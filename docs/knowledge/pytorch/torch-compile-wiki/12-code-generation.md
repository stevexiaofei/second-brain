---
title: torch.compile 代码生成
type: concept
status: seed
tags: [PyTorch, torch.compile, 代码生成, Triton, C++]
created: 2026-08-17
updated: 2026-08-17
source:
  - d:\project\pytorch-2.8.0\wiki\12_codegen.html
---

# 十二、代码生成

*Triton GPU 内核与 C++ CPU 代码的生成*

代码生成是 Inductor 编译流水线的最终阶段。它接收调度器输出的融合后的 IR 节点列表，将每个融合组转换为目标平台可执行代码：在 GPU 上生成 `Triton` 内核，在 CPU 上生成 `C++` 代码。核心代码位于 `torch/_inductor/codegen/`。

> **💡 提示：核心思路：**代码生成并不是"逐算子翻译"，而是把一整个融合组（FusedSchedulerNode）翻译成**一个内核**。融合组内所有点级/归约操作的中间结果都留在寄存器中，只在最后写回显存，从而把多次内存访问压缩为一次。

### 12.1 后端选择与类继承体系

Inductor 通过设备类型选择代码生成后端。两个后端共享一套抽象基类，差异通过子类覆写实现：

| 后端 | 文件 | 目标平台 | 核心类 |
| --- | --- | --- | --- |
| Triton | `codegen/triton.py` | NVIDIA / AMD GPU | `TritonKernel`, `TritonScheduling` |
| C++ | `codegen/cpp.py` | CPU | `CppKernel`, `CppScheduling` |
| SIMD 共享 | `codegen/simd.py` | Triton/Halide 共用 | `SIMDKernel`, `SIMDScheduling` |
| Common | `codegen/common.py` | 共享逻辑 | `CodeGen`, `Kernel`, `CSE` |

```text
CodeGen (common.py#L1940)                BaseScheduling
   │                                         │
   └── Kernel (common.py#L1953)              │
          │  loads/compute/stores            │
          │  cse: CSE                         │
          │                                  │
          ├── SIMDKernel (simd.py#L371) ──── SIMDScheduling (simd.py#L1114)
          │      "flattened indexing,        "fusion across backends"
          │       no loop nests"                   │
          │            │                            │
          │            └── TritonKernel            └── TritonScheduling
          │               (triton.py#L1620)         (triton.py#L4099)
          │
          └── CppKernel (cpp.py#L1876) ──── CppScheduling (cpp.py#L4453)
```

### 12.2 CodeGen / Kernel 基类

定义于 `codegen/common.py#L1940`，是所有代码生成器的根基：

*torch/_inductor/codegen/common.py*

```python
class CodeGen:
    """代码生成器基类，管理上下文栈"""
    def __init__(self):
        self.exit_stack = contextlib.ExitStack()


class Kernel(CodeGen, Generic[CSEVariableType]):
    """内核基类，每个 Kernel 对应一个生成的内核函数"""
    newvar_prefix: str = ""   # C++ 用 "auto ", Triton 用 ""
    suffix: str = ""          # C++ 用 ";", Triton 用 ""

    def __init__(self, args=None, increase_kernel_count=True):
        super().__init__()
        if increase_kernel_count:
            metrics.generated_kernel_count += 1
        self.args = args or KernelArgs()
        self.loads = IndentedBuffer()    # 加载 (tl.load) 代码段
        self.compute = IndentedBuffer()  # 计算 (寄存器内) 代码段
        self.stores = IndentedBuffer()   # 存储 (tl.store) 代码段
        self.cse = CSE(self.newvar_prefix, self.suffix)  # 公共子表达式消除
        self.must_keep_buffers = OrderedSet()
        self.removed_buffers = OrderedSet()
        self.inplace_update_buffers = {}  # 就地复用：写缓冲可重用读缓冲内存
        self.min_elem_per_thread = 1
        self.kernel_name = None

    # 子类必须实现的核心接口:
    def load(self, name, index): ...          # 生成加载代码
    def store(self, name, index, value): ...  # 生成存储代码
    def reduction(self, dtype, src_dtype, reduction_type, value): ...
```

`Kernel` 把生成代码拆成三段缓冲：**loads**（输入加载）、**compute**（寄存器内计算）、**stores**（输出写回）。这种分段使得融合分析可以精确地把中间表达式留在 compute 段而不落回显存。

#### IndentedBuffer：缩进感知的代码缓冲

定义于 `torch/_inductor/utils.py#L1198`，是所有代码生成的"画布"：

*torch/_inductor/utils.py#L1198*

```python
class IndentedBuffer:
    tabwidth = 4
    def __init__(self, initial_indent=0):
        self._lines = []
        self._indent = initial_indent

    # 自动管理缩进层级，支持 push/pop
    def writeline(self, line): ...
    def splice(self, code): ...        # 拼接一段带缩进的代码
    def indent(self): ...            # 上下文管理器，进入下一级缩进
    def getvalue(self): ...         # 输出最终字符串
```

它还支持 `DeferredLine`（延迟求值的行）和 `LineContext`（行级元信息，用于把生成代码映射回源 FX 节点），便于调试和 profiling。

### 12.3 SIMDKernel：Triton 与 Halide 的共享基类

定义于 `codegen/simd.py#L371`，`TritonKernel` 直接继承自它：

*torch/_inductor/codegen/simd.py#L371*

```python
class SIMDKernel(Kernel[CSEVariableType], Generic[CSEVariableType]):
    """
    Common base class for Triton/Halide codegen which both use
    flattened indexing rather than loop nests.
    """

    def __init__(self, tiling, features, pid_cache=None, ...):
        super().__init__()
        self.features = features
        self.mutations = features.get_mutations()
        self.body = IndentedBuffer()          # 内核主体
        self.indexing_code = IndentedBuffer() # 索引计算代码
        self.numels = {...}                   # 各维度的元素数 (xnumel, rnumel)
        self.range_trees = []                 # 迭代范围树 (循环展开结构)
        self.inside_reduction = features.is_reduction()  # 是否含归约
        self.cooperative_reduction = self.should_use_cooperative_reduction()
        self.persistent_reduction = self.should_use_persistent_reduction()
        self.no_x_dim = self.want_no_x_dim()
        self.initialize_range_tree(pid_cache)
```

> **✨ 技巧：SIMD 的关键设计：**"flattened indexing rather than loop nests"——SIMD 后端**不生成嵌套 for 循环**，而是把多维索引**展平**为一维，用 `range_tree` 描述并行维度与归约维度的层次结构，再映射到 Triton 的 `tl.program_id` / `tl.arange` / `tl.reduce`。这正是 Triton 的 SPMD 编程模型所要求的。

#### SIMDScheduling：融合决策的共享逻辑

定义于 `codegen/simd.py#L1114`，是 `TritonScheduling` 的父类：

*torch/_inductor/codegen/simd.py#L1114*

```python
class SIMDScheduling(BaseScheduling):
    """
    Single Instruction Multiple Data parent class used for fusion
    across multiple different backends.
    """
    kernel_type = SIMDKernel  # 子类覆写为 TritonKernel

    def can_fuse(self, node1, node2):
        """调度器调用：判断两个节点能否融合"""
        # 比较 numel/rnumel、归约类型、split-scan 兼容性等
        ...

    def codegen_node(self, node):
        """给定一个融合节点，生成一个 Triton 内核"""
        nodes = node.get_nodes()
        _, (numel, rnumel) = max(nodes, key=lambda x: int(x.is_reduction())).group
        node_schedule = self.generate_node_schedule(nodes, numel, rnumel)
        return self.codegen_node_schedule(
            SIMDKernelFeatures(node_schedule, numel, rnumel, ...)
        )
```

### 12.4 TritonKernel：生成 Triton GPU 内核

定义于 `codegen/triton.py#L1620`，`TritonKernel(SIMDKernel[TritonCSEVariable])`：

*torch/_inductor/codegen/triton.py#L1620*

```python
class TritonKernel(SIMDKernel[TritonCSEVariable]):
    overrides = TritonKernelOverrides
    kexpr = texpr                # sympy 表达式 → Triton 字符串
    allow_block_ptr = True        # 允许使用 tl.make_block_ptr 优化

    def __init__(self, tiling, min_elem_per_thread=0,
                 optimize_mask=True, fixed_config=None, **kwargs):
        super().__init__(tiling, **kwargs)
        self.optimize_mask = optimize_mask
        self.fixed_config = fixed_config      # max-autotune 选定的配置
        self.cse = TritonCSE(self.newvar_prefix, self.suffix)
        self.post_loop_combine = IndentedBuffer()
        self.post_loop_store = IndentedBuffer()
        self.outside_loop_vars = OrderedSet()
        self.min_elem_per_thread = min_elem_per_thread
        self.block_ptr_id = itertools.count()
        self.helper_functions = HelperFunctions()
        self.autotune_hints = OrderedSet()    # 传给 Triton autotuner 的提示
        self.triton_meta = None

        if self.inside_reduction:
            self.codegen_reduction_numels(self.body)
        if self.cooperative_reduction:
            self.init_cooperative_reduction()
        self.codegen_range_tree()
```

`TritonKernel` 把 IR 操作转换为 Triton 内联表达式。例如 `Pointwise(op=lambda a, b: a + b)` 会被翻译成 `tmp0 + tmp1`，其中 `tmp0/tmp1` 来自 `tl.load`。所有融合的计算都通过 `CSE` 命名为 `tmp0, tmp1, ...` 串联在 compute 段。

#### load / store / reduction 三个核心方法

定义于 `triton.py#L2191 / #L2326 / #L2454`：

*TritonKernel 核心方法*

```python
def load(self, name, index):
    # 生成: tmp = tl.load(in_ptr + offset, mask=..., other=...)
    # 自动处理边界 mask、block_ptr 优化、向量化
    ...

def store(self, name, index, value, mode=None):
    # 生成: tl.store(out_ptr + offset, value, mask=...)
    # mode 支持 "atomic_add" 等原子操作
    ...

def reduction(self, dtype, src_dtype, reduction_type, value):
    # reduction_type: "sum" / "max" / "min" / "argmax" ...
    # 自动把 FP16/BF16 提升为 FP32 累加以保证精度
    # 生成: tl.sum(value, axis=...) / tl.max(...) 等
    # 支持持久化归约与协作归约 (cooperative reduction)
    ...
```

#### 生成的 Triton 内核示例

*Inductor 生成的 Triton 代码 (示意: relu((a+b)*c))*

```python
@triton.jit
def triton_poi_fused_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
                       xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel

    # === loads 段 ===
    tmp0 = tl.load(in_ptr0 + xindex, mask=xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + xindex, mask=xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + xindex, mask=xmask, other=0.0)

    # === compute 段 (全在寄存器中，无显存访问) ===
    tmp3 = tmp0 + tmp1
    tmp4 = tmp3 * tmp2
    tmp5 = tl.maximum(tmp4, 0)   # relu

    # === stores 段 ===
    tl.store(out_ptr0 + xindex, tmp5, mask=xmask)
```

### 12.5 Pointwise 与 Reduction 内核的差异

调度器区分两类融合组，`SIMDKernel` 通过 `inside_reduction` 标志走不同代码路径：

| 维度 | Pointwise 内核 | Reduction 内核 |
| --- | --- | --- |
| IR 类型 | `ir.Pointwise` | `ir.Reduction` |
| `inside_reduction` | `False` | `True` |
| 并行结构 | 单层 `xnumel` 网格，每线程处理一个元素 | 外层 `xnumel` + 内层 `rnumel` 归约维度 |
| Tile 大小 | `XBLOCK` | `XBLOCK` + `RBLOCK` |
| 归约操作 | 无 | `tl.sum` / `tl.max` 沿 RBLOCK 归约 |
| Mask 处理 | 仅 `xmask` | `xmask` + `rmask` 双 mask |
| 特殊优化 | block_ptr、向量化 load | 持久化归约、协作归约、welford (方差) |
| 生成的 grid | 1D grid | 1D 或 2D grid (含 rsplit 维) |

**归约内核额外步骤**（见 `codegen_reduction_numels` / `codegen_reduction_indices`，triton.py#L4009/#L4054）：

- **持久化归约 (persistent reduction)：**当归约维度较小时，单个 program 处理多行输出，避免 kernel 启动开销
- **协作归约 (cooperative reduction)：**跨多个 SM 协作完成大归约，通过 `semaphores` 同步，适合超大 rnumel
- **Welford 算法：**用于 mean/var 归约，数值稳定地累加均值与方差
- **精度提升：**FP16/BF16 归约自动提升到 FP32 累加（见 `reduction` 方法 `maybe_upcast`）

### 12.6 CppKernel：生成 C++ CPU 代码

定义于 `codegen/cpp.py#L1876`，`CppKernel(Kernel)`。CPU 后端使用**嵌套 for 循环**而非 SIMD 展平索引：

*torch/_inductor/codegen/cpp.py#L1876*

```python
class CppKernel(Kernel):
    overrides = CppOverrides
    sexpr = cexpr
    newvar_prefix = "auto "   # 生成的变量用 auto 声明
    suffix = ";"             # 每条语句以分号结尾

    def __init__(self, args, num_threads):
        super().__init__(args)
        self.active_ranges = {}      # 内核激活区间 (尾循环优化)
        self.inner_itervars = []     # 内层循环变量
        self.call_ranges = None
        self.ranges = []             # 各维度范围
        self.itervars = []           # 循环变量名
        self.reduction_prefix = IndentedBuffer()
        self.reduction_suffix = IndentedBuffer()
        self.parallel_reduction_prefix = IndentedBuffer()
        self.parallel_reduction_suffix = IndentedBuffer()
        self.local_reduction_init = IndentedBuffer()
        self.local_reduction_stores = IndentedBuffer()
        self.reduction_cse = CSE(self.newvar_prefix, self.suffix, name_prefix="tmp_acc")
        self.welford_helper_cse = CSE(..., name_prefix="welford_helper")
        self.preloads = IndentedBuffer()
        self.poststores = IndentedBuffer()
        self.num_threads = num_threads  # 特化的线程数
```

#### CPU 并行化与向量化

- **OpenMP 并行：**外层循环用 `#pragma omp parallel for`，线程数由 `parallel_num_threads()` 决定，支持 `config.cpp.dynamic_threads` 动态调整（`omp_get_max_threads()`）
- **归约并行：**每个线程维护本地累加器 `acc_local`，存入 `acc_arr[tid]`，最后串行合并（见 `_gen_parallel_reduction_buffers`）
- **SIMD 向量化：**依赖编译器（gcc/clang）的自动向量化，配合 `at::vec::Vectorized` 在支持的 dtype 上使用 AVX/AVX2/AVX512/NEON 指令
- **尾循环优化：**`active_ranges` 记录 `{x0, {24, 26}}` 表示该内核在 x0 ∈ [24,26) 激活，用于处理非对齐尾部，主循环用向量化、尾部用标量
- **内层循环下沉：**`move_code_under_inner_loop` 把计算移入最内层循环以提升寄存器复用

*生成的 C++ 代码 (示意)*

```c
// 融合内核: relu((a+b)*c)
extern "C" void kernel0(const float* in_ptr0,
                          const float* in_ptr1,
                          const float* in_ptr2,
                          float* out_ptr0,
                          int64_t xnumel) {
    #pragma omp parallel for
    for (int64_t x0 = 0; x0 < xnumel; ++x0) {
        auto tmp0 = in_ptr0[x0];
        auto tmp1 = in_ptr1[x0];
        auto tmp2 = in_ptr2[x0];
        auto tmp3 = tmp0 + tmp1;
        auto tmp4 = tmp3 * tmp2;
        auto tmp5 = tmp4 > 0 ? tmp4 : 0;   // relu
        out_ptr0[x0] = tmp5;
    }
}
```

### 12.7 CSE：公共子表达式消除

定义于 `codegen/common.py#L1772`，是 `Kernel` 内置的优化：

*torch/_inductor/codegen/common.py#L1725*

```python
class CSEVariable:
    """一个表达式的名字，可被后端附加注释"""
    def __init__(self, name, bounds, dtype=None):
        self.name = name          # 如 "tmp3"
        self.bounds = bounds      # 值范围，用于 bound 推导
        self.use_count = 1        # 被引用次数
        self.dtype = dtype

class CSE(Generic[CSEVariableType]):
    """Common subexpression elimination"""
    def __init__(self, prefix, suffix, name_prefix="tmp", ...):
        self._cache = {}             # 表达式 → 变量名
        self.store_cache = {}        # 存储缓存
        self.reduction_cache = {}    # 归约缓存
        self.varname_map = {}

    def generate(self, buffer, code, ...):
        # 若 code 已在 _cache 中 → 复用变量名，不重复生成
        # 否则 → 生成新变量 "tmp{N}"，写入 buffer
        ...
    def invalidate(self, keep_vars):
        # 当 store 可能改变内存值时，失效相关缓存
        ...
```

CSE 的工作原理：每次要生成一个表达式 `a + b` 时，先查 `_cache`；若同样的表达式已生成过，就直接复用 `tmp3` 而不重复计算。`store_cache` 还能识别"加载刚存储的值"这种模式，直接复用寄存器变量，省去一次 `tl.load`。`TritonCSEVariable` 还会附加 `update_on_args` 注解，为 Triton 编译器提供额外信息。

### 12.8 生成代码的 Python 包装

生成的内核源码不能直接运行，需要被包装成可调用的 Python 函数。这一步由 `torch/_inductor/codegen/wrapper.py` 的 `WrapperCodegen` 完成：

*包装流程*

```python
# 1. 每个融合内核被 define_kernel() 写入一个 .py 文件
#    (triton.py: TritonScheduling.define_kernel)
#    路径: /tmp/torchinductor_<user>/<hash>.py

# 2. wrapper 生成一个 Python 函数，按调度顺序调用各内核
def call(args_0, args_1, ...):
    # 输入是扁平化的张量列表
    kernel0.run(args_0, args_1, args_2, out_ptr0, xnumel,
                grid=grid_fn, stream=stream)
    kernel1.run(out_ptr0, out_ptr1, ...)
    return (out_ptr0, out_ptr1,)

# 3. Triton 内核首次调用时自动调优 (autotune)
#    - 遍历候选配置 (num_stages, num_warps, XBLOCK)
#    - 实测每条配置耗时，选最优
#    - 结果缓存到 ~/.triton/cache

# 4. 最终返回的 CompiledFxGraph 持有:
#    - 编译后的 wrapper.py 模块
#    - 输入/输出的张量布局映射
#    - CUDA Graph 句柄 (若启用)
```

> **📝 说明：代码缓存：**每个生成的内核根据源码计算 `code_hash`，存为 `<hash>.py`。下次遇到相同内核直接复用文件，跳过 Triton 编译。这就是 Inductor 的**冷启动后第二次编译更快**的原因之一。缓存目录可通过 `torch._inductor.config.cache_dir` 配置。

### 12.9 模板代码生成 (Template Codegen)

普通融合内核是"自动生成"的逐元素/归约代码。但对于 **matmul** 和 **conv** 这类计算密集型算子，Inductor 在 `max-autotune` 模式下使用**模板**代码生成，由 `codegen/common.py#L2260` 的 `KernelTemplate` 统一抽象：

*torch/_inductor/codegen/common.py#L2260*

```python
class KernelTemplate:
    """
    Base class for defining kernel templates.
    Children classes: TritonTemplate, CUDATemplate
    """
    @staticmethod
    def _template_from_string(source):
        # 用 Jinja2 模板引擎渲染内核源码
        env = jinja2_env()
        return env.from_string(source)
```

### TritonTemplate

用 Triton 编写的 matmul 模板（如 `ttgir`）。通过 epilogue 融合，把后续的 pointwise（bias、relu、gelu 等）嵌入 matmul 模板的 epilogue 段，避免额外内核。需要 `config.max_autotune` 启用。

### CUDATemplate

基于 cuBLAS / cuDNN / cutlass 的 CUDA 模板，用于 matmul 与卷积。同样支持 epilogue 融合。需要 `max-autotune` 且在 GPU 上启用。

### 自动调优 (Autotuning)

模板代码生成会枚举多组配置（tile、stage、warp、K 分割），对每组实际编译并运行 benchmark，选出最优配置。这是 `max-autotune` 编译时间长的主因。

> **⚠️ 注意：模板融合条件：**要触发模板融合，前置 pointwise 必须能被吸收进 matmul 的 prologue，后置 pointwise 必须能进 epilogue。需要 `config.epilogue_fusion=True`（`max-autotune` 默认开启）。

### 12.10 CUDA Graphs 集成

在 `reduce-overhead` 模式或启用 `triton.cudagraphs` 时，Inductor 把整批内核调用包装为 CUDA Graphs，定义于 `torch/_inductor/cudagraph_trees.py`：

- **首次执行录制：**捕获所有内核启动、参数与内存地址到一个 CUDA Graph
- **后续执行重放：**直接 `cudaGraphLaunch`，消除每次内核启动的 CPU 端开销（约 5-10μs/内核）
- **工作空间缓存：**缓存 Graph 的工作内存，避免重复分配（代价是显存占用增加）
- **限制：**要求输入/输出地址固定、不支持动态形状、不支持输入突变；仅 CUDA-only 图生效
- **多 CUDA Graph 树：**通过 `CUDAGraphTree` 管理不同输入形状对应的多个 Graph，按需切换

> **✨ 技巧：如何查看生成的代码：**设置 `TORCH_LOGS=output_code` 会打印 Inductor 生成的全部 Triton/C++ 内核源码。也可用 `TORCH_COMPILE_DEBUG=1` 把生成的代码、调度日志、融合结果写到 `torchinductor_<user>/debug/` 目录。配合 `TORCH_LOGS=fusion` 可查看融合决策。

### 12.11 后端特性矩阵

| BackendFeature | TritonScheduling | CppScheduling | 说明 |
| --- | --- | --- | --- |
| FOREACH | ✓ | ✗ | 支持 `foreach` 算子组融合 |
| BUCKETIZE | ✓ | ✗ | 支持 `bucketize` |
| INPLACE_BUFFERS | ✓ | ✓ | 允许就地缓冲区复用 |
| MASKED_SCATTER_WITH_INDEX | ✓ | ✗ | 带索引的 masked_scatter |
| SCAN | ✓ | ✗ | 支持 cumulative scan (前缀和) |
| SORT | ✓ | ✗ | 支持排序算子 |
| TRITON_TEMPLATES | ✓ | ✗ | 支持 Triton 模板 (matmul/conv) |
| TUPLE_REDUCTION | ✓ | ✗ | 支持多元组归约 (如 mean+var) |
| REDUCE_TO_SINGLE_ELEMENT | 可选* | ✓ | 归约到单元素（*启用 cooperative_reductions 时） |

调度器在 `can_fuse` 决策时会查询这些特性，决定某类算子是否可在当前后端融合。

## Related

- [11 调度器与融合](./11-scheduler-fusion.md)
- [13 完整编译流程总结](./13-full-compile-pipeline.md)
- [PyTorch 索引](../index.md)
