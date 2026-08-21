---
title: torch.compile 配置与模式
type: concept
status: seed
tags: [PyTorch, torch.compile, 配置, mode, 调试]
created: 2026-08-17
updated: 2026-08-17
source:
  - d:\project\pytorch-2.8.0\wiki\14_config.html
---

# 十四、配置与模式

*编译模式、关键配置与调试工具*

torch.compile 提供丰富的配置体系，覆盖从编译模式选择到细粒度优化开关。本章梳理 `mode` 参数、`torch._dynamo.config` 与 `torch._inductor.config` 两套配置对象、调试工具链，以及自定义后端的注册方式，最后给出性能调优清单与最佳实践。

> **💡 提示：配置分层：**Dynamo（前端）的配置在 `torch/_dynamo/config.py`，控制追踪/缓存/Guard 行为；Inductor（后端）的配置在 `torch/_inductor/config.py`，控制 Lowering/融合/代码生成。两者独立，但 `mode` 参数会一次性设置两边的相关项。

### 14.1 编译模式 (mode) 详解

`mode` 是 `torch.compile()` 的高层预设，定义于 `torch/__init__.py#L2520` 的文档与 `torch/_inductor/codegen/autotuner.py` 的 `list_mode_options()`。可选值与官方说明：

| 模式 | 官方说明 | 适用场景 |
| --- | --- | --- |
| `"default"` | 默认模式，性能与编译开销的良好平衡 | 通用场景、首次试用 |
| `"reduce-overhead"` | 用 CUDA Graphs 降低 Python 开销，适合小 batch；以更多显存换更低延迟。仅对纯 CUDA、不突变输入的图生效 | 小 batch 推理/训练、内核多且小 |
| `"max-autotune"` | 用 Triton/模板 matmul 与 Triton 卷积；GPU 上默认启用 CUDA Graphs | 追求极致性能、能接受长编译 |
| `"max-autotune-no-cudagraphs"` | 同 max-autotune 但不启用 CUDA Graphs | 显存敏感或动态形状 + 模板调优 |

#### 各模式具体启用的配置

下表列出每个模式相对 `default` 的差异（可通过 `torch._inductor.list_mode_options()` 查看）：

| 配置项 | default | reduce-overhead | max-autotune | max-autotune-no-cudagraphs |
| --- | --- | --- | --- | --- |
| `triton.cudagraphs` | False | True | True | False |
| `max_autotune` | False | False | True | True |
| `max_autotune_pointwise` | False | False | True | True |
| `max_autotune_gemm` | False | False | True | True |
| `epilogue_fusion` | False | False | True | True |
| `coordinate_descent_tuning` | False | False | True | True |
| `cuda.enable_cuda_graph` | False | True | True | False |

> **⚠️ 注意：互斥约束：**`mode` 与 `options` 不能同时指定（见 `__init__.py#L2607`）。若 `mode=None` 且 `options=None`，则默认 `mode="default"`。需要细粒度控制时用 `options={...}` 并保持 `mode=None`。

### 14.2 关键 API 参数

| 参数 | 说明 | 默认值 | 典型取值 |
| --- | --- | --- | --- |
| `fullgraph` | 要求整图编译，不允许图断裂 | `False` | 调试/极致性能时 `True` |
| `dynamic` | 动态形状支持：`None` 自动、`True` 强制动态、`False` 强制静态 | `None` | 变长输入 `True` |
| `backend` | 编译后端 | `"inductor"` | 字符串或自定义 callable |
| `disable` | 禁用编译，变 no-op | `False` | 调试/对比基准 |

#### dynamic 的三种语义

- **`None`（自动）：**首次按静态编译；重编译时若检测到形状变化，自动生成更动态的内核
- **`True`：**主动把维度标记为符号，尽量生成动态内核，减少重编译
- **`False`：**永不生成动态内核，全部特殊化，性能最优但形状变化即重编译

### 14.3 torch._dynamo.config vs torch._inductor.config

两套配置对象分别控制前端与后端，可在运行时直接修改：

| 配置对象 | 文件 | 管辖范围 | 常用项 |
| --- | --- | --- | --- |
| `torch._dynamo.config` | `torch/_dynamo/config.py` | 前端：追踪、Guard、缓存 | `recompile_limit`, `cache_size_limit`, `suppress_errors`, `verbose` |
| `torch._inductor.config` | `torch/_inductor/config.py` | 后端：Lowering、融合、代码生成 | `triton.cudagraphs`, `max_autotune`, `epilogue_fusion`, `fallback_random` |

*配置使用示例*

```python
import torch
import torch._dynamo.config as dynamo_config
import torch._inductor.config as inductor_config

# 前端: 提高重编译上限 (调试形状抖动)
dynamo_config.recompile_limit = 64
dynamo_config.cache_size_limit = 256

# 后端: 关闭 CUDA Graphs, 开启 max-autotune
inductor_config.triton.cudagraphs = False
inductor_config.max_autotune = True
inductor_config.epilogue_fusion = True

# 调试: 错误时不静默回退
dynamo_config.suppress_errors = False

# 列出所有 Inductor 配置项及当前值
print(torch._inductor.list_options())
```

### 14.4 常用 Inductor 配置项

| 配置项 | 说明 | 默认值 |
| --- | --- | --- |
| `triton.cudagraphs` | 启用 CUDA Graphs，降低 Python 开销 | `False` |
| `max_autotune` | 启用 matmul/conv 自动调优 | `False` |
| `epilogue_fusion` | 将 pointwise 融合进模板（需 `max_autotune`） | `False` |
| `fallback_random` | 用 eager 随机数（调试精度问题） | `False` |
| `shape_padding` | 填充矩阵形状以优化 GPU 对齐/张量核 | `False` |
| `trace.enabled` | 最全面的调试日志 | `False` |
| `trace.graph_diagram` | 输出融合后的计算图图示 | `False` |
| `coordinate_descent_tuning` | 对 pointwise 内核做坐标下降调优 | `False` |
| `force_disable_caches` | 禁用所有缓存（强制重编译） | `False` |
| `cache_dir` | 生成代码缓存目录 | `/tmp/torchinductor_<user>` |
| `triton.descriptive_names` | 内核名包含融合算子描述 | `True` |
| `triton.cooperative_reductions` | 启用协作归约 | `True` |

### 14.5 调试工具与环境变量

| 工具 / 环境变量 | 用途 |
| --- | --- |
| `TORCH_LOGS=guards` | 查看 Guard 失败原因 |
| `TORCH_LOGS=recompiles` | 追踪重编译事件与触发原因 |
| `TORCH_LOGS=dynamic` | 调试过度特殊化 / 动态形状问题 |
| `TORCH_LOGS=output_code` | 打印生成的 Triton/C++ 内核源码 |
| `TORCH_LOGS=fusion` | 查看融合决策与原因 |
| `TORCH_LOGS=perf_hints` | 调试性能提示（如 CUDA Graphs 不适用） |
| `TORCH_COMPILE_DEBUG=1` | 把调试产物写入 `torchinductor_*/debug/`（含代码、调度、时间） |
| `TORCHDYNAMO_DISABLE=1` | 全局禁用 torch.compile |
| `TORCHINDUCTOR_COMPILE_THREADS=N` | 并行编译线程数 |
| `torch._dynamo.explain(fn)` | 分析图断裂原因与编译统计 |
| `torch._inductor.list_options()` | 列出所有 Inductor 配置项 |
| `torch._inductor.list_mode_options()` | 列出各 mode 启用的配置 |
| `torch._dynamo.list_backends()` | 列出可用后端 |
| `torch._dynamo.utils.counters` | 查看编译过程计数器 |
| `torch.compiler.assume_constant_by_default` | 调试：把标量视为常量 |

> **✨ 技巧：组合使用：**`TORCH_LOGS` 支持逗号组合，如 `TORCH_LOGS=guards,recompiles,fusion` 一次看多项。可用 `torch._logging.set_logs(guards=True)` 在代码中动态开启。配合 `TORCH_COMPILE_DEBUG=1` 能把完整调试快照落盘，便于离线分析。

### 14.6 自定义后端注册

后端协议：`backend(gm: GraphModule, example_inputs) → Callable`。注册后即可用字符串引用：

*注册自定义后端 (基本)*

```python
from torch._dynamo.backends.registry import register_backend

@register_backend
def my_custom_backend(gm: torch.fx.GraphModule, example_inputs):
    """接收 FX GraphModule，返回可执行函数"""
    gm = optimize_graph(gm)          # 可选: 自定义图优化
    return gm.forward     # 最简: 直接返回 (相当于 eager)

# 使用
model = torch.compile(model, backend="my_custom_backend")
```

*复用 AOTAutograd + Inductor 的自定义后端*

```python
from torch._dynamo.backends.registry import register_backend
from torch._functorch.aot_autograd import aot_autograd
from torch._inductor.compile_fx import compile_fx_inner
import torch._inductor.config as config

@register_backend
def my_inductor_variant(gm, example_inputs):
    # 自定义 Inductor 配置
    config.coordinate_descent_tuning = True

    # 用 AOTAutograd 拆分前后向, 再调 Inductor 内层编译
    return aot_autograd(
        fw_compiler=compile_fx_inner,
        bw_compiler=compile_fx_inner,
        decompositions=torch._inductor.select_decomp_table(),
    )(gm, example_inputs)

model = torch.compile(model, backend="my_inductor_variant")
```

*纯 eager 调试后端 (打印图)*

```python
@register_backend
def print_graph_backend(gm, example_inputs):
    print(gm.graph)               # 打印 FX 图
    print(f"节点数: {len(gm.graph.nodes)}")
    return gm.forward      # 不优化, 直接 eager 执行

model = torch.compile(model, backend="print_graph_backend")
```

> **📝 说明：后端协议要点：**返回的可调用对象接收与原始函数相同的参数，返回相同的输出。后端可只优化部分图（其余回退），也可完全不做优化。注册后端后，`torch._dynamo.list_backends()` 会列出它。完整自定义后端指南见 [官方文档](https://pytorch.org/docs/main/torch.compiler_custom_backends.html)。

### 14.7 性能调优检查清单

### ① 形状与输入

- □ batch size 是否固定？变长用 `dynamic=True`
- □ 输入 dtype/device 是否一致？避免抖动
- □ 是否避免 `tensor.item()` 等数据依赖控制流？

### ② 图完整性

- □ `torch._dynamo.explain` 检查图断裂
- □ 能否消除不支持的 Python 特性？
- □ 稳定后试 `fullgraph=True` 进一步提效

### ③ 重编译

- □ `TORCH_LOGS=recompiles` 是否为 0？
- □ `counters['stats']['recompiles']` 是否接近 8？
- □ 必要时 `dynamic=True` 或 `recompile_limit` 调大

### ④ 融合质量

- □ `TORCH_LOGS=output_code` 内核数是否合理？
- □ `TORCH_LOGS=fusion` 查看融合原因
- □ pointwise 链是否被合并为单内核？

### ⑤ 模式选择

- □ 小 batch 试 `reduce-overhead`
- □ 计算密集试 `max-autotune`
- □ 显存敏感用 `max-autotune-no-cudagraphs`

### ⑥ 显存与开销

- □ CUDA Graph 会增加显存占用
- □ `cache_size_limit` 是否够用？
- □ 关闭 `trace.*` 调试开关后是否更快？

### 14.8 最佳实践 (Do's and Don'ts)

| 场景 | ✅ 推荐 (Do) | ❌ 避免 (Don't) |
| --- | --- | --- |
| 输入形状 | 固定形状用默认；变长用 `dynamic=True` | 在训练中频繁切换 batch size 而不开 dynamic |
| 控制流 | 用 `torch.where` / 布尔 mask 替代 `if tensor` | 在编译区调用 `.item()` 触发数据依赖分支 |
| 随机数 | 信任 Inductor 的 RNG 融合 | 精度问题先怀疑 RNG 时，用 `fallback_random=True` 定位 |
| 编译模式 | 先用 `default` 验证正确性，再开 `max-autotune` | 一上来就 `max-autotune` 调试，编译太慢 |
| fullgraph | 稳态后尝试 `fullgraph=True` 提效 | 开发期就 `fullgraph=True`，图断裂直接报错难迭代 |
| CUDA Graph | 小 batch + 纯 CUDA 图用 `reduce-overhead` | 动态形状 + CUDA Graph（会失效或重录） |
| 调试 | `TORCH_COMPILE_DEBUG=1` 离线分析 | 生产环境长期开 `trace.enabled`（有开销） |
| 缓存 | CI/部署时复用 `cache_dir` 与 Triton 缓存 | 频繁清缓存导致每次冷启动全编译 |
| 副作用 | 编译函数保持纯函数语义 | 在 forward 里修改全局变量 / 打印张量值 |
| 混合精度 | 用 `torch.autocast` 配合 compile | 手动在编译区反复 `.to(float)` / `.half()` |

### 14.9 常见陷阱与规避

> **⚠️ 注意：陷阱 1：隐式重编译风暴。**DataLoader 输出最后一个不完整的 batch（`drop_last=False`）会导致形状变化触发重编译。**规避：**设 `drop_last=True`，或 `dynamic=True`。

> **⚠️ 注意：陷阱 2：模型处于 train/eval 切换。**`model.train()` 与 `model.eval()` 会改变 `BatchNorm`/`Dropout` 行为，触发 Guard 失败重编译。**规避：**切换模式是正常的，Dynamo 会为每种模式各编译一次（计入 recompile 计数）；若模式来回切换频繁，确保 `recompile_limit` 足够。

> **⚠️ 注意：陷阱 3：CUDA Graph 输入地址变化。**CUDA Graph 要求输入张量地址固定，若每次 `to(device)` 分配新内存会失效。**规避：**Inductor 在 `reduce-overhead` 下会管理静态输入池；不要在编译函数内部对新输入做不必要的拷贝。

> **⚠️ 注意：陷阱 4：`fullgraph=True` 下报错难定位。**图断裂在严格模式直接抛异常，栈追踪可能不直观。**规避：**先用 `fullgraph=False` + `torch._dynamo.explain` 定位断裂点，修复后再开严格模式。

### 14.10 快速验证模板

*验证编译收益的标准流程*

```python
import torch, time

model = MyModel().cuda().eval()
x = torch.randn(B, C, H, W, device="cuda")

# 1. eager 基准
with torch.no_grad():
    t0 = time.perf_counter()
    for _ in range(50): model(x)
    torch.cuda.synchronize()
    print("eager:", (time.perf_counter() - t0) / 50)

# 2. 编译 (不含编译时间的稳态)
compiled = torch.compile(model, mode="max-autotune")
with torch.no_grad():
    compiled(x)  # 预热 (触发首次编译)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50): compiled(x)
    torch.cuda.synchronize()
    print("compiled:", (time.perf_counter() - t0) / 50)

# 3. 检查重编译
print("recompiles:", torch._dynamo.utils.counters["stats"].get("recompiles", 0))
```

> **✨ 技巧：测试要领：**① 始终 `torch.cuda.synchronize()` 后再计时；② 至少 1 次预热触发编译，编译时间不计入稳态；③ 对比 eager 与 compiled 的**稳态吞吐**，而非首调用；④ 用 `torch profiler` 进一步定位内核级瓶颈。

## Related

- [13 完整编译流程总结](./13-full-compile-pipeline.md) — 编译全流程中 mode/配置如何影响首次编译耗时
- [15 torch.fx 专题](./15-torch-fx-special.md) — 自定义后端协议基于 FX GraphModule
- [PyTorch 索引](../index.md)
