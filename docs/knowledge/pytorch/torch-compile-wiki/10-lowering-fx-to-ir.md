---
title: "Lowering: FX → IR"
type: concept
status: seed
tags: [PyTorch, torch.compile, Lowering, IR, Inductor]
created: 2026-08-17
updated: 2026-08-17
source:
  - d:\project\pytorch-2.8.0\wiki\10_lowering.html
---

# Lowering: FX → IR

*从计算图到中间表示的转换*

## ⬇️ 10.1 Lowering 概述

Lowering 是将 FX 图节点转换为 Inductor 内部 IR (中间表示) 的过程。前一章我们看到 `compile_fx` 的三个编译回调最终都进入 `inner_compile`（`compile_fx_inner`），后者会创建 `GraphLowering` 实例并启动 Lowering。核心文件：`torch/_inductor/lowering.py` 与 `torch/_inductor/ir.py`。

Lowering 的目标是把高层的 ATen 算子（如 `aten.add`、`aten.mm`）翻译为少量、规整的 IR 节点（`Pointwise`、`Reduction`、`View`、模板算子等），使后续调度器能高效地做融合与代码生成。

## 🔄 10.2 GraphLowering：FX 解释器模式

定义于 [graph.py#L288](file:///d:/project/pytorch-2.8.0/torch/_inductor/graph.py#L288)，`GraphLowering` 继承自 `torch.fx.Interpreter`：

**torch/_inductor/graph.py#L288**

```python
class GraphLowering(torch.fx.Interpreter):
    graph_outputs: list[ir.IRNode]

    def __init__(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: Optional[Sequence[object]] = None,
        shape_env: Optional[ShapeEnv] = None,
        graph_id: Optional[int] = None,
        cpp_wrapper: bool = False,
        aot_mode: bool = False,
        layout_opt: Optional[bool] = None,
        extern_node_serializer: Optional[...] = None,
        is_inference: bool = False,
        is_backward: bool = False,
        is_const_graph: bool = False,
        ...
    ) -> None:
        super().__init__(gm)
        self.example_inputs = example_inputs
        self.layout_opt = (
            layout_opt if layout_opt is not None
            else self.decide_layout_opt(gm, is_inference=is_inference)
        )
        ...
        self.sizevars = SizeVarAllocator(shape_env)
        self.graph_inputs: dict[str, Union[TensorBox, TorchBindObject, sympy.Expr]] = {}
        self.graph_outputs: list[ir.IRNode] = []
        self.buffers: list[ir.Buffer] = []
        self.operations: list[ir.Operation] = []
```

### 10.2.1 解释器模式工作原理

`torch.fx.Interpreter` 会按拓扑顺序逐节点遍历 FX 图，并为 **每种 FX 节点类型** 调用对应的处理方法。GraphLowering 重写了这些方法，把 FX 节点转换为 IR：

| FX 节点类型 (node.op) | Interpreter 方法 | GraphLowering 行为 |
| --- | --- | --- |
| `placeholder` | `placeholder()` | 创建 `InputBuffer`，包装为 `TensorBox` 存入 `graph_inputs` |
| `call_function` | `call_function()` | 查 `lowerings` 表，调用对应 lowering 函数生成 IR |
| `get_attr` | `get_attr()` | 读取模块属性，创建 `ConstantBuffer` |
| `output` | `output()` | 收集输出 IR 节点到 `graph_outputs` |

每个 lowering 函数接收若干 `TensorBox`（或其他 IR 节点）作为输入，返回一个新的 `TensorBox` 作为输出。解释器用环境字典 `env` 维护节点名 → IR 节点的映射，从而把整张 FX 图"重新执行"一遍，但执行的不是真实张量计算，而是 IR 构造。

> **💡 提示：** **为何用解释器模式：**FX 图本身就是可解释的数据结构，解释器能天然地按依赖顺序处理节点，且无需重新实现一遍拓扑排序。每个算子 lowering 函数独立注册，便于扩展——新增算子只需 `@register_lowering` 一个函数，无需改动 GraphLowering 主体。

## 🧱 10.3 IR 节点体系

定义于 [ir.py](file:///d:/project/pytorch-2.8.0/torch/_inductor/ir.py)，是 Inductor 的核心数据结构。IR 节点呈层次结构：

**torch/_inductor/ir.py 核心类层次**

```python
class IRNode:                       # 所有 IR 节点基类 (L519)
    name: str
    device: Optional[torch.device]
    dtype: torch.dtype
    layout: Layout

class Loops(IRNode):                   # 循环计算基类 (L847)
    device, dtype, inner_fn, ranges

class Pointwise(Loops):               # 元素级操作 (L984)
    # 对每个元素独立计算: x+y, x.relu(), x.exp()
    # make_loader() 返回 inner_fn

class Scatter(Pointwise):             # 散射写操作 (L1017)
    output_indexer: Callable
    scatter_mode: StoreMode
    # 例: index_put, scatter, masked_scatter

class Reduction(Loops):               # 归约操作 (L1125)
    reduction_ranges: Sequence
    reduction_type: ReductionType  # sum/max/min/prod/...
    src_dtype: torch.dtype
    reduction_hint: ReductionHint
    # 例: x.sum(dim=0), x.max()

# ---- 视图类 (不复制数据, 改变访问方式) ----
class BaseView(IRNode)                # 视图基类
class ExpandView(BaseView):           # 广播扩展 (L2758)
class PermuteView(BaseView):          # 维度置换/转置 (L2837)
class SqueezeView(BaseView):          # 去除尺寸为 1 的维度 (L2881)
class GenericView(BaseView)           # 通用视图 (L2940)
class View(GenericView):              # reshape/view (L2970)

# ---- 缓冲区类 ----
class Buffer(IRNode, CodegenSymbol):  # 直接拥有存储的分配 (L3979)
    layout: Layout
class OperationBuffer(Buffer, Operation):  # L4110
class ComputedBuffer(OperationBuffer):      # L4197
    data: Loops                       # 已物化的计算结果

# ---- 顶层句柄 ----
class TensorBox(MutableBox):          # 张量句柄 (L7551)
class StorageBox(MutableBox):         # 可写存储容器 (L7559)
```

## 📦 10.4 TensorBox / StorageBox / View 关系

这是 Inductor IR 的核心抽象，用于精确追踪数据的读写与依赖关系。

```text
   用户视角的 Tensor
         │
         ▼
   ┌─────────────┐  指向   ┌──────────────┐
   │  TensorBox  │ ──────► │ StorageBox   │  可写存储容器
   │  (句柄)     │         │  .data: IRNode│
   └─────────────┘         └──────┬───────┘
        ▲                         │ 持有
        │ 多个 TensorBox 可       ▼
        │ 指向同一 StorageBox  ┌──────────────┐
        │ (alias)             │ Pointwise /  │  未物化的计算
        │                     │ Reduction    │  (lazy, 可融合)
        │                     └──────┬───────┘
        │                            │ realize()
        │                            ▼
        │                     ┌──────────────┐
        │                     │ComputedBuffer│  已物化缓冲区
        │                     │  (有 name)   │  (注册到 buffers)
        │                     └──────────────┘
        │
   ┌─────────────┐  指向   ┌──────────────┐
   │  TensorBox  │ ──────► │    View      │  只读视图
   │  (句柄)     │         │ (Expand/     │  (不复制数据)
   └─────────────┘         │  Permute/    │
                           │  Squeeze)    │
                           └──────────────┘
```

**三者的协作：**

- **TensorBox：**张量的"句柄"，是 lowering 函数的输入输出类型。`TensorBox.create(data)` 会把任意 IR 节点包进一个 `StorageBox`。
- **StorageBox：**可写的存储容器，允许多次读写。当多个 TensorBox 指向同一个 StorageBox 时，它们构成 **别名 (alias)** 关系——调度器据此精确追踪数据依赖。
- **realize()：**当需要把"未物化"的计算（`Pointwise`/`Reduction`）落地为具体缓冲区时，`StorageBox.realize()` 会创建一个 `ComputedBuffer`（赋予名字、注册到 `V.graph.buffers` 与 `V.graph.operations`），并替换内部 data 指针。
- **View：**只读视图（如 `ExpandView`、`PermuteView`），不复制数据，只改变访问方式。视图使得 reshape/transpose/broadcast 不产生额外内核。

**StorageBox.realize() 关键逻辑 (ir.py#L7571)**

```python
def realize(self) -> Optional[str]:
    if isinstance(self.data, (ComputedBuffer, InputsKernel, InputBuffer,
                              ReinterpretView, TemplateBuffer)):
        return self.data.get_name()   # 已物化, 直接返回名字
    assert isinstance(self.data, (Pointwise, Reduction, Scan, Sort)), type(self.data)
    origin_node = self.data.get_origin_node()
    # 将未物化的 Loops 包装为 ComputedBuffer
    self.data = ComputedBuffer(
        name=None,
        layout=FlexibleLayout(
            device=self.data.get_device(),
            dtype=self.data.get_dtype(),
            size=self.data.get_size(),
        ),
        data=self.data,
    )
    self.data.name = V.graph.register_buffer(self.data)        # 注册到 buffers
    V.graph.register_operation(self.data)                      # 注册到 operations
    return self.data.name
```

> **💡 提示：** **Lazy 物化的好处：**Lowering 阶段尽量保持计算为"未物化"的 `Pointwise`/`Reduction` 形式（只记录 `inner_fn` 闭包），直到调度器决定是否融合。如果某个 `Pointwise` 被融合进下游内核，它就永远不需要单独 realize，从而省去一次内存写回。

## 🏷️ 10.5 Lowering 注册机制

定义于 [lowering.py#L100](file:///d:/project/pytorch-2.8.0/torch/_inductor/lowering.py#L100)。Inductor 维护一个全局字典 `lowerings`，建立 `OpOverload → lowering 函数` 的映射：

**torch/_inductor/lowering.py#L100 / #L457**

```python
lowerings: dict[Union[Callable[..., Any], str], Callable[..., Any]] = {}

# 通过装饰器注册每个 ATen 操作的 lowering 函数
def register_lowering(
    aten_fn,
    broadcast=False,
    type_promotion_kind: Optional[ELEMENTWISE_TYPE_PROMOTION_KIND] = ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    convert_input_to_bool=False,
    lowering_dict=lowerings,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """Shim to support decorator syntax."""
    return functools.partial(
        _register_lowering,
        aten_fn,
        broadcast=broadcast,
        type_promotion_kind=type_promotion_kind,
        convert_input_to_bool=convert_input_to_bool,
        lowering_dict=lowering_dict,
    )
```

相关的辅助设施：

- **`needs_realized_inputs`**：某些算子（如 `aten.convolution`、`aten.mm`）要求输入必须是已物化的具体缓冲区，不能是未物化的 `Pointwise`。它们在注册时被加入此集合，lowering 前会强制 realize 输入。
- **`_maybe_layout_constraints`**：基于算子 tag 懒加载的布局约束。由 `tag_to_layout_constraint` 把 `torch._C.Tag` 映射为约束函数。
- **`fallbacks`**：未实现 lowering 的算子回退到 eager 执行（生成 extern 调用）。

### 10.5.1 元素级算子 lowering 示例

**元素级 lowering (lowering.py)**

```python
@register_lowering(aten.add.Tensor)
def add(a, b):
    return pointwise(
        a, b,
        inner_fns=[lambda a, b: a + b],   # 实际计算 (延迟闭包)
        ...,
    )

@register_lowering(aten.relu.default)
def relu(a):
    return pointwise(
        a,
        inner_fns=[lambda a: torch.relu(a)],
        ...,
    )

@register_lowering(prims.convert_element_type, type_promotion_kind=None)
def convert_element_type(x, dtype):
    return pointwise(
        x,
        inner_fns=[lambda x: ops.to_dtype(x, dtype)],
        ...,
    )
```

这些 lowering 都返回 `TensorBox`（其内部是包装 `Pointwise` 的 `StorageBox`），计算被延迟到 `inner_fn` 闭包中，等待调度器决定是否融合。

### 10.5.2 matmul / conv / batch_norm 的 lowering

计算密集型算子（`aten.mm`、`aten.convolution`、`aten._native_batch_norm`）不走 `Pointwise`/`Reduction` IR，而是通过 **模板 / 外部内核** 方式 lowering：

- **加入 `needs_realized_inputs`**：`aten.convolution`、`aten.mm` 等（见 [lowering.py#L199](file:///d:/project/pytorch-2.8.0/torch/_inductor/lowering.py#L199) 起）要求输入先 realize，因为底层 cuBLDNN/cuBLAS 需要连续内存。
- **模板融合 (epilogue fusion)：**在 `max-autotune` 模式下，matmul/conv 后接的 pointwise epilogue（如 `matmul(x,w) + bias → relu`）会被融合进模板内核，通过 `TemplateBuffer` 表示。
- **未支持的回退 (fallback)：**`aten.convolution_backward` 等通过 `make_fallback(aten.convolution_backward, constrain_to_fx_strides)` 注册为外部调用，在生成的代码中以 `at::convolution_backward` 形式调用。
- **batch_norm：**通常先被分解 (decomposition) 为更原子的归约与 pointwise，再走标准 lowering；或调用 fused 内核。

## 🗺️ 10.6 FX 节点 → lowering 函数 → IR 节点映射

**典型映射关系**

```text
FX Node (call_function)        lowering 函数            生成的 IR 节点
─────────────────────────      ─────────────────       ────────────────────────
aten.add.Tensor(a, b)    ──►  add(a, b)          ──►  TensorBox(StorageBox(Pointwise))
aten.relu.default(a)      ──►  relu(a)            ──►  TensorBox(StorageBox(Pointwise))
aten.sum(dim=0)           ──►  sum(a, dims)       ──►  TensorBox(StorageBox(Reduction))
aten.mm(a, b)             ──►  mm(a, b)           ──►  TensorBox(StorageBox(TemplateBuffer
                                                                / ExternKernel))
aten.view(a, shape)       ──►  view(a, shape)     ──►  TensorBox(View)
aten.expand(a, shape)     ──►  expand(a, shape)   ──►  TensorBox(ExpandView)
aten.permute(a, dims)     ──►  permute(a, dims)   ──►  TensorBox(PermuteView)
aten.squeeze(a, dim)      ──►  squeeze(a, dim)    ──►  TensorBox(SqueezeView)
aten.index_put(...)       ──►  index_put(...)     ──►  TensorBox(StorageBox(Scatter))
prims.convert_element_type ─►  convert_elem_type  ──►  TensorBox(StorageBox(Pointwise))
aten.convolution(...)     ──►  convolution(...)   ──►  TemplateBuffer (需 realized inputs)
aten.convolution_backward ──►  make_fallback(...) ──►  ExternKernelNode (回退)

注: View 类不立即产生计算, 只改变后续 Pointwise/Reduction 的索引方式
```

## 📐 10.7 布局优化 (Layout Optimization)

Inductor 会根据图的结构决定是否启用布局优化，核心由 `GraphLowering.decide_layout_opt` 完成，定义于 [graph.py#L594](file:///d:/project/pytorch-2.8.0/torch/_inductor/graph.py#L594)：

**torch/_inductor/graph.py#L594**

```python
def decide_layout_opt(gm: GraphModule, *, is_inference: bool) -> bool:
    """
    基于启发式决定本图是否启用布局优化。
    """
    if not config.layout_optimization:
        return False                  # 全局开关关闭

    if config.force_layout_optimization:
        return True                   # 强制启用

    conv_nodes = [n for n in gm.graph.nodes
                  if n.target == torch.ops.aten.convolution.default]
    nconv = len(conv_nodes)
    if nconv == 0:
        return False                  # 无卷积则不启用

    # CPU + mkldnn: 始终用 channels_last
    if (torch.backends.mkldnn.enabled
            and torch.backends.mkldnn.is_available()
            and all(n.args[idx].meta["val"].device.type in SUPPORTED_MKLDNN_DEVICES
                    for n in conv_nodes for idx in [0, 1])):
        return True

    # 卷积太少 (节点数 >= 300 * 卷积数) 则跳过
    if len(list(gm.graph.nodes)) >= 300 * nconv:
        log.debug("Skipped layout opt because only a few conv")
        return False

    # 动态形状会触发性能回退, 跳过
    if any(has_free_symbols(n.args[idx].meta["val"])
            for n in conv_nodes for idx in [0, 1]):
        log.debug("See perf regression with dynamic shape.")
        return False
    ...
```

**布局优化的本质：**把卷积相关的张量从默认的 NCHW (channels-first) 转换为 **channels-last (NHWC)** 内存布局，使卷积内核访问更连续、更适合 Tensor Core。决策启发式包括：

- **必须存在卷积：**没有卷积的图不需要布局转换。
- **卷积占比：**若图节点数远大于卷积数（≥300 倍），说明卷积非热点，转换开销不划算。
- **动态形状：**已知动态形状在 channels-last 下有性能回退，保守跳过。
- **分组卷积检查：**某些分组卷积布局不兼容，需进一步判定。
- **CPU/mkldnn：**在 CPU 且 mkldnn 可用时，强制使用 channels-last 以获得最佳性能。

启用后，`convert_conv_weights_to_channels_last`（在 freezing 阶段）会把卷积权重预先转成 channels-last 并固化；后续 lowering 生成的 `FlexibleLayout` 会优先选择匹配的 stride 顺序。

### 10.7.1 布局约束 tag

算子可通过 `torch._C.Tag` 声明对输入布局的要求，`tag_to_layout_constraint` 把 tag 映射为约束函数：

| Tag | 约束函数 | 含义 |
| --- | --- | --- |
| `needs_exact_strides` | `constrain_to_fake_tensors` | 输入 stride 必须与编译期 fake tensor 完全一致 |
| `needs_contiguous_strides` | `require_contiguous_strides` | 输入必须连续 |
| `needs_fixed_stride_order` | `constrain_to_fx_strides` | 输入 stride 顺序固定（如 NHWC） |
| `flexible_layout` | `None` | 布局可任意，无约束 |

这些约束在 lowering 时被懒加载到 `_maybe_layout_constraints`，并通过 guard 在运行时校验，确保编译期假设的布局与运行时实际布局一致。

## 📦 10.8 Lowering 产物

Lowering 完成后，`GraphLowering` 持有两组关键数据，作为调度器的输入：

- **`buffers`：**`list[ir.Buffer]`，所有已物化的缓冲区（输入缓冲、常量缓冲、计算缓冲）。
- **`operations`：**`list[ir.Operation]`，所有需要生成代码的操作（主要是 `ComputedBuffer`，以及模板/外部算子）。
- **`graph_outputs`：**`list[ir.IRNode]`，图的输出 IR 节点。

这些产物随后交给 [第十一章 调度器](./11-scheduler-fusion.md) 进行依赖分析、融合与执行排序，最终由代码生成器产出 Triton/C++ 内核。

> **✨ 技巧：** **Lowering 的设计哲学：**尽可能把计算保留为"延迟的 `Pointwise`/`Reduction`"，把布局/形状变换保留为"零成本的 `View`"。这样调度器在 IR 层面看到的是一组无副作用的纯函数闭包，能自由地重排与融合，而把"何时物化、以何种布局物化"的决策推迟到最后一刻。

## Related

- [09 TorchInductor 后端](./09-torchinductor-backend.md) — `inner_compile`（compile_fx_inner）创建 GraphLowering 启动 lowering
- [11 调度器与融合](./11-scheduler-fusion.md) — lowering 产出的 buffers/operations 交给调度器做融合与排序
- [PyTorch 索引](../index.md)
