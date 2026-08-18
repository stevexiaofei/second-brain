---
title: TorchInductor 后端
type: concept
status: seed
tags: [PyTorch, torch.compile, Inductor, 后端]
created: 2026-08-17
updated: 2026-08-17
source:
  - d:\project\pytorch-2.8.0\wiki\09_inductor.html
---

# TorchInductor 后端

*从 FX 图到高效可执行代码*

## ⚙️ 9.1 Inductor 概述

Inductor 是 torch.compile 的默认后端，位于 `torch/_inductor/`。它将 FX 图转换为高效的 Triton (GPU) 或 C++ (CPU) 代码。尽管入口函数 `compile_fx` 位于 `_inductor` 包内，但它实际承担了 **编排端到端编译流程** 的职责——包括调用 AOTAutograd 进行联合图追踪与分区，最终通过 `inner_compile` 回调执行真正的代码生成。

> **💡 提示：** **设计要点：**`compile_fx` 接管输入 `model_` 的所有权，可能对其进行变异。若需保留原始 GraphModule，应在调用前自行拷贝。

## 🛣️ 9.2 Inductor 编译流水线

```text
FX GraphModule (来自 Dynamo)
    │
    ▼
┌─────────────────────────────┐
│  1. Pre-grad 图优化          │  compile_fx.py: _recursive_pre_grad_passes()
│     - 模式匹配与替换         │  config.add_pre_grad_passes / remove_pre_grad_passes
│     - 通用图变换             │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  2. AOTAutograd 调用         │  aot_autograd()
│     - 联合图追踪             │  → fw_compiler / bw_compiler / inference_compiler
│     - partition_fn 分区       │
└─────────────┬───────────────┘
              │
    ┌─────────┴─────────┐
    ▼                   ▼
  fw_compiler         bw_compiler
    │                   │
    ▼                   ▼
┌─────────────────────────────┐
│  3. Joint 图优化             │  _recursive_joint_graph_passes()
│     - 常量折叠               │  joint_graph_passes()
│     - 死代码消除             │
│     - 布局优化               │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  4. GraphLowering            │  graph.py: GraphLowering
│     - 逐节点解释 FX 图        │  (FX Interpreter)
│     - 调用 lowering 函数      │
│     - FX Node → IR Node     │  lowering.py
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  5. 调度器                   │  scheduler.py: Scheduler
│     - 依赖分析               │
│     - 算子融合               │
│     - 执行排序               │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  6. 代码生成                 │  codegen/
│     - Triton 内核 (GPU)      │  codegen/triton.py
│     - C++ 代码 (CPU)         │  codegen/cpp.py
│     - CUDA Graphs (可选)     │
└─────────────┬───────────────┘
              │
              ▼
        编译后的可执行函数
```

## 🚪 9.3 compile_fx：Inductor 入口

定义于 [compile_fx.py#L1977](file:///d:/project/pytorch-2.8.0/torch/_inductor/compile_fx.py#L1977)：

**torch/_inductor/compile_fx.py#L1977**

```python
def compile_fx(
    model_: GraphModule,
    example_inputs_: Sequence[InputType],
    inner_compile: Callable[..., OutputCode] = compile_fx_inner,
    config_patches: Optional[dict[str, Any]] = None,
    decompositions: Optional[dict[OpOverload, Callable[..., Any]]] = None,
    ignore_shape_env: bool = False,
) -> Union[Callable, str, list[str], Weights]:
    """
    Inductor 主入口。尽管位于 _inductor 中，此函数负责调用 AOT Autograd，
    最终通过 inner_compile 回调执行实际编译。
    本函数对输入 model_ 拥有所有权并可能变异它。
    """

    # 1. config_patches 短路：在配置补丁上下文中递归调用自身
    if config_patches:
        with config.patch(config_patches):
            return compile_fx(
                model_,
                example_inputs_,
                inner_compile=config.patch(config_patches)(inner_compile),
                decompositions=decompositions,
                ignore_shape_env=ignore_shape_env,
            )

    # 2. cpp_wrapper 短路：递归以打破 cpp_wrapper 递归
    if config.cpp_wrapper:
        with config.patch({"cpp_wrapper": False, **get_cpp_wrapper_config()}), \
             V.set_real_inputs(example_inputs_):
            ...
            return compile_fx(patched_mod, fake_args, ...)

    # 3. 真正的工作
    with (_use_lazy_graph_module(...), enable_python_dispatcher(), ...):
        model_ = _recursive_pre_grad_passes(model_, example_inputs_)
        ...
```

## 🧩 9.4 config_patches 机制

`compile_fx` 接受 `config_patches` 参数，允许调用方在不全局修改配置的前提下，为本次编译注入临时配置：

**config_patches 处理**

```python
if config_patches:
    with config.patch(config_patches):
        return compile_fx(
            model_,
            example_inputs_,
            # 反向编译在另一个作用域中执行，需要额外包裹一层 patch
            inner_compile=config.patch(config_patches)(inner_compile),
            decompositions=decompositions,
            ignore_shape_env=ignore_shape_env,
        )
```

> **📝 备注：** **config_patches 工作原理：**`config.patch(patches)` 是一个上下文管理器，进入时将 `patches` 中的键值临时覆盖到 `torch._inductor.config` 上，退出时恢复原值。由于反向图编译发生在外层 `with` 作用域之外（在 AOTAutograd 的回调里），所以 `inner_compile` 也被 `config.patch(config_patches)` 再次装饰，确保反向编译时配置依然生效。这种"短路递归"模式让配置补丁对所有阶段（前向/反向/推理）都一致生效。

## 🧹 9.5 _recursive_pre_grad_passes：梯度前优化

定义于 [compile_fx.py#L456](file:///d:/project/pytorch-2.8.0/torch/_inductor/compile_fx.py#L456)，在 AOTAutograd 追踪 *之前* 执行的图优化阶段：

**torch/_inductor/compile_fx.py#L456**

```python
def _recursive_pre_grad_passes(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
) -> GraphModule:
    with dynamo_timed("_recursive_pre_grad_passes",
                     log_pt2_compile_event=True,
                     dynamo_compile_column_us="pre_grad_pass_time_us"):
        add_passes = config.add_pre_grad_passes
        remove_passes = config.remove_pre_grad_passes
        # 递归处理所有子图 (如 invoke_subgraph HOP)
        for subgraph_name in _get_subgraph_names(gm):
            subgraph = getattr(gm, subgraph_name)
            new_subgraph = _recursive_pre_grad_passes(subgraph, ())
            setattr(gm, subgraph_name, new_subgraph)
        return pre_grad_passes(gm, example_inputs, add_passes, remove_passes)
```

**主要优化内容：**

- **模式匹配与替换：**将常见算子组合替换为更高效的形式（如 fused attention 模式、SDP 初始化）。
- **通用图变换：**常量折叠、冗余算子消除、形状推断简化。
- **可配置 pass 列表：**通过 `config.add_pre_grad_passes` / `config.remove_pre_grad_passes` 动态增删优化 pass。
- **递归子图：**对 `invoke_subgraph` 等高阶算子的子图递归应用同一组 pass，保证嵌套结构也被优化。

由于此阶段发生在 autograd 追踪之前，所做变换会同时影响前向与反向图。

## 🔗 9.6 _recursive_joint_graph_passes：联合图优化

定义于 [compile_fx.py#L475](file:///d:/project/pytorch-2.8.0/torch/_inductor/compile_fx.py#L475)，在 AOTAutograd 追踪得到联合图后、分区前执行：

**torch/_inductor/compile_fx.py#L475**

```python
def _recursive_joint_graph_passes(
    gm: GraphModule, skip_invoke_subgraph: bool = False
) -> None:
    with dynamo_timed("_recursive_joint_graph_passes",
                     log_pt2_compile_event=True,
                     dynamo_compile_column_us="joint_graph_pass_time_us"):
        # invoke_subgraph 已在 run_joint_graph_passes_on_hops 中分区前递归处理过
        for subgraph_name in _get_subgraph_names(gm, skip_invoke_subgraph):
            subgraph = getattr(gm, subgraph_name)
            _recursive_joint_graph_passes(subgraph, skip_invoke_subgraph)
        joint_graph_passes(gm)
```

该函数在 `partition_fn` 内被调用（见 9.7），对联合前向-反向图执行优化：

- **常量折叠：**将可静态求值的常量子图提前计算并内联。
- **死代码消除：**移除无输出消费者的节点。
- **布局优化预备：**为后续 Inductor 的 `layout_opt` 决策铺垫。
- **跳过 invoke_subgraph：**当外层已对 HOP 子图递归处理时，通过 `skip_invoke_subgraph=True` 避免重复处理。

## 🎯 9.7 三个编译回调：fw_compiler / bw_compiler / inference_compiler

`compile_fx` 在调用 `aot_autograd` 时传入三个编译器回调，分别处理不同场景的子图：

**三个编译回调定义 (compile_fx.py#L2260-L2345)**

```python
# 前向编译器: 委托给 fw_compiler_base, 标记为非推理
fw_compiler: Callable = functools.partial(fw_compiler_base, is_inference=False)
fw_compiler = SerializableAOTDispatchCompiler(OutputCode, fw_compiler)

# 推理编译器: 若开启 freezing 且非梯度模式, 用 fw_compiler_freezing
if config.freezing and not torch.is_grad_enabled():
    inference_compiler = functools.partial(
        fw_compiler_freezing,
        dynamo_model=model_,
        num_example_inputs=num_example_inputs,
        inner_compile=inner_compile,
        cudagraphs=cudagraphs,
        ...
    )
else:
    inference_compiler = functools.partial(fw_compiler_base, is_inference=True)

# 反向编译器: 调用 inner_compile, 标记 is_backward=True
@compile_time_strobelight_meta(phase_name="backward")
def bw_compiler(gm, example_inputs):
    with dynamo_utils.dynamo_timed("compile_fx.<locals>.bw_compiler"), compile_lock:
        fixed = count_tangents(gm)
        return inner_compile(
            gm, example_inputs,
            static_input_idxs=list(range(fixed)),
            cudagraphs=cudagraphs,
            is_backward=True,
            graph_id=graph_id,
            ...
        )

# 分区函数: 先联合图优化, 再最小割分区
def partition_fn(gm, joint_inputs, **kwargs):
    with cuda_context:
        _recursive_joint_graph_passes(gm, skip_invoke_subgraph=True)
    return min_cut_rematerialization_partition(
        gm, joint_inputs, compiler="inductor", **kwargs
    )
```

| 回调 | 触发场景 | 核心行为 |
| --- | --- | --- |
| `fw_compiler` | 训练模式的前向子图 | `fw_compiler_base(is_inference=False)` → `inner_compile` |
| `bw_compiler` | 训练模式的反向子图 | `inner_compile(is_backward=True)`，统计 tangents 数量 |
| `inference_compiler` | 推理模式（无梯度） | 开启 freezing 时走 `fw_compiler_freezing`；否则 `fw_compiler_base(is_inference=True)` |
| `partition_fn` | AOTAutograd 分区时 | 先 `_recursive_joint_graph_passes`，再 `min_cut_rematerialization_partition` |

三者最终都汇聚到 `inner_compile`（即 `compile_fx_inner`），后者负责真正的 Lowering → 调度 → 代码生成。

## ❄️ 9.8 模型冻结 (Model Freezing)

当 `config.freezing` 为真且当前非梯度模式（`torch.is_grad_enabled()` 为 False）时，推理路径会走 `fw_compiler_freezing`，定义于 [compile_fx.py#L1838](file:///d:/project/pytorch-2.8.0/torch/_inductor/compile_fx.py#L1838)。

**torch/_inductor/compile_fx.py#L1838**

```python
def fw_compiler_freezing(
    aot_autograd_model: GraphModule,
    aot_example_inputs: Sequence[InputType],
    dynamo_model: GraphModule,
    num_example_inputs: int,
    inner_compile: Callable[..., Any],
    cudagraphs: BoxedBool,
    graph_id: int,
    forward_device: BoxedDeviceIndex,
) -> Callable:
    from torch._inductor.freezing import convert_conv_weights_to_channels_last, freeze

    # partition_fn 不会被调用 (推理无反向)
    _recursive_joint_graph_passes(aot_autograd_model)

    layout_opt = GraphLowering.decide_layout_opt(aot_autograd_model, is_inference=True)
    if layout_opt:
        fake_tensor_prop(aot_autograd_model, aot_example_inputs, True)
        convert_conv_weights_to_channels_last(aot_autograd_model)

    # 核心: 冻结常量参数, 折叠可静态计算的部分
    opt_model, preserved_arg_indices = freeze(
        dynamo_model, aot_autograd_model, aot_example_inputs,
    )
    aot_example_inputs = [aot_example_inputs[ind] for ind in preserved_arg_indices]
    ...
```

**Freezing 的核心步骤：**

- **跳过分区：**推理模式下无反向图，`partition_fn` 不会被调用，直接对整个前向图做联合优化。
- **布局优化决策：**调用 `GraphLowering.decide_layout_opt(..., is_inference=True)` 决定是否启用布局优化（如 channels-last）。
- **卷积权重转换：**若启用布局优化，`convert_conv_weights_to_channels_last` 将卷积权重预先转换为 channels-last 布局并固化。
- **常量折叠 (freeze)：**将模型参数视为常量，把可在编译期求值的子图（如常量张量构造、形状计算）提前算出并内联到代码中，减少运行时开销。返回保留的输入索引 `preserved_arg_indices`。
- **用户可见输出标记：**确保冻结后图的输出仍对用户可见，避免被死代码消除误删。

> **✨ 技巧：** **何时触发：**仅当 `config.freezing=True` 且 `torch.is_grad_enabled()=False` 时触发。即典型的推理部署场景：参数固定、无需反向，可把权重烘焙进生成的内核中。

## 📦 9.9 compile_fx_aot：AOT 编译导出

定义于 [compile_fx.py#L1773](file:///d:/project/pytorch-2.8.0/torch/_inductor/compile_fx.py#L1773)，用于 **提前编译导出 (AOT Inductor)**，将模型编译为可独立部署的 `.so` 共享库。

**torch/_inductor/compile_fx.py#L1773**

```python
def compile_fx_aot(
    model_: GraphModule,
    example_inputs_: list[InputType],
    inner_compile: _CompileFxCallable = compile_fx_inner,
    config_patches: Optional[dict[str, str]] = None,
) -> Union[list[Union[str, Weights]], str]:
    assert isinstance(model_, GraphModule), model_

    # [See NOTE] Unwrapping subclasses AOT
    unwrap_tensor_subclass_parameters(model_)

    # AOT 模式强制启用 cpp_wrapper
    config_patches = (
        {"cpp_wrapper": True}
        if config_patches is None
        else {**config_patches, "cpp_wrapper": True}
    )

    output_path = config_patches.get(
        "aot_inductor.output_path", config.aot_inductor.output_path
    )
    if output_path:
        assert not output_path.endswith(".pt2"), ...
    else:
        config_patches = {
            **config_patches,
            "aot_inductor.output_path": code_hash(model_.code),
        }

    with V.set_aot_compilation(True), \
         torch._guards.compile_context(saved_compile_context), \
         chromium_event_timed("compile_fx_aot", ...):
        compiled_artifacts = compile_fx(
            model_, example_inputs_,
            inner_compile=functools.partial(inner_compile, extern_node_serializer=extern_node_serializer),
            config_patches=config_patches,
        )
        assert isinstance(compiled_artifacts, CompiledAOTI)
        return compiled_artifacts.filename
```

| 对比维度 | compile_fx | compile_fx_aot |
| --- | --- | --- |
| 用途 | 即时编译 (JIT)，运行时执行 | 提前编译 (AOT)，导出部署产物 |
| cpp_wrapper | 可选 | 强制启用 |
| 输出 | 可调用对象 | `.so` 文件路径 (CompiledAOTI.filename) |
| 张量子类 | 保留 | `unwrap_tensor_subclass_parameters` 解包 |
| V.aot_compilation | False | True |
| output_path | 无 | 显式指定或按代码哈希自动生成 |
| extern_node_serializer | 无 | 传入以序列化外部算子 |

简言之，`compile_fx_aot` 是 `compile_fx` 的"导出版本"：它在 `V.set_aot_compilation(True)` 上下文中复用 `compile_fx` 的全部流程，但额外强制 C++ wrapper、解包子类、并最终产出可序列化的 `.so`，便于脱离 Python 环境部署（如通过 `torch._inductor.aoti_load_package` 加载）。

## 📚 9.10 编译流水线全景

把上述阶段串起来，`compile_fx` 的完整流程如下：

```text
compile_fx(model_, example_inputs_, ...)
  │
  ├─ config_patches?  →  with config.patch: 递归 compile_fx
  ├─ cpp_wrapper?     →  with cpp 配置: 递归 compile_fx
  │
  └─ with lazy_graph_module / python_dispatcher / preserve_node_meta:
       │
       ├─ 1. _recursive_pre_grad_passes(model_)         [pre-grad 优化]
       │
       ├─ 2. 定义 fw_compiler / bw_compiler / inference_compiler / partition_fn
       │     (partition_fn = _recursive_joint_graph_passes + min_cut_partition)
       │
       ├─ 3. 选择 decompositions (select_decomp_table)
       │
       └─ 4. aot_autograd(fw_compiler, bw_compiler, inference_compiler,
                          decompositions, partition_fn, cudagraphs)
              │
              ├─ 联合图追踪 (前向 + 反向)
              ├─ partition_fn:
              │    ├─ _recursive_joint_graph_passes  [joint 优化]
              │    └─ min_cut_rematerialization_partition  [分区]
              │
              ├─ fw_compiler(fwd_gm)  ──► inner_compile ──► Lowering/Schedule/Codegen
              ├─ bw_compiler(bwd_gm)  ──► inner_compile ──► Lowering/Schedule/Codegen
              └─ inference_compiler   ──► (freezing? fw_compiler_freezing : fw_compiler_base)
                                          ──► inner_compile ──► Lowering/Schedule/Codegen
```

三个回调最终都进入 `inner_compile`（`compile_fx_inner`），后者创建 `GraphLowering` 实例，逐节点解释 FX 图并生成 IR，再交由调度器与代码生成器产出最终的 Triton/C++ 内核。这部分将在 [第十章 Lowering](./10-lowering-fx-to-ir.md) 与 [第十一章 调度器](./11-scheduler-fusion.md) 详述。

## Related

- [08 前向/反向分区策略](./08-partition-strategy.md) — Inductor 的 `partition_fn` 使用最小割分区拆分联合图
- [10 Lowering: FX → IR](./10-lowering-fx-to-ir.md) — `inner_compile` 创建 GraphLowering 逐节点生成 IR
- [PyTorch 索引](../index.md)
