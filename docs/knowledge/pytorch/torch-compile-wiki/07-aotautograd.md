---
title: AOTAutograd 中间层
type: concept
status: seed
tags: [PyTorch, torch.compile, AOTAutograd, 联合图追踪]
created: 2026-08-17
updated: 2026-08-17
source:
  - d:\project\pytorch-2.8.0\wiki\07_aot_autograd.html
---

# AOTAutograd 中间层

*前向与反向计算图的分离引擎*

AOTAutograd (Ahead-Of-Time Autograd) 是连接 Dynamo 和 Inductor 的中间层，位于 [aot_autograd.py](file:///d:/project/pytorch-2.8.0/torch/_functorch/aot_autograd.py)。它负责将前向和反向计算图分离，并应用梯度重计算优化。与传统的"运行时 autograd"不同，AOTAutograd 在编译期就把前向+反向**联合追踪**为一张图，再分区成独立的前向图与反向图，使后端能对二者分别优化、融合。

## 🔀 7.1 核心职责

- **联合图追踪：**通过 functorch 的 dispatch 机制追踪前向 + 反向的联合计算图
- **算子分解：**将复杂算子分解为更原子的操作，便于后端优化
- **前向/反向分区：**使用最小割算法将联合图拆分为独立的前向图和反向图
- **子类处理：**处理 Tensor 子类（如 DTensor、SparseTensor）的转换

## 🚪 7.2 调用入口

Inductor 的 `compile_fx` 在完成 pre-grad 优化后，调用 `aot_autograd()`：

**torch/_inductor/compile_fx.py**

```python
# Inductor 调用 AOTAutograd
return aot_autograd(
    fw_compiler=fw_compiler,        # 前向图编译器 → inductor inner_compile
    bw_compiler=bw_compiler,        # 反向图编译器 → inductor inner_compile
    inference_compiler=inference_compiler,  # 推理模式编译器
    decompositions=decompositions,  # 算子分解表
    partition_fn=partition_fn,      # min_cut_rematerialization_partition
    keep_inference_input_mutations=True,
    cudagraphs=cudagraphs,
)(model_, example_inputs_)
```

## 🧭 7.3 functorch dispatch 追踪机制

AOTAutograd 的"联合图追踪"建立在 functorch 的 **dispatch tracing** 之上。PyTorch 的算子都注册在 dispatcher 上，每个算子根据 dispatch key（CPU/CUDA/Autograd/CompositeImplicitAutograd 等）路由到不同实现。AOTAutograd 临时切换到一个"追踪 dispatch key"，让所有算子调用被记录为 FX 节点，而非真正执行。

### dispatch 追踪的关键点

- **不真正计算：**追踪时算子被替换为"记录到 FX 图"的实现，输出是携带元数据的 Fake Tensor，不分配真实存储。
- **捕获 autograd 语义：**在 Autograd dispatch 层追踪，能同时看到前向算子与反向算子，从而把二者纳入同一张联合图。
- **与 eager 行为一致：**追踪复用真实的 dispatcher 路由逻辑，因此能正确处理 autograd混合、in-place 操作、视图关系等复杂语义。

这套机制与 `torch.func`（functorch 的公开 API）共享底层。事实上，AOTAutograd 大量使用 `torch.func.functional_call`、`torch.func.functional_grad` 等原语来"函数化"地执行模块并获取梯度。

## 👻 7.4 Fake Tensor 与联合图追踪

追踪期间，所有张量都是 **Fake Tensor**——只携带 dtype/device/shape/stride 等元数据，不持有真实数据。Fake Tensor 让 AOTAutograd 能在没有真实输入的情况下"跑完"前向+反向，把每个算子记录为图节点。

### 联合图追踪的详细流程

```text
Dynamo 产出: forward_graph (GraphModule) + example_inputs
                    │
                    ▼
        ┌───────────────────────────────────────────┐
        │  AOTAutograd 追踪 (aot_autograd)           │
        │                                           │
        │  1. 从 example_inputs 构造 Fake Tensors    │
        │     (携带 shape/dtype/device 元数据)       │
        │                                           │
        │  2. 应用 decompositions 分解表              │
        │     把复杂算子替换为原子算子序列             │
        │                                           │
        │  3. 在 dispatch 层执行前向                  │
        │     - 算子被 trace 实现拦截 → FX 节点        │
        │     - 输出 Fake Tensor (无真实计算)         │
        │     - 记录前向算子 + 中间 Fake Tensor       │
        │                                           │
        │  4. 构造 grad 输出 (ones_like 前向输出)     │
        │                                           │
        │  5. 在 Autograd dispatch 层执行反向         │
        │     (torch.func.functional_grad)           │
        │     - 反向算子被 trace 拦截 → FX 节点        │
        │     - 记录反向算子 + 梯度 Fake Tensor       │
        │                                           │
        │  产出: joint_graph (前向+反向合一)           │
        │        + 节点标记 (哪些属于 fwd / bwd)       │
        └───────────────────┬───────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────────┐
        │  分区 (partition_fn, 详见第八章)            │
        │                                           │
        │  最小割算法: 在 joint_graph 上求最小割      │
        │  - 割的一侧: 保存的前向中间结果 (fwd 输出)  │
        │  - 割的另一侧: 反向时重计算的部分           │
        │                                           │
        │  产出: fwd_graph + bwd_graph              │
        └───────────────────┬───────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
       fw_compiler(fwd_gm)         bw_compiler(bwd_gm)
       → Inductor 编译              → Inductor 编译
       (Triton/C++ 内核)            (Triton/C++ 内核)
```

> **💡 提示：** **Fake Tensor 的意义：**它把"图捕获"与"数值计算"解耦。AOTAutograd 无需真实数据就能确定算子序列、形状传播与梯度结构，因此可以在模型首次调用时完成完整的前向+反向编译，而不必等到反向传播真实发生。

## 📦 7.5 functional_call 与参数提升

`nn.Module` 的参数和缓冲区是"状态"，而追踪需要一个纯函数。AOTAutograd 通过 `aot_module`（[aot_autograd.py#L961](file:///d:/project/pytorch-2.8.0/torch/_functorch/aot_autograd.py#L961)）把参数/缓冲区**提升**为函数输入，使模块变成可追踪的纯函数。

**torch/_functorch/aot_autograd.py#L961**

```python
def aot_module(mod: nn.Module, *args, **kwargs) -> nn.Module:
    """
    追踪 mod 的前向和反向图，使用 torch dispatch tracing 机制。
    它是 aot_function 的包装，底层调用 aot_function 完成追踪与编译。

    aot_module 把 nn.Module 的参数和缓冲区提升为新可调用对象的输入，
    然后通过 aot_function 编译。
    """
    # 见 Note: [Fake Modules and AOTAutograd]
    torch._dynamo.utils.assert_no_fake_params_or_buffers(mod)

    def functional_call(named_params, named_buffers, *args, **kwargs):
        params_and_buffers = {**named_params, **named_buffers}
        return torch.func.functional_call(mod, params_and_buffers, args, kwargs)

    named_params = dict(mod.named_parameters(remove_duplicate=False))
    named_buffers = dict(mod.named_buffers(remove_duplicate=False))
    num_params_buffers = len(named_params) + len(named_buffers)
    compiled_f = aot_function(
        functional_call, *args, num_params_buffers=num_params_buffers, **kwargs
    )

    class AOTModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.orig_module = mod

        def forward(self, *args, **kwargs):
            return compiled_f(
                named_params,
                named_buffers,
                *args,
                **kwargs,
            )

    return AOTModule()
```

### 提升的细节

- **functional_call：**`torch.func.functional_call(mod, params_and_buffers, args)` 在不修改模块状态的情况下，用给定参数/缓冲区执行 `mod`，把"有状态模块"变成"纯函数"。
- **参数去重：**`remove_duplicate=False` 保留所有参数（包括共享参数的不同名字），使追踪期能精确建模每个参数的使用。
- **num_params_buffers：**记录参数+缓冲区总数，让 `aot_function` 知道前多少个输入是" lifted"的参数，从而正确分区前向/反向。
- **AOTModule.forward：**运行时把当前模块的 `named_params/named_buffers` 作为前几个输入传给编译后的函数，使参数更新（如优化器步进）能反映到下次调用。

## 🧩 7.6 算子分解 (Decomposition)

分解表位于 [torch/_decomp/](file:///d:/project/pytorch-2.8.0/torch/_decomp/)，将复杂算子拆分为更简单的原子操作。分解在 AOTAutograd 追踪**之前**应用——追踪开始时，`select_decomp_table()` 返回的分解表被注册到 dispatch，使被分解的算子在追踪期直接走原子实现。

**torch/_decomp/decompositions.py 顶部结构**

```python
# 注册机制: @register_decomposition(aten.XXX) 把函数注册为 XXX 的分解
from torch._decomp import register_decomposition
aten = torch._ops.ops.aten

# type_casts 装饰器统一处理类型提升逻辑
def type_casts(f, type_promotion, compute_dtype_only=False, ...):
    @functools.wraps(f)
    def inner(*args, **kwargs):
        # 计算 computation_dtype / result_dtype
        # 把输入提升到 computation_dtype 后调用 f
        # 再把结果降回 result_dtype
        ...
    return inner

compute_only_pw_cast_for_opmath = partial(type_casts, ...)
```

### 分解示例：softmax

**分解示例**

```python
# torch._decomp/decompositions.py

# softmax 分解为 exp + sum + div 等原子操作
@register_decomposition(aten._softmax)
def softmax(x, dim, half_to_float):
    x = x.to(None if half_to_float else torch.float)
    max_val = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_val)
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    return exp_x / sum_exp

# 分解的好处:
# 1. 让后端 (Inductor) 看到更细粒度的操作，便于融合
# 2. 减少后端需要支持的算子数量
# 3. 统一不同框架的操作粒度
```

### 更多分解类别

分解表覆盖大量 ATen 算子，主要分几大类（具体实现见 `torch/_decomp/decompositions.py`）：

- **归约类：**如 `var` 分解为 $mean(x^2) - mean(x)^2$，让后端能融合归约与后续元素级操作。
- **激活类：**如 `softplus`、`celu` 等分解为 `exp/log/maximum` 等原子操作（含数值稳定处理）。
- **归一化类：**如 `layer_norm`、`softmax` 分解为归约 + 元素级组合，便于 Inductor 融合。
- **类型提升：**`type_casts` 装饰器统一封装各分解的类型提升策略（`ELEMENTWISE_TYPE_PROMOTION_KIND`），保证分解前后语义一致。

> **✨ 技巧：** **分解的收益：**把 `softmax` 这样的"复合算子"分解后，Inductor 看到 `exp/sub/sum/div` 等点级+归约算子，可以将它们与前后算子融合进同一内核，显著减少内存访问。这是 torch.compile 在 memory-bound 场景下大幅超越 eager 的关键之一。

## 🎭 7.7 Tensor 子类处理

PyTorch 支持多种 **Tensor 子类**：DTensor（分布式）、SparseTensor、FakeTensor、NestedTensor 等。这些子类在 dispatch 时有特殊行为，AOTAutograd 必须正确处理它们，否则追踪结果会失真。

- **子类 → 真实张量转换：**追踪前，AOTAutograd 会把输入中的子类张量"去子类化"为普通张量（或 Fake Tensor），使追踪在统一的 dispatch key 上进行。
- **子类元数据 Guard：**Dynamo 侧通过 `TENSOR_SUBCLASS_METADATA_MATCH` Guard 确保运行时输入的子类身份与编译期一致，不一致则重编译。
- **梯度重计算兼容：**分区器在求最小割时，会考虑子类张量的特殊性，避免把不可重算的子类操作（如涉及 RNG 的分布式同步）放到重计算侧。
- **输入变异处理：**`keep_inference_input_mutations` 等选项控制推理模式下对输入的变异是否保留，子类的变异语义比普通张量更复杂，需特殊处理。

> **⚠️ 注意：** **子类与图断裂：**某些 Tensor 子类（特别是改变 dispatch 行为的）可能在 AOTAutograd 阶段触发额外处理甚至回退。如果遇到子类相关的编译失败，可先用 `torch._dynamo.explain` 排查，确认是否需要显式去子类化。

## 🔗 7.8 与 torch.func 的关系

> **📝 备注：** **AOTAutograd 与 torch.func：**AOTAutograd 与 `torch.func`（functorch 的公开 API）共享同一套底层 dispatch 追踪基础设施。AOTAutograd 内部大量使用 `torch.func.functional_call`（函数化执行模块）、`torch.func.functional_grad`（函数化求梯度）等原语。可以说：`torch.func` 提供了"函数化变换"的能力，而 AOTAutograd 把这些能力组合成"编译期前向+反向联合追踪"的流水线，再交给后端（Inductor）生成代码。理解 `torch.func` 的 `vmap`/`grad`/`functional_call` 有助于深入理解 AOTAutograd 的工作原理。

二者的关键区别在于**触发时机**：`torch.func` 是用户显式调用的函数化变换（运行时），而 AOTAutograd 在 torch.compile 编译期自动应用，对用户透明。这种"编译期 autograd"使得后端能对前向与反向分别优化（甚至跨算子融合），是 torch.compile 在训练场景下加速的核心机制之一。

## Related

- [06 图断裂](./06-graph-break.md) — Dynamo 产出的子图进入 AOTAutograd 做联合图追踪
- [08 前向/反向分区策略](./08-partition-strategy.md) — AOTAutograd 追踪后的联合图由分区器拆分为前向/反向
- [PyTorch 索引](../index.md)
