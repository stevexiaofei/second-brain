---
title: 前向/反向分区策略
type: concept
status: seed
tags: [PyTorch, torch.compile, 分区, 最小割, 重计算]
created: 2026-08-17
updated: 2026-08-21
source:
  - d:\project\pytorch-2.8.0\wiki\08_partitioner.html
---

# 前向/反向分区策略

*最小割算法与梯度重计算优化*

## ✂️ 分区概述

分区 (Partitioning) 是 AOTAutograd 的关键优化阶段。AOTAutograd 追踪得到的 `joint_module` 是一个前向与反向合一的计算图，分区器需要将其拆分为两个独立的 `fx.GraphModule`：前向图 (`fwd_module`) 与反向图 (`bwd_module`)。

核心难点在于：联合图中的某些前向中间结果既可能被"保存"供反向直接使用，也可能在反向时"重计算"。如何在这两者之间取舍，是一个典型的 **计算-内存权衡** 问题。PyTorch 通过 **最小割 (Min-Cut)** 算法自动求解。所有逻辑定义于 `torch/_functorch/partitioners.py`。

> **💡 提示：** **分区的输入输出：**输入为 `joint_module`（前向+反向联合图）和 `num_fwd_outputs`（前向输出数量），输出为 `(fwd_module, bwd_module)` 二元组。前向图的额外输出即为"保存给反向的中间结果"。

## 🟦 8.1 default_partition：简单分区

定义于 `torch/_functorch/partitioners.py#L915`，行为最接近原始 `.forward()` / `.backward()` 的划分。

**torch/_functorch/partitioners.py#L915**

```python
def default_partition(
    joint_module: fx.GraphModule,
    _joint_inputs,
    *,
    num_fwd_outputs,
    static_lifetime_input_indices: Optional[list[int]] = None,
    static_lifetime_input_nodes: Optional[OrderedSet[fx.Node]] = None,
) -> tuple[fx.GraphModule, fx.GraphModule]:
    """
    将 joint_module 按原始 forward/backward 行为划分。
    forward 图包含前向输入到前向输出之间的算子，
    其余算子放入 backward 图。需要 stashed 的张量成为 forward 图的输出。
    """
    # 关键：若图存在可重计算算子，则自动升级为最小割分区
    if has_recomputable_ops(joint_module):
        return min_cut_rematerialization_partition(
            joint_module,
            _joint_inputs,
            num_fwd_outputs=num_fwd_outputs,
            static_lifetime_input_indices=static_lifetime_input_indices,
        )
    primal_inputs = list(filter(_is_primal, joint_module.graph.nodes))
    fwd_seed_offset_inputs = list(filter(_is_fwd_seed_offset, joint_module.graph.nodes))
    inputs = primal_inputs + fwd_seed_offset_inputs
    fwd_outputs, bwd_outputs = _extract_fwd_bwd_outputs(
        joint_module, num_fwd_outputs=num_fwd_outputs
    )
    forward_only_graph = _extract_graph_with_inputs_outputs(
        joint_module.graph, inputs, fwd_outputs, "forward"
    )
```

其工作流程：

- **可重计算检测：**先调用 `has_recomputable_ops(joint_module)`，若存在可重计算的算子，则自动委派给 `min_cut_rematerialization_partition`，以获得更好的内存效率。
- **节点分类：**通过 `_is_primal` 过滤 primal 输入，通过 `_is_fwd_seed_offset` 过滤前向 RNG 种子/偏移输入。
- **子图抽取：**用 `_extract_graph_with_inputs_outputs` 抽取前向子图，剩余作为反向子图。

**特点：**实现简单，前向所有中间结果全部保存，无额外重计算开销；缺点是内存占用高。

## 🔬 8.2 min_cut_rematerialization_partition：最小割分区

定义于 `torch/_functorch/partitioners.py#L2491`，是 Inductor 的默认分区策略。核心思想是 **用计算换内存带宽**：反向传播时重计算部分前向操作，从而减少需要保存的中间结果。

**torch/_functorch/partitioners.py#L2491**

```python
def min_cut_rematerialization_partition(
    joint_module: fx.GraphModule,
    _joint_inputs,
    compiler="inductor",
    *,
    num_fwd_outputs,
    static_lifetime_input_indices: Optional[list[int]] = None,
) -> tuple[fx.GraphModule, fx.GraphModule]:
    """
    通过最小割算法分区，使反向传播时重计算部分前向操作，
    以牺牲计算换取内存带宽的优化。

    核心思想:
    1. 将联合图建模为流网络
    2. 节点之间的边有权重（代表保存该中间结果的"代价"）
    3. 求最小割: 决定哪些中间结果保存（在割的一侧）vs 重计算（在另一侧）
    4. 可重计算的算子（如 pointwise ops）权重低 → 更倾向于重计算
    5. 不可重计算的算子（如 dropout, RNG）权重高 → 必须保存
    """

    joint_module.graph.eliminate_dead_code()
    joint_module.recompile()
    fx_g = joint_module.graph

    # 1. CSE (公共子表达式消除) pass
    if config.cse:
        cse_graph = fx_graph_cse(fx_g)
        joint_module.graph = cse_graph
    joint_graph = joint_module.graph

    # 2. 检测可重计算算子 / 可重计算 RNG 算子
    graph_has_recomputable_ops = has_recomputable_ops(joint_module)
    graph_has_recomputable_rng_ops = has_recomputable_rng_ops(joint_module)
    if graph_has_recomputable_ops:
        joint_module = cleanup_recompute_tags(joint_module)
    if not config.unsafe_allow_optimization_of_collectives:
        force_save_collectives(joint_module)

    # 3. 分类节点: 哪些必须在反向
    required_bw_nodes = classify_nodes(joint_module, static_lifetime_input_indices)
    ...
```

## 🧮 8.3 最小割算法详解

### 8.3.1 节点分类 (classify_nodes)

分区器首先调用 `classify_nodes` 确定哪些节点必须位于反向图中：

**classify_nodes 流程**

```python
def classify_nodes(joint_module, static_lifetime_input_indices):
    name_to_node = get_name_to_node(joint_module.graph)
    required_bw_nodes: OrderedSet[fx.Node] = OrderedSet()
    for node in joint_module.graph.nodes:
        # 反向输入 (tangents) 必须在反向
        if node.op == "placeholder" and "tangents" in node.target:
            required_bw_nodes.add(node)
        # 通过 _must_be_in_backward 判定（如归约的梯度）
        elif _must_be_in_backward(node):
            required_bw_nodes.add(node)

        # 传播: 节点的所有下游用户也必须在反向
        if node in required_bw_nodes:
            required_bw_nodes.update(node.users)
    return required_bw_nodes
```

分类规则：

- **primal inputs：**模型原始输入（参数、激活），属于前向。
- **tangents：**反向传播的梯度输入，必须位于反向图。
- **fwd_seed_offset_inputs：**RNG 种子与偏移，用于重计算随机算子。
- **_must_be_in_backward：**某些算子（如归约的反向）天然属于反向。
- **传播规则：**一旦某节点被标记为反向节点，其所有 user（下游消费者）也必须留在反向图中——这是为了保证反向图的闭包性。

### 8.3.2 流网络构建 (Flow Network)

最小割算法将联合图建模为一个 **有向流网络**，包含虚拟的源点 (source) 与汇点 (sink)：

**流网络构建示意**

```text
                    SOURCE (前向侧)
                      │
           ┌──────────┴──────────┐
           ▼                     ▼
       ┌───────┐  edge=∞    ┌───────┐
       │ fwd A │ ─────────► │ fwd B │   前向节点链
       │ (conv)│             │ (relu)│   (可重计算)
       └───┬───┘             └───┬───┘
           │                     │
           │  edge = save_cost   │  edge = save_cost
           │  = size * factor    │  (低权重 → 倾向重计算)
           ▼                     ▼
       ┌───────┐             ┌───────┐
       │ bwd A'│ ◄───────── │ bwd B'│   反向节点链
       │ (grad)│  edge=∞    │ (grad)│
       └───┬───┘             └───┬───┘
           │                     │
           └──────────┬──────────┘
                      ▼
                    SINK (反向侧)

    最小割 = 在 SOURCE 与 SINK 之间切代价最小的边集
    ─ 被切断的 "save_cost" 边 → 该中间结果【重计算】
    ─ 未被切断的边        → 该中间结果【保存】
```

**边的构造与权重：**

- **同侧节点之间：**权重设为 `∞`（无穷大），表示不应切断正常的数据流。
- **"保存 vs 重计算" 边：**对每个可重计算的前向中间结果，添加一条从"前向侧"到"反向侧"的边，权重为 **保存代价**。
- **保存代价公式：**$save\_cost = tensor\_size \times runtime\_factor$。`tensor_size` 由 fake tensor 的元素数 × dtype 字节估算；`runtime_factor` 是经验权重，反映保存该张量的访存代价。
- **不可重计算算子：**对于 dropout、RNG 等无法确定性重算的算子，其输出边权重设为 `∞`，强制保存。
- **collective 算子：**在未开启 `unsafe_allow_optimization_of_collectives` 时，`force_save_collectives` 会强制保存通信算子的输出。

### 8.3.3 求解最小割

构建完流网络后，调用 **最大流/最小割** 算法（基于 `networkx` 的实现）求从 SOURCE 到 SINK 的最小割。最小割将节点集合划分为两侧：

- **割的前向侧（含 SOURCE）：**这些节点的输出若被反向需要，则 **保存** 为前向图的额外输出。
- **割的反向侧（含 SINK）：**这些前向节点会在反向图中 **重计算**，不保存其结果。

最后根据最小割结果，分别为前向图和反向图设置输出节点，并运行 `eliminate_dead_code` 清理无用节点，生成最终的 `fwd_module` 和 `bwd_module`。

## 🔁 8.4 可重计算算子 (Recomputable Ops)

最小割的有效性依赖于"哪些算子可以被廉价地重计算"。Inductor 通过 `has_recomputable_ops` 与 `has_recomputable_rng_ops` 进行检测。

### 8.4.1 has_recomputable_ops

遍历联合图，检查是否存在被标记为 `recompute=` 的算子。判定的核心原则：

- **计算廉价：**元素级算子（pointwise ops）如 `add`、`mul`、`relu`、`sigmoid`、`tanh` 计算成本极低，适合重计算。
- **无副作用：**纯函数式算子可重算；带 inplace 修改、 RNG、IO 的算子不可重算。
- **输入可恢复：**若算子的输入本身可重计算或可保存，该算子才可重算。
- **计算密集度低：**归约 (Reduction)、矩阵乘 (matmul)、卷积 (conv) 等计算密集型算子一般不重计算（重算代价过高）。

### 8.4.2 has_recomputable_rng_ops

专门检测是否存在 **可重计算的随机算子**（如 dropout）。这类算子之所以"可重计算"，是因为 AOTAutograd 在追踪时记录了前向的 RNG 种子与偏移 (`fwd_seed_offset_inputs`)，反向时只要用相同种子重新调用 RNG，就能得到与原始前向完全相同的随机掩码。

> **✨ 技巧：** **RNG 重计算的关键：**反向重算 dropout 时必须复现前向的随机掩码，否则梯度方向错误。这就是为什么联合图要把 `fwd_seed` 与 `fwd_offset` 作为输入保存并传给反向图的原因。

### 8.4.3 cleanup_recompute_tags

当 `has_recomputable_ops` 返回真时，调用 `cleanup_recompute_tags(joint_module)` 清理/规整节点上的 `recompute` 标记，确保后续最小割分析看到一致的状态。

## 📊 8.5 直观示意

```text
联合图 (前向 → 反向):

  前向部分                              反向部分
  ┌─────────────────────────┐          ┌──────────────────┐
  │  A ──→ B ──→ C ──→ D    │          │ D' ──→ C' ──→ ...│
  │  (conv)  (relu) (bn)    │          │ (grad) (grad)    │
  └─────────────────────────┘          └──────────────────┘

default_partition (全保存):
  前向输出: {A, B, C, D}  →  反向直接使用
  内存: 高 (保存所有中间结果)
  计算: 无额外重计算

min_cut_partition (重计算优化):
  前向输出: {A, D}  →  只保存不可重计算的 (conv 输出 / bn 输入)
  反向时重计算: B (relu), C (bn)  ← 这些算子计算便宜
  内存: 低 (只保存必要的)
  计算: 略增 (重计算 relu, bn)

  → 中间结果 B、C 留在寄存器中由 Inductor 后续融合处理，无需写回显存
```

## ⚖️ 8.6 两种分区策略对比

| 对比维度 | default_partition | min_cut_rematerialization_partition |
| --- | --- | --- |
| 实现位置 | `partitioners.py#L915` | `partitioners.py#L2491` |
| 内存占用 | 高（保存全部前向中间结果） | 低（仅保存最小割决定的必要结果） |
| 计算开销 | 无额外重计算 | 略有增加（重算廉价算子） |
| 算法复杂度 | $O(V+E)$ 图遍历 | 最大流算法，复杂度较高 |
| 可重计算检测 | 仅作是否升级的判定 | 核心环节，逐节点判定权重 |
| 是否默认 | 否（但 `has_recomputable_ops` 为假时回退到此） | 是（Inductor 默认） |
| 适用场景 | 计算密集、显存充裕；或无可重计算算子 | memory-bound 训练、大模型、激活内存占比高 |
| 额外步骤 | 无 CSE | CSE 公共子表达式消除 + RNG 种子保存 |

> **✨ 技巧：** **设计权衡：**最小割分区用计算换内存。对于 memory-bound 的训练场景特别有效，因为减少中间结果的保存意味着更少的内存访问和更小的显存占用。点级算子（如 ReLU、GELU）计算成本极低但中间结果可能很大，非常适合重计算。

## 🤝 8.7 force_save_collectives 与分布式

在分布式训练中，前向图常包含 collective 通信算子（如 `all_reduce`、`all_gather`）。这些算子的输出 **原则上可以重计算**，但重算会触发一次额外的通信，在多卡场景下代价巨大且可能改变通信调度。

> **⚠️ 注意：** **force_save_collectives：**当 `config.unsafe_allow_optimization_of_collectives` 为 `False`（默认）时，分区器会调用 `force_save_collectives(joint_module)`，将所有 collective 算子的输出强制标记为"必须保存"，使其不会出现在重计算路径中。这样可以避免反向时重复通信带来的性能与正确性风险。仅当用户显式确认安全时才可关闭该保护。

这一保护机制体现了 Inductor 在"激进优化"与"数值/性能安全"之间的谨慎取舍：通信算子的重计算不仅消耗带宽，还可能与 NCCL 通信节奏冲突，因此默认保守处理。

## 🧩 8.8 分区后的产物

分区器返回 `(fwd_module, bwd_module)` 后，AOTAutograd 将二者分别交给 `fw_compiler` 与 `bw_compiler`（通常都是 Inductor 的 `inner_compile`）进行编译：

```text
joint_module (前向+反向合一)
        │
        ▼  min_cut_rematerialization_partition
┌───────────────┬───────────────┐
│  fwd_module   │  bwd_module   │
│ (含 saved 输出)│ (含重计算路径)│
└───────┬───────┴───────┬───────┘
        │               │
        ▼               ▼
   fw_compiler     bw_compiler
   (Inductor)      (Inductor)
        │               │
        ▼               ▼
   前向可执行函数   反向可执行函数
```

这种"先分区再分别编译"的设计，让前向与反向可以独立应用 Inductor 的融合、调度与代码生成优化，同时通过 saved tensors 列表在运行时传递中间结果。

## Related

- [07 AOTAutograd 中间层](./07-aotautograd.md) — 分区器的输入 `joint_module` 由 AOTAutograd 联合追踪产出
- [09 TorchInductor 后端](./09-torchinductor-backend.md) — 分区产物交给 fw/bw_compiler（Inductor）编译
- [PyTorch 索引](../index.md)
