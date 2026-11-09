---
title: 专家并行与 MoE：token 走向专家的路由
type: concept
status: growing
tags: [expert-parallel, ep, moe, all-to-all, routing, capacity-factor, deepseek, mixtral, distributed-training]
created: 2026-11-09
updated: 2026-11-09
source: Switch Transformer / GShard / Mixtral / DeepSeek-V3 论文 + 个人整理
---

# 专家并行与 MoE：token 走向专家的路由

## 一句话理解

> MoE 把 FFN 替换为多个专家，每个 token 只激活其中少数专家。**固定专家宽度与 Top-k 时**，增加专家总数主要增加参数容量，不按比例增加每个 token 的专家计算量；路由与通信仍有额外开销。EP 是这些专家在设备间的放置方式，常用 dispatch/combine 两次 all-to-all 把 token 发给专家再送回。MoE 是模型结构，EP 是并行策略，二者不是同义词。

## 为什么重要

- **解耦总参数与激活参数**：例如 DeepSeek-V3 报告 671B 总参数、每 token 37B 激活参数。理解这种区别，才能分别估算参数驻留显存与实际 FLOPs，不能把所有 MoE 的激活比例套成同一个值。
- **让 all-to-all 的语义落到具体业务**：不同 token 去不同专家，每个 rank 发给各对端的内容和大小可能不同。路由、负载均衡与网络拓扑共同决定速度；TP 的重分片等其他策略也可能使用 A2A，它并非 EP 独占。

## 核心机制：路由、dispatch、计算与 combine

一个常见的 Top-k MoE 层可以写为：

$$
p(x)=\operatorname{softmax}(W_r x),\qquad
\mathcal{I}(x)=\operatorname{TopK}(p(x)),\qquad
y=\sum_{i\in\mathcal{I}(x)} a_i(x)\operatorname{FFN}_i(x)
$$

$W_r$ 是路由器，专家参数相互独立。$a_i(x)$ 是被选中专家的组合权重：有的实现对入选概率重新归一化，有的采用其他 gate/缩放方案，不能把这一选择当成 MoE 的统一定义。共享专家若存在，还要额外加入其输出。

1. **路由（本地）**：对 token 计算专家编号与 gate 权重；Top-k 大于 1 时，同一 token 可能生成多份专家任务。
2. **dispatch**：按目标 rank 打包 token 和必要的还原索引，常用 A2A 发给专家所在设备。
3. **计算（本地）**：按专家分组执行本地 FFN，可能使用 grouped GEMM。
4. **combine**：把结果送回来源 rank，按索引恢复 token 顺序，依据 gate 权重求和。

为什么常用 A2A 而非 AG？AG 会把所有来源的数据都收齐，A2A 则只把每段数据送给指定目的地。EP 可以使用其他 dispatcher（例如 AG 或专用 P2P/融合通信），但其冗余量、内存和拓扑适用范围不同，不能说语义上“必须 A2A”。训练反向还需对 dispatch/combine 及 gate 计算传播梯度；“两次 A2A”这里只计前向。

## 关键工程问题

### 负载均衡：路由的自由与热点

路由器可能把大量 token 发给少数专家，其他专家空闲，整层等待最忙的设备。专家负载、rank 负载和链路负载相关但并不相同。

以 Switch 的 **Top-1** 辅助均衡损失为例，设一批有 $T$ 个 token、$N$ 个专家：

$$
f_i=\frac{1}{T}\sum_{t=1}^{T}\mathbf{1}[\mathcal{I}(x_t)=\{i\}],\qquad
P_i=\frac{1}{T}\sum_{t=1}^{T}p_i(x_t),\qquad
L_{\text{aux}}=\alpha N\sum_{i=1}^{N}f_iP_i
$$

$f_i$ 是硬路由占比，$P_i$ 是平均路由概率；硬选择通常不直接求导，损失通过概率项影响路由。这是鼓励均衡的代理目标，不应声称它无条件保证均匀分配或唯一最小值。推广到 Top-k 必须明确计数与归一化方式，不能直接把 Top-1 公式中的占比混用。

**DeepSeek-V3 的 auxiliary-loss-free balancing**：专家偏置参与 Top-k 选择，按负载调整；组合权重不简单使用偏置后的分数。该机制减少主要均衡目标与主任务梯度的冲突，但报告仍使用很小的序列级辅助均衡损失；“auxiliary-loss-free”不能解释成训练完全不存在任何辅助损失。

### Capacity Factor、padding 与 dropless

**固定容量是一类实现选择，不是 all-to-all 的必要条件。** 常见容量预算约为：

$$
C\approx\left\lceil\mathrm{CF}\cdot\frac{Tk}{N}\right\rceil
$$

这里 $T$ 必须是该容量预算覆盖的路由域 token 总数，$k$ 为每个 token 的专家任务数。具体实现可能还有最小容量、对齐与溢出规则。

- **固定容量/填充实现**：预分配每专家容量，溢出时可能丢弃部分专家任务、改路由或采用其他溢出策略。CF 太小可能增大 drop rate，太大增加 padding 和显存。
- **Dropless/可变长度实现**：先根据路由交换计数，再按真实数量分配与发送；`all_to_all_single` 支持不同 dim-0 split，无需让所有专家收到相同 token 数。也可配合分组/块稀疏计算。
- **Dropless 不等于无限资源**：严重倾斜仍可能造成热点、超出显存预算或需要排队；推断策略也没有统一的“CF=1 且不 drop”规则。

**共享专家是模型结构，不是通用溢出缓冲。** DeepSeek 系列使用共享专家；原始 **Mixtral 8x7B 是 8 个 routed experts，每 token 选择 2 个，没有“2 个 shared experts”**。即使有共享专家，也不能自动推出被丢弃的 routed 分支任务会转交给它。

可变长度通信的计数交换与 split 约束，见[集合通信原语](../pytorch/pytorch-collective-communication.md)。

### 放置与物理拓扑

- MoE 层的 dispatch/combine 发生频繁，高带宽域通常有优势；但 EP 可以跨节点扩展，是否值得取决于专家容量、消息大小、链路和通信/计算重叠。
- A2A 可能均匀，也可能因路由而高度倾斜；不能断言它“没有规整结构可利用”。层次化路由、拓扑感知分组与融合 dispatcher 都是在利用结构。
- attention、共享专家与 routed experts 可采用不同的 TP/DP/EP 方案。`parallelize_module` 的普通列/行计划本身不会自动完成 router、token permutation 与 dispatcher。

### 训练/推断的不对称

| 维度 | 训练 | 推断 |
|---|---|---|
| 路由器 | 参数持续学习，常伴随均衡机制 | 参数通常固定，但逐 token 路由仍依赖输入 |
| 专家放置 | 常按专家分组，需同步副本梯度 | 可按热度复制或动态调整放置 |
| 主要矛盾 | 负载、激活显存、通信与梯度同步 | prefill/decode 的批量差异、通信延迟与热点 |

## 与其他并行轴的组合

- **EP × TP**：一个专家 FFN 也可由 TP 组承载；是否切单专家取决于专家大小与互联，不能仅凭模型采用 MoE 推断不需要 TP。
- **EP × DP**：非专家参数和专家参数的复制组可能不同。同一 rank 可以处理非专家副本上的输入 token，同时承载部分专家；专家副本之间还需正确同步梯度。
- **EP × PP**：不同 stage 的 MoE 层分别组织专家组。简单均分、无专家复制的 EP 组内有 $N=E\,n_{\text{local}}$，其中 $E$ 为 EP 度数。加入复制、TP 或非均匀放置后需重新定义映射。
- **不要盲目相乘所有并行度**：EP 可能复用非专家部分的数据并行设备集合；SP 也常复用 TP 组。“DP×TP×PP×EP×SP”不是无条件的 world size 公式。必须按模块核对通信组与参数副本。

## PyTorch 侧现状与读码路线

本笔记对应的本地 PyTorch v2.9.0a0 源码中，没有与 TP 的 `parallelize_module` 等价、可直接替代完整 MoE 层的统一 EP 入口。

- **原语**：`all_to_all_single` 可用于手写 dispatcher；变长 split、计数交换、反向与排列还原都要正确实现。DTensor 的 A2A 重分片只解决布局变换的一部分，不自动变成 token router。
- **训练生态**：Megatron-Core、DeepSpeed 等有 MoE/EP 实现，但接口、专家分组和 dispatcher 不完全相同。
- **组合框架**：[torchtitan](https://github.com/pytorch/torchtitan) 的模型与并行支持会演进，应按具体版本检查 MoE 模型、并行转换与 dispatcher；不再把 `models/flux` 当作 MoE 的源码入口。
- **推断生态**：vLLM / SGLang 等提供 EP 与专家执行优化，需区分其推断调度与训练梯度语义。

建议先跟一次 token 的“源 rank → 专家 rank → 源 rank”数据与索引流，再看对应通信组及反向，而不是只记住一个名为 `ExpertParallel` 的类。

## 我的理解

- **MoE 是条件计算，不是免费扩容**：固定激活专家规模时能提高总参数容量，但参数驻留、router、通信和负载不均都需支付成本。
- **路由问题可以类比任务调度**：token 是任务，专家是执行者，router 决定去向；容量限制和热点与队列问题相似。但这是理解类比，不代表两者的优化目标完全相同。
- **细粒度专家与共享专家改变了容量/计算形状**：DeepSeek-V3 使用 256 个 routed experts、每 token 激活其中 8 个，另有 1 个共享专家。这是具体模型设计，不能泛化成 Mixtral 或所有 MoE 的配置。
- **历史要分清**：2017 年的 sparsely-gated MoE 推动了现代大规模稀疏专家研究，GShard 是 2020 年工作，不是“2017 GShard”。

## 常见误区

1. **总参数增加必然同比增加 token FLOPs** → 固定专家宽度和 Top-k 时通常不同比增加，但改变专家宽度或 Top-k 就会改变计算量。
2. **EP 就是切单个专家矩阵** → EP 主要分配专家；专家内部的矩阵分片仍属于 TP。
3. **all-to-all 等同于 all-gather，或所有 EP 必须 A2A** → 它们的目标数据分布不同；选择其他 dispatcher 需要支付不同的冗余/内存成本。
4. **所有 MoE 都有容量丢弃** → 固定容量是选择，dropless 也存在；排查时先确认实现，再看 drop rate、负载与显存。
5. **均衡只是训练损失技巧** → 最忙的专家、设备与链路都可能决定耗时，推断也有负载均衡问题。
6. **共享专家自动兜底所有溢出** → 模型必经共享分支不等于溢出转发机制；Mixtral 8x7B 本身也没有共享专家。

## Related

- [并行策略总览](./parallelism-overview.md) — 并行维度与按模块混合指定
- [张量并行：Megatron 式](./tensor-parallel-megatron.md) — 单专家或 dense 模块内部切分
- [流水线并行](./pipeline-parallelism.md) — stage 内组合 EP
- [集合通信原语](../pytorch/pytorch-collective-communication.md) — all-to-all 与可变长度 split
- [PyTorch 分布式训练](../pytorch/pytorch-distributed.md) — DeviceMesh 与通信组
- [千卡集群网络拓扑与 NCCL 排查](../../../engineering/mlops-platform/thousand-gpu-network-topology.md) — 跨节点通信路径

## References

- *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*（Shazeer et al., 2017）
- *GShard: Scaling Giant Models with Conditional Computation*（Lepikhin et al., 2020）
- *Switch Transformers*（Fedus et al., 2021）— Top-1 与辅助均衡损失
- *Mixtral of Experts*（Jiang et al., 2024）— 8 个 routed experts、Top-2，无共享专家
- *DeepSeek-V3 Technical Report*（2024）— 细粒度/共享专家、偏置均衡与序列级辅助损失
- PyTorch 源码：`torch/distributed/distributed_c10d.py`（`all_to_all_single`）；`torch/distributed/tensor/_api.py`（重分片）
- 生态代码：[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)、[DeepSpeed](https://github.com/deepspeedai/DeepSpeed)、[torchtitan](https://github.com/pytorch/torchtitan)
