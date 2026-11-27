---
title: 专家并行与 MoE：token 走向专家的路由
type: concept
status: growing
tags: [expert-parallel, ep, moe, all-to-all, routing, capacity-factor, deepseek, mixtral, distributed-training]
created: 2026-11-09
updated: 2026-11-18
source: Switch Transformer / GShard / Mixtral / DeepSeek-V3 论文 + 个人整理
---

# 专家并行与 MoE：token 走向专家的路由

## 一句话理解

> MoE 把 FFN 替换为多个专家，每个 token 只激活其中少数专家。**固定专家宽度与 Top-k 时**，增加专家总数主要增加参数容量，不按比例增加每个 token 的专家计算量；路由与通信仍有额外开销。EP 是这些专家在设备间的放置方式，常用 dispatch/combine 两次 all-to-all 把 token 发给专家再送回。MoE 是模型结构，EP 是并行策略，二者不是同义词。

## 为什么重要

- **解耦总参数与激活参数**：例如 DeepSeek-V3 报告 671B 总参数、每 token 37B 激活参数。理解这种区别，才能分别估算参数驻留显存与实际 FLOPs，不能把所有 MoE 的激活比例套成同一个值。
- **让 all-to-all 的语义落到具体业务**：不同 token 去不同专家，每个 rank 发给各对端的内容和大小可能不同。路由、负载均衡与网络拓扑共同决定速度；TP 的重分片等其他策略也可能使用 A2A，它并非 EP 独占。

## 先把 MoE 想清楚：一个 FFN 变成多条可选择的路

### 普通 Transformer 里的 FFN

可以先把 Transformer 的一层粗略看成：

```text
token 向量 → Attention → FFN → 下一层输入
```

Attention 让 token 汇集上下文信息，FFN 再对各 token 的向量分别做非线性变换。这里省略了残差连接和归一化等结构。

这里的 FFN（feed-forward network，前馈网络）接收一个 token 的向量，经过自己的可学习参数和非线性变换后，输出一个新向量。普通的 dense Transformer 在同一层中只有一套 FFN 参数，因此所有 token 都经过同一个 FFN：

```text
token A ─┐
token B ─┼──> 同一个 FFN ──> 各自的输出向量
token C ─┘
```

这里的“同一个”指共享同一套参数，不是说所有 token 的向量会混成一个 token。

### MoE 的核心改动

MoE（Mixture of Experts，混合专家）把某些层的一个 FFN 换成多个独立的 FFN，分别称为专家（expert），再增加一个路由器（router/gate）：

```text
专家池：E0、E1、E2、E3（各有独立 FFN 参数）

token 向量 → 路由器 → 只选少数专家 → 选中专家分别计算 → 合并输出
```

路由器根据当前 token 的向量给专家打分；Top-$k$ 路由只选择分数最高的 $k$ 个专家。**模型准备了多套参数，但每个 token 只激活其中少数几套。** 因而 MoE 是模型结构与条件计算机制，不是多个完整模型分别生成文本后投票。

例如 4 个专家、Top-2 时：

```text
token A ──> 路由器选 E0、E2 ──> E0 和 E2 分别计算 ──> 加权合并
token B ──> 路由器选 E1、E3 ──> E1 和 E3 分别计算 ──> 加权合并
```

专家通常只是某个 MoE 层里的 FFN，并不是四个完整的聊天模型；同一个 token 到下一层 MoE 时还会重新路由。训练可能让专家形成一定分工，但不能先验断言某个专家只负责数学、中文或代码。

### 一个 token 的数值路径

令 $x$ 为一个 token 进入 MoE 层时的向量，$N$ 为专家总数，$W_r$ 为路由器参数。路由器先产生专家分数：

$$
q(x)=\operatorname{softmax}(W_r x)
$$

再取分数最高的 $k$ 个专家，得到集合 $\mathcal{I}(x)=\operatorname{TopK}(q(x))$。选中的专家分别产生向量 $h_i=\operatorname{FFN}_i(x)$，最后按组合权重 $a_i(x)$ 合并：

$$
 y=\sum_{i\in\mathcal{I}(x)} a_i(x)h_i
$$

其中 $x$ 是输入 token 向量，$q(x)$ 是路由概率或分数，$\mathcal{I}(x)$ 是入选专家集合，$h_i$ 是专家 $i$ 的输出，$a_i(x)$ 是组合权重，$y$ 是 MoE 层输出。不同实现可能对入选概率重新归一化，或使用其他 gate/缩放方案；上式表达的是共同结构，不是所有模型的唯一细节。

例如路由器给 E1、E2 的分数分别为 $0.50$、$0.30$，采用入选后归一化，则：

$$
 y=\frac{0.50}{0.50+0.30}h_1+\frac{0.30}{0.50+0.30}h_2=0.625h_1+0.375h_2
$$

这里是在**向量空间中加权求和**，不是把两个专家生成的文字拼接起来。

### 为什么需要 MoE

固定每个专家的宽度与 Top-$k$ 时，增加专家总数主要增加模型的参数容量，而不按专家总数同比增加每个 token 的专家计算量：

| 设计 | 该层准备的 FFN 参数 | 每个 token 激活的专家数 |
|---|---:|---:|
| 普通 FFN | 1 套 | 1 套 |
| 4 专家、Top-2 | 4 套 | 2 套 |
| 16 专家、Top-2 | 16 套 | 2 套 |

所以 MoE 试图解耦“模型总共拥有多少参数”和“每个 token 实际使用多少参数”。但它不是免费扩容：没被某个 token 选中的专家参数通常仍需驻留在设备或可访问的存储中；路由器、token 重排、通信、负载不均和容量管理都会产生代价。**少算不等于少存，也不等于端到端一定更快。**

### MoE 与 EP 的关系

如果所有专家都在一张卡上，仍然是 MoE，只是没有跨设备的专家并行。若把专家分布到不同设备，例如：

```text
GPU 0：E0、E1                 GPU 1：E2、E3
```

某个来自 GPU 0 的 token 若被选中 E1、E3，就需要：

```text
token（GPU 0）
   ├──> 本卡 E1 计算 ─────────────┐
   └──> 发到 GPU 1 的 E3 计算 ──> 送回
                                  └──> 按 gate 加权合并
```

因此：**MoE 决定 token 使用哪些专家；EP（Expert Parallelism）决定这些专家放在哪些设备上。** dispatch 是把专家任务送到目标专家所在 rank，combine 是把结果送回来源 rank 并按 gate 权重恢复 token 顺序。MoE 是模型结构，EP 是并行策略，二者不是同义词。

## 核心机制：路由、dispatch、计算与 combine

前面描述的是一个 token 的逻辑计算路径；当专家分布到多个 rank 时，需要把它落实为以下数据流。共享专家若存在，还要额外加入其输出。

1. **路由（本地）**：对 token 计算专家编号与 gate 权重；Top-k 大于 1 时，同一 token 可能生成多份专家任务。
2. **dispatch**：按目标 rank 打包 token 和必要的还原索引，常用 A2A 发给专家所在设备。
3. **计算（本地）**：按专家分组执行本地 FFN，可能使用 grouped GEMM。
4. **combine**：把结果送回来源 rank，按索引恢复 token 顺序，依据 gate 权重求和。

下面的演示跟踪 4 个 token 的完整流程：先观察路由如何产生倾斜的任务分布，再看 dispatch 中跨 rank 的 all-to-all，最后对比有无容量限制时专家计算与 combine 的差异（每专家容量 2 格，E0 有 3 份任务，必然有一份溢出）。

<MoeDispatchDemo />

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
