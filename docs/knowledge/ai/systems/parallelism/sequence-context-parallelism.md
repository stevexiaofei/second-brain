---
title: 序列并行 SP 与上下文并行 CP
type: concept
status: growing
tags: [sequence-parallel, context-parallel, attention, megatron, ring-attention, distributed-training]
created: 2026-11-13
updated: 2026-11-13
source: Megatron-LM / DeepSpeed Ulysses / Ring Attention 论文与 PyTorch experimental API；个人整理
---

# 序列并行 SP 与上下文并行 CP

## 一句话理解

> **SP（本文特指 Megatron-SP）**把 TP 中若干 token-local 操作之间的完整激活改为沿序列分片，主要减少这些区间的重复激活；**CP**把长序列的 attention 上下文分给多个 rank，但每个 query 仍须得到它所需的全局 K/V 信息。两者都涉及序列维，却解决不同瓶颈，通信对象和边界也不同。

“Sequence Parallel”在不同论文和框架中可能泛指多种沿序列切分的方法。为避免术语混用，本文的 **SP** 专指 Megatron-LM 常见的、与 TP 配合的 sequence parallelism；**CP**（Context Parallelism）专指将 attention 的上下文计算跨 rank 分摊的策略。

## 为什么需要它们

长序列 Transformer 的中间激活会随序列长度增长。Attention 的计算与注意力矩阵相关，若直接在一张卡上处理整段长上下文，显存和计算都会成为限制。并行化时要问清楚：

1. **哪些张量被切？** 是 token-local 区间中的激活，还是 attention 的 Q/K/V 上下文？
2. **每个 rank 为了算本地结果需要什么数据？** 本地 token-wise 运算只需要本地 token；attention 的 query 通常需要更广的 K/V。
3. **切分后在哪里恢复或交换数据？** SP 通常在 TP 边界使用 RS/AG；CP 则按具体算法交换、收集或重排 attention 数据。

下面都用两张卡处理一个长度为 8 的样本说明。省略 batch 维时，模型激活写作 $[S,B,H]$，其中 $S=8$ 是序列长度、$B$ 是 batch、$H$ 是 hidden size。Attention 投影后可写作 $[S,B,n_h,d_h]$，且 $H=n_h d_h$。

## SP：切开 TP 中间的 token-local 激活

### 从经典 TP 的布局看问题

Megatron 式 TP 的 MLP/Attention 常由列并行和行并行线性层组成。行并行出口处，各 rank 对同一 token 算出部分贡献；需要先求和。经典布局用 all-reduce（AR），因此每张卡都持有完整序列。紧接着的 LayerNorm、Dropout、残差等操作往往逐 token 进行，复制完整序列意味着这些区间的激活在 TP rank 间重复驻留。

### Megatron-SP 的布局变化

SP 在行并行出口把 AR 换成 reduce-scatter（RS）：跨 rank 求和后，结果按序列维分给各 rank。token-local 区间只处理本地 token 分片；下一列并行层需要完整输入时，再 all-gather（AG）恢复复制布局。

两卡、8 token 的示意：

| 阶段 | rank 0 持有 | rank 1 持有 |
|---|---|---|
| 行并行输出，归约前 | 8 个 token 的部分贡献 | 8 个 token 的部分贡献 |
| 经典 TP：AR 后 | token 0–7 的归约结果 | token 0–7 的归约结果 |
| SP：RS 后 | token 0–3 的归约结果 | token 4–7 的归约结果 |
| SP：token-local 区间 | 只处理 token 0–3 | 只处理 token 4–7 |
| SP：AG 后 | token 0–7 完整结果 | token 0–7 完整结果 |

可将核心布局关系概括为：

```text
经典 TP：行并行 → AR → Replicate(完整序列) → token-local 区间 → 下一列并行
Megatron-SP：行并行 → RS → Shard(序列) → token-local 区间 → AG → 下一列并行
```

残差相加要求两支布局相容：主分支和 shortcut 必须在同一 rank 上对应相同 token 片段。不能把完整序列与半段序列直接相加。SP 也不是把 attention 当成完全独立的逐 token 运算；attention 的 query 依赖上下文，需由模型布局和实现处理相应的数据需求。

### 通信量与显存收益要分开看

设完整激活大小为 $S_a$（字节），TP 组大小为 $N$。在常见 ring 算法、按每 rank 发送量计的简化口径下：

$$
V_{\text{AR}}=2S_a\frac{N-1}{N},\qquad
V_{\text{RS}}=S_a\frac{N-1}{N},\qquad
V_{\text{AG}}=S_a\frac{N-1}{N}.
$$

所以 RS+AG 的发送字节量之和与一次 AR 相同。这个比较**不代表总训练通信时间相同**：启动延迟、拓扑、实现、计算重叠和调用次数都会影响耗时。SP 的主要收益是 token-local 区间的激活从每 rank 一份完整序列变为约 $1/N$ 序列分片；AG 后完整张量会重新出现，因此不能推导出“整模型激活显存必然减少 $N$ 倍”。实际收益取决于哪些激活被保存、这些区间占总激活的比例以及具体实现。

更多 Megatron-SP 的两卡逐步例子和 PyTorch `SequenceParallel` 输入布局边界，见 [Megatron 式张量并行中的 SP 详解](./tensor-parallel-megatron.md#变体与工程细节)。

## CP：分摊 attention 的全局上下文

### 关键约束：本地 Q 需要可见的 K/V

CP 把序列块分给不同 rank。每个 rank 可以用自己的 query（Q）计算本地 query 位置的输出，但 attention 结果仍取决于这些 query 可见的 key/value（K/V）。因此，“每卡只算本地 token”本身还不够；实现必须让本地 query 获得所有需要的 K/V，并正确处理全局位置与 mask。

两个 rank 的初始分布可以想成：

| rank | 本地 query 区间 | 初始持有的 K/V 区间 |
|---|---|---|
| 0 | token 0–3 | token 0–3 |
| 1 | token 4–7 | token 4–7 |

以非因果 attention 为例，rank 0 的 Q 也需要 token 4–7 的 K/V；rank 1 的 Q 也需要 token 0–3 的 K/V。一个概念性的 ring 流程是每个 rank 保留本地 Q，分块计算当前 K/V 对本地 Q 的贡献，再把 K/V 块传给环上的下一个 rank，直到处理完所需块。真实实现会在数据传输与计算之间做调度/重叠；这里的描述不规定具体框架的发送顺序、缓冲区或同步细节。

### 分块 attention 为什么需要在线 softmax

不能分别对每个 K/V 块算一次 softmax，再把各块的 attention 输出直接相加：每块 softmax 的分母不同，直接相加会改变归一化结果。可以用数值稳定的在线 softmax 将块结果合并。

先按单个 batch、单个 attention head 看 shape：本 rank 的 query 块 $Q\in\mathbb{R}^{L_q\times d_h}$，当前收到的 key 块 $K_j\in\mathbb{R}^{L_k\times d_h}$、value 块 $V_j\in\mathbb{R}^{L_k\times d_v}$。因此

$$
X_j=\frac{QK_j^T}{\sqrt{d_h}}\in\mathbb{R}^{L_q\times L_k}.
$$

在前面的两卡、8-token 例子中，每个 rank 的 query 块和每个 KV 块各含 4 个 token，所以 $L_q=L_k=4$，$Q,K_j,V_j$ 的 shape 分别是 $[4,d_h]$、$[4,d_h]$、$[4,d_v]$，logits $X_j$ 是 $[4,4]$。以下下标含义固定为：$j$ 是当前处理的 **K/V 块编号**，$r$ 是本 rank 的 query 块内行号，$t$ 是 K/V 块内的 key 位置；$j$ 不是 head 编号。对 $X_j$ 的每一行（一个 query）沿 key 位置维 $t$ 求统计量，得到 $m_j,\ell_j\in\mathbb{R}^{L_q}$、$o_j\in\mathbb{R}^{L_q\times d_v}$。若本地 query 块的全局起点是 $q_0$，第 $j$ 个 K/V 块的全局起点是 $k_{j,0}$，则该行 query 的全局位置为 $q_0+r$，块内 key $t$ 的全局位置为 $k_{j,0}+t$。例如 rank 0 的 query 块覆盖 token 0–3，处理覆盖 token 4–7 的块 $j=1$ 时，$r=2$ 对应 query token 2，$t=3$ 对应 key token 7。统计量为：

$$
m_{j,r}=\max_t(X_{j,r,t}),\qquad
\ell_{j,r}=\sum_t e^{X_{j,r,t}-m_{j,r}},\qquad
o_{j,r,:}=\sum_t e^{X_{j,r,t}-m_{j,r}}V_{j,t,:}.
$$

已有累计状态 $m,\ell\in\mathbb{R}^{L_q}$、$o\in\mathbb{R}^{L_q\times d_v}$ 与新块合并时，对每个 query 行 $r$ 令 $m'_r=\max(m_r,m_{j,r})$，则：

$$
\ell'_r=e^{m_r-m'_r}\ell_r+e^{m_{j,r}-m'_r}\ell_{j,r},\qquad
o'_{r,:}=e^{m_r-m'_r}o_{r,:}+e^{m_{j,r}-m'_r}o_{j,r,:}.
$$

多 batch、多 attention head 时，这些统计量对每个 $(b,h,q)$ 独立维护；这里只把 batch/head 省略了。

所有可见块处理完后，输出为 $o/\ell$。因果 mask 可能让某个块对当前 query 完全不可见；这类空块应跳过或由实现用等价的安全逻辑处理，不能把 $m_j=-\infty$ 直接代入普通合并式而产生未定义的 $-\infty-(-\infty)$。这是说明“跨 K/V 块仍要保持同一个 softmax 归一化”的概念公式，不是某个框架 kernel 的逐行实现；实际 kernel 还会按 query/head/块形状组织计算。

### 因果 attention 的全局 mask 与负载

因果 mask 必须根据 query 和 key 的**全局 token 位置**判断。记 query 的全局位置为 $p_q$、key 的全局位置为 $p_k$，则该 key 可见当且仅当 $p_k\le p_q$。这里特意不用 $i,j$ 表示全局位置，避免与前文的 K/V 块编号 $j$ 混淆。仅按各 rank 内部的局部索引判断会在跨块边界处错误地屏蔽过去 token，或错误地放行未来 token。

因果约束也会带来工作量不均：前面的 query 块可见的 K/V 较少，后面的 query 块可见的 K/V 较多。不同系统可能采用不同的数据排布或调度策略改善负载均衡；不能把某一种排布方式当成所有 CP 实现共有的规则。

## CP 的通信实现并非只有一种

“CP”描述的是上下文计算如何跨设备分担，不是唯一通信原语。常见设计包括：

| 思路 | 数据流概念 | 取舍提示 |
|---|---|---|
| Ring / P2P | 本地 Q 不动，K/V 分块在 rank 间环式传递；边收边算 | 避免一次性在每卡复制全部 K/V，但依赖多轮传输与调度；具体 overlap、因果跳过和反向实现各异 |
| Gather 类 | 收集其他 rank 的 K/V 后计算本地 query | 数据流直观，但临时 K/V 驻留量、通信和内存峰值取决于 gather 范围与实现 |
| All-to-all 重排（如 Ulysses 思路） | 在序列分片和 head 分片之间做维度重排，让 attention 在新的 head 布局下计算，再重排回来 | 通信模式与 ring 不同；head 数、并行度、序列长度和拓扑会影响适配性 |

不要仅凭“用了 CP”就给出固定通信量、固定显存峰值或固定反向通信公式。不同方法可能传输不同张量、使用不同分块和重计算策略；因果与非因果 attention 的有效计算量也不同。训练反向还需正确累积跨块的梯度，具体传输路径依算法与框架而变。

## SP 与 CP 对照

| 维度 | Megatron-SP（本文用法） | CP |
|---|---|---|
| 主要目标 | 减少 TP rank 间 token-local 激活的重复驻留 | 将长序列 attention 的上下文计算分到多 rank |
| 主要切分/布局 | TP 边界间的激活序列布局：复制 ↔ 分片 | 序列上下文（Q/K/V 的序列块及其跨 rank 可见性） |
| 关键算子/约束 | token-local 运算在本地分片上进行；残差布局须匹配 | 每个 query 必须访问其可见的全局 K/V，并使用正确的全局 mask |
| 典型通信 | 与 TP 配合的 RS/AG 边界 | 依实现可为 ring P2P、gather、A2A 重排等 |
| 常见并行组关系 | 通常复用 TP 组 | 通常是单独的 CP 组/mesh 维度；具体取决于框架配置 |
| 能否互相替代 | 不能由 SP 自动得到 attention 上下文分片 | CP 本身不等同于 Megatron-SP 的 token-local 激活布局优化 |

实际系统可以组合 TP、SP 与 CP：例如不同 mesh 维度分别负责权重矩阵切分和上下文分组；但哪些 rank 属于哪个组、某个张量在哪一步是 replicated 还是 sharded，必须按框架的 mesh/layout 配置确认。“5D 并行”是理解并行轴的视角，不意味着所有框架都有统一固定的五个物理 mesh 轴。

## PyTorch API 边界

PyTorch 文档/源码中的 `torch.distributed.tensor.experimental.context_parallel` 明确属于 experimental/prototype 接口。其实现会在上下文中调整 SDPA 行为，并按传入配置处理指定 buffer 的序列切分；API、限制与行为可能随 PyTorch 版本变化。它适合作为理解特定 PyTorch 实验接口的入口，不应被描述为稳定、通用的 CP 标准，也不能仅凭该 API 名称推断它与 Megatron 或 DeepSpeed 的通信时序完全一致。

## 常见误区

1. **“SP 和 CP 都是切序列，所以是一回事。”** 不对。本文 SP 关注 TP 中间 token-local 激活的复制/分片；CP 关注 attention query 如何获取全局上下文。
2. **“SP 就把整个模型激活显存除以 TP 度数。”** 不对。仅部分区间分片，AG 后完整激活又会出现。
3. **“CP 每个 rank 只需要自己的 K/V。”** 通常不对。除非 query 可见范围恰好只落在本地，否则必须让所需远端上下文参与 attention。
4. **“对每个 K/V 块独立 softmax 后把输出相加即可。”** 不对。分块结果必须按一致的归一化统计量合并。
5. **“所有 CP 都是 ring attention。”** 不对。Ring 是一种常见方案；也存在 gather、A2A 重排等不同数据流。
6. **“RS+AG 与 AR 字节相同，SP 一定不影响性能。”** 不对。发送字节总量的简化估算不等于延迟、计算重叠或端到端吞吐相同。

## 实践检查清单

- 明确该实现里的 SP/CP 术语定义和具体版本；不要只看配置项名称。
- 记录每个并行维度的 process group/mesh，以及 TP、DP、CP 组之间的关系。
- 按阶段标注张量布局：复制、序列分片、head 分片，及其转换发生在哪里。
- 对 CP 检查全局 token 位置、因果 mask、跨块 softmax 合并和最后一个不完整块等边界。
- 对 SP 检查残差两支布局、norm/dropout 输入是否已分片、下一列并行前是否恢复所需布局。
- 通过 profile/trace 核验通信原语、消息尺寸、rank 拓扑和通信计算重叠；不要从概念图推断性能。
- 比较显存时注明统计范围（激活、KV、workspace、临时 gather buffer）和是否启用重计算。

## 参考资料

- Shoeybi et al., [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) — Megatron 式张量并行的基础。
- Liu et al., [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889) — 分块 attention 与环式上下文通信思路。
- Jacobs et al., [DeepSpeed Ulysses: System Optimizations for Enabling Training of Extremely Long Sequence Transformer Models](https://arxiv.org/abs/2309.14509) — 序列与 attention head 维度间 all-to-all 重排思路。
- [PyTorch `context_parallel` experimental API source](https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/experimental/_attention.py) — 版本可能变化的实验性实现。
- [集合通信原语：all-gather、reduce-scatter 与它们的亲戚](../pytorch/pytorch-collective-communication.md) — AR、RS、AG、A2A 的通信语义。
- [并行策略总览：5D 并行的组合逻辑](./parallelism-overview.md) — 并行维度组合与 mesh 关系。
