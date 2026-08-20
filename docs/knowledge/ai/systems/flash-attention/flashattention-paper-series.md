---
title: FlashAttention 三篇论文精读导览
type: paper
status: seed
authors: [Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani]
year: 2022
tags: [FlashAttention, Attention, GPU, Kernel, IO-aware, Hopper, CUDA]
created: 2026-08-18
updated: 2026-08-18
paper_versions:
  - 2205.14135v2
  - 2307.08691v1
  - 2407.08608v2
source:
  - https://arxiv.org/abs/2205.14135
  - https://arxiv.org/abs/2307.08691
  - https://arxiv.org/abs/2407.08608
---

# FlashAttention 三篇论文精读导览

## 一句话理解

这三篇论文不是三个孤立的 attention trick，而是一条连续的系统优化路线：

- **FlashAttention-1**：解决 attention 的 **HBM / IO 瓶颈**
- **FlashAttention-2**：解决 attention 的 **GPU 并行度与 work partitioning 瓶颈**
- **FlashAttention-3**：进一步解决 **Hopper 架构上的异步流水线与低精度利用问题**

它们的共同目标是：

> 在不改变标准 attention 数学结果的前提下，重新安排计算、数据移动和并行任务，使 GPU 尽可能少访问慢速显存、尽可能多使用片上存储和矩阵乘法单元。

---

## 0. 先建立标准 attention 的问题模型

给定：

- $Q \in \mathbb{R}^{N_q \times d}$
- $K \in \mathbb{R}^{N_k \times d}$
- $V \in \mathbb{R}^{N_k \times d}$

标准 attention 是：

$$
O = \operatorname{softmax}(S)V,
\qquad
S = \frac{QK^T}{\sqrt d}
$$

其中：

- $S$ 的形状是 $N_q \times N_k$
- 当 $N_q=N_k=N$ 时，中间矩阵是 $N \times N$
- 每个 score 通常还要经过 mask、softmax、dropout 等处理

### 普通实现的计算和 IO 路径

一个朴素实现大致是：

```text
Q, K 从 HBM 读入
        ↓
计算 S = QKᵀ
        ↓
把 S 写回 HBM
        ↓
从 HBM 读 S，做 softmax
        ↓
把 P 写回或保留在 HBM
        ↓
读取 P 和 V
        ↓
计算 O = PV
```

问题不只是 FLOPs 多，而是中间的 $N \times N$ 矩阵需要：

- 占用大量显存
- 被多次写入和读取
- 在 forward 中产生巨大带宽压力
- 在 backward 中还要保存或重新读取更多中间量

因此，attention 经常不是算力受限，而是 **memory-bound / IO-bound**。

### FlashAttention 的核心转变

FlashAttention 不改变：

$$
\operatorname{softmax}\left(\frac{QK^T}{\sqrt d}\right)V
$$

它改变的是计算顺序：

```text
一次加载一个 Q tile
    ↓
依次加载 K/V tile
    ↓
片上计算 score、mask、softmax
    ↓
在线更新归一化状态和 O
    ↓
只把最终 O 与必要摘要写回 HBM
```

所以它优化的是 **dataflow**，而不是 attention 的数学定义。

---

# 1. FlashAttention-1：IO-aware exact attention

论文：

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)

## 1.1 它要解决什么问题

第一篇论文针对的是两个事实：

1. 标准 attention 需要物化 $N \times N$ 的 attention matrix
2. GPU 的 HBM 很大但很慢，SRAM / shared memory 很快但很小

如果只追求减少 FLOPs，通常并不能解决真实瓶颈；关键是减少 GPU memory hierarchy 之间的数据搬运。

## 1.2 IO-aware 的含义

GPU 可以粗略看成几级存储层次：

```text
寄存器       最快、最小、线程私有
   ↓
Shared Memory / SRAM
             快、容量小、线程块共享
   ↓
HBM / Global Memory
             容量大、带宽高，但访问代价更高
```

FlashAttention 的策略是：

- 从 HBM 加载一个 tile 到 SRAM
- 尽量在 SRAM / registers 中完成该 tile 相关的矩阵乘、softmax、PV
- 处理下一个 tile 时复用已经存在的 Q 或状态
- 不把完整的 $S$ 或 $P$ 写回 HBM

### 关键不是“片上存储更大”

SRAM 的容量远小于完整 attention matrix，所以 FlashAttention 并不是试图把整个矩阵放进 SRAM，而是：

> 只让当前需要的局部块进入 SRAM，并用一个可合并的状态表示已经处理过的历史块。

这个“可合并状态”就是 online softmax 的关键。

## 1.3 Tiling：把矩阵计算切成块

设一个 query tile 为 $Q_i$，key/value tile 为 $K_j,V_j$：

$$
S_{ij} = Q_iK_j^T
$$

每个 tile 只产生局部 score：

$$
S_{ij} \in \mathbb{R}^{B_r \times B_c}
$$

其中：

- $B_r$：一个 query block 的行数
- $B_c$：一个 key/value block 的列数

单个 CTA 大致做：

1. 将 $Q_i$ 载入片上存储
2. 依次载入 $K_j,V_j$
3. 计算 $Q_iK_j^T$
4. 应用 causal / local mask
5. 更新该 query block 的 softmax 状态
6. 更新输出累积
7. 循环处理所有 $j$
8. 写回最终输出

## 1.4 Online softmax 的数学原理

普通 softmax 对一行 score $s_1,\dots,s_N$ 的计算是：

$$
\operatorname{softmax}(s_j)=\frac{e^{s_j}}{\sum_k e^{s_k}}
$$

为了数值稳定，通常写成：

$$
\operatorname{softmax}(s_j)=\frac{e^{s_j-m}}{\sum_k e^{s_k-m}},
\qquad
m=\max_k s_k
$$

问题是：如果 score 被分成多个 tile，事先不知道全局最大值 $m$。

### 维护两个在线状态

处理到第 $t$ 个 tile 时，对每个 query row 维护：

- $m^{(t)}$：到目前为止的最大 score
- $\ell^{(t)}$：到目前为止、以当前最大值为基准的指数和

新 tile 的局部最大值为：

$$
\widetilde m^{(t)} = \max_j S^{(t)}_j
$$

合并新旧最大值：

$$
 m^{(t)} = \max\left(m^{(t-1)},\widetilde m^{(t)}\right)
$$

更新归一化和：

$$
\ell^{(t)}
=
 e^{m^{(t-1)}-m^{(t)}}\ell^{(t-1)}
+
\sum_j e^{S^{(t)}_j-m^{(t)}}
$$

这里第一项是旧 tile 的贡献重新换基，第二项是当前 tile 的贡献。

### 输出也必须同步重标定

设旧输出的未归一化累积为 $O^{(t-1)}$，当前 tile 的局部概率权重为：

$$
P^{(t)}=e^{S^{(t)}-m^{(t)}}
$$

则输出累积可以更新为：

$$
O^{(t)}
=
 e^{m^{(t-1)}-m^{(t)}}O^{(t-1)}
+
P^{(t)}V^{(t)}
$$

最后输出：

$$
O=\frac{O^{(T)}}{\ell^{(T)}}
$$

并保存：

$$
\operatorname{LSE}=m^{(T)}+\log \ell^{(T)}
$$

这就是代码中的 `softmax_lse` 的数学来源。

### 为什么这仍然是 exact

每个 tile 的贡献都被转换到了同一个全局归一化尺度；旧状态和新状态的合并是代数上精确的，而不是抽样、剪枝或近似。

因此 FlashAttention 的主要变化是：

- 不保存全部中间矩阵
- 用 $(m,\ell,O)$ 摘要历史 tile
- 仍然得到标准 softmax attention 的结果

## 1.5 IO 复杂度：为什么 tile 化能减少 HBM 访问

论文把 GPU 简化为两级存储模型：容量较大的 HBM 与容量为 $M$ 的片上 SRAM。设序列长度为 $N$、head dimension 为 $d$，并假设 $d \le M \le Nd$。

在这个模型下，论文给出的主项是：

- 标准 attention 的 HBM 访问量：

$$
\Theta(Nd + N^2)
$$

- FlashAttention 的 HBM 访问量：

$$
\Theta\left(\frac{N^2d^2}{M}\right)
$$

当典型的 $d=64\sim128$、SRAM 约为百 KB 时，$d^2/M$ 远小于 1，因此 FlashAttention 对二次项有显著削减。

### 这个复杂度从哪里来

若一个 K/V tile 大约占据 $M$ 个片上元素，则每个 tile 可容纳约 $M/d$ 个 token。K/V 方向大约需要：

$$
\frac{Nd}{M}
$$

个 tile。每处理一个 K/V tile，算法会扫描 Q/O 与在线 softmax 状态，其数据规模为 $\Theta(Nd)$，因此主项约为：

$$
\frac{Nd}{M}\cdot Nd
=
\frac{N^2d^2}{M}
$$

这也揭示了一个容易忽视的事实：FlashAttention 并不是“Q/K/V 各读一次”。在 FA1 的循环顺序里，Q/O 状态仍可能被多遍扫描；它的优势是这些扫描的总量远小于把 $N^2$ 的 S/P 矩阵反复写入 HBM。

### 最优性应该怎样表述

论文证明的是一个带范围的下界：不存在一个 exact-attention 算法，能对允许范围内的**所有** SRAM 大小 $M$ 都渐近优于上述 HBM 访问量。它不是“对每一个固定 $M$ 都单独证明唯一全局最优”的更强断言。

论文的 GPT-2 medium 示例也直观展示了 IO 与 FLOPs 的差异：FlashAttention 的 FLOPs 略高，但 HBM 读写从约 40.3 GB 降到约 4.4 GB，forward+backward 时间从约 41.7 ms 降到约 7.3 ms。这里的额外 FLOPs 主要来自 backward 重计算。

## 1.6 Forward 为什么只保存 LSE

完整 attention matrix $P$ 太大，FlashAttention forward 通常不保存它。
它只需要保存：

- 输出 $O$
- 每行的 `softmax_lse`
- dropout 场景下的 RNG 状态

backward 再按 tile 重新计算局部 $S$ 和 $P$。

这是典型的：

> 用少量状态 + 额外计算，换取巨大的显存节省。

## 1.7 Backward 的基本原理

令：

$$
O=PV,
\qquad P=\operatorname{softmax}(S),
\qquad S=QK^T\cdot \text{scale}
$$

给定上游梯度 $dO$：

### 对 V 的梯度

$$
 dV=P^TdO
$$

### 对 P 的梯度

$$
 dP=dOV^T
$$

### softmax 的梯度

softmax 是按行计算的。对第 $i$ 行：

$$
P_{ij}=\frac{e^{S_{ij}}}{\sum_k e^{S_{ik}}}
$$

它的不同输出并不独立：提高一个 score 不仅会提高对应概率，还会通过分母降低同一行的其他概率。因此不能简单写成 $dS=P\odot dP$，而必须加入整行的归一化修正。

#### 从 Jacobian 推导

softmax 的元素级偏导为：

$$
\frac{\partial P_{ij}}{\partial S_{ik}}
=
P_{ij}(\delta_{jk}-P_{ik})
$$

其中 $\delta_{jk}$ 是 Kronecker delta：当 $j=k$ 时为 $1$，否则为 $0$。根据链式法则：

$$
\begin{aligned}
dS_{ik}
&=\sum_j dP_{ij}\frac{\partial P_{ij}}{\partial S_{ik}}\\
&=\sum_j dP_{ij}P_{ij}(\delta_{jk}-P_{ik})\\
&=P_{ik}\left(dP_{ik}-\sum_jP_{ij}dP_{ij}\right)
\end{aligned}
$$

定义每一行的修正项：

$$
D_i=\sum_jP_{ij}dP_{ij}
$$

就得到：

$$
\boxed{
dS_{ij}=P_{ij}(dP_{ij}-D_i)
}
$$

矩阵形式为：

$$
\boxed{
dS=P\odot\left(dP-
\operatorname{rowsum}(dP\odot P)\right)
}
$$

这里 `rowsum` 产生每行一个标量，减法时会在该行所有列上广播。

#### 修正项的直觉

$D_i$ 是 $dP_i$ 在概率分布 $P_i$ 下的加权平均：

$$
D_i=\mathbb{E}_{j\sim P_i}[dP_{ij}]
$$

因此，score 梯度取决于当前位置相对于整行加权平均的差异：

- $dP_{ij}>D_i$ 时，$dS_{ij}>0$
- $dP_{ij}<D_i$ 时，$dS_{ij}<0$
- $dP_{ij}=D_i$ 时，$dS_{ij}=0$

这体现了 softmax 的相对竞争关系，而不是每个位置彼此独立地变化。

#### 为什么一行 $dS$ 的和为零

$$
\begin{aligned}
\sum_jdS_{ij}
&=\sum_jP_{ij}(dP_{ij}-D_i)\\
&=D_i-D_i\sum_jP_{ij}\\
&=0
\end{aligned}
$$

这对应 softmax 的平移不变性：

$$
\operatorname{softmax}(S_i+c)=\operatorname{softmax}(S_i)
$$

把一行 score 同时增加相同常数不会改变输出，所以梯度在“整行同时移动”的方向上必须为零。

#### 为什么 FlashAttention 可以直接计算 $D_i=dO_i\cdot O_i$

由 $O=PV$：

$$
dP_{ij}=dO_i\cdot V_j
$$

代入 $D_i$：

$$
\begin{aligned}
D_i
&=\sum_jP_{ij}dP_{ij}\\
&=\sum_jP_{ij}(dO_i\cdot V_j)\\
&=dO_i\cdot\left(\sum_jP_{ij}V_j\right)\\
&=dO_i\cdot O_i
\end{aligned}
$$

也就是：

$$
\boxed{
D_i=\operatorname{rowsum}(dO_i\odot O_i)
}
$$

因此 FlashAttention 不需要物化完整的 $P$ 和 $dP$ 来计算行修正项。backward 可以先从已经保存的输出 $O$ 和上游梯度 $dO$ 得到 $D$，再逐 tile 重建：

$$
P_{ij}=\exp(S_{ij}-\operatorname{LSE}_i)
$$

以及：

$$
dP_{ij}=dO_i\cdot V_j
$$

最后在 tile 内计算：

$$
dS_{ij}=P_{ij}(dP_{ij}-D_i)
$$

这正是 FlashAttention backward 能够避免保存或重建完整 $N\times N$ 概率矩阵的关键之一。

### 对 Q、K 的梯度

$$
 dQ=dS K,
\qquad
 dK=dS^T Q
$$

FlashAttention backward 不需要保存完整 $P$：

1. 读取 Q/K/V/O 和 `softmax_lse`
2. 重新计算当前 Q/K tile 的 score
3. 用 LSE 恢复该 tile 对应的 softmax 概率
4. 计算局部 $dV,dK,dQ$
5. 累积到最终梯度

这解释了为什么 backward 会更复杂，也解释了为什么 forward 需要保存 `softmax_lse`。

## 1.8 FlashAttention-1 的实验结论与边界

论文报告的结果需要连同实验环境理解，而不应视为对所有 shape 和硬件都成立的常数：

- 常见序列长度下，attention microbenchmark 相对 PyTorch exact attention 最多约 $3\times$。
- GPT-2 端到端训练相对不同基线报告最高约 $3\times$。
- 显存随序列长度线性增长；相对 exact-attention 基线最高约 $20\times$ 更省显存。
- 在论文测试中，部分近似/稀疏方法在序列长度约 512–1024 后开始与 dense FlashAttention 出现运行时间交叉；这提醒我们 dense exact attention 的计算复杂度仍是 $\Theta(N^2d)$。

它用以下代价换取显存和 IO 优势：

- kernel 实现复杂，需要为新的 attention 变体编写或生成专门 kernel
- 对 tile size、head dim、register/shared-memory 容量很敏感
- backward 需要重计算，因此 FLOPs 可能比物化中间矩阵的实现更多
- 实现不一定可以直接跨 GPU 架构复用
- FA1 的并行度主要来自 batch 和 head；小 batch、少 head、长序列时可能无法占满 SM
- 论文的单 GPU IO 下界并不直接覆盖 multi-GPU 通信成本

这正是 FlashAttention-2 要进一步解决的问题。

---

# 2. FlashAttention-2：更好的并行与 work partitioning

论文：

- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)

## 2.1 FA2 为什么还要重新设计

FA1 已经解决了主要 IO 问题，但 GPU kernel 的性能还受另一个因素影响：

> 不是所有时间都在做矩阵乘；如果 CTA、warp、sequence/head 之间分工不合理，Tensor Core 很快也没用。

FA2 的重点从“减少 HBM 访问”扩展到：

- 让更多 SM 同时有活干
- 让每个 CTA 的工作量更均衡
- 减少非 matmul FLOPs
- 让 forward 和 backward 都能处理小 batch / 短 query / 长 KV

## 2.2 Work partitioning 是什么

一个 attention 问题至少可以沿这些维度切分：

- batch 维度 $B$
- query head 维度 $H$
- query sequence 维度 $N_q$
- key/value sequence 维度 $N_k$
- head dimension $d$

朴素策略可能只沿 batch 和 head 切：

```text
CTA 0 → batch 0, head 0
CTA 1 → batch 0, head 1
CTA 2 → batch 1, head 0
...
```

当 batch 很小、head 数不多、或者 decode 时 $N_q=1$，CTA 数量就不足，很多 SM 会空闲。

FA2 论文进一步沿 sequence 方向切分，使可并行的任务数量增加；当前仓库的推理实现还可沿 KV 方向切分，但那是需要单独区分的 split-KV 扩展。

## 2.3 Forward 并行化：论文里的 sequence parallelism

FA1 主要以一个 batch/head 对应一个 thread block，因此总 thread-block 数大致是：

$$
B\times H
$$

A100 有 108 个 SM；当 $B\times H$ 足够大时，这种调度有效，但长序列通常伴随较小 batch，可能导致 SM 工作不足。

FA2 把 forward 的外层循环改为 Q row blocks，并把不同 Q block 分给不同 CTA：

$$
\text{parallel tasks}
\approx
B\times H\times \left\lceil\frac{N_q}{B_r}\right\rceil
$$

不同 Q row blocks 的输出互不重叠，因此 CTA 之间不需要通信。这是 FA2 论文所谓沿 sequence-length dimension 并行化的核心。

### Backward 的 sequence parallelism

backward 则以 attention matrix 的 column/KV block 为 CTA 工作单元。每个 column block 可以独立累积对应的 $dK,dV$，但多个 column-block CTA 都会对 $dQ$ 贡献梯度，因此论文实现使用 atomic add 来更新 $dQ$。

```text
forward：每个 CTA 负责一组 attention rows → 输出互不冲突
backward：每个 CTA 负责一组 attention columns → dK/dV 局部，dQ 需跨 CTA 汇合
```

### 与源码中的 split-KV 区分

**重要纠正：split-KV 不是 FA2 论文的三个核心贡献之一。**

论文主要讨论：

1. 降低 non-matmul FLOPs
2. CTA 之间沿 sequence length 并行
3. CTA 内从 sliced-K 改为 sliced-Q

当前 flash-attention 仓库的推理/kvcache 路径还实现了 split-KV；它与 FA2 共享“扩大并行任务、再精确合并”的系统思想，但应该作为**后续实现机制**单独理解，而不能直接写成 FA2 论文提出的主要算法。

### 实现扩展：split-KV 的原理

对于短 query、长 KV 的 decode，仅沿 Q row blocks 切分仍然不够。这时实现可把 K/V 序列切成 $S$ 个分片：

$$
K=[K^{(1)},K^{(2)},\dots,K^{(S)}]
$$

每个 CTA 得到局部归一化后的输出 $O^{(s)}$ 和局部 LSE $L^{(s)}$。它们不能直接相加，因为每个 split 的 softmax 分母不同。

正确 combine 是：

$$
L=\log\sum_s e^{L^{(s)}}
$$

$$
O=\sum_s e^{L^{(s)}-L}O^{(s)}
$$

因此 split-KV 仍然是 exact 的。它适合 `seqlen_q` 极短、`seqlen_k` 很长、batch/head 并行任务不足的情况，但会引入 partial-output/LSE buffers、combine kernel 和额外 HBM IO，所以实现通常通过启发式选择 `num_splits`。

## 2.4 减少非矩阵乘开销

GPU 的 Tensor Core 对矩阵乘非常高效，但以下操作可能成为相对瓶颈：

- max / sum reduction
- exp / log
- mask
- rescale
- dropout
- tile 间同步
- 中间状态读写

FA2 的一个重要方向是减少这些 non-matmul FLOPs 和不必要的数据转换，使矩阵乘占比更高。论文以 A100 为例：FP16/BF16 matmul 峰值约 312 TFLOPs/s，而 FP32 non-matmul 约 19.5 TFLOPs/s，因此从峰值吞吐角度看，一个 non-matmul FLOP 可比一个 matmul FLOP“昂贵”约 $16\times$。

论文对 online-softmax 更新做了两个具体简化：

1. 内层循环维护未最终归一化的输出累积，只在所有 KV blocks 完成后除以最终 $\ell$；避免每轮同时对输出更新的两项都做完整缩放。
2. backward 不再同时保存行最大值 $m$ 和指数和 $\ell$，只保存：

$$
L=m+\log \ell
$$

即 `softmax_lse`。局部概率可以直接由 $P_{ij}=\exp(S_{ij}-L_i)$ 恢复。

可以用一个简单的性能视角理解：

```text
总时间
= 矩阵乘时间
+ softmax / reduction 时间
+ 数据搬运时间
+ 同步与调度时间
```

FA1 主要降低数据搬运时间；FA2 进一步压低其余三项相对于矩阵乘的比例。

## 2.5 Warp-level work partitioning：从 sliced-K 到 sliced-Q

FA1 的一个重要实现选择可概括为 **sliced-K**：CTA 内多个 warp 分别处理不同 K/V 分片。这样每个 warp 都会对同一 Q tile 产生部分 score 和部分输出，最后还需要跨 warp 合并。

```text
sliced-K（概念图）
warp 0 → Q tile × K/V slice 0 ─┐
warp 1 → Q tile × K/V slice 1 ─┼→ 跨 warp 合并 score / 输出
warp 2 → Q tile × K/V slice 2 ─┘
```

这种方式能分摊 K/V 方向工作，但代价是：

- 同一 Q tile 被多个 warp 使用
- softmax 和输出累积包含跨 warp 归约
- 需要更多 shared memory 通信和同步

FA2 的关键改动之一是 **sliced-Q**：不同 warp 负责不同的 Q 行；CTA 内的 K/V tile 被共享，但每个 warp 独立完成自己那部分 query rows 的 score、online softmax 与输出。

```text
sliced-Q（概念图）
warp 0 → Q rows 0..r  × 同一个 K/V tile → 自己的 softmax / O
warp 1 → Q rows r..2r × 同一个 K/V tile → 自己的 softmax / O
warp 2 → Q rows 2r..3r × 同一个 K/V tile → 自己的 softmax / O
```

这并不意味着完全没有 CTA 级同步；K/V tile 仍需协同装载。但它避免了把每个 score tile、softmax 状态和输出结果在 warp 间反复交换与归约。

因此 FA2 的收益可以更精确地理解为：

- Q/K/V tile 的加载与复用更有效
- 每个 warp 对自己的 Q rows 形成更完整、更独立的计算路径
- shared memory 的读写、warp 间通信和同步减少
- Tensor Core 矩阵乘更连续
- registers、shared memory 与 occupancy 的平衡更好

这也是“减少 non-matmul FLOPs”在 kernel 层的具体含义之一：不仅减少算术指令，也减少为跨线程协作而产生的 reduction、共享内存访问和同步。

### Block size 的权衡

论文通常在 $\{64,128\}\times\{64,128\}$ 的候选 tile 中按 head dimension 和设备手工调优。tile 较大通常减少 shared-memory load/store，但也增加 register 和 shared-memory 占用；超过阈值会 register spilling，甚至因 shared memory 不足而无法启动 kernel。

## 2.6 实验结果与边界

在论文的 A100 80GB SXM4 测试范围内，FA2 报告：

- 相对 FA1 快约 $1.7\times$–$3.0\times$
- 相对 Triton FlashAttention 快约 $1.3\times$–$2.5\times$
- 相对标准 attention 快约 $3\times$–$10\times$
- forward 峰值约 230 TFLOPs/s，即 A100 理论峰值的 73%
- GPT 风格模型端到端训练相对 FA1 最高约 $1.3\times$，达到每卡约 225 TFLOPs/s、72% model FLOPs utilization

这些结果不是固定加速比：它们取决于 causal/non-causal、head dimension、序列长度、batch、硬件和比较基线。

论文还留下了明显的工程边界：block size 仍按少量候选手工调优；FA2 在 H100 上没有充分利用 Hopper 专属异步能力，这成为 FA3 的出发点。

## 2.7 FA2 的本质

FA2 不是简单把 FA1 的 block size 改一下，而是重新思考：

> 一个 attention 问题应该怎样映射到 CTA、warp、SM 和 sequence blocks？

所以 FA2 论文的关键词不是新的 softmax 数学，而是：

- 减少 non-matmul FLOPs
- sequence-length parallelism
- sliced-Q work partitioning
- load balance 与 occupancy

`split-KV` 则是理解当前仓库推理实现时需要额外连接的后续机制。

---

# 3. FlashAttention-3：面向 Hopper 的异步与低精度

论文：

- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608)

## 3.1 FA3 为什么不是简单的 FA2.1

Hopper GPU 引入了更强的硬件能力，例如：

- Tensor Memory Accelerator（TMA）
- WGMMA / warp-group matrix multiply
- 更适合异步数据搬运和矩阵乘的执行模型
- FP8 等低精度计算支持

如果仍然按照较老 GPU 的同步执行方式：

```text
load → wait → matmul → wait → softmax → load next tile
```

就会浪费 Hopper 的能力。

FA3 的核心方向是：

> 让数据搬运、矩阵乘和 softmax 等不同阶段尽可能重叠，而不是排成严格串行流水线。

## 3.2 Hopper 提供了哪些异步原语

论文关注的 H100 层次参数包括：

- HBM/GMEM：80 GiB，带宽约 3.35 TB/s
- 每个 SM 的 shared memory：228 KiB
- TMA：专用的 GMEM↔SMEM 异步搬运单元
- WGMMA：warpgroup 级异步矩阵乘，可直接从 shared memory 取操作数
- `setmaxnreg`：可在 warpgroup 之间动态重分配 registers

这使 CTA 内可以明确分工：producer warpgroup 主要发射 TMA，consumer warpgroup 主要发射 WGMMA 和处理 softmax；producer 需要的 registers 少，可以把更多 register 配额让给 consumer。

## 3.3 Producer–consumer 与 ping-pong

一个理想化的流水线是：

```text
时间 →

Load tile 0: █████
Matmul tile 0:    █████
Softmax tile 0:       ███
Load tile 1:      █████
Matmul tile 1:          █████
Softmax tile 1:             ███
```

当 tile 0 做矩阵乘时，可以同时搬运 tile 1；
当矩阵乘单元忙时，softmax 或下一阶段也可以尽量推进。

这要求：

- producer warp / consumer warp 分工
- 双缓冲或 ping-pong buffer
- 明确的依赖和同步
- TMA 异步加载
- WGMMA 异步矩阵乘

### Warp specialization

FA3 会把 CTA 内的 warp group 按角色划分：

- producer：负责从 global memory 把 tile 搬到 shared memory
- consumer：负责 WGMMA / Tensor Core 计算
- 其他阶段负责 softmax、归约或输出处理

这样可以让数据搬运和计算并行，而不是让同一组线程在所有阶段之间同步切换。

### 双缓冲 / ping-pong

使用两组片上 buffer：

```text
buffer A：当前正在被矩阵乘读取
buffer B：后台加载下一个 tile

下一轮交换 A/B
```

这种 ping-pong 结构可以减少“计算等数据”的空泡。

FA3 的 ping-pong 还有更具体的一层：两个 consumer warpgroups 轮换工作，让一个 warpgroup 做 softmax 时，另一个 warpgroup 发射 GEMM。论文在 head dimension 128、sequence length 8192 的 FP16 forward 示例中，报告这一调度把性能从约 570 TFLOPs/s 提高到约 620–640 TFLOPs/s。

## 3.4 两级 GEMM–softmax 重叠

softmax 是流水线难点，因为：

矩阵乘可以异步发射，但 softmax 不是普通矩阵乘：

- 需要 row-wise max reduction
- 需要 exp
- 需要 sum reduction
- 新 tile 会改变历史状态的缩放系数
- 输出累积依赖前一 tile 的状态

所以不能无限制地并行 softmax；必须保持正确的数据依赖：

```text
tile t 的 score
    ↓
更新 m / l
    ↓
得到新的 rescale
    ↓
更新 O
```

FA3 的挑战就是在不破坏 online softmax 依赖的前提下，把：

- 下一 tile 的加载
- 当前 tile 的 WGMMA
- 当前 tile 的 softmax

尽量重叠起来。

FA3 的两级 pipeline 跨相邻 KV 迭代打破局部串行链：在当前局部概率与 $V$ 做第二个 GEMM 的同时，异步计算下一个 $QK^T$；在下一块 score 上做 softmax 时，前一块的 $PV$ WGMMA 仍可推进。这里“2-stage”指寄存器中同时维持相邻迭代的状态，不等同于 shared-memory circular buffer 的 stage 数。

代价是需要额外保留下一块 score accumulator，增加 register pressure。更深的 3-stage pipeline 理论上能增加重叠，但占用更多 registers，还受到编译器指令重排影响；论文明确把 tile size 与 pipeline depth 视为需要 profiling 的权衡。

## 3.5 FP8：不只是把输入 cast 成 8 bit

FP8 能显著提升 Tensor Core 的吞吐和降低数据搬运量，但 attention 对数值误差敏感，尤其是：

- $QK^T$ 的 score 可能有较大动态范围
- softmax 对 max 附近的差异敏感
- 长序列会放大累积误差
- $V$ 的低精度误差会直接传到输出

因此低精度 attention 不能只做简单 cast。

需要考虑：

- scale 的选择
- block-wise quantization
- accumulation 使用更高精度
- score / softmax 的数值稳定性
- Q/K 的分布和 outlier
- 两个连续 WGMMA 之间不兼容的数据布局

### FP8 layout 转换

Hopper 的 FP8 WGMMA 要求 shared-memory 操作数是 k-major。通常 Q/K/V 按 head dimension 连续，但第二个 GEMM $PV$ 需要 V tile 在 sequence 维连续。论文选择在 kernel 内转置 V tile：

- producer warpgroup 使用 LDSM/STSM 在 SMEM↔register 搬运时完成 tile transpose
- 从第 2 个 tile 起，把下一块 V 的转置隐藏在当前 K/V 相关 WGMMA 后面
- 用 byte-permute 调整第一个 FP8 WGMMA 的 FP32 accumulator，使其满足第二个 WGMMA 的寄存器操作数布局

这说明低精度加速不仅是数值问题，也是 layout-conformance 问题。

### Block quantization

per-tensor quantization 为整个 Q、K、V 各使用一个 scale，容易被少量 outlier 主导。FA3 改为每个 tile/block 使用 scale；由于 attention kernel 本来就按 block 工作，可在每个 score block 中吸收相应缩放。论文指出量化本身还可与 rotary embedding 等 bandwidth-bound 前序操作融合。

### Incoherent processing 的精确机制

论文在量化前对 Q 和 K 右乘同一个随机正交矩阵 $M$：

$$
Q'=QM,\qquad K'=KM
$$

因为 $M M^T=I$：

$$
Q'K'^T
=QM(KM)^T
=QMM^TK^T
=QK^T
$$

所以在没有量化误差时，score 完全不变；变换的作用是把少数大 outlier 分散到更多坐标，减少 FP8 量化误差。实现采用随机 $\pm1$ 对角矩阵与 Hadamard 矩阵的乘积，将变换复杂度从朴素的 $O(d^2)$ 降为 $O(d\log d)$，并可与 rotary embedding 融合。

```text
原始 Q/K：少数维度有巨大 outlier
      ↓ 同一个正交变换
能量分散，但 QKᵀ 不变
      ↓ block-wise FP8 quantization
减少有限动态范围被 outlier 独占
```

这里要特别区分：

- FA1 / FA2 的核心路径强调 exact attention
- FA3 的 FP8 路径是在低精度表示下尽量保持准确性
- “低精度近似误差”与“算法层面的稀疏/低秩近似”不是一回事

## 3.6 FA3 的优化层次

可以把 FA3 的优化分成三层：

### 算法/数据流层

- 仍然采用 tile 化和 online softmax
- 继续避免完整 attention matrix
- 继续用 LSE / 局部状态合并结果

### 执行模型层

- 异步 TMA load
- 异步 WGMMA
- warp specialization
- 双缓冲 / ping-pong
- 尽量隐藏 global memory latency 和 softmax 开销

### 数值层

- FP8 forward
- block quantization
- accumulation 精度控制
- 低精度下的误差补偿 / 激活处理

## 3.7 实验结果、准确性与限制

论文在 H100 80GB SXM5 上报告：

- FP16 forward 相对 FA2 快约 $1.5\times$–$2.0\times$，最高约 740 TFLOPs/s，即理论峰值的 75%
- FP16 backward 相对 FA2 快约 $1.5\times$–$1.75\times$
- FP8 forward 接近 1.2 PFLOPs/s
- FP16 FA3 与 FP16 FA2 的数值误差相同；两者都因 softmax 等中间量保持 FP32，在论文测试中比标准 FP16 实现 RMSE 更低
- 在带 0.1% 人工 outlier 的测试分布上，FP8 FA3 的 RMSE 为 $9.1\times10^{-3}$，per-tensor baseline 为 $2.4\times10^{-2}$，即约 $2.6\times$ 更低误差

这些准确性数字来自论文特定的合成 outlier 实验，不能直接外推成任意模型训练的质量保证。

论文明确列出的限制包括：

- 尚需针对 LLM inference 优化
- FP8 kernel 尚未集成 persistent-kernel 设计；这部分解释了其在短序列和 causal masking 下不如 FP8 cuDNN
- 低精度 attention 对大规模训练的影响仍需研究
- 2/3-stage pipeline 会增加 register pressure，并受 NVCC 指令重排影响
- 论文实现和基准主要针对 Hopper；思想可能泛化，但具体性能依赖硬件异步与低精度能力

## 3.8 FA3 的本质

FA3 的关键不是重新发明 online softmax，而是：

> 在 Hopper 上，把 online softmax attention 重新安排成更深的异步流水线，并用低精度获得更高吞吐。

---

# 4. 三篇论文的统一数据流

## 4.1 Forward 的统一抽象

```text
for each Q tile:
    load Q tile
    initialize m, l, O

    for each K/V tile:
        load K/V tile
        S = Q_tile @ K_tileᵀ
        apply scale / mask / bias
        update online softmax state (m, l)
        rescale old O
        O += local_probability @ V_tile

    write O and LSE
```

三代的差别主要在于：

- FA1：怎样让这个循环不产生巨大 HBM IO
- FA2：怎样把这个循环分给更多 CTA / warp
- FA3：怎样让 load、WGMMA、softmax 在 Hopper 上异步重叠

## 4.2 Backward 的统一抽象

```text
precompute row correction D from dO and O

for each Q/K/V tile pair:
    recompute S
    recover P using saved LSE
    regenerate dropout mask if needed
    compute dV
    compute dS using softmax derivative
    compute dQ and dK
    accumulate gradients
```

关键事实：

- forward 不保存 $N \times N$ 的 $P$
- backward 用重计算恢复 $P$
- `softmax_lse` 是恢复概率的必要摘要
- dropout 必须用可复现 RNG 重新生成相同 mask

## 4.3 Split-KV 的统一抽象

```text
KV sequence
 ├── split 0 → local O₀, local LSE₀
 ├── split 1 → local O₁, local LSE₁
 ├── split 2 → local O₂, local LSE₂
 └── split S → local Oₛ, local LSEₛ

combine:
    LSE = logsumexp(local LSEs)
    O = Σ exp(local LSEs - LSE) * local Os
```

这张图是理解 FA2 和推理 decode 路径的关键。

---

# 5. 从论文到 CUDA 代码的映射

| 论文思想 | 代码中应重点寻找的对象 |
|---|---|
| IO-aware | tile copy、shared memory、register accumulator |
| online softmax | `m`、`l`、`softmax_lse`、normalize / rescale 逻辑 |
| exact backward | `softmax_lse`、`softmax_d`、重算 score / probability |
| dropout 重建 | Philox、`rng_state`、tile / lane 定位 |
| work partitioning | launch template、CTA mapping、`num_splits` |
| split-KV | split forward kernel、LSE accumulator、combine kernel |
| MQA/GQA | `h`、`h_k`、head ratio、expanded dK/dV |
| varlen | `cu_seqlens_q`、`cu_seqlens_k` |
| paged KV cache | `block_table`、page block size |
| Hopper async | TMA、WGMMA、warp specialization、ping-pong buffer |
| FP8 | quantization scale、block quantization、higher-precision accumulation |

对应已有源码笔记：

- [FlashAttention 源码精读](./flash-attention-source-reading.md)
- [FlashAttention 系统地图](./flash-attention-system-map.md)
- [FlashAttention 接口与 Autograd](./flash-attention-interface-and-autograd.md)
- [FlashAttention Kernel 与 Launch 机制](./flash-attention-kernel-and-launch.md)
- [FlashAttention Kernel 细节补充](./flash-attention-kernel-details.md)
- [FlashAttention PyTorch ATen 接入层](./flash-attention-pytorch-aten-integration.md)

---

# 6. 关键状态表

| 状态 | 数学/工程含义 | 主要出现在哪条路径 |
|---|---|---|
| $m$ | 当前已处理 score 的行最大值 | forward online softmax |
| $\ell$ | 以当前 $m$ 为基准的指数和 | forward online softmax |
| `softmax_lse` | 最终 $m+\log \ell$ | forward 输出、backward、split combine |
| `out` / $O$ | 未归一化或最终 attention 输出 | forward、backward |
| `softmax_d` / $D$ | softmax backward 的行修正项 | backward |
| `rng_state` | dropout mask 可复现所需状态 | dropout forward/backward |
| `num_splits` | KV 被切分的数量 | 仓库的 split-KV / decode 实现 |
| `dq_accum` | dQ 的中间累积缓冲 | backward / deterministic |
| `cu_seqlens_*` | varlen token 边界前缀和 | varlen |
| `cache_seqlens` | 每个序列当前 cache 长度 | kvcache 推理 |
| `block_table` | 逻辑序列到物理 cache block 的映射 | paged cache |

## 最重要的两个状态

### 训练态：`softmax_lse`

它把完整 softmax matrix 压缩成每行一个数，同时保留 backward 恢复归一化所需的信息。

### 推理态：`cache_seqlens`

它告诉 kernel 新 token 应写到 cache 的什么位置，以及当前 query 的 causal / local 位置语义如何对齐。

---

# 7. 三代优化的本质对比

| 维度 | FA1 | FA2 | FA3 |
|---|---|---|---|
| 主要瓶颈 | HBM IO / 中间矩阵 | 并行度 / 负载均衡 / non-matmul 开销 | Hopper 执行与低精度利用 |
| 核心机制 | tiling + online softmax | sequence parallelism + sliced-Q + 更少 rescale | TMA/WGMMA async pipeline + warp specialization |
| 重点硬件 | SRAM / shared memory | SM / CTA / warp / Tensor Core | TMA / WGMMA / Hopper Tensor Core |
| 中间状态 | $m,\ell,O$、LSE | 仅保存 LSE、warp-local Q-row 状态 | circular buffer、相邻迭代 score、FP8 block scale |
| 主要代价 | backward 重计算、kernel 复杂 | atomic dQ、tile 手工调优、register/shared-memory 权衡 | register pressure、layout 转换、硬件专用、量化误差 |
| 最适合记忆的句子 | 不要写出中间矩阵 | 不要让 GPU 闲着 | 不要让 Hopper 等数据 |

---

# 8. 常见误解

## 误解 1：FlashAttention 是近似 attention

不是。FA1/FA2 的核心目标是 exact attention；它们主要通过改变数据流减少 IO。

## 误解 2：只要不保存 attention matrix 就一定更快

不一定。若 tile 太小、同步太多、并行度不足、combine 成本过高，理论上的显存优势不一定转化为端到端速度优势。

## 误解 3：split-KV 可以把局部输出直接相加

不能。每个 split 的 softmax 归一化基准不同，必须通过 LSE 做 log-sum-exp combine。

## 误解 4：FA3 只是把 head dimension 改成 FP8

不是。FA3 的重要变化还包括 Hopper 上的异步数据搬运、WGMMA、warp specialization 和流水线重叠。

## 误解 5：shared memory 越多越好

不是。shared memory、register 使用量、occupancy、tile size、Tensor Core 利用率之间需要平衡。

## 误解 6：训练和推理使用同一套最优调度

不一定：

- 训练通常 $N_q,N_k$ 都较大，重视吞吐和 backward
- decode 通常 $N_q$ 很小、$N_k$ 很大，重视 split-KV、cache layout 和短 query 并行度

---

# 9. 建议的论文阅读顺序

## 第一遍：只看主线

1. 读 FA1 的摘要、Introduction、Conclusion
2. 读 FA2 的摘要、Introduction、改进点
3. 读 FA3 的摘要、Introduction、Hopper 优化概览
4. 回来看本笔记的“三代优化对比表”

目标：先记住 `IO → parallelism → Hopper async/low precision`。

## 第二遍：理解数学和数据流

1. FA1：online softmax 推导
2. FA1：forward tiling、IO 复杂度和 backward recomputation
3. FA2：sequence parallelism、sliced-Q 和 non-matmul 优化
4. 实现扩展：split-KV 的 LSE combine
5. FA3：异步流水线、layout 转换和低精度数值策略

目标：能画出 forward、backward、split combine 三张图。

## 第三遍：对照源码

1. 看 `flash.h` 的参数结构
2. 看 `flash_fwd_launch_template.h`
3. 看 `flash_fwd_kernel.h`
4. 看 `flash_bwd_launch_template.h`
5. 看 `flash_bwd_kernel.h`
6. 看 `flash_attn_with_kvcache`
7. 最后看 PyTorch ATen 接入

目标：把论文中的每个关键词定位到代码中的参数、函数和 kernel。

---

# 10. 最后压缩成四句话

1. **FA1：用 tiling 和 online softmax，在不写出 $N \times N$ 矩阵的情况下精确计算 attention。**
2. **FA2：减少 non-matmul 开销，并用 sequence parallelism 与 sliced-Q 重新设计 CTA/warp 的工作划分。**
3. **FA3：利用 Hopper 的 TMA、WGMMA、warp specialization 和低精度能力，把数据搬运与计算异步重叠。**
4. **整个系列的底层原理是：数学不变，改变数据流、内存层次、并行切分和硬件执行方式。**

## 关联知识

- [FlashAttention 阅读导览](./flash-attention-reading-guide.md)
- [FlashAttention 术语表与关键状态表](./flash-attention-glossary-and-state-table.md)
- [FlashAttention 系统地图](./flash-attention-system-map.md)
- [AI Infra 方向论文地图](../../../../inbox/ai-infra-papers-map.md)

## References

- https://arxiv.org/abs/2205.14135
- https://arxiv.org/abs/2307.08691
- https://arxiv.org/abs/2407.08608
