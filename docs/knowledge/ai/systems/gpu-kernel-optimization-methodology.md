---
title: GPU 算子优化方法论：计算、通信、存储
type: concept
status: seed
tags: [AI, CUDA, GPU, Performance, Memory, TensorCore, Occupancy]
created: 2026-09-11
updated: 2026-09-11
source: 知乎《CUTLASS 笔记 (3)：Tiled MMA》第 1 节（https://zhuanlan.zhihu.com/p/1950555644814946318）+ 个人整理
---

# GPU 算子优化方法论：计算、通信、存储

## 一句话理解

GPU 算子优化只有三个可下手的地方：**计算（FLOPS / Tensor Core 利用率）、通信（访存 latency）、存储（GMEM/SMEM/RMEM 的容量与 Occupancy）**。在 Tensor Core 时代，算力增长快于访存带宽增长，所以大部分常见算子是 **memory bound** —— 当 FLOPS 不达预期时，第一反应应该是查通信与存储，而不是改算法。

## 为什么重要

- 它给出一个**排障顺序**：FLOPS 低 → 先看是不是访存卡住了 Tensor Core，而不是盲optimize计算。
- 它解释了后面所有分块（Tiling）、流水线（pipeline）、向量化设计的**动机来源**：每一种优化手段都在回应某个维度的瓶颈。
- 三个维度相互牵制（下面表格），任何单点优化都可能从别处引入副作用。

## 三个维度总览

| 维度 | 关注指标 | 典型瓶颈表现 | 优化方向 |
|---|---|---|---|
| **计算** | FLOPS、Tensor Core 利用率 | 利用率低 | 算法层减少 FLOPs；合理安排依赖让 Tensor Core 满载 |
| **通信** | 数据在存储介质间搬运的 latency | memory bound | 减少高 latency 通信量；用流水线把通信时间藏在计算后面 |
| **存储** | Occupancy（SMEM / RMEM 利用率） | SMEM 溢出 / register spilling | 以存储换效率，同时避免副作用 |

三者**三位一体**：加 warp 提并行度会吃掉寄存器，减少 GMEM 访存要用 SMEM 做中转，SMEM 又要占容量。优化本质上是在这三者之间找平衡点。

## 计算层面

常用指标是每秒浮点运算次数 FLOPS。在以 Tensor Core 为计算核心的架构上，更实用的指标是 **Tensor Core 利用率**。

受**功耗墙**限制，无法发挥 Tensor Core 的理论最大算力，但总可以在算子内塞满 mma 指令逼近**实际**算力上限。因此计算层的优化主要落在**算法层面**：

- 用更少的 FLOPs 实现同样的计算（如 FlashAttention 的 IO-aware 重排、recompute）；
- 合理安排计算依赖，让 Tensor Core 不出现空转等待。

> **关键判断**：FLOPS 低 ≠ 计算层面需要优化。很多情况下是通信或存储瓶颈让 Tensor Core 拿不到数据、接不到任务。必须具体问题具体分析。

## 通信层面

单卡场景下主要关注**数据从一个存储介质搬到另一个存储介质的耗时（latency）**。

核心事实：**Tensor Core 算力极大，计算数据的时间往往小于搬运数据的时间**，因此当前大多数算子在 Tensor Core 面前都是 memory bound。

优化手段只有两条：

1. **减少高 latency 的通信量** —— 两个介质间的 latency 由硬件决定，改不了，只能少搬；
2. **用流水线掩盖** —— 通过多级缓冲让通信与计算重叠，把通信时间藏在计算时间背后。

## 存储层面

GPU 当前可编程的存储单元有三种：

```text
               容量          延迟        可见范围
GMEM (显存)     最大       最高(最慢)    多算子共享（片外）
  ↑
SMEM (共享内存) 几十~几百 KB 中等         一个 thread block 内共享（片内）
  ↑
RMEM (寄存器)   最受限      最低(最快)    计算单元直接读取（片内）
```

FlashAttention 正是利用 GMEM 与 SMEM 的访存效率差异，用 SMEM 作为中间介质大幅减少 GMEM 读写，从而优化 Attention 算子。

### 容量比效率更致命

在很多场景下，SMEM / 寄存器的**存储容量**比访存效率更关键，这就是 **Occupancy**（存储资源利用率）。

SM90（Hopper）/ SM100（Blackwell）的硬约束：

| 资源 | 上限 |
|---|---|
| SMEM | 单 thread block 最多 **227 KB** |
| RMEM | 单 block 最多 **64K 个寄存器** |
| RMEM | 编译器限制：单线程最多 **255 个寄存器** |

两种溢出的后果**不对称**：

- **SMEM 溢出** → 算子根本跑不起来（硬失败）；
- **RMEM 溢出（register spilling）** → 溢出到 Local Memory，最坏情况下其访存效率约等于 GMEM。而寄存器在典型场景下会被访问上千万次，因此代价极大。

Occupancy 不足本身不会有严重后果，但它提示算子仍有访存效率与数据复用率的优化空间。所以存储层优化要在**尽可能提高 Occupancy 的前提下避免任何副作用**，用存储换效率。

## 我的理解

（以下为个人理解，非原文结论）

- 这三个维度不是并列的三条路，而是一条**因果链**：存储容量限制 → 决定能搬多少数据 → 决定通信是否成为瓶颈 → 最终表现为 Tensor Core 利用率不足。所以"FLOPS 低"往往是最下游的症状。
- 真正稀缺的资源是**寄存器**：SMEM 溢出会立刻报错，反而容易发现；寄存器溢出只表现为性能下降，最容易被忽略，也最难调。
- 这条方法论对 FlashAttention 同样成立：FA 的 split-KV、多级缓冲、recompute 都能在这三个维度上找到对应解释。

## 常见误区

1. **"FLOPS 低就是计算不够优"** → 错。先排查 memory bound。
2. **"Occupancy 越高越好"** → 不一定。Occupancy 只是必要条件，数据复用率和访存效率同样重要，且提高 Occupancy 常以牺牲每线程数据复用为代价。
3. **"SMEM 和寄存器只是快慢差别"** → 错。它们的**容量**在 Tensor Core 越做越大的今天已成为硬约束。
4. **"通信 latency 可以靠调 API 降低"** → 介质间的 latency 由硬件决定，可优化的只有通信量与重叠程度。

## Related

- [GEMM 三级 Tiling](./cutlass/06-gemm-three-level-tiling.md) — 这套方法论在 GEMM 上的直接落地
- [CUTLASS/CuTe 02：Copy Atom 与线程分区](./cutlass/02-cute-copy-and-thread-partition.md) — 通信层的具体编程抽象
- [CUTLASS/CuTe 05：Copy 规模核算与 128-bit 向量化](./cutlass/05-cute-copy-scaling-and-vectorization.md) — 通信量如何折算成指令数
- [FlashAttention 系统地图](./flash-attention/flash-attention-system-map.md) — memory bound 优化的代表案例
- [Profiling 专题](../../engineering/profiling/) — **这三个维度各自怎么测出来**：机器层 → 框架层 → 时间线 → 单 kernel
- [Nsight Compute（ncu）](../../engineering/profiling/06-nsight-compute-ncu.md) — `Speed of Light` 的 SM%/Memory% 正是"算力 / 带宽"两轴的直接测量
- [AI Systems](./)

## References

- 知乎《CUTLASS 笔记 (3)：Tiled MMA》第 1 节"算子优化方法论"，作者杨远航，2025-09-14：<https://zhuanlan.zhihu.com/p/1950555644814946318>
- 功耗墙讨论：<https://www.thonking.ai/p/strangely-matrix-multiplications>
