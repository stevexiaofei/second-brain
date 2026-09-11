---
title: GPU 全局内存访存模型：向量化与合并访存
type: concept
status: seed
tags: [AI, CUDA, GPU, Memory, Coalescing, Vectorization, Sector, Transaction]
created: 2026-09-11
updated: 2026-09-11
source: 知乎《CUTLASS 笔记 (4)：Tiled Copy》第 1 节（https://zhuanlan.zhihu.com/p/1968745447741972494）+ 个人整理
---

# GPU 全局内存访存模型：向量化与合并访存

## 一句话理解

GMEM 的访存效率由两件**正交**的事决定：

- **向量化访存** —— 单条指令一次搬多宽（上限 128 bit），决定**指令条数**；
- **合并访存** —— 一个 warp 的 32 个线程地址是否连续且对齐，决定**实际搬运的字节数**。

后者的硬件最小单元是 **32 字节的 sector（sector = transaction 的最小粒度）**。只要一个 warp 的访存不连续或不对齐，硬件就会把用不到的字节一起搬走 —— **实际 GMEM 访存量可能成倍于指令所需**。

## 为什么重要

- 这是[算子优化方法论](../gpu-kernel-optimization-methodology.md)里"通信层面"的**微观物理基础**：memory bound 不是抽象概念，而是可以用 sector 数算出来的。
- 它解释了为什么 TiledCopy 的 thread / value layout 不能随便选 —— 布局直接决定每个 warp 的访问形状。
- 它给出一个可量化的排障指标：**ncu 报告的 "多余的 GMEM 访存量" 百分比**。

## 一、向量化访存

目标是**用更长的访存指令**减少指令调度次数、提升指令级并行度、吃满带宽。

单条 PTX / SASS 指令支持的长度（**当前单指令上限为 128 bit**）：

| 长度 | PTX | SASS |
|---|---|---|
| 32 bit | `ld.global.u32` | `LDG.E` |
| 64 bit | `ld.global.v2.u32` | `LDG.E.64` |
| 128 bit | `ld.global.v4.u32` | `LDG.E.128` |

**编译器只在数据物理连续时才会选更长的指令。** Minimal GEMM 里读 A 矩阵用了两条 32-bit 指令（`LDG.E`）而不是一条 64-bit，正是因为该线程的两个 32-bit 数据块在内存中不连续。

> 实践规则：**尽可能让单个线程访存的数据在物理上连续**，这样编译器才能合并成更宽的指令。

## 二、合并访存

NV GPU 采用 SIMT 架构：一个 warp 的 32 个线程**同时**执行同一条访存指令。硬件层面会把这次请求合并成若干个 **transaction**。

核心模型：

```text
transaction  硬件访存的最小单元
   └─ 一次 transaction 访问一段连续且内存对齐的 32 bytes = 1 个 sector
```

三条必须记住的推论：

1. **sector 是 32 字节**，`384 bytes / 32 = 12` 次 transaction（连续对齐的最优情况）；
2. 从 GMEM 读出的 sector **无论是否被实际使用，都会经过各级 Cache** —— 用不到的部分纯属浪费；
3. 只要满足以下任一条，实际 GMEM 访存量就会 **大于** 指令所需的数据量：
   - 一次访存 < 32 bytes（读不满一个 sector）；
   - 访存区域跨多个 sector 且不连续；
   - 起始地址未按 32 字节对齐。

> 原文关于 transaction 粒度的说明：**SM60 以下**的架构，访存经过 L1 Cache 时一次 transaction 的数据量会变为 **128 bytes**；**SM60 及以上**则无论是否经过 L1，一次 transaction 固定为 **32 bytes**。

## 三、案例：Tiled MMA 的访存问题诊断

上一篇的 Tiled MMA 算子被 ncu 提示"部分指令多读了 GMEM 数据"。用上面的模型可以把它算清楚。

### 理论最优：16 个 sector

以 A 矩阵、第一个 warp（T0–T31）为例：

```text
最优情况
  第一个 warp 所需数据全部落在 A(0,0) 与 A(0,1) 区域
  每一行的长度 = 32 bytes = 恰好 1 个 sector
  ⇒ 读取 A 只需 16 个 sector，即 16 次 transaction
```

### 实际：32 个 sector

但每个线程持有的是 **4 块不连续的数据**，因此实际访存由 **4 条 `LDG.E`** 完成：

```text
实际情况
  每条 LDG.E 访问的是不连续的内存区域
  ⇒ 单条 LDG.E 就跨越/读取了 8 个 sector，而实际只需要其中一半
  ⇒ 4 条 LDG.E 共读取 32 个 sector（最优只需 16 个）
  ⇒ 实际 GMEM 访存量 = 最优的 2 倍
```

这就是 ncu 报告"**每条 `LDG.E` 有 50% 的 GMEM 访存是多余的**"的来源（因为有 Cache，访存**时间**的增加小于一倍，但**字节量**确实翻倍）。

### 两条看似可行的方案，为什么都不行

| 方案 | 思路 | 为什么不可行 |
|---|---|---|
| ① 让单线程的数据块物理连续 | 按向量化思路重排 | **无法实现**。没有办法通过控制 MMA Permutation（任意交换 MMA Atom 的行和列）让同一个 Atom、同一个线程对应的两个不连续数据块变连续 |
| ② 把 K 维长度固定为 8 | 按合并访存思路，保证 LDG.E 区域连续 | **牺牲太大**。这样就无法在 K 维度扩展 MMA 规模，处理大矩阵时性能受限 |

> 原文补充的第三条路：让线程先拷贝一段连续数据，再通过 **warp 内线程数据交换**（shuffle）拿到各自所需。代价是增加一个数据交换步骤，编程复杂度较大。

### 正解：引入 SMEM

要彻底解决这个访存问题，必须使用 **Shared Memory**：

```text
GMEM --(合并 + 向量化的连续访问)--> SMEM --(按 MMA 需要的方式重排)--> 寄存器
         ↑ 这一步可以做到完美连续对齐        ↑ SMEM 内部重排的代价远低于 GMEM 重复读
```

**这正是 GEMM kernel 里 `gmem → smem → register` 这条路径存在的真正原因** —— 不是为了"多一级缓存"，而是为了**把两个互相矛盾的需求解耦**：GMEM 侧要连续对齐，MMA 侧要特定排布。

## 我的理解

（以下为个人理解）

- 把访存问题拆成"**宽度**（向量化）"和"**形状**（合并）"两把尺子，绝大多数 memory bound 问题都能定位：宽度不够 → 指令太多；形状不对 → 字节数浪费。
- sector 模型最有价值的地方是**让浪费变得可数**。8 个 sector 里只用 4 个 → 50% 浪费，这是能在 ncu 里直接对上的数字，比"访存效率低"这种定性描述有用得多。
- 注意两者的**作用域不同**：向量化是**单线程内**的问题（我一次读几个），合并访存是**跨线程**的问题（32 个线程的空间关系）。所以它们需要 TiledCopy 里两个独立的 layout 分别控制 —— 这正好对应 `thread_layout` 与 `value_layout` 的分工。
- 还有一个隐含结论：**布局选择不是审美问题，而是物理问题**。同一个逻辑 tile，换个 thread/value layout 就可能从 16 个 sector 变成 32 个。

## 常见误区

1. **"读得少就等于搬得少"** → 错。小于 32 bytes 或跨 sector 的访问，硬件会把整个 sector 搬走。
2. **"128 bit 向量化是自动的"** → 错。需要数据物理连续 **且** 起始地址 16 字节对齐，编译器才可能选用。
3. **"合并访存是编译器优化的结果"** → 不准确。合并发生在硬件层面（warp 请求 → transaction），但**是否合并得起来由程序员写的布局决定**。
4. **"访问不连续只是慢一点"** → 错。字节量可能翻倍，而 GMEM 带宽是 memory bound kernel 的硬上限。
5. **"有了 Cache 就不必在意浪费"** → 错。Cache 只减小**时间**上的损失，被读走的 sector 仍然占用了带宽。

## Related

- [GPU 算子优化方法论：计算、通信、存储](../gpu-kernel-optimization-methodology.md) — 通信层面的宏观框架
- [Nsight Compute（ncu）：单 kernel 的硬件计数器分析](../../engineering/profiling/06-nsight-compute-ncu.md) — 本篇的 sector 模型在 ncu 里就是 `Sectors Per Request` 这一个指标
- [Profiling 专题](../../engineering/profiling/) — 怎么把"实际访存量翻倍"测出来
- [CUTLASS/CuTe 09：TiledCopy 核心原理](./cutlass/09-cute-tiled-copy-principle.md) — 用 S/D/R Layout 描述"谁访问哪些地址"
- [CUTLASS/CuTe 02：Copy Atom、TiledCopy 与线程分区](./cutlass/02-cute-copy-and-thread-partition.md) — thread/value layout 如何决定访问形状
- [CUTLASS/CuTe 05：Copy 规模核算与 128-bit 向量化](./cutlass/05-cute-copy-scaling-and-vectorization.md) — 向量化约束的 TiledCopy 侧推导
- [AI Systems](./)

## References

- 知乎《CUTLASS 笔记 (4)：Tiled Copy》第 1 节"NV GPU 的全局内存访存特性"，作者杨远航，CUTLASS 4.1.0 / SM90：<https://zhuanlan.zhihu.com/p/1968745447741972494>
- PTX ISA：`ld.global` 的向量化形式（`.v2` / `.v4`）
