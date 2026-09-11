---
title: GEMM 三级 Tiling：Global → Block → Tile → MMA Atom
type: concept
status: seed
tags: [AI, CUDA, CUTLASS, GEMM, Tiling, TensorCore, SMEM]
created: 2026-09-11
updated: 2026-09-11
source: 知乎《CUTLASS 笔记 (3)：Tiled MMA》第 2 节（https://zhuanlan.zhihu.com/p/1950555644814946318）+ 个人整理
---

# GEMM 三级 Tiling：Global → Block → Tile → MMA Atom

## 一句话理解

GEMM 的三级 Tiling 不是"为了分块而分块"，而是**每一条硬件特性的一条对应措施**：

```text
① 想用多个 SM 并行  →  把 Global 切成 Block
② 想用 SMEM 低延迟 + 数据复用  →  把 Block 切成 Tile
③ 想喂饱 SM 内的多个 Tensor Core  →  给每个 Tile 塞足够多的 MMA Atom
```

因为动机来自硬件，所以**硬件换代时 Tiling 方式一定会变**（例如 Blackwell 的 2SM MMA 借 Distributed SMEM 多出了一层 Cluster 级 Tiling）。

## 为什么重要

- 它是理解一切高性能 GEMM kernel（CUTLASS、cuBLAS、FlashAttention 的 GEMM 部分）的**概念模型**：先看它怎么切三层，再看每层怎么优化访存。
- 它把"性能调优"翻译成了具体的三个可调参数：block 大小、tile 大小、每线程 mma 条数。
- 它解释了一个反直觉的事实：**写一个正确的 GEMM 不难，难的是利用硬件和指令集特性写出性能最佳的 GEMM。**

## 逐级推演

### 起点：只有一条 MMA 指令

前两篇实现的是单指令 `16x8x8` MMA。真实矩阵规模远大于此，需要把单指令扩展到大得多的规模。理论上可以只做**一级 Tiling**：单 block 内循环执行 `16x8x8` 就算完任意规模 GEMM。

但这浪费了硬件：**完全没有用到多 SM 并行**。

### 第一级：Global → Block（要的是 SM 并行）

把 D 矩阵按 `16x8` 切分成若干 tile，tile 之间互相独立可并行，一个 tile 交给一个 block。

block 内部沿 K 维循环：每轮从 GMEM 拷 `16x8` 的 A 分片与 `8x8` 的 B 分片到寄存器，执行 MMA，累加，最后一并写回 GMEM。

**新问题**：SM 并行起来了，但单个 SM 的算力没吃满。

- 一次循环只有一个 warp 执行一次 mma，而 FP16 `16x8x8` 每线程只需 5–7 个寄存器，一个 warp 最多约 224 个寄存器 —— 远低于单 block 的 64K；
- 单个 SM 有 **4 个 Tensor Core**，单 warp 循环只能用到 1 个。

### 第二级：Tile 内扩展（要的是 Tensor Core 满载）

两条思路同时用：

1. **增加并行度**：扩线程数、加 warp 数量；
2. **增加每 warp 的指令数**：在每个 warp 内循环执行多条 mma。

这两件事不仅提高算力利用率，还能**一次性从 GMEM 拷更多数据**，充分利用 GMEM → RMEM 的带宽。

文章的例子：线程扩 8 倍（M 维 ×2、N 维 ×4），每 warp 的 mma 沿 K 维扩 2 倍 → tile 变为 `32x32x16`，共 256 线程。

**不是可以无限扩**：

| 约束 | 后果 |
|---|---|
| 单 block 线程数 ≤ 2048 | 硬上限 |
| 线程扩展 / 指令扩展都要更多寄存器 | register spilling → 性能损失 + 大量 GMEM 占用 + 编译时间暴涨 |

### 第三级：Block 内扩展（要的是数据复用）

tile 级别时 GMEM 访存量仍然很大，而且**重复访存严重**。

关键观察：GEMM 中**同一行的 tile 共用同一行 A 分片，同一列的 tile 共用同一列 B 分片**。如果一个 block 内循环完成多个 tile，把这批 tile 共用的 A/B 分片从 GMEM 拷到 **SMEM**，就能显著减少重复搬运。

文章例子：把 tile 再按 $(M,N,K)=(4,4,2)$ 扩展 → block 规模 `128x128x32`。

**同样不能无限扩**：

| 约束 | 后果 |
|---|---|
| 单 block 的 SMEM 容量有限 | 一次性放不下整个 A/B 分片，必须分批拷贝 |
| block 过大 → block 总数变少 | SM 维度并行度下降 |

### 三级全景

```text
                        K →
      ┌─────────────────────────────────────┐
 M    │  Block(0,0)    │   Block(0,1)        │   ← Global → Block：多 SM 并行
 ↓    ├────────────────┼─────────────────────┤
      │  Block(1,0)    │   Block(1,1)        │
      └─────────────────────────────────────┘
             │  block 内沿 K 循环，每轮拷一对 A/B 分块进 SMEM（数据复用）
             ▼
      ┌───────────────────┐
      │  Tile  128x128x32 │   ← Block → Tile：吃满 SMEM 复用 + 多 warp 协作
      └───────────────────┘
             │  Tile 由多个 MMA Atom 拼成
             ▼
      ┌──────────┐
      │ MMP Atom │   ← Tile → MMA Atom：一条硬件指令，落到 Tensor Core
      │ 16x8x8   │
      └──────────┘
```

## 三级与硬件特性的对应

| 层级 | 切分动机 | 依赖的硬件特性 | 主要代价 |
|---|---|---|---|
| Global → Block | 多 SM 并行 | 多 SM / 多 CTA | 重复访存（A/B 分片被多次读取） |
| Block → Tile | 减少 GMEM 访存、数据复用 | SMEM 容量 + 低延迟 | SMEM 占用、block 数减少 |
| Tile → MMA Atom | 吃满 Tensor Core | 单 SM 多 Tensor Core | 寄存器压力、register spilling |

## 我的理解

（以下为个人理解，非原文结论）

- 三级 Tiling 的本质是**用一个中间存储层换一次访存优化**：Block 换 SMEM，Tile 换寄存器。每一级都在回答"上一层留下的浪费怎么消掉"。
- 三层约束的**失败模式不同**：SMEM 溢出会直接编译/运行失败（容易暴露），寄存器溢出只表现为性能下降（隐蔽），block 数不足只是并行度低（最不易察觉但影响最大）。
- 记住"动机 → 硬件特性"的对应关系，比记住 `128x128x32` 这类具体数字更重要 —— 数字是给定架构下的经验最优解，架构一变就失效。这也解释了为什么 CUTLASS 用模板参数把这三层全参数化，而不是写死。

## 常见误区

1. **"三级 Tiling 是 GEMM 的数学性质决定的"** → 错。它是硬件约束（多 SM / SMEM 容量 / Tensor Core 数量）的映射，换硬件就要重切。
2. **"tile 越大越好"** → 错。受限线程数 2048、SMEM 容量与寄存器压力，过大反而降性能。
3. **"只做一级 Tiling 也能算，所以多级只是为了好看"** → 错。一级 Tiling 放弃了 SM 并行与数据复用，性能差几个量级。
4. **"增加线程数就一定能提高利用率"** → 不一定。线程数与每线程指令数需要与 tile 规模匹配，否则大量线程空转。

## Related

- [GPU 算子优化方法论：计算、通信、存储](../gpu-kernel-optimization-methodology.md) — 三级 Tiling 的理论动机来源
- [CUTLASS/CuTe 03：TiledMMA 与 fragment](./03-cute-tiled-mma.md) — 第三级"Tile → MMA Atom"的具体实现
- [CUTLASS/CuTe 04：GEMM 数据流](./04-cute-gemm-pipeline.md) — 三级 Tiling 在代码中的视图流转
- [CUTLASS/CuTe 07：CuTe Permutation Layout](./07-cute-permutation-layout.md) — 同一 Tile 内 Atom 的排列方式
- [FlashAttention 系统地图](../flash-attention/flash-attention-system-map.md) — 同样的分层思想在 Attention 上的应用

## References

- 知乎《CUTLASS 笔记 (3)：Tiled MMA》第 2 节"GEMM 三级 Tiling"，作者杨远航，CUTLASS 4.1.0 / SM90：<https://zhuanlan.zhihu.com/p/1950555644814946318>
- 前序：<https://zhuanlan.zhihu.com/p/1937517614084650073>（笔记 1）、<https://zhuanlan.zhihu.com/p/1940158874255602181>（笔记 2）
