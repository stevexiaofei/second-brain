---
title: CUTE TiledMMA：一条 MMA 指令如何"复制"成大 tile
type: concept
status: growing
tags: [AI, CUDA, CUTE, CUTLASS, MMA, TiledMMA, Kernel]
created: 2026-09-01
updated: 2026-09-01
source: cutlass-notes/01-minimal-gemm + 03-tiled-mma + 15-tiled-gemm（本地实践，SM80）
---

# CUTE TiledMMA：一条 MMA 指令如何"复制"成大 tile

## 一句话理解

TiledMMA 是把单条 `mma.sync` 指令（一个 warp 算一个小矩阵）在 M/N/K 三个方向"复制"若干份组成的大 MMA：**线程层复制（ThrExpand）决定放多少个 warp，数值层复制（ValExpand）决定每个线程把指令执行几遍**。kernel 的 tile 大小 = 指令形状 × 这两个复制系数的乘积，所以 tile 不必等于指令，只需是它的整数倍。

## 为什么重要

- 真实 GEMM 的输出 tile（64×64、128×128 等）远大于单条指令（16×8），没有 TiledMMA 就无法表达"一个大 tile 如何由多条指令拼成"。
- ThrExpand / ValExpand 这两个旋钮直接决定线程数、寄存器压力、每线程指令数，是 GEMM 调性能的核心变量。
- 读懂 `partition_fragment_*` 返回的片段大小与 `gemm()` 的展开次数，才能理解"每个线程到底在算什么"。

## 核心概念

### MMA atom：最小计算单元

以 SM80 的 `SM80_16x8x8_F32F16F16F32_TN` 为例（源码 `mma_traits_sm80.hpp` 中 `ThrID = Layout<_32>`）：

- 指令形状 $(M,N,K)=(16,8,8)$，**32 线程（1 个 warp）共同执行一条**
- 一次算 $C[16\times8] = A[16\times8]\cdot B[8\times8]$，共 $16\times8\times8 = 1024$ 次乘加
- 一条指令里每个线程持有：A 4 个元素、B 2 个元素、C 4 个元素

### ThrExpand：线程层复制（放更多 warp）

`make_tiled_mma` 的第 2 个参数（ThrLayout），形状如 $(2,4,1)$：把整个 warp 在 M 方向摆 2 组、N 方向摆 4 组 → 线程数 $32 \times 2 \times 4 = 256$。K 方向不加线程——MMA 的线程本来就铺在 M×N 平面上，K 方向加线程没有意义。

### ValExpand：数值层复制（每个线程多干几遍）

第 3 个参数（ValLayout）直接给出大 tile 的形状，隐含每个线程在 M/N/K 方向各重复执行的次数。如 $(2,2,2)$：每条线程的指令在 M、N、K 方向各重复 2 遍。

### tile 形状公式

$$\text{TiledMMA tile} = \text{指令形状} \times \text{ThrExpand} \times \text{ValExpand}$$

## 工作原理

### 一个具体配置（cutlass-notes/15-tiled-gemm，均经实测验证）

```
指令 SM80_16x8x8_F32F16F16F32_TN：(16, 8, 8)，32 线程
ThrExpand (2, 4, 1)   → 线程 = 32 × 2 × 4 = 256
ValExpand (2, 2, 2)
TiledMMA tile = (16×2×2, 8×4×2, 8×1×2) = (64, 64, 16)
```

kernel 的 tile 必须等于 TiledMMA 的 tile（`partition_*` 按 TiledMMA 的 tile 为单位切分），代码里用 `static_assert` 兜底。

### 每个线程的片段（实测，不是朴素除法！）

`partition_fragment_*` 实测结果：

| 片段 | 大小 | 构成 |
|---|---|---|
| A | 16 个 fp16 | $2(M)\times2(K)\times4(atom)$ |
| B | 8 个 fp16 | $2(N)\times2(K)\times2(atom)$ |
| C | 16 个 fp32 | $2(M)\times2(N)\times4(atom)$ |

**关键认知**：片段大小 ≠ tile 元素数 ÷ 线程数。A/B 含"副本"——同一份 A 数据被 N 方向的 4 组线程各持一份（它们都要用它做各自的累加），同一份 B 被 M 方向的 2 组线程各持一份；只有 C 每个元素属于唯一线程。

### gemm() 不是一条指令，而是一个展开循环

一次 `gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)`，每个线程会串行发

$$\text{MMA\_M} \times \text{MMA\_N} \times \text{MMA\_K} = 2\times2\times2 = 8 \text{ 条 } mma.sync$$

即 ValExpand 的乘积。用 MAC 数交叉验证：一个 block 一次 gemm 算 $64\times64\times16 = 65536$ MACs，每条指令 1024 MACs → 64 条指令 ÷ 8 个 warp = 每条线程 8 条 ✓。

### 整个 kernel 的指令量

K 主循环把全矩阵 $K=64$ 切成 4 段 16，每次循环调一次 gemm()：每个线程共执行 $4 \times 8 = 32$ 条 `mma.sync`。

### local_tile + Step 投影：给三个张量切块

用同一个 3D tiler $(64,64,16)$ 分别给 A(B,C) 切块时，`Step<_1,X,_1>` 表示从 tiler 里挑出需要的两维：

```
gA = local_tile(mA, tiler, coord, Step<_1, X, _1>{})  // (64, 16, K/16)
gB = local_tile(mB, tiler, coord, Step< X, _1, _1>{})  // (64, 16, K/16)
gC = local_tile(mC, tiler, coord, Step<_1, _1, X>{})   // (64, 64)
```

结果的前两维是 tile 本身，最后一维是 K-tile 序号（K 主循环用 `gA(_, _, ktile)` 逐个取）。注意 `_` 放在 coord 里 = "剩余块全保留，做成新维度"；放具体数字 = 只取那一块。

## 我的理解

（以下为个人理解，非 cute 文档原文）

- 三个旋钮的分工：ThrExpand 决定"并发度"（warp 数），ValExpand 决定"单线程工作量"（指令重复次数），K 主循环决定"调用几次 gemm()"。三者都在增大总计算量，但占用的硬件资源形态完全不同——加 warp 吃 SM 上的并行槽位，加 ValExpand 吃寄存器，加 K 循环吃时间。
- 片段副本的方向恰好和复制方向相反：A 的副本数是 ThrExpandN，B 的副本数是 ThrExpandM，C 无副本。所以"A 片段 16、B 片段 8"不是 bug，而是布局使然。
- 与其背公式，不如会算：由 C 片段大小 ÷ 单条指令的 C 贡献（4）得到 M×N 方向指令数，K 方向的重数看 fragment 的 MMA_K 模式。

## 常见误区

1. **"tile 必须等于指令大小"** → 错。只需 tile 各维是指令各维的整数倍，倍数 = ThrExpand × ValExpand。
2. **"片段大小 = tile 元素数 ÷ 线程数"** → 错。A/B 有副本（见上），只有 C 满足这个朴素除法。
3. **"一次 gemm() 调用 = 一条指令"** → 错。它是按片段模式展开的多条指令序列。
4. **"kernel 参数 k（全矩阵 K）就是 tile 的 K"** → 错。全矩阵 K 被切成 K/kTileK 段：gA 中间维是 kTileK，最后一维才是段数。

## Related

- [CUTE 入门：从 C CUDA 视角理解 flash-attn 的 kernel 写法](./cute-basics.md) — Tensor / Layout / local_tile / 线程分区的前置知识
- [FlashAttention 源码精读](./flash-attention-source-reading.md) — 上层 kernel 中 TiledMMA 的实际用法
- [FlashAttention Kernel 与 Launch 机制](./flash-attention-kernel-and-launch.md) — tile、launch 与 split-KV

## References

- 实践代码：`/root/autodl-tmp/project/cutlass-notes/01-minimal-gemm`、`03-tiled-mma`、`15-tiled-gemm`
- CUTE 源码：`third_party/cutlass/include/cute/atom/mma_traits_sm80.hpp`（`ThrID = Layout<_32>`、ALayout/CLayout）
- `cute/tensor_impl.hpp` 的 `local_tile` / `inner_partition` 注释（Step 投影语义）
