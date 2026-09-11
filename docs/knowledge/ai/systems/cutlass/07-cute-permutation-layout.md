---
title: CuTe Permutation Layout：从"旧位置"到"新位置"的映射
type: concept
status: seed
tags: [AI, CUDA, CUTE, CUTLASS, Layout, Permutation, MMA]
created: 2026-09-11
updated: 2026-09-11
source: 知乎《CUTLASS 笔记 (3)：Tiled MMA》第 3.1 节（https://zhuanlan.zhihu.com/p/1950555644814946318）+ 个人整理
---

# CuTe Permutation Layout：从"旧位置"到"新位置"的映射

## 一句话理解

普通 Layout 描述的是 **坐标 → offset**；而 **Permutation Layout 描述的是 old_index → new_index 的位置重排**。它是 CUTLASS 四大重要 Layout 的最后一种，在 TiledMMA 中通过 `MMATileLayout` 使用，用来调整各个 MMA Atom 在一个 tile 内的**排列次序（Permutation）**。排列一变，数据落在哪个线程、哪个地址就变，访存模式也就跟着变。

## 为什么重要

- 它是**唯一一种不改变数据内容、只改变"谁在哪儿"的 Layout**，因此是纯粹的布局控制手段。
- 在 TiledMMA 里，`ThrExpand` 决定"一个 Atom 由哪些线程算"，而 `MMATileLayout` 决定"Atom 之间怎么摆"。后者直接决定访存模式 —— 这正是 Tiled Copy 效率问题的根源。
- 它解释了为什么 CUTLASS 能在**不改 kernel 主体代码**的前提下，通过改一个模板参数来切换数据布局。

## 与普通 Layout 的区别

```text
普通 Layout       ： 坐标 (i, j, k)  →  物理 offset
Permutation Layout： 旧位置 old_index →  新位置 new_index
```

读法完全一致（都是 `shape:stride`），区别只在**"输入输出分别代表什么"**：普通 Layout 输出的是内存偏移，Permutation Layout 输出的仍是"位置"。

## 具体例子：`(4,4,2):(1,8,4)`

原文给出的官方示例，Shape `(4,4,2)`、Stride `(1,8,4)`，共 32 个位置：

```text
old m-coord:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
new m-coord:  0  1  2  3  8  9 10 11 16 17 18 19 24 25 26 27  4  5  6  7 12 13 14 15 20 21 22 23 28 29 30 31
```

**计算方式**（左端最快展开，与普通 Layout 一致）：把 old_index $p$ 按 Shape 拆成坐标 $(i_0,i_1,i_2)$，其中

$$
i_0 = p \bmod 4,\qquad i_1 = \left\lfloor \frac{p}{4}\right\rfloor \bmod 4,\qquad i_2 = \left\lfloor \frac{p}{16}\right\rfloor
$$

再用 Stride 组合得到新位置：

$$
\text{new\_index} = i_0 \cdot 1 + i_1 \cdot 8 + i_2 \cdot 4
$$

验算几个点：

| old_index $p$ | $(i_0,i_1,i_2)$ | new_index | 上表核对 |
|---|---|---|---|
| 1 | (1,0,0) | 1 | ✔ |
| 4 | (0,1,0) | 8 | ✔ |
| 5 | (1,1,0) | 9 | ✔ |
| 8 | (0,2,0) | 16 | ✔ |
| 16 | (0,0,1) | 4 | ✔ |
| 17 | (1,0,1) | 5 | ✔ |

可以看出 Stride 的语义是："第 $i_0$ 个 Atom 紧挨着放（stride 1），第 $i_1$ 个 Atom 间隔 8 个位置，$i_2$ 组间隔 4 个位置"。也就是把原来 `(4,4,2)` 的紧凑排列，按 stride 重新铺开。

## 在 TiledMMA 中的用法

`make_tiled_mma` 的第 3 个参数 `MMATileLayout` 既给出 tile 总规模，也承载排列信息：

```cpp
using MMATileLayout = Tile<Int<kMmaTileM>, Int<kMmaTileN>, Int<kMmaTileK>>;
//                   Tile<32,           32,           16>
using TiledMMA = decltype(make_tiled_mma(MMA_op{}, MMAThrLayout{}, MMATileLayout{}));
```

`MMATileLayout` 是长度为 3 的 tuple，**分别对应 M / N / K 三个维度的排列，每个维度用一个 Layout 表示**。调整某个维度的 Layout，就改变了各个 MMA Atom 在该维度上的排列次序 —— 这就是把 Permutation Layout 用在了 Atom 的排列上。

一个非常实用的性质：**改完 `MMATileLayout` 后，`copy` 和 `gemm` API 一行都不用动**。这两个 API 会根据 TiledMMA 的扩展情况，自动处理 MMA Atom 的循环计算。

## 我的理解

（以下为个人理解，非原文结论）

- 把 Permutation Layout 理解成**"重排索引的置换表"**最直观：它不产生新数据，只是把一份已有数据换个位置坐。因此它天然适合做"同一份计算、换一种访存顺序"的优化旋钮。
- 它与 TiledCopy 的关系是**因果关系**：Atom 的排列决定了每个线程持有哪些数据，而线程持有哪些数据决定了 gmem → smem → rmem 的搬运是否连续、能否向量化、会不会 bank conflict。所以"改排列"表面上是布局问题，实质上是**性能问题**。
- `Tile<32,32,16>` 这种写法是"紧凑（identity）排列"的简写；一旦需要非紧凑排列，就要显式给出各维度的 Layout。这也解释了为什么 `MMATileLayout` 在文档里被描述成"tuple of Layout"而不是"tuple of size"。

## 常见误区

1. **"Permutation Layout 也是坐标 → offset"** → 语义不同：它输出的是新位置，而不是内存偏移。
2. **"改排列只是好看，不影响性能"** → 错。排列直接决定访存模式，是本系列后续 Tiled Copy 优化的核心杠杆。
3. **"改了 `MMATileLayout` 就要重写 copy / gemm"** → 错。这两个 API 会自动适配 TiledMMA 的扩展与排列。
4. **"Permutation Layout 只在 MMA 里用"** → 它是通用 Layout 类型，TiledMMA 只是其中一个典型用法。

## Related

- [CUTLASS/CuTe 01：Tensor、Layout 与坐标映射](./01-cute-tensor-and-layout.md) — 普通 Layout 与嵌套 Layout 的基础
- [CUTLASS/CuTe 03：TiledMMA 与 fragment](./03-cute-tiled-mma.md) — Permutation Layout 在 TiledMMA 中的使用上下文
- [CUTLASS/CuTe 06：GEMM 三级 Tiling](./06-gemm-three-level-tiling.md) — 排列发生的那一层
- [CUTLASS/CuTe 02：Copy Atom 与线程分区](./02-cute-copy-and-thread-partition.md) — 排列影响的下一环

## References

- 知乎《CUTLASS 笔记 (3)：Tiled MMA》第 3.1 节"make_tiled_mma API"，作者杨远航，CUTLASS 4.1.0：<https://zhuanlan.zhihu.com/p/1950555644814946318>
- CuTe 官方 Layout 文档：`third_party/cutlass/media/docs/cpp/cute/0x_layout.md`
