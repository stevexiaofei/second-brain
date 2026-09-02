---
title: CUTLASS/CuTe 01：Tensor、Layout 与坐标映射
type: concept
status: seed
tags: [AI, CUDA, CUTE, CUTLASS, Tensor, Layout]
created: 2026-09-02
updated: 2026-09-02
source: third_party/cutlass/include/cute/ + CUTLASS 官方教程；知乎 CUTLASS 系列来源待逐篇核对
---

# CUTLASS/CuTe 01：Tensor、Layout 与坐标映射

## 一句话理解

CuTe 的 Tensor 可以看作“数据访问引擎 + 逻辑布局”：Engine 说明数据在哪里，Layout 说明逻辑坐标如何映射到物理 offset。CUTLASS 因此可以用同一套抽象表达 global memory、shared memory、register fragment，以及线程和数据的多层分工。

## 为什么重要

传统 CUDA kernel 常把形状、步长、线程索引和指针偏移混在一起。CuTe 将它们拆开后，可以单独检查：

- 逻辑张量有哪些坐标；
- 每个坐标对应哪个物理元素；
- 哪一层 memory 保存数据；
- 哪套 layout 被用于 copy、MMA 或边界判断。

后续的 `local_tile`、`partition_*`、`TiledCopy` 和 `TiledMMA` 都建立在这个模型上。

## Tensor = Engine + Layout

```text
Tensor<Engine, Layout>
  Engine：数据引擎，包含指针或寄存器存储方式
  Layout：逻辑坐标 → 物理 offset 的映射
```

一个二维行主序矩阵可以写成：

```cpp
float raw[6] = {0, 1, 2, 3, 4, 5};
Tensor T = make_tensor(raw, make_shape(2, 3), make_stride(3, 1));

// T(1, 2) = raw[1 * 3 + 2] = raw[5]
```

Shape `(2, 3)` 表示 2 行 3 列，Stride `(3, 1)` 表示行坐标增加一格时 offset 增加 3，列坐标增加一格时 offset 增加 1。`make_tensor` 创建的是视图，不会自动复制完整矩阵。

### `mA`、`gA`、`sA`、`rA`

在 GEMM 示例中，经常用前缀提示数据所处的阶段：

```text
mA：较完整的矩阵视图
 gA：CTA 负责的 global-memory A tile
 sA：shared-memory 中的 A tile
 rA：当前线程 register 中的 A fragment
```

这是一种代码阅读约定，不是 C++ 类型系统。`gA` 和 `sA` 可能有不同的 layout，因为 global memory 的合并访问、shared memory 的 bank-conflict 避免和 MMA 的寄存器契约通常并不相同。

## Layout：Shape 与 Stride

普通二维 Layout 可理解为：

$$
(i,j) \mapsto i \times stride_0 + j \times stride_1
$$

例如：

```text
(3, 4):(4, 1)
```

表示 3 行 4 列的行主序布局：第 `(i,j)` 个逻辑元素位于 offset `4i+j`。而：

```text
(3, 4):(1, 3)
```

是列主序布局：offset 为 `i+3j`。

```text
(3, 3):(4, 1)
```

则表示每个逻辑行有 3 个有效元素，但物理行跨度为 4；每行末尾留下一个 padding/gap。这种布局可用于对齐或为 shared-memory 访问降低 bank conflict。

### 嵌套 Layout

CuTe 的 Shape 和 Stride 可以是嵌套 tuple，用来表达 tile、warp、lane 或更复杂的硬件层次。

```text
(3,(2,2)):(4,(2,1))
```

坐标写成 `(m,(n0,n1))`，offset 为：

$$
4m + 2n_0 + n_1
$$

内层坐标按 `(n0,n1)` 展开时，一个 `m` 对应的 offset 顺序是 `0,2,1,3`；不同 `m` 之间以 4 为步长。

另一个例子：

```text
((2,2),(2,2)):((1,8),(4,2))
```

坐标为 `((r0,r1),(c0,c1))`，offset 为：

$$
 r_0 + 8r_1 + 4c_0 + 2c_1
$$

它可以表达“外层 tile 坐标 + tile 内坐标”或“线程层次 + value 层次”的组合。理解嵌套 Layout 的关键不是把 tuple 当成普通数组，而是逐层展开“坐标模式”和“步长模式”。

## Layout 不等于数据

```text
Layout：坐标如何走
Tensor：用某个 Engine 挂上这张坐标地图
copy：按照 source/destination layout 实际搬数据
```

所以 `local_tile` 通常只是从大 Tensor 投影出一个 tile 视图；`partition_*` 通常只是为当前线程生成局部视图；只有调用 `copy` 或实际 MMA 指令时，才发生相应的数据移动或计算。

## 小结

阅读任何 CuTe Tensor 时，先回答三件事：

1. Engine 是 global、shared 还是 register？
2. Shape/Stride 如何把逻辑坐标映射到 offset？
3. 这个 view 会被谁使用：copy、MMA、边界判断还是写回？

## 我的理解

CuTe 最核心的价值不是隐藏索引，而是把索引数学变成可组合、可检查的对象。学习 CUTLASS 的第一步不是背 API，而是能够在纸上把一个 layout 的坐标映射展开，并知道同一数据为什么需要多个不同 layout 的视图。

## Related

- [CUTLASS/CuTe 02：Copy Atom 与线程分区](./02-cute-copy-and-thread-partition.md)
- [CUTLASS/CuTe 03：TiledMMA 与 fragment](./03-cute-tiled-mma.md)
- [CUTLASS/CuTe 04：从 global → shared → MMA 串起 GEMM](./04-cute-gemm-pipeline.md)
- [CUTE 入门：FlashAttention kernel 中的 Tensor/Layout](../flash-attention/cute-basics.md)

## References

- `third_party/cutlass/include/cute/tensor.hpp`
- `third_party/cutlass/include/cute/layout.hpp`
- `third_party/cutlass/include/cute/tensor_impl.hpp`
- [CUTLASS GitHub](https://github.com/NVIDIA/cutlass)
- [知乎 CUTLASS 系列来源](https://zhuanlan.zhihu.com/p/1937220431728845963)（当前环境无法稳定读取，篇目待核对）
