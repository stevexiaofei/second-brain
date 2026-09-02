---
title: CUTLASS/CuTe 04：从 Global 到 Shared 再到 MMA 的 GEMM 数据流
type: concept
status: seed
tags: [AI, CUDA, CUTE, CUTLASS, GEMM, Pipeline]
created: 2026-09-02
updated: 2026-09-02
source: CUTLASS 官方 CuTe GEMM 教程与本地实现；知乎 CUTLASS 系列来源待逐篇核对
---

# CUTLASS/CuTe 04：从 Global 到 Shared 再到 MMA 的 GEMM 数据流

## 一句话理解

一个 CuTe GEMM kernel 通常不是“global memory 直接乘 register”，而是沿着 `global → shared → register/MMA → accumulator → global` 的路径流动；不同阶段使用不同 Tensor view 和不同 partition pattern，但它们通过 layout 保持逻辑元素的对应关系。

## 全流程地图

```text
mA / mB / mC                         更完整的矩阵视图
      │ local_tile
      ▼
gA / gB / gC                         CTA 负责的 global tile
      │ tA/tB copy partition
      ▼
tAgA / tBgB                         当前线程负责读取的 source fragment
      │ copy
      ▼
tAsA / tBsB                         当前线程负责写入 shared tile 的 destination fragment
      ▼
sA / sB                              shared-memory staging tile
      │ tC MMA partition
      ▼
tCsA / tCsB                         MMA 的 A/B operand fragment
      │ register fragment
      ▼
tCrA / tCrB / tCrC                  线程私有寄存器片段与 accumulator
      │ gemm / mma.sync
      ▼
C/D accumulator                      输出 tile 的累加结果
      │ epilogue + store partition
      ▼
gC                                    写回 global memory
```

箭头不一定代表每一步都创建新物理存储；`local_tile` 和 `partition_*` 很多时候只是视图，`copy` 和 MMA 指令才是实际搬运或计算。

## 两套 partition：copy 与 MMA

### Copy 阶段

```cpp
Tensor tAgA = thr_copy_a.partition_S(gA);
Tensor tAsA = thr_copy_a.partition_D(sA);
Tensor tArA = make_fragment_like(tAsA);

copy(thr_copy_a, tAgA, tAsA);
```

这里的 `tA` 是 copy partition pattern：它回答“当前线程从 global A 的哪些槽位读取，以及向 shared A 的哪些槽位写入”。`S` / `D` 分别表示 source / destination。

### MMA 阶段

```cpp
Tensor tCsA = thr_mma.partition_A(sA);
Tensor tCsB = thr_mma.partition_B(sB);
Tensor tCgC = thr_mma.partition_C(gC);
Tensor tCrC = thr_mma.make_fragment_C(tCgC);
```

这里的 `tC` 是围绕 MMA 的分区模式：它回答“当前线程在 MMA atom/tiled MMA 中拥有 A、B 的哪些 fragment，以及 C/D accumulator 的哪些寄存器”。它不是说“数据属于 C 矩阵”。

同一个 `sA` 同时拥有 `tAsA` 和 `tCsA` 两种视图是正常的：

```text
 tAsA：为了搬运，服从 copy layout
 tCsA：为了计算，服从 MMA operand layout
```

## 一个最小的结构化伪代码

```cpp
// 1. 创建全局矩阵和 CTA tile
Tensor mA = ...;
Tensor mB = ...;
Tensor mC = ...;
Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

// 2. 线程协作搬 global → shared
auto thr_copy_a = tiled_copy_a.get_thread_slice(threadIdx.x);
Tensor tAgA = thr_copy_a.partition_S(gA);
Tensor tAsA = thr_copy_a.partition_D(sA);
copy(tiled_copy_a, tAgA, tAsA);

// 3. 让 MMA 线程看到 shared tile 的 operand fragment
auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
Tensor tCsA = thr_mma.partition_A(sA);
Tensor tCsB = thr_mma.partition_B(sB);
Tensor tCrC = thr_mma.make_fragment_C(gC);

// 4. 对 K tile 循环执行 MMA
gemm(tiled_mma, tCrC, tCsA, tCsB, tCrC);

// 5. epilogue 后按输出 partition 写回 global
Tensor tCgC = thr_mma.partition_C(gC);
copy(tiled_copy_c, tCrC, tCgC);
```

真实代码还会加入双缓冲、异步 copy、同步、K 主循环、尾部谓词和 epilogue；这个伪代码只保留职责边界。

## 为什么要使用多套 layout

同一矩阵在不同硬件阶段的最优访问方式不同：

| 阶段 | 典型目标 |
|---|---|
| global load | 连续、合并、足够宽的向量化访问 |
| shared staging | 避免 bank conflict，并适配 `ldmatrix` 或 MMA 读取 |
| register fragment | 匹配 MMA 指令规定的 lane/value 映射 |
| output store | 让写回地址连续、处理尾行尾列 |

因此“逻辑上都是 A”不意味着“所有 Tensor view 都有同一个 Layout”。CuTe 的组合方式让这些物理布局可以不同，同时让开发者明确它们代表同一个算法对象。

## `tCgC` 与 `tCrC` 的区别

```cpp
Tensor tCgC = thr_mma.partition_C(gC);
Tensor tCrC = thr_mma.make_fragment_C(tCgC);
```

- `tCgC`：按照 MMA 的 C partition，当前线程对应的 global C/output 位置；
- `tCrC`：按照同一 C fragment 形状创建的寄存器 accumulator；
- 前者描述“最终写到哪里”，后者保存“当前算出了什么”。

MMA 只更新 `tCrC`，不会自动把结果写入 global memory；写回需要显式的 epilogue/store 阶段。

## 与 FlashAttention 的对应关系

FlashAttention 的 Q/K/V kernel 不是标准单次 GEMM，但其内存层次和线程视图仍遵循同一思想：

```text
gQ/gK/gV
  → tQgQ/tKgK/tVgV
  → tQsQ/tKsK/tVsV
  → sQ/sK/sV
  → tSrQ/tSrK/tOrVt
  → QKᵀ 与 PV 的 MMA
  → acc_o
  → O 写回
```

这也是为什么单独学习 CUTLASS/CuTe 的 Tensor、Copy 和 MMA，再回到 FlashAttention，会比直接从 `flash_fwd_kernel.h` 的模板类型开始更容易。

## 常见误区

1. **`gA → sA` 的每个箭头都等于一次复制**：视图构造不等于数据移动。
2. **`tC` 代表 C 矩阵存储位置**：通常它表示 MMA partition family。
3. **MMA 会自动写回 C**：累加结果通常先在 register fragment 中，写回是单独阶段。
4. **copy layout 和 MMA layout 必须相同**：它们面向不同硬件契约，通常不同。
5. **只看变量名就能确定语义**：必须追踪来源 Tensor、构造 API 和后续操作。

## 我的理解

CuTe GEMM 的主线可以归纳为一句话：**先用 copy layout 把数据以适合内存层次的方式搬到位，再用 MMA layout 把同一逻辑 tile 重新解释成硬件所需的 fragment。** 这两个阶段不是重复设计，而是分别解决带宽和计算映射问题。

## Related

- [CUTLASS/CuTe 01：Tensor、Layout 与坐标映射](./01-cute-tensor-and-layout.md)
- [CUTLASS/CuTe 02：Copy Atom 与线程分区](./02-cute-copy-and-thread-partition.md)
- [CUTLASS/CuTe 03：TiledMMA 与 fragment](./03-cute-tiled-mma.md)
- [FlashAttention 源码精读](../flash-attention/flash-attention-source-reading.md)

## References

- `third_party/cutlass/media/docs/cpp/cute/0x_gemm_tutorial.md`
- `third_party/cutlass/include/cute/atom/copy_atom.hpp`
- `third_party/cutlass/include/cute/atom/mma_traits_sm80.hpp`
- [CUTLASS GitHub](https://github.com/NVIDIA/cutlass)
- [知乎 CUTLASS 系列来源](https://zhuanlan.zhihu.com/p/1937220431728845963)（当前环境无法稳定读取，篇目待核对）
