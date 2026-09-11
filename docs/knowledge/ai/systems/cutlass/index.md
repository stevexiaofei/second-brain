---
title: CUTLASS / CuTe 专题
type: index
status: growing
tags: [AI, CUDA, CUTLASS, CUTE, GPU]
created: 2026-09-02
updated: 2026-09-11
source: CUTLASS 官方文档、third_party/cutlass 本地源码与知乎 CUTLASS 系列（已可读取，见文末映射）
---

# CUTLASS / CuTe 专题

## 一句话理解

CUTLASS 是 NVIDIA 面向高性能矩阵运算的 CUDA 模板库；CuTe 是其中负责描述 Tensor、Layout、线程分工、Copy 和 MMA 组合关系的核心抽象。学习这套体系的目标，不是背模板，而是能够从逻辑 tile 追踪到线程、内存和硬件指令。

## 为什么单独建立这个专题

CuTe 知识会在 FlashAttention、GEMM、卷积和其他 GPU kernel 中反复出现，但应用专题通常只解释“这里为什么这样写”。独立的 CUTLASS 目录负责沉淀可迁移的基础：

```text
Tensor / Layout
      ↓
Copy Atom / TiledCopy / thread partition
      ↓
MMA Atom / TiledMMA / fragment
      ↓
GEMM pipeline
      ↓
FlashAttention 等真实 kernel
```

[FlashAttention 专题](../flash-attention/) 保留算法、kernel launch、split-KV 和 PyTorch 接入等应用内容；这里则集中解释 CUTLASS/CuTe 的通用编程模型。

## 推荐阅读顺序

> 文件编号是**创建顺序**，下面是**学习顺序**。核心原则：先建立"为什么要分层"的模型，再看每一层怎么做。

1. **硬件前置** → [GPU 全局内存访存模型：向量化与合并访存](../gpu-memory-access-model.md)：先搞清访存效率由什么决定（sector / transaction），后面所有布局取舍才有物理依据；
2. [06 GEMM 三级 Tiling](./06-gemm-three-level-tiling.md)：理解 Global → Block → Tile → MMA Atom 为什么要分三级、每级受什么硬件约束（配合[算子优化方法论](../gpu-kernel-optimization-methodology.md)）；
3. [01 Tensor、Layout 与坐标映射](./01-cute-tensor-and-layout.md)：理解 Engine、Shape、Stride 和嵌套 Layout；
4. [02 Copy Atom、TiledCopy 与线程分区](./02-cute-copy-and-thread-partition.md)：理解 `make_tiled_copy`、`Copy_Traits` 三 Layout、`retile_S/D` 与 metadata；
5. [09 TiledCopy 核心原理](./09-cute-tiled-copy-principle.md)：理解 `partition_S/D` 背后的 $S$/$D$/$R$ 复合映射；
6. [03 TiledMMA 与 fragment](./03-cute-tiled-mma.md)：理解 MMA atom 如何复制成大 tile（第三级的具体实现）；
7. [08 MMA 指令语义与累加方向](./08-mma-instruction-and-accumulation.md)：理解展开后的每一条 mma 到底在算什么、为什么 K 是串行链；
8. [07 Permutation Layout](./07-cute-permutation-layout.md)：理解同一 tile 内 Atom 的排列如何被重排；
9. [04 GEMM 数据流](./04-cute-gemm-pipeline.md)：把 copy、shared memory、MMA、accumulator 和写回串起来；
10. [05 Copy 规模核算与 128-bit 向量化](./05-cute-copy-scaling-and-vectorization.md)：把 thread/value layout 算到 warp/CTA 吞吐，并核对向量化约束；
11. 回到 [FlashAttention 中的 CuTe 入门](../flash-attention/cute-basics.md)：用真实 attention kernel 验证这些抽象。

## 知识地图

| 层次 | 关键问题 | 对应笔记 |
|---|---|---|
| 优化动机 | 算子优化有哪三个维度？瓶颈通常在哪？ | [GPU 算子优化方法论](../gpu-kernel-optimization-methodology.md) |
| 访存硬件 | 向量化与合并访存如何决定实际搬运的字节数？ | [GPU 访存模型](../gpu-memory-access-model.md) |
| 分层动机 | 为什么 GEMM 要三级 Tiling？每级对应哪条硬件特性？ | [06](./06-gemm-three-level-tiling.md) |
| 数据表示 | Tensor 如何把 Engine 和 Layout 结合？ | [01](./01-cute-tensor-and-layout.md) |
| 坐标映射 | Shape/Stride 如何映射到 offset？ | [01](./01-cute-tensor-and-layout.md) |
| 位置重排 | 如何在保留数据的前提下改变位置次序？ | [07](./07-cute-permutation-layout.md) |
| 数据搬运 | 谁从 global 读、谁向 shared 写？ | [02](./02-cute-copy-and-thread-partition.md) |
| 拷贝原理 | `partition_S/D` 背后的复合映射是什么？ | [09](./09-cute-tiled-copy-principle.md) |
| 线程分工 | `get_thread_slice` 和 `partition_S/D` 做什么？ | [02](./02-cute-copy-and-thread-partition.md) |
| 规模与约束 | tile 槽位如何换算成 warp/CTA 吞吐？128-bit 向量化要求什么？ | [05](./05-cute-copy-scaling-and-vectorization.md) |
| 矩阵计算 | 一条 MMA 如何扩展成大 tile？ | [03](./03-cute-tiled-mma.md) |
| 指令语义 | 一条 mma 内部算什么？累加器归谁？K 为什么不能拆线程？ | [08](./08-mma-instruction-and-accumulation.md) |
| 完整流水线 | copy → shared → fragment → MMA → store 如何连接？ | [04](./04-cute-gemm-pipeline.md) |
| 真实应用 | Q/K/V tile 如何使用这些视图？ | [FlashAttention](../flash-attention/cute-basics.md) |

## 一组必须掌握的命名

```text
mA / gA / sA / rA       矩阵 A 在不同视图或存储层次的表示
tAgA / tAsA             copy partition 应用于 global/shared A
tCsA / tCsB             MMA partition 应用于 shared A/B
tCgC / tCrC             MMA C partition 的 global 目标 / register accumulator

tQgQ / tQsQ             FlashAttention 中 Q 的 global source / shared destination
```

这些是代码阅读约定，不是 C++ 语法。应结合 Tensor 来源和 `partition_*` API 解释变量名。

## 与知乎系列的关系

本专题以 [知乎 CUTLASS 系列](https://zhuanlan.zhihu.com/p/1937220431728845963)（作者杨远航）作为整理入口，同时用 CUTLASS 官方文档和本地源码校验概念。该系列页面现已可读取（2026-09-11 起），因此可以建立文章 → 原子笔记的映射：

| 原文章节 | 已沉淀笔记 |
|---|---|
| 笔记 (1)：Minimal GEMM Kernel | [01](./01-cute-tensor-and-layout.md)、[03](./03-cute-tiled-mma.md) |
| 笔记 (2)：混合精度 GEMM Kernel | [03](./03-cute-tiled-mma.md)（混合精度部分） |
| 笔记 (3) §1 算子优化方法论 | [GPU 算子优化方法论：计算、通信、存储](../gpu-kernel-optimization-methodology.md) |
| 笔记 (3) §2 GEMM 三级 Tiling | [06 GEMM 三级 Tiling](./06-gemm-three-level-tiling.md) |
| 笔记 (3) §3.1 `make_tiled_mma` API | [03 TiledMMA 与 fragment](./03-cute-tiled-mma.md) |
| 笔记 (3) §3.1 Permutation Layout | [07 Permutation Layout](./07-cute-permutation-layout.md) |
| 笔记 (3) §3.2 Tensor Metadata 打印格式 | [03 TiledMMA 与 fragment](./03-cute-tiled-mma.md) |
| 笔记 (3) §3.3 SASS 分析（mma 语义与累加） | [03](./03-cute-tiled-mma.md)、[08 MMA 指令语义与累加方向](./08-mma-instruction-and-accumulation.md) |
| 笔记 (4) §1 NV GPU 的全局内存访存特性 | [GPU 全局内存访存模型：向量化与合并访存](../gpu-memory-access-model.md) |
| 笔记 (4) §2 TiledCopy 的核心原理 | [09 TiledCopy 核心原理](./09-cute-tiled-copy-principle.md) |
| 笔记 (4) §3 Tiled Copy 实现（Copy_Traits / ThrCopy / make_tiled_copy） | [02](./02-cute-copy-and-thread-partition.md) |
| 笔记 (4) §3.5 metadata 解析与 LaTeX 图解 | [02](./02-cute-copy-and-thread-partition.md)、[05](./05-cute-copy-scaling-and-vectorization.md) |

原文章节链接：

- 导读：<https://zhuanlan.zhihu.com/p/1937220431728845963>
- 笔记 (1)：<https://zhuanlan.zhihu.com/p/1937517614084650073>
- 笔记 (2)：<https://zhuanlan.zhihu.com/p/1940158874255602181>
- 笔记 (3)：<https://zhuanlan.zhihu.com/p/1950555644814946318>
- 笔记 (4)：<https://zhuanlan.zhihu.com/p/1968745447741972494>
- 笔记 (5)：<https://zhuanlan.zhihu.com/p/1970162570636816559>

整理原则：

- 不整篇复制原文；只记录自己的理解、必要的公式和最小代码；
- 原文观点、官方事实和个人推断分开标注；
- 每新增一个主要概念，优先新增一个可独立复习的 Markdown 文件；
- 发现已有笔记覆盖同一概念时，更新链接或重组归属，不重复粘贴。

## 实践主线

建议用一个小型 FP16/BF16 GEMM 做实验：

```text
1. 写 naive global-memory CUDA GEMM，建立正确性基线
2. 用 local_tile 表达 CTA tile
3. 用 TiledCopy 搬 global → shared
4. 用 TiledMMA / mma.sync 计算 fragment
5. 用 accumulator 和 epilogue 写回
6. 改变 tile、warp、copy vector width，观察寄存器与性能
7. 对照 FlashAttention 的 QKᵀ/PV 两个 MMA 阶段
```

性能结论应通过编译、运行和 profiler 验证；不能仅凭变量名或模板形状推断性能。

## Related

- [FlashAttention 专题](../flash-attention/) — CuTe 在 attention kernel 中的应用
- [CUDA 初学者学习路径](../../../../inbox/cuda-beginner-learning-path.md) — CUDA、Tensor Core 和 CUTLASS 的前置路线
- [CUDA 硬件与编程模型地图](../../../../inbox/cuda-hardware-and-programming-model-map.md) — warp、shared memory、Tensor Core 等背景
- [AI Systems](../)

## References

- [CUTLASS 官方 GitHub](https://github.com/NVIDIA/cutlass)
- [CUTLASS 官方文档](https://github.com/NVIDIA/cutlass/tree/main/media/docs)
- [知乎 CUTLASS 系列来源](https://zhuanlan.zhihu.com/p/1937220431728845963)
