---
title: CUTLASS / CuTe 专题
type: index
status: growing
tags: [AI, CUDA, CUTLASS, CUTE, GPU]
created: 2026-09-02
updated: 2026-09-02
source: CUTLASS 官方文档、third_party/cutlass 本地源码与知乎 CUTLASS 系列
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

1. [Tensor、Layout 与坐标映射](./01-cute-tensor-and-layout.md)：先理解 Engine、Shape、Stride 和嵌套 Layout；
2. [Copy Atom、TiledCopy 与线程分区](./02-cute-copy-and-thread-partition.md)：理解 global → shared 搬运和 `partition_S/D`；
3. [TiledMMA 与 fragment](./03-cute-tiled-mma.md)：理解 MMA atom 如何复制成大 tile；
4. [GEMM 数据流](./04-cute-gemm-pipeline.md)：把 copy、shared memory、MMA、accumulator 和写回串起来；
5. 回到 [FlashAttention 中的 CuTe 入门](../flash-attention/cute-basics.md)：用真实 attention kernel 验证这些抽象。

## 知识地图

| 层次 | 关键问题 | 对应笔记 |
|---|---|---|
| 数据表示 | Tensor 如何把 Engine 和 Layout 结合？ | [01](./01-cute-tensor-and-layout.md) |
| 坐标映射 | Shape/Stride 如何映射到 offset？ | [01](./01-cute-tensor-and-layout.md) |
| 数据搬运 | 谁从 global 读、谁向 shared 写？ | [02](./02-cute-copy-and-thread-partition.md) |
| 线程分工 | `get_thread_slice` 和 `partition_S/D` 做什么？ | [02](./02-cute-copy-and-thread-partition.md) |
| 矩阵计算 | 一条 MMA 如何扩展成大 tile？ | [03](./03-cute-tiled-mma.md) |
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

本专题以用户提供的 [知乎 CUTLASS 系列来源文章](https://zhuanlan.zhihu.com/p/1937220431728845963) 作为后续整理入口，同时用 CUTLASS 官方文档和本地源码校验概念。目前环境无法稳定读取该知乎页面，因而本轮不臆造原文的作者、篇目标题或文章顺序；已确认并沉淀的笔记先按知识依赖组织。后续获得可读页面或用户提供系列目录后，再将每篇来源映射到对应原子笔记，并补充文章级链接。

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
