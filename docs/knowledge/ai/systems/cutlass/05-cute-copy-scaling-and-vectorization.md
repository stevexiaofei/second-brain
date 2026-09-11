---
title: CUTLASS/CuTe 05：TiledCopy 规模核算与 128-bit 向量化约束
type: concept
status: seed
tags: [AI, CUDA, CUTE, CUTLASS, Copy, TiledCopy, Vectorization]
created: 2026-09-07
updated: 2026-09-07
source: DeepSeek 分享对话（2026-09-07 抓取）https://chat.deepseek.com/share/etkkl7tbliko5g3a2o ；公式与语义与 02 号笔记交叉核对
---

# CUTLASS/CuTe 05：TiledCopy 规模核算与 128-bit 向量化约束

## 一句话理解

一次 gmem→smem 搬运其实是在做三笔可数的账：**单线程一次拿几个值**（value layout）、**多少个线程一起搬**（thread layout，二者乘积 = 一个 tile 的槽位数）、以及**每线程在连续地址上拿的字节够不够凑一条 128-bit 指令**（向量化）。`TiledCopy` 只负责"一个 tile"之内的事；"这个 tile 要重复几次才铺满整张矩阵"由 kernel 的 CTA/tile 循环负责，不写在布局里。

## 这条对话的来龙去脉

在 DeepSeek 上从 `Cutelass tiled copy 是什么` 一路追问到 `thread layout / value layout 与 128-bit 的约束`，问题链刚好覆盖 02 号笔记公式的一整套配套练习：

1. `TiledCopy` 是什么 → Copy Atom + thread layout + value layout 的组装（公式已在 [02](./02-cute-copy-and-thread-partition.md)）
2. 要一个详细例子 → 用 128×128 fp32 全局→共享搬运来建立数量感（注意：这是教学数字，不是真实 GEMM 写法）
3. "那每个线程要跑两次？" → 暴露了把"布局内的每线程值数"和"外层 tile 重复次数"混为一谈
4. "一个 warp 是不是能搬 16×32 个 float？" → 把每线程值数换算成 warp 吞吐
5. "一个 CTA 有 256 线程呢？" → thread layout 必须覆盖实际 launch 的线程，否则大量 warp 空转
6. "想提高效率，layout 和 128-bit 有什么约束？" → 向量化的硬条件

沉淀价值：02 给公式与语义，05 记下配套的**数量级换算**和最容易错的两个点，便于独立复习。

## 账本一：一个 TiledCopy = 一个 tile 的分工表

`make_tiled_copy(atom, thr_layout, val_layout)` 合成的 `TiledCopy`，其"覆盖表"是一个 `(thr, val) -> (M, N)` 的一一对应（02 已按源码解释），因此槽位守恒：

$$
M_{\text{tile}} \times N_{\text{tile}} = \text{size(thr_layout)} \times \text{size(val_layout)}
$$

当 thread/value layout 都能沿 $M,N$ 分离时（教学例最常用），还能按维展开：

$$
m_{\text{tile}} = \text{Thr}_M \times \text{Val}_M, \qquad n_{\text{tile}} = \text{Thr}_N \times \text{Val}_N
$$

这条对话里的例子：thread layout $4\times 8=32$、value layout $4\times 4=16$，则单 tile = $32\times 16 = 512$ 个元素，展开成 2D 是 $16\times 32$（$m=4\times4$，$n=8\times4$）。**这里已经能回答"每线程拿 16 个"**——它来自 `size(val_layout)`，是布局内常量。

## 账本二：把每线程值数换算成 warp / CTA 吞吐

硬件以 warp（32 线程）为调度与执行粒度，所以"每线程 16 个 fp32"换算成物理吞吐：

- 一个 warp 每"趟"搬运：$32 \times 16 = 512$ 个 float = **2 KB**
- 一个 CTA 有 256 线程（8 warps）时，把 thread layout 扩到 $16\times16=256$：单 tile 变 $64\times64=4096$ 个 float = **16 KB**，一次 `copy` 整个 CTA 满载参与

不同配置对比（fp32、value layout 固定 $4\times4$ 的教学场景）：

| thread layout | 线程数 | 单 tile | 每 warp 每趟 | 256 线程 CTA 利用率 |
|---|---|---|---|---|
| $4\times8$ | 32 | $16\times32=512$ (2 KB) | 2 KB | 12.5%（只 1 个 warp 在搬） |
| $16\times16$ | 256 | $64\times64=4096$ (16 KB) | 2 KB | 100%（8 warps 满载） |

由此得到一条硬性设计规则：**`thr_layout` 的总线程数应覆盖（通常等于）实际 launch 的 CTA 线程数**。布局只覆盖 32 线程而 CTA 有 256 线程，意味着 7/8 的线程不参与这次搬运——这在真实 kernel 里是不可接受的空转。02 讨论的"寄存器预算"限制的是单线程值数，05 这条限制的是**参与线程数**，两者是互补的两个约束。

## 账本三：重复次数归 kernel 管，不归 TiledCopy 管

对话里"每个线程要跑几次"的正确答案是：`TiledCopy` 描述的是**一个 tile 内部**线程与值的分配，它不包含"这张矩阵要切几块、每块搬几次"的信息。次数来自 kernel 结构：

- 本 CTA 负责的全局 tile（如 GEMM 中 CTA 的 $M\times N$ 输出块）
- K 方向的主循环（每轮搬一段 A/B）
- CTA 网格如何在整张矩阵上铺开

典型写法（与 [04](./04-cute-gemm-pipeline.md) 对应）：用与 CTA 线程数匹配的 `TiledCopy`，每次只搬"当前 K 段"的那一片，K 由外层循环推进；而不是把整张 128×128 矩阵塞进一次 copy。因此"每线程跑 $128/16 \times 128/32 = 32$ 次"这类数字只是用一个小布局去覆盖大矩阵时的算术推演，真实 kernel 会换更大的 thread layout 让单次覆盖等于本阶段的 tile，次数由外层循环承担。

## 128-bit 向量化的硬条件

> 硬件层面的完整阐释（sector / transaction / 合并访存）见 [GPU 全局内存访存模型：向量化与合并访存](../gpu-memory-access-model.md)。本节的表只是它在 TiledCopy 布局参数上的落地形式。

### 先看指令

向量化的上限与对齐来自 `Copy_Atom` 对应的硬件指令：`cp.async` 支持 4/8/16 字节；`AutoVectorizingCopyWithAssumedAlignment<128>` 表示"按 128-bit（16 字节）对齐/向量化"的假设（不是 128 线程）。要真正合成 128-bit 访问，必须满足：

1. 每线程在**物理连续**方向上一次性覆盖 16 字节；
2. 源与目标基地址按 16 字节对齐（stride 也要随之满足）。

### 再看 value layout

关键约束落在 `val_layout` 的**最快 mode（内存连续方向）**上：它的长度 × 元素字节数必须 ≥ 16B 才有向量化前提：

| 元素类型 | 字节/元素 | 128-bit 需要的连续元素数 | 行主序时 value layout 最快维 |
|---|---|---|---|
| fp32 | 4 | 4 | $(1,4)$ |
| fp16 / bf16 | 2 | 8 | $(1,8)$ |
| fp64 | 8 | 2 | $(1,2)$ |

列主序则把连续维放到 M 方向，对应 $(4,1)$ / $(8,1)$ 形式。对话结论"fp16 连续维要 8、fp32 要 4"与上面的表一致——注意这是对**最快维**的要求，不是 value layout 的整体形状。

### thread layout 的配合

thread layout 决定 warp 内 32 个线程的地址推进方式：想要 gmem 合并访问与 smem 低 bank 冲突，通常让相邻线程落在相邻的 16B 段上，再由 value layout 在段内做连续向量读。可自洽的检查是同时满足三件事：

- $\text{size(val_layout)}$ = 每线程元素数；
- $\text{size(thr_layout)} \times \text{size(val_layout)}$ = 单 tile 槽位数；
- 每线程连续段字节数能整除 16B，且段起始地址按 16B 对齐。

## 对话里踩过的两个易混点

1. **"每个线程要跑两次？"** 原例 thread layout N 方向 8 个线程、value layout N 方向 4，$8/4=2$ 只是"N 方向上线程槽位与值槽位的比例"，不是单个线程的执行次数。把"每线程 16 个值"误当成"每线程要循环几次"是概念错位：16 是布局内每线程的静态值数，重复次数属于账本三的外层结构。
2. **"一行 `cute::copy` 没有 for 循环，是不是只搬一次？"** 结论取决于分区视图里是否带着额外的重复 mode/外层 tile。只对"当前 CTA tile 的当前 K 段"调用时，一行 copy 就是把该段一次搬完；若把一个小 `TiledCopy` 直接套到更大的矩阵视图上，视图里就会带上多次重复，copy 的展开方式见"待核对"。

## 我的理解

`TiledCopy` 布局设计是一个**三约束预算题**：thread layout 定"多少个线程"、value layout 定"每线程静态拿几个值（以及拿得连不连续）"、两者乘积必须等于目标 tile 的槽位数；落到硬件上还要再检查"连续段够不够 16B（向量化）"和"thread 总数是否覆盖 CTA（空转）"。而最容易错的地方是记住：这套预算**只算一个 tile**，矩阵更大时"铺几遍"由 kernel 的循环和 CTA 划分承担——把这两个层面分开，规模换算就不会再被"两次/32 次"这类问题绕进去。

## 待核对（对话中的表述，需与源码/实测确认后采信）

- 对话称"每线程 16 个 fp32 ≈ 4 条 16B 拷贝指令"：按 $64\,\mathrm{B}/16\,\mathrm{B}=4$ 数学自洽，但实际条数由 `Copy_Atom` 的具体向量宽度决定，不是固定值。
- 对话称"2KB/warp 粒度天然无 bank 冲突"：这是特定 thread/value 排布下的结论，不是一般规律。
- `cute::copy` 对含重复 mode 的分区视图如何展开（编译期递归 vs 需要显式循环）需对照 `cute/algorithm/copy.hpp` 与 partition 返回形状确认。
- 128-bit 向量化在 smem 一侧还受分配 stride 与对齐影响；失败点可能是编译期 static assert，也可能需要运行时对齐保证。

## Related

- [CUTLASS/CuTe 01：Tensor、Layout 与坐标映射](./01-cute-tensor-and-layout.md)
- [CUTLASS/CuTe 02：Copy Atom、TiledCopy 与线程分区](./02-cute-copy-and-thread-partition.md) — 本笔记的公式与 `make_tiled_copy` 语义来源
- [CUTLASS/CuTe 04：从 Global 到 Shared 再到 MMA 的 GEMM 数据流](./04-cute-gemm-pipeline.md)
- [CUTLASS/CuTe 09：TiledCopy 核心原理](./09-cute-tiled-copy-principle.md) — 向量化与合并访存如何落到 `partition_S/D` 的复合映射上
- [GPU 全局内存访存模型：向量化与合并访存](../gpu-memory-access-model.md) — 128-bit 向量化与 sector 模型的硬件依据
- [CUTE 入门：FlashAttention 中的 GmemTiledCopyO 与边界处理](../flash-attention/cute-basics.md)

## References

- DeepSeek 分享对话《Cutelass tiled copy》：https://chat.deepseek.com/share/etkkl7tbliko5g3a2o （2026-09-07 抓取；为教学型问答，结论以 02 笔记和 CUTLASS 源码核对为准）
- `third_party/cutlass/include/cute/atom/copy_atom.hpp`、`third_party/cutlass/media/docs/cpp/cute/0x_gemm_tutorial.md`（经 02 号笔记转引）
- [CUTLASS GitHub](https://github.com/NVIDIA/cutlass)
