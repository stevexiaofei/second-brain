---
title: 混合精度 MMA 与自定义 FP8 MMA op（TV Layout / MN Layout）
type: concept
status: growing
tags: [AI, CUDA, CUTLASS, CUTE, MMA, FP8, 混合精度, Kernel]
created: 2026-09-07
updated: 2026-09-07
source: 知乎《CUTLASS 笔记 (2)：混合精度 GEMM kernel》 + cutlass-notes 仓库 + PTX 文档
---

# 混合精度 MMA 与自定义 FP8 MMA op（TV Layout / MN Layout）

> 本文整理自知乎专栏《CUTLASS 笔记 (2)：混合精度 GEMM kernel》（作者杨远航，收录于 [cutlass-notes](https://github.com/ArthurinRUC/cutlass-notes)），并对照 CUTLASS 4.1 / SM90 源码理解。原文部分配图与代码段在网络抓取中截断，涉及细节标注"原文截断，以仓库为准"。

## 一句话理解

单条 `mma.sync` 指令的输入、输出、累加精度是被 PTX 固定死的组合；而真实算子往往需要"A 用 BF16、累加用 FP32、输出 BF16"这类混合精度。实现混合精度有两种手段：**换一条支持该组合的 MMA 指令**，或**在寄存器里手动转换精度**。当 PTX 支持但 CUTLASS 没封装某种组合（如 FP8 E4M3×E5M2）时，就要按 TV Layout / MN Layout 手写一个自定义 MMA op。

## 为什么重要

- 控制数值精度直接决定训练/推理质量：PyTorch 原生 API（`matmul`/`addmm`）无法自由指定累加精度，也要求 A/B/C 同 dtype，只能靠自定义算子
- 低精度（FP8/BF16）计算必然引入混合精度：DeepSeek-V3 的 FP8 Linear 就是"BF16 输入量化成 FP8 → FP8×FP8 + FP32 累加 → 输出转回 BF16"
- 学会手写 MMA op = 真正理解 CUTLASS/CuTe 里 MMA 的寄存器映射（TV Layout），也能扩展 CUTLASS 尚未覆盖的 PTX 指令

## MMA 的精度体系

一个 MMA（如 $D = A \times B + C$）涉及四类精度，见 PTX 指令命名：

```
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
                         │        │     │     │     └─ D 精度（= ComputeTypeC）
                         │        │     │     └─────── B 精度（ComputeTypeB）
                         │        │     └───────────── A 精度（ComputeTypeA）
                         │        └─────────────────── 累加精度 AccType（实际用）
                         └──────────────────────────── 指令 shape (M,N,K)
```

- **AccType**：A×B 结果与 C 相加时的**实际计算精度**（不等于字面输出精度）
- **ComputeTypeA/B/C**：A、B、C 各自存储/输入精度
- PTX 文档按 shape / sparsity / 架构列出支持的精度组合；**不是任意组合都有指令**
- 本文所有例子约定 `AccType == ComputeTypeC`（绝大多数指令如此）

## 实现混合精度的两种方式

### 方式一：换 MMA 指令（精度组合恰好存在）

目标 `fp32 = bf16 × bf16 + fp32` 有现成 PTX 指令，CUTLASS 也封装好了，只换 op：

```cpp
using MMA_op = SM80_16x8x8_F32BF16BF16F32_TN;
```

PTX 变化（对比全 FP16 版只是精度描述变了）：

```asm
mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32
```

### 方式二：寄存器里手动转精度（组合不存在时）

目标 `bf16 = bf16 × bf16 + fp32`（输出要从 fp32 转 bf16）：先按 FP32 累加出结果，再用 `copy` 转精度到 BF16 寄存器 Tensor，最后才写回 global：

```cpp
auto tCrO = make_tensor_like<OutType>(tCrC);  // 与累加片段同 shape，但元素类型 = OutType(bf16)
copy(tCrC, tCrO);                              // fp32 → bf16 的逐元素精度转换
```

`copy` 在这的语义就是循环赋值（CuTe 中类型不同的拷贝自动变成数值转换）：

```cpp
for (int i = 0; i < size(tCrC); ++i) tCrO(i) = tCrC(i);
```

#### PTX / SASS 佐证

转换层 PTX 会为每线程 4 个结果寄存器各加一条 cvt（默认 `.rn` = round to nearest even）：

```asm
cvt.rn.bf16.f32 %rs2, %f2;
```

SASS 侧更省：2 条 `F2FP.BF16.F32.PACK_AB` 把 4 个 FP32 寄存器打包成 2 个（各含 2 个 BF16），再 `STG` 写 global。**PACK 指令说明 BF16 输出常按 2 个/寄存器紧凑存储。**

## 手写自定义 FP8 MMA op

场景：Ada(SM90) 起 PTX 支持 FP8，且**允许 A/B 分别用两种 FP8 format**：

```
mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32   (shape 16×8×32)
```

但 CUTLASS 没有封装该组合 → 自己写。三步：写 MMA op（内联汇编）→ 写 MMA Traits（寄存器映射）→ 引用。

### 步骤 1：MMA op —— 封装一条 PTX 指令

`MMA op` 描述"执行什么指令、每个线程每种矩阵占几个寄存器"。FP8 版每线程：A 16 个元素（fp8×16=16B→4 reg）、B 8 个元素（2 reg）、C/D 4 个 fp32（4 reg）：

```cpp
struct SM90_16x8x32_F32E4M3E5M2F32_TN {
    using DRegisters = float[4];
    using ARegisters = uint32_t[4];   // 8 个 fp8 塞进 1 个 uint32
    using BRegisters = uint32_t[2];
    using CRegisters = float[4];
    CUTE_HOST_DEVICE
    static void fma(float& d0..d3, const uint32_t a0..a3,
                    const uint32_t b0,b1, const float c0..c3) {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
            : "=f"(d0).. : "r"(a0).. : ...);
    }
};
```

对照封装 FP16 的 `SM80_16x8x8_F16F16F16F16_TN`（每线程 A 4 元素、B 2 元素、C/D 2 个 fp16 存 uint32[2]），可见寄存器数量由"每线程元素数 ÷ 每寄存器可塞元素数"决定。

### 步骤 2：MMA Traits —— 描述指令内生的寄存器映射

`MMA op` 告诉硬件"算 16×8×32 用谁累加到哪"，但**没说每个线程拿哪些元素**。这个映射由 MMA Traits 的 `ALayout/BLayout/CLayout`（都是 TV Layout）表达。FP8 版（以作者推导结果为例，原文配图截断，常量级结论已与仓库核对思路）：

```cpp
template <>
struct MMA_Traits<SM90_16x8x32_F32E4M3E5M2F32_TN> {
    using ValTypeA = float_e4m3_t; using ValTypeB = float_e5m2_t;
    using ValTypeC = float; using ValTypeD = float;
    using Shape_MNK = Shape<_16, _8, _32>;
    using ThrID = Layout<_32>;                       // 32 线程一起执行一条指令
    using ALayout = Layout<Shape<Shape<_4, _8>, Shape<_4, _2, _2>>,
                          Stride<Stride<_64, _1>, Stride<_16, _8, _256>>>;
    ...
};
```

## TV Layout 与 MN Layout：核心难点

### 三种 Layout 的关系

CuTe 里"Layout"在不同语境指不同映射，MMA 这里一次遇到三种：

| 名称 | 映射 | 含义 |
|---|---|---|
| **（普通）Layout** | 坐标 → 内存 offset | Tensor 的 shape/stride |
| **TV Layout** | (T, V) → (M, N) | 线程 T 的第 V 个元素对应矩阵哪个位置（T=thread id，V=matrix element index） |
| **MN Layout** | (M, N) → (T, V) | TV 的逆映射，PTX 文档给的"warp 里谁持谁"图 |

TV 与 MN 互为逆映射（通常双射）。MN Layout 就是 PTX 文档中"一个 warp 内矩阵元素到线程/寄存器"的经典分配图。

### 一个算例：ALayout 怎么用

以作者例子的 FP16 A 布局 `((4,8),(2,2)) : ((32,1),(16,8))` 推导"线程 11 的第 2 个元素"取哪个矩阵坐标：

- 外层：T 维 shape `(4,8)`，把 `T=11` 转成嵌套坐标：`11 → (11%4, 11/4) = (3, 2)`
- 外层：V 维 shape `(2,2)`，`V=2 → (0, 1)`（原文此处截断，按 CuTe 坐标展开惯例补全推导）
- 合起来用 stride 算线性 index：坐标各自点乘 stride，映射进 (M,N) 空间得到 idx
- 内层：M×N 总 shape 如 `(16,8)`，把 idx 反解成 `(row, col)`

作者算例（FP16 布局）：T11 V2 → MN = (10, 6)。核验 MN Layout 图确实一致。（注：图 4 原文配图无法抓取，此处以文字推导记录结论）

### 为什么要能"反推"TV Layout

PTX 文档只给 MN Layout（图），CuTe 需要 TV Layout（代码）。推导技巧（作者给的口诀）：**逐 mode 看相邻线程间的 stride，stride 突变处就是一个 mode 的边界**——把 T0V0→T1V0→…→T31 的步长分段，段长就是该 mode 的 shape，段内步长就是该 mode 的 stride。FP8 的 ALayout `(64,1),(16,8,256)` 就是这么推出来的（原文推导有截断，结论常量可见）。

### 组装

```cpp
using MMA_op = SM90_16x8x32_F32E4M3E5M2F32_TN;  // op + traits 共同构成 MMA Atom
```

之后用 `make_tiled_mma` 扩成更大 tile 的用法与普通 MMA op 一致。

## 验证

作者共写了 4 个 kernel（fp32/bf16 输出 × MM/MMA 场景），外加自定义 FP8 kernel，跑测试比对结果（原文测试输出日志截断）。PTX 检查确认 FP8 版内联汇编原样出现；SASS 侧出现一串 `F2FP.F16.E4M3.UNPACK_B` / `F2FP.F16.E5M2.UNPACK_B` 打包指令，最终由 `HMMA.16816.F32` 执行（作者注：中间 pack/unpack 原理尚未完全搞清）。

## 我的理解

（个人理解，非原文）

- **精度控制本质是"表达方式"问题，不是数学问题**：三种机制（换指令 / 寄存器 cvt / 手写 op）都在回答"让硬件把什么精度放进 MMA 管线"。寄存器 cvt 最通用但也最贵（多读改写），换指令零成本但受 PTX 支持表限制
- **FP8 打包**：`uint32[4]` 存 16 个 E4M3 说明硬件层面 8 个 fp8 挤一个寄存器；E4M3/E5M2 混用其实只是"同一 MMA 引擎两路输入各用一套 8-bit 解释"，所以 SASS 要 UNPACK 对齐
- **TV/MN Layout 是"谁拿到哪个元素"的完整契约**：写算子时若 Layout 和 PTX 文档不一致，结果必然错位且极难排查——这正是模板把正确性编译进来的价值

## 常见误区

1. **"累加精度 = C 矩阵精度"** → 错。AccType 是实际累加位宽（如 fp16 指令也常 fp32 累加），C 的存储精度只是输入。
2. **"混合精度 = 换一个 op 就行"** → 只在目标组合存在指令时成立；不存在就要寄存器 cvt 或手写 op。
3. **"TV Layout 只能查表"** → 可以从 MN Layout 按 stride 突变反推，不依赖背图。

## Related

- [03 TiledMMA 与 fragment](./03-cute-tiled-mma.md) — MMA atom 如何复制成大 tile（本文是 atom 内部映射的补充）
- [01 Tensor 与 Layout](./01-cute-tensor-and-layout.md) — Layout 基本概念与坐标映射
- [CUTE 入门：从 C CUDA 视角理解 flash-attn 的 kernel 写法](../flash-attention/cute-basics.md) — 无 cute 基础先看这篇
- [FlashAttention 源码精读](../flash-attention/flash-attention-source-reading.md) — 真实 kernel 中的 TiledMMA 用法

## References

- 知乎《CUTLASS 笔记 (2)：混合精度 GEMM kernel》，https://zhuanlan.zhihu.com/p/1940158874255602181
- cutlass-notes 开源仓库，https://github.com/ArthurinRUC/cutlass-notes
- NVIDIA PTX ISA：Warp-level Matrix Shape 指令精度表
