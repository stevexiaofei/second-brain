---
title: CUTE TiledMMA：一条 MMA 指令如何"复制"成大 tile
type: concept
status: growing
tags: [AI, CUDA, CUTE, CUTLASS, MMA, TiledMMA, Kernel]
created: 2026-09-01
updated: 2026-09-11
source: cutlass-notes/01-minimal-gemm + 03-tiled-mma + 15-tiled-gemm（本地实践，SM80）；知乎《CUTLASS 笔记 (3)：Tiled MMA》（SM90 + BF16，https://zhuanlan.zhihu.com/p/1950555644814946318）
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

`make_tiled_mma` 的第 2 个参数（ThrLayout），形状如 $(2,4,1)$：把整个 warp 在 M 方向摆 2 组、N 方向摆 4 组 → 线程数 $32 \times 2 \times 4 = 256$。

**精确读法**：`ThrLayout` 本身是一个 Layout，语义是 **(M,N,K) 坐标 → warp_idx 的映射**。这里的 (M,N,K) 坐标指的是"第几个 MMA Atom"，输出的是"这个 Atom 交给几号 warp 算"。

```
make_layout(make_shape (2,4,1))  →  (2,4,1):(1,2,8)      // 默认紧凑 stride
warp_idx = m*1 + n*2 + k*8
```

验证：坐标 $(M,N,K)=(1,2,0)$ → $1\times1 + 2\times2 + 0\times8 = 5$ → **warp 5 = T160–T191**。

**为什么 K 维的 ThrExpand 一般为 1（不在 K 方向扩线程）？**
因为 $D = A \times B$ 的累加器由**某一个线程独占**。若把 K 拆给不同线程，同一个 $(m,n)$ 的部分和就散落在多个线程里，必须做**跨线程归约**（shuffle / SMEM），代价极高。因此 K 方向的扩展必须落在**同一线程的寄存器里**，用"多条 mma 顺序累加"完成 —— 这正是后文 SASS 中两条 HMMA 复用同一累加器的原因。

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

### 打印格式：如何读一个 Tensor（定位 MMA_M/N/K）

`partition_*` 返回的 6 个 Tensor，原文打印如下：

```text
tCgA  gmem_ptr[16b](0x7f61c3e00000) o ((_2,_2),_1,_2):((_1,128),_0,_8)
tCgB  gmem_ptr[16b](0x7f61c3e00400) o ( _2,_1,_2):( _1,_0,_8)
tCgC  gmem_ptr[32b](0x7f61c3e01800) o ((_2,_2),_1,_1):((_1,256),_0,_0)
tCrA  ptr[16b](0x7f61d9fffca0)      o ((_2,_2),_1,_2):((_1,_2),_0,_4)
tCrB  ptr[16b](0x7f61d9fffcb0)      o ( _2,_1,_2):( _1,_0,_2)
tCrC  ptr[32b](0x7f61d9fffcc0)      o ((_2,_2),_1,_1):((_1,_2),_0,_0)
```

先破一个可能的坎：**`_1` 就是数字 1，`_2` 就是数字 2** —— `_N` 只是 CuTe 里 `Int<N>`（编译期常量）的打印形式，不是特殊标记。

#### 第一步：解剖一行打印

```text
gmem_ptr[16b](0x7f61c3e00000)  o  ((_2,_2),_1,_2) : ((_1,128),_0,_8)
└──────── data ────────┘      │  └── shape ──┘    └── stride ──┘
                              └ 分隔符：左边 data，右边 Layout
```

| 字段 | 含义 |
|---|---|
| `gmem_ptr` / `smem_ptr` / `rmem_ptr` / `tmem_ptr` | 存储介质前缀；**无前缀的 `ptr` 是普通指针，不携带存储介质信息** |
| `[16b]` / `[32b]` | 单个数据的位宽（bf16 为 16b，fp32 累加器为 32b） |
| `(0x...)` | 数据基地址 |
| `o` | 分隔符，左边 Tensor data、右边 Tensor Layout |
| `shape : stride` | 顶层是嵌套 tuple，**要按逗号分层数 mode** |

`shape` 是嵌套的，所以顶层 mode 数**不等于**数字个数：

```text
( (_2,_2) , _1 , _2 )
   ↑mode0   ↑m1   ↑m2       顶层 3 个 mode；mode0 内部还嵌了一层 2×2
```

**顶层 mode 数 = 3**，正好对应 `(MMA, 两个扩展维)`。

#### 第二步：用 partition API 定下三个 mode 的语义

打印**不会告诉你**哪个 mode 是 M、哪个是 K，这个信息来自 API 约定：

| partition API | mode0 | mode1 | mode2 |
|---|---|---|---|
| `partition_A` / `partition_fragment_A` | MMA | **MMA_M** | **MMA_K** |
| `partition_B` / `partition_fragment_B` | MMA | **MMA_N** | **MMA_K** |
| `partition_C` / `partition_fragment_C` | MMA | **MMA_M** | **MMA_N** |

注意 **C 的第三个 mode 是 N 不是 K** —— C 是累加结果，K 已被累加掉，不存在该维度（这也是 `tCrC` 不随 K 扩展变大的原因）。

#### 第三步：逐一对号

| Tensor | 打印 shape | 拆分 | 元素数 |
|---|---|---|---|
| `tCgA` | `((_2,_2),_1,_2)` | MMA=(2,2)，**MMA_M=1**，**MMA_K=2** | 8 |
| `tCgB` | `(_2,_1,_2)` | MMA=(2)，**MMA_N=1**，**MMA_K=2** | 4 |
| `tCgC` | `((_2,_2),_1,_1)` | MMA=(2,2)，**MMA_M=1**，**MMA_N=1** | 4 |
| `tCrA` | `((_2,_2),_1,_2)` | 同 `tCgA` | 8 |
| `tCrB` | `(_2,_1,_2)` | 同 `tCgB` | 4 |
| `tCrC` | `((_2,_2),_1,_1)` | 同 `tCgC` | 4 |

`tCg*`（global）与 `tCr*`（register）的 **shape 完全相同，只有基地址和 stride 不同** —— 它们是同一个 partition 作用在不同存储介质上的结果。

三个量都能从两个不同 Tensor 各读一次、互相印证：

```text
MMA_M : tCgA.mode1 = 1  ≡  tCgC.mode1 = 1   →  1
MMA_N : tCgB.mode1 = 1  ≡  tCgC.mode2 = 1   →  1
MMA_K : tCgA.mode2 = 2  ≡  tCgB.mode2 = 2   →  2
```

$$\Rightarrow (\text{MMA\_M}, \text{MMA\_N}, \text{MMA\_K}) = (1,\ 1,\ 2)$$

#### 它就是 ValExpand

原文配置 `kMmaValExpandM/N/K = (1,1,2)`，与打印结果一致 —— **`MMA_M/N/K` 和 `kMmaValExpandM/N/K` 是同一个东西的两种名字**：

- 在 `make_tiled_mma` 里，它决定 TiledMMA 的 tile 大小；
- 在 `partition_*` 的结果里，它表现为多出来的 2 个 mode（加 mode0 共 3 个）。

翻译成人话：**这两个扩展 mode 记录的是"这个线程还要把指令重复执行几遍"，而不是"这个线程拥有多少不同的数据"**；mode0 才是"一个 atom 内该线程真正持有的那几个元素"。这印证了「partition 结果多出来的 `MMA_M/N/K` 维度，就是 mma 指令的扩展维度」。

#### 自校验 ①：元素数守恒

$$\text{每线程元素数} = \underbrace{\text{atom 中该线程持有的元素数}}_{\text{mode0 展开}} \times (\text{对应的两个 MMA\_})$$

| Tensor | 校验 |
|---|---|
| `tCrA` | atom A 每线程 4 个 × MMA_M(1) × MMA_K(2) = **8**，打印 $2\times2\times1\times2 = 8$ ✔ |
| `tCrB` | atom B 每线程 2 个 × MMA_N(1) × MMA_K(2) = **4**，打印 $2\times1\times2 = 4$ ✔ |
| `tCrC` | atom C 每线程 4 个 × MMA_M(1) × MMA_N(1) = **4**，打印 $2\times2\times1\times1 = 4$ ✔ |

**注意 `tCrC` 是 4 而不是 8** —— 累加器不随 K 增长，这正是"K 扩展后 STG 仍然只有 2 条"的原因。

#### 自校验 ②：stride 反查 SASS

`tCgA` 的 MMA_K stride 是 `_8`，含义是"第二个 K 半段在内存中往后挪 8 个元素"。bf16 每元素 2 字节：

$$8 \times 2\ \text{B} = 16\ \text{B} = \texttt{0x10}$$

对照本页后面「SASS 观察」一节里的 `LDG.E R2, [R14.64+0x10]` —— **偏移量完全对得上**。打印出的 Layout 与最终 SASS 的地址计算用的是同一套数学，这是验证自己是否真正读懂的硬核办法。

#### 三个容易踩的坑

1. **数错 mode 数**：`((_2,_2),_1,_2)` 顶层是 3 个 mode，不是 4 个。mode0 内部的嵌套是"atom 内线程布局"，属于 MMA 那一维，别把它当成 MMA_M。
2. **A 的 mode0 嵌套 `(_2,_2)`、B 的扁平 `_2`**：因为 16×8×8 atom 里 A 每线程 4 个元素（自然两两成对）、B 每线程 2 个。**这是 atom 固有属性，与 ValExpand 无关**，别误以为是扩展出来的。
3. **stride `_0` 不是"没有 stride"**：CuTe 对 size=1 的 mode 记作 0（该维不存在，地址不变）。所以 `tCgA` 的 MMA_M stride 为 `_0` 恰好说明 MMA_M = 1。

### gemm() 不是一条指令，而是一个展开循环

一次 `gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)`，每个线程会串行发

$$\text{MMA\_M} \times \text{MMA\_N} \times \text{MMA\_K} = 2\times2\times2 = 8 \text{ 条 } mma.sync$$

即 ValExpand 的乘积。用 MAC 数交叉验证：一个 block 一次 gemm 算 $64\times64\times16 = 65536$ MACs，每条指令 1024 MACs → 64 条指令 ÷ 8 个 warp = 每条线程 8 条 ✓。

**两个容易搞混的点**：

1. **进入这个乘积的是 ValExpand，不是 ThrExpand。** ThrExpand 决定"哪个 warp 算哪个 atom"（改变**谁干**），ValExpand 决定"同一线程多发几条"（改变**干多少**）。所以线程数从 32 变到 256，单线程指令数不变。
2. **"2 条"是每线程的发射次数，也就是每 warp 的指令数**（`mma.sync` 是 warp 级指令，32 线程协同执行一条）。整个 block 要再乘 warp 数。

| 视角 | 文章配置（`32x32x16`，ValExpand `(1,1,2)`） | 本节配置（`64x64x16`，ValExpand `(2,2,2)`） |
|---|---|---|
| 每线程 = 每 warp | $1\times1\times2 = 2$ | $2\times2\times2 = 8$ |
| 每 block | $8 \times 2 = 16$ | $8 \times 8 = 64$ |

对照文章配置的 MAC 法核对：一个 block 一次 gemm 算 $32\times32\times16 = 16384$ MACs，每条指令 1024 MACs → 16 条 ÷ 8 warp = 每条线程 2 条 ✓（与 SASS 里 2 条 HMMA 完全一致）。

### 整个 kernel 的指令量

K 主循环把全矩阵 $K=64$ 切成 4 段 16，每次循环调一次 gemm()：每个线程共执行 $4 \times 8 = 32$ 条 `mma.sync`。

一般式：

$$\text{每线程总 mma} = \underbrace{\frac{K}{K_{\text{tile}}}}_{\text{K 主循环轮数}} \times \underbrace{(\text{MMA\_M} \times \text{MMA\_N} \times \text{MMA\_K})}_{\text{单轮 gemm() 展开条数}}$$

**先算单轮，再乘轮数** —— 这句话能覆盖绝大多数"这个 kernel 到底发多少条 mma"的问题。

### local_tile + Step 投影：给三个张量切块

用同一个 3D tiler $(64,64,16)$ 分别给 A(B,C) 切块时，`Step<_1,X,_1>` 表示从 tiler 里挑出需要的两维：

```
gA = local_tile(mA, tiler, coord, Step<_1, X, _1>{})  // (64, 16, K/16)
gB = local_tile(mB, tiler, coord, Step< X, _1, _1>{})  // (64, 16, K/16)
gC = local_tile(mC, tiler, coord, Step<_1, _1, X>{})   // (64, 64)
```

结果的前两维是 tile 本身，最后一维是 K-tile 序号（K 主循环用 `gA(_, _, ktile)` 逐个取）。注意 `_` 放在 coord 里 = "剩余块全保留，做成新维度"；放具体数字 = 只取那一块。

## SASS 观察：K 维扩展后发生了什么

另一组实测配置（知乎《CUTLASS 笔记 (3)》，SM90 + BF16，tile `32x32x16`，ThrExpand `(2,4,1)`，ValExpand `(1,1,2)`）：

| 指令 | 单指令时 | Tiled 时 | 原因 |
|---|---|---|---|
| LDG | 3 | **6** | 每条 mma 各需一套 A/B 数据 |
| HMMA | 1 | **2** | K 维扩了 2 倍 |
| STG | 2 | 2 | 累加器只有一个，只需写回一次 |

关键点：K 维的两条 HMMA 是**链式累加**的：

```text
HMMA.1688.F32.BF16 R4, R4,  R6, RZ     ← 第一条：累加器从 RZ(=0) 起
HMMA.1688.F32.BF16 R4, R2, R11, R4     ← 第二条：复用 R4 作为累加初值
```

因为两次 $A \times B$ 计算的是**同一个 D**，第二条 mma 必须用第一条的结果作为累加初值 —— 这和「K 维不扩线程」是同一件事的两面。指令语义（`D = A×B + C`）、`RZ` 清零、以及为什么 K 是串行链而 M/N 是独立链，见 [08：MMA 指令语义与累加方向](./08-mma-instruction-and-accumulation.md)。

原文还指出：此时访存指令前会多出感叹号标记，说明这些 LDG **重复读取了 GMEM 数据**，即 tile 级别下访存行为已经开始劣化（数据复用没做好），将在下一篇 Tiled Copy 中解决。

## K 维扩展 vs K 主循环：是同一件事的两种写法吗

先说结论：**在指令数和依赖结构上完全等价，差别只在"循环写在哪一层"和"寄存器占用"**。

### 等价性

文章配置的 tile K $= 8 \times \text{ValExpandK} = 16$，而问题 K 也恰好是 16，所以没有 K 主循环。如果改成 `kMmaValExpandK = 1`：

**方案 A（ValExpandK = 2，无循环）**

```text
一次 gemm():
  LDG ×6     (A 32x16, B 16x32)
  HMMA        k=0..7
  HMMA        k=8..15
  STG ×2
```

**方案 B（ValExpandK = 1，K 循环 2 轮）**

```text
k=0:  LDG ×3 → HMMA
k=1:  LDG ×3 → HMMA
      STG ×2
```

总数完全一样：**6 LDG、2 HMMA、2 STG**。

这不是巧合——全局 mma 总数只由 $\frac{M}{16}\times\frac{N}{8}\times\frac{K}{8}$ 决定。**tile 只改变"工作量如何分组"，不改变总指令量。**

### 那差别在哪

| | K 主循环（ValExpandK = 1） | K 维扩展（ValExpandK > 1） |
|---|---|---|
| 循环存在形式 | kernel 里的**运行时** `for` | TiledMMA **类型里的编译期展开** |
| 循环开销 | 每轮：分支、计数器、地址递增 | 无 |
| `partition_*` / 切片 | 每轮做一次 | 只做一次 |
| 每线程 A/B fragment | 只持有**单步** K 的数据 | 持有**整个 tile K** 的数据 |
| 寄存器压力 | 小 | **大** ← 这是代价 |
| 搬运粒度 | 小批多次 | 一次搬更多 |

所以 ValExpandK 的定位是**展开因子 / 粒度旋钮**：用寄存器换循环开销与指令连续性，**不是"多做一遍计算"**。

### 为什么文章这里必须写 2

不是性能需要，而是 CuTe 的类型约束。Block tile 的 K $=16$，而 TiledMMA tile 的 K $= 8 \times \text{ValExpandK}$。若取 1，TiledMMA 只覆盖 K=8，与 Block tile 的 K=16 不匹配（`static_assert` 过不了），必须在外层再套一个 K 循环。

取 2 是为了让"**一次 `gemm()` 调用 = 一个完整 Block tile**"这个教学叙事成立 —— 这是**教学取舍，不是性能取舍**。

### 真实 GEMM 里 ValExpandK 的价值

真实 GEMM 的 block tile K（如 64 / 128）远大于 8，结构是嵌套的：

```text
for k_outer in tiles(BK):                  ← K 主循环：GMEM → SMEM（有真实通信/流水线代价）
    for k_inner in 0 .. BK/(8*ValExpandK):  ← 从 SMEM 取 operand 切片
        gemm(...)                           ← 内部再展开 ValExpand ×(M/N 方向)
```

此时 ValExpandK 决定"**一次从 SMEM 取数后能连续发多少条 mma**"，摊薄内层循环与切片的开销。收益随 $BK$ 增大而摊薄，所以工程上通常取 2 左右，不会取很大。

### 别忘了寄存器的账

ValExpandK 增大的是 K 方向的 **A/B fragment** 寄存器占用（累加器 C 的大小不受影响）。文章开头就强调过：扩展 mma 指令需要更多寄存器，扩过头会 register spilling，反而拖垮性能 —— 这是这笔交易的另一半。

## 我的理解

（以下为个人理解，非 cute 文档原文）

- 三个旋钮的分工：ThrExpand 决定"并发度"（warp 数），ValExpand 决定"单线程工作量"（指令重复次数），K 主循环决定"调用几次 gemm()"。三者都在增大总计算量，但占用的硬件资源形态完全不同——加 warp 吃 SM 上的并行槽位，加 ValExpand 吃寄存器，加 K 循环吃时间。
- 片段副本的方向恰好和复制方向相反：A 的副本数是 ThrExpandN，B 的副本数是 ThrExpandM，C 无副本。所以"A 片段 16、B 片段 8"不是 bug，而是布局使然。
- 与其背公式，不如会算：由 C 片段大小 ÷ 单条指令的 C 贡献（4）得到 M×N 方向指令数，K 方向的重数看 fragment 的 MMA_K 模式。

## 常见误区

1. **"tile 必须等于指令大小"** → 错。只需 tile 各维是指令各维的整数倍，倍数 = ThrExpand × ValExpand。
2. **"片段大小 = tile 元素数 ÷ 线程数"** → 错。A/B 有副本（见上），只有 C 满足这个朴素除法。
3. **"一次 gemm() 调用 = 一条指令"** → 错。它是按片段模式展开的多条指令序列；条数 = ValExpand 乘积，与 ThrExpand 无关。
4. **"kernel 参数 k（全矩阵 K）就是 tile 的 K"** → 错。全矩阵 K 被切成 K/kTileK 段：gA 中间维是 kTileK，最后一维才是段数。
5. **"K 维扩展是为了多做计算 / 提高并行度"** → 错。总指令数不变（$M/16 \times N/8 \times K/8$ 与 tile 无关），K 扩展是**展开因子**；且 K 方向共用同一个累加器，是串行链，**不产生并行度**（见 [08](./08-mma-instruction-and-accumulation.md)）。
6. **"K 方向的两条 mma 各自独立累加"** → 错。第二条必须拿第一条的结果作为累加初值，这是 `D = A×B + C` 的直接后果。

## Related

- [CUTLASS/CuTe 01：Tensor、Layout 与坐标映射](./01-cute-tensor-and-layout.md) — Layout 与分区的前置知识
- [CUTLASS/CuTe 06：GEMM 三级 Tiling](./06-gemm-three-level-tiling.md) — TiledMMA 处在三级中的哪一级、为什么要分层
- [CUTLASS/CuTe 07：CuTe Permutation Layout](./07-cute-permutation-layout.md) — `MMATileLayout` 背后的第 4 种 Layout
- [CUTLASS/CuTe 08：MMA 指令语义与累加方向](./08-mma-instruction-and-accumulation.md) — `D = A×B + C`、K 串行链与 M/N 独立链
- [CUTE 入门：从 C CUDA 视角理解 flash-attn 的 kernel 写法](../flash-attention/cute-basics.md) — Tensor / Layout / local_tile / 线程分区的前置知识
- [FlashAttention 源码精读](../flash-attention/flash-attention-source-reading.md) — 上层 kernel 中 TiledMMA 的实际用法
- [FlashAttention Kernel 与 Launch 机制](../flash-attention/flash-attention-kernel-and-launch.md) — tile、launch 与 split-KV

## References

- 实践代码：`/root/autodl-tmp/project/cutlass-notes/01-minimal-gemm`、`03-tiled-mma`、`15-tiled-gemm`
- CUTE 源码：`third_party/cutlass/include/cute/atom/mma_traits_sm80.hpp`（`ThrID = Layout<_32>`、ALayout/CLayout）
- `cute/tensor_impl.hpp` 的 `local_tile` / `inner_partition` 注释（Step 投影语义）
- 知乎《CUTLASS 笔记 (3)：Tiled MMA》，作者杨远航，CUTLASS 4.1.0 / SM90 + BF16：<https://zhuanlan.zhihu.com/p/1950555644814946318>（ThrLayout 的 warp 映射、Tensor 打印格式、SASS 分析出自此文）
