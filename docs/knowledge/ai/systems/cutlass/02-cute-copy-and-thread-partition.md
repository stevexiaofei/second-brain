---
title: CUTLASS/CuTe 02：Copy Atom、TiledCopy 与线程分区
type: concept
status: seed
tags: [AI, CUDA, CUTE, CUTLASS, Copy, TiledCopy, Thread]
created: 2026-09-02
updated: 2026-09-11
source: third_party/cutlass/include/cute/atom/copy_atom.hpp、cute/algorithm/copy.hpp；知乎《CUTLASS 笔记 (4)：Tiled Copy》（https://zhuanlan.zhihu.com/p/1968745447741972494）
---

# CUTLASS/CuTe 02：Copy Atom、TiledCopy 与线程分区

## 一句话理解

`TiledCopy` 描述整个线程块如何协作搬运一个 tile；`get_thread_slice(tid)` 取出一个线程的局部分工；`partition_S` 和 `partition_D` 再分别生成 source 与 destination 的 Tensor view。它们是“谁搬哪些元素”的类型化表达，不是一次自动发生的拷贝。

本笔记讲的是 **API 语义层**（参数是什么、结果长什么样）；`partition_S/D` 背后的**数学原理**（$S$/$D$/$R$ 三个 Layout 与复合映射）在 [09：TiledCopy 核心原理](./09-cute-tiled-copy-principle.md)。

## 从 Copy Atom 到 TiledCopy

可以把层次看成：

```text
Copy Atom      一次底层 copy 的元素类型、向量宽度和基本契约
线程布局       哪些线程参与、线程如何排布
value 布局     每个线程一次携带哪些值
TiledCopy      把上述模式平铺到完整 tile
thread slice   某个 tid 在这套模式中的局部视图
```

典型形态：

```cpp
using CopyAtom = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>;
auto tiled_copy = make_tiled_copy(CopyAtom{}, thread_layout, value_layout);
auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);
```

`128` 在 `AutoVectorizingCopyWithAssumedAlignment<128>` 中表示最多按 128 bit（16 byte）对齐/向量化的假设，不表示 128 个线程。线程数由 thread layout 和 launch 配置决定。

### thread_layout 与 value_layout：参数补齐与语义

上面骨架省略了两个 layout 的构造。它们的语义来自 `make_tiled_copy` 的签名注释（`third_party/cutlass/include/cute/atom/copy_atom.hpp`，以下代码与注释均据此）：

```cpp
/** The thread and value layouts map coordinates to thr_idx and val_idx. */
make_tiled_copy(Copy_Atom const& atom,
                ThrLayout const& thr_layout = {},   // (m,n) -> thr_idx  线程布局
                ValLayout const& val_layout = {});  // (m,n) -> val_idx  值布局
```

- **`thread_layout`**：把目标 tile 的坐标映射到"线程编号"——决定 tile 被切成多少个线程槽位、每个线程对应哪个子位置。默认 `Layout<_1>`。
- **`value_layout`**：把同一份 tile 坐标映射到"值编号"——决定一个线程在它的子位置里还要细分出几个元素槽位。默认 `Layout<_1>`。

`make_tiled_copy` 内部把它们合成为 `TiledCopy` 的"覆盖表"：

```cpp
auto layout_mn = raked_product(thr_layout, val_layout);   // (M,N)                -> (thr_idx, val_idx)
auto layout_tv = right_inverse(layout_mn).with_shape(
                   make_shape(size(thr_layout), size(val_layout)));  // (thr_idx,val_idx) -> (M,N)
auto tiler     = product_each(shape(layout_mn));          // 这个 tile 在 M、N 上的尺寸
```

`layout_tv` 的 domain 形状是 `(size(thr_layout), size(val_layout))` = `(线程数, 每线程值数)`，并且它是 `(thr,val) -> (M,N)` 的一一对应（`right_inverse`）。因此槽位总数守恒，这是后面两个小节的前提：

$$
\text{tile 元素数} = M \times N = \underbrace{\text{prod(shape(thread\_layout))}}_{\text{线程数}} \times \underbrace{\text{prod(shape(value\_layout))}}_{\text{每线程值数}}
$$

### raked_product 返回什么：blocked vs raked

`raked_product(block, tiler)` 返回一个 **Layout**（`third_party/cutlass/include/cute/layout.hpp`）：把 `block` 平铺到 `tiler` 上，但采用**逐元素交错（cyclic）**而非成块（blocked）排布。实现上是 `logical_product` 后再按 mode 重新 zip，与 `blocked_product` 唯一区别是 zip 顺序相反（raked 先 tiler 后 block）。官方文档（`third_party/cutlass/media/docs/cpp/cute/02_layout_algebra.md`）称 blocked 为成块分布、raked 为 cyclic distribution：

```text
blocked_product（成块）:  [A0 A0 | A1 A1 | A2 A2]
raked_product（交错）   :  [A0 A1 A2 A0 A1 A2]
```

在 `make_tiled_copy` 里它是 `(M,N) -> (thr_idx, val_idx)` 的分配表：决定"线程与值"在坐标上的交错顺序（影响合并访问与向量化），**不改变槽位总数**。

### tile 尺寸、线程数、寄存器三者的关系

CUTE 教学例（`third_party/cutlass/media/docs/cpp/cute/0x_gemm_tutorial.md`）给出了完整数字：

```cpp
TiledCopy copyA = make_tiled_copy(
    Copy_Atom<UniversalCopy<uint128_t>, float>{},  // 128-bit 指令 = 4 个 float
    Layout<Shape<_32,_8>>{},                       // thread_layout：32×8 = 256 线程
    Layout<Shape<_4,_1>>{});                       // value_layout：每线程 4×1 = 4 个值
```

文档原文："each thread reads **4x1** TA elements and there are **32x8** threads." 代入公式：256 × 4 = **1024 个元素**；当两个 layout 都是 2D 且可分离时还能按维再乘：

$$
m = \text{ThrM} \times \text{ValM}, \qquad n = \text{ThrN} \times \text{ValN}
$$

由此得到"寄存器视角"的关键澄清：**寄存器限制的不是"tile 不能太大"，而是"单个线程同时握在手里的值不能太多"**：

- 大 tile 不需要塞进单个线程——它被 `#threads` 摊分，再被 partition 结果里的 Rest/时间维（如 `k`）按步处理，每步只占一小片寄存器
- `prod(value_layout)` 直接决定每线程的寄存器 footprint：过大 → 寄存器溢出（spill）或 occupancy 崩溃；过小 → 浪费向量化/指令宽度
- 但 gmem→smem 的 copy 里寄存器只是**中转**（load 后立刻 store），压力远小于 MMA 累加器那种"必须长期抱着"的场景
- 这也是 `make_tiled_copy` 把"线程怎么排"和"每线程几个值"拆成两个独立参数的原因：两者受到的物理约束不同

## Copy_Traits：S / D / R 三个 Layout

上面讲的是 `make_tiled_copy` 的两个参数。往下钻一层，一条**拷贝指令**本身需要记录三个映射（推导见 [09：TiledCopy 核心原理](./09-cute-tiled-copy-principle.md)）。以 `SM75_U32x4_LDSM_N`（即 `ldmatrix`）为例：

```cpp
template <>
struct Copy_Traits<SM75_U32x4_LDSM_N>
{
  // Logical thread id to thread idx (warp)
  using ThrID = Layout<_32>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape < _32,_128>,
                           Stride<_128,  _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_32,Shape <_32,   _4>>,
                           Stride<_32,Stride< _1,_1024>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
};
```

三点值得注意：

1. **注释里的 "bit" 是字面意思**：`SrcLayout` / `DstLayout` 的 codomain 是**位偏移**，不是元素序号（上例 val 维长度 128 就是 128 bit）。
2. **`SrcLayout ≠ DstLayout` 正是"会做线程间交换"的证据**：
   - `SrcLayout`：每线程拿到 **128 个连续 bit**（$Shape\langle32,128\rangle : Stride\langle128,1\rangle$）；
   - `DstLayout`：每线程拿到 **4 段、每段 32 bit、段间隔 1024 bit**（$Shape\langle32,\langle32,4\rangle\rangle : Stride\langle32,\langle1,1024\rangle\rangle$）。
   两者总位数都是 $32\times128=4096$ bit，但分布完全不同 —— 这就是 `ldmatrix` 在 warp 内重排数据。
3. **`RefLayout = DstLayout`**：这条指令选了 $R = D$。

`Copy_Atom` 从 Traits 取出这些 Layout 并做**单位换算**（与 MMA Atom / MMA Traits 的协作模式一致）：

```cpp
template <class... Args, class CopyInternalType>
struct Copy_Atom<Copy_Traits<Args...>, CopyInternalType>
  : Copy_Traits<Args...>
{
  using ThrID        = typename Traits::ThrID;
  using BitLayoutSrc = typename Traits::SrcLayout;
  using BitLayoutDst = typename Traits::DstLayout;
  using BitLayoutRef = typename Traits::RefLayout;

  using ValType = CopyInternalType;

  using ValLayoutSrc = decltype(recast_layout<uint1_t, ValType>(BitLayoutSrc{}));
  using ValLayoutDst = decltype(recast_layout<uint1_t, ValType>(BitLayoutDst{}));
  using ValLayoutRef = decltype(recast_layout<uint1_t, ValType>(BitLayoutRef{}));
  ...
};
```

**`recast_layout<uint1_t, ValType>` 就是"从 bit 折算成元素"这一步**，此后 Tensor 层面的 partition 才能按元素数推理。

## TiledCopy：把 CopyAtom 扩展成 tile

`TiledCopy` 在 `CopyAtom` 基础上沿**线程维（T）**和**数据维（V）**扩展，与 `TiledMMA` 的 ThrExpand / ValExpand 同构：

```cpp
template <class Copy_Atom,
          class LayoutCopy_TV,  // (tid,vid) -> coord   [Need not be 2D...]
          class ShapeTiler_MN>  // coord space
struct TiledCopy : Copy_Atom
{
  // Layout information from the CopyAtom
  using AtomLayoutRef = typename Copy_Atom::ValLayoutRef; // (thr,val) -> offset
  using AtomNumThr = decltype(size<0>(AtomLayoutRef{}));
  using AtomNumVal = decltype(size<1>(AtomLayoutRef{}));

  // Layout information for the TiledCopy
  using Tiler_MN       = ShapeTiler_MN;    // 拷贝的总规模
  using TiledLayout_TV = LayoutCopy_TV;    // (tid,vid) -> coord  ← Ref TV Layout
  using TiledNumThr    = decltype(size<0>(TiledLayout_TV{}));
  using TiledNumVal    = decltype(size<1>(TiledLayout_TV{}));
  ...
}
```

| 成员 | 作用 |
|---|---|
| `TiledLayout_TV` | 记录 **Ref TV Layout** $r$，用于变换 src / dst（即 `partition_S/D`） |
| `Tiler_MN` | 这个 TiledCopy 拷贝的**总规模**；size 与 `TiledLayout_TV` 的 size 相同（都是拷贝元素个数） |

**线程维与值维的扩展，就是通过扩展 `TiledLayout_TV` 的 T 维和 V 维实现的。**

`ThrCopy`（由 `get_slice(tid)` 取得，对应 `ThrMMA`）提供两类能力：

| API | 作用 |
|---|---|
| `partition_S` / `partition_D` | 把 Tensor 经复合映射变换后，**选出当前线程的分块** |
| `retile_S` / `retile_D` | **改变已有 Tensor 分块的 Layout 排布**（size 不变），使其适配拷贝操作的形状 |

`retile` 的用法：已经拿到 TiledMMA 的分块时，不必重新 partition，直接 retile 即可：

```cpp
Tensor tCgA = thr_mma.partition_A(gA);           // (MMA, MMA_M, MMA_K)
Tensor tCrA = thr_mma.partition_fragment_A(gA);  // (MMA, MMA_M, MMA_K)

Tensor tAgA = g2r_thr_copy_a.retile_S(tCgA);     // (CPY, CPY_M, CPY_K)
Tensor tArA = g2r_thr_copy_a.retile_D(tCrA);     // (CPY, CPY_M, CPY_K)

copy(g2r_tiled_copy_a, tAgA, tArA);
```

`partition_S/D` 与 `retile_S/D` 都只是 TiledCopy 核心原理的工程实现，代码解析留待讲完 CuTe Layout 之后。

## 从 TiledMMA 直接造 TiledCopy

`make_tiled_copy` 和 `make_tiled_mma` 一样接收 `CopyAtom`、`ThrLayout`、`ValLayout` 三个参数，内部据此算出 `Ref TV Layout` 与 `Tiler_MN`。

而当**需要的 Ref TV Layout 恰好就是 TiledMMA 的 TV Layout** 时，可以直接把 TiledMMA 实例传给 `make_tiled_copy_A/B/C`：

```cpp
using Copy_op = AutoVectorizingCopy;
using CopyA_atom = Copy_Atom<Copy_op, ComputeTypeA>;

using TiledCopyA = decltype(make_tiled_copy_A(CopyA_atom{}, TiledMMA{}));
```

**什么时候"恰好就是"？** 就是「**拷贝完马上交给 MMA 运算**」或者「**算完马上拷贝出去**」的场景 —— 因为此时用 MMA 的 TV Layout 作 Ref TV Layout，就能**在不改变 Tensor 排布的前提下**把拷来的 Tensor 直接喂给 MMA，或者算完直接搬走，省掉一次重排。

## CPY 与 MMA 的差别（原文含勘误）

`partition_S` 结果的三个 mode 记作 `(CPY, CPY_M, CPY_K)`，很容易与 TiledMMA 的 `(MMA, MMA_M, MMA_K)` 类比，但**语义不同**。

> **原文勘误（作者本人更正）**
> 早先版本写作"CPY 是每个线程每个拷贝指令参与的数据量，CPY_M / CPY_K 是 TiledCopy 在 M 和 K 维度的数据扩展规模"。
> **正确说法**：`CPY` 指**每个线程拷贝单个 Tile 的总数据量**，而非每个线程每个 Copy Atom 的数据量；`CPY_M` / `CPY_K` 指**从 Block 层面看** M 和 K 方向扩展了多少个 Tile。因为当前还没把 Tile 扩展到 Block，所以这里的 `CPY_M` 和 `CPY_K` 都是 **1**。

> **待核对**：CuTe 源码注释中 `partition_S` 的返回形状常被标注为 `(CPY, CPY_M, CPY_K)`，但其 mode0 究竟是"atom 的每线程值数"还是"整个 tile 的每线程总量"，可能随 `TiledCopy` 的构造方式（`make_tiled_copy` vs `make_tiled_copy_A(atom, tiled_mma)`）而变。此处按原文更正后的表述记录，待对照 `copy_atom.hpp` / `tensor_impl.hpp` 与实测打印确认。

## 读 TiledCopy 的 metadata

TiledCopy 支持打印与 LaTeX 可视化：

```cpp
cute::print(typename Spec::TiledCopyA{});
cute::print_latex(typename Spec::TiledCopyA{});
```

原文 `TiledCopyA` 的打印结果：

```text
TiledCopy
  Tiler_MN:       (_32,_16)
  TiledLayout_TV: ((_4,_8,_2,_4),((_2,_2,_2),(_1,_1))):((_64,_1,_16,_0),((_32,_8,_256),(_0,_0)))
Copy_Atom
  ThrID:        _1:_0
  ValLayoutSrc: (_1,_1):(_0,_0)
  ValLayoutDst: (_1,_1):(_0,_0)
  ValLayoutRef: (_1,_1):(_0,_0)
  ValueType:    16b
```

逐项读：

- `Tiler_MN: (_32,_16)` → 一次覆盖 M=32、K=16，与 TiledMMA 的 A 分块一致；
- `TiledLayout_TV` → domain 是 `(thr, val)`、codomain 是 `(M,K)`，即原理部分的 $r$；
- `ValLayout*: (_1,_1):(_0,_0)` → 这个 CopyAtom 是**逐元素**的（`AutoVectorizingCopy` 的向量化在编译期完成，本身不做线程间交换）；
- `ValueType: 16b` → BF16。

LaTeX 图给出 **Src MN Layout**（$s$ 的逆）与 **Dst MN Layout**（$d$ 的逆），**左右两侧相同坐标对应同一份数据**。本例两张图完全相同，且与 TiledMMA 的 A TV Layout 相同，说明：

$$\text{无线程间交换} \iff s = d \iff S = D$$

这也与 metadata 里 `SrcLayout` / `DstLayout` 完全一致相印证。

## `partition_S` 与 `partition_D`

```cpp
Tensor tAgA = thr_copy.partition_S(gA);
Tensor tAsA = thr_copy.partition_D(sA);
```

读法是：

```text
tAgA：当前线程从 global A 读取的 source 槽位
tAsA：当前线程向 shared A 写入的 destination 槽位
```

其中 `S` 是 Source，`D` 是 Destination。两个 partition 使用同一套线程和值布局，使 source 与 destination 的逻辑槽位可以对应：

```cpp
for (每个被 tid 分配的 value) {
    tAsA[value] = tAgA[value];
}
```

真正的实现由 `cute::copy(tiled_copy, tAgA, tAsA)` 展开；partition 本身主要创建视图。

## 命名约定

CUTLASS 代码常用以下形式帮助读者恢复数据流：

```text
t + partition pattern + storage + logical object
```

例如：

| 名字 | 含义 |
|---|---|
| `gA` | global-memory A tile |
| `sA` | shared-memory A tile |
| `rA` | register A fragment |
| `tAgA` | `tA` copy partition applied to `gA` |
| `tAsA` | `tA` copy partition applied to `sA` |
| `tArA` | 与 `tA` 对应的 register fragment |

这套名字是约定而不是语言规则。遇到不熟悉的变量，必须结合构造 API、来源 Tensor 和后续操作确认含义。

## FlashAttention 中的 Q/K/V 实例

`flash_fwd_kernel.h` 中的代码是同一模式在实际 attention kernel 中的应用：

```cpp
typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
```

其中：

- `gQ/gK/gV` 是 global-memory 中当前 CTA 的 tile；
- `sQ/sK/sV` 是 shared-memory staging tile；
- `gmem_thr_copy_QKV` 是 QKV copy scheme 中当前线程的 slice；
- `tQgQ`、`tKgK`、`tVgV` 是当前线程要读取的 source 槽位；
- `tQsQ`、`tKsK`、`tVsV` 是当前线程要写入的 destination 槽位。

因此 Q 的搬运可读成：

```text
gQ --partition_S--> tQgQ --copy--> tQsQ <--partition_D-- sQ
```

K 和 V 完全类似。`Q/K/V` 是逻辑对象，`g/s` 是存储层次；不要把第二个字母误解成固定的 GEMM A/B/C 语法。

## 边界谓词：partition 不负责判断合法性

一个编译期 tile 可能覆盖运行时矩阵之外的尾行或尾列。常见做法是并行创建数据 view 与坐标 view：

```cpp
Tensor tOgO = thr_copy.partition_D(gO);
Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));
Tensor tOcO = thr_copy.partition_D(cO);
```

因为二者用了同一个 thread slice 和 destination partition，`tOcO` 的每个槽位与 `tOgO` 的目标地址一一对应。代码可以根据坐标判断行列是否小于实际尺寸，再调用 predicated copy。

职责分离如下：

```text
partition：决定本线程可能负责哪些槽位
identity tensor：告诉这些槽位的逻辑坐标
predicate：决定当前运行时尺寸下哪些槽位真正合法
copy：执行通过谓词检查的搬运
```

## 常见误区

1. **`partition_D` 会立刻搬数据**：错，它主要建立 destination view。
2. **所有线程平均各拿一个标量**：错，向量化 copy 中一个线程可能拥有连续多个元素，布局还可能跨多个 tile 重复。
3. **`128` 一定是 128 threads**：错，向量化对齐参数和线程数是两件事。
4. **global/shared 必须使用相同物理 layout**：错，逻辑对应可以相同，物理 layout 往往为合并访问、bank conflict 和 MMA 要求而不同。
5. **partition 名字决定 C++ 类型**：错，真正语义来自 layout、engine 和 API。
6. **“tile 太大装不进寄存器”**：错，寄存器只限制单线程同时持有的值数（≈ `prod(value_layout)`）；大 tile 靠线程数和 Rest/时间维分步消化。
7. **`value_layout` 取值很随意**：错，它决定每线程寄存器 footprint 与向量化是否成立（如 128-bit 指令要求 4 个 float 对齐，否则编译期 static fail）。
8. **`SrcLayout` / `DstLayout` 的 codomain 是元素序号**：错，是 **bit 偏移**；元素单位由 `Copy_Atom` 用 `recast_layout<uint1_t, ValType>` 折算得到。
9. **`RefLayout` 是独立于 `SrcLayout`/`DstLayout` 的第三张表**：错，它**就是**其中之一（`ldmatrix` 选了 `DstLayout`）。
10. **`partition_S` 与 `retile_S` 只是同一件事的两个名字**：错。`partition_S` 做完整复合映射并选线程分块；`retile_S` 只是**重排已有分块的 Layout**，不做坐标空间变换。

## 我的理解

TiledCopy 的难点不是“复制”本身，而是把一个 copy 操作拆成三个可验证问题：整个 CTA 的覆盖范围、当前线程的责任、当前 value 的合法坐标。读懂这三个层次后，复杂的 `partition_S/D` 就不再像隐藏的魔法索引。

我目前的理解还可以再加一层：thread/value layout 本质是在给"槽位归属"建模——`thread_layout` 分线程、`value_layout` 分值，覆盖总数满足 $M \times N = \#threads \times \text{values/thread}$，而寄存器只约束后一个因子。这样"tile 尺寸、线程数、每线程值数、occupancy"就不再是四个孤立问题，而是同一道预算题。

再往下还有一层是本文补上的：**`partition_S/D` 是"复合"而不是"切片"**。真正的机制是用数据 ID 当桥梁，把 src 和 dst 的 $(t,v)$ 空间经 $R^{-1}$ 对齐到同一个参考系，再套一套 Ref TV Layout。理解了这一点，`make_tiled_copy_A(atom, tiled_mma)` 这种"直接复用 MMA 布局"的写法就不再神秘——它只是把 MMA 的 TV Layout 拿来做 $r$ 而已。

源码路径为 `third_party/cutlass` 本地检出（源码核对到 `copy_atom.hpp` 与 `media/docs/cpp/cute/` 文档）。

## Related

- [CUTLASS/CuTe 09：TiledCopy 核心原理（S/D/R Layout 与 Ref TV Layout）](./09-cute-tiled-copy-principle.md) — 本笔记的数学原理层
- [CUTLASS/CuTe 01：Tensor、Layout 与坐标映射](./01-cute-tensor-and-layout.md)
- [CUTLASS/CuTe 03：TiledMMA 与 fragment](./03-cute-tiled-mma.md)
- [CUTLASS/CuTe 04：从 global → shared → MMA 串起 GEMM](./04-cute-gemm-pipeline.md)
- [CUTLASS/CuTe 05：Copy 规模核算与 128-bit 向量化](./05-cute-copy-scaling-and-vectorization.md)
- [GPU 全局内存访存模型：向量化与合并访存](../gpu-memory-access-model.md) — 布局选择为什么影响实际访存量
- [CUTE 入门：FlashAttention 中的 GmemTiledCopyO 与边界处理](../flash-attention/cute-basics.md)

## References

- `third_party/cutlass/include/cute/atom/copy_atom.hpp`（`make_tiled_copy`、`TiledCopy`、`ThrCopy`、`Copy_Atom::recast_layout`）
- `third_party/cutlass/include/cute/atom/copy_traits.hpp`（`Copy_Traits` 的 `SrcLayout` / `DstLayout` / `RefLayout`）
- `third_party/cutlass/include/cute/layout.hpp`（`blocked_product` / `raked_product`）
- `third_party/cutlass/include/cute/algorithm/copy.hpp`
- `third_party/cutlass/media/docs/cpp/cute/02_layout_algebra.md`（logical/blocked/raked product）
- `third_party/cutlass/media/docs/cpp/cute/0x_gemm_tutorial.md`（TiledCopy 教学例：32×8 线程 × 4×1 值）
- `third_party/flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h`
- `third_party/flash-attention/csrc/flash_attn/src/utils.h`
- 知乎《CUTLASS 笔记 (4)：Tiled Copy》第 3 节"Tiled Copy 实现"，作者杨远航，CUTLASS 4.1.0 / SM90：<https://zhuanlan.zhihu.com/p/1968745447741972494>
- [CUTLASS GitHub](https://github.com/NVIDIA/cutlass)
