---
title: TiledCopy 核心原理：S/D/R Layout 与 Ref TV Layout
type: concept
status: seed
tags: [AI, CUDA, CUTE, CUTLASS, TiledCopy, Layout, CopyAtom, TVLayout]
created: 2026-09-11
updated: 2026-09-11
source: 知乎《CUTLASS 笔记 (4)：Tiled Copy》第 2 节（https://zhuanlan.zhihu.com/p/1968745447741972494）+ 个人整理
---

# TiledCopy 核心原理：S/D/R Layout 与 Ref TV Layout

## 一句话理解

拷贝的本质是 $dst(i) = src(i)$，但 SIMT 下每个线程必须拿自己的分块，于是索引要写成 $(t, v)$ 二元组。CuTe 的做法是**以"数据 ID"作为唯一桥梁**，在 `CopyAtom` 里记录三个 Layout：

- **$S$（SrcLayout）**：$(src_t, src_v) \to ID$
- **$D$（DstLayout）**：$(dst_t, dst_v) \to ID$
- **$R$（RefLayout）**：$(ref_t, ref_v) \to ID$，取 $S$ 或 $D$ 之一，用来把 src 和 dst 的 $(t,v)$ **统一到同一个空间**

有了 $R$，就能构造出复合映射 $src' = src \circ r \circ R^{-1} \circ S$ 与 $dst' = dst \circ r \circ R^{-1} \circ D$ —— **这就是 `partition_S` / `partition_D` 的全部原理**。

## 为什么重要

- 它把 `partition_S` / `partition_D` / `retile_S` / `retile_D` 从"魔法索引"变成一条可推导的复合链。
- 它解释了为什么 `TiledCopy` 必须记录 `TiledLayout_TV`：那是 $r$ 的载体。
- 它给出了理解 **ldmatrix 这类会做线程间交换的指令**的正确框架 —— 对它们来说 $(src_t,src_v) \to (dst_t,dst_v)$ 不再是恒等映射。
- 这是走向"用 CuTe Layout 代数写算子"的关键一步：一旦明白 partition 只是复合，就能自己构造布局而不是抄模板。

## 第一步：拷贝的本质

所有数据拷贝归根到底是一行：

```cpp
dst = src;        // 即 *dst_ptr = *src_ptr;
```

在 CuTe 中 Tensor = **Data 头指针 + Layout**。知道了这两者，就知道 Tensor 中任意数据的地址，于是最朴素的拷贝就是按坐标遍历：

```cpp
copy(dst, src);

// 等价于
for (int i = 0; i < size(src); ++i) {
  dst(i) = src(i);
}
```

### 更一般的拷贝模式

倒序拷贝、移位拷贝等都无法用 `copy(dst, src)` 表达。抽象一下，它们都建立了 src 坐标到 dst 坐标的映射 $f$：

$$dst\_idx = f(src\_idx)$$

```cpp
for (int i = 0; i < size(src); ++i) {
  dst(f(i)) = src(i);
}
```

**CuTe 的一个隐含假设是：$f$ 一定是恒等映射。** 因为总可以变换 dst 的映射，令 $dst' = dst \circ f$，把非恒等的 $f$ 吸收进 dst 本身：

```cpp
dst' = composition(dst, f);

for (int i = 0; i < size(src); ++i) {
  dst'(i) = src(i);
}
```

因此后续一律用 $dst(i) = src(i)$ 表示拷贝。

> 严格说，`ldmatrix` 之类会做线程间数据交换的指令无法简单建模成 $dst(i) = src(i)$，更准确的写法是 $copy\_instruction(dst(i), src(i))$ —— 把 src 的数据交给拷贝指令，指令返回的数据放到 dst。本文为表述方便仍写成 $dst(i) = src(i)$，但要注意**同一行代码在不同拷贝指令下底层机制不同**。

## 第二步：SIMT 下必须用 (t, v) 索引

每个线程要拿到各自的数据分块，所以不是直接按坐标索引 Tensor，而是用 $(t, v)$ 二元组：

```cpp
// 全局视角
for (int t = 0; t < size<0>(src); ++t)
  for (int v = 0; v < size<1>(src); ++v)
    dst(t, v) = src(t, v);

// 线程视角
int t = threadIdx.x;
src_frg = src(t, _);
dst_frg = dst(t, _);
copy(dst_frg, src_frg);      // for (v) dst_frg(v) = src_frg(v);
```

关键问题变成：**如何构造 src 和 dst，使得 $src(t,v) = dst(t,v)$ 就能完成拷贝？**

答案需要两个 TV Layout：

| Layout | 含义 |
|---|---|
| $s$（Src TV Layout） | $s(src\_t, src\_v) = src\_idx$，每个线程从 src 的哪个坐标读 |
| $d$（Dst TV Layout） | $d(dst\_t, dst\_v) = dst\_idx$，每个线程把数据写到 dst 的哪个坐标 |

## 第三步：桥梁只能是"数据 ID"

常规思路是 $(src_t, src_v)$ 经 $s$ 得到 $src\_idx$、$(dst_t, dst_v)$ 经 $d$ 得到 $dst\_idx$。但**怎么能保证对同一个 $(t,v)$，两个映射得到的 idx 相同？**

**这个桥梁必然是数据本身**，理由：拷贝不改变数据 —— src 里的每份数据都能在 dst 里找到对应，反之亦然。无论存储介质、地址、内存是否连续对齐，都不会改变这个特征。所以建立 $(src_t, src_v)$ 与 $(dst_t, dst_v)$ 的联系时，中间一定是**具体的数据**，而不是坐标、offset、位置。

做法是给数据编序号（ID，$0 \sim N-1$），于是形成：

$$(src\_t, src\_v) \longleftrightarrow ID \longleftrightarrow (dst\_t, dst\_v)$$

而这个对应关系**由拷贝指令的本质特性决定** —— 只有指令能确定"哪个线程的哪个数据能搬到哪个线程的哪个数据"，程序无法改变。所以这两张映射表必须记录在 `CopyAtom` 里：

```text
SrcLayout  S: (src_t, src_v) -> ID
DstLayout  D: (dst_t, dst_v) -> ID
```

那么 $(src_t, src_v) \to (dst_t, dst_v)$ 的映射就是复合函数 $D^{-1} \circ S$（同理 $S^{-1} \circ D$ 反方向）。

> **什么时候这个复合不是恒等映射？** 常规的 $dst = src$ 类拷贝不会在线程间交换数据，同一个线程取到的数据在搬运过程中始终不变，此时 $(src_t, src_v) = (dst_t, dst_v)$，两个复合都是恒等映射。但 NV GPU 的 **`ldmatrix`** 等指令会做线程间交换 —— 某线程在 $(src_t, src_v)$ 拿到的数据最终会被**另一个**线程在 $(dst_t, dst_v)$ 拿到，此时复合**不是**恒等映射。

## 第四步：用 Ref Layout 统一 (t, v) 空间

只要把 $(src_t, src_v)$ 映射到 $(dst_t, dst_v)$（或反向），两者就能共用同一套 TV Layout。但选哪个方向都行，实际取决于**哪个 TV Layout 更容易获取或计算**。

CuTe 定义了一个映射 $R$，取 $S$ 与 $D$ 之一：

$$R(ref\_t, ref\_v) = ID$$

- $R = S$ 时：$R(src\_t, src\_v) = ID$，也就是 Src TV Layout；
- $R = D$ 时：$R(dst\_t, dst\_v) = ID$，也就是 Dst TV Layout。

把 $R^{-1}$ 分别与 $S$、$D$ 复合：

$$\underbrace{R^{-1} \circ S}_{(src\_t,src\_v) \to (ref\_t,ref\_v)}, \qquad \underbrace{R^{-1} \circ D}_{(dst\_t,dst\_v) \to (ref\_t,ref\_v)}$$

两个 $(t,v)$ 就都落到**同一个 $(ref\_t, ref\_v)$ 空间**了。此时：

- $R = S$：$(src_t,src_v)$ 走的是恒等映射，$(ref_t, ref_v) = (src_t, src_v)$，需要记录 **Src TV Layout** 把 $(src_t,src_v)$ 映射为 idx；
- $R = D$：两个都变成 $(dst_t, dst_v)$，$(ref_t, ref_v) = (dst_t, dst_v)$，需要记录 **Dst TV Layout**。

记录下来的这套 TV Layout 称为 **Ref TV Layout**，记作 $r$（等于 $s$ 或 $d$ 之一）。不同拷贝指令下 $r$ 的获取难度不同，所以 $R$ 到底取 $S$ 还是 $D$ **也和拷贝指令相关** —— 因此 $R$ 与 $S$、$D$ 一样被记录在 `CopyAtom` 中。

### 最终的复合映射

对 src：从 $(src_t, src_v)$ 出发，经 $R^{-1} \circ S$ 到 $(ref_t, ref_v)$，再经 Ref TV Layout 得到 idx，最后交给 Src Tensor Layout 求出实际地址：

$$\boxed{src' = src \circ r \circ R^{-1} \circ S}$$

同理：

$$\boxed{dst' = dst \circ r \circ R^{-1} \circ D}$$

**这个复合 Layout 的构建过程，就是 `TiledCopy` 中 `partition_S` 和 `partition_D` 的原理。**

反过来也能用 $r$ 还原出两个 TV Layout：

$$s = r \circ R^{-1} \circ S, \qquad d = r \circ R^{-1} \circ D$$

当 $R = S$ 时有 $s = r$，即 **Ref TV Layout 就是 Src TV Layout**；$R = D$ 同理。

### 全流程串起来

```text
        (src_t, src_v)                          (dst_t, dst_v)
              │                                        │
              │  R⁻¹ ∘ S                    R⁻¹ ∘ D    │
              ▼                                        ▼
                       (ref_t, ref_v)   ← 统一空间
                              │
                              │  Ref TV Layout r        ← CopyAtom 提供
                              ▼
                             idx
                    ┌─────────┴─────────┐
                    ▼                   ▼
            Src Tensor Layout     Dst Tensor Layout
                    │                   │
                    ▼                   ▼
              src 地址偏移         dst 地址偏移
```

一句话：**用 ID 把"数据"这件事说清楚，用 $R$ 把两边的线程/值空间对齐，剩下的都交给 Layout 复合。**

## 我的理解

（以下为个人理解）

- 这套推导的巧妙之处在于**把"数据身份"和"数据位置"彻底分开**。$S$ / $D$ 描述的是"位置 ↔ 身份"，$R$ 描述的是"选哪个位置空间当参考系"。换存储介质、换布局，$S$/$D$ 变，但"数据身份"这条主线不变。
- 它同时解释了为什么 `TiledCopy` 只需要**一个** `TiledLayout_TV`：因为 src 和 dst 已经被 $R^{-1}$ 对齐到同一个空间了，一套 TV Layout 就够。
- `R = S` 还是 `R = D` 这个看似随意的选择，实际是**工程便利性**的取舍 —— 哪个布局在源码里拿得到就用哪个。所以 `RefLayout` 在不同 `Copy_Traits` 里会不一样（例如 LDSM 就选了 `DstLayout`）。
- 我理解 $S$/$D$ 里那个 "ID" 在源码里具体是 **bit 偏移**，`Copy_Atom` 再用 `recast_layout<uint1_t, ValType>` 把它折算成以元素为单位的 `ValLayout*`。本文按原文用 "ID" 表述便于理解，但读源码时要注意这个单位换算。

## 常见误区

1. **"`partition_S` 只是按线程切一刀"** → 错。它是 $src \circ r \circ R^{-1} \circ S$ 的复合，包含**坐标空间变换**，不只是切片。
2. **"$(src_t,src_v)$ 和 $(dst_t,dst_v)$ 总是一一对应"** → 只有不交换线程数据的指令才成立；`ldmatrix` 这类指令下映射非恒等。
3. **"$R$ 可以随便取 $S$ 或 $D$"** → 形式上都行，但取哪个由**拷贝指令**决定，所以它被写死在 `Copy_Traits` 里。
4. **"$r$ 是独立于 $s$/$d$ 的第三个布局"** → 错。$r$ **就是** $s$ 或 $d$，是同一份布局的另一个名字。
5. **"非恒等映射 $f$ 需要特殊处理"** → 不需要。`composition(dst, f)` 把它吸收进 dst 的 Layout，之后一律按 $dst(i) = src(i)$ 处理。

## Related

- [CUTLASS/CuTe 02：Copy Atom、TiledCopy 与线程分区](./02-cute-copy-and-thread-partition.md) — `Copy_Traits` 三个 Layout 的源码、`retile_S/D`、`make_tiled_copy_A/B/C`
- [CUTLASS/CuTe 01：Tensor、Layout 与坐标映射](./01-cute-tensor-and-layout.md) — Tensor = Data + Layout、composition 等 Layout 代数
- [CUTLASS/CuTe 05：Copy 规模核算与 128-bit 向量化](./05-cute-copy-scaling-and-vectorization.md) — TiledCopy 的规模与向量化约束
- [GPU 全局内存访存模型：向量化与合并访存](../gpu-memory-access-model.md) — 布局选择为什么会影响性能
- [CUTLASS/CuTe 03：TiledMMA 与 fragment](./03-cute-tiled-mma.md) — 与之对称的 MMA partition 机制

## References

- 知乎《CUTLASS 笔记 (4)：Tiled Copy》第 2 节"理解 TiledCopy 的核心原理"，作者杨远航，CUTLASS 4.1.0 / SM90：<https://zhuanlan.zhihu.com/p/1968745447741972494>
- `third_party/cutlass/include/cute/atom/copy_traits.hpp`（`Copy_Traits` 的 `SrcLayout` / `DstLayout` / `RefLayout`）
- `third_party/cutlass/include/cute/atom/copy_atom.hpp`（`Copy_Atom::recast_layout`、`TiledCopy`、`ThrCopy::partition_S/D`）
