---
title: 群论与抽象代数专题
type: index
status: growing
tags: [Mathematics, Group Theory, Abstract Algebra, NTU MH2220]
created: 2026-10-14
updated: 2026-10-14
source: 知乎《群论及抽象代数学习笔记》系列（作者大PPP，NTU MH2220 Algebra I 课程笔记）
---

# 群论与抽象代数专题

## 一句话理解

群是"具有满足某些良好性质的二元运算的集合"。这个专题按一门完整课程（抽代 1）的顺序沉淀群论基础：从群与子群的定义出发，经过几类标准群（模 n 加法群、二面体群、对称群、循环群），建立同态/同构语言，再由拉格朗日定理、商群与同构定理逐步深入到"用小块拼大块"的构造技术（直积、半直积）与最终产物（有限阿贝尔群分类）。

## 为什么单独建立这个专题

- [魔方群的数学结构](../rubiks-cube-group-theory.md) 已经用到了大量群论概念（置换群、正规子群、交换子、共轭、商群），但当时缺少一份系统的前置知识沉淀——这个专题补齐它
- 群论是"对称性的数学"，与线性代数（可逆矩阵)、数论（模运算）、拓扑（基本群）、物理（守恒律）共享同一套语言，是可长期复用的基础设施
- 该系列结构完整（公理 → 例子 → 性质 → 构造 → 分类定理），适合按原子化方式逐篇内化，而不是一篇大综述

## 全景图

```text
公理层     群 / 阿贝尔群 / 子群 / 阶                       (01)
例子层     Z/nZ、D_n、S_n、循环群                          (02, 03)
性质层     同态与同构、sgn、A_n、拉格朗日、Cayley            (04, 05)
构造层     正规子群、商群、正规化子/中心化子                 (06)
           自同构、直积、半直积                             (07)
定理层     三大同构定理                                     (08)
           子群对应定理、有限阿贝尔群分类、阶 8 的群          (09)
```

## 推荐阅读顺序

1. [01 群论导论](./01-introduction-to-group-theory.md)：二元运算、群的三公理、单位元/逆元唯一性、阶、子群判定引理、生成子群
2. [02 整数模 n 加法群与二面体群](./mod-n-and-dihedral-groups.md)：同余 → 等价类 → $\mathbb{Z}/n\mathbb{Z}$；正多边形的对称群 $D_n$
3. [03 对称群与循环群](./symmetric-and-cyclic-groups.md)：排列与 cycle 分解、cycle type、$o(\sigma)=\operatorname{lcm}$、循环群初步
4. [04 群的性质（上）](./group-properties-I.md)：同态/同构、循环群分类（只有 $\mathbb{Z}$ 与 $\mathbb{Z}/n\mathbb{Z}$）、置换矩阵与 sgn
5. [05 群的性质（下）](./group-properties-II.md)：交错子群 $A_n$、陪集、拉格朗日定理、阶 6 分类、Cayley 定理
6. [06 正规子群与商群](./normal-subgroup-and-quotient-group.md)：正规性、商群良定义性、$\mathbb{R}/\mathbb{Z}\cong$ 单位圆、正规化子与中心化子
7. [07 自同构、直积与半直积](./automorphism-and-products.md)：$Aut(C_n)\cong(\mathbb{Z}/n\mathbb{Z})^\times$、内/外直积、半直积与 $S_n=A_n\rtimes\langle(1,2)\rangle$
8. [08 同构定理（上）](./isomorphism-theorems-I.md)：核是正规子群、交换图提升引理、第一/二/三同构定理
9. [09 同构定理（下）与有限阿贝尔群分类](./isomorphism-theorems-II.md)：子群对应定理、有限阿贝尔群分类、阶 8 群的完全分类

## 知识地图

| 关键问题 | 对应笔记 |
|---|---|
| 什么是群？单位元/逆元为什么唯一？ | [01](./01-introduction-to-group-theory.md) |
| 怎么判断一个子集是不是子群？ | [01](./01-introduction-to-group-theory.md)（引理：非空 + 封闭 + 逆元封闭） |
| 钟表算术为什么构成群？ | [02](./mod-n-and-dihedral-groups.md) |
| 正多边形的对称有几个？怎么用 $r,m$ 生成？ | [02](./mod-n-and-dihedral-groups.md) |
| 排列怎么写、怎么乘、怎么求阶？ | [03](./symmetric-and-cyclic-groups.md) |
| 为什么循环群只有两种？ | [04](./group-properties-I.md) |
| 排列的"奇偶性"是什么？为什么 $\operatorname{sgn}$ 是同态？ | [04](./group-properties-I.md) |
| 为什么 $|H|$ 整除 $\|G\|$？ | [05](./group-properties-II.md) |
| 任意有限群都藏在某个 $S_n$ 里？ | [05](./group-properties-II.md)（Cayley 定理） |
| 为什么只有正规子群才能造商群？ | [06](./normal-subgroup-and-quotient-group.md) |
| $\mathbb{R}/\mathbb{Z}$ 是什么？ | [06](./normal-subgroup-and-quotient-group.md)（单位圆） |
| $D_n$ 能拆成旋转 × 翻转吗？ | [07](./automorphism-and-products.md)（半直积） |
| $G/\ker\phi\cong\operatorname{im}\phi$ 怎么证、怎么用？ | [08](./isomorphism-theorems-I.md) |
| $G/N$ 的子群长什么样？ | [09](./isomorphism-theorems-II.md) |
| 有限阿贝尔群为什么由阶完全决定？ | [09](./isomorphism-theorems-II.md) |

## 与已有笔记的连接

- [魔方群的数学结构](../rubiks-cube-group-theory.md)：$G/\ker\pi$ 的论证（同态基本定理求 $|G|$）正是 [08 第一同构定理](./isomorphism-theorems-I.md) 的直接应用；$A_8,A_{12}$ 来自 [05 交错子群](./group-properties-II.md)；交换子/共轭在 [07](./automorphism-and-products.md) 中以"内自同构"的面目再次出现
- [魔方层先法实操](../rubiks-cube-solving-guide.md)：层先法的"七步"结构对应"把大群按正规子群链逐步约化"的思想
- [奇异值分解 SVD](../singular-value-decomposition.md)：同构视角下"换坐标系不改变结构"的另一个实例

## 整理原则

- 原文是他人笔记（作者大PPP），沉淀时区分**事实**（定义、定理、证明）与**我的理解**（标注"我的理解"小节）
- 每篇保留：定义/定理/证明骨架 + 一句话动机 + 与本库其他笔记的链接
- 练习题在原文中是图片，无法转为文本，未搬运；需要时可回到原文练习
- 原文中的示意图（Cayley 表、多边形对称等）以 `[图]` 占位，不影响正文证明链完整性

## 原文链接

- (1) 群论导论：<https://zhuanlan.zhihu.com/p/603762948>
- (2) 一些群的介绍（上）：<https://zhuanlan.zhihu.com/p/604295896>
- (3) 一些群的介绍（下）：<https://zhuanlan.zhihu.com/p/604557952>
- (4) 群的性质（上）：<https://zhuanlan.zhihu.com/p/613909324>
- (5) 群的性质（下）：<https://zhuanlan.zhihu.com/p/614344709>
- (6) 群的构造（上）：<https://zhuanlan.zhihu.com/p/617509036>
- (7) 群的构造（下）：<https://zhuanlan.zhihu.com/p/617626701>
- (8) 同构定理（上）：<https://zhuanlan.zhihu.com/p/620855557>
- (9) 同构定理（下）完结篇：<https://zhuanlan.zhihu.com/p/620893105>
