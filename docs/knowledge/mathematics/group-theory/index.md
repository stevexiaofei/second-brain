---
title: 群论与抽象代数专题
type: index
status: growing
tags: [Mathematics, Group Theory, Abstract Algebra, NTU MH2220, Permutation, Rubiks Cube]
created: 2026-10-07
updated: 2026-10-07
source: 知乎《群论及抽象代数学习笔记》系列（作者大PPP，NTU MH2220 Algebra I 课程笔记）；另含本站自写四篇（群论基础、置换的符号与行列式、魔方群的数学结构、魔方层先法实操）
---

# 群论与抽象代数专题

## 一句话理解

群是"具有满足某些良好性质的二元运算的集合"。本目录有两条互补的线：**课程线（01–09）** 按一门完整课程（抽代 1）的顺序走完"公理 → 标准群 → 同态/同构 → 拉格朗日 → 商群 → 直积/半直积 → 同构定理 → 有限阿贝尔群分类"；**自写线（4 篇）** 是一份更紧凑的参考手册，并把抽象定义落到置换群与魔方群这两个具体载体上。

## 为什么单独建立这个专题

- [魔方群的数学结构](./rubiks-cube-group-theory.md) 已经用到了大量群论概念（置换群、正规子群、交换子、共轭、商群），但当时缺少一份系统的前置知识沉淀——这个专题补齐它
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
应用层     置换符号 ↔ 行列式、魔方群、层先法                (自写 1–4)
```

## 课程线：推荐阅读顺序

1. [01 群论导论](./01-introduction-to-group-theory.md)：二元运算、群的三公理、单位元/逆元唯一性、阶、子群判定引理、生成子群
2. [02 整数模 n 加法群与二面体群](./02-mod-n-and-dihedral-groups.md)：同余 → 等价类 → $\mathbb{Z}/n\mathbb{Z}$；正多边形的对称群 $D_n$
3. [03 对称群与循环群](./03-symmetric-and-cyclic-groups.md)：排列与 cycle 分解、cycle type、$o(\sigma)=\operatorname{lcm}$、循环群初步
4. [04 群的性质（上）](./04-group-properties-I.md)：同态/同构、循环群分类（只有 $\mathbb{Z}$ 与 $\mathbb{Z}/n\mathbb{Z}$）、置换矩阵与 sgn
5. [05 群的性质（下）](./05-group-properties-II.md)：交错子群 $A_n$、陪集、拉格朗日定理、阶 6 分类、Cayley 定理
6. [06 正规子群与商群](./06-normal-subgroup-and-quotient-group.md)：正规性、商群良定义性、$\mathbb{R}/\mathbb{Z}\cong$ 单位圆、正规化子与中心化子
7. [07 自同构、直积与半直积](./07-automorphism-and-products.md)：$Aut(C_n)\cong(\mathbb{Z}/n\mathbb{Z})^\times$、内/外直积、半直积与 $S_n=A_n\rtimes\langle(1,2)\rangle$
8. [08 同构定理（上）](./08-isomorphism-theorems-I.md)：核是正规子群、交换图提升引理、第一/二/三同构定理
9. [09 同构定理（下）与有限阿贝尔群分类](./09-isomorphism-theorems-II.md)：子群对应定理、有限阿贝尔群分类、阶 8 群的完全分类

## 自写线：参考手册

按"抽象 → 具体"的依赖顺序排列：

1. [群论基础](./group-theory-basics.md) — 起点。群与子群、陪集与拉格朗日定理（含四条推论的完整证明）、同态与核、正规子群与商群（含"良定义"论证）、群作用与凯莱定理、交换子与换位子群。后三篇的所有工具都在这里定义。
2. [置换的符号与行列式](./permutation-sign-and-determinant.md) — 第一个具体载体 $S_n$。逆序数定义、对换翻转奇偶性、$k$-循环符号公式 $(-1)^{k-1}$、莱布尼茨公式，以及核心同一性 $\operatorname{sgn}(\sigma)=\det(P_\sigma)$（群论与线性代数的交汇点）。
3. [魔方群的数学结构](./rubiks-cube-group-theory.md) — 最大的应用实例。状态空间 $\lvert G\rvert=\frac{8!\,12!}{2}3^72^{11}\approx4.33\times10^{19}$、三个守恒律（奇偶同步 / 棱块翻转 / 角块扭转）的完整证明、交换子与共轭构造、上帝之数 20。
4. [魔方层先法实操指南](./rubiks-cube-solving-guide.md) — 落地。七步入门解法、公式速查、常见卡点；最后回到理论解释"每个公式为什么只做它该做的事"。

### 怎么选

- **只想过一遍数学**：自写线 1 → 2
- **想跟着课程系统学**：课程线 01 → 09
- **想理解魔方为什么转不出某些状态**：自写线 1 → 3
- **只想学会还原魔方**：直接读自写线 4，不需要群论

## 知识地图

| 关键问题 | 对应笔记 |
|---|---|
| 什么是群？单位元/逆元为什么唯一？ | [01](./01-introduction-to-group-theory.md) |
| 怎么判断一个子集是不是子群？ | [01](./01-introduction-to-group-theory.md)（引理：非空 + 封闭 + 逆元封闭） |
| 钟表算术为什么构成群？ | [02](./02-mod-n-and-dihedral-groups.md) |
| 正多边形的对称有几个？怎么用 $r,m$ 生成？ | [02](./02-mod-n-and-dihedral-groups.md) |
| 排列怎么写、怎么乘、怎么求阶？ | [03](./03-symmetric-and-cyclic-groups.md) |
| 为什么循环群只有两种？ | [04](./04-group-properties-I.md) |
| 排列的"奇偶性"是什么？为什么 $\operatorname{sgn}$ 是同态？ | [04](./04-group-properties-I.md) |
| 为什么 $\lvert H\rvert$ 整除 $\lvert G\rvert$？ | [05](./05-group-properties-II.md) |
| 任意有限群都藏在某个 $S_n$ 里？ | [05](./05-group-properties-II.md)（Cayley 定理） |
| 为什么只有正规子群才能造商群？ | [06](./06-normal-subgroup-and-quotient-group.md) |
| $\mathbb{R}/\mathbb{Z}$ 是什么？ | [06](./06-normal-subgroup-and-quotient-group.md)（单位圆） |
| $D_n$ 能拆成旋转 × 翻转吗？ | [07](./07-automorphism-and-products.md)（半直积） |
| $G/\ker\phi\cong\operatorname{im}\phi$ 怎么证、怎么用？ | [08](./08-isomorphism-theorems-I.md) |
| $G/N$ 的子群长什么样？ | [09](./09-isomorphism-theorems-II.md) |
| 有限阿贝尔群为什么由阶完全决定？ | [09](./09-isomorphism-theorems-II.md) |
| $\operatorname{sgn}$ 和行列式是什么关系？ | [置换的符号与行列式](./permutation-sign-and-determinant.md) |
| 魔方为什么有状态转不出来？ | [魔方群的数学结构](./rubiks-cube-group-theory.md) |

## 与已有笔记的连接

- [魔方群的数学结构](./rubiks-cube-group-theory.md)：$G/\ker\pi$ 的论证（同态基本定理求 $|G|$）正是 [08 第一同构定理](./08-isomorphism-theorems-I.md) 的直接应用；$A_8,A_{12}$ 来自 [05 交错子群](./05-group-properties-II.md)；交换子/共轭在 [07](./07-automorphism-and-products.md) 中以"内自同构"的面目再次出现
- [魔方层先法实操](./rubiks-cube-solving-guide.md)：层先法的"七步"结构对应"把大群按正规子群链逐步约化"的思想
- [奇异值分解 SVD](../singular-value-decomposition.md)：同构视角下"换坐标系不改变结构"的另一个实例
- [Mathematics 总览](../index.md)：本目录的父级索引
- [Lagrangian 与约束优化](../lagrangian-and-constrained-optimization.md)：同样体现"结构约束决定可行域"的思维，可与群论的"守恒律限制可达状态"对照

## 整理原则

- 课程线原文是他人笔记（作者大PPP），沉淀时区分**事实**（定义、定理、证明）与**我的理解**（标注"我的理解"小节）
- 每篇保留：定义/定理/证明骨架 + 一句话动机 + 与本库其他笔记的链接
- 练习题在原文中是图片，无法转为文本，未搬运；需要时可回到原文练习
- 原文中的示意图（Cayley 表、多边形对称等）以 `[图]` 占位，不影响正文证明链完整性

## Open Questions

- 把"所有可想象的构型"记为 $W=(\mathbb{Z}_3\wr S_8)\times(\mathbb{Z}_2\wr S_{12})$（角块/棱块各自的位置置换 + 朝向），则 $\lvert W\rvert=3^8\,8!\cdot2^{12}\,12!$，而 $\lvert G\rvert=3^7\,8!\cdot2^{11}\,12!/2$，故 $[W:G]=12=3\times2\times2$——**恰好**是三个守恒律的乘积，说明三者独立且已穷尽所有限制。那么能否从"转动不改变块自身朝向"这一物理事实**直接**推出这三个守恒律，而不必逐个构造证明？
- 同态 $\operatorname{sgn}$ 与 $\det$ 之外，还有哪些"到 $\{\pm1\}$ 的同态"是物理/工程中有意义的守恒量？
- Kociemba 两阶段算法利用的子群链 $G\supseteq G_1\supseteq G_2$ 与一般的"子群塔 + Schreier–Sims"算法之间能否互相启发？

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
