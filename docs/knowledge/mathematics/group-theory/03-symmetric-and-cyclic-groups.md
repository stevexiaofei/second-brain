---
title: 对称群与循环群
type: concept
status: growing
tags: [Mathematics, Group Theory, Symmetric Group, Cyclic Group, Permutation]
created: 2026-10-14
updated: 2026-10-14
source: 知乎《群论及抽象代数学习笔记（3）- 一些群的介绍（下）》（大PPP，NTU MH2220）
---

# 对称群与循环群

> 系列第二章下。对称群 $S_n$（排列、cycle 记号、cycle 分解、阶）与循环群初步（$\langle g\rangle$、$o(g^k)=n/\gcd(n,k)$）。原文：<https://zhuanlan.zhihu.com/p/604557952>

## 一句话理解

$S_n$ 把"重排 $n$ 个对象"变成群，cycle 记号让排列可读可乘可求阶；循环群 $\langle g\rangle$ 是"一个元素的全部幂"，是最简单也最基本的一类群。

## 什么是它

### 对称群

**定义 2.18**：集合 $X$ 的**排列**（permutation）是双射 $X\to X$；$X=\{1,\dots,n\}$ 时全体排列记为 $S_n$，$|S_n|=n!$。

**命题 2.21**：$(S_n,\circ)$ 是群（复合为运算，恒等映射为单位元，逆映射为逆元）。

### cycle 记号

**定义 2.22**：cycle $(a_1,\dots,a_k)$ 是排列 $\sigma(a_1)=a_2,\ \sigma(a_2)=a_3,\ \dots,\ \sigma(a_k)=a_1$，其余元素不动；$k$ 称为长度。同一 cycle 可以轮转书写：$(1,2,3)=(2,3,1)=(3,1,2)$。

cycle 相乘即复合，**从右往左**作用。

**例 2.25（分解算法）**：跟踪每个元素的轨道：$1\to5\to3\to2\to1$ 与 $7\to8\to7$，不动点略去：
$$\sigma=(1,5,3,2)(4)(6)(7,8)=(1,5,3,2)(7,8).$$

**命题 2.26**：任何排列都能写成**不相交 cycles 之积**；不相交 cycles 互相交换。

证明用等价关系 $a\sim b\iff\sigma^k(a)=b$ 把 $X$ 切成轨道，每条轨道给出一个 cycle。

**定理 2.28（排列的阶）**：
1. $k$-cycle 的阶是 $k$；
2. 不相交 cycles 积 $\sigma=\sigma_1\cdots\sigma_m$（各 $\sigma_i$ 长 $\ell_i$）满足
$$o(\sigma)=\operatorname{lcm}(\ell_1,\dots,\ell_m).$$

> cycle type（把长度排成不增序列）是排列的共轭不变量——两个排列共轭当且仅当 type 相同。这是 [魔方群](../rubiks-cube-group-theory.md) 里"块的位置怎么排"的全部语言。

### 循环群初步

**定义 2.30**：$g\in G$ 生成的子集 $\langle g\rangle=\{g^k:k\in\mathbb{Z}\}$。

**推论 2.32**：$\langle g\rangle$ 是子群；且若 $o(g)=m$，则
$$\langle g\rangle=\{1,g,\dots,g^{m-1}\},\qquad |\langle g\rangle|=o(g).$$
证明：$k=lm+r$（带余除法）$\Rightarrow g^k=(g^m)^l g^r=g^r$，幂按周期重复。

**定义 2.34**：$G=\langle g\rangle$ 时称 $G$ 为**循环群**。例：$\mathbb{Z}=\langle 1\rangle=\langle-1\rangle$（生成元不唯一）；$\mathbb{Z}/n\mathbb{Z}=\langle\bar1\rangle$。

**命题 2.38**：循环群必为阿贝尔群（$g^kg^l=g^{k+l}=g^lg^k$）。

**定理 2.39**：有限群 $G$（$|G|=n$）是循环群 $\iff$ 存在 $g$ 使 $o(g)=n$。

**定理 2.41（幂的阶公式）**：$G=\langle g\rangle$，$|G|=n$，则
$$o(g^k)=\frac{n}{\gcd(n,k)}.$$
证明：$d=\gcd(n,k)$ 时 $(g^k)^{n/d}=1$ 给出上界；Bézout 恒等式 $sn+tk=d$ 给出下界。

例：$|G|=6$ 时 $o(g^2)=3,\ o(g^3)=2,\ o(g^4)=3,\ o(g^5)=6$——阶为 6 的循环群有 $\varphi(6)=2$ 个生成元。

## 为什么重要

- $S_n$ 是"所有有限群的容器"——Cayley 定理（[05](./group-properties-II.md)）说任何有限群都嵌入某个 $S_n$
- $o(\sigma)=\operatorname{lcm}$ 与 $o(g^k)=n/\gcd(n,k)$ 是两个最常用的**计算工具**，出现在魔方元素阶（1260）的计算里
- 循环群是后面一切分类的"基元"：[循环群分类](./group-properties-I.md)（只有 $\mathbb{Z}$ 与 $\mathbb{Z}/n\mathbb{Z}$）→ [有限阿贝尔群分类](./isomorphism-theorems-II.md)（循环群直积）

## 局限 / 边界

- cycle 分解不唯一（轮转书写、不相交因子顺序都可换），唯一的是 type
- $S_n$ 对复合虽成群，但 cycle 相乘**不交换**——即使不相交才交换；交叠 cycles 乘积要逐点追踪（例 2.27：$\sigma\circ\tau=(1,6,3,5,2)(4)$）

## 我的理解

- "排列 = 双射"与"cycle = 轨道"是同一对象的两种视角：前者是函数视角（好复合），后者是动力系统视角（好看结构）；求阶公式是两种视角的合成
- $|\langle g\rangle|=o(g)$ 把"元素的阶"和"子群的大小"焊在一起——拉格朗日定理（$o(g)\mid|G|$）只是这一步加上"子群阶整除"的推论

## Related

- [01 群论导论](./01-introduction-to-group-theory.md) — $\langle S\rangle$ 的定义
- [02 整数模 n 加法群与二面体群](./mod-n-and-dihedral-groups.md) — $\mathbb{Z}/n\mathbb{Z}=\langle\bar1\rangle$
- [04 群的性质（上）](./group-properties-I.md) — 循环群分类与 sgn
- [05 群的性质（下）](./group-properties-II.md) — Cayley 定理、拉格朗日定理
- [魔方群的数学结构](../rubiks-cube-group-theory.md) — cycle type 与置换奇偶性的应用场
