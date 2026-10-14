---
title: 整数模 n 加法群与二面体群
type: concept
status: growing
tags: [Mathematics, Group Theory, Modular Arithmetic, Dihedral Group]
created: 2026-10-14
updated: 2026-10-14
source: 知乎《群论及抽象代数学习笔记（2）- 一些群的介绍（上）》（大PPP，NTU MH2220）
---

# 整数模 n 加法群与二面体群

> 系列第二章上。从同余关系出发构造 $(\mathbb{Z}/n\mathbb{Z},+)$ 与 $((\mathbb{Z}/n\mathbb{Z})^\times,\times)$，再从正多边形对称构造二面体群 $D_n$。原文：<https://zhuanlan.zhihu.com/p/604295896>

## 一句话理解

钟表算术给出第一批有限群（模 n 加法/乘法群），多边形对称给出第一批非阿贝尔群（$D_n$）——"等价类"与"生成元+关系"是构造群的两种基本手法。

## 什么是它

### 同余与商集

**定义 2.1**：$a\equiv b \bmod n \iff a=b+kn$（某个整数 $k$）。等价说法：余数相同 / $n\mid a-b$。

**推论 2.2**：模 $n$ 同余是等价关系（自反、对称、传递）。

**定义 2.6**：等价类 $\bar a=\{b\in\mathbb{Z}:a\equiv b\bmod n\}$；所有不同等价类的集合记为 $\mathbb{Z}/n\mathbb{Z}$。例如 $\mathbb{Z}/2\mathbb{Z}=\{\bar0,\bar1\}$。

特例：$n=0$ 时 $\mathbb{Z}/0\mathbb{Z}\cong\mathbb{Z}$（逐元素）；$n=1$ 时 $\mathbb{Z}/1\mathbb{Z}=\{\bar0\}$ 只有一个元素。

### 模 n 加法群

**命题 2.8**：定义 $\bar a+\bar b=\overline{a+b}$，则 $(\mathbb{Z}/n\mathbb{Z},+)$ 是阿贝尔群。

证明里最重要的一步是**良定义性（well-defined）**：若换代表元 $\bar a=\bar c,\bar b=\bar d$，则 $a+b=c+d+(k+l)n$，故 $\overline{a+b}=\overline{c+d}$。即在等价类上定义运算必须验证"结果与代表元选取无关"。

### 模 n 乘法群

先看反例：$(\mathbb{Z}/2\mathbb{Z},\times)$ 不是群——$\bar0$ 没有逆元。

**定义 2.11 / 命题 2.12**：取其中有逆元的部分
$$(\mathbb{Z}/n\mathbb{Z})^\times=\{\bar a:\bar a\bar b=\bar 1\ \text{某个}\ \bar b\}=\{\bar a:\gcd(a,n)=1,\ 0\le a\le n-1\},$$
则 $((\mathbb{Z}/n\mathbb{Z})^\times,\times)$ 是阿贝尔群。

证明用 Bézout：$\gcd(a,n)=1\iff\exists b,c,\ ab+nc=1\iff \bar a\bar b=\bar1$。

例：$p$ 为质数时 $|(\mathbb{Z}/p\mathbb{Z})^\times|=p-1$。

### 二面体群

**背景**：正 $n$ 边形的全部对称（保持图形的平面等距变换）——$n$ 个旋转 $1,r,\dots,r^{n-1}$（$r$ 为 $2\pi/n$ 逆时针旋转）与 $n$ 个翻转 $m,rm,\dots,r^{n-1}m$（$m$ 为某对称轴翻转）。

**命题 2.17**：这些对称在复合下成群，且满足关系
$$r^n=1,\qquad m^2=1,\qquad mr=r^{-1}m,$$
于是每个元素都可写成 $r^i m^j$（$0\le i<n$，$j\in\{0,1\}$），$|D_n|=2n$：
$$D_n=\langle r,m: r^n=1=m^2,\ mr=r^{-1}m\rangle.$$

证明骨架：任何对称 $H$ 把顶点 $w_0$ 送到 $w_k$ 后，$w_1$ 只能去 $w_{k+1}$（纯旋转 $r^k$）或 $w_{k-1}$（翻转 $r^km$），二选一穷尽所有可能；$mr=r^{-1}m$ 用复数坐标 $w^k\mapsto e^{2\pi ik/n}$ 逐点验证。

> **为什么这是非阿贝尔群的第一课**：$mr=r^{-1}m\ne rm$（$n\ge3$ 时）——翻转和旋转"不能交换，但交换有代价公式"。$D_n$ 里所有运算都能机械化化简，正是"生成元 + 关系"表示法的示范。

## 为什么重要

- $\mathbb{Z}/n\mathbb{Z}$ 是**良定义性检查**的标准训练场——后续商群（[06](./normal-subgroup-and-quotient-group.md)）、同构定理（[08](./isomorphism-theorems-I.md)）里每个"在等价类上定义的映射"都要过这一关
- $(\mathbb{Z}/n\mathbb{Z})^\times$ 把数论（gcd、Bézout）变成群论对象，后面 $Aut(C_n)\cong(\mathbb{Z}/n\mathbb{Z})^\times$（[07](./automorphism-and-products.md)）把两者接通
- $D_n$ 是最小的非阿贝尔群家族，也是[魔方群](../rubiks-cube-group-theory.md)"转动生成对称"的最简模型

## 局限 / 边界

- 原文证明 $D_n$ 元素个数时默认"对称由顶点像决定"——这依赖"等距变换由不共线三点像决定"，正文未展开
- $n\ge 3$ 的约束是必要的：$n=2$ 时正二边形退化，讨论无意义

## 我的理解

- "良定义性"第一印象是形式主义，其实是**商结构可用性的全部门槛**：$\mathbb{Z}/n\mathbb{Z}$ 里 $\times$ 对全体类不构成群（$\bar 0$ 拖后腿），必须筛出可逆类——"结构保留多少，取决于等价关系多粗"
- $mr=r^{-1}m$ 说明非阿贝尔不等于"无序"：交换的**代价是可计算的**（$r$ 变 $r^{-1}$）。魔方里 $F R\ne R F$ 但 $FR=RF^{-1}$ 型的代价公式同样存在

## Related

- [01 群论导论](./01-introduction-to-group-theory.md) — 子群判定引理
- [03 对称群与循环群](./symmetric-and-cyclic-groups.md) — $\langle r\rangle=\{1,r,\dots,r^{n-1}\}$ 展开
- [07 自同构、直积与半直积](./automorphism-and-products.md) — $Aut(D_3)\cong D_3$
- [魔方群的数学结构](../rubiks-cube-group-theory.md) — 更大的"对称生成群"实例
