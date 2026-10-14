---
title: 群的构造（上）：正规子群与商群、正规化子与中心化子
type: concept
status: growing
tags: [Mathematics, Group Theory, Normal Subgroup, Quotient Group, Normalizer, Centralizer]
created: 2026-10-14
updated: 2026-10-14
source: 知乎《群论及抽象代数学习笔记（6）- 群的构造（上）》（大PPP，NTU MH2220）
---

# 群的构造（上）：正规子群与商群、正规化子与中心化子

> 系列第四章上。回答"什么样的子群上可以造商群"，并给出度量"多不交换"的两个刻度：正规化子与中心化子。原文：<https://zhuanlan.zhihu.com/p/617509036>

## 一句话理解

正规子群 $gH=Hg$ 让"陪集相乘"与代表元选取无关，于是商集 $G/H$ 升级成商群；中心化子度量一个元素"有多被交换"，正规化子度量一个子群"有多被共轭稳定"。

## 什么是它

### 正规子群

**定义 4.1**：$H\triangleleft G\iff gH=Hg$ 对所有 $g\in G$。

> - $\{e\}$ 与 $G$ 总是正规的；阿贝尔群的一切子群都正规
> - $gH=Hg$ 是**集合相等**，不等于逐点交换（见下反例）

**定理 4.2**：$[G:H]=2\Rightarrow H\triangleleft G$。
证明：$g\in H$ 时 $gH=H=Hg$；$g\notin H$ 时左、右陪集都只能是 $G\setminus H$。例：$A_n\triangleleft S_n$。

**反例 4.3**：$A_3\triangleleft S_3$ 且 $(1,2)A_3=A_3(1,2)$，但 $(1,2)(1,2,3)\ne(1,2,3)(1,2)$——集合相等 $\ne$ 逐点相等。

**反例 4.4**：$H=\langle(1,2)\rangle\le S_3$ 不正规：$(1,3)H=\{(1,3),(1,2,3)\}\ne\{(1,3),(1,3,2)\}=H(1,3)$。

### 商群

**定理 4.6**：$H\triangleleft G$ 时，定义 $(xH)*(yH)=(xy)H$，则 $G/H$ 成群（单位元 $H$，逆元 $x^{-1}H$）。

证明核心还是**良定义**：$x=x'b,\ y=y'a$（$a,b\in H$）时
$$xy=x'(by')a=x'(y'c)a\in x'y'H,$$
其中 $by'=y'c$ 正是**用到正规性的一步**——没有 $gH=Hg$ 这一步就过不去。这也是"只有正规子群才有商群"的原因。

**例 4.7**：$S_n/A_n\cong C_2$（$n\ge2$）。

**例 4.11**：$G=(\mathbb{Z}/16\mathbb{Z})^\times$，$H=\langle\bar7\rangle=\{\bar1,\bar7\}$（$G$ 阿贝尔故 $H$ 正规），$G/H=\{H,\bar3H,\bar9H,\bar{11}H\}\cong C_4$。

**推论 4.12**：$G$ 阿贝尔 $\Rightarrow G/H$ 阿贝尔；$G$ 循环 $\Rightarrow G/H$ 循环（$G/H=\langle xH\rangle$）。

**例 4.13（$\mathbb{R}/\mathbb{Z}\cong$ 单位圆）**：$\psi(r+\mathbb{Z})=e^{2r\pi i}$ 是良定义双射同态：
- 良定义：$r-r'=n\in\mathbb{Z}$ 时 $e^{2r\pi i}=e^{2r'\pi i}e^{2n\pi i}=e^{2r'\pi i}$
- 直观：把数轴按整数周期"卷起来"，分数部分就是圆上的角度

### 正规化子与中心化子

**定义 4.14**：$H\le G$ 的**正规化子** $N_G(H)=\{g:gHg^{-1}=H\}$；$H\triangleleft G\iff N_G(H)=G$。

**定义 4.15**：$x\in G$ 的**中心化子** $C_G(x)=\{g:gx=xg\}$；对子集 $X$，$C_G(X)=\{g:gx=xg,\ \forall x\in X\}$；$X=G$ 时记 $Z(G)$，称 $G$ 的**中心**，且 $Z(G)=\bigcap_{x\in G}C_G(x)$。

## 为什么重要

- 商群是"把群除掉一个正规部分"的运算——所有同构定理（[08](./isomorphism-theorems-I.md)、[09](./isomorphism-theorems-II.md)）都是商群的性质；[魔方群](../rubiks-cube-group-theory.md) 求 $|G|$ 用的 $G/\ker\pi\cong P$ 就是一次商群操作
- 中心化子/正规化子是共轭作用的"稳定子"，为 Sylow 理论（未覆盖）与群作用计数铺路
- $C_{p^2}\ne C_p\times C_p$ 这类"同阶不同构"的分辨工作依赖商群与直积工具（[07](./automorphism-and-products.md)）

## 局限 / 边界

- 正规性是"可除性"的条件，不是"可交换性"的条件——商群 $G/H$ 阿贝尔只需要 $G$ 阿贝尔（或更弱条件），与 $H$ 内部交换与否无关
- $N_G(H)$、$C_G(x)$ 本身一般是子群但未必正规

## 我的理解

- 商群的困难全在良定义，而良定义的钥匙只有一把：$gH=Hg$。可以把正规子群理解为"商运算的许可证"——它保证你换代表元时运算结果不动
- $\mathbb{R}/\mathbb{Z}\cong S^1$ 是"商 = 卷绕"的最干净实例：模掉 $\mathbb{Z}$ 就是丢掉绝对刻度、只留相位。这个直觉对理解 $U(1)$、周期现象都有用

## Related

- [05 群的性质（下）](./group-properties-II.md) — 陪集与指数是商群的前置
- [07 自同构、直积与半直积](./automorphism-and-products.md) — $N\rtimes H$ 中 $N$ 必须正规
- [08 同构定理（上）](./isomorphism-theorems-I.md) — 商群性质的集中爆发
- [魔方群的数学结构](../rubiks-cube-group-theory.md) — $\ker\Phi$ 是正规子群、$G/G_0\cong\mathbb{Z}_3\times\mathbb{Z}_2\times\mathbb{Z}_2$
