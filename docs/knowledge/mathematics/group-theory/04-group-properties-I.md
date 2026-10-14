---
title: 群的性质（上）：同态与同构、循环群 II、对称群 II
type: concept
status: growing
tags: [Mathematics, Group Theory, Homomorphism, Isomorphism, Signature]
created: 2026-10-14
updated: 2026-10-14
source: 知乎《群论及抽象代数学习笔记（4）- 群的性质（上）》（大PPP，NTU MH2220）
---

# 群的性质（上）：同态与同构、循环群 II、对称群 II

> 系列第三章上。建立同态/同构语言，对循环群做完全分类，并用置换矩阵给每个排列配上奇偶符号 sgn。原文：<https://zhuanlan.zhihu.com/p/613909324>

## 一句话理解

同构是"结构相同"的严格化——同态保运算，双射的同态让两个群可以互相翻译；sgn 则是 $S_n$ 上第一个非平凡的同态，它把排列分成奇偶两类。

## 什么是它

### 群同态

**定义 3.2**：$\varphi:(G,*)\to(H,\cdot)$ 是**同态**，若对所有 $x,y\in G$：
$$\varphi(x*y)=\varphi(x)\cdot\varphi(y).$$
（左边用 $G$ 的运算，右边用 $H$ 的运算。）

**命题 3.5**：$\varphi:G\to H$ 同态，则：
1. 任意子群 $N\le G$ 的像 $\varphi(N)$ 是 $H$ 的子群；
2. **核** $\ker\varphi=\{g\in G:\varphi(g)=e_H\}$ 是 $G$ 的子群；
3. 同态的复合仍是同态。

### 群同构

**定义 3.6**：双射的同态叫**同构**，记 $G\cong H$。

**命题 3.7（同构不变量）**：同构保持：cardinality、元素阶（$o(g)=o(\varphi(g))$）、阿贝尔性、循环性。

> 这是"用不变量排除同构"的工具箱：[05](./group-properties-II.md) 里证 $A_4\not\cong D_6$（前者无 6 阶元素）正是用法示范。

### 循环群分类

**定理 3.10**：无限循环群 $\cong(\mathbb{Z},+)$；$n$ 阶有限循环群 $\cong(\mathbb{Z}/n\mathbb{Z},+)$。

证明：$\varphi(g^k)=k$（或 $\bar k$），验证良定义、双射、同态三件事。

> 同构意义下循环群**只有两种**：$C_\infty$ 与 $C_n$。以后写 $C_n:=\mathbb{Z}/n\mathbb{Z}$。

**推论 3.11**：循环群的子群都是循环群（$\mathbb{Z}$ 的子群 = $m\mathbb{Z}$；有限情形取最小正代表元）。

### 对称群 II：置换矩阵与 sgn

**定义 3.13（置换矩阵）**：$(Q_\sigma)_{ij}=1\iff\sigma(j)=i$（每行每列恰一个 1）。

**推论 3.16**：
1. $Q_\sigma Q_\sigma^\top=I_n$（逆就是转置）
2. $\det(Q_\sigma)=\pm1$
3. $Q_\sigma Q_\tau=Q_{\sigma\tau}$（复合变矩阵乘法——$S_n$ 嵌入 $GL_n$）

**定义 3.17**：$\operatorname{sgn}(\sigma):=\det(Q_\sigma)$，称为排列的 signature（sign / parity）。

**推论 3.18**：$\operatorname{sgn}(\sigma\tau)=\operatorname{sgn}(\sigma)\operatorname{sgn}(\tau)$——即 $sgn:S_n\to\{\pm1\}\cong C_2$ 是群同态。

**引理 3.19**：$k$-cycle 满足 $\operatorname{sgn}=-1$ 当 $k$ 偶（分解成 $k-1$ 个对换）。

**定理 3.21**：把 $\sigma$ 写成 cycles（不必不相交）之积，则
$$\operatorname{sgn}(\sigma)=(-1)^d,$$
$d$ 为**偶长度** cycle 的个数。

例：$(4,2,5)(7,2,4,3)(3,1)$ 有两个偶长度 cycle，$\operatorname{sgn}=+1$。

## 为什么重要

- 同态/同构是整个专题后半部（[商群](./normal-subgroup-and-quotient-group.md)、[同构定理](./isomorphism-theorems-I.md)）的语言基础；$\ker\varphi$ 在这里作为子群登场，[08](./isomorphism-theorems-I.md) 里它升级为正规子群并给出 $G/\ker\varphi\cong\operatorname{im}\varphi$
- sgn 是第一个"从群到 $C_2$ 的非平凡同态"，其核就是[交错子群 $A_n$](./group-properties-II.md)；[魔方群](../rubiks-cube-group-theory.md) 的守恒律一（角/棱奇偶性同步）正是 sgn 思想的二维版本
- 置换矩阵把 $S_n$ 线性代数化——"群表示论"的第一次预演

## 局限 / 边界

- $\operatorname{sgn}$ 的定义依赖 $\det$，看起来"线性代数气"很重；纯群论的定义（对换个数）需要另外证明良定义，本篇用矩阵路线绕开了这个问题
- 同构不变量只能**否定**同构（找到不一致的不变量），**肯定**同构必须构造映射

## 我的理解

- "同构保阶"让阶成为指纹：想证明两个群不同构，数一数各阶元素个数往往一击致命（$A_4$ vs $D_6$、$C_6$ vs $S_3$）
- $\operatorname{sgn}$ 用行列式定义的妙处：$\det(AB)=\det(A)\det(B)$ 免费送出"sgn 是同态"；剩下的全是行列式计算

## Related

- [03 对称群与循环群](./symmetric-and-cyclic-groups.md) — cycle 分解是 sgn 计算的输入
- [05 群的性质（下）](./group-properties-II.md) — $A_n=\ker(sgn)$、拉格朗日定理
- [08 同构定理（上）](./isomorphism-theorems-I.md) — $\ker\varphi$ 的完整故事
- [魔方群的数学结构](../rubiks-cube-group-theory.md) — 奇偶性守恒 = sgn 思想的二维推广
