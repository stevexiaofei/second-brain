---
title: 同构定理（下）与有限阿贝尔群分类：子群对应定理、分类定理、阶 8 的群
type: concept
status: growing
tags: [Mathematics, Group Theory, Subgroup Lattice, Classification of Finite Abelian Groups]
created: 2026-10-14
updated: 2026-10-14
source: 知乎《群论及抽象代数学习笔记（9）完结篇》（大PPP，NTU MH2220）
---

# 同构定理（下）与有限阿贝尔群分类

> 系列第五章下（完结篇）。子群对应定理把 $G$ 的子群格与 $G/N$ 的子群格对齐；终极应用是有限阿贝尔群的完全分类。原文：<https://zhuanlan.zhihu.com/p/620893105>

## 一句话理解

商群 $G/N$ 的子群恰好是 $G$ 中"包含 $N$ 的子群"的缩略图——用这张对应表加上直积工具，能把所有有限阿贝尔群一口气分类完毕。

## 什么是它

### 子群对应定理

**引理 5.11**：$\phi:G\to H$ 同态、$K\le H$，则 $\phi^{-1}(K)$ 是 $G$ 中包含 $\ker\phi$ 的子群。

**定理 5.12**：$N\triangleleft G$、$\pi:G\to G/N$ 典范满射，则
$$\Phi(H)=\pi(H)=H/N$$
是 $\{G\text{ 的包含 }N\text{ 的子群}\}\ \longrightarrow\ \{G/N\text{ 的子群}\}$ 的**双射**，逆为 $\Psi(K)=\pi^{-1}(K)$。且对应保持包含关系与正规性：
1. $H\le H'\iff\pi(H)\le\pi(H')$
2. $N\le H\triangleleft G\iff H/N\triangleleft G/N$

证明关键在 $\pi^{-1}(\pi(H))=H$：$\pi(g)=\pi(h)$ 给出 $gh^{-1}\in N\subseteq H$，故 $g\in H$——又是"$N$ 正规 + 包含"的配合。

**例 5.13**：$\mathbb{Z}$ 的子群格 $m\mathbb{Z}\le n\mathbb{Z}\iff n\mid m$（**整除方向与包含方向相反**）；模掉 $6\mathbb{Z}$ 后的格与 $\mathbb{Z}/6\mathbb{Z}$ 的子群格逐点对应。

**例 5.14**：$S_4$ 的 $N=\langle(1,2)(3,4),(1,3)(2,4)\rangle$，$S_4/N\cong S_3$——直接列 $S_4$ 中含 $N$ 的子群很痛苦，改看 $S_3$ 的子群格即可。

### 有限阿贝尔群分类

**引理 5.15**：$G$ 有限阿贝尔：
1. $|G|=p^am$（$p\nmid m$）时 $G=H\times K$，其中 $H=\{x:x^{p^a}=e\}$、$K=\{x:x^m=e\}$（$p$-部分与 $p'$-部分自动分离）
2. $n=p_1^{n_1}\cdots p_r^{n_r}$ 时 $G=G_1\times\cdots\times G_r$，$|G_j|=p_j^{n_j}$

**中间结论（Cauchy 型）**：$p\mid|G|$（$G$ 有限阿贝尔）$\Rightarrow$ 存在 $p$ 阶元素（对 $|G|$ 归纳，先在循环因子里找，找不到就下放到商群）。

**定理 5.16（分类）**：$G$ 有限阿贝尔 $\Rightarrow$
$$G\cong C_{p_1^{n_1}}\times\cdots\times C_{p_s^{n_s}}\qquad(p_i\ \text{质数，可重复}).$$

证明最重的一步是引理：$H$ 为 $G$ 中最高阶循环子群 $\Rightarrow G=H\times K$。证明对 $|G|$ 归纳，反复使用第二同构定理与子群对应定理把问题降到 $G/N$。

**定理 5.17（不变因子形式）**：等价写法 $G\cong C_{d_1}\times\cdots\times C_{d_k}$，$d_k\mid\cdots\mid d_2\mid d_1$。

**例（阶 8 完全分类）**：阿贝尔：$C_8,\ C_4\times C_2,\ C_2^3$；非阿贝尔：$D_4,\ Q_8$。共 5 个。

## 为什么重要

- 这是本系列的**收官定理**：它宣布有限阿贝尔群在"质数幂循环群直积"意义下完全可知——阶定了，可能性就有限
- 子群对应定理是实用工具：大群的子群结构难列时，转去小商群里列（$S_4\to S_3$ 的降维打击）
- [魔方群](../rubiks-cube-group-theory.md) 的商 $G/G_0\cong\mathbb{Z}_3\times\mathbb{Z}_2\times\mathbb{Z}_2$ 可以用本篇立刻验证：$3,2,2$ 两两互质，直积不塌缩

## 局限 / 边界

- 分类只覆盖**阿贝尔**；非阿贝尔没有类似的总定理（小阶靠 Sylow + 群作用，本系列未覆盖）
- 质数幂循环群部分 $C_{p^2}\ne C_p\times C_p$（[07](./automorphism-and-products.md) 例 4.24）——同幂次不互质，分解必须到质数幂一层为止
- $Q_8$（四元数群）在原文只是提及，未展开

## 我的理解

- 有限阿贝尔群分类的证明结构值得单独记：**先用互质性把不同质数拆开（引理 5.15），再在固定质数幂内部归纳切出最高阶循环因子**。第一步靠 Bézout，第二步靠同构定理链条
- 子群对应定理最反直觉的点是 $\pi^{-1}(\pi(H))=H$ 会**严格**成立（而非只包含）——原因是 $N\subseteq H$；没有这个包含，$\pi^{-1}(\pi(H))=HN$ 只是个乘积
- 阶 8 的 5 个群是检验工具箱的试金石：3 个阿贝尔来自分类定理，2 个非阿贝尔靠 $C_4\rtimes C_2$ 两种 $\phi$ 与四元数

## Related

- [08 同构定理（上）](./isomorphism-theorems-I.md) — 本篇所有证明的引擎
- [07 自同构、直积与半直积](./automorphism-and-products.md) — $C_n$ 质数幂分解、内直积判定
- [05 群的性质（下）](./group-properties-II.md) — 阶 6 分类（分类思想的小型预演）
- [魔方群的数学结构](../rubiks-cube-group-theory.md) — $G/G_0\cong\mathbb{Z}_3\times\mathbb{Z}_2\times\mathbb{Z}_2$
