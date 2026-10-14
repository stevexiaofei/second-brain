---
title: 群的构造（下）：自同构、直积与半直积
type: concept
status: growing
tags: [Mathematics, Group Theory, Automorphism, Direct Product, Semidirect Product]
created: 2026-10-14
updated: 2026-10-14
source: 知乎《群论及抽象代数学习笔记（7）- 群的构造（下）》（大PPP，NTU MH2220）
---

# 群的构造（下）：自同构、直积与半直积

> 系列第四章下。从"群到自己 的同构"（自同构）出发，建立两种把小群拼成大群的运算：直积（无扭）与半直积（带扭曲）。原文：<https://zhuanlan.zhihu.com/p/617626701>

## 一句话理解

直积是"两群并排各过各的"，半直积是"一个群通过自同构去扭另一个群"——$D_n$ 的旋转 × 翻转、$S_n=A_n\rtimes\langle(1,2)\rangle$ 都是半直积。

## 什么是它

### 自同构

**定义 4.16**：**自同构**是 $G\to G$ 的群同构；全体记 $Aut(G)$（复合为运算，成群）。

**定义 4.17（内自同构）**：$\lambda_g(x)=gxg^{-1}$ 是自同构（单射显然，同态因 $g(xy)g^{-1}=(gxg^{-1})(gyg^{-1})$）。全体内自同构记 $Inn(G)$；非内的称**外自同构**。

**例 4.18**：$G$ 阿贝尔时 $Inn(G)=\{1\}$（$gxg^{-1}=x$）——阿贝尔群没有内扭曲。

**例 4.19（$Aut(D_3)\cong D_3$）**：自同构保阶，故 $\phi(r)\in\{r,r^2\}$、$\phi(m)\in\{m,rm,r^2m\}$，共 6 个候选 $\phi_{x,y}$；逐一验证满足 $x^{-1}y=yx$ 的 6 个都是同态。再由"6 阶群只能是 $D_3$ 或 $C_6$"+ "$Aut(D_3)$ 非阿贝尔"（给出具体两个自同构不交换）排除 $C_6$。

**定理 4.20**：$Aut(C_n)\cong(\mathbb{Z}/n\mathbb{Z})^\times$。
证明：$\phi(g)$ 必须是生成元，生成元恰是 $g^k\ (\gcd(n,k)=1)$，共 $\varphi(n)$ 个；复合对应 $\phi_k\circ\phi_l=\phi_{kl}$，即模 $n$ 乘法。

### 直积

**定义 4.21**：
- **外直积** $H\times K$：集合为笛卡尔积，运算逐分量 $(h,k)(h',k')=(hh',kk')$
- **内直积**：$G\cong H\times K$ 经 $\phi((h,k))=hk$

**推论 4.22**：$(G_1\times G_2)\times G_3\cong G_1\times(G_2\times G_3)$（结合律，意义下）。

**定理 4.23（循环群按质数幂分解）**：$n=p_1^{a_1}\cdots p_k^{a_k}$（质因数分解，两两互质）时
$$C_n\cong C_{p_1^{a_1}}\times\cdots\times C_{p_k^{a_k}}.$$
证明核心：$\gcd(r,s)=1$ 时 $C_{rs}\cong C_r\times C_s$——用 Bézout 找到 $ar+bs=1$ 把 $g$ 写成两块生成元的乘积。

**警示（例 4.24）**：$C_{p^2}\ne C_p\times C_p$——前者有 $p^2$ 阶元素，后者的非单位元阶全是 $p$。互质才能拆，同质数幂不能拆。

**定理 4.25（内直积判定）**：$G$ 是子群 $H,K$ 的内直积 $\iff$
1. $G=HK$；2. $H\cap K=\{1\}$；3. $H,K\triangleleft G$。

### 半直积

**定义 4.27**：给定同态 $\phi:H\to Aut(N)$，**半直积** $N\rtimes_\phi H$ 的集合是 $N\times H$，运算为
$$(n,h)*(n',h')=(n\,[\phi(h)(n')],\ hh').$$

- $\phi$ 平凡时退化为直积
- $(n,h)^{-1}=(\phi(h^{-1})(n^{-1}),\ h^{-1})$
- base group $\{(n,1)\}\cong N$ 总是半直积的**正规**子群

**定理 4.30**：$G=N H$、$H\cap N=\{1\}$、$N\triangleleft G$ 时 $G\cong N\rtimes H$（$H$ 作用于 $N$ 的方式取共轭 $\lambda_h(n)=hnh^{-1}$）。

**例 4.31（本篇点睛）**：$A_n\triangleleft S_n$、$K=\langle(1,2)\rangle$、$A_n\cap K=\{1\}$、$S_n=A_nK$，且 $K$ 不正规（$n\ge3$）——所以
$$S_n=A_n\rtimes\langle(1,2)\rangle\quad(\text{不是直积}).$$

## 为什么重要

- "拼群"是分类定理的原材料：[有限阿贝尔群分类](./isomorphism-theorems-II.md)的结论就是"循环群质数幂的直积"；[阶 8 分类](./isomorphism-theorems-II.md)里 $D_4=C_4\rtimes C_2$ 型结构
- $Aut(C_n)\cong(\mathbb{Z}/n\mathbb{Z})^\times$ 把[模 n 乘法群](./mod-n-and-dihedral-groups.md)接回群论主线
- 内自同构 $\lambda_g$ 正是[魔方笔记](../rubiks-cube-group-theory.md)里的"共轭搬运"：$o(x)=o(\lambda_g(x))$（[01](./01-introduction-to-group-theory.md) 命题 1.13）在这里得到代数身份

## 局限 / 边界

- 半直积依赖 $\phi$ 的选取：同一对 $(N,H)$ 可拼出不同构的群（$C_4\rtimes C_2$ 有平凡与非平凡两种 $\phi$，得 $C_4\times C_2$ 与 $D_4$）
- 定理 4.30 要求 $N$ 正规而 $H$ 不必——方向别记反

## 我的理解

- 直积 vs 半直积的区别就一句话：$hk=h'k'$ 拆开时，$k$ 穿过 $h'$ 要不要"付费"。直积免费（$h'k=kh'$），半直积付自同构的税（$h'k=\phi(k)(h')\,k$）
- $S_n=A_n\rtimes C_2$ 说明半直积不是冷门构造，而是"正规部分 + 一点外在扭曲"的标准形状；魔方群同样是 $\ker\pi$（朝向）与位置置换部分的拼接

## Related

- [06 正规子群与商群](./normal-subgroup-and-quotient-group.md) — 正规性是半直积的前提
- [08 同构定理（上）](./isomorphism-theorems-I.md) — $HK/H\cong K/(H\cap K)$ 配合内直积使用
- [09 同构定理（下）与有限阿贝尔群分类](./isomorphism-theorems-II.md) — 直积是分类的表达语言
- [魔方群的数学结构](../rubiks-cube-group-theory.md) — 共轭与交换子
