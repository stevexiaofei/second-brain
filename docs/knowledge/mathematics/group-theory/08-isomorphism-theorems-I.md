---
title: 同构定理（上）：第一、二、三同构定理
type: concept
status: growing
tags: [Mathematics, Group Theory, Isomorphism Theorems, Quotient Group]
created: 2026-10-14
updated: 2026-10-14
source: 知乎《群论及抽象代数学习笔记（8）- 同构定理（上）》（大PPP，NTU MH2220）
---

# 同构定理（上）：第一、二、三同构定理

> 系列第五章上。抽象代数最经典的三件工具：$G/\ker\phi\cong\operatorname{im}\phi$ 及其两个推论。原文：<https://zhuanlan.zhihu.com/p/620855557>

## 一句话理解

第一同构定理说"同态的像 = 群模掉核"——把映射问题变成商群问题；第二、三定理是它在乘积与嵌套情形下的换算公式。

## 什么是它

### 预备：核是正规的

**命题 5.1**：$\phi:G\to H$ 同态，则：
1. $\ker\phi\triangleleft G$：$\phi(xgx^{-1})=\phi(x)\phi(g)\phi(x)^{-1}=1$
2. $\phi$ 单射 $\iff\ker\phi=\{1\}$

**典范满射**：$\pi:G\to G/N,\ \pi(g)=gN$ 是同态，且 $\ker\pi=N$、$\operatorname{im}\pi=G/N$。

### 第一同构定理

**引理 5.2（交换图提升）**：$N\triangleleft G$、$N\le\ker\phi$，则存在**唯一**同态 $\psi:G/N\to H$ 使 $\phi=\psi\circ\pi$；且：
1. $\phi$ 满射 $\iff\psi$ 满射
2. $N=\ker\phi\iff\psi$ 单射

（$\psi$ 必为 $\psi(gN)=\phi(g)$；良定义性正是"$N\le\ker\phi"$ 的用武之地。）

**定理 5.3（第一同构定理）**：
$$G/\ker\phi\ \cong\ \operatorname{im}\phi.$$

**例 5.4（指数 2 重证）**：$[G:H]=2$ 时定义 $\phi(g)=1$（$g\in H$）或 $-1$（$g\notin H$），四种情形验证同态，得 $G/H\cong C_2$——比[逐陪集论证](./normal-subgroup-and-quotient-group.md)更机械。

**例 5.5**：$S_n/A_n\cong\{\pm1\}$ 由 $sgn$ 同态直接读出。

### 第二同构定理

**引理 5.7**：$H,K\le G$ 且 $K\le N_G(H)$，则 $HK\le G$、$H\triangleleft HK$、$H\cap K\triangleleft K$。

**定理 5.8**：
$$HK/H\ \cong\ K/(H\cap K).$$
证明：$\phi:K\to HK/H,\ \phi(k)=kH$ 是满同态，$\ker\phi=H\cap K$，用第一定理收尾。

**推论（阶公式）**：$|HK|=\dfrac{|H|\,|K|}{|H\cap K|}$。

**例 5.9**：$K\le S_n$ 含至少一个奇排列时，$K/(A_n\cap K)\cong C_2$——$K$ 里恰好一半是偶排列，不论 $K$ 是什么。

### 第三同构定理

**定理 5.10**：$N\le H$、$N\triangleleft G$、$H\triangleleft G$ 时：
$$G/H\ \cong\ (G/N)/(H/N).$$
证明：$\phi(gN)=gH$ 是 $G/N\to G/H$ 的满同态，$\ker\phi=H/N$。

> 形式记忆："分数约分" $(G/N)/(H/N)\approx G/H$。

## 为什么重要

- 三定理合起来是**商群运算律**：第一定理给出"模掉核"的标准操作，第二、三定理给出两个交换/结合的换算规则——后续一切分类证明都在这套语言里进行
- [魔方群](../rubiks-cube-group-theory.md) 求 $|G|$ 的核心一步 $G/\ker\pi\cong P$ 就是第一同构定理的直用：核是朝向部分（$3^7 2^{11}$），像是位置部分（$(8!12!/2)$）
- 第二定理的阶公式是估算子群乘积大小的唯一直接工具

## 局限 / 边界

- 引理 5.2 要求 $N\le\ker\phi$：$\psi$ 的存在性不免费，"$\phi$ 分解穿过商"必须有核吸收 $N$
- 第二定理要求 $K\le N_G(H)$（保证 $HK$ 是子群），任意两个子群的乘积 $HK$ 未必是子群

## 我的理解

- 三定理的公共证明模式：**构造一个显然满的映射 → 算出核 → 套第一定理**。第一定理是引擎，另两个是应用
- "约分"直觉对第三定理有效，但第一定理不能这么记——它是"像与商同构"，方向是"把映射压扁"，不是分数运算

## Related

- [06 正规子群与商群](./normal-subgroup-and-quotient-group.md) — 商群定义与良定义性
- [07 自同构、直积与半直积](./automorphism-and-products.md) — 内直积判定条件与第二定理配合
- [09 同构定理（下）与有限阿贝尔群分类](./isomorphism-theorems-II.md) — 子群对应定理继续这条线
- [魔方群的数学结构](../rubiks-cube-group-theory.md) — $G/\ker\pi\cong P$ 的完整应用
