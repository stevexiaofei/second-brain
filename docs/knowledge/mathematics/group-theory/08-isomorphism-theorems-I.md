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
1. $\ker\phi\triangleleft G$
2. $\phi$ 单射 $\iff\ker\phi=\{e_G\}$

**证明：**

1. **核正规**：用[定义 4.1 的等价说法 "$gxg^{-1}\in\ker\phi$ 对一切 $x\in\ker\phi$"](./06-normal-subgroup-and-quotient-group.md)。任取 $g\in G$、$x\in\ker\phi$：
   $$\phi(gxg^{-1})=\phi(g)\phi(x)\phi(g^{-1})=\phi(g)\,e_H\,\phi(g)^{-1}=\phi(g)\phi(g)^{-1}=e_H,$$
   故 $gxg^{-1}\in\ker\phi$，$K=\ker\phi$ 正规 ✓。（这里也看到了共轭的代数身份：同态把共轭送到共轭。）
2. **单射判据**：
   - **($\Rightarrow$)** $\phi$ 单射时，$\phi(x)=e_H=\phi(e_G)$（[命题 3.5](#) 已证 $\phi(e_G)=e_H$）给出 $x=e_G$，故 $\ker\phi=\{e_G\}$。
   - **($\Leftarrow$)** 设 $\ker\phi=\{e_G\}$。若 $\phi(x)=\phi(y)$，则 $\phi(xy^{-1})=\phi(x)\phi(y)^{-1}=e_H$，故 $xy^{-1}\in\ker\phi=\{e_G\}$，即 $xy^{-1}=e_G$，$x=y$ ✓。$\blacksquare$

> **(2) 是"用核检查单射"的万能钥匙**：以后凡是要证"某个同态是单射"，一律算核。

**典范满射**：$\pi:G\to G/N,\ \pi(g)=gN$（$N\triangleleft G$）是同态，且 $\ker\pi=N$、$\operatorname{im}\pi=G/N$。

**验证**：
- **同态**：$\pi(xy)=(xy)N=(xN)(yN)=\pi(x)\pi(y)$ ✓（**商群乘法的定义正是为了让这件事成立**）。
- **满射**：任意陪集 $gN$ 都是 $\pi(g)$ ✓。
- **核**：$\pi(g)=N$（商群的单位元）$\iff$ $gN=N$ $\iff$ $g\in N$（[引理 3.30(1)](./05-group-properties-II.md)）✓。

### 第一同构定理

**引理 5.2（交换图提升）**：$N\triangleleft G$、$N\le\ker\phi$，则存在**唯一**同态 $\psi:G/N\to H$ 使下图交换（即 $\phi=\psi\circ\pi$）：

```text
        φ
   G ──────────► H
   │             ▲
 π │             │ ψ        (ψ∘π = φ)
   ▼             │
  G/N ───────────┘
```

且：
1. $\phi$ 满射 $\iff$ $\psi$ 满射
2. $N=\ker\phi$ $\iff$ $\psi$ 单射

**证明：**

**(0) $\psi$ 的形式被唯一确定。** 由 $\phi=\psi\circ\pi$；而任意 $gN$ 都能写成 $\pi(g)$，故
$$\psi(gN)=\psi(\pi(g))=\phi(g).$$
所以**唯一性免费**：$\psi$ 只能是 $\psi(gN)=\phi(g)$。

**(1) 良定义。** 设 $gN=g'N$，即 $g^{-1}g'\in N\subseteq\ker\phi$（**这里用到假设 $N\le\ker\phi$**）。则
$$\phi(g^{-1}g')=e_H\ \Rightarrow\ \phi(g)^{-1}\phi(g')=e_H\ \Rightarrow\ \phi(g)=\phi(g'),$$
即 $\psi(gN)=\psi(g'N)$ ✓。**没有 $N\le\ker\phi$，$\psi$ 无从定义**——这就是引理要求"核要能吸收 $N$"的原因。

**(2) $\psi$ 是同态**：$\psi\bigl((gN)(g'N)\bigr)=\psi\bigl(gg'N\bigr)=\phi(gg')=\phi(g)\phi(g')=\psi(gN)\psi(g'N)$ ✓。

**(3) 满射的对应。** $\operatorname{im}\psi=\psi(G/N)=\{\phi(g):g\in G\}=\phi(G)=\operatorname{im}\phi$（**正是 (0) 那个公式直接给出的**）。故 $\phi$ 满 $\iff$ $\psi$ 满 ✓。

**(4) 单射的对应。**
$$\ker\psi=\{gN:\psi(gN)=e_H\}=\{gN:\phi(g)=e_H\}=\{gN:g\in\ker\phi\}=\ker\phi/N,$$
（最后一步是陪集的集合：$\{gN:g\in\ker\phi\}=\ker\phi/N$，需要 $N\le\ker\phi$ 保证这是良定义的集合）。于是
$$\psi\text{ 单射}\iff\ker\psi\text{ 平凡}\iff\ker\phi/N=\{N\}\iff\ker\phi=N$$
（第二个 $\iff$ 因为 $gN=N\iff g\in N$，故 $\ker\phi/N=\{N\}\iff$ 每个 $g\in\ker\phi$ 都在 $N$ 中 $\iff\ker\phi\subseteq N$；结合 $N\subseteq\ker\phi$ 得相等）✓。$\blacksquare$

> **图的意义**：引理说"任何在 $G$ 上定义、且核包含 $N$ 的同态，都能**下降（descend）到商群**"——把 $N$ 抹掉不影响 $\phi$。这是商群的**泛性质**，也是后面三定理的统一思路来源。

**定理 5.3（第一同构定理）**：
$$G/\ker\phi\ \cong\ \operatorname{im}\phi.$$

**证明**：取 $N=\ker\phi$（由命题 5.1 正规）。此时引理的假设 $N\le\ker\phi$ 平凡成立（相等），于是存在同态 $\psi:G/\ker\phi\to H$、$\psi(gK)=\phi(g)$（$K=\ker\phi$）。由引理 (3)(4)：$\psi$ 的像是 $\operatorname{im}\phi$、核是 $K/K=\{K\}$ 故单射。于是 $\psi$ 是 $G/K\to\operatorname{im}\phi$ 的**双射同态**，即同构。$\blacksquare$

> **这条定理为什么是"引擎"**：它把"研究一个同态"直接等价为"研究一个商群"。**用法固定三步**：
> 1. 造一个显然满的同态 $\phi:G\to H'$；
> 2. 算出 $\ker\phi$；
> 3. 得 $G/\ker\phi\cong H'$。
>
> 第二、第三同构定理都是这个三步的实例（见下）。**例（三个实例）**：
> - $sgn:S_n\to\{\pm1\}$：核 $A_n$，得 $S_n/A_n\cong C_2$。
> - $\det:GL_n(\mathbb{R})\to\mathbb{R}^\times$：核 $SL_n$，得 $GL_n(\mathbb{R})/SL_n(\mathbb{R})\cong\mathbb{R}^\times$。
> - $\psi:\mathbb{R}\to S^1,\ \psi(t)=e^{2\pi it}$：核 $\mathbb{Z}$，得 $\mathbb{R}/\mathbb{Z}\cong S^1$（[06 例 4.13](./06-normal-subgroup-and-quotient-group.md) 的另一种做法）。

**例 5.4（指数 2 重证）**：设 $[G:H]=2$，要证 $G/H\cong C_2$。

在引理 5.2 中有个更漂亮的思路（无需验算商运算）：取 $N=H$、目标群 $C_2=\{\pm1\}$，定义
$$\phi:G\to\{\pm1\},\qquad \phi(g)=\begin{cases}1,&g\in H\\-1,&g\notin H.\end{cases}\quad$$

**验证 $\phi$ 是同态**（四种情形，用 $H$ 与 $gH=G\setminus H$）：

- $g_1,g_2\in H$：$g_1g_2\in H$，故 $\phi(g_1g_2)=1=1\cdot1$ ✓。
- $g_1\in H,\ g_2\notin H$：$g_1g_2\in g_1(G\setminus H)=G\setminus H$（左乘 $g_1\in H$ 是双射且把 $H$ 送到 $H$、故把补集送到补集），故 $\phi(g_1g_2)=-1=1\cdot(-1)$ ✓。
- $g_1\notin H,\ g_2\in H$：$g_1g_2\in(G\setminus H)g_2=G\setminus H$（同理，右乘），$\phi=-1=(-1)\cdot1$ ✓。
- $g_1,g_2\notin H$：断言 $g_1g_2\in H$。反设 $g_1g_2\notin H$，则 $g_1g_2\in G\setminus H=g_1H$（因 $g_1H=G\setminus H$），故 $g_1g_2=g_1h$ 某 $h\in H$，左消去得 $g_2=h\in H$，与 $g_2\notin H$ 矛盾 ✓。故 $\phi(g_1g_2)=1=(-1)(-1)$ ✓。

$\phi$ 是满射（$H\ne G$ 故有 $g\notin H$），且 $\ker\phi=H$。由引理 5.2（或 [定理 4.6](./06-normal-subgroup-and-quotient-group.md) 的抽象版本）得 $G/H\cong C_2$ $\blacksquare$。

> **这个例子说明 $\phi$ 为什么"显然"是符号函数**：$\phi$ 就是把 $H$ 当"偶"、补集当"奇"——与 [04 的 $sgn$](./04-group-properties-I.md) 是同一构造。**任何指数 2 的子群都自动给出一个到 $C_2$ 的满同态**，这就是[定理 4.2（指数 2 ⇒ 正规）](./06-normal-subgroup-and-quotient-group.md) 的"同态版"证明；比逐陪集论证更机械，而且揭示了正规性的真正来源（**能当核**）。

**例 5.5**：$S_n/A_n\cong\{\pm1\}\cong C_2$（$n\ge2$）。由 $sgn:S_n\to\{\pm1\}$ 满射、$\ker sgn=A_n$，直接套第一同构定理 ✓。**注**：$\{\pm1\}$ 与 $C_2$ 同构（$1\leftrightarrow\bar0,\ -1\leftrightarrow\bar1$）。

### 第二同构定理

**引理 5.7**：$H,K\le G$ 且 $K\le N_G(H)$，则
$$HK\le G,\qquad H\triangleleft HK,\qquad H\cap K\triangleleft K.$$

**证明：** 依次证三件事。

**(1) $HK=\{hk:h\in H,k\in K\}$ 是子群。** 用[判别法](./01-introduction-to-group-theory.md)。

- **非空**：$e=ee\in HK$ ✓。
- **封闭**：取 $h_1k_1,h_2k_2\in HK$。要在中间把 $k_1$ 搬到左边，需要 $k_1h_2$ 的写法：因 $K\le N_G(H)$，有 $k_1Hk_1^{-1}=H$，即 $k_1H=Hk_1$，故存在 $h_3\in H$ 使 $k_1h_2=h_3k_1$。于是
  $$h_1k_1h_2k_2=h_1(h_3k_1)k_2=(h_1h_3)(k_1k_2)\in HK$$
  ✓（**这一步就是"$K$ 必须能正规化 $H$"的全部用处**）。
- **逆封闭**：$(hk)^{-1}=k^{-1}h^{-1}$；用同样的搬移（$k^{-1}\in K\le N_G(H)$，故 $k^{-1}h^{-1}=h_4k^{-1}$ 某 $h_4\in H$），得 $(hk)^{-1}=h_4k^{-1}\in HK$ ✓。

**(2) $H\triangleleft HK$。** 由 (1) $HK\le G$。任取 $hk\in HK$、$x\in H$：
$$(hk)x(hk)^{-1}=h\,(kxk^{-1})\,h^{-1}\in H$$
（因 $k\in K\le N_G(H)$ 故 $kxk^{-1}\in H$，再共轭 $h\,(\cdot)\,h^{-1}$ 仍在 $H$）✓。故 $H\triangleleft HK$。

**(3) $H\cap K\triangleleft K$。** $H\cap K\le K$ 是子群（交）。任取 $k\in K$、$x\in H\cap K$：$kxk^{-1}\in H$（因 $K\le N_G(H)$）且 $kxk^{-1}\in K$（因 $k,x\in K$ 封闭），故 $kxk^{-1}\in H\cap K$ ✓。$\blacksquare$

> **注意 $H$ 不必正规于 $G$**——引理只要求 $K\le N_G(H)$。**反例**：$G=S_3$、$H=\langle(1,2)\rangle$、$K=\langle(1,3)\rangle$。$K\not\le N_G(H)$（$(1,3)$ 不正规化 $\langle(1,2)\rangle$），引理不适用，而 $HK=\{e,(1,2),(1,3),(1,3,2)\}$ 确实不是子群（$|HK|=4\nmid6$）✓。

**定理 5.8（第二同构定理）**：
$$HK/H\ \cong\ K/(H\cap K).$$

**证明（三步固定套路）：**

**(1) 构造显然满的同态。** 定义
$$\phi:K\to HK/H,\qquad \phi(k)=kH.$$
（目标群是商群 $HK/H$，由引理 (2) $H\triangleleft HK$ 保证这是群。）

- **$\phi$ 是同态**：$\phi(k_1k_2)=(k_1k_2)H=k_1H\cdot k_2H=\phi(k_1)\phi(k_2)$ ✓。
- **$\phi$ 是满射**：$HK/H$ 的任意元素形如 $(hk)H$，而 $(hk)H=h(kH)=(kH)$（因 $h\in H$ 故 $h(kH)=kH$），故 $=kH=\phi(k)$ ✓。

**(2) 算核。**
$$\ker\phi=\{k\in K:kH=H\}=\{k\in K:k\in H\}=K\cap H$$
（中间用[引理 3.30(1)](./05-group-properties-II.md)：$kH=H\iff k\in H$）✓。

**(3) 套第一同构定理。** $K/\ker\phi\cong\operatorname{im}\phi$，即
$$K/(H\cap K)\ \cong\ HK/H.$$
$\blacksquare$

**推论（阶公式）**：$|HK|=\dfrac{|H|\,|K|}{|H\cap K|}$。

**证明**：对定理 5.8 两边取阶。左边 $|HK/H|=[HK:H]=\frac{|HK|}{|H|}$（拉格朗日）；右边 $|K/(H\cap K)|=\frac{|K|}{|H\cap K|}$。相等即
$$\frac{|HK|}{|H|}=\frac{|K|}{|H\cap K|}\ \Longrightarrow\ |HK|=\frac{|H|\,|K|}{|H\cap K|}.$$
$\blacksquare$

> **这是"估算子群乘积大小"的唯一直接工具**。**例**：$|H|=6,\ |K|=10,\ |H\cap K|=2$ 时 $|HK|=30$；若 $|H\cap K|=1$ 则 $|HK|=60$。**上界**总是 $|H||K|$，取等当且仅当交平凡。

**例 5.9**：$K\le S_n$ 且 $K$ 含至少一个奇排列时
$$K/(A_n\cap K)\cong C_2.$$

**证明**：取 $H=A_n\triangleleft S_n$（指数 2）。则 $K\le S_n=N_{S_n}(A_n)$ 自动成立，由第二同构定理
$$A_nK/A_n\cong K/(A_n\cap K).$$
左边是 $S_n/A_n\cong C_2$ 的子群；又 $K$ 含奇排列意味着 $A_nK\supsetneq A_n$，故 $A_nK/A_n\ne\{A_n\}$，即左边**非平凡**，只能是整个 $C_2$。故 $K/(A_n\cap K)\cong C_2$ ✓。$\blacksquare$

> **推论：$K$ 里恰好一半是偶排列**。由阶公式（取 $H=A_n$）
> $$|A_nK|=\frac{|A_n||K|}{|A_n\cap K|},\qquad \frac{|K|}{|A_n\cap K|}=|K/(A_n\cap K)|=2,$$
> 故 $|A_n\cap K|=|K|/2$——**不论 $K$ 是什么子群**。
>
> **例**：$K=S_n$ 时 $|A_n\cap S_n|=|A_n|=n!/2$ ✓；$K=D_4\le S_4$（$|K|=8$）时 $K\cap A_4$ 恰好 4 个元素（$D_4$ 中的偶排列：$e,r^2$ 与两个双对换型 $rm,r^3m$）；$K=\langle(1,2),(1,2,3,4)\rangle$ 同法。**这个"半半"现象对任意含奇排列的子群都成立**，是 $sgn$ 同态的直接后果。

### 第三同构定理

**定理 5.10（第三同构定理）**：$N\le H$、$N\triangleleft G$、$H\triangleleft G$ 时：
$$G/H\ \cong\ (G/N)/(H/N).$$

**证明（同样三步）：**

**(0) 先把用语交代清楚。** 因 $N\triangleleft G$、$H\triangleleft G$，商群 $G/N$、$G/H$ 都有定义。又 $N\le H\triangleleft G$ ⇒ $H/N\triangleleft G/N$（[06 引理 5.7 型的事实](./06-normal-subgroup-and-quotient-group.md)：正规传递到商群），故商 $(G/N)/(H/N)$ 有定义。

**(1) 构造显然满的同态。** 定义
$$\phi:G/N\to G/H,\qquad \phi(gN)=gH.$$
（这是"把更粗的等价关系 $N$ 换成更细的 $H$"——$N$ 是 $H$ 的子群，$H$-陪集比 $N$-陪集大。）

- **良定义**：$gN=g'N$ 即 $g^{-1}g'\in N\subseteq H$，故 $gH=g'H$（[引理 3.30(2)](./05-group-properties-II.md)）✓。**这里用了 $N\le H$**。
- **同态**：$\phi\bigl((gN)(g'N)\bigr)=\phi(gg'N)=gg'H=(gH)(g'H)=\phi(gN)\phi(g'N)$ ✓。
- **满射**：任意 $gH\in G/H$ 都是 $\phi(gN)$ ✓。

**(2) 算核。**
$$\ker\phi=\{gN:gH=H\}=\{gN:g\in H\}=H/N$$
（第一个 $=$ 是 $\phi$ 的定义加"$gH$ 是 $G/H$ 的单位元即 $H$"；第二个用引理 3.30(1)）✓。

**(3) 套第一同构定理。**
$$(G/N)/\ker\phi\cong\operatorname{im}\phi\ \Longrightarrow\ (G/N)/(H/N)\cong G/H.$$
$\blacksquare$

> **形式记忆："分数约分"** $(G/N)/(H/N)\approx G/H$——把上下共同的 $N$ 约掉。**但注意顺序**：分子是更大的商 $G/N$、分母是 $H/N$，约分后剩 $G/H$。
>
> **例（$\mathbb{Z}$ 上的约分）**：$G=\mathbb{Z}$（加法）、$N=12\mathbb{Z}$、$H=4\mathbb{Z}$（注意 $N\subseteq H$，即 $4\mid12$）。则 $H/N=4\mathbb{Z}/12\mathbb{Z}$（在 $\mathbb{Z}/12\mathbb{Z}$ 中大小 $12/4=3$ 的子群，同构于 $C_3$），$G/N=\mathbb{Z}/12\mathbb{Z}$，$G/H=\mathbb{Z}/4\mathbb{Z}$。第三定理给出
> $$(\mathbb{Z}/12\mathbb{Z})\big/\bigl(4\mathbb{Z}/12\mathbb{Z}\bigr)\cong\mathbb{Z}/4\mathbb{Z},$$
> 即"12 个刻度的钟除以 3 个一组的粗刻度 = 4 个刻度的钟"✓。用阶验证：$|(\mathbb{Z}/12)/(C_3)|=12/3=4=|C_4|$ ✓。
>
> **例（$S_4$ 上的一次，配 [09 例 5.14](./09-isomorphism-theorems-II.md)）**：$N=V_4=\{e,(1,2)(3,4),(1,3)(2,4),(1,4)(2,3)\}$（正规，因是共轭类之并）、$H=A_4$（$V_4\le A_4$）。则
> $$S_4/A_4\cong(S_4/V_4)/(A_4/V_4),$$
> 左边 $\cong C_2$（指数 2），右边 $=(S_3)/(C_3)$（因 $S_4/V_4\cong S_3$、$A_4/V_4\cong C_3$），故 $S_3/C_3\cong C_2$ ✓——**与 $S_3/A_3\cong C_2$ 一致**，第三定理把两个看似不同的商串了起来。

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

- [06 正规子群与商群](./06-normal-subgroup-and-quotient-group.md) — 商群定义与良定义性
- [07 自同构、直积与半直积](./07-automorphism-and-products.md) — 内直积判定条件与第二定理配合
- [09 同构定理（下）与有限阿贝尔群分类](./09-isomorphism-theorems-II.md) — 子群对应定理继续这条线
- [魔方群的数学结构](../rubiks-cube-group-theory.md) — $G/\ker\pi\cong P$ 的完整应用
