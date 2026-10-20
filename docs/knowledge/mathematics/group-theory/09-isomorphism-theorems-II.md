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

**引理 5.11**：$\phi:G\to H$ 同态、$K\le H$，则 $\phi^{-1}(K)=\{g\in G:\phi(g)\in K\}$ 是 $G$ 的子群，且 $\ker\phi\le\phi^{-1}(K)$。

**证明：**

**(1) 是子群**（判别法）：$e_G\in\phi^{-1}(K)$（因 $\phi(e_G)=e_H\in K$），非空；$x,y\in\phi^{-1}(K)$ 时 $\phi(xy)=\phi(x)\phi(y)\in K$（$K$ 封闭）；$\phi(x^{-1})=\phi(x)^{-1}\in K$（$K$ 取逆封闭）✓。

**(2) 包含核**：$x\in\ker\phi$ 时 $\phi(x)=e_H\in K$，故 $x\in\phi^{-1}(K)$ ✓。$\blacksquare$

> **例（原像往往比像"更好"）**：$sgn^{-1}(\{1\})=A_n$ ✓；$\det^{-1}(\{1\})=SL_n$ ✓；$\pi^{-1}(K)$ 正是[子群对应定理](#子群对应定理)的逆向映射 $\Psi$。

**定理 5.12（子群对应定理）**：$N\triangleleft G$、$\pi:G\to G/N$ 典范满射，则
$$\Phi(H)=\pi(H)=H/N$$
是
$$\{G\text{ 的包含 }N\text{ 的子群}\}\ \xrightarrow{\ \sim\ }\ \{G/N\text{ 的子群}\}$$
的**双射**，其逆为 $\Psi(K)=\pi^{-1}(K)$。而且对应保持包含关系与正规性：
1. $H\le H'\iff\pi(H)\le\pi(H')$
2. $N\le H\triangleleft G\iff H/N\triangleleft G/N$

**证明。**

**(0) 两个映射确实落在目标集合里。** $\Phi(H)=H/N$ 是 $G/N$ 的子群（同态的像，[命题 3.5](./04-group-properties-I.md)）；$\Psi(K)=\pi^{-1}(K)$ 是 $G$ 的子群且含 $\ker\pi=N$（引理 5.11），故是"包含 $N$ 的子群"✓。

**(1) $\Psi\circ\Phi=\mathrm{id}$，即 $\pi^{-1}(\pi(H))=H$（$N\le H$）**——**这是证明的关键一步，也是一处包含关系必须严格成立（不能只证 $\supseteq$）的地方**。

- $(\supseteq)$ 显然：$h\in H$ ⇒ $\pi(h)\in\pi(H)$ ⇒ $h\in\pi^{-1}(\pi(H))$。
- $(\subseteq)$ 取 $g\in\pi^{-1}(\pi(H))$，则 $\pi(g)\in\pi(H)$，即 $gN=hN$ 某 $h\in H$；由[引理 3.30(2)](./05-group-properties-II.md) 得 $h^{-1}g\in N$，而 $N\le H$ 故 $h^{-1}g\in H$，于是 $g=h\cdot(h^{-1}g)\in H$ ✓。

  > **若没有 $N\le H$**，一般只能得到 $\pi^{-1}(\pi(H))=HN\supseteq H$（"$H$ 的膨胀"）。**例**：$G=S_3$、$N=A_3$、$H=\langle(1,2)\rangle$，则 $\pi^{-1}\pi(H)=HN=S_3\ne H$——包含 $N$ 这个条件不可省。

**(2) $\Phi\circ\Psi=\mathrm{id}$，即 $\pi(\pi^{-1}(K))=K$（$K\le G/N$）**：

- $(\subseteq)$ $\pi(\pi^{-1}(K))\subseteq K$ 由原像定义。
- $(\supseteq)$ 取 $k\in K\le G/N$，则 $k$ 是某个陪集 $gN$；由 $\pi$ 满射取 $g\in G$ 使 $\pi(g)=k$，则 $g\in\pi^{-1}(K)$，故 $k=\pi(g)\in\pi(\pi^{-1}(K))$ ✓。

由 (1)(2)，$\Phi$ 与 $\Psi$ 互逆，是双射 ✓。

**(3) 保持包含。** $H\subseteq H'\Rightarrow\pi(H)\subseteq\pi(H')$ 显然。反向：设 $\pi(H)\subseteq\pi(H')$、取 $h\in H$，则 $\pi(h)\in\pi(H)\subseteq\pi(H')$，即 $hN=h'N$ 某 $h'\in H'$；由 (1) 的论证（$N\subseteq H'$）得 $h\in H'$ ✓。

**(4) 保持正规性。**
- **($\Rightarrow$)** $H\triangleleft G$、$N\le H$ 时 $H/N\triangleleft G/N$：任取 $gN\in G/N$、$hN\in H/N$，
  $$(gN)(hN)(gN)^{-1}=(ghg^{-1})N\in H/N\quad(\text{因}\ ghg^{-1}\in H)✓。$$
- **($\Leftarrow$)** 设 $H/N\triangleleft G/N$。任取 $g\in G,h\in H$：$\pi(ghg^{-1})=(gN)(hN)(gN)^{-1}\in H/N$，故 $ghg^{-1}\in\pi^{-1}(H/N)=H$（用 (1)）✓。

故 $H\triangleleft G$ ✓。$\blacksquare$

> **这条定理的直观**：$G/N$ 的子群格就是 $G$ 的子群格中"**$N$ 以上**"那一段的**缩略图**——把 $N$ 及以下全压成一个点。所以"列大群的子群"可以先"列小商的子群"再往上还原（见例 5.14）。
>
> **例 5.13（$\mathbb{Z}$ 的子群格）**：$G=\mathbb{Z}$，$N=6\mathbb{Z}$。$\mathbb{Z}$ 中所有子群是 $m\mathbb{Z}$（[例 1.18](./01-introduction-to-group-theory.md)），其中包含 $6\mathbb{Z}$ 的恰是 $6\mid m$ 的那些（$m\mathbb{Z}\supseteq6\mathbb{Z}\iff6\in m\mathbb{Z}\iff m\mid6$）。于是
>
> | $G$ 中含 $N$ 的子群 | $\mathbb{Z}$ | $2\mathbb{Z}$ | $3\mathbb{Z}$ | $6\mathbb{Z}$ |
> |---|---|---|---|---|
> | 对应 $G/N=\mathbb{Z}/6\mathbb{Z}$ 的子群 | 全体（阶 6） | $\{\bar0,\bar2,\bar4\}$（阶 3） | $\{\bar0,\bar3\}$（阶 2） | $\{\bar0\}$（阶 1） |
>
> **注意整除方向与包含方向相反**：$6\mid2$ 对应 $6\mathbb{Z}\subseteq2\mathbb{Z}$——即"$m$ 越小，子群越大"。**这个表把 $\mathbb{Z}$ 的无穷子群格在模掉 $6\mathbb{Z}$ 后压成了 4 个点**，正是"商 = 缩略图"的展示。

**例 5.14（$S_4/V_4\cong S_3$，以及用它省力）**：$N=V_4=\{e,(1,2)(3,4),(1,3)(2,4),(1,4)(2,3)\}$（$S_4$ 的正规子群，因为它是 3 个双对换的共轭类再并上 $\{e\}$，[06 共轭类筛子](./06-normal-subgroup-and-quotient-group.md)）。

**(1) $N\triangleleft S_4$**：$V_4$ 是 $\{e\}$ 与"3 个双对换"这个共轭类之并，正好是 $S_4$ 的两个共轭类之并（[03 的表](./03-symmetric-and-cyclic-groups.md)）✓。

**(2) $|S_4/N|=24/4=6$**。

**(3) $S_4/V_4\cong S_3$**：
- **$S_4/V_4$ 无 6 阶元素**：$S_4$ 中的 6 阶元素不存在（$S_4$ 元素阶只有 $1,2,3,4$——按 cycle type 看：长度为 $1,2,3,4$ 的 cycle 阶分别为 $1,2,3,4$，两个不相交 cycle 的组合如 $(3,1)$ 给 3、$(2,2)$ 给 2，故无 6），故商群里也没有 6 阶元素。
- 于是 $|S_4/V_4|=6$ 且无 6 阶元素，由[阶 6 分类（命题 3.41）](./05-group-properties-II.md) $S_4/V_4\cong S_3$ ✓（**非 $C_6$**，因为 $C_6$ 有 6 阶元素）。
- **也可直接构造同构**：$S_4$ 作用在"$V_4$ 的三个非单位元"上给出 $\phi:S_4\to S_3$（**这 3 个元素被置换**），核正是 $V_4$，由第一同构定理 $S_4/V_4\hookrightarrow S_3$，两边阶都是 6，故同构 ✓。

**(4) 用对应定理省力的地方**：想知道 $S_4$ 中含 $V_4$ 的子群结构（$S_4$ 的 30 个子群中哪些含 $V_4$？很痛苦），改看 $S_3$ 的子群格（只有 6 个元素，子群格小得多）：
$$S_3\text{ 的子群格}: \{e\}\subset C_3\subset S_3,\quad \{e\}\subset C_2\subset S_3\ (\text{3 个不同的 } C_2).$$
对应回 $S_4$：含 $V_4$ 的子群恰是 $V_4$、$A_4$（$\leftrightarrow C_3$）、$D_4$ 型的三个（$\leftrightarrow$ 三个 $C_2$）、$S_4$——**树状结构一目了然**。

### 有限阿贝尔群分类

这一节把整个系列收口。**证明链条**：Cauchy 引理 → $p$-部分分离（引理 5.15）→ 关键引理（最高阶元素生成直因子）→ 定理 5.16。原文把 Cauchy 放在引理 5.15 之后，但引理 5.15 的完整证明要用到它，故这里**先证 Cauchy**。

**引理 5.15a（Cauchy 型引理）**：$G$ 有限阿贝尔，$p$ 为质数且 $p\mid|G|$ $\Rightarrow$ $G$ 含 $p$ 阶元素。

**证明（对 $|G|$ 归纳）：**

- **$G$ 循环**（$G\cong\mathbb{Z}/n$，$n=|G|$）：取生成元 $g$，$o(g)=n$。因 $p\mid n$，元素 $g^{n/p}$ 的阶是 $n/\gcd(n,n/p)=n/(n/p)=p$ ✓。
- **$G$ 非循环**：取非单位元 $x$，令 $H=\langle x\rangle$，则 $1<|H|<|G|$（非循环故 $H\neq G$）。
  - **若 $p\mid|H|$**：$H$ 循环且 $p$ 整除其阶，由上一情形 $H$ 中有 $p$ 阶元素 ✓。
  - **若 $p\nmid|H|$**：由拉格朗日 $|G|=|H|\cdot|G/H|$，故 $p\mid|G/H|$。又 $|G/H|=|G|/|H|<|G|$，由归纳假设商群 $G/H$ 有 $p$ 阶元素，记为 $yH$。
    则由 $o(yH)=p$ 有 $y\notin H$ 且 $y^p\in H$。又由同态性 $o(yH)\mid o(y)$（[命题 3.7](./04-group-properties-I.md) 的"同态不升阶"），故 $p\mid o(y)$。令 $o(y)=ps$。
    取 $z:=y^{s}$。则
    $$o(z)=\frac{o(y)}{\gcd(o(y),s)}=\frac{ps}{s}=p$$
    （用 $o(y^s)=o(y)/\gcd(o(y),s)$，即[定理 2.41](./03-symmetric-and-cyclic-groups.md)）✓。

  $\blacksquare$

  > 这个引理对**非阿贝尔**群也成立（真正的 Cauchy 定理），但证明需更细的群作用论证；本系列只用到阿贝尔情形。它马上会在引理 5.15 证明的第 (5) 步派上用场。

**引理 5.15（$p$-部分与 $p'$-部分分离）**：$G$ 有限阿贝尔，$|G|=p^am$（$p\nmid m$）。令
$$H=\{x\in G:x^{p^a}=e\},\qquad K=\{x\in G:x^m=e\}.$$
则 $H,K\le G$、$H\cap K=\{e\}$、$G=HK$，故
$$G=H\times K,\qquad |H|=p^a,\quad |K|=m.$$

**证明：**

**(1) $H,K$ 是子群。** 它们分别是同态 $\phi_{p^a}(x)=x^{p^a}$ 与 $\phi_m(x)=x^m$ 的核；而 $G$ 阿贝尔时 $x\mapsto x^k$ 是同态（$(xy)^k=x^ky^k$，[01 交换情形幂律](./01-introduction-to-group-theory.md)）。由[命题 5.1(1)](./08-isomorphism-theorems-I.md) 核是子群 ✓。

**(2) $H\cap K=\{e\}$。** 取 $x\in H\cap K$，则 $o(x)\mid p^a$（$x^{p^a}=e$）且 $o(x)\mid m$。因 $\gcd(p^a,m)=1$，故 $o(x)\mid1$，即 $o(x)=1$，$x=e$ ✓。

**(3) $G=HK$：用 Bézout 分解每个元素。** 因 $\gcd(p^a,m)=1$，存在整数 $r,s$ 使
$$rp^a+sm=1.$$
任取 $g\in G$，则
$$g=g^{1}=g^{rp^a+sm}=\underbrace{g^{rp^a}}_{\in K}\cdot\underbrace{g^{sm}}_{\in H},$$
其中 $g^{rp^a}\in K$ 因为 $(g^{rp^a})^m=(g^{p^am})^{r}=e^r=e$（**用了 $|G|=p^am$ 推出 $g^{p^am}=e$**，[推论 3.39(2)](./05-group-properties-II.md)）；$g^{sm}\in H$ 同理。故 $g\in KH=HK$ ✓。

**(4) 结论。** $G$ 阿贝尔故 $H,K$ 都正规；配合 (2)(3) 用[内直积判定（定理 4.25）](./07-automorphism-and-products.md) 得 $G=H\times K$ ✓。

**(5) 大小。** $G=H\times K$ 给出 $|G|=|H|\cdot|K|$。$H$ 中每个元素的阶都整除 $p^a$，故 $|H|$ 是 $p$ 的幂（否则某质数 $q\ne p$ 整除 $|H|$，由引理 5.15a $H$ 有 $q$ 阶元素——它的阶整除 $q$ 却不整除 $p^a$，矛盾）。写 $|H|=p^b$（由拉格朗日 $p^b\mid p^am$ 且 $p\nmid m$，得 $b\le a$），于是 $|K|=|G|/|H|=p^{a-b}m$。
若 $b<a$，则 $p\mid|K|$，由引理 5.15a 得 $K$ 中有 $p$ 阶元素 $y$；但 $K$ 中元素都满足 $y^m=e$，故 $o(y)\mid m$，而 $o(y)=p$ 给出 $p\mid m$，与 $\gcd(p^a,m)=1$ 矛盾。故 $b=a$，即 $|H|=p^a$、$|K|=m$ ✓。$\blacksquare$

> **推论**（原文引理 5.15(2)）：$|G|=p_1^{n_1}\cdots p_r^{n_r}$ 时反复应用引理 5.15，得
> $$G=G_1\times\cdots\times G_r,\qquad |G_j|=p_j^{n_j},$$
> 其中 $G_j=\{x:x^{p_j^{n_j}}=e\}$ 是唯一的 $p_j$-Sylow 子群。**"阿贝尔群按质数分裂"这一步是免费的**（Bézout 送上门），而**非阿贝尔群做不到**（$S_3$ 不能拆成 $C_2\times C_3$，它是 $C_3\rtimes C_2$）——这正是[半直积](./07-automorphism-and-products.md)存在的理由。
>
> **例**：$G=C_{6}\times C_{10}$（$|G|=60=2^2\cdot3\cdot5$）时
> $$G\cong(C_2\times C_2)\times C_3\times C_5,$$
> 因为 $C_6=C_2\times C_3$、$C_{10}=C_2\times C_5$（[定理 4.23](./07-automorphism-and-products.md)）。

**引理 5.15b（关键引理：最高阶元素生成直因子）**：$G$ 有限阿贝尔，$x\in G$ 的阶**最大**（记为 $n$）。则存在 $K\le G$ 使
$$G=\langle x\rangle\times K.$$

**证明分两步。**

**第一步，$x^n=e$ 对一切 $x\in G$ 成立**（即 $n$ 就是 $G$ 的**指数**）。等价地：**每个元素的阶都整除 $n$**。

反设存在 $y$ 使 $o(y)=m\nmid n$。取质数 $p$ 使 $v_p(m)>v_p(n)$（$v_p$ 表示素数指数；存在这样的 $p$，否则 $m\mid n$）。令
$$t:=\frac{n}{p^{v_p(n)}},\qquad u:=\frac{m}{p^{v_p(m)}},$$
则 $o(x^{p^{v_p(n)}})=n/p^{v_p(n)}=t$（[定理 2.41](./03-symmetric-and-cyclic-groups.md)）且 $o(y^{u})=p^{v_p(m)}$。两个阶 $t$ 与 $p^{v_p(m)}$ **互质**（$t$ 不含因子 $p$，而 $p^{v_p(m)}$ 是纯 $p$ 幂），故由 $G$ 的交换性
$$o\bigl(x^{p^{v_p(n)}}y^{u}\bigr)=t\cdot p^{v_p(m)}=\frac{n}{p^{v_p(n)}}\cdot p^{v_p(m)}=n\cdot p^{v_p(m)-v_p(n)}>n,$$
与 $n$ 的最大性矛盾 ✓。

**第二步，构造 $K$（对 $|G|$ 归纳）。**

- **$G=\langle x\rangle$**（含 $|G|=1$ 的情形）：取 $K=\{e\}$ ✓。
- **否则** $\langle x\rangle<G$，故 $\bar G:=G/\langle x\rangle$ 是非平凡有限阿贝尔群。取 $\bar G$ 的一个**极大子群** $\bar M$（有限性保证存在），则 $\bar G/\bar M$ 是有限阿贝尔单群，故 $\cong C_p$（某质数 $p$——若阶为合数，由引理 5.15a 有质数阶子群，是既非平凡也非全体的真子群，与单性矛盾）。由[引理 5.11 与子群对应定理（定理 5.12）](#子群对应定理)把 $\bar M$ 拉回：$N:=\pi^{-1}(\bar M)$ 满足 $\langle x\rangle\le N<G$；再由[第三同构定理（定理 5.10）](./08-isomorphism-theorems-I.md)，
  $$G/N\cong\bar G/\bar M\cong C_p,\qquad [G:N]=p.$$
  **"到商群里找子群、再拉回大群"——本篇的主角恰好在这个证明里上岗。**
  由归纳假设（$N$ 中元素的阶都不超过 $n$，而 $x\in N$ 的阶恰为 $n$，故 $x$ 在 $N$ 中仍阶最大；$|N|=|G|/p<|G|$）：
  $$N=\langle x\rangle\times K_1\qquad\text{某}\ K_1\le N.$$
- **先证 $p\mid n$**。取 $y\in G\setminus N$。因 $[G:N]=p$，$G/N\cong C_p$ 故 $y^p\in N$；且 $yN$ 的阶为 $p$，由**同态不升阶**得 $p\mid o(y)$，而 $o(y)\mid n$（第一步），故 $p\mid n$ ✓。
- **关键计算**：由第一步，$y^n=e$，故 $(y^p)^{n/p}=y^n=e$，即 $o(y^p)\mid n/p$。
  写 $y^p\in N=\langle x\rangle\times K_1$ 为
  $$y^p=h\,k,\qquad h\in\langle x\rangle,\ k\in K_1.$$
  在直积 $N=\langle x\rangle\times K_1$ 中把 $y^p=hk$ 升 $n/p$ 次幂（交换性下分量各自进行）：$e=(y^p)^{n/p}=h^{n/p}k^{n/p}$，由直积表示的唯一性得 $h^{n/p}=e$（同时 $k^{n/p}=e$），故 $o(h)\mid n/p$。而 $\langle x\rangle\cong\mathbb{Z}/n$ 中**阶整除 $n/p$ 的元素恰是 $p$ 次幂**：$o(x^i)=n/\gcd(n,i)\mid n/p\iff p\mid\gcd(n,i)\iff p\mid i$，即 $x^i\in\langle x^p\rangle$。故
  $$h=(x^{s})^{p}\quad\text{某整数 } s.$$
- **修正 $y$，把 $y^p$ 推进 $K_1$**：令
  $$y':=y\,(x^{s})^{-1}.$$
  则 $y'\notin N$（因 $y'\in yN$，且 $y\notin N$），且
  $$(y')^{p}=y^{p}\bigl((x^{s})^{p}\bigr)^{-1}=(h\,k)h^{-1}=k\in K_1$$
  ✓（用了交换性）。**这就把"$y^p$ 的 $\langle x\rangle$-分量"消掉了**——本证明的核心一招。
- **验证 $G=\langle x\rangle\times K$，其中 $K:=K_1\langle y'\rangle$**：
  - **$\langle x\rangle K=G$**：$\langle x\rangle K\supseteq\langle x\rangle K_1=N$，且 $y'\in K$、$y'\notin N$，故 $\langle x\rangle K$ 是**严格包含 $N$ 的子群**（$G$ 阿贝尔，子群之和仍是子群）；$[G:N]=p$ 为质数故 $N$ 极大，于是 $\langle x\rangle K=G$ ✓。
  - **$\langle x\rangle\cap K=\{e\}$**：取 $g\in\langle x\rangle\cap K$。$K=K_1\langle y'\rangle$，写 $g=k_1\,(y')^{i}$（$k_1\in K_1$，$i\in\mathbb{Z}$）。
    投到 $G/N\cong C_p$（其中 $\bar{k_1}=N$、$\bar{g}=N$ 因 $g\in\langle x\rangle\le N$）：
    $$N=\bar g=\bar{k_1}\,(\bar{y'})^{i}=(\bar{y'})^{i}.$$
    因 $\bar{y'}$ 是 $G/N$ 的生成元（阶 $p$），故 $p\mid i$，写 $i=pj$。于是
    $$(y')^{i}=\bigl((y')^{p}\bigr)^{j}=k^{j}\in K_1,$$
    故 $g=k_1k^{j}\in K_1$。而 $g\in\langle x\rangle$，故 $g\in\langle x\rangle\cap K_1=\{e\}$ ✓。
  - 由 $\langle x\rangle,K$ 都是正规子群（$G$ 阿贝尔）、交平凡、乘积为 $G$，用[内直积判定](./07-automorphism-and-products.md) 得 $G=\langle x\rangle\times K$ ✓。

$\blacksquare$

> **这个引理的陡度在哪**：它说"**最高阶元素所在的那个循环子群，总能干净地切下来**"。有了它，分类变成机械的归纳：切一刀得 $G=\langle x\rangle\times K$，再对 $K$ 切一刀，直到切完——**每一步都把一个循环因子剥离**。
>
> **例（$C_6\times C_4$，切一刀看）**：$|G|=24$，指数 $n=\operatorname{lcm}(6,4)=12$，最高阶元素如 $x=(g_6,g_4)$（$o(x)=\operatorname{lcm}(6,4)=12$）。引理断言 $G=\langle x\rangle\times K$、$|K|=2$——$K$ 是谁？取 $K=\langle(g_6^3,e)\rangle\cong C_2$，验证内直积两条件：
> - **交平凡**：$(g_6^i,g_4^i)\in K$ 要求第二坐标 $g_4^i=e$（即 $4\mid i$）且第一坐标 $g_6^i\in\{e,g_6^3\}$（即 $i\equiv0,3\pmod 6$）；$i\equiv3\pmod6$ 与 $4\mid i$ 不相容（一奇一偶），$i\equiv0\pmod6$ 且 $4\mid i$ 给 $12\mid i$，此时元素为 $(e,e)$。故 $\langle x\rangle\cap K=\{(e,e)\}$ ✓。
> - **覆盖**：$|\langle x\rangle\cdot K|=12\cdot2=24=|G|$ ✓。
>
> 于是 $C_6\times C_4\cong C_{12}\times C_2$。**注意 $K$ 不能取 $\{e\}\times\langle g_4^2\rangle$**：$(g_6^6,g_4^6)=(e,g_4^2)$ 同时落在 $\langle x\rangle$ 与这个候选里，交不平凡——**引理保证合适的 $K$ 存在，但不是随手抓一个 2 阶子群都行**。$C_6\times C_4$ 与 $C_{12}\times C_2$ 是"不同写法同一个群"，这正是下面"不变因子"的意义。

**定理 5.16（有限阿贝尔群分类）**：$G$ 有限阿贝尔 $\Rightarrow$
$$G\cong C_{p_1^{n_1}}\times\cdots\times C_{p_s^{n_s}}\qquad(p_i\ \text{质数，可重复}).$$

**证明（对 $|G|$ 归纳）：**

- **$|G|=1$**：空乘积，成立。
- **归纳步**：取 $x\in G$ 阶最大，由引理 5.15b 得 $G=\langle x\rangle\times K$。分两种情况：
  - 若 $\langle x\rangle=\{e\}$，则 $G=K=\{e\}$，属于基例 ✓。
  - 若 $\langle x\rangle\ne\{e\}$，则 $|K|=|G|/o(x)<|G|$（$o(x)>1$），由归纳假设 $K$ 分解为质数幂循环群之积。而循环群 $\langle x\rangle$ 由[定理 4.23](./07-automorphism-and-products.md) 分解为质数幂循环群之积。于是 $G$ 是二者之积，也是质数幂循环群之积 ✓。$\blacksquare$

**唯一性（为什么"阶决定一切"）**：设
$$G\cong C_{p^{a_1}}\times\cdots\times C_{p^{a_k}}\qquad(\text{固定质数 } p,\ a_1\ge\cdots\ge a_k).$$
定义**不变量**（不依赖分解，只依赖 $G$ 的同构型）
$$N_j:=\bigl|\{x\in G:x^{p^j}=e\}\bigr|.$$
在分解下 $\{x:x^{p^j}=e\}=\bigoplus_i\{x\in C_{p^{a_i}}:x^{p^j}=e\}$，每个因子贡献 $p^{\min(j,a_i)}$ 个元素，故
$$N_j=p^{\sum_i\min(j,a_i)}.$$
设 $e_j:=\#\{i:a_i\ge j\}$，则 $\sum_i\min(j,a_i)=\sum_{l=1}^{j}e_l$，于是
$$\frac{N_j}{N_{j-1}}=p^{e_j}\ \Longrightarrow\ e_j=\log_p\frac{N_j}{N_{j-1}},$$
即每个 $e_j$ 由 $G$ 唯一决定，从而重数多重集 $\{a_1,\dots,a_k\}$ 唯一决定。再结合引理 5.15 按质数分裂，**整个同构型由 $G$ 唯一决定**。$\blacksquare$

> **这个唯一性证明的技巧值得记**：**不看群内部，只看"被 $p^j$ 杀死的元素有多少个"**。这类"数元素个数得结构不变量"的手法（配合 $\Omega_k(G)=\{x:x^{p^k}=e\}$ 的分层）在 $p$-群理论里是标准工具。

**定理 5.17（不变因子形式）**：分类的等价写法是
$$G\cong C_{d_1}\times\cdots\times C_{d_k}\qquad(d_k\mid d_{k-1}\mid\cdots\mid d_1).$$

**从 5.16 到 5.17 的换算**：把 5.16 的质数幂因子按质数分组，每组内按指数从大到小排齐，**竖着相乘**。**例**：
$$C_2\times C_4\times C_3\times C_9\to\begin{cases}d_1=4\cdot9=36\\ d_2=2\cdot3=6\end{cases}\ \Longrightarrow\ G\cong C_{36}\times C_6,$$
且 $6\mid36$ ✓。**不变因子形式的优点**：$d_1$ 是 $G$ 的指数（最大元素阶），$\prod_i d_i=|G|$，各 $d_i$ 都不依赖质数的罗列顺序。**注意整除链 $d_{i+1}\mid d_i$ 是"小的整除大的"**。

**例（阶 8 完全分类，共 5 个群）**：

**阿贝尔（3 个）**：由定理 5.16，$8=2^3$，故 $G$ 是 8 阶阿贝尔 2-群，分解对应 $3$ 的**分拆**：
$$3=3\ \to\ C_8;\qquad 3=2+1\ \to\ C_4\times C_2;\qquad 3=1+1+1\ \to\ C_2^3.$$

**非阿贝尔（2 个）**：设 $G$ 是 8 阶非阿贝尔群。

- 由拉格朗日与引理 5.15a，元素阶只能是 $1,2,4,8$。若**存在 8 阶元素** $g$，则 $G=\langle g\rangle$ 循环，阿贝尔，排除。故元素阶 $\in\{1,2,4\}$。
- **若所有非单位元阶为 2**：$G$ 阿贝尔（$xy=(xy)^{-1}=yx$），排除。故**存在 4 阶元素** $a$。
- 令 $H=\langle a\rangle$，$[G:H]=8/4=2$，故 $H\triangleleft G$（[定理 4.2](./06-normal-subgroup-and-quotient-group.md)）。取 $b\in G\setminus H$，则 $G=H\cup bH$，且 $b^2\in H$（由 $[G:H]=2$ 或 $G/H\cong C_2$）。
  - $b^2\in\{e,a,a^2,a^3\}$，而 $o(b)\in\{2,4\}$（非 8），故 $b^2\in\{e,a^2\}$（$b^2=a$ 或 $a^3$ 给 $o(b)=8$，排除）。
  - 又由 $H\triangleleft G$，$bab^{-1}\in H$，且 $bab^{-1}$ 与 $a$ 同阶（共轭保阶）故 $\in\{a,a^3\}$。分**四种情形**：

  | $b^2$ | $bab^{-1}$ | 群 | 依据 |
  |---|---|---|---|
  | $e$ | $a$ | $C_4\times C_2$ | 由[内直积判定](./07-automorphism-and-products.md)（$b$ 与 $a$ 交换、交平凡、覆盖） |
  | $e$ | $a^{-1}$ | $D_4$ | 正是 $D_4=\langle r,m:r^4=m^2=e,mrm^{-1}=r^{-1}\rangle$ |
  | $a^2$ | $a$ | $C_4\times C_2$ | 见下 |
  | $a^2$ | $a^{-1}$ | $Q_8$ | 见下 |

  **第 3 行（$b^2=a^2,\ bab^{-1}=a$）**：此时 $a,b$ 交换，故 $G=\langle a,b\rangle$ 阿贝尔。而 $(ab)^2=abab=a^2b^2=a^2a^2=a^4=e$，且 $ab\notin H$（否则 $b\in H$），故 $o(ab)=2$，$\langle a\rangle\cap\langle ab\rangle=\{e\}$，于是 $G=\langle a\rangle\times\langle ab\rangle\cong C_4\times C_2$ ✓。
  **第 4 行（$b^2=a^2,\ bab^{-1}=a^{-1}$）**：这就是**四元数群** $Q_8=\{\pm1,\pm i,\pm j,\pm k\}$（令 $a\mapsto i,\ b\mapsto j$，则 $ab\mapsto k$）。$Q_8$ 非阿贝尔 ✓。

  **第 1、3 行给出同一个群 $C_4\times C_2$**（不同选取 $b$ 的结果），故非阿贝尔只有 $D_4$ 与 $Q_8$ 两个 ✓。

- **$D_4\not\cong Q_8$ 的最后确认**：两者的元素阶谱都是 $1,2,4$，靠阶谱分不开，改用**各阶元素的个数**：
  - $D_4$：$r,r^3$ 是 4 阶（2 个）；$r^2$ 与 4 个翻转是 2 阶（5 个）。
  - $Q_8$：$i,j,k,-i,-j,-k$ 全是 4 阶（6 个）；只有 $-1$ 是 2 阶（1 个）。

  **同一个群中 4 阶元素个数是共轭/自同构不变量**（更简单地说：$\#\{x\in G:o(x)=4\}$ 是同构不变量），$2\neq6$ 故**不同构** ✓。

**总表**：

| $G$（$|G|=8$） | 阿贝尔 | 2 阶元素个数 | 4 阶元素个数 | 结构 |
|---|---|---|---|---|
| $C_8$ | 是 | 1 | 2 | $\langle r\rangle,\ r^8=e$ |
| $C_4\times C_2$ | 是 | 3 | 4 | $\langle r\rangle\times\langle s\rangle$ |
| $C_2^3$ | 是 | 7 | 0 | 初等阿贝尔 |
| $D_4$ | 否 | 5 | 2 | $C_4\rtimes C_2$ |
| $Q_8$ | 否 | 1 | 6 | 非分裂扩张 $1\to C_4\to Q_8\to C_2\to1$ |

> **"结构"一列的对比**：$D_4=C_4\rtimes C_2$（非平凡 $\phi$，[07 例 4.24 后](./07-automorphism-and-products.md)）；而 $Q_8$ **不能**写成 $C_4\rtimes C_2$——$Q_8$ 的 4 阶子群 $\langle i\rangle$ 指数为 2、正规，但**没有**补子群与之交平凡且覆盖（$\langle i\rangle\cap\langle j\rangle=\{\pm1\}$）。**所以 $Q_8$ 是非分裂扩张的第一个例子**：$C_4$ 被 $C_2$ 扩张而成，但不裂。
>
> **这张表也是检验工具箱的试金石**：3 个阿贝尔来自分类定理（分拆 3 的三种方式），2 个非阿贝尔来自上面的分支枚举。**阶 8 是"分类能靠手算撑住"的极限**——再往上一级（16 阶有 14 个群）就需要 Sylow 与群作用理论了。

## 为什么重要

- 这是本系列的**收官定理**：它宣布有限阿贝尔群在"质数幂循环群直积"意义下完全可知——阶定了，可能性就有限
- 子群对应定理是实用工具：大群的子群结构难列时，转去小商群里列（$S_4\to S_3$ 的降维打击）
- [魔方群](../rubiks-cube-group-theory.md) 的商 $G/G_0\cong\mathbb{Z}_3\times\mathbb{Z}_2\times\mathbb{Z}_2$ 可以用本篇立刻验证：$3,2,2$ 两两互质，直积不塌缩

## 局限 / 边界

- 分类只覆盖**阿贝尔**；非阿贝尔没有类似的总定理（小阶靠 Sylow + 群作用，本系列未覆盖）
- 质数幂循环群部分 $C_{p^2}\ne C_p\times C_p$（[07](./07-automorphism-and-products.md) 例 4.24）——同幂次不互质，分解必须到质数幂一层为止
- $Q_8$（四元数群）在原文只是提及，未展开

## 我的理解

- 有限阿贝尔群分类的证明结构值得单独记：**先用互质性把不同质数拆开（引理 5.15），再在固定质数幂内部归纳切出最高阶循环因子**。第一步靠 Bézout，第二步靠同构定理链条
- 子群对应定理最反直觉的点是 $\pi^{-1}(\pi(H))=H$ 会**严格**成立（而非只包含）——原因是 $N\subseteq H$；没有这个包含，$\pi^{-1}(\pi(H))=HN$ 只是个乘积
- 阶 8 的 5 个群是检验工具箱的试金石：3 个阿贝尔来自分类定理（分拆 3 的三种方式），$D_4$ 来自 $C_4\rtimes C_2$ 的非平凡 $\phi$，$Q_8$ 是四元数群（第一个非分裂扩张）

## Related

- [08 同构定理（上）](./08-isomorphism-theorems-I.md) — 本篇所有证明的引擎
- [07 自同构、直积与半直积](./07-automorphism-and-products.md) — $C_n$ 质数幂分解、内直积判定
- [05 群的性质（下）](./05-group-properties-II.md) — 阶 6 分类（分类思想的小型预演）
- [魔方群的数学结构](../rubiks-cube-group-theory.md) — $G/G_0\cong\mathbb{Z}_3\times\mathbb{Z}_2\times\mathbb{Z}_2$
