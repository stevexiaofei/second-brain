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

**定义 4.16**：**自同构**是 $G\to G$ 的群同构；全体记 $Aut(G)$。

**$Aut(G)$ 成群**：封闭（同构的复合是同构）、结合律（复合）、单位元（$\mathrm{id}_G$）、逆元（$\varphi^{-1}$ 也是同构，[04](#)）。

**定义 4.17（内自同构）**：$\lambda_g(x)=gxg^{-1}$（叫"用 $g$ 共轭"）。

**证明 $\lambda_g$ 是自同构**：

- **同态**：$\lambda_g(xy)=g(xy)g^{-1}=(gxg^{-1})(gyg^{-1})=\lambda_g(x)\lambda_g(y)$ ✓（中间插入 $g^{-1}g$）。
- **双射**：$\lambda_g\circ\lambda_{g^{-1}}(x)=g(g^{-1}xg)g^{-1}=x$，同理反向，故 $\lambda_{g^{-1}}=\lambda_g^{-1}$ ✓。

全体内自同构记 $Inn(G)=\{\lambda_g:g\in G\}$；非内的自同构称**外自同构**。

**$g\mapsto\lambda_g$ 是同态 $G\to Aut(G)$**：$\lambda_{gh}(x)=(gh)x(gh)^{-1}=g(hxh^{-1})g^{-1}=\lambda_g(\lambda_h(x))$，故 $\lambda_{gh}=\lambda_g\circ\lambda_h$ ✓。**它的核是** $\{g:\lambda_g=\mathrm{id}\}=\{g:gxg^{-1}=x\ \forall x\}=Z(G)$，故由[第一同构定理](./08-isomorphism-theorems-I.md)
$$Inn(G)\cong G/Z(G).$$

**例 4.18**：$G$ 阿贝尔时 $\lambda_g(x)=gxg^{-1}=x$，故 $Inn(G)=\{1\}$——**阿贝尔群没有内扭曲**。与上面的同构式一致（$Z(G)=G$，$G/Z(G)$ 平凡）。

**例 4.19（$Aut(D_3)\cong D_3$）**：

**第一步，$|Aut(D_3)|=6$。** 自同构保阶（[04 命题 3.7](./04-group-properties-I.md)），故 $\phi(r)$ 必须是 3 阶元素（$D_3$ 中 3 阶元素只有 $r,r^2$）、$\phi(m)$ 必须是 2 阶元素（$D_3$ 中 2 阶元素有 $m,rm,r^2m$ 三个）。且 $\phi$ 被 $\phi(r),\phi(m)$ 完全决定（$r,m$ 生成 $D_3$），故 $|Aut(D_3)|\le2\cdot3=6$。

**第二步，6 个候选都是同构。** 但要排除"$\phi(r)=x,\phi(m)=y$ 任意配"的情形——**必须保持关系** $mrm^{-1}=r^{-1}$，即
$$yxy^{-1}=x^{-1}.$$
- 取 $x$ 为 3 阶元素、$y\in\{m,rm,r^2m\}$ 逐个验证：例如 $x=r,\ y=rm$ 时
  $$(rm)r(rm)^{-1}=(rm)r(m^{-1}r^{-1})=(rm)r(m r^2)=r(mrm)r^2=r(r^{-1}m m)r^2=r(r^{-1})r^2=r^2=x^{-1}$$
  ✓。其余 5 个同法验证（用 $mr=r^{-1}m$ 化简），6 个候选**全部满足**关系式。
- 这说明 6 个候选确实给出 $D_3$ 的自同构，故 $|Aut(D_3)|=6$。

**第三步，$Aut(D_3)$ 非阿贝尔，故 $\cong D_3$。** 取两个自同构
$$\alpha:\ r\mapsto r,\ m\mapsto rm;\qquad \beta:\ r\mapsto r^2,\ m\mapsto m$$
（两者都保 $r$ 的阶与 $m$ 的阶，且由第二步满足关系式，故确实是自同构）。比较 $\alpha\beta$ 与 $\beta\alpha$ 在 $m$ 上的作用：
$$(\alpha\beta)(m)=\alpha\bigl(\beta(m)\bigr)=\alpha(m)=rm,\qquad (\beta\alpha)(m)=\beta(rm)=\beta(r)\beta(m)=r^2m.$$
因 $rm\neq r^2m$，故 $\alpha\beta\neq\beta\alpha$，$Aut(D_3)$ 非阿贝尔 ✓。

又 $|Aut(D_3)|=6$，而 6 阶群只有 $C_6$ 与 $S_3$ 两种（[05 命题 3.41](./05-group-properties-II.md)）——非阿贝尔的那个是 $S_3\cong D_3$，故
$$Aut(D_3)\cong D_3.$$

> **顺带一条**：$\alpha$ 是内自同构吗？$Inn(D_3)\cong D_3/Z(D_3)=D_3/\{e\}\cong D_3$，而 $|Inn(D_3)|=|Aut(D_3)|=6$，故**$D_3$ 的每个自同构都是内自同构**，$Out(D_3)=1$。这也顺带把上面对 $\alpha,\beta$ 的计算与"内自同构非阿贝尔"接起来了（$D_3$ 非阿贝尔）。

> **一般结论（未证）**：$n\ge3$ 时 $Aut(D_n)\cong(\mathbb{Z}/n\mathbb{Z})^\times\rtimes\mathbb{Z}/n\mathbb{Z}$，大小是 $n\varphi(n)$。对 $n=3$：$3\cdot2=6$ ✓。

**定理 4.20**：$Aut(C_n)\cong(\mathbb{Z}/n\mathbb{Z})^\times$。

**证明：** 设 $C_n=\langle g\rangle$。

**(1) 自同构把生成元送到生成元。** 若 $\phi\in Aut(C_n)$，则对任意 $x=g^k$ 有 $\phi(x)=\phi(g)^k$，故 $\operatorname{im}\phi=\langle\phi(g)\rangle$。而 $\phi$ 满射，故 $\langle\phi(g)\rangle=C_n$，即 $\phi(g)$ 是生成元。

**(2) $C_n$ 的生成元恰是 $g^k$（$\gcd(n,k)=1$）。** 由[定理 2.41](./03-symmetric-and-cyclic-groups.md)，$o(g^k)=n/\gcd(n,k)$，而 $g^k$ 生成 $C_n$ $\iff$ $o(g^k)=n$ $\iff$ $\gcd(n,k)=1$ ✓。

**(3) 对每个 $k=\gcd(n,k)$ 互质的 $k$，定义 $\phi_k(g)=g^k$ 是自同构。** 良定义不需要（$g$ 固定），只需：$\phi_k$ 是映射 $C_n\to C_n$（由 $\phi_k(g^i)=g^{ki}$），同态（$\phi_k(g^i g^j)=g^{k(i+j)}=\phi_k(g^i)\phi_k(g^j)$ ✓），双射（$C_n$ 有限，同态单射即双射；$\phi_k(g^i)=e\Rightarrow n\mid ki\Rightarrow n\mid i$ 因为 $\gcd(k,n)=1$，故 $g^i=e$，单射 ✓）。

**(4) 复合对应乘法。** $\phi_k\circ\phi_l(g)=\phi_k(g^l)=g^{kl}=\phi_{kl}(g)$，故在生成元上
$$\phi_k\circ\phi_l=\phi_{kl}.$$
即映射
$$\Phi:(\mathbb{Z}/n\mathbb{Z})^\times\to Aut(C_n),\qquad \Phi(\bar k)=\phi_k$$
是**良定义的同态**（$\bar k$ 是模 $n$ 类，而 $\phi_k$ 只依赖 $k\bmod n$，因为 $g^{k+nl}=g^k$ ✓）。

**(5) $\Phi$ 是双射。** 满射：由 (1)(2)，任何 $\phi$ 都形如 $\phi_k$ ✓。单射：$\Phi(\bar k)=\mathrm{id}$ $\Rightarrow$ $g^k=g$ $\Rightarrow$ $g^{k-1}=e$ $\Rightarrow$ $n\mid k-1$ $\Rightarrow$ $\bar k=\bar1$ ✓。

故 $Aut(C_n)\cong(\mathbb{Z}/n\mathbb{Z})^\times$。$\blacksquare$

> **例（逐个算）**：
> - $n=5$：$Aut(C_5)\cong(\mathbb{Z}/5\mathbb{Z})^\times\cong C_4$，$|Aut(C_5)|=4$。四个自同构是 $\phi_1,\phi_2,\phi_3,\phi_4$。$\phi_2$ 的性质：$\phi_2(g^i)=g^{2i}$，故 $\phi_2$ 把生成元 $g$ 送 $g^2$。
> - $n=12$：$(\mathbb{Z}/12\mathbb{Z})^\times=\{\bar1,\bar5,\bar7,\bar{11}\}\cong C_2\times C_2$，故 $Aut(C_{12})\cong C_2\times C_2$（$|Aut|=4$）——**非循环**（[02 例](./02-mod-n-and-dihedral-groups.md)），所以自同构群不必循环。
> - $n=6$：$(\mathbb{Z}/6\mathbb{Z})^\times=\{\bar1,\bar5\}\cong C_2$，故 $Aut(C_6)\cong C_2$，只有恒等与"$g\mapsto g^5=g^{-1}$"两个自同构。
> - **一般的大小**是 $\varphi(n)$。如 $\varphi(8)=4$，$Aut(C_8)\cong C_2\times C_2$。
>
> **这个定理的意义**：数论的 $(\mathbb{Z}/n\mathbb{Z})^\times$（[02](./02-mod-n-and-dihedral-groups.md)）与群论的 $Aut(C_n)$ 被接上了。**更深的用处**：建造半直积 $C_n\rtimes_\phi H$ 时，$\phi:H\to Aut(C_n)$ 就是"$H$ 通过哪个模 $n$ 单位作用在每个循环因子上"——$D_n=C_n\rtimes C_2$ 中那个非平凡 $\phi$ 就是 $\bar{-1}\in(\mathbb{Z}/n\mathbb{Z})^\times$（见下）。

### 直积

**定义 4.21**：
- **外直积** $H\times K$：集合为笛卡尔积，运算逐分量 $(h,k)(h',k')=(hh',kk')$
- **内直积**：$G\cong H\times K$ 经 $\phi((h,k))=hk$

**推论 4.22**：$(G_1\times G_2)\times G_3\cong G_1\times(G_2\times G_3)$（结合律，同构意义下）。

证明：映射 $((g_1,g_2),g_3)\mapsto(g_1,(g_2,g_3))$ 是双射同态（逐分量验证）✓。同理可换也是同构，故直积的运算次序无关紧要，可以写 $G_1\times G_2\times G_3$。

**定理 4.23（循环群按质数幂分解）**：$n=p_1^{a_1}\cdots p_k^{a_k}$（质因数分解，$p_i$ 两两不同）时
$$C_n\cong C_{p_1^{a_1}}\times\cdots\times C_{p_k^{a_k}}.$$

**证明：只需证 $\gcd(r,s)=1$ 时 $C_{rs}\cong C_r\times C_s$，再反复使用（因 $p_i^{a_i}$ 两两互质）。**

设 $\gcd(r,s)=1$，$C_{rs}=\langle g\rangle$。定义
$$\phi:C_{rs}\to C_r\times C_s,\qquad \phi(g^k)=(g^{sk},\ g^{rk}).$$
（注意两个分量都落在"较小"的群里：$g^{sk}$ 的周期整除 $r$，$g^{rk}$ 的周期整除 $s$。）

- **同态**：$\phi(g^k g^l)=\phi(g^{k+l})=(g^{s(k+l)},g^{r(k+l)})=(g^{sk},g^{rk})(g^{sl},g^{rl})$ ✓。
- **单射**：$\phi(g^k)=(e,e)$ 即 $g^{sk}=e$ 且 $g^{rk}=e$，即 $rs\mid sk$ 且 $rs\mid rk$；由 $rs\mid sk$ 约去 $s$ 得 $r\mid k$（$r,s$ 互质），故 $g^k=e$ ✓。
- **双射**：$|C_{rs}|=rs=|C_r\times C_s|$，单射即满射 ✓。

故同构。**另一种构造（更显式，用 Bézout）**：找 $a,b$ 使 $ar+bs=1$。则 $g=g^{ar+bs}=(g^r)^b(g^s)^a$——**每个元素都是"$r$-部分的幂"与"$s$-部分的幂"之积**，其中 $g^s$ 的阶是 $r$、$g^r$ 的阶是 $s$。于是
$$\psi:C_r\times C_s\to C_{rs},\qquad \psi(x,y)=xy\ \ (\text{在 } C_{rs} \text{ 中相乘})$$
是同构（由上面的分解保证满射，阶数相同保证单射）。$\blacksquare$

> **例（$C_6\cong C_2\times C_3$，逐元素对上）**：$C_6=\langle g\rangle$，$g^3$ 阶为 2、$g^2$ 阶为 3，$C_6=\langle g^3\rangle\times\langle g^2\rangle$：
>
> | $C_6$ | $e$ | $g$ | $g^2$ | $g^3$ | $g^4$ | $g^5$ |
> |---|---|---|---|---|---|---|
> | 写成 $\langle g^3\rangle\times\langle g^2\rangle$ | $(e,e)$ | $(g^3,g^2)$ | $(e,g^4)$ | $(g^3,e)$ | $(e,g^2)$ | $(g^3,g^4)$ |
>
> （$g=(g^3)^1(g^2)^{-1}=(g^3)(g^4)$ ✓。）**注意"互质才能拆"**：$C_4\not\cong C_2\times C_2$。

**警示（例 4.24）**：$C_{p^2}\ne C_p\times C_p$——前者有 $p^2$ 阶元素（$g$ 本身），后者的非单位元阶全是 $p$（每个分量阶整除 $p$）。**互质才能拆，同质数幂不能拆**。这正是[有限阿贝尔群分类](./09-isomorphism-theorems-II.md)里"分解必须到质数幂一层为止"的原因。

**定理 4.25（内直积判定）**：$G$ 是子群 $H,K$ 的内直积（即 $G\cong H\times K$ 经 $\phi((h,k))=hk$）$\iff$
1. $G=HK$；2. $H\cap K=\{1\}$；3. $H,K\triangleleft G$。

**证明：**

**($\Rightarrow$)** 设 $\phi:H\times K\to G$、$\phi(h,k)=hk$ 是同构。

1. **$G=HK$**：$\phi$ 满射，故每个 $g\in G$ 写作 $hk$ ✓。
2. **$H\cap K=\{1\}$**：取 $x\in H\cap K$。则 $\phi(x,1)=x=\phi(1,x)$，而 $\phi$ 单射，故 $(x,1)=(1,x)$，即 $x=1$ ✓。
3. **$H,K\triangleleft G$**：$\phi$ 是同构，而 $H\times\{1\}\triangleleft H\times K$（直积的因子总是正规的：$(h,k)(h',1)(h,k)^{-1}=(hh'h^{-1},kkk^{-1})=(hh'h^{-1},1)\in H\times\{1\}$ ✓），故 $\phi(H\times\{1\})=H\triangleleft G$ ✓；同理 $K\triangleleft G$。

**($\Leftarrow$)** 设三条成立。定义 $\phi:H\times K\to G,\ \phi(h,k)=hk$。

- **$\phi$ 是映射且满射**：由条件 1，$G=HK$，故每个 $g$ 写成 $hk$ ✓。
- **$\phi$ 是双射**：满射已证；单射：若 $h_1k_1=h_2k_2$，则 $h_2^{-1}h_1=k_2k_1^{-1}\in H\cap K$，由条件 2 得 $h_2^{-1}h_1=1=k_2k_1^{-1}$，即 $h_1=h_2,\ k_1=k_2$ ✓。
- **$\phi$ 是同态**（这就是要用条件 3 的地方）：先证一条引理。

  > **引理**：$H,K\triangleleft G$ 且 $H\cap K=\{1\}$ $\Rightarrow$ $hk=kh$ 对一切 $h\in H,k\in K$。

  引理证明：考察换位子 $hkh^{-1}k^{-1}$。因 $K\triangleleft G$，有 $hkh^{-1}\in K$，故 $hkh^{-1}k^{-1}\in K\cdot K=K$；又因 $H\triangleleft G$，有 $kh^{-1}k^{-1}\in H$，故 $hkh^{-1}k^{-1}=h\cdot(kh^{-1}k^{-1})\in H\cdot H=H$。于是它 $\in H\cap K=\{1\}$，即 $hkh^{-1}k^{-1}=1$，两边右乘 $kh$ 得 $hk=kh$ ✓。

  有了交换性：
  $$\phi\bigl((h_1,k_1)(h_2,k_2)\bigr)=h_1h_2k_1k_2=h_1k_1h_2k_2=\phi(h_1,k_1)\phi(h_2,k_2)$$
  （中间用了引理：$h_2$ 与 $k_1$ 交换）✓。

于是 $\phi$ 是同态且双射，是 $G\cong H\times K$ 的同构 ✓。$\blacksquare$

> **关键的一步是"$H\cap K=\{1\}$ + 都正规 ⇒ 交换"**，即 $hk=kh$ 对一切 $h\in H,k\in K$。**例**：$C_6=\langle g\rangle$ 中 $H=\langle g^3\rangle\cong C_2$、$K=\langle g^2\rangle\cong C_3$，两块都正规（$C_6$ 阿贝尔）、交为 $\{e\}$、乘积覆盖全体，故 $C_6=C_2\times C_3$ ✓（与定理 4.23 一致）。
>
> **反例（缺一条就不行）**：
> - **缺"交为 $\{1\}$"**：$S_3$ 中 $H=A_3$、$K=A_3$，$HK=A_3\ne S_3$ ✗（连条件 1 都不满足）。
> - **缺"正规"**：$S_3$ 中 $H=\langle(1,2)\rangle$、$K=\langle(1,3)\rangle$，$H\cap K=\{e\}$、$|HK|=4$，但 $HK\not\le S_3$（$HK$ 不封闭）——**乘积不是子群，谈不上直积**。这就是[第二同构定理](./08-isomorphism-theorems-I.md)要加 $K\le N_G(H)$ 的原因。
> - **缺"覆盖"**：$\mathbb{Z}/6\mathbb{Z}$ 中 $H=\{\bar0,\bar2,\bar4\}$、$K=\{\bar0,\bar3\}$，$H\cap K=\{\bar0\}$、都正规、$H+K=\mathbb{Z}/6\mathbb{Z}$ 覆盖 ✓——**这个例子三条都满足**，故 $\mathbb{Z}/6\mathbb{Z}=C_3\times C_2$ ✓。

### 半直积

**定义 4.27**：给定同态 $\phi:H\to Aut(N)$，**半直积** $N\rtimes_\phi H$ 的集合是 $N\times H$，运算为
$$(n,h)*(n',h')=\bigl(n\,[\phi(h)(n')],\ hh'\bigr).$$

**验证它确实是群**（四条公理，注意所有扭曲都在 $\phi(h)(n')$ 这一项上）：

1. **封闭**：$\phi(h)(n')\in N$，$hh'\in H$，故结果是 $N\times H$ 中元素 ✓（**这就是要 $\phi(h)\in Aut(N)$ 而不是随便一个映射的原因**）。
2. **结合律**：分别算两边。
   $$\bigl[(n,h)*(n',h')\bigr]*(n'',h'')=\bigl(n\,\phi(h)(n'),\ hh'\bigr)*(n'',h'')=\bigl(n\,\phi(h)(n')\,\phi(hh')(n''),\ hh'h''\bigr).$$
   另一边
   $$(n,h)*\bigl[(n',h')*(n'',h'')\bigr]=(n,h)*\bigl(n'\,\phi(h')(n''),\ h'h''\bigr)=\bigl(n\,\phi(h)\bigl(n'\phi(h')(n'')\bigr),\ hh'h''\bigr).$$
   比较 $N$ 分量，只需
   $$\phi(h)(n')\,\phi(hh')(n'')=\phi(h)\bigl(n'\,\phi(h')(n'')\bigr)=\phi(h)(n')\,\phi(h)\bigl(\phi(h')(n'')\bigr),$$
   即 $\phi(hh')=\phi(h)\circ\phi(h')$——**这正是 $\phi$ 是同态** ✓。
3. **单位元**：$(1_N,1_H)$。验算 $(n,h)*(1,1)=(n\,\phi(h)(1),h)=(n,h)$ ✓，右侧同理（$\phi(1)=\mathrm{id}$）。
4. **逆元**：$(n,h)^{-1}=\bigl(\phi(h^{-1})(n^{-1}),\ h^{-1}\bigr)$。验算：
   $$(n,h)*\bigl(\phi(h^{-1})(n^{-1}),h^{-1}\bigr)=\bigl(n\,\phi(h)(\phi(h^{-1})(n^{-1})),\ hh^{-1}\bigr)=\bigl(n\,n^{-1},\ 1\bigr)=(1,1)$$
   （用了 $\phi(h)\circ\phi(h^{-1})=\phi(1)=\mathrm{id}$）✓；另一侧同理。$\blacksquare$

**即刻可见的结构**：

- **$\phi$ 平凡时退化为直积**：$\phi(h)=\mathrm{id}$ 时 $(n,h)*(n',h')=(nn',hh')$ ✓。
- **base group $\{(n,1)\}\cong N$ 总是正规的**：
  $$(n,h)*(n',1)=\bigl(n\,\phi(h)(n'),\ h\bigr),$$
  再右乘 $(n,h)^{-1}=\bigl(\phi(h^{-1})(n^{-1}),h^{-1}\bigr)$：
  $$\bigl(n\,\phi(h)(n'),\ h\bigr)*\bigl(\phi(h^{-1})(n^{-1}),h^{-1}\bigr)=\bigl(n\,\phi(h)(n')\,\phi(h)\phi(h^{-1})(n^{-1}),\ hh^{-1}\bigr)=\bigl(n\,\phi(h)(n')\,n^{-1},\ 1\bigr).$$
  结果的第二个坐标是 $1$，故属于 $\{(n,1)\}\cong N$ ✓。**注意 $\phi(h)(n')$ 出现在中间**——共轭公式带着 $\phi$，"扭曲"在这里现形。这也是半直积计算最容易出错的一步。
- **$\{(1,h)\}\cong H$ 一般不正规**（只有 $\phi$ 平凡时才正规）——这正是半直积与直积的唯一差别。
- 一般元素可唯一写成 $(n,h)=(n,1)*(1,h)$（"$N$ 部分 $\cdot$ $H$ 部分"），且 $h$ 对 $n$ 的共轭由 $\phi$ 给出：取上面的公式中 $n=1$ 得
  $$(1,h)*(n',1)*(1,h)^{-1}=\bigl(\phi(h)(n'),1\bigr)$$
  ——**这就是 $\phi$ 的几何意义：$\phi(h)$ 就是"$h$ 共轭作用在 $N$ 上"**（与定理 4.30 的 $\lambda_h|_N$ 一致）。

**定理 4.30**：$G=NH$、$H\cap N=\{1\}$、$N\triangleleft G$ 时 $G\cong N\rtimes_\phi H$，其中 $\phi(h)=\lambda_h|_N$（$H$ 通过共轭作用在 $N$ 上）。

**证明**：定义
$$\Phi:N\rtimes_\phi H\to G,\qquad \Phi(n,h)=nh.$$

- **$\phi$ 确实是 $H\to Aut(N)$ 的同态**：$\lambda_h(N)=hNh^{-1}\subseteq N$（$N\triangleleft G$），且 $\lambda_h|_N$ 是自同构（限制共轭）；$\phi(h_1h_2)=\lambda_{h_1h_2}|_N=\lambda_{h_1}\lambda_{h_2}|_N=\phi(h_1)\phi(h_2)$ ✓。
- **$\Phi$ 是双射**：满射由 $G=NH$；单射由 $H\cap N=\{1\}$（同[定理 4.25](#) 的单射论证）。
- **$\Phi$ 是同态**：任取 $(n,h),(n',h')$，
  $$\Phi\bigl((n,h)*(n',h')\bigr)=\Phi\bigl(n\,\phi(h)(n'),hh'\bigr)=n\,\phi(h)(n')\,hh'=n\,(hn'h^{-1})\,hh'=nh\,n'h'=\Phi(n,h)\Phi(n',h')$$
  ✓（在 $G$ 中验证，比在抽象半直积里算方便得多）。

故 $\Phi$ 是同构。$\blacksquare$

> **与直积判定的对照**（值得并列记）：定理 4.25 的直积要三条——覆盖、交平凡、**两块都正规**；定理 4.30 的半直积也是三条——覆盖、交平凡、**只需 $N$ 正规**。**差别只有"$H$ 是否正规"这一条**，而这一条决定"要不要付费"。
>
> **例（$D_n=C_n\rtimes C_2$）**：$G=D_n$、$N=\langle r\rangle\cong C_n$（正规）、$H=\langle m\rangle\cong C_2$（不正规）、$N\cap H=\{1\}$、$NH=D_n$ ✓。$\phi(m)(r)=mrm^{-1}=r^{-1}$，即 $\phi$ 把生成元 $m$ 送到 $\lambda_{-1}\in Aut(C_n)\cong(\mathbb{Z}/n\mathbb{Z})^\times$（[定理 4.20](#)）。**这就是 $mr=r^{-1}m$ 的抽象来源**：$D_n$ 不是直积，因为翻转"付费"把 $r$ 翻成 $r^{-1}$。
>
> **例（$C_4\rtimes C_2$ 两种 $\phi$ 给出两个群）**：$N=C_4=\langle a\rangle$、$H=C_2=\langle b\rangle$，$Aut(C_4)\cong(\mathbb{Z}/4\mathbb{Z})^\times=\{\bar1,\bar3\}\cong C_2$，故 $\phi:H\to Aut(C_4)$ 只有两种。

- **$\phi$ 平凡**：$C_4\times C_2$（阶 8 阿贝尔，有 4 阶元素但非循环）。
- **$\phi$ 非平凡**（$\phi(b)(a)=a^{-1}=a^3$）：得到 $D_4$，因为 $D_4=\langle r,m\rangle$ 中 $mam^{-1}=a^{-1}$。
  $$\text{验证：}(1,b)*(a,1)*(1,b)^{-1}=\bigl(\phi(b)(a),1\bigr)=(a^{-1},1),$$
  与 $D_4$ 中 $mrm^{-1}=r^{-1}$ 完全一致 ✓（$a$ 对应 $r$、$b$ 对应 $m$）。

**同一对 $(N,H)$ 拼出不同构的群**——这是[阶 8 分类](./09-isomorphism-theorems-II.md)里 $C_4\times C_2$ 与 $D_4$ 并列的原因。**故半直积不是"由 $N,H$ 决定"，而是"由 $(N,H,\phi)$ 三元组决定"。**

**例 4.31（本篇点睛）**：$n\ge3$ 时
$$S_n=A_n\rtimes\langle(1,2)\rangle\qquad(\text{不是直积}).$$

验证三条：
1. **$A_n\triangleleft S_n$**：$[S_n:A_n]=2$（[05 命题 3.25](./05-group-properties-II.md)）＋[定理 4.2](./06-normal-subgroup-and-quotient-group.md) ✓。
2. **$A_n\cap K=\{1\}$**（$K=\langle(1,2)\rangle=\{e,(1,2)\}$）：$(1,2)\notin A_n$（是奇排列）✓。
3. **$S_n=A_nK$**：任意 $\sigma\in S_n$，若 $\sigma$ 偶则 $\sigma=\sigma e\in A_nK$；若 $\sigma$ 奇则 $\sigma(1,2)^{-1}=\sigma(1,2)$ 是偶（sgn 乘 $-1$ 变 $+1$），故 $\sigma=\bigl(\sigma(1,2)\bigr)(1,2)\in A_nK$ ✓。也可用阶数：$|A_nK|=\frac{|A_n||K|}{|A_n\cap K|}=\frac{(n!/2)\cdot2}{1}=n!=|S_n|$ ✓。

而 **$K$ 不正规**（$n\ge3$）：$K$ 只有一个 2 阶元素 $(1,2)$，但 $S_n$ 中的 2 阶元素有多个（所有对换），若 $K\triangleleft S_n$ 则 $(1,2)$ 的**共轭类**要落在 $K$ 里——而 $(1,2)$ 的共轭类含全部对换，共 $\binom n2\ge3>2$ 个，装不进 $|K|=2$ ✓。

故由定理 4.30，$S_n\cong A_n\rtimes_\phi\langle(1,2)\rangle$，$\phi$ 由 $(1,2)$ 对 $A_n$ 的共轭给出。**不是直积**（$K$ 不正规 ⇒ 不能是直积，因为直积要求两块都正规）。

> **具体看 $n=3$**：$S_3=A_3\rtimes\langle(1,2)\rangle$，$\phi((1,2))$ 在 $A_3=\langle(1,2,3)\rangle$ 上的作用是共轭：$(1,2)(1,2,3)(1,2)=(1,3,2)=(1,2,3)^{-1}$——**正是 $C_3$ 的取逆自同构**，于是 $S_3\cong C_3\rtimes C_2=D_3$ ✓ 与 [05 阶 6 分类](./05-group-properties-II.md)接上。
>
> **这个例子的分量**：半直积不是冷门构造，而是"**正规部分 + 一点外在扭曲**"的**标准形状**。$S_n$、$D_n$、甚至[魔方群](../rubiks-cube-group-theory.md)（$\ker\pi$ 朝向部分 $\rtimes$ 位置置换部分）都是这个形状。

## 为什么重要

- "拼群"是分类定理的原材料：[有限阿贝尔群分类](./09-isomorphism-theorems-II.md)的结论就是"循环群质数幂的直积"；[阶 8 分类](./09-isomorphism-theorems-II.md)里 $D_4=C_4\rtimes C_2$ 型结构
- $Aut(C_n)\cong(\mathbb{Z}/n\mathbb{Z})^\times$ 把[模 n 乘法群](./02-mod-n-and-dihedral-groups.md)接回群论主线
- 内自同构 $\lambda_g$ 正是[魔方笔记](../rubiks-cube-group-theory.md)里的"共轭搬运"：$o(x)=o(\lambda_g(x))$（[01](./01-introduction-to-group-theory.md) 命题 1.13）在这里得到代数身份

## 局限 / 边界

- 半直积依赖 $\phi$ 的选取：同一对 $(N,H)$ 可拼出不同构的群（$C_4\rtimes C_2$ 有平凡与非平凡两种 $\phi$，得 $C_4\times C_2$ 与 $D_4$）
- 定理 4.30 要求 $N$ 正规而 $H$ 不必——方向别记反

## 我的理解

- 直积 vs 半直积的区别就一句话：$hk=h'k'$ 拆开时，$k$ 穿过 $h'$ 要不要"付费"。直积免费（$h'k=kh'$），半直积付自同构的税（$h'k=\phi(k)(h')\,k$）
- $S_n=A_n\rtimes C_2$ 说明半直积不是冷门构造，而是"正规部分 + 一点外在扭曲"的标准形状；魔方群同样是 $\ker\pi$（朝向）与位置置换部分的拼接

## Related

- [06 正规子群与商群](./06-normal-subgroup-and-quotient-group.md) — 正规性是半直积的前提
- [08 同构定理（上）](./08-isomorphism-theorems-I.md) — $HK/H\cong K/(H\cap K)$ 配合内直积使用
- [09 同构定理（下）与有限阿贝尔群分类](./09-isomorphism-theorems-II.md) — 直积是分类的表达语言
- [魔方群的数学结构](../rubiks-cube-group-theory.md) — 共轭与交换子
