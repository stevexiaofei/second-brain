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

**证明（三条都用子群判别法）：**

0. **先备两条基本事实**：$\varphi(e_G)=e_H$（因为 $\varphi(e)=\varphi(ee)=\varphi(e)\varphi(e)$，两边左乘 $\varphi(e)^{-1}$ 得 $e=\varphi(e)$）；以及 $\varphi(g^{-1})=\varphi(g)^{-1}$（因为 $\varphi(g)\varphi(g^{-1})=\varphi(gg^{-1})=\varphi(e)=e$，由逆元唯一性即得）。

1. **像**：$N$ 非空故 $\varphi(N)$ 非空。取 $\varphi(x),\varphi(y)\in\varphi(N)$（$x,y\in N$）：
   - 封闭：$\varphi(x)\varphi(y)=\varphi(xy)\in\varphi(N)$（$xy\in N$）；
   - 逆封闭：$\varphi(x)^{-1}=\varphi(x^{-1})\in\varphi(N)$（$x^{-1}\in N$）。
2. **核**：$\ker\varphi$ 非空（含 $e_G$）。取 $x,y\in\ker\varphi$：
   - 封闭：$\varphi(xy)=\varphi(x)\varphi(y)=e_He_H=e_H$，故 $xy\in\ker\varphi$；
   - 逆封闭：$\varphi(x^{-1})=\varphi(x)^{-1}=e_H^{-1}=e_H$，故 $x^{-1}\in\ker\varphi$。
3. **复合**：$\psi\circ\varphi$ 满足 $(\psi\circ\varphi)(xy)=\psi(\varphi(xy))=\psi(\varphi(x)\varphi(y))=\psi(\varphi(x))\psi(\varphi(y))$。$\blacksquare$

> **注意 $\varphi^{-1}$ 不一定是同态**：同态可逆（作为映射）时其逆同态需另证。事实上定理 3.7 与[命题 5.1](./08-isomorphism-theorems-I.md) 会给出一条干净判据：$\varphi$ 是双射 $\iff$ 满射且 $\ker\varphi=\{e\}$。

**例（几个标准同态与核）**：

| 同态 | 核 |
|---|---|
| $sgn:S_n\to\{\pm1\}$ | $A_n$（偶排列全体） |
| $\det:GL_n(\mathbb{R})\to\mathbb{R}^\times$ | $SL_n(\mathbb{R})$ |
| $\pi:\mathbb{Z}\to\mathbb{Z}/n\mathbb{Z},\ \pi(k)=\bar k$ | $n\mathbb{Z}$ |
| $\psi:\mathbb{R}\to S^1,\ \psi(t)=e^{2\pi it}$ | $\mathbb{Z}$ |
| $\varphi:(\mathbb{Z}/n\mathbb{Z})^\times\to(\mathbb{Z}/n\mathbb{Z})^\times,\ \varphi(x)=x^2$ | 平方为单位元的元素（阶整除 2 的那些） |

这张表把后面几篇的主角一次列齐：$A_n$（[05](./05-group-properties-II.md)）、$n\mathbb{Z}$（[06](./06-normal-subgroup-and-quotient-group.md)）、$\mathbb{Z}$ 作为 $\ker$（[06 例 4.13](./06-normal-subgroup-and-quotient-group.md)）。

### 群同构

**定义 3.6**：双射的同态叫**同构**，记 $G\cong H$。

**命题 3.7（同构不变量）**：同构保持：cardinality、元素阶（$o(g)=o(\varphi(g))$）、阿贝尔性、循环性。

**证明四条：**

设 $\varphi:G\to H$ 是同构。

1. **cardinality**：$\varphi$ 是双射，故 $|G|=|H|$。
2. **元素阶**：由同态性对 $m\ge1$ 归纳得 $\varphi(g^m)=\varphi(g)^m$。于是
   $$\varphi(g)^m=\varphi(g^m)=e_H\iff g^m\in\ker\varphi=\{e_G\}\iff g^m=e_G.$$
   两边 $m$ 的取值集合相同，故最小正整数相同，$o(\varphi(g))=o(g)$。（$\ker\varphi=\{e_G\}$ 因为 $\varphi$ 单射。）
3. **阿贝尔性**：若 $G$ 阿贝尔，则对任意 $x'=\varphi(x),y'=\varphi(y)\in H$，
   $$x'y'=\varphi(x)\varphi(y)=\varphi(xy)=\varphi(yx)=\varphi(y)\varphi(x)=y'x'.$$
   故 $H$ 阿贝尔。反向用 $\varphi^{-1}$（也是同构，见下）。
4. **循环性**：若 $G=\langle g\rangle$，则任意 $x\in G$ 有 $x=g^k$，故 $\varphi(x)=\varphi(g)^k$，说明 $H=\langle\varphi(g)\rangle$ 循环。反向同理。

> **$\varphi^{-1}$ 也是同构**（这一步在上面用了，值得单独交代）：$\varphi$ 是双射故 $\varphi^{-1}$ 是映射。验证同态：任取 $a,b\in H$，写 $a=\varphi(x),b=\varphi(y)$，则
> $$\varphi^{-1}(ab)=\varphi^{-1}\bigl(\varphi(x)\varphi(y)\bigr)=\varphi^{-1}(\varphi(xy))=xy=\varphi^{-1}(a)\varphi^{-1}(b).$$

> 这是"用不变量**排除**同构"的工具箱：[05](./05-group-properties-II.md) 里证 $A_4\not\cong D_6$（$D_6$ 有 6 阶元素 $r$，$A_4$ 的元素阶只有 $1,2,3$——$o(\varphi(r))$ 憋不出 6）正是用法示范。**注意只能否定**：找到不一致的不变量 ⇒ 不同构；找不到 ⇒ 什么也没说明。

**例（用不变量一击分辨）**：

- $C_6\not\cong S_3$：$|C_6|=|S_3|=6$、都非平凡，但 $C_6$ 阿贝尔、$S_3$ 不阿贝尔 ⇒ 不同构。
- $C_4\not\cong C_2\times C_2$：同阶同为阿贝尔，但 $C_4$ 有 4 阶元素、$C_2\times C_2$ 没有 ⇒ 不同构。
- $D_4\not\cong C_8$：同阶 8，但 $D_4$ 非阿贝尔 ⇒ 不同构。
- $D_4$ 与 $Q_8$ **不能**靠阶谱分开（都有元素阶 $1,2,4$，都是 8 阶非阿贝尔）——需要别的不变量，如**元素阶的个数分布**：$D_4$ 有 2 个 4 阶元素（$r,r^3$），$Q_8$ 有 6 个（$i,j,k,\pm i,\pm j,\pm k$ 中的 6 个）⇒ 不同构（[09](#)）。

### 循环群分类

**定理 3.10**：无限循环群 $\cong(\mathbb{Z},+)$；$n$ 阶有限循环群 $\cong(\mathbb{Z}/n\mathbb{Z},+)$。

**证明分两种情形，每种都验证"良定义 / 双射 / 同态"三件事。**

**情形 A：$G=\langle g\rangle$ 无限（$o(g)=\infty$）。** 定义
$$\varphi:\mathbb{Z}\to G,\qquad \varphi(k)=g^k.$$

- **良定义**：这是普通函数（$k\mapsto g^k$ 无代表元问题），天然良定义。**但要注意 $k\mapsto g^k$ 是不是单射**——见下。
- **同态**：$\varphi(k+l)=g^{k+l}=g^kg^l=\varphi(k)\varphi(l)$ ✓。
- **满射**：$G=\langle g\rangle=\{g^k\}$，显然 ✓。
- **单射**：若 $\varphi(k)=\varphi(l)$，即 $g^{k-l}=e$。若 $k\ne l$，则 $|k-l|\ge1$ 且 $g^{|k-l|}=e$（或取逆），故 $o(g)\le|k-l|<\infty$，与无限矛盾。所以 $k=l$ ✓。

故 $\varphi$ 是同构，$G\cong\mathbb{Z}$。

**情形 B：$G=\langle g\rangle$，$o(g)=n<\infty$。** 定义
$$\varphi:\mathbb{Z}/n\mathbb{Z}\to G,\qquad \varphi(\bar k)=g^k.$$

- **良定义（关键一步）**：$\bar k=\bar l$ 即 $n\mid k-l$，即 $k=l+nq$。则
  $$\varphi(\bar k)=g^k=g^{l+nq}=g^l(g^n)^q=g^l=\varphi(\bar l).$$
  结果与代表元无关 ✓。（如果没有 $g^n=e$，这一步就崩了——**良定义的代价正是"关系"**。）
- **同态**：$\varphi(\bar k+\bar l)=\varphi(\overline{k+l})=g^{k+l}=g^kg^l$ ✓。
- **满射**：$G=\langle g\rangle=\{g^k:k\in\mathbb{Z}\}$ ✓。
- **单射**：设 $\varphi(\bar k)=e$，即 $g^k=e$。由推论 2.32 的"幂按周期重复"，等价于 $n\mid k$，即 $\bar k=\bar0$，故 $\ker\varphi=\{\bar0\}$，$\varphi$ 单射 ✓。

故 $G\cong\mathbb{Z}/n\mathbb{Z}$。$\blacksquare$

> 同构意义下循环群**只有两种**：$C_\infty\cong\mathbb{Z}$ 与 $C_n:=\mathbb{Z}/n\mathbb{Z}$。以后统一写 $C_n$。
>
> **例（找具体生成元，把抽象的 $\mathbb{Z}/n\mathbb{Z}$ 认出来）**：
> - $A_3=\{e,(1,2,3),(1,3,2)\}$：其中 $(1,2,3)$ 的阶是 3，故 $A_3=\langle(1,2,3)\rangle\cong C_3$。
> - $(\mathbb{Z}/7\mathbb{Z})^\times$：$\bar3$ 的阶是 6，故 $(\mathbb{Z}/7\mathbb{Z})^\times=\langle\bar3\rangle\cong C_6$。
> - $D_2$（若按"翻转"定义而不退化）通常写成 $C_2\times C_2$——**它不是** $C_4$，说明"阶相同 + 都阿贝尔"仍不够，得有 4 阶元素才行。

**推论 3.11**：循环群的子群都是循环群。

**证明（两种情形）：**

**情形 A：$G=\mathbb{Z}$。** 这正是[例 1.18](./01-introduction-to-group-theory.md)：任何子群 $H=m\mathbb{Z}=\langle m\rangle$ 循环。

**情形 B：$G=C_n=\langle g\rangle$。** 设 $H\le G$。

- 若 $H=\{e\}$，则 $H=\langle e\rangle$ 循环。
- 否则取 $H$ 中**最小正幂次**：$s:=\min\{k\ge1:g^k\in H\}$（非空因 $H\ne\{e\}$ 且幂按周期重复使足标落在 $[1,n]$；由良序性取到最小）。
  断言 $H=\langle g^s\rangle$：
  - $(\supseteq)$ $g^s\in H$ 且 $H$ 封闭，故 $g^{s}$ 的幂全在 $H$。
  - $(\subseteq)$ 任取 $h=g^k\in H$，写 $k=qs+r$（$0\le r<s$）。则
    $$g^r=g^{k-qs}=g^k\cdot(g^s)^{-q}\in H,$$
    由 $s$ 的最小性，$r=0$，故 $k=qs$，$h=(g^s)^q\in\langle g^s\rangle$。

故 $H=\langle g^s\rangle$ 循环，且 $|H|=n/s$，即 $s=n/|H|$——这也就是"每个因子对应唯一子群"的构造（[03 定理 2.41 后的推论](./03-symmetric-and-cyclic-groups.md)）。$\blacksquare$

> **推论 3.11 的威力**：$\mathbb{Z}$ 与 $C_n$ 的子群格因此完全可算。例 $C_{12}$：$|H|$ 取遍 $12$ 的因子 $1,2,3,4,6,12$，每个恰有一个子群，分别由 $g^{12},g^6,g^4,g^3,g^2,g$ 生成。**"子群个数 = 因子个数"是循环群的特征**（在有限阿贝尔群中只有循环群如此）。

### 对称群 II：置换矩阵与 sgn

**定义 3.13（置换矩阵）**：$(Q_\sigma)_{ij}=1\iff\sigma(j)=i$（每行每列恰一个 1）。

**推论 3.16**：
1. $Q_\sigma Q_\sigma^\top=I_n$（逆就是转置）
2. $\det(Q_\sigma)=\pm1$
3. $Q_\sigma Q_\tau=Q_{\sigma\tau}$（复合变矩阵乘法——$S_n$ 嵌入 $GL_n$）

**证明：**

(1) 计算 $\bigl(Q_\sigma Q_\sigma^\top\bigr)_{ij}=\sum_k (Q_\sigma)_{ik}(Q_\sigma^\top)_{kj}=\sum_k (Q_\sigma)_{ik}(Q_\sigma)_{jk}$。按定义 $(Q_\sigma)_{ik}=1\iff\sigma(k)=i$，$(Q_\sigma)_{jk}=1\iff\sigma(k)=j$。因 $\sigma$ 是双射，"存在 $k$ 使 $\sigma(k)=i$ 且 $\sigma(k)=j$"当且仅当 $i=j$，且此时恰有一个这样的 $k$（$k=\sigma^{-1}(i)$）。故该和为 $1$（$i=j$）或 $0$（$i\ne j$），即 $=\delta_{ij}$，$Q_\sigma Q_\sigma^\top=I_n$。

(2) 由 (1) 取行列式：$\det(Q_\sigma)^2=\det(Q_\sigma Q_\sigma^\top)=\det(I_n)=1$，故 $\det(Q_\sigma)=\pm1$（实矩阵行列式是实数）。

(3) 两边第 $(i,k)$ 项：
$$\bigl(Q_\sigma Q_\tau\bigr)_{ik}=\sum_j (Q_\sigma)_{ij}(Q_\tau)_{jk}.$$
$(Q_\tau)_{jk}=1\iff\tau(k)=j$，故和中只有 $j=\tau(k)$ 一项非零，给出 $(Q_\sigma)_{i,\tau(k)}$，而它 $=1\iff\sigma(\tau(k))=i\iff(\sigma\tau)(k)=i$。这正是 $(Q_{\sigma\tau})_{ik}$。故 $Q_\sigma Q_\tau=Q_{\sigma\tau}$。$\blacksquare$

> (3) 是最有价值的一条：它说 $\sigma\mapsto Q_\sigma$ 是**群同态** $S_n\to GL_n(\mathbb{R})$，且由 (1) 每个 $Q_\sigma$ 可逆（$Q_\sigma^{-1}=Q_\sigma^\top$）、由 (3) 保运算、单射（$Q_\sigma$ 唯一决定 $\sigma$），故 $S_n$ 同构于 $GL_n(\mathbb{R})$ 的一个子群——**置换矩阵把抽象置换线性代数化**，这是群表示论的第一次预演。
>
> **例（$n=3$ 手算一次）**：
> $$Q_{(1,2)}=\begin{pmatrix}0&1&0\\1&0&0\\0&0&1\end{pmatrix},\qquad Q_{(1,2,3)}=\begin{pmatrix}0&0&1\\1&0&0\\0&1&0\end{pmatrix}.$$
> 直接乘：$Q_{(1,2)}Q_{(2,3)}$ 应等于 $Q_{(1,2)(2,3)}=Q_{(1,2,3)}$——可以验证 $Q_{(1,2)}Q_{(2,3)}=Q_{(1,2,3)}$，正如 $(1,2)(2,3)=(1,2,3)$。

**定义 3.17**：$\operatorname{sgn}(\sigma):=\det(Q_\sigma)$，称为排列的 signature（sign / parity）。

**推论 3.18**：$\operatorname{sgn}(\sigma\tau)=\operatorname{sgn}(\sigma)\operatorname{sgn}(\tau)$——即 $sgn:S_n\to\{\pm1\}\cong C_2$ 是群同态。

**证明**：由推论 3.16(3) 与行列式的乘法性
$$\operatorname{sgn}(\sigma\tau)=\det(Q_{\sigma\tau})=\det(Q_\sigma Q_\tau)=\det(Q_\sigma)\det(Q_\tau)=\operatorname{sgn}(\sigma)\operatorname{sgn}(\tau).$$
$\blacksquare$

> **注意 $\{\pm1\}$ 是群**（乘法，单位元 $1$）且 $\{\pm1\}\cong C_2$（把 $-1$ 送到 $\bar1$）。所以 $sgn$ 是一个**从 $S_n$ 到 $C_2$ 的同态**——它的核就是[交错子群 $A_n$](./05-group-properties-II.md)。
>
> **为什么用 $\det$ 定义（而不是"对换个数的奇偶"）**：用对换定义要额外证明"$\sigma$ 的对换分解长度奇偶性良定义"（换一种分解长度可能不同，但奇偶性相同）——这是历史上很难的一步。**走 $\det$ 这条路可以绕过它**：行列式是良定义的，"$\det(AB)=\det A\det B$"免费给出同态性，剩下的全是具体计算。代价是抽象的"奇偶性"直觉晚一点才出现。

**引理 3.19**：$k$-cycle 满足 $\operatorname{sgn}=-1$ 当且仅当 $k$ 为偶数。

**证明：**

**(1) 任何 $k$-cycle 可写成 $k-1$ 个对换。** 直接给分解（从右往左作用）：
$$(a_1,a_2,\dots,a_k)=(a_1,a_k)(a_1,a_{k-1})\cdots(a_1,a_3)(a_1,a_2).$$
逐点验证：右端先把 $a_1\mapsto a_2$，再不被其余对换影响（其余对换只动 $a_1$ 与 $a_j\ (j\ge3)$），故 $a_1\mapsto a_2$ ✓；$a_2$ 先不动，然后被 $(a_1,a_2)$ 送回 $a_1$……完整验证按"右起第 $j$ 个对换把 $a_{j+1}$ 送到 $a_1$"逐步跟踪即可得 $a_i\mapsto a_{i+1}$、$a_k\mapsto a_1$，其余不动 ✓。

**(2) 对换的 sgn 是 $-1$。** 对换 $\tau=(a,b)$ 的置换矩阵是交换 $a,b$ 两行（或两列）的单位阵。单位阵交换两行后行列式为 $-1$（行列式的反对称性），故 $\operatorname{sgn}(\tau)=-1$。

**(3) 由同态性把 (1)(2) 拼起来**：
$$\operatorname{sgn}\bigl((a_1,\dots,a_k)\bigr)=\prod_{j=1}^{k-1}\operatorname{sgn}\bigl((a_1,a_{j+1})\bigr)=(-1)^{k-1}.$$
故 $\operatorname{sgn}=-1\iff k-1$ 奇 $\iff k$ 偶。$\blacksquare$

> **由此得到 $A_n$ 的最初印象**：$n\ge2$ 时偶排列 = 偶长度 cycle 之积（等价地，偶数个对换之积）。

**定理 3.21**：把 $\sigma$ 写成 cycles $\sigma=\sigma_1\sigma_2\cdots\sigma_m$（**不必**不相交），则
$$\operatorname{sgn}(\sigma)=(-1)^d,$$
$d$ 为**偶长度** cycle 的个数。

**证明**：由推论 3.18 同态性和引理 3.19，
$$\operatorname{sgn}(\sigma)=\prod_{j=1}^{m}\operatorname{sgn}(\sigma_j)=\prod_{j=1}^{m}(-1)^{\ell_j-1}=(-1)^{\sum(\ell_j-1)}.$$
只需验证 $\sum_{j=1}^m(\ell_j-1)\equiv d\pmod 2$，其中 $d$ 是偶长度 cycle 的个数。注意 $\ell_j-1$ 是奇数 $\iff\ell_j$ 是偶数，故 $\sum(\ell_j-1)$ 的奇偶性 = 偶长度 cycle 的个数 $d$ 的奇偶性（奇数的个数与和同奇偶）。故 $(-1)^{\sum(\ell_j-1)}=(-1)^d$。$\blacksquare$

> **注意定理 3.21 与"分解成不相交 cycles"的关系**：定理 3.21 允许任意 cycles 分解（含交叠、含长度为 1 的），结论只用"偶长度 cycle 的个数"——**这本身就证明了奇偶性良定义**：不同分解若给不同答案就矛盾了。特别地，对不相交分解，"偶长度 cycle 的个数"和"$\sigma$ 的阶的一半"之类的想法都对不上，**只数偶长度 cycle 的个数**（这是原文给的公式）。
>
> **例（原文的例子）**：$\sigma=(4,2,5)(7,2,4,3)(3,1)$。三个 cycle 长度 $3,4,2$，其中偶数长度的有 $(7,2,4,3)$（长 4）与 $(3,1)$（长 2），故 $d=2$，$\operatorname{sgn}=(-1)^2=+1$ ✓。
>
> **例（"不必不相交"到底省了什么）**：取 $\rho=(1,2,3)(2,3,4)$，两个因子交叠。
>
> **路线一：直接用定理 3.21 数偶长度 cycle。** 两个因子长度都是 3（奇），故 $d=0$，$\operatorname{sgn}=+1$。
>
> **路线二：先化成不相交分解再数。** 从右往左逐点追踪（$(2,3,4)$ 先作用）：
> $$1\xrightarrow{(2,3,4)}1\xrightarrow{(1,2,3)}2,\qquad 2\xrightarrow{(2,3,4)}3\xrightarrow{(1,2,3)}1,$$
> $$3\xrightarrow{(2,3,4)}4\xrightarrow{(1,2,3)}4,\qquad 4\xrightarrow{(2,3,4)}2\xrightarrow{(1,2,3)}3.$$
> 故 $\rho=(1,2)(3,4)$，两个偶长度 cycle，$d=2$，$\operatorname{sgn}=(-1)^2=+1$。
>
> 两条路线一致 ✓。**定理 3.21 的价值就在这里**：交叠分解虽然不方便读结构（路线二要先算），但**奇偶性能直接数出来**（路线一）。而且两种分解给同一个答案，这本身就是在验证 $sgn$ 的良定义性。
>
> **这门"数偶长度 cycle"的手艺**：[魔方群](../rubiks-cube-group-theory.md) 里角块与棱块的奇偶性同步守恒，就是 $sgn$ 在两类块上的同时应用（转角数不变 $\leftrightarrow$ 奇偶性不变）。

## 为什么重要

- 同态/同构是整个专题后半部（[商群](./06-normal-subgroup-and-quotient-group.md)、[同构定理](./08-isomorphism-theorems-I.md)）的语言基础；$\ker\varphi$ 在这里作为子群登场，[08](./08-isomorphism-theorems-I.md) 里它升级为正规子群并给出 $G/\ker\varphi\cong\operatorname{im}\varphi$
- sgn 是第一个"从群到 $C_2$ 的非平凡同态"，其核就是[交错子群 $A_n$](./05-group-properties-II.md)；[魔方群](../rubiks-cube-group-theory.md) 的守恒律一（角/棱奇偶性同步）正是 sgn 思想的二维版本
- 置换矩阵把 $S_n$ 线性代数化——"群表示论"的第一次预演

## 局限 / 边界

- $\operatorname{sgn}$ 的定义依赖 $\det$，看起来"线性代数气"很重；纯群论的定义（对换个数）需要另外证明良定义，本篇用矩阵路线绕开了这个问题
- 同构不变量只能**否定**同构（找到不一致的不变量），**肯定**同构必须构造映射

## 我的理解

- "同构保阶"让阶成为指纹：想证明两个群不同构，数一数各阶元素个数往往一击致命（$A_4$ vs $D_6$、$C_6$ vs $S_3$）
- $\operatorname{sgn}$ 用行列式定义的妙处：$\det(AB)=\det(A)\det(B)$ 免费送出"sgn 是同态"；剩下的全是行列式计算

## Related

- [03 对称群与循环群](./03-symmetric-and-cyclic-groups.md) — cycle 分解是 sgn 计算的输入
- [05 群的性质（下）](./05-group-properties-II.md) — $A_n=\ker(sgn)$、拉格朗日定理
- [08 同构定理（上）](./08-isomorphism-theorems-I.md) — $\ker\varphi$ 的完整故事
- [魔方群的数学结构](../rubiks-cube-group-theory.md) — 奇偶性守恒 = sgn 思想的二维推广
