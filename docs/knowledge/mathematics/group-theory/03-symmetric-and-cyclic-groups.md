---
title: 对称群与循环群
type: concept
status: growing
tags: [Mathematics, Group Theory, Symmetric Group, Cyclic Group, Permutation]
created: 2026-10-14
updated: 2026-10-14
source: 知乎《群论及抽象代数学习笔记（3）- 一些群的介绍（下）》（大PPP，NTU MH2220）
---

# 对称群与循环群

> 系列第二章下。对称群 $S_n$（排列、cycle 记号、cycle 分解、阶）与循环群初步（$\langle g\rangle$、$o(g^k)=n/\gcd(n,k)$）。原文：<https://zhuanlan.zhihu.com/p/604557952>

## 一句话理解

$S_n$ 把"重排 $n$ 个对象"变成群，cycle 记号让排列可读可乘可求阶；循环群 $\langle g\rangle$ 是"一个元素的全部幂"，是最简单也最基本的一类群。

## 什么是它

### 对称群

**定义 2.18**：集合 $X$ 的**排列**（permutation）是双射 $X\to X$；$X=\{1,\dots,n\}$ 时全体排列记为 $S_n$，$|S_n|=n!$。

**证明 $|S_n|=n!$**：排列 $\sigma$ 由 $\sigma(1),\dots,\sigma(n)$ 这 $n$ 个值决定，而它们必须是 $1,\dots,n$ 的一个排列（每值恰用一次）。构造：$\sigma(1)$ 有 $n$ 种选法，$\sigma(2)$ 剩 $n-1$ 种……由乘法原理总共 $n(n-1)\cdots1=n!$ 个。（严格说是"$\sigma$ 与有序元组 $(\sigma(1),\dots,\sigma(n))$ 一一对应，而后者就是 $1..n$ 的所有排列"。）

**命题 2.21**：$(S_n,\circ)$ 是群（复合为运算，恒等映射为单位元，逆映射为逆元）。

**证明：**

1. **封闭**：两个双射的复合仍是双射（逆映射为 $\sigma^{-1}\circ\tau^{-1}$），故 $\circ:S_n\times S_n\to S_n$ 是二元运算。
2. **结合律**：映射复合天然满足 $(\sigma\circ\tau)\circ\rho=\sigma\circ(\tau\circ\rho)$（两边对任意 $x$ 给出同一个值）。
3. **单位元**：恒等映射 $\mathrm{id}(x)=x$ 满足 $\sigma\circ\mathrm{id}=\sigma=\mathrm{id}\circ\sigma$。
4. **逆元**：$\sigma$ 是双射，故有逆映射 $\sigma^{-1}$ 且也是双射，$\sigma\circ\sigma^{-1}=\mathrm{id}=\sigma^{-1}\circ\sigma$。

故 $(S_n,\circ)$ 是群。$\blacksquare$

> **$S_n$ 是"所有有限群的容器"（预告）**：[05 Cayley 定理](./05-group-properties-II.md) 说任何 $n$ 阶群都同构于 $S_n$ 的某个子群——所以 $S_n$ 的构造虽简单，却是全体有限群的家。这也解释了为什么 $S_n$ 值得单独细讲。

> **记号约定**：本笔记与原文一致，**从右往左**作用：$\sigma\tau$ 表示先做 $\tau$ 再做 $\sigma$。这是函数复合的标准写法 $\sigma\circ\tau$，但和"从左往右读"的直觉相反，遇到别人的材料时要先确认约定。

**例（$S_3$ 的 6 个元素全表）**：

| 元素 | cycle 记号 | 阶 | sgn |
|---|---|---|---|
| $e$ | $(1)(2)(3)$ | 1 | $+1$ |
| $(1,2)$ | 对换 | 2 | $-1$ |
| $(1,3)$ | 对换 | 2 | $-1$ |
| $(2,3)$ | 对换 | 2 | $-1$ |
| $(1,2,3)$ | 3-cycle | 3 | $+1$ |
| $(1,3,2)$ | 3-cycle | 3 | $+1$ |

$|S_3|=3!=6$，与 $D_3$ 同阶；事实上 $S_3\cong D_3$（[02 例](./02-mod-n-and-dihedral-groups.md)）。$S_3$ 是**最小的非阿贝尔群**：$|S_3|=6$，而阶 $<6$ 的群全阿贝尔（阶 1,2,3,5 是质数阶必循环，阶 4 只能是 $C_4$ 或 $C_2\times C_2$）。

> **$n\ge3$ 时 $S_n$ 非阿贝尔**：取 $\sigma=(1,2),\ \tau=(2,3)$，则 $\sigma\tau=(1,2,3)$ 而 $\tau\sigma=(1,3,2)$，两者不同。$n=1,2$ 时 $|S_n|\le2$，必阿贝尔。

### cycle 记号

**定义 2.22**：cycle $(a_1,\dots,a_k)$ 是排列 $\sigma(a_1)=a_2,\ \sigma(a_2)=a_3,\ \dots,\ \sigma(a_k)=a_1$，其余元素不动；$k$ 称为长度。同一 cycle 可以轮转书写：$(1,2,3)=(2,3,1)=(3,1,2)$。

cycle 相乘即复合，**从右往左**作用。

**命题 2.26**：任何排列都能写成**不相交 cycles 之积**；不相交 cycles 互相交换。

**证明分三步。**

**(1) 定义轨道与等价关系。** 在 $X=\{1,\dots,n\}$ 上定义
$$a\sim b\iff \exists k\ge0\ \text{使}\ \sigma^k(a)=b.$$
先验证这是等价关系：

- **自反**：$\sigma^0(a)=a$。
- **对称**：$\sigma^k(a)=b$ 时，$\sigma$ 是双射故 $\sigma^{-k}(b)=a$；而 $-\!k$ 若为负可用 $\sigma$ 的逆幂（$\sigma^{-1}$ 也是排列），故 $b\sim a$。
- **传递**：$\sigma^k(a)=b$、$\sigma^l(b)=c$ 时 $\sigma^{k+l}(a)=c$，故 $a\sim c$。

于是 $X$ 被切成若干**等价类（轨道）** $B_1,\dots,B_r$，两两不交且并集为 $X$。

**(2) 每条轨道给出一个 cycle。** 取轨道 $B$，$|B|=k$。任取 $a\in B$，由轨道的定义 $B=\{\sigma^j(a):j\in\mathbb{Z}\}$。断言 $a,\sigma(a),\dots,\sigma^{k-1}(a)$ 两两不同：若 $\sigma^i(a)=\sigma^j(a)$（$0\le i<j\le k-1$），则 $\sigma^{j-i}(a)=a$，于是 $a$ 在 $\sigma$ 的迭代下周期为 $j-i<k$，轨道大小 $\le j-i<k$，矛盾。故这 $k$ 个元素互异且穷尽 $B$，于是 $\sigma$ 在 $B$ 上的作用恰是 cycle
$$\sigma_B=(a,\ \sigma(a),\ \dots,\ \sigma^{k-1}(a)),$$
把 $B$ 内元素循环移位、$B$ 外元素不动。

**(3) 拼起来得到分解，并证交换性。** 由构造，$\sigma$ 与 $\sigma_{B_1}\cdots\sigma_{B_r}$ 在每个元素上作用相同（该元素只属于一个轨道，其余因子把它固定），故
$$\sigma=\sigma_{B_1}\sigma_{B_2}\cdots\sigma_{B_r},$$
这是**不相交** cycles 之积（各支撑 $B_i$ 两两不交）。最后，两个支撑不交的 cycle 必交换：设 $\alpha$ 只动 $A$、$\beta$ 只动 $B$、$A\cap B=\emptyset$。任取 $x$——

- $x\in A$：$\alpha\beta(x)=\alpha(x)$（$\beta$ 不动 $x$）且 $\beta\alpha(x)=\beta(\alpha(x))=\alpha(x)$（$\alpha(x)\in A$，$\beta$ 不动）；
- $x\in B$：对称同上；
- $x\notin A\cup B$：两边都是 $x$。

故 $\alpha\beta=\beta\alpha$。$\blacksquare$

> **"不相交才交换"反过来差不多也成立**：两个 cycle 若支撑相交且都不为恒等，一般**不**交换。例：$(1,2)(1,3)=(1,3,2)$ 而 $(1,3)(1,2)=(1,2,3)$，两者不同。所以下面"局限"里说的"$S_n$ 对复合不交换"具体就落在交叠 cycle 上。

**例 2.25（分解算法，逐步走一遍）**：设 $\sigma$ 由下表给出（原文图）

$$\sigma(1)=5,\ \sigma(2)=1,\ \sigma(3)=2,\ \sigma(4)=4,\ \sigma(5)=3,\ \sigma(6)=6,\ \sigma(7)=8,\ \sigma(8)=7.$$

- 从最小的未处理元素 $1$ 出发跟踪轨道：$1\to5\to3\to2\to1$，回到起点，得 cycle $(1,5,3,2)$。
- 下一个未处理元素是 $4$：$\sigma(4)=4$，长度 1，即不动点 $(4)$。
- 接着 $6$：$\sigma(6)=6$，得 $(6)$。
- 接着 $7$：$7\to8\to7$，得 $(7,8)$。

略去不动点，于是
$$\sigma=(1,5,3,2)(7,8).$$
> **算法要点**：每次从**最小的未处理元素**开始，保证结果是"标准形"，便于比较两个排列是否相等。这才是原文例 2.25 的完整步骤。

**例（cycle type 与共轭类）**：$S_5$ 中
$$\sigma=(1,2)(3,4),\qquad \tau=(1,3)(2,5)$$
都是"两个 2-cycle"型，type 记为 $(2,2,1)$（补上不动点 5 的长度 1）。取 $\rho=(1,3,4)$ 计算共轭：
$$\rho\sigma\rho^{-1}=(1,2)(3,4)\ \text{在}\ \rho\ \text{下的重命名}=\ (\rho(1),\rho(2))(\rho(3),\rho(4))=(3,2)(4,1),$$
仍是两个 2-cycle 型。反过来，任何同 type 的 $\tau$ 都能由某个 $\rho$ 共轭得到——这就是"共轭类 = cycle type"（定理 2.28 后的注）。**例**：$S_4$ 中的共轭类与 type 一一对应：

| type | 代表元 | 类大小 |
|---|---|---|
| $(1,1,1,1)$ | $e$ | 1 |
| $(2,1,1)$ | $(1,2)$ | 6 |
| $(2,2)$ | $(1,2)(3,4)$ | 3 |
| $(3,1)$ | $(1,2,3)$ | 8 |
| $(4)$ | $(1,2,3,4)$ | 6 |

总和 $1+6+3+8+6=24=4!$ ✓（类大小公式 $\frac{n!}{\prod_j j^{m_j}m_j!}$，$m_j$ 为长度 $j$ 的 cycle 个数：如 $(2,2)$ 型给 $\frac{24}{2^2\cdot2!}=3$）。这张表在 [09 例 5.14](./09-isomorphism-theorems-II.md) 里还会用到。

**定理 2.28（排列的阶）**：
1. $k$-cycle 的阶是 $k$；
2. 不相交 cycles 积 $\sigma=\sigma_1\cdots\sigma_m$（各 $\sigma_i$ 长 $\ell_i$）满足
$$o(\sigma)=\operatorname{lcm}(\ell_1,\dots,\ell_m).$$

**证明：**

**(1) $k$-cycle 的阶是 $k$。** 设 $\alpha=(a_1,\dots,a_k)$。对 $1\le j\le k$，$\alpha^j$ 把 $a_i$ 送到 $a_{i+j}$（下标模 $k$）——对 $j$ 归纳即得。于是 $\alpha^k$ 把每个 $a_i$ 送到 $a_i$，即 $\alpha^k=\mathrm{id}$；而 $1\le j<k$ 时 $\alpha^j(a_1)=a_{j+1}\ne a_1$，故 $\alpha^j\ne\mathrm{id}$。所以 $o(\alpha)=k$。

**(2) 不相交 cycles 积的阶是 lcm。** 设 $\sigma=\sigma_1\cdots\sigma_m$，支撑 $B_i$ 两两不交，$\sigma_i$ 在 $B_i$ 上是长度 $\ell_i$ 的 cycle。关键事实（由不相交性 + 命题 2.26 的交换性）：
$$\sigma^k=\sigma_1^k\sigma_2^k\cdots\sigma_m^k.$$
于是对任意 $k\ge1$
$$\sigma^k=\mathrm{id}\iff \sigma_i^k=\mathrm{id}\ \text{对每个}\ i\iff \ell_i\mid k\ \text{对每个}\ i\iff \operatorname{lcm}(\ell_1,\dots,\ell_m)\mid k.$$
（第二个 $\iff$：$\sigma_i^k$ 只在 $B_i$ 上动，而各 $\sigma_i^k$ 相乘后若恒等，则限制在每个 $B_i$ 上分别恒等；第三个 $\iff$ 用 (1)。）故 $o(\sigma)$ 是能被所有 $\ell_i$ 整除的最小正整数，即 $\operatorname{lcm}(\ell_1,\dots,\ell_m)$。$\blacksquare$

> **例（阶的计算，含魔方 1260 的来源）**：
> - $\sigma=(1,2,3)(4,5)$：$o=\operatorname{lcm}(3,2)=6$。
> - $\sigma=(1,2,3,4)(5,6,7,8,9)$：$o=\operatorname{lcm}(4,5)=20$。
> - **最大值问题**：$S_9$ 中元素阶最大是多少？要把 9 拆成和不超过 9 的两两互质部分使 lcm 最大：$9=4+3+2$，$\operatorname{lcm}(4,3,2)=12$；而 $9=5+4$ 给 $\operatorname{lcm}=20$，更大；$9=5+3+1$ 给 15。最优是 $9=5+4\Rightarrow o_{\max}=20$。（魔方群里著名的 $1260=\operatorname{lcm}(9,7,5,4,3)$ 是同一套"拆和求最大 lcm"的算法在更多数字上的结果。）
> - **推论**：$S_n$ 的**指数**（所有元素阶的 lcm）即 Landau 函数 $g(n)$，增长极快：$g(10)=30,\ g(20)=420,\ g(50)=180180$。

> cycle type（把长度排成不增序列）是排列的共轭不变量——两个排列共轭当且仅当 type 相同。这是 [魔方群](../rubiks-cube-group-theory.md) 里"块的位置怎么排"的全部语言。

### 循环群初步

**定义 2.30**：$g\in G$ 生成的子集 $\langle g\rangle=\{g^k:k\in\mathbb{Z}\}$。

**推论 2.32**：$\langle g\rangle$ 是子群；且若 $o(g)=m$，则
$$\langle g\rangle=\{1,g,\dots,g^{m-1}\},\qquad |\langle g\rangle|=o(g).$$

**证明：**

**(1) $\langle g\rangle$ 是子群**（用[判别法](./01-introduction-to-group-theory.md)）：含 $g^0=1$，非空；$g^k\cdot g^l=g^{k+l}$ 封闭；$(g^k)^{-1}=g^{-k}$ 逆元封闭。

**(2) 幂按周期重复，且 $\{1,g,\dots,g^{m-1}\}$ 恰有 $m$ 个元素。**

任取 $k\in\mathbb{Z}$，带余除法 $k=lm+r$（$0\le r<m$，$l\in\mathbb{Z}$ 可为负），则
$$g^k=g^{lm+r}=(g^m)^l g^r=e^l g^r=g^r,$$
故 $\langle g\rangle\subseteq\{1,g,\dots,g^{m-1}\}$，反向包含显然，故相等。

再看这 $m$ 个幂两两不同：若 $g^i=g^j$（$0\le i<j<m$），则 $g^{j-i}=e$ 且 $0<j-i<m$，与 $m=o(g)$ 的最小性矛盾。故 $|\langle g\rangle|=m=o(g)$。$\blacksquare$

> 这个"$|\langle g\rangle|=o(g)$"是[拉格朗日定理](./05-group-properties-II.md)的最常用入口：$o(g)=|\langle g\rangle|$ 整除 $|G|$，于是 $o(g)\mid|G|$。
>
> **例（逐个数幂）**：$G=\mathbb{Z}/8\mathbb{Z}$ 加法，取 $g=\bar3$（阿贝尔群用加法记号，$g^k$ 读作 $k\bar3$）：$o(\bar3)=8$，$\langle\bar3\rangle=\{\bar0,\bar3,\bar6,\bar1,\bar4,\bar7,\bar2,\bar5\}$ 遍历全部 8 个元素，故 $G=\langle\bar3\rangle$，即 $\bar3$ 也是生成元（与 $\bar1$ 一样）。而 $\langle\bar2\rangle=\{\bar0,\bar2,\bar4,\bar6\}$，大小 4 $=o(\bar2)$ ✓。

**定义 2.34**：$G=\langle g\rangle$ 时称 $G$ 为**循环群**。例：$\mathbb{Z}=\langle 1\rangle=\langle-1\rangle$（生成元不唯一）；$\mathbb{Z}/n\mathbb{Z}=\langle\bar1\rangle$。

**命题 2.38**：循环群必为阿贝尔群。

证明：设 $G=\langle g\rangle$，则任意 $x,y\in G$ 可写作 $x=g^k,\ y=g^l$，于是
$$xy=g^kg^l=g^{k+l}=g^{l+k}=g^lg^k=yx.$$
$\blacksquare$

> **逆命题不成立**：阿贝尔不必循环。最小反例是 $C_2\times C_2=\{e,a,b,ab\}$（Klein 四元群，$a^2=b^2=e$）：它阿贝尔，但每个非单位元素阶为 2，无 4 阶元素，故不循环（[05 推论 3.39](./05-group-properties-II.md)：阶 4 的循环群应有 4 阶元素）。这正说明"阿贝尔"比"循环"严格地弱。

**定理 2.39**：有限群 $G$（$|G|=n$）是循环群 $\iff$ 存在 $g$ 使 $o(g)=n$。

**证明：**

- **($\Rightarrow$)** 设 $G=\langle g\rangle$。由推论 2.32，$o(g)=|\langle g\rangle|=|G|=n$。
- **($\Leftarrow$)** 设 $o(g)=n$。由推论 2.32，$\langle g\rangle$ 是 $G$ 的子群且 $|\langle g\rangle|=o(g)=n=|G|$。子集与全集元素个数相同即相等，故 $G=\langle g\rangle$，即 $G$ 循环。$\blacksquare$

> **这个等价让小阶分类变得机械**：判断一个 $n$ 阶群是否循环，只需"数一数有没有 $n$ 阶元素"。**例**：$\mathbb{Z}/6\mathbb{Z}$ 中 $\bar1$ 的阶是 6，故 $C_6$；而 $S_3$ 中最大阶是 3（见 03 开头的表），没有 6 阶元素，故 $S_3$ 不循环。**这正是 [05 阶 6 分类](./05-group-properties-II.md) 的判据**：$|G|=6$ 时若存在 6 阶元素则 $G\cong C_6$，否则只能是 $S_3$（或 $D_3$，与之同构）。

**定理 2.41（幂的阶公式）**：$G=\langle g\rangle$，$|G|=n$，则
$$o(g^k)=\frac{n}{\gcd(n,k)}.$$

**证明**：记 $d=\gcd(n,k)$，要证 $o(g^k)=n/d$。分两个不等式。

**上界（$(g^k)^{n/d}=e$）**：$n/d$ 是整数（$d\mid n$）。由推论 2.32，$g^n=e$，故
$$(g^k)^{n/d}=g^{k\cdot n/d}=\bigl(g^n\bigr)^{k/d}=e^{k/d}=e.$$
（$k/d$ 是整数因为 $d\mid k$。）于是由阶的最小性，$o(g^k)\le n/d$。

**下界（$n/d\mid o(g^k)$，即 $o(g^k)\ge n/d$）**：设 $(g^k)^t=e$，即 $g^{kt}=e$。由推论 2.32 的"幂按周期重复"，$g^{kt}=e\iff n\mid kt$。于是问题变成数论命题：

> **引理**：$n\mid kt\Rightarrow \dfrac{n}{d}\mid t$，其中 $d=\gcd(n,k)$。

引理证明（用 Bézout）：$d=\gcd(n,k)$ 时 $d=sn+tk_0$ 对某些整数 $s,k_0$ 成立（Bézout 恒等式）；且 $n=d n',\ k=d k'$ 且 $\gcd(n',k')=1$。由 $n\mid kt$ 得 $d n'\mid d k' t$，即 $n'\mid k't$；而 $\gcd(n',k')=1$，故 $n'\mid t$（Euclid 引理：互质因子可从乘积中约去），即 $n/d\mid t$。

于是任何使 $(g^k)^t=e$ 的 $t$ 都被 $n/d$ 整除，特别地取 $t=o(g^k)$ 得 $n/d\mid o(g^k)$，故 $o(g^k)\ge n/d$。

两侧合并：$o(g^k)=n/d=\dfrac{n}{\gcd(n,k)}$。$\blacksquare$

> **例（$n=6$ 逐个算，含原文的表）**：$G=C_6=\langle g\rangle$，
> $$o(g^0)=1,\quad o(g^1)=\tfrac62=6,\quad o(g^2)=\tfrac{6}{2}=3,\quad o(g^3)=\tfrac63=2,\quad o(g^4)=\tfrac62=3,\quad o(g^5)=6.$$
> 与 03 开头那张 $\mathbb{Z}/6\mathbb{Z}$ 的表完全一致。**有几个生成元**？$o(g^k)=6\iff\gcd(6,k)=1\iff k\in\{1,5\}$，共 $\varphi(6)=2$ 个——这就是"$\varphi(n)$ = $C_n$ 的生成元个数"。
>
> **例（$n=12$）**：$C_{12}$ 中 $o(g^k)=12/\gcd(12,k)$：$k=1..11$ 时依次为 $12,6,4,3,12,2,12,3,4,6,12$。生成元 $\varphi(12)=4$ 个（$k=1,5,7,11$），与 $(\mathbb{Z}/12\mathbb{Z})^\times=\{\bar1,\bar5,\bar7,\bar{11}\}$ 的元素完全对应——这不是巧合，而是 [07 定理 4.20](./07-automorphism-and-products.md) $Aut(C_n)\cong(\mathbb{Z}/n\mathbb{Z})^\times$ 的预告。
>
> **推论（$C_n$ 的子群结构）**：对 $n$ 的每个正因子 $d$，$C_n$ 恰有一个 $d$ 阶子群，即 $\langle g^{n/d}\rangle$。例 $n=12$：因子 $1,2,3,4,6,12$ 各对应唯一子群，$\langle g^6\rangle\cong C_2$、$\langle g^4\rangle\cong C_3$、$\langle g^3\rangle\cong C_4$、$\langle g^2\rangle\cong C_6$、$\langle g\rangle=C_{12}$。

## 为什么重要

- $S_n$ 是"所有有限群的容器"——Cayley 定理（[05](./05-group-properties-II.md)）说任何有限群都嵌入某个 $S_n$
- $o(\sigma)=\operatorname{lcm}$ 与 $o(g^k)=n/\gcd(n,k)$ 是两个最常用的**计算工具**，出现在魔方元素阶（1260）的计算里
- 循环群是后面一切分类的"基元"：[循环群分类](./04-group-properties-I.md)（只有 $\mathbb{Z}$ 与 $\mathbb{Z}/n\mathbb{Z}$）→ [有限阿贝尔群分类](./09-isomorphism-theorems-II.md)（循环群直积）

## 局限 / 边界

- cycle 分解不唯一（轮转书写、不相交因子顺序都可换），唯一的是 type
- $S_n$ 对复合虽成群，但 cycle 相乘**不交换**——即使不相交才交换；交叠 cycles 乘积要逐点追踪（例 2.27：$\sigma\circ\tau=(1,6,3,5,2)(4)$）
- 定理 2.41 的前提 $G=\langle g\rangle$ 不能去掉：一般有限群中 $o(x^k)$ 与 $|G|/\gcd(|G|,k)$ 无关。例 $S_3$ 中 $|G|=6$、$x=(1,2)$ 时 $x^k=x$ 或 $e$，$o$ 只能是 1 或 2，谈不上 $6/\gcd(6,k)$

## 我的理解

- "排列 = 双射"与"cycle = 轨道"是同一对象的两种视角：前者是函数视角（好复合），后者是动力系统视角（好看结构）；求阶公式是两种视角的合成
- $|\langle g\rangle|=o(g)$ 把"元素的阶"和"子群的大小"焊在一起——拉格朗日定理（$o(g)\mid|G|$）只是这一步加上"子群阶整除"的推论
- 定理 2.41 证明的手法值得单独记：**上界用"幂的周期重复"，下界用 Bézout 把整除条件转成互质**。这一"转成数论"的招式在 [07 定理 4.20](./07-automorphism-and-products.md)（$Aut(C_n)\cong(\mathbb{Z}/n\mathbb{Z})^\times$）、[09 引理 5.15](./09-isomorphism-theorems-II.md)（$p$-部分与 $p'$-部分分离）里都会重现

## Related

- [01 群论导论](./01-introduction-to-group-theory.md) — $\langle S\rangle$ 的定义
- [02 整数模 n 加法群与二面体群](./02-mod-n-and-dihedral-groups.md) — $\mathbb{Z}/n\mathbb{Z}=\langle\bar1\rangle$
- [04 群的性质（上）](./04-group-properties-I.md) — 循环群分类与 sgn
- [05 群的性质（下）](./05-group-properties-II.md) — Cayley 定理、拉格朗日定理
- [魔方群的数学结构](../rubiks-cube-group-theory.md) — cycle type 与置换奇偶性的应用场
