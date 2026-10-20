---
title: 群的性质（下）：交错子群、拉格朗日定理、群的分类、Cayley 定理
type: concept
status: growing
tags: [Mathematics, Group Theory, Alternating Group, Lagrange Theorem, Cayley Theorem]
created: 2026-10-14
updated: 2026-10-14
source: 知乎《群论及抽象代数学习笔记（5）- 群的性质（下）》（大PPP，NTU MH2220）
---

# 群的性质（下）：交错子群、拉格朗日定理、群的分类、Cayley 定理

> 系列第三章下。第一座里程碑：拉格朗日定理（子群阶整除群阶）；配上 $A_n$、阶 6 分类与 Cayley 定理。原文：<https://zhuanlan.zhihu.com/p/614344709>

## 一句话理解

陪集把 $G$ 均匀切成 $|H|$ 大小的块，于是 $|H|$ 整除 $|G|$——这是群论第一个"计数型"定理，整个有限群理论都建立在它上面。

## 什么是它

### 交错子群

**定义 3.23**：$A_n=\{\sigma\in S_n:\operatorname{sgn}(\sigma)=1\}$——全部**偶排列**。它是同态 $sgn:S_n\to\{\pm1\}$ 的核，因此是子群。

**例**：$A_1=A_2=\{1\}$；$A_3=\{1,(1,2,3),(1,3,2)\}=\langle(1,2,3)\rangle\cong C_3$。

**命题 3.25**：$n\ge2$ 时 $|A_n|=n!/2$（证明需拉格朗日，见下）。

**命题 3.26**：$A_4$ 非阿贝尔，且与 $D_6$ 不同构。$A_4$ 是第一个不属于已知家族的有限群。

**证明：**

**(1) $A_4$ 的元素阶只有 $1,2,3$。** $A_4$ 含 $4!/2=12$ 个元素，是 $S_4$ 中的偶排列：$e$（1 个，阶 1）、3 个双对换 $(a,b)(c,d)$（阶 2）、8 个 3-cycle（阶 3）。故阶的集合是 $\{1,2,3\}$。

**(2) $D_6$ 有 6 阶元素。** $D_6=\langle r,m\rangle$ 中 $r$ 是转 $60°$，$r^k$ 的阶是 $6/\gcd(6,k)$，特别地 $o(r)=6$。

**(3) 由同构不变量（[04 命题 3.7](./04-group-properties-I.md)）即可判定 $A_4\not\cong D_6$**：若 $A_4\cong D_6$，则 $D_6$ 中 6 阶元素 $r$ 的像在 $A_4$ 中也应阶为 6，但 $A_4$ 无 6 阶元素，矛盾。

**(4) $A_4$ 非阿贝尔。** 取 $\sigma=(1,2,3),\ \tau=(1,2,4)$（都是 3-cycle，故都在 $A_4$），逐点算：
$$\sigma\tau=(1,2,3)(1,2,4)=(1,3,4),\qquad \tau\sigma=(1,2,4)(1,2,3)=(2,3,4),$$
两者不同，故 $A_4$ 非阿贝尔。（顺便：$|A_4|=|D_6|=12$，所以这两个群是"同阶、一个非阿贝尔一个非阿贝尔"的例子——**分辨它们必须用元素阶谱**，这正是"不变量工具箱"的经典用例。）$\blacksquare$

> **$A_4$ 的完整元素表**（12 个，按 cycle type 分类，便于记忆）：
> - 单位元：$e$
> - 3 个双对换：$(1,2)(3,4),\ (1,3)(2,4),\ (1,4)(2,3)$
> - 8 个 3-cycle：$(1,2,3)$ 型，共 $4\times2=8$ 个（选"被固定的元素" 4 种 × 2 个方向）
>
> **顺带得到 $A_4$ 没有 6 阶子群**——这就是下面"三个不能"里提到的标准反例（$6\mid12$ 但无 6 阶子群）。可见拉格朗日的逆命题失败得相当彻底。

### 陪集

**定义 3.27**：$H\le G$，左陪集 $gH=\{gh:h\in H\}$，右陪集 $Hg=\{hg:h\in H\}$。

**引理 3.30（左陪集四性质）**：
1. $xH=H\iff x\in H$
2. $xH=yH\iff y^{-1}x\in H$
3. $xH\cap yH=\emptyset$ 或 $xH=yH$（陪集要么相等要么不交）
4. $h\mapsto xh$ 是 $H\to xH$ 的双射，故 $|H|=|xH|$

**证明：**

**(1)** ($\Leftarrow$) $x\in H$ 时对任意 $h\in H$，$xh\in H$（封闭），故 $xH\subseteq H$；反向：$h=x(x^{-1}h)\in xH$，故 $H\subseteq xH$。( $\Rightarrow$ ) $xH=H$ 时 $x=xe\in xH=H$，故 $x\in H$。

**(2)** ($\Rightarrow$) $xH=yH$ 时 $x\in xH=yH$（用 (1) 的"$y\in yH$"），故 $x=yh$ 某个 $h\in H$，于是 $y^{-1}x=h\in H$。( $\Leftarrow$ ) $y^{-1}x=h_0\in H$，即 $x=yh_0$。则
$$xH=yh_0H=yH\quad(\text{用 (1)：}h_0H=H).$$

**(3)** 设 $xH\cap yH\ne\emptyset$，取 $z\in xH\cap yH$。写 $z=xh_1=yh_2$（$h_1,h_2\in H$），则 $y^{-1}x=h_2h_1^{-1}\in H$，由 (2) 得 $xH=yH$。

**(4)** 映射 $\phi:H\to xH,\ \phi(h)=xh$ 显然满（$xH$ 的定义）。单射：$xh_1=xh_2$ 由左消去律（[01 补充性质 3](./01-introduction-to-group-theory.md)）得 $h_1=h_2$。故是双射，$|xH|=|H|$。$\blacksquare$

> **注意 (3) 的"或"是排他的**——陪集要么完全相同要么完全不相交，**没有中间态**（这正是[推论 2.2 传递性](./02-mod-n-and-dihedral-groups.md)在群里的化身：$g\sim g'$ 当且仅当 $g^{-1}g'\in H$ 是等价关系）。
>
> **完全用不着 $(1)+(2)$ 之外的假设？不，用到了 $H\le G$**：下面反例。

**反例意识（"子群"前提不可去掉）**：$H=\{\bar0,\bar1\}\subset\mathbb{Z}/3\mathbb{Z}$ 不是子群（$\bar1+\bar1=\bar2\notin H$）。取 $G=\mathbb{Z}/3\mathbb{Z}$：
$$\bar0+H=\{\bar0,\bar1\},\qquad \bar2+H=\{\bar2,\bar0\}.$$
两者**相交**（都含 $\bar0$）但**不相等**——(3) 失效。原因就在 (2)：$\bar2+\bar0=\bar2\notin H$ 与（本该成立的）$y^{-1}x\in H$ 不匹配，因为 $H$ 不封闭，$h_0H=H$ 那一步崩了。

**例（$S_3$ 关于 $H=\langle(1,2)\rangle$ 的左陪集）**：$H=\{e,(1,2)\}$，
$$eH=H=\{e,(1,2)\},\quad (1,3)H=\{(1,3),(1,3)(1,2)\}=\{(1,3),(1,3,2)\},\quad (2,3)H=\{(2,3),(1,2,3)\}.$$
共 3 个左陪集，每个大小 2，$|S_3|=3\times2=6$ ✓（拉格朗日的力行）。右陪集则不同：$H(1,3)=\{(1,3),(1,2)(1,3)\}=\{(1,3),(1,2,3)\}\ne(1,3)H$——**左、右陪集分开算是 $H$ 不正规的标志**（[06](./06-normal-subgroup-and-quotient-group.md)）。

### 拉格朗日定理

**定义 3.33**：指数 $[G:H]$ = 不同左陪集个数。

**定理 3.34**：$|G|=[G:H]\,|H|$。有限时 $|H|$ 整除 $|G|$。

**证明（三步）：**

**(1) 陪集铺满 $G$。** 任意 $g\in G$ 都有 $g=ge\in gH$，故 $G=\bigcup_{g\in G}gH$。

**(2) 陪集两两不交。** 由引理 3.30(3)，不同的陪集不交；故上面这个并实际上是**不交并**。

**(3) 每块大小相同。** 由引理 3.30(4)，每个陪集大小都是 $|H|$。

把三条拼起来：$G$ 是 $[G:H]$ 个互不交的、每个大小 $|H|$ 的块之并，故
$$|G|=\underbrace{|H|+\cdots+|H|}_{[G:H]\ \text{个}}=[G:H]\,|H|.$$
有限时两边都是整数且 $|H|>0$，故 $|H|\mid|G|$。$\blacksquare$

> **这个证明的手法比结论更重要**：**用等价关系把大集合切成等大的块**。同一模式出现在"商空间维数相加"（线代）、"完全剩余系"（数论）、[09 子群对应定理](./09-isomorphism-theorems-II.md)（格对应）。

**三个"不能"（拉格朗日的逆命题全都不成立）**：

- **$d\mid|G|$ 不能推出存在 $d$ 阶子群**。标准反例：$A_4$（$|A_4|=12$）没有 6 阶子群。**证明**（自足，只用到定义与拉格朗日）：

  设 $H\le A_4$、$|H|=6$。先确定 $H$ 里元素的阶。

  - **$H$ 含 3 阶元素。** 反设 $H$ 中非单位元阶全为 2。这种群必阿贝尔（$xy=(xy)^{-1}=y^{-1}x^{-1}=yx$），而有限阿贝尔群中所有元素阶整除 2 意味着 $H$ 是 $\mathbb{F}_2$ 上的向量空间，$|H|=2^k$ 是 2 的幂——与 $6$ 矛盾。故存在 $a\in H$、$o(a)=3$。
  - **$H$ 的另一个陪集元素 $b\notin\langle a\rangle$ 阶为 2。** $|H|=6$、$|\langle a\rangle|=3$，故 $[H:\langle a\rangle]=2$，即 $H=\{e,a,a^2\}\cup b\langle a\rangle$。若 $o(b)=3$，则 $\langle b\rangle$ 是另一个 3 阶子群，$\langle a\rangle\cap\langle b\rangle=\{e\}$（两个不同 3 阶子群之交是 $\langle a\rangle$ 的真子群，阶整除 3 故为 1），于是 $|\langle a\rangle\langle b\rangle|=3\cdot3=9>6$ 不可能。故 $o(b)=2$。
  - **$ba=a^2b$。** 因 $[H:\langle a\rangle]=2$ 得 $\langle a\rangle\triangleleft H$，故 $bab^{-1}\in\{a,a^2\}$。若 $bab^{-1}=a$，则 $ab=ba$，此时 $ab$ 的阶是 $\operatorname{lcm}(2,3)=6$——但 $H\subseteq A_4$ 中没有 6 阶元素，矛盾。故 $bab^{-1}=a^2$，即 $ba=a^2b$。
  - **$H=\{e,a,a^2\}\cup\{b,ab,a^2b\}$ 且 $b,ab,a^2b$ 都是 2 阶。** $(ab)^2=abab=a(ba)b=a(a^2b)b=a^3b^2=e$；$a^2b$ 同理。
  - **矛盾。** 那 3 个 2 阶元素必须是 $A_4$ 里的三个双对换（$A_4$ 恰有 3 个 2 阶元素：$(1,2)(3,4),(1,3)(2,4),(1,4)(2,3)$），它们连同单位元构成 $V=\{e,t_1,t_2,t_3\}\le A_4$。于是 $V\subseteq H$、$|V|=4$，而 $4\nmid6=|H|$，与拉格朗日矛盾。

  （用下面命题 3.41 的"阶 6 分类"还有条更短的路：$H\cong C_6$ 或 $S_3$；前者含 6 阶元素而 $A_4$ 无，后者含 3 个 2 阶元素即 3 个双对换、进而含 $V=\{e,t_1,t_2,t_3\}$，而 $4\nmid6$。）

  **记住这个反例的分量**：拉格朗日只给**候选阶**，是必要条件而非充分条件。
- **"大小整除"与"是子群"互不推出**：$\mathbb{Z}/6\mathbb{Z}$ 中 $\{\bar0,\bar2,\bar4\}$ 是 3 阶子群，而 $\{\bar0,\bar2\}$ 虽有 $2\mid6$ 却因 $\bar2+\bar2=\bar4\notin$ 该集合而不封闭——**运算封闭性无法由大小推得**。
- **无限群里 $[G:H]$ 与 $|H|$ 至少一个为 $\infty$**：$\mathbb{Z}$ 与 $m\mathbb{Z}$（$m\ne0$）时 $|m\mathbb{Z}|=\infty$、$[\mathbb{Z}:m\mathbb{Z}]=m<\infty$——商可以有限而子群无限大。**这也说明拉格朗日定理的"无限版本"必须写成乘积**：$|\mathbb{Z}|=\infty=m\cdot\infty$。

**推论 3.39**：
1. $o(g)\mid|G|$
2. $g^{|G|}=e$（费马小定理的群论版）
3. $|G|=p$（质数）$\Rightarrow G\cong C_p$

**证明：**

1. 由[推论 2.32](./03-symmetric-and-cyclic-groups.md)，$|\langle g\rangle|=o(g)$；$\langle g\rangle\le G$，故由拉格朗日定理 $o(g)=|\langle g\rangle|\mid|G|$。
2. 设 $|G|=n$，$o(g)=m$，由 (1) 知 $m\mid n$，写 $n=mk$。则 $g^n=(g^m)^k=e^k=e$。
3. $G$ 中任取非单位元 $g$，则 $o(g)\mid p$ 且 $o(g)>1$，故 $o(g)=p=|G|$；由[定理 2.39](./03-symmetric-and-cyclic-groups.md) $G=\langle g\rangle\cong C_p$。$\blacksquare$

> **例（费马小定理的群论版 = 经典费马小定理）**：取 $G=(\mathbb{Z}/p\mathbb{Z})^\times$（$p$ 质数，$|G|=p-1$）与任一 $a\not\equiv0\pmod p$，推论 3.39(2) 给出
> $$a^{p-1}\equiv1\pmod p\qquad(\text{Fermat's little theorem}).$$
> 例如 $p=7$：$2^6=64\equiv1$、$3^6=729\equiv1$ ✓。**这就是"抽象定理自动产出数论结论"的示范**——费马小定理在数论里要单独证，在群论里免费。
>
> **例（Euler 定理）**：把 $p$ 换成任意 $n$，取 $G=(\mathbb{Z}/n\mathbb{Z})^\times$（$|G|=\varphi(n)$）得
> $$a^{\varphi(n)}\equiv1\pmod n\quad(\gcd(a,n)=1).$$
> 如 $n=12$、$a=5$：$\varphi(12)=4$、$5^4=625\equiv1\pmod{12}$ ✓。
>
> **例（质数阶群只有一种）**：$|G|=5\Rightarrow G\cong C_5$；$|G|=7\Rightarrow G\cong C_7$。这解释了为什么"质数阶无分类问题"——统计上所有质数阶群都是同一个。

**命题 3.25 的证明**：$n\ge2$ 时 $|A_n|=n!/2$。

证明：取对换 $\tau=(1,2)$，它是奇排列（[引理 3.19](./04-group-properties-I.md)，$k=2$ 时 $sgn=(-1)^{2-1}=-1$）。断言
$$S_n=A_n\ \cup\ (1,2)A_n,\qquad A_n\cap(1,2)A_n=\emptyset.$$
- **并集**：任意 $\sigma\in S_n$，若 $\operatorname{sgn}(\sigma)=1$ 则 $\sigma\in A_n$；若 $\operatorname{sgn}(\sigma)=-1$，则 $\operatorname{sgn}((1,2)^{-1}\sigma)=\operatorname{sgn}((1,2))\operatorname{sgn}(\sigma)=(-1)(-1)=1$，故 $(1,2)^{-1}\sigma=(1,2)\sigma\in A_n$，即 $\sigma\in(1,2)A_n$ ✓。
- **不交**：若 $\sigma\in A_n\cap(1,2)A_n$，则 $\sigma=(1,2)\alpha$（$\alpha\in A_n$），取 $sgn$ 得 $1=(-1)\cdot1=-1$，矛盾 ✓。

故 $[S_n:A_n]=2$，由拉格朗日定理 $|A_n|=|S_n|/[S_n:A_n]=n!/2$。$\blacksquare$

> **同一论证证明"指数 2 ⇒ 正规"**（[06 定理 4.2](./06-normal-subgroup-and-quotient-group.md)）：上面两块不交并的论证只用"$[G:H]=2$"，故对任何指数 2 的子群都对：$G=H\cup gH\Rightarrow gH=G\setminus H$，同理 $Hg=G\setminus H$，故 $gH=Hg$。
>
> **例（$A_n$ 的大小）**：$|A_3|=3,\ |A_4|=12,\ |A_5|=60,\ |A_6|=360$。$A_5$ 是**最小的非阿贝尔单群**（$|A_5|=60$，它是单群且非阿贝尔——历史上催生了 Galois 的整个理论）。

### 阶 6 分类

**命题 3.41**：$|G|=6\Rightarrow G\cong C_6$ 或 $G\cong S_3$。

**证明：**

**第一步，判据：有 6 阶元素则循环。** 若存在 $g\in G$ 使 $o(g)=6$，由[定理 2.39](./03-symmetric-and-cyclic-groups.md) $G=\langle g\rangle\cong C_6$。**故以下假设无 6 阶元素**，要证 $G\cong S_3$。

**第二步，元素的阶只能取 1, 2, 3。** 由下文推论 3.39(1)（$o(x)$ 整除群阶），$o(x)\mid6$，故 $o(x)\in\{1,2,3,6\}$；已排除 6，故非单位元阶为 2 或 3。

**第三步，排除"非单位元全为 2 阶"。** 若所有非单位元满足 $x^2=e$，则 $G$ 阿贝尔（$xy=(xy)^{-1}=y^{-1}x^{-1}=yx$），且 $G$ 是 $\mathbb{F}_2$ 上向量空间，$|G|=2^k$。但 $|G|=6$ 不是 2 的幂，矛盾。（也可直接用 Cauchy 型结论"$p\mid|G|\Rightarrow$ 有 $p$ 阶元素"的思路：$3\mid6$ 必须有 3 阶元素——后面 [09](./09-isomorphism-theorems-II.md) 会给出该引理。）故**存在 3 阶元素**。

**第四步，排除"非单位元全为 3 阶"（即无 2 阶元素）。** 取 $b\in G$、$o(b)=3$。若还有 $c\notin\langle b\rangle$ 且 $o(c)=3$，则 $\langle b\rangle\cap\langle c\rangle=\{e\}$（交是 $\langle b\rangle$ 的真子群，其阶整除 3 故为 1），于是
$$|\langle b\rangle\langle c\rangle|=\frac{3\cdot3}{1}=9>6=|G|,$$
矛盾（用[第二同构定理的阶公式](./08-isomorphism-theorems-I.md)或直接说 $9$ 个元素互不相同）。故 $G$ 中所有 3 阶元素都在同一个 $\langle b\rangle=\{e,b,b^2\}$ 里，即除 $e,b,b^2$ 外的 $G\setminus\langle b\rangle$ 中三个元素**都没有 3 阶**——但它们也不是 6 阶（假设），也不能是 1 阶（非单位元），故**都是 2 阶**。特别地取 $a\in G\setminus\langle b\rangle$、$o(a)=2$。

**第五步，$a,b$ 不能交换。** 反设 $ab=ba$。则 $ab$ 的阶为 $\operatorname{lcm}(2,3)=6$（[01 交换情形幂律](./01-introduction-to-group-theory.md)＋阶的定义），与"无 6 阶元素"矛盾。故 $ab\neq ba$。

**第六步，枚举出全部 6 个元素。** 考虑集合
$$T=\{e,\ a,\ b,\ b^2,\ ab,\ ab^2\}.$$
断言这 6 个元素两两不同。（$e\ne a,e\ne b,b\ne b^2$ 显然；$a\ne b^j$ 因 $a\notin\langle b\rangle$；$ab=a b^2\Rightarrow b=b^2$ 矛盾；$ab=b^j\Rightarrow a=b^{j-1}\in\langle b\rangle$ 矛盾；$ab^2=b^j$ 同理。）因 $|G|=6$，故
$$G=\{e,a,b,b^2,ab,ab^2\}.$$

**第七步，算出完整的乘法表。** 关键关系 $ba$：由第五步 $ba\ne ab$。又 $ba\in G=T$，逐个排除：$ba=e$ 会逼出 $a=b^{-1}\in\langle b\rangle$（矛盾，因 $a\notin\langle b\rangle$）；$ba=a$ 由消去律得 $b=e$（矛盾）；$ba=b^j$ 得 $a=b^{j-1}\in\langle b\rangle$（矛盾）。剩下的唯一候选是 $ba=ab^2$。验算它自洽：这等价于 $a^{-1}ba=b^2$，即 $a^{-1}ba=b^{-1}$，也就是"用 $a$ 共轭把 $b$ 翻成 $b^{-1}$"。于是乘法完全由生成元关系
$$a^2=e,\quad b^3=e,\quad aba^{-1}=b^{-1}\ \ (\text{等价地}\ ba=ab^{-1}=ab^2)$$
决定，任何乘积都能化到 $a^i b^j$（$i\in\{0,1\},j\in\{0,1,2\}$）的标准型。

**第八步，构造到 $S_3$ 的同构。** 在 $S_3$ 中取
$$\tilde a=(1,2),\qquad \tilde b=(1,2,3),$$
验证同样满足：$\tilde a^2=e$、$\tilde b^3=e$、而
$$\tilde a\tilde b\tilde a^{-1}=(1,2)(1,2,3)(1,2).$$
用共轭即"重命名"（[01 命题 1.13](./01-introduction-to-group-theory.md) 的机制）：$(1,2)$ 把 $1\leftrightarrow2$ 对换，故 $(1,2,3)$ 被重命名成 $(2,1,3)=(1,3,2)=\tilde b^{-1}$ ✓。故映射
$$\varphi:G\to S_3,\qquad \varphi(a^ib^j)=\tilde a^i\tilde b^j$$
是良定义的（第七步的每个关系在 $S_3$ 中成立）、同态、满射（$|\varphi(G)|$ 至少含 6 个元素，而 $|S_3|=6$）、单射（$|G|=6$）。故 $G\cong S_3$。$\blacksquare$

> **这个证明是"按阶分类群"的完整范式**：① 用元素阶谱排除分支；② 从阶谱逼出生成元；③ 算出生成元之间的关系；④ 把关系"实现"到已知群上完成同构。**结构完全由生成元与关系决定**——[09 阶 8 分类](./09-isomorphism-theorems-II.md)会把同一套路推到 8 阶（结果 5 个群），而再往上就需要 Sylow 定理了。
>
> **具体例子（$D_3$ 与 $S_3$ 的两个"化身"）**：$D_3$ $=C_3\rtimes C_2$（$r$ 对应 $b$、$m$ 对应 $a$，关系 $mrm^{-1}=r^{-1}$ 正是 $aba^{-1}=b^{-1}$）；而 $S_3$ 里 $a=(1,2)$、$b=(1,2,3)$。两者由本命题统一：**$|G|=6$ 的非阿贝尔群只有 $S_3$ 的型**。

### 阶 6 全分类表（把上面的信息压成一张表）

| 群 | 阿贝尔？ | 元素阶谱 | 生成元关系 |
|---|---|---|---|
| $C_6\cong C_2\times C_3$ | 是 | 1 个 1 阶、1 个 2 阶、2 个 3 阶、2 个 6 阶 | $g^6=e$ |
| $S_3\cong D_3$ | 否 | 1 个 1 阶、3 个 2 阶、2 个 3 阶 | $a^2=b^3=e,\ aba^{-1}=b^{-1}$ |

（注意 $C_6$ 里 6 阶元素的个数是 $\varphi(6)=2$，与 $S_3$ 的 3 阶元素个数 2 巧合相同——**别靠单个数字认群，要靠完整阶谱**。）

### Cayley 定理

**定理 3.42（Cayley）**：任何 $n$ 阶有限群 $G$ 都同构于 $S_n$ 的某个子群。

**证明分三步。**

**(1) 左乘是一个排列。** 对每个 $g\in G$ 定义
$$\sigma_g:G\to G,\qquad \sigma_g(x)=gx.$$
则 $\sigma_g$ 是双射：单射由左消去律（$gx=gy\Rightarrow x=y$）；满射是因任意 $y\in G$ 等于 $\sigma_g(g^{-1}y)$。故 $\sigma_g$ 是 $G$ 上的一个排列，即 $\sigma_g\in S_G$（$G$ 的对称群；$|G|=n$ 时 $S_G\cong S_n$）。

**(2) $g\mapsto\sigma_g$ 是同态 $G\to S_G$。** 对任意 $x\in G$，
$$\sigma_{gh}(x)=(gh)x=g(hx)=\sigma_g(\sigma_h(x))=(\sigma_g\circ\sigma_h)(x),$$
故 $\sigma_{gh}=\sigma_g\circ\sigma_h$：**左乘把群乘法翻译成映射复合**。

**(3) 该同态是单射，故 $G\cong\Gamma:=\{\sigma_g:g\in G\}\le S_G$。** 若 $\sigma_g=\mathrm{id}$，则对 $x=e$ 有 $g=g\cdot e=\sigma_g(e)=e$，故 $\ker=\{\sigma_g\text{ 为恒等}\}=\{e\}$；由[命题 5.1](./08-isomorphism-theorems-I.md)（或直接：单射判据）映射是单射。于是 $G\cong\Gamma$，而 $\Gamma$ 是 $S_G$ 的子群、$S_G\cong S_n$。$\blacksquare$

> **所谓"正则表示"**：$g\mapsto\sigma_g$ 就叫 $G$ 的**左正则表示**。$\Gamma\cong G$ 是 $S_G$ 中最"忠实"的拷贝。
>
> **例（$C_4$ 的 Cayley 嵌入）**：$G=\mathbb{Z}/4\mathbb{Z}=\{0,1,2,3\}$（加法，故 $\sigma_{\bar k}(x)=x+k$）——
> $$\sigma_0=\mathrm{id},\qquad \sigma_1=(0,1,2,3),\qquad \sigma_2=(0,2)(1,3),\qquad \sigma_3=(0,3,2,1).$$
> 直接核对：$\sigma_1$ 把 $0\to1,1\to2,2\to3,3\to0$，故 $\sigma_1=(0,1,2,3)$；$\sigma_2$ 把 $0\to2,2\to0,1\to3,3\to1$，故 $\sigma_2=(0,2)(1,3)$；$\sigma_3$ 把 $0\to3,3\to2,2\to1,1\to0$，故 $\sigma_3=(0,3,2,1)$。于是 $\Gamma=\{e,(0,1,2,3),(0,2)(1,3),(0,3,2,1)\}\cong C_4\le S_4$ ✓（注意 $\sigma_1$ 与 $\sigma_3$ 是互逆的 4-cycle）。
>
> **例（$S_3$ 的 Cayley 嵌入是自身）**：$G=S_3$ 时 $\Gamma\le S_{S_3}\cong S_6$，但 $S_3$ 本身已经在 $S_3$ 里——Cayley 给的嵌入通常**不是最优的**（$|G|=n$ 时嵌到 $S_n$，但小群常能嵌到更小的对称群）。
>
> **换位视角（为什么这条定理重要）**：抽象群 $G$ 的元素自己变成"作用在 $G$ 上的变换"。**任何有限群都是具体的**——不存在"纯抽象"的有限群；且"研究群 = 研究它怎么作用于某个集合"这一方法论从此有了地基。这与[魔方群](../rubiks-cube-group-theory.md) 的视角完全一致：魔方群就是"作用在 54 个贴纸上的置换群"。
>
> **变形与加强（本系列未展开）**：若 $G$ 有指数 $k$ 的子群，可嵌到 $S_k$（**广义 Cayley / 群作用**）；$|G|$ 是质数幂时能嵌得更小。这些是 Sylow 与群作用理论的开端。

> **换位视角**：抽象群 $G$ 的元素自己变成"作用在 $G$ 上的变换"。任何有限群都是具体的——不存在"纯抽象"的有限群。

## 为什么重要

- 拉格朗日定理是**有限群论的第一次收网**：$o(g)\mid|G|$、质数阶必循环、$|A_n|=n!/2$ 全是它的即插即用
- 阶 6 分类示范了"按阶分类群"的完整打法——后面[阶 8 分类](./09-isomorphism-theorems-II.md)（5 个群）是同一方法在小阶上的极限
- Cayley 定理给[魔方群](../rubiks-cube-group-theory.md) 这样的"作用群"提供了理论地基：研究群 = 研究它怎么作用于某个集合

## 局限 / 边界

- 拉格朗日是必要条件不是充分条件：知道 $|G|$ 只能列出**候选阶**，存在性要另行构造（这正是后面"分类定理"的难度来源）
- 左陪集等于右陪集不保证逐点交换：$A_3\triangleleft S_3$ 有 $(1,2)A_3=A_3(1,2)$，但 $(1,2)(1,2,3)\ne(1,2,3)(1,2)$——这是通往[正规子群](./06-normal-subgroup-and-quotient-group.md)的伏笔

## 我的理解

- 陪集论证的精髓：**用等价关系把大集合切成等大的块**。这与线代里"商空间维数相加"、数论里"完全剩余系"是同一个模式
- 阶 6 分类的证明表面是暴力枚举，内核是"元素阶的分布约束群结构"——这种计数 + 穷举在阶 $\le 15$ 的分类里通用，再往上就需要 Sylow 定理（本系列未覆盖）

## Related

- [04 群的性质（上）](./04-group-properties-I.md) — sgn 同态，$A_n$ 是其核
- [06 正规子群与商群](./06-normal-subgroup-and-quotient-group.md) — 指数 2 的子群必正规
- [08 同构定理（上）](./08-isomorphism-theorems-I.md) — 例 5.4 用第一同构定理重证"指数 2 正规"
- [魔方群的数学结构](../rubiks-cube-group-theory.md) — $|G|=|\ker\pi|\cdot|P|$ 的计数结构与拉格朗日同源
