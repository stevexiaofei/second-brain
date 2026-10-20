---
title: 群的构造（上）：正规子群与商群、正规化子与中心化子
type: concept
status: growing
tags: [Mathematics, Group Theory, Normal Subgroup, Quotient Group, Normalizer, Centralizer]
created: 2026-10-14
updated: 2026-10-14
source: 知乎《群论及抽象代数学习笔记（6）- 群的构造（上）》（大PPP，NTU MH2220）
---

# 群的构造（上）：正规子群与商群、正规化子与中心化子

> 系列第四章上。回答"什么样的子群上可以造商群"，并给出度量"多不交换"的两个刻度：正规化子与中心化子。原文：<https://zhuanlan.zhihu.com/p/617509036>

## 一句话理解

正规子群 $gH=Hg$ 让"陪集相乘"与代表元选取无关，于是商集 $G/H$ 升级成商群；中心化子度量一个元素"有多被交换"，正规化子度量一个子群"有多被共轭稳定"。

## 什么是它

### 正规子群

**定义 4.1**：$H\triangleleft G\iff gH=Hg$ 对所有 $g\in G$。

**等价的四个说法**（互相等价，用时挑最方便的）：

$$H\triangleleft G\iff \forall g:\ gH=Hg\iff \forall g:\ gHg^{-1}=H\iff \forall g,h:\ ghg^{-1}\in H\iff \forall g:\ gHg^{-1}\subseteq H.$$

**证明（五条首尾相接）：**

1. 逐条验证相邻两条等价。
   - **$(a)\Leftrightarrow(b)$**：$gH=Hg$ 两边右乘 $g^{-1}$ 得 $gHg^{-1}=Hgg^{-1}=H$；反之 $gHg^{-1}=H$ 两边右乘 $g$ 得 $gH=Hg$。**集合等式两边同乘同一元素是合法的**（$A=B\Rightarrow Ax=Bx$）。
   - **$(b)\Leftrightarrow(c)$**：$gHg^{-1}=H$ 是集合等式，等价于"$gHg^{-1}$ 的每个元素都 $\in H$ 且每个 $H$ 中元素都被取到"。而
     $$gHg^{-1}=\{ghg^{-1}:h\in H\},$$
     故 "$gHg^{-1}=H$" $\iff$ "$ghg^{-1}\in H$ 对一切 $h\in H$，且映射 $h\mapsto ghg^{-1}$ 是 $H\to H$ 的双射（它总单射，$g(\cdot)g^{-1}$ 可逆）"。后者自动成立（$H$ 有限时由单射即满射；无限时用 $g^{-1}hg=(g^{-1})h(g^{-1})^{-1}\in H$ 反向）。故 $(b)\Leftrightarrow(c)$。
   - **$(c)\Leftrightarrow(d)$**："$\forall h:ghg^{-1}\in H$" 就是说 $gHg^{-1}\subseteq H$；$(d)\Rightarrow(c)$ 是同一句话。
   - **$(d)\Rightarrow(a)$**：设 $gHg^{-1}\subseteq H$ 对一切 $g$ 成立。取 $x\in gH$，写 $x=gh$，则
     $$x=(ghg^{-1})g\in Hg$$
     （因 $ghg^{-1}\in H$），故 $gH\subseteq Hg$。再对 $g^{-1}$ 用 (d)：$g^{-1}Hg\subseteq H$；两边左乘 $g$、右乘 $g^{-1}$ 得 $H\subseteq gHg^{-1}$。结合假设 $gHg^{-1}\subseteq H$ 得 $gHg^{-1}=H$。因 $g$ 任意，换 $g$ 为 $g^{-1}$ 的同一条给出 $g^{-1}Hg=H$，即 $Hg=gH$，故 $gH=Hg$ ✓。

> - $\{e\}$ 与 $G$ 总是正规的；**阿贝尔群的一切子群都正规**（$ghg^{-1}=gg^{-1}h=h$）
> - $gH=Hg$ 是**集合相等**，不等于逐点交换（见下反例 4.3）

**命题（正规子群的若干等价刻画，实用版）**：
1. $H\triangleleft G\iff N_G(H)=G$（见定义 4.14）
2. $H\triangleleft G$ 且 $K\le G$ $\Rightarrow$ $H\cap K\triangleleft K$、$HK=KH$、$HK\le G$
3. $H\triangleleft G\Rightarrow$ $H$ 是若干共轭类的并（$H$ 对共轭封闭）——这是**实际判断正规性最快的方法**：把 $G$ 按共轭类分解成块，看 $H$ 是不是整块整块地取
4. $H\triangleleft G\iff H$ 是某个同态的核

> **(4) 是"正规性"最深的意义**：一开始 "正规" 看起来只是"能造商群"的技术条件，但它其实就是"能当核"的代数条件。这解释了为什么后面[第一同构定理](./08-isomorphism-theorems-I.md) 里 $\ker\phi\triangleleft G$ 是自然的，也解释了[魔方群](../rubiks-cube-group-theory.md) 中 $\ker\pi$（朝向部分）为什么天然正规。
>
> **例（用共轭类判正规，$S_4$）**：$S_4$ 的共轭类按 cycle type 分成 5 块（大小 $1,6,3,8,6$，见 [03](./03-symmetric-and-cyclic-groups.md) 的表）。正规子群必须是若干块的并就包含 $\{e\}$，且大小整除 $|S_4|=24$：可能的并集大小只有 $1$（$\{e\}$）、$4=1+3$（$V_4=\{e,\text{三个双对换}\}$）、$24$（全体）。故 $S_4$ 的正规子群恰为 $\{e\},V_4,A_4,S_4$（$A_4$ 大小 $12=1+3+8$ ✓）。**这张"共轭类 + 整除"的筛子把无穷的检查压成几步**。

**定理 4.2**：$[G:H]=2\Rightarrow H\triangleleft G$。

**证明**：设 $[G:H]=2$，即左陪集只有两个。任取 $g\in G$：

- **$g\in H$** 时，由[引理 3.30(1)](./05-group-properties-II.md) $gH=H$ 且 $Hg=H$，故 $gH=Hg$ ✓。
- **$g\notin H$** 时，$gH$ 与 $Hg$ 都是**非 $H$ 的陪集**。左陪集只有 $H$ 与 $gH$ 两个（$[G:H]=2$），且由不交性 $gH\cap H=\emptyset$，$|gH|=|H|$，而 $|G|=2|H|$，故 $gH=G\setminus H$；同理 $Hg=G\setminus H$。两式比较得 $gH=Hg$ ✓。

故 $gH=Hg$ 对一切 $g$ 成立，$H\triangleleft G$。$\blacksquare$

> 例：$A_n\triangleleft S_n$（$[S_n:A_n]=2$，见 [05 命题 3.25](./05-group-properties-II.md)）；$SL_n\triangleleft GL_n$ 不行（$[GL_n:SL_n]=\infty$ 或非 2——需另证，用它等于 $\ker\det$ 更省事）。

**反例 4.3（集合相等 $\ne$ 逐点相等）**：$A_3\triangleleft S_3$（$[S_3:A_3]=2$），取 $g=(1,2)$。

**左陪集** $(1,2)A_3$（复合从右往左：先 $A_3$ 中元素，再 $(1,2)$）：
$$(1,2)e=(1,2),\qquad (1,2)(1,2,3)=(2,3),\qquad (1,2)(1,3,2)=(1,3),$$
其中第二个的逐点追踪是 $1\xrightarrow{(1,2,3)}2\xrightarrow{(1,2)}1$、$2\to3\to3$、$3\to1\to2$，得 $(2,3)$；第三个同法得 $(1,3)$。故
$$(1,2)A_3=\{(1,2),(2,3),(1,3)\}.$$

**右陪集** $A_3(1,2)$：
$$e(1,2)=(1,2),\qquad (1,2,3)(1,2)=(1,3),\qquad (1,3,2)(1,2)=(2,3),$$
（第二个：$1\xrightarrow{(1,2)}2\xrightarrow{(1,2,3)}3$、$2\to1\to2$、$3\to3\to1$，得 $(1,3)$。）故
$$A_3(1,2)=\{(1,2),(1,3),(2,3)\}.$$

**两个集合完全相同**——$A_3$ 确实正规。但**逐点比较会失败**：
$$(1,2)\cdot(1,2,3)=(2,3)\ \ne\ (1,2,3)\cdot(1,2)=(1,3).$$
**结论**：$gH=Hg$ 的含义是"对每个 $h_1$ 存在 $h_2$（**依赖 $h_1$**）使 $gh_1=h_2g$"——不是"$gh=hg$ 对所有 $h$"。这里 $(1,2)$ 把 $A_3$ 中三个元素**轮换着洗了一遍**（$(2,3)\leftrightarrow(1,3)$ 互换，"洗牌"的结果恰是同一个集合），但没有固定任何一个。

**反例 4.4**：$H=\langle(1,2)\rangle=\{e,(1,2)\}\le S_3$ 不正规。取 $g=(1,3)$：
$$(1,3)H=\{(1,3),(1,3)(1,2)\}=\{(1,3),(1,3,2)\},$$
$$H(1,3)=\{(1,3),(1,2)(1,3)\}=\{(1,3),(1,2,3)\},$$
两者不同（$(1,3,2)$ vs $(1,2,3)$）。故 $H\ntriangleleft S_3$——这与[推论 4.12](#) 不冲突：$S_3$ 非阿贝尔，故其子群不必正规。

> **与共轭类视角对照**：$\{e,(1,2)\}$ 不是共轭类的并（$(1,2)$ 的共轭类含全部 3 个对换，大小 3），故不正规——与"共轭类筛子"一致。

### 商群

**定理 4.6**：$H\triangleleft G$ 时，定义
$$(xH)*(yH)=(xy)H,$$
则 $G/H$ 成群（单位元 $H$，逆元 $x^{-1}H$）。

**证明分两部分——良定义性是全部难点。**

**第一部分：良定义。** 运算的公式用**代表元** $x,y$，必须验证换代表元不改变结果。设
$$xH=x'H,\qquad yH=y'H.$$
由[引理 3.30(2)](./05-group-properties-II.md)，等价地 $x'=xb$、$y'=ya$，其中 $b=x^{-1}x'\in H$、$a=y^{-1}y'\in H$。目标：证明 $xy$ 与 $x'y'$ 在同一个 $H$-陪集里，即 $(xy)^{-1}(x'y')\in H$。计算
$$(xy)^{-1}(x'y')=y^{-1}x^{-1}x' y'=y^{-1}\underbrace{(x^{-1}x')}_{b}y'=y^{-1}b\,y'.$$
现在 $y^{-1}by$ 与 $y'=ya$ 的乘积——**插入 $H$ 的元素重新结合**：
$$y^{-1}by'=y^{-1}b\,ya=\underbrace{(y^{-1}by)}_{\in H}\underbrace{a}_{\in H}\in H,$$
（第一个因子 $\in H$ 正是**正规性**：$y^{-1}Hy=H$，即 $y^{-1}by\in H$。）故 $(xy)^{-1}(x'y')\in H$，即 $x'y'H=xyH$。良定义 ✓。

> **原文的写法与此等价**：把 $x=x'b,\ y=y'a$ 代入得
> $$xy=x'(by')a,$$
> 而 $by'$ 与 $y'$ 在同一个右陪集里，故 $by'=y'c$（某个 $c\in H$，由 $Hy'=y'H$ —— **这一步用的就是正规性**），于是
> $$xy=x'(y'c)a=x'y'(ca)\in x'y'H.$$
> 两种写法都卡在同一个地方：**必须让 $H$ 从左边穿到右边**。没有 $gH=Hg$ 这一步，整个证明过不去——这就是"只有正规子群才有商群"的全部原因。

**第二部分：四条公理。**

1. **封闭**：$xH*yH=(xy)H$，仍是陪集 ✓（由良定义，这是 $G/H$ 中的元素）。
2. **结合律**：$(xH*yH)*zH=(xy)H*zH=(xyz)H$，而 $xH*(yH*zH)=xH*(yz)H=(xyz)H$ ✓——**下沉到 $G$ 的结合律**。
3. **单位元**：$eH=H$，$H*xH=(ex)H=xH$，$xH*H=(xe)H=xH$ ✓。
4. **逆元**：$xH*x^{-1}H=(xx^{-1})H=H$，$x^{-1}H*xH=H$ ✓。

故 $G/H$ 是群，$|G/H|=[G:H]$。$\blacksquare$

> **$|G/H|=[G:H]$ 由拉格朗日**：$|G|=|G/H|\cdot|H|$。

**例（$S_3/A_3$，手算一次）**：$A_3=\{e,(1,2,3),(1,3,2)\}$，$[S_3:A_3]=2$，故 $S_3/A_3=\{A_3,(1,2)A_3\}$，两个元素：
$$(1,2)A_3*(1,2)A_3=\bigl((1,2)(1,2)\bigr)A_3=eA_3=A_3,$$
故 $S_3/A_3\cong C_2$。**注意**：这里 $(1,2)A_3$ 的平方是单位元，与 $\mathbb{Z}/2\mathbb{Z}$ 一致——商的运算"记住的只是陪集层面的信息"（$A_3$ 内部的三元循环被完全抹平了）。

**例（$\mathbb{Z}/6\mathbb{Z}\big/\{0,2,4\}$）**：$H=\{\bar0,\bar2,\bar4\}\cong C_3$ 是正规子群（$\mathbb{Z}/6\mathbb{Z}$ 阿贝尔）。商群有 $[G:H]=2$ 个元素 $\{H,\bar1+H\}$，故 $\cong C_2$。**这就是 $\mathbb{Z}/6\mathbb{Z}$ 被"除以 3"得到 $\mathbb{Z}/2\mathbb{Z}$**——与说"$\mathbb{Z}/6\mathbb{Z}\cong\mathbb{Z}/2\mathbb{Z}\times\mathbb{Z}/3\mathbb{Z}$ 再投影"是一回事（[07 定理 4.23](./07-automorphism-and-products.md)）。

**例 4.7**：$S_n/A_n\cong C_2$（$n\ge2$）。由 $sgn:S_n\to\{\pm1\}$ 满且 $\ker=sgn=A_n$，套[第一同构定理](./08-isomorphism-theorems-I.md)即得；也可直接：$[S_n:A_n]=2$ 故商群阶 2，而阶 2 的群必 $\cong C_2$。

**例 4.7**：$S_n/A_n\cong C_2$（$n\ge2$）。

**例 4.11**：$G=(\mathbb{Z}/16\mathbb{Z})^\times$，$H=\langle\bar7\rangle=\{\bar1,\bar7\}$（$G$ 阿贝尔故 $H$ 正规，不必验证）。$|G|=\varphi(16)=8$，故 $|G/H|=4$。商群的元素：
$$G/H=\{H,\ \bar3H,\ \bar9H,\ \bar{11}H\},$$
其中 $\bar3H=\{\bar3,\bar3\cdot\bar7=\overline{21}=\bar5\}$、$\bar9H=\{\bar9,\bar{63}=\overline{15}\}$、$\bar{11}H=\{\bar{11},\overline{77}=\overline{13}\}$——四个陪集不交地覆盖 $G=\{\bar1,\bar3,\bar5,\bar7,\bar9,\bar{11},\overline{13},\overline{15}\}$ ✓。**商的阶**：$\bar3H$ 的平方是 $\bar9H$（非 $H$）、三次方是 $\bar{27}H=\bar{11}H$、四次方是 $\overline{33}H=\bar1H=H$，故 $\bar3H$ 的阶为 4，$G/H=\langle\bar3H\rangle\cong C_4$ ✓。

> **注意 $(\mathbb{Z}/16\mathbb{Z})^\times$ 本身是什么**：$\varphi(16)=8$，且 $\bar3$ 的阶是 4（$3^4=81\equiv1$）、$\bar{15}$ 的阶是 2，可以验证 $(\mathbb{Z}/16\mathbb{Z})^\times\cong C_4\times C_2$（两个生成元 $\bar3,\bar{15}$）。**这个例子的价值**：$G$ 非循环却能有 4 阶商群 $G/H$——**商群比原群"更循环"是常态**（把"噪声"模掉后剩下的可能更规整）。

**推论 4.12**：$G$ 阿贝尔 $\Rightarrow G/H$ 阿贝尔；$G$ 循环 $\Rightarrow G/H$ 循环。

**证明：**

1. **阿贝尔**：任意 $xH,yH\in G/H$，
   $$(xH)(yH)=(xy)H\overset{G\text{ 阿贝尔}}{=}(yx)H=(yH)(xH).$$
2. **循环**：设 $G=\langle g\rangle$。任意 $x\in G$ 写 $x=g^k$，则 $xH=(g^k)H=(gH)^k$，故 $G/H=\langle gH\rangle$ 循环。$\blacksquare$

> **反向都不成立**：$G/H$ 阿贝尔推不出 $G$ 阿贝尔（$S_3/A_3\cong C_2$ 阿贝尔而 $S_3$ 不是）；$G/H$ 循环推不出 $G$ 循环（$S_3/A_3$ 循环而 $S_3$ 不是）。**商只会丢信息，不会造信息**。

**例 4.13（$\mathbb{R}/\mathbb{Z}\cong$ 单位圆）**：$H=\mathbb{Z}\triangleleft\mathbb{R}$（$\mathbb{R}$ 阿贝尔）。定义
$$\psi:\mathbb{R}/\mathbb{Z}\to S^1,\qquad \psi(r+\mathbb{Z})=e^{2\pi ri}.$$

**验证四件事：**

- **良定义**：$r+\mathbb{Z}=r'+\mathbb{Z}$ 即 $r-r'=n\in\mathbb{Z}$，则
  $$\psi(r'+\mathbb{Z})=e^{2\pi r'i}=e^{2\pi(r-n)i}=e^{2\pi ri}\cdot e^{-2\pi ni}=e^{2\pi ri}\cdot1=\psi(r+\mathbb{Z})$$
  （$e^{-2\pi ni}=(e^{-2\pi i})^n=1^n=1$）✓。
- **同态**：$\psi\bigl((r+\mathbb{Z})+(s+\mathbb{Z})\bigr)=\psi((r+s)+\mathbb{Z})=e^{2\pi(r+s)i}=e^{2\pi ri}e^{2\pi si}$ ✓。
- **满射**：任意 $z\in S^1$ 写作 $z=e^{2\pi i\theta}$（$\theta\in[0,1)$），则 $z=\psi(\theta+\mathbb{Z})$ ✓。
- **单射**：$\psi(r+\mathbb{Z})=1$ 即 $e^{2\pi ri}=1$，即 $r\in\mathbb{Z}$，即 $r+\mathbb{Z}=\mathbb{Z}$。故 $\ker\psi=\{\mathbb{Z}\}$（商群的单位元），单射 ✓。

由[第一同构定理](./08-isomorphism-theorems-I.md) 也可直接得 $\mathbb{R}/\mathbb{Z}\cong S^1$（把 $\psi$ 看成 $\mathbb{R}\to S^1$、核为 $\mathbb{Z}$）。

> **直观**：把数轴按整数周期"卷起来"，分数部分 $\{r\}$ 就是圆上的角度。**例**：$0.25+\mathbb{Z}$ 对应 $e^{\pi i/2}=i$；$1.75+\mathbb{Z}$ 与 $-0.25+\mathbb{Z}$ 是**同一个陪集**，都对应 $e^{-\pi i/2}=-i$（注意 $-0.25\equiv0.75\pmod1$，$e^{2\pi i\cdot0.75}=e^{3\pi i/2}=-i$ ✓ 一致）。
>
> **这是"商 = 卷绕"的最干净实例**：模掉 $\mathbb{Z}$ 就是丢掉绝对刻度、只留相位。这个直觉对理解 $U(1)$ 规范群、周期现象、傅里叶分析（$\mathbb{R}/\mathbb{Z}$ 上做分析 = 单位圆上做分析）都有用。

### 正规化子与中心化子

**定义 4.14**：$H\le G$ 的**正规化子**
$$N_G(H)=\{g\in G:gHg^{-1}=H\}.$$
是 $G$ 的子群。**证明**：$e\in N_G(H)$；$g_1,g_2\in N_G(H)$ 时
$$(g_1g_2)H(g_1g_2)^{-1}=g_1\bigl(g_2Hg_2^{-1}\bigr)g_1^{-1}=g_1Hg_1^{-1}=H;$$
$g\in N_G(H)$ 时由 $gHg^{-1}=H$ 两边左乘 $g^{-1}$、右乘 $g$ 得 $g^{-1}Hg=H$，故 $g^{-1}\in N_G(H)$（这对"取逆封闭"来说已经够了）。且 $H\le N_G(H)\le G$、$H\triangleleft N_G(H)$；**$H\triangleleft G\iff N_G(H)=G$**——所以 $N_G(H)$ 度量"$H$ 在 $G$ 里有多正规"，它是"使 $H$ 正规的最大子群"。

**定义 4.15**：$x\in G$ 的**中心化子**
$$C_G(x)=\{g\in G:gx=xg\}.$$
对子集 $X\subseteq G$，$C_G(X)=\{g:gx=xg,\ \forall x\in X\}$；$X=G$ 时记 $Z(G)$，称 $G$ 的**中心**：
$$Z(G)=\{g\in G:gx=xg,\ \forall x\in G\}=\bigcap_{x\in G}C_G(x).$$

**证明这些都是子群（用判别法）**：以 $C_G(x)$ 为例——$e\in C_G(x)$；$g_1,g_2$ 都与 $x$ 交换时 $g_1g_2x=g_1xg_2=xg_1g_2$ ✓；$g\in C_G(x)$ 时 $g^{-1}x=g^{-1}xgg^{-1}=g^{-1}gxg^{-1}=xg^{-1}$ ✓。$C_G(X)=\bigcap_{x\in X}C_G(x)$ 是子群之交（[01 定义 1.20](./01-introduction-to-group-theory.md)），同理为子群。

**例（逐个算）**：

- **$Z(S_3)=\{e\}$**：$S_3$ 非阿贝尔，中心只能是 $\{e\}$（若 $g\in Z$，则 $g$ 与所有对换交换，而对换的共轭作用能把它变成任何东西——只有 $e$ 不动）。
- **$Z(D_4)=\{e,r^2\}$**：$r^2$（转 $180°$）与所有元素交换，$r$ 不（$mr\ne rm$），$m$ 不。**中心可以非平凡，哪怕群非阿贝尔**。
- **$C_{S_4}((1,2))=\langle(1,2),(3,4)\rangle\cong C_2\times C_2$**：与 $(1,2)$ 交换的元素有 $(1,2)$、$(3,4)$、$(1,2)(3,4)$、$e$，共 4 个。（检验：$(1,3)\notin C_{S_4}((1,2))$，因 $(1,3)(1,2)=(1,2,3)\ne(1,2,3)^{-1}$。）
- **$C_{S_4}((1,2,3,4))=\langle(1,2,3,4)\rangle\cong C_4$**：在 $S_4$ 中与一个 4-cycle 交换的元素恰是它的幂（一般结论：$n$-cycle 的中心化子在 $S_n$ 中就是它自己生成的循环群）。
- **$N_{S_3}(\langle(1,2)\rangle)=\langle(1,2)\rangle$**：与 $(1,2)$ 交换的元素才可能把 $\langle(1,2)\rangle$ 正规化，而 $C_{S_3}((1,2))=\langle(1,2)\rangle$，故 $N=\langle(1,2)\rangle$——$H\ntriangleleft S_3$ 与 $N_G(H)\ne G$ 一致 ✓。
- **$N_{S_3}(A_3)=S_3$**：$A_3\triangleleft S_3$ ✓（与定理 4.2 一致）。

> **关于 $C$ 与 $N$ 的关系**：$C_G(H)\le N_G(H)$——与 $H$ 逐点交换当然更强。$C_G(H)$ 是"稳定每个点"，$N_G(H)$ 是"稳定集合 $H$"（可以洗牌）。这个"逐点 vs 集合"的区别正是[反例 4.3](#) 的抽象版本。对 $H$ 阿贝尔时等式未必成立，但 $N_G(H)/C_G(H)$ 同构于 $Aut(H)$ 的某个子群（未展开，Sylow 理论的常用工具）。

## 为什么重要

- 商群是"把群除掉一个正规部分"的运算——所有同构定理（[08](./08-isomorphism-theorems-I.md)、[09](./09-isomorphism-theorems-II.md)）都是商群的性质；[魔方群](../rubiks-cube-group-theory.md) 求 $|G|$ 用的 $G/\ker\pi\cong P$ 就是一次商群操作
- 中心化子/正规化子是共轭作用的"稳定子"，为 Sylow 理论（未覆盖）与群作用计数铺路
- $C_{p^2}\ne C_p\times C_p$ 这类"同阶不同构"的分辨工作依赖商群与直积工具（[07](./07-automorphism-and-products.md)）

## 局限 / 边界

- 正规性是"可除性"的条件，不是"可交换性"的条件——商群 $G/H$ 阿贝尔只需要 $G$ 阿贝尔（或更弱条件），与 $H$ 内部交换与否无关
- $N_G(H)$、$C_G(x)$ 本身一般是子群但未必正规

## 我的理解

- 商群的困难全在良定义，而良定义的钥匙只有一把：$gH=Hg$。可以把正规子群理解为"商运算的许可证"——它保证你换代表元时运算结果不动
- $\mathbb{R}/\mathbb{Z}\cong S^1$ 是"商 = 卷绕"的最干净实例：模掉 $\mathbb{Z}$ 就是丢掉绝对刻度、只留相位。这个直觉对理解 $U(1)$、周期现象都有用

## Related

- [05 群的性质（下）](./05-group-properties-II.md) — 陪集与指数是商群的前置
- [07 自同构、直积与半直积](./07-automorphism-and-products.md) — $N\rtimes H$ 中 $N$ 必须正规
- [08 同构定理（上）](./08-isomorphism-theorems-I.md) — 商群性质的集中爆发
- [魔方群的数学结构](../rubiks-cube-group-theory.md) — $\ker\Phi$ 是正规子群、$G/G_0\cong\mathbb{Z}_3\times\mathbb{Z}_2\times\mathbb{Z}_2$
