---
title: 整数模 n 加法群与二面体群
type: concept
status: growing
tags: [Mathematics, Group Theory, Modular Arithmetic, Dihedral Group]
created: 2026-10-14
updated: 2026-10-14
source: 知乎《群论及抽象代数学习笔记（2）- 一些群的介绍（上）》（大PPP，NTU MH2220）
---

# 整数模 n 加法群与二面体群

> 系列第二章上。从同余关系出发构造 $(\mathbb{Z}/n\mathbb{Z},+)$ 与 $((\mathbb{Z}/n\mathbb{Z})^\times,\times)$，再从正多边形对称构造二面体群 $D_n$。原文：<https://zhuanlan.zhihu.com/p/604295896>

## 一句话理解

钟表算术给出第一批有限群（模 n 加法/乘法群），多边形对称给出第一批非阿贝尔群（$D_n$）——"等价类"与"生成元+关系"是构造群的两种基本手法。

## 什么是它

### 同余与商集

**定义 2.1**：$a\equiv b \bmod n \iff a=b+kn$（某个整数 $k$）。等价说法：余数相同 / $n\mid a-b$。

**推论 2.2**：模 $n$ 同余是等价关系（自反、对称、传递）。

证明（三条逐条验证，$a\equiv b$ 即 $n\mid a-b$）：

1. **自反**：$n\mid a-a=0$，故 $a\equiv a$。
2. **对称**：$a\equiv b$ 即 $a-b=kn$，则 $b-a=(-k)n$，故 $b\equiv a$。
3. **传递**：$a-b=kn$、$b-c=ln$，则 $a-c=(a-b)+(b-c)=(k+l)n$，故 $a\equiv c$。

> **传递性是"等价类铺满且不重叠"的唯一保障**：正因为传递，$\bar a$ 与 $\bar b$ 要么完全重合要么完全不交（[05 引理 3.30](./05-group-properties-II.md) 的陪集三性质是完全同构的论证）。等价关系三性质在抽代里出现的地方，背后都是同一个模式。

**定义 2.6**：等价类 $\bar a=\{b\in\mathbb{Z}:a\equiv b\bmod n\}$；所有不同等价类的集合记为 $\mathbb{Z}/n\mathbb{Z}$。例如 $\mathbb{Z}/2\mathbb{Z}=\{\bar0,\bar1\}$。

特例：$n=0$ 时 $\mathbb{Z}/0\mathbb{Z}\cong\mathbb{Z}$（逐元素）；$n=1$ 时 $\mathbb{Z}/1\mathbb{Z}=\{\bar0\}$ 只有一个元素。

### 模 n 加法群

**命题 2.8**：定义 $\bar a+\bar b=\overline{a+b}$，则 $(\mathbb{Z}/n\mathbb{Z},+)$ 是阿贝尔群。

**证明：**

**第一步，良定义性（well-defined）**——这是全篇最重要的一步。运算定义在**等价类**上，但公式右边用了**代表元** $a+b$。必须验证：换代表元不改变结果。

设 $\bar a=\bar c$、$\bar b=\bar d$，即 $a-c=kn$、$b-d=ln$（$k,l\in\mathbb{Z}$）。则
$$(a+b)-(c+d)=(a-c)+(b-d)=(k+l)n,$$
即 $n\mid(a+b)-(c+d)$，于是 $\overline{a+b}=\overline{c+d}$。故加法在等价类上是良定义的：$\bar a+\bar b$ 只依赖于 $\bar a,\bar b$ 本身。

**第二步，四条公理**（全都"下沉"到 $\mathbb{Z}$ 的加法，那里已经知道成立）：

1. **结合律**：$(\bar a+\bar b)+\bar c=\overline{(a+b)+c}=\overline{a+(b+c)}=\bar a+(\bar b+\bar c)$。
2. **单位元**：$\bar0+\bar a=\overline{0+a}=\bar a=\overline{a+0}=\bar a+\bar0$。
3. **逆元**：$\bar a+\overline{-a}=\overline{a+(-a)}=\bar0=\overline{-a}+\bar a$，即 $-\bar a=\overline{-a}$。
4. **阿贝尔**：$\bar a+\bar b=\overline{a+b}=\overline{b+a}=\bar b+\bar a$。

**第三步，$|\mathbb{Z}/n\mathbb{Z}|=n$**（$n\ge1$）：由带余除法，任何 $a=qn+r$（$0\le r<n$），故 $\bar a=\bar r$，所以每个类都被 $\bar0,\dots,\overline{n-1}$ 代表；而这 $n$ 个类两两不同（$0\le r<s<n$ 时 $0<s-r<n$，故 $n\nmid s-r$）。$\blacksquare$

> **良定义性在别处长什么样**：$\psi(r+\mathbb{Z})=e^{2\pi ri}$（[06 例 4.13](./06-normal-subgroup-and-quotient-group.md)）、$G/H$ 的乘法（[06 定理 4.6](./06-normal-subgroup-and-quotient-group.md)）、$G/\ker\phi\cong\operatorname{im}\phi$ 里的 $\psi(gN)=\phi(g)$（[08 引理 5.2](./08-isomorphism-theorems-I.md)）——**每一次在等价类/陪集上定义映射，都要重演上面第一步**。这是抽代里最容易被跳过又最容易出错的一步。

**例（$\mathbb{Z}/6\mathbb{Z}$ 的完整加法表）**：

| $+$ | $\bar0$ | $\bar1$ | $\bar2$ | $\bar3$ | $\bar4$ | $\bar5$ |
|---|---|---|---|---|---|---|
| $\bar0$ | $\bar0$ | $\bar1$ | $\bar2$ | $\bar3$ | $\bar4$ | $\bar5$ |
| $\bar1$ | $\bar1$ | $\bar2$ | $\bar3$ | $\bar4$ | $\bar5$ | $\bar0$ |
| $\bar2$ | $\bar2$ | $\bar3$ | $\bar4$ | $\bar5$ | $\bar0$ | $\bar1$ |
| $\bar3$ | $\bar3$ | $\bar4$ | $\bar5$ | $\bar0$ | $\bar1$ | $\bar2$ |
| $\bar4$ | $\bar4$ | $\bar5$ | $\bar0$ | $\bar1$ | $\bar2$ | $\bar3$ |
| $\bar5$ | $\bar5$ | $\bar0$ | $\bar1$ | $\bar2$ | $\bar3$ | $\bar4$ |

这个表**每行是上一行的循环移位**——阿贝尔群 Cayley 表的典型形状（对称矩阵形式）；非阿贝尔群的 Cayley 表则完全不对称（对比 [05](./05-group-properties-II.md) 的 $S_3$）。顺带读出各元素阶：$o(\bar1)=o(\bar5)=6$、$o(\bar2)=o(\bar4)=3$、$o(\bar3)=2$、$o(\bar0)=1$——正好是 $o(g^k)=n/\gcd(n,k)$（[03 定理 2.41](./03-symmetric-and-cyclic-groups.md)）在 $n=6$ 时的读数。

### 模 n 乘法群

先看反例：$(\mathbb{Z}/2\mathbb{Z},\times)$ 不是群——$\bar0$ 没有逆元（$\bar0\cdot\bar b=\bar0\ne\bar1$ 对一切 $\bar b$）。更一般地，$(\mathbb{Z}/n\mathbb{Z},\times)$（$n\ge2$）的 $\bar0$ 永远没有逆元，所以乘法在全体等价类上**不可能**成群。

**定义 2.11 / 命题 2.12**：取其中有逆元的部分
$$(\mathbb{Z}/n\mathbb{Z})^\times=\{\bar a:\bar a\bar b=\bar 1\ \text{某个}\ \bar b\}=\{\bar a:\gcd(a,n)=1,\ 0\le a\le n-1\},$$
则 $((\mathbb{Z}/n\mathbb{Z})^\times,\times)$ 是阿贝尔群。

**证明：**

**第一步，乘法良定义**：与加法同理。$\bar a=\bar c$、$\bar b=\bar d$ 时 $a=c+kn$、$b=d+ln$，则
$$ab-cd=a(b-d)+d(a-c)=aln+dkn=(al+dk)n,$$
故 $\overline{ab}=\overline{cd}$。（注意这里比加法多了一项交叉项 $a(b-d)$——乘法不是逐分量线性的，但 $n$ 依然被"带出去"。）

**第二步，逆元集合的刻画**（本命题的核心，两边互相推出）：

> **引理（Bézout 恒等式）**：$\gcd(a,n)=1\iff\exists b,c\in\mathbb{Z}$ 使 $ab+nc=1$。

- **($\Leftarrow$) 由 Bézout 到可逆**：若 $ab+nc=1$，两边模 $n$ 得 $\overline{ab}=\bar1$（因为 $\overline{nc}=\bar0$），即 $\bar a\bar b=\bar1$。故 $\bar a$ 可逆，且逆元就是 $\bar b$。
- **($\Rightarrow$) 由可逆到互质**：设 $\bar a\bar b=\bar1$，即 $\overline{ab}=\bar1$，即 $ab=1+kn$，即 $ab-kn=1$。设 $d=\gcd(a,n)$，则 $d\mid a$ 且 $d\mid n$，故 $d\mid(ab-kn)=1$，即 $d=1$。

两边合起来即
$$\bar a\in(\mathbb{Z}/n\mathbb{Z})^\times\iff\overline{ab}=\bar1\ \text{有解}\ \bar b\iff ab+nc=1\ \text{有解}\iff\gcd(a,n)=1.$$

（Bézout 正向就是通常的"最大公因子可写成线性组合"；反向就是"公因子整除线性组合"。）

**第三步，$(\mathbb{Z}/n\mathbb{Z})^\times$ 是群**（用子群判别法，把它看成"$\mathbb{Z}/n\mathbb{Z}$ 中乘法封闭子集"）：

1. **非空**：$\bar1\in(\mathbb{Z}/n\mathbb{Z})^\times$（$\bar1\cdot\bar1=\bar1$）。
2. **乘法封闭**：$\gcd(a,n)=\gcd(b,n)=1$ 时，若质数 $p\mid ab$ 且 $p\mid n$，则 $p\mid a$ 或 $p\mid b$，与两者都互质矛盾；故 $\gcd(ab,n)=1$，即 $\overline{ab}\in(\mathbb{Z}/n\mathbb{Z})^\times$。
3. **逆元封闭**：$\gcd(a,n)=1$ 时由 Bézout 得 $ab+nc=1$，对称地也有 $ba+nc=1$，故 $\gcd(b,n)=1$，即 $\bar b$ 也在集合里。
4. **阿贝尔**：$\bar a\bar b=\overline{ab}=\overline{ba}=\bar b\bar a$；单位元 $\bar1$；结合律继承自 $\mathbb{Z}$。

> 第 2、3 步用"互质"表述做起来干净，这是第一步把它刻画出来之后的红利。

**例（$n=12$ 逐个筛）**：$a=0..11$ 中与 12 互质的是 $1,5,7,11$，故 $|(\mathbb{Z}/12\mathbb{Z})^\times|=4$：

| $\times$ | $\bar1$ | $\bar5$ | $\bar7$ | $\bar{11}$ |
|---|---|---|---|---|
| $\bar1$ | $\bar1$ | $\bar5$ | $\bar7$ | $\bar{11}$ |
| $\bar5$ | $\bar5$ | $\bar1$ | $\bar{11}$ | $\bar7$ |
| $\bar7$ | $\bar7$ | $\bar{11}$ | $\bar1$ | $\bar5$ |
| $\bar{11}$ | $\bar{11}$ | $\bar7$ | $\bar5$ | $\bar1$ |

每个非单位元素的阶都是 2（$\bar5^2=\bar1$、$\bar7^2=\bar1$、$\bar{11}^2=\bar1$），故
$$(\mathbb{Z}/12\mathbb{Z})^\times\cong C_2\times C_2\quad(\text{不是 }C_4).$$
这个"要不要 $\times C_2$ 还是 $C_4$"的分辨，正是 [09 有限阿贝尔群分类](./09-isomorphism-theorems-II.md) 要一般化的东西。

**例（质数下必循环）**：$p$ 为质数时 $1,\dots,p-1$ 全部与 $p$ 互质，故 $|(\mathbb{Z}/p\mathbb{Z})^\times|=p-1$。更强的事实（本系列未证）：$(\mathbb{Z}/p\mathbb{Z})^\times$ 是**循环群**——例如 $p=7$ 时 $\bar3$ 是生成元，$\bar3^1..\bar3^6=\bar3,\bar2,\bar6,\bar4,\bar5,\bar1$ 遍历全部 6 个元素，故 $(\mathbb{Z}/7\mathbb{Z})^\times=\langle\bar3\rangle\cong C_6$。（对比上面 $n=12$ 的例子：$12$ 不是质数，$(\mathbb{Z}/12\mathbb{Z})^\times$ 就不是循环的——可见"循环"是质数带来的额外恩惠。）

### 二面体群

**背景**：正 $n$ 边形的全部对称（保持图形的平面等距变换）——$n$ 个旋转 $1,r,\dots,r^{n-1}$（$r$ 为 $2\pi/n$ 逆时针旋转）与 $n$ 个翻转 $m,rm,\dots,r^{n-1}m$（$m$ 为某对称轴翻转）。

**命题 2.17**：这些对称在复合下成群，且满足关系
$$r^n=1,\qquad m^2=1,\qquad mr=r^{-1}m,$$
于是每个元素都可写成 $r^i m^j$（$0\le i<n$，$j\in\{0,1\}$），$|D_n|=2n$：
$$D_n=\langle r,m: r^n=1=m^2,\ mr=r^{-1}m\rangle.$$

**证明分四步。**

**第一步，给对称一个坐标。** 把正 $n$ 边形内接于单位圆，顶点记为
$$w_k=e^{2\pi ik/n}\qquad(k=0,1,\dots,n-1).$$
用复数运算表示变换：逆时针旋转 $2\pi/n$ 是 $r(z)=e^{2\pi i/n}z$，即 $r(w_k)=w_{k+1}$（下标模 $n$）；关于实轴翻转是 $m(z)=\bar z$（复共轭），即 $m(w_k)=w_{-k}$。先记住这两条基本作用：
$$r(w_k)=w_{k+1},\qquad m(w_k)=w_{-k}.$$
（下面所有逐点验证都只是这两式的迭代。）

**第二步，$|D_n|\le2n$：任何对称由 $w_0$ 与 $w_1$ 的像唯一决定。**

设 $H$ 是任意一个保持正 $n$ 边形的平面等距变换。它把顶点集合送到自身，特别地 $H(w_0)$ 必是某个 $w_k$（$k=0,\dots,n-1$）；于是 $H(w_0)$ 有 $n$ 种可能。

- **关键观察**：正 $n$ 边形的**全部点**（不只是顶点）都由 $w_0,w_1,\dots,w_{n-1}$ 决定，而等距变换由两个不同点的像唯一决定……但两点还不够（平面上两点之像只钉住一个"二值"自由度）。这里用更精确的论证：

  等距变换由**三个不共线点的像**唯一决定，而"哪些像是允许的"由**是否保持定向**分两类。设 $H(w_0)=w_k$。因为 $H$ 是等距变换，它保持距离，特别地
  $$d(H(w_0),H(w_1))=d(w_0,w_1)=|1-e^{2\pi i/n}|,$$
  而在正 $n$ 边形上，与 $w_k$ 距离等于 $d(w_0,w_1)$ 的顶点**恰有两个**：$w_{k+1}$ 与 $w_{k-1}$（相邻的两个）。故
  $$H(w_1)=w_{k+1}\quad\text{或}\quad H(w_1)=w_{k-1}.$$
  二选一，**穷尽了所有可能**。（这一步正是原文的"$w_1$ 只能去 $w_{k+1}$ 或 $w_{k-1}$"。）

- $H(w_0)=w_k$ 有 $n$ 种，$H(w_1)$ 二选一，故候选不超过 $2n$ 个，即 $|D_n|\le2n$。

**第三步，两类各由 $r,m$ 生成，且 $|D_n|\ge2n$。** 注意 $r,m$ 本身都是对称，故都在 $D_n$ 中。

- **第一类（$w_1\mapsto w_{k+1}$，保定向）**：$r^k(w_0)=w_k$ 且 $r^k(w_1)=w_{k+1}$，两个点上的作用与 $H$ 一致。两者都保持正 $n$ 边形状（都是保定向等距），而**保定向等距由一点及其像唯一决定**（因为保定向即旋转 + 平移，在平面里由一对点唯一确定），故 $H=r^k$。
- **第二类（$w_1\mapsto w_{k-1}$，反定向）**：看复合 $r^km$：由第一步
  $$(r^km)(w_0)=r^k(w_0)=w_k,\qquad(r^km)(w_1)=r^k(w_{-1})=w_{k-1}.$$
  两个点上的作用都与 $H$ 一致，同理（反定向等距由一对点唯一决定）$H=r^km$。

于是 $D_n=\{1,r,\dots,r^{n-1}\}\cup\{m,rm,\dots,r^{n-1}m\}$，正好 $2n$ 个；结合第二步 $|D_n|\le2n$ 得**恰有** $|D_n|=2n$，且每个元素都写成 $r^im^j$ 的形式（$0\le i<n$，$j\in\{0,1\}$）。

**第四步，三条关系逐点验证。**

- $r^n=1$：$r^n(w_k)=w_{k+n}=w_k$（下标模 $n$），对一切 $k$ 成立，故 $r^n$ 是恒等变换。
- $m^2=1$：$m^2(w_k)=m(w_{-k})=w_{k}$，故 $m^2=1$。
- $mr=r^{-1}m$——用复数算最干净。对 $z=e^{2\pi ik/n}$：
  $$(mr)(z)=m\bigl(e^{2\pi i/n}z\bigr)=\overline{e^{2\pi i/n}z}=e^{-2\pi i/n}\bar z,$$
  而
  $$(r^{-1}m)(z)=r^{-1}(\bar z)=e^{-2\pi i/n}\bar z.$$
  两者相等，故 $mr=r^{-1}m$。$\blacksquare$

> **由 $mr=r^{-1}m$ 能机械化化简一切**：反复把 $m$ 往左推，每穿过一个 $r$ 就翻一次符号：
> $$r^im\cdot r^jm=r^i(mr^j)m=r^i(r^{-j}m)m=r^{i-j}m^2=r^{i-j}.$$
> 于是 $D_n$ 中任意两个元素相乘都能算成 $r^\ast m^\ast$ 的标准型——这就是"生成元 + 关系"表示法的全部威力。**例**（$n=6$）：$r^4m\cdot r^5m=r^{-1}=r^5$。

> **为什么这是非阿贝尔群的第一课**：$n\ge3$ 时 $mr=r^{-1}m\ne rm$。**验证**：$n\ge3$ 时 $r^{-1}\ne r$（否则 $r^2=1$，但 $r$ 是 $2\pi/n$ 旋转，只有当 $n=1,2$ 才有 $r^2=1$）；求逆得 $rm=r^{-1}m\cdot r^{-1}\cdot r$ 不成立也可见两者不同。具体地 $n=3$ 时用 $S_3$ 的置换记号 $r=(1,2,3)$、$m=(2,3)$：
> $$mr=(2,3)(1,2,3)=(1,3),\qquad rm=(1,2,3)(2,3)=(1,2),$$
> $(1,3)\ne(1,2)$，确认非交换。翻转与旋转"不能交换，但交换有代价公式"。

**例（$D_3\cong S_3$）**：$|D_3|=6$，且把三个顶点重新标号后，$D_3$ 的每个对称就是 $\{1,2,3\}$ 的一个排列，这样得到的映射 $D_3\to S_3$ 是双射同态，故 $D_3\cong S_3$。$r=(1,2,3)$ 是 3-cycle，$m=(2,3)$ 是对换。这个同构是"几何对称 = 置换"的第一次现身，[05 阶 6 分类](./05-group-properties-II.md) 会独立地把 $D_3$ 认成 $S_3$。

**例（$D_4$，正方形的 8 个对称，逐个列出）**：取顶点 $1,2,3,4$ 顺时针排列，$r=(1,2,3,4)$ 是转 $90°$，$m=(2,4)$ 是沿过顶点 1、3 的对角线翻转。则
$$D_4=\{e,\ r,\ r^2,\ r^3,\ m,\ rm,\ r^2m,\ r^3m\}$$
上的元素逐条读作置换：

| 元素 | 作为置换 | 几何含义 |
|---|---|---|
| $e$ | $(1)(2)(3)(4)$ | 不动 |
| $r$ | $(1,2,3,4)$ | 逆时针 $90°$ |
| $r^2$ | $(1,3)(2,4)$ | 转 $180°$（点对称） |
| $r^3$ | $(1,4,3,2)$ | 逆时针 $270°$ |
| $m$ | $(2,4)$ | 沿对角 1–3 翻转 |
| $rm$ | $(1,2)(3,4)$ | 沿水平中轴翻转 |
| $r^2m$ | $(1,3)$ | 沿对角 2–4 翻转 |
| $r^3m$ | $(1,4)(2,3)$ | 沿竖直中轴翻转 |

这给出 $D_4\le S_4$ 的一个具体嵌入。顺带可见 $D_4\not\cong S_4$（$|D_4|=8\ne24=|S_4|$），且 $D_4$ 的元素阶只有 $1,2,4$。（阶 8 的另一类群里 $Q_8$ 的元素阶也为 $1,2,4$，故这两者不能靠阶谱区分，得另想不变量——见 [09 阶 8 分类](./09-isomorphism-theorems-II.md)）。

## 为什么重要

- $\mathbb{Z}/n\mathbb{Z}$ 是**良定义性检查**的标准训练场——后续商群（[06](./06-normal-subgroup-and-quotient-group.md)）、同构定理（[08](./08-isomorphism-theorems-I.md)）里每个"在等价类上定义的映射"都要过这一关
- $(\mathbb{Z}/n\mathbb{Z})^\times$ 把数论（gcd、Bézout）变成群论对象，后面 $Aut(C_n)\cong(\mathbb{Z}/n\mathbb{Z})^\times$（[07](./07-automorphism-and-products.md)）把两者接通
- $D_n$ 是最小的非阿贝尔群家族，也是[魔方群](../rubiks-cube-group-theory.md)"转动生成对称"的最简模型

## 局限 / 边界

- 原文证明 $D_n$ 元素个数时默认"对称由顶点像决定"——这依赖平面等距变换的分类（保定向 = 旋转+平移、反定向 = 反射+平移，由一对点唯一决定）。上面的证明里把这一步显式拆成了"两点定 $H$"，比原文的骨架更紧，但**没有从零推导欧氏等距分类**——那是几何侧的事，群论这边只需承认"图形对称 = 等距变换"
- $n\ge 3$ 的约束是必要的：$n=2$ 时正二边形退化，$r$ 与 $m$ 的关系不再是 $mr=r^{-1}m$ 而出现 $r=r^{-1}$，整个 $D_2$ 塌缩成 $C_2\times C_2$，讨论失去一般性
- $(\mathbb{Z}/n\mathbb{Z})^\times$ 为循环只在 $n=1,2,4,p^k,2p^k$（$p$ 奇质数）时成立；这是数论结论（原根存在定理），本系列未展开，[07 $Aut(C_n)$](./07-automorphism-and-products.md) 会用到"$Aut$ 的大小"但不依赖循环性

## 我的理解

- "良定义性"第一印象是形式主义，其实是**商结构可用性的全部门槛**：$\mathbb{Z}/n\mathbb{Z}$ 里 $\times$ 对全体类不构成群（$\bar 0$ 拖后腿），必须筛出可逆类——"结构保留多少，取决于等价关系多粗"
- $mr=r^{-1}m$ 说明非阿贝尔不等于"无序"：交换的**代价是可计算的**（$r$ 变 $r^{-1}$）。魔方里 $F R\ne R F$ 但 $FR=RF^{-1}$ 型的代价公式同样存在
- 加法与乘法的对比很值得记：两者都是"良定义 + 下沉"，但乘法的良定义多了一个交叉项、且必须**筛掉不可逆元**才能成群。这说明"同一个集合 + 不同运算"可以给出完全不同的群结构——$\mathbb{Z}/n\mathbb{Z}$ 在加法下是 $C_n$，在乘法下连群都不是（全体），在乘法下取可逆部分才是 $(\mathbb{Z}/n\mathbb{Z})^\times$
- $D_n$ 的"两类各 $n$ 个"结构（$n$ 个旋转 + $n$ 个翻转）是**半直积**的原型：$D_n=C_n\rtimes C_2$（[07](./07-automorphism-and-products.md) 例 4.31 旁的讨论）。翻转 $m$ 通过 $r\mapsto r^{-1}$ 这个自同构去扭旋转——$mr=r^{-1}m$ 就是这个自同构的方程式

## Related

- [01 群论导论](./01-introduction-to-group-theory.md) — 子群判定引理
- [03 对称群与循环群](./03-symmetric-and-cyclic-groups.md) — $\langle r\rangle=\{1,r,\dots,r^{n-1}\}$ 展开
- [07 自同构、直积与半直积](./07-automorphism-and-products.md) — $Aut(D_3)\cong D_3$
- [魔方群的数学结构](../rubiks-cube-group-theory.md) — 更大的"对称生成群"实例
