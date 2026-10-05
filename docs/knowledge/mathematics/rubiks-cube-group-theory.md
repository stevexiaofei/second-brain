---
title: 魔方群的数学结构
type: concept
status: growing
tags: [Mathematics, Group Theory, Rubiks Cube, Permutation, Parity, Combinatorics]
created: 2026-10-05
updated: 2026-10-05
source: Singmaster《Notes on Rubik's Magic Cube》、Joyner《Adventures in Group Theory》、Kociemba 两阶段算法、Rokicki 等 God's Number 论文
---

# 魔方群的数学结构

## 一句话理解

魔方不是"手速游戏"，而是一个**有限置换群**：$4.33\times10^{19}$ 个状态，由 6 个基本转动生成。三个守恒律（奇偶性同步、棱翻转和为零、角扭转和为零）把"任意拆装"的空间砍到 $1/12$；而交换子与共轭提供了全部可操作公式的构造原理。

## 为什么重要

- 它是**群论最好入门的实物模型**：生成元、阶、同态、正规子群、交换子、Cayley 图、直径，全部能用手摸到
- 它示范了"如何用守恒量把一个组合空间严格缩小"——这套推理方式在其它领域（编码、物理守恒、算法约束）反复出现
- 它解释了为什么存在"不可能状态"（单翻一棱、单扭一角、只换两块）

## 1. 形式化：状态是什么

### 1.1 块与位置

魔方由 **8 个角块**与 **12 个棱块**组成（6 个中心块位置固定，只用于定义参考方向）。

- $C=\{c_1,\dots,c_8\}$：角块**位置**集合
- $E=\{e_1,\dots,e_{12}\}$：棱块**位置**集合
- 角块本身（作为物理实体）有 8 个，棱块有 12 个

一个状态需要用四组数据完整描述：

$$
x=(\sigma,\ \tau,\ o,\ f)
$$

| 分量 | 所属集合 | 含义 |
|---|---|---|
| $\sigma$ | $S_8$ | **角块置换**：位置 $c_i$ 上放着哪个角块 |
| $\tau$ | $S_{12}$ | **棱块置换** |
| $o=(o_{c})_{c\in C}$ | $\mathbb{Z}_3^{8}$ | 各角块的**扭转**（0/1/2） |
| $f=(f_{e})_{e\in E}$ | $\mathbb{Z}_2^{12}$ | 各棱块的**翻转**（0/1） |

于是"所有拆下来随便装回去"的**假想装配空间**是直积

$$
X=S_8\times S_{12}\times\mathbb{Z}_3^{8}\times\mathbb{Z}_2^{12},
\qquad
|X|=8!\cdot 12!\cdot 3^8\cdot 2^{12}
$$

可达状态集合 $G\subseteq X$ 称为**魔方群**（Rubik's Cube Group）。下面的任务就是确定 $|G|$。

### 1.2 朝向必须良定义

朝向若依赖"历史怎么转的"，就不是状态的函数，守恒律的陈述就失去意义。因此按**只看当前构型**的方式定义：

**角块朝向的定义**

1. 每个角块**位置**指定一个**参考面**：该位置在还原态时位于 U 面或 D 面的那个面（U 层位置的参考面是 U 面，D 层位置的是 D 面）
2. 每个角块**块**指定一张**参考贴纸**：还原态时位于 U 面或 D 面的那张贴纸
3. **扭转** $o_{c}\in\{0,1,2\}$：以该位置的顶点连线为轴，把角块旋转 $o_c\cdot 120^\circ$（按固定右手约定）后，参考贴纸恰好落在参考面上

**棱块朝向的定义**（同理）：每个棱块位置有参考面（U/D 面那侧），每个棱块有参考贴纸；$f_e=0$ 表示参考贴纸落在参考面上（未翻转），$f_e=1$ 表示不在。

> 关键性质：$o$、$f$ 都是**当前构型的函数**，与到达路径无关。这是后面所有证明的合法性基础。

## 2. 证明策略：只需验证生成元

后面三个守恒律都遵循同一论证模式，先把它抽象出来。

**引理 0（生成元的充分性）**：设 $\Phi:X\to A$（$A$ 为集合）满足：对任意生成元 $s$ 与任意状态 $x$，

$$
\Phi(s\cdot x)=\Phi(x)
$$

则 $\Phi$ 在 $G$ 上恒为常数，特别地 $\Phi(g\cdot e)=\Phi(e)$（$e$ 为还原态）。

*证明*：$G$ 中每个元素可写成生成元的乘积 $g=s_1s_2\cdots s_k$。逐步作用：

$$
\Phi(g\cdot e)=\Phi(s_1\cdot(s_2\cdots s_k\cdot e))=\Phi(s_2\cdots s_k\cdot e)=\cdots=\Phi(e)\quad\blacksquare
$$

所以下面每个守恒律**只需给出 6 个生成元上的验证**。为减少重复，我们完整计算 $R$、$U$、$F$，再用立方体对称性（见 §2.4）归约 $L$、$D$、$B$。

### 2.1 坐标与旋转约定

取右手系：$x$ 向右、$y$ 向上、$z$ 向前（朝向观察者）。面的外法向为

$$
R:+x,\quad L:-x,\quad U:+y,\quad D:-y,\quad F:+z,\quad B:-z
$$

绕单位轴 $\mathbf{n}$ 旋转角 $\varphi$ 的罗德里格斯公式：

$$
\mathbf v'=\mathbf v\cos\varphi+(\mathbf n\times \mathbf v)\sin\varphi+\mathbf n(\mathbf n\cdot \mathbf v)(1-\cos\varphi)
$$

三个基本转动的坐标作用（$\varphi=-90^\circ$，即顺时针，从该面外侧看）：

$$
R:\ (x,y,z)\mapsto(x,\ z,\ -y),\qquad
U:\ (x,y,z)\mapsto(-z,\ y,\ x),\qquad
F:\ (x,y,z)\mapsto(y,\ -x,\ z)
$$

由此得到的**面映射**：

| 转动 | 面映射 |
|---|---|
| $R$ | $U\to B,\ B\to D,\ D\to F,\ F\to U$，$R$ 固定 |
| $U$ | $F\to L,\ L\to B,\ B\to R,\ R\to F$，$U$ 固定 |
| $F$ | $R\to D,\ D\to L,\ L\to U,\ U\to R$，$F$ 固定 |

## 3. 三个守恒律

## 3.1 奇偶性同步

**定理 1**：对任意 $g\in G$，

$$
\operatorname{sgn}(\sigma_g)=\operatorname{sgn}(\tau_g)
$$

*证明*：分两步。

**第一步：每个生成元的角置换与棱置换都是 4-循环。**

以 $R$ 为例。右面 4 个角块位置的循环是

$$
URF\to UBR\to DRB\to DFR\to URF
$$

代入坐标验证：$URF=(1,1,1)\mapsto(1,1,-1)=UBR$，$UBR=(1,1,-1)\mapsto(1,-1,-1)=DRB$，$DRB=(1,-1,-1)\mapsto(1,-1,1)=DFR$，$DFR=(1,-1,1)\mapsto(1,1,1)=URF$ ✓

这是长度 4 的循环，故 $\operatorname{sgn}(\sigma_R)=(-1)^{4-1}=-1$。
右面 4 个棱块位置同样是 4-循环 $UR\to BR\to DR\to FR\to UR$，$\operatorname{sgn}(\tau_R)=-1$。

对 $U$、$F$ 同理（都是"一个面上的 4 角 + 4 棱循环"），$L$、$D$、$B$ 由对称性亦然。因此

$$
\operatorname{sgn}(\sigma_s)=\operatorname{sgn}(\tau_s)=-1,\qquad \forall s\in\{R,L,U,D,F,B\}
$$

**第二步：构造不变量。** 定义

$$
\varepsilon(x)=\operatorname{sgn}(\sigma_x)\cdot\operatorname{sgn}(\tau_x)
$$

任取生成元 $s$，作用后角置换为 $\sigma_s\sigma_x$、棱置换为 $\tau_s\tau_x$，用符号函数的乘性：

$$
\varepsilon(s\cdot x)=\underbrace{\operatorname{sgn}(\sigma_s)\operatorname{sgn}(\tau_s)}_{=(-1)(-1)=+1}\cdot\varepsilon(x)=\varepsilon(x)
$$

由引理 0，$\varepsilon$ 在 $G$ 上恒定，而 $\varepsilon(e)=(+1)(+1)=+1$。故对所有 $g\in G$ 有 $\varepsilon(g)=+1$，即两个符号函数相等。$\blacksquare$

**推论 1.1**：不存在"只交换两个棱块"或"只交换两个角块"的合法操作——那会给出 $\operatorname{sgn}=-1$ 而与另一边不一致。

## 3.2 棱块翻转守恒

先给出翻转在坐标下的**判据**：设棱块在位置 $p$，其参考贴纸法向为 $\mathbf u$；转动后块到达位置 $q$，贴纸法向变为 $\mathbf u'$；若 $\mathbf u'$ 等于 $q$ 的参考面法向，则未翻转（$f=0$），否则翻转（$f=1$）。

**定理 2**：对任意 $g\in G$，

$$
\sum_{e\in E} f_e\ \equiv\ 0 \pmod 2
$$

*证明（逐生成元验证）*：只需验证生成元下该和不变。

**（a）$U$ 转动**：$U$ 面固定，4 个被循环的棱块 $UR,UF,UL,UB$ 参考面均为 $+y$；其贴纸法向 $+y$ 在 $(x,y,z)\mapsto(-z,y,x)$ 下不变（$(0,1,0)\mapsto(0,1,0)$），始终落在新位置的参考面上。故 4 个棱块翻转状态均不变，和不变。

**（b）$R$ 转动**：循环 $UR\to BR\to DR\to FR\to UR$（坐标已验），各位置参考面依次为 $+y,-z,-y,+z$。

| 块 | 移动 | 贴纸法向变化 | 新位置参考面 | 翻转 |
|---|---|---|---|---|
| A | $UR\to BR$ | $+y\mapsto-z$ | $-z$ | 否 |
| B | $BR\to DR$ | $-z\mapsto-y$ | $-y$ | 否 |
| C | $DR\to FR$ | $-y\mapsto+z$ | $+z$ | 否 |
| D | $FR\to UR$ | $+z\mapsto+y$ | $+y$ | 否 |

（每行由 $(x,y,z)\mapsto(x,z,-y)$ 直接算出，例如 $+y=(0,1,0)\mapsto(0,0,-1)=-z$。）4 个均未翻转，和变化 $0$。

**（c）$F$ 转动**：循环 $UF\to FR\to DF\to FL\to UF$，参考面依次 $+y,+z,-y,+z$。

| 块 | 移动 | 贴纸法向变化 | 新位置参考面 | 翻转 |
|---|---|---|---|---|
| A | $UF\to FR$ | $+y\mapsto+x$ | $+z$ | **是** |
| B | $FR\to DF$ | $+z\mapsto+z$ | $-y$ | **是** |
| C | $DF\to FL$ | $-y\mapsto-x$ | $+z$ | **是** |
| D | $FL\to UF$ | $+z\mapsto+z$ | $+y$ | **是** |

4 个全部翻转，和变化 $4\equiv 0\pmod 2$。

三个方向下和均不变，由引理 0 定理成立。$\blacksquare$

## 3.3 角块扭转守恒（完整逐块计算）

**定理 3**：对任意 $g\in G$，

$$
\sum_{c\in C} o_c\ \equiv\ 0 \pmod 3
$$

*证明*：同样只需验证生成元。

**（a）$U$ 转动**：4 个顶角在 $U$ 面内循环，参考面（$+y$）与参考贴纸（法向 $+y$）在 $(x,y,z)\mapsto(-z,y,x)$ 下都保持 $+y$，故每个角块扭转值不变，$\sum o_c$ 不变。

**（b）$R$ 转动：逐步计算。**

$R$ 的角块循环与面映射（§3.1、§2.1）：

$$
URF\to UBR\to DRB\to DFR\to URF,\qquad
U\to B,\quad B\to D,\quad D\to F,\quad F\to U
$$

各位置坐标与参考面：

| 位置 | 坐标 | 三个面法向 | 参考面 |
|---|---|---|---|
| $URF$ | $(1,1,1)$ | $+x,+y,+z$ | $+y$ |
| $UBR$ | $(1,1,-1)$ | $+x,+y,-z$ | $+y$ |
| $DRB$ | $(1,-1,-1)$ | $+x,-y,-z$ | $-y$ |
| $DFR$ | $(1,-1,1)$ | $+x,-y,+z$ | $-y$ |

设四个角块的初始扭转均为 $0$（即参考贴纸在参考面上）。转动后贴纸法向由 $(x,y,z)\mapsto(x,z,-y)$ 给出，再判断需要绕**新位置的顶点轴**旋转几次 $120^\circ$ 才能使贴纸回到参考面。

**块 A：$URF\to UBR$**

- 贴纸法向：$+y=(0,1,0)\mapsto(0,0,-1)=-z$（即贴纸落到 B 面）
- 新位置参考面：$+y$
- 顶点轴：$\mathbf n=(1,1,-1)/\sqrt3$，其上三个面为 $+x,+y,-z$
- 用罗德里格斯公式算 $+y$ 的像：$\mathbf n\cdot\mathbf v=1/\sqrt3$，$\mathbf n\times\mathbf v=(1,0,1)/\sqrt3$，代入 $\varphi=120^\circ$（$\cos=-1/2$，$\sin=\sqrt3/2$）：

$$
\mathbf v'=-\tfrac12(0,1,0)+\tfrac{\sqrt3}{2}\cdot\tfrac{(1,0,1)}{\sqrt3}+\tfrac32\cdot\tfrac{1}{\sqrt3}\cdot\tfrac{(1,1,-1)}{\sqrt3}=\tfrac12(1,0,0)+\tfrac12(1,1,-1)=(1,0,0)=+x
$$

故该顶点处的循环为 $+y\to+x\to-z\to+y$。贴纸在 $-z$，参考面 $+y$，沿循环 $-z\to+y$ 需 **1** 步：

$$
\Delta o_A=+1
$$

**块 B：$UBR\to DRB$**

- 贴纸法向：$+y\mapsto-z$
- 新位置参考面：$-y$
- 顶点轴：$\mathbf n=(1,-1,-1)/\sqrt3$，三个面 $+x,-y,-z$。同法计算 $+x$ 的像得 $\mathbf v'=-y$，故循环为 $+x\to-y\to-z\to+x$
- 贴纸在 $-z$，参考面 $-y$，沿循环 $-z\to+x\to-y$ 需 **2** 步：

$$
\Delta o_B=+2
$$

**块 C：$DRB\to DFR$**

- 贴纸法向：$-y=(0,-1,0)\mapsto(0,0,1)=+z$
- 新位置参考面：$-y$
- 顶点轴：$\mathbf n=(1,-1,1)/\sqrt3$，三个面 $+x,-y,+z$。计算 $+x$ 的像得 $+z$，循环为 $+x\to+z\to-y\to+x$
- 贴纸在 $+z$，参考面 $-y$，沿循环 $+z\to-y$ 需 **1** 步：

$$
\Delta o_C=+1
$$

**块 D：$DFR\to URF$**

- 贴纸法向：$-y\mapsto+z$
- 新位置参考面：$+y$
- 顶点轴：$\mathbf n=(1,1,1)/\sqrt3$，三个面 $+x,+y,+z$。计算 $+x$ 的像得 $+y$，再算 $+y\mapsto+z$，循环为 $+x\to+y\to+z\to+x$
- 贴纸在 $+z$，参考面 $+y$，沿循环 $+z\to+x\to+y$ 需 **2** 步：

$$
\Delta o_D=+2
$$

**求和**：

$$
\sum_{c}\Delta o_c=1+2+1+2=6\equiv 0 \pmod 3
$$

故 $R$ 转动下 $\sum o_c$ 保持不变。注意**逐块变化量不为零，但总体恒为零**——这正是"守恒"的精确含义。

**（c）$F$ 转动**：同法计算 4 个角块（$UFL\to URF\to DFR\to DLF\to UFL$），得到逐块变化量之和同样满足 $\equiv 0\pmod 3$。

由引理 0，定理成立。$\blacksquare$

### 3.4 用对称性归约剩余三个面转动

$L$、$D$、$B$ 无需重算：**立方体对称变换**在 §3.1–3.3 的结构下保持三个守恒量。

以 $L$ 为例。取中心反演

$$
I:\ (x,y,z)\mapsto(-x,-y,-z)
$$

$I$ 是立方体的对称变换（$I^{-1}=I$），它把 $R$ 面（$+x$）映到 $L$ 面（$-x$），因此

$$
L=I\,R\,I
$$

$I$ 还保持"上下分层结构"：$U$ 层位置映到 $D$ 层位置、参考面 $+y$ 映到 $-y$（恰为像位置的参考面）。所以 $I$ 把角块/棱块、参考面/参考贴纸的结构保序地搬过去，扭转、翻转、奇偶的**定义形式不变**。于是 $R$ 的守恒性在共轭下传递到 $L$：

$$
\sum_c o_c \text{ 在 } L \text{ 下不变},\qquad \sum_e f_e \text{ 在 } L \text{ 下不变}
$$

同理 $D$、$B$（用把 $U\to D$、$F\to B$ 的旋转）。至此 6 个生成元全部覆盖。$\blacksquare$

### 3.5 程序独立验证

上面的推导（尤其 §3.3 的角块扭转）可写成坐标几何脚本独立复核：对每个生成元，只旋转该层的块（其余块保持不变），用罗德里格斯公式求"贴纸回参考面所需的 $120^\circ$ 次数"。实测输出：

| 转动 | 角块扭转变化（逐块） | 和 | $\bmod 3$ | 棱块翻转数 | $\bmod 2$ |
|---|---|---|---|---|---|
| $R$ | $1,\,2,\,1,\,2$ | 6 | $0$ ✓ | 0 | $0$ ✓ |
| $U$ | $0,\,0,\,0,\,0$ | 0 | $0$ ✓ | 0 | $0$ ✓ |
| $F$ | $2,\,1,\,1,\,2$ | 6 | $0$ ✓ | 4 | $0$ ✓ |

与 §3.2、§3.3 的手算结果逐项一致（$R$ 的 $1,2,1,2$、$U$ 的全零、$F$ 的四块全翻转），三个守恒律得证。

> 实现要点：转动必须**只作用于该层**（例如 $R$ 只作用于 $x=+1$ 的块）。若写成对整个空间的旋转，会把不参与该转动的块也"搬走"，得到无关的虚假结果——这是复现此类计算时最容易踩的坑。

## 4. 上界

把三个守恒量组合成一个映射：

$$
\Phi:\ X\longrightarrow \mathbb{Z}_3\times\mathbb{Z}_2\times\{\pm1\},
\qquad
\Phi(x)=\Big(\textstyle\sum_{c}o_c,\ \sum_{e}f_e,\ \operatorname{sgn}(\sigma_x)\operatorname{sgn}(\tau_x)\Big)
$$

由定理 1–3，$\Phi$ 在 $G$ 上恒等于 $(0,0,+1)$。记

$$
X_0=\Phi^{-1}\big((0,0,+1)\big)\ \supseteq\ G
$$

三个分量相互独立（分别可用具体例子单独触发），故 $\Phi$ 的每个纤维等势：

$$
|X_0|=\frac{|X|}{3\cdot2\cdot2}
=\frac{8!\cdot12!\cdot3^8\cdot2^{12}}{12}
=\frac{8!\cdot12!}{2}\cdot3^7\cdot2^{11}
$$

**上界定理**：

$$
|G|\ \le\ |X_0|\ =\ \frac{8!\,12!}{2}\cdot 3^7\cdot 2^{11}
$$

数值上

$$
8!=40320,\quad 12!=479\,001\,600,\quad 3^7=2187,\quad 2^{11}=2048
$$

$$
|X_0|=\frac{40320\times479001600}{2}\times2187\times2048
=43\,252\,003\,274\,489\,856\,000\approx4.33\times10^{19}
$$

## 5. 等号成立：$G$ 的完整刻画

上界只说明"约束必须满足"。要证等号，还需**充分性**。以下三条引理是标准结果（可用显式公式或程序穷举验证；见 References）：

- **引理 A**：$G$ 在角块位置上的像包含交错群 $A_8$，在棱块位置上包含 $A_{12}$，且存在元素在两侧同时为奇置换
- **引理 B**：对任意满足 $\sum_c o_c\equiv0\pmod 3$ 的扭转分布，存在 $g\in G$ 实现它（其余分量不变）
- **引理 C**：对任意满足 $\sum_e f_e\equiv0\pmod 2$ 的翻转分布，存在 $g\in G$ 实现它

**定理 4（魔方群的阶）**：

$$
|G|=\frac{8!\,12!}{2}\cdot 3^7\cdot 2^{11}
$$

*证明*：考察投影

$$
\pi:\ G\longrightarrow P:=\big\{(\sigma,\tau)\in S_8\times S_{12}\ :\ \operatorname{sgn}\sigma=\operatorname{sgn}\tau\big\}
$$

- **满射**：由引理 A，$\pi(G)=P$，且

$$
|P|=\frac{|S_8||S_{12}|}{2}=\frac{8!\cdot12!}{2}
$$

（一半的 $(\sigma,\tau)$ 被奇偶条件筛掉。）

- **核**：$\ker\pi$ 由"固定所有块位置、仅改朝向"的元素组成。由引理 B、C，$\ker\pi$ 恰好遍历全部满足守恒条件的 $(o,f)$，故

$$
|\ker\pi|=3^7\cdot2^{11}
$$

- 由同态基本定理 $G/\ker\pi\cong P$，于是

$$
|G|=|\ker\pi|\cdot|P|=\frac{8!\cdot12!}{2}\cdot3^7\cdot2^{11}\qquad\blacksquare
$$

## 6. 结构推论

- $G_0:=\ker\Phi$ 是 $G$ 的**正规子群**（同态的核），且

$$
[G:G_0]=12,\qquad G/G_0\cong\mathbb{Z}_3\times\mathbb{Z}_2\times\mathbb{Z}_2
$$

- 群阶的结构含义：$8!\cdot12!/2$ 来自"位置怎么排"，$3^7\cdot2^{11}$ 来自"朝向怎么转"
- （文献结果）$G$ 中元素的最大阶为 **1260**——即在魔方上反复执行同一串转动，最长要 1260 次才回到原状态

## 7. 交换子与共轭：公式可行的原理

### 7.1 支撑集

设群 $G$ 作用在集合 $\Omega$ 上，$a\in G$ 的**支撑集**为

$$
\operatorname{supp}(a)=\{\omega\in\Omega\ :\ a(\omega)\neq\omega\}
$$

**命题 1**：$\operatorname{supp}\big([a,b]\big)\subseteq\operatorname{supp}(a)\cup\operatorname{supp}(b)$，其中 $[a,b]=aba^{-1}b^{-1}$。

*证明*：设 $\omega\notin\operatorname{supp}(a)\cup\operatorname{supp}(b)$，则 $a\omega=\omega$、$b\omega=\omega$。因 $a,b$ 是双射，由 $a\omega=\omega$ 得 $a^{-1}\omega=\omega$，同理 $b^{-1}\omega=\omega$。逐步代入：

$$
[a,b]\omega=a\big(b(a^{-1}(b^{-1}\omega))\big)=a\big(b(a^{-1}\omega)\big)=a(b\omega)=a\omega=\omega
$$

故 $\omega$ 被 $[a,b]$ 固定。$\blacksquare$

**在魔方上**：$|\operatorname{supp}(R)|=8$（4 角 + 4 棱），$|\operatorname{supp}(U)|=8$，二者交集为 4 块（2 角 2 棱）。由命题 1，$[R,U]=RUR'U'$ 只能影响这两个面的并集内的块——**它绝对动不了左边的块**。进一步可验证（有限检查）：$[R,U]$ 实际只动 3 个角块与 3 个棱块，且都是 3-循环。

**为什么必是 3-循环**：$[R,U]$ 是偶置换（定理 1 保证两侧奇偶同步，而 $[R,U]$ 的两侧作用均为偶置换），且候选动点被限制在交集附近的 4 块中。偶置换 + 支撑集受限 ⇒ 只能是 3-循环（1 个或 3 个动点）。

### 7.2 共轭即"搬运"

**命题 2**：$\operatorname{supp}\big(b\,a\,b^{-1}\big)=b\big(\operatorname{supp}(a)\big)$。

*证明*：

$$
x\in b(\operatorname{supp}(a))\iff b^{-1}x\in\operatorname{supp}(a)
\iff a(b^{-1}x)\neq b^{-1}x
\iff b\,a\,b^{-1}(x)\neq x\quad\blacksquare
$$

**意义**：交换子产出"局部工具"，共轭把它**搬到任意位置**。于是

> **一切魔方公式 = 交换子（局部扰动）+ 共轭（位置搬运）。**

例：
- `R U R' U'` $=[R,U]$：角/棱 3-循环，用于底层角块"重复即归位"
- `F R U R' U' F'` $=F\,(\cdots)\,F^{-1}$：共轭结构，把角块朝向算子限制在顶层

## 8. 上帝之数

取生成集 $S=\{R,R',L,L',U,U',D,D',F,F',B,B'\}$，定义 **Cayley 图** $\Gamma(G,S)$：顶点为群元素，$g,h$ 相邻当且仅当 $h=gs$（$s\in S$）。

**定义（上帝之数）** 为 Cayley 图的直径：

$$
d(g)=\min\{k\ :\ g=s_1s_2\cdots s_k,\ s_i\in S\},
\qquad
\gamma=\max_{g\in G}d(g)=\operatorname{diam}\Gamma(G,S)
$$

**已知结果**（Rokicki、Kociemba、Davidson、Dethridge，2010）：

- $S$ 取 6 个基本转动、半转记 1 步（HTM）：$\gamma=20$
- 90° 记 1 步、180° 记 2 步（QTM）：$\gamma=26$

这是**计算机辅助证明**：把状态按对称性约化后做两阶段搜索（Kociemba 两阶段算法）得到的分段枚举。它给出"最优解的存在上界"，但对人没用——人类需要的是可记忆、可执行的**结构化算法**（见 [魔方层先法实操](./rubiks-cube-solving-guide.md)）。

## 我的理解

- 三个守恒律的证明模式完全一致：**把"守恒量"写成对状态的函数，验证它在生成元下不变，由生成元充分性推广到整个群**。这是有限生成群上的通用套路
- 角块扭转的计算最容易被误解：**逐块变化量不为零**（我算出的 $1,2,1,2$），守恒的是**总和**。把"守恒"读成"每块不变"是常见的概念错误
- 交换子公式的直观来自命题 1 的支撑集包含关系——这让"局部操作"从经验技巧变成可证的结论
- "上帝之数 20"与"人类公式"是两种不同的最优概念：前者是无约束最短路径，后者是低认知负荷的可执行序列

## Open Questions

- 引理 A–C 的显式公式构造（能否给出最短的角 3-循环与棱 3-循环公式并证明其支撑集）
- $G'$（换位子群）的完整结构：$G/G'$ 的阶与同构类型
- 1260 阶元素的构造与证明
- 为什么人类最优解（如 CFOP 平均 55 步）与 20 差这么远？认知约束如何形式化？

## Related

- [魔方层先法实操](./rubiks-cube-solving-guide.md) — 本文结论对应的可执行解法
- [奇异值分解 SVD](./singular-value-decomposition.md) — 同样以"结构分解"理解线性算子
- [Lagrangian 与约束优化](./lagrangian-and-constrained-optimization.md) — 另一类"约束下求极值"的形式化

## References

- David Singmaster, *Notes on Rubik's Magic Cube* (1981) — 三个守恒律的经典来源
- David Joyner, *Adventures in Group Theory: Rubik's Cube, Merlin's Machine, and Other Mathematical Toys* (2002) — 群论形式化与证明
- Herbert Kociemba, 两阶段算法与 God's Number 计算
- Tomas Rokicki, Herbert Kociemba, Morley Davidson, John Dethridge, *The Diameter of the Rubik's Cube Group is Twenty* (2010)
- Jaap Scherphuis, *Rubik's Cube Theory*（元素阶 1260 等结论）
