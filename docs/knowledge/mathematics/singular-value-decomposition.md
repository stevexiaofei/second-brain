---
title: 奇异值分解 SVD：代数推导、几何直觉与在优化器中的用法
type: concept
status: seed
tags: [mathematics, linear-algebra, SVD, eigenvalues, spectral-norm, polar-decomposition, low-rank]
created: 2026-09-11
updated: 2026-09-11
source: 线性代数标准结论 + 本仓库 Muon 系列笔记（msign / 谱范数 / 极分解）反查 + 个人整理
---

# 奇异值分解 SVD：代数推导、几何直觉与在优化器中的用法

## 一句话理解

**任何**实矩阵 $A \in \mathbb{R}^{m\times n}$ 都可以分解成"**旋转 → 沿坐标轴拉伸 → 旋转**"三步：

$$A = U\Sigma V^\top$$

- $V$ 的列 $v_1,\dots,v_n$：输入空间中的一组**正交方向**（右奇异向量）；
- $U$ 的列 $u_1,\dots,u_m$：输出空间中的一组**正交方向**（左奇异向量）；
- $\Sigma$ 的对角元 $\sigma_1 \ge \sigma_2 \ge \cdots \ge 0$：**奇异值**，即第 $i$ 个输入方向被放大成第 $i$ 个输出方向的倍数。

核心等式只有一条：

$$\boxed{\ A\,v_i = \sigma_i\,u_i\ }$$

读作：**$A$ 把输入空间的一组正交方向，一对一地映射成输出空间的一组正交方向，只改变长度。**

## 为什么重要

SVD 是"把任意线性变换化为标准形"的工具，几乎所有涉及矩阵的问题都会用到它：

| 用途 | SVD 给出的量 |
|---|---|
| 矩阵的"大小" | 谱范数 $\|A\|_2 = \sigma_1$，Frobenius $\|A\|_F = \sqrt{\sum\sigma_i^2}$，核范数 $\|A\|_* = \sum\sigma_i$ |
| 最接近的半正交矩阵 | 极因子 $UV^\top$（即 $\mathrm{msign}$） |
| 最佳低秩近似 | 截断 SVD（Eckart–Young） |
| 数值稳定性 | 条件数 $\kappa = \sigma_1/\sigma_r$ |
| 数据降维 | PCA = 中心化数据的 SVD |

在**本仓库的优化器主线**里 SVD 出现的具体位置：

- [最速下降的范数对偶框架](../ai/foundations/training-optimization/steepest-descent-duality-map.md)：谱范数下的对偶映射就是 $\mathrm{msign}(g) = UV^\top$；顺带导出对偶范数是核范数 $\sum\sigma_i$。
- [谱范数与 RMS 几何](../ai/foundations/training-optimization/spectral-norm-rms-geometry.md)：约束谱范数 = 约束最大奇异值 = 控制最坏输出增幅。
- [Muon 优化器](../ai/foundations/training-optimization/muon-optimizer.md)：Newton–Schulz 迭代的本质是**只在奇异值上做标量迭代**（$\varphi(G) = U\varphi(\Sigma)V^\top$）。

---

## 一、两种写法：完整 SVD 与薄 SVD

### 1.1 完整 SVD

$$\underbrace{A}_{m\times n} = \underbrace{U}_{m\times m}\ \underbrace{\Sigma}_{m\times n}\ \underbrace{V^\top}_{n\times n}$$

- $U$、$V$ 是**正交矩阵**：$U^\top U = UU^\top = I_m$，$V^\top V = VV^\top = I_n$（为正交，不是"列正交"；完整 SVD 里的 $U,V$ 都是方阵）；
- $\Sigma$ 是"矩形对角阵"：只有 $\Sigma_{ii}$ 可能非零，且 $\Sigma_{11} \ge \Sigma_{22} \ge \cdots \ge 0$。

**当 $m \ne n$ 时 $\Sigma$ 有一整块零**：若 $m > n$（高瘦矩阵），$\Sigma$ 的下半部分是零行；若 $m < n$（矮胖矩阵），右半部分是零列。这些零对应"被压没了的维度"。

### 1.2 薄 SVD（economy / reduced SVD）

设 $r = \mathrm{rank}(A)$。因为 $\sigma_{r+1} = \cdots = 0$，可以只保留前 $r$ 项：

$$A = U_r\Sigma_r V_r^\top,\qquad U_r \in \mathbb{R}^{m\times r},\ \Sigma_r \in \mathbb{R}^{r\times r},\ V_r \in \mathbb{R}^{n\times r}$$

这里 $U_r^\top U_r = I_r$ 但 $U_rU_r^\top \ne I_m$（"列正交"而非"正交"）。**这是实际计算与理论推导中最常用的形式**——因为后面的项全是零，保留它们只是浪费存储。

$$\boxed{\ A = \sum_{i=1}^{r}\sigma_i\,u_i v_i^\top\ }$$

这个"**秩一外积之和**"的写法是理解 SVD 的一把钥匙：**$A$ 是 $r$ 个秩一矩阵的加权叠加，权重就是奇异值。** 把小的 $\sigma_i$ 丢掉，就得到低秩近似（见第五节）。

---

## 二、几何解释：三步变换

### 2.1 $A = U\Sigma V^\top$ 是三步复合

把 $A$ 作用在向量 $x$ 上，按从右到左的顺序：

```text
        x                V^T x              Σ V^T x            U Σ V^T x = A x
   ∈ ℝⁿ  ──① V^T ──→  ∈ ℝⁿ  ──② Σ ──→  ∈ ℝᵐ  ──③ U ──→  ∈ ℝᵐ
          旋转/反射            拉伸/压扁            旋转/反射
       （输入空间内）      （可能升降维）        （输出空间内）
```

| 步骤 | 做什么 | 作用在基上 |
|---|---|---|
| ① $V^\top$ | 输入空间里的旋转（正交变换，不改变长度） | 把 $v_i$ 转到标准基 $\mathbf{e}_i$ |
| ② $\Sigma$ | 沿各坐标轴拉伸 $\sigma_i$ 倍（$m\ne n$ 时还改变维度） | $\mathbf{e}_i \mapsto \sigma_i\mathbf{e}_i$ |
| ③ $U$ | 输出空间里的旋转 | 把 $\mathbf{e}_i$ 转到 $u_i$ |

三步串起来：

$$v_i \xrightarrow{V^\top} \mathbf{e}_i \xrightarrow{\Sigma} \sigma_i\mathbf{e}_i \xrightarrow{U} \sigma_i u_i
\quad\Longrightarrow\quad Av_i = \sigma_i u_i \quad\checkmark$$

**直觉**：**任意线性变换都可以被"看穿"成一次拉伸。** 你只需要站在正确的角度看（用 $V$ 当输入坐标架、用 $U$ 当输出坐标架），再复杂的矩阵也只是一个对角矩阵。

### 2.2 单位球 → 椭球（最重要的几何图像）

$A$ 把输入空间的**单位球** $S^{n-1} = \{x : \|x\|_2 = 1\}$ 映射为输出空间中的一个**椭球**：

```text
   输入空间 ℝⁿ                        输出空间 ℝᵐ

       v₂                                  σ₁u₁ ← 最长半轴
        ↑    ┌───────┐                        ╱
        │   ╱  单位球  ╲                     ╱ ← A 的像：椭球
        │  │    ●      │      ──A──→    ┌──●──┐
        │   ╲         ╱                    ╲   ╱
        └──→ v₁                             σ₂u₂ ← 最短半轴
```

| 椭球的几何量 | 对应的代数量 |
|---|---|
| 最长半轴长度 | $\sigma_1 = \|A\|_2$（**谱范数**） |
| 全部半轴长度之和 | $\sum\sigma_i = \|A\|_*$（**核范数**） |
| 半轴平方和 | $\sum\sigma_i^2 = \|A\|_F^2$ |
| 半轴个数（非零半轴） | $r = \mathrm{rank}(A)$ |
| 半轴方向 | $u_1, u_2, \dots$（左奇异向量） |
| 取到最长半轴的单位输入 | $v_1$（右奇异向量） |
| 半轴的"扁率" | $\kappa = \sigma_1/\sigma_r$（**条件数**） |

**由此可以立刻"看懂"一堆中心结论：**

- **谱范数**：所有方向中，$A$ 最多把长度放大 $\sigma_1$ 倍。$\|Ax\|_2 \le \sigma_1\|x\|_2$，等号在 $x \parallel v_1$ 时取得。
- **最小奇异值**：存在方向 $v_r$，$A$ 只会把它放大 $\sigma_r$ 倍。$\sigma_r$ 越小，矩阵越"接近降秩"。
- **方阵的行列式**：$|\det A| = \prod_i\sigma_i$，即**体积的缩放倍数 = 诸半轴长度之积**。
- **秩的几何含义**：零奇异值意味着椭球在那些方向被压成 0——**维度丢失了，信息不可恢复**。

### 2.3 为什么"旋转 + 拉伸 + 旋转"是最一般的图景

想象任意一个线性变换。它在原来的坐标架下看起来可能很复杂（既扭曲又旋转）。但它的**内在作用只有两种**：

1. 把某些方向**拉长**、某些方向**压短**（这是"实质"）；
2. 把整体**转了个方向**（这是"表象"，与坐标系选取有关）。

SVD 就是把这两件事彻底分开：$\Sigma$ 只负责"拉长压短"，$U,V$ 只负责"换个坐标架看"。**所以奇异值是变换的内在不变量，而奇异向量依赖方向约定。**

### 2.4 旋转矩阵的例子：奇异值完全看不见旋转

取 90° 旋转

$$A = \begin{pmatrix}0&-1\\1&0\end{pmatrix}$$

- **特征值**：$\det(A-\lambda I) = \lambda^2 + 1 = 0 \Rightarrow \lambda = \pm i$ —— **没有实特征值**，谈"不变方向"完全无从下手；
- **奇异值**：$A^\top A = I$，所以 $\sigma_1 = \sigma_2 = 1$ —— **两个方向都没被拉伸**。

这正是"奇异值 = 拉伸量"的最干净证据：**旋转没有拉伸任何方向，所以奇异值全为 1；而它在实数域里根本没有特征方向。** 这解释了为什么优化器理论（关心"放大多少倍"）更爱用 SVD 而不是特征分解。

---

## 三、代数上怎么算出来

### 3.1 从 $A^\top A$ 的特征分解构造

**关键观察**：$A^\top A$ 一定是对称半正定矩阵，因而必可正交对角化（谱定理）。

**第 1 步：展开 $A^\top A$。** 假设已有 $A = U\Sigma V^\top$，则

$$A^\top A = (U\Sigma V^\top)^\top(U\Sigma V^\top) = V\Sigma U^\top U\Sigma V^\top = V\Sigma^2 V^\top$$

因为 $\Sigma$ 是矩形对角阵，$\Sigma^\top\Sigma$ 是 $n\times n$ 对角阵，对角元为 $\sigma_i^2$。所以

$$\boxed{\ A^\top A = V\,\mathrm{diag}(\sigma_1^2,\dots,\sigma_n^2)\,V^\top\ }$$

**这一步信息量极大**：它说明

- $\sigma_i^2$ 是 $A^\top A$ 的**特征值**；
- $v_i$ 是 $A^\top A$ 的**特征向量**；
- 由于 $A^\top A$ 半正定，$\sigma_i^2 \ge 0$，所以 $\sigma_i = \sqrt{\lambda_i \ge 0}$ 总是实数非负。

**第 2 步：反推 $u_i$。** 对 $\sigma_i > 0$，由 $Av_i = \sigma_i u_i$ 得

$$u_i = \frac{Av_i}{\sigma_i}$$

**第 3 步：验证 $u_i$ 确实是正交单位向量。**

*长度*：

$$\|u_i\|_2^2 = \frac{v_i^\top A^\top A v_i}{\sigma_i^2} = \frac{\lambda_i\, v_i^\top v_i}{\sigma_i^2} = \frac{\sigma_i^2 \cdot 1}{\sigma_i^2} = 1$$

*正交*：对 $i \ne j$，注意 $A^\top A v_j = \lambda_j v_j$，故

$$u_i^\top u_j = \frac{v_i^\top A^\top A v_j}{\sigma_i\sigma_j} = \frac{\lambda_j\, v_i^\top v_j}{\sigma_i\sigma_j} = \frac{\sigma_j^2 \cdot 0}{\sigma_i\sigma_j} = 0$$

**第 4 步：补全到完整正交基。** 若 $r < m$，上面只得到 $r$ 个 $u_i$。任取一组与它们正交的单位向量补足 $U$ 的其余列（并让对应 $\sigma_i = 0$），即得完整 SVD。

> **这段构造就是 SVD 的存在性证明**：因为对称半正定矩阵总能正交对角化，所以 SVD 对**任何**实矩阵都存在。

### 3.2 对称形式：从 $AA^\top$ 看

同理

$$AA^\top = U\Sigma\Sigma^\top U^\top = U\,\mathrm{diag}(\sigma_1^2,\dots,\sigma_m^2)\,U^\top$$

**两边的非零特征值完全相同**（都是 $\sigma_1^2,\dots,\sigma_r^2$），只是 $A^\top A$ 给了 $v_i$、$AA^\top$ 给了 $u_i$。这解释了为什么高瘦矩阵（$m \gg n$）时用 $A^\top A$（尺寸 $n\times n$）更省算力。

### 3.3 一个完整的手算例子

取

$$A = \begin{pmatrix}3&0\\4&5\end{pmatrix}$$

**① 算 $A^\top A$**

$$A^\top A = \begin{pmatrix}3&4\\0&5\end{pmatrix}\begin{pmatrix}3&0\\4&5\end{pmatrix} = \begin{pmatrix}9+16 & 20\\ 20 & 25\end{pmatrix} = \begin{pmatrix}25&20\\20&25\end{pmatrix}$$

**② 求特征值**（$\lambda = \sigma^2$）

$$\det\begin{pmatrix}25-\lambda & 20\\ 20 & 25-\lambda\end{pmatrix} = (25-\lambda)^2 - 400 = 0 \;\Longrightarrow\; 25-\lambda = \pm 20$$

$$\lambda_1 = 45,\quad \lambda_2 = 5 \;\Longrightarrow\; \sigma_1 = 3\sqrt5 \approx 6.708,\quad \sigma_2 = \sqrt5 \approx 2.236$$

**③ 求右奇异向量 $v_i$（$A^\top A$ 的单位特征向量）**

$\lambda_1 = 45$：$-20v_1 + 20v_2 = 0 \Rightarrow v_1 = v_2$，归一化得 $v_1 = \frac{1}{\sqrt2}\binom{1}{1}$

$\lambda_2 = 5$：$20v_1 + 20v_2 = 0 \Rightarrow v_2 = -v_1$，得 $v_2 = \frac{1}{\sqrt2}\binom{\ 1}{-1}$

**④ 求左奇异向量 $u_i = Av_i/\sigma_i$**

$$u_1 = \frac{1}{3\sqrt5}\cdot\frac{1}{\sqrt2}\binom{3\cdot1 + 0\cdot1}{4\cdot1+5\cdot1} = \frac{1}{3\sqrt{10}}\binom{3}{9} = \frac{1}{\sqrt{10}}\binom{1}{3}$$

$$u_2 = \frac{1}{\sqrt5}\cdot\frac{1}{\sqrt2}\binom{3\cdot1+0\cdot(-1)}{4\cdot1+5\cdot(-1)} = \frac{1}{\sqrt{10}}\binom{3}{-1}$$

**⑤ 验算 $A = U\Sigma V^\top$**

$$U\Sigma V^\top = \frac{1}{\sqrt{10}}\begin{pmatrix}1&3\\3&-1\end{pmatrix}\begin{pmatrix}3\sqrt5&0\\0&\sqrt5\end{pmatrix}\frac{1}{\sqrt2}\begin{pmatrix}1&1\\1&-1\end{pmatrix} = \frac{1}{2}\begin{pmatrix}6&0\\8&10\end{pmatrix} = \begin{pmatrix}3&0\\4&5\end{pmatrix}\ \checkmark$$

（中间用到 $\frac{\sqrt5}{\sqrt{20}} = \frac12$。）

几何核对：单位圆被 $A$ 映射成半轴为 $6.708$ 与 $2.236$、方向分别为 $\frac{1}{\sqrt{10}}(1,3)$ 与 $\frac{1}{\sqrt{10}}(3,-1)$ 的椭圆。

### 3.4 与特征分解的关系与区别

| | 特征分解 $A = P\Lambda P^{-1}$ | SVD $A = U\Sigma V^\top$ |
|---|---|---|
| 适用矩阵 | 只对可对角化矩阵 | **对任意矩阵都存在** |
| 是否总存在 | 否 | **是** |
| 变换的基 | 输入/输出用**同一组**基 | 输入用 $V$、输出用 $U$，**可以不同** |
| 是否正交 | 一般不是 | **$U,V$ 一定正交** |
| 数值稳定性 | 可能病态（$P$ 可能接近奇异） | **极其稳定**（正交变换不放大误差） |
| 标量 | $\lambda_i$ 可为**复数**、可正可负 | $\sigma_i$ 一定**实数非负** |

**对对称半正定矩阵，两者重合**：此时 $A = Q\Lambda Q^\top$，于是 $U = V = Q$、$\Sigma = \Lambda$。**所以 SVD 是特征分解在"非对称/非方阵"情形下的推广与修复。**

**两个经典反例**（都值得记住）：

| 矩阵 | 特征值 | 奇异值 | 说明 |
|---|---|---|---|
| $\begin{pmatrix}0&1\\0&0\end{pmatrix}$ | $0,\ 0$ | $1,\ 0$ | 特征值全零却"能量"为 1，$\|A\|_2 = 1 > \rho(A) = 0$ |
| $\begin{pmatrix}0&-1\\1&0\end{pmatrix}$ | $\pm i$ | $1,\ 1$ | 无实特征方向，但完全不拉伸 |

> **直觉总结**：**特征值描述"方向不变时被放大多少"，奇异值描述"把方向也允许改变时最多放大多少"。** 因为允许改变方向，奇异值总能取到更大的值，且永远良定义。

---

## 四、等价刻画（严谨版）

这些刻画把"定义"变成"可优化的极值问题"，是很多理论的起点。

### 4.1 谱范数的变分刻画

$$\|A\|_2 = \sigma_1 = \max_{\|x\|_2 = 1}\|Ax\|_2 = \max_{x\ne0}\frac{\|Ax\|_2}{\|x\|_2}$$

**推导**：任取 $x = \sum_i c_i v_i$（$\{v_i\}$ 是正交基，$\|x\|^2 = \sum c_i^2$），则

$$\|Ax\|_2^2 = \Bigl\|\sum_i c_i \sigma_i u_i\Bigr\|_2^2 = \sum_i c_i^2\sigma_i^2 \le \sigma_1^2\sum_i c_i^2 = \sigma_1^2\|x\|_2^2$$

（中间用了 $\{u_i\}$ 的正交性消掉交叉项。）等号在 $c_1 = 1$、其余为 0 时成立，即 $x = v_1$。

**推广到第 $i$ 个奇异值（Courant–Fischer 极小极大原理）**：

$$\sigma_i = \max_{\substack{S \subseteq \mathbb{R}^n \\ \dim S = i}}\ \min_{\substack{x \in S \\ \|x\|=1}}\|Ax\|_2
= \min_{\substack{S \subseteq \mathbb{R}^n \\ \dim S = n-i+1}}\ \max_{\substack{x \in S \\ \|x\|=1}}\|Ax\|_2$$

**直白解释**：在所有 $i$ 维子空间里挑一个，使得"$A$ 在该子空间上的最小放大倍数"尽可能大——那个最优值就是 $\sigma_i$。等价说法：$\sigma_i = \|A\|_2$ 限制在前 $i-1$ 个右奇异向量的正交补上。

### 4.2 各种"矩阵范数"与奇异值

| 范数 | 定义 | 用奇异值表示 | 几何含义 |
|---|---|---|---|
| 谱范数 $\|A\|_2$ | $\max_{\|x\|_2=1}\|Ax\|_2$ | $\sigma_1$ | 最长半轴 |
| 核范数 $\|A\|_*$ | 奇异值之和 | $\sum_i\sigma_i$ | 半轴长度和 |
| Frobenius $\|A\|_F$ | $\sqrt{\sum_{ij}A_{ij}^2}$ | $\sqrt{\sum_i\sigma_i^2}$ | 半轴长度平方和的根 |

**$\|A\|_F$ 的推导**：

$$\|A\|_F^2 = \mathrm{tr}(A^\top A) = \mathrm{tr}(V\Sigma^2V^\top) = \mathrm{tr}(\Sigma^2\underbrace{V^\top V}_{I}) = \mathrm{tr}(\Sigma^2) = \sum_i\sigma_i^2$$

**三者的大小关系**（可由上面的表示直接看出）：

$$\|A\|_2 \;\le\; \|A\|_F \;\le\; \|A\|_* \;\le\; \sqrt{r}\,\|A\|_F$$

（第一条：$\sigma_1 = \sqrt{\sigma_1^2} \le \sqrt{\sum\sigma_i^2}$；最后一条：$\sum\sigma_i \le \sqrt{r}\sqrt{\sum\sigma_i^2}$，Cauchy–Schwarz。）

> 这三条在优化器理论里直接对应不同约束：约束谱范数 → 对偶是核范数；约束 Frobenius → 对偶是自身。

### 4.3 行列式与体积

对 $n\times n$ 方阵：

$$|\det A| = |\det U| \cdot |\det\Sigma| \cdot |\det V^\top| = 1 \cdot \prod_i\sigma_i \cdot 1 = \prod_i\sigma_i$$

（正交矩阵行列式为 $\pm1$。）即**体积缩放倍数 = 诸奇异值之积**，与"椭球半轴之积"一致。

### 4.4 奇异值对扰动的稳定性（Weyl 不等式）

$$|\sigma_i(A) - \sigma_i(B)| \le \|A - B\|_2$$

即**奇异值对矩阵扰动是 1-Lipschitz 的**。这是 SVD 数值上可靠的根本原因之一（对比之下，非正规矩阵的特征值可以极度敏感）。

---

## 五、低秩近似：Eckart–Young–Mirsky 定理

由 $A = \sum_{i=1}^r \sigma_i u_iv_i^\top$，最自然的想法是"掐掉小奇异值"：

$$A_k = \sum_{i=1}^{k}\sigma_i\,u_iv_i^\top$$

**定理（Eckart–Young–Mirsky）**：$A_k$ 是所有秩不超过 $k$ 的矩阵中，在谱范数与 Frobenius 范数下**同时最优**的近似：

$$\min_{\mathrm{rank}(B)\le k}\|A - B\|_2 = \sigma_{k+1},\qquad
\min_{\mathrm{rank}(B)\le k}\|A - B\|_F = \sqrt{\sum_{i>k}\sigma_i^2}$$

且最优解都是 $B = A_k$。

**几何解释**：

```text
   秩 r 的椭球                秩 k 的近似（丢掉 r-k 个最短半轴）

      σ₁ ╱                            σ₁ ╱
        ╱ ← 半轴 σ₂, ..., σ_r          ╱ ← 只保留前 k 个
      ●                               ●
```

**"丢掉最小的半轴"就是最优的低秩近似**——因为误差恰好等于被丢掉的那部分能量。

**与 PCA 的关系**：对中心化数据矩阵 $X \in \mathbb{R}^{N\times d}$，$X = U\Sigma V^\top$，则主成分方向就是 $v_1,\dots,v_k$（$X^\top X$ 的特征向量），投影即 $X V_k = U_k\Sigma_k$。**PCA 就是 SVD 的截断。**

---

## 六、数值计算：为什么不该显式算 $A^\top A$

### 6.1 条件数被平方

$$\kappa(A^\top A) = \frac{\sigma_1(A^\top A)}{\sigma_r(A^\top A)} = \frac{\sigma_1^2}{\sigma_r^2} = \kappa(A)^2$$

**这意味着**：若 $\kappa(A) = 10^8$（双精度尚可），则 $\kappa(A^\top A) = 10^{16}$，超出 double 的精度（约 $10^{-16}$），**小奇异值会被完全冲掉**。

**实践结论**：**永远不要用"算 $A^\top A$ 再做特征分解"来求 SVD**（除非矩阵很小且条件数很好）。这是教学用推导与生产用算法的分界线。

### 6.2 实际算法

| 场景 | 算法 |
|---|---|
| 一般中小矩阵、需要完整 SVD | **Golub–Kahan 双对角化** + 隐式 QR 迭代（LAPACK `gesdd`） |
| 只要前 $k$ 个奇异三元组 | **Lanczos**（Krylov 子空间）或 **随机化 SVD**（randomized range finder） |
| 只需极因子 / 半正交化、且要大量矩阵乘法且低精度 | **Newton–Schulz 迭代**（Muon 走的路） |

### 6.3 在 Muon 语境下的替代方案

Muon 不显式做 SVD，而是用 Newton–Schulz 迭代求极因子。它的代数依据是

$$\varphi(G) = U\,\varphi(\Sigma)\,V^\top$$

即**任何奇多项式都只作用在奇异值上、不动奇异向量**（推导见 [Muon 优化器 §3](../ai/foundations/training-optimization/muon-optimizer.md)）。所以只要设计一个把 $\sigma$ 推向 1 的标量多项式，整个矩阵迭代就会收敛到 $UV^\top$——**用一串矩阵乘法代替了 SVD**。

---

## 七、极分解：SVD 的直接推论

由 $A = U\Sigma V^\top$ 可以立刻写出

$$A = \underbrace{(UV^\top)}_{\text{正交}}\ \underbrace{(V\Sigma V^\top)}_{\text{对称半正定}}$$

验证：

$$UV^\top\cdot V\Sigma V^\top = U\Sigma V^\top = A \quad\checkmark$$

而 $V\Sigma V^\top = \sqrt{A^\top A}$（因为 $A^\top A = V\Sigma^2V^\top$，开根号即得）。所以

$$\boxed{\ A = QH,\qquad Q = UV^\top,\qquad H = \sqrt{A^\top A}\ }$$

**这是"极坐标"的类比**：复数 $z = re^{i\theta}$ 中 $r$ 是长度、$e^{i\theta}$ 是方向；矩阵分解中 $H$（半正定）扮演"长度"，$Q = UV^\top$（正交）扮演"方向"。$Q$ 就是**极因子**，也叫矩阵的"zeroth power"。

**这就是 $\mathrm{msign}$**：

$$\mathrm{msign}(A) = UV^\top = \operatorname{polar}(A)$$

也就是"**只保留方向、把所有奇异值压成 1**"。

**为什么它是最接近 $A$ 的正交矩阵**（Frobenius 意义）？

为清晰起见设 $A$ 为 $n\times n$ 方阵，用**完整** SVD（此时 $U,V$ 是方阵正交矩阵）。

**第 1 步：把 $Q$ 搬进对角坐标。** 因为 $UU^\top = I$、$VV^\top = I$，所以 $Q = U(U^\top QV)V^\top$，于是

$$A - Q = U\Sigma V^\top - U(U^\top QV)V^\top = U\bigl(\Sigma - W\bigr)V^\top,\qquad W := U^\top QV$$

**第 2 步：正交变换不改变 Frobenius 范数。** 对正交 $U,V$，

$$\|UMV^\top\|_F^2 = \mathrm{tr}(VM^\top U^\top UMV^\top) = \mathrm{tr}(M^\top M) = \|M\|_F^2$$

且 $W = U^\top QV$ 作为正交矩阵的乘积仍正交。于是问题化为

$$\min_{Q\ \text{正交}}\|A-Q\|_F^2 = \min_{W\ \text{正交}}\|\Sigma - W\|_F^2$$

**第 3 步：展开目标函数。**

$$\|\Sigma - W\|_F^2 = \|\Sigma\|_F^2 + \|W\|_F^2 - 2\,\mathrm{tr}(\Sigma^\top W) = \sum_i\sigma_i^2 + n - 2\sum_i\sigma_i W_{ii}$$

关键一步用到：**正交矩阵的 Frobenius 范数恒为 $\sqrt n$**（其全部奇异值都是 1），所以 $\|W\|_F^2 = n$ 与 $W$ 无关。

**第 4 步：只剩最大化 $\sum_i\sigma_i W_{ii}$。** 正交矩阵的对角元满足 $|W_{ii}| \le \|W_{:,i}\|_2 = 1$（对角元是第 $i$ 列的一个分量，不会超过该列范数）。又因 $\sigma_i \ge 0$，

$$\sum_i\sigma_i W_{ii} \le \sum_i\sigma_i\,|W_{ii}| \le \sum_i\sigma_i$$

等号取到当且仅当所有 $W_{ii} = 1$，此时 $W = I$。

**结论**：$W = I$ 时目标最小，对应 $U^\top QV = I$，即 $Q = UV^\top$，且

$$\min_{Q}\|A-Q\|_F = \sqrt{\sum_i(\sigma_i-1)^2}\qquad\square$$

---

## 八、常见误区

1. **"SVD 和特征分解差不多"** → 差在根本处：SVD 用**两组**基（输入 $V$、输出 $U$），特征分解用**一组**基；SVD 总存在且数值稳定，特征分解不然。
2. **"奇异值就是特征值的绝对值"** → 只对**对称半正定**矩阵成立。旋转矩阵 $\binom{0\ -1}{1\ \ 0}$ 特征值 $\pm i$、奇异值 $1,1$，完全对不上。
3. **"$\sigma_i$ 可以为负"** → 不可能。$\sigma_i = \sqrt{\lambda_i(A^\top A)} \ge 0$。所以 $\mathrm{msign}$ 里的"sign"是"把所有非零奇异值置 1"，而不是正负号。
4. **"$\|A\|_2$ 就是最大特征值的绝对值"** → 是**最大奇异值**。$\binom{0\ 1}{0\ 0}$ 的谱半径是 0，谱范数是 1。
5. **"算 SVD 就是算 $A^\top A$ 的特征分解"** → 数学上等价，**数值上是灾难**（条件数被平方）。
6. **"零奇异值说明矩阵是零"** → 只说明**降秩**：有些方向被压没了，信息不可恢复。这才是低秩近似有意义的原因。
7. **"奇异向量唯一"** → 只有当奇异值**互不相同**时才唯一（且仍可选择符号）。有重根时，对应子空间内的任意正交基都行。

---

## 我的理解

（以下为个人理解）

- 我理解 SVD 的本质是**"给线性变换找最舒服的坐标系"**。任何矩阵在错误的角度下都显得复杂，但只要允许输入输出**各自**选一组正交基，它立刻变成对角。这解释了为什么 SVD 的表达能力比特征分解强：**它多给了一个自由度（输出基可以与输入基不同）**，而正是这个自由度换来了"对任意矩阵都存在"。
- 三个层级的心智模型，我按使用频次排序：
  1. **$Av_i = \sigma_i u_i$** —— 一行核心等式，随身携带；
  2. **单位球 → 椭球** —— 把谱范数、核范数、Frobenius、行列式、条件数、秩全部统一成"椭球的某个几何量"；
  3. **$A = \sum\sigma_i u_iv_i^\top$** —— 把矩阵看成"秩一块的加权和"，这是低秩近似、PCA、LoRA 的共同起点。
- 一个让我印象很深的对照：**特征值回答"如果方向不许变，能放大多少"，奇异值回答"允许换方向时最多能放大多少"**。这解释了为什么几乎所有"稳定性/最坏情况/鲁棒性"的理论都建立在奇异值上——它们问的本来就是"允许最坏方向时会发生什么"。
- 在优化器语境里，SVD 的价值可以用一句话概括：**它把"矩阵该怎么改"拆成了两个正交的问题——改方向（$U,V$）还是改幅度（$\Sigma$）。** `msign` 只动 $\Sigma$（压平），weight decay 也只动 $\Sigma$（等比缩放），而 weight decay 甚至不动 $U,V$。理解了这层，Muon 的许多设计就不再零散。

## Related

- [Lagrangian 与约束优化](./lagrangian-and-constrained-optimization.md) — 变分刻画（Courant–Fischer）与约束极值的方法论
- [高斯分布](./gaussian-distribution.md) — 另一个"标准形"：把一般分布化为标准正态
- [最速下降的范数对偶框架](../ai/foundations/training-optimization/steepest-descent-duality-map.md) — 谱范数下的对偶映射 = $\mathrm{msign} = UV^\top$，对偶范数 = 核范数
- [谱范数与 RMS 几何](../ai/foundations/training-optimization/spectral-norm-rms-geometry.md) — 为什么约束最大奇异值等于控制最坏输出增幅
- [Muon 优化器](../ai/foundations/training-optimization/muon-optimizer.md) — Newton–Schulz 迭代：$\varphi(G) = U\varphi(\Sigma)V^\top$
- [Bi-Maxwell：Muon 的物理响应与双时间尺度动量](../ai/foundations/training-optimization/bimaxwell-muon-physical-response.md) — 核范数上界的物理表述

## References

- 线性代数标准结论（谱定理、Courant–Fischer 极小极大原理、Eckart–Young–Mirsky 定理、Weyl 不等式）
- Golub & Van Loan, *Matrix Computations* — Golub–Kahan 双对角化与 SVD 数值算法
- 知乎《Muon优化器科普，但从最速下降的本质出发》：<https://zhuanlan.zhihu.com/p/1954634867791869927>（谱范数、msign、核范数部分）
- 知乎《范数的定义、性质、边界图像》：<https://zhuanlan.zhihu.com/p/671815885>
- 本笔记的第 2.2、3.1、4.1、4.2、七 节含本人补齐的中间推导；第 3.3 节手算例子为本人计算并已代回验算。
