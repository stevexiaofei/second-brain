---
title: 奇异值分解 SVD：代数推导、几何直觉与在优化器中的用法
type: concept
status: seed
tags: [mathematics, linear-algebra, SVD, eigenvalues, spectral-norm, polar-decomposition, low-rank, courant-fischer, weyl-inequality]
created: 2026-09-11
updated: 2026-09-14
source: 线性代数标准结论 + 本仓库 Muon 系列笔记（msign / 谱范数 / 极分解）反查 + DeepSeek 对话《SVD 几何直觉与证明》整理 + 个人整理
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

> **一句话记住 $U$ 与 $V$ 的分工**：$v_i$ 是"**发力方向**"，$u_i$ 是"**受力后的方向**"。
>
> - $v_i$ 属于**输入空间**，回答"$A$ 在哪些方向上放大最明显"——它是单位球上那组"最佳发力点"；
> - $u_i$ 属于**输出空间**，回答"放大之后，这些方向在输出空间里长什么样"——它是椭球的半轴朝向；
> - $\sigma_i$ 是与这对方向绑定的**放大倍数**（权重）。
>
> 换成数据科学的语言：把 $A$ 看成"用户-电影评分矩阵"，则 $v_i$ 是**电影侧**的潜在模式（"由什么构成"）、$u_i$ 是**用户侧**的潜在模式（"谁喜欢"）、$\sigma_i$ 说明这个模式有多重要。这就是 SVD 被当作推荐系统与 PCA 基础的直觉来源。

### 2.3 两个细节的严格回答

#### 为什么椭球主轴"恰好"对齐 $U$ 的列向量？

设 $x$ 是输入单位球上的点（$\|x\|_2 = 1$），输出 $y = Ax$。代入 $A = U\Sigma V^\top$：

$$y = U\Sigma(V^\top x)$$

按从右到左三步看：

1. **$V^\top$ 只旋转**：$z = V^\top x$ 仍是单位球上的点（$\|z\|_2 = \|x\|_2 = 1$）。**此时还不能谈"主轴"**——单位球各向同性，没有长轴短轴之分。
2. **$\Sigma$ 只拉伸**：$w = \Sigma z$，分量形式 $w_i = \sigma_i z_i$。因为 $\|z\|_2 = 1$，

   $$\|w\|_2^2 = \sum_i \sigma_i^2 z_i^2 \le \sigma_1^2\sum_i z_i^2 = \sigma_1^2$$

   $w$ 的全体构成一个**轴对齐**的椭球：第 $i$ 根半轴恰好落在标准基 $\mathbf{e}_i$ 上，长度为 $\sigma_i$。
3. **$U$ 只旋转**：$y = Uw$。正交变换把"轴对齐椭球"整体转过去，于是原本指向 $\mathbf{e}_i$ 的半轴，被转到了 $U\mathbf{e}_i = u_i$ 的方向。

**所以"椭球半轴方向是 $u_i$、长度是 $\sigma_i$"不是巧合**，而是"$\Sigma$ 造出轴对齐椭球 + $U$ 转动坐标架"的必然结果。

> 关键点：**单位球本身没有主轴**（各向同性），主轴是 $\Sigma$ 拉伸这一步才产生的；$U$ 与 $V$ 都不改变形状，只改朝向。这也顺便解释了为什么**奇异值只由 $\Sigma$ 决定，与 $U,V$ 无关**。

#### 为什么"取到最长半轴的单位输入"是 $v_1$？

由核心等式

$$A v_1 = \sigma_1 u_1,\qquad \|v_1\|_2 = \|u_1\|_2 = 1 \;\Longrightarrow\; \|Av_1\|_2 = \sigma_1$$

而任意单位向量 $x = \sum_i c_i v_i$ 满足 $\|Ax\|_2^2 = \sum_i c_i^2\sigma_i^2 \le \sigma_1^2$，**等号当且仅当 $x \parallel v_1$**（见 §4.1 的推导）。

即：**在输入单位球上沿 $v_1$ 方向"发力"，$A$ 的放大倍数最大（$= \sigma_1$），输出方向是 $u_1$。** 这就是 $\|A\|_2 = \sigma_1 = \max_{\|x\|=1}\|Ax\|_2$ 的几何含义。

### 2.4 为什么"旋转 + 拉伸 + 旋转"是最一般的图景

想象任意一个线性变换。它在原来的坐标架下看起来可能很复杂（既扭曲又旋转）。但它的**内在作用只有两种**：

1. 把某些方向**拉长**、某些方向**压短**（这是"实质"）；
2. 把整体**转了个方向**（这是"表象"，与坐标系选取有关）。

SVD 就是把这两件事彻底分开：$\Sigma$ 只负责"拉长压短"，$U,V$ 只负责"换个坐标架看"。**所以奇异值是变换的内在不变量，而奇异向量依赖方向约定。**

### 2.5 旋转矩阵的例子：奇异值完全看不见旋转

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

### 3.5 已知特征值，如何求特征向量

从 $A^\top A$ 解出 $\lambda_i = \sigma_i^2$ 之后，求 $v_i$ 是一个标准的**解齐次方程组**过程，三步走：

**第 1 步：构造 $A^\top A - \lambda I$。** 由定义 $A^\top A v = \lambda v$ 移项得

$$(A^\top A - \lambda I)v = 0$$

**第 2 步：解这个齐次方程组。** 因为 $\lambda$ 是特征值，$\det(A^\top A - \lambda I) = 0$，所以矩阵必然奇异、必有非零解。行化简后取**基础解系**，其维数为

$$n - \mathrm{rank}(A^\top A - \lambda I)$$

即 $\lambda$ 的**几何重数**。

**第 3 步：单位化。** SVD 语境下我们要的是**单位**特征向量（对应"单位球上的输入"），所以把基础解系的每个向量除以自己的模长。

**完整小例。** 取 $A^\top A = \begin{pmatrix}2&1\\1&2\end{pmatrix}$，已知 $\lambda_1 = 3$、$\lambda_2 = 1$。

$\lambda_1 = 3$：

$$A^\top A - 3I = \begin{pmatrix}-1&1\\1&-1\end{pmatrix}\;\Longrightarrow\; -x_1 + x_2 = 0 \;\Longrightarrow\; v \propto \binom{1}{1} \;\Longrightarrow\; v_1 = \frac{1}{\sqrt2}\binom{1}{1}$$

$\lambda_2 = 1$：

$$A^\top A - I = \begin{pmatrix}1&1\\1&1\end{pmatrix}\;\Longrightarrow\; x_1 + x_2 = 0 \;\Longrightarrow\; v_2 = \frac{1}{\sqrt2}\binom{\ 1}{-1}$$

**重根要特别小心。** 若 $\lambda$ 是 $k$ 重根，其**特征子空间**是 $k$ 维的，基础解系里任取一组基都行，但**必须先 Gram–Schmidt 正交化、再单位化**，否则拼不出正交矩阵 $V$（$V^\top V = I$ 会失效）。

> 好消息：$A^\top A$ 是**对称**矩阵，由谱定理，**不同特征值**对应的特征向量天然正交；只有重根内部需要人为正交化。

**顺带：有了 $v_i$，不必再去算 $A A^\top$。** 由核心等式 $Av_i = \sigma_i u_i$ 直接得

$$u_i = \frac{1}{\sigma_i}Av_i \qquad (\sigma_i > 0)$$

这与 §3.1 第 2 步是同一件事——只需求一次对称矩阵的特征分解（$A^\top A$），另一半靠一次矩阵乘法补出来。

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

**证明**。记 $x = \sum_j c_j v_j$，则 $\|x\|_2^2 = \sum_j c_j^2$、$\|Ax\|_2^2 = \sum_j c_j^2\sigma_j^2$（$\{v_j\}$ 正交、$\{u_j\}$ 正交，交叉项全消）。全程用一个事实：**维数相交引理** $\dim(P\cap Q) \ge \dim P + \dim Q - n$。

**（a）极大–极小形式 $\sigma_i = \max_{\dim S = i}\min_{\|x\|=1, x\in S}\|Ax\|_2$**

- *下界（$\ge$）*：取 $S_i = \mathrm{span}\{v_1,\dots,v_i\}$，$\dim S_i = i$。对单位向量 $x\in S_i$，

  $$\|Ax\|_2^2 = \sum_{j\le i}c_j^2\sigma_j^2 \;\ge\; \sigma_i^2\sum_{j\le i}c_j^2 = \sigma_i^2$$

  又 $x = v_i$ 时取到等号，故 $\min_{x\in S_i}\|Ax\|_2 = \sigma_i$，于是 $\max_{\dim S=i}\min \ge \sigma_i$。
- *上界（$\le$）*：任取 $\dim S = i$，令 $W = \mathrm{span}\{v_i,v_{i+1},\dots,v_n\}$，$\dim W = n-i+1$。由相交引理 $\dim(S\cap W)\ge 1$，取其中单位向量 $x$。此时 $x = \sum_{j\ge i}c_jv_j$，故

  $$\|Ax\|_2^2 = \sum_{j\ge i}c_j^2\sigma_j^2 \;\le\; \sigma_i^2\|x\|_2^2 = \sigma_i^2$$

  于是 $\min_{x\in S}\|Ax\|_2 \le \|Ax\|_2 \le \sigma_i$。由 $S$ 任意，$\max_{\dim S=i}\min \le \sigma_i$。

**（b）极小–极大形式 $\sigma_i = \min_{\dim S = n-i+1}\max_{\|x\|=1, x\in S}\|Ax\|_2$**

- *上界（$\le$）*：取 $T = \mathrm{span}\{v_i,\dots,v_n\}$，$\dim T = n-i+1$。对单位 $x\in T$，$\|Ax\|_2^2 = \sum_{j\ge i}c_j^2\sigma_j^2 \le \sigma_i^2$，且 $x = v_i$ 取等，故 $\max_{x\in T}\|Ax\|_2 = \sigma_i$，于是 $\min\max \le \sigma_i$。
- *下界（$\ge$）*：任取 $\dim S = n-i+1$，令 $V_i = \mathrm{span}\{v_1,\dots,v_i\}$，$\dim V_i = i$。由相交引理 $\dim(S\cap V_i)\ge 1$，取其中单位向量 $x = \sum_{j\le i}c_jv_j$，则

  $$\|Ax\|_2^2 = \sum_{j\le i}c_j^2\sigma_j^2 \;\ge\; \sigma_i^2$$

  故 $\max_{x\in S}\|Ax\|_2 \ge \sigma_i$。由 $S$ 任意，$\min\max \ge \sigma_i$。

两个形式各自双向夹逼，故相等。$\square$

**两个形式的直觉对照**：

| 形式 | 选的子空间 | 最优子空间 | 说了什么 |
|---|---|---|---|
| 极大–极小 | $\dim S = i$ | $\mathrm{span}(v_1,\dots,v_i)$ | "最差的方向能保证多少放大"——挑一个**保证最好**的小空间 |
| 极小–极大 | $\dim S = n-i+1$ | $\mathrm{span}(v_i,\dots,v_n)$ | "最好的方向最多放大多少"——挑一个**已经避开前 $i-1$ 根长轴**的大空间 |

**证明的统一套路**（记住这两招，Weyl 不等式也靠它）：

1. **构造性下界**：把 $x$ 限制在"前 $i$ 个奇异方向"或"后 $n-i+1$ 个奇异方向"里，让坐标只落在 $\sigma_j \ge \sigma_i$ 或 $\sigma_j \le \sigma_i$ 的分量上，直接读出界。
2. **维数相交引理**：要证明"任何 $S$ 都逃不掉"，就构造一个维数足够大的空间，逼它与 $S$ 相交，再在交里取向量。

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

### 4.4 Weyl 不等式与奇异值的扰动稳定性

#### 4.4.1 加法形式

对同型的 $A, B$，奇异值均降序排列，则对任意满足 $i+j-1 \le \min(m,n)$ 的 $i,j$：

$$\boxed{\ \sigma_{i+j-1}(A+B) \;\le\; \sigma_i(A) + \sigma_j(B)\ }$$

特例 $i = j = 1$ 即谱范数的三角不等式 $\|A+B\|_2 \le \|A\|_2 + \|B\|_2$。

**证明**（只用 §4.1 的极小极大原理 + 维数相交引理）。记 $v_k(A)$ 为 $A$ 的第 $k$ 个右奇异向量，并把 $\sigma_k$ 在 $k > \min(m,n)$ 时补 0。

*第 1 步：写出极小极大表达式。* 设 $S^\star$ 是取到

$$\sigma_{i+j-1}(A+B) = \max_{\dim S = i+j-1}\ \min_{\substack{x\in S\\ \|x\|=1}}\|(A+B)x\|_2$$

的子空间。**只要在 $S^\star$ 里找到一个单位向量 $x$ 使 $\|(A+B)x\|_2 \le \sigma_i(A)+\sigma_j(B)$ 即可**——因为左边是 $S^\star$ 上的**最小**值，必不超过任意一点的函数值。

*第 2 步：造两个"只含小奇异方向"的空间。*

$$U_A = \mathrm{span}\{v_i(A),\dots,v_n(A)\},\qquad \dim U_A = n-i+1$$

对 $x\in U_A$（写 $x = \sum_{k\ge i}c_kv_k(A)$）：

$$\|Ax\|_2^2 = \sum_{k\ge i}c_k^2\sigma_k(A)^2 \;\le\; \sigma_i(A)^2\|x\|_2^2$$

同理 $U_B = \mathrm{span}\{v_j(B),\dots,v_n(B)\}$，$\dim U_B = n-j+1$，且 $x\in U_B \Rightarrow \|Bx\|_2 \le \sigma_j(B)\|x\|_2$。

*第 3 步：维数相交引理给出非零向量。* 连续用两次 $\dim(P\cap Q)\ge\dim P+\dim Q-n$：

$$\dim(S^\star\cap U_A\cap U_B) \;\ge\; (i+j-1)+(n-i+1)+(n-j+1)-2n \;=\; 1$$

（第一次：$(i+j-1)+(n-i+1)-n = j$；第二次：$j+(n-j+1)-n = 1$。）所以存在**单位**向量 $x\in S^\star\cap U_A\cap U_B$。

*第 4 步：三角不等式收尾。*

$$\|(A+B)x\|_2 = \|Ax+Bx\|_2 \le \|Ax\|_2+\|Bx\|_2 \le \sigma_i(A)+\sigma_j(B)$$

于是

$$\sigma_{i+j-1}(A+B) = \min_{\substack{x\in S^\star\\ \|x\|=1}}\|(A+B)x\|_2 \;\le\; \|(A+B)x\|_2 \;\le\; \sigma_i(A)+\sigma_j(B) \qquad\square$$

> 注意第 1 步和第 4 步的配合方式：**$\sigma_{i+j-1}$ 是某个"最小"，所以任何一点的取值都自动是它的上界。** 这是"max-min 型定理"证明的固定套路。

#### 4.4.2 扰动定理

取 $A = B + (A-B)$、$j = 1$，代入加法形式：

$$\sigma_i(A) \le \sigma_i(B) + \sigma_1(A-B) \;\Longrightarrow\; \sigma_i(A)-\sigma_i(B) \le \sigma_1(A-B) = \|A-B\|_2$$

交换 $A,B$ 得反向不等式，合起来：

$$\boxed{\ |\sigma_i(A)-\sigma_i(B)| \;\le\; \|A-B\|_2\ }$$

即**奇异值函数 $\sigma_i(\cdot)$ 关于谱范数是 1-Lipschitz 的**。

#### 4.4.3 意义

- **抗噪**：实测数据 $A_{\text{noisy}} = A_{\text{true}} + E$ 时，**不管 $A_{\text{true}}$ 的条件数多差**，奇异值的绝对误差都被 $\|E\|_2$ 锁死。
- **SVD 比 EVD 稳的根本原因**：非正规矩阵的特征值可以极度敏感——例如 $n\times n$ 的 Jordan 块受 $\epsilon$ 扰动，特征值变化可达 $\epsilon^{1/n}$ 量级（$2\times2$ 情形 $\epsilon^{1/2}$：$\begin{pmatrix}0&1\\ \epsilon&0\end{pmatrix}$ 的特征值是 $\pm\sqrt{\epsilon}$），而同一扰动下奇异值只变 $O(\epsilon)$。这就是 SVD 数值可靠的根源，也是 §六 强调"不要显式算 $A^\top A$"之外的另一个理由。
- **低秩近似的前提**：只有当"小奇异值确实小"这件事对噪声稳健时，§五 的截断 SVD 才有意义。

> 更细的扰动界（如 $\sum_i(\sigma_i(A)-\sigma_i(B))^2 \le \|A-B\|_F^2$，即 Wielandt–Hoffman 不等式）可由同样的极小极大框架得到，这里不展开。

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

其中 $Q = UV^\top$ 确实是**正交矩阵**，只需验证它转置乘自己等于单位阵：

$$(UV^\top)^\top(UV^\top) = (VU^\top)(UV^\top) = V(U^\top U)V^\top = VV^\top = I$$

（用到 $(UV^\top)^\top = (V^\top)^\top U^\top = VU^\top$、矩阵乘法结合律，以及 $U,V$ 正交。）同理 $Q Q^\top = I$，故 $Q$ 正交。

> 注意：这只在 $U,V$ 为**方阵**（完整 SVD）时成立。若 $A$ 是长方形矩阵，$UV^\top$ 只能截取前 $r$ 列，此时 $U_r^\top U_r = I_r$ 但 $U_rU_r^\top \ne I_m$，只能叫**半正交**（partial isometry），$Q$ 是"保范但不满"的映射。

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
8. **"$u_i$ 和 $v_i$ 正交"** → $u_i \in \mathbb{R}^m$、$v_i \in \mathbb{R}^n$ 属于**不同空间**，$m \ne n$ 时连内积都没定义，谈不上正交。"正交"说的是**同一组内**互相正交（$u_i \perp u_j$、$v_i \perp v_j$），以及 $U,V$ 是正交矩阵。而"$UV^\top$ 是正交矩阵"是另一个命题（见 §七）。
9. **"$\sigma_i$ 只由 $U,V$ 决定"** → 恰好相反：$\sigma_i$ 完全由 $\Sigma$ 决定，$U,V$ 只负责朝向（见 §2.3）。这也意味着**奇异值是矩阵的内在不变量**，而奇异向量依赖方向约定。

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
- 关于几何直觉，有一点是"点破"式的：**"椭球的主轴对齐 $u_i$"听起来像巧合，其实只是"先造轴对齐椭球（$\Sigma$）、再转坐标架（$U$）"的机械结果**（§2.3）。所以几何图像里 $U$ 和 $V$ 完全不参与"形状"，只参与"朝向"——这与"奇异值只由 $\Sigma$ 决定、与 $U,V$ 无关"是同一件事的两种说法。
- 我意识到 §4.1 与 §4.4 的证明其实是**同一套两招**：① 把 $x$ 限制在"前 $i$ 个 / 后 $n-i+1$ 个奇异方向"上直接读出界；② 用维数相交引理 $\dim(P\cap Q)\ge\dim P+\dim Q-n$ 逼出一个必然存在的向量。抓住这两招，Courant–Fischer 和 Weyl 这两个看起来"高深"的定理就都变成了机械操作——**这也是我判断"自己是否真懂一个变分型定理"的标准：能不能说清它用的是哪种构造。**
- 从"求解流程"角度，SVD 的计算可以记成一条流水线：**对称矩阵特征分解 → 得 $v_i,\sigma_i$ →（重根才需要 Gram–Schmidt）→ 一次矩阵乘法得 $u_i = Av_i/\sigma_i$**。只有 $A^\top A$ 需要真正做特征分解，另一半是免费的；而重根是唯一需要额外处理正交性的地方（§3.5）。

## Related

- [Lagrangian 与约束优化](./lagrangian-and-constrained-optimization.md) — 变分刻画（Courant–Fischer）与约束极值的方法论
- [高斯分布](./gaussian-distribution.md) — 另一个"标准形"：把一般分布化为标准正态
- [最速下降的范数对偶框架](../ai/foundations/training-optimization/steepest-descent-duality-map.md) — 谱范数下的对偶映射 = $\mathrm{msign} = UV^\top$，对偶范数 = 核范数
- [谱范数与 RMS 几何](../ai/foundations/training-optimization/spectral-norm-rms-geometry.md) — 为什么约束最大奇异值等于控制最坏输出增幅
- [Muon 优化器](../ai/foundations/training-optimization/muon-optimizer.md) — Newton–Schulz 迭代：$\varphi(G) = U\varphi(\Sigma)V^\top$
- [Bi-Maxwell：Muon 的物理响应与双时间尺度动量](../ai/foundations/training-optimization/bimaxwell-muon-physical-response.md) — 核范数上界的物理表述

## References

- 线性代数标准结论（谱定理、Courant–Fischer 极小极大原理、Eckart–Young–Mirsky 定理、Weyl 不等式、Wielandt–Hoffman 不等式）
- Golub & Van Loan, *Matrix Computations* — Golub–Kahan 双对角化与 SVD 数值算法
- 知乎《Muon优化器科普，但从最速下降的本质出发》：<https://zhuanlan.zhihu.com/p/1954634867791869927>（谱范数、msign、核范数部分）
- 知乎《范数的定义、性质、边界图像》：<https://zhuanlan.zhihu.com/p/671815885>
- DeepSeek 分享对话：《SVD 几何直觉与证明》（单位球→椭球、左/右奇异向量、极小极大定理、迹循环置换、Weyl 不等式与扰动定理、$UV^\top$ 正交性与极分解）：<https://chat.deepseek.com/share/knz4jig0i6bq1rsndp>
- 本笔记中以下推导为本人补齐并逐行核对：§2.2 表格后的中心结论、§3.1 的 $u_i$ 正交性验证、§3.5 的求特征向量流程、§4.1 的 Courant–Fischer 双向证明、§4.2 的 $\|A\|_F$ 推导与范数大小关系、§4.4 的 Weyl 加法形式与扰动定理证明、§七 中 $UV^\top$ 的正交性与 $F$-范数下最优性的四步证明；第 3.3 节手算例子为本人计算并已代回验算。
