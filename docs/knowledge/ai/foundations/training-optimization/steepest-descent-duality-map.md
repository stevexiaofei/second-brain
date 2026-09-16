---
title: 最速下降的范数对偶框架：SGD、SignSGD 与 Muon 是同一条公式
type: concept
status: seed
tags: [AI, optimization, norm, dual-norm, duality-map, polar-set, support-function, msign, spectral-norm, Muon, SignSGD]
created: 2026-09-11
updated: 2026-09-14
source: 知乎《Muon优化器科普，但从最速下降的本质出发》（https://zhuanlan.zhihu.com/p/1954634867791869927）+ Bernstein & Newhouse《Old Optimizer, New Norm》+ 本人补齐的推导中间步骤
---

# 最速下降的范数对偶框架：SGD、SignSGD 与 Muon 是同一条公式

> 本笔记只记录**推导**。Muon 的算法总览、参数分组、实验证据见 [Muon 优化器](./muon-optimizer.md)；谱范数为什么被选中见 [谱范数与 RMS 几何](./spectral-norm-rms-geometry.md)。

## 一句话理解

所有人都在解**同一个最速下降方程**，区别只在于「用什么范数去约束更新量」。一旦把方程解出来，最优更新量必然长成

$$\Delta w^\star = -\frac{\|g\|^\dagger}{\lambda}\,\mathrm{dualize}_{\|\cdot\|}(g)$$

其中 $\|g\|^\dagger$ 是**对偶范数**、$\mathrm{dualize}(g)$ 是**对偶映射**。于是：

- 约束用 $\ell_2$ → 对偶映射是 $g/\|g\|_2$ → **SGD**
- 约束用 $\ell_\infty$ → 对偶映射是 $\mathrm{sign}(g)$ → **SignSGD / 关动量的 Adam**
- 约束用谱范数 → 对偶映射是 $\mathrm{msign}(g) = UV^\top$ → **Shampoo / Muon**

**所谓"设计优化器"，本质上是设计对偶映射。**

## 为什么重要

- 它把"Adam 和 Muon 哪个好"这类经验之争，转化为"哪个范数更契合网络的几何"这个**可论证**的问题。
- 它给出一个很硬的洞察：**参数和梯度根本不在同一个空间**（$V$ vs $V^*$），所以 `weight - lr * grad` 在数学上是类型错误，而对偶映射正是修复这个类型错误的算子。
- 它是理解 [为什么是谱范数](./spectral-norm-rms-geometry.md) 的前置：先有统一框架，才能谈"选哪个几何"。

---

## 一、列出最速下降方程

### 1.1 二阶泰勒展开

记当前参数 $W \in V$、loss 为 $L(W)$、梯度 $g = \nabla_W L(W)$。对一个小更新量 $\Delta w \in V$，二阶泰勒展开为

$$L(W + \Delta w) = L(W) + \langle g, \Delta w \rangle + \frac{\lambda}{2}\|\Delta w\|^2 ,\qquad \lambda > 0$$

三项的物理含义：

| 项 | 含义 |
|---|---|
| $L(W)$ | 常数，优化时无关 |
| $\langle g, \Delta w\rangle$ | 一阶方向导数——**唯一与 $\Delta w$ 方向有关**的项 |
| $\frac{\lambda}{2}\|\Delta w\|^2$ | 二阶/阻尼项，同时起正则作用 |

其中 $\langle A, B\rangle$ 是 Frobenius 内积，$\langle A, B\rangle = \mathrm{tr}(A^\top B)$ = 对应元素乘积之和。

### 1.2 要解的方程

$$\Delta w^\star = \arg\min_{\Delta w \in V}\left[\langle g, \Delta w\rangle + \frac{\lambda}{2}\|\Delta w\|^2\right]$$

**注意范数 $\|\cdot\|$ 是自由的**——它由我们选定，正是这个自由度区分了不同优化器。

---

## 二、解这个方程

### 2.1 第一步：把问题拆成「方向」与「模长」

因为只有 $\langle g, \Delta w\rangle$ 与方向有关，而 $\|\Delta w\|$ 与方向无关，所以可以先固定模长 $\|\Delta w\| = r$，单独求最优方向。

对任意单位向量 $t$（$\|t\| = 1$），令 $\Delta w = r\,t$，则

$$\langle g, \Delta w\rangle = r\langle g, t\rangle$$

固定 $r$ 时要最小化 $\langle g, \Delta w\rangle$，就是要**最小化** $\langle g,t\rangle$，即令 $t$ 取相反方向：

$$\arg\min_{\|t\|=1}\langle g, t\rangle = -\arg\max_{\|t\|=1}\langle g, t\rangle$$

**这就是对偶映射第一次出现的地方。**

### 2.2 第二步：得到一个最优下界（关键不等式）

对任意非零 $\Delta w$，令 $t = \Delta w/\|\Delta w\|$，则 $\langle g,\Delta w\rangle = \|\Delta w\|\langle g,t\rangle$。由

$$\|g\|^\dagger := \max_{\|t\|=1}\langle g,t\rangle \;\Longrightarrow\; \langle g,t\rangle \ge -\|g\|^\dagger\ \ (\forall \|t\|=1)$$

可得

$$\boxed{\ \langle g,\Delta w\rangle \ \ge\ -\|g\|^\dagger\,\|\Delta w\|\ }$$

等号成立当且仅当 $t = \mathrm{dualize}(g)$，即 $\Delta w$ 平行于最速上升方向。于是原问题可以**降到一维**：

$$\arg\min_{\Delta w}\left[\langle g,\Delta w\rangle + \frac{\lambda}{2}\|\Delta w\|^2\right] = \arg\min_{r \ge 0}\left[-\|g\|^\dagger r + \frac{\lambda}{2}r^2\right]$$

### 2.3 第三步：解一维二次函数

这是初中题。令 $\phi(r) = -\|g\|^\dagger r + \frac{\lambda}{2}r^2$，则

$$\phi'(r) = -\|g\|^\dagger + \lambda r = 0 \;\Longrightarrow\; r^\star = \frac{\|g\|^\dagger}{\lambda}$$

（$\phi'' = \lambda > 0$，确为极小值。）代入 $\phi$ 得最小值 $-\dfrac{(\|g\|^\dagger)^2}{2\lambda}$。

### 2.4 结果

$$\boxed{\ \Delta w^\star = -\frac{\|g\|^\dagger}{\lambda}\cdot \mathrm{dualize}_{\|\cdot\|}(g)\ }$$

其中对偶映射定义为

$$\mathrm{dualize}_{\|\cdot\|}(g) = \arg\max_{\|t\|=1}\langle g,t\rangle$$

**这个结果看似"废话文学"（因为它就是用定义写的），但它把优化器的设计空间完全显式化了：**

```text
模型无关的公共部分          优化器真正能设计的部分
──────────────────         ──────────────────────
模长  ‖g‖† / λ      ×      方向  dualize(g)
（实际由学习率接管）           ↑ 换范数 = 换优化器
```

### 2.5 关于步长

最优步长 $\|g\|^\dagger/\lambda$ 在实践中通常**不使用**——步长由学习率、schedule 等外部机制控制。所以分析的重点全部落在**对偶映射**上。

---

## 三、把"对偶"这个词讲清楚

第 19 行的公式里"对偶"出现了两次：$\|g\|^\dagger$ 叫**对偶范数**，$\mathrm{dualize}$ 叫**对偶映射**。这个词之所以显得玄，是因为它其实同时指**三个层层依赖的概念**；拆开看就清楚了。

| 层次 | 对偶的对象 | 记法 | 一句话 |
|---|---|---|---|
| ① 空间对偶 | 向量空间 $V$ | $V^*$ | $V$ 上全体**线性泛函**构成的空间 |
| ② 范数对偶 | 范数 $\|\cdot\|$ | $\|\cdot\|^\dagger$ | $V^*$ 上"配套"的那个范数 |
| ③ 映射对偶 | 方向 | $\mathrm{dualize}$ | 把泛函送回向量：$V^* \to V$ |

依赖关系是**层层往下**的：先有配对定义出 $V^*$，再由 $\|\cdot\|$ 在 $V^*$ 上诱导出 $\|\cdot\|^\dagger$，最后"取极值点"给出 $\mathrm{dualize}$。

### 3.1 空间对偶：泛函住在另一个空间（$V^*$）

**核心机制只有一个：配对（pairing）** $\langle\cdot,\cdot\rangle : V^*\times V \to \mathbb{R}$。

一个对象要"吃掉"$V$ 里的向量并吐出一个数，它就属于 $V^*$：

- 参数 $W$、更新量 $\Delta w$ 住在 $V$（因为要和 $W$ 相加）；
- 梯度 $g = \nabla_W L$ 的作用方式是"输入一个方向 $v$，输出该方向的方向导数"$\langle g,v\rangle$ —— 这正是**线性泛函**的定义，所以 $g$ 住在 $V^*$。

最具体的图像就是**行向量 vs 列向量**：

```text
     g ∈ V*（行向量）  [ g₁  g₂ ] · [ v₁ ]  =  标量 ⟨g, v⟩
                                   [ v₂ ]
     v ∈ V （列向量）
```

**"对偶对象 = 专门作用在它身上的东西"**，这就是 dual 最原始的语义。有限维时 $\dim V^* = \dim V$，且 $(V^*)^* = V$ —— **对偶的对偶回到自己**，这是对偶性最本质的特征，下面每一层都会有它的版本。

### 3.2 范数对偶：极集与支撑函数

$V$ 上有了范数 $\|\cdot\|$ 之后，$V^*$ 上的元素该怎么量大小？标准（也是唯一自然的）答案是

$$\boxed{\ \|g\|^\dagger := \max_{\|t\|=1}\langle g,t\rangle \;=\; \max_{\|t\|=1}|\langle g,t\rangle|\ }$$

（第二个等号用了范数球的对称性：$\|t\|=1 \Rightarrow \|-t\|=1$。）

**为什么写成 max？** 因为这样才能让广义 Cauchy–Schwarz

$$|\langle g,t\rangle| \le \|g\|^\dagger\,\|t\|,\qquad \forall t$$

成立，而且常数 $\|g\|^\dagger$ **最优、不能再小**（取 $t$ 为达到 max 的那个单位向量即取等号）。

> **操作性含义**：$\|g\|^\dagger$ 是使 $|\langle g,t\rangle|\le C\|t\|$ 恒成立的最小常数 $C$。它回答的不是"$g$ 有多大"，而是"**$g$ 最多能从单位球里榨出多少**"。

**几何图像：极集（polar）。** 把 $\|g\|^\dagger\le1$ 展开：

$$\|g\|^\dagger \le 1 \iff \langle g,t\rangle \le 1,\quad \forall\, t \text{ with } \|t\|\le1$$

这正是 $V$ 的单位球 $B$ 的**极集** $B^\circ$。所以 $B^\circ$ 恰好是 $\|\cdot\|^\dagger$ 的单位球：

```text
      ℓ₁ 球（二维菱形）              ℓ∞ 球（二维正方形）= 它的极集

             ▲                            ┌─────────┐
            ╱ ╲                           │         │
           ◀   ▶           ⟷              │         │
            ╲ ╱                           │         │
             ▼                            └─────────┘

       |t₁| + |t₂| ≤ 1             max(|t₁|, |t₂|) ≤ 1
```

完整对照：

| $\|\cdot\|$ 的单位球 | 极集 $B^\circ$（= $\|\cdot\|^\dagger$ 的单位球） |
|---|---|
| $\ell_2$ 球 | $\ell_2$ 球（**自对偶**） |
| $\ell_1$ 球（二维菱形） | $\ell_\infty$ 球（正方形） |
| $\ell_\infty$ 球（正方形） | $\ell_1$ 球（菱形） |
| 谱范数球 $\{t:\sigma_1(t)\le1\}$ | 核范数球 $\{g:\sum_i\sigma_i(g)\le1\}$ |

**"菱形 ↔ 正方形"是这一节最值得记住的一对**（二维可画、高维也成立）。极集还能再取一次：$B^{\circ\circ}=B$（单位球是闭凸集）—— 又一次"回到自己"。

**与层次 ① 的连通：对偶范数就是算子范数。** 把 $g\in V^*$ 看成线性映射 $g: (V,\|\cdot\|)\to(\mathbb{R},|\cdot|)$，则

$$\|g\|^\dagger = \sup_{t\ne0}\frac{|g(t)|}{\|t\|}$$

**这就是算子范数。** 所以"对偶范数"不是新东西，只是"把泛函当算子看"的算子范数。本笔记里谱范数与核范数的关系正是这个机制的一个实例：

$$\|g\|_{2\to2} = \sigma_1(g),\qquad \|g\|^\dagger_{2\to2} = \sum_i\sigma_i(g)$$

"最大奇异值 ↔ 奇异值之和"与"最大分量 ↔ 绝对值之和"是**同一机制的两种尺度**。

### 3.3 对偶映射：在球上取支撑点

层次 ② 给的是**数值**，层次 ③ 给的是**方向**：

$$\mathrm{dualize}_{\|\cdot\|}(g) \;=\; \arg\max_{\|t\|=1}\langle g,t\rangle$$

几何上就是：**把 $g$ 当作法向量，在单位球上找那个"最支撑"的点。**

| 范数 | 单位球的性格 | 支撑点（对偶映射） | 对应优化器 |
|---|---|---|---|
| $\ell_2$ | 光滑，无棱角 | $g/\|g\|_2$（唯一） | SGD |
| $\ell_\infty$ | 超立方体，极值在**顶点** | $\mathrm{sign}(g)$ | SignSGD |
| 谱范数 | $\mathrm{conv}\{uv^\top\}$，极值在**极端点** | $UV^\top$ | Shampoo / Muon |

这三行就是 SGD / SignSGD / Muon 差异的全部来源。

**三条自洽性检查**（用来判断有没有算对）：

1. **配对取到最大**：$\langle g,\ \mathrm{dualize}(g)\rangle = \|g\|^\dagger$；
2. **落在球边界上**：$\|\mathrm{dualize}(g)\| = 1$；
3. **对偶的对偶回到自己**：$(\|\cdot\|^\dagger)^\dagger = \|\cdot\|$。

以 $\ell_\infty$ 验证：$\langle g,\mathrm{sign}(g)\rangle = \sum_i|g_i| = \|g\|_1 = \|g\|^\dagger_\infty$ ✓，$\|\mathrm{sign}(g)\|_\infty = 1$ ✓。

### 3.4 为什么叫"对偶"，而不是"归一化"

因为它满足两个只有对偶才有的性质：

**（1）与转置（伴随）的范数守恒。** 对偶性的经典原型是"算子与它的伴随"：对 $A : (V,\|\cdot\|_a)\to(W,\|\cdot\|_b)$，有 $A^\top : (W,\|\cdot\|_b^\dagger)\to(V,\|\cdot\|_a^\dagger)$ 且

$$\|A\|_{a\to b} \;=\; \|A^\top\|_{b^\dagger\to a^\dagger}$$

例：$\|A\|_{2\to2} = \|A^\top\|_{2\to2} = \sigma_1$；$\|A\|_{1\to\infty} = \|A^\top\|_{1\to\infty} = \max_{ij}|a_{ij}|$。**"转置"就是对偶的具体化身。**

**（2）对偶的对偶回到自己。** $V^{**}=V$、$B^{\circ\circ}=B$、$(\|\cdot\|^\dagger)^\dagger = \|\cdot\|$。一个操作做两次回到原点，才算"对偶"：$\ell_1 \leftrightarrow \ell_\infty$ 成对，$\ell_2$ 与自己成对。

**（3）与 Lagrangian 对偶同源。** 把第 19 行的式子改写成硬约束形式

$$\min_{\Delta w}\ \langle g,\Delta w\rangle \quad \text{s.t.}\quad \|\Delta w\| \le r$$

它的解恰是 $\Delta w = -r\,\mathrm{dualize}(g)$ —— 与二次阻尼版本共享**同一个方向**，只在模长上通过 $r = \|g\|^\dagger/\lambda$ 换算；此时 $\lambda$ 扮演的就是约束的 **Lagrange 乘子**。**"对偶范数"与"Lagrangian 对偶"共用同一个配对 $\langle\cdot,\cdot\rangle$**，这正是本笔记 Related 里连到 [Lagrangian 与约束优化](../../../mathematics/lagrangian-and-constrained-optimization.md) 的原因——名字相同，底层机制（配对 + 极值）也相同。

### 3.5 为什么不能直接减梯度（原文论述 + 一点严谨性补充）

> **严谨性补充（本人添加，非原文内容）**：在有限维、且固定了内积（这里是 Frobenius 内积）之后，Riesz 表示定理给出 $V\cong V^*$，**所以 $W-\eta g$ 在代数上并不是非法的**。需要精确化的是：**"该用哪个等同"取决于你选的范数**。内积诱导的那个等同只是众多等同中的一种；当声明"要在谱范数几何下做最速下降"时，参数空间带的已经是**另一个范数**，正确的搬运规则是**范数的对偶映射**，而不是内积代表元。
>
> 两者关系可以一句话说清：**范数对偶映射 = 内积诱导的等同，当且仅当这个范数就是内积诱导的范数（$\ell_2$）**。于是
>
> - $\ell_2$ 几何下，直接减梯度就是对的 —— 这解释了 SGD 为什么"看起来天经地义"；
> - 换成 $\ell_\infty$ 或谱范数后，直接减梯度就变成"用 $\ell_2$ 的答案回答 $\ell_\infty$ 的问题"——**方向也许对，形状完全错**（$g$ 与 $\mathrm{sign}(g)$、$g$ 与 $UV^\top$ 的差别就在这里）。

**以下是原文论述，其中"类型错误"一词请按上面的补充理解：**

这是全文最有洞察力的一步。

**参数与梯度不在同一个空间。**

- 参数 $W$ 住在向量空间 $V$；
- 梯度 $g = \nabla_W L$ 本质是一个**线性泛函** $f : V \to \mathbb{R}$，$f(v) = \langle g, v\rangle$（输入一个方向，输出该方向的方向导数）。它住在 $V$ 的**对偶空间** $V^*$ 里。

所以把两者直接相减在数学上是**类型错误**：

```text
weight - LR * weight.grad            type error!
weight - LR * dualize(weight.grad)   all good!
```

**对偶映射的作用就是 $V^* \to V$**：根据对偶空间里的梯度，找到参数所在向量空间中的最速下降方向。

> 我的理解：这就是为什么"对偶映射"这个词必须出现，而不是简单说"换个方向"。它不是在 $V$ 里挑方向，而是做了一次**空间之间的搬运**。这也解释了为什么不同范数会给出形状完全不同的更新（$\mathrm{sign}$ vs $UV^\top$）——它们是不同的"搬运规则"。

---

## 四、不同范数 → 不同对偶映射 → 不同优化器

### 4.1 对偶范数的定义与计算

$$\|g\|^\dagger = \max_{t \in V,\ \|t\|=1}\langle g,t\rangle$$

三个最基本的对偶关系（原文说"证明留做习题"，这里补上）：

**① $\ell_2$ 的对偶是 $\ell_2$**

$$\max_{\|t\|_2 = 1} g^\top t = \|g\|_2$$

由 Cauchy–Schwarz，等号在 $t = g/\|g\|_2$ 时成立。

**② $\ell_1$ 的对偶是 $\ell_\infty$**

$$\max_{\|t\|_1 = 1} \sum_i g_i t_i = \max_i |g_i| = \|g\|_\infty$$

取 $t = \mathrm{sign}(g_i)\,e_i$（把全部质量压到最大的那个分量上），即得。它是一个线性规划：单形的极点在坐标轴方向上。

**③ $\ell_\infty$ 的对偶是 $\ell_1$**

$$\max_{\|t\|_\infty = 1} \sum_i g_i t_i = \sum_i |g_i| = \|g\|_1$$

取 $t_i = \mathrm{sign}(g_i)$（超立方体的顶点）即得。

> 规律：$\ell_p$ 与 $\ell_q$ 对偶 $\iff$ $1/p + 1/q = 1$（$p,q \ge 1$）。$\ell_2$ 自对偶，$\ell_1 \leftrightarrow \ell_\infty$。

### 4.2 汇总表

| 约束范数 $\|\cdot\|$ | 对偶范数 $\|\cdot\|^\dagger$ | 对偶映射 $\mathrm{dualize}(g)$ | 对应优化器 |
|---|---|---|---|
| $\ell_2$ | $\ell_2$ | $g / \|g\|_2$ | **SGD** |
| $\ell_\infty$ | $\ell_1$ | $\mathrm{sign}(g)$ | **SignSGD / AdamW（关动量）** |
| 谱范数 $\|\cdot\|_{2\to2}$ | 核范数 $\|\cdot\|_*$ | $\mathrm{msign}(g) = UV^\top$ | **Shampoo / Muon** |

### 4.3 SGD 的推导

$$\mathrm{dualize}_{\ell_2}(g) = \arg\max_{\|t\|_2=1}g^\top t = \frac{g}{\|g\|_2}$$

所以 $\Delta w^\star \propto -g$：**$\ell_2$ 约束下的最速下降就是沿梯度反方向，即 SGD。**

### 4.4 SignSGD 的推导

$$\mathrm{dualize}_{\ell_\infty}(g) = \arg\max_{\|t\|_\infty=1}\sum_i g_i t_i = \mathrm{sign}(g)
\quad\text{（逐元素）}$$

即 **$\ell_\infty$ 约束下的最速下降是 SignSGD**。（每个分量的可行区间是 $[-1,1]$，各自独立取端点即可。）

### 4.5 Adam 关掉动量后就是 SignSGD

原 Adam 更新：

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,\qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2,\qquad
W_t = W_{t-1} - \eta\frac{\hat m_t}{\sqrt{\hat v_t}}$$

其中 $\hat m_t = m_t/(1-\beta_1^t)$、$\hat v_t = v_t/(1-\beta_2^t)$ 是偏差修正。

**关掉动量**即 $\beta_1 = \beta_2 = 0$，此时（逐步代入）：

$$m_t = g_t \;\Rightarrow\; \hat m_t = \frac{g_t}{1 - 0^t} = g_t$$

$$v_t = g_t^2 \;\Rightarrow\; \hat v_t = g_t^2$$

注意 $\beta^t = 0$ 当 $t\ge1$，所以偏差修正因子是 1。于是

$$W_t = W_{t-1} - \eta\frac{g_t}{\sqrt{g_t^2}} = W_{t-1} - \eta\,\mathrm{sign}(g_t)$$

（逐元素，且 $\sqrt{g_t^2} = |g_t|$。）**因此 Adam 同样归入 $\ell_\infty$ 范数下的最速下降。**

> 我的理解：这个结论说明 Adam 的"自适应"在关动量后只是逐元素符号。真正带来"每个坐标不同步长"的是逐元素范数结构——它在数学上等价于用一个**各向异性（对角）的 $\ell_\infty$** 去约束更新。

---

## 五、谱范数下的对偶映射 = msign（完整推导）

这是全文最核心的一段推导。

### 5.1 定义

记 $g$ 的 SVD 为 $g = U\Sigma V^\top$，其中 $U \in \mathbb{R}^{n\times r}$、$V \in \mathbb{R}^{m\times r}$ 列正交，$\Sigma = \mathrm{diag}(\sigma_1,\dots,\sigma_r)$，$\sigma_i \ge 0$，$r = \mathrm{rank}(g)$。

矩阵的**谱范数**（$2\to2$ 算子范数）就是最大奇异值：

$$\|g\|_{2\to2} = \max_{\|x\|_2=1}\|gx\|_2 = \sigma_1(g)$$

**msign 的定义**：

$$\mathrm{msign}(g) = UV^\top$$

也就是**只保留左右奇异子空间、把所有非零奇异值统一换成 1**（奇异值非负，所以"符号操作"就是置 1）。

### 5.2 推导：把 $g$ 的 SVD 代入

目标：求 $\mathrm{dualize}_{\text{spectral}}(g) = \arg\max_{\|t\|_{2\to2}=1}\langle g,t\rangle$。

**第 1 步：内积写成 trace。**

$$\langle g,t\rangle = \mathrm{tr}(g^\top t)$$

**第 2 步：展开 $g^\top$ 的 SVD 形式。**

因为 $g = U\Sigma V^\top = \sum_i \sigma_i u_i v_i^\top$，所以

$$g^\top = \sum_i \sigma_i\, v_i u_i^\top
\quad\Longrightarrow\quad
\mathrm{tr}(g^\top t) = \mathrm{tr}\!\left(\sum_i \sigma_i\, v_i u_i^\top t\right) = \sum_i \sigma_i\,\mathrm{tr}(v_i u_i^\top t)$$

**第 3 步：用 $\mathrm{tr}(AB) = \mathrm{tr}(BA)$ 把标量提出。**

取 $A = v_i$、$B = u_i^\top t$，则 $\mathrm{tr}(v_i u_i^\top t) = \mathrm{tr}(u_i^\top t v_i)$。

而 $u_i^\top t\,v_i$ 是一个**标量**（$1\times n$ 乘 $n\times m$ 乘 $m\times 1$），标量的 trace 就是它自己：

$$\mathrm{tr}(g^\top t) = \sum_i \sigma_i\, u_i^\top t\, v_i$$

**第 4 步：逐项上界。**

$u_i^\top t v_i = \langle u_i,\ t v_i\rangle$（内积），由 Cauchy–Schwarz 且 $\|u_i\|_2 = 1$：

$$|u_i^\top t v_i| \le \|u_i\|_2\,\|t v_i\|_2 = \|t v_i\|_2$$

再用算子范数的定义 $\|t v_i\|_2 \le \|t\|_{2\to2}\|v_i\|_2 = \|t\|_{2\to2}\cdot 1 = 1$（在约束 $\|t\|_{2\to2}=1$ 下），得

$$u_i^\top t v_i \le 1$$

**第 5 步：求和。**

$$\langle g,t\rangle = \sum_i \sigma_i\, u_i^\top t\,v_i \le \sum_i \sigma_i \cdot 1 = \sum_i \sigma_i = \|g\|_*$$

（最后一步用到了**核范数** $\|g\|_* = \sum_i \sigma_i$ 的定义。）

### 5.3 验证等号可取

取 $t^\star = UV^\top = \sum_j u_j v_j^\top$。则对每个 $i$：

$$t^\star v_i = \left(\sum_j u_j v_j^\top\right) v_i = \sum_j u_j \underbrace{(v_j^\top v_i)}_{=\ \delta_{ij}} = u_i$$

（用到 $V$ 的列正交性 $v_j^\top v_i = \delta_{ij}$。）于是

$$u_i^\top t^\star v_i = u_i^\top u_i = 1 \quad(\|u_i\|_2 = 1)$$

同时 $\|UV^\top\|_{2\to2} = 1$（其全部非零奇异值都是 1），满足约束。因此

$$\boxed{\ \mathrm{dualize}_{\|\cdot\|_{\text{spectral}}}(g) = UV^\top = \mathrm{msign}(g)\ }$$

**顺带得到一个副产品**：由第 5 步，谱范数的对偶范数是**核范数**

$$\|g\|^\dagger_{\text{spectral}} = \|g\|_* = \sum_i \sigma_i$$

这解释了 [Bi-Maxwell](./bimaxwell-muon-physical-response.md) 里出现的 $-\dot\Phi_{\max} = c\|X\|_*$ 为什么长这样：**约束是谱范数时，"能榨出的最大下降速率"就是核范数**（即奇异值之和）。

### 5.4 为什么 msign 出现在 Shampoo / Muon 里

Shampoo 的更新是

$$W_{t+1} = W_t - \eta\left(L_t^{-1/4}G_tR_t^{-1/4}\right),\qquad
L_t = \textstyle\sum G G^\top,\quad R_t = \textstyle\sum G^\top G$$

关掉动量时，$L_t$ 与 $G_t$ 的左右奇异子空间一致：若 $G = U\Sigma V^\top$，则 $L \propto U\Sigma^2U^\top$、$R \propto V\Sigma^2V^\top$，于是

$$L^{-1/4}GR^{-1/4} = U\Sigma^{-1/2}\,U^\top\cdot U\Sigma V^\top\cdot V\Sigma^{-1/2}V^\top = UV^\top = \mathrm{msign}(G)$$

**所以 Shampoo 早就在做 msign 了**（2018 年）。Muon 的贡献不是"发现了 msign"，而是**找到了不用开四次方根就能算出 msign 的工程方法**（见 [Muon 优化器 §3](./muon-optimizer.md)）。

---

## 六、常见误区

1. **"最速下降就是沿负梯度方向"** → 只有约束取 $\ell_2$ 时才对。换范数就换方向（$\ell_1$ / $\ell_\infty$ / 谱范数给的方向都不同）。
2. **"不同优化器的差别在于步长/动量"** → 更本质的差别是**对偶映射**（约束范数）。步长只是 $\|g\|^\dagger/\lambda$ 的替代品。
3. **"梯度就是参数空间里的一个向量"** → 错。梯度住在对偶空间 $V^*$；必须先经对偶映射才能与参数相加。
4. **"msign 是某种魔法正交化"** → 它就是**谱范数下的对偶映射**，由 $\arg\max_{\|t\|_{2\to2}=1}\langle g,t\rangle$ 唯一确定。
5. **"$\mathrm{sign}$ 和 $\mathrm{msign}$ 是两个无关的技巧"** → 它们是同一公式在 $\ell_\infty$ 与谱范数下的两个实例。$\mathrm{msign}$ 是 $\mathrm{sign}$ 的矩阵版：**奇异值的 sign**。
6. **"放大小奇异值是为了保护长尾信息"** → 这是过度解读。从最速下降看，把奇异值压平只是为了**榨干谱范数约束下允许的最大更新量**。而且 AdamW/SignSGD 同样放大了小值，用"长尾信息"无法解释为什么它们不行。

## 我的理解

（以下为个人理解）

- 这个框架真正漂亮的地方是**把"方向"和"幅度"彻底解耦**：$\mathrm{dualize}(g)$ 决定方向（几何问题），$\|g\|^\dagger/\lambda$ 决定幅度（标量问题）。所以换优化器 = 换几何，而学习率 = 幅度控制，两者正交。
- 对偶映射 $\mathrm{dualize} : V^* \to V$ 之所以"像"一个归一化，是因为它要在**约束球的边界**上取极值——球不同，极点就不同：$\ell_2$ 球的极点是圆滑的（法向 = 梯度方向），$\ell_\infty$ 球的极点在顶点（符号），谱范数球的极点是 $\{uv^\top\}$ 的凸包（$UV^\top$ 的直觉来源）。
- 一个自洽的检查方式：把 $\mathrm{dualize}(g)$ 代回，应当有 $\langle g, \mathrm{dualize}(g)\rangle = \|g\|^\dagger$。对谱范数即 $\langle g, UV^\top\rangle = \mathrm{tr}(V\Sigma U^\top\cdot UV^\top) = \mathrm{tr}(V\Sigma V^\top) = \mathrm{tr}(\Sigma) = \sum\sigma_i$ ✓。

## Related

- [Muon 优化器](./muon-optimizer.md) — 算法总览与 [Newton–Schulz 代数推导](./muon-optimizer.md)
- [谱范数与 RMS 几何：为什么 Muon 用算子范数](./spectral-norm-rms-geometry.md) — 本篇框架在 Muon 上的落地：为什么选谱范数、AdamW 的几何缺陷
- [Bi-Maxwell：Muon 的物理响应与双时间尺度动量](./bimaxwell-muon-physical-response.md) — 核范数上界与 Polar 方向的另一种条件化推导
- [平滑矩阵 Polar 谱梯度流](./smoothed-matrix-polar-spectral-gradient-flows.md) — 连续时间视角
- [Lagrangian 与约束优化](../../../mathematics/lagrangian-and-constrained-optimization.md) — 等式约束下的极值方法论

## References

- 知乎《Muon优化器科普，但从最速下降的本质出发》，作者 resnet-65536，2025-11-10：<https://zhuanlan.zhihu.com/p/1954634867791869927>
- Jeremy Bernstein, Laker Newhouse. *Old Optimizer, New Norm: An Anthology*. [arXiv:2409.20325](https://arxiv.org/abs/2409.20325)
- Jeremy Bernstein. *Deriving Muon*: <https://jeremybernste.in/writing/deriving-muon>
- Laker Newhouse. *Duality, Weight Decay, and Metrized Deep Learning*: <https://www.lakernewhouse.com/thesis.pdf>
- Vineet Gupta, Tomer Koren, Yoram Singer. *Shampoo: Preconditioned Stochastic Tensor Optimization*, ICML 2018.
- 范数基础（定义、性质、边界图像）：<https://zhuanlan.zhihu.com/p/671815885>
- 本笔记的推导中间步骤（第 2.2、4.1、5.2 节）为对照原文补齐，非原文逐字内容。
- 第 三 节（三层对偶、极集/支撑函数几何、对偶映射与自洽性、与 Lagrangian 对偶的同源关系）为本人整理与补充，非原文内容；其中 §3.5 的"严谨性补充"是对原文"类型错误"这一说法的精确化，原文论述已在该节完整保留。
