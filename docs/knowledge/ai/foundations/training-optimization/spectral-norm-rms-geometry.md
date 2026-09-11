---
title: 谱范数与 RMS 几何：为什么 Muon 用算子范数，而 AdamW 的几何不匹配
type: concept
status: seed
tags: [AI, optimization, spectral-norm, operator-norm, RMS, AdamW, Muon, geometry]
created: 2026-09-11
updated: 2026-09-11
source: 知乎《Muon优化器科普，但从最速下降的本质出发》（https://zhuanlan.zhihu.com/p/1954634867791869927）+ Bernstein《Deriving Muon》+ 本人补齐的推导中间步骤
---

# 谱范数与 RMS 几何：为什么 Muon 用算子范数，而 AdamW 的几何不匹配

> 前置：[最速下降的范数对偶框架](./steepest-descent-duality-map.md)。本篇回答的是框架里的**自由参数怎么选**：为什么是谱范数（更准确说 RMS→RMS 算子范数），以及 AdamW 的隐式几何为什么与网络的前向传播不匹配。

## 一句话理解

线性层 $y = Wx$。我们真正想控制的是**激活值的幅度**（用 RMS 范数度量），于是"给定输入 RMS，限制输出的最大可能增幅"这件事，对应的几何恰好是

$$\|W\|_{\mathrm{RMS}\to\mathrm{RMS}} = \sqrt{\frac{\text{fan-in}}{\text{fan-out}}}\,\|W\|_{2\to2}$$

也就是**谱范数乘一个维度因子**。所以约束 $\|\Delta W\|_{\mathrm{RMS}\to\mathrm{RMS}}$ = 约束谱范数，等价于**约束"最坏情况下输出的 RMS 变化"**。

而 AdamW（关动量即 SignSGD）的隐式约束是 $\ell_1 \to \ell_\infty$，它把输出空间当 $\ell_\infty$、输入空间当 $\ell_1$，**层与层之间几何类型必须来回切换**，与"各层 RMS 稳定"的自然几何不一致。

## 为什么重要

- 它给出了"为什么是 Muon"的**可论证理由**，而不是"正交化梯度看起来很酷"。
- RMS 范数的**维度不敏感**性质，直接解释了 Muon 超参为什么有强尺度不变性（可跨网络宽度迁移）。
- 它同时给出了 AdamW 的局限的**计算层面**证据（$\ell_\infty\to\ell_1$ 是 NP-hard），而不只是经验对比。

---

## 一、RMS 范数

对向量 $v \in \mathbb{R}^d$：

$$\|v\|_{\mathrm{RMS}} = \sqrt{\frac{1}{d}\sum_{i=1}^d v_i^2} = \frac{1}{\sqrt{d}}\|v\|_2$$

**为什么用 RMS 而不是 $\ell_2$？** 原文给出两点：

1. **它度量"平均意义下"的幅度**，能直接反映每个分量的典型绝对值大小。大模型常用 SiLU 等激活函数，在 0 附近非线性最强，把激活幅度控制在适中区间有利于发挥表达能力。
2. **它对维度不敏感**：所有分量都为 1 时，无论 $d$ 多大，$\|v\|_{\mathrm{RMS}} = 1$。因此按 RMS 做约束的优化器，其最优超参可以直接在不同宽度的网络间迁移。

> 注意第 2 点是纯数学性质，直接读定义就能验证：$\|(1,\dots,1)\|_{\mathrm{RMS}} = \sqrt{\frac{1}{d}\cdot d} = 1$。$\ell_2$ 范数下这个值是 $\sqrt d$，所以 $\ell_2$ 约束的超参会随宽度漂移。

## 二、算子范数

对任意向量空间范数 $\|\cdot\|_A : A \to \mathbb{R}$ 与 $\|\cdot\|_B : B \to \mathbb{R}$，线性算子 $W : A \to B$ 的**算子范数**（诱导范数）定义为

$$\|W\|_{A\to B} = \max_{\|x\|_A = 1}\|Wx\|_B = \sup_{x \ne 0}\frac{\|Wx\|_B}{\|x\|_A}$$

（第二个等号对线性算子成立：把 $\|x\|_A$ 提出来即可。）

它回答的问题是：**给定输入范数不超过 1，输出范数最大能被放大到多少。**

## 三、RMS → RMS 算子范数 = 维度因子 × 谱范数

设 $W \in \mathbb{R}^{n_{out}\times n_{in}}$，输入 $x \in \mathbb{R}^{n_{in}}$、输出 $y = Wx \in \mathbb{R}^{n_{out}}$。记 $\text{fan-in} = n_{in}$、$\text{fan-out} = n_{out}$。

**推导**（把 RMS 的定义代进去）：

$$
\begin{aligned}
\|W\|_{\mathrm{RMS}\to\mathrm{RMS}}
&= \sup_{x\ne0}\frac{\|Wx\|_{\mathrm{RMS}}}{\|x\|_{\mathrm{RMS}}}
= \sup_{x\ne0}\frac{\dfrac{1}{\sqrt{n_{out}}}\|Wx\|_2}{\dfrac{1}{\sqrt{n_{in}}}\|x\|_2}
= \sqrt{\frac{n_{in}}{n_{out}}}\sup_{x\ne0}\frac{\|Wx\|_2}{\|x\|_2}
= \sqrt{\frac{n_{in}}{n_{out}}}\,\|W\|_{2\to2}
\end{aligned}
$$

即

$$\boxed{\ \|W\|_{\mathrm{RMS}\to\mathrm{RMS}} = \sqrt{\frac{\text{fan-in}}{\text{fan-out}}}\;\|W\|_{2\to2}\ }$$

而 $\|W\|_{2\to2} = \sigma_{\max}(W)$ 就是谱范数（最大奇异值）。

**结论**：RMS→RMS 算子范数与谱范数**只差一个维度因子**。所以"控制 RMS 几何下的更新量" ⟺ "控制谱范数（带缩放系数）"。

> 我的理解：这个 $\sqrt{n_{in}/n_{out}}$ 因子不是可有可无的装饰——它正是"为什么 Muon 要做 shape scaling"的几何来源，也解释了为什么缩放必须依赖矩阵形状而不是一个常数。

## 四、输出扰动的上界

对权重更新 $\Delta W$，输出的扰动是 $\Delta y = \Delta W x$（因为 $y = Wx$ 是线性的，$W \to W + \Delta W$ 时 $\Delta y = \Delta W x$）。由算子范数的定义：

$$\|\Delta y\|_{\mathrm{RMS}} = \|\Delta W x\|_{\mathrm{RMS}} \le \|\Delta W\|_{\mathrm{RMS}\to\mathrm{RMS}}\cdot\|x\|_{\mathrm{RMS}}$$

**推导**：

$$\|\Delta W x\|_{\mathrm{RMS}} = \|x\|_{\mathrm{RMS}}\cdot\underbrace{\frac{\|\Delta W x\|_{\mathrm{RMS}}}{\|x\|_{\mathrm{RMS}}}}_{\le\ \sup_{x'\ne0}\frac{\|\Delta W x'\|_{\mathrm{RMS}}}{\|x'\|_{\mathrm{RMS}}} = \|\Delta W\|_{\mathrm{RMS}\to\mathrm{RMS}}}$$

**于是优化器的目标被明确成一句话：只要控制住 $\|\Delta W\|_{\mathrm{RMS}\to\mathrm{RMS}}$，就能控制住每层输出 RMS 的增幅上界。** 这就是把"训练稳定性"翻译成了一个范数约束。

---

## 五、AdamW 的隐式几何：$\ell_1 \to \ell_\infty$

### 5.1 计算 $\ell_1\to\ell_\infty$ 诱导范数

$$
\begin{aligned}
\|\Delta W\|_{1\to\infty}
&= \max_{\|x\|_1 = 1}\|\Delta W x\|_\infty \\
&= \max_{\|x\|_1 = 1}\ \max_i\left|\sum_j \Delta W_{ij}x_j\right| \\
&= \max_i\ \max_{\|x\|_1 = 1}\left|\sum_j \Delta W_{ij}x_j\right| \qquad(\text{两个 }\max\text{ 可交换}) \\
&= \max_i\ \|\mathrm{row}_i(\Delta W)\|_\infty \qquad(\text{逐行：}\ell_1\text{ 的对偶是 }\ell_\infty) \\
&= \max_i\ \max_j |\Delta W_{ij}| \\
&= \|\mathrm{flatten}(\Delta W)\|_\infty
\end{aligned}
$$

**关键观察**：$\|\cdot\|_{1\to\infty}$ 的**单位球恰好是逐元素的立方体**：

$$\{W : \|W\|_{1\to\infty} \le 1\} = \{W : |W_{ij}| \le 1,\ \forall i,j\}$$

### 5.2 该范数下的对偶映射就是 sign

在该单位球上最大化 $\langle g, \Delta W\rangle_F = \sum_{ij} g_{ij}\Delta W_{ij}$，各分量独立，各自取端点：

$$\Delta W_{ij} = \mathrm{sign}(g_{ij})$$

**所以：给定 $\langle g,\Delta w\rangle$ 与约束 $\max(\Delta W) \le \eta$，最优下降方向就是 $-\eta\,\mathrm{sign}(g)$。**

这与 [最速下降框架](./steepest-descent-duality-map.md) 里 $\ell_\infty$ 情形的结论一致——**SignSGD 等价于在 $\ell_1\to\ell_\infty$ 几何下的最速下降**。

### 5.3 由此推出的几何结构

```text
输入空间隐式采用 ℓ₁
        │
        ▼
   ΔW: ℓ₁ → ℓ∞
        │
        ▼
输出空间隐式采用 ℓ∞
```

## 六、为什么这个几何"不匹配"

### 6.1 层间需要反复切换几何类型

一层输出是下一层输入。若某层把**输出空间**当 $\ell_\infty$，那么下一层要与之匹配，其**输入空间**就得是 $\ell_\infty$——对应约束变成 $\ell_\infty \to \ell_1$。

于是层与层之间，几何类型必须在

$$\ell_1\to\ell_\infty \quad\longleftrightarrow\quad \ell_\infty\to\ell_1$$

之间来回切换。**这与前向传播的自然几何（我们希望各层保持 RMS norm 稳定）并不一致。**

### 6.2 而且切不过来：NP-hard

$\ell_\infty\to\ell_1$ 的诱导范数计算是 **NP-hard**（Tropp, 2004）。也就是说，"交替施加 $\ell_1\to\ell_\infty$ 与 $\ell_\infty\to\ell_1$ 两种算子范数约束"在**计算上不可行**——不是一个"工程上麻烦"的问题，而是没有多项式算法。

（直觉：$\|W\|_{\infty\to1} = \max_{\|x\|_\infty\le1}\sum_i|\sum_j W_{ij}x_j|$。逐行看，$\sum_j W_{ij}x_j$ 关于 $x$ 是线性的，而外层取绝对值再求和使它成为凸函数；在盒子上最大化凸函数必在顶点取到，于是问题退化成组合优化，与 max-cut 一类问题同源。）

### 6.3 对比结论

| | AdamW / SignSGD | Muon |
|---|---|---|
| 隐式范数 | $\ell_1\to\ell_\infty$ | RMS→RMS（$\propto$ 谱范数） |
| 输入空间 | $\ell_1$ | RMS |
| 输出空间 | $\ell_\infty$ | RMS |
| 层间几何 | 需要 $\ell_\infty\leftrightarrow\ell_1$ 切换 | **始终保持在 RMS 几何** |
| 计算可行性 | $\ell_\infty\to\ell_1$ 侧 NP-hard | 谱范数有高效近似（NS 迭代） |
| 超参迁移性 | $\ell_\infty$ 对维度敏感 | **RMS 尺度不变 → 超参可跨宽度迁移** |

**结论**：在谱范数约束下，对激活值的约束始终停留在 RMS 几何，从而获得尺度不变性；而 Adam 系列并没有很好地实现对激活值的约束。

---

## 七、对"正交化叙事"的一个反驳

原文有一段碎碎念，值得记录成一个判断：

> 很多故事把 Muon 的优势仅仅归结为"正交化梯度本身"，例如"Muon 放大了（相对大奇异值而言的）小奇异值中蕴含的长尾信息，让它更能记住长尾知识"。**这个说法的逻辑漏洞是明显的**：
> 1. 小奇异值对应的子空间**并不代表**里面装的就是长尾信息；
> 2. AdamW 等 SignSGD 类优化器**同样放大了更新量里的小值**，为什么它们就不行？

从最速下降的本质看：**放大小奇异值只是为了榨干谱范数约束下允许的最大更新量**，不宜过度解读。

---

## 常见误区

1. **"Muon 控制的是梯度的范数"** → 不准确。它约束的是**权重更新**的算子范数，进而控制**激活值的 RMS 变化上界**。
2. **"RMS 和 $\ell_2$ 是同一个东西换个名字"** → 差一个 $1/\sqrt d$ 因子，而这个因子正是维度不敏感性的来源。
3. **"$\ell_1\to\ell_\infty$ 只是个记号"** → 它的单位球就是逐元素立方体，这直接导致对偶映射退化成逐元素 $\mathrm{sign}$。
4. **"AdamW 的几何缺陷只是经验观察"** → 有计算复杂度层面的硬证据（$\ell_\infty\to\ell_1$ NP-hard）。
5. **"换 AdamW 的优化器就等于换 AdamW 的几何"** → 两者必须一起考虑：Muon 需要配套的学习率/shape scaling 才能对齐到目标 RMS 几何（见 [Muon 优化器](./muon-optimizer.md) 中 Kimi 的 RMS 对齐做法）。

## 我的理解

（以下为个人理解）

- 这篇推导的真正价值是**把"稳定性"变成了一个几何匹配问题**：我们希望每一层的输入输出都活在 RMS 这个参考系里，那么更新算子就应该是一个"保持该参考系"的算子。AdamW 的 $\ell_1\to\ell_\infty$ 让参考系在层间翻转，而谱范数（乘以维度因子后）恰好是 RMS→RMS 的"等距型"约束。
- 谱范数约束球在矩阵空间里的几何值得想一下：$\{W : \|W\|_{2\to2}\le1\}$ 是**所有秩一外积 $uv^\top$（$\|u\|=\|v\|=1$）的凸包**。它的极点就是秩一矩阵——所以极值总在"最像秩一"的地方取到。这也解释了为什么最优解会是 $UV^\top$ 这种"等奇异值"的形式。
- $\sqrt{\text{fan-in}/\text{fan-out}}$ 这个因子很容易被忽略，但它是**Muon shape scaling 的理论出处**：它不是调参经验，而是"把谱范数翻译成 RMS 几何"的汇率。

## Related

- [最速下降的范数对偶框架](./steepest-descent-duality-map.md) — 本篇的通用框架（对偶范数、对偶映射、msign 推导）
- [Muon 优化器](./muon-optimizer.md) — Newton–Schulz 代数推导与 Kimi 的 RMS 对齐/QK-Clip
- [Bi-Maxwell：Muon 的物理响应与双时间尺度动量](./bimaxwell-muon-physical-response.md) — 同样从"输出扰动预算"出发的条件化推导
- [平滑矩阵 Polar 谱梯度流](./smoothed-matrix-polar-spectral-gradient-flows.md) — 连续时间稳定性
- [Training Optimization](./) — 主题入口

## References

- 知乎《Muon优化器科普，但从最速下降的本质出发》，作者 resnet-65536，2025-11-10：<https://zhuanlan.zhihu.com/p/1954634867791869927>
- Jeremy Bernstein. *Deriving Muon*: <https://jeremybernste.in/writing/deriving-muon>
- Laker Newhouse. *Duality, Weight Decay, and Metrized Deep Learning*: <https://www.lakernewhouse.com/thesis.pdf>
- Joel A. Tropp. *Topics in Sparse Approximation*, PhD thesis, UT Austin, 2004（$\ell_\infty\to\ell_1$ 诱导范数的计算复杂度）
- 本笔记的推导中间步骤（第三、四、5.1、6.2 节）为对照原文补齐，非原文逐字内容。
