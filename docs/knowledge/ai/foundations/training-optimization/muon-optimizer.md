---
title: Muon 优化器
subtitle: 矩阵更新的动量与近似极分解
type: concept
status: seed
tags: [AI, deep-learning, optimization, Muon, Newton-Schulz, polar-decomposition]
created: 2026-09-06
updated: 2026-09-11
source:
  - https://kellerjordan.github.io/posts/muon/
  - https://github.com/KellerJordan/Muon
  - https://mp.weixin.qq.com/s/Gq7k9yDgDdVlT6NHoKJizQ
  - https://mp.weixin.qq.com/s/zapTq9YsHqAGLsymEt9gcQ?scene=1
  - 知乎《Muon优化器科普，但从最速下降的本质出发》 https://zhuanlan.zhihu.com/p/1954634867791869927 （Newton–Schulz 代数、Moonlight weight decay / RMS 对齐 / QK-Clip）
---

# Muon 优化器

> **证据状态：** Muon 的更新几何、Bi-Maxwell 的实验结论和谱梯度流的理论结论来自不同层级的资料，不能混为同一证明链。本文只保留 Muon 的算法总览；两篇相关论文分别记录在 [Bi-Maxwell 物理响应与双时间尺度动量](./bimaxwell-muon-physical-response.md) 和 [平滑矩阵 Polar 谱梯度流](./smoothed-matrix-polar-spectral-gradient-flows.md)，并明确各自的假设和外推边界。

## 一句话理解

Muon（MomentUm Orthogonalized by Newton-schulz）是面向**二维及更高维权重矩阵**的优化器：先对梯度做动量累积，再把该矩阵更新方向近似投影为半正交矩阵，最后沿这个尺度受控的方向更新参数。它不是“对所有参数都用 Muon”的完整训练配方，通常要与 AdamW 等优化器共同处理 embedding、bias、归一化参数和其他非矩阵参数。

## 它想解决什么问题

设某个矩阵参数为 $W \in \mathbb{R}^{m \times n}$，损失为 $\mathcal{L}$，其梯度为

$$
G_t = \nabla_W \mathcal{L}(W_t).
$$

普通 SGD 直接使用 $G_t$ 更新。矩阵梯度的奇异值可能高度不均衡：若

$$
G_t = U\Sigma V^\top,
$$

则直接更新会让大奇异值方向比小奇异值方向移动得更多。Muon 的基本几何直觉是：保留梯度的左右奇异向量 $U, V$ 所编码的方向结构，但不让各个非零奇异方向的更新幅度完全由 $\Sigma$ 决定。

这不是普适的“梯度幅度一定有害”的结论。奇异值缩放是否有用，取决于模型、参数类型、学习率、batch size、训练阶段和其他优化器超参数；应由对照实验判断。

## 算法骨架

### 1. 对梯度做动量

Muon 的名字指向先形成矩阵动量。以常见的动量形式为例：

$$
M_t = \beta M_{t-1} + (1-\beta)G_t,
$$

其中 $\beta$ 是动量系数。不同实现也可能采用不含 $(1-\beta)$ 的缩放约定；阅读代码时必须以实现的实际公式为准。

有些配方会先使用 Nesterov 型方向：

$$
\widetilde{M}_t = \beta M_t + (1-\beta)G_t.
$$

它改变的是送入正交化步骤的矩阵，而不是 Muon 的唯一必要定义。

### 2. 取最近的半正交方向

对满秩矩阵 $X \in \mathbb{R}^{m \times n}$，令其薄 SVD 为

$$
X = U\Sigma V^\top.
$$

极分解的极因子（polar factor）为

$$
\operatorname{polar}(X) = UV^\top.
$$

它在 Frobenius 范数意义下是最接近 $X$ 的半正交矩阵之一：

$$
\operatorname{polar}(X)
= \underset{Q}{\arg\min}\; \lVert X-Q \rVert_F,
$$

其中可行域取满足 $Q^\top Q=I$ 或 $QQ^\top=I$ 的适当 Stiefel 流形，取决于矩阵形状。直观地说，$UV^\top$ 保留 $X$ 的左右奇异子空间，同时将其非零奇异值替换为 $1$。

Muon 用这一变换作用于动量方向：

$$
O_t \approx \operatorname{polar}(\widetilde{M}_t).
$$

### 3. 近似极分解，而非每步 SVD

逐步做 SVD 的成本通常不适合大模型训练。常见实现用 Newton–Schulz 类迭代，只含矩阵乘法，近似计算极因子或矩阵“zeroth power”。概念上，在做过尺度归一化的 $X_0$ 上迭代形如

$$
X_{k+1} = X_k\left(aI + bX_k^\top X_k + c(X_k^\top X_k)^2 + \cdots\right),
$$

使 $X_k$ 的奇异值向 $1$ 收缩。

具体多项式系数、迭代次数、归一化方式、是否转置宽矩阵，以及低精度下的稳定性都是**实现细节**，不同 Muon 版本可能不同。它们不能由上式推断，应以所选代码版本和实验复现为准。

#### 为什么"多项式作用在矩阵上"等于"多项式作用在奇异值上"

这是 Newton–Schulz 能工作的代数根据，值得完整推一遍。

设 $G = U\Sigma V^\top$（$U \in \mathbb{R}^{n\times r}$、$V \in \mathbb{R}^{m\times r}$ 列正交，$\Sigma = \mathrm{diag}(\sigma_1,\dots,\sigma_r)$、$\sigma_i \ge 0$）。反复使用 $U^\top U = I_r$、$V^\top V = I_r$：

**① 二阶组合**

$$G^\top G = (U\Sigma V^\top)^\top(U\Sigma V^\top) = V\Sigma \underbrace{U^\top U}_{I}\Sigma V^\top = V\Sigma^2 V^\top$$

**② 由结合律逐步升幂**

$$G^\top G = V\Sigma^2 V^\top \;\Longrightarrow\; (G^\top G)^n = V\Sigma^{2n}V^\top$$

（因为 $(V\Sigma^2V^\top)^k (V\Sigma^2V^\top) = V\Sigma^{2k}\underbrace{V^\top V}_{I}\Sigma^2V^\top = V\Sigma^{2k+2}V^\top$，望远镜式消掉。）

**③ 与 $G$ 相乘**

$$G\,(G^\top G)^n = U\Sigma V^\top \cdot V\Sigma^{2n}V^\top = U\Sigma^{2n+1}V^\top$$

> **原文勘误**：知乎原文此处写作 $G^\top(G^\top G)^n = U\Sigma V^\top(V\Sigma^2V^\top)^n = U\Sigma^{2n+1}V^\top$，左端应为 $G\,(G^\top G)^n$（$G^\top$ 的分解是 $V\Sigma U^\top$，代入后得到的是 $V\Sigma^{2n+1}U^\top$，即转置）。右端的计算过程本身是对的，只是左端符号笔误。下面统一用正确的 $G\,(G^\top G)^n$。

**④ 代入五阶奇多项式**

取 $\varphi(X) = aX + bXX^\top X + cXX^\top XX^\top X$。逐项计算：

```text
XX^T      = UΣ²U^T
XX^T X    = (UΣ²U^T)(UΣV^T)            = UΣ³V^T        ← 对应 n = 1
XX^T XX^T X = (UΣ²U^T)(UΣ²U^T)(UΣV^T) = UΣ⁵V^T        ← 对应 n = 2
```

于是

$$\varphi(G) = a\,U\Sigma V^\top + b\,U\Sigma^3 V^\top + c\,U\Sigma^5 V^\top = U\,(a\Sigma + b\Sigma^3 + c\Sigma^5)\,V^\top = U\,\varphi(\Sigma)\,V^\top$$

**核心结论**：

$$\boxed{\ \varphi(G) = U\,\varphi(\Sigma)\,V^\top\ }$$

**多项式 $\varphi$ 只作用在奇异值上，左右奇异向量 $U,V$ 完全不变。** 因此整个 NS 迭代可以退化成 $r$ 个**独立的标量迭代**：

$$\sigma_i \longleftarrow \varphi(\sigma_i) = a\sigma_i + b\sigma_i^3 + c\sigma_i^5$$

#### 尺度归一化与收敛

迭代方案：

$$G_0 = \frac{G}{\|G\|_F},\qquad G_{t+1} = \varphi(G_t)$$

**为什么除以 Frobenius 范数能把奇异值压进 $[0,1]$？** 因为

$$\|G\|_F = \sqrt{\sum_i \sigma_i^2} \;\ge\; \max_i\sigma_i = \sigma_1$$

即 F 范数**不小于谱范数**，所以 $G_0$ 的奇异值 $\sigma_i/\|G\|_F \le 1$，且非负，落在 $[0,1]$。

此后标量迭代 $\sigma \mapsto a\sigma + b\sigma^3 + c\sigma^5$ 把 $[0,1]$ 内的奇异值推向 $1$，从而 $G_t \to UV^\top$，即正交矩阵。

**关于系数的两点说明**（原文说"这篇不讲"，这里只记录可验证的部分）：

1. 系数需要综合考虑不动点迭代的收敛条件、收敛速度与精度。
2. Keller Jordan 给出的 $(a,b,c) = (3.4445,\,-4.7750,\,2.0315)$，迭代 5 步。**他放松了误差要求，允许奇异值收敛到 $[0.7, 1.3]$ 区间**，经验上这个误差对训模型影响不大。

> **我的验算**：把 $\sigma = 1$ 代入得 $3.4445 - 4.7750 + 2.0315 = 0.7010 \ne 1$，可见 $\sigma = 1$ **不是精确不动点** —— 这正与"允许收敛到一个带，而不是精确的 1"相符。也就是说 Muon 实际拿到的是**近似半正交**矩阵，不是严格 $UV^\top$。这一点在评估近似误差时必须记住。

目前 SOTA 方案是 [Polar Express](https://arxiv.org/abs/2505.16932)，科学空间也有记载（<https://www.spaces.ac.cn/archives/10996>）。

### 4. 按形状缩放并更新

半正交矩阵的 Frobenius 范数会随秩而变。实际 Muon 配方通常对 $O_t$ 乘以与 $m/n$ 相关的缩放，再进行参数更新：

$$
W_{t+1} = W_t - \eta\,s(m,n)\,O_t.
$$

缩放 $s(m,n)$ 的精确定义、学习率 $\eta$、weight decay 的放置方式和参数分组同样是实现/配方的一部分。比较 Muon 与 AdamW 时，必须比较完整配方，而不仅是替换 `optimizer.step()` 中的一行。

## 一种受限的理论动机：输出扰动预算

Bi-Maxwell 论文给出了一种**条件化**的解释。将单层 $W$ 看作 $y=Wx$，令 $X=-\partial\Phi/\partial W$ 为损失势 $\Phi(W)$ 的负梯度，$V=\dot W$ 为瞬时更新速度。若要求任意输入方向上的最坏输出扰动满足

$$
A_{\mathrm{out}}(Vx) \le \varepsilon A_{\mathrm{in}}(x),\qquad \forall x,
$$

在该文使用的 RMS 通道归一化下，可以写成谱范数约束 $\lVert V\rVert_2\le c$。在该硬约束内最大化瞬时损失下降功率

$$
\max_{\lVert V\rVert_2\le c}\langle X,V\rangle_F
$$

会得到

$$
V^\star=c\operatorname{polar}(X),\qquad
-\dot\Phi_{\max}=c\lVert X\rVert_*.
$$

这说明：在**欧氏度量、各向同性、对所有输入成立的最坏情况硬上限**下，Polar 方向是预算内的最大瞬时耗散方向。它不是 Muon 无条件优于梯度下降的证明。若约束改为按实际 activation 分布的平均扰动，论文会导出类似 $V\propto XC^{-1}$ 的协方差预条件方向；再考虑下游敏感度则可能是 $V\propto S^{-1}XC^{-1}$。完整推导、记忆核与实验见 [Bi-Maxwell 物理响应与双时间尺度动量](./bimaxwell-muon-physical-response.md)。

## 为什么不直接沿原始梯度走

### 可以从定义得到的性质

对于 $X=U\Sigma V^\top$，极因子 $UV^\top$：

- 保留左、右奇异向量，即保留矩阵更新的主方向结构；
- 将非零奇异值变为 $1$，从而不再按原始梯度奇异值比例分配步长；
- 对秩亏或近奇异输入，结果及其数值近似需要额外小心。

因此，Muon 不是“丢掉梯度方向”，而是改变矩阵更新在不同奇异方向上的相对幅度。

### 需要经验验证的机制解释

“奇异值更均匀的更新会带来更快收敛、更好泛化或更稳定训练”是可检验假设，不是仅由极分解定义就能推出的定理。要验证它，至少应控制：

1. 模型、数据、训练 token 数、随机种子与精度；
2. 所有参数的优化器分组及 AdamW 的超参数；
3. Muon 的动量、Newton–Schulz 迭代、缩放、学习率与 weight decay；
4. 训练损失、验证指标、梯度/更新的奇异值谱、wall-clock 吞吐和峰值显存；
5. 是否因更大稳定学习率或更好 kernel 实现造成收益，而非正交化本身。

## 参数分组：Muon 不是全参数替代品

Muon 的核心操作面向矩阵。一个常见的研究问题是如何划分参数，而不是只问“用不用 Muon”：

| 参数类别 | 为什么需要单独判断 | 典型选择（需按实现确认） |
| --- | --- | --- |
| 线性层/投影层权重 | 通常是二维矩阵，适合讨论矩阵极因子 | Muon 候选 |
| 卷积核 | 可展平为矩阵，但展平方式改变几何含义 | 需要实验确认 |
| embedding 表 | 虽是矩阵，但稀疏访问、行语义和规模不同 | 常保留 AdamW 类方案 |
| bias、LayerNorm/RMSNorm 尺度 | 向量或标量，不能直接套用矩阵半正交更新 | 常保留 AdamW 类方案 |
| 标量参数 | 没有矩阵奇异谱 | 不使用 Muon |

这里的“典型选择”是待查阅官方训练配方后才能固定的工程结论，不能机械迁移到任意任务。

## 配方改型：Kimi / Moonlight 的两个可解析结论

原版 Muon 没有 weight decay；Moonlight 给它加上了，并做了一件很聪明的事——**用 RMS 对齐来复用 AdamW 的超参**。这两件事各自有一个干净的解析结论。

### 1. weight decay 等价于"对奇异值做 decay"

设参数矩阵 $W \in \mathbb{R}^{m\times n}$、SVD 为 $W = U\Sigma V^\top$。Weight decay 的更新为

$$W^{+} = (1-\eta\lambda)\,W,\qquad \eta > 0,\ \lambda \ge 0$$

代入 SVD：

$$W^{+} = (1-\eta\lambda)\,U\Sigma V^\top = U\big((1-\eta\lambda)\Sigma\big)V^\top$$

与 $W^{+} = U^{+}\Sigma^{+}(V^{+})^\top$ 对比，得

$$U^{+} = U,\qquad V^{+} = V,\qquad \Sigma^{+} = (1-\eta\lambda)\,\Sigma$$

**结论**：weight decay **不改变左右奇异子空间，只把所有奇异值按同一比例缩放**。由此奇异值在训练过程中便是有界的。

> 我的理解：这个结论很实用——它说明 weight decay 在 Muon 下是一个**纯粹的谱操作**，与 msign（只改奇异值的符号/幅度、不改子空间）作用在同一个对象上，两者不冲突。

### 2. Muon 更新量的 RMS 可以解析算出来

Moonlight 观察到 Adam 更新量的 RMS 比较稳定（通常在 $0.2 \sim 0.4$），于是**建议用 RMS Norm 把新优化器的 update RMS 对齐到 $0.2$**，从而复用 AdamW 搜好的学习率。

Muon 的更新矩阵 $\Phi_t$ 可以写成低秩分解形式

$$\Phi_t = U_{[:,:r]}\,V_{[:,:r]}^\top$$

其中 $U_{[:,:r]}$、$V_{[:,:r]}$ 的列是正交单位向量。**它的更新 RMS 可以精确求出**：

$$\|\Phi_t\|_F^2 = \mathrm{tr}(\Phi_t^\top\Phi_t) = \mathrm{tr}\big(V_{[:,:r]}\underbrace{U_{[:,:r]}^\top U_{[:,:r]}}_{I_r}V_{[:,:r]}^\top\big) = \mathrm{tr}\big(V_{[:,:r]}V_{[:,:r]}^\top\big) = \|V_{[:,:r]}\|_F^2 = r$$

（用到了 $U_{[:,:r]}^\top U_{[:,:r]} = I_r$ 与 $\|V_{[:,:r]}\|_F^2 = \mathrm{tr}(V^\top V) = r$。）

由 $\mathrm{RMS}(\Phi_t)^2 = \frac{1}{nm}\sum_{i,j}\Phi_{ij}^2 = \frac{\|\Phi_t\|_F^2}{nm}$ 得

$$\mathrm{RMS}(\Phi_t)^2 = \frac{r}{nm}\quad\Longrightarrow\quad \mathrm{RMS}(\Phi_t) = \sqrt{\frac{r}{nm}}$$

**注意这里的正交性是关键**：原文用指标形式写成

$$nm\,\mathrm{RMS}(\Phi_t)^2 = \sum_{i=1}^{n}\sum_{j=1}^{m}\sum_{k=1}^{r}U_{ik}^2V_{kj}^2 = \sum_{k=1}^{r}\Bigl(\sum_i U_{ik}^2\Bigr)\Bigl(\sum_j V_{kj}^2\Bigr) = \sum_{k=1}^r 1 = r$$

这个写法默认了 $\Phi_{ij}^2 = \sum_k U_{ik}^2V_{kj}^2$，即忽略了 $k$ 交叉项。交叉项消掉的**真正原因正是列正交性**（$U^\top U = I_r$），所以上面的 trace 推导才是严格版本，二者结论一致。

实践中严格低秩较少见，通常近似取 $r = \min(n,m)$，于是

$$\mathrm{RMS}(\Phi_t) \approx \sqrt{\frac{\min(n,m)}{nm}} = \frac{1}{\sqrt{\max(n,m)}}$$

（因为 $nm/\min(n,m) = \max(n,m)$。）

### 3. 对齐后的最终更新式

令学习率 $\eta_t$、解耦 L2 系数 $\lambda$，把 RMS 归一化与 weight decay 一并写进去：

$$W_t = W_{t-1} - \eta_t\left(0.2\,\frac{\Phi_t}{\mathrm{RMS}(\Phi_t)} + \lambda W_{t-1}\right) = W_{t-1} - \eta_t\left(0.2\,\Phi_t\sqrt{\max(n,m)} + \lambda W_{t-1}\right)$$

> 我的理解：这一步非常工程化也很漂亮——它把"要不要换优化器就得重搜学习率"这个迁移成本消掉了。注意 $0.2$ 是**目标 RMS**（对齐到 Adam 的经验值），而 $\sqrt{\max(n,m)}$ 正是 [RMS→RMS 几何](./spectral-norm-rms-geometry.md) 里那个维度因子的具体体现。

### 4. QK-Clip：把 attention 缩放吸收进权重

训 K2 时遇到 attention 矩阵最大值过大导致训练不稳定，Kimi 给出的修正是 **QK-Clip**。

记 attention 矩阵 $S = QK^\top$，最大值为 $S_{\max}$，期望阈值为 $\tau$。当 $S_{\max} > \tau$ 时，直接令 $S \leftarrow \gamma S$，其中 $\gamma = \tau/S_{\max}$。

现在的任务是把 $\gamma$ **吸收进权重**。对普通 MHA：

$$\begin{aligned}
W_t &= \mathrm{Optimizer}(W_{t-1}, G_t) \\
\text{if } S^{(l)}_{\max} &> \tau \text{ and } W \in \{W^{(l)}_q, W^{(l)}_k\}: \\
W_t &\leftarrow W_t \times \sqrt{\frac{\tau}{S^{(l)}_{\max}}}
\end{aligned}$$

其中 $S^{(l)}_{\max}$ 是第 $l$ 层 attention 矩阵的最大值，$W^{(l)}_q, W^{(l)}_k$ 是该层的 Q、K 权重。

**为什么是 $\sqrt{\gamma}$ 而不是 $\gamma$？** 因为 $S = QK^\top$ 对 $Q$ 和 $K$ 是双线性的：

$$(\sqrt\gamma\,Q)(\sqrt\gamma\,K)^\top = \sqrt\gamma\sqrt\gamma\,QK^\top = \gamma\,QK^\top$$

所以**给 $W_q$ 和 $W_k$ 各乘 $\sqrt\gamma$，等价于给 $S$ 乘 $\gamma$**。对 MLA 的情况更复杂，见[苏神博客](https://kexue.fm/archives/11126)。

## 与常见优化器的关系

| 方法 | 核心变换 | 状态开销（每参数） | 主要代价/边界 |
| --- | --- | ---:| --- |
| SGD | 沿梯度更新 | 可为 $0$ | 对参数尺度和学习率敏感 |
| Momentum SGD | 梯度的一阶指数移动平均 | 一阶状态 | 仍使用原始坐标下的方向幅度 |
| AdamW | 一阶/二阶矩的逐元素自适应缩放 + 解耦衰减 | 一阶 + 二阶状态 | 状态和逐元素预条件开销较大 |
| Shampoo 类 | 使用张量/矩阵二阶统计做预条件 | 结构相关 | 预条件矩阵、逆根和通信/计算更复杂 |
| Muon | 矩阵动量的近似极分解/半正交化 | 主要是一阶动量 | 仅定义矩阵参数路径；极分解近似有计算与数值边界 |

Muon 和 AdamW 的区别不应简化成“一个高级、一个落后”：AdamW 用每个元素的二阶统计调整步长；Muon 直接改变一个矩阵更新的奇异值结构。混合使用两者是参数类型不同的工程选择。

## 实现阅读：从伪代码到真实代码

下面是刻意省略具体系数与缩放约定的伪代码，用来标出数据流，不是可复现实装：

```python
for parameter_group in groups:
    for W in parameter_group.matrix_parameters:
        G = W.grad
        M = momentum * state[W].M + gradient_scale * G
        state[W].M = M

        direction = nesterov_transform(M, G)  # 可选，依实现而定
        direction = normalize_for_iteration(direction)
        O = newton_schulz_polar_approximation(direction, steps)
        W.add_(O, alpha=-lr * shape_scale(W))

    # 向量、标量、embedding 等参数由另一优化器处理。
```

阅读一个具体实现时，依次回答：

1. 何种张量形状进入 Muon，何种张量被排除？
2. 动量使用何种缩放约定，是否使用 Nesterov？
3. 宽矩阵如何处理，输入归一化采用哪种范数？
4. Newton–Schulz 多项式和迭代次数是什么，在哪种 dtype 执行？
5. shape scaling、weight decay 和学习率如何定义？
6. `foreach`、fused kernel、分布式分片与混合精度怎样影响性能和数值？

## 最小实验计划

### 环境记录

记录 GPU、CUDA/PyTorch/实现 commit、模型规模、数据集或 tokenization、全局 batch size、序列长度、精度、并行策略和随机种子。

### 对照与消融

1. 统一训练预算下对比 AdamW 与混合 Muon + AdamW；
2. 固定其余设置，只扫描 Muon 学习率、动量和迭代次数；
3. 比较原始动量、精确 polar（小矩阵离线基线）和 Newton–Schulz 近似的误差：

$$
\frac{\lVert O_{\mathrm{NS}} - UV^\top \rVert_F}{\lVert UV^\top \rVert_F};
$$

4. 记录每个训练阶段的更新谱、训练/验证曲线、每 token 时间、峰值显存和 NaN/overflow；
5. 单独改变参数分组，检验收益来自哪些矩阵而非混合配方的其他变化。

### 不应跳过的检查

- 确认宽矩阵/高矩阵的半正交条件与实际实现相符；
- 确认 Newton–Schulz 输入处于迭代收敛所需的尺度范围；
- 同时报告收敛速度、最终质量和总训练成本；
- 不把单一规模、单一种子或训练早期的 loss 优势泛化为通用结论。

## 两篇论文如何补充理解 Muon

两篇微信文章实际对应不同的一手研究，不能互相替代验证：

| 工作 | 研究维度 | 最稳妥的结论 | 不应推出的结论 |
| --- | --- | --- | --- |
| [Bi-Maxwell 物理响应与双时间尺度动量](./bimaxwell-muon-physical-response.md) | **时间**：动量记住多长历史 | 在一个 $124\text{M}$ GPT-2 / FineWeb 公开基准中，双时间尺度记忆核在特定冻结训练栈里优于匹配平均滞后的单 EMA 对照 | 对任意规模、任务或 Muon 配方普遍加速；完全解释 Muon 的全部行为 |
| [平滑矩阵 Polar 谱梯度流](./smoothed-matrix-polar-spectral-gradient-flows.md) | **空间/谱**：为何使用 Polar 型矩阵方向 | 平滑连续时间模型具有稳定性结论；在特定单层平方损失与局部二次尺度比较下，可得 Polar 局部占优条件 | 实际离散 Muon 在真实网络中必然快于 SGD / AdamW |

因此可以把 Muon 研究为一条两维路线：先问梯度历史如何被动量积累，再问积累后的矩阵方向如何按奇异谱重新缩放。两篇论文尚未将这两部分联合成一个经过真实大模型验证的完整理论。

## 我的当前理解

Muon 的关键不只是“用 Newton–Schulz 加速矩阵运算”，而是选择把**矩阵更新的奇异值谱**作为优化对象：动量整合跨 step 的梯度信息，极因子把更新从“按原梯度谱伸缩”变成“按子空间方向但更均匀地移动”。

我目前把它理解为一种矩阵级的更新几何/预条件选择，而不是对 AdamW、自然梯度或二阶优化的简单替代。这个理解尚须通过原始实现、训练配方和受控实验检验，尤其是“为什么对 Transformer 的某些权重有效”与“什么情况下会失败”。

## Open Questions

1. Muon 的原始实现对不同张量类别究竟采用怎样的参数分组和默认超参数？
2. 其 Newton–Schulz 多项式、迭代次数和范数归一化怎样影响低精度稳定性与 polar 近似误差？
3. shape scaling 的理论和经验动机是什么？它与学习率 scaling rule 如何耦合？
4. 对 Transformer 的 attention/MLP 投影、embedding 与卷积核，Muon 的收益和失败模式是否不同？
5. 更新谱均衡与训练损失、泛化、条件数或表示学习之间存在何种可检验关系？
6. [Bi-Maxwell](./bimaxwell-muon-physical-response.md) 与 [谱梯度流](./smoothed-matrix-polar-spectral-gradient-flows.md)如何在同一离散 Muon 训练栈中联结？是否能产生可检验的联合预测？
7. 在 FSDP、ZeRO、张量并行或 fused optimizer 中，矩阵的分片边界是否改变 Muon 的数学含义？

## Related Knowledge

- [Training Optimization](./) — 优化器与训练稳定性的主题入口
- [最速下降的范数对偶框架](./steepest-descent-duality-map.md) — 把 SGD / SignSGD / Muon 统一成"换对偶映射"，含 msign 的完整推导
- [谱范数与 RMS 几何](./spectral-norm-rms-geometry.md) — 为什么是谱范数、AdamW 的 $\ell_1\to\ell_\infty$ 几何缺陷
- [torch.optim — 优化算法](../../systems/pytorch/pytorch-optim.md) — AdamW、optimizer state 和 PyTorch 源码入口
- [自动微分 autograd](../../systems/pytorch/pytorch-autograd.md) — Muon 消费的 `.grad` 如何产生
- [分布式训练](../../systems/pytorch/pytorch-distributed.md) — 优化器状态分片与并行训练边界
- [CUTLASS / CuTe](../../systems/cutlass/) — Newton–Schulz 迭代依赖的矩阵乘法数据流与 GPU 实现背景
- [Bi-Maxwell 物理响应与双时间尺度动量](./bimaxwell-muon-physical-response.md) — Muon 动量核的物理建模与 $124\text{M}$ 公开基准
- [平滑矩阵 Polar 谱梯度流](./smoothed-matrix-polar-spectral-gradient-flows.md) — 连续时间稳定性与局部谱方向条件

## References

### 算法与实现

- Keller Jordan, [Muon: An optimizer for hidden weight matrices](https://kellerjordan.github.io/posts/muon/) — Muon 的公开说明入口
- [KellerJordan/Muon](https://github.com/KellerJordan/Muon) — 参考实现入口
- [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) — 大规模训练研究
- [Old Optimizer, New Norm: An Anthology](https://arxiv.org/abs/2409.20325) — 与谱范数最速下降相关的理论背景

### 本篇引用的推导来源

- resnet-65536，[Muon优化器科普，但从最速下降的本质出发](https://zhuanlan.zhihu.com/p/1954634867791869927)，2025-11-10 — 最速下降的范数对偶框架、Newton–Schulz 代数、Moonlight 的 weight decay / RMS 对齐、QK-Clip
- Jeremy Bernstein, [Deriving Muon](https://jeremybernste.in/writing/deriving-muon)
- Laker Newhouse, [Duality, Weight Decay, and Metrized Deep Learning](https://www.lakernewhouse.com/thesis.pdf)
- 苏剑林，[QK-Clip：让Muon在Scaleup之路上更进一步](https://kexue.fm/archives/11126)

### 对应一手论文与二手解读

- [A Physical Response-and-Memory Model for Muon Optimization](https://arxiv.org/abs/2608.22994) — Bi-Maxwell 论文，2026-08 预印本
- [A Continuous-Time Analysis of Smoothed Matrix-Polar Spectral Gradient Flows for Muon-Type Optimization](https://arxiv.org/abs/2608.01911) — 谱梯度流论文，2026-08 预印本
- 立与青，《[Muon优化器的物理模型——解释它为什么 work，还搞出了个 Bi-Maxwell](https://mp.weixin.qq.com/s/Gq7k9yDgDdVlT6NHoKJizQ)》，2026-08-25
- 阳仔的控智笔记，《[控制论，从未退场⑤：浙大团队最新成果，Muon 为什么不直接沿梯度走？这篇论文把矩阵优化写成了“谱反馈系统”](https://mp.weixin.qq.com/s/zapTq9YsHqAGLsymEt9gcQ?scene=1)》，2026-08-15
