---
title: 平滑矩阵 Polar 谱梯度流
subtitle: Muon-type 优化的连续时间稳定性与局部优势条件
type: paper
status: seed
tags: [AI, deep-learning, optimization, Muon, polar-decomposition, spectral-gradient-flow, control-theory, paper]
created: 2026-09-06
updated: 2026-09-06
source:
  - https://arxiv.org/abs/2608.01911
  - https://mp.weixin.qq.com/s/zapTq9YsHqAGLsymEt9gcQ?scene=1
---

# 平滑矩阵 Polar 谱梯度流

> **论文：** Jinlin Liu、Song Chen、Jiaxu Liu、Chao Xu，[*A Continuous-Time Analysis of Smoothed Matrix-Polar Spectral Gradient Flows for Muon-Type Optimization*](https://arxiv.org/abs/2608.01911)，arXiv:2608.01911v1，2026-08-03。作者来自浙江大学、新加坡国立大学、湖州师范学院等机构。这是一篇关于 Muon-type **连续时间模型**的理论预印本；它没有在真实神经网络上训练，也没有给出可直接替换实际 Muon 的新离散优化器。

## 一句话理解

理想 Polar 映射在矩阵接近秩亏时不光滑或病态。论文用一个平滑的谱反馈 $h_\epsilon$ 代替它，得到可用 ODE 与 Lyapunov 工具分析的 Muon-type 流：证明连续系统的稳定性和收敛性质，并在严格限定的局部二次模型中给出“何时谱方向优于普通梯度”的条件。

## Problem

普通梯度流为

$$
\dot W=-\nabla f(W).
$$

理想的 Matrix-Polar 流写成

$$
\dot W=-\operatorname{polar}(\nabla f(W)).
$$

若 $G=U\Sigma V^\top$，则

$$
\operatorname{polar}(G)=UV^\top=G(G^\top G)^{-1/2}.
$$

当最小奇异值趋近 $0$ 或秩发生变化时，该映射的光滑性和条件数会成为理论分析障碍。因此论文的问题不是“发布一个已验证的现实训练优化器”，而是：如何构造平滑的 Polar-like 反馈，并分析其稳定性和局部下降条件？

## 平滑谱反馈

论文定义

$$
h_\epsilon(Z)=Z(Z^\top Z+\epsilon I)^{-1/2},
\qquad \epsilon>0.
$$

若

$$
Z=U\operatorname{diag}(\sigma_i)V^\top,
$$

则

$$
h_\epsilon(Z)=
U\operatorname{diag}\left(
\frac{\sigma_i}{\sqrt{\sigma_i^2+\epsilon}}
\right)V^\top.
$$

它在不同尺度的行为：

| 奇异值尺度 | 响应 | 含义 |
| --- | --- | --- |
| $\sigma_i\gg\sqrt\epsilon$ | $\sigma_i/\sqrt{\sigma_i^2+\epsilon}\approx1$ | 接近理想 Polar，近似压平强模式 |
| $\sigma_i\ll\sqrt\epsilon$ | $\sigma_i/\sqrt{\sigma_i^2+\epsilon}\approx\sigma_i/\sqrt\epsilon$ | 小信号时近似线性，不把极小模式强制放大到单位幅度 |

相应势函数为

$$
\Phi_\epsilon(Z)=
\operatorname{Tr}\left[(Z^\top Z+\epsilon I)^{1/2}-\sqrt\epsilon I\right],
$$

并满足

$$
\nabla\Phi_\epsilon(Z)=h_\epsilon(Z).
$$

论文证明 $h_\epsilon$ 连续可微、单调且全局 Lipschitz：

$$
\lVert h_\epsilon(M)-h_\epsilon(N)\rVert_F
\le
\frac{1}{\sqrt\epsilon}\lVert M-N\rVert_F.
$$

此外，若 $d=\min(m,n)$，有

$$
\lVert h_\epsilon(M)\rVert_F\le\sqrt d.
$$

## 直接谱梯度流

论文先研究

$$
\dot W=-h_\epsilon(\nabla f(W)).
$$

设梯度的奇异值为 $\gamma_i$，则沿轨迹：

$$
\frac{d}{dt}f(W(t))
=-\left\langle G,h_\epsilon(G)\right\rangle
=-\sum_i\frac{\gamma_i^2}{\sqrt{\gamma_i^2+\epsilon}}
\le0.
$$

在 $f$ 有下界、coercive 且梯度局部 Lipschitz 等条件下，论文证明全局解存在唯一、轨迹有界、梯度趋于零并逼近驻点集合。

### 收敛率的正确阅读方式

| 假设 | 论文给出的量级 | 不代表什么 |
| --- | --- | --- |
| 一般非凸 | $\min_{0\le t\le T}\lVert\nabla f(W(t))\rVert_F^2=O(1/T)$ | 不是深度网络全局最优保证 |
| 凸 | $f(W(t))-f_*=O(1/t)$ | 不是实际离散 Muon 的步数结论 |
| PL 条件 | 指数衰减 | 不说明所有训练损失都满足 PL |

这些结果是平滑 ODE 的稳定性/收敛证书，不是 Muon 对普通梯度流全局更快的证明。

## Momentum 扩维系统

论文进一步研究带记忆的系统：

$$
\dot W=-h_\epsilon(M),
$$

$$
\dot M=a\nabla f(W)-bM,
\qquad a,b>0.
$$

采用 Lyapunov 函数

$$
V(W,M)=a\left(f(W)-f_{\inf}\right)+\Phi_\epsilon(M),
$$

其导数中交叉项抵消：

$$
\dot V=-b\langle h_\epsilon(M),M\rangle\le0.
$$

由此得到全局解、有界性和极限点条件

$$
\nabla f(W)=0,
\qquad M=0.
$$

非凸情况下有最小梯度平方 $O(1/T)$ 的界；凸情形对 Cesàro 时间平均

$$
\bar W_T=\frac1T\int_0^T W(t)\,dt
$$

给出 $f(\bar W_T)-f_*=O(T^{-1/2})$。

> **边界：** 这是“与 Muon momentum 结构相似”的连续扩维模型，不能被称作实际离散 Muon 的严格连续化。有限学习率、随机梯度、有限 Newton–Schulz 步数、混合精度和参数分组均不在这个定理中。

## 何时谱方向局部更好

### 一般局部二次比较

对点 $W$、梯度 $G$ 与候选方向 $D$，局部二次模型是

$$
f(W-\alpha D)
\approx
f(W)-\alpha\langle G,D\rangle
+\frac{\alpha^2}{2}\langle D,\mathcal{H}_W[D]\rangle.
$$

在下降对齐与正方向曲率条件下，该方向的局部最优步长为

$$
\alpha_D^\star=
\frac{\langle G,D\rangle}
{\langle D,\mathcal{H}_W[D]\rangle},
$$

归一化局部下降率为

$$
R_W(D)=
\frac{\langle G,D\rangle^2}
{\langle D,\mathcal{H}_W[D]\rangle}.
$$

将 Frobenius 梯度与平滑谱方向分别代入：

$$
R_F=\frac{\lVert G\rVert_F^4}
{\langle G,\mathcal{H}_W[G]\rangle},
$$

$$
R_\epsilon=
\frac{\langle G,h_\epsilon(G)\rangle^2}
{\langle h_\epsilon(G),\mathcal{H}_W[h_\epsilon(G)]\rangle}.
$$

定义 $\Gamma_\epsilon=R_\epsilon/R_F$。只有在 **每个方向都分别使用局部二次最优尺度** 的条件下，$\Gamma_\epsilon>1$ 才表示谱方向在该局部模型中更快；它不是固定同一学习率的离散训练比较。

### 单层平方损失的漂亮条件

对

$$
f(W)=\frac12\lVert WA-Y\rVert_F^2,
$$

有

$$
G=(WA-Y)A^\top,
\qquad
\mathcal{H}_W[D]=DAA^\top.
$$

在方阵、满秩、$\epsilon\to0$ 的理想 Polar 极限以及正曲率等附加条件下，论文得到

$$
\Gamma_0=
r_{\mathrm{eff}}(G)
\operatorname{sim}_{\mathrm{HS}}(\rho_G,\rho_A),
$$

其中

$$
r_{\mathrm{eff}}(G)=
\frac{\lVert G\rVert_*^2}{\lVert G\rVert_F^2},
$$

$$
\rho_G=\frac{G^\top G}{\lVert G\rVert_F^2},
\qquad
\rho_A=\frac{AA^\top}{\lVert A\rVert_F^2},
$$

$$
\operatorname{sim}_{\mathrm{HS}}(\rho_G,\rho_A)
=\langle\rho_G,\rho_A\rangle.
$$

因此谱方向局部占优的充要条件为

$$
r_{\mathrm{eff}}(G)
\operatorname{sim}_{\mathrm{HS}}(\rho_G,\rho_A)>1.
$$

可解释为：梯度有效秩高、且梯度右侧谱结构与 activation covariance 对齐时，压平谱的方向有更好的局部曲率—下降权衡。它需要两者共同成立，反驳了“Polar 总会更快”的简单说法。

## Experiments

论文用 NumPy 矩阵运算和 RK4 做连续时间数值实验，**没有真实神经网络训练**：

1. 在 $f(W)=\frac12\lVert W\rVert_F^2$、$4\times4$ 矩阵上，将平滑 ODE 与离散 exact-Polar 更新比较；远离秩退化的早期轨迹接近，接近最优点时小奇异值导致累计差异增大。
2. 在两个构造的 $2\times2$ sweep 上验证局部阈值：一个固定 $r_{\mathrm{eff}}=1.6$，另一个固定相似度 $0.7$；实际初始下降率在理论阈值附近交叉。
3. 在 $12\times12$ 强凸/PL 与一般凸二次目标上，观察 objective gap 位于理论上界以下；上界较保守。

这些实验主要验证推导与数值实现自洽，不能视为 LLM 或 Transformer 性能证据。

## Limitations

1. **不是实际训练算法评测：** 没有真实神经网络、随机梯度、有限 batch、有限学习率或 AdamW/Muon 基线比较。
2. **平滑近似只在 rank-safe 有限时间区间逼近 ideal Polar：** 需要相关最小奇异值统一远离零；接近秩变化时该保证不适用。
3. **coercivity 是强假设：** 深度网络常有尺度对称性和非紧子水平集，未必满足。
4. **一般情形只逼近驻点集合：** 不保证唯一驻点或排除不良驻点。
5. **局部阈值适用范围窄：** 它依赖单层线性平方损失、方阵、满秩、理想 Polar 极限、正方向曲率与各方向独立最优缩放。
6. **实现层因素被排除：** Newton–Schulz 的近似误差、分布式矩阵分片、fused kernel 与低精度都可能改变实际行为。
7. **预印本状态：** 论文未提供公开代码仓库；数值构造足够清楚但复现便利性有限。

## 我的当前理解

“谱反馈系统”是一个有用但容易被夸张的说法：论文真正建立的是平滑非线性反馈 $h_\epsilon$ 与 Lyapunov 结构，而不是证明真实训练中的 Muon 已被控制理论完整解释。

最值得带回实证研究的预测是：是否受益不只取决于“梯度有多大”，还取决于梯度有效秩和它与局部曲率/activation 结构的对齐。将这个预测用于真实 Transformer 前，需要把连续理论的量与离散训练日志中的谱统计对应起来。

## Open Questions

1. 是否能从实际 Muon 的 Newton–Schulz 多项式和参数规模导出可用的 $\epsilon$ 对应关系？
2. 有限学习率与随机梯度下，$\Gamma_\epsilon$ 或其近似能否预测哪些 layer 受益？
3. 在非方阵、卷积核、attention 投影和张量并行分片中，有效秩—相似度条件应如何改写？
4. 连续模型的单指数记忆能否与 [Bi-Maxwell](./bimaxwell-muon-physical-response.md) 的双时间尺度核结合，并维持可分析的 Lyapunov 结构？
5. 不满足 coercivity 的深度网络中，需要什么替代条件来保证轨迹有界或稳定？

## Related Knowledge

- [Muon 优化器](./muon-optimizer.md) — 算法骨架、Newton–Schulz 与参数分组
- [Bi-Maxwell：Muon 的物理响应与双时间尺度动量](./bimaxwell-muon-physical-response.md) — 时间记忆核与 $124\text{M}$ 公开基准
- [torch.optim — 优化算法](../../systems/pytorch/pytorch-optim.md) — 实际优化器状态与训练循环背景

## References

### 一手资料

- Liu et al., [A Continuous-Time Analysis of Smoothed Matrix-Polar Spectral Gradient Flows for Muon-Type Optimization](https://arxiv.org/abs/2608.01911), arXiv:2608.01911v1, 2026-08-03
- [论文 HTML](https://arxiv.org/html/2608.01911)
- [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)
- [Old Optimizer, New Norm: An Anthology](https://arxiv.org/abs/2409.20325)
- [On the Convergence of Muon](https://arxiv.org/abs/2502.02900)
- [Muon Convergence Analysis](https://arxiv.org/abs/2505.23737)
- [Non-Euclidean Trust Region for Gradient Orthogonalization](https://arxiv.org/abs/2503.12645)

### 二手解读

- 阳仔的控智笔记，《[控制论，从未退场⑤：浙大团队最新成果，Muon 为什么不直接沿梯度走？这篇论文把矩阵优化写成了“谱反馈系统”](https://mp.weixin.qq.com/s/zapTq9YsHqAGLsymEt9gcQ?scene=1)》，2026-08-15。该文对原论文的理论边界转述总体克制；本笔记仍以原预印本的假设为准。
