---
title: Bi-Maxwell：Muon 的物理响应与双时间尺度动量
subtitle: 从输出扰动预算到双指数记忆核
type: paper
status: seed
tags: [AI, deep-learning, optimization, Muon, momentum, Bi-Maxwell, paper]
created: 2026-09-06
updated: 2026-09-06
source:
  - https://arxiv.org/abs/2608.22994
  - https://github.com/orange4664/bimaxwell-track3-reproduction
  - https://github.com/KellerJordan/modded-nanogpt/pull/339
---

# Bi-Maxwell：Muon 的物理响应与双时间尺度动量

> **论文：** Yinze Hu、Hongjun Xiang、Xingao Gong、Hongyu Yu，[*A Physical Response-and-Memory Model for Muon Optimization*](https://arxiv.org/abs/2608.22994)，arXiv:2608.22994v1，2026-08-24。论文以 $124\text{M}$ GPT-2 / FineWeb 单 GPU 公开基准研究 Muon 的动量记忆核；它是预印本，不应当作大模型训练的普适定论。

## 一句话理解

论文把矩阵权重看作响应介质：在“任意输入方向的最坏输出扰动不超过预算”的特定约束下，Muon 的 Polar 更新是最大瞬时耗散方向；将单指数动量解释为单一应力弛豫后，论文提出快、慢两条 EMA 支路组成的 **Bi-Maxwell** 记忆核，并在冻结的训练栈中报告了优于单 EMA 对照的步数结果。

## Problem

Muon 对矩阵动量做近似 Polar 变换，但仍有两个问题：

1. 为什么在特定场景中，半正交矩阵方向可能是合理的更新选择？
2. 单个 EMA 的一个时间常数，能否同时处理短期梯度变化和长期记忆？

本文没有重定义 Muon 的矩阵 Polar 步骤，而是尝试用受限优化解释其方向选择，并替换其时间维度上的记忆核。

## 从输出安全预算得到 Polar 方向

### 权重变化作为响应

对单层线性映射 $y=Wx$，令 $\Phi(W)$ 是失配势或损失，定义驱动力

$$
X=-\frac{\partial\Phi}{\partial W},
$$

并令权重变化率为

$$
V=\dot W.
$$

势能释放功率为

$$
P=-\dot\Phi=\langle X,V\rangle_F.
$$

### 受限问题

论文约束的不是每个参数元素，而是任意输入 $x$ 上的最坏输出扰动：

$$
A_{\mathrm{out}}(Vx)\le\varepsilon A_{\mathrm{in}}(x),\qquad \forall x.
$$

在其 RMS 通道归一化下，这等价于

$$
\lVert V\rVert_2\le c,
\qquad
c=\varepsilon\sqrt{\frac{m}{n}}.
$$

在此约束下最大化即时耗散，若 $X=U\Sigma V^\top$，则

$$
\max_{\lVert V\rVert_2\le c}\langle X,V\rangle_F
\quad\Longrightarrow\quad
V^\star=cUV^\top=c\operatorname{polar}(X),
$$

并有

$$
P_{\max}=c\lVert X\rVert_*.
$$

这给出 Polar 方向的一种条件化动机：它把每个非零奇异模式推到相同的更新幅度，同时服从最坏情况输出增益上限。

### 这个推导不说明什么

这个结论依赖欧氏度量、各向同性和最坏情况硬上限。它**不**证明 Muon 对所有训练问题最优：

- 若约束改为按 activation 分布加权的平均扰动，论文推得类似 $V\propto XC^{-1}$ 的协方差预条件；
- 若加入下游敏感度 $S$，方向可变为 $V\propto S^{-1}XC^{-1}$；
- 完整离散 Muon 还包含有限步 Newton–Schulz、学习率、形状缩放、权重衰减及参数分组，它们不由这个连续瞬时问题推出。

## Momentum 是记忆核，不是粒子惯性

### 单 EMA 对应 Maxwell 弛豫

论文采用介质内部应力的图像：

$$
\tau\dot M=X-M.
$$

移除驱动力后，状态按 $e^{-t/\tau}$ 衰减。以 step 间隔 $\Delta t$ 精确离散化得到归一化 EMA：

$$
M_t=\beta M_{t-1}+(1-\beta)X_t,
\qquad
\beta=e^{-\Delta t/\tau}.
$$

因此，单 EMA 是单一时间尺度的记忆核。它的平均滞后为

$$
n(\beta)=\frac{\beta}{1-\beta}.
$$

### 广义 Maxwell 与 Bi-Maxwell

广义 Maxwell 模型以多个正权重弛豫分量表示记忆：

$$
M(t)=\sum_jw_jM_j(t),
\qquad
w_j\ge0,\quad\sum_jw_j=1.
$$

论文取最低阶的两个分量：

$$
M_t^f=\beta_fM_{t-1}^f+(1-\beta_f)X_t,
$$

$$
M_t^s=\beta_sM_{t-1}^s+(1-\beta_s)X_t,
$$

$$
M_t^{\mathrm{BM}}=wM_t^f+(1-w)M_t^s,
\qquad 0\le w\le1.
$$

对应的离散核为

$$
J_{\mathrm{BM}}(\ell)=
w(1-\beta_f)\beta_f^\ell+
(1-w)(1-\beta_s)\beta_s^\ell.
$$

其思想是将快速响应与长期记忆从单个时间常数中解耦；这是一种模型和算法设计选择，不是神经网络被实验证明具有真实黏弹性微观结构。

## 实际训练堆栈中的 kernel swap

论文的基线动量为

$$
M_t=\mu_tM_{t-1}+(1-\mu_t)X_t,
$$

下游采用 Nesterov 型读出：

$$
R_t=(1-\mu_t)X_t+\mu_tM_t.
$$

从 $T_{\mathrm{on}}=1000$ 起，它只把状态替换为

$$
M_t^{\mathrm{eff}}=wM_t^f+(1-w)M_t^s,
$$

而瞬时梯度分支、$\mu_t$ 调度、Newton–Schulz、学习率、权重衰减和其余训练组件维持不变。切换时将两条新支路初始化为基线状态，因此切换当步与基线更新 bitwise identical，之后才发生分化。

论文默认报告的核参数为：

$$
\beta_f=0.85,
\qquad
\beta_s=0.98,
\qquad
w=0.4385.
$$

两支路的平均滞后约为 $5.67$ 与 $49$ steps，混合约为 $30$ steps。这些参数与 $T_{\mathrm{on}}$ 都具有训练协议依赖，不能直接迁移。

## Experiments

### 基准设置

| 项目 | 设置 |
| --- | --- |
| 模型 | GPT-2，$124\text{M}$ 参数，12 层、隐藏维 $768$ |
| 数据 | FineWeb |
| 每步 token | $524{,}288$ |
| 目标 | held-out validation loss $3.28$ |
| 基准协议 | modded-nanogpt Track 3 |
| 主要指标 | 首个满足 $(3.28-\bar L)\sqrt n\ge0.004$ 的同步验证 step |

主记录栈还含 SOAP、Tail-EMA、RowFloor、layer-radius pinning、PowerCool 学习率等组件。因此结论是对**这个完整配方中的动量核替换**，不是裸 Muon 对 AdamW 的比较。

### 报告结果

| 设置 | 单时间尺度 | Bi-Maxwell | 注释 |
| --- | ---:| ---:| --- |
| bare tuned-Muon | $3250$ steps，10 seeds，H100 | $3210$ steps，8 seeds，A800 | 硬件不同，不能视为严格墙钟对照 |
| record stack | $2690$ steps，8 seeds，A40 | $2635$ steps，8 seeds，A800 | 主结果也是跨硬件对照 |
| record stack 独立复现 | — | $2645$ steps，8 seeds，H100 | 有助于检验结果不只存在于 A800 |
| 匹配平均滞后 $30$ 的单 EMA | $2775$ steps | $2635$ steps | 支持核形状而非仅历史长度的重要性 |
| 去掉 Tail-EMA readout | $2735$ steps | $2690$ steps | 用于检查 readout 交互 |

论文报告 A800 主实验 $p=0.0065$、H100 复现 $p=0.0035$、两硬件池化 $p=7.0\times10^{-4}$。这些是基准协议中的 arm/均值统计，不是“每一个随机种子分别显著”。

### 不同谱方向是否需要不同记忆

论文构造只读 $K^\star$ 探针，估计梯度漂移强度 $D$ 与 batch noise $T$。在“随机游走漂移 + 独立测量噪声”的简化模型中，单指数最优平均滞后为

$$
n^\star=\frac{-1+\sqrt{1+4T/D}}{2}.
$$

论文将 SOAP whitened momentum 按奇异模式强度分 band，报告固定高学习率时强模式更偏向长记忆、部分中弱模式更偏向短记忆。这是支持方向依赖记忆的探针证据；绝对估计受基旋转、方向相关噪声与模型假设影响，作者也将谱细节定位为探索性分析。

## Limitations

1. **规模和任务有限：** 只研究 $124\text{M}$ GPT-2 / FineWeb，没有 $1\text{B}$、$7\text{B}$、多任务或长训练结论。
2. **关键主比较跨 GPU 型号：** 基线与实验 arm 分别使用 H100/A800 或 A40/A800；H100 复现加强了证据，但不是完整同硬件、同种子的 factorial 对照。
3. **它报告 step-count，不等于 wall-clock 加速百分比。** 论文称额外每 step 时间落在重复运行噪声内，但分布式训练未测试。
4. **显存成本可观：** 论文报告额外两个同形状 FP32 缓冲，在 $124\text{M}$ 配置中约 $680\text{MB}$，并随规模线性增长。
5. **超参数依赖协议：** $w$、平均滞后和启用时机均做过扫描；某些权重在 bare 与 record stack 上甚至出现相反效果方向。
6. **预印本和公开基准提交不等于同行评审：** 复现实验与日志公开是优点，但截至本文记录时相关 modded-nanogpt PR 仍是公开提交，不能表述为全部已合并的官方最终结论。
7. **物理类比有边界：** “正弛豫谱”是选择的广义 Maxwell 模型类；稳定的非互易系统也可能具有非单调弛豫。

## 我的当前理解

这篇论文把“动量是平均梯度”换成了更具体的问题：不同时间尺度的梯度历史在后续 Polar 更新前应如何组合。最有价值的实验不是单独的 $40$ 或 $55$ step 改善，而是匹配平均滞后的单 EMA 仍落后于 Bi-Maxwell 的控制：它至少在该协议中支持“核形状”而非单纯“记得更久”是变量。

但空间极分解、时间记忆和训练配方依然耦合。Bi-Maxwell 的成功不能自动证明 Polar 几何是唯一原因，也不能验证 [平滑矩阵 Polar 谱梯度流](./smoothed-matrix-polar-spectral-gradient-flows.md) 中的连续时间定理。

## Open Questions

1. 在同硬件、同随机种子配对下，Bi-Maxwell 的效应量和 wall-clock 成本是多少？
2. 在更大语言模型、不同数据、不同 batch size 与分布式 ZeRO/FSDP 下，最优时间尺度是否稳定？
3. 是否能使用每个谱模式自适应的记忆核，而不会使状态、通信和超参数不可控？
4. 输出扰动预算能否由真实 activation covariance、下游敏感度和非欧氏度量更准确地刻画？
5. 该物理模型能否推导实际 Muon 的 Newton–Schulz 系数、shape scaling 或 weight decay，而不是仅解释理想 Polar 方向？

## Related Knowledge

- [Muon 优化器](./muon-optimizer.md) — 算法总览、参数分组与最小实验
- [平滑矩阵 Polar 谱梯度流](./smoothed-matrix-polar-spectral-gradient-flows.md) — Muon-type 连续时间谱反馈的另一条理论线
- [torch.optim — 优化算法](../../systems/pytorch/pytorch-optim.md) — EMA 状态与优化器实现背景

## References

### 一手资料

- Hu et al., [A Physical Response-and-Memory Model for Muon Optimization](https://arxiv.org/abs/2608.22994), arXiv:2608.22994v1, 2026-08-24
- [论文 HTML](https://arxiv.org/html/2608.22994)
- [冻结脚本、per-seed 日志和探针数据](https://github.com/orange4664/bimaxwell-track3-reproduction)
- [固定复现仓库提交](https://github.com/orange4664/bimaxwell-track3-reproduction/commit/d125cebb54786bb743082acc93758df64c2df9c3)
- [record-stack Bi-Maxwell $2635$-step 公开提交](https://github.com/KellerJordan/modded-nanogpt/pull/339)
- [bare tuned-Muon Bi-Maxwell $3210$-step 公开提交](https://github.com/KellerJordan/modded-nanogpt/pull/340)
- [record-stack $2690$-step 基线](https://github.com/KellerJordan/modded-nanogpt/pull/328)

### 二手解读

- 立与青，《[Muon优化器的物理模型——解释它为什么 work，还搞出了个 Bi-Maxwell](https://mp.weixin.qq.com/s/Gq7k9yDgDdVlT6NHoKJizQ)》，2026-08-25。本文对其关于“完全解释 Muon”“所有种子显著”“Muon 比 AdamW 快省”等易过度外推的表述作了收紧。
