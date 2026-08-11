---
title: DDIM (Denoising Diffusion Implicit Models)
type: paper
status: reading
authors: [Jiaming Song, Chenlin Meng, Stefano Ermon]
year: 2020
tags: [Diffusion, Generative, DDIM, Variational, KL]
created: 2026-08-11
updated: 2026-08-11
source: 与 DeepSeek 的对话整理（https://chat.deepseek.com/share/kanpr0fferkd76bwlz）
arxiv: 2010.02502v4
---

# DDIM (Denoising Diffusion Implicit Models)

## Paper Information

- Authors: Jiaming Song, Chenlin Meng, Stefano Ermon
- Year: 2020
- arXiv: [2010.02502v4](https://arxiv.org/abs/2010.02502)

## TL;DR

DDIM 通过引入**非马尔可夫前向过程**，把 DDPM 的训练目标和采样过程解耦：训练目标保持不变（仍是预测噪声的 MSE），但采样时可以通过调节 $\sigma$ 实现快速确定性采样（10~20 步即可生成），无需重新训练模型。

## Problem

DDPM 能生成高质量图片，但采样需要 1000 步迭代，非常慢。能否在不重训模型的前提下加速采样？

## Motivation

DDPM 的训练目标是变分下界（VLB），它依赖马尔可夫前向过程。如果改成非马尔可夫过程，是否能保持训练目标不变？如果能，就可以直接复用预训练的 DDPM 模型，只改采样过程来加速。

## Key Idea

定义一族**非马尔可夫前向过程** $q_\sigma(x_{1:T} | x_0)$，其边际分布与 DDPM 完全一致，但联合分布不同。证明这族过程对应的变分目标 $J_\sigma$ 等价于 DDPM 的加权去噪损失 $L_\gamma$，仅差一个常数——这意味着训练目标对 $\sigma$ 不敏感，$\sigma$ 可以纯粹作为推理阶段的"旋钮"。

## Method

### 非马尔可夫前向过程

- DDPM 的前向是马尔可夫的：$q(x_{t-1} | x_t)$
- DDIM 让前向过程显式依赖 $x_0$：$q_\sigma(x_{t-1} | x_t, x_0)$
- 边际分布 $q(x_t | x_0) = \mathcal{N}(\sqrt{\alpha_t} x_0, (1-\alpha_t) I)$ 保持不变

### 公式 (11) 的核心：变分推断目标

公式 (11) 位于论文第 4 页，第 3.2 节"生成过程和统一变分推理目标"之后，是 DDIM 的核心变分推断目标函数，定义了在非马尔可夫前向过程族下生成模型需要优化的损失函数：

$$
J_\sigma(\epsilon_\theta) \triangleq \mathbb{E}_{x_{0:T} \sim q_\sigma(x_{0:T})} \left[ \log q_\sigma(x_{1:T} | x_0) - \log p_\theta(x_{0:T}) \right]
$$

展开为四项：

$$
= \mathbb{E}_{x_{0:T}} \left[ \log q_\sigma(x_T | x_0) + \sum_{t=2}^T \log q_\sigma(x_{t-1} | x_t, x_0) - \sum_{t=1}^T \log p_\theta^{(t)}(x_{t-1} | x_t) - \log p_\theta(x_T) \right]
$$

#### 第一行解读：宏观目标（变分下界 / 负 ELBO）

- $J_\sigma$：需要最小化的损失函数
- $q_\sigma(x_{1:T} | x_0)$：推理过程（前向过程）。给定真实图片 $x_0$，生成加噪隐变量序列 $x_1$ 到 $x_T$ 的概率。下标 $\sigma$ 代表这是一个**非马尔可夫过程**（即 $x_{t-1}$ 不仅依赖 $x_t$，还直接依赖初始的 $x_0$）
- $p_\theta(x_{0:T})$：生成过程（逆向过程）。由噪声 $x_T$ 开始，逐步还原出 $x_0$ 的概率（即我们训练的模型）

**数学本质**：$J_\sigma$ 实际上是 $q$（真实加噪过程）和 $p_\theta$（模型生成过程）之间的 KL 散度（忽略 $q(x_0)$ 的常数项）。由于 $\log q - \log p = \log(q/p)$，最小化 $J_\sigma$ 等价于最大化对数似然 $\log p_\theta(x_0)$ 的变分下界（ELBO）——让模型的生成分布尽可能拟合真实数据的加噪逆过程。

#### 第二行解读：展开拆解（四项的物理含义）

| 项 | 含义 | 是否含 $\theta$ |
|---|---|---|
| $\log q_\sigma(x_T \| x_0)$ | 先验分布：从原图 $x_0$ 加噪到纯噪声 $x_T$ 的对数概率。由于 $\alpha_T \approx 0$，这一项是标准高斯分布，**不含可训练参数 $\theta$**，在优化中视为常数（通常忽略） | 否 |
| $\sum_{t=2}^T \log q_\sigma(x_{t-1} \| x_t, x_0)$ | **"真实"逆向核**。最重要的设计：标准 DDPM 中逆过程是马尔可夫的 $q(x_{t-1} \| x_t)$，但 DDIM 中显式依赖 $x_0$。这一项充当"老师"的角色，告诉模型当给定 $x_t$ 和原始 $x_0$ 时 $x_{t-1}$ 应该落在哪里 | 否 |
| $\sum_{t=1}^T \log p_\theta^{(t)}(x_{t-1} \| x_t)$ | 模型学习的逆过程。神经网络 $\epsilon_\theta$ 拟合的分布。它试图模仿第二项的"真实逆过程"，但由于它**看不到 $x_0$**，只能通过神经网络预测噪声间接估计 $x_0$ | **是**（核心优化对象） |
| $\log p_\theta(x_T)$ | 初始噪声的先验，通常固定为 $\mathcal{N}(0, I)$，不参与训练 | 否 |

#### 核心洞察

作者特意把 $q$ 写成 $x_{t-1} | x_t, x_0$ 的形式，是为了让 $p_\theta$ 也能利用同样的形式（用模型预测的 $\hat{x}_0$ 替换真实 $x_0$），从而把复杂的变分目标转化为简单的去噪匹配目标。

#### 为什么这个公式极其重要？（论文的逻辑支点）

1. **统一了训练目标**：尽管前向过程 $q_\sigma$ 变了（从马尔可夫变成非马尔可夫），但最终的变分目标依然可以拆解成"去噪"的形式
2. **引出了定理 1**：论文紧接着证明，这个复杂的 $J_\sigma$ 等价于公式 (5) 中的简单去噪损失 $L_\gamma$（即预测随机噪声 $\epsilon$ 的 MSE）。这意味着完全不需要重新训练模型，直接用原本 DDPM 训练好的模型，只要调整采样时的 $\sigma$，就能实现 DDIM 的采样
3. **解耦了训练与推理**：由于 $J_\sigma$ 中只有 $p_\theta$ 项是优化的对象，而 $q_\sigma$（带有 $\sigma$）只参与 KL 散度的计算但不参与梯度回传的权重调整，因此 $\sigma$ 变成了一个纯粹在推理（采样）阶段调节的"旋钮"——当 $\sigma = 0$ 时变成确定性的 DDIM（快速采样），当 $\sigma > 0$ 时变成随机性的 DDPM（高质量但慢）

#### 直观类比

把生成图片想象成从模糊马赛克（$x_T$）还原高清图（$x_0$）：

- 第二项 $q_\sigma$ 告诉你"如果原图长这样，那么中间步骤的模糊图应该长什么样"（标准答案）
- 第三项 $p_\theta$ 是模型在猜"虽然我不知道原图，但根据当前模糊图，我猜测上一步应该是这样"
- 公式 (11) 就是在计算模型猜测与标准答案之间的差距。通过最小化这个差距，模型就学会了去噪

**总结**：公式 (11) 是 DDIM 的理论基石，它证明了即使破坏扩散过程的马尔可夫性质，我们依然拥有合法的变分下界，且该下界最终转化为一个我们早已熟悉的去噪得分匹配问题，从而实现了"训练不变，采样加速"的神奇效果。

### KL 散度（在 DDIM 中的角色）

KL 散度（Kullback-Leibler Divergence，相对熵）是衡量两个概率分布之间差异的非对称性度量。在机器学习（尤其是变分推断）中，它是构建损失函数（ELBO）的核心数学工具。

#### 数学定义

对于定义在同一个随机变量 $x$ 上的两个概率分布 $P$（真实分布）和 $Q$（拟合分布），$Q$ 对 $P$ 的 KL 散度定义为：

- **离散型**：$D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$
- **连续型**：$D_{KL}(P \| Q) = \int_{-\infty}^{\infty} p(x) \log \frac{p(x)}{q(x)} dx$

#### 直观理解（它到底在算什么？）

KL 散度衡量的是"**用分布 $Q$ 去近似分布 $P$ 时，损失了多少信息**"（或者说，浪费了多少编码长度）。

- 如果 $P$ 和 $Q$ 完全一致（即 $P(x) = Q(x)$ 对所有 $x$ 成立），则 $D_{KL} = 0$
- 如果两者差异很大，KL 散度会是一个较大的正数（因为对数函数内比值变大）

你可以把它想象成"两个概率分布之间的距离"，但要注意，它不是真正的"距离"（因为不满足对称性）。

#### 三大性质

1. **非负性**：$D_{KL}(P \| Q) \geq 0$，且当且仅当 $P = Q$ 时取等号。这是由**吉布斯不等式**保证的
2. **非对称性**（不可作为距离）：$D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$
   - $D_{KL}(P \| Q)$ 称为 **Forward KL**：强迫 $Q$ 覆盖 $P$ 的所有峰值（**均值搜索**，追求覆盖面广）
   - $D_{KL}(Q \| P)$ 称为 **Reverse KL**：强迫 $Q$ 集中在 $P$ 的某一个峰值上（**模式搜索**，追求质量高）
   - 在生成模型中（如 VAE 和扩散模型）选择哪种方向非常重要
3. **链式法则**：联合分布的 KL 散度可以拆解为边缘分布的 KL 加条件分布的 KL 期望，这在推导变分下界（ELBO）时极其常用

#### 在公式 (11) 中的具体作用

公式 (11) 虽然写成了期望的对数差形式，但它的本质就是在求 KL 散度。可以重写为：

$$
J_\sigma = \mathbb{E}_{q(x_0)} \left[ \underbrace{D_{KL}(q_\sigma(x_{1:T} | x_0) \| p_\theta(x_{0:T}))}_{\text{这就是 KL 散度的定义！}} \right] + \text{常数}
$$

- $P$（真实）= 前向加噪过程 $q_\sigma(x_{1:T} | x_0)$
- $Q$（拟合）= 逆向生成过程 $p_\theta(x_{0:T})$

**为什么引入 KL 散度**：因为我们无法直接计算 $p_\theta(x_0)$ 的真实似然（积分太复杂），所以转而最小化两个分布的 KL 散度。当 KL 最小时，模型的逆向过程 $p_\theta$ 无限逼近真实的加噪逆过程，此时模型生成的图片质量最高。

#### 直觉口诀和例子

> "KL 散度 = 用 $Q$ 编码 $P$ 时，平均多出来的比特数。"

**天气例子**：

- 真实情况 $P$ 是"80% 晴天，20% 雨天"
- 模型 $Q$ 预测是"50% 晴天，50% 雨天"
- $D_{KL}(P \| Q)$ 就会比较大，因为预测的确定性不够（信息熵增大了）

**关键结论**：在扩散模型中，KL 散度项最终被简化成了"预测噪声的均方误差（MSE）"，这也是为什么 DDPM 和 DDIM 最后训练的损失函数看起来只是在做简单的回归任务！

### 定理 1 的证明：$J_\sigma = L_\gamma + C$

#### 定理陈述

对于所有 $\sigma > 0$，存在 $\gamma \in \mathbb{R}_{>0}^T$ 和 $C \in \mathbb{R}$，使得 $J_\sigma = L_\gamma + C$。

#### 含义

非马尔可夫推理过程的变分目标等价于标准 DDPM 去噪目标，仅差一个常数和重新加权。可以直接用预训练 DDPM 模型，通过改变 $\sigma$ 来实现 DDIM 采样。

#### 证明步骤（来自论文 Appendix B，第 14 页）

##### Step 1：写出 $J_\sigma$ 并扔掉常数项

首先，生成过程 $p_\theta$ 的定义是：

$$
p_\theta^{(t)}(x_{t-1} | x_t) = q_\sigma(x_{t-1} | x_t, f_\theta^{(t)}(x_t))
$$

即用模型预测的 $f_\theta^{(t)}(x_t)$ 去替代真实的 $x_0$。

证明中引入符号 $\equiv$（表示"等价于，忽略不依赖于 $\theta$ 的常数项"）。

**推导过程**：从公式 (11) 的展开形式出发：

$$
J_\sigma = \mathbb{E}_{x_{0:T}} \left[ \underbrace{\log q_\sigma(x_T | x_0)}_{\text{常数项 1}} + \sum_{t=2}^T \log q_\sigma(x_{t-1} | x_t, x_0) - \sum_{t=1}^T \log p_\theta^{(t)}(x_{t-1} | x_t) - \underbrace{\log p_\theta(x_T)}_{\text{常数项 2}} \right]
$$

**第 1 步：扔掉与 $\theta$ 无关的常数项**

- 1. $\log q_\sigma(x_T | x_0)$：前向加噪到最终噪声的对数概率，不含 $\theta$
- 2. $\log p_\theta(x_T)$：虽然带 $\theta$ 下标，但 $p_\theta(x_T)$ 通常固定为 $\mathcal{N}(0, I)$，不参与训练

两项合并进常数 $C$，得到：

$$
J_\sigma \equiv \mathbb{E}_{x_{0:T}} \left[ \sum_{t=2}^T \log q_\sigma(x_{t-1} | x_t, x_0) - \sum_{t=1}^T \log p_\theta^{(t)}(x_{t-1} | x_t) \right]
$$

**第 2 步：把 $t=1$ 项单独拿出来**

第二个求和 $\sum_{t=1}^T$ 拆成 $t=1$ 和 $t=2..T$ 两部分：

$$
J_\sigma \equiv \mathbb{E}_{x_{0:T}} \left[ \sum_{t=2}^T \left( \log q_\sigma(x_{t-1} | x_t, x_0) - \log p_\theta^{(t)}(x_{t-1} | x_t) \right) - \log p_\theta^{(1)}(x_0 | x_1) \right]
$$

**第 3 步：识别 KL 散度**

注意 $\log q - \log p = \log \frac{q}{p}$，对 $x_{t-1}$ 在整个空间积分（取期望）后正是 KL 散度的定义：

$$
\mathbb{E}_{x_{t-1}} \left[ \log \frac{q_\sigma(x_{t-1} | x_t, x_0)}{p_\theta^{(t)}(x_{t-1} | x_t)} \right] = D_{KL}\left( q_\sigma(x_{t-1} | x_t, x_0) \| p_\theta^{(t)}(x_{t-1} | x_t) \right)
$$

因此公式 (11) 化简为：

$$
J_\sigma(\epsilon_\theta) \equiv \mathbb{E}_q \left[ \sum_{t=2}^T D_{KL}(q_\sigma(x_{t-1} | x_t, x_0) \| p_\theta^{(t)}(x_{t-1} | x_t)) - \log p_\theta^{(1)}(x_0 | x_1) \right]
$$

##### Step 2：将 KL 散度转化为高斯分布的平方差

在 DDIM 的定义中，$q_\sigma(x_{t-1} | x_t, x_0)$ 是一个高斯分布：

$$
q_\sigma(x_{t-1} | x_t, x_0) = \mathcal{N}\left(\sqrt{\alpha_{t-1}} x_0 + \frac{1-\alpha_{t-1}-\sigma_t^2}{\sqrt{1-\alpha_t}} \cdot \frac{x_t - \sqrt{\alpha_t} x_0}{\sqrt{1-\alpha_t}}, \sigma_t^2 I\right)
$$

而模型 $p_\theta^{(t)}$ 是把这个公式中的 $x_0$ 换成了模型的预测值 $f_\theta^{(t)}(x_t)$。

**关键点**：这两个高斯分布的**协方差矩阵完全相同**（都是 $\sigma_t^2 I$），只有均值不同。

**引理（协方差相同的高斯 KL 散度）**：对于两个协方差相同的高斯分布 $P = \mathcal{N}(\mu_1, \Sigma)$ 和 $Q = \mathcal{N}(\mu_2, \Sigma)$，其 KL 散度简化为：

$$
D_{KL}(P \| Q) = \frac{1}{2} (\mu_1 - \mu_2)^T \Sigma^{-1} (\mu_1 - \mu_2)
$$

**证明**：从一般高斯 KL 公式出发：

$$
D_{KL}(\mathcal{N}(\mu_1, \Sigma_1) \| \mathcal{N}(\mu_2, \Sigma_2)) = \frac{1}{2} \left[ \text{tr}(\Sigma_2^{-1} \Sigma_1) - d + (\mu_2 - \mu_1)^T \Sigma_2^{-1} (\mu_2 - \mu_1) + \log \frac{\det \Sigma_2}{\det \Sigma_1} \right]
$$

当 $\Sigma_1 = \Sigma_2 = \Sigma$ 时：

- $\text{tr}(\Sigma^{-1} \Sigma) = \text{tr}(I) = d$，与 $-d$ 相消
- $\log \frac{\det \Sigma}{\det \Sigma} = \log 1 = 0$

只剩均值项：

$$
D_{KL}(P \| Q) = \frac{1}{2} (\mu_1 - \mu_2)^T \Sigma^{-1} (\mu_1 - \mu_2)
$$

当 $\Sigma = \sigma_t^2 I$ 时，$\Sigma^{-1} = \frac{1}{\sigma_t^2} I$，进一步简化为：

$$
D_{KL}(P \| Q) = \frac{\| \mu_1 - \mu_2 \|^2}{2 \sigma_t^2}
$$

**应用引理**：在 DDIM 中，两个高斯分布的均值分别是用 $x_0$ 和 $f_\theta^{(t)}(x_t)$ 计算的，由于均值表达式对 $x_0$ 是线性的，均值之差恰好正比于 $x_0 - f_\theta^{(t)}(x_t)$。因此，对于 $t > 1$：

$$
D_{KL}(q_\sigma \| p_\theta^{(t)}) \equiv \frac{\| x_0 - f_\theta^{(t)}(x_t) \|^2}{2 \sigma_t^2}
$$

同理，对于 $t = 1$ 项（论文将 $-\log p_\theta^{(1)}$ 也视为类似形式的 KL 散度）：

$$
-\log p_\theta^{(1)}(x_0 | x_1) \equiv \frac{\| x_0 - f_\theta^{(1)}(x_1) \|^2}{2 \sigma_1^2}
$$

##### Step 3：将预测 $x_0$ 转化为预测噪声 $\epsilon$

根据论文公式 (4) 和 (9)：

$$
x_t = \sqrt{\alpha_t} x_0 + \sqrt{1-\alpha_t} \epsilon
$$

$$
f_\theta^{(t)}(x_t) = \frac{x_t - \sqrt{1-\alpha_t} \cdot \epsilon_\theta^{(t)}(x_t)}{\sqrt{\alpha_t}}
$$

把 $x_t$ 和 $f_\theta^{(t)}$ 的表达式代入上一步的 $\| x_0 - f_\theta^{(t)}(x_t) \|^2$ 中，展开化简：

**第 1 步：代入 $x_t$ 的表达式**

$$
x_0 - f_\theta^{(t)}(x_t) = x_0 - \frac{x_t - \sqrt{1-\alpha_t} \cdot \epsilon_\theta^{(t)}(x_t)}{\sqrt{\alpha_t}} = x_0 - \frac{(\sqrt{\alpha_t} x_0 + \sqrt{1-\alpha_t} \epsilon) - \sqrt{1-\alpha_t} \cdot \epsilon_\theta^{(t)}(x_t)}{\sqrt{\alpha_t}}
$$

**第 2 步：通分，把 $x_0$ 写成 $\frac{\sqrt{\alpha_t} x_0}{\sqrt{\alpha_t}}$**

$$
= \frac{\sqrt{\alpha_t} x_0 - \left[ (\sqrt{\alpha_t} x_0 + \sqrt{1-\alpha_t} \epsilon) - \sqrt{1-\alpha_t} \cdot \epsilon_\theta^{(t)}(x_t) \right]}{\sqrt{\alpha_t}}
$$

**第 3 步：展开分子，$\sqrt{\alpha_t} x_0$ 相消**

$$
= \frac{\sqrt{\alpha_t} x_0 - \sqrt{\alpha_t} x_0 - \sqrt{1-\alpha_t} \epsilon + \sqrt{1-\alpha_t} \cdot \epsilon_\theta^{(t)}(x_t)}{\sqrt{\alpha_t}} = \frac{\sqrt{1-\alpha_t} \left( \epsilon_\theta^{(t)}(x_t) - \epsilon \right)}{\sqrt{\alpha_t}}
$$

**第 4 步：整理得到最终形式**

$$
x_0 - f_\theta^{(t)}(x_t) = \frac{\sqrt{1-\alpha_t}}{\sqrt{\alpha_t}} \left( \epsilon_\theta^{(t)}(x_t) - \epsilon \right)
$$

**第 5 步：取二范数平方**（利用 $\| c \cdot v \|^2 = c^2 \cdot \| v \|^2$）

$$
\| x_0 - f_\theta^{(t)}(x_t) \|^2 = \frac{1-\alpha_t}{\alpha_t} \| \epsilon_\theta^{(t)}(x_t) - \epsilon \|^2
$$

**关键洞察**：这一步的精妙之处在于——原本在像素空间 $x_0$ 上的预测误差，通过重参数化转化为噪声空间 $\epsilon$ 上的预测误差。$\frac{1-\alpha_t}{\alpha_t}$ 是一个只与时间步 $t$ 有关的标量系数，会被吸收进 $\gamma_t$ 中。

##### Step 4：代回原式，整合系数

将 Step 3 的结果代回 Step 2 的期望中。对于 $t$ 时刻的项：

$$
\mathbb{E}\left[ \frac{\| x_0 - f_\theta^{(t)} \|^2}{2 \sigma_t^2} \right] = \mathbb{E}\left[ \frac{1-\alpha_t}{2 \sigma_t^2 \alpha_t} \| \epsilon_\theta^{(t)}(x_t) - \epsilon \|^2 \right]
$$

将所有 $t$ 从 $1$ 到 $T$ 的项累加起来：

$$
J_\sigma(\epsilon_\theta) \equiv \sum_{t=1}^T \frac{1-\alpha_t}{2 \sigma_t^2 \alpha_t} \cdot \mathbb{E}\left[ \| \epsilon_\theta^{(t)}(x_t) - \epsilon \|^2 \right]
$$

##### Step 5：对照 $L_\gamma$ 的定义，得出定理结论

论文公式 (5) 定义的 $L_\gamma$ 是：

$$
L_\gamma(\epsilon_\theta) = \sum_{t=1}^T \gamma_t \cdot \mathbb{E}\left[ \| \epsilon_\theta^{(t)}(\sqrt{\alpha_t} x_0 + \sqrt{1-\alpha_t} \epsilon_t) - \epsilon_t \|^2 \right]
$$

对比 Step 4 得到的系数，令：

$$
\gamma_t = \frac{1-\alpha_t}{2 \sigma_t^2 \alpha_t}
$$

（论文附录中写的是 $\frac{1}{2d\sigma_t^2 \alpha_t}$，其中 $d$ 是维度，概念上完全一致）。

因为 $\sigma_t > 0$ 且 $\alpha_t \in (0, 1]$，所以 $\gamma_t > 0$。

因此我们证明了：

$$
J_\sigma = L_\gamma + C
$$

其中 $C$ 包含了所有与 $\theta$ 无关的项（比如 $\log q(x_T | x_0)$ 和维度常数）。

#### 证明的精妙之处（背后的直觉）

1. **协方差锁定**：DDIM 巧妙地将 $p_\theta$ 的方差固定为与 $q_\sigma$ 完全相同。这使得复杂的 KL 散度退化为简单的均方误差（MSE），避免了训练的不稳定性
2. **$x_0$ 的重参数化**：通过将"预测 $x_0$"转为"预测噪声 $\epsilon$"，复杂的图像空间像素误差转化为了标准高斯噪声空间中的残差拟合，这与**得分匹配（Score Matching）**的理论完美契合
3. **结论的巨大威力**：由于 $\gamma_t$ 是由我们自由选择的 $\sigma_t$ 计算得来的，这意味着无论我们想采样 10 步还是 1000 步，只需要改变 $\sigma_t$ 和子序列 $\tau$，而**不需要重新训练神经网络** $\epsilon_\theta$。这就是 DDIM 能直接加载 DDPM 预训练模型进行加速采样的根本数学原因！

## Mathematical Formulation

见上文 Method 部分。关键公式编号：

- 公式 (1)：DDPM 逆向过程的因子分解 $p_\theta(x_{0:T}) = p(x_T) \prod p_\theta^{(t)}(x_{t-1} | x_t)$
- 公式 (4)：$x_t$ 的重参数化 $x_t = \sqrt{\alpha_t} x_0 + \sqrt{1-\alpha_t} \epsilon$
- 公式 (5)：DDPM 加权去噪损失 $L_\gamma$
- 公式 (6)：非马尔可夫前向过程的因子分解
- 公式 (9)：$f_\theta^{(t)}(x_t)$ 的定义
- 公式 (10)：$p_\theta^{(t)}(x_{t-1} | x_t) = q_\sigma(x_{t-1} | x_t, f_\theta^{(t)}(x_t))$
- 公式 (11)：变分目标 $J_\sigma$
- 定理 1：$J_\sigma = L_\gamma + C$

## Experiments

（待补——可后续补充论文中的加速采样实验、ImageNet 生成质量对比）

## Results

- 用 DDPM 预训练模型直接做 DDIM 采样，10~20 步即可生成可接受质量的图片
- $\sigma = 0$ 时是确定性采样（同一噪声 → 同一图片），便于可控生成
- $\sigma > 0$ 时退化为随机采样，类似 DDPM

## Limitations

- 仍然需要 DDPM 的完整训练成本
- 加速比有上限（不能少于 ~10 步而不损失质量）
- 确定性采样可能错过某些 mode

## My Understanding

（待补——需要结合实际跑过/调过 diffusion 模型的经验）

## What I Learned

1. **训练-推理解耦**的核心技巧：让训练目标对某些超参不敏感，推理时再自由调节
2. **协方差锁定**：把复杂 KL 散度简化为 MSE 的关键设计
3. **重参数化**：把预测目标从图像空间换到噪声空间，与 Score Matching 理论统一
4. **非马尔可夫过程不必然改变边际分布**：可以让联合分布变但边际不变，从而保留训练目标

## Related Knowledge

- [DDPM](../knowledge/ai/ddpm.md) — TODO（DDIM 的前置论文）
- [Score Matching](../knowledge/ai/score-matching.md) — TODO（DDIM 与之理论契合）
- [ELBO / VLB](../knowledge/ai/elbo.md) — TODO（公式 11 的本质）
- [KL 散度](./kl-divergence.md) — TODO（拆出独立 concept 笔记）
- [VAE](../knowledge/ai/vae.md) — TODO（变分推断的另一种应用，Forward/Reverse KL 的选择差异）

## Open Questions

- 为什么非马尔可夫前向过程的边际分布能保持不变？（需要更深的数学推导）
- 定理 1 证明中 $\alpha_t$ 的具体定义和约束？（需要回查 DDPM 原文）
- DDIM 采样时如何选择子序列 $\tau$？（实践层面的问题）
- 加速采样的最优步数和 $\sigma$ 调度策略是什么？
- Forward KL vs Reverse KL 在 VAE 和扩散模型中的具体选择差异？

## References

- 原论文 arXiv: https://arxiv.org/abs/2010.02502
- DDPM 原论文（Ho et al., 2020）: https://arxiv.org/abs/2006.11239
- DeepSeek 对话来源: https://chat.deepseek.com/share/kanpr0fferkd76bwlz
