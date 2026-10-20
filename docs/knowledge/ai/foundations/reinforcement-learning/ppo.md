---
title: PPO
type: concept
status: seed
tags: [AI, RL]
created: 2026-08-11
updated: 2026-10-20
source: 整理自 PPO（Proximal Policy Optimization, Schulman et al., 2017）学习过程
---

# PPO

## 一句话理解

PPO（Proximal Policy Optimization）通过限制新旧策略的更新幅度，使策略优化更加稳定。

## 核心思想

典型 PPO 使用概率比值：

```text
r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)
```

并通过 clipping 限制策略变化。

## Related

- [GRPO](./grpo)
