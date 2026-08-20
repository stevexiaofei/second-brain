---
title: PPO
type: concept
status: seed
tags: [AI, RL]
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
