---
title: GRPO
type: concept
status: growing
tags: [AI, RL, LLM]
created: 2026-08-11
updated: 2026-10-20
source: 整理自 GRPO（DeepSeekMath 提出）学习过程
---

# GRPO

## 一句话理解

GRPO（Group Relative Policy Optimization）通过同一问题的一组采样结果构造相对优势信号，从而减少对独立 Critic / Value Model 的依赖。

## 为什么重要

它是理解现代 LLM 后训练、reasoning、PPO 与 reward optimization 的重要入口。

## 我的理解

可以先理解为：

```text
PPO
 ├── policy
 ├── reward
 └── advantage

GRPO
 ├── policy
 ├── group sampling
 ├── relative reward
 └── group-relative advantage
```

## Related

- [PPO](./ppo)
