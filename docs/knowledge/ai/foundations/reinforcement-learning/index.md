# Reinforcement Learning

这一组笔记从价值函数的递推结构，走到稳定的策略优化，再走到面向 LLM 推理训练的组内相对优化。

## 推荐顺序

1. [状态价值与 Bellman 方程](./state-values-and-bellman-equation.md) — 理解长期回报如何递推
2. [PPO](./ppo.md) — 理解如何限制策略更新幅度
3. [GRPO](./grpo.md) — 理解如何用组内相对奖励减少对独立 Critic 的依赖

## 待补基础

以下主题尚未形成笔记：MDP、动作价值函数、Policy Gradient、Actor-Critic、GAE、RLHF、DPO、RLOO。

## Related

- [AI Foundations](../)
- [Reasoning & Inference](../reasoning/) — GRPO 等方法与推理模型训练的连接
