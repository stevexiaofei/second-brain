---
title: 状态价值与 Bellman 方程
type: concept
status: growing
tags: [AI, RL, MDP, Value Function, Bellman Equation]
created: 2026-08-17
updated: 2026-08-17
source:
  - https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning/blob/main/3%20-%20Chapter%202%20State%20Values%20and%20Bellman%20Equation.pdf
---

# 状态价值与 Bellman 方程

## 一句话理解

这一章回答的是：**在一个马尔可夫决策过程里，一个状态到底“有多好”**，以及这个“好不好”如何被拆成“眼前回报 + 未来价值”的递推关系。

## 核心概念

强化学习里，智能体不是只看当前一步，而是要评估一个状态在某个策略下的长期回报。

因此会引出两个最基础的量：

- **状态价值函数**：描述在状态 $s$ 下，沿着某个策略继续行动时，能获得多少期望累计回报
- **动作价值函数**：描述在状态 $s$ 采取动作 $a$ 后，再继续执行策略时，能获得多少期望累计回报

它们本质上都在回答同一个问题：
> 从现在开始，未来总共还能期待多少收益？

## 状态价值函数

状态价值函数通常记作 $V^\pi(s)$，表示在策略 $\pi$ 下，从状态 $s$ 出发的期望回报。

直观上：

- $V^\pi(s)$ 越大，说明这个状态在策略 $\pi$ 下越“值得待着”
- 它把未来所有可能的回报都折算成当前状态的一个分数

## Bellman 期望方程

Bellman 方程的核心思想是：

> 一个状态的价值，可以拆成“立刻得到的奖励”加上“下一状态的折扣价值”的期望。

对于状态价值函数，这种递推关系就是 **Bellman expectation equation**。

它的形式可以理解为：

```text
V^π(s) = E_π[ R_{t+1} + γ V^π(S_{t+1}) | S_t = s ]
```

这里：

- $R_{t+1}$ 是下一步立即获得的奖励
- $γ$ 是折扣因子
- $S_{t+1}$ 是下一状态
- 期望是对策略和环境随机性一起取的

这说明：状态价值不是孤立定义的，而是**自洽递推**出来的。

## Bellman 最优方程

如果我们不满足于“某个策略下的价值”，而是想知道**最优情况下**一个状态能有多好，就会得到 Bellman 最优方程。

它表达的是：

> 最优状态价值等于在所有动作中，选取能让未来总回报最大的那个动作所对应的价值。

也就是说，最优价值满足“最大化的递推关系”。

这也是动态规划和很多强化学习算法的理论基础。

## 和策略的关系

同一个状态，在不同策略下会有不同的价值。

所以：

- **策略决定行为方式**
- **价值函数衡量这个行为方式有多好**
- **Bellman 方程把策略和价值连接起来**

这也是强化学习里“policy”和“value”两条主线能够相互配合的原因。

## 为什么重要

Bellman 方程的重要性在于它把一个“看起来很长远”的问题，拆成了局部递推问题：

- 不需要一次性展开整条未来轨迹
- 可以把长期规划转成逐步更新
- 可以迭代逼近价值函数

这就是为什么 value iteration、policy evaluation、policy iteration 都离不开 Bellman 方程。

## 你应该记住的关键词

- Markov Decision Process（MDP）
- State Value Function $V^\pi(s)$
- Action Value Function $Q^\pi(s,a)$
- Bellman Expectation Equation
- Bellman Optimality Equation
- Discount Factor $γ$
- Dynamic Programming

## 我的一句话理解

Bellman 方程的精髓就是：**把长期回报写成“现在 + 未来”的递推形式**，这样就能用局部更新去处理全局最优问题。

## Related

- [PPO](./ppo.md)
- [GRPO](./grpo.md)
- [数学索引](../mathematics/index.md)
