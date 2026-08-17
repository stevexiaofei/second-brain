# 🤖 Artificial Intelligence

## 笔记索引

- [GRPO](./grpo.md) — Group Relative Policy Optimization，无需独立 Critic 的策略优化方法
- [PPO](./ppo.md) — Proximal Policy Optimization，信任区域近似的策略优化
- [DDIM (Denoising Diffusion Implicit Models)](./ddim-paper.md) — 扩散模型加速采样论文笔记
- [AI 开源项目源码精读指南](./ai-open-source-source-reading.md) — 12 个精读项目、Top 5 学习路线、Design Pattern 对照表
- [nanobot 源码阅读指南](./nanobot-source-reading-guide.md) — 按真实代码结构的阅读路径：入口 → 核心链路 → 三大注册中心 → 状态
- [nanobot 核心架构总览](./nanobot-architecture-overview.md) — 目录地图、分层架构、MessageBus 与七阶段管线
- [nanobot AgentLoop 与 AgentRunner 源码精读](./nanobot-agentloop-runner.md) — 七阶段 turn 管线、_run_core 迭代循环、injection、checkpoint、崩溃恢复
- [nanobot ContextBuilder 源码精读](./nanobot-contextbuilder.md) — system prompt 六大 section、图片 base64、role 交替合并
- [nanobot Providers Registry 源码精读](./nanobot-providers-registry.md) — ProviderSpec 元数据表、三种识别策略、各家 reasoning 参数方言
- [nanobot Tool Registry 源码精读](./nanobot-tool-registry.md) — tool call 的 coerce/cast/validate 三段式、schema 缓存、ToolResult
- [nanobot Channel Manager 源码精读](./nanobot-channel-manager.md) — 渠道插件发现、outbound 分发、流式 delta 合并、重试去重
- [Semantica — AI Agent 的知识图谱基础设施](./semantica.md) — Graph-Native 存储与推理层、决策智能、PROV-O 溯源
- [状态价值与 Bellman 方程](./state-values-and-bellman-equation.md) — MDP 中状态价值函数、Bellman 期望方程与最优方程

## Large Language Models

- Transformer
- Attention
- LLM
- Pre-training
- Post-training
- RLHF
- DPO
- PPO
- GRPO

## Reinforcement Learning

- MDP
- Policy Gradient
- Actor-Critic
- PPO
- GRPO
- RLOO

## Agents

- Tool Use
- Function Calling
- MCP
- Planning
- Memory
- Reflection
