# 🤖 Artificial Intelligence

## 笔记索引

- [GRPO](./grpo.md) — Group Relative Policy Optimization，无需独立 Critic 的策略优化方法
- [PPO](./ppo.md) — Proximal Policy Optimization，信任区域近似的策略优化
- [DDIM (Denoising Diffusion Implicit Models)](./ddim-paper.md) — 扩散模型加速采样论文笔记
- [AI 开源项目源码精读指南](./ai-open-source-source-reading.md) — 12 个精读项目、Top 5 学习路线、Design Pattern 对照表
- [新代码库阅读导览](./codebase-reading-guide.md) — 通用源码阅读流程、系统图、关键状态、笔记输出模板
- [代码库审计 / 上手 / 深读 三分法](./codebase-review-modes.md) — 审计、快速上手、深读三种目标的阅读分流器
- [nanobot 源码阅读指南](./nanobot-source-reading-guide.md) — 按真实代码结构的阅读路径：入口 → 核心链路 → 三大注册中心 → 状态
- [nanobot 核心架构总览](./nanobot-architecture-overview.md) — 目录地图、分层架构、MessageBus 与七阶段管线
- [nanobot AgentLoop 与 AgentRunner 源码精读](./nanobot-agentloop-runner.md) — 七阶段 turn 管线、_run_core 迭代循环、injection、checkpoint、崩溃恢复
- [nanobot ContextBuilder 源码精读](./nanobot-contextbuilder.md) — system prompt 六大 section、图片 base64、role 交替合并
- [nanobot Providers Registry 源码精读](./nanobot-providers-registry.md) — ProviderSpec 元数据表、三种识别策略、各家 reasoning 参数方言
- [nanobot Tool Registry 源码精读](./nanobot-tool-registry.md) — tool call 的 coerce/cast/validate 三段式、schema 缓存、ToolResult
- [nanobot Channel Manager 源码精读](./nanobot-channel-manager.md) — 渠道插件发现、outbound 分发、流式 delta 合并、重试去重
- [Semantica — AI Agent 的知识图谱基础设施](./semantica.md) — Graph-Native 存储与推理层、决策智能、PROV-O 溯源
- [FlashAttention 阅读导览](./flash-attention-reading-guide.md) — FlashAttention 的回看路线图：训练态、推理态、总图、专题笔记的入口
- [FlashAttention 术语表与关键状态表](./flash-attention-glossary-and-state-table.md) — `softmax_lse`、`rng_state`、`cache_seqlens`、`block_table` 等核心状态索引
- [FlashAttention 系统地图](./flash-attention-system-map.md) — 论文→接口→C++→kernel→ATen→模型层→测试的完整闭环
- [FlashAttention 源码精读](./flash-attention-source-reading.md) — 论文→接口→C++→CUDA→kernel 的完整链路、在线 softmax、work partitioning、KV cache
- [FlashAttention 接口与 Autograd](./flash-attention-interface-and-autograd.md) — Python API、autograd Function、ctx 保存、packed / varlen / KV cache
- [FlashAttention Kernel 与 Launch 机制](./flash-attention-kernel-and-launch.md) — params、launch template、tile 计算、online softmax、split-KV 调度
- [FlashAttention Kernel 细节补充](./flash-attention-kernel-details.md) — dropouts、split combine、sequence-parallel、tile 级 RNG 语义
- [FlashAttention PyTorch ATen 接入层](./flash-attention-pytorch-aten-integration.md) — ATen backend、参数检查、RNG、ALiBi、dense / varlen 派发
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
