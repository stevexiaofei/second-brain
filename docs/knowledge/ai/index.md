# 🤖 Artificial Intelligence

AI 知识按两条主线组织：**基础与方法**回答模型为什么有效，**系统与实现**回答它们如何在真实软件和硬件上运行。

## Foundations

- [AI Foundations](./foundations/) — 算法、模型与训练方法总入口
- [Reinforcement Learning](./foundations/reinforcement-learning/) — 状态价值、Bellman 方程、PPO 与 GRPO
- [Diffusion Models](./foundations/diffusion/) — 扩散模型与生成过程；当前从 DDIM 论文切入
- [Reasoning & Inference](./foundations/reasoning/) — Chain-of-Thought 及其推理方法演进

## Systems

- [AI Systems](./systems/) — AI 基础设施、源码阅读与工程实现总入口
- [AI 开源项目源码精读指南](./systems/ai-open-source-source-reading.md) — Framework、Runtime、Compiler、Distributed 与 GPU 项目路线
- [PyTorch 专题](./systems/pytorch/) — 框架架构、autograd、编译栈、分布式训练与源码阅读
- [FlashAttention 专题](./systems/flash-attention/) — 论文原理、系统地图、源码、kernel 与 PyTorch 接入
- [nanobot 专题](./systems/nanobot/) — Agent Runtime 的架构与组件源码精读
- [Semantica](./systems/semantica.md) — Agent 的知识图谱、溯源和确定性推理基础设施

## 目前的知识缺口

以下是后续应逐步补成原子笔记的基础主题，暂不创建空链接：

- LLM：Transformer、Attention、Pre-training、Post-training、RLHF、DPO
- Reinforcement Learning：MDP、Policy Gradient、Actor-Critic、GAE、RLOO
- Diffusion：DDPM、Score Matching、ELBO / VLB、KL 散度、VAE
- Agents：Tool Use、Function Calling、MCP、Planning、Memory、Reflection
