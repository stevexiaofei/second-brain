# AI Systems

这里关注 AI 模型如何在框架、编译器、运行时、kernel、分布式系统和 Agent 基础设施中真正运行。

## 学习路线

- [AI 开源项目源码精读指南](./ai-open-source-source-reading.md) — 从 Framework、Runtime、Compiler、Distributed 到 GPU 的项目选择地图
- [GPU 算子优化方法论：计算、通信、存储](./gpu-kernel-optimization-methodology.md) — 性能瓶颈的排查顺序与三层存储约束
- [GPU 全局内存访存模型：向量化与合并访存](./gpu-memory-access-model.md) — sector / transaction 模型与实际访存量的放大
- [PyTorch 专题](./pytorch/) — 框架架构、autograd、编译栈、分布式训练与源码阅读
- [FlashAttention 专题](./flash-attention/) — 从 IO-aware 算法到 CUDA kernel 与 PyTorch 接入
- [CUTLASS / CuTe 专题](./cutlass/) — Tensor、Layout、TiledCopy、TiledMMA 与 GEMM 数据流
- [nanobot 专题](./nanobot/) — 从消息总线到 AgentLoop、工具、Provider 与 Channel
- [Semantica](./semantica.md) — 知识图谱、决策溯源与确定性推理基础设施

## 通用方法

通用的代码库阅读方法不属于 AI 专有知识，已归入 [源码阅读方法](../../learning/code-reading/)。

## Related

- [AI Foundations](../foundations/) — 系统实现背后的模型、算法和论文
- [AI 总览](../)
