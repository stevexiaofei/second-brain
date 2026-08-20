# FlashAttention 专题

这组笔记把 FlashAttention 分成两条互相校验的主线：**论文解释算法为什么成立**，**源码解释算法如何在 GPU 和 PyTorch 中落地**。

## 入口

- [FlashAttention 阅读导览](./flash-attention-reading-guide.md) — 按目标选择最短阅读路径
- [FlashAttention 系统地图](./flash-attention-system-map.md) — 论文、接口、C++、kernel、ATen、模型层与测试的连接
- [术语表与关键状态表](./flash-attention-glossary-and-state-table.md) — 回看 `softmax_lse`、`rng_state`、`cache_seqlens` 等关键状态

## 论文原理

- [FlashAttention 三篇论文精读导览](./flashattention-paper-series.md) — FA1、FA2、FA3 的 IO-aware、work partitioning 与 Hopper 异步流水线

这是一篇约 1200 行的综合笔记。本轮保留其统一推导和跨版本比较；后续可拆成三篇独立论文笔记，再让原文只承担导览与对比。

## 源码与集成

建议按层次阅读：

1. [FlashAttention 源码精读](./flash-attention-source-reading.md) — 先建立仓库与核心链路全局图
2. [接口与 Autograd](./flash-attention-interface-and-autograd.md) — Python API、ctx、packed / varlen 与训练态
3. [Kernel 与 Launch 机制](./flash-attention-kernel-and-launch.md) — params、launch specialization、tile 与 split-KV
4. [Kernel 细节补充](./flash-attention-kernel-details.md) — backward、RNG、sequence parallel 与 split combine
5. [PyTorch ATen 接入层](./flash-attention-pytorch-aten-integration.md) — backend、参数检查、RNG 与派发

## Related

- [AI Systems](../)
- [PyTorch 源码理解](../../../pytorch/)
- [源码阅读方法](../../../learning/code-reading/)
