# 并行策略（5D Parallelism）

大模型训练的并行策略系列：把"5D 并行"从名词表展开成可推导的句子。姊妹篇是 [集合通信原语](../pytorch/pytorch-collective-communication.md)——那边是名词（AR/AG/RS/A2A 的语义与通信量），这边是句子（各范式如何组织这些原语）。

## 笔记列表

- [并行策略总览：5D 并行的组合逻辑](./parallelism-overview.md) — 从"单卡放不下/算不满"推导出五根轴；通信频率与物理拓扑的映射法则
- [张量并行：Megatron 式 TP](./tensor-parallel-megatron.md) — 列切→行切的分块结构、经典 TP 每层前向/反向各 2 次 AR、TP×FSDP mesh
- [序列并行 SP 与上下文并行 CP](./sequence-context-parallelism.md) — Megatron-SP 的激活布局、CP 的上下文分片与注意力通信方案
- [流水线并行：GPipe、1F1B 与零气泡](./pipeline-parallelism.md) — 气泡公式与调度演进（GPipe → 1F1B → interleaved → zero-bubble）、PyTorch pipelining
- [专家并行与 MoE](./expert-parallel-moe.md) — dispatch/combine、负载均衡、固定容量与 dropless、DeepSeek/Mixtral 架构区别及 EP 组合

## 学习路径

```text
总览（为什么要切、切哪里）→ TP（层内切分的数学结构）
→ SP/CP（序列激活与 attention 上下文）→ PP（调度与气泡）
→ EP/MoE（数据路由与负载均衡）
```

DP/FSDP/ZeRO 这条数据并行进化线不在此系列，见 [PyTorch 分布式训练](../pytorch/pytorch-distributed.md) 与 [FSDP 源码阅读指南](../pytorch/pytorch-fsdp.md)。

## Related

- [PyTorch 专题](../pytorch/) — 分布式栈的源码级基础
- [千卡集群网络拓扑与 NCCL 排查](../../../engineering/mlops-platform/thousand-gpu-network-topology.md) — 并行通信模式的物理承载
- [PyTorch 分布式训练性能分析](../../../engineering/profiling/07-pytorch-distributed-profiling.md) — 用 trace 验证本系列的通信代价
