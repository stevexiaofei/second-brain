---
title: AI Infra 方向论文地图（ChatGPT 对话整理）
type: concept
status: seed
tags: [AI Infra, LLM Serving, GPU Kernel, AI Compiler, vLLM, FlashAttention, Triton, Distributed Training, 论文推荐]
created: 2026-08-17
updated: 2026-08-17
source:
  - https://chatgpt.com/share/6a83d059-f1f8-83e8-8ccd-7fe9db8df487
---

# AI Infra 方向论文地图

> 本文整理自一次 ChatGPT 对话（"ai infra方向有哪些有意思的论文"），结合提问者已有的 RVV/算子/GPU/Docker/BEV/大模型基础，给出按 **Kernel → Compiler → Runtime → Serving → Distributed System → Hardware** 链路的论文阅读路线。原始对话见 [share 链接](https://chatgpt.com/share/6a83d059-f1f8-83e8-8ccd-7fe9db8df487)。

## 一句话理解

AI Infra = "让 AI 模型跑得更快、更便宜、更稳定" 的系统层。不要按论文年份读，而应按 **Kernel → Compiler → Runtime → Serving → Distributed GPU Cluster** 的层次读。

## 核心演进主线（最值得跟踪）

```text
PagedAttention（解决 KV Cache / Memory）
     ↓
   vLLM（解决 batching / scheduling）
     ↓
Sarathi-Serve（解决 prefill/decode 干扰 → chunked prefill）
     ↓
 DistServe（prefill/decode 分离 → disaggregated serving）
     ↓
AI Cluster
```

## 第一梯队：强烈推荐先读的 15 篇

### GPU Kernel / IO-aware（与 RVV 思维最相通）

| 论文 | 核心思想 | 关键概念 |
|---|---|---|
| **PagedAttention / vLLM**（SOSP 2023） | 把 OS 的 virtual memory / paging 思想搬到 GPU KV Cache | OS→Virtual Memory→Page Table 映射到 LLM→KV Cache→Block Table→GPU Memory；2–4× throughput |
| **FlashAttention** | IO-aware attention：不是算得快，而是**重新思考 GPU memory hierarchy** | tiling、IO complexity、SRAM、register、occupancy、CUDA kernel |
| **FlashAttention-2** | 第一版解决 IO，第二版解决 GPU parallelism | thread block、warp、SM、sequence/head parallelism、warp partition |
| **FlashInfer** | LLM GPU Kernel 库 | paged KV cache、variable length、batching、decode/prefill |

> FlashAttention 的思维方式（tiling + 数据布局 + memory access）与 RVV kernel 优化**高度相通**——做 RVV 算子的人读这篇收益最大。

### LLM Serving（目前 AI Infra 最有意思的方向之一）

| 论文 | 核心思想 | 关键概念 |
|---|---|---|
| **Orca** | iteration-level scheduling | **continuous batching** 的思想基础 |
| **Sarathi-Serve**（OSDI 2024） | chunked prefill：把 Prefill 切块，避免 decode 的 generation stall | Mistral-7B 单 A100 提升 2.6×，Yi-34B 双卡最高 3.7× |
| **DistServe** | **Disaggregated Serving**：Prefill/Decode 分离到不同 GPU 集群 | 类似 storage disaggregation / compute-storage separation |
| **Splitwise** | 和 DistServe 思路接近：compute-intensive 的 prefill 与 memory-intensive 的 decode 分到不同 GPU | GPU cluster 组织方式 |

### Distributed Training

| 论文 | 核心思想 | 关键概念 |
|---|---|---|
| **ZeRO**（DeepSpeed） | 把 Model/Optimizer/Gradient 跨 GPU partition，消除冗余 | 理解 DeepSpeed / FSDP / Megatron 的前置 |
| **Megatron-LM** | 3D 并行 | **TP / PP / DP** 三轴并行，所有后续系统的概念基础 |

### AI Compiler（喜欢 RVV 的人最适合这条线）

| 论文 | 核心思想 | 关键概念 |
|---|---|---|
| **TVM** | Automated end-to-end optimizing compiler | PyTorch→Graph IR→Tensor IR→Schedule→Codegen→CPU/GPU/Accelerator |
| **Ansor** | 自动寻找最优 kernel | search space→schedule→benchmark→cost model→search |
| **MLIR** | 多级 IR 编译器基础设施 | High-level IR→Tensor→Linalg→Vector→LLVM；做 RVV AI Compiler 绕不开 |
| **Triton** | 让 AI 工程师写 kernel 但不直面 CUDA 细节 | Python-like DSL→Triton IR→LLVM→PTX→GPU |

## 推荐的深度学习路线

```text
                AI Infra
                   │
       ┌───────────┼────────────┐
       │           │            │
     Kernel      Compiler     Distributed
       │           │            │
       ↓           ↓            ↓
 FlashAttention  Triton       Megatron
       │         MLIR           │
       ↓           │            ↓
 FlashInfer      TVM           ZeRO
       │           │            │
       └───────────┼────────────┘
                   ↓
                Runtime
                   │
                   ↓
                 vLLM
                   │
             ┌─────┴─────┐
             ↓           ↓
        Sarathi       DistServe
             │           │
             └─────┬─────┘
                   ↓
              AI Cluster
```

## 如果只选 10 篇

| 阶段 | 论文 |
|---|---|
| 一、GPU / Kernel | FlashAttention、FlashAttention-2、Triton |
| 二、AI Compiler | TVM、Ansor、MLIR |
| 三、LLM Runtime | Orca、PagedAttention / vLLM |
| 四、AI Cluster | Sarathi-Serve、DistServe |

## 6 个重点方向（结合已有基础）

| 方向 | 代表论文/项目 | 难度 | 推荐度 |
|---|---|---:|---:|
| GPU Kernel | FlashAttention | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| LLM Serving | vLLM | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Serving Scheduler | Sarathi-Serve | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Distributed Inference | DistServe | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| AI Compiler | TVM / MLIR | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Distributed Training | Megatron / ZeRO | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

## 思维升级点

- **Splitwise / DistServe** 标志着你从"怎么优化一个 kernel"升级到"整个 GPU cluster 应该怎么组织"——这是 **AI System / AI Infra 思维升级**
- **AI Infra 正在变成一种新的 Operating System**：Scheduler / Memory / Compiler 围绕 GPU 构成新的资源管理层
- 推荐学习路径串联：`C softmax → RVV softmax → CUDA softmax → Triton softmax → FlashAttention`，把 CPU SIMD / RVV / GPU / Compiler / Kernel 全部串起来

## 下一步建议（对话中的提议）

做一张 **AI Infra 50 篇论文地图**：按 GPU Architecture → CUDA Kernel → Compiler → Runtime → Training System → Inference → Serving → Distributed GPU Cluster 八层，每层挑 5–8 篇，标注"必须精读 / 了解思想 / 配合源码读"，并把 vLLM、Triton、FlashInfer、Megatron、DeepSpeed、TensorRT-LLM 的源码对应到论文。

## Related

- [RVV 算子开发必备基础知识](../knowledge/engineering/rvv-operator-development.md) — FlashAttention/Triton 思路与 RVV kernel 优化的相通点
- [RVV 算子设计大赛备考指南](../knowledge/engineering/rvv-operator-challenge.md) — 算子实现经验
- [分布式存储系统知识地图](../knowledge/engineering/distributed-storage-knowledge-map.md) — AI Infra 的另一面（数据层）
- [思维链（CoT）论文详解](./chain-of-thought-papers.md) — 算法层论文的另一条主线
