# 📥 Inbox

这里是第二大脑的入口。任何突然想到的东西、看到的文章、论文、代码经验、问题、想法，都可以先放这里。

不要一开始就纠结分类。

## Rules

1. 先记录
2. 后整理
3. 不要因为分类而停止记录
4. 定期让 AI 助手帮助整理

## Notes

> 这里仅保留尚待验证、扩展或消化的内容。已经形成稳定结构的笔记会迁移到 [Knowledge](../knowledge/)；项目类内容进入 [Projects](../projects/)。

- [AI Infra 方向论文地图](./ai-infra-papers-map.md) — Kernel → Compiler → Runtime → Serving → Cluster 的初步论文阅读线索；保留前需补齐一手来源并逐篇核验
  - 成熟度：待核验的阅读路线，不是已确认的论文结论
  - 来源：ChatGPT 对话整理
  - 关键词：`AI Infra`、`vLLM`、`FlashAttention`、`Triton`、`Sarathi-Serve`、`DistServe`、`ZeRO`、`Megatron`
- [NVIDIA CUDA 硬件与编程模型研究地图](./cuda-hardware-and-programming-model-map.md) — 从 GPU/SM/warp 到内存层次、同步、Tensor Core、异步编程和性能分析的待验证研究母地图
  - 成熟度：待逐项研读官方文档、阅读源码并通过实验 / profiler 验证
  - 来源：NVIDIA CUDA Programming Guide、Best Practices Guide、PTX 文档及架构调优指南
  - 关键词：`CUDA`、`SM`、`CTA`、`warp`、`SIMT`、`shared memory`、`occupancy`、`Tensor Core`、`TMA`、`WGMMA`
- [CUDA 初学者学习路径与最小实验](./cuda-beginner-learning-path.md) — CUDA 研究母地图的互补入门路线：从 Thread/Block/Grid/warp 到内存层次、GEMM、Softmax 和 FlashAttention
  - 成熟度：待运行和测量的最小实验路线，不与研究母地图静默合并
  - 来源：ChatGPT 分享对话
  - 关键词：`CUDA`、`Vector Add`、`Transpose`、`Reduction`、`GEMM`、`Softmax`、`FlashAttention`


## Example

```text
今天发现 Docker 在 NFS 上执行 git status 特别慢。
可能和 inode / metadata / network filesystem 有关。
```
