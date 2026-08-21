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

- [AI Infra 方向论文地图](./ai-infra-papers-map.md) — Kernel → Compiler → Runtime → Serving → Cluster 的初步论文阅读路线，下一步需要补齐一手来源并扩展
  - 来源：ChatGPT 对话整理
  - 关键词：`AI Infra`、`vLLM`、`FlashAttention`、`Triton`、`Sarathi-Serve`、`DistServe`、`ZeRO`、`Megatron`
- [NVIDIA CUDA 硬件与编程模型研究地图](./cuda-hardware-and-programming-model-map.md) — 从 GPU/SM/warp 到内存层次、同步、Tensor Core、异步编程和性能分析的系统学习路线
  - 来源：NVIDIA CUDA Programming Guide、Best Practices Guide、PTX 文档及架构调优指南（待逐项研读与实验验证）
  - 关键词：`CUDA`、`SM`、`CTA`、`warp`、`SIMT`、`shared memory`、`occupancy`、`Tensor Core`、`TMA`、`WGMMA`


## Example

```text
今天发现 Docker 在 NFS 上执行 git status 特别慢。
可能和 inode / metadata / network filesystem 有关。
```
