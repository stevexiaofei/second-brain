---
title: MLOps 平台专题
type: map
status: seed
tags: [MLOps, Training Platform, Hybrid Cloud, Kubernetes, ACK, GPU]
created: 2026-10-29
updated: 2026-10-29
source: 个人整理
---

# MLOps 平台专题

## 这是什么

记录模型训练平台 / AI 基础设施的架构知识：算力调度、存储分层、任务入口、可观测性。核心问题是——**如何把分散的 GPU 算力组织成用户可以声明式使用的训练资源池**。

## 笔记

- [黑芝麻模型训练平台架构解析](./black-sesame-training-platform.md) — 混合云架构实例：双专线互联、双入口管理、ACK Edge Namespace 统一调度云上云下 GPU、NAS/CPFS/OSS 存储分层

## 与其他专题的关系

- [Kubernetes 入门 Wiki](../kubernetes/) — 训练平台的调度底座
- [分布式存储专题](../../distributed-systems/distributed-storage/) — 存储分层选型的通用原理
- [Profiling 专题](../profiling/) — 训练任务性能分析
- [PyTorch 分布式训练](../../ai/systems/pytorch/pytorch-distributed.md) — 平台承载的典型负载
