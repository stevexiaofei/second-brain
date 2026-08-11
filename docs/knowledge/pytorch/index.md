# 🔥 PyTorch 源码理解

> 基于 `pytorch-main` 仓库（v2.9.0a0）源码静态分析，覆盖架构、模块职责、关键类与运行方式。

## 架构与核心

- [项目概述](./pytorch-overview/) — PyTorch 是什么、设计哲学、核心组件
- [整体架构](./pytorch-architecture/) — 五层分层、分发机制、新旧编译栈
- [C++ 核心模块](./pytorch-cpp-core/) — c10、ATen、caffe2、torch/csrc 四大模块
- [Python 包结构](./pytorch-python-package/) — torch 顶层包与子包职责

## 主要子系统

- [自动微分 autograd](./pytorch-autograd/) — 反向模式 AD、autograd 图、Engine 引擎
- [神经网络 nn](./pytorch-nn/) — Module 基类、层、参数、函数式算子
- [优化器 optim](./pytorch-optim/) — SGD/Adam 等算法、学习率调度
- [TorchScript jit](./pytorch-jit/) — script/trace 捕获、IR、优化 pass
- [FX 图变换](./pytorch-fx/) — 符号追踪、Graph/Node IR、代码生成

## 编译栈与分布式

- [torch.compile](./pytorch-compile/) — Dynamo 字节码捕获 + Inductor 内核生成
- [分布式训练](./pytorch-distributed/) — DDP/FSDP/RPC、ProcessGroup、torchrun
- [程序导出 export](./pytorch-export/) — ExportedProgram IR、动态形状
- [Reducer 类设计详解](./pytorch-reducer/) — DDP 梯度归约协调器深度解析

## 工程基建

- [代码生成 torchgen](./pytorch-torchgen/) — 从 YAML schema 生成 C++/Python 绑定
- [依赖关系](./pytorch-dependencies/) — 模块依赖、第三方库、可选后端
- [构建与运行](./pytorch-build-run/) — 源码编译、Docker、测试、文档
- [关键类参考](./pytorch-key-classes/) — C++ 与 Python 核心类/函数速查表
