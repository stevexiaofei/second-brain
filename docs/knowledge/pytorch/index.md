# 🔥 PyTorch 源码理解

> 基于 `pytorch-main` 仓库（v2.9.0a0）源码静态分析，覆盖架构、模块职责、关键类与运行方式。

## 架构与核心

- [项目概述](./pytorch-overview/) — PyTorch 是什么、设计哲学、核心组件
- [整体架构](./pytorch-architecture/) — 五层分层、分发机制、新旧编译栈
- [C++ 核心模块](./pytorch-cpp-core/) — c10、ATen、caffe2、torch/csrc 四大模块
- [Python 包结构](./pytorch-python-package/) — torch 顶层包与子包职责
- [FlashAttention PyTorch ATen 接入层](../ai/systems/flash-attention/flash-attention-pytorch-aten-integration.md) — ATen backend、参数检查、RNG、ALiBi、dense / varlen 派发

## 主要子系统

- [自动微分 autograd](./pytorch-autograd/) — 反向模式 AD、autograd 图、Engine 引擎
- [神经网络 nn](./pytorch-nn/) — Module 基类、层、参数、函数式算子
- [优化器 optim](./pytorch-optim/) — SGD/Adam 等算法、学习率调度
- [TorchScript jit](./pytorch-jit/) — script/trace 捕获、IR、优化 pass
- [FX 图变换](./pytorch-fx/) — 符号追踪、Graph/Node IR、代码生成

## 编译栈与分布式

- [torch.compile](./pytorch-compile/) — Dynamo 字节码捕获 + Inductor 内核生成
- [torch.compile Wiki 系列（17 章）](./torch-compile-wiki/01-architecture-overview.md) — 基于 PyTorch 2.8.0 源码的全链路精读
- [分布式训练](./pytorch-distributed/) — DDP/FSDP/RPC、ProcessGroup、torchrun
- [程序导出 export](./pytorch-export/) — ExportedProgram IR、动态形状
- [Reducer 类设计详解](./pytorch-reducer/) — DDP 梯度归约协调器深度解析

## torch.compile Wiki 系列（基于 PyTorch 2.8.0 源码）

- [01 架构总览](./torch-compile-wiki/01-architecture-overview.md) — 三大组件定位、字节码追踪设计理念、宏观数据流
- [02 torch.compile 入口](./torch-compile-wiki/02-torch-compile-entry.md) — 公共 API、_dynamo.optimize 分发、OptimizedModule 包装
- [03 TorchDynamo 前端](./torch-compile-wiki/03-torchdynamo-frontend.md) — PEP 523 帧钩子、字节码符号化追踪、OutputGraph 构建
- [04 Guard 系统](./torch-compile-wiki/04-guard-system.md) — Guard 本质、类型体系、C 函数编译、失败重编译
- [05 缓存机制](./torch-compile-wiki/05-cache-mechanism.md) — CacheEntry 链表、缓存查找流程、per-code-object 设计
- [06 图断裂](./torch-compile-wiki/06-graph-break.md) — 触发条件、处理流程、fullgraph 模式、subgraph 拼接
- [07 AOTAutograd 中间层](./torch-compile-wiki/07-aotautograd.md) — 联合图追踪、算子分解、dispatch 机制、子类处理
- [08 前向/反向分区策略](./torch-compile-wiki/08-partition-strategy.md) — 最小割算法与梯度重计算优化
- [09 TorchInductor 后端](./torch-compile-wiki/09-torchinductor-backend.md) — 编译流水线、compile_fx 入口、后端选择
- [10 Lowering: FX → IR](./torch-compile-wiki/10-lowering-fx-to-ir.md) — GraphLowering 解释器、IR 节点体系、register_lowering
- [11 调度器与融合](./torch-compile-wiki/11-scheduler-fusion.md) — Scheduler 类、融合策略、依赖分析、FusedSchedulerNode
- [12 代码生成](./torch-compile-wiki/12-code-generation.md) — Triton GPU 内核、C++ CPU 代码、CUDA Graphs、CodeGen 基类
- [13 完整编译流程总结](./torch-compile-wiki/13-full-compile-pipeline.md) — 首次编译、缓存复用、Guard 失败重编译三条路径
- [14 配置与模式](./torch-compile-wiki/14-config-and-modes.md) — 4 种编译模式、关键配置项、调试工具、自定义后端
- [15 torch.fx 专题](./torch-compile-wiki/15-torch-fx-special.md) — 符号追踪、Proxy 与 __torch_function__、6 种 Node、GraphModule
- [16 TorchDynamo 深入](./torch-compile-wiki/16-torchdynamo-deep.md) — CPython 执行流程、三种图捕获方式、ByteCode rewrite 实战
- [17 torch.compile 后端](./torch-compile-wiki/17-compile-backend.md) — 后端协议、内置后端清单、自定义注册、Minifier 调试

## 工程基建

- [代码生成 torchgen](./pytorch-torchgen/) — 从 YAML schema 生成 C++/Python 绑定
- [依赖关系](./pytorch-dependencies/) — 模块依赖、第三方库、可选后端
- [构建与运行](./pytorch-build-run/) — 源码编译、Docker、测试、文档
- [关键类参考](./pytorch-key-classes/) — C++ 与 Python 核心类/函数速查表
