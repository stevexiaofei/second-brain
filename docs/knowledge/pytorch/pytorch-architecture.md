---
title: PyTorch 整体架构
type: concept
status: seed
tags: [pytorch, source-code]
created: 2026-08-11
updated: 2026-08-11
---

# PyTorch 整体架构

## 一句话理解

> PyTorch 是一个严格分层的系统：从后端无关的 `c10` 核心库，到 `ATen` 算子与分发器，再到 `torch/csrc` 的 JIT/autograd/distributed 子系统，最后经 `torch._C` 绑定暴露为 Python 顶层包；所有算子通过 `DispatchKey` 路由，所有绑定由 YAML schema 经 `torchgen` 单一来源生成。

## 为什么重要

架构分层决定了 "改一个东西要去哪里找"。理解这五层与分发机制，才能定位性能瓶颈、扩展自定义后端、或在编译栈（TorchScript / `torch.compile`）之间做选择。同时，"单一事实来源"（native_functions.yaml + derivatives.yaml + torchgen）是 PyTorch 能在数千算子、多后端下维持一致性的工程基石。

## 核心概念

### 五层分层架构

自底向上，PyTorch 可划分为五层。上层依赖下层，最底层 `c10` 不依赖任何后端或 Python：

<div class="diagram">
  <div class="v-steps" style="--dot-bg:#eef2ff;--dot-border:#6366f1;">
    <div class="step-row">
      <div class="step-dot" style="background:#eef2ff;border-color:#6366f1;color:#4338ca;">L5</div>
      <div class="step-body">
        <b>Python 用户层</b>
        <small><code>torch.nn</code> · <code>torch.optim</code> · <code>torch.distributed</code> · <code>torch.compile</code> · <code>torch.export</code> 等纯 Python 子包。</small>
      </div>
    </div>
    <div class="step-row">
      <div class="step-dot" style="background:#e0e7ff;border-color:#4f46e5;color:#3730a3;">L4</div>
      <div class="step-body">
        <b>Python 绑定层 · <code>torch._C</code></b>
        <small>由 <code>torch/csrc/Module.cpp</code> 统一注册所有 C++ 子系统到 CPython 扩展模块。</small>
      </div>
    </div>
    <div class="step-row">
      <div class="step-dot" style="background:#ddd6fe;border-color:#7c3aed;color:#4c1d95;">L3</div>
      <div class="step-body">
        <b>C++ 子系统层 · <code>torch/csrc</code></b>
        <small>JIT / Autograd 引擎 / Distributed(c10d+RPC) / Dynamo / Inductor / libtorch C++ 前端 API。</small>
      </div>
    </div>
    <div class="step-row">
      <div class="step-dot" style="background:#f3e8ff;border-color:#a855f7;color:#581c87;">L2</div>
      <div class="step-body">
        <b>ATen 张量运算库 · <code>aten/src/ATen</code></b>
        <small>原生算子实现（native/） + <code>Dispatcher</code>（基于 <code>DispatchKeySet</code> 路由后端内核）。</small>
      </div>
    </div>
    <div class="step-row">
      <div class="step-dot" style="background:#faf5ff;border-color:#c084fc;color:#6b21a8;">L1</div>
      <div class="step-body">
        <b>c10 最小 C++ 核心库</b>
        <small><code>TensorImpl</code> / <code>Storage</code> / <code>Device</code> / <code>ScalarType</code> / <code>DispatchKey</code> / <code>intrusive_ptr</code> · 零后端、零 Python 依赖。</small>
      </div>
    </div>
  </div>
  <div class="d-note">
    <b>依赖方向：</b>L5 → L4 → L3 → L2 → L1（上层依赖下层）。c10 是地基，所有编译/链接顺序先编 c10，再 ATen，再 torch/csrc，最后拼成 <code>torch</code> 库与 <code>torch._C</code> 扩展。
  </div>
</div>

### 分发机制（Dispatch）

PyTorch 的算子分发基于 **DispatchKey**。每个张量携带一个 `DispatchKeySet`，记录它所属的所有分发键（CPU、CUDA、Autograd、Meta、Sparse、Quantized、FuncTorchBatched 等）。当调用一个算子时，`Dispatcher` 根据输入张量的键集合查找并调用对应内核。算子通过 `TORCH_LIBRARY`/`TORCH_LIBRARY_IMPL` 宏注册（见 `aten/src/ATen/core/Library.h`）。复合分发键（`CompositeImplicitAutograd`、`CompositeExplicitAutograd`）允许一个内核服务所有后端。

### 单一事实来源（Single Source of Truth）

- `aten/src/ATen/native/native_functions.yaml` 声明所有原生算子的 schema；
- `tools/autograd/derivatives.yaml` 声明 autograd 导数公式；
- `torchgen` 在构建时解析这些 YAML，生成 CPU/CUDA/Meta/Autograd/Composite/Functionalization/Lazy/Python 绑定等所有代码。

一条 YAML 条目即可生成一切，避免手编数十个文件。

## 工作原理

### 各层职责与关键点

1. **c10 是基石**：`c10::TensorImpl` 持有 `Storage`（数据缓冲）、`DispatchKeySet`（分发键集合，决定算子路由）、`VariableVersion`（版本计数器，供 autograd 检测就地修改）和 `caffe2::TypeMeta`（数据类型）。c10 不依赖任何后端或 Python。

2. **ATen 建立在 c10 之上**：`at::Tensor` 是 `intrusive_ptr<c10::TensorImpl>` 的句柄；ATen 的 `native/` 目录实现算子，`core/dispatch/Dispatcher` 读取每个张量的 `DispatchKeySet` 来选择内核。

3. **caffe2 现在主要是构建编排层**：`caffe2/CMakeLists.txt` 把 c10 + ATen + torch/csrc 组装成 `torch_cpu`、`torch_cuda`/`torch_hip`、`torch` 等库，并运行代码生成。残留的运行时代码主要是序列化、线程池和性能内核（移动端/边缘端）。

4. **torch/csrc 是最顶层**：依赖 c10、ATen、JIT/autograd/distributed 库，通过单一入口 `torch/csrc/Module.cpp` 把所有 C++ 子系统注册到 Python 扩展模块 `_C` 中。它包含 JIT 编译器、autograd 引擎、分布式运行时、AOTI/Dynamo 以及 libtorch C++ 前端 API。

5. **Python 用户层**：`torch._C` 之上是 `torch.nn`、`torch.optim`、`torch.distributed` 等纯 Python 子包，提供高层 API。

### 新旧编译栈并存

- **旧栈**：TorchScript（`torch.jit`）—— script/trace 捕获模型为可序列化 IR，用于 C++/移动端部署。
- **新栈**：`torch.compile`（Dynamo + Inductor）—— 基于 PEP 523 字节码帧钩子捕获 FX 图，lowered 到 Triton/C++ 内核，支持 CUDA Graph 与 AOTI 提前编译。

新栈在训练/推理优化上已基本取代 TorchScript，但 TorchScript 仍是 C++/移动端部署的 IR。

### 顶层目录与架构层的映射

下表把顶层目录归入对应的架构角色（去重后的关键信息）：

| 目录 | 架构角色 |
| ---- | ---- |
| `c10/` | 第 1 层：最小 C++ 核心库（TensorImpl、Storage、Device、ScalarType、分发键、工具类型） |
| `aten/` | 第 2 层：ATen 张量运算库，原生算子实现 + 分发器 |
| `torch/csrc/` | 第 3 层：C++ 子系统（JIT/autograd/distributed/dynamo/inductor）+ Python 绑定源码 |
| `torch/`（非 csrc） | 第 4–5 层：Python 顶层包 + `lib/` 编译产物 |
| `caffe2/` | 构建编排层（CMakeLists.txt 组装主库）+ 序列化/线程池/性能内核 |
| `torchgen/` | 代码生成系统：从 YAML schema 生成 C++/Python 绑定（构建时运行） |
| `functorch/` | functorch（vmap、函数变换）C++ 扩展 |
| `tools/` | 构建辅助脚本、autograd 模板、pyi 生成、lint |
| `cmake/` | CMake 模块与辅助脚本 |
| `third_party/` | 第三方子模块（gloo、cpuinfo、onnx、fbgemm、cutlass 等） |
| `test/` | Python 与 C++ 测试套件 |
| `docs/` | Sphinx 文档源 |
| `.ci/`、`android/`、`benchmarks/` | CI、Android 构建、性能基准 |

顶层关键文件：`setup.py`（Python 包构建入口，配置 CMake/扩展/打包）、`CMakeLists.txt`（定义所有 `USE_*` 编译选项）、`version.txt`（版本号 `2.9.0a0`）、`pyproject.toml`（项目元数据与构建后端）。

## 我的理解

- **分层 = 依赖方向 = 编译顺序**。五层不仅是逻辑划分，也是实际编译/链接顺序：先编 c10，再 ATen，再 torch/csrc，最后拼成 `torch` 库与 `torch._C` 扩展。这解释了为什么 c10 必须零后端依赖——它是地基，不能反过来依赖上层。
- **DispatchKey 是 PyTorch 的"多态"实现**。不同于 OOP 的虚函数，PyTorch 用一个键集合做穷举式路由。一个张量可以同时携带 Autograd + CUDA + FuncTorchBatched 等多个键，分发器按优先级逐层剥离，这让 autograd、functionalization、批处理变换都能以"中间层"身份插入，而无需侵入每个算子实现。
- **caffe2 的 "隐退" 是个值得记住的工程演化**：曾经独立的框架现在退化为构建编排层，残留代码只剩序列化/线程池/性能内核。这说明 "合并胜过并存"——把构建系统收敛到一处降低了维护成本。
- **新旧编译栈并存不是技术债，是过渡策略**：TorchScript 在 C++/移动端部署仍有不可替代性，而 `torch.compile` 在训练优化上更强。两者共享部分 IR 概念但定位不同，短期内不会互相替换。
- **torchgen 的存在让 "加一个算子" 从工程问题降级为声明问题**。这是大型系统控制复杂度的典范：用代码生成守住 schema 与实现的一致性。

## Related

- [PyTorch 项目概述](./pytorch-overview/)
- [C++ 核心模块](./pytorch-cpp-core/)
- [Python 包结构](./pytorch-python-package/)

## References

- 源码仓库 `pytorch-main`，版本 `2.9.0a0`
- `caffe2/CMakeLists.txt`（构建编排入口）
- `torch/csrc/Module.cpp`（C++ 子系统注册到 Python 的单一入口）
- `aten/src/ATen/native/native_functions.yaml`（算子 schema 单一事实来源）
- `tools/autograd/derivatives.yaml`（autograd 导数公式）
