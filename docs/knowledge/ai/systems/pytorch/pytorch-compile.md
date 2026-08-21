---
title: torch.compile 编译栈（Dynamo + Inductor）
type: concept
status: seed
tags: [pytorch, source-code]
created: 2026-08-11
updated: 2026-08-11
---

# torch.compile 编译栈（Dynamo + Inductor）

## 一句话理解

> torch.compile 通过 TorchDynamo 钩入 CPython 帧求值 API（PEP 523）把 PyTorch 程序的字节码符号化为受 Guard 保护的 FX 图，再由默认后端 TorchInductor 将其 lower 成融合的 Triton/C++ 内核，构成 PyTorch 新一代 Python 级 JIT 编译栈。

## 为什么重要

- **取代 TorchScript**：Dynamo/Inductor 在训练与推理优化上基本取代了 TorchScript，成为 PyTorch 主流的编译加速路径；TorchScript 现主要作为 C++ 运行时的部署 IR 保留。
- **未修改程序即可加速**：Dynamo 的目标是"让未修改的 PyTorch 程序更快"，无需用户重写模型。
- **算子融合 + 通信/计算重叠**：Inductor 通过融合 pass 与内核生成大幅减少内核启动与访存开销，是大规模训练性能的关键。
- **可定制后端**：后端可插拔（Inductor、TensorRT、TVM、ONNX Runtime、TorchXLA、cuDGRAPH 等），同一套捕获机制服务多种目标。

## 核心概念

### TorchDynamo —— 捕获层
- **PEP 523 帧求值钩子**：Dynamo 安装自定义帧求值器，在执行前动态修改 Python 字节码（`torch/_dynamo/eval_frame.py`）。
- **符号字节码解释**：`symbolic_convert.py` 用 `variables/` 下的变量跟踪器（tensor、nn_module、builder、user_defined、higher_level_ops、optimizer 等）遍历字节码，把 PyTorch 算子序列提取成 FX `GraphModule`（`convert_frame.py`、`output_graph.py`）。
- **Guard（守卫）**：`guards.py` 为生成的图附加假设（形状、dtype、Python 值等）；假设失败时触发重编译或回退 eager，这是 Dynamo 安全性的基石。
- **Graph break**：遇到不支持的构造时拆分图，混合 Python 执行与编译后端，而非整体失败。
- **compiled_autograd**：`compiled_autograd.py` 可编译反向传播，把 autograd 图也纳入编译。

### TorchInductor —— 默认后端
- **入口**：`torch/_inductor/__init__.py` 暴露 `compile(gm, example_inputs, options)`，委托 `compile_fx.compile_fx`。
- **IR 与调度**：`lowering.py` 把 ATen 算子 lower 到 Inductor IR（`ir.py`），`graph.py` 构建计算图，`scheduler.py` 调度内核。
- **融合 pass**：`fx_passes/` 提供前/后梯度融合模式（`fuse_attention.py`、`b2b_gemm.py`、`post_grad.py`、`fsdp.py`、`ddp_fusion.py`、`quantization.py` 等）。
- **代码生成**：`codegen/` 按后端生成内核——`triton.py`（CUDA）、`cpp.py`（CPU）、`cuda/`（CUTLASS 模板）、`rocm/`（CK 模板）、`mps.py` 等。
- **专用内核族**：`kernel/` 含 `mm.py`、`bmm.py`、`conv.py`、`flex/`（flex_attention）等高优内核。
- **CUDA graph**：`cudagraph_trees.py` 管理 CUDA graph replay，减少启动开销。
- **自动调优**：`runtime/` 含 coordinate-descent tuner 与 triton 启发式；`autoheuristic/` 学习 matmul 启发式（A100/H100 排名）。
- **AOT-Inductor**：`aoti_eager.py`、`package/`、`cpp_wrapper_*.py` 可持久化编译产物，用于无 Python 部署。

### FX 作为 IR 基质
Dynamo 产出 FX 图，Inductor 消费 FX `GraphModule`。FX 也是独立的图变换工具（量化、图手术、形状传播）。其 Python 原生 IR 人类可读且可往返源码，对开发者和编译器都理想。

## 工作原理

torch.compile 的端到端流水线如下：用户调用 `torch.compile(model)` 后，Dynamo 安装 PEP 523 帧求值器；首次运行时符号化字节码产出受 Guard 保护的 FX 图；FX 图交给后端（默认 Inductor），Inductor 做 lowering → 调度 → 融合 pass → 生成 Triton/C++ 内核，并常包装进 CUDA graph；AOT-Inductor 还可把产物持久化用于无 Python 部署。

```mermaid
flowchart TD
    s0["<b>0 <code>torch.compile(model)</code> → 安装 PEP 523 帧求值器</b><br/><small><code>torch/_dynamo/eval_frame.py</code> 把自定义 <code>_dynamo_eval_frame</code> 挂到每个正在解释的 Python 帧上，后续字节码执行优先经 Dynamo 的符号化路径。</small>"]
    s1["<b>1 首次命中：<code>convert_frame → symbolic_convert</code></b><br/><small>把字节码符号化地"跑一遍"，<code>variables/</code> 追踪 Python 对象，产出一份"能翻成 FX Graph"的执行序列；<b>graph break</b> 把无法符号化的部分切出回 Python。</small>"]
    s2["<b>2 受 Guard 保护的 FX GraphModule</b><br/><small>由 <code>output_graph.py</code> 组装 GraphModule；<code>guards.py</code> 生成一组 guard 表达式（形状、dtype、类型、参数 id…）用于缓存命中判定。<b>Guard 失败</b>时回到 Step 1 重新捕获。</small>"]
    s3["<b>3 后端选择：默认 TorchInductor（<code>compile_fx</code>）</b><br/><small>也可替换为 TensorRT / TVM / ONNX RT / TorchXLA 等自定义后端。</small>"]
    s4["<b>4 Inductor lowering → 内核调度 → fx_passes → codegen</b><br/><small><b>① lowering.py：</b>ATen 算子 → Inductor IR（<code>ir.py</code> / <code>graph.py</code>）<br/><b>② scheduler.py：</b>分组调度，决定哪些节点融合为一个内核<br/><b>③ fx_passes：</b><code>fuse_attention</code> · <code>b2b_gemm</code> · <code>post_grad</code> 等前后梯度大融合<br/><b>④ codegen/：</b><code>triton.py</code>(CUDA) · <code>cpp.py</code>(CPU) · CUTLASS / ROCm CK 生成最终内核代码</small>"]
    s5["<b>5 可选包装：CUDA Graph Trees · AOT-Inductor 持久化</b><br/><small><code>cudagraph_trees.py</code> 把生成内核包进 CUDA Graph 减启动；<b>AOT-Inductor</b>（<code>aoti_eager / package / cpp_wrapper</code>）把 Triton/C++ 产物持久化，实现<b>无 Python 部署</b>。</small>"]
    s0 --> s1 --> s2 --> s3 --> s4 --> s5
    class s0,s1,s2,s3,s4,s5 step

classDef step     fill:#eef2ff,stroke:#c7d2fe,color:#312e81,stroke-width:1.5px
classDef action   fill:#fff7ed,stroke:#fdba74,color:#7c2d12,stroke-width:1.5px
classDef decide   fill:#fef3c7,stroke:#fcd34d,color:#78350f,stroke-width:1.5px
classDef branchNo fill:#f0fdf4,stroke:#86efac,color:#166534,stroke-width:1.5px
classDef branchYes fill:#eef2ff,stroke:#c7d2fe,color:#3730a3,stroke-width:1.5px
```

> **正确性 > 性能：** Dynamo 的核心哲学是 Guard + graph break 保护边界，绝不"硬编"不保证的代码；Inductor 的收益主要来自**融合**（少 kernel launch + 少访存），FX 作为统一 IR 让"Guard / 图变换 / 编译 / 持久化"四层共享同一种图表达。

## 我的理解

- Dynamo 的核心难点是**正确性**：要在不改变语义的前提下把 Python 控制流、副作用、闭包等"剥"成纯 FX 图，Guard + graph break 是它妥协与安全的机制——宁可拆图或回退，也不编错。
- Inductor 的核心收益是**融合**：把多个点乘/激活/归约合成一个内核，省访存与启动开销；Triton 让 GPU 内核生成不需要手写 CUDA。
- FX 作为统一 IR 是整个新栈的"脊梁"：Dynamo、Inductor、量化、export 都建立在 FX 之上，这也是为什么 `torch.export` 与 `torch.compile` 共享大量基础设施。
- AOT-Inductor 与 `torch.export` 是互补的部署故事：export 负责"捕获稳定 IR"，Inductor 负责"把 IR 编成快内核"。

## Related

- [torch.fx 图捕获与变换](./pytorch-fx.md) — FX 是 Dynamo/Inductor 的 IR 基质，编译栈的"脊梁"
- [torch.distributed 分布式训练](./pytorch-distributed.md) — Inductor 的 `fx_passes/fsdp.py`、`ddp_fusion.py` 与分布式训练协同优化
- [torch.export 程序导出](./pytorch-export.md) — 共享 FX IR，export 产出可被 Inductor AOT 编译的稳定图
- [Reducer 类设计详解](./pytorch-reducer.md) — DDP 梯度归约协调器，与编译栈在分布式训练中互补

## References

- 源码目录 `torch/_dynamo/`、`torch/_inductor/`
- `torch/_dynamo/eval_frame.py`、`convert_frame.py`、`symbolic_convert.py`、`guards.py`、`output_graph.py`
- `torch/_inductor/compile_fx.py`、`lowering.py`、`ir.py`、`scheduler.py`、`codegen/triton.py`、`fx_passes/`
