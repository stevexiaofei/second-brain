---
title: torch.compile 完整编译流程总结
type: concept
status: seed
tags: [PyTorch, torch.compile, 编译流程, 缓存, 重编译]
created: 2026-08-17
updated: 2026-08-17
source:
  - d:\project\pytorch-2.8.0\wiki\13_pipeline.html
---

# 十三、完整编译流程总结

*首次编译、缓存复用与重编译三条路径*

本章把前面各章串联起来，从**运行时视角**完整剖析一个 `torch.compile` 模型从首次调用到稳态运行的全生命周期。理解三条路径——首次编译、缓存命中、Guard 失败重编译——的耗时分布，是诊断"为什么我的模型第一次慢、之后快"以及"为什么偶尔卡顿"的关键。

> **💡 提示：三条路径的本质：**Dynamo 的帧评估钩子每次被触发时，先在 C 层遍历 `CacheEntry` 链表执行 Guard 检查。命中则走**快速路径**（微秒级）；全部 miss 则触发**编译路径**（秒级）。重编译会向链表追加新条目，直至达到 `recompile_limit`（默认 8）后回退 eager。

### 13.1 路径一：首次调用——完整编译链路

下面逐步还原 `model = torch.compile(model, mode="max-autotune"); out = model(input)` 的首次执行：

```text
用户: model = torch.compile(model, mode="max-autotune")
      output = model(input)

┌─────────────────────────────────────────────────────────────────────┐
│ 步骤 1: torch.compile() 装饰阶段  [毫秒级]                           │
│   torch/__init__.py#L2466                                            │
│   → 创建 _TorchCompileInductorWrapper(mode, options, dynamic)        │
│   → torch._dynamo.optimize(backend, nopython=fullgraph)(model)       │
│   → 返回 OptimizedModule (仅包装, 尚未编译)                          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│ 步骤 2: 帧评估钩子触发  [微秒级]                                     │
│   torch/_dynamo/eval_frame.py (PEP 523)                              │
│   → model(input) 调用 forward → C 层钩子拦截                         │
│   → 查找 code object 缓存: 无缓存 → 进入编译                         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│ 步骤 3: Dynamo 字节码追踪  [十~百毫秒级]                             │
│   torch/_dynamo/convert_frame.py#L685                                │
│   → InstructionTranslator 逐条解释字节码                             │
│   → 遇到 torch.* → 在 OutputGraph 创建 FX 节点                       │
│   → 遇到不支持的代码 → 图断裂                                        │
│   → 产出: FX GraphModule + Guard 集合                                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│ 步骤 4: Guard 编译 + 调用后端  [毫秒级]                              │
│   → Guard 编译为 C 检查函数 (CheckFunctionManager, guards.py)        │
│   → FX GraphModule 传递给 inductor compile_fx()                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│ 步骤 5: Inductor Pre-grad 优化  [毫秒级]                             │
│   torch/_inductor/compile_fx.py#L1977                                │
│   → _recursive_pre_grad_passes(): 模式匹配、图变换                    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│ 步骤 6: AOTAutograd 联合追踪 + 分区  [百毫秒~秒级]                    │
│   torch/_functorch/aot_autograd.py                                   │
│   → 算子分解 (decompositions)                                        │
│   → 追踪前向+反向联合图                                               │
│   → min_cut_rematerialization_partition: 最小割分区                   │
│   → 产出: fwd_graph + bwd_graph                                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
┌──────────────────────────┐     ┌──────────────────────────┐
│ 步骤 7a: 前向编译        │     │ 步骤 7b: 反向编译        │
│  fw_compiler(fwd_gm)     │     │  bw_compiler(bwd_gm)     │
│                          │     │                          │
│  → Joint 图优化          │     │  → Joint 图优化          │
│  → GraphLowering:        │     │  → GraphLowering:        │
│    FX → IR (lowering.py) │     │    FX → IR               │
│  → Scheduler:            │     │  → Scheduler:            │
│    融合 + 排序           │     │    融合 + 排序           │
│  → Codegen:              │     │  → Codegen:              │
│    Triton / C++ 内核     │     │    Triton / C++ 内核     │
│  → 模板自动调优 (可选)   │     │                          │
│  → CUDA Graphs (可选)    │     │                          │
└──────────────┬───────────┘     └──────────────┬───────────┘
               │                                │
               └────────────┬───────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────┐
│ 步骤 8: 组装编译结果  [毫秒级]                                       │
│   → 前向可执行函数 + 反向可执行函数                                  │
│   → 生成包装字节码 (Dynamo codegen.py)                               │
│   → 存入 code object 缓存 (CacheEntry)                               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
                    执行编译后的代码
```

### 13.2 各阶段耗时与瓶颈分析

下表给出典型模型（如 ResNet/Transformer）在 GPU 上首次编译各阶段的量级耗时与主要开销来源：

| 阶段 | 典型耗时 | 主要开销 | 是否可并行/缓存 |
| --- | --- | --- | --- |
| 1. 装饰 + 钩子 | 毫秒级 | 对象包装、C 钩子安装 | 否 |
| 3. Dynamo 追踪 | 十~百毫秒 | 字节码解释、符号化、Guard 生成 | 否 |
| 4. Guard 编译 | 毫秒级 | Guard C 代码编译为 .so | 否 |
| 5. Pre-grad 优化 | 毫秒级 | FX 图遍历与改写 | 否 |
| 6. AOTAutograd 追踪+分区 | 百毫秒~秒级 | 联合图追踪、最小割求解 | 否 |
| 7. Lowering | 百毫秒级 | 逐节点 IR 构造 | 否 |
| 7. Scheduler | 百毫秒级 | 融合分析、依赖排序 | 否 |
| 7. Codegen | 百毫秒级 | 内核源码生成 | 源码可缓存 (code_hash) |
| 7. Triton 编译 | **秒~十秒级** | MLIR/LLVM 编译、autotune | Triton 缓存 |
| 7. 模板自动调优 | **十秒~分钟级** | 多配置 benchmark | 调优结果缓存 |
| 8. 组装 + CUDA Graph | 百毫秒级 | 字节码生成、Graph 录制 | 否 |

> **⚠️ 注意：编译时间主要瓶颈：**通常是 **Triton 内核编译**（MLIR→LLVM）与 **max-autotune 的模板自动调优**。前者通过 Triton 自身的 `~/.triton/cache` 缓存缓解；后者通过 Inductor 的 `code_hash` 缓存。首次冷启动时这两项可占总编译时间 80% 以上。

### 13.3 路径二：后续调用——缓存复用（快速路径）

```text
用户: output = model(input2)  # 第二次调用, 输入"形状一致"

┌──────────────────────────────────────────────────┐
│ 1. PEP 523 帧评估钩子触发                          │
│ 2. 查找 code object 缓存                          │
│ 3. 遍历 CacheEntry 链表                           │
│    → 执行 Guard 检查函数 (C 函数, 极快)            │
│    → Guard 全部通过 ✓                             │
│ 4. 直接执行缓存的编译字节码                         │
│    → 调用 Inductor 生成的 Triton/C++ 内核          │
│    → 无需重新编译                                  │
└──────────────────────────────────────────────────┘

耗时: ~微秒级 (Guard 检查) vs ~秒级 (首次编译)
```

快速路径完全在 C 层完成（Guard 检查 + 字节码跳转），不进入 Python 解释器，因此开销极低。这也是 `torch.compile` 能在稳态下接近手写 CUDA 性能的根本原因。

### 13.4 路径三：Guard 失败——重编译

```text
用户: output = model(input3)  # input3 形状/dtype/device 不同

1. 帧评估钩子触发
2. 遍历 CacheEntry 链表
   → CacheEntry[0]: Guard 检查 → 形状不匹配 ✗
   → CacheEntry[1]: Guard 检查 → 形状不匹配 ✗ (若有)
3. 缓存未命中 → 重新触发编译流程 (步骤 3-8)
4. 新编译结果 + 新 Guard 追加到缓存链表
5. 执行新编译的代码

注意: 重编译次数累计 >= recompile_limit (默认 8) 后
      → 放弃编译, 回退到 eager 模式执行
      → 触发 TORCH_LOGS=recompiles 警告
```

### 13.5 三条路径对比

| 维度 | 首次编译 | 缓存命中 | Guard 失败重编译 |
| --- | --- | --- | --- |
| 触发条件 | 无 CacheEntry | 某条 CacheEntry 的 Guard 全通过 | 所有 CacheEntry 的 Guard 均失败 |
| 执行位置 | Python 层 callback | C 层帧评估 | Python 层 callback |
| 典型耗时 | 秒~分钟级 | 微秒级 | 秒~分钟级 |
| 是否走 Dynamo 追踪 | 是 | 否 | 是 |
| 是否走 AOTAutograd+Inductor | 是 | 否 | 是 |
| 是否生成新 Guard | 是 | 否 | 是 |
| 是否追加 CacheEntry | 是 (链表头) | 否 | 是 (链表尾追加) |
| 是否受 recompile_limit 限制 | 否 (首次) | 否 | 是 (累计计数) |
| 可观测性 | TORCH_LOGS=dynamic | trace.compile | TORCH_LOGS=recompiles/guards |

### 13.6 编译函数的完整生命周期

```text
         ┌──────────────────────────────────────────────────────────┐
         │              torch.compile(model) 返回 OptimizedModule    │
         │              (未编译, 仅安装帧评估钩子)                    │
         └────────────────────────────┬─────────────────────────────┘
                                      │
                                      ▼
   第 1 次调用 ───────► [无缓存] ──► Dynamo 追踪 + Inductor 编译
                                      │
                                      ▼
                          ┌───────────────────────┐
                          │ CacheEntry[0] 入缓存   │
                          │ code + Guard           │
                          └───────────┬───────────┘
                                      │
   第 2 次调用 ──────────────────────►│ Guard 检查
                                      ▼
                            ┌──── 检查通过? ────┐
                            │                   │
                           是                  否
                            │                   │
                            ▼                   ▼
                       [快速路径]         [重编译路径]
                       执行缓存字节码      追加 CacheEntry[N]
                       (微秒级)           (秒级, 计数+1)
                            │                   │
                            │                   ▼
                            │           累计 < 8? ──── 是 ──► 执行新代码
                            │                   │
                            │                   否
                            │                   ▼
                            │           [回退 eager]
                            │           放弃编译, 原生执行
                            │           触发 recompiles 警告
                            │                   │
                            └────────┬──────────┘
                                     │
                                     ▼
                            后续调用重复"快速路径"
                            或"重编译路径"
```

### 13.7 异步编译 (Async Compilation)

对于 **反向传播**，Inductor 支持异步编译，让前向图先编译并执行，反向图在后台并行编译。相关配置在 `torch/_inductor/compile_fx.py`：

- **动机：**训练循环中前向先执行，反向稍后才执行。若前向编译完即可运行，反向可在前向执行期间并行编译，隐藏编译延迟
- **触发：**默认对 CUDA 反向图启用异步编译（受 `config.compile_fx` 内部逻辑控制）
- **实现：**反向图编译提交到后台线程池，主线程立即返回一个 `Boxed` callable；首次反向调用时若编译未完成则阻塞等待
- **收益：**显著缩短首次训练迭代的总等待时间（前向编译 + 反向编译 → max(前向, 反向) + 前向执行）
- **限制：**异步编译的反向图在首次调用时仍可能等待；CUDA Graph 录制需所有内核就绪

> **✨ 技巧：异步编译与 CUDA Graph：**启用 `reduce-overhead` 时，CUDA Graph 录制需要前向所有内核编译完成。异步编译只针对反向，不影响前向的 Graph 录制时序。

### 13.8 错误处理与回退机制

torch.compile 在多个层级设有容错回退，确保编译失败不会让用户程序崩溃：

| 层级 | 失败场景 | 回退行为 | 相关配置/日志 |
| --- | --- | --- | --- |
| Dynamo 追踪 | 遇到不支持的 Python 特性 | 图断裂，分段编译 + 中间 eager | `fullgraph=False` (默认) |
| Dynamo 追踪 | 异常崩溃 | 回退整个帧到 eager，打印警告 | `TORCH_LOGS=dynamic` |
| 重编译上限 | Guard 失败次数 ≥ limit | 回退 eager，不再尝试编译 | `recompile_limit=8` |
| Inductor 编译 | Lowering/Codegen 异常 | 抛出异常（默认）或回退 eager | `config.fallback_random` |
| Triton 编译 | 内核编译失败 | Inductor 报错；可降级到非模板路径 | 禁用 `max_autotune` |
| 整图禁用 | 调试需求 | `disable=True` 变 no-op | `TORCHDYNAMO_DISABLE=1` |

*容错回退伪代码*

```python
# convert_frame._compile 内部
try:
    graph = trace_bytecode(frame)         # 追踪
    if graph_break:
        if fullgraph:
            raise Unsupported(...)   # 严格模式直接报错
        else:
            split_and_compile(graph)      # 分段编译
    compiled = backend(graph)             # Inductor 编译
except Exception:
    log.warning("torch.compile failed, falling back to eager")
    return None                  # 回退 eager 执行
```

### 13.9 编译统计：torch._dynamo.utils.counters

Dynamo 与 Inductor 内部维护一组全局计数器，反映编译过程发生了什么：

*计数器使用*

```python
import torch._dynamo.utils as dynamo_utils

model = torch.compile(model)
out = model(input)   # 触发编译

# 查看 Dynamo 统计
print(dynamo_utils.counters)
# {'stats': {'unique_graphs': 1, 'graph_break': 0, ...},
#  'compilation': {...}, 'guards': {...}}

# 重置计数器
dynamo_utils.counters.clear()

# Inductor 的指标
from torch._inductor import metrics
print(metrics.generated_kernel_count)   # 生成的内核数
print(metrics.inductor_meta)
```

> **📝 说明：关键计数器：**`unique_graphs`（捕获的不同图数）、`graph_break`（图断裂次数）、`recompiles`（重编译次数）、`guards`（各类 Guard 计数）。若 `recompiles` 接近 8，说明存在形状/dtype 抖动，应考虑 `dynamic=True`。可用 `torch._dynamo.explain(fn)(*inputs)` 获取更详细的图断裂与编译分析报告。

### 13.10 实战诊断流程

### ① 编译太慢？

用 `TORCH_COMPILE_DEBUG=1` 看 `torchinductor_*/debug/` 下的各阶段时间。瓶颈多在 Triton 编译或 autotune。可先 `mode="default"` 验证，再开 `max-autotune`。

### ② 运行时偶尔卡顿？

用 `TORCH_LOGS=recompiles` 看是否有 Guard 失败触发重编译。常见原因是 batch size 或序列长度抖动，可用 `dynamic=True` 缓解。

### ③ 性能不达预期？

用 `TORCH_LOGS=output_code` 检查生成的内核数与融合情况；用 `TORCH_LOGS=fusion` 看融合决策。过多小内核说明融合不充分，可能存在图断裂。

### ④ 正确性问题？

用 `config.fallback_random=True` 排查 RNG 差异；用 `TORCH_LOGS=guards` 确认 Guard 是否覆盖了关键假设；逐步用 `fullgraph=False` 定位图断裂段。

## Related

- [12 代码生成](./12-code-generation.md)
- [14 配置与模式](./14-config-and-modes.md)
- [PyTorch 索引](../index.md)
