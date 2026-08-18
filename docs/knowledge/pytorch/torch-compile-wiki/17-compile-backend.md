---
title: torch.compile 后端
type: concept
status: seed
tags: [PyTorch, torch.compile, 后端, backend, AOTAutograd, Inductor, Minifier]
created: 2026-08-17
updated: 2026-08-17
source:
  - d:\project\pytorch-2.8.0\wiki\17_compile_backend.html
---

# 十七、torch.compile 后端

> 后端协议、内置后端、自定义后端注册与 AOTAutograd 包装

`torch.compile` 通过 TorchDynamo 捕获 FX Graph 后，需要交给一个**后端（backend）**把图编译成可执行函数。默认后端是 TorchInductor。本章系统梳理后端协议、内置后端清单、自定义后端的注册方式、以及如何用 AOTAutograd 包装后端以支持训练，最后给出后端调试与最小化方法。

> **💡 提示：**
>
> **本章定位：**第九章深入了默认后端 Inductor 的内部流水线；第十四章 14.6 节给出了自定义后端注册的速查示例。本章是后端主题的完整专题——把"后端是什么、有哪些、怎么写、怎么调试"一次讲透。

### 17.1 后端协议与契约

TorchDynamo 在追踪出一个 FX Graph 后，会以一个**后端函数**调用它。后端函数的契约是：

```text
backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]) -> Callable
```

后端函数被 TorchDynamo 调用后，应返回一个与原始 `torch.fx.GraphModule.forward` 同契约的**可调用对象**：`(*args: torch.Tensor) -> List[torch.Tensor]`，且语义上等价于被追踪的 FX Graph。最简单的后端就是直接返回 `gm.forward`（相当于 eager，不做任何优化）。

```python
import torch

def my_custom_backend(gm, example_inputs):
    return gm.forward      # 不优化，直接 eager 执行

def f(x):
    return torch.relu(x) + 1

f_opt = torch.compile(f, backend=my_custom_backend)
```

### 17.2 内置后端清单

用 `torch._dynamo.list_backends()` 可查看可用的非实验性 in-tree 后端；`torch._dynamo.list_backends(None)` 可查看实验性/调试后端。常用后端如下：

| 后端 | 说明 | 前向/反向 |
| --- | --- | --- |
| `inductor`（默认） | TorchInductor：生成 Triton(GPU) / C++·OpenMP(CPU) 内核（见第九章） | 训练 + 推理 |
| `aot_eager` | 用 AOTAutograd 拆分前后向，但前后向都不做图优化（eager 执行） | 训练 + 推理 |
| `cudagraphs` | CUDA Graphs + AOTAutograd，降低 Python 开销 | 训练 + 推理 |
| `eager` | 调试用：Dynamo 捕获图后直接 eager，不走 AOTAutograd | 仅前向 |
| `aot_ts` | AOTAutograd + TorchScript 编译器 | 训练 + 推理 |
| `ipex` | Intel IPEX，CPU/GPU 优化 | 训练 + 推理 |
| `onnxrt` | ONNX Runtime，CPU/GPU 训练 | 训练 + 推理 |
| `tensorrt` | Torch-TensorRT，推理优化 | 推理 |
| `tvm` | Apache TVM，推理优化 | 推理 |

> **✨ 技巧：**
>
> **调试三板斧：**遇到 `torch.compile` 正确性问题，按 `eager` → `aot_eager` → `inductor` 顺序排查：若 `eager` 错则是 Dynamo 捕获问题；若 `aot_eager` 错则是 AOTAutograd/分解问题；若只有 `inductor` 错则是 Inductor 代码生成问题。

### 17.3 自定义后端注册

#### 方式一：register_backend 装饰器

用 `@register_backend` 装饰后，即可用字符串引用该后端：

```python
from torch._dynamo.backends.registry import register_backend

@register_backend
def my_compiler(gm: torch.fx.GraphModule, example_inputs):
    print("my_compiler() called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward

# 用字符串引用
model = torch.compile(model, backend="my_compiler")
```

#### 方式二：entry_points（外部包注册）

若后端在另一个 Python 包中，可通过 Python 包的 `entry_points` 注册，使包安装后即可被发现。在包的 `setup.py` 中加入：

```python
setup(
    ...
    entry_points={
        'torch_dynamo_backends': [
            'my_compiler = your_module.submodule:my_compiler',
        ]
    },
    ...
)
```

调用 `torch.compile(model, backend="my_compiler")` 时，PyTorch 先在 `register_backend` 注册表中查找；找不到则继续在所有经 `entry_points` 注册的后端中查找。

> **📝 说明：**
>
> **注册的两个用途：**① 可用字符串而非函数对象传给 `torch.compile`；② **minifier 必需**——minifier 生成的代码会用 `import` 语句触发后端注册，故后端必须可被 import 注册。

### 17.4 用 AOTAutograd 包装后端（支持训练）

直接传给 TorchDynamo 的后端只能拿到**前向图**，无法支持训练。要让后端支持训练，需让它在 AOTAutograd *之后*被调用——此时后端拿到的是**分离后的前向/反向图**，且由 canonical Aten 算子组成（算子集显著小于整个 torch/Aten 算子集）。

用 `torch._dynamo.backends.common.aot_autograd` 包装后端，通过 `fw_compiler` / `bw_compiler` 传入；若不指定 `bw_compiler`，默认与 `fw_compiler` 相同：

```python
from torch._dynamo.backends.common import aot_autograd
from functorch.compile import make_boxed_func

def my_compiler(gm, example_inputs):
    # AOTAutograd 要求返回的编译函数是 "boxed" 的
    return make_boxed_func(gm.forward)

# 包装：fw/bw 都用 my_compiler 编译
my_backend = aot_autograd(fw_compiler=my_compiler)

model_opt = torch.compile(model, backend=my_backend)
```

> **⚠️ 注意：**
>
> **boxed func 要点：**AOTAutograd 要求后端返回的编译函数是 "boxed" 的，需用 `functorch.compile.make_boxed_func` 包裹 `gm.forward`。这是常见的踩坑点。复用 AOTAutograd + Inductor 内层的完整示例见第十四章 14.6。

### 17.5 自定义后端示例集

#### 示例 A：打印图 + eager 执行（调试用）

```python
@register_backend
def print_graph_backend(gm, example_inputs):
    print(gm.graph)                       # 打印 FX 图
    print(f"节点数: {len(gm.graph.nodes)}")
    return gm.forward              # 不优化，直接 eager 执行

model = torch.compile(model, backend="print_graph_backend")
```

#### 示例 B：自定义图优化 Pass

```python
@register_backend
def my_opt_backend(gm: torch.fx.GraphModule, example_inputs):
    # 在这里可做任意 FX 图改写（见第十五章 15.5 图改写模式）
    for n in gm.graph.nodes:
        if n.op == "call_function" and n.target == torch.add:
            # 例如：把某些算子替换为自定义融合算子
            ...
    gm.recompile()                       # 重新代码生成
    return gm.forward
```

#### 示例 C：复用 Inductor 内层编译（自定义配置）

```python
from torch._functorch.aot_autograd import aot_autograd
from torch._inductor.compile_fx import compile_fx_inner
import torch._inductor.config as config

@register_backend
def my_inductor_variant(gm, example_inputs):
    config.coordinate_descent_tuning = True     # 自定义 Inductor 配置
    return aot_autograd(
        fw_compiler=compile_fx_inner,
        bw_compiler=compile_fx_inner,
        decompositions=torch._inductor.select_decomp_table(),
    )(gm, example_inputs)

model = torch.compile(model, backend="my_inductor_variant")
```

### 17.6 后端调试与最小化（Minifier）

当后端产生错误结果或崩溃时，**minifier** 能自动把触发问题的完整模型缩减到最小复现用例，极大方便 bug 上报与定位。开启方式：

```text
# 方式一：环境变量
TORCHDYNAMO_REPRO_AFTER="compile" python my_script.py
# 或 REPRO_LEVEL=3 (在 compile 之后插入复现脚本生成)

# 方式二：代码内
import torch._dynamo
torch._dynamo.repro_after("compile")
torch._dynamo.repro_level(3)
```

minifier 会在触发后生成一个 `minified_launcher.py`，其中含一段最小化代码与 `torch.compile(model, backend="your_backend")` 调用。**这就是为何自定义后端必须可被 import 注册**——否则生成的复现脚本无法用字符串引用它。

### 17.7 后端调用链全景

```text
用户: torch.compile(model, backend="inductor")
  │
  ▼
TorchDynamo 捕获 FX Graph (gm: GraphModule, 前向图)
  │
  │  backend 字符串 -> lookup_backend() 查注册表 / entry_points
  ▼
backend_fn = lookup_backend("inductor")  ──►  inductor = aot_autograd(fw_compiler=compile_fx_inner, ...)
  │                                              │
  │                                              ▼
  │                                     AOTAutograd: 联合图追踪 + 最小割分区
  │                                              │  产出 fwd_gm, bwd_gm (canonical Aten)
  │                                              ▼
  │                                     compile_fx_inner(fwd_gm) + compile_fx_inner(bwd_gm)
  │                                              │
  │                                              ▼
  │                                     Inductor: Pre-grad → Lowering → Scheduler → Codegen
  │                                              │
  │                                              ▼
  │                                     可执行函数 (Triton / C++)
  ▼
backend_fn(gm, example_inputs) 返回编译后的可调用对象
  │
  ▼
TorchDynamo 把它包进改写后的 ByteCode (见第十六章)
  │
  ▼
运行时: Guard 通过则直接执行编译函数；否则重编译
```

> **✨ 技巧：**
>
> **后端可替换性的关键：**整个栈通过明确接口解耦——TorchDynamo 只产出 FX Graph，不关心后端怎么编译；AOTAutograd 只做前后向分离，不关心 `fw_compiler` 是 Inductor 还是自定义；Inductor 只接受 FX 图产出内核。任何一环都可被替换，这正是自定义后端能无缝接入的根本原因。

### 17.8 编写自定义后端的检查清单

#### ① 契约正确

- □ 签名为 `(gm, example_inputs) -> Callable`
- □ 返回的 callable 接收与原 forward 相同参数
- □ 输出与原 forward 语义等价

#### ② 训练支持

- □ 需反向 → 用 `aot_autograd` 包装
- □ 返回 boxed func：`make_boxed_func(gm.forward)`
- □ 指定 `fw_compiler` / `bw_compiler`

#### ③ 可发现性

- □ 用 `@register_backend` 或 `entry_points` 注册
- □ `torch._dynamo.list_backends()` 能列出
- □ 可被 `import`（minifier 必需）

#### ④ 调试友好

- □ 先用 `eager` / `aot_eager` 定位问题层级
- □ `TORCH_COMPILE_DEBUG=1` 落盘调试快照
- □ minifier 缩减复现用例

> **📝 说明：**
>
> **本章与第九、十四章的关系：**第九章讲 Inductor 内部流水线（默认后端的"内幕"）；第十四章 14.6 给出注册速查；本章把后端作为一个完整主题——协议、清单、注册、训练包装、调试、调用链——一次讲透。三者互为补充。

参考：`torch/_dynamo/backends/registry.py` `torch/_dynamo/backends/common.py` `torch/_functorch/aot_autograd.py` `torch/_inductor/compile_fx.py` `官方文档：Custom Backends`

## Related

- [16 TorchDynamo 深入](./16-torchdynamo-deep.md)
- [PyTorch 索引](../index.md)
