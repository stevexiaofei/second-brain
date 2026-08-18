---
title: torch.fx 专题
type: concept
status: seed
tags: [PyTorch, torch.compile, torch.fx, 符号追踪, GraphModule, IR]
created: 2026-08-17
updated: 2026-08-17
source:
  - d:\project\pytorch-2.8.0\wiki\15_torch_fx.html
---

# 十五、torch.fx 专题

> Python-to-Python 代码变换工具链：Symbolic Trace → IR → Codegen

torch.fx 是 PyTorch 官方发布的 Python-to-Python 代码变换工具。它把 `nn.Module` 追踪成由 6 种基础节点组成的计算图，基于该图可方便地做各种变换，变换后的图再生成可执行的 `nn.Module`。torch 2.0 的 `torch.compile`（TorchDynamo）默认把代码转换为 torch.fx 的 `GraphModule`，进一步强化了 torch.fx 的重要性。

> **💡 提示：**
>
> **本章定位：**前 14 章聚焦 `torch.compile` 全链路，本章与[第十六章](./16-torchdynamo-deep.md)、[第十七章](./17-compile-backend.md)一起作为"专题补充"，深入 torch.fx 这个底层 IR 容器——它是 Dynamo 输出、AOTAutograd 输入、Inductor Lowering 起点的共同枢纽。理解 fx 是理解整个编译栈的前提。

### 15.1 三大基础功能

torch.fx 有三块基础功能，构成一条 Python-to-Python 变换流水线：

#### ① Symbolic Trace（图捕获）

- 把 `torch.nn.Module` 转换成 `fx.GraphModule`。用假输入（Proxy）走一遍 forward，记录执行路径，形成符号化计算图。

#### ② IR 与图改写（中间表达）

- 对 `fx.Graph`（节点列表）做增删查改：插入、删除、替换节点，遍历改写。这是 Pass 的载体。

#### ③ Python 代码生成（Codegen）

- 把变换后的 Graph 重新生成合法 Python 代码（即 `GraphModule.forward`），像普通 `nn.Module` 一样执行。

### 15.2 最小用例

定义一个有代表性的 `nn.Module`，覆盖 fx 要处理的 6 种基础操作（取参数、算子、子模块、tensor 方法）：

```python
import torch

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        return self.linear(x + self.param).clamp(min=0.0, max=1.0)

module = MyModule()
```

使用第一个基础功能 `symbolic_trace`，它把 torch python 代码转换成符号化表达 `fx.GraphModule`：

```python
from torch.fx import symbolic_trace

symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)
```

`fx.GraphModule` 执行时的行为和 `nn.Module` 相同，但同时内含一个可用图遍历操作的计算图。打印该图可以看到 IR 表达：

```python
print(symbolic_traced.graph)
"""
graph():
  %x      : [#users=1] = placeholder[target=x]
  %param  : [#users=1] = get_attr[target=param]
  %add    : [#users=1] = call_function[target=operator.add](args = (%x, %param), kwargs = {})
  %linear : [#users=1] = call_module[target=linear](args = (%add,), kwargs = {})
  %clamp  : [#users=1] = call_method[target=clamp](args = (%linear,), kwargs = {min: 0.0, max: 1.0})
  return clamp
"""
```

Graph 内含的计算图可再被转成 torch python 代码，即代码生成功能：

```python
print(symbolic_traced.code)
"""
def forward(self, x):
    param = self.param
    add = x + param;  x = param = None
    linear = self.linear(add);  add = None
    clamp = linear.clamp(min = 0.0, max = 1.0);  linear = None
    return clamp
"""
```

### 15.3 图生成：Symbolic Trace 原理

fx 构图的方法是 symbolic trace：把假输入传入 `nn.Module`，执行假输入时不是真执行，而是记录操作路径，最后形成完整执行记录即一张图。`symbolic_trace` 的实质是调用 `Tracer.trace` 返回 `fx.Graph`，再用该 Graph 和原 root 构造 `fx.GraphModule`：

```python
def symbolic_trace(root, concrete_args=None) -> GraphModule:
    tracer = Tracer()
    graph = tracer.trace(root, concrete_args)
    name = root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    return GraphModule(tracer.root, graph, name)
```

#### Proxy 与 __torch_function__ 协议

trace 的机制依赖于把输入转换成抽象值 **Proxy**。Proxy 起到代理 tensor 执行的作用——把 tensor 都转成 Proxy 在代码中传递，且 Proxy 可以输入常规 torch 操作。

Proxy 之所以能输入常规 torch 操作，依赖 `__torch_function__` 协议：一个类型支持了 `__torch_function__`，就可以传给 torch 的常规函数执行，执行时调用的逻辑在 `__torch_function__` 中定义。给 Proxy 的 `__torch_function__` 定义"记录操作到图"的逻辑，即可完成 trace。

```python
class ProxyTensor(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=None, kwargs=None):
        if func.__name__ == 'add':
            # 自定义加法行为：符号化而非真执行
            return args[0].symbolic() + " + " + args[1].symbolic()
        return super().__torch_function__(func, types, args=args, kwargs=kwargs)

    def symbolic(self):
        return "tensor(" + str(self.shape) + ")"

x = ProxyTensor([4, 5, 6])
y = ProxyTensor([1, 2, 3])
print(x - y)   # 减法：常规 tensor 运算 -> ProxyTensor([3., 3., 3.])
print(x + y)   # 加法：被拦截 -> tensor(torch.Size([3])) + tensor(torch.Size([3]))
```

Tracer 的 trace 实质就是做类似的事：把 `nn.Module` 输入转换成 graph 中的 Node，再把 Node 包装到 Proxy 作为新输入；torch 操作执行 Proxy 时触发自定义 `__torch_function__`，把操作记录为图中的 Node，并把 Node 包装为 Proxy 作为结果继续传递，如此便构造出计算图。

> **⚠️ 注意：**
>
> **trace 机制的局限性：**对于控制流、非 torch 内置操作，trace 会被 Python 真正执行但 trace "看不到"它们——`if` 只记录一个分支，`for` 被展开，Python 计算结果被当作常量传入 torch 操作。这是 trace 机制的根本局限，也是 TorchDynamo 选择字节码级别追踪（见[第十六章](./16-torchdynamo-deep.md)）以绕开该局限的动机。若希望某个自定义函数被当作内置操作 trace，可用 `torch.fx.wrap` 注册。

### 15.4 fx.Node 与 fx.Graph

`fx.Node` 和 `fx.Graph` 是 fx 中间表达的核心数据结构。每行 IR 对应一个 Node（`return` 对应 `output` 类型 Node）。Node 的 `op` 属性标识类型，共 6 种：

| op 类型 | 含义 | target 含义 |
| --- | --- | --- |
| `placeholder` | 整个被 trace 的 Module/函数的输入 | 参数名（字符串） |
| `get_attr` | 从 Module 上获取属性（如参数） | 属性的点分路径字符串 |
| `call_function` | 函数调用（torch op 或 Python 函数） | 函数对象本身 |
| `call_method` | 对象上的方法调用（如 `.clamp()`） | 方法名字符串 |
| `call_module` | nn.Module 的调用 | 查找该 module 的 key 字符串 |
| `output` | 整个被 trace 的 Module/函数的输出 | 字符串 |

> **📝 说明：**
>
> **target 的歧义设计：**对于 `call_module` 和 `get_attr`，target 是字符串而非对象。要通过它找到实例需：对 `call_module` 用 `dict(gm.named_modules())[node.target]`；对 `get_attr` 用 `getattr(gm, node.target)`。Node 的 `meta` 属性含对象信息与代码调用栈，对 Debug 极有帮助。

`fx.Graph` 主要支持图上的增删查改：

- **nodes**：获取图中所有 Node 的列表
- **create_node** / `call_module` / `call_method` 等：添加新 Node（后者是语法糖）
- **erase_node**：删除 Node
- **inserting_after** / `inserting_before`：设置新 Node 插入点
- **eliminate_dead_code**：删除未被使用的 Node
- **lint**：图结构检查
- **on_generate_code**：代码生成时插入自定义操作

`fx.Node` 支持 `append` / `prepend`（前后插入）、`replace_all_uses_with`（替换所有依赖）等改写操作。

### 15.5 图改写：两种典型模式

#### 模式一：图遍历改写

最典型的图改写模式——用 `fx.Graph.nodes` 获取节点并修改。下面把 `add` 操作替换为 `bitwise_and`：

```python
import torch, operator
from torch.fx import symbolic_trace

class M(torch.nn.Module):
    def forward(self, x, y):
        return x + y, torch.add(x, y), x.add(y)

traced = symbolic_trace(M())
patterns = set([operator.add, torch.add, "add"])

for n in traced.graph.nodes:
    if any(n.target == pattern for pattern in patterns):
        with traced.graph.inserting_after(n):
            new_node = traced.graph.call_function(torch.bitwise_and, n.args, n.kwargs)
            n.replace_all_uses_with(new_node)
        traced.graph.erase_node(n)   # 清理过时 Node

traced.recompile()                   # 重新代码生成，得到新 GraphModule
```

处理复杂输入时，`map_aggregate` 函数提供对参数的通用变换：对由 node 组成的 tuple/list/dict，提供一个 node 处理函数 `fn`，返回同结构、每个 node 都被 `fn` 变换过的新输入。

#### 模式二：Interpreter 模式

Interpreter 模式提供"边执行边修改图"的能力——遍历图中节点同时挨个执行。`Node.target` 可获得节点实例（如 nn.Module）并执行。典型的 **ShapeProp** 通过执行 Node 记录其输出 tensor 的 shape 和 dtype：

```python
for node in self.graph.nodes:
    if node.op == 'placeholder':
        result = next(args_iter)
    elif node.op == 'get_attr':
        result = fetch_attr(node.target)
    elif node.op == 'call_function':
        result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
    elif node.op == 'call_method':
        self_obj, *args = load_arg(node.args)
        result = getattr(self_obj, node.target)(*args, **load_arg(node.kwargs))
    elif node.op == 'call_module':
        result = self.modules[node.target](*load_arg(node.args), **load_arg(node.kwargs))
    if isinstance(result, torch.Tensor):
        node.shape = result.shape
        node.dtype = result.dtype
```

`fx.Interpreter` 是该模式的语法糖，接受 `fx.GraphModule`，用 `run` 方法执行，`run_node` 调度到各类 Node 的执行方法（`placeholder()` / `get_attr()` / `call_function()` / `call_method()` / `call_module()`）。重载这些方法即可自定义执行逻辑。

### 15.6 fx.GraphModule

`fx.GraphModule` 继承自 `nn.Module`，主要行为与 `nn.Module` 一致，特别之处在于：

- 其 `forward` 是从内含的 `fx.Graph` 生成的
- `graph` 属性：获取内部计算图
- `code` 属性（str 类型）：从 graph 生成的 Python 文本代码，`forward` 方法是该文本代码经编译得到的

由 `symbolic_trace` 生成的 `GraphModule` 通常当做普通 `nn.Module` 使用即可，体现了 fx 良好的易用性。

### 15.7 自定义 Tracer

通常使用涉及不到自定义 Tracer，但需要时可继承并覆盖。可自定义的方法包括：

| 方法 | 作用 |
| --- | --- |
| `create_node` | Tracer 往 graph 插入节点时调用，返回 node |
| `create_proxy` | 所有操作调用的输入/输出转 Proxy 时调用 |
| `create_args_for_root` | 创建被 trace 的 Module/函数的输入 |
| `create_arg` | 创建内部函数的输入 |
| `call_module` | 遇到 nn.Module 时触发对应 node 创建 |
| `getattr` | 从 nn.Module 获取属性时触发 node 创建 |

> **⚠️ 注意：**
>
> **注意：**以上方法与 Tracer 行为耦合紧密，自定义时需小心处理并结合 Tracer 源码实现。

### 15.8 与 torch.compile 栈的衔接

torch.fx 在 `torch.compile` 全链路中是承上启下的枢纽：

```text
TorchDynamo (字节码符号化追踪)
    │
    │  输出：fx.GraphModule (前向图, 含 aten/prim 算子)
    ▼
AOTAutograd (autograd dispatch 追踪)
    │
    │  输入/输出仍是 fx.GraphModule；在此做算子分解、联合图追踪、最小割分区
    │  最终产出 fwd_gm + bwd_gm 两个 GraphModule
    ▼
TorchInductor (代码生成)
    │
    │  GraphLowering 解释器遍历 fx.Graph，把 fx.Node 逐个 Lowering 为 Inductor IR
    │  (见第十章 Lowering: FX → IR)
    ▼
Triton / C++ 内核
```

> **✨ 技巧：**
>
> **为何 Dynamo 仍用 fx 而非新造 IR？**fx 的 IR 极简（仅 6 种 Node）、纯 Python、可直接生成可执行代码、与 `nn.Module` 无缝兼容。这让 Dynamo 产出的图可被任何 fx Pass 处理，也便于调试（直接打印就是 Python 代码）。Dynamo 解决了"如何从 Python 字节码捕获图"，而 fx 提供了"图该长什么样"的成熟答案。

> **📝 说明：**
>
> **本章与第十章的关系：**第十章讲 Inductor 如何把 fx 图 Lowering 为 IR；本章讲 fx 图本身如何被构造与改写。两者衔接点在 `GraphLowering` 解释器——它本质就是一个遍历 fx.Node 并发射 IR 的 fx Interpreter（见 15.5 模式二）。

参考：`torch/fx/` `torch/fx/_symbolic_trace.py` `torch/fx/graph.py` `torch/fx/node.py` `torch/fx/proxy.py` `torch/fx/interpreter.py`

## Related

- [16 TorchDynamo 深入](./16-torchdynamo-deep.md)
- [PyTorch 索引](../index.md)
