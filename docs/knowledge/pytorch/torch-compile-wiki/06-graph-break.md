---
title: 图断裂 (Graph Break)
type: concept
status: seed
tags: [PyTorch, torch.compile, 图断裂, Dynamo]
created: 2026-08-17
updated: 2026-08-17
source:
  - d:\project\pytorch-2.8.0\wiki\06_graph_break.html
---

# 图断裂 (Graph Break)

*不可追踪代码的处理策略*

当 Dynamo 遇到无法符号化追踪的代码时（如调用不支持的内建函数、数据依赖的控制流等），会触发**图断裂**，将计算图断开为多个可独立编译的子图。图断裂是 Dynamo 容错性的核心：它让"部分可加速"的代码也能从 torch.compile 受益，而不要求整个函数完全可追踪。

> **💡 提示：** **核心思想：**Dynamo 不要求"全有或全无"。遇到无法追踪的代码段，它在该处断开，前一段单独编译为子图，不可追踪段回退 eager 执行，后一段再编译为下一个子图。运行时按顺序调用各段，把不可追踪段"缝合"在两个编译子图之间。

## ⚡ 6.1 图断裂的触发条件

| 触发原因 | 示例 | 处理方式 |
| --- | --- | --- |
| 不支持的内建函数 | `print()`, `input()` | 断开图，在 eager 模式执行该调用 |
| 数据依赖的控制流 | `if tensor.item() > 0:` | 断开图，运行时求值后重新追踪 |
| 不支持的 Python 特性 | 某些生成器、异步操作 | 断开图 |
| 显式调用 | `torch._dynamo.graph_break()` | 手动断开 |
| 递归深度限制 | 过深的嵌套追踪 | 断开以保证安全 |

## 🔬 6.2 触发原因详析

### trace_rules：可追踪性规则表

Dynamo 维护一张"可追踪性规则表"（`torch/_dynamo/trace_rules.py`），对每个 Python 内建/库函数标注是否可在追踪期符号化。规则分几类：

- **支持（traceable）：**该调用可在追踪期被符号化，加入 FX 图。
- **不支持（unsupported）：**调用无法符号化，触发图断裂。
- **恒等/常量：**结果可在编译期确定，直接折叠为常量。

规则表覆盖 `builtins`、`math`、`itertools`、`functools`、`os`、`sys` 等大量模块。当用户代码调用了未在表中标记为 traceable 的函数，Dynamo 默认会图断裂（除非通过 `torch._dynamo.allow_in_graph` 显式放行）。

### 不支持的内建函数

具有副作用的内建（如 `print`、`input`、`open`）天然无法在追踪期符号化——它们的语义依赖运行时外部状态。Dynamo 在遇到此类调用时图断裂，让该调用在 eager 模式真实执行。

### 数据依赖的控制流

Dynamo 能展开**静态**控制流（条件在编译期可求值），但当代件分支依赖**张量的运行时值**（如 `tensor.item() > 0`）时，编译期无法决定走哪条分支，必须图断裂。断裂后，前一段编译为子图，运行时求值条件，再根据分支结果追踪后一段。

### 其他触发场景

- **不支持的 Python 特性：**某些生成器、`async/await`、`yield`、特定形式的 `eval/exec`。
- **递归深度限制：**为防止栈溢出，Dynamo 对嵌套追踪深度设限，超限则图断裂。
- **显式图断裂：**用户调用 `torch._dynamo.graph_break()` 强制断开，常用于调试或隔离不可追踪段。

`graph_break_hints.py`（[graph_break_hints.py](file:///d:/project/pytorch-2.8.0/torch/_dynamo/graph_break_hints.py)）为每种图断裂原因定义了用户提示文案，帮助开发者理解断裂性质：

**torch/_dynamo/graph_break_hints.py**

```python
USER_ERROR = [
    "Dynamo has detected that tracing the code will result in an error when running in eager. "
    "Please double check that your code doesn't contain a similar error...",
]
DYNAMO_BUG = [
    "This is likely to be a Dynamo bug. Please report an issue to PyTorch.",
]
FUNDAMENTAL = [
    "This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through "
    "your code. Consider finding a workaround.",
]
SUPPORTABLE = [
    "It may be possible to write Dynamo tracing rules for this code. Please report an issue...",
]
INFERENCE_MODE = [
    "Avoid using `tensor.is_inference()` and `torch.is_inference_mode_enabled()` ... "
    "Consider using `torch.no_grad` instead ...",
]
```

## 💻 6.3 触发图断裂的代码示例

下面的函数包含两处图断裂：一处是数据依赖控制流，一处是 `print` 副作用。

**触发图断裂的函数**

```python
import torch

@torch.compile
def fn(x):
    y = x + 1          # 可追踪: 加入子图 1
    z = y * 2          # 可追踪: 加入子图 1

    if z.item() > 0:    # 数据依赖控制流 → 图断裂!
        w = z.relu()
    else:
        w = z.neg()

    print(w)           # print 副作用 → 再次图断裂
    return w + 1      # 可追踪: 子图 3
```

Dynamo 把 `fn` 拆为三段，运行时按顺序执行：

**Dynamo 的拆分结果**

```text
# 子图 1: 可追踪段 (y = x+1, z = y*2)
GraphModule_1(x) -> z          # → 编译为内核 1

# eager 段: 数据依赖控制流
z_val = run_kernel_1(x)         # 先执行子图 1 得到 z 的真实值
if z_val.item() > 0:           # 运行时求值条件
    w = z_val.relu()            # eager 执行分支
else:
    w = z_val.neg()

# eager 段: print 副作用
print(w)                        # eager 执行 print

# 子图 3: 可追踪段 (w + 1)
GraphModule_3(w) -> out         # → 编译为内核 3
out = run_kernel_3(w)
```

## ✂️ 6.4 图断裂的处理流程

```text
  代码段 A (可追踪)     代码段 B (不可追踪)     代码段 C (可追踪)
  ┌─────────────┐       ┌──────────────┐       ┌─────────────┐
  │ x = a + b   │       │ print(x)     │       │ y = x * 2   │
  │ x = x.relu()│  ───► │              │  ───► │ return y    │
  └─────────────┘       └──────────────┘       └─────────────┘
     子图 1                  eager 执行              子图 2

  编译结果:
  ├─ GraphModule_1 (a, b → x)        → 编译为内核 1
  ├─ x = eager_execute(print, x)     → 原始 Python 执行
  └─ GraphModule_2 (x → y)           → 编译为内核 2

  运行时:
  调用内核 1 → 执行 print → 调用内核 2
```

> **⚠️ 注意：** **fullgraph=True 时：**图断裂会直接报错而非静默处理。这确保整图被编译为单个内核，但要求代码完全可追踪。生产场景中，建议先用 `torch._dynamo.explain` 排查图断裂，再决定是否启用 `fullgraph`。

图断裂追踪逻辑位于 [graph_break_hints.py](file:///d:/project/pytorch-2.8.0/torch/_dynamo/graph_break_hints.py)（用户提示）和 [graph_region_tracker.py](file:///d:/project/pytorch-2.8.0/torch/_dynamo/graph_region_tracker.py)（子图区域追踪）。

## 🗺️ 6.5 子图管理与 graph_region_tracker

`GraphRegionTracker`（[graph_region_tracker.py#L189](file:///d:/project/pytorch-2.8.0/torch/_dynamo/graph_region_tracker.py#L189)）追踪每个被加入输出图的节点，并基于**源码位置、指令指针、输入形状、全局状态**计算哈希，把哈希相同的节点归为"相同节点组"。它的核心目的是识别图中**重复的子区域**，为去重等图变换优化提供基础。

**torch/_dynamo/graph_region_tracker.py#L189**

```python
class GraphRegionTracker:
    """
    追踪加入输出图的每个节点，基于源码位置、指令指针、输入形状和
    全局状态生成 key。相同 key 的节点被归入同一 identical_nodes 列表。

    hash_to_duplicates: dict[str, IdenticalNodes]  key → 相同节点列表
    node_to_duplicates: dict[Node, IdenticalNodes] 节点 → 所属相同节点列表
    input_pickler: InputPickler  用于生成节点哈希
    """

    def __init__(self):
        self.hash_to_duplicates: dict[str, IdenticalNodes] = defaultdict(list)
        self.node_to_duplicates: dict[Node, IdenticalNodes] = {}
        self.node_to_mutated_arg_positions: dict[Node, OrderedSet[int]] = {}
        self.input_pickler = InputPickler()

    def _hash_node(self, filename, lineno, instruction_pointer, node):
        key = (
            get_global_state_key(),   # grad/inference_mode/dtype/...
            filename,
            lineno,
            instruction_pointer,
            _normalize_args(node),     # 规范化的参数 (含 shape)
        )
        return sha256_hash(self.input_pickler.dumps(key))

    def track_node(self, tx, node):
        # 主入口: 哈希节点, 把相同哈希的节点分组
        duplicates = self.hash_to_duplicates[
            self._hash_node(tx.f_code.co_filename, tx.lineno, tx.instruction_pointer, node)
        ]
        duplicates.append(node)
        self.node_to_duplicates[node] = duplicates
```

`get_identical_regions` 在图断裂产生的多个子图之间识别"最大相同区域组"，使用反向 BFS（`BackwardBfsArgIter`）从拓扑最晚的节点开始扩展区域，确保先形成最大的不重叠区域。这为子图去重（Subgraph HOP）等优化打下基础：若两个子图存在大段相同区域，可合并为一次编译。

> **📝 备注：** **与图断裂的关系：**`GraphRegionTracker` 不是"产生图断裂"的组件，而是"管理图断裂后多个子图"的组件。它让 Dynamo 在追踪期就记录节点的来源指纹，事后能识别哪些子图段是重复的，从而减少重复编译、支持更激进的图优化。

## 🧵 6.6 子图在运行时的拼接

Dynamo 通过**自定义字节码**把多个编译子图与 eager 段缝合在一起。具体地，追踪结束后 `OutputGraph` 产出 `output_instructions`——一组 Dynamo 重写后的字节码指令，它们：

- 调用子图 1 的编译入口（Inductor 生成的内核），把输出张量压入栈；
- 执行 eager 段的原始字节码（不可追踪部分）；
- 把 eager 段输出作为输入，调用子图 2 的编译入口；
- 以此类推，直到函数返回。

这套自定义字节码与对应的 `CacheEntry` 一同存入 code object 缓存（见第五章）。Guard 失败时重编译会重新生成字节码。需要注意的是：**每个子图有自己的 Guard 集合**，但它们共享同一个 code object 的缓存链表项——整段函数的 Guard 通过即复用整段字节码。

```text
Dynamo 生成的自定义字节码 (单条 CacheEntry):
┌─────────────────────────────────────────────────────┐
│ 1. LOAD cache_entry[0].compiled_fn_1 (子图 1 入口)   │
│ 2. CALL → 栈顶得到 z                                 │
│ 3. (eager) LOAD z.item, COMPARE > 0, POP_JUMP        │  ← 不可追踪段原样保留
│ 4. (eager) 根据 z.item 结果走 relu / neg 分支        │
│ 5. LOAD cache_entry[0].compiled_fn_3 (子图 3 入口)   │
│ 6. CALL → 栈顶得到 out                               │
│ 7. RETURN out                                        │
└─────────────────────────────────────────────────────┘
        ↑ compiled_fn_1 / compiled_fn_3 由 Inductor 生成
```

## 🛠️ 6.7 常见图断裂原因与解决方案

| 图断裂原因 | 典型表现 | 解决方案 |
| --- | --- | --- |
| 调用 `print`/`input` 等内建 | 日志中出现 `unsupported` 内建 | 移出编译区域，或用 `torch._dynamo.disable` 包裹 |
| 数据依赖控制流 | `if tensor.item() > 0` | 改用 `torch.where`/`torch.cond` 等可追踪替代 |
| 调用了未注册的第三方函数 | 该函数被判定 unsupported | 用 `torch._dynamo.allow_in_graph` 显式放行（若安全） |
| 使用 `inference_mode` | 提示 INFERENCE_MODE 图断裂 | 改用 `torch.no_grad`（提示文案明确建议） |
| 递归过深 | 提示递归深度限制 | 拆分函数，或用 `graph_break` 显式断点 |
| 修改全局 dict | `DICT_VERSION` Guard 频繁失败 | 避免在编译函数内修改全局状态 |
| 动态 shape 频繁变化 | 反复重编译 | 启用 `dynamic=True` 或标记动态维度 |
| FUNDAMENTAL 图断裂 | 提示"fundamental" | 该特性 Dynamo 不会支持，需重构代码绕开 |

## 🐛 6.8 调试图断裂

> **✨ 技巧：** **调试利器：**`torch._dynamo.explain(fn)(*args)` 会运行一次追踪并返回结构化报告，列出所有图断裂位置、原因（USER_ERROR / FUNDAMENTAL / SUPPORTABLE 等分类）、产生的子图数量、Guard 数量等编译统计。这是排查"为什么我的模型没有被完整编译"的首选工具。配合 `TORCH_LOGS=graph_breaks` 可获得每个断裂点的详细堆栈。

其他常用调试手段：

- `TORCH_LOGS=graph_breaks`：打印每个图断裂点及其原因分类。
- `torch._dynamo.explain`：返回 `Explanation` 对象，含 `graph_breaks` 列表与 `ops_per_graph` 统计。
- `fullgraph=True` 配合 `torch._dynamo.explain`：先解释清楚所有断裂点，再尝试整图编译。

## Related

- [05 缓存机制](./05-cache-mechanism.md) — 自定义字节码与 CacheEntry 一同存入 code object 缓存
- [07 AOTAutograd 中间层](./07-aotautograd.md) — AOTAutograd 对图断裂产出的子图做联合图追踪与前向/反向分区
- [PyTorch 索引](../index.md)
