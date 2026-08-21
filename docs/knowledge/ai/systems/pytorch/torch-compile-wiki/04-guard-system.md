---
title: Guard 系统
type: concept
status: seed
tags: [PyTorch, torch.compile, Guard]
created: 2026-08-17
updated: 2026-08-17
source:
  - d:\project\pytorch-2.8.0\wiki\04_guards.html
---

# Guard 系统

> 编译结果复用的核心保障机制

Guard 是 Dynamo 实现高效复用的核心机制。每次编译时，Dynamo 记录一组假设条件（Guard），后续调用时只需快速检查这些条件是否满足即可决定是否复用缓存。Guard 系统定义于 `torch/_dynamo/guards.py`，并在 C 层（`torch/_C/_dynamo/guards`）提供高性能检查实现。

> **💡 提示：** 设计哲学：Guard 把"是否需要重新编译"这个昂贵的决策，转化为运行时一组廉价的布尔断言。所有断言被组织成树形 `GuardManager`，在 C 层以 `check_nopybind` 一次遍历完成，避免回退到 Python 解释器，单次检查耗时在微秒量级。

## 🛡️ 4.1 Guard 的本质

Guard 是一个布尔表达式，形如"这个张量的 dtype 是 float32"或"这个整数的值是 42"。在追踪期，Dynamo 把每个符号变量所依赖的运行时假设记录为 Guard；在缓存命中检查时，逐条验证这些假设。

```python
# 张量 Guard: 检查张量属性
GUARD: tensor_self.dtype == torch.float32
GUARD: tensor_self.device == cuda:0
GUARD: tensor_self.requires_grad == False
GUARD: 'size' not in tensor_self._named_keys

# 形状 Guard: 检查张量形状（可能含符号维度）
GUARD: tensor_self.size()[0] == 128          # 静态形状
GUARD: tensor_self.size()[1] == s0            # 动态形状（符号变量）

# Python 值 Guard
GUARD: ___class_value == int
GUARD: ___check_type_id(5678, 12345)         # 类型 ID 检查

# 全局变量 Guard
GUARD: ___check_global_state()                # 全局状态检查
GUARD: ___check_obj_attribute(fn, 'mode', 1) # 对象属性检查
```

每个 Guard 由三部分组成：**Source**（被检查对象的来源，如 `LocalSource`、`GlobalSource`、`TensorPropertySource`）、**create_fn**（Guard 类型对应的构建函数名，如 `TENSOR_MATCH`）以及**检查语义**。这三者由 `torch._guards.Guard` 数据类承载。

## 🔧 4.2 GuardBuilder：追踪期构建 Guard

Guard 的构建发生在 Dynamo 符号化追踪期间。核心类 `GuardBuilder`（定义于 `guards.py#L637`）继承自 `GuardBuilderBase`，它是一个"分发器"：每个 Guard 的 `create_fn` 名字对应 `GuardBuilder` 上的一个同名方法。

```python
class GuardBuilder(GuardBuilderBase):
    def __init__(
        self,
        f_code: types.CodeType,
        id_ref: Callable[[Any, str], str],
        source_ref: Callable[[Source], str],
        lookup_weakrefs: Callable[[object], ReferenceType[object]],
        local_scope: dict[str, object],
        global_scope: dict[str, object],
        guard_manager: GuardManagerWrapper,
        check_fn_manager: CheckFunctionManager,
        serialization_mode: Optional[str] = None,
    ):
        self.f_code = f_code
        self.id_ref = id_ref
        self.source_ref = source_ref
        self.lookup_weakrefs = lookup_weakrefs
        self.scope = {"L": local_scope, "G": global_scope}
        self.scope["__builtins__"] = builtins.__dict__.copy()
        ...
```

### 构建流程

当追踪结束、`OutputGraph` 准备生成 `GuardedCode` 时，`CheckFunctionManager` 会：

1. 从 `output_graph.guards` 取出所有 Guard，按 `Guard.sort_key` 排序，保证构建顺序稳定。
2. 实例化 `GuardBuilder`，传入帧的局部/全局作用域、`f_code` 等。
3. 对每个 Guard，调用 `getattr(guard_builder, guard.create_fn.__name__)(guard)` 分发到对应方法。
4. 每个方法生成 C 代码片段，并通过 `self.get_guard_manager(guard)` 在树形 `GuardManager` 上挂载对应类型的检查节点（如 `add_type_match_guard`、`add_id_match_guard`、`add_tensor_guard` 等）。

> **✨ 技巧：** 分发的妙处：`GuardBuilder` 上每个 `XXX_MATCH` 方法既是 Guard 类型的"名字"，也是构建逻辑的"实现"。Guard 数据类只保存 `create_fn` 的引用，构建延迟到 `CheckFunctionManager` 阶段执行，使追踪期与检查函数生成期解耦。

例如 `TYPE_MATCH` 方法（`guards.py#L1470`）生成 `___check_type_id(arg, type_id)`，并调用 `add_type_match_guard` 在 manager 上注册；`ID_MATCH` 方法（`guards.py#L1548`）生成 `___check_obj_id(arg, obj_id)`，并把 `nn.Module` 类型的 ID_MATCH 对象记录到 `id_matched_objs`（供缓存大小计数使用，见第五章）。

## 📋 4.3 常见 Guard 类型与 C 代码生成模式

下表列出 `GuardBuilder` 中常见的 Guard 类型、触发场景、生成的检查代码模式及检查语义。其中 `___` 前缀的标识符是 Dynamo 在 C 扩展中预注册的内建检查函数。

| Guard 类型 | 触发场景 | C 代码生成模式 | 检查语义 |
|---|---|---|---|
| `TENSOR_MATCH` | 输入张量 | C++ `TensorGuard`（多字段批量检查） | dtype / device / requires_grad / ndimension / sizes / strides |
| `TYPE_MATCH` | 任意对象的类型 | `___check_type_id(arg, type_id)` | `id(type(x)) == y` |
| `ID_MATCH` | nn.Module / 对象身份 | `___check_obj_id(arg, obj_id)` | `id(x) == y` |
| `EQUALS_MATCH` | int / float / str 值 | `arg == value` | 值相等（对应旧文档中的 INT_MATCH/FLOAT_MATCH/STRING_MATCH） |
| `CONSTANT_MATCH` | Python 常量 | `arg == value` | 常量值相等 |
| `BOOL_MATCH` | 布尔值 | `arg == True` / `arg == False` | 布尔相等（分别用 true/false match guard） |
| `NONE_MATCH` | None 检查 | `arg is None` | 对象为 None |
| `NOT_NONE_MATCH` | 非 None 检查 | `arg is not None` | 对象非 None |
| `DICT_VERSION` | dict 全局变量 | `___dict_version(arg) == version` | dict 版本号未变 |
| `DICT_CONTAINS` | dict 键存在性 | `___dict_contains(key, arg)` | 键存在/不存在 |
| `NAME_MATCH` | 函数/类名 | `arg.__name__ == 'name'` | 名称匹配 |
| `GRAD_MODE` | 全局梯度模式 | `torch.is_grad_enabled() == val` | autograd grad 模式 |
| `TORCH_FUNCTION_STATE` | torch_function 栈 | 栈状态检查 | `TorchFunctionMode` 栈一致 |
| `DISPATCH_KEY_SET_MATCH` | dispatch key 集 | `arg.raw_repr() == val.raw_repr()` | dispatch key 集一致 |
| `FUNCTION_MATCH` / `BUILTIN_MATCH` | 函数对象 | 基于 ID_MATCH | 函数身份一致 |
| `CLOSURE_MATCH` | 闭包 cell | cell 内容检查 | 闭包变量一致 |
| `FUNCTORCH_STACK_MATCH` | functorch 栈 | 栈深度与状态 | functorch transform 栈一致 |
| `TENSOR_SUBCLASS_METADATA_MATCH` | Tensor 子类 | 子类元数据检查 | 子类身份与属性 |
| `NN_MODULE` | nn.Module | 复合检查 | 模块属性/训练状态 |
| `FSDP_TRAINING_STATE` | FSDP 状态 | 特定检查 | FSDP 训练阶段 |
| `RANGE_ITERATOR_MATCH` | range 迭代器 | 迭代器剩余检查 | range 迭代器状态 |

> **📝 备注：** TENSOR_MATCH 的特殊性：对张量而言，若 `match_on_id_for_tensor(guard)` 为真（即张量已是 FX 图的输入），`TENSOR_MATCH` 会退化为 `ID_MATCH`，只校验对象身份；否则才在 C++ `TensorGuard` 中逐字段比对 dtype/device/shape/strides 等。导出模式（`export=True`）下则不使用 C++ TensorGuard，而是生成显式的 Python 表达式（`tensor.dtype == ...` 等）以便序列化。

## 📌 4.4 Guard 如何挂载到 Code Object

Guard 不是孤立存在的，它必须与"哪个函数的哪次编译"绑定。Dynamo 通过 `code object` 的 `co_extra` 暂存区挂载缓存与 Guard。

构建完成后，`CheckFunctionManager` 产出一个 `GuardManagerWrapper`，它持有根 manager `RootGuardManager`。随后 `create_backgraph_blacklist` / `CacheEntry` 构造逻辑把 `GuardManagerWrapper` 与编译后的自定义字节码一起封装为 `GuardedCode`，再包成 `CacheEntry`，写入 `f_code.co_extra`。

```python
# GuardManagerWrapper 持有的关键字段（guards.py#L190）
class GuardManagerWrapper:
    def __init__(self, root=None):
        self.root = RootGuardManager()   # C 层根 manager
        self.diff_guard_root = None        # 仅含"差异 Guard"的克隆树
        self.closure_vars = None
        self.args = None
        self.code_parts = []             # 可读的 Guard 代码片段
        self.verbose_code_parts = None    # 调试用详细片段
        self.global_scope = None
        self.guard_fail_fn = None         # Guard 失败回调
        self.cache_entry = None           # 反向指针 → CacheEntry
        self.id_matched_objs = {}        # ID_MATCH 对象 → 缓存计数
        self.no_tensor_aliasing_sources = []
```

`GuardManagerWrapper` 被显式存在缓存项中，使得 C 层帧评估能直接通过 `cache_entry.guard_manager.root.check_nopybind(...)` 调用检查，无需进入 Python。同时 `diff_guard_root` 是一棵"差异 Guard"克隆树（只包含与张量相关的 Guard），用于在重编译时快速定位是哪类输入变化导致 Guard 失败（`collect_diff_guard_sources` / `populate_diff_guard_manager`）。

## 🎛️ 4.5 guard_filter_fn：控制保存哪些 Guard

`CheckFunctionManager` 接受可选的 `guard_filter_fn`（`guards.py#L2770`），允许调用方在 Guard 被写入检查函数前进行筛选。这是 export、torch.export 等高级场景过滤"运行时无意义 Guard"的钩子。

```python
if guard_filter_fn:

    def make_guard_filter_entry(guard):
        MISSING = object()
        name = strip_local_scope(guard.name)
        if name == "":
            has_value = False
            value = MISSING
        else:
            has_value = True
            value = builder.get(guard.name)
        is_global = get_global_source_name(guard.originating_source) is not None
        guard_fn = guard.create_fn
        if isinstance(guard_fn, functools.partial):
            guard_fn = guard_fn.func
        return GuardFilterEntry(
            name=name,
            has_value=has_value,
            value=value,
            guard_type=guard_fn.__name__,
            derived_guard_types=tuple(guard.guard_types) if guard.guard_types else (),
            is_global=is_global,
            orig_guard=guard,
        )

    filter_results = guard_filter_fn(
        [make_guard_filter_entry(guard) for guard in sorted_guards]
    )
    assert len(filter_results) == len(sorted_guards)
    sorted_guards = [
        guard for i, guard in enumerate(sorted_guards) if filter_results[i]
    ]
    # 用筛选后的 guard 重新构建一次 manager
    builder, guard_manager = self.build_guards(
        sorted_guards, existing_diff_guard_sources, f_code, output_graph,
        self.guards_serialization_mode,
    )
```

`guard_filter_fn` 接收一个 `GuardFilterEntry` 列表（每个条目携带 name、value、guard_type、is_global 等元信息），返回同长度的布尔列表：`True` 表示保留该 Guard，`False` 表示丢弃。筛选完成后会**重新调用** `build_guards`，用过滤后的集合重建 manager，保证 manager 树与最终保留的 Guard 严格一致。

- **导出场景：** `torch.export` 通过 `guard_filter_fn` 移除仅与运行时 dispatch 相关的 Guard（如 dispatch key），生成可序列化的图。
- **调试场景：** 可用于只保留某类 Guard 以观察其对重编译的影响。

## ⚡ 4.6 Guard 检查的 C 层执行

Dynamo 将所有 Guard 编译为一棵树形 `GuardManager`，在帧评估时由 C 层高效执行。`CheckFunctionManager`（`guards.py#L2763`）负责把 `output_graph.guards` 转化为可执行的检查树。

```python
class CheckFunctionManager:
    """将所有 Guard 编译为高效的 C 检查函数"""

    def __init__(self, f_code, output_graph=None, cache_entry=None,
                 guard_fail_fn=None, guard_filter_fn=None, ...):
        guards = output_graph.guards if output_graph else None
        self._weakrefs = {}

        existing_diff_guard_sources = update_diff_guard_managers_for_existing_cache_entries(cache_entry)
        self.output_graph = output_graph

        # 1. 排序保证构建稳定
        sorted_guards = sorted(guards or (), key=Guard.sort_key)

        # 2. 构建 GuardManager 树
        builder, guard_manager = self.build_guards(
            sorted_guards, existing_diff_guard_sources, f_code, output_graph,
            None if guard_filter_fn else self.guards_serialization_mode,
        )

        # 3. 可选: 用 guard_filter_fn 筛选并重建 (见 4.5)
        ...

        self.guard_manager = guard_manager
        self.compile_check_fn(builder, sorted_guards, guard_fail_fn)
```

执行时，C 层不回退到 Python 解释器，而是直接遍历 `RootGuardManager` 子树。下图展示了一次 Guard 检查在 C 层的执行流：

```text
Python 帧评估 (PEP 523 钩子)
    │
    ▼
C 层: eval_frame_callback
    │
    ├─ 获取 frame->f_code
    ├─ 从 code->co_extra 取出 cache_entry 链表
    │
    ▼
遍历 CacheEntry 链表
    │
    每个 CacheEntry:
    │
    ├─ 调用 guard_manager.root.check_nopybind(frame_locals, ...)
    │      │
    │      ├─ RootGuardManager 树形遍历
    │      │   ├─ TypeMatchGuard      → ___check_type_id(...)
    │      │   ├─ IdMatchGuard        → ___check_obj_id(...)
    │      │   ├─ TensorGuard         → dtype/device/sizes/strides 批量比较
    │      │   ├─ DictVersionGuard    → ___dict_version(...)
    │      │   ├─ GradModeGuard       → torch.is_grad_enabled()
    │      │   └─ ... (所有挂载的 guard 节点)
    │      │
    │      └─ 任一节点失败 → 立即短路返回 False
    │
    ├─ 返回 True  → 命中! 使用此 entry 的 out_code (自定义字节码)
    │
    └─ 返回 False → 跳到下一个 CacheEntry
    │
    ▼
全部 entry 不命中 → 调用 Python callback 触发重编译
```

> **✨ 技巧：** 短路求值：GuardManager 树采用短路语义——任一节点失败立即返回，不再检查后续节点。配合 C 层 `check_nopybind`（无 pybind11 转换开销），单次完整 Guard 检查通常在亚微秒到微秒级，相比秒级的编译开销可忽略。

## 🔁 4.7 Guard 失败与重编译

当运行时输入与编译期假设不一致（如张量形状变化、dtype 变化、模块实例变化），对应的 Guard 失败，Dynamo 触发重编译。

```text
首次调用: 无缓存 → 编译 (记录 Guard) → 执行
                                     │
第二次调用: 检查 Guard                │
  ├─ 全部通过 → 复用缓存 ✓ (快速路径) │
  └─ 有 Guard 失败 → 重编译          │
       ├─ 重编译次数 < recompile_limit (默认 8)
       │    → 生成新的编译结果 + 新 Guard → 存入缓存
       └─ 重编译次数 >= limit
            → 放弃编译，回退到 eager 执行
```

重编译时，`diff_guard_root`（差异 Guard 树）会被用来快速定位是哪些"张量相关"的 Guard 发生了变化，便于日志输出与调试。失败的 Guard 通过 `guard_fail_fn` 回调上报，配合 `GuardFail` 数据结构记录失败原因与位置。

> **⚠️ 注意：** 注意：每个 code object 最多编译 `torch._dynamo.config.recompile_limit`（默认 8）次。超过后回退到 eager 模式。实际上还有第二个上限 `accumulated_recompile_limit`（累计上限，默认 64），用于在 ID_MATCH 模块多实例场景下兼顾复用与尽早回退（详见第五章 5.4 节）。

## 🐛 4.8 调试 Guard

Guard 系统提供专门的日志通道。`guards.py` 顶部注册了多个 artifact logger：

- `guards`：Guard 创建与检查的基本日志
- `recompiles`：重编译事件
- `recompiles_verbose`：重编译详细原因
- `verbose_guards`：最详细的 Guard 状态

> **✨ 技巧：** 调试技巧：使用环境变量 `TORCH_LOGS=guards` 可查看每次 Guard 检查的通过/失败情况及失败原因；`TORCH_LOGS=recompiles` 专注于重编译事件；`TORCH_LOGS="guards,recompiles_verbose"` 可同时获得 Guard 失败与重编译的完整诊断信息。当出现"意外的重编译"时，这是定位根因的首选工具。

此外，`verbose_code_parts` 字段保留了人类可读的 Guard 代码片段，可在日志中直接看到形如 `tensor_self.size()[0] == 128` 的断言，便于理解 Dynamo 对当前帧所做的假设。

## Related

- [03 TorchDynamo 前端](./03-torchdynamo-frontend.md)
- [05 缓存机制](./05-cache-mechanism.md)
- [PyTorch 索引](../index.md)
