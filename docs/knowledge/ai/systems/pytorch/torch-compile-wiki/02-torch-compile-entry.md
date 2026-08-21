---
title: torch.compile 入口
type: concept
status: seed
tags: [PyTorch, torch.compile, 入口]
created: 2026-08-17
updated: 2026-08-17
source:
  - d:\project\pytorch-2.8.0\wiki\02_entry.html
---

# torch.compile 入口

> 从用户 API 到 Dynamo 上下文

本章追溯从用户调用 `torch.compile()` 到 Dynamo 帧评估钩子被激活的完整路径。理解这一链路是定位"编译未触发"、"行为不符合预期"等问题的关键。

### 2.1 公共 API 定义

用户通过 `torch.compile()` 进入编译流程。该函数定义于 `torch/__init__.py#L2466`，是整个编译系统的统一入口。下面是完整的函数签名与核心实现逻辑：

```python
def compile(
    model: _Optional[_Callable[_InputT, _RetT]] = None,
    *,
    fullgraph: builtins.bool = False,        # 是否要求整图编译（不允许图断裂）
    dynamic: _Optional[builtins.bool] = None,  # 动态形状支持
    backend: _Union[str, _Callable] = "inductor",  # 编译后端
    mode: _Union[str, None] = None,      # 编译模式
    options: _Optional[dict] = None,   # 后端选项
    disable: builtins.bool = False,          # 禁用编译
) -> _Union[...]:
    """
    Optimizes given model/function using TorchDynamo and specified backend.
    """
    import sysconfig
    _C._log_api_usage_once("torch.compile")

    # 1. 版本兼容性检查
    if sys.version_info >= (3, 14):
        raise RuntimeError("torch.compile is not supported on Python 3.14+")

    # 2. 装饰器模式: @torch.compile (无括号或带括号但无模型)
    if model is None:
        def fn(model):
            if model is None:
                raise RuntimeError("Model can't be None")
            return compile(model, fullgraph=fullgraph, dynamic=dynamic,
                          backend=backend, mode=mode, options=options, disable=disable)
        return fn

    # 3. mode 与 options 互斥校验
    if mode is not None and options is not None:
        raise RuntimeError(
            "Either mode or options can be specified, but both can't be specified at the same time.")
    if mode is None and options is None:
        mode = "default"  # 默认模式

    # 4. 编译器二分查找（调试用）
    from torch._inductor.compiler_bisector import CompilerBisector
    if bisect_backend := CompilerBisector.get_backend():
        backend = bisect_backend

    # 5. 提取 guard_filter_fn（不安全的高级选项）
    guard_filter_fn = None
    if options and isinstance(options, dict):
        guard_filter_fn = options.pop("guard_filter_fn", None)

    # 6. 选择后端包装器
    if backend == "inductor":
        backend = _TorchCompileInductorWrapper(mode, options, dynamic)
    else:
        backend = _TorchCompileWrapper(backend, mode, options, dynamic)

    # 7. 委托给 _dynamo.optimize
    return torch._dynamo.optimize(
        backend=backend,
        nopython=fullgraph,       # fullgraph=True → nopython=True
        dynamic=dynamic,
        disable=disable,
        guard_filter_fn=guard_filter_fn,
    )(model)
```

> **💡 提示：** 关键参数映射：`fullgraph=True` 会被映射为 `nopython=True` 传给 Dynamo。这是命名上的细微差异——"nopython"是 Dynamo 内部术语，表示"不允许回退到 Python 解释器执行"，等价于"不允许图断裂"。

### 2.2 后端包装器：_TorchCompileInductorWrapper vs _TorchCompileWrapper

`torch.compile()` 根据 `backend` 参数选择不同的后端包装器，二者职责不同：

### _TorchCompileInductorWrapper —— 默认

当 `backend == "inductor"` 时使用。专为 Inductor 后端设计，负责：

- 解析 mode（default/reduce-overhead/max-autotune 等）为 Inductor 配置
- 合并 options 字典与 mode 默认值
- 处理 dynamic 形状策略
- 返回真正的 `compile_fx` 编译函数

它理解 Inductor 的内部配置体系，能将高层 mode 翻译为细粒度选项。

### _TorchCompileWrapper —— 通用

当 backend 为字符串（非 "inductor"）或可调用对象时使用。职责较薄：

- 查找注册的后端函数（通过 `lookup_backend`）
- 对 mode 进行基本校验
- 透传 options 给后端
- 返回后端可调用对象

不假设后端理解 Inductor 配置，保持通用性。

> **⚠️ 注意：** mode 与 options 互斥：从源码可见，`mode` 和 `options` 不能同时指定。若需细粒度控制，应使用 `options` 字典；若使用预设模式，则用 `mode` 字符串。这是为了避免配置冲突的显式约束。

### 2.3 编译模式 (mode) 解析

mode 参数将常用配置组合预设为字符串，避免用户记忆繁杂的选项键名。各模式的语义与底层配置如下：

| 模式 | 语义 | 关键配置 | 适用场景 |
|---|---|---|---|
| `"default"` | 平衡性能与编译开销 | 标准融合策略，无 CUDA Graphs | 通用场景、首次尝试 |
| `"reduce-overhead"` | 减少 Python 开销 | 启用 CUDA Graphs，缓存工作空间内存 | 小 batch、CPU 受限场景 |
| `"max-autotune"` | 最大性能，编译时间最长 | Triton matmul 模板、卷积自动调优、启用 CUDA Graphs | 对延迟敏感的生产部署 |
| `"max-autotune-no-cudagraphs"` | 最大调优但禁用 CUDA Graphs | 同 max-autotune 但禁用 CUDA Graphs | 动态形状或 CUDA Graphs 不兼容时 |

> **✨ 技巧：** 查看模式对应配置：可通过 `torch._inductor.list_mode_options()` 查看每个 mode 具体设置了哪些 Inductor 配置项。这对于理解"为什么 reduce-overhead 比 default 快"等问题的根因非常有用。

> **⚠️ 注意：** CUDA Graphs 限制：reduce-overhead 和 max-autotune 默认启用 CUDA Graphs，但有严格限制：仅适用于纯 CUDA 图（无 CPU 操作）、不修改输入张量、且静态形状。若不满足，可通过 `TORCH_LOGS=perf_hints` 查看不适用原因，并改用 max-autotune-no-cudagraphs。

### 2.4 _dynamo.optimize: 核心入口

定义于 `torch/_dynamo/eval_frame.py#L1055`，是 Dynamo 的真正入口。它接收后端包装器，构建编译上下文：

```python
def optimize(*args, **kwargs):
    # rebuild_ctx 用于在重编译时重建上下文
    def rebuild_ctx():
        # 支持 compiled_autograd 的 kwargs 覆盖
        ca_kwargs_override = config.compiled_autograd_kwargs_override
        if ca_kwargs_override:
            assert set(ca_kwargs_override.keys()) == {"fullgraph"}
            kwargs["nopython"] = ca_kwargs_override["fullgraph"]
        return optimize(*args, **kwargs)
    return _optimize(rebuild_ctx, *args, **kwargs)


def _optimize(
    rebuild_ctx: Callable[[], Union[OptimizeContext, _NullDecorator]],
    backend="inductor",
    *,
    nopython=False,       # True = 不允许图断裂
    guard_export_fn=None,
    guard_fail_fn=None,
    guard_filter_fn=None,
    disable=False,
    dynamic=None,         # 动态形状模式
    package=None,
) -> Union[OptimizeContext, _NullDecorator]:
    """The main entrypoint of TorchDynamo."""
    check_if_dynamo_supported()
    check_for_incompatible_configs()

    # 构建 Hooks 对象，串联 guard 回调
    hooks = Hooks(guard_export_fn=guard_export_fn, guard_fail_fn=guard_fail_fn,
                  guard_filter_fn=guard_filter_fn)

    # 禁用短路：返回空装饰器
    if disable or os.environ.get("TORCHDYNAMO_DISABLE") == "1" \
            or not justknobs_check("pytorch/compiler:enable_dynamo"):
        return _NullDecorator()

    # nopython 模式：断言整图编译
    if nopython:
        return optimize_assert(backend, dynamic=dynamic, hooks=hooks,
                            rebuild_ctx=rebuild_ctx, package=package)

    backend = get_compiler_fn(backend)
    backend_ctx_ctor = getattr(backend, "backend_ctx_ctor", null_context)

    # 构建 convert_frame 回调 → 包装到 _optimize_catch_errors
    return _optimize_catch_errors(
        convert_frame.convert_frame(backend, hooks=hooks, package=package),
        hooks,
        backend_ctx_ctor,
        dynamic=dynamic,
        compiler_config=(backend.get_compiler_config()
                         if hasattr(backend, "get_compiler_config") else None),
        rebuild_ctx=rebuild_ctx,
        package=package,
    )
```

> **📝 备注：** 多层嵌套的意义：`_optimize_catch_errors` 包装 `convert_frame` 包装 `backend`。这种洋葱式结构让每层各司其职：backend 负责实际编译，convert_frame 负责字节码到 FX 的转换，_optimize_catch_errors 负责错误捕获与上下文管理。

### 2.5 _TorchDynamoContext: 上下文管理

定义于 `torch/_dynamo/eval_frame.py#L555`，是连接 Python 调用与帧评估钩子的核心类。它的 `__call__` 方法决定了如何包装用户传入的可调用对象：

```python
class _TorchDynamoContext:
    def __init__(self, callback, on_enter=nothing, backend_ctx_ctor=null_context,
                 patch_fn=nothing, first_ctx=False, *, export=False,
                 dynamic=None, compiler_config=None, package=None):
        assert callable(callback) or callback is False or callback is None
        self.callback = callback          # convert_frame 编译回调
        self._backend_ctx_ctor = backend_ctx_ctor
        self.prior = unset
        self.first_ctx = first_ctx
        self.export = export
        self._dynamic = dynamic
        self.compiler_config = compiler_config
        self.cleanup_fns = []
        self.enter_exit_hooks = []
        self._package = package
        patch_fn()

        # 保存后端以便 reset 时清理
        backend = innermost_fn(callback)
        cached_backends.setdefault(id(backend), backend)

        # 动态形状钩子
        if dynamic is not None:
            self.enter_exit_hooks.append(make_set_enable_dynamic(dynamic))

    def __enter__(self):
        # 上下文管理器模式: with torch._dynamo.optimize(...):
        if config.raise_on_ctx_manager_usage:
            raise RuntimeError("请使用装饰器/直接调用，而非 with 语句")
        self.prior = set_eval_frame(None)  # 激活钩子
        self.cleanup_fns = [enter() for enter in self.enter_exit_hooks]
        ...

    def __call__(self, fn):
        fn = innermost_fn(fn)

        # 情况 1: nn.Module → 包装为 OptimizedModule
        if isinstance(fn, torch.nn.Module):
            mod = fn
            new_mod = OptimizedModule(mod, self)
            new_mod._torchdynamo_orig_callable = mod.forward
            new_mod.get_compiler_config = self.compiler_config
            return new_mod

        # 情况 2: 类 → 包装 __call__ 和 _call_impl
        if inspect.isclass(fn):
            cls_obj = fn
            cls_obj.__call__ = self(cls_obj.__call__)
            if issubclass(cls_obj, torch.nn.Module):
                cls_obj._call_impl = self(cls_obj._call_impl)
            return cls_obj

        # 情况 3: 普通函数 → compile_wrapper
        @functools.wraps(fn)
        def compile_wrapper(*args, **kwargs):
            prior = set_eval_frame(None)  # 激活帧评估钩子
            try:
                # 检测嵌套 FX / JIT 追踪并报错
                if is_fx_tracing():
                    raise RuntimeError("不支持嵌套 FX 追踪")
                if is_jit_tracing():
                    raise RuntimeError("不支持嵌套 JIT 追踪")
                return fn(*args, **kwargs)  # 执行 → 钩子拦截 → 编译
            finally:
                set_eval_frame(prior)  # 恢复

        return compile_wrapper
```

> **💡 提示：** 关键机制：`set_eval_frame(callback)` 通过 PEP 523 设置 C 层帧评估钩子。当 Python 执行被编译函数的字节码时，钩子被触发，Dynamo 接管执行流程进行符号化追踪。钩子是进程级的，因此必须用 try/finally 保证恢复。

> **⚠️ 注意：** 三种包装路径：注意 `__call__` 区分了三种情况：nn.Module 返回 OptimizedModule（新对象）；类则原地修改其 `__call__`；普通函数返回 compile_wrapper 闭包。理解这三种路径有助于解释"为什么编译后的对象类型不同"。

### 2.6 OptimizedModule: nn.Module 的编译包装

对于 `nn.Module`，Dynamo 返回 `OptimizedModule` 包装器而非简单的闭包。这是为了保留 nn.Module 的语义（参数、缓冲区、子模块访问等），同时在其 `forward()` 中激活帧评估钩子。

```python
class OptimizedModule(torch.nn.Module):
    """
    编译后的 nn.Module 包装器。
    - 保留原模块的所有参数/缓冲区/子模块（通过 __getattr__ 转发）
    - forward() 中激活 Dynamo 帧评估钩子
    - 钩子拦截 orig_callable (即原始 forward) 的字节码
    """

    def __init__(self, mod, dynamo_ctx):
        # 真正的编译发生在第一次 forward 调用时
        # 这里仅保存原始模块和上下文
        super().__init__()
        self._orig_module = mod
        self._dynamo_ctx = dynamo_ctx
        # 保存原始 forward 以便 Dynamo 拦截
        self._torchdynamo_orig_callable = mod.forward

    def forward(self, *args, **kwargs):
        # 激活帧评估钩子 → 调用原始 forward → 钩子拦截编译
        return self._dynamo_ctx(self._torchdynamo_orig_callable)(*args, **kwargs)

    # 关键: 透明转发属性访问到原始模块
    # 使得 model.parameters() / model.layer1 等仍可用
```

> **✨ 技巧：** 为什么用 OptimizedModule 而非闭包？nn.Module 有丰富的协议（parameters、buffers、state_dict、子模块树等）。若用普通函数闭包包装，这些协议会丢失，导致 `optimizer = Adam(model.parameters())` 等代码失效。OptimizedModule 继承 nn.Module 并转发属性，保证与生态兼容。

> **📝 备注：** 惰性编译：OptimizedModule 在构造时**不**触发编译。真正的编译发生在第一次 `forward()` 调用时——此时帧评估钩子才被激活并拦截字节码。这意味着 `torch.compile(model)` 本身开销极小，真正的开销在首次推理。

### 2.7 完整调用流程图

下图展示从用户调用到 OptimizedModule 返回的完整链路：

```text
用户: model = torch.compile(model, mode="max-autotune")
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ torch.compile()  [torch/__init__.py#L2466]              │
│                                                         │
│  1. 版本检查 (Python 3.14+ 拒绝)                         │
│  2. 装饰器模式判断 (model is None?)                      │
│  3. mode/options 互斥校验                                │
│  4. mode=None & options=None → mode="default"           │
│  5. 提取 guard_filter_fn                                │
│                                                         │
│  6. 选择后端包装器:                                       │
│     ┌─ "inductor" → _TorchCompileInductorWrapper        │
│     │   (解析 mode → Inductor 配置 → compile_fx)        │
│     └─ 其他      → _TorchCompileWrapper                 │
│         (查找注册后端 → 透传)                            │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│ torch._dynamo.optimize(backend, nopython=fullgraph)     │
│  [eval_frame.py#L1040]                                  │
│                                                         │
│  → _optimize(rebuild_ctx, backend, nopython, dynamic)   │
│                                                         │
│  1. 检查 disable / TORCHDYNAMO_DISABLE → _NullDecorator │
│  2. nopython=True → optimize_assert (整图模式)          │
│  3. backend = get_compiler_fn(backend)                  │
│  4. 构建 convert_frame(backend, hooks) 回调              │
│  5. 返回 _optimize_catch_errors(...)                    │
│     → 即 _TorchDynamoContext 实例                       │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│ _TorchDynamoContext.__call__(model)  [eval_frame.py#631]│
│                                                         │
│  isinstance(model, nn.Module)?                          │
│     │                                                   │
│     ├─ 是 → OptimizedModule(model, self)                │
│     │      保存 _torchdynamo_orig_callable = model.forward│
│     │      返回新的 OptimizedModule 实例                 │
│     │                                                   │
│     └─ 否 → compile_wrapper 闭包                        │
│            调用时: set_eval_frame(None) → fn() → 恢复   │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
        返回 OptimizedModule (或 compile_wrapper)

  ※ 注意: 此时编译尚未发生！
    首次 model(input) 调用 forward 时才触发 PEP 523 钩子 → 编译
```

### 2.8 装饰器 vs 直接调用模式

torch.compile 支持多种使用方式，理解其差异有助于选择合适模式：

### 直接调用 —— 推荐

`model = torch.compile(model)`

显式、清晰，返回新的 OptimizedModule。原 model 不变，可对比编译前后效果。

### 装饰器 (带参数) —— 常用

`@torch.compile(mode="max-autotune")`
`def fn(x): ...`

利用 model is None 的装饰器分支，返回包装函数。适合函数或小型模块。

### 装饰器 (无参数) —— 便捷

`@torch.compile`
`def fn(x): ...`

model 即 fn 本身，直接进入主路径。最简形式。

> **⚠️ 注意：** 装饰器陷阱：`@torch.compile`（无括号）与 `@torch.compile()`（带括号）行为不同。前者 model=fn，立即进入主路径；后者 model=None，先返回装饰器 fn 再调用。两者最终效果一致，但带括号形式可传参，无括号形式只能用默认配置。

> **✨ 技巧：** inplace 编译：除 `torch.compile(model)` 返回新对象外，`nn.Module` 还提供 `model.compile()` 方法进行原地编译（修改自身而不返回新对象）。这在需要保持对象引用不变的场景（如已传给 optimizer 的模型）很有用。

### 2.9 入口层关键文件索引

| 文件 | 行号 | 核心符号 | 职责 |
|---|---|---|---|
| `torch/__init__.py` | 2466-2634 | `compile()` | 公共 API 入口 |
| `torch/__init__.py` | - | `_TorchCompileInductorWrapper` | Inductor 后端包装器 |
| `torch/__init__.py` | - | `_TorchCompileWrapper` | 通用后端包装器 |
| `torch/_dynamo/eval_frame.py` | 1040-1141 | `optimize()`, `_optimize()` | Dynamo 主入口 |
| `torch/_dynamo/eval_frame.py` | 555-714 | `_TorchDynamoContext` | 上下文管理与包装 |
| `torch/_dynamo/eval_frame.py` | - | `set_eval_frame()` | PEP 523 钩子激活 |
| `torch/_dynamo/eval_frame.py` | - | `OptimizedModule` | nn.Module 编译包装器 |
| `torch/_dynamo/convert_frame.py` | - | `convert_frame()` | 帧转换回调构造 |

> **💡 提示：** 下一章预告：至此，我们已经把"用户 API → Dynamo 上下文"的路径走通。下一章将深入 Dynamo 内部，剖析 PEP 523 帧评估钩子触发后，`InstructionTranslator` 如何逐条解释字节码并构建 FX 计算图。

## Related

- [01 架构总览](./01-architecture-overview.md)
- [03 TorchDynamo 前端](./03-torchdynamo-frontend.md)
- [PyTorch 索引](../index.md)
