---
title: TorchDynamo 前端
type: concept
status: seed
tags: [PyTorch, torch.compile, TorchDynamo]
created: 2026-08-17
updated: 2026-08-17
source:
  - d:\project\pytorch-2.8.0\wiki\03_dynamo.html
---

# TorchDynamo 前端

> 字节码级别的计算图捕获引擎

Dynamo 是 torch.compile 的图捕获前端，核心位于 `torch/_dynamo/`。它通过 Python 字节码分析而非 AST 分析来捕获计算图，能精确处理 Python 控制流和数据结构。本章深入 Dynamo 的核心机制：PEP 523 钩子、字节码符号化追踪、变量系统、OutputGraph 构建与 tracing context 管理。

> **💡 提示：** Dynamo 的定位：Dynamo 是"翻译官"——它将 Python 字节码翻译为 FX 计算图，但**不**负责反向图构造或内核生成。它的输入是字节码，输出是 FX GraphModule + Guard 集合 + 自定义字节码。

### 3.1 帧评估钩子 (PEP 523)

Dynamo 利用 CPython 的 PEP 523 帧评估 API 在 C 层拦截函数执行。这是整个编译系统的"启动开关"。

#### PEP 523 是什么？

PEP 523（"Adding a frame evaluation API to CPython"）为 CPython 引入了一个 C 级别的钩子，允许第三方代码替换默认的帧评估函数 `_PyEval_EvalFrameDefault`。每当 Python 即将执行一个帧（函数调用）时，这个钩子会被调用，钩子可以决定：

- **自己处理该帧：** 返回执行结果（Dynamo 的编译路径）
- **委托默认评估器：** 回退到标准 Python 解释器（eager 路径）

#### set_eval_frame 的工作原理

Dynamo 的 `set_eval_frame(callback)` 是 PEP 523 的封装。它将 Dynamo 的回调安装到 CPython 的帧评估槽位：

```python
# torch/_dynamo/eval_frame.py (简化)
def set_eval_frame(callback):
    """
    设置 CPython 的帧评估钩子。

    callback=None  → 安装 Dynamo 回调 (激活编译)
    callback=False → 卸载钩子 (恢复默认 eager)
    callback=<callable> → 安装指定回调

    返回值: 之前的回调 (用于恢复)
    """
    # 调用 C 扩展设置 _PyEval_EvalFrameDefault 的替代函数
    return torch._C._dynamo.eval_frame.set_eval_frame(callback)

# C 层伪代码 (eval_frame.c)
# Dynamo 安装的钩子大致如下:
PyObject* dynamo_eval_frame(PyThreadState* tstate, PyFrameObject* frame, ...) {
    code = frame->f_code;

    # 1. 遍历该 code object 的缓存链表
    CacheEntry* entry = code->cache_entry;
    while (entry != NULL) {
        # 执行 Guard 检查函数 (编译时生成的 C 函数)
        if (entry->guarded_code.check_fn(frame)) {
            # Guard 通过! 使用缓存的编译字节码
            return PyEval_EvalCode(entry->guarded_code.code, ...);
        }
        entry = entry->next;
    }

    # 2. 无缓存命中 → 调用 Python 层 callback 进行编译
    return call_python_callback(frame);
}
```

> **⚠️ 注意：** 钩子是进程级的：PEP 523 钩子作用于整个 CPython 解释器，而非单个线程或函数。这就是为什么 `compile_wrapper` 必须用 try/finally 保证 `set_eval_frame(prior)` 恢复——否则钩子会泄漏到不相关的代码，导致意外编译。

#### 帧评估的完整流程

```text
用户调用 model(inputs)
  │
  ├─ CPython 准备执行 model.forward 的字节码
  │
  ├─ PEP 523 帧评估钩子被触发
  │    │
  │    ├─ 检查缓存: 该 code object 是否已有编译结果?
  │    │    │
  │    │    ├─ 有缓存 → 执行 Guard 检查
  │    │    │    ├─ Guard 通过 → 执行缓存的编译代码 ✓
  │    │    │    └─ Guard 失败 → 尝试下一个缓存 / 重新编译
  │    │    │
  │    │    └─ 无缓存 → 进入编译流程
  │    │
  │    └─ 编译流程: convert_frame._compile()
  │         ├─ 字节码分析 + 符号化追踪
  │         ├─ 生成 FX GraphModule
  │         ├─ 调用后端编译 (inductor)
  │         └─ 生成自定义字节码 + 存入缓存
  │
  └─ 执行编译后的字节码
```

### 3.2 字节码符号化追踪

核心类 `InstructionTranslator` 定义于 `torch/_dynamo/symbolic_convert.py`，它逐条解释执行 Python 字节码指令，将操作转换为符号变量。这是 Dynamo 的"心脏"。

```python
def _compile(
    code: CodeType, globals, locals, builtins, closure,
    compiler_fn: CompilerFn, one_graph: bool, export: bool,
    export_constraints, hooks, cache_entry, cache_size,
    frame=None, frame_state=None, *,
    compile_id, skip=0, package=None,
) -> ConvertFrameReturn:
    """将 Python 帧完全转换为 FX 图"""
    output = None
    tracer = None

    tf_mode_stack = torch.overrides._get_current_function_mode_stack()

    @preserve_global_state
    def transform(instructions, code_options):
        nonlocal output, tracer
        speculation_log.restart()
        exn_vt_stack = ExceptionStack()

        # 创建指令翻译器
        tracer = InstructionTranslator(
            instructions, code, locals, globals, builtins, closure,
            tf_mode_stack, code_options, compiler_fn, one_graph, export,
            export_constraints, frame_state=frame_state,
            speculation_log=speculation_log, exn_vt_stack=exn_vt_stack,
            distributed_state=distributed_state, package=package,
        )

        # 执行符号化追踪: 逐条解释字节码
        try:
            tracer.output.mark_bytecode_tracing_start()
            with tracing(tracer.output.tracing_context), tracer.set_current_tx():
                tracer.run()
        except exc.UnspecializeRestartAnalysis:
            speculation_log.clear()
            raise
        except (exc.SpeculationRestartAnalysis,
                exc.TensorifyScalarRestartAnalysis, exc.SkipFrame):
            raise
        finally:
            tracer.output.call_cleanup_hooks()

        # 获取生成的 FX 图和自定义字节码
        output = tracer.output
        assert output.output_instructions
        instructions[:] = output.output_instructions
        code_options.update(output.code_options)
        # 死代码消除
        instructions[:] = remove_pointless_jumps(remove_dead_code(instructions))

    def _compile_inner(code, one_graph, hooks, transform):
        # 多次尝试编译 (支持重启分析)
        for attempt in itertools.count():
            try:
                # 调用 transform 进行字节码转换
                ...
            except exc.RestartAnalysis:
                continue  # 重启追踪

    return _compile_inner(code, one_graph, hooks, transform)
```

### 3.3 InstructionTranslator.run() 循环

`InstructionTranslator` 继承自 `InstructionPointer`，其 `run()` 方法是一个指令调度循环。它维护一个符号栈，对每条字节码指令调用对应的处理方法：

```python
class InstructionTranslator(InstructionPointer, LoggingBase):
    """逐条解释字节码，模拟栈式虚拟机"""

    def run(self):
        # 主循环: 遍历所有指令
        while True:
            inst = self.next_instruction()  # 取下一条指令
            if inst is None:
                break

            # 根据指令名分发到对应处理方法
            # 例: LOAD_FAST → self.LOAD_FAST(inst)
            #     BINARY_ADD → self.BINARY_ADD(inst)
            #     CALL_FUNCTION → self.CALL_FUNCTION(inst)
            opname = inst.opname
            if hasattr(self, opname):
                fn = getattr(self, opname)
                fn(inst)  # 调用处理方法
            else:
                self.unhandled_instruction(inst)  # 触发图断裂

    # 每条指令的处理方法操作符号栈 (self.stack)
    def LOAD_FAST(self, inst):
        # 从局部变量加载到栈顶
        val = self.local(varname)
        self.push(val)

    def BINARY_ADD(self, inst):
        # 弹出两个操作数，执行符号化加法
        b = self.pop()
        a = self.pop()
        self.push(a.__add__(b))  # 调用 VariableTracker.__add__

    def CALL_FUNCTION(self, inst):
        # 弹出函数和参数，执行符号化调用
        args = self.popn(inst.arg)
        fn = self.pop()
        self.push(fn.call_function(self, args, {}))
```

#### 字节码追踪示例

下面通过一个简单函数展示 Dynamo 如何将其字节码转换为 FX 图：

```python
# 原始 Python 函数
def f(x):
    return x + 1

# 对应的字节码 (CPython 3.11)
#   LOAD_FAST    0 (x)      ← 加载局部变量 x 到栈顶
#   LOAD_CONST   1 (1)      ← 加载常量 1 到栈顶
#   BINARY_ADD              ← 弹出 x 和 1，相加，结果入栈
#   RETURN_VALUE            ← 返回栈顶

# Dynamo 追踪过程:
# ┌─────────────────────────────────────────────────────────────┐
# │ 指令              │ 符号栈变化              │ FX 图操作       │
# ├─────────────────────────────────────────────────────────────┤
# │ LOAD_FAST 0 (x)  │ [TensorVariable(x)]    │ 创建 placeholder│
# │                  │                        │  节点: %x        │
# │ LOAD_CONST 1 (1) │ [TensorVariable(x),    │ (常量，不入图)   │
# │                  │  ConstantVariable(1)]  │                 │
# │ BINARY_ADD       │ [ComputationNode]      │ 创建 call_function│
# │                  │                        │  节点: %add =    │
# │                  │                        │  aten.add(%x, 1) │
# │ RETURN_VALUE     │ []                     │ 创建 output 节点 │
# │                  │                        │  output(%add)    │
# └─────────────────────────────────────────────────────────────┘

# 最终生成的 FX 图:
# Graph:
#   %x : Tensor [num_users=1] = placeholder[target=x]
#   %add : Tensor [num_users=1] = call_function[target=aten.add.Tensor]
#                              (args=(%x,), kwargs={})
#   return add

# 生成对应的自定义字节码 (调用编译后的内核)
#   LOAD_FAST    0 (x)
#   LOAD_GLOBAL  compiled_kernel  ← Dynamo 注入
#   CALL_FUNCTION 1
#   RETURN_VALUE
```

> **✨ 技巧：** 理解关键：Dynamo 并非"执行"字节码，而是"模拟执行"。它不真正计算张量值，而是记录操作语义到 FX 图。当遇到 `torch.*` 调用时，就在 OutputGraph 中创建对应的 FX 节点；遇到纯 Python 操作（如整数加法）则在符号层面求值，不产生 FX 节点。

#### InstructionTranslator 的工作方式总结

- **模拟栈式虚拟机：** 维护一个符号变量栈，每条字节码指令都有对应的处理方法
- **变量系统：** Python 对象被抽象为 `VariableTracker` 子类（TensorVariable、ListVariable、ConstantVariable 等）
- **FX 图构建：** 当遇到 `torch.*` 操作时，在 OutputGraph 中创建对应的 FX 节点
- **控制流处理：** 遇到 `if/for/while` 时，尝试展开为静态图；无法展开时触发图断裂
- **重启分析：** 遇到某些特殊情况（如需要去特殊化）时，会清空状态重新追踪

### 3.4 符号化变量系统

位于 `torch/_dynamo/variables/`，定义了符号化追踪时的变量抽象。所有变量都继承自 `VariableTracker`，它定义了符号操作的标准接口（如 `__add__`、`call_function`、`call_method` 等）。

> **💡 提示：** 为什么需要变量系统？Dynamo 不操作真实 Python 对象，而是操作它们的"符号代理"。这样可以在不真正执行计算的情况下推理程序语义。例如，`TensorVariable` 不持有真实张量，只记录其属性（dtype、shape、device 等），当遇到 `x + y` 时，它创建一个新的 FX 节点而非真正相加。

| 变量类 | 文件 | 代表的 Python 对象 | 关键行为 |
|---|---|---|---|
| `TensorVariable` | variables/tensor.py | torch.Tensor | torch.* 调用 → 创建 FX 节点 |
| `NNModuleVariable` | variables/nn_module.py | torch.nn.Module | 展开 forward 调用，内联参数 |
| `ListVariable` | variables/builder.py | list / tuple | 支持索引、迭代、append |
| `DictVariable` | variables/builder.py | dict | 支持键访问、迭代 |
| `ConstantVariable` | variables/constant.py | int / float / str / bool | 常量折叠，编译时求值 |
| `UserFunctionVariable` | variables/functions.py | 用户自定义函数 | 递归追踪函数体 |
| `UserMethodVariable` | variables/functions.py | 用户自定义方法 | 绑定 self 后追踪 |
| `BuiltinVariable` | variables/builtin.py | Python 内建函数 | 部分支持，否则图断裂 |
| `GetAttrVariable` | variables/base.py | 属性访问 (a.b) | 解析属性链 |
| `BannedVariable` | variables/builtin.py | 被禁用的对象 | 访问即报错（如某些不安全 API） |
| `NewGlobalVariable` | variables/base.py | 追踪期间新创建的全局 | 处理动态全局变量 |
| `ContextWrappingVariable` | variables/builtin.py | 上下文管理器 | 处理 with 语句 |
| `SymNodeVariable` | variables/symnode.py | 符号化形状表达式 | 动态形状推理的核心 |
| `base.VariableTracker` | variables/base.py | 所有变量的基类 | 定义符号操作接口 |

> **📝 备注：** 变量系统的可扩展性：每种 Python 对象类型都有对应的 VariableTracker 子类。当 Dynamo 遇到不认识的类型时，会尝试用 builder.py 中的 `variable_builder` 函数构造合适的变量；若仍无法处理，则触发图断裂。这种设计使得 Dynamo 能逐步扩展支持范围。

### 3.5 OutputGraph: FX 图构建

定义于 `torch/_dynamo/output_graph.py`，是 Dynamo 追踪的输出产物。它不仅构建 FX 图，还管理 Guard、字节码生成和图断裂协调。

```python
class OutputGraph:
    """Dynamo 追踪的输出，包含 FX 图 + Guard + 字节码"""

    # 1. FX 图构建
    graph: torch.fx.Graph           # 符号化追踪产生的计算图
    graphmodule: GraphModule        # 最终的 GraphModule

    # 2. Guard 收集
    guards: GuardSet                # 所有 Guard 条件

    # 3. 字节码生成
    output_instructions: list       # 生成的自定义字节码指令
    code_options: dict              # 修改后的 code object 属性

    # 4. 图断裂管理
    graph_calls: list               # 子图调用列表

    # 5. 追踪状态
    tracing_context: TracingContext # 当前追踪的上下文
    shape_env: ShapeEnv             # 符号形状环境

    def create_node(self, target, args, kwargs, name=None):
        """
        在 FX 图中创建新节点。
        当 InstructionTranslator 遇到 torch.* 调用时调用此方法。

        例: 遇到 x + y → create_node(aten.add.Tensor, (x, y), {})
        """
        node = self.graph.create_node("call_function", target, args, kwargs, name)
        return node

    def compile_and_call_fx(self, compiler_fn, ...):
        """调用后端编译器编译 FX 图"""
        # 1. 消除死代码
        self.graph.eliminate_dead_code()
        # 2. 构建 GraphModule
        gm = self._create_graph_module()
        # 3. 调用后端编译器
        compiled = compiler_fn(gm, example_inputs)
        return compiled

    def add_output_instructions(self, instructions):
        """将编译后的内核调用注入为字节码"""
        self.output_instructions.extend(instructions)
```

#### FX 节点创建流程

当 InstructionTranslator 遇到 `torch.*` 操作时，OutputGraph 如何创建 FX 节点：

```text
InstructionTranslator 遇到 CALL_FUNCTION 指令
  │
  ├─ 弹出函数对象: UserFunctionVariable(aten.add.Tensor)
  ├─ 弹出参数: [TensorVariable(x), TensorVariable(y)]
  │
  ▼
TensorVariable.call_function(tx, fn, args, kwargs)
  │
  ├─ 检查是否是支持的 torch 操作
  ├─ 收集 Guard (dtype、device、shape 等)
  │
  ▼
OutputGraph.create_node(
    target=aten.add.Tensor,
    args=(x_node, y_node),
    kwargs={}
)
  │
  ├─ self.graph.create_node("call_function", ...)
  │   → 返回新的 fx.Node
  │
  ▼
返回新的 TensorVariable，指向新创建的 fx.Node
  │
  ▼
InstructionTranslator 将新 TensorVariable 压入符号栈
```

> **💡 提示：** OutputGraph 的双重角色：它既是 FX 图的"构建器"（create_node），又是编译结果的"装配器"（compile_and_call_fx）。追踪阶段不断调用 create_node 累积节点；追踪结束后调用 compile_and_call_fx 触发后端编译，并将编译结果注入字节码。

### 3.6 追踪上下文 (Tracing Context)

Dynamo 通过 `TracingContext` 管理追踪期间的全局状态。它确保追踪过程中的元信息（如当前节点、FakeTensor 假设、符号形状）能被各组件访问。

```python
class TracingContext:
    """
    追踪期间的上下文，管理:
    - 当前 OutputGraph
    - 当前 InstructionTranslator
    - FakeTensorMode (用于创建 fake tensors)
    - ShapeEnv (符号形状环境)
    - Guard 收集状态
    - 调用栈与源码位置信息
    """
    output: OutputGraph
    fake_mode: FakeTensorMode
    shape_env: ShapeEnv

    # 使用 contextvar 实现线程安全的上下文传播
    current_ctx: contextvars.ContextVar

# 追踪时通过 with 语句激活:
with tracing(tracer.output.tracing_context), tracer.set_current_tx():
    tracer.run()  # 在此期间，任何代码都能访问 current_ctx
```

TracingContext 的关键作用包括：

- **状态传播：** 追踪期间调用的任意代码（包括 PyTorch 内部代码）都能获取当前追踪状态
- **源码定位：** 记录每个 FX 节点对应的源码位置，用于错误信息和调试
- **FakeTensor 共享：** 整个追踪过程共享同一个 FakeTensorMode，确保 fake tensor 一致性
- **Guard 累积：** 各处代码可以向上下文添加 Guard 条件
- **Graph 栈管理：** 支持嵌套追踪（如子图断裂后追踪新图）

> **✨ 技巧：** contextvars 的作用：Dynamo 使用 Python 的 `contextvars` 模块实现上下文传播。这意味着即使追踪过程中调用了 C 扩展或异步回调，只要回到同一个协程上下文，就能恢复 TracingContext。这是 Dynamo 能透明追踪复杂代码的基础。

### 3.7 FakeTensor: 追踪时的张量替身

Dynamo 在追踪时不操作真实张量，而是使用 **FakeTensor**——一种仅记录元数据（dtype、shape、device、stride）而不分配实际存储的张量替身。这是实现高效符号化追踪的关键。

```python
# FakeTensor 位于 torch/_subclasses/fake_tensor.py
class FakeTensor(Tensor):
    """
    一个不分配真实存储的张量子类。
    仅记录 dtype, shape, device, stride 等元数据。
    所有操作在 meta 设备上执行，不产生真实计算。
    """
    fake_device: torch.device
    real_tensor: None  # 不持有真实数据

# 追踪时:
# 1. 用户输入的真实张量被转换为 FakeTensor
real_x = torch.randn(128, 768)  # 真实张量
fake_x = FakeTensor.from_real(real_x)
# fake_x.shape == (128, 768), 但不占内存

# 2. 所有 torch 操作在 FakeTensor 上执行
fake_y = torch.relu(fake_x)
# fake_y.shape == (128, 768)  ← 仅推理形状
# 不真正计算 relu，只记录操作到 FX 图

# 3. 编译完成后，运行时使用真实张量执行编译代码
```

> **📝 备注：** FakeTensor 的价值：(1) **内存效率**：追踪时不分配真实存储，大模型追踪也不会 OOM；(2) **形状推理**：通过执行 meta 内核推断每步输出的形状，用于 Guard 生成；(3) **正确性**：FakeTensor 上的操作语义与真实张量一致，保证 FX 图正确。

> **⚠️ 注意：** FakeTensor 的局限：少数操作无法在 meta 设备上推理形状（如依赖数据的控制流 `if x.item() > 0`），这类操作会触发图断裂。此外，FakeTensor 不能用于验证数值正确性——它只检查形状与类型，不检查计算结果。

### 3.8 符号化形状 (Symbolic Shapes)

定义于 `torch/_dynamo/symbolic_shapes.py`，是 Dynamo 支持动态形状的核心：

- **ShapeEnv：** 管理所有符号维度变量及其约束关系
- **特殊化 vs 泛化：** 默认对遇到的每个具体数值创建符号变量，后续根据运行时变化决定是否特殊化
- **guards 中的形状检查：** 编译时记录形状假设（如 `s0 > 0`），运行时快速验证
- **动态模式 (`dynamic=True`)：** 主动将已知维度标记为动态，避免因形状变化触发重编译

> **✨ 技巧：** 动态形状的默认策略：当 `dynamic=None`（默认）时，Dynamo 采用"自动检测"策略：首次编译假设静态形状；若后续调用形状变化，则重编译为更动态的版本。这种渐进式策略在多数场景下兼顾性能与灵活性。

### 3.9 Dynamo 关键文件索引

| 文件 | 核心符号 | 职责 |
|---|---|---|
| `torch/_dynamo/eval_frame.py` | `set_eval_frame`, `_TorchDynamoContext` | PEP 523 钩子与上下文管理 |
| `torch/_dynamo/convert_frame.py` | `_compile`, `convert_frame` | 帧转换主流程 |
| `torch/_dynamo/symbolic_convert.py` | `InstructionTranslator` | 字节码符号化追踪 |
| `torch/_dynamo/output_graph.py` | `OutputGraph` | FX 图构建与编译协调 |
| `torch/_dynamo/variables/base.py` | `VariableTracker` | 符号变量基类 |
| `torch/_dynamo/variables/builder.py` | `variable_builder` | Python 对象 → 变量映射 |
| `torch/_dynamo/symbolic_shapes.py` | `ShapeEnv` | 符号化形状推理 |
| `torch/_dynamo/guards.py` | `GuardedCode`, `CheckFunctionManager` | Guard 生成与检查 |
| `torch/_dynamo/codegen.py` | 字节码生成函数 | 生成自定义字节码 |
| `torch/_subclasses/fake_tensor.py` | `FakeTensor`, `FakeTensorMode` | FakeTensor 实现 |

> **💡 提示：** 本章小结：Dynamo 通过 PEP 523 拦截 Python 帧执行，由 InstructionTranslator 逐条解释字节码，将 torch.* 操作转换为 FX 节点，同时收集 Guard 条件。追踪过程使用 FakeTensor 推理形状、VariableTracker 抽象 Python 对象、TracingContext 管理全局状态。最终产出 FX GraphModule 交给后端编译。下一章将深入 Guard 系统——它是 Dynamo 实现高效缓存复用的核心。

## Related

- [02 torch.compile 入口](./02-torch-compile-entry.md)
- [04 Guard 系统](./04-guard-system.md)
- [PyTorch 索引](../index.md)
