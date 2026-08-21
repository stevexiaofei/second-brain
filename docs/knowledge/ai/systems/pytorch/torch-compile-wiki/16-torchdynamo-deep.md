---
title: TorchDynamo 深入
type: concept
status: seed
tags: [PyTorch, torch.compile, TorchDynamo, PEP 523, 字节码改写, CPython]
created: 2026-08-17
updated: 2026-08-17
source:
  - d:\project\pytorch-2.8.0\wiki\16_torchdynamo_deep.html
---

# 十六、TorchDynamo 深入

> 从 CPython 执行模型到 PEP 523 字节码改写

第三章已介绍 TorchDynamo 作为 `torch.compile` 前端的整体职责。本章从**底层原理**切入，回到 CPython 解释器本身，回答两个根本问题：Dynamo 为什么选择在字节码级别工作？它是如何"动态修改字节码"的？理解这些，才能透彻把握 Dynamo 相对传统 trace/AST 方案的本质优势。

> **💡 提示：**
>
> **本章定位：**第三章讲 Dynamo"做什么"（PEP 523 帧钩子、符号化追踪、OutputGraph）；本章讲 Dynamo"为什么这么做"——CPython 执行流水线、三种计算图捕获方式的对比、字节码改写的真实示例。是第三章的原理性补充。

### 16.1 背景：计算图捕获的三种方式

深度学习框架做编译优化时，需要先根据计算逻辑形成逻辑计算图，再改写计算图，最后执行改写后的图。计算图的**生成方式**有三种，关键区别在于"在 Python 执行流水线的哪个阶段介入"：

#### ① 基于 Trace Tensor（执行后）

- 跟踪 tensor 的执行路径。tensor 执行时基于函数重载，落到框架自定义的 C++ 函数中，该函数生成 Operation 的符号表达（如加法记录为符号化加法算子）。一连串运算即被转成符号化计算图。
- **介入时机：** Python VM 执行 ByteCode *之后*（tensor 真正被计算时）。
- **代表：** torch.jit.trace、Lazy Tensor。

#### ② 基于 AST 解析（执行前）

- 在代码执行前，直接根据 Python 文本代码得到 AST，再根据 AST 翻译成计算图（中间代码 IR）。
- **介入时机：** Python 源码 → AST 阶段。
- **代表：** torch.jit.script。

#### ③ 基于 ByteCode 改写（Dynamo）

- 动态修改 Python ByteCode，让 VM 第三阶段执行的就是修改后的 ByteCode。思想类似 DynamoRIO（动态修改 x86 机器码）。
- **介入时机：** AST → ByteCode 之后、ByteCode 执行之前。
- **代表：** TorchDynamo。

> **✨ 技巧：**
>
> **Dynamo 的独特价值：**Trace 方式受限于"执行才看得到"、AST 方式需重实现 Python 语义。Dynamo 工作在 ByteCode 阶段——既复用 CPython 官方语义（零歧义），又能在执行前改写（可缓存可复用），且支持任何 Python 语法（不支持就回退原 Frame）。

### 16.2 CPython 标准执行流程

理解 Dynamo 必须先理解 CPython 的三阶段执行流程：

```text
  Python 源码 (文本)
        │  ① 解析
        ▼
  AST (抽象语法树)
        │  ② 编译
        ▼
  ByteCode (字节码, PyCodeObject)
        │  ③ 执行
        ▼
  Frame (PyFrameObject) 在 VM 中执行
```

AST 解析的计算图生成发生在**阶段①**；基于 trace tensor 的计算图生成发生在**阶段③之后**；TorchDynamo 工作在**阶段②与③之间**，动态修改 ByteCode，使阶段③执行的已是改写后的 ByteCode。

#### 用标准组件观察这条流水线

Python 的标准组件非常易用，可在 Python 层用 `ast` 查看 AST、用 `compile` 编译 ByteCode、用 `exec` 执行 ByteCode。构造一段示例代码（含普通乘法、tensor 标量加法、打印当前 Frame 的 ByteCode）：

```python
import ast, dis, sys

src_code = """
# normal python operation
x = 1
x = x * 2

# tensor operation
y = dl_framework.ones((1, 2))
z = x + y
print(z)

# print python frame
f = sys._getframe()
print(f.f_code)
"""

# ① 源码 -> AST
ast_obj = ast.parse(src_code)
print(ast.dump(ast_obj))   # Module(body=[Assign(...), Assign(...), ...])

# ② AST -> ByteCode
code_obj = compile(ast_obj, filename="", mode="exec")
print(code_obj)            # <code object <module> at 0x7ff79bb5c660, ...>

# 展示 ByteCode 指令
print(dis.Bytecode(code_obj).dis())

# ③ 执行 ByteCode
import torch as dl_framework
exec(code_obj)
```

AST 基本是一棵多叉树，每个节点对应一个表达式。以 `x = x * 2` 为例：`Assign` 节点，被赋值的是 `x`，赋值的值是一个二元乘法 `BinOp(left=Name('x'), op=Mult(), right=Constant(2))`。AST 解析是纯文本层面的，`dl_framework` 还未被 import，AST 解析仍可正常工作。

`dis.Bytecode(...).dis()` 输出每条字节码指令，每行对应一条指令，通过字面含义即可看出在做什么：

```text
# x = 1
  3           0 LOAD_CONST               0 (1)
              2 STORE_NAME               0 (x)

# x = x * 2
  4           4 LOAD_NAME                0 (x)
              6 LOAD_CONST               1 (2)
              8 BINARY_MULTIPLY
             10 STORE_NAME               0 (x)

# y = dl_framework.ones((1, 2))
  7          12 LOAD_NAME                1 (dl_framework)
             14 LOAD_METHOD              2 (ones)
             16 LOAD_CONST               2 ((1, 2))
             18 CALL_METHOD              1
             20 STORE_NAME               3 (y)

# z = x + y
  8          22 LOAD_NAME                0 (x)
             24 LOAD_NAME                3 (y)
             26 BINARY_ADD
             28 STORE_NAME               4 (z)
```

> **📝 说明：**
>
> **关键观察：**运行时可获取当前 frame，通过 `frame.f_code` 拿到 frame 里包含的 ByteCode（code object），会发现它的指针就是之前编译时生成的那个。Frame（运行时对象，对应一次函数调用栈）中要执行的指令就是之前创建的 ByteCode。可以想象：如果这些指令被修改，Python VM 就会执行自定义指令——这正是 Dynamo 的入口。

### 16.3 CPython 帧执行的 C 层实现

CPython 在 C 层面执行 Frame 的入口是 `_PyEval_EvalFrameDefault`，主逻辑就是"取 ByteCode 指令 + 执行指令"：

```c
co = f->f_code;                          # 从 PyFrameObject* f 取出 PyCodeObject*
names = co->co_names;
consts = co->co_consts;
fastlocals = f->f_localsplus;
first_instr = (_Py_CODEUNIT *) PyBytes_AS_STRING(co->co_code);
next_instr = first_instr;

# 每条指令由 opcode + oparg 组成
# define NEXTOPARG()  ... next_instr++ ...

for (;;) {
    NEXTOPARG();                  # 取当前指令
    switch (opcode) {
        # case LOAD_FAST: ...
        # case STORE_FAST: ...
        # case BINARY_ADD: ...
        # case CALL_FUNCTION: ...
    }
}
```

每个指令类型对应一个 opcode（数值）。例如 `BINARY_ADD` 从栈顶弹出 right、次顶取 left，调用 `PyNumber_Add`，结果压栈；`CALL_FUNCTION` 调用 `call_function` 并把结果压栈。

### 16.4 PEP 523：帧评估钩子

CPython 提供了执行**自定义 Frame Evaluation API** 的能力（PEP 523）。默认的 Eval Frame 入口是 `_PyEval_EvalFrame`：默认情况下它直接调用 `_PyEval_EvalFrameDefault()` 处理未被修改的 frame；但如果发现存在一个自定义的 eval frame 函数，就会执行那个自定义函数：

```c
static inline PyObject * _PyEval_EvalFrame(
    PyThreadState *tstate,
    struct _PyInterpreterFrame *frame,
    int throwflag)
{
    if (tstate->interp->eval_frame == NULL) {
        // 默认 eval frame
        return _PyEval_EvalFrameDefault(tstate, frame, throwflag);
    }
    // 如果设置了 eval_frame 就会被执行
    return tstate->interp->eval_frame(tstate, frame, throwflag);
}
```

所以只要在 ByteCode 执行前设置一个自定义的 eval frame 函数即可。TorchDynamo 正是这么做的——在 Python 层基于 ContextManager 在进入 Dynamo 作用域时触发 `eval_frame` 的设置：

```python
# torch._dynamo.optimize(...) 对应的 context manager
class _TorchDynamoContext:
    def __init__(self, callback: DynamoCallback):
        self.callback = callback

    def __enter__(self):
        # 安装自定义 eval frame 钩子
        set_eval_frame(self.callback)

    def __exit__(self, *exc):
        # 恢复默认 eval frame
        set_eval_frame(None)
```

> **⚠️ 注意：**
>
> **版本耦合代价：**字节码追踪与 CPython 版本强耦合（不同 Python 版本字节码指令集不同），这是 `torch.compile` 对 Python 版本有要求（如不支持 3.14+）的原因。PEP 523 是 CPython 专属 API，也限制了跨解释器兼容性。

### 16.5 ByteCode Rewrite 实战

Dynamo 在标准 Python 执行流程中做的核心改变是支持修改 Frame 执行前的 ByteCode，即 `Source → ByteCode → [ByteCode rewrite] → Evaluate`。ByteCode rewrite 的工作方式：把一段 ByteCode 转成 FX Graph，调用用户自定义的 FX Graph 改写执行逻辑，生成一个可编译的执行函数；然后把该段 ByteCode 替换成"函数调用 ByteCode"，调用的就是那个编译过的函数。

下面用一个 `fn()` 函数的编译示例展开。原始函数与字节码：

```python
def fn(a, b):
    x = a + b
    x = x / 2.0
    if x.sum() < 0:
        return x * -1.0
    return x

with torchdynamo.optimize(custom_compiler):
    fn(torch.randn(10), torch.randn(10))

# 原始 ByteCode（与代码对应关系见注释）
# x = a + b
  0 LOAD_FAST    0 (a)
  2 LOAD_FAST    1 (b)
  4 BINARY_ADD
  6 STORE_FAST   2 (x)
# x = x / 2.0
  8 LOAD_FAST    2 (x)
 10 LOAD_CONST   1 (2.0)
 12 BINARY_TRUE_DIVIDE
 14 STORE_FAST   2 (x)
# if x.sum() < 0:
 16 LOAD_FAST    2 (x)
 18 LOAD_METHOD  0 (sum)
 20 CALL_METHOD  0
 22 LOAD_CONST   2 (0)
 24 COMPARE_OP   0 (<)
 26 POP_JUMP_IF_FALSE 36
# return x * -1.0
 28 LOAD_FAST    2 (x)
 30 LOAD_CONST   3 (-1.0)
 32 BINARY_MULTIPLY
 34 RETURN_VALUE
# return x
 36 LOAD_FAST    2 (x)
 38 RETURN_VALUE
```

经过 TorchDynamo 动态改写后的 ByteCode：

```text
# x = a + b ; x = x / 2.0 ; x.sum() < 0  被转换成 __compiled_fn_0
# __compiled_fn_0 返回 (x, x.sum()<0) 组成的 tuple
  0 LOAD_GLOBAL    1 (__compiled_fn_0)
  2 LOAD_FAST      0 (a)
  4 LOAD_FAST      1 (b)
  6 CALL_FUNCTION  2
  8 UNPACK_SEQUENCE 2
 10 STORE_FAST     2 (x)
 12 POP_JUMP_IF_FALSE 22
# x * -1.0 被转换成 __compiled_fn_1
 14 LOAD_GLOBAL    2 (__compiled_fn_1)
 16 LOAD_FAST      2 (x)
 18 CALL_FUNCTION  1
 20 RETURN_VALUE
# return x
 22 LOAD_FAST      2 (x)
 24 RETURN_VALUE
```

可以看到新增了两个函数调用 `__compiled_fn_0` 和 `__compiled_fn_1`，它们对应的 FX Graph 如下：

```text
# __compiled_fn_0
opcode         name   target                    args           kwargs
-------------  ------  -------------------------  --------------  --------
placeholder    a_0    a_0                        ()             {}
call_function  add    <built-in function add>    (a_0, b_1)     {}
call_function  truediv <built-in function truediv> (add, 2.0)    {}
call_method    sum_1  sum                        (truediv,)     {}
call_function  lt     <built-in function lt>     (sum_1, 0)     {}
output         output output                     ((truediv, lt),) {}

# __compiled_fn_1
placeholder    x_4    x_4                        ()             {}
call_function  mul    <built-in function mul>    (x_4, -1.0)    {}
output         output output                     ((mul,),)      {}
```

在 ByteCode rewrite 的最后，TorchDynamo 为这段代码的输入创建两个 Guard：局部参数 `a` 必须是 Tensor、局部参数 `b` 必须是 Tensor。该 `fn` 被再次调用时，若符合这两个条件即可命中缓存的 Dynamo 处理结果；否则触发新的 ByteCode 分析和变换。对于与 tensor 无关的、比较特别的 Python 代码，其 ByteCode 保持原状——这样达到了"不需要用户标注区域、自动寻找优化机会"的设计目标。

### 16.6 改写后的 Frame 全景

TorchDynamo 把原来的 `PyFrameObject` 替换成 Patched `PyFrameObject`（CPython 支持的特性）。这个 Patched Frame 中最主要的改动是 Frame 中的 ByteCode（`PyCodeObject`）被修改了：原来的 `PyCodeObject` 变成 Transformed `PyCodeObject`。被改写的 `PyCodeObject` 如上节所示，主要是部分 ByteCode 被替换成"调用被编译过函数"，该函数支持自定义编译逻辑，当前默认的编译接口是 FX Graph。

```text
  原始 PyFrameObject                 Patched PyFrameObject
 ┌────────────────────┐             ┌────────────────────────────┐
 │  ...               │             │  ...                       │
 │  PyCodeObject ─────┼──改写──▶    │  Transformed PyCodeObject  │
 │  (原始 ByteCode)   │             │  (部分 ByteCode 被替换为    │
 │  ...               │             │   调用 __compiled_fn_N)    │
 └────────────────────┘             └────────────────────────────┘
        │                                     │
        ▼                                     ▼
  _PyEval_EvalFrameDefault            _PyEval_EvalFrameDefault
  执行原始 ByteCode                    执行改写后的 ByteCode
                                            │
                                            ▼
                                  __compiled_fn_N (FX Graph → 编译后函数)
```

### 16.7 Dynamo 的优势小结

| 优势 | 说明 |
| --- | --- |
| 支持所有 Python 语法 | 自定义 Frame 过程中任何一点发现不支持，都可回退到原 Frame |
| 开销少 | 劫持发生在 Python 执行较早阶段（ByteCode 生成/优化），而非执行后；多次 ByteCode 调用被融合为一次可缩减开销 |
| 可做到不增加编译延迟 | 可在识别热点代码后单独开线程做编译而不影响主线程（PEP 523 有延迟编译样例，如 Pyjion） |
| 动态性 | 同一段源码每次执行都走到 rewrite 步骤，可选择是否 rewrite、做何种 rewrite，并支持结果缓存复用 |
| FX Graph 作为接口 | 把 ByteCode 段转成 FX Graph，提供 Python 层 Pass 便利性（见[第十五章 torch.fx](./15-torch-fx-special.md)） |

> **📝 说明：**
>
> **与第三章的衔接：**第三章描述的 `InstructionTranslator`（逐条解释字节码）、`OutputGraph`（构建 FX Graph）、`guards.py`（生成 Guard）、`codegen.py`（生成自定义字节码+缓存）——正是本章所述 ByteCode rewrite 流程在源码层面的落地。本章是"为什么"，第三章是"怎么做"。

参考：`torch/_dynamo/eval_frame.py` `torch/_dynamo/convert_frame.py` `torch/_dynamo/symbolic_convert.py` `torch/_dynamo/codegen.py` `PEP 523`

## Related

- [15 torch.fx 专题](./15-torch-fx-special.md)
- [17 torch.compile 后端](./17-compile-backend.md)
- [PyTorch 索引](../index.md)
