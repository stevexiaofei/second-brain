---
title: nanobot Tool Registry 源码精读
type: concept
status: seed
tags: [AI, Tools, Registry, Function Calling, Nanobot, MCP, Source Code]
created: 2026-08-17
updated: 2026-08-17
source:
  - d:\project\nanobot\nanobot\agent\tools\registry.py
  - d:\project\nanobot\nanobot\agent\tools\base.py
---

# nanobot Tool Registry 源码精读

> 本文基于本地源码 `d:\project\nanobot\nanobot\agent\tools\registry.py`（212 行）与 `d:\project\nanobot\nanobot\agent\tools\base.py` 拆解。目标：看懂模型发出的 tool call 是如何被解析、校验、执行、回传的。

## 一句话理解

`ToolRegistry` 是 nanobot 的**工具调度中心**：持有工具字典、生成给模型的 schema、把模型返回的 JSON 参数解析校验成可执行的函数调用。`Tool`/`ToolResult`/`Schema` 三个基类定义了工具的协议。

## 1. 数据结构

```python
class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._cached_definitions: list | None = None  # schema 缓存，register/unregister 时失效
```

工具本体是 `Tool`（ABC，[base.py#L159](file:///d:/project/nanobot/nanobot/agent/tools/base.py#L159)），每个工具实现：
- `name` / `description`：模型契约的标识
- `parameters`：JSON Schema（`to_schema()` 生成给模型的定义）
- `cast_params()`：参数类型转换
- `validate_params()`：参数校验
- `execute(**params)`：真正执行
- `concurrency_safe`：能否并行（runner 用它做批处理）

## 2. 两个最核心的方法

### 2.1 `get_definitions()`：生成给模型的 tool schema

[registry.py#L86-L108](file:///d:/project/nanobot/nanobot/agent/tools/registry.py#L86-L108)：

```python
def get_definitions(self):
    if self._cached_definitions is None:
        definitions = [tool.to_schema() for tool in self._tools.values()]
        builtins, mcp_tools = [], []
        for schema in definitions:
            name = self._schema_name(schema)
            (mcp_tools if name.startswith("mcp_") else builtins).append(schema)
        builtins.sort(key=self._schema_name)   # 内置工具稳定排序
        mcp_tools.sort(key=self._schema_name)  # MCP 工具追加在尾部
        self._cached_definitions = builtins + mcp_tools
    return self._cached_definitions
```

**设计要点**：
1. **结果缓存**，register/unregister 时才失效——因为 tool 定义直接决定 prompt cache 命中率，稳定顺序很重要。
2. **MCP 工具全部放最后**：内置工具作为稳定前缀（利于 prompt caching），MCP 工具（名字带 `mcp_` 前缀）排序追加。

### 2.2 `prepare_call()`：解析 + 校验一次 tool call

[registry.py#L110-L147](file:///d:/project/nanobot/nanobot/agent/tools/registry.py#L110-L147)：

```python
def prepare_call(self, name, params):
    tool = self._tools.get(name)
    if not tool:
        # 找不到工具 → 用"去标点小写"模糊匹配给建议
        suggestion = self._suggest_name(name)
        return None, params, ToolResult.error(
            f"Error: Tool '{name}' not found.{hint} Available: {', '.join(self.tool_names)}")
    if isinstance(tool, ContextAware) and (ctx := current_request_context()):
        tool.set_context(ctx)      # 兼容旧协议：注入请求上下文
    params = self._coerce_params(tool, params)   # ① 参数粗转换
    if not isinstance(params, dict):
        return tool, params, ToolResult.error("参数必须是 JSON object ...")
    cast_params = tool.cast_params(params)        # ② 类型转换
    errors = tool.validate_params(cast_params)    # ③ Schema 校验
    if errors:
        return tool, cast_params, ToolResult.error(
            f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors))
    return tool, cast_params, None                # ④ 成功
```

三段式：**coerce（JSON 字符串→dict）→ cast（类型转换）→ validate（schema 校验）**。任何一个失败都返回 `ToolResult.error`（带 `is_error=True`），runner 拿到后追加 `[Analyze the error above and try a different approach.]` 提示让模型自我纠错。

### 2.3 `_coerce_params`：处理 LLM 返回的"字符串参数"

[registry.py#L149-L185](file:///d:/project/nanobot/nanobot/agent/tools/registry.py#L149-L185)：

```python
@classmethod
def _coerce_argument_value(cls, value):
    # 字符串且以 { 或 [ 开头 → 尝试 json.loads 解析成 dict/list
    if isinstance(value, str) and stripped.startswith(("{", "[")):
        return json.loads(stripped) or value
    return value

@classmethod
def _unwrap_arguments_payload(cls, tool, params):
    # 处理 {"arguments": {...}} 包装形态（部分 provider 的 API 形态）
    if set(params) == {"arguments"}:
        if tool.parameters 里确实有 arguments 属性: 原样返回
        return cls._coerce_argument_value(params["arguments"])
```

**为什么需要**：不同 provider 的 function calling 返回形态不同——有的直接给结构化 JSON，有的给 JSON 字符串，还有的套一层 `{"arguments": ...}`。这里统一抹平。

## 3. `execute()`：便捷入口

[registry.py#L187-L201](file:///d:/project/nanobot/nanobot/agent/tools/registry.py#L187-L201)：

```python
async def execute(self, name, params):
    tool, params, error = self.prepare_call(name, params)
    if error:
        return ToolResult.error(str(error) + hint)
    try:
        result = await tool.execute(**params)
        if is_tool_error_result(result):
            return ToolResult.error(str(result) + hint)
        return result
    except Exception as e:
        return ToolResult.error(f"Error executing {name}: {str(e)}" + hint)
```

注意 runner 里其实走的是 `prepare_call` + `tool.execute(**params)` 两段式（[runner.py#L1440-L1470](file:///d:/project/nanobot/nanobot/agent/runner.py#L1440-L1470)），`ToolRegistry.execute` 是简化路径。

## 4. `ToolResult`：带错误标志的字符串

[base.py#L144-L156](file:///d:/project/nanobot/nanobot/agent/tools/base.py#L144-L156)：

```python
class ToolResult(str):
    """String-compatible tool output with structured status."""
    is_error: bool
    def __new__(cls, content, *, is_error=False):
        obj = str.__new__(cls, content)
        obj.is_error = is_error
        return obj
    @classmethod
    def error(cls, content): return cls(content, is_error=True)
```

**继承 `str` 但带 `is_error` 标志**——工具结果可以当字符串用（省得包对象），同时 runner 用 `is_tool_error_result()` 判断错误。

## 5. Schema 校验层（base.py 里的隐藏细节）

`Schema.validate_json_schema_value`（[base.py#L51-L121](file:///d:/project/nanobot/nanobot/agent/tools/base.py#L51-L121)）是一个完整的 JSON Schema 校验器：

- 类型检查：`integer`（且排除 bool！）、`number`（必须有限）、`string`/`array`/`object`
- 约束检查：`enum`、`minimum/maximum`、`minLength/maxLength`、`required`、`additionalProperties: false`、`minItems/maxItems`
- **错误信息带路径**：`missing required params.path`，格式对模型友好

## 6. 工具执行全链路（runner 视角）

```text
模型返回 tool_calls: [{name, arguments}]
  → AgentRunner._run_tool（runner.py#L1410）
      ├── repeated_external_lookup_error 限流检查
      ├── spec.tools.prepare_call(name, arguments)   ← 本篇核心
      │     ├── coerce（JSON 字符串→dict，unwrap arguments 包装）
      │     ├── cast_params（类型转换）
      │     └── validate_params（Schema 校验）
      ├── hook.before_execute_tool
      ├── tool.execute(**params)
      ├── 异常/错误结果 → _classify_violation（SSRF/workspace 分类）
      └── 返回 (result, event, error)
  → tool 消息 {role: "tool", tool_call_id, name, content} append 回 messages
  → 再次请求模型
```

## 7. 你该重点看的源码细节

| 问题 | 位置 |
|---|---|
| schema 缓存为什么重要 | `get_definitions`（[registry.py#L86-L108](file:///d:/project/nanobot/nanobot/agent/tools/registry.py#L86-L108)） |
| 参数包装形态怎么抹平 | `_unwrap_arguments_payload`（[registry.py#L175-L185](file:///d:/project/nanobot/nanobot/agent/tools/registry.py#L175-L185)） |
| 工具找不到给什么提示 | `_suggest_name`（[registry.py#L58-L69](file:///d:/project/nanobot/nanobot/agent/tools/registry.py#L58-L69)） |
| ToolResult 如何携带错误 | [base.py#L144-L156](file:///d:/project/nanobot/nanobot/agent/tools/base.py#L144-L156) |
| 完整 schema 校验 | `Schema.validate_json_schema_value`（[base.py#L51-L121](file:///d:/project/nanobot/nanobot/agent/tools/base.py#L51-L121)） |

## 8. 常见坑点

1. **工具定义顺序影响 prompt cache**：改工具顺序会破坏缓存前缀——这就是为什么 builtin 排序、MCP 垫底、结果缓存。
2. **不要直接改 schema/错误文案**：`[Analyze the error above and try a different approach.]` 这类后缀是给模型的纠错指令，是"对话式容错"设计的一部分。
3. **JSON 字符串参数必须 coerce**：模型经常返回字符串化 JSON，不解析就会在 `cast_params` 炸掉。
4. **参数校验错误信息要带路径**：模型要靠"哪个字段错了"来修正，信息越结构化收敛越快。

## Related

- [nanobot AgentLoop 与 AgentRunner 源码精读](./nanobot-agentloop-runner.md) — runner 如何调用 registry
- [nanobot ContextBuilder 源码精读](./nanobot-contextbuilder.md) — tool contract 如何进入 prompt
- [nanobot 源码阅读指南](./nanobot-source-reading-guide.md)
- [AI 索引](./index.md)
