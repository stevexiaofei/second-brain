---
title: nanobot AgentLoop 与 AgentRunner 源码精读
type: concept
status: seed
tags: [AI, Agents, AgentLoop, AgentRunner, Runtime, Nanobot, Source Code, MessageBus]
created: 2026-08-17
updated: 2026-08-17
source:
  - d:\project\nanobot\nanobot\agent\loop.py
  - d:\project\nanobot\nanobot\agent\runner.py
---

# nanobot AgentLoop 与 AgentRunner 源码精读

> 本文基于本地源码 `d:\project\nanobot\nanobot\agent\loop.py`（2358 行）与 `d:\project\nanobot\nanobot\agent\runner.py`（1670 行）逐段拆解，目标是让你能直接对着源码读懂每一层在干什么。

## 一句话理解

- `AgentLoop` = **产品层 turn 编排器**：从总线收消息 → 恢复/构造 session → 组装上下文 → 调 runner → 落盘 → 组 outbound 发回。
- `AgentRunner` = **模型层执行循环**：发请求 → 解析 response → 执行工具 → 结果回写 → 再发，直到 final answer。

二者分工清晰：**Loop 管"这一轮对话的完整生命周期"，Runner 管"模型+工具的迭代执行"**。

## 1. AgentLoop 内部结构总览（loop.py）

### 1.1 关键数据类型

`TurnKind` 枚举（[loop.py#L112-L114](file:///d:/project/nanobot/nanobot/agent/loop.py#L112-L114)）：

```python
class TurnKind(Enum):
    USER = auto()      # 用户消息
    SYSTEM = auto()    # 内部消息（subagent 结果、cron 触发等）
```

`TurnContext` dataclass（[loop.py#L117-L165](file:///d:/project/nanobot/nanobot/agent/loop.py#L117-L165)）——**贯穿一个 turn 所有阶段的"书包"**，字段按用途可分五组：

| 分组 | 字段 | 作用 |
|---|---|---|
| 来源 | `msg` / `session_key` / `turn_id` / `kind` | 消息本身 + 标识 |
| 运行时 | `runtime`(LLMRuntime) / `delivery`(TurnDelivery) | 模型运行时 + 投递通道 |
| 上下文 | `history` / `initial_messages` / `provider_state` / `request_context` / `runtime_context_blocks` | 给 LLM 的输入 |
| 结果 | `final_content` / `all_messages` / `stop_reason` / `had_injections` / `streamed_content` | 一轮产出 |
| 回调 | `on_progress` / `on_stream` / `on_stream_end` / `on_retry_wait` / `pending_queue` | 流式/进度/注入 |

两个守卫方法（[loop.py#L167-L177](file:///d:/project/nanobot/nanobot/agent/loop.py#L167-L177)）：`require_runtime()`（BUILD 阶段后才能用）、`require_session()`（RESTORE 阶段后才能用）——**这从机制上强制了阶段顺序**。

### 1.2 `__init__` 装配了哪些组件

`AgentLoop.__init__`（[loop.py#L256-L459](file:///d:/project/nanobot/nanobot/agent/loop.py#L256-L459)）是最大的装配点，全部依赖都在这里建立：

```python
self.bus = bus                    # MessageBus（inbound/outbound 队列）
self.context = ContextBuilder(...)  # prompt 组装（见 context 笔记）
self.sessions = SessionManager(workspace)  # session 持久化
self.tools = ToolRegistry()       # 工具注册表
self.runner = AgentRunner()       # 模型执行循环（本篇核心）
self.subagents = SubagentManager(...)  # 子 agent
self.consolidator = Consolidator(...)   # 记忆整理
self.auto_compact = AutoCompact(...)    # 过期 session 压缩
self._cron_turns = CronTurnCoordinator(...)
self._local_trigger_turns = LocalTriggerTurnCoordinator(...)
# 并发控制：NANOBOT_MAX_CONCURRENT_REQUESTS，默认 3，<=0 不限
self._concurrency_gate = asyncio.Semaphore(...)
# 每个 session 一个锁（weakref 字典，空闲自动回收）
self._session_locks = weakref.WeakValueDictionary(...)
self.commands = CommandRouter()   # 内置命令（/new /stop 等）
```

**关键设计**：per-session 锁用 `WeakValueDictionary`，锁跟随 session 活跃度自动过期，避免内存泄漏。

### 1.3 `run()` 主循环：消息从总线进来的第一条路径

[loop.py#L1172-L1266](file:///d:/project/nanobot/nanobot/agent/loop.py#L1172-L1266)：

```python
while self._running:
    msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
    # 1. 超时（1s 无消息）→ 检查过期 session 压缩，继续
    # 2. runtime control 消息（如 session discard）→ 立即处理
    # 3. 优先级命令（/stop 等）→ 内联分发
    # 4. automation turns（cron/local trigger）→ 若 session 活跃则 defer
    # 5. 若该 session 已有活跃 pending queue → 消息路由进队列（mid-turn 注入）
    # 6. 否则 create_task(self._dispatch(msg))
```

**每 session 一个 pending queue**（[loop.py#L416](file:///d:/project/nanobot/nanobot/agent/loop.py#L416)）是 nanobot 实现"回合中注入"（injection）的基石：当 agent 正在跑工具时又来一条消息，它不是新起任务，而是**塞进队列，由 runner 的 injection_callback 在合适时机拉取**（见 Runner 部分）。

### 1.4 `_dispatch()`：串行化 + 全生命周期

[loop.py#L1268-L1382](file:///d:/project/nanobot/nanobot/agent/loop.py#L1268-L1382)：

```python
async def _dispatch(self, msg):
    session_key = self._effective_session_key(msg)
    lock = self._get_session_lock(session_key)   # per-session 串行
    gate = self._concurrency_gate or nullcontext()
    async with lock, gate:
        pending = asyncio.Queue(maxsize=20)
        self._pending_queues[session_key] = pending
        delivery = self.turn_delivery_factory.create(msg, session_key, enable_stream=True)
        response = await self._process_message(msg, ...)  # 核心管线
        await delivery.complete(response, ...)
    finally:
        # 清理：残留 pending 消息重新 publish 到总线（不丢失）
        # 恢复 runtime checkpoint（取消/崩溃后保住部分上下文）
```

`_effective_session_key`（[loop.py#L839-L843](file:///d:/project/nanobot/nanobot/agent/loop.py#L839-L843)）处理 **unified session**：开启后所有 channel 的消息归并到 `UNIFIED_SESSION_KEY`。

### 1.5 `_process_message()`：七阶段 turn 管线（最值得画图的一段）

[loop.py#L1446-L1564](file:///d:/project/nanobot/nanobot/agent/loop.py#L1446-L1564)：

```text
restore → compact → command → build → run → save → respond
```

```python
await self._run_turn_stage(ctx, "restore", self._restore_turn)
await self._run_turn_stage(ctx, "compact", self._compact_session)
if await self._run_turn_stage(ctx, "command", self._dispatch_command):
    return ctx.outbound   # 命令走捷径，跳过 BUILD/SAVE
await self._run_turn_stage(ctx, "build", self._build_turn)
await self._run_turn_stage(ctx, "run", self._run_turn)
await self._run_turn_stage(ctx, "save", self._persist_turn)
await self._run_turn_stage(ctx, "respond", self._prepare_outbound)
return ctx.outbound
```

各阶段职责：

| 阶段 | 方法 | 干什么 |
|---|---|---|
| restore | `_restore_turn` | 非图片附件转引用；`get_or_create` session；按 `session.policy.disabled_tools` 过滤工具；恢复崩溃前的 runtime checkpoint |
| compact | `_compact_session` | 检查过期 session 是否需要压缩/总结，产出 `pending_summary` |
| command | `_dispatch_command` | 命中 `/xx` 命令直接返回，跳过模型调用 |
| build | `_build_turn` | 解析 runtime、consolidate、取 history、构造 `initial_messages`、早期持久化用户消息 |
| run | `_run_turn` | 调 `_run_agent_loop` → 最终 `AgentRunner.run` |
| save | `_persist_turn` | `_save_turn` 写入 session（截断/清洗工具结果）、清 checkpoint |
| respond | `_prepare_outbound` | 组装 `OutboundMessage`，可能附加 `StreamedResponseEvent` |

**崩溃恢复双机制**（理解 nanobot 健壮性的关键）：

1. **runtime checkpoint**（[loop.py#L2142-L2265](file:///d:/project/nanobot/nanobot/agent/loop.py#L2142-L2265)）：runner 在每个 phase 调 `checkpoint_callback`，把 `assistant_message` + `completed_tool_results` + `pending_tool_calls` 存进 `session.metadata`。`_restore_runtime_checkpoint` 在下一轮把它物化进 history，并通过消息键比对去重 overlap。
2. **pending user turn**（[loop.py#L2267-L2286](file:///d:/project/nanobot/nanobot/agent/loop.py#L2267-L2286)）：用户消息已落盘但 assistant 回复未生成就崩溃 → 补一条"Task interrupted"占位，避免对话断裂。

### 1.6 `_build_turn` 与 provider_state

[loop.py#L1741-L1864](file:///d:/project/nanobot/nanobot/agent/loop.py#L1741-L1864) 中有一个值得注意的机制：**provider conversation state（连续对话状态）**。

```python
if stored_state is not None and runtime.provider.can_resume_conversation_state(stored_state, runtime.model):
    current_provider_message = self.context.build_current_message(...)
    ctx.provider_state = stored_state.with_pending_messages([
        *stored_state.pending_messages, current_provider_message,
    ])
```

即：某些 provider（如 OpenAI Responses API）把多轮工具调用存为"待处理消息栈"，`build` 阶段把当前用户消息 append 上去恢复，而不是从 history 重放。这是对"同一会话多轮工具调用"的 provider 级优化。

## 2. AgentRunner 内部结构总览（runner.py）

### 2.1 三个数据类

- `AgentRunSpec`（[runner.py#L90-L117](file:///d:/project/nanobot/nanobot/agent/runner.py#L90-L117)）：一次执行的**全部配置**——`initial_messages`、`tools`、`runtime`、`max_iterations`、`max_tool_result_chars`、`hook`、`concurrent_tools`、`workspace`、各种 callback（progress/retry_wait/checkpoint/**injection**/goal）。
- `AgentRunResult`（[runner.py#L120-L134](file:///d:/project/nanobot/nanobot/agent/runner.py#L120-L134)）：产出——`final_content`、`messages`（完整消息史，供 loop 落盘）、`tools_used`、`usage`、`stop_reason`、`tool_events`、`had_injections`、`provider_state`。
- 关键常量（[runner.py#L73-L76](file:///d:/project/nanobot/nanobot/agent/runner.py#L73-L76)）：

```python
_MAX_EMPTY_RETRIES = 2        # 空回复最多重试 2 次
_MAX_LENGTH_RECOVERIES = 3    # 截断最多续写 3 段
_MAX_INJECTIONS_PER_TURN = 3  # 每轮最多注入 3 条消息
_MAX_INJECTION_CYCLES = 5     # 全轮最多 5 个注入循环
```

### 2.2 `run()`：hook 生命周期外壳

[runner.py#L371-L417](file:///d:/project/nanobot/nanobot/agent/runner.py#L371-L417)：

```python
async def run(self, spec):
    await hook.before_run(context)
    try:
        result = await self._run_core(spec, hook, messages)
    except CancelledError:
        ...
    except Exception as exc:
        await hook.on_error(context)
        raise
    else:
        if context.error: await hook.on_error(context)
        await hook.after_run(context)
        return result
    finally:
        await hook.on_finally(context)
```

Hook 生命周期：`before_run → (每轮 before_iteration/after_iteration) → on_error? → after_run → on_finally`。流式输出、进度、reasoning 全通过 hook 分发。

### 2.3 `_run_core()`：主迭代循环（全文核心）

[runner.py#L419-L873](file:///d:/project/nanobot/nanobot/agent/runner.py#L419-L873)：

```text
for iteration in range(spec.max_iterations):
    messages_for_model = context_governor.prepare_for_model(...)   # 上下文治理（截断/修复）
    response = await self._request_model(...)                        # ① 调模型
    conversation_state.observe_response(...)
    extract_reasoning(...)  # 抽出 thinking 文本

    if response.should_execute_tools:          # ② 模型要调工具
        messages.append(assistant_message with tool_calls)
        checkpoint("awaiting_tools")
        results = await self._execute_tools(...)
        messages.append(tool 结果消息)
        checkpoint("tools_completed")
        drain_injections(...)                   # ③ 注入缓冲的新消息
        continue                                # → 回到 ① 再问模型

    # 无工具 → 处理最终回答
    if finish_reason == "length":               # ④ 输出截断 → 续写
        messages.append(assistant) + messages.append(length_recovery)
        continue
    if finish_reason == "error":                # ⑤ 模型报错
        final_content = error 或欠费提示; stop_reason="error"; break
    messages.append(final assistant_message)
    checkpoint("final_response")
    break
else:  # max_iterations 耗尽
    stop_reason = "max_iterations"
    final_content = _try_finalize_after_max_iterations(...)  # 无工具收尾
```

**状态机**：`awaiting_tools → tools_completed → final_response` 三个 checkpoint phase 用于崩溃恢复；`stop_reason` 有 `completed / tool_error / error / max_iterations / cancelled / empty_final_response`。

### 2.4 `_request_model()`：三种请求模式

[runner.py#L895-L1102](file:///d:/project/nanobot/nanobot/agent/runner.py#L895-L1102)：

按 hook 是否要流式，选三条路径：

| 模式 | 条件 | 调用 | 回调 |
|---|---|---|---|
| 全流式 | `hook.wants_streaming()` | `provider.chat_stream_with_retry` | `on_content_delta` / `on_thinking_delta`（增量 reasoning） |
| 进度流 | 非流式但有 progress | 同上 | `IncrementalThinkExtractor` + `on_content_delta` |
| 非流式 | 默认 | `provider.chat_with_retry` | 无 |

超时控制（[runner.py#L906-L917](file:///d:/project/nanobot/nanobot/agent/runner.py#L906-L917)）：默认 `NANOBOT_LLM_TIMEOUT_S=300`；流式请求用 `max(300, timeout*2)` 作为墙钟超时，`asyncio.wait_for` 兜底。

**坏工具调用清洗**（[runner.py#L1104-L1140](file:///d:/project/nanobot/nanobot/agent/runner.py#L1104-L1140)）：`_drop_malformed_tool_calls` 剥离 name 缺失的 tool call——如果不删，它会进 assistant 消息并被每次重放，导致 provider 校验永久失败"wedge"住 session。全删则用 `_malformed_tool_call_retry_messages` 重试一次，再不行降级为 no-tools 请求。

### 2.5 `_execute_tools()` 与 `_run_tool()`：工具执行

批量与并发（[runner.py#L1647-L1670](file:///d:/project/nanobot/nanobot/agent/runner.py#L1647-L1670)）：`_partition_tool_batches` 把 `concurrency_safe=True` 的连续工具合并成一批，`asyncio.gather` 并发执行；不安全的工具单独串行。

`_run_tool`（[runner.py#L1410-L1523](file:///d:/project/nanobot/nanobot/agent/runner.py#L1410-L1523)）完整流程：

```python
# 1. 重复外部查询限流（同一工具+参数最多 N 次）
lookup_error = repeated_external_lookup_error(...)
# 2. prepare_call：解析/校验参数（ToolRegistry 职责）
tool, params, prep_error = prepare_call(name, arguments)
# 3. hook.before_execute_tool → tool.execute(**params)
# 4. 异常/错误结果 → _classify_violation 分类
```

### 2.6 安全边界分类：SSRF 与 workspace 违规

[runner.py#L1525-L1615](file:///d:/project/nanobot/nanobot/agent/runner.py#L1525-L1615)：

```python
_SSRF_MARKERS = ("internal/private url detected", "private/internal address", ...)
_WORKSPACE_VIOLATION_MARKERS = ("outside the configured workspace", "path traversal detected", ...)
```

`_classify_violation` 做三件事：
1. **SSRF 命中** → 返回带 `_SSRF_BOUNDARY_NOTE` 的**不可重试**错误（明确告诉模型"别用 curl/编码 IP 绕过"），并提示 `tools.ssrfWhitelist`。
2. **workspace 违规** → 首次软返回（`soft_payload` 让模型换方法），重复则 `repeated_workspace_violation_error` 升级提示。
3. 都不命中 → 返回 `None` 走常规错误路径。

**设计思想**：安全不是让 agent 崩溃，而是**让 agent 在对话层面恢复**——错误信息本身是给 LLM 的行为指导。

### 2.7 Injection：回合中注入机制（理解 nanobot 复杂度的钥匙）

[runner.py#L240-L369](file:///d:/project/nanobot/nanobot/agent/runner.py#L240-L369)：

```text
用户 turn 进行中，又来新消息
  → loop 把它塞进该 session 的 pending_queue（不新起任务）
  → runner 在三个时机调 injection_callback = _drain_pending：
      ① after tool execution
      ② after final response（最终回复前）
      ③ after error / max_iterations
  → 拉到的消息转成 user 消息注入 messages，循环继续
```

`_try_drain_injections`（[runner.py#L240-L299](file:///d:/project/nanobot/nanobot/agent/runner.py#L240-L299)）返回值 `(should_continue, cycles)`，配合 `_MAX_INJECTION_CYCLES=5` 防止无限循环。还有 **sustained goal 注入**（[runner.py#L301-L309](file:///d:/project/nanobot/nanobot/agent/runner.py#L301-L309)）：`goal_active_predicate` 为真时注入"继续执行目标"的提示。

`_append_injected_messages`（[runner.py#L162-L238](file:///d:/project/nanobot/nanobot/agent/runner.py#L162-L238)）：注入时做 **role 交替合并**——两条连续 user 消息会合并 content 与 runtime context blocks，避免违反 OpenAI 的 role 交替约束。

### 2.8 输出恢复三件套

| 场景 | 机制 | 位置 |
|---|---|---|
| 空回复 | 重试 ≤2 次，仍空 → `_finalization_retry_messages` 无工具重发 | [runner.py#L632-L673](file:///d:/project/nanobot/nanobot/agent/runner.py#L632-L673) |
| 长度截断（length） | 续写 ≤3 段，`length_recovery_parts` 拼接，`build_length_recovery_message` | [runner.py#L675-L700](file:///d:/project/nanobot/nanobot/agent/runner.py#L675-L700) |
| 迭代耗尽 | `_try_finalize_after_max_iterations` 无工具收尾，失败回退模板 | [runner.py#L1195-L1278](file:///d:/project/nanobot/nanobot/agent/runner.py#L1195-L1278) |

## 3. 一条完整消息的生命周期（串联两个类）

```text
用户发消息
  → bus.publish_inbound(msg)
  → AgentLoop.run() 消费
  → _dispatch(): 拿 session 锁 + 并发信号量
  → _process_message() 七阶段：
      restore（恢复 session / checkpoint）
      compact（检查压缩）
      command（不是命令则跳过）
      build（ContextBuilder 组装 initial_messages + provider_state 恢复）
      run → _run_agent_loop → AgentRunner.run(spec)
        └→ _run_core 迭代：模型 ↔ 工具 循环，期间 checkpoint 落盘
            └→ 注入 pending_queue 的新消息，继续循环
      save（_save_turn 清洗后写入 session.messages）
      respond（组装 OutboundMessage）
  → delivery.complete() → bus.publish_outbound
  → ChannelManager._dispatch_outbound 投递到对应 channel
```

## 4. 阅读建议（按代码顺序）

1. **先读 `_process_message`** 的七阶段（loop.py#L1556-L1564），建立管线心智。
2. **再读 `run()` 主循环**（loop.py#L1172），理解消息怎么进来、pending queue 怎么工作。
3. **然后读 `AgentRunner._run_core`**（runner.py#L419），这是唯一真正的"模型循环"。
4. **接着读 `_request_model` + `_execute_tools`**，理解流式三模式与工具批处理。
5. **最后读 injection 和 checkpoint**，理解 nanobot 最复杂的两块健壮性设计。

## 5. 常见坑点（源码里的教训）

1. **不要小看 pending queue**：mid-turn 消息若不进队列而直接建任务，会和当前任务竞争 session 锁导致乱序——queue + injection 是刻意设计。
2. **checkpoint 三阶段不能乱**：`awaiting_tools` 时 pending_tool_calls 非空，恢复时会给它们补"Task interrupted"错误消息。
3. **malformed tool call 必须 drop**：带 name=None 的调用会污染 assistant 历史，每次重放都让 provider 报 `tool_use.name` 校验错。
4. **injection 有上限**：`_MAX_INJECTIONS_PER_TURN=3`、`_MAX_INJECTION_CYCLES=5`，超出即丢（记 warning），防死循环。
5. **role 交替**：注入消息与上一条同 role 时合并 content，不合并会触发 provider 400。

## Related

- [nanobot ContextBuilder 源码精读](./nanobot-contextbuilder.md) — 七阶段中 build 阶段调用的类
- [nanobot Tool Registry 源码精读](./nanobot-tool-registry.md) — runner 执行工具时用的注册表
- [nanobot Providers Registry 源码精读](./nanobot-providers-registry.md) — runtime 如何选 provider
- [nanobot Channel Manager 源码精读](./nanobot-channel-manager.md) — outbound 如何投递
- [nanobot 源码阅读指南](./nanobot-source-reading-guide.md)
- [AI 索引](./index.md)
