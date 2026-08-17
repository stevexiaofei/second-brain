---
title: nanobot Channel Manager 源码精读
type: concept
status: seed
tags: [AI, Channels, WebUI, Chat Apps, Gateway, Nanobot, Message Routing, Source Code]
created: 2026-08-17
updated: 2026-08-17
source:
  - d:\project\nanobot\nanobot\channels\manager.py
---

# nanobot Channel Manager 源码精读

> 本文基于本地源码 `d:\project\nanobot\nanobot\channels\manager.py`（1017 行）拆解。目标：看懂 Telegram/Discord/Slack/Feishu/WebSocket 等渠道如何被初始化、启动、路由消息、投递回复。

## 一句话理解

`ChannelManager` 是 nanobot 的**外部接入层调度器**：启动时扫描并实例化启用的渠道，运行时有一个 outbound 分发循环把 `OutboundMessage` 按 `msg.channel` 路由给对应渠道实例发送。

## 1. 核心数据与常量

[manager.py#L59-L67](file:///d:/project/nanobot/nanobot/channels/manager.py#L59-L67)：

```python
_SEND_RETRY_DELAYS = (1, 2, 4)   # 指数退避：1s, 2s, 4s
_BOOL_CAMEL_ALIASES = {"send_progress": "sendProgress", ...}  # JSON config 兼容 camelCase
```

实例状态字典（[manager.py#L129-L136](file:///d:/project/nanobot/nanobot/channels/manager.py#L129-L136)）：

| 字段 | 作用 |
|---|---|
| `self.channels: dict[str, BaseChannel]` | runtime_name → channel 实例 |
| `self._channel_owners: dict[str, str]` | runtime_name → 插件名（多实例归属） |
| `self._channel_runtime_specs` | runtime_name → (owner, instance_id) |
| `self._channel_errors` | 启动失败的错误信息（供 WebUI 状态展示） |
| `self._channel_tasks` | 每个 channel 的 asyncio.Task |

**关键概念：runtime_name vs 插件名**。一个插件（如 QQ）可以配置多个实例，每个实例有一个唯一 `runtime_name`（如 `qq:main`、`qq:backup`），消息路由按 runtime_name 走。

## 2. 初始化：`_init_channels()`（插件扫描）

[manager.py#L221-L298](file:///d:/project/nanobot/nanobot/channels/manager.py#L221-L298)：

```python
def _init_channels(self):
    plugins = discover_plugins()      # 1. 发现所有渠道插件
    for name, plugin in plugins.items():
        section = self._channel_section(name, ...)   # 2. 取该渠道的 config 段
        if section is None: continue
        channel_setup_spec(name, plugin=plugin)      # 3. 检查依赖（缺失则标错）
        specs = channel_instance_specs(plugin, section)  # 4. 解析出多个实例
        # 5. runtime_name 冲突检查
        # 6. 依赖检查 ensure_enabled_channel_dependencies
        # 7. plugin.load_channel_class() → _build_channel() 实例化
    self._validate_allow_from()   # 8. 校验 allowFrom（无则进入 pairing-only 模式）
```

**Channel 插件机制**：每个渠道在 `nanobot/channels/<name>/` 下有 `manifest.py`（插件描述）、`runtime.py`（`BaseChannel` 子类）、可选 `validation.py`。`discover_plugins` 从这些"无依赖描述符"发现插件，真正的类按需加载——**延迟导入保证未安装依赖的渠道不会拖垮启动**。

### `_build_channel()` 的特例：websocket

[manager.py#L162-L219](file:///d:/project/nanobot/nanobot/channels/manager.py#L162-L219)：websocket 渠道特殊，它要 `build_gateway_services` 构建整套 WebUI 网关服务（静态资源、session API、cron API、MCP 状态等）再作为 channel 注入——**WebUI 本质上是一个渠道**。

## 3. 生命周期：start / stop

### `start_all()`

[manager.py#L577-L595](file:///d:/project/nanobot/nanobot/channels/manager.py#L577-L595)：

```python
async def start_all(self):
    self._started = True
    self._dispatch_task = asyncio.create_task(self._dispatch_outbound())  # 启动分发循环
    tasks = [self._start_channel_task(name, ch) for name, ch in self.channels.items()]
    self._notify_restart_done_if_needed()
    await asyncio.gather(*tasks, return_exceptions=True)  # channel 常驻，正常永不返回
```

每个 channel 的 `start()` 内部通常是一个长循环（监听平台消息）。`_start_channel` 捕获异常并写入 `_channel_errors`（[manager.py#L360-L376](file:///d:/project/nanobot/nanobot/channels/manager.py#L360-L376)）。

### `stop_all()` / 热开关

`stop_all`（[manager.py#L641-L654](file:///d:/project/nanobot/nanobot/channels/manager.py#L641-L654)）取消 dispatcher + 逐个停 channel。`apply_channel_feature_action`（[manager.py#L408-L575](file:///d:/project/nanobot/nanobot/channels/manager.py#L408-L575)）支持 **WebUI 里不重启启停渠道**：disable 走 `_stop_channel` + 移除；enable 重新构建实例并 `_start_channel_task`。

## 4. Outbound 分发循环（核心）

### `_dispatch_outbound()`

[manager.py#L683-L766](file:///d:/project/nanobot/nanobot/channels/manager.py#L683-L766)：

```python
while True:
    msg = await asyncio.wait_for(self.bus.consume_outbound(), timeout=1.0)
    event = outbound_event_from_message(msg)   # 解出事件类型

    # 1. reasoning 事件：只有 show_reasoning=True 的 channel 才收
    # 2. ProgressEvent：按 send_progress / send_tool_hints 开关过滤
    # 3. RetryWaitEvent：直接丢弃（只用于计时提示）
    # 4. StreamDeltaEvent → _coalesce_stream_deltas 合并（下面细说）
    channel = self.channels.get(msg.channel)
    if channel:
        # 去重：同源消息内容指纹一致则抑制
        if not isinstance(event, StreamDeltaEvent|StreamEndEvent|StreamedResponseEvent):
            if self._should_suppress_outbound(msg): continue
        await self._send_with_retry(channel, msg)
```

**事件类型**（`outbound_events.py`）：`StreamDeltaEvent`（流式增量）、`StreamEndEvent`、`StreamedResponseEvent`（最终响应）、`ProgressEvent`（进度/reasoning/tool hint）、`RetryWaitEvent`。

### 流式 delta 合并（降低 API 调用）

`_coalesce_stream_deltas`（[manager.py#L849-L912](file:///d:/project/nanobot/nanobot/channels/manager.py#L849-L912)）：

```python
# 把队列里连续的同 (channel, chat_id, stream_id) 的 delta 拼成一条
# 直到遇到不同 stream 的消息或 stream_end 为止
# 不匹配的消息放进 pending 缓冲，下次循环先处理（模拟 push_front）
```

**为什么要合并**：LLM 生成速度 > 渠道发送速度时，队列会积压大量 delta；逐条发 API 又贵又慢。合并后一次调用发一整段。队列积压的消息用本地 `pending` 列表模拟"插回队头"。

### 发送重试

`_send_with_retry`（[manager.py#L914-L972](file:///d:/project/nanobot/nanobot/channels/manager.py#L914-L972)）：

```python
while True:
    try:
        await self._send_once(channel, msg)
        return
    except CancelledError: raise
    except Exception as e:
        if not channel.should_retry_send_error(e): return  # 非可重试错误直接放弃
        delay = _SEND_RETRY_DELAYS[min(attempt-1, 2)]       # 1/2/4s 退避
        await asyncio.sleep(delay)
```

**设计**：`should_retry_send_error` 由渠道决定什么错可重试（如 429/网络抖动），什么错不可重试（如 403 权限），避免无限重试。

### 去重抑制

`_should_suppress_outbound`（[manager.py#L661-L681](file:///d:/project/nanobot/nanobot/channels/manager.py#L661-L681)）：对回复的**原文消息**（`origin_message_id`）算内容 SHA1 指纹，同指纹不重复发——防止模型在同一会话里把同一段话发两次。

## 5. `_send_once`：事件 → 渠道方法分发

[manager.py#L824-L847](file:///d:/project/nanobot/nanobot/channels/manager.py#L824-L847)：

```python
event = outbound_event_from_message(msg)
if isinstance(event, ProgressEvent) and event.reasoning_end:  → channel.send_reasoning_end(...)
elif ... reasoning_delta:                                      → channel.send_reasoning_delta(...)
elif ... reasoning:                                            → channel.send_reasoning(...)
elif ... file_edit_events:                                     → channel.send_file_edit_events(...)
elif isinstance(event, StreamDeltaEvent):                      → channel.send_delta(stream_id=..., stream_end=False)
elif isinstance(event, StreamEndEvent):                        → channel.send_delta(stream_id=..., stream_end=True, resuming=...)
elif not isinstance(event, StreamedResponseEvent):             → channel.send(msg)
```

每个渠道只需实现 `send` / `send_delta` / `send_reasoning` 等几个原语，事件类型统一由 manager 翻译。

## 6. 状态查询：`get_status()`

[manager.py#L978-L1012](file:///d:/project/nanobot/nanobot/channels/manager.py#L978-L1012)：

```python
for runtime_name, (owner, instance_id) in runtime_specs.items():
    running = bool(channel and channel.is_running)
    if error: state = "failed"
    elif running: state = "running"
    elif task is not None and not task.done(): state = "starting"
    else: state = "stopped"
    status[runtime_name] = {"enabled": True, "running": running, "state": state, "owner": owner, "instance_id": instance_id, ...}
```

提供给 WebUI 的渠道状态面板：每个 runtime 的四态机（failed/running/starting/stopped）。

## 7. 一条回复的完整投递链路

```text
AgentLoop._prepare_outbound → OutboundMessage(channel=..., chat_id=...)
  → bus.publish_outbound(msg)
  → ChannelManager._dispatch_outbound 消费
      ├── 事件分类（delta/progress/reasoning/final）
      ├── 开关过滤（send_progress / show_reasoning）
      ├── StreamDeltaEvent 合并
      ├── 去重抑制（内容指纹）
      └── _send_with_retry → _send_once → channel.send / send_delta
  → 平台 API 实际发送
```

## 8. 你该重点看的源码细节

| 问题 | 位置 |
|---|---|
| 插件如何被发现和延迟加载 | `_init_channels` + `discover_plugins`（[manager.py#L221-L298](file:///d:/project/nanobot/nanobot/channels/manager.py#L221-L298)） |
| WebUI 为什么是渠道 | `_build_channel` 的 websocket 分支（[manager.py#L171-L202](file:///d:/project/nanobot/nanobot/channels/manager.py#L171-L202)） |
| delta 合并怎么保证顺序 | `_coalesce_stream_deltas` + pending 缓冲（[manager.py#L849-L912](file:///d:/project/nanobot/nanobot/channels/manager.py#L849-L912)） |
| 什么错误可重试 | `channel.should_retry_send_error`（[manager.py#L938-L946](file:///d:/project/nanobot/nanobot/channels/manager.py#L938-L946)） |
| 渠道状态机 | `get_status`（[manager.py#L978-L1012](file:///d:/project/nanobot/nanobot/channels/manager.py#L978-L1012)） |

## 9. 常见坑点

1. **WebUI 不是独立服务，是 websocket 渠道**：读代码时把 webui 相关功能都在 `_build_channel` 的 websocket 分支里找。
2. **runtime_name 决定路由**：多实例渠道（如 QQ 开两个号）靠 runtime_name 区分，`msg.channel` 里存的就是 runtime_name。
3. **合并 delta 不能跨 stream**：`_coalesce_stream_deltas` 遇到不同 `stream_id` 立即停，不匹配消息走 pending 缓冲保序。
4. **启动失败不阻塞其他渠道**：`_start_channel` 捕获异常写 `_channel_errors`，WebUI 状态面板显示 failed，其他渠道照常跑。

## Related

- [nanobot AgentLoop 与 AgentRunner 源码精读](./nanobot-agentloop-runner.md) — outbound 消息的源头
- [nanobot 源码阅读指南](./nanobot-source-reading-guide.md)
- [AI 索引](./index.md)
