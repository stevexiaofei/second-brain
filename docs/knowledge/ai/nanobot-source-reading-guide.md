---
title: nanobot 源码阅读指南（本地版）
type: concept
status: seed
tags: [AI, Agents, Source Code, Architecture, Nanobot, Python, WebUI, MCP]
created: 2026-08-17
updated: 2026-08-17
source:
  - d:\project\nanobot\nanobot\__main__.py
  - d:\project\nanobot\nanobot\nanobot.py
  - d:\project\nanobot\nanobot\agent\loop.py
  - d:\project\nanobot\nanobot\agent\runner.py
---

# nanobot 源码阅读指南（本地版）

> 本地源码位于 `d:\project\nanobot\nanobot\`。本文给出**按真实代码结构设计**的阅读路径，配合各篇"源码精读"笔记使用。

## 一句话理解

nanobot 是一个**自托管的个人 AI Agent Runtime**。读它源码的关键不是"功能多"，而是抓住三条主线：

1. **一条消息的完整生命周期**（inbound → loop → runner → outbound）
2. **三大注册中心**（provider / tool / channel 都是"数据表 + 工厂 + 基类"）
3. **一个迭代执行循环**（runner 的模型 ↔ 工具循环 + 各种恢复机制）

## 第 0 步：先建立总图（5 分钟）

按顺序读这三篇，不要碰源码：

1. [nanobot 核心架构总览](./nanobot-architecture-overview.md) — 目录地图 + 分层图
2. [nanobot AgentLoop 与 AgentRunner 源码精读](./nanobot-agentloop-runner.md) — 主线
3. 官方 [docs/architecture.md](file:///d:/project/nanobot/docs/architecture.md)

回答三个问题：
- 消息从哪进？（bus）
- 核心处理在哪？（loop + runner）
- 结果怎么出来？（channel manager）

## 第 1 步：读入口，验证直觉（10 分钟）

- [`nanobot/__main__.py`](file:///d:/project/nanobot/nanobot/__main__.py)：只是把入口交给 CLI。
- [`nanobot/nanobot.py`](file:///d:/project/nanobot/nanobot/nanobot.py)：`Nanobot` 类，Python SDK facade，`run()` / `run_streamed()` 一次性封装，`from_config` 从配置构建。
- [`nanobot/cli/commands.py`](file:///d:/project/nanobot/nanobot/cli/commands.py)：`onboard` / `agent` / `gateway` / `webui` / `trigger` / `channels_status` 等命令。

## 第 2 步：读核心链路（1-2 小时，最重要）

按这个顺序读，每个文件配对应笔记：

| 顺序 | 文件 | 笔记 | 要点 |
|---|---|---|---|
| 1 | `agent/loop.py`（2358 行） | [AgentLoop 精读](./nanobot-agentloop-runner.md) | `run()` 主循环、`_dispatch`、`_process_message` 七阶段 |
| 2 | `agent/runner.py`（1670 行） | 同上 | `_run_core` 迭代循环、`_request_model`、`_execute_tools`、injection、checkpoint |
| 3 | `agent/context.py`（315 行） | [ContextBuilder 精读](./nanobot-contextbuilder.md) | system prompt 六大 section、图片 base64、role 交替 |
| 4 | `bus/queue.py` + `bus/events.py` | — | `MessageBus` 进出站队列、`Inbound/OutboundMessage` |

**读 loop.py 时的抓手**：先看 `_process_message` 那 7 行阶段调用（L1556-L1564），再往上看 `run()` 主循环，然后逐个阶段展开。

**读 runner.py 时的抓手**：先看 `_run_core` 的 for 循环骨架，把"工具循环 / 最终回答 / 错误处理"三个分支画下来，再看 `_request_model` 和 `_run_tool` 的实现。

## 第 3 步：读三大注册中心（每篇 30 分钟）

| 顺序 | 文件 | 笔记 | 核心问题 |
|---|---|---|---|
| 1 | `providers/registry.py` | [Providers 精读](./nanobot-providers-registry.md) | 元数据表怎么支撑几十个后端？识别/参数方言？ |
| 2 | `agent/tools/registry.py` + `tools/base.py` | [Tool Registry 精读](./nanobot-tool-registry.md) | tool call 怎么被解析校验执行？schema 缓存？ |
| 3 | `channels/manager.py` | [Channel Manager 精读](./nanobot-channel-manager.md) | 渠道怎么发现/启动？outbound 怎么分发/合并/重试？ |

## 第 4 步：读状态与配置

- `session/manager.py`：session 持久化、`get_history` / `add_message`
- `agent/memory.py`：`MemoryStore` / `Consolidator`（Dream）
- `config/schema.py` + `loader.py`：配置结构（camelCase 写回、env 插值）

## 第 5 步：按需深入高级能力

- **流式投递**：`agent/turn_delivery.py`（TurnDelivery 抽象）
- **网关**：`gateway/` + `cli/gateway.py`（远程接入模式）
- **WebUI**：`webui/`（静态 + API）+ `channels/websocket/runtime.py`
- **安全**：`security/`（workspace_access / workspace_policy / network）
- **MCP**：`agent/tools/mcp.py`（工具系统的外部扩展）
- **Cron**：`cron/service.py`（定时任务）
- **Subagent**：`agent/subagent.py`（子 agent 管理）

## 阅读心态：不要被 400+ 文件吓住

nanobot 的实际核心很小：

```text
真正必读（~6 个文件）：
  agent/loop.py        ← turn 编排
  agent/runner.py      ← 模型执行循环
  agent/context.py     ← prompt 组装
  providers/registry.py ← 模型后端元数据
  agent/tools/registry.py ← 工具调度
  channels/manager.py  ← 消息投递

其余全是"扩展点 + 具体实现"：
  channels/<平台>/     ← 每个平台一个适配器
  providers/<实现>.py   ← 每个后端一个实现
  agent/tools/<工具>.py ← 每个工具一个文件
```

**主线固定，扩展面大**——这正是 agent runtime 的标准组织方式。

## 我理解的 nanobot 设计哲学

1. **数据驱动优于代码分支**：provider 参数方言、工具 schema、渠道插件全用数据/注册表描述，核心循环不做 if/else 地狱。
2. **对话式容错优于崩溃**：工具错误、SSRF、workspace 违规全部转成"给模型的提示"，让 agent 在对话里自我修正。
3. **健壮性靠 checkpoint + 恢复**：runtime checkpoint、pending user turn、injection 上限，都是为了"崩溃后不丢上下文、不死循环"。
4. **多入口单核心**：CLI/WebUI/聊天渠道全部收敛到 `AgentLoop`，差异只体现在 `TurnDelivery` 和 channel 适配器。

## Related

- [nanobot 核心架构总览](./nanobot-architecture-overview.md)
- [nanobot AgentLoop 与 AgentRunner 源码精读](./nanobot-agentloop-runner.md)
- [nanobot ContextBuilder 源码精读](./nanobot-contextbuilder.md)
- [nanobot Tool Registry 源码精读](./nanobot-tool-registry.md)
- [nanobot Providers Registry 源码精读](./nanobot-providers-registry.md)
- [nanobot Channel Manager 源码精读](./nanobot-channel-manager.md)
- [AI 开源项目源码精读指南](./ai-open-source-source-reading.md)
- [AI 索引](./index.md)
