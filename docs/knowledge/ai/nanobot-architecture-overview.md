---
title: nanobot 核心架构总览（源码视角）
type: concept
status: seed
tags: [AI, Agents, Architecture, Nanobot, Runtime, WebUI, Gateway]
created: 2026-08-17
updated: 2026-08-17
source:
  - d:\project\nanobot\nanobot\nanobot.py
  - d:\project\nanobot\nanobot\agent\loop.py
  - d:\project\nanobot\nanobot\agent\runner.py
---

# nanobot 核心架构总览（源码视角）

> 本文基于本地源码 `d:\project\nanobot\nanobot\` 目录树，画出一条主线：**消息从哪进、经过哪些层、最后怎么回来**。各层的细节见对应的"源码精读"笔记。

## 一句话理解

nanobot 是一个**自托管的个人 AI Agent Runtime**：外部消息（CLI/WebUI/Telegram/Discord/Feishu/...）统一进入 `MessageBus` → `AgentLoop`（turn 编排）→ `AgentRunner`（模型+工具循环），结果再经 `ChannelManager` 投回外部世界。

## 1. 源码目录地图（nanobot/ 顶层）

```text
nanobot/
├── nanobot.py            ← Python SDK facade（Nanobot 类）
├── __main__.py           ← python -m nanobot 入口
├── cli/                  ← CLI 命令（commands.py: onboard/agent/gateway/webui/trigger/...）
├── gateway/              ← 网关模式（远程设备/WebUI 接入）
├── agent/                ← ★ 核心：loop / runner / context / memory / skills / subagent
│   └── tools/            ← ★ 工具：registry / base / filesystem / shell / web / mcp / ...
├── providers/            ← LLM 后端：registry（数据表）/ factory / base / 各实现
├── channels/             ← 渠道：manager + 每个平台一个子包（feishu/telegram/qq/...）
├── session/              ← session 持久化、model_selection、turn_continuation
├── bus/                  ← MessageBus + outbound 事件类型
├── config/               ← schema / loader / paths / watcher
├── command/              ← 内置命令（/new /stop 等）
├── cron/  triggers/      ← 定时任务 / 本地触发器
├── security/             ← workspace 访问控制、网络策略
├── webui/                ← WebUI 后端 API（静态资源、session、settings、MCP 等）
├── api/                  ← OpenAI 兼容 API server
├── apps/                 ← CLI app 协议
├── sdk/                  ← Python SDK
└── utils/                ← 工具函数（llm_runtime / prompt_templates / helpers）
```

## 2. 分层架构

```text
┌──────────────────────────────────────────────────────────┐
│ 入口层（多种，殊途同归）                                    │
│  CLI / WebUI(websocket) / Gateway / OpenAI API / Python   │
│  SDK / 聊天渠道(Telegram、Feishu、QQ、Slack、Email...)     │
└──────────────┬───────────────────────────────────────────┘
               ▼ publish_inbound
┌──────────────────────────────────────────────────────────┐
│ 调度层：AgentLoop（agent/loop.py）                        │
│  run() 主循环 → _dispatch() → _process_message()          │
│  七阶段：restore → compact → command → build → run →      │
│          save → respond                                  │
└──────────────┬───────────────────────────────────────────┘
               ▼ build 阶段产出 initial_messages
┌──────────────────────────────────────────────────────────┐
│ 执行层：AgentRunner（agent/runner.py）                    │
│  _run_core 迭代循环：                                     │
│    模型请求(_request_model) ↔ 工具执行(_execute_tools)     │
│    注入(injection) / 恢复(empty/length/max_iterations)    │
└──────────────┬───────────────────────────────────────────┘
               ▼ 关联组件
┌──────────────────────────────────────────────────────────┐
│ 支撑层                                                    │
│  ContextBuilder（prompt 组装）  SessionManager（会话）     │
│  ToolRegistry（工具）           Providers（模型后端）      │
│  Memory/Consolidator/Dream      Security（权限边界）      │
└──────────────┬───────────────────────────────────────────┘
               ▼ publish_outbound
┌──────────────────────────────────────────────────────────┐
│ 投递层：ChannelManager（channels/manager.py）             │
│  _dispatch_outbound → 事件分类 → 合并 delta → 去重 →       │
│  重试发送到对应 channel                                    │
└──────────────────────────────────────────────────────────┘
```

## 3. 五大核心概念

### 3.1 三"注册中心"（数据驱动）

| 注册中心 | 文件 | 管什么 |
|---|---|---|
| Provider | `providers/registry.py` | LLM 后端的元数据表（识别/路由/参数方言） |
| Tool | `agent/tools/registry.py` | 工具 schema + 参数解析校验 + 执行 |
| Channel 插件 | `channels/registry.py` | 渠道插件发现、延迟加载、多实例 |

三者都是**"注册表 + 工厂 + 基类"**模式：新增能力只需加一条数据/一个插件，核心循环不变。

### 3.2 MessageBus：松耦合的核心

`bus/queue.py` 的 `MessageBus` 是**进出站两个 asyncio.Queue**：

```python
publish_inbound(msg)   → AgentLoop.run() 消费
publish_outbound(msg)  → ChannelManager._dispatch_outbound() 消费
```

Channel 只往 bus 发 inbound、AgentLoop 只消费 inbound 发 outbound、ChannelManager 只消费 outbound——三者互不直接调用，这就是"多入口共享同一核心"的结构基础。

### 3.3 AgentLoop 的七阶段管线

`_process_message()`（[loop.py#L1446-L1564](file:///d:/project/nanobot/nanobot/agent/loop.py#L1446-L1564)）是 nanobot 对"一轮对话"的完整定义：

```text
restore（恢复 session/checkpoint）
  → compact（检查压缩）
  → command（内置命令捷径）
  → build（组装 prompt + provider_state）
  → run（AgentRunner 迭代）
  → save（落盘，清洗）
  → respond（组 outbound）
```

### 3.4 AgentRunner 的迭代循环

`_run_core()`（[runner.py#L419-L873](file:///d:/project/nanobot/nanobot/agent/runner.py#L419-L873)）是唯一真正的"模型循环"：

```text
请求模型 → 有 tool call？→ 执行工具 → 结果回写 → 再请求模型
         → 无工具 → 处理 final（截断续写/空回复重试/错误）
         → 期间注入 pending 新消息、checkpoint 落盘
```

### 3.5 TurnDelivery：入口无关的投递抽象

`agent/turn_delivery.py` 的 `TurnDelivery` 把"如何把进度/流式/最终结果发给用户"从 loop 中解耦——CLI 打印、WebUI WebSocket、聊天渠道各自实现自己的 delivery，但核心循环完全一样。

## 4. 状态系统三层

| 层 | 载体 | 内容 | 生命周期 |
|---|---|---|---|
| Session | `session/manager.py` + 磁盘 | 短期对话历史 | 按 session_key 隔离，可 TTL 过期 |
| Memory | `agent/memory.py` | 长期记忆（MEMORY.md） | 跨 session 共享 |
| Dream | `agent/memory.py` Consolidator | 记忆整理/压缩 | 后台异步 |

## 5. 安全边界

- `security/workspace_access.py`：workspace scope 解析与绑定（contextvars）
- `security/workspace_policy.py`：工具可访问路径白名单
- `security/network.py`：SSRF 防护
- runner 的错误分类（SSRF 硬边界 / workspace 违规软恢复）

## 6. 推荐阅读顺序（从主线到分支）

1. 本文（总图）
2. [AgentLoop 与 AgentRunner 源码精读](./nanobot-agentloop-runner.md)（主线）
3. [ContextBuilder 源码精读](./nanobot-contextbuilder.md)（build 阶段）
4. [Tool Registry 源码精读](./nanobot-tool-registry.md)（工具）
5. [Providers Registry 源码精读](./nanobot-providers-registry.md)（模型）
6. [Channel Manager 源码精读](./nanobot-channel-manager.md)（投递）
7. 其余：session / memory / gateway / webui / security

## Related

- [nanobot 源码阅读指南](./nanobot-source-reading-guide.md)
- [nanobot AgentLoop 与 AgentRunner 源码精读](./nanobot-agentloop-runner.md)
- [nanobot ContextBuilder 源码精读](./nanobot-contextbuilder.md)
- [nanobot Tool Registry 源码精读](./nanobot-tool-registry.md)
- [nanobot Providers Registry 源码精读](./nanobot-providers-registry.md)
- [nanobot Channel Manager 源码精读](./nanobot-channel-manager.md)
- [AI 开源项目源码精读指南](./ai-open-source-source-reading.md)
- [AI 索引](./index.md)
