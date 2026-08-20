# nanobot 专题

这组笔记从源码视角理解 nanobot 这一自托管 Agent Runtime：消息如何进入系统，模型与工具如何循环，状态如何恢复，结果又如何投递到不同渠道。

## 推荐顺序

1. [nanobot 源码阅读指南](./nanobot-source-reading-guide.md) — 阅读范围、关键文件与主线问题
2. [核心架构总览](./nanobot-architecture-overview.md) — 分层、MessageBus、七阶段管线和状态系统
3. [AgentLoop 与 AgentRunner](./nanobot-agentloop-runner.md) — turn 编排与模型—工具迭代循环
4. [ContextBuilder](./nanobot-contextbuilder.md) — system prompt、图片、角色交替和工具 contract
5. [Tool Registry](./nanobot-tool-registry.md) — 参数解析、校验、执行与结果封装
6. [Providers Registry](./nanobot-providers-registry.md) — 后端识别、路由和参数方言
7. [Channel Manager](./nanobot-channel-manager.md) — 插件发现、outbound 分发、流式合并与重试

## 适用边界

这些笔记基于特定日期的本地源码快照。文中的路径用于定位被分析的源码，不代表 VitePress 站内页面；复用结论前应先核对当前 nanobot 版本。

## Related

- [AI Systems](../)
- [AI 开源项目源码精读指南](../ai-open-source-source-reading.md)
- [源码阅读方法](../../../learning/code-reading/)
