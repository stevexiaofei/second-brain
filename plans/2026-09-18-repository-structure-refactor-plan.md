# 仓库结构重构计划

## 审计背景

本次仓库级审计聚焦于 VitePress 知识导航、站内链接、主题归属与发布兼容性。目标是以最小结构调整修复已确认的问题，不重写知识内容，也不创建占位笔记。

## 受保护的现有 Git 状态

开始时已有以下未提交修改，均不属于本计划，未编辑也未纳入变更：

- `AGENTS.md`
- `docs/knowledge/engineering/mongodb-pymongo-getting-started.md`

## 已确认发现与实施边界

### 1. Attention 笔记的真实失效链接与本机链接

- **证据**：`docs/knowledge/ai/attention-head-variants.md` 的 Related Knowledge 区曾从 `ai/` 目录错误引用 4 个不存在的 `./flash-attention-*.md` 文件；实际目标位于 `ai/systems/flash-attention/`。同一笔记还包含 `file:///Users/.../flash_api.cpp`。
- **实施**：改为真实的相对 Markdown 路径；将本机 URL 改成仓库相对源码位置的文字说明，不伪造公共链接；更新该笔记的 `updated` 日期。

### 2. 分布式存储主题的双入口与重复索引

- **证据**：稳定的分布式存储基础及子笔记位于 `docs/knowledge/distributed-systems/distributed-storage/`，但场景化工程地图原在 `docs/knowledge/engineering/`；Engineering 索引同时重复列出该主题的全部子笔记。
- **实施**：
  - 使用 Git 感知移动，将地图迁至 `docs/knowledge/distributed-systems/distributed-storage/ai-training-and-multicloud-storage-map.md`；不删除内容。
  - 更新全仓业务引用、父级 `index.md`、VitePress Distributed Systems 侧栏与 Engineering 索引，使分布式存储有一个主题入口。
  - 在存储基础总览中明确基础模型与 AI 训练/多云工程场景地图的分工，并补入推荐学习顺序。
  - 对移动笔记及存储总览中已修改的 `flowchart` 补齐共享的 5 个 Mermaid `classDef` 与节点分类。

### 3. 高层可发现性与过期状态快照

- **证据**：`docs/index.md` 未暴露已有的 Distributed Systems、Investing、Learning 知识区域；`docs/projects/second-brain-iteration-roadmap.md` 将 Inbox 描述为仅一条笔记，而 `docs/inbox/index.md` 已列出三条待核验学习地图。
- **实施**：补齐首页三区入口，并将路线图中的当前 Inbox 描述修正为三条地图。

## 不在本次范围内

- 不拆分或重写长笔记。
- 不新增空白知识页。
- 不修改其他未确认的问题、自动化检查/CI 或大范围 frontmatter。
- 不提交、推送或发布。

## 验证步骤

1. 搜索旧的 `engineering/distributed-storage-knowledge-map` 路径，确认仅历史计划保留记录。
2. 确认 Attention 笔记的 4 个 FlashAttention 相对链接均解析到实际文件，并搜索 `file://` 确认该发布阻塞链接已消除。
3. 确认移动后的地图同时出现在其父 `index.md`、Distributed Systems 索引和 VitePress 侧栏，且旧 Engineering 侧栏路由已移除。
4. 检查修改的 Mermaid `flowchart` 含共享 5 个 `classDef` 且节点已分类。
5. 运行 `npm run docs:build`、Markdown 相对链接扫描、`git diff --check`，最后审阅 Git 状态和差异，确认受保护的预存修改未被触及。

## 实施结果

待验证完成后填写。
