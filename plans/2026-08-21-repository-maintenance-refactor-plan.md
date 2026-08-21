---
title: 仓库维护重构计划
type: plan
status: approved
created: 2026-08-21
updated: 2026-08-21
scope: repository-wide maintenance follow-up
---

# 仓库维护重构计划

## 范围与安全边界

- 审计基线：干净提交 `49b12b9`（`docs: reorganize knowledge and add refactor skill`）。
- 没有需要保护的既有未提交改动。
- 此前的 AI、Inbox 与 Learning 重组已经提交；本计划不重新讨论其目录设计。
- 本轮不包含内容移动、删除、占位笔记、sidebar 分类调整、大范围 frontmatter 迁移、长笔记重写、CI 新增、提交或推送。
- 审计期间曾尝试运行 `npm run docs:build`，但被本地工具安全分类器的临时故障阻断。这是环境限制，不代表构建结果。

## 已确认的问题

### 1. PyTorch 源码链接只能在本机使用

以下五篇笔记中共有 18 个不适合发布的 `file:///d:/project/pytorch-2.8.0/...` 超链接：

- `docs/knowledge/pytorch/torch-compile-wiki/06-graph-break.md`
- `docs/knowledge/pytorch/torch-compile-wiki/07-aotautograd.md`
- `docs/knowledge/pytorch/torch-compile-wiki/08-partition-strategy.md`
- `docs/knowledge/pytorch/torch-compile-wiki/09-torchinductor-backend.md`
- `docs/knowledge/pytorch/torch-compile-wiki/10-lowering-fx-to-ir.md`

**影响：**这些链接仅在原始机器上可用，但对读者来说看起来像可访问的网站链接。

**方案：**将每个超链接替换为可读的源码路径和行号文本，同时保留 frontmatter 中本地源码快照的来源信息。

### 2. PyTorch 链接将 Markdown 文件当作目录

`docs/knowledge/pytorch/index.md` 和 13 篇关联的 PyTorch 笔记至少有 70 个类似 `./pytorch-overview/` 的链接，但真实目标是 `pytorch-overview.md` 文件。

**影响：**这不符合仓库“链接必须指向真实 Markdown 文件”的规则。VitePress 当前可能会宽松解析，但该行为含糊且不可移植。

**方案：**将受影响的 `./pytorch-*/` 链接统一改为明确的 `.md` 目标；不移动或重命名笔记。

### 3. KKT 条件链接无效

`docs/knowledge/mathematics/lagrangian-and-constrained-optimization.md` 链接到不存在的 `./kkt-conditions.md`。全局 VitePress 死链忽略配置掩盖了这个问题。

**影响：**它违反了“只使用真实链接”的规则，并为读者提供了断裂路径。

**方案：**将链接改为纯文本待办项；不创建没有内容的 KKT 占位笔记；本轮不修改死链配置。

### 4. 迭代路线图将过时盘点呈现为当前状态

`docs/projects/second-brain-iteration-roadmap.md` 仍写着 Inbox 有八篇 seed 笔记、AI 只有 GRPO/PPO、Mathematics 与 Autonomous Driving 为空。这些描述已不符合仓库现状。

**影响：**读者会得到错误的知识覆盖面和下一步优先级。

**方案：**保留 2026-08-12 的盘点作为历史基线；增加当前状态；将优先级更新为：消化 Inbox 研究地图、深化 Mathematics / Autonomous Driving、实现链接健康度检查、成熟现有 seed 笔记；并刷新 `updated` 日期。

### 5. 公共入口页暗示 TRAE 是唯一助手

- `docs/index.md` 写着“TRAE is the assistant”。
- `docs/inbox/index.md` 建议读者让 TRAE 整理 Inbox。
- `docs/projects/index.md` 将工作流描述为与 TRAE 协作。

**影响：**这与仓库的工具中立工作流指引不一致。

**方案：**在这些入口页使用“AI 助手”；保留详细工作流笔记中的历史性或工具专属 TRAE 说明，因为该笔记已经解释了其更广泛的适用性。

## 精确实施边界

仅编辑：

1. 上述五篇 `torch-compile-wiki` 笔记，移除 18 个本地文件系统超链接。
2. `docs/knowledge/pytorch/index.md` 与 13 篇包含目录式 Markdown 文件链接的 PyTorch 笔记。
3. `docs/knowledge/mathematics/lagrangian-and-constrained-optimization.md`，移除无效的 KKT 超链接。
4. `docs/projects/second-brain-iteration-roadmap.md`，标注历史盘点、补充当前状态并刷新优先级。
5. `docs/index.md`、`docs/inbox/index.md` 与 `docs/projects/index.md`，改为工具中立的助手表述。
6. `prompts/repository-refactor.md`，要求后续运行将计划归档至 `plans/`。
7. 本计划归档：`plans/2026-08-21-repository-maintenance-refactor-plan.md`。

## 延期工作

- 在单独的导览优化中，更清楚地区分基于 PyTorch 2.8.0 的 `torch.compile` Wiki 笔记和基于 PyTorch 2.9.0a0 的其他 PyTorch 笔记。
- 对比大型 Engineering 分布式存储知识地图与较新的分布式存储专题笔记，再决定是否拆分、重构或迁移。
- 保留 seed 笔记中有意留空的“个人理解 / 实践”部分；不虚构个人经验。
- 将 index、sidebar、死链和孤立笔记检查设计为独立脚本或 CI 功能。

## 验证计划

1. 在修改过的 PyTorch 笔记中搜索 `file:///`（预期：0）。
2. 在修改过的 PyTorch 范围中搜索 `./pytorch-*/` 目标（预期：0）。
3. 运行 Markdown 相对链接扫描，并区分真实失败和代码示例造成的误报。
4. 确认 VitePress sidebar 的目标都存在。
5. 运行 `git diff --check`。
6. 重试 `npm run docs:build`；如被环境阻断，如实报告。

## 批准记录

用户于 2026-08-21 批准本计划，并要求今后的 `repository-refactor` 运行将生成的重构计划归档到 `plans/`。

## 实施结果

- 已完成上述有界编辑；未移动、删除笔记，也未创建占位笔记。
- 已将 18 个本地 PyTorch 源码超链接替换为可读的仓库路径，并将批准范围内的 PyTorch 文件链接规范为明确的 `.md` 目标。
- 已将不存在的 KKT 页面链接改为待办说明；已刷新路线图和公共页面的助手表述，并更新实质修改笔记的 `updated` 字段。
- 已在 `prompts/repository-refactor.md` 中加入未来计划归档要求。
- 修改后的定向搜索显示：五篇变更的 `torch-compile-wiki` 笔记中没有 `file:///` 链接；批准范围的 PyTorch 笔记中没有目录式 `./pytorch-*/` 链接。
- `npm run docs:build` 因本地工具安全分类器暂时不可用而未能执行；这既不是构建通过，也不是构建失败。完整相对链接扫描与最终的 `git diff --check` 同样被该环境问题阻断，仍需在环境恢复后重跑。
