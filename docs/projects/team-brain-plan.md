---
title: Team Brain 计划 — 从个人 Second Brain 到感知算法团队知识库
type: plan
status: seed
tags: [Team-Brain, Knowledge-Management, GitLab, Perception, MLOps]
created: 2026-10-30
updated: 2026-10-30
---

# Team Brain 计划：从个人 Second Brain 到感知算法团队知识库

## 一句话理解

Team Brain = 把现有 second-brain 的**规则层**（AGENTS.md / 模板 / inbox→knowledge 流程 / VitePress）几乎原样迁移到 GitLab，把**内容层**从"我的学习领域"重构为"感知仓库的子系统 + 团队工程经验"，再补上个人版没有的三件新东西：**权限**、**AI 助手的团队化接入**、**与感知仓库的双向链接**。

## 背景

### 现状：second-brain 是什么

当前仓库（VitePress 站点）已验证可复用的资产：

| 资产 | 位置 | 对 Team Brain 的价值 |
|---|---|---|
| AI 行为规则 | `AGENTS.md`（搜索先行、原子笔记、链接不虚构、Mermaid 规范、LaTeX 规范） | **直接复用**，这是最核心的资产 |
| 6 类模板 | `templates/`（concept / paper / experience / idea / book / project） | 直接复用，可能补 1-2 个团队模板 |
| Inbox 流程 | `docs/inbox/` + index 登记 + 消化管线 | 直接复用，规则改为"任何人可写入，定期轮流消化" |
| Mermaid 配色规范 | `docs/.vitepress/mermaid-style.md`（5 个 classDef） | 直接复用 |
| 目录约定 | knowledge 按 domain 分层 + 每层 `index.md` + sidebar 注册 | 复用结构，替换领域划分 |
| 发布管线 | VitePress build + CI 自动部署 | 复用，从 GitHub Actions 换成 GitLab CI |

### 目标：Team Brain 要解决什么

一个感知算法团队（模型训练 / 推理 / 数据库管理 / 数据挖掘 / infra 工具同仓）的典型知识流失场景：

1. **踩坑经验失传**：某个 CUDA bug 的 workaround 只存在于某个人的聊天记录里，三个月后别人再踩一遍。
2. **数据闭环黑盒**：数据库里某个 sensor 字段的语义、某批数据的采集背景，只有当时的处理人知道。
3. **新人 onboarding 慢**：感知仓库 50w 行代码，新人靠读代码 + 口口相传，两周才能提交第一个 PR。
4. **决策无据可查**："当时为什么选 BEVFormer 而不是 StreamPETR"散落在会议记录里。

Team Brain 的定位：**感知仓库的"注释层"**——代码记录怎么做，Team Brain 记录为什么、踩过什么坑、边界在哪。

## 总体架构

```mermaid
flowchart LR
    subgraph Repo["感知仓库 GitLab"]
        code["模型训练/推理/数据挖掘/infra 代码"]
        ci["GitLab CI"]
        mr["Merge Request"]
    end
    subgraph Brain["Team Brain 仓库 GitLab"]
        inbox["docs/inbox"] --> knowledge["docs/knowledge"]
        knowledge --> site["GitLab Pages 站点"]
        kb["GitLab CI 部署"]
    end
    mr -- "MR 模板引导：踩坑/决策 → inbox" --> inbox
    ci -- "CI 失败复盘 → experience 模板" --> inbox
    code -. "双向链接：代码注释指向 brain 页面" .-> site
    site -- "notes 里引用 MR/文件路径" .-> code

    classDef step     fill:#eef2ff,stroke:#c7d2fe,color:#312e81,stroke-width:1.5px
    classDef action   fill:#fff7ed,stroke:#fdba74,color:#7c2d12,stroke-width:1.5px
    classDef decide   fill:#fef3c7,stroke:#fcd34d,color:#78350f,stroke-width:1.5px
    classDef branchNo fill:#f0fdf4,stroke:#86efac,color:#166534,stroke-width:1.5px
    classDef branchYes fill:#eef2ff,stroke:#c7d2fe,color:#3730a3,stroke-width:1.5px

    class code,ci,mr step
    class inbox,knowledge,site,kb step
```

两个仓库、一个站点：

- **感知仓库**：代码是真理之源，保持现状不动。
- **Team Brain 仓库**：Markdown 是真理之源，独立 GitLab 仓库，GitLab Pages 发布（内部可见）。
- 不把 Team Brain 塞进感知仓库：权限、CI、clone 体积、review 流程都会互相干扰。感知仓库只在 MR 模板和 README 里留指向 Team Brain 的链接。

## 第一阶段：迁移与骨架（第 1-2 周）

### 1. GitLab 仓库与 CI

- [ ] 新建 GitLab 项目 `team-brain`，把 second-brain 内容迁移过去（保留 git 历史，`git push` 整仓即可）
- [ ] 删除/归档个人内容：`investing/`、个人 reading 笔记等不迁，`archives/` 里个人内容留在旧仓库
- [ ] `.gitlab-ci.yml`：替换 GitHub Actions，`npm ci && npm run docs:build`，`artifacts` 指向 `docs/.vitepress/dist`，用 GitLab Pages 发布；`environment: internal`（登录才可见）
- [ ] GitLab Pages 域名确认：内网 GitLab 一般是 `https://<group>.pages.example.com/team-brain`，确认公司网络可达

### 2. 内容结构重构

`docs/knowledge/` 从"学习领域"改为"仓库子系统"优先：

```
docs/knowledge/
├── perception-training/     # 模型训练：数据集协议、增强策略、训练 trick、loss 设计
│   ├── index.md
│   └── bev-benchmark-protocol.md
├── perception-inference/    # 推理：部署目标、量化、时延预算、算子兼容
├── data-platform/           # 数据库管理 + 数据挖掘：schema 语义、挖掘 pipeline、采集任务
├── infra-tools/             # infra 工具：分布式存储、集群调度、监控
├── ai-foundations/          # 底层理论（BEV/多传感器融合/时序融合等），不绑定单一仓库模块
├── engineering/             # 通用工程（Git、C++、 profiling、K8s）——可直接整体迁移现有内容
└── onboarding/              # 新人指南：环境搭建、第一次跑通训练/推理、常见报错
```

- 顶层 `index.md` 重写为团队入口：站内搜索 + 各子系统入口 + onboarding 置顶
- `engineering/` 现有内容（Git on NFS、K8s、profiling、pyproject 等）直接迁移，这些本来就是团队会用到的

### 3. 团队版规则层

- [ ] `AGENTS.md` 改写：删掉个人化表述（"my personal second brain"），保留全部硬规则（搜索先行 / 原子笔记 / 链接不虚构 / Mermaid classDef / LaTeX 规范），新增团队规则（见第三阶段）
- [ ] 模板调整：
  - `experience.md` **升级为最重要模板**（团队最高频场景是踩坑复盘），字段对齐：Environment / Symptoms / Investigation / Root Cause / Solution / Why / Lessons / **Affected Repo Paths** / **Related MRs**
  - 新增 `template/decision.md`（ADR 风格）：Context / Options / Decision / Consequences / Status——对应"为什么选 A 不选 B"
  - `paper.md` 保留（感知团队也读论文）
  - `book.md` / 个人 idea 可删或降权
- [ ] frontmatter 增加 `owner` 字段：每篇笔记有明确 owner，避免"公共的没人管"

## 第二阶段：与感知仓库打通（第 3-4 周）

这是个人版没有的、也是 Team Brain 价值最大的部分。

### 1. MR 模板引导沉淀

- [ ] 感知仓库 `.gitlab/merge_request_templates/default.md` 加一个 section：

  ```markdown
  ## Knowledge Capture（可选但强烈建议）
  - 本次改动是否包含踩坑/非显然决策？
  - 如是：在 team-brain `docs/inbox/` 新建笔记（用 experience / decision 模板），
    并在下方粘贴笔记路径：____
  - CI 失败的 root cause 若有普适性，同样适用
  ```

- [ ] 不强制：MR review 里 reviewer 看到值得沉淀的内容时口头提醒即可，强制会让人敷衍

### 2. 双向链接约定

- Brain → Code：笔记里用**相对稳定的引用**写仓库路径 + MR 链接（`perception/tools/dump_bev.py` + `!1234`），不引用行号（行号会漂移）
- Code → Brain：约定一个轻量格式，代码注释或 docstring 顶部：
  ```python
  # See team-brain: knowledge/perception-training/bev-benchmark-protocol.md
  # （GitLab Pages 内网可达）
  ```
- [ ] 可选增强：写一个 `.gitlab-ci.yml` 的 lint job，检查笔记里引用的仓库路径是否真实存在（clone 感知仓库浅克隆 + 校验路径），防止链接腐烂

### 3. CI 失败自动建 issue（可选，第二期再做）

- 感知仓库 CI 失败 → GitLab webhook → 自动在 team-brain 仓库开 issue/或追加到一篇 `ci-failures.md`
- 价值在于积累"哪些坑反复出现"的统计，人再决定哪些值得写成 experience 笔记

## 第三阶段：团队化运行机制（第 5-8 周起长期）

个人版规则不需要变的部分：原子笔记、先 inbox 后整理、链接必须真实存在、Mermaid 统一配色、个人理解与外部事实分开标注。

需要**新增**的团队规则（写进 Team Brain 的 `AGENTS.md`）：

### 1. 所有权与消化

- 每篇笔记有 `owner`；inbox 笔记无 owner，进去时自动归属提交者
- **消化轮值**：每人每月花 1-2 小时把 inbox 里自己领域的碎片整理进 `knowledge/`（个人版是 AI + 一个人做，团队版必须分工，否则 inbox 会无限堆积）
- AI assistant 辅助消化，但 owner 把关：个人理解部分必须由本人确认，AI 不能替人写"我的理解"

### 2. 质量守门

- `knowledge/` 目录改动走 MR + 至少 1 人 review（review 重点：链接有效性、与既有笔记是否重复、个人理解是否冒充事实）
- `inbox/` 直推不 review（保持低摩擦，inbox 的价值就是随手记）
- 定期（每月）跑一次全库死链检查脚本，结果贴到团队频道

### 3. AI 助手团队化接入

个人版是每人本地跑 Claude Code / TRAE + 各自 API key；团队版建议：

- [ ] 统一规则入口：Team Brain `AGENTS.md` 就是 AI 助手的 system prompt 之源，任何人 clone 后本地 assistant 自动遵守同一套规则
- [ ] 维护一个 `prompts/` 目录（现有 prompt 模板可迁移），沉淀"整理笔记"“生成索引”等团队共用 prompt
- [ ] 不建议初期搞共享 RAG 服务：VitePress 本地搜索 + 每人本地 AI 助手读仓库，已能覆盖 90% 需求；等笔记量 >500 篇、本地搜索不够用时再评估

### 4. 激励与节奏

- 团会固定 10 分钟"本周知识沉淀"环节（展示 1-2 篇新笔记，比强制制度有效得多）
- 把"笔记引用了你的内容"当作可见的荣誉（GitLab Pages 上每篇笔记顶部显示 owner 和最近更新）
- 季度清理一次 `archives/`

## 里程碑与验收

| 阶段 | 时间 | 验收标准 |
|---|---|---|
| 一：迁移骨架 | 第 1-2 周 | GitLab Pages 站点内网可访问；新人按 `onboarding/` 能跑通一次训练+推理；`engineering/` 旧内容全部迁完且死链为 0 |
| 二：仓库打通 | 第 3-4 周 | 下一个真实 MR 里出现第一篇由 MR 引导的 experience/decision 笔记；双向链接格式在团队内达成一致 |
| 三：运行机制 | 第 5-8 周 | inbox 消化轮值运行满一个周期；`knowledge/` 改动全部走 MR review；每周至少 2 篇新笔记进 inbox |

## 风险与对策

| 风险 | 对策 |
|---|---|
| 没人写（最常见死法） | 从 onboarding + 踩坑两类最高频场景切入，不做"大而全"；团会 10 分钟展示代替 KPI |
| inbox 堆积腐烂 | 消化轮值 + inbox 笔记带 `created` 日期，超过 1 个月的每周提醒 |
| 笔记质量参差（AI 生成内容冒充事实） | AGENTS.md 强制"个人理解"与"外部事实"分开标注；knowledge/ 走 MR review |
| 与代码同步腐烂 | 引用仓库路径而非行号；路径校验 lint；季度归档清理 |
| 权限泄漏（数据 schema 等敏感内容） | Team Brain 仓库 + Pages 都设 internal 可见性；敏感字段语义类笔记单独 review |

## 非目标（明确不做）

- 不做对外公开站点——内网知识库
- 不替换 GitLab issue / wiki——GitLab wiki 适合流水账，Team Brain 适合需要结构、导航和长期维护的知识；两者在 MR 模板里各自指向
- 不在初期引入数据库/图谱/共享 RAG——先让 Markdown 闭环跑起来

## Related Knowledge

- [Second Brain 使用工作流](./second-brain-workflow.md) — 数据流与目录约定，Team Brain 直接继承
- [Second Brain 迭代路线图](./second-brain-iteration-roadmap.md) — 个人版的迭代经验（内容闭环优先于堆规则），同样适用于 Team Brain
- Shared AI Prompt Templates（`prompts/`）— 团队共用 prompt 的起点
- Claude Compatibility Layer（`.claude/`）— AI 助手接入方式的参考
