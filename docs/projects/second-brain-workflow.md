---
title: Second Brain 使用工作流
type: concept
status: seed
tags: [Second-Brain, Workflow, TRAE]
created: 2026-08-11
updated: 2026-08-11
source: 与 TRAE 的对话整理
---

# Second Brain 使用工作流

## 核心心智模型

Second Brain 不是笔记本，是「外置大脑」。原则（见 [AGENTS.md](../../AGENTS.md)）：

> 这份工作流最初按 TRAE 写，但同样适用于 Claude Code；下文中的“TRAE”可以理解为“你正在使用的 AI assistant”。


- Markdown 是真理之源，HTML 只是呈现层
- 原子化：一概念一文
- 先进 [inbox](../inbox/index.md)，后整理到 knowledge
- 强调个人理解和连接，不是抄录

## 数据流

```
Capture / Learn / Think
       ↓
    📥 Inbox              ← 任何碎片先丢这里
       ↓
     AI assistant         ← 搜索 → 去重 → 分类 → 连接
       ↓
  🧠 Knowledge            ← 按领域归档到 docs/knowledge/
       ↓
   Git + GitHub           ← commit / push
       ↓
   VitePress Wiki         ← Actions 自动部署
```

## 目录与用途

| 目录 | 作用 | 何时放进去 |
|---|---|---|
| `docs/inbox/` | 临时收件箱 | 想到/看到/遇到任何东西，先丢这 |
| `docs/knowledge/` | 稳定知识库（按领域分子目录） | 已理解、能讲清楚、可复用 |
| `docs/ideas/` | 想法 / 假设 | 还没验证、不想立刻深挖的念头 |
| `docs/projects/` | 项目复盘 | 一个项目阶段性总结 |
| `docs/reading/` | 阅读清单/笔记 | 书、长文 |
| `docs/archives/` | 归档 | 过时但舍不得删 |
| `templates/` | 6 种模板 | 新建笔记时套用 |

模板 6 类：[concept](../../templates/concept.md)、[paper](../../templates/paper.md)、[experience](../../templates/experience.md)、[idea](../../templates/idea.md)、[book](../../templates/book.md)、[project](../../templates/project.md)。

共享 AI prompt 模板见 [prompts/](../../prompts/)；TRAE 兼容层见 [.trae/agents/](../../.trae/agents/)；Claude 兼容层见 [.claude/agents/](../../.claude/agents/)。

## 日常工作流场景

每个场景的对话样例都是"你口述 → TRAE 执行"的真实交互形态，不需要复制对话原文，AI assistant 会按模板把结构化结论落到 Markdown。

### 场景 1：读论文

**Step 1 — 记录到 inbox**

```
你：我读了 GRPO 这篇论文，记到 inbox

TRAE：
  1. 搜仓库 → 发现 docs/knowledge/ai/foundations/reinforcement-learning/ppo.md 但没有 grpo，不重复
  2. 套 paper 模板，写到 docs/inbox/grpo-notes.md
     - Problem / Motivation / Key Idea / Method / 留空
     - My Understanding / Open Questions 先空着
  3. 在 docs/inbox/index.md 的 Notes 加链接
  4. 反问你：
     - 看的是 DeepSeek-Math 那版还是原 GRPO 论文？
     - Motivation 你看到的是什么？
```

**Step 2 — 补全结构**

```
你：DeepSeek-Math 那版，去掉了 critic，用 group baseline

TRAE：
  1. 补全 Method 节：去掉 value network，组内均值做 baseline
  2. 补全 Mathematical Formulation：写出 A_i = r_i - mean(r_group) 的公式
  3. 反问：和 PPO 相比省了哪些显存？你的理解是什么？

你：省了 critic 参数和它的激活，但代价是方差更大

TRAE：补到 "My Understanding" 节，明确标注「我的理解」
     → status: seed → growing
```

**Step 3 — 整理到 knowledge**

```
你：把 grpo-notes 整理成正式笔记，放到 knowledge/ai/

TRAE：
  1. 搜重复 → 无冲突
  2. 补全 Related Knowledge，加链接到 [ppo.md](../knowledge/ai/foundations/reinforcement-learning/ppo.md)
  3. 移动 docs/inbox/grpo-notes.md → docs/knowledge/ai/foundations/reinforcement-learning/grpo.md
  4. 从 docs/inbox/index.md 删条目
  5. 在 docs/knowledge/ai/index.md 的 RLHF 区块加 GRPO 链接
  6. git commit -m "docs: add GRPO knowledge"
```

参考已有样例：[grpo.md](../knowledge/ai/foundations/reinforcement-learning/grpo.md)、[ppo.md](../knowledge/ai/foundations/reinforcement-learning/ppo.md)。

### 场景 2：工程踩坑

**Step 1 — 立即口述，先丢 inbox**

```
你：刚才在 NFS 上 git status 卡了好几秒，记到 inbox

TRAE：
  1. 套 experience 模板，写到 docs/inbox/git-on-nfs.md
     - Problem: 已知现象
     - Environment: NFS / Git（细节留空，反问补全）
     - Investigation / Root Cause / Why / Lessons Learned 先空
  2. 在 docs/inbox/index.md 加链接
  3. 反问：
     - NFS 版本？mount 选项？
     - 仓库文件数量级？
     - 是 git status 慢，还是 git add / commit 也慢？
```

**Step 2 — 定位根因后补全**

```
你：根因是 metadata latency，文件 50w+，git 在扫 working tree

TRAE：
  1. 补全 Investigation：列出实际查过的指标（RTT / inode cache / file count）
  2. 补全 Root Cause：NFS metadata 操作延迟被 working-tree scan 放大
  3. 补全 Why：解释为什么带宽不是瓶颈，metadata 才是
  4. 补全 Lessons Learned：以后 NFS + 大仓库要配 fsmonitor / sparse-checkout
  5. status: seed → growing
```

**Step 3 — 移到 knowledge**

```
你：整理到 knowledge/engineering/

TRAE：
  1. 移动 docs/inbox/git-on-nfs.md → docs/knowledge/engineering/git-on-nfs.md
  2. 更新 docs/inbox/index.md（删条目）
  3. 更新 docs/knowledge/engineering/index.md（加条目）
  4. git commit -m "docs: add git-on-nfs experience"
```

参考样例：[git-on-nfs.md](../knowledge/engineering/git-on-nfs.md)。

### 场景 3：学习新概念

**Step 1 — 记录初印象**

```
你：我在学 Self-Attention，记到 inbox

TRAE：
  1. 搜仓库 → 没有 self-attention，但有 grpo/ppo（不冲突）
  2. 套 concept 模板，写到 docs/inbox/self-attention.md
     - What is it / Why it matters / How it works 先空
  3. 在 docs/inbox/index.md 加链接
  4. 反问：是 scaled dot-product？multi-head？你想从哪一层入手？
```

**Step 2 — 真正理解后补全**

```
你：scaled dot-product，公式 softmax(QK^T / sqrt(d_k)) V

TRAE：
  1. 补全 How it works：写公式 + 解释 Q/K/V 含义
  2. 反问：为什么除以 sqrt(d_k)？你的理解是什么？

你：避免点积过大让 softmax 饱和

TRAE：
  1. 写到 "My Understanding" 节，明确标注「我的理解」
  2. 反问：这个解释是你自己想的，还是看某篇博客？要不要标注来源？
  3. status: seed → growing
```

**Step 3 — 链接与归档**

```
你：整理到 knowledge/ai/，链接到 Transformer

TRAE：
  1. 搜知识库 → 发现 knowledge/ai/ 没有 transformer.md
  2. 反问：要不要先建一个 Transformer stub？还是先把 Self-Attention 放进去，等学到 Transformer 再补？
  3. 移动 self-attention.md → docs/knowledge/ai/self-attention.md
  4. 更新 index，添加 "Related: Transformer (TODO)" 占位链接
  5. git commit -m "docs: add self-attention concept"
```

### 场景 4：突发灵感

**Step 1 — 快速记下，不展开**

```
你：突然想到用 GRPO 的 group baseline 替代 PPO 的 advantage baseline，先记到 ideas

TRAE：
  1. 套 idea 模板，写到 docs/ideas/grpo-baseline-for-ppo.md
     - Hypothesis: GRPO 的 group baseline 可以替代 PPO 的 GAE
     - Motivation / Expected benefit 简短一两句
     - status: hypothesis
  2. 在 docs/ideas/index.md 加链接
  3. 反问（轻量，不深挖）：
     - 你预期最大的收益是省 critic 还是别的？
     - 留一句话就行，详细推导等之后再说
  4. 不放进 knowledge（未验证）
```

**Step 2 — 之后某天回来验证**

```
你：把 grpo-baseline-for-ppo 这个想法验证一下

TRAE：
  1. 搜相关论文 → 找到 RLOO / RAFT 等类似工作
  2. 反问：要不要先看 RLOO 这篇？还是直接跑个小实验？
  3. 根据你的选择执行：
     - 验证成立 → 移到 docs/knowledge/ai/，标注来源 + 实验记录
     - 验证失败 → 在原文档写 "Why it doesn't work"，移到 docs/archives/
  4. 更新对应 index
```

灵感不要立刻展开，避免打断主线工作。先记，后验证。

### 场景 5：项目复盘

**Step 1 — 启动复盘**

```
你：把这次 DDP 多卡训练项目复盘一下

TRAE：
  1. 套 project 模板，写到 docs/projects/ddp-training-2026-08.md
     - Goal / Context / Decisions / What went well / What went wrong /
       Lessons / Artifacts 全部留空
  2. 在 docs/projects/index.md 加链接
  3. 反问关键点：
     - Goal 是什么？业务指标还是技术指标？
     - 多少卡？什么拓扑？什么模型规模？
     - 哪几个决策事后看是对的？哪些是错的？
     - 卡在哪一步：通信？显存？数据加载？
```

**Step 2 — 口述补全**

```
你：（口述项目细节，可以乱序、可以啰嗦）

TRAE：
  1. 把口述内容拆到对应字段：
     - "用了 8 卡 A100" → Context
     - "选了 DDP 不是 FSDP，因为模型放得下" → Decisions
     - "通信瓶颈卡在 backward" → What went wrong
     - "下次先 profile 再选策略" → Lessons
  2. 反问：通信瓶颈这一条要不要单独抽成 experience 笔记？
```

**Step 3 — 抽取关联笔记**

```
你：通信瓶颈那条单独抽

TRAE：
  1. 新建 docs/knowledge/engineering/ddp-communication-bottleneck.md
     - 套 experience 模板
     - 复用项目里的 Investigation / Root Cause
  2. 双向链接：
     - project 笔记 → experience 笔记
     - experience 笔记 → project 笔记
  3. 更新 docs/knowledge/engineering/index.md
  4. git commit -m "docs: add DDP project retro + communication bottleneck"
```

### 场景 6：读书

**Step 1 — 建立读书骨架**

```
你：我在读《DDIA》，每章一条笔记

TRAE：
  1. 套 book 模板
  2. 反问：每章一份，还是每章一节？要不要先建目录？

你：每章一份，原子化

TRAE：
  1. 新建 docs/reading/ddia/ 目录
  2. 写 docs/reading/ddia/index.md 作为整书的章节索引（先空，每读完一章补一条）
  3. 在 docs/reading/index.md 加 DDIA 入口
  4. 反问：第一章你看完了吗？要现在记，还是等读完再记？
```

**Step 2 — 每章一份原子笔记**

```
你：第一章读完了，讲 reliable / scalable / maintainable

TRAE：
  1. 新建 docs/reading/ddia/ch01-reliable-scalable-maintainable.md
  2. 套 book 模板补全：
     - Summary: 一句话讲什么
     - Key Ideas: 三个关键词各一段
     - My Understanding: 你怎么看作者举的电商例子？有实际项目能对照吗？
     - Open Questions: 留疑问
  3. 反问：
     - "reliability" 这一节你工作里有对照场景吗？
     - "scalability" 的负载参数你想用哪个项目举例？
  4. 在 docs/reading/ddia/index.md 加第一章链接
```

**Step 3 — 读完后跨章串联**

```
你：DDIA 全读完了，用 index 把各章串起来

TRAE：
  1. 在 docs/reading/ddia/index.md 写：
     - 章节索引（按全书顺序）
     - 跨章连接（例如第 5 章 replication ↔ 第 6 章 partitioning）
     - 标注哪些章节对你的实际工作有启发
  2. 反问：要不要从读书笔记里抽出几条放到 knowledge/？
     （例如把 "scalability" 抽成 concept，链接回原章节）
  3. git commit -m "docs: finish DDIA reading notes"
```

每读完一章重复 Step 2；全部读完做 Step 3。

### 场景 7：读源码 / 系统化研究一个项目

读源码本质上是"系统化研究一个项目"，介于学习概念和读书之间，但有独特之处：源码会变（必须锁版本）、需要代码引用（带文件路径+行号）、强调调用链。归档位置是 `docs/knowledge/engineering/<项目名>/`，每个模块一份原子笔记。

**准备工作 — 用 multi-root workspace 同时挂载两个 folder**

为了让 TRAE 既能读源码、又能写 second-brain 笔记，建议在 Trae 里用 multi-root workspace 把两个目录挂到同一个窗口：

1. 在 Trae 里先打开 second-brain
2. `File → Add Folder to Workspace…` → 选源码目录（例如 `D:\code\pytorch`）
3. `File → Save Workspace As…` → 保存为 `second-brain.code-workspace`
   - **保存到 second-brain 仓库外**（例如上一级目录），避免污染 git
4. 以后双击这个 `.code-workspace` 文件即可同时打开两个 folder

注意事项：

- AI 的 primary working directory 默认是第一个 folder（second-brain），写笔记时会正确落到 `docs/knowledge/engineering/<项目名>/`
- 读源码时要在指令里明确路径，例如：
  > 读 `D:\code\pytorch\torch\distributed\distributed_c10d.py` 里 `init_process_group` 的实现，整理成笔记写到 `docs/knowledge/engineering/pytorch/process-group.md`
- [AGENTS.md](../../AGENTS.md) 规则只在 second-brain 子目录生效：读源码不受约束，写笔记时遵守 second-brain 规则
- AI 不会动源码目录的文件（除非你明确要求），它只在 second-brain 里写

**Step 1 — 启动：建立阅读骨架**

```
你：我想读 PyTorch 的 DDP 源码，做成笔记

TRAE：
  1. 搜仓库 → docs/knowledge/engineering/ 下没有 pytorch 子目录，不重复
  2. 反问关键点：
     - 目标是什么？理解实现细节 / 排查问题 / 自己造一个？
     - PyTorch 版本 + commit hash？（源码会变，必须锁定）
     - 按模块切（ProcessGroup / Bucket / Reducer / Hook）还是按调用链
       （forward → backward → allreduce）？
  3. 根据你的回答建骨架：
     - docs/knowledge/engineering/pytorch/index.md（先空，每读一块补一条）
     - docs/knowledge/engineering/pytorch/ 下每个模块一份原子笔记
  4. 在 docs/knowledge/engineering/index.md 加 PyTorch 区块
  5. 不丢 inbox——系统化研究直接进 knowledge，inbox 只放碎片
```

**Step 2 — 每个模块一份原子笔记**

```
你：今天看了 ProcessGroup，重点是 init_process_group 的实现

TRAE：
  1. 新建 docs/knowledge/engineering/pytorch/process-group.md
     - 套 concept 模板，补源码特定字段：
       - Source: PyTorch v2.4.0 / commit abc1234
       - Files: torch/distributed/distributed_c10d.py#L500-L700
     - 一句话理解 / 为什么重要 / 工作原理 先空
     - Open Questions 留空待补
  2. 反问：
     - init_process_group 你看到几种 backend？默认是哪个？
     - rendezvous 过程你的理解是什么？
     - 哪一段代码你看不懂？

你：rendezvous 用 env:// 时读 MASTER_ADDR/MASTER_PORT，
    但我不懂为什么要有 store 这层抽象

TRAE：
  1. 补 "工作原理"：rendezvous 流程 + 关键代码引用（带文件路径和行号）
  2. 补 "我的理解"：明确标注「我的理解」 vs 「代码事实」
  3. Open Questions 记下："为什么需要 store 抽象？"
  4. 反问：要不要下次专门写一篇 TCPStore 的笔记？
  5. 在 pytorch/index.md 加 ProcessGroup 链接
  6. status: seed → growing
```

**Step 3 — 串联 + 抽取关联笔记**

```
你：DDP 主线读完了，把各模块串起来

TRAE：
  1. 在 docs/knowledge/engineering/pytorch/index.md 写：
     - 模块索引（按数据流顺序：ProcessGroup → Bucket → Reducer → Hook）
     - 跨模块连接（例如 Reducer 依赖 ProcessGroup）
     - 调用链总览：forward → autograd hook → bucket ready → allreduce → optimizer step
  2. 反问：
     - 通信瓶颈这条要不要单独抽成 experience 笔记？
     - 你之前有没有 DDP 踩坑可以双向链接过来？
  3. git commit -m "docs: add PyTorch DDP source reading notes"
```

**读源码笔记的特殊规则**

1. **必须锁定版本**：frontmatter 加 `source_version: PyTorch v2.4.0 @ abc1234`，否则半年后行号全失效
2. **代码引用用文件路径+行号**：`torch/distributed/distributed_c10d.py#L500-L700`，方便回查
3. **严格区分三层**：代码事实 / 我的理解 / 推测（符合 AGENTS.md 的 Personal understanding）
4. **看不懂的留 Open Questions**：不要硬编造解释，下次读到相关模块时回填

## 与 AI assistant 协作的常用指令

| 你说 | TRAE 做 |
|---|---|
| "把这段记到 inbox：…" | 套最合适的模板，写到 `docs/inbox/` |
| "整理 inbox 里的 X" | 搜索重复 → 补全结构 → 移到 `docs/knowledge/<领域>/` |
| "X 和 Y 有什么联系" | 跨笔记找连接，加双向链接 |
| "搜一下我之前记过关于 X 的内容" | 全仓库检索，避免重复造轮子 |
| "把 X 的状态从 growing 改成 evergreen" | 更新 frontmatter 的 `status` |
| "提交这次改动" | 按规则 commit（如 `docs: add GRPO`） |

笔记 `status` 字段：`seed`（刚记）→ `growing`（在补）→ `evergreen`（成熟）。

## 发布成 Wiki

提交到 GitHub 后，[.github/workflows/deploy.yml](../../.github/workflows/deploy.yml) 会自动构建 VitePress 并部署。

```bash
npm run docs:dev      # 本地实时预览
npm run docs:build    # 本地构建
```

## 入门建议

不要一开始就追求完美分类。第一周只做两件事：

1. 任何想法/读到的东西/遇到的坑，先丢 [inbox](index.md)，哪怕一句话。
2. 每周末花 10 分钟让 TRAE 帮你整理 inbox：分类、去重、链接、归档。

跑两周自然摸出自己的节奏，再调整目录和模板。

## Related

- [AGENTS.md](../../AGENTS.md) — 仓库规则
- [README.md](../../README.md) — 项目说明
- [Inbox 规则](index.md)
- [templates/](../../templates/)

## Open Questions

- 个人节奏（每天/每周整理一次）还没定下来
- 跨领域笔记的链接密度应该多大，目前没经验
