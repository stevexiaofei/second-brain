# 仓库链接与导航修复重构计划

## 审计背景

本次仓库级审计聚焦四个维度：导航完备性、站内相对链接、frontmatter 一致性、结构/重复。审计方式为全量机械扫描（223 个 .md 文件逐一核对侧栏路由、索引条目、相对链接解析），并对关键发现逐行人工复核。目标是以最小 diff 修复已确认的缺陷，不重写知识内容，不创建占位笔记。

**审计日期**：2026-10-19（计划归档 2026-10-20）

## 受保护的现有 Git 状态

开始时已有以下未提交修改，均不属于本计划：

- `docs/knowledge/mathematics/group-theory/01…09-*.md`（9 篇）——本会话前半段"扩写定理证明"任务的成果，属于内容增强。**本计划会修改这 9 篇文件，但只动链接字符串，不触碰任何已扩写的内容**；修复后 diff 中链接行与内容行需可分辨。
- `print/` 下产物（gitignore 范围，不入库）。

## 已确认发现与实施边界

### 1. 群论专题 148 行 / 166 处缺数字前缀的死链（最高优先级）

- **证据**：`docs/knowledge/mathematics/group-theory/` 全部 10 个文件引用 8 个兄弟笔记时省略了文件名的数字前缀（如 `./mod-n-and-dihedral-groups.md`，实际文件是 `02-mod-n-and-dihedral-groups.md`）。这 8 个无前缀文件名在整个仓库中从未存在过。逐文件行数：index 22、01→8、02→12、03→13、04→10、05→19（含 163 行一处**自我链接**，实际想指向本文 121 行的"推论 3.39"，应去掉链接或改锚点）、06→13、07→16、08→12、09→23。
- **影响**：站点上专题内所有交叉引用全部 404；打印工具靠容错回退才能解析。HEAD 版本已坏（非本次会话引入），且未提交的扩写在持续新增同类死链。
- **实施**：对 8 个目标名做全仓库范围的 `NN-` 前缀补全（仅 `docs/knowledge/mathematics/group-theory/` 内命中；`_sidebar`/其他目录无引用，已扫描确认）。05:163 的自链接改为纯文本"推论 3.39（见上文）"。同步把 `group-theory/index.md` 的 `updated` 改为实施日期；9 篇正文因内容行数多、且 `updated` 仅要求"实质变更时更新"，本次只改链接字符串，**统一不动正文 frontmatter**（避免 9 处无意义日期翻新）。
- **置信度**：确认（逐行核过抽样 + 全量计数）。
- **工作量/风险**：低。纯字符串替换，替换后跑链接扫描归零验证。

### 2. 群论之外的 12 处真实死链（4 个文件）

- **证据**（逐行核过）：
  - `docs/knowledge/ai/systems/gpu-memory-access-model.md:24,134` → `../gpu-kernel-optimization-methodology.md` 应为 `./gpu-kernel-optimization-methodology.md`（目标文件是 `ai/systems/` 的兄弟，不在 `ai/`）。
  - `docs/knowledge/ai/systems/cutlass/mma-atom-official-doc-zh.md:455,471` → `./06-cute-mixed-precision-mma.md` 应为 `./10-cute-mixed-precision-mma.md`（06 是 CuTe 上游教程编号泄漏；侧栏里该笔记就叫"10 混合精度"）。
  - `docs/knowledge/ai/systems/flash-attention/flash-attention-interface-and-autograd.md:352` → `./attention-head-variants.md` 应为 `../../attention-head-variants.md`（实际位于 `ai/` 根，从 `ai/systems/flash-attention/` 需上两级）。
  - `docs/knowledge/distributed-systems/distributed-storage/ai-training-and-multicloud-storage-map.md` 6 处系统性深度错误：328 行 `../distributed-systems/…`→`../…`；361 行 `../ai/…`→`../../ai/…`；411 行同上；412 行 `./rvv-operator-development.md`→`../../engineering/rvv-operator-development.md`；413 行 `../ai/systems/pytorch/`→`../../ai/systems/pytorch/`；414 行 `../../projects/…`→`../../../projects/…`。
- **影响**：站点死链（构建因 `ignoreDeadLinks: true` 不报错，但点击 404）。
- **实施**：按上述逐一改为正确相对路径。该地图笔记的 `updated` 顺带更新（它整段 Related 都在改）。
- **置信度**：确认。
- **工作量/风险**：低。

### 3. 侧栏缺 28 个已存在笔记 + archives 孤儿

- **证据**：
  - PyTorch 段（config.mts:98–117）只收了 torch-compile-wiki 17 章；`docs/knowledge/ai/systems/pytorch/` 下另有 17 篇（pytorch-overview、pytorch-architecture…pytorch-key-classes）只在父 index.md 可达。
  - Distributed Systems 段（config.mts:122–127）只收目录 index 与地图；`distributed-storage/` 下另有 7 篇（block-file-object-comparison、ceph-lustre-minio-overview、consistency-consensus-failure、metadata-and-performance、object-storage、parallel-file-systems、replication-erasure-coding-placement）只在祖父 index 可达。
  - `docs/archives/index.md` 无 nav、无侧栏段、首页无入口——只能手敲 URL 到达。
- **影响**：28 篇已构建页面在侧栏不可发现；archives 完全不可达。
- **实施**：
  - PyTorch 段内新增折叠子段"核心笔记（源码阅读）"列出 17 篇（保持 torch-compile-wiki 原样，PyTorch 段改 `collapsed: true` 已是，无需动）。
  - Distributed Systems 段新增折叠子项"分布式存储基础系列"列出 7 篇。
  - nav 加 Archives 入口 + 新增 `/archives/` 侧栏段（Overview 一项）+ `docs/index.md` 补一个 Archives 区块入口。
  - 同时修复发现 4 的连带项：`distributed-storage/index.md` 补上自己 7 篇子笔记的 Notes 清单（现在只字未提其中 2 篇，全部 7 篇靠祖父 index 列出）。
- **置信度**：确认。
- **工作量/风险**：中（config.mts 结构性新增约 30 行，需构建验证侧栏渲染）。

### 4. 侧栏重复段：群论 10 条目双份

- **证据**：config.mts:203–212（Mathematics 段内）与 214–225（独立"群论与抽象代数"折叠段）完全重复。
- **影响**：侧栏出现两份相同入口，维护时容易只改一份（本次群论证明扩写期间就实际发生过）。
- **实施**：删除 Mathematics 段内的 8 篇正文条目（203 行"群论与抽象代数专题"总览入口保留），保留独立折叠段为唯一入口。
- **置信度**：确认。
- **工作量/风险**：低。

### 5. frontmatter 缺口（仅补 3 篇硬缺，不做批量回填）

- **证据**：`grpo.md`、`ppo.md`（缺 created/updated/source）、`git-on-nfs.md`（同）——全部内容笔记都有 title/type/status/tags，这 3 篇连日期都没有。另有 ~42 篇缺 `source`，但那是按目录批量导入形成的模式，属风格选择，不算缺陷。
- **实施**：只给 3 篇补 `created`（用 git log 首次提交日期）、`updated`（实施日期）、`source`（无法确定的写"整理自学习过程"）。42 篇的 source 回填**明确不做**，记录为待办。
- **置信度**：确认。
- **工作量/风险**：低。

### 6. Inbox 老化与知识→inbox 反向依赖（只记录 + 修依赖，不移动）

- **证据**：3 篇 inbox 笔记已静置 8–9 周（created 2026-08-17/21，最后实质提交 2026-09-06，全部 status: seed）。`cutlass/index.md:144-145` 两处链接从稳定知识区指向 inbox（`../../../../inbox/cuda-*.md`，路径可解析，站点可跳，但语义上是"稳定内容依赖暂存区"）。
- **实施**：**本次不移动 inbox 笔记**（分类判断需通读内容，超出本边界）。仅在 `docs/inbox/index.md` 的 3 个条目后追加老化备注（"已静置 8+ 周，待定去留"），把"是否晋升到 knowledge/"记入待办。cutlass 的两处链接保持原样（当前可解析，移动时再修）。
- **置信度**：确认（事实）；处理方式属判断，已按最小动作收缩。

## 不在本次范围内（明确搁置）

- 不拆分、不重写任何长笔记；不动 9 篇群论笔记的证明内容。
- 不做 ~42 篇 `source` 字段的批量回填。
- 不移动 3 篇 inbox 笔记到 knowledge/（需内容通读后另行决定）。
- 不修 `docs/knowledge/index.md:8-9` 与 `docs/projects/index.md:13-14` 指向仓库根 `prompts/`、`.claude/` 的站外链接——这是 `ignoreDeadLinks: true` 的既定理由（仓库文件非站点页面），保留现状。
- 不动 `mermaid-style.md:41` 的 `../../.vitepress/config.mts`（非站点内容目录，纯装饰性）。
- 不新增自动化链接检查 CI（候选后续工作）。
- 不提交、不推送。

## 验证步骤

1. 链接扫描脚本复扫全仓 `docs/**/*.md` 相对链接：群论 8 个无前缀目标命中数 = 0；12 处群论外死链解析成功。
2. `node -e` 侧栏校验：config.mts 提取的全部 link 路径对应文件存在；28 篇此前缺失的笔记出现在侧栏。
3. `NODE_OPTIONS=--max-old-space-size=8192 npm run docs:build` 构建通过（build 完成且无新增死链告警）。
4. `git diff --check`（无空白错误）；`git status` 确认改动仅限计划边界内文件 + 本计划归档文件。
5. 抽查 5 处修复后的链接在构建产物（`docs/.vitepress/dist`）中渲染为正确 href。

## 实施结果（2026-10-20 完成）

全部 7 项按边界实施完毕，验证全部通过：

1. **群论死链**：脚本精确替换 166 处（index 24、01→10、02→16、03→15、04→14、05→19、06→14、07→18、08→13、09→23）；05:163 自链接改为"由下文推论 3.39(1)"纯文字。复扫 8 个无前缀目标命中 0。
2. **群论外 12 处死链**：gpu-memory-access-model 2 处、cutlass 译文 2 处（06→10）、flash-attention 1 处、存储地图 6 处，全部改为正确相对路径；两篇 storage 笔记 `updated` → 2026-10-20。
3. **侧栏补齐**：PyTorch 段新增 17 篇核心笔记并把 17 章 wiki 收进折叠子段；Distributed Systems 段新增折叠子段"分布式存储基础系列"列 7 篇（标题与各篇 frontmatter 对齐）；nav + 侧栏 + 首页均加 Archives。
4. **群论重复段**：删除 Mathematics 段内 8 篇正文条目（总览入口保留），独立折叠段为唯一正文入口。
5. **distributed-storage/index.md**：新增"本目录笔记"8 条清单，学习顺序补第 1 步（对比篇）。
6. **frontmatter**：grpo/ppo/git-on-nfs 补 created=2026-08-11（git log 首提交）、updated=2026-10-20、source（保守表述，不虚构出处）。
7. **inbox**：index.md 加状态备注（静置 8 周以上 + cutlass 反向依赖提示），未移动任何笔记。

**验证记录**：

- 全仓相对链接复扫（223 篇，剔代码块）：broken = 0
- 侧栏校验：192 个 link 全部解析到实际文件（唯一"missing"是 socialLinks 的 GitHub 外链，属预期）；24 篇此前缺失笔记全部入栏
- `NODE_OPTIONS=--max-old-space-size=8192 npm run docs:build` 构建通过（157s）
- 构建产物抽查：群论 index 的 02- 链接、cutlass 10- 链接、存储地图 ddim 链接渲染正确；archives/index.html 与 object-storage.html 已产出
- `git diff --check` 干净（修复了 09:111 一处早前遗留的尾随空行）

**备注**：`plans/2026-10-20-navigation-links-refactor-plan.md` 是同一审计的另一份归档（并行会话产出，内容一致、行号口径略异），两份均保留作为审计记录；实施以本文件为准。
