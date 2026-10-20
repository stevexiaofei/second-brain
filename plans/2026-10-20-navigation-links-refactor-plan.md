# 仓库导航与链接修复重构计划（2026-10-20）

## 审计范围

全仓 223 篇 Markdown（docs/ 下），四个维度：导航完整性、相对链接有效性、frontmatter 一致性、结构/重复。审计方法为全量机械扫描 + 人工抽查核实。

## 受保护的现有 Git 状态

开始时已有以下未提交修改，**均不属于本计划的目标，但与本计划第 1 项有交集**：

- `docs/knowledge/mathematics/group-theory/01–09-*.md` 共 9 篇——本会话前半段"扩写定理证明"的内容增强成果。**这些文件的第 1 项修复会在其现有未提交改动之上叠加少量单行链接替换**；除此之外不触碰其内容。
- `print/` 目录下 4 个产物文件（在 .gitignore 范围内，不入库）。

## 已确认发现（含证据）与实施边界

### 1. group-theory 专题 148+ 行死链（最高优先级，量最大）

- **证据**：9 篇编号笔记 + index.md 中所有指向兄弟笔记的链接均缺数字前缀（如 `./mod-n-and-dihedral-groups.md`，实际文件为 `./02-mod-n-and-dihedral-groups.md`），8 个目标名全部无前缀形式在全仓不存在。按文件行数分布：index 22 行、01→8、02→12、03→13、04→10、05→19（含 163 行一处**自链接**，应为指向自身锚点或删除链接语法）、06→13、07→16、08→12、09→23。逐行清单由审计脚本产出，实施时以机械替换 + 复扫为准。
- **额外确认**：这些死链在 HEAD 里就存在（index.md 工作区未修改也有 22 行死链），且本轮证明扩写在**新增**死链——只修 index 追不上。
- **实施**：对 10 个文件做唯一机械替换——8 个无前缀名 → 对应带前缀名（映射固定：mod-n→02、symmetric→03、group-properties-I→04、group-properties-II→05、normal-subgroup→06、automorphism→07、isomorphism-theorems-I→08、isomorphism-theorems-II→09）。05:163 的自链接单独处理（改为纯文字"上文推论 3.39"）。不改动其他任何内容。
- **置信度**：确认。**风险**：低（纯链接文本替换）。

### 2. group-theory 之外的 12 处死链（4 个文件）

- **证据**（逐条人工核实）：
  - `docs/knowledge/ai/systems/gpu-memory-access-model.md:24,134` → `../gpu-kernel-optimization-methodology.md` 应为 `./gpu-kernel-optimization-methodology.md`（目标实际是兄弟文件）
  - `docs/knowledge/ai/systems/cutlass/mma-atom-official-doc-zh.md:455,471` → `./06-cute-mixed-precision-mma.md` 应为 `./10-cute-mixed-precision-mma.md`（上游教程编号泄漏）
  - `docs/knowledge/ai/systems/flash-attention/flash-attention-interface-and-autograd.md:352` → `./attention-head-variants.md` 应为 `../../attention-head-variants.md`
  - `docs/knowledge/distributed-systems/distributed-storage/ai-training-and-multicloud-storage-map.md:328,361,411,412,413,414` → 6 处系统性深度错误（`../distributed-systems/…`→`../…`、`../ai/…`→`../../ai/…`、`./rvv…`→`../../engineering/rvv…`、`../ai/systems/pytorch/`→`../../ai/systems/pytorch/`、`../../projects/…`→`../../../projects/…`）
- **实施**：逐条修正为上述正确路径。
- **置信度**：确认（每个目标文件都已 ls 验证存在）。**风险**：低。

### 3. 侧栏缺 28 个内容页 + archives 孤儿 + distributed-storage 父索引缺口

- **证据**：
  - PyTorch 段（config.mts:98–117）只收录 torch-compile-wiki 17 章，`pytorch/` 下另外 17 篇（pytorch-overview、pytorch-architecture…pytorch-key-classes）全部缺席（父 index.md 有列）。
  - Distributed Systems 段（config.mts:122–127）缺 `distributed-storage/` 下 7 篇子笔记。
  - 群论 10 条目在侧栏出现两次：Mathematics 段内（config.mts:203–212）+ 独立"群论与抽象代数"段（config.mts:214–225），完全重复。
  - `docs/archives/index.md` 无 nav、无侧栏、无任何入口链接——纯孤儿页。
  - `distributed-storage/index.md` 未列出自己目录下的 `block-file-object-comparison.md` 与 `ceph-lustre-minio-overview.md`（祖级 index 列了，父级没列，违反 AGENTS.md"父目录 index 必须列出"的约定）。
- **实施**：
  - 侧栏：PyTorch 段加 17 篇（按 pytorch/index.md 的阅读顺序，collapsed 子段"核心笔记"或平铺插入）；Distributed Systems 段加 7 篇；**删除重复的"群论与抽象代数"独立段**（保留 Mathematics 段内的 10 条）。
  - nav：加 Archives 入口（不新加侧栏段，archives 只有 index 一页）。
  - `distributed-storage/index.md` 补两行笔记列表项。
- **置信度**：确认。**风险**：低—中（config.mts 改动需构建验证）。

### 4. frontmatter 缺口（低优先级，仅补关键缺失）

- **证据**：`grpo.md`、`ppo.md`（缺 created/updated/source）、`git-on-nfs.md`（同）；另有约 42 篇按目录批量缺 `source`（pytorch 17 篇、distributed-storage 9 篇等）。
- **实施边界**：本次**只补** grpo/ppo/git-on-nfs 三篇的 `created`/`updated`（以 git 首次提交日期为准）与可考的 `source`。**42 篇批量 `source` 回填不在本次范围**（工作量大且来源考据需逐篇确认，记为延后项）。
- **置信度**：确认（字段缺失）；部分 `source` 值需考据，宁可留空不造。**风险**：低。

## 明确延后（不在本次实施）

- 约 42 篇的批量 `source` 字段回填。
- 3 篇 inbox 笔记（滞留 8–9 周）的去留：升级进 knowledge/ 或标记放弃。需用户逐篇决策，且 cutlass/index.md:144–145 依赖其中两篇（`../../../../inbox/…` 路径经解析可达，非死链），移动须连带改链接——作为独立任务处理。
- `docs/knowledge/index.md:8-9` 与 `docs/projects/index.md:13-14` 指向 VitePress 根外的链接（`../../prompts/`、`../../.claude/README.md`）——属已知有意设计（ignoreDeadLinks 注释言明），维持现状。
- cutlass `cute-basics.md` 与 CUTLASS 01–03 的内容重叠（仓库自身已记录迁移意图）——内容级合并，非本次结构修复。
- 打印脚本 `--out` 参数把相对路径叠到 `print/` 之下产生 `print/print/` 的小缺陷（本会话实测踩到）——工具改进，另行处理。

## 验证步骤

1. 全量相对链接复扫（同审计脚本逻辑）：group-theory 0 死链 + 第 2 项 12 处 0 死链。
2. 侧栏目标核查：config.mts 每个 link 均解析到实际文件；重复段消除后总数对账。
3. `NODE_OPTIONS=--max-old-space-size=8192 npm run docs:build`（4GB 默认堆会 OOM）构建通过。
4. `git diff --check`；逐文件审阅 diff，确认第 1 项只改了链接文本、9 篇笔记的证明内容未被触碰。
5. 受保护的预存修改（证明扩写）在 diff 中可辨识且未被回退。

## 实施结果

待验证完成后填写。
