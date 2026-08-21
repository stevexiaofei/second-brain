---
title: PyTorch 迁入 AI Systems 重构计划
type: plan
status: implemented
created: 2026-08-21
updated: 2026-08-21
scope: move PyTorch knowledge topic under AI Systems
---

# PyTorch 迁入 AI Systems 重构计划

## 范围与安全边界

- 审计基线为提交 `49b12b9`；工作区当前已有上一轮仓库维护的未提交修改，必须保留并随迁移更新其中涉及 PyTorch 路径的内容。
- 迁移目标是 `docs/knowledge/ai/systems/pytorch/`，而不是 `docs/knowledge/ai/` 根目录。
- 保持 PyTorch 作为完整独立专题：不拆分其框架、编译栈、分布式训练和 `torch-compile-wiki/` 系列，也不改写其主题内容。
- 不保留旧的 `docs/knowledge/pytorch/` 占位页或重复副本；Git 历史保留移动记录。
- 不提交或推送。

## 已确认的依据

1. `docs/knowledge/ai/index.md` 将 AI 分为 Foundations 与 Systems；Systems 用于框架、编译器、运行时、kernel、分布式与 Agent 基础设施。
2. `docs/knowledge/ai/systems/index.md` 已把 PyTorch 作为“框架与编译栈源码知识”关联项。
3. PyTorch 专题当前包含框架结构、autograd、FX、`torch.compile`、Inductor、AOTAutograd、分布式与源码阅读；这些内容与 FlashAttention 的 PyTorch / ATen 接入笔记存在直接连接。
4. 当前 PyTorch 入口有明确的站点导航与跨领域引用：Knowledge 总索引、首页、AI Systems、Learning、Engineering、Distributed Systems、FlashAttention、路线图及 VitePress sidebar。

## 实施边界

### 1. 移动专题

- 使用版本控制感知的移动，将整个 `docs/knowledge/pytorch/` 移至 `docs/knowledge/ai/systems/pytorch/`。
- 保持所有文件名和 `torch-compile-wiki/` 子目录不变。
- 迁移后的专题首页仍为 `docs/knowledge/ai/systems/pytorch/index.md`。

### 2. 修复专题内部与跨专题链接

- 更新 PyTorch 首页到 FlashAttention 的链接，使其从新位置指向同级的 `flash-attention/`。
- 保留 `torch-compile-wiki/` 内部的相对链接：其相对拓扑不变。
- 更新下列外部引用为新位置：
  - `docs/knowledge/ai/systems/index.md`
  - `docs/knowledge/ai/systems/ai-open-source-source-reading.md`
  - FlashAttention 专题的 PyTorch / ATen 相关笔记和专题索引
  - `docs/knowledge/learning/code-reading/index.md`
  - `docs/knowledge/distributed-systems/distributed-systems-foundations.md`
  - `docs/knowledge/engineering/rvv-operator-development.md`
  - `docs/knowledge/engineering/distributed-storage-knowledge-map.md`
  - `docs/index.md`
  - 当前路线图中的现状描述。
- 保留旧维护计划中的旧路径作为历史审计记录，不将其改写为当前路径。

### 3. 更新信息架构和站点导航

- 在 `docs/knowledge/ai/index.md` 的 Systems 区增加 PyTorch 专题入口。
- 在 `docs/knowledge/ai/systems/index.md` 的学习路线中将 PyTorch 从 Related 提升为明确的专题入口。
- 从 `docs/knowledge/index.md` 移除顶层 PyTorch 区域入口。
- 从 `docs/index.md` 移除顶层 PyTorch 入口，避免与其作为 AI Systems 子专题的归属重复。
- 将 `docs/.vitepress/config.mts` 中独立的 PyTorch sidebar 组迁入 AI 区域，作为与 FlashAttention、nanobot 并列的折叠专题组；更新全部 `/knowledge/pytorch/...` 路由。

### 4. 记录与元数据

- 在移动和实质修改的索引页更新 `updated`，但不对仅路径随 Git 移动的每篇笔记做无意义日期迁移。
- 在本计划中记录实施结果与验证结论，并将状态更新为 `implemented`。

## 预期导航结构

```text
knowledge/
└── ai/
    ├── foundations/
    └── systems/
        ├── flash-attention/
        ├── nanobot/
        └── pytorch/
            ├── index.md
            ├── pytorch-*.md
            └── torch-compile-wiki/
```

## 验证计划

1. 搜索仓库中未作为历史记录保留的 `knowledge/pytorch`、`/knowledge/pytorch` 和旧相对路径引用；预期为零。
2. 对迁移后的 Markdown 运行相对链接扫描，并单独识别代码示例造成的误报。
3. 验证 VitePress sidebar 全部目标存在。
4. 运行 `git diff --check`。
5. 运行 `npm run docs:build`；若环境仍阻断命令，明确记录为未执行而非通过。
6. 审查最终 Git 状态，确认没有删除内容、没有旧目录残留，且没有提交或推送。

## 实施结果

- 已将整个 PyTorch 专题从 `docs/knowledge/pytorch/` 移至 `docs/knowledge/ai/systems/pytorch/`，保留原有文件名和 `torch-compile-wiki/` 阅读路径。
- 已将 PyTorch 专题注册到 AI 总览、AI Systems 学习路线与 VitePress 的 AI sidebar；原有顶层 PyTorch sidebar 和顶层索引入口已移除。
- 已更新 FlashAttention、Learning、Engineering、Distributed Systems、首页和路线图中的当前路径引用；上一轮维护计划和本计划中的旧路径保留为历史记录。
- 未移动、删除或拆分任何笔记内容，未创建兼容性副本，未提交或推送。
- 验证结果将在完成路径扫描、sidebar 目标检查、`git diff --check` 和文档构建后补充。
