# Mermaid 流程图配色规范(权威源)

> 本文件定义全站 Mermaid 流程图的**统一配色**。所有 `docs/**/*.md` 中的 mermaid 代码块必须使用本文件定义的 classDef,确保视觉一致。

## 5 个核心 classDef

每个 classDef 对应原手写 HTML 流程图中的一种语义角色。复制以下整段,粘贴到 mermaid 代码块末尾的 `classDef ... end` 区域。

```mermaid
classDef step     fill:#eef2ff,stroke:#c7d2fe,color:#312e81,stroke-width:1.5px
classDef action   fill:#fff7ed,stroke:#fdba74,color:#7c2d12,stroke-width:1.5px
classDef decide   fill:#fef3c7,stroke:#fcd34d,color:#78350f,stroke-width:1.5px
classDef branchNo fill:#f0fdf4,stroke:#86efac,color:#166534,stroke-width:1.5px
classDef branchYes fill:#eef2ff,stroke:#c7d2fe,color:#3730a3,stroke-width:1.5px
```

## 角色对照

| classDef 名 | 语义角色 | 配色(浅色背景 / 边框 / 文字) | 原 CSS 类 |
|---|---|---|---|
| `step` | 普通步骤节点 | `#eef2ff` / `#c7d2fe` / `#312e81`(淡紫) | `.d-node` |
| `action` | 关键动作/可变状态节点 | `#fff7ed` / `#fdba74` / `#7c2d12`(淡橙) | `.d-node-active` |
| `decide` | 决策菱形 | `#fef3c7` / `#fcd34d` / `#78350f`(淡黄) | `.d-node-decide` |
| `branchNo` | 分支:否/等待 | `#f0fdf4` / `#86efac` / `#166534`(淡绿) | `.d-label`(绿) |
| `branchYes` | 分支:是/收尾 | `#eef2ff` / `#c7d2fe` / `#3730a3`(淡靛) | `.d-label`(靛) |

## 节点形状约定

| 含义 | Mermaid 语法 |
|---|---|
| 普通矩形节点 | `id["文本"]` |
| 决策菱形 | `id{"文本"}` |
| 圆角节点(用于入口/收尾) | `id("文本")` |
| 圆柱(用于数据存储) | `id[(文本)]` |
| 子图分组 | `subgraph 标题 ... end` |
| 节点内换行 | `id["第一行<br/>第二行"]` |
| 边标签 | `A -- "是" --> B` |

## 节点字号

VitePress 的 `theme` 在 [config.mts](../../.vitepress/config.mts) 已配置 `fontSize: 20px` / `noteFontSize: 16px`,无需在每张图重复声明。

## 反例(避免)

- **不要**再用 `<div class="diagram"><span class="d-node">` 拼流程图——字符箭头无法精准对齐
- **不要**在每张图里重复发明配色——直接复制本文件的 5 行 classDef
- **不要**用 ASCII 画流程图(AGENTS.md 已禁)
- **不要**在 Mermaid 节点里直接写未转义的双引号(用 `\"` 或单引号)

## 维护

修改本文件后,需同步更新:
- [AGENTS.md](../../AGENTS.md) 中的"Diagrams"章节
- 所有已迁移的 mermaid 图无需改动(它们引用的是颜色值,不是变量名)
