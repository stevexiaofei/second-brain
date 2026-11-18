---
title: 把某个主题打印成一份文档
type: guide
status: growing
tags: [Second-Brain, Workflow, Tooling, Print, PDF, VitePress, KaTeX, Mermaid]
created: 2026-10-07
updated: 2026-10-07
source: 本仓库实践；脚本见 scripts/print-notes.mjs
---

# 把某个主题打印成一份文档

## 一句话理解

**`npm run print -- <某个主题目录> --pdf`**，一条命令把这个目录下的所有笔记合并成一份文档：带篇名标题、可点击目录、每篇分页，并直接导出 PDF。产物在 `print/`（已 gitignore）。

```bash
# 例：把群论专题（14 篇）打成一份 PDF
npm run print -- docs/knowledge/mathematics/group-theory --pdf
```

## 适用场景

- 想离线通读某个主题（地铁上、纸上批注），但不想一篇文章一篇文章地开网页打印
- 想把某个专题发给别人（对方不需要装环境、不需要网络）
- 想给一段时间的学习留一个"可归档的成品"

## 为什么不直接合并 HTML

这是本方法最关键、也最反直觉的一点：**不能抓构建产物 `dist/*.html` 来拼**。

VitePress 站点里，数学公式与图表都是**浏览器端**渲染的，构建产物中它们只是占位符：

| 内容 | `dist/*.html` 里实际有什么 |
|---|---|
| KaTeX 公式 | `<!---->`（空注释） |
| 块级公式 | `<section><!----></section>` |
| Mermaid 图 | 空容器，等客户端脚本注入 `<svg>` |

所以无论"抓 HTML 再拼"还是"逐页浏览器打印"，拿到的都是残缺的。必须**从 Markdown 源码重新渲染**。

## 脚本做了什么

[scripts/print-notes.mjs](../../scripts/print-notes.mjs) 的流程：

| 步骤 | 做法 |
|---|---|
| 决定顺序 | 只取该目录**直接**的 `*.md`（不递归）；`index.md` 恒排最前，其余按文件名自然排序（`02` 在 `10` 前） |
| Markdown → HTML | 复用 **VitePress 自己的渲染管线**（`resolveConfig` + `createMarkdownRenderer`），因此公式、容器、代码块与站点完全一致；代码高亮用单一 shiki 主题并把颜色内联成行内样式 |
| Mermaid | `mermaid` 代码块改成 `<pre class="mermaid">`，页面内联一份 `mermaid.min.js`，打开时渲染 |
| 链接 | 批内 `.md` 互链 → 改成页内锚点 `#note-N`；指向批外的笔记 → 降级为纯文本（并打印告警） |
| 图片 | 笔记引用的本地图片复制到 `<输出名>.assets/`，并改写 `src`（`./images/x.png` 在输出目录本来会断链） |
| 字体 | KaTeX 的 CSS 与 woff2/woff 字体全部内联成 data URI——`file://` 下加载本地字体容易被浏览器拦住；`texmath.css` 一并内联 |
| 排版 | A4、篇名标题、自动目录（H2/H3）、每篇 `break-before: page`、公式/表格/代码块不跨页断开 |
| 出 PDF | 用本机 Chrome/Edge 的 `--headless --print-to-pdf`；`--virtual-time-budget` 保证 Mermaid 画完再打印 |

## 用法

```bash
npm run print -- <目录>                    # 只生成 HTML
npm run print -- <目录> --pdf              # 额外生成 PDF
npm run print -- <目录> --out 名字.html     # 指定输出文件名
npm run print -- <目录> --title "标题"      # 指定封面标题
npm run print -- <目录> --no-toc           # 不生成目录
```

产物结构：

```
print/
├── <目录名>.html      # 单文件、字体/脚本内联，脱网可看
├── <目录名>.pdf       # A4，可直接打印
└── <目录名>.assets/   # 该主题笔记里引用的图片
```

依赖都在 `devDependencies` 里（`katex`、`markdown-it-texmath`、`mermaid`、`vitepress`）；能找到本机 Chrome/Edge 就能出 PDF，找不到时只保留 HTML 并提示。

## 怎么用在任何主题上

不需要改配置。三步：

1. 把该主题的 `.md` 放在**同一个目录**里（脚本不递归扫描子目录）
2. 决定篇序：想让某篇排最前就叫 `index.md`；其余用 `01-`、`02-` 前缀，自然排序会照做
3. 跑 `npm run print -- <目录> --pdf`

## 边界与注意事项

- 只合并该目录**直接**的 `*.md`，子目录里的笔记要在父目录也放一份（或单独对子目录跑一次）
- 指向批外的 `.md` 链接会降级为纯文本并告警——告警列表值得看一眼，能顺手发现断链
- 图片会**复制**到 `print/`，所以产物体积可能比源目录大几倍
- 产物不进版本库（`.gitignore` 已含 `print/`）
- 目录里若有 `.vitepress`、`public` 之类的目录不受影响（本来就不在扫描范围）

## 踩过的坑

1. **构建产物里公式是空的**。第一版想直接从 `dist/` 抓已渲染的 HTML，结果发现 KaTeX 与 Mermaid 都是客户端渲染，静态文件里只有占位符。→ 只能从 Markdown 源码重新渲染。
2. **markdown-it 无法直接 `require`**。VitePress 把 markdown-it 打包进了自己的 dist，`node_modules/markdown-it` 并不存在。→ 用 `resolveConfig` + `createMarkdownRenderer` 复用站点管线，反而让渲染结果与站点 100% 一致。
3. **链接改写必须在 token 层做**。`link_open` 里的 `href` 还是 `./x.md`，渲染后会被 VitePress 改写成 `/x.html`——那时再按 `.md` 查表就对不上了。
4. **`file://` 下 ES module 被 CORS 拦**。所以 Mermaid 用经典脚本（`mermaid.min.js`）内联，换成 `import` 会失败。
5. **`file://` 下本地字体加载受限**。KaTeX 的 `katex.min.css` 里 `url(fonts/*.woff2)` 若保持相对路径，公式会掉字体。→ 全部内联成 data URI。
6. **Chrome headless 打完 PDF 不退出**。最初用 `spawnSync` 等它结束，结果每次卡满超时；PDF 其实早就写好了。→ 改成"后台启动 → 轮询 PDF 落盘且体积稳定 → 杀整个进程组"，同一份文档从 90 秒降到十几秒。
7. **笔记搬家后相对链接会断**。把笔记移进子目录时，`../` 的层数变了但链接没跟着改，一次搬迁曾产生 26 条断链。→ 搬完就把每个 `](./x.md)` 解析成真实路径扫一遍，确认目标存在。

> 验证手法（改脚本时复用）：用 `--dump-dom` 在相同参数下检查 `<pre class="mermaid">` 里是否出现 `<svg data-processed="true">`；PDF 内容流是压缩的，直接搜文本/颜色不可靠，可对比"去掉 Mermaid 脚本"前后产物是否变化。

## Related

- [Second Brain 使用工作流](./second-brain-workflow.md) — 上游：笔记怎么从 inbox 沉淀到 knowledge；本篇是其下游的"输出/分发"环节
- [群论与抽象代数专题](../knowledge/mathematics/group-theory/index.md) — 本方法的首次大规模应用（14 篇）
- [Second Brain 迭代路线图](./second-brain-iteration-roadmap.md) — 发布治理相关的后续计划

## References

- 脚本：[scripts/print-notes.mjs](../../scripts/print-notes.mjs)；用法另见仓库 [README](../../README.md)
- Chromium 的 `--virtual-time-budget` / `--print-to-pdf` 命令行开关
