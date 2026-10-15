#!/usr/bin/env node
// print-notes.mjs — 把一个知识目录的全部 Markdown 笔记合成一份自包含的可打印 HTML（可选 PDF）。
//
// 用法：
//   npm run print -- docs/knowledge/mathematics/group-theory [--pdf] [--out <file>] [--title <text>] [--no-toc]
//
// 产物：print/<目录名>.html（自包含：KaTeX CSS+字体 data URI、mermaid.js、打印样式全内联）
//       加 --pdf 时另产出同名 .pdf（headless Chrome 打印）。
//
// 设计要点（与 AGENTS.md 的"Markdown 是唯一事实源"一致）：只读 Markdown 与
// docs/.vitepress/config.mts，不改动站点；复用 VitePress 的 markdown 渲染管线，
// 保证公式/代码高亮与站点一致。

import fs from 'node:fs'
import path from 'node:path'
import { spawnSync } from 'node:child_process'
import { pathToFileURL, fileURLToPath } from 'node:url'
import { createRequire } from 'node:module'
import { resolveConfig, createMarkdownRenderer } from 'vitepress'

const require = createRequire(import.meta.url)
const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')

// ---------- CLI ----------

function parseArgv(argv) {
  const opts = { dir: null, out: null, pdf: false, title: null, toc: true }
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (a === '--pdf') opts.pdf = true
    else if (a === '--no-toc') opts.toc = false
    else if (a === '--out') opts.out = argv[++i]
    else if (a === '--title') opts.title = argv[++i]
    else if (a === '-h' || a === '--help') usage()
    else if (a.startsWith('-')) {
      console.error(`未知选项: ${a}`)
      usage()
    } else if (opts.dir === null) opts.dir = a
    else {
      console.error(`多余的位置参数: ${a}`)
      usage()
    }
  }
  if (opts.dir === null) usage()
  return opts
}

function usage() {
  console.error(`用法: npm run print -- <dir> [--pdf] [--out <file>] [--title <text>] [--no-toc]

  <dir>        笔记目录（相对仓库根或绝对路径），只合并该目录直接的 *.md
  --pdf        额外生成同名 PDF（默认只出 HTML）
  --out <file> 输出 HTML 路径（默认 print/<目录名>.html）
  --title      封面标题（默认取 index.md frontmatter title，否则目录名）
  --no-toc     不生成目录`)
  process.exit(1)
}

const opts = parseArgv(process.argv.slice(2))
const dirAbs = path.isAbsolute(opts.dir) ? opts.dir : path.join(REPO, opts.dir)
if (!fs.existsSync(dirAbs) || !fs.statSync(dirAbs).isDirectory()) {
  console.error(`目录不存在: ${dirAbs}`)
  process.exit(1)
}

// ---------- 1. 收集与排序：index.md 最先，其余自然排序 ----------

const files = fs
  .readdirSync(dirAbs)
  .filter((f) => f.endsWith('.md'))
  .sort((a, b) => {
    if (a === 'index.md') return -1
    if (b === 'index.md') return 1
    return a.localeCompare(b, 'en', { numeric: true, sensitivity: 'base' })
  })
if (files.length === 0) {
  console.error(`目录中没有 .md 文件: ${dirAbs}`)
  process.exit(1)
}
console.log(`共 ${files.length} 篇: ${files.join(', ')}`)

// ---------- 2. 渲染管线（与站点同款，单一 shiki 主题 → 内联颜色） ----------

const DOCS = path.join(REPO, 'docs')
const cfg = await resolveConfig(DOCS, 'build')
const md = await createMarkdownRenderer(
  DOCS,
  { ...cfg.markdown, theme: 'github-light' },
  cfg.site.base || '/',
  cfg.logger
)
const mermaidCfg = { ...(cfg.userConfig?.mermaid ?? {}), startOnLoad: false, suppressErrors: true }

// withMermaid 注入的 fence 规则输出 Vue 组件（静态 HTML 无法用），覆盖之：
// mermaid fence → <pre class="mermaid">，其余 fence 交给原规则（保留 shiki 高亮）。
const origFence = md.renderer.rules.fence
md.renderer.rules.fence = (tokens, idx, options, env, self) => {
  const t = tokens[idx]
  if ((t.info || '').trim() === 'mermaid') {
    const raw = t.content.replace(/<\/pre/gi, '<\\/pre') // 唯一能截断 <pre> 的序列
    return `<pre class="mermaid">${raw}</pre>\n`
  }
  return origFence(tokens, idx, options, env, self)
}

function stripFrontmatter(src) {
  const m = /^---\r?\n[\s\S]*?\r?\n---\r?\n?/.exec(src)
  const body = m ? src.slice(m[0].length) : src
  // 去掉正文首个 H1（篇名由脚本统一插入，避免重复）
  return body.replace(/^#\s+.+\r?\n+/, '')
}

// env.links 与渲染后 href 的公共形态：去扩展名、保留 ./ 或 ../ 前缀。
// 统一归一化为「无前缀、无扩展名」的 basename 再比对。
function normLinkKey(href) {
  return href.replace(/\.md$/i, '').replace(/^\.\//, '').replace(/^\/__print__\//, '')
}

// 批内链接解析：先按文件名精确匹配，再去掉开头的 "NN-" 前缀重试
//（index.md 的推荐阅读顺序曾以无前缀名链接，容错保证仍可解析）。
function resolveBatchLink(href, fileSet) {
  let base = path.posix.basename(href.replace(/\.md$/i, ''))
  if (fileSet.has(base)) return base
  const unprefixed = base.replace(/^\d+-/, '')
  for (const f of fileSet) {
    if (f.replace(/\.md$/i, '').replace(/^\d+-/, '') === unprefixed) return f
  }
  return null
}

const warnings = []
const sections = []
const tocEntries = []
const fileSet = new Set(files)

for (let i = 0; i < files.length; i++) {
  const file = files[i]
  const srcRaw = fs.readFileSync(path.join(dirAbs, file), 'utf8')
  const fmTitle = /^title:\s*(.+)\s*$/m.exec(srcRaw.match(/^---\r?\n([\s\S]*?)\r?\n---/)?.[1] ?? '')
  const src = stripFrontmatter(srcRaw)

  const env = { path: `/__print__/${file}` }
  const tokens = md.parse(src, env)

  // 链接改写要在 token 层做：link_open 携带原始 .md href，渲染期被 VitePress
  // 规则改写成 .html——那时再按 .md 键查表就对不上了（首轮实现踩过这个坑）。
  // link_open 位于 inline token 的 children 里，需要递归遍历。
  const deadAttrs = [['class', 'deadlink']]
  const rewriteLinkTokens = (toks) => {
    for (const { type, attrs, children } of toks) {
      if (children) rewriteLinkTokens(children)
      if (type !== 'link_open') continue
      const href = attrs.find(([n]) => n === 'href')?.[1] ?? ''
      if (!href || href.startsWith('#') || /^(https?:|\/\/)/.test(href)) continue
      const resolved = resolveBatchLink(href, fileSet)
      if (resolved) {
        attrs[attrs.findIndex(([n]) => n === 'href')] = ['href', `#note-${files.indexOf(resolved)}`]
      } else {
        attrs.length = 0
        attrs.push(...deadAttrs)
        warnings.push(`${file}: 批外链接（降级为纯文本）: ${href}`)
      }
    }
  }
  rewriteLinkTokens(tokens)

  let html = md.renderer.render(tokens, md.options, env)

  // 标题 id 加前缀避免跨篇 slug 冲突；同篇 #锚点 同步改写
  const p = `note-${i}`
  html = html.replace(/(<h[123] id=")([^"]+)(")/g, `$1${p}-$2$3`)
  html = html.replace(/href="#([^"]+)"/g, `href="#${p}-$1"`)

  // 原文自带一个 H1，与插入的篇名重复 → 去掉首个 H1（含其 header-anchor）
  html = html.replace(/^\s*<h1 id="[^"]*"[^>]*>[\s\S]*?<\/h1>\s*/i, '')

  sections.push(
    `<section class="note" id="note-${i}">\n<h1 class="note-title">${fmTitle?.[1]?.trim() ?? file.replace(/\.md$/, '')}</h1>\n${html}\n</section>`
  )

  if (opts.toc) {
    // 从渲染后的 HTML 收集 h2/h3（h1 即篇名，已进目录顶层）
    const re = /<h([23]) id="([^"]+)"[^>]*>([\s\S]*?)<\/h\1>/g
    let m2
    while ((m2 = re.exec(html))) {
      const inner = m2[3].replace(/<a[^>]*>[\s\S]*?<\/a>/g, '').replace(/<[^>]+>/g, '').trim()
      tocEntries.push({ level: +m2[1], id: m2[2], text: inner, note: i })
    }
  }
}

// ---------- 3. 目录 ----------

let tocHtml = ''
if (opts.toc) {
  const rows = []
  files.forEach((f, i) => {
    const srcRaw = fs.readFileSync(path.join(dirAbs, f), 'utf8')
    const fmTitle = /^title:\s*(.+)\s*$/m.exec(srcRaw.match(/^---\r?\n([\s\S]*?)\r?\n---/)?.[1] ?? '')
    rows.push(`<li class="toc-note"><a href="#note-${i}">${fmTitle?.[1]?.trim() ?? f.replace(/\.md$/, '')}</a></li>`)
    for (const e of tocEntries.filter((t) => t.note === i)) {
      rows.push(`<li class="toc-h${e.level}"><a href="#${e.id}">${e.text}</a></li>`)
    }
  })
  tocHtml = `<nav class="toc"><h2 class="toc-title">目录</h2><ul>${rows.join('')}</ul></nav>`
}

// ---------- 4. 自包含资源内联 ----------

function readDep(...p) {
  return fs.readFileSync(path.join(REPO, 'node_modules', ...p))
}

function inlineKatexCss() {
  let css = readDep('katex', 'dist', 'katex.min.css').toString('utf8')
  const fontDir = path.join(REPO, 'node_modules', 'katex', 'dist', 'fonts')
  css = css.replace(/url\(fonts\/([^)]+)\)/g, (m, f) => {
    const fp = path.join(fontDir, f)
    if (!fs.existsSync(fp)) return m
    const mime = f.endsWith('.woff2') ? 'font/woff2' : f.endsWith('.woff') ? 'font/woff' : 'application/octet-stream'
    return `url(data:${mime};base64,${fs.readFileSync(fp).toString('base64')})`
  })
  return css
}

const katexCss = inlineKatexCss()
const texmathCss = readDep('markdown-it-texmath', 'css', 'texmath.css').toString('utf8')
const mermaidJs = readDep('mermaid', 'dist', 'mermaid.min.js').toString('utf8')

const printCss = `
@page { size: A4; margin: 18mm 16mm; }
* { box-sizing: border-box; }
body { font-family: "Microsoft YaHei", "PingFang SC", system-ui, sans-serif; line-height: 1.7; max-width: 210mm; margin: 0 auto; padding: 12mm 4mm; color: #1f2937; }
h1, h2, h3, h4 { line-height: 1.3; }
h1, h2, h3 { break-after: avoid; }
pre, table, .katex-display { break-inside: avoid; }
.deadlink { color: #6b7280; text-decoration: underline dotted; }
.note { break-before: page; }
.note:first-of-type { break-before: auto; }
.note-title { border-bottom: 2px solid #e5e7eb; padding-bottom: .3em; }
.toc { break-after: page; }
.toc ul { list-style: none; padding-left: 0; }
.toc li { margin: .35em 0; }
.toc-h2, .toc-h3 { padding-left: 2em; font-size: .92em; color: #374151; }
.toc a { text-decoration: none; color: inherit; }
.header-anchor { display: none; }
pre.shiki { background: #f6f8fa; padding: 12px 14px; border-radius: 6px; overflow-x: auto; font-size: .88em; }
pre code { font-family: Consolas, "Courier New", monospace; }
section.eqn, section.eqno, eqn { break-inside: avoid; }
eqn { display: block; text-align: center; margin: .8em 0; overflow-x: auto; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #d1d5db; padding: 6px 10px; text-align: left; }
th { background: #f3f4f6; }
button.copy, .copy, .lang { display: none !important; }
@media print { body { padding: 0; max-width: none; } }
`

const bootstrap = `
document.addEventListener('DOMContentLoaded', async () => {
  const nodes = [...document.querySelectorAll('pre.mermaid')]
  if (nodes.length && window.mermaid) {
    try {
      mermaid.initialize(${JSON.stringify(mermaidCfg)})
      await mermaid.run({ nodes })
    } catch (e) { console.error('mermaid render failed:', e) }
  }
  document.documentElement.dataset.printReady = 'true'
})
`

const title =
  opts.title ??
  (files.includes('index.md')
    ? (/^title:\s*(.+)\s*$/m.exec(
        fs.readFileSync(path.join(dirAbs, 'index.md'), 'utf8').match(/^---\r?\n([\s\S]*?)\r?\n---/)?.[1] ?? ''
      )?.[1]?.trim() ?? path.basename(dirAbs))
    : path.basename(dirAbs))

const htmlDoc = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>${title}</title>
<style>${katexCss}</style>
<style>${texmathCss}</style>
<style>${printCss}</style>
</head>
<body>
<h1 class="doc-title">${title}</h1>
${tocHtml}
${sections.join('\n')}
<script>${mermaidJs}</script>
<script>${bootstrap}</script>
</body>
</html>`

// ---------- 5. 输出 ----------

const outPath = opts.out
  ? path.isAbsolute(opts.out)
    ? opts.out
    : path.join(REPO, 'print', opts.out)
  : path.join(REPO, 'print', `${path.basename(dirAbs)}.html`)
fs.mkdirSync(path.dirname(outPath), { recursive: true })
fs.writeFileSync(outPath, htmlDoc)
console.log(`HTML: ${outPath} (${(htmlDoc.length / 1024 / 1024).toFixed(1)} MB)`)

// ---------- 6. PDF（可选，headless Chrome） ----------

if (opts.pdf) {
  const candidates =
    process.platform === 'win32'
      ? [
          process.env.CHROME_PATH,
          'C:/Program Files/Google/Chrome/Application/chrome.exe',
          'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
          path.join(process.env.LOCALAPPDATA ?? '', 'Google/Chrome/Application/chrome.exe'),
          'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
          'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'
        ]
      : process.platform === 'darwin'
        ? ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome']
        : ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser']
  const chrome = candidates.find((c) => c && fs.existsSync(c))
  if (!chrome) {
    console.error('未找到 Chrome/Edge，无法生成 PDF（HTML 已保留）。可设 CHROME_PATH 指定浏览器。')
    process.exit(1)
  }
  const pdfPath = outPath.replace(/\.html$/, '.pdf')
  const r = spawnSync(
    chrome,
    [
      '--headless=new',
      '--disable-gpu',
      '--no-sandbox',
      '--no-pdf-header-footer',
      '--run-all-compositor-stages-before-draw',
      '--virtual-time-budget=20000',
      `--print-to-pdf=${pdfPath}`,
      pathToFileURL(outPath).href
    ],
    { stdio: 'inherit', timeout: 120000 }
  )
  if (r.status === 0 && fs.existsSync(pdfPath)) {
    console.log(`PDF:  ${pdfPath} (${(fs.statSync(pdfPath).size / 1024 / 1024).toFixed(1)} MB)`)
  } else {
    console.error(`PDF 生成失败（exit=${r.status}），HTML 已保留: ${outPath}`)
    process.exit(1)
  }
}

if (warnings.length) {
  console.warn(`\n${warnings.length} 条告警:`)
  for (const w of warnings) console.warn(`  - ${w}`)
}
