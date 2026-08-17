import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import texmath from 'markdown-it-texmath'
import katex from 'katex'

const isGitHubActions = process.env.GITHUB_ACTIONS === 'true'

export default withMermaid(
  defineConfig({
    title: 'My Second Brain',
    description: 'A personal knowledge system',
    lang: 'zh-CN',
  base: isGitHubActions ? '/second-brain/' : '/',
  lastUpdated: true,
  head: [],
    // AGENTS.md / README.md / templates/*.md 位于仓库根目录（docs/ 之外），
    // 但确实是真实存在的 Markdown，VitePress 无法把它们当站内页面解析，
    // 这里跳过死链检查。
    ignoreDeadLinks: true,
    themeConfig: {
    siteTitle: '🧠 My Second Brain',
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Knowledge', link: '/knowledge/' },
      { text: 'Projects', link: '/projects/' },
      { text: 'Ideas', link: '/ideas/' },
      { text: 'Reading', link: '/reading/' },
      { text: 'Inbox', link: '/inbox/' }
    ],
    sidebar: {
      '/knowledge/': [{ text: 'Knowledge', items: [
        { text: 'Overview', link: '/knowledge/' },
        { text: 'AI', link: '/knowledge/ai/' },
        { text: 'Autonomous Driving', link: '/knowledge/autonomous-driving/' },
        { text: 'Engineering', link: '/knowledge/engineering/' },
        { text: 'Mathematics', link: '/knowledge/mathematics/' },
        { text: 'PyTorch', link: '/knowledge/pytorch/' }
      ]}],
      '/projects/': [{ text: 'Projects', items: [{ text: 'Overview', link: '/projects/' }] }],
      '/ideas/': [{ text: 'Ideas', items: [{ text: 'Overview', link: '/ideas/' }] }],
      '/reading/': [{ text: 'Reading', items: [{ text: 'Overview', link: '/reading/' }] }],
      '/inbox/': [{ text: 'Inbox', items: [
        { text: 'Overview', link: '/inbox/' },
        { text: 'Second Brain 工作流', link: '/inbox/second-brain-workflow' },
        { text: 'ER 图', link: '/inbox/er-diagram' },
        { text: 'UML 类图', link: '/inbox/uml-class-diagram' },
        { text: 'DFD 数据流图', link: '/inbox/dfd' },
        { text: 'DDIM 论文', link: '/inbox/ddim-paper' },
        { text: 'uv Python 包管理', link: '/inbox/uv-python-package-manager' },
        { text: 'AI 开源项目源码精读', link: '/inbox/ai-open-source-source-reading' },
        { text: 'RVV 算子开发', link: '/inbox/rvv-operator-development' },
        { text: '迭代路线图', link: '/inbox/second-brain-iteration-roadmap' },
        { text: '分布式存储知识地图', link: '/inbox/distributed-storage-knowledge-map' },
        { text: 'RVV 算子设计大赛', link: '/inbox/rvv-operator-challenge' },
        { text: 'Fenwick Tree 加权采样', link: '/inbox/fenwick-tree-weighted-sampling' },
        { text: 'Semantica 知识图谱', link: '/inbox/semantica' }
      ]}]
    },
    search: { provider: 'local' },
    socialLinks: [{ icon: 'github', link: 'https://github.com/stevexiaofei/second-brain' }],
    editLink: {
      pattern: 'https://github.com/stevexiaofei/second-brain/edit/main/docs/:path',
      text: 'Edit this page on GitHub'
    },
    outline: { level: [2, 3] },
    footer: { message: 'Knowledge belongs to me.', copyright: 'My Second Brain' }
  },
  markdown: {
    config: (md) => {
      md.use(texmath, { engine: katex, delimiters: 'dollars', katexOptions: { throwOnError: false } })
    }
  },
  // vitepress-plugin-mermaid 通过 withMermaid(config) 增强 UserConfig，
  // mermaid 配置必须作为 defineConfig 的顶层字段，而不是第二个参数。
  mermaid: {
    theme: 'neutral',
    themeVariables: {
      fontSize: '20px',
      fontFamily: '"Inter","PingFang SC","Microsoft YaHei",system-ui,sans-serif',
      noteFontSize: '16px',
      labelFontSize: '18px',
      edgeLabelBackground: '#f1f5f9',
      tertiaryColor: '#f1f5f9'
    },
    startOnLoad: true,
    securityLevel: 'loose',
    flowchart: { useMaxWidth: false, htmlLabels: true, curve: 'basis' },
    sequence: { useMaxWidth: false, wrap: true, actorMargin: 50 },
    class: { useMaxWidth: false },
    state: { useMaxWidth: false },
    er: { useMaxWidth: false },
    gantt: { useMaxWidth: false }
  }
  })
)
