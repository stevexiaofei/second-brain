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
      '/knowledge/': [
        { text: 'Knowledge', items: [
          { text: 'Overview', link: '/knowledge/' }
        ]},
        { text: 'AI', items: [
          { text: 'Overview', link: '/knowledge/ai/' },
          { text: 'GRPO', link: '/knowledge/ai/grpo' },
          { text: 'PPO', link: '/knowledge/ai/ppo' },
          { text: 'DDIM 论文', link: '/knowledge/ai/ddim-paper' },
          { text: 'AI 开源源码精读', link: '/knowledge/ai/ai-open-source-source-reading' },
          { text: 'Semantica 知识图谱', link: '/knowledge/ai/semantica' },
          { text: 'nanobot 阅读指南', link: '/knowledge/ai/nanobot-source-reading-guide' },
          { text: 'nanobot 架构总览', link: '/knowledge/ai/nanobot-architecture-overview' },
          { text: 'nanobot AgentLoop/Runner', link: '/knowledge/ai/nanobot-agentloop-runner' },
          { text: 'nanobot ContextBuilder', link: '/knowledge/ai/nanobot-contextbuilder' },
          { text: 'nanobot Providers', link: '/knowledge/ai/nanobot-providers-registry' },
          { text: 'nanobot Tool Registry', link: '/knowledge/ai/nanobot-tool-registry' },
          { text: 'nanobot Channel Manager', link: '/knowledge/ai/nanobot-channel-manager' }
        ]},
        { text: 'Autonomous Driving', items: [
          { text: 'Overview', link: '/knowledge/autonomous-driving/' },
          { text: '训练数据管理', link: '/knowledge/autonomous-driving/data-management' }
        ]},
        { text: 'Distributed Systems', items: [
          { text: 'Overview', link: '/knowledge/distributed-systems/' },
          { text: '核心理论基础', link: '/knowledge/distributed-systems/distributed-systems-foundations' }
        ]},
        { text: 'Engineering', items: [
          { text: 'Overview', link: '/knowledge/engineering/' },
          { text: 'RVV 算子开发', link: '/knowledge/engineering/rvv-operator-development' },
          { text: 'RVV 算子设计大赛', link: '/knowledge/engineering/rvv-operator-challenge' },
          { text: '分布式存储知识地图', link: '/knowledge/engineering/distributed-storage-knowledge-map' },
          { text: 'Fenwick Tree 加权采样', link: '/knowledge/engineering/fenwick-tree-weighted-sampling' },
          { text: 'uv Python 包管理', link: '/knowledge/engineering/uv-python-package-manager' },
          { text: 'ER 图', link: '/knowledge/engineering/er-diagram' },
          { text: 'UML 类图', link: '/knowledge/engineering/uml-class-diagram' },
          { text: 'DFD 数据流图', link: '/knowledge/engineering/dfd' },
          { text: 'Git on NFS', link: '/knowledge/engineering/git-on-nfs' }
        ]},
        { text: 'Mathematics', items: [
          { text: 'Overview', link: '/knowledge/mathematics/' }
        ]},
        { text: 'PyTorch', items: [
          { text: 'Overview', link: '/knowledge/pytorch/' }
        ]}
      ],
      '/projects/': [{ text: 'Projects', items: [
        { text: 'Overview', link: '/projects/' },
        { text: 'Second Brain 工作流', link: '/projects/second-brain-workflow' },
        { text: '迭代路线图', link: '/projects/second-brain-iteration-roadmap' }
      ]}],
      '/ideas/': [{ text: 'Ideas', items: [{ text: 'Overview', link: '/ideas/' }] }],
      '/reading/': [{ text: 'Reading', items: [{ text: 'Overview', link: '/reading/' }] }],
      '/inbox/': [{ text: 'Inbox', items: [
        { text: 'Overview', link: '/inbox/' }
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
