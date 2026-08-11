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
  cleanUrls: true,
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
        { text: 'Mathematics', link: '/knowledge/mathematics/' }
      ]}],
      '/projects/': [{ text: 'Projects', items: [{ text: 'Overview', link: '/projects/' }] }],
      '/ideas/': [{ text: 'Ideas', items: [{ text: 'Overview', link: '/ideas/' }] }],
      '/reading/': [{ text: 'Reading', items: [{ text: 'Overview', link: '/reading/' }] }],
      '/inbox/': [{ text: 'Inbox', items: [{ text: 'Overview', link: '/inbox/' }] }]
    },
    search: { provider: 'local' },
    socialLinks: [{ icon: 'github', link: 'https://github.com/YOUR_USERNAME/second-brain' }],
    editLink: {
      pattern: 'https://github.com/YOUR_USERNAME/second-brain/edit/main/docs/:path',
      text: 'Edit this page on GitHub'
    },
    outline: { level: [2, 3] },
    footer: { message: 'Knowledge belongs to me.', copyright: 'My Second Brain' }
  },
  markdown: {
    config: (md) => {
      md.use(texmath, { engine: katex, delimiters: 'dollars', katexOptions: { throwOnError: false } })
    }
  }
}),
  { mermaid: { theme: 'default' } }
)
