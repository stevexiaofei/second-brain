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
          { text: 'Foundations', link: '/knowledge/ai/foundations/' },
          { text: 'Reinforcement Learning', link: '/knowledge/ai/foundations/reinforcement-learning/' },
          { text: '状态价值与 Bellman 方程', link: '/knowledge/ai/foundations/reinforcement-learning/state-values-and-bellman-equation' },
          { text: 'PPO', link: '/knowledge/ai/foundations/reinforcement-learning/ppo' },
          { text: 'GRPO', link: '/knowledge/ai/foundations/reinforcement-learning/grpo' },
          { text: 'Diffusion Models', link: '/knowledge/ai/foundations/diffusion/' },
          { text: 'DDIM 论文', link: '/knowledge/ai/foundations/diffusion/ddim-paper' },
          { text: 'Reasoning & Inference', link: '/knowledge/ai/foundations/reasoning/' },
          { text: '思维链论文详解', link: '/knowledge/ai/foundations/reasoning/chain-of-thought-papers' },
          { text: 'AI Systems', link: '/knowledge/ai/systems/' },
          { text: 'AI 开源源码精读', link: '/knowledge/ai/systems/ai-open-source-source-reading' },
          { text: 'CUTLASS / CuTe', link: '/knowledge/ai/systems/cutlass/' },
          { text: 'Semantica 知识图谱', link: '/knowledge/ai/systems/semantica' },
          { text: 'Attention 头变体：MHA/MQA/GQA/MLA', link: '/knowledge/ai/attention-head-variants' }
        ]},
        { text: 'CUTLASS / CuTe', collapsed: true, items: [
          { text: '专题总览', link: '/knowledge/ai/systems/cutlass/' },
          { text: '01 Tensor 与 Layout', link: '/knowledge/ai/systems/cutlass/01-cute-tensor-and-layout' },
          { text: '02 Copy 与线程分区', link: '/knowledge/ai/systems/cutlass/02-cute-copy-and-thread-partition' },
          { text: '03 TiledMMA 与 fragment', link: '/knowledge/ai/systems/cutlass/03-cute-tiled-mma' },
          { text: '04 GEMM 数据流', link: '/knowledge/ai/systems/cutlass/04-cute-gemm-pipeline' }
        ]},
        { text: 'FlashAttention', collapsed: true, items: [
          { text: '专题总览', link: '/knowledge/ai/systems/flash-attention/' },
          { text: '阅读导览', link: '/knowledge/ai/systems/flash-attention/flash-attention-reading-guide' },
          { text: '论文精读', link: '/knowledge/ai/systems/flash-attention/flashattention-paper-series' },
          { text: 'CUTE 入门', link: '/knowledge/ai/systems/flash-attention/cute-basics' },
          { text: 'CUTE TiledMMA（已归档至 CUTLASS）', link: '/knowledge/ai/systems/cutlass/03-cute-tiled-mma' },
          { text: '术语表与状态', link: '/knowledge/ai/systems/flash-attention/flash-attention-glossary-and-state-table' },
          { text: '系统地图', link: '/knowledge/ai/systems/flash-attention/flash-attention-system-map' },
          { text: '源码精读', link: '/knowledge/ai/systems/flash-attention/flash-attention-source-reading' },
          { text: '接口与 Autograd', link: '/knowledge/ai/systems/flash-attention/flash-attention-interface-and-autograd' },
          { text: 'Kernel 与 Launch', link: '/knowledge/ai/systems/flash-attention/flash-attention-kernel-and-launch' },
          { text: 'Kernel 细节', link: '/knowledge/ai/systems/flash-attention/flash-attention-kernel-details' },
          { text: 'PyTorch ATen 接入', link: '/knowledge/ai/systems/flash-attention/flash-attention-pytorch-aten-integration' }
        ]},
        { text: 'nanobot', collapsed: true, items: [
          { text: '专题总览', link: '/knowledge/ai/systems/nanobot/' },
          { text: '阅读指南', link: '/knowledge/ai/systems/nanobot/nanobot-source-reading-guide' },
          { text: '架构总览', link: '/knowledge/ai/systems/nanobot/nanobot-architecture-overview' },
          { text: 'AgentLoop / Runner', link: '/knowledge/ai/systems/nanobot/nanobot-agentloop-runner' },
          { text: 'ContextBuilder', link: '/knowledge/ai/systems/nanobot/nanobot-contextbuilder' },
          { text: 'Tool Registry', link: '/knowledge/ai/systems/nanobot/nanobot-tool-registry' },
          { text: 'Providers Registry', link: '/knowledge/ai/systems/nanobot/nanobot-providers-registry' },
          { text: 'Channel Manager', link: '/knowledge/ai/systems/nanobot/nanobot-channel-manager' }
        ]},
        { text: 'PyTorch', collapsed: true, items: [
          { text: '专题总览', link: '/knowledge/ai/systems/pytorch/' },
          { text: '01 架构总览', link: '/knowledge/ai/systems/pytorch/torch-compile-wiki/01-architecture-overview' },
          { text: '02 torch.compile 入口', link: '/knowledge/ai/systems/pytorch/torch-compile-wiki/02-torch-compile-entry' },
          { text: '03 TorchDynamo 前端', link: '/knowledge/ai/systems/pytorch/torch-compile-wiki/03-torchdynamo-frontend' },
          { text: '04 Guard 系统', link: '/knowledge/ai/systems/pytorch/torch-compile-wiki/04-guard-system' },
          { text: '05 缓存机制', link: '/knowledge/ai/systems/pytorch/torch-compile-wiki/05-cache-mechanism' },
          { text: '06 图断裂', link: '/knowledge/ai/systems/pytorch/torch-compile-wiki/06-graph-break' },
          { text: '07 AOTAutograd 中间层', link: '/knowledge/ai/systems/pytorch/torch-compile-wiki/07-aotautograd' },
          { text: '08 前向/反向分区策略', link: '/knowledge/ai/systems/pytorch/torch-compile-wiki/08-partition-strategy' },
          { text: '09 TorchInductor 后端', link: '/knowledge/ai/systems/pytorch/torch-compile-wiki/09-torchinductor-backend' },
          { text: '10 Lowering: FX → IR', link: '/knowledge/ai/systems/pytorch/torch-compile-wiki/10-lowering-fx-to-ir' },
          { text: '11 调度器与融合', link: '/knowledge/ai/systems/pytorch/torch-compile-wiki/11-scheduler-fusion' },
          { text: '12 代码生成', link: '/knowledge/ai/systems/pytorch/torch-compile-wiki/12-code-generation' },
          { text: '13 完整编译流程', link: '/knowledge/ai/systems/pytorch/torch-compile-wiki/13-full-compile-pipeline' },
          { text: '14 配置与模式', link: '/knowledge/ai/systems/pytorch/torch-compile-wiki/14-config-and-modes' },
          { text: '15 torch.fx 专题', link: '/knowledge/ai/systems/pytorch/torch-compile-wiki/15-torch-fx-special' },
          { text: '16 TorchDynamo 深入', link: '/knowledge/ai/systems/pytorch/torch-compile-wiki/16-torchdynamo-deep' },
          { text: '17 torch.compile 后端', link: '/knowledge/ai/systems/pytorch/torch-compile-wiki/17-compile-backend' }
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
          { text: 'pyproject.toml', link: '/knowledge/engineering/pyproject-toml' },
          { text: 'MongoDB 与 PyMongo', link: '/knowledge/engineering/mongodb-pymongo-getting-started' },
          { text: 'ER 图', link: '/knowledge/engineering/er-diagram' },
          { text: 'UML 类图', link: '/knowledge/engineering/uml-class-diagram' },
          { text: 'DFD 数据流图', link: '/knowledge/engineering/dfd' },
          { text: 'Git on NFS', link: '/knowledge/engineering/git-on-nfs' }
        ]},
        { text: 'Kubernetes', collapsed: true, items: [
          { text: '专题总览', link: '/knowledge/engineering/kubernetes/' },
          { text: '01 容器与网络前置', link: '/knowledge/engineering/kubernetes/01-prerequisites-containers-and-networking' },
          { text: '02 架构与控制循环', link: '/knowledge/engineering/kubernetes/02-architecture-and-control-loop' },
          { text: '03 kubectl、对象与 YAML', link: '/knowledge/engineering/kubernetes/03-kubectl-objects-and-yaml' },
          { text: '04 Pod 与多容器模式', link: '/knowledge/engineering/kubernetes/04-pods-and-container-patterns' },
          { text: '05 Workload 与任务', link: '/knowledge/engineering/kubernetes/05-workloads-deployments-and-jobs' },
          { text: '06 Service、Ingress 与 DNS', link: '/knowledge/engineering/kubernetes/06-services-networking-ingress-and-dns' },
          { text: '07 ConfigMap 与 Secret', link: '/knowledge/engineering/kubernetes/07-configuration-configmap-and-secrets' },
          { text: '08 PV、PVC 与 CSI', link: '/knowledge/engineering/kubernetes/08-storage-pv-pvc-and-csi' },
          { text: '09 资源、调度与扩缩容', link: '/knowledge/engineering/kubernetes/09-resources-scheduling-and-autoscaling' },
          { text: '10 RBAC 与 Pod 安全', link: '/knowledge/engineering/kubernetes/10-security-rbac-and-pod-security' },
          { text: '11 Helm 包管理', link: '/knowledge/engineering/kubernetes/11-helm-and-package-management' },
          { text: '12 可观测性与排障', link: '/knowledge/engineering/kubernetes/12-observability-and-troubleshooting' },
          { text: '13 CRD、Operator 与生产', link: '/knowledge/engineering/kubernetes/13-crd-operators-and-production-next-steps' }
        ]},
        { text: 'Learning', items: [
          { text: 'Overview', link: '/knowledge/learning/' },
          { text: '如何学习：方法地图', link: '/knowledge/learning/how-to-learn-systematically' },
          { text: '源码阅读方法', link: '/knowledge/learning/code-reading/' },
          { text: '审计 / 上手 / 深读', link: '/knowledge/learning/code-reading/codebase-review-modes' },
          { text: '新代码库阅读导览', link: '/knowledge/learning/code-reading/codebase-reading-guide' }
        ]},
        { text: 'Investing', items: [
          { text: 'Overview', link: '/knowledge/investing/' },
          { text: '价值投资学习路线图', link: '/knowledge/investing/value-investing-learning-roadmap' },
          { text: '价值投资入门', link: '/knowledge/investing/value-investing-intro' },
          { text: '上市公司财报阅读指南', link: '/knowledge/investing/financial-statement-reading' }
        ]},
        { text: 'Mathematics', items: [
          { text: 'Overview', link: '/knowledge/mathematics/' }
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
