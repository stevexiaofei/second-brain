# 🧠 My Second Brain

A personal knowledge system built with Markdown, Git, GitHub, VitePress and TRAE.

## Philosophy

> Capture everything. Understand deeply. Connect ideas. Keep what matters.

## Architecture

```text
Capture / Learn / Think
          ↓
       📥 Inbox
          ↓
        TRAE
   ↙       ↓       ↘
Search   Organize   Connect
          ↓
     🧠 Knowledge
          ↓
         Git
          ↓
       GitHub
          ↓
      VitePress
          ↓
     Static HTML
```

## Start

```bash
npm install
npm run docs:dev
```

Build:

```bash
npm run docs:build
```

## Deployed Site

<https://stevexiaofei.github.io/second-brain/>

## Workflow

1. Put rough thoughts into `docs/inbox/`.
2. Ask TRAE to organize them.
3. Move stable knowledge into `docs/knowledge/`.
4. Link related notes.
5. Commit and push.
6. GitHub Actions deploys the Wiki.

Markdown is the source of truth. HTML is the presentation layer. Git is the memory. TRAE is the assistant.
