# 🧠 Second Brain AI Rules

You are operating inside my personal second brain. This is a long-term personal knowledge system.

## Source of truth
Markdown files are the source of truth. Never treat generated HTML as the source of knowledge.

## Core philosophy
Prefer understanding over summary, connection over collection, experience over copying, and first principles over memorization.

## Before creating knowledge
Always search the repository first. Check for existing or similar notes. Update/extend/link existing notes instead of blindly creating duplicates.

## Atomic knowledge
Prefer one concept per document. Examples: `Transformer.md`, `Attention.md`, `PPO.md`, `GRPO.md` rather than one huge AI.md.

## Good knowledge notes
When appropriate answer: What is it? Why does it matter? How does it work? What problem does it solve? Limitations? My understanding? Related knowledge?

## Personal understanding
Clearly distinguish established facts, external sources, my interpretation, my experience, and hypotheses. Never present speculation as fact.

## Links
Use normal Markdown links to files that actually exist. Do not invent links.

## Diagrams
Choose the clearest representation for the context:
- **ASCII art / character-drawn boxes**: Better for simple inline sketches, quick architecture overviews, or when alignment matters more than visual polish.
- **Mermaid**: Better for complex diagrams (ER, UML, DFD, long flowcharts, sequence) that need precise arrow routing, styling consistency across notes, and editability as text blocks.

Use whichever communicates the idea more clearly. No hard requirements on format.

## Math formulas
Use LaTeX syntax for all mathematical formulas and symbols: `$...$` for inline, `$$...$$` for block. KaTeX renders them automatically in VitePress. Do not use backticks (e.g. `q_σ`) or plain text for math symbols — they will display as raw characters. Examples: write `$\sigma$` not `\`σ\``, write `$q_\sigma(x_{1:T} \| x_0)$` not `\`q_σ(x_{1:T} | x_0)\``.

## Inbox
Inbox is temporary. Understand → search → detect duplicates → classify → move stable knowledge → link → preserve useful context. Never delete information silently.

When creating or moving a note into `docs/inbox/`, always also add a link entry to the `Notes` section of `docs/inbox/index.md`. VitePress does not auto-scan directories; without an index entry the page is built but unreachable from navigation. Same rule applies when moving a note out of inbox: remove its entry from `docs/inbox/index.md`.

## Experience
For engineering/debugging notes capture Environment, Symptoms, Investigation, Root Cause, Solution, Why, Lessons Learned.

## Papers
Capture Problem, Motivation, Key Idea, Method, Mathematical Formulation when useful, Experiments, Results, Limitations, My Understanding, Open Questions, Related Knowledge.

## Knowledge connections
Look for prerequisites, extensions, alternatives, shared principles, applications, analogies, and missing concepts.

## Git
Keep changes focused. Do not modify unrelated files. Prefer meaningful commits such as `docs: add GRPO knowledge`.

## AI behavior
When given rough notes: Understand → Search → Compare → Organize → Connect → Write. Do not immediately turn every fragment into a polished essay.

## Second brain principle
The repository should gradually become a model of my understanding: what I learned, what I believe, why I believe it, what I experienced, how ideas connect, and what I still do not understand.
