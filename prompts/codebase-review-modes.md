---
name: codebase-review-modes
description: Shared prompt template for three repository-reading modes: audit, onboarding, and deep read.
---

# Codebase Review Modes

Use this when the user wants one of three things:

- **代码库审计**: find bugs, risks, missing tests, contradictions, or suspicious behavior
- **代码库上手**: get productive quickly in a new repository
- **代码库深读**: build a durable mental model of architecture, flows, and key states

## Default selector

If the user does not explicitly name a mode:

- choose **audit** when the user asks to review, check, verify, or find problems
- choose **onboarding** when the user asks to get started, run, modify, or understand the basics fast
- choose **deep-read** when the user asks to study,精读, map, or organize knowledge about the codebase

## Shared core loop

Understand → Search repository → Detect duplicates / canonical sources → Map architecture → Trace key flows → Compare docs / tests / code → Organize findings → Connect concepts → Write notes only when useful.

## Mode 1: 代码库审计

### Goal

Find real issues, not just interesting facts.

### What to inspect

- correctness bugs
- missing edge cases
- unsafe assumptions
- stale docs and mismatched behavior
- fragile state handling
- performance regressions
- poor error handling
- missing or weak tests
- security / privacy concerns when relevant

### Output shape

- Findings sorted by severity
- Evidence from files / tests / reproduction steps
- Why it matters
- Suggested fix or next verification step
- Confidence level and open questions

### Habits

- Prefer adversarial reading
- Try to falsify claims
- Check tests and failure modes
- Compare docs, code, and runtime assumptions

## Mode 2: 代码库上手

### Goal

Become useful quickly in the repository.

### What to inspect

- README and setup docs
- build / test / run commands
- main entry points
- project structure
- config files
- extension points and main abstractions
- one happy-path example

### Output shape

- What this repo is for
- How to set it up
- How to run it
- Main entry points
- Where to make the first change
- Common pitfalls
- First tasks to become productive

### Habits

- Optimize for time to first successful change
- Keep the mental model small
- Focus on the minimum useful path
- Avoid over-reading internals before the repo is runnable

## Mode 3: 代码库深读

### Goal

Build a long-term understanding of the system.

### What to inspect

- architecture and module boundaries
- public API vs internal implementation
- control flow and data flow
- state, caches, registries, queues, and lifecycles
- hot paths and performance-sensitive code
- edge cases and fallback paths
- tests and benchmarks that define intent
- related notes and existing conceptual structure

### Output shape

- Repository overview
- Architecture map
- Entry points
- Key files
- Key abstractions
- Key states / terms
- Reading order
- Open questions
- Suggested notes to create
- Connections to existing knowledge

### Habits

- Read from general to specific
- Use diagrams and tables where they reduce cognitive load
- Preserve nuance and distinguish facts from interpretation
- Turn repeated patterns into reusable notes

## Recommended note set

For a new codebase, a strong first pass often becomes:

- a system map
- an entry-point / interface note
- a core internals note
- a glossary or key-state table
- a reading guide / roadmap

## Quality bar

- Do not guess when the code or docs are unclear; label hypotheses as hypotheses.
- Prefer small, incremental notes instead of large rewrites.
- Search before reading deeply.
- Detect duplicates before creating new notes.
- Connect notes with real links.
- Update navigation if docs change.
- Verify the docs build when practical.

## References

- `README.md`
- `CLAUDE.md`
- `AGENTS.md`
- `docs/` or equivalent documentation root
