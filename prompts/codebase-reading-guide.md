---
name: codebase-reading-guide
description: Shared prompt template for reading a new codebase systematically: search first, map architecture, trace key flows, detect duplicates, and turn findings into reusable notes or a reading plan.
---

# Codebase Reading Guide

Use this when starting a new repository or subsystem.

## Core loop

Understand → Search repository → Detect duplicates and canonical sources → Map architecture → Trace key flows → Compare docs/tests/code → Organize findings → Connect concepts → Write notes only when needed.

## What to do first

1. Read the repository's own instructions first: `README`, `CLAUDE.md`, `AGENTS.md`, contributing docs, build scripts, and test entry points.
2. Identify the stack and the canonical entry points.
3. Find existing docs, indices, diagrams, or notes so you don't duplicate them.
4. Prefer searching before reading deeply.

## What to map

Build a mental model of:

- Repository layout
- Entry points
- Core abstractions
- Runtime modes
- Data flow
- Control flow
- State / caches / queues / registries
- Public APIs vs internal implementation
- Tests and benchmarks that define intended behavior

## Reading order

A good default order is:

1. Repository metadata and README files
2. Build / package / config files
3. Tests and examples
4. Public APIs and wrappers
5. Core implementation
6. Hot paths and performance-sensitive code
7. Edge cases, fallbacks, and error handling
8. Documentation gaps and contradictions

## Output shape

When you report back, prefer these sections:

- Repository overview
- Architecture map
- Entry points
- Key files
- Key abstractions
- Key states / terms
- Reading order
- Open questions
- Suggested notes to create
- Next actions

## If you are saving notes

- Keep notes atomic: one concept, one file.
- Preserve existing content; prefer small diffs.
- Detect duplicates and extend existing notes instead of creating new ones blindly.
- Connect related notes with real markdown links.
- If the repo has docs navigation or indexes, update them too.
- If docs changed, verify the docs build when practical.

## Good habits

- Do not guess when the code or docs are unclear; label hypotheses as hypotheses.
- Cite file paths and line numbers when possible.
- Use tables for comparisons and state summaries.
- Use Mermaid or ASCII diagrams when they make the architecture easier to see.
- Keep a separation between established facts, interpretation, and open questions.

## Recommended note set

For a new codebase, a useful first pass is often:

- A system map or architecture overview
- An entry-point / interface note
- A core internals note
- A glossary or key-state table
- A reading guide / roadmap

## References

- `README.md`
- `CLAUDE.md`
- `AGENTS.md`
- `docs/` or equivalent documentation root
