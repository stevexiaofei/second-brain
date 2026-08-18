@AGENTS.md

# Claude Code notes

This repository was originally written around TRAE, but the workflow is tool-agnostic.
Use the same Markdown-first, search-first, incremental-edit approach.

## Claude-specific expectations

- Search the repository before creating or moving notes.
- Preserve existing content; prefer small diffs.
- When adding or moving notes under `docs/inbox/` or `docs/knowledge/`, update the matching index files and VitePress sidebar per [AGENTS.md](AGENTS.md).
- Treat TRAE mentions in user-facing docs as historical unless the text is explicitly tool-specific.
- If documentation or navigation changed, build the site before handing off when practical.
