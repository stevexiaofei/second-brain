---
name: repository-refactor
description: Scan a repository for structural and maintenance problems, propose an evidence-based refactor plan, then implement an approved bounded refactor with validation.
---

# Repository Refactor

Use this skill when the user wants to scan a repository or documentation system for unreasonable structure, navigation defects, duplication, stale links, organizational drift, or other refactor opportunities—and wants both a practical plan and implementation.

## Core principle

**Audit first → plan explicitly → obtain an implementation boundary → make focused changes → validate.**

Do not turn a broad request to “refactor” into unbounded rewriting. Preserve useful content, distinguish verified defects from hypotheses, and keep unrelated pre-existing work untouched.

## Phase 0: Establish context and safety boundaries

1. Read repository instructions first: `README.md`, `CLAUDE.md`, `AGENTS.md`, contribution guides, build configuration, and documentation configuration.
2. Inspect the current Git status before making recommendations or edits.
3. Treat pre-existing uncommitted changes as protected:
   - do not reset, revert, overwrite, or silently fold them into the refactor;
   - identify them in the report when they affect the audit;
   - stop and ask when their relationship to the requested refactor is ambiguous.
4. Identify canonical sources of truth. In this repository, Markdown is the source of truth; generated site output is not.
5. Establish the scope: repository-wide, a subsystem, or a documentation area. Start broad enough to find cross-boundary failures, but keep the eventual implementation bounded.

## Phase 1: Scan in independent dimensions

Search before reading deeply. Build an evidence-backed finding list across these dimensions.

### 1. Structure and information architecture

Look for:

- mixed concerns in one directory or topic root;
- concept, paper, source-reading, project, and temporary material mixed without a useful hierarchy;
- duplicate or strongly overlapping notes;
- oversized overview pages that should become topic maps;
- orphaned pages, unrepresented areas, and misleading directory names;
- unclear ownership or canonical-source conflicts.

For a knowledge repository, classify content before proposing moves:

| Content type | Typical role |
| --- | --- |
| concept | Reusable explanation of one stable idea |
| paper | Problem, method, evidence, limitations, and interpretation |
| source-reading | Architecture, control/data flow, states, interfaces, and source locations |
| project map / guide | Reading order, scope, and links among a topic’s notes |
| Inbox material | Temporary, unverified, or not yet fully understood |

Use subject/topic directories as the default organization. When a topic includes both theory and implementation, keep them together under a topic index while making the reading paths explicit.

### 2. Navigation and discoverability

Inspect parent `index.md` files, VitePress configuration, menus, sidebars, and any generated navigation rules.

Look for:

- a note that exists but is absent from its parent index;
- a note that is in an index but absent from the sidebar, or the reverse;
- stale sidebar routes after moves;
- duplicate entrances or unintuitive navigation depth;
- topic directories without an overview or reading path.

### 3. Links, paths, and metadata

Check for:

- broken relative Markdown links;
- stale paths after renames/moves;
- local-only `file:///` URLs in publishable content;
- links that resolve but point to the wrong semantic target;
- links to nonexistent placeholder pages;
- inconsistent or stale frontmatter when the repository uses it;
- documentation examples that can confuse simplistic link scanners; classify these separately from real broken links.

### 4. Content quality and maintenance

Inspect for:

- stale project snapshots or roadmaps that contradict current structure;
- missing validation coverage for navigation, links, or documentation builds;
- weak separation between established facts, personal interpretation, and open questions;
- overly broad changes that would create churn without improving retrieval or understanding.

Do not treat style preference as a defect unless it harms correctness, retrieval, navigation, or maintenance.

## Phase 2: Report findings and design the refactor

Before modifying files, report findings grouped by impact and confidence.

For every proposed change, state:

- **Finding** — what is wrong or unclear;
- **Evidence** — exact file paths and relevant lines or search results;
- **Impact** — correctness, discoverability, maintainability, or cognitive cost;
- **Recommendation** — smallest coherent improvement;
- **Confidence** — confirmed or hypothesis;
- **Effort / risk** — low, medium, or high.

Then present a concrete refactor plan with:

1. target directory and topic architecture;
2. exact files to move, add, modify, or delete;
3. required parent-index and sidebar updates;
4. link and metadata repairs;
5. validation commands and expected checks;
6. explicitly deferred work, especially large content rewrites or uncertain classifications.

Prefer a small structural pass over a simultaneous structural rewrite and content rewrite. Do not create empty placeholder notes merely to make a taxonomy appear complete. Prefer a plain-text backlog for missing concepts until a real note exists.

## Phase 3: Archive and confirm the implementation boundary

Before presenting the refactor plan for approval, archive it in the repository under `plans/`:

1. Create `plans/` if it does not exist.
2. Use a descriptive, date-prefixed filename: `YYYY-MM-DD-<scope>-refactor-plan.md`.
3. Write the plan in Chinese by default, unless the user explicitly requests another language. Include the audit scope, protected Git state, confirmed findings with evidence, exact implementation boundary, deferred work, and validation steps.
4. Treat the archived file as an approved-plan record after the user confirms it; update it with the implementation outcome only if doing so keeps the original proposed boundary clear.
5. Add the archived plan to the final change summary. Do not put plans in generated VitePress output unless the user explicitly asks to publish them.

Do not mutate the repository outside the plan archive until the user has approved the exact refactor plan, unless their instruction explicitly authorizes a fully specified implementation.

The plan archive itself is a record of the proposal, not approval to apply destructive or ambiguous changes.

## Confirm the implementation boundary

Do not mutate the repository until the user has approved the exact refactor plan, unless their instruction explicitly authorizes a fully specified implementation.

Always ask before:

- deleting material rather than moving or preserving it;
- overwriting content not created during the task;
- applying a broad metadata migration;
- changing content classification when evidence is inconclusive;
- committing, pushing, or publishing.

If the user approves only a structural refactor, do not expand into rewriting or splitting long notes. Record that work as deferred.

## Phase 4: Implement the approved refactor

1. Make moves with version-control-aware commands when available.
2. Preserve content. Keep diffs small and avoid formatting churn.
3. For every content page added, moved, or removed under a VitePress content directory:
   - update its parent `index.md`;
   - update the relevant VitePress sidebar/navigation configuration;
   - repair links from related notes and cross-domain references.
4. For Inbox moves, update `docs/inbox/index.md` in both directions.
5. Replace publish-hostile local filesystem links with useful textual source references unless a real public destination exists.
6. Update `updated` only in notes materially changed during the refactor.
7. Keep unrelated defects out of the implementation; capture them as follow-up work instead.

## Phase 5: Validate and hand off

Run the strongest practical validation set:

- repository-specific documentation build or tests;
- a Markdown relative-link scan, separating real failures from code-example false positives;
- VitePress sidebar/navigation target checks;
- a search for stale old paths and local-only links within the changed scope;
- `git diff --check`;
- final Git status and diff review.

Report results faithfully:

- list completed moves and architecture changes;
- list repaired links and navigation changes;
- state the exact validation outcomes;
- clearly distinguish validation that passed, failed, or could not run because of an environment limitation;
- list deferred and out-of-scope issues with paths;
- do not commit or push unless explicitly requested.

## Repository-specific quality bar

Follow `AGENTS.md` throughout:

- Markdown is the source of truth.
- Search before creating or moving knowledge.
- Prefer atomic knowledge and genuine connections over superficial collection.
- Use only real Markdown links to existing pages.
- Maintain every required Inbox, Knowledge index, and VitePress sidebar entry.
- Preserve the distinction among established facts, interpretation, experience, and hypotheses.
