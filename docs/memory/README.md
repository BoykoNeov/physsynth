# Project memory (mirror)

This directory is a **version-controlled mirror** of Claude Code's per-project auto-memory.

- **Canonical source** (single source of truth, read/written by Claude Code):
  `~/.claude/projects/M--claud-projects-physical-synthesis/memory/`
- **This copy** is a downstream snapshot, committed so the project history and any
  human/agent picking up the repo can see the accumulated design state without access to the
  local `~/.claude` dir. Re-synced at each batch end (see `commit-push-at-batch-end.md`);
  do not hand-edit files here — edit the canonical dir and re-copy.

## What's here

`MEMORY.md` is the index (one line per memory). Each other file is a single fact with YAML
frontmatter (`name`, `description`, `metadata.type`). Types:

- **project** — state of an ongoing model/deliverable (e.g. `beam-state.md`, `plate-state.md`).
- **feedback** — how to work in this repo (e.g. `respect-ruff-line-length.md`,
  `commit-push-at-batch-end.md`).

Bodies cross-link with `[[name]]` wiki-links pointing at other files' `name:` slugs.

## Caveat

Each file is a **point-in-time observation**, not live state — `file:line` citations and claims
about code behavior reflect when the note was written and may have drifted. Verify against the
current code before relying on a specific detail. The authoritative, always-current artifacts are
the code, the tests, and the plan docs under `docs/dev/`.
