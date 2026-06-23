---
name: commit-push-at-batch-end
description: "Standing rule — at end of a work batch or on 'session end', always update memory + docs, then commit and push to main"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 8232fb2b-c7b2-4556-bdf9-553bd1dcba6f
---

At the end of a work batch, or whenever the human says "session end", **always**: (1) update
memory and docs to reflect the finished state, (2) commit, and (3) push to `main`. This is durable
standing authorization — no need to ask before committing/pushing at a batch boundary (overrides the
default "commit only when asked" harness rule for *this* repo's batch-end checkpoints).

**Why:** the human wants each batch to land as a durable, pushed checkpoint with docs/memory in
sync — not left as an uncommitted working tree they have to chase.

**How to apply:** treat a completed deliverable (e.g. the §6.5 dispersion polish) as a batch end.
Keep logically separate changes as separate commits (see [[milestone-1-state]] commit boundaries).
Remote is now configured (2026-06-21): `origin` = git@github.com:BoykoNeov/physsynth.git (PUBLIC),
local `main` tracks `origin/main`, so `git push` works directly at batch end.

**Sync the memory mirror (added 2026-06-23, human's request):** the per-project memory is now also
version-controlled inside the repo at `docs/memory/` — a verbatim mirror of this canonical dir
(`~/.claude/projects/M--claud-projects-physical-synthesis/memory/`). As part of step (1), copy the
canonical memory dir into `docs/memory/` so each pushed checkpoint carries the latest memory + index.
The canonical `~/.claude` dir stays the single source of truth (Claude Code reads/writes it); the repo
copy is a downstream mirror — re-sync it at batch end, don't hand-edit it. The repo is PUBLIC, so
never put a credential or private path in a memory file (today they are all project-technical).
