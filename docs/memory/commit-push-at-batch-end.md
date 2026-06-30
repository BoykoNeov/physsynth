---
name: commit-push-at-batch-end
description: "Standing rule — at end of a work batch or on 'session end', always update memory + docs, then commit and push to main"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 8232fb2b-c7b2-4556-bdf9-553bd1dcba6f
---

At the end of any of these: a work batch, a docs update, a memory update, or a planning stage
— **always**: (1) update memory and docs to reflect the finished state, (2) commit, and (3) push
to `main`. This is durable standing authorization — no need to ask before committing/pushing at
any of these boundaries (overrides the default "commit only when asked" harness rule).

**Commit ⟹ push, always paired (human, 2026-06-30):** "commit and push" are a single action here.
When the human says just "commit" (or I reach a commit boundary), **push in the same step** — never
commit-only and hold the push back as a separate outward-facing approval gate. Do not split them.

**Why:** the human wants every meaningful checkpoint to land as a durable pushed commit — code
batches, docs/memory refreshes, and planning milestones alike. Nothing left in an uncommitted
working tree.

**How to apply:** treat the end of a work batch, any docs/memory write, or a planning stage as a
commit boundary. Keep logically separate changes as separate commits (see [[milestone-1-state]]
commit boundaries).
Remote is now configured (2026-06-21): `origin` = git@github.com:BoykoNeov/physsynth.git (PUBLIC),
local `main` tracks `origin/main`, so `git push` works directly at batch end.

**Sync the memory mirror (added 2026-06-23, human's request):** the per-project memory is now also
version-controlled inside the repo at `docs/memory/` — a verbatim mirror of this canonical dir
(`~/.claude/projects/M--claud-projects-physical-synthesis/memory/`). As part of step (1), copy the
canonical memory dir into `docs/memory/` so each pushed checkpoint carries the latest memory + index.
The canonical `~/.claude` dir stays the single source of truth (Claude Code reads/writes it); the repo
copy is a downstream mirror — re-sync it at batch end, don't hand-edit it. The repo is PUBLIC, so
never put a credential or private path in a memory file (today they are all project-technical).
