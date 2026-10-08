---
name: claude-md-audit-unapplied
description: "An UNAPPLIED 2026-10-05 audit of this project's CLAUDE.md (325 → 180 lines, 13 stale statements) is kept in docs/private/, git-excluded — never commit it"
metadata:
  node_type: memory
  type: project
  originSessionId: caf7a610-33f2-4885-afd7-81f4946e66ef
  modified: 2026-10-08T15:42:25.227Z
---

`docs/private/claude-md-audit-2026-10-05/` holds a prompt audit of the project CLAUDE.md made on
2026-10-05: `audit-report.md` (findings), `CLAUDE.md.diff` / `CLAUDE.md.proposed` (the rewrite),
`CLAUDE.md.orig` (the file as audited), `CLAUDE.md.copy`. Nothing in it was ever applied — on
2026-10-08 CLAUDE.md was still the long diary version it proposed cutting. Moved out of
`W:\temp\claude` during a temp cleanup, at the human's choice ("keep, not committed").

**Why:** it names private paths under the user profile and reviews the user's global CLAUDE.md, and
this repo is public. The folder is excluded through `.git/info/exclude` (`/docs/private/`), not
`.gitignore`, so the exclusion is local to this clone.

**How to apply:** if the human asks to slim or refresh CLAUDE.md, start from this report — but its
line numbers and several findings predate phases C–F, so re-check each against the current file.
Never `git add -f` anything under `docs/private/`. The copy is named `CLAUDE.md.copy`, not
`CLAUDE.md`, so it is never loaded as instructions.
