---
name: claude-md-audit-unapplied
description: "The 2026-10-05 CLAUDE.md audit was APPLIED 2026-10-18 (439 → 127 lines) and its folder DELETED the same day at the human's request; CLAUDE.md text that code comments cite must survive edits"
metadata:
  node_type: memory
  type: project
  originSessionId: 46d96f22-0d6c-475d-8a10-35209504793c
  modified: 2026-10-18T06:48:58.952Z
---

A prompt audit of the project CLAUDE.md, made 2026-10-05 and kept git-excluded under
`docs/private/`, was **applied 2026-10-18** (the human: "your call on what to apply"). Its prepared
diff was NOT used — it predated phases C–F and still named `physsynth/core/`, pytest and a Python
frozen file. The finding table was used as a checklist and the file rewritten against today's
text: item 3 went from a ~390-line migration diary to a short "Rust, no Python" item; #4 names the
crate; HANDOFF §11's Python decision marked superseded by the human's 2026-08-26 call.

**Folder deleted 2026-10-18** at the human's request ("delete it"); `docs/private/` is gone too.
The `/docs/private/` line in `.git/info/exclude` was left in place — harmless, and still keeps any
future private notes out of this public repo.

**Why:** a record that the audit is done, so nobody goes looking for it or re-runs it from memory.

**How to apply:** never `git add -f` anything under `docs/private/` if it comes back. Nothing reads
CLAUDE.md programmatically (checked 2026-10-18); code comments cite its 1e-10 bar, "do not
tighten", closed-form validation, oversampling and "#4" — keep those when editing it.
