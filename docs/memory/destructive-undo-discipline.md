---
name: destructive-undo-discipline
description: "Never undo a small temporary edit with `git checkout -- <file>` — revert the edit itself; commit a checkpoint before deliberately breaking code to test a suite"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 838aa134-bec2-4851-94ca-5b63171d941b
  modified: 2026-08-10T04:12:30.144Z
---

When I make a **temporary** edit to test something — inverting a sign to check that a test suite
actually catches it, stubbing a function, forcing a branch — I must undo it by reverting **that
edit**, never by running `git checkout -- <file>` (or `git restore`, or `git stash` without care).

On 2026-08-10, mid air-box batch 4, I inverted the `-q`/`+q` injection order in
`physsynth/core/airbox.py` to confirm the new dipole suite would fail. It did (6 of 6 sign tests,
plus conservation). Then I ran `git checkout -- physsynth/core/airbox.py` to undo the inversion —
and it **discarded ~1000 lines of uncommitted batch-4 core work**: `add_cut`, `_PatchPort`,
`InteriorSurfacePort`, `RoomSuspendedPlate`. It was recoverable only because the whole batch was
still in the session transcript and a golden-number pin could prove the replay byte-faithful.

**Why:** `git checkout -- <file>` does not undo "the last edit" — it restores the file to HEAD, and
in a repo where the interesting work is by definition uncommitted, that is the maximum-damage
operation dressed as a cleanup. The narrow intent ("undo my two-line test hack") and the actual
effect ("delete everything since the last commit") differ by the entire session.

**How to apply:**
- Undo a temporary edit with the inverse edit (an `Edit` call swapping the strings back), or write
  the original text back from a copy taken before the change.
- **Commit a durable checkpoint before deliberately breaking code**, so the destructive path is
  survivable even if taken. A break-it-and-watch-the-suite-fail check is worth doing; do it on top
  of a commit.
- Before any `git checkout`/`restore`/`reset`/`clean` on a path, run `git status --short` and
  `git diff --stat` on that path and read what would be lost. If the answer is "work from this
  session", it is the wrong command.
- This applies to overwriting files generally: look at the target before discarding it.

Related: [[commit-push-at-batch-end]] — committing at batch end is the floor, not the ceiling; a
checkpoint mid-batch is what makes a destructive slip cheap.
