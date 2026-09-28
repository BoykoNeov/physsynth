---
name: viewer-stays-python
description: "The human's call 2026-09-03 — the viewer backend stays Python and talks to Rust; the import audit it left open is DONE (2026-09-06) and closed as a COVERAGE question, not a gate"
metadata: 
  node_type: memory
  type: project
  originSessionId: f1ee07fe-8634-4127-b790-8fec42400ec8
  modified: 2026-09-05T21:13:30.851Z
---

**SUPERSEDED 2026-09-07 — the human reversed this.** The viewer backend does NOT stay Python;
route (a), a Rust HTTP server, is back on. Everything below is an accurate record of the
2026-09-03 decision and of the import audit that closed under it, and none of it is current
direction. See [[python-retirement-state]].

**Decided 2026-09-03 by the human, closing plan §35.5's two routes: route (b).** `web/serialize.py`
stays Python and talks to Rust through the `physsynth_rs` binding. The browser side was never in
question. Do not re-open this — it now sits with HANDOFF §11's five closed decisions.

**Why:** "Python goes, all of it" (the 2026-08-26 supersession) was about the *physics*, not about a
JSON serializer. `serialize.py` builds models, steps them and hands the browser numbers; rewriting
9,915 lines of that in Rust buys no fidelity, no portability and no speed the binding does not
already give.

**The audit it left open is FINISHED — plan §50, 2026-09-06 — and the framing it was written under
was already obsolete when it ran.** §35.5 called it the gate on every deletion: find the names
`serialize.py` reaches past a public constructor, because each must exist on the Rust object before
the Python body behind it can go. The deletions finished first (§40–§49), so that half is discharged
**by construction** — a name the binding does not expose is an `AttributeError` in the viewer *today*,
not a future hazard. The half that does not answer itself is the one no grep reaches: **a reach only
proves the binding exposes it if the line executes.** So it was run as line coverage, not an
inventory.

**How to apply:**

* **Do not re-run this as an import inventory.** The reproducible scripts are in
  `M:\claud_projects\temp\viewer-audit\` (`reaches.py`, `superset.py`, `intersect.py`); after any
  viewer change, re-run `superset.py`, which *derives* the surface from `pkgutil.iter_modules` over
  **both** `physsynth.core` and `physsynth.analysis`, module-level names included, rather than a
  list anyone maintains. Its first draft walked `core` only and only class `dir()` — the narrowing
  finding #69 is *about*, committed inside the scan that measured #69, and it left every reach onto
  an analysis oracle's result outside the population. Widening moved the numbers (23 modules / 408
  names / 635 lines → 29 / 633 / 680) and not the conclusion.
* **The old "47 call sites" number is retired**, and was not reconciled with the new one on purpose
  (§45.7/§47.6 forbid splitting the difference): the AST measurement finds 12 private lines
  (6 names), 35 raw-state lines, 9 operator lines, 6 state *writes* onto a Rust model, and **zero**
  `._lu` reaches. Grep counted prose — 3 of 13 "private reaches" were a comment and two docstrings.
* **Measure writes separately from reads.** A `#[getter]` with no `#[setter]` is silently read-only
  (the §33 scar), and the viewer writes six state arrays on the geometric string at
  `web/serialize.py:2698–2699` to seed an exact rotating-wave history.
* **Result: 94% coverage, 680 candidate reach lines, 3 dark** — two defensive branches provably
  unreachable from any call site (a closed-closed bore, an empty plate-frame fallback) and one name
  collision. The one *live* gap was the bow's empty settle window: asking for less audio than the
  animation window captures `bow.state` at step 0, a path never run. Now tested. Reachability
  analysis, not coverage alone, is what says whether a dark line needs a test or needs nothing.
* **Plan §5's Phase 8 is struck** and **§35.6's "the room waits on the viewer" is void** — both were
  consequences of a Rust viewer that is not being written.

**The Rust migration now has no remaining named work.** What is left in the project is elective:
the test suite's move to native bars (§35.4), the two costed physics proposals in
`docs/dev/scientific-hurdles.md` §4–§5, and the real-time port.

`CLAUDE.md`'s non-negotiable #3 carries the exception and the audit's result. Related:
[[rust-migration-state]], [[rust-deletion-phase-state]], [[web-viewer-state]],
[[rust-phase7-oracles-state]].
