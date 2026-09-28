---
name: python-retirement-state
description: "The human reversed both Python carve-outs 2026-09-07 — Python goes to ZERO; killing it also kills 19k lines of RUST (the pyo3 binding), and a third of the 2,713 tests evaporates rather than ports"
metadata: 
  node_type: memory
  type: project
  originSessionId: 9458bd92-fbc0-4375-9531-e3beb4b28d3a
  modified: 2026-09-07T20:23:10.659Z
---

**2026-09-07, the human's call: Python goes to ZERO in this repo.** This reverses two of the
human's own earlier decisions — `rust-migration-plan.md` §35.4 ("the test suite's move to native
bars is *elective*") and §35.5 ("route (b), the viewer backend stays Python — do not re-open").
They were the human's to set and are the human's to unset; **do not quote either back as a
constraint**. `CLAUDE.md` non-negotiable #3 is amended to say so, and the rest of that item is
now history rather than direction. Plan: `M:\claud_projects\physical synthesis\docs\dev\python-retirement-plan.md`.
Supersedes [[viewer-stays-python]].

**The finding that reframes the whole thing: deleting Python deletes 19,024 lines of RUST.**
`crates/physsynth-py` is a pyo3 binding and has exactly one purpose — letting Python call Rust.
No Python, no binding. So the accounting is ~58,300 lines of Python **plus** 19,024 of Rust, about
**77,000 lines removed**, with less written back. That is a stronger argument than "Python is
retired" and it is the one to lead with.

**Second finding: a third of the suite has nothing to port to.** Of 2,713 tests (measured
2026-09-07; the stored 1,808 was stale), ~462 are claims *about Python* and evaporate — 275
Rust-vs-Python parity, 60 binding surface, 21 package shape, 106 pytest-runner machinery. 429 more
are the viewer backend and follow the viewer. The real work is **~1,744 physics bars**, against
**550 native tests** that already exist (448 core + 102 analysis).

**The order is forced, not chosen.** The binding dies when its last caller does, and it has exactly
three classes of caller: tests, viewer, scripts. So viewer and scripts proceed in parallel with the
bar port, and the binding is deleted last. The four `import("physsynth…")` Rust-to-Python reaches
look like a blocker and are **free** — they live inside the crate being deleted.

**Two decisions taken inside this one:**

* **The plotting island was chosen and withdrawn the same hour.** There is no Python corner. The
  plotting capability *leaves* — where a diagnostic drew a figure, Rust emits the data and the
  figure is drawn outside the repo. Had the island survived it would have had to read **files**,
  never import the physics, or the binding could not be deleted and the 19k lines stayed.
* **`banded.py`'s Python half is LAPACK, not a transcription** — the yardstick every θ-scheme
  string acceptance number was measured against. Deleting it retires a reference oracle, which is
  a physics decision, not cleanup. Two measured facts make it acceptable (2026-08-27, in that
  file's header): drift 2.7e-12 vs LAPACK's 2.7e-12 against a 1e-10 bar, and the four models'
  bit-identity *with each other* is untouched because all four call the same solver either way.

**The named risk:** the Python suite IS the acceptance contract, and this phase deletes the
checker. §35.4's rule is the only defence and it is a discipline, not a mechanism — nothing *fails*
if a bar is dropped silently. Every retiring commit must name the native test that replaces the
Python one. See also [[rust-deletion-phase-state]] for how the body deletions were sequenced.

**Phase B is DONE (2026-09-07) and it revised both sizing numbers.** The map is §9 of the plan.

* **The unit was wrong.** `pytest`'s 2,713 are *cases*; 1,389 are further rows of a `@parametrize`
  and the suite is **1,324 functions** (one function = 85 cases, another = 80). Rust has no
  parametrize — a native test loops inside one `#[test]` — so 2,713-vs-550 compares two units.
  Counted as functions the physics is **899 vs 475**.
* **The first reading said nine models were at zero. EIGHT of the nine were the JOIN**, not a
  hole — and this is the durable lesson: **a join between two test suites indexed on different
  axes measures the indexing, not the coverage.** Python test files are named `<model>_<claim>`;
  Rust test files are named after the **source module**. `plate.rs` alone covers the supported,
  free, orthotropic, guitar and von Kármán plates because in Rust they are one module (~24 VK
  bars, ~5 orthotropic, ~4 guitar, ~3 free), and the damped string's four bars live inside
  `string_stiff.rs` (a `damped_params` helper + `sigma1_zero_is_the_stiff_string_exactly`).
  **The tell was `plate` at 39/42** — suspiciously balanced for a file carrying five variants.
  The mapping was *complete* on both sides and still *wrong*: complete and correct are different
  properties, and the mismatch lands wherever a Python-facing model is a **configuration** of a
  Rust type rather than a type of its own. Before believing a zero, read the test NAMES in the
  file that would plausibly hold the bar — two greps overturned eight.
* **Genuinely empty: sympathetic strings (15 functions), and most of the orthotropic free plate
  (29 functions, 1 bar).** The connection is partial (`body.rs` has the read-outs and the rank-one
  guard; the coupled-energy bars are only Python's). So the job is mostly **verification, not
  authorship** — the "212 functions with nothing on the far side" was wrong by an order of
  magnitude. **Sympathetic strings is the recommended first phase-C batch**, not the damped string.
* **The room is the biggest gap by volume, not a zero**: `airbox` 188 vs 37.
* **A populated cell is NOT proof of coverage** — only that both suites say something about that
  model under that heading. Same-bar-same-fixture is a per-claim read belonging to each batch.
* Model axis hand-mapped over all 88 py + 41 rs files, nothing unclassified; claim axis is
  keyword-derived and indicative (a third of both sides lands in `other`).
