---
name: retirement-phase-c-carrying-state
description: "Phase C CARRYING batches (retire a Python physics file whose model is already native) — batch 1 done 2026-09-29: free orthotropic plate; record SciPy with LAPACK dense, ARPACK was 7e-7 off"
metadata:
  node_type: memory
  type: project
  originSessionId: bc58c454-d6f0-40f5-8259-4ca90ee23d46
  modified: 2026-09-28T21:56:51.910Z
---

Phase C of `docs/dev/python-retirement-plan.md`, second kind of work: the model is already Rust,
so a batch = read each Python test's ASSERTIONS (not names), find/write the native bar, plant
breakages, delete the file. Plan §24 is batch 1 (2026-09-29): `tests/test_free_plate_orthotropic.py`
(29 functions) → `crates/physsynth-core/tests/plate_free_grain.rs` (30 bars). Next: 
`tests/test_plate_orthotropic.py` (19, the supported grained plate). Remaining after §24: 62 physics
files / 625 functions — §9.4's map was 3 weeks stale (sympathetic strings were NOT at zero; §12
retired them), so re-derive before scoping.

**Rules this batch set:**
- A bar needing core + analysis: `physsynth-analysis` is a TEST-ONLY dev-dependency of
  `physsynth-core` (the human's call; `deps.rs` walks normal/build edges only, no cycle).
- SciPy-derived truth: record the numbers BEFORE deleting (reinstall the wheel first). Record with
  LAPACK dense `eigh`, not whatever the test called: ARPACK at the test's shift `-1e-4` was 7.2e-7
  off; native shift-invert and native dense were within 5e-11 of LAPACK. But at N=80 (6,561
  unknowns) dense LAPACK's own floor (`eps·mu_max`, `mu_max ~ h^-4`) is ~1e-9, and the SHIFTED
  solvers agree — bar 1e-8 there.
- A "bit-identical to the helper" test where Rust has one code path (`unwrap_or`) is a VERDICT, not
  a carried equality (finding #78) — keep a cheap bar pinning the premise.
- Planted breakages: snapshot src to `W:\temp\claude\...`, `str.count==1` patterns, restore by copy,
  byte-compare, `git status`. Script: `W:\temp\claude\free-ortho\mutate.py`.
- pytest count drops by (cases + 1): `test_xdist_groups` is parametrized per test file.
- New native files run in BOTH CI profiles unless added to the Rust job's `release_only` list; this
  one is 1.5 s release / ~30 s debug, left in both (reported, the human's call).
- Bash heredocs with backticks/quotes in markdown fail to parse here — append docs via Edit.

Related: [[python-retirement-state]], [[retirement-phase-d-state]], [[retirement-phase-a-state]],
[[free-plate-orthotropic-state]], [[rust-airbox-native-bars]].
