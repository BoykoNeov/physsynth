---
name: retirement-phase-c-carrying-state
description: "Phase C CARRYING batches (retire a Python physics file whose model is already native) — batches 1-4 done 2026-09-29 (ortho plates, plain plate, free plate); 56 files / 566 functions left"
metadata:
  node_type: memory
  type: project
  originSessionId: bc58c454-d6f0-40f5-8259-4ca90ee23d46
  modified: 2026-09-29T04:45:04.898Z
---

Phase C of `docs/dev/python-retirement-plan.md`, second kind of work: the model is already Rust,
so a batch = read each Python test's ASSERTIONS (not names), find/write the native bar, plant
breakages, delete the file.
- §24 batch 1: `tests/test_free_plate_orthotropic.py` → `crates/physsynth-core/tests/plate_free_grain.rs`.
- §25 batch 2: `tests/test_plate_orthotropic.py` (19 fns / 22 cases) → `tests/plate_grain.rs` (19).
- §26 batch 3 (human chose "plain plate"): `test_plate_{energy,modal,stability}.py` (21 fns / 45 cases)
  → `tests/plate_kirchhoff.rs` (19). Three tests were ABOUT SciPy's `L @ L` row order: SciPy's
  `csr_matmat` transcribed into the test and certified by reproducing recorded digests.
- §27 batch 4 (human chose "free plate"): `test_free_plate_{energy,modal}.py` (19 fns / 21 cases)
  → `tests/plate_free.rs` (18). LAPACK recorded at N=20..80, bars in units of eps·mu_max.
Remaining after §27: **56 physics files / 566 functions**. Next batch not chosen — scope by model;
the user picks the family (asked via options last time).

**Rules these batches set:**
- A bar needing core + analysis: `physsynth-analysis` is a TEST-ONLY dev-dependency of
  `physsynth-core` (the human's call; `deps.rs` walks normal/build edges only, no cycle).
- Only NumPy/SciPy/LAPACK numbers are independent referees — everything else in a Python test
  already runs through the Rust binding (Rust checking Rust). Record those BEFORE deleting
  (reinstall the wheel first), with LAPACK dense `eigh` not ARPACK (7e-7 off in §24).
- A test whose SUBJECT is SciPy: transcribe the SciPy kernel and certify it against a recorded
  digest (§26.1), rather than dropping the test.
- A carried bar can certify a building block the model no longer USES (§27.3: `collocated_d2_1d`);
  plant the error in the model's real path and rewrite the bar against the real builder.
- The 1e-10 acceptance bar is never moved even at 2.9x headroom (§26.2 mu=0.5 drift 3.5e-11) — record it.
- A dense-eigen bar is written in units of `eps·lambda_max` (§25 uses 20 of them), not a bare 1e-9.
- A hand-built stand-in for a SciPy product is PROVED faithful by reproducing SciPy's recorded
  number exactly (§25: the L@L gap 1.70601310856e-16 over 195 entries), not assumed.
- Measure EVERY margin (finding 80): §25's sine residual had 2.2x (eps·|B|/q floor) → widened.
- Plant a breakage in the ORACLE too, not just the model: bars comparing the formula with itself
  pass a broken oracle; check something independent (FFT of a real run) catches it.
- A "control" identical to its subject asserts nothing — make it differ in the last bit and ASSERT
  that it differs (§25 ledger control: moduli scaled by 1.1).
- A "bit-identical to the helper" test where Rust has one code path is a VERDICT (finding #78).
- Planted breakages: snapshot src to `W:\temp\claude\...`, `str.count==1` patterns, restore by copy,
  byte-compare, `git status`. Script: `W:\temp\claude\ortho-supported\mutate.py`.
- pytest count drops by (cases + 1): `test_xdist_groups` is parametrized per test file. Reconcile
  against a `git worktree` at HEAD with `--collect-only` — a count from an older section is stale.
- New native files run in BOTH CI profiles unless added to `rust-debug`'s `release_only`. §25's
  (8 s release / 304 s debug on CI) WAS added, the human's call: a long-trajectory file whose exact
  checks have no transcendental/constant exponent goes release-only. Ask; don't add unasked.
  §26's plate_kirchhoff (45 s / 1,072 s debug) also release-only, asked BEFORE pushing because it
  would have become the gate's slowest job.
- Bash heredocs with backticks/quotes in markdown fail to parse here — append docs via Edit.

Related: [[python-retirement-state]], [[retirement-phase-d-state]], [[retirement-phase-a-state]],
[[free-plate-orthotropic-state]], [[orthotropic-plate-state]], [[rust-airbox-native-bars]].
