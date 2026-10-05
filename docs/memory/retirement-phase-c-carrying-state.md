---
name: retirement-phase-c-carrying-state
description: "Phase C CARRYING batches (retire a Python physics file whose model is already native) — batches 1-11 done by 2026-10-05 (every plate family, membrane, beam, ideal/stiff/damped/tension string); 41 files / 430 functions left"
metadata:
  node_type: memory
  type: project
  originSessionId: bc58c454-d6f0-40f5-8259-4ca90ee23d46
  modified: 2026-09-29T14:07:47.776Z
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
- §28 batch 5 (human chose "guitar"): `test_guitar_plate.py` (21 fns / 113 cases) → `tests/plate_outline.rs`
  (18). NumPy outline masks, SciPy Bessel quotients, LAPACK disk eigenvalues frozen into
  `crates/physsynth-core/tests/reference/guitar_plate.json` (first frozen reference in core).
- §29 batch 6 (human took the recommendation, "membrane"): `test_membrane_{energy,modal,stability,dispersion}.py`
  (19 fns / 42 cases) → `tests/membrane_harness.rs` (14) + 4 dispersion bars in `physsynth-analysis/tests/modal.rs`
  (they touch no model). LAPACK eigenvalues of the disk (N=32/64/128) + SciPy Bessel zeros frozen into
  `tests/reference/membrane.json`; ARPACK at shift 0 (no nullspace) agreed with LAPACK to 5e-16.
- §30 batch 7 (human: "the natural next batch is the beam - do it"): `test_beam_{energy,modal,stability}.py`
  (16 fns / 32 cases) → EXTENDED the existing `tests/beam.rs` (17 → 24) instead of a new harness file,
  because it was already at `make_beam`'s parameters. `impl Resonator for FreeBeam` added to engine.rs.
  Frozen `tests/reference/beam.json` (7 grids). 12 plants all red.
- §31 batch 8 (human took the recommendation, "ideal string"): `test_{energy,modal,convergence,dispersion}.py`
  whole + the first 4 fns of `test_stability.py` (20 fns / 35 cases) → EXTENDED `tests/string_ideal.rs`
  (13 → 21) + one bar in `physsynth-analysis/tests/oracles.rs`. NO outside referee: nothing frozen. Orphans
  `make_string` and `measure_mode_frequencies` removed from helpers. 10 plants all red.
- §32 batch 9 (human took the recommendation, "stiff string"): `test_stiff_string.py` (20 fns / 48 cases)
  → NEW `tests/string_stiff_harness.rs` (12; not an extension: string_stiff.rs runs another fixture and holds
  the both-profiles `squaring_is_pow_not_multiply`) + 3 in `ops.rs` + 5 in analysis `oracles.rs`. Nothing
  frozen; every Python figure reproduced to the digit. 14 plants all red; start-up bar added (F).
- §33 batch 10 (human took the recommendation, "damped string"): `test_damped_string.py` (16 fns / 28 cases)
  → NEW `tests/string_damped_harness.rs` (14: 12 carried + start-up bar + θ sweep) + 6 in analysis `oracles.rs`
  (5 carried + the analysis freeze's T60 row). `np.polyfit` decay fit transcribed + certified (gap 0). Every
  Python figure reproduced to the digit. 14 plants; F (T60 const), H/I (θ hard-coded) seen by NOTHING
  workspace-wide → asked the human, who chose to guard both. 3.5 s release / 15 s debug locally,
  1.7 s / 27.3 s on CI (run 36622026049, green; debug job 7.8 min vs release 12.0): both profiles.
  Review caught 3 prose overclaims (a miscounted bar list, an inferred catch written as measured,
  a missing Windows-only caveat on the viewer freeze) — check §-prose against the logs before pushing.
- §34 batch 11 (2026-10-05, human took the recommendation, "tension string"): `test_tension_string.py`
  (30 fns / 39 cases) → NEW `tests/string_nonlinear_harness.rs` (24: 23 carried + a θ sweep) + 6 in analysis
  `oracles.rs`. CORE GAINED `string_coefficients_from_material` (+ `StringCoefficients`, `MaterialError`) —
  four tests were about the Python-only helper; ported unasked (plate precedent), NumPy's six fields asserted
  to the bit. Python copy stays (geometric tests import it). Every trajectory figure reproduced to the digit;
  only projection read-outs (np.dot BLAS) differ in last digits. 13 plants: A (tol ignored) only the tol bar;
  D/E (θ hard-coded) seen by NOTHING → human chose the θ sweep; start-up plants F/G seen by the Richardson
  bar (a physics bar, not a twin) → no start-up bar added. `apply_Ainv` refusal: no analogue (absent = type).
  0.71 s release / 1.5 s debug: both profiles. §27's loose end closed (`test_arpack_oracles_are_bit_reproducible`
  + `free_plate_low_eigenfrequencies` deleted). pytest 1,216 → 1,175. CI run 37302929849 green: 0.41 s release /
  6.22 s debug (jobs 12.0 / 10.0 min). Material bar exact on glibc too: the six recorded UCRT `pow`s were
  checked correctly rounded first (`fractions.Fraction`) — do that check for ANY bit-exact `pow` record.
Remaining after §34: **41 physics files / 430 functions** (§24.1 count). Next: the geometric string (six
`test_geometric_*` files, the last string family; it still imports the Python material helper). Next batch
not chosen — the user picks the family; a recommendation + "go with it" is an accepted answer.
Loose end (from §27): `test_arpack_oracles_are_bit_reproducible`'s surviving free-plate half guards a
helper whose only caller is the guard itself.

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
- Frozen floats in JSON need serde_json `float_roundtrip` (now on in core's dev-dep): without it a
  1-ulp parse error became a 31-ulp failure (§28.1).
- Native `eigsh_shift_invert` fails (300-iter cap) at shift -1e-8 next to a 3-D rigid nullspace;
  use -1e-3·mu_1 (§28.2).
- When a planted breakage goes uncaught, find WHERE it is pinned (other test files, viewer freezes)
  before calling it a gap (§28.4).
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
  would have become the gate's slowest job. §27's plate_free also release-only. §28's
  plate_outline stays in BOTH (CI: 5.3 s release / 75.7 s debug; debug job 2.6 → 4.6 min, still
  ~7 min shorter than the release job) — first candidate if the debug job becomes the long pole.
- Bash heredocs with backticks/quotes in markdown fail to parse here — append docs via Edit.
- A bar about a CONSTANT must not import it from the model (§29.3: CFL bars read `lambda_max()`, so a
  planted `1.1/sqrt2` moved bar and model together). Write the constant in the test, as the Python did.
- A carried "worst of" must propagate NaN (`fold(_, f64::max)` drops it): §29's passivity bar passed a
  NaN run until fixed in review. Use a `nan_max` / assert every element, like `np.all`.
- A rectangle oracle bug that swaps axes is invisible on a SQUARE — check which bars use Lx != Ly.
- Python edits via `open(p,'w')` on Windows write CRLF; use `newline=''` (git warns on the .rs files).
- A truth STRONGER than LAPACK: the 50-digit (mpmath) Rayleigh quotient of LAPACK's eigenvector — exact for
  that vector, error second order in the vector's. Record it beside LAPACK and measure the native solvers
  against it (§30.1). ARPACK was the outlier a third time (22.6 eps·mu_max, N=120).
- LAPACK `eigh` eigenvalues-only (jobz='N') and with-vectors are DIFFERENT algorithms (the rigid pair moved
  -1.4e-6 → -6.3e-6); say which path produced the record.
- Native `eigsh_shift_invert`'s error = eps·mu_max floor PLUS its 1e-10 RELATIVE stopping rule. Neither unit
  alone is a bar (172 eps·mu_max at the top of a 24-mode window; 7e-9 relative at N=400's fundamental).
  Bar per mode: `20 eps mu_max + 1e-9 |mu|`, the 1e-9 written as a literal (§30.2).
- Plant the START-UP too: dropping the ½ in u^{-1} passed the WHOLE workspace (energy can't see any u^{-1},
  the discrete cosine holds from any start, FFT too coarse). Pin it with the exact eigenmode identities:
  at rest u¹−u⁻¹ = θc²/(1+θc)·u⁰; launched, centred velocity = V exactly (§30.4).
- Extending an existing bar to the Python's length must not SHORTEN it anywhere: use max(Python length,
  old length) — the first draft cut μ=16 from 8,000 steps to 5,120 (caught in review).
- A margin row must be the measured worst over EVERY step, not a sampled one (review caught "never rose").
- §30 beam.rs: 0.70 s release / 14.5 s debug locally, 1.1 s / 9.5 s on CI (run 36553822722, green on Linux, first Linux run of the frozen referee) — both profiles (debug job 3.2 min vs release 11.6).
- An existing native bar at ONE parameter value is blind to a defect that vanishes there (§31: an energy with
  `(h/k)²` for `c²` is exact at λ=1 and passed every pre-existing bar in string_ideal.rs). Carry the Python's
  SWEEP, and plant a defect that is invisible at the old parameter to prove it.
- "Only X catches it" must be measured WORKSPACE-wide (`cargo test --workspace --release --no-fail-fast`, script
  `W:\temp\claude\ideal-string\mutate_ws.py`) before it is written. §31's first draft ran one file; B and C then
  went red in 11 binaries (connection*, 7 viewer files incl. frozen). Review caught it; §31.4 corrected.
- Grep each helper name ALONE before calling it live: an OR'd grep reported callers that belonged to the
  other names, and `make_string` was actually an orphan (§31.5).
- A refiner mutation can leave a bar green for a structural reason: when every tone lands on an exact FFT bin
  the parabolic correction is 0 whatever its sign (§31.4 J). Explain it, don't call it a gap.
- §31 string_ideal.rs: 0.16 s release / 3.3 s debug locally, 0.36 s / 6.2 s on CI (run 36578409910, green) — both profiles.
- A red TWIN anchor (two transcriptions compared) says two copies differ, not which is wrong — §32's
  dropped-½ start-up was seen only by the stiff↔damped anchor (+ viewer freeze) until the exact eigenmode
  start-up bar was added. Credit a plant to a physics bar, not to a twin.
- The viewer freeze `physsynth-viewer --test frozen` is a second witness for string defects but compares
  exactly only on Windows CI; on Linux it is structure only — say so when crediting it (§32.4).
- A refusal bar must assert the error VARIANT: with the ρ check dropped, ρ=−1 still errors (NaN pivot →
  NotFactorable), so `pytest.raises(ValueError)`-style "some error" would have passed (§32.4 D).
- §32 string_stiff_harness.rs: 6.9 s release / 92 s debug locally, 8.1 s / 99.9 s on CI (run 36593245615,
  green) — both profiles, but the debug job grew 6.0 → 7.3 min vs release 9.7: the gap is closing, weigh
  `release_only` (ask the human) before the next long both-profile file.
- The viewer freeze caught 5 of §32's 7 re-planted plants (A, B, F, J, K) — re-plant EVERY "only" claim
  workspace-wide, including ones in the prose (§32.2's J/K claim was caught by the advisor, not by me).
- A constant that appears on BOTH sides of a round trip cancels out of it (§33 F: T60 constant +1% passed
  both inversion round trips). Pin it with an independent recorded answer (the analysis freeze's row).
- A model PARAMETER every test builds at its default is unguarded (§33 H/I: damped θ hard-coded to 0.28 seen
  by nothing). Grep the fixtures for the parameter's range, not just the model's bars.
- If a plant is seen by NOTHING, ask the human (plain words, recommendation first) before adding or omitting a
  bar; a third instance of an already-established bar (the start-up bar) was added without asking.
- A Python-only HELPER whose tests are retiring gets ported into core unasked when a sibling precedent exists
  (§34: the string material helper, after plate's `grain_ratios_from_material`); say so in the report.
- Before adding the start-up bar, check whether a convergence bar against an exact solution already sees the
  plant: §34's Richardson run dropped to first order under both start-up plants, so no bar was needed.
- Projection read-outs (`np.dot` = BLAS) differ in the last digits from a left-to-right sum even when the
  trajectory is bit-identical — attribute last-digit differences to the read-out, not the model (§34.1).
- Bash heredocs choke on markdown with backticks: Write the section to a temp file, append with Python.
- §29 core file: 3.4 s release / 59 s debug locally, 3.9 s / 37 s on CI (run 36546609975, green on Linux) — stayed in both profiles (debug job 3.0 min vs release 11.2).

Related: [[python-retirement-state]], [[retirement-phase-d-state]], [[retirement-phase-a-state]],
[[free-plate-orthotropic-state]], [[orthotropic-plate-state]], [[rust-airbox-native-bars]].
