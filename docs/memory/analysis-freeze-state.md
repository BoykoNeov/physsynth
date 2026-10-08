---
name: analysis-freeze-state
description: "The analysis instrument was DELETED (2026-09-03) but its numbers were FROZEN first — really 74 fixtures, not the 62 everyone quoted; since 2026-10-08 (§47) the record is native JSON in the analysis crate, INPUTS frozen too"
metadata: 
  node_type: memory
  type: project
  originSessionId: 1384bf6b-dd8f-4158-a97a-7ca431d94d28
  modified: 2026-09-03T16:19:50.332Z
---

`physsynth/analysis/` — the thing that *measures* every model — has one implementation now.
Plan `docs/dev/rust-migration-plan.md` **§44**, findings **#47–#50**.

## The human's condition, and why it mattered

`PHYSSYNTH_RS_ANALYSIS` existed so the acceptance run could put a **Rust model under a Python
ruler**: a misreading shared by a model and its detector would cancel, and two implementations
cannot easily share one (§36.4). Deleting the Python instrument ends that. The human's call
(2026-09-03) was **delete, but freeze the numbers first** — so the check can still be *repeated*,
just never *re-derived*.

`tests/analysis_frozen_values.py` is the record: **62 fixtures, 3,708 floats, 55 ints**, taken from
the Python implementation the day before it was deleted. `tests/analysis_frozen_cases.py` holds the
inputs (deterministic, no clock, no environment), `scripts/freeze_analysis.py` regenerates it (and
**only works while a second implementation exists** — after this, never again), and
`tests/test_analysis_frozen.py` asserts it on every run, flagged or not.

## The design decisions worth reusing for any freeze

- **Structure and integers compared EXACTLY; floats on one tolerance.** A root search returning a
  different *number* of roots, a changed multiplicity, `converged=False` — none is a small error and
  no tolerance describes one. That split is where the teeth are.
- **One bar, 1e-13 (Group A), on the amplitude-normalised gap** — never an equality. This compares a
  Rust answer computed now against a Python answer computed on another machine (ledger #28 plus an
  extra axis). 51 of 62 recorded gaps were exactly 0.0; the worst was 3.5e-15.
- **Measure the two sides through the PUBLIC names, in a subprocess with the flag set.** The first
  generator guessed the binding's name from the Python one and reported "no Rust twin" for **15 of
  62 cases that have one** — the names are not a mechanical transform (`free_free_beam_betaL` →
  `modal_free_free_beam_beta_l`, `solve_rotating_wave` → `rotating_wave_solve`), and several public
  names are wrappers that *adapt arguments*, so a gap against the raw binding measures something
  nobody runs.
- **Three guards on the freeze itself**: the case list is derived from each module's `__all__` (a
  new oracle cannot be added unfrozen), every case must carry a real measured gap rather than a
  string saying why one could not be taken, and the two files must cover exactly the same keys.
- **A recording mechanism applies ONE rule to everything it records** — and that is how the freeze
  shipped a bug (§44.9, finding #51). `flatten` collects every integer and the test compares them
  exactly, which is right for a mode label or a root count and **wrong for `RotatingWave.iterations`**,
  the one quantity finding #33 says is not comparable. It went green because neither fixture happened
  to be a witness, one commit after deleting the test whose whole job was to forbid that assertion.
  **Sweep every int and bool in a record**: structure (length, label, count → exact) or decision
  (iteration count, predicate → needs a recorded margin or an exclusion). The same sweep found the
  `#over` damping fixture returning `True` like its sibling, so the pair covered one arm twice.
- **What a freeze cannot do**: catch an error the Python made too. That is
  `crates/physsynth-analysis/tests/`'s job — and §37.11 is the precedent, a native bar finding a
  **544% defect the Python always had**.

## Two scars from the same batch

- **A module header is a claim and nothing checks it.** `crates/physsynth-analysis/tests/rotating_wave.rs`
  listed the Jacobian identities among what it asserted, for a phase, while they lived only in
  `tests/test_geometric_rotating_wave.py`. The deletion is what found it. When the blocked test
  reaches **private internals**, the cure is a `mod tests` *inside* the module — an integration test
  cannot reach them either.
- **Scan a finite-difference step, never choose it.** The gap ran 3.0e-9 · 4.9e-8 · 1.5e-6 · 1.5e-5
  as the step shrank from 1e-5 to 1e-8 — it *grows* as the step shrinks, so the small step was
  measuring its own subtraction. A step carried over from another language's arithmetic is carrying
  an accident.

See [[rust-deletion-phase-state]], [[spectrum-detector-guard]], [[rust-phase7-oracles-state]],
[[rust-phase7-rotating-wave-state]], [[numpy-libm-cpu-dispatch]].

## Moved native, 2026-10-08 (retirement plan §47, phase F step 1)

The Python test and both data modules are deleted. The record is
`crates/physsynth-analysis/tests/reference/analysis_frozen.json`, read by that crate's
`tests/analysis_frozen.rs` (4 bars). Three things worth keeping:

- **"62 fixtures, 3,708 floats" was stale for a month** — the horizon plan added 12 rows and nothing
  re-derived the number. It is **74 rows, 3,754 floats, 179 ints, 59 functions**. Count from the
  data, never from a document.
- **Freeze the INPUTS, not only the answers, when moving a record across languages.** Several inputs
  were built by NumPy's seeded RNG, `sin`, `exp`, `linspace` — nothing native rebuilds them. They
  went in as exact doubles, with every Python-wrapper default written out (the shims that held the
  defaults are going). The reader fails a case whose recorded argument it never read.
- **Self-check that worked:** same Rust behind the binding + same input doubles ⇒ the native
  per-case gap must equal pytest's to the bit. 74/74 did. A checksum over every double is the only
  thing that sees a record parsed without `float_roundtrip` (a last bit is far inside the 1e-13 bar).

The `__all__`-derived coverage guard was NOT carried: nothing can freeze a new oracle any more.
