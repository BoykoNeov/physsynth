---
name: rust-phase7-oracles-state
description: "Phase 7 batch 2 — the closed-form oracles and their special functions; a Bessel bar is ABSOLUTE, a dependency's direction is a grep, and a native bar found a defect the Python always had"
metadata: 
  node_type: memory
  type: project
  originSessionId: f1ee07fe-8634-4127-b790-8fec42400ec8
  modified: 2026-09-03T10:57:47.302Z
---

Phase 7 batch 2 (2026-09-03): `analysis/modal.py`, `damping.py`, `dispersion.py`, `duffing.py`
and the Bessel/elliptic functions they stand on are in Rust, behind `PHYSSYNTH_RS_ANALYSIS`.
`analysis/` has one file left, `rotating_wave.py`. Plan §37; ledger #30 and #31.

**The denominator of an agreement bar is a modelling choice — the batch's finding, three times
over.** `J_n` agrees with SciPy to **6.7e-16 absolute** and **2.1e-12 relative**, and those are the
same measurement: the relative worst case sits where `J_3` itself is `-9.8e-05`, near one of its own
zeros, so the *denominator* collapsed rather than the error growing. Same shape twice more:
`duffing_displacement` is 1.4e-17 absolute on an amplitude of 0.07 and reads 2.7e-4 *relative* at a
zero crossing; `duffing_frequency_shift` is 3.5e-15 of the shift and 1.6e-16 of the frequency.
And the first attempt at the disk-scan margin said 1e-35 because it divided by the scan's global
maximum — the determinant near `λ → 0` is legitimately 1e-187 and perfectly signed. Dividing by the
**local cancellation** instead gives the real margin: 4.6e-6 against a 1e-15 perturbation, ~5e9×.
This is [[rust-phase2-radiation-state]]'s amplitude-not-pointwise scar arriving in a module with no
trajectory in it.

**A near-cycle whose direction was settled by one grep.** `modal` needs core's Brent transcription;
`piston_radiation_resistance` in `core/radiation.py` needs a Bessel `J1`. That reads as
analysis→core and core→analysis, which Cargo refuses. But the piston helper is **never called inside
a model's `step()`** — two tests, two scripts, three docstrings — so it is an oracle in a core file
and the second edge does not exist. Brent is reached by a `#[path]` **source include**, not a Cargo
edge: one copy, nothing for `deps.rs` to catch, both allowlists still empty. Taking `physsynth-core`
as a dependency was refused (it inverts the argument the crate split exists to make); copying the
file was refused (it duplicates a method whose justification is reproducing SciPy exactly).

**A native bar found a defect a parity test structurally cannot.** A bar asserting
`piston_radiation_resistance`'s two branches meet at their threshold failed — and the defect is in
the *Python*: its `ka < 1e-8` series guard is three decades too small, so just above it
`1 - J1(2ka)/ka` cancels sixteen digits and the shipped function is **544% wrong**. Two `J1`s
differing in a last bit therefore disagree by 300% there. Reproduced deliberately (changing a
shipped physics number inside a porting batch is not a port); registered as
`docs/dev/scientific-hurdles.md` **§14** — and the human called it the same day, so §14 opened and
closed on 2026-09-03. The general lesson: **a parity test compares two implementations of the same
mistake**, so a native bar checking two independent routes to one number is the only thing that
finds this class.

**The fix, and three corrections it produced.** Three Taylor terms in Horner form below `ka = 3e-2`;
worst relative error over `ka` in [1e-10, 10] goes **5.24 → 6.7e-13**. (1) The algebraic crossover
estimate written into the first draft of hurdles §14 said `ka ≈ 7.2e-4`; **measured it is 2e-4** —
off by 3.6× — and the shipped threshold is at neither, because three terms let it sit *past* the
direct form's own noisy region rather than at the crossover. **§36.2's "measure first" applies to
fixes, not only ports.** (2) The first draft claimed the whole function was now bit-identical across
languages; measured, **only the series branch is** (0 of 3,000), while the direct branch differs in
1,444 of 3,000 at 9.8e-13 — two different `J1` implementations, amplified 2,200× by the residual
cancellation at the seam, which *is* the threshold's justification. (3) A test asserting a defect
must be **replaced by one asserting the property the fix established** (the branches meet at the
seam), not deleted — and writing it surfaced §27 again: `(scale*ka2)*rest` ≠ `scale*(ka2*rest)`.

**Almost everything is bit-identical and none of it is asserted.** Every function agrees to the bit
except six, including all the `sin`/`arcsin`/`arccos`/`log2` ones — but requiring that would be
[[numpy-libm-cpu-dispatch]]'s mistake, so equality is required only where IEEE-754 requires it
(eighteen functions), plus one *physics* equality: a lossless string's decay factor is exactly 1.0.

**A derived CI file list was wrong by 43 of 65 files, and its own safety argument was outside it.**
The step that runs "the instrument's clients with a Rust instrument" was widened from the detector's
three names to files importing `physsynth.analysis` — 27 matched. But `tests/helpers.py` imports the
instrument and measures on its callers' behalf, so **65** test files reach it. Among the 38 silent
ones: `test_radiation.py`, which carries the ONLY check holding the ported Bessel `J1` against
`scipy.special.j1`. **A floor cannot catch a grep that is asking a narrower question than its name
claims** — it only catches one that stopped matching. Pattern now
`physsynth\.analysis|import helpers|from helpers`, floor 60. Run separately with `PHYSSYNTH_RS=1`:
93 passed, and the composed piston resistance is **bit-identical** to the Cephes-built expectation.
Ledger #32.

**Also:** a search's *count* is a claim a tolerance cannot make (the disk scan asserts how many roots
it found, because a missing one is the dangerous direction and all three self-checks catch a
spurious one instead). `dispersion.py`'s early `from .modal import` is **safe** — a footer runs
inside its own module body — pinned by a subprocess probe rather than argued. `free_free_beam_betaL`'s
docstring says the roots approach `(2i+1)π/2` "from above", which is true of the first and false of
the second. See [[rust-phase7-spectrum-state]] for batch 1 and [[viewer-stays-python]] for the scope
decision taken the same day.
