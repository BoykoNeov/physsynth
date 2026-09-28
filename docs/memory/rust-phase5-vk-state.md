---
name: rust-phase5-vk-state
description: "Rust migration Phase 5 batch 3 — the nonlinear plate finishes operators2d; an association moved a SUM not a product, and a chaotic trajectory decorrelates while every physics bar stays green"
metadata: 
  node_type: memory
  type: project
  originSessionId: 6b704d3a-a52d-42a1-87db-17429ad85d5b
  modified: 2026-08-28T17:44:55.439Z
---

Phase 5, batch 3 (2026-08-28), plan §27. Ports the last of `operators2d`: the five private 1-D
differences (`collocated`/`forward`/`centered`/`clamped`/`avg`), `VonKarmanBracket` and
`AiryStressSolver`. **`operators2d` is now ported in full**; what is left of the core is `plate`,
`connection`, `string_geometric`, then `airbox` and `analysis/`. `plate.py` needed **no edit** —
`VKPlate` is now a NumPy loop around three Rust objects.

**The finding — §26.5's mantissa rule is about PRODUCTS, and an association can move the SUM.**
`BᵀWB` written with no parentheses is left-associated by Python. Ask §26.5's question (do the outer
factors share a mantissa?) and the answer is yes, so **not one term differs** — and the two matrices
differ anyway, because the two bracketings contract through differently-*ordered* intermediates:
SciPy hands `Lc_rᵀ @ Wa` back **descending**, and a sparse product contracts the shared index in the
stored order of its left operand's rows. Live on **2 of the 22 grids the suite actually builds**
(46 entries of 1,889 at one) — so unlike [[rust-phase3-collision-state]]'s blind fixture, the
suite's own grids contain the witness. **Enumerate the shipped grids by instrumenting a run; do not
sample.**

**The remedy is a THIRD kind, and the cheapest.** After "reproduce the values" (§26.2) and "sort the
storage" ([[rust-phase3-strings-state]]'s `portable.canonical`), the third is **re-associate**: pick
the bracketing whose contraction runs over an operand that is already canonical. One pair of
parentheses, no module, no run-time cost. The question that finds it: *which operand's stored order
does this contraction run over, and did I choose it or did SciPy?*

**Rust could not have fixed it.** `Csr::from_rows` sorts, so a descending row is inexpressible in
the crate — [[rust-phase1-state]]'s canonical-`Csr` decision paying off a third time.

**A CSC scatter and a canonical CSR gather are the same sum** — a lemma, not a measurement: both
accumulate each output entry over increasing column index. That is why the bracket is bit-identical
even though `Acell.T` is a CSC in SciPy and a CSR in Rust.

**One constant is the wrong shape for a Group D bar.** The Airy solve's gap runs 3.1e-16 at 4×4 to
**5.2e-10** at the 160×128 the airbox tests build — that is `N⁴`, a biharmonic's condition number.
Both sides are **backward stable** (asserted, so the growing tolerance cannot be misread as the port
decaying), so the forward difference is conditioning×epsilon and says nothing about either
implementation. [[rust-phase4-beam-state]]'s §24.4 manoeuvre — Python solver on the Rust
factorization — makes the two **bit-identical**, which is the only reason the assembly claim is
sayable at all.

**A seventh agreement regime, and the first where the trajectory becomes unrelated while every
physics bar stays green.** Same plate, same amplitude, twelve orders apart: one regime random-walks
to 1.1e-13 at 1,000 steps (and is *bit-identical* for 4,000 below `w/e ≈ 0.1`), the other is chaotic
and e-folds every **~57 steps**, reaching **0.59 of the peak by step 2,000**. Energy 5.1e-14, drift
2.6e-14 on both sides throughout. **The bar reads the ENERGY, never `max|du|/amp`** — and the
general form is that *a conserved quantity is not a trajectory comparison*.

**What separates the two regimes is the PICARD SWEEP COUNT, not the fixture and not the amplitude.**
Both of those were tried and falsified — a broadband start normalised to the same peak random-walks,
and the window is flat at ~1e-13 from `w/e = 0.5` to `w/e = 10`. Two to six sweeps a step is the
random walk, eleven or more is exponential. `n_iters` is a public attribute, so this is checkable
without rebuilding the fixture set.

**Nothing in the von Karman/airbox suites compares a displacement across implementations** — the
airbox bars are 1e-12 relative on an *energy*, and every `array_equal` there compares two
configurations of the same implementation. That is why the flagged step is green at 160x128 (gap
5.2e-10) for a reason rather than by luck. The parity file asserts the solve gap as a **scaling
law**, `1e-17·(n_x n_y)²`, parametrised over the large grids too.

**The Rust factorization is 16x slower than SuperLU (2.5 s vs 0.16 s at 160x128) and it does not
show**: the same 24 files ran **350.9 s on the default path against 291.6 s with the flag**, back to
back on one machine — the per-step overhead the rest of the tree wins back is larger.

The Picard convergence branch was watched: first flip at step 1,553, when the deviation is already
2.1e-3 — a consequence, not a cause.

Two housekeeping scars. `operators2d` had never been in the swap guard's **class** derive tuple, so
a `*Py` alias here would have been invisible — [[rust-phase3-barrier-state]]'s §23.7 on schedule.
And §19.7's YAML line continuation happened a **third** time, from a third tool, and never reached
CI because the scar had been turned into a test.
