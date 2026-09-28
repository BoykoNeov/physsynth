---
name: rust-phase5-matrices-state
description: "Rust migration Phase 5 batch 2 (the plate's matrices) — values vs stored ORDER are two separate questions; the free plate needed nothing; a reassociation is invisible when the outer factors share a mantissa"
metadata: 
  node_type: memory
  type: project
  originSessionId: bcbe663c-9f10-40ad-8910-96ad32f14ef0
  modified: 2026-08-28T16:51:07.940Z
---

Phase 5 batch 2, shipped 2026-08-28: `biharmonic_from_mask`, `dirichlet_interior_d2_1d`,
`orthotropic_biharmonic` and both `free_plate_stiffness*` ported to Rust, plus `Csr::add` /
`kron` / `identity`. Plan §26. **All bit-identical** — no tolerance-level row in the batch.

**The headline: values and stored order are TWO questions.** Every previous batch that met a SciPy
sparse product asked whether the *values* could be matched. Here they matched immediately and
everywhere (an ascending-`k` accumulation reproduces SMMP bit for bit, 0 differing entries of 2,629
at N=16, across 7 grids and 2 staircased outlines). Only the **stored column order** differed — and
that reaches the trajectory because a CSR matvec sums a row in stored order and `Plate.step` forms
`B @ u` twice per timestep. Fix: `portable.canonical` on the Python side, which is [[rust-phase3-strings-state]]'s
manoeuvre landing where §18.4 predicted. **§18.2's own rule was too specific**: it found the 1-D
biharmonic **descending**; in 2-D the order is **neither ascending nor descending in 600 of 610
rows**. Re-measure a rule before relying on it somewhere new.

**The free plate needed nothing, and that is what made the batch safe.** `free_plate_stiffness`'s
`K` is a `AᵀWB` Gram product and SciPy returns those already sorted — measured canonical in *every*
row of every rectangle, disk and guitar. So only the **supported** plate moved (1.2e-13 of amplitude
over 2,000 steps at N=12; drift unmoved, 2.2e-14 → 2.3e-14 against the 1e-10 bar) and the free,
orthotropic-free and guitar plates are bit-identical. Had `K` needed the sort, [[rust-phase4-beam-state]]'s
t² integration along the `{1,x,y}` nullspace would have turned porting a matrix into a re-tolerancing
of three models.

**Ask whether the outer factors share a mantissa before assuming a reassociation is visible.**
`operators2d.py` writes `C2x.T @ (Wa @ C2y)` (right-associated, explicit parens) and
`AiryStressSolver` writes `Lc_r.T @ Wa @ Lc_r` (left, by Python's rule) — the same form `BᵀWB`,
opposite association, and a shared helper would silently pick one. ~35% of random value triples
distinguish `(x·w)·z` from `x·(w·z)` (69,943/200,000) — and **none** of this operator's do, because
every curvature entry is `1/h²` times an exact power of two, so both reduce to the same product by
commutativity. Costs no measurement; [[rust-phase3-barrier-state]]'s cheap-question move applied to a
product instead of a sum.

**A spelling pin must SEARCH, not assert a constant.** The association witness was first written as
three hand-picked numbers; they landed in the agreeing two-thirds and the test went **red**. That is
the empty-witness-search hazard seen from the other side — only a searching form can distinguish
"no difference exists" from "I did not look in the right place".

**Two model classes were bound by an anchor again.** `plate.py` spelled `L @ L` **inline in two
classes** (`Plate` and `VKPlate`) and never called `biharmonic_from_mask` at all; `VKPlate(nonlinear=False)`
must be `array_equal` to `Plate`, so both call sites had to move to the shared builder in one edit.

**The swap guard caught its own hazard inside the batch that made it.** `dirichlet_interior_d2_1d`
is the module's first private-but-swapped name; the natural alias `_dirichlet_interior_d2_1d_py` is
invisible to a derive that filters leading underscores. Use `collision`'s convention — bare
`<name>_py` for an underscored function.

**No speed win, and none expected:** Rust `free_plate_stiffness` is **slower** (3.49 ms vs 2.53 ms
at N=32) — five compiled SMMP calls with nothing around them to win. Construction-time only.

**Next batch (3):** what is left of `operators2d` is the von Kármán half — `_collocated_d2_1d`,
`_forward_d1_1d`, `_centered_d2_1d`, `_avg_d1_1d`, `VonKarmanBracket`, `AiryStressSolver`. Its claim
is a **tolerance**, settled in advance by [[rust-phase4-beam-state]] (SuperLU is supernodal).
`_clamped_d2_1d` is built through `lil` + two scalar assignments — the only construction in the
module that is not a single kernel call, so check its index order rather than assuming.
