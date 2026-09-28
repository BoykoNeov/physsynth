---
name: rust-phase3-banded-state
description: "Rust migration Phase 3 batch 1 — the shared banded Cholesky; a bit-identity anchor BINDS models into one unit, so the solver ported instead of a model; the first swap that changes the numbers; shim validation cost more than the port saved"
metadata: 
  node_type: memory
  type: project
  originSessionId: 53a1f7af-812d-4703-84a6-f71463f2c050
  modified: 2026-08-27T06:17:04.349Z
---

Phase 3's first batch (2026-08-27) ported **no model**. It ported `physsynth/core/banded.py`, the
banded Cholesky that four theta-scheme strings share. Plan `docs/dev/rust-migration-plan.md` §15.

**The headline, and it is about `tests/` not about arithmetic: a bit-identity anchor between two
DIFFERENT model classes is a porting constraint that binds them into one indivisible unit.** Three
`array_equal` reduction anchors chain `string_stiff` ⟷ `string_damped` ⟷ `string_nonlinear` ⟷
`string_geometric` (`sigma1=0`, `EA=0`, `EA=T`). Port one and an intra-Python comparison becomes a
cross-language one — which the banded solve cannot carry. And one of the four is Group D, i.e.
Phase 5. So the plan's "port `string_stiff` first" is unrunnable, and the way out is Phase 1's move
one level down: port the **solver**, the thing four models are built out of. Every anchor stays
valid because all four call the same code. **Before porting any model, grep the suite for
`array_equal` against a different class** — same-class comparisons (bowed string vs bare string)
survive any port; cross-class ones are a chain, and the chain is the unit of work.

**The factor is transcribable, the solve is not, and both halves were measured.** SciPy →
`dpbtrf` (unblocked `DPBTF2` at kd=2) and `dpbtrs` (two `DTBSV`). Factor agreement with OpenBLAS
over 120 of the family's matrices: **120/120 with reciprocal-once AND fma; 82/120 with
reciprocal-once alone; 19/120 dividing per element.** `DSCAL` forming `1/ajj` once and multiplying
is the reference algorithm's own behaviour → transcribed. The fma is a `DSYR`-kernel property
picked by `DYNAMIC_ARCH` → **deliberately not** transcribed (§14.2's line). The solve admits **no
scalar recipe at all**: per-element elimination over {forward,reverse}×{plain,fma}×{divide,recip}
gave disjoint candidate sets (element 2 needs divide, 5 needs fma, 9 needs forward, 14 excludes
forward) and **element 7 admitted nothing**. `DTBSV` is blocked/vectorised.

**This is the first swap in the migration that changes the numbers on purpose.** Group A's
"~1e-13 over a short run" is run-length-dependent and shorter than it looks: a fed-back *reduction*
held it to 2,000 steps ([[rust-phase2-radiation-state]] §14.4); a fed-back **solve** holds it only
to ~**100** (1.1e-13 at 100, 9.7e-13 at 2,000, 3.2e-12 at 20,000, as a fraction of amplitude).
**The energy bar does not move** — lossless drift 1.16e-12 Rust vs 1.14e-12 LAPACK, bar 1e-10. What
replaces bit-identity as the sharp claim is that the four models still agree with **each other**
exactly.

**Validation written into a swap block is on the hot path.** One `np.isfinite(ab).all()` in the
Python shim turned a **1.47x win into a 0.96x loss** — the win being spent is per-call overhead
([[rust-phase2-state]] §11.6) and that check is another call of exactly that kind. Moved into the
pass the binding already makes: primitive 2.2–2.9x (solve) and ~4x (factor); whole model 1.47x /
1.35x / 1.17x at N = 64 / 256 / 1024. It also *has* to be there — a NaN diagonal otherwise comes
back as `NotPositiveDefinite` instead of `ValueError`.

Smaller scars: `LinAlgError` is **not** a `ValueError` subclass, so the binding raises its own
`NotPositiveDefinite` and the shim re-raises the type SciPy promises; `b/√a/√a` **is not** `b/a`
(differs in the last bit at a=2), so the native test asserts the form the algorithm computes;
`kappa=0` hands over a pentadiagonal band whose second superdiagonal is *numerically* zero, not
structurally absent, so kd=2 must equal kd=1 to the bit. **Nothing was linked** — the Cargo
allowlist is still empty, ~150 lines of transcription sufficed, and §4.1's SuperLU hypothesis stays
untested so `beam` keeps its Phase 4 de-risking job.

Next: `collision` (Group C, dense LU) — it *wraps* a `DampedStiffString` rather than being one, so
it does not join the chain. See [[rust-migration-state]] and [[rust-phase2-radiation-state]].
