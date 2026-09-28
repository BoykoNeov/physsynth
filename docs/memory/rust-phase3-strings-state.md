---
name: rust-phase3-strings-state
description: "Rust migration Phase 3 batch 3 — the two theta-scheme strings; both obstacles were an evaluation ORDER, fixed on the Python side; Phase 1's canonical-Csr reason expired but the decision survived; a linear model's divergence is a random walk"
metadata: 
  node_type: memory
  type: project
  originSessionId: 8cc22250-fe57-44ef-8044-aad508516b6f
  modified: 2026-08-27T09:01:19.403Z
---

Phase 3 batch 3 (2026-08-27): `string_stiff` and `string_damped` — models #2 and #3 — are Rust.
First models out of the four-string chain. Plan §18. See [[rust-phase3-banded-state]] for the
solver that had to go first, and [[rust-migration-state]] for the destination.

**The headline: neither obstacle was in Rust.** §15.9 predicted that with the solver common the
bit-identity anchors would stop blocking. They still blocked, twice, and both times the blocker was
an **order of evaluation** SciPy picked that no portable implementation reproduces:

1. **The reduction.** `np.dot` is BLAS `ddot`; it disagrees with a left-to-right sum in **16,797 of
   20,000** vectors at n=99. All three chain anchors assert `a.energy() == b.energy()` *across model
   classes*, so a Rust model's energy has to equal a Python model's exactly.
2. **The matrix's column order.** `biharmonic_matrix` is `D2 @ D2`, and SciPy's sparse-product
   kernel emits each row as a **stack** — columns DESCENDING, `has_sorted_indices == False`. A CSR
   matvec sums a row in *stored* order, so `L @ u` differs between the two spellings in **2,000 of
   2,000** vectors at every grid size. And `L @ u` builds every timestep's right-hand side, so this
   is a different trajectory from step one, not a different reported number.

**Both fixed by changing the PYTHON side**, in a new module `physsynth/core/portable.py` holding
`dot` (left-to-right, spelled `np.cumsum(a*b)[-1]` — the one NumPy reduction that is sequential by
construction, 0/3300 disagreements with a naive loop, still compiled) and `canonical` (sorts the CSR
indices). Applied to all four theta-scheme strings at once. **This is the Phase 1 manoeuvre a third
time and one level lower again**: `operators` was not a model (swung five), `banded` was not a model
(swung four), this is neither a model nor a solver but an *order* (swings the same four).

**It changes the reference implementation's numbers unconditionally** — unlike `banded`, whose swap
only changes them under the flag. Gating a *Python* model's arithmetic on an env var would be worse.
156 string tests green on the default path; no physics bar moves.

**`np.sum` / `arr.sum()` / `np.add.reduce` are PAIRWISE above blocksize 128** — a third answer, not
a fix. `math.fsum` is correctly-rounded — a fourth.

## The decision that had to be re-examined

Phase 1 kept the Rust `Csr` canonical (ascending) and gave two reasons. One — *"nothing downstream
reads `.data` or `.indices`"* — **expired** the moment a model started multiplying by the operator in
its inner loop. The other — *"reproducing SciPy's stack order pins the port to a SciPy internal"* —
got **stronger**: under the alternative a SciPy point release reordering SMMP would silently move
every string trajectory in the project. General form: **a decision justified by "nothing downstream
depends on this" must be re-examined when something downstream ports, and the question is whether
the conclusion survives losing its stated reason.** Here it did, on the other argument.

**This is latent in the plate family.** `plate.py` does `self.B @ self.u_prev` every step with `B`
from `biharmonic_matrix`; `beam.py` does the same with `K`. Phase 5 hits the same wall and the
answer is written down in advance (§18.4). Deliberately NOT fixed now — those models have shipped
parity measurements built on their current behaviour.

## The other two things worth carrying

**Ask whether a reduction reaches the next timestep, not whether it is a reduction.**
`string_nonlinear._stretch` is also an `np.dot` and sits *inside* the tension solve's residual —
changing it would move model #9's trajectory. Left on `np.dot` deliberately; the anchors don't reach
it (`EA = 0` returns early). A port of #9 must therefore **match** a BLAS reduction rather than route
around it — the first time this migration faces that.

**A linear model does not amplify a difference at all** — the third agreement regime, next to the
barrier's chaotic separation ([[barrier-collision-state]], ~1.2k steps) and the mallet's transient
non-separation ([[rust-phase2-mallet-state]], never). With a shared solver the two strings are
bit-identical at 20,000 steps. Without one (SciPy's blocked `DTBSV` vs the transcribed reference
`DTBSV`, §15.3's gap, not this batch's) the divergence grows like a **random walk** — 1.7e-14 →
4.8e-14 → 8.6e-14 → 1.6e-13 across 100/500/2,000/20,000 steps, ~sqrt(n). **What sets the window is
amplification, not perturbation size.**

## Smaller, but load-bearing

- **The two Rust cores are deliberate near-duplicates.** One superset `Params` would be ~150 lines
  less and would make the `sigma1 = 0` anchor **vacuous under the flag** — an anchor compares two
  transcriptions, and two names for one implementation compare equal for free. §17.6's lesson.
- **The class half of the swap guard is now DERIVED** (it was a block of hand-pasted `assert X is
  physsynth_rs.X` lines with exactly the hole §17.6 found in the function half).
- **`A` is never assembled** — only its three diagonals are read, and `csr.diagonal(d)` is
  order-independent, so the sort provably cannot move the Cholesky factor. Measured over 288
  parameter combinations before any Rust was written: **0** bands changed. That was the one
  assumption that would have invalidated the batch silently.
- **A parity test must be able to separate its causes.** Bit-identity here is only claimable with a
  shared banded solver, so `tests/test_rust_parity_strings.py` patches the captured
  `cho_solve_banded` — which is what makes "bit-identical" a claim about the *port* rather than about
  OpenBLAS.

Measured: 215 native tests (both profiles, now enforced in CI); 1,104 parity tests flagged /
1,103 + 1 skip unflagged; 297 in the batch's flagged CI step. Speed: **19.8x / 10.1x / 4.1x / 1.8x /
1.2x** at N = 16/64/256/1024/4096 — §11.6's crossover, unchanged in shape.

**Next:** `string_nonlinear` (last chain member that avoids Group D), then `bow` (whose Newton
iteration count is compared by nothing in the repo — that has to be added).

**`portable.py` itself is GONE (2026-09-07).** The two spellings outlived their callers by four
days: every model that imported them is Rust, `airbox.py` inlined the `sort_indices()` call, and a
sweep found zero importers anywhere in the repo — no test, no script, no viewer, and not the
binding's four `import("physsynth...")` reaches. Deleted with no guard edit, because the module
population is scanned by `pkgutil.iter_modules` with **named positive controls** rather than a
count (see [[rust-deletion-connection-state]]). The *manoeuvre* is what these notes are about and
it is undamaged — the finding lives in `docs/dev/rust-migration-plan.md` §18.2 and the Rust doc
comments that used to point at the file now point there.
