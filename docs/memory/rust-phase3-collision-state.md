---
name: rust-phase3-collision-state
description: "Rust migration Phase 3 batch 2 (collision) — a vectorized NumPy function is TWO computations; the default fixture does not exercise the solver; and a chaotic model's agreement window is set by its dynamics"
metadata: 
  node_type: memory
  type: project
  originSessionId: 79289d02-fa09-4b1b-b551-bb99cbf7f17d
  modified: 2026-08-27T07:00:31.083Z
---

Phase 3 batch 2 (2026-08-27) ported `physsynth/core/collision.py` — the contact primitives, the
scalar solve (the mallet's) and the vector solve (the barrier's) — plus `dense`, the project's one
dense LU (Group C). `BarrierString` itself did **not** port: it wraps a still-Python
`DampedStiffString`, the same reason [[rust-phase3-banded-state]] ported no model. Under the flag
this swings *two* models anyway, because `mallet.py` re-exports these primitives — which is
§11.2.2's "Phase 2 finishes after Phase 3 starts" coming due. **`mallet` is the next batch and it
closes Phase 2.**

**The finding, and it is about NumPy rather than about Rust: a vectorized function called with an
array and with a scalar is TWO different computations.** NumPy's float64 power *ufunc loop* carries
a fast-path ladder for the exponents -1, 0, 0.5, 1 and 2 — `x**0.5` becomes `sqrt`, `x**2` becomes
`x*x` — and its *scalar* path takes no shortcut and calls libm `pow`. Measured over 200k
penetrations they disagree in 94 cases at exponent 0.5 and 53 at exponent 2. The exponents this
module uses are `α+1`, `α`, `α−1`, so the split lands exactly on `α=1` (the closed-form-oracle
case) and `α=1.5` (the barrier default). The Rust side therefore carries **both spellings** and
picks by argument rank. A corollary that is a fact about the Python original: `_force_total_vec`'s
docstring claiming it is "numerically identical to calling contact_force_total per component" is
**wrong** — 174 of 200,000 disagree at `α=1`, to 1.5e-12. Fixing it would move both models'
trajectories, so the port reproduces it.

**The second finding is about what the tests can see, and it generalises [[rust-phase2-radiation-state]]'s
blindness: the fixture the suite uses most is the one that does NOT exercise the thing being
ported.** The divergence tracks how far the Newton Jacobian `I + G·diag(F')` is from the identity —
*not* the number of contact nodes. At the default barrier (`K=1e6`, `α=1.5`) cond(J) is 1.004, so
the new dense LU is effectively solving `I·δ = -r` and **79 nodes in contact still come out
bit-identical for 2,000 steps**, even though the factor differs from LAPACK in 5,088 of 6,241
entries. Raise `K` to 1e8 (cond 1.14) and it separates by step 250. So a parity test must bring a
fixture chosen to exercise the **solver**, not the physics — and the test asserts *both* halves
(soft identical AND stiff non-zero) so the stiff case cannot be quietly dropped.

**The third retires a question rather than answering it: for a nonlinear model the agreement window
is set by the DYNAMICS, not by the port.** A string buzzing on a one-sided barrier is chaotic, so
the trajectories do not drift apart, they **separate**: first over 1e-13 at step 1,175 (`α=1`) and
1,584 (stiff), then 3.4e-12 at 10,000 and **1.1e-7 at 20,000**. Every earlier batch's divergence
grew like the run length; this one grows like an exponent. The physics bars do not move (1.12e-12
vs 1.14e-12, bar 1e-10). Before porting the four remaining nonlinear models the question is "how
long before it cannot be compared", not "how well does it agree".

**The cause-separator that made any of this attributable: the single contact node.** With one
finite barrier node `G` is 1×1, the matvec is one multiply and the LU a scalar divide, so it *must*
be bit-identical — and it is, as is `m=2`, the whole scalar solve (18k configs), and the entire
mallet trajectory including its `brentq` fallback counts. Run it with the string's banded solve
left on SciPy, or §15's change confounds the reading.

**What did NOT fire, and was watched for:** the Armijo test `0.5*(r_try@r_try) < (1-1e-4*t)*f0` is
two reductions inside a **branch**, where a last-bit flip halves the step and changes the iterate by
O(1). `newton_iters` came out identical at every step out to 20,000 across every fixture. §15.9
noted nothing in the repo compares iteration counts; the parity test now does.

Smaller things worth keeping: the LU chases **pivot equality** (a discrete decision) but not
arithmetic — `IDAMAX` is strictly-greater so a tie takes the FIRST row; `np.max(np.abs(r))`
propagates NaN where `f64::max` discards it (that comparison decides whether to warn); the Armijo
loop has **no failure exit** (40 rejections and the step is taken anyway); `G` is *borrowed* through
`PyReadonlyArray2` rather than copied per step; and the non-convergence warning stays in Python
because `stacklevel=2` cannot mean "my caller" from inside an extension module.

Speed: **2.97x** (barrier, m=79), **3.39x** (m=1), **1.42x** (mallet's scalar solve). Native tests
189 (from 174); parity tests 74; the Cargo dependency allowlist is still **EMPTY**.
