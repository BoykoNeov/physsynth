---
name: rust-phase3-barrier-state
description: "Rust migration Phase 3 batch 6 — the distributed barrier, PHASE 3 COMPLETE; a TWO-term reduction's error is correlated with its own smallness (so exactness is a claim about the LENGTH of the sum), the sister model spells the same prefactor differently, LLVM's fold hit the test written to catch it, and porting a class silently EMPTIED a parity section"
metadata: 
  node_type: memory
  type: project
  originSessionId: dd5355d3-2318-4605-8018-56ab58328d32
  modified: 2026-08-27T17:44:13.979Z
---

Phase 3 batch 6 (2026-08-27), plan §23: `physsynth/core/collision.BarrierString` — a damped stiff
string against a one-sided distributed barrier (fret buzz, sitar/tanpura jawari, tanpura thread).
**PHASE 3 IS COMPLETE**: Group B and Group C are ported. `string_geometric` is the only string left
and it is a **Phase 5** model, not a Phase 3 leftover — it needs a sparse LU as well as the banded
Cholesky. Next is `beam`, whose §4.1 SuperLU de-risking job is untouched.

Almost pure shell — primitives, vector solve and dense LU came in §16, the string in §18, the banded
solve in §15. Speed: **47.6x** at one contact node, 5.3x at 79, 2.6x at 199.

**A TWO-TERM REDUCTION'S ERROR IS CORRELATED WITH ITS OWN SMALLNESS — so bit-identity across a
ported reduction is decided by HOW MANY TERMS IT HAS.** The shell injects force through
`u[1:-1] += force_pref * (cols_mat @ f)`, a BLAS matvec on the **update path** that nothing compared
across the languages while the shell was Python on both sides. Measured over 2,000 steps: 0 of
158,000 rows differ at m=1 (one product), **2,232–3,719 differ at m=2 and NONE reach `u`**, 45,822
differ at m=79 and **30 reach `u`**. The first explanation drafted — "the correction is a small
fraction of the field, so its ulps fall off the end" — is **WRONG**, and the m=79 row refutes it
(the ratio there is *smaller*, 2.1e-3 vs 5.7e-3, and it reaches the state anyway). The real
mechanism: two doubles sum the same in either order **unless they cancel**, and a cancelled sum is
tiny — so restricted to the rows where the matvec differs, the correction is median **2.5e-18** /
max **9.3e-13** of `u` at m=2, against median **1.2e-4** at m=79. One ulp of a 9.3e-13-of-`u`
quantity cannot survive the addition; a 1.2e-4 one crosses a rounding boundary about 1 time in
1/1.2e-4 (predicts ~36 of 44,653, observed 30). **Ask how many terms a reduction has before
asserting bit-identity across it** — two is provably safe, many is not, and it costs no measurement.
Both halves are asserted: the exact m=2 test would read as "the matvecs agree" (false) without the
m=79 control beside it.

**`portable.py` was considered and REJECTED, on evidence** — the first time §18.2's manoeuvre was
declined. It buys nothing at m=1/m=2 (already exact, structurally) and nothing is *available* at
m=79 (the solve's own `G @ F` already spent it; the shell contributes ≤1.9e-14 of peak at 500 steps
against a 1e-13 bar, and first crosses at steps 1,597–3,076 where the solve's window is 1,175–1,584)
— while it would change a shipped model's reference numbers and the viewer's fret/jawari/juari
output. Rule: `portable.py` is for a reduction whose order is *the only thing* between two
implementations.

**THE SISTER MODEL SPELLS THE SAME PREFACTOR DIFFERENTLY, so "follow the bow" would have been
wrong.** `collision.py` writes `string.k ** 2 / string.rho` (`float.__pow__` → libm `pow`);
`bow.py` writes `self.k * self.k / (rho*h)` at the structurally identical spot. Different doubles in
**86 of 200,000** sample rates in range, so the Rust barrier must use `scalar_pow` where the Rust bow
correctly uses a multiply. Invisible to every energy bar, and the two models are never compared to
each other.

**§17.2's constant fold, a THIRD time — inside the test written to catch it.** The pin for the above
searches for a witness sample rate at runtime rather than hardcoding one, because *which* arguments
`pow` rounds differently is a property of the runner ([[numpy-libm-cpu-dispatch]] §22.1) and a
hardcoded witness is §21.6's machine-decided bar. Written the obvious way the Rust sweep passed in
debug and found **nothing** in `--release`: LLVM folds `powf(x, 2.0)` into `x*x`, so the search's own
predicate became a tautology and the test went green having asserted nothing. Routed through
`scalar_pow`. Second scar in the same search: the first witness a `k**2 != k*k` sweep returned had
its difference **absorbed by the `/rho`** that follows, so the negative control compared a value to
itself — **the predicate must be the whole expression**, not the sub-expression you suspect.

**PORTING A CLASS SILENTLY EMPTIED AN EXISTING PARITY SECTION.** §16 measured the vector solve
*through* the Python `BarrierString` by pinning `collision.solve_contact_vector`. The moment the
class swaps, the Rust model never looks that name up — so with the flag set every test in that
section compared Rust against Rust and passed. Verified, not assumed (a pin that raises on call was
never called). Same class as [[rust-phase2-mallet-state]]'s empty guard and §16.8's empty CI job,
third door: **a comparison whose two sides are selected by a MODULE-LEVEL NAME stops being a
comparison when that name is rebound.** When a model class ports, grep the parity suite for anything
pinning its collaborators by name.

**The class swap guard was one name short, for six batches.** §17.6's fix derives the swapped set
from `<Name>Py` aliases — but over a **hand-written tuple of modules**, and `collision` was not in
it. So a `BarrierStringPy` alias could have appeared unnoticed in the very half §17.6 claimed to be
fixing. Closed the way §17.6 says: added the module first, **watched the guard fail**, then updated
the expectation. **A derive is only as wide as the list it derives over.**

**The non-convergence warning had nowhere left to be raised from.** §16's swap block deliberately
kept it on the Python side because `stacklevel=2` names `BarrierString.step` — a frame that stops
existing once the model is Rust. No test pinned the attribution, so it would have been lost
*silently*. Decision taken rather than discovered: raise from Rust with `stacklevel=1` (naming the
Python caller of `step()`), message byte-for-byte, and a new parity test stalls both and compares the
text. Implementation detail that matters: raise **after** dropping the string's `borrow_mut` — a
`UserWarning` can be promoted to an exception and unwinding through a live PyO3 borrow panics.

**§12.2's underscore rule, in its strongest form yet: four underscored attributes are ASSIGNED by
clients**, not merely read — `_G` and `_force_pref` (doubling both is the coupling-magnitude
negative control), `_b` (flattening a curved bridge to a rail at its own crest — the jawari's
travel contrast), and `penetration` (hand-seating a static equilibrium). A getters-only binding
looked complete and left three physics tests unable to run; `_b` was **missed on the first pass and
found by running the suite, not by reading it**. `_b`'s setter deliberately does not rebuild `G` —
neither does the original, and it is right: the support is chosen by which heights are *finite*.

See [[rust-migration-state]], [[rust-phase3-collision-state]], [[rust-phase3-bow-state]],
[[barrier-collision-state]], [[jawari-state]].
