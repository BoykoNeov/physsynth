---
name: rust-phase7-rotating-wave-state
description: Phase 7 batch 3, the rotating-wave BVP — **`analysis/` FINISHED**; a margin measured at ONE fixture is a claim about one fixture (the iteration count moved in 17/108 and the first header said it didn't), a differing iteration count is only a defect when the iterate is FED FORWARD, and the natural LU order fills this matrix DENSELY
metadata:
  node_type: memory
  type: project
---

Phase 7 batch 3 (2026-09-03) ported `physsynth/analysis/rotating_wave.py` — the geometrically exact
string's Tier B oracle, the exact rotating helix. **`physsynth/analysis/` is now FINISHED**, and with
`core/` already done ([[rust-phase5-connection-state]]) **the translation of models is over**. What
is left of the migration: the test-suite port, the viewer *import audit* ([[viewer-stays-python]]),
the deletions. Plan §38; ledger #33 and #34.

**The package's only nonlinear solve** — Newton on a sparse Jacobian inside an 8-step amplitude
continuation — so the only Group D member outside `core/` ([[rust-phase4-beam-state]]).

# The batch's finding: a margin measured at ONE fixture is a claim about ONE fixture

**I got this wrong in the module header and the parity file caught it inside an hour.** Following
§36.2 ([[rust-phase7-spectrum-state]]) I measured the margin *before* porting: perturb the Python
Newton step by a relative 1e-10 — six orders beyond anything two LU implementations differ by — and
`Ω` moves **one ulp** while the iteration count sits unmoved at 24. I wrote that the branch hazard
[[rust-phase3-tension-state]] records "does not exist here". **False.** That probe held `N=32`,
mode 1, `κ=0`. Over a 126-fixture grid the count differs in **17 of the 108 that converge, by up to
13**. The fix is a **grid, not a better probe**.

**But the hazard is real AND harmless, and separating those is the actual lesson.** In a timestepping
model the iterate **is** the state fed forward, so a differing count differs the trajectory. Here the
iterate is *discarded* and only the root survives — the widest gap (28 iters vs 41) still agrees on
`Ω` to **9.0e-16**. So: **ask what the solver RETURNS, not how it got there.** Ledger #33.

Measured agreement, 126 fixtures: `Ω` worst **8.4e-15**; `converged` agrees **126/126** (the same 18
fail on both sides). Far inside the Group D tolerance budget §24 granted.

# The natural LU column order fills this matrix DENSELY — §29.2 a second time

Unknowns are `[φ; ψ; s]`: two `(N-1)` blocks stacked **BY FIELD**, bordered by the amplitude row and
the `∂F/∂s` column — while the φ↔ψ coupling is **cell-local**. So every coupling sits `N-1` columns
off the diagonal and the elimination fills the whole envelope. `nnz(L+U)` natural is **`≈dim²/2`** —
dense triangular factors. Interleaving `(φ_i, ψ_i)` by node with `s` last:

| N | dim | COLAMD | natural | interleaved |
|---|---|---|---|---|
| 32 | 63 | 1,128 | 2,278 | **593** |
| 512 | 1,023 | 9,949 | 528,139 | **9,944** |

Beats COLAMD outright at small N, matches it at large N, **closed form in N** so no ordering
heuristic ports. `SparseLu::factor_permuted` already existed — [[rust-phase5-geometric-state]] built
it for the identical shape on the core's own Jacobian. **No physics bar in this project could catch
the dense version**: same root, dense factorization. A cost cliff.

# Two more things worth carrying

**Where the BVP does NOT converge, compare only the flag.** 18/126 fail; past that `Δφ/φ` reaches
0.5 and nothing is asserted. Same boundary as that morning's `λ_long` sweep
([[geometric-string-state]]): **a margin measured in the converged regime says nothing about the
failing one** — there the iteration path *is* the answer, not a means to it.

**Two last-bit divergences REFUSED, on ledger precedent.** `1/lam**3` ≠ `1/(l*l*l)` (NumPy's `**3`
calls `pow`; 191/768 entries differ by ≤5.8e-16) and a Python **scalar**'s `**2` goes through libm
`pow` (1/200, 1.4e-16). Matching means `powf(const)` in Rust — **precisely** the spelling
[[rust-phase2-mallet-state]] records turning CI red, because LLVM folds it back in release. One ulp
on a tolerance-ported limit oracle does not buy re-entering a twice-recorded trap.

**Crate boundary rule (ledger #34):** `sparse.rs` + `sparse_lu.rs` join `root.rs` as `#[path]`
includes. The line is **what the shared text IS** — a sparse LU and a root-find are numerical methods
with no physics in them; the second difference and the SBP pair are **the discretisation under test**
and are rebuilt locally, or the oracle would agree with a divergent core by construction.

**Binding returns a TUPLE not a `#[pyclass]`** — not because of [[rust-phase5-membrane-state]]'s
read-only-getter trap (grepped: nothing assigns, nothing calls `_replace`) but because consumers read
it **as a tuple**. PyO3 caps `IntoPyObject` at **12** elements and this carries 15, so it splits along
arrays/scalars. The non-convergence **warning** is raised Python-side from two extra returned values.

**Cross-crate identity test (`planar_hessian_cells == 2·core._dg_jacobian`) STAYS in Python** —
a native test needs the dep edge `tests/deps.rs` refuses. Decided up front, not discovered at the end.

**§19.7's line continuation, a SEVENTH time**, from a seventh tool, adding the parity file to CI.
`test_ci_workflow.py` caught it again. Seven occurrences, zero reaching CI.

Green: 29 unmodified Python tests under the analysis flag, 100 across the geometric family with both
flags, 408 web-backend, **1,418** in the derived instrument-clients set, 22 parity, 11 native bars,
`cargo test` in **debug and release**, `clippy -D warnings` clean.
