---
name: beam-state
description: Phase-3 model #5b-pre (1D free-free Euler-Bernoulli beam) — the free-edge plate de-risk, built & passing; first free-boundary resonator, energy-first K from a Gram product
metadata:
  node_type: memory
  type: project
  originSessionId: 2cabff3c-9fbb-420e-9f6c-1bbbf8a6d6a5
---

Phase-3 **model #5b-pre — 1D free-free Euler–Bernoulli beam** built & all green (263 tests total).
`core/beam.py` (`FreeBeam`, implicit θ-scheme), `operators.free_beam_stiffness`,
`analysis/modal.{free_free_beam_betaL, free_free_beam_freqs, discrete_beam_eigenfrequency}`,
`tests/test_beam_{modal,energy,stability}.py`. Plan + measured results in
`docs/dev/plate-free-edge-plan.md` Part 0. This is the **de-risk rehearsal** of the free-edge Chladni
plate ([[plate-state]] §Next): it isolates the free-boundary flexural stencil + energy-first operator
symmetry **without** the 2D corners or Poisson's ratio. PDE `u_tt = -κ²u_xxxx - 2σu_t`, bending-only.

**The first FREE-boundary resonator. The construction is energy-first (build K FROM the energy, not
the reverse):** assemble the symmetric stiffness `K` as a Gram product representing the bending strain
energy `∫(u_xx)²dx`, so symmetry (→ energy conservation), the natural free BCs (zero moment `u_xx=0`,
zero shear `u_xxx=0`), and the rigid-body nullspace `{1, x}` all fall out **by construction** — never
ghost-point elimination on a boundary stencil.

**Load-bearing facts (the ones that will carry into the 2D free plate):**
- **`K = h·D2ᵀD2`**, D2 = (N−1)×(N+1) interior second-difference (curvature-quadrature weight
  Wc=h·I); mass `W = diag(h/2, h,…, h, h/2)` (trapezoidal, **half-cells at the free ends**). All
  h-quadrature weights live INSIDE K and W (no extra scalar h anywhere).
- **THE insight (advisor-flagged):** the free-end closure is supplied by the **h/2 end masses, NOT by
  hand-coded stiffness rows.** `W⁻¹K` reproduces Bilbao's free-free bar exactly — end row
  `[2,−4,2]/h⁴`, next `[−2,5,−4,1]/h⁴`, interior `[1,−4,6,−4,1]/h⁴`. **Expected to recur in 2D:** the
  edge-½ / corner-¼ `W` weights will supply the corner closure there too.
- **θ-scheme, `A = (1+σk)W + θk²κ²K`** — SPD because **W is SPD even though K is only PSD** (K has the
  `{1,x}` nullspace). Factor once with `scipy.sparse.linalg.splu` (NOT cholesky — singular K).
  θ=0.28 default. **No CFL** (unconditionally stable θ≥¼); beam-Courant `μ=κk/h²` reported only.
- **Closed-form oracle** (the reason the beam is built first, before the 2D plate which has none):
  `f_n = κ(β_nL)²/(2πL²)`, `β_nL` = roots of `cos(βL)·cosh(βL)=1` (4.730041, 7.853205, 10.995608,
  14.137165, 17.278760; → (2n+1)π/2). Found via `brentq` on the overflow-safe `cos(x)−sech(x)=0`.
  Two rigid-body zero modes `{1, x}`.
- **Generalized eigenproblem** `Kφ=μWφ`, μ=ω²/κ²→β⁴, `f=κ√μ/2π`. K is PSD ⇒ shift-invert at sigma=0
  is singular: use a **small NEGATIVE shift** `sigma=-1e-3·(4.730041/L)⁴`. Discard the 2 rigid modes.
- **Damping caveat is BROAD** (same shape as [[plate-state]]): rate `2σ(1−θQk²)` with `Q=κ²μ` is
  4th-power across the whole spectrum, so mid/high modes under-damp; assert 2σ for **low modes only**.

**Measured (test bars set from data, per project culture):** `‖K−Kᵀ‖=0`; `K@1=0` exactly, `K@x`
rel~1e-18, `K@x²` rel~1e-6 (the not-everything-killed counter-check); modal 0.18–1.45 cents at N=200
mu=0.5; convergence order → 2.00; energy drift 2–5e-12 (incl. μ=16 ≫ ¼, the unconditional showcase).

**Next:** Part 1 — **2D free-edge Kirchhoff plate + Chladni** (model #5b), task #3, per the same plan
doc: `free_plate_stiffness(Nx,Ny,h,nu)→(K,W,index_map)` with the (1−ν) Gaussian-curvature term + the
diagonal edge-½/corner-¼ W; money tests `K{1,x,y}=0` & `K(xy)≠0`; validate vs eigsh + Leissa FFFF
square + O(h²) + energy; render the curved Chladni nodal-line figures. NB: `test_plate_stability.py`
currently expects `boundary="free"` to RAISE — must flip when the free branch lands. See
[[commit-push-at-batch-end]]; mind [[respect-ruff-line-length]].
