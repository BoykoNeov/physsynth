---
name: plate-state
description: Phase-3 model
metadata: 
  node_type: memory
  type: project
  originSessionId: 7bbe6ebf-45e7-42c3-9a97-8278f943879b
---

Phase-3 **model #5 — Kirchhoff plate, simply-supported (Navier) rectangle** built & all green
(231 tests total). `core/plate.py` (implicit θ-scheme), `operators2d.biharmonic_from_mask`,
`analysis/modal.py::{rectangular_plate_freqs, discrete_plate_eigenfrequency}`,
`tests/test_plate_{energy,modal,stability}.py`, `scripts/diagnose_plate.py`. Plan + build results in
`docs/dev/plate-plan.md`. Human decision (2026-06-23): **SS now, free-edge Chladni as a follow-on**.

**The plate is the composition of the two prior advances — no new machinery:** membrane's masked
Dirichlet Laplacian `L` ([[membrane-state]]) + stiff-string's implicit θ-scheme & biharmonic-by-
squaring ([[stiff-string-state]]). The 2D biharmonic `∇⁴=(∇²)²` is **`B = L @ L`**; because `L` is
the Dirichlet (zero-ghost) Laplacian, `L²` **automatically** bakes in both SS conditions (`u=0` and
`∇²u=0`) and keeps `sin·sin` an exact discrete eigenvector with eigenvalue `Λ²` — the 2D analog of
the 1D `D2²`. Operator/energy use the assembled `B` everywhere (P(f,g)=κ²h²(Bf)·g).

**Load-bearing facts (advisor-caught traps that mechanical template-reuse would break):**
- **Bending-ONLY** — `𝓛 = −κ²B`, *no* `c²∇²` wave term in operator or energy. Oracle is pure
  4th-power: `f_mn=(π/2)√(D/ρ_s)[(m/Lx)²+(n/Ly)²]`. One param `κ=√(D/ρ_s)` (ν drops out for SS).
- **No κ=0 regression anchor** (κ=0 ⇒ `u_tt=0`, degenerate). Operator-correctness money test instead:
  assembled `B=L²` eigenvalues == squared Laplacian eigenvalues `Λ²` (got 2.4e-13, machine precision).
- **Damping caveat is BROADER than the stiff string**: per-mode rate `2σ(1−θQk²)` with `Q=κ²Λ²` is
  4th-power across the *whole* spectrum (no gentle `c²p²` term), so θ-scheme under-damping bites
  mid-spectrum, not just top partials. Passivity still unconditional. Cure = freq-dependent loss (later).
- **Solver = `scipy.sparse.linalg.splu`** (factored once), NOT `cholesky_banded`: `B=L²` is a 13-point
  stencil (bandwidth ~2Nx) and scipy has no sparse Cholesky.
- **`Q=κ²·Λ²` uses the biharmonic eigenvalue** (Λ=Laplacian eigenvalue, biharmonic=Λ²). Pin it.
- **Unconditional for θ≥¼** (θ=0.28 default, inherited). No CFL; plate-Courant `μ=κk/h²` reported only
  (explicit bound μ≤¼). Energy drift 2e-13 even at μ=16 — the unconditional-stability showcase.
- **TIGHT bar held** (machine-precise eigenvalues ⇒ 1D-style ~1-cent), NOT loosened like the membrane
  circle's staircase tier (that loosening was a staircase artifact absent here). Low modes 0.15–0.62 c
  at N=96; FFT 0.01 c. SS nodal lines = rectangular GRID, not curved Chladni (those need free edges).

**Gotcha:** `Ly` is **snapped** to an integer number of cells at construction (square cells). Any
oracle/comparison must use the snapped `p.Ly`, not the requested value — a ~0.26% mismatch reads as a
spurious ~10-cent error (this bit the diagnose script; tests use a square plate so were immune).

**Next:** free-edge Chladni plate (no closed form → validate via `eigsh` + Leissa frequency tables +
energy; ν re-enters) and/or a web-viewer plate panel ([[web-viewer-state]]), then **model #6
nonlinear (von Kármán) plate** — the gong/cymbal deep end. See [[commit-push-at-batch-end]].
