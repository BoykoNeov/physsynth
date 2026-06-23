---
name: membrane-state
description: Phase-3 model
metadata: 
  node_type: memory
  type: project
  originSessionId: cb030015-8285-4c6c-88ad-8de6570289ae
---

Phase 3 (2D) model #4 — **circular membrane** — built & passing. Full suite now **160 tests**
(was 118 after the string family [[damped-string-state]]). First 2D resonator; crosses out of the
1D string family.

**Human decisions (2026-06-21, asked not guessed):** circular drumhead first (not rectangular);
round rim = **staircased Dirichlet mask** on a Cartesian grid; Bessel modal test judged by
**convergence-rate** (error shrinks at the ~O(h) staircase rate + a loose absolute cents bound), NOT
the 1D ~1-cent bar; rectangle built as a **harness unit-test** (exact sin·sin, O(h²)) to de-risk the
2D machinery; **interactive web viewer** chosen as the viz target but **sequenced after** the
validated solver (CLAUDE.md #4: viz depends on a validated core).

**The load-bearing fact (advisor-confirmed, made a test):** energy conservation needs only the
Laplacian's **symmetry**, not boundary fidelity. The masked 5-point Laplacian is a principal
submatrix of the symmetric full-grid operator → still symmetric → the **staircased circle conserves
E^n to ~1e-15 exactly like the rectangle**. Staircasing taxes only the *Bessel match* (→ ~O(h)),
not conservation. **Energy ⊥ geometry** — this is why "circular first" was viable with no
embedded-boundary machinery. `test_circle_conserves_like_rectangle` asserts it directly.

**2D specifics that do NOT carry over from 1D:** CFL is **λ = ck/h ≤ 1/√2** (5-point Laplacian
spectral radius 8/h², double the 1D 4/h²) — asserted at construction. And **no λ is dispersionless**
in 2D: the 5-point scheme is anisotropic (axial vs diagonal phase speed differ at every λ). The 1D
"tune toward λ=1 for exactness" headline is gone; `test_membrane_dispersion` characterizes the
anisotropy instead of bounding it.

**Files:** `core/operators2d.py` (`laplacian_from_mask` + masks + `inner2d`/`embed`),
`core/membrane.py` (`Membrane`, domain="rectangle"|"circle", same resonator interface; cross-time
potential via the same masked L, exactly like [[stiff-string-state]]), `analysis/modal.py`
(+rectangular/circular oracles, Bessel via `scipy.special.jn_zeros`, `discrete_membrane_eigenfrequency`),
`core/exciter.py` (`raised_cosine_2d`), tests `test_membrane_{energy,modal,stability,dispersion}.py`,
viz `plots.py` (`plot_membrane_field` Chladni heatmap, `save_membrane_animation`),
`scripts/diagnose_membrane.py`, plan `docs/dev/membrane-plan.md`.

**Money test = the discrete-eigenvalue oracle:** eigsh on −L → `discrete_membrane_eigenfrequency`.
Rectangle: assembled-L eigenvalues match closed-form Λ_mn to 2e-14 (proves the operator), continuum
f_mn at clean O(h²). Circle: f_disc vs Bessel converges at p≈0.66→0.87 (→ first-order staircase),
~−9 cents at N=128. End-to-end FFT fundamental matches the discrete value to −0.01 cents.

**Degeneracy trap (pre-flagged, bit me in the diagnose script only):** m≥1 circular modes are cos/sin
pairs; the numeric spectrum has BOTH copies while the (m,n) oracle list has one entry each → naive
index-pairing mis-matches past the first degenerate mode. Tests do it right (expand by degeneracy +
sort); the diagnose table had to be fixed the same way. Always match eigenvalues by sorted value +
count, never by per-peak label.

**Portability:** auto-covered — `test_stability.py`'s core-submodule sweep picks up `operators2d.py`
/ `membrane.py` against the headless/allowlist guards with no edits (both import only numpy+scipy).

**Next:** Phase 3 continues — (a) the **interactive web viewer** (precomputed frames → browser
player, Python source of truth, no solver port), then (b) model #5 the **Kirchhoff plate**
(biharmonic, Chladni patterns). θ-artifact + portability-test loose ends from
[[stiff-string-state]] still untouched (don't gate Phase 3).
