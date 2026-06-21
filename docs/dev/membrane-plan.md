# Membrane (2D) — Phase 3 Plan (model #4, circular drumhead)

> **Status: PLANNED (2026-06-21).** First 2D model. Crosses from the 1D string family (#1–3) into
> Phase 3 (HANDOFF §5 row 4, §9). Core in `core/membrane.py` + `core/operators2d.py`, oracles in
> `analysis/modal.py`, suite in `tests/test_membrane_*.py`, viz extended in `viz/plots.py`.
> The **interactive web viewer** (precomputed frames → browser player) is a *separate, later*
> deliverable, sequenced after this solver+harness passes (CLAUDE.md #4: viz depends on a validated
> core, never before it).
>
> **Human decisions taken (2026-06-21):**
> - **Circular membrane first** (the canonical drumhead, Bessel-zero oracle) — not rectangular.
> - Round rim realised as a **staircased Dirichlet mask** on a Cartesian grid.
> - **Modal criterion = convergence-rate** (Bessel error shrinks at the staircase rate under
>   refinement + a loose absolute bound), *not* the 1D ~1-cent bar. Energy test stays at 1e-10.
> - Rectangle is the **harness unit-test** (exact sin·sin modes, O(h²)) — built first to de-risk the
>   2D machinery, but circular stays model #4.

## Goal

The first 2D resonator and the "visual showpiece" milestone: a vibrating drumhead. The deliverable is
again **the resonator + the rig that measures its deviation from theory** — here the theory is the
Bessel-zero modal series. HANDOFF §5 row 4 names the validation: `f_{mn} = c · j_{m,n} / (2π a)`.

## Physics

Transverse displacement `u(x,y,t)` of a uniform membrane under tension:

```
u_tt = c² ∇²u − 2σ u_t,     c² = T/ρ,     ∇² = ∂_xx + ∂_yy
```

- `T` = tension per unit length (N/m); `ρ` = areal density (kg/m²) → wave speed `c = sqrt(T/ρ)`.
- Fixed rim (Dirichlet `u = 0` on the boundary) — the only termination with a clean closed-form
  oracle, and the physically right one for a clamped drumhead.
- `σ ≥ 0`: the same frequency-independent loss as model #1 (`−2σ u_t`). Frequency-dependent
  membrane loss is deferred (it mirrors model #3's `σ₁` machinery and is not needed for #4).

Continuous energy (kinetic + strain), with `∫∫ over the domain Ω`:

```
E(t) = (ρ/2) ∫∫ (u_t)²  +  (T/2) ∫∫ |∇u|²
dE/dt = T ∮_{∂Ω} u_t (∇u·n)  −  2σ ρ ∫∫ (u_t)²
```

Fixed rim kills the boundary flux (`u_t = 0` on `∂Ω`), so lossless ⇒ `dE/dt = 0`, lossy ⇒ `≤ 0`.

### Modal oracles

- **Circular (model #4), radius `a`:** modes `J_m(j_{m,n} r/a)·{cos,sin}(mθ)`,
  ```
  f_{mn} = c · j_{m,n} / (2π a),    j_{m,n} = n-th positive zero of Bessel J_m.
  ```
  `m=0` modes are non-degenerate; `m≥1` come in a cos/sin degenerate pair. `scipy.special.jn_zeros`.
- **Rectangular (harness unit-test only), `[0,Lx]×[0,Ly]`:** modes `sin(mπx/Lx)·sin(nπy/Ly)`,
  ```
  f_{mn} = (c/2)·sqrt( (m/Lx)² + (n/Ly)² ).
  ```
  These `sin·sin` products are **exact discrete eigenvectors** of the 5-point Laplacian (tensor
  product of the 1D case), so the rectangle is the clean O(h²) reference that pins down a harness
  bug *before* the staircase error enters.

## Scheme — explicit 5-point FDTD

```
u^{n+1} = 2u^n − u^{n-1} + c²k² (Δ_h u^n)               (lossless)
Δ_h u[i,j] = (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] − 4u[i,j]) / h²   (square cells, h_x=h_y=h)
```

With loss, the same `(1±σk)` split as `IdealString`:
`u^{n+1} = (2u^n − (1−σk)u^{n-1} + c²k² Δ_h u^n) / (1+σk)`.

### 2D CFL — λ ≤ 1/√2 (NOT 1), and no dispersionless λ

The 5-point Laplacian's spectral radius is `8/h²` (double the 1D `4/h²`). Positivity of `E^n` ⇒

```
λ = c k / h  ≤  1/√2 ≈ 0.7071.
```

Assert this at construction (generalises the CLAUDE.md `λ≤1` rule). **Critically, the 1D "λ=1 is
exact / zero dispersion" headline does NOT transfer:** the 5-point scheme is *anisotropic* — axial
and diagonal phase speeds differ at every λ — so no λ removes numerical dispersion in 2D. The
dispersion diagnostic should *show* this direction-dependent error, not try to tune it away.

### Discrete energy — cross-time potential via the masked Laplacian

Mirror the 1D/stiff form. Let `L` be the **masked 5-point Laplacian matrix** (below) on the live
(interior) unknowns; `−L` is symmetric positive-definite. With the 2D inner product
`⟨f,g⟩ = h² Σ f g`:

```
E^n = ρ [ ½ ‖δ_t- u^n‖²  +  (c²/2) P(u^n, u^{n-1}) ],     P(f,g) = ⟨−L f, g⟩ = ⟨∇_+ f, ∇_+ g⟩ ≥ 0
```

The potential is the gradient inner product **across two time levels** (the non-obvious move that
makes conservation exact, HANDOFF §4.1). Evaluating `P` through the *same* matrix `L` used in the
update makes `E^{n+1} = E^n` an exact algebraic identity (machine-precision lossless; monotone
decreasing lossy). Fixed rim ⇒ boundary velocity is 0, so no trapezoidal edge-weight subtlety
(unlike the 1D free boundary). `set_state` uses the Taylor start `u^{-1} = u^0 − k v^0 + ½c²k² L u^0`.

## The masked Laplacian — why staircasing keeps energy exact

Mark a grid node **live** (an unknown) iff it lies strictly inside the domain; everything else is held
at `u=0` (Dirichlet). The 5-point stencil at a live node references 4 neighbours; a neighbour that is
not live contributes its `u=0` (a zero ghost) and simply drops from the row. The resulting `L` on the
live nodes is a **principal submatrix of the symmetric full-grid Laplacian → still symmetric.**

> **Load-bearing fact (advisor-confirmed): energy conservation needs only symmetry, not boundary
> fidelity.** So the staircased circle conserves `E^n` to 1e-10 *exactly like the rectangle*. What
> staircasing taxes is the **Bessel match**: the staircased domain isn't a true circle, so its
> eigenvalues converge to `j_{m,n}` at ~O(h), not the clean O(h²) of 1D. **Energy and geometry
> decouple** — that is the whole reason "circular first" is viable without embedded-boundary
> machinery.

`L` is built once from the boolean mask + an index map `(i,j) → flat unknown`. Rectangle mask = all
interior nodes (recovers the exact tensor-product operator); disk mask = `(x−x_c)²+(y−y_c)² < a²`.

### Discrete eigenfrequency oracle (the tight "money" test)

Insert an eigenmode `u = φ z^n` with `L φ = −Λ φ` (`Λ>0`) into the scheme:
`cos(ωk) = 1 − c²k²Λ/2`, so each eigenvalue `Λ` of `−L` gives a discrete temporal frequency

```
f_disc(Λ) = arccos(1 − c²k²Λ/2) / (2π k).
```

Compute the low eigenvalues of `−L` with `scipy.sparse.linalg.eigsh` (degeneracy-robust: sort and
match by count). Then:

- **Rectangle:** assembled-`L` eigenvalues must equal the closed-form
  `Λ_{mn} = (4/h²)[sin²(mπ/2N_x) + sin²(nπ/2N_y)]` to machine precision (proves `L` is assembled
  right), and `f_disc → f_{mn}` continuum at **O(h²)**.
- **Circle:** `f_disc` from the staircased `L`, compared to Bessel `f_{mn}`, converges at **~O(h)**
  with a loose absolute bound at the finest grid (the agreed criterion).

This eigenvalue test is the 2D analogue of model #3's per-mode `g_m` money test. A complementary
end-to-end FFT test (pluck → pickup spectrum → detected peaks land on `f_disc`) validates the actual
time-stepping; single-mode runs use the eigenvectors from `eigsh` (circle) or analytic `sin·sin`
(rectangle) as initial conditions so the field stays one clean tone.

## Work breakdown (build order de-risks)

**Order:** (1) operators2d + rectangle energy green → (2) rectangle modal exact/O(h²) → (3) circle
mask + energy green → (4) circle Bessel convergence → (5) stability/NaN guards → (6) viz heatmaps.
Energy first at every geometry (HANDOFF §4.1 recipe). The web viewer is a separate later batch.

1. **`core/operators2d.py`** (new, pure NumPy/SciPy) — `rectangle_mask`, `disk_mask`,
   `laplacian_from_mask(mask, h) → (L, index_map)`, `inner2d`/`norm2_2d`. Symmetry is the
   energy guarantee; assert it in a test.
2. **`core/membrane.py`** (new) — `Membrane(domain=..., ...)` with the standard resonator interface
   (`__init__`, `set_state`, `step`, `state`, `energy`, `displacement_at`). `state` returns the 2D
   field (live nodes embedded back into the grid, zeros elsewhere) for heatmaps. CFL `λ≤1/√2`
   asserted at construction; `domain` selects rectangle vs circle via the mask.
3. **`analysis/modal.py`** — add `rectangular_membrane_freqs`, `rectangular_mode_field`,
   `circular_membrane_freqs` (Bessel via `scipy.special.jn_zeros`, returned sorted with `(m,n)`),
   `discrete_membrane_eigenfrequency(Lambda, c, k)`.
4. **`core/exciter.py`** — `raised_cosine_2d` (smooth radial bump) for broadband excitation.
5. **`tests/helpers.py`** — `make_membrane(...)` (build at a target λ via `fs = cN/(L λ)`), 2D
   single-mode init + frequency/eigenvalue measurement helpers.
6. **`tests/test_membrane_energy.py`**, **`test_membrane_modal.py`**, **`test_membrane_stability.py`**
   — the suite (below).
7. **`viz/plots.py`** + **`scripts/diagnose_membrane.py`** — mode-shape heatmaps, detected-vs-Bessel,
   energy trace, a displacement-field animation frame dump (the seed of the web viewer).

## Tests — acceptance criteria

1. **Energy conserved (lossless), both geometries:** drift `< 1e-10` over ≥ 1 s — rectangle *and*
   staircased circle (the decoupling claim, made a test).
2. **Passivity (lossy):** `σ>0` ⇒ energy monotone non-increasing; decay rate ≈ `2σ`.
3. **Rectangle modal — exact + convergent:** assembled-`L` eigenvalues match the closed-form
   `Λ_{mn}` to machine precision; continuum `f_{mn}` error shrinks at **O(h²)** under refinement.
4. **Circle modal — convergence-rate to Bessel:** `f_disc` vs `c·j_{m,n}/(2πa)` error decreases at
   the **expected staircase rate** under refinement (empirical order `p` near the staircase value),
   with a **loose absolute bound** (in cents) at the finest grid. *Not* a 1-cent single-grid bar.
5. **End-to-end FFT (sanity):** pluck a circle, detected pickup peaks land on the low `f_disc`.
6. **Stability guards:** `λ > 1/√2` rejected at construction; NaN-free over a sweep of valid λ;
   non-physical params rejected.
7. **Anisotropy demonstrated (dispersion):** axial vs diagonal phase speed differ and no λ removes
   the gap (the 2D dispersion fact, shown — assertion is *direction-dependence exists*, not a bound).
8. **Portability:** auto-covered — `test_stability.py`'s core sweep checks `membrane.py` /
   `operators2d.py` against the headless/allowlist guards with no edits.

## Traps (pre-flagged)

- **Do not carry `λ=1`** from 1D. The 2D limit is `1/√2`, and *no* λ is dispersionless. Asserting a
  λ=1 build, or a zero-dispersion λ, is the top trap.
- **Don't loosen the energy bar for the circle.** Staircasing degrades the *Bessel* match, not
  conservation — the 1e-10 drift test must hold for the circle too. If it doesn't, the mask broke
  `L`'s symmetry (check the neighbour-dropping logic), exactly the HANDOFF §4.4 boundary warning.
- **Validate the harness on the rectangle first.** A 2D-Laplacian/peak-detection bug and the
  staircase error are confounded in the circle run; the rectangle (exact modes) separates them.
- **Degeneracy in peak→mode assignment.** `m≥1` circular modes are cos/sin pairs; square rectangles
  add `(m,n)↔(n,m)`. Match eigenvalues by sorted value + count, never by a fragile per-peak label.
- **Convergence test honesty.** Report the measured order and the absolute cents at the finest grid;
  don't present the loosened circular tolerance as if it equalled the 1D standard (advisor caveat).
- **`L` is constant only while geometry/params are fixed** — any future parameter setter must rebuild
  the mask/operator.
