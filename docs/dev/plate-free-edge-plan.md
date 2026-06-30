# Free-edge plate (Chladni) — Phase 3 Plan (model #5b, free rectangular plate)

> **Status: Part 0 DONE (2026-06-23), Part 1 DONE (2026-06-30).** Follow-on to the simply-supported
> plate
> (`docs/dev/plate-plan.md`, model #5). This is the *visual showpiece* half of HANDOFF §5 row 5 — the
> iconic curved Chladni nodal figures, which require **free** edges. Built **energy-first** per the Bilbao framework (CLAUDE.md #2);
> validated against the free-free **closed-form 1D beam** oracle, the **rigid-body nullspace**, **O(h²)
> self-convergence**, and **Leissa's tabulated free-square-plate frequency parameters**.
>
> **Human decisions taken (2026-06-23):**
> - **Build a 1D free-free Euler–Bernoulli beam FIRST** as a de-risk sub-step (model #5b-pre). It has a
>   *real closed-form oracle* and isolates the free-end moment/shear stencil + operator symmetry
>   **without** the 2D corner conditions or ν. Squarely in the closed-form-validation culture.
> - **ν re-enters; default ν = 0.3** (matches Leissa's tables). Core input surface = **`(kappa, nu)`**
>   (one stiffness number + Poisson's ratio); `(E, e, ν, ρ_s)` macro deferred as before.
> - **Render the Chladni figures in THIS batch.** They are the headline deliverable, not viz polish.
>   They come straight from the `eigsh` eigenvectors of the stiffness operator, so they are
>   *independent of the time-stepper* and double as validation. (Validate-core-first still holds — the
>   operator + nullspace + eigenfrequency checks land before the figures.)
>
> **Advisor-confirmed approach (the load-bearing decision):** build the **operator from the energy**,
> not the energy from the operator. Define the discrete strain-energy bilinear form first; let the
> stiffness matrix `K` be the matrix that represents it. This gives — *by construction* — symmetry
> (→ energy conservation), the correct **natural** free boundary conditions (zero bending moment, zero
> Kirchhoff/effective shear, corner force), and the right rigid-body nullspace, **none hand-imposed**.
> The alternative (ghost-point elimination on the 13-point biharmonic stencil) breaks symmetry unless
> done very carefully — **avoid it**.

## Why `B = L²` does NOT carry over (the whole reason this is a new model)

The SS plate got everything for free from `B = L @ L` with the *Dirichlet* Laplacian: the zero ghost
baked in `u = 0` **and** `∇²u = 0`, kept `sin·sin` an exact eigenvector, and gave machine-precise
eigenvalues. **None of that survives free edges:**

- Free edges are **`u ≠ 0`** on the rim — every boundary node is now an unknown. There is no Dirichlet
  rim and no `L` to square.
- The free natural BCs are **zero bending moment** (`M_n = 0`) and **zero effective (Kirchhoff)
  shear** (`V_n = 0`), plus a **corner condition** (`u_xy = 0` at free corners). These involve ν.
- **ν re-enters.** For SS the twisting (Gaussian-curvature) term integrated to a vanishing boundary
  term; for free edges it is alive in both the operator and the energy.
- **No closed-form modal oracle.** The tight ~1-cent bar is gone. Validation shifts to: closed-form
  *nullspace*, O(h²) *convergence order*, and *absolute* anchoring to Leissa's tabulated values.

## Goal

The first **free-boundary** flexural resonator — the plate whose free edges produce the curved Chladni
nodal figures. The deliverable is, as always, **the resonator + the rig that measures its deviation
from theory**. Here the theory is supplied by three independent oracles (no single closed form):
the rigid-body nullspace (machine-precise), the convergence *order* (O(h²)), and Leissa's
dimensionless frequency parameters (percent-level absolute anchor).

---

## Part 0 — 1D free-free beam de-risk (model #5b-pre) ✅ DONE

> **Built & green (2026-06-23).** `core/beam.py` (`FreeBeam`), `operators.free_beam_stiffness`,
> `analysis/modal.{free_free_beam_betaL,free_free_beam_freqs,discrete_beam_eigenfrequency}`,
> `tests/test_beam_{modal,energy,stability}.py` — 263 tests total, lint-clean. Measured results:
> `‖K−Kᵀ‖=0`; nullspace `K@1=0` / `K@x` rel~1e-18 / `K@x²` rel~1e-6; modal 0.18–1.45 cents at
> N=200, mu=0.5; convergence order → 2.00; energy drift 2–5e-12. **Key construction insight:** the
> free-end closure is supplied by the **half-cell end masses (`W` diag `h/2` at the ends)**, NOT by
> hand-coded stiffness rows — `W⁻¹K` reproduces Bilbao's free-free bar exactly. Expected to recur in
> 2D, where the edge-½ / corner-¼ `W` weights supply the free closure.

A cheap, fully closed-form rehearsal of the free-end stencil + energy-first symmetry, with **no
corners and no ν**. If this is green, the 2D build inherits a trusted free-edge construction and only
the genuinely-2D risks (corners, the (1−ν) term) remain.

### Physics

Euler–Bernoulli beam, bending only (transverse displacement `u(x, t)` on `[0, L]`):

```
u_tt = -kappa² u_xxxx ,     kappa² = EI / (rho A)   (same kappa convention as the stiff string)
```

**Free-free natural BCs** at both ends: zero moment `u_xx = 0` and zero shear `u_xxx = 0`. These are
*natural* — they fall out of the energy variation, so the energy-first construction enforces them
without hand-coded boundary rows.

Strain (bending) energy and the conserved total:

```
U(t) = (rho A · kappa² / 2) ∫ (u_xx)² dx ,    E = (rho A/2)∫(u_t)²  +  U ,    dE/dt = -2σ·(…) ≤ 0
```

### Modal oracle (the reason we do the beam first)

Free-free Euler–Bernoulli has a **closed-form** spectrum:

```
ω_n = kappa · β_n² ,    f_n = kappa · (β_n L)² / (2π L²)
```

where `β_n L` are the roots of the **frequency equation `cos(βL)·cosh(βL) = 1`**:

```
β₁L ≈ 4.730041, β₂L ≈ 7.853205, β₃L ≈ 10.995608, β₄L ≈ 14.137165, …  (→ (2n+1)π/2 for large n)
```

Plus **two rigid-body zero modes** (`ω = 0`): translation `u ≡ 1` and tilt `u ≡ x`. These are the 1D
analog of the 2D `{1, x, y}` nullspace and are the operator money test here too.

### Energy-first construction (1D)

Let `D2` be the **interior** second-difference operator mapping the `N+1` nodal values to the `N−1`
interior curvatures `(u_{l+1} − 2u_l + u_{l−1})/h²` (an `(N−1)×(N+1)` matrix — note it acts on the
*full* node set, end values included, since the ends are free unknowns). Discretize the bending energy
as a quadratic form and read off the symmetric stiffness:

```
∫(u_xx)² dx  ≈  uᵀ (D2ᵀ Wc D2) u  ≡  uᵀ K u  ,     K = D2ᵀ Wc D2   (symmetric PSD by construction)
```

with `Wc` the curvature-quadrature weights. **`K`'s nullspace is exactly `{1, x}`** (any linear `u`
has zero second difference), which *is* the free-free rigid-body space — the natural BCs come for free.
The mass/quadrature weight `W` (diagonal: interior `h`, ends `h/2`) sits on the LHS:

```
W u_tt = -kappa² K u  →  (theta-scheme)  A = (1+σk) W + θk²κ² K ,   A SPD → factor once (splu)
```

**Boundary accuracy is THE crux of Part 0.** The bare `K = D2ᵀ(h·I)D2` gives the correct *interior*
operator, but the free-end stencil it implies is only a low-order approximation of `u_xxx = 0`. Assert
the convergence **order explicitly** (Richardson-extrapolate the low eigenvalues) — and do **not** be
alarmed if the boundary-adjacent *pointwise* error lags the interior; the low **eigenvalues** can still
come in at O(h²) by symmetry, and that is the signal you tune `Wc`/the end-weights against. This is the
entire point of doing the beam first. Reference: Bilbao *Numerical Sound Synthesis*, the bar chapter
(free-free bar with the energy method).

### Beam acceptance criteria

1. **Operator symmetry:** `‖K − Kᵀ‖ = 0` (machine precision) — test *before* any time-stepping.
2. **Nullspace money test:** `K @ 1 = 0` and `K @ x = 0` to machine precision; `K @ x² ≠ 0` (a
   parabola has real curvature energy) as the "not-everything-is-killed" counter-check.
3. **Energy drift < 1e-10** (lossless) over ≥ 1 s; **passivity** (σ>0) monotone non-increasing.
4. **Modal frequencies vs the `cosβL·coshβL = 1` oracle:** generalized eigenvalues `K φ = (ω²/κ²) W φ`
   (skip the 2 zero modes), matched by sorted eigenvalue, within a **tight** bar at fine N (this *is*
   closed-form), and the continuum error shrinking at **O(h²)** under refinement.
5. **Portability:** auto-covered by the `core/` headless/allowlist sweep.

---

## Part 1 — 2D free-edge Kirchhoff plate (model #5b)

> **Built & green (2026-06-30).** `operators2d.free_plate_stiffness(Nx, Ny, h, nu) → (K, W,
> index_map)`, `Plate(boundary="free", nu=…)` (W-weighted θ-scheme, splu),
> `analysis/modal.{free_plate_ffff_square_lambdas, free_plate_freq_from_lambda}`,
> `viz/plots.plot_chladni`, `tests/test_free_plate_{modal,energy}.py` (+ flipped
> `test_plate_stability`), `scripts/diagnose_free_plate.py`. **The advisor's collapse held: the only
> genuinely new code was the ν-coupling.** Bending-diagonal = the validated free *beam* operator per
> axis (`K = C2xᵀWaC2x + C2yᵀWaC2y + ν·cross + 2(1-ν)·h²·DxyᵀDxy`, `Wa = kron(m_y,m_x)` →
> edge-½/corner-¼ for free); twist = **cell-centered** `kron(d1y,d1x)` (no checkerboard mode). Money
> tests: `‖K−Kᵀ‖=0`; `K{1,x,y}=0` ~1e-18 with `K(xy)≠0` scaling **exactly** ∝ (1-ν); **exactly 3**
> near-zero generalized eigenvalues; kron build == an independent explicit-loop assembly. Validation:
> O(h²) self-convergence (order ≈ 2.1–2.3); **Leissa FFFF-square anchor matched to 0.01 %** at N=80
> (λ = 13.467/19.598/24.269/34.803/34.803 vs Narita-table 13.468/19.596/24.270/34.801/34.801);
> energy drift 1e-13 even at mu = 16; fundamental = the **saddle/twist** (not a bulge). Chladni
> figures rendered (cross, X, ring, S-curves, stripes). **NB:** the Narita (2022) values *improve on*
> Leissa's classic monograph; cited in `modal.free_plate_ffff_square_lambdas`. The portability
> allowlist test (`test_stability.py`) was also fixed to measure the *delta* of core-pulled modules,
> not the absolute set (excludes the Windows pywin32 startup baggage that newly leaked in).

### Physics

Same PDE as model #5 — **bending only** — but **free** edges and **ν alive**:

```
u_tt = -kappa² ∇⁴u - 2σ u_t ,    kappa² = D/rho_s ,    D = E e³/(12(1−ν²))
```

The Kirchhoff **strain energy** keeps the twisting term that vanished for SS:

```
U = (rho_s·kappa²/2) ∫∫ [ (∇²u)²  −  2(1−ν)(u_xx·u_yy − u_xy²) ] dA
```

- The `(∇²u)²` term is the SS part. The **`(1−ν)(u_xx u_yy − u_xy²)`** (Gaussian-curvature) term is
  what re-enters; it is the part that makes the **saddle `xy` carry energy** and is the tell for a
  correct ν.
- Continuum total energy and dissipation:
  `E = (rho_s/2)∫∫(u_t)² + U`, `dE/dt = -2σ ρ_s∫∫(u_t)² ≤ 0` (free-edge boundary flux cancels —
  the natural BCs are exactly the ones that kill it).

### Energy-first operator (the build)

Following the advisor and Bilbao NSS **Ch. 12** (free plate, energy method): assemble the **symmetric
stiffness matrix `K`** as the discrete representation of the bilinear form

```
P(f, g) = ∫∫ [ (∇²f)(∇²g)  −  (1−ν)( f_xx g_yy + f_yy g_xx − 2 f_xy g_xy ) ] dA
```

so `U = (rho_s κ²/2)·fᵀ K f`. Build it from the component second-difference operators
(`D_xx`, `D_yy`, `D_xy`) as Gram-style products `Dᵀ (weights) D`, which is **symmetric by
construction** and yields the **natural** free BCs (`M_n = 0`, `V_n = 0`, corner force) automatically —
no hand-coded 13-point boundary/corner rows, no ghost elimination. Use Bilbao's stencils for the edge
and corner weighting rather than re-deriving the corner terms.

A **diagonal weight matrix `W`** (the lumped quadrature / mass weights: **interior `h²`, edge `h²/2`,
corner `h²/4`**) replaces the SS model's scalar `h²` and sits on the LHS. The θ-scheme becomes a
*principled generalization* of model #5, not a rewrite:

```
W δ_tt u = -kappa² K (θ u^{n+1} + (1−2θ) u^n + θ u^{n−1}) - 2σ W δ_t. u

A = (1+σk) W + θ k² κ² K        (SPD → factor once with scipy.sparse.linalg.splu, back-sub each step)
```

> **SS is the special case `W = h²I`, `K = h²B`** — note the **`h²` is baked into `K`** (because
> `∫∫(∇²u)² dA ≈ h²·uᵀB u`), so `W⁻¹K = B`. For the rectangle the global `h²` cancels through the
> solve, so the *solution* matches model #5, but the matrices are **not** identical: the existing SS
> path uses `A_SS = (1+σk)I + θk²κ²B` (no `h²` on the LHS — it factors out), whereas the unified scheme
> gives `A = h²·A_SS`. **Pin this convention before assembling `K`** — for the *free* plate the edge-½
> / corner-¼ weights do **not** factor out, which is the entire reason `W` exists. Keep the validated
> SS branch untouched and add a parallel `free` branch; the regression test compares **frequencies and
> energy**, not the A-matrices (see test #8). Symptom of a `K=B` (missing-`h²`) slip: stiffness and
> mass scale apart by exactly `h²`.

Energy (θ-dependent), the W/K generalization of model #5's form:

```
E^n = rho_s [ ½ (δ_t⁻u)ᵀ W (δ_t⁻u)  +  κ²·( (θ/2)(⟨Ku,u⟩_n + ⟨Ku,u⟩_p) + (½−θ)⟨Ku,u⟩_np ) ]
```

evaluated through the **same** `K`, `W` as the update → `E^{n+1}=E^n` exact (lossless), monotone (lossy).

### Operator money test — the rigid-body nullspace (replaces `B=L²==Λ²`)

A completely free plate has **exactly 3 zero-energy modes**: `w ∈ {1, x, y}` (vertical translation +
two tilts). With no closed-form modal oracle, this is the machine-precision correctness anchor:

1. **`K @ 1 = 0`, `K @ x = 0`, `K @ y = 0`** to machine precision (rigid-body nullspace) — and `W`
   positive diagonal, `‖K − Kᵀ‖ = 0`.
2. **`K @ (x·y) ≠ 0`** — the saddle/twist. `∇⁴(xy) = 0`, so a wrong operator that forgot the
   `(1−ν)` term would *spuriously kill `xy`*. A nonzero, correct `xy` energy is **the single best
   catch for a dropped-ν / wrong-free-BC bug.** (Cross-check: scale with `(1−ν)`.)

### Validation oracles (no closed form → three independent anchors)

- **O(h²) self-convergence (tight, no table needed).** Refine `h → h/2`; Richardson-extrapolate the
  low generalized eigenvalues `K φ = (ω²/κ²) W φ`. The *order* converging to ≈ 2 is a tight test that
  needs no external data — the strongest non-energy correctness check here.
- **Leissa absolute anchor (percent-level).** Lowest ~6 **elastic** modes of the **completely free
  square plate** (FFFF, ν = 0.3) via `eigsh` on the generalized problem, matched by **sorted
  eigenvalue** (a square plate has heavy degeneracy — *never* per-peak label), converging toward the
  table as N grows. Conversion is clean in our parameters: Leissa's `λ = ω a² √(ρ_s/D) = ω a² / κ`
  (since `κ = √(D/ρ_s)`), so **`f = λ · κ / (2π a²)`** — no ρ/thickness factor to fumble.
  - **Pull the exact `λ` values from Leissa (1969) at build time** and pin them; the lowest elastic
    mode is ≈ 13.5 (the twist/saddle), with the well-known degenerate pairs above it. Do **not** hard
    code remembered digits — cite the source row.
  - **Sanity sign:** the *fundamental* free-plate mode is the **saddle/twist** (diagonal nodal lines),
    **not** a drum-like central bulge. If the lowest elastic mode looks like a membrane fundamental,
    the operator is wrong.
  - The 3 rigid-body modes are **skipped** in the comparison. `K` is PSD, so shift-invert at
    `sigma=0` is **singular** — use a small **negative** shift (`eigsh(K, M=W, sigma=-ε, which="LM")`
    so `K − σW` is SPD), find the lowest eigenvalues, discard the 3 ≈ 0. `A` stays SPD for the
    time-step even though `K` is only PSD.
  - **The anchor is FFFF-*square*-specific.** Keep it square (accept the degeneracy, match by sorted
    eigenvalue). A rectangle is only a *separate* degeneracy-splitting sanity run with its own
    (different) table — never anchored against the square λ.
- **Energy drift < 1e-10 / passivity** — needs only `K` symmetry, so it is the *first* gate.

### Chladni figures (this batch, from the eigenvectors)

The iconic curved nodal patterns are the **nodal lines (zero contours) of the free-plate
eigenvectors** — rendered straight from the `eigsh` modes, independent of the time-stepper. They
double as a qualitative oracle (the low FFFF-square patterns are textbook: diagonal cross, ring,
grid). Reuse the existing 2D field/heatmap plotters (`viz/plots.py`) + a nodal-line (sign / |u|≈0)
overlay; write to `out/`. **Honest labeling:** these *are* the curved Chladni figures (unlike the SS
plate's plain rectangular grid).

---

## Work breakdown (build order de-risks; each gate green before the next)

**Beam first, then plate; operator + nullspace before time-stepping; figures last.**

1. **Beam (`core/beam.py` + oracle in `analysis/modal.py`)** — energy-first `K = D2ᵀWcD2`, `W`,
   θ-scheme; `free_free_beam_freqs` oracle (roots of `cosβL·coshβL=1`). Tests:
   `tests/test_beam_{energy,modal,stability}.py`. **Gate: closed-form βL match + O(h²) + nullspace +
   energy drift.**
2. **2D operator (`core/operators2d.py`)** — `free_plate_stiffness(Nx, Ny, h, nu) → (K, W, index_map)`
   (all nodes live; build `D_xx,D_yy,D_xy`, assemble symmetric `K` + diagonal `W`; document the
   energy-first construction and the natural BCs). **Gate: `K=Kᵀ`, `K{1,x,y}=0`, `K(xy)≠0`,
   eigenvalues O(h²).**
3. **Plate resonator (`core/plate.py`)** — add `boundary="free"`: build `K, W`, SPD
   `A = (1+σk)W + θk²κ²K`, factor once (`splu`); W-weighted `step()`, energy, `set_state`. New input
   `nu` (default 0.3). **Keep the `"supported"` branch byte-identical** (regression). No CFL limit.
4. **Oracles (`analysis/modal.py`)** — `free_plate_leissa_lambdas` (tabulated, cited), the
   `λ→f` conversion `f = λκ/(2πa²)`, and a generalized-eigensolve helper (or in tests).
5. **Tests** — `tests/test_plate_*` gain a `free` config: energy/passivity, nullspace money test,
   O(h²) self-convergence, Leissa sorted-eigenvalue anchor, SS-regression equivalence, stability/NaN,
   param rejection (ν range, etc.). Reuse `tests/helpers.py` builders.
6. **Viz (`viz/plots.py` + `scripts/diagnose_free_plate.py`)** — Chladni nodal-line figures from the
   eigenvectors, detected-vs-Leissa table, energy trace, displacement animation. Writes to `out/`.

---

## Acceptance criteria (summary)

| # | Test | Bar |
|---|------|-----|
| 1 | `K` symmetric (`‖K−Kᵀ‖`) | machine precision — **first gate** |
| 2 | Rigid-body nullspace `K{1,x,y}=0`; `K(xy)≠0` | machine precision / nonzero — **operator money test** |
| 3 | Energy drift (lossless) | `< 1e-10` over ≥ 1 s, κ & (N,fs) sweeps |
| 4 | Passivity (σ>0) | monotone non-increasing; `2σ` low modes only (broad damping caveat carries over) |
| 5 | Self-convergence | low eigenvalues at **O(h²)** under refinement |
| 6 | Leissa anchor (FFFF square, ν=0.3) | lowest ~6 elastic modes, **sorted-eigenvalue** match, percent-level, → table as N grows |
| 7 | Beam closed-form (Part 0) | `cosβL·coshβL=1` roots, tight at fine N, O(h²) |
| 8 | SS regression | free machinery on a rectangle reproduces model #5's **frequencies + energy** (compare results, *not* the A-matrix; SS is W=h²I, K=h²B) |
| 9 | Stability / params | NaN-free κ,(N,fs) sweeps; reject ν∉(−1,½), θ∉(0,1], non-physical params |
| 10 | Portability | auto-covered by `core/` headless/allowlist/no-sibling sweep |

---

## Open decisions (all resolved at plan time — recorded, not re-litigated)

- **De-risk order: 1D free-free beam first.** ✅ (human, 2026-06-23) — closed-form oracle, isolates the
  free-end stencil + symmetry before corners/ν.
- **ν: re-enters; default 0.3; surface `(kappa, nu)`.** ✅ — matches Leissa; `(E,e,ν,ρ_s)` macro later.
- **Chladni figures in this batch.** ✅ — the headline deliverable; rendered from `eigsh` eigenvectors.
- **Construction: energy-first (`K = Dᵀ W D`), Bilbao stencils; NOT ghost-point elimination.** ✅
  (advisor) — symmetry, natural BCs, and nullspace by construction.
- **Solver `splu`/`factorized`** (13-point + edge/corner rows; no sparse Cholesky in scipy). ✅

## Traps (pre-flagged)

- **`B = L²` is gone.** Free edges have `u ≠ 0` on the rim; there is no Dirichlet `L` to square and no
  exact `sin·sin` eigenvector. Build `K` from the energy.
- **Build the operator FROM the energy.** The reverse (hand-coded 13-point + ghost elimination) breaks
  symmetry → energy drift → a debugging swamp. Symmetric Gram form first.
- **The `(1−ν)` twisting term is alive.** Dropping it silently *passes* the `∇⁴` interior check but
  **kills `xy` energy** and corrupts the free BCs. The `K(xy)≠0` test is the catch.
- **Diagonal weight `W` (interior h², edge h²/2, corner h²/4)** replaces the scalar `h²` on the LHS
  *and* in the energy. Forgetting the edge/corner halving is a subtle O(h) accuracy leak.
- **No tight cents bar.** No closed-form modal oracle for the 2D free plate. Anchor with nullspace
  (machine), convergence *order* (O(h²)), Leissa (percent). Do not invent a 1-cent claim.
- **Match modes by sorted eigenvalue, never per-peak label.** The free *square* plate is heavily
  degenerate; per-peak labeling is fragile. (Use a rectangle to split degeneracies if needed.)
- **Skip the 3 rigid-body modes** when comparing to Leissa. `K` is PSD ⇒ shift-invert at `sigma=0` is
  singular; use a small **negative** shift (`K − σW` SPD) and discard the 3 ≈ 0 eigenvalues. `A` stays
  SPD for time-stepping even though `K` is PSD.
- **`a`-convention in Leissa's `λ`.** `a` is the **side length** of the square (analog of the
  model-#5 snapped-`Ly` gotcha). Use the *snapped* side length and confirm the convention against the
  cited table, or a ~0.5% length error reads as a spurious frequency error.
- **Fundamental = saddle/twist, not a bulge.** A drum-like lowest elastic mode means a wrong operator.
- **Damping caveat carries over (broad).** `Q ~ κ²Λ²` 4th-power across the spectrum → mid-spectrum
  under-damps; assert `2σ` for low modes only. Frequency-dependent loss is a later model.
- **`K`, `W` (hence the LU factor) are constant only while geometry/params/ν are fixed** — any future
  setter must rebuild and re-factor.
- **Energy-drift triangulation:** wrong drift ⇒ the **energy measure / `K` symmetry** is wrong; wrong
  frequency ⇒ the **operator assembly / ν / weights** are wrong. Never relax the tolerance (HANDOFF §8).

## References

- Bilbao, *Numerical Sound Synthesis* (2009): the **bar chapter** (free-free bar, energy method) for
  Part 0; **Ch. 12** (thin plate, free edges, energy-based difference scheme) for Part 1.
- Leissa, *Vibration of Plates* (NASA SP-160, 1969): tabulated dimensionless frequency parameters
  `λ = ω a² √(ρ_s/D)` for the completely free (FFFF) rectangular/square plate — the absolute anchor.
- `docs/dev/plate-plan.md` (model #5, SS plate) — the machinery this generalizes.
