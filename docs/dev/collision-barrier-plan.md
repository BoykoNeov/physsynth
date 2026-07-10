# Distributed-barrier collision — string against a rigid/nonlinear barrier (model #8)

> **Status: BUILT & GREEN (2026-07-10).** `core/collision.py` (promoted contact primitives +
> `solve_contact_vector` + `BarrierString`), `mallet.py` now imports the primitives back
> (bit-identical, its 26 tests still green), `tests/test_collision_{energy,modal,signature}.py`
> (12+3+4 = 19 tests), `scripts/diagnose_collision.py`, `make_barrier_string` helper. Full suite
> **692** green, ruff clean. Gates: lossless conservation **drift ~2e-12** through genuine contact
> (peak force ~700 N, ~1000 contact-steps); **static-equilibrium magnitude oracle** holds the
> closed-form `S u*=(K/ρ)b` to **3.4e-15** with a `force_pref*=2` negative control that blows drift
> up >1e4×; single-active-node → scalar `solve_contact` collapse to **7e-16**; **drift ∝ newton_tol**
> self-cert; passivity (σ>0 / λ_h>0) monotone; out-of-reach barrier bit-identical to the bare string;
> buzz signatures (barrier brightens the tone, closer/harder barrier brighter/shorter, intermittent
> contact). The second consumer of the mallet's contact primitives, and the model that promotes them
> into `core/collision.py`. A **string vibrating against a one-sided distributed barrier** —
> string–fret buzz, the tanpura/sitar *jawari* bridge, prepared-piano rattle, a snare's snap. Model
> #7 (mallet) was a **single** lumped DOF in contact at **one** node; model #8 is the **field** in
> contact along a **profile** of nodes. The genuinely new machinery is the **vector contact solve**
> through the string's dense driving-point admittance.
>
> **Platform gotcha (2026-07-10):** the dense per-step Newton solve uses `scipy.linalg.lu_solve`, not
> `np.linalg.solve` — NumPy 2.4's threaded BLAS on this Windows box has a catastrophic cliff (0.05 ms
> at n≤90 → ~250 ms at n≥100), which hung the N=120 diagnostic. scipy stays ~1 ms across sizes. The
> full-support solve is `|𝒞|×|𝒞|` dense per step (as designed — fine offline); a future active-set
> reduction to only the in-contact nodes is the natural optimization.
>
> **Advisor-confirmed direction (load-bearing):**
> - **This is the bow pattern generalized from rank-1 to a vector.** Force-free `step()`, read
>   `u_free` at the contact nodes, vector Newton for the contact forces, apply the exact rank-`m`
>   correction `u += (k²/ρ) Σⱼ Fⱼ·aⱼ`. **Zero edits to `string_damped.py`** — `apply_Ainv` +
>   external `u +=` already suffices (the bow proved this).
> - **Force *density*, not force. `force_pref = k²/ρ` (NOT `k²/(ρh)`).** The bow's extra `h` exists
>   only because the bow injects a point force (N) at one node and must divide to a density. Here
>   `φ'(ηⱼ)` is already a density (N/m) because the potential `V = h·Σⱼ φ(ηⱼ)` makes `φ` an energy
>   *density*. So force `Fⱼ = φ'(ηⱼ)`, update `+ (k²/ρ)·A⁻¹F`, energy `h·Σ φ(ηⱼ)`. Internally
>   consistent — do not second-guess it.
> - **Vector solver = damped Newton + Armijo line-search on ‖r‖². NOT brentq.** The mallet's
>   scalar bracketed fallback does not transfer to a vector. Well-posedness is *provable*, not
>   hoped-for: `J = I + G·diag(F')`, `G` SPD (A SPD ⇒ A⁻¹ SPD), `diag(F') ⪰ 0` ⇒ `G·D` similar to
>   the PSD `D^½ G D^½` ⇒ **every eigenvalue of J is ≥ 1, everywhere**. So the root is **unique**
>   and Newton converges globally — *stronger* than the bow (which had non-unique roots needing
>   branch-picking). **Do not build branch-selection machinery.**
> - **Let the vectorized `[η]₊^α` force self-select the active set.** `F` and `F'` are exactly `0`
>   for `η < 0`, so plain Newton on the full contact-support vector gives `J = I` (⇒
>   `ηⱼ = η_free,j`) at inactive nodes automatically — the coupling lives only among active nodes.
>   **Do not hand-roll active-set bookkeeping.** Precompute `aⱼ = apply_Ainv(eⱼ)` only for nodes in
>   the barrier's spatial *support* (not all `N−1`), cache them (A is fixed). Point fret → 1 column
>   (collapses to the mallet scalar). Full-length barrier → the whole support block, O(N)
>   back-solves **once**. Fine offline; `log`/comment the cost so a future active-set-only variant
>   is possible.
> - **The quantitative *magnitude* oracle: the static-equilibrium test** (revised 2026-07-10 after
>   an advisor reconcile). Energy conservation proves only *internal consistency* — the force
>   injection and the `h·Σφ` PE telescope together, but *both could carry a compensating scale
>   factor* and still conserve. A separate oracle must pin the **absolute** coupling magnitude
>   against known `K/ρ`. The *frequency* eigenvalue test (measured partials == eig of `−L +
>   (K/ρ)·diag(mask)`) turns out **not tight**: for `α = 1` the discrete-gradient force is
>   `K(η⁺+η⁻)/2` — time-weights `(½,0,½)` vs the elastic θ-term's `(θ,1−2θ,θ)` — so the discrete
>   partials are warped away from the continuous augmented eig at finite `k` (they agree only at
>   `θ = ½` or `k → 0`). **Instead: the static equilibrium is exact.** At rest `η⁺ = η⁻`, so the DG
>   hits its Taylor branch (no averaging warp), and the scheme's discrete fixed point equals the
>   *continuous* augmented equilibrium `S u* = (K/ρ) b` (with `S = −L + (K/ρ)·diag(mask)`) to
>   machine precision (verified **3.4e-15**). Start both history levels at that closed-form `u*` with
>   zero velocity ⇒ the sim holds `u*` to ~1e-14. A wrong `force_pref`, wrong sign, or wrong `G`
>   off-diagonal all *move* the true fixed point, so the sim drifts off the analytic `u*`. **Negative
>   control:** `force_pref *= 2` blows the drift up by orders of magnitude — the gate has teeth. The
>   frequency shift stays a *diagnostic signature* (direction + rough magnitude vs augmented eig),
>   not a gate.

## Why this is genuinely a new model (not a mallet re-run)

The mallet's host resonator (the `Membrane`) is **explicit**: a node force touches only that node
next step, so the driving-point admittance is the bare local nodal mass and the contact solve is a
single **scalar** equation (`η = η_free − g·f(η)`). The barrier's host is the **implicit**
θ-scheme `DampedStiffString`: `A u^{n+1} = rhs`, so a force at contact node `j` propagates through
`A⁻¹` to **every** node. With contact along a set `𝒞` of nodes the coupled unknown is the whole
penetration **vector** `η_𝒞`:

```
u^{n+1}   = u_free^{n+1} + (k²/ρ) Σ_{j∈𝒞} Fⱼ · aⱼ ,      aⱼ = A⁻¹ eⱼ         (dense columns)
η_i^{n+1} = b_i − u_i^{n+1}                                                   (i ∈ 𝒞)
⇒  η_𝒞 = η_free,𝒞 − G · F(η_𝒞) ,   G_ij = (k²/ρ)(A⁻¹)_ij  (the string admittance block on 𝒞)
```

`G` is a dense `|𝒞|×|𝒞|` symmetric-PD matrix (the driving-point admittance restricted to the
barrier support). `F(η)ⱼ = ` per-node `[DG]` force density. This is a **vector Newton solve** — the
new algebraic object, the analog of the mallet's scalar `solve_contact` promoted to a vector.

## Physics — string against a one-sided distributed barrier

String displacement `u(x,t)`, a fixed barrier profile `b(x)` (the fret crown, the curved jawari
bridge, the flat rail). Convention (mirror the mallet, `η > 0` ⇔ in contact / interpenetrating):

```
η(x,t) = b(x) − u(x,t)          (> 0  ⟺  the string has pushed into the barrier)
```

The barrier is below the string when `b < u_rest`; contact happens where the vibrating string
crosses `b`. Per-node one-sided nonlinear spring, **contact potential density** and force density

```
φ(η) = (K/(α+1)) [η]₊^(α+1) ,   φ'(η) = K [η]₊^α ,   [η]₊ = max(η,0) ,   K > 0, α ≥ 1
```

Equation of motion (barrier reaction pushes the string *away* from the barrier, i.e. toward
decreasing η):

```
ρ u_tt = c²ρ u_xx − κ²ρ u_xxxx − 2ρσ₀ u_t + 2ρσ₁ u_txx + φ'(η)      (φ'≥0 pushes u toward b→ contact repels)
```

Sign check: `η = b − u`, so `∂η/∂u = −1`; the potential energy density is `φ(η)`, and the
generalized force on `u` is `−∂φ/∂u = +φ'(η)·(−∂η/∂u)·(−1)`… → the force **opposes penetration**
(pushes the string out of the barrier). **Eyeball the sign in code** — wrong sign grows the energy
and the lossless-drift test catches it, but it is the classic spot.

Continuous energy and balance:

```
E(t) = E_string + ∫ φ(η) dx
dE/dt = −(string loss ≥ 0) − (barrier hysteresis ≥ 0) ≤ 0 ,   = 0 exactly when σ = 0, λ_h = 0.
```

## Numerical scheme — vector discrete gradient through the implicit admittance

**String:** unchanged; one force-free `DampedStiffString.step()` per step, then a rank-`|𝒞|`
correction. **Time-centering:** the `[DG]` uses `η^{n+1}` and `η^{n-1}` only — `u^n` does **not**
appear in `F` (this is what telescopes to the money-test PE `h·Σⱼ ½(φ(ηⱼ^n)+φ(ηⱼ^{n-1}))`; confirm
`u^n` absent from the residual, same `(eta_next, eta_prev) = (n+1, n-1)` as the mallet).

**Vector `[DG]` = component-wise scalar `[DG]`** because the potential `V = h·Σⱼ φ(ηⱼ)` is
**separable**: `∂V/∂ηⱼ` depends on `ηⱼ` only, so each node's force is the same scalar discrete
gradient the mallet uses, applied independently. `Fⱼ = contact_force_dg(ηⱼ^{n+1}, ηⱼ^{n-1}, K, α,
tol)` (+ optional per-node Hunt–Crossley). The **coupling** is entirely in `G` (through `A⁻¹`), not
in `F`.

**The vector residual and Jacobian:**

```
r(η) = η − η_free + G · F(η) = 0            (η, η_free, F ∈ ℝ^{|𝒞|};  G ∈ ℝ^{|𝒞|×|𝒞|})
J(η) = I + G · diag(F'ⱼ) ,   F'ⱼ = ∂Fⱼ/∂ηⱼ = _contact_force_total_deriv(...)  ≥ 0
```

**Solver — damped Newton with Armijo backtracking on `½‖r‖²`:**
1. seed `η = η_free` (or continuation from the previous step's penetration on 𝒞);
2. `δ = −J⁻¹ r` (dense `|𝒞|×|𝒞|` solve — small; `|𝒞|` ≤ barrier support);
3. backtrack `t ∈ {1, ½, ¼, …}` until `½‖r(η+tδ)‖² < (1 − c·t)·½‖r(η)‖²` (Armijo, `c≈1e-4`);
4. accept `η ← η + tδ`; stop at `‖r‖∞ ≤ newton_tol`.
Because `λ_min(J) ≥ 1` everywhere the full Newton step is always a descent direction and the root
is unique — the line-search only guards the semismooth kink at `η = 0` (the `[η]₊` breakaway; for
`α > 1`, `F ∈ C¹`; for `α = 1`, `F` is piecewise-linear). An extra iteration near contact-onset /
breakaway is expected, not a bug. **No brentq, no branch-picking.**

**Apply the force exactly** (like the bow/mallet): `u += (k²/ρ) Σⱼ Fⱼ aⱼ`, so the reported energy
is machine-precision regardless of the Newton residual.

**Energy identity (why it telescopes).** Per node the elastic `[DG]` gives `Fⱼ·δ_t·ηⱼ = δ_t·φ(ηⱼ)`.
Summed over 𝒞 with the `h` weight and combined with the string's own SBP telescoping, the contact
work cancels the stored barrier PE exactly, leaving `−(losses) ≤ 0`. So
`E_string^n + h·Σⱼ ½(φ(ηⱼ^n)+φ(ηⱼ^{n-1}))` is **constant** (`σ = λ_h = 0`) or **monotone
decreasing** (`σ > 0` or `λ_h > 0`).

## Files

- `core/collision.py` — **NEW.** Home of the promoted contact primitives (moved verbatim from
  `mallet.py`, which now imports them back — mallet public behavior stays **bit-identical**, the
  673 existing tests must still pass): `contact_potential`, `contact_force_elastic`,
  `contact_stiffness`, `contact_force_dg`, `_contact_force_dg_deriv`, the Hunt–Crossley pair,
  `contact_force_total` / `_contact_force_total_deriv`, and the scalar `solve_contact`. **NEW**
  alongside them: `solve_contact_vector(eta_free, eta_prev, G, K, α, λ_h, k, *, tol, seed,
  newton_tol, maxiter)` (damped Newton + Armijo) and the `BarrierString` resonator
  (`step`/`energy`/`state`/`k`/`displacement_at`), holding a `DampedStiffString` + a barrier
  profile `b(x)` + the cached admittance columns `aⱼ` on the support 𝒞.
- `physsynth/core/mallet.py` — **EDIT:** delete the moved free-function bodies; `from .collision
  import (contact_potential, contact_force_elastic, contact_stiffness, contact_force_dg,
  contact_force_total, solve_contact)` (keep them re-exported in `__all__` so any importer of
  `mallet.contact_*` still resolves). No behavioral change.
- `tests/test_collision_energy.py` — coupled money test: lossless string + elastic barrier ⇒
  `E_string + h·Σ½(φ^n+φ^{n-1})` flat to `~1e-12`, at an amplitude where a meaningful fraction of
  the string touches the barrier; `K=0` (or barrier out of reach) ⇒ **bit-identical** to the bare
  `DampedStiffString`; passivity with `σ > 0` and/or `λ_h > 0`; **drift ∝ newton_tol**
  self-certification.
- `tests/test_collision_modal.py` — **the quantitative magnitude oracle + degenerate path:** `α=1`,
  full-interior linear-spring bed (barrier a hair above rest so `η > 0` always). Start both history
  levels at the closed-form continuous equilibrium `u* = S⁻¹(K/ρ)b`, `S = −L + (K/ρ)I` ⇒ the sim
  holds `u*` to ~1e-14 (**exact** magnitude anchor); **negative control** `force_pref *= 2` ⇒ drift
  blows up orders of magnitude. Plus **single-active-node → scalar collapse**: one genuinely
  contacting node ⇒ the vector `η` equals the imported scalar `solve_contact` with `g = G_jj =
  (k²/ρ)(A⁻¹)_jj` to ~newton_tol (verified **7e-16**; the two solvers differ — Newton+Armijo vs
  Newton+brentq — so this checks same-root convergence, not shared code).
- `tests/test_collision_signature.py` — physical signatures: barrier contact **adds high partials**
  (buzz) vs the free string; a raised barrier **shifts pitch up** (shortened effective length /
  stiffening); harder/closer barrier → brighter/longer buzz; tanpura-style curved barrier sustains
  a richer spectrum. Diagnostic-only (like Schelleng), not gates.
- `scripts/diagnose_collision.py` — string-vs-barrier animation frames, contact-force map over
  (x,t), energy partition (string ⇄ barrier PE, flat total elastic / decaying hysteretic),
  spectrum with/without barrier (the buzz), pitch-vs-barrier-height sweep.

## Validation ladder (mirrors the mallet's two-tier)

1. **Degenerate scalar collapse — reuse the proven path.** One contacting node ⇒ the vector solve
   *is* the mallet's scalar `solve_contact` with `G_jj = (k²/ρ)(A⁻¹)_jj`. Cross-check numerically.
   De-risks the vector machinery against already-green code before any distributed physics.
2. **Quantitative magnitude oracle — the static-equilibrium test.** `α=1`, full-interior spring bed,
   started at the closed-form continuous equilibrium `u* = S⁻¹(K/ρ)b` ⇒ the sim holds `u*` to
   ~1e-14; `force_pref *= 2` blows it up (negative control). Catches a wrong coupling *magnitude*
   that energy conservation (internal consistency only) would miss. (The frequency-eigenvalue variant
   is warped by the `α=1` DG time-averaging → diagnostic-only, not a gate.)
3. **Coupled energy conservation of the whole.** `E_string + h·Σ½(φ^n+φ^{n-1})` flat to `~1e-12`
   (lossless, elastic); bit-identical miss check; drift ∝ newton_tol.
4. **Passivity.** `σ > 0` and/or `λ_h > 0` ⇒ `E` monotone decreasing.
5. **Signatures** (diagnostic): buzz partials, pitch-vs-barrier-height, tanpura sustain.

## Resolved / open decisions

- **Host resonator = the string** (`DampedStiffString`), the implicit θ-scheme — that is the whole
  point (it forces the vector solve). Membrane-vs-barrier (2D snare) is a later config swap reusing
  the same vector collapse with the membrane's local admittance (would collapse the dense G to
  diagonal). Default taken.
- **Primitives promoted to `core/collision.py`**, mallet imports back (the mallet plan pre-committed
  this "when the second consumer lands"). Now it lands. Default taken.
- **Vector solver = damped Newton + Armijo**, no brentq, no branch-picking (unique root, λ_min(J)≥1).
  Advisor-confirmed.
- **Active set self-selects** via `[η]₊^α`; precompute admittance columns on the barrier support
  only, cached. Advisor-confirmed.
- **Hunt–Crossley hysteresis:** reuse the per-node primitive (already vector-ready). Include it this
  batch for the passivity gate, as the mallet did. Default taken.
- **Barrier profile `b(x)`:** support flat (fret rail), point (single fret), and curved (jawari)
  profiles as a callable/array. Snap to grid nodes (bow/mallet precedent); sub-grid interpolation is
  a shared later refinement. Default taken.

## Non-goals for this batch

Two-sided barriers, moving/deformable barriers (finger stopping), 2D snare-on-membrane, sub-grid
contact interpolation, sympathetic-string coupling through a shared barrier. All reuse the promoted
`φ`/`[DG]` primitives and the vector solve.
