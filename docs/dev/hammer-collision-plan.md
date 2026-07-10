# Mallet–membrane collision — Contact & Collisions Plan (model #7, first contact model)

> **Status: BUILT & GREEN (2026-07-10).** `core/mallet.py` (`MalletMembrane` + standalone `MalletWall`
> + vector-ready contact primitives), `tests/test_mallet_{wall,energy,signature}.py` (9+11+5 = 25
> tests), `scripts/diagnose_mallet.py`. Full suite **673** green, ruff clean. Gates: standalone wall
> contact-time `π√(M/K)` + exact velocity reversal (restitution 1) at α=1; coupled conservation
> **drift ~1e-12** with the head taking ~65 % of the strike energy at peak; **drift ∝ newton_tol**
> self-cert; passivity (σ>0 / λ_h>0) monotone; miss = bit-identical to the bare membrane; circle
> drumhead conserves too; signatures (broadband strike, harder→shorter+brighter, single bounce,
> (2,1)-mode nulled by a centre strike). First model of the **nonlinear-contact family** (HANDOFF
> §12.B). A new *exciter* class: a **lumped mass in one-sided contact** — a **mallet striking a
> drumhead** (timpani/tom).
>
> **Human decisions (2026-07-10):**
> - **Target resonator = membrane** (mallet on a drumhead), not the string. The membrane is the
>   iconic soft-mallet target and the visual showpiece. Plate/bar is a trivial config swap (see below).
> - **Include Hunt–Crossley/Stulov felt hysteresis in this batch** (not deferred). So this batch
>   proves **both** gates: conservation (elastic, `λ_h = 0`) *and* passivity (hysteretic, `λ_h > 0`).
>
> **Advisor-confirmed direction (load-bearing):**
> - **Mallet-first is the de-risk.** A single lumped DOF and one contact point isolates the
>   collision-potential crux before distributed contact (string–fret / barrier / snare). Mirrors the
>   beam→plate and SS→free-edge staging.
> - **Energy-conserving force = the discrete gradient of the contact potential**
>   `f = (φ(η^{n+1}) − φ(η^{n-1})) / (η^{n+1} − η^{n-1})` (Chatziioannou–van Walstijn / Bilbao). This
>   is what makes `E` telescope *exactly*, not approximately. Reuse the bow's safeguarded
>   Newton/brentq solver almost verbatim — **not** the ψ-quadratization non-iterative variant.
> - **The discrete gradient has a removable `0/0`.** When `η^{n+1} ≈ η^{n-1}` both numerator and
>   denominator vanish; branch to the Taylor fallback `→ φ'(η^n)` below a tolerance, or it NaNs in the
>   quiet regions. The #1 implementation trap — build it in from draft.
> - **The closed-form contact-time oracle lives at a RIGID WALL, not the membrane.** Against the
>   membrane, the head carries energy away → no closed-form contact time. Stage two-tier (below).
> - **Elastic felt → conservation; hysteretic felt → passivity only.** The elastic part is
>   potential-derived (conserves); the Hunt–Crossley term is dissipative (`λ_h[η]₊^α · η̇`, power
>   `= λ_h[η]₊^α(δ_t·η)² ≥ 0`), so it can only ever remove energy. Assert conservation at `λ_h = 0`,
>   passivity at `λ_h > 0`.
> - **Stability is by construction** (non-negative contact PE + exact telescoping). The real risk is
>   **under-resolving / aliasing the stiff contact** (contact frequency `~√(K/M)`), so oversample per
>   HANDOFF §8. The collision does **not** tighten the membrane's CFL `λ = ck/h ≤ 1/√2`.

## Why the explicit membrane makes the coupling *simpler* than the bow

The bow coupled to an **implicit** string, so it needed the precomputed driving-point admittance
`a = A⁻¹eᵢ` (one banded solve baked into a vector). The membrane is **explicit**
(`u^{n+1} = (2u^n − (1−σk)u^{n-1} + c²k² L u^n)/(1+σk)`), so a point force `f` at the struck live node
`i` affects **only node `i`** at step `n+1` — the Laplacian coupling is already frozen in `u^n`. The
driving-point admittance is therefore the bare **local nodal mass**:

```
u_i^{n+1} = u_i,free^{n+1} + g_s f ,     g_s = k² / (ρ h² (1 + σk))
```

No `A⁻¹` solve, and **zero edits to `membrane.py`**: do the force-free `membrane.step()`, read
`u_i,free`, solve the scalar contact equation, then apply `membrane.u[i] += g_s · f` — algebraically
identical to putting `f` in the update numerator (the `(1+σk)` divide is why `g_s` carries it). This
mirrors the bow's non-invasive `string.u += force_pref · f_B · a_full`.

## Why this is genuinely a new model (not a template reuse)

The bow (`core/bow.py`) is the closest analog and we reuse its scalar-solve machinery, but three
things are new:

- **The exciter now has state and stores energy.** The bow is *memoryless* → `BowedString.energy()`
  is exactly the string energy and correctness is a *balance*. The mallet is a **mass** that stores
  kinetic energy `½M v_H²` and, through the felt, **potential** energy `φ(η)`. So the money test flips
  back to strict **conservation**: `E_membrane + ½M v_H² + φ(η) = const` to `~1e-13` (lossless,
  elastic).
- **The coupling is a one-sided nonlinear spring across a moving gap.** `φ` is a genuine potential, so
  the force is its *gradient* — and the energy-conserving discretization uses the **discrete
  gradient**, not `φ'` at a point. This is the new algebraic object (the analog of the VK bracket): the
  discrete-gradient identity is exactly what telescopes.
- **A second time-marched DOF.** The mallet obeys `M u_H'' = −f`, integrated with the *same* centered
  `δ_tt` scheme, so its KE telescopes in lockstep with the contact PE. Consistent velocity centering
  is the "potential across two time levels" discipline applied to a lumped mass (advisor point 4).

## Physics — mass in one-sided nonlinear contact

Mallet (lumped mass `M`, kg) at transverse position `u_H(t)`; membrane displacement `u(x,y,t)`;
contact at live node `i` (physical point nearest `(x_h, y_h)`). Penetration

```
η(t) = u_H(t) − u(x_h, t)          (η > 0  ⟺  mallet has pushed past the head surface)
```

with the mallet approaching from `+`. Felt = one-sided nonlinear spring, **contact potential**

```
φ(η) = (K / (α+1)) · [η]₊^(α+1) ,      [η]₊ = max(η, 0) ,      K > 0 ,  α ≥ 1
```

(`α = 1` linear; real felt `α ≈ 2–3`). Elastic force `φ'(η) = K[η]₊^α`. **Hunt–Crossley/Stulov**
hysteresis adds a velocity-dependent, penetration-gated damping:

```
f_contact = K[η]₊^α  +  λ_h [η]₊^α η̇ ,     η̇ = dη/dt ,   λ_h ≥ 0
```

Equations of motion (`J = (1/h²)eᵢ` the 2D spread, `I` the read):

```
M u_H''      = −f_contact
ρ u_tt = T ∇²u − 2ρσ u_t + J f_contact
```

Continuous energy and balance:

```
E(t) = E_membrane + ½ M (u_H')² + φ(η)
dE/dt = −(membrane loss ≥0) − λ_h[η]₊^α (η̇)²  ≤ 0
      = 0   exactly when  σ = 0 and λ_h = 0.
```

## Numerical scheme — energy-conserving discrete gradient + hysteresis

**Membrane:** unchanged; one force-free advance per step, then a local correction at node `i`.

**Mallet:** centered `δ_tt`,   `u_H^{n+1} = 2u_H^n − u_H^{n-1} − (k²/M) f`.

**Contact force `f` (the crux):**

```
f = f_elastic + f_hyst
f_elastic = (φ(η^{n+1}) − φ(η^{n-1})) / (η^{n+1} − η^{n-1})        [DG]  (discrete gradient)
            → φ'(η^n)                     when |η^{n+1} − η^{n-1}| < η_tol   (removable 0/0)
f_hyst    = λ_h · ⟦η⟧^α · (η^{n+1} − η^{n-1}) / (2k)              (centered η̇, penetration-gated)
```

where `⟦η⟧^α` is a nonnegative centered penetration weight (e.g. `[½(η^{n+1}+η^{n-1})]₊^α`, or the
`[DG]`-consistent `f_elastic/K`-style average) — chosen so `f_hyst · δ_t·η = λ_h⟦η⟧^α(δ_t·η)² ≥ 0`,
i.e. **provably dissipative** regardless of sign. `λ_h = 0` recovers the pure conservative scheme.

**Energy identity (why it telescopes).** Discrete power into the contact is `f · δ_t·η`. The elastic
`[DG]` gives `f_elastic · δ_t·η = (φ(η^{n+1}) − φ(η^{n-1}))/(2k) = δ_t· φ` — a perfect telescoping of
stored PE. The mallet KE telescopes via the same `δ_tt`/`δ_t·` identity every model uses. Summing the
membrane's force-work `+f·δ_t·u_i`, the mallet KE `−f·δ_t·u_H`, and the contact PE `+δ_t·φ` cancels to
`−f_hyst·δ_t·η ≤ 0`. So `E_membrane^n + ½M‖δ_{t-}u_H‖² + φ(η^n)` is **constant** (`λ_h=σ=0`) or
**monotone decreasing** (`λ_h>0` or `σ>0`).

**Scalar collapse (the bow shape).** Both DOFs are linear in `f` except through the contact:

```
u_i^{n+1}  = u_i,free^{n+1}  + g_s f ,     g_s = k² / (ρ h² (1+σk))     (membrane, LOCAL — no A⁻¹)
u_H^{n+1}  = u_H,free^{n+1}  − g_H f ,     g_H = k² / M                 (mallet)
⇒  η^{n+1} = η_free^{n+1} − (g_s + g_H) f(η^{n+1}) ,   g ≡ g_s + g_H
```

One scalar equation in `η^{n+1}`. Solve with the bow's **safeguarded Newton seeded by continuation**
(+ guaranteed **brentq** bracket fallback). Apply `f` **exactly** via the local membrane correction and
the closed-form mallet update, so the reported energy balance is machine-precision regardless of Newton
residual. `α = 1, λ_h = 0` makes `f` affine in `η^{n+1}` → a single linear step (first-light plumbing
check).

**Grazing / separation:** `η^{n+1} ≤ 0 ∧ η^{n-1} ≤ 0` ⇒ both `φ = 0` ⇒ `f_elastic = 0` (`[η]₊`
handles it); `f_hyst` gated by `⟦η⟧₊^α = 0` too ⇒ `f = 0`, mallet flies free. The `η_tol` Taylor
branch covers the *in-contact* near-stationary case (`η^{n+1} ≈ η^{n-1} > 0`). Both required.

## Files

- `core/mallet.py` — `MalletMembrane` resonator (engine `Resonator` protocol: `step`/`energy`/`state`/
  `k`/`displacement_at`), holding a `Membrane` + lumped-mass state, structured like `bow.py`. Free
  functions written **vector-ready** (scalar = size-1 case, so the later distributed-barrier model
  reuses them): `contact_potential(η, K, α)`, `contact_force_elastic(η, K, α)`,
  `contact_force_dg(η_next, η_prev, K, α, tol)` (the `[DG]` + Taylor branch), and the hysteresis term.
- `tests/test_mallet_wall.py` — **standalone** mass-vs-fixed-wall: closed-form contact time
  `π√(M/K)` and exact velocity reversal at `α = 1, λ_h = 0`; energy conservation of `½M v² + φ(η)`;
  the `[DG]` `0/0` branch exercised (grazing seed); with `λ_h > 0`, monotone energy loss (passivity).
- `tests/test_mallet_energy.py` — **coupled** mallet→membrane: conservation
  `E_membrane + ½M v_H² + φ(η) = const` to `~1e-13` (lossless, `λ_h=0`), at a strike amplitude where
  contact PE is a meaningful fraction of the total; no-op-when-mallet-misses = **bit-identical** to the
  bare membrane (the `K=0` analog); passivity with `λ_h > 0` and/or `σ > 0`; drift ∝ `newton_tol`/
  `η_tol` self-certification.
- `tests/test_mallet_signature.py` — physical signatures: strike excites many membrane modes; harder
  mallet (larger `K`/`v_H`) → brighter spectrum + shorter contact; the mallet **bounces** (single
  contact then separation); strike at the head **centre** favours axisymmetric modes vs an off-centre
  strike (mode content shifts).
- `scripts/diagnose_mallet.py` — contact-force pulse vs time, mallet/head trajectory through contact,
  energy-partition trace (membrane ⇄ KE ⇄ PE, flat total elastic / decaying hysteretic), spectrum vs
  mallet hardness, hysteresis loop (`f` vs `η`, showing the loading≠unloading area = energy lost).

## Validation ladder (two-tier, mirrors VK bracket-before-loop)

1. **Standalone (no membrane) — the closed form lives here.** Integrate `M η'' = −φ'(η)` against a
   fixed wall with the `[DG]` scheme. At `α = 1, λ_h = 0`: contact = half-period of `ω = √(K/M)`, so
   contact duration `= π√(M/K)` and exit velocity `= −`entry velocity — assert both tightly. Energy
   `½M v² + φ(η)` conserved to `~1e-13`. Then `λ_h > 0`: energy monotone decreasing, exit speed <
   entry (coefficient of restitution < 1). De-risks the collision scheme in isolation before coupling,
   exactly as `test_vk_bracket.py` did.
2. **Coupled — energy conservation of the whole.** Money test: lossless membrane + elastic felt ⇒
   `E_membrane + ½M v_H² + φ(η)` flat to `~1e-13`. Plus the bit-identical miss check and the
   drift-∝-tol self-certification.
3. **Passivity.** `λ_h > 0` and/or `σ > 0` ⇒ `E` monotone decreasing.
4. **Signatures** (test_mallet_signature): mode content vs strike position, hardness→brightness,
   single-bounce contact.

## Resolved / remaining decisions

- **Target resonator: membrane** (human, 2026-07-10). Plate/bar follow-on reuses the same scalar
  collapse with `g_s` from the plate's driving-point admittance (`Plate.step(f_ext)` already exists
  from the StringPlateBridge work) instead of the local nodal mass.
- **Hysteresis: included this batch** (human, 2026-07-10) — conservative core first inside the same
  file, hysteresis term added and passivity-tested alongside.
- **Contact primitives live in `core/mallet.py`, written vector-ready**; promote to `core/collision.py`
  only when the distributed-barrier model (the second consumer) actually lands (avoid premature
  abstraction). Default taken.
- **Felt exponent:** any `α ≥ 1` conserves; demo default `α = 2.3`, `K` chosen so `√(K/M)` is
  well-resolved at the render `fs` (warn if the contact is under-resolved). `α = 1` used for the
  closed-form oracle. Default taken.
- **Sub-grid contact position:** snap to nearest live node for v1 (bow precedent); interpolated read/
  spread is a shared later refinement. Default taken.

## Non-goals for this batch

Distributed contact (string–fret, rigid barrier, snare buzz), multiple simultaneous mallets,
two-point/rolling contact, mallet-shaft dynamics, sub-grid interpolation. All reuse the `φ`/`[DG]`
primitives and are natural follow-ons once the lumped single-point case is proven.
