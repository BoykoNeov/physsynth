# Frequency-dependent radiation — the rational impedance `Z_a(ω)` (radiation batch 3)

> **Status: BUILT & GREEN (2026-07-27).** `core/radiation.py` gains `RationalAirLoad` (+
> `from_sphere`, `impedance`, `impedance_discrete`, `loaded_mode`, `far_field_pressure`) and
> `ReactiveRadiatedBody`; `tests/test_radiation.py` 28 → **71**;
> `scripts/diagnose_radiation_impedance.py` (3 figures); `helpers.make_reactive_body`. Zero edits
> to `AirRadiation`, `RadiatedBody`, `body.py` or `connection.py`. **Measured:** impedance sweep vs the pre-warped closed form **8.3e-16**;
> three-way energy identity flat to **7.4e-14**; per-mode decay and the added-mass pitch drop within
> **0.5%** of the closed form (frequency to 5 digits); both reductions bit-identical.
>
> **The one thing the plan got wrong, found by measuring:** the per-mode oracle
> `alpha = a² Re Z_a(omega_i)/(2 m_i)` is incomplete — it misses the **reactance**, an added mass
> `m_add = a² Im Z_a/omega` that lowers the frequency *and* the denominator. With that included
> (self-consistently, since both depend on the frequency they shift) the agreement goes from 26% off
> to 0.5%, and the test now exercises **both** parts of `Z_a` instead of only its real part. Shipped
> as `RationalAirLoad.loaded_mode()`. A second trap surfaced with it: fitting the decay rate
> through the log of the *modal energy* is badly biased, because that expression assumes the bare
> `omega` the loaded mode no longer oscillates at — fit the envelope peaks instead.

> *(Original plan below, as written before the build.)* The first core batch after Phase D
> closed. Extends `core/radiation.py`: batch 1 was the read-out (`AirRadiation`), batch 2 the *resistive* load
> (`RadiatedBody`, one constant `R`), and this is the **full first-order impedance** — resistance
> **plus reactance** (the radiation mass). It is the one thing batch 2 explicitly refused, in its own
> docstring: *"the constant-`R` `RadiatedBody` evaluates this at one reference frequency; true
> per-mode spectral shaping is a later, frequency-dependent batch."* This is that batch.

---

## 1. Why — the refusal being discharged

`RadiatedBody` loads the body with **one number**. Air does not do that: the free-space monopole
resistance is `R_a(ω) = ρ₀ω²/(4πc₀)` — it rises as `ω²`. A constant `R` therefore over-damps the
low modes and under-damps the high ones by whatever ratio the reference frequency happens to sit at.
The audible content of radiation damping — *high partials die first because they radiate better* —
is exactly the part a constant `R` cannot produce. Batch 2 was honest about this and shipped the
constant anyway (right call: the passive rank-1 machinery had to exist first). This batch makes the
load frequency-dependent **without** giving up the two properties that made batch 2 trustworthy:
provable passivity and a closed-form oracle.

The key realisation: we do **not** need a filter-design approximation. The exact monopole radiation
impedance is a **first-order rational function of `jω`**, so it is realisable in the time domain
*exactly*, with **one** auxiliary state.

---

## 2. The physics — a parallel `R`–`M_a` pair, exact

For a **pulsating sphere** of radius `a` (the canonical compact monopole), the acoustic radiation
impedance seen by its volume velocity `U` is exactly

```
Z_a(jω) = (ρ₀c₀/S) · jka/(1 + jka) ,      S = 4πa² ,   k = ω/c₀ .
```

Write it as a resistance in **parallel** with an inertance:

```
Z_a(jω) = R · jωτ/(1 + jωτ) ,    R = ρ₀c₀/S = ρ₀c₀/(4πa²) ,
                                 M_a = ρ₀/(4πa) ,        τ = M_a/R = a/c₀ .
```

`M_a` is the classic **radiation mass** (acoustic inertance, kg/m⁴) — the air that gets dragged
along without being radiated. The circuit reading is the whole scheme:

```
U = U_R + U_L ,        p = R·U_R = M_a·dU_L/dt .
```

The volume velocity **splits**: the part through the resistor is radiated (lost to the far field),
the part through the inertance is stored (returned). That split is what makes the frequency
dependence *passive by construction* — no filter to check for positive-realness, the network **is**
the proof.

**Two closed-form anchors, both already in the repo's language:**

| limit | value | anchor |
|---|---|---|
| `ka → 0` | `Re Z_a → ρ₀ω²/(4πc₀)` | **exactly** the existing `monopole_radiation_resistance()` |
| `ka → ∞` | `Re Z_a → ρ₀c₀/S` | plane-wave (fully-loaded) saturation |

So batch 2's helper is not superseded — it becomes the **low-frequency limit** of this batch's
impedance, which is the continuity check between them. `Im Z_a` peaks at `ka = 1`.

---

## 3. Scope — and what is deliberately deferred

**In scope:** the general first-order positive-real acoustic load `(R, M_a)`, its passive
implicit discretisation, the energy identity with a *stored* air term, the impedance oracle, the
per-mode radiation-damping oracle, the far-field power balance, and drop-in composition with
`StringBodyBridge`.

**Out of scope (say so, don't drift):**

- **3-D FDTD air box** — HANDOFF §12H. A different animal (a spatial grid, not a lumped port). The
  lumped rational impedance is precisely the tier *below* it, and shipping it does not commit us.
- **Non-rational loads (the baffled piston's `1 − J₁(2ka)/(ka)`)** — not a finite-order rational
  function, so an exact one-state realisation does not exist; it would need a fitted multi-pole
  approximation plus a positive-realness proof. `piston_radiation_resistance()` stays what it is: a
  closed-form *modeling* oracle, not the load.
- **Distributed / per-node radiation on a grid resonator (plate, membrane)** — still deferred from
  batch 2 (area-weighted `W`, implicit θ-solve). This batch stays on the modal port.
- **A viewer batch.** Phase D is closed; this is core work. If the human wants it surfaced later,
  it is a separate decision.

---

## 4. API — effective coefficients first, physics as a helper

Primary parameters are **`R` (Pa·s/m³) and `M_a` (kg/m⁴), independently** — *not* a sphere radius.
Three reasons:

1. It follows the standing project rule (`unphysical-params-are-a-feature`): expose effective
   coefficients that permit inconsistent combinations, offer realism through a constructor helper,
   never impose it.
2. **The reduction to batch 2 is only reachable this way.** `M_a → ∞` at fixed `R` is the
   constant-resistance load; you cannot reach it from any radius (a sphere ties `M_a` to `R`).
3. It is the general first-order positive-real load, so a future fitted piston/horn termination
   reuses the same object.

```python
RationalAirLoad(R=..., M_a=...)                    # general first-order positive-real load
RationalAirLoad.from_sphere(radius=a, ...)         # the physically consistent pulsating sphere
ReactiveRadiatedBody(body=..., load=...)           # the ModalBody drop-in (batch 2's shape)
```

**`M_a = np.inf` must be exactly batch 2.** In IEEE arithmetic `kR/(2·inf) = 0.0` and `p/inf = 0.0`,
so `R_eff = R` *exactly* and the auxiliary state stays *exactly* zero — provided the operations are
ordered to match `RadiatedBody.step` term-for-term (see §5). That gives the family's signature test
for free, alongside `R = 0` ⇒ bare `ModalBody`:

> `R=0` ↔ bare body · `M_a=inf` ↔ `RadiatedBody` · `σ₁=0` ↔ model #2 · `nonlinear=False` ↔ #5

**Naming.** `ReactiveRadiatedBody` — batch 2's load is purely *resistive*; what is new here is the
*reactance*. (Considered and rejected: `FrequencyDependentRadiatedBody`, too long;
`RadiatedBodyRational`, obscures the physics.) Splitting the load into its own small class is
deliberate — see the impedance oracle in §7.2, which needs to drive the load standalone.

---

## 5. The discrete scheme — trapezoid aux state, same Sherman–Morrison collapse

Batch 2's structure survives intact. Centered (implicit) volume velocity, as before:

```
U^n = aᵀ(q^{n+1} − q^{n−1}) / (2k) ,     free step ⇒ U_free ,   U = U_free − p·G ,
G = (k/2) Σ aᵢ²/(mᵢ(1+σᵢk)) ,            correction  qᵢ = q̃ᵢ − p·cᵢ ,  cᵢ = k²aᵢ/(mᵢ(1+σᵢk)) .
```

The only change is that the load pressure `p` is no longer `R·U`. Trapezoid (= bilinear) on the
inertance branch, with the aux state on **half-steps** so it aligns with the body's `n+1/2` energy:

```
M_a (U_L^{n+1/2} − U_L^{n−1/2}) / k = p^n ,     U_L^n = (U_L^{n+1/2} + U_L^{n−1/2})/2 ,
p^n = R·U_R^n = R (U^n − U_L^n) .
```

Eliminate `U_L^n` with `L⁻ ≡ U_L^{n−1/2}` known:

```
p = R_eff (U − L⁻) ,        R_eff = R / (1 + kR/(2M_a)) .
```

**That is the whole batch on the body side: `R → R_eff`, plus a known offset `L⁻`.** Substituting
`U = U_free − pG` and solving the scalar:

```
u* = (U_free − L⁻) / (1 + R_eff·G)      p = R_eff·u*        U = u* + L⁻    (exact identity)
q  = q̃ − p·c                            U_L^{n+1/2} = L⁻ + (k/M_a)·p      U_R = U − U_L^n
```

**Order of operations is load-bearing** for the bit-identity claim: compute `u*` first and multiply
by `R_eff` (batch 2 computes `u` then `R*u`), and take `U = u* + L⁻` rather than `U_free − pG` — the
two are algebraically equal but only the former is bit-identical when `L⁻ = 0`.

**Unconditional passivity survives:** `R_eff ∈ [0, R]` for all `M_a > 0`, so `1 + R_eff·G ≥ 1` — the
scalar solve is never singular, at any `R`, any `M_a`, any `k`. No CFL, no guard, same as batch 2.

### 5.1 Energy — a *stored* term joins the channel

```
E_total = E_body  +  ½ M_a (U_L^{n+1/2})²  +  Σ_n k·R·(U_R^n)²   =  const     (σ = 0)
          ────────  ──────────────────────    ───────────────────
           body      air kinetic (STORED)      radiated (DISSIPATED)
```

Telescoping, exactly:

```
ΔE_air = ½M_a[(L⁺)² − (L⁻)²] = M_a·U_L^n·(L⁺−L⁻) = k·p·U_L^n
k·p·U^n = k·p·U_R^n + k·p·U_L^n = k·R·(U_R)²  +  ΔE_air
```

and the body sheds exactly `k·p·U^n` (batch 2's identity with `RU → p`). So work-out = stored +
dissipated, term for term, to machine precision. This is **new structure**, not a relabelling:
batch 2's air could only *take*; this air can also *give back*. A wrong reactance shows up as drift,
which is why the identity is still the primary bug detector.

**`M_a = inf` trap:** `½·inf·0.0²` is `nan`, not `0`. The energy accessor must special-case
`isinf(M_a) → 0.0` (or store the air energy incrementally). This would poison the *entire* suite
through the reduction test, so it is written down here.

---

## 6. Traps (advisor-flagged, before a line is written)

1. **Batch 1 will *not* balance against this load — and it looks like a bug.** `AirRadiation` is the
   `a → 0` read-out; for a finite sphere the far field is additionally low-passed by `1/(1+jka)`, so
   a power balance against it misses by `1/(1+(ka)²)`. The correct far field is the surface pressure
   scaled and retarded, `p_far(r,t) = (a/r)·p_load(t − (r−a)/c₀)`, and with *that* the balance is
   **exact at all `ka`** because `S·|Z_a|²/(ρ₀c₀) ≡ Re Z_a`. Put the read-out on the new class;
   **do not touch batch 1** (its 16 tests stay untouched). Test the two paths agreeing as `ka → 0`
   separately, and print the `1/(1+(ka)²)` miss as the negative control.
2. **Pre-warping.** Trapezoid *is* the bilinear transform, so the discrete load's impedance is `Z_a`
   evaluated at `s = (2j/k)·tan(ωk/2)`, **not** at `jω`. Assert machine precision against the
   **pre-warped** form; report the deviation from continuous `Z_a(jω)` separately as the honest
   "how close is the discrete load to physics" number (and check it falls at `O(k²)`).
3. **Batch 2's debrief catch recurs, with a second factor.** The `(1+σᵢk)` in `G`/`c` was unpinned
   because every test ran at `σ = 0`; now `R_eff`'s `k`-dependence is *also* unpinned if every test
   runs at one `k`. The σ>0 dense-coupled-solve cross-check must be extended to include the aux
   state and run at **two different `k`**.
4. **Far-field read-out needs sphere consistency.** For a general `(R, M_a)` pair the equivalent
   radius `a_eq = c₀M_a/R` and equivalent area `S_eq = ρ₀c₀/R` need not satisfy `4πa_eq² = S_eq`.
   Construction stays permissive (the standing rule); only `far_field_pressure()` — the read-out
   that genuinely requires the interpretation — refuses, with a message naming the mismatch.

---

## 7. Oracles — what must pass

**Structural (the identity):**

1. **Energy conservation.** σ=0, several `(R, M_a)` incl. extremes (`R=1e12`, `M_a` tiny/huge):
   `E_body + E_air + ∫P_rad` drifts `< 1e-12` relative. Passivity: σ>0 ⇒ monotone decrease.
2. **Reductions, bit-identical.** `R=0` ⇒ bare `ModalBody`; `M_a=inf` ⇒ `RadiatedBody(R)` — equal
   arrays, `==`, not `allclose`.

**Spectral (the money oracles — what batch 2 could not have):**

3. **Impedance sweep.** Drive the load standalone with a prescribed sinusoidal `U` (that is exactly
   `G = 0` — the rigid piston: `p = R_eff(U − L⁻)`, `U = U_free`), measure `p/U` in steady state,
   compare magnitude *and* phase to the **pre-warped** `Z_a`: machine precision. Then report the
   deviation vs continuous `Z_a(jω)` and assert it converges at `O(k²)`.
4. **Both limits.** `Re Z(ω→0)` matches `monopole_radiation_resistance(ω)` (the batch-2 helper — the
   continuity check); `Re Z(ka≫1)` saturates at `ρ₀c₀/S`.
5. **Per-mode radiation damping — the physics claim.** A weakly-loaded multi-mode body: each mode's
   measured decay rate matches `α_i = a_i²·Re Z_a(ω_i)/(2m_i)` predicted from the impedance at that
   mode's frequency. **This is the thing a constant `R` gets wrong**, so the same test with
   `RadiatedBody` at a reference frequency is the negative control: it must visibly fail on the
   modes far from the reference. High partials dying first, *measured*, not asserted.
6. **Far-field power balance** (§6.1). Time-averaged `4πr²⟨p_far²⟩/(ρ₀c₀)` equals the booked
   `⟨R·U_R²⟩`, at several `ka` including `ka ≈ 1`.

**Cross-checks:**

7. **σ>0 dense coupled solve, at two `k`** (trap 3) — the only test that pins `(1+σk)` *and*
   `R_eff(k)` absolutely.
8. **Full chain.** `ReactiveRadiatedBody` as the body inside `StringBodyBridge`:
   `E_string + E_body + E_conn + E_air + ∫P_rad` conserves, with **zero edits** to `connection.py`
   (the `__getattr__` delegation must carry it, as it did in batch 2).

---

## 8. Deliverables

- `physsynth/core/radiation.py`: `RationalAirLoad` (+ `from_sphere`, `impedance`,
  `impedance_discrete`, `far_field_pressure`) and `ReactiveRadiatedBody`. No edits to
  `AirRadiation`; no edits to `RadiatedBody`; no edits to `body.py` or `connection.py`.
- `tests/test_radiation.py`: extended (batch 3 block). `tests/helpers.py`: `make_reactive_body`
  + defaults.
- `scripts/diagnose_radiation_impedance.py`: `Z_a(ω)` measured-vs-closed-form (magnitude + phase),
  the energy triple-channel flat total, and the per-mode decay-rate comparison against constant-`R`.
- Docs: HANDOFF/README radiation lines, memory `radiation-state`.

**Acceptance:** all of §7 green, full suite green (baseline pinned below), `ruff check .` clean.

**Baseline pinned at `1533b51`.** Suite after the batch: see the `feat:` commit that lands it.
