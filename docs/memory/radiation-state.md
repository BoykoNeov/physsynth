---
name: radiation-state
description: "Air/radiation node batches 1 (read-out AirRadiation) + 2 (constant-R LOAD RadiatedBody) + 3 (FREQUENCY-DEPENDENT load RationalAirLoad/ReactiveRadiatedBody, exact rational Z_a) ALL built & green and batch 3 SURFACED in the viewer as `airload`; batch 2's R and batch 3's R are DIFFERENT quantities (compact-source at one frequency vs saturated plane-wave)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 93bc8edb-eee0-4c24-8331-79fb2e33247c
  modified: 2026-07-27T18:44:46.376Z
---

**Air/radiation node — the last node of `exciter → resonator → body/radiation`.** Human decisions
(2026-07-09): tier = **lumped passive impedance** (NOT 3D FDTD air box, which is §12H later); depth =
**read-out first, load next** (two batches). This closes the chain start→ *radiated sound*.

**Batch 1 DONE & GREEN (2026-07-09): `physsynth/core/radiation.py` `AirRadiation`.** Pure passive
**output transform** — free-space monopole (simple source): `p_far(t,r) = ρ₀/(4πr)·Q''(t − r/c₀)`,
where `Q'' = body.pressure()` is the net **volume acceleration** (already exposed project-wide on
`ModalBody`/`Plate`/bridges — pressure = monopole ∝ volume accel). **No back-reaction** → body energy
untouched → node deliberately has **no `energy()`** (stores/dissipates none; that's batch 2).
- Retardation = **integer-sample** delay line `round(r/c₀·fs)` → amplitude preserved EXACTLY (a
  fractional/interp delay would low-pass and blur the amplitude oracle); sub-sample residual exposed
  on `.retardation_residual`. `retarded=False` → instantaneous gain (delay is inaudible latency for
  one source, matters when summing sources at a listener). First `latency_samples` outputs = silence
  (wavefront in transit).
- API: `process(volume_accel)`, `radiate(source)` (duck-types on `.pressure()`), `reset()`; attrs
  `gain=ρ₀/4πr`, `latency_samples`, `retardation_seconds`. Constants `RHO0_AIR=1.2041`, `C0_AIR=343`.
- **Oracles (closed-form, no ears):** gain exact `p=gain·Q''`; monopole amplitude from prescribed
  volume velocity `U₀sin(ωt)` → `|p|=ρ₀ωU₀/(4πr)`; inverse-distance `p∝1/r`; retardation = exact
  N-sample delay (impulse test); linearity/superposition; end-to-end ModalBody + full
  string→bridge→body chain radiate finite non-trivial pressure = gain·volume-accel (delayed).
- `tests/test_radiation.py` (16), `scripts/diagnose_radiation.py` (3 figs: source-vs-radiated w/
  wavefront-arrival marker, 1/r collapse, radiated spectrum = body spectrum tilted ×ω²).
  helpers.py: `make_radiation` + `RADIATION_DISTANCE_DEFAULT`. Suite **539** (was 523), ruff clean.

**Batch 2 DONE & GREEN (2026-07-10): `RadiatedBody` (in `core/radiation.py`) = radiation LOAD +
back-reaction.** A **rank-1 passive dashpot** on a `ModalBody`: air presents constant acoustic
resistance `R` (Pa·s/m³) to the body's net volume velocity `U=Σaᵢq̇ᵢ` (**reciprocity**: the same
read-out weights `aᵢ` couple the uniform back-pressure `R·U` into each mode); removed power
`P_rad=R·U²` tracked as `radiated_energy=∫P_rad dt` → identity `E_body+∫P_rad=const` (σ=0 drift
**7.86e-14**), full chain `E_string+E_body+E_conn+∫P_rad` (drift <1e-9); `R=0` **bit-identical** to
bare body.
- **Scheme = centered (implicit) → single scalar Sherman-Morrison** (advisor-gated batch): force per
  mode `-aᵢRU` with `U` **centered** `aᵀ(qⁿ⁺¹−qⁿ⁻¹)/(2k)` (implicit, like body's own `2σq̇` → why
  dissipation `kRU²≥0` provable; explicit one-sided U blows up at large R). Free step
  `q̃=ModalBody.step(force)`, then `U=Ũ/(1+RG)`, `G=(k/2)Σaᵢ²/(mᵢ(1+σᵢk))`, correct
  `qᵢ=q̃ᵢ−[k²aᵢ/(mᵢ(1+σᵢk))]RU`. Corrected-state centered velocity **==U** (`Ũ−RUG=U`) → energy
  telescopes machine-precision. Mirrors [[bow-state]] rank-1 but **linear (no Newton)**.
- **UNCONDITIONALLY passive — no CFL, NO guard** (`1+RG≥1` ∀R≥0, never singular): unlike the bridge
  springs whose rank-1 block can go negative. Tested at R=1e12 → still finite + conserved.
- **Advisor corrections (the "advisor first" payoff):** (1) **units** — `R·U²` needs **ACOUSTIC**
  `R_a=(ρ₀c/S)[1−J₁(2ka)/(ka)]` (Pa·s/m³), NOT the memory's **mechanical** `ρ₀c·πa²·[…]` (=`S²·R_a`,
  off by S², not watts); (2) **keep `(1+σk)`** in G AND corr (I'd dropped it — invisible at σ=0);
  (3) constant R = right scope, honest = no spectral fidelity (`R_a∝ω²`), freq-dep R is a separate
  harder batch; (4) **free-space (4π)** to match batch 1, baffled piston is half-space (2π) = 2×.
- **`RadiatedBody` is a `ModalBody` DROP-IN** (`__getattr__` delegates phi/m/omega/q/…; overrides
  step/energy/pressure) → slots into `StringBodyBridge` **as the body** with zero bridge edits → full
  chain money test for free. `pressure()` refreshes `body._accel` from the CORRECTED 2nd-diff.
- Helpers `monopole_radiation_resistance` (`ρ₀ω²/(4πc)`, ∝ω²) + `piston_radiation_resistance`
  (baffled Bessel, Rayleigh `(ka)²/2`→`ρ₀ω²/(2πc)`=2×monopole) = the closed-form **modeling** oracle
  (tested separately vs Rayleigh limit — NOT asserting spectrum matches a piston).
- **Advisor debrief catch (before done):** ALL σ=0 tests → `(1+σk)` factor **unpinned** (a
  wrong-but-consistent factor passes self-consistency AND monotonicity). Fix = σ=3,R=1500 loaded step
  **vs exact dense coupled solve** `[diag(1+σk)+(kR/2)(a/m)aᵀ]q=free_rhs+(kR/2)(a·qⁿ⁻¹)(a/m)`; correct
  matches 3e-19, dropped-factor diverges 2.4e-11, atol=1e-13.
- `tests/test_radiation.py` +12 (28 total), `scripts/diagnose_radiation_load.py` (energy-channel
  flat-total, ring-down vs R, R_a(f) oracle), `helpers.make_radiated_body`+`R_RADIATION_DEFAULT`.
  Suite **550** (was 539), ruff clean. Builds on [[body-bridge-state]] / [[bow-state]].

**Batch 3 DONE & GREEN (2026-07-27): `RationalAirLoad` + `ReactiveRadiatedBody`** — the
frequency-dependent load, discharging batch 2's own docstring refusal. Plan:
`docs/dev/radiation-frequency-dependent-plan.md`. **The insight that made it cheap: no filter fit is
needed.** The pulsating sphere's acoustic impedance is *already* first-order rational,
`Z_a = R·jωτ/(1+jωτ)` = **resistance R in PARALLEL with the radiation mass `M_a`** (`R=ρ₀c₀/S`,
`M_a=ρ₀/(4πa)`, `τ=M_a/R=a/c₀`), so **ONE aux state realises it EXACTLY** and the R–M_a network **is**
the passivity proof — `U` splits, resistor branch radiated, inertance branch stored.
- **Energy gains a STORED term:** `E_body + ½M_a U_L² + ∫R U_R² = const` (7.4e-14). Batch 2's air
  could only take; this one **gives back** (the stored curve rises AND falls — that's a test).
  `½·inf·0.0²` is **NaN** → `stored_energy()` special-cases `isinf(M_a)`; this would poison the
  whole suite through the reduction test.
- **Scheme = trapezoid on the inertance = the BILINEAR transform.** Collapses to `p = R_eff(U−L⁻)`,
  `R_eff = R/(1+kR/(2M_a))`, so batch 2's scalar Sherman–Morrison is **unchanged in shape** (just
  `R→R_eff` + a known offset). `R_eff∈[0,R]` ⇒ `1+R_eff·G ≥ 1` ⇒ still **unconditionally passive**,
  no CFL, no guard (tested at R=1e12).
- **API = `(R, M_a)` effective coefficients, NOT a radius** (`from_sphere` is the consistency
  helper) — the [[unphysical-params-are-a-feature]] rule, and the *only* parameterisation that
  reaches `M_a=inf` = batch 2 **bit-identical**. Bit-identity requires matching batch 2's **operation
  order** (form `u*` first, then `×R_eff`; take `U = u* + L⁻`, not `U_free − pG`).
- **THE TRAP the advisor caught pre-build:** batch 1 is the `a→0` compact read-out; a *finite*
  sphere's far field carries an extra `1/(1+jka)`, so a power balance against `AirRadiation` misses
  by `1+(ka)²` and **reads as a bug**. Correct form `p_far=(a/r)·p_load` balances **exactly at every
  ka** (`S|Z_a|²/(ρ₀c₀) ≡ Re Z_a`). Shipped the miss as a *negative control*, batch 1 untouched.
- **Pre-warping:** the scheme realises `Z_a` at `s=(2j/k)tan(ωk/2)`, so the measured sweep matches
  `impedance_discrete` to **8.3e-16** — comparing to continuous `Z_a(jω)` shows an O((ωk)²) gap that
  looks like a scheme bug. Sweep drive = `G=0` (a rigid piston at prescribed `U`) — no extra rig.
- **THE ORACLE I GOT WRONG, found by measuring (26% off):** per-mode `α = a²ReZ/(2m)` ignores the
  **REACTANCE**, which is an added mass `m_add = a²ImZ/ω` lowering the frequency *and* the
  denominator. Self-consistent `loaded_mode()` → **0.5%**, frequency to **5 digits**. So the headline
  is TWO claims: high partials die first (α spread >20× over 3 octaves) **and the air FLATTENS the
  pitch** — a constant R can do neither (`Im Z = 0`), which is the negative control.
- **Second measurement trap:** fitting decay through the log of the **modal energy** is badly biased
  (that expression assumes the bare ω the loaded mode no longer oscillates at). **Fit envelope
  peaks.** Also: batch 2's unpinned `(1+σk)` recurs with `R_eff(k)` beside it → the dense coupled
  cross-check now runs at **two timesteps**.
- Zero edits to `AirRadiation`/`RadiatedBody`/`body.py`/`connection.py`; `__getattr__` delegation
  carries the full string→bridge→loaded-body chain free. `tests/test_radiation.py` 28 → **72**,
  `scripts/diagnose_radiation_impedance.py` (3 figs), `helpers.make_reactive_body`. Suite **1173**
  green (was 1129 at Phase D close; +44 all in `test_radiation.py`, so the delta *is* the batch),
  ruff clean.

**SURFACED IN THE VIEWER 2026-08-10 as `airload` (web-viewer batch 17)** — see
[[web-viewer-state]]. The one thing that batch changed about *this* note: **batch 2's `R` and batch
3's `R` are DIFFERENT PHYSICAL QUANTITIES and I nearly shipped a headline built on conflating
them.** Batch 2's default came from `monopole_radiation_resistance` = the **compact-source**
(`ka→0`) law at one frequency; batch 3's is the **saturated plane-wave** `ρ₀c₀/S`. Reading the
former as a saturation implies a 0.497 m sphere whose exact `Re Z_a` over 110–440 Hz is
66.6/101.3/113.0/125.3 — **−50…−94 %** against the law batch 2 quoted — and would have "proved"
batch 2 confessed to a sin it never committed. The sphere batch 2's number actually describes is
the **5 cm** one (132.1/410.4/710.7/1836.8, i.e. −1.0/−3.1/−5.4/−14.0 %), so batch 2's honesty note
was CORRECT and its numbers were good to 14 %. The two limits meet only at `ka = 1`, which is also
why `√(Rc₀/(ρ₀π))` ≡ batch 2's `f_match` — true, but near-tautological, so it is a sanity check and
not evidence of continuity. *Two batches can name a slider `R`, cite the same medium, and mean
different limits of the same formula.*

**Next horizon** = distributed/3D FDTD air box (§12H — four batches SHIPPED in core, see
[[air-box-state]], but still UNSURFACED in the viewer and needing a 3-D field type first) or
plate/grid radiation load (deferred: area-weighted W, implicit θ-solve). See [[web-viewer-state]]
for the fork list.
