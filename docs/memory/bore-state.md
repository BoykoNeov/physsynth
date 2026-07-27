---
name: bore-state
description: "Wind/brass leg — linear acoustic bore + radiating bell (clarinet), batches 1-2 (ALL 3 wind batches now done; reed = [[reed-state]]); staggered p-U leapfrog, energy-first half-cell wall, odd-harmonic oracle, passive radiation-resistance bell (RadiatedBody pattern on a boundary node)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 44c5c948-3546-4f21-a6e1-1dd6ade18ada
---

**Wind/brass leg — the last major method family (HANDOFF §12.A). Human chose direction "wind/bore
+ reed" over web-viewer-catchup / spatial-audio / A-B-recordings (2026-07-10).** Staging (advisor,
load-bearing order): **batch 1 = linear bore → batch 2 = radiating bell (loss) → batch 3 = dynamic
reed exciter.** The order is NOT bore→reed: a reed on a *lossless* bore diverges (the bow only
settled because the string had σ₀>0), so a passive loss channel must exist before the reed. Target
instrument = **clarinet** (closed-open cylinder, cleanest validation signature); reed scope = human
chose **dynamic mass-spring-damper reed** (not quasi-static) for batch 3 — a coupled 1-DOF
oscillator whose tip modulates the Bernoulli flow area + its own swept-volume flow.

**BATCH 1 DONE & GREEN (2026-07-10): `core/bore.py` `Bore` resonator, 43 tests, full suite 594.**
The **first acoustic resonator** of `exciter→resonator→body/radiation`. Do NOT re-open FDTD-vs-
waveguide: non-negotiable #2 (energy-FDTD) settles it; Bilbao does wind as energy-FDTD of the 1D
tube, which is what yields the mandatory `energy()`. DWGs are §12.A *future* breadth, not the
foundation.

- **Scheme = first-order STAGGERED p/U leapfrog** (Yee cell in 1D), Webster horn eq. Pressure `p` at
  N+1 integer nodes; volume velocity `U=Sv` at N half-nodes, half a step offset in time.
  `U^{n+1/2}=[(1-σk)U^{n-1/2} - (kS_seg/(ρ₀h))(p_{l+1}-p_l)]/(1+σk)`; `p^{n+1}=p^n -
  (k/C_l)·div(U)`. **NOT** a 2nd-order pressure PDE — the reed injects *flow* and the bell is an
  *impedance* p=ZU, both natural in p-U (advisor). CFL `λ=c₀k/h≤1`, **exact/dispersionless at λ=1**.
- **Energy-first boundaries (the free-beam/string lesson AGAIN).** `E = ½Σ(w_l S_l/ρ₀c₀²)p_l² +
  ½Σ(hρ₀/S_j)·U_j^{n+1/2}U_j^{n-1/2}`. The velocity term is the **CROSS-TIME product** (never the
  same-time square) — collapsing it drifts, same trap as the string potential term. Telescopes to
  **exactly zero** algebraically (continuity uses Gᵀ against the same G momentum uses; independent of
  the diagonal weights). A **rigid/closed wall = the trapezoidal `h/2` half-cell** at that node — NO
  ghost stencil (the free-beam end-mass mechanism); an **open end pins p=0** (Dirichlet, carries no
  energy, flux pU=0 → an ideal open end is LOSSLESS, radiates nothing yet). Per-end
  `("closed","open")` like the string; clarinet default `("closed","open")`.
- **Clarinet signature (the money test).** Closed-open → **ODD harmonics only** `fₙ=(2n-1)c₀/4L`;
  open-open → full series `nc₀/2L`. Confirmed **TWICE independently** (advisor stressed this proves
  it's structural, not a pluck artifact): measured spectrum odd/even ≈ 5.6e5×, AND the operator's
  own eigenvalues (`bore_low_eigenfrequencies`) return only 1,3,5,7,9,11×f₁.
- **λ=1 at 0.000 cents is NOT luck** — it structurally certifies the half-cell wall (a wrong `h/2`
  weight would shift the eigenvalues off the continuum). The one number I was least sure of, proven.
- **Webster S(x) from day one**: `S_node[N+1]`, `S_seg[N]` arrays throughout → cone/flare/sax = a
  different area profile, NOT a rewrite. Cylinder = `radius` scalar → `S=πr²`.
- **Loss = `-2σU` FREQUENCY-INDEPENDENT viscous drag** — a **passivity PLACEHOLDER** (ideal-string-σ
  role before model #3), single mode decays at 2σ (0.18% match), monotone. **NOT** the physical √ω
  viscothermal wall loss (Zwikker–Kosten) — real freq-dependent + bell losses come in batch 2. σ=0
  bit-for-bit lossless.
- **Modal oracle**: eliminating U → `C δ_tt p = -k²L p`, `L=GᵀM⁻¹G` (pressure stiffness, exposed as
  `bore.Lop`), `C`=`bore.Cmat` compliance mass, `bore.dof`=free (non-open) node indices. Generalized
  eig `(L,C)` on free dof → ω²; map through `modal.discrete_bore_eigenfrequency(ω², k)` =
  `arcsin(k√ω²/2)/(πk)` (leapfrog SHO). Closed-open/open-open are PD (open end pins) → σ=0 shift-
  invert; closed-closed has constant-pressure nullspace → negative shift + drop ω≈0 mode.
- Files: `physsynth/core/bore.py`, oracles in `analysis/modal.py`
  (`bore_resonance_frequencies`, `discrete_bore_eigenfrequency`), helpers `make_bore` +
  `bore_low_eigenfrequencies` (`tests/helpers.py`), `tests/test_bore_{energy,modal,stability}.py`,
  `scripts/diagnose_bore.py` (energy trace, odd/even bars, standing-wave pressure shapes,
  convergence). Air constants `RHO0_AIR=1.2041`, `C0_AIR=343.0` reused from radiation. Gates: drift
  1.85e-14, worst partial 0.0000 cents, convergence order 2.00 (O(h²) dispersion at λ<1), odd/even
  5.6e5×. Bore auto-covered by the portability allowlist sweep (numpy+scipy only).

**BATCH 2 DONE & GREEN (2026-07-10): radiating bell — inline `"radiating"` End token + `R_bell`,
21 tests, full suite 615.** Reused the `RadiatedBody` pattern ([[radiation-state]]) at the *math*
level, NOT as a wrapper class — **inline boundary, per the "not a new class" idiom** (VKPlate-free /
StringPlateBridge-free). **Decisive reason it MUST be inline (advisor):** `Bore.step()` computes
`U^{n+3/2}` from `p^{n+1}`, so a post-hoc wrapper correction to `p_N` would leave the stored
`U_{N-1}` derived from the *uncorrected* `p̃_N` → broken telescoping. Inline solves `p_next[N]`
*before* `_momentum`, so consistency is automatic.
- **The load = a rank-1, unconditionally-passive resistive dashpot on the terminating half-cell
  node.** Bell resistance `R_bell` (acoustic Pa·s/m³) relates end pressure to outgoing volume
  velocity, `p=R·U_out`, evaluated at the **CENTERED** end pressure `p̄=(p^{n+1}+p^n)/2` (implicit —
  the VK/bow/radiation-load lesson). The un-pinned `p_next[end]` from `_divergence` IS already the
  rigid force-free half-cell step `p_rigid`; the centered resistor turns it into a **1×1 solve**
  `p^{n+1}=(a·p_rigid − b·p^n)/(a+b)`, `a=C_end/k`, `b=1/(2R)`. `U_out=b(p^{n+1}+p^n)`. `a+b>0`
  ∀R>0 → never singular → **unconditionally passive, no guard** beyond interior CFL.
- **Energy identity (money test):** telescopes to `E^{n+1}−E^n = −k·R·U_out² ≤ 0`; book
  `radiated_energy += k·R·U_out²` → **`energy()` returns `acoustic_energy()+radiated_energy`
  conserved** (folded in like RadiatedBody so `res.energy_drift` "just works"). Drift **1.0e-14**.
  One `_radiate_node` formula serves either end (dissipation is `R·U_out²`, sign-agnostic) → a
  **left-end sign bug only shows as drift** → the `("radiating","radiating")` + each-single-end
  identity test is what catches it. No radiating end → `radiated_energy=0`, `energy()` bit-identical.
- **Independent oracle with TEETH (advisor — the energy identity is self-consistent, doesn't prove R
  is physical):** pressure reflection coeff `r=(R−Z0)/(R+Z0)`, `Z0=ρ0c0/S` (`bore.Z0`). A stationary
  Gaussian at center splits (λ=1, exact) into two E0/2 halves; one right-bounce sheds
  `½(1−r²)=½·4RZ0/(R+Z0)²` of E0 — **measured lands on oracle to 1.6e-15 across R/Z0∈[0.03,30]**,
  peaking at matched `R=Z0` (r=0, **anechoic**, absorbs everything). This is the batch-2 signature.
- **R interpolates the two ideal ends:** `R→0` = open pressure-release (odd-harmonic clarinet, but
  NOT bit-identical to the `"open"` pin — it pins `p̄` not `p`, don't claim it); `R→∞` = rigid closed
  wall (closed-closed full series, even harmonics reappear). Interior odd/even ratio 1.5e5 (open-ish)
  vs <1 (rigid) — 12 orders apart.
- **Overdamped-end Nyquist artifact (advisor #1, the one that bites):** at physical `R≪Z0`
  (`R/Z0≈3e-4`, `b/a` huge) the update is `p_end^{n+1}≈−p_end^n` → the *raw* end node carries a
  marginal Nyquist ripple. It **cancels exactly in `U_out∝(p^{n+1}+p^n)`** → radiated energy, the
  far-field read-out, and interior pickups are all clean. **Read the far field from `U_out`, put
  spectral pickups in the interior (node 1).** Verified: far-field `pressure()` Nyquist frac 1.8e-4
  vs raw node 3.6e-3 (20× cleaner).
- **`pressure()` far-field read-out** = bell volume acceleration `dU_out/dt` → composes with batch-1
  `AirRadiation.radiate(bore)` (that's "how sound leaves"). Physical `R` sized from
  `piston_radiation_resistance(ω1, radius)` ≈ 649 (R/Z0≈3e-4, lightly radiating).
- Files: `physsynth/core/bore.py` (`"radiating"` token, `R_bell`, `acoustic_energy()`, `pressure()`,
  `Z0`, `radiated_energy`, `_apply_radiating_ends`/`_radiate_node`), `make_radiating_bore`
  (`tests/helpers.py`), `tests/test_bore_radiation.py` (21), `scripts/diagnose_bore_bell.py` (energy
  channel, reflection-oracle sweep, R-interpolation spectra, clean-far-field vs raw-node).

**BATCH 3 (dynamic reed) — NOT STARTED**; bow-shaped implicit scalar coupling
([[bow-state]]), but note the reed injects flow into node 0's continuity and **node 0 is the
half-cell → the injection prefactor divides by `C₀=(h/2)S₀/ρc²`, not the interior h-cell** (advisor
— the area-weight/paste gotcha the [[free-plate-bridge-state]] already flags: "W already carries
area weight"). Reed is a 2nd coupled DOF (or local 2×2). Evaluate Bernoulli flow at the *centered*
pressure (bow's O(k²)-drift lesson), oversample, bracketed-root fallback for the beating-reed clamp.

Per [[commit-push-at-batch-end]]: committed + pushed at batch-1 end.
