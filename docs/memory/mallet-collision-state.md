---
name: mallet-collision-state
description: "Model #7 = mallet->membrane collision (core/mallet.py) — the FIRST contact/collision model (HANDOFF §12.B), a lumped mass in one-sided nonlinear felt contact striking a drumhead. Built & green (673 tests). Energy-CONSERVING discrete-gradient force (Chatziioannou-van Walstijn) with the 0/0 Taylor branch; explicit membrane => LOCAL admittance no A^-1; standalone MalletWall closed-form oracle; hysteresis=passivity. Opens the collisions family."
metadata: 
  node_type: memory
  type: project
  originSessionId: 20acc75a-c906-4641-9489-c59c9ccd8162
---

**MODEL #7 = mallet→membrane collision, BUILT & GREEN (2026-07-10): `core/mallet.py`.** The FIRST
model of the **nonlinear-contact family** (HANDOFF §12.B — string-fret / hammer/mallet / snares /
beating reeds). A new exciter class: a **lumped mass in one-sided nonlinear contact** with a
resonator = a soft **mallet striking a drumhead** ([[membrane-state]], timpani/tom). Human decisions
(2026-07-10): target = **membrane** (not string), and **include hysteresis this batch**. Plan =
`docs/dev/hammer-collision-plan.md`. Full suite **673** green, ruff clean.

**Why new vs the bow [[bow-state]] (closest analog).** Bow = *memoryless* → correctness is an energy
*balance*. The mallet is a **mass with state that STORES energy** (KE `½M v_H²` + felt PE `φ(η)`), so
the money test flips back to strict **CONSERVATION**: `H = E_membrane + ½M(δ_t- z_H)² +
½(φ(η^n)+φ(η^{n-1})) = const` (lossless, elastic). Two more new things: a genuine **contact
potential** whose force is its *gradient* (the new algebraic object, VK-bracket analog), and a
**second δ_tt-marched DOF** (the mass) centered consistently with the contact PE.

**Physics / sign convention.** `z_H` mallet, `u_S` head surface, common +z, mallet from above.
Penetration **`η = u_S − z_H`** (>0 = contact). One-sided felt `φ(η)=K/(α+1)[η]₊^(α+1)`,
`φ'(η)=K[η]₊^α`. Force on mallet `+φ'` (up), on head `−φ'` (down). `M z_H''=+f`, head node gets `−f`.
Continuous `dE/dt = −(membrane loss) − λ_h[η]₊^α(η̇)² ≤ 0`.

**THE crux = energy-conserving DISCRETE GRADIENT force (Chatziioannou–van Walstijn / Bilbao), NOT
`φ'` at a point** (point-eval drifts O(k²) — the VK/bow lesson):
`f = (φ(η^{n+1})−φ(η^{n-1}))/(η^{n+1}−η^{n-1})` → telescopes `f·δ_t·η = δ_t·φ` EXACTLY. **The #1 trap
= its removable 0/0** when `η^{n+1}≈η^{n-1}` (stick/grazing) → **branch to Taylor `φ'(½(η⁺+η⁻))` below
`η_tol` (1e-12)** or it NaNs in quiet regions. Advisor: use the discrete gradient + bow's
Newton/brentq, NOT ψ-quadratization.

**Hysteresis (Hunt–Crossley/Stulov) = passivity ONLY, not conservation.** `f_hyst = λ_h⟦η⟧₊^α·δ_t·η`,
`⟦η⟧=½(η⁺+η⁻)` → dissipation `f_hyst·δ_t·η = λ_h⟦η⟧₊^α(δ_t·η)² ≥ 0` sign-definite (removes energy
regardless of loading/unloading sign). `λ_h=0` recovers the conservative scheme. Build elastic-first,
assert conservation; hysteretic asserts passivity + restitution<1.

**Explicit membrane ⇒ coupling SIMPLER than the bow (no A^-1).** Membrane is EXPLICIT
(`u^{n+1}=(2u^n−(1−σk)u^{n-1}+c²k²Lu^n)/(1+σk)`), so a node force hits ONLY that node next step → the
driving-point admittance is the **bare local nodal mass** `g_s = k²/(ρ h²(1+σk))` — no banded solve.
Mallet `g_H = k²/M`. **ZERO edits to `membrane.py`**: force-free `mem.step()`, read `u_i,free`, solve
scalar, then `mem.u[i] -= g_s·f` (exact, mirrors bow's `string.u += ...`). Plate/bar follow-on: swap
`g_s` for the plate's `Plate.step(f_ext)` driving-point admittance ([[body-bridge-state]] machinery).

**Scalar collapse = the bow shape, but EASIER (unique root, no multivalued branch).** Both DOFs linear
in `f` → `η^{n+1} = η_free − g·f(η^{n+1})`, `g=g_s+g_H`. Residual `r(η)=η−η_free+g·f(η)` is **monotone
increasing** (convex potential ⇒ `f` non-decreasing) → single root. Safeguarded Newton (continuation-
seeded) + guaranteed brentq-scan fallback (nearest seed). `f` applied **exactly**.

**Money-test STRUCTURE differs from the bow (advisor).** Bow balance was residual-INdependent (true-
force trick). Here **conservation REQUIRES the applied f == the discrete gradient**, so a loose solve
drifts **∝ newton_tol** — a real self-cert (VK's drift∝couple_tol analog): tight-tol drift ~1e-12,
loosening 1e-14→1e-6 grows drift ×100+. Keep newton_tol tight (default 1e-14). PE in `energy()` MUST
be the **two-time-average** `½(φ(η^n)+φ(η^{n-1}))` (the form that telescopes) — NOT `φ(η^n)` (plan
draft was wrong; the derivation corrected it).

**Two-tier validation (VK bracket-before-loop culture).** (1) **Standalone `MalletWall`** (rigid wall,
`g_s=0`, reuses `solve_contact` verbatim) = the closed-form oracle: contact time `π√(M/K)` + **exact
velocity reversal (restitution 1)** at α=1,λ_h=0 (measured to 0.5% / 1e-9); ~190 steps/contact at the
helper defaults. The string/membrane carries energy away → NO analytic contact time coupled, so the
oracle MUST live at the wall. (2) **Coupled** = conservation `H=const` ~1e-12 at an amplitude where
the head takes ~65% of the strike energy (bracket bug can't hide — advisor's large-amp caveat), miss
= bit-identical to bare membrane (K=0 analog), circle drumhead conserves too (energy⊥geometry).

**Signatures** (`test_mallet_signature`): strike is broadband (≥3 analytic modes via projection, NOT
FFT peak-count — a soft felt low-passes so the FFT shows mostly the fundamental); **harder felt (↑K)
→ shorter contact + higher spectral centroid**; the mallet **BOUNCES** (velocity flips −→+, single
contact episode — elastic felt returns most energy to the mallet, so the head's *instantaneous* post-
separation energy is small, measure PEAK energy); **(2,1) mode NULLED by a centre strike** of a square
head (sin(2πx)=0 at x=L/2 node line) but excited off-centre — the mode-comb.

**FINDING — a point mass is an INEFFICIENT membrane exciter (physics, not a bug; advisor-confirmed).**
Coupled **elastic restitution ≈ 1.000** (really 0.99995 = √(1−1e-4), display rounding) → the mallet
bounces off carrying ~all its energy back; the head retains only **~0.01 %** of the strike (rings
faintly but detectably — 3 modes). This is the **coupled analog of the wall oracle's exact
restitution 1**, and it *emerged* un-enforced → reassuring, not suspicious. Mechanism = the **local
reactive near-field**: a point load on a 2D membrane has large (log-divergent) local compliance → it
forms a reactive dimple that relaxes and returns energy, radiating little into propagating modes.
**Timing check confirms it's near-field not boundary-bounce:** strike 0.3 m from the edge, c=200 →
round-trip 3 ms > ~2 ms contact, so no reflection returns during contact. Hysteresis lowers restitution
(0.84 @ λ_h=3e4) and raises retention only 0.01→0.09 % (directional, not meaningful). **Do NOT tune
params to ring louder** (rabbit hole — report like the reed threshold, don't assert); the honest fix =
a **spatially-extended / distributed contact footprint**, which is exactly the named follow-on → this
finding MOTIVATES that follow-on, it is not a defect in this one.

**Stability by construction** (non-negative PE + exact telescoping); real risk = under-resolving the
stiff contact `√(K/M)` → **warns if <8 steps/half-period** (energy still conserves even under-resolved
— the whole point of the energy method — it just aliases spectral content). Collision does NOT tighten
the membrane CFL λ≤1/√2 (bounded force applied exactly, not added stencil stiffness).

**API.** `MalletMembrane(membrane=, mass=, stiffness=K, alpha=2.3, hysteresis=λ_h, strike_x=,
strike_y=, strike_velocity=, gap=0, newton_tol=1e-14)`; Resonator protocol (`step/energy/state/k/
displacement_at`) + `.penetration/.contact_force/.in_contact/.mallet_velocity()/.fallbacks`.
`MalletWall(mass=, stiffness=, fs=, alpha=1, hysteresis=, strike_velocity=, gap=)`. Free funcs
(vector-ready, scalar=size-1): `contact_potential/contact_force_elastic/contact_stiffness/
contact_force_dg/contact_force_total/solve_contact`. Helpers `make_mallet`, `make_mallet_wall`.
`scripts/diagnose_mallet.py` (4 figs: contact pulse+trajectories, energy partition, hardness→spectrum,
hysteresis loop). Canonical rig N=40 λ=0.5 K=5e4 M=0.02 α=2.3 v0=3.

**NEXT collision follow-ons (non-goals this batch):** distributed contact (string-fret / rigid barrier
/ snare buzz) reuses the SAME `φ`/`[DG]` primitives on a per-node VECTOR η (promote to
`core/collision.py` when that 2nd consumer lands); mallet on plate/beam (config swap). Also latent:
reed BEATING (contact) deferred here per [[reed-state]]. Pre-existing E501 in `scripts/diagnose_reed.py`
(line 179) fixed in passing. Per [[commit-push-at-batch-end]]: commit + push at batch end.
