---
name: body-bridge-state
description: "Body/radiation node built & green — Steps 1-3 (modal body + string-terminus bridge + radiation) AND Step 4 (grid plate #5 as a distributed body via driving-point coupling, exact Sherman-Morrison guard); explicit-linear-spring exactness, per-end string boundary"
metadata: 
  node_type: memory
  type: project
  originSessionId: ccb757d5-f7d0-401e-a75d-e796d32e0ba6
---

**Body/radiation node** (third stage of `exciter → resonator → body/radiation`, HANDOFF §3.2) —
advisor Steps 1–3 BUILT & GREEN (417 tests total; 2026-07-02). Human decisions: body = **modal set
first, then reuse connection for plate #5** (Step 4 follow-on, NOT built); bridge topology = **string
TERMINUS (guitar-like)**, deviating from advisor's interior-point MVP → the coupled end is a moving,
force-driven **free** end (not untouched stencil).

**Step 1 — `ModalBody` (`core/body.py`):** bank of damped SHOs `q̈ᵢ+2σᵢq̇ᵢ+ωᵢ²qᵢ=φᵢF/mᵢ`, same
explicit leapfrog + **cross-time potential `½mω²qⁿqⁿ⁻¹`** as the string (naive same-time drifts —
same warning). Modal CFL `ωᵢk<2`. Drift 6.8e-14; single lossless mode = **exact discrete cosine**
`sin(Ωk/2)=ωk/2` to 4e-17. Read-outs `bridge_displacement=Σφᵢqᵢ`, `pressure=Σaᵢq̈ᵢ`. 9 tests.

**Step 2 — `StringBodyBridge` (`core/connection.py`), the core deliverable:** fixed-left/free-right
string, free end loaded by linear spring `F=K·ηⁿ`, `η=u_N−w_b`. String end gets reaction `−β_s F`,
body `+β_i F`. **EXPLICIT spring is EXACT here** (advisor reconciled my tie: implicit was only for
the *nonlinear* VK quartic; a linear spring makes the whole system ONE linear leapfrog conserving
`H=½(δt⁻x)ᵀM δt⁻x + ½xⁿᵀS xⁿ⁻¹` exactly, `S=…+K aaᵀ`, `a=[+1(end),−φᵢ(modes)]`) — so
`E_string+E_body+E_conn` is a rigorous decomposition, NOT a seam; `E_conn=½Kηⁿηⁿ⁻¹`. **`E_string`
alone is NOT conserved once coupled — assert on the total.** Bridge mass weighting (advisor's #1 drift
risk): `β_s=2k²/(ρh)` (string end **h/2** half-cell), `β_i=k²φᵢ/mᵢ`. Total drift 3.9e-14; body
carries up to ~62 % of energy (real transfer). 16 tests.

**Stability — cheap guard is a FOOTGUN, use exact:** the 2-DOF `K(β_s+β_b)<4` is
NECESSARY-NOT-SUFFICIENT (system NaN'd at ~0.5×that bound — the rank-1 spring lifts the top coupled
eigenvalue above both parts' isolated maxima). Replaced with **exact guard**: assemble the coupled
leapfrog operator `A` matrix-free via `_apply_A` (reuses the SAME step stencils so it can't disagree
with the scheme; ~N+M≈104 DOF, dense `eigvals`), assert `k²λ_max(A)<4`. Drift holds machine-precision
right up to k²λ_max→4; rejects unstable K. **λ=1 trap** (advisor): string Nyquist sits marginally at
λ=1, spring pushes it unstable → run **λ<1** (0.9 fine). `cfl_2dof` kept only as a diagnostic attr.

**Step 3 — radiation:** `ModalBody.pressure()=Σaᵢ·_accel` where `_accel` = the **actual second
difference `(qⁿ⁺¹−2qⁿ+qⁿ⁻¹)/k²`** stored each step (captures the connection force) — NOT reconstructed
`−ω²q−2σq̇` (advisor catch: that omits the coupling term the moment the string drives the body).

**Infra:** `IdealString.boundary` now accepts a **`("fixed","free")` tuple** (per-end; backward-compat
— a bare string = both ends). `string_stiff`/`string_damped` are independent classes (own
`"supported"`), untouched. **K=0 → bit-identical** to uncoupled `IdealString`+`ModalBody`.

**Diagnostics:** `scripts/diagnose_body_bridge.py` — (1) energy-exchange **flat total over
string↔body** (drift 3.2e-14), (2) radiated-pressure spectrum = string partials **acceleration-tilted
(p∝ω²) + boosted near body modes** (honest: body modes off-harmonic → no pure formants), (3)
terminus fundamental **free c/4L≈50 → clamped c/2L≈100 Hz** as K stiffens.

**Radiation is verified UNDER COUPLING** (advisor catch): `test_pressure_includes_coupling_term`
warms up to F≠0 then asserts `_accel == −ω²q + φF/m` — the standalone F=0 test can't tell the fixed
form from naive `−ω²q`, so a revert would pass all 417 tests silently.

**Step 4 — `StringPlateBridge` (`core/connection.py`), plate #5 as a distributed body — BUILT &
GREEN (2026-07-02).** The advisor resolved the two open questions before any code: (a) **build the
grid plate, don't reduce to modes** — the memory already decided it; modal reduction is a dead end
(#5b free-edge and nonlinear plates have no closed-form modes), the **driving-point adapter** is the
reusable path; (b) architecture question was NOT for the human — already recorded. String terminus →
single plate **live node `drive_index`** (default off-symmetry `pickup_index_at(0.3Lx,0.4Ly)` so it
excites the low spectrum) via the same explicit spring `F=Kη^n`, `η=u_end−w_dp`.

**The one genuinely-new piece was coupling an explicit string+spring to the *implicit* plate.**
String reaction is post-step (`u_end −= β_s F`, exact for a linear leapfrog, β_s=2k²/ρh as before);
the plate's `+F` source must enter the θ-scheme **RHS before the solve** (a post-solve node
correction is invalid — the implicit `A`-solve couples all nodes) as `k²F/(ρ_s h²)` at `dp`
(`Plate.step` gained an optional `f_ext` live-force vector). **The θ-average touches only the elastic
term, so a time-n source contributes `kF·δ_{t·}w_dp` regardless of θ** — the three increments
(`E_str −kFδu_end`, `E_plate +kFδw_dp`, `E_conn +kFδη`) telescope to zero because `η=u_end−w_dp`, the
proven modal cancellation one distributed body up. `E_conn=½Kη^nη^{n-1}`; energy = string+plate+conn.

**Stability guard is EXACT and cleaner than the modal case (advisor):** don't reach for an
amplification-matrix ρ (fragile on the unit circle). Energy-positivity `G=M_aug−(k²/4)S ≻ 0` where
`M_aug` absorbs the plate's implicit θ-augmentation → on the plate block reduces to
`ρ_s h²[I+(θ−¼)k²κ²B] ⪰ 0` for θ≥¼ — that's **why** the implicit plate is unconditionally stable, so
**it only adds headroom; the sole destabiliser is the rank-1 spring** (string block PD for λ<1). So
**Sherman–Morrison** gives a closed form: `(k²/4)K[(G0_str⁻¹)_end,end + (G0_plate⁻¹)_dp,dp] < 1`, two
small solves at construction (string tridiag `G0_str=M_str−(k²/4)(T/h)DᵀD`, plate `G0_plate` SPD
splu). **Margin is exactly LINEAR in K** (rank-1) — a money test. Stored on `stability_margin`.

**Plate radiation read-out (advisor catch, same as Step 3):** `Plate` gained `_accel`=actual second
difference stored each step + `pressure()=h²·Σ _accel` (monopole volume accel). Reconstructing
`−κ²B u` would drop the coupling force the instant the string drives the plate; the F=0 standalone
test can't tell → `test_pressure_includes_coupling_term` compares the forced step's `_accel` against
an identical unforced plate: the difference must equal `A⁻¹ s_F` exactly.

Build order (advisor): coupling+energy+**K=0 bit-identity** first (catch sign/timing before the
guard), guard last. Everything passed first try. Gates: **drift 1.7e-13**, plate carries **86 %** of
energy, passivity exact (0 positive steps), K=0 bit-identical (string+plate), margin **linear in K**.
**Guard "EXACT" is divergence-VERIFIED, not just self-consistent** (advisor catch — the two obvious
guard tests are weak: "rejects its own ceiling+5 %" is tautological, "0.93× conserves" only rules out
grossly-too-tight; a ~5 % inflated ceiling from a rank-1 slip would pass both and NaN a user
mid-render, the modal footgun one class down). Fix = `test_ceiling_is_the_true_instability_onset`:
0.99×Kc conserves AND bumping `.K=1.05×Kc` after construction (β_s/f_ext K-independent → clean guard
bypass) **actually diverges to inf in 0.3 s** — brackets the physical onset tight around the SM
ceiling K≈13.97k. Also **λ<1 hard-required at construction** (string block G0_str singular at λ=1,
Nyquist trap) with a clean error, not a bare spsolve warning. `Plate.step(f_ext=…)` keeps all 231
plate tests bit-identical. `tests/test_plate_connection.py` (23), `scripts/diagnose_string_plate.py`
(energy exchange flat total, plate colours the sound = string partials accel-tilted + boosted near
discrete plate modes, terminus fundamental 54.8→98 Hz free-c/4L→clamped-c/2L as K stiffens).

**Step 5 — free-edge plate #5b as a body — DONE (2026-07-02), see [[free-plate-bridge-state]].**
`StringPlateBridge` was **generalized** (not subclassed) to `boundary="free"`; only the guard's plate
block changed (`ρ_s[W+(θ−¼)k²κ²K]`, no extra `h²`), everything else delegates to `Plate`'s existing
boundary branches. No rigid drift (point-force orthogonal to `w_dp=0` nullspace modes). **This closes
the body-coupling family (Steps 1–5).** See [[plate-state]], [[free-plate-state]],
[[milestone-1-state]], [[respect-ruff-line-length]], [[commit-push-at-batch-end]].
