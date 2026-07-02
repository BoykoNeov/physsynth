---
name: free-plate-bridge-state
description: "Step 5 of the body/radiation node — free-edge plate #5b (suspended cymbal/gong) as a distributed body; generalized StringPlateBridge to boundary='free' with only the guard block changing; no rigid drift"
metadata: 
  node_type: memory
  type: project
  originSessionId: 89e0f29e-c2d0-4667-ac2d-3426ff1ddbc5
---

**Step 5 — free-edge plate #5b as a distributed body — BUILT & GREEN (2026-07-02).** The suspended
cymbal/gong (FFFF free-edge Kirchhoff plate) is now a body node under the string terminus. **Did NOT
write a new class** — generalized [[body-bridge-state]]'s `StringPlateBridge` to accept
`boundary="free"` (advisor: mirrors the `VKPlate(boundary="free")` pattern). Crux insight (why it was
cheap): the bridge already delegates `step(f_ext)`, `energy()`, `pressure()` to `Plate`, which
**already branches internally on `boundary`**, so **the ONLY boundary-specific code is the
Sherman–Morrison guard's plate block** — `plate.py` needed **zero edits**.

**Guard block (the one real change).** General form is `G0 = M + (θ−¼)k²S` (mass M, stiffness S); it
reproduces both known-correct cases — SS plate at that θ, and the **string block is the θ=0 special
case** `M−¼k²S` (already in code). Free plate: `M=ρ_s W`, `S=ρ_s κ²K` →
**`G0_plate = ρ_s [W + (θ−¼)k²κ²K]`**. **Arithmetic gotcha (advisor):** `W` already carries the `h²`
area weight → **NO extra `h²` prefactor** (do NOT paste the SS branch's `ρ_s h²[I+…]`). PD for θ≥¼
because the **diagonal `W` anchors the `{1,x,y}` rigid-body nullspace of K** (at θ=¼ it's exactly
`ρ_s W`, diagonal-positive), so the guard stays a clean two-solve computation and the free plate
contributes only headroom — same as SS. Force injection unchanged: `plate.step` free branch already
does `k²f_ext/ρ` (the W-divide is handled by the A-solve → per-node mass `ρ_s W_ii`), so the
telescoping energy identity `E_plate += kFδ_{t·}w_dp` holds identically.

**No rigid drift (advisor sharpened my caveat — it's better than "drifts but conserves").** The
single-point spring only couples to plate modes with `w_dp≠0`. Surviving zero-modes lie in
`span{1,x,y}` with `v(dp)=0` (one constraint leaves 2 of 3) → the dp point-force is **orthogonal** to
them → **starting from rest they are NEVER excited**. No piston/tilt ramp; the `w_dp≠0` combinations
become bounded oscillations on the spring (plate bouncing on the bridge). Empirically: **same
~1e-13/1e-14 drift as the SS bridge** (a `q~t` ramp would grow KE unbounded → caught by the drift
bound). `test_no_rigid_body_drift` tracks the area-weighted mean displacement directly (bounded,
never ramps).

**Gates (all green).** Free bridge stability margin **identical to SS** (0.2148 — interior dp with
`W_dp=h²` and θ≈¼ gives near-identical driving-point admittance), K=0 drift 3.6e-15, K=3000 drift
1.7e-14, free plate carries up to **89%** of energy. **Divergence-VERIFIED guard** (Step-4 discipline,
the real money test, not "doesn't crash"): margin linear in K (probe self-cert) AND
`test_ceiling_is_the_true_instability_onset` — 0.99×Kc conserves, `.K=1.05×Kc` bypass **diverges by
step ~104** (E/E0=1.25e6). Ceiling K≈13.97k.

**Files.** `tests/test_free_plate_connection.py` (23 tests, full SS battery + rigid-drift check),
`scripts/diagnose_string_free_plate.py` (energy-exchange flat total drift 1.35e-13, cymbal colours
the sound = string partials accel-tilted + boosted near free-plate curved-Chladni modes, terminus
fundamental 60→118 Hz as K stiffens). **Physical distinction from the SS bridge (advisor
report-faithfully catch):** the floating free plate **OVERSHOOTS** the rigid-clamp c/2L=100 Hz
(lands ~118), because below its first elastic mode (~569 Hz here) it loads the string end as a
*reactive mass-spring* (its {1,x,y} rigid-body modes act mass-like), NOT a rigid anchor — unlike the
pinned SS plate (Step 4) which lands cleanly near c/2L≈98. Diagnostic label was corrected to say so
(was overstating "toward clamped c/2L, identical terminus physics"). `tests/helpers.py` gained `make_free_plate_bridge` (**`nu`
plumbed through** — re-enters for free edges, unlike SS). SS `test_plate_must_be_supported` flipped to
`test_free_plate_is_accepted` (old behavior deliberately removed). All 231 plate + 23 SS-bridge tests
untouched/bit-identical.

**This closes the body-coupling family** (Steps 1–5: modal body → string-terminus bridge → radiation
→ grid SS plate → free-edge plate). **Full suite 464 tests, all green.** See [[free-plate-state]],
[[plate-state]], [[body-bridge-state]], [[respect-ruff-line-length]], [[commit-push-at-batch-end]].
