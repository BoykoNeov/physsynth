---
name: string-vk-bridge-state
description: "StringVKPlateBridge (string on the nonlinear plate) — the exact guard stays SUFFICIENT but the failure mode MIGRATES to Picard non-convergence; headline = departure from the plate's own linear self, order 2 off a bit-exact zero"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7df21e04-fbde-4b74-96fd-b2b30a47443b
  modified: 2026-08-17T10:13:18.736Z
---

`StringVKPlateBridge` — a string terminated on the von Kármán plate (model #6) — **SHIPPED
2026-08-17**, the last item `HANDOFF.md` §12H had left out. Plan + post-build record:
`docs/dev/string-vk-plate-bridge-plan.md`. Sits between [[body-bridge-state]] /
[[free-plate-bridge-state]] (the linear bridges it copies) and [[von-karman-plate-state]] (the body).

## The three named blockers, and only one resolved as framed

* **`plate.step(f_ext=...)`** — added to `VKPlate`. The force is **sweep-invariant** (`F = K η^n`
  depends only on time-`n` state), so it goes into the RHS **once, OUTSIDE the Picard loop**. That
  is why this batch needed **no `solve()` hook**, unlike [[air-box-state]] batch 6, whose room load
  is linear in the *unknown* `w^{n+1}` and had to fold into `A`. Simpler than its predecessor —
  worth checking before assuming a nonlinear coupling needs the loop.
* **`plate.rho`** → `rho_s`. The per-node force mass is now a shared `VKPlate.force_denominator`
  that airbox's `_VKPlateSurface` **reads**. The sharing paid immediately: falsifying it turns **46**
  tests red across *both* consumers, where a duplicated expression would have reddened one.
* **`plate.pressure()`** — **REFUSED, not supplied.** Batch 6 measured the compact monopole at 3e-7
  of the truth and pointing the *wrong way* for a cymbal; adding it back to satisfy a protocol would
  re-open a retired read-out. The absence is **asserted by a test** so a later batch argues with a
  test, not a comment. Checked first: nothing calls `.pressure()` on a body generically.

## The stability answer — and §12H's question was stale

§12H asked what a *linear 2-DOF* guard means here. That estimate is `StringBodyBridge`'s footgun;
`StringPlateBridge` has used the **exact Sherman–Morrison** margin for two batches. The sharper
question, answered: **the exact linear margin stays SUFFICIENT.** The conserved total is
`E_lin + H_mem + E_conn`, and `H_mem = ½(H(F^{n+1}) + H(F^n))` is a **sum of two squared norms with
no cross-time term** — unlike the θ-weighted bending potential, which carries an indefinite one — so
it cannot subtract, and a PD linear form stays coercive at any amplitude. Verified at 95% of the
ceiling under a strongly nonlinear run.

**But sufficient ≠ enough. The failure mode MIGRATES.** Conservation holds only *at* the Picard
fixed point, so a configuration the guard **passes** can still die by non-convergence (first failure
at step 182 at 90% of the margin) — a statement about a quadratic form cannot see a statement about
a fixed point. Hence the bridge surfaces `converged`/`n_iters`/`last_residual`, read **per step**.

## The headline, and the one that died on the shipping rig

**Claimed:** departure from the *same plate's* linear self,
`max_t‖u_nl − u_lin‖_∞ / max_t‖u_lin‖_∞`, which is identically **`0.0`** for a linear body (doubling
the pluck doubles a leapfrog and an LU back-substitution *exactly* — a machine-precision zero with
no oracle) and grows at **second order** in the pluck: 1.99/1.94 supported, 1.97/1.87 free, on both
rigs. **Saturates near 0.81** once the trajectories decorrelate, so only the small-amplitude orders
are claimed.

**DIED:** the energy-share magnitude. On the probe rig it gave +21.1% with orders 1.99/1.97/1.84;
on the shipping rig it goes non-monotone and **changes sign**, because **peak share is a BOUNDED
observable** and at 82% it has no headroom left. Its *control* half survives and is lovely (a linear
body's share is amplitude-invariant to machine precision, since every energy scales as amplitude²
and a ratio does not scale at all) — but neither size nor sign is claimed. Fourth time in this
family: ratios survive, magnitudes do not. Also: peak share is reached **early**, so it reports the
**drive point's** impedance, not the whole plate's (which is why both boundaries agree to 9 digits).

## Traps and rig facts a later batch needs

* **The cost claim inverts.** Wall-clock runs the *right* way (no room, no 3-D CFL), but the
  coupling's difficulty goes like `k²/h⁴`, so **shrinking the PLATE breaks the fixed point**: 40 cm
  converges to `w = 9e` in ≤13 sweeps, 8 cm caps out by `w = 6e` (drift 2.8e-2). Consequence:
  `f11 ≈ 3 Hz`, **below the audio band**, and **audio-band + string-drivable + Picard-convergent
  cannot all hold** at this sample rate — audio modes need ~7 cm (won't converge) and a thicker
  plate costs ~`e⁵` in energy to reach `w ≈ e`. **Want the gong impression? Use a MALLET, not a
  bigger budget.** [[air-box-state]] batch 6's wall, reached by geometry instead of grid.
* **A pure REASSOCIATION** of the `f_ext` term (`k²·f/d` → `(k²/d)·f`) is caught, but only by **2 of
  4** parametrisations — so parametrise a bit-identity regression over boundary **and** loss, never
  one case. (Batch 6 saw 5/8.)
* **Two things called `rho`.** The test helper first took one `rho` passthrough; `VKPlate` calls its
  *volumetric* density `rho` and the string family calls the *string's linear* density `rho`, so one
  argument retuned the wrong object. It failed loudly only by luck (the margin hit 204 and the
  constructor refused). Split into `rho` / `rho_plate`. The batch's own `rho_v`/`rho_s` trap one
  level up.
* Guard margins are **identical** for `supported` and `free` here (0.2139) — the drive point is an
  interior node, same lumped mass either way.
* 30 tests, 19 s. Picard sweeps 6/8/10/13 at `w/e` = 0.8/1.5/3.4/9.0.

## Still out

A **viewer batch** for this (Phase D's rule is satisfied: new model *and* new claim). See
[[web-viewer-state]].

**The room arm is DONE** — [[string-vk-room-chain-state]], shipped 2026-08-17 — and it retired two
things written here. The "third fixed point (spring, Picard, room load)" **does not exist**: there
is one, because the spring force and the room's terms are both sweep-invariant, so bare and loaded
bridges take the same sweep count. And this file's closing line — *want the gong impression, use a
MALLET* — is upgraded from assertion to **mechanism**, and sharpened: it is **not a budget question
at all**. A string cannot deliver the *shape* at any budget unless its band reaches the plate's,
because a point force feeds the free plate's rigid nullspace and rigid motion stretches nothing.
`w/e` is not an amplitude when the drive is a point force. See [[air-box-state]].
