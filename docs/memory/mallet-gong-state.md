---
name: mallet-gong-state
description: "Model #7g, mallet on a von Karman plate — the nested solve; the predicted 10–100x cost was 1.9–2.3x, the \"no closed-form derivative\" was false, and the predicted inner-tolerance floor does not exist"
metadata: 
  node_type: memory
  type: project
  originSessionId: 565f5b37-96ed-48e6-93f6-bfdd76d1ff6d
  modified: 2026-09-07T13:24:23.761Z
---

Model **#7g `MalletVKPlate`** — a mallet striking a nonlinear (von Kármán) plate. Shipped
2026-09-06. Core `crates/physsynth-core/src/mallet.rs` (gong section), bound in
`crates/physsynth-py/src/mallet.rs`, 12 native bars in `crates/physsynth-core/tests/mallet_gong.rs`,
22 in `tests/test_mallet_gong.py`, helpers `make_mallet_gong` / `gong_linear_twin`. Plan:
`M:\claud_projects\physical synthesis\docs\dev\mallet-gong-plan.md`. Closes the half of HANDOFF
§14.1 that [[mallet-plate-state]] deferred.

**The algorithm is a chord whose frozen tangent is the LINEAR plate's own `g_s`.** Superposition is
gone (the VK step is not affine in `f_ext`, so no influence column exists), but it fails *by a
little* — the coupling reaches the one-step response scaled by `k²` — so the linear column is an
excellent frozen tangent exactly where it is a useless exact solution. One outer iteration is
`u_eff = Psi(f_j) + g_s·f_j` fed to model #7's `solve_contact`, **unchanged**. Only `u_eff` is new
arithmetic.

**Three predictions in the previous batch's plan were wrong, and that is the batch's content:**

- **Cost: 1.9–2.3× a bare gong step, not 10–100×.** Denominator must be a bare `VkPlate` step from
  the *identical state*. Outer loop: mean 1.74–1.89, **max 2**. "Not exact" was read as "not
  useful"; the gap between those readings is two orders of magnitude of cost.
- **The outer tangent IS closed-form**: `dη/df = −([J⁻¹ influence]_node + g_h)`, one GMRES against
  `VkCoupledStep::jacobian_vector`. Shipped as an *instrument* (`vk_drive_point_tangent`), not a
  step. Contraction bound `|1 − g_exact/g|` = 2e-6 … 1e-2.
- **The predicted inner-tolerance floor does not exist.** Expected the outer residual to plateau at
  `couple_tol·‖w‖/(g·force_scale)` ≈ 9.7e-14. Measured: at `outer_tol=1e-16` it converges in ~2.6
  iterations, and **four orders of `couple_tol` (1e-9…1e-15) move the count in the 2nd decimal**.
  Reason: the inner error is a **bias, not noise** — the inner solve is cold-seeded and
  deterministic, so the chord converges to the perturbed map's own fixed point. A floor would need
  `Psi` discontinuous in `f`; sweeping the trial force by parts in 1e12 leaves the inner sweep count
  pinned at 5. (Contrast [[tension-string-state]], where a root-find *did* branch on an iteration
  count.)

**The mallet enters the contraction ONLY through `g`, asserted bitwise.** `|g − g_exact|` is
`7.2757e-10` — the *same double* — for five masses spanning 200×. So the mass dependence is exactly
`1/g` and it **saturates** (`g→g_s` as `M→∞`): heavier is worse by a bounded 1.64×, not by an
amount that grows. The first measurement appeared to show 60× growth with mass; that was the
*amplitude* (a heavy head delivers a bigger peak force). **Hold the plate state fixed to separate a
mechanism from a correlation.** Return `response` separately from `g_exact`: recovering it as
`g_exact − g_h` is not the identity over doubles when they differ by four orders, and a bitwise
claim about a plate-only quantity must not route through a mallet-sized addition.

**`nonlinear=False` degenerates EXACTLY to #7p, and `n_outer == 1` is the structural anchor** — an
affine plate makes `u_eff = u_free` whatever `f_j` was. A sign error in `f_ext = −f·e_node` cannot
survive it. Agreement 2.08e-14 and **asserted NOT bitwise** (RHS-before-solve vs column-after-solve).
A miss is bit-identical to a bare gong, structurally: `force == 0.0` short-circuits to the
force-free advance rather than passing a zero force *vector* (which would add `+0.0` and flip `−0.0`).

**Conservation 2.7e-12 … 6.7e-12 over 2000 steps** (contract 1e-10), passivity 1.2e-15/4.8e-15.
Assert the **membrane share** (1.8–4.1%), never the velocity — otherwise it re-tests the linear
θ-scheme. The gong helper defaults to a **50 g** head, not the shared 20 g: 20 g at 3 m/s leaves the
membrane at 0.6%.

**Tightening a solver tolerance UNCOVERED a read-out defect.** At the first `outer_tol=1e-12` the
free branch measured *no worse* than the supported one — the nested solve's own error was larger
than [[mallet-plate-state]]'s quadratic rigid-drift cancellation error. At 1e-14 the law is plain
(`drift/rigid²` = 7.3e-8/6.9e-8/7.0e-8) and the supported control is 550× better.

**Shipped `outer_tol = 1e-13`** — one decade looser than the 1e-14 knee where drift stops improving
(that limit is the *energy read-out's* reduction rounding, ~6e-13, not the solver) — because 1e-13
is the tightest setting at which **no step stalls**, and a config that stalls once per run makes
every "did it converge" assertion probabilistic. A **stagnation exit** (`!(resid < prev)`, same
shape as `solve_contact`'s) ships anyway: one in-contact step in ~900 stops contracting near 5e-14,
and without the exit it spent all 20 outer iterations.

**In a nested solve the outer solver reports the INNER one's failure in its own words** — and those
words lie. A non-converged Picard makes `solve_contact` find no root, whose message says that is
"impossible for the monotone convex-potential force". It is. So `VkContactError::Contact` carries
`inner_converged` and the message names the solver that actually gave up.

**Physics payoff, and the control is NOT the identity this first claimed.** At `alpha = 1` the
exciter's *physics* is homogeneous of degree one, so a linear plate's departure from an exactly
scaled response reads **0.000e+00** on Windows — and **3.52e-13** on every Linux CI runner, which
took nineteen runs to notice because it was asserted `== 0.0`. An exact power-of-two loud/quiet
ratio is **necessary and not sufficient**: the *solver* carries two ABSOLUTE scales (the 0/0
Taylor-branch threshold, the bracketed root find's exit), so `ratio=8` and `strike_velocity=1.5`
each read 4.30e-13 on the machine where this pair reads 0.0. Bar is now tier-1 `1e-10`; the claim
was always the SEPARATION, not the zero. The gong's is **2.12**.
So #7p's "the felt exponent is the only source of dynamic timbre" is a statement about a *linear
resonator*. A power-weighted spectral centroid splits the same pair 450:1 (felt +0.16%, plate +74%
over a 16× dynamic range) where the scaled-response test splits it under 2:1 — two detectors, two
very different splits ([[air-box-state]]'s rule).

Not carried over from #7p: **rectangles only** (`VkParams` hardcodes the outline), **no
`pressure()`** (`VKPlate` has no accel field). **The inner Picard is not the limit at audio rates**
— zero non-convergent inner solves up to `w/e = 7.8`; the wall needs `fs = 8 kHz`. **Newton buys
nothing here** (same deflection to 11 figures, **1.47-1.72x** the solves across three masses) — its advantage is at pluck amplitudes
([[vk-newton-batch-state]]). Does **not** compose with `RoomLoadedVKPlate` (that seam replaces the
plate's `solve`; this model calls `vk_step` directly).

Scar: `cargo clippy --workspace --all-targets -- -D warnings` was **already red on `main`** from the
#7p batch (`needless_range_loop` in `plate_inject_u`/`_accel`). Run the Rust gates before assuming a
red one is yours.
