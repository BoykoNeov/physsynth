---
name: vk-newton-part0-state
description: VK Newton plan Part 0 (the cap-vs-wall verdict) — landed 2026-09-06; the plan's "two residuals already exist" premise was FALSE, the exit ratio is NOT the within-solve rho, and the wall is decided on step ZERO
metadata:
  type: project
---

Part 0 of `M:\claud_projects\physical synthesis\docs\dev\vk-newton-plan.md` is done (results in its
§9). The nonlinear plate now reports `residual_ratio` (stored) and `couple_outcome`
(`converged`/`capped`/`expansive`/`unknown`, **derived on every read**, no setter) so that
`converged = false` stops conflating "the cap ran out" with "no cap would have helped".

Four things worth carrying forward:

- **The plan's premise was wrong about its own code.** §5 Part 0 said the model "already holds two
  consecutive residuals, so rho is one division". It does not — `last_residual` is overwritten every
  sweep. Struck through in the plan rather than deleted. *A plan written from reading a file can
  still be wrong about that file; re-read before quoting it as a fact.*
- **The exit ratio is a different quantity from the probe's rho, by construction.** `probe_rho.py`
  measures sweeps 1-8 of step 1; the new field measures the exit sweep of the last step, and §2.4's
  own finding is that the factor climbs *within* one solve. So the field not reproducing the probe's
  table is correct behaviour, not a transcription bug — and the exit value is the better answer to
  "would more sweeps help from here". Named `ratio`, never `rho`.
- **Classify `capped` POSITIVELY** — finite *and* below one. An overflowed step carries NaN, and
  `NaN >= 1.0` is false exactly as `NaN < 1.0` is, so a negative test files a blown-up plate under
  the one verdict a bigger cap fixes. A native bar asserts this on the classifier directly.
- **The wall is decided on step ZERO** — and the 300-step run does *not* show this, because the step
  at which a fixture's energy overflows **moves with the cap**. It took a direct one-step probe
  (`part0_first_step.py`): every divergent fixture is `expansive` on its first step at every cap.
  Part 3's map therefore costs **one step per point**, not a 300-step run. *A death at step 0 is not
  a verdict about step 0; measure the thing the cost estimate rests on.*
- **The binding does not delegate to the core model.** `VkPlate::step` and `PyVKPlate::step` each
  hold their own fields and copy the diagnostics in a hand-written block, so an edit to one cannot
  fail the other — the §45.9 `THETA_DEFAULT` fork's shape. Part 2's `couple_method` lands in both.

Gate met: no cap-limited fixture reported `expansive`, no divergent one reported `capped`, and the
three cap-limited ones come back green at a bigger cap (143/724/76 sweeps, drift ~4e-13) for 10-30%
wall clock. `couple_max_iter`'s default did **not** move. Next: Part 1, the Jacobian-vector product
asserted against finite differences before Newton uses it. See [[von-karman-plate-state]].
