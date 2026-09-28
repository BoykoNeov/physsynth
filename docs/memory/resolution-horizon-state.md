---
name: resolution-horizon-state
description: "Where each scheme stops being in tune, measured 2026-09-06 — TWO families with opposite signs (implicit errors compound into a hard floor, explicit ones CANCEL at the magic Courant number), and the 2-D cancellation is DIAGONAL-ONLY"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3c0c4a48-d743-45a8-a700-73c01ddab289
  modified: 2026-09-06T20:32:05.442Z
---

The human's follow-on to [[theta-loss-lock-state]], chosen 2026-09-06: the probe's real content was
that the **pitch** error bounds every accuracy claim this project makes about a mode, and no model
had that boundary written down. Now it does.
`M:\claud_projects\physical synthesis\docs\dev\resolution-horizon-plan.md`,
`M:\claud_projects\physical synthesis\tests\test_resolution_horizon.py` (32 tests, 4 s),
`M:\claud_projects\physical synthesis\tests\helpers.py` (`pitch_horizon`, `pitch_error_cents`,
`spatial_operator_horizon`). New `scientific-hurdles.md` **§15**.

**The horizon is not `min(time floor, space floor)` — that draft was wrong and the membrane data
caught it.** Whether the time and space errors add or cancel depends on their **signs**:

* **Implicit θ-scheme** (stiff/damped string, both plates, beam): time factor `1/√(1+θk²Q)` and
  spatial `sinc(u)` are **both flat**, so they **compound**. There is a hard floor no sample rate
  passes.
* **Explicit leapfrog** (ideal string, membrane): the time factor is **sharp** and cancels the
  spatial droop exactly at `λ = 1`. The ideal string at `λ=1` resolves **12× its own space floor**.

So **"refine the timestep" is right for one family and actively wrong for the other** — below
`λ=1` the ideal string goes 255 → 24 modes as `λ` falls to 0.5. That is the finding most likely to
change what someone does.

Four numbers worth remembering:

* **The space floor has a closed form with no `k`, `c`, `L` or `fs` in it**: `sin(u)/u =
  2^(−cents/1200)`, `m*/N = 2u/π` — **8.38% of the grid** at 5 cents. Stiffness lowers it (6.6% at
  κ=2, 5.9% at κ=8) *and keeps lowering it under refinement*, because a finer `h` admits higher `p²`
  where the quartic term dominates. This is the only bar in the file checkable from **outside** the
  measurement, and it is analytic rather than recorded.
* The canonical `λ=1` damped string at `N=256` resolves **eleven partials**. 64× the sample rate
  buys eight more and then stops dead.
* **The plate is space-limited at every `fs` the suite uses** — a 40× change of sample rate moves
  its horizon by ≤1 mode. At the `N=16` fixture the **fundamental is 9.6 cents flat**.
* **The membrane's CFL-ceiling cancellation is DIAGONAL-ONLY**: at `λ=1/√2`, 127 of 127 diagonal
  modes are exact and only 15 axial ones are. Reading the diagonal alone would have shipped "the
  membrane is in tune at the ceiling" — a claim about one mode family. *Check a second mode family
  before any 2-D dispersion headline.*

Three method points:

* **The primitive returns `(horizon, monotone)`, never just the integer.** "Leading prefix all in
  tune" and "highest mode that happens to be in tune" differ exactly when the curve is non-monotone,
  and this project already has one non-monotone boundary on the record.
* **Nothing is a frozen integer.** A horizon is a property of `(model, N, k, params)`; the tests
  assert *family behaviour* — which way it moves, that a floor exists and is reached, that the other
  family beats it — plus the one closed form.
* **The primitive stayed in `tests/helpers.py` deliberately.** A new public `analysis/` name trips
  `test_analysis_frozen.py`'s derived guard and **cannot satisfy it** (no Python left to freeze
  against). Widening a guard to admit the first thing that does not fit is how a guard stops meaning
  anything — promotion is its own batch, needing a native bar plus a stated guard amendment.

**Four inventory rows are explicitly NOT done, each with its reason**: staircased domains (circular
membrane, guitar outline) fail a cents comparison because the reference is a different *shape*; the
free plate's reference is a table, not a formula; the nonlinear family needs a *refinement* horizon
(one point exists in `vk-newton-plan.md` §13); the bore's question is the area function. **The named
follow-on** is converting the suites' hand-picked assertion bands (`[1..16]` and siblings) into
derived ones — deliberately not done here, because doing it alongside the primitive that computes
them would make any failure ambiguous.

Related: [[theta-loss-lock-state]], [[damped-string-state]], [[plate-state]], [[membrane-state]],
[[beam-state]], [[milestone-1-state]], [[vk-newton-batch-state]].
