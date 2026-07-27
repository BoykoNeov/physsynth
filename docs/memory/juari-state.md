---
name: juari-state
description: "Juari / tanpura cotton thread (viewer batch 14) — a SINGLE-NODE point contact, the jawari's position-SELECTIVE complement; headline = the TUNING CURVE (buzz vs thread position), NOT more buzz; zero core; the buzz is a SETTLED quantity so the sweep is a canonical fixed duration decoupled from audio"
metadata:
  node_type: memory
  type: project
  originSessionId: f2cc751e-a184-4af7-9f14-a31e755c18c1
  modified: 2026-07-21T04:09:17.713Z
---

**Juari / tanpura cotton thread** — web viewer **batch 14** (built & browser-verified 2026-07-21).
The last unshown member of the barrier family and the **jawari's complement, not a preset**. A
single thread of cotton laid on the bridge = **ONE barrier node**, a *point* contact, config of
model #8 ([[barrier-collision-state]] `BarrierString`); **zero core code**, all in `web/serialize.py`
(`_build_juari` / `_juari_tuning` / `_juari_signature_block` / `_build_payload_juari`) + `app.js`
(`drawJuariViz` / `drawJuari` / `drawJuariTuning` / `drawJuariSpectrum`). Contrast with
[[jawari-state]] (a CURVED distributed bridge).

**The whole batch was gated on a separability PROBE before any wiring** (advisor: "the probe is the
whole decision"). The redundancy trap is real: a thread node *inside* the span the curved jawari
already wraps is a no-op → the honest framing is the thread **ALONE**, not thread-on-curve. Probe
(3 passes, `M:\claud_projects\temp\juari_probe\`) confirmed a distinct model.

**THE HEADLINE IS POSITION-SELECTIVITY — the TUNING CURVE — not "more buzz."** A point contact
clips the partials with an **antinode** at the thread and spares those with a **node** there, so
sliding the thread reshapes *which* partials buzz — the tanpura's jvari-tuning gesture, and the
clean separator from the jawari's **position-INDEPENDENT** broadband shimmer. Measured (jawari's
string params: loss ON σ₀=0.5, mode-1 IC, N=100, λ=0.4, K=2e6, α=1.5, amp 8 mm, **generic pickup
0.83 L** not 0.5 — at 0.5 the even partials sit on the pickup's own node and flatten the curve):
clean late-centroid **100 Hz = f₁**; juari sweeps **1.0× (nut) → ~3.0× sweet spot (x≈0.10–0.11,
node 10–11) → ~1.7× mid-string**; the sweet spot **rivals the whole curved jawari bridge (3.45×,
FLAT)** — a well-placed point thread is as bright as the distributed wrap but position-tunable.
Mechanism (probe 2): the thread **suppresses the fundamental** where it has an antinode there (node
50: f₁ −18.8 dB) and pumps energy into the **low-order** antinode partials — clean at n=3/5,
**washing out at high n** as nonlinear clipping spreads energy broadly (NOT idealized odd/even).

**THE money design decision, measured beyond the plan: the buzz is a SETTLED quantity.** The
spectrum keeps cascading upward for ~20 fundamental periods, so a short run peaks at the WRONG node
(0.10 s → node 15, 0.24 s → 11, 0.35 s → 9). Hence the tuning-curve sweep runs at a **CANONICAL
fixed `JUARI_SWEEP_DUR`=0.24 s, DECOUPLED from the audio slider** — lengthening the sound must not
shift (or re-price) the map. Pinned by `test_juari_tuning_curve_is_decoupled_from_the_audio_length`.
A hidden `sweep_duration` override (never in the UI) lets tests run fast.

**Grid quantization is load-bearing and stated honestly:** `thread_position` snaps to the nearest
node, so the curve's resolution IS the node spacing `h = L/N` — coarsest exactly near the nut where
a real thread lives (~10 distinct positions in x ≤ 0.15 L). The curve is drawn as **discrete points
at node resolution**, not a smooth line.

**Correctness (measured on the SHIPPED config, not by analogy):** σ=0 lossless **drift 1.08e-12**
through 587 real contact steps; σ>0 **passivity** (−2.78e-17); the config-specific money oracle
**m=1 scalar-collapse** (vector == scalar `solve_contact`) **1.93e-15** at K=2e6/α=1.5 — a dedicated
core test `test_juari_thread_config_collapses_to_scalar_solver` (the generic single-node test uses
K=8e5). `decay_oracle` stays **TRUE** (a lossless elastic point contact dissipates none, so the 2σ₀
oracle survives). **Cost: the SWEEP is the driver** (~11 thread positions × the settled duration;
per-node *construction* dominates, not stepping) → ~37 s at the default; `JUARI_WORK_MAX`=240k.

**Panels:** animation (string over the thread peg, lighting up the frames it is *caught*) +
diagnostics split — LEFT the tuning curve (discrete points, clean-1× and jawari-flat reference
lines, sweet-spot ring, thread marker on the curve), RIGHT the band spectrum (thread vs clean, which
partials buzz). Tests: 12 web (`tests/test_web_backend.py`) + 1 core; headless-verifier case `juari`
in `scripts/verify_web_headless.py`. Param hygiene: `thread_position` is a NEW name; reuses
`bridge_stiffness` (α fixed 1.5 server-side). Below-signal → LABEL never FAIL (bow/jawari rule).

The barrier family is now fully surfaced. See [[web-viewer-state]], [[jawari-state]],
[[barrier-collision-state]], [[commit-push-at-batch-end]], [[respect-ruff-line-length]].
