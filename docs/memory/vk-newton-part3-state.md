---
name: vk-newton-part3-state
description: "VK Newton Part 3 — the convergence map: a one-step map answers one-step questions, the outcome is NOT monotone (bisection returns a clean wrong boundary), and trap 3's answer is graded along curvature"
metadata:
  node_type: memory
  type: project
  originSessionId: 8d72a366-f87e-40e3-b252-a42a1f72c712
  modified: 2026-09-06T09:26:25.449Z
---

Part 3 of the von Kármán Newton batch (`docs/dev/vk-newton-plan.md` §13), landed 2026-09-06. Both
methods over `(w/e, strike curvature, fs)` — 480 points, one step each, Picard at cap 2000 so the
baseline is best-effort. Scripts in `M:\claud_projects\temp\vk-newton\` (`part3_map.py`,
`part3_runmap.py`, `part3_refine.py`, `part3_refine2.py`); two native bars appended to
`crates/physsynth-core/tests/plate.rs`. Follows [[vk-newton-part2-state]].

**The gate is met and is the smallest part of the result.** Newton's amplitude boundary is 2–4×
Picard's wherever the comparison is not censored at `w = 40e` (six of fifteen cells are). But
Newton is **cheaper only at the hard end**: 0.55–0.85× (i.e. Picard wins) across the easy half,
68× at the worst point. `couple_method` is a real choice, and that is now the measured argument for
Picard staying the default.

**A one-step map answers one-step questions — check what the affordability claim measured.** §9.4's
"the wall is decided on step zero" is true, and it is about **Picard's divergence from a struck rest
state**. It does not transfer to a Newton *run*: in five cells of fifteen the 300-step boundary sits
2–12 lower in `w/e`, first failing at step 2, 16, 24, 24 and 50. When a prior section makes a cheap
scan affordable, read what it actually measured before reading the scan as a claim about runs.

**The outcome is NOT monotone in the sweep axis, so a bisection returns a clean wrong answer.**
Supported plate, 3 cm strike, struck 0.12 m off-centre, 48 kHz: Newton is `expansive` at `w = 18e`
and **converges at `w = 20e`** to a residual of 7e-15. Report a boundary as two columns — largest
that converged, smallest that failed — so a gap is visible instead of averaged away. The same cell
shows the boundary is a **cost ramp**: 50 solves at 6e, 466 at 16e, 7266 at 20e, two orders of
Krylov work before the verdict moves at all.

**Energy cannot answer "is this root a plate", and the refined-`k` check needs a BENIGN CONTROL to
be readable.** Any root of the discrete-gradient equation conserves exactly, so 1e-13 drift is what
a root must do. Refining `k` at **fixed `h`** (only `fs` moves), identical IC in physical units,
matched physical time: two easy fixtures hold ratio **4.0** at every checkpoint — that is what makes
the rest interpretable, and without them a degraded ratio reads as a broken rig. Then the answer is
**graded along curvature**: an 8 cm strike at `w = 20e` (Picard-impossible) converges at 4.0 out to
667 µs — a real plate; a 3 cm strike at `w = 12e` never does and is **37% apart between 48 and
96 kHz at 83 µs** — a genuine root that is not a resolved plate. **The curvature axis that sets
Picard's wall also sets the resolution horizon, and Newton moves only the first.** So [[bow-state]]-
style broad excitation gains real territory and a narrow strike gains arithmetic. §3's refusal of an
audio-band string-drivable gong stands, now with a mechanism. Newton and Picard at 384 kHz land on
the same trajectory (8.7e-15 to 7.9e-10), so the root is not an iteration artefact.

**The line-search question could only be asked in Rust.** `n_line_search`, `gmres_products` and
`gmres_stalls` live on `VkNewtonReport` and deliberately not on `VkStep` ([[vk-newton-part2-state]]:
a field costs two edits), so no Python client sees them. Over fourteen Newton-only points: **zero**
halvings, **zero** stalls, worst 38 Krylov products and 7 iterations. §11.6 asked this on six
comfortable fixtures; fourteen hard ones give the same answer.

**How to apply:** state what a sweep holds fixed — this map holds **peak amplitude** on the width
axis (energy falls), which is §2.2's knob, and a fixed-energy sweep would draw a different boundary
that no column reveals. Newton's four pins are **reported, not tuned** (GMRES restart 30, 200
products, forcing 1e-4, Armijo 1e-4), per `plate.rs`'s own instruction. A free-boundary fixture must
be struck **off-centre** or it is not one ([[vk-newton-part1-state]] §10.4).
