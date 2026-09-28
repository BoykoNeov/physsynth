---
name: vk-newton-state
description: "The von Karman Picard wall (scientific-hurdles §5), chosen 2026-09-06 — probing falsified the doc's mechanism AND found a third of the wall is just the sweep cap"
metadata: 
  node_type: memory
  type: project
  originSessionId: 6fd48d83-d7ed-46c3-999d-d871b8ddfd89
  modified: 2026-09-05T22:11:06.758Z
---

The human picked `docs/dev/scientific-hurdles.md` **§5** (the nonlinear plate's iteration wall)
over §4 (the θ-scheme's damping suppression) on 2026-09-06. Plan lives at
`M:\claud_projects\physical synthesis\docs\dev\vk-newton-plan.md`; probe scripts at
`M:\claud_projects\temp\vk-newton\`. **Probed before any code, and the probe overturned two of
§5's three claims.**

- **The `k²/h⁴` scaling is wrong on the `h` half.** Refining the grid 4.3× at fixed plate size
  *and fixed absolute strike width* goes 12 → 22 sweeps and flattens. Shrinking the plate at
  **fixed h** hits the cap, and so does narrowing the *strike* alone at fixed plate, grid and peak
  amplitude. The driver is the **strain** (curvature of the deflection), not the grid spacing. The
  `k²` half survives — the sample-rate sweep confirms it. Both §5 and the comment at
  `tests/helpers.py:729-733` state the wrong mechanism; their *observations* stand.
- **A third of the wall is the 50-sweep cap, not divergence.** Three of six failing fixtures come
  back green on the energy bar at cap 400+ (143 / 76 / **724** sweeps, drift ~4e-13) at essentially
  no wall-clock cost, because the expensive steps are rare (mean sweeps barely move). So **the
  baseline Newton is measured against must be best-effort Picard** — comparing against cap-50
  would credit Newton with territory a constant recovers. Handled by ordering (Part 0 first), not
  by a warning. Do **not** change `couple_max_iter`'s default: it is public and reached from
  `tests/helpers.py`, the airbox VK surfaces and the viewer payloads.
- **In the divergent cases ρ is not constant — it climbs through 1** (0.243 → 1.011;
  0.334 → 1.114), stalling at 1e-3/1e-4 first. That is the recoverable kind. The unrecoverable
  ones sit at residual ≈ 1 from sweep 1 and go NaN. The **7 cm audio-band plate is in that set at
  w=6e but converges cleanly at w=e** (ρ 0.183, 8 sweeps) — so the ceiling is an *amplitude*, and
  the honest headline is that it **moves, measured**, not "audio-band gong solved".

Two structural facts about the scheme, neither in §5:

- The Picard map is **Richardson on a linear system**, so Picard *is* Newton with `J ≈ I`, and its
  convergence is the spectral radius of `c·A⁻¹K`.
- `J = I − c·A⁻¹K` is **already left-preconditioned** by the plate's own factorization (`J = I`
  exactly at zero amplitude), so §5's proposed `splu` preconditioner is present by construction and
  the batch needs no second factorization.

Related: [[von-karman-plate-state]], [[string-vk-bridge-state]], [[air-box-state]],
[[rust-migration-state]].
