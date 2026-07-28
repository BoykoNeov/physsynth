---
name: air-box-state
description: 3-D FDTD air box (HANDOFF §12H) batch 1 BUILT & GREEN, batch 2 (two-way room<->body port) PLANNED — the DISTRIBUTED air tier above the lumped port; the reward and the price both live at λ=1/√3; a flat energy is NOT a stability certificate there
metadata: 
  node_type: memory
  type: project
  originSessionId: 8506c3d9-c98e-4229-87cf-6160a3fd6c48
  modified: 2026-07-28T02:37:05.779Z
---

**The distributed air node — `physsynth/core/airbox.py` `AirBox`.** The human's pick at the
post-Phase-D fork (2026-07-28), taken right after the `R(ω)` arm landed. Plan
`docs/dev/air-box-plan.md`, built same day. The lumped tier is [[radiation-state]]; the 1-D ancestor
whose Yee cell this tensors up is [[bore-state]].

**The claim, in one line: the distributed air CONTAINS the lumped air as its free-field limit.**
Far from the walls the room reproduces `AirRadiation`'s own monopole law (`max|gain−1| = 6.6e-3` at
N=64); one cell thick it tracks the repo's `Bore` to 5.3e-14. Everything else is what a lumped port
*structurally cannot* do: room modes, travel time, comb filtering, more than one listener.

**Node-centered p was a deliberate choice, not a default.** Room acoustics usually goes
cell-centered; node-centered was picked so **the bore's boundary machinery transfers unchanged** —
a rigid wall is the `h/2` trapezoidal half-cell (no ghost stencil), an impedance wall is the bore's
radiating-end 1×1 collapse. Both already proven in this repo. It paid off exactly as hoped.

**ONE wall closure, THREE boundary types** — the reduction-ledger entry.
`p^{n+1} = (p_rigid − β p^n)/(1+β)`, `β = kρ₀c₀²/(2 Z w_wall)`. `Z=inf ⇒ β=0 ⇒` rigid
(bit-identical); `Z=0 ⇒` open, exactly 0.0. **Edge/corner nodes SUM ADMITTANCES** and the solve
stays 1×1 — no coupled solve anywhere in 3-D. Carry the open face as a boolean MASK, not an `inf`
in `β`, or NaN appears in the update.

**THE ORDERING DECISION, and what it dissolved.** The plan's trap §6.1 (measured on the prototype)
was that `E^n` must be snapshotted *mid-step*, and an off-by-one reads as a 2.2e-2 drift that looks
like a broken scheme. **Adopting `Bore`'s ordering — pressure, walls, then velocity — makes
`energy()` a pure function of stored state and the trap evaporates.** After `step()` the object
holds `p^{n+1}` with `u^{n+3/2}` and `u^{n+1/2}`, so the cross-time product is `E^{n+1}`. Two knock-
ons: `set_state` must mirror Bore's contract exactly (`u0` is `u^{-1/2}`, the ctor derives `u^{1/2}`)
or the Bore cross-check reads 1e-2 instead of 1e-14 and you hunt the scheme instead of the seam;
and **the plan's −1.5-sample read-out lag VANISHED** (measured `|lag| < 0.4 sa`). That offset was an
artefact of injecting the source *before* the velocity sub-step — a property of where in the step
the source lands, not of the read-out. A plan number can be ordering-dependent; re-measure it.

**λ = 1/√3 — THE REWARD AND THE PRICE ARE THE SAME NUMBER.** This is the batch's real finding and
it is not in the plan.
- **Reward:** a mode along the **grid diagonal** — `l/Nx = m/Ny = n/Nz`, which on a cube is
  `l=m=n`, and on the (9,7,6) room is the CORNER mode (9,7,6) — comes out at the **exact continuum
  frequency, 1e-16**. Nothing else is exact at any λ. Axial modes are never exact (>1e-3 always).
- **Price:** the corner mode's dispersion argument reaches exactly 1, `ω_d k = π`, and the
  amplification matrix goes **defective**. Broadband content grows **LINEARLY** — measured peak ×15
  by 1000 steps, ×88 by 6000, windowed peaks in a clean 1,2,3,4 progression. Not exponential.
- **AND THE ENERGY IDENTITY SURVIVES IT UNTOUCHED**, because the discrete energy is only positive
  **semi**-definite at the ceiling: potential and kinetic grow together with cancelling signs.
  ⇒ **A flat energy is NOT a stability certificate at λ_max.** The one place in this repo where the
  project's primary bug detector is blind. Construction allows the ceiling deliberately (textbook
  CFL, and the diagonal exactness is a real reason to want it); helpers default to `0.9/√3`.
- Corollary that also broke a first-draft test: dispersion does **not** simply worsen toward the
  ceiling. Spatial and temporal errors partly cancel, so "smaller λ is more accurate" — true in 1-D
  — is FALSE here. Assert that no λ is dispersionless, not that big λ is worse.

**The modal oracle is a TIER ABOVE the membrane's Bessel test.** The tensor cosine is an *exact*
eigenvector of the discrete Neumann Laplacian **including at the h/2 wall nodes** — a grid-aligned
rectangle staircases nothing. So it splits: (a) spatial, the eigen*value*, asserted over EVERY index
0..N in one step each (cheap, exhaustive); (b) temporal, the eigen*frequency*, tracked against
`cos(ω_d n k)·mode` for 500 steps (1e-14). A shape-only test passes with a wrong `ω_d`, and `ω_d` is
the whole point. The exact half-step-back velocity `u^{-1/2} = (k/(2ρ₀h))·diff(p⁰)` is **ω-free** —
the plausible continuum form leaves a 1e-4..1e-1 error that masquerades as scheme inaccuracy.

**Free field: `gain == 1` IS the 1/r law.** The closed form already carries `1/(4πr)`, so a
per-radius LSQ gain fit beats a log-log slope fit on dispersive pulse peaks — the naive peak-ratio
estimator drifts +5% where the fit sits within 0.7%. Two traps: (1) the window is **per-probe and
hard-edged** (`t < (L−r)/c₀`) because node-centered p makes a rigid wall a pressure ANTINODE — the
reflection arrives at full amplitude, and one global stop time turns the slope into −2.5; (2) let
the lag search go **negative** or it pins at its boundary. Measured `max|gain−1|`: 9.9e-2 (N=32) →
2.0e-2 (48) → 6.6e-3 (64). **Do NOT tighten this into a convergence order** — `r_min`/`r_max` move
with N, so the implied rate is not a property of the scheme.

**Scope: READ-OUT ONLY.** The source drives the room; the room does not push back. Deliberately the
same position `AirRadiation` held in radiation batch 1, and the natural next batch is the two-way
passive port — exactly the 1→2 pattern that module already ran. Also deferred: PML (a
locally-reacting `Z=ρ₀c₀` wall is matched at NORMAL INCIDENCE ONLY, so the free-field oracle leans
on windowing and never on absorption), HRTF/ambisonics, scattering objects / non-rectangular rooms,
viscothermal loss.

**`inject` takes VOLUME VELOCITY** because that is the continuity source term; the chain drives from
`ReactiveRadiatedBody.volume_velocity` (already public) so **zero edits** to `body.py` /
`connection.py` / `radiation.py` / `bore.py`. Do NOT integrate `Q''` internally to fake the
`_VolumeAccelerationSource` protocol — an accumulating integrator has an unrestored DC drift mode.
Booking is `k·pbar_src·q` with `pbar` taken POST wall-solve, so a source sitting on a wall books
correctly (there is a test for exactly that).

**Cost is a design constraint here for the first time in this repo.** At fs=44.1 kHz the CFL forces
`h ≥ √3c₀/fs ≈ 1.35 cm`, so 1 m³ is ~74³ nodes and one second of audio is ~4 minutes. Structural and
modal oracles are grid-size-independent and run on ~10×8×6 (560 nodes) where they are free; the
free-field pair tops out at N=64. Audio-rate rooms belong in the diagnose script, never the suite.

**Files:** `physsynth/core/airbox.py`; `tests/test_airbox_{energy,modal,freefield}.py`;
`tests/helpers.py::make_airbox`/`airbox_noise`/`gaussian_pulse`; `scripts/diagnose_airbox.py`.
Suite **1235** green (was 1173); the batch's 62 run in 2.97 s.

---

**BATCH 2 — the room pushes back — PLANNED (2026-07-28), not built.** Plan
`docs/dev/air-box-back-reaction-plan.md`. The room seen from a port node is a **Thevenin source**:
`pbar = pbar_free + R_room*q`, so the whole two-way coupling is ONE DIVISION,
`U = (U_free - G*pbar_free)/(1 + G*R_room)` — structurally `RadiatedBody`'s rank-1 solve reused
verbatim, unconditionally passive (`1 + G*R_room >= 1` always, no CFL, no guard). Seven traps
measured on a prototype BEFORE any core code; the numbers that would otherwise cost a rediscovery:

- **`R_room = k*rho0*c0^2 / (2*W*(1+beta))` — the `(1+beta)` is NOT optional.** batch 1's `step()`
  adds the source BEFORE the wall closure, so a port on a lossy wall gets divided by `(1+beta)` too.
  Measured drift: **8.4e-15 with** the factor, **1.9e-2 without**. Wall-mounted ports are supported
  *because* of it; the naive spelling passes every interior-port test and leaks only on a wall.
- **A port on an OPEN face is silent and the books say nothing is wrong.** `p=0` pins `pbar=0` and
  `R_room=0`, so `injected` and `acoustic` are **exactly 0.0** forever while the drift is 8.6e-15.
  The energy report — this project's primary bug detector — is structurally blind to it. Refuse.
- **A POINT port's load diverges as `1/h` and refining makes it WORSE.** Measured `|Z|` **x2.00 per
  halving** over three grids (37³/74³/148³), = `3.16..3.29 x |Z_sphere(a=h)|` at every frequency:
  **a point port is a pulsating sphere of radius ~h/3.2**, i.e. mostly an ADDED MASS (detune), not
  damping. NOT the cell compliance (that would be `1/h³`) — it is the monopole near field at the
  only radius the grid has.
- **A fixed-radius SPREAD port converges** (x0.96 then x1.01 over three grids, 203 -> 1743 -> 13613
  nodes; non-monotone because the STAIRCASED ball's discrete volume wobbles around `4*pi*a^3/3`), so
  it ships. But it converges to the **uniformly-injecting BALL**, not the pulsating shell — the
  classic **6/5** factor (equivalent shell radius `5a/6`); measured 1.11x at ka=0.23. An oracle
  against `from_sphere(a)` would be wrong by design.
- **Disjoint ports are EXACTLY independent (7.1e-15); coincident ports drift 3.6e-2.** Same
  measurement read both ways: it scopes N instruments IN and refuses overlap. Snapping is what makes
  the hazard real — two ports 5 mm apart on a 13.5 mm grid ARE one port.
- **The arrival oracle is MANHATTAN, not Euclidean.** The 7-point stencil spreads one node per step,
  so a second body first moves at `Manhattan + 1` steps — **measured 7** where `r/c0` says **17.5**.
  A causality test written against `r/c0` fails 2.5x early for reasons unrelated to the coupling.

**Design decision: the port does NOT step the room — the caller does, once, after every port has
solved.** That is what lets N instruments share a room and lets a string-driven member work at all
(the *bridge* owns `body.step`). `free_pressure_at` deliberately ignores queued injections, which is
exact iff ports are disjoint; the forgotten-`room.step()` guard is therefore **per-port**, not a
global "is `_pending` empty" (that would fire on the second instrument of every scene).
`RoomLoadedBody.energy()` must be an EXPLICIT override — `__getattr__` delegation would silently
return the bare modal energy without its coupling channel.
