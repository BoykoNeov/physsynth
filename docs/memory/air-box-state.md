---
name: air-box-state
description: 3-D FDTD air box (HANDOFF §12H) batches 1–6 ALL SHIPPED — §12H's MODEL LIST IS COMPLETE (b6 built 2026-08-17); the DISTRIBUTED air tier above the lumped port; reward and price both live at λ=1/√3; b2's port is a Thevenin source in ONE division and the ROOM contaminated the port's own measured size by more than the effect; b3 = distributed area coupling, CONSERVATION IS BLIND to a wrong coupling constant; b4 = the plate becomes an OBJECT (two-sided dipole), the MONEY TEST is not sufficient either; b5 = the drumhead — NO SINGLE ONE OF THE THREE DETECTORS IS SUFFICIENT, and a drumhead is quiet because it is COMPACT not because it is subsonic; **b6 = the GONG — the first radiating object whose acoustic character depends on HOW HARD IT WAS HIT (σ_shape moves 46% loud vs 1.4% quiet, 33× the control), the money test is blind for a THIRD distinct reason (it is arithmetic on whatever w^{n+1} came out of the solve), the LEDGER is the wrong observable because the ROOM's own build-up moves the control as much as the effect moves the claim, and — the finding a later batch most needs — THE AIR GRID CANNOT BE COARSENED TO BUY AFFORDABILITY BECAUSE COARSENING THE ROOM BREAKS THE PLATE'S FIXED POINT** (72 Picard sweeps at 57.9 kHz, NaN at 33 kHz); the seam CALLS the model's _linear_rhs() rather than transcribing it, the opposite of _PlateSurface and for a reason; rho_v-vs-rho_s is a 1000× slip no ledger sees; §7.7 directivity REFUSED on a costed CONTRADICTION (a 120 ms window needs a 41 m room)
metadata: 
  node_type: memory
  type: project
  originSessionId: 8506c3d9-c98e-4229-87cf-6160a3fd6c48
  modified: 2026-08-10T04:11:39.214Z
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

**BATCH 2 — the room pushes back — BUILT & GREEN (2026-07-28).** Plan
`docs/dev/air-box-back-reaction-plan.md`. `RoomPort` + `RoomLoadedBody`, both in `airbox.py`; ZERO
edits to `radiation.py` / `body.py` / `connection.py` / `bore.py`. All seven planned traps held.

**The whole batch is one observation about batch 1's `step()`:** within a step an injection changes
the pressure at its OWN nodes and nowhere else (propagation waits for the next momentum sub-step).
So the room seen from a port is a **Thevenin source**, `pbar = pbar_free + R_room*q`, and the body's
rank-1 solve closes in ONE DIVISION: `U = (U_free - G*pbar_free)/(1 + G*R_room)`. `1 + G*R_room >= 1`
always ⇒ **unconditionally passive**, no CFL of its own, no guard. Conservation of
`Σ inst.energy() + room.energy()` measured **~1e-14** at interior / wall / edge / corner / spread
ports, rigid / matched / mixed walls, coarse and fine grids, one and two instruments.

**THE PLAN'S CROSS-TIER ORACLE DID NOT SURVIVE CONTACT — and the reason is the finding.**
- **The oracle became the port's EQUIVALENT RADIUS, not `|Z|`.** `a_eff = rho0/(4*pi*M_a)` with
  `M_a = Im Z / omega`. One number saying what the port IS as a sphere, and it does BOTH halves of
  §7.10 at once. `|Z|` carries the room's modal wiggle; the near-field MASS does not.
- **THE ROOM CONTAMINATED THE MAGNITUDE BY MORE THAN THE EFFECT BEING MEASURED.** Same port, same
  `h`, room swept: `a_eff/(5a/6)` = **1.086 (0.5 m) → 1.040 (0.7) → 1.003 (1.0) → 0.977 (1.4)**. The
  plan's "1.11 at ka=0.23, consistent with 6/5" was reading the ROOM's reactance. The 6/5 shape
  factor is real and now confirmed to **0.3%** — but only where the port is COMPACT. **A ratio
  survives a small cheap room; a MAGNITUDE does not.** Get the room out of the way before comparing
  anything to a closed form.
- **`a_eff` is COURANT-INVARIANT to 5 significant figures** (45.231/45.232/45.233/45.234 mm across
  cfl 0.5→0.998). It is a *static* near-field quantity ⇒ run the sweep at the cheapest λ, and the
  measurement is not a dispersion artifact — not assumable in a scheme with NO dispersionless λ. The
  first sweep grids were accidentally at **0.998 of the CFL ceiling** (where the corner mode goes
  defective); harmless here, but only because it was checked.
- Contrast, three grids: point `a_eff/h` = 0.324/0.320/0.317 (ratios **0.493, 0.496** — it HALVES)
  vs ball ratios **1.045, 1.038**. A factor of twenty in grid sensitivity. THAT is the assertion.

**Arrival constants, measured with the SHIPPED code (the plan's prototype numbers were off by one):**
- reflection returns at **`2d + 1`** steps — measured at d = 3, 5, 7 (three geometries, one law),
  asserted as BIT-IDENTITY between two rooms differing only in `Lz`. Careful: the reference room
  must be deep enough that its OWN reflection is still in transit (Nz=12 → step 19 vs largest t=15).
- a second body first moves at **exactly Manhattan** (not `Manhattan+1`), at separations 6 and 12,
  against `r/c0` = 11.5 and 13.6. Euclidean is the wrong oracle by up to 2x.

**Build-time gotchas worth not rediscovering:**
- **The room must book `injected` from its OWN post-closure `pbar`**, never a number the port hands
  back. Port books `k*pbar_predicted*U`; the two agree only if `R_room` is exactly right, so their
  difference IS the bug detector. Shortcut it and the conservation test becomes a tautology.
- **A spread port must NOT go through `_pending`** — that path is a scalar Python loop and a 5 cm
  port is 203 nodes at h=13.5 mm, 13613 at 3.4 mm. Separate vectorized queue (`_pending_ports`).
- **The forgotten-`room.step()` guard keys on `room.n`**, per port — not mark-clearing, and NOT a
  global "is anything pending" (that fires on the second instrument of every scene). `set_state`
  clears every port's mark.
- `self.body = body` must be the FIRST assignment (else `__getattr__` recurses). `energy()`,
  `set_state()` and `reset()` are all explicit overrides — delegation returns a stale ledger.
- **`bridge.energy()` ALREADY includes `inst.energy()` including `radiated_energy`** — adding it
  again double-counts, and it shows up as a 3.8e-3 drift that reads like a scheme bug.
- The local `O(port)` free-pressure read is asserted **BIT-IDENTICAL** to the full-array
  `_divergence()`-then-closure at interior/wall/edge/corner. Not in the plan; added on review,
  because an off-by-one there SURVIVES every energy test (port and room would still agree).

**Refusals, all measured rather than argued:** port on an **open** face (`injected` and `acoustic`
exactly 0.0, drift 8.6e-15 — perfectly conservative, perfectly silent, and the energy report is
structurally BLIND to it); **overlapping** ports (3.6e-2 vs 7.1e-15 disjoint — the same measurement
scopes N instruments IN); a radius the grid cannot resolve (it would silently be a point port); a
sample-rate mismatch; a port solved twice without `room.step()`. Checks run over the WHOLE mask —
an interior centre can still reach a wall once the ball is laid down.

**The port does NOT step the room** — the caller does, once, after every port has solved. That is
what lets N instruments share a room and lets a string-driven member work (the *bridge* owns
`body.step`). Disjoint ports are bit-identical under solve ORDER, which is why `free_pressure()` may
ignore queued injections.

**Files:** `physsynth/core/airbox.py`; `tests/test_airbox_port.py` (structural),
`tests/test_airbox_scene.py` (physical + cross-tier); `tests/helpers.py::make_room_loaded_body` /
`room_scene_energy`; `scripts/diagnose_airbox_port.py`. Suite **1287** green (was 1235);
the batch adds **52 tests in 8.9 s** (~7 s of that is the one cross-tier sweep).

---

**BATCH 3 — the plate radiates from EVERY NODE — COMPLETE & SHIPPED (2026-08-09).** Plan
`docs/dev/air-box-area-coupling-plan.md`, whose **§10 is the post-build record** — read it before
trusting any number in that plan's body. `SurfacePort` + `RoomLoadedPlate`, both in `airbox.py`;
**zero edits** to `AirBox` itself, `plate.py`, `connection.py`, `body.py`, `radiation.py`, `bore.py`.
**Full suite re-run at batch end: 1355 green, exit 0** (was 1287); the batch adds **68 tests**, and
1355 − 1287 = 68 = exactly `test_airbox_surface.py`, so nothing anywhere else moved.

**The claim: a surface radiates by the SHAPE of its motion, not its net volume displacement.** An
even-index supported-plate mode has *exactly* zero net volume velocity (`Σ sin(mπi/N) = 0` is an
identity, and `B = L²` keeps the sine product an exact eigenvector), so every one-port in the repo
reports exact silence — and the distributed port radiates **5.6× the (1,1) mode's energy** from it
while `|U|/A` stays at 6e-14 for the whole 200-step run. Structural, not numerical: a one-port has
no length scale on its surface.

**The mechanism is one line up from batch 2.** The room's instantaneous response over a node SET is
**diagonal** (off-diagonal measured *exactly* `0.00e+00`), so the load is `TᵀRT` — constant,
symmetric, PSD, sparse — and it **folds into the plate's own `splu`**. Nothing new is solved.
Batch 2 IS the rank-1 case (`T = w1ᵀ` → Sherman–Morrison). Passivity is a property of the matrix,
not an inequality to check.

**THE BATCH'S REAL FINDING — CONSERVATION IS BLIND, AND THIS INVERTS THE PROJECT'S INSTINCT.** The
plate's ledger telescopes against *whatever* pressure it used and the room's against *whatever* it
received, so the scene total is the sum of two separately-exact identities and stays flat when the
two disagree. Measured (dropping `(1+β)` from `R_j` on a lossy mounting wall, 300 steps): drift
**4.9e-15 — SMALLER than the correct run's 2.0e-14**, i.e. green and not even suspicious — while
`|radiated − injected|` goes from **exactly 0.00e+00 to 18%** of the channel. **So the money test is
`radiated == injected` plus the differential per-node `R_j`, never the total.** This also corrects
`RoomPort.R_room`'s shipped docstring, which said the missing factor "leaks ~2% of the run's
energy": nothing leaks — that 1.9e-2 was the LEDGER GAP. A test asserts BOTH halves so the framing
fails loudly if it ever stops being true.

**THREE MORE PLAN CLAIMS DID NOT SURVIVE MEASUREMENT — all in the same direction (the prototype
measured them where the effect was hidden). Re-measure a plan number in YOUR geometry.**
1. **Bilinear's mirror-equivariance is NOT offset-independent.** It needs `S = 2·(surface centre)/
   h_air` **integral**: 1.0e-15 there, **1.6e-01…3.8e-01** elsewhere (16 offsets), confirmed two
   independent ways (`TᵀRT` under the plate's mirror permutation; `T` itself under that mirror
   composed with the air grid's) and by the algebra — the mirror sends node `i` to cell fraction
   `frac(S − t_i)`, which is the `1 − f_i` that reverses a bilinear weight pair only for integral
   `S`. ⇒ **centring is load-bearing TWICE** (the load's equivariance AND the scene's symmetry),
   which strengthens the centred default rather than weakening it.
2. **The symmetry argument does not discriminate the spreading operators at all.** Centred, an even
   mode's `|U|/A` stays at rounding under BOTH bilinear and nearest at N=8/16/24 — the plan's "18%
   leak under nearest" does not reproduce. What decides bilinear is **coverage**: 10×–100× flatter
   interior assignment and *converging* (0.082/0.062/0.051/0.031) where nearest wanders
   (0.83/1.03/0.64/0.46). Nearest's equivariance is exact at an EVEN `S` and broken at an ODD one,
   where the centre node lands on a round-half-to-even tie — an accident of the rounding rule.
3. **"Partition of unity" promises more than it delivers.** Bilinear's interior assigned area is
   exactly `h_air²` (5e-16) **only when `h_air/h_surface` is an integer**, and off it ripples
   non-monotonically in the ratio (2.93 → 0.0077 but the finer 4.40 → 0.0207). Poisson summation on
   the hat: the `k`-th coefficient `sinc²(πk h_air/h_p)` vanishes when `k·ratio` is an integer, so
   WHICH harmonics cancel is arithmetic, not size. That ripple is the coupling's accuracy floor.

**Design decisions worth not re-deriving:**
- **The sign convention is LOCAL** — positive `u` along the port face's **inward normal**, so `T` is
  entrywise non-negative on all six faces and no inward normal appears in the code. Necessary
  because a consistent flip is **bit-identical** in `radiated_energy` (`TᵀRT` is sign-invariant);
  the only detector is the surface pressure at **n = 1**. The test asserts the bit-identity of the
  WRONG run — that is the point of it.
- **Refuse the face RIM, do not clip the stencil.** A rim node touches a SECOND wall, so it carries
  half `W` and the SUM of two admittances ⇒ `R_j` stops being uniform ⇒ `TᵀRT = R·TᵀT` (the whole
  equivariance argument) stops holding. Clipping would keep every ledger green with the geometry
  quietly wrong.
- **Zero `AirBox` edits** because `step()` is LINEAR in the port weights: a `SurfacePort` passes the
  per-node volume-velocity VECTOR as `weights` with `U = 1.0`, and both the injection and the
  read-back come out right.
- `self.plate` FIRST (or `__getattr__` recurses); never shadow a name `StringPlateBridge` reads
  (`K W B L w mask theta kappa rho h u u_prev …`) — hence `_lu_loaded`, `load_matrix`.
- **The guard errs SAFE, measured:** the margin is **bit-identical** loaded vs bare (the load is
  dissipative — it enters `A`, never `G0`), and adding the load block to `G0` anyway *reduces*
  `(G0⁻¹)_dp` (ratio 0.500 supported / 0.995 free). **The SIGN is the claim, not the size.**
- **The load DOES thicken the factorization** — LU fill **1.55×/3.50×/5.29×** at
  `h_plate/h_air = 0.45/0.23/0.15`, against the plan's expectation that it would not. `lu_nnz` is
  exposed because stored `nnz` (2.9×/8.7×/18.2×) is NOT what `splu` pays.
- **The channel is a property of the MOTION, not the coupling:** a struck bump gives 0.002 of `E₀`
  (fine patterns radiate badly — the short circuit working), the free plate's **piston** gives
  0.9974. Ship the piston config so a conservation assertion is non-vacuous.
- `net_area` is `((N−1)/N)²·LxLy` supported (dead rim nodes displace nothing) and exactly `LxLy`
  free. `T = 0` reduces to bare `Plate` **bit-identically** (structural zeros eliminated before
  factoring) — the available reduction, since `R = 0` happens only on a refused open face.

**THE DIAGNOSE SCRIPT CORRECTED TWO MORE OF THE PLAN'S OWN READINGS** (`scripts/diagnose_airbox_
surface.py`, ~18 s, three figures) — and the pattern is now five-for-five: **every plan claim that
died, died because the prototype measured it where the effect was hidden.**
- **Coincidence is a SCALING COLLAPSE, not a knee.** At fixed `f` the five patterns span **39×**;
  plotted against `f/f_c` they collapse to within 1.5×–5.5× and every curve **PEAKS at `f/f_c = 1`**.
  The unity crossing is on the rising flank, `[0.70, 0.85] f_c` — a bracket **one sweep interval
  wide**, i.e. exactly as tight as the frequency grid and no tighter. Say so or it reads as a
  measurement.
- **The fine plate-mode rows are limited by the AIR grid's SPACE axis, not the plate's time axis**
  (the plan said time). `(4,2)` radiates 0.018 → 0.870 → 0.9998 across a 4× air refinement. But that
  sweep cannot attribute anything, because `h_air = c₀√3/(CFL·fs)` moves BOTH axes together — **the
  control is to pin `h_air` and reach the same rates by lowering the Courant fraction** (0.900,
  0.450, 0.225): 4× the time resolution moves `(4,2)` not at all (0.0181, 0.0155, 0.0228). Caveat
  kept in the open: the control runs at a different dispersion. And it is ONE row repeated — `(3,1)`
  weak support, the other four flat at ~1.0000.
- **Once every mode is resolved the RANKING INVERTS** — per cycle at equal rms velocity: 1.000,
  0.448, 0.448, 0.260, 0.213, 0.091, strictly decreasing. The plan's "finer radiates MORE, up to
  7.1×" was measuring the **cycle count** over a fixed window. The zero survives untouched
  (2e-15…3e-14 at every level) — the resolution-free claim is resolution-free, the other was not.
- **Rig facts that would each have produced a confident wrong figure:** equal rms **velocity** is
  what makes modes comparable (equal displacement puts 4700× more energy in the finest); a **lossy**
  room is what makes "did it radiate" one-way (a rigid box hands it back and the fraction wanders);
  the drive needs a **raised-cosine ramp** (a hard start's click beats the steady-state power at
  fine patterns); and the **resolvability floor prints FIRST**, because an aliased point on a
  monotonicity curve looks exactly like a clean result.

**Files:** `physsynth/core/airbox.py`; `tests/test_airbox_surface.py`;
`tests/helpers.py::make_surface_room` / `make_room_loaded_plate` / `plate_bump` /
`plate_mode_shape` / `surface_scene_energy`; `scripts/diagnose_airbox_surface.py`. Docs closed at
batch end: HANDOFF §12H (three batches, batch 4 named), README's model catalogue, the plan's status
block + new §10, and `tests/test_airbox_port.py`'s header — which had gone on repeating the
"leaks ~2%" wording that §6.1 disproved, in the very file whose job is that trap.

---

**BATCH 4 — the plate stops being a SOURCE and becomes an OBJECT — COMPLETE & SHIPPED
(2026-08-10).** Plan `docs/dev/air-box-dipole-plan.md`, whose **§10 is the post-build record — read
it before trusting any number in that plan's body.** `AirBox.add_cut`/`cut_faces`, a private
`_PatchPort` base, `InteriorSurfacePort`, `RoomSuspendedPlate`, all in `airbox.py`; **zero edits** to
`plate.py`, `connection.py`, `body.py`, `radiation.py`, `bore.py`. Full suite at batch end:
**1453 green, exit 0** (was 1355); the batch adds **98** = exactly `test_airbox_cut.py` (42) +
`test_airbox_dipole.py` (56), so nothing anywhere else moved. Deferred beyond it, unchanged: PML /
higher-order absorbing boundaries, scattering objects and non-rectangular rooms, a plate not aligned
with a grid plane or of finite thickness, `Membrane`/#6 as suspended surfaces (the port needs **no
change** for either — neither is wired up), moving ports, viscothermal absorption.

**PROCESS SCAR, worth more than any number here: `git checkout -- <file>` to undo a small temporary
edit DISCARDED the entire batch's core work.** It was replayable from the transcript and the golden
pin proved the replay byte-faithful, but the right move is to revert the edit itself (or stash), and
to commit a durable checkpoint before touching the file to test a deliberate break.

**The plan's own §3 fear was wrong, and pleasantly.** It budgeted for new boundary machinery
("replacing their momentum update with the plate's motion"). Measured: prescribing the face velocity
and injecting a `−q/+q` pair on the two node planes straddling the plane are **the same arithmetic**,
because `A_face·w_z = W` identically. ⇒ **`AirBox` needs exactly ONE new thing: the cut** (zeroing `u`
on a face set, applied inside `_momentum` so `step()` and `set_state()` are both covered). Port
protocol, weights and the `injected` ledger are batch 3's untouched. Batch 1's step ordering paying
off a third time. The load is `2·TᵀRT` — same sparsity, same fill, still PSD, still folds into `splu`.

**A free new machine-precision oracle came with the cut.** It lies on a FACE, half a cell past the
last node, so a full cut splits the room into two **half-offset** sub-rooms of length `(m+½)h` and
`(N−m−½)h` — summing to `L` exactly, no cell lost — whose exact modes are `cos(nπi/(m+½))`, **not**
the room's own `cos(nπi/N)`. Measured 1.2e-14. Full cut isolates at exactly `0.000e+00`; a partial
cut passes 2.2e-01 (the diffraction an unbaffled plate lives on). NB this validates the **cut
primitive**, not the port — §6.3's rim refusal means a *port* can never seal the room.

**THE METHODOLOGICAL FINDING, and it outranks the physics: the money test is NOT sufficient either.**
Batch 3 established the conserved total is blind to a wrong `R_j` and crowned `radiated == injected`.
Batch 4 has a coefficient batch 3 lacked (the **2**), and **each ledger is blind to a different way of
getting it wrong**: `1×` inside the factorization ONLY → money test 1.5e-16 (blind), total 1.3e-4
(caught); `1×` consistently → total 1.2e-15 (blind), money test 1.45× channel (caught). **Only the
coupled residual catches both** — the achieved `u^{n+1}` put back into the PDE with the force taken
from the ROOM's own post-closure pressure jump, at TWO timesteps. And `assert load == 2*one_sided` is
a tautology; the 2 is earned differentially (`∂p̄_lo/∂q = −R_j`, `∂p̄_hi/∂q = +R_j`, same `R_j`,
1.15e-16 rel, off-diagonal exactly `0.00e+00`).

**Two observables TRIED AND REJECTED as radiation measures.** `radiated_energy` and `t50` both count
the **reactive near field** as though it had left, and batch 4's channel is **50.9% negative
increments** where batch 3's was 0.0% — a reservoir, not a drain. They disagreed about *direction* in
the prototype. ⇒ radiation efficiency needs a **prescribed-velocity** rig (batch 3's even-mode test
already wrote that rule for its own reason). Any **mode ranking is refused twice over**.

**Design decisions not to re-derive:**
- **The cut is ADDITIVE, and disjointness must cover cut FACES, not only pressure nodes.** A
  single-slot cut lets a second plate silently un-block the first — it degrades to a pure `−q/+q`
  source with every ledger green. Two ports can have disjoint NODES and overlapping CUTS (the node
  sets live on different planes while the cuts share one), which is why this refusal is genuinely new.
- **The cut is the SUPPORT OF `T`**, not the plate's footprint (clipping `T` is batch 3's refused
  failure shape).
- **The sign convention CANNOT be local like batch 3's** (an interior plane has no inward normal), so
  `+x/+y/+z` it is. Refinement over the plan: the **consistent** flip (`∓q` order AND jump direction
  together) is the invisible one — load matrix, solve, `pressure_jump` and `radiated_energy` all
  BIT-IDENTICAL while the room's field is exactly inverted; flip only ONE and the conserved total
  catches it. ⇒ the detector is the sign of the **ROOM's own** pressure at step 1, not the port's
  `pressure_jump`. (The port's own jump inverts at step **8** here, not the plan's 7.)
- Use the plate's **centered** velocity in the `n+½` face slot — forced, because the forward
  difference is an **added mass** and would land in the guard's `G0`. Bounded by a `k`-only
  refinement: **0.14%** across `λ = 0.52 → 0.065`.
- The phantom (inject, don't cut) is **bit-identically two `AirBox.inject` monopoles** (`0.000e+00`).
  `T = 0` → bare `Plate` bit-identically **even though the cut is still there** — the cut belongs to
  the ROOM and the load to the PLATE.
- `index` is a FACE index and its legal range is **`1 ≤ index ≤ N−2`** (the plan said `N−1`): both
  straddling NODE planes must be interior or `R_j` differs between the sides. Consequence: an
  interior patch can never touch any wall, so **no open-face refusal is needed** — provable, not an
  omission.

**IT RETIRES A PREDICTION OF BATCH 3'S — measured, and the docstring + plan §6.8 now say so.** The
face cut does **not** make the load non-dissipative: it removes air inertia from the *room's* ledger,
where it was never in the plate's `G0`, while the load stays ∝ `u^{n+1} − u^{n−1}` and enters `A`.
Margin bit-identical, and it is **batch 3's own** `0.2061806714931906` / `0.2061840079056186`, not
the plan's `0.2052…` — because the guard never saw either load.

**FOUR MORE PLAN CLAIMS DIED ON MEASUREMENT — including the plan's own pass criterion. The
five-for-five pattern from batch 3 holds: every one died because the prototype measured it where the
effect was hidden.**
1. **`ratio ≤ 2` is the WRONG pass criterion** — the very test §7.7 invented to catch its own
   contaminated prototype. Windowed re-measurement: the **ratio** hits 2.31 (2.49 refined) while each
   **arm** is fine — baffled 0.281→0.915 rising monotonically to its asymptote of 1, dipole topping
   at 2.12–2.30. The ratio exceeds 2 because the baffled arm has not saturated, and a piston
   overshoots its own asymptote near its first maximum. ⇒ the criterion belongs on **each arm,
   asymptotically**; the baffled arm's textbook shape is what makes the dipole arm believable.
2. **The dipole arm's MAGNITUDE does not converge, and the obstacle is why.** At `ka = 1.0` under air
   refinement it reads 0.279, 0.231, **0.360**, 0.225 — tracking `blocked_area` (1.44, 1.36, **1.78**,
   1.44), not `h` — while the baffled arm, using the SAME `T`, converges smoothly. Only the ratio's
   sign against 1 is a claim.
3. **§6.4's "±3% while the area moves 33%" is true of the FRACTION and false of everything else.**
   Two probes disagree: `rad/E0` moves 1.03× while `t50` moves 1.45× (piston) / 2.75× (a (2,1) mode),
   and the prescribed-velocity `R` agrees with the RATE. The fraction is the one observable that
   cannot see the effect. (A mode's fraction saturates at 1.0000 — not the same as insensitive.)
4. **The overshoot's denominator was wrong.** `blocked/plate` = 1.49…1.18 "trending to 1" is the
   **free** branch only; the **supported** branch goes *below* 1 (1.513, 0.927, 0.925, 0.799) because
   the clamped rim is not in `T`. Against the **LIVE** rectangle — the moving surface, the honest
   denominator — it reads 2.690, 1.647, 1.644, 1.421 and does trend to 1.

**The physics claims, as measured in the build:**
1. **Directivity — survives everything.** `1.000, 0.928, 0.786, 0.565, 0.347, 0.164, 0.012` vs
   `cos θ`: an **85× in-plane null**, against a baffled arc reaching only 0.530 with no null, and
   lumped one-ports with none *possible*. Reported with **SNAPPED** radii and angles (0.0/14.3/30.3/
   45.9/59.2/74.2/88.8 at r = 1.175–1.222 m) and with the baffled 90° probe flagged as sitting ON the
   wall. **The phantom has the SAME pattern (in-plane 0.010) at 5.2× less** ⇒ directivity identifies
   the dipole, the BLOCKAGE sets its strength; both tests are needed.
2. **Unbaffling CROSSES 1** — 0.278, 0.569, 1.339, 2.257, 2.314, 1.965 over `ka = 0.8…2.8`, crossing
   between 1.0 and 1.3. No constant, hence no `R(ω)`, reproduces a crossing.
3. **The cut is un-omittable** — phantom/dipole `t50` 5.2 → 19.3 → **40.8**, i.e. it *diverges*; the
   free piston's phantom never reaches `t50`. A doublet at separation `h` has moment ∝ `h` by
   construction, so this is an **implementation** control (exactly what a clobbered cut produces),
   not a claim about dipole sources in general.
4. **`t50` and radiated power point OPPOSITE ways and both are right.** The suspended free piston
   sheds energy ~20× FASTER than baffled (`dip/baf` 0.047/0.039/0.041) while RADIATING far less at
   that `ka` — because `t50` counts the reactive near field as though it had left. That pair is the
   proof that neither a decay time nor `radiated_energy` is a radiation measure, and it is why the
   attribute's docstring leads with "the work done on the air — and about half comes back".

**Files:** `physsynth/core/airbox.py`; `tests/test_airbox_cut.py`, `tests/test_airbox_dipole.py`;
`tests/helpers.py::make_cut_room` / `make_suspended_plate` / `sub_room_mode`;
`scripts/diagnose_airbox_dipole.py` (~7 min, ~1 GB peak, three figures). A golden-number test pins
batch 3's `SurfacePort` **bit-identical** across the `_PatchPort` refactor, captured from the
pre-refactor code — the only way to cash that claim once the old code is gone.

---

## Batch 5 — the drumhead in the room (SHIPPED 2026-08-10)

Plan `docs/dev/air-box-membrane-plan.md`. `Membrane` (#4) suspended + baffled. VK plate (#6) is
**batch 6**, deliberately split off. Two commits pushed: the plan, then the plan's own corrections.

**HANDOFF §12H's "the port needs no change for either" is DEAD — for the disk only.**
`_check_footprint` builds its required set as a bounding **BOX** (a per-axis min/max outer product),
so a *circular* membrane is refused at every resolution — 20/48/40 unfed at `N=40/56/72`, identical
on both tiers — and **refining makes it worse, not better**, because the bbox corners sit ~0.41·R
from the nearest live node and are not under the surface at all. A rectangle of the same spacing is
accepted. Fix (validated before writing): required set = **rows ∪ columns** span-wise. Measured as
the *union*, not inferred from rows — the first probe measured rows alone and the plan sentence
described the union, which is strictly stronger. Union is 0 for every disk incl. odd `N=25/41/57` at
`R=0.13` (non-square air footprint), 0 for non-square rectangles, and still refuses every comb
(64/96/96 vs bbox 68/112/132). Comb threshold unmoved under both: `h_mem/h_air ∈ (2.02, 2.20]`.

**The probe killed three more claims, two of them the plan's own:**
- Draft's footprint ceiling `h_mem ≤ h_air` — **wrong**, a bilinear stencil spans **TWO** air cells,
  so the ceiling is ~2. Hence `c/c₀ ≤ √2/λ_air = 2.449` at the room's CFL, **not** `√(3/2)=1.2247`.
  The supersonic arm is roomy, not a marginal point. (Advisor independently guessed ~1.2 too.)
- `airbox.py` L1974-6 says #4/#5/#5b all have a `pressure()` read-out. **`Membrane` does not**, and
  cannot cheaply (two-level roll keeps no `_accel`) → doc fix, read-out stays absent.
- Expectation that the channel would be ~1e-6 at a realistic head. **0.84 of E0** at `c/c₀=0.31`,
  (0,1) bulge, rigid room (bump 0.23 — the short circuit *in the channel*). §7.2 needs no supersonic
  head and no lossy room. A membrane has **no piston** (clamped rim, no rigid-body nullspace), so
  batch 3's 0.9974 configuration doesn't exist — the `(0,1)` bulge is the named one.

**The headline and its inverting trap.** `c = √(T/ρ)` has no `ω` in it → a membrane has **no
coincidence frequency**: subsonic at every mode or supersonic at every mode, set by the single
number the player tunes. (Real Mylar ≈ 0.31.) But the 5-point scheme's dispersion drops `c_ph` below
`c`, so a marginally supersonic head falls **back** below `c₀` on the grid — `βh = 1.445` at
`c/c₀ = 1.05`, ~2.2 nodes/wave — manufacturing exactly the coincidence the claim denies. Ships
**bracketed**, band held below the 1% knee (`βh = 0.686`, ~4.6 nodes/wave at `λ=1/√2`). En route:
**at `λ = 1/√2` the DIAGONAL is exactly dispersionless** (1.0000 across the band) while the axis
degrades to 0.707 at Nyquist — the 2-D analogue of 1-D `λ=1`, on one direction only.

**Design facts for the build.** The load goes in `A`, so the **explicit** membrane acquires a solve
it never had — first batch in the family with that choice (every earlier load landed inside a solve
that already existed); lagged-explicit ships once as a measured negative control, `spreading=
"nearest"`'s precedent. Prototype already validates the scheme: `|radiated−injected|/channel`
1e-16…1e-15, scene total flat to 1e-13, across 16 configurations. `coords = X[mask]` excludes the
**dead clamped rim**, so area is the live sum not `πR²` (0.9943) and `blocked_area` is 1.228× the
moving surface — batch 4 says the dipole magnitude tracks `blocked_area`, so this is exactly the
shape that yields a plausible-wrong magnitude. Seam extraction (`a_bare`/`rhs`/`denominator`/
`surface`/`commit`) goes **first and ALONE**, guarded by batches 3/4's bit-identity margins
`0.2061806714931906` / `0.2061840079056186`; the membrane adapter must NOT be in that commit.
`Membrane.step()` takes no `f_ext` (same gap as `VKPlate`) so the adapter's `f_ext` term is **new**
arithmetic with no bit-identity anchor. VK is batch 6 because its conservation holds only at the
**Picard fixed point** → its seam is a *loop hook*, and its money test becomes a two-parameter
`couple_tol` claim.

**Probe scripts:** `M:\claud_projects\temp\airbox-b5-probe\probe{,2,3}.py`.

### What the build changed (2026-08-10, four more commits — SHIPPED)

`RoomLoadedMembrane` + `RoomSuspendedMembrane`, both domains, on an extracted seam. 348 air-box
tests (44 new). `plate.py` and `membrane.py` untouched, as promised.

**The seam went first and alone, and its guard had to be BUILT — the suite could not check it.**
The plan said the refactor was "guarded by construction" by batches 3/4's pinned stability margins.
It is not: those come from `StringPlateBridge._stability_margin`, i.e. from `G0`, which the air load
**never enters** — so they are blind to `rhs()` and `commit()` and would pass with a wrong RHS.
Everything else asserts physics to a tolerance and a changed float operation order slides under
1e-15. So: a byte-exact baseline captured at HEAD *before* the edit (16 configurations = 2 tiers ×
supported/free × f_ext absent/present × σ 0/4; sha256 of the raw bytes of `u`, `u_prev`, `_accel`
and the room's pressure field, `float.hex()` of the ledgers, `nnz_growth`, `lu_nnz`), compared after.
All 16 bit-identical. `M:\claud_projects	empirbox-b5-seam\pin.py`. σ is in the cross product
because σ=0 makes the `sk·u^{n-1}` term vanish, so a slip there would have been invisible.
**No base class** — one implementation is unexercised scaffolding, and `commit()`'s `_accel` write
is the plate's, not the seam's (Membrane has no `_accel`). Batch 6 adds the base when #6 arrives.

**THE THIRD DETECTOR — the finding that outranks the physics.** The measured negative control (lag
the load velocity at `(uⁿ−u^{n-1})/k` and keep the membrane explicit) drifts the scene total by
**3.8e-2 of E0** while `|radiated − injected|` stays at **1.6e-16**. So the MONEY TEST is the blind
one here and the CONSERVED TOTAL is the detector — the exact inverse of batches 3 and 4. Reason:
`radiated == injected` is a property of the **port relation alone** (the room gets exactly the `q`
it was handed at exactly the pressure it then has), so it cannot see which velocity produced the
`q`. Batch 3: total blind. Batch 4: money test blind. Batch 5: money test blind again, opposite
mechanism. **No single one of the three is sufficient** — that is now the family's standing rule,
not a per-batch surprise.

**The headline splits in two and only one half is size-free.** Kinematic half, exact, no simulation:
`k₀/β = c/c₀` at EVERY membrane mode (measured 0.313 four times over modes (1,1)…(4,4)) against the
plate's `√(κω)/c₀` = 0.864 → 1.727 → 2.591 → 3.454, which CROSSES. Operational half — "therefore
short-circuited" — is **size-dependent, and that killed the plan's framing**. The corrugated-surface
rig (prescribe `sin(βx)sin(βy)sin(ωt)` at FIXED ω, sweep β, so only `k₀/β` moves) measures at
`ka = 8`: knee exactly at 1, each arm saturating at its OWN plane-wave asymptote (1 and 2 — which
also retires batch 4's "ratio ≤ 2 is the wrong criterion", since there the baffled arm had simply
not saturated), and a **70.2× / 74.8×** rise across the knee. **The multiplier is NOT the claim and
I published a wrong one first**: the full sweep's 3773× / 8859× has its bottom point at 2.7 air
cells per structural wave, i.e. an ALIASING floor as much as a cancellation floor, so it is an
upper bound; 70× is the figure over the resolved band (`k₀/β` 0.60–1.25, ≥5 cells/wave). The KNEE
is robust across 5.3 / 7.1 / 8.9 cells; batch 4's "the crossing is the claim, the magnitudes are
not" applies to this batch's own headline number. At `ka = 1.2`, where a real head's first
modes live (0.30 m Mylar fundamental = 253 Hz, ka = 0.78): **no knee at all**, 69×, peaking at
`k₀/β = 0.60`, because the patch carries 0.3 structural periods. **A real drumhead's fundamental is
quiet because the head is COMPACT, not because it is subsonic.** Both halves ship; stating only the
first would have been true and misleading.

**Five claims died, three of them the plan's own** (plan §10.1):
1. §2.1's "comb threshold unmoved at (2.02, 2.20]" — probe 3's bbox column was computed from the
   REACHED set; the shipped one is from the COORDINATE extent, inset by up to one node per side, so
   the new criterion is slightly **stricter** (20 unfed vs 17 at 2.02) and that interval was another
   room's. What survives: both criteria change verdict between the same two spacings.
2. §7.6's "the lagged load breaks the ledger identity" — see the third detector above.
3. §7.7's rig — a prescribed UNIFORM piston has **no structural wavenumber**, so `c` never enters
   and the sweep would have been constant by construction. Redesigned as the corrugated surface.
4. §1.1's operational half — size-dependent, above.
5. The `f_ext` one-step oracle needed the load REMOVED: even from rest the first step's centred
   velocity is nonzero, so the room loads the head immediately and moves the answer **0.96%** —
   exactly the size an "approximately equal" test waves through. Pinned unloaded, plus a static
   deflection `u_ss = −L⁻¹f/(T h²)` for the sign and the operator, because `Membrane.step()` takes
   no force and this arithmetic has nothing to be bit-identical to.

**Other build facts worth keeping.** The footprint tests assert the fix *discriminates* — each disk
case recomputes the superseded required set and asserts it was nonzero, so a no-op could not pass.
A membrane has **no piston** (clamped rim ⇒ no rigid-body nullspace), so batch 3's 0.9974 non-vacuous
configuration does not exist; the single-signed bulge replaces it (0.14–0.82 of E0). The two areas
bracket the nominal from opposite sides — the clamped rim makes the mover smaller, the staircased
footprint makes the obstacle larger. Diagnose script `scripts/diagnose_airbox_membrane.py`; its
figure 2 needs no room at all.


## Batch 6 — the gong in the room (SHIPPED 2026-08-17)

Model #6, the von Kármán nonlinear plate ([[von-karman-plate-state]]), as a baffled and a suspended
surface, both boundary branches (`supported` = gong, `free` = cymbal). Plan
`docs/dev/air-box-vk-plate-plan.md`. **This completes §12H's model list**: every resonator in
`physsynth/core/` that can be a surface now can be one *in* the room.

**Batch 5's handover was right about exactly one thing, and it was the load-bearing one.** The air
load is linear in `w^{n+1}` and **independent of the Airy stress `F`** — that orthogonality is what
makes the batch tractable at all. So it folds into `A` exactly once and the Picard loop is otherwise
untouched. What the seam needed was a **loop hook**, not a second `rhs()`:
`solve(lu, rhs_fixed) -> (w^{n+1}, F^{n+1})`, because the loaded back-substitution sits *inside* the
fixed-point iteration. `rhs_fixed` is sweep-invariant (it depends only on `p̄_free` and `w^{n-1}`),
which is why the hook is cheap. `commit(u_next, F_next)` rolls **two** histories; there is no
`_accel` (model #6 has none, hence no `pressure()` on the wrapper, following batch 5).

**The seam does the OPPOSITE of `_PlateSurface` in one place, deliberately.** `rhs()` **calls**
`VKPlate._linear_rhs()` instead of transcribing it. The asymmetry is a fact about the two models:
`Plate.step` *inlines* its theta-scheme RHS, so batch 3 had no choice but to copy; model #6 already
hoists it out for the Picard loop. Calling it keeps `plate.py` untouched *and* deletes a whole class
of transcription slip. Verified, not assumed — see below.

**The trap: `rho -> rho_s`.** `VKPlate` has **no** `rho`. It has `rho_v` (volumetric) and `rho_s`
(areal), differing by the thickness — **1000× at `e = 1 mm`**. Write `rho_v` and the load is 1000×
too weak while every ledger still telescopes against the pressure it used: nothing green turns red.
Only the `nonlinear=False` bit-identical regression catches it, which is why that regression is a
commit of its own — and it was **falsified deliberately**, off-tree: `rho_v` everywhere fails 8/8
cases, `rho_v` in the `f_ext` divide alone fails 8/8, and a **pure reassociation** of the `f_ext`
term (same maths, different rounding) still fails 5/8. `vk_linear_twin()` exists because three of
the comparison plate's inputs fail in a way that reads exactly like a load bug: `rho=vk.rho_s`, the
**snapped** `Ly=vk.Ly` (re-snapping a nominal `Ly` can give a different `Ny`, hence `n_live`), and
`nu=vk.nu` (inert supported, load-bearing free).

### The headline — the first radiating object whose character depends on how hard it was hit

Every radiator here before #6 is linear in its excitation, so radiated fraction, directivity and
dipole-over-baffled are amplitude-**invariant** by construction. The VK coupling is quadratic, so the
**shape** of the motion evolves during a single strike — and shape is what `SurfacePort` exists to
make audible (batch 3). Hence:

> **A loud plate's radiation is time-varying at fixed geometry, and a quiet one's is not.**

No `R(ω)` can state it — a scalar-per-frequency load has *one* pattern per frequency.

| tier | `w/e` | modal drift | `σ_shape` spread | resolved-band |
|---|---|---|---|---|
| baffled | 0.05 | 0.029 | **1.4%** | 0.2% |
| baffled | 3 | 0.362 | **46.0%** (33× control) | 4.6% (20×) |
| suspended | 0.05 | 0.009 | **0.4%** | 0.1% |
| suspended | 3 | 0.336 | **17.3%** (39× control) | 1.6% (18×) |

**Batch 5's doctrine bit its successor immediately**: only **17 of 289** modes keep ≥5 air cells per
structural wave, and the cascade's destination modes are exactly the ones the air grid resolves
worst. **The SEPARATION is the claim and survives the restriction (20×/18×); the MULTIPLIER is an
upper bound and is not claimed.**

**The compact limit does not merely under-read — it points the wrong way.** The monopole (everything
`AirRadiation` / `RadiatedBody` / `RationalAirLoad` can see) is 3e-7…3e-6 of the true figure, and for
the suspended cymbal at `w/e = 3` it **rises 1.38× while the true efficiency falls to 0.93×**. A
lumped one-port would report the cymbal getting brighter as it dulls. This is plan §7.8, chosen over
§7.7 on measurement as the plan required.

### The detector finding — the money test is blind for a THIRD distinct reason

A new error axis: model #6 conserves only at the **Picard fixed point**, so `couple_tol` sits beside
the air load as an error source.

| `couple_tol` | scene total | \|radiated − injected\| | `last_residual` | sweeps |
|---|---|---|---|---|
| 1e-13 | 1.2e-13 | 2.2e-15 | 9.9e-14 | 19 |
| 1e-6 | 5.1e-7 | 8.7e-16 | 9.9e-7 | 9 |
| 1e-3 | 1.0e-3 | 1.6e-15 | 1.0e-3 | 4 |

`radiated == injected` is arithmetic on **whatever `w^{n+1}` came out of the solve**, so an
under-converged one is ported *self-consistently*: the books balance to rounding while the physics is
wrong by a part in a thousand. Batch 3's blind spot was the total, batch 4's a `2` in the
factorization, batch 5's which velocity made the `q`, batch 6's this. **Four batches, four ways for a
single detector to be insufficient.** The plan's *more* interesting hoped-for outcome — that the
scene total might also be blind — **did not happen**, and the reason it was wrong is worth keeping:
the committed `F^{n+1}` is the Airy solve of the **previous** iterate while `w^{n+1}` is the current
one, and the gap between them *is* the increment the tolerance bounds. Self-certifying half passed:
loaded drift falls with `couple_tol` at the same rate as unloaded, within 1.2×.

### FIVE more claims died, two of them the plan's own

1. **The ledger is the wrong observable for the headline.** Radiated energy per window does not
   separate the runs: the **room's own build-up** moves the quiet *control* 1.79× while the effect
   moves the claim 3.64×. Batch 2's lesson recurring — a magnitude wearing a ratio's clothes
   (denominator the plate, numerator the room). What ships instead is a functional of the **shape
   alone**, `σ_shape = vᵀ(TᵀRT)v / (ρ₀c₀A⟨v²⟩)`: the room's own resistive operator applied to the
   plate's own *coupled* motion. The run stays coupled; only the read-out is a fixed quadratic form.
2. **§7.7's directivity panel — refused on a COSTED CONTRADICTION, not on cost.** The pattern change
   needs a 120 ms window; a reflection-free 120 ms needs a room **41 m** across. No room size
   resolves it. Stronger than "too expensive".
3. **§4's cost model, 8× optimistic.** The build cell is 109 ms per room step at 2.35 M nodes, i.e.
   ~21 min for 0.2 s against the costed 2–3 min. Shipped room is 0.6 m — legitimate because every
   claim is a ratio.
4. **§0.5's convergence characterisation — and the correction is a genuinely new coupling between
   the two grids.** The Picard sweep count was measured against plate *geometry* at a fixed 96 kHz.
   It is a strong function of the **timestep** too, and **the room sets the timestep**: at `w/e = 3`,
   **72 sweeps at 57.9 kHz, NO convergence (NaN) at 33.0 kHz, and at 22.0 kHz even `w/e = 2`
   diverges.** So **the air grid cannot be coarsened to buy affordability — coarsening the ROOM
   breaks the PLATE's fixed point.** A second, independent reason this family's cost runs the wrong
   way, on top of the 3-D CFL's `h⁻⁴` ([[airbox-viewer-state]]). `couple_max_iter`'s default of 50
   caps out at the build cell; the script raises it to 120.
5. **§0.3's spectral peak, confirmed dead in practice:** identify modes by projection under the mass
   matrix, never by an FFT peak — at `w/e = 3` a peak tracker reads a mode at 0.53× its own linear
   frequency because the field has gone broadband.

### Numbers a later batch will want

* The **velocity** piston is the free plate's fat channel (27.6% of `E0` baffled, 4.6% suspended, vs
  0.25%/0.13% for a strike) — and a rigid translation carries no stretching, so the VK coupling is
  **asleep** there. Both configurations run in the suite for that reason. A **displacement** piston
  is not a piston: no velocity, so it sits still and radiates nothing.
* The coupled residual: 1.4e-14…3.5e-14 correct, vs 8.6e-2 with the VK term dropped, 4.3e-2 halved,
  1.2e-2 with the air load halved — the one guard that sees the nonlinear force and the air load
  **separately**.
* **`F^{n-1}` must be captured BEFORE the step** in any external residual: `commit()` rolls it away,
  and the `μ`-average is `(F^{n+1} + F^{n-1})/2`, not `(F^{n+1} + F^n)/2`. And the coupling must be
  rebuilt from the **committed** state or it reports the Picard increment instead.
* A **narrow** strike does not converge at large amplitude — the broad strike (0.20 `Lx`) is not a
  stylistic choice, it is the only one that runs.

**Files:** `physsynth/core/airbox.py` (`_VKPlateSurface`, `_RoomLoadedVKPlateMixin`,
`RoomLoadedVKPlate`, `RoomSuspendedVKPlate`); `tests/test_airbox_vk.py` (64 tests, ~45 s);
`tests/helpers.py::make_air_vk_plate` / `make_room_loaded_vk_plate` / `make_suspended_vk_plate` /
`vk_linear_twin` / `vk_strike`; `scripts/diagnose_airbox_vk.py` (~4 min, one figure).

**Still out, and now askable rather than blocked:** `StringVKPlateBridge` is its own batch —
`connection.py` reads `plate.rho`, calls `plate.step(f_ext=...)` and delegates `plate.pressure()`,
none of which model #6 has, and what a *linear* 2-DOF stability guard means for an
amplitude-dependent stiffness is a real question, not plumbing. Because no bridge composes with these
wrappers yet, `RoomLoadedVKPlate.__getattr__` is free of `RoomLoadedPlate`'s "NOTHING here may shadow
a name the bridge reads" constraint. A **viewer batch** now has both the new model and the new claim
its own rule requires ([[web-viewer-state]]).
