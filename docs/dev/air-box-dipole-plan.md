# The plate stops being a source and becomes an object — the interior two-sided (dipole) plate (air-box batch 4)

Batch 3 mounted a plate flush in a wall and let it radiate from every node. The wall did the rest:
it was the textbook infinite baffle, the plate's back face was unloaded, and the plate was — for all
the room could tell — a *source*, a patch of wall that moved. Batch 4 takes the wall away. The plate
hangs **in** the room, radiates from both faces, and is driven by the pressure *jump* across it.

That is not batch 3 with a sign. A source adds sound to a room. An object also **removes** paths
through it, and this one does so whether or not it is moving. Three measurements say the difference
is the whole point rather than a correction to it:

* **The blockage is not a refinement of the source — the source alone converges to silence.** Drive
  the ±`q` pair with no cut (the "phantom": a legal, perfectly conservative dipole *source* with the
  plate's own motion and no obstacle) and the plate's `t50` runs 5.2×, 22×, then **45×** longer than
  the blocking plate's as the air grid is refined 1× → 2× → 3×. It diverges, because a transparent
  doublet at separation `h` has moment ∝ `h`. There is no grid at which "a dipole source" is a
  coarse version of "a plate".
* **Unbaffling is not a factor — it changes sign.** Under prescribed uniform motion, the ratio of
  radiation resistances `dipole / baffled` measures **0.057** at `ka = 0.275` and reaches **1** near
  `ka ≈ 0.9`: at low frequency the two-sided plate is the far weaker radiator, and by mid-band it is
  the stronger. A ratio that **crosses 1** cannot be reproduced by any constant, and therefore not by
  any `R(ω)` fitted to the baffled case — the same structural argument as batch 2's delayed echo and
  batch 3's short circuit, a third time. (The crossing is the claim; the prototype's high-`ka`
  magnitudes are room-contaminated and are not — see §7.7.)
* **It has a direction, and nothing else in this repo does.** Windowed free-field, the two-sided
  plate's pattern is `1.000, 0.924, 0.839, 0.599, 0.448, 0.163, 0.034` from on-axis to in-plane,
  against `cos θ = 1.000, 0.966, 0.866, 0.707, 0.500, 0.259, 0.000` — a **29× null in the plate's
  own plane**. The baffled plate over the same arc is `0.79 … 1.00`, no null anywhere, and every
  lumped one-port in this repo (`AirRadiation`, `RadiatedBody`, `RationalAirLoad`, `RoomPort`) is a
  monopole with no angular dependence at all. An angular null is not a magnitude, so it survives
  every caveat the magnitudes carry.

And the implementation news is better than §3 of the area-coupling plan feared. That section
budgeted for new boundary machinery — "putting the plate on a plane of velocity faces and replacing
their momentum update with the plate's motion". Measured, prescribing the face velocity and
injecting a ∓`q` pair on the two node planes that straddle the plane are **the same arithmetic**:

```
    node (i,j,m)  divergence term  +u_face / w_z   ->  dp = -k rho0 c0^2 u_face / w_z
    a soft source q at that node                   ->  dp = +k rho0 c0^2 q / W,   W = A_face * w_z
```

and `A_face * w_z = W` identically, so with `q = A_face * u_face` the two agree term for term, with
the sign flipping between the planes. **`AirBox` therefore needs exactly one new thing: the cut** —
zeroing `u` on a set of faces. The port protocol, the injection weights and the `injected` ledger are
batch 3's, unchanged. Batch 1's step ordering paid for batch 3 the same way (`SurfacePort`: "a soft
injection at a wall node *is* the moving-wall condition"); this is that lesson one geometry further.

---

## 1. Why — the refusal being discharged

The area-coupling plan §3 deferred this in as many words:

> **The interior, two-sided (dipole) plate.** A real cymbal radiates from both faces and the pressure
> *jump* across it drives it. […] It is the natural batch 4, it is a genuinely different object (an
> internal moving boundary, not a source), and mixing it in here would double the batch.

and HANDOFF §12H closes on it: "What is left is a plate the room can reach from **both** sides."
Every instrument in the free-plate family — the suspended cymbal of model #5b, the gong of #6 — is
physically an object hanging in air, not a patch of wall. `RoomLoadedPlate` models a loudspeaker
cone or a soundboard in a cabinet honestly and a cymbal dishonestly, and the dishonesty is not a
small one: at `ka = 0.275` it overstates the radiation resistance by **17×**.

---

## 2. The physics — a cut, and a ∓`q` pair that straddles it

### 2.1 The cut

The plate lies on a plane of velocity **faces**: `u_z[i, j, m]`, the faces between node planes `m`
and `m+1` (and likewise for an `x`- or `y`-normal plate). Zero those faces at every half-step and
the air on either side of the plane is disconnected there. That is a rigid, zero-thickness,
perfectly reflecting internal partition, and it costs nothing to book:

* a cut face's `u` is identically zero at every half-step, so its contribution to the kinetic sum
  `½ρ₀ Σ W_f u^{n+1/2} u^{n-1/2}` is identically zero — **the energy identity needs no new term and
  no exclusion list.** §3 anticipated "removing those faces from the kinetic-energy sum"; they
  remove themselves.
* the potential/kinetic telescoping is untouched, because it only ever used the divergence's being
  the transpose of the gradient, which the zeroing does not disturb.

Measured: energy flat to **6.5e-16** of the acoustic scale (rigid room) and **7.2e-15** (lossy),
with a cut and a driven source, over 400 steps.

### 2.2 The cut has its own exact modal oracle — and it is a *new* one

Cut the full cross-section at face index `m` and the room becomes two independent rooms. Their
lengths are **`(m + ½)h` and `(N − m − ½)h`**, which sum to `Nh` exactly: the cut lies on a face,
half a cell past the last node on each side, so no cell is lost and neither sub-room is the room's
own grid restricted. The cut end is **face-centered** (the mirror plane sits between nodes), so the
ghost condition is `p_{m+1} = p_m` and the exact discrete eigenvector along the cut axis is

```
    cos( n pi i / (m + 1/2) )          NOT  cos( n pi i / N )
```

tensored with ordinary node-centered cosines in plane. This is a batch-1-tier oracle — machine
precision, not a convergence rate. Measured across three cut positions and both sides: field error
**1.16e-14 … 1.69e-14** against `amplitude · cos(2π f n k) · shape`, energy drift **2.0e-16 …
5.0e-16**.

Two things follow that must be said plainly in the suite, or it reads as validating a coupling it
does not touch:

* this oracle exercises the **cut primitive** (`AirBox.add_cut`), not the port. §6.3's rim refusal
  means a legal `InteriorSurfacePort` can never span a full cross-section, so a *port* can never
  seal the room. A hand-placed cut can, and that is what the oracle uses.
* the corollary is physical, not a limitation to hide: **a legal interior plate always has a
  diffraction path around it.** Measured — a full cut passes **exactly `0.000e+00`** across, a
  partial cut passes **2.232e-01** of the same source.

### 2.3 The load is `2 TᵀRT`, and the room's response is diagonal on *both* planes

With `T` the batch-3 volume-weight matrix (now surface → **face**), `q = T v` the per-face volume
velocity, and the injection weights `(−q, +q)` on planes `m` and `m+1`:

```
    pbar_lo = pbar_free_lo  -  R q
    pbar_hi = pbar_free_hi  +  R q
    f       = -T^T (pbar_hi - pbar_lo)  =  -T^T d_free  -  2 T^T R T v
```

with the same `R_j = k ρ₀ c₀² / (2 W_j (1 + β_j))` on both planes. So the load is exactly **twice**
the one-sided load on the same `T` and `R` — two faces, two resistances — and it is still constant,
symmetric, PSD and sparse, so it still folds into the plate's own `splu` with nothing new solved.
The family's ladder gains its last rung: `RadiatedBody` scalar `R` → one division; `RationalAirLoad`
`R_eff` + one aux state → one division; `RoomPort` scalar `R_room` → one division; `SurfacePort`
diagonal `R` with matrix `T` → absorbed into the LU; **this** the same, doubled.

**The factor 2 must be earned by measurement, not by construction** (§6.1). Measured differentially
off the room — perturb `q` at one patch face, read the room's own post-closure `p̄` at both planes:

```
    d pbar_lo / dq  =  -R_j     to 1.15e-16 relative
    d pbar_hi / dq  =  +R_j     to 1.15e-16 relative
    off-diagonal    =   0.00e+00  exactly       (rigid and lossy-wall rooms alike)
```

That — the same `R_j`, opposite signs, exactly zero elsewhere — is what the 2 *is*.

### 2.4 What the dipole does that the baffled plate cannot

| | baffled (batch 3) | interior dipole (batch 4) |
|---|---|---|
| faces loaded | one | two |
| local load | `TᵀRT` | `2 TᵀRT` |
| blocks a path | no (it *is* the wall) | **yes**, and even at rest |
| far field | no angular null | `cos θ`, **29× null in plane** |
| coupling channel | monotone (lossy room, piston): **0.0%** negative increments | **50.9%** negative increments — a reservoir, not a drain |
| `R_rad` vs baffled | — | **0.057 at `ka = 0.275`, crossing 1 near `ka ≈ 0.9`** |

The reactive row is the one that changes how this batch must be *measured*, and §7.7 is built on it.

---

## 3. Scope — and what is deliberately deferred

**In scope:**

* **`AirBox.add_cut`** — an additive, rigid, zero-thickness internal partition on a plane of faces,
  with the half-offset sub-room modal oracle. Public, because it is a physical object in its own
  right (a room divider, a duct termination) and because the oracle needs it without a port.
* **`InteriorSurfacePort`** — a surface on an interior plane of faces: the cut over the support of
  `T`, plus the ∓`q` pair. Written against batch 3's **surface protocol** (nodal coordinates, nodal
  areas) so `Membrane` (#4) and any future grid resonator mount with no port edit.
* **`RoomSuspendedPlate`** — `Plate` (#5 `"supported"` and #5b `"free"`) hanging in the room, loaded
  on both faces. A drop-in for `Plate` by `__getattr__` delegation, so `StringPlateBridge` takes it
  as the body and `string → bridge → plate → room` comes free — **measured, not inferred** (§6.6).
* The **shared spreading operator**: `SurfacePort._spread` and the footprint/open-face/disjointness
  checks factored out and used by both ports, with a test pinning that `SurfacePort`'s own behaviour
  is **bit-identical** to before the refactor.
* The **cut-disjointness** refusal (§6.2), which is a genuinely new failure mode and not inherited.
* The energy ledger, the cross-ledger identity, the differential `∓R_j` oracle, the **coupled
  residual** (§6.1 — promoted from batch 3's supporting cast to this batch's primary guard), the
  first-step sign oracle, and the refusals of §6.
* **N instruments in one room**, mixed tiers, once §6.2's refusal exists to make it safe.

**Out of scope — say so, do not drift:**

* **A plate not aligned with a grid plane**, and a non-rectangular footprint. Staircasing in 3-D is
  the membrane batch's lesson and deserves its own measurement.
* **A plate of finite thickness**, and therefore any edge-diffraction detail at the plate's own rim
  beyond what a zero-thickness cut gives.
* **A sealing plate** (a partition spanning a full cross-section as a *port*). §6.3's rim refusal
  forbids it; `add_cut` provides the rigid version, and a *moving* seal needs the non-uniform `R_j`
  that refusal exists to prevent.
* **Any absolute radiated-power magnitude** as a suite assertion, and **any ranking of plate modes
  by radiated energy** — batch 3's even-mode test already refuses the latter in terms ("a plate mode
  locks spatial fineness to frequency […] belongs only to a prescribed-velocity rig"), and §7.7
  explains why batch 4 must refuse it twice as hard.
* **`Membrane` (#4) and the von Kármán plate (#6)** as suspended surfaces. The port is written so
  neither needs port changes; neither is wired up or tested here.
* **Everything batches 1–3 deferred stays deferred**: PML and higher-order absorbing boundaries,
  HRTF/ambisonics, scattering objects and non-rectangular rooms, viscothermal absorption, moving
  ports, overlapping ports.
* **A viewer batch.** Core work, per batches 1–3 and radiation batch 3.

---

## 4. API

```python
PLANES = ("x", "y", "z")

class AirBox:
    def add_cut(self, plane: str, index: int, extent=None) -> None: ...
    @property
    def cut_faces(self) -> int: ...          # how many faces are currently cut (reported, not tuned)

class InteriorSurfacePort:
    def __init__(self, *, room, plane, index, coords, areas, origin=None,
                 spreading="bilinear") -> None: ...
    node_count: int          # 2 x n_face -- BOTH planes
    face_count: int          # the faces the surface cuts
    net_area: float          # sum of nodal areas (batch 3's meaning, unchanged)
    blocked_area: float      # the cut rectangle -- NOT net_area; see 6.4
    def free_pressure(self): ...             # (pbar_free_lo, pbar_free_hi), O(patch)
    def inject(self, q) -> None: ...         # the per-FACE volume velocity; queues (-q, +q)

class RoomSuspendedPlate:
    def __init__(self, *, plate, room, plane, index, origin=None,
                 spreading="bilinear") -> None: ...
    def step(self, f_ext=None) -> None: ...
    def energy(self) -> float: ...           # plate.energy() + radiated_energy
    radiated_energy: float                   # WORK DONE ON THE AIR -- about half of it comes back
    pressure_jump: np.ndarray                # last (pbar_hi - pbar_lo), Pa per face
    nodal_volume_velocity: np.ndarray        # last q, m^3/s per face
    volume_velocity: float                   # sum_j q_j -- the lumped tier's coupling, the control
```

**`radiated_energy` keeps batch 3's name and must not keep its connotation.** Batch 3's channel was a
one-way drain, so "radiated" was honest there. Batch 4's is **half reservoir** — 50.9% of its
increments are negative (§7.7) — so a reader who takes `0.136` to mean "13.6% radiated" is wrong by
roughly a factor of two in a direction the number cannot show. The docstring must say, in its **first
line**, that this is the *work done on the air* and that a large fraction of it comes back, with the
same directness batch 3 used for "the conserved total is structurally blind to a wrong `R_j`". The
alternative — renaming it — is refused because the identity `radiated_energy == room.injected` is the
money test and both sides must obviously name the same thing.

Naming, stated so it can be vetoed: `InteriorSurfacePort` / `RoomSuspendedPlate` follow the family's
habit of naming for *structure* (`RoomPort`, `SurfacePort`, `RoomLoadedPlate`) rather than for the
physics (`DipolePort` was the alternative). `add_cut` uses `plane="z", index=m` rather than a
`FACES`-style token, because an interior plane has no "end".

**The sign convention, and it is different from batch 3's.** `SurfacePort` could make the inward
normal disappear by defining positive displacement *along* it, which is why "no inward normal appears
in the code at all" there. An interior plane has two sides and no inward normal, so batch 4 must
choose: **positive surface displacement is along the plane's own axis, `+x`/`+y`/`+z`**, so the
`index+1` plane is the one a positive velocity compresses. The `−q`/`+q` order in `inject` is
therefore load-bearing and is *not* the same kind of invisible as batch 3's flip (§6.5).

---

## 5. The discrete scheme

Per step, in this order (`p̄_free` must be read from the stored `u^{n+1/2}` **before** the room
advances — batch 2's contract, unchanged):

```
1.  (lo_free, hi_free) = port.free_pressure()      # two vectors, O(patch)
    d_free = hi_free - lo_free
2.  rhs = the plate's own force-free RHS  +  k^2 f_ext / rho_s
          -  k^2 T^T d_free / rho_s
          +  (k / 2 rho_s) (2 T^T R T) u^{n-1}
3.  u^{n+1} = LU_loaded.solve(rhs)                 # A + (k / 2 rho_s) 2 T^T R T, factored ONCE
4.  q      = T (u^{n+1} - u^{n-1}) / (2k)          # per-face volume velocity, m^3/s
5.  d_pbar = d_free + 2 R q
6.  radiated_energy += k (d_pbar . q) ;  port.inject(q)   # queues (-q, +q)
7.  (caller) room.step()                           # zeroes the cut faces after its momentum step
```

`rho_s` is `rho h²` on the supported branch and `rho` on the free branch, exactly as batch 3.

**Why the plate's *centered* velocity goes into a face slot that lives at `n+½`.** The face
velocity is a half-step quantity; the plate's natural velocity at time `n` is
`(u^{n+1} − u^{n-1})/(2k)`. Using the centered one is what makes both ledgers telescope against the
*same* `(q, p̄)` pair, which is what makes `radiated == injected` an identity rather than an
approximation. The alternative — the forward difference `(u^{n+1} − u^n)/k`, which is the *formally*
correct half-step object — puts a term proportional to `u^{n+1}` alone into the load, i.e. an
**added mass**, which would land in the stability guard's `G0` and break §6.6's bit-identity. So the
choice is forced, and it is the same half-step mixing batch 3 already ships.

Batch 3's channel was resistive, so its half-step placement barely mattered; batch 4's is dominantly
**reactive** (§7.7), and a reactive coupling is more sensitive to it. Nothing in the ledgers can see
this — both sides use the same pair, and a dense reference built with the same convention reproduces
the convention — so it is bounded by a **`k`-only refinement**: fix `h` and the geometry, raise `fs`
so `λ` drops. Measured, the free plate's piston, `λ = 0.5196 → 0.2598 → 0.1299 → 0.0650`:

```
    radiated / E0  =  0.464558,  0.465338,  0.464871,  0.464809
```

— converged to **0.2%**, non-monotone at the 1e-3 level. The half-step placement is worth less than
that, isolated from the blockage overshoot that confounds a space refinement.

**No `_accel` correction**, for batch 3's reason: the load is inside the solve, so the second
difference already carries it.

---

## 6. Traps — measured before a line of core code

### 6.1 Neither ledger catches a wrong load coefficient. The *residual* catches both halves.

Batch 3 established that the conserved total is blind to a wrong `R_j` and promoted
`radiated == injected` to the money test. Batch 4 has a coefficient batch 3 did not — the **2** — and
the money test is blind to half of the ways to get it wrong. Two negative controls, both realistic:

| error | coupled residual | `\|rad − inj\|` / channel | scene drift / `E0` |
|---|---|---|---|
| *(correct)* | 4.3e-14 / 1.1e-13 | 3.0e-16 / 3.7e-16 | 3.9e-15 / 2.6e-15 |
| **A** — `1×` inside the **factorization only** (the prediction still says `2R`) | **1.7e-3** | 1.5e-16 — **blind** | 1.3e-4 — caught |
| **B** — `1×` **consistently** ("I forgot the plate has two faces") | **4.4e-3** | **1.45×** — caught | 1.2e-15 — **blind** |

(pairs are supported / free.) So: **the money test alone is not sufficient either**, which is the
sharpest correction batch 4 makes to batch 3's methodology. What catches both is putting the achieved
`u^{n+1}` back into the coupled PDE with the force computed from the **room's own post-closure
pressure jump** — a number the port never touched — at **two** timesteps, because a wrong-but-
consistent `k`-dependent factor passes at one. `assert load_matrix == 2 * one_sided` is *not* a test;
it re-checks arithmetic the same file just wrote.

### 6.2 The cut is not single-slot, or a second plate silently un-blocks the first

A naive `room._cut = ...` is overwritten by the next port. Measured with two plates in one room: the
second plate's cut replaces the first's, plate A **degrades to the phantom** — it keeps injecting
its ∓`q` and stops blocking — and the scene stays perfectly green (`drift/E0 = 1.7e-10`,
`|rad − inj| = 1.5e-21`). At the refinement where the phantom is 45× weaker, that is a silent
45× error in one instrument.

So the cut is **additive** (a boolean face mask, accumulated), and the disjointness refusal must
cover **cut faces as well as pressure nodes** — two ports can have disjoint node sets and
overlapping cuts, because the node sets live on different planes while the cuts can share one. This
is a design decision, not a retrofit: it is why `add_cut` is additive and why a port records its own
face set.

### 6.3 The rim refusal, inherited and doubled

Batch 3 refuses a stencil reaching the mounting face's rim, because a rim node touches a second wall
and `R_j` stops being uniform. Batch 4 needs it **twice**: in-plane (a face on the room's wall
carries a half or quarter transverse area, and its node a halved `W`), and along the plate's normal
(the two straddling node planes must both be interior, `1 ≤ index ≤ N − 1`). Both keep `R_j` uniform
across the patch, which is what the `TᵀRT = R(TᵀT)` equivariance argument rests on. The consequence
— no port can seal the room — is stated in §2.2, not hidden.

### 6.4 The obstacle is up to one air cell larger than the plate, and that is the honest choice

The cut is the **support of `T`**, not the plate's footprint. It has to be: the identity in the
header holds only where a face carrying `q` is also cut, so a face with `q` and no cut is a phantom
patch, and clipping `T` instead is exactly the "volume conserved, ledgers green, geometry quietly
wrong" failure batch 3 refuses. The cost, reported rather than discovered — `blocked_area /
plate_area` under air-grid refinement:

```
    1.490,  1.571,  1.352,  1.183      (h_p/h_air = 0.23 -> 0.54)
```

non-monotone (the node count is a rounding of the footprint) and trending to 1. What it is *worth*
is much less than it looks: the same runs give a piston `radiated/E0` of `0.1429, 0.1514, 0.1504,
0.1479` — **±3% while the blocked area moves 33%**. Report the overshoot; do not treat it as the
accuracy floor of the radiated number. **To be re-measured in the build with a mode-shaped motion as
well as a piston** — the overshoot extends the rectangle asymmetrically relative to a mode's nodal
lines, and only the piston has been checked.

### 6.5 The orientation flip is invisible to every energetic quantity — read the **first** step

`2 TᵀRT` is sign-invariant, so swapping the `−q`/`+q` order leaves the load matrix, the solve and
`radiated_energy` bit-identical while the plate is **anti-driven** by the room's incident field:
passive, conservative, green, and pushing the wrong way. This is batch 3's §6.2 in its batch-4 form,
and it is worse here because batch 4's convention (§4) genuinely has a per-plane sign where batch
3's did not.

Its only detector is the sign of the pressure jump on the **first** step. Measured, a free plate
given a uniform `+z` velocity, `kz = 4` in a room with `Nz = 9` (so the scene is exactly symmetric
about the cut):

```
    step 1:  pbar_lo = -4.6362e-01   pbar_hi = +4.6362e-01     <- correct: it compresses what it moves toward
    step 7:  pbar_lo = +7.097e-02    pbar_hi = -7.097e-02      <- inverted
```

The sign flips at step 7. Batch 3 recorded the same trap ("a six-step read gave the *wrong* answer
because the plate's own half period was five steps"); here it is reproduced with a different period,
which is the point — the number of steps is not the lesson, reading step 1 is. Note also that
`pbar_lo == -pbar_hi` to the last digit in a mirror-symmetric scene, which is a free second oracle.

### 6.6 The stability guard stays bit-identical — and batch 3's prediction of this is **wrong**

`tests/test_airbox_surface.py::test_string_bridge_plate_room_chain` says, in its docstring:

> Pinning the bit-identity here means a future change making the load non-dissipative — the two-sided
> dipole plate of batch 4, whose face cut removes air mass — fails loudly instead of silently
> mis-guarding.

It does not fail, and the reasoning behind the prediction does not survive. The face cut removes air
inertia from the **room's** ledger, where it was never part of the plate's `G0`; the load itself
stays proportional to `u^{n+1} − u^{n-1}` — dissipative, merely doubled — so it enters `A` and never
`G0`, exactly as in batch 3. Measured, `StringPlateBridge` over a `RoomSuspendedPlate` vs a bare
`Plate`:

```
    supported:  margin  0.2052148342611817  both, to the last digit
    free:       margin  0.20521483773139457 both, to the last digit
    (G0^-1)_dp with the load block added anyway:  ratio 0.9994 (both)  -- below 1, so it errs safe
```

Batch 3 already said it is "the sign of that ratio, not its size, that is the claim", so the size
difference from its own 0.500/0.995 is parameters, not physics. **Correcting that docstring and the
area-coupling plan's §6.8 prediction is a deliverable of this batch** (§9), in the repo's habit of
retiring a plan claim by commit.

### 6.7 The phantom is bit-identically two monopoles — which is what makes it a fair control

Injecting `(−q, +q)` with no cut is not an approximation of anything; it is exactly two
`AirBox.inject` soft sources. Measured over 60 randomised injections: `max|p_ref − p_port| =
0.000e+00`. That matters because the phantom is this batch's headline control (§7.6), and a control
is only worth its bit-identity to the tier it stands in for — here, batch 1's monopole, twice.

### 6.8 `_free_pressure_nodes` needs no change, and it is worth saying why

The shared local read replicates `AirBox.step`'s divergence exactly. On the two straddling planes the
face between them is cut, i.e. `u = 0`, which the read picks up from the stored array like any other
value — the `idx < N` / `idx > 0` branches are unchanged and no cut-awareness is needed. The open-face
refusal carries over unchanged and for batch 3's second reason as well (the local read does not pin
`p = 0` where `step` does).

---

## 7. Oracles — what must pass (prototype numbers in brackets)

### 7.1 The cut conserves, and its sub-rooms have exact modes

Energy flat with a cut plus a driven source, rigid and lossy [6.5e-16, 7.2e-15 of the acoustic
scale]. Full-cut sub-room modes `cos(nπi/(m+½))` exact, both sides, three cut positions [field error
1.16e-14 … 1.69e-14; drift 2.0e-16 … 5.0e-16]. Sub-room lengths sum to `Lz` exactly. Parametrized
over all three plane orientations.

### 7.2 A full cut isolates exactly; a partial cut does not

Peak `|p|` across the plane from a source on one side: full cut **exactly `0.000e+00`**, partial cut
`2.232e-01`. The exact zero is the assertion that a cut is a *rigid* boundary and not a strong
impedance.

### 7.3 The money test, and the differential ∓`R_j`

`|radiated − room.injected| ≤ 1e-12 × |radiated|` across both plate boundaries and rigid / all-lossy
rooms [8.5e-21 absolute, against a channel of 8.4e-2 `E0`]. And `∂p̄/∂q` measured off the room:
`−R_j` on the low plane, `+R_j` on the high plane, **the same `R_j`**, off-diagonal asserted
`== 0.0` exactly [1.15e-16 relative; `0.00e+00`].

### 7.4 The coupled residual, at two timesteps, against the room's own pressure

§6.1's table, as a test — including **both** negative controls, since each is blind to a different
one of the two ledgers [correct 4.3e-14 / 1.1e-13; controls 1.7e-3 and 4.4e-3].

### 7.5 The sign oracle, at step 1, over three orientations

The pressure jump's sign read on the first step, parametrized over `x`/`y`/`z` planes and both plate
boundaries, plus the mirror-symmetry identity `p̄_lo == −p̄_hi` [±4.6362e-01]. And, batch 3's shape:
a test that asserts the **bit-identity of the wrong run's** `radiated_energy` — that is what makes
"invisible to every energetic quantity" a measured claim rather than a warning.

### 7.6 The headline: the source alone converges to silence

`t50` (steps to lose half the plate's energy) for the same plate and the same motion, three
mountings, under air-grid refinement 1× / 2× / 3×:

```
    supported (1,1) mode:   dipole/baffled  = 0.75, 0.75, 0.79     (converged)
                            phantom/dipole  = 5.2,  22,   45       (DIVERGES)
    free rigid-body piston: dipole/baffled  = 0.12, 0.10, 0.09
                            phantom         = never reaches t50
```

The assertion is the divergence, not a value: the phantom/dipole ratio must **grow** with refinement.

**Be exact about what that proves.** The phantom is a `(−q, +q)` doublet at separation `h`, so its
moment is ∝ `h` *by construction* and of course it vanishes under refinement. That makes it a precise
**implementation** control and not a general claim about source-only tiers: it is exactly what batch 4
degrades to if the cut is omitted or clobbered (§6.2), so the divergence is what proves **the cut is
load-bearing and cannot be quietly dropped** — at 3× refinement, omitting it is a 45× error that every
ledger calls green. A physically-motivated dipole source would hold its moment fixed and would not
vanish; that is a different object and not what this control stands in for.

(`t50` is used here as a *contrast* between mountings, never as a radiation figure; see §7.7.)

### 7.7 Radiation efficiency needs a prescribed-velocity rig — and this is the batch's methodological finding

`radiated_energy` and `t50` both count the **reactive near field** as though it had left. Batch 4's
channel is dominantly reactive where batch 3's was not: over a lossy-room piston run,
`radiated_energy` has **50.9%** negative increments for the dipole against **0.0%** for the baffled
plate, peaking at 0.180 of `E0` and settling to 0.136. So a decay time and a radiated fraction can
disagree about direction, and in this prototype they did — which is why neither ships as a radiation
measure.

Batch 3 arrived at the same rule from the other side and wrote it into its even-mode test: a ranking
"belongs only to a prescribed-velocity rig where frequency is a knob." Batch 4 adopts it. Prescribe a
uniform piston on the same `T`, run whole cycles so the reactive part integrates out, and read
`room.injected` — the radiation resistance:

```
    ka       0.275   0.412   0.605   0.907   1.374   2.061
    baffled  2.94    5.39    8.97    13.1    17.0    17.9      (R, kg/s)
    dipole   0.167   0.409   1.57    12.6    63.9    83.8
    dip/baf  0.057   0.076   0.175   0.961   3.749   4.691
```

**Read the top of that table against a physical ceiling before believing it.** At high `ka` a baffled
piston tends to `R → ρ₀c₀A` and a two-sided one radiates plane waves from *each* face, so
`R → 2ρ₀c₀A`: **the ratio's ceiling is 2.** The measured 3.75 and 4.69 are above it, so the dipole
arm is contaminated at the top of the sweep — a mid-room source in a 1.24 m `ζ = 1` box sees a modal
input impedance that a wall-mounted source does not, and `room.injected` over whole cycles is the real
part of *that*, not a free-field radiation resistance. The baffled column corroborates that reading
rather than undermining it: it has the classic `(ka)² →` constant shape and saturates, which is what a
sane arm looks like.

So the assertion is neither the exponent (`ratio/(ka)²` wanders 0.45 … 1.99 — a modal box, again) nor
the magnitudes. It is the **crossing**: the ratio is `≪ 1` at low `ka` and `≥ 1` by `ka ≈ 0.9`, so no
constant reproduces it. The build re-measures this properly — tone bursts at three or four centre
frequencies in §7.8's larger rigid room, `|p|²` integrated over the same quarter arc inside the
window, per batch 1's "windowing, never absorption" doctrine — and there the `≤ 2` ceiling becomes a
**pass criterion** rather than a warning sign. Whether the low-`ka` slope tightens toward `(ka)²` is
reported, not asserted.

**Explicitly refused, twice over: any ranking of plate modes by radiated energy.** Batch 3 refuses it
because a plate mode locks fineness to frequency; batch 4 must refuse it again because the observable
itself is half reactive. A prototype attempt produced `penalty = 1.0061, 0.9997, 0.5987` at one
resolution and `0.9920, 1.0002, 1.0014` at another — i.e. nothing.

### 7.8 Directivity — the claim that survives every caveat above

Windowed free field (rigid 3.3 m cube, read truncated before the first image arrives), normalized
peak `|p|` on a quarter arc from on-axis to in-plane at `r = 0.90 m`:

```
    theta      0     15     30     45     60     75     90
    dipole   1.000  0.924  0.839  0.599  0.448  0.163  0.034
    cos      1.000  0.966  0.866  0.707  0.500  0.259  0.000
    baffled  0.935  0.896  1.000  0.880  0.913  0.786  0.793
```

A **29× in-plane null** against a baffled plate that has none. The assertion is the null and the
monotone decrease, not agreement with `cos θ` to a tolerance, and three separate things forbid a
tighter claim:

* **the angles above are the ones requested, not the ones the grid gave.** `AirBox.snapped`'s own
  docstring forbids exactly this — "an oracle that compares against the *requested* `r` silently
  charges that error to the physics" — and at `r/h = 10.9` the snap is up to `h/2` in radius and moves
  `θ` with it. The build must report `snapped()` and the **actual** `θ` each probe landed on. This is
  the likeliest reason the baffled row scatters ±12% and peaks at 30° rather than on-axis.
* the source is not compact at this radius (`a = 0.15 m`, `r = 0.90 m`), which is why 45°/60° sit
  above the cosine;
* the baffled arm's `θ = 90°` probe sits **on the `z0` wall** — a half-weight node on the baffle
  surface, not the free-space counterpart of the dipole's in-plane point. The two rows' end columns
  are not the same measurement and the table must not imply they are.

Note what directivity does *not* distinguish: the phantom's pattern is the same shape (in-plane
0.049) at 4–5× smaller magnitude. **Directivity identifies the dipole; the blockage sets its
strength.** Both tests are needed and they are testing different things.

### 7.9 Reductions and refusals

* `T = 0` (a zero-area surface) → the bare `Plate`, **bit-identically**, cut or no cut [`0.000e+00`
  over 200 steps]. Structural zeros eliminated before factoring, as batch 3.
* The phantom → two `AirBox.inject` monopoles, **bit-identically** [`0.000e+00`, §6.7].
* `SurfacePort` after the shared-spreading refactor → **bit-identical** to before it.
* Refusals, each with its own test: a cut plane on a wall; an in-plane stencil reaching the room's
  rim; a surface touching an `open` face; a patch sharing pressure nodes with an existing port; a
  patch sharing **cut faces** with an existing port or a manual `add_cut` (§6.2); a sample-rate
  mismatch; an unknown plane or spreading name.

### 7.10 The chain, and the guard

`string → StringPlateBridge → RoomSuspendedPlate → AirBox` runs and conserves, with **no edit to
`connection.py`**, and the stability margin comes out **bit-identical** to the bare plate's
[0.2052148342611817 supported, 0.20521483773139457 free] — §6.6, which also corrects batch 3's
prediction that it would not.

---

## 8. Cost budget — owned, not discovered

* **The cut** is one fancy-index assignment per step over the cut faces — `O(patch)`, no allocation
  if the mask is preallocated. It is applied inside `_momentum`, which is the single place both
  `step()` and `set_state()` produce velocities, so a cut room can never hold a live velocity on a
  cut face at any half-step including the consistent start.
* **The load** has batch 3's sparsity pattern exactly (same `T`, same `R`, scaled by 2), so its
  fill-in in the plate's `splu` is batch 3's — `nnz_growth` and `lu_nnz` are reported by the class
  for the same reason batch 3 reports them (stored entries and actual fill say different things).
* **The port** costs two `free_pressure` reads per step instead of one, both `O(patch)`.
* **The suite** runs on ~12³ grids where the modal oracles are free. §7.7's prescribed-velocity
  sweep and §7.8's 40³ windowed room are **diagnose-script** work, not suite work — batch 1's rule
  that "genuinely audio-rate rooms belong in a diagnose script" applies to a 3.3 m cube too.
* The refinement sweeps in §6.4, §7.6 and §5 are diagnose-script work for the same reason; the suite
  asserts the *direction* of §7.6's divergence at two resolutions, not the whole sweep.

---

## 9. Deliverables

1. `physsynth/core/airbox.py`: `PLANES`, `AirBox.add_cut` / `cut_faces` (additive mask, applied in
   `_momentum`), the shared spreading/refusal helpers, `InteriorSurfacePort`, `RoomSuspendedPlate`;
   `__all__` updated. **No edit to `plate.py` or `connection.py`.**
2. `tests/test_airbox_cut.py` — the cut primitive: energy, the half-offset modal oracle, full vs
   partial isolation, additivity, the refusals. Parametrized over three orientations.
3. `tests/test_airbox_dipole.py` — the coupling: §7.3–§7.6, §7.9, §7.10, both negative controls.
4. `tests/helpers.py` — `make_cut_room`, `make_suspended_plate`, mirroring batch 3's helpers.
5. A test pinning `SurfacePort`'s bit-identity across the shared-spreading refactor.
6. `scripts/diagnose_airbox_dipole.py` — the prescribed-velocity resistance comparison, re-measured
   as **windowed tone bursts in the large rigid room** with the `≤ 2` plane-wave ceiling as a pass
   criterion (§7.7); the directivity arc reporting **snapped** radii and actual angles (§7.8); the
   `t50`-by-mounting refinement (§7.6); the blockage-overshoot convergence (§6.4) **including a
   mode-shaped motion, not only a piston**; and the `k`-only refinement (§5).
7. **Corrections to the record**: `tests/test_airbox_surface.py::test_string_bridge_plate_room_chain`'s
   docstring and `docs/dev/air-box-area-coupling-plan.md` §6.8 / §10.6, both of which predict that
   batch 4's face cut makes the load non-dissipative. It does not (§6.6).
8. `HANDOFF.md` §12H: batch 4 shipped, and what batch 5 is.
9. This plan's own §10, the post-build record.

---

## 10. What the build changed — the post-build record

**Status: SHIPPED (2026-08-10).** `physsynth/core/airbox.py` gains `PLANES`, `AirBox.add_cut` /
`cut_faces`, a private `_PatchPort` base holding the spreading operator and four refusals,
`InteriorSurfacePort` and `RoomSuspendedPlate`. No edit to `plate.py` or `connection.py`.
`tests/test_airbox_cut.py` (42) and `tests/test_airbox_dipole.py` (56) ship alongside
`scripts/diagnose_airbox_dipole.py`. Suite 1355 → **1453**, green.

The plan's structure survived essentially intact — the cut *is* the only new machinery, the load
*is* `2 TᵀRT`, the half-offset oracle *is* exact, and §6.1's "neither ledger catches a wrong load
coefficient" reproduced with both controls. What follows is only where measurement in the build
disagreed with the plan's prototype.

### 10.1 The pass criterion of §7.7 is wrong, and the right one is per-arm

§7.7 proposed **`ratio ≤ 2`** as the criterion that would separate an honest sweep from the
prototype's room-contaminated 3.75 and 4.69. Re-measured properly — windowed tone bursts in a 5 m
rigid room, whole cycles, truncated before the first reflection reaches the source — the sweep is
clean and the criterion still fails:

```
    ka       0.80   1.00   1.30   1.70   2.20   2.80
    baffled  0.281  0.406  0.595  0.797  0.915  0.905     (R / rho0 c0 A)
    dipole   0.078  0.231  0.797  1.798  2.117  1.778
    dip/baf  0.278  0.569  1.339  2.257  2.314  1.965
```

The **ratio** reaches 2.31 (2.49 across the refinement) — over the ceiling — while each **arm** is
where it should be: the baffled column has the textbook `(ka)² →` constant shape and rises
monotonically toward 1, and the dipole tops out at 2.12–2.30. The ratio exceeds 2 because the
baffled arm has not saturated yet, and a piston legitimately overshoots its own asymptote near its
first maximum. So the criterion belongs on **each arm separately, asymptotically**, and the
baffled arm's shape is what makes the dipole arm believable at all. The prototype's 3.75/4.69 were
still contaminated; the diagnosis was right and the test it proposed was not.

**The crossing, which §7.7 correctly identified as the claim, is at `ka` between 1.0 and 1.3** and
survives every refinement.

### 10.2 The dipole arm's magnitude does not converge, and the reason is the obstacle

New, and the sharpest correction here. Under air-grid refinement at fixed room and plate:

```
    h_air     0.060  0.050  0.040  0.030
    baffled   0.405  0.406  0.409  0.412     (ka = 1.0, R / rho0 c0 A)
    dipole    0.279  0.231  0.360  0.225
    blocked   1.44   1.36   1.78   1.44      (blocked_area / plate area)
```

The dipole arm tracks **`blocked_area`, not `h`** — at low `ka` its magnitude is set by the
obstacle, and the obstacle is a rounding of the footprint onto the air grid, so it does not
converge. The baffled arm, whose source uses the *same* `T`, converges smoothly. Only the ratio's
sign against 1 is a claim.

### 10.3 §6.4 was right about the fraction and wrong about everything else

§6.4 reported the overshoot as worth "**±3% while the blocked area moves 33%**" and asked for a
re-measurement with a mode-shaped motion. Both done, and the two probes **disagree**:

```
    free / piston      blocked  2.27  1.53  1.31  1.37   (1.73x)
                       t50/ms   0.500 0.375 0.375 0.344  (1.45x)
                       rad/E0   0.4647 0.4695 0.4615 0.4764  (1.03x)
    supported / (2,1)  blocked  1.51  0.93  0.92  0.80   (1.89x)
                       t50/ms   1.375 0.562 0.500 0.500  (2.75x)
                       rad/E0   1.0000 x4 — SATURATED, which is not the same as insensitive
```

The **fraction** radiated is insensitive; the **rate** is not, and §10.2's prescribed-velocity `R`
agrees with the rate. §6.4's number was measured on the fraction, which is the one observable that
cannot see the effect. (A mode's fraction saturates outright, so it is reported as saturated rather
than as a result.)

### 10.4 The overshoot ratio's denominator was wrong, and the supported branch goes *below* 1

§6.4 quoted `blocked_area / plate_area` = 1.490, 1.571, 1.352, 1.183 "trending to 1". Measured, that
is the **free** branch only. The supported branch reads 1.513, 0.927, 0.925, 0.799 — **below 1** —
because the cut is the support of `T` and a supported plate's clamped rim is not in `T`. Against
the **live** rectangle, which is the moving surface and therefore the honest denominator, it reads
2.690, 1.647, 1.644, 1.421 and does trend to 1. `blocked_area`'s docstring says so, and the
consequence is worth stating: the free plate is the configuration where the obstacle and the
physical plate coincide, which is also the one a physically suspended object *is*.

### 10.5 The numbers that moved without changing a claim

* **§6.5's sign flip lands at step 8**, not step 7. The plan said the step count is not the lesson;
  it moved, and it still is not. More importantly the flip that is genuinely invisible is the
  **consistent** one (both the `∓q` order and the jump direction) — flip only one and the conserved
  total *does* catch it, which the plan did not distinguish. The detector is the sign of the
  **room's own** pressure on the first step, not the port's `pressure_jump`, which the consistent
  flip leaves bit-identical.
* **§6.3's index range is `1 ≤ index ≤ N − 2`**, not `N − 1`: with `index` a *face* index, both
  straddling node planes are interior only up to `N − 2`.
* **§7.8's null is 85×**, the arc is 1.000/0.928/0.786/0.565/0.347/0.164/0.012, the baffled arc
  reaches 0.530, and the phantom is 5.2× down at the same shape. The plan's caveats all held —
  including that the requested angles are not the ones the grid gives (0.0/14.3/30.3/45.9/59.2/
  74.2/88.8 at radii 1.175–1.222 m).
* **§7.6's ratios are 5.2, 19.3, 40.8** (plan: 5.2, 22, 45) and the free piston's phantom never
  reaches `t50` — the divergence, as asserted.
* **§6.6 confirmed**: margins `0.2061806714931906` supported and `0.2061840079056186` free,
  bit-identical loaded or bare, and they are *batch 3's own* numbers rather than the plan's
  `0.2052…` — the guard never saw either load, so it reports the same margin for both batches.
* **§7.1's energy drifts** are 6.5e-15 (rigid) and 8.0e-15 (lossy) of the acoustic scale, and the
  modal oracle 1.0e-14 … 7.4e-14 field error at 3.3e-16 … 8.2e-16 drift — a decade looser than the
  prototype's, on a broadband IC rather than a single mode.
* **§7.2's partial cut passes 0.83 of the uncut room's peak**, not the prototype's 0.223 of "the
  same source": a half-width partition in a room this size is barely an obstacle. The exact
  `0.000e+00` for the full cut is unchanged, and it is the assertion.

### 10.6 What was scoped out and stayed out

Everything §3 deferred stayed deferred. One item is worth naming because it was *close*: `Membrane`
(#4) and the von Kármán plate (#6) mount on `InteriorSurfacePort` with no port change — the surface
protocol is batch 3's, unedited — but neither is wired up or tested, so neither is claimed.
