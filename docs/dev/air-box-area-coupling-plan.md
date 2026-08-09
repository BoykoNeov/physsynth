# The plate radiates from every node — the distributed area coupling (air-box batch 3)

> **Status: PLANNED.** Every number in §6 and §7 below was measured on a prototype before a line of
> core code was written — the house rule since batch 1. Seven traps were measured; **three changed
> the design**, one of them changing what the batch's own money test *is*, and one of them is a
> correction to a claim batch 2 shipped in a comment.
>
> The three that changed the design, in order of how much they cost to learn:
>
> 1. **The conserved total is structurally BLIND to a wrong coupling constant** (§6.1). It is not
>    merely weak — it is *exactly* as flat with a 17 % error in `R_j` as without it, because each
>    side's ledger telescopes against whatever pressure it used, and the sum of two
>    internally-consistent identities is conserved even when the two disagree with each other.
>    Measured on the **shipped batch-2 code**: the naive `R_room` leaves the scene total drifting
>    6.3e-15 (green) while the two ledgers disagree by **88 %**. So the money test of this batch is
>    not conservation. It is `radiated == injected`, plus a differential per-node measurement of
>    `R_j` read straight off the room.
> 2. **Nearest-node area assignment is the wrong spreading operator, and bilinear is measured
>    better on both counts that matter** (§6.5): zero uncovered footprint nodes at *every* grid
>    ratio (nearest leaves 8 of 12 empty where the plate is coarser than the air), and an interior
>    assigned-area spread of **exactly 0.000** at every refinement, where nearest-node's spread is
>    0.16–0.83 `h_air²` and does not converge.
> 3. **The headline is a ratio, not a located transition** (§7.6). The `ω = c₀k_p` coincidence law is
>    the *infinite* corrugated-plane result; a finite staircased patch in a box radiates from its
>    edges below coincidence, by an amount that is a property of the patch, not of a citable closed
>    form. Measured: the power ratio is **not** monotone in frequency (room modes plus the piston's
>    own response), so the assertions are strict monotonicity in *pattern fineness* at fixed
>    frequency, an endpoint lift, and a *bracket* containing each pattern's own `c₀k_p`.
>
> The fourth finding worth reading before the build: **a consistent sign flip in `T`/`Tᵀ` is
> invisible to every energetic quantity in the batch**, including `radiated_energy`, which comes out
> **bit-identical** (§6.2). `TᵀRT` is sign-invariant. The only detector is the sign of the surface
> pressure on the *first* steps — and the naive six-step read gives the *wrong* answer, because the
> plate's own half period was five steps.

---

## 1. Why — the refusal being discharged

Batch 2's docstring names this batch in its own deferral list, and HANDOFF §12H puts it first in
"rough order of appetite":

> *"Still deferred: … and the distributed **area** coupling (a body radiating from every node rather
> than through one port)."*

Every air node this project has ever built — `AirRadiation`, `RadiatedBody`, `RationalAirLoad`,
`RoomPort` — is a **one-port**. It couples to the resonator through exactly one scalar, the net
volume velocity `U = Σᵢ aᵢ q̇ᵢ`, and hands back exactly one scalar, a pressure. That is the right
shape for a `ModalBody`, which has no geometry to speak of: its radiation weights `aᵢ` *are* its
surface, collapsed to a point. It is the wrong shape for a **grid** resonator, which has a real
surface with a real shape on it, and the difference is not a refinement — it is a whole physical
effect that the lumped tier predicts to be **exactly zero**.

**The claim this batch exists to make, in one line:** *a surface radiates according to the shape of
its motion, not only its net volume displacement — so a mode with zero net volume velocity, which
every lumped tier in this repo says is silent, is not.*

That effect is the **acoustic short circuit**, and it is the reason a plate's radiation is a
frequency-dependent efficiency rather than a resistance. Take any plate mode above the fundamental:
its `+` and `−` regions displace almost exactly cancelling volumes, so `U ≈ 0` and every one-port in
the repo reports silence. What actually happens is that each patch of surface pushes on the air
*locally*, and the cancellation is only as complete as the acoustic wavelength's ability to bridge
the distance between a `+` region and its neighbouring `−` region. Below coincidence the cancellation
is nearly complete and radiation comes from the edges; above it, the surface pattern is supersonic
and radiates freely. A one-port has no length scale on its surface, so it cannot represent any of
this at any `R(ω)` — the same structural argument by which batch 2's delayed echo was beyond the
lumped tier, one tier further in.

This is also the batch that makes the *existing* grid resonators radiate honestly. Models #4, #5 and
#5b have had a `pressure()` read-out since they were built, and that read-out is `Σᵢ areaᵢ u''ᵢ` — a
monopole, i.e. exactly the net volume acceleration, i.e. exactly the quantity this batch shows is
insufficient. The read-out is not wrong; it is the `a → 0` compact limit of it, and this batch is
where the plate stops being a point source.

---

## 2. The physics — the room's instantaneous response is *diagonal*, so the load is `TᵀRT`

The whole batch rests on the same observation batch 2 rested on, used differently. Batch 2 read it as
"the room is a Thévenin source at a node". The stronger reading is:

> **Within a single timestep an injection at a node changes the pressure at that node and nowhere
> else.** So the room's instantaneous response over *any* set of nodes is not merely linear — it is
> **diagonal**. There is no cross-resistance to compute, at any separation, including between nodes
> one cell apart.

Measured, not argued (§6.3): stepping the room twice from an identical saved state, once with `q = 0`
and once with `q = e_j`, the incremental centered pressure at node `j` matches `R_j` to **2.7e-16**
and the response at every *other* node of the patch is **exactly 0.00e+00** — at interior, lossy-wall
and lossy-edge patches alike.

That single fact is what makes a distributed coupling cheap. Let

* `T` be the **volume-weight matrix**, `(n_air × n_plate)`, mapping plate nodal velocities to air
  nodal volume velocities: `q = T v`, with `q` in m³/s;
* `R = diag(R_j)`, `R_j = k ρ₀ c₀² / (2 W_j (1 + β_j))` — batch 2's constant, per node, with the
  wall-closure denominator (§6.1);
* `pbar = pbar_free + R q`, now a **vector** open-circuit pressure.

By reciprocity the *same* matrix transposed carries the pressure back as a nodal force,
`f = −Tᵀ pbar` — the identical structural fact `RadiatedBody` leans on when it uses the read-out
weights `aᵢ` as the coupling weights, and the reason the coupling is passive rather than merely
plausible. Substituting the plate's own linear response into that pair gives the load

```
    f_load = −Tᵀ pbar_free  −  Tᵀ R T · (u^{n+1} − u^{n−1}) / (2k)
```

whose unknown part is `−(1/2k) TᵀRT u^{n+1}`: a **constant, symmetric, positive-semidefinite, sparse
matrix** multiplying the unknown. The plate already solves an implicit SPD system with a prefactored
`splu` every step (`A = (1+σk)I + θk²κ²B`, or the `W`-weighted free-plate form). A PSD addition to an
SPD matrix is SPD. So:

> **The air load folds into the plate's own factorization. Nothing new is solved — the matrix is
> assembled and factored once at construction, and the per-step cost of the room load is two
> sparse matrix-vector products.**

That is this batch's structural payoff, and it is the exact analogue of batch 2's "one division" one
rank up. Indeed **batch 2 *is* the rank-1 case**: put every plate node on one air node and `T`
becomes `w·1ᵀ`, `TᵀRT` becomes the rank-1 `R_room w wᵀ`, and the Sherman–Morrison division is what
you get by inverting `A + rank-1` instead of refactoring. The family's ladder, complete:

| tier | coupling | solve |
|---|---|---|
| `RadiatedBody` | scalar `R` | one division (Sherman–Morrison) |
| `RationalAirLoad` | scalar `R_eff` + one aux state | one division |
| `RoomPort` | scalar `R_room`, rich `p̄_free` | one division |
| **this batch** | **diagonal `R`, matrix `T`** | **absorbed into the resonator's own factorization** |

**Passivity is unconditional, and for a better reason than before.** Batch 2 argued it from
`1 + G R_room ≥ 1`. Here it is a property of the matrix: `TᵀRT` is PSD because `R_j ≥ 0`, so
`A + (k/2ρ)TᵀRT` is SPD for every timestep, every grid, every patch, and can never be singular. The
room's CFL and the plate's unconditional stability are both untouched; coupling them adds no third
condition.

**The energy ledger telescopes exactly**, and the reason is worth stating because it is the one
derivation the whole batch rests on. The θ-average applies to the *conservative* spatial operator,
not to the load: `f_ext` enters the plate's update as a plain force at time `n`, so multiplying the
update by the centered velocity `δ_t· u = (u^{n+1} − u^{n−1})/(2k)` and summing gives

```
    ΔE_plate = k · f_load · δ_t· u  =  −k · pbar · T δ_t· u  =  −k · pbar · q
```

with **exactly** the same `q` that is injected into the room. So `E_plate + ∫pbar·q dt` is conserved
for any `T`, any `R`, and the room's own identity is exact for any injection. Which is precisely why
their sum proves nothing on its own — see §6.1.

---

## 3. Scope — and what is deliberately deferred

**In scope:**

- **`SurfacePort`** — a set of air nodes on one **wall face** of the room, plus the volume-weight
  matrix `T` and the diagonal `R`. Written against a minimal *surface protocol* (nodal coordinates,
  nodal areas, a live-node count) rather than against `Plate`, so model #4 (`Membrane`) and any
  future grid resonator can be mounted later with no port edits.
- **`RoomLoadedPlate`** — a `Plate` (model #5 `"supported"` **and** #5b `"free"`) mounted flush in a
  wall face and loaded by the room over its whole surface. A drop-in for `Plate` by `__getattr__`
  delegation, so `StringPlateBridge` accepts it as the body and the full
  `string → bridge → plate → room` chain comes free, exactly as batch 2 got its chain.
- The **baffled** geometry: the plate lies in a wall, so the rigid wall is the textbook infinite
  baffle and the plate's back face is unloaded. §6.4 shows this needs **no new boundary machinery at
  all** — a soft injection at a wall node *is* the moving-wall condition.
- The energy ledger and the **cross-ledger** identity `radiated == injected` (§6.1), the differential
  `R_j` oracle, the sign oracle (§6.2), and the refusals of §6.
- **N instruments in one room** and mixed tiers: a `RoomLoadedPlate` and a batch-2 `RoomLoadedBody`
  in the same room, provided their node sets are disjoint — inherited unchanged, since disjointness
  is what makes each port's solve exact and the check is already `room._ports`-based.

**Out of scope — say so, do not drift:**

- **The interior, two-sided (dipole) plate.** A real cymbal radiates from both faces and the
  pressure *jump* across it drives it. Doing that properly means putting the plate on a plane of
  velocity **faces** and replacing their momentum update with the plate's motion — cutting the room
  in two, removing those faces from the kinetic-energy sum, and injecting `±q` on the two node
  planes that straddle the cut. It is the natural batch 4, it is a genuinely different object (an
  internal moving boundary, not a source), and mixing it in here would double the batch.
- **Everything batches 1 and 2 deferred stays deferred**: PML and higher-order absorbing boundaries,
  HRTF/ambisonics, scattering objects, non-rectangular rooms, viscothermal absorption, moving ports,
  overlapping ports.
- **A plate not parallel to a wall face**, and a non-rectangular footprint. The staircasing question
  is the membrane batch's lesson and deserves its own measurement.
- **`Membrane` (#4) and the von Kármán plate (#6)** as mounted surfaces. The port is written so
  neither needs port changes, but neither is wired up or tested here. #6 additionally interacts with
  the Picard iteration and must be reasoned about separately.
- **A viewer batch.** Core work, per batch 1, batch 2 and radiation batch 3's precedent.
- **Any absolute radiated-power magnitude** as a suite assertion (§7.6, §8).

---

## 4. API

```python
room = AirBox(L=(1.3, 1.2, 1.0), fs=16000.0, h=0.0413, walls=impedance_from_zeta(4.0))

plate = Plate(Lx=0.45, Ly=0.45, kappa=20.0, rho=0.5, fs=16000.0, N=24, boundary="free")

inst = RoomLoadedPlate(
    plate=plate,
    room=room,
    face="z0",            # which of the six walls the plate is mounted flush in
    origin=(0.40, 0.35),  # the plate's (0, 0) corner, in that face's own two coordinates (m)
)

for n in range(n_steps):
    inst.step(f_ext)      # or bridge.step(), which owns plate.step
    room.step()           # ONE room step, after every port has solved — batch 2's contract
```

`SurfacePort` (constructed by `RoomLoadedPlate`, usable directly for a prescribed-velocity surface):

| member | meaning |
|---|---|
| `nodes` | fancy-index triple of the air nodes the surface touches |
| `T` | sparse `(n_air × n_surface)` volume-weight matrix, `q = T v` |
| `R` | vector `R_j` (Pa·s/m³), the room's diagonal self-response |
| `free_pressure()` | **vector** `p̄_free`, `O(patch)`, read before `room.step()` |
| `inject(q)` | queue the per-node volume-velocity **vector** |
| `net_area` | `Σ_n area_n` — the radiating area, which is *not* `Lx·Ly` (§6.6) |
| `footprint_empty` | air nodes inside the footprint that no surface node reaches (§6.5) |
| `require_ready()`, `reset()` | batch 2's per-port guard and mark, inherited verbatim |

`RoomLoadedPlate` adds `radiated_energy`, `nodal_volume_velocity` (the vector `q`),
`surface_pressure` (the vector `p̄`), and `volume_velocity` (the scalar `Σ q_j`, kept because it is
exactly what the lumped tier would have coupled through — which makes it the natural negative
control rather than a diagnostic).

**Zero edits to `AirBox` are needed to drive it.** The room's existing port queue holds
`(nodes, weights, U)` and computes `w·U` for the injection and `w·p̄` for the booking — both linear
in `w` — so a per-node volume-velocity *vector* goes in as `weights=q, U=1.0` and both come out
exactly right. Only the type comment on `_pending_ports` (which says "normalized") needs relaxing.
`plate.py`, `body.py`, `radiation.py`, `connection.py` and `bore.py` are untouched, as in batch 2.

---

## 5. The discrete scheme

Per step, in this order (the ordering is load-bearing — `p̄_free` must be read from the stored
`u^{n+1/2}` **before** the room advances):

```
1.  p̄_free = port.free_pressure()                       # vector, O(patch)
2.  rhs    = plate's own force-free RHS  +  k²/(ρ_s h²) f_ext
             − (k²/(ρ_s h²)) Tᵀ p̄_free
             + (k /(2 ρ_s h²)) TᵀRT u^{n−1}
3.  u^{n+1} = LU_loaded.solve(rhs)                       # prefactored ONCE at construction:
                                                         #   A + (k/(2 ρ_s h²)) TᵀRT
4.  q      = T (u^{n+1} − u^{n−1}) / (2k)                # per-node volume velocity, m³/s
5.  p̄      = p̄_free + R ⊙ q
6.  radiated_energy += k p̄·q ;  port.inject(q)
7.  (caller) room.step()
```

For `boundary="free"` every `ρ_s h²` becomes `ρ_s`: the lumped mass `W` lives inside `A` and is
divided out by the solve, exactly as `Plate.step`'s own `f_ext` path already does.

Three notes that are not obvious from the algebra:

- **`_accel` needs no correction.** `RoomLoadedBody` had to refresh `body._accel` *after* its rank-1
  correction, because the load was applied post-solve. Here the load is inside the solve, so
  `(u^{n+1} − 2u^n + u^{n−1})/k²` already carries it and `Plate.pressure()` is right for free. The
  one-line override that batch 2 needed is *absent* here, deliberately, and a test should say so.
- **`(1+σk)` is not hand-derived.** The radiation leg's recurring unpinned factor lives inside `A`
  already; there is no separate `G` to get wrong. The dense cross-check still runs at two timesteps
  (§7.3) because `R_j ∝ k` sits beside it.
- **The load matrix's sparsity is the spreading operator's stencil squared.** With bilinear weights
  each air node gathers from the plate nodes in its four surrounding cells, so `TᵀRT` couples plate
  nodes sharing an air node — blocks of about `(h_air/h_plate)²` nodes. At the ratios that matter
  (§6.5) this is a small fraction of `B`'s own 13-point bandwidth, so the factorization does not
  meaningfully thicken. Measure and report `nnz` growth rather than assuming it.

---

## 6. Traps — seven, measured before a line of core code

### 6.1 The conserved total is BLIND to a wrong coupling constant — so it is not the money test

This is the batch's most important finding and it invalidates the instinct the whole project runs on.

The reasoning is three lines. The plate's energy identity telescopes to `−k p̄·q` for **whatever**
`p̄` was used in the force (§2). The room's identity `acoustic + dissipated − injected` is exact for
**whatever** injection it received. So `E_plate + radiated + room.energy()` is the sum of two
*separately* exact identities, and it stays flat even when the pressure the plate was pushed by and
the pressure the room actually developed are **different numbers**. A wrong `R_j` does not leak
energy from the total; it silently creates energy on one side and destroys the same amount on the
other, and the total never notices.

Measured in the prototype (plate rig, `R_j` deliberately 17 % wrong by dropping `1+β`):

| | conserved-total drift | `radiated − injected` |
|---|---|---|
| correct `R_j` | 7.5e-15 | 1.0e-25 |
| `R_j` without `(1+β)` | 1.7e-14 | **13 % of the channel** |

And, because this is a claim about a *shipped* module, measured again on **batch 2's own code** —
free ring-down, port in a corner (`β = 1.559`, so the naive `R_room` is 2.56× too big), 400 steps:

| | scene-total drift / `E₀` | `\|radiated − injected\| / \|radiated\|` |
|---|---|---|
| correct `R_room` | 4.7e-15 | 4.2e-16 |
| naive `R_room` | **6.3e-15 — still green** | **88 %** |

So `tests/test_airbox_port.py`'s header claim — *"Without the factor an interior-port suite stays
perfectly green and a wall-mounted port leaks"* — is **wrong about the mechanism**. Nothing leaks;
the suite stays green in both cases. What batch 2 got *right* is its actual pinning test,
`test_R_room_is_what_the_room_does`, which measures `R_room` **differentially** off the room and never
consults the energy total. The memory's "leaks ~2 % of the run's energy (1.9e-2)" is the *ledger
gap*, not an energy leak. **Correcting that comment is part of this batch's deliverables.**

Consequences for §7, which is why this trap is first:

- The money test is **`radiated == injected`** — two independently computed numbers that agree only
  if `R_j` is exactly right — not the conserved total.
- The conserved total still ships (it catches a genuinely broken scheme, and it is the statement a
  reader wants), but it is documented as **necessary and not sufficient**, with a pointer here.
- The **differential per-node `R_j`** measurement is mandatory, and it doubles as the diagonality
  oracle (§6.3).
- Every conservation assertion must additionally report **how big the channel is** — a conservation
  test on a channel worth 1e-14 of the total passes with the coupling disconnected. Measured
  channel: **75–100 % of `E₀`** in every configuration below, which is what makes the tests
  non-vacuous. (Air loading a light plate is genuinely violent: a 45 g, 0.09 m² plate at 700 Hz has
  a radiation time constant of ~2.4 ms.)

### 6.2 A consistent sign flip in `T`/`Tᵀ` is invisible to every energetic quantity

`TᵀRT` is **sign-invariant**, so flipping the convention leaves the load matrix bit-identical, the
solve bit-identical, and the self-term `R q²` bit-identical. Measured: with `T → −T` the
`radiated_energy` after four steps is `+1.442517e-04` — *bit-identical* to the correct run — and only
the room's field is inverted. Batch 2's open-face port all over again: perfectly conservative,
perfectly wrong.

The detector, measured:

```
   T correct : plate velocity +z (into the room) -> surface p at n=1 = +1.048e+02 Pa
   T FLIPPED : plate velocity +z (into the room) -> surface p at n=1 = −1.048e+02 Pa
```

**The convention, stated so it can be tested:** positive plate displacement is *into the room*, i.e.
along the inward normal of the face it is mounted in, so an outward-moving plate **compresses** the
air at its own surface.

**And read it on the first step.** The naive six-step read reported `−7.6e+01` for the *correct*
`T` — the plate's own half period was five steps, so the velocity had already reversed. A sign test
whose read-out time is longer than half a period of the thing under test measures the sign of the
oscillation, not the sign of the coupling.

### 6.3 The room's response is diagonal — measured as *exactly zero* off-diagonal

The claim §2 rests on, measured differentially: step the room twice from an identical saved state,
once with `q = 0` and once with `q = amp·e_j`, and read `(p̄(e_j) − p̄(0))/amp` straight off the room
for every `j`.

| patch mounting | max rel. error on `R_j` | max off-diagonal / `R_j` |
|---|---|---|
| rigid wall | 2.3e-16 | **0.00e+00** |
| lossy mounting wall (`ζ=3`) | 2.7e-16 | **0.00e+00** |
| lossy wall + lossy edge | 2.7e-16 | **0.00e+00** |

Exactly zero, not small: there is no cross-resistance even between adjacent nodes. This is the test
that makes the diagonal load *provable* rather than plausible, and it is the direct generalisation of
batch 2's differential instrument. Note the second column is the one that would catch a "the port is
really a dense matrix" misconception, and the first is the one that pins `(1+β)`.

**Decision this settles:** patch nodes **keep the mounting wall's impedance**. `AirBox.step` divides
the injection by `1 + β` whether one likes it or not, so `R_j` must say so, and the result is
physically the right object — a piston in an *absorbing* baffle, consistent and passive. At least one
test must therefore mount the plate on a **finite-`Z`** wall, or the factor is unpinned exactly as
the radiation leg's `(1+σk)` was, twice.

### 6.4 The baffled piston needs no new boundary machinery — the wall source *is* the moving wall

A soft injection of `q = A_n·v` at a **wall** node is the moving-wall (piston) boundary condition,
not an approximation of it. The wall node's divergence is `u_face/w_z` with `w_z = h/2`, so a wall
moving at `v` adds `k ρ₀c₀² v/(h/2)`; a soft source of `q = h²v` into `W = h³/2` adds
`k ρ₀c₀² (h²v)/(h³/2)` — the same number.

Measured against an independent moving-wall implementation (a subclass whose `_divergence` carries the
wall velocity and which books `k·p̄·A·v` directly), 60 steps, random per-node velocities:

| mounting wall | max \|Δp\| (field scale 0.2–0.3) | \|Δinjected\| | \|Δenergy\| |
|---|---|---|---|
| rigid | 2.4e-16 | 1.7e-24 | 8.3e-25 |
| mixed rigid/lossy | 2.2e-16 | 5.0e-24 | 4.1e-24 |
| lossy `z0` | 6.9e-17 | 4.1e-24 | 5.8e-24 |

≈1e-15 **relative**, fields and both energy books — but **not bit-identical**, because the two
expressions divide by `W` and by `w_z` in different orders. Claim machine precision, not bit
identity; this repo's bit-identity claims are load-bearing and this one cannot be cashed.

That it also holds on a **lossy** wall is what licenses §6.3's decision: the injection and the wall's
own absorption are divided by the same `1 + β`, consistently.

### 6.5 Nearest-node assignment is the wrong spreading operator — bilinear, measured

`T` must (a) conserve volume exactly, (b) be usable as `Tᵀ` for the pressure, and (c) leave no air
node in the plate's footprint unfed — an unfed node makes the source a comb at the grid scale.
Nearest-node assignment satisfies (a) and (b) and fails (c) badly. Bilinear spreading — each plate
node's area distributed over the four surrounding air nodes with weights summing to 1 — satisfies all
three. Measured on a 0.3 m plate, `h_air = 8.25 cm`, footprint 12 face nodes:

| `N_plate` | `h_plate/h_air` | empty (nearest) | empty (bilinear) | interior spread (nearest) | interior spread (bilinear) |
|---|---|---|---|---|---|
| 3 | 1.212 | 8 | **0** | 0.000 | **0.000** |
| 4 | 0.909 | 6 | **0** | 0.826 | **0.000** |
| 6 | 0.606 | 0 | **0** | 0.367 | **0.000** |
| 8 | 0.454 | 0 | **0** | 0.620 | **0.000** |
| 16 | 0.227 | 0 | **0** | 0.258 | **0.000** |
| 24 | 0.151 | 0 | **0** | 0.161 | **0.000** |

Two things to read off it. First, **`h_plate ≤ h_air` is not the refusal condition**: at ratio 0.909
— comfortably inside the inequality — nearest-node still leaves half the footprint unfed. The refusal
must **count empty footprint rows**, batch 2's form (which counted nodes in the ball rather than
testing an inequality on the radius). With bilinear it will rarely fire, and it still ships, because
a plate coarse enough to skip whole air cells is a real thing to refuse.

Second, bilinear's interior assigned area is **exactly** `h_air²` at every ratio — spread `0.000`,
a partition-of-unity consequence — while nearest-node's spread is 0.16–0.83 `h_air²` and **does not
converge with refinement**. That is a lumpy source at the grid scale for no reason. Bilinear costs
~10 lines and is what ships.

### 6.6 The radiating area is not `Lx·Ly`, and the shortfall is exactly the clamped rim

Measured: total assigned area / `Lx·Ly` = 0.5625, 0.7656, 0.8403, 0.8789, 0.9184 at
`N_plate` = 4, 8, 12, 16, 24 — identical for both spreading operators, and exactly `((N−1)/N)²`
(0.7656 = `(7/8)²`). The simply-supported plate's rim nodes are *dead*: they do not move, so they
displace no volume. The shortfall is **physics, not a defect** — but it means `net_area` must be
reported (`RoomPort.volume`'s precedent), and any comparison against a closed form for a piston of
area `Lx·Ly` is wrong by that factor at coarse `N`. The free plate has no dead rim and should come
out at exactly 1.0 — which is a test.

### 6.7 The per-node `p̄_free` read is bit-identical to the full-array closure — *unless* an open face
is touched

Batch 2 added the scalar version of this check on review, because an off-by-one in the local read
survives every energy test (port and room would still agree). On a whole **wall plane** it is more
exposed, not less. Measured over the entire `z0` plane against the full-array
`_divergence()`-then-closure:

| walls | max \|local − full\| | bit-identical |
|---|---|---|
| rigid | 0.0 | **yes** |
| all lossy `ζ=4` | 0.0 | **yes** |
| lossy `z0` + **open** `x0` | **4.3e-04** | no |

The disagreement is at the nodes the plane shares with the pressure-release face: `AirBox.step` pins
`p = 0` there and the local read does not. Batch 2 already refuses a port touching an open face, for
the reason that such a port is silent and the energy report is blind to it. **This is a second,
independent reason for the same refusal** — inherit it verbatim, and the local read is then
bit-identical by construction rather than by luck.

---

## 7. Oracles — what must pass (prototype numbers in brackets)

### 7.1 The money test: the two ledgers agree

`|radiated_energy − room.injected|` ≤ 1e-12 × `|radiated_energy|`, across both plate boundaries, and
across rigid / all-lossy / lossy-**mounting**-wall rooms. Two numbers computed from opposite sides of
the same terminal — the port from its predicted `p̄`, the room from its own post-closure field —
agreeing only if `R_j` is exactly right. [prototype: 1.0e-25 absolute, vs 13 % of the channel with
`(1+β)` dropped]

### 7.2 The differential `R_j`, and the exactly-zero off-diagonal

§6.3's table, as a test: per-node `R_j` to 1e-12 relative, and the off-diagonal response asserted
`== 0.0` exactly. Includes a finite-`Z` mounting wall, and an **edge** patch where `β` sums two
admittances. [2.7e-16 · 0.00e+00]

### 7.3 Conservation, with the channel size asserted alongside

`plate.energy() + radiated_energy + room.energy()` flat to 1e-12 relative — **and** the coupling
channel asserted to be a real fraction of `E₀`, so the test cannot pass on a disconnected coupling.
Both boundaries × {rigid, all-lossy, lossy mounting wall}; monotone non-increasing with `σ_plate > 0`.
[drift 2.6e-15 – 7.0e-13; channel 0.75–1.00 of `E₀`; `E_end/E₀ = 0.9785` at `σ = 3`]

### 7.4 The dense coupled cross-check, at two timesteps

One step of the loaded plate against an explicitly assembled dense coupled system
`[A + (k/2ρ)TᵀRT] u = rhs`, at a small grid, with `σ > 0` and a lossy mounting wall so nothing is
invisible — run at **two different timesteps**, radiation batch 3's fix, because a wrong-but-
consistent `k`-dependent factor passes at one. This is the test that pins the `ρ_s h²` versus `ρ_s`
branch difference between the supported and free plates.

### 7.5 The sign oracle

For a plate given a uniformly positive (into-the-room) velocity, the mean surface pressure at
**`n = 1`** is positive; and with `T` negated it is the negative of that, while `radiated_energy` is
bit-identical. Asserting the bit-identity of the *wrong* run is the point: it records that no
energetic quantity can catch this. [+1.048e+02 vs −1.048e+02 Pa; radiated identical to all digits]

### 7.6 The headline: the acoustic short circuit

Rig: a `SurfacePort` driven at **prescribed** surface velocity (no plate — the `G = 0` rigid-piston
trick radiation batch 3 used for its impedance sweep, so no extra machinery). Patterns are square
waves of period `p` nodes with the **uniform component projected out** (so `Σ q_j` is exactly `0.0`,
not approximately) and then **rms-normalised** (so "equal rms surface velocity" is exact). Radiated
power is the time-average of `Σ_j p̄_j q_j` over whole periods.

Three assertions, in descending robustness:

1. **The lumped tier predicts zero.** `Σ_j q_j` is zero to rounding for every non-uniform pattern by
   construction — measured `0.0` exactly for three of the four patterns and `6.8e-21` for the
   fourth, against `2.5e-04` for the piston, so the assertion is a relative one (`< 1e-15` of the
   uniform pattern's `U`) and not an `== 0.0`. Hence `AirRadiation`, `RadiatedBody`,
   `RationalAirLoad` and `RoomPort` all report silence while the distributed port reports a definite
   nonzero power. This is the batch's claim, and it is structural rather than numerical.
2. **At fixed frequency below coincidence, radiated power falls strictly monotonically as the
   pattern gets finer.** [150 Hz: 1.00, 0.280, 0.0753, 0.0346, 0.0106 · 300 Hz: 1.00, 0.535, 0.148,
   0.0684, 0.0214 — strict in both]
3. **The suppression lifts with frequency, and the lift grows with fineness.**
   [ratio(3800 Hz)/ratio(150 Hz) = 3.9× · 16.5× · 46.4× · **138.9×** for periods 12, 6, 4, 2 nodes]

**And what must NOT be asserted.** The ratio is **not monotone in frequency** (measured: room modes
and the piston's own response make it wander by tens of percent), so no monotone-in-`f` test. The
coincidence law *is* visible — each pattern's ratio crosses unity in the interval bracketing its own
`f_c = c₀/(2 p h_air)`: predicted 346, 693, 1039, 2078 Hz, observed crossings in 300–600, 600–1000,
1000–1500, 1500–2200. Four patterns, four brackets, one closed form. **Assert the bracket, never a
located knee** — locating it needs a room big enough and a grid fine enough that the number stops
being a property of the patch. Absolute radiated power is a diagnose-script figure only (§8).

Rig constraint worth writing down: the pattern period must not exceed the patch node count, or the
projection annihilates the pattern and the rms normalisation divides by zero (a `NaN`, met once).

### 7.7 Reductions and refusals

- **Volume conservation of `T`, exactly:** `Σ_j q_j == Σ_n area_n v_n` to machine precision — the
  identity that makes the lumped monopole the low-frequency limit of the distributed port. Plus
  `Tᵀ1 == areas` (every plate node's area fully distributed) and `net_area == ((N−1)/N)² Lx Ly` for
  the supported plate, `== Lx Ly` for the free plate (§6.6).
- **The rank-1 reduction:** a surface port collapsed onto a single air node, against a hand-assembled
  rank-1 Sherman–Morrison solve on the *same plate*, to machine precision. **Not** a bit-identity
  claim — different code paths — per the house rule about not spending claims that cannot be cashed.
- **No `_accel` correction is needed** (§5): `Plate.pressure()` on a loaded plate equals
  `Σ area·(second difference)` of the *loaded* state, with no post-solve refresh.
- **`A_loaded` is SPD** and `TᵀRT` is symmetric to machine precision; `nnz` growth over `A` reported.
- **Refusals**, each measured rather than argued: a sample-rate mismatch; a footprint with unfed air
  nodes (with the count in the message, §6.5); a patch touching an **open** face (§6.7, batch 2's
  refusal, now with two reasons); a patch overlapping an existing port; a plate whose footprint
  falls outside the face; a port solved twice without `room.step()`.

---

## 8. Cost budget — owned, not discovered

The structural, differential, reduction and sign oracles are all **grid-size-independent** and run on
the ~10×8×6 rooms where batch 2 put its own structural suite; they are effectively free.

The headline is the only test with a real bill, and it was sized before the build:

| configuration | cost | verdict |
|---|---|---|
| `fs=8000`, `N=(9,8,7)`, patch 6, 500 steps | 0.20 s / 5 runs | too small — patch cannot hold a period-12 pattern |
| **`fs=8000`, `N=(12,11,9)`, patch 8, 700 steps** | **0.31 s / 5 runs** | **ships: monotonicity clean** |
| `fs=12000`, `N=(18,17,15)`, patch 10, 800 steps | 0.53 s / 5 runs | ships if the margins want widening |
| `fs=16000`, `N=(32,29,24)`, patch 12, 8 freqs × 5 patterns | **32 s** | **diagnose script only** |

So assertion 7.6.2 costs ~0.3 s and 7.6.3 about twice that (two frequencies). The 8-frequency sweep
that produces the coincidence brackets is a **`scripts/diagnose_airbox_surface.py` figure**, with a
coarse two-or-three-frequency version in the suite. Nothing here needs an audio-rate room, and the
batch should stay under ~10 s total — batch 2's cross-tier sweep alone was ~7 s of its 8.9 s.

---

## 9. Deliverables

- `physsynth/core/airbox.py`: `SurfacePort`, `RoomLoadedPlate`, and the surface protocol they are
  written against. No new module — this is the air box's third batch and belongs beside its port.
- **The only edits outside it**: the `_pending_ports` type comment (§4), and the corrected header
  claim in `tests/test_airbox_port.py` plus `RoomPort.R_room`'s docstring paragraph (§6.1) — a
  documentation correction to a shipped module, measured, not a behaviour change.
- `tests/test_airbox_surface.py` (structural: the two ledgers, differential `R_j`, diagonality,
  dense cross-check, sign, reductions, refusals) and the physical assertions of §7.6. Helpers
  `make_room_loaded_plate` and a prescribed-velocity `surface_drive` in `tests/helpers.py`.
- `scripts/diagnose_airbox_surface.py`: the radiated-power-versus-pattern-fineness sweep with the
  coincidence brackets marked, the energy-channel flat-total figure, and the surface pressure field.
- Docs: HANDOFF §12H updated (batch 3 shipped, the two-sided dipole plate named as batch 4), this
  plan's status block rewritten with what the build changed, and the memory mirror synced.
