# The plate radiates from every node — the distributed area coupling (air-box batch 3)

> **Status: BUILT (2026-08-09).** `SurfacePort` + `RoomLoadedPlate` shipped in
> `physsynth/core/airbox.py`, with **zero edits** to `AirBox` itself, `plate.py`, `connection.py`,
> `body.py`, `radiation.py` or `bore.py` — as §9 promised. Suite **1355** green, was 1287; the batch
> adds **68 tests**, all of them in `tests/test_airbox_surface.py`, and that arithmetic closes
> exactly (1355 − 1287 = 68), so nothing anywhere else moved.
>
> **§10 is the post-build record: what shipped, and the four plan claims the build measured
> false.** Read it before trusting a number in the body of this document. Three of the four are in
> §6.5 and §7.6.4, and they all failed in the same direction — the prototype measured them in a
> configuration where the effect it was looking for was hidden. The sites are marked inline.
>
> Everything below this line is the **pre-build** prediction, kept as written. Every number in §6
> and §7 was measured on a prototype before a line of core code existed — the house rule since
> batch 1. Eight traps were measured; **four changed the design**, one of them changing what the
> batch's own money test *is*, one of them changing a default in the API, and one of them is a
> correction to a claim batch 2 shipped in a comment.
>
> The four that changed the design, in order of how much they cost to learn:
>
> 1. **The conserved total is structurally BLIND to a wrong coupling constant** (§6.1). It is not
>    merely weak — it is *exactly* as flat with a 17 % error in `R_j` as without it, because each
>    side's ledger telescopes against whatever pressure it used, and the sum of two
>    internally-consistent identities is conserved even when the two disagree with each other.
>    Measured on the **shipped batch-2 code**: the naive `R_room` leaves the scene total drifting
>    6.3e-15 (green) while the two ledgers disagree by **88 %**. So the money test of this batch is
>    not conservation. It is `radiated == injected`, plus a differential per-node measurement of
>    `R_j` read straight off the room.
> 2. **Nearest-node area assignment is the wrong spreading operator — and the argument that decides
>    it is symmetry, not lumpiness** (§6.5). **[CORRECTED by the build — §10.1 and §10.2. The
>    conclusion stands and the reason inverts: symmetry does *not* discriminate the two operators,
>    coverage does, and bilinear's equivariance is not offset-independent.]** The coverage numbers
>    came first (zero uncovered footprint
>    nodes at every grid ratio against nearest-node's 8 of 12; interior assigned-area spread exactly
>    `0.000` at every refinement against 0.16–0.83 `h_air²` that never converges) and they are merely
>    aesthetic. The decisive one: bilinear spreading is **reflection-equivariant** and nearest-node is
>    not — symmetry defect of `TᵀRT` measured at **1e-15 versus 0.42–0.73** — and the monopole that
>    leaks out of an even plate mode tracks that defect one-for-one, **1e-15 versus 18 %**. Under
>    nearest-node the batch's own headline claim holds at `t = 0` and dissolves within 200 steps.
>    Bilinear's equivariance holds at **every** plate offset (measured at eight, flat at 9e-16) for a
>    stateable reason: `R_j` is uniform across a face's interior, so `TᵀRT = R·TᵀT` and only relative
>    offsets enter. Nearest-node's defect is flat too — its failure is not an alignment matter either.
> 3. **The load matrix is only half the coupling, and the even mode's silence is a property of the
>    whole SCENE** (§7.6.4) — the finding that came closest to shipping as a false claim, and it was
>    found by asking whether §6.5's measurement was conditional on the plate being centred. It is, but
>    not for the expected reason. The incoming `Tᵀ p̄_free` is the **room's** field, so the
>    antisymmetric subspace survives only if the room is mirror-symmetric about the mode's own
>    antisymmetry plane. A **perfectly centred** plate in a room made asymmetric *in x* (lossy `x0`,
>    rigid `x1`) leaks **3.5e-02** — the largest figure in the study — while the same asymmetry in *y*
>    leaves a `(2,1)` mode at 3.0e-15. Six different room widths with the plate centred, spanning two
>    grid alignments, all sit at rounding: **grid commensurability is not the criterion, centring is.**
>    And there is no tolerance band — the leak is *linear* in the offset over four decades (`δ/h_air`
>    of 1e-6 gives 1.0e-07), so "approximately centred" is not approximately silent. Consequences:
>    `origin` **defaults to centred**, the oracle fixes both halves of the symmetry, and the
>    asymmetric case ships as a **diagnose figure rather than a bug** — a room re-exciting a plate's
>    *shape* is one more thing no `R(ω)` one-port can represent.
> 4. **The headline is a ratio, not a located transition** (§7.6). The `ω = c₀k_p` coincidence law is
>    the *infinite* corrugated-plane result; a finite staircased patch in a box radiates from its
>    edges below coincidence, by an amount that is a property of the patch, not of a citable closed
>    form. Measured: the power ratio is **not** monotone in frequency (room modes plus the piston's
>    own response), so the assertions are strict monotonicity in *pattern fineness* at fixed
>    frequency, an endpoint lift, and a *bracket* containing each pattern's own `c₀k_p`.
>
> The fifth finding worth reading before the build: **a consistent sign flip in `T`/`Tᵀ` is
> invisible to every energetic quantity in the batch**, including `radiated_energy`, which comes out
> **bit-identical** (§6.2). `TᵀRT` is sign-invariant. The only detector is the sign of the surface
> pressure on the *first* steps — and the naive six-step read gives the *wrong* answer, because the
> plate's own half period was five steps. This is also why the API defines positive `u` along the
> **inward normal of the port's face** rather than along a global axis (§7.5): the local convention
> makes `T` entrywise non-negative on all six walls, so the per-face sign that no ledger can catch
> never exists to be got wrong.
>
> And one thing the prototype learned by getting the *direction* of an effect wrong — **and then got
> wrong itself, §10.3**: with a **real plate mode** driving the surface (§7.6.4, the assertion that
> puts a plate into §1's claim), the finer mode radiates **more**, up to 7.1×, not less. A plate mode locks fineness to frequency, so the
> higher mode completes ten times the cycles and its count beats the per-cycle suppression. The
> fineness law belongs to the prescribed-velocity rig where `f` is a knob; the plate-mode oracle
> asserts the exactly-zero net volume velocity and a nonzero power, and ranks nothing.

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
frequency-dependent efficiency rather than a resistance. Take an **even-index** mode of a simply
supported plate — `(2,1)`, `(2,2)`, `(4,2)`: its `+` and `−` regions displace **exactly** cancelling
volumes, `U = 0` to rounding and not merely small, so every one-port in the repo reports exact
silence. (Not *every* higher mode: `(3,1)` keeps a third of the fundamental's net volume, measured in
§7.6.4 — the cancellation is a symmetry statement, not a high-order one, and the plan should say the
true thing.) What actually happens is that each patch of surface pushes on the air
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
  `string → bridge → plate → room` chain comes free — **measured, not inferred** (§6.8), because
  unlike batch 2's `ModalBody` this bridge reassembles the body's operator behind the delegation.
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
    origin=None,          # the plate's (0, 0) corner in that face's own two coordinates (m);
                          # DEFAULTS TO CENTRED, and §7.6.4 is why — an off-centre plate in a
                          # mirror-symmetric room is legal and physical, but its even modes stop
                          # being exactly silent, linearly in the offset with no threshold.
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

**The sign convention is part of the API, and it is local.** Positive `u` displaces the plate along
the room's **inward normal at `face`** — `+z` for `z0`, `−z` for `z1`, and so on. So `T` is entrywise
non-negative on all six walls, `TᵀRT` is PSD by inspection, and the per-face sign that no energy
report can catch (§7.5) does not exist in the code. All six faces are supported and all six are
tested; the alternative convention (positive `u` along the global axis) is measured as the negative
control, because it produces a perfectly conservative simulation of a plate pushing the wrong way.

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

## 6. Traps — eight, measured before a line of core code

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

One modelling caveat to state where the API is documented, so nobody reads the lossy-wall *test* as
the recommended *configuration*: a lossy mounting wall absorbs **through the plate's own footprint**,
i.e. it models a plate that is porous to the wall behind it. That is energy-consistent (§6.4 measured
it) and it is the only way to pin `(1+β)`, but a **rigid** mounting wall — the true infinite baffle —
is the sane default and is what the helpers should default to.

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

> **[CORRECTED by the build — §10.2.]** "Exactly `h_air²` at every ratio" is too strong: it is exact
> only when `h_air/h_surface` is an **integer**, and off an integer it ripples non-monotonically in
> the ratio. Bilinear is still 10×–100× flatter than nearest at every refinement *and* converging
> where nearest wanders, so what this paragraph concludes survives — but the residual ripple is the
> coupling's accuracy floor and this batch shipped knowing its size.

**And then a third measurement, which is the one that actually decides it.** Lumpiness is an
aesthetic argument; symmetry is not. Bilinear spreading is **reflection-equivariant** — mirror the
geometry and the four weights mirror with it — while nearest-node rounding is not, because ties and
unequal group sizes break one way. Measured as the relative defect
`‖PᵀLP − L‖/‖L‖` of the load matrix under the plate's own `x → Lx − x` permutation `P`, together with
the monopole that leaks out of an even-index plate mode (§7.6.4) over 200 coupled steps:

| spreading | `N_plate` | `h_p/h_air` | symmetry defect of `TᵀRT` | peak leaked \|U\|/A |
|---|---|---|---|---|
| nearest | 8 | 0.454 | 7.3e-01 | 1.7e-01 |
| nearest | 16 | 0.227 | 5.2e-01 | 1.8e-01 |
| nearest | 24 | 0.151 | 4.2e-01 | 1.2e-01 |
| **bilinear** | 8 | 0.454 | **1.3e-15** | **5.3e-16** |
| **bilinear** | 16 | 0.227 | **9.1e-16** | **4.6e-15** |
| **bilinear** | 24 | 0.151 | **1.4e-15** | **1.4e-14** |

The leak tracks the defect one-for-one across fourteen orders of magnitude, and the causal chain is
closed by the one case where nearest-node *accidentally* becomes symmetric: refining the air grid to
`h_air = 2.75 cm` under a fixed `N=16` plate happens to align the assignment, the defect drops to
**exactly `0.00e+00`**, and the leak drops with it to 8.2e-16. Nearest-node's symmetry is an accident
of alignment; bilinear's is not.

> **[CORRECTED by the build — §10.1 and §10.2.]** The table above and the two paragraphs that
> follow it are the plan's biggest miss, and it is a miss of *reasoning*, not of conclusion. The
> shipped measurement finds the coupled monopole leak does **not** discriminate the operators at all
> (centred, an even mode stays at rounding under both), and bilinear's equivariance is **not**
> offset-independent — it needs `S = 2·(surface centre)/h_air` integral. The operator that ships is
> unchanged and the argument for it is now **coverage**; the leak's real determinant is the
> **scene**, which is §7.6.4's subsection below and the thing this plan got right.

**And bilinear's equivariance is genuinely offset-independent, which is worth stating because the
obvious worry is that it is not.** Bilinear is mirror-equivariant only if the plate's mirror maps the
*air* grid to itself, which sounds like a commensurability condition on where the plate sits. Measured
at eight `x` offsets spanning `0 … h_air` in steps of `h/8`, the defect is **9.0e-16 at every one of
them** — flat, no periodicity. The reason: `R_j` is uniform across the interior of a face, so
`TᵀRT = R · TᵀT`, and `TᵀT`'s entries depend only on the *relative* offsets of plate nodes within
their air cells, which mirroring reverses without changing. The load matrix is equivariant wherever
the plate sits. Nearest-node's defect is likewise flat (5.2e-01 at every offset) — its failure is not
an alignment matter either.

Why this outranks the lumpiness argument: a load that breaks the mode's mirror symmetry **mixes the
odd modes in**, and the odd modes are the ones with net volume. Under nearest-node an even plate mode
starts with exactly zero net volume velocity (1.3e-17) and within 200 steps has grown a monopole 18 %
of the plate's own scale — so §1's headline claim would hold at `t = 0` and dissolve while you watch.
Under bilinear the load never breaks the antisymmetric subspace. **But the load is only half of the
coupling**, and the other half turns out to matter more — §7.6.4.

Two consequences of bilinear that must be handled rather than inherited from the prototype:

- **Clipping the stencil at the face boundary is a silent geometric fold — refuse instead.** The
  prototype's `np.clip(i0 + di, 0, N)` means a plate node within one air cell of the face's edge
  dumps its outboard weight back onto the boundary node. The weights still sum to 1, so volume is
  still conserved and **every ledger stays green while the source geometry is quietly wrong** — the
  same failure shape as the sign flip of §6.2. `AirBox.node_index` already refuses to relocate an
  out-of-room point rather than snapping it; match that. **The refusal condition is that the
  footprint plus one air cell lies strictly inside the face**, and it belongs next to the empty-row
  count.
- **The acoustic source is up to one air cell larger than the plate**, because bilinear legitimately
  spreads onto the nodes just outboard of the plate's rectangle. Say so where `net_area` is reported
  (§6.6), and make the disjointness check use the **actual node set**, not the footprint.

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

**And the check must be run on a HIGH face too, because that is a different branch.** The local
divergence read takes `where(idx < N_axis, u[...], 0.0)` on the plus side and the mirrored form on the
minus side; a port on `z1`/`x1`/`y1` exercises the branch a `z0` port never touches, and the six-face
ledger agreement of §7.5 does *not* test it — the port and the room would agree on a wrong value just
as happily. Measured over all six faces after 23 coupled steps, against the full-array
`_divergence()`-then-closure: **`0.0` exactly, bit-identical, on every face, rigid and all-lossy**
(against a `|p̄|` scale of 4.3 – 36 Pa). So the local read is right on the high faces as well — but the
assertion ships, one line per face, because the fact that it *would* have been invisible is the reason
batch 2 added the scalar version on review in the first place.

### 6.8 The bridge computes its stability guard on the **unloaded** plate — measured safe, not assumed

Batch 2 earned its "the chain comes free" claim against `ModalBody`, whose bridge only ever touches
public state and `body.step(force)`. `StringPlateBridge` is a different animal:
`_stability_margin` **reassembles the plate's `G0` block from scratch** out of
`theta, rho, h, kappa, B / W, K` — every one of which a `__getattr__`-delegating wrapper hands over
happily. So the guard would be computed against physics that is not happening, and the delegation
would hide it perfectly. This is exactly the kind of claim that must not ship on inference.

Measured, and the news is good on both counts:

- **The chain runs and conserves.** `E_string + E_plate + radiated + E_conn + room.energy()` over 600
  steps of a plucked string: drift/`E₀` = 2.6e-15 (supported, rigid), 2.2e-15 (supported, lossy),
  2.3e-13 (free, rigid), 2.7e-14 (free, lossy), with the coupling channel at 0.35 % of `E₀` for the
  supported plate and **35 %** for the free one — a hundredfold difference that is itself a result
  (§7.8: the free plate's rigid-body translation is a piston; the supported plate's clamped rim is a
  poor radiator).
- **The guard's blindness is safe, for a stateable reason.** `G0 = M + (θ − ¼)k²S` is a statement
  about **mass and θ-excess stiffness**, and the air load is **dissipative** — it enters `A`, never
  `G0`. So the margin comes out **bit-identical** with a loaded or a bare body (0.57868295 both,
  supported and free). And adding the load block to `G0` anyway *reduces* `(G0⁻¹)_dp` by 0.7 %
  (ratio 0.993108 supported, 0.993159 free), i.e. the true margin is *smaller* than the guard
  reports: **the guard is conservative, so its blindness errs safe.**

Therefore **no `connection.py` edit**, and the plan says why rather than that it happened to work. A
test must pin the bit-identity of the margin, so that a future change making the load non-dissipative
(the two-sided plate of §3, whose face-cut removes air mass) fails loudly here instead of silently
mis-guarding.

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

### 7.5 The sign oracle, parametrized over all six faces

For a plate given a uniformly positive (into-the-room) velocity, the mean surface pressure at
**`n = 1`** is positive; and with `T` negated it is the negative of that, while `radiated_energy` is
bit-identical. Asserting the bit-identity of the *wrong* run is the point: it records that no
energetic quantity can catch this. [+1.048e+02 vs −1.048e+02 Pa; radiated identical to all digits]

**One face is not enough, and the reason is a design decision the plan should state rather than
discover.** A surface port takes a *face*, not a position, and the inward normal is `+axis` at a low
face but `−axis` at a high one. There are two ways to write that:

- **positive `u` displaces along the global axis** — then `T` needs an explicit per-face sign, and
  getting it wrong on three of the six faces is invisible to every energetic quantity;
- **positive `u` displaces along the room's inward normal at the port's face** — a local convention,
  under which `T` is entrywise non-negative on all six faces and **no inward normal appears in the
  code at all**.

The second ships. It is the one that makes `TᵀRT` PSD by inspection rather than by cancellation, and
it disarms the trap instead of testing around it. The oracle is then the *uniformity* of the
convention: mount the same plate on each of `x0, x1, y0, y1, z0, z1` and assert

1. the port's nodes lie on the requested wall (`index[axis] == 0` or `== N[axis]`) — the assertion
   that catches an axis-permutation or a low/high mix-up, neither of which any ledger sees, because a
   port on the wrong wall of a symmetric room is perfectly self-consistent;
2. the ledgers still agree and the total is still flat, on every face [gap 2.7e-20 – 5.4e-20, drift
   6.7e-16 – 4.2e-15];
3. **one sign on all six faces** — measured `+6.824e+01` on the four side walls and `+8.560e+01` on
   `z0`/`z1`, the difference being the node count the plate lands on (20 vs 16) in a room that is not
   cubic, not a sign;
4. the negative control, on a **high** face where the naive implementation differs: taking the sign
   from the global axis instead of the inward normal flips the surface pressure to `−8.560e+01` and
   leaves `radiated_energy` **bit-identical to all thirteen digits** (`2.434414196444e-04`) and the
   drift unchanged at 4.2e-15.

Point 4 is the whole justification for points 1–3 existing: the wrong convention is a perfectly
passive, perfectly conservative, perfectly green simulation of a plate pushing the wrong way.

### 7.6 The headline: the acoustic short circuit

Rig: a `SurfacePort` driven at **prescribed** surface velocity (no plate — the `G = 0` rigid-piston
trick radiation batch 3 used for its impedance sweep, so no extra machinery). Patterns are square
waves of period `p` nodes with the **uniform component projected out** (so `Σ q_j` is exactly `0.0`,
not approximately) and then **rms-normalised** (so "equal rms surface velocity" is exact). Radiated
power is the time-average of `Σ_j p̄_j q_j` over whole periods.

Three assertions on that rig, in descending robustness, and a fourth below on a real plate:

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

#### 7.6.4 The same claim with a real plate mode in it

Assertions 1–3 drive a prescribed velocity pattern with no plate anywhere in the rig, while §1's claim
is entirely about *plate modes*. The gap closes with the one hook available exactly: for
`boundary="supported"` the scheme's modes are the **exact** discrete `sin×sin`, and
`Σ_{i=1}^{N−1} sin(mπi/N) = 0` identically for even `m`. So an even-index mode has **exactly** zero net
volume displacement — not small, zero — and every lumped tier in the repo predicts exact silence from
a mode the distributed port radiates from definitely.

Measured on a real coupled `RoomLoadedPlate` (N=16, equal rms surface velocity, 200 steps, bilinear):

| mode | \|U\|/A at `t=0` | peak \|U\|/A over 200 steps | radiated / (1,1) | `f` (continuum, Hz) | band |
|---|---|---|---|---|---|
| (1,1) | 8.59e-01 | 5.94e-01 | 1.000 | 698 | `< fs/4` ✔ |
| **(2,1)** | **1.26e-17** | **1.86e-15** | 1.398 | 1745 | `< fs/4` ✔ |
| **(1,2)** | **1.81e-17** | **2.49e-15** | 1.398 | 1745 | `< fs/4` ✔ |
| **(2,2)** | **7.79e-18** | **1.16e-15** | 2.134 | 2793 | `fs/4 … fs/2` |
| (3,1) | 2.79e-01 | 2.56e-01 | 2.763 | 3491 | near Nyquist |
| **(4,2)** | **4.75e-18** | **4.38e-15** | 7.100 | 6982 | **above Nyquist** |

> **[CORRECTED by the build — §10.3.]** The `radiated / (1,1)` column of this table is an artifact
> of the air grid it was measured on, and its direction **inverts** once every mode is resolved (per
> cycle at equal rms velocity: 1.000, 0.448, 0.448, 0.260, 0.213, 0.091). The `|U|/A` columns —
> the zero, which is a symmetry statement — hold at every refinement level, exactly as the paragraph
> below predicts they would.

The frequency column is the continuum closed form `f = κπ((m/Lx)² + (n/Ly)²)/2`, and it is there to
mark the band, not as a measured quantity — at `fs = 8000` only the first three rows sit in the
trustworthy `f < fs/4`, `(3,1)` is up against Nyquist and `(4,2)`'s 6982 Hz is *not attainable* at
this sample rate at all. **Which affects the two columns differently, and that asymmetry is the useful
part.** The zero is a symmetry statement and is therefore resolution-independent — `(4,2)`'s 4.4e-15
is as good a number as `(2,1)`'s, unresolved mode or not. The `radiated / (1,1)` column is *not*:
comparing energies across rows means comparing an unresolved mode against a resolved one. So the
suite's assertions live on `(1,1)`, `(2,1)`, `(1,2)` and the zero-only rows; the ranking column and
rows below `(2,2)` are diagnose-script material (§8).

The even rows hold the zero **for all 200 steps, at rounding**. So the assertion is: for an even-index
mode, `max|Σ_j q_j| / Σ areas < 1e-13` over the whole run, **and** `radiated_energy` is a definite
nonzero — bounded below by a real fraction of the (1,1) mode's, so it cannot pass on a disconnected
coupling. `(1,2)` against `(2,1)` on a square plate is a free symmetry check (identical to all quoted
digits).

##### The zero is a property of the whole scene, not of the plate or the spreading operator

This is the finding that came closest to shipping as a false claim, and it is worth the space. §6.5's
equivariance is necessary but **not sufficient**: the load matrix is only half the coupling, and the
incoming `Tᵀ p̄_free` is the *room's* field. The antisymmetric subspace survives only if the whole
scene is mirror-symmetric about the mode's own antisymmetry plane. Measured three ways:

| configuration | load defect | peak \|U\|/A |
|---|---|---|
| plate centred, **six different room widths** (`N_x` = 7…11, two distinct grid alignments) | 9.0e-16 | **1.9e-15 – 2.3e-15** |
| plate off-centre by `h_air/3`, same six rooms | 9.0e-16 | 4.7e-03 – 9.7e-03 |
| plate **centred**, all-lossy or all-rigid room (symmetric) | 9.1e-16 | **1.9e-15 – 4.6e-15** |
| plate **centred**, lossy `x0` against rigid `x1` (asymmetric **in x**) | 9.1e-16 | **3.5e-02** |
| plate **centred**, lossy `x0` **and** `x1` (symmetric in x again) | 9.1e-16 | **3.3e-15** |
| plate **centred**, lossy `y0` only (asymmetric in y, *not* in x) | 9.1e-16 | **3.0e-15** |

Three things fall out of that table, in order of how much they matter:

1. **Grid commensurability is not the criterion.** Six room widths give two different `org/h_air`
   fractional alignments and two different `n_air` (30 and 36), and all six are at rounding. Centring
   is the criterion; where the plate lands on the grid is irrelevant.
2. **The room's own asymmetry is what leaks, and it leaks in the mode's axis only.** A perfectly
   centred plate in a room made asymmetric *in x* leaks 3.5e-02 — the largest figure in the whole
   study — while the same asymmetry in *y* leaves the `(2,1)` mode's zero untouched at 3.0e-15. That
   is the mechanism identified, not merely correlated: the room's asymmetric echoes drive the plate's
   odd modes.
3. **There is no tolerance band.** The leak is *linear* in the offset with no threshold: `δ/h_air` of
   1e-6, 1e-4 and 1e-2 give peak `|U|/A` of 1.0e-07, 1.0e-05 and 1.0e-03 — a clean factor of `0.1·δ/h`
   over four decades. So an "approximately centred" plate is not approximately silent. The oracle must
   be built on an **exactly** symmetric scene, and the assertion is `< 1e-13`, not a loose bound.

Consequences for the build, all three of which would otherwise be discovered by a mysterious failure:

- **`RoomLoadedPlate` defaults `origin` to centred in the face**, and `make_room_loaded_plate` does
  too, each with a comment giving this reason. §4's illustrative `origin=(0.40, 0.35)` is *outside*
  the symmetric regime (measured load defect 2.9e-01 in the room that fits it, versus 3.1e-15
  centred) — legal, physical, and simply not the configuration the plate-mode oracle can run in.
- **The §7.6.4 test fixes both halves**: centred plate **and** wall impedances symmetric in the mode's
  antisymmetric axis. A test that used `walls={"x0": ...}` would fail at the 3.5e-02 level and look
  like a coupling bug.
- **And the asymmetric case is not a defect to hide — it is another thing only this tier can do.**
  A room that is asymmetric about the plate re-excites the plate's *shape*, converting an
  acoustically-silent even mode into a radiating one at the 1–3 % level. No `R(ω)` one-port can
  represent that at all: a lumped port couples through a single scalar and has no shape for the room
  to push on. It belongs in the diagnose script as a figure, with the symmetric case beside it.

> **[CORRECTED by the build — §10.3.]** The `radiated / (1,1)` column of the table above is an
> artifact of the air grid it was measured on, and its direction **inverts** once every mode is
> resolved. What the paragraph below concludes — *do not rank modes by radiated energy in this
> oracle* — is right, and now for a second and better reason than the one it gives.

**And the trap in this table, which is the reason it is a separate assertion from 7.6.2.** The
radiated column goes the *wrong* way: the finer mode radiates **more**, up to 7.1× at (4,2). This does
not contradict 7.6.2 — it is the reason 7.6.2 must hold frequency fixed. A plate mode locks spatial
fineness to frequency (`f ∝ m² + n²`), so a finer mode completes several times the cycles in the same
window and that count beats the per-cycle suppression. **The plate-mode oracle must therefore assert
the exact zero and a nonzero power, and must NOT rank the modes by radiated energy** — that ranking
belongs only to the prescribed-velocity rig where `f` is a knob. A test that "fixed" the direction by
normalising per cycle would be asserting the square-wave result through a mode that cannot isolate it.

Resolution note for the rig: the air nodes under the plate (36 here, a 6×6 patch) bound the surface
pattern the room can resolve, which is a second and independent reason the fine rows are
diagnose-script material.

**And what must NOT be asserted.** The ratio is **not monotone in frequency** (measured: room modes
and the piston's own response make it wander by tens of percent), so no monotone-in-`f` test. The
coincidence law *is* visible — each pattern's ratio crosses unity in the interval bracketing its own
`f_c = c₀/(2 p h_air)`: predicted 346, 693, 1039, 2078 Hz, observed crossings in 300–600, 600–1000,
1000–1500, 1500–2200. Four patterns, four brackets, one closed form. **Assert the bracket, never a
located knee** — locating it needs a room big enough and a grid fine enough that the number stops
being a property of the patch. Absolute radiated power is a diagnose-script figure only (§8).

> **[REFINED by the build — §10.4.]** "Assert the bracket" was right for a better reason than this
> paragraph gives: plotted against `f/f_c` the patterns' curves **collapse**, and each one *peaks*
> at `f/f_c = 1`. The unity crossing sits on the rising flank rather than at the peak, and the
> common `[0.70, 0.85] f_c` bracket the diagnose script reports is exactly **one sweep interval
> wide** — as tight as the frequency grid and no tighter.

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

### 7.8 The free plate's rigid-body translation now radiates

The free plate's stiffness nullspace is exactly `{1, x, y}` (model #5b's own oracle). Bare, that is a
motion nothing resists: give the plate a uniform velocity and it translates forever at **constant**
energy — measured `E/E₀ = 1.0000` at every one of 400, 800, 1200, 1600, 2000 steps. Mount the same
plate flush in a baffle and that identical motion **is a piston**, the most efficient radiator the
geometry has. So the room must take it, and the bare plate is the negative control that makes the
statement mean something.

| configuration | `E_plate/E₀` at 400 … 2000 steps | `radiated/E₀` |
|---|---|---|
| **bare, no room** | 1.0000 1.0000 1.0000 1.0000 1.0000 | — |
| baffled, **lossy** room | 0.0000 0.0000 0.0000 0.0000 0.0000 | **1.0000** |
| baffled, rigid room | 0.0451 0.1585 0.0247 0.0472 0.0268 | 0.9732 |

Two assertions, and one temptation to refuse:

1. **The lossy room takes all of it.** `E_plate/E₀ < 1e-3` by 400 steps and `radiated/E₀ → 1.0000`:
   the nullspace motion is *fully* converted, which no bare free plate and no lumped body-loss
   coefficient can do. This is the assertion, because it is the one that is monotone in the physics.
2. **The bare control does not move at all** — `E/E₀ == 1.0` to 1e-12 over the same 2000 steps. Cheap,
   and it is what turns row 1 from a number into a contrast.

**Do not assert monotone decay in the rigid room.** Measured, the plate drops to 4.5 % and then
*climbs back* to 15.9 % before wandering — a rigid box is closed, so the piston's energy comes back to
it. Only the **total** is monotone there (it is exactly flat, drift/`E₀` = 2.0e-13), and only the
lossy room gives the plate a one-way trip. Asserting the rigid-room drop as a decay would be a test
that passes on the sampling instants and fails on the physics.

This also retires the loose end §6.8 left: the coupling channel being 0.35 % of `E₀` for the supported
plate and **35 %** for the free one is this effect, quantified — the clamped rim is a poor radiator and
the free plate's rigid body is a perfect one, a hundredfold in the same rig.

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

The three oracles added on review are in the free tier too, measured rather than assumed:

| oracle | suite configuration | cost |
|---|---|---|
| §7.5 six-face sign + ledgers | 6 faces × 120 steps, `N_plate=8` | **0.12 s** |
| §7.6.4 real plate modes | 3 modes × 200 steps, `N_plate=16` | **0.12 s** |
| §7.8 free rigid body | 2 rooms + bare control × 600 steps | **0.21 s** |
| §6.5 symmetry defect | two load assemblies, **no time-stepping at all** | **0.01 s** |
| §6.7 six-face `free_pressure` bit-identity | 6 faces × 2 wall sets × 23 steps | **~0.05 s** |
| §7.6.4 scene-symmetry controls | off-centre + asymmetric-room positive controls, 200 steps each | **~0.10 s** |

Two step counts worth fixing here rather than in review. §7.8's 2000-step prototype run is
**unnecessary**: `E_plate/E₀` is already 4.3e-05 at **200 steps** with `radiated/E₀ = 0.999957`, so the
assertion ships at 200–400 and the 2000-step table of §7.8 is a diagnose-script figure. And §7.6.4's
leak bound is **saturated by 100 steps** (peak `|U|/A` = 1.86e-15 at 100, 200 and 400 steps alike), so
200 is already generous — the number is a rounding floor, not a growing quantity, which is the point.

So assertion 7.6.2 costs ~0.3 s and 7.6.3 about twice that (two frequencies). The 8-frequency sweep
that produces the coincidence brackets is a **`scripts/diagnose_airbox_surface.py` figure**, with a
coarse two-or-three-frequency version in the suite. Nothing here needs an audio-rate room, and the
batch should stay under ~10 s total — batch 2's cross-tier sweep alone was ~7 s of its 8.9 s. With the
review additions at ~0.6 s combined, the headline is still the only line item with a bill.

---

## 9. Deliverables

- `physsynth/core/airbox.py`: `SurfacePort`, `RoomLoadedPlate`, and the surface protocol they are
  written against. No new module — this is the air box's third batch and belongs beside its port.
  `SurfacePort` takes a **face**, supports all six, and defines positive `u` along that face's inward
  normal (§7.5) so that no per-face sign exists to get wrong. Bilinear spreading, on the strength of
  §6.5's symmetry measurement rather than its lumpiness one. `origin` **defaults to centred in the
  face**, with §7.6.4's reason in the comment.
- **No `connection.py` edit** — `StringPlateBridge` accepts a `RoomLoadedPlate` as its body unchanged,
  and its guard's blindness to the load is measured conservative rather than assumed harmless (§6.8).
- **The only edits outside `airbox.py`**: the `_pending_ports` type comment (§4), and the corrected
  header claim in `tests/test_airbox_port.py` plus `RoomPort.R_room`'s docstring paragraph (§6.1) — a
  documentation correction to a shipped module, measured, not a behaviour change.
- `tests/test_airbox_surface.py`:
  - structural — the two ledgers (§7.1), differential `R_j` and exact-zero off-diagonal (§7.2),
    conservation with the channel size asserted alongside (§7.3), the dense cross-check at two
    timesteps (§7.4), the reductions and refusals (§7.7);
  - the **six-face** sign oracle with its global-axis negative control (§7.5);
  - the load matrix's **reflection equivariance** under bilinear spreading, with nearest-node as the
    measured negative control (§6.5), asserted **offset-independently** so the reason it holds
    (`TᵀRT = R·TᵀT`, relative offsets only) is what is pinned rather than a lucky alignment;
  - `free_pressure()` bit-identical to the full-array closure on **all six faces**, high ones
    included, because the high-face branch is new in this batch and no ledger sees it (§6.7);
  - the short-circuit headline (§7.6), including **§7.6.4's real plate mode**: the even-index mode's
    exactly-zero net volume velocity held for the whole run, and a nonzero radiated energy — on an
    exactly symmetric scene, with the off-centre and asymmetric-room cases as the measured
    **positive** controls (the zero is supposed to break there, and by how much is a number);
  - the **free plate's rigid-body translation** damped to a stop against a bare-plate control (§7.8);
  - the `StringPlateBridge` margin's bit-identity loaded vs bare (§6.8), so that a future
    non-dissipative load fails loudly there.
  Helpers `make_room_loaded_plate` and a prescribed-velocity `surface_drive` in `tests/helpers.py`.
- `scripts/diagnose_airbox_surface.py`: the radiated-power-versus-pattern-fineness sweep with the
  coincidence brackets marked, the energy-channel flat-total figure, the surface pressure field, the
  plate-mode table of §7.6.4 including the rows above `fs/4` that the suite cannot rank, and the
  **asymmetric-room figure** — an even mode's silence broken by the room's own asymmetry at the 1–3 %
  level, which is a thing no one-port can do and reads as a result rather than an error.
- Docs: HANDOFF §12H updated (batch 3 shipped, the two-sided dipole plate named as batch 4), this
  plan's status block rewritten with what the build changed, and the memory mirror synced.

---

## 10. What the build changed — the post-build record

Everything above this line is the pre-build prediction. This section is what the shipped code
measured, and it is the authority wherever the two disagree. The four corrections below are marked
inline at their sites.

`SurfacePort` and `RoomLoadedPlate` shipped in `physsynth/core/airbox.py` exactly as §9 scoped them
— **zero edits** to `AirBox`, `plate.py`, `connection.py`, `body.py`, `radiation.py` or `bore.py`.
The one structural surprise was pleasant: `AirBox.step()` is *linear* in the port weights, so a
`SurfacePort` passes the per-node volume-velocity **vector** as `weights` with `U = 1.0` and both
the injection and the read-back come out right without the room knowing a surface exists. Batch 2's
`_pending_ports` type comment ("normalized") was relaxed to say so — that, and this document, and
`tests/test_airbox_port.py`'s header claim (§10.5), were the only edits outside `airbox.py`.

The batch's headline claims all held as written: the off-diagonal of the room's instantaneous
response is **exactly `0.00e+00`**, the load `TᵀRT` folds into the plate's own `splu` with nothing
new solved, an even-index supported mode holds `|U|/A` at rounding for a whole run while radiating
**5.6×** the (1,1) mode's energy, and `|radiated − injected|` is **exactly `0.00e+00`** where a
wrong `R_j` puts it at 18 % of the channel with the conserved total still green (§6.1, the finding
this batch exists to have made).

### 10.1 Bilinear's equivariance is *not* offset-independent — it needs `S` integral

§6.5's "9.0e-16 at every one of eight offsets, flat, no periodicity" does not reproduce. Measured on
the shipped code two independent ways — the defect of `TᵀRT` under the surface's own mirror
permutation, and the defect of `T` itself under that mirror composed with the air grid's — bilinear
is exactly equivariant when `S = 2·(surface centre)/h_air` is an **integer** (defect **1.0e-15**)
and fails smoothly otherwise: **1.6e-01 … 3.8e-01** across sixteen offsets spanning one air cell,
peaking at the half-cell. The algebra agrees, which is why this is a finding rather than a doubt:
the mirror sends node `i` to cell fraction `frac(S − t_i)`, which equals the `1 − f_i` that reverses
a bilinear weight pair only for integral `S`. The plan's eight-offset sweep evidently sampled a
period it could not see.

Nearest-node's failure is patchier and the *way* it fails is the tell: exact at an **even** `S`
(measured `0.00e+00`), broken at an **odd** one (5.2e-01), because there the surface's centre node
lands on a rounding tie that round-half-to-even resolves the same way from both directions. So
nearest-node's symmetry is an accident of alignment *and of the rounding rule*; bilinear's is a
property of the geometry, holding on a stateable condition rather than everywhere.

**This strengthens the centred default rather than weakening it.** Centring is now load-bearing
**twice** — it is what makes `S` integral and the load equivariant at all, on top of §7.6.4's scene
symmetry — and the two reasons are independent.

### 10.2 The symmetry argument does not discriminate the operators at all — coverage does

§6.5's decisive third measurement is the one that dissolved. With the surface **centred**, an even
plate mode's `|U|/A` stays at rounding (1.3e-14 … 3.3e-13 over 200 steps) under **both** spreadings
at `N_plate = 8, 16, 24`. The plan's "18 % leak under nearest-node, tracking the load defect
one-for-one across fourteen orders of magnitude" was measured with the plate somewhere that made
the *scene* asymmetric, and it was the scene doing the leaking. What actually breaks the zero, both
measured on the shipped code: **2.3e-01** for a plate off-centre by `h_air/3`, and **7.4e-02** for a
perfectly centred plate in a room made asymmetric in the mode's own axis — against **7.2e-14**
centred and symmetric.

So bilinear ships for the argument §6.5 called merely aesthetic. It leaves **0** unfed footprint
nodes at every ratio where nearest leaves up to 8 of 12, and its interior assignment is 10×–100×
flatter *and converging* — **0.082, 0.062, 0.051, 0.031** over four refinements against nearest's
**0.83, 1.03, 0.64, 0.46**, which wander.

And the flatness claim needs its own correction, because "partition of unity" promises more than it
delivers. Bilinear's interior assignment is `h_air²` **exactly** (5e-16 … 2e-15) only when
`h_air/h_surface` is an **integer**. Off an integer it ripples, and not monotonically in the ratio:
2.93 gives 0.0077 while the *finer* 4.40 gives 0.0207. Poisson summation on the periodised hat says
why — the `k`-th coefficient is `sinc²(πk·h_air/h_surface)`, vanishing exactly when
`k·h_air/h_surface` is a nonzero integer, so **which** harmonics cancel is the ratio's arithmetic
rather than its size. The residual is a small ripple in the source's amplitude across the surface,
it shrinks under refinement, and it is **this coupling's accuracy floor** — worth knowing before
reading a radiated magnitude as physics.

### 10.3 The plate-mode ranking inverts, and the binding constraint is the air grid's *space* axis

§7.6.4's `radiated / (1,1)` column — and its "the finer mode radiates **more**, up to 7.1×" — is an
artifact of the grid it was measured on. §7.6.4 attributed the untrustworthy rows to the **plate's
time** axis (modes above `fs/4`). Measured across a 4× air-grid refinement at fixed physical room
and duration, the binding constraint is the **air grid's space** axis: `(4,2)` radiates **0.018,
0.870, 0.9998** of its energy at `h_air` = 82.5, 41.3, 20.6 mm.

That refinement sweep cannot by itself separate the two axes, because `h_air = c₀√3/(CFL·fs)` ties
them: raising `fs` refines the air grid *and* lifts Nyquist in lockstep. The discriminator is to pin
`h_air` at the coarsest 82.5 mm and reach the same three sample rates by lowering the Courant
fraction instead (0.900, 0.450, 0.225 — all legal). Time resolution improves 4×; space does not
move. Measured:

| mode | space **and** time refined | time alone (`h_air` pinned) |
|---|---|---|
| (4,2) | 0.0181 → 0.8695 → 0.9998 | 0.0181 → 0.0155 → 0.0228 |
| (3,1) | 0.8396 → 0.9993 → 0.9999 | 0.8396 → 0.7616 → 0.7893 |

Four times the time resolution moves `(4,2)` not at all; it recovers only when the space axis moves.
Two honesties the diagnose script prints beside the table: the control runs at a different Courant
fraction and therefore different numerical dispersion, which does not plausibly account for 50× but
is not nothing; and the correction rests on the `(4,2)` row with `(3,1)` as weak support — the other
four modes sit flat at ~1.0000, so this is one measurement repeated, not six independent
confirmations.

**Once every mode is resolved the ranking inverts.** Per cycle of the mode's own oscillation at
equal rms velocity: **1.000, 0.448, 0.448, 0.260, 0.213, 0.091** — strictly decreasing. That is
§7.6.2's fineness law showing up on *real plate modes*, and it means §7.6.4's ranking column was
measuring the **cycle count** over a fixed window. The plan's instruction — the plate-mode oracle
asserts the exact zero and a nonzero power and **ranks nothing** — was right, and is now right for
a second reason.

**What survives untouched is the zero.** Peak `|U|/A` sits at 2e-15 … 3e-14 for every even mode at
every refinement level, exactly as a symmetry statement should. The claim that is resolution-free is
resolution-free; the claim that is not, was not.

### 10.4 Coincidence is a scaling collapse, and the bracket is one sweep interval wide

§7.6's refusal to assert a located knee was right, for a better reason than it gave. Driven at
prescribed velocity, power falls **strictly** with fineness below every `f_c` — 0.564, 0.159, 0.070,
0.038, 0.015 of the piston at 150 Hz, strict at 100 and 250 Hz too. At fixed frequency the five
patterns span **39×**; plotted against `f/f_c` the same points **collapse to within 1.5×–5.5×**, and
every curve **peaks at `f/f_c = 1`**. So the coincidence law locates the *peak*; the unity crossing
sits on the rising flank, in the same `[0.70, 0.85] f_c` for every pattern that crosses, across a
factor of three in fineness. That bracket is **one sweep interval wide** — exactly as tight as the
frequency grid and no tighter, which the script says beside it.

Three rig facts measured rather than assumed, each of which would otherwise have produced a
confident wrong figure:

- **Equal rms *velocity* is what makes two modes comparable.** Equal rms displacement puts 4700×
  more energy in the finest mode and ranks amplitudes instead of modes.
- **A lossy room is what gives "did it radiate" a one-way answer.** A rigid box hands the energy
  back, and a fixed-window fraction then wanders.
- **The drive needs a raised-cosine ramp.** A hard start radiates a click larger than the
  steady-state power at the fine patterns.

And the resolvability floor is the **first** thing the diagnose script prints, because a pattern the
air grid cannot carry aliases, and an aliased point on a monotonicity curve looks exactly like a
clean result. All five shipped patterns clear `λ_p ≥ 4 h_air`.

### 10.5 Smaller things the build settled

- **The load *does* thicken the factorization**, against §2's expectation that it would not: LU fill
  **1.55× / 3.50× / 5.29×** at `h_plate/h_air` = 0.45 / 0.23 / 0.15. `lu_nnz` is exposed because the
  stored `nnz` (2.9× / 8.7× / 18.2×) is **not** what `splu` pays.
- **§6.8's guard errs safe, and the sign is the claim, not the size.** The bridge margin is
  *bit-identical* loaded versus bare (the load enters `A`, never `G0`), and adding the load block to
  `G0` anyway *reduces* `(G0⁻¹)_dp` — ratio 0.500 supported, 0.995 free.
- **The radiated channel is a property of the motion, not of the coupling.** A struck bump gives
  0.002 of `E₀` (fine patterns radiate badly — the short circuit working), the free plate's
  **piston** gives 0.9974. The conservation assertion ships on the piston config so it is not
  vacuous.
- **The face RIM is refused rather than clipped.** A rim node touches a *second* wall, carrying half
  `W` and the **sum** of two admittances, so `R_j` stops being uniform and `TᵀRT = R·TᵀT` — the
  whole equivariance argument — stops holding. Clipping would keep every ledger green with the
  geometry quietly wrong.
- **Reductions:** `net_area` is `((N−1)/N)²·Lx·Ly` supported (dead rim nodes displace nothing) and
  exactly `Lx·Ly` free, as §6.6 predicted. `T = 0` reduces to a bare `Plate` **bit-identically**
  (structural zeros eliminated before factoring) — that is the reduction available, since `R = 0`
  happens only on a refused open face.
- **`tests/test_airbox_port.py`'s header was corrected**, alongside `RoomPort.R_room`'s docstring:
  both said omitting the `1/(1+β)` factor "leaks ~2 %". Nothing leaks. The 1.9e-2 was the **ledger
  gap** (§6.1), and the total drifts *less* with the factor dropped than with it right.

### 10.6 What comes next

**Batch 4 is the interior two-sided (dipole) plate** — a plate hanging *in* the room rather than
flush in a wall, radiating from both faces, which is an internal moving boundary rather than a
source and therefore a genuinely different object from anything shipped here. §3 named it and the
scope it defers is unchanged: PML or higher-order absorbing boundaries, scattering objects and
non-rectangular rooms, moving ports, and viscothermal air absorption.
