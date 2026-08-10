# The drumhead in the room — the membrane as a suspended and baffled surface (air-box batch 5)

Batches 3 and 4 gave the room a *plate*: mounted flush in a wall (`RoomLoadedPlate`, a source) and
hanging in the air (`RoomSuspendedPlate`, an object). Batch 5 hangs a **membrane** (model #4) there
instead — the frame drum, the tom head, the tambourine — and mounts one in a baffle for the
reference arm. HANDOFF §12H names it, in as many words:

> `Membrane` (#4) and the von Kármán plate (#6) as suspended surfaces (the port needs no change for
> either — neither is wired up)

**Half of that sentence is wrong, and the probe is what says so.** The port needs no change for a
*rectangular* membrane and refuses a **circular** one at every resolution — which is the interesting
case, because a drumhead is round. The refusal is structural rather than a tuning problem, it is one
localized edit to fix, and the fix is measured below before a line of it is written. §2.1.

And the batch is worth doing for a reason that is not "the next resonator in the list". A plate's
bending-wave speed grows as `√ω`, so it **crosses** `c₀` at one frequency — coincidence, which
air-box batch 3 measured as a scaling collapse. A membrane's wave speed is the constant `c = √(T/ρ)`.
It therefore has **no coincidence frequency at all**: it is subsonic at every mode or supersonic at
every mode, and which one is decided by a single dimensionless number `c/c₀` that the player sets by
tuning the head. That is a structurally different radiation story from anything in this repo, and it
is the fourth entry in the series batch 2 (the delayed echo), batch 3 (the acoustic short circuit)
and batch 4 (the ratio that crosses 1) belong to: a claim no `R(ω)` fitted to a plate can make.

The trap in that headline is in §6.1 and it would *invert* the result if ignored.

---

## 0. What the probe killed before this document was written

Following batch 18's precedent — four claims died in that probe — the probe ran first. Scripts:
`M:\claud_projects\temp\airbox-b5-probe\probe.py` and `probe2.py`.

| # | The claim | Verdict |
|---|-----------|---------|
| 1 | HANDOFF §12H: "the port needs no change for either" | **Dead** for the disk — refused at every resolution, both tiers |
| 2 | This plan's own first draft: the footprint ceiling is `h_mem ≤ h_air` | **Dead** — measured ceiling is **two** air cells, so the reachable `c/c₀` is **2× wider** than drafted |
| 3 | `airbox.py` L1974-6: "models #4, #5 and #5b have had a `pressure()` read-out since they were built" | **Dead** for #4 — `Membrane` has no `pressure()` |
| 4 | Implicit in "just wire it up": the membrane stays an **explicit** scheme | **Dead** — the load is in `A`, so #4 acquires a solve it never had (§5.1) |

Claim 2 is the one to note, because it was *this document's* and it moved a number in the direction
that makes the batch easier. The draft reasoned that a bilinear stencil reaches one air cell, so a
membrane coarser than `h_air` must leave gaps. Measured, consecutive membrane nodes may straddle
floors two apart and still leave nothing unfed, because each node's stencil is `{i, i+1}`:

```
    h_mem / h_air  =  0.505  0.918  1.010  1.188  1.554  1.836  2.020 | 2.244  2.525  2.886  3.367
    footprint check   ok     ok     ok     ok     ok     ok     ok    | UNFED  UNFED  UNFED  UNFED
```

with the same threshold centred and offset by `h_air/3`, i.e. alignment is **not** load-bearing
there. So the ceiling is `h_mem ≲ 2 h_air`, and §4's feasibility window doubles.

---

## 1. Why — what a membrane says that a plate cannot

### 1.1 One global number instead of one frequency

Kirchhoff bending gives `c_b(ω) = √(κω)`, monotonically increasing and unbounded, so there is always
a coincidence frequency `ω_c = c₀²/κ` where `c_b = c₀`; below it the plate is a poor radiator, above
it a good one. Every plate in this repo has one, and batch 3's diagnose script measured it as a
scaling collapse with the bracket set by one sweep interval.

A membrane's `c = √(T/ρ)` has no `ω` in it. The consequences are all sharp:

* There is **no** frequency at which the character changes. The membrane is subsonic at every mode
  or supersonic at every mode.
* The control is the **material and the tuning**, not the mode number: `c/c₀ = √(T/ρ)/343`. A player
  raising the head tension walks the whole instrument across the threshold at once.
* Real drumheads sit *well* below it — a Mylar head at `T ≈ 3000 N/m`, `ρ ≈ 0.26 kg/m²` gives
  `c ≈ 107 m/s`, i.e. `c/c₀ ≈ 0.31`. The acoustic short circuit is therefore not an edge case for a
  drum: it is the drum's normal operating point, and it is why a drumhead with no shell is quiet.
* Both arms are reachable on this grid (§4), so the crossing can be **measured**, not argued.

### 1.2 What that makes this batch, beside "the next model"

The three earlier batches each produced one structural claim. This is the fourth, and it differs in
kind from theirs: batches 2–4's claims are about the *coupling* (a delay, a short circuit, a
direction). This one is about the **resonator's own dispersion relation**, read through the
coupling. It is the first time the air box says something that distinguishes two *resonator
families* rather than two ways of attaching one.

---

## 2. What must change in `airbox.py` — and nothing else

### 2.1 `_check_footprint` assumes the surface is a rectangle. Measured.

The check exists to refuse a surface too coarse to feed every air node under it — a comb at the grid
scale — and it builds the required set as a **bounding box**:

```python
    inside.append(np.nonzero((grid >= lo) & (grid <= hi))[0])       # per axis, from min/max
    foot = (inside[0][:, None] * (room.N[t1] + 1) + inside[1][None, :]).ravel()   # outer product
```

For a disk the bbox corners are ~0.41·R from the nearest live node, so they are unfed *by
construction* and refining makes it no better:

```
    circle N= 40  h_mem/h_air = 0.495  n_live= 1245  ->  REFUSED: 20 of 324 air nodes unfed
    circle N= 56  h_mem/h_air = 0.354  n_live= 2449  ->  REFUSED: 48 of 400 air nodes unfed
    circle N= 72  h_mem/h_air = 0.275  n_live= 4049  ->  REFUSED: 40 of 400 air nodes unfed
    rect   N= 40/56/72  (same spacings)              ->  accepted
```

Both tiers refuse it — `SurfacePort` on a wall gives the identical 48/400. **The refusal is right
and its required set is wrong**: those corner nodes are not "under the surface" at all.

**The fix, validated in the probe before being written:** build the required set **span-wise** —
per air-node row, the columns between that row's own min and max reached column; per column, the
rows between its own min and max; and require their **union**. For a rectangle every row spans the
same columns and every column the same rows, so this reduces to the bounding box **by
construction**, which is what protects batches 3 and 4.

The union is strictly stronger than rows alone, so it is measured as the union rather than inferred
from it — the disk's symmetry would make rows and columns agree, and a disk on a non-square air
footprint or an odd node count need not:

```
    case                    bbox   rows   cols   UNION      case                bbox rows cols UNION
    circle R=0.15 N=24        64      0      0       0      rect 0.30x0.18 N=40    0    0    0     0
    circle R=0.15 N=40        56      0      0       0      rect 0.29x0.13 N=37    0    0    0     0
    circle R=0.15 N=56       108      0      0       0      rect 0.30x0.30 N=56    0    0    0     0
    circle R=0.15 N=72       100      0      0       0
    circle R=0.15 N=96       100      0      0       0      <- disks: UNION is 0 at every N
    circle R=0.13 N=25        48      0      0       0      <- odd N, non-square air footprint
    circle R=0.13 N=41        40      0      0       0
    circle R=0.13 N=57        40      0      0       0      <- rects: unchanged, as required

    negative control (a genuinely coarse comb must still be caught):
    h_mem/h_air = 1.98   bbox   0   rows  0   cols  0   UNION  0    <- both pass; not a comb
    h_mem/h_air = 2.20   bbox  68   rows 32   cols 32   UNION 64    <- both refuse
    h_mem/h_air = 2.83   bbox 112   rows 48   cols 48   UNION 96    <- both refuse
    h_mem/h_air = 3.96   bbox 132   rows 48   cols 48   UNION 96    <- both refuse
```

So the weaker criterion does not weaken the check where the check earns its keep: the comb threshold
is unmoved — it lies in `(2.02, 2.20]` under both — and only the *shape* assumption is dropped. The
`footprint_empty` attribute, the refusal text and the "count, not an inequality on `h_surface/h_air`"
rationale all survive; the message gains the shape it now knows about.

**This is the batch's only edit to shipped port machinery.** Everything else the disk needs is
already there — probe 2 bypassed the check and constructed the port to the end:

```
    port.face_count 376  ==  room.cut_faces 376        # the staircased cut registers as a set
    load_matrix (2449, 2449) nnz 165529  sym-defect 1.084e-19
    R across the patch: ptp = 0.000e+00                # one R for both planes, as batch 4 requires
```

`_register_cut` takes an arbitrary `(i0, i1)` node set, so a **staircased** disk cut needs no work.
Only the *public* `add_cut` is rectangle-only (its `extent=` is a pair of inclusive ranges), and a
port never goes through it. That stays as it is; §3 defers a general hand-placed cut.

### 2.2 The doc line that is wrong about model #4

`airbox.py` L1974-6 says models #4, #5 and #5b "have had a `pressure()` read-out since they were
built". `Plate` has one (L374); **`Membrane` does not**. The coupling does not use it — the port
reads the *state*, not the monopole — and adding one would need an `_accel` that `Membrane.step`'s
two-level roll does not keep. So this is a **one-line doc fix**, not a core edit, and the read-out
stays absent. Recording it because the sentence is load-bearing rhetoric in batch 3's docstring: the
point it makes ("that read-out is the compact `a → 0` limit of this") is true of #5/#5b and simply
does not apply to #4.

---

## 3. Scope — and what is deliberately deferred

**In:** `RoomLoadedMembrane` (baffled, flush in a wall) and `RoomSuspendedMembrane` (hanging, two
faces, cut) for both domains — `rectangle` and, once §2.1 lands, `circle`. The row-wise footprint
fix. The seam extraction of §5.2. Oracles per §7.

**Both tiers ship, and completeness is not the reason.** The `c/c₀` claim is a *comparison*: batch
4's dipole/baffled resistance ratio is the instrument that made "unbaffling changes sign" a claim
rather than a magnitude, and the same ratio is what will make the threshold visible here. The
baffled arm is the reference the suspended arm is read against. With the §5.2 seam in place the
second tier is a constructor and a `2`.

**Out — the von Kármán plate (#6) is batch 6, not the second half of this one.** The reason is
concrete rather than budgetary. `VKPlate.step()` takes **no** `f_ext`, its conservation holds only
*at the Picard fixed point*, and its state lives in two coupled fields (`u`, `F`) with the roll
tying `F` to `(u, u_prev)`. Its extension point is named here so batch 6 does not rediscover it:

* The load is **linear in `u^{n+1}` and independent of `F`**, so it folds into `A` exactly once and
  the Picard loop is untouched. That orthogonality is what makes #6 tractable at all.
* Therefore the seam #6 needs is a **loop hook**, not the single `rhs()` §5.2 builds — the coupling
  term is recomputed each sweep and the loaded solve must sit inside that. Designing for it now
  would be speculative; §5.2 says where it goes.
* And #6's money test becomes a **two-parameter** convergence claim: `|radiated − injected|` will
  depend on `couple_tol` as well as on the load, which is the analogue of the existing
  "drift falls with `couple_tol`" self-certifying test. Batch 6 owns it.

`Membrane` first also de-risks `VKPlate`, which is the relationship `beam → free plate` already has
in this repo.

**Also out:** a `pressure()` read-out for `Membrane` (§2.2); a general non-rectangular `add_cut`
(§2.1); the viewer (a batch-18-style surfacing is a separate question and needs a new claim, not a
new model — the web-viewer plan's own rule); `StringMembraneBridge` (no such coupling exists and the
batch does not need one); and PML, non-rectangular rooms, moving ports and viscothermal absorption,
which stay on HANDOFF §12H's deferred list untouched.

---

## 4. The feasibility window — computed up front, not discovered mid-sweep

Three constraints pull against each other and the honest thing is to state the window before the
sweep rather than to hit its edge during one (batch 4's `ka ∈ [1.0, 1.3]` precedent — a sweep only
ever brackets a crossing to one interval).

```
    membrane CFL    h_mem >= sqrt(2) c  k        (5-point Laplacian, lambda_mem <= 1/sqrt(2))
    footprint       h_mem <= 2 h_air             (MEASURED, §0 claim 2 -- not 1 h_air)
    room CFL        h_air >= sqrt(3) c0 k        (lambda_air <= 1/sqrt(3))

    =>   c / c0  <=  sqrt(2) / lambda_air        [ fs-independent ]
```

| `λ_air` | max `c/c₀` | note |
|---------|-----------|------|
| `1/√3` = 0.5774 | **2.449** | the room at its own CFL ceiling — the cheapest room |
| 0.50 | 2.828 | |
| 0.40 | 3.536 | room nodes scale as `λ_air⁻³` |
| 0.30 | 4.714 | |

So the crossing at `c/c₀ = 1` sits comfortably inside the window even in the cheapest room, with
headroom to 2.4×, and **the sweep does not have to be designed around the constraint**. The draft's
`√(3/2) = 1.2247` — which would have made the supersonic arm a single marginal point — was the
casualty of §0 claim 2.

The cost that *is* real: the membrane's CFL floor means a supersonic head is a **coarse** grid at
fixed `fs`. Measured at `fs = 40 kHz`, a 0.30 m square patch:

```
    c/c0 = 0.30  N = 82  n_live = 6561        c/c0 = 1.10  N = 22  n_live = 441
    c/c0 = 0.60  N = 41  n_live = 1600        c/c0 = 1.20  N = 20  n_live = 361
    c/c0 = 0.90  N = 27  n_live =  676        c/c0 = 1.40  N = 17  n_live = 256
```

A 17×17 head resolves few modes, and §6.1 is why that specifically matters. Raising `fs` refines
both grids together at the room's `h⁻⁴` cost (the airbox-viewer batch's finding: the 3-D CFL runs
the wrong way, so the **room** sets `fs`). §8 budgets it.

---

## 5. The discrete scheme

### 5.1 The membrane stops being explicit, and that is the batch's main design fact

Every earlier air-load landed inside a solve that already existed — the plate back-substitutes a
prefactored SPD system every step regardless. `Membrane` does not: it is a pure explicit update,

```
    u^{n+1} = ( 2 u^n - (1 - sigma k) u^{n-1} + c^2 k^2 L u^n ) / (1 + sigma k)
```

and the load `f_load = -Tᵀ p̄_free - Tᵀ R T (u^{n+1} - u^{n-1}) / 2k` has an unknown in `u^{n+1}`.
Two options, and this is the first batch in the family that has a choice:

**Ship: the load in `A`.** With per-node mass `ρ h²` (uniform — no `W`, no dead-rim weighting):

```
    A = (1 + sigma k) I  +  (k / 2 rho h^2) TT R T            <- SPD (PSD added to SPD), splu ONCE
    rhs = 2 u^n - (1 - sigma k) u^{n-1} + c^2 k^2 L u^n
          - k^2 TT p_free / (rho h^2)  +  (k / 2 rho h^2) (TT R T) u^{n-1}   [ +  k^2 f_ext / rho h^2 ]
```

and for the suspended tier the load matrix and the `p̄` term double, exactly as batch 4. Passivity
is then a property of the matrix rather than an inequality to check, `radiated == injected` stays an
**identity**, and the membrane's `λ ≤ 1/√2` and the room's `λ ≤ 1/√3` remain the only two stability
conditions — coupling them adds no third. The price is honest and must be reported, not buried: a
model whose whole character was "explicit, one matvec" acquires a sparse factorization and a
back-substitution per step.

**Measure once, as a negative control, then discard: the lagged-explicit load.** Evaluate the load
velocity at `(u^n - u^{n-1})/k` and the scheme stays explicit. It also stops being unconditionally
passive and breaks the cross-ledger identity. `spreading="nearest"` is the precedent for shipping
exactly one measured negative control, and this deserves the same treatment: one number showing what
it costs (§7.6), not a configuration flag.

### 5.2 Extract the seam first, in its own zero-behaviour commit

Batches 3 and 4 each reassemble the plate's `A` and its θ-scheme RHS, in two boundary branches — four
copies today, and a naive batch 5 makes it eight (× two domains × two tiers). The RHS reassembly is
already the thing batch 3 flagged as needing a dense coupled cross-check, so multiplying it is
exactly the wrong direction.

Extract a small adapter **inside `airbox.py`** (so "`plate.py` untouched" survives) exposing what a
loaded surface resonator has to provide:

```
    a_bare()        the unloaded system matrix                     denominator   rho_s per node
    rhs(f_ext)      the force-free RHS + the f_ext path            surface()     (coords, areas)
    commit(u_next)  roll the history, refresh what the model caches
```

with a `Plate` adapter (both boundary branches) and a `Membrane` adapter (`a_bare = (1+σk) I`,
`denominator = ρh²`), and rewire `RoomLoadedPlate` / `RoomSuspendedPlate` onto it. VK's loop hook
(§3) is a fourth method that batch 6 adds; do not add it now.

**The membrane adapter's `rhs(f_ext)` is *new* arithmetic, and that is a difference worth naming.**
Batches 3 and 4 reproduce `Plate.step`'s own `f_ext` path line for line, so a transcription error
shows up against the model itself. `Membrane.step()` takes **no** `f_ext` at all — the same gap that
sends `VKPlate` to batch 6 (§3) — so the membrane's `k² f_ext / ρh²` term has no counterpart to be
bit-identical *to*. It is easy arithmetic (uniform per-node mass, no `W`, no θ), but the only thing
pinning it is §7.1's `T = 0` reduction plus §7.2's ledger, and no caller in this batch passes an
`f_ext` at all. Either wire the term and test it directly, or omit it and let batch 6 add it with
its first real caller — do not ship it untested because the signature looked like batch 3's.

**The refactor is guarded by construction, which is why it goes first and alone.** Batches 3 and 4
already pin exact numbers — the `StringPlateBridge` stability margins `0.2061806714931906`
(supported) and `0.2061840079056186` (free), bit-identical loaded and bare, plus `nnz_growth` and
`lu_nnz`. A behaviour-preserving extraction reproduces those to the last digit and a broken one fails
loudly there. Combined with new physics in one commit, the ability to say *which* change moved a
number is gone.

### 5.3 The rim ring is not part of the radiating surface

`coords = X[mask], Y[mask]` gives the **live** nodes, and unlike the free plate — where `mask` is
all-ones — a membrane's rim is clamped and **dead**. So the moving surface is one cell inside the
nominal boundary and its area is the live sum, not `πR²`. Measured on the probe's disk
(`R = 0.15 m`, `N = 56`):

```
    live area  sum(h^2)     0.070284 m^2      live / pi R^2  =  0.9943
    net_area   (moving)     0.070284 m^2      <- agrees, as it must
    blocked_area (the cut)  0.086293 m^2      =  1.228 x the moving surface
```

Both numbers must be reported by name in any radiated magnitude, because batch 4 measured that the
dipole's magnitude tracks `blocked_area` rather than `h` — so an area quietly taken as `πR²` yields
a plausible, wrong, and green-ledgered magnitude. `InteriorSurfacePort.blocked_area`'s docstring
already anticipates exactly this for the supported plate ("the moving surface is the *live*
footprint, so the clamped rim is not part of the obstacle at all"); a membrane is that case always,
never the free plate's.

---

## 6. Traps — measured before a line of core code

### 6.1 The scheme manufactures a spurious coincidence, and it would invert the headline

§1.1 is a **continuum** claim. `membrane.py`'s own docstring says the 5-point Laplacian is
anisotropic and that *no* `λ` is dispersionless, so the discrete phase speed falls below `c` at high
wavenumber. A supersonic head therefore falls **back** below `c₀` somewhere on the grid — the exact
coincidence the headline says a membrane does not have. Measured (`c_ph/c`, axis vs diagonal):

```
    lambda_mem = 1/sqrt(2):  axis  beta*h/pi = 0.1 .. 1.0 : 0.998 0.992 0.981 0.965 0.943 0.877 0.707
                             diag  beta*h/pi = 0.1 .. 1.0 : 1.000 1.000 1.000 1.000 1.000 1.000 1.000
    lambda_mem = 0.50     :  axis                          : 0.997 0.988 0.972 0.950 0.920 0.840 0.667
                             diag                          : 0.999 0.996 0.991 0.983 0.973 0.944 0.874
```

Two things fall out, and the first is a small gift: **at `λ = 1/√2` the diagonal is exactly
dispersionless** — 1.0000 to four places across the whole band — while the axis degrades to 0.707 at
the grid Nyquist. That is the 2-D analogue of the 1-D `λ = 1` exactness, surviving on one direction
only, and it is the sharpest reason to run the membrane **at** its CFL ceiling here. (`λ = 1/√2` is
also the anisotropy *maximum*: the axis is worst there. Both statements are in the table.)

The knees, along the worse (axis) direction:

```
    lambda_mem = 1/sqrt(2):  1% knee at beta*h = 0.686 (0.218 pi) ~ 4.6 nodes/wavelength
                             5% knee at beta*h = 1.478 (0.470 pi) ~ 2.1 nodes/wavelength
    lambda_mem = 0.50     :  1% knee at beta*h = 0.564 (0.180 pi) ~ 5.6 nodes/wavelength
```

and the inversion itself — the wavenumber at which a supersonic head's `c_ph` falls back to `c₀`:

```
    c/c0 = 1.05,   lambda = 1/sqrt(2):  beta*h = 1.445 (0.460 pi)  ~ 2.2 nodes/wavelength
    c/c0 = 1.10,   lambda = 1/sqrt(2):  beta*h = 1.930 (0.614 pi)  ~ 1.6 nodes/wavelength
    c/c0 = 1.2247, lambda = 1/sqrt(2):  beta*h = 2.594 (0.826 pi)  ~ 1.2 nodes/wavelength
```

Every one of those is **on the grid** (`βh < π`), so a broadband strike on a marginally supersonic
head genuinely contains a subsonic upper spectrum. Read together with §4's node counts — `c/c₀ =
1.2` is a 20×20 head whose upper modes sit near Nyquist — this is not a remote corner.

**So the claim ships as bracketed, not sharp.** The continuum threshold is at `c/c₀ = 1`; the scheme
smears it over a band whose width is computable and is reported with the result, and the measurement
band is restricted below the 1% knee. Stating "a membrane has no coincidence frequency" beside a
sweep that silently crossed the inversion would measure the opposite of the claim and look clean
doing it.

### 6.2 The disk is staircased, and its rim is the membrane batch's own lesson

Model #4's circular domain is a staircased Dirichlet rim: energy stays exact (the masked `L` is
symmetric) while the Bessel match degrades to ~`O(h)`. Radiation adds nothing new to that, but it
does add a second staircase — the **cut**, which follows the reached air-node set, not the membrane
mask (§5.3's 1.228× is that). Two staircases at two different spacings is a thing to report rather
than discover, and it is why every disk oracle here is a *ratio* or a *rate*, never a magnitude.

### 6.3 Everything batches 3 and 4 already pay for, inherited unchanged

The centred-velocity choice into a half-step face slot (batch 4: forcing the alternative would add
mass and land in the bridge's guard); the centred `origin` default (an off-centre surface's even
modes stop being exactly silent, *linearly* in the offset with no threshold); the in-plane rim
refusal; the additive cut; `p̄_free` read **before** `room.step()`. None of these change and none is
re-derived here — but the membrane is the first surface for which `origin=None` centres a *disk*, so
the equivariance argument is re-measured rather than assumed (§7.5).

### 6.4 The conserved total is necessary and not sufficient — for the third time

Batch 3 measured that dropping the `1 + β` wall factor from `R_j` leaves the scene total flat to
4.9e-15 while `radiated − injected` goes to 18% of the channel; batch 4 measured that the money test
is *itself* blind to half the ways to get its `2` wrong, and that only the coupled residual against
the room's own post-closure jump catches both. Both findings transfer verbatim and the guards ship
here as they stand. Nothing about this batch is expected to change them — which is exactly why a
green total will not be read as a pass.

---

## 7. Oracles — what must pass

**7.1 Reduction.** `T = 0` (zero areas) → bit-identical to a bare `Membrane`, both tiers, both
domains. This is the load's `eliminate_zeros()` path and it is what makes the whole batch falsifiable
at one end.

**7.2 The money test.** `radiated_energy == room.injected` to rounding, and the scene total
`membrane.energy() + radiated + room.energy()` flat — reported **with the channel size**, because a
conservation test on a channel worth 1e-14 of the total passes disconnected.

Batch 3's precedent is to *name* the non-vacuous configuration, and here the naming had to be
probed, because **a membrane has no piston**: the rim is clamped, there is no rigid-body nullspace,
and batch 3's `0.9974` configuration therefore does not exist. The expectation going in was that a
realistic head (`c/c₀ ≈ 0.31`) would leave a channel too small to assert on. **Measured, it does
not** — the prototype of §5.1, suspended, 500 steps, `ρ = 0.26 kg/m²`, 0.30 m square:

```
    c/c0  walls  shape    N   channel/E0   |rad-inj|/channel   total drift/E0   E_end/E0
    0.31  rigid  mode01  79    8.424e-01         1.7e-15           1.20e-13       0.2032
    0.31  rigid  bump    79    2.307e-01         1.8e-15           6.08e-15       0.8296
    0.31  lossy  mode01  79    6.393e-01         1.9e-15           9.33e-14       0.4370
    0.60  lossy  mode01  41    8.438e-01         3.0e-16           2.94e-15       0.2124
    1.00  lossy  mode01  24    9.981e-01         1.7e-15           1.11e-15       0.0031
    1.40  lossy  mode01  17    1.000e+00         3.2e-15           4.74e-15       0.0000
```

So §7.2 is non-vacuous at the drum's **actual** operating point and needs neither a supersonic head
nor an all-lossy room; the `(0,1)` bulge is the configuration to name, and the narrow bump is the
contrast (0.23 vs 0.84 at `c/c₀ = 0.31` — the acoustic short circuit, visible in the channel itself).

**Two warnings must travel with that table or it will be misread.** First, `channel` is
`max |radiated_energy|`, and for a *suspended* surface that ledger is dominantly **reactive** — batch
4 measured 50.2% of its per-step increments negative and warned in as many words that reading it as
"84% radiated" is wrong by about a factor of two. It is the right measure of whether a conservation
assertion has anything to bite on, and it is **not** a radiation figure; §7.7's prescribed-velocity
rig is. Second, `E_end/E0` falls hard with `c/c₀` (0.44 → 0.21 → 0.003 → 0.000) which is the
headline's shape — but `N` falls with it too (79 → 41 → 24 → 17) and 500 steps is a different number
of periods at each `c`. It is an encouraging **sighting**, not a claim, and §7.7 is where it becomes
one or dies.

The same run validates §5.1's arithmetic ahead of commit 4: `|radiated − injected| / channel` is
1e-16…1e-15 and the scene total is flat to 1e-16…1e-13 across all sixteen configurations.

**7.3 The coupled residual at two timesteps** against the room's own post-closure pressure — batch
4's guard, the only one that catches both ways of getting the `2` wrong, applied to the doubled
membrane load.

**7.4 Passivity and the differential `R_j`.** `σ > 0` → monotone; `R_j` measured off the room
per-node rather than trusted from the assembly line.

**7.5 The disk's geometry.** The row-wise footprint criterion accepts the disk at `N = 24…96` and
still refuses a comb at `h_mem/h_air ≥ 2.24` (§2.1's table, promoted to tests); a rectangle's
required set is **identical** under old and new criteria; `cut_faces == port.face_count`; and the
centred `origin` default gives an equivariant load for a *disk* as it does for a rectangle.

**7.6 The negative control.** The lagged-explicit load (§5.1), measured once: how far
`radiated − injected` departs from zero, and at what `λ_mem` it goes unstable. One number, not a
flag.

**7.7 The headline, bracketed.** The dipole/baffled radiation-resistance ratio under prescribed
uniform motion, swept across `c/c₀` on both sides of 1, with the measurement band held below §6.1's
1% knee and the inversion wavenumber reported beside it. The claim is that the ratio's behaviour is
governed by `c/c₀` and by no frequency — bracketed to one sweep interval, batch 4's rule.

**7.8 Refusals.** CFL both sides; sample-rate mismatch; the rim refusal; a surface larger than the
plane; the port's disjointness.

---

## 8. Cost budget — owned, not discovered

The room sets `fs` (the 3-D CFL runs the wrong way — the airbox-viewer batch's `h⁻⁴`), so a membrane
oracle is cheap only if the room is small. Budget from batch 3/4's shipped tests: a 0.9 m room at
`fs = 40 kHz` is `61³ ≈ 2.3e5` nodes; the sweep of §7.7 is the cost driver, exactly as the juari
batch's was. Decide the sweep's arc count and duration **before** running it, mark the sweep `slow`,
and keep the per-tier oracles on the smallest room that keeps the channel non-vacuous. The suite is
1489 tests / ~23 min today and the `slow` marker means a claim, not a stopwatch reading.

---

## 9. Deliverables

1. `docs/dev/air-box-membrane-plan.md` — this document. *(commit 1)*
2. The §5.2 seam extraction, `airbox.py` only, **zero behaviour change**, pinned by batches 3/4's
   existing bit-identity numbers. *(commit 2, alone — and the `Membrane` adapter must **not** be in
   it. "Zero behaviour change" is only a meaningful claim about a commit that introduces no
   unexercised code path; the membrane adapter arrives in commit 4 with its first caller.)*
3. The §2.1 row-wise footprint criterion + the §2.2 doc fix. *(commit 3)*
4. `RoomLoadedMembrane` + `RoomSuspendedMembrane` and their tests. *(commit 4)*
5. A diagnose script for §7.7's sweep, and the post-build record — including every claim above that
   dies on measurement, which on this project's record will not be zero.

---

## 10. What the build changed — the post-build record

*(To be written after the build, per batches 3 and 4. If §1.1's headline survives contact with
§6.1's inversion unchanged, say so explicitly and say at what band — a claim that never moved is
either well-probed or under-measured, and the reader cannot tell which unless it is stated.)*
