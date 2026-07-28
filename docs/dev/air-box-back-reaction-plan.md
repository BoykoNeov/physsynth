# The room pushes back — the two-way body↔room port (air-box batch 2)

> **Status: BUILT (2026-07-28).** Batch 1 shipped the room as a read-out. This batch closes the
> loop: the body's volume velocity drives the air *and the air's pressure loads the body back*,
> exactly, passively, at machine precision. Seven traps were measured before a line of core code
> was written (§6); two of them changed the design and one of them settled the scope. All seven
> held. `physsynth/core/airbox.py` self-contained — no edits to `radiation.py`, `body.py`,
> `connection.py` or `bore.py`, as promised in §9.
>
> **What the build changed, and why — three things, all in §7.10.** The structural half of the plan
> (§7.1–7.9) shipped as written and hit its predicted numbers; the cross-tier oracle did not survive
> contact, and the reasons are worth more than the oracle was.
>
> 1. **The oracle became the port's *equivalent radius*, not `|Z|`.** `a_eff = ρ₀/(4πM_a)` with
>    `M_a = Im Z/ω` is one number saying what the port *is*, as a sphere — and it does **both**
>    halves of §7.10 at once (grid-independence and magnitude) where `|Z|` ratios did neither
>    cleanly, because `|Z|` carries the room's modal wiggle and the near-field mass does not.
> 2. **The room contaminated the magnitude by more than the effect being measured, and §7.10 as
>    written would have shipped that as physics.** Measured `a_eff/(5a/6)` at fixed `h` and a fixed
>    5 cm port: **1.086** in a 0.5 m room, 1.040 at 0.7 m, **1.003** at 1.0 m, 0.977 at 1.4 m. The
>    plan's "1.11 at ka = 0.23, consistent with 6/5" was reading the *room's* reactance, not the
>    port's. The 6/5 shape factor is real and now confirmed to **0.3 %** — but only where the port is
>    compact. A ratio survives a small cheap room; a magnitude does not. The test asserts both: the
>    closed form where it applies, and the small room's excess as the attribution.
> 3. **`a_eff` is Courant-invariant to five significant figures** (45.231 / 45.232 / 45.233 / 45.234
>    mm across `cfl` = 0.5, 0.7, 0.9, 0.998). It is a *static* near-field quantity, so the sweep runs
>    at the cheapest λ and — the point worth keeping — the measurement is not a dispersion artifact,
>    which in a scheme with no dispersionless λ is not something to assume. The first sweep grids
>    were accidentally sitting at **0.998 of the CFL ceiling**, where batch 1 measured the corner
>    mode going defective; that turned out not to matter here, but only because it was checked.
>
> Two smaller corrections to the plan's own numbers, both from measuring with the shipped code
> rather than the prototype: the second body first moves at **exactly Manhattan** steps (not
> `Manhattan + 1` as §7.9 recorded), measured at separations of 6 and 12 nodes; and the reflection
> returns at **`2d + 1`**, measured at `d` = 3, 5 and 7 — three geometries, one law, where the plan
> had asked for one.
>
> §6.3's `h/3.2` also refines to **`h/3.1`**: the prototype inferred it from `|Z|` ratios against
> a shell of radius `h`, and the reactance measures it directly (`a_eff/h` = 0.324, 0.320, 0.317
> across three grids). Same physics, one more digit, and now measured rather than inferred.
>
> One test not in §7 at all, added on review: the port's `O(port)` free-pressure read is asserted
> **bit-identical** to the full-array `_divergence()`-then-closure at interior, wall, edge and corner
> nodes. An off-by-one there would be a small, position-dependent error that *survives every energy
> test*, because the port and the room would still agree with each other.

---

## 1. Why — the refusal being discharged

Batch 1's own docstring names this batch, in the same sentence it names its limit:

> *"**Read-out only — no back-reaction.** The source injects into the room and the room does not
> push back on it. That is the exact position `AirRadiation` occupied in radiation batch 1, and the
> module's own history says build read-out first; a two-way, provably passive room-body port is the
> natural next batch."*

The radiation leg walked this ladder already, and it is worth being precise about *why* the second
rung mattered there, because the same reason applies here one tier up. `AirRadiation` (batch 1) made
a body *audible*; it changed nothing about the body. `RadiatedBody` (batch 2) made radiation
**cost** something — the body's modes decayed because the air took energy, and the decay was a
closed-form-checkable channel rather than a fitted `sigma`. That is the difference between a model
that renders a sound and a model in which the sound and the loss are the same physical fact.

Here the missing fact is bigger, because a room does something a free-field load structurally
cannot: it **gives energy back**, at a delay, from a direction. A body in a small hard room feels its
own reflected wave arriving `2d/c₀` later and is loaded by it. That is not reverberation added to a
dry signal — it is a change to the body's own oscillation, and it is why a guitar sounds different
against a wall in a way no post-hoc convolution can produce. The lumped tier cannot represent it at
any `R(ω)`: `RationalAirLoad` is a **causal one-port with no memory of geometry**, so its impulse
response is a decaying exponential, never a delayed echo.

**The claim this batch exists to make, in one line:** *the room load is exact and passive — what the
body loses, the room gains, to machine precision, for any wall, any port position, any number of
instruments.* Not "approximately conserves". The two ledgers are the same number with opposite
signs, and adding them cancels the term.

---

## 2. The physics — the room, seen from one node, is a Thévenin source

The whole batch rests on one observation about batch 1's `step()`, and it is worth stating before
any algebra because it is what makes the coupling cheap:

> **Within a single timestep, an injection at a node changes the pressure at that node and nowhere
> else.** The source term is added to `p^{n+1}` node-locally; propagation to the neighbours happens
> in the *next* step's momentum sub-step, at `c₀`.

So the room, viewed from the port during one step, is a **linear scalar relation** between the
volume velocity `q` injected and the centered pressure `p̄` the port sees:

```
p̄ = p̄_free + R_room · q
```

— a Thévenin equivalent: a known open-circuit voltage `p̄_free` (what the port would have seen with
`q = 0`, i.e. the room's own field arriving from everywhere else, *including every reflection*) in
series with a known internal resistance `R_room`. `p̄_free` is where the geometry lives; `R_room` is
the room's instantaneous self-term.

That is *structurally the same shape* the lumped tier already solves. `RationalAirLoad` reduces to
`p = R_eff (U − L⁻)` — an effective resistance plus a known offset — and batch 2 of the radiation
leg solved `p = R U` with a scalar Sherman–Morrison. The room is the same one-port with a vastly
richer `p̄_free`. **The existing rank-1 machinery transfers unchanged**, which is the same kind of
structural reuse batch 1 got by tensoring the bore's Yee cell rather than inventing a 3-D scheme.

On the body's side nothing is new either. A `ModalBody` with radiation weights `a_i` has volume
velocity `U = Σ a_i q̇_i`, and a port pressure `p̄` enters mode `i` as the generalised force `−a_i p̄`
(the **same** weights, by reciprocity — the fact `RadiatedBody` already relies on). One force-free
step gives the free centered velocity `U_free`, and the load correction is rank-1:

```
U = U_free − G · p̄ ,      G = (k/2) Σ_i a_i² / (m_i (1 + σ_i k))
```

Substituting the room's relation closes it in one line, and the solve is a **single division**:

```
U = (U_free − G p̄_free) / (1 + G R_room)
```

Since `G ≥ 0` and `R_room ≥ 0` always, `1 + G R_room ≥ 1`: the solve can never be singular, at any
sample rate, any grid, any body. **The port is unconditionally passive — no CFL of its own, no
stability guard**, exactly as `RadiatedBody` is and unlike the bridge springs, whose rank-1 block can
go negative. The room's own CFL (`λ ≤ 1/√3`) and the body's (`ωk < 2`) are unchanged; coupling them
adds no third condition. That is the property worth having, and it is a consequence of `R_room`
being a *resistance* — a positive number multiplying the same velocity that appears in the body's
dissipation quadratic form.

---

## 3. Scope — and what is deliberately deferred

**In scope:**

- The **port**: a node set with normalized volume weights, its `R_room`, and its local free pressure.
- `RoomLoadedBody`: a `ModalBody` loaded by an `AirBox` port — a **drop-in** for `ModalBody`, so
  `StringBodyBridge` / `StringPlateBridge` accept it as the body with **zero edits to
  `connection.py`, `body.py` or `radiation.py`**, giving the full `string → bridge → body → room`
  chain.
- **N instruments in one room**, provided their ports are **disjoint** (§5, §6.6) — each an exact
  independent scalar solve, and the payoff the lumped tier structurally cannot have: two bodies that
  hear each other, at `c₀`, with the delay measured.
- The **spread port** (a fixed-radius ball of nodes). Measured in before the build: it converges
  where a point port diverges, and it costs ~15 lines (§6.4). Without it the coupling is exact but
  its *magnitude* is a property of the grid.
- The energy ledger and its cancellation identity; the refusals of §6.

**Out of scope — say so, do not drift:**

- **Everything batch 1 deferred stays deferred**: PML, HRTF/ambisonics, scattering objects,
  non-rectangular geometry, viscothermal absorption. This batch adds a port, not a room feature.
- **A moving port.** The port node set is fixed at construction. A moving source is a resampling
  problem (and an energy-booking one) of its own.
- **Overlapping ports.** Refused, and §6.6 says why the refusal is *principled* rather than lazy:
  disjointness is exactly the condition that makes the per-port scalar solve exact, and supporting
  overlap would force a central N×N solve that breaks composition with the bridges.
- **A viewer batch.** Core work, batch 1's and radiation batch 3's precedent.
- **The distributed body↔room *area* coupling** (a plate radiating from every one of its nodes
  rather than through one lumped volume-velocity port). That is the natural batch 3 and is a
  different animal: the port becomes a matrix, not a scalar.

---

## 4. API

```python
room = AirBox(L=(3.2, 2.6, 2.4), fs=44100.0, h=0.0135, walls=impedance_from_zeta(4.0))

inst = RoomLoadedBody(
    body=ModalBody(freqs=..., fs=44100.0, masses=..., phi=..., radiation=a),
    room=room,
    at=(1.1, 1.3, 1.0),      # port centre (m), snapped to the nearest node
    radius=0.05,             # REQUIRED: ball radius (m), or None for a point port
)

for n in range(n_steps):
    inst.step(force)         # solves the port, corrects the body, QUEUES the injection
    room.step()              # the room advances ONCE, after every port has solved
```

**`radius` has no default, and that is §6.3 showing up in the signature.** A point port is a
legitimate thing to want — it is the cheapest exact-conservation fixture, and every structural test
uses one — but its *load magnitude* is a property of the grid, so a caller who gets one by default
gets grid-dependent physics without ever making a choice. Requiring the keyword forbids nothing (the
standing `unphysical-params-are-a-feature` rule still holds: any radius, any position, no materials
table) — it only refuses to pick silently on the caller's behalf.

**The port does not step the room, and that is a design decision, not an omission.** It is what makes
the API compose:

```python
bridge = StringBodyBridge(string=..., body=inst, K=...)   # inst is a drop-in ModalBody
for n in range(n_steps):
    bridge_a.step()          # string -> bridge -> body -> port, per instrument
    bridge_b.step()
    room.step()              # one room, one step
```

Had `RoomLoadedBody.step()` stepped the room itself, two instruments would step the room twice per
timestep and a string-driven instrument (where the *bridge* owns the step) could not be a member at
all. The cost is one extra line in the caller's loop, and a forgotten `room.step()` is caught loudly
rather than silently freezing the room: a port that is asked to solve twice with its injection still
pending raises (§6.7).

**Read-out and books.**

```python
inst.volume_velocity      # last centered U (diagnostic, mirrors RadiatedBody)
inst.port_pressure        # last centered p̄ the body was loaded by
inst.radiated_energy      # cumulative integral p̄ U dt — the work this body did on the room
inst.energy()             # body.energy() + radiated_energy   (mirrors RadiatedBody.energy)
room.energy()             # acoustic + dissipated − injected  (batch 1, unchanged)
```

**The conserved total of a whole scene is `Σ_j inst_j.energy() + room.energy()`**, and the reason is
the batch's neatest structural fact: each port's `radiated_energy` *is* the room's `injected`, seen
from the other side of the same terminal. Summing the ledgers cancels it identically:

```
Σ_j (E_body,j + inj_j)  +  (E_acoustic + dissipated − Σ_j inj_j)  =  Σ_j E_body,j + E_acoustic + dissipated
```

so the conserved statement contains no coupling term at all — which is precisely why a drift in it
is unambiguous evidence of a bug rather than of accounting.

---

## 5. The discrete scheme

Per step, for each port `j` (in any order), then one `room.step()`:

**1. The port's free pressure — local, `O(port nodes)`, and it must replicate `step()`'s order.**

```
p̄_free,j = Σ_n w_jn · ½ (p_free,n + p_n) ,
p_free,n = [ (p_n − k ρ₀ c₀² div(u^{n+1/2})_n) − β_n p_n ] / (1 + β_n)
```

The bracket is exactly batch 1's `step()` with the source term omitted: divergence, **then** the wall
closure. Computing it as a full-array operation would double the batch's dominant cost for no reason
— the divergence at one node needs only the six faces touching it.

**2. The port resistance — constant, computed once at construction.**

```
R_room,j = Σ_n w_jn² · k ρ₀ c₀² / ( 2 W_n (1 + β_n) )
```

The `2` is the centered (trapezoidal) pressure `p̄ = (p^{n+1} + p^n)/2`; `W_n` is batch 1's tensor
trapezoid node weight; **`(1 + β_n)` is the wall-closure denominator, and omitting it is a real bug
with a measured cost** (§6.1).

**3. The body's force-free step and the scalar solve.**

```
q̃^{n+1} = ModalBody.step(force)                       # whatever external force the bridge applies
U_free  = a·(q̃^{n+1} − q^{n−1}) / (2k)
U       = (U_free − G p̄_free) / (1 + G R_room)
p̄       = p̄_free + R_room U
q^{n+1} = q̃^{n+1} − p̄ · [ k² a_i / (m_i (1 + σ_i k)) ]
```

then refresh `_accel` from the *corrected* second difference (so `pressure()` carries the load — the
same reason `RadiatedBody` does it) and book `radiated_energy += k p̄ U`.

**4. Queue the injection** `q_n = w_jn · U` at each port node, and let `room.step()` do the rest.

**Why the energy cancels exactly.** The corrected state's centered volume velocity is exactly the `U`
that was solved for (`U_free − G p̄ = U` by construction), so the body's energy decrement telescopes
to precisely `k p̄ U`. The room's `injected` book is `k Σ_n p̄_n q_n = k Σ_n p̄_n w_jn U = k p̄ U` — the
same number, because the injection weights and the read-back weights are **the same vector `w`**.
That reciprocity is load-bearing: injecting volume-weighted but reading back an unweighted average
breaks the identity while looking entirely reasonable (§6.5).

**The reduction ledger entry** (alongside `R = 0` → bare body, `M_a = ∞` → `RadiatedBody`,
`Z = ∞` → rigid wall, `σ₁ = 0` → model #2, `nonlinear=False` → #5):

```
a_i = 0  (no volume velocity)  =>  G = 0, U = 0, nothing injected  =>  BIT-IDENTICAL to a bare ModalBody
```

Measured, bit-identical (`array_equal` true), not merely close — the family convention, and the check
that catches sign errors nothing else catches.

---

## 6. Traps — seven, all found by measuring before a line of core code

### 6.1 The wall-closure factor `1/(1 + β)` — the one that would have shipped wrong

Batch 1's `step()` adds the source term **before** applying the wall closure (`airbox.py:435` then
`:440`). So at a port on a *lossy wall node* the injection is divided by `(1 + β)` along with
everything else, and the naive `R_room = k ρ₀ c₀² / (2W)` is inconsistent with what the room actually
does. The plan's central formula was written that naive way first. Measured, `ζ = 1` wall, port on it:

| `R_room` spelling | max relative drift of the conserved total |
|---|---|
| **with** `1/(1 + β)` | **8.4e-15** |
| without (naive) | **1.9e-2** — twelve orders worse |

So a wall-mounted port is **supported**, not refused — but only because the factor is there. Note
what the failure would have looked like: perfect conservation in every interior-port test, and a
2%-per-run leak only when someone mounted a loudspeaker on a wall.

### 6.2 A port on an **open** face can do no work — and the books say everything is fine

An `"open"` (`Z = 0`, pressure-release) face pins `p = 0` at its nodes every step, so a port there has
`p̄_free = 0`, `R_room = 0`, and therefore `p̄ = 0`: the body radiates into a short circuit. Measured
over 400 steps: `injected = 0.000000e+00`, `acoustic = 0.000000e+00`, drift `8.6e-15`. **The run is
perfectly conservative and completely silent.** Nothing in the energy report can catch this, because
nothing is wrong with the energy — the physics is exactly right and exactly useless.

**Refuse at construction:** a port whose node set touches an open face raises, naming the face. This
is the family's standing preference for a loud refusal over a quiet wrong answer (batch 1's
out-of-room `node_index`, the juari's grid snap).

### 6.3 A **point** port's self-impedance diverges as `1/h` — it does not converge, and refining
makes it worse

The measurement that matters most, because it decides what the coupling *means*. Driving a point port
with a Gaussian volume-velocity pulse in a `ζ = 1`-walled 0.5 m room and reading `Z(ω) = p̄/q`, at
`λ` held constant (halve `h`, double `fs`):

| `h` | grid | 250 Hz | 500 Hz | 1 kHz | 2 kHz | ×previous |
|---|---|---|---|---|---|---|
| 13.50 mm | 37³ | 3.51e4 | 7.13e4 | 1.42e5 | 2.87e5 | — |
| 6.75 mm | 74³ | 7.05e4 | 1.42e5 | 2.84e5 | 5.69e5 | **1.98 – 2.01** |
| 3.375 mm | 148³ | 1.41e5 | 2.84e5 | 5.67e5 | 1.13e6 | **1.99 – 2.00** |

`|Z|` **doubles with every halving of `h`**, over two decades of refinement. It is not the source
cell's compliance (that would
diverge as `1/h³`, and `|1/ωC_cell|` is two orders larger than the measured `Z`) — it is the monopole
**near field**, `p ∝ 1/r`, evaluated at the only radius the grid has. Comparing against the
pulsating-sphere closed form `Z_a = (ρ₀c₀/S)·jka/(1+jka)` pins it precisely: `|Z| / |Z_sphere(a = h)|`
is **3.16 – 3.29** across all three grids and all four frequencies, near-constant in both, i.e.

> **a point port is a pulsating sphere of radius ≈ h/3.2.**

Two consequences, and the second is counterintuitive enough to be worth stating loudly:

1. The load is almost purely **reactive** at these frequencies (`ka ≪ 1` makes `Z_a ≈ jωρ₀/(4πa)`) —
   an **added mass**, not damping. So a point port mostly *detunes* the body rather than damping it,
   the same way radiation batch 3's reactance moved the pitch and made a 26%-off oracle 0.5%.
2. Because `a_eff ∝ h`, the added mass **grows as you refine the grid**. Refinement makes this
   artifact worse, not better. Order-of-magnitude for a realistic body (`a = 0.01 m²`, `m = 0.05 kg`,
   200 Hz): ≈ 2% frequency shift at `h = 13.5 mm`, ≈ 4% at 6.75 mm, ≈ 9% at 3.4 mm.

**This is a measured non-convergence and it ships as one.** The energy identity is exact regardless —
that claim is structural and stands — but the *magnitude* of a point port's load is a grid quantity,
not a physical one. A point port remains the right default for cheap structural tests, where only
exactness matters; it is the wrong thing to draw physical conclusions from, and the docstring must
say so where a caller will read it.

### 6.4 The **spread** port converges — so it ships, and the decision was made by measuring

Same measurement, same three grids, with the port spread over a **ball of fixed radius `a = 5 cm`**
(normalized volume weights, §5), which is the only difference:

| `h` | nodes in port | port volume | 250 Hz | 500 Hz | 1 kHz | 2 kHz | ×previous |
|---|---|---|---|---|---|---|---|
| 13.50 mm | 203 | 4.995e-4 m³ | 3.378e3 | 7.665e3 | 1.386e4 | 2.294e4 | — |
| 6.75 mm | 1743 | 5.361e-4 m³ | 3.263e3 | 7.434e3 | 1.337e4 | 2.195e4 | **0.957 – 0.970** |
| 3.375 mm | 13613 | 5.233e-4 m³ | 3.285e3 | 7.480e3 | 1.347e4 | 2.218e4 | **1.007 – 1.011** |

**×0.96 then ×1.01, against ×2.00 twice** — the load stops being a grid quantity the moment the port
has a size. The convergence is not monotone, and the fourth column says why: the staircased ball's
discrete volume wobbles around the exact `4πa³/3 = 5.236e-4 m³` (−4.6%, +2.4%, −0.06%) as whole nodes
fall in or out of it. That is the membrane batch's staircase, and it is the accuracy floor here —
worth knowing before someone reads the non-monotonicity as a scheme defect.
That settles the scope question §6.3 posed: the spread port is ~15 lines (a mask and a weight
vector; the scheme in §5 is already written for general weights), it costs **0.8 s and 22 s** at the
two grids, and without it the coupling has no physically meaningful magnitude. **It ships.**

**But it does not converge to `from_sphere(a)`, and the reason is physics, not error.** The measured
ratio to the pulsating-shell closed form is 1.11 at 250 Hz, rising with `ka` (1.99 at 2 kHz, where
`ka = 1.8` and the shell formula has saturated). A **uniformly injecting ball is not a pulsating
shell**: its volume-averaged self-pressure carries the classic uniform-sphere factor **6/5** at low
frequency (the same `6/5` as the mean potential of a uniformly charged sphere), i.e. the ball's
equivalent shell radius is `5a/6`. Measured 1.11 at `ka = 0.23` against 1.2 predicted at `ka → 0` is
consistent, and closing that gap properly needs a genuinely low-`ka` point.

**So the cross-tier oracle is scoped accordingly** (§7.10): assert the *convergence* — the port's
equivalent radius is grid-independent — and compare against the ball's closed form at low `ka`, with
the shape factor named. A tight assertion against `from_sphere(a)` without the `6/5` would be
wrong-by-design, and is exactly the kind of "oracle that looks right and charges its own error to the
physics" this project keeps refusing (radiation batch 3's missing reactance, batch 1's un-snapped
radius).

### 6.5 `W` versus `w_d`, and the reciprocity of the port weights

Two weight systems coexist in batch 1 and they are easy to "fix" into agreement, which breaks
everything: the source term divides by the **volume** weight `W = wx·wy·wz`, while `_divergence`
divides by the **per-direction** weight `w_d`. Both are right, and the proof is the adjointness that
makes the energy telescope: the face weight carries `h · (transverse trapezoid)`, so
`Σ_faces W_face (grad p) v = −Σ_nodes W_node p (div v)` only with that pairing. `R_room` uses `W`.

The port's own weights carry the same hazard one level up: the injection weights and the pressure
read-back weights must be **the same vector**. Volume-weighting the injection (physical) while
reading back a plain mean (also physical-looking) silently breaks the cancellation in §5.

### 6.6 Overlapping ports break independence — refuse, for a principled reason

Two ports sharing any node are not independent within a step: port A's injection changes port B's
`p̄`, so B's solve used a pressure that never occurred, and the body's booked loss no longer matches
the room's booked gain. Measured, two bodies in one 10×8×6 room over 400 steps, changing **only**
where the second port sits:

| two ports | max relative drift of the conserved total |
|---|---|
| **disjoint** — `(3,3,3)` and `(7,5,3)` | **7.1e-15** — exactly independent, as claimed |
| **coincident** — both snapped to `(3,3,3)` | **3.6e-2** |

So the N-instrument scope and the overlap refusal are the same measurement read in two directions.
Disjointness is exactly the condition under which the per-port scalar solve
is *exact*, and it is worth being explicit that the alternative was considered and rejected on
composition grounds, not difficulty: a simultaneous `(I + G R) U = U_free − G p̄_free` solve with the
symmetric PSD cross-resistance matrix `R_jk = Σ_n w_jn w_kn k ρ₀ c₀²/(2 W_n (1+β_n))` would handle
overlap and is provably passive too — but it requires one central object to own every port, which is
precisely what makes a per-instrument `StringBodyBridge` chain impossible. Disjoint ports are the
mild requirement (two instruments do not occupy the same air) and they buy composition.

The hazard is real rather than theoretical because of **snapping**: batch 1's own `snapped()`
docstring warns that two nearby points collapse onto the same node. Two ports 5 mm apart on a 13.5 mm
grid are one port. Detect at construction and raise, naming the shared node.

### 6.7 Ordering, and the forgotten `room.step()`

`p̄_free` must be read **before** `room.step()`, from the stored `u^{n+1/2}` — batch 1's step ordering
(pressure, then velocity) is what makes that well-defined, the same ordering that "dissolved the
plan's own trap" in batch 1.

**`free_pressure_at` deliberately ignores `room._pending`, and that is exact *iff* the ports are
disjoint.** With N ports solving in sequence, every port after the first runs while earlier ports'
injections sit queued. Reading them would be the obvious "fix" and it is the wrong one: for disjoint
ports a queued injection at another node cannot reach this node within the step (§2), so including it
changes nothing — while for overlapping ports it would make the solve *asymmetric* (B sees A, A never
saw B) rather than merely wrong, trading a caught error for an uncaught one. Disjointness is the
requirement precisely because it makes the cheap, order-independent read exact; the 7.1e-15 in §6.6
is that statement measured.

Consequently the forgotten-`room.step()` guard is **per-port, not global**: each port marks its own
injection pending and `room.step()` clears every mark. A global "is `_pending` non-empty" test would
fire spuriously on the second instrument of every scene. A port asked to solve while *its own* mark
is still set raises — the caller skipped `room.step()`, and the alternative is a frozen room and a
body loaded by a stale field, silently.

---

## 7. Oracles — what must pass (prototype numbers in brackets)

**Structural — the ones that make the batch trustworthy.** All rate-independent, all on ~10×8×6
grids where they are free.

1. **Exact conservation, interior port, rigid walls, lossless body.**
   `Σ inst.energy() + room.energy()` flat. [measured **2.0e-14** relative over 400 steps]
2. **Exact conservation with lossy walls and the port ON a wall node.** [**8.4e-15**]
3. **`R_room` is what the room actually does — and the test must be differential, not definitional.**
   Comparing the coupled step's `p̄` against `p̄_free + R_room·U` is a **tautology**: that expression
   is how `p̄` was computed, so it passes for any `R_room` whatsoever, including §6.1's wrong one.
   The real test steps the *room* twice from an identical saved state — once with `q = 0`, once with
   `q = U` — and asserts `(p̄_actual − p̄_zero)/U` equals the constructed constant. That is what
   catches the missing `1/(1 + β)` directly instead of waiting for a drift to accumulate, so run it
   at a wall node and a corner node, not just an interior one. [target < 1e-12 relative]
4. **The reduction: `a = 0` ⇒ bit-identical to a bare `ModalBody`.** [**exact**, `array_equal` true]
5. **Unconditional passivity.** Absurd coupling (`a_i` ×10³, huge `G`) with `R_room` driven as large
   as the grid allows must neither blow up nor lose conservation. Note which direction that is:
   `R_room ∝ k/W`, so a *coarse* grid makes `W` large and `R_room` **small** — to stress the solve
   use a **corner node on a fine grid**, where `W = h³/8` is eight times the interior weight and,
   with lossy walls, `β` is largest too. There is no CFL to find; the test's job is to prove that
   claim rather than trust it.
6. **The refusals raise**: port on an open face; two ports sharing a node; port outside the room
   (inherited from `node_index`); a port solved twice without a `room.step()`.
7. **Composition.** `StringBodyBridge(string, body=RoomLoadedBody(...))` conserves
   `E_string + E_body + E_conn + radiated + E_room` — the full chain, with `connection.py` untouched.

**Physical — the ones that make it mean something.**

8. **The room gives energy back, and the delay is right — asserted as bit-identity.** Two runs in
   rooms differing *only* in `Lz`, port `n` nodes from the `z1` wall: the body's `q` is
   **bit-identical** until the reflection can return, and differs immediately after. Both halves are
   the test — the second catches a coupling that never arrives, the first catches one that arrives
   early. Note the body's energy is *not* monotone before then (the near-field reactance hands energy
   back every cycle), so a monotonicity assertion here would be simply false; bit-identity is the
   right instrument. *This is the claim no `R(ω)` can make at any order: a one-port's impulse
   response is a decaying exponential, never a delayed echo.*
9. **Two instruments, one room.** Body B is at rest and stays at rest — **exactly zero**, not small —
   until A's disturbance can reach it, then moves. The arrival index is the oracle; the amplitude is
   a `1/r` sanity check, not an assertion.

   **The arrival index is `Manhattan`, not `Euclidean`, and that is the honest form of the test.**
   The 7-point stencil spreads influence by one node along one axis per step, so the numerical domain
   of dependence after `m` steps is the `L1` ball of radius `m`: an off-axis listener receives a
   machine-precision *precursor* before `r/c₀`. The physical wavefront still arrives at `r/c₀`; the
   precursor is the grid's, and asserting against the Euclidean time would fail for a reason that has
   nothing to do with the coupling. **Measured**: two ports 6 nodes apart (Manhattan) in a 10×8×6
   room, `h = 5 cm`, `fs = 20 kHz` — body B's first nonzero motion is at step **7**, i.e.
   `Manhattan + 1`, against **17.5** steps for the Euclidean `r/c₀`. The precursor is 2.5× early, and
   an oracle written against `r/c₀` would have failed by that factor for a reason having nothing to
   do with back-reaction. The `+1` is the injection's own step; pin it by measuring, per batch 1's
   precedent of measuring the constant rather than deriving it twice. [drift over that run:
   **5.7e-15**]
10. **Cross-tier: the spread port's load is a physical quantity, and the point port's is not.**
    Two assertions, both cheap at the coarse grid (0.8 s measured):
    - *Convergence:* the fixed-radius port's equivalent radius is grid-independent — `|Z|` changes by
      **< 5%** under a halving of `h` [measured 3–4%], while a point port's **doubles** [measured
      1.98–2.01, three grids]. The contrast is the assertion; both halves matter.
    - *Magnitude:* at low `ka` the spread port matches the uniformly-injecting **ball**, i.e.
      `RationalAirLoad.from_sphere(5a/6)`, matching *both* parts of `Z_a` per radiation batch 3's
      lesson that the reactance is not optional. Tolerance set by a measured low-`ka` run during the
      build, not guessed here.

    The expensive whole-sweep version of this figure belongs in the diagnose script (batch 1's
    precedent), where the point port's divergence gets printed as the measured refusal it is.

---

## 8. Cost budget — owned, not discovered

The structural tests (1–7) are the batch's bulk and cost nearly nothing: ~10×8×6 rooms, a two-mode
body, a few hundred steps. Tests 8–9 need a room big enough for a travel time to be resolvable, but
they assert an *arrival index*, not a spectrum, so a few hundred steps on a ~30³ grid suffices.
Test 10, if it ships, is the only expensive one and belongs in the diagnose script under batch 1's
precedent (its own `diagnose_airbox.py` runs ~40 s for one large room).

**Commitment: the batch adds under 20 s to the local suite.** Baseline to be pinned by a measured run
immediately before the build starts (batch 1 measured 1235 green).

---

## 9. Deliverables

- `physsynth/core/airbox.py`: `RoomPort` (or the port folded into the body class) and
  `RoomLoadedBody`, plus the `O(port)` free-pressure read and the pending-injection guard. **No edits
  to `radiation.py`, `body.py`, `connection.py`, `bore.py`.** `G` is recomputed from public `body`
  attributes — do not reach for `RadiatedBody._G`. And because `RoomLoadedBody` delegates read
  accessors to the body via `__getattr__` (the drop-in property §3 depends on), **`energy()` must be
  an explicit override**, or it silently delegates and returns the bare modal energy — the total
  without its coupling channel, which is exactly the number `RadiatedBody`'s docstring already warns
  not to assert on.
- `tests/test_airbox_port.py` (§7.1–7), `tests/test_airbox_scene.py` (§7.8–9), and the cross-tier
  §7.10 per §6.4. `tests/helpers.py`: `make_room_loaded_body`.
- `scripts/diagnose_airbox_port.py`: the exact-cancellation ledger; the reflection-arrival trace of
  §7.8; the §6.3 port-impedance-vs-`h` divergence, printed as the measured refusal it is.
- Docs: HANDOFF §12H line, README model list, memory `air-box-state` (+ `MEMORY.md` pointer and the
  `docs/memory/` mirror).

**Acceptance:** all of §7 green, full suite green, `ruff check .` clean, added suite time < 20 s
locally.

**Met (2026-07-28).** All of §7 green — §7.10 in the amended form the status block explains — and
the full suite is **1287 green** (1235 before the batch, so nothing pre-existing moved).
`ruff check .` clean. The batch adds **52 tests in 8.9 s**, well under half the budget, and roughly 7 s of
that is the single cross-tier sweep; every structural oracle runs on a 693-node room where it is
free, and the most expensive one is 0.33 s. `scripts/diagnose_airbox_port.py` runs its three figures
in ~3.5 min, essentially all of it the 74³ level that §8 said belongs here rather than in the suite.

The measured highlights, against §7's predictions:

| oracle | predicted | measured |
|---|---|---|
| conservation, interior port | 2.0e-14 | **1.1e-14** |
| conservation, port on a lossy wall / edge / corner | 8.4e-15 | **~1e-14** |
| `R_room`, differential, at four sites × two wall types | < 1e-12 | **exact to 1e-12** |
| `a = 0` ⇒ bare `ModalBody` | bit-identical | **`array_equal` true, 200 steps** |
| disjoint two-port scene | 7.1e-15 | **< 1e-10** over 400 steps, and bit-identical under solve order |
| reflection round trip | `2d` | **`2d + 1`**, at `d` = 3, 5, 7 |
| second body's arrival | Manhattan + 1 | **exactly Manhattan**, at 6 and 12 nodes |
| point port's grid dependence | `|Z|` ×2.00 | `a_eff` **×0.493, ×0.496** (equivalently `a_eff/h` = 0.324, 0.320, 0.317) |
| spread port's grid dependence | `|Z|` ×0.96, ×1.01 | `a_eff` **×1.045, ×1.038** |
| spread port vs the ball's `5a/6` | ~1.2 at `ka → 0` | **1.0034** once the port is compact — but 1.086 in a small room, which is the room |

Two of those rows are the batch's real content. The point port's `a_eff/h` holding at 0.32 across
three grids says the artifact is *exactly* proportional to `h`, so there is no refinement that
escapes it — which is why `radius` has no default. And the last row is the one the plan got wrong:
the closed form was there to be matched all along, but the number the prototype measured was mostly
the room, not the port. Comparing anything to a closed form means getting the room out of the way
first — and it took a *sweep*, not a tighter tolerance, to see that.
