# The 3-D air box — a room the sound actually crosses (air-box batch 1)

> **Status: PLANNED (2026-07-28).** The human's pick at the post-Phase-D fork, taken after the
> `R(ω)` arm landed (`radiation-frequency-dependent-plan.md`). Every number quoted in §6 and §7
> below was **measured on a throwaway prototype before this plan was written** — the layout, the
> energy identity, the wall closure and the free-field window were all settled empirically, not
> assumed. Prototype lives outside the repo (`M:\claud_projects\temp\airbox`).

---

## 1. Why — the refusal being discharged

`core/radiation.py` models the air as a **lumped port**: one volume-velocity terminal, one
impedance, one listening distance. All three radiation batches share that assumption, and each
one said so. The lumped air cannot represent anything that depends on *where you are*:

- **room modes** — the air's own resonances, which colour every recorded instrument;
- **the finite travel time across a space**, and the comb filtering of a direct sound against its
  reflections;
- **more than one listener**, or a listener who moves;
- **any source that is not compact** — a plate is not a point, and at 4 kHz a 40 cm plate is three
  wavelengths across.

Batch 3 named this explicitly in its own §3: *"3-D FDTD air box — HANDOFF §12H. A different animal
(a spatial grid, not a lumped port). The lumped rational impedance is precisely the tier below
it."* This is that animal.

**The claim this batch exists to make, in one line:** *the distributed air contains the lumped air
as its free-field limit.* Feed the same source to `AirRadiation` (batch 1) and to `AirBox`, put the
walls far enough away, and the room reproduces the monopole law `p = ρ₀Q''/(4πr)` — measured in the
window before the first reflection arrives. Then turn the walls up and get everything the lumped
tier structurally cannot.

---

## 2. The physics — the bore's Yee cell, one dimension up

Same two first-order conservation laws as `core/bore.py`, now with a vector velocity and no area
profile:

```
rho0 du/dt = -grad p          (momentum / Euler)
dp/dt      = -rho0 c0^2 div u (continuity / mass)
```

**State layout — node-centered pressure, face-centered velocity.** `p` lives at the
`(Nx+1)(Ny+1)(Nz+1)` grid nodes, *including the walls*; `u_x` lives on the `Nx(Ny+1)(Nz+1)` faces
between neighbouring nodes in `x`, and likewise `u_y`, `u_z` — each half a timestep offset in
time. This is the bore's layout tensored up, and it is chosen over the (more common in room
acoustics) cell-centered convention for one reason: **the bore's boundary machinery transfers
unchanged.** A rigid wall is the `h/2` half-cell trapezoidal node weight and needs no ghost
stencil; an impedance wall is the bore's radiating-end 1×1 collapse. Both are already proven in
this repo.

Uniform spacing `h` in all three directions is **required**, not a convenience: an anisotropic grid
breaks the isotropic CFL and the tensor-cosine exactness that §7.2 rests on.

**Leapfrog, with the same adjoint pairing that makes the bore conserve:**

```
u_x^{n+1/2} = u_x^{n-1/2} - (k / (rho0 h)) (p_{i+1} - p_i)^n        [and y, z]
p^{n+1}     = p^n - k rho0 c0^2 * div(u^{n+1/2})
```

**CFL.** `lambda = c0 k / h <= 1/sqrt(3)` in 3-D. Assert at construction, reject otherwise. Unlike
the 1-D string there is **no dispersionless `lambda`** — the same fact the membrane batch recorded
for 2-D. At `lambda = 1/sqrt(3)` the standard Yee stencil is exact along the grid *diagonals* only;
axis-aligned propagation stays dispersive. Say this in the docstring; do not tune toward it as if
it were the string's `lambda = 1`.

---

## 3. Scope — and what is deliberately deferred

**In scope:** the rigid rectangular box; tensor-trapezoid node weights; the exact discrete energy;
one wall closure covering rigid / open / locally-reacting impedance; a point volume-velocity
source booked as an energy channel; a listener read-out; the room-mode oracle in both its exact
and continuum tiers; the free-field cross-tier oracle against batch 1.

**Out of scope — say so, do not drift:**

- **Back-reaction (the room loading the source).** This batch is the **read-out tier** — the exact
  position `AirRadiation` occupied in batch 1, and the module's own history says build read-out
  first. The source injects into the room and the room does not push back on it. A two-way port
  (room ↔ body, provably passive) is the natural batch 2 and repeats the 1→2 pattern.
- **PML / higher-order absorbing boundaries.** The locally-reacting impedance wall is a
  *first-order, normal-incidence* absorber. It is passive and closed-form-checkable, which is what
  this project asks of a boundary; it is not an anechoic chamber (see trap §6.4).
- **HRTF / ambisonics / directivity encoding** (HANDOFF §12H's other half). The box produces a
  pressure field; turning that into a binaural or B-format signal is a separate, non-physics batch.
- **Scattering objects inside the box, moving sources, and room geometry beyond a rectangle.**
  A staircased obstacle is the membrane batch's lesson repeated in 3-D and deserves its own batch.
- **Viscothermal / air-absorption loss.** Frequency-dependent bulk loss (the `sqrt(omega)` and
  humidity-dependent terms) is the bore's deferred wall-loss problem in 3-D.
- **A viewer batch.** Phase D is closed. This is core work — batch 3's precedent.
- **Audio-rate whole-room runs as a routine thing.** See §8; the cost is real and the plan owns it.

---

## 4. API

```python
AirBox(
    L=(Lx, Ly, Lz),          # room dimensions (m)
    fs=44100.0,              # sample rate; k = 1/fs
    h=0.02,                  # requested grid spacing (m); uniform, all three axes
    walls="rigid",           # or a per-face dict, see below
    rho0=RHO0_AIR, c0=C0_AIR,
)
```

**Grid snap is the resolution, and it is reported.** `N_d = round(L_d / h)` and the *actual* room is
`L_actual = (Nx h, Ny h, Nz h)`, exposed as `box.L_actual` alongside the requested `L`. This is the
juari batch's precedent (`grid snap IS the resolution`) rather than silently resampling or
silently refusing.

**Walls — one parameter, three behaviours.** `walls` is either a token or a dict over the six faces
`"x0" "x1" "y0" "y1" "z0" "z1"`. Each face is `"rigid"` (≡ `Z = inf`), `"open"` (≡ `Z = 0`, a
pressure-release / Dirichlet face), or a **float** specific acoustic impedance `Z` in Pa·s/m.
`Z = rho0*c0` is the normal-incidence matched value. Per the standing
`unphysical-params-are-a-feature` rule the primary parameter is the **effective** `Z`, with
`zeta = Z / (rho0 c0)` available as a helper — never a materials table.

**Source and read-out.**

```python
box.inject(q)                      # volume velocity q (m^3/s) at the source node — the primitive
box.step()
box.pressure_at(point)             # nearest-node pressure (the snap is reported, not hidden)
box.energy(); box.dissipated_energy(); box.injected_energy()
```

`inject` takes **volume velocity**, because that is what the continuity equation's source term is.
The lumped tier's `_VolumeAccelerationSource` protocol hands out `Q''` instead, so driving the box
from the existing chain uses `ReactiveRadiatedBody.volume_velocity` — the exact quantity, already
public, so the full `string → bridge → body → lumped load → room` chain works with **zero edits to
`body.py`, `connection.py` or `radiation.py`**. Do *not* integrate `Q''` internally to fake the
protocol: an accumulating integrator has a DC drift mode with nothing to restore it, and it would
be a silent one.

---

## 5. The discrete scheme

**Weights.** Node weight is the tensor trapezoid `W_ijk = wx_i wy_j wz_k`, with `w = h` interior and
`h/2` at each wall — so an edge node carries `h·h/2·h/2` and a corner `h³/8`. The `x`-face weight is
`h · wy_j · wz_k` (full `h` along the face normal, trapezoid transverse). Divergence at a node is
`sum_d (Delta_d u_d) / w_d` — note the **per-direction** weight, not `W`.

**Energy — the bore's cross-time product, in 3-D:**

```
E^n = (1/2) sum_nodes  (W / (rho0 c0^2)) (p^n)^2
    + (1/2) rho0 sum_faces W_face  u^{n+1/2} u^{n-1/2}
```

The velocity term is the **cross-time product** of the staggered variable, never the same-time
square — the string/bore lesson, and the reason this conserves to machine precision instead of
oscillating. It telescopes to exactly zero change per step because divergence uses the transpose of
the gradient momentum uses, and the weights cancel in the pairing.

**One wall closure, three boundary types.** At a wall node, the outward normal velocity is
`u_n = pbar / Z` with `pbar = (p^{n+1} + p^n)/2` — centered, i.e. implicit, the standing
VK/bow/radiation-load lesson. Substituting into the boundary node's divergence collapses to a
scalar per node:

```
p^{n+1} = (p_rigid - beta p^n) / (1 + beta) ,     beta = k rho0 c0^2 / (2 Z w_wall)
```

where `p_rigid` is the force-free rigid-wall update. `w_wall = h/2` for a face node; a node on an
**edge or corner touches two or three walls and simply sums their `beta`** (admittances add) — still
1×1, no coupled solve anywhere. The two special cases fall out of the same line:

> **`Z = inf` ⇒ `beta = 0` ⇒ exactly the rigid update** (bit-identical, measured) ·
> **`Z = 0` ⇒ `beta = inf` ⇒ `p^{n+1} = 0`**, the open face.

That is this batch's entry in the family's reduction ledger, alongside `R=0 ↔ bare body`,
`M_a=inf ↔ RadiatedBody`, `sigma_1=0 ↔ model #2`, `nonlinear=False ↔ #5`.

**Energy identity with walls.** The same telescoping now leaves exactly the wall flux:

```
E^{n+1} - E^n = -k sum_wallnodes A_node * pbar^2 / Z   <= 0
```

so `E + integral(dissipated)` is flat and `E` alone is monotone. `A_node` is the node's wall area
(the transverse trapezoid weights).

**Source.** A **soft** (transparent) source: `p_src += k rho0 c0^2 q / W_src` before the step. It
injects `integral p q dt`, booked as `injected_energy()`, so the conserved statement is
`E + dissipated - injected = const`. A *hard* source (assigning `p`) would not be passive and is
not used.

---

## 6. Traps — all four found by measuring, before a line of core code

1. **The energy/dissipation pairing instant.** `E^n` pairs `p^n` with `u^{n±1/2}`, so it must be
   evaluated *mid-step* — after the velocity update, before the pressure update. Pairing `p^{n+1}`
   with the `u` levels straddling `n` instead produced a **2.2e-2** "drift" in the prototype that
   read exactly like a broken scheme; the dissipation accumulator has the same off-by-one. Once
   both were snapshotted at the same instant the total went to **7.5e-16**. This is the single
   most likely place a drift hunt lands, and it is a *bookkeeping* bug, not a physics one.
2. **The exact mode initialiser is `omega`-free.** For `p^n = cos(omega_d n k)·mode`, the exact
   discrete half-step-back velocity is `u^{-1/2} = (k / (2 rho0 h)) * diff(mode)` — no `omega` in
   it. The plausible-looking continuum form `sin(omega k/2)/(rho0 omega h)` is *nearly* right and
   leaves a `1e-4`–`1e-1` shape error that masquerades as scheme inaccuracy. Deriving the exact
   one also *is* the derivation of the dispersion relation in §7.2.
3. **The free-field window is per-probe and it is hard-edged.** With `p` on the walls a rigid wall
   is a pressure **antinode**, so the first reflection arrives at *full* amplitude — there is no
   forgiving roll-off. At radius `r` in a box of side `L` with the source centred, the clean window
   is `t ∈ [r/c0, (L-r)/c0]`, of duration `(L-2r)/c0`, which **collapses to zero as `r → L/2`**.
   Running every probe to one global stop time truncates the far probes' pulses before they even
   arrive and turns the measured `1/r` slope into `-2.5`. Sizing rule, to be applied in the test:
   for a pulse of effective duration `T_p ≈ 8 sigma` (`sigma = 1/(2 pi f0)`),
   `r_max = (L - c0 T_p)/2`, and each probe is windowed at its own `(L-r)/c0`.
4. **A locally-reacting `Z = rho0 c0` wall is not anechoic.** It is matched at **normal incidence
   only**; obliquely it reflects, and the free-field oracle must not lean on it. The window in
   trap 3 is the mechanism; absorption is a convenience. Measured normal-incidence reflection
   against the closed form `|R| = |(zeta-1)/(zeta+1)|` is a *separate* oracle (§7.5), not a
   substitute for windowing.
5. **The read-out has a fixed sub-sample lag — report it, do not assert it away.** The recorded
   sample `n` holds `p^{n+1}`, and the staggering adds another half step, giving a **constant
   ≈ -1.5 sample** offset, independent of `h` and of `r` (spread across radii < 0.51 sample).
   Batch 1's delay line is an integer-sample, dispersionless construction; the FDTD arrival is
   dispersive with an O(h) effective source origin. So **assert gain, report lag** — exactly the
   split the advisor called for. Fitting gain without allowing a *negative* lag pins the search at
   its boundary and inflates the residual from `9e-3` to `0.59`.

---

## 7. Oracles — what must pass (prototype numbers in brackets)

**Structural:**

1. **Energy conservation, lossless rigid box.** Random-field start, several aspect ratios: relative
   drift `< 1e-12`. *[measured **5.3e-16 … 2.0e-15**]*
2. **Passivity + booked total, impedance walls.** `E` monotone non-increasing; `E + dissipated`
   flat to `< 1e-12`. *[measured **7.5e-16**, monotone true]*
3. **Reductions, bit-identical.** `Z = inf` on every face ≡ the rigid box, `np.array_equal`, not
   `allclose`. `Z = 0` ⇒ that face is exactly `0.0`. *[measured: both hold exactly]*
4. **Source booking.** With a soft source, `E + dissipated - injected` flat to `< 1e-12`.

**Spectral — the money oracles, in two tiers:**

5. **The exact discrete room mode.** The tensor cosine `cos(l pi i/Nx) cos(m pi j/Ny)
   cos(n pi k/Nz)` is an **exact** eigenvector of the discrete Neumann Laplacian *including at the
   `h/2` wall nodes*, with eigenvalue `-mu^2`, `mu^2 = (4/h^2) sum_d sin^2(l_d pi / (2 N_d))`.
   Initialised with the §6.2 velocity it oscillates at exactly
   `omega_d = (2/k) arcsin(c0 k mu / 2)` — assert the field stays proportional to the mode shape to
   machine precision over hundreds of steps. *[measured shape error **1.2e-14 … 4.7e-14** over 500
   steps]* This is a **tier above** the membrane's Bessel test, which was convergence-rate only,
   because the rectangle is grid-aligned and nothing is staircased.
6. **The continuum room modes.** `f_lmn = (c0/2) sqrt((l/Lx)^2 + (m/Ly)^2 + (n/Lz)^2)` — the
   textbook rectangular-room formula — recovered at **order 2**. *[measured rates **2.012, 2.003,
   2.001**]*
7. **CFL.** `lambda > 1/sqrt(3)` rejected at construction; a run at `lambda` just under it stays
   bounded over a long run.

**Cross-tier — the headline:**

8. **Free field = the lumped monopole.** Point source, walls far, per-probe reflection-free window
   (§6.3). At each radius, least-squares fit gain and lag against
   `p = rho0 qdot(t - r/c0) / (4 pi r)`. Since the closed form already contains `1/(4 pi r)`,
   **`gain == 1` at every radius *is* the `1/r` law** — a far better estimator than a log-log slope
   fit on dispersive pulse peaks. Assert `max|gain - 1|` falls with refinement; **report** the lag.
   *[measured `max|gain-1|` = **1.97e-2 (N=48) → 6.85e-3 (N=64) → 1.65e-3 (N=96)**; lag ≈ -1.5
   samples, spread < 0.51 sa; post-fit residual 0.175 → 0.037, order ≈ 2]*
9. **Wall reflection coefficient.** Quasi-1D duct, pulse onto an impedance face: measured `|R|`
   vs `|(zeta-1)/(zeta+1)|` at several `zeta`. *[measured err **3.1e-3 … 1.3e-2**]* — a
   convergence tier, not machine precision; assert accordingly and let it tighten with `h`.
10. **Cross-model agreement with `Bore`.** A box one cell thick in `y` and `z`, rigid on the
    transverse faces, reproduces the 1-D bore's closed-open odd-harmonic series. **This is an
    `allclose` cross-model check, not a family reduction** — `Bore` carries the area `S` through
    both updates and the box carries none, so the float operation order differs and bit-identity
    is not promised.
11. **Full chain, zero edits elsewhere.** `string → bridge → ReactiveRadiatedBody → AirBox`, driven
    by the body's public `volume_velocity`, runs with no edits to `body.py`, `connection.py` or
    `radiation.py`.

---

## 8. Cost budget — owned, not discovered

3-D is the first model in this repo where grid cost is a design constraint rather than a footnote.
Measured single-step cost (NumPy, this machine): `N=32` 0.21 ms · `N=48` 1.74 ms · `N=64` 3.52 ms ·
`N=96` ≈ 12 ms. The free-field runs above cost 0.31 s (`N=48`), 1.02 s (`N=64`), 6.5 s (`N=96`).

CI runs `pytest -q` with **no deselect on a 2-core runner**, so a `slow` mark still costs — assume
2–3× the local time. **Commitment: the batch adds under 30 s locally**, which means the free-field
convergence pair tops out at `N=64` and `N=96` appears at most once. Structural and modal tests run
on grids of ~`10×8×6`, where they are effectively free. If a test needs more than that, it needs a
reason written next to it.

Note also that a genuinely audio-rate room is *not* what these tests do: at `fs = 44.1 kHz` the CFL
forces `h >= sqrt(3) c0 / fs ≈ 1.35 cm`, so a 1 m³ room is ~74³ nodes and one second of audio is
44 100 steps ≈ 4 minutes. Tests choose `(h, k)` as a consistent pair at whatever rate is cheapest;
the modal oracles are rate-independent. The diagnose script may run one audio-rate room; the test
suite may not.

---

## 9. Deliverables

- `physsynth/core/airbox.py`: `AirBox`. No edits to `radiation.py`, `body.py`, `connection.py`,
  `bore.py`.
- `tests/test_airbox_energy.py` (§7.1–4), `tests/test_airbox_modal.py` (§7.5–7),
  `tests/test_airbox_freefield.py` (§7.8–9), plus the cross-checks §7.10–11.
  `tests/helpers.py`: `make_airbox` + defaults.
- `scripts/diagnose_airbox.py`: the room-mode spectrum measured against the closed form, the
  three-channel flat energy total, and the free-field `1/r` fit against batch 1's lumped law.
- Docs: HANDOFF §12H line, README model list, memory `air-box-state` (+ `MEMORY.md` pointer, and
  the mirror in `docs/memory/`).

**Acceptance:** all of §7 green, full suite green, `ruff check .` clean, added suite time < 30 s
locally. Baseline to be pinned by a measured run immediately before the build starts.
