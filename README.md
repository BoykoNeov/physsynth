# Physical Synthesis Simulator

An accuracy-first, **energy-based** physical-modeling sound-synthesis engine. The foundation is the
Bilbao finite-difference (FDTD) framework: numerical schemes whose discrete energy is provably
conserved (lossless) or monotonically dissipated (lossy), so correctness is **measured against
closed-form physics**, not judged by ear.

> Orientation: read [`CLAUDE.md`](CLAUDE.md) for the non-negotiables and [`HANDOFF.md`](HANDOFF.md)
> for the full spec, math, and milestone definitions. This README is the operational quickstart.

## Why this design

- **Validation is code, not listening.** Every resonator exposes `energy()`. A lossless run must keep
  `max|Eⁿ − E⁰| / E⁰ < 1e-10`; a lossy run must decrease monotonically. Detected partials are checked
  against analytic oracles (e.g. `fₙ = n·c/(2L)` for the ideal string).
- **Headless core.** `physsynth/core/` is pure NumPy/SciPy — no audio I/O, no plotting. Visualization
  and (future) wrappers depend on the core, never the reverse, so the physics ports cleanly later.
- **Unifying abstraction:** `exciter → resonator (± nonlinear coupling) → body/radiation`.

## Layout

```
physsynth/
  core/        # headless DSP: operators, resonators, exciters, engine (no I/O, no graphics)
  analysis/    # analytic oracles (modal frequencies) + spectral partial detection
  viz/         # diagnostic plots (matplotlib, Agg backend) — imports core, never vice versa
web/           # interactive viewer (wrapper): local HTTP backend + static frontend — imports core
tests/         # validation harness: energy, modal, convergence, stability
scripts/       # runnable diagnostics (e.g. diagnose_ideal_string.py)
docs/dev/      # per-feature dev-docs (plan / context / tasks)
```

## Setup

```bash
python -m venv .venv
. .venv/Scripts/activate        # Windows; use .venv/bin/activate on POSIX
pip install -e ".[dev]"
```

(NumPy/SciPy/Matplotlib/pytest may already be present in your global Python; the editable install
just wires up the `physsynth` package and dev extras.)

## Run the validation suite

```bash
pytest                  # full harness
pytest -m "not slow"    # skip the longer convergence/spectral sweeps
pytest -n 8 --dist loadgroup      # same tests, spread across cores (needs pytest-xdist)
```

The full harness is long — it simulates every model in the repo. Prefer the parallel form for a
whole-suite run and the plain form when debugging a single test, where worker startup is pure
overhead and the serial run gets the full multi-threaded BLAS (measurably faster per test).

Pick the worker count deliberately rather than reaching for `-n auto`: `auto` takes one worker per
*logical* core, which oversubscribes a hyperthreaded desktop that is also doing other work. `-n 8`
is the configuration actually measured here. CI uses `-n auto` because its runners are small and
otherwise idle.

`--dist loadgroup` matters: a handful of modules carry `pytest.mark.xdist_group` because their
module-scoped fixtures are expensive (the whirling and phantom sweeps run 30–110 s of simulation
*before* their first assertion). The mark keeps those on one worker so the fixture is built once;
everything else — including the 336-test web-backend module — scatters freely.

A parallel run saturates every core, which makes the rest of the desktop sluggish. To keep working
while it runs, prefix any invocation with the priority wrapper — it forwards its arguments to pytest
unchanged, and the xdist workers inherit the lowered priority:

```bash
python scripts/nicepytest.py -n 8 --dist loadgroup
```

## Generate diagnostics

```bash
python scripts/diagnose_ideal_string.py
```

Writes `out/` figures: energy-vs-time, detected-vs-analytic partials, and a grid-convergence plot.

## Interactive web viewer

A local backend recomputes a model **offline** on each parameter change and streams the displacement
field + audio + energy to a browser, which animates the string (slow-motion, so the vibration is
visible), plays the sound, and shows the live energy-drift / passivity and partials diagnostics.
Accuracy-first: no in-browser physics, no real-time port — the validated Python core stays the single
source of truth (architecture B; see `docs/dev/web-viewer-plan.md`).

One model is deliberately **viz-only**: the geometrically-exact string carries a longitudinal wave at
`c_long = sqrt(EA/rho)` ≈ 22× the transverse speed you hear, so resolving it (which is the whole
point of that model) forces a sample rate ~22× higher and a second of audio would be ~10 minutes of
compute. It shows orbits instead — including one thing no other model here can draw at all.

```bash
python web/server.py            # then open http://localhost:8000
```

The physics lives entirely in `physsynth/core`; `web/` is a wrapper (`serialize.py` packs the payload,
`server.py` is a thin `ThreadingHTTPServer` shell) and never the other way around.

## Status

Complete and validated:

- **String family** — #1 ideal, #2 stiff, #3 frequency-dependent damped, **#9 tension-modulated**
  and **#10 geometrically-exact** (nonlinear; both below).
- **2D** — #4 circular membrane, #5 simply-supported Kirchhoff plate, #5b free-edge (FFFF) plate with
  Chladni patterns; plus the #5b-pre free–free Euler–Bernoulli beam (free-boundary de-risk).
- **Nonlinear** — #6 von Kármán coupled plate, **all 6 Parts** (bracket, Airy stress solve,
  conservative Picard resonator, validation, pitch-glide/energy-exchange diagnostics, and Part 6 the
  **free-edge cymbal/gong** — energy-conserving nonlinear coupling on a free rectangle, with the
  crash cascade and curved-Chladni modes).
- **Tension-modulated string** — #9, the **string family's nonlinearity** (`core/string_nonlinear.py`):
  displacing a string *stretches* it, raising tension, raising pitch — hit it hard and the note starts
  sharp and glides down (+80 % measured). Kirchhoff–Carrier: the tension `T₀+(EA/2L)∫u_x²` is a scalar
  functional of the whole state, so the quartic potential needs a conservative *implicit* scheme (model
  #6's lesson) — but because it is **quadratic in the stretch**, the energy-conserving tension is just
  the plain midpoint `T₀+(EA/2L)·Ī`, and the step reduces to a **scalar** root-find. Only the
  *nonlinear excess* is averaged at θ=½, so `EA=0` is model #3 **bit-for-bit**. Lossless drift 3.5e-13
  at 82 % nonlinear energy and 10× tension, and 1.4e-13 from a broadband pluck.
  Unlike model #6 it keeps a **closed-form nonlinear oracle**: a single mode reduces *exactly* to a
  Duffing oscillator, so hardening has an elliptic-integral frequency and a `cn` waveform the FDTD
  lands on (1.3e-3 / 3.7e-4). It also reproduces the **parametric instability of single-mode motion** —
  above `ΔT/T₀≈3` the tension, pumping at `2ω_m`, drives the neighbouring modes through Mathieu
  resonance and the mode disintegrates *while energy is conserved to 1e-13* (physics, not a blow-up:
  refinement-invariant onset and unstable modes). Planar modal exchange only — out-of-plane whirling
  and true phantom partials need a geometrically-exact (two-polarization / longitudinal) string.
- **Geometrically-exact string** — #10 (`core/string_geometric.py`), which **pays both of those
  debts**. Three coupled fields — two transverse polarizations `u`, `w` and the longitudinal `v` —
  with the exact stretch `Λ=√((1+v_x)²+u_x²+w_x²)`, so tension is a **field**, not #9's scalar
  functional. `EA=T₀` is model #3 bit-for-bit ×3; a discrete gradient on `mean(Λ)` (*not* `Λ(mean)`)
  keeps the lossless drift at 1.5e-16 through it.
  - **Phantom partials** (Conklin 1999): the excess carries `r²v_x/2`, quadratic in the transverse
    fields and *linear* in the longitudinal one, so two partials `f₁`, `f₂` drive `v` at `f₁±f₂` and
    `2f₁`, `2f₂` — combination tones landing where **no** partial exists, read at the bridge force
    `EA·v_x(0)`, the channel that actually radiates in a piano. #9 has no `v` to put them in.
  - **The polarization discriminator**: a *circular* mode holds `r²` time-independent, so the
    longitudinal forcing is **static**. Same string, same amplitude, opposite longitudinal spectrum,
    from polarization alone — and the null is not a quiet string: the circular run is **2× as
    energetic and 2× as stretched**, and radiates **113,000×** less. The nonlinearity is not off; it
    is on and silent.
  - **Out-of-plane whirling** (Gough 1984) — and the honest version is that an isotropic string
    **cannot** whirl: `w→−w` is a reflection symmetry, so a planar IC stays planar *bit-exactly*
    (`max|w| == 0.0`), and the rotation generator pins both Floquet multipliers at `+1` — marginal,
    never exponential (measured: a degenerate string's seeded envelope grows **secularly**, `1:2:3:4`
    to 1.3 %). Whirling is a **threshold** instability and needs the degeneracy broken, which is what
    the per-polarization `κ_u ≠ κ_w` (a non-circular cross-section) is for. Break it and the same
    `2Ω` tension pump that disintegrates #9's planar mode aims at the *other polarization* instead of
    the neighbouring *modes* — the same Mathieu resonance, a different target, and the one #9
    structurally cannot have. The tongue is `0 < Δω₀² < εA²/2`, mapped: growth
    `1.0 → 14.7 → 76.3 → 37.4 → 8.4 → 1.63×` across `Δ/εA² = 0 → 0.8`, peaking at the predicted
    `0.25`, with the rate matching `(Ω/2)√(q_M²−σ²)` to 5–11 %. Gough's threshold `A_c=√(2Δ/ε)` moves
    as **`√Δ`** (verified by re-crossing it), and — the sharpest claim in the model — **only the
    *soft* plane whirls**: same string, same amplitude, same seed, **76.3× vs 1.00×**. All of it
    while energy is flat to 1e-12, which is what separates a whirl from a blow-up.
  - **The rotating wave** (`analysis/rotating_wave.py`) — the model's *one* exact oracle, and its
    escape from "measure the residual, don't promise cents". A helix `u=φcosΩt`, `w=φsinΩt`, `v=ψ(x)`
    solves the full nonlinear PDE **exactly**: `r²=φ'²` is time-independent, so the stretch, the
    tension field and the longitudinal forcing all freeze — the string is bent into a fixed shape and
    spun. Solved as a boundary-value problem for `(φ, ψ, Ω)`, it sharpens the polarization
    discriminator from *five* orders to **twenty-three**: seeded from a converged helix the
    longitudinal field does not move at all (`long_kin/E = 1.3e-26`, against `6.6e-3` planar). And it
    turns the Kirchhoff–Carrier frequency error from a bare residual into a **mechanism**: KC assumes
    `φ` is a sine, but a rigid helix is stretched *non-uniformly* — most near the nodes, where `φ'`
    is largest — so the true `φ` is a deformed sine and the frequency error is
    **`(4/3)×` the shape deformation**, a ratio that holds across `EA/T` and mode number. The whole
    discrepancy is a tension field whose spread along `x` is **0.5 % of its own rise**: that is why
    KC is a *good* oracle, and why it is still the wrong one.
- **Bowed string** — the first continuous **nonlinear exciter** (`core/bow.py`): a friction bow on
  a damped string, closing the `exciter →` leg of the abstraction. Stick-slip via the smooth
  friction curve `Φ(v)=F·√(2a)·v·e^{-av²+½}`, evaluated at the *centered* relative velocity — so the
  force is implicit and reduces to one scalar equation (rank-1 driving-point admittance `a=A⁻¹eᵢ`,
  continuation-seeded Newton + bracketed fallback through the multivalued Helmholtz regime). The
  friction force is applied *exactly*, so the discrete **energy balance** `E − E₀ = bow_work` holds
  to machine precision (the bow is *active*, not conservative). Reproduces sustained **Helmholtz
  motion** (one slip per period, slip fraction = β, bow-speed-independent pitch, amplitude ∝ bow
  speed) and the **Schelleng** min/max-force playability wedge.
- **Body / radiation** — the third node of `exciter → resonator → body/radiation`: a **modal body**
  (bank of damped oscillators, `core/body.py`) coupled to a string *terminus* through an
  **energy-conserving bridge** (`core/connection.py`). The linear spring makes the whole system one
  leapfrog, so `E_string + E_body + E_conn` is conserved to machine precision (explicit, exact — no
  implicit solve); an exact coupled-eigenvalue guard bounds the spring stiffness. Radiated pressure
  is read out as `Σ aᵢ q̈ᵢ` (monopole ∝ volume acceleration).
- **Air / radiation** — the last node of the chain (`core/radiation.py`), in three tiers. A
  free-space **monopole read-out** (`AirRadiation`) turns the body's volume acceleration into the
  far-field `ρ₀Q″(t − r/c₀)/(4πr)` with an exact integer-sample retardation. A **radiation load**
  (`RadiatedBody`) lets the air push back — a passive rank-1 dashpot solved by one scalar
  Sherman–Morrison — so what the air takes is *booked*: `E_body + ∫R U² dt` is conserved to ~1e-14
  and `R = 0` is bit-identical to the bare body. And a **frequency-dependent load**
  (`RationalAirLoad`) replaces that one constant `R` with the exact first-order impedance of a
  pulsating sphere: a resistance in *parallel* with the radiation mass, `Z_a = R·jωτ/(1+jωτ)`. No
  filter fitting is involved — that impedance is already rational, so a single auxiliary state
  realises it *exactly*, and the R–`M_a` network **is** the passivity proof (unconditional: no CFL,
  no guard). The volume velocity splits, so the energy identity gains a **stored** term beside the
  dissipated one, `E_body + ½M_a U_L² + ∫R U_R² dt`, flat to ~1e-14; `M_a → ∞` collapses it back to
  the constant-`R` load bit-for-bit. A measured impedance sweep matches the **pre-warped** closed
  form to 8e-16 (trapezoid *is* the bilinear transform, so the scheme realises `Z_a` at
  `s = (2j/k)·tan(ωk/2)` — comparing against `Z_a(jω)` instead shows a warp that reads as a bug).
  The payoff is what a constant `R` structurally cannot do: high partials radiate better and die
  first, at `α = a²·Re Z_a/(2m_eff)`, while the **reactance** adds mass and drops the pitch — both
  measured against the closed form to better than 2%.
- **The 3-D air box** — the **distributed** tier above all three (`core/airbox.py`): a room of
  actual air on a Yee grid, node-centred pressure and face-centred velocity, which is the bore's
  1-D cell tensored up so its boundary machinery transfers unchanged. A lumped port has one
  terminal, one impedance, one distance, and so cannot represent room modes, travel time, comb
  filtering, several listeners, or a source that is not compact. The energy is the same cross-time
  product under tensor-trapezoid weights (drift ~1e-15), and **one** wall closure covers all three
  boundary types — `p^{n+1} = (p_rigid − βp^n)/(1+β)`, a 1×1 solve where edges and corners simply
  *sum admittances* — with `Z = ∞` recovering the rigid box bit-identically and `Z = 0` the open
  face. The soft volume-velocity source is booked as its own channel, so `stored + absorbed −
  injected` is flat while the room drains 99.8% of what it was given. Its money oracle is rare:
  the tensor cosine is an **exact** eigenvector of the discrete Laplacian *including at the h/2
  wall nodes*, so both the mode shape and its frequency `ω_d = (2/k)·arcsin(c₀kμ/2)` are asserted
  to ~1e-14 — a tier above the membrane's Bessel test, which staircasing could only ever make a
  convergence check. Two things 3-D teaches that 1-D does not: **no `λ` is dispersionless**, only a
  *direction* is (on the grid diagonal at `λ = 1/√3`, exact to 1e-16, axial modes never), and at
  that same ceiling the corner mode goes defective so broadband content grows **linearly** while
  the energy stays flat — the one place in this repo where a flat energy is not a stability
  certificate. The headline is cross-tier: far from the walls the room reproduces `AirRadiation`'s
  own monopole law to `max|gain−1| = 6.6e-3`, and one cell thick it tracks `Bore` to 5.3e-14.
- **The room pushes back** *(air box, batch 2)* — `RoomPort` / `RoomLoadedBody`: the two-way
  body↔room coupling, which is to the air box what `RadiatedBody` was to `AirRadiation`. It rests on
  one observation about the scheme: within a single step an injection changes the pressure at its
  own nodes and **nowhere else** (propagation waits for the next momentum sub-step), so the room
  seen from a port is a **Thévenin source**, `p̄ = p̄_free + R_room·q` — a known open-circuit
  pressure carrying the room's entire history *including every reflection*, in series with one
  positive constant. The body's rank-1 volume-velocity solve then closes in a **single division**,
  `U = (U_free − G·p̄_free)/(1 + G·R_room)`, and since `1 + G·R_room ≥ 1` the port is
  **unconditionally passive**: no CFL of its own, no stability guard. The two ledgers are the same
  number twice — each port's `∫p̄U dt` *is* the room's `injected` — so summing them removes the
  coupling term from the conserved statement entirely and `Σ inst.energy() + room.energy()` is flat
  to ~1e-14 for any wall, any port position, any number of instruments. The payoff is what no `R(ω)`
  can produce at any order: a **delayed echo**. In two rooms differing *only* in `Lz` the body's
  trajectory is **bit-identical** until the round trip completes and different immediately after,
  at `2d + 1` steps for `d` = 3, 5, 7 — three geometries, one law; and two instruments in one room
  hear each other, the second staying *exactly* zero until the first's wave arrives. That arrival is
  **Manhattan**, not `r/c₀`: the 7-point stencil spreads one node per step, so an off-axis listener
  gets a machine-precision precursor (measured step 6 where `r/c₀` says 11.5). Two refusals came out
  of measuring rather than arguing. A port on an **open** face is perfectly conservative and
  perfectly *silent* (`injected` and `acoustic` exactly zero, drift 8.6e-15) — the energy report is
  structurally blind to it, so it raises at construction. And **overlapping** ports drift 3.6e-2
  where disjoint ones hold 7.1e-15, which is the same measurement that scopes N instruments in.
  Finally, a measured non-convergence, shipped as one: a **point** port is a sphere of radius
  `≈ h/3.1`, so refining the grid *halves* its equivalent radius and doubles the added mass it hangs
  on the body — refinement makes the artifact worse — while a fixed-radius **ball** barely moves
  (0.493 and 0.496 against 1.045 and 1.038 over two halvings: a factor of twenty in grid
  sensitivity). The energy identity is exact either way; the *magnitude* of a point port's load is a
  grid quantity, which is why `radius` has no default. What a spread port *is* comes out as the
  **uniformly injecting ball**, equivalent shell radius `5a/6` — the classic 6/5 shape factor, not
  a pulsating shell of radius `a` — confirmed to **0.3%**, but only after the room is got out of the
  way: the same port reads 8.6% high in a room ten times its radius, and that excess is the room's
  own reactance rather than the port's. The number is Courant-invariant to five figures, so it is a
  static near-field quantity and not a dispersion artifact.
- **Acoustic bore** *(wind leg, batch 1 of 3)* — the first **acoustic** resonator (`core/bore.py`):
  the 1D air column of a clarinet, a staggered pressure/volume-velocity leapfrog of Webster's horn
  equation. Energy-first — the trapezoidal `h/2` half-cell closes a rigid wall (no ghost stencil,
  the free-beam lesson) and the **cross-time** `Uⁿ⁺¹ᐟ² Uⁿ⁻¹ᐟ²` product conserves energy to machine
  precision (drift ~1e-14). A **closed-open** cylinder rings the **odd** harmonics
  `fₙ=(2n-1)c₀/4L` (the clarinet signature — even harmonics ~5·10⁵× down, confirmed both in the
  measured spectrum *and* the operator's own eigenvalues); **open-open** gives the full series.
  `S(x)` is carried from day one (cone/flare = a different area profile, not a rewrite). A
  frequency-independent `-2σU` drag is the passivity placeholder (real viscothermal / bell losses
  come with batch 2's radiating bell, then batch 3's dynamic mass-spring reed exciter).
- **Web viewer** — interactive offline recompute, and as of Phase D batch 16 **every resonator the
  core defines except the free beam has a panel**. Phase A wired the linear string family, Phase B
  the membrane, Phase C both Kirchhoff plates (#5 supported, #5b free) and the von Kármán nonlinear
  plate (#6 gong + cymbal); Phase D was chosen as *consolidation* over more physics and surfaced the
  rest — the tension-modulated string #9, the bow, the geometrically-exact string #10 and its
  phantom partials, the mallet #7, sympathetic strings and Weinreich two-stage decay, the jawari,
  fret and juari configurations of the barrier model #8, the acoustic bore and its dynamic reed, the
  modal and plate bodies, the radiation load, and the parametric instability.
  Each panel is built around the claim its model can actually support rather than a uniform readout:
  the tension panel leads with the **shift** `ω(A) − ω(A→0)` against its exact Duffing closed form
  (the difference cancels the θ-scheme's linear dispersion error, which an absolute frequency would
  carry); the bow's energy panel is a third verdict type — an energy **balance**, because for a
  driven model both older verdicts are actively wrong rather than merely weaker; the geometric
  string's is the **orbit** that model #9 structurally cannot draw. Nonlinear panels read the energy
  verdict through a solver-convergence gate, and a run that dissipates or radiates keeps the
  conservation verdict when its loss is a *booked* channel. The newest panel puts a mode's
  disintegration and the same run's energy drift on one log axis — a straight line up beside a flat
  line at machine precision, which is what separates redistribution from a diverging solve. See
  `docs/dev/web-viewer-plan.md`.

The deliverable for each model is the resonator *and the rig that measures its deviation from
theory*. See `docs/dev/` for the live plans.

## License

Boyko Non-Commercial License v1.0 (BNCL-1.0) — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
Non-commercial use only; commercial use requires a separate license from the copyright holder.
