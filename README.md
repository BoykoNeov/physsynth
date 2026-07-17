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
- **Web viewer** — interactive offline recompute. Wired so far: the linear string family (Phase A),
  the membrane (Phase B), both Kirchhoff plates (#5 supported, #5b free) and the von Kármán
  nonlinear plate (#6 supported gong + free cymbal) as 2D heatmap models (Phase C), and the
  tension-modulated string #9 (Phase D batch 1). The nonlinear panels read the energy verdict through
  a solver-convergence gate and the spectrum as an amplitude *hardening shift* — not a cents error.
  The tension panel leads with the **shift** `ω(A) − ω(A→0)` against its exact Duffing closed form
  (the difference cancels the θ-scheme's linear dispersion error, which an absolute frequency would
  carry), measured on a *lossless* pair of runs while the audio stays lossy so the downward pitch
  glide is audible. Phase D is *consolidation* — the exciters (bow, reed), the wind bore, the
  collision models and the coupled/radiating chains are built and tested but not yet surfaced; see
  `docs/dev/web-viewer-plan.md`.

The deliverable for each model is the resonator *and the rig that measures its deviation from
theory*. See `docs/dev/` for the live plans.

## License

Boyko Non-Commercial License v1.0 (BNCL-1.0) — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
Non-commercial use only; commercial use requires a separate license from the copyright holder.
