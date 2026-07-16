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
  (nonlinear; below).
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
- **Web viewer** — interactive offline recompute for every model: the string family (Phase A), the
  membrane (Phase B), and both Kirchhoff plates (#5 supported, #5b free) and the von Kármán
  nonlinear plate (#6 supported gong + free cymbal) as 2D heatmap models. The nonlinear panel reads
  the energy verdict through a Picard-convergence gate and the spectrum as an amplitude *hardening
  shift* (not a cents error).

The deliverable for each model is the resonator *and the rig that measures its deviation from
theory*. See `docs/dev/` for the live plans.

## License

Boyko Non-Commercial License v1.0 (BNCL-1.0) — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
Non-commercial use only; commercial use requires a separate license from the copyright holder.
