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

Models #1–4 complete and validated (string family: ideal / stiff / damped; 2D circular membrane) plus
the interactive web viewer for the string family (Phase A). The deliverable for each model is the
resonator *and the rig that measures its deviation from theory*. See `docs/dev/` for the live plans.

## License

MIT.
