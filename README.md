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

## Status

**Milestone 1 (current):** ideal-string FDTD solver + validation harness (HANDOFF §10). The deliverable
is the string *and the rig that measures its deviation from theory*. See `docs/dev/` for the live plan.

## License

MIT.
