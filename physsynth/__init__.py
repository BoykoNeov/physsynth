"""Physical Synthesis Simulator — energy-based physical-modeling sound synthesis.

The package is layered (see CLAUDE.md / HANDOFF.md):

- ``physsynth.core``     headless DSP: difference operators, resonators, exciters, engine.
                         Pure NumPy/SciPy — no audio I/O, no plotting.

``physsynth.analysis`` was deleted at retirement plan §48: by then a row of shims over
``crates/physsynth-analysis``, which is where the oracles and the partial detector live.
"""

__version__ = "0.1.0"
