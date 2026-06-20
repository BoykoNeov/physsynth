"""Physical Synthesis Simulator — energy-based physical-modeling sound synthesis.

The package is layered (see CLAUDE.md / HANDOFF.md):

- ``physsynth.core``     headless DSP: difference operators, resonators, exciters, engine.
                         Pure NumPy/SciPy — no audio I/O, no plotting.
- ``physsynth.analysis`` analytic oracles (modal frequencies) and spectral partial detection.
- ``physsynth.viz``      diagnostic plots (matplotlib). Depends on core, never the reverse.
"""

__version__ = "0.1.0"
