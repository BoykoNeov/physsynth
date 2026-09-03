"""Numerical dispersion oracles — what a scheme's partials do as the mode number rises.

**The implementation is Rust.** ``crates/physsynth-analysis/src/dispersion.rs``, re-exported
through the wrappers below (``docs/dev/rust-migration-plan.md`` §44, unit 10).
"""

from __future__ import annotations

import numpy as np
import physsynth_rs as _rs

__all__ = ["dispersion_frequencies", "stiff_dispersion_frequencies", "phase_velocity"]


# The Python bodies are gone and this module delegates. Three things follow, and they are the same
# three that follow for every deleted module in this project:
#
#   * `physsynth_rs` is a hard requirement to import this module, and therefore to collect most of
#     the suite. There is no `PHYSSYNTH_RS_ANALYSIS` switch any more: the flag chose between two
#     implementations and there is one.
#   * what these wrappers do beyond delegating is real work, not ceremony -- coercing whatever a
#     caller passed into the contiguous float64 arrays and Python ints the binding requires. That
#     is why this is a shim and not a row of re-exports, the way `airbox.py`'s wrapper tier and
#     `operators2d.py` are shims for their own reasons.
#   * what the Python implementation *said* is not lost. `tests/analysis_frozen_values.py` holds
#     62 fixtures' worth of its answers, recorded to the last digit before it was deleted, and
#     `tests/test_analysis_frozen.py` asserts them on every run. That is the human's condition on
#     this deletion (plan §44) and it is the only remaining check of these oracles against a second
#     implementation.


def _ints(modes):
    """`np.atleast_1d(modes)` as the list of Python ints the binding extracts."""
    return [int(m) for m in np.atleast_1d(modes)]


def dispersion_frequencies(c, L, N, lam, modes):
    return _rs.dispersion_frequencies(c, L, int(N), lam, _ints(modes))


def stiff_dispersion_frequencies(c, L, N, kappa, k, theta, modes):
    return _rs.dispersion_stiff_frequencies(c, L, int(N), kappa, k, theta, _ints(modes))


def phase_velocity(f, L, modes):
    arr = np.ascontiguousarray(np.asarray(f, dtype=float).ravel())
    return _rs.dispersion_phase_velocity(arr, L, _ints(modes))
