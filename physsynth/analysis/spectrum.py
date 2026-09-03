"""Spectral measurement — the partial detector every modal test in this project reads.

**The implementation is Rust.** ``crates/physsynth-analysis/src/spectrum.rs``, re-exported through
the wrappers below (``docs/dev/rust-migration-plan.md`` §44, unit 11). The windowed magnitude
spectrum, the parabolic refinement and its local-maximum guard, and the peak picker are documented
there.
"""

from __future__ import annotations

import numpy as np
import physsynth_rs as _rs
from numpy.typing import NDArray

__all__ = ["magnitude_spectrum", "measure_partials_near", "detect_peaks"]


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


def _asarray(a: object) -> NDArray[np.float64]:
    """Whatever the caller passed, as the contiguous float64 array the binding requires."""
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64))


def magnitude_spectrum(signal, fs, zero_pad_factor=2):
    return _rs.spectrum_magnitude_spectrum(_asarray(signal), fs, zero_pad_factor)


def _parabolic_refine(mag, i, fs, nfft):
    return _rs.spectrum_parabolic_refine(_asarray(mag), int(i), fs, nfft)


def measure_partials_near(signal, fs, expected, search_hz=None):
    return _rs.spectrum_measure_partials_near(_asarray(signal), fs, _asarray(expected), search_hz)


def detect_peaks(signal, fs, n_peaks, f_min=1.0, min_separation_hz=None):
    return _rs.spectrum_detect_peaks(_asarray(signal), fs, int(n_peaks), f_min, min_separation_hz)
