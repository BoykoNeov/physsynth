"""The exact Duffing oracle — the nonlinear string's closed form, in Jacobi elliptic functions.

**The implementation is Rust.** ``crates/physsynth-analysis/src/duffing.rs``, with the elliptic
functions themselves in ``crates/physsynth-analysis/src/elliptic.rs``; re-exported through the
wrappers below (``docs/dev/rust-migration-plan.md`` §44, unit 10). The Kirchhoff-Carrier mode
reduction, the elliptic parameter, the exact frequency and its small-amplitude expansion are
documented there.
"""

from __future__ import annotations

import numpy as np
import physsynth_rs as _rs

__all__ = [
    "kc_mode_coefficients",
    "kc_mode_stretch",
    "duffing_frequency",
    "duffing_frequency_shift",
    "duffing_frequency_expansion",
    "duffing_elliptic_parameter",
    "duffing_displacement",
]


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


def kc_mode_coefficients(*, c, kappa, EA, rho, p2, L):
    return _rs.duffing_kc_mode_coefficients(c, kappa, EA, rho, p2, L)


def kc_mode_stretch(amplitude, *, p2, L):
    return _rs.duffing_kc_mode_stretch(amplitude, p2, L)


def duffing_elliptic_parameter(amplitude, omega0_sq, eps):
    return _rs.duffing_elliptic_parameter(amplitude, omega0_sq, eps)


def duffing_frequency(amplitude, omega0_sq, eps):
    return _rs.duffing_frequency(amplitude, omega0_sq, eps)


def duffing_frequency_shift(amplitude, omega0_sq, eps):
    return _rs.duffing_frequency_shift(amplitude, omega0_sq, eps)


def duffing_displacement(t, amplitude, omega0_sq, eps):
    arr = np.asarray(t, dtype=float)
    flat = np.ascontiguousarray(arr.ravel())
    out = _rs.duffing_displacement(flat, amplitude, omega0_sq, eps)
    return out.reshape(arr.shape) if arr.ndim else out[0]


def duffing_frequency_expansion(amplitude, omega0_sq, eps):
    return _rs.duffing_frequency_expansion(amplitude, omega0_sq, eps)
