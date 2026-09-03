"""Frequency-dependent damping oracles — decay rates the damped schemes are checked against.

**The implementation is Rust.** ``crates/physsynth-analysis/src/damping.rs``, re-exported through
the wrappers below (``docs/dev/rust-migration-plan.md`` §44, unit 10). The continuum rate, the
discrete per-mode decay of the theta-scheme, the underdamped predicate and the T60 inversion are
documented there.
"""

from __future__ import annotations

import numpy as np
import physsynth_rs as _rs

__all__ = [
    "spatial_eigenvalue_p2",
    "modal_loss_rate_continuum",
    "discrete_damped_mode_decay",
    "discrete_damped_mode_rate",
    "discrete_damped_mode_is_underdamped",
    "loss_coefficients_from_T60",
    "T60_SECONDS_PER_RATE",
]

T60_SECONDS_PER_RATE = 3.0 * np.log(10.0)  # ~= 6.9078; T60 = this / sigma_eff


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


def spatial_eigenvalue_p2(N, h, m):
    return _rs.damping_spatial_eigenvalue_p2(int(N), h, int(m))


def modal_loss_rate_continuum(c, L, kappa, sigma0, sigma1, m):
    return _rs.damping_modal_loss_rate_continuum(c, L, kappa, sigma0, sigma1, int(m))


def discrete_damped_mode_decay(c, L, N, kappa, k, theta, sigma0, sigma1, m):
    return _rs.damping_discrete_damped_mode_decay(
        c, L, int(N), kappa, k, theta, sigma0, sigma1, int(m)
    )


def discrete_damped_mode_rate(c, L, N, kappa, k, theta, sigma0, sigma1, m):
    return _rs.damping_discrete_damped_mode_rate(
        c, L, int(N), kappa, k, theta, sigma0, sigma1, int(m)
    )


def discrete_damped_mode_is_underdamped(c, L, N, kappa, k, theta, sigma0, sigma1, m):
    return _rs.damping_discrete_damped_mode_is_underdamped(
        c, L, int(N), kappa, k, theta, sigma0, sigma1, int(m)
    )


def loss_coefficients_from_T60(c, L, kappa, f1, T60_1, f2, T60_2):
    return _rs.damping_loss_coefficients_from_t60(c, L, kappa, f1, T60_1, f2, T60_2)
