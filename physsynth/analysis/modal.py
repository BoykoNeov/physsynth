"""Analytic modal oracles — closed-form frequencies the numerical schemes are checked against.

**The implementation is Rust.** This module is what is left of the Python one after the
migration's seventh deletion (``docs/dev/rust-migration-plan.md`` §44, unit 10): every oracle here
lives in ``crates/physsynth-analysis/src/modal.rs``, is bound in ``crates/physsynth-py``, and is
re-exported through the thin wrappers below. The derivations — the ideal and stiff string series,
the rectangular and circular membrane, the supported and free plate (isotropic and orthotropic),
the free-free beam, the free circular plate's root search, and the bore — are documented at length
in that Rust module and in ``docs/dev/``'s plans.
"""

from __future__ import annotations

import numpy as np
import physsynth_rs as _rs
from numpy.typing import NDArray

__all__ = [
    "harmonic_frequencies",
    "mode_shape",
    "discrete_mode_frequency",
    "inharmonicity_B",
    "stiff_harmonic_frequencies",
    "discrete_stiff_mode_frequency",
    "cents",
    # 2D membrane (model #4)
    "rectangular_membrane_freqs",
    "rectangular_mode_field",
    "rectangular_discrete_eigenvalues",
    "circular_membrane_freqs",
    "discrete_membrane_eigenfrequency",
    # 2D plate (model #5)
    "rectangular_plate_freqs",
    "discrete_plate_eigenfrequency",
    # 2D orthotropic plate -- model #5 with a grain direction
    "orthotropic_plate_freqs",
    "discrete_orthotropic_plate_eigenfrequency",
    "dirichlet_axis_eigenvalue",
    # 1D free-free beam (model #5b-pre)
    "free_free_beam_betaL",
    "free_free_beam_freqs",
    "discrete_beam_eigenfrequency",
    # 2D free-edge (FFFF) plate (model #5b)
    "free_plate_ffff_square_lambdas",
    "free_plate_freq_from_lambda",
    # 2D free-edge ORTHOTROPIC plate -- model #5b with a grain (model #5of)
    "free_plate_twist_bound",
    "free_plate_coupling_form",
    # 2D free-edge CIRCULAR plate -- the derived oracle for a staircased outline (model #5g)
    "free_circular_plate_lambda_roots",
    "free_circular_plate_lambdas",
    "free_circular_plate_saddle_bound",
    # 1D acoustic bore (wind leg)
    "bore_resonance_frequencies",
    "discrete_bore_eigenfrequency",
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


def _flat(a) -> NDArray[np.float64]:
    """Whatever the caller passed, as the contiguous flat float64 array the binding wants."""
    return np.ascontiguousarray(np.asarray(a, dtype=float).ravel())


def _elemwise(fn, a, *args):
    """Call a flat-in/flat-out binding and restore NumPy's own shape convention.

    A 0-d input must come back as a scalar, not a one-element array -- see the note above.
    """
    arr = np.asarray(a, dtype=float)
    out = fn(_flat(arr), *args)
    return out.reshape(arr.shape) if arr.ndim else out[0]


def _modes(modes):
    """A mode list as the ``[(m, n), ...]`` of Python ints the binding extracts."""
    return [(int(m), int(n)) for m, n in np.asarray(modes).reshape(-1, 2)]


def harmonic_frequencies(c, L, n_partials):
    return _rs.modal_harmonic_frequencies(c, L, int(n_partials))


def mode_shape(x, L, m):
    arr = np.asarray(x, dtype=float)
    out = _rs.modal_mode_shape(_flat(arr), L, int(m))
    return out.reshape(arr.shape) if arr.ndim else out[0]


def discrete_mode_frequency(c, L, N, lam, m):
    return _rs.modal_discrete_mode_frequency(c, L, int(N), lam, int(m))


def inharmonicity_B(c, L, kappa):
    return _rs.modal_inharmonicity_b(c, L, kappa)


def stiff_harmonic_frequencies(c, L, kappa, n_partials):
    return _rs.modal_stiff_harmonic_frequencies(c, L, kappa, int(n_partials))


def discrete_stiff_mode_frequency(c, L, N, kappa, k, m, theta):
    return _rs.modal_discrete_stiff_mode_frequency(c, L, int(N), kappa, k, int(m), theta)


def cents(f, f_ref):
    # Broadcast first, exactly as the ufunc would, then hand over one flat pair -- so the whole
    # expression 1200*log2(f/f_ref) lives on the Rust side rather than half of it.
    a, b = np.broadcast_arrays(np.asarray(f, dtype=float), np.asarray(f_ref, dtype=float))
    out = _rs.modal_cents(_flat(a), _flat(b))
    return out.reshape(a.shape) if a.ndim else out[0]


def rectangular_membrane_freqs(c, Lx, Ly, modes):
    return _rs.modal_rectangular_membrane_freqs(c, Lx, Ly, _modes(modes))


def rectangular_mode_field(X, Y, Lx, Ly, m, n):
    arr = np.asarray(X, dtype=float)
    out = _rs.modal_rectangular_mode_field(_flat(X), _flat(Y), Lx, Ly, int(m), int(n))
    return out.reshape(arr.shape) if arr.ndim else out[0]


def rectangular_discrete_eigenvalues(h, Nx, Ny, modes):
    return _rs.modal_rectangular_discrete_eigenvalues(h, int(Nx), int(Ny), _modes(modes))


def circular_membrane_freqs(c, a, n_modes, m_max=12, n_max=12):
    return _rs.modal_circular_membrane_freqs(c, a, int(n_modes), int(m_max), int(n_max))


def discrete_membrane_eigenfrequency(Lambda, c, k):
    return _elemwise(_rs.modal_discrete_membrane_eigenfrequency, Lambda, c, k)


def rectangular_plate_freqs(kappa, Lx, Ly, modes):
    return _rs.modal_rectangular_plate_freqs(kappa, Lx, Ly, _modes(modes))


def discrete_plate_eigenfrequency(Lambda_lap, kappa, k, theta):
    return _elemwise(_rs.modal_discrete_plate_eigenfrequency, Lambda_lap, kappa, k, theta)


def orthotropic_plate_freqs(kappa, Lx, Ly, modes, grain_x=1.0, grain_cross=1.0, grain_y=1.0):
    return _rs.modal_orthotropic_plate_freqs(
        kappa, Lx, Ly, _modes(modes), grain_x, grain_cross, grain_y
    )


def discrete_orthotropic_plate_eigenfrequency(
    lam_x, lam_y, kappa, k, theta, grain_x=1.0, grain_cross=1.0, grain_y=1.0
):
    x, y = np.broadcast_arrays(np.asarray(lam_x, dtype=float), np.asarray(lam_y, dtype=float))
    out = _rs.modal_discrete_orthotropic_plate_eigenfrequency(
        _flat(x), _flat(y), kappa, k, theta, grain_x, grain_cross, grain_y
    )
    return out.reshape(x.shape) if x.ndim else out[0]


def dirichlet_axis_eigenvalue(m, L, h):
    return _elemwise(_rs.modal_dirichlet_axis_eigenvalue, m, L, h)


def free_free_beam_betaL(n_modes):
    return _rs.modal_free_free_beam_beta_l(int(n_modes))


def free_free_beam_freqs(kappa, L, n_modes):
    return _rs.modal_free_free_beam_freqs(kappa, L, int(n_modes))


def free_plate_ffff_square_lambdas():
    return _rs.modal_free_plate_ffff_square_lambdas()


def free_plate_freq_from_lambda(lam, kappa, a):
    return _elemwise(_rs.modal_free_plate_freq_from_lambda, lam, kappa, a)


def free_plate_twist_bound(kappa, a, b, grain_torsion=0.5):
    return _rs.modal_free_plate_twist_bound(kappa, a, b, grain_torsion)


def free_circular_plate_lambda_roots(nu, n, lam_max=14.0, scan=20000):
    return _rs.modal_free_circular_plate_lambda_roots(nu, int(n), float(lam_max), int(scan))


def free_circular_plate_lambdas(nu=0.3, n_modes=7, n_max=8):
    return _rs.modal_free_circular_plate_lambdas(nu, int(n_modes), int(n_max))


def free_circular_plate_saddle_bound(nu):
    return _rs.modal_free_circular_plate_saddle_bound(nu)


def free_plate_coupling_form(grain_coupling, h, Nx, Ny):
    return _rs.modal_free_plate_coupling_form(grain_coupling, h, int(Nx), int(Ny))


def bore_resonance_frequencies(c0, L, n_partials, boundary="closed-open"):
    return _rs.modal_bore_resonance_frequencies(c0, L, int(n_partials), boundary)


def discrete_bore_eigenfrequency(omega2, k):
    return _elemwise(_rs.modal_discrete_bore_eigenfrequency, omega2, k)


def discrete_beam_eigenfrequency(mu, kappa, k, theta):
    return _elemwise(_rs.modal_discrete_beam_eigenfrequency, mu, kappa, k, theta)
