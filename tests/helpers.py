"""Shared test fixtures and physical parameters for the validation harness.

A single canonical string (L=1 m, T=200 N, rho=0.005 kg/m -> c=200 m/s, f1=100 Hz) is reused so the
expected frequencies are easy to reason about.
"""

from __future__ import annotations

import numpy as np

from physsynth.analysis import modal, spectrum
from physsynth.core.engine import simulate
from physsynth.core.string_ideal import Boundary, IdealString
from physsynth.core.string_stiff import THETA_DEFAULT, StiffString

L_DEFAULT = 1.0
T_DEFAULT = 200.0
RHO_DEFAULT = 0.005  # -> c = sqrt(T/rho) = 200 m/s, fundamental f1 = c/(2L) = 100 Hz
KAPPA_DEFAULT = 2.0  # -> B = pi^2 kappa^2 / (c^2 L^2) ~ 9.87e-4, a piano-ish inharmonicity


def wave_speed(T: float = T_DEFAULT, rho: float = RHO_DEFAULT) -> float:
    return float(np.sqrt(T / rho))


def make_string(
    *,
    N: int = 100,
    lam: float = 1.0,
    boundary: Boundary = "fixed",
    sigma: float = 0.0,
    L: float = L_DEFAULT,
    T: float = T_DEFAULT,
    rho: float = RHO_DEFAULT,
) -> IdealString:
    """Build a string whose Courant number is exactly ``lam`` by choosing ``fs = c N / (L lam)``."""
    c = wave_speed(T, rho)
    fs = c * N / (L * lam)
    return IdealString(L=L, T=T, rho=rho, fs=fs, N=N, boundary=boundary, sigma=sigma)


def make_stiff_string(
    *,
    N: int = 100,
    lam: float = 1.0,
    kappa: float = KAPPA_DEFAULT,
    sigma: float = 0.0,
    theta: float = THETA_DEFAULT,
    L: float = L_DEFAULT,
    T: float = T_DEFAULT,
    rho: float = RHO_DEFAULT,
) -> StiffString:
    """Build a stiff string at Courant number ``lam`` via ``fs = c N / (L lam)``.

    Unlike :func:`make_string`, ``lam > 1`` is allowed (the implicit scheme is unconditionally
    stable) -- a coarse-grid / large-timestep regime the explicit stiff scheme could not run.
    """
    c = wave_speed(T, rho)
    fs = c * N / (L * lam)
    return StiffString(
        L=L, T=T, rho=rho, fs=fs, N=N, kappa=kappa, sigma=sigma, theta=theta
    )


def convergence_orders(errors: np.ndarray, step_sizes: np.ndarray) -> np.ndarray:
    """Empirical orders ``p`` between consecutive (h, error) pairs: ``error ~ C h^p``."""
    errors = np.asarray(errors, dtype=float)
    step_sizes = np.asarray(step_sizes, dtype=float)
    return np.log(errors[:-1] / errors[1:]) / np.log(step_sizes[:-1] / step_sizes[1:])


def measure_mode_frequencies(
    modes: np.ndarray,
    *,
    N: int,
    lam: float,
    secs: float = 0.5,
    amplitude: float = 1e-3,
) -> np.ndarray:
    """Measure the FDTD oscillation frequency (Hz) of each single spatial mode in ``modes``.

    Each run is initialised with one exact discrete eigenvector ``phi_m = sin(m pi x / L)``. Because
    that eigenvector is the *only* thing present, the field stays ``u(t) = q(t) phi_m``, so the
    modal coordinate ``q(t) = <u(t), phi_m>`` is a pure tone at the mode's discrete frequency. We
    measure that projection rather than a single grid pickup: a point pickup can land on a node of
    the mode (e.g. mode N/2 vanishes at every even grid node), whereas the projection has maximal
    SNR for every mode and any N. The peak search is anchored at the closed-form dispersion oracle.
    Returns one frequency per mode (the dispersion curve, measured).
    """
    c, L = wave_speed(), L_DEFAULT
    out = []
    for m in np.atleast_1d(modes):
        m = int(m)
        s = make_string(N=N, lam=lam)
        phi = modal.mode_shape(s.x, L, m)
        s.set_state(phi * amplitude)
        res = simulate(s, num_steps=int(secs * s.fs), snapshot_stride=1)
        states = np.array([state for _, state in res.snapshots])  # (T+1, N+1)
        q = states @ phi  # modal coordinate time series (proportional to a pure cosine)
        f_oracle = modal.discrete_mode_frequency(c, L, N, lam, m)
        out.append(spectrum.measure_partials_near(q, res.fs, np.array([f_oracle]))[0])
    return np.array(out)


def measure_stiff_mode_frequencies(
    modes: np.ndarray,
    *,
    N: int,
    lam: float,
    kappa: float = KAPPA_DEFAULT,
    theta: float = THETA_DEFAULT,
    secs: float = 0.5,
    amplitude: float = 1e-3,
) -> np.ndarray:
    """Measure the stiff-string FDTD frequency (Hz) of each single spatial mode in ``modes``.

    Identical to :func:`measure_mode_frequencies` but for :class:`StiffString`: ``sin(m pi x / L)``
    is still an *exact* discrete eigenvector (the biharmonic block is ``(delta_xx)^2``), so a
    single-mode initial condition stays a pure tone and its modal-coordinate projection is a clean
    cosine at the stiff discrete frequency. The peak search is anchored at the closed-form stiff
    dispersion oracle (which depends on ``theta``).
    """
    c, L = wave_speed(), L_DEFAULT
    out = []
    for m in np.atleast_1d(modes):
        m = int(m)
        s = make_stiff_string(N=N, lam=lam, kappa=kappa, theta=theta)
        phi = modal.mode_shape(s.x, L, m)
        s.set_state(phi * amplitude)
        res = simulate(s, num_steps=int(secs * s.fs), snapshot_stride=1)
        states = np.array([state for _, state in res.snapshots])
        q = states @ phi
        f_oracle = modal.discrete_stiff_mode_frequency(c, L, N, kappa, s.k, m, theta)
        out.append(spectrum.measure_partials_near(q, res.fs, np.array([f_oracle]))[0])
    return np.array(out)
