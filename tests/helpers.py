"""Shared test fixtures and physical parameters for the validation harness.

A single canonical string (L=1 m, T=200 N, rho=0.005 kg/m -> c=200 m/s, f1=100 Hz) is reused so the
expected frequencies are easy to reason about.
"""

from __future__ import annotations

import numpy as np

from physsynth.core.string_ideal import Boundary, IdealString

L_DEFAULT = 1.0
T_DEFAULT = 200.0
RHO_DEFAULT = 0.005  # -> c = sqrt(T/rho) = 200 m/s, fundamental f1 = c/(2L) = 100 Hz


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


def convergence_orders(errors: np.ndarray, step_sizes: np.ndarray) -> np.ndarray:
    """Empirical orders ``p`` between consecutive (h, error) pairs: ``error ~ C h^p``."""
    errors = np.asarray(errors, dtype=float)
    step_sizes = np.asarray(step_sizes, dtype=float)
    return np.log(errors[:-1] / errors[1:]) / np.log(step_sizes[:-1] / step_sizes[1:])
