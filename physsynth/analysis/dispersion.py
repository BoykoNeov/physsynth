"""Numerical-dispersion analysis for the explicit string scheme (HANDOFF §6 test 5, §7).

The explicit scheme propagates spatial mode ``m`` at the discrete frequency

    f_m = arcsin( lambda * sin(m*pi/(2N)) ) / (pi*k),     k = lambda*L/(c*N)

(see :func:`physsynth.analysis.modal.discrete_mode_frequency`). At ``lambda = 1`` this equals the
continuum ``m*c/(2L)`` for *every* mode — zero numerical dispersion, the d'Alembert-exact case. For
``lambda < 1`` it lies below the continuum value and the gap widens with ``m``: high partials travel
too slowly.

HANDOFF §6.5 asks for this as a *phase velocity vs frequency* curve. A standing mode ``m`` is two
counter-propagating waves of wavenumber ``k_m = m*pi/L`` and angular frequency ``omega = 2*pi*f_m``,
so its phase velocity is

    v_p(m) = omega / k_m = 2*L*f_m / m,

which equals ``c`` for the continuum and droops below ``c`` once ``lambda < 1``.
:func:`phase_velocity` returns it for measured *or* oracle frequencies; dividing by ``c`` gives the
dimensionless curve that is flat at ``1`` for ``lambda = 1`` and sags for ``lambda < 1``.

Pure NumPy; independent of the core (an oracle must not depend on the code it validates).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .modal import discrete_mode_frequency, discrete_stiff_mode_frequency

__all__ = ["dispersion_frequencies", "stiff_dispersion_frequencies", "phase_velocity"]


def dispersion_frequencies(
    c: float, L: float, N: int, lam: float, modes: NDArray[np.int_]
) -> NDArray[np.float64]:
    """Discrete-scheme frequencies (Hz) for each mode in ``modes`` — the vectorised oracle.

    A vectorised companion to :func:`physsynth.analysis.modal.discrete_mode_frequency`; the closed
    form lives there so there is a single source of truth for the dispersion relation.
    """
    return np.array(
        [discrete_mode_frequency(c, L, N, lam, int(m)) for m in np.atleast_1d(modes)]
    )


def stiff_dispersion_frequencies(
    c: float, L: float, N: int, kappa: float, k: float, theta: float, modes: NDArray[np.int_]
) -> NDArray[np.float64]:
    """Discrete stiff-string frequencies (Hz) for each mode -- the vectorised stiff oracle.

    Wraps :func:`physsynth.analysis.modal.discrete_stiff_mode_frequency` (the single source of
    truth). Unlike the ideal case the curve droops *more* steeply with mode number: bending adds a
    ``kappa^2 p^4`` term to ``Q`` that stiffens high partials, so the phase velocity rises with mode
    (the partials sharpen) while numerical dispersion drags it the other way -- the net is the
    measured dispersion curve.
    """
    return np.array(
        [
            discrete_stiff_mode_frequency(c, L, N, kappa, k, int(m), theta)
            for m in np.atleast_1d(modes)
        ]
    )


def phase_velocity(
    f: NDArray[np.float64], L: float, modes: NDArray[np.int_]
) -> NDArray[np.float64]:
    """Modal phase velocity ``v_p = 2*L*f/m`` (m/s) for frequencies ``f`` of modes ``modes``.

    Accepts measured or analytic frequencies. For the continuum ``f = m*c/(2L)`` it returns ``c``
    for every mode; numerical dispersion (``lambda < 1``) makes it droop below ``c`` as ``m`` grows.
    """
    f = np.asarray(f, dtype=float)
    m = np.asarray(modes, dtype=float)
    return 2.0 * L * f / m
