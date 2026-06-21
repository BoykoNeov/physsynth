"""Membrane (model #4): the 2D scheme is anisotropic — no lambda removes numerical dispersion.

The 1D headline ("lambda = 1 is exact, zero dispersion") does NOT transfer to 2D. The 5-point
Laplacian's plane-wave symbol is

    Lambda(kx, ky) = (4/h²) [ sin²(kx h/2) + sin²(ky h/2) ],

so for a fixed wavenumber magnitude the discrete phase speed depends on the propagation *direction*
(axial vs diagonal). This characterizes that fact rather than bounding it: the assertion is that the
direction-dependence is real at every admissible lambda, while vanishing (isotropic) only in the
continuum limit kappa -> 0.
"""

import numpy as np
import pytest
from helpers import wave_speed

from physsynth.analysis import modal

C = wave_speed()  # 200 m/s
LAMBDA_MAX = 1.0 / np.sqrt(2.0)


def _phase_speed(kappa: float, direction: str, lam: float, h: float = 1e-3) -> float:
    """Discrete phase speed v_p = 2*pi*f_disc / kappa for a plane wave of magnitude ``kappa``.

    ``direction`` is "axial" (kx=kappa, ky=0) or "diagonal" (kx=ky=kappa/sqrt2). ``k`` (timestep) is
    set from the chosen ``lam`` and ``h`` so the comparison is at a fixed Courant number.
    """
    k = lam * h / C  # lam = c k / h
    if direction == "axial":
        kx, ky = kappa, 0.0
    else:
        kx, ky = kappa / np.sqrt(2.0), kappa / np.sqrt(2.0)
    Lam = (4.0 / h**2) * (np.sin(kx * h / 2.0) ** 2 + np.sin(ky * h / 2.0) ** 2)
    f_disc = modal.discrete_membrane_eigenfrequency(Lam, C, k)
    return float(2.0 * np.pi * f_disc / kappa)


def test_isotropic_in_continuum_limit():
    # At long wavelength (kappa h << 1) both directions recover the continuum speed c.
    h = 1e-3
    kappa = 0.02 / h  # kappa*h = 0.02
    for direction in ("axial", "diagonal"):
        v = _phase_speed(kappa, direction, LAMBDA_MAX, h)
        assert abs(v - C) / C < 1e-3, f"{direction} speed {v:.3f} != c at long wavelength"


@pytest.mark.parametrize("lam", [LAMBDA_MAX, 0.6, 0.4, 0.2])
def test_anisotropic_at_every_lambda(lam):
    # Near Nyquist the axial and diagonal phase speeds differ — for EVERY admissible lambda.
    h = 1e-3
    kappa = (0.6 * np.pi) / h  # a short wave, well inside the band
    v_ax = _phase_speed(kappa, "axial", lam, h)
    v_di = _phase_speed(kappa, "diagonal", lam, h)
    rel_gap = abs(v_ax - v_di) / C
    assert rel_gap > 1e-3, (
        f"expected directional anisotropy at lam={lam}, got axial={v_ax:.3f} diagonal={v_di:.3f}"
    )


def test_both_directions_subluminal():
    # Numerical dispersion slows short waves (phase speed <= c) in both directions, as in 1D.
    h = 1e-3
    kappa = (0.6 * np.pi) / h
    for direction in ("axial", "diagonal"):
        v = _phase_speed(kappa, direction, LAMBDA_MAX, h)
        assert v <= C * (1.0 + 1e-9), f"{direction} phase speed {v:.3f} exceeds c"
