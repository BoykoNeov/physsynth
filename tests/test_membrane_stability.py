"""Membrane (model #4): stability and construction guards (criterion 4, in 2D).

- No NaN / blow-up across a sweep of valid lambda in (0, 1/sqrt(2)].
- The 2D CFL ceiling is **1/sqrt(2)** (the 5-point Laplacian's spectral radius is 8/h², double the
  1D 4/h²); lambda above it is rejected at construction, not run into an overflow.
- Non-physical / missing-geometry parameters are rejected.

The headless / dependency-allowlist guards in test_stability.py sweep every physsynth.core
submodule, so operators2d.py and membrane.py are covered there automatically (no edits needed).
"""

import numpy as np
import pytest
from helpers import make_membrane

from physsynth.core.engine import simulate
from physsynth.core.exciter import raised_cosine_2d
from physsynth.core.membrane import Membrane

LAMBDA_MAX = 1.0 / np.sqrt(2.0)


@pytest.mark.parametrize("domain", ["rectangle", "circle"])
@pytest.mark.parametrize("lam", [0.7071, 0.65, 0.5, 0.3, 0.1])
def test_no_nan_across_valid_lambda(domain, lam):
    m = make_membrane(domain=domain, N=40, lam=lam)
    field = raised_cosine_2d(m.X, m.Y, (0.0, 0.0), 0.3 * (m.radius or m.Lx), amplitude=1e-3)
    field[~m.mask] = 0.0
    m.set_state(field)
    res = simulate(m, num_steps=int(0.3 * m.fs), pickup_index=0)
    assert np.all(np.isfinite(res.output))
    assert np.all(np.isfinite(res.energy))


def test_lambda_above_cfl_rejected_at_construction():
    # lambda = c / (fs h): a LOWER fs raises lambda. Build one just past 1/sqrt(2).
    c = float(np.sqrt(200.0 / 0.005))  # 200 m/s
    N, a = 40, 0.5
    h = 2.0 * a / N
    fs_unstable = c / (1.05 * LAMBDA_MAX * h)  # forces lambda = 1.05 * (1/sqrt2) > ceiling
    with pytest.raises(ValueError, match="CFL"):
        Membrane(domain="circle", T=200.0, rho=0.005, fs=fs_unstable, N=N, radius=a)


def test_lambda_at_cfl_ceiling_accepted():
    # The guard must not reject the exact CFL-limit lambda = 1/sqrt(2) on round-off.
    m = make_membrane(domain="rectangle", N=40, lam=LAMBDA_MAX)
    assert m.lam == pytest.approx(LAMBDA_MAX, abs=1e-9)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rho": -1.0},
        {"T": 0.0},
        {"sigma": -0.1},
        {"N": 1},
        {"domain": "triangle"},
        {"radius": -0.5},
        {"radius": None},  # circle without geometry
    ],
)
def test_invalid_parameters_rejected(kwargs):
    base = {"domain": "circle", "T": 200.0, "rho": 0.005, "fs": 200000.0, "N": 40, "radius": 0.5}
    base.update(kwargs)
    with pytest.raises(ValueError):
        Membrane(**base)


def test_rectangle_requires_sides():
    with pytest.raises(ValueError, match="rectangle"):
        Membrane(domain="rectangle", T=200.0, rho=0.005, fs=200000.0, N=40)
