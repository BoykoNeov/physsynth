"""Acoustic bore (wind leg): stability and construction guards (criterion 4).

- No NaN / blow-up across the valid Courant range and every boundary combination.
- ``lambda = c0 k / h > 1`` is rejected at construction (explicit CFL), and exactly ``1`` accepted.
- Non-physical parameters and unknown boundary tokens are rejected.

The headless / dependency-allowlist guards in test_stability.py sweep every physsynth.core
submodule, so bore.py is covered there automatically (it uses only numpy + scipy.sparse).
"""

import numpy as np
import pytest
from helpers import BORE_LENGTH_DEFAULT, make_bore

from physsynth.core.bore import C0_AIR, Bore
from physsynth.core.engine import simulate


def _run(bore, secs=0.2):
    x = bore.x
    p0 = 1e-3 * np.exp(-((x - 0.3 * bore.L) ** 2) / (2.0 * (0.08 * bore.L) ** 2))
    bore.set_state(p0)
    return simulate(bore, num_steps=int(secs * bore.fs), pickup_index=1)


@pytest.mark.parametrize("lam", [0.999, 0.95, 0.9, 0.75, 0.5, 0.3, 0.1])
def test_no_nan_across_valid_lambda(lam):
    bore = make_bore(N=150, lam=lam, boundary=("closed", "open"))
    res = _run(bore)
    assert np.all(np.isfinite(res.output))
    assert np.all(np.isfinite(res.energy))


@pytest.mark.parametrize(
    "boundary",
    [("closed", "open"), ("open", "open"), ("closed", "closed"), ("open", "closed")],
)
def test_no_nan_across_boundaries(boundary):
    bore = make_bore(N=150, lam=0.9, boundary=boundary)
    res = _run(bore)
    assert np.all(np.isfinite(res.output))
    assert np.all(np.isfinite(res.energy))


def test_lambda_above_one_rejected_at_construction():
    L, N = BORE_LENGTH_DEFAULT, 150
    # lambda = c0*N / (fs*L): a LOWER fs (coarser timestep) raises lambda, so divide by 1.05.
    fs_unstable = C0_AIR * N / (L * 1.05)  # forces lambda = 1.05 > 1
    with pytest.raises(ValueError, match="CFL"):
        Bore(L=L, fs=fs_unstable, N=N)


def test_lambda_exactly_one_is_accepted():
    bore = make_bore(N=150, lam=1.0)
    assert bore.lam == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"L": 0.0},
        {"L": -1.0},
        {"fs": 0.0},
        {"radius": 0.0},
        {"radius": -0.01},
        {"rho0": 0.0},
        {"c0": -1.0},
        {"sigma": -0.1},
        {"N": 1},
        {"boundary": "rigid"},
        {"boundary": ("closed", "leaky")},
    ],
)
def test_invalid_parameters_rejected(kwargs):
    base = {"L": 0.5, "fs": 400000.0, "N": 150}
    base.update(kwargs)
    with pytest.raises(ValueError):
        Bore(**base)
