"""Plate (model #5): stability and construction guards (criterion 4, flexural case).

- No NaN / blow-up across a sweep of the plate-Courant number ``mu = kappa k / h²``, **including
  values far past the explicit bound 1/4** — the implicit theta-scheme (theta >= 1/4) is
  unconditionally stable, so there is no CFL ceiling to reject (unlike the membrane's 1/sqrt(2)).
- Non-physical parameters and an unsupported boundary are rejected at construction.

The headless / dependency-allowlist guards in test_stability.py sweep every physsynth.core
submodule, so plate.py and operators2d.py are covered there automatically (scipy.sparse(.linalg)
stays within the allowlist).
"""

import numpy as np
import pytest
from helpers import make_plate

from physsynth.core.engine import simulate
from physsynth.core.exciter import raised_cosine_2d
from physsynth.core.plate import Plate


def _run(p, secs=0.3):
    field = raised_cosine_2d(p.X, p.Y, (0.4 * p.Lx, 0.5 * p.Ly), 0.3 * p.Lx, amplitude=1e-3)
    field[~p.mask] = 0.0
    p.set_state(field)
    return simulate(p, num_steps=int(secs * p.fs), pickup_index=0)


@pytest.mark.parametrize("mu", [0.1, 0.5, 2.0, 8.0, 32.0])
def test_no_nan_across_mu(mu):
    p = make_plate(N=40, mu=mu)
    res = _run(p)
    assert np.all(np.isfinite(res.output))
    assert np.all(np.isfinite(res.energy))


def test_explicit_unstable_config_runs_stably():
    # mu = 50 is ~200x the explicit plate bound (1/4); the implicit scheme runs it without blow-up.
    p = make_plate(N=40, mu=50.0)
    assert p.mu > 0.25
    res = _run(p, secs=0.5)
    assert np.all(np.isfinite(res.energy))
    assert res.energy_drift < 1e-9  # still conserves, just at a coarse timestep


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kappa": 0.0},
        {"kappa": -1.0},
        {"rho": 0.0},
        {"rho": -1.0},
        {"fs": 0.0},
        {"Lx": 0.0},
        {"Ly": -1.0},
        {"sigma": -0.1},
        {"N": 1},
        {"theta": 0.0},
        {"theta": 1.5},
        {"boundary": "free"},
    ],
)
def test_invalid_parameters_rejected(kwargs):
    base = {
        "Lx": 1.0, "Ly": 1.0, "kappa": 20.0, "rho": 0.005, "fs": 50000.0, "N": 40,
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        Plate(**base)
