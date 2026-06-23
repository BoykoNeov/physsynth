"""Free-free beam (model #5b-pre): stability and construction guards.

- No NaN / blow-up across a sweep of the beam-Courant number ``mu = kappa k / h²``,
  **including values
  far past the explicit bound 1/4** — the implicit theta-scheme (theta >= 1/4) is unconditionally
  stable, so there is no CFL ceiling to reject.
- Non-physical parameters and an unsupported boundary are rejected at construction.

The headless / dependency-allowlist guards in test_stability.py sweep every
physsynth.core submodule,
so beam.py is covered there automatically (it uses only numpy + scipy.sparse(.linalg)).
"""

import numpy as np
import pytest
from helpers import make_beam

from physsynth.core.beam import FreeBeam
from physsynth.core.engine import simulate
from physsynth.core.exciter import raised_cosine


def _run(beam, secs=0.3):
    beam.set_state(raised_cosine(beam.x, beam.L, 0.4 * beam.L, 0.25 * beam.L, amplitude=1e-3))
    return simulate(beam, num_steps=int(secs * beam.fs), pickup_index=0)


@pytest.mark.parametrize("mu", [0.1, 0.5, 2.0, 8.0, 32.0])
def test_no_nan_across_mu(mu):
    beam = make_beam(N=40, mu=mu)
    res = _run(beam)
    assert np.all(np.isfinite(res.output))
    assert np.all(np.isfinite(res.energy))


def test_explicit_unstable_config_runs_stably():
    # mu = 50 is ~200x the explicit beam bound (1/4); the implicit scheme runs it without blow-up.
    beam = make_beam(N=40, mu=50.0)
    assert beam.mu > 0.25
    res = _run(beam, secs=0.5)
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
        {"L": 0.0},
        {"sigma": -0.1},
        {"N": 3},  # < 4: too few interior nodes for the free-free modes
        {"theta": 0.0},
        {"theta": 1.5},
        {"boundary": "supported"},
    ],
)
def test_invalid_parameters_rejected(kwargs):
    base = {"L": 1.0, "rho": 0.005, "fs": 50000.0, "N": 40, "kappa": 20.0}
    base.update(kwargs)
    with pytest.raises(ValueError):
        FreeBeam(**base)
