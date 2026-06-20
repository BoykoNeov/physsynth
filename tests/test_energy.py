"""Energy conservation (lossless) and passivity (lossy) — the primary correctness tests.

These implement HANDOFF §6.1-§6.2 and acceptance criterion 1 (§10).
"""

import numpy as np
import pytest
from helpers import RHO_DEFAULT, make_string

from physsynth.core.engine import simulate
from physsynth.core.exciter import triangular_pluck

DRIFT_TOL = 1e-10  # acceptance criterion 1


def _run(string, secs=2.0, pickup_frac=0.241):
    u0 = triangular_pluck(string.x, string.L, 0.137 * string.L, amplitude=1e-3)
    string.set_state(u0)
    steps = int(secs * string.fs)
    return simulate(string, num_steps=steps, pickup_index=int(round(pickup_frac * string.N)))


# -- Criterion 1: lossless energy is conserved, and for ALL valid lambda (the identity is
#    algebraic, not a lambda=1 special case). --------------------------------------------------
@pytest.mark.parametrize("lam", [1.0, 0.99, 0.9, 0.7, 0.5])
def test_energy_conserved_across_lambda(lam):
    string = make_string(N=100, lam=lam)
    res = _run(string, secs=2.0)
    assert res.energy_drift < DRIFT_TOL, f"drift {res.energy_drift:.2e} at lambda={lam}"


def test_energy_conserved_free_boundary():
    string = make_string(N=100, lam=0.9, boundary="free")
    res = _run(string, secs=2.0)
    assert res.energy_drift < DRIFT_TOL, f"free-boundary drift {res.energy_drift:.2e}"


def test_energy_strictly_positive_when_lossless():
    string = make_string(N=100, lam=0.9)
    res = _run(string, secs=1.0)
    assert np.all(res.energy > 0.0)


# -- Criterion (passivity): with loss, energy decreases monotonically and at the analytic rate. --
def test_passivity_monotonic_decrease():
    string = make_string(N=100, lam=1.0, sigma=5.0)
    res = _run(string, secs=2.0)
    e0 = res.energy[0]
    steps_up = np.diff(res.energy)
    # No step may increase the energy (allow a hair of round-off relative to E0).
    assert np.all(steps_up <= 1e-12 * e0), f"max positive step {steps_up.max():.2e}"


def test_decay_rate_matches_2sigma():
    # A uniformly damped string loses energy as E(t) ~ E0 * exp(-2 sigma t).
    sigma = 4.0
    secs = 1.0
    string = make_string(N=100, lam=1.0, sigma=sigma)
    res = _run(string, secs=secs)
    measured_ratio = res.energy[-1] / res.energy[0]
    expected_ratio = np.exp(-2.0 * sigma * secs)
    # Compare in log space; the discrete rate matches the continuous one to a few percent.
    rel = abs(np.log(measured_ratio) - np.log(expected_ratio)) / abs(np.log(expected_ratio))
    assert rel < 0.02, (
        f"decay rate off by {rel:.3%} (got {measured_ratio:.3e}, want {expected_ratio:.3e})"
    )


def test_energy_units_scale_with_density():
    # energy() returns Joules and scales linearly with rho. Double both rho and T to keep the wave
    # speed c (hence the grid and the displacement field) identical, isolating the rho prefactor.
    s1 = make_string(N=80, lam=0.9, rho=RHO_DEFAULT, T=200.0)
    s2 = make_string(N=80, lam=0.9, rho=2.0 * RHO_DEFAULT, T=400.0)
    for s in (s1, s2):
        s.set_state(triangular_pluck(s.x, s.L, 0.3 * s.L, amplitude=1e-3))
    assert np.isclose(s2.energy() / s1.energy(), 2.0, rtol=1e-12)
