"""Membrane (model #4): energy conservation (lossless) and passivity (lossy).

HANDOFF §6.1-§6.2 in 2D. The headline result is the **decoupling**: the staircased *circular* rim
conserves the discrete energy to machine precision exactly like the rectangle, because the masked
5-point Laplacian stays symmetric. Staircasing taxes the Bessel match (test_membrane_modal), not
conservation.
"""

import numpy as np
import pytest
from helpers import make_membrane

from physsynth.core.engine import simulate
from physsynth.core.exciter import raised_cosine_2d

DRIFT_TOL = 1e-10  # the 1D acceptance bar, carried into 2D unchanged


def _pluck(m, amplitude=1e-3):
    """A smooth off-centre radial bump, clamped to the live region."""
    if m.domain == "rectangle":
        center = (0.4 * m.Lx, 0.55 * m.Ly)
        width = 0.25 * min(m.Lx, m.Ly)
    else:
        center = (0.2 * m.radius, -0.15 * m.radius)
        width = 0.55 * m.radius
    field = raised_cosine_2d(m.X, m.Y, center, width, amplitude=amplitude)
    field[~m.mask] = 0.0
    return field


def _run(m, secs=1.0):
    m.set_state(_pluck(m))
    return simulate(m, num_steps=int(secs * m.fs))


# -- Conservation: lossless energy is flat to machine precision, BOTH geometries, across lambda. --
@pytest.mark.parametrize("domain", ["rectangle", "circle"])
@pytest.mark.parametrize("lam", [0.7071, 0.6, 0.4])
def test_energy_conserved(domain, lam):
    m = make_membrane(domain=domain, N=48, lam=lam)
    res = _run(m, secs=1.0)
    assert res.energy_drift < DRIFT_TOL, f"{domain} drift {res.energy_drift:.2e} at lam={lam}"


def test_circle_conserves_like_rectangle():
    """The decoupling claim, made a single direct comparison: staircasing does not leak energy."""
    rect = make_membrane(domain="rectangle", N=48, lam=0.6)
    circ = make_membrane(domain="circle", N=48, lam=0.6)
    drifts = [_run(m, secs=1.0).energy_drift for m in (rect, circ)]
    assert max(drifts) < DRIFT_TOL, f"drifts rect/circle = {drifts}"


def test_energy_strictly_positive_when_lossless():
    m = make_membrane(domain="circle", N=48, lam=0.6)
    res = _run(m, secs=0.5)
    assert np.all(res.energy > 0.0)


# -- Passivity: with loss, energy decreases monotonically and at ~2 sigma. --
def test_passivity_monotonic_decrease():
    m = make_membrane(domain="circle", N=48, lam=0.6, sigma=8.0)
    res = _run(m, secs=1.0)
    e0 = res.energy[0]
    steps_up = np.diff(res.energy)
    assert np.all(steps_up <= 1e-12 * e0), f"max positive step {steps_up.max():.2e}"


def test_decay_rate_matches_2sigma():
    # A uniformly damped membrane loses energy as E(t) ~ E0 exp(-2 sigma t).
    sigma, secs = 6.0, 0.6
    m = make_membrane(domain="rectangle", N=48, lam=0.6, sigma=sigma)
    res = _run(m, secs=secs)
    measured = res.energy[-1] / res.energy[0]
    expected = np.exp(-2.0 * sigma * secs)
    rel = abs(np.log(measured) - np.log(expected)) / abs(np.log(expected))
    assert rel < 0.02, f"decay rate off by {rel:.3%} (got {measured:.3e}, want {expected:.3e})"


def test_energy_units_scale_with_density():
    # energy() is in Joules and scales linearly with areal density rho; double rho and T together to
    # hold c (hence the grid and field) fixed, isolating the rho prefactor.
    m1 = make_membrane(domain="circle", N=40, lam=0.6, rho=0.005, T=200.0)
    m2 = make_membrane(domain="circle", N=40, lam=0.6, rho=0.010, T=400.0)
    for m in (m1, m2):
        m.set_state(_pluck(m))
    assert np.isclose(m2.energy() / m1.energy(), 2.0, rtol=1e-12)
