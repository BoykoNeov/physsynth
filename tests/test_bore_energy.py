"""Acoustic bore (wind leg): energy conservation (lossless) and passivity (lossy).

The primary correctness tests (HANDOFF §6.1-§6.2), now for the staggered p/U acoustic scheme:

- the cross-time acoustic energy is flat to machine precision for a lossless run, across the whole
  valid Courant range AND every boundary combination (closed/open at each end);
- with the viscous ``-2 sigma U`` loss, energy is monotone non-increasing and a single mode decays
  at the analytic ``2 sigma`` rate.

The velocity term is the **cross-time product** ``U^{n+1/2} U^{n-1/2}`` — collapsing it to a
same-time square is the classic bug this test is built to catch.
"""

import numpy as np
import pytest
from helpers import make_bore
from scipy.sparse.linalg import eigsh

from physsynth.core.engine import simulate

DRIFT_TOL = 1e-10  # acceptance criterion 1 (same bar as every other resonator)


def _bump(bore, center_frac=0.3, width_frac=0.08, amplitude=1e-3):
    """A smooth interior Gaussian pressure pulse (open ends get zeroed by ``set_state``)."""
    c = center_frac * bore.L
    w = width_frac * bore.L
    return amplitude * np.exp(-((bore.x - c) ** 2) / (2.0 * w * w))


def _run(bore, secs=0.3):
    bore.set_state(_bump(bore))
    return simulate(bore, num_steps=int(secs * bore.fs), pickup_index=1)


# -- Conservation: lossless energy is flat to machine precision, across lambda (algebraic identity,
#    not a lambda=1 special case). --------------------------------------------------------------
@pytest.mark.parametrize("lam", [1.0, 0.99, 0.9, 0.7, 0.5])
def test_energy_conserved_across_lambda(lam):
    bore = make_bore(N=200, lam=lam, boundary=("closed", "open"))
    res = _run(bore, secs=0.3)
    assert res.energy_drift < DRIFT_TOL, f"drift {res.energy_drift:.2e} at lambda={lam}"


# -- Conservation holds for EVERY boundary combination: the closed wall (half-cell) and the open end
#    (pressure-release pin) are both lossless reflectors. -----------------------------------------
@pytest.mark.parametrize(
    "boundary",
    [("closed", "open"), ("open", "open"), ("closed", "closed"), ("open", "closed")],
)
def test_energy_conserved_all_boundaries(boundary):
    bore = make_bore(N=200, lam=0.9, boundary=boundary)
    res = _run(bore, secs=0.3)
    assert res.energy_drift < DRIFT_TOL, f"{boundary} drift {res.energy_drift:.2e}"


def test_energy_strictly_positive_when_lossless():
    bore = make_bore(N=200, lam=0.9, boundary=("closed", "open"))
    res = _run(bore, secs=0.2)
    assert np.all(res.energy > 0.0)


# -- Passivity: with viscous loss, energy decreases monotonically. -------------------------------
def test_passivity_monotonic_decrease():
    bore = make_bore(N=200, lam=0.9, boundary=("closed", "open"), sigma=40.0)
    res = _run(bore, secs=0.3)
    e0 = res.energy[0]
    steps_up = np.diff(res.energy)
    assert np.all(steps_up <= 1e-10 * e0), f"max positive step {steps_up.max() / e0:.2e} * E0"


def test_decay_rate_matches_2sigma_single_mode():
    # A single mode loses energy as E(t) ~ E0 exp(-2 sigma t): the viscous term damps the kinetic
    # (velocity) half, and over a cycle the loss averages to 2 sigma of the total (kinetic ==
    # potential on average). The acoustic analogue of the string's uniform-damping decay test.
    sigma, secs = 40.0, 0.15
    bore = make_bore(N=150, lam=0.8, boundary=("closed", "open"), sigma=sigma)
    dof = bore.dof
    _, vec = eigsh(bore.Lop[dof][:, dof], k=1, M=bore.Cmat[dof][:, dof], sigma=0.0, which="LM")
    phi = np.zeros(bore.N + 1)
    phi[dof] = vec[:, 0]
    bore.set_state(phi * 1e-3)
    res = simulate(bore, num_steps=int(secs * bore.fs))
    measured = res.energy[-1] / res.energy[0]
    expected = np.exp(-2.0 * sigma * secs)
    rel = abs(np.log(measured) - np.log(expected)) / abs(np.log(expected))
    assert rel < 0.02, f"decay rate off by {rel:.3%} (got {measured:.3e}, want {expected:.3e})"


def test_energy_units_scale_with_area():
    # energy() is in Joules. For a fixed pressure field it scales linearly with the cross-section
    # S = pi r^2: at a fixed pressure IC the initial energy is pure compliance ~ S, so doubling S
    # (radius * sqrt(2)) doubles E for the same field.
    b1 = make_bore(N=120, lam=0.9, radius=0.008)
    b2 = make_bore(N=120, lam=0.9, radius=0.008 * np.sqrt(2.0))
    for b in (b1, b2):
        b.set_state(_bump(b))
    assert np.isclose(b2.energy() / b1.energy(), 2.0, rtol=1e-12)
