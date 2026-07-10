"""Standalone collision oracle: a lumped mass vs a fixed rigid wall (model #7, first contact model).

The closed form lives here (not in the coupled mallet-membrane run, where the head carries energy
away and there is no analytic contact time). A mass hitting a **fixed linear spring** (``alpha=1``,
no hysteresis) is exactly a half-period of ``omega = sqrt(K/M)``: contact lasts ``pi*sqrt(M/K)`` and
the mass leaves with its entry speed exactly reversed (restitution 1). This unit-tests
the energy-conserving discrete-gradient scheme in isolation — the analog of testing the von Karman
bracket before the time loop. The discrete-gradient ``0/0`` removable singularity is exercised
directly, and the hysteretic felt is checked to be strictly dissipative (restitution < 1).
"""

import numpy as np
import pytest
from helpers import make_mallet_wall

from physsynth.core.mallet import (
    contact_force_dg,
    contact_force_elastic,
    contact_potential,
    contact_stiffness,
)

CONSERVE_TOL = 1e-11  # lossless: relative drift of (KE + PE); observed ~2e-13


def _run(wall, steps):
    """Step ``wall`` for ``steps``, returning per-step (energy, penetration, velocity) arrays."""
    e = np.empty(steps + 1)
    eta = np.empty(steps + 1)
    v = np.empty(steps + 1)
    e[0], eta[0], v[0] = wall.energy(), wall.penetration, wall.velocity()
    for i in range(1, steps + 1):
        wall.step()
        e[i], eta[i], v[i] = wall.energy(), wall.penetration, wall.velocity()
    return e, eta, v


# -- Criterion 1 (money test): the closed-form contact time and exact velocity reversal ----------
def test_contact_time_and_velocity_reversal_linear_felt():
    # alpha = 1 (linear felt), lossless: contact duration = pi*sqrt(M/K), restitution exactly 1.
    M, K, v0, fs = 0.02, 5.0e4, 2.0, 96000.0
    wall = make_mallet_wall(mass=M, K=K, alpha=1.0, hysteresis=0.0, fs=fs, strike_velocity=v0)
    e, eta, v = _run(wall, 700)
    k = wall.k

    # Interpolated zero-crossings of the penetration (entry: eta -> +, exit: eta -> -).
    up = np.where((eta[:-1] <= 0.0) & (eta[1:] > 0.0))[0][0]
    dn = np.where((eta[:-1] > 0.0) & (eta[1:] <= 0.0))[0][0]
    t_in = (up + eta[up] / (eta[up] - eta[up + 1])) * k
    t_out = (dn + eta[dn] / (eta[dn] - eta[dn + 1])) * k
    contact_time = t_out - t_in
    theory = np.pi * np.sqrt(M / K)
    assert abs(contact_time - theory) / theory < 5e-3, (
        f"contact time {contact_time*1e6:.2f}us vs theory {theory*1e6:.2f}us"
    )

    # Velocity reversal: the mass leaves with +v0 (energy is fully returned by the elastic felt).
    v_exit = v[-1]
    assert v_exit > 0.0, "mass did not rebound"
    assert abs(v_exit - v0) / v0 < 1e-9, f"restitution not 1: v_exit={v_exit:.6f} vs v0={v0}"


# -- Criterion 2: energy conservation of the standalone rig (any alpha) --------------------------
@pytest.mark.parametrize("alpha", [1.0, 2.0, 2.3, 3.0])
def test_standalone_energy_conserved(alpha):
    wall = make_mallet_wall(alpha=alpha, hysteresis=0.0, strike_velocity=2.5)
    e, eta, _ = _run(wall, 800)
    assert np.all(np.isfinite(e)), "non-finite energy"
    assert eta.max() > 0.0, "the mass never made contact"
    drift = np.max(np.abs(e - e[0])) / abs(e[0])
    assert drift < CONSERVE_TOL, f"energy drift {drift:.2e} (alpha={alpha})"


# -- Criterion 3: the discrete-gradient 0/0 removable singularity is handled ----------------------
def test_discrete_gradient_removable_singularity():
    K, alpha, tol = 3.0e4, 2.3, 1e-12
    # Exactly equal arguments: the DG is 0/0 -> the Taylor branch returns phi'(eta), finite.
    for eta in (0.5e-3, 1e-3, 2e-3):
        f = contact_force_dg(eta, eta, K, alpha, tol)
        assert np.isfinite(f)
        assert f == pytest.approx(float(contact_force_elastic(eta, alpha=alpha, K=K)), rel=1e-12)
    # Just inside the tolerance: still the Taylor branch, still finite (would NaN without it).
    f_near = contact_force_dg(1e-3 + 1e-14, 1e-3, K, alpha, tol)
    assert np.isfinite(f_near)
    # Well outside the tolerance the DG -> phi'(midpoint) as the gap shrinks (consistency).
    a, b = 1e-3 + 1e-7, 1e-3
    f_dg = contact_force_dg(a, b, K, alpha, tol)
    f_mid = float(contact_force_elastic(0.5 * (a + b), alpha=alpha, K=K))
    assert f_dg == pytest.approx(f_mid, rel=1e-4)


# -- Criterion 4: hysteretic felt is strictly dissipative (passivity, restitution < 1) -----------
def test_hysteresis_is_passive():
    v0 = 2.0
    wall = make_mallet_wall(alpha=1.5, hysteresis=5.0e3, fs=96000.0, strike_velocity=v0)
    e, eta, v = _run(wall, 900)
    assert np.all(np.isfinite(e))
    # Energy is monotone non-increasing (each step removes >= 0, within a tiny tolerance).
    assert np.max(np.diff(e)) <= 1e-9 * e[0], "hysteresis added energy (non-passive)"
    # The mass rebounds slower than it arrived: coefficient of restitution < 1.
    v_exit = v[-1]
    assert 0.0 < v_exit < v0, f"restitution not in (0,1): v_exit={v_exit:.4f} vs v0={v0}"


# -- primitives ---------------------------------------------------------------------------------
def test_contact_primitives_are_one_sided():
    K, alpha = 1.0e4, 2.0
    # No force / potential / stiffness for a non-penetrating (eta <= 0) gap.
    for eta in (-1.0, -1e-9, 0.0):
        assert float(contact_potential(eta, K, alpha)) == 0.0
        assert float(contact_force_elastic(eta, K, alpha)) == 0.0
        assert float(contact_stiffness(eta, K, alpha)) == 0.0
    # Positive and monotone in contact; force = d(potential)/d(eta), stiffness = d(force)/d(eta).
    eta = 1e-3
    d = 1e-9
    dphi = float(contact_potential(eta + d, K, alpha)) - float(contact_potential(eta - d, K, alpha))
    assert dphi / (2 * d) == pytest.approx(float(contact_force_elastic(eta, K, alpha)), rel=1e-5)
    df = (float(contact_force_elastic(eta + d, K, alpha))
          - float(contact_force_elastic(eta - d, K, alpha)))
    assert df / (2 * d) == pytest.approx(float(contact_stiffness(eta, K, alpha)), rel=1e-5)


def test_stiffness_alpha_one_edge_case():
    # alpha = 1: phi'' = K for eta > 0, but must stay 0 for eta <= 0 (the 0**0 trap).
    K = 1.0e4
    assert float(contact_stiffness(1e-3, K, 1.0)) == pytest.approx(K, rel=1e-12)
    assert float(contact_stiffness(-1e-3, K, 1.0)) == 0.0
    assert float(contact_stiffness(0.0, K, 1.0)) == 0.0
