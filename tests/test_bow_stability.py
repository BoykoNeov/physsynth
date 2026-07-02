"""Stability, robustness, and construction validation for the bowed string.

The bow is *stable, not conservative*: friction is bounded (``|Phi|_max = force``) and the bow input
per step is bounded, so the scheme never blows up — and the per-step friction root always exists
(``r(v) = v - v_free + g Phi(v)`` runs from ``-inf`` to ``+inf``). These tests sweep the playable
parameter space for finiteness, exercise the multivalued Helmholtz regime that trips pure Newton
(exercising the bracketed fallback), check the ``helmholtz_number`` diagnostic marks the
single-valued / multivalued boundary, and reject non-physical construction.
"""

import numpy as np
import pytest
from helpers import make_bowed_string

from physsynth.core.bow import BowedString, friction_smooth, friction_smooth_deriv
from physsynth.core.string_damped import DampedStiffString


# -- no blow-up across the playable parameter space -----------------------------------------
@pytest.mark.parametrize("force", [0.2, 1.0, 3.0, 8.0])
@pytest.mark.parametrize("v_bow", [0.02, 0.1, 0.4])
def test_no_blowup_force_speed_sweep(force, v_bow):
    bow = make_bowed_string(force=force, v_bow=v_bow)
    for _ in range(3000):
        bow.step()
    assert np.all(np.isfinite(bow.state)), f"non-finite state (force={force}, v_bow={v_bow})"
    assert np.isfinite(bow.energy())


@pytest.mark.parametrize("sharpness", [20.0, 60.0, 150.0, 400.0])
@pytest.mark.parametrize("beta_pos", [0.08, 0.2, 0.35, 0.5])
def test_no_blowup_sharpness_position_sweep(sharpness, beta_pos):
    bow = make_bowed_string(sharpness=sharpness, bow_position=beta_pos)
    for _ in range(3000):
        bow.step()
    assert np.all(np.isfinite(bow.state))


def test_friction_solve_always_converges_in_helmholtz_regime():
    # A strong bow is deep in the multivalued regime; the solve must still return a root every step
    # (via the bracketed fallback) without raising.
    bow = make_bowed_string(force=4.0, sharpness=120.0)
    assert bow.helmholtz_number > 1.0
    for _ in range(4000):
        bow.step()  # would raise RuntimeError if the root search ever failed
    assert bow.fallbacks > 0, "expected some slip-event fallbacks in the multivalued regime"


# -- the helmholtz_number diagnostic marks the single-valued / multivalued boundary ----------
def test_helmholtz_number_below_one_is_single_valued():
    # A weak bow (helmholtz_number < 1) has a single friction root everywhere -> Newton never needs
    # the bracketed fallback.
    bow = make_bowed_string(force=0.02, sharpness=60.0)
    assert bow.helmholtz_number < 1.0
    for _ in range(3000):
        bow.step()
    assert bow.fallbacks == 0, "single-valued regime should need no root fallback"


def test_helmholtz_number_formula():
    bow = make_bowed_string(force=1.0, sharpness=60.0)
    expected = bow._g * bow.force * np.sqrt(2.0 * bow.sharpness) * np.exp(0.5)
    assert np.isclose(bow.helmholtz_number, expected)


# -- friction characteristic shape -----------------------------------------------------------
def test_friction_curve_shape():
    force, a = 1.5, 80.0
    assert friction_smooth(0.0, force, a) == 0.0            # odd, zero at origin
    v_peak = 1.0 / np.sqrt(2.0 * a)
    assert np.isclose(friction_smooth(v_peak, force, a), force)   # peak magnitude == force
    assert np.isclose(friction_smooth(-0.03, force, a), -friction_smooth(0.03, force, a))  # odd
    # derivative matches a finite difference
    v = 0.05
    fd = (friction_smooth(v + 1e-7, force, a) - friction_smooth(v - 1e-7, force, a)) / 2e-7
    assert np.isclose(friction_smooth_deriv(v, force, a), fd, rtol=1e-5)


# -- construction validation -----------------------------------------------------------------
def _string(N=100):
    return DampedStiffString(L=1.0, T=200.0, rho=0.005, fs=40000.0, N=N, kappa=0.0, sigma0=0.5)


def test_rejects_negative_force():
    with pytest.raises(ValueError, match="force"):
        BowedString(string=_string(), bow_position=0.13, v_bow=0.1, force=-1.0)


def test_rejects_nonpositive_sharpness():
    with pytest.raises(ValueError, match="sharpness"):
        BowedString(string=_string(), bow_position=0.13, v_bow=0.1, force=1.0, sharpness=0.0)


@pytest.mark.parametrize("pos", [-0.1, 0.0, 1.0, 1.5])
def test_rejects_bow_position_out_of_range(pos):
    with pytest.raises(ValueError, match="bow_position"):
        BowedString(string=_string(), bow_position=pos, v_bow=0.1, force=1.0)


def test_bow_node_is_interior_and_snapped():
    s = _string(N=100)
    # A bow requested very close to the nut still lands on an interior node (1..N-1).
    bow = BowedString(string=s, bow_position=0.004, v_bow=0.1, force=1.0)
    assert 1 <= bow.node <= s.N - 1
    assert 0.0 < bow.x_bow < s.L
