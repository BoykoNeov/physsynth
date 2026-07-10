"""Stability, validation, and the inert-hook regression for the single-reed mouthpiece (wind #3).

- The ``Bore.step`` ``source`` hook is **inert** when unused: ``source=None`` and a no-op source are
  bit-for-bit the un-driven bore (the batch-1/2 regression — nothing changed under the reed's seam).
- The reed stays **bounded** across a wide blowing range (it is energy-passive: the jet and reed
  damping cap the stored energy), even blown hard into the beating regime.
- Parameter guards: the reed rides a ``"closed"`` left end, needs an oversampled reed
  (``wr k < 2``), and rejects non-physical values.
"""

import numpy as np
import pytest
from helpers import make_reed_bore

from physsynth.core.bore import Bore
from physsynth.core.reed import ReedBore, bernoulli_flow


def _make_bore(boundary=("closed", "open"), **kw):
    return Bore(L=0.5, fs=1e6, N=200, radius=0.008, boundary=boundary, **kw)


# -- The source hook is inert when unused (batch-1/2 bit-identical regression). ----------------
def test_source_none_is_bit_identical():
    a, b = _make_bore(), _make_bore()
    for bore in (a, b):
        bore.set_state(1e-3 * np.exp(-((bore.x - 0.15) ** 2) / (2.0 * 0.04**2)))
    for _ in range(3000):
        a.step()
        b.step(source=None)
    assert np.array_equal(a.p, b.p) and np.array_equal(a.U, b.U)


def test_noop_source_is_bit_identical():
    a, b = _make_bore(), _make_bore()
    for bore in (a, b):
        bore.set_state(1e-3 * np.exp(-((bore.x - 0.15) ** 2) / (2.0 * 0.04**2)))
    for _ in range(3000):
        a.step()
        b.step(source=lambda p: None)
    assert np.array_equal(a.p, b.p)


# -- Bounded / stable across the blowing range (passivity caps the amplitude). -----------------
@pytest.mark.parametrize("p_mouth", [500.0, 1500.0, 2500.0, 4000.0])
def test_bounded_across_blowing(p_mouth):
    reed = make_reed_bore(p_mouth=p_mouth)
    mp = np.empty(int(0.4 / reed.k))
    for i in range(len(mp)):
        reed.step()
        mp[i] = reed.mouthpiece_pressure()
    assert np.all(np.isfinite(mp)), f"blew up at p_mouth={p_mouth}"
    # The stored energy can never exceed the breath work put in (passive): amplitude stays sane.
    assert np.max(np.abs(mp)) < 20.0 * p_mouth, f"amplitude ran away at p_mouth={p_mouth}"


def test_reed_is_a_resonator():
    # Duck-types the engine's Resonator protocol.
    reed = make_reed_bore()
    assert isinstance(reed, ReedBore)
    assert hasattr(reed, "k") and callable(reed.step) and callable(reed.energy)
    assert reed.state.shape == (reed.bore.N + 1,)
    assert isinstance(reed.displacement_at(10), float)


# -- Bernoulli flow helper (the nonlinearity in isolation). ------------------------------------
def test_bernoulli_flow_is_odd_and_passive():
    # Signed by dp, zero when shut, and dp * U_B >= 0 (a passive resistor).
    assert bernoulli_flow(0.0, 4e-4, 1.5e-2, 1.2) == 0.0
    assert bernoulli_flow(1000.0, 0.0, 1.5e-2, 1.2) == 0.0        # shut reed passes no air
    up = bernoulli_flow(1000.0, 4e-4, 1.5e-2, 1.2)
    dn = bernoulli_flow(-1000.0, 4e-4, 1.5e-2, 1.2)
    assert up > 0.0 and dn == pytest.approx(-up)                  # odd in dp
    assert 1000.0 * up >= 0.0                                     # dp * U_B >= 0


def test_flow_scales_as_sqrt_dp():
    # Quadrupling the pressure drop doubles the flow (Bernoulli sqrt law).
    u1 = bernoulli_flow(500.0, 4e-4, 1.5e-2, 1.2)
    u4 = bernoulli_flow(2000.0, 4e-4, 1.5e-2, 1.2)
    assert u4 == pytest.approx(2.0 * u1, rel=1e-12)


# -- Parameter guards. -------------------------------------------------------------------------
def test_non_closed_left_end_rejected():
    with pytest.raises(ValueError, match="closed"):
        ReedBore(bore=_make_bore(boundary=("open", "closed")), p_mouth=1500.0)


def test_reed_cfl_rejected():
    # An absurdly stiff reed on a coarse-time bore: wr k >= 2.
    bore = Bore(L=0.5, fs=2e4, N=20, radius=0.008, boundary=("closed", "open"))
    with pytest.raises(ValueError, match="reed CFL"):
        ReedBore(bore=bore, p_mouth=1500.0, f_reed=8000.0)


@pytest.mark.parametrize("bad", [{"mu": 0.0}, {"H0": -1e-4}, {"Sr": 0.0}, {"q_reed": -1.0}])
def test_nonphysical_params_rejected(bad):
    with pytest.raises(ValueError):
        ReedBore(bore=_make_bore(), p_mouth=1500.0, **bad)


def test_p_closing_and_gamma():
    # p_closing = mu wr^2 H0 and gamma = p_mouth / p_closing (the control parameter).
    reed = make_reed_bore(p_mouth=1000.0)
    expected = reed.mu * reed.wr**2 * reed.H0
    assert reed.p_closing == pytest.approx(expected, rel=1e-12)
    assert reed.gamma == pytest.approx(1000.0 / expected, rel=1e-12)
