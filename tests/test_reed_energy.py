"""Energy validation for the single-reed mouthpiece (nonlinear self-oscillating exciter, wind #3).

The reed is an *active* element — a steady mouth pressure feeds energy in — so ``E`` is **not**
conserved. And unlike the bow, the reed *stores* energy (it is a mass-spring), so the conserved book
is ``E = E_bore + E_reed``. The correctness statement is the discrete work-energy **balance**:

    E^n - E^0 == mouth_work - jet_loss - reed_damp_work        (lossless bore)

with every dissipation channel sign-definite — ``jet_loss = sum k dp_bar U_B >= 0`` (Bernoulli),
``reed_damp_work = sum k Mr g y'^2 >= 0`` (reed damping) — and ``mouth_work = sum k p_m U`` the
active breath input. The reactive reed-force / reed-sweep coupling telescopes exactly *because* the
bore node and the reed both use the same centered ``p_bar`` and ``y'`` — an off-centered pressure
would leak O(k). Unlike the bow, whose memoryless force makes its balance residual-*independent*,
the reed's per-step error is ``k p_bar R / p_pref0`` — **linear in the scalar residual** ``R`` — so
this test both requires *and verifies* a converged solve each step; it reads machine precision
because ``newton_tol ~ 1e-10`` keeps ``R`` tiny. This is the exciter's money test.
"""

import numpy as np
import pytest
from helpers import make_reed_bore

BALANCE_TOL = 1e-11  # lossless: |dE - (mouth - jet - reed_damp)| / scale (observed ~4e-15)


def _run(reed, steps):
    """Step ``reed``, returning per-step total energy and the three cumulative work channels."""
    e = np.empty(steps + 1)
    mouth = np.empty(steps + 1)
    jet = np.empty(steps + 1)
    damp = np.empty(steps + 1)
    e[0], mouth[0], jet[0], damp[0] = reed.energy(), 0.0, 0.0, 0.0
    for i in range(1, steps + 1):
        reed.step()
        e[i] = reed.energy()
        mouth[i] = reed.mouth_work
        jet[i] = reed.jet_loss
        damp[i] = reed.reed_damp_work
    return e, mouth, jet, damp


# -- Criterion 1 (money test): lossless energy balance to machine precision. -------------------
@pytest.mark.parametrize("p_mouth", [1000.0, 1500.0, 2500.0])
def test_lossless_energy_balance(p_mouth):
    # Lossless bore (closed-open, sigma=0): every joule of stored energy is accounted for by the
    # breath minus the (sign-definite) jet and reed-damping losses. Machine precision regardless of
    # whether the reed is beating (high p_mouth) or not.
    reed = make_reed_bore(boundary=("closed", "open"), sigma=0.0, p_mouth=p_mouth)
    e, mouth, jet, damp = _run(reed, 8000)
    lhs = e - e[0]
    rhs = mouth - jet - damp
    scale = np.abs(e) + np.abs(mouth) + 1e-30
    rel = np.max(np.abs(lhs - rhs) / scale)
    assert rel < BALANCE_TOL, f"energy-balance error {rel:.2e} at p_mouth={p_mouth}"


def test_reed_injects_energy():
    # Sanity that the breath actually drives the air column up from rest.
    reed = make_reed_bore(boundary=("closed", "open"), sigma=0.0, p_mouth=1500.0)
    _run(reed, 4000)
    assert reed.mouth_work > 0.0, "the mouth did no net work"
    assert reed.energy() > 1e-8, "the air column never gained energy from the reed"


# -- Criterion 2: every dissipation channel is passive (sign-definite). ------------------------
def test_dissipation_channels_are_nonnegative():
    reed = make_reed_bore(boundary=("closed", "open"), sigma=0.0, p_mouth=1500.0)
    _run(reed, 6000)
    assert reed.jet_loss >= 0.0, "Bernoulli jet returned energy (non-passive)"
    assert reed.reed_damp_work >= 0.0, "reed damping returned energy (non-passive)"


def test_jet_and_reed_damp_are_monotone():
    # Each accumulator only ever grows: dp_bar U_B >= 0 and Mr g y'^2 >= 0 every step.
    reed = make_reed_bore(boundary=("closed", "open"), sigma=0.0, p_mouth=1800.0)
    jet = np.empty(4001)
    damp = np.empty(4001)
    jet[0] = damp[0] = 0.0
    for i in range(1, 4001):
        reed.step()
        jet[i], damp[i] = reed.jet_loss, reed.reed_damp_work
    assert np.min(np.diff(jet)) >= -1e-18, "a jet-loss step returned energy"
    assert np.min(np.diff(damp)) >= -1e-18, "a reed-damping step returned energy"


# -- Criterion 3: the identity survives a radiating bell (extra passive channel). --------------
def test_balance_with_radiating_bell():
    # Add the bell's shed energy to the right-hand side: E_reed + E_bore(field) change equals
    # breath minus jet minus reed-damp minus what the bell radiated. energy() already folds the
    # bore's radiated channel into E_bore, so the same identity holds verbatim.
    reed = make_reed_bore(boundary=("closed", "radiating"), R_bell=5e4, p_mouth=1500.0)
    e, mouth, jet, damp = _run(reed, 6000)
    lhs = e - e[0]
    rhs = mouth - jet - damp
    scale = np.abs(e) + np.abs(mouth) + 1e-30
    rel = np.max(np.abs(lhs - rhs) / scale)
    assert rel < BALANCE_TOL, f"balance error with radiating bell {rel:.2e}"
    assert reed.bore.radiated_energy > 0.0, "the bell should shed some energy"


# -- Delegation / protocol. --------------------------------------------------------------------
def test_energy_is_bore_plus_reed():
    reed = make_reed_bore(boundary=("closed", "open"), sigma=0.0)
    for _ in range(200):
        reed.step()
    assert np.isclose(reed.energy(), reed.bore.energy() + reed.reed_energy(), rtol=0, atol=1e-18)


def test_reed_energy_zero_at_rest():
    reed = make_reed_bore(boundary=("closed", "open"), sigma=0.0)
    assert reed.reed_energy() == 0.0
    assert reed.energy() == 0.0
