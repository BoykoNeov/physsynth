"""Energy validation for the bowed string (nonlinear friction exciter).

The bow is an *active* element — it feeds energy into the string, so ``E`` is **not** conserved.
The correctness statement is instead the discrete work–energy *balance*: the change in stored
energy equals the accumulated bow work (minus any string loss). Because the friction force is
applied *exactly* and the power is read from the true post-correction velocity, the lossless balance
holds to machine precision regardless of the per-step friction solve — the exciter's money test.
Loss only ever removes energy (passivity survives the coupling), and ``force = 0`` decouples the bow
bit-for-bit.
"""

import numpy as np
import pytest
from helpers import make_bowed_string, make_damped_string

from physsynth.analysis import modal
from physsynth.core.bow import BowedString

BALANCE_TOL = 1e-11  # lossless: |dE - bow_work| / scale (observed ~6e-15)


def _run(bow, steps):
    """Step ``bow`` for ``steps``, returning per-step energy and cumulative bow work arrays."""
    e = np.empty(steps + 1)
    w = np.empty(steps + 1)
    e[0], w[0] = bow.energy(), bow.bow_work
    for i in range(1, steps + 1):
        bow.step()
        e[i], w[i] = bow.energy(), bow.bow_work
    return e, w


# -- Criterion 1 (money test): lossless energy balance to machine precision ------------------
@pytest.mark.parametrize("force,sharpness", [(1.0, 60.0), (2.0, 100.0), (0.5, 40.0)])
def test_lossless_energy_balance(force, sharpness):
    # No string loss: every joule of stored energy must be accounted for by the bow's work.
    bow = make_bowed_string(sigma0=0.0, sigma1=0.0, force=force, sharpness=sharpness)
    e, w = _run(bow, 4000)
    scale = np.abs(e) + np.abs(w) + 1e-30
    rel = np.max(np.abs((e - e[0]) - w) / scale)
    assert rel < BALANCE_TOL, f"energy-balance error {rel:.2e} (force={force}, a={sharpness})"


def test_bow_injects_energy():
    # Sanity that the bow is actually doing work (drives the string up from rest).
    bow = make_bowed_string(sigma0=0.0, sigma1=0.0)
    _run(bow, 2000)
    assert bow.bow_work > 0.0, "bow did no net work"
    assert bow.energy() > 1e-8, "string never gained energy from the bow"


# -- Criterion 2: passivity survives the bow — loss only removes energy -----------------------
@pytest.mark.parametrize("sigma0,sigma1", [(0.5, 0.05), (2.0, 0.0), (0.0, 0.1)])
def test_loss_only_removes_energy(sigma0, sigma1):
    # With loss present, the stored energy can never exceed the bow work put in: the balance
    # E - E0 = bow_work - dissipation with dissipation >= 0. (No per-step loss formula needed.)
    bow = make_bowed_string(sigma0=sigma0, sigma1=sigma1)
    e, w = _run(bow, 6000)
    dissipation = w - (e - e[0])  # accumulated energy removed by loss
    assert np.all(np.isfinite(e)), "non-finite energy"
    # Dissipation is monotone non-decreasing (each step removes >= 0), within a tiny tolerance.
    assert dissipation[-1] >= -BALANCE_TOL * (abs(w[-1]) + 1.0), "loss added energy (non-passive)"
    assert np.min(np.diff(dissipation)) >= -1e-9 * (abs(w[-1]) + 1.0), "a loss step added energy"


# -- Criterion 3: force = 0 decouples the bow bit-for-bit -------------------------------------
def test_zero_force_is_bit_identical_to_bare_string():
    # A plucked string wrapped in a force-free bow must evolve identically to the standalone string.
    N, steps = 100, 500
    bare = make_damped_string(N=N, lam=0.9, kappa=0.0, sigma0=0.5, sigma1=0.05)
    phi = modal.mode_shape(bare.x, bare.L, 3) * 1e-3
    bare.set_state(phi)

    bowed = make_bowed_string(N=N, lam=0.9, force=0.0, sigma0=0.5, sigma1=0.05)
    bowed.string.set_state(phi.copy())

    for _ in range(steps):
        bare.step()
        bowed.step()
    assert bowed.bow_force == 0.0 and bowed.bow_work == 0.0
    np.testing.assert_array_equal(bowed.state, bare.state)


def test_energy_method_delegates_to_string():
    bow = make_bowed_string()
    assert bow.energy() == bow.string.energy()
    for _ in range(50):
        bow.step()
    assert bow.energy() == bow.string.energy()


def test_bowed_string_is_a_resonator():
    # Duck-types the engine's Resonator protocol.
    bow = make_bowed_string()
    assert isinstance(bow, BowedString)
    assert hasattr(bow, "k") and callable(bow.step) and callable(bow.energy)
    assert bow.state.shape == (bow.string.N + 1,)
    assert isinstance(bow.displacement_at(10), float)
