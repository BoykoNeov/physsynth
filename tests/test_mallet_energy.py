"""Energy validation for the coupled mallet -> membrane collision (model #7, first contact model).

The mallet stores kinetic *and* (through the felt) potential energy, so — unlike the bow — the
correctness statement is strict **conservation** of the total ``H = E_membrane + KE + PE_felt``:
lossless membrane + elastic felt keeps it flat to machine precision. The conservation rests on the
applied force being the exact discrete gradient of the potential, so a deliberately loose solve
makes the drift grow **in proportion to the solve tolerance** — the self-certification that the
discrete-gradient telescoping is what conserves energy. Loss (membrane ``sigma``) or a lossy felt
(``lambda_h``) only ever removes energy (passivity survives the coupling), and a mallet that never
makes contact leaves the membrane bit-for-bit unchanged.
"""

import numpy as np
import pytest
from helpers import make_mallet, make_membrane

from physsynth.core.mallet import MalletMembrane

CONSERVE_TOL = 1e-10  # lossless, elastic: relative drift of H; observed ~1e-12


def _run(mal, steps):
    """Step ``mal`` for ``steps``, returning the per-step total-energy array."""
    e = np.empty(steps + 1)
    e[0] = mal.energy()
    for i in range(1, steps + 1):
        mal.step()
        e[i] = mal.energy()
    return e


# -- Criterion 1 (money test): lossless coupled energy conservation to machine precision ----------
# Vary alpha too (the wall covers alpha in {1,2,3} at g_s=0; here g_s != 0 so this closes the
# "alpha != 2.3 and coupled" gap — the only untested force x coupling interaction).
@pytest.mark.parametrize("K,mass,v0,alpha", [
    (5.0e4, 0.02, 3.0, 2.3), (2.0e4, 0.05, 3.0, 1.0),
    (3.0e4, 0.03, 4.0, 3.0), (5.0e4, 0.02, 3.0, 1.5),
])
def test_lossless_energy_conserved(K, mass, v0, alpha):
    # No membrane loss, elastic felt: E_membrane + mallet KE + felt PE is conserved exactly.
    mal = make_mallet(K=K, mass=mass, alpha=alpha, strike_velocity=v0, hysteresis=0.0, sigma=0.0)
    e = _run(mal, 6000)
    drift = np.max(np.abs(e - e[0])) / abs(e[0])
    assert drift < CONSERVE_TOL, f"energy drift {drift:.2e} (K={K}, M={mass}, v0={v0})"


def test_strike_actually_couples():
    # Sanity: the head must receive a real share of the strike energy (else conservation is a
    # trivial linear re-test and a bracket bug would hide — advisor's large-amplitude caveat).
    mal = make_mallet()
    mem = mal.membrane
    frac = 0.0
    for _ in range(4000):
        mal.step()
        frac = max(frac, mem.energy() / mal.energy())
    assert frac > 0.3, f"membrane took only {frac:.2f} of the energy — coupling too weak to test"


# -- Criterion 2: conservation is discrete-gradient-limited (drift proportional to solve tol) -----
def test_drift_scales_with_newton_tolerance():
    # Tightening the scalar contact solve tightens the conservation: the applied force approaches
    # the exact discrete gradient. A loose solve drifts proportionally -> telescoping conserves.
    drifts = []
    tols = [1e-14, 1e-10, 1e-6]
    for tol in tols:
        membrane = make_membrane(domain="rectangle", N=40, lam=0.5, sigma=0.0)
        mal = MalletMembrane(
            membrane=membrane, mass=0.02, stiffness=5.0e4, alpha=2.3, hysteresis=0.0,
            strike_x=0.5, strike_y=0.5, strike_velocity=3.0, newton_tol=tol,
        )
        e = _run(mal, 2500)
        drifts.append(np.max(np.abs(e - e[0])) / abs(e[0]))
    assert drifts[0] < CONSERVE_TOL, f"tight-tol drift {drifts[0]:.2e} not machine precision"
    assert drifts[-1] > drifts[0] * 100, (
        f"loosening the solve did not increase drift ({drifts}); conservation is not solve-limited"
    )


# -- Criterion 3: passivity survives the coupling — loss only removes energy ---------------------
@pytest.mark.parametrize("sigma,lam_h", [(2.0, 0.0), (0.0, 5.0e3), (1.0, 2.0e3)])
def test_loss_only_removes_energy(sigma, lam_h):
    mal = make_mallet(sigma=sigma, hysteresis=lam_h)
    e = _run(mal, 5000)
    assert np.all(np.isfinite(e)), "non-finite energy"
    # Total energy is monotone non-increasing (each step removes >= 0, within a tiny tolerance).
    assert np.max(np.diff(e)) <= 1e-9 * e[0], f"loss added energy (sigma={sigma}, lam_h={lam_h})"


# -- Criterion 4: a mallet that never contacts leaves the membrane bit-for-bit unchanged ----------
def test_missing_mallet_is_bit_identical_to_bare_membrane():
    # A plucked membrane wrapped in a mallet held clear of the head must evolve identically to the
    # bare membrane (the collision's K=0 analog: no contact -> zero coupling force every step).
    N, steps = 40, 400
    bare = make_membrane(domain="rectangle", N=N, lam=0.5, sigma=0.3)
    # A low mode as the initial displacement (a soft pluck of the drumhead).
    phi = np.sin(np.pi * bare.X) * np.sin(np.pi * bare.Y) * 1e-3
    bare.set_state(phi)

    membrane = make_membrane(domain="rectangle", N=N, lam=0.5, sigma=0.3)
    membrane.set_state(phi.copy())
    # gap huge and the mallet moving away -> it never reaches the head within the window.
    mal = MalletMembrane(
        membrane=membrane, mass=0.02, stiffness=5.0e4, alpha=2.3,
        strike_x=0.5, strike_y=0.5, strike_velocity=-1.0, gap=10.0,
    )
    for _ in range(steps):
        bare.step()
        mal.step()
    assert mal.contact_force == 0.0 and not mal.in_contact
    np.testing.assert_array_equal(mal.membrane.u, bare.u)


def test_mallet_membrane_is_a_resonator():
    # Duck-types the engine's Resonator protocol.
    mal = make_mallet()
    assert isinstance(mal, MalletMembrane)
    assert hasattr(mal, "k") and callable(mal.step) and callable(mal.energy)
    assert mal.state.shape == mal.membrane.mask.shape
    assert isinstance(mal.displacement_at(0), float)


def test_circle_membrane_strike_conserves():
    # The staircased circular drumhead conserves too (energy is geometry-independent).
    mal = make_mallet(domain="circle", N=40, lam=0.5, strike_x=0.0, strike_y=0.0, sigma=0.0)
    e = _run(mal, 4000)
    drift = np.max(np.abs(e - e[0])) / abs(e[0])
    assert drift < CONSERVE_TOL, f"circular-membrane energy drift {drift:.2e}"
