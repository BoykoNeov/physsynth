"""Energy validation for the string <-> distributed-barrier collision (model #8, first distributed
contact model).

The barrier stores potential energy ``h*sum_j phi(eta_j)``, so — like the mallet — the correctness
statement is strict **conservation** of the total ``E = E_string + PE_barrier``: a lossless string
against an elastic barrier keeps it flat to machine precision. The conservation rests on the
applied contact force being the exact discrete gradient of the potential (now a *vector* discrete
gradient, one component per contact node), so a deliberately loose vector solve makes the drift grow
**in proportion to the solve tolerance** — the self-certification that the discrete-gradient
telescoping is what conserves energy. This gate proves only *internal consistency*; the absolute
coupling **magnitude** is pinned separately by the static-equilibrium oracle in
``test_collision_modal.py``. Loss (string ``sigma``) or a lossy barrier (``lambda_h``) only ever
removes energy (passivity survives the vector coupling), and a barrier out of reach leaves the
string bit-for-bit unchanged.
"""

import numpy as np
import pytest
from helpers import make_barrier_string, make_damped_string

from physsynth.core.collision import BarrierString

CONSERVE_TOL = 1e-10  # lossless, elastic: relative drift of E; observed ~2e-12


def _pluck(bar, amplitude):
    """Displace the string as an upward first-mode pluck (peaks above the rail); on release its
    antinode swings *down* through the barrier -> genuine intermittent contact (buzz)."""
    x = bar.string.x
    phi = amplitude * np.sin(np.pi * x / bar.string.L)
    bar.set_state(phi)


def _run(bar, steps):
    """Step ``bar`` for ``steps``, returning the per-step total-energy array.

    Also asserts the vector contact solve never hits its iteration cap: energy exactness holds only
    at the converged root (the applied force *is* the discrete gradient there), so a stall would
    silently corrupt the balance. This documents the convergence guarantee at its source rather than
    inferring a stall from downstream drift.
    """
    e = np.empty(steps + 1)
    e[0] = bar.energy()
    for i in range(1, steps + 1):
        bar.step()
        assert bar.newton_iters < bar.newton_maxiter, f"contact solve stalled at step {i}"
        e[i] = bar.energy()
    return e


# -- Criterion 1 (money test): lossless coupled energy conservation to machine precision ----------
@pytest.mark.parametrize("K,alpha", [
    (1.0e6, 1.5), (5.0e5, 1.0), (2.0e6, 2.0), (1.0e6, 3.0),
])
def test_lossless_energy_conserved(K, alpha):
    # No string loss, elastic barrier: E_string + barrier PE is conserved exactly through contact.
    bar = make_barrier_string(K=K, alpha=alpha, lam=0.4, sigma0=0.0, sigma1=0.0, hysteresis=0.0)
    _pluck(bar, 5.0e-3)
    e = _run(bar, 6000)
    drift = np.max(np.abs(e - e[0])) / abs(e[0])
    assert drift < CONSERVE_TOL, f"energy drift {drift:.2e} (K={K}, alpha={alpha})"


def test_string_actually_contacts():
    # Sanity: the barrier must actually be hit for a real fraction of the run (else conservation is
    # a trivial free-string re-test and a vector-solve bug would hide). Mirror the mallet's
    # "coupling is real" guard.
    bar = make_barrier_string(lam=0.4)
    _pluck(bar, 5.0e-3)
    contact_steps = 0
    max_force = 0.0
    for _ in range(6000):
        bar.step()
        if np.any(bar.penetration > 0.0):
            contact_steps += 1
        max_force = max(max_force, np.max(np.abs(bar.contact_force)))
    assert contact_steps > 300, f"only {contact_steps} contact steps — barrier barely touched"
    assert max_force > 1.0, f"peak contact force {max_force:.2e} N too small to test the coupling"


# -- Criterion 2: conservation is discrete-gradient-limited (drift proportional to solve tol) -----
def test_drift_scales_with_newton_tolerance():
    # Tightening the vector contact solve tightens the conservation: the applied force approaches
    # the exact vector discrete gradient. A loose solve drifts proportionally -> telescoping wins.
    drifts = []
    tols = [1e-13, 1e-9, 1e-5]
    for tol in tols:
        bar = make_barrier_string(K=1.0e6, alpha=1.5, lam=0.4, newton_tol=tol)
        _pluck(bar, 5.0e-3)
        e = _run(bar, 2500)
        drifts.append(np.max(np.abs(e - e[0])) / abs(e[0]))
    assert drifts[0] < CONSERVE_TOL, f"tight-tol drift {drifts[0]:.2e} not machine precision"
    assert drifts[-1] > drifts[0] * 100, (
        f"loosening the solve did not increase drift ({drifts}); conservation is not solve-limited"
    )


# -- Criterion 3: passivity survives the coupling — loss only removes energy ---------------------
@pytest.mark.parametrize("sigma0,sigma1,lam_h", [
    (2.0, 0.0, 0.0), (0.0, 0.0, 5.0e4), (1.0, 0.05, 2.0e4),
])
def test_loss_only_removes_energy(sigma0, sigma1, lam_h):
    bar = make_barrier_string(
        K=1.0e6, alpha=1.5, lam=0.4, sigma0=sigma0, sigma1=sigma1, hysteresis=lam_h
    )
    _pluck(bar, 5.0e-3)
    e = _run(bar, 5000)
    assert np.all(np.isfinite(e)), "non-finite energy"
    # Total energy is monotone non-increasing (each step removes >= 0, within a roundoff tolerance).
    assert np.max(np.diff(e)) <= 1e-9 * e[0], (
        f"loss added energy (sigma0={sigma0}, sigma1={sigma1}, lam_h={lam_h}): "
        f"max rise {np.max(np.diff(e)):.2e}"
    )


# -- Criterion 4: a barrier out of reach leaves the string bit-for-bit unchanged (the K=0 analog) --
def test_out_of_reach_barrier_is_bit_identical_to_bare_string():
    # A plucked string wrapped in a barrier positioned far below it must evolve identically to the
    # bare DampedStiffString: no penetration -> zero contact force every step -> u += 0 exactly.
    N, steps = 80, 500
    amp = 5.0e-3
    x = np.linspace(0.0, 1.0, N + 1)
    phi = amp * np.sin(np.pi * x)

    bare = make_damped_string(N=N, lam=0.9, kappa=0.0, sigma0=0.3)
    bare.set_state(phi.copy())

    bar = make_barrier_string(N=N, lam=0.9, kappa=0.0, sigma0=0.3, barrier=-100.0)
    bar.set_state(phi.copy())

    for _ in range(steps):
        bare.step()
        bar.step()
    assert np.all(bar.contact_force == 0.0) and not np.any(bar.penetration > 0.0)
    np.testing.assert_array_equal(bar.string.u, bare.u)


def test_barrier_string_is_a_resonator():
    # Duck-types the engine's Resonator protocol.
    bar = make_barrier_string()
    assert isinstance(bar, BarrierString)
    assert hasattr(bar, "k") and callable(bar.step) and callable(bar.energy)
    assert bar.state.shape == bar.string.u.shape
    assert isinstance(bar.displacement_at(1), float)


def test_stiff_string_barrier_conserves():
    # A stiff (kappa > 0) string against the barrier conserves too (the biharmonic term is inside
    # string.energy(); the contact scheme is agnostic to the host operator).
    bar = make_barrier_string(K=1.0e6, alpha=1.5, lam=0.4, kappa=2.0)
    _pluck(bar, 5.0e-3)
    e = _run(bar, 5000)
    drift = np.max(np.abs(e - e[0])) / abs(e[0])
    assert drift < CONSERVE_TOL, f"stiff-string-barrier energy drift {drift:.2e}"
