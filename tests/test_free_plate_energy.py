"""Free-edge (FFFF) plate (model #5b): energy conservation (lossless) and passivity (lossy).

The energy-first construction makes ``K`` symmetric and ``W`` SPD, so the W-weighted theta-scheme
conserves the discrete energy to machine precision (lossless) and is monotone non-increasing (lossy)
**unconditionally** — including at a plate-Courant ``mu = kappa k / h²`` far past the explicit bound
1/4. The broad damping caveat (``Q = kappa²·mu`` 4th-power across the spectrum -> mid/high modes
under-damp) carries over from the beam/SS plate and is itself pinned as a test.
"""

import numpy as np
import pytest
from helpers import make_free_plate
from scipy.sparse.linalg import eigsh

from physsynth.core.engine import simulate
from physsynth.core.exciter import raised_cosine_2d

DRIFT_TOL = 1e-10  # the 1D/2D acceptance bar, unchanged


def _pluck(p, amplitude=1e-3):
    """A smooth off-centre bump (every free node is live, so nothing is clamped)."""
    a = p.Lx
    return raised_cosine_2d(p.X, p.Y, (0.4 * a, 0.55 * a), 0.25 * a, amplitude)


def _elastic_mode(p, which):
    """The ``which``-th elastic eigenvector (0 = saddle fundamental), as a full live-node vector."""
    a = p.Lx
    mu1 = (13.0 / (a * a)) ** 2
    vals, vecs = eigsh(p.K, k=which + 4, M=p.W, sigma=-1e-3 * mu1, which="LM")
    order = np.argsort(vals)
    return vecs[:, order[3 + which]]


def _run(p, secs=1.0):
    p.set_state(_pluck(p))
    return simulate(p, num_steps=int(secs * p.fs))


# -- Conservation: lossless energy is flat to machine precision, across mu (incl. explicit-illegal).
@pytest.mark.parametrize("mu", [0.5, 2.0, 8.0])
def test_energy_conserved(mu):
    p = make_free_plate(N=32, mu=mu)
    res = _run(p, secs=1.0)
    assert res.energy_drift < DRIFT_TOL, f"drift {res.energy_drift:.2e} at mu={mu}"


def test_energy_conserved_with_timestep_explicit_could_not_run():
    """The unconditional-stability claim, made a direct test: mu = 16 (>> 1/4) still conserves."""
    p = make_free_plate(N=32, mu=16.0)
    assert p.mu > 0.25  # would blow up an explicit plate immediately
    res = _run(p, secs=1.0)
    assert res.energy_drift < DRIFT_TOL, f"drift {res.energy_drift:.2e} at mu={p.mu}"


def test_energy_strictly_positive_when_lossless():
    p = make_free_plate(N=32, mu=2.0)
    res = _run(p, secs=0.5)
    assert np.all(res.energy > 0.0)


# -- Passivity: with loss, energy decreases monotonically. --
def test_passivity_monotonic_decrease():
    p = make_free_plate(N=32, mu=2.0, sigma=8.0)
    res = _run(p, secs=1.0)
    e0 = res.energy[0]
    steps_up = np.diff(res.energy)
    assert np.all(steps_up <= 1e-10 * e0), f"max positive step {steps_up.max() / e0:.2e} * E0"


def test_decay_rate_matches_2sigma_low_mode():
    """A single LOW mode (the saddle fundamental, Q k² « 1) decays at ~2 sigma."""
    sigma, secs = 6.0, 0.5
    p = make_free_plate(N=32, mu=2.0, sigma=sigma)
    p.set_state(_elastic_mode(p, 0) * 1e-3)
    res = simulate(p, num_steps=int(secs * p.fs))
    measured = res.energy[-1] / res.energy[0]
    expected = np.exp(-2.0 * sigma * secs)
    rel = abs(np.log(measured) - np.log(expected)) / abs(np.log(expected))
    assert rel < 0.03, f"low-mode decay off by {rel:.3%} (got {measured:.3e}, want {expected:.3e})"


def test_higher_mode_underdamps_relative_to_lower():
    """The damping caveat, pinned: rate 2σ(1 - θ Q k²) falls with mode (Q = kappa²·mu), so a higher
    mode retains MORE energy than the fundamental after the same time -- the opposite of a real
    plate, and the reason frequency-dependent loss (a later model) is needed. Coarse timestep (large
    mu -> large k) so θ Q k² is non-negligible for the high mode."""
    sigma, secs = 6.0, 0.3
    ratios = []
    for which in (0, 12):  # saddle fundamental vs a much higher mode
        p = make_free_plate(N=32, mu=8.0, sigma=sigma)  # large mu -> large k -> visible θ Q k²
        p.set_state(_elastic_mode(p, which) * 1e-3)
        res = simulate(p, num_steps=int(secs * p.fs))
        ratios.append(res.energy[-1] / res.energy[0])
    assert ratios[1] > ratios[0], f"high/low retained energy = {ratios} (expected high > low)"


def test_energy_units_scale_with_density():
    # energy() is in Joules and scales linearly with areal density rho_s; doubling rho doubles E for
    # the same field/geometry (kappa fixed -> same grid, same K/W, same frequencies).
    p1 = make_free_plate(N=32, mu=2.0, rho=0.005)
    p2 = make_free_plate(N=32, mu=2.0, rho=0.010)
    for p in (p1, p2):
        p.set_state(_pluck(p))
    assert np.isclose(p2.energy() / p1.energy(), 2.0, rtol=1e-12)
