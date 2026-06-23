"""Free-free beam (model #5b-pre): energy conservation (lossless) and passivity (lossy).

The headline checks mirror the plate's (HANDOFF §6.1-§6.2), now for a **free-boundary** flexural
operator built energy-first:

- the cross-time energy is flat to machine precision even at ``mu = kappa k / h²`` well past the
  explicit bound 1/4 (the unconditional-stability win of the implicit theta-scheme);
- with loss, energy is monotone non-increasing, a single *low* mode decays at ~2 sigma, and
  the broad
  damping caveat (mid/high modes under-damp because ``Q = kappa²·mu`` is 4th-power) is
  itself pinned.

The free-free beam has no closed-form spatial mode shape, so the single-mode tests initialise
with the
numerically-computed eigenvector of ``K φ = mu W φ`` (the 2D plate will do the same for its modes).
"""

import numpy as np
import pytest
from helpers import make_beam
from scipy.sparse.linalg import eigsh

from physsynth.core.engine import simulate
from physsynth.core.exciter import raised_cosine

DRIFT_TOL = 1e-10  # the 1D/2D acceptance bar, unchanged


def _pluck(beam, amplitude=1e-3):
    """A smooth off-centre bump (band-limited; ends untouched, so nothing is clamped)."""
    return raised_cosine(beam.x, beam.L, 0.4 * beam.L, 0.2 * beam.L, amplitude)


def _run(beam, secs=1.0):
    beam.set_state(_pluck(beam))
    return simulate(beam, num_steps=int(secs * beam.fs))


def _elastic_eigenvector(beam, which):
    """The ``which``-th **elastic** eigenvector of ``K φ = mu W φ`` (0 = fundamental; skips the 2
    rigid-body modes). W-orthonormal (φᵀWφ = 1)."""
    mu1_est = (4.730041 / beam.L) ** 4
    vals, vecs = eigsh(beam.K, k=which + 3, M=beam.W, sigma=-1e-3 * mu1_est, which="LM")
    order = np.argsort(vals)
    return vecs[:, order[2 + which]]


# -- Conservation: lossless energy is flat to machine precision, across mu (incl.
# explicit-illegal). --
@pytest.mark.parametrize("mu", [0.5, 2.0, 8.0])
def test_energy_conserved(mu):
    beam = make_beam(N=64, mu=mu)
    res = _run(beam, secs=1.0)
    assert res.energy_drift < DRIFT_TOL, f"drift {res.energy_drift:.2e} at mu={mu}"


def test_energy_conserved_with_timestep_explicit_could_not_run():
    """Unconditional stability as a direct test: mu = 16 (>> 1/4) still conserves to machine eps."""
    beam = make_beam(N=64, mu=16.0)
    assert beam.mu > 0.25  # would blow up an explicit beam immediately
    res = _run(beam, secs=1.0)
    assert res.energy_drift < DRIFT_TOL, f"drift {res.energy_drift:.2e} at mu={beam.mu}"


def test_energy_strictly_positive_when_lossless():
    beam = make_beam(N=64, mu=2.0)
    res = _run(beam, secs=0.5)
    assert np.all(res.energy > 0.0)


# -- Passivity: with loss, energy decreases monotonically. --
def test_passivity_monotonic_decrease():
    beam = make_beam(N=64, mu=2.0, sigma=8.0)
    res = _run(beam, secs=1.0)
    e0 = res.energy[0]
    steps_up = np.diff(res.energy)
    assert np.all(steps_up <= 1e-10 * e0), f"max positive step {steps_up.max() / e0:.2e} * E0"


def test_decay_rate_matches_2sigma_low_mode():
    # A single LOW mode (Q k² « 1) decays at ~2 sigma; this is where the analytic rate holds.
    sigma, secs = 6.0, 0.5
    beam = make_beam(N=64, mu=2.0, sigma=sigma)
    phi = _elastic_eigenvector(beam, 0)  # fundamental
    beam.set_state(phi * 1e-3)
    res = simulate(beam, num_steps=int(secs * beam.fs))
    measured = res.energy[-1] / res.energy[0]
    expected = np.exp(-2.0 * sigma * secs)
    rel = abs(np.log(measured) - np.log(expected)) / abs(np.log(expected))
    assert rel < 0.02, f"low-mode decay off by {rel:.3%} (got {measured:.3e}, want {expected:.3e})"


def test_higher_mode_underdamps_relative_to_lower():
    """The damping caveat, pinned: rate 2 sigma (1 - theta Q k²) falls with mode (Q = kappa²·mu),
    so a
    higher mode retains MORE energy than a lower one after the same time — the opposite of a
    real beam,
    and the reason frequency-dependent loss (a later model) is needed."""
    sigma, secs = 6.0, 0.3
    ratios = []
    for which in (0, 6):  # fundamental vs a much higher elastic mode
        beam = make_beam(N=48, mu=8.0, sigma=sigma)  # large mu -> large k -> visible theta Q k²
        phi = _elastic_eigenvector(beam, which)
        beam.set_state(phi * 1e-3)
        res = simulate(beam, num_steps=int(secs * beam.fs))
        ratios.append(res.energy[-1] / res.energy[0])
    assert ratios[1] > ratios[0], f"high/low retained energy = {ratios} (expected high > low)"


def test_energy_units_scale_with_density():
    # energy() is in Joules and scales linearly with linear density rho (kappa fixed -> same grid,
    # same K, same frequencies); doubling rho doubles E for the same field/geometry.
    beam1 = make_beam(N=64, mu=2.0, rho=0.005)
    beam2 = make_beam(N=64, mu=2.0, rho=0.010)
    for beam in (beam1, beam2):
        beam.set_state(_pluck(beam))
    assert np.isclose(beam2.energy() / beam1.energy(), 2.0, rtol=1e-12)
