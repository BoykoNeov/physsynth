"""Plate (model #5): energy conservation (lossless) and passivity (lossy).

HANDOFF §6.1-§6.2 for the flexural (bending-only) operator. The headline checks:

- the cross-time energy is flat to machine precision **even at mu = kappa k / h² well past the
  explicit bound 1/4** — the unconditional-stability win the implicit theta-scheme buys (a regime
  the explicit plate could not run at all);
- with loss, energy is monotone non-increasing, and a *single low mode* decays at ~2 sigma. The
  caveat (mid/high modes under-damp because Q = kappa² Λ² is 4th-power across the whole spectrum)
  is
  itself pinned as a test: a higher mode decays strictly slower than a lower one.
"""

import numpy as np
import pytest
from helpers import make_plate

from physsynth.analysis import modal
from physsynth.core.engine import simulate
from physsynth.core.exciter import raised_cosine_2d

DRIFT_TOL = 1e-10  # the 1D/2D acceptance bar, unchanged


def _pluck(p, amplitude=1e-3):
    """A smooth off-centre bump, clamped to the live region."""
    field = raised_cosine_2d(p.X, p.Y, (0.4 * p.Lx, 0.55 * p.Ly), 0.25 * min(p.Lx, p.Ly), amplitude)
    field[~p.mask] = 0.0
    return field


def _run(p, secs=1.0):
    p.set_state(_pluck(p))
    return simulate(p, num_steps=int(secs * p.fs))


# -- Conservation: lossless energy is flat to machine precision, across mu (incl.
# explicit-illegal). --
@pytest.mark.parametrize("mu", [0.5, 2.0, 8.0])
def test_energy_conserved(mu):
    # mu > 1/4 is unstable for an EXPLICIT plate; the implicit theta-scheme conserves regardless.
    p = make_plate(N=32, mu=mu)
    res = _run(p, secs=1.0)
    assert res.energy_drift < DRIFT_TOL, f"drift {res.energy_drift:.2e} at mu={mu}"


def test_energy_conserved_with_timestep_explicit_could_not_run():
    """The unconditional-stability claim, made a direct test: mu = 16 (>> 1/4) still conserves."""
    p = make_plate(N=32, mu=16.0)
    assert p.mu > 0.25  # would blow up an explicit plate immediately
    res = _run(p, secs=1.0)
    assert res.energy_drift < DRIFT_TOL, f"drift {res.energy_drift:.2e} at mu={p.mu}"


def test_energy_strictly_positive_when_lossless():
    p = make_plate(N=32, mu=2.0)
    res = _run(p, secs=0.5)
    assert np.all(res.energy > 0.0)


# -- Passivity: with loss, energy decreases monotonically. --
def test_passivity_monotonic_decrease():
    p = make_plate(N=32, mu=2.0, sigma=8.0)
    res = _run(p, secs=1.0)
    e0 = res.energy[0]
    steps_up = np.diff(res.energy)
    assert np.all(steps_up <= 1e-10 * e0), f"max positive step {steps_up.max() / e0:.2e} * E0"


def test_decay_rate_matches_2sigma_low_mode():
    # A single LOW mode (Q k² « 1) decays at ~2 sigma; this is where the analytic rate holds.
    sigma, secs = 6.0, 0.5
    p = make_plate(N=32, mu=2.0, sigma=sigma)
    phi = modal.rectangular_mode_field(p.X, p.Y, p.Lx, p.Ly, 1, 1)
    p.set_state(phi * 1e-3)
    res = simulate(p, num_steps=int(secs * p.fs))
    measured = res.energy[-1] / res.energy[0]
    expected = np.exp(-2.0 * sigma * secs)
    rel = abs(np.log(measured) - np.log(expected)) / abs(np.log(expected))
    assert rel < 0.02, f"low-mode decay off by {rel:.3%} (got {measured:.3e}, want {expected:.3e})"


def test_higher_mode_underdamps_relative_to_lower():
    """The damping caveat, pinned: rate 2 sigma (1 - theta Q k²) falls with mode
    (Q = kappa² Λ²),

    so a higher mode retains MORE energy than a lower one after the same time — the opposite of a
    real plate, and the reason frequency-dependent loss (a later model) is needed. We compare a low
    mode against a much higher one at a deliberately coarse timestep (large k) so theta Q k² is
    non-negligible for the high mode.
    """
    sigma, secs = 6.0, 0.3
    ratios = []
    for m in (1, 8):  # mode (m, m): Q grows ~ (m²)² , so the effect is large by m = 8
        p = make_plate(N=32, mu=8.0, sigma=sigma)  # large mu -> large k -> visible theta Q k²
        phi = modal.rectangular_mode_field(p.X, p.Y, p.Lx, p.Ly, m, m)
        p.set_state(phi * 1e-3)
        res = simulate(p, num_steps=int(secs * p.fs))
        ratios.append(res.energy[-1] / res.energy[0])
    # The high mode retains strictly more energy (under-damps) than the low mode.
    assert ratios[1] > ratios[0], f"high/low retained energy = {ratios} (expected high > low)"


def test_energy_units_scale_with_density():
    # energy() is in Joules and scales linearly with areal density rho_s; doubling rho doubles E for
    # the same field/geometry (kappa fixed -> same grid, same B, same frequencies).
    p1 = make_plate(N=32, mu=2.0, rho=0.005)
    p2 = make_plate(N=32, mu=2.0, rho=0.010)
    for p in (p1, p2):
        p.set_state(_pluck(p))
    assert np.isclose(p2.energy() / p1.energy(), 2.0, rtol=1e-12)
