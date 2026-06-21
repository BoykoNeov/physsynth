"""Numerical-dispersion relation (HANDOFF §6 test 5).

The other modal tests cover the *exact* corner (partials at lambda = 1) and the convergence *rate*
as h -> 0. Neither checks that ``step()`` reproduces the predicted frequency across the *whole mode
range at lambda < 1* — that is what dispersion fills, and it is where a coefficient or indexing bug
hides while remaining invisible at lambda = 1.

Three assertions, separating "is the code right" from "is the physics right":

* **measured <-> oracle, lambda < 1** (``test_dispersion_matches_oracle_below_lambda_one``): the
  FDTD frequency of every swept mode lands on the closed-form discrete-dispersion oracle. This is
  the discriminating test of the implementation.
* **measured <-> continuum, lambda = 1** (``test_dispersion_flat_at_lambda_one``): at lambda = 1 the
  scheme is dispersionless, so the measured curve sits on the *continuum* ``n*c/(2L)`` — anchoring
  the result to physics, not only to an oracle derived from the same recurrence.
* **droop direction** (``test_phase_velocity_droops_with_mode_below_lambda_one``): for lambda < 1
  the phase velocity falls below ``c`` and keeps falling as the mode number rises (high partials
  travel too slowly).

Tolerances are provisional (HANDOFF §11.5 is still the human's call); they are set a few times
looser than the observed clean residuals so a real regression trips them without flakiness.
"""

import numpy as np
import pytest
from helpers import L_DEFAULT, measure_mode_frequencies, wave_speed

from physsynth.analysis import dispersion, modal

# A coarse enough grid that low modes are well below Nyquist, with a sub-Nyquist lambda that makes
# dispersion plainly visible. Modes span up to 0.75 N (the top decile is measurement-limited).
N = 128
MODES = np.array([2, 4, 8, 16, 32, 48, 64, 80, 96])
LAMBDA_DISPERSIVE = 0.8

# Provisional bars (see module docstring / HANDOFF §11.5), set a few times above the observed clean
# residuals: at lambda=0.8 the worst measured-vs-oracle error is ~1.3e-5 (the lowest mode, limited
# only by FFT resolution over the window); at lambda=1 the scheme is exact and the worst
# measured-vs-continuum error is ~8e-10 (machine precision bar the lowest mode).
ORACLE_RTOL = 1e-4  # measured FDTD frequency vs the discrete oracle
CONTINUUM_RTOL = 1e-7  # measured FDTD frequency vs the continuum, at lambda = 1


@pytest.mark.slow
def test_dispersion_matches_oracle_below_lambda_one():
    # The implementation test: every swept mode's measured frequency must match the closed-form
    # discrete-dispersion oracle at lambda < 1 (a regime the lambda=1 modal test never exercises).
    c = wave_speed()
    measured = measure_mode_frequencies(MODES, N=N, lam=LAMBDA_DISPERSIVE)
    oracle = dispersion.dispersion_frequencies(c, L_DEFAULT, N, LAMBDA_DISPERSIVE, MODES)

    assert not np.any(np.isnan(measured)), "a mode frequency was not detected"
    rel = np.abs(measured - oracle) / oracle
    assert np.max(rel) < ORACLE_RTOL, (
        f"worst measured-vs-oracle error {np.max(rel):.2e} at mode {MODES[np.argmax(rel)]}"
    )


@pytest.mark.slow
def test_dispersion_flat_at_lambda_one():
    # The physics anchor: at lambda = 1 the scheme is dispersionless, so the measured curve sits on
    # the *continuum* harmonic series (not just on an oracle drawn from the same recurrence).
    c = wave_speed()
    measured = measure_mode_frequencies(MODES, N=N, lam=1.0)
    continuum = MODES * c / (2.0 * L_DEFAULT)

    assert not np.any(np.isnan(measured)), "a mode frequency was not detected"
    rel = np.abs(measured - continuum) / continuum
    assert np.max(rel) < CONTINUUM_RTOL, (
        f"worst measured-vs-continuum error at lambda=1: {np.max(rel):.2e}"
    )

    vp_over_c = dispersion.phase_velocity(measured, L_DEFAULT, MODES) / c
    assert np.allclose(vp_over_c, 1.0, atol=CONTINUUM_RTOL), "phase velocity not flat at lambda=1"


@pytest.mark.slow
def test_phase_velocity_droops_with_mode_below_lambda_one():
    # The direction-of-physics anchor: dispersion slows high partials, so v_p / c < 1 and falls
    # monotonically with the mode number for lambda < 1.
    c = wave_speed()
    measured = measure_mode_frequencies(MODES, N=N, lam=LAMBDA_DISPERSIVE)
    vp_over_c = dispersion.phase_velocity(measured, L_DEFAULT, MODES) / c

    assert np.all(vp_over_c < 1.0), f"a phase velocity was not below c: {vp_over_c}"
    # Strictly decreasing across the (increasing) mode sweep.
    assert np.all(np.diff(vp_over_c) < 0.0), (
        f"phase velocity not monotonically drooping: {vp_over_c}"
    )


# -- fast unit tests for the pure dispersion helpers (no simulation) ----------------------


def test_phase_velocity_recovers_c_for_continuum():
    # v_p = 2 L f / m applied to the continuum series f = m c / (2L) must return c for every mode.
    c, L = wave_speed(), L_DEFAULT
    modes = np.arange(1, 21)
    f_cont = modes * c / (2.0 * L)
    assert np.allclose(dispersion.phase_velocity(f_cont, L, modes), c, rtol=1e-12)


def test_dispersion_frequencies_match_scalar_oracle_and_droop():
    # The vectorised oracle equals the scalar one, equals the continuum at lambda = 1, and lies
    # below it for lambda < 1.
    c, L = wave_speed(), L_DEFAULT
    modes = np.array([1, 5, 10, 25, 50])
    vec = dispersion.dispersion_frequencies(c, L, N, LAMBDA_DISPERSIVE, modes)
    scalar = np.array([modal.discrete_mode_frequency(c, L, N, LAMBDA_DISPERSIVE, m) for m in modes])
    assert np.allclose(vec, scalar, rtol=1e-15)

    at_one = dispersion.dispersion_frequencies(c, L, N, 1.0, modes)
    continuum = modes * c / (2.0 * L)
    assert np.allclose(at_one, continuum, rtol=1e-12)
    assert np.all(vec < continuum)  # dispersion lowers every mode for lambda < 1
