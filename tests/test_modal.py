"""Modal-frequency match: detected partials vs the analytic harmonic series (criterion 2)."""

import numpy as np
from helpers import make_string, wave_speed

from physsynth.analysis import modal, spectrum
from physsynth.core.engine import simulate
from physsynth.core.exciter import triangular_pluck

ONE_CENT = 1.0  # acceptance criterion 2 tolerance


def _pluck_and_run(string, secs=2.0, pickup_frac=0.241):
    u0 = triangular_pluck(string.x, string.L, 0.137 * string.L, amplitude=1e-3)
    string.set_state(u0)
    steps = int(secs * string.fs)
    return simulate(string, num_steps=steps, pickup_index=int(round(pickup_frac * string.N)))


def test_partials_within_one_cent_at_lambda_one():
    # At lambda = 1 the explicit scheme is dispersion-free: partials should sit on n*c/(2L).
    string = make_string(N=100, lam=1.0)
    res = _pluck_and_run(string, secs=2.0)

    n_partials = 10
    analytic = modal.harmonic_frequencies(wave_speed(), string.L, n_partials)
    detected = spectrum.measure_partials_near(res.output, res.fs, analytic)
    err_cents = modal.cents(detected, analytic)

    assert not np.any(np.isnan(detected)), "a partial was not detected"
    assert np.max(np.abs(err_cents)) < ONE_CENT, (
        f"worst partial error {np.max(np.abs(err_cents)):.4f} cents"
    )


def test_blind_detection_finds_harmonic_series():
    # With no prior knowledge of where partials are, every strong peak the detector finds must lie
    # on the harmonic grid n*f1 (a guard against the guided search merely confirming its own
    # assumption). Which harmonics dominate depends on pluck/pickup position, so we check that each
    # detected peak snaps to *some* integer harmonic within a cent, and that the fundamental is hit.
    string = make_string(N=100, lam=1.0)
    res = _pluck_and_run(string, secs=2.0)

    f1 = wave_speed() / (2 * string.L)
    peaks = spectrum.detect_peaks(res.output, res.fs, n_peaks=6, f_min=50.0)
    assert len(peaks) == 6

    nearest_n = np.round(peaks / f1)
    err_cents = modal.cents(peaks, nearest_n * f1)
    assert np.max(np.abs(err_cents)) < ONE_CENT, f"a peak is off-grid: {err_cents}"
    assert len(np.unique(nearest_n)) == 6, "detected peaks collapsed onto duplicate harmonics"
    assert nearest_n.min() == 1, "fundamental not among the detected peaks"


def test_discrete_oracle_matches_continuous_at_lambda_one():
    # Internal consistency of the dispersion oracle: at lambda = 1 it equals the continuum value.
    c, L, N = wave_speed(), 1.0, 100
    for m in (1, 5, 10, 25):
        f_disc = modal.discrete_mode_frequency(c, L, N, lam=1.0, m=m)
        f_cont = m * c / (2 * L)
        assert np.isclose(f_disc, f_cont, rtol=1e-12)
