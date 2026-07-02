"""Modal / Helmholtz validation for the bowed string (the exciter-specific correctness tests).

A sustained bowed note is **Helmholtz motion**: a single travelling corner, so the bow-point
transverse velocity is a two-state waveform — it *sticks* at the bow speed for a fraction
``1 - beta`` of the period, then *slips* once (fraction ``beta`` = bow-to-nut distance / L). The
fundamental is the string's own ``f_1 = c/2L``, **independent of bow speed and force** (that is why
a violinist controls loudness, not pitch, with the bow), while the amplitude scales with bow speed.
These facts — correct pitch, bow-speed-independent pitch, amplitude ∝ bow speed, and slip-fraction
``= beta`` with one slip per period — are what actually exercise the friction coupling.

**Schelleng.** A clean fundamental Helmholtz exists only when the bow force sits inside the
``[F_min, F_max]`` window *for that geometry*, and that window scales with ``v_bow`` and narrows as
the bow moves off the bridge (``F_max`` falls with ``beta``). The scheme reproduces this: too much
force for a given ``beta`` breaks into subharmonics/raucous motion, too little (relative to
``v_bow``) crushes the note. So these tests pick ``(beta, force, v_bow)`` inside the window
(``force <= 0.4`` clean up to ``beta = 0.25``; ``force ~ 4 v_bow`` holds the window across speeds).
"""

import numpy as np
import pytest
from helpers import L_DEFAULT, make_bowed_string, wave_speed

from physsynth.analysis import spectrum

F1 = wave_speed() / (2.0 * L_DEFAULT)  # 100 Hz on the canonical rig


def _bow_to_steady(bow, secs=2.5, tail_frac=0.4):
    """Run ``bow`` for ``secs`` and return the steady-tail pickup signal, bow-relative-velocity
    trace, and sample rate. The tail (last ``tail_frac``) is the settled Helmholtz regime."""
    fs = bow.string.fs
    steps = int(secs * fs)
    pickup = bow.string.N // 3
    sig = np.empty(steps)
    vrel = np.empty(steps)
    for n in range(steps):
        bow.step()
        sig[n] = bow.displacement_at(pickup)
        vrel[n] = bow.v_rel
    i0 = int((1.0 - tail_frac) * steps)
    return sig[i0:], vrel[i0:], fs


def _pitch(sig, fs, search=0.15):
    sig = sig - sig.mean()
    return float(spectrum.measure_partials_near(sig, fs, np.array([F1]), search_hz=search * F1)[0])


def _slip_mask(vrel_tail, v_bow):
    """Two-state split: the string *sticks* when it co-moves with the bow (``|v_rel| ~ 0``) and
    *slips* when it swings away. An absolute threshold at half the bow speed cleanly separates them
    and captures the whole slip *duration* (the smooth friction curve rounds the corner, so a
    relative-to-peak threshold would only catch the trough)."""
    return np.abs(vrel_tail) >= 0.5 * v_bow


# -- Criterion 1: sustained Helmholtz sits at the string fundamental f_1 = c/2L ---------------
def test_helmholtz_pitch_at_fundamental():
    bow = make_bowed_string()  # beta=0.13, force=1.0, v_bow=0.1 -> clean Helmholtz
    sig, _, fs = _bow_to_steady(bow)
    cents = 1200.0 * np.log2(_pitch(sig, fs) / F1)
    assert abs(cents) < 60.0, f"bowed pitch is {cents:.1f} cents off f1={F1:.1f}"


# -- Criterion 2: pitch is independent of bow speed (force ~ v_bow to stay in the window) -----
def test_pitch_independent_of_bow_speed():
    freqs = []
    for v_bow in (0.1, 0.15, 0.2):
        bow = make_bowed_string(bow_position=0.13, v_bow=v_bow, force=4.0 * v_bow)
        sig, _, fs = _bow_to_steady(bow)
        freqs.append(_pitch(sig, fs))
    freqs = np.array(freqs)
    spread_cents = 1200.0 * np.log2(freqs.max() / freqs.min())
    assert spread_cents < 25.0, f"pitch moved {spread_cents:.1f} cents across bow speeds {freqs}"


def test_amplitude_scales_with_bow_speed():
    # Helmholtz amplitude is ~ proportional to bow speed: doubling the bow speed (holding the
    # position in the Schelleng window via force ~ v_bow) roughly doubles the note's amplitude.
    amps = []
    for v_bow in (0.1, 0.2):
        bow = make_bowed_string(bow_position=0.13, v_bow=v_bow, force=4.0 * v_bow)
        sig, _, _ = _bow_to_steady(bow)
        amps.append(np.max(np.abs(sig - sig.mean())))
    ratio = amps[1] / amps[0]
    assert 1.5 < ratio < 2.5, f"amplitude ratio {ratio:.2f} for 2x bow speed (expected ~2): {amps}"


# -- Criterion 3 (money test): slip fraction = beta, one slip per period ----------------------
@pytest.mark.parametrize("beta_pos", [0.13, 0.2, 0.25])
def test_slip_fraction_matches_beta(beta_pos):
    # force = 0.4 is inside the Helmholtz window across this beta range (Schelleng).
    bow = make_bowed_string(bow_position=beta_pos, force=0.4)
    _, vrel, _ = _bow_to_steady(bow)
    slip = float(np.mean(_slip_mask(vrel, bow.v_bow)))
    assert abs(slip - bow.beta) < 0.05, f"slip fraction {slip:.3f} != beta {bow.beta:.3f}"


@pytest.mark.parametrize("beta_pos", [0.13, 0.2, 0.25])
def test_one_slip_per_period(beta_pos):
    bow = make_bowed_string(bow_position=beta_pos, force=0.4)
    _, vrel, fs = _bow_to_steady(bow)
    slipping = _slip_mask(vrel, bow.v_bow)
    onsets = int(np.sum((~slipping[:-1]) & (slipping[1:])))
    slips_per_period = onsets / (len(vrel) * F1 / fs)  # onsets / number-of-periods-in-tail
    assert 0.85 < slips_per_period < 1.25, (
        f"{slips_per_period:.2f} slips/period (expected ~1) at beta={bow.beta:.3f}"
    )


def test_slip_velocity_matches_helmholtz_prediction():
    # During slip the bow-point velocity swings to ~ -v_bow (1-beta)/beta (the two-slope corner).
    bow = make_bowed_string(bow_position=0.13, force=0.4)
    _, vrel, _ = _bow_to_steady(bow)
    v_slip_measured = (vrel + bow.v_bow).min()  # most-negative string velocity at the bow node
    v_slip_ideal = -bow.v_bow * (1.0 - bow.beta) / bow.beta
    assert abs(v_slip_measured - v_slip_ideal) < 0.4 * abs(v_slip_ideal), (
        f"slip velocity {v_slip_measured:.3f} vs ideal {v_slip_ideal:.3f}"
    )
