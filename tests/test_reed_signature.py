"""Signature validation for the single-reed mouthpiece — the *physics* oracle (wind #3).

The energy balance (``test_reed_energy``) is necessary but **not sufficient**: a wrong-sign / dead
reed rings down and the balance still passes on silence. The independent gate is the **clarinet
signature** — that the model actually plays:

- **self-sustained oscillation above a blowing threshold**, and decay below it (the reed is an
  *active* exciter, like the bow's Helmholtz motion);
- the tone is **locked near the bore's quarter-wave** ``f ~ c / 4L``, set by the *air column* not
  the reed (inward-striking) — so sweeping the reed frequency barely moves the pitch;
- **odd harmonics dominate** (the clarinet's hallmark: the fundamental dwarfs the second harmonic);
- the reed **beats shut** under strong blowing (the once-per-period slam), not at gentle blowing.

The clarinet's small-oscillation threshold is around ``gamma = p_mouth / p_closing ~ 1/3``
(Dalmont/Kergomard); here it sits a touch above that (this bore's losses), so the tests only assert
*speaks above / silent below* with comfortable margins rather than hard-asserting ``1/3``.
"""

import numpy as np
from helpers import make_reed_bore

from physsynth.analysis import modal, spectrum
from physsynth.core.bore import C0_AIR

F1_CO = modal.bore_resonance_frequencies(C0_AIR, 0.5, 1, "closed-open")[0]  # 171.5 Hz


def _mouthpiece(reed, secs):
    n = int(secs / reed.k)
    sig = np.empty(n)
    for i in range(n):
        reed.step()
        sig[i] = reed.mouthpiece_pressure()
    return sig


def _interior(reed, secs, node=1):
    n = int(secs / reed.k)
    sig = np.empty(n)
    for i in range(n):
        reed.step()
        sig[i] = reed.displacement_at(node)
    return sig


# -- Self-oscillation: above threshold the note speaks and sustains. ---------------------------
def test_self_sustains_above_threshold():
    reed = make_reed_bore(p_mouth=1500.0)  # gamma ~ 0.5
    assert reed.gamma > 0.4
    sig = _mouthpiece(reed, 0.6)
    first, second = sig[: len(sig) // 2], sig[len(sig) // 2 :]
    rms1, rms2 = np.sqrt(np.mean(first**2)), np.sqrt(np.mean(second**2))
    assert rms2 > 100.0, f"note did not speak (2nd-half rms {rms2:.2f} Pa)"
    assert rms2 > 0.5 * rms1, "amplitude decaying, not a sustained regime"


def test_silent_below_threshold():
    reed = make_reed_bore(p_mouth=300.0)  # gamma ~ 0.1, well below threshold
    assert reed.gamma < 0.2
    sig = _mouthpiece(reed, 0.6)
    rms_end = np.sqrt(np.mean(sig[len(sig) // 2 :] ** 2))
    assert rms_end < 5.0, f"note spoke below threshold (rms {rms_end:.3f} Pa)"


def test_threshold_is_between_gentle_and_hard_blowing():
    # A monotone speak-onset: gentle -> silent, hard -> loud. Brackets the threshold ~1/3..1/2.
    quiet = make_reed_bore(p_mouth=600.0)   # gamma ~ 0.2
    loud = make_reed_bore(p_mouth=1500.0)   # gamma ~ 0.5
    rq = np.sqrt(np.mean(_mouthpiece(quiet, 0.5)[-4000:] ** 2))
    rl = np.sqrt(np.mean(_mouthpiece(loud, 0.5)[-4000:] ** 2))
    assert rl > 50.0 * (rq + 1e-9), f"no clear speak onset (quiet {rq:.2f}, loud {rl:.2f})"


# -- Pitch: locked near the bore quarter-wave, set by the air column not the reed. -------------
def test_pitch_locked_near_quarter_wave():
    reed = make_reed_bore(p_mouth=1500.0)
    sig = _interior(reed, 0.8)
    freqs, mag, _ = spectrum.magnitude_spectrum(sig[len(sig) // 2 :], 1.0 / reed.k)
    f0 = freqs[1 + int(np.argmax(mag[1:]))]
    assert abs(f0 - F1_CO) / F1_CO < 0.1, f"pitch {f0:.1f} Hz not near c/4L = {F1_CO:.1f} Hz"


def test_pitch_set_by_bore_not_reed():
    # Sweep the reed frequency at fixed blowing ratio gamma (p_mouth scaled with p_closing, i.e.
    # f_reed^2) so both speak. The playing pitch barely moves — the BORE sets it (inward-striking).
    f0s = []
    for f_reed in (2000.0, 3000.0):
        p_closing = 0.03 * (2.0 * np.pi * f_reed) ** 2 * 4.0e-4
        reed = make_reed_bore(p_mouth=0.5 * p_closing, f_reed=f_reed)
        sig = _interior(reed, 0.8)
        freqs, mag, _ = spectrum.magnitude_spectrum(sig[len(sig) // 2 :], 1.0 / reed.k)
        f0s.append(freqs[1 + int(np.argmax(mag[1:]))])
    assert max(abs(f - F1_CO) / F1_CO for f in f0s) < 0.1, f"pitch tracks the reed: {f0s}"
    assert abs(f0s[0] - f0s[1]) / F1_CO < 0.06, f"pitch moved with the reed by {f0s}"


# -- Odd harmonics dominate (the clarinet hallmark). -------------------------------------------
def test_odd_harmonics_dominate():
    reed = make_reed_bore(p_mouth=1500.0)
    sig = _interior(reed, 0.8)
    freqs, mag, _ = spectrum.magnitude_spectrum(sig[len(sig) // 2 :], 1.0 / reed.k)
    df = freqs[1] - freqs[0]

    def peak(f):
        i = int(round(f / df))
        return mag[max(1, i - 3) : i + 4].max()

    assert peak(F1_CO) > 100.0 * peak(2 * F1_CO), "fundamental does not dwarf the 2nd harmonic"
    assert peak(3 * F1_CO) > peak(2 * F1_CO), "3rd (odd) harmonic weaker than 2nd (even)"


# -- The reed beats shut under strong blowing, stays open when gentle. -------------------------
def test_reed_beats_shut_when_blown_hard():
    reed = make_reed_bore(p_mouth=1800.0)  # gamma ~ 0.6
    beat = False
    for _ in range(int(0.5 / reed.k)):
        reed.step()
        if reed.reed_opening() <= 0.0:
            beat = True
    assert beat, "reed never beat shut under hard blowing"


def test_reed_stays_open_below_threshold():
    reed = make_reed_bore(p_mouth=300.0)  # gamma ~ 0.1
    min_open = np.inf
    for _ in range(int(0.4 / reed.k)):
        reed.step()
        min_open = min(min_open, reed.reed_opening())
    assert min_open > 0.0, "reed beat shut although it never spoke"
