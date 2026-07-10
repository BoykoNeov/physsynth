"""Physical-signature checks for the string <-> distributed-barrier collision (model #8).

These are the *diagnostic* tier (like Schelleng for the bow): qualitative behaviours a real buzzing
string must show, not machine-precision gates. The barrier turns a clean plucked tone into a buzz —
it injects energy into high partials, brightens as the contact gets harder/closer, and it is an
*intermittent* contact (the string slaps the barrier and springs off, repeatedly), not a string
permanently pinned to a shorter length.
"""

import numpy as np
from helpers import make_barrier_string


def _spectral_centroid(sig, fs):
    """Amplitude-weighted mean frequency (Hz) of ``sig`` — a brightness proxy."""
    w = np.hanning(len(sig))
    mag = np.abs(np.fft.rfft(sig * w))
    freqs = np.fft.rfftfreq(len(sig), 1.0 / fs)
    return float(np.sum(freqs * mag) / np.sum(mag))


def _pluck_and_record(bar, amplitude, steps, pickup_frac=0.1):
    """Pluck (upward first mode), run ``steps``, return the pickup time series (a node near one end)
    and the per-step in-contact flag."""
    x = bar.string.x
    bar.set_state(amplitude * np.sin(np.pi * x / bar.string.L))
    node = max(1, int(pickup_frac * bar.string.N))
    pickup = np.empty(steps)
    contact = np.zeros(steps, dtype=bool)
    for i in range(steps):
        bar.step()
        pickup[i] = bar.string.displacement_at(node)
        contact[i] = np.any(bar.penetration > 0.0)
    return pickup, contact


def test_barrier_brightens_the_tone():
    # Same string, same pluck: a barrier in reach must raise the spectral centroid vs one out of
    # reach (the barrier contact pumps energy into high partials — audible buzz).
    steps = 8000
    free = make_barrier_string(lam=0.4, sigma0=0.5, barrier=-100.0)  # out of reach -> clean tone
    buzz = make_barrier_string(lam=0.4, sigma0=0.5, barrier=-2.0e-3)  # in reach -> buzz
    p_free, _ = _pluck_and_record(free, 5.0e-3, steps)
    p_buzz, c_buzz = _pluck_and_record(buzz, 5.0e-3, steps)
    assert np.any(c_buzz), "buzz case never contacted the barrier"
    cf = _spectral_centroid(p_free, 1.0 / free.k)
    cb = _spectral_centroid(p_buzz, 1.0 / buzz.k)
    assert cb > 1.3 * cf, f"barrier did not brighten the tone: centroid free={cf:.0f} buzz={cb:.0f}"


def test_closer_barrier_is_brighter():
    # A closer barrier is struck more often / harder -> more high-partial energy -> brighter.
    steps = 8000
    far = make_barrier_string(lam=0.4, sigma0=0.5, barrier=-4.0e-3)
    near = make_barrier_string(lam=0.4, sigma0=0.5, barrier=-1.0e-3)
    p_far, _ = _pluck_and_record(far, 5.0e-3, steps)
    p_near, _ = _pluck_and_record(near, 5.0e-3, steps)
    c_far = _spectral_centroid(p_far, 1.0 / far.k)
    c_near = _spectral_centroid(p_near, 1.0 / near.k)
    assert c_near > c_far, f"closer barrier not brighter: far={c_far:.0f} near={c_near:.0f}"


def test_contact_is_intermittent():
    # A buzzing string slaps the barrier and springs off many times — it must show multiple
    # separate contact episodes, and must NOT be permanently pinned (some steps are contact-free).
    steps = 8000
    bar = make_barrier_string(lam=0.4, sigma0=0.2, barrier=-2.0e-3)
    _, contact = _pluck_and_record(bar, 5.0e-3, steps)
    onsets = int(np.sum(contact[1:] & ~contact[:-1]))  # no-contact -> contact transitions
    frac_contact = float(np.mean(contact))
    assert onsets >= 3, f"expected repeated slaps, saw {onsets} contact onsets"
    assert frac_contact < 0.9, f"string is essentially pinned (in contact {frac_contact:.0%})"


def test_harder_barrier_shortens_contact():
    # A stiffer barrier (higher K) rebounds the string faster -> a smaller fraction of time in
    # contact than a soft one, at the same pluck (the felt/wood-hardness -> shorter-slap signature).
    steps = 8000
    soft = make_barrier_string(lam=0.4, sigma0=0.2, K=1.0e5, alpha=1.5, barrier=-2.0e-3)
    hard = make_barrier_string(lam=0.4, sigma0=0.2, K=5.0e6, alpha=1.5, barrier=-2.0e-3)
    _, c_soft = _pluck_and_record(soft, 5.0e-3, steps)
    _, c_hard = _pluck_and_record(hard, 5.0e-3, steps)
    assert np.mean(c_hard) < np.mean(c_soft), (
        f"harder barrier not shorter contact: soft={np.mean(c_soft):.3f} hard={np.mean(c_hard):.3f}"
    )
