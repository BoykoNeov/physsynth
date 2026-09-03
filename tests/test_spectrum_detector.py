"""The partial detector itself, tested directly -- the one measurement primitive that never was.

Every modal test in this suite asks :func:`~physsynth.analysis.spectrum.measure_partials_near`
where a partial is and then compares that answer to an oracle. Fifteen test files depend on it and
not one of them tests *it*: they all assert ``abs(found - oracle) < tol``, which is a statement
about the model, and which is satisfied just as well by a detector that is quietly wrong somewhere
the models never look. This file looks there.

The failure that prompted it: the detector returned **-502 Hz**. ``measure_partials_near`` takes the
argmax *inside a search window*, and a window whose magnitudes fall monotonically puts that argmax
on the window's own edge -- a bin with no peak under it. The refiner then fitted a parabola through
three near-collinear log-magnitudes, the curvature came out at -7.9e-3, and the correction was
**-22.1 bins** from bin 4, throwing the estimate clean out of the window and past zero. Downstream,
``modal.cents`` took a log of that negative frequency and produced the NaN that ``json.dumps``
refuses, which is a 500 on the viewer -- but the frequency was already wrong long before anything
serialised it.

Note what does *not* discriminate the case: both real witnesses had **negative** curvature, so
testing the sign of the denominator catches neither. ``a=3, b=2, c=0`` is concave and decreasing and
still refines by -1.5 bins. The property that makes the interpolation meaningful, and the one that
bounds ``|delta| <= 1/2``, is that the bin is a genuine local maximum. That is what is asserted.
"""

from __future__ import annotations

import numpy as np
import pytest

from physsynth.analysis.spectrum import (
    _parabolic_refine,
    magnitude_spectrum,
    measure_partials_near,
)

FS = 8000.0
"""Sample rate for the synthetic records here -- arbitrary, nothing physical depends on it."""

WITNESSES = [
    ([0.00054482, 0.00045897, 0.00038359], -22.096),
    ([0.00037953, 0.00033298, 0.00028228], -4.311),
]
"""The two magnitude triples that produced the bug, and the correction each used to receive.

Taken from ``plate:guitar`` at ``Lx=0.15, Ly=0.80, N=16`` and ``Lx=0.30, Ly=0.80, N=24`` rendered
for 0.01 s -- a short record, so the FFT is 256 bins wide and the +-0.3*f1 window spans six of them
with no peak inside. Kept as literals rather than re-derived from a simulation: the arithmetic that
goes wrong is in the refiner, and it goes wrong for *any* signal that hands it these three numbers.
"""


@pytest.mark.parametrize("mag3, old_delta", WITNESSES)
def test_the_refiner_declines_a_bin_that_is_not_a_peak(mag3, old_delta):
    """A monotone triple must not move the estimate at all, let alone off the end of the axis."""
    nfft, i = 256, 4
    mag = np.zeros(nfft // 2 + 1)
    mag[i - 1 : i + 2] = mag3
    df = FS / nfft

    # what the unguarded formula would have said, recomputed here so the claim is not hearsay
    a, b, c = (np.log(x) for x in mag3)
    assert (a - 2.0 * b + c) < 0.0, "the witness is CONCAVE -- a sign-of-curvature guard misses it"
    assert 0.5 * (a - c) / (a - 2.0 * b + c) == pytest.approx(old_delta, abs=1e-3)

    f = _parabolic_refine(mag, i, FS, nfft)
    assert f == pytest.approx(i * df), "a non-peak bin has no sub-bin information to offer"
    assert f > 0.0


def test_a_genuine_peak_is_still_refined_below_the_bin_grid():
    """The guard must decline non-peaks, not refinement -- a detector pinned to bin centres would
    satisfy every other test in this file and quietly cost the modal suite its sub-cent accuracy."""
    n, nfft = 2000, 4096
    df = FS / nfft
    f0 = 100.4 * df  # deliberately off-grid: the bin centre is 0.4 bins = 40% of a bin away
    t = np.arange(n) / FS
    sig = np.sin(2.0 * np.pi * f0 * t)

    _, mag, got_nfft = magnitude_spectrum(sig, FS)
    assert got_nfft == nfft
    i = int(np.argmax(mag[1:])) + 1
    assert abs(i * df - f0) > 0.3 * df, "the tone must sit well off a bin centre for this to bite"

    f = _parabolic_refine(mag, i, FS, nfft)
    assert abs(f - f0) < 0.1 * df, "refinement still has to happen at a real peak"


@pytest.mark.parametrize("f_probe", [40.0, 137.0, 400.0, 1200.0, 2500.0, 3800.0])
def test_a_measured_partial_never_leaves_the_window_it_was_asked_about(f_probe):
    """``measure_partials_near`` answers "where is the peak near here"; the answer must be *near
    here*. The record is a short, heavily damped 190 Hz tone -- the shape the viewer renders when
    ``audio_duration`` is small -- so most probes land on a skirt with no peak in the window at all.

    The probe at 137 Hz is the one that used to escape, and it is worth reading closely: its window
    stops just short of the real 190 Hz tone, the argmax pins to the *upper* edge on a rising slope,
    and the refiner extrapolates +7 bins to land on the peak **outside** the window, at 191.5 Hz.
    That answer is positive and finite, so it would survive any amount of sanity checking on its
    sign; it is simply the wrong partial. The negative frequencies above are the loud version of
    this, not a separate bug."""
    n = 1500
    t = np.arange(n) / FS
    sig = np.sin(2.0 * np.pi * 190.0 * t) * np.exp(-t / 0.004)
    _, _, nfft = magnitude_spectrum(sig, FS)
    df = FS / nfft
    search_hz = 0.3 * f_probe

    f = float(measure_partials_near(sig, FS, np.asarray([f_probe]), search_hz=search_hz)[0])

    assert np.isfinite(f)
    assert f > 0.0, "a frequency is positive -- log-based readouts downstream take that on trust"
    assert abs(f - f_probe) <= search_hz + 2.0 * df, "the estimate escaped its own search window"


# -- three claims that arrived from `test_rust_parity_spectrum.py` when unit 11 was deleted -------
#
# That file was the detector's Python half against its Rust half (plan §36), and its central
# finding was that this module has **two axes with different exactness**: the frequency axis comes
# from `+ - * /` alone and is bit-identical on any conforming machine, while the magnitude axis sits
# on NumPy's CPU-dispatched `cos`, pocketfft and `hypot` and is a tolerance port. The comparisons
# died with the Python body (their values are frozen in `tests/analysis_frozen_values.py`). These
# three did not, because none of them is a comparison between two implementations: each is a claim
# about the shipped detector that happens to have been discovered while porting it.
#
# The two guard tests that also lived there -- the recorded -502 Hz witnesses and the sub-bin
# refinement of a genuine peak -- are NOT copied down: they are already the first two tests in this
# file, which is where the guard's own batch put them.


def test_the_integer_fft_length_agrees_with_the_float_spelling_over_every_length():
    """The one transcendental the port refused, and the measurement that says the refusal is free.

    The Python original spelled the FFT length ``int(2 ** np.ceil(np.log2(max(n, 2))))`` -- a
    ``log2`` inside a **discrete** decision, which is the shape to distrust, because a last bit next
    to an integer is a different FFT length rather than a different last digit. The shipped
    implementation uses integer arithmetic, which cannot round at all.

    That substitution is only safe if the two agree over the range anyone can reach, so it is
    measured rather than argued: every length from 1 to 2^20, and each of ``2^k - 1``, ``2^k``,
    ``2^k + 1`` up to 2^31. Kept after the Python body was deleted because the claim is about the
    *substitution*, and the float spelling -- the thing that might have disagreed -- is still
    computable right here.
    """
    n = np.arange(1, 1 << 20, dtype=np.int64)
    float_way = (2 ** np.ceil(np.log2(np.maximum(n, 2)))).astype(np.int64)
    int_way = np.array([1 << int(m - 1).bit_length() for m in np.maximum(n, 2)], dtype=np.int64)
    bad = np.flatnonzero(float_way != int_way)
    assert bad.size == 0, f"the two spellings part company at lengths {n[bad][:5]}"

    for k in range(20, 32):
        for m in (2**k - 1, 2**k, 2**k + 1):
            f = int(2 ** np.ceil(np.log2(max(m, 2))))
            assert f == 1 << (max(m, 2) - 1).bit_length(), f"disagreement at {m}"


def test_the_peak_separation_comparison_sits_on_a_zero_margin_and_the_spelling_is_load_bearing():
    """``detect_peaks`` compares quantities that are *exactly equal*, so the spelling decides.

    Candidate peaks are found on a bin grid, so a pair four bins apart is compared against a
    threshold of exactly four bins: ``|i*val - c*val| >= 4.0*val``, both sides of which are the same
    real number. The margin is **zero** and the verdict is decided entirely by rounding.

    The first draft of this claim asserted that the gap always clears. It does not -- at 100 kHz
    with ``nfft = 16`` it comes out short and the candidate is rejected -- so this is a live
    comparison whose answer changes with the sample rate, not a theoretical hazard that happens to
    be benign. The half that is asserted here is that it stays live.

    The second half is why the ``val`` chain is spelled the way it is rather than tidied to
    ``fs/nfft``. At the rates this project uses -- 8 k, 44.1 k, 48 k, 22.05 k, 96 k -- the two
    spellings agree at every power-of-two size, so a test built from those rates would "prove" the
    tidy form fine and leave the code accidentally correct rather than correct. Searched over random
    rates they differ for about one pair in eight, and that search is asserted to keep finding one.
    """
    verdicts = []
    for k in range(4, 13):
        nfft = 1 << k
        for fs in (8000.0, 44100.0, 48000.0, 22050.0, 96000.0, 100000.0):
            t = np.arange(nfft) / fs
            sig = np.ascontiguousarray(np.sin(2.0 * np.pi * (fs / 64.0) * t))
            freqs, _, got = magnitude_spectrum(sig, fs, 1)
            assert got == nfft
            verdicts.append(np.abs(freqs[4:] - freqs[:-4]) >= 4.0 * (freqs[1] - freqs[0]))
    assert not all(v.all() for v in verdicts), (
        "the separation comparison now clears everywhere searched, so this test no longer "
        "exercises the knife edge it exists for -- widen the sweep before trusting it"
    )

    rng = np.random.default_rng(20260903)
    rates = rng.uniform(1000.0, 96000.0, 2000)
    tidy_differs = sum(
        1
        for fs in rates
        for k in (10, 11, 12, 13)
        if 1.0 / ((1 << k) * (1.0 / fs)) != fs / (1 << k)
    )
    assert tidy_differs > 0, (
        "the two spellings of the bin width now agree everywhere searched, so nothing here "
        "demonstrates why the literal chain is spelled as it is -- widen the search or retire this"
    )


def test_a_window_with_nothing_in_it_comes_back_as_not_a_number():
    """Both ends of the spectrum, where the *clamps* rather than the width empty the window.

    A probe above Nyquist and a probe below the first bin have no window to search, and the answer
    has to be ``nan`` rather than a bin index, a zero or the nearest edge. Every caller in this
    project treats ``nan`` as "no partial here"; a plausible number would be read as a measurement.
    """
    fs, n = 44100.0, 1024
    t = np.arange(n) / fs
    rng = np.random.default_rng(7)
    sig = np.ascontiguousarray(
        np.sin(2.0 * np.pi * 220.0 * t) * np.exp(-3.0 * t) + 1e-6 * rng.standard_normal(n)
    )
    got = measure_partials_near(sig, fs, np.array([30000.0, 0.5, 220.0]), search_hz=10.0)
    assert np.isnan(got[0]), "a probe above Nyquist has no window and must not return a number"
    assert np.isnan(got[1]), "a probe below the first bin has no window either"
    assert not np.isnan(got[2]) and abs(got[2] - 220.0) < 10.0
