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

from physsynth.analysis.spectrum import _parabolic_refine, magnitude_spectrum, measure_partials_near

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
