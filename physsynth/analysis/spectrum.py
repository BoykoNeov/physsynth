"""Spectral partial detection for validating modal frequencies.

A Hann-windowed FFT plus parabolic interpolation on the log-magnitude spectrum recovers partial
frequencies to well under a cent for long, stationary records — accurate enough to check the FDTD
output against the analytic harmonic series.

Pure NumPy. No plotting (this is analysis, not viz).
"""

from __future__ import annotations

import os

import numpy as np
from numpy.typing import NDArray

__all__ = ["magnitude_spectrum", "measure_partials_near", "detect_peaks"]


def magnitude_spectrum(
    signal: NDArray[np.float64],
    fs: float,
    zero_pad_factor: int = 2,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Return ``(freqs, magnitude, nfft)`` of the DC-removed, Hann-windowed signal.

    Zero-padding (default 2x, rounded up to a power of two) densifies the bin grid, which improves
    the parabolic-interpolation estimate; it does not add real resolution.
    """
    sig = np.asarray(signal, dtype=float)
    sig = sig - np.mean(sig)
    win = np.hanning(len(sig))
    sigw = sig * win
    nfft = int(2 ** np.ceil(np.log2(max(len(sigw) * zero_pad_factor, 2))))
    spec = np.fft.rfft(sigw, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    return freqs, np.abs(spec), nfft


def _parabolic_refine(mag: NDArray[np.float64], i: int, fs: float, nfft: int) -> float:
    """Sub-bin frequency (Hz) of the peak at bin ``i`` via log-magnitude parabolic interpolation.

    Refines only a bin that **is** a local maximum. :func:`measure_partials_near` takes its argmax
    *inside a search window*, and a window whose magnitudes fall monotonically puts that argmax on
    the window's own edge — a bin with no peak under it. Fitting a parabola through three
    near-collinear log-magnitudes there makes the curvature ``a - 2b + c`` vanish and ``delta``
    diverge: measured at -22.1 bins from bin 4, i.e. a "refined" frequency of **-502 Hz**, which
    downstream became a NaN out of ``modal.cents``. The quieter form of the same fault extrapolates
    to a *positive* frequency outside the window — the wrong partial, checked by nothing.

    Note which guard this has to be. Both real witnesses had **negative** curvature, so testing the
    sign of ``denom`` catches neither (``a=3, b=2, c=0`` is concave, decreasing, and still moves the
    estimate 1.5 bins). The property that makes the interpolation meaningful, and that bounds
    ``|delta| <= 1/2``, is that ``i`` is a genuine local max — so that is what is tested, and a
    non-peak bin gets its honest bin centre back. See ``tests/test_spectrum_detector.py``.
    """
    if i <= 0 or i >= len(mag) - 1:
        return i * fs / nfft
    if mag[i] < mag[i - 1] or mag[i] < mag[i + 1]:
        return i * fs / nfft
    a = np.log(mag[i - 1] + 1e-300)
    b = np.log(mag[i] + 1e-300)
    c = np.log(mag[i + 1] + 1e-300)
    denom = a - 2.0 * b + c
    delta = 0.5 * (a - c) / denom if denom != 0.0 else 0.0
    return (i + delta) * fs / nfft


def measure_partials_near(
    signal: NDArray[np.float64],
    fs: float,
    expected: NDArray[np.float64],
    search_hz: float | None = None,
) -> NDArray[np.float64]:
    """Measure the partial frequencies nearest each value in ``expected``.

    For every expected frequency, the magnitude peak within ``±search_hz`` is located and refined by
    parabolic interpolation. ``search_hz`` defaults to 40% of the lowest expected frequency (i.e.
    40% of the harmonic spacing), keeping each search window clear of neighbouring partials.
    Returns an array the same length as ``expected`` (``NaN`` where no bin falls in the window).
    """
    expected = np.asarray(expected, dtype=float)
    freqs, mag, nfft = magnitude_spectrum(signal, fs)
    df = freqs[1] - freqs[0]
    if search_hz is None:
        search_hz = 0.4 * float(expected.min())

    out = np.full(expected.shape, np.nan)
    for j, fe in enumerate(expected):
        lo = max(1, int(np.floor((fe - search_hz) / df)))
        hi = min(len(mag) - 1, int(np.ceil((fe + search_hz) / df)))
        if hi <= lo:
            continue
        i = lo + int(np.argmax(mag[lo : hi + 1]))
        out[j] = _parabolic_refine(mag, i, fs, nfft)
    return out


def detect_peaks(
    signal: NDArray[np.float64],
    fs: float,
    n_peaks: int,
    f_min: float = 1.0,
    min_separation_hz: float | None = None,
) -> NDArray[np.float64]:
    """Blindly detect the ``n_peaks`` strongest spectral peaks above ``f_min`` (ascending Hz).

    Unlike :func:`measure_partials_near` this uses no prior knowledge of where partials should be —
    useful as an independent cross-check that the detector finds the harmonic series on its own.
    ``min_separation_hz`` greedily suppresses weaker peaks closer than that to an already-selected
    stronger one, which rejects window sidelobes around a strong tone (default: 4 raw FFT bins, ~one
    Hann main-lobe half-width).
    """
    freqs, mag, nfft = magnitude_spectrum(signal, fs)
    df = freqs[1] - freqs[0]
    if min_separation_hz is None:
        min_separation_hz = 4.0 * df
    # Local maxima above the noise floor.
    interior = np.arange(1, len(mag) - 1)
    is_peak = (mag[interior] > mag[interior - 1]) & (mag[interior] > mag[interior + 1])
    cand = interior[is_peak]
    cand = cand[freqs[cand] >= f_min]
    if len(cand) == 0:
        return np.array([])
    # Greedy strongest-first selection with a minimum frequency separation.
    cand = cand[np.argsort(mag[cand])[::-1]]
    chosen: list[int] = []
    for i in cand:
        f = freqs[i]
        if all(abs(f - freqs[c]) >= min_separation_hz for c in chosen):
            chosen.append(int(i))
        if len(chosen) >= n_peaks:
            break
    refined = np.array([_parabolic_refine(mag, int(i), fs, nfft) for i in chosen])
    return np.sort(refined)


# --- the Rust swap (docs/dev/rust-migration-plan.md, Phase 7) ------------------------------------
#
# **This module reads a DIFFERENT environment variable from every other swap in the project, and
# that is the point rather than an accident.** `PHYSSYNTH_RS=1` replaces the models in
# `physsynth/core` and runs the existing suite against them; what makes that run worth anything is
# that the *instrument* does not move -- a Rust string is measured by the same Python detector,
# against the same analytic oracle, that the Python string was. Put this module behind the same
# flag and both the model and the ruler become Rust at once, so a shared misreading would cancel
# and the acceptance gate would quietly stop asserting what it says it asserts. Plan Sec 7
# scheduled `analysis/` late for exactly that reason and Sec 35.3 re-planned the order without
# re-taking the argument; this is where it is re-taken.
#
# So:  PHYSSYNTH_RS=1                         Rust models, Python instrument -- the acceptance run.
#      PHYSSYNTH_RS=1 + PHYSSYNTH_RS_ANALYSIS both Rust -- exercises this port through its real
#                                             clients, which is the extra CI step.
#      PHYSSYNTH_RS_ANALYSIS=1 alone          Python models, Rust instrument -- the sharpest test
#                                             of this module alone, and what the parity file uses.
#
# Nothing in `physsynth/core` imports `physsynth/analysis`, checked rather than assumed, so the two
# flags are genuinely independent and not merely written that way.
#
# **What is and is not claimed about agreement.** The frequency axis is exact: `freqs`, `df`, the
# window bounds and `min_separation_hz` are built from `+ - * /`, which IEEE-754 pins, so both sides
# compute them bit for bit. That is load-bearing rather than tidy -- the greedy separation test in
# `detect_peaks` compares a gap of exactly four bins against a threshold of exactly four bins, so
# its margin is ZERO and its answer is decided by rounding alone. It does not always clear (at
# 100 kHz with nfft=16 it does not), so the claim is that both sides agree on whatever it decides,
# which holds only because the `1.0/(nfft*(1.0/fs))` chain is transcribed and not tidied.
#
# The magnitudes are NOT exact and no attempt is made to pretend
# otherwise -- three library kernels sit on that path (NumPy's own CPU-dispatched `cos` inside the
# Hann window, pocketfft, and `hypot`), none of which a transcription reaches. The port is safe
# because every decision with a tight margin lives on the exact axis and every decision on the
# inexact one clears by ten orders of magnitude more than a rounding can move it; the six margins
# were measured over the dependent suite's real calls before this was written, and they are in plan
# Sec 36.2 and in the Rust module's header.
magnitude_spectrum_py = magnitude_spectrum
_parabolic_refine_py = _parabolic_refine
measure_partials_near_py = measure_partials_near
detect_peaks_py = detect_peaks

_USE_RUST = os.environ.get("PHYSSYNTH_RS_ANALYSIS", "").strip() not in ("", "0", "false", "False")

if _USE_RUST:  # pragma: no cover - exercised by the dedicated CI step, not the default gate
    import physsynth_rs as _rs

    def _asarray(a: object) -> NDArray[np.float64]:
        """Whatever the caller passed, as the contiguous float64 array the binding requires."""
        return np.ascontiguousarray(np.asarray(a, dtype=np.float64))

    def magnitude_spectrum(signal, fs, zero_pad_factor=2):  # type: ignore[misc]  # noqa: F811
        return _rs.spectrum_magnitude_spectrum(_asarray(signal), fs, zero_pad_factor)

    def _parabolic_refine(mag, i, fs, nfft):  # type: ignore[misc]  # noqa: F811
        return _rs.spectrum_parabolic_refine(_asarray(mag), int(i), fs, nfft)

    def measure_partials_near(signal, fs, expected, search_hz=None):  # type: ignore[misc]  # noqa: F811,E501
        return _rs.spectrum_measure_partials_near(
            _asarray(signal), fs, _asarray(expected), search_hz
        )

    def detect_peaks(signal, fs, n_peaks, f_min=1.0, min_separation_hz=None):  # type: ignore[misc]  # noqa: F811,E501
        return _rs.spectrum_detect_peaks(
            _asarray(signal), fs, int(n_peaks), f_min, min_separation_hz
        )
