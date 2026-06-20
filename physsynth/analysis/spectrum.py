"""Spectral partial detection for validating modal frequencies.

A Hann-windowed FFT plus parabolic interpolation on the log-magnitude spectrum recovers partial
frequencies to well under a cent for long, stationary records — accurate enough to check the FDTD
output against the analytic harmonic series.

Pure NumPy. No plotting (this is analysis, not viz).
"""

from __future__ import annotations

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
    """Sub-bin frequency (Hz) of the peak at bin ``i`` via log-magnitude parabolic interpolation."""
    if i <= 0 or i >= len(mag) - 1:
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
