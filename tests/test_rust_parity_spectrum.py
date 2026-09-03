"""The partial detector, Python against Rust -- and the first port whose two axes disagree.

Every parity file before this one had a single question: does the Rust side reproduce the Python
side's arithmetic, and if not, over what window does the divergence stay under a bar. This module
splits in two, and the split is the batch's finding (``docs/dev/rust-migration-plan.md`` §36).

* **The frequency axis is exact and is asserted exactly.** ``freqs``, ``df``, the search-window
  bounds and ``min_separation_hz`` come from ``+ - * /`` alone, which IEEE-754 pins, so both sides
  agree bit for bit on any machine. That is not a nicety: measured over the suite's real calls, the
  greedy separation test compares quantities that are *exactly equal* -- candidates precisely four
  bins apart against a threshold of precisely four bins -- so its margin is **zero** and the ``>=``
  is decided purely by how the chain was spelled. ``fs/nfft`` is not ``1.0/(nfft*(1.0/fs))``; they
  differ in 20 of 240 (size, rate) pairs. Simplifying it would flip a live comparison.

* **The magnitude axis is a tolerance port and is asserted as one.** Three library kernels sit on
  it -- NumPy's own CPU-dispatched ``cos`` in the Hann window (§22.1), pocketfft, and ``hypot`` --
  and none is reachable by transcription. So the magnitudes are compared against a bar and the
  observed agreement is *reported*, never required. Requiring it would be the §35.2 mistake one
  batch after it was written down: a predicate that is really a claim about which CPU ran CI.

What licenses porting a module whose output is a *decision* under those conditions is that every
decision with a tight margin lives on the exact axis, and every decision on the inexact axis clears
by ten orders of magnitude more than a rounding can move. Those margins were measured before the
port and are re-measured here, so that a future fixture which narrows one fails *this* file rather
than silently returning a different partial.
"""

from __future__ import annotations

import numpy as np
import pytest

import physsynth.analysis.spectrum as S

rs = pytest.importorskip("physsynth_rs")


def _record(n: int, fs: float, f1: float, seed: int) -> np.ndarray:
    """A decaying harmonic series plus a little noise -- the shape every modal test measures.

    The noise matters. A pure sum of sinusoids gives a spectrum with exactly-zero bins whose
    relative disagreement is meaningless, and it makes the runner-up in a search window
    unrealistically weak, which would flatter the margin this file is here to police.
    """
    t = np.arange(n) / fs
    rng = np.random.default_rng(seed)
    sig = sum(
        (1.0 / m) * np.sin(2 * np.pi * f1 * m * t) * np.exp(-3.0 * m * t) for m in range(1, 7)
    )
    return np.ascontiguousarray(sig + 1e-6 * rng.standard_normal(n))


CASES = [
    # (n, fs, f1) -- powers of two and not, one short record, one long, two sample rates.
    (1024, 8000.0, 300.0),
    (4096, 44100.0, 220.0),
    (5000, 44100.0, 220.0),
    (1 << 15, 44100.0, 110.0),
    (777, 22050.0, 500.0),
]
"""Deliberately mixed lengths: 5000 and 777 are not powers of two, so the zero-padding path and the
`next_pow2` decision are exercised rather than assumed away by round numbers."""


def _both(n, fs, f1, seed=20260903):
    sig = _record(n, fs, f1, seed)
    py = S.magnitude_spectrum_py(sig, fs)
    rust = rs.spectrum_magnitude_spectrum(sig, fs, 2)
    return sig, py, rust


def _bin_of(freq: float, fs: float, nfft: int) -> int:
    """Which bin a refined frequency came from -- the parabolic correction is bounded by half a bin.

    This is how the *decision* is compared exactly while the *value* is compared on a tolerance.
    Without it the two would be one assertion, and a tolerance loose enough for the magnitudes
    would also be loose enough to hide a different peak.

    What makes it trustworthy is a property asserted somewhere else, and the coupling is worth
    naming here because it is invisible from this file: recovering the bin by rounding only works
    while ``|delta| <= 1/2``, and that bound is precisely what the refiner's local-max guard exists
    to give. It is asserted directly in the native suite
    (``the_refiner_moves_a_genuine_peak_by_less_than_half_a_bin``) and again below on a swept
    sub-bin offset. Were the bound ever to fail on both sides in the same direction, this helper
    would quietly turn a disagreement about *which bin* into a pass.
    """
    return int(round(freq * nfft / fs))


@pytest.mark.parametrize("n, fs, f1", CASES)
def test_the_fft_length_and_the_frequency_axis_are_bit_identical(n, fs, f1):
    _, (fp, _, nfft_py), (fr, _, nfft_rs) = _both(n, fs, f1)
    assert nfft_py == nfft_rs, "a different FFT length is a different spectrum, not a last bit"
    assert np.array_equal(fp, fr), "the frequency axis is pure +-*/ and must agree exactly"


def test_the_integer_fft_length_agrees_with_the_float_spelling_over_every_length():
    """The one transcendental that was refused, and the measurement that says the refusal is free.

    The original spells the FFT length ``int(2 ** np.ceil(np.log2(max(n, 2))))`` -- a ``log2``
    inside a discrete decision, which is the shape §25 says to distrust, because a last bit next to
    an integer would be a different FFT length rather than a different last digit. The Rust side
    uses integer arithmetic, which cannot round at all.

    That substitution is only safe if the two agree over the range anyone can reach, so it is
    measured rather than argued: every length from 1 to 2^20, and each of ``2^k - 1``, ``2^k``,
    ``2^k + 1`` up to 2^31.
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


@pytest.mark.parametrize("n, fs, f1", CASES)
def test_the_magnitudes_agree_to_about_a_last_bit_and_the_number_is_reported(n, fs, f1, capsys):
    """Reported, not required -- see this module's docstring on the §35.2 mistake."""
    _, (_, mp, _), (_, mr, _) = _both(n, fs, f1)
    scale = mp.max()
    worst = np.abs(mp - mr).max() / scale
    identical = int((mp == mr).sum())
    with capsys.disabled():
        print(
            f"\n  n={n} fs={fs}: worst magnitude diff {worst:.3e} of peak; "
            f"{identical}/{len(mp)} bins bit-identical"
        )
    assert worst < 1e-12, "a divergence this large is not the window and the FFT, it is a bug"


@pytest.mark.parametrize("n, fs, f1", CASES)
def test_measure_partials_near_picks_the_same_bins_and_lands_in_the_same_place(n, fs, f1, capsys):
    """The exact claim is the bin; the value is a tolerance. Two assertions, deliberately."""
    sig, (fp, _, nfft), _ = _both(n, fs, f1)
    expected = np.array([f1 * m for m in range(1, 7)])
    py = S.measure_partials_near_py(sig, fs, expected)
    rust = rs.spectrum_measure_partials_near(sig, fs, np.ascontiguousarray(expected), None)

    assert np.array_equal(np.isnan(py), np.isnan(rust)), "the empty-window verdicts must match"
    ok = ~np.isnan(py)
    assert [_bin_of(f, fs, nfft) for f in py[ok]] == [_bin_of(f, fs, nfft) for f in rust[ok]], (
        "a different bin won a search window -- this is the decision the port must not move"
    )
    df = fp[1] - fp[0]
    worst_bins = float(np.max(np.abs(py[ok] - rust[ok])) / df)
    with capsys.disabled():
        print(f"\n  n={n} fs={fs}: worst refined-frequency diff {worst_bins:.3e} bins")
    assert worst_bins < 1e-6, "the sub-bin correction moved further than a rounding can explain"


@pytest.mark.parametrize("n, fs, f1", CASES)
def test_the_peak_decision_still_has_room_to_spare(n, fs, f1, capsys):
    """The licence, re-measured. If this ever tightens, the port's exact claim expires.

    The bin that wins a search window is decided by comparing magnitudes, which the two sides do
    *not* compute identically. That is safe only while the winner clears the runner-up by far more
    than a rounding can move -- measured at >= 1.4e12 ulps over the dependent suite's 384 real
    calls before the port was written. This asserts the property on these fixtures rather than
    trusting that measurement to stay true as fixtures are added.
    """
    sig, (fp, mp, _), _ = _both(n, fs, f1)
    df = fp[1] - fp[0]
    search = 0.4 * f1
    worst = np.inf
    for m in range(1, 7):
        fe = f1 * m
        lo = max(1, int(np.floor((fe - search) / df)))
        hi = min(len(mp) - 1, int(np.ceil((fe + search) / df)))
        if hi <= lo:
            continue
        window = np.sort(mp[lo : hi + 1])[::-1]
        margin = (window[0] - window[1]) / np.spacing(window[0])
        worst = min(worst, margin)
    with capsys.disabled():
        print(f"\n  n={n} fs={fs}: tightest winner-vs-runner-up margin {worst:.3e} ulps")
    assert worst > 1e6, (
        "a search window's winner is within a million ulps of its runner-up, so which bin wins is "
        "no longer decided by the physics -- the exact-peak claim in spectrum.rs does not hold for "
        "this fixture and the port cannot be trusted on it"
    )


@pytest.mark.parametrize("n, fs, f1", CASES)
def test_detect_peaks_chooses_the_same_peaks(n, fs, f1):
    """The greedy cascade, which is the output that changes as a *set* rather than as a value."""
    sig, (_, _, nfft), _ = _both(n, fs, f1)
    py = S.detect_peaks_py(sig, fs, 5, 50.0)
    rust = rs.spectrum_detect_peaks(sig, fs, 5, 50.0, None)
    assert len(py) == len(rust), f"different number of peaks: {len(py)} vs {len(rust)}"
    assert [_bin_of(f, fs, nfft) for f in py] == [_bin_of(f, fs, nfft) for f in rust]


def test_the_zero_margin_separation_comparison_returns_the_same_verdict_on_both_sides():
    """The knife-edge comparison, and the claim about it is *agreement*, not *outcome*.

    Candidate peaks are found on a bin grid, so a pair four bins apart is compared against a
    threshold of exactly four bins: ``|i*val - c*val| >= 4.0*val``, both sides of which are the same
    real number. The margin is **zero**, so the verdict is decided entirely by rounding.

    The first draft of this test asserted that the gap always clears. It does not. At 100 kHz with
    ``nfft = 16`` it comes out *short*, and the candidate is rejected -- so this is not a
    theoretical hazard that happens to be benign, it is a live comparison whose answer changes with
    the sample rate. What the port must guarantee is therefore not that the comparison passes but
    that **both implementations reach the same verdict**, which they do for the structural reason
    the module header gives: the frequency axis is built from ``+ - * /`` alone and is bit-identical
    between them, so every comparison built on it is too.

    The second half is why the ``val`` chain was transcribed literally rather than tidied to
    ``fs/nfft``. At the rates this project uses -- 8 k, 44.1 k, 48 k, 22.05 k, 96 k -- the two
    spellings agree at every power-of-two size, so a test built from those rates would "prove" the
    tidy form fine and leave the port accidentally correct rather than correct. Searched over
    random rates they differ for about one pair in eight. So the sweep searches, per this repo's
    scar about a hand-picked spelling witness that went red.
    """
    verdicts = []
    for k in range(4, 13):
        nfft = 1 << k
        for fs in (8000.0, 44100.0, 48000.0, 22050.0, 96000.0, 100000.0):
            sig = _record(nfft, fs, fs / 64.0, 11)
            fp, _, n_py = S.magnitude_spectrum_py(sig, fs, 1)
            fr, _, n_rs = rs.spectrum_magnitude_spectrum(sig, fs, 1)
            assert n_py == n_rs == nfft
            assert np.array_equal(fp, fr), f"the axis itself differs at nfft={nfft}, fs={fs}"
            for freqs in (fp, fr):
                verdicts.append(np.abs(freqs[4:] - freqs[:-4]) >= 4.0 * (freqs[1] - freqs[0]))
    for py_v, rs_v in zip(verdicts[0::2], verdicts[1::2], strict=True):
        assert np.array_equal(py_v, rs_v), "the two sides disagree about a zero-margin comparison"
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
        "demonstrates why the literal chain was transcribed -- widen the search or retire this"
    )


def test_the_guard_declines_the_recorded_witnesses_on_both_sides():
    """The two triples that produced -502 Hz, through both refiners.

    Exact equality is fair here and nowhere else in this file: the guard *returns before* touching
    a logarithm, so both sides compute the bin centre ``i * fs / nfft`` and nothing else. If either
    side ever starts refining these, the difference will be enormous rather than subtle.
    """
    witnesses = [
        [0.00054482, 0.00045897, 0.00038359],
        [0.00037953, 0.00033298, 0.00028228],
    ]
    fs, nfft, i = 8000.0, 256, 4
    for triple in witnesses:
        mag = np.zeros(nfft // 2 + 1)
        mag[i - 1 : i + 2] = triple
        py = S._parabolic_refine_py(mag, i, fs, nfft)
        rust = rs.spectrum_parabolic_refine(np.ascontiguousarray(mag), i, fs, nfft)
        assert py == rust == i * fs / nfft, f"the guard parted company: {py} vs {rust}"


def test_a_genuine_peak_is_refined_the_same_way_on_both_sides():
    """The other half of the guard: when it does *not* decline, the parabola must agree too."""
    fs, nfft, i = 8000.0, 256, 40
    for offset in np.linspace(-0.45, 0.45, 19):
        mag = np.exp(-((np.arange(nfft // 2 + 1) - (i + offset)) ** 2))
        py = S._parabolic_refine_py(mag, i, fs, nfft)
        rust = rs.spectrum_parabolic_refine(np.ascontiguousarray(mag), i, fs, nfft)
        assert abs(py - rust) < 1e-9 * fs / nfft, f"at offset {offset}: {py} vs {rust}"
        assert abs(py * nfft / fs - i) <= 0.5 + 1e-12, "the half-bin bound is the guard's promise"


def test_the_empty_window_verdicts_agree():
    """Both ends of the spectrum, where the clamps rather than the width empty the window."""
    fs, n = 44100.0, 1024
    sig = _record(n, fs, 220.0, 7)
    expected = np.array([30000.0, 0.5, 220.0])
    py = S.measure_partials_near_py(sig, fs, expected, 10.0)
    rust = rs.spectrum_measure_partials_near(sig, fs, np.ascontiguousarray(expected), 10.0)
    assert np.array_equal(np.isnan(py), np.isnan(rust))
    assert np.isnan(py[0]) and np.isnan(py[1]) and not np.isnan(py[2])
