//! The detector's own bars, asserted without an interpreter in the room.
//!
//! These are not parity tests — `tests/test_rust_parity_spectrum.py` compares this code against
//! NumPy. These assert that the code is *right on its own terms*, which is the half a parity test
//! cannot reach: two implementations can agree beautifully and both be wrong, and a hand-written
//! FFT is exactly the kind of thing that agrees with nothing until it is checked against the
//! definition.
//!
//! So the FFT is checked against a directly evaluated DFT and against Parseval; the detector is
//! checked against signals whose partials are known by construction; and the guard that this whole
//! module exists to carry is checked on the two recorded witnesses that produced the original bug.

use physsynth_analysis::spectrum::{
    detect_peaks, hann, magnitude_spectrum, measure_partials_near, next_pow2, parabolic_refine,
    rfftfreq,
};
use std::f64::consts::PI;

/// A record of `n` samples of a sum of sinusoids at `fs`, so the answer is known before measuring.
fn tones(n: usize, fs: f64, partials: &[(f64, f64)]) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            partials
                .iter()
                .map(|&(f, a)| a * (2.0 * PI * f * t).sin())
                .sum::<f64>()
        })
        .collect()
}

/// The forward DFT, straight from the definition. O(n^2) and correct by inspection.
fn dft(x: &[f64]) -> Vec<(f64, f64)> {
    let n = x.len();
    (0..n)
        .map(|k| {
            let mut re = 0.0;
            let mut im = 0.0;
            for (j, &xj) in x.iter().enumerate() {
                let ang = -2.0 * PI * (k * j) as f64 / n as f64;
                re += xj * ang.cos();
                im += xj * ang.sin();
            }
            (re, im)
        })
        .collect()
}

#[test]
fn the_fft_agrees_with_the_definition() {
    // Through `magnitude_spectrum` with the window and DC removal reproduced by hand, since the
    // FFT itself is private -- the point is that what the module computes IS a DFT.
    let fs = 8000.0;
    for &n in &[16usize, 64, 256] {
        let sig = tones(n, fs, &[(500.0, 1.0), (1300.0, 0.3), (2100.0, 0.07)]);
        let out = magnitude_spectrum(&sig, fs, 1);
        assert_eq!(out.nfft, n, "a power-of-two record needs no padding");

        let mean = sig.iter().sum::<f64>() / n as f64;
        let win = hann(n);
        let windowed: Vec<f64> = (0..n).map(|i| (sig[i] - mean) * win[i]).collect();
        let reference = dft(&windowed);

        // O(eps * log n) against a reduction that is itself O(eps * n); scaled by the largest
        // magnitude, because an absolute bar on a near-null bin is a claim about cancellation.
        let scale = out.mag.iter().cloned().fold(0.0, f64::max);
        for (k, &(re, im)) in reference.iter().enumerate().take(n / 2 + 1) {
            let want = re.hypot(im);
            let got = out.mag[k];
            assert!(
                (got - want).abs() <= 1e-12 * scale,
                "bin {k} at n={n}: fft {got:e} vs dft {want:e}"
            );
        }
    }
}

#[test]
fn parseval_holds_across_the_transform() {
    // The FFT's global check, and the one that catches a wrong normalisation or a dropped
    // butterfly that a per-bin comparison at one size might tolerate.
    let fs = 44100.0;
    let n = 1024;
    let sig = tones(n, fs, &[(440.0, 1.0), (880.0, 0.5), (1320.0, 0.25)]);
    let out = magnitude_spectrum(&sig, fs, 1);

    let mean = sig.iter().sum::<f64>() / n as f64;
    let win = hann(n);
    let time: f64 = (0..n).map(|i| ((sig[i] - mean) * win[i]).powi(2)).sum();

    // One-sided sum: interior bins carry both halves of a conjugate pair, DC and Nyquist do not.
    let mut freq = out.mag[0] * out.mag[0] + out.mag[n / 2] * out.mag[n / 2];
    for k in 1..n / 2 {
        freq += 2.0 * out.mag[k] * out.mag[k];
    }
    let rel = (freq / n as f64 - time).abs() / time;
    assert!(rel < 1e-13, "Parseval violated by {rel:e}");
}

#[test]
fn the_window_is_numpys_spelling_and_its_edge_cases() {
    assert!(hann(0).is_empty());
    assert_eq!(hann(1), vec![1.0]);
    let w = hann(8);
    assert_eq!(w.len(), 8);
    // Symmetric, zero at both ends, unity in the middle for odd lengths.
    for i in 0..4 {
        assert_eq!(w[i], w[7 - i], "the Hann window is symmetric");
    }
    assert!(w[0].abs() < 1e-16, "the endpoints vanish, got {}", w[0]);
    let w9 = hann(9);
    assert!((w9[4] - 1.0).abs() < 1e-15, "the centre is unity");
}

#[test]
fn the_fft_length_is_the_ceiling_power_of_two() {
    // The one place a transcendental was refused in favour of integers. The equivalence with
    // `2 ** ceil(log2(n))` over 1..2^20 is asserted below, in
    // `the_integer_fft_length_agrees_with_the_float_spelling_over_every_length`.
    assert_eq!(next_pow2(0), 2);
    assert_eq!(next_pow2(1), 2);
    assert_eq!(next_pow2(2), 2, "a power of two is its own ceiling");
    assert_eq!(next_pow2(3), 4);
    assert_eq!(next_pow2(1023), 1024);
    assert_eq!(next_pow2(1024), 1024);
    assert_eq!(next_pow2(1025), 2048);
}

#[test]
fn the_frequency_axis_is_built_by_multiplication_not_accumulation() {
    // The exact axis the header's zero-margin comparison lives on. Accumulating `f += df` would
    // drift and is the obvious way to write this wrongly.
    let fs = 44100.0;
    let n = 1024;
    let f = rfftfreq(n, 1.0 / fs);
    assert_eq!(f.len(), n / 2 + 1);
    assert_eq!(f[0], 0.0);
    let val = 1.0 / (n as f64 * (1.0 / fs));
    for (i, &fi) in f.iter().enumerate() {
        assert_eq!(fi, i as f64 * val, "bin {i} is not i*val exactly");
    }

    // And the comparison itself, which has zero margin: a gap of exactly four bins against a
    // threshold of exactly four bins. At THIS rate it clears everywhere -- at 100 kHz with
    // nfft = 16 it does not (`the_peak_separation_comparison_sits_on_a_zero_margin_and_stays_live`
    // below records where).
    let df = f[1] - f[0];
    let sep = 4.0 * df;
    for c in 1..f.len() - 5 {
        assert!(
            (f[c + 4] - f[c]).abs() >= sep,
            "the zero-margin separation test flipped at bin {c}"
        );
    }
}

#[test]
fn the_refiner_declines_a_bin_that_is_not_a_peak() {
    // The two magnitude triples that produced the -502 Hz report, carried over verbatim from
    // `tests/test_spectrum_detector.py` (retired in retirement plan §43). Both are CONCAVE, which
    // is why a sign-of-curvature guard catches neither and the guard has to be about
    // local-maximality.
    //
    // Each comes with the correction the unguarded formula used to hand it, recomputed below so the
    // claim is not hearsay (taken from `plate:guitar` at Lx = 0.15 / 0.30, Ly = 0.80, N = 16 / 24,
    // rendered for 0.01 s: a 256-bin FFT whose +-0.3 f1 window spans six bins with no peak inside).
    let witnesses: [([f64; 3], f64); 2] = [
        ([0.00054482, 0.00045897, 0.00038359], -22.096),
        ([0.00037953, 0.00033298, 0.00028228], -4.311),
    ];
    let (fs, nfft, i) = (8000.0, 256usize, 4usize);
    for (w, old_delta) in witnesses {
        let mut mag = vec![0.0; nfft / 2 + 1];
        mag[i - 1..i + 2].copy_from_slice(&w);

        let (a, b, c) = (w[0].ln(), w[1].ln(), w[2].ln());
        assert!(a - 2.0 * b + c < 0.0, "the witness must be concave");
        let unguarded = 0.5 * (a - c) / (a - 2.0 * b + c);
        assert!(
            (unguarded - old_delta).abs() < 1e-3,
            "the unguarded correction is {unguarded}, recorded {old_delta}"
        );

        let f = parabolic_refine(&mag, i, fs, nfft);
        assert_eq!(
            f,
            i as f64 * fs / nfft as f64,
            "a non-peak keeps its bin centre"
        );
        assert!(f > 0.0, "and is certainly not negative");
    }
}

#[test]
fn the_refiner_moves_a_genuine_peak_by_less_than_half_a_bin() {
    // The bound the guard exists to guarantee. Asserted over a swept sub-bin offset rather than at
    // one point, because |delta| <= 1/2 is the property, not a value.
    let (fs, nfft) = (8000.0, 256usize);
    for step in 0..20 {
        let d = -0.5 + 0.05 * step as f64;
        let i = 40usize;
        // A concave-in-log triple with its apex at offset d.
        let mag: Vec<f64> = (0..nfft / 2 + 1)
            .map(|k| (-((k as f64 - (i as f64 + d)).powi(2))).exp())
            .collect();
        let f = parabolic_refine(&mag, i, fs, nfft);
        let bins = f * nfft as f64 / fs - i as f64;
        assert!(
            bins.abs() <= 0.5 + 1e-12,
            "refined by {bins} bins at offset {d}, past the half-bin bound"
        );
    }
}

#[test]
fn a_harmonic_series_is_recovered_to_well_under_a_cent() {
    // The bar the whole module is for: a long stationary record's partials, measured against the
    // frequencies they were synthesised at.
    let fs = 44100.0;
    let f1 = 220.0;
    let expected: Vec<f64> = (1..=5).map(|m| f1 * m as f64).collect();
    let partials: Vec<(f64, f64)> = expected
        .iter()
        .enumerate()
        .map(|(m, &f)| (f, 1.0 / (m as f64 + 1.0)))
        .collect();
    let sig = tones(1 << 15, fs, &partials);

    let found = measure_partials_near(&sig, fs, &expected, None);
    for (&want, &got) in expected.iter().zip(found.iter()) {
        let cents = 1200.0 * (got / want).log2();
        assert!(
            cents.abs() < 0.5,
            "{want} Hz measured at {got} Hz ({cents} cents)"
        );
    }
}

#[test]
fn the_blind_detector_finds_the_same_series_without_being_told() {
    let fs = 44100.0;
    let f1 = 330.0;
    let partials: Vec<(f64, f64)> = (1..=4).map(|m| (f1 * m as f64, 1.0 / m as f64)).collect();
    let sig = tones(1 << 15, fs, &partials);

    let found = detect_peaks(&sig, fs, 4, 50.0, None);
    assert_eq!(found.len(), 4, "four tones, four peaks");
    for (m, &got) in found.iter().enumerate() {
        let want = f1 * (m + 1) as f64;
        let cents = 1200.0 * (got / want).log2();
        assert!(
            cents.abs() < 1.0,
            "partial {m}: {want} Hz measured at {got} Hz"
        );
    }
    assert!(found.windows(2).all(|w| w[0] < w[1]), "ascending");
}

#[test]
fn a_window_off_the_end_of_the_spectrum_reports_nan_rather_than_a_wrong_answer() {
    // NaN is the original's answer here and it is the right one: a caller comparing to an oracle
    // fails loudly, where a clamped bin centre would pass quietly at the wrong frequency.
    //
    // Note *which* windows are actually empty, because the obvious guess is wrong: a window
    // narrower than one bin is NOT empty. `lo` floors and `hi` ceils, so any interval that does
    // not straddle nothing still spans the two bins either side of it -- a 2e-6 Hz window at
    // 440 Hz measures the partial perfectly well. Emptiness comes from the CLAMPS instead: an
    // expected partial above Nyquist has `lo` past the last bin, and one below the first has both
    // ends pinned to bin 1. Both are reachable from real physics -- a high partial off the top of
    // the grid's range, and a mode near DC.
    //
    // The third probe is the control, carried from `tests/test_spectrum_detector.py` (retirement
    // plan §43): in the same call, on the same record, a probe that DOES have a window must come
    // back as a measurement -- without it a detector answering NaN to everything passes. The
    // record is that file's: a decaying 220 Hz tone under 1e-6 of structureless noise.
    let fs = 44100.0;
    let n = 1024;
    let noise = signal(n);
    let sig: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            (2.0 * PI * 220.0 * t).sin() * (-3.0 * t).exp() + 1e-6 * noise[i]
        })
        .collect();
    let found = measure_partials_near(&sig, fs, &[30000.0, 0.5, 220.0], Some(10.0));
    assert!(
        !found[2].is_nan() && (found[2] - 220.0).abs() < 10.0,
        "the control probe at 220 Hz must be measured, got {}",
        found[2]
    );
    assert!(
        found[0].is_nan(),
        "above Nyquist: expected NaN, got {}",
        found[0]
    );
    assert!(
        found[1].is_nan(),
        "below the first bin: expected NaN, got {}",
        found[1]
    );
}

#[test]
fn the_separation_rule_suppresses_a_sidelobe_next_to_a_strong_tone() {
    // What `min_separation_hz` is for: one loud partial's window sidelobes are local maxima and
    // would otherwise fill the answer.
    let fs = 44100.0;
    let sig = tones(1 << 14, fs, &[(1000.0, 1.0)]);
    let found = detect_peaks(&sig, fs, 3, 50.0, Some(500.0));
    for pair in found.windows(2) {
        assert!(
            pair[1] - pair[0] >= 500.0,
            "peaks at {} and {} are closer than the separation asked for",
            pair[0],
            pair[1]
        );
    }
}

// -- rfft at the record's own length (the viewer's brightness and band panels)

/// A directly evaluated one-sided DFT, in extended care: the angle is reduced exactly first.
fn direct_rfft(x: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    (0..n / 2 + 1)
        .map(|k| {
            let (mut re, mut im) = (0.0, 0.0);
            for (j, &xj) in x.iter().enumerate() {
                let r = ((j * k) % n) as f64;
                let (s, c) = (-2.0 * std::f64::consts::PI * r / n as f64).sin_cos();
                re += xj * c;
                im += xj * s;
            }
            (re, im)
        })
        .unzip()
}

/// A deterministic, structureless test signal (no RNG crate: a fixed quadratic-residue hash).
fn signal(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let h = ((i as u64 * 2_654_435_761 + 97) % 1_000_003) as f64 / 1_000_003.0;
            (h - 0.5) + 0.3 * (0.37 * i as f64).sin()
        })
        .collect()
}

#[test]
fn rfft_matches_a_direct_dft_at_prime_composite_and_power_of_two_lengths() {
    for n in [
        1, 2, 3, 4, 5, 7, 8, 12, 17, 31, 32, 60, 97, 100, 128, 255, 256, 257, 1000, 1009,
    ] {
        let x = signal(n);
        let (re, im) = physsynth_analysis::spectrum::rfft(&x);
        let (dr, di) = direct_rfft(&x);
        assert_eq!(re.len(), n / 2 + 1, "n = {n}");
        let scale: f64 = x.iter().map(|v| v.abs()).sum::<f64>().max(1e-300);
        for k in 0..re.len() {
            let err = (re[k] - dr[k]).hypot(im[k] - di[k]) / scale;
            assert!(err < 1e-13, "n = {n}, bin {k}: relative error {err:e}");
        }
    }
}

#[test]
fn rfft_satisfies_parseval_and_real_input_edge_bins() {
    for n in [99, 100, 4097, 44_101] {
        let x = signal(n);
        let (re, im) = physsynth_analysis::spectrum::rfft(&x);
        // DC and (for even n) Nyquist bins of a real signal are real.
        assert!(im[0].abs() <= 1e-9 * re[0].abs().max(1.0), "n = {n}");
        if n % 2 == 0 {
            assert!(im[n / 2].abs() <= 1e-9, "n = {n}");
        }
        // Parseval, one-sided: sum x^2 = (|X0|^2 + 2 sum |Xk|^2 [+ |X_{n/2}|^2]) / n.
        let time: f64 = x.iter().map(|v| v * v).sum();
        let mut freq = re[0] * re[0];
        for k in 1..re.len() {
            let p = re[k] * re[k] + im[k] * im[k];
            freq += if n % 2 == 0 && k == n / 2 { p } else { 2.0 * p };
        }
        freq /= n as f64;
        assert!(
            (time - freq).abs() <= 1e-11 * time,
            "n = {n}: {time} vs {freq}"
        );
    }
}

#[test]
fn rfft_finds_a_pure_tone_in_its_own_bin_at_an_awkward_length() {
    // A tone exactly on bin 7 of a length-1001 record: all the energy in bin 7, none elsewhere.
    let n = 1001;
    let x: Vec<f64> = (0..n)
        .map(|i| (2.0 * std::f64::consts::PI * 7.0 * i as f64 / n as f64).cos())
        .collect();
    let mag = physsynth_analysis::spectrum::rfft_mag(&x);
    assert!((mag[7] - n as f64 / 2.0).abs() < 1e-9 * n as f64);
    for (k, &m) in mag.iter().enumerate() {
        if k != 7 {
            assert!(m < 1e-9 * n as f64, "bin {k}: {m}");
        }
    }
}

// -- carried from `tests/test_spectrum_detector.py` (retirement plan §43) ------------------------
//
// That file tested the detector directly -- the one measurement primitive every modal bar in the
// project leans on, and which none of them tests: they all assert `|found - oracle| < tol`, which a
// detector that is quietly wrong somewhere the models never look satisfies just as well. These are
// its claims that the bars above did not already make.

#[test]
fn a_genuine_peak_off_the_bin_grid_is_refined_to_a_tenth_of_a_bin() {
    // The guard must decline non-peaks, not refinement. The half-bin bound above is satisfied by a
    // refiner pinned to bin centres (delta = 0 is inside it), which would quietly cost every modal
    // bar its sub-cent accuracy; this one is not. A real Hann spectrum of a tone 0.4 bins off-grid.
    let fs = 8000.0;
    let (n, nfft) = (2000usize, 4096usize);
    let df = fs / nfft as f64;
    let f0 = 100.4 * df;
    let sig: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * f0 * (i as f64 / fs)).sin())
        .collect();

    let s = magnitude_spectrum(&sig, fs, 2);
    assert_eq!(s.nfft, nfft);
    // `np.argmax(mag[1:]) + 1`: the first maximum above DC.
    let mut i = 1usize;
    for j in 2..s.mag.len() {
        if s.mag[j] > s.mag[i] {
            i = j;
        }
    }
    assert!(
        (i as f64 * df - f0).abs() > 0.3 * df,
        "the tone must sit well off a bin centre for this to bite"
    );
    // Measured 0.0023 bins off.
    let f = parabolic_refine(&s.mag, i, fs, nfft);
    assert!(
        (f - f0).abs() < 0.1 * df,
        "a real peak at {f0} Hz refined to {f} Hz, {} bins off",
        (f - f0) / df
    );
}

#[test]
fn a_measured_partial_never_leaves_the_window_it_was_asked_about() {
    // `measure_partials_near` answers "where is the peak near here", and the answer must be near
    // here. The record is a short, heavily damped 190 Hz tone -- what the viewer renders when the
    // audio is short -- so most probes land on a skirt with no peak in their window at all.
    //
    // The probe at 137 Hz is the one that used to escape: its window stops just short of the real
    // tone, the argmax pins to the window's upper edge on a rising slope, and the unguarded
    // refiner extrapolated +7 bins to 191.5 Hz -- positive, finite, and the wrong partial. It is
    // also the tightest case now: 2.32 Hz of slack, 1.2 bins (measured on the Python run).
    let fs = 8000.0;
    let n = 1500usize;
    let sig: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            (2.0 * PI * 190.0 * t).sin() * (-t / 0.004).exp()
        })
        .collect();
    let nfft = magnitude_spectrum(&sig, fs, 2).nfft;
    let df = fs / nfft as f64;

    for f_probe in [40.0, 137.0, 400.0, 1200.0, 2500.0, 3800.0] {
        let search_hz = 0.3 * f_probe;
        let f = measure_partials_near(&sig, fs, &[f_probe], Some(search_hz))[0];
        assert!(f.is_finite(), "probe {f_probe}: {f}");
        assert!(f > 0.0, "probe {f_probe}: a frequency is positive, got {f}");
        assert!(
            (f - f_probe).abs() <= search_hz + 2.0 * df,
            "probe {f_probe}: the estimate {f} escaped its own search window"
        );
    }
}

#[test]
fn the_integer_fft_length_agrees_with_the_float_spelling_over_every_length() {
    // The one transcendental the port refused, and the measurement that says the refusal is free.
    // The original spelled the FFT length `int(2 ** np.ceil(np.log2(max(n, 2))))` -- a `log2`
    // inside a DISCRETE decision, where a last bit next to an integer is a different FFT length
    // rather than a different last digit. `next_pow2` is integer arithmetic and cannot round.
    //
    // Every length 1 .. 2^20, and each of 2^k - 1, 2^k, 2^k + 1 up to 2^31. The Python test this
    // carries compared two Python spellings with each other; this one is aimed at the shipped
    // `next_pow2`, and its float referee is the platform's `log2` rather than NumPy's. The deciding
    // inputs are the exact powers of two, where any conforming `log2` returns an exact integer.
    let float_way = |n: usize| -> usize { 2f64.powf((n.max(2) as f64).log2().ceil()) as usize };
    for n in 1..(1usize << 20) {
        assert_eq!(
            next_pow2(n),
            float_way(n),
            "the two spellings part company at {n}"
        );
    }
    for k in 20..32 {
        for m in [(1usize << k) - 1, 1 << k, (1 << k) + 1] {
            assert_eq!(next_pow2(m), float_way(m), "disagreement at {m}");
        }
    }
}

/// A structureless uniform deviate in [0, 1) for draw `i` (SplitMix64's finaliser), standing in for
/// the Python's seeded PCG64: the claim it feeds is "some rate exists", not a value.
fn uniform(i: u64) -> f64 {
    let mut z = i.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

#[test]
fn the_peak_separation_comparison_sits_on_a_zero_margin_and_stays_live() {
    // `detect_peaks` keeps a candidate only if it is at least `min_sep` from every kept one, and on
    // a bin grid a pair four bins apart meets a threshold of exactly four bins: both sides are the
    // same real number, the margin is ZERO, and the verdict is decided by rounding. The axis is
    // built from `+ - * /` alone (no transcendental), so which pairs clear is the same on every
    // conforming machine -- recorded here exactly as the Python run found it: at the project's
    // rates every gap clears, and at 100 kHz it falls short at 1, 5, 9, ... 33 bins, one size more
    // for every doubling of nfft.
    let mut short = Vec::new();
    for k in 4..13 {
        let nfft = 1usize << k;
        for fs in [8000.0, 44100.0, 48000.0, 22050.0, 96000.0, 100000.0] {
            let sig: Vec<f64> = (0..nfft)
                .map(|i| (2.0 * PI * (fs / 64.0) * (i as f64 / fs)).sin())
                .collect();
            let s = magnitude_spectrum(&sig, fs, 1);
            assert_eq!(s.nfft, nfft);
            let f = &s.freqs;
            let sep = 4.0 * (f[1] - f[0]);
            let misses = (4..f.len())
                .filter(|&j| (f[j] - f[j - 4]).abs() < sep)
                .count();
            if misses > 0 {
                short.push((nfft, fs, misses));
            }
        }
    }
    let recorded: Vec<(usize, f64, usize)> = (4..13)
        .map(|k| (1usize << k, 100000.0, 1 + 4 * (k - 4)))
        .collect();
    assert_eq!(
        short, recorded,
        "the zero-margin comparison's verdicts moved; if it now clears everywhere this test no \
         longer exercises the knife edge it exists for"
    );

    // Why the bin width is spelled `1 / (n (1 / fs))` and not tidied to `fs / n`: at the rates
    // above the two agree at every power-of-two size, so a test built from them would "prove" the
    // tidy form fine. Over arbitrary rates they differ for about one pair in eight, and the search
    // must keep finding one.
    let mut tidy_differs = 0usize;
    for i in 0..2000u64 {
        let fs = 1000.0 + 95000.0 * uniform(i);
        for k in 10..14 {
            let n = (1u64 << k) as f64;
            if 1.0 / (n * (1.0 / fs)) != fs / n {
                tidy_differs += 1;
            }
        }
    }
    assert!(
        tidy_differs > 0,
        "the two spellings of the bin width agree everywhere searched"
    );
}

#[test]
fn the_axis_is_spelled_as_the_chain_at_a_length_where_the_spellings_differ() {
    // Added in §43 (the human's call). `rfftfreq` claims to transcribe `1 / (n d)` exactly, and
    // the bar above checks `f[i] == i val` with `val` spelled that way -- but at 44.1 kHz with
    // n = 1024, where a power-of-two `n` makes `1 / (n d)` and `(1 / d) / n` the SAME float, so a
    // reordered `val` passed the whole workspace and the whole Python suite. The detector itself
    // only ever asks for power-of-two lengths; the viewer's panels and several harnesses ask
    // `rfftfreq` for the record's own length, where about one (n, fs) pair in three differs.
    // n = 1001 at 8 kHz is one: the chain gives 7.992007992007991, the other two ...992.
    let (n, fs) = (1001usize, 8000.0);
    let d = 1.0 / fs;
    let f = rfftfreq(n, d);
    let val = 1.0 / (n as f64 * d);
    assert_eq!(val, 7.992_007_992_007_991);
    // The controls must differ, or this asserts nothing (plan §25).
    assert_ne!(
        val,
        (1.0 / d) / n as f64,
        "the reordered spelling no longer differs here"
    );
    assert_ne!(
        val,
        fs / n as f64,
        "the tidy spelling no longer differs here"
    );
    assert_eq!(f.len(), n / 2 + 1);
    for (i, &fi) in f.iter().enumerate() {
        assert_eq!(fi, i as f64 * val, "bin {i} is not i*val exactly");
    }
}

#[test]
fn two_peaks_exactly_the_minimum_separation_apart_are_both_kept() {
    // Added in §43 (the human's call). `detect_peaks` keeps a candidate when it is AT LEAST
    // `min_sep` from every kept peak; `>` for `>=` drops a pair sitting exactly on the limit and
    // was seen only by the Windows-exact viewer freeze. Two pure tones on exact bins of the padded
    // FFT, 400 bins apart, with the separation asked for spelled from the same axis the comparison
    // reads, so the gap and the threshold are the same float. Under `>` the second slot goes to a
    // sidelobe five bins away instead (1372.0 Hz; the tones sit within 3e-7 bins of their own).
    let fs = 8000.0;
    let n = 4096usize;
    let axis = rfftfreq(2 * n, 1.0 / fs); // `magnitude_spectrum` pads by 2: nfft = 8192
    let (i1, i2) = (1000usize, 1400usize);
    let (f1, f2) = (axis[i1], axis[i2]);
    let sig = tones(n, fs, &[(f1, 1.0), (f2, 0.5)]);
    let sep = axis[i2] - axis[i1];

    let found = detect_peaks(&sig, fs, 2, 50.0, Some(sep));
    assert_eq!(found.len(), 2, "two tones, two peaks: {found:?}");
    let df = axis[1];
    assert!(
        (found[0] - f1).abs() < 1e-3 * df && (found[1] - f2).abs() < 1e-3 * df,
        "the pair exactly {sep} Hz apart was not both kept: found {found:?}, tones {f1} and {f2}"
    );
}
