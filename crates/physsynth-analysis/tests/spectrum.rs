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
    // `2 ** ceil(log2(n))` over 1..2^20 is asserted on the Python side, where both exist.
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
    // nfft = 16 it does not, which is why the parity file asserts that the two implementations
    // agree rather than that the comparison passes.
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
    // `tests/test_spectrum_detector.py`. Both are CONCAVE, which is why a sign-of-curvature guard
    // catches neither and the guard has to be about local-maximality.
    let witnesses: [[f64; 3]; 2] = [
        [0.00054482, 0.00045897, 0.00038359],
        [0.00037953, 0.00033298, 0.00028228],
    ];
    let (fs, nfft, i) = (8000.0, 256usize, 4usize);
    for w in witnesses {
        let mut mag = vec![0.0; nfft / 2 + 1];
        mag[i - 1..i + 2].copy_from_slice(&w);

        let (a, b, c) = (w[0].ln(), w[1].ln(), w[2].ln());
        assert!(a - 2.0 * b + c < 0.0, "the witness must be concave");

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
    let fs = 44100.0;
    let sig = tones(1024, fs, &[(440.0, 1.0)]);
    let found = measure_partials_near(&sig, fs, &[30000.0, 0.5], Some(10.0));
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
