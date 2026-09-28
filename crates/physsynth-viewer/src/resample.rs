//! The audio path's rational resampler — `scipy.signal.resample_poly` with its defaults, and the
//! `Fraction.limit_denominator` that picked its ratio.
//!
//! The pickup is recorded at the simulation rate, which rides `N` and `c` and can be far above what
//! a browser `AudioBuffer` accepts, so every scene ships its audio at a fixed 48 kHz (the viewer
//! plan's catch #1). The reference did that with SciPy; this is SciPy's algorithm transcribed from
//! the installed 1.17.1 source, so the output agrees with the reference to within the last bits of
//! the filter taps rather than being a different, merely-good resampler:
//!
//! * the ratio is `Fraction(48000, round(fs)).limit_denominator(2000)`, reduced;
//! * the filter is `firwin(2 * 10 * max(up, down) + 1, 1 / max(up, down), window=('kaiser', 5.0))`
//!   — a windowed sinc, normalized to unit DC gain, then scaled by `up`;
//! * it is zero-padded in front so the output samples land on the filter's centre, applied by
//!   `upfirdn` (a polyphase loop that accumulates each output in increasing input order), and the
//!   leading `n_pre_remove` outputs are dropped.
//!
//! **Why it cannot be bit-identical, and why that does not matter.** The taps call `sin` and the
//! Kaiser window calls a Bessel `I0`. NumPy computes `sin` with its own CPU-dispatched routine
//! (findings §22.1) and SciPy's `i0` is Cephes' Chebyshev expansion, where this uses the series in
//! `physsynth-analysis`. So a tap can differ in its last bit, and the audio with it. The audio is
//! then shipped as **float32**, which discards 29 of those bits; the one-time comparison in
//! retirement plan §23 measured what survives.

use physsynth_analysis::bessel::iv;
use physsynth_core::reduce::sum;
use std::f64::consts::PI;

/// `Fraction(num, den).limit_denominator(max_den)`, as `(numerator, denominator)` in lowest terms.
///
/// CPython 3.12+'s algorithm: the continued-fraction convergents until the next denominator would
/// pass the bound, then the better of the last convergent and the best semiconvergent — "better"
/// decided in exact integer arithmetic, so there is no float in it to disagree about.
pub fn limit_denominator(num: u64, den: u64, max_den: u64) -> (u64, u64) {
    let g = gcd(num, den);
    let (num, den) = (num / g, den / g);
    if den <= max_den {
        return (num, den);
    }
    let (mut p0, mut q0, mut p1, mut q1) = (0u128, 1u128, 1u128, 0u128);
    let (mut n, mut d) = (u128::from(num), u128::from(den));
    let max_d = u128::from(max_den);
    loop {
        let a = n / d;
        let q2 = q0 + a * q1;
        if q2 > max_d {
            break;
        }
        (p0, q0, p1, q1) = (p1, q1, p0 + a * p1, q2);
        (n, d) = (d, n - a * d);
    }
    let k = (max_d - q0) / q1;
    // `if 2*d*(q0+k*q1) <= self._denominator: p1/q1 else (p0+k*p1)/(q0+k*q1)`
    if 2 * d * (q0 + k * q1) <= u128::from(den) {
        (p1 as u64, q1 as u64)
    } else {
        ((p0 + k * p1) as u64, (q0 + k * q1) as u64)
    }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// `np.sinc(x) = sin(pi x) / (pi x)`, with `x == 0` replaced by `1e-20` so the ratio is 1.
fn sinc(x: f64) -> f64 {
    let y = PI * if x == 0.0 { 1.0e-20 } else { x };
    y.sin() / y
}

/// `scipy.signal.windows.kaiser(m, beta, sym=True)`.
fn kaiser(m: usize, beta: f64) -> Vec<f64> {
    if m <= 1 {
        return vec![1.0; m];
    }
    let alpha = (m - 1) as f64 / 2.0;
    let denom = iv(0, beta);
    (0..m)
        .map(|n| {
            let r = (n as f64 - alpha) / alpha;
            iv(0, beta * (1.0 - r.powf(2.0)).sqrt()) / denom
        })
        .collect()
}

/// `firwin(numtaps, cutoff, window=('kaiser', 5.0))` — low-pass from DC, `scale=True`.
///
/// The band is `[0, cutoff]`, so the left edge contributes `0 * sinc(0 * m) == 0` and is omitted;
/// the scale frequency is 0, so the normalizer is the plain (pairwise) sum of the taps.
fn firwin_lowpass_kaiser(numtaps: usize, cutoff: f64) -> Vec<f64> {
    let alpha = 0.5 * (numtaps - 1) as f64;
    let win = kaiser(numtaps, 5.0);
    let mut h: Vec<f64> = (0..numtaps)
        .map(|i| {
            let m = i as f64 - alpha;
            cutoff * sinc(cutoff * m) * win[i]
        })
        .collect();
    let s = sum(&h);
    for v in &mut h {
        *v /= s;
    }
    h
}

/// `scipy.signal._upfirdn._output_len`.
fn output_len(len_h: usize, in_len: usize, up: usize, down: usize) -> usize {
    let padded = len_h + (up - len_h % up) % up;
    let nt = (in_len + padded / up - 1) * up;
    nt.div_ceil(down)
}

/// `upfirdn(h, x, up, down)` in `'constant'` mode with a zero pad — the polyphase apply loop.
fn upfirdn(h: &[f64], x: &[f64], up: usize, down: usize) -> Vec<f64> {
    let len_out = output_len(h.len(), x.len(), up, down);
    // `_pad_h`: pad to a multiple of `up`, then `reshape(-1, up).T[:, ::-1].ravel()` — phase `p`
    // holds taps p, p+up, p+2up, ... in REVERSE order.
    let padlen = h.len() + (up - h.len() % up) % up;
    let per_phase = padlen / up;
    let mut flip = vec![0.0; padlen];
    for p in 0..up {
        for j in 0..per_phase {
            let src = (per_phase - 1 - j) * up + p;
            flip[p * per_phase + j] = if src < h.len() { h[src] } else { 0.0 };
        }
    }

    let mut out = vec![0.0; len_out];
    if len_out == 0 {
        return out;
    }
    let len_x = x.len() as isize;
    let hpp = per_phase as isize;
    let padded_len = len_x + hpp - 1;
    let (mut x_idx, mut y_idx, mut t) = (0isize, 0usize, 0isize);
    let (up_i, down_i) = (up as isize, down as isize);
    while x_idx < len_x {
        let mut h_idx = t * hpp;
        let mut start = x_idx - hpp + 1;
        if start < 0 {
            h_idx -= start;
            start = 0;
        }
        for xc in start..=x_idx {
            out[y_idx] += x[xc as usize] * flip[h_idx as usize];
            h_idx += 1;
        }
        y_idx += 1;
        if y_idx >= len_out {
            return out;
        }
        t += down_i;
        x_idx += t / up_i;
        t %= up_i;
    }
    // The flush loop: past the end of `x` the zero pad contributes nothing.
    while x_idx < padded_len {
        for (h_idx, xc) in (t * hpp..).zip((x_idx - hpp + 1)..=x_idx) {
            if xc >= 0 && xc < len_x {
                out[y_idx] += x[xc as usize] * flip[h_idx as usize];
            }
        }
        y_idx += 1;
        if y_idx >= len_out {
            return out;
        }
        t += down_i;
        x_idx += t / up_i;
        t %= up_i;
    }
    out
}

/// `scipy.signal.resample_poly(x, up, down)` with the default Kaiser window and zero padding.
pub fn resample_poly(x: &[f64], up: u64, down: u64) -> Vec<f64> {
    let g = gcd(up, down);
    let (up, down) = ((up / g) as usize, (down / g) as usize);
    if up == 1 && down == 1 {
        return x.to_vec();
    }
    let n_in = x.len();
    let n_out = (n_in * up).div_ceil(down);
    let max_rate = up.max(down);
    let half_len = 10 * max_rate;
    let mut h = firwin_lowpass_kaiser(2 * half_len + 1, 1.0 / max_rate as f64);
    for v in &mut h {
        *v *= up as f64;
    }
    let n_pre_pad = down - half_len % down;
    let n_pre_remove = (half_len + n_pre_pad) / down;
    let mut n_post_pad = 0;
    while output_len(h.len() + n_pre_pad + n_post_pad, n_in, up, down) < n_out + n_pre_remove {
        n_post_pad += 1;
    }
    let mut padded = vec![0.0; n_pre_pad];
    padded.extend_from_slice(&h);
    padded.extend(std::iter::repeat_n(0.0, n_post_pad));
    let y = upfirdn(&padded, x, up, down);
    y[n_pre_remove..n_pre_remove + n_out].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limit_denominator_matches_cpython_on_known_cases() {
        // Values from `Fraction(a, b).limit_denominator(2000)` in CPython 3.14.
        assert_eq!(limit_denominator(48000, 12800, 2000), (15, 4));
        assert_eq!(limit_denominator(48000, 48000, 2000), (1, 1));
        // 3.14159... -> the classic 355/113 under a bound of 1000.
        assert_eq!(limit_denominator(314159265, 100000000, 1000), (355, 113));
        assert_eq!(limit_denominator(48000, 44101, 2000), (911, 837));
        assert_eq!(limit_denominator(48000, 25601, 2000), (3193, 1703));
    }

    #[test]
    fn identity_ratio_is_a_copy() {
        let x = [1.0, -2.0, 3.0];
        assert_eq!(resample_poly(&x, 7, 7), x.to_vec());
    }

    #[test]
    fn output_length_is_ceil_of_n_up_over_down() {
        for (n, up, down) in [(100, 15, 4), (99, 4, 15), (1, 3, 2), (1000, 911, 837)] {
            assert_eq!(
                resample_poly(&vec![0.5; n], up, down).len(),
                (n * up as usize).div_ceil(down as usize)
            );
        }
    }

    #[test]
    fn a_tone_well_inside_both_bands_keeps_its_amplitude() {
        // 200 Hz at 12.8 kHz, up to 48 kHz: after the filter's start-up, the resampled tone must
        // match the analytic one sampled at the new rate. The Kaiser(5) low-pass is flat to ~1e-3
        // this far below cutoff, and the phase is exact because the filter is centred.
        let fs_in = 12_800.0;
        let x: Vec<f64> = (0..2_000)
            .map(|i| (2.0 * PI * 200.0 * i as f64 / fs_in).sin())
            .collect();
        let y = resample_poly(&x, 15, 4);
        let fs_out = 48_000.0;
        for (i, &v) in y.iter().enumerate().skip(1_000).take(4_000) {
            let want = (2.0 * PI * 200.0 * i as f64 / fs_out).sin();
            assert!((v - want).abs() < 2e-3, "sample {i}: {v} vs {want}");
        }
    }

    #[test]
    fn filter_taps_sum_to_one_before_the_up_gain() {
        let h = firwin_lowpass_kaiser(2 * 10 * 15 + 1, 1.0 / 15.0);
        assert!((sum(&h) - 1.0).abs() < 1e-15);
        // Linear phase: the taps are symmetric.
        for i in 0..h.len() / 2 {
            assert_eq!(h[i], h[h.len() - 1 - i]);
        }
    }
}
