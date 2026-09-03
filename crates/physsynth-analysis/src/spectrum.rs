//! The partial detector — `physsynth/analysis/spectrum.py`, transcribed.
//!
//! A Hann-windowed FFT plus parabolic interpolation on the log-magnitude spectrum, which is how
//! every modal test in this project asks where a partial actually is before comparing it to an
//! analytic oracle. Fifteen test files stand on it.
//!
//! # What is exact here and what is not, and why the line falls where it does
//!
//! This is the first module in the migration whose two halves have *different* answers, and the
//! split is not the usual one between a reduction and a step. It is between two axes:
//!
//! * **The frequency axis is exact.** `freqs`, `df`, the search-window bounds, `min_separation_hz`
//!   and every comparison between them are built from `+ - * /` alone, which IEEE-754 specifies
//!   exactly, so a transcription reproduces them bit for bit on any machine. That matters more
//!   than it sounds: measured over the 17 real `detect_peaks` calls the suite makes, the greedy
//!   separation test compares two quantities that are **exactly equal** — candidate peaks sitting
//!   precisely four bins apart against a threshold of precisely four bins — so its margin is
//!   *zero*, and the `>=` is decided by whether `i*val - c*val` happens to land on `4.0*val`.
//!
//!   Note what is and is not claimed there, because the first draft of the parity test claimed the
//!   wrong thing. The gap does **not** always clear: at 100 kHz with `nfft = 16` it comes out short
//!   and the candidate is rejected. So this is a live comparison whose answer changes with the
//!   sample rate, and what a port must guarantee is not the outcome but that both sides reach the
//!   *same* outcome — which follows from the axis being bit-identical, and only while the chain is
//!   transcribed rather than tidied. `fs/nfft` is not `1.0/(nfft*(1.0/fs))`: at this project's own
//!   rates the two agree at every power-of-two size, but searched over random rates they differ for
//!   about one pair in eight. Tidying it would leave the port accidentally correct here and wrong
//!   elsewhere.
//!
//! * **The magnitude axis is a tolerance port, and deliberately so.** Three things on it are
//!   library kernels that no transcription reaches: NumPy's `cos` inside the Hann window (its own
//!   CPU-dispatched routine, not libm — plan §22.1), `np.fft.rfft` (pocketfft, whose blocking is
//!   not radix-2 and whose bit pattern is not a target), and `np.abs` on a complex array (`hypot`).
//!   So the magnitudes agree to about a last bit and no claim is made that they agree exactly.
//!
//! The reason that split is *safe* is a measurement rather than an argument, taken before this
//! file was written, over the 384 real `measure_partials_near` calls and 92,261 candidate peaks the
//! dependent suite generates (plan §36.2):
//!
//! | decision | lives on | margin |
//! |---|---|---|
//! | which bin wins a search window | magnitudes | >= 1.4e12 ulps (3.0e-4 relative) |
//! | is that bin a genuine local max | magnitudes | >= 1.6e10 ulps |
//! | ordering of candidates by strength | magnitudes | >= 7.6e7 ulps, zero exact ties |
//! | does a candidate clear the separation | frequencies | **exactly zero** |
//! | is a candidate above `f_min` | frequencies | 0.2 Hz |
//! | `nfft` | integers | exact by construction |
//!
//! Every decision with a tight margin is on the exact axis; every decision on the inexact axis
//! clears by ten orders of magnitude more than a rounding can move it. That is the whole licence
//! for porting a discrete-output module without bit-identical arithmetic, and if a future fixture
//! ever puts two partials within 3e-4 of each other in magnitude inside one search window, the
//! licence expires and the parity file is where that will show up.

use std::f64::consts::PI;

/// The FFT length for a record of `n` samples: the next power of two at or above `max(n, 2)`.
///
/// The original spells this `int(2 ** np.ceil(np.log2(max(n, 2))))`, which puts a **transcendental
/// inside a discrete decision** — the one shape plan §25 says to distrust, since a last bit in
/// `log2` next to an integer would be a different FFT length and therefore a different spectrum,
/// not a different last digit. Here it is integer arithmetic instead, which cannot round at all.
///
/// That is a substitution, so it is measured rather than assumed: `tests/test_rust_parity_spectrum.py`
/// checks the two spellings agree for **every** length from 1 to 2^20 and at each of `2^k - 1`,
/// `2^k`, `2^k + 1` up to 2^31. They do, everywhere. The float path is safe on this range and the
/// integer one is safe on every range, so the integer one is what runs.
pub fn next_pow2(n: usize) -> usize {
    let m = n.max(2);
    // `(m - 1).next_power_of_two()` would be wrong for m already a power of two; this is the
    // ceiling form: 8 -> 8, 9 -> 16.
    m.next_power_of_two()
}

/// `np.hanning(m)`, in NumPy's spelling rather than the textbook one.
///
/// The textbook Hann window is `0.5 - 0.5*cos(2*pi*k/(M-1))` for `k` in `0..M`. NumPy computes
/// something algebraically identical and numerically different: `n = arange(1-M, M, 2)`, then
/// `0.5 + 0.5*cos(pi*n/(M-1))` — a symmetric argument about zero, the opposite sign, and one
/// multiply-then-divide instead of a divide-then-multiply. Writing the textbook form here would
/// move the window's last bits for no reason and against no oracle.
///
/// The two early returns are NumPy's too, and the `M == 1` one is not decoration: the general
/// formula divides by `M - 1`.
pub fn hann(m: usize) -> Vec<f64> {
    // NumPy widens M to f64 first (`values = np.array([0.0, M]); M = values[1]`), which is why the
    // arithmetic below is float throughout rather than integer-then-cast.
    let mf = m as f64;
    if mf < 1.0 {
        return Vec::new();
    }
    if mf == 1.0 {
        return vec![1.0];
    }
    // `arange(1-M, M, 2)` on floats is `start + i*step`, not repeated addition.
    (0..m)
        .map(|i| {
            let ni = (1.0 - mf) + 2.0 * (i as f64);
            0.5 + 0.5 * ((PI * ni) / (mf - 1.0)).cos()
        })
        .collect()
}

/// `np.fft.rfftfreq(n, d)` — the one-sided frequency axis.
///
/// The chain is transcribed exactly, including that `val` is `1.0 / (n * d)` and each frequency is
/// `i * val`. See this module's header for why: a candidate pair exactly four bins apart is
/// compared against a threshold of exactly four bins, and that comparison has zero margin.
pub fn rfftfreq(n: usize, d: f64) -> Vec<f64> {
    let val = 1.0 / (n as f64 * d);
    (0..n / 2 + 1).map(|i| i as f64 * val).collect()
}

/// In-place forward DFT, radix-2 decimation-in-time, `re.len()` a power of two.
///
/// Hand-written for the reason `sparse.rs` gives about CSR and `tests/deps.rs` repeats: a library
/// FFT would pin this project's spectra to a third party's blocking order, and these numbers are
/// the acceptance contract. No attempt is made to match pocketfft — see the header. What *is*
/// required is that it be a correct DFT to within the usual `O(eps * log n)`, which the native
/// tests check against a directly evaluated DFT and against Parseval.
fn fft_in_place(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    debug_assert!(n.is_power_of_two() && im.len() == n);
    if n < 2 {
        return;
    }

    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    // Butterflies. Each twiddle is computed from `sin_cos` directly rather than advanced by
    // repeated complex multiplication: the recurrence is cheaper and drifts, and an FFT whose
    // error grows with n would eat the margin this module's header depends on.
    //
    // The twiddles for a stage are tabulated once and then read by the block loop, which is worth
    // the paragraph because both of the obvious spellings are measurably worse and they fail in
    // opposite directions (measured end to end against NumPy at n = 2^10 .. 2^18):
    //
    //   blocks outside, `sin_cos` inside   0.60x .. 0.21x   -- recomputes each twiddle n/len times,
    //                                                          `(n/2)*log2(n)` transcendental pairs
    //   k outside, blocks inside           1.28x .. 0.11x   -- `n-1` pairs, but the inner loop then
    //                                                          strides by `len` and falls out of cache
    //   tabulate per stage, blocks outside 1.25x .. 0.51x   -- `n-1` pairs AND a unit-stride inner
    //                                                          loop, which is this
    //
    // The middle one is the instructive failure: hoisting the transcendental made the small
    // transforms twice as fast and the large ones five times slower, because what it actually
    // hoisted was the memory access pattern. The table costs `n/2` pairs of f64 of scratch.
    //
    // Where it lands: faster than NumPy on short records and about half its speed on long ones.
    // pocketfft is mixed-radix and vectorised and this is a textbook radix-2, so 2x is the honest
    // price of not taking a dependency (`tests/deps.rs`), and it is paid by measurement code that
    // runs once per test rather than by anything in a timestep loop.
    let mut tw_cos = vec![0.0; n / 2];
    let mut tw_sin = vec![0.0; n / 2];
    let mut len = 2usize;
    while len <= n {
        let half = len / 2;
        let ang = -2.0 * PI / (len as f64);
        for k in 0..half {
            let (sn, cs) = (ang * k as f64).sin_cos();
            tw_cos[k] = cs;
            tw_sin[k] = sn;
        }
        for start in (0..n).step_by(len) {
            for k in 0..half {
                let (wc, ws) = (tw_cos[k], tw_sin[k]);
                let (ur, ui) = (re[start + k], im[start + k]);
                let (vr, vi) = (re[start + k + half], im[start + k + half]);
                let tr = vr * wc - vi * ws;
                let ti = vr * ws + vi * wc;
                re[start + k] = ur + tr;
                im[start + k] = ui + ti;
                re[start + k + half] = ur - tr;
                im[start + k + half] = ui - ti;
            }
        }
        len <<= 1;
    }
}

/// `(freqs, magnitude, nfft)` of the DC-removed, Hann-windowed signal.
///
/// The mean is a plain left-to-right sum, **not** NumPy's pairwise one, and that is a deliberate
/// refusal rather than an oversight. `reduce::sum` next door exists precisely to make a ported
/// `np.sum` bit-identical (plan §31.2), and it would work here — but it would buy a bit-identical
/// *input* to a path that then goes through NumPy's `cos`, pocketfft and `hypot`, none of which
/// can be matched. A half-exact path is worse than an honestly inexact one: it invites the next
/// reader to believe an equality that was never true. The exact claims this module makes are the
/// discrete ones listed in the header, and they do not run through here.
pub fn magnitude_spectrum(signal: &[f64], fs: f64, zero_pad_factor: usize) -> SpectrumOut {
    let len = signal.len();
    let mut acc = 0.0;
    for &x in signal {
        acc += x;
    }
    let mean = acc / len as f64;

    let win = hann(len);
    let nfft = next_pow2(len.saturating_mul(zero_pad_factor));

    // `np.fft.rfft(x, n=nfft)` zero-pads a short record and truncates a long one.
    let mut re = vec![0.0; nfft];
    let mut im = vec![0.0; nfft];
    for i in 0..len.min(nfft) {
        re[i] = (signal[i] - mean) * win[i];
    }
    fft_in_place(&mut re, &mut im);

    let freqs = rfftfreq(nfft, 1.0 / fs);
    // `np.abs` on a complex array is `hypot`, not `sqrt(re*re + im*im)`; they differ in the last
    // bit and hypot is the one that does not overflow. Matching the spelling costs nothing.
    let mag = (0..nfft / 2 + 1).map(|i| re[i].hypot(im[i])).collect();
    SpectrumOut { freqs, mag, nfft }
}

/// What [`magnitude_spectrum`] returns: the frequency axis, the magnitudes, and the FFT length.
pub struct SpectrumOut {
    pub freqs: Vec<f64>,
    pub mag: Vec<f64>,
    pub nfft: usize,
}

/// Sub-bin frequency (Hz) of the peak at bin `i`, by parabolic interpolation on log-magnitudes.
///
/// **The guard is the point of this function, not the parabola.** `measure_partials_near` takes its
/// argmax inside a search window, and a window whose magnitudes fall monotonically puts that argmax
/// on the window's own edge — a bin with no peak under it. Fitting a parabola through three
/// near-collinear log-magnitudes there makes the curvature vanish and the correction diverge: the
/// witness that prompted the guard moved -22.1 bins from bin 4, i.e. a "refined" frequency of
/// **-502 Hz**, which became a NaN out of `modal.cents` and a 500 in the viewer.
///
/// Note which guard it has to be. Both real witnesses were **concave**, so testing the sign of the
/// denominator catches neither. The property that makes the interpolation meaningful, and that
/// bounds `|delta| <= 1/2`, is that `i` is a genuine local maximum. A non-peak bin gets its honest
/// bin centre back.
///
/// This is not a hypothetical branch being carried across for completeness: over the dependent
/// suite the guard fires on **14 of 384** real calls, and 17 of those calls take an edge argmax.
/// Its tightest decision clears by 1.6e10 ulps, which is why it survives a magnitude path that is
/// only accurate to a last bit.
pub fn parabolic_refine(mag: &[f64], i: usize, fs: f64, nfft: usize) -> f64 {
    let bin_centre = i as f64 * fs / nfft as f64;
    if i == 0 || i + 1 >= mag.len() {
        return bin_centre;
    }
    if mag[i] < mag[i - 1] || mag[i] < mag[i + 1] {
        return bin_centre;
    }
    // The `1e-300` floor is the original's, and it is load-bearing for an all-zero window: without
    // it a silent bin takes `ln(0)` and the correction becomes NaN rather than zero.
    let a = (mag[i - 1] + 1e-300).ln();
    let b = (mag[i] + 1e-300).ln();
    let c = (mag[i + 1] + 1e-300).ln();
    let denom = a - 2.0 * b + c;
    let delta = if denom != 0.0 {
        0.5 * (a - c) / denom
    } else {
        0.0
    };
    (i as f64 + delta) * fs / nfft as f64
}

/// Measure the partial frequencies nearest each value in `expected`; NaN where the window is empty.
///
/// `search_hz` defaults to 40% of the lowest expected frequency — 40% of the harmonic spacing,
/// which keeps each window clear of its neighbours.
pub fn measure_partials_near(
    signal: &[f64],
    fs: f64,
    expected: &[f64],
    search_hz: Option<f64>,
) -> Vec<f64> {
    let s = magnitude_spectrum(signal, fs, 2);
    let df = s.freqs[1] - s.freqs[0];
    let search = match search_hz {
        Some(v) => v,
        // `expected.min()` on an empty array raises in the original; there is nothing to return.
        None => 0.4 * expected.iter().copied().fold(f64::INFINITY, f64::min),
    };

    let last = s.mag.len() - 1;
    expected
        .iter()
        .map(|&fe| {
            // `int(np.floor(x))` and `int(np.ceil(x))`: floor/ceil first, then to integer, so a
            // negative lower edge stays negative before `max(1, ..)` clamps it.
            let lo = (((fe - search) / df).floor() as i64).max(1) as usize;
            let hi = ((((fe + search) / df).ceil() as i64).max(0) as usize).min(last);
            if hi <= lo {
                return f64::NAN;
            }
            // `np.argmax` returns the FIRST maximum, which `>` (not `>=`) reproduces.
            let mut i = lo;
            for j in lo + 1..=hi {
                if s.mag[j] > s.mag[i] {
                    i = j;
                }
            }
            parabolic_refine(&s.mag, i, fs, s.nfft)
        })
        .collect()
}

/// Blindly detect the `n_peaks` strongest spectral peaks above `f_min`, ascending in Hz.
///
/// Uses no prior knowledge of where partials should be — the independent cross-check that the
/// detector finds a harmonic series on its own.
///
/// **The sort is a stable descending sort, where NumPy's is `argsort` reversed and unstable.** The
/// two agree exactly when no two candidate magnitudes are equal, and that is a measured
/// precondition rather than an identity: over the 17 real calls in the dependent suite, 92,261
/// candidate peaks produced **zero** exact magnitude ties, with the tightest adjacent gap at 7.6e7
/// ulps. If a fixture ever produces a tie the two orders can differ, and because the selection
/// below is a greedy cascade that would change *which peaks come out*, not one peak's last digit.
pub fn detect_peaks(
    signal: &[f64],
    fs: f64,
    n_peaks: usize,
    f_min: f64,
    min_separation_hz: Option<f64>,
) -> Vec<f64> {
    let s = magnitude_spectrum(signal, fs, 2);
    let df = s.freqs[1] - s.freqs[0];
    let sep = min_separation_hz.unwrap_or(4.0 * df);

    let mut cand: Vec<usize> = (1..s.mag.len().saturating_sub(1))
        .filter(|&i| s.mag[i] > s.mag[i - 1] && s.mag[i] > s.mag[i + 1])
        .filter(|&i| s.freqs[i] >= f_min)
        .collect();
    if cand.is_empty() {
        return Vec::new();
    }

    cand.sort_by(|&a, &b| s.mag[b].total_cmp(&s.mag[a]));
    let mut chosen: Vec<usize> = Vec::new();
    for i in cand {
        let f = s.freqs[i];
        if chosen.iter().all(|&c| (f - s.freqs[c]).abs() >= sep) {
            chosen.push(i);
        }
        if chosen.len() >= n_peaks {
            break;
        }
    }

    let mut refined: Vec<f64> = chosen
        .iter()
        .map(|&i| parabolic_refine(&s.mag, i, fs, s.nfft))
        .collect();
    refined.sort_by(f64::total_cmp);
    refined
}
