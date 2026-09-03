//! Bessel functions of integer order — the oracles' one library debt, paid.
//!
//! Three call sites in this project need a Bessel function and all three are oracles:
//!
//! * `circular_membrane_freqs` needs `j_{m,n}`, the zeros of `J_m` — the drumhead's analytic
//!   frequencies (HANDOFF §5 row 4).
//! * `free_circular_plate_lambda_roots` needs `J_n` and `I_n` and their first three derivatives —
//!   the free circular plate's frequency determinant, which is the guitar-shaped plate's oracle.
//! * `piston_radiation_resistance` needs `J_1` — Rayleigh's baffled-piston resistance. It lives in
//!   `physsynth/core/radiation.py` rather than in `analysis/`, and plan §14 parked it *there* for
//!   exactly this file: "a file's Bessel call does not drag the file into Phase 7; it drags the
//!   call." That is the debt this module clears, and it is why the plan made these one batch.
//!
//! # Why a transcription and not a crate
//!
//! `tests/deps.rs` keeps this crate's dependency list empty and says the first entry arrives as a
//! reviewed edit with its reason next to it. A Bessel crate does not supply that reason. The
//! requirement is not "a Bessel function" — every oracle here is compared against SciPy by a
//! parity test, and SciPy's are Cephes and AMOS. A different-but-equally-good `J_0` would move the
//! numbers this project's acceptance runs were taken with, and would move them silently, since
//! the physics bars downstream are percentage-level and would absorb a 1e-9 oracle shift without
//! a word. So this is written against the *definitions*, checked against SciPy over the whole
//! domain the callers reach, and the agreement is measured rather than assumed.
//!
//! # The finding this module is the evidence for: a Bessel function's bar is ABSOLUTE
//!
//! Measured against `scipy.special.jv` over `x ∈ (0, 60]`, `n ∈ 0..13` before any of this was
//! written (800 × 14 points):
//!
//! | quantity | worst agreement with SciPy |
//! |---|---|
//! | `J_n(x)` | **6.7e-16 absolute** — and 2.1e-12 *relative* |
//! | `jvp(n, x, k)`, k ≤ 3, x ≤ 14 | **4.1e-16 absolute** — 1.8e-12 relative |
//! | `I_n(x)`, x ≤ 14 | 2.0e-15 relative |
//! | `jn_zeros(m, n)`, m ≤ 12, n ≤ 12 | 3.2e-16 relative |
//!
//! Those two columns are not in tension and reading them as a warning would be the mistake. The
//! relative worst case sits at `x = 32.0655, n = 3`, where `J_3` itself is `-9.8e-05` — a point
//! near one of its own zeros. The absolute error there is the same 1e-16 it is everywhere; it is
//! the *denominator* that collapsed. A Bessel function is an oscillation with unit amplitude that
//! passes through zero infinitely often, so a relative bar on it is a bar that tightens without
//! limit at points of no physical significance, and any test that states one is really stating a
//! claim about how close a fixture landed to a root.
//!
//! What consumes these values decides which column matters, and here both consumers want the
//! absolute one:
//!
//! * the determinant in `modal::free_circular_plate_lambda_roots` **adds** J and I values, so an
//!   absolute error propagates additively and a relative one near a zero does not propagate at all;
//! * Newton's step on a zero is `J/J'`, and near the root `J → 0` while `J'` does not, so the
//!   error in the root is `|ΔJ| / |J'|` — again absolute over a bounded slope.
//!
//! So every native bar below and every parity assertion in `tests/test_rust_parity_analysis.py`
//! is absolute, with the relative figure reported alongside and never required. Ledger #30.
//!
//! # The algorithms, and why these three
//!
//! **`J_n` by Miller's downward recurrence with the self-normalising sum.** The obvious spelling —
//! the ascending power series — is a trap at the arguments this project reaches. `jn_zeros(12, 12)`
//! is 54.44, and the series for `J_0(54)` has a largest term near 1e17 summing to a result near
//! 0.1: eighteen digits of cancellation, so the answer is noise. The Hankel asymptotic form does
//! not rescue it either — at the crossover `x ≈ 8` its terms bottom out around 1e-9, five orders
//! short of what a determinant needs. Downward recurrence is stable in the direction `J_k` decays,
//! needs no coefficient tables, and normalises itself through `J_0 + 2ΣJ_{2k} = 1`, which is what
//! makes it accurate rather than merely stable.
//!
//! **`I_n` by the ascending series, which for the modified function has no cancellation at all.**
//! Every term of `Σ (x/2)^{n+2k} / (k!(n+k)!)` is positive. The sum and its largest term are the
//! same size by construction, so the series that is useless for `J` is exact for `I` — the only
//! difference between them is the `(-1)^k`. Bounded to `x ≤ 14` here, which is the free plate's
//! whole scan; past that `I_n` overflows the `f64` range the caller wants anyway.
//!
//! **Zeros by interlacing, not by asymptotics-plus-Newton.** `j_{m,n}` lies strictly between
//! `j_{m-1,n}` and `j_{m-1,n+1}` — a theorem, not an approximation — so the zeros of `J_0` (where
//! McMahon's expansion is excellent and Newton converges from it in three steps) bracket those of
//! `J_1`, which bracket those of `J_2`, and so on up. Every zero past the first order is therefore
//! found by [`crate::root::brentq`] inside a bracket that is *proved* to contain exactly one, which
//! is the property a bare Newton iteration cannot offer: Newton on an oscillating function can
//! walk to a neighbouring root and return a plausible wrong answer, and nothing downstream would
//! catch it — the frequency it produces is a real mode of the drum, just not the one asked for.

use crate::root::brentq;
use std::f64::consts::PI;

/// SciPy's `brentq` defaults, which is what the Python original passes when it does not say.
const ZERO_XTOL: f64 = 1e-15;
const ZERO_RTOL: f64 = 8.881_784_197_001_252e-16;

/// `J_0(x) .. J_nmax(x)` for `x >= 0`, by Miller's downward recurrence.
///
/// The starting order is `max(nmax, ceil(x)) + 40`, rounded up to even so the normalising sum
/// `J_0 + 2(J_2 + J_4 + ...)` ends on a term it includes. Forty is not tuned: the recurrence's
/// unwanted `Y_n` component grows downward like `n!`-ish and is gone within a handful of orders
/// past `x`, and the cost of overshooting is one multiply per order.
///
/// The rescale guard is the reason the seed can be arbitrary. Starting from `1e-290` the values
/// climb by roughly `2n/x` per step; over a hundred steps at small `x` that overflows, so whenever
/// the running value passes `1e250` everything computed so far is divided down. Scaling the whole
/// sequence by a constant is invisible to the normalisation, which is the point of it.
pub fn jn_all(x: f64, nmax: usize) -> Vec<f64> {
    let mut out = vec![0.0; nmax + 1];
    if x == 0.0 {
        out[0] = 1.0;
        return out;
    }
    let x = x.abs();
    let mut m = nmax.max(x.ceil() as usize) + 40;
    m += m & 1;

    let mut f = vec![0.0f64; m + 2];
    f[m] = 1e-290;
    for k in (1..=m).rev() {
        f[k - 1] = (2.0 * k as f64 / x) * f[k] - f[k + 1];
        if f[k - 1].abs() > 1e250 {
            for v in f.iter_mut().skip(k - 1) {
                *v *= 1e-250;
            }
        }
    }

    let mut s = f[0];
    let mut k = 2;
    while k <= m {
        s += 2.0 * f[k];
        k += 2;
    }
    for (n, o) in out.iter_mut().enumerate() {
        *o = f[n] / s;
    }
    out
}

/// `J_n(x)` for integer `n` of either sign, `x >= 0`.
///
/// `J_{-n} = (-1)^n J_n` is an identity rather than a convention, and the derivative recursion
/// below reaches negative orders for every `n < k`, so it is not an edge case here.
pub fn jn(n: i32, x: f64) -> f64 {
    let a = n.unsigned_abs() as usize;
    let v = jn_all(x, a)[a];
    if n < 0 && a % 2 == 1 {
        -v
    } else {
        v
    }
}

/// `J_1(x)`, the one name `physsynth/core/radiation.py` needs (plan §14's parked call).
pub fn j1(x: f64) -> f64 {
    jn(1, x)
}

/// `I_n(x)` for integer `n` of either sign, `0 <= x`, by the all-positive ascending series.
///
/// `I_{-n} = I_n`, unlike `J`. Terms are accumulated until one falls below `1e-18` of the running
/// sum, which for `x = 14` — the largest the free plate's scan reaches — takes about 35 of them.
pub fn iv(n: i32, x: f64) -> f64 {
    let n = n.unsigned_abs();
    if x == 0.0 {
        return if n == 0 { 1.0 } else { 0.0 };
    }
    let h = 0.5 * x.abs();
    let mut t = 1.0;
    for k in 1..=n {
        t *= h / k as f64;
    }
    let mut s = t;
    let h2 = h * h;
    let mut k = 1u32;
    while k <= 400 {
        t *= h2 / (k as f64 * (n + k) as f64);
        s += t;
        if t <= 1e-18 * s {
            break;
        }
        k += 1;
    }
    s
}

/// The `k`-th derivative of `J_n` at `x` — `scipy.special.jvp(n, x, k)`.
///
/// SciPy computes this as `2^-k Σ_{i=0..k} (-1)^i C(k,i) J_{n-k+2i}(x)`, and that identity is
/// transcribed rather than replaced by, say, three applications of `J_n' = (J_{n-1} - J_{n+1})/2`.
/// The two are equal in exact arithmetic and are not equal in doubles: the same terms in a
/// different association, which plan §22 has now cost this migration six times.
pub fn jvp(n: i32, x: f64, k: u32) -> f64 {
    if k == 0 {
        return jn(n, x);
    }
    let mut s = 0.0;
    for i in 0..=k {
        let c = binomial(k, i) as f64;
        let term = c * jn(n - k as i32 + 2 * i as i32, x);
        s += if i % 2 == 1 { -term } else { term };
    }
    s / (2.0f64).powi(k as i32)
}

/// The `k`-th derivative of `I_n` at `x` — `scipy.special.ivp(n, x, k)`. Same recursion, no signs.
pub fn ivp(n: i32, x: f64, k: u32) -> f64 {
    if k == 0 {
        return iv(n, x);
    }
    let mut s = 0.0;
    for i in 0..=k {
        s += binomial(k, i) as f64 * iv(n - k as i32 + 2 * i as i32, x);
    }
    s / (2.0f64).powi(k as i32)
}

/// `C(n, k)` for the small `n` the derivative recursion uses (`k <= 3` at every call site).
fn binomial(n: u32, k: u32) -> u64 {
    let k = k.min(n - k);
    let mut r = 1u64;
    for i in 0..k {
        r = r * (n - i) as u64 / (i + 1) as u64;
    }
    r
}

/// The first `n` positive zeros of `J_0`, by McMahon's expansion refined with Newton.
///
/// McMahon gives `j_{0,k} ≈ β + 1/(8β) - 124/(3(8β)³) + 120928/(15(8β)⁵)` with `β = (k - ¼)π`,
/// which is already good to about 1e-10 at `k = 1` and better after. Newton on `J_0` with
/// `J_0' = -J_1` then converges in two or three steps; the loop stops on a step below `1e-16 x`
/// rather than on a fixed count, and the cap exists only so a pathological input cannot spin.
pub fn j0_zeros(n: usize) -> Vec<f64> {
    (1..=n)
        .map(|k| {
            let b = (k as f64 - 0.25) * PI;
            let m8 = 8.0 * b;
            let mut x = b + 1.0 / m8 - 124.0 / (3.0 * m8.powi(3)) + 120_928.0 / (15.0 * m8.powi(5));
            for _ in 0..60 {
                let j = jn_all(x, 1);
                let step = j[0] / -j[1];
                x -= step;
                if step.abs() < 1e-16 * x {
                    break;
                }
            }
            x
        })
        .collect()
}

/// The first `n` positive zeros of `J_m` — `scipy.special.jn_zeros(m, n)`.
///
/// Climbs the orders one at a time on the interlacing theorem: `j_{m,i}` is the unique zero of
/// `J_m` in `(j_{m-1,i}, j_{m-1,i+1})`, so each is bracketed before it is found and `brentq`
/// cannot return the wrong one. Reaching order `m` with `n` zeros needs `n + m + 1` zeros of
/// `J_0`, one being consumed by each step up, which is what the `+ 1` accounts for.
pub fn jn_zeros(m: u32, n: usize) -> Vec<f64> {
    let mut prev = j0_zeros(n + m as usize + 1);
    for order in 1..=m as i32 {
        prev = (0..prev.len() - 1)
            .map(|i| {
                brentq(
                    |t| jn(order, t),
                    prev[i],
                    prev[i + 1],
                    ZERO_XTOL,
                    ZERO_RTOL,
                    100,
                )
                .expect("interlacing guarantees a sign change in this bracket")
            })
            .collect();
    }
    prev.truncate(n);
    prev
}
