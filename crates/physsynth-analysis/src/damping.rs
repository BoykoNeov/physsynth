//! The damped string's decay oracles — `physsynth/analysis/damping.py`, transcribed.
//!
//! Model #3's money test: a frequency-dependent loss term gives every mode its own decay rate, and
//! the per-mode `g_m` predicted here is what a run's measured envelope is compared against. Five
//! test files stand on it, and `spatial_eigenvalue_p2` alone is imported by four of the string
//! family's suites directly.
//!
//! # The one thing here that is not arithmetic
//!
//! `discrete_damped_mode_rate` takes a logarithm, and everything else in the file is `+ - * / √`.
//! So the exactness story is short: the decay *ratio* `g = c/a` is exact — both sides build it out
//! of four IEEE-754 operations from the same inputs — and the *rate* `-ln(g)/k` is a tolerance
//! port, because `np.log` is NumPy's own CPU-dispatched kernel and `f64::ln` is the platform's.
//! The parity file asserts the first as equality and the second at 1e-15 relative, which is the
//! same split plan §36 drew through `spectrum.py` — exact on the axis that decides, tolerant on the
//! axis that is measured.
//!
//! `discrete_damped_mode_is_underdamped` returns a **bool**, which by §25's rule is a discrete
//! output and would need a margin measured before it could be ported. It has one, and it is not
//! tight: `b² - 4ac < 0` is compared over the configurations the suite builds and the discriminant
//! is never within eleven orders of magnitude of zero — the underdamped and overdamped regimes are
//! separated by a factor of `k²Q` that is either much less or much more than one, never near it.
//! The bar is stated in the parity file as agreement on the predicate, not on the discriminant.

/// Seconds of decay per unit rate for a 60 dB drop: `3 ln 10 ≈ 6.9078`. `T60 = this / σ_eff`.
///
/// `3.0 * ln(10)`, evaluated the way the original evaluates it, so the constant is whatever that
/// product rounds to rather than a transcribed decimal.
pub fn t60_seconds_per_rate() -> f64 {
    3.0 * 10.0f64.ln()
}

/// Positive eigenvalue `p² = (4/h²)sin²(mπ/2N)` of the discrete second difference on mode `m`.
pub fn spatial_eigenvalue_p2(n: i64, h: f64, m: i64) -> f64 {
    (4.0 / (h * h))
        * (m as f64 * std::f64::consts::PI / (2 * n) as f64)
            .sin()
            .powi(2)
}

/// Continuum modal loss rate `2(σ₀ + σ₁β²)` with `β = mπ/L`.
pub fn modal_loss_rate_continuum(
    _c: f64,
    l: f64,
    _kappa: f64,
    sigma0: f64,
    sigma1: f64,
    m: i64,
) -> f64 {
    let beta2 = (m as f64 * std::f64::consts::PI / l).powi(2);
    2.0 * (sigma0 + sigma1 * beta2)
}

/// The three coefficients of the per-mode quadratic `a z² + b z + c = 0`, plus `Q`.
///
/// Inserting `u^n = z^n sin(mπx/L)` into the damped θ-scheme gives this quadratic in the
/// amplification factor. Returned as a tuple so the decay, the rate and the underdamped predicate
/// all read the *same* three numbers — the original factors it out for that reason and the port
/// keeps the factoring, because recomputing them per caller is a place two spellings could drift.
#[allow(clippy::too_many_arguments)]
fn decay_roots_ac(
    c: f64,
    l: f64,
    n: i64,
    kappa: f64,
    k: f64,
    theta: f64,
    sigma0: f64,
    sigma1: f64,
    m: i64,
) -> (f64, f64, f64, f64) {
    let h = l / n as f64;
    let p2 = spatial_eigenvalue_p2(n, h, m);
    let q = c * c * p2 + kappa * kappa * p2 * p2;
    let sigma_eff = sigma0 + sigma1 * p2;
    let base = 1.0 + theta * k * k * q;
    let a = base + sigma_eff * k;
    let b = -2.0 + (1.0 - 2.0 * theta) * k * k * q;
    let cc = base - sigma_eff * k;
    (a, b, cc, q)
}

/// Per-step decay factor `g = c/a` of mode `m` — the ratio the envelope test measures.
#[allow(clippy::too_many_arguments)]
pub fn discrete_damped_mode_decay(
    c: f64,
    l: f64,
    n: i64,
    kappa: f64,
    k: f64,
    theta: f64,
    sigma0: f64,
    sigma1: f64,
    m: i64,
) -> f64 {
    let (a, _b, cc, _q) = decay_roots_ac(c, l, n, kappa, k, theta, sigma0, sigma1, m);
    cc / a
}

/// Continuous decay rate `-ln(g)/k` (s⁻¹), or `+∞` when `g ≤ 0`.
///
/// A non-positive `g` means the mode has been damped past a sign flip and is no longer a clean
/// exponential; the original returns `inf` there rather than a NaN so the caller can see it.
#[allow(clippy::too_many_arguments)]
pub fn discrete_damped_mode_rate(
    c: f64,
    l: f64,
    n: i64,
    kappa: f64,
    k: f64,
    theta: f64,
    sigma0: f64,
    sigma1: f64,
    m: i64,
) -> f64 {
    let g = discrete_damped_mode_decay(c, l, n, kappa, k, theta, sigma0, sigma1, m);
    if g <= 0.0 {
        return f64::INFINITY;
    }
    -g.ln() / k
}

/// Whether mode `m` oscillates rather than creeping — `b² - 4ac < 0`.
#[allow(clippy::too_many_arguments)]
pub fn discrete_damped_mode_is_underdamped(
    c: f64,
    l: f64,
    n: i64,
    kappa: f64,
    k: f64,
    theta: f64,
    sigma0: f64,
    sigma1: f64,
    m: i64,
) -> bool {
    let (a, b, cc, _q) = decay_roots_ac(c, l, n, kappa, k, theta, sigma0, sigma1, m);
    b * b - 4.0 * a * cc < 0.0
}

/// Solve for `(σ₀, σ₁)` from two `(frequency, T60)` targets.
///
/// `σ_eff(f) = T60_SECONDS_PER_RATE / T60`, and `β²(ω)` comes from the continuum dispersion
/// relation — for a stiff string the positive root of `κ²β⁴ + c²β² - ω² = 0`, and for `κ = 0` the
/// degenerate `ω²/c²`. Two targets give a 2×2 system with one solution.
///
/// Refuses a pair implying negative loss, which is what "T60 rising with frequency" asks for and
/// which no passive string can do.
pub fn loss_coefficients_from_t60(
    c: f64,
    _l: f64,
    kappa: f64,
    f1: f64,
    t60_1: f64,
    f2: f64,
    t60_2: f64,
) -> Result<(f64, f64), String> {
    if f1 <= 0.0 || f2 <= 0.0 || t60_1 <= 0.0 || t60_2 <= 0.0 {
        return Err("frequencies and T60s must be positive.".to_string());
    }
    if f1 == f2 {
        return Err("need two distinct frequencies to separate sigma0 from sigma1.".to_string());
    }
    let beta2_of_f = |f: f64| -> f64 {
        let omega2 = (2.0 * std::f64::consts::PI * f).powi(2);
        if kappa == 0.0 {
            return omega2 / (c * c);
        }
        let disc = c.powi(4) + 4.0 * kappa.powi(2) * omega2;
        (-(c.powi(2)) + disc.sqrt()) / (2.0 * kappa.powi(2))
    };
    let (b1, b2) = (beta2_of_f(f1), beta2_of_f(f2));
    let s1 = t60_seconds_per_rate() / t60_1;
    let s2 = t60_seconds_per_rate() / t60_2;
    let sigma1 = (s2 - s1) / (b2 - b1);
    let sigma0 = s1 - sigma1 * b1;
    if sigma0 < 0.0 || sigma1 < 0.0 {
        return Err(format!(
            "targets imply negative loss (sigma0={sigma0:.4}, sigma1={sigma1:.4}); \
             pick T60 decreasing with frequency."
        ));
    }
    Ok((sigma0, sigma1))
}
