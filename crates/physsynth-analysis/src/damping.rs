//! The damped string's decay oracles — `physsynth/analysis/damping.py`, transcribed.
//!
//! Model #3's money test: a frequency-dependent loss term gives every mode its own decay rate, and
//! the per-mode `g_m` predicted here is what a run's measured envelope is compared against. Five
//! Python test files stood on it, and `spatial_eigenvalue_p2` alone was imported by four of the
//! string family's suites directly.
//!
//! The Python module's docstrings were the physics, and this header documented only what the
//! translation had to decide. The Python bodies went at deletion 7 (`b1d37c8`) and the package at
//! retirement plan §48, so the derivations are carried here — the module docstring under "The
//! physics", each function's on the item of the same name — verbatim from `b1d37c8^`, save for
//! cross-references respelled natively (retirement plan §53).
//!
//! # The physics
//!
//! Frequency-dependent damping oracles for the damped stiff string (model #3, HANDOFF §5 row 3).
//!
//! The damped stiff string adds a frequency-dependent loss term `+2 sigma1 u_txx` to model #2:
//!
//! ```text
//! u_tt = c^2 u_xx - kappa^2 u_xxxx - 2 sigma0 u_t + 2 sigma1 u_txx.
//! ```
//!
//! In the modal domain `delta_xx -> -p^2` so the loss becomes a velocity term with a
//! mode-dependent coefficient: a mode decays in **energy** as `exp(-2 sigma_eff t)` with
//! `sigma_eff = sigma0 + sigma1 * (wavenumber)^2` — high modes die faster, the physically-correct
//! ordering real strings have (and model #2's frequency-*independent* loss got backwards).
//!
//! This module is the single source of truth for that decay relation, in three layers:
//!
//! - the **continuum** rate `2 sigma_eff` from the true wavenumber `beta = m pi / L` — the physics
//!   the scheme should approach as the grid refines;
//! - the **discrete** per-step energy factor `g_m` of the implicit theta-scheme — the *exact*
//!   thing a simulated single mode decays by, the tight test oracle. It includes the
//!   theta-scheme's `1/(1+theta Q k^2)` rate-suppression (the model #2 artifact), so the per-mode
//!   decay test stays tight without pretending the artifact is gone;
//! - the **T60 -> (sigma0, sigma1)** inversion — a luthier specifies decay times, this returns the
//!   loss coefficients (continuum mapping; the round-trip lands a few % off, a documented demo).
//!
//! Independent of the core (an oracle must not depend on the code it validates; in the Python,
//! pure NumPy). See `docs/dev/damped-string-plan.md`.
//!
//! # The one thing here that is not arithmetic
//!
//! `discrete_damped_mode_rate` takes a logarithm, and everything else in the file is `+ - * / √`.
//! So the exactness story is short: the decay *ratio* `g = c/a` is exact — both sides built it out
//! of four IEEE-754 operations from the same inputs — and the *rate* `-ln(g)/k` was a tolerance
//! port, because `np.log` is NumPy's own CPU-dispatched kernel and `f64::ln` is the platform's.
//! The parity file asserted the first as equality and the second at 1e-15 relative until it was
//! deleted with the Python bodies (plan §44), which is the same split plan §36 drew through
//! `spectrum.py` — exact on the axis that decides, tolerant on the axis that is measured.
//!
//! `discrete_damped_mode_is_underdamped` returns a **bool**, which by §25's rule is a discrete
//! output and would need a margin measured before it could be ported. It has one, and it is not
//! tight: `b² - 4ac < 0` was compared over the configurations the suite built and the
//! discriminant was never within eleven orders of magnitude of zero — the underdamped and
//! overdamped regimes are separated by a factor of `k²Q` that is either much less or much more
//! than one, never near it. The parity file's bar was agreement on the predicate, not on the
//! discriminant.

/// Seconds of decay per unit rate for a 60 dB drop: `3 ln 10 ≈ 6.9078`. `T60 = this / σ_eff`.
///
/// A 60 dB amplitude drop is a factor 10^-3; amplitude ~ exp(-sigma_eff t), so
/// exp(-sigma_eff T60) = 10^-3  =>  T60 = 3 ln(10) / sigma_eff. (Energy, being amplitude^2, drops
/// 60 dB in T60/2, but T60 is conventionally an amplitude/level figure — keep it amplitude-based.)
///
/// `3.0 * ln(10)`, evaluated the way the original evaluates it, so the constant is whatever that
/// product rounds to rather than a transcribed decimal.
pub fn t60_seconds_per_rate() -> f64 {
    3.0 * 10.0f64.ln()
}

/// Discrete eigenvalue `p^2 = (4/h^2) sin^2(m pi / 2N)` of `-delta_xx` for mode `m`.
///
/// `sin(m pi x / L)` is an exact eigenvector of the second-difference matrix with eigenvalue
/// `-p^2` (and of the biharmonic `(delta_xx)^2` with `+p^4`). Tends to the continuum
/// `beta^2 = (m pi / L)^2` as `h -> 0`.
pub fn spatial_eigenvalue_p2(n: i64, h: f64, m: i64) -> f64 {
    (4.0 / (h * h))
        * (m as f64 * std::f64::consts::PI / (2 * n) as f64)
            .sin()
            .powi(2)
}

/// Continuum **energy** decay rate `2 sigma_eff = 2(sigma0 + sigma1 beta_m^2)` (1/s).
///
/// `E_m(t) ~ E_m(0) exp(-rate * t)` with `beta_m = m pi / L` the true wavenumber. `kappa` is
/// unused here (the loss rate depends on the wavenumber, not the stiffness; stiffness enters only
/// via the oscillation frequency) — kept in the signature so callers pass the full parameter set
/// symmetrically with the discrete oracle. The discrete scheme approaches this as the grid
/// refines.
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
/// From `u^n = z^n sin(m pi x / L)` in the implicit theta-scheme with both loss terms:
/// `a = 1 + theta k^2 Q + sigma_eff k`, `b = -2 + (1-2theta) k^2 Q`,
/// `cc = 1 + theta k^2 Q - sigma_eff k`, with `Q = c^2 p^2 + kappa^2 p^4` and
/// `sigma_eff = sigma0 + sigma1 p^2`.
///
/// Returned as a tuple so the decay, the rate and the underdamped predicate
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

/// Exact per-step **energy** decay factor `g_m = cc/a` of mode `m` (implicit theta-scheme).
///
/// For an underdamped mode the two roots are a complex-conjugate pair with `|z|^2 = cc/a`, so the
/// energy (amplitude squared) is multiplied by
///
/// ```text
/// g_m = (1 + theta k^2 Q - sigma_eff k) / (1 + theta k^2 Q + sigma_eff k)  in (0, 1]
/// ```
///
/// every step (`E_m(t) = E_m(0) g_m^(t/k)`). This is the *exact* thing a simulated single mode
/// decays by — the tight test oracle. It keeps the theta-scheme's `1/(1+theta Q k^2)` factor (the
/// model #2 high-mode rate-suppression), so the test is tight without claiming that artifact is
/// gone. `sigma0 = sigma1 = 0` gives `g_m = 1` (lossless).
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

/// Per-second **energy** decay rate `Gamma_m = -(1/k) ln g_m` (1/s); 0 when lossless.
///
/// Tends to the continuum `2(sigma0 + sigma1 beta^2)` as `k, h -> 0`. Not monotone in `m`
/// forever: the numerator grows like `p^2` but the `1/(1+theta Q k^2)` denominator like `p^4`
/// with stiffness, so the rate turns over at very high modes (see
/// `docs/dev/damped-string-plan.md`).
///
/// Returns `+∞` when `g ≤ 0`. A non-positive `g` means the mode has been damped past a sign flip
/// and is no longer a clean exponential; the original returned `inf` there rather than a NaN so
/// the caller can see it.
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

/// True if mode `m` is underdamped (complex-conjugate roots, `b^2 - 4 a cc < 0`).
///
/// The `g_m = cc/a` energy oracle assumes a clean oscillatory decay; a heavily overdamped mode
/// has two distinct real roots and is not a single exponential. Tests pick modes where this
/// holds.
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

/// Invert two `(frequency, T60)` targets to `(sigma0, sigma1)` (continuum mapping).
///
/// A luthier specifies decay times; this returns the loss coefficients. With amplitude decay
/// `exp(-sigma_eff t)` a 60 dB drop takes `T60 = T60_SECONDS_PER_RATE / sigma_eff`
/// ([`t60_seconds_per_rate`]), so `sigma_eff(f) = T60_SECONDS_PER_RATE / T60`. Using
/// `beta^2(omega)` from the continuum dispersion `omega^2 = c^2 beta^2 + kappa^2 beta^4` gives
/// the linear system
///
/// ```text
/// [1, beta1^2; 1, beta2^2] @ [sigma0; sigma1] = [sigma_eff1; sigma_eff2].
/// ```
///
/// **Pick targets with T60 decreasing in frequency** (`f2 > f1` and `T60_2 < T60_1`) so
/// `sigma0, sigma1 >= 0`. This is a *continuum* inversion: a simulation decays at the discrete
/// rate (with the theta-factor), so the round-trip lands a few % off — a physics demo, not a
/// tight test. Refuses (`Err`) if the targets are degenerate (equal frequencies) or imply a
/// negative coefficient.
///
/// `beta^2` is the positive root of `κ²β⁴ + c²β² - ω² = 0`, and for `κ = 0` the degenerate
/// `ω²/c²`. Negative loss is what "T60 rising with frequency" asks for, and no passive string
/// can do it.
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
