//! Complete and Jacobian elliptic functions — the Duffing oracle's exact solution.
//!
//! `analysis/duffing.py` is the exact solution of `q'' + ω₀²q + εq³ = 0`, which is what makes the
//! tension-modulated string (model #9) and the geometrically exact string (model #10) testable
//! against something other than themselves. Two SciPy names carry it: `ellipk(m)`, the quarter
//! period, and `ellipj(u, m)`, whose `cn` *is* the waveform. Both are transcribed here for the
//! reason [`crate::bessel`] gives at length — a different-but-equally-good elliptic integral would
//! move this project's acceptance numbers silently, since every physics bar downstream is
//! percentage-level and would absorb the shift without a word.
//!
//! Both use SciPy's *parameter* convention: `m = k²`, not the modulus `k`. Getting that wrong is
//! not a rounding error, it is a different function, and `duffing.py`'s own docstring says so.
//!
//! # Measured against SciPy before this was written
//!
//! * `ellipk`: **3.5e-16 relative**, worst over `m ∈ [0, 0.999]` at 500 points.
//! * `ellipj`: **bit-identical** at the small arguments (`|u| ≲ 2`), growing to 1.7e-14 absolute
//!   at `|u| = 18` and 1.6e-12 at `|u| = 2000`, worst over `m ∈ [0, 0.95]`.
//!
//! That growth is the finding worth stating, because it looks like a defect and is not. `sn`, `cn`
//! and `dn` are oscillations of unit amplitude in `u`, so an error of one ulp *in the argument* —
//! which is all `2ⁿ aₙ u` can promise — is an absolute error of about `|u| · 1e-16` in the result,
//! and there is nothing an implementation can do about it: the function has genuinely lost that
//! much information by the time it is asked. Cephes descends the same Landen sequence and pays the
//! same price. The bar is therefore stated as `1e-15 · (1 + |u|)` absolute, and the callers clear
//! it by orders: `duffing_displacement` is used as a convergence oracle whose own errors are
//! `O(h²)` — around 1e-6 at the finest grid its tests build — against an oracle disagreement of
//! 1e-13 at the arguments those tests reach.
//!
//! One case is exactly reproduced rather than approximately, and a test pins it: at `m = 0` the
//! Landen sequence terminates before its first step (`c₀ = √0 = 0`), `φ` collapses to `u`, and
//! `cn(u, 0) = cos(u)` on the nose — which is what `tests/test_tension_string.py` compares
//! `duffing_displacement(t, A, ω₀², 0)` against at `atol = 1e-14`.

use std::f64::consts::PI;

/// The complete elliptic integral of the first kind `K(m)`, by the arithmetic–geometric mean.
///
/// `K(m) = π / (2·AGM(1, √(1-m)))`, which converges quadratically — five iterations reach a double's
/// precision from any `m` this project uses, and the loop stops when the two sequences agree to
/// `1e-17` relative rather than on a count.
///
/// Two edges are handled explicitly rather than left to the loop, and both were caught by a native
/// bar rather than by reading. `m = 1` makes `b` exactly zero, where `AGM(a, 0) = 0` and `K = ∞` —
/// but the iteration only *halves* `a` each step, so sixty passes leave it at `8.7e-19` and the
/// quotient comes back as a finite `3.6e18` instead of the `inf` SciPy returns. `m > 1` needs no
/// help: `√(1-m)` is NaN, the convergence test is false for NaN so the loop runs out, and the
/// result is NaN, which is also what SciPy returns. Negative `m` is ordinary and works as written.
pub fn ellipk(m: f64) -> f64 {
    let (mut a, mut b) = (1.0, (1.0 - m).sqrt());
    if b == 0.0 {
        return f64::INFINITY;
    }
    for _ in 0..60 {
        if (a - b).abs() <= 1e-17 * a.abs() {
            break;
        }
        let a_next = 0.5 * (a + b);
        b = (a * b).sqrt();
        a = a_next;
    }
    PI / (a + b)
}

/// The Jacobian elliptic functions `(sn, cn, dn)` at `u` with parameter `m` — SciPy's `ellipj`
/// without the amplitude `ph`, which nothing in this project reads.
///
/// The descending Landen transformation of Abramowitz & Stegun 16.4: build the AGM sequence
/// `a₀ = 1, c₀ = √m` down to `cₙ ≈ 0`, take `φ = 2ⁿ aₙ u`, then walk the sequence back up through
/// `φ_{k-1} = ½(φ_k + arcsin(c_k sin φ_k / a_k))`. `sn = sin φ`, `cn = cos φ`,
/// `dn = √(1 - m sin²φ)`.
///
/// `dn` is computed from the definition rather than from the descent, which is deliberate: it is
/// the branch that stays correct at `m → 1` where `cn → sech` and the sequence needs many more
/// steps than the `60` cap allows.
pub fn ellipj(u: f64, m: f64) -> (f64, f64, f64) {
    // The separatrix. At `m = 1` the descent degenerates — every step halves `c` and it never
    // reaches the cutoff — and the functions become hyperbolic: `sn → tanh`, `cn = dn → sech`.
    // Nothing in this project reaches it (`duffing_elliptic_parameter` is bounded above by ½ for
    // any positive `ω₀²`), which is exactly why it is written down: an edge no caller visits is an
    // edge no test visits either, and the loop's answer here is plausible and wrong.
    if m == 1.0 {
        return (u.tanh(), 1.0 / u.cosh(), 1.0 / u.cosh());
    }
    let mut a = vec![1.0f64];
    let mut c = vec![m.sqrt()];
    let mut b = (1.0 - m).sqrt();
    let mut n = 0usize;
    while c[n].abs() > 1e-17 && n < 60 {
        a.push(0.5 * (a[n] + b));
        c.push(0.5 * (a[n] - b));
        b = (a[n] * b).sqrt();
        n += 1;
    }

    let mut phi = (2.0f64).powi(n as i32) * a[n] * u;
    for k in (1..=n).rev() {
        phi = 0.5 * (phi + (c[k] * phi.sin() / a[k]).asin());
    }
    let (s, cs) = (phi.sin(), phi.cos());
    (s, cs, (1.0 - m * s * s).sqrt())
}
