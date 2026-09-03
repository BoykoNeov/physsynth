//! The closed-form oracles — `physsynth/analysis/modal.py`, transcribed.
//!
//! Thirty-one functions and one job: say where a mode *should* be, from theory, so a simulation
//! can be told how far off it is. Every modal test in this project is a comparison between a
//! number from here and a number from [`crate::spectrum`], and the two files are the whole
//! instrument. Fourteen test files import this module.
//!
//! # Why this is a tolerance port and what "tolerance" means here
//!
//! [`crate::spectrum`] is the one file in `analysis/` whose output is a *decision* — which peak —
//! and it was ported alone for that reason (plan §36). Everything here returns a **number**, and
//! every consumer compares it through a cents bar or a relative bar of 1e-9 at the very tightest.
//! So exact reproduction is not the target and claiming it would be dishonest: seven of these
//! functions call `arcsin`, `arccos`, `log2` or `sqrt`, and NumPy computes transcendentals with its
//! own CPU-dispatched kernels rather than the platform libm (plan §22.1, and the CI incident where
//! one failure became eighteen on unchanged code). A bit-identity assertion over `arcsin` is a
//! claim about which machine ran CI.
//!
//! What *is* claimed, and what the parity file asserts:
//!
//! * **The pure arithmetic is exact.** `harmonic_frequencies`, `inharmonicity_B`,
//!   `free_plate_freq_from_lambda`, `free_plate_coupling_form`, `bore_resonance_frequencies`,
//!   `free_plate_ffff_square_lambdas`, `rectangular_discrete_eigenvalues`'s `4/h²` factor — `+ - *
//!   /` and nothing else, which IEEE-754 pins. These are asserted equal, not close.
//! * **`cents` on identical inputs is exact**, because both sides reach the same `log2` for the
//!   ratio only when the ratio is itself bit-identical — so the assertion is on `cents(f, f)` and
//!   on ratios built by exact arithmetic, which is the anchor plan §35.3 said the *tests* own.
//! * **Everything with a transcendental in it is a tolerance port at 1e-15 relative**, which is
//!   two or three ulps and far inside every bar downstream.
//! * **The two root-finds are the interesting half** and have their own accounting below.
//!
//! # The two root-finds, and the margin measured before either was written
//!
//! `free_free_beam_betaL` and `free_circular_plate_lambda_roots` are the only functions here that
//! *search*, and a search can fail in a way a tolerance cannot describe: it can return a different
//! number of answers. Both were measured against the Python original first (plan §36.2's
//! discipline — measure the margin, then decide whether the port is licensed):
//!
//! * **The beam.** `cos(x) - sech(x)` on `(iπ, (i+1)π)`, one root per bracket by construction, and
//!   the bracket endpoints are exact multiples of `π`. There is nothing to decide; the port is the
//!   root-find and the agreement is `brentq`'s tolerance.
//! * **The free circular plate.** This one *does* decide something: it scans 20,000 points and
//!   keeps a bracket wherever `vs[i] * vs[i+1] < 0`. A sign that flips in the last bit adds or
//!   drops a root, and a **missing** root is the dangerous direction — the docstring's three
//!   self-checks (rigid-body annihilation, the Rayleigh quotient returning `λ⁴`, the saddle bound)
//!   all catch a *spurious* root and none of them catches an absent one.
//!
//!   Measured over `ν = 0.3`, `n = 0..8`, all 20,000 scan points, comparing each `|det|` against
//!   the cancellation `|m₀₀m₁₁| + |m₀₁m₁₀|` that produced it, and excluding the two points either
//!   side of a genuine crossing (where the ratio goes to zero because the function really is zero):
//!   the worst ratio anywhere is **4.6e-6**, at `λ = 0.0105, n = 8`. A Bessel routine agreeing to
//!   1e-15 can move the determinant by about 1e-15 of that cancellation scale, so the margin is
//!   **~5e9×**. Unlike §36's separation test, whose margin was exactly zero, this decision is not
//!   on a knife edge and the port does not need the arithmetic to be identical to be safe.
//!
//!   The scan grid itself is transcribed rather than tidied for the same reason §36 transcribed
//!   `1.0/(nfft*(1.0/fs))`: NumPy's `linspace` is `i·step + start` with the last point *assigned*
//!   `stop`, not `start + i*(stop-start)/(n-1)` evaluated at `i = n-1`, and the two differ in the
//!   last bit at most grid sizes.

use crate::bessel::{ivp, jn_zeros, jvp};
use crate::root::brentq;
use std::f64::consts::PI;

/// `np.linspace(start, stop, num)` with `endpoint=True`, spelled the way NumPy spells it.
///
/// `step = (stop - start) / (num - 1)`, values are `i * step + start`, and the final element is
/// **assigned** `stop` rather than computed. That last detail is the one that matters: the computed
/// value differs from `stop` in the last bit for most grids, and here the grid feeds a sign test.
fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    if num == 0 {
        return Vec::new();
    }
    if num == 1 {
        return vec![start];
    }
    let step = (stop - start) / (num - 1) as f64;
    let mut v: Vec<f64> = (0..num).map(|i| i as f64 * step + start).collect();
    v[num - 1] = stop;
    v
}

/// Continuous fixed–fixed harmonic series `f_n = n c / (2L)` for `n = 1 ..= n_partials`.
pub fn harmonic_frequencies(c: f64, l: f64, n_partials: usize) -> Vec<f64> {
    (1..=n_partials).map(|n| n as f64 * c / (2.0 * l)).collect()
}

/// The `m`-th fixed–fixed spatial mode `sin(mπx/L)` sampled on grid `x`.
pub fn mode_shape(x: &[f64], l: f64, m: i64) -> Vec<f64> {
    x.iter().map(|&xi| (m as f64 * PI * xi / l).sin()).collect()
}

/// Exact oscillation frequency (Hz) of mode `m` for the explicit scheme.
///
/// `k = λL/(cN)`, then `f = arcsin(λ sin(mπ/2N)) / (πk)`. At `λ = 1` this collapses to `mc/(2L)`.
pub fn discrete_mode_frequency(c: f64, l: f64, n: i64, lam: f64, m: i64) -> f64 {
    let k = lam * l / (c * n as f64);
    (lam * (m as f64 * PI / (2 * n) as f64).sin()).asin() / (PI * k)
}

/// Inharmonicity coefficient `B = π²κ²/(c²L²)` (simply-supported stiff string).
///
/// `PI.powi(2)` rather than `PI * PI`: the original writes `np.pi ** 2`, and plan §12's finding is
/// that `h**4` and `h*h*h*h` disagree in 1,400 of 3,998 measured cases. `powi(2)` is the spelling
/// that matches NumPy's power ufunc on an exponent of two.
pub fn inharmonicity_b(c: f64, l: f64, kappa: f64) -> f64 {
    PI.powi(2) * kappa.powi(2) / (c.powi(2) * l.powi(2))
}

/// Continuum stretched partials `f_n = n f₀ √(1 + Bn²)`, `f₀ = c/(2L)`.
pub fn stiff_harmonic_frequencies(c: f64, l: f64, kappa: f64, n_partials: usize) -> Vec<f64> {
    let f0 = c / (2.0 * l);
    let b = inharmonicity_b(c, l, kappa);
    (1..=n_partials)
        .map(|n| {
            let n = n as f64;
            n * f0 * (1.0 + b * n * n).sqrt()
        })
        .collect()
}

/// Exact oscillation frequency (Hz) of mode `m` for the implicit θ-scheme stiff string.
///
/// `p² = (4/h²)sin²(mπ/2N)`, `Q = c²p² + κ²p⁴`, `s = Qk²/(4 + 4θQk²)`, `f = arcsin(√s)/(πk)`.
/// The frequency depends on `θ`, so callers pass their resonator's own.
pub fn discrete_stiff_mode_frequency(
    c: f64,
    l: f64,
    n: i64,
    kappa: f64,
    k: f64,
    m: i64,
    theta: f64,
) -> f64 {
    let h = l / n as f64;
    let p2 = (4.0 / (h * h)) * (m as f64 * PI / (2 * n) as f64).sin().powi(2);
    let q = c * c * p2 + kappa * kappa * p2 * p2;
    let s = q * k * k / (4.0 + 4.0 * theta * q * k * k);
    s.sqrt().asin() / (PI * k)
}

/// Pitch error in cents: `1200 log₂(f / f_ref)`.
pub fn cents(f: f64, f_ref: f64) -> f64 {
    1200.0 * (f / f_ref).log2()
}

/// Rectangular-membrane frequencies `f = (c/2)√((m/Lx)² + (n/Ly)²)` for each `(m, n)`.
pub fn rectangular_membrane_freqs(c: f64, lx: f64, ly: f64, modes: &[(i64, i64)]) -> Vec<f64> {
    modes
        .iter()
        .map(|&(m, n)| 0.5 * c * ((m as f64 / lx).powi(2) + (n as f64 / ly).powi(2)).sqrt())
        .collect()
}

/// The `(m, n)` rectangular mode field `sin(mπX/Lx)·sin(nπY/Ly)` on a flattened grid.
pub fn rectangular_mode_field(
    xs: &[f64],
    ys: &[f64],
    lx: f64,
    ly: f64,
    m: i64,
    n: i64,
) -> Vec<f64> {
    xs.iter()
        .zip(ys.iter())
        .map(|(&x, &y)| (m as f64 * PI * x / lx).sin() * (n as f64 * PI * y / ly).sin())
        .collect()
}

/// Discrete 2-D Dirichlet Laplacian eigenvalues `(4/h²)(sin²(mπ/2Nx) + sin²(nπ/2Ny))`.
pub fn rectangular_discrete_eigenvalues(
    h: f64,
    nx: i64,
    ny: i64,
    modes: &[(i64, i64)],
) -> Vec<f64> {
    modes
        .iter()
        .map(|&(m, n)| {
            let sx = (m as f64 * PI / (2 * nx) as f64).sin().powi(2);
            let sy = (n as f64 * PI / (2 * ny) as f64).sin().powi(2);
            (4.0 / (h * h)) * (sx + sy)
        })
        .collect()
}

/// One entry of the circular-membrane spectrum: `(m, n, frequency, degeneracy)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CircularMembraneMode {
    /// Nodal diameters.
    pub m: u32,
    /// Radial index, from 1.
    pub n: usize,
    /// Frequency in Hz.
    pub freq: f64,
    /// 1 for `m = 0`, else 2 — the cos/sin pair.
    pub degeneracy: u32,
}

/// Lowest `n_modes` circular-membrane frequencies, each tagged `(m, n, freq, degeneracy)`.
///
/// `f_{mn} = c·j_{m,n}/(2πa)`. Sorted by frequency; `m = 0` is non-degenerate, `m ≥ 1` comes as a
/// cos/sin pair. The sort is **stable**, matching Python's `list.sort`, which decides the order of
/// the exact ties a symmetric membrane can produce.
pub fn circular_membrane_freqs(
    c: f64,
    a: f64,
    n_modes: usize,
    m_max: u32,
    n_max: usize,
) -> Vec<CircularMembraneMode> {
    let mut entries = Vec::new();
    for m in 0..=m_max {
        let zeros = jn_zeros(m, n_max);
        let degeneracy = if m == 0 { 1 } else { 2 };
        for (i, &z) in zeros.iter().enumerate() {
            entries.push(CircularMembraneMode {
                m,
                n: i + 1,
                freq: c * z / (2.0 * PI * a),
                degeneracy,
            });
        }
    }
    entries.sort_by(|x, y| x.freq.partial_cmp(&y.freq).expect("frequencies are finite"));
    entries.truncate(n_modes);
    entries
}

/// Discrete temporal frequency (Hz) of a membrane eigenmode with Laplacian eigenvalue `Λ`.
///
/// `f = arccos(clip(1 - c²k²Λ/2, -1, 1)) / (2πk)`.
pub fn discrete_membrane_eigenfrequency(lambda: f64, c: f64, k: f64) -> f64 {
    let arg = 1.0 - 0.5 * (c * k).powi(2) * lambda;
    arg.clamp(-1.0, 1.0).acos() / (2.0 * PI * k)
}

/// Simply-supported rectangular-plate frequencies `f = κγ/(2π)`, `γ = (mπ/Lx)² + (nπ/Ly)²`.
pub fn rectangular_plate_freqs(kappa: f64, lx: f64, ly: f64, modes: &[(i64, i64)]) -> Vec<f64> {
    modes
        .iter()
        .map(|&(m, n)| {
            let gamma = (m as f64 * PI / lx).powi(2) + (n as f64 * PI / ly).powi(2);
            kappa * gamma / (2.0 * PI)
        })
        .collect()
}

/// Discrete plate frequency from a Laplacian eigenvalue: `Q = κ²Λ²`, `s = Qk²/(4 + 4θQk²)`.
pub fn discrete_plate_eigenfrequency(lambda_lap: f64, kappa: f64, k: f64, theta: f64) -> f64 {
    let q = kappa * kappa * lambda_lap * lambda_lap;
    let s = q * k * k / (4.0 + 4.0 * theta * q * k * k);
    s.sqrt().asin() / (PI * k)
}

/// Orthotropic plate frequencies. `q = Dx a² + 2Dxy ab + Dy b²` with `a = (m/Lx)²`, `b = (n/Ly)²`.
///
/// Returns `Err` when any mode's modal stiffness is non-positive, which is the same refusal the
/// Python original raises — `grain_cross` must exceed `-√(grain_x·grain_y)`.
pub fn orthotropic_plate_freqs(
    kappa: f64,
    lx: f64,
    ly: f64,
    modes: &[(i64, i64)],
    grain_x: f64,
    grain_cross: f64,
    grain_y: f64,
) -> Result<Vec<f64>, String> {
    let qs: Vec<f64> = modes
        .iter()
        .map(|&(m, n)| {
            let a = (m as f64 / lx).powi(2);
            let b = (n as f64 / ly).powi(2);
            grain_x * a * a + 2.0 * grain_cross * a * b + grain_y * b * b
        })
        .collect();
    if qs.iter().any(|&q| q <= 0.0) {
        return Err(NONPOSITIVE_STIFFNESS.to_string());
    }
    Ok(qs.iter().map(|&q| 0.5 * PI * kappa * q.sqrt()).collect())
}

/// The refusal both orthotropic oracles share, word for word with the Python original.
const NONPOSITIVE_STIFFNESS: &str =
    "orthotropic modal stiffness is non-positive for at least one mode; grain_cross must \
     exceed -sqrt(grain_x*grain_y).";

/// Discrete orthotropic plate frequency from the two per-axis eigenvalues.
/// Eight parameters because the Python original takes eight; a struct here would be a nicer Rust
/// signature and a worse transcription, since the binding has to accept the call the viewer and
/// the tests already make.
#[allow(clippy::too_many_arguments)]
pub fn discrete_orthotropic_plate_eigenfrequency(
    lam_x: f64,
    lam_y: f64,
    kappa: f64,
    k: f64,
    theta: f64,
    grain_x: f64,
    grain_cross: f64,
    grain_y: f64,
) -> Result<f64, String> {
    let q = grain_x * lam_x * lam_x + 2.0 * grain_cross * lam_x * lam_y + grain_y * lam_y * lam_y;
    if q <= 0.0 {
        return Err(NONPOSITIVE_STIFFNESS.to_string());
    }
    let big_q = kappa * kappa * q;
    let s = big_q * k * k / (4.0 + 4.0 * theta * big_q * k * k);
    Ok(s.sqrt().asin() / (PI * k))
}

/// Positive 1-D Dirichlet second-difference eigenvalue `(4/h²)sin²(mπh/2L)`.
pub fn dirichlet_axis_eigenvalue(m: f64, l: f64, h: f64) -> f64 {
    (4.0 / (h * h)) * (m * PI * h / (2.0 * l)).sin().powi(2)
}

/// First `n_modes` positive roots `βₙL` of the free–free frequency equation `cos(x) = sech(x)`.
///
/// Root `i` lives in `(iπ, (i+1)π)` — one per bracket, so `brentq` cannot pick the wrong one, and
/// there is no scan and no bracket decision here. `sech` is spelled `1/cosh` exactly as the
/// original does, which underflows harmlessly for large `x`. The double root at `x = 0` (the two
/// rigid-body modes) is deliberately not returned.
pub fn free_free_beam_beta_l(n_modes: usize) -> Result<Vec<f64>, String> {
    if n_modes < 1 {
        return Err("n_modes must be >= 1.".to_string());
    }
    let f = |x: f64| x.cos() - 1.0 / x.cosh();
    Ok((1..=n_modes)
        .map(|i| {
            brentq(
                f,
                i as f64 * PI,
                (i + 1) as f64 * PI,
                SCIPY_XTOL,
                SCIPY_RTOL,
                100,
            )
            .expect("cos - sech changes sign once on every (i*pi, (i+1)*pi)")
        })
        .collect())
}

/// `scipy.optimize.brentq`'s defaults, which the Python original relies on by not passing them.
const SCIPY_XTOL: f64 = 2e-12;
const SCIPY_RTOL: f64 = 8.881_784_197_001_252e-16;

/// Closed-form free–free Euler–Bernoulli bending frequencies `f = κ(βₙL)²/(2πL²)`.
pub fn free_free_beam_freqs(kappa: f64, l: f64, n_modes: usize) -> Result<Vec<f64>, String> {
    Ok(free_free_beam_beta_l(n_modes)?
        .iter()
        .map(|&bl| kappa * bl * bl / (2.0 * PI * l * l))
        .collect())
}

/// The five lowest FFFF square-plate frequency parameters, cited from Leissa/Narita.
///
/// Literal digits, so both sides are exactly equal by construction — the one function here that
/// cannot disagree.
pub fn free_plate_ffff_square_lambdas() -> Vec<f64> {
    vec![13.468, 19.596, 24.270, 34.801, 34.801]
}

/// `f = λκ/(2πa²)` — a frequency parameter turned into Hz.
pub fn free_plate_freq_from_lambda(lam: f64, kappa: f64, a: f64) -> f64 {
    lam * kappa / (2.0 * PI * a * a)
}

/// The pure-twist Rayleigh bound on a free rectangle's fundamental, `24κ√(D_xy)/(2πab)`.
pub fn free_plate_twist_bound(
    kappa: f64,
    a: f64,
    b: f64,
    grain_torsion: f64,
) -> Result<f64, String> {
    if kappa <= 0.0 || a <= 0.0 || b <= 0.0 {
        return Err("kappa, a and b must all be positive.".to_string());
    }
    if grain_torsion <= 0.0 {
        return Err(format!(
            "grain_torsion (D_xy/D_ref) must be positive — at zero the saddle carries no energy \
             and joins the rigid-body nullspace; got {grain_torsion}"
        ));
    }
    Ok(24.0 * kappa * grain_torsion.sqrt() / (2.0 * PI * a * b))
}

/// The free circular plate's frequency determinant at `λ`, for `n` nodal diameters.
///
/// `W(ρ) = A Jₙ(λρ) + B Iₙ(λρ)`; the two columns are the free-edge conditions `M_r = 0` and
/// `V_r = 0` evaluated at `ρ = 1`. `Iₙ` grows like `e^λ`, so past `λ = 1` its column is scaled by
/// `e^{-λ}` to keep the determinant finite — a positive scalar column scaling moves no root.
fn free_circular_plate_det(nu: f64, n: i32, lam: f64) -> f64 {
    let mut m = [[0.0f64; 2]; 2];
    // Column 0 is the `J` solution, column 1 the `I` one — the same two rows built from a
    // different radial function, which is what the Python original's `enumerate((jvp, ivp))` says.
    let columns: [fn(i32, f64, u32) -> f64; 2] = [jvp, ivp];
    for (col, f) in columns.iter().enumerate() {
        let (d0, d1) = (f(n, lam, 0), f(n, lam, 1) * lam);
        let (d2, d3) = (f(n, lam, 2) * lam.powi(2), f(n, lam, 3) * lam.powi(3));
        let nn = (n * n) as f64;
        m[0][col] = d2 + nu * (d1 - nn * d0);
        m[1][col] = d3 + d2 - (1.0 + nn * (2.0 - nu)) * d1 + nn * (3.0 - nu) * d0;
    }
    if lam > 1.0 {
        let s = (-lam).exp();
        m[0][1] *= s;
        m[1][1] *= s;
    }
    m[0][0] * m[1][1] - m[0][1] * m[1][0]
}

/// Roots `λ = ka` of the free circular Kirchhoff plate for `n` nodal diameters.
///
/// Scans `[1e-6, lam_max]` on `scan` points, keeps every bracket where the determinant changes
/// sign, and refines each with `brentq` at `xtol = 1e-13`. Roots at or below `1e-3` are dropped:
/// `λ = 0` is a root for `n = 0, 1` (the rigid-body translation and tilt) and the caller wants
/// elastic modes. That filter is also what makes the near-origin end of the scan harmless — the
/// determinant there is astronomically small (`1e-187` at `n = 8`) but its sign is well determined,
/// and any bracket it produced would be discarded anyway.
pub fn free_circular_plate_lambda_roots(
    nu: f64,
    n: i32,
    lam_max: f64,
    scan: usize,
) -> Result<Vec<f64>, String> {
    if !(-1.0 < nu && nu < 0.5) {
        return Err(format!(
            "nu (Poisson's ratio) must be in (-1, 1/2), got {nu}"
        ));
    }
    if n < 0 {
        return Err(format!("n (nodal diameters) must be >= 0, got {n}"));
    }
    let xs = linspace(1e-6, lam_max, scan);
    let vs: Vec<f64> = xs
        .iter()
        .map(|&x| free_circular_plate_det(nu, n, x))
        .collect();
    let mut out = Vec::new();
    for i in 0..xs.len().saturating_sub(1) {
        if vs[i].is_finite() && vs[i + 1].is_finite() && vs[i] * vs[i + 1] < 0.0 {
            let r = brentq(
                |t| free_circular_plate_det(nu, n, t),
                xs[i],
                xs[i + 1],
                1e-13,
                SCIPY_RTOL,
                100,
            )
            .expect("the bracket was selected by a sign change");
            out.push(r);
        }
    }
    out.retain(|&v| v > 1e-3);
    Ok(out)
}

/// Lowest `n_modes` elastic frequency parameters `Λ = λ²` of a free circular plate, with
/// multiplicity, plus each mode's nodal-diameter count.
///
/// Every `n ≥ 1` mode is a degenerate cos/sin pair and is emitted **twice**. Omitting that
/// misaligns every comparison past the first entry and reads as a 26–42% physics error with no bug
/// behind it, which is why the expansion is here and not at the call site.
pub fn free_circular_plate_lambdas(
    nu: f64,
    n_modes: usize,
    n_max: u32,
) -> Result<(Vec<f64>, Vec<i64>), String> {
    if n_modes < 1 {
        return Err(format!("n_modes must be >= 1, got {n_modes}"));
    }
    let mut found: Vec<(f64, i64)> = Vec::new();
    for n in 0..=n_max as i32 {
        for lam in free_circular_plate_lambda_roots(nu, n, 14.0, 20000)? {
            let copies = if n == 0 { 1 } else { 2 };
            for _ in 0..copies {
                found.push((lam * lam, n as i64));
            }
        }
    }
    // Python's `list.sort()` on tuples is lexicographic and stable; a degenerate pair has an
    // identical first element, so the second decides nothing and the stability decides nothing
    // either — but matching it costs one word and removes the question.
    found.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .expect("frequency parameters are finite")
            .then(a.1.cmp(&b.1))
    });
    if found.len() < n_modes {
        return Err(format!(
            "only {} modes below the root-scan ceiling; raise lam_max or n_max.",
            found.len()
        ));
    }
    found.truncate(n_modes);
    Ok((
        found.iter().map(|&(v, _)| v).collect(),
        found.iter().map(|&(_, n)| n).collect(),
    ))
}

/// The pure-saddle bound on a free disk's fundamental, `√(48(1-ν))` — no Bessel function in it.
pub fn free_circular_plate_saddle_bound(nu: f64) -> Result<f64, String> {
    if !(-1.0 < nu && nu < 0.5) {
        return Err(format!(
            "nu (Poisson's ratio) must be in (-1, 1/2), got {nu}"
        ));
    }
    Ok((48.0 * (1.0 - nu)).sqrt())
}

/// The free plate's cross-coupling quadratic form `4·D_xy·h²(Nx-1)(Ny-1)`.
pub fn free_plate_coupling_form(
    grain_coupling: f64,
    h: f64,
    nx: i64,
    ny: i64,
) -> Result<f64, String> {
    if h <= 0.0 {
        return Err("h must be positive.".to_string());
    }
    if nx < 2 || ny < 2 {
        return Err(format!("Nx and Ny must both be >= 2, got ({nx}, {ny})"));
    }
    Ok(4.0 * grain_coupling * h * h * (nx - 1) as f64 * (ny - 1) as f64)
}

/// Continuum acoustic-tube resonances — the clarinet oracle.
///
/// `closed-open` gives the odd series `(2n-1)c₀/(4L)`; both `open-open` and `closed-closed` give
/// the full series `nc₀/(2L)`.
pub fn bore_resonance_frequencies(
    c0: f64,
    l: f64,
    n_partials: usize,
    boundary: &str,
) -> Result<Vec<f64>, String> {
    match boundary {
        "closed-open" => Ok((1..=n_partials)
            .map(|n| (2 * n - 1) as f64 * c0 / (4.0 * l))
            .collect()),
        "open-open" | "closed-closed" => Ok((1..=n_partials)
            .map(|n| n as f64 * c0 / (2.0 * l))
            .collect()),
        other => Err(format!(
            "boundary must be 'closed-open', 'open-open', or 'closed-closed', got '{other}'"
        )),
    }
}

/// Discrete bore frequency from `ω²`: `arcsin(clip(½k√(max(ω²,0)), -1, 1)) / (πk)`.
pub fn discrete_bore_eigenfrequency(omega2: f64, k: f64) -> f64 {
    let arg = 0.5 * k * omega2.max(0.0).sqrt();
    arg.clamp(-1.0, 1.0).asin() / (PI * k)
}

/// Discrete beam frequency from a biharmonic eigenvalue `μ`: `Q = κ²μ`, `s = Qk²/(4 + 4θQk²)`.
pub fn discrete_beam_eigenfrequency(mu: f64, kappa: f64, k: f64, theta: f64) -> f64 {
    let q = kappa * kappa * mu;
    let s = q * k * k / (4.0 + 4.0 * theta * q * k * k);
    s.sqrt().asin() / (PI * k)
}
