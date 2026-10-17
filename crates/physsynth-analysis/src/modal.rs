//! The closed-form oracles — `physsynth/analysis/modal.py`, transcribed.
//!
//! Thirty-one functions and one job: say where a mode *should* be, from theory, so a simulation
//! can be told how far off it is. Every modal test in this project is a comparison between a
//! number from here and a number from [`crate::spectrum`], and the two files are the whole
//! instrument. Fourteen Python test files imported the original.
//!
//! That module's docstrings were the physics, and this header documented only what the
//! translation had to decide. The Python bodies went at deletion 7 (`b1d37c8`) and the package at
//! retirement plan §48, so the derivations are carried here — the module docstring under "The
//! physics", each function's on the item of the same name — verbatim from `b1d37c8^`, save for
//! cross-references respelled natively and SciPy plumbing dated (retirement plan §53).
//!
//! **The measured figures in the carried text are the Python implementation's** — the free disk's
//! pair splittings and the coupling form's agreement across grids. They were measured before the
//! port and have not been re-measured on this one; re-measure, and save the native print, before
//! quoting any of them as a Rust result.
//!
//! # The physics
//!
//! *(The module docstring was written at the first milestone, when the file held only the ideal
//! string's three oracles, and was never widened as the other twenty-eight arrived. It is carried
//! as it stood; every later oracle's derivation is on its own item.)*
//!
//! Analytic modal oracles for the ideal string.
//!
//! These are the closed-form references the FDTD output is validated against (HANDOFF §5, §6):
//!
//! - the continuous harmonic series `f_n = n * c / (2L)` (fixed–fixed string),
//! - the spatial mode shapes `sin(n pi x / L)`,
//! - the *exact* oscillation frequency of the discrete explicit scheme (the numerical-dispersion
//!   oracle), which lets convergence be predicted in closed form rather than only measured.
//!
//! No dependency on the core (oracles are independent of the implementation under test). In the
//! Python that meant pure NumPy; here it is the edge `tests/deps.rs` refuses.
//!
//! # Why this is a tolerance port and what "tolerance" means here
//!
//! [`crate::spectrum`] is the one file in `analysis/` whose output is a *decision* — which peak —
//! and it was ported alone for that reason (plan §36). Everything here returns a **number**, and
//! every consumer compares it through a cents bar or a relative bar of 1e-9 at the very tightest.
//! So exact reproduction was not the target and claiming it would have been dishonest: seven of
//! these functions call `arcsin`, `arccos`, `log2` or `sqrt`, and NumPy computes transcendentals
//! with its own CPU-dispatched kernels rather than the platform libm (plan §22.1, and the CI
//! incident where one failure became eighteen on unchanged code). A bit-identity assertion over
//! `arcsin` would have been a claim about which machine ran CI.
//!
//! What *was* claimed, and what the parity file asserted until it was deleted with the Python
//! bodies (plan §44) — the Python's answers on the freeze's own fixtures are in
//! `tests/analysis_frozen.rs`:
//!
//! * **The pure arithmetic is exact.** `harmonic_frequencies`, `inharmonicity_B`,
//!   `free_plate_freq_from_lambda`, `free_plate_coupling_form`, `bore_resonance_frequencies`,
//!   `free_plate_ffff_square_lambdas`, `rectangular_discrete_eigenvalues`'s `4/h²` factor — `+ - *
//!   /` and nothing else, which IEEE-754 pins. These were asserted equal, not close.
//! * **`cents` on identical inputs is exact**, because both sides reach the same `log2` for the
//!   ratio only when the ratio is itself bit-identical — so the assertion was on `cents(f, f)` and
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
//!   drops a root, and a **missing** root is the dangerous direction — the three self-checks on
//!   [`free_circular_plate_lambda_roots`] (rigid-body annihilation, the Rayleigh quotient
//!   returning `λ⁴`, the saddle bound) all catch a *spurious* root and none of them catches an
//!   absent one.
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
///
/// This is an exact eigenvector of the discrete second-difference operator, so initializing the
/// string with it yields a clean single-frequency oscillation — ideal for a convergence study.
pub fn mode_shape(x: &[f64], l: f64, m: i64) -> Vec<f64> {
    x.iter().map(|&xi| (m as f64 * PI * xi / l).sin()).collect()
}

/// Exact oscillation frequency (Hz) of mode `m` for the explicit scheme.
///
/// Solving the scheme's amplification relation gives `sin(omega k / 2) = lambda sin(m pi / 2N)`,
/// hence
///
/// ```text
/// f_m^discrete = (1 / (pi k)) * arcsin( lambda * sin(m pi / (2N)) ),    k = lambda L / (c N).
/// ```
///
/// At `lambda = 1` this collapses to the exact `m c / (2L)`; for `lambda < 1` it lies below it
/// (numerical dispersion), with the gap shrinking as O(h^2). Provided as an independent oracle
/// for the convergence test.
pub fn discrete_mode_frequency(c: f64, l: f64, n: i64, lam: f64, m: i64) -> f64 {
    let k = lam * l / (c * n as f64);
    (lam * (m as f64 * PI / (2 * n) as f64).sin()).asin() / (PI * k)
}

/// Inharmonicity coefficient `B = π²κ²/(c²L²)` (simply-supported stiff string).
///
/// The single audible stiffness parameter; for a piano `B ~ 1e-4 .. 1e-3`. Scales as `kappa^2` —
/// the relationship the B-vs-kappa sweep test pins down quantitatively.
///
/// `PI.powi(2)` rather than `PI * PI`: the original writes `np.pi ** 2`, and plan §12's finding is
/// that `h**4` and `h*h*h*h` disagree in 1,400 of 3,998 measured cases. `powi(2)` is the spelling
/// that matches NumPy's power ufunc on an exponent of two.
pub fn inharmonicity_b(c: f64, l: f64, kappa: f64) -> f64 {
    PI.powi(2) * kappa.powi(2) / (c.powi(2) * l.powi(2))
}

/// Continuum stretched partials `f_n = n f₀ √(1 + Bn²)`, `f₀ = c/(2L)`.
///
/// The simply-supported stiff-string oracle (HANDOFF §5, Bilbao Ch. 7): bending stiffness pushes
/// every partial sharp, including the fundamental (`f_1 = f0 sqrt(1 + B)`, *not* `f0`). This is
/// the "theory" the FDTD partials are measured against.
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
/// Inserting `u^n = z^n sin(m pi x / L)` (an exact discrete eigenvector) with the spatial
/// eigenvalue `p^2 = (4/h^2) sin^2(m pi / 2N)` (so `delta_xx -> -p^2` and `delta_xxxx -> +p^4`)
/// and `Q = c^2 p^2 + kappa^2 p^4` gives
///
/// ```text
/// sin^2(omega k / 2) = s = Q k^2 / (4 + 4 theta Q k^2),
/// f_m = arcsin(sqrt(s)) / (pi k).
/// ```
///
/// As `h, k -> 0` this tends to the continuum `f_n = n f0 sqrt(1 + B n^2)` for any `theta`.
/// **The frequency depends on `theta`** (the temporal scheme), so callers must pass the
/// resonator's own `theta`. At `kappa = 0` it is the implicit wave scheme's oracle (which is *not*
/// the explicit one — not exact even at `lambda = 1`).
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
///
/// `modes` is a list of `(m, n)` mode indices (both >= 1). Fixed-rim modes are
/// `sin(mπx/Lx) sin(nπy/Ly)`. This is the exact, O(h²)-clean oracle for the harness unit-test
/// geometry (the rectangle), used before the staircase error enters on the circle.
pub fn rectangular_membrane_freqs(c: f64, lx: f64, ly: f64, modes: &[(i64, i64)]) -> Vec<f64> {
    modes
        .iter()
        .map(|&(m, n)| 0.5 * c * ((m as f64 / lx).powi(2) + (n as f64 / ly).powi(2)).sqrt())
        .collect()
}

/// The `(m, n)` rectangular mode field `sin(mπX/Lx)·sin(nπY/Ly)` on a flattened grid.
///
/// An *exact* discrete eigenvector of the 5-point Laplacian (tensor product of the 1D
/// `sin(mπl/N)` eigenvector), so a single-mode initial condition stays one clean tone — the 2D
/// analogue of [`mode_shape`].
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

/// Closed-form eigenvalues `Λ_{mn}` of `-Δ_h` on a rectangle (`-L` is SPD, `Λ > 0`).
///
/// `Λ_{mn} = (4/h²)[sin²(mπ/(2Nx)) + sin²(nπ/(2Ny))]` for `m = 1..Nx-1`, `n = 1..Ny-1` — the 2D
/// tensor product of the 1D second-difference spectrum (see
/// `physsynth_core::ops::second_difference_matrix`). The assembled masked Laplacian must
/// reproduce these to machine precision, which is what proves the operator is wired correctly
/// before any continuum/Bessel comparison.
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
/// `f_{mn} = c · j_{m,n} / (2π a)` where `j_{m,n}` is the n-th positive zero of the Bessel
/// function `J_m` (mode shape `J_m(j_{m,n} r/a)·{cos,sin}(mθ)`). `m = 0` modes are non-degenerate
/// (`degeneracy = 1`); `m >= 1` come as a cos/sin pair (`degeneracy = 2`). Returned sorted by
/// frequency. `m_max, n_max` bound the search grid of zeros (raise if `n_modes` is large). The
/// drumhead oracle (HANDOFF §5 row 4).
///
/// The sort is **stable**, matching Python's `list.sort`, which decides the order of the exact
/// ties a symmetric membrane can produce.
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

/// Discrete temporal frequency (Hz) of a membrane eigenmode with Laplacian eigenvalue `Λ` (of
/// `-L`).
///
/// Inserting `u^n = z^n φ` with `L φ = -Λ φ` into the explicit scheme gives
/// `cos(ω k) = 1 − c²k²Λ/2`, hence `f = arccos(1 − c²k²Λ/2) / (2π k)` (the argument clipped to
/// `[-1, 1]`). As `k → 0` this tends to the continuum `f = c sqrt(Λ_cont)/(2π)`. The temporal
/// companion to the spatial eigenvalue test: spectrum (`Λ` of the masked Laplacian) → measurable
/// frequency.
pub fn discrete_membrane_eigenfrequency(lambda: f64, c: f64, k: f64) -> f64 {
    let arg = 1.0 - 0.5 * (c * k).powi(2) * lambda;
    arg.clamp(-1.0, 1.0).acos() / (2.0 * PI * k)
}

/// Continuum simply-supported rectangular-plate frequencies (HANDOFF §5 row 5).
///
/// With stiffness `kappa = sqrt(D/rho_s)` (D = flexural rigidity, rho_s = areal density), the
/// Navier modes are `sin(mπx/Lx) sin(nπy/Ly)` with `ω_{mn} = kappa·γ_{mn}` where
/// `γ_{mn} = (mπ/Lx)² + (nπ/Ly)²` is the Laplacian eigenvalue magnitude, hence
///
/// ```text
/// f_{mn} = kappa·γ_{mn} / (2π) = (π/2)·kappa·[(m/Lx)² + (n/Ly)²].
/// ```
///
/// Pure 4th-power in the wavenumbers (no tension term — the plate is bending-only). `modes` is a
/// list of `(m, n)` indices (both >= 1). Poisson's ratio drops out for simply-supported edges, so
/// `kappa` is the single stiffness parameter.
pub fn rectangular_plate_freqs(kappa: f64, lx: f64, ly: f64, modes: &[(i64, i64)]) -> Vec<f64> {
    modes
        .iter()
        .map(|&(m, n)| {
            let gamma = (m as f64 * PI / lx).powi(2) + (n as f64 * PI / ly).powi(2);
            kappa * gamma / (2.0 * PI)
        })
        .collect()
}

/// Discrete temporal frequency (Hz) of a plate eigenmode, implicit theta-scheme.
///
/// `lambda_lap` is the **Laplacian** eigenvalue magnitude `Λ` (of `-L`); the *biharmonic*
/// eigenvalue is `Λ²`, so the plate's modal stiffness is `Q = kappa²·Λ²` (4th power — easy to
/// under-square or double-square, so it is pinned here). Inserting `u^n = z^n φ` with
/// `B φ = Λ² φ` into `δ_tt u = -kappa² B (θ u^{n+1} + (1-2θ) u^n + θ u^{n-1})` gives
///
/// ```text
/// sin²(ω k / 2) = s = Q k² / (4 + 4 θ Q k²),   f = arcsin(sqrt(s)) / (π k).
/// ```
///
/// As `k → 0` this tends to the continuum `f = kappa·Λ/(2π)`. Unconditionally stable for
/// `θ >= 1/4` (`s <= 1`); the frequency depends on `θ` (the temporal scheme), so callers must
/// pass the resonator's own `theta`.
pub fn discrete_plate_eigenfrequency(lambda_lap: f64, kappa: f64, k: f64, theta: f64) -> f64 {
    let q = kappa * kappa * lambda_lap * lambda_lap;
    let s = q * k * k / (4.0 + 4.0 * theta * q * k * k);
    s.sqrt().asin() / (PI * k)
}

/// Continuum simply-supported **orthotropic** rectangular-plate frequencies.
///
/// The plate of a material with a grain — stiffer along one axis than across. The Navier modes
/// are still exactly `sin(mπx/Lx) sin(nπy/Ly)` (orthotropy does not mix them on a rectangle whose
/// axes are the material axes), but the frequency is no longer a function of the Laplacian
/// eigenvalue alone:
///
/// ```text
/// f_mn = (π/2) sqrt( [D_x (m/Lx)⁴ + 2H (m/Lx)²(n/Ly)² + D_y (n/Ly)⁴] / rho_s )
/// ```
///
/// written in this module's ratio convention (`g = D/D_ref`, `kappa² = D_ref/rho_s`) as
///
/// ```text
/// f_mn = (kappa/2π) · π² · sqrt( g_x a² + 2 g_h a b + g_y b² ),   a = (m/Lx)², b = (n/Ly)²
/// ```
///
/// **The factor of 2 is on the cross term, and `H = D_1 + 2 D_xy`** — see
/// `physsynth_core::plate::grain_ratios_from_material`. At `g_x = g_h = g_y = 1` this is
/// [`rectangular_plate_freqs`] exactly (`a² + 2ab + b² = (a+b)²`), which the suite asserts
/// rather than assumes.
///
/// Unlike the isotropic case, Poisson's ratio does **not** drop out — it enters through `H`.
///
/// Returns `Err` when any mode's modal stiffness is non-positive, which is the same refusal the
/// Python original raised — `grain_cross` must exceed `-√(grain_x·grain_y)`.
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

/// Discrete temporal frequency (Hz) of an **orthotropic** plate eigenmode, theta-scheme.
///
/// A separate function from [`discrete_plate_eigenfrequency`] rather than an optional argument on
/// it, because the isotropic one is parameterised by the single Laplacian eigenvalue `Λ` and this
/// one genuinely needs the two axes apart: `lam_x = (4/h²)sin²(mπh/2Lx)` and `lam_y` likewise,
/// both **positive**. The modal stiffness is
///
/// ```text
/// Q_mn = kappa² ( g_x lam_x² + 2 g_h lam_x lam_y + g_y lam_y² )
/// ```
///
/// and from there the algebra is the isotropic one verbatim:
///
/// ```text
/// sin²(ω k/2) = s = Q k² / (4 + 4 θ Q k²),   f = arcsin(sqrt(s)) / (π k).
/// ```
///
/// Because `lam_x + lam_y = Λ` and `Q` collapses to `kappa² Λ²` at `g = (1,1,1)`, this agrees
/// with [`discrete_plate_eigenfrequency`] there to machine precision.
///
/// Eight parameters because the Python original took eight; a struct would have been a nicer
/// Rust signature and a worse transcription while the binding had to accept the call the viewer
/// and the Python tests made. The binding is gone (retirement plan §51) and the signature stayed.
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
///
/// The per-axis factor the orthotropic oracle needs. `-` this is the eigenvalue of `δ_xx` on
/// `sin(mπx/L)` sampled at the interior nodes; the isotropic Laplacian eigenvalue is the *sum*
/// of the two axes' values, which is why the isotropic oracle can take one number and the
/// orthotropic one cannot. Tends to `(mπ/L)²` as `h → 0` (the O(h²) source in both).
pub fn dirichlet_axis_eigenvalue(m: f64, l: f64, h: f64) -> f64 {
    (4.0 / (h * h)) * (m * PI * h / (2.0 * l)).sin().powi(2)
}

/// First `n_modes` positive roots `β_n L` of the free-free frequency equation.
///
/// The elastic modes of a free-free Euler–Bernoulli beam satisfy `cos(βL)·cosh(βL) = 1`. The
/// roots are found by `brentq` on the overflow-safe rearrangement `cos(x) − sech(x) = 0` (`sech`
/// underflows harmlessly for large `x`); root `i` lives in `(iπ, (i+1)π)` and tends to
/// `(2i+1)π/2`. The first few are `4.730041, 7.853205, 10.995608, 14.137165, …`.
///
/// *(The Python docstring said the roots tend to `(2i+1)π/2` "from above". That holds for the
/// first (4.73004 > 4.71239) and not the second (7.85320 < 7.85398): the approach alternates, and
/// `tests/modal.rs` asserts only the proximity.)*
///
/// The double root at `x = 0` (the two rigid-body modes `u ≡ 1` and `u ≡ x`, `ω = 0`) is **not**
/// returned — those are the operator's nullspace, checked separately.
///
/// One root per bracket, so `brentq` cannot pick the wrong one, and there is no scan and no
/// bracket decision here. `sech` is spelled `1/cosh` exactly as the original does.
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

/// Closed-form free-free Euler–Bernoulli bending frequencies (the Part-0 oracle).
///
/// With stiffness `kappa = sqrt(E I / (rho A))` (same convention as the stiff string), the
/// bending PDE `u_tt = -kappa² u_xxxx` on a *free-free* beam has `ω_n = kappa·β_n²` where
/// `β_n L` are the roots of `cos(βL)·cosh(βL) = 1` ([`free_free_beam_beta_l`]), hence
///
/// ```text
/// f_n = kappa · (β_n L)² / (2π L²).
/// ```
///
/// This is a *genuine closed form* (unlike the 2D free plate), which is exactly why the beam is
/// built first: it gives a tight oracle for the free-edge stencil before the 2D corners and
/// Poisson term. Returns the lowest `n_modes` elastic frequencies (the two `ω = 0` rigid-body
/// modes excluded).
pub fn free_free_beam_freqs(kappa: f64, l: f64, n_modes: usize) -> Result<Vec<f64>, String> {
    Ok(free_free_beam_beta_l(n_modes)?
        .iter()
        .map(|&bl| kappa * bl * bl / (2.0 * PI * l * l))
        .collect())
}

/// Tabulated dimensionless frequencies of the **completely free (FFFF) square** plate, ν = 0.3.
///
/// The free plate has **no closed-form** modal oracle (unlike the simply-supported plate or the
/// free-free beam), so this is the *absolute* percent-level anchor for the free-edge plate. The
/// values are the dimensionless frequency parameters
///
/// ```text
/// λ = ω a² √(ρ_s / D) = ω a² / κ        (a = square side length, κ = √(D/ρ_s)),
/// ```
///
/// so the physical frequency is `f = λ κ / (2π a²)` ([`free_plate_freq_from_lambda`]).
///
/// Returned are the **lowest 5 elastic modes** (the 3 rigid-body modes `{1, x, y}` are excluded):
///
/// ```text
/// λ = [13.468, 19.596, 24.270, 34.801, 34.801]
/// ```
///
/// The **fundamental is the saddle/twist** (diagonal nodal lines, λ₁ ≈ 13.47) — *not* a drum-like
/// bulge; a bulge-shaped lowest mode signals a wrong (ν-dropped) operator. Modes 4 and 5 are a
/// degenerate pair (the square's symmetry), so the comparison must be by **sorted eigenvalue**,
/// not per-mode label.
///
/// Source (exact digits, not remembered): Y. Narita, "Natural Frequencies of Isotropic
/// Rectangular Plates in Improved Accuracy", *EPI International Journal of Engineering* **5**(1),
/// 2022, pp. 26–36 (DOI 10.25042/epi-ije.022022.05), Table 1, F-F-F-F, a/b = 1, 12×12 Ritz. These
/// are converged upper-bound Ritz values that *improve on* Leissa's classic 1969 monograph
/// (NASA SP-160), whose corresponding parameters (≈13.49, 19.79, 24.43, 35.02) are slightly
/// higher.
///
/// Literal digits — the one function here that could not disagree with the Python.
pub fn free_plate_ffff_square_lambdas() -> Vec<f64> {
    vec![13.468, 19.596, 24.270, 34.801, 34.801]
}

/// Convert a dimensionless free-plate frequency parameter `λ` to a physical frequency (Hz).
///
/// With `λ = ω a² √(ρ_s/D) = ω a²/κ` (`κ = √(D/ρ_s)`, `a` = square side length), the frequency
/// is
///
/// ```text
/// f = ω / (2π) = λ κ / (2π a²).
/// ```
///
/// No areal-density or thickness factor to fumble — `κ` is the single stiffness parameter (as for
/// the simply-supported plate). Pairs with [`free_plate_ffff_square_lambdas`].
pub fn free_plate_freq_from_lambda(lam: f64, kappa: f64, a: f64) -> f64 {
    lam * kappa / (2.0 * PI * a * a)
}

// -- 2D free-edge ORTHOTROPIC plate (model #5of): the free plate with a grain -----------------
//
// The free orthotropic plate has no closed-form spectrum (neither does the isotropic one) and no
// freely-citable table either, so its four bending constants are validated by four independent
// probes instead. Two of them are closed-form and live here; the other two are exact reductions
// to the shipped 1-D free beam (see docs/dev/orthotropic-free-plate-plan.md §1.1).

/// Rayleigh upper bound on the **first elastic frequency** of a free orthotropic plate (Hz).
///
/// The centred saddle `w = x·y` (coordinates from the plate's centroid) has `w_xx = w_yy = 0`
/// and `w_xy = 1`, so in the orthotropic bending energy every term dies except the torsional one
/// and the Rayleigh quotient closes in one line:
///
/// ```text
/// U = 2 D_xy a b,   ∫∫ w² dA = a³b³/144   =>   omega² <= 576 D_xy / (rho_s a² b²)
/// omega_1 <= 24 sqrt(D_xy / rho_s) / (a b),    f_1 <= 24 kappa sqrt(g_xy) / (2π a b)
/// ```
///
/// written in this module's ratio convention (`g_xy = D_xy/D_ref`, `kappa² = D_ref/rho_s`). The
/// saddle is odd about both centre lines, so on a symmetric grid it is exactly orthogonal to the
/// rigid-body space `{1, x, y}` and the bound is legitimate rather than contaminated by the zero
/// modes.
///
/// Three things this is and is not:
///
/// - **It depends on the torsional rigidity alone.** `D_x`, `D_y` and the coupling rigidity `D_1`
///   cannot appear, which is what makes it a probe of `g_xy` in isolation — and is why the wood
///   literature reads `G_xy` off a tapped free plate (Caldersmith, *Acustica* **56** (1984)
///   144–152). The default `grain_torsion = 0.5` is the isotropic `ν = 0` value; for isotropic
///   material pass `(1 - nu)/2`.
/// - **It is informative, not vacuous.** Isotropic square at `ν = 0.3`: the bound is
///   `24 sqrt(0.35) = 14.20` in `λ = ω a²/κ` units against the tabulated 13.468
///   ([`free_plate_ffff_square_lambdas`]) — a 5.4% one-term overshoot.
/// - **It is ONE-SIDED.** A uniformly too-soft operator satisfies it comfortably. It must be used
///   alongside a two-sided check (the free-beam reduction) — see
///   `docs/dev/orthotropic-free-plate-plan.md` §7.
///
/// *(The "default" is the Python keyword's; natively every argument is passed.)*
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

/// Roots `lambda = k a` of the **free circular** Kirchhoff plate for `n` nodal diameters.
///
/// The oracle the guitar-shaped plate (#5g) is validated against, **derived rather than cited** —
/// the same policy #5of adopted when its orthotropic tables turned out to be paywalled. A disk
/// exercises the identical staircased-mask machinery as a guitar outline and, unlike a guitar,
/// has a real answer.
///
/// With `w = W(r) cos(n theta)`, the regular solution is `W(rho) = A J_n(lam rho) +
/// B I_n(lam rho)` on `rho = r/a`, and `k⁴ = rho_s omega²/D` gives `omega = kappa lam²/a²`.
/// **Note the square:** the frequency parameter tabulated in the plate literature is
/// `Lambda = omega a²/kappa = lam²`, not `lam` ([`free_circular_plate_lambdas`]). Getting that
/// backwards makes a correct root look like a 2.3x error.
///
/// The two free-edge conditions at `rho = 1`, with primes `d/drho` (the `a`-powers cancel after
/// multiplying the second by `a³`):
///
/// ```text
/// M_r = 0:   W'' + nu (W' - n² W) = 0
/// V_r = 0:   W''' + W'' - [1 + n²(2-nu)] W' + n²(3-nu) W = 0
/// ```
///
/// the second assembled from the Kirchhoff shear `V_r = Q_r + (1/r) dM_rtheta/dtheta`, whose
/// `(1-nu) n²` twisting contribution is what makes the `W'` and `W` coefficients depend on `nu`
/// at all.
///
/// **This derivation is self-checked three ways**, because a plausible sign error survives most
/// checks (the tests assert all three — the first two in `physsynth-core`'s
/// `tests/plate_outline.rs`, the third in this crate's `tests/modal.rs`):
///
/// 1. **Rigid-body roots.** `W = 1` at `n = 0` and `W = rho` at `n = 1` annihilate both lines
///    identically — a translation and a tilt cost a free plate nothing.
/// 2. **The plate's own energy.** Build `W` from each root's nullvector and evaluate the bending
///    Rayleigh quotient in polar coordinates: a genuine mode returns `lam⁴` exactly. Every root
///    found here passes, so the equation has no spurious roots to filter.
/// 3. **A closed-form bound containing no Bessel function.** The pure saddle `w = xy` is
///    orthogonal to `{1, x, y}` on a disk and has `w_xx = w_yy = 0`, `w_xy = 1`, giving
///    `P = 2(1-nu) pi a²` against `∫w² = pi a⁶/24` and hence
///
///    ```text
///    Lambda_1 <= sqrt(48 (1 - nu))
///    ```
///
///    — 5.79655 at `nu = 0.3` against a derived fundamental of 5.35833, an **8.18%** one-term
///    overshoot, the same character as the free *rectangle*'s 5.4% ([`free_plate_twist_bound`],
///    which is this same probe on the other outline).
///
/// `lam = 0` is a root for `n = 0, 1` (those rigid-body modes) and is deliberately **not**
/// returned: the caller wants elastic modes.
///
/// # The search
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

/// Lowest `n_modes` elastic frequency parameters `Lambda = omega a² sqrt(rho_s/D)` of a
/// completely free circular plate, **with multiplicity**, plus each mode's nodal-diameter count.
///
/// `Lambda = lam²` from [`free_circular_plate_lambda_roots`]. At `nu = 0.3` the first entries are
/// 5.35833 (n=2), 9.00314 (n=0), 12.43899 (n=3), 20.47455 (n=1), 21.83516 (n=4).
///
/// **Multiplicity is not decoration and omitting it is a trap that looks like a physics bug.**
/// Every `n >= 1` mode is a *degenerate pair* — `cos(n theta)` and `sin(n theta)` are independent
/// and cost the same energy — while `n = 0` modes are single. A discrete spectrum contains both
/// members, so comparing it against a list that names each pair once misaligns everything past
/// the first entry and reads as 26–42% errors that grow worse under refinement. That is a
/// *comparison* failure with no bug behind it, and it is why the pairs are expanded here rather
/// than at the call site.
///
/// The degeneracy is also a **zero-valued detector** in its own right: a square grid relates a
/// pair's axis-aligned and 45° members by no symmetry, so it splits them, and the exact answer
/// for that split is zero. Measured on a staircased disk (the Python plate — see the module
/// header): 0.69%, 1.01%, 0.06%, 0.52%, 0.17%, 0.013% at N = 24…128 — shrinking but **not
/// monotone**, so assert a ceiling that shrinks with `h`, never monotonicity.
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

/// Closed-form Rayleigh bound `Lambda_1 <= sqrt(48(1-nu))` on the free circular plate.
///
/// The disk's version of [`free_plate_twist_bound`] — same trial function `w = xy`, same one-term
/// Rayleigh argument, different domain — and the one line of
/// [`free_circular_plate_lambda_roots`]'s derivation a reader can check by hand. Contains no
/// Bessel function, so it cannot share a bug with the frequency equation it validates.
pub fn free_circular_plate_saddle_bound(nu: f64) -> Result<f64, String> {
    if !(-1.0 < nu && nu < 0.5) {
        return Err(format!(
            "nu (Poisson's ratio) must be in (-1, 1/2), got {nu}"
        ));
    }
    Ok((48.0 * (1.0 - nu)).sqrt())
}

/// Exact value of the free-plate bending form on the pair `(x², y²)` — the `D_1` probe.
///
/// On `f = x²`, `g = y²` every term of the orthotropic bilinear form vanishes except the coupling
/// one (`f_yy = g_xx = 0`, `f_xy = g_xy = 0`), leaving a closed form:
///
/// ```text
/// P(x², y²) = ∫∫ D_1 (f_xx g_yy + f_yy g_xx) dA = 4 D_1 a b        (continuum)
/// ```
///
/// Discretely the collocated second differences return `2` at every interior node and `0` on the
/// respective free edges — where the *natural* boundary condition puts them — so the discrete
/// form is short of exactly one boundary strip and is itself **exact**:
///
/// ```text
/// fᵀ K g = 4 g_1 h² (Nx - 1) (Ny - 1)          (this function; machine precision)
///        -> 4 g_1 a b  as h -> 0               (the O(h) continuum statement)
/// ```
///
/// Returned is the **discrete** value, in units of the reference rigidity (multiply by
/// `rho_s kappa²` for Joules). Measured (on the Python plate — see the module header) to
/// 5.9e-16 … 1.9e-14 relative across grids.
///
/// This is the **only** one of the free plate's four probes that responds to the coupling
/// rigidity: the twist bound is blind to it by construction, and the free-beam reduction is exact
/// only *at* `g_1 = 0`. It is therefore load-bearing despite being the least glamorous of the
/// three, and is not to be dropped for looking like a unit test of an array sum.
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

/// Continuum acoustic-tube resonances (HANDOFF §12.A, the clarinet oracle).
///
/// A tube of length `L` in air of sound speed `c0` resonates at
///
/// - `"closed-open"` -> `f_n = (2n - 1) c0 / (4 L)` — the **odd** harmonics only (a quarter-wave
///   resonator: pressure antinode at the rigid wall, node at the open end). This is the clarinet
///   signature and the model-specific correctness test for the bore.
/// - `"open-open"` or `"closed-closed"` -> `f_n = n c0 / (2 L)` — the **full** harmonic series
///   (half-wave resonator; symmetric ends, matching boundary condition at both).
///
/// Returns the lowest `n_partials` resonances (Hz), ascending. The exact air constant `c0` is the
/// analogue of the string's wave speed here; unlike string tension it is a property of the
/// medium, not tunable.
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

/// Discrete temporal frequency (Hz) of a bore eigenmode from its angular-frequency-squared.
///
/// The staggered p/U leapfrog eliminates the velocity to `C δ_tt p = -k² L p` (`L = Gᵀ M⁻¹ G`,
/// `C` the compliance mass), so an eigenmode `L φ = ω² C φ` obeys the same simple-harmonic
/// leapfrog as every other second-order scheme here:
///
/// ```text
/// sin(Ω k / 2) = (k/2) ω = (k/2) sqrt(omega2),    f = arcsin((k/2) sqrt(omega2)) / (π k).
/// ```
///
/// `omega2` is the **generalized eigenvalue** of `(L, C)` restricted to the free (non-open)
/// pressure nodes — already `ω²` (no extra wave-speed factor: `c0` is baked into `L` and `C`). As
/// `k → 0` this tends to the continuum `f = sqrt(omega2)/(2π)`, whose values are the resonances
/// of [`bore_resonance_frequencies`]. Stable for all valid `λ` (the argument stays `<= 1` when
/// `λ <= 1`). Computed as `arcsin(clip(½k√(max(ω²,0)), -1, 1)) / (πk)`.
pub fn discrete_bore_eigenfrequency(omega2: f64, k: f64) -> f64 {
    let arg = 0.5 * k * omega2.max(0.0).sqrt();
    arg.clamp(-1.0, 1.0).asin() / (PI * k)
}

/// Discrete temporal frequency (Hz) of a beam eigenmode, implicit theta-scheme.
///
/// `mu` is the **4th-power spatial eigenvalue** `ω²/κ²` — the generalized eigenvalue of
/// `K φ = mu W φ` for the energy-first operator (`physsynth_core::ops::free_beam_stiffness`),
/// `mu → β⁴` in the continuum. The modal stiffness is `Q = kappa²·mu = ω²`. Inserting
/// `u^n = z^n φ` into `W δ_tt u = -kappa² K (θ u^{n+1} + (1-2θ) u^n + θ u^{n-1})` gives the same
/// relation as the plate (with `Q` from the beam operator rather than `kappa²Λ²`):
///
/// ```text
/// sin²(ω k / 2) = s = Q k² / (4 + 4 θ Q k²),    f = arcsin(sqrt(s)) / (π k).
/// ```
///
/// As `k → 0` this tends to the spatial `f = kappa·sqrt(mu)/(2π)`. Unconditionally stable for
/// `θ >= 1/4`; depends on `θ`, so callers pass the resonator's own `theta`.
pub fn discrete_beam_eigenfrequency(mu: f64, kappa: f64, k: f64, theta: f64) -> f64 {
    let q = kappa * kappa * mu;
    let s = q * k * k / (4.0 + 4.0 * theta * q * k * k);
    s.sqrt().asin() / (PI * k)
}
