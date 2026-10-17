//! The rotating-wave BVP — `physsynth/analysis/rotating_wave.py`, transcribed.
//!
//! The geometrically exact string's Tier B oracle: the *exact* relative equilibrium of the scheme,
//! a rigid helix `u = φ cos Ωt`, `w = φ sin Ωt`, `v = ψ`. Seed a `GeometricString` with it and the
//! string rotates to round-off; that is the sharpest statement this project makes about model #10,
//! and it is the reason a 602-line BVP solver lives in an analysis package.
//!
//! The Python module's docstrings were the physics, and this header documented only what the
//! translation had to decide. The Python bodies went at deletion 7 (`b1d37c8`) and the package at
//! retirement plan §48, so the derivations are carried here — the module docstring under "The
//! physics", the constants', fields' and functions' on the items of the same name — verbatim from
//! `b1d37c8^`, save for cross-references respelled natively and SciPy plumbing dated (retirement
//! plan §53). The Python's non-convergence `RuntimeWarning` has no native counterpart: natively a
//! failed continuation reports [`RotatingWave::converged`] `== false` and nothing is printed.
//!
//! # The physics
//!
//! Rotating-wave relative equilibrium of the geometrically-exact string (model #10, Tier B).
//!
//! **The one exact frequency oracle model #10 has**, and the only escape from its Tier C "measure
//! the residual, do not promise cents" taxonomy. Where model #9's tension is a spatial *scalar* —
//! which is what buys it an exact elliptic Duffing oracle — model #10 promotes the tension to a
//! **field**, and every closed form degrades to a limit. Every one but this one.
//!
//! **The ansatz.** A helix spinning rigidly about the string axis:
//!
//! ```text
//! u(x,t) = phi(x) cos(Omega t),    w(x,t) = phi(x) sin(Omega t),    v(x,t) = psi(x)
//! ```
//!
//! solves the **full** nonlinear PDE exactly — no truncation, no small-slope assumption — because
//! the transverse amplitude `r^2 = u_x^2 + w_x^2 = phi'(x)^2 (cos^2 + sin^2) = phi'(x)^2` is
//! **time-independent**. The potential sees the two polarizations *only* through `r^2`, so the
//! whole nonlinearity goes static: the local stretch `Lambda`, the tension field, and the
//! longitudinal forcing are all frozen. The string is bent into a fixed helical shape and spun.
//!
//! That is also **exactly why** the core's `GeometricString::longitudinal_energy` has nothing to
//! do: a circular polarization does not pump the longitudinal field at all, where a planar one
//! pumps it at `2 Omega` (phantom partials). This module is what sharpens that claim — Tier A/3's
//! discriminator — from *orders of magnitude apart* to **bit-zero longitudinal kinetic energy**.
//!
//! **What it is an oracle FOR, stated honestly.** Two different claims, and blurring them is the
//! trap:
//!
//! 1. **Scheme-consistency (exact, ~1e-15).** Seeded from a converged BVP the *discrete* scheme
//!    rotates rigidly forever. This certifies that the discrete-gradient scheme admits an exact
//!    discrete relative equilibrium and that this solver finds it. It is **not** a claim about
//!    physics — the scheme is being checked against its own equations.
//! 2. **Physics: the mechanism behind Tier C/8 (a limit).** The Kirchhoff-Carrier prediction
//!    `Omega = sqrt(omega0^2 + eps R^2)` ([`crate::duffing::kc_mode_coefficients`], with
//!    `EA_#9 <-> a = EA - T0`, and `eps R^2` unhalved because a circular mode's stretch is **2x**
//!    a planar mode's time-average) assumes `phi` is a *sine*. It is not. A rigid helix is
//!    stretched **non-uniformly** — most near the nodes, where `phi'` is largest — so `T(x)` is
//!    non-uniform and the true `phi` is a deformed sine. [`RotatingWave::shape_residual`] measures
//!    that deformation, and it *is* the reason the KC frequency is wrong. The residual gets a
//!    **mechanism**, not just an amplitude scaling.
//!
//! **The equations.** With `q = (u_x, w_x, v_x)`, `a = EA - T0`, and the nonlinear excess
//! `V_nl = a[r^2/2 + 1 + v_x - Lambda]`, substituting the ansatz into the continuum PDE gives:
//!
//! ```text
//! -rho Omega^2 phi = ( T(x) phi' )' ,     T(x) = T0 + a chi(x),   chi = 1 - 1/Lambda
//! d/dx [ dV/dv_x ] = 0    <=>    EA psi' + a (1 - (1 + psi')/Lambda) = const
//! ```
//!
//! the second being the plan's `dV/dv_x = const`: the longitudinal field is in **static
//! equilibrium** against a frozen transverse load. On the grid the discrete divergence `Gm` has
//! exactly the constants in its kernel, so the discrete second equation reproduces the continuum
//! statement *as a statement*, not merely to O(h^2).
//!
//! **The two time-discrete factors — the crux of this module.** The oracle would be worthless at
//! 1e-15 if it solved the continuum-time BVP, because the scheme does not. Substituting the
//! *discrete* rotating wave `u_j^n = phi_j cos(Omega n k)` into the theta-scheme gives two
//! factors, and a naive port drops the second:
//!
//! - `delta_tt -> -s` with `s = (4/k^2) sin^2(Omega k / 2)` — and the theta-average
//!   `theta u+ + (1-2theta) u^n + theta u-` contributes `(1 - theta k^2 s)`. Familiar from
//!   [`crate::modal::discrete_stiff_mode_frequency`].
//! - The discrete gradient pairs `q^{n+1}` against `q^{n-1}` — it spans **2k**, not `k` — so its
//!   midpoint `qbar = (q+ + q-)/2` picks up `cos(Omega k)` on the *transverse* strains:
//!   `ubar_x = phi' cos(Omega n k) cos(Omega k)`. Hence the DG force carries a factor
//!   `cos(Omega k) = 1 - k^2 s / 2`.
//!
//! Both factors are **linear in `s`**, which is why `s` (not `Omega`) is the natural unknown. The
//! longitudinal row carries **neither**: `psi` is static, so its theta-average is `psi` for any
//! `theta`, and its DG component is time-independent. Set `time_discrete: false` to drop both and
//! get the semi-discrete (continuous-time) BVP — the difference between the two is the scheme's
//! temporal dispersion, which is what must be kept *out* of the Tier C/8 comparison.
//!
//! **The Jacobian is not symmetric, and that is not a bug.** The reduced (phi, psi) system looks
//! variational — its cell blocks are the Hessian of `V_nl` on the **planar** strain slice
//! `(p, 0, z)` (see [`planar_hessian_cells`]), which is symmetric. But `cos(Omega k)` multiplies
//! the transverse row *only*, so `dF_phi/dpsi = cos(Omega k) dF_psi/dphi`: symmetric only as
//! `k -> 0`. Solved with a general sparse LU ([`crate::sparse_lu`]; `splu` in the Python),
//! exactly as the core solves its own non-symmetric DG Jacobian.
//!
//! **Degeneracy required.** The rotating wave needs `kappa_u == kappa_w`: the `u` and `w` rows
//! must be the *same* equation for one `phi` to serve both. On a non-degenerate string
//! (`GeometricString::is_degenerate` false — the configuration that *whirls*) no rigid rotation
//! exists. This module takes a single `kappa` for that reason.
//!
//! Independent of the core (an oracle must not depend on the code it validates), so the operators
//! are rebuilt here rather than imported. `theta` is required rather than defaulted — the
//! frequency depends on it, so the caller must pass the resonator's own, as
//! [`crate::modal::discrete_stiff_mode_frequency`] also demands.
//!
//! See `docs/dev/geometrically-exact-string-plan.md` (Tier B, decision #5).
//!
//! # The port
//!
//! It is the package's only member that solves a nonlinear system — Newton on a sparse Jacobian,
//! wrapped in an eight-step amplitude continuation — and therefore its only member of the
//! migration plan's **Group D**, the sparse-LU models. Two measurements were taken before a line
//! of this file was written, following §36.2's rule that a margin is measured before it is claimed.
//! Both changed the shape of the port, so both are recorded here rather than in a commit message.
//!
//! # 1. The margin on the converged root is enormous; the iteration count is the one exception
//!
//! §24 put Group D on measured tolerance rather than bit-identity, because SuperLU is supernodal
//! and its blocking is a property of how SciPy was *built*. The question that leaves open is how
//! much a different elimination actually costs *here*, and the answer is: nothing worth naming.
//!
//! Measured on the test file's own fixture (`N = 32`, amplitude 5e-3, mode 1) by perturbing the
//! Python solve's Newton step by a relative amount and re-reading the root:
//!
//! | perturbation of the Newton step | `ΔΩ/Ω` | `Δψ` (rel) | iterations |
//! |---|---|---|---|
//! | `permc_spec="NATURAL"` (this module's ordering) | 1.8e-16 | 1.0e-15 | 24 |
//! | relative 1e-16 noise | 3.6e-16 | 4.5e-15 | 24 |
//! | relative 1e-14 noise | 1.8e-16 | 5.0e-15 | 24 |
//! | relative 1e-12 noise | 1.8e-16 | 1.7e-15 | 24 |
//! | relative 1e-10 noise | 1.8e-16 | 2.4e-15 | 24 |
//!
//! A perturbation **six orders of magnitude larger** than any difference between two LU
//! implementations moves `Ω` by one ulp. The reason is structural rather than lucky: Newton's step
//! is a *means*, the residual defines the root, and the convergence bar is a relative bar on the
//! step, which a relative perturbation of an already tiny step barely changes.
//!
//! **The iteration count is the exception, and the perturbation probe was too narrow to see it.**
//! That probe held the fixture at `N = 32`, mode 1, `κ = 0` and reported the count unmoved at 24,
//! which is true — and is a claim about *that fixture*. Run the finished port against the Python
//! over 126 fixtures (`N` 16…128, modes 1–3, `κ ∈ {0, 2}`, three amplitudes) and the count differs
//! in **17 of the 108 that converge**, by as much as 13. So the branch hazard §19.2 and §20.3
//! record for the core's timestepping solve *does* exist here; what is different is that it is
//! harmless, because the two paths land on the same root — the worst `ΔΩ/Ω` across those 108 is
//! **8.4e-15**, and the fixture with the widest iteration gap (28 against 41) agrees to 9.0e-16.
//! The lesson is the general one, and it cost a wrong sentence in this header to learn:
//! **a margin measured at one fixture is a claim about one fixture.**
//!
//! Two things *were* claimed across every fixture tried, and the parity file asserted both until
//! it was deleted with the Python bodies (plan §44):
//!
//! * **`converged` agrees in 126 of 126.** Eighteen fixtures fail to converge, and they are the
//!   same eighteen on both sides. That is the one flag a caller is told to branch on, so it is the
//!   one that has to agree.
//! * **Among the converged, `Ω`, `s`, `φ`, `ψ` and the tension field agree to ~1e-14 or better.**
//!
//! Where the BVP does *not* converge the two implementations produced different answers and the
//! parity file asserted nothing about them beyond the flag. That is not a dodge: `converged ==
//! false` means "the helix is not a relative equilibrium; do not treat this as physics". It is also
//! the same lesson `docs/dev/scientific-hurdles.md` §6 landed on the same day — **a margin measured
//! in the converged regime says nothing about the failing one.** There the iteration path *is* the
//! answer rather than a means to it, so the two paths are free to diverge, and they do.
//!
//! **What this licenses:** a tolerance port with a wide budget, and no need to chase SuperLU's
//! digits or reproduce COLAMD. **What it does not licence:** an exact assertion on `Ω`, on `ψ`, or
//! on `iterations`. The one exact thing on the path — `arcsin`, which turns `s` back into `Ω` — is
//! a `np.<transcendental>` and therefore forbidden from an exact cross-language claim by
//! `docs/dev/scientific-hurdles.md` §3 in any case.
//!
//! # 2. The natural ordering fills this matrix completely, and the fix is §29.2's, again
//!
//! The Jacobian is `[φ; ψ; s]`: two `(N-1)`-blocks stacked **by field**, plus one bordering row
//! (the amplitude normalisation) and one bordering column (`∂F/∂s`). §24 wrote that this module's
//! LU orders columns naturally because "every Group D matrix in this project is a banded FDTD
//! operator whose natural order already has none [no fill] to speak of", and added that if a later
//! model made fill the constraint, an ordering would go in front of the solver. `string_geometric`
//! was the first such model (§29.2); this is the second, and it is the same mechanism one level
//! up — the `φ`↔`ψ` coupling is a *cell-local* term, so stacking by field puts every coupling
//! `N-1` columns off the diagonal and the elimination fills the whole envelope between.
//!
//! Measured on the real Jacobian, nonzeros stored in `L + U`:
//!
//! | `N` | dim | SuperLU COLAMD | natural | interleaved by node |
//! |---|---|---|---|---|
//! | 32 | 63 | 1,128 | 2,278 | **593** |
//! | 128 | 255 | 10,956 | 33,722 | **2,460** |
//! | 256 | 511 | 4,959 | 132,992 | **4,954** |
//! | 512 | 1,023 | 9,949 | 528,139 | **9,944** |
//!
//! The natural column is `≈ dim²/2` — dense triangular factors, the whole matrix. Interleaving
//! `(φ_i, ψ_i)` and leaving `s` last is linear in `dim`, beats COLAMD outright at small `N` and
//! matches it at large `N`, and is a **closed form in `N`** rather than an ordering heuristic — so
//! [`crate::sparse_lu::SparseLu::factor_permuted`] takes it and no COLAMD needs porting. §29.2's
//! numbers for the core's own Jacobian (2,645 interleaved against COLAMD's 2,788) are the same
//! result on the same shape, which is why this needed measuring but not deciding.
//!
//! Nothing in this project could have caught the natural-order version by a physics bar: it
//! computes the identical root, just through a dense factorization. It is a cost cliff, not a
//! correctness one, and it is written down because that is exactly the class §29 said no bar here
//! can see.
//!
//! # What is shared with the core, and why it is not a dependency
//!
//! [`crate::sparse`] and [`crate::sparse_lu`] are `#[path]` includes of `physsynth-core`'s files,
//! the manoeuvre `crate::root` already uses and whose argument is in this crate's `lib.rs`: one
//! copy of the text, compiled into both crates, and no edge for `tests/deps.rs` to see. The
//! *operators* are the opposite case and are rebuilt locally below, because an oracle must not be
//! assembled out of the discretisation it checks. A sparse LU is a numerical method with no physics
//! in it; a second difference is the thing under test.

use crate::sparse::Csr;
use crate::sparse_lu::{SparseLu, DIAG_PIVOT_THRESH};

/// Amplitude-continuation steps from the linear mode up to the requested amplitude.
///
/// The `R -> 0` seed (a sine at the theta-scheme's linear frequency) is exact only in the limit,
/// and the mode shape deforms as `R` grows. Stepping the amplitude and reusing each solution as the
/// next seed is what keeps Newton in its basin at amplitudes where the deformation is large.
pub const CONTINUATION_STEPS_DEFAULT: usize = 8;

/// **Relative** convergence bar on the Newton *step*, not on the residual.
///
/// The residual's three blocks carry different units — a transverse force density, a longitudinal
/// force density and a displacement — so a single max-norm bar on it would be a unit-mixing
/// accident. The step is compared against the solution's own scale, which is unit-free per block
/// and is what Newton's quadratic convergence makes a proxy for the error.
pub const NEWTON_TOL_DEFAULT: f64 = 1e-14;

/// Cap on Newton iterations per continuation step. Reaching it reports `converged == false`; it
/// never silently renders. (The Python also raised a `RuntimeWarning`; natively there is none.)
pub const NEWTON_MAXITER_DEFAULT: usize = 50;

/// A converged rotating-wave relative equilibrium: the helix, its stretch, its frequency.
///
/// Field for field the Python `RotatingWave` NamedTuple, plus [`Self::failed_step`] and
/// [`Self::failed_amplitude`], which carried what the Python side needed to raise its
/// non-convergence warning with the same text, and now say where a starved continuation stopped.
#[derive(Clone, Debug)]
pub struct RotatingWave {
    /// Transverse profile on the **full grid** `(N+1)` (m), zero at both clamped ends.
    ///
    /// The helix is `u = phi cos(Omega t)`, `w = phi sin(Omega t)`. A *deformed* sine — see
    /// [`Self::shape_residual`].
    pub phi: Vec<f64>,
    /// Longitudinal profile on the **full grid** `(N+1)` (m), zero at both clamped ends.
    ///
    /// **Static, and nonzero.** The helix holds a fixed longitudinal stretch against the frozen
    /// transverse load. Do not test the rotating wave by asserting `v == 0`; assert that `v` does
    /// not *move* (zero longitudinal kinetic energy).
    pub psi: Vec<f64>,
    /// Angular frequency (rad/s) of the rotation.
    pub omega: f64,
    /// `Omega / (2 pi)` (Hz).
    pub frequency: f64,
    /// The scheme's `s = (4/k^2) sin^2(Omega k / 2)` (rad^2/s^2) — the unknown solved for.
    ///
    /// Equals `Omega^2` exactly when `time_discrete` is false. Both time-discrete factors are
    /// linear in this variable, which is why it, and not `Omega`, is the natural unknown.
    pub s: f64,
    /// The imposed modal amplitude `R` (m) — `phi`'s projection onto `sin(m pi x / L)`.
    pub amplitude: f64,
    /// The mode number `m` continued up from.
    pub mode: usize,
    /// Per-cell `Lambda` `(N)` — the frozen stretch field. `1` at rest.
    pub stretch_ratio: Vec<f64>,
    /// Per-cell axial tension `EA Lambda - a` (N) `(N)` — **non-uniform**, which is the point.
    ///
    /// A rigid helix is stretched most near the nodes, where `phi'` is largest. That
    /// non-uniformity is exactly what a Kirchhoff-Carrier tension (a single scalar) cannot
    /// represent, and it is the mechanism behind [`Self::shape_residual`] and Tier C/8's breakage.
    pub tension: Vec<f64>,
    /// `max|phi - R sin(m pi x / L)| / max|phi|` — the deviation from a sine.
    ///
    /// **The physics payoff of this module.** `-> 0` with amplitude; grows as the helix stretches
    /// non-uniformly. This is *why* [`kc_circular_frequency`] is wrong, and reporting it turns
    /// Tier C/8's frequency residual from a bare number into a mechanism.
    pub shape_residual: f64,
    /// Total Newton iterations summed over the continuation.
    pub iterations: usize,
    /// Whether every continuation step converged. Never render a false one as physics.
    pub converged: bool,
    /// Whether the scheme's `(1 - theta k² s)` and `cos(Omega k)` factors were included.
    pub time_discrete: bool,
    /// The 1-based continuation step that failed, or 0 when everything converged.
    pub failed_step: usize,
    /// The target amplitude of the failed step, or 0.0 when everything converged.
    pub failed_amplitude: f64,
}

/// `(D2, Gp, Gm)`: the Dirichlet second difference and the SBP node↔cell adjoint pair.
///
/// Rebuilt here rather than taken from the core — an oracle must not depend on the code it
/// validates, and a silently divergent discretisation is exactly what the seeded-rotation test
/// exists to catch. `Gm @ Gp == D2` exactly, and `Gm`'s kernel is the constants, which is what
/// makes the discrete `dV/dv_x = const` statement exact.
fn operators(n_cells: usize, h: f64) -> (Csr, Csr, Csr) {
    let n_int = n_cells - 1;
    let inv_hh = 1.0 / (h * h);
    let d2 = Csr::from_rows(
        n_int,
        n_int,
        (0..n_int)
            .map(|i| {
                let mut row = Vec::with_capacity(3);
                if i > 0 {
                    row.push((i - 1, inv_hh));
                }
                row.push((i, -2.0 * inv_hh));
                if i + 1 < n_int {
                    row.push((i + 1, inv_hh));
                }
                row
            })
            .collect(),
    );
    let inv_h = 1.0 / h;
    // `sparse.diags([+1/h at offset 0, -1/h at offset -1], shape=(N, N-1))`: row `i` of the
    // node->cell gradient differences node `i+1` from node `i`.
    let gp = Csr::from_rows(
        n_cells,
        n_int,
        (0..n_cells)
            .map(|i| {
                let mut row = Vec::with_capacity(2);
                if i > 0 {
                    row.push((i - 1, -inv_h));
                }
                if i < n_int {
                    row.push((i, inv_h));
                }
                row
            })
            .collect(),
    );
    let gm = gp.transpose().scaled(-1.0);
    (d2, gp, gm)
}

/// `(Lambda, chi, d)` per cell from planar strains `p = phi'`, `z = psi'`.
///
/// `chi = 1 - 1/Lambda` and `d = Lambda - 1 - z`, both through the **cancellation-free**
/// rearrangements the core uses (`physsynth_core::string_geometric`'s stretch terms):
///
/// ```text
/// Lambda - 1     = (z (2 + z) + p^2) / (Lambda + 1)
/// Lambda - 1 - z = p^2 / (Lambda + 1 + z)
/// ```
///
/// This is not tidiness and not duplicated code for its own sake: both are `O(strain)` quantities
/// assembled from `O(1)` ones, and musical strings run at strain 1e-4 .. 1e-3. Evaluating them
/// literally would converge this BVP to a root of a *slightly different* equation than the one the
/// core steps, and the seeded helix would then shed that difference into the longitudinal field —
/// the 1e-15 gate would fail for a reason unrelated to the physics.
fn stretch(p: &[f64], z: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = p.len();
    let mut lam = Vec::with_capacity(n);
    let mut chi = Vec::with_capacity(n);
    let mut d = Vec::with_capacity(n);
    for i in 0..n {
        let (pi, zi) = (p[i], z[i]);
        let one_z = 1.0 + zi;
        let l = (one_z * one_z + pi * pi).sqrt();
        let lam_m1 = (zi * (2.0 + zi) + pi * pi) / (l + 1.0);
        let denom = l + 1.0 + zi;
        // Physical cells sit at denom ~ 2; only an inverted element nears 0.
        let di = if denom > 1.0 {
            pi * pi / denom
        } else {
            l - 1.0 - zi
        };
        lam.push(l);
        chi.push(lam_m1 / l);
        d.push(di);
    }
    (lam, chi, d)
}

/// `(H_pp, H_pz, H_zz)`: the Hessian of `V_nl` on the **planar** strain slice, per cell.
///
/// The rotating wave's linearisation. Although the helix is fully three-dimensional, its reduced
/// `(phi, psi)` system linearises on the *planar* slice `q = (p, 0, z)`: `V_nl` sees the
/// polarizations only through `r^2 = p^2 (cos^2 + sin^2) = p^2`, which is the same value the
/// planar strain `(p, 0, z)` produces. So the 2x2 reduced block is the full 3x3 Hessian with the
/// `w` row and column struck out:
///
/// ```text
/// H = a [ chi I3 + m m^T / Lambda^3 - e3 e3^T ] ,   m = (u_x, w_x, 1 + v_x)
/// ```
///
/// Three entries survive, and the third simplifies exactly — `(1+z)^2 - Lambda^2 = -p^2`:
///
/// ```text
/// H_pp = a (chi + p^2 / Lambda^3)
/// H_pz = a p (1 + z) / Lambda^3
/// H_zz = a ((1+z)^2 / Lambda^3 - 1/Lambda) = -a p^2 / Lambda^3
/// ```
///
/// That last identity written literally is a difference of `O(1)` terms and is catastrophic at
/// musical strain; written this way it has no cancellation at all. `H_zz` is *negative* for a
/// hardening string — the nonlinear excess softens the longitudinal direction — but the full
/// longitudinal diagonal adds `EA` and stays positive.
///
/// Public because it cross-checks against the core: `H/2` is exactly
/// `string_geometric::dg_jacobian(q, q)`, the `1/2` being `d(qbar)/d(q+)`. That check cannot live
/// in this crate without the dependency edge `tests/deps.rs` refuses, so it is a core test reaching
/// this crate through its test-only dev-dependency (retirement plan §24.2):
/// `crates/physsynth-core/tests/string_geometric_helix.rs`, at 1e-12 since the core's `(v,v)`
/// block was made cancellation-free. It was `test_geometric_rotating_wave.py` until retirement
/// plan §36.
pub fn planar_hessian_cells(p: &[f64], z: &[f64], a: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (lam, chi, _) = stretch(p, z);
    let n = p.len();
    let mut h_pp = Vec::with_capacity(n);
    let mut h_pz = Vec::with_capacity(n);
    let mut h_zz = Vec::with_capacity(n);
    for i in 0..n {
        let l = lam[i];
        let inv_lam3 = 1.0 / (l * l * l);
        h_pp.push(a * (chi[i] + p[i] * p[i] * inv_lam3));
        h_pz.push(a * p[i] * (1.0 + z[i]) * inv_lam3);
        h_zz.push(-a * p[i] * p[i] * inv_lam3);
    }
    (h_pp, h_pz, h_zz)
}

/// Kirchhoff–Carrier's `Omega = sqrt(omega0² + eps R²)` for a **circular** mode (rad/s).
///
/// A **limit** oracle, not an exact one — it is here to be measured *against*
/// [`solve_rotating_wave`], and it breaks by mode-shape deformation
/// ([`RotatingWave::shape_residual`]).
///
/// Take `(omega0_sq, eps)` from [`crate::duffing::kc_mode_coefficients`] with
/// `EA = a = EA_#10 - T0` (mind the identification) and the **discrete**
/// `p2 = spatial_eigenvalue_p2(...)` ([`crate::damping::spatial_eigenvalue_p2`]), so the only
/// thing left to explain is the shape.
///
/// **Note the missing half.** For a *planar* mode the stretch `I ~ cos^2(Omega t)` averages to
/// `R^2 p^2 L / 4`; a circular mode's is `R^2 p^2 L / 2` — **exactly twice**, and constant rather
/// than pulsing, because `cos^2 + sin^2 = 1`. So the tension shift is the full `eps R^2` with no
/// factor of `1/2`, and this is *not* the same expression as the planar Duffing frequency
/// ([`crate::duffing::duffing_frequency`]) evaluated at the same `R`.
pub fn kc_circular_frequency(omega0_sq: f64, eps: f64, amplitude: f64) -> Result<f64, String> {
    let val = omega0_sq + eps * amplitude * amplitude;
    if val <= 0.0 {
        return Err(format!("omega0^2 + eps R^2 must be positive, got {val}."));
    }
    Ok(val.sqrt())
}

/// The **exact** two-level history `(u0, w0, v0, u_prev, w_prev, v_prev)` seeding the helix.
///
/// Assign these straight onto the resonator's fields — do **not** route them through `set_state`,
/// whose `y^{-1}` is a second-order Taylor start: consistent but not exact, so it would seed an
/// `O(k³)` history error that the helix sheds into the longitudinal field. The whole 1e-15 claim
/// lives in this function.
///
/// The exact history is just the ansatz evaluated at `n = 0` and `n = -1`:
///
/// ```text
/// u^0 = phi,              w^0 = 0,                v^0 = psi
/// u^-1 = phi cos(Omega k), w^-1 = -phi sin(Omega k), v^-1 = psi
/// ```
///
/// `fs` must be the resonator's own (`Omega` was solved for at that `k`, if `time_discrete`).
#[allow(clippy::type_complexity)]
pub fn rotating_wave_history(
    wave: &RotatingWave,
    fs: f64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>), String> {
    if fs <= 0.0 {
        return Err("fs must be positive.".to_string());
    }
    let k = 1.0 / fs;
    let cos_k = (wave.omega * k).cos();
    let sin_k = (wave.omega * k).sin();
    let zero = vec![0.0; wave.phi.len()];
    Ok((
        wave.phi.clone(),
        zero.clone(),
        wave.psi.clone(),
        wave.phi.iter().map(|v| v * cos_k).collect(),
        wave.phi.iter().map(|v| -v * sin_k).collect(),
        wave.psi.clone(),
    ))
}

/// Everything [`solve_rotating_wave`] needs, mirroring `GeometricString`'s own parameters.
///
/// They must match the resonator the result will seed — `theta`, `fs` and `n_cells` included,
/// since the frequency depends on all three.
#[derive(Clone, Copy, Debug)]
pub struct BvpParams {
    /// The string, exactly as the resonator was built: length (m).
    pub l: f64,
    /// Tension `T0` (N).
    pub t: f64,
    /// Linear density (kg/m).
    pub rho: f64,
    /// Axial stiffness (N). `EA < T` is refused (see [`solve_rotating_wave`]).
    pub ea: f64,
    /// Sample rate (Hz).
    pub fs: f64,
    /// Cells `N`.
    pub n_cells: usize,
    /// The resonator's own theta.
    pub theta: f64,
    /// The modal amplitude `R` (m): `phi`'s projection onto `sin(m pi x / L)`, normalised so that
    /// `R` means what it means in [`crate::duffing::kc_mode_coefficients`]. Chosen over
    /// `max|phi|` precisely so the Tier C/8 comparison against [`kc_circular_frequency`] is
    /// amplitude-consistent.
    pub amplitude: f64,
    /// Mode number `m` to continue up from (the linear sine the helix deforms away from). The
    /// Python defaulted it to 1.
    pub mode: usize,
    /// Bending stiffness, applied to **both** polarizations: a rotating wave exists only on a
    /// degenerate (`kappa_u == kappa_w`) string. The Python defaulted it to 0.
    pub kappa: f64,
    /// Include the scheme's `(1 - theta k^2 s)` and `cos(Omega k)` factors (the Python's
    /// default). Required for the seeded helix to rotate rigidly to machine precision. Pass
    /// `false` for the semi-discrete BVP, whose `Omega` carries no temporal dispersion — the
    /// honest comparison for Tier C/8.
    pub time_discrete: bool,
    /// Continuation steps ([`CONTINUATION_STEPS_DEFAULT`]).
    pub continuation_steps: usize,
    /// A **relative** bar on the Newton step ([`NEWTON_TOL_DEFAULT`]).
    pub tol: f64,
    /// Newton iterations per continuation step ([`NEWTON_MAXITER_DEFAULT`]).
    pub maxiter: usize,
}

/// The fill-reducing reordering: `(phi_i, psi_i)` interleaved by node, `s` left last.
///
/// `q[factored_index] = caller_index`, which is [`SparseLu::factor_permuted`]'s convention. See
/// this module's header for what it buys — the natural order fills the factors densely.
fn interleave_permutation(n_int: usize) -> Vec<usize> {
    let mut q = Vec::with_capacity(2 * n_int + 1);
    for i in 0..n_int {
        q.push(i);
        q.push(n_int + i);
    }
    q.push(2 * n_int);
    q
}

/// Solve the rotating-wave BVP for `(phi, psi, Omega)` by amplitude continuation.
///
/// Returns the converged helix; check [`RotatingWave::converged`].
///
/// Refuses (`Err`) on non-physical parameters, `EA < T` (softening: outside this oracle's scope),
/// a `mode` outside `1 .. N-1`, or a rotation past Nyquist (`Omega k > pi`, where the discrete
/// `s` no longer maps back to a frequency).
pub fn solve_rotating_wave(p: &BvpParams) -> Result<RotatingWave, String> {
    if p.l.min(p.t).min(p.rho).min(p.fs).min(p.ea) <= 0.0 {
        return Err("L, T, rho, fs, EA must all be positive.".to_string());
    }
    if p.n_cells < 2 {
        return Err("N must be >= 2.".to_string());
    }
    if p.mode < 1 || p.mode > p.n_cells - 1 {
        return Err(format!(
            "mode must be in 1 .. {}, got {}.",
            p.n_cells - 1,
            p.mode
        ));
    }
    if p.kappa < 0.0 {
        return Err("kappa must be >= 0.".to_string());
    }
    if !(p.theta > 0.0 && p.theta <= 1.0) {
        return Err(format!("theta must be in (0, 1], got {}.", p.theta));
    }
    if p.ea < p.t {
        return Err(format!(
            "EA ({}) < T ({}) is a softening string, outside this oracle's scope: the continuation \
             seed is the hardening branch. The resonator itself permits it via allow_softening.",
            p.ea, p.t
        ));
    }
    if p.continuation_steps < 1 {
        return Err("continuation_steps must be >= 1.".to_string());
    }
    if p.maxiter < 1 {
        return Err("maxiter must be >= 1.".to_string());
    }
    if p.tol <= 0.0 {
        return Err("tol must be > 0.".to_string());
    }

    let n_cells = p.n_cells;
    let n_int = n_cells - 1;
    let h = p.l / n_cells as f64;
    let k = 1.0 / p.fs;
    let c_sq = p.t / p.rho;
    let a = p.ea - p.t;
    let (d2, gp, gm) = operators(n_cells, h);
    let op_u = if p.kappa != 0.0 {
        d2.scaled(c_sq)
            .sub(&d2.matmul(&d2).scaled(p.kappa * p.kappa))
    } else {
        d2.scaled(c_sq)
    };
    let op_v = d2.scaled(p.ea / p.rho);
    let ident = Csr::identity(n_int);

    // The interior grid, and the mode the continuation starts from.
    let shape: Vec<f64> = (1..n_cells)
        .map(|i| {
            let x = p.l * i as f64 / n_cells as f64;
            (p.mode as f64 * std::f64::consts::PI * x / p.l).sin()
        })
        .collect();
    // `h * sum(sin^2) == L/2` exactly on the interior grid, which is what makes this projection the
    // same amplitude convention as `duffing::kc_mode_stretch`.
    let proj: Vec<f64> = shape.iter().map(|s| (2.0 / p.l) * h * s).collect();

    // Linear seed: the theta-scheme's own discrete dispersion, algebraically identical to
    // `modal::discrete_stiff_mode_frequency` — the `R -> 0` gate on this whole module.
    let p2 = (4.0 / (h * h))
        * (p.mode as f64 * std::f64::consts::PI / (2 * n_cells) as f64)
            .sin()
            .powi(2);
    let q_lin = c_sq * p2 + p.kappa * p.kappa * p2 * p2;
    let mut s = if p.time_discrete {
        q_lin / (1.0 + p.theta * k * k * q_lin)
    } else {
        q_lin
    };

    let perm = interleave_permutation(n_int);
    let mut phi = vec![0.0f64; n_int];
    let mut psi = vec![0.0f64; n_int];
    let mut total_iters = 0usize;
    let mut converged = true;
    let mut failed_step = 0usize;
    let mut failed_amplitude = 0.0f64;

    for step in 1..=p.continuation_steps {
        let target = p.amplitude * step as f64 / p.continuation_steps as f64;
        if step == 1 {
            // The linear mode: exact as R -> 0.
            phi = shape.iter().map(|v| target * v).collect();
        } else {
            // `phi` scales linearly in R, `psi` quadratically (it is driven by the r² v_x / 2
            // term). Cheap, and it keeps Newton's first step small where deformation is strongest.
            let ratio = step as f64 / (step - 1) as f64;
            for v in phi.iter_mut() {
                *v *= ratio;
            }
            for v in psi.iter_mut() {
                *v *= ratio * ratio;
            }
        }
        let ctx = NewtonCtx {
            op_u: &op_u,
            op_v: &op_v,
            gp: &gp,
            gm: &gm,
            ident: &ident,
            proj: &proj,
            perm: &perm,
            rho: p.rho,
            a,
            k,
            theta: p.theta,
            time_discrete: p.time_discrete,
        };
        let (iters, ok) = newton(&mut phi, &mut psi, &mut s, target, &ctx, p.tol, p.maxiter)?;
        total_iters += iters;
        if !ok {
            converged = false;
            failed_step = step;
            failed_amplitude = target;
            break;
        }
    }

    let omega = if p.time_discrete {
        let arg = 0.5 * k * s.max(0.0).sqrt();
        if arg >= 1.0 {
            return Err(format!(
                "The converged rotation is at or past Nyquist (s = {s:.4} gives \
                 sin(Omega k / 2) = {arg:.4} >= 1): the discrete frequency does not map back to an \
                 Omega. Raise fs."
            ));
        }
        2.0 * arg.asin() / k
    } else {
        s.max(0.0).sqrt()
    };

    let (lam, _, _) = stretch(&gp.matvec(&phi), &gp.matvec(&psi));
    let mut phi_full = vec![0.0f64; n_cells + 1];
    let mut psi_full = vec![0.0f64; n_cells + 1];
    phi_full[1..n_cells].copy_from_slice(&phi);
    psi_full[1..n_cells].copy_from_slice(&psi);
    let peak = phi.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let shape_residual = if peak > 0.0 {
        phi.iter()
            .zip(&shape)
            .fold(0.0f64, |m, (v, sh)| m.max((v - p.amplitude * sh).abs()))
            / peak
    } else {
        0.0
    };

    Ok(RotatingWave {
        phi: phi_full,
        psi: psi_full,
        omega,
        frequency: omega / (2.0 * std::f64::consts::PI),
        s,
        amplitude: p.amplitude,
        mode: p.mode,
        tension: lam.iter().map(|l| p.ea * l - a).collect(),
        stretch_ratio: lam,
        shape_residual,
        iterations: total_iters,
        converged,
        time_discrete: p.time_discrete,
        failed_step,
        failed_amplitude,
    })
}

/// Everything the residual, the Jacobian and the Newton loop share, gathered so the three
/// signatures stay readable.
struct NewtonCtx<'a> {
    op_u: &'a Csr,
    op_v: &'a Csr,
    gp: &'a Csr,
    gm: &'a Csr,
    ident: &'a Csr,
    proj: &'a [f64],
    perm: &'a [usize],
    rho: f64,
    a: f64,
    k: f64,
    theta: f64,
    time_discrete: bool,
}

impl NewtonCtx<'_> {
    /// `(1 - theta k² s)` on the linear row and `cos(Omega k) = 1 - k² s / 2` on the DG row.
    ///
    /// The second factor is the crux of the whole BVP and a naive derivation drops it: the discrete
    /// gradient pairs `q^{n+1}` against `q^{n-1}`, spanning `2k` and not `k`, so the midpoint
    /// strain picks up `cos(Omega k)` on the *transverse* terms only. Both factors are linear in
    /// `s`, which is why `s` and not `Omega` is the natural unknown.
    fn factors(&self, s: f64) -> (f64, f64) {
        if self.time_discrete {
            (
                1.0 - self.theta * self.k * self.k * s,
                1.0 - 0.5 * self.k * self.k * s,
            )
        } else {
            (1.0, 1.0)
        }
    }
}

/// The BVP residual `[F_phi; F_psi; F_norm]`.
///
/// ```text
/// F_phi  = rho (1 - theta k² s) L_u phi + cos(Omega k) Gm[a chi phi'] + rho s phi
/// F_psi  = rho L_v psi + Gm[a (Lambda - 1 - psi')/Lambda]      (no time factors: psi is static)
/// F_norm = <phi, sin_m> (2/L) h - R
/// ```
fn residual(phi: &[f64], psi: &[f64], s: f64, target: f64, c: &NewtonCtx) -> Vec<f64> {
    let n_int = phi.len();
    let p = c.gp.matvec(phi);
    let z = c.gp.matvec(psi);
    let (lam, chi, d) = stretch(&p, &z);
    let (fac_lin, fac_dg) = c.factors(s);

    let u_term = c.op_u.matvec(phi);
    let dg_u: Vec<f64> = (0..p.len()).map(|i| c.a * chi[i] * p[i]).collect();
    let dg_u = c.gm.matvec(&dg_u);
    let v_term = c.op_v.matvec(psi);
    let dg_v: Vec<f64> = (0..p.len()).map(|i| c.a * d[i] / lam[i]).collect();
    let dg_v = c.gm.matvec(&dg_v);

    let mut out = Vec::with_capacity(2 * n_int + 1);
    for i in 0..n_int {
        out.push(c.rho * fac_lin * u_term[i] + fac_dg * dg_u[i] + c.rho * s * phi[i]);
    }
    for i in 0..n_int {
        out.push(c.rho * v_term[i] + dg_v[i]);
    }
    out.push(dot(c.proj, phi) - target);
    out
}

/// The one reduction on the path. Left to right, as `np.dot` is not — see the module header on why
/// nothing here is claimed exact.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The exact (non-symmetric) Jacobian of [`residual`], bordered by the `d/ds` column and the
/// amplitude row.
///
/// Non-symmetric because `cos(Omega k)` multiplies the transverse row only, so
/// `dF_phi/dpsi = cos(Omega k) dF_psi/dphi`. The cell blocks are [`planar_hessian_cells`]; the
/// `d/ds` column is where a dropped factor hides, which is why the tests finite-difference-check
/// it against [`residual`].
fn jacobian(phi: &[f64], psi: &[f64], s: f64, c: &NewtonCtx) -> Csr {
    let n_int = phi.len();
    let p = c.gp.matvec(phi);
    let z = c.gp.matvec(psi);
    let (h_pp, h_pz, h_zz) = planar_hessian_cells(&p, &z, c.a);
    let (_, chi, _) = stretch(&p, &z);
    let (fac_lin, fac_dg) = c.factors(s);

    let sandwich = |diag: &[f64]| c.gm.matmul(&Csr::diagonal(diag)).matmul(c.gp);
    let j_pp = c
        .op_u
        .scaled(c.rho * fac_lin)
        .add(&sandwich(&h_pp).scaled(fac_dg))
        .add(&c.ident.scaled(c.rho * s));
    let j_pz = sandwich(&h_pz).scaled(fac_dg);
    let j_zp = sandwich(&h_pz);
    let j_zz = c.op_v.scaled(c.rho).add(&sandwich(&h_zz));

    let d_phi_ds: Vec<f64> = if c.time_discrete {
        let u_term = c.op_u.matvec(phi);
        let dg_u: Vec<f64> = (0..p.len()).map(|i| c.a * chi[i] * p[i]).collect();
        let dg_u = c.gm.matvec(&dg_u);
        (0..n_int)
            .map(|i| {
                -c.rho * c.theta * c.k * c.k * u_term[i] - 0.5 * c.k * c.k * dg_u[i]
                    + c.rho * phi[i]
            })
            .collect()
    } else {
        phi.iter().map(|v| c.rho * v).collect()
    };

    let dim = 2 * n_int + 1;
    let mut rows: Vec<Vec<(usize, f64)>> = Vec::with_capacity(dim);
    for (i, ds) in d_phi_ds.iter().enumerate() {
        let mut row: Vec<(usize, f64)> = csr_row(&j_pp, i).collect();
        row.extend(csr_row(&j_pz, i).map(|(j, v)| (j + n_int, v)));
        row.push((2 * n_int, *ds));
        rows.push(row);
    }
    for i in 0..n_int {
        let mut row: Vec<(usize, f64)> = csr_row(&j_zp, i).collect();
        row.extend(csr_row(&j_zz, i).map(|(j, v)| (j + n_int, v)));
        rows.push(row);
    }
    rows.push(c.proj.iter().copied().enumerate().collect());
    Csr::from_rows(dim, dim, rows)
}

/// Row `i` of a CSR matrix as `(column, value)` pairs.
fn csr_row(m: &Csr, i: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
    let (lo, hi) = (m.indptr()[i], m.indptr()[i + 1]);
    (lo..hi).map(move |p| (m.indices()[p], m.data()[p]))
}

/// Newton on `[phi; psi; s]`, in place; returns `(iterations, converged)`.
///
/// The convergence bar is on the **relative step**: the residual's blocks carry different units
/// (two force densities and a displacement), so a max-norm bar on it would mix them. Displacement
/// steps are measured against `max|phi|` — `psi` included, since it can legitimately be ~0 at small
/// amplitude and would otherwise divide by nothing.
fn newton(
    phi: &mut [f64],
    psi: &mut [f64],
    s: &mut f64,
    target: f64,
    c: &NewtonCtx,
    tol: f64,
    maxiter: usize,
) -> Result<(usize, bool), String> {
    let n_int = phi.len();
    for it in 1..=maxiter {
        let r = residual(phi, psi, *s, target, c);
        let jac = jacobian(phi, psi, *s, c);
        let lu = SparseLu::factor_permuted(&jac, c.perm, DIAG_PIVOT_THRESH)
            .map_err(|e| format!("the rotating-wave Jacobian did not factor: {e}"))?;
        let rhs: Vec<f64> = r.iter().map(|v| -v).collect();
        let delta = lu
            .solve(&rhs)
            .map_err(|e| format!("the rotating-wave Jacobian did not solve: {e}"))?;

        for i in 0..n_int {
            phi[i] += delta[i];
            psi[i] += delta[n_int + i];
        }
        *s += delta[2 * n_int];

        let scale = phi
            .iter()
            .fold(0.0f64, |m, v| m.max(v.abs()))
            .max(f64::MIN_POSITIVE);
        let step = delta[..2 * n_int]
            .iter()
            .fold(0.0f64, |m, v| m.max(v.abs()));
        let err = (step / scale).max(delta[2 * n_int].abs() / s.abs().max(f64::MIN_POSITIVE));
        if err <= tol {
            return Ok((it, true));
        }
    }
    Ok((maxiter, false))
}

#[cfg(test)]
mod tests {
    //! The two structural claims about the Jacobian, which live here rather than in
    //! `tests/rotating_wave.rs` because `residual`, `jacobian` and `NewtonCtx` are private and
    //! ought to stay so — an integration test cannot reach them.
    //!
    //! They were in `tests/test_geometric_rotating_wave.py` until unit 10's deletion (plan §44),
    //! reaching for `rw._operators`, `rw._residual` and `rw._jacobian` on the Python module. That
    //! module has no body any more, and this file's own header had been claiming these bars for a
    //! phase without them existing — an overclaim the deletion is what found.
    //!
    //! The fixture is the retired `tests/helpers.py` string, evaluated: `L = 1`, `T = 200`,
    //! `rho = 0.005`, `EA = 1e5`, `kappa = 0` (bending is irrelevant to the relative equilibrium),
    //! `N = 32`, `theta = 0.28`, and the `fs` the helper derives from `lam_long = 0.5`.

    use super::*;

    const L: f64 = 1.0;
    const T: f64 = 200.0;
    const RHO: f64 = 0.005;
    const EA: f64 = 1.0e5;
    const N_CELLS: usize = 32;
    const THETA: f64 = 0.28;
    const FS: f64 = 286_216.701_119_973_1;
    const AMP: f64 = 5e-3;

    /// Everything `residual` and `jacobian` need, plus the state to evaluate them at.
    struct Fixture {
        op_u: Csr,
        op_v: Csr,
        gp: Csr,
        gm: Csr,
        ident: Csr,
        proj: Vec<f64>,
        perm: Vec<usize>,
        phi: Vec<f64>,
        psi: Vec<f64>,
        s: f64,
    }

    fn fixture(perturb: bool) -> Fixture {
        let n_int = N_CELLS - 1;
        let h = L / N_CELLS as f64;
        let (d2, gp, gm) = operators(N_CELLS, h);
        let shape: Vec<f64> = (1..N_CELLS)
            .map(|i| (std::f64::consts::PI * (L * i as f64 / N_CELLS as f64) / L).sin())
            .collect();
        // A deterministic stand-in for the Python test's seeded normals: a wobble that is
        // incommensurate with the mode, so no symmetry of the operator can hide a wrong entry.
        let wobble = |i: usize, scale: f64| scale * ((i as f64 * 1.7 + 0.3).sin());
        let phi: Vec<f64> = shape
            .iter()
            .enumerate()
            .map(|(i, v)| AMP * v + if perturb { wobble(i, 1e-4) } else { 0.0 })
            .collect();
        let psi: Vec<f64> = (0..n_int)
            .map(|i| if perturb { wobble(i + 7, 1e-5) } else { 0.0 })
            .collect();
        Fixture {
            op_u: d2.scaled(T / RHO),
            op_v: d2.scaled(EA / RHO),
            gp,
            gm,
            ident: Csr::identity(n_int),
            proj: shape.iter().map(|v| (2.0 / L) * h * v).collect(),
            perm: interleave_permutation(n_int),
            phi,
            psi,
            s: (200.0 * std::f64::consts::PI) * (200.0 * std::f64::consts::PI),
        }
    }

    fn ctx(f: &Fixture, time_discrete: bool) -> NewtonCtx<'_> {
        NewtonCtx {
            op_u: &f.op_u,
            op_v: &f.op_v,
            gp: &f.gp,
            gm: &f.gm,
            ident: &f.ident,
            proj: &f.proj,
            perm: &f.perm,
            rho: RHO,
            a: EA - T,
            k: 1.0 / FS,
            theta: THETA,
            time_discrete,
        }
    }

    /// The Jacobian against central differences of the residual — including the `d/ds` column.
    ///
    /// Newton converges to a root of the *residual*, so a wrong Jacobian shows up as slow
    /// convergence rather than a wrong answer, which means it can be wrong for a long time without
    /// anything failing. The `d/ds` column is the one to watch: it carries the derivative of
    /// **both** time factors, and dropping either leaves a column that is merely *nearly* right.
    #[test]
    fn the_jacobian_matches_central_differences_of_the_residual() {
        let f = fixture(true);
        let n = N_CELLS - 1;
        for time_discrete in [true, false] {
            let c = ctx(&f, time_discrete);
            let jac = jacobian(&f.phi, &f.psi, f.s, &c);
            let mut worst_gap = 0.0f64;
            let mut scale = 0.0f64;
            for j in 0..2 * n + 1 {
                // The step is 1e-6 of the entry and not smaller, and that is measured rather
                // than guessed. Scanning it here gives 3.0e-9 at 1e-5, 4.9e-8 at 1e-6, 1.5e-6 at
                // 1e-7 and 1.5e-5 at 1e-8 — the gap *grows* as the step shrinks, which is
                // cancellation in the difference and not truncation in the Jacobian. A test that
                // took the smallest step it could would be measuring its own subtraction.
                let eps = if j < n {
                    1e-6 * f.phi[j].abs().max(1e-6)
                } else if j < 2 * n {
                    1e-6 * f.psi[j - n].abs().max(1e-9)
                } else {
                    1e-6 * f.s
                };
                let mut cols: Vec<Vec<f64>> = Vec::with_capacity(2);
                for sign in [1.0f64, -1.0] {
                    let mut phi = f.phi.clone();
                    let mut psi = f.psi.clone();
                    let mut s = f.s;
                    if j < n {
                        phi[j] += sign * eps;
                    } else if j < 2 * n {
                        psi[j - n] += sign * eps;
                    } else {
                        s += sign * eps;
                    }
                    cols.push(residual(&phi, &psi, s, AMP, &c));
                }
                for (i, (plus, minus)) in cols[0].iter().zip(cols[1].iter()).enumerate() {
                    let fd = (plus - minus) / (2.0 * eps);
                    scale = scale.max(fd.abs());
                    worst_gap = worst_gap.max((jac.get(i, j) - fd).abs());
                }
            }
            assert!(
                worst_gap / scale < 1e-6,
                "time_discrete={time_discrete}: the Jacobian is {:.3e} of the finite-difference \
                 scale away from it",
                worst_gap / scale
            );
        }
    }

    /// `dF_phi/dpsi == cos(Omega k) dF_psi/dphi` — the structural signature of the 2k-wide DG.
    ///
    /// The reduced system *looks* variational: its cell blocks are the Hessian of `V_nl` on the
    /// planar strain slice, which is symmetric. It is not, and the asymmetry is not roundoff — it
    /// is the discrete gradient spanning `q^{n+1}` to `q^{n-1}`, which puts a `cos(Omega k)` on the
    /// transverse row and nothing on the longitudinal one.
    ///
    /// Assuming the symmetry (and reaching for a Cholesky, as the family's *linear* models do)
    /// would stall Newton against a Jacobian wrong by one part in 2e5 at these settings — small
    /// enough to look like a conditioning problem rather than a bug. So: the exact relation, the
    /// plain symmetry **failing**, and the symmetry restored as `k -> 0`.
    #[test]
    fn the_jacobian_asymmetry_is_exactly_the_discrete_gradient_time_factor() {
        let f = fixture(false);
        let n = N_CELLS - 1;
        let k = 1.0 / FS;

        let jac = jacobian(&f.phi, &f.psi, f.s, &ctx(&f, true));
        let cos_k = 1.0 - 0.5 * k * k * f.s;
        let mut scale = 0.0f64;
        let mut worst_relation = 0.0f64;
        let mut worst_symmetry = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let pz = jac.get(i, n + j);
                let zp = jac.get(n + i, j);
                scale = scale.max(pz.abs());
                worst_relation = worst_relation.max((pz - cos_k * zp).abs());
                worst_symmetry = worst_symmetry.max((pz - zp).abs());
            }
        }
        assert!(
            worst_relation / scale < 1e-14,
            "the cos(Omega k) relation is off by {:.3e}",
            worst_relation / scale
        );
        assert!(
            worst_symmetry / scale > 1e-9,
            "the Jacobian came back symmetric ({:.3e}), so the discrete gradient's 2k span has \
             gone missing and Newton is being handed a variational lie",
            worst_symmetry / scale
        );

        // ... and at `k -> 0` (the semi-discrete branch) it IS symmetric.
        let jac0 = jacobian(&f.phi, &f.psi, f.s, &ctx(&f, false));
        let mut scale0 = 0.0f64;
        let mut worst0 = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let pz = jac0.get(i, n + j);
                scale0 = scale0.max(pz.abs());
                worst0 = worst0.max((pz - jac0.get(n + i, j)).abs());
            }
        }
        assert!(
            worst0 / scale0 < 1e-14,
            "the semi-discrete Jacobian is not symmetric ({:.3e})",
            worst0 / scale0
        );
    }
}
