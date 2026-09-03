//! The rotating-wave BVP — `physsynth/analysis/rotating_wave.py`, transcribed.
//!
//! The geometrically exact string's Tier B oracle: the *exact* relative equilibrium of the scheme,
//! a rigid helix `u = φ cos Ωt`, `w = φ sin Ωt`, `v = ψ`. Seed a `GeometricString` with it and the
//! string rotates to round-off; that is the sharpest statement this project makes about model #10,
//! and it is the reason a 602-line BVP solver lives in an analysis package.
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
//! Two things *are* claimed across every fixture tried, and the parity file asserts both:
//!
//! * **`converged` agrees in 126 of 126.** Eighteen fixtures fail to converge, and they are the
//!   same eighteen on both sides. That is the one flag a caller is told to branch on, so it is the
//!   one that has to agree.
//! * **Among the converged, `Ω`, `s`, `φ`, `ψ` and the tension field agree to ~1e-14 or better.**
//!
//! Where the BVP does *not* converge the two implementations produce different answers and the
//! parity file asserts nothing about them beyond the flag. That is not a dodge: `converged == false`
//! means "the helix is not a relative equilibrium; do not treat this as physics". It is also the
//! same lesson `docs/dev/scientific-hurdles.md` §6 landed on the same day — **a margin measured in
//! the converged regime says nothing about the failing one.** There the iteration path *is* the
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
pub const CONTINUATION_STEPS_DEFAULT: usize = 8;

/// **Relative** convergence bar on the Newton *step*, not on the residual.
///
/// The residual's three blocks carry different units — a transverse force density, a longitudinal
/// force density and a displacement — so a single max-norm bar on it would be a unit-mixing
/// accident. The step is compared against the solution's own scale, which is unit-free per block.
pub const NEWTON_TOL_DEFAULT: f64 = 1e-14;

/// Cap on Newton iterations per continuation step. Reaching it reports `converged == false`.
pub const NEWTON_MAXITER_DEFAULT: usize = 50;

/// A converged rotating-wave relative equilibrium: the helix, its stretch, its frequency.
///
/// Field for field the Python `RotatingWave` NamedTuple, plus [`Self::failed_step`] and
/// [`Self::failed_amplitude`], which carry what the Python side needs to raise its non-convergence
/// warning with the same text. The Python tuple is immutable and no caller assigns to it or calls
/// `_replace` (grepped over `tests/`, `scripts/` and `web/serialize.py`), so the binding exposes
/// getters only.
#[derive(Clone, Debug)]
pub struct RotatingWave {
    /// Transverse profile on the **full grid** `(N+1)`, zero at both clamped ends.
    pub phi: Vec<f64>,
    /// Longitudinal profile on the **full grid** `(N+1)`. **Static, and nonzero.**
    pub psi: Vec<f64>,
    /// Angular frequency (rad/s) of the rotation.
    pub omega: f64,
    /// `Omega / (2 pi)` (Hz).
    pub frequency: f64,
    /// The scheme's `s = (4/k²) sin²(Ωk/2)` — the unknown solved for.
    pub s: f64,
    /// The imposed modal amplitude `R` (m).
    pub amplitude: f64,
    /// The mode number continued up from.
    pub mode: usize,
    /// Per-cell `Lambda` `(N)` — the frozen stretch field.
    pub stretch_ratio: Vec<f64>,
    /// Per-cell axial tension `EA Lambda - a` (N) `(N)` — **non-uniform**, which is the point.
    pub tension: Vec<f64>,
    /// `max|phi - R sin(m pi x / L)| / max|phi|` — the deviation from a sine.
    pub shape_residual: f64,
    /// Total Newton iterations summed over the continuation.
    pub iterations: usize,
    /// Whether every continuation step converged.
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
/// rearrangements the core uses. Both are `O(strain)` quantities assembled from `O(1)` ones, and
/// musical strings run at strain 1e-4 .. 1e-3; evaluating them literally would converge this BVP to
/// a root of a slightly *different* equation than the one the core steps, and the seeded helix
/// would shed the difference into the longitudinal field.
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
/// The helix is three-dimensional but its reduced `(phi, psi)` system linearises on the planar
/// slice `q = (p, 0, z)`, because `V_nl` sees the polarizations only through `r² = p²`. Three
/// entries survive, and the third simplifies exactly through `(1+z)² - Lambda² = -p²`:
///
/// ```text
/// H_pp = a (chi + p² / Lambda³)
/// H_pz = a p (1 + z) / Lambda³
/// H_zz = -a p² / Lambda³
/// ```
///
/// That last identity written literally is a difference of `O(1)` terms and is catastrophic at
/// musical strain; written this way it has no cancellation at all. `H_zz` is *negative* for a
/// hardening string — the nonlinear excess softens the longitudinal direction — but the full
/// longitudinal diagonal adds `EA` and stays positive.
///
/// Public because it cross-checks against the core: `H/2` is exactly
/// `GeometricString::dg_jacobian(q, q)`, the `1/2` being `d(qbar)/d(q+)`. That check cannot be a
/// native test in this crate without the dependency edge `tests/deps.rs` refuses, so it stays the
/// Python test it has always been (`test_geometric_rotating_wave.py`, at 1e-12 since the core's
/// `(v,v)` block was made cancellation-free).
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
/// Note the missing half: a *planar* mode's stretch averages to `R²p²L/4`, a circular mode's to
/// `R²p²L/2` — exactly twice, and constant rather than pulsing, because `cos² + sin² = 1`. So the
/// tension shift is the full `eps R²` with no factor of `1/2`, and this is *not* the planar Duffing
/// frequency at the same `R`.
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
/// lives in this function. `fs` must be the resonator's own, since `Omega` was solved for at that
/// `k` when `time_discrete`.
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
#[derive(Clone, Copy, Debug)]
pub struct BvpParams {
    pub l: f64,
    pub t: f64,
    pub rho: f64,
    pub ea: f64,
    pub fs: f64,
    pub n_cells: usize,
    pub theta: f64,
    pub amplitude: f64,
    pub mode: usize,
    pub kappa: f64,
    pub time_discrete: bool,
    pub continuation_steps: usize,
    pub tol: f64,
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
    //! The fixture is `tests/helpers.py`'s own string, evaluated: `L = 1`, `T = 200`,
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
