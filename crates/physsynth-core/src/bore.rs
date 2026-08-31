//! Acoustic bore — the 1-D air column of a wind instrument (Webster's horn equation).
//!
//! Port of `physsynth/core/bore.py`, the wind leg's resonator and the first **acoustic** model in
//! the project. Pressure `p` lives on `N + 1` integer nodes, volume velocity `U = S v` on the `N`
//! half-nodes between them and half a timestep offset in time — a Yee cell in one dimension:
//!
//! ```text
//! U_{l+1/2}^{n+1/2} = U^{n-1/2} - (k S_{l+1/2} / (rho0 h)) (p_{l+1}^n - p_l^n)
//! p_l^{n+1}         = p_l^n - (k rho0 c0^2 / (w_l S_l)) (U_{l+1/2}^{n+1/2} - U_{l-1/2}^{n+1/2})
//! ```
//!
//! Pressure updates from the divergence of the velocity, velocity from the gradient of the
//! pressure — the same difference operator and its transpose, which is exactly what makes the
//! discrete energy telescope. The reference docstring in `bore.py` is the physics; this module
//! documents only what the translation had to decide.
//!
//! # This is the first model in the crate whose step takes a caller's hook
//!
//! `Bore.step(source=...)` hands the freshly-updated pressure field to an outside party which
//! mutates it in place, **between** the pressure and momentum sub-steps. That is the seam an
//! implicit boundary exciter injects through, and `reed::ReedBore` is the caller it exists for.
//!
//! Here it is a plain Rust closure, [`Source`]. It is deliberately *not* a PyO3 callable: putting
//! a Python type in this crate would break the empty dependency list that
//! `crates/physsynth-core/tests/deps.rs` exists to guard, and that is the one mistake in this
//! batch which would be expensive to undo. The binding wraps a Python callable into one of these
//! instead, so the capability survives without the dependency. The plan's §12.8 recorded that the
//! hook is load-bearing rather than transitional: `tests/test_reed_stability.py` passes its own
//! `lambda p: None` to assert the hook is inert when unused, so porting `reed` removes the *hot*
//! crossing, never the capability.
//!
//! # The ordering inside a step is load-bearing and no energy test can see it
//!
//! Open-end pin, then the `source` hook, then the radiating drain, then the momentum sub-step —
//! so that `U^{n+3/2}` sees a node pressure corrected by both the exciter and the bell. Reorder it
//! and the reed still oscillates and the books still roughly balance; the project's own reed work
//! established that balance is not a sufficient detector there and the *signature* oracle is.
//!
//! # Operation order
//!
//! As everywhere in this crate, the kernels reproduce NumPy's evaluation order rather than merely
//! its algebra, so the elementwise updates agree bit-for-bit. Two spots needed measuring rather
//! than reasoning, both recorded at their definitions: the `c0 ** 2` in the node compliance (see
//! [`Params::new`]) and the zero-seeded accumulation in [`divergence_into`]. The only read-outs
//! that cannot be bit-identical are the two `np.dot` reductions in [`acoustic_energy`].

use crate::pyfloat::scalar_pow;
use crate::sparse::Csr;

/// A per-end termination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum End {
    /// Rigid wall: `U = 0` just outside, a pressure antinode. Needs no ghost stencil — the `h/2`
    /// trapezoidal half-cell at that node *is* the closure.
    Closed,
    /// Pressure-release: `p = 0` pinned (Dirichlet), a pressure node. Lossless and perfectly
    /// reflecting; it radiates nothing.
    Open,
    /// A passively-lossy bell of resistance `R_bell`: a live half-cell node drained by a rank-1
    /// dashpot, which sheds energy to the far field.
    Radiating,
}

impl End {
    /// Parse the Python spelling. `None` for anything not in `("closed", "open", "radiating")`.
    pub fn parse(s: &str) -> Option<End> {
        match s {
            "closed" => Some(End::Closed),
            "open" => Some(End::Open),
            "radiating" => Some(End::Radiating),
            _ => None,
        }
    }

    /// The Python spelling, for echoing `.boundary` back.
    pub fn name(self) -> &'static str {
        match self {
            End::Closed => "closed",
            End::Open => "open",
            End::Radiating => "radiating",
        }
    }
}

/// A construction-time rejection. Every variant's `Display` is the Python original's message
/// verbatim, because the suite matches on the text.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamError {
    /// One of `L`, `fs`, `radius`, `rho0`, `c0` was not positive.
    NonPositiveScalar,
    /// `N < 2`.
    TooFewSegments,
    /// `sigma < 0`.
    NegativeSigma,
    /// `R_bell < 0`.
    NegativeRBell,
    /// A boundary token was not one of the three. The message quotes the caller's object, which
    /// only the caller can `repr()`, so it is formatted at the parse site instead — as
    /// `string_ideal::ParamError::BadBoundary` already is.
    BadBoundary,
    /// A `"radiating"` end without `R_bell > 0`.
    RadiatingNeedsResistance,
    /// `lambda = c0 k / h > 1`. Carries the offending `lambda`.
    CflViolated(f64),
}

impl std::fmt::Display for ParamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamError::NonPositiveScalar => {
                write!(f, "L, fs, radius, rho0, c0 must all be positive.")
            }
            ParamError::TooFewSegments => {
                write!(f, "N must be >= 2 (need at least one interior node).")
            }
            ParamError::NegativeSigma => write!(f, "sigma (loss) must be >= 0."),
            ParamError::NegativeRBell => {
                write!(f, "R_bell (radiation resistance) must be >= 0.")
            }
            ParamError::BadBoundary => write!(
                f,
                "each boundary end must be one of ('closed', 'open', 'radiating')."
            ),
            ParamError::RadiatingNeedsResistance => write!(
                f,
                "a 'radiating' end needs R_bell > 0 (the bell's radiation resistance). \
                 Use 'open' for the ideal lossless pressure-release end (R -> 0)."
            ),
            ParamError::CflViolated(lam) => write!(
                f,
                "CFL violated: lambda = c0*k/h = {lam:.6} > 1. \
                 Reduce fs, refine the grid (increase N), or shorten the tube."
            ),
        }
    }
}

impl std::error::Error for ParamError {}

/// The slack in the CFL check, matching `bore._LAMBDA_TOL`.
const LAMBDA_TOL: f64 = 1e-12;

/// The validated, immutable parameter set: everything about a bore that is not its state.
#[derive(Debug, Clone)]
pub struct Params {
    /// Tube length (m).
    pub l: f64,
    /// Sample rate (Hz).
    pub fs: f64,
    /// Number of segments; `N + 1` pressure nodes and `N` velocity half-nodes.
    pub n: usize,
    /// Bore radius (m).
    pub radius: f64,
    /// Left-end termination.
    pub bc_left: End,
    /// Right-end termination.
    pub bc_right: End,
    /// Viscous loss coefficient in the `-2 sigma U` drag.
    pub sigma: f64,
    /// Radiation resistance of a `"radiating"` end (Pa·s/m³).
    pub r_bell: f64,
    /// Air density (kg/m³).
    pub rho0: f64,
    /// Sound speed (m/s).
    pub c0: f64,
    /// Node spacing `L / N` (m).
    pub h: f64,
    /// Timestep `1 / fs` (s).
    pub k: f64,
    /// Courant number `c0 k / h`.
    pub lam: f64,
    /// Cross-section at the pressure nodes; length `N + 1`.
    pub s_node: Vec<f64>,
    /// Cross-section at the velocity half-nodes; length `N`.
    pub s_seg: Vec<f64>,
    /// Trapezoidal node weight: `h` interior, `h/2` at each end. Length `N + 1`.
    pub w: Vec<f64>,
    /// Diagonal node compliance `C_l = w_l S_l / (rho0 c0^2)`; length `N + 1`.
    pub c: Vec<f64>,
    /// Diagonal segment inductance `M_j = h rho0 / S_j`; length `N`.
    pub m: Vec<f64>,
    /// Pressure update prefactor `k / C_l`; length `N + 1`.
    pub p_pref: Vec<f64>,
    /// Momentum update prefactor `k / M_j`; length `N`.
    pub u_pref: Vec<f64>,
    /// `C[0] / k` — the left node's compliance rate, the `a` of the rank-1 radiating solve.
    pub a_left: f64,
    /// `C[N] / k` — the right node's compliance rate.
    pub a_right: f64,
    /// Characteristic acoustic impedance `rho0 c0 / S`.
    pub z0: f64,
}

impl Params {
    /// Validate and derive.
    ///
    /// The checks run in the original's order — scalars, `N`, `sigma`, `R_bell`, the boundary
    /// tokens, the radiating end's resistance, then CFL — so a call with two faults reports the
    /// same one. `boundary` arrives already parsed because the rejection message for a bad token
    /// quotes the caller's object.
    ///
    /// # The `c0 ** 2`
    ///
    /// The compliance denominator is `rho0 * c0**2`, and `c0 ** 2` in Python calls libm's `pow`,
    /// not a multiply. The two agree at the ambient 343 m/s (343² = 117649 is exact in doubles)
    /// but **not in general**: measured over 200,000 random positive doubles, `x ** 2` and `x * x`
    /// disagree in 79. Same class of finding as the plan's `h ** 4` (§10.3), so it is spelled the
    /// same way — a real `pow`, not `c0 * c0`.
    ///
    /// It has to go through [`crate::pyfloat::scalar_pow`] rather than a bare `c0.powf(2.0)`,
    /// because LLVM folds a literal exponent of 2.0 straight back into a multiply and does so only
    /// in `--release` (§17.2). Written as a bare `powf` this call was the fold's *shipped* case for
    /// six batches: at the ambient 343 m/s nothing can see it (343² is exact), and one ulp above
    /// it the release build's bore and the reference's separate at 9.4e-15 of amplitude over 200
    /// steps. Measured 2026-08-31, before and after.
    ///
    /// This matters twice over, because `reed` computes the *same physical quantity* with the
    /// other spelling — `rho0 * c0 * c0` — and the two disagree by one ulp in 3,531 of 3,552
    /// tube/grid combinations. That divergence predates the migration and is deliberately
    /// preserved on both sides; see `reed::Params::new`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        l: f64,
        fs: f64,
        n: usize,
        radius: f64,
        boundary: Option<(End, End)>,
        sigma: f64,
        r_bell: f64,
        rho0: f64,
        c0: f64,
    ) -> Result<Params, ParamError> {
        if l <= 0.0 || fs <= 0.0 || radius <= 0.0 || rho0 <= 0.0 || c0 <= 0.0 {
            return Err(ParamError::NonPositiveScalar);
        }
        if n < 2 {
            return Err(ParamError::TooFewSegments);
        }
        if sigma < 0.0 {
            return Err(ParamError::NegativeSigma);
        }
        if r_bell < 0.0 {
            return Err(ParamError::NegativeRBell);
        }
        let (bc_left, bc_right) = boundary.ok_or(ParamError::BadBoundary)?;
        let radiating = bc_left == End::Radiating || bc_right == End::Radiating;
        if radiating && r_bell <= 0.0 {
            return Err(ParamError::RadiatingNeedsResistance);
        }

        let h = l / (n as f64);
        let k = 1.0 / fs;
        // `self.c0 * self.k / self.h`, left to right.
        let lam = (c0 * k) / h;
        if lam > 1.0 + LAMBDA_TOL {
            return Err(ParamError::CflViolated(lam));
        }

        // `np.pi * self.radius * self.radius`, left to right.
        let area = (std::f64::consts::PI * radius) * radius;
        let s_node = vec![area; n + 1];
        let s_seg = vec![area; n];

        let mut w = vec![h; n + 1];
        w[0] = 0.5 * h;
        w[n] = 0.5 * h;

        // `self._w * self.S_node / (self.rho0 * self.c0**2)` — the product before the divide, and
        // the denominator formed once. See the note above on `powf(2.0)`.
        let denom = rho0 * scalar_pow(c0, 2.0);
        let c: Vec<f64> = (0..=n).map(|i| (w[i] * s_node[i]) / denom).collect();
        // `self.h * self.rho0 / self.S_seg`.
        let m: Vec<f64> = (0..n).map(|j| (h * rho0) / s_seg[j]).collect();
        let p_pref: Vec<f64> = c.iter().map(|&ci| k / ci).collect();
        let u_pref: Vec<f64> = m.iter().map(|&mj| k / mj).collect();

        let a_left = c[0] / k;
        let a_right = c[n] / k;
        // `self.rho0 * self.c0 / area`, left to right.
        let z0 = (rho0 * c0) / area;

        Ok(Params {
            l,
            fs,
            n,
            radius,
            bc_left,
            bc_right,
            sigma,
            r_bell,
            rho0,
            c0,
            h,
            k,
            lam,
            s_node,
            s_seg,
            w,
            c,
            m,
            p_pref,
            u_pref,
            a_left,
            a_right,
            z0,
        })
    }

    /// Number of pressure nodes, `N + 1`.
    pub fn nodes(&self) -> usize {
        self.n + 1
    }

    /// Whether the left end is pinned to `p = 0`.
    pub fn open_left(&self) -> bool {
        self.bc_left == End::Open
    }

    /// Whether the right end is pinned to `p = 0`.
    pub fn open_right(&self) -> bool {
        self.bc_right == End::Open
    }

    /// Whether the left end is a radiating bell.
    pub fn rad_left(&self) -> bool {
        self.bc_left == End::Radiating
    }

    /// Whether the right end is a radiating bell.
    pub fn rad_right(&self) -> bool {
        self.bc_right == End::Radiating
    }

    /// Pressure-node positions — `np.linspace(0.0, L, N + 1)`, reproduced exactly.
    ///
    /// NumPy computes `i * step` with `step = L / N` and then *overwrites* the last entry with the
    /// endpoint, so `x[N]` is `L` exactly rather than `N * (L / N)`. The two differ in the last bit
    /// for most lengths, and `x` reaches the viewer and the analysis layer.
    pub fn grid(&self) -> Vec<f64> {
        let step = self.l / (self.n as f64);
        let mut x: Vec<f64> = (0..self.nodes()).map(|i| (i as f64) * step).collect();
        x[self.n] = self.l;
        x
    }

    /// Velocity half-node positions — `0.5 * (x[1:] + x[:-1])`.
    pub fn grid_u(&self) -> Vec<f64> {
        let x = self.grid();
        (0..self.n).map(|j| 0.5 * (x[j + 1] + x[j])).collect()
    }

    /// The free (non-open) pressure nodes: the DOFs of the modal eigenproblem.
    ///
    /// `np.nonzero(free)[0]` — ascending, and `int64` on the Python side.
    pub fn dof(&self) -> Vec<usize> {
        (0..self.nodes())
            .filter(|&i| !((i == 0 && self.open_left()) || (i == self.n && self.open_right())))
            .collect()
    }

    /// The pressure stiffness `L = G^T M^{-1} G` and the mass `C`, for the modal oracle.
    ///
    /// `G` is the `N x (N+1)` gradient with `-1` at column `j` and `+1` at column `j+1`. The
    /// scheme eliminates `U` to `C d_tt p = -k^2 L p`, so the generalized eigenvalues of `(L, C)`
    /// restricted to [`Self::dof`] are `omega^2`.
    ///
    /// The product is associated `(G^T @ Minv) @ G`, which is how Python's left-to-right `@` chain
    /// associates it, and [`Csr::matmul`] accumulates in ascending order of the contracted index
    /// as SciPy's kernel does — so the stored values agree bit-for-bit even though the sparsity
    /// *ordering* may not (see `sparse`'s header).
    pub fn pressure_operator(&self) -> (Csr, Csr) {
        let n_seg = self.n;
        let rows: Vec<Vec<(usize, f64)>> =
            (0..n_seg).map(|j| vec![(j, -1.0), (j + 1, 1.0)]).collect();
        let g = Csr::from_rows(n_seg, self.nodes(), rows);
        let minv = Csr::diagonal(&self.m.iter().map(|&mj| 1.0 / mj).collect::<Vec<f64>>());
        let lop = g.transpose().matmul(&minv).matmul(&g);
        let cmat = Csr::diagonal(&self.c);
        (lop, cmat)
    }
}

/// The seam an implicit boundary exciter injects through: a caller's in-place corrector applied to
/// the freshly-updated pressure field, after the open-end pin and before the radiating drain.
///
/// See the module header for why this is a Rust closure and not a Python callable.
pub type Source<'a> = &'a mut dyn FnMut(&mut [f64]);

// -- kernels ---------------------------------------------------------------------------------
//
// Free functions over slices, as in `string_ideal`, `membrane` and `body`: they hold no state and
// write only into what they are given, so the native struct below can keep its buffers in `Vec`s
// while the Python binding keeps the same buffers in NumPy arrays.

/// One momentum half-step `U^{+1/2} = [(1 - sk) U^{-1/2} - u_pref (grad p)] / (1 + sk)`.
///
/// # Panics
/// If any slice has the wrong length.
pub fn momentum_into(p: &[f64], u_prev: &[f64], out: &mut [f64], params: &Params) {
    assert_eq!(p.len(), params.nodes(), "p must have N+1 entries");
    assert_eq!(u_prev.len(), params.n, "u_prev must have N entries");
    assert_eq!(out.len(), params.n, "out must have N entries");

    let sk = params.sigma * params.k;
    for j in 0..params.n {
        let grad_p = p[j + 1] - p[j];
        out[j] = ((1.0 - sk) * u_prev[j] - params.u_pref[j] * grad_p) / (1.0 + sk);
    }
}

/// Discrete divergence `(G^T U)_l = U_{l+1/2} - U_{l-1/2}`, with zero wall ghosts.
///
/// Node `0` sees only the segment to its right and node `N` only the one to its left — the
/// rigid-wall closure, no ghost velocity needed.
///
/// **The zero seeding is reproduced literally.** The original allocates `np.zeros(N+1)` and then
/// runs two whole-array updates, `div[:-1] += u` followed by `div[1:] -= u`. For a finite value
/// `0.0 + u` is `u`, but for `u = -0.0` it is `+0.0` — so writing `out[l] = u[l] - u[l-1]`
/// directly would differ in the sign of zero at a node whose two segments are both negative zero.
/// That is invisible in every energy bar and visible to `np.array_equal`, which is precisely the
/// class of difference the parity files exist to catch.
///
/// # Panics
/// If any slice has the wrong length.
pub fn divergence_into(u: &[f64], out: &mut [f64], params: &Params) {
    assert_eq!(u.len(), params.n, "u must have N entries");
    assert_eq!(out.len(), params.nodes(), "out must have N+1 entries");

    out.fill(0.0);
    for l in 0..params.n {
        out[l] += u[l];
    }
    for l in 1..=params.n {
        out[l] -= u[l - 1];
    }
}

/// Pin any open end to `p = 0` (Dirichlet).
pub fn apply_open_ends(p: &mut [f64], params: &Params) {
    if params.open_left() {
        p[0] = 0.0;
    }
    if params.open_right() {
        p[params.n] = 0.0;
    }
}

/// The pressure sub-step `p^{n+1} = p^n - p_pref * div(U^{n+1/2})`, followed by the open-end pin.
///
/// `scratch` is the divergence workspace; it is a parameter rather than a local so that the
/// binding can hoist the allocation out of the timestep if it ever wants to.
///
/// # Panics
/// If any slice has the wrong length.
pub fn pressure_into(p: &[f64], u: &[f64], out: &mut [f64], scratch: &mut [f64], params: &Params) {
    assert_eq!(p.len(), params.nodes(), "p must have N+1 entries");
    assert_eq!(out.len(), params.nodes(), "out must have N+1 entries");

    divergence_into(u, scratch, params);
    for l in 0..params.nodes() {
        out[l] = p[l] - params.p_pref[l] * scratch[l];
    }
    apply_open_ends(out, params);
}

/// The rank-1 dashpot at one end node: correct `p_next[idx]` in place, book the energy it shed
/// into `radiated_energy`, and return its `U_out`.
///
/// The un-pinned value in `p_next` is already the **rigid** (force-free) half-cell step
/// `p_rigid`. The centered resistor `U_out = (p^{n+1} + p^n) / (2R)` turns that into the 1×1 solve
/// `p^{n+1} = (a p_rigid - b p^n) / (a + b)` with `a = C_end / k` and `b = 1 / (2R)`. Since
/// `a + b > 0` for any `R > 0` it is never singular — the load is unconditionally passive.
///
/// The booking happens **here**, per node, rather than once from the summed `U_out`, because the
/// original books inside its own `_radiate_node`. With a bell at both ends the two spellings
/// differ — `(E + e_l) + e_r` is not `E + (e_l + e_r)` in floating point — and `both ends` is a
/// configuration the bore's own tests build.
pub fn radiate_node(
    p_next: &mut [f64],
    p_old: f64,
    idx: usize,
    a: f64,
    b: f64,
    radiated_energy: &mut f64,
    params: &Params,
) -> f64 {
    let p_rigid = p_next[idx];
    let p_new = (a * p_rigid - b * p_old) / (a + b);
    p_next[idx] = p_new;
    let u_out = b * (p_new + p_old);
    *radiated_energy += params.k * params.r_bell * u_out * u_out;
    u_out
}

/// Drain every radiating end, booking the shed energy, and return the total outgoing `U_out`.
///
/// One formula serves either end — the outgoing direction differs but the dissipated
/// `k R U_out^2` does not — so a left-end sign error can only show up as energy drift, which is
/// what the both-ends test is for.
///
/// Returns `None` when `R_bell <= 0`, which is the original's early return: the caller must then
/// leave `U_out`/`U_out_prev` untouched. Note that `R_bell > 0` with *neither* end radiating is a
/// legal bore, and there this returns `Some(0.0)` — so the read-out pair is still rotated, exactly
/// as the original rotates it.
///
/// `p_old` is the *pre-step* pressure field `p^n`, which the caller still holds because the bore
/// has not committed the step yet.
pub fn apply_radiating_ends(
    p_next: &mut [f64],
    p_old: &[f64],
    radiated_energy: &mut f64,
    params: &Params,
) -> Option<f64> {
    if params.r_bell <= 0.0 {
        return None;
    }
    let b = 0.5 / params.r_bell;
    let mut u_out_total = 0.0;
    if params.rad_left() {
        u_out_total += radiate_node(
            p_next,
            p_old[0],
            0,
            params.a_left,
            b,
            radiated_energy,
            params,
        );
    }
    if params.rad_right() {
        u_out_total += radiate_node(
            p_next,
            p_old[params.n],
            params.n,
            params.a_right,
            b,
            radiated_energy,
            params,
        );
    }
    Some(u_out_total)
}

/// Energy **stored in the air column** (Joules), excluding what has already radiated away.
///
/// Compliance `p^2` plus the **cross-time** inductive `U^{n+1/2} U^{n-1/2}` term — never the
/// same-time square, the same "do not collapse the two time levels" trick as the string's
/// potential. Both reductions are `np.dot` on the Python side and go through BLAS, so this is the
/// one read-out here that cannot be bit-identical (plan §2.1).
///
/// # Panics
/// If any slice has the wrong length.
pub fn acoustic_energy(p: &[f64], u: &[f64], u_prev: &[f64], params: &Params) -> f64 {
    assert_eq!(p.len(), params.nodes(), "p must have N+1 entries");
    assert_eq!(u.len(), params.n, "u must have N entries");
    assert_eq!(u_prev.len(), params.n, "u_prev must have N entries");

    // `0.5 * np.dot(self._C, self.p * self.p)` — the elementwise square before the reduction, and
    // the accumulation strictly in index order (which is the part BLAS does not promise).
    let mut compliance = 0.0;
    for (&c, &pl) in params.c.iter().zip(p) {
        compliance += c * (pl * pl);
    }
    let mut inductive = 0.0;
    for ((&m, &uj), &up) in params.m.iter().zip(u).zip(u_prev) {
        inductive += m * (uj * up);
    }
    0.5 * compliance + 0.5 * inductive
}

// -- the native owning struct ----------------------------------------------------------------

/// An acoustic bore owning its own state.
///
/// The Rust caller's view and what `cargo test` exercises. The Python binding does **not** wrap
/// this — it holds NumPy arrays and calls the kernels above directly, because §9.3's buffer
/// contract requires Python to own anything a client can write to.
#[derive(Debug, Clone)]
pub struct Bore {
    p: Params,
    pressure: Vec<f64>,
    u: Vec<f64>,
    u_prev: Vec<f64>,
    radiated_energy: f64,
    u_out: f64,
    u_out_prev: f64,
    n: usize,
    /// Divergence workspace, hoisted out of the step.
    scratch: Vec<f64>,
}

impl Bore {
    /// Build from a validated parameter set, at rest.
    pub fn new(p: Params) -> Bore {
        let nodes = p.nodes();
        let n_seg = p.n;
        Bore {
            p,
            pressure: vec![0.0; nodes],
            u: vec![0.0; n_seg],
            u_prev: vec![0.0; n_seg],
            radiated_energy: 0.0,
            u_out: 0.0,
            u_out_prev: 0.0,
            n: 0,
            scratch: vec![0.0; nodes],
        }
    }

    /// The parameters.
    pub fn params(&self) -> &Params {
        &self.p
    }

    /// Current pressure field `p^n`.
    pub fn p(&self) -> &[f64] {
        &self.pressure
    }

    /// Current pressure field, mutable — a boundary exciter writes node 0 through this.
    pub fn p_mut(&mut self) -> &mut Vec<f64> {
        &mut self.pressure
    }

    /// Volume velocity `U^{n+1/2}`.
    pub fn u(&self) -> &[f64] {
        &self.u
    }

    /// Volume velocity `U^{n-1/2}`.
    pub fn u_prev(&self) -> &[f64] {
        &self.u_prev
    }

    /// Energy shed to the far field through radiating ends.
    pub fn radiated_energy(&self) -> f64 {
        self.radiated_energy
    }

    /// Outgoing volume velocity `U_out^{n+1/2}`.
    pub fn u_out(&self) -> f64 {
        self.u_out
    }

    /// Completed steps.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Set the initial pressure field (and optional half-node volume velocity).
    ///
    /// Starts from rest by default: `U^{1/2}` is taken as one consistent momentum half-step from
    /// `p^0`, so a single-mode pressure IC oscillates cleanly and the initial energy is exactly the
    /// acoustic potential. Also resets the far-field channel for a fresh run.
    ///
    /// # Panics
    /// If `p0` does not have `N + 1` entries or `u0` does not have `N`.
    pub fn set_state(&mut self, p0: &[f64], u0: &[f64]) {
        assert_eq!(p0.len(), self.p.nodes(), "p0 must have N+1 entries");
        assert_eq!(u0.len(), self.p.n, "u0 must have N entries");
        self.pressure.copy_from_slice(p0);
        apply_open_ends(&mut self.pressure, &self.p);
        self.u_prev.copy_from_slice(u0);
        let mut u_next = vec![0.0; self.p.n];
        momentum_into(&self.pressure, &self.u_prev, &mut u_next, &self.p);
        self.u = u_next;
        self.radiated_energy = 0.0;
        self.u_out = 0.0;
        self.u_out_prev = 0.0;
        self.n = 0;
    }

    /// Advance one timestep: pressure from the current velocity, then velocity from it.
    ///
    /// `source` is applied to the freshly-updated pressure field after the open-end pin and before
    /// the radiating drain — the ordering the module header calls load-bearing.
    pub fn step(&mut self, source: Option<Source<'_>>) {
        let mut p_next = vec![0.0; self.p.nodes()];
        pressure_into(
            &self.pressure,
            &self.u,
            &mut p_next,
            &mut self.scratch,
            &self.p,
        );
        if let Some(f) = source {
            f(&mut p_next);
        }
        if let Some(u_out) = apply_radiating_ends(
            &mut p_next,
            &self.pressure,
            &mut self.radiated_energy,
            &self.p,
        ) {
            self.u_out_prev = self.u_out;
            self.u_out = u_out;
        }
        let mut u_next = vec![0.0; self.p.n];
        momentum_into(&p_next, &self.u, &mut u_next, &self.p);

        std::mem::swap(&mut self.u_prev, &mut self.u);
        self.u = u_next;
        self.pressure = p_next;
        self.n += 1;
    }

    /// Energy stored in the air column (Joules), excluding what has radiated away.
    pub fn acoustic_energy(&self) -> f64 {
        acoustic_energy(&self.pressure, &self.u, &self.u_prev, &self.p)
    }

    /// Total conserved energy `E_bore + radiated_energy` (Joules).
    ///
    /// Conserved to machine precision when `sigma = 0` (any `R_bell`); monotonically decreasing if
    /// the air column is itself viscous. Assert conservation on this total, not on
    /// [`Self::acoustic_energy`] alone, which falls as the bell radiates.
    pub fn energy(&self) -> f64 {
        self.acoustic_energy() + self.radiated_energy
    }

    /// Pressure at node `index` — a microphone pickup.
    pub fn displacement_at(&self, index: usize) -> f64 {
        self.pressure[index]
    }

    /// Far-field monopole read-out: the bell's net volume acceleration `dU_out/dt` (m³/s²).
    ///
    /// Computed from the outgoing volume velocity, whose Nyquist part has cancelled, so it is
    /// clean even though the raw terminating node carries the cosmetic ripple. `0` when no end
    /// radiates.
    pub fn radiated_pressure(&self) -> f64 {
        (self.u_out - self.u_out_prev) / self.p.k
    }
}
