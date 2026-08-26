//! 2-D membrane (drumhead) resonator — explicit 5-point FDTD.
//!
//! Port of `physsynth/core/membrane.py`, HANDOFF §5 model #4:
//!
//! ```text
//! u_tt = c^2 (u_xx + u_yy) - 2 sigma u_t,     c^2 = T / rho
//! ```
//!
//! stepped explicitly as
//!
//! ```text
//! u^{n+1} = (2 u^n - (1 - sigma k) u^{n-1} + c^2 k^2 L u^n) / (1 + sigma k)
//! ```
//!
//! where `L` is the 5-point Laplacian restricted to the live nodes of a domain mask
//! (`ops2d::laplacian_from_mask`). **Two domains share this one type via the mask**: a rectangle,
//! whose `sin·sin` modes are exact and which is therefore the harness's unit test, and a circle —
//! the actual drumhead — whose round rim is *staircased* onto the Cartesian grid.
//!
//! # Energy is perpendicular to geometry
//!
//! The defining feature, as in 1-D, is [`energy`], with the **cross-time** potential
//!
//! ```text
//! E^n = rho [ 1/2 ||delta_t- u^n||^2  +  (c^2/2) P(u^n, u^{n-1}) ],   P(f, g) = <-L f, g> >= 0
//! ```
//!
//! evaluated through the *same* masked `L` the update uses. Because that `L` is symmetric —
//! a principal submatrix of the symmetric full-grid Laplacian — `E^{n+1} = E^n` is an exact
//! identity **for the staircased circle too**. The staircase costs accuracy against the Bessel
//! oracle (~O(h)); it costs the energy ledger nothing. That separation is the thing to remember
//! about this model: a green energy test says nothing about whether the rim is the right shape.
//!
//! # 2-D CFL
//!
//! `lambda = c k / h <= 1/sqrt(2)` — the 5-point Laplacian's spectral radius is `8/h^2`, double
//! the 1-D case. And unlike 1-D there is **no dispersionless lambda**: the 5-point scheme is
//! anisotropic at every Courant number, so tuning to the ceiling buys stability margin, not
//! exactness.
//!
//! # Operation order
//!
//! As in `string_ideal`, every expression reproduces NumPy's *evaluation order*, not merely its
//! algebra, so the elementwise kernels agree bit-for-bit rather than to a tolerance. The
//! reductions in [`energy`] go through `np.dot`/BLAS on the Python side and cannot; they are held
//! to the plan's Group A target instead.

use crate::ops2d::{self, Mask};
use crate::sparse::Csr;

/// Which of the two shapes a membrane is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Domain {
    /// Axis-aligned rectangle; `Lx` and `Ly` required. Exact `sin·sin` modes.
    Rectangle,
    /// Disk; `radius` required. The rim staircases onto the grid.
    Circle,
}

impl Domain {
    /// Parse the Python spelling. `None` for anything else — the caller owns the error message,
    /// because it is the one holding the object the user actually passed.
    pub fn parse(name: &str) -> Option<Domain> {
        match name {
            "rectangle" => Some(Domain::Rectangle),
            "circle" => Some(Domain::Circle),
            _ => None,
        }
    }
}

/// A construction-time rejection. Every variant's `Display` is the Python original's message
/// verbatim, because the suite matches on the text.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamError {
    /// One of `T`, `rho`, `fs` was not positive.
    NonPositive,
    /// Fewer than two grid segments.
    TooFewSegments,
    /// Negative loss coefficient.
    NegativeSigma,
    /// `domain="rectangle"` without both side lengths.
    RectangleNeedsSides,
    /// A side length was not positive.
    NonPositiveSides,
    /// `domain="circle"` without a radius.
    CircleNeedsRadius,
    /// The radius was not positive.
    NonPositiveRadius,
    /// The domain spec did not name a [`Domain`]. The caller formats the message, because it
    /// quotes the object that was passed.
    BadDomain,
    /// `lambda = c k / h > 1/sqrt(2)`. Carries the offending value.
    CflViolated(f64),
    /// The mask selected no unknowns — the grid is coarser than the shape.
    EmptyMask,
}

impl std::fmt::Display for ParamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamError::NonPositive => write!(f, "T, rho, fs must all be positive."),
            ParamError::TooFewSegments => write!(f, "N must be >= 2."),
            ParamError::NegativeSigma => write!(f, "sigma (loss) must be >= 0."),
            ParamError::RectangleNeedsSides => {
                write!(f, "domain='rectangle' requires Lx and Ly.")
            }
            ParamError::NonPositiveSides => write!(f, "Lx, Ly must be positive."),
            ParamError::CircleNeedsRadius => write!(f, "domain='circle' requires radius."),
            ParamError::NonPositiveRadius => write!(f, "radius must be positive."),
            ParamError::BadDomain => write!(f, "domain must be 'rectangle' or 'circle'."),
            ParamError::CflViolated(lam) => write!(
                f,
                "CFL violated: lambda = c*k/h = {lam:.6} > 1/sqrt(2) = {:.6}. \
                 Reduce fs, refine the grid (increase N), or lower the wave speed.",
                lambda_max()
            ),
            ParamError::EmptyMask => write!(
                f,
                "the domain mask has no interior (live) nodes; refine the grid."
            ),
        }
    }
}

impl std::error::Error for ParamError {}

/// The 2-D CFL ceiling, `1/sqrt(2)`.
///
/// Spelled as the division the original spells (`1.0 / np.sqrt(2.0)`) rather than as
/// `std::f64::consts::FRAC_1_SQRT_2`, because the two are not required to be the same double and
/// this value appears in a rejection boundary *and* in the message text.
pub fn lambda_max() -> f64 {
    1.0 / 2.0f64.sqrt()
}

/// A hair of slack so a requested `lambda == 1/sqrt(2)` is not spuriously rejected by round-off.
const LAMBDA_TOL: f64 = 1e-12;

/// Python's `round()` — round-half-to-**even** — which Rust's `f64::round` is not.
///
/// This matters at exactly one place, `Ny = max(int(round(Ly / h)), 1)`, and it matters a lot:
/// `Ly / h` lands on an exact half-integer for perfectly ordinary inputs (`Lx = 1`, `N = 2`,
/// `Ly = 1.25` gives `2.5`), and there Python snaps *down* to 2 cells while `f64::round` snaps up
/// to 3. That is a **different membrane** — different `Ly`, different mask, different spectrum —
/// and every detector this project owns stays green on it, because a 2-cell rectangle is as
/// energy-conserving as a 3-cell one. Same class of trap as the plan's `h ** 4` finding (§10.3):
/// a one-ulp spelling difference with a macroscopic consequence and no alarm attached.
fn round_ties_even(x: f64) -> f64 {
    x.round_ties_even()
}

/// The validated, immutable parameter set: everything about a membrane that is not its state.
///
/// Unlike the string's, this carries the assembled Laplacian. Building `L` is `O(n_live)` work
/// with an allocation, and both the update and `energy` need it every step, so it is derived once
/// here rather than rebuilt — the same reason the Python original stores it on the instance.
#[derive(Debug, Clone)]
pub struct Params {
    /// Which shape.
    pub domain: Domain,
    /// Tension per unit length (N/m).
    pub t: f64,
    /// Areal density (kg/m^2).
    pub rho: f64,
    /// Sample rate (Hz).
    pub fs: f64,
    /// Resolution control: segments along x (rectangle) or across the bounding box (circle).
    pub n: usize,
    /// Frequency-independent loss for the `-2 sigma u_t` term.
    pub sigma: f64,
    /// Wave speed `sqrt(T / rho)` (m/s).
    pub c: f64,
    /// Timestep `1 / fs` (s).
    pub k: f64,
    /// Grid spacing (m); cells are square.
    pub h: f64,
    /// Rectangle width (m); `None` for a circle.
    pub lx: Option<f64>,
    /// Rectangle height (m) **after snapping** to a whole number of cells; `None` for a circle.
    pub ly: Option<f64>,
    /// Disk radius (m); `None` for a rectangle.
    pub radius: Option<f64>,
    /// Courant number `c k / h`.
    pub lam: f64,
    /// Which nodes are unknowns.
    pub mask: Mask,
    /// Flat unknown index per node, `-1` at dead nodes; the mask's shape, row-major.
    pub index_map: Vec<i64>,
    /// The masked 5-point Laplacian, `(n_live x n_live)`.
    pub l: Csr,
    /// x-coordinate of every node, row-major over the mask's shape.
    pub x: Vec<f64>,
    /// y-coordinate of every node, row-major over the mask's shape.
    pub y: Vec<f64>,
}

impl Params {
    /// Validate and derive.
    ///
    /// `domain` is `None` when the caller could not make sense of the spec it was handed; passing
    /// it in that shape (rather than rejecting earlier) keeps the *order* of the checks identical
    /// to Python's, so a call with two faults reports the same one. `n` is taken as `i64` so that
    /// `N = 1` and `N = -3` are both rejected by the documented path rather than by a cast.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        domain: Option<Domain>,
        t: f64,
        rho: f64,
        fs: f64,
        n: i64,
        lx: Option<f64>,
        ly: Option<f64>,
        radius: Option<f64>,
        sigma: f64,
    ) -> Result<Params, ParamError> {
        // `min(T, rho, fs) <= 0` in the original. Deliberately no NaN guard: the Python side has
        // none either, and inventing one here would be a divergence in the one place a caller
        // could not have anticipated it.
        if t <= 0.0 || rho <= 0.0 || fs <= 0.0 {
            return Err(ParamError::NonPositive);
        }
        if n < 2 {
            return Err(ParamError::TooFewSegments);
        }
        if sigma < 0.0 {
            return Err(ParamError::NegativeSigma);
        }
        let n = n as usize;

        let c = (t / rho).sqrt();
        let k = 1.0 / fs;

        // Geometry. Each branch fixes `h`, the coordinate fields and the mask; nothing after this
        // point knows which shape it is holding, which is the point of the mask.
        let (h, x, y, mask, lx_out, ly_out, radius_out) = match domain {
            None => return Err(ParamError::BadDomain),
            Some(Domain::Rectangle) => {
                let (lx, ly) = match (lx, ly) {
                    (Some(a), Some(b)) => (a, b),
                    _ => return Err(ParamError::RectangleNeedsSides),
                };
                if lx <= 0.0 || ly <= 0.0 {
                    return Err(ParamError::NonPositiveSides);
                }
                let h = lx / (n as f64);
                // `Ny = max(int(round(Ly / h)), 1)`, then `Ly` is snapped to `Ny * h` so cells stay
                // square. See `round_ties_even` for why the rounding mode is load-bearing.
                let ny = (round_ties_even(ly / h) as i64).max(1) as usize;
                let ly_snapped = (ny as f64) * h;

                // `np.linspace(0, Lx, N+1)` and `np.linspace(0, Ly, Ny+1)`: `i * step + start`,
                // with the last entry overwritten by the endpoint. The y step is `Ly / Ny`, which
                // is NOT bit-identical to `h` in general even though `Ly` was just built from it.
                let xs = linspace_from_zero(lx, n);
                let ys = linspace_from_zero(ly_snapped, ny);
                let (fx, fy) = meshgrid(&xs, &ys);
                let mask = ops2d::rectangle_mask(n, ny);
                (h, fx, fy, mask, Some(lx), Some(ly_snapped), None)
            }
            Some(Domain::Circle) => {
                let radius = match radius {
                    Some(r) => r,
                    None => return Err(ParamError::CircleNeedsRadius),
                };
                if radius <= 0.0 {
                    return Err(ParamError::NonPositiveRadius);
                }
                let (fx, fy, h) = ops2d::grid_coords(n, radius);
                let mask = ops2d::disk_mask(&fx, &fy, radius, n + 1, n + 1);
                (h, fx, fy, mask, None, None, Some(radius))
            }
        };

        let lam = c * k / h;
        if lam > lambda_max() + LAMBDA_TOL {
            return Err(ParamError::CflViolated(lam));
        }

        let (l, index_map) = ops2d::laplacian_from_mask(&mask, h);
        if l.nrows() < 1 {
            return Err(ParamError::EmptyMask);
        }

        Ok(Params {
            domain: domain.expect("checked above"),
            t,
            rho,
            fs,
            n,
            sigma,
            c,
            k,
            h,
            lx: lx_out,
            ly: ly_out,
            radius: radius_out,
            lam,
            mask,
            index_map,
            l,
            x,
            y,
        })
    }

    /// Number of unknowns.
    pub fn n_live(&self) -> usize {
        self.l.nrows()
    }

    /// The mask's shape, `(nrows, ncols)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.mask.nrows(), self.mask.ncols())
    }

    /// Select the live-node values from a full row-major node field.
    ///
    /// # Panics
    /// If `field` is not the mask's shape.
    pub fn to_live(&self, field: &[f64]) -> Vec<f64> {
        assert_eq!(
            field.len(),
            self.mask.flags().len(),
            "field must have the mask's shape"
        );
        field
            .iter()
            .zip(self.mask.flags().iter())
            .filter_map(|(&v, &alive)| if alive { Some(v) } else { None })
            .collect()
    }

    /// Flat live-node index nearest the physical point `(x, y)` — for placing a pickup.
    ///
    /// Ties go to the lower index, as `np.argmin` does.
    pub fn pickup_index_at(&self, px: f64, py: f64) -> usize {
        let mut best = 0usize;
        let mut best_d2 = f64::INFINITY;
        let mut p = 0usize;
        for (idx, &alive) in self.index_map.iter().enumerate() {
            if alive < 0 {
                continue;
            }
            let dx = self.x[idx] - px;
            let dy = self.y[idx] - py;
            let d2 = dx * dx + dy * dy;
            if d2 < best_d2 {
                best_d2 = d2;
                best = p;
            }
            p += 1;
        }
        best
    }
}

/// `np.linspace(0.0, stop, n + 1)`, reproduced operation for operation.
///
/// NumPy forms `i * step` with `step = stop / n` and then overwrites the last entry with `stop`,
/// so `out[n]` is exactly `stop` rather than `n * (stop / n)`. Those differ in the last bit for
/// most lengths, and the coordinates reach the analysis layer.
fn linspace_from_zero(stop: f64, n: usize) -> Vec<f64> {
    let step = stop / (n as f64);
    let mut out: Vec<f64> = (0..=n).map(|i| (i as f64) * step).collect();
    out[n] = stop;
    out
}

/// `np.meshgrid(xs, ys)` — `x` varies along the second axis, `y` along the first. Row-major.
fn meshgrid(xs: &[f64], ys: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut fx = Vec::with_capacity(xs.len() * ys.len());
    let mut fy = Vec::with_capacity(xs.len() * ys.len());
    for &yv in ys {
        for &xv in xs {
            fx.push(xv);
            fy.push(yv);
        }
    }
    (fx, fy)
}

// -- kernels ---------------------------------------------------------------------------------
//
// Free functions over slices, as in `string_ideal`: they hold no state and allocate only what they
// return, so the native struct below can keep its buffers in `Vec`s while the Python binding keeps
// the same buffers in NumPy arrays. One copy of the physics, two owners.

/// One timestep: write `u^{n+1}` into `out` from `u^n` and `u^{n-1}`.
///
/// # Panics
/// If the three slices do not all have `n_live` entries.
pub fn step_into(u: &[f64], u_prev: &[f64], out: &mut [f64], p: &Params) {
    let n = p.n_live();
    assert_eq!(u.len(), n, "u must have n_live entries");
    assert_eq!(u_prev.len(), n, "u_prev must have n_live entries");
    assert_eq!(out.len(), n, "out must have n_live entries");

    let sk = p.sigma * p.k;
    let c2k2 = p.c * p.c * p.k * p.k;
    let one_minus_sk = 1.0 - sk;
    let one_plus_sk = 1.0 + sk;
    let lu = p.l.matvec(u);

    for i in 0..n {
        out[i] = ((2.0 * u[i] - one_minus_sk * u_prev[i]) + c2k2 * lu[i]) / one_plus_sk;
    }
}

/// The consistent second-order start: `u^{-1} = u^0 - k v^0 + 1/2 c^2 k^2 L u^0`.
///
/// This is what makes a single eigenmode oscillate as a clean discrete cosine and zero initial
/// velocity exact to second order. There is no boundary clamp to apply — a dead node is not in
/// the live vector at all.
///
/// # Panics
/// If `u0` or `v0` do not have `n_live` entries.
pub fn initial_previous(u0: &[f64], v0: &[f64], p: &Params) -> Vec<f64> {
    let n = p.n_live();
    assert_eq!(u0.len(), n, "u0 must have n_live entries");
    assert_eq!(v0.len(), n, "v0 must have n_live entries");

    let c2k2 = p.c * p.c * p.k * p.k;
    let half_c2k2 = 0.5 * c2k2;
    let lu = p.l.matvec(u0);
    (0..n)
        .map(|i| (u0[i] - p.k * v0[i]) + half_c2k2 * lu[i])
        .collect()
}

/// Discrete energy `E^n` (Joules) using the cross-time potential term.
///
/// Lossless -> conserved to machine precision; `sigma > 0` -> monotone decreasing (passive).
/// Both reductions here are `np.dot` on the Python side, so this is the one function in the model
/// that is not bit-identical to it.
///
/// # Panics
/// If `u` or `u_prev` do not have `n_live` entries.
pub fn energy(u: &[f64], u_prev: &[f64], p: &Params) -> f64 {
    let n = p.n_live();
    assert_eq!(u.len(), n, "u must have n_live entries");
    assert_eq!(u_prev.len(), n, "u_prev must have n_live entries");

    let h2 = p.h * p.h;

    // `0.5 * h2 * np.dot(dt_u, dt_u)`; the elementwise square happens before the reduction.
    let mut acc = 0.0;
    for i in 0..n {
        let dt_u = (u[i] - u_prev[i]) / p.k;
        acc += dt_u * dt_u;
    }
    let kinetic = (0.5 * h2) * acc;

    // `P(u^n, u^{n-1}) = <-L u^n, u^{n-1}> = -h^2 (L u^n) . u^{n-1}`, non-negative since -L is SPD.
    let lu = p.l.matvec(u);
    let mut dot = 0.0;
    for i in 0..n {
        dot += lu[i] * u_prev[i];
    }
    let p_np = -h2 * dot;
    let potential = ((0.5 * p.c) * p.c) * p_np;

    p.rho * (kinetic + potential)
}

// -- the native owning struct ----------------------------------------------------------------

/// A discretized membrane resonator, owning its own state.
///
/// The Rust caller's view and what `cargo test` exercises. The Python binding does **not** wrap
/// this: its buffers have to be Python objects so a reference held across a step stays valid (see
/// `physsynth-py`), so it owns NumPy arrays and calls the kernels above directly.
#[derive(Debug, Clone)]
pub struct Membrane {
    params: Params,
    /// Current displacement at the live nodes, `u^n`.
    pub u: Vec<f64>,
    /// Previous displacement at the live nodes, `u^{n-1}`.
    pub u_prev: Vec<f64>,
    /// Number of completed steps.
    pub n_steps: usize,
}

impl Membrane {
    /// Build from validated parameters. Both state buffers start at rest.
    pub fn new(params: Params) -> Membrane {
        let n = params.n_live();
        Membrane {
            params,
            u: vec![0.0; n],
            u_prev: vec![0.0; n],
            n_steps: 0,
        }
    }

    /// The parameter set this membrane was built from.
    pub fn params(&self) -> &Params {
        &self.params
    }

    /// Set the initial displacement and velocity at the live nodes, resetting the step count.
    ///
    /// # Panics
    /// If `u0` or `v0` do not have `n_live` entries.
    pub fn set_state(&mut self, u0: &[f64], v0: &[f64]) {
        self.u_prev = initial_previous(u0, v0, &self.params);
        self.u = u0.to_vec();
        self.n_steps = 0;
    }

    /// Set the initial displacement with zero initial velocity.
    pub fn set_displacement(&mut self, u0: &[f64]) {
        let v0 = vec![0.0; self.params.n_live()];
        self.set_state(u0, &v0);
    }

    /// Advance one timestep, rolling the history.
    pub fn step(&mut self) {
        let mut next = vec![0.0; self.params.n_live()];
        step_into(&self.u, &self.u_prev, &mut next, &self.params);
        std::mem::swap(&mut self.u_prev, &mut self.u);
        self.u = next;
        self.n_steps += 1;
    }

    /// Discrete energy `E^n` (Joules) — the primary bug detector.
    pub fn energy(&self) -> f64 {
        energy(&self.u, &self.u_prev, &self.params)
    }

    /// Current displacement as a full row-major node field, zeros at dead nodes.
    pub fn state(&self) -> Vec<f64> {
        ops2d::embed(&self.u, &self.params.index_map)
    }

    /// Displacement at flat live-node `index` — a pickup for spectral analysis.
    ///
    /// # Panics
    /// If `index` is past the last live node.
    pub fn displacement_at(&self, index: usize) -> f64 {
        self.u[index]
    }
}
