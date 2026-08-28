//! Kirchhoff plate — models #5, #5b, #5o, #5of, #5g and #6, the whole of `plate.py`.
//!
//! Two classes live here because the suite binds them into one: `tests/test_vk_energy.py` and
//! `tests/test_vk_free.py` each assert that a [`VkPlate`] with the coupling switched off is
//! **bit-identical** to a [`Plate`], `array_equal` on the state at every one of 150 steps and `==`
//! on the energy. A port that moved one and not the other would break that anchor for a reason
//! having nothing to do with either model — §15.2's finding, reaching two classes in one file
//! rather than four models across four.
//!
//! # What the port moves, and what it does not
//!
//! Everything up to the solve is bit-identical: the masks, the operators (ported in §25–§27), the
//! right-hand side and the energy's matvecs. The solve is not, and cannot be — `A` is factored by
//! [`SparseLu`] here and by SuperLU there, and §24.2 settled that those disagree at ~4e-16 per
//! entry for reasons that are a property of how SciPy was *built*. So this batch is the inverse of
//! the last one: it moves the last bits of **every plate trajectory under the flag** while leaving
//! the default path untouched, where §27 moved two matrix entries on two grids and nothing else.
//!
//! The exactness claim is therefore made the way §24.4 made the beam's: hold the solver constant —
//! drive the Python model through *this* factorization — and the two implementations agree to the
//! bit at every branch. That is the only test that can see a reassociation in the right-hand side,
//! and here there are four of them (supported/free, with and without an external force).
//!
//! # Three parity bars, because there are three ways this plate diverges
//!
//! * **supported, linear** — a linear model does not amplify, so the per-step solver gap random
//!   walks (§18.6).
//! * **free, linear** — the stiffness has the rigid-body nullspace `{1, x, y}`, along which the
//!   plate is a free particle and a per-step gap is integrated *twice*. §24.5 measured that on the
//!   beam's two-dimensional nullspace; here it is three-dimensional. Read the rigid/elastic split
//!   or the energy, never `max|du|/amp`.
//! * **von Karman** — either regime of §27.5. Read the **energy**.
//!
//! # Reductions
//!
//! `energy()` and `_P` are `np.dot`, and `pressure()` is `np.sum`. Neither is reproduced: the
//! first is §14.2's BLAS reduction and the second is NumPy's pairwise blocking, which `ops2d`'s
//! `guitar_area` and `collision`'s `barrier_energy` already declined for the same stated reason.
//! Both are read-outs that reach no timestep — `AirRadiation` consumes `pressure()` and feeds
//! nothing back, and the room-loaded tiers couple through the system matrix rather than through
//! the monopole. `portable.py` is declined a third time and on §24.6's grounds: exactness is not
//! *available* downstream of the solve, so buying it in the reduction would buy nothing.

use crate::fmt::{py_float, py_general};
use crate::ops2d::{
    biharmonic_from_mask, disk_mask, embed, free_plate_stiffness_from_mask, guitar_area,
    guitar_half_width, guitar_mask, guitar_scale, laplacian_from_mask, orthotropic_biharmonic,
    prune_to_area_carrying, rectangle_mask, AiryStressSolver, Mask, VonKarmanBracket,
};
use crate::pyfloat::scalar_pow;
use crate::radiation::py_round;
use crate::sparse::Csr;
use crate::sparse_lu::{SparseLu, SparseLuError};

/// A hair above `1/4` — near the minimal-dispersion weight, with a positivity margin.
pub const THETA_DEFAULT: f64 = 0.28;

/// Which edge condition the plate carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary {
    /// Simply-supported (Navier) — the closed-form-oracle case, `B = L * L`.
    Supported,
    /// Completely free — the curved-Chladni plate, assembled from the strain energy.
    Free,
}

/// The plate's outline, orthogonal to its boundary condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Domain {
    /// Every plate the core built before model #5g.
    Rectangle,
    /// A disk, staircased onto the Cartesian grid.
    Circle,
    /// The guitar outline of model #5g.
    Guitar,
}

impl Domain {
    /// The spelling the Python argument uses — the one refusals quote.
    pub fn name(self) -> &'static str {
        match self {
            Domain::Rectangle => "rectangle",
            Domain::Circle => "circle",
            Domain::Guitar => "guitar",
        }
    }
}

// -- the material helper -------------------------------------------------------------------

/// Everything [`Params`] needs from a material: `kappa`, the areal density, and five ratios.
///
/// `rho_s` is the **areal** density, and it is returned named for the reason the Python original
/// gives: the material density fed in is a *volume* density, and passing that straight through to
/// the plate leaves every frequency right and every energy wrong by a factor of the thickness.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrainSpec {
    /// `sqrt(D_x / rho_s)` (m^2/s).
    pub kappa: f64,
    /// Areal density `rho * thickness` (kg/m^2).
    pub rho_s: f64,
    /// `D_x / D_ref`, which is `1` by the choice of reference.
    pub grain_x: f64,
    /// `H / D_ref`, the supported branch's cross term.
    pub grain_cross: f64,
    /// `D_y / D_ref`.
    pub grain_y: f64,
    /// `D_1 / D_ref`, the coupling rigidity — the free branch's fourth constant.
    pub grain_coupling: f64,
    /// `D_xy / D_ref`, the torsional rigidity.
    pub grain_torsion: f64,
}

/// Why a material was refused.
#[derive(Debug, Clone, PartialEq)]
pub enum MaterialError {
    /// One of `E_x`, `E_y`, `G_xy`, `thickness`, `rho` was not positive.
    NonPositive,
    /// `1 - nu_xy nu_yx <= 0` — thermodynamically inadmissible.
    Inadmissible {
        /// The computed `1 - nu_xy nu_yx`.
        den: f64,
        /// The major Poisson ratio as given.
        nu_xy: f64,
        /// The minor Poisson ratio implied by reciprocity.
        nu_yx: f64,
    },
}

impl std::fmt::Display for MaterialError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MaterialError::NonPositive => {
                write!(f, "E_x, E_y, G_xy, thickness and rho must all be positive.")
            }
            MaterialError::Inadmissible { den, nu_xy, nu_yx } => write!(
                f,
                "1 - nu_xy*nu_yx must be positive (thermodynamic admissibility); got {} from \
                 nu_xy={}, nu_yx={}.",
                py_general(*den, 6),
                py_float(*nu_xy),
                py_general(*nu_yx, 6)
            ),
        }
    }
}

impl std::error::Error for MaterialError {}

/// Real orthotropic material to the ratios [`Params`] takes.
///
/// `thickness ** 3` goes through [`scalar_pow`] rather than a multiply: it is a Python `float`
/// raised to a power, which is the C library's `pow`, and §17.2's constant fold is exactly what
/// would turn a literal exponent into a chain of multiplies that rounds differently.
///
/// # Errors
/// A non-positive constant, or a Poisson pair outside thermodynamic admissibility.
pub fn grain_ratios_from_material(
    e_x: f64,
    e_y: f64,
    nu_xy: f64,
    g_xy: f64,
    thickness: f64,
    rho: f64,
) -> Result<GrainSpec, MaterialError> {
    if e_x.min(e_y).min(g_xy).min(thickness).min(rho) <= 0.0 {
        return Err(MaterialError::NonPositive);
    }
    let nu_yx = nu_xy * e_y / e_x;
    let den = 1.0 - nu_xy * nu_yx;
    if den <= 0.0 {
        return Err(MaterialError::Inadmissible { den, nu_xy, nu_yx });
    }
    let t3 = scalar_pow(thickness, 3.0);
    let d_x = e_x * t3 / (12.0 * den);
    let d_y = e_y * t3 / (12.0 * den);
    let d_1 = nu_yx * d_x;
    let d_xy = g_xy * t3 / 12.0;
    let h = d_1 + 2.0 * d_xy;
    let rho_s = rho * thickness;
    // `H / D_x` rather than a recombination of the two halves: that is the shipped #5o expression
    // and it is what lands *exactly* 1.0 for isotropic material.
    Ok(GrainSpec {
        kappa: (d_x / rho_s).sqrt(),
        rho_s,
        grain_x: 1.0,
        grain_cross: h / d_x,
        grain_y: d_y / d_x,
        grain_coupling: d_1 / d_x,
        grain_torsion: d_xy / d_x,
    })
}

// -- geometry --------------------------------------------------------------------------------

/// Number of 4-connected components of a live-node mask.
///
/// Hand-rolled here for the reason it is hand-rolled there: the Python original will not reach for
/// a new SciPy subpackage, and this crate depends on nothing at all. Only the *count* is used, so
/// the traversal order is free.
pub fn count_components(mask: &Mask) -> usize {
    let (nrows, ncols) = (mask.nrows(), mask.ncols());
    let live = mask.flags();
    let mut seen = vec![false; nrows * ncols];
    let mut n = 0usize;
    let mut stack: Vec<usize> = Vec::new();
    for start in 0..nrows * ncols {
        if !live[start] || seen[start] {
            continue;
        }
        n += 1;
        seen[start] = true;
        stack.push(start);
        while let Some(p) = stack.pop() {
            let (j, i) = (p / ncols, p % ncols);
            let mut neighbours = [usize::MAX; 4];
            if j + 1 < nrows {
                neighbours[0] = (j + 1) * ncols + i;
            }
            if j > 0 {
                neighbours[1] = (j - 1) * ncols + i;
            }
            if i + 1 < ncols {
                neighbours[2] = j * ncols + i + 1;
            }
            if i > 0 {
                neighbours[3] = j * ncols + i - 1;
            }
            for &q in &neighbours {
                if q != usize::MAX && live[q] && !seen[q] {
                    seen[q] = true;
                    stack.push(q);
                }
            }
        }
    }
    n
}

/// `np.linspace(0.0, stop, num)`, reproduced operation for operation.
///
/// NumPy forms `i * step` with `step = stop / (num - 1)` and then **overwrites** the last entry
/// with `stop`, so `xs[num-1]` is exactly `stop` rather than `(num-1) * step`. Those differ in the
/// last bit for most extents, and on a curved outline one ulp is a live node or a dead one (§25.3).
///
/// # Panics
/// If `num < 2` — the original never reaches the `div == 0` branch, and neither does this.
pub fn linspace0(stop: f64, num: usize) -> Vec<f64> {
    assert!(num >= 2, "linspace0 needs at least two points");
    let step = stop / ((num - 1) as f64);
    let mut v: Vec<f64> = (0..num).map(|i| (i as f64) * step).collect();
    v[num - 1] = stop;
    v
}

/// The `(X, Y)` node coordinate fields of `np.meshgrid(xs, ys)`, flattened row-major.
///
/// `X[j * ncols + i] == xs[i]` and `Y[j * ncols + i] == ys[j]`, which is the C-order layout every
/// live-node vector in the project is indexed by.
fn meshgrid(xs: &[f64], ys: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let (ncols, nrows) = (xs.len(), ys.len());
    let mut x = Vec::with_capacity(nrows * ncols);
    let mut y = Vec::with_capacity(nrows * ncols);
    for &yv in ys {
        for &xv in xs {
            x.push(xv);
            y.push(yv);
        }
    }
    (x, y)
}

// -- the linear plate's parameters -------------------------------------------------------------

/// Why a plate was refused at construction.
///
/// The variant order is the order the Python original checks them in, so a call that is wrong in
/// more than one way reports the same fault either way.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamError {
    /// `min(Lx, Ly, fs) <= 0`.
    NonPositive,
    /// `kappa <= 0`.
    NonPositiveKappa,
    /// `rho <= 0`.
    NonPositiveRho,
    /// `N < 2`.
    TooFewSegments,
    /// `sigma < 0`.
    NegativeSigma,
    /// `theta` outside `(0, 1]`.
    BadTheta(f64),
    /// The boundary argument was neither spelling.
    BadBoundary,
    /// Half of `(grain_coupling, grain_torsion)` was given.
    HalfSplit,
    /// `nu` alongside a split on the free branch.
    NuWithSplit,
    /// `nu` outside `(-1, 1/2)`.
    BadNu(f64),
    /// `grain_cross` contradicts `grain_coupling + 2 grain_torsion`.
    SplitContradiction {
        /// The `grain_cross` the caller passed.
        given: f64,
        /// `grain_coupling + 2 * grain_torsion`.
        effective: f64,
    },
    /// `grain_x <= 0` or `grain_y <= 0`.
    NonPositiveGrain(f64, f64),
    /// A grained free plate with no split at all.
    FreeNeedsSplit,
    /// `grain_cross <= -sqrt(grain_x grain_y)` — the supported operator is indefinite.
    IndefiniteCross {
        /// `-sqrt(grain_x * grain_y)`.
        floor: f64,
        /// The effective cross term.
        given: f64,
    },
    /// `|grain_coupling| >= sqrt(grain_x grain_y)` — the free-edge energy is indefinite.
    IndefiniteCoupling {
        /// `sqrt(grain_x * grain_y)`.
        ceiling: f64,
        /// The coupling rigidity.
        given: f64,
    },
    /// `grain_torsion <= 0` — a degenerate plate, not a stiff one.
    NonPositiveTorsion(f64),
    /// The domain argument was none of the three spellings.
    BadDomain,
    /// A curved outline on the supported branch — a refusal, not a limitation.
    CurvedSupported(Domain),
    /// The mask has no unknowns.
    NoLiveNodes,
    /// A node was pruned away from the rim: the mask is no longer the shape asked for.
    PrunedInside {
        /// How deep, in multiples of `h`.
        depth_in_h: f64,
        /// The outline that was asked for.
        domain: Domain,
        /// The grid the plate was built on.
        n: usize,
    },
    /// The outline staircased into more than one piece.
    Disconnected {
        /// How many pieces.
        parts: usize,
        /// The outline that was asked for.
        domain: Domain,
        /// The grid the plate was built on.
        n: usize,
        /// The waist that produced it.
        waist: f64,
    },
    /// `A` had no admissible pivot.
    NotFactorable(SparseLuError),
}

impl std::fmt::Display for ParamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamError::NonPositive => write!(f, "Lx, Ly, fs must all be positive."),
            ParamError::NonPositiveKappa => write!(f, "kappa (stiffness) must be positive."),
            ParamError::NonPositiveRho => write!(f, "rho (areal density) must be positive."),
            ParamError::TooFewSegments => {
                write!(f, "N must be >= 2 (need at least one interior node).")
            }
            ParamError::NegativeSigma => write!(f, "sigma (loss) must be >= 0."),
            ParamError::BadTheta(t) => {
                write!(f, "theta must be in (0, 1], got {}.", py_float(*t))
            }
            ParamError::BadBoundary => Ok(()),
            ParamError::HalfSplit => write!(
                f,
                "grain_coupling (D_1/D_ref) and grain_torsion (D_xy/D_ref) must be given together \
                 -- defaulting one half of the split from Poisson's ratio while the other is \
                 material data would silently mix two conventions."
            ),
            ParamError::NuWithSplit => write!(
                f,
                "pass either nu or (grain_coupling, grain_torsion), not both: on the free branch \
                 the split carries Poisson's ratio (D_1 = nu_yx D_x), so nu would be a silently \
                 ignored argument. The implied nu_yx = grain_coupling/grain_x is exposed as .nu."
            ),
            ParamError::BadNu(v) => write!(
                f,
                "nu (Poisson's ratio) must be in (-1, 1/2), got {}.",
                py_float(*v)
            ),
            ParamError::SplitContradiction { given, effective } => write!(
                f,
                "grain_cross={} contradicts the split (grain_coupling + 2*grain_torsion = {}). \
                 Pass one or the other; H = D_1 + 2 D_xy is not an independent number.",
                py_float(*given),
                py_general(*effective, 12)
            ),
            ParamError::NonPositiveGrain(gx, gy) => write!(
                f,
                "grain_x and grain_y (along/across bending stiffness ratios) must be positive, \
                 got ({}, {}).",
                py_float(*gx),
                py_float(*gy)
            ),
            ParamError::FreeNeedsSplit => write!(
                f,
                "the free branch needs the grain's coupling and torsional rigidities separately: \
                 pass grain_coupling (D_1/D_ref) and grain_torsion (D_xy/D_ref) alongside \
                 grain_x/grain_y. Their combination H = D_1 + 2 D_xy is enough for \
                 boundary='supported' but not for a free edge, whose corner force is pure torsion. \
                 grain_ratios_from_material returns both. See \
                 docs/dev/orthotropic-free-plate-plan.md."
            ),
            ParamError::IndefiniteCross { floor, given } => write!(
                f,
                "grain_cross must exceed -sqrt(grain_x*grain_y) = {} or the bending operator is \
                 indefinite (unstable); got {}.",
                py_general(*floor, 6),
                py_float(*given)
            ),
            ParamError::IndefiniteCoupling { ceiling, given } => write!(
                f,
                "|grain_coupling| must be < sqrt(grain_x*grain_y) = {} or the free-edge bending \
                 energy is indefinite (unstable); got {}.",
                py_general(*ceiling, 6),
                py_float(*given)
            ),
            ParamError::NonPositiveTorsion(v) => write!(
                f,
                "grain_torsion (D_xy/D_ref) must be positive; got {}. At zero the saddle xy \
                 carries no energy and joins the rigid-body nullspace as a 4th zero mode, which is \
                 a different (and lower) plate, not a stiff one.",
                py_float(*v)
            ),
            ParamError::BadDomain => Ok(()),
            ParamError::CurvedSupported(d) => write!(
                f,
                "domain='{}' is offered on boundary='free' only. A simply-supported plate on a \
                 curved outline has the membrane's spectrum squared (B = L @ L), so it is a model \
                 surface with no content -- use Membrane(domain=...) and square it.",
                d.name()
            ),
            ParamError::NoLiveNodes => {
                write!(
                    f,
                    "the plate has no interior (live) nodes; refine the grid."
                )
            }
            ParamError::PrunedInside {
                depth_in_h,
                domain,
                n,
            } => write!(
                f,
                "prune_to_area_carrying dropped a node {depth_in_h:.2} h inside the {} outline, \
                 not at its rim: the mask is no longer the shape that was asked for. Refine the \
                 grid (N = {n}) or soften the outline.",
                domain.name()
            ),
            ParamError::Disconnected {
                parts,
                domain,
                n,
                waist,
            } => write!(
                f,
                "the {} outline staircases into {parts} disconnected pieces at N = {n}: that is \
                 {parts} independent plates with a {}-dimensional rigid-body nullspace, not one \
                 plate. Refine the grid or reduce waist (= {}).",
                domain.name(),
                3 * parts,
                py_float(*waist)
            ),
            ParamError::NotFactorable(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ParamError {}

// -- the linear plate ---------------------------------------------------------------------------

/// The constructor arguments, in the shape the Python signature has them.
///
/// Every argument that is optional there is `Option` here, because several of the refusals turn on
/// *whether* a value was passed rather than on what it was: `nu` is an input in the isotropic case
/// and an output in the orthotropic one, and `grain_cross` alongside a split is a contradiction
/// rather than an override. §24.7 is the same lesson one layer up — an omitted argument and an
/// explicit `None` must not collapse.
#[derive(Debug, Clone, PartialEq)]
pub struct PlateSpec {
    /// Rectangle side length along `x` (m); the disk's diameter, the guitar's widest span.
    pub lx: f64,
    /// Side length along `y` (m), snapped to an integer number of square cells.
    pub ly: f64,
    /// Stiffness coefficient `sqrt(D / rho_s)` (m^2/s).
    pub kappa: f64,
    /// Areal density (kg/m^2).
    pub rho: f64,
    /// Sample rate (Hz).
    pub fs: f64,
    /// Segments along `x`; taken as `i64` so `N = -3` is rejected by the documented path.
    pub n: i64,
    /// Frequency-independent loss.
    pub sigma: f64,
    /// Time-averaging weight in `(0, 1]`.
    pub theta: f64,
    /// `None` when the caller could not make sense of the boundary it was handed.
    pub boundary: Option<Boundary>,
    /// `None` when the caller could not make sense of the domain it was handed.
    pub domain: Option<Domain>,
    /// Guitar waist depth.
    pub waist: f64,
    /// Guitar bout asymmetry.
    pub asym: f64,
    /// Poisson's ratio — free branch only, and refused alongside a split there.
    pub nu: Option<f64>,
    /// `D_x / D_ref`.
    pub grain_x: f64,
    /// `H / D_ref` — the supported branch's third number.
    pub grain_cross: Option<f64>,
    /// `D_y / D_ref`.
    pub grain_y: f64,
    /// `D_1 / D_ref` — half of the free branch's split.
    pub grain_coupling: Option<f64>,
    /// `D_xy / D_ref` — the other half.
    pub grain_torsion: Option<f64>,
}

impl Default for PlateSpec {
    /// The Python signature's defaults, for the arguments that have one.
    fn default() -> Self {
        PlateSpec {
            lx: 0.0,
            ly: 0.0,
            kappa: 0.0,
            rho: 0.0,
            fs: 0.0,
            n: 0,
            sigma: 0.0,
            theta: THETA_DEFAULT,
            boundary: Some(Boundary::Supported),
            domain: Some(Domain::Rectangle),
            waist: 0.42,
            asym: 0.30,
            nu: None,
            grain_x: 1.0,
            grain_cross: None,
            grain_y: 1.0,
            grain_coupling: None,
            grain_torsion: None,
        }
    }
}

/// The validated parameter set plus everything that is constant in time.
///
/// The operators and the factorization are built **once**, which for the plate is not the §11.4
/// precaution it was for the membrane: a guitar plate's `splu` is the most expensive thing this
/// model does, and rebuilding it per access would pass every physics bar while making the flagged
/// run slower than the Python one.
#[derive(Debug, Clone)]
pub struct Params {
    /// Side length along `x` (m).
    pub lx: f64,
    /// Side length along `y` (m) — the **snapped** value, `Ny * h`.
    pub ly: f64,
    /// Stiffness coefficient (m^2/s).
    pub kappa: f64,
    /// Areal density (kg/m^2).
    pub rho: f64,
    /// Sample rate (Hz).
    pub fs: f64,
    /// Segments along `x`.
    pub n: usize,
    /// Segments along `y`, from `int(round(Ly / h))`.
    pub ny: usize,
    /// Frequency-independent loss.
    pub sigma: f64,
    /// Time-averaging weight.
    pub theta: f64,
    /// Which edge condition.
    pub boundary: Boundary,
    /// Which outline.
    pub domain: Domain,
    /// Guitar waist depth.
    pub waist: f64,
    /// Guitar bout asymmetry.
    pub asym: f64,
    /// Poisson's ratio — an input isotropically, `grain_coupling / grain_x` with a split.
    pub nu: f64,
    /// `D_x / D_ref`.
    pub grain_x: f64,
    /// `H / D_ref`, resolved from the split when one was given.
    pub grain_cross: f64,
    /// `D_y / D_ref`.
    pub grain_y: f64,
    /// `D_1 / D_ref`.
    pub grain_coupling: f64,
    /// `D_xy / D_ref`.
    pub grain_torsion: f64,
    /// Selects the untouched `B = L @ L` line — a flag about the *assembly*, not the material.
    pub grain_is_isotropic: bool,
    /// Timestep `1 / fs` (s).
    pub k: f64,
    /// Square cell spacing `Lx / N` (m).
    pub h: f64,
    /// Plate "Courant" number `kappa k / h^2` — reported only.
    pub mu: f64,
    /// Node `x` coordinates, row-major over `(ny+1) x (n+1)`.
    pub x: Vec<f64>,
    /// Node `y` coordinates, same layout.
    pub y: Vec<f64>,
    /// Which nodes are unknowns.
    pub mask: Mask,
    /// Flat unknown index per node, `-1` at dead nodes.
    pub index_map: Vec<i64>,
    /// The masked Dirichlet Laplacian — supported branch only.
    pub laplacian: Option<Csr>,
    /// The operator the step applies: `B` on the supported branch, `K` on the free one.
    pub stiffness: Csr,
    /// The lumped mass `W` — free branch only.
    pub mass: Option<Csr>,
    /// `W`'s diagonal; empty on the supported branch, where the mass is the scalar `h^2`.
    pub w: Vec<f64>,
    /// Number of unknowns.
    pub n_live: usize,
    /// How many staircase spikes `prune_to_area_carrying` removed.
    pub n_pruned: usize,
    /// How deep inside the outline the deepest pruned node sat (m); `0.0` when nothing was pruned.
    pub prune_depth_max: f64,
    /// Area of the continuum outline (m^2).
    pub outline_area: f64,
    /// Area the quadrature weights actually carry (m^2).
    pub area: f64,
    /// `area / outline_area - 1` — reported, never applied.
    pub area_deficit: f64,
    /// Factorization of `A`.
    pub lu: SparseLu,
}

impl Params {
    /// Validate, derive, assemble the operators and factor `A`.
    ///
    /// The checks run in the Python original's order, so a call that is wrong in more than one way
    /// reports the same fault either way.
    ///
    /// # Errors
    /// Any of [`ParamError`].
    pub fn new(spec: &PlateSpec) -> Result<Params, ParamError> {
        if spec.lx.min(spec.ly).min(spec.fs) <= 0.0 {
            return Err(ParamError::NonPositive);
        }
        if spec.kappa <= 0.0 {
            return Err(ParamError::NonPositiveKappa);
        }
        if spec.rho <= 0.0 {
            return Err(ParamError::NonPositiveRho);
        }
        if spec.n < 2 {
            return Err(ParamError::TooFewSegments);
        }
        if spec.sigma < 0.0 {
            return Err(ParamError::NegativeSigma);
        }
        if !(spec.theta > 0.0 && spec.theta <= 1.0) {
            return Err(ParamError::BadTheta(spec.theta));
        }
        let Some(boundary) = spec.boundary else {
            return Err(ParamError::BadBoundary);
        };

        // -- resolve the grain: three numbers (supported) or four (free) --------------------
        let split_given = spec.grain_coupling.is_some() || spec.grain_torsion.is_some();
        if split_given && (spec.grain_coupling.is_none() || spec.grain_torsion.is_none()) {
            return Err(ParamError::HalfSplit);
        }
        if split_given && spec.nu.is_some() && boundary == Boundary::Free {
            return Err(ParamError::NuWithSplit);
        }
        if let Some(nu) = spec.nu {
            if !(-1.0 < nu && nu < 0.5) {
                return Err(ParamError::BadNu(nu));
            }
        }

        let (grain_x, grain_y) = (spec.grain_x, spec.grain_y);
        let (g_1, g_xy, cross_eff);
        if split_given {
            g_1 = spec.grain_coupling.expect("both halves present");
            g_xy = spec.grain_torsion.expect("both halves present");
            cross_eff = g_1 + 2.0 * g_xy;
            if let Some(given) = spec.grain_cross {
                if (given - cross_eff).abs() > 1e-12 * 1.0f64.max(cross_eff.abs()) {
                    return Err(ParamError::SplitContradiction {
                        given,
                        effective: cross_eff,
                    });
                }
            }
        } else {
            cross_eff = spec.grain_cross.unwrap_or(1.0);
            // The nu-derived isotropic split. Bit-identical to the shipped isotropic assembly.
            let nu_for_split = spec.nu.unwrap_or(0.3);
            g_1 = nu_for_split;
            g_xy = 0.5 * (1.0 - nu_for_split);
        }
        let grain_cross = cross_eff;
        let grain_is_isotropic = grain_x == 1.0 && grain_cross == 1.0 && grain_y == 1.0;
        if grain_x <= 0.0 || grain_y <= 0.0 {
            return Err(ParamError::NonPositiveGrain(grain_x, grain_y));
        }
        if boundary == Boundary::Free && !split_given && !(grain_x == 1.0 && grain_y == 1.0) {
            return Err(ParamError::FreeNeedsSplit);
        }
        if boundary == Boundary::Supported {
            let cross_floor = -(grain_x * grain_y).sqrt();
            if grain_cross <= cross_floor {
                return Err(ParamError::IndefiniteCross {
                    floor: cross_floor,
                    given: grain_cross,
                });
            }
        } else {
            let coupling_ceiling = (grain_x * grain_y).sqrt();
            if g_1.abs() >= coupling_ceiling {
                return Err(ParamError::IndefiniteCoupling {
                    ceiling: coupling_ceiling,
                    given: g_1,
                });
            }
            if g_xy <= 0.0 {
                return Err(ParamError::NonPositiveTorsion(g_xy));
            }
        }

        let nu = match spec.nu {
            Some(v) => v,
            None if split_given => g_1 / grain_x,
            None => 0.3,
        };
        let Some(domain) = spec.domain else {
            return Err(ParamError::BadDomain);
        };
        if domain != Domain::Rectangle && boundary != Boundary::Free {
            return Err(ParamError::CurvedSupported(domain));
        }

        let n = spec.n as usize;
        let k = 1.0 / spec.fs;
        // A circle takes Lx as its DIAMETER and squares the bounding box before snapping.
        let ly_in = if domain == Domain::Circle {
            spec.lx
        } else {
            spec.ly
        };
        let h = spec.lx / (n as f64);
        let ny = (py_round(ly_in / h) as i64).max(1) as usize;
        let lx = spec.lx;
        let ly = (ny as f64) * h; // snapped so cells are square
        let xs = linspace0(lx, n + 1);
        let ys = linspace0(ly, ny + 1);
        let (x, y) = meshgrid(&xs, &ys);
        let mu = spec.kappa * k / (h * h);

        let sk = spec.sigma * k;
        let coeff = spec.theta * k * k * spec.kappa * spec.kappa;

        // Outline bookkeeping, defined for every plate so nothing downstream branches on the
        // domain to read it. A rectangle prunes nothing and its area deficit comes out at
        // rounding rather than being forced to zero.
        let mut n_pruned = 0usize;
        let mut prune_depth_max = 0.0f64;
        let mut outline_area = lx * ly;
        let mut area = outline_area;
        let mut area_deficit = 0.0f64;

        let (mask, index_map, laplacian, stiffness, mass, w);
        if boundary == Boundary::Supported {
            let m = rectangle_mask(n, ny);
            let (l, im) = laplacian_from_mask(&m, h);
            let b = if grain_is_isotropic {
                // Deliberately the shared builder rather than a second spelling of `L @ L`: both
                // are multiplied by every timestep and both need the canonical column order.
                biharmonic_from_mask(&m, h).0
            } else {
                // Grain: a *separate* path -- routing the isotropic default through it would agree
                // only to ~2e-16 and perturb every shipped plate number in the last digit.
                orthotropic_biharmonic(n, ny, h, grain_x, grain_cross, grain_y).0
            };
            if b.nrows() < 1 {
                return Err(ParamError::NoLiveNodes);
            }
            mask = m;
            index_map = im;
            laplacian = Some(l);
            stiffness = b;
            mass = None;
            w = Vec::new();
        } else {
            let raw = outline_mask(domain, &x, &y, lx, ly, n, ny, spec.waist, spec.asym);
            // THE MASK IS NOT THE OUTLINE. A curved rim staircases into one-node spikes touching
            // no complete cell, whose trapezoidal weight is exactly 0 -- and two of those make the
            // factorisation fail outright.
            let (pruned, dropped) = prune_to_area_carrying(&raw);
            n_pruned = dropped;
            prune_depth_max = check_pruned_nodes_are_at_the_rim(
                domain, &raw, &pruned, &x, &y, lx, ly, h, n, spec.waist, spec.asym,
            )?;
            // The four constants are always passed explicitly -- ONE code path for isotropic and
            // orthotropic; at the nu-derived split the coefficients are byte-identical to the
            // pre-grain assembly.
            let (kk, ww, im) = free_plate_stiffness_from_mask(
                &pruned,
                h,
                nu,
                grain_x,
                grain_y,
                Some(g_1),
                Some(g_xy),
            );
            let wdiag: Vec<f64> = (0..kk.nrows()).map(|i| ww.get(i, i)).collect();
            // `W.diagonal().sum()` is `ndarray.sum()`, i.e. NumPy's pairwise blocking; summed left
            // to right here for the reason `ops2d::guitar_area` gives. It is divided into the
            // outline area to report how converged a staircase is and reaches no timestep.
            area = wdiag.iter().sum();
            outline_area = true_outline_area(domain, lx, ly, spec.waist, spec.asym);
            area_deficit = area / outline_area - 1.0;
            mask = pruned;
            index_map = im;
            laplacian = None;
            stiffness = kk;
            mass = Some(ww);
            w = wdiag;
        }
        let n_live = stiffness.nrows();

        // A = (1 + sigma k) I + theta k^2 kappa^2 B, or its W form. SPD either way.
        let a = match &mass {
            Some(ww) => ww.scaled(1.0 + sk).add(&stiffness.scaled(coeff)),
            None => Csr::identity(n_live)
                .scaled(1.0 + sk)
                .add(&stiffness.scaled(coeff)),
        };
        let lu = SparseLu::factor(&a).map_err(ParamError::NotFactorable)?;

        Ok(Params {
            lx,
            ly,
            kappa: spec.kappa,
            rho: spec.rho,
            fs: spec.fs,
            n,
            ny,
            sigma: spec.sigma,
            theta: spec.theta,
            boundary,
            domain,
            waist: spec.waist,
            asym: spec.asym,
            nu,
            grain_x,
            grain_cross,
            grain_y,
            grain_coupling: g_1,
            grain_torsion: g_xy,
            grain_is_isotropic,
            k,
            h,
            mu,
            x,
            y,
            mask,
            index_map,
            laplacian,
            stiffness,
            mass,
            w,
            n_live,
            n_pruned,
            prune_depth_max,
            outline_area,
            area,
            area_deficit,
            lu,
        })
    }
}

/// Raw live-node mask for `domain`, before pruning.
///
/// A rectangle is the mask that happens to be all-ones — the case the free plate always was, now
/// stated as one outline among three rather than as the absence of an outline.
#[allow(clippy::too_many_arguments)]
fn outline_mask(
    domain: Domain,
    x: &[f64],
    y: &[f64],
    lx: f64,
    ly: f64,
    n: usize,
    ny: usize,
    waist: f64,
    asym: f64,
) -> Mask {
    let (nrows, ncols) = (ny + 1, n + 1);
    match domain {
        Domain::Rectangle => Mask::new(nrows, ncols, vec![true; nrows * ncols]),
        Domain::Circle => {
            let r = 0.5 * lx;
            let xc: Vec<f64> = x.iter().map(|v| v - r).collect();
            let yc: Vec<f64> = y.iter().map(|v| v - r).collect();
            disk_mask(&xc, &yc, r, nrows, ncols)
        }
        Domain::Guitar => {
            let xc: Vec<f64> = x.iter().map(|v| v - 0.5 * lx).collect();
            guitar_mask(&xc, y, ly, lx, waist, asym, nrows, ncols)
        }
    }
}

/// Area of the continuum outline — the denominator of the area deficit.
fn true_outline_area(domain: Domain, lx: f64, ly: f64, waist: f64, asym: f64) -> f64 {
    match domain {
        Domain::Rectangle => lx * ly,
        Domain::Circle => std::f64::consts::PI * scalar_pow(0.5 * lx, 2.0),
        Domain::Guitar => guitar_area(ly, lx, waist, asym),
    }
}

/// Distance from `(x, y)` to the outline boundary, measured *inwards*.
fn depth_inside_outline(
    domain: Domain,
    x: f64,
    y: f64,
    lx: f64,
    ly: f64,
    waist: f64,
    asym: f64,
) -> f64 {
    if domain == Domain::Circle {
        let r = 0.5 * lx;
        return r - (x - r).hypot(y - r);
    }
    let half = guitar_scale(lx, waist, asym) * guitar_half_width(y / ly, waist, asym);
    let across = half - (x - 0.5 * lx).abs();
    let along = y.min(ly - y);
    across.min(along)
}

/// Assert the pruned mask is still the plate that was asked for, and report how deep it cut.
///
/// Two failure modes, neither visible to any other detector this family has — energy conserves,
/// the nullspace stays 3-dimensional and the spectrum stays plausible through both. The depth is
/// *returned* rather than only compared, because a bar that raises only on violation is never
/// observed on a grid that passes.
#[allow(clippy::too_many_arguments)]
fn check_pruned_nodes_are_at_the_rim(
    domain: Domain,
    raw: &Mask,
    pruned: &Mask,
    x: &[f64],
    y: &[f64],
    lx: f64,
    ly: f64,
    h: f64,
    n: usize,
    waist: f64,
    asym: f64,
) -> Result<f64, ParamError> {
    if domain == Domain::Rectangle {
        return Ok(0.0);
    }
    let mut depth_max = 0.0f64;
    let mut any = false;
    for (idx, (&was, &is)) in raw.flags().iter().zip(pruned.flags().iter()).enumerate() {
        if was && !is {
            let d = depth_inside_outline(domain, x[idx], y[idx], lx, ly, waist, asym);
            depth_max = if any { depth_max.max(d) } else { d };
            any = true;
        }
    }
    if any && depth_max > 1.0001 * h {
        return Err(ParamError::PrunedInside {
            depth_in_h: depth_max / h,
            domain,
            n,
        });
    }
    let parts = count_components(pruned);
    if parts != 1 {
        return Err(ParamError::Disconnected {
            parts,
            domain,
            n,
            waist,
        });
    }
    Ok(if any { depth_max } else { 0.0 })
}

// -- the scheme ---------------------------------------------------------------------------------

/// `a . b`, left to right.
///
/// Stands in for `np.dot`, which is BLAS `ddot` and is not reproducible by any portable loop
/// (§14.2). Every caller here is a read-out — see the module header for why that is enough.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        s += x * y;
    }
    s
}

/// `u^{-1} = u^0 - k v^0 + 1/2 k^2 a^0` — the consistent second-order start.
///
/// The bending acceleration is `-kappa^2 B u^0` on the supported branch and `-kappa^2 W^-1 K u^0`
/// on the free one. Spelled as the original spells it: one scalar `0.5 k^2 kappa^2` formed left to
/// right and *subtracted*, which is a different rounding from the beam's `+ half_k2 * accel`.
pub fn initial_previous(u0: &[f64], v0: &[f64], p: &Params) -> Vec<f64> {
    let su = p.stiffness.matvec(u0);
    let half_k2_kappa2 = 0.5 * p.k * p.k * p.kappa * p.kappa;
    (0..p.n_live)
        .map(|i| {
            let accel_term = match p.boundary {
                Boundary::Supported => half_k2_kappa2 * su[i],
                Boundary::Free => half_k2_kappa2 * su[i] / p.w[i],
            };
            u0[i] - p.k * v0[i] - accel_term
        })
        .collect()
}

/// `a^0 = -kappa^2 B u^0` (supported) or `-kappa^2 W^-1 K u^0` (free).
///
/// Seeded so `pressure()` is meaningful before the first step.
pub fn initial_accel(u0: &[f64], p: &Params) -> Vec<f64> {
    let su = p.stiffness.matvec(u0);
    let neg_kappa2 = -p.kappa * p.kappa;
    (0..p.n_live)
        .map(|i| match p.boundary {
            Boundary::Supported => neg_kappa2 * su[i],
            Boundary::Free => neg_kappa2 * su[i] / p.w[i],
        })
        .collect()
}

/// The step's right-hand side, in NumPy's summation order.
///
/// Four spellings live here — supported and free, each with and without an external nodal force —
/// and they are the four the exactness claim of §24.4's manoeuvre is made about. The `f_ext` term
/// is added *before* the solve because a post-solve correction is invalid: `A` couples all nodes.
pub fn step_rhs(u: &[f64], u_prev: &[f64], f_ext: Option<&[f64]>, p: &Params) -> Vec<f64> {
    let sk = p.sigma * p.k;
    let k2 = p.k * p.k;
    let kappa2 = p.kappa * p.kappa;
    let su = p.stiffness.matvec(u);
    let su_prev = p.stiffness.matvec(u_prev);
    // `(1 - 2 theta) * k2` and `theta * k2` are scalar-times-scalar before the array multiply,
    // because `float * float * ndarray` associates to the left.
    let c_now = (1.0 - 2.0 * p.theta) * k2;
    let c_prev = p.theta * k2;
    let force_den = match p.boundary {
        Boundary::Supported => p.rho * p.h * p.h,
        Boundary::Free => p.rho,
    };
    (0..p.n_live)
        .map(|i| {
            let lop_u = -kappa2 * su[i];
            let lop_prev = -kappa2 * su_prev[i];
            let base = match p.boundary {
                Boundary::Supported => {
                    2.0 * u[i] + c_now * lop_u - u_prev[i] + c_prev * lop_prev + sk * u_prev[i]
                }
                Boundary::Free => {
                    p.w[i] * (2.0 * u[i] - u_prev[i])
                        + c_now * lop_u
                        + c_prev * lop_prev
                        + sk * (p.w[i] * u_prev[i])
                }
            };
            match f_ext {
                Some(f) => base + k2 * f[i] / force_den,
                None => base,
            }
        })
        .collect()
}

/// Potential bilinear form `P(f, g) = <-L f, g> >= 0`.
///
/// `kappa^2 h^2 (B f) . g` supported, `kappa^2 (K f) . g` free — each through the *same* matrix as
/// the update, which is what makes the energy identity exact rather than approximate.
pub fn potential_form(f: &[f64], g: &[f64], p: &Params) -> f64 {
    let sf = p.stiffness.matvec(f);
    match p.boundary {
        Boundary::Supported => p.kappa * p.kappa * p.h * p.h * dot(&sf, g),
        Boundary::Free => p.kappa * p.kappa * dot(&sf, g),
    }
}

/// Kinetic plus bending energy `E^n` (Joules) for the implicit theta-scheme.
pub fn energy(u: &[f64], u_prev: &[f64], p: &Params) -> f64 {
    let dt_u: Vec<f64> = (0..p.n_live).map(|i| (u[i] - u_prev[i]) / p.k).collect();
    let kinetic = match p.boundary {
        Boundary::Supported => 0.5 * (p.h * p.h) * dot(&dt_u, &dt_u),
        Boundary::Free => {
            let weighted: Vec<f64> = (0..p.n_live).map(|i| p.w[i] * dt_u[i]).collect();
            0.5 * dot(&dt_u, &weighted)
        }
    };
    let p_nn = potential_form(u, u, p);
    let p_pp = potential_form(u_prev, u_prev, p);
    let p_np = potential_form(u, u_prev, p);
    let potential = 0.5 * p.theta * (p_nn + p_pp) + (0.5 - p.theta) * p_np;
    p.rho * (kinetic + potential)
}

/// Radiated pressure read-out `p = sum_i area_i u_i''` — a monopole, proportional to volume
/// acceleration.
///
/// The supported branch's `np.sum` is NumPy's *pairwise* blocking above eight elements and is
/// summed left to right here, for the reason `ops2d::guitar_area` gives. Nothing feeds this back.
pub fn pressure(accel: &[f64], p: &Params) -> f64 {
    match p.boundary {
        Boundary::Supported => {
            let mut s = 0.0;
            for &a in accel {
                s += a;
            }
            p.h * p.h * s
        }
        Boundary::Free => dot(&p.w, accel),
    }
}

/// Flat live-node index nearest the physical point `(x, y)`.
///
/// A **discrete** output (§25.2), so the squares are spelled `d * d`: the original squares a NumPy
/// *array*, where `** 2` is the ufunc ladder's `x * x` and not a `pow` call (§16.2). Ties go to the
/// first minimum, as `np.argmin` does.
pub fn pickup_index_at(x: f64, y: f64, p: &Params) -> usize {
    let mut best = 0usize;
    let mut best_d2 = f64::INFINITY;
    let mut live = 0usize;
    for (idx, &alive) in p.mask.flags().iter().enumerate() {
        if !alive {
            continue;
        }
        let dx = p.x[idx] - x;
        let dy = p.y[idx] - y;
        let d2 = dx * dx + dy * dy;
        if d2 < best_d2 {
            best_d2 = d2;
            best = live;
        }
        live += 1;
    }
    best
}

/// A discretized Kirchhoff plate — parameters plus the four state buffers.
#[derive(Debug, Clone)]
pub struct Plate {
    /// Everything constant in time.
    pub p: Params,
    /// Current displacement `u^n` over the live nodes.
    pub u: Vec<f64>,
    /// Previous displacement `u^{n-1}`.
    pub u_prev: Vec<f64>,
    /// Transverse acceleration of the most recent step, from the *actual* second difference.
    pub accel: Vec<f64>,
    /// Completed steps.
    pub n: usize,
}

impl Plate {
    /// A plate at rest.
    pub fn new(p: Params) -> Self {
        let n_live = p.n_live;
        Plate {
            p,
            u: vec![0.0; n_live],
            u_prev: vec![0.0; n_live],
            accel: vec![0.0; n_live],
            n: 0,
        }
    }

    /// Set the initial displacement and velocity over the live nodes.
    pub fn set_state(&mut self, u0: &[f64], v0: &[f64]) {
        self.u_prev = initial_previous(u0, v0, &self.p);
        self.accel = initial_accel(u0, &self.p);
        self.u = u0.to_vec();
        self.n = 0;
    }

    /// Advance one timestep, rolling the history.
    ///
    /// # Panics
    /// If the factorization cannot back-substitute — which for an SPD `A` means the caller built
    /// something that is not one.
    pub fn step(&mut self, f_ext: Option<&[f64]>) {
        let rhs = step_rhs(&self.u, &self.u_prev, f_ext, &self.p);
        let next = self.p.lu.solve(&rhs).expect("A is SPD and was factored");
        let k2 = self.p.k * self.p.k;
        self.accel = (0..self.p.n_live)
            .map(|i| (next[i] - 2.0 * self.u[i] + self.u_prev[i]) / k2)
            .collect();
        self.u_prev = std::mem::replace(&mut self.u, next);
        self.n += 1;
    }

    /// Current displacement as a full-grid field, dead nodes zero.
    pub fn state(&self) -> Vec<f64> {
        embed(&self.u, &self.p.index_map)
    }

    /// Discrete energy `E^n` (Joules).
    pub fn energy(&self) -> f64 {
        energy(&self.u, &self.u_prev, &self.p)
    }

    /// Radiated pressure read-out.
    pub fn pressure(&self) -> f64 {
        pressure(&self.accel, &self.p)
    }
}

// -- the von Karman plate -------------------------------------------------------------------

/// Why a nonlinear plate was refused at construction.
///
/// Variant order is the Python original's check order. The three that a linear plate also refuses
/// are delegated: [`Params::new`] raises them with the same text.
#[derive(Debug, Clone, PartialEq)]
pub enum VkParamError {
    /// `min(Lx, Ly, fs) <= 0`.
    NonPositive,
    /// `E <= 0`.
    NonPositiveYoung,
    /// `e <= 0`.
    NonPositiveThickness,
    /// `rho <= 0`.
    NonPositiveDensity,
    /// `N < 2`.
    TooFewSegments,
    /// `sigma < 0`.
    NegativeSigma,
    /// `theta` outside `(0, 1]`.
    BadTheta(f64),
    /// `nu` outside `(-1, 1/2)`.
    BadNu(f64),
    /// The boundary argument was neither spelling.
    BadBoundary,
    /// `couple_tol <= 0`.
    NonPositiveTol,
    /// `couple_max_iter < 1`.
    TooFewSweeps,
    /// Whatever the linear half refused — the mask, or the factorization.
    Linear(ParamError),
    /// `B_F` had no admissible pivot.
    AiryNotFactorable(SparseLuError),
}

impl std::fmt::Display for VkParamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VkParamError::NonPositive => write!(f, "Lx, Ly, fs must all be positive."),
            VkParamError::NonPositiveYoung => write!(f, "E (Young's modulus) must be positive."),
            VkParamError::NonPositiveThickness => write!(f, "e (thickness) must be positive."),
            VkParamError::NonPositiveDensity => write!(f, "rho (density) must be positive."),
            VkParamError::TooFewSegments => {
                write!(f, "N must be >= 2 (need at least one interior node).")
            }
            VkParamError::NegativeSigma => write!(f, "sigma (loss) must be >= 0."),
            VkParamError::BadTheta(t) => {
                write!(f, "theta must be in (0, 1], got {}.", py_float(*t))
            }
            VkParamError::BadNu(v) => write!(
                f,
                "nu (Poisson's ratio) must be in (-1, 1/2), got {}.",
                py_float(*v)
            ),
            VkParamError::BadBoundary => Ok(()),
            VkParamError::NonPositiveTol => write!(f, "couple_tol must be positive."),
            VkParamError::TooFewSweeps => write!(f, "couple_max_iter must be >= 1."),
            VkParamError::Linear(e) => write!(f, "{e}"),
            VkParamError::AiryNotFactorable(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for VkParamError {}

/// The nonlinear plate's constructor arguments, in the Python signature's shape.
#[derive(Debug, Clone, PartialEq)]
pub struct VkSpec {
    /// Side length along `x` (m).
    pub lx: f64,
    /// Side length along `y` (m), snapped to square cells.
    pub ly: f64,
    /// Young's modulus (Pa).
    pub young: f64,
    /// Thickness (m).
    pub thickness: f64,
    /// Poisson's ratio.
    pub nu: f64,
    /// **Volume** density (kg/m^3) — the areal one is derived.
    pub rho: f64,
    /// Sample rate (Hz).
    pub fs: f64,
    /// Segments along `x`.
    pub n: i64,
    /// Frequency-independent loss.
    pub sigma: f64,
    /// Time-averaging weight.
    pub theta: f64,
    /// `None` when the caller could not make sense of the boundary it was handed.
    pub boundary: Option<Boundary>,
    /// Whether the membrane coupling is on.
    pub nonlinear: bool,
    /// Picard convergence threshold on the relative increment.
    pub couple_tol: f64,
    /// Picard sweep cap.
    pub couple_max_iter: i64,
}

impl Default for VkSpec {
    /// The Python signature's defaults, for the arguments that have one.
    fn default() -> Self {
        VkSpec {
            lx: 0.0,
            ly: 0.0,
            young: 0.0,
            thickness: 0.0,
            nu: 0.0,
            rho: 0.0,
            fs: 0.0,
            n: 0,
            sigma: 0.0,
            theta: THETA_DEFAULT,
            boundary: Some(Boundary::Supported),
            nonlinear: true,
            couple_tol: 1e-13,
            couple_max_iter: 50,
        }
    }
}

/// The nonlinear plate's constants: the whole linear plate, plus the two nonlinear operators.
///
/// **The linear half is a [`Params`], not a transcription of one.** The Python original keeps the
/// two classes bit-identical by writing the theta-scheme out twice and saying so in a docstring;
/// here `nonlinear=False` reduces to [`Plate`] by *construction*, because it is literally the same
/// struct and the same [`step_rhs`]. That is what makes §15.2's anchor — 150 steps of `array_equal`
/// across the two classes — structural rather than a claim about two pieces of transcription
/// staying in step.
#[derive(Debug, Clone)]
pub struct VkParams {
    /// The linear plate this one reduces to, whose `rho` is the **areal** density.
    pub lin: Params,
    /// Young's modulus (Pa).
    pub young: f64,
    /// Thickness (m).
    pub thickness: f64,
    /// Volume density (kg/m^3).
    pub rho_v: f64,
    /// Areal density `rho_v * thickness` (kg/m^2).
    pub rho_s: f64,
    /// Flexural rigidity `E e^3 / (12 (1 - nu^2))`.
    pub d: f64,
    /// Membrane coefficient `E e`.
    pub y_mem: f64,
    /// Whether the membrane coupling is on.
    pub nonlinear: bool,
    /// Picard convergence threshold.
    pub couple_tol: f64,
    /// Picard sweep cap.
    pub couple_max_iter: usize,
    /// Per-node mass an external force divides by — `rho_s h^2` supported, `rho_s` free.
    pub force_denominator: f64,
    /// The Monge-Ampere bracket, shared by the `F`-source and the coupling force.
    pub bracket: VonKarmanBracket,
    /// The clamped Airy stress solve.
    pub airy: AiryStressSolver,
    /// Nodes on the full grid, `(N+1)(Ny+1)`.
    pub n_nodes: usize,
}

impl VkParams {
    /// Validate, derive, and build both halves.
    ///
    /// # Errors
    /// Any of [`VkParamError`].
    pub fn new(spec: &VkSpec) -> Result<VkParams, VkParamError> {
        if spec.lx.min(spec.ly).min(spec.fs) <= 0.0 {
            return Err(VkParamError::NonPositive);
        }
        if spec.young <= 0.0 {
            return Err(VkParamError::NonPositiveYoung);
        }
        if spec.thickness <= 0.0 {
            return Err(VkParamError::NonPositiveThickness);
        }
        if spec.rho <= 0.0 {
            return Err(VkParamError::NonPositiveDensity);
        }
        if spec.n < 2 {
            return Err(VkParamError::TooFewSegments);
        }
        if spec.sigma < 0.0 {
            return Err(VkParamError::NegativeSigma);
        }
        if !(spec.theta > 0.0 && spec.theta <= 1.0) {
            return Err(VkParamError::BadTheta(spec.theta));
        }
        if !(-1.0 < spec.nu && spec.nu < 0.5) {
            return Err(VkParamError::BadNu(spec.nu));
        }
        let Some(boundary) = spec.boundary else {
            return Err(VkParamError::BadBoundary);
        };
        if spec.couple_tol <= 0.0 {
            return Err(VkParamError::NonPositiveTol);
        }
        if spec.couple_max_iter < 1 {
            return Err(VkParamError::TooFewSweeps);
        }

        let rho_v = spec.rho;
        let rho_s = rho_v * spec.thickness;
        let d = spec.young * scalar_pow(spec.thickness, 3.0)
            / (12.0 * (1.0 - scalar_pow(spec.nu, 2.0)));
        let kappa = (d / rho_s).sqrt();
        let y_mem = spec.young * spec.thickness;

        let lin = Params::new(&PlateSpec {
            lx: spec.lx,
            ly: spec.ly,
            kappa,
            rho: rho_s,
            fs: spec.fs,
            n: spec.n,
            sigma: spec.sigma,
            theta: spec.theta,
            boundary: Some(boundary),
            domain: Some(Domain::Rectangle),
            nu: Some(spec.nu),
            ..PlateSpec::default()
        })
        .map_err(VkParamError::Linear)?;

        let force_denominator = match boundary {
            Boundary::Supported => rho_s * lin.h * lin.h,
            Boundary::Free => rho_s,
        };
        let bracket = VonKarmanBracket::new(lin.n, lin.ny, lin.h);
        let airy =
            AiryStressSolver::new(lin.n, lin.ny, lin.h).map_err(VkParamError::AiryNotFactorable)?;
        let n_nodes = (lin.n + 1) * (lin.ny + 1);

        Ok(VkParams {
            lin,
            young: spec.young,
            thickness: spec.thickness,
            rho_v,
            rho_s,
            d,
            y_mem,
            nonlinear: spec.nonlinear,
            couple_tol: spec.couple_tol,
            couple_max_iter: spec.couple_max_iter as usize,
            force_denominator,
            bracket,
            airy,
            n_nodes,
        })
    }

    /// Scatter a live-node vector to the full grid, rim held at 0.
    pub fn to_full(&self, u_live: &[f64]) -> Vec<f64> {
        embed(u_live, &self.lin.index_map)
    }

    /// Restrict a full-grid vector to the live nodes, C-order.
    pub fn to_live(&self, full: &[f64]) -> Vec<f64> {
        full.iter()
            .zip(self.lin.mask.flags().iter())
            .filter(|(_, &alive)| alive)
            .map(|(&v, _)| v)
            .collect()
    }

    /// Solve `nabla^4 F = -(Y/2) l(w, w)` for the stress function from a full-grid `w`.
    ///
    /// # Errors
    /// If the factorization cannot back-substitute.
    pub fn airy_f(&self, w_full: &[f64]) -> Result<Vec<f64>, SparseLuError> {
        let br = self.bracket.eval(w_full, w_full);
        // `-0.5 * Y * arr` is `(-0.5 * Y) * arr` -- one scalar formed first, as Python associates.
        let c = -0.5 * self.y_mem;
        let source: Vec<f64> = br.iter().map(|&v| c * v).collect();
        self.airy.solve(&source)
    }

    /// Membrane potential `(1/2Y) F^T B_F F` for a full-grid `F`.
    pub fn membrane_energy_of(&self, f_full: &[f64]) -> f64 {
        self.airy.laplacian_norm_sq(f_full) / (2.0 * self.y_mem)
    }
}

/// `sqrt(x . x)` — `np.linalg.norm` for a 1-D array, which is `sqrt(np.dot(x, x))`.
///
/// This one **is** branched on: `last_residual <= couple_tol` decides whether the Picard loop
/// stops, which is §19.2's hazard exactly. §20.3's refinement says to ask how far the fed quantity
/// sits from the threshold rather than only whether it reaches one, and §27.5 answered it for this
/// loop: the residual falls geometrically and crosses `couple_tol` with orders of room, so the
/// counts never differ on a smooth fixture and differ in the chaotic regime only long after the
/// trajectories have separated for other reasons.
fn norm2(x: &[f64]) -> f64 {
    dot(x, x).sqrt()
}

/// A von Karman plate — the linear plate plus two cached stress functions.
#[derive(Debug, Clone)]
pub struct VkPlate {
    /// Everything constant in time.
    pub p: VkParams,
    /// Current displacement `w^n` over the live nodes.
    pub u: Vec<f64>,
    /// Previous displacement `w^{n-1}`.
    pub u_prev: Vec<f64>,
    /// `F(w^n)` on the full grid, rim 0.
    pub f: Vec<f64>,
    /// `F(w^{n-1})` on the full grid.
    pub f_prev: Vec<f64>,
    /// Completed steps.
    pub n: usize,
    /// Picard sweeps used by the last step.
    pub n_iters: usize,
    /// Did the last step's Picard iteration reach `couple_tol`?
    pub converged: bool,
    /// Its final relative increment.
    pub last_residual: f64,
}

impl VkPlate {
    /// A plate at rest.
    pub fn new(p: VkParams) -> Self {
        let (n_live, n_nodes) = (p.lin.n_live, p.n_nodes);
        VkPlate {
            p,
            u: vec![0.0; n_live],
            u_prev: vec![0.0; n_live],
            f: vec![0.0; n_nodes],
            f_prev: vec![0.0; n_nodes],
            n: 0,
            n_iters: 0,
            converged: true,
            last_residual: 0.0,
        }
    }

    /// Set the initial displacement and velocity, seeding both cached stress functions.
    ///
    /// # Errors
    /// If the Airy factorization cannot back-substitute.
    pub fn set_state(&mut self, u0: &[f64], v0: &[f64]) -> Result<(), SparseLuError> {
        let start = vk_initial_state(u0, v0, &self.p)?;
        self.u = u0.to_vec();
        self.u_prev = start.u_prev;
        self.f = start.f;
        self.f_prev = start.f_prev;
        self.n = 0;
        Ok(())
    }

    /// The linear theta-scheme right-hand side — [`Plate`]'s own, by construction.
    pub fn linear_rhs(&self) -> Vec<f64> {
        step_rhs(&self.u, &self.u_prev, None, &self.p.lin)
    }

    /// Advance one timestep: one solve when linear, a Picard loop when not.
    ///
    /// # Errors
    /// If either factorization cannot back-substitute.
    pub fn step(&mut self, f_ext: Option<&[f64]>) -> Result<(), SparseLuError> {
        let out = vk_step(&self.u, &self.u_prev, &self.f, &self.f_prev, f_ext, &self.p)?;
        self.u_prev = std::mem::replace(&mut self.u, out.u);
        if let Some(f_new) = out.f {
            self.f_prev = std::mem::replace(&mut self.f, f_new);
        }
        self.n += 1;
        self.n_iters = out.n_iters;
        self.converged = out.converged;
        self.last_residual = out.last_residual;
        Ok(())
    }

    /// Current displacement as a full-grid field, rim zero.
    pub fn state(&self) -> Vec<f64> {
        embed(&self.u, &self.p.lin.index_map)
    }

    /// Kinetic plus bending energy — the linear theta-scheme energy, [`Plate`]'s own.
    pub fn linear_energy(&self) -> f64 {
        energy(&self.u, &self.u_prev, &self.p.lin)
    }

    /// Half-step membrane energy; zero when the coupling is off.
    pub fn membrane_energy(&self) -> f64 {
        if !self.p.nonlinear {
            return 0.0;
        }
        0.5 * (self.p.membrane_energy_of(&self.f) + self.p.membrane_energy_of(&self.f_prev))
    }

    /// Total discrete energy — conserved to machine precision when lossless and converged.
    pub fn energy(&self) -> f64 {
        self.linear_energy() + self.membrane_energy()
    }
}

// -- the nonlinear scheme, as free functions over `&VkParams` -------------------------------
//
// Free rather than methods on [`VkPlate`] because the Python binding does not own a `VkPlate`: its
// state buffers are Python arrays (§9.3), and it holds one `VkParams` for the life of the object.
// A step that needed a `VkPlate` would have to clone those parameters — which means cloning **two
// factorizations** — on every call. That is the shape §11.4 warns about, arriving through the
// binding rather than through a getter.

/// The consistent second-order start's three derived buffers.
#[derive(Debug, Clone, PartialEq)]
pub struct VkStart {
    /// `w^{-1}`, bending plus (when nonlinear) the coupling contribution.
    pub u_prev: Vec<f64>,
    /// `F(w^0)`.
    pub f: Vec<f64>,
    /// `F(w^{-1})`.
    pub f_prev: Vec<f64>,
}

/// `w^{-1} = w^0 - k v^0 + 1/2 k^2 a^0` with the full acceleration, plus the seeded `F` cache.
///
/// The bending half is [`initial_previous`] — the *linear* model's exact arithmetic, which is why
/// `nonlinear=False` starts bit-identically to [`Plate`]. The coupling contribution is added on
/// top of it.
///
/// # Errors
/// If the Airy factorization cannot back-substitute.
pub fn vk_initial_state(u0: &[f64], v0: &[f64], p: &VkParams) -> Result<VkStart, SparseLuError> {
    let mut prev = initial_previous(u0, v0, &p.lin);
    if !p.nonlinear {
        return Ok(VkStart {
            u_prev: prev,
            f: vec![0.0; p.n_nodes],
            f_prev: vec![0.0; p.n_nodes],
        });
    }
    let half_k2 = 0.5 * p.lin.k * p.lin.k;
    let u0_full = p.to_full(u0);
    let f0 = p.airy_f(&u0_full)?;
    let coupling_force = p.to_live(&p.bracket.eval(&u0_full, &f0));
    match p.lin.boundary {
        Boundary::Supported => {
            for (q, &c) in prev.iter_mut().zip(coupling_force.iter()) {
                *q += half_k2 * c / p.rho_s;
            }
        }
        Boundary::Free => {
            // The uniform-h^2 coupling force needs the h^2 that A carries; the /W is per-node.
            let h2 = p.lin.h * p.lin.h;
            let scale = half_k2 * (h2 / p.rho_s);
            for (i, (q, &c)) in prev.iter_mut().zip(coupling_force.iter()).enumerate() {
                *q += scale * c / p.lin.w[i];
            }
        }
    }
    let f_prev = p.airy_f(&p.to_full(&prev))?;
    Ok(VkStart {
        u_prev: prev,
        f: f0,
        f_prev,
    })
}

/// One step's outputs: the new displacement, the new stress function, and the loop's diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct VkStep {
    /// `w^{n+1}`.
    pub u: Vec<f64>,
    /// `F^{n+1}` — `None` on the linear path, which does **not** roll the `F` cache. The Python
    /// original returns early there, so a caller that had written `F` by hand keeps `F_prev`.
    pub f: Option<Vec<f64>>,
    /// Picard sweeps used.
    pub n_iters: usize,
    /// Did the loop reach `couple_tol`?
    pub converged: bool,
    /// Its final relative increment.
    pub last_residual: f64,
}

/// Advance one timestep: one prefactored solve when linear, a Picard loop when not.
///
/// The `f_ext` term is added **once, outside** the loop, and that is a fact about the force rather
/// than a shortcut: a bridge spring's `F = K eta^n` depends only on time-`n` state, so it is
/// invariant across the sweeps.
///
/// # Errors
/// If either factorization cannot back-substitute.
pub fn vk_step(
    u: &[f64],
    u_prev: &[f64],
    f: &[f64],
    f_prev: &[f64],
    f_ext: Option<&[f64]>,
    p: &VkParams,
) -> Result<VkStep, SparseLuError> {
    let n_live = p.lin.n_live;
    let k2 = p.lin.k * p.lin.k;
    let mut rhs_lin = step_rhs(u, u_prev, None, &p.lin);
    if let Some(force) = f_ext {
        for (r, &v) in rhs_lin.iter_mut().zip(force.iter()) {
            *r += k2 * v / p.force_denominator;
        }
    }
    if !p.nonlinear {
        return Ok(VkStep {
            u: p.lin.lu.solve(&rhs_lin)?,
            f: None,
            n_iters: 1,
            converged: true,
            last_residual: 0.0,
        });
    }

    // "supported": k^2 l / rho_s (scalar mass, no h^2). "free": the mass matrix A carries W's h^2,
    // so the uniform-h^2 coupling force needs the matching h^2; the /W is the solve's, not ours.
    let mut couple_factor = k2 / p.rho_s;
    if p.lin.boundary == Boundary::Free {
        couple_factor *= p.lin.h * p.lin.h;
    }
    let w_prev_full = p.to_full(u_prev);
    let mut w_j: Vec<f64> = (0..n_live).map(|i| 2.0 * u[i] - u_prev[i]).collect();
    let mut f_new_full = f.to_vec();
    let mut n_iters = 0usize;
    let mut converged = false;
    let mut last_residual = 0.0f64;
    for sweep in 1..=p.couple_max_iter {
        n_iters = sweep;
        let w_j_full = p.to_full(&w_j);
        f_new_full = p.airy_f(&w_j_full)?;
        let w_avg: Vec<f64> = (0..p.n_nodes)
            .map(|i| 0.5 * (w_j_full[i] + w_prev_full[i]))
            .collect();
        let f_avg: Vec<f64> = (0..p.n_nodes)
            .map(|i| 0.5 * (f_new_full[i] + f_prev[i]))
            .collect();
        let coupling = p.to_live(&p.bracket.eval(&w_avg, &f_avg));
        let rhs: Vec<f64> = (0..n_live)
            .map(|i| rhs_lin[i] + couple_factor * coupling[i])
            .collect();
        let w_next = p.lin.lu.solve(&rhs)?;
        let diff: Vec<f64> = (0..n_live).map(|i| w_next[i] - w_j[i]).collect();
        let incr = norm2(&diff);
        let scale = norm2(&w_next);
        w_j = w_next;
        last_residual = incr / scale.max(1e-30);
        if last_residual <= p.couple_tol {
            converged = true;
            break;
        }
    }
    Ok(VkStep {
        u: w_j,
        f: Some(f_new_full),
        n_iters,
        converged,
        last_residual,
    })
}
