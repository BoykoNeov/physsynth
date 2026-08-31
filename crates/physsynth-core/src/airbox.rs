//! The 3-D air box — a rectangular room of air on a Yee grid (HANDOFF §12.H).
//!
//! Port of the `AirBox` half of `physsynth/core/airbox.py`, the largest file in the project. The
//! reference docstrings there are the physics; this module documents only what the translation had
//! to decide.
//!
//! Node-centered pressure on `(Nx+1)(Ny+1)(Nz+1)` nodes, face-centered velocity on the
//! `Nx(Ny+1)(Nz+1)` (and `y`, `z`) faces between them, half a timestep offset in time — the
//! [`crate::bore`]'s Yee cell with two more dimensions and no area profile:
//!
//! ```text
//! u_x^{n+1/2} = u_x^{n-1/2} - (k / (rho0 h)) (p_{i+1} - p_i)^n      [and y, z]
//! p^{n+1}     = p^n - k rho0 c0^2 div(u^{n+1/2})
//! ```
//!
//! # This half of the file, and where the seam is
//!
//! `airbox.py` is 3,976 lines and holds three tiers above `AirBox` itself: the ports (`RoomPort`,
//! `SurfacePort`, `InteriorSurfacePort`) and the six `RoomLoaded*` / `RoomSuspended*` wrappers.
//! None of those is ported here. They reach into the room through a **duck-typed private surface**
//! — `room._w`, `room._W`, `room._beta`, `room._open`, `room._has_walls`, `room._pending`,
//! `room._pending_ports`, `room._ports`, `room._cut_mask`, `room._cut_index`, `room._cuts`,
//! `room._register_cut`, `room._plane_axis` and `room._divergence` — fourteen private names, four
//! of which are **written from outside** (a test clears the three cut fields; a port appends to
//! `_pending_ports`). The binding therefore exposes every one of them as a settable attribute
//! rather than mirroring any of it in Rust: §12.2's finding ("a leading underscore is not a
//! statement about the interface") at its widest so far.
//!
//! # What is bit-identical, and what is not
//!
//! The field is: `p`, `ux`, `uy`, `uz` and both stored half-steps agree with the reference to the
//! bit, because every update is elementwise and every kernel below reproduces NumPy's evaluation
//! order rather than merely its algebra.
//!
//! The **energy books** are not, and it is §14.2's rule with the roles reversed. `acoustic_energy`
//! sums a whole volume with `np.sum` — pairwise blocking above 128 elements — and `step` books the
//! wall flux and the port injection the same way. Matching that would mean transcribing NumPy's
//! blocking, which is the bargain §18.2 refused for SciPy's sparse product and `ops2d::guitar_area`
//! refused for a quadrature: a claim about a library internal, and after §22.1 a claim about the
//! CPU as well. It can be refused *here* for a reason the earlier refusals did not have: the two
//! accumulators `dissipated` and `injected` are pure bookkeeping. Nothing in the update path reads
//! them, so the reduction reaches no timestep — §14.2's question asked and answered "no".
//!
//! # Two discrete outputs, and one transcendental
//!
//! `N = int(round(L / h))` and `node_index` are **decisions**, not numbers (§25.2): a wrong one is a
//! different room or a different source location, and every energy bar passes. Python's `round` is
//! half-to-**even** and Rust's `f64::round` is half-away-from-zero, so both go through
//! `round_ties_even` — the scar `membrane` and `radiation` already carry.
//!
//! `mode_shape` is a tensor cosine, and §22.1 makes `np.cos` a claim about the runner's CPU. It
//! seeds an *initial condition* through `set_mode`, so a last bit there is a different run rather
//! than a different read-out. The Python side therefore builds its three cosine vectors with
//! `math.cos` — §22.3's portable-spelling manoeuvre a sixth time, and free here at ~60 calls per
//! room.

use crate::fmt::py_float;
use crate::pyfloat::scalar_pow;

/// Ambient air density (kg/m^3) — matches `crate::radiation` and `crate::bore`.
pub const RHO0_AIR: f64 = 1.2041;
/// Ambient speed of sound (m/s).
pub const C0_AIR: f64 = 343.0;

/// The six faces of the box, named `<axis><end>`.
pub const FACES: [&str; 6] = ["x0", "x1", "y0", "y1", "z0", "z1"];
/// The three interior plane orientations, named by their normal axis.
pub const PLANES: [&str; 3] = ["x", "y", "z"];
/// Axis names, indexed by axis number.
pub const AXES: [&str; 3] = ["x", "y", "z"];

/// The slack in the CFL check, matching `airbox._LAMBDA_TOL`.
const LAMBDA_TOL: f64 = 1e-12;

/// The 3-D CFL ceiling `1 / sqrt(3)`.
///
/// Spelled as the division rather than as a literal so it is the same double as
/// `1.0 / np.sqrt(3.0)`: a fixture built exactly at the ceiling must construct on both sides or
/// raise on both.
#[inline]
pub fn lambda_max() -> f64 {
    1.0 / 3.0_f64.sqrt()
}

/// Python's `round()` — round-half-to-**even**, which Rust's `f64::round` is not.
///
/// `N` and `node_index` are discrete outputs (§25.2). See the module header.
#[inline]
fn py_round(x: f64) -> f64 {
    x.round_ties_even()
}

/// `(normal axis, end, in-plane axis 0, in-plane axis 1)` for a face index into [`FACES`].
///
/// The two in-plane axes are taken in increasing order — `x0` spans `(y, z)`, `y0` spans `(x, z)`,
/// `z0` spans `(x, y)`. No axis is mirrored on a high face.
pub fn face_axes(face: usize) -> (usize, usize, usize, usize) {
    let axis = face / 2;
    let end = face % 2;
    let (t0, t1) = other_axes(axis);
    (axis, end, t0, t1)
}

/// The two axes that are not `axis`, in increasing order.
#[inline]
pub fn other_axes(axis: usize) -> (usize, usize) {
    match axis {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    }
}

/// Specific acoustic impedance `Z = zeta rho0 c0` from the normalized `zeta`.
#[inline]
pub fn impedance_from_zeta(zeta: f64, rho0: f64, c0: f64) -> f64 {
    zeta * rho0 * c0
}

/// A wall specification, per face: the three spellings `walls=` accepts, already parsed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Wall {
    /// `"rigid"` — infinite impedance, zero admittance.
    Rigid,
    /// `"open"` — a pressure-release face, `p = 0` pinned.
    Open,
    /// A finite specific acoustic impedance `Z > 0` (Pa*s/m).
    Impedance(f64),
}

impl Wall {
    /// The impedance this wall carries, in the units `AirBox.walls` reports: `inf` for rigid,
    /// `0.0` for open.
    pub fn z(self) -> f64 {
        match self {
            Wall::Rigid => f64::INFINITY,
            Wall::Open => 0.0,
            Wall::Impedance(z) => z,
        }
    }

    /// Classify an impedance the way `_normalize_walls` does after it has produced its dict:
    /// `inf` is rigid, `0.0` is open, anything else is a live impedance.
    pub fn from_z(z: f64) -> Wall {
        if z.is_infinite() {
            Wall::Rigid
        } else if z == 0.0 {
            Wall::Open
        } else {
            Wall::Impedance(z)
        }
    }
}

/// A construction-time rejection. Every `Display` is the Python original's message verbatim,
/// because the suite matches on the text.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamError {
    /// One of `L`, `fs`, `h`, `rho0`, `c0` was not positive.
    NonPositiveScalar,
    /// `h` is coarser than the room on some axis.
    TooCoarse {
        /// The offending `h`.
        h: f64,
        /// The room, as passed.
        l: [f64; 3],
        /// The cell counts that came out.
        n: [i64; 3],
    },
    /// `lambda = c0 k / h > 1/sqrt(3)`. Carries the offending `lambda`.
    CflViolated(f64),
    /// A wall impedance was negative or NaN. Carries the face name and the value.
    BadImpedance(&'static str, f64),
    /// A point lies outside the room. Carries everything the message quotes.
    OutsideRoom {
        /// The offending point (m).
        point: [f64; 3],
        /// The node index it would have snapped to.
        index: [i64; 3],
        /// The room the grid spans (m).
        l_actual: [f64; 3],
        /// Cells per axis.
        n: [usize; 3],
    },
}

/// `str()` of a Python tuple of three floats.
fn tuple3(v: &[f64; 3]) -> String {
    format!(
        "({}, {}, {})",
        py_float(v[0]),
        py_float(v[1]),
        py_float(v[2])
    )
}

/// `str()` of a Python tuple of three ints.
fn itup3(v: &[i64; 3]) -> String {
    format!("({}, {}, {})", v[0], v[1], v[2])
}

impl std::fmt::Display for ParamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamError::NonPositiveScalar => {
                write!(f, "L, fs, h, rho0, c0 must all be positive.")
            }
            ParamError::TooCoarse { h, l, n } => write!(
                f,
                "h = {} is coarser than the room {}: N = {} has an axis with no cells. \
                 Refine h (or enlarge L) so every axis takes at least one cell.",
                py_float(*h),
                tuple3(l),
                itup3(n),
            ),
            ParamError::CflViolated(lam) => write!(
                f,
                "CFL violated: lambda = c0*k/h = {:.6} > 1/sqrt(3) = {:.6}. Raise fs, or coarsen \
                 the grid (increase h). Note 3-D has no dispersionless lambda \u{2014} do not tune \
                 toward the ceiling as if it were the string.",
                lam,
                lambda_max()
            ),
            ParamError::BadImpedance(face, z) => write!(
                f,
                "wall '{face}': impedance Z must be >= 0, got {}.",
                py_float(*z)
            ),
            ParamError::OutsideRoom {
                point,
                index,
                l_actual,
                n,
            } => write!(
                f,
                "point {} lies outside the room (0, 0, 0)..{} m. Nearest node index would be {}, \
                 valid range 0..{} per axis.",
                tuple3(point),
                tuple3(l_actual),
                itup3(index),
                itup3(&[n[0] as i64, n[1] as i64, n[2] as i64]),
            ),
        }
    }
}

impl std::error::Error for ParamError {}

/// One lossy face's contribution to the wall ledger: where it is, the per-node transverse area,
/// and its impedance.
#[derive(Debug, Clone)]
pub struct LossyFace {
    /// The face's normal axis.
    pub axis: usize,
    /// `0` for the low face, `1` for the high one.
    pub end: usize,
    /// Per-node area on the slab, in `(t0, t1)` C order: `w[t0][a] * w[t1][b]`.
    pub area: Vec<f64>,
    /// The face's specific acoustic impedance.
    pub z: f64,
}

/// The validated, immutable parameter set: everything about a room that is not its state.
#[derive(Debug, Clone)]
pub struct Params {
    /// Room dimensions as passed (m).
    pub l: [f64; 3],
    /// Sample rate (Hz).
    pub fs: f64,
    /// Grid spacing (m), uniform in all three directions.
    pub h: f64,
    /// Air density (kg/m^3).
    pub rho0: f64,
    /// Speed of sound (m/s).
    pub c0: f64,
    /// Timestep `1 / fs` (s).
    pub k: f64,
    /// Cells per axis.
    pub n: [usize; 3],
    /// The room the grid actually spans, `N h` (m).
    pub l_actual: [f64; 3],
    /// Courant number `c0 k / h`.
    pub lam: f64,
    /// Per-face impedance, in [`FACES`] order.
    pub walls: [f64; 6],
    /// Trapezoidal node weights per axis: `h` interior, `h/2` at each wall.
    pub w: [Vec<f64>; 3],
    /// Volume node weights, shape `(N0+1, N1+1, N2+1)`.
    pub wv: Vec<f64>,
    /// Face weights per axis: `h` along the normal, trapezoid transverse.
    pub wf: [Vec<f64>; 3],
    /// Per-node wall admittance `beta`, shape `(N0+1, N1+1, N2+1)`.
    pub beta: Vec<f64>,
    /// Pressure-release node mask, same shape.
    pub open: Vec<bool>,
    /// Whether any wall is not rigid — the branch `step` takes.
    pub has_walls: bool,
    /// The lossy faces, in [`FACES`] order.
    pub lossy: Vec<LossyFace>,
    /// The default injection node.
    pub source_index: [usize; 3],
    /// `k * rho0 * c0**2` — the compliance gain, formed once in Python's order.
    pub gain: f64,
}

/// Row-major flat index into a `(d0, d1, d2)` array.
#[inline]
pub fn flat(shape: [usize; 3], i: usize, j: usize, k: usize) -> usize {
    (i * shape[1] + j) * shape[2] + k
}

impl Params {
    /// The pressure array's shape.
    #[inline]
    pub fn p_shape(&self) -> [usize; 3] {
        [self.n[0] + 1, self.n[1] + 1, self.n[2] + 1]
    }

    /// The velocity array's shape on `axis`: one *fewer* node along the normal.
    #[inline]
    pub fn u_shape(&self, axis: usize) -> [usize; 3] {
        let mut s = self.p_shape();
        s[axis] -= 1;
        s
    }

    /// Number of pressure nodes.
    #[inline]
    pub fn n_nodes(&self) -> usize {
        let s = self.p_shape();
        s[0] * s[1] * s[2]
    }

    /// Number of velocity faces on `axis`.
    #[inline]
    pub fn n_faces(&self, axis: usize) -> usize {
        let s = self.u_shape(axis);
        s[0] * s[1] * s[2]
    }

    /// Build and validate, in the original's order: shape, positivity, `N`, coarseness, CFL, then
    /// the walls. A call with two faults reports the same one the reference does.
    ///
    /// # The `c0 ** 2`
    ///
    /// `self.c0 ** 2` is a Python *scalar* power, i.e. libm's `pow`, and it is not `c0 * c0` —
    /// they disagree in 79 of 200,000 random positive doubles ([`crate::bore`] measured it, and
    /// §17.2 established that only an opaque call keeps them apart in `--release`). It multiplies
    /// the divergence at every timestep, so it goes through [`scalar_pow`].
    ///
    /// The compliance gain `k * rho0 * c0**2` is formed **once**, left to right, exactly as the
    /// three sites in the reference spell it: `(k * rho0) * pow(c0, 2)`.
    pub fn new(
        l: [f64; 3],
        fs: f64,
        h: f64,
        walls: [Wall; 6],
        source: Option<[f64; 3]>,
        rho0: f64,
        c0: f64,
    ) -> Result<Params, ParamError> {
        // `min(*L, fs, h, rho0, c0) <= 0`, with Python's `min` semantics (keep the first, replace
        // only on a strict `<`), so a NaN behaves the way the reference's does.
        let mut smallest = l[0];
        for &v in &[l[1], l[2], fs, h, rho0, c0] {
            if v < smallest {
                smallest = v;
            }
        }
        if smallest <= 0.0 {
            return Err(ParamError::NonPositiveScalar);
        }
        let k = 1.0 / fs;
        let counts = [
            py_round(l[0] / h) as i64,
            py_round(l[1] / h) as i64,
            py_round(l[2] / h) as i64,
        ];
        if counts.iter().copied().min().unwrap_or(0) < 1 {
            return Err(ParamError::TooCoarse { h, l, n: counts });
        }
        let n = [counts[0] as usize, counts[1] as usize, counts[2] as usize];
        let l_actual = [n[0] as f64 * h, n[1] as f64 * h, n[2] as f64 * h];
        let lam = c0 * k / h;
        if lam > lambda_max() + LAMBDA_TOL {
            return Err(ParamError::CflViolated(lam));
        }
        for (face, wall) in walls.iter().enumerate() {
            if let Wall::Impedance(z) = wall {
                if *z < 0.0 || z.is_nan() {
                    return Err(ParamError::BadImpedance(FACES[face], *z));
                }
            }
        }
        let wall_z = [
            walls[0].z(),
            walls[1].z(),
            walls[2].z(),
            walls[3].z(),
            walls[4].z(),
            walls[5].z(),
        ];

        let w = [trapezoid(n[0], h), trapezoid(n[1], h), trapezoid(n[2], h)];
        let shape = [n[0] + 1, n[1] + 1, n[2] + 1];
        let mut wv = vec![0.0; shape[0] * shape[1] * shape[2]];
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for kk in 0..shape[2] {
                    wv[flat(shape, i, j, kk)] = w[0][i] * w[1][j] * w[2][kk];
                }
            }
        }
        // Face weights: `h` along the normal, the transverse trapezoid product elsewhere. The
        // reference writes `self.h * broadcast(...)`, so `h` is the *left* factor.
        let mut wf: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for (axis, slot) in wf.iter_mut().enumerate() {
            let mut us = shape;
            us[axis] -= 1;
            let (t0, t1) = other_axes(axis);
            let mut buf = vec![0.0; us[0] * us[1] * us[2]];
            for i in 0..us[0] {
                for j in 0..us[1] {
                    for kk in 0..us[2] {
                        let idx = [i, j, kk];
                        buf[flat(us, i, j, kk)] = h * (w[t0][idx[t0]] * w[t1][idx[t1]]);
                    }
                }
            }
            *slot = buf;
        }

        let (beta, open, lossy, has_walls) = wall_closure(&wall_z, &w, shape, k, rho0, c0);

        let default_source = [0.5 * l_actual[0], 0.5 * l_actual[1], 0.5 * l_actual[2]];
        let point = source.unwrap_or(default_source);
        // Only reachable through an explicit `source=`; the default is the room's centre.
        let source_index = node_index(point, h, n).ok_or(ParamError::OutsideRoom {
            point,
            index: node_index_raw(point, h),
            l_actual,
            n,
        })?;

        Ok(Params {
            l,
            fs,
            h,
            rho0,
            c0,
            k,
            n,
            l_actual,
            lam,
            walls: wall_z,
            w,
            wv,
            wf,
            beta,
            open,
            has_walls,
            lossy,
            source_index,
            gain: (k * rho0) * scalar_pow(c0, 2.0),
        })
    }
}

/// Trapezoidal node weights on one axis: `h` interior, `h/2` at each wall.
///
/// The half-cell **is** the rigid-wall closure — the free-beam end-mass lesson, twice generalized.
pub fn trapezoid(n: usize, h: f64) -> Vec<f64> {
    let mut w = vec![h; n + 1];
    w[0] = 0.5 * h;
    let last = w.len() - 1;
    w[last] = 0.5 * h;
    w
}

/// Assemble the per-node `beta` field, the open-face mask and the lossy face list.
///
/// A node on an edge or corner touches several walls; their **admittances add**, so `beta` is a
/// plain sum over the faces that node belongs to and the solve stays 1x1. An `open` face wins over
/// any finite wall it shares a node with.
#[allow(clippy::type_complexity)]
fn wall_closure(
    walls: &[f64; 6],
    w: &[Vec<f64>; 3],
    shape: [usize; 3],
    k: f64,
    rho0: f64,
    c0: f64,
) -> (Vec<f64>, Vec<bool>, Vec<LossyFace>, bool) {
    let n_nodes = shape[0] * shape[1] * shape[2];
    let mut beta = vec![0.0; n_nodes];
    let mut open = vec![false; n_nodes];
    let mut lossy: Vec<LossyFace> = Vec::new();
    let gain = (k * rho0) * scalar_pow(c0, 2.0);
    for (face, &z) in walls.iter().enumerate() {
        if z.is_infinite() {
            continue; // rigid: zero admittance, contributes nothing anywhere
        }
        let (axis, end, t0, t1) = face_axes(face);
        let plane = if end == 0 { 0 } else { shape[axis] - 1 };
        if z == 0.0 {
            for_slab(shape, axis, plane, |idx| open[idx] = true);
            continue; // an ideal pressure-release face carries no energy and sheds none
        }
        let w_wall = if end == 0 {
            w[axis][0]
        } else {
            w[axis][w[axis].len() - 1]
        };
        let contribution = gain / ((2.0 * z) * w_wall);
        for_slab(shape, axis, plane, |idx| beta[idx] += contribution);
        let mut area = Vec::with_capacity(w[t0].len() * w[t1].len());
        for &a in &w[t0] {
            for &b in &w[t1] {
                area.push(a * b);
            }
        }
        lossy.push(LossyFace { axis, end, area, z });
    }
    let has_walls = beta.iter().any(|&b| b != 0.0) || open.iter().any(|&o| o);
    (beta, open, lossy, has_walls)
}

/// Call `f` with the flat index of every node on the slab `axis == plane`, in `(t0, t1)` C order.
fn for_slab<F: FnMut(usize)>(shape: [usize; 3], axis: usize, plane: usize, mut f: F) {
    let (t0, t1) = other_axes(axis);
    let mut idx = [0usize; 3];
    idx[axis] = plane;
    for a in 0..shape[t0] {
        idx[t0] = a;
        for b in 0..shape[t1] {
            idx[t1] = b;
            f(flat(shape, idx[0], idx[1], idx[2]));
        }
    }
}

/// Index of the grid node nearest `point`, or `None` if it lies outside the room.
///
/// A **discrete** output: [`py_round`], not `f64::round`. See the module header.
pub fn node_index(point: [f64; 3], h: f64, n: [usize; 3]) -> Option<[usize; 3]> {
    let raw = [
        py_round(point[0] / h) as i64,
        py_round(point[1] / h) as i64,
        py_round(point[2] / h) as i64,
    ];
    for (d, &i) in raw.iter().enumerate() {
        if i < 0 || i > n[d] as i64 {
            return None;
        }
    }
    Some([raw[0] as usize, raw[1] as usize, raw[2] as usize])
}

/// The same, but reporting the out-of-range triple the reference's message quotes.
pub fn node_index_raw(point: [f64; 3], h: f64) -> [i64; 3] {
    [
        py_round(point[0] / h) as i64,
        py_round(point[1] / h) as i64,
        py_round(point[2] / h) as i64,
    ]
}

// -- the kernels ---------------------------------------------------------------------------------

/// Discrete divergence at every node — the **transpose** of the momentum gradient.
///
/// Each node accumulates the face velocities on either side of it (a wall node sees only the one
/// face it has, which *is* the rigid closure), then divides by the **per-direction** weight `w_d`.
///
/// NumPy's spelling is `dx / wx + dy / wy + dz / wz`, which is `((dx/wx) + (dy/wy)) + (dz/wz)`
/// elementwise — three divisions and two additions per node, in that association. No reduction.
pub fn divergence(p: &Params, ux: &[f64], uy: &[f64], uz: &[f64]) -> Vec<f64> {
    let shape = p.p_shape();
    let mut out = vec![0.0; shape[0] * shape[1] * shape[2]];
    let us = [p.u_shape(0), p.u_shape(1), p.u_shape(2)];
    let u = [ux, uy, uz];
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for kk in 0..shape[2] {
                let node = [i, j, kk];
                let mut acc = 0.0;
                for axis in 0..3 {
                    let mut d = 0.0;
                    if node[axis] < us[axis][axis] {
                        d += u[axis][flat(us[axis], node[0], node[1], node[2])];
                    }
                    if node[axis] > 0 {
                        let mut m = node;
                        m[axis] -= 1;
                        d -= u[axis][flat(us[axis], m[0], m[1], m[2])];
                    }
                    let term = d / p.w[axis][node[axis]];
                    if axis == 0 {
                        acc = term;
                    } else {
                        acc += term;
                    }
                }
                out[flat(shape, i, j, kk)] = acc;
            }
        }
    }
    out
}

/// One momentum half-step `u^{+1/2} = u^{-1/2} - (k / (rho0 h)) grad p`, per axis.
///
/// Cut faces are zeroed here, which is the whole implementation of `add_cut`: this is the single
/// place both `step` and `set_state` produce velocities, so a cut room can never hold a live
/// velocity on a cut face at *any* half-step.
pub fn momentum(
    p: &Params,
    pressure: &[f64],
    u_prev: [&[f64]; 3],
    cuts: &[Vec<usize>; 3],
) -> [Vec<f64>; 3] {
    let c = p.k / (p.rho0 * p.h);
    let shape = p.p_shape();
    let mut out: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for axis in 0..3 {
        let us = p.u_shape(axis);
        let mut buf = vec![0.0; us[0] * us[1] * us[2]];
        for i in 0..us[0] {
            for j in 0..us[1] {
                for kk in 0..us[2] {
                    let mut hi = [i, j, kk];
                    hi[axis] += 1;
                    let d = pressure[flat(shape, hi[0], hi[1], hi[2])]
                        - pressure[flat(shape, i, j, kk)];
                    let idx = flat(us, i, j, kk);
                    buf[idx] = u_prev[axis][idx] - c * d;
                }
            }
        }
        out[axis] = buf;
    }
    apply_cut(&mut out, cuts);
    out
}

/// Zero the cut faces of a `(ux, uy, uz)` triple, in place — `O(cut faces)`.
pub fn apply_cut(u: &mut [Vec<f64>; 3], cuts: &[Vec<usize>; 3]) {
    for (axis, idx) in cuts.iter().enumerate() {
        for &i in idx {
            u[axis][i] = 0.0;
        }
    }
}

/// The pressure sub-step before source and walls: `p^{n+1} = p^n - k rho0 c0^2 div u`.
///
/// The reference writes `self.k * self.rho0 * self.c0**2 * self._divergence()`, which Python
/// evaluates as `((k * rho0) * pow(c0, 2)) * div` — the scalar gain formed first, once, then
/// broadcast. [`Params::gain`] is that scalar.
pub fn pressure_step(p: &Params, p_old: &[f64], div: &[f64]) -> Vec<f64> {
    p_old
        .iter()
        .zip(div.iter())
        .map(|(&po, &d)| po - p.gain * d)
        .collect()
}

/// A queued scalar injection: `(node, q)`.
pub type Injection = ([usize; 3], f64);

/// Apply the queued scalar injections to `p_next` — `p_next[i] += gain * q / W[i]`.
pub fn inject_scalar(p: &Params, p_next: &mut [f64], pending: &[Injection]) {
    let shape = p.p_shape();
    for &(node, q) in pending {
        let idx = flat(shape, node[0], node[1], node[2]);
        p_next[idx] += (p.gain * q) / p.wv[idx];
    }
}

/// Apply one queued port injection: `p_next[nodes] += gain * q * w / W[nodes]`.
///
/// Node `n` takes the share `w[n]` of the volume velocity `q`. Python groups this as
/// `((gain * q) * w[n]) / W[node]` — the two scalars multiplied first, then the weight array, then
/// the elementwise division.
pub fn inject_port(p: &Params, p_next: &mut [f64], nodes: &[usize], w: &[f64], q: f64) {
    let gq = p.gain * q;
    for (n, &idx) in nodes.iter().enumerate() {
        p_next[idx] += (gq * w[n]) / p.wv[idx];
    }
}

/// Apply the wall closure to a freshly-updated pressure field, in place, and book the energy the
/// lossy faces absorbed onto `dissipated`.
///
/// `p_next = (p_next - beta p_old) / (1 + beta)`, then the open faces are pinned to zero. The wall
/// flux is booked at the centered pressure, `k A pbar^2 / Z >= 0` per face; an edge node pays into
/// every lossy face it belongs to, which is what summing admittances into `beta` already charged
/// it.
///
/// The per-face sum is `np.sum` in the reference — pairwise above eight elements — and is summed
/// left to right here. It reaches no timestep; see the module header.
///
/// `dissipated` is accumulated **per face**, not per step, because that is what the reference
/// does: with two lossy walls it forms `(D + f0) + f1`, and adding a per-step subtotal instead
/// would be `D + (f0 + f1)` — a different association, on a running accumulator. It costs nothing
/// to get right and the alternative is a claim in a comment that is quietly false.
pub fn apply_walls(p: &Params, p_next: &mut [f64], p_old: &[f64], dissipated: &mut f64) {
    for (i, pn) in p_next.iter_mut().enumerate() {
        *pn = (*pn - p.beta[i] * p_old[i]) / (1.0 + p.beta[i]);
    }
    for (i, &o) in p.open.iter().enumerate() {
        if o {
            p_next[i] = 0.0;
        }
    }
    let shape = p.p_shape();
    for face in &p.lossy {
        let plane = if face.end == 0 {
            0
        } else {
            shape[face.axis] - 1
        };
        let mut s = 0.0;
        let mut a = 0usize;
        for_slab(shape, face.axis, plane, |idx| {
            let pbar = 0.5 * (p_next[idx] + p_old[idx]);
            s += face.area[a] * (pbar * pbar);
            a += 1;
        });
        *dissipated += (p.k * s) / face.z;
    }
}

/// Work done by one scalar injection: `k * 0.5 * (p_next + p_old) * q`.
#[inline]
pub fn booked_scalar(p: &Params, p_next: &[f64], p_old: &[f64], node: [usize; 3], q: f64) -> f64 {
    let idx = flat(p.p_shape(), node[0], node[1], node[2]);
    p.k * 0.5 * (p_next[idx] + p_old[idx]) * q
}

/// Work done by one port injection, booked from the room's **own** post-closure pressure.
///
/// `pbar_port` is `np.sum(w * 0.5 * (p_next[nodes] + p_old[nodes]))`, summed left to right here;
/// it reaches no timestep. See the module header.
pub fn booked_port(
    p: &Params,
    p_next: &[f64],
    p_old: &[f64],
    nodes: &[usize],
    w: &[f64],
    q: f64,
) -> f64 {
    let mut s = 0.0;
    for (n, &idx) in nodes.iter().enumerate() {
        s += w[n] * (0.5 * (p_next[idx] + p_old[idx]));
    }
    p.k * s * q
}

/// Energy **stored in the air**: compliance `p^2` plus the **cross-time** inductive
/// `u^{n+1/2} u^{n-1/2}` term.
///
/// The four `np.sum`s are pairwise in the reference and left to right here; see the module header.
/// The compliance denominator is `rho0 * c0**2`, the scalar power again.
pub fn acoustic_energy(p: &Params, pressure: &[f64], u: [&[f64]; 3], u_prev: [&[f64]; 3]) -> f64 {
    let mut spot = 0.0;
    for (i, &pv) in pressure.iter().enumerate() {
        spot += p.wv[i] * pv * pv;
    }
    let pot = 0.5 * spot / (p.rho0 * scalar_pow(p.c0, 2.0));
    let mut parts = [0.0; 3];
    for axis in 0..3 {
        let mut s = 0.0;
        for i in 0..u[axis].len() {
            s += p.wf[axis][i] * u[axis][i] * u_prev[axis][i];
        }
        parts[axis] = s;
    }
    let kin = 0.5 * p.rho0 * (parts[0] + parts[1] + parts[2]);
    pot + kin
}

/// The rigid-room mode `cos(l pi i/Nx) cos(m pi j/Ny) cos(n pi k/Nz)` on the grid.
///
/// An **exact** eigenvector of the discrete Neumann Laplacian including at the `h/2` wall nodes.
/// The cosines use `f64::cos` (libm); the Python side uses `math.cos` for the same reason — see
/// the module header on §22.1.
pub fn mode_shape(p: &Params, idx: [usize; 3]) -> Vec<f64> {
    let shape = p.p_shape();
    let mut cos: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for axis in 0..3 {
        let nn = p.n[axis] as f64;
        let q = idx[axis] as f64;
        cos[axis] = (0..shape[axis])
            .map(|i| (((q * std::f64::consts::PI) * i as f64) / nn).cos())
            .collect();
    }
    let mut out = vec![0.0; shape[0] * shape[1] * shape[2]];
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for kk in 0..shape[2] {
                out[flat(shape, i, j, kk)] = cos[0][i] * cos[1][j] * cos[2][kk];
            }
        }
    }
    out
}

/// `mu^2 = (4/h^2) sum_d sin^2(l_d pi / (2 N_d))` — the spatial eigenvalue of mode `idx`.
pub fn mu_squared(p: &Params, idx: [usize; 3]) -> f64 {
    let mut s = 0.0;
    for (axis, &q) in idx.iter().enumerate() {
        let arg = (q as f64 * std::f64::consts::PI) / ((2 * p.n[axis]) as f64);
        let sn = arg.sin();
        s += scalar_pow(sn, 2.0);
    }
    4.0 * s / (p.h * p.h)
}

/// The **exact discrete** frequency (Hz) of mode `idx` — what the scheme will do.
pub fn mode_frequency(p: &Params, idx: [usize; 3]) -> f64 {
    let mu = mu_squared(p, idx).sqrt();
    let arg = (p.c0 * p.k * mu / 2.0).clamp(-1.0, 1.0);
    let omega = (2.0 / p.k) * arg.asin();
    omega / (2.0 * std::f64::consts::PI)
}

/// The **textbook** rigid rectangular-room frequency (Hz), on `l_actual`.
pub fn continuum_mode_frequency(p: &Params, idx: [usize; 3]) -> f64 {
    let mut s = 0.0;
    for (axis, &q) in idx.iter().enumerate() {
        s += scalar_pow(q as f64 / p.l_actual[axis], 2.0);
    }
    0.5 * p.c0 * s.sqrt()
}

/// The half-step-back velocity that makes `set_mode` exact: `u^{-1/2} = (k / (2 rho0 h)) grad p^0`.
///
/// Note it is **omega-free**. The plausible-looking continuum form is *nearly* right and leaves a
/// shape error that masquerades as scheme inaccuracy.
pub fn mode_velocity(p: &Params, p0: &[f64]) -> [Vec<f64>; 3] {
    let s = p.k / (2.0 * p.rho0 * p.h);
    let shape = p.p_shape();
    let mut out: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for axis in 0..3 {
        let us = p.u_shape(axis);
        let mut buf = vec![0.0; us[0] * us[1] * us[2]];
        for i in 0..us[0] {
            for j in 0..us[1] {
                for kk in 0..us[2] {
                    let mut hi = [i, j, kk];
                    hi[axis] += 1;
                    let d = p0[flat(shape, hi[0], hi[1], hi[2])] - p0[flat(shape, i, j, kk)];
                    buf[flat(us, i, j, kk)] = s * d;
                }
            }
        }
        out[axis] = buf;
    }
    out
}

// -- the native owning struct --------------------------------------------------------------------

/// A rectangular room of air, owning its own buffers — for Rust callers and `cargo test`.
///
/// The binding does **not** use this: its buffers must be Python-owned NumPy arrays (§9.3), and
/// its `_pending` / `_pending_ports` / `_cut_*` fields must be Python objects a client can write.
/// This is the thin shell over the kernels that keeps the crate testable without an interpreter.
#[derive(Debug, Clone)]
pub struct AirBox {
    /// The validated parameters.
    pub p: Params,
    /// Pressure field `p^n`.
    pub pressure: Vec<f64>,
    /// Velocity `u^{n+1/2}` per axis.
    pub u: [Vec<f64>; 3],
    /// Velocity `u^{n-1/2}` per axis.
    pub u_prev: [Vec<f64>; 3],
    /// Flat face indices cut on each axis.
    pub cuts: [Vec<usize>; 3],
    /// Cumulative energy absorbed by the walls (>= 0).
    pub dissipated: f64,
    /// Cumulative work done by the soft source.
    pub injected: f64,
    /// Steps taken.
    pub n: usize,
    /// Queued scalar injections for the next step.
    pub pending: Vec<Injection>,
}

impl AirBox {
    /// A room at rest.
    pub fn new(p: Params) -> AirBox {
        let u: [Vec<f64>; 3] = [
            vec![0.0; p.n_faces(0)],
            vec![0.0; p.n_faces(1)],
            vec![0.0; p.n_faces(2)],
        ];
        AirBox {
            pressure: vec![0.0; p.n_nodes()],
            u_prev: u.clone(),
            u,
            cuts: [Vec::new(), Vec::new(), Vec::new()],
            dissipated: 0.0,
            injected: 0.0,
            n: 0,
            pending: Vec::new(),
            p,
        }
    }

    /// Set `p^0` (and optionally `u^{-1/2}`), deriving `u^{1/2}` by one consistent momentum
    /// half-step. Open faces are pinned to zero and the energy books reset.
    pub fn set_state(&mut self, p0: &[f64], u0: Option<[&[f64]; 3]>) {
        let mut p0v = p0.to_vec();
        for (i, &o) in self.p.open.iter().enumerate() {
            if o {
                p0v[i] = 0.0;
            }
        }
        let mut prev: [Vec<f64>; 3] = match u0 {
            Some(u) => [u[0].to_vec(), u[1].to_vec(), u[2].to_vec()],
            None => [
                vec![0.0; self.p.n_faces(0)],
                vec![0.0; self.p.n_faces(1)],
                vec![0.0; self.p.n_faces(2)],
            ],
        };
        apply_cut(&mut prev, &self.cuts);
        self.pressure = p0v;
        self.u = momentum(
            &self.p,
            &self.pressure,
            [&prev[0], &prev[1], &prev[2]],
            &self.cuts,
        );
        self.u_prev = prev;
        self.dissipated = 0.0;
        self.injected = 0.0;
        self.pending.clear();
        self.n = 0;
    }

    /// Queue a soft point injection of volume velocity `q` for the next [`AirBox::step`].
    pub fn inject(&mut self, q: f64, at: Option<[usize; 3]>) {
        self.pending.push((at.unwrap_or(self.p.source_index), q));
    }

    /// Advance one timestep: pressure (plus source and walls) first, then velocity.
    pub fn step(&mut self) {
        let div = divergence(&self.p, &self.u[0], &self.u[1], &self.u[2]);
        let mut p_next = pressure_step(&self.p, &self.pressure, &div);
        if !self.pending.is_empty() {
            inject_scalar(&self.p, &mut p_next, &self.pending);
        }
        if self.p.has_walls {
            apply_walls(&self.p, &mut p_next, &self.pressure, &mut self.dissipated);
        }
        if !self.pending.is_empty() {
            for &(node, q) in &self.pending {
                self.injected += booked_scalar(&self.p, &p_next, &self.pressure, node, q);
            }
            self.pending.clear();
        }
        let u_next = momentum(
            &self.p,
            &p_next,
            [&self.u[0], &self.u[1], &self.u[2]],
            &self.cuts,
        );
        self.u_prev = std::mem::replace(&mut self.u, u_next);
        self.pressure = p_next;
        self.n += 1;
    }

    /// Energy stored in the air (Joules).
    pub fn acoustic_energy(&self) -> f64 {
        acoustic_energy(
            &self.p,
            &self.pressure,
            [&self.u[0], &self.u[1], &self.u[2]],
            [&self.u_prev[0], &self.u_prev[1], &self.u_prev[2]],
        )
    }

    /// The **conserved** total `acoustic + dissipated - injected` (Joules).
    pub fn energy(&self) -> f64 {
        self.acoustic_energy() + self.dissipated - self.injected
    }

    /// Cut every face of the plane `axis == index`, the full cross-section.
    ///
    /// The native shell carries only the unrestricted cut, which is what its tests need; the
    /// binding implements the full `add_cut` (extents, the shared-face refusal and the additive
    /// bookkeeping) against Python objects a client can also write.
    pub fn cut_plane(&mut self, axis: usize, index: usize) {
        let us = self.p.u_shape(axis);
        let (t0, t1) = other_axes(axis);
        let mut idx = [0usize; 3];
        idx[axis] = index;
        for a in 0..us[t0] {
            idx[t0] = a;
            for b in 0..us[t1] {
                idx[t1] = b;
                self.cuts[axis].push(flat(us, idx[0], idx[1], idx[2]));
            }
        }
        self.cuts[axis].sort_unstable();
        self.cuts[axis].dedup();
        let cuts = self.cuts.clone();
        apply_cut(&mut self.u, &cuts);
        apply_cut(&mut self.u_prev, &cuts);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(walls: [Wall; 6]) -> Params {
        // 0.30 x 0.24 x 0.18 m at h = 3 cm -> N = (10, 8, 6), lambda = 0.9/sqrt(3).
        let h = 0.03;
        let fs = C0_AIR * 3.0_f64.sqrt() / (0.9 * h);
        Params::new([0.30, 0.24, 0.18], fs, h, walls, None, RHO0_AIR, C0_AIR).unwrap()
    }

    fn rigid() -> [Wall; 6] {
        [Wall::Rigid; 6]
    }

    /// A cheap deterministic field, so `cargo test` needs no RNG dependency.
    fn noise(n: usize) -> Vec<f64> {
        let mut s: u64 = 0x2545_F491_4F6C_DD1D;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn the_grid_counts_and_the_courant_number() {
        let p = params(rigid());
        assert_eq!(p.n, [10, 8, 6]);
        assert_eq!(p.p_shape(), [11, 9, 7]);
        assert!((p.lam - 0.9 * lambda_max()).abs() < 1e-15);
        assert_eq!(p.source_index, [5, 4, 3]);
    }

    #[test]
    fn the_cfl_ceiling_is_allowed_and_a_hair_past_it_is_not() {
        let h = 0.03;
        let at = C0_AIR * 3.0_f64.sqrt() / h;
        assert!(Params::new([0.3, 0.24, 0.18], at, h, rigid(), None, RHO0_AIR, C0_AIR).is_ok());
        // Lowering fs raises lambda: the ceiling is a *floor* on the sample rate.
        let past = at / 1.001;
        assert!(matches!(
            Params::new([0.3, 0.24, 0.18], past, h, rigid(), None, RHO0_AIR, C0_AIR),
            Err(ParamError::CflViolated(_))
        ));
    }

    #[test]
    fn a_rigid_room_conserves_energy() {
        let mut box_ = AirBox::new(params(rigid()));
        let p0 = noise(box_.p.n_nodes());
        box_.set_state(&p0, None);
        let e0 = box_.energy();
        for _ in 0..2000 {
            box_.step();
        }
        let drift = (box_.energy() - e0).abs() / e0.abs();
        assert!(drift < 1e-12, "drift {drift:e}");
    }

    #[test]
    fn a_lossy_room_is_passive_and_the_ledger_closes() {
        let mut walls = rigid();
        walls[0] = Wall::Impedance(2.0 * RHO0_AIR * C0_AIR);
        walls[3] = Wall::Impedance(0.5 * RHO0_AIR * C0_AIR);
        let mut box_ = AirBox::new(params(walls));
        let p0 = noise(box_.p.n_nodes());
        box_.set_state(&p0, None);
        let e0 = box_.energy();
        let mut last = box_.acoustic_energy();
        for _ in 0..1000 {
            box_.step();
            let now = box_.acoustic_energy();
            assert!(now <= last * (1.0 + 1e-12), "acoustic energy rose");
            last = now;
        }
        assert!(box_.dissipated > 0.0);
        let drift = (box_.energy() - e0).abs() / e0.abs();
        assert!(drift < 1e-12, "ledger drift {drift:e}");
    }

    #[test]
    fn an_open_face_is_pinned_to_exactly_zero() {
        let mut walls = rigid();
        walls[3] = Wall::Open; // y1
        let mut box_ = AirBox::new(params(walls));
        let p0 = noise(box_.p.n_nodes());
        box_.set_state(&p0, None);
        let shape = box_.p.p_shape();
        for _ in 0..50 {
            box_.step();
            for i in 0..shape[0] {
                for k in 0..shape[2] {
                    assert_eq!(box_.pressure[flat(shape, i, shape[1] - 1, k)], 0.0);
                }
            }
        }
    }

    #[test]
    fn a_soft_source_books_exactly_what_it_puts_in() {
        let mut box_ = AirBox::new(params(rigid()));
        box_.set_state(&vec![0.0; box_.p.n_nodes()], None);
        for i in 0..200 {
            box_.inject(1e-3 * ((i as f64) * 0.1).sin(), None);
            box_.step();
        }
        assert!(box_.injected.abs() > 1e-12);
        let drift = box_.energy().abs() / box_.injected.abs();
        assert!(drift < 1e-10, "source ledger drift {drift:e}");
    }

    #[test]
    fn the_exact_discrete_mode_oscillates_at_its_own_frequency() {
        let p = params(rigid());
        let idx = [1, 0, 0];
        let f = mode_frequency(&p, idx);
        let shape_vec = mode_shape(&p, idx);
        let u0 = mode_velocity(&p, &shape_vec);
        let mut box_ = AirBox::new(p);
        box_.set_state(&shape_vec, Some([&u0[0], &u0[1], &u0[2]]));
        let probe = 0usize;
        for n in 1..=400 {
            box_.step();
            let want = (2.0 * std::f64::consts::PI * f * (n as f64) * box_.p.k).cos();
            assert!(
                (box_.pressure[probe] - want).abs() < 1e-11,
                "step {n}: {} vs {want}",
                box_.pressure[probe]
            );
        }
    }

    #[test]
    fn a_full_cut_makes_two_rooms_and_costs_no_energy() {
        let mut box_ = AirBox::new(params(rigid()));
        let p0 = noise(box_.p.n_nodes());
        box_.set_state(&p0, None);
        box_.cut_plane(2, 2);
        // Re-seed after the cut so the consistent start sees it.
        box_.set_state(&p0, None);
        let e0 = box_.energy();
        for _ in 0..500 {
            box_.step();
            for &i in &box_.cuts[2] {
                assert_eq!(box_.u[2][i], 0.0);
            }
        }
        let drift = (box_.energy() - e0).abs() / e0.abs();
        assert!(drift < 1e-12, "cut drift {drift:e}");
    }

    #[test]
    fn the_wall_admittances_add_at_an_edge() {
        let mut walls = rigid();
        let z = 3.0 * RHO0_AIR * C0_AIR;
        walls[0] = Wall::Impedance(z); // x0
        walls[2] = Wall::Impedance(z); // y0
        let p = params(walls);
        let shape = p.p_shape();
        let face_only = p.beta[flat(shape, 0, 3, 3)];
        let edge = p.beta[flat(shape, 0, 0, 3)];
        assert!(face_only > 0.0);
        // The x0 node weight and the y0 node weight are both h/2, so the edge is exactly twice.
        assert_eq!(edge, face_only + face_only);
    }

    #[test]
    fn round_is_half_to_even_where_rust_rounds_away() {
        // L / h = 2.5 lands on the tie: Python's round gives 2, f64::round gives 3.
        let h = 0.2;
        let p = Params::new([0.5, 0.5, 0.5], 5000.0, h, rigid(), None, RHO0_AIR, C0_AIR).unwrap();
        assert_eq!(p.n, [2, 2, 2]);
    }
}
