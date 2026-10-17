//! The port tier of `physsynth/core/airbox.py` — a room's two-way terminals.
//!
//! Port of `RoomPort`, `_PatchPort`, `SurfacePort` and `InteriorSurfacePort`, plus the two module
//! helpers they share (`_face_axes`, `_free_pressure_nodes`). The reference docstrings were the
//! physics and this header documented only what the translation had to decide. They were deleted
//! with the room's first half (deletion 9, `58654e5`), so their derivations are carried from
//! `58654e5^` onto the native items of the same names: the three port classes' docstrings on
//! [`RoomPort`], [`SurfacePort`] and [`InteriorSurfacePort`] (their parameter lists left out — the
//! native constructors and fields carry those — and their notes kept), and the method docstrings
//! that carry a derivation on the native methods and helpers. Each is verbatim save for its
//! cross-references. Where the text says the port "registers itself with the room", read the
//! native ownership described on [`RoomPort`]. The room's own physics is [`crate::airbox`]'s.
//!
//! **The measured figures in the carried text are the Python implementation's.** They were
//! measured before the port and have not been re-measured on this one; re-measure, and save
//! the native print, before quoting any of them as a Rust result.
//!
//! # Why the ports go before the wrappers
//!
//! `airbox.py` has three tiers: the room (ported in plan §30), the ports, and six `RoomLoaded*` /
//! `RoomSuspended*` wrappers. The obvious argument for taking the ports first is dependency order,
//! and it is the weaker one. The real reason is §13.2: a wrapper's `step` calls
//! `port.free_pressure()`, solves, then calls `port.inject(q)` — it hands control **out** twice per
//! step. A Rust wrapper over a Python port would therefore be a `&mut self` pymethod that must
//! release and re-take its own state mid-step, which is exactly what `bore` and `reed` hit and
//! exactly what PyO3 refuses. A Python wrapper over a Rust port is the ordinary direction and needs
//! no such contortion. Take the callee first.
//!
//! The second reason is that this tier owns **no** factorization. All six `splu` calls in the file
//! are in the wrapper tier, so nothing here is in plan §4's sparse-LU risk group and every claim
//! below can be exact.
//!
//! # The room is read through Python, not through `Params`
//!
//! Every kernel here takes a [`RoomView`] — plain slices — and the binding fills one by reading the
//! room's attributes. It does *not* require the room to be the Rust `AirBox`. That is deliberate:
//! §29.1 found `connection` polymorphic over its collaborators' types, and the port tier has the
//! same shape one level down. A port must work against `AirBoxPy` and against `_rs.AirBox` alike,
//! because the parity file builds one of each and because a caller may keep the flag off for the
//! room and on for the port. Duck typing is the interface; a `Params` would have been a narrowing.
//!
//! # Reductions: this tier's reach the timestep, so they are exact
//!
//! §30.2 established `np.sum`'s cutoff and declined to transcribe the blocking above it, on the
//! grounds that the room's two energy books are pure bookkeeping. Here the answer to §14.2's
//! question is **yes**: `w = W / W.sum()` is the share of the volume velocity each node receives,
//! `R_room` is the resistance the coupled solve divides by, and `free_pressure` is the pressure the
//! body is pushed by. A last bit in any of the three is a different trajectory, not a different
//! read-out. So they all go through [`crate::reduce::sum`], which reproduces `np.sum` exactly at
//! every length — see that module for what the claim rests on.
//!
//! # The triple product, and the one association that is observable
//!
//! `load_matrix = (T.T @ diags(R) @ T).tocsr()` is a sparse contraction, which after §26 and §27 is
//! three separate questions. Measured over the fixtures the suite's own builders make (§27.2's
//! method — enumerate, do not sample):
//!
//! * **Stored order**: not an issue here, unusually. `T` comes from `coo_matrix(...).tocsr()`, which
//!   canonicalizes, and the product's `.tocsr()` from a CSC canonicalizes again — measured
//!   ascending in every row of every fixture, so §18.2's `portable.canonical` is not needed and
//!   [`crate::sparse::Csr`]'s own sorting is already right.
//! * **Values**: an ascending-`k` accumulation reproduces SciPy's kernel bit for bit — §26.2's
//!   finding holding again, 0 differing entries of 6,845 over five fixtures.
//! * **Association**: this one *is* live. `diags(R)` sits between the two factors and Python
//!   left-associates, so SciPy forms `(T_ki R_k) T_kj` and **not** `T_ki (R_k T_kj)`. Those are
//!   different doubles in **2,028 of 6,845** entries. The diagonal is not neutral, and which side
//!   it folds into has to be copied rather than chosen.
//!
//! The blind fixture is worth naming, because it is one of the six the golden test in
//! `tests/test_airbox_dipole.py` pins. With `spreading="nearest"` every surface node lands on one
//! air node with weight exactly 1, so every stored entry in a row of `T` is the same uniform node
//! area — and `(x d) x` and `x (d x)` are the same double **identically**, for every `x` and `d`
//! (measured 0 differences in 200,000 random pairs, against 69,943 when the two outer factors
//! differ). So the association is unobservable on that fixture and observable on all the others:
//! §26.5's "do the outer factors share a mantissa" question with a sharper answer — here they are
//! not merely commensurate, they are *the same number*.

use crate::airbox::AirBox;
use crate::fmt::{py_float, py_general};
use crate::pyfloat::scalar_pow;
use crate::reduce;
use crate::sparse::Csr;

/// The face names, in the reference's order.
pub const FACES: [&str; 6] = ["x0", "x1", "y0", "y1", "z0", "z1"];
/// The interior-plane names, in the reference's order.
pub const PLANES: [&str; 3] = ["x", "y", "z"];
/// Axis letters, indexed by axis number.
pub const AXES: [char; 3] = ['x', 'y', 'z'];

/// How a surface node's area is distributed over the air nodes under it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Spreading {
    /// Bilinear over the `{i, i+1}` stencil in each in-plane direction — the default.
    Bilinear,
    /// The whole area on the nearest node. The measured negative control, not a configuration.
    Nearest,
}

impl Spreading {
    /// Parse the reference's spelling, or `None` for an unknown one (the caller quotes it).
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "bilinear" => Some(Spreading::Bilinear),
            "nearest" => Some(Spreading::Nearest),
            _ => None,
        }
    }

    /// The reference's spelling, for an error message that quotes it back.
    pub fn name(self) -> &'static str {
        match self {
            Spreading::Bilinear => "bilinear",
            Spreading::Nearest => "nearest",
        }
    }
}

/// `(normal axis, end, in-plane axis 0, in-plane axis 1)` for a face name — `_face_axes`.
///
/// The two in-plane axes come back in increasing order and no axis is mirrored on a high face; the
/// inward normal is carried by the sign convention, never by flipping a coordinate.
pub fn face_axes(face: &str) -> Option<(usize, usize, usize, usize)> {
    let pos = FACES.iter().position(|&f| f == face)?;
    let axis = pos / 2;
    let end = pos % 2;
    let (t0, t1) = match axis {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    Some((axis, end, t0, t1))
}

/// The axis a plane name names — `AirBox._plane_axis`.
pub fn plane_axis(plane: &str) -> Option<usize> {
    PLANES.iter().position(|&p| p == plane)
}

/// Everything a port kernel needs from its room, as borrowed slices.
///
/// Filled by the binding from the room's Python attributes, so it is agnostic to whether the room
/// itself is Python or Rust. Every field is read-only: a port never writes a room's state, it
/// appends to `_pending_ports` and lets `AirBox.step` do the work.
pub struct RoomView<'a> {
    /// Cells per axis, so the node shape is `[n0 + 1, n1 + 1, n2 + 1]`.
    pub n: [usize; 3],
    /// Grid spacing (m).
    pub h: f64,
    /// Timestep (s).
    pub k: f64,
    /// Air density (kg/m^3).
    pub rho0: f64,
    /// Sound speed (m/s).
    pub c0: f64,
    /// Node-centered pressure, C order over the node shape.
    pub p: &'a [f64],
    /// Face-centered velocity per axis; axis `a` has extent `n[a]` along `a` and `n + 1` elsewhere.
    pub u: [&'a [f64]; 3],
    /// Per-direction trapezoid weights, `_w[axis]`, length `n[axis] + 1`.
    pub w: [&'a [f64]; 3],
    /// The tensor volume weight `_W`, C order over the node shape.
    pub node_w: &'a [f64],
    /// The wall-closure denominator field `_beta`, C order over the node shape.
    pub beta: &'a [f64],
    /// Whether the room has any lossy or open face — `_has_walls`.
    pub has_walls: bool,
}

impl RoomView<'_> {
    /// The pressure node shape.
    pub fn node_shape(&self) -> [usize; 3] {
        [self.n[0] + 1, self.n[1] + 1, self.n[2] + 1]
    }

    /// Flat C-order offset of a pressure node.
    pub fn flat(&self, i: [usize; 3]) -> usize {
        let s = self.node_shape();
        (i[0] * s[1] + i[1]) * s[2] + i[2]
    }

    /// `k rho0 c0^2`, as NumPy evaluates it — `(k * rho0) * c0**2`, the power through
    /// [`scalar_pow`] because CPython's `float.__pow__` is libm's `pow` and not a multiply (§30.7).
    pub fn gain(&self) -> f64 {
        (self.k * self.rho0) * scalar_pow(self.c0, 2.0)
    }
}

/// A port's node set: three parallel index arrays, the fancy-index triple the reference holds.
pub type Nodes = [Vec<usize>; 3];

/// Per-node open-circuit centered pressure at `nodes` — `_free_pressure_nodes`.
///
/// Replicates `AirBox.step`'s order exactly: divergence, then the wall closure. The three axis
/// contributions accumulate in axis order into a running `div` that starts at `0.0`, which is the
/// reference's `div += ...` and matters because `0.0 + (-0.0)` is `+0.0`.
///
/// The local divergence read is bit-identical to the full-array divergence followed by the closure,
/// on low **and** high faces alike (the `idx < N` / `idx > 0` branches are different code paths and
/// both are exercised), for every wall type *except* an `open` face — where [`AirBox::step`] pins
/// `p = 0` and this read does not. Ports touching an open face are refused at construction, so the
/// identity holds by construction rather than by luck.
pub fn free_pressure_nodes(view: &RoomView<'_>, nodes: &[&[usize]; 3]) -> Vec<f64> {
    let count = nodes[0].len();
    let shape = view.node_shape();
    let gain = view.gain();
    let mut out = Vec::with_capacity(count);
    for (&i0, (&i1, &i2)) in nodes[0].iter().zip(nodes[1].iter().zip(nodes[2].iter())) {
        let idx = [i0, i1, i2];
        let mut div = 0.0;
        for axis in 0..3 {
            let n_face = view.n[axis];
            let mut ushape = shape;
            ushape[axis] = n_face;
            let i = idx[axis];
            let plus = if i < n_face {
                let mut pick = idx;
                pick[axis] = i.min(n_face - 1);
                view.u[axis][(pick[0] * ushape[1] + pick[1]) * ushape[2] + pick[2]]
            } else {
                0.0
            };
            let minus = if i > 0 {
                let mut pick = idx;
                pick[axis] = i - 1;
                view.u[axis][(pick[0] * ushape[1] + pick[1]) * ushape[2] + pick[2]]
            } else {
                0.0
            };
            div += (plus - minus) / view.w[axis][i];
        }
        let flat = view.flat(idx);
        let p_node = view.p[flat];
        let mut p_free = p_node - gain * div;
        if view.has_walls {
            let beta = view.beta[flat];
            p_free = (p_free - beta * p_node) / (1.0 + beta);
        }
        out.push(0.5 * (p_free + p_node));
    }
    out
}

// -- RoomPort: the lumped tier ------------------------------------------------------------------

/// The nodes of a ball of radius `radius` around `index`, in C order — `RoomPort`'s node set.
///
/// `np.nonzero` on a 3-D boolean yields C order, so the nested loop below is the same order. The
/// squared offsets go through a plain multiply because NumPy's power *ufunc loop* spells `x**2` as
/// `x * x` (§16.2) and this is the array path.
pub fn ball_nodes(n: [usize; 3], h: f64, index: [usize; 3], radius: f64) -> Nodes {
    let offs: Vec<Vec<f64>> = (0..3)
        .map(|d| {
            (0..=n[d])
                .map(|i| h * ((i as i64 - index[d] as i64) as f64))
                .collect()
        })
        .collect();
    let r2 = radius * radius;
    let mut out: Nodes = [Vec::new(), Vec::new(), Vec::new()];
    for i0 in 0..=n[0] {
        let a = offs[0][i0] * offs[0][i0];
        for i1 in 0..=n[1] {
            let ab = a + offs[1][i1] * offs[1][i1];
            for (i2, &o2) in offs[2].iter().enumerate() {
                if ab + o2 * o2 <= r2 {
                    out[0].push(i0);
                    out[1].push(i1);
                    out[2].push(i2);
                }
            }
        }
    }
    out
}

/// The normalized volume weights `w = W / W.sum()` and the node weights `W` they came from.
///
/// The reduction is `np.sum`, and it reaches the timestep — the room injects `w * U` — so it goes
/// through [`reduce::sum`] rather than a plain loop.
pub fn port_weights(view: &RoomView<'_>, nodes: &[&[usize]; 3]) -> (Vec<f64>, Vec<f64>) {
    let big_w: Vec<f64> = (0..nodes[0].len())
        .map(|m| view.node_w[view.flat([nodes[0][m], nodes[1][m], nodes[2][m]])])
        .collect();
    let total = reduce::sum(&big_w);
    let w = big_w.iter().map(|&x| x / total).collect();
    (w, big_w)
}

/// The lumped internal resistance `R_room = sum_n w_n^2 k rho0 c0^2 / (2 W_n (1 + beta_n))`.
///
/// The evaluation order is the reference's and is not the same as [`patch_resistance`]'s: NumPy
/// reads `self.w * self.w * room.k * room.rho0 * room.c0**2 / (...)` left to right, so the
/// numerator is `((((w w) k) rho0) c0^2)` — four separate roundings — where the patch tier writes
/// `room.k * room.rho0 * room.c0**2` as one numerator and divides once.
pub fn r_room(view: &RoomView<'_>, nodes: &[&[usize]; 3], w: &[f64], big_w: &[f64]) -> f64 {
    let c0_sq = scalar_pow(view.c0, 2.0);
    let terms: Vec<f64> = (0..w.len())
        .map(|m| {
            let beta = view.beta[view.flat([nodes[0][m], nodes[1][m], nodes[2][m]])];
            (((w[m] * w[m]) * view.k) * view.rho0) * c0_sq / ((2.0 * big_w[m]) * (1.0 + beta))
        })
        .collect();
    reduce::sum(&terms)
}

// -- RoomPort: the lumped tier, as a value ---------------------------------------------------

/// A construction- or call-time rejection from the lumped port.
///
/// Every `Display` is the reference's message verbatim, because the retired Python suite matched
/// on the text and the native bars match on it now.
#[derive(Debug, Clone, PartialEq)]
pub enum PortError {
    /// The requested centre is not in the room. Carries the room's own refusal.
    OutsideRoom(crate::airbox::ParamError),
    /// `radius` was not a positive finite length. Carries it.
    BadRadius(f64),
    /// The ball is finer than the grid, so it is a point port wearing a radius.
    UnresolvableRadius {
        /// The radius asked for (m).
        radius: f64,
        /// The grid spacing (m).
        h: f64,
        /// The centre node it collapsed onto.
        index: [usize; 3],
    },
    /// The footprint reaches a pressure-release face. Carries the centre and the faces.
    OnOpenFace {
        /// The centre node index.
        index: [usize; 3],
        /// The offending face names, quoted, in [`FACES`] order.
        faces: Vec<String>,
    },
    /// Two ports share a node. Carries everything the message quotes.
    Overlapping {
        /// The centre node index of the port being built.
        index: [usize; 3],
        /// The first shared node, unravelled.
        node: [usize; 3],
        /// How the existing port names itself.
        other: String,
        /// How many nodes the two have in common.
        count: usize,
        /// The grid spacing, for the snapping note (m).
        h: f64,
    },
    /// A second solve inside one room step. Carries the centre and the room's step count.
    NotReady {
        /// The centre node index.
        index: [usize; 3],
        /// `room.n` at the moment of the refusal.
        n: usize,
    },
    /// `areas` was not one value per surface node.
    AreasLength {
        /// How many surface nodes `coords` carries.
        expected: usize,
        /// How many areas were passed.
        got: usize,
    },
    /// A surface node area was negative or not finite.
    BadAreas,
    /// The placed footprint reaches outside the plane it is mounted on.
    FootprintOutside {
        /// How the port names its mounting.
        where_: String,
        /// The footprint's low edge on the offending axis (m).
        lo: f64,
        /// Its high edge (m).
        hi: f64,
        /// The offending axis.
        axis: usize,
        /// How far the plane reaches on that axis (m).
        extent: f64,
        /// Where the footprint currently sits on that axis (m).
        origin: f64,
    },
    /// The spread stencil reaches a node on the plane's own rim, which touches a second wall.
    InPlaneRim {
        /// How the port names its mounting.
        where_: String,
        /// The lowest in-plane node index reached.
        lo: i64,
        /// The highest.
        hi: i64,
        /// The offending in-plane axis.
        axis: usize,
        /// Cells on that axis.
        n_axis: usize,
    },
    /// Air nodes under the footprint that no surface node feeds — the comb refusal.
    TooCoarseFootprint {
        /// How the port names its mounting.
        where_: String,
        /// How many footprint nodes are fed by nothing.
        unfed: usize,
        /// How big the footprint is.
        foot: usize,
        /// The grid spacing (m).
        h: f64,
        /// The spreading in force.
        spreading: Spreading,
    },
    /// The surface touches a pressure-release face.
    SurfaceOnOpenFace {
        /// How the port names its mounting.
        where_: String,
        /// The offending face names, quoted, in [`FACES`] order.
        faces: Vec<String>,
    },
    /// The surface shares air nodes with a port already on the room.
    SurfaceOverlapping {
        /// How the port names its mounting.
        where_: String,
        /// How many nodes this surface covers.
        count: usize,
        /// The first shared node, unravelled.
        node: [usize; 3],
        /// How the existing port names itself.
        other: String,
        /// How many nodes the two have in common.
        shared: usize,
    },
    /// A second solve inside one room step, from a distributed port.
    SurfaceNotReady {
        /// How the port names its mounting.
        where_: String,
        /// `room.n` at the moment of the refusal.
        n: usize,
    },
    /// The face name is not one of [`FACES`]. Carries it.
    UnknownFace(String),
    /// An interior surface's face index does not leave both straddling node planes interior.
    InteriorIndexOutOfRange {
        /// The plane name, as passed.
        plane: String,
        /// The index asked for.
        index: i64,
        /// How many faces the plane has there.
        n_face: usize,
    },
    /// `inject` was handed a `q` of the wrong length.
    QLength {
        /// How long it should have been.
        expected: usize,
        /// How long it was.
        got: usize,
        /// Whether the vector is per-face (interior) rather than per-node (wall-mounted).
        per_face: bool,
        /// The port's node count, which the per-face message quotes.
        node_count: usize,
    },
    /// The room refused this port's cut. Carries its refusal.
    Cut(crate::airbox::CutError),
}

impl std::fmt::Display for PortError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PortError::OutsideRoom(e) => write!(f, "{e}"),
            PortError::BadRadius(r) => write!(
                f,
                "port radius must be a positive length, got {}.",
                py_float(*r)
            ),
            PortError::UnresolvableRadius { radius, h, index } => write!(
                f,
                "port radius {} m is smaller than the grid can resolve (h = {}): the ball \
                 contains only the centre node {}, so this would silently be a point port with a \
                 grid-dependent load magnitude. Coarsen the request, refine h, or pass \
                 radius=None to ask for a point port on purpose.",
                py_float(*radius),
                py_float(*h),
                index_repr(*index),
            ),
            PortError::OnOpenFace { index, faces } => write!(
                f,
                "port at {} touches the open (pressure-release) face(s) [{}], where p is pinned \
                 to 0: pbar_free and R_room are both exactly zero, so the body would radiate into \
                 a short circuit \u{2014} perfectly conservative, perfectly silent, and invisible \
                 to the energy report. Move the port off that face, or give the face a finite \
                 impedance.",
                index_repr(*index),
                faces.join(", "),
            ),
            PortError::Overlapping {
                index,
                node,
                other,
                count,
                h,
            } => write!(
                f,
                "port at {} shares node {} with the existing port at {other} ({count} node(s) in \
                 common). Overlapping ports are not independent within a step, so each one's \
                 solve uses a pressure that never occurred and the energy ledgers stop matching. \
                 Note grid snapping: two nearby centres collapse onto one node at h = {}.",
                index_repr(*index),
                index_repr(*node),
                py_float(*h),
            ),
            PortError::NotReady { index, n } => write!(
                f,
                "port at {} was asked to solve twice within one room step (room.n = {n}). A port \
                 does not step its room \u{2014} the caller does, once, after every port has \
                 solved:  for inst in instruments: inst.step(...)  then  room.step(). Without it \
                 the room is frozen and the body is loaded by a stale field, silently.",
                index_repr(*index),
            ),
            PortError::AreasLength { expected, got } => write!(
                f,
                "areas must have shape ({expected},) (one per surface node), got ({got},)."
            ),
            PortError::BadAreas => write!(f, "surface node areas must be finite and >= 0 (m^2)."),
            PortError::FootprintOutside {
                where_,
                lo,
                hi,
                axis,
                extent,
                origin,
            } => write!(
                f,
                "the surface's footprint spans {}..{} m along {}, outside {where_}, which is \
                 0..{} m there. Move it with origin= (it currently sits at {} m on that axis), or \
                 enlarge the room.",
                py_general(*lo, 6),
                py_general(*hi, 6),
                AXES[*axis],
                py_general(*extent, 6),
                py_general(*origin, 6),
            ),
            PortError::InPlaneRim {
                where_,
                lo,
                hi,
                axis,
                n_axis,
            } => write!(
                f,
                "the surface's spread stencil reaches air node index {lo}..{hi} along {} on \
                 {where_}, but a node on the plane's own rim (0 or {n_axis}) touches a SECOND \
                 wall: it carries half the node weight W and the sum of two wall admittances, so \
                 R_j stops being uniform across the patch and the spreading operator's reflection \
                 equivariance stops holding. Keep the footprint plus one air cell strictly inside \
                 the plane -- move it with origin=, enlarge the room, or shrink the surface.",
                AXES[*axis]
            ),
            PortError::TooCoarseFootprint {
                where_,
                unfed,
                foot,
                h,
                spreading,
            } => write!(
                f,
                "{unfed} of {foot} air node(s) under the surface's footprint on {where_} are fed \
                 by no surface node, so the acoustic source would be a comb at the grid scale. \
                 The footprint is measured span-wise (per row and per column of air nodes), so \
                 this is about the surface's spacing and not its outline: too coarse for h_air = \
                 {} m (spreading='{}'). Refine the surface, or coarsen the air grid.",
                py_general(*h, 6),
                spreading.name()
            ),
            PortError::SurfaceOnOpenFace { where_, faces } => write!(
                f,
                "the surface on {where_} touches the open (pressure-release) face(s) [{}], where \
                 p is pinned to 0: pbar_free and every R_j are exactly zero, so the surface would \
                 radiate into a short circuit -- perfectly conservative, perfectly silent, and \
                 invisible to the energy report. Give that face a finite impedance, or mount the \
                 surface elsewhere.",
                faces.join(", ")
            ),
            PortError::SurfaceOverlapping {
                where_,
                count,
                node,
                other,
                shared,
            } => write!(
                f,
                "the surface on {where_} ({count} nodes) shares node {} with the existing port at \
                 {other} ({shared} node(s) in common). Overlapping ports are not independent \
                 within a step, so each one's solve uses a pressure that never occurred and the \
                 energy ledgers stop matching. Note the acoustic source is up to one air cell \
                 LARGER than the surface itself (bilinear spreads outboard), so footprints that \
                 merely look separate can still collide.",
                index_repr(*node)
            ),
            PortError::SurfaceNotReady { where_, n } => write!(
                f,
                "the surface port on {where_} was asked to solve twice within one room step \
                 (room.n = {n}). A port does not step its room -- the caller does, once, after \
                 every port has solved:  for inst in instruments: inst.step(...)  then  \
                 room.step(). Without it the room is frozen and the surface is loaded by a stale \
                 field, silently."
            ),
            PortError::UnknownFace(face) => write!(
                f,
                "unknown face '{face}'; expected one of ('x0', 'x1', 'y0', 'y1', 'z0', 'z1')."
            ),
            PortError::InteriorIndexOutOfRange {
                plane,
                index,
                n_face,
            } => write!(
                f,
                "interior surface index {index} on plane '{plane}' is out of range 1..{} (the \
                 room has {n_face} face(s) there). The two node planes straddling the surface \
                 must BOTH be strictly interior: a node plane on a wall carries half the node \
                 weight W and the wall's admittance in beta, so R_j would differ between the two \
                 sides and the load would stop being 2 T^T R T with a single R.",
                *n_face as i64 - 2
            ),
            PortError::QLength {
                expected,
                got,
                per_face,
                node_count,
            } => {
                if *per_face {
                    write!(
                        f,
                        "q must be the per-FACE volume-velocity vector, shape ({expected},), got \
                         ({got},). Note this is HALF the node count ({node_count}): the two node \
                         planes share one q, with opposite signs."
                    )
                } else {
                    write!(
                        f,
                        "q must be the per-node volume-velocity vector, shape ({expected},), got \
                         ({got},). (Pass q = port.T @ v, not the scalar sum -- the scalar is \
                         exactly what the lumped tier would have coupled through, i.e. the \
                         negative control.)"
                    )
                }
            }
            PortError::Cut(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for PortError {}

/// `str()` of a Python tuple of three indices — the shape every port message quotes.
fn index_repr(i: [usize; 3]) -> String {
    format!("({}, {}, {})", i[0], i[1], i[2])
}

/// A lumped two-way terminal between a body and a room: one node, or a staircased ball of them.
///
/// The reference (`airbox.py`'s `RoomPort`) is a Python object the room holds a reference to, and
/// which reaches back into the room to append its injection and to read the open-circuit pressure.
/// Rust cannot have that cycle, and the retirement's second batch chose the shape the rest of the
/// tier will follow: **the port is a value the caller owns and the room is passed at each call.**
///
/// Two invariants the reference kept on the room survive the change:
///
/// * **Disjointness.** Two ports may not share a node — overlapping ports are not independent
///   within a step. [`RoomPort::new`] takes `&mut AirBox` and records its footprint in
///   [`crate::airbox::AirBox::claims`], which is `room._ports` reduced to what the refusal reads.
/// * **Unsticking.** `AirBox::set_state` cannot write `_queued_at = -1` into ports it does not
///   hold, so it bumps `AirBox::epoch` and the port compares against it. See
///   [`RoomPort::require_ready`].
///
/// A port is a set of grid nodes with normalized volume weights `w_n` (summing to 1). It answers
/// exactly one question, and answers it as a *linear scalar relation*: if a volume velocity `q` is
/// injected here this step, what centered pressure `pbar` will the injector feel?
///
/// ```text
/// pbar = pbar_free + R_room q
/// ```
///
/// [`RoomPort::free_pressure`] is `pbar_free`, the open-circuit term — what the port would feel
/// with `q = 0`, carrying the room's entire history *including every reflection off every wall*.
/// [`RoomPort::r_room`] is the constant internal resistance. Together they are everything a coupled
/// solver needs, which is why the body's load collapses to one division
/// ([`crate::airbox_wrap::RoomLoadedBody`]) instead of a field solve.
///
/// **Why it is only two numbers.** Within a single step an injection changes the pressure at its
/// own nodes and nowhere else: [`AirBox::step`] adds the source term node-locally to `p^{n+1}`, and
/// propagation to the neighbours waits for the *next* momentum sub-step. So the room is
/// linear-in-`q` with a diagonal instantaneous response, and
///
/// ```text
/// R_room = sum_n w_n^2 k rho0 c0^2 / (2 W_n (1 + beta_n))
/// ```
///
/// where the `2` is the centered (trapezoidal) pressure, `W_n` the tensor-trapezoid node weight,
/// and `(1 + beta_n)` the **wall closure denominator**. That last factor is easy to omit — the
/// naive `k rho0 c0^2 / (2 W)` is right at every interior node — and it matters as soon as a port
/// is mounted on a lossy wall: `step()` injects *before* it closes the wall, so the injection is
/// divided by `1 + beta` along with everything else, and `R_room` must say so.
///
/// **What omitting it costs is NOT an energy leak, and this repo's primary bug detector cannot see
/// it.** Corrected on measurement (the area-coupling batch §6.1,
/// `docs/dev/air-box-area-coupling-plan.md`): the body's energy identity telescopes to `-k pbar U`
/// for *whatever* `pbar` was used in the force, and the room's identity is exact for *whatever*
/// injection it received, so the scene total is the sum of two separately-exact identities and
/// stays flat even when the pressure the body was pushed by and the pressure the room developed are
/// different numbers. Measured here on batch 3's distributed port, whose `R_j` is this same
/// constant per node — the factor dropped on a lossy mounting wall (`zeta = 3`, mean `beta =
/// 0.173`), 300 steps: the scene total drifts **4.9e-15, which is SMALLER than the correct run's
/// 2.0e-14** — green either way, and not even in the suspicious direction — while `|radiated -
/// injected|` goes from **exactly 0.00e+00** to **18% of the channel** (12.9% on a free plate). A
/// wrong `R` does not leak energy; it creates it on one side and destroys the same amount on the
/// other. The tests that catch it are the *cross-ledger* identity `radiated == injected` and the
/// differential measurement (natively `tests/airbox_port.rs`'
/// `r_room_is_what_the_room_actually_does`), never the conserved total.
///
/// **Point port versus spread port — a measured non-convergence.** `radius: None` puts the whole
/// volume velocity on one node. That is exact and perfectly conservative, and its *magnitude* is a
/// property of the grid rather than of the physics: the port behaves as a pulsating sphere of
/// equivalent radius `~ h / 3.1` (measured `a_eff / h` = 0.324, 0.320, 0.317 on three grids), which
/// at low frequency is an added **mass** loading the body. Refining `h` therefore makes the
/// artifact *worse*, not better — it halves the equivalent radius and doubles the load (measured
/// ratios 0.493 and 0.496 across two halvings). A ball of fixed `radius` does not move: 1.045 and
/// 1.038 over the same halvings, a factor of twenty less grid-sensitive. **So use a point port for
/// structural tests, where only exactness matters, and a spread port whenever the magnitude of the
/// coupling is meant to mean something.** `radius` has no default so that choice is always made,
/// never inherited.
///
/// A spread port's absolute size is the **uniformly injecting ball**, whose equivalent shell radius
/// is `5a/6` — the classic **6/5** shape factor, the same one as the mean potential of a uniformly
/// charged sphere — and *not* a pulsating shell of radius `a`. Confirmed to 0.3% in a room large
/// enough for the port to be compact. Read that caveat as a real one: the same measurement in a
/// room only 10x the port reads 8.6% high, and that excess is the **room's** own reactance rather
/// than the port's (measured ratios to `5a/6` of 1.086, 1.040, 1.003, 0.977 for rooms of 0.5, 0.7,
/// 1.0 and 1.4 m). The number is insensitive to the Courant number to five significant figures, so
/// it is a static near-field quantity and not a dispersion artifact.
///
/// **Notes**
/// **An open face is refused, and the energy report cannot catch why.** An `"open"` (`Z = 0`) face
/// pins `p = 0`, so a port on it has `pbar_free = 0` and `R_room = 0`: the body radiates into a
/// short circuit. Measured over 400 steps, `injected` and `acoustic` are *exactly* zero and the
/// conserved total drifts 8.6e-15. The run is perfectly conservative and completely silent — the
/// physics is exactly right and exactly useless — so this project's primary bug detector is
/// structurally blind to it and a construction-time refusal is the only place to catch it.
///
/// **Overlap is refused for a reason that is also the reason N instruments work.** Two ports
/// sharing a node are not independent within a step: A's injection changes B's `pbar`, so B solved
/// against a pressure that never occurred and the two ledgers stop matching (measured 3.6e-2 drift,
/// against 7.1e-15 for disjoint ports in the same room). Disjointness is exactly the condition that
/// makes the cheap per-port scalar solve *exact*. A simultaneous `(I + G R) U` solve with the
/// cross-resistance matrix would handle overlap and is passive too, but it needs one central object
/// owning every port — which is precisely what makes a per-instrument
/// [`crate::connection::StringBodyBridge`] chain impossible. The hazard is real because of
/// **snapping**: two ports 5 mm apart on a 13.5 mm grid are one port.
///
/// **Do not mix [`AirBox::inject`] with ports** in a run whose energy you intend to assert on. That
/// source's work lands in the room's `injected` book with no port ledger to cancel it, so `sum_j
/// inst_j.energy() + room.energy()` is no longer the conserved total (it picks up the source's
/// contribution). Ports and the read-out source are both fine, just not in one ledger.
#[derive(Debug, Clone)]
pub struct RoomPort {
    index: [usize; 3],
    radius: Option<f64>,
    nodes: Nodes,
    flat: Vec<usize>,
    w: Vec<f64>,
    r_room: f64,
    /// The room step this port last queued at, and the epoch it was queued in.
    queued: Option<(u64, usize)>,
}

impl RoomPort {
    /// Place a port at `at` (m), covering one node or a ball of radius `radius` (m).
    ///
    /// The two refusals run in the reference's order — open faces before disjointness — because a
    /// port that is both gets the message it always got.
    pub fn new(
        room: &mut AirBox,
        at: [f64; 3],
        radius: Option<f64>,
    ) -> Result<RoomPort, PortError> {
        let index = match crate::airbox::node_index(at, room.p.h, room.p.n) {
            Some(i) => i,
            None => {
                return Err(PortError::OutsideRoom(
                    crate::airbox::ParamError::OutsideRoom {
                        point: at,
                        index: crate::airbox::node_index_raw(at, room.p.h),
                        l_actual: room.p.l_actual,
                        n: room.p.n,
                    },
                ))
            }
        };
        let view = room.view();
        let nodes: Nodes = match radius {
            None => [vec![index[0]], vec![index[1]], vec![index[2]]],
            Some(r) => {
                if r <= 0.0 || !r.is_finite() {
                    return Err(PortError::BadRadius(r));
                }
                let ball = ball_nodes(view.n, view.h, index, r);
                if ball[0].len() == 1 {
                    return Err(PortError::UnresolvableRadius {
                        radius: r,
                        h: view.h,
                        index,
                    });
                }
                ball
            }
        };

        let touched = touched_open_faces(&room.p, &nodes);
        if !touched.is_empty() {
            return Err(PortError::OnOpenFace {
                index,
                faces: touched,
            });
        }

        let shape = view.node_shape();
        let flat = ravel(&[&nodes[0], &nodes[1], &nodes[2]], shape);
        for claim in &room.claims {
            let (first, count) = shared_nodes(&flat, &claim.nodes);
            if let Some(first) = first {
                return Err(PortError::Overlapping {
                    index,
                    node: unravel(first, shape),
                    other: claim.label.clone(),
                    count,
                    h: view.h,
                });
            }
        }

        let cols = [&nodes[0][..], &nodes[1][..], &nodes[2][..]];
        let (w, big_w) = port_weights(&view, &cols);
        let r = r_room(&view, &cols, &w, &big_w);

        room.claims.push(crate::airbox::PortClaim {
            nodes: flat.clone(),
            label: index_repr(index),
        });
        Ok(RoomPort {
            index,
            radius,
            nodes,
            flat,
            w,
            r_room: r,
            queued: None,
        })
    }

    /// The centre node index.
    pub fn index(&self) -> [usize; 3] {
        self.index
    }

    /// The ball radius as asked for (m), or `None` for a point port.
    pub fn radius(&self) -> Option<f64> {
        self.radius
    }

    /// The port's node set, as three parallel index arrays.
    pub fn nodes(&self) -> &Nodes {
        &self.nodes
    }

    /// The port's nodes as flat C-order pressure indices.
    pub fn flat(&self) -> &[usize] {
        &self.flat
    }

    /// The per-node share of the volume velocity, `w = W / sum W`.
    pub fn w(&self) -> &[f64] {
        &self.w
    }

    /// The lumped internal resistance the body sees looking into the room (Pa s / m^3).
    pub fn r_room(&self) -> f64 {
        self.r_room
    }

    /// How many grid nodes the port actually covers, clipping at walls included.
    pub fn node_count(&self) -> usize {
        self.nodes[0].len()
    }

    /// The port's discrete volume `sum_n W_n` (m^3) — the staircased ball, made visible.
    ///
    /// Compare against `4 pi a^3 / 3`: the wobble as whole nodes fall in or out of the ball is the
    /// membrane batch's staircase and it is this port's accuracy floor (measured -4.6%, +1.5% on
    /// two grids), not a defect of the scheme.
    pub fn volume(&self, room: &AirBox) -> f64 {
        let view = room.view();
        let vals: Vec<f64> = self.flat.iter().map(|&f| view.node_w[f]).collect();
        reduce::sum(&vals)
    }

    /// The open-circuit centered pressure `pbar_free` this port would feel with `q = 0`.
    ///
    /// `O(port nodes)`: the divergence at a node needs only the six faces touching it, so this
    /// never touches the whole field. It replicates [`AirBox::step`]'s order exactly — divergence,
    /// **then** the wall closure — and must be read *before* `room.step()`, from the stored
    /// `u^{n+1/2}`.
    ///
    /// It deliberately ignores other ports' queued injections. For **disjoint** ports that is not
    /// an approximation: a queued injection at another node cannot reach this node within the step,
    /// so including it would change nothing — while for overlapping ports (refused at construction)
    /// reading it would make the solve *asymmetric* rather than merely wrong.
    pub fn free_pressure(&self, room: &AirBox) -> f64 {
        let view = room.view();
        let cols = [&self.nodes[0][..], &self.nodes[1][..], &self.nodes[2][..]];
        let pbar = free_pressure_nodes(&view, &cols);
        let terms: Vec<f64> = self.w.iter().zip(pbar.iter()).map(|(a, b)| a * b).collect();
        reduce::sum(&terms)
    }

    /// Refuse if this port's previous injection is still pending — i.e. no `room.step()`.
    ///
    /// The reference compares its `_queued_at` mark against `room.n` and relies on the room to
    /// clear the mark when it is restarted. Here the mark carries the room's epoch, so a
    /// `set_state` invalidates it without the room reaching in: a mark from epoch `e` says nothing
    /// about a room now in epoch `e + 1`, whatever its step count.
    pub fn require_ready(&self, room: &AirBox) -> Result<(), PortError> {
        if self.queued == Some((room.epoch, room.n)) {
            return Err(PortError::NotReady {
                index: self.index,
                n: room.n,
            });
        }
        Ok(())
    }

    /// Queue this port's volume velocity `q` (m^3/s) for the room's next `AirBox::step`.
    ///
    /// Node `n` receives the share `w_n q`. The room books the work at its own post-closure
    /// centered pressure — never at a number handed back from here, which is what keeps the two
    /// ledgers an independent check on each other rather than a tautology.
    pub fn inject(&mut self, room: &mut AirBox, q: f64) -> Result<(), PortError> {
        self.require_ready(room)?;
        room.pending_ports.push(crate::airbox::PortInjection {
            nodes: self.flat.clone(),
            w: self.w.clone(),
            q,
        });
        self.queued = Some((room.epoch, room.n));
        Ok(())
    }

    /// Forget any pending-injection mark — for reusing the port on a fresh run.
    pub fn reset(&mut self) {
        self.queued = None;
    }
}

/// Which pressure-release faces a node set touches, in [`FACES`] order and quoted.
///
/// A rigid or lossy room short-circuits: `_open` is all false and the reference's `np.any` is the
/// same early return.
///
/// An `open` (`Z = 0`) face pins `p = 0`, so `pbar_free` and every `R_j` are exactly zero: the
/// surface radiates into a short circuit, perfectly conservative and completely silent, and the
/// energy report is structurally blind to it. There is a **second, independent** reason here:
/// [`free_pressure_nodes`]' local read does not pin `p = 0` where [`AirBox::step`] does, so a patch
/// touching an open face is the one case where the local read stops being bit-identical to the
/// full-array closure (measured 4.3e-04, against exactly 0.0 everywhere else). Refusing makes that
/// identity hold by construction rather than by luck.
fn touched_open_faces(p: &crate::airbox::Params, nodes: &Nodes) -> Vec<String> {
    if !p.open.iter().any(|&o| o) {
        return Vec::new();
    }
    let mut touched = Vec::new();
    for (i, face) in FACES.iter().enumerate() {
        if p.walls[i] != 0.0 {
            continue;
        }
        let axis = AXES
            .iter()
            .position(|&c| c == face.as_bytes()[0] as char)
            .expect("a face name starts with an axis letter");
        let end = if face.as_bytes()[1] == b'0' {
            0
        } else {
            p.n[axis]
        };
        if nodes[axis].contains(&end) {
            touched.push(format!("'{face}'"));
        }
    }
    touched
}

// -- the distributed tier ------------------------------------------------------------------------

/// The `T` entries, unassembled — `_PatchPort._spread`.
///
/// Returns `(row, col, value)` in the plane's flat `i0 * (N1 + 1) + i1` indexing. Rows are `i64`
/// because a footprint that starts a hair below zero floors to `-1`, exactly as NumPy's
/// `np.floor(t).astype(np.intp)` does; `_check_in_plane_rim` refuses that before anything indexes
/// an array with it, so the negative never escapes.
///
/// Entries whose *geometric* weight is exactly zero are dropped; entries whose weight is nonzero
/// are kept even when the node's **area** is zero, so a zero-area surface still names the nodes it
/// covers and the `T = 0` reduction to the bare resonator stays exercisable.
pub fn spread(
    face_coords: &[[f64; 2]],
    areas: &[f64],
    h: f64,
    n_axis: [usize; 2],
    spreading: Spreading,
) -> (Vec<i64>, Vec<usize>, Vec<f64>) {
    let n_surface = face_coords.len();
    // Per in-plane direction: the stencil offsets and their weights, one entry per surface node.
    let mut stencil: Vec<Vec<(Vec<i64>, Vec<f64>)>> = Vec::with_capacity(2);
    for d in 0..2 {
        let t: Vec<f64> = face_coords.iter().map(|c| c[d] / h).collect();
        match spreading {
            Spreading::Nearest => {
                let i: Vec<i64> = t
                    .iter()
                    .map(|&x| (x.round_ties_even() as i64).clamp(0, n_axis[d] as i64))
                    .collect();
                stencil.push(vec![(i, vec![1.0; n_surface])]);
            }
            Spreading::Bilinear => {
                // floor, with the top edge folded down one cell so the stencil is always
                // {i0, i0+1} and the outboard node carries weight exactly 0 there.
                let i0: Vec<i64> = t
                    .iter()
                    .map(|&x| (x.floor() as i64).min(n_axis[d] as i64 - 1))
                    .collect();
                let f: Vec<f64> = t
                    .iter()
                    .zip(i0.iter())
                    .map(|(&x, &i)| x - i as f64)
                    .collect();
                let lo = (i0.clone(), f.iter().map(|&x| 1.0 - x).collect::<Vec<f64>>());
                let hi = (i0.iter().map(|&i| i + 1).collect::<Vec<i64>>(), f);
                stencil.push(vec![lo, hi]);
            }
        }
    }
    let n1 = n_axis[1] as i64;
    let (mut rows, mut cols, mut vals) = (Vec::new(), Vec::new(), Vec::new());
    for (a0, w0) in &stencil[0] {
        for (a1, w1) in &stencil[1] {
            for s in 0..n_surface {
                let w = w0[s] * w1[s];
                if w == 0.0 {
                    continue; // the `keep` mask: a geometric weight of exactly zero is not an entry
                }
                rows.push(a0[s] * (n1 + 1) + a1[s]);
                cols.push(s);
                vals.push(areas[s] * w);
            }
        }
    }
    (rows, cols, vals)
}

/// Unique in-plane node indices `(i0, i1)` of the spread stencil, plus the sorted rows.
///
/// The two divisions are Python's floor semantics, so a negative row splits the way NumPy's `//`
/// and `%` do rather than the way Rust's `/` and `%` do.
pub fn plane_nodes(rows: &[i64], n1: usize) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    let mut sorted: Vec<i64> = rows.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let d = n1 as i64 + 1;
    let i0 = sorted.iter().map(|&r| r.div_euclid(d)).collect();
    let i1 = sorted.iter().map(|&r| r.rem_euclid(d)).collect();
    (i0, i1, sorted)
}

/// Assemble `T` from the spread entries — `_PatchPort._build_T`.
///
/// `coo_matrix((vals, (pos, cols))).tocsr()` keeps explicit zeros and sums duplicates; there are no
/// duplicates by construction (each stencil corner is a distinct node), so the only thing that has
/// to be reproduced is that a stored `0.0` stays stored.
pub fn build_t(
    rows: &[i64],
    cols: &[usize],
    vals: &[f64],
    plane_nodes: &[i64],
    n_surface: usize,
) -> Csr {
    let mut by_row: Vec<Vec<(usize, f64)>> = vec![Vec::new(); plane_nodes.len()];
    for e in 0..rows.len() {
        let pos = plane_nodes.partition_point(|&x| x < rows[e]);
        by_row[pos].push((cols[e], vals[e]));
    }
    Csr::from_rows_keeping_zeros(plane_nodes.len(), n_surface, by_row)
}

/// The per-node resistance of a patch, `R = k rho0 c0^2 / (2 W (1 + beta))`.
///
/// One numerator, one division — and deliberately not spelled like [`r_room`], which folds the same
/// constants into the summand one multiply at a time. The two are different doubles and the
/// reference writes both.
pub fn patch_resistance(view: &RoomView<'_>, nodes: &[&[usize]; 3]) -> Vec<f64> {
    let num = (view.k * view.rho0) * scalar_pow(view.c0, 2.0);
    (0..nodes[0].len())
        .map(|m| {
            let flat = view.flat([nodes[0][m], nodes[1][m], nodes[2][m]]);
            num / ((2.0 * view.node_w[flat]) * (1.0 + view.beta[flat]))
        })
        .collect()
}

/// `scale * T^T diag(R) T`, in SciPy's own association and accumulation order.
///
/// The contraction runs over `k` (an air node) ascending, and each term is `(T_ki R_k) T_kj` —
/// the diagonal folded into the **left** factor, because `T.T @ diags(R) @ T` left-associates in
/// Python. Folding it right instead changes 2,028 of 6,845 entries across the suite's fixtures; see
/// the module docs for why `spreading="nearest"` cannot see the difference.
///
/// `scale` is `1.0` for a wall-mounted patch and `2.0` for an interior one (two faces, two
/// resistances). SciPy applies it as `2.0 * matrix`, i.e. to the assembled data, and `2.0 * x` is
/// exact, so where it is applied does not matter — unusually.
pub fn load_matrix(t: &Csr, r: &[f64], scale: f64) -> Csr {
    let n = t.ncols();
    // Columns of T in ascending row order — i.e. the rows of T^T, which is what the contraction
    // walks. Built by a counting sort so the ascending-k order is structural, not sorted for.
    let mut colptr = vec![0usize; n + 1];
    for &j in t.indices() {
        colptr[j + 1] += 1;
    }
    for j in 0..n {
        colptr[j + 1] += colptr[j];
    }
    let mut fill = colptr.clone();
    let mut col_k = vec![0usize; t.nnz()];
    let mut col_v = vec![0.0f64; t.nnz()];
    for k in 0..t.nrows() {
        for a in t.indptr()[k]..t.indptr()[k + 1] {
            let j = t.indices()[a];
            col_k[fill[j]] = k;
            col_v[fill[j]] = t.data()[a];
            fill[j] += 1;
        }
    }
    let mut rows: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);
    let mut acc = vec![0.0f64; n];
    let mut touched: Vec<usize> = Vec::new();
    let mut seen = vec![false; n];
    for i in 0..n {
        touched.clear();
        for c in colptr[i]..colptr[i + 1] {
            let k = col_k[c];
            let left = col_v[c] * r[k];
            for a in t.indptr()[k]..t.indptr()[k + 1] {
                let j = t.indices()[a];
                if !seen[j] {
                    seen[j] = true;
                    acc[j] = 0.0;
                    touched.push(j);
                }
                acc[j] += left * t.data()[a];
            }
        }
        touched.sort_unstable();
        let row: Vec<(usize, f64)> = touched.iter().map(|&j| (j, scale * acc[j])).collect();
        for &j in &touched {
            seen[j] = false;
        }
        rows.push(row);
    }
    // `from_rows`, which DROPS exact zeros -- and note that [`build_t`] two functions up uses the
    // constructor that keeps them. That is not an inconsistency, it is SciPy's: the two routines
    // this one expression calls disagree with each other. `coo_matrix(...).tocsr()` keeps a stored
    // `0.0`, and `csr_matmat` prunes one (it writes an entry only `if (sums[head] != 0)`). So on a
    // surface with zero-area nodes the reference's `T` has 182 stored entries and its load matrix
    // has 91 where a uniform treatment would give 208. No fixture the suite builds contains an
    // explicit zero, so nothing measures this -- §16.4's blind fixture, in the library rather than
    // in the model, and the parity file constructs a zero-area surface on purpose to catch it.
    Csr::from_rows(n, n, rows)
}

/// How many air nodes under the footprint no surface node reaches — `_check_footprint`'s count.
///
/// "Under the footprint" is **span-wise**, not a bounding box: per reached row, the columns between
/// that row's own first and last reached column; per reached column, the rows between its own first
/// and last; and their union. For a rectangle this reduces to the bounding box by construction,
/// and for a staircased disk it is the difference between refusing at every resolution and refusing
/// at none. Returns `(unfed, footprint size)`.
///
/// Counts air nodes **under** the footprint that no surface node reaches. With bilinear spreading
/// this is 0 at every grid ratio measured, so the refusal rarely fires — and it still ships,
/// because a surface coarse enough to skip whole air cells is a real thing to refuse. Note the
/// condition is a **count of unfed nodes**, not an inequality on `h_surface/h_air`: at ratio 0.909,
/// comfortably inside the naive inequality, nearest-node still left half the footprint unfed.
///
/// **"Under the footprint" is span-wise, not a bounding box** — which it was until the membrane
/// batch, and which quietly assumed every surface is a *rectangle*. The required set is now: per
/// reached air-node row, the columns between that row's own first and last reached column; per
/// reached column, the rows between its own first and last; and their **union**. For a rectangle
/// every row spans the same columns and every column the same rows, so this reduces to the bounding
/// box **by construction** — batches 3 and 4 are unmoved.
///
/// The new set is *contained* in the old one's outer bound and *contains* the old one itself (the
/// coordinate box is inset by up to one node per side), so it refuses everything the box refused
/// and counts a few more nodes when it does — measured 20 against 17 at `h_surface/h_air = 2.02`.
/// **The verdict boundary does not move**: swept over a rectangle at 1.01, 1.21, 1.35, 1.52, 1.73,
/// 2.02, 2.42, 3.03, both criteria change their mind between the same two spacings. Where that
/// boundary *sits* depends on alignment and not only on the ratio — a differently-placed patch put
/// it in (2.02, 2.24] — which is the same reason this is a count of unfed nodes and not an
/// inequality on `h_surface/h_air`.
///
/// The bounding box was not a conservative approximation, it was the **wrong set**. A staircased
/// *disk* — model #4's circular domain, i.e. a drumhead — has bbox corners about `0.41 R` from the
/// nearest live node, so they are unreachable by construction and the refusal fired at *every*
/// resolution, getting worse under refinement: 20, 48 and 40 unfed nodes at `N = 40, 56, 72`, on
/// both port tiers. Those corners are not under the surface at all. Span-wise, the same disks leave
/// **zero** unfed at `N = 24 … 96`.
///
/// What this does still refuse, and honestly so, is a surface with an interior *hole* — an annulus
/// spans its own gap row-wise. No resonator in this repo has one; if one arrives, this is the line
/// that has to learn about it, and the refusal names the shape rather than hiding it.
pub fn footprint_unfed(i0: &[usize], i1: &[usize], n1: usize) -> (usize, usize) {
    let stride = n1 + 1;
    let mut spans: Vec<usize> = Vec::new();
    let mut keys: Vec<usize> = i0.to_vec();
    keys.sort_unstable();
    keys.dedup();
    for &key in &keys {
        let (lo, hi) = min_max(i1, i0, key);
        for c in lo..=hi {
            spans.push(key * stride + c);
        }
    }
    let mut keys: Vec<usize> = i1.to_vec();
    keys.sort_unstable();
    keys.dedup();
    for &key in &keys {
        let (lo, hi) = min_max(i0, i1, key);
        for r in lo..=hi {
            spans.push(r * stride + key);
        }
    }
    spans.sort_unstable();
    spans.dedup();
    let mut reached: Vec<usize> = (0..i0.len()).map(|m| i0[m] * stride + i1[m]).collect();
    reached.sort_unstable();
    reached.dedup();
    let unfed = spans
        .iter()
        .filter(|s| reached.binary_search(s).is_err())
        .count();
    (unfed, spans.len())
}

/// `(min, max)` of `values` over the positions where `keys` equals `key`.
fn min_max(values: &[usize], keys: &[usize], key: usize) -> (usize, usize) {
    let mut lo = usize::MAX;
    let mut hi = 0usize;
    for m in 0..keys.len() {
        if keys[m] == key {
            lo = lo.min(values[m]);
            hi = hi.max(values[m]);
        }
    }
    (lo, hi)
}

/// Flat C-order offsets of a node set within the pressure array — `np.ravel_multi_index`.
pub fn ravel(nodes: &[&[usize]; 3], shape: [usize; 3]) -> Vec<usize> {
    (0..nodes[0].len())
        .map(|m| (nodes[0][m] * shape[1] + nodes[1][m]) * shape[2] + nodes[2][m])
        .collect()
}

/// The first flat offset two sorted-or-not node sets share, and how many they share.
///
/// `np.intersect1d` sorts both sides and returns the shared values sorted, so "the first shared
/// one" is the smallest — which is what the refusal message names.
pub fn shared_nodes(a: &[usize], b: &[usize]) -> (Option<usize>, usize) {
    let mut sb: Vec<usize> = b.to_vec();
    sb.sort_unstable();
    sb.dedup();
    let mut sa: Vec<usize> = a.to_vec();
    sa.sort_unstable();
    sa.dedup();
    let shared: Vec<usize> = sa
        .into_iter()
        .filter(|x| sb.binary_search(x).is_ok())
        .collect();
    (shared.first().copied(), shared.len())
}

/// Undo a flat C-order offset — `np.unravel_index`, for the refusal messages that name a node.
pub fn unravel(flat: usize, shape: [usize; 3]) -> [usize; 3] {
    let i2 = flat % shape[2];
    let rest = flat / shape[2];
    [rest / shape[1], rest % shape[1], i2]
}

/// Format a float the way Python's `repr` does — re-exported so the binding's messages match.
pub fn repr_float(x: f64) -> String {
    py_float(x)
}

// -- the distributed tier, as values --------------------------------------------------------------

/// Everything `_PatchPort` holds, shared by the two distributed ports.
///
/// The reference's base class exists to hold code; here it holds state and the code is the free
/// functions above plus the two `impl` blocks below. The four attributes the retired Python suite
/// *wrote* on a port — `T`, `R`, `load_matrix` and `areas` — have setters, because the tests that
/// switch a coupling off or halve it do it by replacing one of them, not by rebuilding the port.
#[derive(Debug, Clone)]
struct Patch {
    spreading: Spreading,
    in_plane_axes: (usize, usize),
    coords: Vec<[f64; 2]>,
    areas: Vec<f64>,
    origin: (f64, f64),
    face_coords: Vec<[f64; 2]>,
    nodes: Nodes,
    index: [usize; 3],
    flat: Vec<usize>,
    t: Csr,
    r: Vec<f64>,
    load: Csr,
    footprint_empty: usize,
    where_: String,
    /// The room step this port last queued at, and the epoch it was queued in — see
    /// [`RoomPort::require_ready`] for why the epoch is half of the mark.
    queued: Option<(u64, usize)>,
}

/// The placed footprint and the origin it was placed at — what [`accept_surface`] hands back.
type Placed = (Vec<[f64; 2]>, (f64, f64));

/// Validate the surface, place its footprint in the plane, and return the placed coordinates.
///
/// The reference's `_accept_surface` also refuses a `coords` that is not `(n_surface, 2)` and an
/// `origin` that is not a pair. Both are claims about the shape of a Python argument and neither
/// has a variant here: `coords: &[[f64; 2]]` and `origin: Option<(f64, f64)>` are the same claims
/// made by the compiler.
fn accept_surface(
    n: [usize; 3],
    h: f64,
    coords: &[[f64; 2]],
    areas: &[f64],
    origin: Option<(f64, f64)>,
    in_plane_axes: (usize, usize),
    where_: &str,
) -> Result<Placed, PortError> {
    if areas.len() != coords.len() {
        return Err(PortError::AreasLength {
            expected: coords.len(),
            got: areas.len(),
        });
    }
    if areas.iter().any(|&a| a < 0.0 || !a.is_finite()) {
        return Err(PortError::BadAreas);
    }
    let (t0, t1) = in_plane_axes;
    let extent = [n[t0] as f64 * h, n[t1] as f64 * h];

    let origin = match origin {
        Some(o) => o,
        None => {
            // Centred: the footprint's midpoint lands on the plane's midpoint, so the grid's own
            // mirror maps the surface to itself. Not an aesthetic default — it is what makes the
            // load equivariant and what lets the scene be symmetric.
            let mut out = [0.0f64; 2];
            for (d, slot) in out.iter_mut().enumerate() {
                let mut lo = f64::INFINITY;
                let mut hi = f64::NEG_INFINITY;
                for c in coords {
                    lo = lo.min(c[d]);
                    hi = hi.max(c[d]);
                }
                *slot = 0.5 * (extent[d] - (lo + hi));
            }
            (out[0], out[1])
        }
    };

    let face_coords: Vec<[f64; 2]> = coords
        .iter()
        .map(|c| [c[0] + origin.0, c[1] + origin.1])
        .collect();
    let tol = 1e-9 * h;
    let axes = [t0, t1];
    for d in 0..2 {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for c in &face_coords {
            lo = lo.min(c[d]);
            hi = hi.max(c[d]);
        }
        if lo < -tol || hi > extent[d] + tol {
            return Err(PortError::FootprintOutside {
                where_: where_.to_owned(),
                lo,
                hi,
                axis: axes[d],
                extent: extent[d],
                origin: if d == 0 { origin.0 } else { origin.1 },
            });
        }
    }
    Ok((face_coords, origin))
}

/// The shared parts of both distributed constructors, from the spread entries to the load matrix.
struct Assembled {
    nodes: Nodes,
    index: [usize; 3],
    flat: Vec<usize>,
    t: Csr,
    r: Vec<f64>,
    load: Csr,
    in_plane: (Vec<i64>, Vec<i64>),
}

#[allow(clippy::too_many_arguments)]
fn assemble(
    view: &RoomView<'_>,
    face_coords: &[[f64; 2]],
    areas: &[f64],
    spreading: Spreading,
    axes: (usize, usize),
    where_: &str,
    place: &dyn Fn(&[i64], &[i64]) -> Nodes,
    r_from: &dyn Fn(&Nodes) -> Nodes,
    scale: f64,
) -> Result<Assembled, PortError> {
    let (t0, t1) = axes;
    let (rows, cols, vals) = spread(
        face_coords,
        areas,
        view.h,
        [view.n[t0], view.n[t1]],
        spreading,
    );
    let (i0, i1, plane) = plane_nodes(&rows, view.n[t1]);
    // The rim refusal runs before anything indexes an array with these, which is what keeps the
    // `-1` a floor can produce from ever escaping into a node index.
    for (idx, ax) in [(&i0, t0), (&i1, t1)] {
        let lo = *idx.iter().min().unwrap_or(&0);
        let hi = *idx.iter().max().unwrap_or(&0);
        if lo < 1 || hi > view.n[ax] as i64 - 1 {
            return Err(PortError::InPlaneRim {
                where_: where_.to_owned(),
                lo,
                hi,
                axis: ax,
                n_axis: view.n[ax],
            });
        }
    }
    let t = build_t(&rows, &cols, &vals, &plane, face_coords.len());

    let nodes = place(&i0, &i1);
    let index = [nodes[0][0], nodes[1][0], nodes[2][0]];
    let flat = ravel(&[&nodes[0], &nodes[1], &nodes[2]], view.node_shape());
    let r_nodes = r_from(&nodes);
    let r = patch_resistance(view, &[&r_nodes[0], &r_nodes[1], &r_nodes[2]]);
    let load = load_matrix(&t, &r, scale);
    Ok(Assembled {
        nodes,
        index,
        flat,
        t,
        r,
        load,
        in_plane: (i0, i1),
    })
}

/// Every accessor the two distributed ports share, generated once against their `patch` field.
macro_rules! patch_accessors {
    ($ty:ty) => {
        impl $ty {
            /// How the surface's node areas are distributed over the air nodes under it.
            pub fn spreading(&self) -> Spreading {
                self.patch.spreading
            }

            /// The plane's two in-plane axes, in increasing order.
            pub fn in_plane_axes(&self) -> (usize, usize) {
                self.patch.in_plane_axes
            }

            /// The surface's own in-plane node positions (m), as passed.
            pub fn coords(&self) -> &[[f64; 2]] {
                &self.patch.coords
            }

            /// The per-node areas (m^2).
            pub fn areas(&self) -> &[f64] {
                &self.patch.areas
            }

            /// Replace the per-node areas — how a test switches a surface's coupling off.
            pub fn set_areas(&mut self, areas: Vec<f64>) {
                self.patch.areas = areas;
            }

            /// How many surface nodes the port carries.
            pub fn n_surface(&self) -> usize {
                self.patch.coords.len()
            }

            /// Where the footprint was placed in the plane's own axes (m).
            pub fn origin(&self) -> (f64, f64) {
                self.patch.origin
            }

            /// The air nodes the source covers, as three parallel index arrays.
            pub fn nodes(&self) -> &Nodes {
                &self.patch.nodes
            }

            /// The first node of the set, which is how the port names itself in a refusal.
            pub fn index(&self) -> [usize; 3] {
                self.patch.index
            }

            /// The node set as flat C-order pressure indices.
            pub fn flat(&self) -> &[usize] {
                &self.patch.flat
            }

            /// The footprint's placed coordinates (m) — `coords + origin`.
            pub fn face_coords(&self) -> &[[f64; 2]] {
                &self.patch.face_coords
            }

            /// How the port names its mounting in a refusal — the reference's `_where`.
            pub fn where_(&self) -> &str {
                &self.patch.where_
            }

            /// How many air nodes under the footprint no surface node reaches — reported, and zero
            /// on any port that was accepted.
            pub fn footprint_empty(&self) -> usize {
                self.patch.footprint_empty
            }

            /// The spreading operator `T`: air nodes by surface nodes, carrying areas.
            pub fn t(&self) -> &Csr {
                &self.patch.t
            }

            /// Replace `T` — the reference allows it and one retired test rebuilt it rescaled.
            pub fn set_t(&mut self, t: Csr) {
                self.patch.t = t;
            }

            /// The per-air-node resistance `R_j` (Pa s / m^3).
            pub fn r(&self) -> &[f64] {
                &self.patch.r
            }

            /// Replace `R` — a wrong one is exactly what the conserved total cannot see.
            pub fn set_r(&mut self, r: Vec<f64>) {
                self.patch.r = r;
            }

            /// The surface-side load `scale * T^T diag(R) T` (Pa s / m^3 per node pair).
            pub fn load_matrix(&self) -> &Csr {
                &self.patch.load
            }

            /// Replace the load matrix.
            pub fn set_load_matrix(&mut self, load: Csr) {
                self.patch.load = load;
            }

            /// How many air nodes the surface's spread source actually covers.
            pub fn node_count(&self) -> usize {
                self.patch.nodes[0].len()
            }

            /// The **radiating** area `sum_n area_n` (m^2) — which is not the bounding rectangle.
            ///
            /// Reads `areas` live rather than from a cache, so zeroing it works.
            ///
            /// A simply-supported plate's rim nodes are dead: they do not move, so they displace no
            /// volume, and the total comes out at exactly `((N-1)/N)^2 Lx Ly` (measured 0.5625,
            /// 0.7656, 0.8403, 0.8789, 0.9184 at `N = 4, 8, 12, 16, 24`). That shortfall is
            /// **physics, not a defect** — but it means any comparison against a closed form for a
            /// piston of area `Lx Ly` is wrong by that factor at coarse `N`, so the number is
            /// reported ([`RoomPort::volume`]'s precedent). A free plate has no dead rim and comes
            /// out at exactly `Lx Ly`.
            pub fn net_area(&self) -> f64 {
                reduce::sum(&self.patch.areas)
            }

            /// Refuse if this port's previous injection is still pending — i.e. no `room.step()`.
            pub fn require_ready(&self, room: &AirBox) -> Result<(), PortError> {
                if self.patch.queued == Some((room.epoch, room.n)) {
                    return Err(PortError::SurfaceNotReady {
                        where_: self.patch.where_.clone(),
                        n: room.n,
                    });
                }
                Ok(())
            }

            /// Forget any pending-injection mark — for reusing the port on a fresh run.
            pub fn reset(&mut self) {
                self.patch.queued = None;
            }
        }
    };
}

/// A surface mounted flush in one of the room's walls, radiating from every node.
///
/// The distributed counterpart of [`RoomPort`]: where the lumped tier couples one scalar volume
/// velocity through one internal resistance, this couples a **vector** of them through
/// `T^T diag(R) T`, so a mode that moves the same net volume as another can still radiate a
/// completely different field. Built as a value the caller owns, with the room passed at each
/// call — the shape the retirement's second batch chose and this tier inherits.
///
/// [`RoomPort`] is a *one-port*: it couples through exactly one scalar, the net volume velocity
/// `U = sum_i a_i q_i'`, and hands back exactly one scalar pressure. That is the right shape for a
/// [`crate::body::ModalBody`], whose radiation weights *are* its surface collapsed to a point. It
/// is the wrong shape for a **grid** resonator, which has a real surface with a real shape on it —
/// and the difference is not a refinement but a whole physical effect that every lumped tier in
/// this repo predicts to be **exactly zero**:
///
/// ```text
/// a surface radiates according to the **shape** of its motion, not only its net volume
/// displacement — so a mode with zero net volume velocity, which every one-port here calls
/// silent, is not.
/// ```
///
/// That is the **acoustic short circuit**. An even-index mode of a simply supported plate —
/// `(2,1)`, `(2,2)`, `(4,2)` — has `+` and `-` regions displacing *exactly* cancelling volumes
/// (`sum_i sin(m pi i/N) = 0` identically for even `m`, so `U = 0` to rounding and not merely
/// small). What actually happens is that each patch of surface pushes on the air *locally*, and the
/// cancellation is only as complete as the acoustic wavelength's ability to bridge a `+` region and
/// its `-` neighbour. Below coincidence the cancellation is nearly complete and the radiation comes
/// from the edges; above it the surface pattern is supersonic and radiates freely. A one-port has
/// no length scale on its surface, so no `R(omega)` reproduces this at any order — the same
/// structural argument as batch 2's delayed echo, one tier further in.
///
/// **The room's instantaneous response is diagonal, so the load is `T^T R T`.** Batch 2 read "an
/// injection changes the pressure at its own node and nowhere else within a step" as a Thevenin
/// source at *a* node. The stronger reading is that the room's instantaneous response over *any*
/// set of nodes is **diagonal** — no cross-resistance at any separation, including one cell
/// (measured: off-diagonal exactly `0.00e+00` at interior, lossy-wall and lossy-edge patches
/// alike). So with
///
/// * `T` the **volume-weight matrix** `(n_patch x n_surface)`, `q = T v` in m^3/s,
/// * `R = diag(R_j)`, `R_j = k rho0 c0^2 / (2 W_j (1 + beta_j))` — [`RoomPort::r_room`]'s constant
///   per node, wall-closure denominator included,
/// * `pbar = pbar_free + R q` a **vector** Thevenin relation,
///
/// reciprocity carries the pressure back as `f = -T^T pbar` (the same structural fact
/// [`crate::radiation::RadiatedBody`] leans on when it reuses its read-out weights as coupling
/// weights), and the unknown part of the load is `-(1/2k) T^T R T u^{n+1}`: a constant, symmetric,
/// positive-semidefinite, sparse matrix times the unknown. It **folds into the resonator's own
/// factorization** — nothing new is solved (see [`crate::airbox_wrap::RoomGrid`]). The family's
/// ladder, complete: [`crate::radiation::RadiatedBody`] scalar `R` -> one division;
/// [`crate::radiation::RationalAirLoad`] `R_eff` + one aux state -> one division; [`RoomPort`]
/// scalar `R_room` with a rich `pbar_free` -> one division; **this** diagonal `R` with a matrix `T`
/// -> absorbed into the resonator's LU. Batch 2 *is* the rank-1 case: put every surface node on one
/// air node and `T` becomes `w 1^T`, `T^T R T` the rank-1 `R_room w w^T`, and Sherman-Morrison is
/// what inverting `A + rank-1` gives instead of refactoring. Passivity is unconditional and for a
/// better reason than before — `T^T R T` is PSD because `R_j >= 0`, so the loaded matrix is SPD for
/// every timestep, every grid, every patch, and can never be singular.
///
/// **The sign convention is part of the API, and it is local.** Positive surface displacement is
/// along the room's **inward normal at** [`SurfacePort::face`] — `+z` for `z0`, `-z` for `z1`, and
/// so on — so an outward-moving surface *compresses* the air at its own face. Hence `T` is
/// entrywise non-negative on all six walls and **no inward normal appears in the code at all**.
/// This is not cosmetic: a consistent sign flip in `T`/`T^T` is invisible to every energetic
/// quantity in the batch — `T^T R T` is sign-invariant, so the load matrix, the solve and
/// `radiated_energy` all come out **bit-identical** while the room's field is inverted (measured).
/// The alternative convention, "positive displacement along the global axis", needs an explicit
/// per-face sign, and getting it wrong on three of the six faces is a perfectly passive, perfectly
/// conservative, perfectly green simulation of a plate pushing the wrong way. The local convention
/// disarms the trap instead of testing around it. (Its only detector is the sign of the surface
/// pressure on the **first** step: a six-step read gave the *wrong* answer in the prototype because
/// the plate's own half period was five steps.)
///
/// **Bilinear spreading, and the argument that decides it is COVERAGE — the symmetry argument does
/// not survive measurement.** `T` must (a) conserve volume exactly, (b) be usable as `T^T` for the
/// pressure, and (c) leave no air node in the footprint unfed. Nearest-node assignment satisfies
/// (a) and (b) and fails (c): at `h_surface/h_air = 1.21` it leaves footprint nodes unfed, which
/// this class refuses. Where it does construct, its interior assigned area per air node is lumpy at
/// the grid scale and **does not converge** — measured spread 0.83, 1.03, 0.64, 0.46 `h_air^2` at
/// `N_plate = 8, 16, 24, 32`, wandering rather than shrinking. Bilinear spreading (each surface
/// node's area distributed over the four surrounding air nodes with weights summing to 1) leaves
/// **0** unfed at every ratio and is 10x to 100x flatter at every one: 0.082, 0.062, 0.051, 0.031
/// over the same refinements, decreasing. That is what ships, and [`Spreading::Nearest`] exists
/// **only** as its measured negative control.
///
/// **How flat, exactly — because "partition of unity" promises more than it delivers.** Bilinear's
/// interior assignment is `h_air^2` *exactly* (measured 5e-16 … 2e-15) when `h_air/h_surface` is an
/// **integer**, and only then. Off an integer it ripples, and not monotonically in the ratio: 2.93
/// gives 0.0077 while the finer 4.40 gives 0.0207. Poisson summation says why — the periodised hat
/// sum has Fourier coefficients `sinc^2(pi k h_air/h_surface)`, whose `k`-th term vanishes exactly
/// when `k h_air/h_surface` is a nonzero integer, so which harmonics cancel depends on the ratio's
/// arithmetic rather than on its size. The residual is a small ripple in the source's *amplitude*
/// across the surface, it shrinks under refinement, and it is the accuracy floor of this coupling —
/// worth knowing before reading a radiated magnitude as physics.
///
/// **What was expected to decide it, and what was actually found.** The plan for this batch argued
/// from *reflection equivariance* — that bilinear's load matrix is mirror-symmetric at every offset
/// while nearest-node's is not. Measured here (two independent routes: the defect of `T^T R T`
/// under the surface's own mirror permutation, and the defect of `T` itself under that mirror
/// composed with the air grid's), **bilinear's equivariance is not offset-independent.** It holds
/// exactly when `S = 2 (surface centre) / h_air` is an **integer** — defect 1.0e-15 there — and
/// fails smoothly otherwise, 1.6e-01 … 3.8e-01 across sixteen offsets spanning one air cell,
/// peaking at the half-cell offset. The algebra says the same thing: the surface's mirror sends
/// node `i` to a cell fraction `frac(S - t_i)`, which equals the `1 - f_i` that reverses a bilinear
/// weight pair *only* for integral `S`. Nearest-node's equivariance is patchier still, and the way
/// it fails is the tell: it is exact at an **even** `S` (measured `0.00e+00`) and breaks at an
/// **odd** one (5.2e-01), because there the surface's own centre node lands exactly on a rounding
/// tie, which round-half-to-even resolves the same way from both directions. So nearest-node's
/// symmetry is an accident of alignment and of the rounding rule; bilinear's is a property of the
/// geometry.
///
/// **Two consequences, and neither weakens the design.** First, centring is load-bearing for *two*
/// independent reasons rather than one — it is what makes `S` integral and the load matrix
/// equivariant at all, on top of the scene-symmetry argument below. Second, the coupled monopole
/// leak does **not** discriminate the two operators: with the surface centred, an even plate mode's
/// `|U|/A` stays at rounding (1.3e-14 … 3.3e-13 over 200 steps) under *both* spreadings at
/// `N_plate = 8, 16, 24`. What breaks the zero is the **scene**, not the spreading operator —
/// measured 2.3e-01 for a plate off-centre by `h_air/3` and 7.4e-02 for a centred plate in a room
/// made asymmetric in the mode's own axis, against 7.2e-14 centred and symmetric.
///
/// **The zero is a property of the whole SCENE, not of the surface or the spreading operator.**
/// Equivariance of the load is necessary and *not* sufficient, because the load is only half the
/// coupling: the incoming `T^T pbar_free` is the **room's** field, so an antisymmetric surface mode
/// stays silent only if the room is mirror-symmetric about that mode's own antisymmetry plane. A
/// **perfectly centred** plate in a room made asymmetric *in x* (lossy `x0`, rigid `x1`) leaks
/// 3.5e-02 — the largest figure in the study — while the same asymmetry in *y* leaves a `(2,1)`
/// mode at 3.0e-15. Six different room widths with the plate centred, spanning two grid alignments,
/// all sit at rounding: **grid commensurability is not the criterion, centring is.** And there is
/// no tolerance band — the leak is *linear* in the offset over four decades (`delta/h_air = 1e-6`
/// gives 1.0e-07), so "approximately centred" is not approximately silent. Hence an omitted origin
/// (`None`) **is centred in the face**. The asymmetric case is not a defect to hide: a room that
/// re-excites a surface's *shape*, converting an acoustically silent mode into a radiating one at
/// the 1-3% level, is one more thing no one-port can represent at all.
///
/// **A baffled surface needs no new boundary machinery, and that is algebra rather than a
/// measurement.** A soft injection of `q = A_n v` at a **wall** node *is* the moving-wall (piston)
/// condition, not an approximation of it: the wall node's divergence is `u_face/w_z` with `w_z =
/// h/2`, so a wall moving at `v` adds `k rho0 c0^2 v/(h/2)`, while a soft source `q = h^2 v` into
/// `W = h^3/2` adds `k rho0 c0^2 (h^2 v)/(h^3/2)` — and `h^2/(h^3/2) = 1/(h/2)`, so these are the
/// same number identically, on every wall type. Note the two expressions divide by `W` and by `w_z`
/// in different orders, so an implementation written the other way would agree to machine precision
/// and **not** bit-for-bit; this repo's bit-identity claims are load-bearing and that one could not
/// be cashed.
///
/// **Notes**
/// **Why the rim of the face is refused rather than clipped.** Clipping the stencil at the face
/// boundary (`np.clip(i0 + di, 0, N)`) folds a node's outboard weight back onto the boundary node.
/// The weights still sum to 1, so volume is still conserved and **every ledger stays green while
/// the source geometry is quietly wrong** — the same failure shape as the sign flip above.
/// [`crate::airbox::node_index`] already refuses to relocate an out-of-room point rather than
/// snapping it; this matches. The refusal is stated on the *stencil*: every air node that receives
/// weight must be strictly interior to the face. That has a second, load-bearing consequence — a
/// node on the face's rim also touches a second wall, so it carries half the node weight `W` and
/// the **sum** of two wall admittances in `beta`. `R_j` would stop being uniform across the patch,
/// and with it the `T^T R T = R (T^T T)` identity the offset-independent equivariance rests on. The
/// refusal is therefore what keeps the symmetry argument true, not merely what keeps the geometry
/// tidy. It also means a legal patch can only ever touch its own mounting face.
///
/// **The acoustic source is up to one air cell larger than the surface**, because bilinear
/// legitimately spreads onto the nodes just outboard of the surface's rectangle. The disjointness
/// check therefore uses the **actual node set**, never the footprint.
///
/// **A lossy mounting wall is a test configuration, not a recommended one.** Patch nodes keep the
/// mounting wall's impedance: [`AirBox::step`] divides the injection by `1 + beta` whether one
/// likes it or not, so `R_j` must say so, and the result is physically a piston in an *absorbing*
/// baffle — consistent, passive, and the only way to pin the `(1 + beta)` factor (which the
/// radiation leg left unpinned twice in its history). But such a wall absorbs **through the
/// surface's own footprint**, i.e. it models a plate porous to the wall behind it. A **rigid**
/// mounting wall is the true infinite baffle and is the sane default.
#[derive(Debug, Clone)]
pub struct SurfacePort {
    patch: Patch,
    face: String,
    axis: usize,
}

/// A surface hanging on an interior plane of faces, radiating from **both** sides.
///
/// The same patch, mounted on a velocity plane rather than a wall: it blocks the faces it occupies
/// (a rigid, zero-thickness partition registered with the room) and injects `-q` on the low node
/// plane and `+q` on the high one, so the two sides are one object and the dipole is exact.
///
/// [`SurfacePort`] mounts a surface flush in a wall. The wall then does the rest: it is the
/// textbook infinite baffle, the surface's back face is unloaded, and the surface is — for all the
/// room can tell — a **source**, a patch of wall that moves. This class takes the wall away. The
/// surface hangs *in* the room, radiates from **both** faces, and is driven by the pressure
/// **jump** across it.
///
/// That is not batch 3 with a sign. A source adds sound to a room; an **object** also *removes*
/// paths through it, and does so whether or not it is moving. Every instrument in the free-plate
/// family — the suspended cymbal of model #5b, the gong of #6 — is physically an object hanging in
/// air, so [`crate::airbox_wrap::RoomGrid`] models a loudspeaker cone or a soundboard in a cabinet
/// honestly and a cymbal dishonestly.
///
/// **The implementation news is better than the area-coupling plan feared.** That plan budgeted for
/// new boundary machinery — "putting the plate on a plane of velocity faces and replacing their
/// momentum update with the plate's motion". Measured, prescribing the face velocity and injecting
/// a `-q`/`+q` pair on the two node planes straddling the plane are **the same arithmetic**:
///
/// ```text
/// node (i,j,m)  divergence term  +u_face / w_z   ->  dp = -k rho0 c0^2 u_face / w_z
/// a soft source q at that node                   ->  dp = +k rho0 c0^2 q / W,  W = A_face w_z
/// ```
///
/// and `A_face * w_z = W` identically, so with `q = A_face u_face` the two agree term for term,
/// with the sign flipping between the planes. **The only new machinery is the cut** —
/// [`AirBox::add_cut`], zeroing `u` on the faces under the surface. The port protocol, the
/// injection weights and the `injected` ledger are batch 3's, unchanged. (Batch 1's step ordering
/// paid for batch 3 the same way — "a soft injection at a wall node *is* the moving-wall
/// condition"; this is that lesson one geometry further.)
///
/// **The load is `2 T^T R T`, and the room's response is diagonal on both planes.** With `T` the
/// volume-weight matrix (surface -> **face**), `q = T v` the per-face volume velocity, and the
/// injection weights `(-q, +q)` on planes `m` and `m+1`:
///
/// ```text
/// pbar_lo = pbar_free_lo  -  R q
/// pbar_hi = pbar_free_hi  +  R q
/// f       = -T^T (pbar_hi - pbar_lo)  =  -T^T d_free  -  2 T^T R T v
/// ```
///
/// with the **same** `R_j = k rho0 c0^2 / (2 W_j (1 + beta_j))` on both planes — which is what the
/// rim refusals below are for. So the load is exactly *twice* the one-sided load on the same `T`
/// and `R`, still constant, symmetric, PSD and sparse, so it still folds into the resonator's own
/// sparse LU with nothing new solved. The family's ladder gains its last rung:
/// [`crate::radiation::RadiatedBody`] scalar `R` -> one division;
/// [`crate::radiation::RationalAirLoad`] `R_eff` + one aux state -> one division; [`RoomPort`]
/// scalar `R_room` -> one division; [`SurfacePort`] diagonal `R` with a matrix `T` -> absorbed into
/// the LU; **this**, the same, doubled.
///
/// **The factor 2 must be earned by measurement, not by construction.** `assert load_matrix == 2 *
/// one_sided` is not a test — it re-checks arithmetic this file just wrote. Two realistic ways to
/// get it wrong are blind to *different* ledgers: a `1x` inside the factorization only leaves
/// `|radiated - injected|` at rounding (4.5e-16) while the scene drifts (1.8e-02), and a consistent
/// `1x` ("I forgot the plate has two faces") leaves the scene flat to 2.4e-14 while the ledgers
/// disagree by 1.8x. **So the money test is not sufficient either** — the guard that catches both
/// is the coupled residual at two timesteps, against the room's own post-closure pressure jump.
///
/// **The sign convention is different from batch 3's, and it is load-bearing.** [`SurfacePort`] can
/// make the inward normal disappear by defining positive displacement *along* it, which is why no
/// inward normal appears in that code at all. An interior plane has two sides and no inward normal,
/// so this class must choose: **positive surface displacement is along the plane's own axis**
/// (`+x`/`+y`/`+z`), so the `index+1` plane is the one a positive velocity compresses. Flip the
/// `-q`/`+q` order **and** the jump direction together and every energetic quantity is
/// *bit-identical* — the load matrix, the solve, the pressure jump and `radiated_energy` — while
/// the room's field is exactly inverted and the plate is anti-driven by it: passive, conservative,
/// green, and pushing the wrong way. (Flip only one of the two and the conserved total does catch
/// it.) The detector is the sign of the room's own pressure on the **first** step.
///
/// **The obstacle is up to one air cell larger than the surface, and that is the honest choice.**
/// The cut is the **support of** `T`, not the surface's footprint. It has to be: the identity above
/// holds only where a face carrying `q` is also cut, so a face with `q` and no cut is a phantom
/// patch — and clipping `T` instead is exactly the "volume conserved, ledgers green, geometry
/// quietly wrong" failure batch 3 refuses. [`InteriorSurfacePort::blocked_area`] reports the cost.
///
/// **A legal interior surface always has a diffraction path around it**, and that is a corollary
/// rather than a limitation to hide: the rim refusal means a port can never span a full
/// cross-section, so it can never seal the room. [`AirBox::add_cut`] provides the sealing version,
/// which is what the sub-room modal oracle uses.
///
/// **Notes**
/// **There is no open-face refusal here, and that is provable rather than an omission.** The
/// in-plane rim refusal keeps every patch node strictly interior on both in-plane axes, and the
/// `index` refusal does the same along the normal — so an interior patch cannot touch any wall,
/// open or otherwise, and [`SurfacePort`]'s two reasons for that refusal cannot arise.
///
/// **Cut-face disjointness is a genuinely new failure mode, not an inherited one.** Two ports can
/// have disjoint *node* sets and overlapping *cuts*, because the node sets live on different planes
/// while the cuts can share one. That is why [`AirBox::add_cut`] is additive and why a port records
/// its own face set.
#[derive(Debug, Clone)]
pub struct InteriorSurfacePort {
    patch: Patch,
    plane: String,
    axis: usize,
    face_index: usize,
    nodes_lo: Nodes,
    nodes_hi: Nodes,
    in_plane: (Vec<i64>, Vec<i64>),
}

patch_accessors!(SurfacePort);
patch_accessors!(InteriorSurfacePort);

impl SurfacePort {
    /// Mount a surface flush in `face`, with per-node positions `coords` (m) and areas `areas`.
    ///
    /// `origin` places the footprint in the plane's own two axes; `None` centres it, which is what
    /// makes the load equivariant under the grid's own mirror. Every refusal runs before the port
    /// claims its footprint on the room, so a rejected port leaves the room exactly as it was.
    pub fn new(
        room: &mut AirBox,
        face: &str,
        coords: &[[f64; 2]],
        areas: &[f64],
        origin: Option<(f64, f64)>,
        spreading: Spreading,
    ) -> Result<SurfacePort, PortError> {
        let (axis, end, t0, t1) =
            face_axes(face).ok_or_else(|| PortError::UnknownFace(face.to_owned()))?;
        let where_ = format!("face '{face}'");
        let (face_coords, origin) =
            accept_surface(room.p.n, room.p.h, coords, areas, origin, (t0, t1), &where_)?;

        let view = room.view();
        let n_axis_end = if end == 0 { 0usize } else { view.n[axis] };
        let asm = assemble(
            &view,
            &face_coords,
            areas,
            spreading,
            (t0, t1),
            &where_,
            &|i0, i1| {
                let mut out: Nodes = [Vec::new(), Vec::new(), Vec::new()];
                out[axis] = vec![n_axis_end; i0.len()];
                out[t0] = i0.iter().map(|&v| v as usize).collect();
                out[t1] = i1.iter().map(|&v| v as usize).collect();
                out
            },
            &|nodes| nodes.clone(),
            1.0,
        )?;

        let (unfed, foot) = footprint_unfed(&asm.nodes[t0], &asm.nodes[t1], view.n[t1]);
        if unfed != 0 {
            return Err(PortError::TooCoarseFootprint {
                where_,
                unfed,
                foot,
                h: view.h,
                spreading,
            });
        }
        let touched = touched_open_faces(&room.p, &asm.nodes);
        if !touched.is_empty() {
            return Err(PortError::SurfaceOnOpenFace {
                where_,
                faces: touched,
            });
        }
        let shape = view.node_shape();
        check_disjoint(&room.claims, &asm.flat, shape, &where_, asm.nodes[0].len())?;

        room.claims.push(crate::airbox::PortClaim {
            nodes: asm.flat.clone(),
            label: index_repr(asm.index),
        });
        Ok(SurfacePort {
            patch: Patch {
                spreading,
                in_plane_axes: (t0, t1),
                coords: coords.to_vec(),
                areas: areas.to_vec(),
                origin,
                face_coords,
                nodes: asm.nodes,
                index: asm.index,
                flat: asm.flat,
                t: asm.t,
                r: asm.r,
                load: asm.load,
                footprint_empty: unfed,
                where_,
                queued: None,
            },
            face: face.to_owned(),
            axis,
        })
    }

    /// The wall face the surface is mounted in.
    pub fn face(&self) -> &str {
        &self.face
    }

    /// The face's normal axis.
    pub fn axis(&self) -> usize {
        self.axis
    }

    /// The open-circuit centered pressure **vector** `pbar_free` over the patch, `O(patch)`.
    pub fn free_pressure(&self, room: &AirBox) -> Vec<f64> {
        let view = room.view();
        let n = &self.patch.nodes;
        free_pressure_nodes(&view, &[&n[0], &n[1], &n[2]])
    }

    /// Queue the **per-node** volume-velocity vector `q` (m^3/s) for the room's next step.
    pub fn inject(&mut self, room: &mut AirBox, q: &[f64]) -> Result<(), PortError> {
        let count = self.patch.nodes[0].len();
        if q.len() != count {
            return Err(PortError::QLength {
                expected: count,
                got: q.len(),
                per_face: false,
                node_count: count,
            });
        }
        self.require_ready(room)?;
        queue_vector(room, &self.patch.flat, q);
        self.patch.queued = Some((room.epoch, room.n));
        Ok(())
    }
}

impl InteriorSurfacePort {
    /// Hang a surface on the `index`-th velocity face plane normal to `plane`.
    ///
    /// The two node planes straddling the surface must both be strictly interior, which is what
    /// makes one `R` correct for both sides. Every refusal — including the room's own shared-face
    /// one — runs before the port claims its footprint, so a rejected port leaves `room.claims`
    /// and `room.cuts` exactly as it found them.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        room: &mut AirBox,
        plane: &str,
        index: i64,
        coords: &[[f64; 2]],
        areas: &[f64],
        origin: Option<(f64, f64)>,
        spreading: Spreading,
    ) -> Result<InteriorSurfacePort, PortError> {
        let axis = AirBox::plane_axis(plane).map_err(PortError::Cut)?;
        let n_face = room.p.n[axis];
        if !(1..=n_face as i64 - 2).contains(&index) {
            return Err(PortError::InteriorIndexOutOfRange {
                plane: plane.to_owned(),
                index,
                n_face,
            });
        }
        let face_index = index as usize;
        let (t0, t1) = crate::airbox::other_axes(axis);
        let where_ = format!("the plane '{plane}' cross-section at index {index}");
        let (face_coords, origin) =
            accept_surface(room.p.n, room.p.h, coords, areas, origin, (t0, t1), &where_)?;

        let view = room.view();
        // The combined node set is both straddling planes, low first — what the disjointness check
        // and the room's bookkeeping see. `R` is built from the LOW plane alone: one resistance for
        // both, which the rim refusal is exactly what makes true.
        let asm = assemble(
            &view,
            &face_coords,
            areas,
            spreading,
            (t0, t1),
            &where_,
            &|i0, i1| {
                let mut out: Nodes = [Vec::new(), Vec::new(), Vec::new()];
                for offset in 0..2usize {
                    for m in 0..i0.len() {
                        out[axis].push(face_index + offset);
                        out[t0].push(i0[m] as usize);
                        out[t1].push(i1[m] as usize);
                    }
                }
                out
            },
            &|nodes| {
                let half = nodes[0].len() / 2;
                [
                    nodes[0][..half].to_vec(),
                    nodes[1][..half].to_vec(),
                    nodes[2][..half].to_vec(),
                ]
            },
            2.0,
        )?;
        let half = asm.nodes[0].len() / 2;
        let total = asm.nodes[0].len();
        let split = |range: std::ops::Range<usize>| -> Nodes {
            [
                asm.nodes[0][range.clone()].to_vec(),
                asm.nodes[1][range.clone()].to_vec(),
                asm.nodes[2][range].to_vec(),
            ]
        };
        let nodes_lo = split(0..half);
        let nodes_hi = split(half..total);
        let index_triple = [nodes_lo[0][0], nodes_lo[1][0], nodes_lo[2][0]];

        let (unfed, foot) = footprint_unfed(&nodes_lo[t0], &nodes_lo[t1], view.n[t1]);
        if unfed != 0 {
            return Err(PortError::TooCoarseFootprint {
                where_,
                unfed,
                foot,
                h: view.h,
                spreading,
            });
        }
        let shape = view.node_shape();
        check_disjoint(&room.claims, &asm.flat, shape, &where_, total)?;

        // The cut is the only registration that WRITES the room's state, so it goes last among the
        // things that can fail — and it can fail: the room refuses a cut that shares a face with
        // an existing port's. Its refusal fires before `claims` is touched, so a rejected interior
        // port leaves both books as it found them.
        let (i0, i1) = &asm.in_plane;
        let cut_i0: Vec<usize> = i0.iter().map(|&v| v as usize).collect();
        let cut_i1: Vec<usize> = i1.iter().map(|&v| v as usize).collect();
        room.register_cut(
            Some(index_repr(index_triple)),
            axis,
            face_index,
            &cut_i0,
            &cut_i1,
        )
        .map_err(PortError::Cut)?;
        room.claims.push(crate::airbox::PortClaim {
            nodes: asm.flat.clone(),
            label: index_repr(index_triple),
        });

        Ok(InteriorSurfacePort {
            patch: Patch {
                spreading,
                in_plane_axes: (t0, t1),
                coords: coords.to_vec(),
                areas: areas.to_vec(),
                origin,
                face_coords,
                nodes: asm.nodes,
                index: index_triple,
                flat: asm.flat,
                t: asm.t,
                r: asm.r,
                load: asm.load,
                footprint_empty: unfed,
                where_,
                queued: None,
            },
            plane: plane.to_owned(),
            axis,
            face_index,
            nodes_lo,
            nodes_hi,
            in_plane: (i0.clone(), i1.clone()),
        })
    }

    /// The interior plane the surface hangs on.
    pub fn plane(&self) -> &str {
        &self.plane
    }

    /// The plane's normal axis.
    pub fn axis(&self) -> usize {
        self.axis
    }

    /// The velocity-face index the surface sits at.
    pub fn face_index(&self) -> usize {
        self.face_index
    }

    /// The low-side node plane.
    pub fn nodes_lo(&self) -> &Nodes {
        &self.nodes_lo
    }

    /// The high-side node plane.
    pub fn nodes_hi(&self) -> &Nodes {
        &self.nodes_hi
    }

    /// The in-plane node indices the surface reaches, on the plane's own two axes.
    pub fn in_plane(&self) -> (&[i64], &[i64]) {
        (&self.in_plane.0, &self.in_plane.1)
    }

    /// How many velocity faces the surface cuts — half of [`Self::node_count`].
    pub fn face_count(&self) -> usize {
        self.nodes_lo[0].len()
    }

    /// The **cut** area (m^2) — `face_count * h^2`, and not [`Self::net_area`].
    ///
    /// The obstacle is the support of `T`, so it is up to one air cell larger than the surface in
    /// each direction. Reported rather than discovered. Two things to know before blaming a
    /// radiated number on it: under air-grid refinement the ratio to the moving surface's own
    /// bounding rectangle is non-monotone (the node count is a rounding of the footprint) and
    /// trends to 1; and on a **supported** plate the moving surface is the *live* footprint, so the
    /// clamped rim is not part of the obstacle at all — the free plate is the configuration where
    /// the obstacle and the physical plate coincide, which is also the one a suspended object is.
    pub fn blocked_area(&self, room: &AirBox) -> f64 {
        self.nodes_lo[0].len() as f64 * scalar_pow(room.p.h, 2.0)
    }

    /// The open-circuit centered pressure on the **low** and **high** node planes, `O(patch)`.
    ///
    /// Batch 3's contract, twice: read *before* `room.step()`, from the stored `u^{n+1/2}`. The
    /// shared local read needs no cut-awareness — on the two straddling planes the face between
    /// them is cut, i.e. `u = 0`, which the read picks up from the stored array like any other
    /// value. See [`free_pressure_nodes`].
    pub fn free_pressure(&self, room: &AirBox) -> (Vec<f64>, Vec<f64>) {
        let view = room.view();
        let lo = &self.nodes_lo;
        let hi = &self.nodes_hi;
        (
            free_pressure_nodes(&view, &[&lo[0], &lo[1], &lo[2]]),
            free_pressure_nodes(&view, &[&hi[0], &hi[1], &hi[2]]),
        )
    }

    /// Queue the **per-face** volume-velocity vector `q` (m^3/s) as a `-q` / `+q` pair.
    ///
    /// `-q` goes to the node plane at [`InteriorSurfacePort::face_index`] and `+q` to the one above
    /// it, which is the sign convention of the class docstring: a positive velocity along the
    /// plane's own axis compresses the air it moves toward. Both are ordinary soft injections
    /// ([`AirBox::inject`]'s own source term, vectorized), so the room books the work at its own
    /// post-closure pressure from both planes and the two ledgers stay an independent check on each
    /// other: their sum is `k q . (pbar_hi - pbar_lo)`, exactly the port's own channel.
    pub fn inject(&mut self, room: &mut AirBox, q: &[f64]) -> Result<(), PortError> {
        let faces = self.nodes_lo[0].len();
        if q.len() != faces {
            return Err(PortError::QLength {
                expected: faces,
                got: q.len(),
                per_face: true,
                node_count: self.patch.nodes[0].len(),
            });
        }
        self.require_ready(room)?;
        let neg: Vec<f64> = q.iter().map(|&v| -v).collect();
        queue_vector(room, &self.patch.flat[..faces], &neg);
        queue_vector(room, &self.patch.flat[faces..], q);
        self.patch.queued = Some((room.epoch, room.n));
        Ok(())
    }
}

/// Append one distributed injection to the room's pending queue.
///
/// The reference queues `(nodes, q_vector, 1.0)` and the room's step multiplies the two, which is
/// exactly [`crate::airbox::PortInjection`] with `q = 1.0` and the per-node vector in `w`. So the
/// distributed tier needs no new room-side machinery at all — it reuses the lumped one's, with the
/// roles of "share" and "magnitude" swapped.
fn queue_vector(room: &mut AirBox, flat: &[usize], q: &[f64]) {
    room.pending_ports.push(crate::airbox::PortInjection {
        nodes: flat.to_vec(),
        w: q.to_vec(),
        q: 1.0,
    });
}

/// The disjointness refusal, over whichever ports the room already holds.
fn check_disjoint(
    claims: &[crate::airbox::PortClaim],
    flat: &[usize],
    shape: [usize; 3],
    where_: &str,
    count: usize,
) -> Result<(), PortError> {
    for claim in claims {
        let (first, shared) = shared_nodes(flat, &claim.nodes);
        if let Some(first) = first {
            return Err(PortError::SurfaceOverlapping {
                where_: where_.to_owned(),
                count,
                node: unravel(first, shape),
                other: claim.label.clone(),
                shared,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `(p, _W, _w, _beta, spare)` for a 2x2x2-cell room at h = 0.1.
    type Room = (Vec<f64>, Vec<f64>, [Vec<f64>; 3], Vec<f64>, Vec<f64>);

    fn uniform_room() -> Room {
        // A 2x2x2-cell room: 27 pressure nodes, h = 0.1.
        let n = [2usize, 2, 2];
        let h = 0.1;
        let w: [Vec<f64>; 3] = std::array::from_fn(|d| {
            let mut v = vec![h; n[d] + 1];
            v[0] = 0.5 * h;
            let last = v.len() - 1;
            v[last] = 0.5 * h;
            v
        });
        let mut node_w = Vec::new();
        for i0 in 0..=n[0] {
            for i1 in 0..=n[1] {
                for i2 in 0..=n[2] {
                    node_w.push(w[0][i0] * w[1][i1] * w[2][i2]);
                }
            }
        }
        let p: Vec<f64> = (0..27).map(|i| 0.001 * (i as f64 + 1.0)).collect();
        let beta = vec![0.0; 27];
        (p, node_w, w, beta, vec![0.0; 27])
    }

    #[test]
    fn face_axes_matches_the_reference_table() {
        assert_eq!(face_axes("x0"), Some((0, 0, 1, 2)));
        assert_eq!(face_axes("x1"), Some((0, 1, 1, 2)));
        assert_eq!(face_axes("y0"), Some((1, 0, 0, 2)));
        assert_eq!(face_axes("z1"), Some((2, 1, 0, 1)));
        assert_eq!(face_axes("q0"), None);
        assert_eq!(plane_axis("y"), Some(1));
        assert_eq!(plane_axis("w"), None);
    }

    #[test]
    fn a_point_ball_is_one_node_and_a_wide_one_is_the_whole_room() {
        let n = [2usize, 2, 2];
        let tiny = ball_nodes(n, 0.1, [1, 1, 1], 0.01);
        assert_eq!(tiny[0].len(), 1);
        assert_eq!((tiny[0][0], tiny[1][0], tiny[2][0]), (1, 1, 1));
        let all = ball_nodes(n, 0.1, [1, 1, 1], 10.0);
        assert_eq!(all[0].len(), 27);
        // C order: the first node is (0,0,0) and the last is (2,2,2).
        assert_eq!((all[0][0], all[1][0], all[2][0]), (0, 0, 0));
        assert_eq!((all[0][26], all[1][26], all[2][26]), (2, 2, 2));
    }

    #[test]
    fn free_pressure_at_rest_is_the_stored_pressure() {
        let (p, node_w, w, beta, _) = uniform_room();
        let zero = vec![0.0; 18];
        let view = RoomView {
            n: [2, 2, 2],
            h: 0.1,
            k: 1.0 / 40_000.0,
            rho0: 1.2,
            c0: 343.0,
            p: &p,
            u: [&zero, &zero, &zero],
            w: [&w[0], &w[1], &w[2]],
            node_w: &node_w,
            beta: &beta,
            has_walls: false,
        };
        let nodes = [vec![1usize], vec![1], vec![1]];
        let out = free_pressure_nodes(&view, &[&nodes[0], &nodes[1], &nodes[2]]);
        assert_eq!(out, vec![p[view.flat([1, 1, 1])]]);
    }

    #[test]
    fn the_port_weights_sum_to_one_and_use_the_pairwise_reduction() {
        let (p, node_w, w, beta, _) = uniform_room();
        let zero = vec![0.0; 18];
        let view = RoomView {
            n: [2, 2, 2],
            h: 0.1,
            k: 1.0 / 40_000.0,
            rho0: 1.2,
            c0: 343.0,
            p: &p,
            u: [&zero, &zero, &zero],
            w: [&w[0], &w[1], &w[2]],
            node_w: &node_w,
            beta: &beta,
            has_walls: false,
        };
        let all = ball_nodes([2, 2, 2], 0.1, [1, 1, 1], 10.0);
        let (weights, big) = port_weights(&view, &[&all[0], &all[1], &all[2]]);
        assert_eq!(weights.len(), 27);
        // The normaliser has to be the pairwise sum and not a left-to-right one: 27 terms is above
        // §30.2's eight-element cutoff, so the two spellings are genuinely different computations
        // and this fixture is a witness. Asserted as a *difference* so the choice is pinned rather
        // than assumed.
        let plain: f64 = big.iter().fold(0.0, |a, &b| a + b);
        assert_ne!(
            reduce::sum(&big),
            plain,
            "the fixture cannot see the blocking"
        );
        let total: f64 = reduce::sum(&weights);
        assert!((total - 1.0).abs() < 1e-15);
    }

    #[test]
    fn the_diagonals_side_is_observable_and_the_left_one_is_ours() {
        // The batch's association finding, as a native pin. It is written as a SEARCH and not as
        // three hand-picked numbers: the first draft was three constants, they landed in the
        // agreeing two-thirds, and the test went red having found nothing. §26.6, arriving inside
        // the test written to record §26.6's own subject.
        let mut s: u64 = 987_654_321;
        let mut rand = move || {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let mut differed = 0;
        let mut total = 0;
        for _ in 0..200 {
            let n = 3;
            let rows: Vec<Vec<(usize, f64)>> = (0..n)
                .map(|_| (0..n).map(|j| (j, rand())).collect())
                .collect();
            let t = Csr::from_rows_keeping_zeros(n, n, rows);
            let r: Vec<f64> = (0..n).map(|_| rand()).collect();
            let left = load_matrix(&t, &r, 1.0);
            for i in 0..n {
                for j in 0..n {
                    // The same product with the diagonal folded RIGHT, longhand, over k ascending.
                    let mut right = 0.0;
                    for (k, &rk) in r.iter().enumerate() {
                        right += t.get(k, i) * (rk * t.get(k, j));
                    }
                    total += 1;
                    if left.get(i, j) != right {
                        differed += 1;
                    }
                }
            }
        }
        assert!(
            differed > 0,
            "the two associations agreed at all {total} entries searched"
        );
    }

    #[test]
    fn an_all_equal_row_cannot_see_the_association() {
        // The mechanism behind the blind fixture: (x d) x and x (d x) are the same double for
        // every x and d, because the two outer factors are literally the same number.
        let mut s: u64 = 12345;
        for _ in 0..200_000 {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let x = ((s >> 11) as f64) / ((1u64 << 53) as f64);
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let d = ((s >> 11) as f64) / ((1u64 << 53) as f64);
            assert_eq!((x * d) * x, x * (d * x));
        }
    }

    #[test]
    fn a_rectangle_footprint_has_nothing_unfed() {
        let mut i0 = Vec::new();
        let mut i1 = Vec::new();
        for a in 2..6 {
            for b in 3..7 {
                i0.push(a);
                i1.push(b);
            }
        }
        assert_eq!(footprint_unfed(&i0, &i1, 10), (0, 16));
        // Punch a hole and the span-wise set names it.
        let hole = i0
            .iter()
            .zip(i1.iter())
            .position(|(&a, &b)| a == 3 && b == 4)
            .unwrap();
        i0.remove(hole);
        i1.remove(hole);
        assert_eq!(footprint_unfed(&i0, &i1, 10), (1, 16));
    }

    #[test]
    fn plane_nodes_uses_pythons_floor_division() {
        // A negative row is what a footprint starting a hair below zero produces; NumPy's `//`
        // floors toward minus infinity and its `%` is non-negative, unlike Rust's `/` and `%`.
        let (i0, i1, sorted) = plane_nodes(&[-1, 5, 12], 3);
        assert_eq!(sorted, vec![-1, 5, 12]);
        assert_eq!(i0, vec![-1, 1, 3]);
        assert_eq!(i1, vec![3, 1, 0]);
    }

    #[test]
    fn unravel_inverts_ravel() {
        let shape = [4usize, 5, 6];
        for flat in 0..(4 * 5 * 6) {
            let i = unravel(flat, shape);
            assert_eq!((i[0] * shape[1] + i[1]) * shape[2] + i[2], flat);
        }
    }

    // -- the lumped port, measured against the room rather than against its own formula --------
    //
    // `r_room` is a *predicted constant*: the incremental centered pressure the room presents per
    // unit of injected volume velocity. The bars below measure it by driving the room and never
    // consult the formula except in the final comparison — §45.3's two-routes-to-one-number, and
    // the shape that found the piston defect. They need no port object at all, which is the point:
    // `RoomPort` itself lives in the binding crate (§39.3's permanent blocker), but the arithmetic
    // it is a wrapper around is here, and so is the room it makes a claim about.

    use crate::airbox::{self, AirBox, Params, Wall, C0_AIR, RHO0_AIR};

    /// The port fixtures' room — `AIRBOX_PORT_ROOM_DEFAULT` at `AIRBOX_PORT_H_DEFAULT`,
    /// `N = (10, 8, 6)` at 0.9 of the CFL ceiling.
    fn port_room(walls: [Wall; 6]) -> Params {
        let h = 0.05;
        let fs = C0_AIR * 3.0_f64.sqrt() / (0.9 * h);
        Params::new([0.5, 0.4, 0.3], fs, h, walls, None, RHO0_AIR, C0_AIR).unwrap()
    }

    /// A `RoomView` borrowed off a native room — what the binding fills in from Python attributes.
    fn view_of(b: &AirBox) -> RoomView<'_> {
        RoomView {
            n: b.p.n,
            h: b.p.h,
            k: b.p.k,
            rho0: b.p.rho0,
            c0: b.p.c0,
            p: &b.pressure,
            u: [&b.u[0], &b.u[1], &b.u[2]],
            w: [&b.p.w[0], &b.p.w[1], &b.p.w[2]],
            node_w: &b.p.wv,
            beta: &b.p.beta,
            has_walls: b.p.has_walls,
        }
    }

    /// A cheap deterministic field, matching `airbox`'s own test seed.
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

    /// The port-weighted centered pressure one step on from `b`'s current state, injecting `q`.
    ///
    /// Takes the room by shared reference and mutates nothing, so the two calls the differential
    /// needs start from an identical state by construction — where the Python bar has to snapshot
    /// and restore ten fields, because it drives the room's own `step`.
    fn pbar_after(b: &AirBox, flat: &[usize], w: &[f64], q: f64) -> f64 {
        let div = airbox::divergence(&b.p, &b.u[0], &b.u[1], &b.u[2]);
        let mut p_next = airbox::pressure_step(&b.p, &b.pressure, &div);
        airbox::inject_port(&b.p, &mut p_next, flat, w, q);
        if b.p.has_walls {
            let mut booked = 0.0;
            airbox::apply_walls(&b.p, &mut p_next, &b.pressure, &mut booked);
        }
        reduce::sum_by(flat.len(), |m| {
            w[m] * (0.5 * (p_next[flat[m]] + b.pressure[flat[m]]))
        })
    }

    #[test]
    fn r_room_is_what_the_room_actually_does() {
        // Four mounting points, each summing a different number of wall admittances into beta,
        // times rigid and matched walls. The three non-interior sites with lossy walls are where
        // the 1/(1 + beta) factor lives; without it the interior case still passes.
        let matched = airbox::impedance_from_zeta(1.0, RHO0_AIR, C0_AIR);
        for walls in [[Wall::Rigid; 6], [Wall::Impedance(matched); 6]] {
            for at in [[3usize, 3, 3], [0, 3, 3], [0, 0, 3], [0, 0, 0]] {
                let mut b = AirBox::new(port_room(walls));
                let seed = noise(b.p.n_nodes());
                b.set_state(&seed, None);
                for _ in 0..23 {
                    b.step();
                }
                // A point port: one node, weight 1.
                let nodes: Nodes = [vec![at[0]], vec![at[1]], vec![at[2]]];
                let refs = [&nodes[0][..], &nodes[1][..], &nodes[2][..]];
                let v = view_of(&b);
                let (w, big_w) = port_weights(&v, &refs);
                let predicted = r_room(&v, &refs, &w, &big_w);
                let flat = ravel(&refs, v.node_shape());

                let u = 3.7e-4;
                let measured = (pbar_after(&b, &flat, &w, u) - pbar_after(&b, &flat, &w, 0.0)) / u;
                assert!(
                    (measured - predicted).abs() <= 1e-12 * predicted,
                    "at {at:?}: measured {measured:e} vs predicted {predicted:e}"
                );
            }
        }
    }

    #[test]
    fn a_spread_port_is_the_same_claim_over_many_nodes() {
        // Many nodes with differing W and beta, so the weighted sums are exercised rather than
        // collapsing to one term — and the ball straddles a lossy wall, where the per-node beta
        // differs across the port.
        let matched = airbox::impedance_from_zeta(1.0, RHO0_AIR, C0_AIR);
        let mut walls = [Wall::Rigid; 6];
        walls[0] = Wall::Impedance(matched);
        let mut b = AirBox::new(port_room(walls));
        let seed = noise(b.p.n_nodes());
        b.set_state(&seed, None);
        for _ in 0..23 {
            b.step();
        }
        let nodes = ball_nodes(b.p.n, b.p.h, [1, 3, 3], 0.12);
        let refs = [&nodes[0][..], &nodes[1][..], &nodes[2][..]];
        assert!(
            refs[0].len() > 20,
            "the ball collapsed to {} nodes",
            refs[0].len()
        );
        let v = view_of(&b);
        let (w, big_w) = port_weights(&v, &refs);
        assert!((reduce::sum(&w) - 1.0).abs() < 1e-15);
        // The weights are genuinely uneven: a wall node carries half an interior one.
        let (lo, hi) = w
            .iter()
            .fold((f64::MAX, 0.0f64), |(l, h), &x| (l.min(x), h.max(x)));
        assert!(
            hi / lo > 1.9,
            "the port's node weights are uniform: {lo:e}..{hi:e}"
        );
        let predicted = r_room(&v, &refs, &w, &big_w);
        let flat = ravel(&refs, v.node_shape());
        let u = 3.7e-4;
        let measured = (pbar_after(&b, &flat, &w, u) - pbar_after(&b, &flat, &w, 0.0)) / u;
        assert!(
            (measured - predicted).abs() <= 1e-12 * predicted,
            "measured {measured:e} vs predicted {predicted:e}"
        );
    }

    #[test]
    fn the_wall_factor_in_r_room_is_not_free() {
        // Pin the trap: on a lossy wall the naive `k rho c^2 / (2 W)` differs from the truth by
        // exactly `1 + beta`, so this is not a factor that "cancels anyway". A corner sums three
        // admittances, which makes it a big one.
        let matched = airbox::impedance_from_zeta(1.0, RHO0_AIR, C0_AIR);
        let b = AirBox::new(port_room([Wall::Impedance(matched); 6]));
        let nodes: Nodes = [vec![0usize], vec![0], vec![0]];
        let refs = [&nodes[0][..], &nodes[1][..], &nodes[2][..]];
        let v = view_of(&b);
        let (w, big_w) = port_weights(&v, &refs);
        let beta = v.beta[v.flat([0, 0, 0])];
        assert!(
            beta > 0.5,
            "a corner should sum three admittances, got beta = {beta}"
        );
        let naive = reduce::sum_by(w.len(), |m| {
            (((w[m] * w[m]) * v.k) * v.rho0) * crate::pyfloat::scalar_pow(v.c0, 2.0)
                / (2.0 * big_w[m])
        });
        let ratio = naive / r_room(&v, &refs, &w, &big_w);
        assert!(
            (ratio - (1.0 + beta)).abs() < 1e-13 * (1.0 + beta),
            "ratio {ratio} is not 1 + beta = {}",
            1.0 + beta
        );
    }
}
