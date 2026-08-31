//! The port tier of `physsynth/core/airbox.py` — a room's two-way terminals.
//!
//! Port of `RoomPort`, `_PatchPort`, `SurfacePort` and `InteriorSurfacePort`, plus the two module
//! helpers they share (`_face_axes`, `_free_pressure_nodes`). The reference docstrings there are
//! the physics; this module documents only what the translation had to decide.
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

use crate::fmt::py_float;
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
}
