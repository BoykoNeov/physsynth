//! Two-dimensional grid geometry, the guitar outline, and the masked 5-point Laplacian.
//!
//! Port of the *builder* half of `physsynth/core/operators2d.py`, HANDOFF §5 model #4, plus the
//! outline geometry of model #5g.
//!
//! # Why only part of that module is here
//!
//! The migration plan's §4 risk map is keyed to **files**, and it puts `operators2d` in Group D
//! (sparse LU, Phase 5) because the module contains `VonKarmanBracket` and `AiryStressSolver`,
//! which factor with SuperLU. That grouping is right about the module and wrong about the *unit of
//! porting*: `membrane` is a Phase 2 model, and what it reaches for — `grid_coords`,
//! `rectangle_mask`, `disk_mask`, `laplacian_from_mask`, `embed` — never solves anything. It
//! assembles. So the file split in two, and this is the half with no solver in it.
//!
//! Phase 5's first batch adds the **geometry** of the plate family: `guitar_half_width`,
//! `guitar_scale`, `guitar_mask`, `guitar_area`, `live_cells`, `cells_per_node` and
//! `prune_to_area_carrying`. They are grouped together and ported apart from the operators they
//! serve because they are the part of the module whose output is **discrete** — a node is an
//! unknown or it is not — which makes a last bit of `sin` a geometry change rather than a
//! rounding (plan §22.6, and `guitar_mask` here).
//!
//! Phase 5's second batch adds the **matrices** those masks are for: `biharmonic_from_mask`,
//! `dirichlet_interior_d2_1d`, `orthotropic_biharmonic` and `free_plate_stiffness*`. Every one of
//! them is bit-identical to SciPy's, `data`, `indices` and `nnz` alike — but that took an edit to
//! the *reference*, not to this file. The values were never in question; the stored column order
//! was, because SciPy's sparse product hands back each row in the order its kernel happened to
//! touch the columns and a CSR matvec sums a row in stored order. `portable.canonical` sorts the
//! Python side, which is §18.2's manoeuvre for the string family landing exactly where §18.4 said
//! in advance that it would. See `biharmonic_from_mask` below.
//!
//! Phase 5's **third** batch finishes the module with the *nonlinear* plate: the five private 1-D
//! differences, `VonKarmanBracket` and `AiryStressSolver`. Two of the three kinds of claim in this
//! file meet there. The bracket is **exact** — every matvec in it is a canonical row gather, the
//! `Acell.T` one included, for a reason that is a lemma rather than a measurement (see the type).
//! The Airy solve is **not and cannot be**: it factors with SuperLU on the other side and §24.2
//! settled that in advance. What separates them is §24.4's manoeuvre — put the Python solver on
//! the Rust factorization and the two go bit-identical, so the whole residue is the solver.
//!
//! That batch also cost the reference one edit, and it is a *pair of parentheses*: `BᵀWB` has two
//! bracketings, the values are identical under either (both outer factors share a mantissa,
//! §26.5), and the association nonetheless moves the **sum**, because SciPy hands the
//! left bracketing's intermediate back descending. Note what this crate could not have done about
//! it: `Csr::from_rows` sorts, so a descending row is not expressible here at all. See
//! `AiryStressSolver`.
//!
//! # Flat ordering, everywhere
//!
//! A 2-D field of shape `(nrows, ncols)` is a flat `Vec<f64>` in **row-major (C) order**, index
//! `j * ncols + i` for row `j`, column `i` — which is what NumPy hands out and what `mask.shape`
//! means in the original. The *live-node* vector is a different, shorter thing: one entry per
//! live node, in C-order over the live positions, matching `np.nonzero(mask)`. `index_map` is the
//! map between them and `-1` marks a dead node. Confusing the two is the obvious bug here and the
//! reason both are named in every signature.

use crate::sparse::Csr;
use crate::sparse_lu::{SparseLu, SparseLuError};

/// A live-node mask over a `(nrows, ncols)` node grid — which nodes are unknowns.
///
/// `nrows`/`ncols` count **nodes**, not cells, so a rectangle of `nx` by `ny` segments has
/// `(ny + 1, nx + 1)` here. The rim nodes of a Dirichlet domain are *dead*: they are held at zero
/// and never stepped, so they are not unknowns and do not appear in the live vector at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mask {
    nrows: usize,
    ncols: usize,
    live: Vec<bool>,
}

impl Mask {
    /// Build from a row-major flag array.
    ///
    /// # Panics
    /// If `live` does not have `nrows * ncols` entries.
    pub fn new(nrows: usize, ncols: usize, live: Vec<bool>) -> Mask {
        assert_eq!(
            live.len(),
            nrows * ncols,
            "mask must have nrows * ncols entries"
        );
        Mask { nrows, ncols, live }
    }

    /// Number of node rows (the first axis, `y`).
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of node columns (the second axis, `x`).
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// The flags, row-major.
    pub fn flags(&self) -> &[bool] {
        &self.live
    }

    /// Is node `(j, i)` an unknown?
    pub fn at(&self, j: usize, i: usize) -> bool {
        self.live[j * self.ncols + i]
    }

    /// How many unknowns the mask carries.
    pub fn n_live(&self) -> usize {
        self.live.iter().filter(|&&b| b).count()
    }

    /// Flat unknown index per node, `-1` at dead nodes; row-major, same shape as the mask.
    ///
    /// The numbering is C-order over the live positions, which is what `np.nonzero(mask)` yields
    /// and therefore what every live-node vector in the project is indexed by.
    pub fn index_map(&self) -> Vec<i64> {
        let mut map = vec![-1i64; self.live.len()];
        let mut p = 0i64;
        for (slot, &alive) in map.iter_mut().zip(self.live.iter()) {
            if alive {
                *slot = p;
                p += 1;
            }
        }
        map
    }
}

/// Square grid of `n + 1` nodes per axis over `[-half_extent, half_extent]^2`.
///
/// Returns `(x, y, h)` with `x` and `y` flat row-major fields of `(n+1)^2` entries —
/// `x[j*(n+1) + i]` is the x-coordinate of node `(j, i)` — and `h = 2*half_extent/n` the square
/// cell spacing.
///
/// The coordinates reproduce `np.linspace(-half, half, n+1)` operation for operation: NumPy forms
/// `i * step + start` with `step = 2*half/n` and then **overwrites** the last entry with the
/// endpoint, so `coords[n]` is `half` exactly rather than `n * step - half`. Those differ in the
/// last bit for most extents, and the coordinates reach `disk_mask` — where a node within one ulp
/// of the rim is the difference between a live node and a dead one.
///
/// # Panics
/// If `n == 0`.
pub fn grid_coords(n: usize, half_extent: f64) -> (Vec<f64>, Vec<f64>, f64) {
    assert!(n >= 1, "grid_coords needs at least one segment");
    let nodes = n + 1;
    let h = 2.0 * half_extent / (n as f64);
    let mut coords: Vec<f64> = (0..nodes)
        .map(|i| (i as f64) * h + (-half_extent))
        .collect();
    coords[n] = half_extent;

    let mut x = Vec::with_capacity(nodes * nodes);
    let mut y = Vec::with_capacity(nodes * nodes);
    for j in 0..nodes {
        for i in 0..nodes {
            x.push(coords[i]);
            y.push(coords[j]);
        }
    }
    (x, y, h)
}

/// Live-node mask for a rectangle: every interior node of an `(ny+1) x (nx+1)` grid.
///
/// The bounding-box edge nodes are the clamped Dirichlet rim; the `(nx-1) * (ny-1)` interior nodes
/// are the unknowns. The Laplacian built from this mask is exactly the tensor-product 5-point
/// operator, whose `sin·sin` eigenvectors are analytic — the clean O(h^2) reference that de-risks
/// the harness before the staircase error enters.
pub fn rectangle_mask(nx: usize, ny: usize) -> Mask {
    let (nrows, ncols) = (ny + 1, nx + 1);
    let mut live = vec![false; nrows * ncols];
    for j in 1..nrows.saturating_sub(1) {
        for i in 1..ncols.saturating_sub(1) {
            live[j * ncols + i] = true;
        }
    }
    Mask::new(nrows, ncols, live)
}

/// Live-node mask for a disk of `radius` centred at the origin on the grid `(x, y)`.
///
/// A node is live iff `x^2 + y^2 < radius^2` — **strict**, so a node exactly on the rim is
/// boundary. The round rim is *staircased* onto the Cartesian grid, which taxes the Bessel match
/// to ~O(h) while leaving energy conservation exact (the masked operator stays symmetric).
///
/// The comparison is written as NumPy writes it, `(x*x + y*y) < (radius*radius)`, rather than with
/// a `hypot` or a square root: the predicate *is* the geometry, and a different spelling moves
/// nodes across the rim.
///
/// # Panics
/// If `x` and `y` do not have `nrows * ncols` entries each.
pub fn disk_mask(x: &[f64], y: &[f64], radius: f64, nrows: usize, ncols: usize) -> Mask {
    assert_eq!(x.len(), nrows * ncols, "x must be a full node field");
    assert_eq!(y.len(), nrows * ncols, "y must be a full node field");
    let r2 = radius * radius;
    let live = x
        .iter()
        .zip(y.iter())
        .map(|(&xv, &yv)| (xv * xv + yv * yv) < r2)
        .collect();
    Mask::new(nrows, ncols, live)
}

/// One point of the guitar outline's half-width profile, un-normalised, at `t = y/L` in `[0, 1]`.
///
/// `W(t) = sin(pi t) * [1 - waist*cos(4 pi (t - 1/2))] * [1 + asym*(t - 1/2)]`
///
/// `sin(pi t)` closes the outline at both ends; the `cos(4 pi ...)` term puts maxima at the two
/// bouts and the minimum at the **waist**; `asym` widens the lower bout. The shape is defined as
/// `|x| < W(y)`, so it is simply connected and vertically convex by construction.
///
/// # Why this is a scalar and the Python side is a loop
///
/// `f64::sin` here and `math.sin` there are the same call — the platform C library, UCRT on
/// Windows and glibc on Linux. `np.sin` on a float64 array is a *third* implementation: NumPy
/// carries its own vectorised routines and selects one at import from the CPU's features, so what
/// it returns is a property of the machine (plan §22.1). Everywhere else in this module that would
/// be a last bit in a reported number. Here it decides whether a node exists — see `guitar_mask` —
/// so `operators2d.py` gives up the vectorised spelling on the mask path and evaluates this
/// formula one point at a time, in order to be saying the same thing as this function.
///
/// The three factors multiply **left to right**. Reassociating is a different double.
pub fn guitar_half_width(t: f64, waist: f64, asym: f64) -> f64 {
    (t * std::f64::consts::PI).sin()
        * (1.0 - waist * (4.0 * std::f64::consts::PI * (t - 0.5)).cos())
        * (1.0 + asym * (t - 0.5))
}

/// Factor taking `guitar_half_width` to a half-width whose maximum is `width/2`.
///
/// The peak is *sampled* over 20,001 points rather than solved for, so the grid is part of the
/// answer: the `t` values are `np.linspace(0.0, 1.0, 20001)` reproduced operation for operation —
/// `i * step` with `step = 1/20000`, and the last entry overwritten with the endpoint exactly, as
/// NumPy does. One ulp in the result is one ulp on every node of the outline at once, because
/// `scale` multiplies every half-width the mask compares against.
pub fn guitar_scale(width: f64, waist: f64, asym: f64) -> f64 {
    const M: usize = 20_000;
    let step = 1.0 / (M as f64);
    let mut peak = f64::NEG_INFINITY;
    for i in 0..=M {
        let t = if i == M { 1.0 } else { (i as f64) * step };
        let w = guitar_half_width(t, waist, asym);
        if w > peak {
            peak = w;
        }
    }
    0.5 * width / peak
}

/// Live-node mask for a guitar-shaped outline (HANDOFF §12B's non-rectangular plate).
///
/// `x` is measured from the centre line and `y` from the neck end, so the region is
/// `|x| < scale * W(y/length)` with `scale` chosen so the widest point spans `width`. The two end
/// rows (`t = 0`, `t = 1`) are excluded.
///
/// **The result is not yet a usable mask.** A curved outline staircases into nodes that carry no
/// area at all, whose trapezoidal weight is exactly zero and whose presence makes the free plate's
/// mass matrix singular. Pass it through `prune_to_area_carrying`.
///
/// The comparison is `|x| < half`, strict, and it is *the geometry* rather than a number about it.
/// How much room a last bit has here was measured before this was ported: over 130 shipped
/// configurations the smallest margin is ~1.9e7 ulps of `half` for every real guitar — and
/// **1 ulp** for the degenerate lens (`waist = 0, asym = 0`), where four nodes sit mathematically
/// *exactly* on the rim because `sin(pi/6)` is 1/2. That case is why the profile is spelled the
/// way it is; see `guitar_half_width`.
///
/// # Panics
/// If `x`/`y` are not `nrows * ncols` long, or the outline parameters are out of range.
#[allow(clippy::too_many_arguments)]
pub fn guitar_mask(
    x: &[f64],
    y: &[f64],
    length: f64,
    width: f64,
    waist: f64,
    asym: f64,
    nrows: usize,
    ncols: usize,
) -> Mask {
    assert_eq!(x.len(), nrows * ncols, "x must be a full node field");
    assert_eq!(y.len(), nrows * ncols, "y must be a full node field");
    assert!(
        length > 0.0 && width > 0.0,
        "length and width must be positive"
    );
    assert!((0.0..1.0).contains(&waist), "waist must lie in [0, 1)");
    assert!(asym.abs() < 2.0, "|asym| must be < 2");
    let scale = guitar_scale(width, waist, asym);
    let live = y
        .iter()
        .zip(x.iter())
        .map(|(&yv, &xv)| {
            let t = yv / length;
            let half = scale * guitar_half_width(t, waist, asym);
            t > 0.0 && t < 1.0 && xv.abs() < half
        })
        .collect();
    Mask::new(nrows, ncols, live)
}

/// Area of the *true* guitar outline — a fine midpoint quadrature of `2 W(y)`.
///
/// The denominator of the area deficit a guitar plate reports, and the one number in this group
/// that is **not** bit-identical across the port, by decision rather than by accident.
///
/// Two million midpoints are summed here left to right and by `np.sum` on the Python side, which
/// is *pairwise* above a blocksize of 128 — a different number in the last few bits, and one that
/// no portable loop reproduces. Matching it would mean transcribing NumPy's blocking, which is the
/// same bargain §18.2 refused for SciPy's sparse-product kernel: a claim about a library internal
/// that a point release may change. So this is a tolerance-level quantity and is measured as one.
/// It can afford to be: nothing branches on it and it reaches no timestep — it is divided into a
/// mask area to report how converged a staircase is (plan §19.2's question, answered "no").
pub fn guitar_area(length: f64, width: f64, waist: f64, asym: f64) -> f64 {
    const M: usize = 2_000_000;
    let scale = guitar_scale(width, waist, asym);
    let mut acc = 0.0;
    for i in 0..M {
        let t = ((i as f64) + 0.5) / (M as f64);
        acc += scale * guitar_half_width(t, waist, asym);
    }
    2.0 * acc * (length / (M as f64))
}

/// Cells of the dual grid whose **four** corner nodes are all live, row-major, `(nrows-1) x
/// (ncols-1)`.
///
/// The quadrature cells of the free-plate energy: the twist `u_xy` is evaluated on them, and a
/// node's area weight counts them. Empty when the mask is thinner than two nodes on either axis.
pub fn live_cells(mask: &Mask) -> Vec<bool> {
    let (nrows, ncols) = (mask.nrows(), mask.ncols());
    let (cr, cc) = (nrows.saturating_sub(1), ncols.saturating_sub(1));
    let mut out = Vec::with_capacity(cr * cc);
    for j in 0..cr {
        for i in 0..cc {
            out.push(
                mask.at(j, i) && mask.at(j + 1, i) && mask.at(j, i + 1) && mask.at(j + 1, i + 1),
            );
        }
    }
    out
}

/// Number of live cells (0..=4) touching each node — `4` interior, `2` edge, `1` corner.
pub fn cells_per_node(mask: &Mask) -> Vec<i64> {
    let (nrows, ncols) = (mask.nrows(), mask.ncols());
    let (cr, cc) = (nrows.saturating_sub(1), ncols.saturating_sub(1));
    let cells = live_cells(mask);
    let mut out = vec![0i64; nrows * ncols];
    for j in 0..cr {
        for i in 0..cc {
            if cells[j * cc + i] {
                out[j * ncols + i] += 1;
                out[j * ncols + i + 1] += 1;
                out[(j + 1) * ncols + i] += 1;
                out[(j + 1) * ncols + i + 1] += 1;
            }
        }
    }
    out
}

/// Drop live nodes that touch **no** live cell, to a fixed point. Returns `(mask, n_dropped)`.
///
/// **The mask is not the outline.** The outline is a predicate on coordinates; the mask is the set
/// of nodes that carry *area*. A curved rim staircases into one-node spikes whose trapezoidal area
/// weight is exactly `0`, and those make the free plate's mass matrix `W` singular — two are
/// enough, and the shipped guitar produces 2-4 at every grid tried.
///
/// Dropping a node can orphan its neighbour, so this iterates to a fixed point. One sweep sufficed
/// everywhere measured; the loop is the correct statement rather than an optimisation.
///
/// **A silent geometry change, and callers must price it.** The rule is purely topological — it
/// says nothing about *where* the node was — so on a coarse grid with a deep waist it can fire in
/// the middle of the plate rather than at a tip, and energy, nullspace and spectrum all look
/// healthy afterwards. The plate asserts every dropped node lay within one `h` of the outline.
pub fn prune_to_area_carrying(mask: &Mask) -> (Mask, usize) {
    let (nrows, ncols) = (mask.nrows(), mask.ncols());
    let mut live = mask.flags().to_vec();
    let before = live.iter().filter(|&&b| b).count();
    loop {
        let current = Mask::new(nrows, ncols, live.clone());
        let touching = cells_per_node(&current);
        let keep: Vec<bool> = live
            .iter()
            .zip(touching.iter())
            .map(|(&alive, &n)| alive && n > 0)
            .collect();
        if keep == live {
            let after = live.iter().filter(|&&b| b).count();
            return (Mask::new(nrows, ncols, live), before - after);
        }
        live = keep;
    }
}

/// Symmetric 5-point Laplacian on the live nodes of `mask`, plus the mask's `index_map`.
///
/// The matrix is `(n_live x n_live)` with `-4/h^2` on the diagonal and `+1/h^2` for each in-domain
/// neighbour (up/down/left/right). A neighbour that is not live simply drops — its `u = 0` ghost
/// contributes nothing — so the result is a principal submatrix of the symmetric full-grid
/// Laplacian, and is therefore **symmetric negative-definite**. That symmetry is what makes the
/// membrane's energy identity exact on a staircased rim as well as on a rectangle.
///
/// The stored values are written `-4.0 * inv_h2` and `inv_h2` with `inv_h2 = 1.0 / (h * h)`,
/// matching the original's spelling rather than its algebra: `-4.0 / (h * h)` is a different
/// rounding and the difference would reach every timestep.
pub fn laplacian_from_mask(mask: &Mask, h: f64) -> (Csr, Vec<i64>) {
    let index_map = mask.index_map();
    let n_live = mask.n_live();
    let (nrows, ncols) = (mask.nrows(), mask.ncols());
    let inv_h2 = 1.0 / (h * h);
    let diag = -4.0 * inv_h2;

    let mut rows: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n_live);
    for j in 0..nrows {
        for i in 0..ncols {
            if !mask.at(j, i) {
                continue;
            }
            let mut row = vec![(index_map[j * ncols + i] as usize, diag)];
            for (dj, di) in [(1i64, 0i64), (-1, 0), (0, 1), (0, -1)] {
                let nj = j as i64 + dj;
                let ni = i as i64 + di;
                if nj < 0 || nj >= nrows as i64 || ni < 0 || ni >= ncols as i64 {
                    continue;
                }
                let (nj, ni) = (nj as usize, ni as usize);
                if mask.at(nj, ni) {
                    row.push((index_map[nj * ncols + ni] as usize, inv_h2));
                }
            }
            rows.push(row);
        }
    }
    (Csr::from_rows(n_live, n_live, rows), index_map)
}

/// Symmetric 2-D biharmonic `∇⁴ = (∇²)²` on the live nodes, built as `B = L @ L`.
///
/// The plate's flexural operator (HANDOFF §5 model #5). `L` is the *Dirichlet* (zero-ghost)
/// Laplacian above, so `w = L u` already vanishes on the rim and applying `L` twice enforces both
/// simply-supported conditions with no hand-coded 13-point boundary rows.
///
/// **This is where the port and the reference part company on storage order, and only there.**
/// SciPy's sparse product returns each row in the order its kernel happened to touch the columns —
/// neither ascending nor descending, and not a property of the algebra. The values agree to the
/// bit (measured at seven grids and two guitar outlines, 0 differing entries out of 2,629), so the
/// disagreement is entirely about *order* — and a CSR matvec sums a row in stored order, which
/// makes `B @ u` a different sum on the two sides. `plate.py` forms `B @ u` twice per timestep, so
/// the fix is `physsynth.core.portable.canonical` on the Python side, exactly as §18.2 did for the
/// string family: the sorted order is the one both languages can express, and the one SciPy itself
/// calls canonical.
pub fn biharmonic_from_mask(mask: &Mask, h: f64) -> (Csr, Vec<i64>) {
    let (l, index_map) = laplacian_from_mask(mask, h);
    (l.matmul(&l), index_map)
}

/// `n_int x n_int` second difference `[1, -2, 1]/h²` on the *interior* nodes of a segment.
///
/// The 1-D Dirichlet operator whose eigenvectors are exactly `sin(m pi x / L)` sampled at the
/// interior nodes. The two end rows have one neighbour each because the rim nodes are not unknowns
/// at all — distinct from a full-grid operator, which keeps them.
///
/// Spelled `-2.0 * inv_h2` and `inv_h2` with `inv_h2 = 1.0 / (h * h)`, as `laplacian_from_mask` is
/// and for the same reason: `-2.0 / (h * h)` is a different rounding.
pub fn dirichlet_interior_d2_1d(n_int: usize, h: f64) -> Csr {
    let inv_h2 = 1.0 / (h * h);
    let main = -2.0 * inv_h2;
    let rows = (0..n_int)
        .map(|i| {
            let mut row = vec![(i, main)];
            if i > 0 {
                row.push((i - 1, inv_h2));
            }
            if i + 1 < n_int {
                row.push((i + 1, inv_h2));
            }
            row
        })
        .collect();
    Csr::from_rows(n_int, n_int, rows)
}

/// Orthotropic (grain-direction) bending operator on a simply-supported rectangle.
///
/// `B = g_x (d_xx)^2 + 2 g_h (d_xx d_yy) + g_y (d_yy)^2`, the discrete form of
/// `D_x w_xxxx + 2H w_xxyy + D_y w_yyyy` over a reference rigidity, so the three grain arguments
/// are dimensionless ratios. **The factor of 2 belongs on the cross term here, in the operator,
/// not inside `H`** — the two rival packagings the orthotropic literature invites both produce a
/// perfectly stable, exactly energy-conserving, wrong plate.
///
/// Live nodes are walked in C order (y outer, x inner), so `x` is the *inner* tensor factor.
///
/// Two orders are load-bearing and both are spelled out rather than left to an optimiser: the
/// three terms are summed **left to right**, `(t1 + t2) + t3`, which is what Python's `+` chain
/// does, and each scalar multiplies the *assembled* product rather than being folded into a
/// factor. Note also what this deliberately does **not** do: at `g_x = g_h = g_y` it is `g · L @ L`
/// in exact arithmetic but only *grid-dependently* so in doubles, which is why `plate.py` keeps
/// the isotropic default on the squared-Laplacian path instead of routing everything through here.
///
/// Definiteness is a condition, not a freebie (`g_h > -sqrt(g_x g_y)`), and this does not enforce
/// it — so a test can build the indefinite case.
///
/// # Panics
/// If `nx` or `ny` is below 2.
pub fn orthotropic_biharmonic(
    nx: usize,
    ny: usize,
    h: f64,
    grain_x: f64,
    grain_cross: f64,
    grain_y: f64,
) -> (Csr, Vec<i64>) {
    assert!(nx >= 2 && ny >= 2, "nx and ny must both be >= 2");
    let mask = rectangle_mask(nx, ny);
    let index_map = mask.index_map();
    let dxx = Csr::identity(ny - 1).kron(&dirichlet_interior_d2_1d(nx - 1, h));
    let dyy = dirichlet_interior_d2_1d(ny - 1, h).kron(&Csr::identity(nx - 1));
    let b = dxx
        .matmul(&dxx)
        .scaled(grain_x)
        .add(&dxx.matmul(&dyy).scaled(2.0 * grain_cross))
        .add(&dyy.matmul(&dyy).scaled(grain_y));
    (b, index_map)
}

/// Energy-first free-edge Kirchhoff-plate bending operator on the live nodes of `mask` — model #5g.
///
/// Returns `(K, W, index_map)`: the symmetric positive-*semi*definite stiffness, the diagonal
/// lumped area weight, and the live-node map. The bilinear form is
///
/// ```text
/// P(f, g) = ∫∫ [ D_x f_xx g_xx + D_y f_yy g_yy + D_1 (f_xx g_yy + f_yy g_xx) + 4 D_xy f_xy g_xy ]
/// ```
///
/// assembled **from the energy**, so symmetry, the natural free-edge conditions and the rigid-body
/// nullspace `{1, x, y}` all fall out by construction rather than from ghost-point elimination on
/// a 13-point stencil. `grain_coupling` / `grain_torsion` default to the `nu`-derived isotropic
/// split `(nu, (1-nu)/2)`, at which the four coefficients are `1`, `1`, `nu` and an
/// exactly-representable halving — bit-identical to the isotropic assembly on every grid.
///
/// Each rectangle rule is the special case of a rule about what is *live*: curvature is centred at
/// a node iff **both** neighbours along that axis are live (a zeroed end row was never a statement
/// about index 0, it was a statement about a missing neighbour), the twist gets one row per live
/// cell, and the area weight is `h² · (live cells touching the node) / 4` — which *is* the
/// trapezoidal weight, three cases collapsed into one expression.
///
/// **The mask must already carry area** (`prune_to_area_carrying`), or a one-node spike gives `W`
/// a zero on the diagonal and the plate's time-step matrix is singular.
///
/// Three spellings here are arithmetic rather than algebra, and each was wrong once or could be:
///
/// * the twist coefficient is `(1/h) * (1/h)`, **not** `1/(h*h)` — this operator is a product of
///   two forward first differences, and the two differ in the last digit whenever `h` is not
///   exactly representable. It showed up on exactly one grid of the seven-grid survey, so checking
///   one grid would have reported success.
/// * the Gram products are **right**-associated, `C2xᵀ @ (Wa @ C2y)`, because that is how
///   `operators2d.py` parenthesises them. Its sibling `AiryStressSolver` writes the same
///   mathematical form `BᵀWB` with **no** parentheses, which Python left-associates — so those two
///   are different matrices in the last bit, and a shared helper would silently pick one of them.
/// * the four terms are summed left to right, `((t1 + t2) + t3) + t4`.
///
/// # Panics
/// If the mask has no live nodes, or `nu` is outside `(-1, 1/2)` where it is actually *used*, i.e.
/// where it supplies a missing half of the split. An orthotropic plate's implied `nu_yx` may
/// legitimately exceed 1/2, so applying the isotropic range to a superseded argument would reject
/// a valid material — and did, once.
#[allow(clippy::too_many_arguments)]
pub fn free_plate_stiffness_from_mask(
    mask: &Mask,
    h: f64,
    nu: f64,
    grain_x: f64,
    grain_y: f64,
    grain_coupling: Option<f64>,
    grain_torsion: Option<f64>,
) -> (Csr, Csr, Vec<i64>) {
    if grain_coupling.is_none() || grain_torsion.is_none() {
        assert!(
            -1.0 < nu && nu < 0.5,
            "nu (Poisson's ratio) must be in (-1, 1/2)"
        );
    }
    let g_1 = grain_coupling.unwrap_or(nu);
    let g_xy = grain_torsion.unwrap_or(0.5 * (1.0 - nu));

    let (nrows, ncols) = (mask.nrows(), mask.ncols());
    let n_live = mask.n_live();
    assert!(n_live >= 1, "the mask has no live nodes");
    let index_map = mask.index_map();
    let inv_h2 = 1.0 / (h * h);
    let main = -2.0 * inv_h2;

    // `[1, -2, 1]/h²` centred at each node with both `(dj, di)` neighbours live; a node without
    // them contributes an empty row, not a zero one.
    let curvature = |dj: i64, di: i64| -> Csr {
        let mut rows: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n_live);
        for j in 0..nrows {
            for i in 0..ncols {
                if !mask.at(j, i) {
                    continue;
                }
                let (jm, im) = (j as i64 - dj, i as i64 - di);
                let (jp, ip) = (j as i64 + dj, i as i64 + di);
                let inside = jm >= 0
                    && im >= 0
                    && jp < nrows as i64
                    && ip < ncols as i64
                    && mask.at(jm as usize, im as usize)
                    && mask.at(jp as usize, ip as usize);
                if !inside {
                    rows.push(Vec::new());
                    continue;
                }
                rows.push(vec![
                    (
                        index_map[jm as usize * ncols + im as usize] as usize,
                        inv_h2,
                    ),
                    (index_map[j * ncols + i] as usize, main),
                    (
                        index_map[jp as usize * ncols + ip as usize] as usize,
                        inv_h2,
                    ),
                ]);
            }
        }
        Csr::from_rows(n_live, n_live, rows)
    };
    let c2x = curvature(0, 1);
    let c2y = curvature(1, 0);

    // Cell-centred twist on the live cells. The coefficient is (1/h)*(1/h) and NOT 1/(h*h).
    let cells = live_cells(mask);
    let (cr, cc) = (nrows.saturating_sub(1), ncols.saturating_sub(1));
    let d1 = 1.0 / h;
    let twist = d1 * d1;
    let mut cell_rows: Vec<Vec<(usize, f64)>> = Vec::new();
    for j in 0..cr {
        for i in 0..cc {
            if !cells[j * cc + i] {
                continue;
            }
            cell_rows.push(vec![
                (index_map[j * ncols + i] as usize, twist),
                (index_map[j * ncols + i + 1] as usize, -twist),
                (index_map[(j + 1) * ncols + i] as usize, -twist),
                (index_map[(j + 1) * ncols + i + 1] as usize, twist),
            ]);
        }
    }
    let dxy = Csr::from_rows(cell_rows.len(), n_live, cell_rows);

    // Area weight: h² * (live cells touching the node)/4 — exactly h², h²/2, h²/4 on a rectangle.
    let touching = cells_per_node(mask);
    let wdiag: Vec<f64> = (0..nrows * ncols)
        .filter(|&p| index_map[p] >= 0)
        .map(|p| (h * h) * ((touching[p] as f64) * 0.25))
        .collect();
    let wa = Csr::diagonal(&wdiag);

    let cross = c2x.transpose().matmul(&wa.matmul(&c2y));
    let k = c2x
        .transpose()
        .matmul(&wa.matmul(&c2x))
        .scaled(grain_x)
        .add(&c2y.transpose().matmul(&wa.matmul(&c2y)).scaled(grain_y))
        .add(&cross.add(&cross.transpose()).scaled(g_1))
        .add(&dxy.transpose().matmul(&dxy).scaled((4.0 * g_xy) * (h * h)));
    (k, wa, index_map)
}

/// `free_plate_stiffness_from_mask` on a full `(ny+1) x (nx+1)` bounding box.
///
/// A full box **is** the rectangle: every node is a free unknown, every cell is live, and the
/// general routine's "live adjacent cells / 4" area rule evaluates to exactly the trapezoidal
/// weight it replaced. One code path, not two.
///
/// # Panics
/// If `nx` or `ny` is below 2 — at least one interior node per axis is needed.
#[allow(clippy::too_many_arguments)]
pub fn free_plate_stiffness(
    nx: usize,
    ny: usize,
    h: f64,
    nu: f64,
    grain_x: f64,
    grain_y: f64,
    grain_coupling: Option<f64>,
    grain_torsion: Option<f64>,
) -> (Csr, Csr, Vec<i64>) {
    assert!(
        nx >= 2 && ny >= 2,
        "nx, ny must be >= 2 (need at least one interior node per axis)"
    );
    let mask = Mask::new(ny + 1, nx + 1, vec![true; (ny + 1) * (nx + 1)]);
    free_plate_stiffness_from_mask(
        &mask,
        h,
        nu,
        grain_x,
        grain_y,
        grain_coupling,
        grain_torsion,
    )
}

// --- the nonlinear plate's 1-D differences ------------------------------------------------------
//
// Five one-axis operators that `VonKarmanBracket` and `AiryStressSolver` lift onto the grid with
// `kron`. They are private in the Python module and public here, for the reason
// `dirichlet_interior_d2_1d` is: the parity test compares them one at a time, and comparing them
// one at a time is the only way to tell an assembly mistake from an ordering one.
//
// Every one of them is a *fixed pattern of exact scalings of one reciprocal*, so all five are
// bit-identical to SciPy's whatever the grid: `inv_h2 = 1.0 / (h * h)` rounds once and `-2.0 *`,
// `2.0 *` and the sign flips are exact. What is NOT free is spelling the reciprocal the same way
// -- `-2.0 / (h * h)` is a different rounding from `-2.0 * (1.0 / (h * h))`, which is why every
// builder in this module writes the second.

/// `(n+1) x (n+1)` collocated second difference `[1, -2, 1]/h^2` with **empty end rows**.
///
/// Row `l` for `l = 1 .. n-1` is the curvature centred at node `l`; rows `0` and `n` carry no
/// entries at all -- not a stored zero, *nothing* -- because the free beam evaluates curvature at
/// interior nodes only. The distinction is visible in `nnz` and the parity test asserts it.
///
/// Annihilates linear data exactly, which is the free plate's `{1, x, y}` nullspace in one axis.
///
/// This has no caller left in the reference: `free_plate_stiffness` builds its curvature from the
/// mask now, and `tests/test_free_plate_modal.py` keeps this as an independent oracle for it. It
/// is ported anyway, because an oracle that is not compared is an oracle nobody is checking.
pub fn collocated_d2_1d(n: usize, h: f64) -> Csr {
    let inv_h2 = 1.0 / (h * h);
    let main = -2.0 * inv_h2;
    let rows = (0..=n)
        .map(|l| {
            if l == 0 || l == n {
                Vec::new()
            } else {
                vec![(l - 1, inv_h2), (l, main), (l + 1, inv_h2)]
            }
        })
        .collect();
    Csr::from_rows(n + 1, n + 1, rows)
}

/// `n x (n+1)` forward first difference: row `i` is `(u[i+1] - u[i])/h`, living on cell `i`.
///
/// The dual-grid difference whose tensor square is the **cell-centred** twist. That choice is
/// physics, not taste: the collocated centred mixed difference has a checkerboard `(-1)^(i+j)`
/// nullspace, which injects spurious near-zero modes into the low plate spectrum.
///
/// Spelled `-1.0 / h` and `1.0 / h`, two divisions, as the original writes them. They round to
/// exact negatives of each other, so nothing turns on it here -- but the *product* of two of these
/// is what `kron` forms for the twist, and `(1/h) * (1/h)` is not `1/(h*h)`; that one has already
/// been wrong once in this module.
pub fn forward_d1_1d(n: usize, h: f64) -> Csr {
    let rows = (0..n)
        .map(|i| vec![(i, -1.0 / h), (i + 1, 1.0 / h)])
        .collect();
    Csr::from_rows(n, n + 1, rows)
}

/// `(n+1) x (n+1)` ordinary tridiagonal second difference `[1, -2, 1]/h^2` at **every** node.
///
/// Unlike `collocated_d2_1d`, the two end rows are kept: they hold the one-sided Dirichlet-ghost
/// curvature `(u[1] - 2u[0])/h^2`. This is the `d_xx` of the von Karman bracket's straight terms;
/// for a field vanishing on the rim those end values meet a zero test field and drop out of the
/// trilinear form, so only the interior curvature is ever weighed.
///
/// # Panics
/// If `n` is 0 -- a one-node axis has no difference on it.
pub fn centered_d2_1d(n: usize, h: f64) -> Csr {
    assert!(n >= 1, "centered_d2_1d needs n >= 1");
    let inv_h2 = 1.0 / (h * h);
    let main = -2.0 * inv_h2;
    let rows = (0..=n)
        .map(|l| {
            let mut row = Vec::with_capacity(3);
            if l > 0 {
                row.push((l - 1, inv_h2));
            }
            row.push((l, main));
            if l < n {
                row.push((l + 1, inv_h2));
            }
            row
        })
        .collect();
    Csr::from_rows(n + 1, n + 1, rows)
}

/// `(n+1) x (n+1)` second difference with the **clamped ghost mirror** at both ends.
///
/// Interior rows are `centered_d2_1d`'s; the two end rows **double** their single off-diagonal.
/// The clamped edge `F,n = 0` gives the mirror `F_{-1} = F_1`, so the boundary-node curvature is
/// `(F_1 - 2F_0 + F_{-1})/h^2 = (2F_1 - 2F_0)/h^2` and row 0 is `[-2, 2, 0, ...]/h^2`.
///
/// The matrix is **not** symmetric -- the end rows are one-sided -- but the Gram form `Lc^T Wa Lc`
/// with the trapezoidal area weight is, and it reproduces the textbook clamped-plate biharmonic
/// exactly: near-boundary diagonal `7`, interior `6`, off-diagonals `-4` and `1`. With `Wa = I`
/// the `7` comes out `9`, a different and wrong operator, so the weight is load-bearing.
///
/// The reference builds this through `lil` and two scalar assignments, which **replace** the
/// end off-diagonals rather than adding to them. `2.0 * inv_h2` here, not `inv_h2 + inv_h2`;
/// they agree, but only one of them is what the original says.
///
/// # Panics
/// If `n` is below 2 -- the mirror needs a distinct interior neighbour at each end.
pub fn clamped_d2_1d(n: usize, h: f64) -> Csr {
    assert!(n >= 2, "clamped_d2_1d needs n >= 2");
    let inv_h2 = 1.0 / (h * h);
    let main = -2.0 * inv_h2;
    let mirror = 2.0 * inv_h2;
    let rows = (0..=n)
        .map(|l| {
            if l == 0 {
                vec![(0, main), (1, mirror)]
            } else if l == n {
                vec![(n - 1, mirror), (n, main)]
            } else {
                vec![(l - 1, inv_h2), (l, main), (l + 1, inv_h2)]
            }
        })
        .collect();
    Csr::from_rows(n + 1, n + 1, rows)
}

/// `n x (n+1)` node-to-cell average `(u[i] + u[i+1])/2` on cell `i` -- `forward_d1_1d`'s partner.
///
/// Carries no `h`. Its tensor product maps a node field to the 0.25-weighted average of a cell's
/// four corners, and the **adjoint** of that scatters a cell-centred quantity back onto nodes --
/// the step that makes the bracket's trilinear form exactly triple self-adjoint.
pub fn avg_d1_1d(n: usize) -> Csr {
    let rows = (0..n).map(|i| vec![(i, 0.5), (i + 1, 0.5)]).collect();
    Csr::from_rows(n, n + 1, rows)
}

/// The discrete von Karman / Monge-Ampere bracket `l(a, b)` on the full `(nx+1) x (ny+1)` grid.
///
/// `L(a, b) = a_xx b_yy + a_yy b_xx - 2 a_xy b_xy` -- the nonlinear coupling of the Foppl-von
/// Karman plate (HANDOFF section 5 model #6). `l(w, w)` sources the Airy stress function and
/// `l(w, F)` is the membrane restoring force. The property the whole conservative scheme rests on
/// is that the trilinear form `T(a, b, c) = <l(a, b), c>` is symmetric under **any** permutation
/// of its three arguments, to machine precision, for fields vanishing on the rim.
///
/// That symmetry is not free and the naive collocated bracket does not have it. It appears only
/// when the twist term is discretised on **cell centres** (`forward_d1_1d` tensored with itself)
/// and its product averaged back to nodes by the adjoint of the corner average (`avg_d1_1d`):
///
/// ```text
/// l(a, b) = (d_xx a)(d_yy b) + (d_yy a)(d_xx b)  -  2 * A^T[ (D_xy a)(D_xy b) ]
/// ```
///
/// **Domain requirement, not a bug.** The cancellation is a summation-by-parts identity with no
/// leftover boundary term only when the fields are zero on the bounding-box rim. Callers pass
/// full-grid vectors of length `(nx+1)(ny+1)` with the rim held at zero.
///
/// # Every matvec here is a canonical row gather, and one of them had to be checked
///
/// `Sxx`, `Syy` and `Dxy` are used as SciPy uses them, `M @ v`, and all three come out of `kron`
/// ascending. `Acell` is used **transposed** -- and in SciPy `csr.T` is a *CSC*, whose matvec
/// scatters columns rather than gathering rows. Those are different orders in general, but not
/// here, and the reason is worth one line because it is a lemma rather than a measurement: a CSC
/// matvec accumulates each output entry over increasing column index, and a sorted-CSR row gather
/// accumulates over increasing column index, so the two coincide for every canonically-stored
/// matrix. Measured 0 differing entries in 21,780 at four grids, as predicted (plan section 27.3).
#[derive(Debug, Clone)]
pub struct VonKarmanBracket {
    nx: usize,
    ny: usize,
    h: f64,
    n_nodes: usize,
    sxx: Csr,
    syy: Csr,
    dxy: Csr,
    acell: Csr,
    acell_t: Csr,
}

impl VonKarmanBracket {
    /// Build the four operators on an `(nx+1) x (ny+1)` node grid of spacing `h`.
    ///
    /// # Panics
    /// If `nx` or `ny` is below 2, or `h` is not positive -- the two refusals the original raises.
    pub fn new(nx: usize, ny: usize, h: f64) -> Self {
        assert!(
            nx >= 2 && ny >= 2,
            "Nx, Ny must be >= 2 (need at least one interior node per axis)."
        );
        assert!(h > 0.0, "h (grid spacing) must be positive.");
        let ix = Csr::identity(nx + 1);
        let iy = Csr::identity(ny + 1);
        // C order: y is the outer tensor factor, x the inner.
        let sxx = iy.kron(&centered_d2_1d(nx, h));
        let syy = centered_d2_1d(ny, h).kron(&ix);
        let dxy = forward_d1_1d(ny, h).kron(&forward_d1_1d(nx, h));
        let acell = avg_d1_1d(ny).kron(&avg_d1_1d(nx));
        let acell_t = acell.transpose();
        VonKarmanBracket {
            nx,
            ny,
            h,
            n_nodes: (nx + 1) * (ny + 1),
            sxx,
            syy,
            dxy,
            acell,
            acell_t,
        }
    }

    /// Segments along x.
    pub fn nx(&self) -> usize {
        self.nx
    }

    /// Segments along y.
    pub fn ny(&self) -> usize {
        self.ny
    }

    /// Grid spacing.
    pub fn h(&self) -> f64 {
        self.h
    }

    /// `(nx+1)(ny+1)` -- the length of every vector this type takes and returns.
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// `d_xx` lifted onto the grid.
    pub fn sxx(&self) -> &Csr {
        &self.sxx
    }

    /// `d_yy` lifted onto the grid.
    pub fn syy(&self) -> &Csr {
        &self.syy
    }

    /// The cell-centred twist `D_xy`, one row per cell.
    pub fn dxy(&self) -> &Csr {
        &self.dxy
    }

    /// The node-to-cell corner average `A`, one row per cell.
    pub fn acell(&self) -> &Csr {
        &self.acell
    }

    /// The nodal field `l(a, b)`, symmetric in its two arguments by construction.
    ///
    /// The assembly order is the original's, spelled out because it is arithmetic: the two
    /// straight products are summed left to right and the twist is subtracted after being doubled,
    /// `(sxx_a * syy_b + syy_a * sxx_b) - 2.0 * twist`.
    ///
    /// # Panics
    /// If `a` or `b` is not a full-grid vector.
    pub fn eval(&self, a: &[f64], b: &[f64]) -> Vec<f64> {
        assert_eq!(a.len(), self.n_nodes, "a must be a full-grid vector");
        assert_eq!(b.len(), self.n_nodes, "b must be a full-grid vector");
        let sxx_a = self.sxx.matvec(a);
        let syy_b = self.syy.matvec(b);
        let syy_a = self.syy.matvec(a);
        let sxx_b = self.sxx.matvec(b);
        let dxy_a = self.dxy.matvec(a);
        let dxy_b = self.dxy.matvec(b);
        let cell: Vec<f64> = dxy_a.iter().zip(dxy_b.iter()).map(|(p, q)| p * q).collect();
        let twist = self.acell_t.matvec(&cell);
        (0..self.n_nodes)
            .map(|i| (sxx_a[i] * syy_b[i] + syy_a[i] * sxx_b[i]) - 2.0 * twist[i])
            .collect()
    }

    /// The trilinear form `T(a, b, c) = <l(a, b), c> = h^2 sum l(a, b) c`.
    ///
    /// Triple self-adjoint to machine precision **iff** the fields vanish on the rim. This is the
    /// one quantity here whose parity claim is a tolerance rather than an equality, and the reason
    /// is `inner2d`: NumPy contracts it with `np.dot`, which is BLAS, and section 14.2 settled
    /// that matching a BLAS reduction would be a claim about a runner.
    pub fn trilinear(&self, a: &[f64], b: &[f64], c: &[f64]) -> f64 {
        inner2d(&self.eval(a, b), c, self.h)
    }
}

/// Elliptic solve for the von Karman **Airy stress function** `F` -- model #6, Part 2.
///
/// Solves `lap^2 F = source` on a rectangular grid with the **clamped** in-plane condition
/// `F = 0, F,n = 0`, which is the physically correct movable-edge condition for a simply-supported
/// plate. It is deliberately *not* the `B = L^2` Navier operator of `biharmonic_from_mask`.
///
/// Built from the energy: the membrane energy is `(1/2Ee)*||lap F||^2`, and for clamped edges that
/// norm needs no mixed term, so the operator is a single Laplacian squared --
///
/// ```text
/// B_F = Lc_r^T Wa Lc_r
/// ```
///
/// with `Lc` the full-grid Laplacian built from `clamped_d2_1d`, `Wa` the trapezoidal area weight
/// (`h^2` interior, `h^2/2` edge, `h^2/4` corner) and `Lc_r` the rim **columns** dropped, all rows
/// kept. Symmetric positive definite by construction -- clamping leaves no rigid-body mode, so
/// unlike the free plate's `{1, x, y}` the nullspace here is empty -- and factored once.
///
/// # The association is the whole arithmetic story, and it is not the multiplication
///
/// The reference writes `Lc_r.T @ Wa @ Lc_r`, which Python left-associates. Ask the two questions
/// section 26.2 separated and they come apart cleanly:
///
/// * **Values.** Every entry of `Lc` is `{1, 2, 4}` times one reciprocal and every entry of `Wa`
///   is `{1, 1/2, 1/4}` times one product, so each *term* `A*W*B` has the same two outer mantissas
///   whichever way it is bracketed, and `fl(fl(aw)a) = fl(a*fl(wa))` by commutativity. Measured
///   over five grids: not one term differs. Section 26.5's rule holds.
/// * **Order.** And it settles nothing, because the association does not move the products, it
///   moves the **sum**. SciPy hands back `Lc_r^T @ Wa` with every row *descending*, so the outer
///   product contracts the shared index in descending order; the right-associated spelling
///   contracts it ascending. Those are different sums on 2 of the 22 grids the test suite builds,
///   up to 46 entries of 1,889.
///
/// So the fix is a pair of parentheses rather than a canonicaliser: the reference now writes
/// `Lc_r^T @ (Wa @ Lc_r)`, whose contraction runs over `Lc_r^T`'s own ascending rows, and this
/// module spells the same association. That is a *third* remedy for an ordering problem, after
/// "reproduce the values" and "sort the storage" (plan section 27.2).
#[derive(Debug, Clone)]
pub struct AiryStressSolver {
    nx: usize,
    ny: usize,
    h: f64,
    n_nodes: usize,
    n_interior: usize,
    mask: Mask,
    index_map: Vec<i64>,
    bf: Csr,
    load_weight: Vec<f64>,
    lu: SparseLu,
}

impl AiryStressSolver {
    /// Assemble and factor `B_F` on an `(nx+1) x (ny+1)` node grid of spacing `h`.
    ///
    /// # Errors
    /// If the assembled operator has no admissible pivot -- which for an SPD matrix means the
    /// caller built something that is not one.
    ///
    /// # Panics
    /// If `nx` or `ny` is below 2, or `h` is not positive.
    pub fn new(nx: usize, ny: usize, h: f64) -> Result<Self, SparseLuError> {
        assert!(
            nx >= 2 && ny >= 2,
            "Nx, Ny must be >= 2 (need at least one interior node per axis)."
        );
        assert!(h > 0.0, "h (grid spacing) must be positive.");
        let mask = rectangle_mask(nx, ny);
        let index_map = mask.index_map();
        let n_interior = mask.n_live();
        let n_nodes = (nx + 1) * (ny + 1);

        let ix = Csr::identity(nx + 1);
        let iy = Csr::identity(ny + 1);
        let lc = iy
            .kron(&clamped_d2_1d(nx, h))
            .add(&clamped_d2_1d(ny, h).kron(&ix));

        // Trapezoidal area weight Wa = kron(m_y, m_x): h^2 interior, h^2/2 edge, h^2/4 corner. The
        // halvings are exact, so every entry shares the mantissa of `h * h` -- the fact the
        // association argument above turns on.
        let mut mx = vec![h; nx + 1];
        mx[0] = 0.5 * h;
        mx[nx] = 0.5 * h;
        let mut my = vec![h; ny + 1];
        my[0] = 0.5 * h;
        my[ny] = 0.5 * h;
        let mut wa_diag = Vec::with_capacity(n_nodes);
        for wy in &my {
            for wx in &mx {
                wa_diag.push(wy * wx);
            }
        }
        let wa = Csr::diagonal(&wa_diag);

        let keep = mask.flags();
        let lc_r = lc.select_columns(keep);
        // Right-associated, deliberately: see the type docstring. `Lc_r^T`'s rows are ascending
        // (the transpose is a counting sort and the column restriction is monotone), so this
        // contracts the shared index in ascending order -- the one order both languages can say.
        let bf = lc_r.transpose().matmul(&wa.matmul(&lc_r));

        let load_weight: Vec<f64> = wa_diag
            .iter()
            .zip(keep.iter())
            .filter(|(_, &k)| k)
            .map(|(w, _)| *w)
            .collect();
        let lu = SparseLu::factor(&bf)?;
        Ok(AiryStressSolver {
            nx,
            ny,
            h,
            n_nodes,
            n_interior,
            mask,
            index_map,
            bf,
            load_weight,
            lu,
        })
    }

    /// Segments along x.
    pub fn nx(&self) -> usize {
        self.nx
    }

    /// Segments along y.
    pub fn ny(&self) -> usize {
        self.ny
    }

    /// Grid spacing.
    pub fn h(&self) -> f64 {
        self.h
    }

    /// `(nx+1)(ny+1)` -- the length of every full-grid vector here.
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// The number of interior unknowns.
    pub fn n_interior(&self) -> usize {
        self.n_interior
    }

    /// The interior mask -- `F` is held at zero on the bounding-box rim.
    pub fn mask(&self) -> &Mask {
        &self.mask
    }

    /// Full-grid node to interior-unknown index, `-1` on the rim.
    pub fn index_map(&self) -> &[i64] {
        &self.index_map
    }

    /// The assembled SPD operator `B_F`, on the interior unknowns.
    pub fn bf(&self) -> &Csr {
        &self.bf
    }

    /// Solve `lap^2 F = source`; both vectors are full-grid with the rim at zero.
    ///
    /// The interior load is `Wa`-weighted here, because `Wa` lives *inside* `B_F` -- the caller
    /// passes the physical source and never has to remember the quadrature weight. Forgetting it
    /// is an O(1) error against a fine operator, not a small one.
    ///
    /// # Errors
    /// If the right-hand side does not match the factorization.
    ///
    /// # Panics
    /// If `source` is not a full-grid vector.
    pub fn solve(&self, source: &[f64]) -> Result<Vec<f64>, SparseLuError> {
        assert_eq!(
            source.len(),
            self.n_nodes,
            "source must be a full-grid vector"
        );
        let rhs: Vec<f64> = (0..self.n_nodes)
            .filter(|&p| self.index_map[p] >= 0)
            .enumerate()
            .map(|(k, p)| self.load_weight[k] * source[p])
            .collect();
        let interior = self.lu.solve(&rhs)?;
        Ok(embed(&interior, &self.index_map))
    }

    /// Discrete `||lap F||^2 = F^T B_F F` for a full-grid `F`; the area weights are already inside.
    ///
    /// A read-out, and the only thing in this type that touches a reduction. `>= 0`.
    ///
    /// # Panics
    /// If `f` is not a full-grid vector.
    pub fn laplacian_norm_sq(&self, f: &[f64]) -> f64 {
        assert_eq!(f.len(), self.n_nodes, "F must be a full-grid vector");
        let fi: Vec<f64> = (0..self.n_nodes)
            .filter(|&p| self.index_map[p] >= 0)
            .map(|p| f[p])
            .collect();
        let bfi = self.bf.matvec(&fi);
        let mut acc = 0.0;
        for (a, b) in fi.iter().zip(bfi.iter()) {
            acc += a * b;
        }
        acc
    }
}

/// Scatter a flat live-node vector back onto the full node grid, zeros at dead nodes.
///
/// The inverse of selecting `field[mask]`. The result is row-major with the same length as
/// `index_map`, i.e. the mask's shape.
///
/// # Panics
/// If `index_map` names a live index that `values` does not have.
pub fn embed(values: &[f64], index_map: &[i64]) -> Vec<f64> {
    index_map
        .iter()
        .map(|&p| if p < 0 { 0.0 } else { values[p as usize] })
        .collect()
}

/// Discrete 2-D inner product `<f, g> = h^2 * sum f g` over the live nodes.
///
/// # Panics
/// If `f` and `g` have different lengths.
pub fn inner2d(f: &[f64], g: &[f64], h: f64) -> f64 {
    assert_eq!(
        f.len(),
        g.len(),
        "inner2d() operands must have equal length"
    );
    let mut acc = 0.0;
    for (a, b) in f.iter().zip(g.iter()) {
        acc += a * b;
    }
    (h * h) * acc
}

/// Squared discrete 2-D norm `||f||^2 = <f, f>` (>= 0).
pub fn norm2_2d(f: &[f64], h: f64) -> f64 {
    inner2d(f, f, h)
}
