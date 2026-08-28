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
//! Still deliberately **not** here: the four private 1-D differences the von Kármán pieces use
//! (`_collocated_d2_1d`, `_forward_d1_1d`, `_centered_d2_1d`, `_avg_d1_1d`), `VonKarmanBracket`
//! and `AiryStressSolver`. Those are the *nonlinear* plate, and the last of them factors with
//! SuperLU — so its parity claim is a measured tolerance rather than an equality (§24.2) and it
//! does not belong in a batch whose whole result is an exact one. Leaving them visibly absent is
//! the same choice Phase 0 made with `ops.rs` — an incomplete module that looks incomplete beats
//! a complete-looking one that is not.
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
