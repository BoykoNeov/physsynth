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
//! Still deliberately **not** here: `biharmonic_from_mask`, `orthotropic_biharmonic`,
//! `free_plate_stiffness*`, `VonKarmanBracket`, `AiryStressSolver`. Those are the matrices, and
//! they arrive with the plate. Leaving them visibly absent is the same choice Phase 0 made with
//! `ops.rs` — an incomplete module that looks incomplete beats a complete-looking one that is not.
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
