//! Native acceptance bars for the **distributed** body↔room terminal — `airbox_port::SurfacePort`
//! and `airbox_port::InteriorSurfacePort` (HANDOFF §12.H, air-box batches 3–5).
//!
//! These carry over the port-tier half of `tests/test_airbox_surface.py`, whose wrapper-tier half
//! stays where it is until the six `RoomLoaded*` / `RoomSuspended*` wrappers are re-homed: this is
//! the retirement's first **partial** file deletion, and the line it draws is stated in the plan
//! (§14). A test retires here when its referent is the port — it builds one directly, or its every
//! assertion reads a port attribute. A test that *drives a wrapper* stays, because the trajectory
//! it asserts is the wrapper's even when the claim is named after the port.
//!
//! The claim under test, in one line: **a surface radiates from every node, and the operator that
//! spreads it onto the air grid conserves area exactly, is mirror-equivariant when centred, and
//! never feeds the grid a comb.** Where the lumped tier couples one scalar volume velocity through
//! one internal resistance, this couples a vector of them through `T^T diag(R) T`, so two modes
//! that move the same net volume can radiate completely different fields — which is the whole
//! reason the tier exists.
//!
//! Three things get their own bars because nothing else would catch them:
//!
//! * **Centring is not cosmetic.** Bilinear's mirror-equivariance holds *exactly when* the
//!   surface's centre lands on an integral half-cell and fails at 1.6e-01 otherwise, so
//!   `origin = None` buys the load's equivariance as well as the scene's symmetry. The negative
//!   control is `spreading = "nearest"`, whose symmetry rides on the rounding rule rather than on
//!   geometry.
//! * **The footprint is span-wise, not a bounding box.** A staircased disk's bbox corners sit
//!   ~0.41 R outside it and are fed by no surface node *by construction*, so a box criterion
//!   refuses every round drumhead at every resolution. Both bars here also recompute the
//!   superseded criterion, so they can show they discriminate rather than merely that the shipped
//!   code passes.
//! * **A refused interior port must leave the room as it found it.** It is the only port that
//!   *writes* the room — it registers a cut — and the cut is the last thing that can fail, so the
//!   ordering is invisible on the happy path. The bar puts a second port in the way of a cut that
//!   would otherwise have registered, which is §13.6's lesson applied in advance.
//!
//! And the association in `load_matrix` gets a bar that is not a comparison with Python, because
//! there will not be one: [`the_diagonal_folds_left_and_only_nearest_cannot_tell`] recomputes the
//! other association and pins *which fixtures could see the difference*.

use physsynth_core::airbox::{impedance_from_zeta, AirBox, Params, Wall, C0_AIR, RHO0_AIR};
use physsynth_core::airbox_port::{InteriorSurfacePort, PortError, Spreading, SurfacePort};
use physsynth_core::membrane;
use physsynth_core::sparse::Csr;

// -- the suite's fixtures, `tests/helpers.py::make_surface_room` --------------------------------

const FS: f64 = 8000.0;
const N_ROOM: [usize; 3] = [12, 11, 9];
const CFL: f64 = 0.9;
/// `tests/test_airbox_surface.py`'s batch-5 room: fine enough that a disk's bbox corners bite.
const DISK_FS: f64 = 40_000.0;
const DISK_N: [usize; 3] = [28, 28, 9];
const DISK_RADIUS: f64 = 0.18;

fn h_for(fs: f64) -> f64 {
    C0_AIR * 3.0f64.sqrt() / (CFL * fs)
}

fn rigid() -> [Wall; 6] {
    [Wall::Rigid; 6]
}

/// A room sized in **cells**, for mounting a surface in one of its walls.
fn make_room(fs: f64, n: [usize; 3], walls: [Wall; 6]) -> AirBox {
    let h = h_for(fs);
    let l = [n[0] as f64 * h, n[1] as f64 * h, n[2] as f64 * h];
    AirBox::new(Params::new(l, fs, h, walls, None, RHO0_AIR, C0_AIR).unwrap())
}

fn surface_room(walls: [Wall; 6]) -> AirBox {
    make_room(FS, N_ROOM, walls)
}

/// The **supported** plate's live nodes: `(N-1)^2` interior points of area `h^2`, no dead rim in
/// the areas because the rim does not move at all.
fn supported_plate(l: f64, n: usize) -> (Vec<[f64; 2]>, Vec<f64>) {
    let h = l / n as f64;
    let mut coords = Vec::new();
    for i in 0..n - 1 {
        for j in 0..n - 1 {
            coords.push([(i + 1) as f64 * h, (j + 1) as f64 * h]);
        }
    }
    let areas = vec![h * h; coords.len()];
    (coords, areas)
}

/// The **free** plate's live nodes: every node moves, so the areas are lumped cells — `h^2`
/// inside, `h^2/2` on an edge, `h^2/4` in a corner — and they sum to `Lx Ly` exactly.
fn free_plate(l: f64, n: usize) -> (Vec<[f64; 2]>, Vec<f64>) {
    let h = l / n as f64;
    let side = n + 1;
    let mut coords = Vec::new();
    let mut areas = Vec::new();
    for i in 0..side {
        for j in 0..side {
            coords.push([i as f64 * h, j as f64 * h]);
            let mut f = 1.0;
            if i == 0 || i == side - 1 {
                f *= 0.5;
            }
            if j == 0 || j == side - 1 {
                f *= 0.5;
            }
            areas.push(f * (h * h));
        }
    }
    (coords, areas)
}

/// A membrane's live nodes handed to a port — the shape that motivated the span-wise criterion.
fn membrane_nodes(
    domain: membrane::Domain,
    n: i64,
    lx: Option<f64>,
    ly: Option<f64>,
    radius: Option<f64>,
) -> (Vec<[f64; 2]>, Vec<f64>, f64) {
    let p = membrane::Params::new(Some(domain), 3000.0, 0.26, DISK_FS, n, lx, ly, radius, 0.0)
        .expect("a legal membrane");
    let mask = &p.mask;
    let mut coords = Vec::new();
    for j in 0..mask.nrows() {
        for i in 0..mask.ncols() {
            if mask.at(j, i) {
                coords.push([p.x[j * mask.ncols() + i], p.y[j * mask.ncols() + i]]);
            }
        }
    }
    let areas = vec![p.h * p.h; coords.len()];
    (coords, areas, p.h)
}

fn port_err(e: PortError) -> String {
    e.to_string()
}

/// The dense form of a sparse matrix, for the two norms the mirror bars take.
fn dense(m: &Csr) -> Vec<Vec<f64>> {
    (0..m.nrows())
        .map(|i| (0..m.ncols()).map(|j| m.get(i, j)).collect())
        .collect()
}

fn frobenius(a: &[Vec<f64>]) -> f64 {
    a.iter()
        .flat_map(|r| r.iter())
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt()
}

// -- the spreading operator conserves area ------------------------------------------------------

/// `test_T_distributes_every_node_area` — `T^T 1 == areas`: every surface node's area is fully
/// distributed over the air nodes under it, none created and none lost.
#[test]
fn t_distributes_every_surface_node_area() {
    for (coords, areas) in [supported_plate(0.30, 8), free_plate(0.30, 8)] {
        let mut room = surface_room(rigid());
        let port = SurfacePort::new(&mut room, "z0", &coords, &areas, None, Spreading::Bilinear)
            .expect("a legal surface");
        let ones = vec![1.0; port.node_count()];
        let distributed = port.t().transpose().matvec(&ones);
        for (m, &want) in areas.iter().enumerate() {
            assert!(
                (distributed[m] - want).abs() <= 1e-18,
                "node {m}: {} vs {want}",
                distributed[m]
            );
        }
    }
}

/// `test_net_area_is_not_the_bounding_rectangle` — `((N-1)/N)^2 Lx Ly` supported, exactly `Lx Ly`
/// free.
///
/// A simply-supported plate's rim nodes are **dead**: they do not move, so they displace no volume.
/// The shortfall is physics, not a defect, but it means comparing against a piston of area
/// `Lx Ly` is wrong by that factor at coarse `N`. The free plate has no dead rim, and that
/// contrast is the bar.
#[test]
fn the_net_area_is_not_the_bounding_rectangle() {
    let (l, n) = (0.30, 8usize);
    let mut room = surface_room(rigid());
    let (coords, areas) = supported_plate(l, n);
    let supported = SurfacePort::new(&mut room, "z0", &coords, &areas, None, Spreading::Bilinear)
        .expect("a legal surface");
    let factor = ((n as f64 - 1.0) / n as f64) * ((n as f64 - 1.0) / n as f64);
    let want = factor * l * l;
    assert!(
        (supported.net_area() - want).abs() <= 1e-15 * want,
        "supported net area {} vs {want}",
        supported.net_area()
    );

    let mut room = surface_room(rigid());
    let (coords, areas) = free_plate(l, n);
    let free = SurfacePort::new(&mut room, "z0", &coords, &areas, None, Spreading::Bilinear)
        .expect("a legal surface");
    assert!(
        (free.net_area() - l * l).abs() <= 1e-15 * l * l,
        "free net area {} vs {}",
        free.net_area(),
        l * l
    );
    assert!(free.net_area() > supported.net_area());
}

// -- mirror equivariance, and why `origin = None` is not cosmetic -------------------------------

/// Relative defect of `T^T R T` under the surface's own mirror, at an in-plane offset of `shift`
/// air cells from centred.
fn mirror_defect(spreading: Spreading, shift: f64) -> f64 {
    let (l, n) = (0.30, 16usize);
    let mut room = surface_room(rigid());
    let (coords, areas) = supported_plate(l, n);
    let origin = (
        0.5 * (N_ROOM[0] as f64 * room.p.h - l) + shift * room.p.h,
        0.5 * (N_ROOM[1] as f64 * room.p.h - l),
    );
    let port = SurfacePort::new(&mut room, "z0", &coords, &areas, Some(origin), spreading)
        .expect("a legal surface");
    // The plate's own `x -> Lx - x` as a permutation of its live nodes. The grid is built as
    // `m = i * (n-1) + j` with `x = (i+1) h`, so the mirror is a reversal of `i` alone — no
    // sort needed, unlike the reference, which had to recover the order from the coordinates.
    let side = n - 1;
    let perm: Vec<usize> = (0..side * side)
        .map(|m| (side - 1 - m / side) * side + (m % side))
        .collect();
    let matrix = dense(port.load_matrix());
    let mirrored: Vec<Vec<f64>> = (0..perm.len())
        .map(|i| (0..perm.len()).map(|j| matrix[perm[i]][perm[j]]).collect())
        .collect();
    let diff: Vec<Vec<f64>> = (0..perm.len())
        .map(|i| {
            (0..perm.len())
                .map(|j| mirrored[i][j] - matrix[i][j])
                .collect()
        })
        .collect();
    frobenius(&diff) / frobenius(&matrix)
}

/// `test_bilinear_equivariance_needs_centring` — equivariance holds **exactly when the surface is
/// centred**, not near it.
///
/// The batch plan expected bilinear's equivariance to be offset-*independent*. It is not: it holds
/// when `S = 2 (surface centre) / h_air` is an integer and fails otherwise, because the mirror
/// sends a node to a cell fraction `frac(S - t)`, which is the `1 - f` that reverses a bilinear
/// weight pair only for integral `S`. So `origin = None` defaults to centred for **two**
/// independent reasons rather than one.
#[test]
fn bilinear_equivariance_needs_centring() {
    for shift in [0.0, 0.5] {
        let d = mirror_defect(Spreading::Bilinear, shift);
        assert!(d < 1e-13, "integral S at shift {shift}: defect {d:e}");
    }
    for shift in [0.125, 0.25, 0.375, 0.625] {
        let d = mirror_defect(Spreading::Bilinear, shift);
        assert!(d > 1e-2, "non-integral S at shift {shift}: defect {d:e}");
    }
}

/// `test_nearest_node_equivariance_is_an_accident` — the negative control: nearest-node's symmetry
/// rides on the rounding rule, not on geometry.
///
/// Exact at an **even** `S` and broken at an **odd** one, because there the surface's own centre
/// node lands on a rounding tie that round-half-to-even resolves the same way from both
/// directions. Bilinear is exact at both. `Spreading::Nearest` exists only for this.
#[test]
fn nearest_node_equivariance_is_an_accident() {
    assert!(mirror_defect(Spreading::Nearest, 0.0) < 1e-13);
    assert!(mirror_defect(Spreading::Nearest, 0.5) > 1e-2);
    assert!(mirror_defect(Spreading::Bilinear, 0.5) < 1e-13);
}

// -- the ripple the spreading leaves ------------------------------------------------------------

/// Spread of the assigned area over air nodes strictly inside the footprint, in `h_air^2`.
fn interior_area_spread(spreading: Spreading, l: f64, n: usize) -> f64 {
    let mut room = surface_room(rigid());
    let h = room.p.h;
    let (coords, areas) = supported_plate(l, n);
    let port = SurfacePort::new(&mut room, "z0", &coords, &areas, None, spreading)
        .expect("a legal surface");
    let assigned = port.t().matvec(&vec![1.0; port.n_surface()]);
    let (t0, t1) = port.in_plane_axes();
    let fc = port.face_coords();
    let (mut lo0, mut hi0, mut lo1, mut hi1) = (
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    );
    for c in fc {
        lo0 = lo0.min(c[0]);
        hi0 = hi0.max(c[0]);
        lo1 = lo1.min(c[1]);
        hi1 = hi1.max(c[1]);
    }
    let (mut lo, mut hi, mut count) = (f64::INFINITY, f64::NEG_INFINITY, 0usize);
    for (m, &value) in assigned.iter().enumerate() {
        let a = port.nodes()[t0][m] as f64 * h;
        let b = port.nodes()[t1][m] as f64 * h;
        if a > lo0 + h && a < hi0 - h && b > lo1 + h && b < hi1 - h {
            lo = lo.min(value);
            hi = hi.max(value);
            count += 1;
        }
    }
    assert!(count > 4, "need a real interior to measure, got {count}");
    (hi - lo) / (h * h)
}

/// `test_bilinear_assignment_is_exact_at_an_integral_grid_ratio` — `h_air^2` per interior air node
/// **exactly**, when `h_air / h_surface` is an integer.
///
/// "Partition of unity" promises more than it delivers. Poisson summation on the periodised hat
/// gives Fourier coefficients `sinc^2(pi k h_air / h_surface)`, whose `k`-th term vanishes exactly
/// when `k h_air / h_surface` is a nonzero integer — so *all* of them vanish only for an integral
/// ratio. This bar builds that ratio deliberately; the next one measures the ripple that remains
/// otherwise.
#[test]
fn bilinear_assignment_is_exact_at_an_integral_grid_ratio() {
    let h = h_for(FS);
    for divisions in [2usize, 3, 4] {
        let n = (0.45 / (h / divisions as f64)).round() as usize;
        let l = n as f64 * h / divisions as f64;
        let spread = interior_area_spread(Spreading::Bilinear, l, n);
        assert!(spread < 1e-14, "h_air/h_p = {divisions}: spread {spread:e}");
    }
}

/// `test_bilinear_beats_nearest_node_at_every_refinement` — off an integral ratio bilinear ripples,
/// and nearest-node is 10x–100x worse and *diverges*.
///
/// This is the argument that actually decides the spreading operator; the symmetry one does not
/// survive (see [`nearest_node_equivariance_is_an_accident`]). Nearest-node wanders with no
/// convergence at all — a lumpy source at the grid scale, for nothing.
#[test]
fn bilinear_beats_nearest_node_at_every_refinement() {
    let counts = [8usize, 16, 24, 32];
    let bilinear: Vec<f64> = counts
        .iter()
        .map(|&n| interior_area_spread(Spreading::Bilinear, 0.60, n))
        .collect();
    let nearest: Vec<f64> = counts
        .iter()
        .map(|&n| interior_area_spread(Spreading::Nearest, 0.60, n))
        .collect();
    for (b, nn) in bilinear.iter().zip(nearest.iter()) {
        assert!(b < &(0.1 * nn), "bilinear {b:e} vs nearest {nn:e}");
    }
    assert!(
        bilinear[3] < 0.5 * bilinear[0],
        "bilinear must improve with refinement: {bilinear:?}"
    );
    assert!(
        nearest[3] > 0.4,
        "nearest-node does not converge, and that is the point: {nearest:?}"
    );
}

// -- the footprint is span-wise, not a bounding box ---------------------------------------------

/// The **superseded** criterion — the outer product of the two per-axis coordinate ranges.
///
/// Recomputed here so the two bars below can show they discriminate: a bar that only asserts the
/// new code passes cannot tell a fix from a no-op.
fn bounding_box_unfed(port: &SurfacePort, room: &AirBox) -> usize {
    let (t0, t1) = port.in_plane_axes();
    let h = room.p.h;
    let tol = 1e-9 * h;
    let mut inside: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
    for (d, &ax) in [t0, t1].iter().enumerate() {
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for c in port.face_coords() {
            lo = lo.min(c[d]);
            hi = hi.max(c[d]);
        }
        for i in 0..=room.p.n[ax] {
            let g = i as f64 * h;
            if g >= lo - tol && g <= hi + tol {
                inside[d].push(i);
            }
        }
    }
    let stride = room.p.n[t1] + 1;
    let mut reached: Vec<usize> = (0..port.node_count())
        .map(|m| port.nodes()[t0][m] * stride + port.nodes()[t1][m])
        .collect();
    reached.sort_unstable();
    reached.dedup();
    let mut unfed = 0;
    for &a in &inside[0] {
        for &b in &inside[1] {
            if reached.binary_search(&(a * stride + b)).is_err() {
                unfed += 1;
            }
        }
    }
    unfed
}

/// `test_accepts_a_staircased_disk_the_bounding_box_refused` — a round drumhead is the interesting
/// membrane, and the bounding box refused it, always.
///
/// The bbox corners of a disk sit ~`0.41 R` outside it, so they are fed by no surface node *by
/// construction*: the refusal was structural and refining made it no better. Span-wise the same
/// disks leave zero. Both tiers, because the criterion is shared and the interior tier measures it
/// on the low plane alone.
#[test]
fn a_staircased_disk_the_bounding_box_refused_is_accepted() {
    for n in [16i64, 24, 32, 48] {
        let (coords, areas, _) =
            membrane_nodes(membrane::Domain::Circle, n, None, None, Some(DISK_RADIUS));

        let mut room = make_room(DISK_FS, DISK_N, rigid());
        let baffled = SurfacePort::new(&mut room, "z0", &coords, &areas, None, Spreading::Bilinear)
            .unwrap_or_else(|e| panic!("N = {n}, baffled: {}", port_err(e)));
        assert_eq!(baffled.footprint_empty(), 0);
        assert!(
            bounding_box_unfed(&baffled, &room) > 0,
            "N = {n}: this configuration no longer discriminates"
        );

        let mut room = make_room(DISK_FS, DISK_N, rigid());
        let suspended = InteriorSurfacePort::new(
            &mut room,
            "z",
            4,
            &coords,
            &areas,
            None,
            Spreading::Bilinear,
        )
        .unwrap_or_else(|e| panic!("N = {n}, suspended: {}", port_err(e)));
        assert_eq!(suspended.footprint_empty(), 0);
    }
}

/// `test_a_rectangle_is_judged_by_the_identical_required_set` — the reduction that protects the
/// earlier batches: for a rectangle, span-wise **is** the box.
///
/// Every row spans the same columns and every column the same rows, so the union of the spans is
/// the bounding box of the reached nodes — by construction, not by tolerance. Asserted as exact set
/// equality rather than as equal counts.
#[test]
fn a_rectangle_is_judged_by_the_identical_required_set() {
    let (coords, areas, _) = membrane_nodes(
        membrane::Domain::Rectangle,
        12,
        Some(0.24),
        Some(0.16),
        None,
    );
    let mut room = make_room(DISK_FS, DISK_N, rigid());
    let port = SurfacePort::new(&mut room, "z0", &coords, &areas, None, Spreading::Bilinear)
        .expect("a legal rectangle");
    let (t0, t1) = port.in_plane_axes();
    let i0 = &port.nodes()[t0];
    let i1 = &port.nodes()[t1];
    let mut reached: Vec<(usize, usize)> = (0..i0.len()).map(|m| (i0[m], i1[m])).collect();
    reached.sort_unstable();
    reached.dedup();
    let (lo0, hi0) = (*i0.iter().min().unwrap(), *i0.iter().max().unwrap());
    let (lo1, hi1) = (*i1.iter().min().unwrap(), *i1.iter().max().unwrap());
    let mut boxed: Vec<(usize, usize)> = Vec::new();
    for r in lo0..=hi0 {
        for c in lo1..=hi1 {
            boxed.push((r, c));
        }
    }
    assert_eq!(reached, boxed, "a rectangle's span-wise set IS its box");
    assert_eq!(port.footprint_empty(), 0);
    assert_eq!(bounding_box_unfed(&port, &room), 0);

    let mut room = make_room(DISK_FS, DISK_N, rigid());
    let suspended = InteriorSurfacePort::new(
        &mut room,
        "z",
        4,
        &coords,
        &areas,
        None,
        Spreading::Bilinear,
    )
    .expect("a legal rectangle on an interior plane");
    assert_eq!(suspended.footprint_empty(), 0);
}

/// `test_the_comb_verdict_changes_where_it_always_did` — dropping the *shape* assumption must not
/// weaken the check where it earns its keep.
///
/// The new required set contains the old one (the coordinate box is inset by up to one node per
/// side), so it counts a few more unfed nodes when it refuses. What must not move is the **verdict
/// boundary**, and it does not: swept over a rectangle, the criterion changes its mind between the
/// same two spacings, and the crossing brackets the two-air-cell ceiling. Note this is *not* a
/// fixed ratio — where the boundary sits depends on how the patch lands on the air grid, which is
/// exactly why the condition is a count of unfed nodes and not an inequality.
#[test]
fn the_comb_verdict_changes_where_it_always_did() {
    let mut ratios = Vec::new();
    let mut verdicts = Vec::new();
    for n in [12i64, 10, 9, 8, 7, 6, 5, 4] {
        let (coords, areas, h_surface) =
            membrane_nodes(membrane::Domain::Rectangle, n, Some(0.20), Some(0.20), None);
        let mut room = make_room(DISK_FS, DISK_N, rigid());
        ratios.push(h_surface / room.p.h);
        match SurfacePort::new(&mut room, "z0", &coords, &areas, None, Spreading::Bilinear) {
            Err(e) => {
                let msg = port_err(e);
                assert!(msg.contains("fed by no surface node"), "N = {n}: {msg}");
                verdicts.push("refused");
            }
            Ok(port) => {
                verdicts.push("accepted");
                assert_eq!(
                    bounding_box_unfed(&port, &room),
                    0,
                    "N = {n}: the old criterion must agree on an ACCEPT"
                );
            }
        }
    }
    let want: Vec<&str> = ["accepted"; 5].into_iter().chain(["refused"; 3]).collect();
    assert_eq!(verdicts, want, "ratios {ratios:?}");
    assert!(
        ratios[4] < 2.0 && 2.0 < ratios[5],
        "the crossing must bracket the two-air-cell ceiling: {ratios:?}"
    );
}

// -- the load matrix -----------------------------------------------------------------------------

/// The half of `test_load_matrix_is_symmetric_and_the_cost_is_reported` whose referent is the port.
///
/// The load matrix is left as the **raw** triple product, never symmetrised: its ~1e-16 asymmetry
/// is the sparse product's summation order, and symmetrising would make this assertion vacuous.
/// The test's other half — that the factorization thickens, so `lu_nnz` is not the stored `nnz` —
/// is the wrapper's and stays in Python with it.
#[test]
fn the_load_matrix_is_symmetric_but_not_symmetrised() {
    let mut room = surface_room(rigid());
    let (coords, areas) = supported_plate(0.30, 16);
    let port = SurfacePort::new(&mut room, "z0", &coords, &areas, None, Spreading::Bilinear)
        .expect("a legal surface");
    let m = dense(port.load_matrix());
    let n = m.len();
    let mut worst: f64 = 0.0;
    let mut scale: f64 = 0.0;
    let mut exact = 0usize;
    for (i, row) in m.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            worst = worst.max((value - m[j][i]).abs());
            scale = scale.max(value.abs());
            if value == m[j][i] {
                exact += 1;
            }
        }
    }
    assert!(
        worst <= 1e-14 * scale,
        "asymmetry {worst:e} against {scale:e}"
    );
    assert!(
        exact < n * n,
        "the raw triple product must NOT be exactly symmetric, or this bar is vacuous"
    );
}

/// The association `(T_ki R_k) T_kj`, pinned by what could and could not detect the other one.
///
/// SciPy left-associates `T.T @ diags(R) @ T`, so the diagonal folds into the **left** factor, and
/// this crate copied that. There is no Python left to compare against, so the bar recomputes the
/// right-folded product and asserts *which fixtures could tell*: with `Spreading::Bilinear` the
/// two differ, and with `Spreading::Nearest` they are identical — every stored entry of a row of
/// `T` is then the same uniform node area, and `(x d) x` and `x (d x)` are the same double for
/// every `x` and `d`. That is the module's blind-fixture warning turned into a live assertion, so
/// a future reader cannot mistake "the nearest fixture agrees" for "the association does not
/// matter".
#[test]
fn the_diagonal_folds_left_and_only_nearest_cannot_tell() {
    for (spreading, must_differ) in [(Spreading::Bilinear, true), (Spreading::Nearest, false)] {
        let mut room = surface_room(rigid());
        let (coords, areas) = supported_plate(0.30, 8);
        let port = SurfacePort::new(&mut room, "z0", &coords, &areas, None, spreading)
            .expect("a legal surface");
        let t = port.t();
        let r = port.r();
        let n = t.ncols();
        // The right-folded contraction, `T_ki (R_k T_kj)`, over the same ascending `k`.
        let mut right = vec![vec![0.0f64; n]; n];
        for (k, &rk) in r.iter().enumerate() {
            for a in t.indptr()[k]..t.indptr()[k + 1] {
                for b in t.indptr()[k]..t.indptr()[k + 1] {
                    let (i, j) = (t.indices()[a], t.indices()[b]);
                    right[i][j] += t.data()[a] * (rk * t.data()[b]);
                }
            }
        }
        let left = dense(port.load_matrix());
        let differing = (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .filter(|&(i, j)| left[i][j] != right[i][j])
            .count();
        if must_differ {
            assert!(
                differing > 0,
                "bilinear: the association must be observable, or the choice is untested"
            );
        } else {
            assert_eq!(
                differing, 0,
                "nearest: the two associations are the same double, entry for entry"
            );
        }
    }
}

// -- the local free-pressure read -----------------------------------------------------------------

/// The room's own full-array open-circuit update: divergence, then the wall closure.
fn full_array_free_pressure(room: &AirBox) -> Vec<f64> {
    let div = physsynth_core::airbox::divergence(&room.p, &room.u[0], &room.u[1], &room.u[2]);
    let mut p_full = physsynth_core::airbox::pressure_step(&room.p, &room.pressure, &div);
    if room.p.has_walls {
        for (i, v) in p_full.iter_mut().enumerate() {
            *v = (*v - room.p.beta[i] * room.pressure[i]) / (1.0 + room.p.beta[i]);
        }
    }
    p_full
}

fn seed(room: &mut AirBox) {
    let p0: Vec<f64> = (0..room.p.n_nodes())
        .map(|m| ((m % 97) as f64 - 48.0) / 97.0)
        .collect();
    room.set_state(&p0, None);
}

/// `test_free_pressure_matches_full_array` — the patch's local `O(patch)` read must replicate the
/// full-array divergence-then-closure **bit for bit**, on every wall and on both node planes of an
/// interior surface.
///
/// The interesting nodes are the ones where the divergence gives a node only the faces it actually
/// has — that absence *is* the rigid closure. An off-by-one in the local read would be a small,
/// plausible, position-dependent error, and it would survive every energy test because the port and
/// the room would still agree with each other.
#[test]
fn the_local_free_pressure_read_is_the_full_array_update_exactly() {
    let z = impedance_from_zeta(4.0, RHO0_AIR, C0_AIR);
    for walls in [rigid(), [Wall::Impedance(z); 6]] {
        for face in ["x0", "x1", "y0", "y1", "z0", "z1"] {
            let mut room = surface_room(walls);
            let (coords, areas) = supported_plate(0.30, 8);
            let port =
                SurfacePort::new(&mut room, face, &coords, &areas, None, Spreading::Bilinear)
                    .unwrap_or_else(|e| panic!("{face}: {}", port_err(e)));
            seed(&mut room);
            let local = port.free_pressure(&room);
            let full = full_array_free_pressure(&room);
            for (m, &i) in port.flat().iter().enumerate() {
                let want = 0.5 * (full[i] + room.pressure[i]);
                assert_eq!(local[m], want, "face {face}, node {m}");
            }
        }

        let mut room = surface_room(walls);
        let (coords, areas) = supported_plate(0.30, 8);
        let port = InteriorSurfacePort::new(
            &mut room,
            "z",
            4,
            &coords,
            &areas,
            None,
            Spreading::Bilinear,
        )
        .expect("a legal interior surface");
        seed(&mut room);
        let (lo, hi) = port.free_pressure(&room);
        let full = full_array_free_pressure(&room);
        let half = port.face_count();
        for m in 0..half {
            let a = port.flat()[m];
            let b = port.flat()[half + m];
            assert_eq!(lo[m], 0.5 * (full[a] + room.pressure[a]), "lo node {m}");
            assert_eq!(hi[m], 0.5 * (full[b] + room.pressure[b]), "hi node {m}");
        }
    }
}

// -- the interior surface's two sides -------------------------------------------------------------

/// The dipole's defining property: one `q` reaches the room as a `-q` / `+q` pair on the two node
/// planes that straddle the surface, and the node set is low-plane-first.
///
/// New. The ordering is a claim nothing else asserts and everything else depends on — the
/// disjointness check reads the concatenation, `R` is built from the low half alone, and `inject`
/// splits the flat index in two at exactly the same place.
#[test]
fn the_two_node_planes_carry_one_q_with_opposite_signs() {
    let mut room = surface_room(rigid());
    let (coords, areas) = supported_plate(0.30, 8);
    let mut port = InteriorSurfacePort::new(
        &mut room,
        "z",
        4,
        &coords,
        &areas,
        None,
        Spreading::Bilinear,
    )
    .expect("a legal interior surface");
    let half = port.face_count();
    assert_eq!(port.node_count(), 2 * half);
    let axis = port.axis();
    assert!(port.nodes_lo()[axis].iter().all(|&i| i == 4));
    assert!(port.nodes_hi()[axis].iter().all(|&i| i == 5));
    for d in 0..3 {
        assert_eq!(port.nodes()[d][..half], port.nodes_lo()[d][..]);
        assert_eq!(port.nodes()[d][half..], port.nodes_hi()[d][..]);
    }

    seed(&mut room);
    let q: Vec<f64> = (0..half).map(|m| 1e-6 * (m as f64 + 1.0)).collect();
    port.inject(&mut room, &q).expect("a first injection");
    assert_eq!(
        room.pending_ports.len(),
        2,
        "one queued pair, not one entry"
    );
    let (a, b) = (&room.pending_ports[0], &room.pending_ports[1]);
    for (m, &want) in q.iter().enumerate() {
        assert_eq!(a.w[m], -b.w[m], "the pair must be exactly antisymmetric");
        assert_eq!(b.w[m], want);
    }
    assert_eq!(a.nodes, port.flat()[..half]);
    assert_eq!(b.nodes, port.flat()[half..]);
}

/// The cut is exactly the faces the surface covers — `face_count` of them, `face_count * h^2` of
/// area, and it is the **cut** area rather than the radiating one.
///
/// New. A partial cut is the only kind a legal interior port can make (the rim refusal is what
/// makes that true), so `cut_faces` must be strictly less than a full cross-section: a legal
/// interior plate always has a diffraction path around it.
#[test]
fn the_interior_surface_cuts_exactly_the_faces_it_covers() {
    let mut room = surface_room(rigid());
    let (coords, areas) = supported_plate(0.30, 8);
    let port = InteriorSurfacePort::new(
        &mut room,
        "z",
        4,
        &coords,
        &areas,
        None,
        Spreading::Bilinear,
    )
    .expect("a legal interior surface");
    assert_eq!(room.cut_faces(), port.face_count());
    // Spelled `scalar_pow(h, 2.0)` and deliberately not `h * h`: the reference writes `h**2`,
    // which is CPython's `float.__pow__` and therefore libm's `pow`, and this project has measured
    // the two disagreeing in the last bit for other values. They happen to agree at this `h`, so
    // writing the multiply here would promote a coincidence of the fixture into the claim.
    let h2 = physsynth_core::pyfloat::scalar_pow(room.p.h, 2.0);
    assert_eq!(port.blocked_area(&room), port.face_count() as f64 * h2);
    assert!(
        port.blocked_area(&room) != port.net_area(),
        "the cut area is not the radiating area"
    );
    let cross_section = (N_ROOM[0] + 1) * (N_ROOM[1] + 1);
    assert!(
        room.cut_faces() < cross_section,
        "a legal interior plate can never seal the room"
    );
    // Every cut face is on the surface's own plane, and there are no others.
    assert!(room.cuts[0].is_empty() && room.cuts[1].is_empty());
    assert_eq!(room.cuts[2].len(), port.face_count());
}

/// A refused interior port leaves `room.claims` **and** `room.cuts` exactly as it found them.
///
/// New, and it is the bar the value-owned shape makes necessary: the reference got the ordering
/// from `__init__` running after `__new__`, and here every refusal has to be spelled out before
/// the two writes. A happy-path test cannot see the wrong order.
///
/// §13.6's lesson applied in advance: the fixture is one where the cut **would otherwise have
/// registered**. The second port overlaps the first's node set, so it is refused for disjointness
/// — the last refusal before the cut — and if the cut ran first the room would be left carrying a
/// partition belonging to an object that does not exist.
#[test]
fn a_refused_interior_port_leaves_the_room_untouched() {
    let mut room = surface_room(rigid());
    let (coords, areas) = supported_plate(0.30, 8);
    let first = InteriorSurfacePort::new(
        &mut room,
        "z",
        4,
        &coords,
        &areas,
        None,
        Spreading::Bilinear,
    )
    .expect("a legal interior surface");
    let claims_before = room.claims.len();
    let cuts_before: Vec<usize> = room.cuts[2].clone();
    let records_before = room.cut_records.len();
    assert_eq!(cuts_before.len(), first.face_count());

    let err = InteriorSurfacePort::new(
        &mut room,
        "z",
        4,
        &coords,
        &areas,
        None,
        Spreading::Bilinear,
    )
    .expect_err("a second surface on the same faces must be refused");
    assert!(port_err(err).contains("shares node"));
    assert_eq!(room.claims.len(), claims_before, "claims must be untouched");
    assert_eq!(room.cuts[2], cuts_before, "cuts must be untouched");
    assert_eq!(
        room.cut_records.len(),
        records_before,
        "and no cut record may be left behind"
    );
}

/// A hand-placed cut may not share faces with a **port**'s, and that refusal reaches the port that
/// asks for it — the one refusal an interior surface inherits from the room.
///
/// New. Two hand-placed partitions occupying the same faces are simply one partition, but a port's
/// cut and its `-q`/`+q` pair are two halves of one object, so sharing makes the pairing ambiguous:
/// the blocked path would belong to one plate and the injection to another, with every ledger
/// staying green.
#[test]
fn an_interior_surface_over_a_hand_placed_cut_is_rejected() {
    let mut room = surface_room(rigid());
    room.add_cut("z", 4, None).expect("a legal full cut");
    let (coords, areas) = supported_plate(0.30, 8);
    let err = InteriorSurfacePort::new(
        &mut room,
        "z",
        4,
        &coords,
        &areas,
        None,
        Spreading::Bilinear,
    )
    .expect_err("a surface over an existing cut must be refused");
    assert!(port_err(err).contains("shares face"));
    assert_eq!(
        room.claims.len(),
        0,
        "and it claimed nothing on the way out"
    );
}

// -- the refusals ---------------------------------------------------------------------------------

/// `test_refuses_a_footprint_reaching_the_face_rim` — a face-rim node touches a **second** wall, so
/// `R_j` would stop being uniform across the patch.
///
/// Clipping the stencil there would fold the outboard weight back onto the boundary node: volume
/// still conserved, every ledger still green, and the source geometry quietly wrong.
#[test]
fn a_footprint_reaching_the_face_rim_is_rejected() {
    let mut room = surface_room(rigid());
    let (coords, areas) = supported_plate(0.85, 16);
    let err = SurfacePort::new(&mut room, "z0", &coords, &areas, None, Spreading::Bilinear)
        .expect_err("a footprint reaching the rim must be refused");
    assert!(port_err(err).contains("rim"));
}

/// `test_refuses_a_footprint_outside_the_face`.
#[test]
fn a_footprint_outside_the_face_is_rejected() {
    let mut room = surface_room(rigid());
    let (coords, areas) = supported_plate(0.30, 8);
    let err = SurfacePort::new(
        &mut room,
        "z0",
        &coords,
        &areas,
        Some((0.9, 0.4)),
        Spreading::Bilinear,
    )
    .expect_err("a footprint outside the face must be refused");
    assert!(port_err(err).contains("outside face 'z0'"));
}

/// `test_refuses_a_surface_too_coarse_for_the_air_grid` — unfed air nodes under the footprint make
/// the acoustic source a comb at the grid scale.
///
/// The condition is a **count of unfed nodes**, not an inequality on `h_surface / h_air`: at a
/// ratio comfortably inside the naive inequality, nearest-node still leaves half the footprint
/// unfed.
#[test]
fn a_surface_too_coarse_for_the_air_grid_is_rejected() {
    let mut room = surface_room(rigid());
    let (coords, areas) = free_plate(0.30, 3);
    let err = SurfacePort::new(&mut room, "z0", &coords, &areas, None, Spreading::Nearest)
        .expect_err("a comb source must be refused");
    assert!(port_err(err).contains("fed by no surface node"));
}

/// `test_refuses_a_surface_on_an_open_face` — perfectly conservative, completely silent, and the
/// energy report is blind to it.
#[test]
fn a_surface_on_an_open_face_is_rejected() {
    let mut walls = rigid();
    walls[4] = Wall::Impedance(0.0); // 'z0' open
    let mut room = surface_room(walls);
    let (coords, areas) = supported_plate(0.30, 8);
    let err = SurfacePort::new(&mut room, "z0", &coords, &areas, None, Spreading::Bilinear)
        .expect_err("a surface on an open face must be refused");
    assert!(port_err(err).contains("open (pressure-release) face(s) ['z0']"));
}

/// `test_refuses_overlapping_ports` — two surfaces sharing an air node are not independent within a
/// step, so each solves against a pressure that never occurred.
///
/// Disjointness is exactly what makes the cheap per-port solve *exact*, and it is what lets N
/// instruments share one room.
#[test]
fn overlapping_surfaces_are_rejected() {
    let mut room = surface_room(rigid());
    let (coords, areas) = supported_plate(0.30, 8);
    SurfacePort::new(&mut room, "z0", &coords, &areas, None, Spreading::Bilinear)
        .expect("the first surface");
    let err = SurfacePort::new(&mut room, "z0", &coords, &areas, None, Spreading::Bilinear)
        .expect_err("a second surface on the same nodes must be refused");
    assert!(port_err(err).contains("shares node"));
}

/// `test_two_disjoint_surfaces_share_one_room`, reduced to its port-tier half: two surfaces on
/// opposite walls are accepted and neither claims the other's nodes. The scene's conservation is
/// the wrapper's and stays in Python with it.
#[test]
fn two_disjoint_surfaces_share_one_room() {
    let mut room = make_room(FS, [12, 11, 12], rigid());
    let (coords, areas) = supported_plate(0.30, 8);
    let top = SurfacePort::new(&mut room, "z0", &coords, &areas, None, Spreading::Bilinear)
        .expect("the top surface");
    let bottom = SurfacePort::new(&mut room, "z1", &coords, &areas, None, Spreading::Bilinear)
        .expect("the bottom surface");
    let shared: Vec<usize> = top
        .flat()
        .iter()
        .filter(|f| bottom.flat().contains(f))
        .copied()
        .collect();
    assert!(shared.is_empty(), "the two node sets must be disjoint");
    assert_eq!(room.claims.len(), 2);
}

/// `test_refuses_solving_twice_without_a_room_step` — a port does not step its room; the caller
/// does, once, after every port has solved.
#[test]
fn solving_twice_without_a_room_step_is_rejected() {
    let mut room = surface_room(rigid());
    let (coords, areas) = supported_plate(0.30, 8);
    let mut port = SurfacePort::new(&mut room, "z0", &coords, &areas, None, Spreading::Bilinear)
        .expect("a legal surface");
    let q = vec![0.0; port.node_count()];
    port.inject(&mut room, &q).expect("the first injection");
    let err = port
        .inject(&mut room, &q)
        .expect_err("a second injection inside one room step must be refused");
    assert!(port_err(err).contains("twice within one room step"));
    room.step();
    port.inject(&mut room, &q).expect("and it recovers");
}

/// The epoch, and not the step count, is what unsticks a distributed port.
///
/// New, and the same trap §13.2 recorded for the lumped tier, restated where it can regress
/// independently: the port queued at step 0 and `set_state` returns the room to step 0, so a bare
/// step-count comparison would still refuse. It passes only because the epoch moved.
#[test]
fn the_epoch_and_not_the_step_count_is_what_unsticks_a_surface_port() {
    let mut room = surface_room(rigid());
    let (coords, areas) = supported_plate(0.30, 8);
    let mut port = SurfacePort::new(&mut room, "z0", &coords, &areas, None, Spreading::Bilinear)
        .expect("a legal surface");
    let q = vec![0.0; port.node_count()];
    port.inject(&mut room, &q).expect("the first injection");
    let (epoch, n) = (room.epoch, room.n);
    seed(&mut room);
    assert_eq!(room.n, n, "set_state must return the room to the same step");
    assert!(room.epoch > epoch, "and it must move the epoch");
    port.inject(&mut room, &q)
        .expect("a restarted room unsticks the port");
}

/// `test_refuses_unknown_face_and_spreading`, minus the half that has no analogue.
///
/// The reference also refuses `spreading="cubic"`, which is a claim about a string argument;
/// [`Spreading`] is an enum and the compiler makes it. The unknown-face half survives because the
/// face is still a name.
#[test]
fn an_unknown_face_is_rejected() {
    let mut room = surface_room(rigid());
    let (coords, areas) = supported_plate(0.30, 8);
    let err = SurfacePort::new(&mut room, "q0", &coords, &areas, None, Spreading::Bilinear)
        .expect_err("an unknown face must be refused");
    assert!(port_err(err).contains("unknown face 'q0'"));
}

/// An interior index that would put a straddling node plane on a wall is refused.
///
/// New. A node plane on a wall carries half the node weight `W` and the wall's admittance in
/// `beta`, so `R_j` would differ between the two sides and the load would stop being `2 T^T R T`
/// with a single `R` — the assumption every other claim in this tier rests on.
#[test]
fn an_interior_index_that_reaches_a_wall_is_rejected() {
    let (coords, areas) = supported_plate(0.30, 8);
    for index in [0i64, -1, (N_ROOM[2] - 1) as i64, 99] {
        let mut room = surface_room(rigid());
        let err = InteriorSurfacePort::new(
            &mut room,
            "z",
            index,
            &coords,
            &areas,
            None,
            Spreading::Bilinear,
        )
        .expect_err("an index reaching a wall must be refused");
        assert!(port_err(err).contains("out of range"), "index {index}");
    }
    let mut room = surface_room(rigid());
    let err = InteriorSurfacePort::new(
        &mut room,
        "z0",
        4,
        &coords,
        &areas,
        None,
        Spreading::Bilinear,
    )
    .expect_err("a wall face is not an interior plane");
    assert!(port_err(err).contains("unknown plane 'z0'"));
}

/// A `q` of the wrong length is refused on both tiers, and the interior tier's message names the
/// **face** count rather than the node count — they differ by exactly a factor of two.
#[test]
fn a_q_of_the_wrong_length_is_rejected() {
    let mut room = surface_room(rigid());
    let (coords, areas) = supported_plate(0.30, 8);
    let mut wall = SurfacePort::new(&mut room, "z0", &coords, &areas, None, Spreading::Bilinear)
        .expect("a legal surface");
    let err = wall
        .inject(&mut room, &[0.0])
        .expect_err("a scalar is not the per-node vector");
    assert!(port_err(err).contains("per-node volume-velocity vector"));

    let mut room = surface_room(rigid());
    let mut interior = InteriorSurfacePort::new(
        &mut room,
        "z",
        4,
        &coords,
        &areas,
        None,
        Spreading::Bilinear,
    )
    .expect("a legal interior surface");
    let full = vec![0.0; interior.node_count()];
    let err = interior
        .inject(&mut room, &full)
        .expect_err("the node-count vector is twice the face count");
    let msg = port_err(err);
    assert!(msg.contains("per-FACE volume-velocity vector"), "{msg}");
    assert!(msg.contains("HALF the node count"), "{msg}");
    interior
        .inject(&mut room, &vec![0.0; interior.face_count()])
        .expect("the per-face vector is accepted");
}

/// Areas must be one per surface node, finite and non-negative.
#[test]
fn malformed_areas_are_rejected() {
    let mut room = surface_room(rigid());
    let (coords, areas) = supported_plate(0.30, 8);
    let err = SurfacePort::new(
        &mut room,
        "z0",
        &coords,
        &areas[..areas.len() - 1],
        None,
        Spreading::Bilinear,
    )
    .expect_err("one area short must be refused");
    assert!(port_err(err).contains("one per surface node"));

    let mut bad = areas.clone();
    bad[3] = -1.0;
    let err = SurfacePort::new(&mut room, "z0", &coords, &bad, None, Spreading::Bilinear)
        .expect_err("a negative area must be refused");
    assert!(port_err(err).contains("finite and >= 0"));

    let mut bad = areas;
    bad[3] = f64::NAN;
    let err = SurfacePort::new(&mut room, "z0", &coords, &bad, None, Spreading::Bilinear)
        .expect_err("a NaN area must be refused");
    assert!(port_err(err).contains("finite and >= 0"));
}
