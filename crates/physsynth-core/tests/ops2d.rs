//! Native validation of the 2-D geometry and the masked Laplacian.
//!
//! What these assert is *structure*, not arithmetic against a stored number: the mask counts, the
//! index map's bijection, the Laplacian's symmetry and its negative-definiteness. The reason is
//! that the membrane's whole energy argument rests on `L` being symmetric — a masked 5-point
//! Laplacian is a principal submatrix of a symmetric operator, so it stays symmetric however ugly
//! the staircased rim gets — and a builder that quietly broke that symmetry would still produce a
//! plausible drum. `tests/test_rust_parity_membrane.py` covers agreement with SciPy; this file
//! covers the properties that make the model correct in the first place.

use physsynth_core::ops2d::{
    biharmonic_from_mask, cells_per_node, dirichlet_interior_d2_1d, disk_mask, embed,
    free_plate_stiffness, free_plate_stiffness_from_mask, grid_coords, guitar_area,
    guitar_half_width, guitar_mask, guitar_scale, inner2d, laplacian_from_mask, live_cells,
    norm2_2d, orthotropic_biharmonic, prune_to_area_carrying, rectangle_mask, Mask,
};
use physsynth_core::sparse::Csr;

/// Row-major flat index of node `(j, i)` on a grid `ncols` nodes wide — the one ordering every
/// live-node vector and every `index_map` in this project is written in.
fn flat(j: usize, i: usize, ncols: usize) -> usize {
    j * ncols + i
}

#[test]
fn a_rectangle_mask_is_exactly_its_interior() {
    let m = rectangle_mask(6, 4);
    assert_eq!((m.nrows(), m.ncols()), (5, 7));
    assert_eq!(m.n_live(), 5 * 3);
    for j in 0..m.nrows() {
        for i in 0..m.ncols() {
            let interior = j > 0 && j < m.nrows() - 1 && i > 0 && i < m.ncols() - 1;
            assert_eq!(m.at(j, i), interior, "node ({j}, {i})");
        }
    }
}

#[test]
fn a_degenerate_rectangle_has_no_unknowns() {
    // Two segments per side leaves exactly one interior node; one segment leaves none. Both must
    // build without panicking, because the membrane rejects the empty one by message, not by crash.
    assert_eq!(rectangle_mask(2, 2).n_live(), 1);
    assert_eq!(rectangle_mask(1, 1).n_live(), 0);
}

#[test]
fn the_index_map_is_a_bijection_onto_the_live_nodes_in_c_order() {
    let (x, y, _) = grid_coords(12, 1.0);
    let m = disk_mask(&x, &y, 0.7, 13, 13);
    let map = m.index_map();

    let mut expected = 0i64;
    for (idx, &alive) in m.flags().iter().enumerate() {
        if alive {
            assert_eq!(map[idx], expected, "live node {idx} out of C-order");
            expected += 1;
        } else {
            assert_eq!(map[idx], -1, "dead node {idx} must map to -1");
        }
    }
    assert_eq!(expected as usize, m.n_live());
}

#[test]
fn the_grid_hits_both_endpoints_exactly() {
    let (x, y, h) = grid_coords(10, 0.15);
    assert_eq!(x[0], -0.15);
    // Last column of the first row.
    assert_eq!(x[10], 0.15);
    assert_eq!(y[0], -0.15);
    // First column of the last row.
    assert_eq!(y[10 * 11], 0.15);
    assert_eq!(h, 2.0 * 0.15 / 10.0);
}

#[test]
fn a_node_on_the_rim_is_dead() {
    // The disk predicate is strict, so the node at exactly x = radius is boundary. With an even
    // segment count and the radius as half-extent, the equator nodes land exactly on the rim.
    let (x, y, _) = grid_coords(8, 1.0);
    let m = disk_mask(&x, &y, 1.0, 9, 9);
    let mid = 4; // the centre row
    assert!(!m.at(mid, 0), "the west rim node must be dead");
    assert!(!m.at(mid, 8), "the east rim node must be dead");
    assert!(m.at(mid, 1), "one node in must be live");
}

#[test]
fn the_masked_laplacian_is_symmetric_on_a_staircased_rim() {
    let (x, y, h) = grid_coords(21, 0.15);
    let m = disk_mask(&x, &y, 0.15, 22, 22);
    let (l, _) = laplacian_from_mask(&m, h);

    assert_eq!(l.nrows(), m.n_live());
    assert!(
        l.is_symmetric(),
        "a masked 5-point Laplacian is a principal submatrix of a symmetric operator"
    );
}

#[test]
fn the_masked_laplacian_is_negative_definite() {
    // `<-L v, v> > 0` for a spread of random-ish vectors. Definiteness is what makes the
    // membrane's potential energy non-negative, which is what makes E^n >= 0 a stability
    // statement rather than a hope.
    let (x, y, h) = grid_coords(17, 0.2);
    let m = disk_mask(&x, &y, 0.2, 18, 18);
    let (l, _) = laplacian_from_mask(&m, h);
    let n = l.nrows();

    for seed in 1..6u64 {
        let v: Vec<f64> = (0..n)
            .map(|i| (((i as u64 * 2654435761 + seed * 40503) % 1000) as f64 / 500.0) - 1.0)
            .collect();
        let lv = l.matvec(&v);
        let quad: f64 = lv.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
        assert!(quad < 0.0, "seed {seed}: <L v, v> = {quad} must be < 0");
    }
}

#[test]
fn the_laplacian_rows_are_the_five_point_stencil() {
    let m = rectangle_mask(5, 5);
    let h = 0.25;
    let (l, map) = laplacian_from_mask(&m, h);
    let inv_h2 = 1.0 / (h * h);

    // The very middle of a 5x5-segment rectangle has all four neighbours live.
    let centre = map[flat(3, 3, 6)] as usize;
    assert_eq!(l.get(centre, centre), -4.0 * inv_h2);
    let mut neighbours = 0;
    for other in 0..l.nrows() {
        if other != centre && l.get(centre, other) != 0.0 {
            assert_eq!(l.get(centre, other), inv_h2);
            neighbours += 1;
        }
    }
    assert_eq!(neighbours, 4, "an interior node has four live neighbours");

    // A node against the rim loses one neighbour to the ghost, and the diagonal does NOT change —
    // the dropped column is the Dirichlet condition, not a modified stencil.
    let edge = map[flat(1, 3, 6)] as usize;
    assert_eq!(l.get(edge, edge), -4.0 * inv_h2);
    let live_neighbours = (0..l.nrows())
        .filter(|&o| o != edge && l.get(edge, o) != 0.0)
        .count();
    assert_eq!(live_neighbours, 3);
}

#[test]
fn embed_is_the_inverse_of_selecting_through_the_mask() {
    let m = Mask::new(
        3,
        4,
        vec![
            false, false, false, false, //
            false, true, true, false, //
            false, false, false, false,
        ],
    );
    let map = m.index_map();
    let field = embed(&[7.0, -2.0], &map);

    assert_eq!(field.len(), 12);
    assert_eq!(field[flat(1, 1, 4)], 7.0);
    assert_eq!(field[flat(1, 2, 4)], -2.0);
    assert_eq!(field.iter().filter(|&&v| v != 0.0).count(), 2);
}

#[test]
fn the_two_dimensional_inner_product_carries_the_cell_area() {
    let f = [1.0, 2.0, 3.0];
    let g = [4.0, -1.0, 0.5];
    let h = 0.125;
    assert_eq!(inner2d(&f, &g, h), (h * h) * (4.0 - 2.0 + 1.5));
    assert_eq!(norm2_2d(&f, h), inner2d(&f, &f, h));
    assert!(norm2_2d(&f, h) > 0.0);
}

// --- the guitar outline (model #5g's geometry) ---------------------------------------------

/// The `(nrows, ncols)` node grid a guitar plate builds on: `x` from the centre line, `y` from the
/// neck end, cells square, and `Ly` snapped to a whole number of them exactly as `plate.py` does.
fn guitar_grid(lx: f64, ly_asked: f64, n: usize) -> (Vec<f64>, Vec<f64>, f64, usize, usize) {
    let h = lx / (n as f64);
    let ny = ((ly_asked / h).round() as usize).max(1);
    let ly = (ny as f64) * h;
    let (nrows, ncols) = (ny + 1, n + 1);
    let mut x = Vec::with_capacity(nrows * ncols);
    let mut y = Vec::with_capacity(nrows * ncols);
    for j in 0..nrows {
        for i in 0..ncols {
            // np.linspace: `i * step`, with the final entry overwritten by the endpoint.
            let xi = if i == n {
                lx
            } else {
                (i as f64) * (lx / (n as f64))
            };
            let yj = if j == ny {
                ly
            } else {
                (j as f64) * (ly / (ny as f64))
            };
            x.push(xi - 0.5 * lx);
            y.push(yj);
        }
    }
    (x, y, ly, nrows, ncols)
}

#[test]
fn the_outline_closes_at_both_ends_and_peaks_where_the_scale_says() {
    for &(waist, asym) in &[(0.42, 0.30), (0.0, 0.0), (0.88, 0.0), (0.60, -0.30)] {
        assert_eq!(guitar_half_width(0.0, waist, asym), 0.0, "t = 0 must close");
        // sin(pi * 1.0) is not exactly 0 in doubles -- pi is not pi -- so the far end closes to
        // within the argument error, which is what the outline's `t < 1` test relies on anyway.
        assert!(
            guitar_half_width(1.0, waist, asym).abs() < 1e-15,
            "t = 1 must close"
        );

        // `scale` normalises the sampled peak to half the requested width. Sampled, so the true
        // maximum can sit between two samples -- hence a tolerance rather than an equality.
        let width = 0.37;
        let scale = guitar_scale(width, waist, asym);
        let peak = (0..=20_000)
            .map(|i| scale * guitar_half_width((i as f64) / 20_000.0, waist, asym))
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (peak - 0.5 * width).abs() < 1e-15,
            "waist {waist}: peak {peak} against {}",
            0.5 * width
        );
    }
}

#[test]
fn the_three_factors_multiply_left_to_right() {
    // Section 17.2's finding, and section 23.5's: a claim about a *spelling* is only testable
    // while the compiler cannot see which spelling was meant. With literal arguments LLVM folds
    // both associations into the same constant and the test asserts nothing -- in release, where
    // it matters, and not in debug, where it would be noticed. `black_box` is what keeps the two
    // multiplications separate, and the search runs until it finds a genuine witness so the test
    // fails loudly if the class of difference ever stops existing.
    let mut witnesses = 0;
    for i in 0..20_000 {
        let t = std::hint::black_box(((i as f64) + 0.5) / 20_000.0);
        let waist = std::hint::black_box(0.42);
        let asym = std::hint::black_box(0.30);
        let a = (t * std::f64::consts::PI).sin();
        let b = 1.0 - waist * (4.0 * std::f64::consts::PI * (t - 0.5)).cos();
        let c = 1.0 + asym * (t - 0.5);
        let left = (a * b) * c;
        let right = a * (b * c);
        assert_eq!(
            guitar_half_width(t, waist, asym),
            left,
            "the implementation must be the left-associated product"
        );
        if left != right {
            witnesses += 1;
        }
    }
    assert!(
        witnesses > 0,
        "no argument distinguished (a*b)*c from a*(b*c) -- this test is asserting nothing"
    );
}

#[test]
fn the_area_quadrature_matches_the_closed_form_for_a_symmetric_outline() {
    // With `asym = 0` the profile integrates in closed form:
    //     int_0^1 sin(pi t) [1 - w cos(4 pi (t - 1/2))] dt = (2/pi) (1 + w/15)
    // so the outline's area is 2 * scale * L * that. A real oracle for the quadrature rather than
    // a stored number -- it is the only bar here that would catch a wrong midpoint rule.
    let (length, width) = (0.48, 0.37);
    for &waist in &[0.0, 0.42, 0.88] {
        let scale = guitar_scale(width, waist, 0.0);
        let exact = 2.0 * scale * length * (2.0 / std::f64::consts::PI) * (1.0 + waist / 15.0);
        let got = guitar_area(length, width, waist, 0.0);
        assert!(
            ((got - exact) / exact).abs() < 1e-12,
            "waist {waist}: quadrature {got} against closed form {exact}"
        );
    }
}

#[test]
fn the_outline_excludes_its_two_end_rows_and_is_mirror_symmetric() {
    let (x, y, ly, nrows, ncols) = guitar_grid(0.37, 0.48, 24);
    let m = guitar_mask(&x, &y, ly, 0.37, 0.42, 0.30, nrows, ncols);
    for i in 0..ncols {
        assert!(!m.at(0, i), "the neck row is not part of the plate");
        assert!(!m.at(nrows - 1, i), "the tail row is not part of the plate");
    }
    // The predicate is `|x| < half`, and the grid is symmetric about the centre line, so the mask
    // has to be too -- a plate that is a little wider on one side is exactly the kind of geometry
    // error that leaves every physics bar green.
    for j in 0..nrows {
        for i in 0..ncols {
            assert_eq!(m.at(j, i), m.at(j, ncols - 1 - i), "node ({j}, {i})");
        }
    }
    assert!(m.n_live() > 0);
}

#[test]
fn a_deeper_waist_removes_nodes_and_a_wider_asymmetry_moves_them() {
    let (x, y, ly, nrows, ncols) = guitar_grid(0.37, 0.48, 32);
    let plain = guitar_mask(&x, &y, ly, 0.37, 0.0, 0.0, nrows, ncols).n_live();
    let waisted = guitar_mask(&x, &y, ly, 0.37, 0.42, 0.0, nrows, ncols).n_live();
    assert!(
        waisted < plain,
        "a waist is a bite out of the outline: {waisted} against {plain}"
    );
    // `asym` widens the lower bout at the expense of the upper, so it redistributes rather than
    // adds. The count is free to move either way; what must not happen is nothing moving.
    let tilted = guitar_mask(&x, &y, ly, 0.37, 0.42, 0.30, nrows, ncols);
    let square = guitar_mask(&x, &y, ly, 0.37, 0.42, 0.0, nrows, ncols);
    assert!(
        (0..nrows).any(|j| (0..ncols).any(|i| tilted.at(j, i) != square.at(j, i))),
        "asym changed no node at all"
    );
}

#[test]
fn every_cell_of_a_full_grid_is_live_and_the_node_counts_are_four_two_one() {
    let m = Mask::new(4, 5, vec![true; 20]);
    assert_eq!(live_cells(&m), vec![true; 12]);
    let counts = cells_per_node(&m);
    for j in 0..4 {
        for i in 0..5 {
            let edge_j = j == 0 || j == 3;
            let edge_i = i == 0 || i == 4;
            let expected = match (edge_j, edge_i) {
                (true, true) => 1,
                (true, false) | (false, true) => 2,
                (false, false) => 4,
            };
            assert_eq!(counts[j * 5 + i], expected, "node ({j}, {i})");
        }
    }
}

#[test]
fn a_grid_thinner_than_a_cell_has_no_cells() {
    assert!(live_cells(&Mask::new(1, 5, vec![true; 5])).is_empty());
    assert_eq!(
        cells_per_node(&Mask::new(1, 5, vec![true; 5])),
        vec![0i64; 5]
    );
}

#[test]
fn pruning_a_full_rectangle_drops_nothing() {
    let m = Mask::new(4, 5, vec![true; 20]);
    let (pruned, dropped) = prune_to_area_carrying(&m);
    assert_eq!(dropped, 0);
    assert_eq!(pruned.flags(), m.flags());
}

#[test]
fn pruning_iterates_to_a_fixed_point_rather_than_sweeping_once() {
    // The case that makes the loop a correctness statement instead of an optimisation: a two-node
    // spike. The tip carries no cell and goes on the first sweep; only then does the node behind
    // it carry none either. A single sweep leaves a node with zero area weight, which is exactly
    // what makes the free plate's mass matrix singular.
    //
    //   row 0:  # # . .      (a 2x2 block of live nodes, plus a two-node tail off its corner)
    //   row 1:  # # . .
    //   row 2:  . . . .
    let mut live = vec![false; 4 * 5];
    for j in 0..2 {
        for i in 0..2 {
            live[j * 5 + i] = true;
        }
    }
    live[2 * 5 + 2] = true; // the second node of the tail
    live[2 * 5 + 3] = true; // the tip
    let (pruned, dropped) = prune_to_area_carrying(&Mask::new(4, 5, live));
    assert_eq!(
        dropped, 2,
        "both tail nodes carry no area, one after the other"
    );
    assert_eq!(pruned.n_live(), 4);
    for j in 0..2 {
        for i in 0..2 {
            assert!(
                pruned.at(j, i),
                "the block is area-carrying and must survive"
            );
        }
    }
}

#[test]
fn the_shipped_guitar_produces_spikes_and_pruning_removes_them_all() {
    // The reason `prune_to_area_carrying` exists: a curved rim staircases into nodes whose
    // trapezoidal weight is exactly zero, and two are enough to make `W` singular. Measured at
    // 2-4 of them on every grid the plate ships.
    for n in [20, 24, 32, 40, 48] {
        let (x, y, ly, nrows, ncols) = guitar_grid(0.37, 0.48, n);
        let raw = guitar_mask(&x, &y, ly, 0.37, 0.42, 0.30, nrows, ncols);
        let (pruned, dropped) = prune_to_area_carrying(&raw);
        assert!(dropped > 0, "N = {n}: the outline produced no spike at all");
        assert!(
            cells_per_node(&pruned)
                .iter()
                .zip(pruned.flags().iter())
                .all(|(&c, &alive)| !alive || c > 0),
            "N = {n}: a live node still carries no area"
        );
    }
}

// --- the matrices (Phase 5 batch 2) ----------------------------------------------------------
//
// What these assert is the same thing the geometry bars above assert: the properties that make the
// operator correct, not agreement with a stored number. `tests/test_rust_parity_ops2d.py` covers
// the bit-for-bit agreement with SciPy. Two of them are different in kind and are marked as such —
// they pin an *arithmetic spelling* rather than a physical property, and both are written to fail
// if the witness they search for does not exist, because a spelling test that finds nothing is a
// test that asserted nothing (plan §23.5, and §17.2 for why that has happened here before).

/// The Dirichlet Laplacian's exact discrete eigenvalue on the `sin·sin` mode of a rectangle.
///
/// `L u = -lambda u` with `lambda = (4/h²)(sin²(m pi /2Nx) + sin²(n pi /2Ny))`.
fn rectangle_mode_eigenvalue(nx: usize, ny: usize, h: f64, m: usize, n: usize) -> (f64, f64) {
    let sx = (m as f64 * std::f64::consts::PI / (2.0 * nx as f64)).sin();
    let sy = (n as f64 * std::f64::consts::PI / (2.0 * ny as f64)).sin();
    (4.0 / (h * h) * sx * sx, 4.0 / (h * h) * sy * sy)
}

/// The `sin·sin` mode sampled on the interior nodes, in live-node order.
fn rectangle_mode(nx: usize, ny: usize, m: usize, n: usize) -> Vec<f64> {
    let mut v = Vec::new();
    for j in 1..ny {
        for i in 1..nx {
            v.push(
                (m as f64 * std::f64::consts::PI * i as f64 / nx as f64).sin()
                    * (n as f64 * std::f64::consts::PI * j as f64 / ny as f64).sin(),
            );
        }
    }
    v
}

#[test]
fn the_interior_second_difference_has_the_analytic_sine_eigenvalue() {
    // The 1-D factor of the tensor-product rectangle. Its eigenvectors are exactly the interior
    // samples of sin(m pi x / L) -- exactly, not to O(h²), which is what makes the plate's modal
    // oracle a closed form rather than a convergence rate.
    let (n_int, h) = (11usize, 0.037);
    let d = dirichlet_interior_d2_1d(n_int, h);
    let nx = n_int + 1;
    for m in 1..=4 {
        let v: Vec<f64> = (1..=n_int)
            .map(|i| (m as f64 * std::f64::consts::PI * i as f64 / nx as f64).sin())
            .collect();
        let sm = (m as f64 * std::f64::consts::PI / (2.0 * nx as f64)).sin();
        let lambda = 4.0 / (h * h) * sm * sm;
        let y = d.matvec(&v);
        for (a, b) in y.iter().zip(v.iter()) {
            assert!(
                (a + lambda * b).abs() <= 1e-9 * lambda,
                "mode {m}: {a} != {}",
                -lambda * b
            );
        }
    }
}

#[test]
fn the_biharmonic_squares_the_laplacian_eigenvalues() {
    // B = L² is the whole content of the simply-supported plate: the rim conditions u = 0 AND
    // Lap u = 0 are both carried by squaring a Dirichlet Laplacian, with no 13-point boundary row
    // written anywhere. The test of that is that the sin-sin modes survive as exact eigenvectors
    // with the eigenvalue squared.
    let (nx, ny, h) = (9usize, 7usize, 0.037);
    let (b, _) = biharmonic_from_mask(&rectangle_mask(nx, ny), h);
    for (m, n) in [(1usize, 1usize), (2, 1), (3, 2)] {
        let (lx, ly) = rectangle_mode_eigenvalue(nx, ny, h, m, n);
        let lambda = lx + ly;
        let v = rectangle_mode(nx, ny, m, n);
        let y = b.matvec(&v);
        for (a, c) in y.iter().zip(v.iter()) {
            assert!(
                (a - lambda * lambda * c).abs() <= 1e-9 * lambda * lambda,
                "mode ({m},{n}): {a} != {}",
                lambda * lambda * c
            );
        }
    }
}

#[test]
fn the_biharmonic_is_symmetric_and_positive_definite_on_a_staircased_rim() {
    // Symmetry is the plate's energy argument, exactly as it is the membrane's, and it survives a
    // staircase for the same reason: B = L² and L is a principal submatrix of a symmetric operator.
    for mask in [
        rectangle_mask(9, 7),
        guitar_test_mask(24),
        disk_test_mask(20),
    ] {
        let (b, _) = biharmonic_from_mask(&mask, 0.037);
        assert!(b.is_symmetric(), "the biharmonic lost its symmetry");
        let n = b.nrows();
        for seed in 0..4u64 {
            let v = pseudorandom(n, seed);
            let q: f64 = b.matvec(&v).iter().zip(v.iter()).map(|(a, c)| a * c).sum();
            assert!(q > 0.0, "the biharmonic is not positive-definite: {q}");
        }
    }
}

#[test]
fn the_orthotropic_operator_has_the_analytic_mode_eigenvalue() {
    // The eigenvalue on mode (m,n) is g_x lx² + 2 g_h lx ly + g_y ly² -- which is also the whole
    // statement of where the factor of 2 lives. Halving it, or moving it inside H, gives a
    // perfectly stable, exactly energy-conserving, WRONG plate, and only this bar can see that.
    let (nx, ny, h) = (9usize, 7usize, 0.037);
    for (gx, gh, gy) in [(1.0, 1.0, 1.0), (1.3, 0.9, 1.1), (1.0, 0.153, 0.073)] {
        let (b, _) = orthotropic_biharmonic(nx, ny, h, gx, gh, gy);
        for (m, n) in [(1usize, 1usize), (3, 2)] {
            let (lx, ly) = rectangle_mode_eigenvalue(nx, ny, h, m, n);
            let mu = gx * lx * lx + 2.0 * gh * lx * ly + gy * ly * ly;
            let v = rectangle_mode(nx, ny, m, n);
            let y = b.matvec(&v);
            for (a, c) in y.iter().zip(v.iter()) {
                assert!(
                    (a - mu * c).abs() <= 1e-9 * mu,
                    "grain ({gx},{gh},{gy}) mode ({m},{n}): {a} != {}",
                    mu * c
                );
            }
        }
    }
}

#[test]
fn the_isotropic_collapse_is_grid_dependent_and_both_halves_are_asserted() {
    // At g_x = g_h = g_y the orthotropic assembly IS `g · L @ L` in exact arithmetic, and in
    // doubles it is bit-identical on some grids and not on others -- the three products are summed
    // in a different order than L is squared. `plate.py` keeps the isotropic default on the
    // squared-Laplacian path precisely because neither behaviour can be relied on, so this asserts
    // BOTH halves: a grid where it is exact and a grid where it is not. Only the second half can
    // rot, and a version of this test that lost it would still pass while saying nothing.
    let (nx, ny) = (7usize, 5usize);
    let exact_grid = 0.5; // 1/h² == 4.0 exactly
    let (a, _) = orthotropic_biharmonic(nx, ny, exact_grid, 1.0, 1.0, 1.0);
    let (b, _) = biharmonic_from_mask(&rectangle_mask(nx, ny), exact_grid);
    assert_eq!(a.data(), b.data(), "the dyadic grid must collapse exactly");

    let odd_grid = 0.037; // 1/h² == 730.4601899196495
    let (a, _) = orthotropic_biharmonic(nx, ny, odd_grid, 1.0, 1.0, 1.0);
    let (b, _) = biharmonic_from_mask(&rectangle_mask(nx, ny), odd_grid);
    assert_eq!(
        a.indices(),
        b.indices(),
        "the two must at least agree on structure"
    );
    let differing = a
        .data()
        .iter()
        .zip(b.data().iter())
        .filter(|(x, y)| x != y)
        .count();
    assert!(
        differing > 0,
        "no witness: the collapse was exact at h = {odd_grid} too, so this test asserts nothing"
    );
    let worst = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max);
    let scale = a.data().iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(
        worst <= 1e-15 * scale,
        "the gap is a reassociation, not a bug"
    );
}

#[test]
fn the_free_plate_nullspace_is_exactly_the_rigid_body_space() {
    // The operator money test for the free edge, and the one that catches a dropped `nu`: the
    // bending blocks kill everything linear per line, the twist block kills everything additively
    // separable, and the intersection is {1, x, y}. The saddle `xy` is NOT null -- it is carried
    // solely by the 2(1-nu) twist term, so a plate that has quietly lost `nu` gets a fourth zero
    // mode and still conserves energy perfectly.
    let (nx, ny, h, nu) = (8usize, 6usize, 0.037, 0.3);
    let (k, _, _) = free_plate_stiffness(nx, ny, h, nu, 1.0, 1.0, None, None);
    let (nrows, ncols) = (ny + 1, nx + 1);
    let coords = |f: &dyn Fn(f64, f64) -> f64| -> Vec<f64> {
        let mut v = Vec::with_capacity(nrows * ncols);
        for j in 0..nrows {
            for i in 0..ncols {
                v.push(f(i as f64 * h, j as f64 * h));
            }
        }
        v
    };
    let scale = k.data().iter().fold(0.0f64, |m, x| m.max(x.abs()));
    for (name, f) in [
        ("1", &(|_x: f64, _y: f64| 1.0) as &dyn Fn(f64, f64) -> f64),
        ("x", &(|x: f64, _y: f64| x) as &dyn Fn(f64, f64) -> f64),
        ("y", &(|_x: f64, y: f64| y) as &dyn Fn(f64, f64) -> f64),
    ] {
        let v = coords(f);
        let worst = k.matvec(&v).iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let amp = v.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1.0);
        assert!(
            worst <= 1e-12 * scale * amp,
            "K did not kill {name}: {worst}"
        );
    }
    let saddle = coords(&|x, y| x * y);
    let twist = k.matvec(&saddle).iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let amp = saddle.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    assert!(
        twist > 1e-6 * scale * amp,
        "the saddle is in the nullspace -- the (1-nu) twist term is missing"
    );
}

#[test]
fn the_free_plate_weight_is_the_trapezoidal_area_and_the_stiffness_is_symmetric() {
    let (nx, ny, h) = (6usize, 5usize, 0.037);
    let (k, w, index_map) = free_plate_stiffness(nx, ny, h, 0.3, 1.0, 1.0, None, None);
    assert!(
        k.is_symmetric(),
        "the free-plate stiffness is not symmetric"
    );
    assert_eq!(index_map.len(), (ny + 1) * (nx + 1));
    // h² interior, h²/2 edge, h²/4 corner -- and it comes out of one expression, `cells/4`.
    let (nrows, ncols) = (ny + 1, nx + 1);
    for j in 0..nrows {
        for i in 0..ncols {
            let p = index_map[j * ncols + i] as usize;
            let edge_j = j == 0 || j == nrows - 1;
            let edge_i = i == 0 || i == ncols - 1;
            let want = match (edge_j, edge_i) {
                (true, true) => 0.25,
                (false, false) => 1.0,
                _ => 0.5,
            } * (h * h);
            assert_eq!(w.get(p, p), want, "node ({j}, {i})");
        }
    }
    // Every plate the core built before model #5g was a rectangle, and the rectangle is now just
    // the mask that happens to be all-ones. One code path, asserted rather than asserted-in-prose.
    let full = Mask::new(nrows, ncols, vec![true; nrows * ncols]);
    let (k2, w2, im2) = free_plate_stiffness_from_mask(&full, h, 0.3, 1.0, 1.0, None, None);
    assert_eq!(k.data(), k2.data());
    assert_eq!(k.indices(), k2.indices());
    assert_eq!(w.data(), w2.data());
    assert_eq!(index_map, im2);
}

#[test]
fn the_free_plate_nullspace_survives_a_staircased_outline() {
    // Model #5g: the same three zero modes on a guitar, where "both neighbours live" is doing real
    // work. A NECESSARY check, not a sufficient one -- it looks identical for an assembly with the
    // wrong boundary treatment, which is why the plan sends the guitar to a derived circular-plate
    // oracle instead.
    let mask = guitar_test_mask(32);
    let h = 1.0 / 32.0;
    let (k, w, index_map) = free_plate_stiffness_from_mask(&mask, h, 0.3, 1.0, 1.0, None, None);
    assert!(k.is_symmetric());
    for p in 0..k.nrows() {
        assert!(w.get(p, p) > 0.0, "a live node carries no area: {p}");
    }
    let ncols = mask.ncols();
    let scale = k.data().iter().fold(0.0f64, |m, x| m.max(x.abs()));
    for (name, f) in [
        ("1", &(|_i: f64, _j: f64| 1.0) as &dyn Fn(f64, f64) -> f64),
        ("x", &(|i: f64, _j: f64| i) as &dyn Fn(f64, f64) -> f64),
        ("y", &(|_i: f64, j: f64| j) as &dyn Fn(f64, f64) -> f64),
    ] {
        let mut v = vec![0.0; k.nrows()];
        for (p, &q) in index_map.iter().enumerate() {
            if q >= 0 {
                v[q as usize] = f((p % ncols) as f64 * h, (p / ncols) as f64 * h);
            }
        }
        let worst = k.matvec(&v).iter().fold(0.0f64, |m, x| m.max(x.abs()));
        let amp = v.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1.0);
        assert!(
            worst <= 1e-12 * scale * amp,
            "K did not kill {name}: {worst}"
        );
    }
}

/// `1/h` computed once and squared — kept out of line so the compiler cannot see which spelling it
/// is being asked for. Plan §17.2: a distinction between two spellings of the same arithmetic is
/// only observable while the compiler cannot fold them into one.
#[inline(never)]
fn twist_as_a_product_of_reciprocals(h: f64) -> f64 {
    let d1 = 1.0 / h;
    d1 * d1
}

/// ... and the same quantity as one reciprocal of a square.
#[inline(never)]
fn twist_as_one_reciprocal(h: f64) -> f64 {
    1.0 / (h * h)
}

#[test]
fn the_twist_coefficient_is_two_reciprocals_and_not_one() {
    // The cell-centred twist is a PRODUCT OF TWO FORWARD FIRST DIFFERENCES, so its coefficient is
    // (1/h)·(1/h). It is not the curvature's 1/h², and the two differ in the last digit whenever h
    // is not exactly representable -- which showed up on exactly one grid of the seven-grid survey
    // this operator was first validated over. Checking one grid would have reported success, so
    // this walks a range and fails if it finds NO witness.
    let grids = [0.037, 0.05, 0.0125, 0.03125, 0.1 / 7.0, 0.5, 1.0 / 3.0];
    let witnesses = grids
        .iter()
        .filter(|&&h| twist_as_a_product_of_reciprocals(h) != twist_as_one_reciprocal(h))
        .count();
    assert!(
        witnesses > 0,
        "no grid distinguishes (1/h)*(1/h) from 1/(h*h) -- this test asserts nothing"
    );
    // ... and the assembly takes the first spelling. A one-cell plate makes the coefficient
    // readable straight off the matrix: with a single live cell the twist block is one row.
    let h = 0.05;
    let full = Mask::new(2, 2, vec![true; 4]);
    let (k, _, _) = free_plate_stiffness_from_mask(&full, h, 0.0, 0.0, 0.0, Some(0.0), Some(0.25));
    // K = 4·g_xy·h²·DxyᵀDxy with everything else switched off; entry (0,0) is (twist)²·h²·4·0.25.
    let t = twist_as_a_product_of_reciprocals(h);
    assert_eq!(k.get(0, 0), (4.0 * 0.25) * (h * h) * (t * t));
    assert_ne!(k.get(0, 0), {
        let u = twist_as_one_reciprocal(h);
        (4.0 * 0.25) * (h * h) * (u * u)
    });
}

#[test]
fn the_gram_association_is_invisible_here_and_that_is_a_property_of_the_values() {
    // `operators2d.py` writes the free plate's Gram products RIGHT-associated, `C2xᵀ @ (Wa @ C2y)`,
    // while its sibling `AiryStressSolver` writes the same mathematical form BᵀWB with no
    // parentheses at all, which Python left-associates. That reads like a hazard -- a shared helper
    // would silently pick one -- and for this operator it turns out to be provably harmless, for a
    // reason that costs no measurement: every curvature entry is the SAME mantissa `1/h²` times an
    // exact power of two, so `(x·w)·z` and `x·(w·z)` differ only by commuting a product, and IEEE
    // multiplication commutes exactly. The port keeps the faithful association anyway, because the
    // property belongs to the values and not to the code.
    //
    // The second half is what stops this being vacuous: a middle factor between two DIFFERENT
    // mantissas does distinguish the two associations, so the argument is about these matrices and
    // not about associativity in general.
    let hh = 0.037 * 0.037;
    let inv_h2 = 1.0 / hh;
    // Two rows of the curvature's own value set: `1/h²` and `-2/h²`, nothing else.
    let a = Csr::from_rows(
        2,
        2,
        vec![
            vec![(0, inv_h2), (1, -2.0 * inv_h2)],
            vec![(0, -2.0 * inv_h2), (1, inv_h2)],
        ],
    );
    // ... and the area weight's own value set, including the 3/4 a staircased rim produces.
    let w = Csr::diagonal(&[0.75 * hh, 0.5 * hh]);
    let right = a.transpose().matmul(&w.matmul(&a));
    let left = a.transpose().matmul(&w).matmul(&a);
    assert_eq!(
        right.data(),
        left.data(),
        "one mantissa: the two must agree"
    );

    // Off that value set the two associations DO part company -- but only for about a third of
    // value triples, so a hand-picked witness is a coin flip and this searches instead. A test
    // that hardcoded three constants and happened to land on the agreeing two-thirds would go
    // green having asserted nothing, which is §23.5 with the roles reversed.
    let mut witness = None;
    for seed in 0..64u64 {
        let v = pseudorandom(8, seed);
        let b = Csr::from_rows(
            2,
            2,
            vec![vec![(0, v[0]), (1, v[1])], vec![(0, v[2]), (1, v[3])]],
        );
        let c = Csr::from_rows(
            2,
            2,
            vec![vec![(0, v[4]), (1, v[5])], vec![(0, v[6]), (1, v[7])]],
        );
        let w2 = Csr::diagonal(&[v[0].abs() + 0.5, v[3].abs() + 0.5]);
        let right2 = b.transpose().matmul(&w2.matmul(&c));
        let left2 = b.transpose().matmul(&w2).matmul(&c);
        if right2.data() != left2.data() {
            witness = Some(seed);
            break;
        }
    }
    assert!(
        witness.is_some(),
        "no witness in 64 draws: the two associations agree even off the plate's value set,          so the first half of this test says nothing"
    );
}

/// A pruned guitar outline at grid `n` — the shipped shape, on the unit square.
fn guitar_test_mask(n: usize) -> Mask {
    let (x, y, _h) = grid_coords(n, 0.5);
    let shifted: Vec<f64> = y.iter().map(|v| v + 0.5).collect();
    let m = guitar_mask(&x, &shifted, 1.0, 0.62, 0.42, 0.30, n + 1, n + 1);
    prune_to_area_carrying(&m).0
}

/// A pruned disk outline at grid `n`.
fn disk_test_mask(n: usize) -> Mask {
    let (x, y, _h) = grid_coords(n, 0.5);
    prune_to_area_carrying(&disk_mask(&x, &y, 0.4, n + 1, n + 1)).0
}

/// A deterministic spread of values in `[-1, 1)` — enough to probe a quadratic form, and no
/// dependency (the crate's Cargo allowlist is empty and stays that way).
fn pseudorandom(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}
