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
    cells_per_node, disk_mask, embed, grid_coords, guitar_area, guitar_half_width, guitar_mask,
    guitar_scale, inner2d, laplacian_from_mask, live_cells, norm2_2d, prune_to_area_carrying,
    rectangle_mask, Mask,
};

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
