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
    disk_mask, embed, grid_coords, inner2d, laplacian_from_mask, norm2_2d, rectangle_mask, Mask,
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
