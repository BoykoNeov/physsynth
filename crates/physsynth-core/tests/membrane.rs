//! Native validation of the membrane — the same acceptance contract as the Python harness,
//! asserted without an interpreter anywhere in the picture.
//!
//! As with `string_ideal.rs`, this is a native *floor* rather than a duplicate of the Python
//! suite: if `cargo test` is green the physics is right independently of the binding, so a red
//! Python run is a binding failure and a red run here is a physics failure.
//!
//! The bars are the project's, unchanged: lossless energy drift below `1e-10`, passivity strictly
//! monotone, and the modal frequency against a **closed form** rather than a previous run. The
//! rectangle's is exact, which is why the rectangle exists — it is the unit test that de-risks
//! the harness before the circle's staircased rim enters and degrades the Bessel match to ~O(h).
//!
//! One thing deliberately asserted twice, on both domains: **energy conservation is perpendicular
//! to geometry.** The staircase is an accuracy error, not an energy error, because the masked
//! Laplacian stays symmetric. A test that only ran the rectangle would leave the reader believing
//! the circle is the risky one for the energy ledger. It is not, and that is worth showing.

use physsynth_core::membrane::{energy, Domain, Membrane, ParamError, Params};
use std::f64::consts::PI;

const T: f64 = 100.0;
const RHO: f64 = 0.26; // -> c ~= 19.6 m/s, a plausible drumhead

fn rectangle(n: i64, lx: f64, ly: f64, fs: f64, sigma: f64) -> Membrane {
    Membrane::new(
        Params::new(
            Some(Domain::Rectangle),
            T,
            RHO,
            fs,
            n,
            Some(lx),
            Some(ly),
            None,
            sigma,
        )
        .expect("valid rectangle"),
    )
}

fn circle(n: i64, radius: f64, fs: f64, sigma: f64) -> Membrane {
    Membrane::new(
        Params::new(
            Some(Domain::Circle),
            T,
            RHO,
            fs,
            n,
            None,
            None,
            Some(radius),
            sigma,
        )
        .expect("valid circle"),
    )
}

/// A sample rate that puts the Courant number exactly at `lam` for the given spacing.
fn fs_for(lam: f64, h: f64) -> f64 {
    let c = (T / RHO).sqrt();
    c / (lam * h)
}

/// A smooth radial bump on the live nodes — band-limited, so it excites a broad band without the
/// slowly-converging high partials of a cornered shape.
fn bump(m: &Membrane, cx: f64, cy: f64, width: f64, amplitude: f64) -> Vec<f64> {
    let p = m.params();
    let full: Vec<f64> =
        p.x.iter()
            .zip(p.y.iter())
            .map(|(&xv, &yv)| {
                let d = ((xv - cx) * (xv - cx) + (yv - cy) * (yv - cy)).sqrt();
                if d < width {
                    amplitude * 0.5 * (1.0 + (PI * d / width).cos())
                } else {
                    0.0
                }
            })
            .collect();
    p.to_live(&full)
}

#[test]
fn a_rectangle_conserves_energy() {
    let (lx, ly) = (0.4, 0.3);
    let n = 40;
    let h = lx / n as f64;
    let mut m = rectangle(n, lx, ly, fs_for(0.6, h), 0.0);
    let u0 = bump(&m, 0.17, 0.13, 0.09, 1e-3);
    m.set_displacement(&u0);

    let e0 = m.energy();
    assert!(e0 > 0.0, "a struck membrane must start with energy");

    let mut worst: f64 = 0.0;
    for _ in 0..4_000 {
        m.step();
        worst = worst.max(((m.energy() - e0) / e0).abs());
    }
    assert!(
        worst < 1e-10,
        "lossless drift {worst:e} exceeds the 1e-10 bar"
    );
}

#[test]
fn a_staircased_circle_conserves_energy_just_as_well() {
    // The headline of model #4: the round rim is a *geometry* error, not an energy error, because
    // the masked Laplacian is still symmetric. Same bar as the rectangle, deliberately.
    let radius = 0.15;
    let n = 44;
    let h = 2.0 * radius / n as f64;
    let mut m = circle(n, radius, fs_for(0.6, h), 0.0);
    let u0 = bump(&m, 0.02, -0.01, 0.06, 1e-3);
    m.set_displacement(&u0);

    let e0 = m.energy();
    assert!(e0 > 0.0);

    let mut worst: f64 = 0.0;
    for _ in 0..4_000 {
        m.step();
        worst = worst.max(((m.energy() - e0) / e0).abs());
    }
    assert!(
        worst < 1e-10,
        "lossless drift {worst:e} exceeds the 1e-10 bar"
    );
}

#[test]
fn loss_makes_the_energy_monotonically_decrease() {
    let radius = 0.15;
    let n = 30;
    let h = 2.0 * radius / n as f64;
    let mut m = circle(n, radius, fs_for(0.6, h), 3.0);
    let u0 = bump(&m, 0.0, 0.0, 0.08, 1e-3);
    m.set_displacement(&u0);

    let e0 = m.energy();
    let mut previous = e0;
    assert!(previous > 0.0);
    for i in 0..2_000 {
        m.step();
        let e = m.energy();
        assert!(
            e <= previous,
            "energy rose at step {i}: {previous:e} -> {e:e}"
        );
        previous = e;
    }
    // Monotonicity alone would be satisfied by a model that loses nothing, so pin the magnitude
    // too: `sigma = 3` over 2,000 steps has to take a visible bite out of it.
    assert!(
        previous < 0.5 * e0,
        "sigma > 0 barely dissipated: {e0:e} -> {previous:e}"
    );
}

#[test]
fn energy_is_non_negative_even_at_the_cfl_ceiling() {
    // E^n >= 0 *is* the stability condition (HANDOFF §4.1 step 3). At lambda = 1/sqrt(2) the
    // potential term is at its least positive-definite, so this is where a wrong sign shows.
    let radius = 0.12;
    let n = 26;
    let h = 2.0 * radius / n as f64;
    let mut m = circle(n, radius, fs_for(1.0 / 2.0f64.sqrt(), h), 0.0);
    let u0 = bump(&m, 0.01, 0.0, 0.05, 1e-3);
    m.set_displacement(&u0);

    for i in 0..3_000 {
        m.step();
        let e = m.energy();
        assert!(e >= 0.0, "E^{i} = {e:e} went negative at the CFL ceiling");
    }
}

#[test]
fn a_rectangle_eigenmode_oscillates_at_the_discrete_frequency() {
    // The money test. The 5-point Laplacian on a rectangle has the exact eigenvector
    // sin(m pi x / Lx) sin(q pi y / Ly) with eigenvalue -Lambda, so the scheme's own solution is
    // u^n = cos(omega n k) u^0 with cos(omega k) = 1 - c^2 k^2 Lambda / 2. Checking the field
    // itself, node by node, is far sharper than checking a spectrum: it pins the operator, the
    // start-up step and the update all at once.
    let (lx, ly) = (0.4, 0.4);
    let n = 24;
    let h = lx / n as f64;
    let mut m = rectangle(n, lx, ly, fs_for(0.5, h), 0.0);
    let p = m.params();
    let (nrows, ncols) = p.shape();
    let ny = nrows - 1;
    let ly = p.ly.expect("a rectangle has Ly");
    let (mm, qq) = (2.0, 3.0);

    let full: Vec<f64> = (0..nrows * ncols)
        .map(|idx| {
            let (j, i) = (idx / ncols, idx % ncols);
            (mm * PI * (i as f64) / (n as f64)).sin() * (qq * PI * (j as f64) / (ny as f64)).sin()
        })
        .collect();
    let u0 = p.to_live(&full);

    let lambda = (4.0 / (h * h))
        * ((mm * PI * h / (2.0 * lx)).sin().powi(2) + (qq * PI * h / (2.0 * ly)).sin().powi(2));
    let ck = p.c * p.k;
    let omega_k = (1.0 - ck * ck * lambda / 2.0).acos();

    m.set_displacement(&u0);
    let u_ref = u0.clone();
    let scale = u_ref.iter().fold(0.0f64, |a, &b| a.max(b.abs()));

    let mut worst: f64 = 0.0;
    for step in 1..=400 {
        m.step();
        let expected = (omega_k * step as f64).cos();
        for (got, &want) in m.u.iter().zip(u_ref.iter()) {
            worst = worst.max((got - want * expected).abs() / scale);
        }
    }
    assert!(
        worst < 1e-11,
        "eigenmode departed the closed form by {worst:e}"
    );
}

#[test]
fn the_rectangle_snaps_ly_to_whole_cells() {
    // Cells stay square, so Ly is rounded to an integer number of them and the snapped value is
    // what the model reports. `round_ties_even` in the model is the reason the boundary case
    // below lands where Python puts it.
    let m = rectangle(10, 1.0, 0.37, 4000.0, 0.0);
    let p = m.params();
    assert_eq!(p.h, 0.1);
    assert_eq!(p.ly, Some(4.0 * 0.1));

    // Ly / h == 2.5 exactly: Python's round() goes to even, i.e. DOWN to 2. Half-away-from-zero
    // (Rust's own `f64::round`) would give 3 here, and the resulting membrane would be a
    // different, entirely healthy-looking one.
    let m = rectangle(2, 1.0, 1.25, 200.0, 0.0);
    assert_eq!(m.params().ly, Some(2.0 * 0.5));

    // And 3.5 rounds UP to 4, which is what makes this ties-to-even rather than always-down.
    let m = rectangle(2, 1.0, 1.75, 200.0, 0.0);
    assert_eq!(m.params().ly, Some(4.0 * 0.5));
}

#[test]
fn construction_rejects_what_it_should() {
    let bad = |domain, t, rho, fs, n, lx, ly, r, sigma| {
        Params::new(domain, t, rho, fs, n, lx, ly, r, sigma).unwrap_err()
    };
    let rect = Some(Domain::Rectangle);
    let circ = Some(Domain::Circle);

    assert_eq!(
        bad(rect, 0.0, RHO, 1000.0, 10, Some(1.0), Some(1.0), None, 0.0),
        ParamError::NonPositive
    );
    assert_eq!(
        bad(rect, T, RHO, 1000.0, 1, Some(1.0), Some(1.0), None, 0.0),
        ParamError::TooFewSegments
    );
    assert_eq!(
        bad(rect, T, RHO, 1000.0, 10, Some(1.0), Some(1.0), None, -1.0),
        ParamError::NegativeSigma
    );
    assert_eq!(
        bad(rect, T, RHO, 1000.0, 10, Some(1.0), None, None, 0.0),
        ParamError::RectangleNeedsSides
    );
    assert_eq!(
        bad(rect, T, RHO, 1000.0, 10, Some(1.0), Some(-1.0), None, 0.0),
        ParamError::NonPositiveSides
    );
    assert_eq!(
        bad(circ, T, RHO, 1000.0, 10, None, None, None, 0.0),
        ParamError::CircleNeedsRadius
    );
    assert_eq!(
        bad(circ, T, RHO, 1000.0, 10, None, None, Some(0.0), 0.0),
        ParamError::NonPositiveRadius
    );
    assert_eq!(
        bad(None, T, RHO, 1000.0, 10, Some(1.0), Some(1.0), None, 0.0),
        ParamError::BadDomain
    );
}

#[test]
fn the_two_dimensional_cfl_is_one_over_root_two_not_one() {
    // The 5-point Laplacian's spectral radius is 8/h^2, double the 1-D case, so a Courant number
    // that is perfectly legal for a string is unstable here. A membrane accepting lambda = 1 would
    // be the single easiest way to port this model wrong.
    let radius = 0.15;
    let n = 30;
    let h = 2.0 * radius / n as f64;

    let at_ceiling = Params::new(
        Some(Domain::Circle),
        T,
        RHO,
        fs_for(1.0 / 2.0f64.sqrt(), h),
        n,
        None,
        None,
        Some(radius),
        0.0,
    );
    assert!(at_ceiling.is_ok(), "lambda = 1/sqrt(2) must be accepted");

    let err = Params::new(
        Some(Domain::Circle),
        T,
        RHO,
        fs_for(1.0, h),
        n,
        None,
        None,
        Some(radius),
        0.0,
    )
    .unwrap_err();
    assert!(matches!(err, ParamError::CflViolated(_)));
    assert!(err.to_string().starts_with("CFL violated"));
}

#[test]
fn a_rectangle_one_cell_tall_has_no_unknowns_and_says_so() {
    // The empty-mask rejection must be a parameter error, not a panic on a zero-by-zero matrix.
    // Reaching it takes a rectangle: `Ly` snaps to a single cell, so there is no interior row.
    //
    // Worth recording, because it is not obvious and it bounds what this guard covers: a **disk**
    // can never trip it. `disk_mask` keeps the node nearest the origin for every legal `N` — the
    // centre node itself when `N` is even, and the four at `r/N` when it is odd — so the circular
    // domain always has at least one unknown however coarse the grid.
    let err = Params::new(
        Some(Domain::Rectangle),
        T,
        RHO,
        200.0,
        2,
        Some(1.0),
        Some(0.5),
        None,
        0.0,
    );
    match err {
        Err(ParamError::EmptyMask) => {}
        Err(other) => panic!("expected an empty mask, got {other}"),
        Ok(_) => panic!("a rectangle one cell tall should have no unknowns"),
    }
}

#[test]
fn the_state_embeds_onto_the_full_grid_with_a_dead_rim() {
    let m = rectangle(6, 0.3, 0.3, 4000.0, 0.0);
    let p = m.params();
    let (nrows, ncols) = p.shape();
    let field = m.state();
    assert_eq!(field.len(), nrows * ncols);
    for i in 0..ncols {
        assert_eq!(field[i], 0.0, "the top rim must be clamped");
        assert_eq!(field[(nrows - 1) * ncols + i], 0.0);
    }
}

#[test]
fn a_membrane_at_rest_has_no_energy() {
    let m = circle(20, 0.1, 20_000.0, 0.0);
    assert_eq!(m.energy(), 0.0);
    assert_eq!(
        energy(
            &vec![0.0; m.params().n_live()],
            &vec![0.0; m.params().n_live()],
            m.params()
        ),
        0.0
    );
}

#[test]
fn a_pickup_lands_on_the_nearest_live_node() {
    let m = circle(20, 0.1, 20_000.0, 0.0);
    let p = m.params();
    let centre = p.pickup_index_at(0.0, 0.0);

    // Recover the node's coordinates through the index map and check nothing is nearer.
    let mut coords = Vec::new();
    for (idx, &live) in p.index_map.iter().enumerate() {
        if live >= 0 {
            coords.push((p.x[idx], p.y[idx]));
        }
    }
    let best = coords[centre];
    let best_d2 = best.0 * best.0 + best.1 * best.1;
    for &(cx, cy) in &coords {
        assert!(cx * cx + cy * cy >= best_d2 - 1e-18);
    }
}
