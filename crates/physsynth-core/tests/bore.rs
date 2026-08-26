//! Native acceptance bars for `bore` — the same claims `tests/test_bore_*.py` make about the
//! Python original, asserted here against the Rust one without a Python interpreter in the way.
//!
//! The project's contract (`CLAUDE.md`) is that correctness is asserted against closed-form
//! physics, not against a reference implementation. These are that: a lossless tube conserves its
//! discrete energy, a viscous one decreases monotonically, a closed-open cylinder resonates on the
//! **odd** harmonics of `c0 / 4L`, a radiating bell sheds exactly the energy it books, and the
//! construction-time refusals fire in the original's order.
//!
//! Two bars here have no counterpart on any earlier model in this crate, and they are the
//! interesting ones:
//!
//! - **the `source` hook is applied at the one point in the step where it can matter**, which no
//!   energy assertion can see (`the_hook_fires_before_the_momentum_substep`), and
//! - **a bell at both ends books each end's energy separately**, which is a floating-point claim
//!   about the *order* of two additions, not about the physics
//!   (`a_two_ended_bell_books_each_end_separately`).

use physsynth_core::bore::{self, Bore, End, ParamError, Params};

const RHO0: f64 = 1.2041;
const C0: f64 = 343.0;

/// A clarinet-shaped tube whose Courant number is exactly `lam`, as `tests/helpers.make_bore` does.
fn params(n: usize, lam: f64, bc: (End, End), sigma: f64, r_bell: f64) -> Params {
    let l = 0.6;
    let h = l / (n as f64);
    let fs = C0 / (lam * h);
    Params::new(l, fs, n, 0.008, Some(bc), sigma, r_bell, RHO0, C0)
        .expect("parameters should be accepted")
}

/// A narrow pressure bump, the initial condition the bore's own tests use.
fn bump(p: &Params, center_frac: f64) -> Vec<f64> {
    let x = p.grid();
    let c = center_frac * p.l;
    let w = 0.04 * p.l;
    x.iter()
        .map(|&xi| (-(((xi - c) / w) * ((xi - c) / w))).exp())
        .collect()
}

// -- construction ------------------------------------------------------------------------------

#[test]
fn non_physical_parameters_are_rejected_in_the_originals_order() {
    let ok = |l, fs, n, radius, sigma, r_bell, rho0, c0| {
        Params::new(
            l,
            fs,
            n,
            radius,
            Some((End::Closed, End::Open)),
            sigma,
            r_bell,
            rho0,
            c0,
        )
    };
    assert_eq!(
        ok(0.0, 48000.0, 60, 0.008, 0.0, 0.0, RHO0, C0).unwrap_err(),
        ParamError::NonPositiveScalar
    );
    assert_eq!(
        ok(0.6, 48000.0, 1, 0.008, 0.0, 0.0, RHO0, C0).unwrap_err(),
        ParamError::TooFewSegments
    );
    assert_eq!(
        ok(0.6, 48000.0, 60, 0.008, -1.0, 0.0, RHO0, C0).unwrap_err(),
        ParamError::NegativeSigma
    );
    assert_eq!(
        ok(0.6, 48000.0, 60, 0.008, 0.0, -1.0, RHO0, C0).unwrap_err(),
        ParamError::NegativeRBell
    );

    // A non-physical scalar wins over a bad `N`, and `N` wins over a negative `sigma` — the order
    // is the original's, so a call with two faults reports the same one.
    assert_eq!(
        ok(0.0, 48000.0, 1, 0.008, -1.0, -1.0, RHO0, C0).unwrap_err(),
        ParamError::NonPositiveScalar
    );
    assert_eq!(
        ok(0.6, 48000.0, 1, 0.008, -1.0, 0.0, RHO0, C0).unwrap_err(),
        ParamError::TooFewSegments
    );
}

#[test]
fn a_radiating_end_needs_a_positive_resistance() {
    let err = Params::new(
        0.6,
        48000.0,
        60,
        0.008,
        Some((End::Closed, End::Radiating)),
        0.0,
        0.0,
        RHO0,
        C0,
    )
    .unwrap_err();
    assert_eq!(err, ParamError::RadiatingNeedsResistance);
    assert!(err
        .to_string()
        .starts_with("a 'radiating' end needs R_bell > 0"));

    // ...but `R_bell > 0` with no radiating end is legal, and does nothing.
    assert!(Params::new(
        0.6,
        48000.0,
        60,
        0.008,
        Some((End::Closed, End::Open)),
        0.0,
        650.0,
        RHO0,
        C0
    )
    .is_ok());
}

#[test]
fn the_boundary_check_runs_after_the_scalar_checks_and_before_cfl() {
    // A bad token with a fine geometry is the boundary rejection...
    assert_eq!(
        Params::new(0.6, 48000.0, 60, 0.008, None, 0.0, 0.0, RHO0, C0).unwrap_err(),
        ParamError::BadBoundary
    );
    // ...and with a CFL-violating rate it is *still* the boundary rejection, because the boundary
    // is parsed first. Getting this backwards changes which message a doubly-wrong call gets.
    assert_eq!(
        Params::new(0.6, 4_000_000.0, 60, 0.008, None, 0.0, 0.0, RHO0, C0).unwrap_err(),
        ParamError::BadBoundary
    );
}

#[test]
fn the_cfl_ceiling_is_lambda_one_and_the_message_quotes_it() {
    let l = 0.6;
    let n = 60usize;
    let h = l / (n as f64);
    // lam = 1 exactly is accepted (it is the dispersionless sweet spot, not a violation).
    let p = Params::new(
        l,
        C0 / h,
        n,
        0.008,
        Some((End::Closed, End::Open)),
        0.0,
        0.0,
        RHO0,
        C0,
    )
    .expect("lambda = 1 is legal");
    assert!((p.lam - 1.0).abs() < 1e-12);

    let err = Params::new(
        l,
        (C0 / h) / 1.5,
        n,
        0.008,
        Some((End::Closed, End::Open)),
        0.0,
        0.0,
        RHO0,
        C0,
    )
    .unwrap_err();
    assert!(matches!(err, ParamError::CflViolated(_)));
    assert_eq!(
        err.to_string(),
        "CFL violated: lambda = c0*k/h = 1.500000 > 1. \
         Reduce fs, refine the grid (increase N), or shorten the tube."
    );
}

#[test]
fn the_trapezoidal_weight_is_a_half_cell_at_each_end() {
    let p = params(60, 1.0, (End::Closed, End::Open), 0.0, 0.0);
    assert_eq!(p.w[0], 0.5 * p.h);
    assert_eq!(p.w[p.n], 0.5 * p.h);
    assert_eq!(p.w[1], p.h);
    // ...and the half-cell shows up in the compliance and hence in the end node's `a`.
    assert_eq!(p.a_left, p.c[0] / p.k);
    assert!((p.c[0] / p.c[1] - 0.5).abs() < 1e-15);
}

#[test]
fn the_open_ends_are_the_ones_dropped_from_the_dof_set() {
    let closed_open = params(8, 1.0, (End::Closed, End::Open), 0.0, 0.0);
    assert_eq!(closed_open.dof(), vec![0, 1, 2, 3, 4, 5, 6, 7]);

    let open_open = params(8, 1.0, (End::Open, End::Open), 0.0, 0.0);
    assert_eq!(open_open.dof(), vec![1, 2, 3, 4, 5, 6, 7]);

    // A radiating end stays a live DOF — it is a half-cell node like a wall, not a pin.
    let radiating = params(8, 1.0, (End::Closed, End::Radiating), 0.0, 650.0);
    assert_eq!(radiating.dof(), vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
}

#[test]
fn the_pressure_operator_is_symmetric_and_annihilates_a_constant() {
    let p = params(16, 1.0, (End::Closed, End::Closed), 0.0, 0.0);
    let (lop, cmat) = p.pressure_operator();
    assert!(lop.is_symmetric());
    assert_eq!(cmat.nnz(), p.nodes());

    // `L = G^T M^{-1} G` with `G` a difference operator, so a constant field is in its nullspace —
    // the closed-closed tube's zero-frequency (uniform compression) mode.
    let ones = vec![1.0; p.nodes()];
    for v in lop.matvec(&ones) {
        assert!(v.abs() < 1e-9, "L annihilates a constant, got {v}");
    }
}

// -- the step ----------------------------------------------------------------------------------

#[test]
fn the_divergence_is_the_wall_closure_at_both_ends() {
    let p = params(4, 1.0, (End::Closed, End::Closed), 0.0, 0.0);
    let u = vec![1.0, 2.0, 4.0, 8.0];
    let mut div = vec![0.0; p.nodes()];
    bore::divergence_into(&u, &mut div, &p);
    // Node 0 sees only the segment to its right, node N only the one to its left.
    assert_eq!(div, vec![1.0, 2.0 - 1.0, 4.0 - 2.0, 8.0 - 4.0, -8.0]);
}

#[test]
fn a_lossless_closed_open_tube_conserves_its_energy() {
    let p = params(60, 1.0, (End::Closed, End::Open), 0.0, 0.0);
    let mut b = Bore::new(p.clone());
    b.set_state(&bump(&p, 0.5), &vec![0.0; p.n]);
    let e0 = b.energy();
    assert!(e0 > 0.0);
    for _ in 0..4000 {
        b.step(None);
        let drift = (b.energy() - e0).abs() / e0;
        assert!(drift < 1e-10, "lossless drift {drift:.3e} exceeds the bar");
    }
}

#[test]
fn a_lossless_closed_closed_tube_conserves_its_energy() {
    // The all-wall case: every node is a live DOF, so a boundary bug has nowhere to hide.
    let p = params(48, 0.9, (End::Closed, End::Closed), 0.0, 0.0);
    let mut b = Bore::new(p.clone());
    b.set_state(&bump(&p, 0.3), &vec![0.0; p.n]);
    let e0 = b.energy();
    for _ in 0..3000 {
        b.step(None);
        assert!((b.energy() - e0).abs() / e0 < 1e-10);
    }
}

#[test]
fn a_viscous_tube_decreases_monotonically() {
    let p = params(60, 1.0, (End::Closed, End::Open), 30.0, 0.0);
    let mut b = Bore::new(p.clone());
    b.set_state(&bump(&p, 0.5), &vec![0.0; p.n]);
    let e0 = b.energy();
    let mut previous = e0;
    for _ in 0..2000 {
        b.step(None);
        let now = b.energy();
        assert!(now <= previous, "energy rose: {previous} -> {now}");
        previous = now;
    }
    // ...and it did not merely fail to rise: the drag actually took most of it.
    assert!(previous < 0.5 * e0, "sigma > 0 should have damped the tube");
}

#[test]
fn the_initial_energy_is_exactly_the_acoustic_potential() {
    // Starting from rest, `U^{-1/2} = 0`, so the cross-time inductive term is zero and `E^0` is
    // the compliance term alone. This is what makes a single-mode IC oscillate cleanly.
    let p = params(60, 1.0, (End::Closed, End::Open), 0.0, 0.0);
    let mut b = Bore::new(p.clone());
    let p0 = bump(&p, 0.5);
    b.set_state(&p0, &vec![0.0; p.n]);
    let want: f64 = 0.5
        * (0..p.nodes())
            .map(|l| p.c[l] * (p0[l] * p0[l]))
            .sum::<f64>();
    assert!((b.energy() - want).abs() / want < 1e-14);
}

#[test]
fn an_open_end_is_pinned_from_the_initial_condition_onward() {
    let p = params(30, 1.0, (End::Open, End::Open), 0.0, 0.0);
    let mut b = Bore::new(p.clone());
    b.set_state(&vec![1.0; p.nodes()], &vec![0.0; p.n]);
    assert_eq!(b.p()[0], 0.0);
    assert_eq!(b.p()[p.n], 0.0);
    for _ in 0..50 {
        b.step(None);
        assert_eq!(b.p()[0], 0.0);
        assert_eq!(b.p()[p.n], 0.0);
    }
}

// -- the clarinet signature --------------------------------------------------------------------

#[test]
fn a_closed_open_cylinder_resonates_on_the_odd_harmonics() {
    // The model-specific oracle: a quarter-wave tube has a pressure antinode at the wall and a node
    // at the open end, so `f_n = (2n-1) c0 / 4L`. Read off the eigenvalues of the pressure
    // operator restricted to the free nodes rather than from a spectrum, so the bar is exact.
    let p = params(200, 1.0, (End::Closed, End::Open), 0.0, 0.0);
    let got = lowest_frequencies(&p, 3);
    let f1 = C0 / (4.0 * p.l);
    for (i, &f) in got.iter().enumerate() {
        let want = ((2 * (i + 1) - 1) as f64) * f1;
        let cents = 1200.0 * (f / want).log2();
        assert!(
            cents.abs() < 5.0,
            "partial {} is {f:.3} Hz, want {want:.3} Hz ({cents:.2} cents off)",
            i + 1
        );
    }
}

#[test]
fn an_open_open_tube_resonates_on_the_full_harmonic_series() {
    let p = params(200, 1.0, (End::Open, End::Open), 0.0, 0.0);
    let got = lowest_frequencies(&p, 3);
    let f1 = C0 / (2.0 * p.l);
    for (i, &f) in got.iter().enumerate() {
        let want = ((i + 1) as f64) * f1;
        let cents = 1200.0 * (f / want).log2();
        assert!(
            cents.abs() < 5.0,
            "partial {} is {f:.3} Hz, want {want:.3}",
            i + 1
        );
    }
}

/// The `n_modes` lowest continuum eigenfrequencies of `(L, C)` on the free nodes.
///
/// The scheme eliminates `U` to `C d_tt p = -L p`, so `Lambda = omega^2`. `tests/helpers.py` hands
/// this to a generalized `eigsh`; this crate has no eigensolver and does not need one for a
/// tridiagonal, so it is inverse iteration with deflation — the matrix is **inverted once** and
/// each iteration is a matvec, which is the difference between this file taking seconds and taking
/// minutes.
fn lowest_frequencies(p: &Params, n_modes: usize) -> Vec<f64> {
    let dof = p.dof();
    let m = dof.len();
    let (lop, _) = p.pressure_operator();

    // `A = C^{-1/2} L C^{-1/2}` restricted to the free nodes: symmetric, so the generalized problem
    // becomes a standard one and plain inverse iteration converges to the smallest eigenvalue.
    let s: Vec<f64> = dof.iter().map(|&i| 1.0 / p.c[i].sqrt()).collect();
    let mut a = vec![vec![0.0; m]; m];
    for (r, &i) in dof.iter().enumerate() {
        for (c, &j) in dof.iter().enumerate() {
            a[r][c] = s[r] * lop.get(i, j) * s[c];
        }
    }
    // A small positive shift keeps the inverse finite if the operator has a nullspace (the
    // closed-closed tube's uniform-compression mode); it is negligible against omega^2 ~ 1e6.
    let mut shifted = a.clone();
    for (r, row) in shifted.iter_mut().enumerate() {
        row[r] += 1.0;
    }
    let inverse = invert(&shifted);

    let mut found: Vec<Vec<f64>> = Vec::new();
    let mut freqs = Vec::new();
    for _ in 0..n_modes {
        let mut v: Vec<f64> = (0..m)
            .map(|i| ((i * 37 + 11) % 101) as f64 - 50.0)
            .collect();
        let mut lambda = 0.0;
        for _ in 0..400 {
            orthogonalize(&mut v, &found);
            v = matvec(&inverse, &v);
            normalize(&mut v);
            lambda = dot(&v, &matvec(&a, &v));
        }
        orthogonalize(&mut v, &found);
        normalize(&mut v);
        found.push(v);
        freqs.push(lambda.max(0.0).sqrt() / (2.0 * std::f64::consts::PI));
    }
    freqs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    freqs
}

fn orthogonalize(v: &mut [f64], basis: &[Vec<f64>]) {
    for b in basis {
        let d = dot(v, b);
        for (x, &bi) in v.iter_mut().zip(b) {
            *x -= d * bi;
        }
    }
}

fn normalize(v: &mut [f64]) {
    let norm = dot(v, v).sqrt();
    for x in v.iter_mut() {
        *x /= norm;
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn matvec(a: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    a.iter().map(|row| dot(row, v)).collect()
}

/// Gauss-Jordan inversion with partial pivoting — a test helper, not physics.
#[allow(clippy::needless_range_loop)] // row-and-column indices; the iterator form is unreadable
fn invert(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut m: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.extend((0..n).map(|j| if i == j { 1.0 } else { 0.0 }));
            r
        })
        .collect();
    for col in 0..n {
        let piv = (col..n)
            .max_by(|&i, &j| m[i][col].abs().partial_cmp(&m[j][col].abs()).unwrap())
            .unwrap();
        m.swap(col, piv);
        let d = m[col][col];
        for j in col..2 * n {
            m[col][j] /= d;
        }
        for row in 0..n {
            if row != col && m[row][col] != 0.0 {
                let f = m[row][col];
                for j in col..2 * n {
                    m[row][j] -= f * m[col][j];
                }
            }
        }
    }
    m.into_iter().map(|row| row[n..].to_vec()).collect()
}

// -- the radiating bell ------------------------------------------------------------------------

#[test]
fn a_radiating_bell_sheds_exactly_what_it_books() {
    // The passivity money test: the field energy falls, the booked `radiated_energy` rises, and
    // the total is conserved to machine precision. A sign error at either end shows up here.
    let p = params(60, 1.0, (End::Closed, End::Radiating), 0.0, 650.0);
    let mut b = Bore::new(p.clone());
    b.set_state(&bump(&p, 0.3), &vec![0.0; p.n]);
    let e0 = b.energy();
    for _ in 0..4000 {
        b.step(None);
        assert!((b.energy() - e0).abs() / e0 < 1e-10);
    }
    assert!(b.radiated_energy() > 0.0, "the bell should have radiated");
    assert!(
        b.acoustic_energy() < e0,
        "the field energy should have fallen"
    );
}

#[test]
fn a_bell_at_both_ends_still_conserves_the_total() {
    let p = params(60, 1.0, (End::Radiating, End::Radiating), 0.0, 650.0);
    let mut b = Bore::new(p.clone());
    b.set_state(&bump(&p, 0.5), &vec![0.0; p.n]);
    let e0 = b.energy();
    for _ in 0..3000 {
        b.step(None);
        assert!((b.energy() - e0).abs() / e0 < 1e-10);
    }
    assert!(b.radiated_energy() > 0.0);
}

#[test]
fn a_two_ended_bell_books_each_end_separately() {
    // Not a physics claim — a floating-point one about the *order* of two additions. The original
    // books inside `_radiate_node`, once per end, so a two-ended bell accumulates `(E + e_l) + e_r`
    // and never `E + (e_l + e_r)`. A single step rarely distinguishes the two; a run of them
    // reliably does, and the difference is invisible to every energy bar in this file.
    let p = params(31, 0.8, (End::Radiating, End::Radiating), 0.0, 650.0);
    let mut b = Bore::new(p.clone());
    b.set_state(&bump(&p, 0.4), &vec![0.0; p.n]);

    let bcoef = 0.5 / p.r_bell;
    let mut summed_first = 0.0;
    for _ in 0..2000 {
        let p_old_left = b.p()[0];
        let p_old_right = b.p()[p.n];
        b.step(None);
        // Recover each end's `U_out` from the committed pressures, as `_radiate_node` computed it.
        let u_l = bcoef * (b.p()[0] + p_old_left);
        let u_r = bcoef * (b.p()[p.n] + p_old_right);
        let e_l = p.k * p.r_bell * u_l * u_l;
        let e_r = p.k * p.r_bell * u_r * u_r;
        summed_first += e_l + e_r;
    }

    // The physics agrees to any tolerance anyone would ask for...
    let sequential = b.radiated_energy();
    assert!((sequential - summed_first).abs() / sequential < 1e-12);
    // ...and the last bits do not, which is the whole point: a port that summed the two ends
    // before booking them would pass every other test in this file.
    assert!(
        sequential != summed_first,
        "the two accumulation orders did not diverge — this configuration cannot see the claim"
    );
}

#[test]
fn a_matched_load_is_anechoic_and_a_stiff_one_is_a_wall() {
    // `r = (R - Z0)/(R + Z0)`: at `R = Z0` a pulse hitting the end is absorbed entirely, and as
    // `R -> infinity` the bell becomes a rigid wall that radiates nothing.
    // `Z0` needs a legal bore to read it off, and a radiating end needs a resistance — so build
    // the reference with an open end, whose `Z0` is the same tube's.
    let z0 = params(120, 1.0, (End::Closed, End::Open), 0.0, 0.0).z0;

    let matched = params(120, 1.0, (End::Closed, End::Radiating), 0.0, z0);
    let mut b = Bore::new(matched.clone());
    b.set_state(&bump(&matched, 0.5), &vec![0.0; matched.n]);
    let e0 = b.energy();
    for _ in 0..1200 {
        b.step(None);
    }
    // One traversal and out: essentially everything has left the tube.
    assert!(
        b.acoustic_energy() < 1e-3 * e0,
        "a matched load should absorb the pulse, {} left of {e0}",
        b.acoustic_energy()
    );

    let stiff = params(120, 1.0, (End::Closed, End::Radiating), 0.0, 1e12 * z0);
    let mut b = Bore::new(stiff.clone());
    b.set_state(&bump(&stiff, 0.5), &vec![0.0; stiff.n]);
    let e0 = b.energy();
    for _ in 0..1200 {
        b.step(None);
    }
    assert!(
        b.acoustic_energy() > 0.999 * e0,
        "a very stiff load should behave as a wall"
    );
}

#[test]
fn the_far_field_readout_is_zero_without_a_radiating_end() {
    let p = params(40, 1.0, (End::Closed, End::Open), 0.0, 0.0);
    let mut b = Bore::new(p.clone());
    b.set_state(&bump(&p, 0.5), &vec![0.0; p.n]);
    for _ in 0..200 {
        b.step(None);
        assert_eq!(b.radiated_pressure(), 0.0);
        assert_eq!(b.radiated_energy(), 0.0);
    }
}

// -- the source hook ---------------------------------------------------------------------------

#[test]
fn an_inert_hook_is_bit_for_bit_the_undriven_bore() {
    // `tests/test_reed_stability.py` asserts exactly this on the Python side, with its own
    // `lambda p: None`. It is the bar that keeps the hook from being a hidden perturbation.
    let p = params(40, 1.0, (End::Closed, End::Radiating), 0.0, 650.0);
    let mut driven = Bore::new(p.clone());
    let mut plain = Bore::new(p.clone());
    driven.set_state(&bump(&p, 0.4), &vec![0.0; p.n]);
    plain.set_state(&bump(&p, 0.4), &vec![0.0; p.n]);
    for _ in 0..500 {
        driven.step(Some(&mut |_: &mut [f64]| {}));
        plain.step(None);
    }
    assert_eq!(driven.p(), plain.p());
    assert_eq!(driven.u(), plain.u());
    assert_eq!(driven.radiated_energy(), plain.radiated_energy());
}

#[test]
fn the_hook_fires_before_the_momentum_substep() {
    // The ordering claim, and the reason it needs its own test: no energy assertion can see it.
    // A hook that bumped node 0 must be visible in `U^{n+3/2}` *this* step, not next.
    let p = params(20, 1.0, (End::Closed, End::Open), 0.0, 0.0);
    let mut hooked = Bore::new(p.clone());
    let mut plain = Bore::new(p.clone());
    hooked.set_state(&bump(&p, 0.5), &vec![0.0; p.n]);
    plain.set_state(&bump(&p, 0.5), &vec![0.0; p.n]);

    hooked.step(Some(&mut |pn: &mut [f64]| pn[0] += 1.0));
    plain.step(None);

    // The pressure difference is the bump itself...
    assert_eq!(hooked.p()[0] - plain.p()[0], 1.0);
    // ...and the velocity in the very first segment has already responded to it, which it could
    // not have done if the hook ran after the momentum sub-step.
    let want = -p.u_pref[0] * (-1.0) / 1.0;
    assert!(
        (hooked.u()[0] - plain.u()[0] - want).abs() < 1e-18,
        "U^{{n+3/2}} did not see the hook's correction"
    );
}

#[test]
fn the_hook_fires_before_the_radiating_drain() {
    // The other half of the ordering: an injection at a *radiating* node must be drained by the
    // bell in the same step, not survive to the next one untouched.
    let p = params(20, 1.0, (End::Closed, End::Radiating), 0.0, 650.0);
    let mut b = Bore::new(p.clone());
    b.set_state(&vec![0.0; p.nodes()], &vec![0.0; p.n]);
    assert_eq!(b.radiated_energy(), 0.0);
    b.step(Some(&mut |pn: &mut [f64]| {
        let last = pn.len() - 1;
        pn[last] += 1.0;
    }));
    assert!(
        b.radiated_energy() > 0.0,
        "the bell should have drained the injection in the same step"
    );
}

// -- the struct and the free functions agree ---------------------------------------------------

#[test]
fn the_free_functions_and_the_struct_agree() {
    let p = params(24, 0.95, (End::Closed, End::Open), 2.0, 0.0);
    let mut b = Bore::new(p.clone());
    let p0 = bump(&p, 0.5);
    b.set_state(&p0, &vec![0.0; p.n]);

    let mut expected_p = p0.clone();
    bore::apply_open_ends(&mut expected_p, &p);
    let mut expected_u = vec![0.0; p.n];
    bore::momentum_into(&expected_p, &vec![0.0; p.n], &mut expected_u, &p);
    assert_eq!(b.p(), expected_p.as_slice());
    assert_eq!(b.u(), expected_u.as_slice());

    for _ in 0..64 {
        let mut next = vec![0.0; p.nodes()];
        let mut scratch = vec![0.0; p.nodes()];
        bore::pressure_into(b.p(), b.u(), &mut next, &mut scratch, &p);
        let mut u_next = vec![0.0; p.n];
        bore::momentum_into(&next, b.u(), &mut u_next, &p);
        let energy = bore::acoustic_energy(b.p(), b.u(), b.u_prev(), &p);
        assert_eq!(energy, b.acoustic_energy());

        b.step(None);
        assert_eq!(b.p(), next.as_slice());
        assert_eq!(b.u(), u_next.as_slice());
    }
}
