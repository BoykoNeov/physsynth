//! The room's structural, modal and cross-model bars that `src/airbox.rs`'s own tests did not
//! already make — carried from `tests/test_airbox_energy.py`, `test_airbox_modal.py` and
//! `test_airbox_freefield.py` (retirement plan §44), at the Python's own rigs.
//!
//! Most of those three files' claims were already native, in `src/airbox.rs`'s `mod tests`
//! (plan §46); each bar here names the Python test it carries. What is here is what was not:
//!
//! * the **three energy channels** with every one of them live — a lossy wall at three
//!   impedances, all six walls absorbing, open faces, a source next to and *on* a lossy wall, and
//!   two sources in one step;
//! * the **refusals**, each asserted by variant and by the original's message;
//! * the **modal tier at every index**: one step from every axial mode, Nyquist included, and from
//!   a 27-mode oblique grid, then five modes tracked over the whole field rather than one node;
//! * the two **cross-model** oracles — the wall's reflection coefficient against
//!   `(zeta - 1)/(zeta + 1)`, and a one-cell-thick room against the repo's [`Bore`] for 4,000
//!   steps.
//!
//! The broadband initial field is a structureless hash rather than NumPy's PCG64 (§15: a bar on a
//! broadband field is a claim about the class of fields), so those rigs cannot reproduce the
//! Python's figures digit for digit; every deterministic rig here does (§44.1).

use physsynth_core::airbox::{
    self, impedance_from_zeta, mode_frequency, mode_shape, mode_velocity, node_index, AirBox,
    ParamError, Params, Wall, C0_AIR, FACES, RHO0_AIR,
};
use physsynth_core::bore::{self, Bore, End};

/// Acceptance criterion 1 — the same bar as every other resonator (`CLAUDE.md`).
const DRIFT_TOL: f64 = 1e-10;
/// The discrete mode is exact, not approximate: held to the energy bar's neighbourhood.
const EXACT_TOL: f64 = 1e-12;

// -- the fixture, `tests/helpers.py::make_airbox` ------------------------------------------------

/// `AIRBOX_ROOM_DEFAULT` at `AIRBOX_H_DEFAULT`: `N = (9, 7, 6)`.
const ROOM: [f64; 3] = [0.9, 0.7, 0.6];
const H: f64 = 0.1;
/// `lambda = CFL / sqrt(3)`.
const CFL: f64 = 0.9;

fn rigid() -> [Wall; 6] {
    [Wall::Rigid; 6]
}

fn z_matched() -> f64 {
    impedance_from_zeta(1.0, RHO0_AIR, C0_AIR)
}

fn face(name: &str) -> usize {
    FACES.iter().position(|f| *f == name).expect("a face name")
}

fn walls_on(faces: &[&str], w: Wall) -> [Wall; 6] {
    let mut out = rigid();
    for f in faces {
        out[face(f)] = w;
    }
    out
}

/// A room whose Courant number is exactly `cfl / sqrt(3)`: `h` fixes the grid, so the sample rate
/// is solved for. The `sqrt(3)` is written here, not read off the model (§29.3).
fn params(l: [f64; 3], h: f64, cfl: f64, walls: [Wall; 6], source: Option<[f64; 3]>) -> Params {
    let fs = C0_AIR * 3.0f64.sqrt() / (cfl * h);
    Params::new(l, fs, h, walls, source, RHO0_AIR, C0_AIR).unwrap()
}

fn room(walls: [Wall; 6]) -> AirBox {
    AirBox::new(params(ROOM, H, CFL, walls, None))
}

/// A structureless field on every node — the broadband initial condition.
fn noise(n: usize) -> Vec<f64> {
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
}

fn seeded(walls: [Wall; 6]) -> AirBox {
    let mut b = room(walls);
    let p0 = noise(b.p.n_nodes());
    b.set_state(&p0, None);
    b
}

/// Largest of `xs`, and NaN if any is — `np.max`, not `fold(_, f64::max)`, which drops a NaN.
fn nan_max<I: IntoIterator<Item = f64>>(xs: I) -> f64 {
    let mut m = f64::NEG_INFINITY;
    for x in xs {
        if x.is_nan() {
            return f64::NAN;
        }
        m = m.max(x);
    }
    m
}

/// Relative spread of the conserved total over **every** step of a run.
fn drift(b: &mut AirBox, steps: usize) -> f64 {
    let e0 = b.energy();
    let (mut lo, mut hi) = (e0, e0);
    for _ in 0..steps {
        b.step();
        let e = b.energy();
        if e.is_nan() {
            return f64::NAN;
        }
        lo = lo.min(e);
        hi = hi.max(e);
    }
    (hi - lo) / e0.abs()
}

/// `tests/helpers.py::gaussian_pulse` — `q(t)` with `sigma = 1/(2 pi f0)`, centred four standard
/// deviations in, amplitude 1e-3.
fn gaussian_q(f0: f64) -> impl Fn(f64) -> f64 {
    let sigma = 1.0 / (2.0 * std::f64::consts::PI * f0);
    let t0 = 4.0 * sigma;
    move |t: f64| 1e-3 * (-((t - t0) * (t - t0)) / (2.0 * sigma * sigma)).exp()
}

// -- the three energy channels ------------------------------------------------------------------

/// `test_energy_positive_and_no_source_or_loss_channel` — with rigid walls and no source the
/// conserved total *is* the stored energy, bit for bit.
#[test]
fn with_no_source_and_no_loss_the_total_is_the_stored_energy_exactly() {
    let mut b = seeded(rigid());
    for n in 0..50 {
        b.step();
        assert!(b.energy() > 0.0, "step {n}: {}", b.energy());
    }
    assert_eq!(b.dissipated, 0.0);
    assert_eq!(b.injected, 0.0);
    assert_eq!(b.energy().to_bits(), b.acoustic_energy().to_bits());
}

/// `test_impedance_wall_passive_and_booked` — the stored energy falls monotonically while
/// `stored + dissipated` stays flat: the wall channel captures exactly what leaves. A sign error
/// in the flux shows up here and nowhere else.
#[test]
fn an_impedance_wall_only_takes_and_books_every_joule() {
    for zeta in [0.25, 1.0, 4.0] {
        let mut b = seeded(walls_on(
            &["x0"],
            Wall::Impedance(impedance_from_zeta(zeta, RHO0_AIR, C0_AIR)),
        ));
        let e0 = b.energy();
        let mut prev = b.acoustic_energy();
        let mut worst = 0.0f64;
        for n in 0..600 {
            b.step();
            worst = nan_max([worst, (b.energy() - e0).abs() / e0.abs()]);
            let stored = b.acoustic_energy();
            assert!(
                stored <= prev + 1e-18,
                "zeta {zeta}, step {n}: stored energy rose — the wall put energy back in"
            );
            prev = stored;
        }
        assert!(worst < DRIFT_TOL, "zeta {zeta}: ledger {worst:e}");
        assert!(b.dissipated > 0.0, "zeta {zeta}: the wall took nothing");
    }
}

/// `test_every_face_absorbing_still_books` — all six walls lossy at once, so edge and corner nodes
/// pay into two and three faces. Summed admittances in `beta` and summed flux in the book have to
/// agree, or the total drifts exactly where the geometry is most awkward.
#[test]
fn every_face_absorbing_still_books() {
    let mut b = seeded([Wall::Impedance(z_matched()); 6]);
    let d = drift(&mut b, 600);
    assert!(d < DRIFT_TOL, "drift {d:e}");
    assert!(
        b.dissipated / b.energy() > 0.9,
        "a matched room should drain"
    );
}

/// `test_open_face_is_lossless` — an ideal pressure-release face reflects perfectly and radiates
/// nothing: its flux `p u` vanishes because `p` does.
#[test]
fn an_open_face_is_lossless() {
    let mut b = seeded(walls_on(&["y1", "x0"], Wall::Open));
    let d = drift(&mut b, 400);
    assert!(d < DRIFT_TOL, "drift {d:e}");
    assert_eq!(b.dissipated, 0.0);
}

/// `test_source_booking_flat` — `stored + dissipated - injected` is flat while a soft source drives
/// an absorbing room: all three channels live at once.
#[test]
fn a_source_driving_an_absorbing_room_books_flat() {
    let mut b = room(walls_on(&["x1"], Wall::Impedance(z_matched())));
    b.set_state(&vec![0.0; b.p.n_nodes()], None);
    let q = gaussian_q(900.0);
    let mut worst = 0.0f64;
    for n in 0..500 {
        b.inject(q(n as f64 * b.p.k), None);
        b.step();
        worst = nan_max([worst, b.energy().abs()]);
    }
    assert!(b.injected > 0.0, "the source did no work");
    assert!(
        b.dissipated > 0.0,
        "the wall took nothing, so its channel was not live"
    );
    assert!(worst / b.injected < DRIFT_TOL, "{:e}", worst / b.injected);
}

/// `test_source_at_a_wall_node_books_correctly` — a source *on* a lossy wall shares its node with
/// the closure. The injection is folded into the pressure update before the 1x1 wall solve, so the
/// booked `pbar` is the post-solve one; taking it pre-solve would leak here and only here.
#[test]
fn a_source_on_a_lossy_wall_node_books_correctly() {
    let p = params(
        ROOM,
        H,
        CFL,
        walls_on(&["x0"], Wall::Impedance(z_matched())),
        Some([0.0, 0.3, 0.3]),
    );
    assert_eq!(p.source_index, [0, 3, 3]);
    assert!(
        p.beta[airbox::flat(p.p_shape(), 0, 3, 3)] > 0.0,
        "the source node is a wall node"
    );
    let mut b = AirBox::new(p);
    b.set_state(&vec![0.0; b.p.n_nodes()], None);
    let q = gaussian_q(900.0);
    for n in 0..400 {
        b.inject(q(n as f64 * b.p.k), None);
        b.step();
    }
    assert!(b.injected > 0.0);
    let ratio = b.energy().abs() / b.injected;
    assert!(ratio < DRIFT_TOL, "{ratio:e}");
}

/// `test_multiple_injections_accumulate` — two sources in one step are two bookings, not one.
/// Opposite signs at different nodes, so each sees its own `pbar` and the two do **not** cancel:
/// the denominator is a real quantity (8.5e-6 J), not a near-zero.
#[test]
fn two_injections_in_one_step_are_two_bookings() {
    let mut b = room(rigid());
    let (n, h) = (b.p.n, b.p.h);
    let a = node_index([0.2, 0.2, 0.2], h, n).unwrap();
    let c = node_index([0.7, 0.5, 0.4], h, n).unwrap();
    assert_eq!((a, c), ([2, 2, 2], [7, 5, 4]));
    b.set_state(&vec![0.0; b.p.n_nodes()], None);
    for _ in 0..20 {
        b.inject(1e-3, Some(a));
        b.inject(-1e-3, Some(c));
        b.step();
    }
    assert!(
        b.injected > 1e-9,
        "the denominator collapsed; the test would be vacuous"
    );
    let ratio = b.energy().abs() / b.injected.abs();
    assert!(ratio < DRIFT_TOL, "{ratio:e}");
}

/// `test_open_token_pins_the_face_to_exactly_zero` — the `"open"` token on `y1` and an impedance of
/// exactly `0.0` on `z0`. The second goes through [`Wall::from_z`], the reduction itself, so it is
/// the half the native open-face bar (which spells `Wall::Open`) did not reach.
#[test]
fn an_open_token_and_a_zero_impedance_both_pin_their_face_to_exactly_zero() {
    assert_eq!(Wall::from_z(0.0), Wall::Open);
    let mut walls = rigid();
    walls[face("y1")] = Wall::Open;
    walls[face("z0")] = Wall::from_z(0.0);
    let mut b = seeded(walls);
    let s = b.p.p_shape();
    for _ in 0..50 {
        b.step();
        for i in 0..s[0] {
            for k in 0..s[2] {
                assert_eq!(b.pressure[airbox::flat(s, i, s[1] - 1, k)], 0.0);
            }
            for j in 0..s[1] {
                assert_eq!(b.pressure[airbox::flat(s, i, j, 0)], 0.0);
            }
        }
    }
}

// -- construction ---------------------------------------------------------------------------------

/// `test_cfl_rejected_above_the_3d_ceiling` and `test_cfl_accepted_exactly_at_the_ceiling`. The
/// ceiling is written here as `1/sqrt(3)`, not read from the model (§29.3).
#[test]
fn the_cfl_ceiling_is_one_over_root_three_and_a_hair_past_it_is_refused() {
    let at = params(ROOM, H, 1.0, rigid(), None);
    assert!((at.lam - 1.0 / 3.0f64.sqrt()).abs() < 1e-15, "{}", at.lam);
    let fs = C0_AIR * 3.0f64.sqrt() / (1.001 * H);
    let err = Params::new(ROOM, fs, H, rigid(), None, RHO0_AIR, C0_AIR).unwrap_err();
    assert!(matches!(err, ParamError::CflViolated(_)), "{err:?}");
    assert!(err.to_string().contains("CFL"), "{err}");
}

/// `test_bad_construction_refused` — the three refusals about a **value**. The other three cases
/// (a two-element room, the token `"squishy"`, the face name `"x2"`) are about the *shape* of a
/// Python argument and have no analogue here: `[f64; 3]`, [`Wall`] and `[Wall; 6]` are types
/// (§14). The viewer, which does parse wall tokens, refuses `"squishy"` in its own
/// `guards_and_budget_are_clean_error_payloads`.
#[test]
fn a_non_physical_room_is_refused_with_the_originals_message() {
    let p = |l: [f64; 3], h: f64, walls: [Wall; 6]| {
        Params::new(l, 12000.0, h, walls, None, RHO0_AIR, C0_AIR).unwrap_err()
    };
    let neg = p([1.0, -1.0, 1.0], 0.1, rigid());
    assert_eq!(neg, ParamError::NonPositiveScalar);
    assert_eq!(neg.to_string(), "L, fs, h, rho0, c0 must all be positive.");

    let coarse = p(ROOM, 5.0, rigid());
    assert!(
        matches!(coarse, ParamError::TooCoarse { n: [0, 0, 0], .. }),
        "{coarse:?}"
    );
    assert_eq!(
        coarse.to_string(),
        "h = 5.0 is coarser than the room (0.9, 0.7, 0.6): N = (0, 0, 0) has an axis with no \
         cells. Refine h (or enlarge L) so every axis takes at least one cell."
    );

    let z = p(ROOM, 0.1, walls_on(&["x0"], Wall::Impedance(-1.0)));
    assert_eq!(z, ParamError::BadImpedance("x0", -1.0));
    assert_eq!(
        z.to_string(),
        "wall 'x0': impedance Z must be >= 0, got -1.0."
    );
}

/// `test_a_point_outside_the_room_is_refused_not_relocated` — snapping *within* the room is the
/// resolution; snapping a point from outside it onto a wall would be a silently wrong answer.
/// Half a cell past the last node still counts as inside: that is rounding, not relocation.
///
/// The Python reached the refusal through `pressure_at`; natively the lookup is [`node_index`]
/// and the refusal with its message comes through a point the room is asked to use (`source`).
/// The messages are the Python's, recorded: every float in them is a product `n * h`.
#[test]
fn a_point_outside_the_room_is_refused_not_relocated() {
    let p = params(ROOM, H, CFL, rigid(), None);
    let [lx, ly, lz] = p.l_actual;
    assert_eq!(node_index([lx + 0.4 * H, ly, lz], H, p.n), Some(p.n));
    let expect = [
        (
            [lx + H, 0.3, 0.3],
            "point (1.0, 0.3, 0.3) lies outside the room (0, 0, 0)..(0.9, 0.7000000000000001, \
             0.6000000000000001) m. Nearest node index would be (10, 3, 3), valid range \
             0..(9, 7, 6) per axis.",
        ),
        (
            [-H, 0.3, 0.3],
            "point (-0.1, 0.3, 0.3) lies outside the room (0, 0, 0)..(0.9, 0.7000000000000001, \
             0.6000000000000001) m. Nearest node index would be (-1, 3, 3), valid range \
             0..(9, 7, 6) per axis.",
        ),
        (
            [0.3, 0.3, lz + H],
            "point (0.3, 0.3, 0.7000000000000001) lies outside the room (0, 0, 0)..(0.9, \
             0.7000000000000001, 0.6000000000000001) m. Nearest node index would be (3, 3, 7), \
             valid range 0..(9, 7, 6) per axis.",
        ),
    ];
    for (point, message) in expect {
        assert_eq!(node_index(point, H, p.n), None, "{point:?}");
        let fs = p.fs;
        let err = Params::new(ROOM, fs, H, rigid(), Some(point), RHO0_AIR, C0_AIR).unwrap_err();
        assert!(matches!(err, ParamError::OutsideRoom { .. }), "{err:?}");
        assert_eq!(err.to_string(), message);
    }
}

/// `test_grid_snap_is_reported_not_hidden` — a room that does not divide by `h` is snapped, and
/// says so: the grid spans `l_actual` while `l` keeps what was asked for.
#[test]
fn the_grid_snap_is_reported_not_hidden() {
    let p = Params::new(
        [0.93, 0.71, 0.58],
        12000.0,
        0.1,
        rigid(),
        None,
        RHO0_AIR,
        C0_AIR,
    )
    .unwrap();
    assert_eq!(p.n, [9, 7, 6]);
    for (got, want) in p.l_actual.iter().zip([0.9, 0.7, 0.6]) {
        assert!((got - want).abs() < 1e-12, "{:?}", p.l_actual);
    }
    assert_eq!(p.l, [0.93, 0.71, 0.58]);
    let snapped = node_index([0.44, 0.0, 0.0], p.h, p.n).unwrap();
    assert_eq!(snapped, [4, 0, 0]);
    assert!((snapped[0] as f64 * p.h - 0.4).abs() < 1e-12);
}

// -- the modal tier, at every index ---------------------------------------------------------------

/// Seed the room in mode `idx` and return the frequency it is predicted to run at.
fn set_mode(b: &mut AirBox, idx: [usize; 3]) -> f64 {
    let shape = mode_shape(&b.p, idx);
    let u0 = mode_velocity(&b.p, &shape);
    b.set_state(&shape, Some([&u0[0], &u0[1], &u0[2]]));
    mode_frequency(&b.p, idx)
}

/// Max deviation of the field from `amp * mode_shape`, relative to the mode's own scale.
fn mode_error(b: &AirBox, idx: [usize; 3], amp: f64) -> f64 {
    let mode = mode_shape(&b.p, idx);
    let scale = nan_max(mode.iter().map(|v| v.abs()));
    nan_max(
        b.pressure
            .iter()
            .zip(&mode)
            .map(|(p, v)| (p - amp * v).abs()),
    ) / scale
}

/// `cos(omega_d t k)`, with the `2 pi` written here.
fn amp(f: f64, t: usize, k: f64) -> f64 {
    (2.0 * std::f64::consts::PI * f * t as f64 * k).cos()
}

/// `test_every_axial_mode_is_an_exact_eigenvector` — one step from the exact initialisation is
/// one application of the scheme's Laplacian, so `p^1 = cos(omega_d k) p^0` holds iff `mu^2` is the
/// operator's true eigenvalue. Every index on each axis, DC and Nyquist included.
#[test]
fn every_axial_mode_is_an_exact_eigenvector() {
    for axis in 0..3 {
        let mut b = room(rigid());
        let mut worst = 0.0f64;
        for q in 0..=b.p.n[axis] {
            let mut idx = [0usize; 3];
            idx[axis] = q;
            let f = set_mode(&mut b, idx);
            assert_eq!(
                mode_error(&b, idx, 1.0),
                0.0,
                "set_mode did not start on the mode"
            );
            b.step();
            worst = nan_max([worst, mode_error(&b, idx, amp(f, 1, b.p.k))]);
        }
        assert!(
            worst < EXACT_TOL,
            "axis {axis}: worst eigenvector residual {worst:e}"
        );
    }
}

/// `test_every_oblique_mode_is_an_exact_eigenvector` — the full 3-D index grid, corners included:
/// `(0,0,0)` is the DC nullspace direction and `(Nx,Ny,Nz)` the corner Nyquist mode, whose arcsin
/// argument is `lambda sqrt(3)`, the value the CFL exists to cap.
#[test]
fn every_oblique_mode_is_an_exact_eigenvector() {
    let mut b = room(rigid());
    let [nx, ny, nz] = b.p.n;
    let mut worst = 0.0f64;
    for l in [0, 1, nx] {
        for m in [0, 2, ny] {
            for n in [0, 3, nz] {
                let f = set_mode(&mut b, [l, m, n]);
                b.step();
                worst = nan_max([worst, mode_error(&b, [l, m, n], amp(f, 1, b.p.k))]);
            }
        }
    }
    assert!(worst < EXACT_TOL, "worst {worst:e}");
}

/// `test_mode_tracks_its_predicted_amplitude` — the field must equal `cos(omega_d n k) * mode` at
/// every step, compared with the predicted amplitude over the **whole** field. A shape-only test
/// passes even when `omega_d` is wrong, and `omega_d` is the point.
#[test]
fn five_modes_track_their_predicted_amplitude_over_the_whole_field() {
    for idx in [[1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 1, 1], [3, 2, 1]] {
        let mut b = room(rigid());
        let f = set_mode(&mut b, idx);
        assert!(f > 0.0);
        let mut worst = 0.0f64;
        for t in 1..=500 {
            b.step();
            worst = nan_max([worst, mode_error(&b, idx, amp(f, t, b.p.k))]);
        }
        assert!(
            worst < EXACT_TOL,
            "mode {idx:?} drifted from cos(omega_d t) by {worst:e}"
        );
    }
}

// -- the cross-model oracles ----------------------------------------------------------------------

/// `test_wall_reflection_coefficient` — a plane pulse onto a locally-reacting impedance face
/// reflects with `|R| = |(zeta - 1)/(zeta + 1)|` at normal incidence. A **convergence** tier, not
/// machine precision. `zeta = 1` absorbs everything here only because the incidence is normal.
#[test]
fn a_wall_reflects_with_the_closed_form_coefficient() {
    let (h, nx) = (0.1, 400usize);
    for zeta in [0.5, 1.0, 2.0, 5.0] {
        let z = impedance_from_zeta(zeta, RHO0_AIR, C0_AIR);
        let mut duct = AirBox::new(params(
            [nx as f64 * h, h, h],
            h,
            CFL,
            walls_on(&["x1"], Wall::Impedance(z)),
            None,
        ));
        assert_eq!(duct.p.n, [nx, 1, 1]);
        let (x0, width) = (8.0, 0.6);
        let p1: Vec<f64> = (0..=nx)
            .map(|i| {
                let x = i as f64 * h;
                (-((x - x0) * (x - x0)) / (2.0 * (width * width))).exp()
            })
            .collect();
        // A purely right-going plane wave: u = p / (rho0 c0) at the faces.
        let f1: Vec<f64> = (0..nx)
            .map(|i| 0.5 * (p1[i] + p1[i + 1]) / (RHO0_AIR * C0_AIR))
            .collect();
        let (ps, us) = (duct.p.p_shape(), duct.p.u_shape(0));
        let mut p0 = vec![0.0; duct.p.n_nodes()];
        for i in 0..ps[0] {
            for j in 0..ps[1] {
                for k in 0..ps[2] {
                    p0[airbox::flat(ps, i, j, k)] = p1[i];
                }
            }
        }
        let mut ux = vec![0.0; us[0] * us[1] * us[2]];
        for i in 0..us[0] {
            for j in 0..us[1] {
                for k in 0..us[2] {
                    ux[airbox::flat(us, i, j, k)] = f1[i];
                }
            }
        }
        let uy = vec![0.0; duct.u[1].len()];
        let uz = vec![0.0; duct.u[2].len()];
        duct.set_state(&p0, Some([&ux, &uy, &uz]));

        let probe = airbox::flat(ps, (20.0 / h).round() as usize, 0, 0);
        let arrival = (nx as f64 * h - x0) / (C0_AIR * duct.p.k);
        let (mut incident, mut reflected) = (0.0f64, 0.0f64);
        for n in 0..(arrival * 3.2) as usize {
            duct.step();
            let v = duct.pressure[probe].abs();
            if (n as f64) < 0.9 * arrival {
                incident = nan_max([incident, v]);
            } else {
                reflected = nan_max([reflected, v]);
            }
        }
        let predicted = ((zeta - 1.0) / (zeta + 1.0)).abs();
        let err = (reflected / incident - predicted).abs();
        assert!(
            err < 0.02,
            "zeta={zeta}: measured |R|={:.4}, closed form {predicted:.4}",
            reflected / incident
        );
    }
}

/// `test_quasi_1d_box_tracks_the_repo_bore` — a box one cell thick in `y` and `z` *is* a 1-D duct,
/// and must track [`Bore`] step for step: 4,000 steps from the same `p^0`, at the same `lambda`
/// (the 3-D CFL caps the box at 0.577, so a bore at its own `lambda = 1` would differ in dispersion
/// for an uninteresting reason).
///
/// A **cross-model** check, not a reduction: `Bore` carries the area `S` through both updates and
/// the box carries none, so bit-identity is not promised — and is asserted absent, so nobody
/// tightens it later. It is also the start-up's independent witness: both models derive `u^{1/2}`
/// from `p^0` themselves, each in its own code.
#[test]
fn a_one_cell_thick_room_tracks_the_bore() {
    let (l, n, fs) = (0.5, 36usize, 48000.0);
    let h = l / n as f64;
    let mut walls = rigid();
    walls[face("x1")] = Wall::Open;
    let mut b = AirBox::new(Params::new([l, h, h], fs, h, walls, None, RHO0_AIR, C0_AIR).unwrap());
    assert_eq!(b.p.n, [n, 1, 1]);
    let tube = bore::Params::new(
        l,
        fs,
        n,
        0.008,
        Some((End::Closed, End::Open)),
        0.0,
        0.0,
        RHO0_AIR,
        C0_AIR,
    )
    .unwrap();
    let mut t = Bore::new(tube);
    assert!(
        (b.p.lam - C0_AIR / fs / h).abs() < 1e-15,
        "the same lambda, by construction"
    );

    let mut p1: Vec<f64> = (0..=n)
        .map(|i| {
            let x = i as f64 * h;
            (-((x - 0.15) * (x - 0.15)) / (2.0 * (0.02 * 0.02))).exp()
        })
        .collect();
    p1[n] = 0.0; // the bore pins its open end; match it so step 0 already agrees
    let ps = b.p.p_shape();
    let mut p0 = vec![0.0; b.p.n_nodes()];
    for i in 0..ps[0] {
        for j in 0..ps[1] {
            for k in 0..ps[2] {
                p0[airbox::flat(ps, i, j, k)] = p1[i];
            }
        }
    }
    b.set_state(&p0, None);
    t.set_state(&p1, &vec![0.0; n]);

    let probe = n / 3;
    let (mut worst, mut peak, mut differs) = (0.0f64, 0.0f64, false);
    for _ in 0..4000 {
        b.step();
        t.step(None);
        let (x, y) = (
            b.pressure[airbox::flat(ps, probe, 0, 0)],
            t.displacement_at(probe),
        );
        worst = nan_max([worst, (x - y).abs()]);
        peak = nan_max([peak, y.abs()]);
        differs |= x.to_bits() != y.to_bits();
    }
    let err = worst / peak;
    assert!(err < 1e-12, "box vs bore {err:e}");
    assert!(differs, "bit-identity is not promised here");
    // The h/2 transverse half-cells must raise no dynamics of their own, even at the open face.
    for i in 0..ps[0] {
        let first = b.pressure[airbox::flat(ps, i, 0, 0)];
        for j in 0..ps[1] {
            for k in 0..ps[2] {
                assert_eq!(
                    b.pressure[airbox::flat(ps, i, j, k)],
                    first,
                    "node {i},{j},{k}"
                );
            }
        }
    }
}

// -- non-standard air -----------------------------------------------------------------------------
//    Added, the human's call (§44.4): the room takes `rho0` and `c0`, and every room the workspace
//    builds — every test here, every core file, the viewer — uses the standard air. Each of the 14
//    places the room or its port reads either constant could be hard-wired to the standard value
//    and nothing would notice. These run the room at an air no fixture uses, so a constant read
//    from the wrong place changes the answer. The values are written here, not taken from the model.

const ODD_RHO0: f64 = 0.9;
const ODD_C0: f64 = 380.0;

fn odd_air(walls: [Wall; 6]) -> AirBox {
    let fs = ODD_C0 * 3.0f64.sqrt() / (CFL * H);
    AirBox::new(Params::new(ROOM, fs, H, walls, None, ODD_RHO0, ODD_C0).unwrap())
}

/// The modal tier at a different air: the whole-field mode tracking (the compliance gain, the
/// momentum step, the discrete frequency and the start-up velocity all read `rho0` or `c0`), and
/// the textbook room frequency against `(c0/2) sqrt(sum (l_d/L_d)^2)` written out here.
#[test]
fn the_modes_are_exact_at_a_non_standard_air() {
    for idx in [[1, 0, 0], [2, 1, 1], [3, 2, 1]] {
        let mut b = odd_air(rigid());
        let f = set_mode(&mut b, idx);
        let mut worst = 0.0f64;
        for t in 1..=200 {
            b.step();
            worst = nan_max([worst, mode_error(&b, idx, amp(f, t, b.p.k))]);
        }
        assert!(worst < EXACT_TOL, "mode {idx:?} at odd air: {worst:e}");
        let s: f64 = (0..3)
            .map(|d| (idx[d] as f64 / b.p.l_actual[d]).powi(2))
            .sum();
        let textbook = 0.5 * ODD_C0 * s.sqrt();
        let got = physsynth_core::airbox::continuum_mode_frequency(&b.p, idx);
        assert!(
            (got - textbook).abs() <= 1e-12 * textbook,
            "{got} vs {textbook}"
        );
    }
}

/// The three channels at a different air: two lossy walls (each wall's closure reads `rho0 c0^2`
/// through the gain) and a source, from a broadband field (the stored energy reads `rho0` twice
/// and `c0` once). Flat over every step.
#[test]
fn the_energy_books_close_at_a_non_standard_air() {
    let z = |zeta: f64| Wall::Impedance(impedance_from_zeta(zeta, ODD_RHO0, ODD_C0));
    let mut walls = rigid();
    walls[face("x0")] = z(0.5);
    walls[face("y1")] = z(2.0);
    let mut b = odd_air(walls);
    let p0 = noise(b.p.n_nodes());
    b.set_state(&p0, None);
    let q = gaussian_q(900.0);
    let e0 = b.energy();
    let mut worst = 0.0f64;
    for n in 0..400 {
        b.inject(q(n as f64 * b.p.k), None);
        b.step();
        worst = nan_max([worst, (b.energy() - e0).abs() / e0.abs()]);
    }
    assert!(
        b.dissipated > 0.0 && b.injected != 0.0,
        "a channel was not live"
    );
    assert!(worst < DRIFT_TOL, "ledger at odd air: {worst:e}");
}

/// The room's own full-array open-circuit update, divergence then wall closure, read the way a
/// point port reads it.
fn full_array_read(room: &AirBox, port: &physsynth_core::airbox_port::RoomPort) -> f64 {
    let div = airbox::divergence(&room.p, &room.u[0], &room.u[1], &room.u[2]);
    let mut p_full = airbox::pressure_step(&room.p, &room.pressure, &div);
    if room.p.has_walls {
        for (i, v) in p_full.iter_mut().enumerate() {
            *v = (*v - room.p.beta[i] * room.pressure[i]) / (1.0 + room.p.beta[i]);
        }
    }
    let terms: Vec<f64> = port
        .flat()
        .iter()
        .enumerate()
        .map(|(m, &i)| port.w()[m] * (0.5 * (p_full[i] + room.pressure[i])))
        .collect();
    physsynth_core::reduce::sum(&terms)
}

/// The port at a different air: its local free-pressure read must be the room's full-array update
/// bit for bit (the read forms its own `k rho0 c0^2`), and `R_room` must be what the room actually
/// does, measured differentially (it forms `rho0 c0^2` separately again). A corner port on lossy
/// walls, so the wall factor is live, and an interior one.
#[test]
fn the_port_reads_and_loads_the_room_at_a_non_standard_air() {
    use physsynth_core::airbox_port::RoomPort;
    let lossy = [Wall::Impedance(impedance_from_zeta(1.0, ODD_RHO0, ODD_C0)); 6];
    for at in [[0.0, 0.0, 0.0], [0.3, 0.3, 0.3]] {
        let mut room = odd_air(lossy);
        let mut port = RoomPort::new(&mut room, at, None).unwrap();
        let p0 = noise(room.p.n_nodes());
        room.set_state(&p0, None);
        for _ in 0..13 {
            room.step();
        }
        assert_eq!(
            port.free_pressure(&room).to_bits(),
            full_array_read(&room, &port).to_bits(),
            "at {at:?}"
        );

        let snap = room.clone();
        let mut pbar_after = |q: f64| {
            let mut r = snap.clone();
            port.reset();
            let p_old = r.pressure.clone();
            port.inject(&mut r, q).unwrap();
            r.step();
            let terms: Vec<f64> = port
                .flat()
                .iter()
                .enumerate()
                .map(|(m, &i)| port.w()[m] * (0.5 * (r.pressure[i] + p_old[i])))
                .collect();
            physsynth_core::reduce::sum(&terms)
        };
        let u = 3.7e-4;
        let measured = (pbar_after(u) - pbar_after(0.0)) / u;
        let want = port.r_room();
        assert!(
            (measured - want).abs() <= 1e-12 * want,
            "at {at:?}: measured {measured:.17e} vs R_room {want:.17e}"
        );
    }
}
