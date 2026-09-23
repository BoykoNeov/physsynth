//! A mallet on a gong **in a room** — `MalletVkRoom`, the nested solve meeting the loaded operator.
//!
//! `docs/dev/mallet-vk-room-plan.md` is the design; `docs/dev/python-retirement-plan.md` §18 is the
//! move out of the binding, and these bars are what `tests/test_mallet_room_gong.py` carried until
//! then. Two models that had never met: the mallet's outer chord re-solves the plate several times
//! from one time-`n` state, and the room wrapper's step is a once-per-step transaction that reads
//! its port, injects once and books the radiated energy once.
//!
//! **What this file is mostly about is that the room is stepped exactly once.** A wrapper driven
//! per trial would inject `n_outer` times, and the scene total that would normally catch it is the
//! very number a double injection corrupts on *both* sides at once. So the bars are structural
//! rather than statistical: the injections are counted, and the volume velocity the room receives
//! is recomputed from the plate's own committed buffers.
//!
//! The other half is the frozen column. The chord's tangent `g_s` comes off the drive-point
//! response of the operator the step actually inverts, and in a room that is `A_loaded`, not `A`.
//! It changes **no physics** — the two `g_s` terms cancel at the fixed point, so the committed
//! force and field are the same either way — and it is worth about five times the outer
//! iterations, which is why the exact `n_outer == 1` bar is the one that discriminates.
//!
//! # What the Python file had that has no bar here, and why
//!
//! * `mal.plate is inst`, the bare-gong cast and the linear-plate refusal were claims about which
//!   **Python object** a constructor accepted and handed back. Here the gong in a room is its own
//!   type holding a `RoomGrid<VkSeam>`; a linear plate's wrapper is a `RoomGrid<PlateSeam>` and
//!   does not compile. The shape of an argument became a type (plan §14.2). The linear-plate
//!   message itself is still asserted, on the bare class, by `tests/test_mallet_gong.py`.
//! * "The port is read once a step" worked by replacing the port's `free_pressure` and
//!   `require_ready` with counting wrappers. A native port is a value with no methods to replace
//!   (§16.6). What that test protected is structural here: [`MalletVkRoom::step`] calls
//!   `prepare` once and the chord receives the half it built by reference.

use physsynth_core::airbox::{AirBox, Params as RoomParams, Wall, C0_AIR, RHO0_AIR};
use physsynth_core::airbox_port::Spreading;
use physsynth_core::airbox_wrap::{GridPort, GridSeam, RoomGrid, VkSeam, WrapError};
use physsynth_core::mallet::{
    vk_drive_point_tangent, MalletVkPlate, MalletVkRoom, RoomGongError, VkPlateParams,
};
use physsynth_core::plate::{vk_step_with, Boundary, VkCoupledStep, VkParams, VkPlate, VkSpec};
use physsynth_core::sparse::Csr;

// -- the suite's fixtures, `tests/helpers.py` ---------------------------------------------------

const FS: f64 = 8000.0;
const CFL: f64 = 0.9;
const N_ROOM: [usize; 3] = [12, 11, 9];
const PLATE_L: f64 = 0.30;
const DIPOLE_INDEX: i64 = 4;
const THICKNESS: f64 = 1.0e-3;

// The mallet the room's gong is struck with: the shipped gong's mass and felt, and a velocity that
// puts this plate (a third the area, fifteen times the areal density of the 48 kHz rig) at
// `w/e ~ 1.9` -- the nonlinearity awake and comfortably inside the Picard wall.
const MASS: f64 = 0.05;
const STIFFNESS: f64 = 5.0e4;
const ALPHA: f64 = 2.3;
const V0: f64 = 6.0;
const OUTER_TOL: f64 = 1e-13;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mount {
    Baffled,
    Suspended,
}

const MOUNTS: [Mount; 2] = [Mount::Baffled, Mount::Suspended];

fn make_room() -> AirBox {
    let h = C0_AIR * 3.0f64.sqrt() / (CFL * FS);
    let l = [
        N_ROOM[0] as f64 * h,
        N_ROOM[1] as f64 * h,
        N_ROOM[2] as f64 * h,
    ];
    AirBox::new(
        RoomParams::new(l, FS, h, [Wall::Rigid; 6], None, RHO0_AIR, C0_AIR).expect("a legal room"),
    )
}

/// `make_air_vk_plate`: a 1 mm steel sheet, 0.3 m square, lossless, `N = 8`.
fn vk_params(nonlinear: bool) -> VkParams {
    VkParams::new(&VkSpec {
        lx: PLATE_L,
        ly: PLATE_L,
        young: 2.0e11,
        thickness: THICKNESS,
        nu: 0.3,
        rho: 7800.0,
        fs: FS,
        n: 8,
        boundary: Some(Boundary::Supported),
        nonlinear,
        couple_tol: 1e-13,
        ..VkSpec::default()
    })
    .expect("a legal von Karman plate")
}

fn mount_in(mount: Mount, room: &mut AirBox, nonlinear: bool) -> RoomGrid<VkSeam> {
    let seam = VkSeam::new(VkPlate::new(vk_params(nonlinear)));
    match mount {
        Mount::Baffled => RoomGrid::baffled(seam, room, "z0", None, Spreading::Bilinear),
        Mount::Suspended => {
            RoomGrid::suspended(seam, room, "z", DIPOLE_INDEX, None, Spreading::Bilinear)
        }
    }
    .expect("a legal mounting")
}

/// The mallet's parameters against `vk`, struck at `(0.3, 0.4)` of the span -- the **bare**
/// column, as `VkPlateParams::new` always builds it.
fn mallet_params(vk: &VkParams, v0: f64, gap: f64) -> VkPlateParams {
    VkPlateParams::new(
        vk,
        MASS,
        STIFFNESS,
        ALPHA,
        0.0,
        0.3 * vk.lin.lx,
        0.4 * vk.lin.ly,
        v0,
        gap,
        1e-12,
        1e-14,
        60,
        OUTER_TOL,
        20,
    )
    .expect("a valid mallet")
}

/// `make_mallet_room_gong`: the struck gong in a fresh rigid room.
fn struck(mount: Mount, nonlinear: bool, v0: f64, gap: f64) -> (AirBox, MalletVkRoom) {
    let mut room = make_room();
    let grid = mount_in(mount, &mut room, nonlinear);
    let params = mallet_params(&grid.seam.plate.p, v0, gap);
    let mal = MalletVkRoom::new(params, grid, gap, v0).expect("the loaded LU back-substitutes");
    (room, mal)
}

/// `(k^2 / force_denominator) A^-1 e_node` off a factorization -- what the chord must have frozen.
fn column_of(grid: &RoomGrid<VkSeam>, node: usize) -> Vec<f64> {
    let vk = &grid.seam.plate.p;
    let mut e = vec![0.0; vk.lin.n_live];
    e[node] = 1.0;
    let scale = vk.lin.k * vk.lin.k / vk.force_denominator;
    let col = grid.lu_loaded().solve(&e).expect("SPD");
    col.iter().map(|&c| scale * c).collect()
}

fn set_load(grid: &mut RoomGrid<VkSeam>, m: Csr) {
    match &mut grid.port {
        GridPort::Baffled(p) => p.set_load_matrix(m),
        GridPort::Suspended(p) => p.set_load_matrix(m),
    }
}

/// Refactor the loaded system **without** its load -- the plate's own operator -- and leave the
/// port's load as built, so only the factorization changes. The largest change a caller can make
/// to the operator, and the native form of the Python test's `inst._lu_loaded = splu(a_bare)`.
fn refactor_without_the_load(grid: &mut RoomGrid<VkSeam>) {
    let original = grid.port.load_matrix().clone();
    let n = original.nrows();
    set_load(grid, Csr::from_rows(n, n, vec![Vec::new(); n]));
    grid.refactor().expect("the plate's own matrix factors");
    set_load(grid, original);
}

fn run(mal: &mut MalletVkRoom, room: &mut AirBox, steps: usize) {
    for _ in 0..steps {
        mal.step(room).expect("a legal step");
        room.step();
    }
}

// -- the frozen column --------------------------------------------------------------------------

/// `g_s` must come off `A_loaded`, the operator the coupled step actually inverts.
///
/// Asserted to the **bit** against the column rebuilt here, and asserted to differ from the bare
/// plate's. The difference is small -- 2.9e-04 relative on this fixture -- and that is exactly why
/// it needs an exact bar: far too small to show in any trajectory, and worth about five times the
/// outer iterations.
#[test]
fn the_chord_freezes_the_loaded_column_not_the_plates_own() {
    for mount in MOUNTS {
        let (_room, mal) = struck(mount, true, V0, 0.0);
        let p = mal.params();
        let want = column_of(&mal.grid, p.node);
        let bare = mallet_params(&mal.grid.seam.plate.p, V0, 0.0);

        assert_eq!(p.influence, want, "{mount:?}");
        assert_eq!(p.g_s, want[p.node]);
        assert_eq!(p.g, p.g_s + p.g_h);
        // The two columns are genuinely different, so the bit-equality above is a real choice.
        assert_ne!(p.g_s, bare.g_s);
        assert!((bare.g_s - p.g_s).abs() / p.g_s.abs() > 1e-5, "{mount:?}");
    }
}

/// The exact bar that discriminates the two columns -- and the reason the loaded one ships.
///
/// With the coupling off the plate is affine in the contact force, `w_node(f) = w_free,node -
/// g_s_true f` with `g_s_true` the inverted operator's. The chord forms `u_eff = w_node(f) + g_s f`,
/// and **only** when `g_s` is `g_s_true` do the two terms cancel and the second contact solve
/// receive the first one's arguments: it exits at one iteration. Frozen on the bare column the
/// residual would contract by `|g_s_bare - g_s_loaded| / g` a pass, and that arithmetic is checked
/// below from the two columns on this fixture rather than quoted from the plan.
#[test]
fn nonlinear_false_in_a_room_exits_the_chord_at_one_iteration() {
    for mount in MOUNTS {
        let (mut room, mut mal) = struck(mount, false, V0, 0.0);
        let mut landed = 0;
        for _ in 0..400 {
            mal.step(&mut room).expect("a legal step");
            room.step();
            if mal.state().in_contact {
                landed += 1;
                let last = mal.last.as_ref().expect("stepped");
                assert_eq!(last.n_outer, 1, "{mount:?}: the affine chord exits at one");
                assert!(last.outer_converged);
                assert!(last.outer_residual <= OUTER_TOL);
            }
        }
        assert!(
            landed > 50,
            "{mount:?}: the mallet has to land for this to mean anything"
        );

        let p = mal.params();
        let bare = mallet_params(&mal.grid.seam.plate.p, V0, 0.0);
        let rate = (bare.g_s - p.g_s).abs() / p.g;
        assert!(1e-5 < rate && rate < 1e-3, "{mount:?}: {rate}");
        let passes = (OUTER_TOL.ln() / rate.ln()).ceil() as i64 + 1;
        assert!((4..=6).contains(&passes), "{mount:?}: {passes}");
    }
}

/// A rebuilt factorization must re-derive the frozen column, and nothing would turn red if it did
/// not: the chord converges to the same root whichever column it froze, so the only symptom is
/// more outer iterations.
///
/// The binding compared the factorization object's pointer each step, because a Python caller
/// could assign a new one. Natively [`RoomGrid::refactor`] is the only way in and bumps a
/// generation stamp, so the bar is the same claim against the stamp. The column is checked against
/// the factorization actually in place -- not a second factorization of the same matrix, which
/// would make the exactness depend on two orderings agreeing.
#[test]
fn a_rebuilt_factorization_refreshes_the_frozen_column() {
    let (mut room, mut mal) = struck(Mount::Baffled, true, V0, 0.0);
    let before = mal.params().g_s;
    assert_eq!(mal.grid.generation(), 0);
    run(&mut mal, &mut room, 3);
    assert_eq!(mal.params().g_s, before, "no rebuild, no refresh");

    refactor_without_the_load(&mut mal.grid);
    assert_eq!(mal.grid.generation(), 1);
    // Stale until the next step looks -- the refresh is the step's first phase.
    assert_eq!(mal.params().g_s, before);
    mal.step(&mut room).expect("a legal step");

    let p = mal.params();
    assert_eq!(p.influence, column_of(&mal.grid, p.node));
    assert_ne!(p.g_s, before);
    assert_eq!(p.g, p.g_s + p.g_h);
    // And it is the plate's OWN column now, bit for bit, because the operator is the plate's own.
    let own = mallet_params(&mal.grid.seam.plate.p, V0, 0.0);
    assert_eq!(p.influence, own.influence);
}

/// The exact tangent is an instrument, and in a room it was once wrong in a specific way: the
/// column already loaded, the operator left on the plate's own factorization -- a loaded
/// right-hand side inverted against the bare Jacobian, the derivative of neither problem.
///
/// The Python bar compared the room's tangent with a bare gong's and asserted they **differ**. They
/// do, but the columns alone make them differ, so that bar passed with the operator unrouted: a
/// deliberate breakage of exactly the defect it was written for turned nothing red (plan §18.4).
/// This one is a **positive** oracle instead: `response = -d w_node / d f` at the root, so a
/// central difference of the room's own trial solve -- the loaded right-hand side against the
/// loaded factorization, from the pre-step state -- has to agree with it.
///
/// Measured on this fixture at the step of largest force, both mounts: routed, the two agree to
/// 2.6e-12 - 9.5e-11 at `delta = 1e-4 f` (the difference's own truncation and the inner solve's
/// tolerance); unrouted, they disagree by 9.1e-6. The bar sits at 1e-8, about a hundred times
/// clear of each.
#[test]
fn the_exact_tangent_is_taken_against_the_loaded_operator() {
    for mount in MOUNTS {
        let (mut room, mut mal) = struck(mount, true, V0, 0.0);
        let mut best: Option<(f64, MalletVkRoom, AirBox, Vec<f64>)> = None;
        for _ in 0..120 {
            let (pre, pre_room) = (mal.clone(), room.clone());
            mal.step(&mut room).expect("a legal step");
            let f = mal.state().contact_force;
            if best.as_ref().is_none_or(|b| f > b.0) {
                best = Some((f, pre, pre_room, mal.grid.seam.plate.u.clone()));
            }
            room.step();
        }
        let (force, pre, pre_room, w) = best.expect("stepped");
        assert!(force > 1.0, "{mount:?}: the mallet has to be pressing");
        let (grid, p) = (&pre.grid, pre.params());
        let pl = &grid.seam.plate;
        let (g_exact, response, products) = pre
            .drive_point_tangent(&pl.u, &pl.u_prev, &pl.f_prev, force, &w)
            .expect("both factorizations back-substitute");
        assert!(products > 0, "a GMRES that spent nothing solved nothing");
        assert_eq!(g_exact, response + p.g_h);

        // The room's trial solve at a given force: exactly what the chord calls.
        let half = grid
            .prepare(&pre_room)
            .expect("the port is ready before the step");
        let w_node = |f: f64| {
            let mut f_ext = vec![0.0; pl.u.len()];
            f_ext[p.node] = -f;
            let rhs = grid.loaded_rhs(&half, Some(&f_ext));
            let lu = grid.lu_loaded();
            let ctx = VkCoupledStep::with_rhs(rhs, &pl.u_prev, &pl.f_prev, &pl.p, lu);
            vk_step_with(&ctx, &pl.u, &pl.u_prev, &pl.f)
                .expect("a trial solve")
                .u[p.node]
        };
        let delta = 1e-4 * force;
        let fd = -(w_node(force + delta) - w_node(force - delta)) / (2.0 * delta);
        let err = (response - fd).abs() / fd.abs();
        assert!(
            err < 1e-8,
            "{mount:?}: tangent {response:e} vs difference {fd:e}, {err:e}"
        );

        // And against the identical mallet with no room: the bare gong's tangent is a different
        // number, by far more than the bar above allows.
        let bare_vk = vk_params(true);
        let twin = mallet_params(&bare_vk, V0, 0.0);
        let mut f_ext = vec![0.0; pl.u.len()];
        f_ext[twin.node] = -force;
        let (_, resp_bare, _) = vk_drive_point_tangent(
            &pl.u,
            &pl.u_prev,
            &pl.f_prev,
            Some(&f_ext),
            &w,
            &twin,
            &bare_vk,
        )
        .expect("both factorizations back-substitute");
        assert!(
            (response - resp_bare).abs() / resp_bare.abs() > 1e-6,
            "{mount:?}"
        );
        // It is about the quantity the chord froze: `|1 - g_exact / g|` bounds the contraction.
        assert!(0.0 < g_exact && g_exact < 10.0 * p.g);
    }
}

// -- the room is stepped exactly once -----------------------------------------------------------

/// A mallet that never lands leaves the room scene bit-identical to the bare wrapper's.
///
/// Structural, not a floating-point coincidence: a zero contact force short-circuits to the
/// force-free advance and returns it **unmodified**, and that trial was assembled by `loaded_rhs`
/// with `f_ext = None` -- the expression `RoomGrid::step` uses. Driving the plate with a zero force
/// *vector* would add `+0.0` into every entry, the identity for every double except `-0.0`. The
/// room is compared too: a mallet that never lands must leave the air where the wrapper leaves it.
#[test]
fn a_mallet_that_never_lands_leaves_the_room_scene_bit_identical() {
    for mount in MOUNTS {
        let (mut room, mut mal) = struck(mount, true, -1.0, 0.01);
        let mut ref_room = make_room();
        let mut reference = mount_in(mount, &mut ref_room, true);

        let n = mal.grid.seam.n_live();
        let ic: Vec<f64> = (0..n)
            .map(|i| 2.0 * THICKNESS * (3.0 * i as f64 / (n - 1) as f64).sin())
            .collect();
        mal.grid.set_state(&ic, &vec![0.0; n]);
        reference.set_state(&ic, &vec![0.0; n]);

        for _ in 0..120 {
            mal.step(&mut room).expect("a legal step");
            room.step();
            reference.step(&mut ref_room, None).expect("a legal step");
            ref_room.step();
            assert_eq!(mal.state().contact_force, 0.0);
            assert_eq!(mal.last.as_ref().expect("stepped").n_outer, 0);
            assert_eq!(mal.grid.seam.plate.u, reference.seam.plate.u, "{mount:?}");
            assert_eq!(mal.grid.seam.plate.f, reference.seam.plate.f);
            assert_eq!(room.pressure, ref_room.pressure, "{mount:?}");
            assert_eq!(mal.grid.radiated_energy(), reference.radiated_energy());
        }
    }
}

/// The failure this whole composition was designed around, asserted directly: one injection per
/// step, whatever `n_outer` was.
///
/// Natively a second injection could not even be miscounted -- the port's `inject` checks
/// `require_ready` and refuses -- so the count below is a cheap statement of the design, and the
/// second half of the bar is the refusal itself: stepping the mallet again before the room has
/// consumed the first injection is refused, and **mutates nothing**.
///
/// "One injection" is counted in the room's queue, and what one injection queues is the **port's**
/// business: one entry flush in a wall, a `-q` / `+q` pair hung in the interior, one per face. So
/// the expected count is read off the bare wrapper's own single step rather than written down.
#[test]
fn the_port_is_injected_once_per_step_however_long_the_chord_runs() {
    for mount in MOUNTS {
        let per_injection = {
            let mut ref_room = make_room();
            let mut reference = mount_in(mount, &mut ref_room, true);
            reference.step(&mut ref_room, None).expect("a legal step");
            ref_room.pending_ports.len()
        };
        assert!(per_injection > 0);

        let (mut room, mut mal) = struck(mount, true, V0, 0.0);
        let mut multi = 0;
        for _ in 0..200 {
            assert!(room.pending_ports.is_empty());
            mal.step(&mut room).expect("a legal step");
            assert_eq!(
                room.pending_ports.len(),
                per_injection,
                "{mount:?}: one injection per step"
            );
            if mal.last.as_ref().expect("stepped").n_outer > 1 {
                multi += 1;
            }
            room.step();
        }
        assert!(
            multi > 20,
            "{mount:?}: the chord has to iterate for this to be about anything"
        );

        mal.step(&mut room).expect("a legal step");
        let (u, z, n) = (mal.grid.seam.plate.u.clone(), mal.state().z_h, mal.grid.n());
        let refused = mal.step(&mut room);
        assert!(
            matches!(refused, Err(RoomGongError::Room(WrapError::Port(_)))),
            "{mount:?}: {refused:?}"
        );
        assert_eq!(mal.grid.seam.plate.u, u);
        assert_eq!(mal.state().z_h, z);
        assert_eq!(mal.grid.n(), n);
        assert_eq!(room.pending_ports.len(), per_injection);
    }
}

/// The one error the inject-once bar cannot see: the room fed a **trial** iterate.
///
/// `q = T (w^{n+1} - w^{n-1}) / 2k`. Hand the room half a trial and the injection count is still
/// one, the radiated ledger still advances plausibly and every energy bar still passes -- the room
/// simply receives a volume velocity the gong never had. So it is recomputed here from the plate's
/// own buffers, taken around the step.
#[test]
fn the_room_is_driven_by_the_committed_field_not_by_a_trial() {
    for mount in MOUNTS {
        let (mut room, mut mal) = struck(mount, true, V0, 0.0);
        let k = mal.grid.k();
        let mut checked = 0;
        for _ in 0..200 {
            let w_n = mal.grid.seam.plate.u.clone();
            let w_nm1 = mal.grid.seam.plate.u_prev.clone();
            mal.step(&mut room).expect("a legal step");
            let pl = &mal.grid.seam.plate;
            assert_eq!(pl.u_prev, w_n, "the roll happened exactly once");
            let vel: Vec<f64> = (0..w_n.len())
                .map(|i| (pl.u[i] - w_nm1[i]) / (2.0 * k))
                .collect();
            let want = mal.grid.port.t().matvec(&vel);
            assert_eq!(mal.grid.nodal_volume_velocity(), &want[..], "{mount:?}");
            if mal.last.as_ref().expect("stepped").n_outer > 1 {
                checked += 1;
            }
            room.step();
        }
        assert!(
            checked > 20,
            "{mount:?}: the chord has to iterate for this to mean anything"
        );
    }
}

/// Every plate read-out a bare step writes, a struck step writes too -- bare gong and room alike.
///
/// Both mallets commit through `VkPlate::record`. Before this, the bare native mallet wrote three
/// of the six read-outs by hand and left `last_residual`, `residual_ratio` and `n_fallbacks` at
/// whatever came before, while the binding wrote all six -- the drift plan §17.1 found in the room
/// seam, a second time. Seeded with garbage so an untouched field cannot pass by coincidence.
#[test]
fn both_mallets_write_every_read_out_through_the_one_commit_path() {
    let (mut room, mut mal) = struck(Mount::Baffled, true, V0, 0.0);
    let vk = vk_params(true);
    let mut bare = MalletVkPlate::new(mallet_params(&vk, V0, 0.0), VkPlate::new(vk), 0.0, V0);
    for pl in [&mut mal.grid.seam.plate, &mut bare.plate] {
        pl.last_residual = -7.0;
        pl.residual_ratio = -7.0;
        pl.n_fallbacks = 99;
        pl.n_iters = 99;
        pl.n_solves = 99;
        pl.converged = false;
    }
    for _ in 0..40 {
        mal.step(&mut room).expect("a legal step");
        room.step();
        bare.step().expect("a legal step");
        for (pl, last) in [
            (&mal.grid.seam.plate, mal.last.as_ref().expect("stepped")),
            (&bare.plate, bare.last.as_ref().expect("stepped")),
        ] {
            assert_eq!(pl.last_residual, last.outer_residual);
            assert!(pl.residual_ratio.is_nan());
            assert_eq!(pl.n_fallbacks, last.n_fallbacks);
            assert_eq!(pl.n_iters, last.inner_iters);
            assert_eq!(pl.n_solves, last.n_solves);
            assert_eq!(pl.converged, last.inner_converged);
        }
    }
    assert_eq!(mal.grid.seam.plate.n, 40);
    assert_eq!(mal.grid.n(), 40);
    assert_eq!(bare.plate.n, 40);
}

// -- energy -------------------------------------------------------------------------------------

/// Lossless gong, rigid room: `mal.energy() + room.energy()` is flat across the contact.
///
/// `mal.energy()` takes its plate term from the **wrapper**, so the radiated channel is inside it;
/// the air itself is the second term. Necessary and not sufficient -- this family's standing rule
/// is that no single one of its detectors is enough -- and with model #6's caveat that it holds
/// only at the Picard fixed point, which is why `inner_converged` is asserted beside it.
#[test]
fn the_scene_total_is_conserved_through_the_strike() {
    for mount in MOUNTS {
        let (mut room, mut mal) = struck(mount, true, V0, 0.0);
        let e0 = mal.energy() + room.energy();
        let (mut lo, mut hi) = (e0, e0);
        let mut landed = 0;
        for _ in 0..600 {
            mal.step(&mut room).expect("a legal step");
            room.step();
            assert!(mal.last.as_ref().expect("stepped").inner_converged);
            landed += usize::from(mal.state().in_contact);
            let e = mal.energy() + room.energy();
            lo = lo.min(e);
            hi = hi.max(e);
        }
        assert!(landed > 50, "{mount:?}");
        assert!(
            (hi - lo) / e0.abs() < 1e-10,
            "{mount:?}: {}",
            (hi - lo) / e0.abs()
        );
    }
}

/// A guard on the trap, not the physics: the plate's own `energy()` is the gong's total
/// **without** the channel it radiates through, short of the wrapper's by exactly the ledger.
#[test]
fn the_plates_own_energy_is_the_wrong_number_and_the_wrapper_knows_it() {
    let (mut room, mut mal) = struck(Mount::Baffled, true, V0, 0.0);
    run(&mut mal, &mut room, 300);
    let bare = mal.grid.seam.plate.energy();
    let wrapped = mal.grid.energy();
    assert_ne!(wrapped, bare);
    assert_eq!(wrapped, bare + mal.grid.radiated_energy());
    assert!(mal.grid.radiated_energy().abs() > 0.0);
}
