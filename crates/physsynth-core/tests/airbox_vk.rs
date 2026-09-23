//! Native acceptance bars for the **gong in the room** — model #6 as a baffled and suspended surface
//! (HANDOFF §12.H, air-box batch 6), `RoomGrid<VkSeam>`.
//!
//! These carry over the wrapper half of `tests/test_airbox_vk.py`. Its other half — the three-way
//! chain `string -> bridge -> room-loaded gong -> room` — went native with the plate bridge and is
//! in `tests/connection_plate.rs` (retirement plan §20); the Python file is gone.
//!
//! # What the gong adds to the family
//!
//! Every other radiating thing in this repo is linear in its excitation, so radiated fraction,
//! directivity and dipole-over-baffled are amplitude-**invariant** by construction. The von Kármán
//! coupling is quadratic, so the *shape* of the motion evolves during a single strike, and shape is
//! what a distributed port was built to make audible. It also adds a second error axis no linear
//! surface has: model #6 conserves only at the **fixed point** of its per-step iteration, so the
//! iteration tolerance `couple_tol` is an error source sitting beside the air load — and the three
//! detectors split cleanly along it ([`couple_tol_moves_the_total_and_not_the_money_test`]).
//!
//! # The anchor that discriminates the whole seam
//!
//! [`nonlinear_false_is_the_linear_room_loaded_plate_bit_identical`]: with the coupling off, the
//! gong in a room must reproduce the *linear* plate in the same room to the last bit, with `sigma >
//! 0` and a live `f_ext` throughout. It is simultaneously a check on three things nothing else here
//! sees:
//!
//! * **the areal density** — model #6 is specified by a *volume* density, and writing `rho_v` where
//!   `rho_s` belongs leaves the air load 1000x too weak at `e = 1 mm` while every ledger still
//!   telescopes against the pressure it used;
//! * **the force path** — the gong's seam adds `f_ext` outside the plate's own right-hand side, the
//!   linear seam inside it, so their agreement is measured rather than shared code;
//! * **the operator** — the seam must iterate against the *loaded* factorization. Wired to the
//!   plate's own (`VkCoupledStep::new` instead of `with_rhs`), it iterates happily and simply is
//!   not in the room.

use physsynth_core::airbox::{
    impedance_from_zeta, AirBox, Params as RoomParams, Wall, C0_AIR, RHO0_AIR,
};
use physsynth_core::airbox_port::Spreading;
use physsynth_core::airbox_wrap::{GridPort, GridSeam, PlateSeam, RoomGrid, VkSeam, WrapError};
use physsynth_core::plate::{
    self, Boundary, CoupleMethod, CoupleOutcome, Domain, Plate, PlateSpec, VkParams, VkPlate,
    VkSpec,
};
use physsynth_core::sparse::Csr;

// -- the suite's fixtures, `tests/helpers.py` ---------------------------------------------------

const CFL: f64 = 0.9;
const FS: f64 = 8000.0;
const N_ROOM: [usize; 3] = [12, 11, 9];
const PLATE_L: f64 = 0.30;
const DIPOLE_INDEX: i64 = 4;

// `AIRBOX_VK_MATERIAL`: a steel sheet, so the nonlinearity's onset is at `w ~ e = 1 mm`.
const YOUNG: f64 = 2.0e11;
const THICKNESS: f64 = 1.0e-3;
const NU: f64 = 0.3;
/// **Volumetric** (kg/m^3). The areal density the air load divides by is `rho_v e`.
const RHO_V: f64 = 7800.0;

/// The scene total, relative — necessary, not sufficient.
const DRIFT_TOL: f64 = 1e-12;
/// `|radiated - injected| / |radiated|` — the money test, ALSO not sufficient.
const LEDGER_TOL: f64 = 1e-12;

const BOUNDARIES: [Boundary; 2] = [Boundary::Supported, Boundary::Free];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mount {
    Baffled,
    Suspended,
}

const MOUNTS: [Mount; 2] = [Mount::Baffled, Mount::Suspended];

fn rigid() -> [Wall; 6] {
    [Wall::Rigid; 6]
}

fn all_lossy() -> [Wall; 6] {
    [Wall::Impedance(impedance_from_zeta(4.0, RHO0_AIR, C0_AIR)); 6]
}

/// A lossy **mounting** wall (`z0`), rigid everywhere else.
fn lossy_mounting_wall() -> [Wall; 6] {
    let mut w = rigid();
    w[4] = Wall::Impedance(impedance_from_zeta(3.0, RHO0_AIR, C0_AIR));
    w
}

fn wall_cases() -> [(&'static str, [Wall; 6]); 3] {
    [
        ("rigid", rigid()),
        ("all-lossy", all_lossy()),
        ("one-lossy-wall", lossy_mounting_wall()),
    ]
}

fn make_room(fs: f64, walls: [Wall; 6]) -> AirBox {
    let h = C0_AIR * 3.0f64.sqrt() / (CFL * fs);
    let l = [
        N_ROOM[0] as f64 * h,
        N_ROOM[1] as f64 * h,
        N_ROOM[2] as f64 * h,
    ];
    AirBox::new(RoomParams::new(l, fs, h, walls, None, RHO0_AIR, C0_AIR).expect("a legal room"))
}

/// `make_air_vk_plate`'s arguments, with its defaults.
#[derive(Debug, Clone, Copy)]
struct Vk {
    boundary: Boundary,
    sigma: f64,
    nonlinear: bool,
    n: i64,
    fs: f64,
    couple_tol: f64,
    couple_max_iter: i64,
    method: CoupleMethod,
}

impl Default for Vk {
    fn default() -> Self {
        Vk {
            boundary: Boundary::Supported,
            sigma: 0.0,
            nonlinear: true,
            n: 8,
            fs: FS,
            couple_tol: 1e-13,
            couple_max_iter: 50,
            method: CoupleMethod::Auto,
        }
    }
}

impl Vk {
    fn params(self) -> VkParams {
        VkParams::new(&VkSpec {
            lx: PLATE_L,
            ly: PLATE_L,
            young: YOUNG,
            thickness: THICKNESS,
            nu: NU,
            rho: RHO_V,
            fs: self.fs,
            n: self.n,
            sigma: self.sigma,
            theta: plate::THETA_DEFAULT,
            boundary: Some(self.boundary),
            nonlinear: self.nonlinear,
            couple_tol: self.couple_tol,
            couple_max_iter: self.couple_max_iter,
            couple_method: Some(self.method),
        })
        .expect("a legal von Karman plate")
    }

    fn plate(self) -> VkPlate {
        VkPlate::new(self.params())
    }

    /// The gong mounted in a fresh room with `walls` -- flush in `z0`, or hung on the interior
    /// `z` plane at [`DIPOLE_INDEX`].
    fn mount(self, mount: Mount, walls: [Wall; 6]) -> (AirBox, RoomGrid<VkSeam>) {
        let mut room = make_room(self.fs, walls);
        let inst = mount_in(mount, VkSeam::new(self.plate()), &mut room);
        (room, inst)
    }
}

fn mount_in<S: GridSeam>(mount: Mount, seam: S, room: &mut AirBox) -> RoomGrid<S> {
    match mount {
        Mount::Baffled => RoomGrid::baffled(seam, room, "z0", None, Spreading::Bilinear),
        Mount::Suspended => {
            RoomGrid::suspended(seam, room, "z", DIPOLE_INDEX, None, Spreading::Bilinear)
        }
    }
    .expect("a legal mounting")
}

/// Iterate the live nodes of a plate's grid in its own C-order, handing back `(x, y)`.
fn live_xy(p: &plate::Params) -> Vec<(f64, f64)> {
    let mut out = Vec::with_capacity(p.n_live);
    for j in 0..p.mask.nrows() {
        for i in 0..p.mask.ncols() {
            if p.mask.at(j, i) {
                let at = j * p.mask.ncols() + i;
                out.push((p.x[at], p.y[at]));
            }
        }
    }
    out
}

/// `tests/helpers.py::vk_strike` — a centred raised Gaussian, peak `amplitude`, width `0.2 Lx`.
///
/// Broad on purpose: a *narrow* strike is what fails to converge at large amplitude.
fn vk_strike(p: &plate::Params, amplitude: f64) -> Vec<f64> {
    let w = 0.20 * p.lx;
    live_xy(p)
        .into_iter()
        .map(|(x, y)| {
            let (dx, dy) = (x - 0.5 * p.lx, y - 0.5 * p.ly);
            amplitude * (-((dx * dx + dy * dy) / (w * w))).exp()
        })
        .collect()
}

/// A constant off-centre nodal force (N) -- large enough that `f_ext` is not decorative: 1 N moves
/// this plate about a thickness over the fifty steps the regressions run.
fn drive(p: &plate::Params) -> Vec<f64> {
    let w = 0.12 * p.lx;
    live_xy(p)
        .into_iter()
        .map(|(x, y)| {
            let (dx, dy) = (x - 0.42 * p.lx, y - 0.38 * p.ly);
            (-((dx * dx + dy * dy) / (w * w))).exp()
        })
        .collect()
}

/// A structureless field (splitmix64 of the index, centred), for a bar whose claim is about a
/// *class* of forces rather than one force.
fn noise(seed: u64, i: usize) -> f64 {
    let mut z = seed
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(i as u64)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    2.0 * ((z >> 11) as f64 / (1u64 << 53) as f64) - 1.0
}

/// A struck gong at `w = 2e` -- the nonlinearity live, 19 iterations at the peak.
///
/// `2e` rather than a louder strike because the iteration count is a strong function of amplitude
/// and a guard that costs four times as much says nothing more.
fn seed_strike(inst: &mut RoomGrid<VkSeam>) {
    let u0 = vk_strike(&inst.seam.plate.p.lin, 2.0 * THICKNESS);
    inst.set_state(&u0, &vec![0.0; u0.len()]);
}

fn run<S: GridSeam>(inst: &mut RoomGrid<S>, room: &mut AirBox, steps: usize, f: Option<&[f64]>) {
    for _ in 0..steps {
        inst.step(room, f).expect("a legal step");
        room.step();
    }
}

fn scene_energy<S: GridSeam>(inst: &RoomGrid<S>, room: &AirBox) -> f64 {
    inst.energy() + room.energy()
}

/// `tests/helpers.py::vk_linear_twin` — the linear [`Plate`] a `nonlinear = false` gong reduces
/// to, built from the gong's **own** derived numbers: `rho = rho_s` (never `rho_v`), the snapped
/// `Ly`, `kappa = sqrt(D / rho_s)`, and `nu`, which is inert supported and load-bearing free.
fn linear_twin(vk: &VkParams) -> Plate {
    let p = &vk.lin;
    Plate::new(
        plate::Params::new(&PlateSpec {
            lx: p.lx,
            ly: p.ly,
            kappa: p.kappa,
            rho: vk.rho_s,
            fs: p.fs,
            n: p.n as i64,
            sigma: p.sigma,
            theta: p.theta,
            boundary: Some(p.boundary),
            domain: Some(Domain::Rectangle),
            nu: Some(p.nu),
            ..PlateSpec::default()
        })
        .expect("a legal linear twin"),
    )
}

/// Rebuild the factorization with the load block scaled by `scale`, leaving the port's own load
/// as built -- so only the *operator* sees the wrong coupling.
fn refactor_with_load_scaled(inst: &mut RoomGrid<VkSeam>, scale: f64) {
    let original = inst.port.load_matrix().clone();
    set_load(inst, original.scaled(scale));
    inst.refactor().expect("a scaled SPD load still factors");
    set_load(inst, original);
}

fn set_load(inst: &mut RoomGrid<VkSeam>, m: Csr) {
    match &mut inst.port {
        GridPort::Baffled(p) => p.set_load_matrix(m),
        GridPort::Suspended(p) => p.set_load_matrix(m),
    }
}

/// Disconnect the coupling: `T = 0`, `T^T R T = 0`, and the factorization rebuilt through the
/// wrapper's own one assembly spelling.
fn zero_the_load(inst: &mut RoomGrid<VkSeam>) {
    let (rows, cols) = {
        let t = inst.port.t();
        (t.nrows(), t.ncols())
    };
    let empty_t = Csr::from_rows(rows, cols, vec![Vec::new(); rows]);
    let empty_load = Csr::from_rows(cols, cols, vec![Vec::new(); cols]);
    match &mut inst.port {
        GridPort::Baffled(p) => {
            p.set_t(empty_t);
            p.set_load_matrix(empty_load);
        }
        GridPort::Suspended(p) => {
            p.set_t(empty_t);
            p.set_load_matrix(empty_load);
        }
    }
    inst.refactor().expect("a bare plate's own matrix factors");
}

/// Flat C-order offset of a pressure node.
fn flat_index(room: &AirBox, i: [usize; 3]) -> usize {
    let s = [room.p.n[0] + 1, room.p.n[1] + 1, room.p.n[2] + 1];
    (i[0] * s[1] + i[1]) * s[2] + i[2]
}

// -- the split point: coupling off is the linear plate, bit for bit ---------------------------

/// `nonlinear = false` in a room must reproduce the linear plate in the same room **exactly**.
///
/// `==` on the displacement every step, then on both stored levels, the coupling ledger, the
/// total and the net volume velocity -- for fifty steps with `sigma > 0` beside a live `f_ext`,
/// which is deliberate: the `(1 + sigma k)` in `A` and the `sigma k w^{n-1}` in the right-hand side
/// are where a density slip could hide *asymmetrically*, and a lossless run never reaches them.
/// See the module header for the three things this one equality discriminates.
///
/// Each side gets its own room: two ports on one set of nodes is refused, and the rooms are
/// identical by construction.
#[test]
fn nonlinear_false_is_the_linear_room_loaded_plate_bit_identical() {
    for boundary in BOUNDARIES {
        for mount in MOUNTS {
            for sigma in [0.0, 2.0] {
                let vk = Vk {
                    boundary,
                    sigma,
                    nonlinear: false,
                    ..Vk::default()
                };
                let (mut room, mut inst) = vk.mount(mount, lossy_mounting_wall());
                let mut ref_room = make_room(FS, lossy_mounting_wall());
                let twin = PlateSeam::new(linear_twin(&inst.seam.plate.p));
                let mut reference = mount_in(mount, twin, &mut ref_room);

                let u0 = vk_strike(&inst.seam.plate.p.lin, 1e-4);
                let v0 = vec![0.0; u0.len()];
                let f_ext = drive(&inst.seam.plate.p.lin);
                inst.set_state(&u0, &v0);
                reference.set_state(&u0, &v0);
                for step in 0..50 {
                    inst.step(&mut room, Some(&f_ext)).expect("a legal step");
                    room.step();
                    reference
                        .step(&mut ref_room, Some(&f_ext))
                        .expect("a legal step");
                    ref_room.step();
                    assert_eq!(
                        inst.seam.plate.u, reference.seam.plate.u,
                        "{boundary:?}/{mount:?}/sigma={sigma}: the linear reduction must be \
                         bit-identical, and it split at step {step}"
                    );
                }
                let tag = format!("{boundary:?}/{mount:?}/sigma={sigma}");
                assert_eq!(inst.seam.plate.u_prev, reference.seam.plate.u_prev, "{tag}");
                assert_eq!(inst.radiated_energy(), reference.radiated_energy(), "{tag}");
                assert_eq!(inst.energy(), reference.energy(), "{tag}");
                assert_eq!(inst.volume_velocity(), reference.volume_velocity(), "{tag}");
                assert_eq!(
                    room.pressure, ref_room.pressure,
                    "{tag}: the rooms heard the same"
                );
            }
        }
    }
}

/// Same `a_bare`, same `load_scale`, same load matrix -- so the same factorization.
///
/// The trajectory bar above would catch a wrong factorization too, but only as a difference in
/// the last digits of a state vector. This says *where* it came from, and it is the bar a `rho_v`
/// slip fails first: `load_scale = k / (2 rho_s)` is one of the two places the areal density
/// enters, and a factor of 1000 there changes the stored fill as well as the values.
#[test]
fn the_loaded_factorization_matches_the_linear_one() {
    for boundary in BOUNDARIES {
        for mount in MOUNTS {
            let vk = Vk {
                boundary,
                nonlinear: false,
                ..Vk::default()
            };
            let (_room, inst) = vk.mount(mount, rigid());
            let mut ref_room = make_room(FS, rigid());
            let twin = PlateSeam::new(linear_twin(&inst.seam.plate.p));
            let reference = mount_in(mount, twin, &mut ref_room);
            let tag = format!("{boundary:?}/{mount:?}");
            assert_eq!(inst.load_scale(), reference.load_scale(), "{tag}");
            assert_eq!(inst.denominator(), reference.denominator(), "{tag}");
            assert_eq!(inst.nnz_growth(), reference.nnz_growth(), "{tag}");
            assert_eq!(inst.lu_nnz(), reference.lu_nnz(), "{tag}");
        }
    }
}

// -- the loaded operator is an argument, never the model's -------------------------------------

/// The loaded factorization reaches the coupled step as an **argument** and never replaces the
/// plate's own.
///
/// The reference asserted `plate._lu is not inst._lu_loaded`. Natively the plate's operator is a
/// field of its parameters and the wrapper's is a separate value, so identity is structural; what
/// is left to assert is the behavioural half -- that a run in the room leaves the plate's own
/// factorization the one a bare plate would have built, so bare-versus-loaded stays observable and
/// every reduction bar in this file stays a comparison rather than a tautology.
#[test]
fn the_loaded_factorization_is_never_assigned_to_the_model() {
    for boundary in BOUNDARIES {
        for mount in MOUNTS {
            let vk = Vk {
                boundary,
                ..Vk::default()
            };
            let (mut room, mut inst) = vk.mount(mount, rigid());
            seed_strike(&mut inst);
            run(&mut inst, &mut room, 5, None);
            let own = inst.seam.plate.p.lin.lu.nnz();
            let fresh = vk.params().lin.lu.nnz();
            let (l, u) = inst.lu_loaded().nnz();
            assert_eq!(
                own, fresh,
                "{boundary:?}/{mount:?}: the plate's own operator changed"
            );
            assert_ne!(
                own.0 + own.1,
                l + u,
                "{boundary:?}/{mount:?}: the load did nothing"
            );
        }
    }
}

/// A smoke test, and named as one: the coupled iteration is live and converging under the load.
///
/// At `w = 2e` the coupled run must *diverge from* the linear one (else the loop is not running),
/// the iteration must converge on **every** step -- never sampled at the end, since a plate can sit
/// at the cap throughout and still converge on its last step -- and both cached stress levels must
/// be non-zero, which is the roll neither linear seam has.
#[test]
fn the_nonlinear_path_runs_inside_the_load() {
    for boundary in BOUNDARIES {
        for mount in MOUNTS {
            let base = Vk {
                boundary,
                ..Vk::default()
            };
            let (mut room, mut inst) = base.mount(mount, rigid());
            let (mut lin_room, mut linear) = Vk {
                nonlinear: false,
                ..base
            }
            .mount(mount, rigid());
            let u0 = vk_strike(&inst.seam.plate.p.lin, 2.0 * THICKNESS);
            let v0 = vec![0.0; u0.len()];
            inst.set_state(&u0, &v0);
            linear.set_state(&u0, &v0);
            for _ in 0..20 {
                inst.step(&mut room, None).expect("a legal step");
                room.step();
                linear.step(&mut lin_room, None).expect("a legal step");
                lin_room.step();
                assert!(
                    inst.seam.plate.converged,
                    "{boundary:?}/{mount:?}: the iteration hit the cap at step {}",
                    inst.n()
                );
            }
            let pl = &inst.seam.plate;
            assert!(pl.u.iter().all(|v| v.is_finite()));
            assert_ne!(pl.u, linear.seam.plate.u, "the coupling did nothing");
            assert!(pl.f.iter().any(|&v| v != 0.0) && pl.f_prev.iter().any(|&v| v != 0.0));
        }
    }
}

// -- the reduction: the load's zero is a clean zero, nonlinear path included --------------------

/// `T = 0` gives **bit-identical** state to a bare [`VkPlate`], coupling and all.
///
/// Both cached stress levels must come out identical too, which pins the seam's commit. It is also
/// the one bar that runs the loaded iteration against an independent driver of it -- the plate's
/// own step -- so a slip in the seed, the coupling factor or the convergence test shows up here as
/// an inequality rather than as a plausible trajectory.
#[test]
fn zero_area_reduces_to_the_bare_vk_plate() {
    for boundary in BOUNDARIES {
        for mount in MOUNTS {
            let vk = Vk {
                boundary,
                n: 6,
                sigma: 3.0,
                ..Vk::default()
            };
            let (mut room, mut inst) = vk.mount(mount, rigid());
            zero_the_load(&mut inst);
            let mut bare = vk.plate();
            let u0 = vk_strike(&bare.p.lin, 2.0 * THICKNESS);
            let v0 = vec![0.0; u0.len()];
            bare.set_state(&u0, &v0).expect("a seeded stress cache");
            inst.set_state(&u0, &v0);
            for _ in 0..50 {
                inst.step(&mut room, None).expect("a legal step");
                room.step();
                bare.step(None).expect("a legal step");
            }
            let pl = &inst.seam.plate;
            let tag = format!("{boundary:?}/{mount:?}");
            assert_eq!(
                pl.u, bare.u,
                "{tag}: the zero load must be bit-identical, not close"
            );
            assert_eq!(pl.u_prev, bare.u_prev, "{tag}");
            assert_eq!(pl.f, bare.f, "{tag}: the stress roll must reduce too");
            assert_eq!(pl.f_prev, bare.f_prev, "{tag}");
            assert_eq!(pl.n_iters, bare.n_iters, "{tag}");
            assert_eq!(pl.n_solves, bare.n_solves, "{tag}");
            assert_eq!(inst.radiated_energy(), 0.0, "{tag}");
        }
    }
}

// -- the money test, the channel it runs through, and the conserved total ----------------------

/// `radiated == injected`, reported **with the channel size**.
///
/// A conservation bar on a channel worth 1e-14 of the total passes with the coupling disconnected,
/// so the channel is asserted beside the identity. At `w = 2e` it is a few percent of `E0` for the
/// supported gong and a few tenths of a percent for the free cymbal -- the acoustic short circuit
/// doing its job on a strike that is a fine spatial pattern.
/// [`the_piston_is_the_free_plate_s_fat_channel`] is the configuration that makes the free arm
/// non-vacuous by a wide margin.
#[test]
fn ledgers_agree_and_the_channel_is_not_vacuous() {
    for boundary in BOUNDARIES {
        for mount in MOUNTS {
            for (wall_name, walls) in wall_cases() {
                let vk = Vk {
                    boundary,
                    ..Vk::default()
                };
                let (mut room, mut inst) = vk.mount(mount, walls);
                seed_strike(&mut inst);
                let e0 = scene_energy(&inst, &room);
                run(&mut inst, &mut room, 300, None);
                let radiated = inst.radiated_energy();
                let gap = (radiated - room.injected).abs();
                let tag = format!("{boundary:?}/{mount:?}/{wall_name}");
                assert!(gap <= LEDGER_TOL * radiated.abs(), "{tag}: gap {gap:.3e}");
                assert!(
                    radiated.abs() > 1e-3 * e0,
                    "{tag}: the channel must be worth asserting on ({:.3e} of E0)",
                    radiated / e0
                );
            }
        }
    }
}

/// A free plate's rigid-body **velocity** piston is still the fat channel.
///
/// Measured in the reference at 27.6% of `E0` baffled and 4.6% suspended. A rigid translation
/// stretches nothing, so `l(w, w) = 0` and the von Kármán coupling is asleep here -- which is the
/// point of running both configurations: this one asserts the ledger on a fat channel, the strike
/// asserts it with the nonlinearity awake.
#[test]
fn the_piston_is_the_free_plate_s_fat_channel() {
    for mount in MOUNTS {
        let vk = Vk {
            boundary: Boundary::Free,
            ..Vk::default()
        };
        let (mut room, mut inst) = vk.mount(mount, all_lossy());
        let n_live = inst.seam.n_live();
        inst.set_state(&vec![0.0; n_live], &vec![1e-3; n_live]);
        let e0 = scene_energy(&inst, &room);
        run(&mut inst, &mut room, 300, None);
        let radiated = inst.radiated_energy();
        assert!(
            radiated.abs() > 0.04 * e0,
            "{mount:?}: {:.3e} of E0",
            radiated / e0
        );
        assert!((radiated - room.injected).abs() <= LEDGER_TOL * radiated.abs());
    }
}

/// Lossless plate, rigid room, nonlinearity live: the scene total is flat.
///
/// Necessary and not sufficient -- and bounded below by `couple_tol` rather than by the air load,
/// because model #6 conserves only at its fixed point.
/// [`couple_tol_moves_the_total_and_not_the_money_test`] separates the two.
#[test]
fn the_scene_total_is_flat() {
    for boundary in BOUNDARIES {
        for mount in MOUNTS {
            let vk = Vk {
                boundary,
                ..Vk::default()
            };
            let (mut room, mut inst) = vk.mount(mount, rigid());
            seed_strike(&mut inst);
            let e0 = scene_energy(&inst, &room);
            let mut worst: f64 = 0.0;
            for _ in 0..300 {
                inst.step(&mut room, None).expect("a legal step");
                room.step();
                assert!(
                    inst.seam.plate.converged,
                    "the iteration hit the cap at {}",
                    inst.n()
                );
                worst = worst.max((scene_energy(&inst, &room) - e0).abs());
            }
            assert!(
                worst <= DRIFT_TOL * e0.abs(),
                "{boundary:?}/{mount:?}: drift {:.3e}",
                worst / e0
            );
        }
    }
}

/// Lossy plate **and** lossy room: the scene total never rises. The reference measured the rise
/// at exactly 0.0.
///
/// The load is proportional to `w^{n+1} - w^{n-1}`, so placing it inside `A` makes it dissipative
/// by construction rather than by an inequality that has to be checked each step.
#[test]
fn the_lossy_scene_total_is_monotone() {
    for boundary in BOUNDARIES {
        for mount in MOUNTS {
            let vk = Vk {
                boundary,
                sigma: 1.0,
                ..Vk::default()
            };
            let (mut room, mut inst) = vk.mount(mount, all_lossy());
            seed_strike(&mut inst);
            let e0 = scene_energy(&inst, &room);
            let mut prev = e0;
            let mut rise: f64 = 0.0;
            for _ in 0..300 {
                inst.step(&mut room, None).expect("a legal step");
                room.step();
                let cur = scene_energy(&inst, &room);
                rise = rise.max(cur - prev);
                prev = cur;
            }
            assert!(
                rise <= 1e-14 * e0.abs(),
                "{boundary:?}/{mount:?}: the scene total rose by {rise:.3e}"
            );
        }
    }
}

/// `energy()` is the plate's **plus** the coupling channel.
///
/// The failure this guards is silent and caught four earlier batches: the plate's own energy is
/// the total *without* its coupling channel -- the number that looks fine and is not conserved.
#[test]
fn energy_is_an_override_and_not_a_delegation() {
    for boundary in BOUNDARIES {
        for mount in MOUNTS {
            let vk = Vk {
                boundary,
                ..Vk::default()
            };
            let (mut room, mut inst) = vk.mount(mount, all_lossy());
            seed_strike(&mut inst);
            run(&mut inst, &mut room, 100, None);
            let own = inst.seam.plate.energy();
            assert_ne!(inst.radiated_energy(), 0.0);
            assert_eq!(inst.energy(), own + inst.radiated_energy());
            assert_ne!(inst.energy(), own);
        }
    }
}

// -- the primary guard: the achieved state back in the COUPLED PDE -----------------------------

/// The coupled residual of **two** fields, rebuilt from the room's own pressure.
///
/// The force comes from the **room's post-closure pressure** -- a number the port never touched --
/// and the von Kármán term from the **committed** `(w^{n+1}, F^{n+1})`. Both halves are
/// load-bearing: built from the last sweep's cached coupling, this reports the iteration's
/// increment instead; and `F^{n-1}` has to be captured *before* the step, because the commit rolls
/// it away and the scheme's average is `(F^{n+1} + F^{n-1}) / 2`.
///
/// Two timesteps, because a wrong-but-consistent `k`-dependent factor passes at one; `sigma > 0`, a
/// live `f_ext` and a lossy wall, so nothing is invisible. Three negative controls, and each must
/// be seen: the coupling dropped, the coupling halved, and the **air load** halved inside the
/// factorization -- so the residual sees the nonlinear force and the air load separately, which no
/// ledger here does.
#[test]
fn the_coupled_residual_at_two_timesteps() {
    for boundary in BOUNDARIES {
        for mount in MOUNTS {
            for fs in [8000.0, 11000.0] {
                let tag = format!("{boundary:?}/{mount:?}/{fs}");
                let correct = coupled_residual(boundary, mount, fs, "correct");
                assert!(correct <= 1e-11, "{tag}: residual {correct:.3e}");
                for control in ["no-coupling", "half-coupling", "half-load"] {
                    let r = coupled_residual(boundary, mount, fs, control);
                    assert!(r > 1e-6, "{tag}: the residual must see {control} ({r:.3e})");
                }
            }
        }
    }
}

fn coupled_residual(boundary: Boundary, mount: Mount, fs: f64, control: &str) -> f64 {
    let vk = Vk {
        boundary,
        n: 6,
        sigma: 2.0,
        fs,
        ..Vk::default()
    };
    let (mut room, mut inst) = vk.mount(mount, lossy_mounting_wall());
    if control == "half-load" {
        refactor_with_load_scaled(&mut inst, 0.5);
    }
    seed_strike(&mut inst);
    run(&mut inst, &mut room, 5, None);

    let n_live = inst.seam.n_live();
    let f_ext: Vec<f64> = (0..n_live).map(|i| 1e-3 * noise(0, i)).collect();
    let u_n = inst.seam.plate.u.clone();
    let u_nm1 = inst.seam.plate.u_prev.clone();
    let f_nm1 = inst.seam.plate.f_prev.clone(); // F^{n-1}: the commit rolls it away
    let p_old = room.pressure.clone();
    inst.step(&mut room, Some(&f_ext)).expect("a legal step");
    room.step();
    let u_np1 = inst.seam.plate.u.clone();
    let f_np1 = inst.seam.plate.f.clone(); // F^{n+1}, COMMITTED -- never a cached sweep

    // The time-averaged pressure the room itself saw, at the port's own nodes.
    let pbar = |a: usize| 0.5 * (room.pressure[a] + p_old[a]);
    let load: Vec<f64> = match &inst.port {
        GridPort::Baffled(p) => {
            let nodes = p.nodes();
            (0..p.node_count())
                .map(|m| pbar(flat_index(&room, [nodes[0][m], nodes[1][m], nodes[2][m]])))
                .collect()
        }
        GridPort::Suspended(p) => {
            let (lo, hi) = (p.nodes_lo(), p.nodes_hi());
            (0..p.face_count())
                .map(|m| {
                    pbar(flat_index(&room, [hi[0][m], hi[1][m], hi[2][m]]))
                        - pbar(flat_index(&room, [lo[0][m], lo[1][m], lo[2][m]]))
                })
                .collect()
        }
    };
    let back = inst.port.t().transpose().matvec(&load);

    let vkp = &inst.seam.plate.p;
    let p = &vkp.lin;
    let (k, theta) = (p.k, p.theta);
    let average: Vec<f64> = (0..n_live)
        .map(|i| theta * u_np1[i] + (1.0 - 2.0 * theta) * u_n[i] + theta * u_nm1[i])
        .collect();
    let s_avg = p.stiffness.matvec(&average);
    let (wn1, wm1) = (vkp.to_full(&u_np1), vkp.to_full(&u_nm1));
    let w_avg: Vec<f64> = (0..vkp.n_nodes).map(|i| 0.5 * (wn1[i] + wm1[i])).collect();
    let f_avg: Vec<f64> = (0..vkp.n_nodes)
        .map(|i| 0.5 * (f_np1[i] + f_nm1[i]))
        .collect();
    let coupling = vkp.to_live(&vkp.bracket.eval(&w_avg, &f_avg));
    let (mass, weight, couple_scale) = match boundary {
        Boundary::Supported => (vkp.rho_s * p.h * p.h, vec![1.0; n_live], k * k / vkp.rho_s),
        Boundary::Free => (vkp.rho_s, p.w.clone(), k * k * p.h * p.h / vkp.rho_s),
    };
    let scale = match control {
        "no-coupling" => 0.0,
        "half-coupling" => 0.5,
        _ => 1.0,
    };
    let mut worst: f64 = 0.0;
    let mut norm: f64 = 0.0;
    for i in 0..n_live {
        let accel = u_np1[i] - 2.0 * u_n[i] + u_nm1[i];
        let velocity = (u_np1[i] - u_nm1[i]) / (2.0 * k);
        let res = weight[i] * accel
            + k * k * p.kappa * p.kappa * s_avg[i]
            + 2.0 * p.sigma * k * k * weight[i] * velocity
            - k * k * (f_ext[i] - back[i]) / mass
            - scale * couple_scale * coupling[i];
        worst = worst.max(res.abs());
        norm = norm.max((weight[i] * accel).abs());
    }
    worst / norm
}

// -- the two-parameter money test: a second error axis, and which detector sees it -------------

/// `couple_tol` is an error source *alongside* the air load, and the three detectors split on it.
///
/// Measured in the reference (supported, baffled; the other three combinations alike):
///
/// ```text
/// couple_tol                 1e-13     1e-6      1e-3
/// scene-total drift / E0     1.2e-13   5.1e-7    1.0e-3
/// |radiated - injected|      2.2e-15   8.7e-16   1.6e-15
/// last_residual              9.9e-14   9.9e-7    1.0e-3
/// iterations                 19        9         4
/// ```
///
/// **The money test is blind, for a third distinct reason in this family:** `radiated == injected`
/// is arithmetic on whatever `w^{n+1}` came out of the solve, so an under-converged one is ported
/// *self-consistently* and the identity holds to rounding while the physics is off by a part in a
/// thousand. The scene total catches it, because the committed `F^{n+1}` is the stress of the
/// *previous* iterate while `w^{n+1}` is the current one, and the gap between them is exactly the
/// increment the tolerance bounds.
///
/// The self-certifying half, and the only part about the air at all: **the loaded drift falls with
/// `couple_tol` at the same rate as the unloaded one** -- the air adds no error floor of its own.
///
/// Not `#[ignore]`d. The reference marked it slow (it cost a large share of the file's 65 s); the
/// same work natively is a fraction of a second in release and a few seconds in debug.
#[test]
fn couple_tol_moves_the_total_and_not_the_money_test() {
    for boundary in BOUNDARIES {
        for mount in MOUNTS {
            let mut drifts = Vec::new();
            for tol in [1e-13, 1e-6, 1e-3] {
                let vk = Vk {
                    boundary,
                    couple_tol: tol,
                    ..Vk::default()
                };
                let (mut room, mut inst) = vk.mount(mount, rigid());
                seed_strike(&mut inst);
                let mut bare = vk.plate();
                let u0 = vk_strike(&bare.p.lin, 2.0 * THICKNESS);
                bare.set_state(&u0, &vec![0.0; u0.len()])
                    .expect("a seeded stress cache");
                let e0 = scene_energy(&inst, &room);
                let b0 = bare.energy();
                let (mut worst, mut bare_worst): (f64, f64) = (0.0, 0.0);
                for _ in 0..200 {
                    inst.step(&mut room, None).expect("a legal step");
                    room.step();
                    bare.step(None).expect("a legal step");
                    worst = worst.max((scene_energy(&inst, &room) - e0).abs());
                    bare_worst = bare_worst.max((bare.energy() - b0).abs());
                }
                let tag = format!("{boundary:?}/{mount:?}/tol={tol:e}");
                let radiated = inst.radiated_energy();
                let gap = (radiated - room.injected).abs() / radiated.abs();
                assert!(
                    gap <= LEDGER_TOL,
                    "{tag}: the money test is supposed to MISS this"
                );
                // Load-bearing at 1e-6 and 1e-3 only: at 1e-13 both sides sit at the rounding
                // floor and the comparison passes for reasons unrelated to the air load.
                assert!(
                    worst <= 3.0 * bare_worst,
                    "{tag}: the air load added an error floor of its own"
                );
                assert!(inst.seam.plate.last_residual <= tol, "{tag}");
                drifts.push(worst / e0.abs());
            }
            let (tight, middle, loose) = (drifts[0], drifts[1], drifts[2]);
            let tag = format!("{boundary:?}/{mount:?}");
            assert!(
                tight <= DRIFT_TOL,
                "{tag}: a converged run must be flat, {tight:.2e}"
            );
            assert!(
                middle > 1e-9,
                "{tag}: the total is supposed to CATCH an under-converged run"
            );
            assert!(
                loose > 1e-5 && loose > middle && middle > tight,
                "{tag}: {drifts:?}"
            );
        }
    }
}

// -- the seam drives the model's own kernel: every read-out, and `couple_method` ----------------

/// All six read-outs, not the three a hand-written copy happened to set.
///
/// The binding's seam once carried its own transcription of the iteration, and `n_solves` and
/// `residual_ratio` sat at their construction values for the life of a room-driven plate. Natively
/// the seam commits through [`VkPlate::record`], the bare step's own path; this bar is what says
/// that path was actually taken. The identity is Picard's -- a sweep is two back-substitutions, one
/// Airy and one theta-scheme -- which is exactly why the count cannot be inferred from `n_iters`
/// once the method can change.
#[test]
fn the_room_seam_writes_every_read_out_the_bare_step_writes() {
    for boundary in BOUNDARIES {
        for mount in MOUNTS {
            let vk = Vk {
                boundary,
                ..Vk::default()
            };
            let (mut room, mut inst) = vk.mount(mount, rigid());
            seed_strike(&mut inst);
            run(&mut inst, &mut room, 5, None);
            let p = &inst.seam.plate;
            let tag = format!("{boundary:?}/{mount:?}");
            assert!(p.converged && p.n_iters >= 2, "{tag}");
            assert_eq!(
                p.n_solves,
                2 * p.n_iters,
                "{tag}: the cost of the solve was not reported"
            );
            assert!(
                p.residual_ratio.is_finite(),
                "{tag}: the exit ratio was never written"
            );
            assert_eq!(p.n_fallbacks, 0, "{tag}");
            assert_eq!(p.outcome(), CoupleOutcome::Converged, "{tag}");
        }
    }
}

/// A step short of sweeps reports `Capped`, not `Expansive`.
///
/// `couple_outcome` tests the exit ratio **positively** -- finite *and* under one -- so a ratio
/// nobody wrote is a NaN that lands in `Expansive`: "no cap at any size helps", said about a step a
/// bigger cap does fix. A cap of two is the cheapest way to make a step genuinely short rather than
/// genuinely diverging.
#[test]
fn a_short_capped_room_step_reports_capped_and_not_expansive() {
    for boundary in BOUNDARIES {
        for mount in MOUNTS {
            let vk = Vk {
                boundary,
                couple_max_iter: 2,
                method: CoupleMethod::Picard,
                ..Vk::default()
            };
            let (mut room, mut inst) = vk.mount(mount, rigid());
            seed_strike(&mut inst);
            run(&mut inst, &mut room, 3, None);
            let p = &inst.seam.plate;
            let tag = format!("{boundary:?}/{mount:?}");
            assert!(!p.converged, "{tag}: a cap of two was not short enough");
            // Pinned: below two sweeps there is no ratio and the verdict is `Unknown`, which would
            // read like this bug rather than like a changed default.
            assert_eq!(p.n_iters, 2, "{tag}");
            assert!(
                p.residual_ratio < 1.0,
                "{tag}: the sweeps were not contracting"
            );
            assert_eq!(p.outcome(), CoupleOutcome::Capped, "{tag}");
        }
    }
}

/// `couple_method` reaches the loaded operator.
///
/// The two methods must **differ** in cost -- a flag that changed nothing would pass a bare "it
/// still runs" -- and land on the **same root**, since they are two iterations on one equation. One
/// step from one seeded state, so the comparison is a solve rather than an accumulated trajectory.
#[test]
fn the_loaded_step_honours_couple_method() {
    for boundary in BOUNDARIES {
        for mount in MOUNTS {
            let one = |method| {
                let vk = Vk {
                    boundary,
                    method,
                    ..Vk::default()
                };
                let (mut room, mut inst) = vk.mount(mount, rigid());
                seed_strike(&mut inst);
                run(&mut inst, &mut room, 1, None);
                inst
            };
            let picard = one(CoupleMethod::Picard);
            let newton = one(CoupleMethod::Newton);
            let (pp, np) = (&picard.seam.plate, &newton.seam.plate);
            let tag = format!("{boundary:?}/{mount:?}");
            assert_ne!(np.n_solves, pp.n_solves, "{tag}: the flag changed nothing");
            assert!(np.converged && pp.converged, "{tag}");
            let scale = pp.u.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let gap =
                pp.u.iter()
                    .zip(&np.u)
                    .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()))
                    / scale;
            assert!(
                gap <= 1e-9,
                "{tag}: the two iterations landed apart ({gap:.2e})"
            );
        }
    }
}

/// Newton carries a strike that kills the loaded Picard loop -- a claim about what **runs**.
///
/// At `w = 6e` on this fixture the loaded fixed-point iteration fails to contract at all and the
/// scene total is NaN within three steps. The air load does not put it there (the bare plate at the
/// same rate dies at the same amplitude); what this asserts is that a scene denied the fix now has
/// it. `6e` rather than a smaller amplitude because **the wall moves with the grid** -- this plate
/// is `N = 8`, where Picard still survives 4.5e -- which is why the bar carries its own control.
#[test]
fn newton_carries_a_strike_that_kills_the_loaded_picard_loop() {
    let amp = 6.0 * THICKNESS;
    for mount in MOUNTS {
        let picard = Vk {
            method: CoupleMethod::Picard,
            ..Vk::default()
        };
        let (mut room, mut inst) = picard.mount(mount, rigid());
        let u0 = vk_strike(&inst.seam.plate.p.lin, amp);
        inst.set_state(&u0, &vec![0.0; u0.len()]);
        run(&mut inst, &mut room, 3, None);
        assert!(
            !inst.seam.plate.converged,
            "{mount:?}: the Picard control is supposed to be past the wall -- if it converges, \
             this bar asserts nothing and the amplitude has to move"
        );
        assert!(
            !scene_energy(&inst, &room).is_finite(),
            "{mount:?}: the wall is supposed to take the scene total with it"
        );

        let newton = Vk {
            method: CoupleMethod::Newton,
            ..Vk::default()
        };
        let (mut room, mut inst) = newton.mount(mount, rigid());
        inst.set_state(&u0, &vec![0.0; u0.len()]);
        let total0 = scene_energy(&inst, &room);
        let (mut lo, mut hi) = (total0, total0);
        for _ in 0..30 {
            inst.step(&mut room, None).expect("a legal step");
            room.step();
            assert!(
                inst.seam.plate.converged,
                "{mount:?}: Newton hit its own cap at step {}",
                inst.n()
            );
            let total = scene_energy(&inst, &room);
            lo = lo.min(total);
            hi = hi.max(total);
        }
        assert!(inst.seam.plate.u.iter().all(|v| v.is_finite()));
        assert!((hi - lo) / total0.abs() <= DRIFT_TOL, "{mount:?}");
    }
}

// -- refusals -----------------------------------------------------------------------------------

/// The port's solve is one timestep shared by both -- a refusal, not a resampling, and it names
/// the resonator the way the caller knows it.
#[test]
fn refuses_a_sample_rate_mismatch() {
    for mount in MOUNTS {
        let mut room = make_room(FS, rigid());
        let seam = VkSeam::new(
            Vk {
                fs: 2.0 * FS,
                ..Vk::default()
            }
            .plate(),
        );
        let refused = match mount {
            Mount::Baffled => RoomGrid::baffled(seam, &mut room, "z0", None, Spreading::Bilinear),
            Mount::Suspended => RoomGrid::suspended(
                seam,
                &mut room,
                "z",
                DIPOLE_INDEX,
                None,
                Spreading::Bilinear,
            ),
        };
        let err = refused.expect_err("a mismatched rate must be refused");
        assert!(
            matches!(err, WrapError::RateMismatch { label: "plate", .. }),
            "{mount:?}: {err:?}"
        );
        assert!(err.to_string().starts_with("sample-rate mismatch"), "{err}");
        assert!(
            room.claims.is_empty(),
            "{mount:?}: a refused wrapper must leave the room alone"
        );
    }
}
