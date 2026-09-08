//! Native acceptance bars for the **grid wrapper** tier — a `Plate` or a `Membrane` loaded by an
//! `AirBox` through a distributed port (HANDOFF §12.H, air-box batches 3–5).
//!
//! These carry over `tests/test_airbox_surface.py` (its wrapper half, the port half having gone in
//! plan §14), `tests/test_airbox_dipole.py` and `tests/test_airbox_membrane.py`. The von Kármán
//! seam and its two wrappers are the next batch and their bars land beside these.
//!
//! # The claim, in one line
//!
//! **A surface radiates according to the SHAPE of its motion, not only its net volume
//! displacement** — so a mode with exactly zero net volume velocity, which every one-port in this
//! repo (`AirRadiation`, `RadiatedBody`, `RationalAirLoad`, `RoomPort`) calls exactly silent, is
//! not. That is the acoustic short circuit, and it is *structural*: a one-port couples through a
//! single scalar and has no length scale on its surface, so no `R(omega)` reproduces it at any
//! order.
//!
//! # The conserved total is NOT the money test, and this file is organised around that
//!
//! The resonator's energy identity telescopes to `-k pbar . q` for *whatever* pressure was used in
//! the force, and the room's identity is exact for *whatever* injection it received — so the scene
//! total is the sum of two separately exact identities and stays flat even when the two disagree
//! with each other. Measured in [`conservation_is_blind_to_a_wrong_r`]: dropping the `1 + beta`
//! wall factor from `R_j` on a lossy mounting wall leaves the total drifting *smaller* than the
//! correct run's own drift — green, and not even in a suspicious direction — while
//! `|radiated - injected|` goes from exactly zero to a fifth of the channel. So conservation ships
//! as **necessary and not sufficient**, and the bars that carry the weight are the two ledgers
//! agreeing, `R_j` measured differentially off the room, the volume identity recomputed from the
//! resonator's own state, and the coupled residual at two timesteps.
//!
//! # Two traps that nothing else would catch
//!
//! * **A consistent sign flip in `T` is invisible to every energetic quantity.** `T^T R T` is
//!   sign-invariant, so the load matrix, the solve and `radiated_energy` come out *bit-identical*
//!   while the room's field is inverted. [`a_sign_flip_is_invisible_to_every_energy_quantity`]
//!   asserts the bit-identity of the **wrong** run: that is the point of it.
//! * **The surface must be centred**, and the second of the two reasons is measured here: a scene
//!   that is not mirror-symmetric about a silent mode's own plane re-excites it, and the leak is
//!   linear in the offset with no threshold, so "approximately centred" is not approximately
//!   silent. (The first reason — the load matrix's own equivariance — is
//!   `airbox_surface.rs::bilinear_equivariance_needs_centring`.)

use physsynth_core::airbox::{
    impedance_from_zeta, AirBox, Params as RoomParams, Wall, C0_AIR, RHO0_AIR,
};
use physsynth_core::airbox_port::Spreading;
use physsynth_core::airbox_wrap::{
    GridPort, GridSeam, MembraneSeam, PlateSeam, RoomGrid, WrapError,
};
use physsynth_core::membrane::{self, Membrane};
use physsynth_core::plate::{self, Boundary, Domain, Plate, PlateSpec};
use physsynth_core::sparse::Csr;
use physsynth_core::sparse_lu::SparseLu;

// -- the suite's fixtures, `tests/helpers.py` ---------------------------------------------------

const CFL: f64 = 0.9;
const FS: f64 = 8000.0;
const N_ROOM: [usize; 3] = [12, 11, 9];
const PLATE_L: f64 = 0.30;
const PLATE_KAPPA: f64 = 20.0;
/// Light enough that the air genuinely loads it.
const PLATE_RHO: f64 = 0.5;
const DIPOLE_INDEX: i64 = 4;

/// The scene total, relative — necessary, not sufficient (see the module header).
const DRIFT_TOL: f64 = 1e-12;
/// `|radiated - injected| / |radiated|` — the money test.
const LEDGER_TOL: f64 = 1e-12;

fn h_for(fs: f64) -> f64 {
    C0_AIR * 3.0f64.sqrt() / (CFL * fs)
}

fn rigid() -> [Wall; 6] {
    [Wall::Rigid; 6]
}

fn all_lossy() -> [Wall; 6] {
    [Wall::Impedance(impedance_from_zeta(4.0, RHO0_AIR, C0_AIR)); 6]
}

/// A lossy **mounting** wall (`z0`) and rigid everywhere else — where the `1 + beta` closure
/// factor in `R_j` is the only thing that matters.
fn lossy_mounting_wall() -> [Wall; 6] {
    let mut w = rigid();
    w[4] = Wall::Impedance(impedance_from_zeta(3.0, RHO0_AIR, C0_AIR));
    w
}

fn make_room(fs: f64, n: [usize; 3], walls: [Wall; 6]) -> AirBox {
    let h = h_for(fs);
    let l = [n[0] as f64 * h, n[1] as f64 * h, n[2] as f64 * h];
    AirBox::new(RoomParams::new(l, fs, h, walls, None, RHO0_AIR, C0_AIR).expect("a legal room"))
}

fn surface_room(walls: [Wall; 6]) -> AirBox {
    make_room(FS, N_ROOM, walls)
}

#[allow(clippy::too_many_arguments)]
fn plate(fs: f64, n: i64, boundary: Boundary, sigma: f64) -> Plate {
    let spec = PlateSpec {
        lx: PLATE_L,
        ly: PLATE_L,
        kappa: PLATE_KAPPA,
        rho: PLATE_RHO,
        fs,
        n,
        sigma,
        theta: plate::THETA_DEFAULT,
        boundary: Some(boundary),
        domain: Some(Domain::Rectangle),
        ..PlateSpec::default()
    };
    Plate::new(plate::Params::new(&spec).expect("a legal plate"))
}

fn seam(fs: f64, n: i64, boundary: Boundary, sigma: f64) -> PlateSeam {
    PlateSeam::new(plate(fs, n, boundary, sigma))
}

/// A `Plate` mounted flush in `face` of a room the caller keeps.
fn baffled(
    room: &mut AirBox,
    fs: f64,
    n: i64,
    boundary: Boundary,
    sigma: f64,
    face: &str,
    origin: Option<(f64, f64)>,
) -> RoomGrid<PlateSeam> {
    RoomGrid::baffled(
        seam(fs, n, boundary, sigma),
        room,
        face,
        origin,
        Spreading::Bilinear,
    )
    .expect("a legal baffled plate")
}

/// A `Plate` hung on the `z` plane at [`DIPOLE_INDEX`] — the dipole tier.
fn suspended(
    room: &mut AirBox,
    fs: f64,
    n: i64,
    boundary: Boundary,
    sigma: f64,
    index: i64,
) -> RoomGrid<PlateSeam> {
    RoomGrid::suspended(
        seam(fs, n, boundary, sigma),
        room,
        "z",
        index,
        None,
        Spreading::Bilinear,
    )
    .expect("a legal suspended plate")
}

/// Which tier a bar is running — the two arms of [`GridPort`], as a bar-side loop variable.
///
/// Batch 3 mounted a plate flush in a wall and let it radiate from every node; the wall did the
/// rest. Batch 4 takes the wall away: the plate hangs *in* the room, radiates from both faces, and
/// is driven by the pressure **jump** across it. That is not batch 3 with a sign — a source adds
/// sound to a room, an **object** also removes paths through it, whether or not it is moving. Every
/// bar the reference ran on both tiers runs on both here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mount {
    Baffled,
    Suspended,
}

const MOUNTS: [Mount; 2] = [Mount::Baffled, Mount::Suspended];

fn mounted(
    mount: Mount,
    room: &mut AirBox,
    fs: f64,
    n: i64,
    boundary: Boundary,
    sigma: f64,
) -> RoomGrid<PlateSeam> {
    match mount {
        Mount::Baffled => baffled(room, fs, n, boundary, sigma, "z0", None),
        Mount::Suspended => suspended(room, fs, n, boundary, sigma, DIPOLE_INDEX),
    }
}

/// Flat C-order offset of a pressure node — `RoomView::flat` without borrowing the room.
fn flat_index(room: &AirBox, i: [usize; 3]) -> usize {
    let s = [room.p.n[0] + 1, room.p.n[1] + 1, room.p.n[2] + 1];
    (i[0] * s[1] + i[1]) * s[2] + i[2]
}

/// Seed a wrapper with the generic struck bump and hand back the zero-velocity pair.
fn seed_bump(inst: &mut RoomGrid<PlateSeam>) {
    let u0 = plate_bump(&inst.seam.plate.p, 1e-3);
    let v0 = vec![0.0; u0.len()];
    inst.set_state(&u0, &v0);
}

/// `tests/helpers.py::plate_bump` — a smooth off-centre bump on the live nodes.
///
/// Off-centre so it is not orthogonal to the antisymmetric modes. The free plate gets its mean
/// removed: a net piston is the most efficient radiator the geometry has, so leaving it in would
/// drown every other channel within a few hundred steps.
fn plate_bump(p: &plate::Params, amplitude: f64) -> Vec<f64> {
    let width = 0.08 * p.lx;
    let mut u0 = Vec::with_capacity(p.n_live);
    for j in 0..p.mask.nrows() {
        for i in 0..p.mask.ncols() {
            if !p.mask.at(j, i) {
                continue;
            }
            let x = p.x[j * p.mask.ncols() + i] - 0.42 * p.lx;
            let y = p.y[j * p.mask.ncols() + i] - 0.38 * p.ly;
            u0.push(amplitude * (-((x * x + y * y) / (width * width))).exp());
        }
    }
    if p.boundary == Boundary::Free {
        let mean = u0.iter().sum::<f64>() / u0.len() as f64;
        for v in u0.iter_mut() {
            *v -= mean;
        }
    }
    u0
}

/// `tests/helpers.py::plate_mode_shape` — the **exact** discrete mode `sin(m pi x/Lx) sin(n pi
/// y/Ly)` of a supported plate, scaled to rms 1.
///
/// Exact because `B = L^2` keeps the sine product an eigenvector of the scheme, which is what makes
/// `sum_i sin(m pi i/N) = 0` for even `m` an *identity* rather than an approximation.
fn plate_mode_shape(p: &plate::Params, m: usize, n: usize) -> Vec<f64> {
    assert_eq!(
        p.boundary,
        Boundary::Supported,
        "the modes are the supported ones"
    );
    let mut shape = Vec::with_capacity(p.n_live);
    for j in 0..p.mask.nrows() {
        for i in 0..p.mask.ncols() {
            if !p.mask.at(j, i) {
                continue;
            }
            let x = p.x[j * p.mask.ncols() + i];
            let y = p.y[j * p.mask.ncols() + i];
            shape.push(
                (m as f64 * std::f64::consts::PI * x / p.lx).sin()
                    * (n as f64 * std::f64::consts::PI * y / p.ly).sin(),
            );
        }
    }
    let rms = (shape.iter().map(|v| v * v).sum::<f64>() / shape.len() as f64).sqrt();
    for v in shape.iter_mut() {
        *v /= rms;
    }
    shape
}

/// A structureless field, for a bar whose claim is about a *class* of forces rather than one force
/// (plan §15): splitmix64 of the index, centred.
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

/// Step instrument and room in the contract's order: the port solves, then one room step.
fn run<S: GridSeam>(inst: &mut RoomGrid<S>, room: &mut AirBox, steps: usize) {
    for _ in 0..steps {
        inst.step(room, None).expect("a legal step");
        room.step();
    }
}

/// The conserved total of a surface scene: `sum_j inst_j.energy() + room.energy()`.
fn scene_energy<S: GridSeam>(insts: &[&RoomGrid<S>], room: &AirBox) -> f64 {
    insts.iter().map(|i| i.energy()).sum::<f64>() + room.energy()
}

const BOUNDARIES: [Boundary; 2] = [Boundary::Supported, Boundary::Free];
const FACES: [&str; 6] = ["x0", "x1", "y0", "y1", "z0", "z1"];

fn wall_cases() -> [(&'static str, [Wall; 6]); 3] {
    [
        ("rigid", rigid()),
        ("all-lossy", all_lossy()),
        ("lossy-mounting-wall", lossy_mounting_wall()),
    ]
}

// -- the money test -----------------------------------------------------------------------------

/// `radiated_energy == room.injected` — the bar a wrong `R_j` cannot survive.
///
/// The port predicts its work from `pbar = pbar_free + R q`; the room books the same work from its
/// **own** post-closure field, never from a number handed back. The two agree only if every `R_j`
/// is exactly right, which is why this and not the conserved total is the money test. The
/// lossy-mounting-wall case is what pins the `(1 + beta)` factor — the radiation leg left exactly
/// that kind of denominator unpinned twice in its history.
///
/// Suspended, the room books `k q . (pbar_hi - pbar_lo)` from its own post-closure pressure on the
/// two node planes, so the two agree only if every `R_j` is right *and* the `-q` / `+q` pair lands
/// on the right planes — but **not** only if the doubling is right, which is what
/// [`the_coupled_residual_catches_both_wrong_2s`] exists for.
#[test]
fn the_two_ledgers_agree() {
    let mut ran = 0;
    for mount in MOUNTS {
        for boundary in BOUNDARIES {
            for (name, walls) in wall_cases() {
                let mut room = surface_room(walls);
                let mut inst = mounted(mount, &mut room, FS, 8, boundary, 0.0);
                seed_bump(&mut inst);
                let e0 = inst.seam.plate.energy();
                run(&mut inst, &mut room, 300);
                let gap = (inst.radiated_energy() - room.injected).abs();
                assert!(
                    gap <= LEDGER_TOL * inst.radiated_energy().abs(),
                    "{mount:?}/{boundary:?}/{name}: ledger gap {gap:.3e}"
                );
                // ... and the channel is not vacuous: something actually went through it.
                assert!(
                    inst.radiated_energy().abs() > 1e-5 * e0,
                    "{mount:?}/{boundary:?}/{name}: the channel carried nothing"
                );
                ran += 1;
            }
        }
    }
    assert_eq!(
        ran, 12,
        "every tier x boundary x wall combination must have run"
    );
}

/// The conserved total stays green with a badly wrong coupling — so it is not sufficient.
///
/// Drops the wall-closure divisor from `R_j` on a **lossy** mounting wall (where it is the only
/// thing that matters) and reruns. This is the measurement the module header rests on, and it is
/// asserted rather than asserted-about: the drift must stay at rounding *and* the ledger gap must
/// become a real fraction. If a future change made the total sensitive to `R`, this bar fails and
/// the framing above is wrong.
///
/// The reference re-derived `A_loaded` by hand here, which was a second spelling of the wrapper's
/// own assembly; [`RoomGrid::refactor`] rebuilds it through the one spelling instead.
#[test]
fn conservation_is_blind_to_a_wrong_r() {
    fn drift_and_gap(naive: bool) -> (f64, f64) {
        let mut room = surface_room(lossy_mounting_wall());
        let mut inst = baffled(&mut room, FS, 8, Boundary::Supported, 0.0, "z0", None);
        if naive {
            let GridPort::Baffled(port) = &mut inst.port else {
                unreachable!("this scene is baffled")
            };
            let beta: Vec<f64> = port.flat().iter().map(|&i| room.p.beta[i]).collect();
            assert!(
                beta.iter().all(|&b| b > 0.0),
                "the mounting wall must be lossy or there is nothing to drop"
            );
            let r: Vec<f64> = port
                .r()
                .iter()
                .zip(beta.iter())
                .map(|(&r, &b)| r * (1.0 + b))
                .collect();
            let load = physsynth_core::airbox_port::load_matrix(port.t(), &r, 1.0);
            port.set_r(r);
            port.set_load_matrix(load);
            inst.refactor().expect("the mutated load still factors");
        }
        let u0 = plate_bump(&inst.seam.plate.p, 1e-3);
        inst.set_state(&u0, &vec![0.0; u0.len()]);
        let e0 = scene_energy(&[&inst], &room);
        let mut worst: f64 = 0.0;
        for _ in 0..300 {
            inst.step(&mut room, None).expect("a legal step");
            room.step();
            worst = worst.max((scene_energy(&[&inst], &room) - e0).abs());
        }
        let gap = (inst.radiated_energy() - room.injected).abs() / inst.radiated_energy().abs();
        (worst / e0.abs(), gap)
    }

    let (good_drift, good_gap) = drift_and_gap(false);
    let (bad_drift, bad_gap) = drift_and_gap(true);
    assert!(
        good_gap <= LEDGER_TOL,
        "the correct run's gap is {good_gap:.3e}"
    );
    assert!(
        good_drift <= DRIFT_TOL,
        "the correct run drifts {good_drift:.3e}"
    );
    assert!(
        bad_drift <= DRIFT_TOL,
        "the point of this bar is that the total does NOT notice; it drifted {bad_drift:.3e}"
    );
    assert!(
        bad_gap > 0.05,
        "the ledger gap must blow up, got {bad_gap:.3e}"
    );
}

/// `R_j` measured **differentially** off the room, and the off-diagonal asserted *exactly* zero.
///
/// Comparing the coupled step's `pbar` against `pbar_free + R q` would be a tautology — that
/// expression is how `pbar` was computed, and it passes for any `R` whatsoever. So: save the room,
/// step it once with nothing injected, restore, step it again with a unit injection at one node,
/// and read the difference straight off the room's own post-closure field.
///
/// The second assertion is what makes the *diagonal* load provable rather than plausible. Within a
/// step an injection changes the pressure at its own node and nowhere else — propagation waits for
/// the next momentum sub-step — so there is no cross-resistance at any separation, including
/// between nodes one cell apart. Exactly zero, not small.
#[test]
fn r_j_is_what_the_room_does() {
    let z3 = impedance_from_zeta(3.0, RHO0_AIR, C0_AIR);
    let z2 = impedance_from_zeta(2.0, RHO0_AIR, C0_AIR);
    let mut side_and_mount = rigid();
    side_and_mount[4] = Wall::Impedance(z3);
    side_and_mount[0] = Wall::Impedance(z2);
    let cases = [rigid(), lossy_mounting_wall(), side_and_mount];

    for (case, walls) in cases.into_iter().enumerate() {
        let mut room = surface_room(walls);
        let mut inst = baffled(&mut room, FS, 6, Boundary::Supported, 0.0, "z0", None);
        let u0 = plate_bump(&inst.seam.plate.p, 1e-3);
        inst.set_state(&u0, &vec![0.0; u0.len()]);
        run(&mut inst, &mut room, 7); // a nontrivial field, so pbar_free is not trivially zero

        let flat: Vec<usize> = match &inst.port {
            GridPort::Baffled(p) => p.flat().to_vec(),
            GridPort::Suspended(_) => unreachable!("this scene is baffled"),
        };
        let count = flat.len();
        let base = (
            room.pressure.clone(),
            room.u.clone(),
            room.u_prev.clone(),
            room.n,
        );
        let restore = |room: &mut AirBox, inst: &mut RoomGrid<PlateSeam>| {
            room.pressure = base.0.clone();
            room.u = base.1.clone();
            room.u_prev = base.2.clone();
            room.n = base.3;
            room.pending.clear();
            room.pending_ports.clear();
            inst.port.reset();
        };

        let p_before: Vec<f64> = flat.iter().map(|&i| room.pressure[i]).collect();
        room.step();
        let pbar0: Vec<f64> = flat
            .iter()
            .enumerate()
            .map(|(m, &i)| 0.5 * (room.pressure[i] + p_before[m]))
            .collect();

        let amp = 1e-4;
        let mut measured = vec![vec![0.0; count]; count];
        for j in 0..count {
            restore(&mut room, &mut inst);
            let mut q = vec![0.0; count];
            q[j] = amp;
            inst.port.inject(&mut room, &q).expect("a legal injection");
            room.step();
            for (m, &i) in flat.iter().enumerate() {
                measured[m][j] = (0.5 * (room.pressure[i] + p_before[m]) - pbar0[m]) / amp;
            }
        }
        restore(&mut room, &mut inst);

        let want = inst.port.r();
        for j in 0..count {
            let rel = (measured[j][j] - want[j]).abs() / want[j].abs();
            assert!(rel <= 1e-12, "case {case}, node {j}: R off by {rel:.3e}");
            for (m, row) in measured.iter().enumerate() {
                if m != j {
                    assert_eq!(row[j], 0.0, "case {case}: cross-resistance from {j} to {m}");
                }
            }
        }
    }
}

// -- the volume identity and the scheme itself --------------------------------------------------

/// `sum_j q_j == sum_n area_n v_n`, with `v` recomputed from the **resonator**.
///
/// The identity that makes the lumped monopole the low-frequency limit of the distributed port —
/// and the only bar a *consistently* wrong `q` factor (a stray `k`, or `2k` for `k`) cannot pass,
/// since both energy ledgers would use the same wrong `q` and agree happily. Which is why `v` is
/// rebuilt here from the plate's own two levels and never read off the port.
///
/// Normalised by `sum area |v|`: for a plate mode the net is a heavily cancelling sum, so a
/// relative tolerance against the net would be measuring the cancellation, not the identity.
#[test]
fn volume_is_conserved_exactly() {
    for mount in MOUNTS {
        for boundary in BOUNDARIES {
            let mut room = surface_room(all_lossy());
            let mut inst = mounted(mount, &mut room, FS, 8, boundary, 0.0);
            seed_bump(&mut inst);
            let k = inst.k();
            for step in 0..40 {
                let u_nm1 = inst.seam.plate.u_prev.clone();
                inst.step(&mut room, None).expect("a legal step");
                room.step();
                let v: Vec<f64> = inst
                    .seam
                    .plate
                    .u
                    .iter()
                    .zip(u_nm1.iter())
                    .map(|(&u, &p)| (u - p) / (2.0 * k))
                    .collect();
                let areas = inst.seam.areas();
                let net: f64 = inst.nodal_volume_velocity().iter().sum();
                let want: f64 = areas.iter().zip(v.iter()).map(|(&a, &v)| a * v).sum();
                let scale: f64 = areas.iter().zip(v.iter()).map(|(&a, &v)| a * v.abs()).sum();
                assert!(
                    (net - want).abs() <= 1e-13 * scale,
                    "{mount:?}/{boundary:?} step {step}: {:.3e}",
                    (net - want).abs() / scale
                );
            }
        }
    }
}

/// Put the achieved `u^{n+1}` back into the coupled PDE and check the residual vanishes.
///
/// Stronger than comparing against a hand-assembled copy of the same algebra: this writes the
/// scheme out — `delta_tt u = L(theta-average) - 2 sigma delta_t. u + f / rho_s` with
/// `f = f_ext - T^T (pbar_free + R q)` — and asks whether the solve satisfied it. Run at **two**
/// timesteps, because a wrong-but-consistent `k`-dependent factor passes at one; with `sigma > 0`,
/// a nonzero `f_ext` and a lossy mounting wall, so nothing is invisible. This is what pins the
/// `rho_s h^2` versus `rho_s` branch difference between the two boundaries.
#[test]
fn the_coupled_scheme_residual_vanishes() {
    for boundary in BOUNDARIES {
        for fs in [8000.0, 11000.0] {
            let mut room = make_room(fs, N_ROOM, lossy_mounting_wall());
            let mut inst = baffled(&mut room, fs, 6, boundary, 2.0, "z0", None);
            let u0 = plate_bump(&inst.seam.plate.p, 1e-3);
            inst.set_state(&u0, &vec![0.0; u0.len()]);
            run(&mut inst, &mut room, 5);

            let n_live = inst.seam.n_live();
            let f_ext: Vec<f64> = (0..n_live).map(|i| 1e-3 * noise(0, i)).collect();
            let u_n = inst.seam.plate.u.clone();
            let u_nm1 = inst.seam.plate.u_prev.clone();
            let pbar_free = inst.port.free_pressure(&room);
            inst.step(&mut room, Some(&f_ext)).expect("a legal step");
            room.step();
            let u_np1 = inst.seam.plate.u.clone();

            let p = &inst.seam.plate.p;
            let (k, theta) = (p.k, p.theta);
            let average: Vec<f64> = (0..n_live)
                .map(|i| theta * u_np1[i] + (1.0 - 2.0 * theta) * u_n[i] + theta * u_nm1[i])
                .collect();
            let velocity: Vec<f64> = (0..n_live)
                .map(|i| (u_np1[i] - u_nm1[i]) / (2.0 * k))
                .collect();
            let q = inst.port.t().matvec(&velocity);
            let r = inst.port.r();
            let terminal: Vec<f64> = (0..q.len()).map(|i| pbar_free[i] + r[i] * q[i]).collect();
            let back = inst.port.t().transpose().matvec(&terminal);
            let f_total: Vec<f64> = (0..n_live).map(|i| f_ext[i] - back[i]).collect();
            let accel: Vec<f64> = (0..n_live)
                .map(|i| u_np1[i] - 2.0 * u_n[i] + u_nm1[i])
                .collect();
            let s_avg = p.stiffness.matvec(&average);
            let (mass, weight): (f64, Vec<f64>) = match boundary {
                Boundary::Supported => (p.rho * p.h * p.h, vec![1.0; n_live]),
                Boundary::Free => (p.rho, p.w.clone()),
            };
            let mut worst: f64 = 0.0;
            let mut scale: f64 = 0.0;
            for i in 0..n_live {
                let res = weight[i] * accel[i]
                    + k * k * p.kappa * p.kappa * s_avg[i]
                    + 2.0 * p.sigma * k * k * weight[i] * velocity[i]
                    - k * k * f_total[i] / mass;
                worst = worst.max(res.abs());
                scale = scale.max((weight[i] * accel[i]).abs());
            }
            assert!(
                worst <= 1e-11 * scale,
                "{boundary:?} at {fs} Hz: residual {worst:.3e} against {scale:.3e}"
            );
        }
    }
}

/// `T = 0` reduces to a bare [`Plate`] **bit for bit** — the family's reduction ledger entry.
///
/// (Alongside `R = 0 ->` bare body, `M_a = inf -> RadiatedBody`, `sigma_1 = 0 ->` model #2,
/// `nonlinear = false ->` #5.) There is no natural `R = 0` reduction here — `R_j` vanishes only on
/// an open face, which is refused — so a disconnected `T` is the available one, and it is worth
/// having because the wrapper reassembles the plate's right-hand side and its `A` rather than
/// calling `Plate::step`. Bit-identity is claimable because an entry whose value is exactly zero is
/// dropped from the sum, so a zero load factors the plate's **own** matrix.
///
/// Suspended, it says one thing more, and it is a statement about what the two halves of that
/// object are: the **cut belongs to the room** and the **load belongs to the plate**, so a
/// zero-area surface still blocks the room while the plate never notices it is in one. The cut is
/// asserted to survive, or the bar would prove less than it claims.
#[test]
fn no_air_load_reproduces_the_bare_plate() {
    for mount in MOUNTS {
        for boundary in BOUNDARIES {
            let mut room = surface_room(rigid());
            let mut inst = mounted(mount, &mut room, FS, 8, boundary, 1.0);
            let mut bare = plate(FS, 8, boundary, 1.0);
            if mount == Mount::Suspended {
                assert!(
                    room.cut_faces() > 0,
                    "the cut must survive a zero-area surface, or this proves less"
                );
            }

            zero_the_load(&mut inst);

            let n_live = inst.seam.n_live();
            let force: Vec<f64> = (0..n_live).map(|i| 1e-3 * noise(1, i)).collect();
            let u0 = plate_bump(&bare.p, 1e-3);
            let zero = vec![0.0; n_live];
            bare.set_state(&u0, &zero);
            inst.set_state(&u0, &zero);
            for _ in 0..60 {
                bare.step(Some(&force));
                inst.step(&mut room, Some(&force)).expect("a legal step");
                room.step();
            }
            assert_eq!(
                inst.seam.plate.u, bare.u,
                "{mount:?}/{boundary:?}: a zero load must be BIT-identical to the bare plate"
            );
            assert_eq!(
                inst.radiated_energy(),
                0.0,
                "{mount:?}/{boundary:?}: nothing was radiated"
            );
        }
    }
}

/// Disconnect a wrapper's coupling: `T = 0`, `T^T R T = 0`, and the factorization rebuilt.
///
/// The reference did this by assigning empty SciPy matrices onto two public port slots and then
/// re-deriving `A_loaded` by hand. Here the slots are still writable — plan §31.6 found that four
/// port attributes are *written* by tests, and they kept their setters — but the re-derivation goes
/// through [`RoomGrid::refactor`], which is the wrapper's own one spelling.
fn zero_the_load(inst: &mut RoomGrid<PlateSeam>) {
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

// -- the sign convention ------------------------------------------------------------------------

/// Positive displacement is along the **inward normal**, on every one of the six faces.
///
/// Two assertions, and the first is the one no ledger sees: a port mounted on the wrong wall of a
/// symmetric room is perfectly self-consistent, so the node indices are checked against the
/// requested wall directly. Then the physics: a surface moving into the room **compresses** the air
/// at its own face, read at `n = 1` — and the read-out time matters, because a six-step read gave
/// the *wrong* answer in this batch's prototype when the plate's own half period was five steps.
#[test]
fn the_sign_convention_is_uniform_over_all_six_faces() {
    for face in FACES {
        let mut room = surface_room(rigid());
        let mut inst = baffled(&mut room, FS, 8, Boundary::Supported, 0.0, face, None);
        let axis = "xyz"
            .find(face.as_bytes()[0] as char)
            .expect("a legal face");
        let wall_index = if face.as_bytes()[1] == b'0' {
            0
        } else {
            room.p.n[axis]
        };
        match &inst.port {
            GridPort::Baffled(p) => {
                assert!(
                    p.nodes()[axis].iter().all(|&i| i == wall_index),
                    "{face}: the port is not on the requested wall"
                );
            }
            GridPort::Suspended(_) => unreachable!("this scene is baffled"),
        }

        let n_live = inst.seam.n_live();
        inst.set_state(&vec![0.0; n_live], &vec![1.0; n_live]); // uniformly into the room
        inst.step(&mut room, None).expect("a legal step");
        room.step();
        let mean = inst.pbar().iter().sum::<f64>() / inst.pbar().len() as f64;
        assert!(
            mean > 0.0,
            "{face}: moving into the room must compress the air"
        );
    }
}

/// The negative control, on a **high** face: flipping `T` is perfectly conservative and wrong.
///
/// `T^T R T` is sign-invariant, so the load matrix, the solve and `radiated_energy` come out
/// **bit-identical** while the room's field is inverted. Asserting the bit-identity of the wrong
/// run *is* the point: it records that the "positive displacement along the global axis"
/// convention — which needs an explicit per-face sign, and is wrong on three of the six faces —
/// cannot be caught by any energy report. The local inward-normal convention disarms that instead
/// of testing around it, which is why no inward normal appears in the code at all.
#[test]
fn a_sign_flip_is_invisible_to_every_energy_quantity() {
    fn go(flip: bool) -> (f64, f64) {
        let mut room = surface_room(rigid());
        let mut inst = baffled(&mut room, FS, 8, Boundary::Supported, 0.0, "z1", None);
        if flip {
            let t = inst.port.t().scaled(-1.0);
            match &mut inst.port {
                GridPort::Baffled(p) => p.set_t(t),
                GridPort::Suspended(_) => unreachable!("this scene is baffled"),
            }
            inst.refactor()
                .expect("a sign-flipped T gives the same SPD load");
        }
        let n_live = inst.seam.n_live();
        inst.set_state(&vec![0.0; n_live], &vec![1.0; n_live]);
        run(&mut inst, &mut room, 4);
        let mean = inst.pbar().iter().sum::<f64>() / inst.pbar().len() as f64;
        (inst.radiated_energy(), mean)
    }

    let (good_energy, good_pressure) = go(false);
    let (bad_energy, bad_pressure) = go(true);
    assert_eq!(
        bad_energy, good_energy,
        "a sign flip must be BIT-identical in the energy ledger"
    );
    let rel = (bad_pressure + good_pressure).abs() / good_pressure.abs();
    assert!(
        rel <= 1e-12,
        "the field must invert exactly; it is off by {rel:.3e}"
    );
}

// -- conservation (necessary, not sufficient) ---------------------------------------------------

/// `plate.energy() + radiated + room.energy()` flat to machine precision.
///
/// Necessary and not sufficient — see [`conservation_is_blind_to_a_wrong_r`] — and it still ships,
/// because it catches a genuinely broken scheme and it is the statement a reader wants. The channel
/// size is asserted alongside so the bar cannot pass on a disconnected coupling; note it is small
/// here (~0.2% of `E0`) because a narrow struck bump excites fine spatial patterns, which radiate
/// badly. That *is* the acoustic short circuit, and [`a_free_plate_piston_is_fully_radiated`] is the
/// configuration where the channel is essentially the whole energy.
#[test]
fn the_scene_total_is_flat() {
    for mount in MOUNTS {
        for boundary in BOUNDARIES {
            for (name, walls) in wall_cases() {
                let mut room = surface_room(walls);
                let mut inst = mounted(mount, &mut room, FS, 8, boundary, 0.0);
                seed_bump(&mut inst);
                let e0 = scene_energy(&[&inst], &room);
                let mut worst: f64 = 0.0;
                for _ in 0..300 {
                    inst.step(&mut room, None).expect("a legal step");
                    room.step();
                    worst = worst.max((scene_energy(&[&inst], &room) - e0).abs());
                }
                assert!(
                    worst <= DRIFT_TOL * e0.abs(),
                    "{mount:?}/{boundary:?}/{name}: drift {:.3e}",
                    worst / e0.abs()
                );
                assert!(
                    inst.radiated_energy().abs() > 1e-5 * e0.abs(),
                    "{mount:?}/{boundary:?}/{name}: the channel carried nothing"
                );
            }
        }
    }
}

/// With `sigma_plate > 0` the scene total is monotone non-increasing — passivity.
#[test]
fn a_lossy_plate_scene_is_monotone() {
    let mut room = surface_room(all_lossy());
    let mut inst = baffled(&mut room, FS, 8, Boundary::Supported, 3.0, "z0", None);
    let u0 = plate_bump(&inst.seam.plate.p, 1e-3);
    inst.set_state(&u0, &vec![0.0; u0.len()]);
    let e0 = scene_energy(&[&inst], &room);
    let mut previous = e0;
    for step in 0..300 {
        inst.step(&mut room, None).expect("a legal step");
        room.step();
        let current = scene_energy(&[&inst], &room);
        assert!(
            current <= previous + 1e-14 * e0.abs(),
            "step {step}: the total rose by {:.3e}",
            current - previous
        );
        previous = current;
    }
    assert!(previous < e0, "a lossy scene must actually lose energy");
}

/// The free plate's rigid-body nullspace is inert bare and **fully radiated** when baffled.
///
/// Model #5b's stiffness nullspace is exactly `{1, x, y}`: give the bare plate a uniform velocity
/// and nothing resists it — it translates forever at *constant* energy, which is the negative
/// control that makes the contrast mean something. Mount the same plate flush in a baffle and that
/// identical motion **is a piston**, the most efficient radiator the geometry has, so a lossy room
/// takes essentially all of it. No bare free plate and no lumped body-loss coefficient can do that.
///
/// This is also the configuration where the coupling channel is ~100% of `E0` rather than the 0.2%
/// a struck bump gives, which is what makes the conservation assertion here non-vacuous.
///
/// Deliberately **not** asserted: monotone decay in a *rigid* room. A closed box gives the piston's
/// energy back — measured, the plate drops to 4.5% and climbs back — so only the total is monotone
/// there, and asserting the drop as a decay would pass on the sampling instants and fail on the
/// physics.
#[test]
fn a_free_plate_piston_is_fully_radiated() {
    let mut bare = plate(FS, 8, Boundary::Free, 0.0);
    let n_bare = bare.p.n_live;
    bare.set_state(&vec![0.0; n_bare], &vec![1.0; n_bare]);
    let e0_bare = bare.energy();
    for _ in 0..400 {
        bare.step(None);
    }
    let rel = (bare.energy() - e0_bare).abs() / e0_bare.abs();
    assert!(
        rel <= 1e-9,
        "the bare piston must translate at constant energy, drift {rel:.3e}"
    );

    let mut room = surface_room(all_lossy());
    let mut inst = baffled(&mut room, FS, 8, Boundary::Free, 0.0, "z0", None);
    let n_live = inst.seam.n_live();
    inst.set_state(&vec![0.0; n_live], &vec![1.0; n_live]);
    let e0 = scene_energy(&[&inst], &room);
    let mut worst: f64 = 0.0;
    for _ in 0..800 {
        inst.step(&mut room, None).expect("a legal step");
        room.step();
        worst = worst.max((scene_energy(&[&inst], &room) - e0).abs());
    }
    assert!(
        inst.seam.plate.energy() < 1e-3 * e0,
        "the plate kept {:.3e} of E0",
        inst.seam.plate.energy() / e0
    );
    let channel = (inst.radiated_energy() - e0).abs() / e0.abs();
    assert!(
        channel <= 1e-2,
        "the channel IS the energy; it is off by {channel:.3e}"
    );
    assert!(worst <= 1e-10 * e0.abs(), "drift {:.3e}", worst / e0.abs());
}

// -- what the factorization costs ---------------------------------------------------------------

/// The factorization's real cost is exposed, and it is not the stored `nnz`.
///
/// The plan expected the factorization not to thicken meaningfully. It does: the load couples every
/// plate node sharing an air node, so LU fill measured at 1.55x, 3.50x and 5.29x a bare plate's at
/// `h_plate / h_air = 0.45, 0.23, 0.15`. Hence [`RoomGrid::lu_nnz`] — stored `nnz` is not what the
/// solver pays.
///
/// (The load matrix's own symmetry is `airbox_surface.rs::the_load_matrix_is_symmetric_but_not_
/// symmetrised`, which is where it belongs: it is a claim about the port.)
#[test]
fn the_factorization_cost_is_reported() {
    let mut room = surface_room(rigid());
    let inst = baffled(&mut room, FS, 16, Boundary::Supported, 0.0, "z0", None);
    let p = &inst.seam.plate.p;
    let coeff = p.theta * p.k * p.k * p.kappa * p.kappa;
    let a_bare = Csr::identity(p.n_live).add(&p.stiffness.scaled(coeff));
    let bare = SparseLu::factor(&a_bare).expect("a bare plate factors");
    let (l, u) = bare.nnz();
    assert!(
        inst.lu_nnz() > l + u,
        "the loaded factorization must cost more: {} vs {}",
        inst.lu_nnz(),
        l + u
    );
    assert!(inst.nnz_growth() > 1.0, "the load must thicken A");
}

// -- the headline -------------------------------------------------------------------------------

/// Largest `|sum_j q_j| / sum_j |q_j|` over a run — the fraction of the surface's own flow that
/// survives cancellation, i.e. all a one-port could ever couple through.
///
/// **Normalised by the run's own flow, deliberately.** The bare `|sum q| / net_area` is
/// dimensional, so its zero sits at whatever roundoff happens to be for this amplitude and this
/// cell size — it read `7e-14` on Windows and `1.6e-13` on Linux against an *absolute* `1e-13`
/// bar, which is a platform assertion wearing a physics assertion's clothes. The ratio is
/// dimensionless and caps at 1, so the zeros (`<1e-11`) and the radiating controls (`~1`) separate
/// by eight orders of magnitude and the bar can sit two decades clear of either.
///
/// It is also the stronger statement: an amplitude-free measure cannot be flattered by a run that
/// barely moves. A whole run with `denom == 0` would be exactly that degenerate pass, so it is
/// refused rather than returned as a zero.
fn peak_monopole(inst: &mut RoomGrid<PlateSeam>, room: &mut AirBox, steps: usize) -> f64 {
    let mut peak: f64 = 0.0;
    let mut moved = false;
    for _ in 0..steps {
        inst.step(room, None).expect("a legal step");
        room.step();
        let denom: f64 = inst.nodal_volume_velocity().iter().map(|v| v.abs()).sum();
        if denom > 0.0 {
            moved = true;
            peak = peak.max(inst.volume_velocity().abs() / denom);
        }
    }
    assert!(
        moved,
        "the surface never moved -- a dead coupling would pass every silence bar vacuously"
    );
    peak
}

/// **The batch's claim.** A mode with exactly zero net volume velocity radiates definitely.
///
/// For a supported plate the scheme's modes are the *exact* discrete `sin x sin`, and
/// `sum_{i=1}^{N-1} sin(m pi i / N) = 0` identically for even `m` — so an even-index mode's net
/// volume displacement is **zero, not small**, and `AirRadiation`, `RadiatedBody`,
/// `RationalAirLoad` and `RoomPort` all report exact silence. The distributed port reports a
/// definite nonzero power, because each patch of surface pushes on the air *locally* and the
/// cancellation is only as complete as the acoustic wavelength's ability to bridge a `+` region and
/// its `-` neighbour.
///
/// The zero must hold for the **whole run**, not at `t = 0`: it is the run-long assertion that
/// would catch a load matrix that mixes the odd modes back in. And the radiated energy is bounded
/// **below**, so the bar cannot pass on a disconnected coupling.
///
/// Deliberately not asserted: any *ranking* of the modes by radiated energy. A plate mode locks
/// spatial fineness to frequency, so a finer mode completes more cycles in the same window and its
/// count beats the per-cycle suppression — the finer mode radiates *more*, which is the opposite of
/// the fineness law and belongs only to a prescribed-velocity rig where frequency is a knob.
#[test]
fn an_even_mode_is_silent_to_every_one_port_and_still_radiates() {
    // x <-> y symmetric, or (2,1) and (1,2) would see two different rooms.
    let square = [12, 12, 9];
    for mode in [(2usize, 1usize), (1, 2)] {
        let mut room = make_room(FS, square, rigid());
        let mut inst = baffled(&mut room, FS, 16, Boundary::Supported, 0.0, "z0", None);
        let shape = plate_mode_shape(&inst.seam.plate.p, mode.0, mode.1);
        let u0: Vec<f64> = shape.iter().map(|v| 1e-3 * v).collect();
        inst.set_state(&u0, &vec![0.0; u0.len()]);

        let mut ref_room = make_room(FS, square, rigid());
        let mut reference = baffled(&mut ref_room, FS, 16, Boundary::Supported, 0.0, "z0", None);
        let ref_shape = plate_mode_shape(&reference.seam.plate.p, 1, 1);
        let ref_u0: Vec<f64> = ref_shape.iter().map(|v| 1e-3 * v).collect();
        reference.set_state(&ref_u0, &vec![0.0; ref_u0.len()]);

        let quiet = peak_monopole(&mut inst, &mut room, 200);
        let loud = peak_monopole(&mut reference, &mut ref_room, 200);
        assert!(quiet < 1e-9, "mode {mode:?} leaked {quiet:.3e}");
        // The (1,1) control cancels not at all -- it reads 1.0.
        assert!(
            loud > 1e-2,
            "the control must radiate as a monopole, got {loud:.3e}"
        );
        assert!(
            inst.radiated_energy() > 0.05 * reference.radiated_energy(),
            "mode {mode:?} radiated only {:.3e} of the control",
            inst.radiated_energy() / reference.radiated_energy()
        );
    }
}

/// The zero survives only in a scene that is mirror-symmetric about the mode's own plane.
///
/// Equivariance of the load is necessary and **not** sufficient: the incoming `T^T pbar_free` is
/// the *room's* field. Measured five ways, and there is **no tolerance band** — the leak is linear
/// in the offset with no threshold, so "approximately centred" is not approximately silent.
///
/// The two positive controls are not defects to hide. A room that is asymmetric about the plate
/// re-excites the plate's *shape*, converting an acoustically silent mode into a radiating one at
/// the percent level — and no `R(omega)` one-port can represent that at all, because a lumped port
/// couples through a single scalar and has no shape for the room to push on.
#[test]
fn the_silence_is_a_property_of_the_whole_scene() {
    fn leak(walls: [Wall; 6], offset: f64) -> f64 {
        let mut room = surface_room(walls);
        let origin = if offset == 0.0 {
            None
        } else {
            Some((
                0.5 * (room.p.n[0] as f64 * room.p.h - PLATE_L) + offset * room.p.h,
                0.5 * (room.p.n[1] as f64 * room.p.h - PLATE_L),
            ))
        };
        let mut inst = baffled(&mut room, FS, 16, Boundary::Supported, 0.0, "z0", origin);
        let shape = plate_mode_shape(&inst.seam.plate.p, 2, 1);
        let u0: Vec<f64> = shape.iter().map(|v| 1e-3 * v).collect();
        inst.set_state(&u0, &vec![0.0; u0.len()]);
        peak_monopole(&mut inst, &mut room, 200)
    }

    let z = impedance_from_zeta(4.0, RHO0_AIR, C0_AIR);
    let mut lossy_y0 = rigid();
    lossy_y0[2] = Wall::Impedance(z);
    let mut lossy_x0 = rigid();
    lossy_x0[0] = Wall::Impedance(z);

    // Symmetric scenes: silent to rounding.
    assert!(leak(rigid(), 0.0) < 1e-9, "a rigid room must stay silent");
    assert!(
        leak([Wall::Impedance(z); 6], 0.0) < 1e-9,
        "a uniformly lossy room too"
    );
    // Asymmetric in y only -- the (2,1) mode is antisymmetric in x, so its zero is untouched.
    assert!(
        leak(lossy_y0, 0.0) < 1e-9,
        "asymmetry off the mode's axis must not leak"
    );
    // Asymmetric in the mode's OWN axis: the room drives the plate's odd modes.
    assert!(
        leak(lossy_x0, 0.0) > 1e-3,
        "asymmetry on the mode's axis must leak"
    );
    // And off-centre by a third of an air cell, in an otherwise perfect room.
    assert!(
        leak(rigid(), 1.0 / 3.0) > 1e-3,
        "an off-centre surface must leak"
    );
}

// -- sharing a room, and the refusals -----------------------------------------------------------

/// N instruments in one room: disjoint node sets, one room step.
#[test]
fn two_disjoint_surfaces_share_one_room() {
    let mut room = make_room(FS, [12, 11, 12], rigid());
    let mut top = baffled(&mut room, FS, 8, Boundary::Supported, 0.0, "z0", None);
    let mut bottom = baffled(&mut room, FS, 8, Boundary::Supported, 0.0, "z1", None);
    let u_top = plate_bump(&top.seam.plate.p, 1e-3);
    let u_bottom = plate_bump(&bottom.seam.plate.p, 1e-3);
    top.set_state(&u_top, &vec![0.0; u_top.len()]);
    bottom.set_state(&u_bottom, &vec![0.0; u_bottom.len()]);
    let e0 = scene_energy(&[&top, &bottom], &room);
    let mut worst: f64 = 0.0;
    for _ in 0..200 {
        top.step(&mut room, None).expect("a legal step");
        bottom.step(&mut room, None).expect("a legal step");
        room.step();
        worst = worst.max((scene_energy(&[&top, &bottom], &room) - e0).abs());
    }
    assert!(
        worst <= DRIFT_TOL * e0.abs(),
        "drift {:.3e}",
        worst / e0.abs()
    );
    let radiated = top.radiated_energy() + bottom.radiated_energy();
    let gap = (radiated - room.injected).abs();
    assert!(gap <= LEDGER_TOL * radiated.abs(), "ledger gap {gap:.3e}");
}

/// A resonator and a room built at different sample rates are refused, and the message names the
/// resonator.
#[test]
fn a_sample_rate_mismatch_is_rejected() {
    let mut room = surface_room(rigid());
    let err = RoomGrid::baffled(
        seam(9000.0, 8, Boundary::Supported, 0.0),
        &mut room,
        "z0",
        None,
        Spreading::Bilinear,
    )
    .expect_err("9 kHz plate in an 8 kHz room");
    assert!(matches!(
        err,
        WrapError::RateMismatch { label: "plate", .. }
    ));
    let msg = err.to_string();
    assert!(msg.contains("sample-rate mismatch"), "{msg}");
    assert!(msg.contains("plate fs"), "{msg}");
    // A refused wrapper must leave the room exactly as it found it.
    assert!(
        room.claims.is_empty(),
        "a refused wrapper claimed a footprint"
    );
}

/// A port does not step its room — the caller does, once, after every port has solved.
#[test]
fn solving_twice_without_a_room_step_is_rejected() {
    let mut room = surface_room(rigid());
    let mut inst = baffled(&mut room, FS, 8, Boundary::Supported, 0.0, "z0", None);
    inst.step(&mut room, None).expect("the first step is legal");
    let err = inst
        .step(&mut room, None)
        .expect_err("the second must refuse");
    assert!(
        err.to_string().contains("twice within one room step"),
        "{err}"
    );
}

// -- the dipole tier: what taking the wall away changes -----------------------------------------
//
// # This tier's methodological finding CORRECTS the baffled tier's
//
// The baffled tier established that the conserved total is blind to a wrong `R_j` and promoted
// `radiated == injected` to the money test. The suspended tier has a coefficient the baffled one
// does not -- the **2** of the two loaded faces -- and the money test is blind to half of the ways
// to get it wrong. Measured in `the_coupled_residual_catches_both_wrong_2s` and
// `each_ledger_is_blind_to_a_different_wrong_2`:
//
// ```text
// error                                     residual      |rad - inj|      scene drift
// (correct)                                 ~1e-15        ~1e-15           ~1e-14
// A -- 1x in the FACTORIZATION only         ~2e-04        blind            ~2e-02
// B -- 1x CONSISTENTLY ("one face?")        ~7e-04        1.8x             blind
// ```
//
// So **the money test alone is not sufficient either**. What catches both is putting the achieved
// `u^{n+1}` back into the coupled PDE with the force computed from the **room's own post-closure
// pressure jump** -- a number the port never touched -- at **two** timesteps, because a
// wrong-but-consistent `k`-dependent factor passes at one. Asserting `load_matrix == 2 * one_sided`
// is not a test; it re-checks arithmetic the same file just wrote.

/// A free plate given a uniform velocity -- the loudest motion the geometry has.
///
/// Baffled, this is a piston in an infinite baffle and the room takes ~all of it. Suspended, the
/// *same* motion is a two-sided plate at low `ka`, which is a far weaker radiator -- so the channel
/// is large but no longer everything, and that contrast is physics.
fn piston(room: &mut AirBox) -> RoomGrid<PlateSeam> {
    let mut inst = suspended(room, FS, 8, Boundary::Free, 0.0, DIPOLE_INDEX);
    let n_live = inst.seam.n_live();
    inst.set_state(&vec![0.0; n_live], &vec![1e-3; n_live]);
    inst
}

/// Rebuild the factorization with the load block scaled by `scale` (`1.0` = as built).
///
/// Temporarily rescales the port's load matrix, refactors, and puts the original back -- so the
/// *factorization* sees `scale x` while the port's own `R` and load stay as they were. That is
/// negative control **A**, and it is the one the money test cannot see.
fn refactor_with_load_scaled(inst: &mut RoomGrid<PlateSeam>, scale: f64) {
    let original = inst.port.load_matrix().clone();
    let scaled = original.scaled(scale);
    set_load(inst, scaled);
    inst.refactor().expect("a scaled SPD load still factors");
    set_load(inst, original);
}

fn set_load(inst: &mut RoomGrid<PlateSeam>, m: Csr) {
    match &mut inst.port {
        GridPort::Baffled(p) => p.set_load_matrix(m),
        GridPort::Suspended(p) => p.set_load_matrix(m),
    }
}

/// Halve `R` *and* the load matrix, then refactor -- "I forgot the plate has two faces".
///
/// Negative control **B**, and the one the conserved total cannot see.
fn halve_the_coupling(inst: &mut RoomGrid<PlateSeam>) {
    let half_r: Vec<f64> = inst.port.r().iter().map(|&r| 0.5 * r).collect();
    let half_load = inst.port.load_matrix().scaled(0.5);
    match &mut inst.port {
        GridPort::Baffled(p) => {
            p.set_r(half_r);
            p.set_load_matrix(half_load);
        }
        GridPort::Suspended(p) => {
            p.set_r(half_r);
            p.set_load_matrix(half_load);
        }
    }
    inst.refactor().expect("a halved SPD load still factors");
}

/// Drop every cut, turning a suspended plate into the **phantom** -- see the headline bar.
///
/// Asserts the room really is uncut afterwards: if the cut ever grows a fourth piece of state this
/// helper stops uncutting, and the headline bar would silently compare the dipole to itself and
/// pass trivially.
fn uncut(room: &mut AirBox) {
    room.cuts = [Vec::new(), Vec::new(), Vec::new()];
    room.cut_records.clear();
    assert_eq!(room.cut_faces(), 0, "the room must really be uncut");
}

/// A free plate's rigid-body piston puts a large fraction of `E0` through the coupling, and it
/// balances.
///
/// Every conservation claim must report how big the channel is: a conservation bar on a channel
/// worth 1e-14 of the total passes with the coupling disconnected. A bump's channel here is ~2e-4
/// of `E0` (fine spatial patterns radiate badly -- the acoustic short circuit doing its job, and a
/// *dipole* short-circuits twice over), so this is the configuration that makes the assertion mean
/// something. Note the contrast with the baffled tier: mounted flush, this same motion loses 99.7%
/// of `E0` to a lossy room; suspended it loses about half, and half of *that* comes back.
#[test]
fn the_piston_is_the_non_vacuous_channel() {
    let mut room = surface_room(all_lossy());
    let mut inst = piston(&mut room);
    let e0 = inst.seam.plate.energy();
    run(&mut inst, &mut room, 400);
    assert!(
        inst.radiated_energy().abs() > 0.2 * e0,
        "the channel carried only {:.3e} of E0",
        inst.radiated_energy().abs() / e0
    );
    let gap = (inst.radiated_energy() - room.injected).abs();
    assert!(
        gap <= LEDGER_TOL * inst.radiated_energy().abs(),
        "ledger gap {gap:.3e}"
    );
}

/// The tier's primary guard: the achieved `u^{n+1}` back in the coupled PDE, twice over.
///
/// The force is rebuilt from the **room's own post-closure pressure jump** -- a number the port
/// never touched -- so this is not a re-check of the port's own arithmetic. Run at two timesteps
/// because a wrong-but-consistent `k`-dependent factor passes at one, with `sigma > 0`, a nonzero
/// `f_ext` and a lossy wall so nothing is invisible. Both negative controls are here because each
/// is blind to a *different* one of the two ledgers (see the section header).
#[test]
fn the_coupled_residual_catches_both_wrong_2s() {
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Control {
        Correct,
        FactorizationOnly,
        Consistent,
    }

    fn residual(boundary: Boundary, fs: f64, control: Control) -> f64 {
        let mut room = make_room(fs, N_ROOM, lossy_mounting_wall());
        let mut inst = suspended(&mut room, fs, 6, boundary, 2.0, DIPOLE_INDEX);
        match control {
            Control::Correct => {}
            Control::FactorizationOnly => refactor_with_load_scaled(&mut inst, 0.5),
            Control::Consistent => halve_the_coupling(&mut inst),
        }
        seed_bump(&mut inst);
        run(&mut inst, &mut room, 5);

        let n_live = inst.seam.n_live();
        let f_ext: Vec<f64> = (0..n_live).map(|i| 1e-3 * noise(0, i)).collect();
        let u_n = inst.seam.plate.u.clone();
        let u_nm1 = inst.seam.plate.u_prev.clone();
        let p_old = room.pressure.clone();
        inst.step(&mut room, Some(&f_ext)).expect("a legal step");
        room.step();
        let u_np1 = inst.seam.plate.u.clone();

        // The room's OWN post-closure jump, not the port's `pbar`.
        let (lo, hi) = match &inst.port {
            GridPort::Suspended(p) => (p.nodes_lo().clone(), p.nodes_hi().clone()),
            GridPort::Baffled(_) => unreachable!("this scene is suspended"),
        };
        let faces = lo[0].len();
        let jump: Vec<f64> = (0..faces)
            .map(|m| {
                let a = flat_index(&room, [lo[0][m], lo[1][m], lo[2][m]]);
                let b = flat_index(&room, [hi[0][m], hi[1][m], hi[2][m]]);
                0.5 * (room.pressure[b] + p_old[b]) - 0.5 * (room.pressure[a] + p_old[a])
            })
            .collect();

        let p = &inst.seam.plate.p;
        let (k, theta) = (p.k, p.theta);
        let average: Vec<f64> = (0..n_live)
            .map(|i| theta * u_np1[i] + (1.0 - 2.0 * theta) * u_n[i] + theta * u_nm1[i])
            .collect();
        let velocity: Vec<f64> = (0..n_live)
            .map(|i| (u_np1[i] - u_nm1[i]) / (2.0 * k))
            .collect();
        let back = inst.port.t().transpose().matvec(&jump);
        let f_total: Vec<f64> = (0..n_live).map(|i| f_ext[i] - back[i]).collect();
        let accel: Vec<f64> = (0..n_live)
            .map(|i| u_np1[i] - 2.0 * u_n[i] + u_nm1[i])
            .collect();
        let s_avg = p.stiffness.matvec(&average);
        let (mass, weight): (f64, Vec<f64>) = match boundary {
            Boundary::Supported => (p.rho * p.h * p.h, vec![1.0; n_live]),
            Boundary::Free => (p.rho, p.w.clone()),
        };
        let mut worst: f64 = 0.0;
        let mut scale: f64 = 0.0;
        for i in 0..n_live {
            let res = weight[i] * accel[i]
                + k * k * p.kappa * p.kappa * s_avg[i]
                + 2.0 * p.sigma * k * k * weight[i] * velocity[i]
                - k * k * f_total[i] / mass;
            worst = worst.max(res.abs());
            scale = scale.max((weight[i] * accel[i]).abs());
        }
        worst / scale
    }

    for boundary in BOUNDARIES {
        for fs in [8000.0, 11000.0] {
            let good = residual(boundary, fs, Control::Correct);
            assert!(
                good <= 1e-11,
                "{boundary:?} at {fs}: correct residual {good:.3e}"
            );
            let a = residual(boundary, fs, Control::FactorizationOnly);
            assert!(
                a > 1e-6,
                "{boundary:?} at {fs}: a 1x factorization passed at {a:.3e}"
            );
            let b = residual(boundary, fs, Control::Consistent);
            assert!(
                b > 1e-6,
                "{boundary:?} at {fs}: a consistent 1x passed at {b:.3e}"
            );
        }
    }
}

/// The measurement the section header rests on, asserted rather than asserted-about.
///
/// Control **A** (`1x` in the factorization only) leaves `|radiated - injected|` at *rounding*:
/// both sides use the same `R` and the same `q`, and neither knows what matrix produced that `q`.
/// Control **B** (`1x` consistently) leaves the scene total at rounding: each side's ledger
/// telescopes against whatever pressure *it* used, and the sum of two internally consistent
/// identities is conserved even when the two disagree. If a future change made either detector
/// sensitive to the other's error, this bar fails and the framing above is wrong.
#[test]
fn each_ledger_is_blind_to_a_different_wrong_2() {
    fn books(control: u8) -> (f64, f64) {
        let mut room = surface_room(lossy_mounting_wall());
        let mut inst = suspended(&mut room, FS, 6, Boundary::Supported, 0.0, DIPOLE_INDEX);
        match control {
            b'A' => refactor_with_load_scaled(&mut inst, 0.5),
            b'B' => halve_the_coupling(&mut inst),
            _ => {}
        }
        seed_bump(&mut inst);
        let e0 = scene_energy(&[&inst], &room);
        let mut worst: f64 = 0.0;
        for _ in 0..300 {
            inst.step(&mut room, None).expect("a legal step");
            room.step();
            worst = worst.max((scene_energy(&[&inst], &room) - e0).abs());
        }
        let gap = (inst.radiated_energy() - room.injected).abs() / inst.radiated_energy().abs();
        (worst / e0.abs(), gap)
    }

    let (good_drift, good_gap) = books(b'-');
    let (a_drift, a_gap) = books(b'A');
    let (b_drift, b_gap) = books(b'B');
    assert!(
        good_drift <= DRIFT_TOL && good_gap <= LEDGER_TOL,
        "{good_drift:.3e} {good_gap:.3e}"
    );
    assert!(
        a_gap <= LEDGER_TOL,
        "the money test is supposed to MISS a wrong factorization"
    );
    assert!(
        a_drift > 1e-6,
        "... and the conserved total is supposed to catch it: {a_drift:.3e}"
    );
    assert!(
        b_drift <= DRIFT_TOL,
        "the conserved total is supposed to MISS a consistent 1x"
    );
    assert!(
        b_gap > 0.5,
        "... and the money test is supposed to catch it, got {b_gap:.3e}"
    );
}

/// `d pbar / dq` measured off the room: `-R_j` low, `+R_j` high, off-diagonal *exactly* zero.
///
/// That -- the same `R_j`, opposite signs, exactly zero elsewhere -- is what the **2** in
/// `2 T^T R T` *is*, measured rather than constructed. Comparing the coupled step's `pbar` against
/// `pbar_free + R q` would be a tautology, so instead: save the room, step it once with nothing
/// injected, restore, step it again with a unit `q` on one face, and read the difference straight
/// off the room's own post-closure field.
///
/// The exact zeros are the stronger half. The room's instantaneous response over a node set is
/// diagonal -- propagation waits for the next momentum sub-step -- and that stays true *across the
/// cut*, one cell apart, which is what makes the two planes independent within a step.
#[test]
fn r_j_is_the_same_on_both_planes_with_opposite_signs() {
    for walls in [rigid(), lossy_mounting_wall()] {
        let mut room = surface_room(walls);
        let mut inst = suspended(&mut room, FS, 6, Boundary::Supported, 0.0, DIPOLE_INDEX);
        seed_bump(&mut inst);
        run(&mut inst, &mut room, 7); // a nontrivial field, so pbar_free is not trivially zero

        // The two planes concatenated, low first -- the port's own flat ordering.
        let flat: Vec<usize> = match &inst.port {
            GridPort::Suspended(p) => p.flat().to_vec(),
            GridPort::Baffled(_) => unreachable!("this scene is suspended"),
        };
        let n = inst.port.ledger_len();
        assert_eq!(flat.len(), 2 * n, "the node set is both planes, low first");

        let base = (
            room.pressure.clone(),
            room.u.clone(),
            room.u_prev.clone(),
            room.n,
        );
        let restore = |room: &mut AirBox, inst: &mut RoomGrid<PlateSeam>| {
            room.pressure = base.0.clone();
            room.u = base.1.clone();
            room.u_prev = base.2.clone();
            room.n = base.3;
            room.pending.clear();
            room.pending_ports.clear();
            inst.port.reset();
        };

        let p_before: Vec<f64> = flat.iter().map(|&i| room.pressure[i]).collect();
        room.step();
        let pbar0: Vec<f64> = flat
            .iter()
            .enumerate()
            .map(|(m, &i)| 0.5 * (room.pressure[i] + p_before[m]))
            .collect();

        let amp = 1e-4;
        let mut measured = vec![vec![0.0; n]; 2 * n];
        for j in 0..n {
            restore(&mut room, &mut inst);
            let mut q = vec![0.0; n];
            q[j] = amp;
            inst.port.inject(&mut room, &q).expect("a legal injection");
            room.step();
            for (m, &i) in flat.iter().enumerate() {
                measured[m][j] = (0.5 * (room.pressure[i] + p_before[m]) - pbar0[m]) / amp;
            }
        }
        restore(&mut room, &mut inst);

        let want = inst.port.r();
        for j in 0..n {
            let lo_rel = (measured[j][j] + want[j]).abs() / want[j].abs();
            let hi_rel = (measured[n + j][j] - want[j]).abs() / want[j].abs();
            assert!(
                lo_rel <= 1e-12,
                "face {j}: the low plane reads {lo_rel:.3e} off -R"
            );
            assert!(
                hi_rel <= 1e-12,
                "face {j}: the high plane reads {hi_rel:.3e} off +R"
            );
            for (m, row) in measured.iter().enumerate() {
                if m != j && m != n + j {
                    assert_eq!(row[j], 0.0, "cross-resistance from face {j} to node {m}");
                }
            }
        }
    }
}

/// Half of `radiated_energy`'s increments are **negative** -- which is why it is not a measure.
///
/// The same free-plate piston into the same lossy room: baffled, `radiated_energy` is essentially
/// monotone (measured 1.2% negative increments -- a drain); suspended, it is 50.2% (a reservoir).
/// The dipole's channel is dominantly the **reactive near field**, which `radiated_energy` counts
/// as though it had left. So a decay time and a radiated fraction can disagree about direction, and
/// neither ships as a radiation figure -- that needs a prescribed-velocity rig over whole cycles.
#[test]
fn the_channel_is_a_reservoir_not_a_drain() {
    fn negative_fraction(inst: &mut RoomGrid<PlateSeam>, room: &mut AirBox) -> f64 {
        let mut prev = 0.0;
        let mut neg = 0;
        for _ in 0..400 {
            inst.step(room, None).expect("a legal step");
            room.step();
            if inst.radiated_energy() < prev {
                neg += 1;
            }
            prev = inst.radiated_energy();
        }
        neg as f64 / 400.0
    }

    let mut room = surface_room(all_lossy());
    let mut inst = piston(&mut room);
    let dipole = negative_fraction(&mut inst, &mut room);

    let mut baffle_room = surface_room(all_lossy());
    let mut flush = baffled(&mut baffle_room, FS, 8, Boundary::Free, 0.0, "z0", None);
    let n_live = flush.seam.n_live();
    flush.set_state(&vec![0.0; n_live], &vec![1e-3; n_live]);
    let drain = negative_fraction(&mut flush, &mut baffle_room);

    assert!(
        drain < 0.1,
        "the baffled channel must be a drain, got {drain:.3}"
    );
    assert!(
        dipole > 0.4,
        "the suspended channel must be a reservoir, got {dipole:.3}"
    );
}

/// A plate moving along `+plane` compresses the air it moves toward -- read at step **1**.
///
/// The baffled tier recorded that a six-step read gave the *wrong* answer because the plate's own
/// half period was five steps; here the flip lands at step 8 instead, which is the point -- the
/// number of steps is not the lesson, reading step 1 is.
///
/// This is also the **only** detector of a consistent orientation flip, because `2 T^T R T` is
/// sign-invariant and every energetic quantity is blind to it.
#[test]
fn the_sign_is_readable_on_the_first_step() {
    for plane in ["x", "y", "z"] {
        for boundary in BOUNDARIES {
            let index = if plane == "z" { 4 } else { 5 };
            let mut room = surface_room(rigid());
            let mut inst = RoomGrid::suspended(
                seam(FS, 8, boundary, 0.0),
                &mut room,
                plane,
                index,
                None,
                Spreading::Bilinear,
            )
            .expect("a legal suspended plate");
            let n_live = inst.seam.n_live();
            // The rim is dead on a supported plate, so a piston is not available there.
            let v0: Vec<f64> = match boundary {
                Boundary::Supported => plate_mode_shape(&inst.seam.plate.p, 1, 1)
                    .iter()
                    .map(|v| 1e-3 * v)
                    .collect(),
                Boundary::Free => vec![1e-3; n_live],
            };
            inst.set_state(&vec![0.0; n_live], &v0);
            inst.step(&mut room, None).expect("a legal step");
            room.step();

            let (lo, hi) = match &inst.port {
                GridPort::Suspended(p) => (p.nodes_lo().clone(), p.nodes_hi().clone()),
                GridPort::Baffled(_) => unreachable!("this scene is suspended"),
            };
            let mean = |nodes: &[Vec<usize>; 3]| -> f64 {
                let n = nodes[0].len();
                (0..n)
                    .map(|m| {
                        room.pressure[flat_index(&room, [nodes[0][m], nodes[1][m], nodes[2][m]])]
                    })
                    .sum::<f64>()
                    / n as f64
            };
            assert!(
                mean(&hi) > 0.0,
                "{plane}/{boundary:?}: the high side must compress"
            );
            assert!(
                mean(&lo) < 0.0,
                "{plane}/{boundary:?}: the low side must rarefy"
            );
        }
    }
}

/// A free second oracle: with the cut on the room's own mirror plane the two sides are exact
/// negatives of each other, to the last digit.
///
/// `N_z = 9` and `index = 4` put the cut at `4.5 h` from both walls.
#[test]
fn a_mirror_symmetric_scene_gives_pbar_lo_equal_to_minus_pbar_hi() {
    let mut room = surface_room(rigid());
    let mut inst = piston(&mut room);
    let face_index = match &inst.port {
        GridPort::Suspended(p) => p.face_index(),
        GridPort::Baffled(_) => unreachable!("this scene is suspended"),
    };
    assert_eq!(
        room.p.n[2],
        2 * face_index + 1,
        "the scene must be symmetric in z"
    );
    run(&mut inst, &mut room, 1);
    let (lo, hi) = match &inst.port {
        GridPort::Suspended(p) => p.free_pressure(&room),
        GridPort::Baffled(_) => unreachable!("this scene is suspended"),
    };
    let scale = hi.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    assert!(
        scale > 0.0,
        "the read must be exercised, not trivially zero"
    );
    for m in 0..lo.len() {
        assert!(
            (lo[m] + hi[m]).abs() <= 1e-13 * scale,
            "face {m}: {} vs {}",
            lo[m],
            hi[m]
        );
    }
}

// -- the headline -------------------------------------------------------------------------------

/// Drop the cut and the coupling *diverges* from the real plate's under refinement.
///
/// The phantom -- the `-q` / `+q` pair with no obstacle -- is a legal, perfectly conservative dipole
/// **source** carrying the plate's own motion. Measured `t50` ratios phantom/dipole of 5.2 and 19.3
/// at 1x and 2x air-grid refinement (40.8 at 3x). The assertion is the *growth*, not a value.
///
/// Be exact about what that proves. A doublet at separation `h` has moment proportional to `h` by
/// construction, so of course it vanishes under refinement; this is a precise **implementation**
/// control, not a general claim about source-only tiers. It is exactly what this tier degrades to
/// if the cut is omitted or clobbered, so the divergence is what proves **the cut is load-bearing
/// and cannot be quietly dropped** -- at 3x, omitting it is a 40x error that every ledger calls
/// green.
#[test]
fn the_source_alone_converges_to_silence() {
    fn t50(refine: usize, phantom: bool) -> usize {
        const LIMIT: usize = 4000;
        let fs = FS * refine as f64;
        let n = [N_ROOM[0] * refine, N_ROOM[1] * refine, N_ROOM[2] * refine];
        let z = impedance_from_zeta(1.0, RHO0_AIR, C0_AIR);
        let mut room = make_room(fs, n, [Wall::Impedance(z); 6]);
        let mut inst = suspended(
            &mut room,
            fs,
            8,
            Boundary::Supported,
            0.0,
            DIPOLE_INDEX * refine as i64,
        );
        if phantom {
            uncut(&mut room);
        }
        let shape = plate_mode_shape(&inst.seam.plate.p, 1, 1);
        let u0: Vec<f64> = shape.iter().map(|v| 1e-3 * v).collect();
        inst.set_state(&u0, &vec![0.0; u0.len()]);
        let e0 = inst.seam.plate.energy();
        for step in 1..=LIMIT {
            inst.step(&mut room, None).expect("a legal step");
            room.step();
            if inst.seam.plate.energy() <= 0.5 * e0 {
                return step;
            }
        }
        panic!("no t50 within {LIMIT} steps");
    }

    let ratios: Vec<f64> = [1usize, 2]
        .into_iter()
        .map(|r| t50(r, true) as f64 / t50(r, false) as f64)
        .collect();
    assert!(
        ratios[0] > 2.0,
        "the phantom must already be much weaker: {ratios:?}"
    );
    assert!(
        ratios[1] > 2.0 * ratios[0],
        "and the gap must GROW with refinement: {ratios:?}"
    );
}

/// Injecting `-q` / `+q` with no cut is not an approximation of anything -- it is exactly two
/// [`AirBox::inject`] soft sources, to `0.000e+00` over 60 injections.
///
/// That matters because the phantom is the headline's control, and a control is only worth its
/// bit-identity to the tier it stands in for -- here, the lumped monopole, twice.
#[test]
fn the_phantom_is_bit_identically_two_monopoles() {
    use physsynth_core::airbox::PortInjection;

    let mut port_room = surface_room(rigid());
    let mut ref_room = surface_room(rigid());
    let h = port_room.p.h;
    let lo = flat_index(&port_room, [4, 4, 4]);
    let hi = flat_index(&port_room, [4, 4, 5]);
    for i in 0..60 {
        let q = noise(7, i);
        port_room.pending_ports.push(PortInjection {
            nodes: vec![lo],
            w: vec![-q],
            q: 1.0,
        });
        port_room.pending_ports.push(PortInjection {
            nodes: vec![hi],
            w: vec![q],
            q: 1.0,
        });
        port_room.step();
        ref_room.inject(-q, Some([4, 4, 4]));
        ref_room.inject(q, Some([4, 4, 5]));
        ref_room.step();
        assert_eq!(port_room.pressure, ref_room.pressure, "injection {i}");
    }
    assert!(h > 0.0);
}

// -- sharing a room, and what the interior port cannot reach -------------------------------------

/// N suspended instruments in one room -- and now they must have disjoint **cuts** as well.
#[test]
fn two_suspended_plates_share_one_room() {
    let mut room = make_room(FS, [12, 11, 12], rigid());
    let mut lower = suspended(&mut room, FS, 8, Boundary::Supported, 0.0, 3);
    let mut upper = suspended(&mut room, FS, 8, Boundary::Supported, 0.0, 8);
    seed_bump(&mut lower);
    seed_bump(&mut upper);
    assert_eq!(
        room.cut_faces(),
        lower.port.ledger_len() + upper.port.ledger_len(),
        "both cuts must be present and disjoint"
    );
    let e0 = scene_energy(&[&lower, &upper], &room);
    let mut worst: f64 = 0.0;
    for _ in 0..200 {
        lower.step(&mut room, None).expect("a legal step");
        upper.step(&mut room, None).expect("a legal step");
        room.step();
        worst = worst.max((scene_energy(&[&lower, &upper], &room) - e0).abs());
    }
    assert!(
        worst <= DRIFT_TOL * e0.abs(),
        "drift {:.3e}",
        worst / e0.abs()
    );
    let total = lower.radiated_energy() + upper.radiated_energy();
    assert!((total - room.injected).abs() <= LEDGER_TOL * total.abs());
}

/// No open-face refusal is needed for an interior surface, and that is provable rather than an
/// omission: the rim and index refusals keep every patch node strictly interior on all three axes.
#[test]
fn the_interior_port_can_never_touch_an_open_face() {
    let mut walls = rigid();
    walls[4] = Wall::Open;
    walls[1] = Wall::Open;
    let mut room = surface_room(walls);
    let inst = suspended(&mut room, FS, 8, Boundary::Supported, 0.0, DIPOLE_INDEX);
    let nodes = match &inst.port {
        GridPort::Suspended(p) => p.nodes().clone(),
        GridPort::Baffled(_) => unreachable!("this scene is suspended"),
    };
    for (axis, along) in nodes.iter().enumerate() {
        assert!(
            along.iter().all(|&i| i > 0 && i < room.p.n[axis]),
            "axis {axis}: a patch node reached a wall"
        );
    }
}

// -- what a shipped refactor's goldens still buy -------------------------------------------------

/// The baffled port's construction-time digests, frozen before a refactor that has since shipped.
///
/// The reference froze six cases here — each `(sum, index-weighted sum)` of a data array, plus the
/// run-end `radiated_energy` and `sum(p)` after 200 coupled steps — to guard a refactor that moved
/// the spreading operator and four refusals into a shared base so the interior port could reuse
/// them. Two thirds of that carries over and one third is spent, and the split is not arbitrary:
///
/// * **The construction digests carry over exactly.** `node_count` and `nnz(T)` are integers, and
///   the index-weighted digest `sum_i a_i * i` moves by *order unity* under any permutation of the
///   data — so it detects a reordering of `T` or of the load matrix outright. These numbers are the
///   reference's own, unchanged, which is a free cross-implementation check on top of the one-time
///   comparison: the native `T` and load matrix reproduce digests measured through SciPy.
/// * **The run-end values are spent and are NOT re-frozen.** They were already downgraded from `==`
///   to a tolerance because they differ in the last ULP between Windows and Linux, and re-recording
///   a 200-step accumulation from *this* machine would promote a pile of incidental digits to a
///   cross-machine claim (plan §15's hazard) — for a guard whose subject, the refactor, shipped
///   long ago. What they were really catching, a permutation, is what the digests above catch
///   exactly.
#[test]
fn the_baffled_port_construction_digests_are_unchanged() {
    fn digest(a: &[f64]) -> (f64, f64) {
        let sum: f64 = a.iter().sum();
        let weighted: f64 = a.iter().enumerate().map(|(i, &v)| v * (i + 1) as f64).sum();
        (sum, weighted)
    }
    fn close(got: (f64, f64), want: (f64, f64), what: &str) {
        for (g, w) in [(got.0, want.0), (got.1, want.1)] {
            assert!(
                (g - w).abs() <= 1e-12 * w.abs(),
                "{what}: {g:?} vs the reference's {w:?}"
            );
        }
    }

    struct Case {
        name: &'static str,
        boundary: Boundary,
        walls: [Wall; 6],
        face: &'static str,
        spreading: Spreading,
        origin: Option<(f64, f64)>,
        node_count: usize,
        nnz_t: usize,
        t: (f64, f64),
        load: (f64, f64),
    }

    let cases = [
        Case {
            name: "supported",
            boundary: Boundary::Supported,
            walls: rigid(),
            face: "z0",
            spreading: Spreading::Bilinear,
            origin: None,
            node_count: 20,
            nnz_t: 182,
            t: (0.06890625, 6.304921875),
            load: (11.019939263311961, 7912.316391057989),
        },
        Case {
            name: "free",
            boundary: Boundary::Free,
            walls: rigid(),
            face: "z0",
            spreading: Spreading::Bilinear,
            origin: None,
            node_count: 30,
            nnz_t: 306,
            t: (0.09, 13.814999999999998),
            load: (14.294317013847094, 17867.896267308868),
        },
        Case {
            name: "lossy",
            boundary: Boundary::Supported,
            walls: lossy_mounting_wall(),
            face: "z0",
            spreading: Spreading::Bilinear,
            origin: None,
            node_count: 20,
            nnz_t: 182,
            t: (0.06890625, 6.304921875),
            load: (9.393020405437131, 6744.18865110386),
        },
        Case {
            name: "nearest",
            boundary: Boundary::Supported,
            walls: rigid(),
            face: "z0",
            spreading: Spreading::Nearest,
            origin: None,
            node_count: 12,
            nnz_t: 49,
            t: (0.06890624999999999, 1.72265625),
            load: (15.89495519000725, 2034.5542643209278),
        },
        Case {
            name: "offcentre",
            boundary: Boundary::Supported,
            walls: rigid(),
            face: "z0",
            spreading: Spreading::Bilinear,
            origin: Some((0.11, 0.13)),
            node_count: 20,
            nnz_t: 196,
            t: (0.06890625, 6.549605168643696),
            load: (10.931278651909219, 6708.01901861878),
        },
        Case {
            name: "y1face",
            boundary: Boundary::Free,
            walls: rigid(),
            face: "y1",
            spreading: Spreading::Bilinear,
            origin: None,
            node_count: 30,
            nnz_t: 306,
            t: (0.09, 13.815000000000001),
            load: (14.294317013847095, 17867.89626730887),
        },
    ];

    for case in cases {
        let mut room = surface_room(case.walls);
        let inst = RoomGrid::baffled(
            seam(FS, 8, case.boundary, 0.0),
            &mut room,
            case.face,
            case.origin,
            case.spreading,
        )
        .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        assert_eq!(
            inst.port.ledger_len(),
            case.node_count,
            "{}: node count",
            case.name
        );
        assert_eq!(inst.port.t().nnz(), case.nnz_t, "{}: nnz(T)", case.name);
        close(digest(inst.port.t().data()), case.t, case.name);
        close(digest(inst.port.load_matrix().data()), case.load, case.name);
    }
}

// -- the drumhead in the room --------------------------------------------------------------------
//
// The two tiers above gave the room a *plate* — mounted flush in a wall (a source) and hanging in
// the air (an object). This hangs a **membrane** there instead, and the physics that changes is not
// a detail.
//
// **A membrane has no coincidence frequency.** Kirchhoff bending gives `c_b(omega) = sqrt(kappa
// omega)`, unbounded, so every plate in this repo *crosses* `c0` at one frequency: poor radiator
// below, good above. A membrane's wave speed is the constant `c = sqrt(T / rho)` with no `omega` in
// it. So the surface is subsonic at **every** mode or supersonic at every mode, the control is the
// single number `c / c0` that a player sets by tightening the head, and a real drumhead sits well
// below it (`c / c0 ~ 0.31` for Mylar) — which is why a head with no shell is quiet.
//
// # What is new here, and is therefore pinned by its own oracle
//
// * `Membrane::step` takes no force, so this tier's `f_ext` path has nothing in the model to be
//   bit-identical *to*. `the_f_ext_term_is_pinned_twice` checks the coefficient exactly (one step
//   from rest) and its sign and operator physically (static deflection).
// * Model #4 was **explicit** — one matvec, no solve. The load's unknown is `u^{n+1}`, so it goes
//   into `A` and the membrane acquires a factorization it never had. That buys passivity by
//   construction, and `the_lagged_explicit_load_is_caught_only_by_the_total` is the measured
//   negative control for the alternative — which turned up **a third detector**. The baffled tier's
//   blind spot was the conserved total, the suspended tier's was the money test; the lagged load is
//   caught by the total (3.8e-2 of `E0`) and is *invisible* to `radiated == injected` (1.6e-16),
//   because that identity is a property of the port relation alone and cannot see which velocity
//   produced the `q`. **Three tiers, three different detectors: the lesson is not which test is the
//   money test, it is that no single one of them is.**
// * The **round** head is the interesting one, and the port refused every disk until this tier. Its
//   two staircases — the clamped rim on the membrane grid, the footprint on the air grid — are why
//   every disk assertion here is a ratio or a rate and never a magnitude.

const MEM_FS: f64 = 40_000.0;
/// The room, in **cells**: `h_air = 16.5 mm`.
const MEM_ROOM_N: [usize; 3] = [28, 28, 9];
const MEM_INDEX: i64 = 4;
/// Square head (m).
const MEM_L: f64 = 0.20;
/// Round head (m).
const MEM_RADIUS: f64 = 0.12;
/// A Mylar drumhead (kg/m^2).
const MEM_RHO: f64 = 0.26;
/// `c = 107.4 m/s`, i.e. `c / c0 = 0.31` — **subsonic**, as a real head is.
const MEM_T: f64 = 3000.0;

const DOMAINS: [membrane::Domain; 2] = [membrane::Domain::Rectangle, membrane::Domain::Circle];

fn membrane_room(walls: [Wall; 6]) -> AirBox {
    make_room(MEM_FS, MEM_ROOM_N, walls)
}

fn air_membrane(domain: membrane::Domain, n: i64, sigma: f64, fs: f64, l: f64) -> Membrane {
    let (lx, ly, radius) = match domain {
        membrane::Domain::Rectangle => (Some(l), Some(l), None),
        membrane::Domain::Circle => (None, None, Some(MEM_RADIUS)),
    };
    Membrane::new(
        membrane::Params::new(Some(domain), MEM_T, MEM_RHO, fs, n, lx, ly, radius, sigma)
            .expect("a legal membrane"),
    )
}

fn head_seam(domain: membrane::Domain, n: i64, sigma: f64, fs: f64) -> MembraneSeam {
    MembraneSeam::new(air_membrane(domain, n, sigma, fs, MEM_L))
}

fn head(
    mount: Mount,
    room: &mut AirBox,
    domain: membrane::Domain,
    n: i64,
    sigma: f64,
    fs: f64,
) -> RoomGrid<MembraneSeam> {
    let seam = head_seam(domain, n, sigma, fs);
    match mount {
        Mount::Baffled => {
            RoomGrid::baffled(seam, room, "z0", None, Spreading::Bilinear).expect("a legal head")
        }
        Mount::Suspended => {
            RoomGrid::suspended(seam, room, "z", MEM_INDEX, None, Spreading::Bilinear)
                .expect("a legal head")
        }
    }
}

/// The live-node coordinates of a membrane, in its own order.
fn head_coords(m: &Membrane) -> Vec<[f64; 2]> {
    let p = m.params();
    let mut out = Vec::with_capacity(p.n_live());
    for j in 0..p.mask.nrows() {
        for i in 0..p.mask.ncols() {
            if p.mask.at(j, i) {
                out.push([p.x[j * p.mask.ncols() + i], p.y[j * p.mask.ncols() + i]]);
            }
        }
    }
    out
}

/// A narrow off-centre strike on the live nodes — the FINE spatial pattern.
///
/// The contrast partner of [`membrane_bulge`]. A membrane has no rigid-body nullspace (the rim is
/// clamped), so unlike [`plate_bump`] there is no mean to remove — and **no piston configuration
/// exists at all**, which is why the free plate's 0.997 channel has no analogue here.
fn membrane_bump(m: &Membrane, amplitude: f64) -> Vec<f64> {
    let c = head_coords(m);
    let (xmin, xmax) = c
        .iter()
        .fold((f64::MAX, f64::MIN), |a, p| (a.0.min(p[0]), a.1.max(p[0])));
    let (ymin, ymax) = c
        .iter()
        .fold((f64::MAX, f64::MIN), |a, p| (a.0.min(p[1]), a.1.max(p[1])));
    let span = xmax - xmin;
    let width = 0.10 * span;
    let x0 = xmin + 0.42 * span;
    let y0 = ymin + 0.38 * (ymax - ymin);
    c.iter()
        .map(|p| {
            let (dx, dy) = (p[0] - x0, p[1] - y0);
            amplitude * (-((dx * dx + dy * dy) / (width * width))).exp()
        })
        .collect()
}

/// The single-signed fundamental bulge — the configuration that makes conservation non-vacuous.
///
/// Maximum net volume velocity for the geometry, so the coupling channel is a large fraction of
/// `E0` (measured 0.84 at `c / c0 = 0.31`, against 0.23 for [`membrane_bump`] — the acoustic short
/// circuit, visible in the channel itself). Sine product on a rectangle, the smooth radial analogue
/// on a disk.
fn membrane_bulge(m: &Membrane, amplitude: f64) -> Vec<f64> {
    let p = m.params();
    head_coords(m)
        .iter()
        .map(|c| match p.domain {
            membrane::Domain::Rectangle => {
                let (lx, ly) = (p.lx.expect("a rectangle"), p.ly.expect("a rectangle"));
                amplitude
                    * (std::f64::consts::PI * c[0] / lx).sin()
                    * (std::f64::consts::PI * c[1] / ly).sin()
            }
            membrane::Domain::Circle => {
                let r = (c[0] * c[0] + c[1] * c[1]).sqrt() / p.radius.expect("a disk");
                amplitude * (0.5 * std::f64::consts::PI * r.clamp(0.0, 1.0)).cos()
            }
        })
        .collect()
}

/// Zero the port's coupling in place — the bare membrane, still inside the wrapper.
///
/// Not the same as building a bare [`Membrane`]: this keeps every line of [`RoomGrid::step`] in
/// play and removes only the room, which is what makes it a usable rig for the `f_ext` oracle (the
/// room has no DC response to compare against).
fn unload(inst: &mut RoomGrid<MembraneSeam>) {
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
    inst.refactor()
        .expect("the head's own (1 + sigma k) I factors");
}

/// `T = 0` must give **bit-identical** state to a bare [`Membrane`], not merely close.
///
/// This is the zero-dropping path: with zero areas the load block is structurally present and
/// numerically zero, so the factorization must come out as the membrane's own `(1 + sigma k) I` and
/// the solve must reproduce the explicit division exactly. It is what makes the whole tier
/// falsifiable at one end — every other assertion here is to a tolerance.
#[test]
fn zero_area_reduces_to_the_bare_membrane() {
    for mount in MOUNTS {
        for domain in DOMAINS {
            let mut room = membrane_room(rigid());
            let mut inst = head(mount, &mut room, domain, 16, 3.0, MEM_FS);
            let n_live = inst.seam.n_live();
            let zero_areas = vec![0.0; n_live];
            match &mut inst.port {
                GridPort::Baffled(p) => p.set_areas(zero_areas),
                GridPort::Suspended(p) => p.set_areas(zero_areas),
            }
            unload(&mut inst);

            let mut bare = air_membrane(domain, 16, 3.0, MEM_FS, MEM_L);
            let u0 = membrane_bump(&bare, 1e-3);
            let v0 = vec![0.0; u0.len()];
            inst.set_state(&u0, &v0);
            bare.set_state(&u0, &v0);
            for _ in 0..50 {
                inst.step(&mut room, None).expect("a legal step");
                room.step();
                bare.step();
            }
            assert_eq!(
                inst.seam.membrane.u, bare.u,
                "{mount:?}/{domain:?}: the zero load must be bit-identical, not close"
            );
            assert_eq!(
                inst.seam.membrane.u_prev, bare.u_prev,
                "{mount:?}/{domain:?}"
            );
            assert_eq!(inst.radiated_energy(), 0.0, "{mount:?}/{domain:?}");
        }
    }
}

/// `radiated_energy == room.injected`, reported **with the channel size**.
///
/// A conservation bar on a channel worth 1e-14 of the total passes with the coupling disconnected,
/// so the channel is asserted too. The plate tiers named the free plate's piston as their
/// non-vacuous configuration; **a membrane has no piston** — the rim is clamped, so there is no
/// rigid-body nullspace at all — and the configuration that had to be found instead is the
/// single-signed fundamental bulge. Measured 0.14 … 0.82 of `E0` across these cases, at the drum's
/// *actual* subsonic operating point and needing neither a fast head nor a lossy room.
#[test]
fn the_heads_two_ledgers_agree_and_the_channel_is_not_vacuous() {
    let mut ran = 0;
    for mount in MOUNTS {
        for domain in DOMAINS {
            for (name, walls) in wall_cases() {
                let mut room = membrane_room(walls);
                let mut inst = head(mount, &mut room, domain, 16, 0.0, MEM_FS);
                let u0 = membrane_bulge(&inst.seam.membrane, 1e-3);
                inst.set_state(&u0, &vec![0.0; u0.len()]);
                let e0 = scene_energy(&[&inst], &room);
                run(&mut inst, &mut room, 300);
                let gap = (inst.radiated_energy() - room.injected).abs();
                assert!(
                    gap <= LEDGER_TOL * inst.radiated_energy().abs(),
                    "{mount:?}/{domain:?}/{name}: ledger gap {gap:.3e}"
                );
                assert!(
                    inst.radiated_energy().abs() > 1e-2 * e0,
                    "{mount:?}/{domain:?}/{name}: the channel must be worth asserting on, it is {:.3e}",
                    inst.radiated_energy().abs() / e0
                );
                ran += 1;
            }
        }
    }
    assert_eq!(
        ran, 12,
        "every tier x domain x wall combination must have run"
    );
}

/// The head's scene total — necessary and not sufficient, and *named* so.
#[test]
fn the_heads_scene_total_is_flat() {
    for mount in MOUNTS {
        for domain in DOMAINS {
            let mut room = membrane_room(all_lossy());
            let mut inst = head(mount, &mut room, domain, 16, 0.0, MEM_FS);
            let u0 = membrane_bulge(&inst.seam.membrane, 1e-3);
            inst.set_state(&u0, &vec![0.0; u0.len()]);
            let e0 = scene_energy(&[&inst], &room);
            let mut worst: f64 = 0.0;
            for _ in 0..300 {
                inst.step(&mut room, None).expect("a legal step");
                room.step();
                worst = worst.max((scene_energy(&[&inst], &room) - e0).abs());
            }
            assert!(
                worst <= DRIFT_TOL * e0.abs(),
                "{mount:?}/{domain:?}: drift {:.3e}",
                worst / e0.abs()
            );
        }
    }
}

/// The fine spatial pattern radiates worse than the smooth one — in the channel itself.
///
/// The same membrane, the same energy, two shapes: the single-signed bulge (large net volume
/// velocity) against a narrow strike (fine pattern, cancelling on the scale of the acoustic
/// wavelength). This is the short circuit measured *without* a mode decomposition, and it is the
/// reason the bulge is the configuration the conservation bars use.
#[test]
fn the_channel_shows_the_acoustic_short_circuit() {
    for mount in MOUNTS {
        let mut channels = [0.0f64; 2];
        for (which, shape) in [membrane_bulge, membrane_bump].into_iter().enumerate() {
            let mut room = membrane_room(all_lossy());
            let mut inst = head(
                mount,
                &mut room,
                membrane::Domain::Rectangle,
                16,
                0.0,
                MEM_FS,
            );
            let u0 = shape(&inst.seam.membrane, 1e-3);
            inst.set_state(&u0, &vec![0.0; u0.len()]);
            let e0 = scene_energy(&[&inst], &room);
            run(&mut inst, &mut room, 300);
            channels[which] = inst.radiated_energy().abs() / e0;
        }
        assert!(
            channels[0] > 2.0 * channels[1],
            "{mount:?}: bulge {:.3e} vs bump {:.3e}",
            channels[0],
            channels[1]
        );
    }
}

/// The suspended tier's guard, applied to the doubled membrane load.
///
/// Inherited because it is the only one that catches both ways of getting the two-loaded-faces
/// factor wrong: control **A** (`1x` in the factorization only) is invisible to
/// `radiated == injected`, control **B** (`1x` consistently) is invisible to the scene total. Two
/// timesteps, because a wrong-but-consistent `k`-dependent factor passes at one.
#[test]
fn the_heads_coupled_residual_catches_both_wrong_2s() {
    fn residual(fs: f64, control: u8) -> f64 {
        // The room takes the head's rate, not the fixture's: this bar varies `fs` to catch a
        // wrong-but-consistent `k`-dependent factor, and both sides share one timestep.
        let mut room = make_room(fs, MEM_ROOM_N, lossy_mounting_wall());
        let mut inst = RoomGrid::suspended(
            head_seam(membrane::Domain::Rectangle, 12, 2.0, fs),
            &mut room,
            "z",
            MEM_INDEX,
            None,
            Spreading::Bilinear,
        )
        .expect("a legal head");
        match control {
            b'A' => {
                let original = inst.port.load_matrix().clone();
                let scaled = original.scaled(0.5);
                match &mut inst.port {
                    GridPort::Baffled(p) => p.set_load_matrix(scaled),
                    GridPort::Suspended(p) => p.set_load_matrix(scaled),
                }
                inst.refactor().expect("a scaled SPD load still factors");
                match &mut inst.port {
                    GridPort::Baffled(p) => p.set_load_matrix(original),
                    GridPort::Suspended(p) => p.set_load_matrix(original),
                }
            }
            b'B' => {
                let half_r: Vec<f64> = inst.port.r().iter().map(|&r| 0.5 * r).collect();
                let half_load = inst.port.load_matrix().scaled(0.5);
                match &mut inst.port {
                    GridPort::Baffled(p) => {
                        p.set_r(half_r);
                        p.set_load_matrix(half_load);
                    }
                    GridPort::Suspended(p) => {
                        p.set_r(half_r);
                        p.set_load_matrix(half_load);
                    }
                }
                inst.refactor().expect("a halved SPD load still factors");
            }
            _ => {}
        }
        let u0 = membrane_bump(&inst.seam.membrane, 1e-3);
        inst.set_state(&u0, &vec![0.0; u0.len()]);
        run(&mut inst, &mut room, 5);

        let n_live = inst.seam.n_live();
        let f_ext: Vec<f64> = (0..n_live).map(|i| 1e-3 * noise(0, i)).collect();
        let u_n = inst.seam.membrane.u.clone();
        let u_nm1 = inst.seam.membrane.u_prev.clone();
        let p_old = room.pressure.clone();
        inst.step(&mut room, Some(&f_ext)).expect("a legal step");
        room.step();
        let u_np1 = inst.seam.membrane.u.clone();

        let (lo, hi) = match &inst.port {
            GridPort::Suspended(p) => (p.nodes_lo().clone(), p.nodes_hi().clone()),
            GridPort::Baffled(_) => unreachable!("this scene is suspended"),
        };
        let jump: Vec<f64> = (0..lo[0].len())
            .map(|m| {
                let a = flat_index(&room, [lo[0][m], lo[1][m], lo[2][m]]);
                let b = flat_index(&room, [hi[0][m], hi[1][m], hi[2][m]]);
                0.5 * (room.pressure[b] + p_old[b]) - 0.5 * (room.pressure[a] + p_old[a])
            })
            .collect();

        let p = inst.seam.membrane.params();
        let k = p.k;
        let velocity: Vec<f64> = (0..n_live)
            .map(|i| (u_np1[i] - u_nm1[i]) / (2.0 * k))
            .collect();
        let back = inst.port.t().transpose().matvec(&jump);
        let lu_n = p.l.matvec(&u_n);
        let mut worst: f64 = 0.0;
        let mut scale: f64 = 0.0;
        for i in 0..n_live {
            let accel = u_np1[i] - 2.0 * u_n[i] + u_nm1[i];
            let f_total = f_ext[i] - back[i];
            let res = accel - k * k * p.c * p.c * lu_n[i] + 2.0 * p.sigma * k * k * velocity[i]
                - k * k * f_total / (p.rho * p.h * p.h);
            worst = worst.max(res.abs());
            scale = scale.max(accel.abs());
        }
        worst / scale
    }

    for fs in [MEM_FS, 0.75 * MEM_FS] {
        let good = residual(fs, b'-');
        assert!(good <= 1e-11, "at {fs}: correct residual {good:.3e}");
        let a = residual(fs, b'A');
        assert!(a > 1e-6, "at {fs}: a 1x factorization passed at {a:.3e}");
        let b = residual(fs, b'B');
        assert!(b > 1e-6, "at {fs}: a consistent 1x passed at {b:.3e}");
    }
}

/// Every channel dissipative -> the scene total decreases at every step, never rises.
#[test]
fn a_lossy_head_in_a_lossy_room_is_monotone() {
    for mount in MOUNTS {
        let mut room = membrane_room(all_lossy());
        let mut inst = head(
            mount,
            &mut room,
            membrane::Domain::Rectangle,
            16,
            40.0,
            MEM_FS,
        );
        let u0 = membrane_bulge(&inst.seam.membrane, 1e-3);
        inst.set_state(&u0, &vec![0.0; u0.len()]);
        let mut previous = scene_energy(&[&inst], &room);
        for step in 0..300 {
            inst.step(&mut room, None).expect("a legal step");
            room.step();
            let now = scene_energy(&[&inst], &room);
            assert!(
                now <= previous + 1e-14 * previous.abs(),
                "{mount:?} step {step}: the total rose by {:.3e}",
                now - previous
            );
            previous = now;
        }
    }
}

/// `R_j` measured off the room per node, not trusted from the assembly line.
///
/// Inject a unit volume velocity at one port node into an otherwise silent room and read the
/// pressure the room itself produces. `pbar = pbar_free + R q` is the Thevenin relation the whole
/// coupling rests on; this is the only bar that checks the `R` in it against the object it claims
/// to describe.
#[test]
fn r_is_the_rooms_own_differential_response() {
    for mount in MOUNTS {
        let mut room = membrane_room(lossy_mounting_wall());
        let mut inst = head(
            mount,
            &mut room,
            membrane::Domain::Rectangle,
            12,
            0.0,
            MEM_FS,
        );
        let n = inst.port.ledger_len();
        let mut q = vec![0.0; n];
        let j = n / 2;
        q[j] = 1.0;
        let before = room.pressure.clone();
        inst.port.inject(&mut room, &q).expect("a legal injection");
        room.step();
        let pbar = |i: usize| 0.5 * (room.pressure[i] + before[i]);

        match &inst.port {
            GridPort::Baffled(p) => {
                let flat = p.flat();
                let response = pbar(flat[j]);
                let expected = p.r()[j] * q[j];
                let rel = (response - expected).abs() / expected.abs();
                assert!(rel <= 1e-12, "{mount:?}: R off by {rel:.3e}");
            }
            GridPort::Suspended(p) => {
                let flat = p.flat();
                let response = pbar(flat[n + j]) - pbar(flat[j]);
                let expected = 2.0 * p.r()[j] * q[j];
                let rel = (response - expected).abs() / expected.abs();
                assert!(rel <= 1e-12, "{mount:?}: 2R off by {rel:.3e}");
            }
        }
    }
}

/// `Membrane::step` takes no force, so this term has nothing to be bit-identical to.
///
/// The plate tiers could copy `Plate`'s own `f_ext` path line for line and a slip would show up
/// against the model. Here it is *new* arithmetic, and the energy ledger would stay green with the
/// coefficient wrong (it telescopes against whatever force was used). So it is pinned twice and
/// neither is a conservation check:
///
/// 1. **The coefficient, exactly.** One step from rest with the load removed must give exactly
///    `k^2 f / (rho h^2) / (1 + sigma k)`. The load has to be removed and that is not a
///    simplification: even from rest the first step's centred velocity `(u^1 - u^{-1}) / 2k` is
///    nonzero, so the room loads the head immediately — measured, that alone moves the answer by
///    0.96%, which is exactly the size of thing an "approximately equal" version of this bar would
///    have waved through.
/// 2. **The sign and the operator, physically.** Held down by a constant force with the room
///    removed, the head must settle on the static deflection `u_ss = -L^-1 f / (T h^2)`, which is
///    the continuum `-T grad^2 u = f` discretized — a different statement from (1), and the one
///    that would catch a sign or a `c^2` in the wrong place.
#[test]
fn the_f_ext_term_is_pinned_twice() {
    let mut room = membrane_room(rigid());
    let mut inst = head(
        Mount::Baffled,
        &mut room,
        membrane::Domain::Rectangle,
        12,
        5.0,
        MEM_FS,
    );
    unload(&mut inst);
    let coords = head_coords(&inst.seam.membrane);
    let f: Vec<f64> = coords
        .iter()
        .map(|c| 7.0 * (11.0 * c[0]).cos() * (5.0 * c[1]).sin())
        .collect();
    let n_live = inst.seam.n_live();
    inst.set_state(&vec![0.0; n_live], &vec![0.0; n_live]);
    assert!(
        room.pressure.iter().all(|&v| v == 0.0),
        "the exact step needs a silent room"
    );
    inst.step(&mut room, Some(&f)).expect("a legal step");
    let p = inst.seam.membrane.params().clone();
    let expected: Vec<f64> = f
        .iter()
        .map(|&v| p.k * p.k * v / (p.rho * p.h * p.h) / (1.0 + p.sigma * p.k))
        .collect();
    assert_eq!(
        inst.seam.membrane.u, expected,
        "the f_ext coefficient must be exact, not close"
    );

    let mut settled_room = membrane_room(rigid());
    let mut settled = head(
        Mount::Baffled,
        &mut settled_room,
        membrane::Domain::Rectangle,
        12,
        4000.0,
        MEM_FS,
    );
    unload(&mut settled);
    let n_live = settled.seam.n_live();
    let f = vec![2.0; n_live];
    settled.set_state(&vec![0.0; n_live], &vec![0.0; n_live]);
    for _ in 0..8000 {
        settled
            .step(&mut settled_room, Some(&f))
            .expect("a legal step");
        settled_room.step();
    }
    let p = settled.seam.membrane.params();
    let solved = SparseLu::factor(&p.l)
        .expect("the masked Dirichlet Laplacian is nonsingular")
        .solve(&f)
        .expect("a legal solve");
    let static_deflection: Vec<f64> = solved.iter().map(|&v| -v / (p.t * p.h * p.h)).collect();
    let scale = static_deflection.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    for (i, &want) in static_deflection.iter().enumerate() {
        let got = settled.seam.membrane.u[i];
        assert!(
            (got - want).abs() <= 2e-6 * scale,
            "node {i}: settled {got:?} against static {want:?}"
        );
    }
}

/// Keeping model #4 explicit costs conservation — and **the money test does not notice**.
///
/// Evaluating the load velocity at the *backward* difference `(u^n - u^{n-1}) / k` leaves the
/// membrane explicit (no factorization, one matvec, its old character intact) and looks entirely
/// reasonable. What it loses is passivity by construction: the load force no longer pairs with the
/// centred velocity the membrane's energy identity uses, so the head's own ledger stops
/// telescoping. Measured over 300 steps: the scene total drifts **3.8e-2 of `E0`**, against 8.6e-15
/// for the shipped scheme.
///
/// **And `radiated == injected` stays at rounding — 1.6e-16 — through all of it.** That inverts the
/// two plate tiers, where the conserved total was the blind one and the cross-ledger identity was
/// the money test. It is blind here for a reason worth stating: `radiated == injected` is a
/// property of the *port relation* alone (the room receives exactly the `q` it was handed, at
/// exactly the pressure `pbar_free + R q` it then has), so it cannot see which velocity produced
/// that `q`.
///
/// `Spreading::Nearest` is the precedent: ship exactly one measured negative control, and do not
/// offer it as a configuration.
#[test]
fn the_lagged_explicit_load_is_caught_only_by_the_total() {
    fn measure(lagged: bool) -> (f64, f64, f64) {
        let mut room = membrane_room(all_lossy());
        let mut inst = head(
            Mount::Baffled,
            &mut room,
            membrane::Domain::Rectangle,
            16,
            0.0,
            MEM_FS,
        );
        let u0 = membrane_bulge(&inst.seam.membrane, 1e-3);
        inst.set_state(&u0, &vec![0.0; u0.len()]);
        // The UNLOADED factorization, which is all an explicit scheme would ever have built.
        let lu = SparseLu::factor(&inst.seam.a_bare()).expect("(1 + sigma k) I factors");
        let e0 = scene_energy(&[&inst], &room);
        let mut worst: f64 = 0.0;
        for _ in 0..300 {
            if !lagged {
                inst.step(&mut room, None).expect("a legal step");
            } else {
                inst.port.require_ready(&room).expect("the port is ready");
                let pbar_free = inst.port.free_pressure(&room);
                let k = inst.k();
                // BACKWARD, not centred: the whole change.
                let u = inst.seam.membrane.u.clone();
                let u_prev = inst.seam.membrane.u_prev.clone();
                let v_lagged: Vec<f64> = (0..u.len()).map(|i| (u[i] - u_prev[i]) / k).collect();
                let q = inst.port.t().matvec(&v_lagged);
                let r = inst.port.r();
                let pbar: Vec<f64> = (0..q.len()).map(|i| pbar_free[i] + r[i] * q[i]).collect();
                let base = inst.seam.rhs(None);
                let back = inst.port.t().transpose().matvec(&pbar);
                let den = inst.denominator();
                let rhs: Vec<f64> = (0..base.len())
                    .map(|i| base[i] - k * k * back[i] / den)
                    .collect();
                inst.seam.commit(lu.solve(&rhs).expect("a legal solve"));
                inst.port.inject(&mut room, &q).expect("a legal injection");
                let booked = inst.radiated_energy() + k * plate::dot(&pbar, &q);
                inst.set_radiated_energy(booked);
                inst.set_n(inst.n() + 1);
            }
            room.step();
            worst = worst.max((scene_energy(&[&inst], &room) - e0).abs());
        }
        let gap = (inst.radiated_energy() - room.injected).abs() / inst.radiated_energy().abs();
        (
            worst / e0.abs(),
            gap,
            inst.radiated_energy().abs() / e0.abs(),
        )
    }

    let (shipped_drift, shipped_gap, shipped_channel) = measure(false);
    let (lagged_drift, lagged_gap, lagged_channel) = measure(true);
    assert!(shipped_drift <= DRIFT_TOL && shipped_gap <= LEDGER_TOL);
    assert!(
        lagged_drift > 1e-3,
        "the lagged load must cost real energy, got {lagged_drift:.3e}"
    );
    assert!(lagged_drift > 1e9 * shipped_drift);
    // The point of the bar: the cross-ledger identity is BLIND to it.
    assert!(
        lagged_gap <= LEDGER_TOL,
        "expected the money test to stay green, got {lagged_gap:.3e}"
    );
    // ... and not because nothing happened -- the channel is the same size either way.
    assert!(lagged_channel > 0.5 * shipped_channel);
}

// -- geometry: the round head, and the two areas that are not the same number --------------------

/// The obstacle is the *air* footprint; the moving surface is the *live* nodes. Not equal.
///
/// The suspended plate tier measured that a surface's radiated magnitude tracks `blocked_area`
/// rather than the air spacing, so an area quietly taken as `pi R^2` would give a plausible, wrong
/// and green-ledgered result. Both numbers are asserted to exist, to disagree, and to bracket the
/// nominal area from opposite sides — the clamped rim makes the mover *smaller* than nominal and
/// the staircased footprint makes the obstacle *larger*.
#[test]
fn the_cut_follows_the_port_and_the_two_areas_differ() {
    for domain in DOMAINS {
        let mut room = membrane_room(rigid());
        let inst = head(Mount::Suspended, &mut room, domain, 16, 0.0, MEM_FS);
        let p = inst.seam.membrane.params().clone();
        let (net_area, blocked, faces) = match &inst.port {
            GridPort::Suspended(port) => {
                (port.net_area(), port.blocked_area(&room), port.face_count())
            }
            GridPort::Baffled(_) => unreachable!("this scene is suspended"),
        };
        assert_eq!(
            faces,
            room.cut_faces(),
            "{domain:?}: the cut must follow the port"
        );
        let nominal = match domain {
            membrane::Domain::Circle => {
                std::f64::consts::PI * p.radius.expect("a disk") * p.radius.expect("a disk")
            }
            membrane::Domain::Rectangle => p.lx.expect("a rectangle") * p.ly.expect("a rectangle"),
        };
        let mover = p.n_live() as f64 * p.h * p.h;
        assert!(
            (net_area - mover).abs() <= 1e-12 * mover,
            "{domain:?}: net_area is the live nodes' own area"
        );
        assert!(
            net_area < nominal && nominal < blocked,
            "{domain:?}: {net_area} < {nominal} < {blocked}"
        );
        assert!(
            blocked > 1.05 * net_area,
            "{domain:?}: the two areas must really differ"
        );
    }
}

/// The first surface for which `origin = None` centres something **round** — re-measured, not
/// assumed.
///
/// A centred surface in a mirror-symmetric room keeps the reflection equivariance the spreading
/// operator's symmetry argument rests on, so the load matrix must be invariant under the in-plane
/// 180-degree rotation that maps the disk to itself.
#[test]
fn the_default_origin_centres_a_disk() {
    let mut room = membrane_room(rigid());
    let inst = head(
        Mount::Suspended,
        &mut room,
        membrane::Domain::Circle,
        16,
        0.0,
        MEM_FS,
    );
    let coords = match &inst.port {
        GridPort::Suspended(p) => p.coords().to_vec(),
        GridPort::Baffled(_) => unreachable!("this scene is suspended"),
    };
    let n = coords.len();
    let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
    for c in &coords {
        for d in 0..2 {
            lo[d] = lo[d].min(c[d]);
            hi[d] = hi[d].max(c[d]);
        }
    }
    let centre = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])];

    // The permutation that maps each node to its 180-degree image, matched on rounded coordinates.
    let key = |c: [f64; 2]| ((c[0] * 1e9).round() as i64, (c[1] * 1e9).round() as i64);
    let mut index = std::collections::HashMap::new();
    for (i, c) in coords.iter().enumerate() {
        index.insert(key(*c), i);
    }
    let perm: Vec<usize> = coords
        .iter()
        .map(|c| {
            let image = [2.0 * centre[0] - c[0], 2.0 * centre[1] - c[1]];
            *index
                .get(&key(image))
                .unwrap_or_else(|| panic!("the disk is not centred: {image:?} has no image"))
        })
        .collect();
    assert_eq!(perm.len(), n);

    // The reference wrote `np.allclose(..., atol=1e-18)` and that reads tighter than it is:
    // `allclose` keeps its **default `rtol = 1e-5`** unless told otherwise, so the bar it actually
    // ran was `|a - b| <= 1e-18 + 1e-5 |b|`. Carrying over the `atol` alone would have made this
    // seven orders of magnitude stricter than the claim it stands for -- and it fails, because the
    // triple product's summation order leaves the two images 2e-15 apart *relatively*. The
    // rotation invariance is a statement about the geometry, so it gets a relative tolerance, and
    // 1e-12 is still seven decades tighter than the reference's.
    let load = inst.port.load_matrix();
    for i in 0..n {
        for j in 0..n {
            let a = load.get(i, j);
            let b = load.get(perm[i], perm[j]);
            assert!(
                (a - b).abs() <= 1e-18 + 1e-12 * b.abs(),
                "the load is not rotation-invariant at ({i}, {j}): {a} vs {b}"
            );
        }
    }
}

// -- refusals, reset, and the override ----------------------------------------------------------

/// A head and a room built at different sample rates are refused, and the message names the
/// **membrane** — on both tiers.
#[test]
fn a_head_rate_mismatch_is_rejected_and_names_the_membrane() {
    for mount in MOUNTS {
        let mut room = membrane_room(rigid());
        let seam = head_seam(membrane::Domain::Rectangle, 16, 0.0, 0.5 * MEM_FS);
        let err = match mount {
            Mount::Baffled => RoomGrid::baffled(seam, &mut room, "z0", None, Spreading::Bilinear)
                .expect_err("half-rate head"),
            Mount::Suspended => {
                RoomGrid::suspended(seam, &mut room, "z", MEM_INDEX, None, Spreading::Bilinear)
                    .expect_err("half-rate head")
            }
        };
        assert!(matches!(
            err,
            WrapError::RateMismatch {
                label: "membrane",
                ..
            }
        ));
        assert!(err.to_string().contains("membrane fs"), "{err}");
        assert!(
            room.claims.is_empty(),
            "{mount:?}: a refused wrapper claimed a footprint"
        );
    }
}

/// The coupling adds no third stability condition.
///
/// `lambda_mem <= 1/sqrt(2)` and the room's `lambda_air <= 1/sqrt(3)` remain the only two, and the
/// first is refused by model #4 itself before a port is ever built — so a head too fast for its own
/// grid never reaches a wrapper at all.
#[test]
fn the_membrane_cfl_is_the_models_own_refusal() {
    let err = membrane::Params::new(
        Some(membrane::Domain::Rectangle),
        3.0e5,
        MEM_RHO,
        MEM_FS,
        32,
        Some(MEM_L),
        Some(MEM_L),
        None,
        0.0,
    )
    .expect_err("a head that fast must be refused");
    assert!(err.to_string().to_lowercase().contains("cfl"), "{err}");
}

/// `energy()` is an **override**, not a delegation.
///
/// The delegated number is the head's total *without* its coupling channel, so a wrapper that
/// simply forwarded `energy()` would report the number that looks fine and is not conserved.
///
/// The reference asserted a whole delegation surface alongside this (`inst.n_live`, `inst.u`,
/// `inst.state`), which Python gave it through `__getattr__`. There is no analogue: Rust has no
/// attribute fallback, so a caller reads `inst.seam.membrane.u` and the question cannot arise.
#[test]
fn the_head_overrides_energy_rather_than_delegating_it() {
    for mount in MOUNTS {
        let mut room = membrane_room(all_lossy());
        let mut inst = head(
            mount,
            &mut room,
            membrane::Domain::Rectangle,
            12,
            0.0,
            MEM_FS,
        );
        let u0 = membrane_bulge(&inst.seam.membrane, 1e-3);
        inst.set_state(&u0, &vec![0.0; u0.len()]);
        run(&mut inst, &mut room, 40);
        assert_eq!(
            inst.energy(),
            inst.seam.membrane.energy() + inst.radiated_energy()
        );
        assert_ne!(
            inst.energy(),
            inst.seam.membrane.energy(),
            "{mount:?}: the channel must be in the total"
        );
    }
}

/// The cut is geometry, not state: it must survive a reset.
#[test]
fn reset_clears_the_ledger_but_not_the_geometry() {
    for mount in MOUNTS {
        let mut room = membrane_room(rigid());
        let mut inst = head(
            mount,
            &mut room,
            membrane::Domain::Rectangle,
            12,
            0.0,
            MEM_FS,
        );
        let u0 = membrane_bulge(&inst.seam.membrane, 1e-3);
        inst.set_state(&u0, &vec![0.0; u0.len()]);
        run(&mut inst, &mut room, 20);
        let faces_before = room.cut_faces();
        inst.reset();
        assert_eq!(inst.radiated_energy(), 0.0, "{mount:?}");
        assert_eq!(inst.volume_velocity(), 0.0, "{mount:?}");
        assert!(inst.seam.membrane.u.iter().all(|&v| v == 0.0), "{mount:?}");
        assert_eq!(
            room.cut_faces(),
            faces_before,
            "{mount:?}: the cut is geometry"
        );
    }
}

/// N heads in one room, inherited unchanged: disjoint node sets, one room step.
#[test]
fn two_heads_share_one_room() {
    let mut room = membrane_room(rigid());
    let mut a = RoomGrid::suspended(
        head_seam(membrane::Domain::Rectangle, 12, 0.0, MEM_FS),
        &mut room,
        "z",
        3,
        None,
        Spreading::Bilinear,
    )
    .expect("a legal head");
    let mut b = RoomGrid::suspended(
        head_seam(membrane::Domain::Rectangle, 12, 0.0, MEM_FS),
        &mut room,
        "z",
        6,
        None,
        Spreading::Bilinear,
    )
    .expect("a legal head");
    let ua = membrane_bulge(&a.seam.membrane, 1e-3);
    let ub = membrane_bump(&b.seam.membrane, 1e-3);
    a.set_state(&ua, &vec![0.0; ua.len()]);
    b.set_state(&ub, &vec![0.0; ub.len()]);
    let e0 = scene_energy(&[&a, &b], &room);
    let mut worst: f64 = 0.0;
    for _ in 0..200 {
        a.step(&mut room, None).expect("a legal step");
        b.step(&mut room, None).expect("a legal step");
        room.step();
        worst = worst.max((scene_energy(&[&a, &b], &room) - e0).abs());
    }
    assert!(
        worst <= DRIFT_TOL * e0.abs(),
        "drift {:.3e}",
        worst / e0.abs()
    );
    let total = a.radiated_energy() + b.radiated_energy();
    assert!((total - room.injected).abs() <= LEDGER_TOL * total.abs());
}
