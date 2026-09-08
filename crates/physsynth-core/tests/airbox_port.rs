//! Native acceptance bars for the lumped body↔room terminal — `airbox_port::RoomPort` and
//! `airbox_wrap::RoomLoadedBody` (HANDOFF §12.H, air-box batch 2).
//!
//! These replace `tests/test_airbox_port.py`, retired in the same commit under the retirement
//! plan's naming rule; each bar below names the Python test it carries over, and the ones that are
//! new say so.
//!
//! The claim under test, in one line: **the room load is exact and passive — what the body loses,
//! the room gains, to machine precision, for any wall, any port position, any number of
//! instruments.** Each port's radiated energy *is* the room's injected, seen from the other side of
//! one terminal, so summing the two ledgers cancels the coupling term identically and
//!
//! ```text
//! sum_j inst_j.energy() + room.energy()
//! ```
//!
//! contains no coupling term at all. That works only because the room books its side from its
//! **own** post-closure pressure, never from a number the port hands back.
//!
//! Two traps get their own bars because nothing else would catch them:
//!
//! * `R_room` must carry the wall-closure factor `1 / (1 + beta)` — the room injects *before* it
//!   closes the wall, so a port on a lossy wall is divided by `1 + beta` along with everything
//!   else. The conserved total is **structurally blind** to getting this wrong: each side's
//!   identity telescopes against whatever pressure *it* used, so the sum of two internally
//!   consistent identities stays flat while the two disagree. The detectors are
//!   [`r_room_is_what_the_room_actually_does`], which measures it **differentially**, and
//!   [`the_two_ledgers_agree_across_the_terminal`]. Never the total.
//! * The port's local `O(port)` free-pressure read must replicate the full-array
//!   divergence-then-closure exactly, *including* at wall, edge and corner nodes where a node sees
//!   only the faces it has. That is asserted as bit-identity.
//!
//! And one refusal the energy report is structurally blind to: a port on an `open` face is
//! perfectly conservative and completely silent — the physics is exactly right and exactly
//! useless — so it must be refused at construction.

use physsynth_core::airbox::{impedance_from_zeta, AirBox, Params, Wall, C0_AIR, FACES, RHO0_AIR};
use physsynth_core::airbox_port::{PortError, RoomPort};
use physsynth_core::airbox_wrap::{RoomLoadedBody, WrapError};
use physsynth_core::body::{self, ModalBody};
use physsynth_core::reduce;

/// Acceptance criterion 1 — the same bar as every other resonator (`CLAUDE.md`).
const DRIFT_TOL: f64 = 1e-10;

// -- the suite's fixture, `tests/helpers.py::make_room_loaded_body` -----------------------------

/// A slightly larger room than the bare air-box fixture's, at a coarser `h`, so a port can sit at
/// an interior node, on a wall and in a corner of the same geometry.
const ROOM: [f64; 3] = [0.5, 0.4, 0.3]; // -> N = (10, 8, 6) at h = 5 cm
const H: f64 = 0.05;
/// `lambda = CFL_FRACTION / sqrt(3)`; `h` fixes the grid so the sample rate is solved for.
const CFL_FRACTION: f64 = 0.9;
/// Interior on every axis — node (3, 3, 3).
const INTERIOR: [f64; 3] = [0.15, 0.15, 0.15];
const ON_WALL: [f64; 3] = [0.0, 0.15, 0.15];
const ON_EDGE: [f64; 3] = [0.0, 0.0, 0.15];
const IN_CORNER: [f64; 3] = [0.0, 0.0, 0.0];
const SITES: [[f64; 3]; 4] = [INTERIOR, ON_WALL, ON_EDGE, IN_CORNER];
/// An inharmonic pair, a soundboard-scale modal mass, and radiation weights (m^2) big enough that
/// the room genuinely loads the body within a few hundred steps.
const FREQS: [f64; 2] = [220.0, 337.0];
const MASS: f64 = 0.05;
const RADIATION: f64 = 2e-3;
const Q0: [f64; 2] = [1e-3, 5e-4];

fn z_matched() -> f64 {
    impedance_from_zeta(1.0, RHO0_AIR, C0_AIR)
}

/// The room's sample rate, given `h`: `fs = c0 sqrt(3) / (cfl h)`.
fn fs_for(h: f64) -> f64 {
    C0_AIR * 3.0f64.sqrt() / (CFL_FRACTION * h)
}

fn rigid() -> [Wall; 6] {
    [Wall::Rigid; 6]
}

/// The wall array with `faces` set to `z`, everything else rigid.
fn walls_on(faces: &[&str], z: Wall) -> [Wall; 6] {
    let mut w = rigid();
    for name in faces {
        w[FACES.iter().position(|f| f == name).expect("a face name")] = z;
    }
    w
}

fn make_room(h: f64, walls: [Wall; 6]) -> AirBox {
    AirBox::new(Params::new(ROOM, fs_for(h), h, walls, None, RHO0_AIR, C0_AIR).unwrap())
}

/// How an instrument is built, mirroring the Python helper's keyword defaults one field at a time.
struct Fixture {
    h: f64,
    walls: [Wall; 6],
    at: [f64; 3],
    radius: Option<f64>,
    sigmas: f64,
    radiation: Option<Vec<f64>>,
    q0: Option<Vec<f64>>,
}

impl Default for Fixture {
    fn default() -> Fixture {
        Fixture {
            h: H,
            walls: rigid(),
            at: INTERIOR,
            radius: None,
            sigmas: 0.0,
            radiation: None,
            q0: Some(Q0.to_vec()),
        }
    }
}

impl Fixture {
    /// The modal body alone, at rest.
    fn body(&self, fs: f64) -> ModalBody {
        let a = self.radiation.clone().unwrap_or_else(|| {
            (0..FREQS.len())
                .map(|i| RADIATION * 0.65f64.powi(i as i32))
                .collect()
        });
        ModalBody::new(
            body::Params::new(
                FREQS.to_vec(),
                fs,
                vec![self.sigmas; FREQS.len()],
                vec![MASS; FREQS.len()],
                vec![1.0; FREQS.len()],
                Some(a),
            )
            .unwrap(),
        )
    }

    /// A fresh room and the instrument mounted in it.
    fn build(&self) -> (AirBox, RoomLoadedBody) {
        let mut room = make_room(self.h, self.walls);
        let inst = self.mount(&mut room);
        (room, inst)
    }

    /// Mount into an **existing** room — a second instrument in a shared scene.
    fn mount(&self, room: &mut AirBox) -> RoomLoadedBody {
        let fs = room.p.fs;
        let mut inst = RoomLoadedBody::new(self.body(fs), room, self.at, self.radius).unwrap();
        if let Some(q0) = &self.q0 {
            let zeros = vec![0.0; q0.len()];
            inst.set_state(q0, &zeros);
        }
        inst
    }
}

/// The conserved total of a whole scene: `sum_j inst_j.energy() + room.energy()`.
///
/// The coupling term cancels identically, which is why a drift in this is unambiguously a bug
/// rather than accounting.
fn scene_energy(instruments: &[&RoomLoadedBody], room: &AirBox) -> f64 {
    instruments.iter().map(|i| i.energy()).sum::<f64>() + room.energy()
}

/// Relative spread of the scene total over a run — the primary bug detector.
fn drift(inst: &mut RoomLoadedBody, room: &mut AirBox, steps: usize) -> f64 {
    let e0 = scene_energy(&[inst], room);
    let (mut lo, mut hi) = (e0, e0);
    for _ in 0..steps {
        inst.step(room, 0.0).unwrap();
        room.step();
        let e = scene_energy(&[inst], room);
        lo = lo.min(e);
        hi = hi.max(e);
    }
    (hi - lo) / e0.abs()
}

// -- 1-2. Exact conservation --------------------------------------------------------------------
//    Rigid walls isolate the coupling channel entirely: the ONLY place energy can go is the room,
//    so a leak has nowhere to hide. The lossy cases then add the wall channel on top, and the ones
//    with the port ON a lossy wall are the ones that need the 1/(1+beta) factor in R_room.

/// `test_conserved_rigid_room`.
#[test]
fn the_scene_total_is_flat_in_a_rigid_room() {
    for at in SITES {
        let (mut room, mut inst) = Fixture {
            at,
            ..Fixture::default()
        }
        .build();
        let d = drift(&mut inst, &mut room, 400);
        assert!(d < DRIFT_TOL, "at {at:?}: drift {d:.3e}");
    }
}

/// `test_conserved_lossy_walls` — the port sits on a lossy wall in three of these four, which is
/// the wall-closure factor's home ground.
#[test]
fn the_scene_total_is_flat_with_lossy_walls() {
    for at in SITES {
        let (mut room, mut inst) = Fixture {
            at,
            walls: [Wall::Impedance(z_matched()); 6],
            ..Fixture::default()
        }
        .build();
        let d = drift(&mut inst, &mut room, 400);
        assert!(d < DRIFT_TOL, "at {at:?}: drift {d:.3e}");
    }
}

/// `test_conserved_spread_port` — a spread port is many nodes with differing `W` and `beta`, so
/// this is the weighted sums exercised rather than the single-node shortcut.
#[test]
fn a_spread_port_conserves_across_wall_sets() {
    let sets: [[Wall; 6]; 3] = [rigid(), [Wall::Impedance(z_matched()); 6], {
        let mut w = walls_on(&["x0"], Wall::Impedance(z_matched()));
        w[FACES.iter().position(|f| *f == "y1").unwrap()] = Wall::Impedance(100.0);
        w
    }];
    for walls in sets {
        let (mut room, mut inst) = Fixture {
            at: INTERIOR,
            radius: Some(0.12),
            walls,
            ..Fixture::default()
        }
        .build();
        assert!(inst.port.node_count() > 1);
        let d = drift(&mut inst, &mut room, 400);
        assert!(d < DRIFT_TOL, "drift {d:.3e}");
    }
}

/// `test_conserved_lossy_body` — with `sigma > 0` the body is no longer lossless, so the scene
/// total **decreases** rather than staying flat. Passivity, the weaker statement that still holds.
#[test]
fn a_lossy_body_makes_the_scene_total_decrease_monotonically() {
    let (mut room, mut inst) = Fixture {
        sigmas: 40.0,
        ..Fixture::default()
    }
    .build();
    let e0 = scene_energy(&[&inst], &room);
    let mut prev = e0;
    for n in 0..400 {
        inst.step(&mut room, 0.0).unwrap();
        room.step();
        let e = scene_energy(&[&inst], &room);
        assert!(
            e <= prev + 1e-14 * e0.abs(),
            "step {n}: {e:.17e} > {prev:.17e}"
        );
        prev = e;
    }
    // And it genuinely decayed, rather than merely failing to grow.
    assert!(prev < 0.9 * e0);
}

// -- The local free-pressure read ---------------------------------------------------------------

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

/// The port's weighted read of a full field, the way the reference spells it.
fn weighted_read(port: &RoomPort, p_full: &[f64], p_old: &[f64]) -> f64 {
    let terms: Vec<f64> = port
        .flat()
        .iter()
        .enumerate()
        .map(|(m, &i)| port.w()[m] * (0.5 * (p_full[i] + p_old[i])))
        .collect();
    reduce::sum(&terms)
}

/// `test_free_pressure_matches_full_array` — `RoomPort::free_pressure` is `O(port)` where the
/// room's own update is a full-array operation, and they must agree **bit for bit**.
///
/// The interesting nodes are the ones where the divergence gives a node only the faces it actually
/// has (that absence *is* the rigid closure). An off-by-one in the local read would be a small,
/// plausible, position-dependent error — exactly the kind that survives an energy test, because
/// the port and the room would still agree with each other.
#[test]
fn the_local_free_pressure_read_is_the_full_array_update_exactly() {
    for walls in [rigid(), [Wall::Impedance(z_matched()); 6]] {
        for at in SITES {
            let (mut room, mut inst) = Fixture {
                at,
                walls,
                ..Fixture::default()
            }
            .build();
            for _ in 0..17 {
                // A field with structure, not a symmetric one.
                inst.step(&mut room, 0.0).unwrap();
                room.step();
            }
            let want = weighted_read(&inst.port, &full_array_free_pressure(&room), &room.pressure);
            assert_eq!(
                inst.port.free_pressure(&room).to_bits(),
                want.to_bits(),
                "at {at:?}"
            );
        }
    }
}

/// `test_free_pressure_matches_full_array_spread` — the same, for a ball straddling a corner:
/// clipped, one-sided, every weight different.
#[test]
fn a_clipped_ball_reads_the_same_as_the_full_array() {
    let (mut room, mut inst) = Fixture {
        at: IN_CORNER,
        radius: Some(0.12),
        walls: [Wall::Impedance(z_matched()); 6],
        ..Fixture::default()
    }
    .build();
    // Genuinely a clipped ball.
    assert!(inst.port.node_count() > 1 && inst.port.node_count() < room.pressure.len());
    for _ in 0..17 {
        inst.step(&mut room, 0.0).unwrap();
        room.step();
    }
    let want = weighted_read(&inst.port, &full_array_free_pressure(&room), &room.pressure);
    assert_eq!(inst.port.free_pressure(&room).to_bits(), want.to_bits());
}

// -- 3. R_room is what the room actually does ---------------------------------------------------

/// `test_R_room_is_what_the_room_does` — **differential**, not definitional.
///
/// Step the room twice from an identical saved state, once with `q = 0` and once with `q = U`, and
/// read the incremental centered pressure per unit volume velocity straight off the room. Nothing
/// here consults `R_room` except the final comparison, so it catches a wrong constant directly
/// instead of waiting for a drift to build. Comparing the coupled step's `pbar` against
/// `pbar_free + R_room U` would be a tautology and would pass for any `R_room` whatsoever.
#[test]
fn r_room_is_what_the_room_actually_does() {
    for walls in [rigid(), [Wall::Impedance(z_matched()); 6]] {
        for at in SITES {
            let (mut room, mut inst) = Fixture {
                at,
                walls,
                ..Fixture::default()
            }
            .build();
            for _ in 0..23 {
                inst.step(&mut room, 0.0).unwrap();
                room.step();
            }
            let snap = room.clone();

            let mut pbar_after = |q: f64| {
                let mut r = snap.clone();
                inst.port.reset();
                let p_old = r.pressure.clone();
                inst.port.inject(&mut r, q).unwrap();
                r.step();
                weighted_read(&inst.port, &r.pressure, &p_old)
            };
            let u = 3.7e-4;
            let measured = (pbar_after(u) - pbar_after(0.0)) / u;
            let want = inst.port.r_room();
            assert!(
                (measured - want).abs() <= 1e-12 * want,
                "at {at:?}: measured {measured:.17e} vs R_room {want:.17e}"
            );
        }
    }
}

/// `test_R_room_wall_factor_is_not_free` — pin the trap: on a lossy wall the naive
/// `k rho c^2 / (2 W)` differs from the truth by exactly `1 + beta`, so this is not a factor that
/// "cancels anyway".
#[test]
fn the_wall_closure_factor_in_r_room_is_not_free() {
    let (room, inst) = Fixture {
        at: IN_CORNER,
        walls: [Wall::Impedance(z_matched()); 6],
        ..Fixture::default()
    }
    .build();
    let port = &inst.port;
    let terms: Vec<f64> = port
        .flat()
        .iter()
        .enumerate()
        .map(|(m, &i)| {
            port.w()[m] * port.w()[m] * room.p.k * room.p.rho0 * room.p.c0.powi(2)
                / (2.0 * room.p.wv[i])
        })
        .collect();
    let naive = reduce::sum(&terms);
    let beta = room.p.beta[port.flat()[0]];
    // A corner sums three admittances: the factor is a big one.
    assert!(beta > 0.5, "beta = {beta}");
    let ratio = naive / port.r_room();
    assert!(
        (ratio - (1.0 + beta)).abs() <= 1e-13 * (1.0 + beta),
        "ratio {ratio} vs 1 + beta {}",
        1.0 + beta
    );
}

/// **New.** The cross-ledger residual, which is the second half of the `R_room` detector and is
/// *not* implied by the conserved total.
///
/// Each instrument's `radiated_energy` and the room's `injected` are the same terminal booked from
/// opposite sides — the port from `pbar_free + R_room U`, the room from its own post-closure
/// field. With `R_room` wrong the two disagree while the total stays flat, because each side's
/// identity telescopes against whatever pressure it used. Two instruments, so the sum is over the
/// scene and a single-port coincidence cannot carry it.
///
/// **One instrument sits in the corner**, and that is the whole of what makes this a detector. At
/// an interior node `beta` is zero and the wall-closure factor is exactly 1, so a version of this
/// bar with both ports inside the room passes with the factor deleted — measured, on the way in.
#[test]
fn the_two_ledgers_agree_across_the_terminal() {
    for walls in [rigid(), [Wall::Impedance(z_matched()); 6]] {
        let mut room = make_room(H, walls);
        let mut a = Fixture {
            at: IN_CORNER,
            ..Fixture::default()
        }
        .mount(&mut room);
        let mut b = Fixture {
            at: [0.35, 0.25, 0.15],
            radius: Some(0.11),
            q0: Some(vec![7e-4, 2e-4]),
            ..Fixture::default()
        }
        .mount(&mut room);
        for _ in 0..300 {
            a.step(&mut room, 0.0).unwrap();
            b.step(&mut room, 0.0).unwrap();
            room.step();
        }
        let radiated = a.radiated_energy() + b.radiated_energy();
        assert!(radiated.abs() > 0.0, "the room was never driven");
        let gap = (radiated - room.injected).abs() / radiated.abs();
        assert!(gap < 1e-12, "cross-ledger gap {gap:.3e}");
    }
}

// -- 4. The reduction ---------------------------------------------------------------------------

/// `test_zero_radiation_is_bit_identical_to_bare_body` — `a = 0` gives `G = 0`, `U = 0` and
/// nothing injected: the family's reduction-ledger entry, asserted as **bit** identity rather than
/// closeness. This is the check that catches sign errors nothing else catches.
#[test]
fn zero_radiation_is_bit_identical_to_the_bare_body() {
    let f = Fixture {
        radiation: Some(vec![0.0; FREQS.len()]),
        ..Fixture::default()
    };
    let (mut room, mut inst) = f.build();
    let mut bare = f.body(room.p.fs);
    bare.set_state(&Q0, &[0.0; 2]);

    for n in 0..200 {
        let force = 0.3 * (0.07 * n as f64).sin();
        inst.step(&mut room, force).unwrap();
        room.step();
        bare.step(force);
        assert_eq!(inst.body.q(), bare.q(), "step {n}");
        assert_eq!(inst.body.accel(), bare.accel(), "accel at step {n}");
    }
    assert_eq!(inst.volume_velocity(), 0.0);
    assert_eq!(inst.radiated_energy(), 0.0);
    assert_eq!(room.injected, 0.0);
}

// -- 5. Unconditional passivity -----------------------------------------------------------------

/// `test_absurd_coupling_stays_passive` — there is no CFL to find here, and the bar's job is to
/// prove that rather than trust it.
///
/// Note which direction stresses the solve: `R_room ~ k / W`, so a *coarse* grid makes `W` large
/// and `R_room` **small**. The stress case is a corner node on a **fine** grid, where `W = h^3/8`
/// is eight times smaller than the interior weight and, with lossy walls, `beta` is largest too —
/// with radiation weights inflated by `10^3` on top.
#[test]
fn an_absurd_coupling_stays_passive() {
    let (mut room, mut inst) = Fixture {
        h: 0.02,
        at: IN_CORNER,
        walls: [Wall::Impedance(z_matched()); 6],
        radiation: Some(vec![2.0, 1.3]),
        ..Fixture::default()
    }
    .build();
    // The solve is genuinely far from the decoupled limit.
    assert!(inst.g() * inst.port.r_room() > 1e3);
    let e0 = scene_energy(&[&inst], &room);
    let (mut lo, mut hi) = (e0, e0);
    let mut peak = 0.0f64;
    for _ in 0..300 {
        inst.step(&mut room, 0.0).unwrap();
        room.step();
        let e = scene_energy(&[&inst], &room);
        lo = lo.min(e);
        hi = hi.max(e);
        peak = peak.max(inst.body.q().iter().fold(0.0f64, |m, v| m.max(v.abs())));
    }
    assert!((hi - lo) / e0.abs() < DRIFT_TOL);
    assert!(room.pressure.iter().all(|v| v.is_finite()));
    // Bounded, not monotone: the room couples the two modes, so one may exceed its own initial
    // value while the scene total does not. A CFL failure would be orders of magnitude and secular
    // growth would show as a slow climb — neither is a few percent.
    assert!(peak < 5.0 * 1e-3, "peak {peak:.3e}");
}

/// `test_passivity_across_grids_and_walls` — the claim is "any wall, any grid", so sweep both
/// rather than asserting it once.
#[test]
fn passivity_holds_across_grids_and_wall_sets() {
    for h in [0.05, 0.025] {
        let sets: [[Wall; 6]; 4] = [
            rigid(),
            [Wall::Impedance(z_matched()); 6],
            [Wall::Impedance(20.0); 6],
            {
                let mut w = walls_on(&["x0"], Wall::Impedance(5.0));
                w[FACES.iter().position(|f| *f == "z1").unwrap()] = Wall::Impedance(z_matched());
                w
            },
        ];
        for walls in sets {
            let (mut room, mut inst) = Fixture {
                h,
                at: ON_WALL,
                walls,
                radius: Some(0.1),
                ..Fixture::default()
            }
            .build();
            let d = drift(&mut inst, &mut room, 150);
            assert!(d < DRIFT_TOL, "h={h} drift {d:.3e}");
        }
    }
}

// -- 6. The refusals ----------------------------------------------------------------------------

/// The refusal's message, so a bar can assert on the wording the way the retired suite did.
fn port_err(e: PortError) -> String {
    e.to_string()
}

/// `test_open_face_port_is_refused`.
#[test]
fn a_port_on_an_open_face_is_rejected() {
    let mut room = make_room(H, walls_on(&["x0"], Wall::Open));
    let e = RoomPort::new(&mut room, [0.0, 0.15, 0.15], None).unwrap_err();
    assert!(port_err(e).contains("open"));
}

/// `test_open_face_reached_by_a_BALL_is_refused` — a port whose *centre* is interior can still
/// reach the face once the ball is laid down, so the check runs over the whole node set.
#[test]
fn an_open_face_reached_by_the_ball_is_rejected() {
    let mut room = make_room(H, walls_on(&["x0"], Wall::Open));
    RoomPort::new(&mut room, [0.3, 0.3, 0.3], Some(0.12)).expect("far enough: fine");
    let e = RoomPort::new(&mut room, [0.1, 0.3, 0.3], Some(0.12)).unwrap_err();
    assert!(port_err(e).contains("open"));
}

/// `test_shared_node_is_refused` — `(0.16, 0.16, 0.16)` snaps onto node `(3, 3, 3)`.
#[test]
fn a_shared_node_is_rejected() {
    let (mut room, _inst) = Fixture::default().build();
    let e = RoomPort::new(&mut room, [0.16, 0.16, 0.16], None).unwrap_err();
    assert!(port_err(e).contains("shares node"));
}

/// `test_overlapping_balls_are_refused`.
#[test]
fn overlapping_balls_are_rejected() {
    let (mut room, _inst) = Fixture {
        radius: Some(0.1),
        ..Fixture::default()
    }
    .build();
    let e = RoomPort::new(&mut room, [0.3, 0.15, 0.15], Some(0.1)).unwrap_err();
    assert!(port_err(e).contains("shares node"));
}

/// `test_disjoint_ports_are_accepted`.
#[test]
fn disjoint_ports_are_accepted() {
    let (mut room, inst) = Fixture {
        radius: Some(0.07),
        ..Fixture::default()
    }
    .build();
    let second = RoomPort::new(&mut room, [0.35, 0.25, 0.15], Some(0.07)).unwrap();
    assert!(second.node_count() > 1);
    assert!(!second.flat().iter().any(|n| inst.port.flat().contains(n)));
}

/// `test_port_outside_the_room_is_refused`.
#[test]
fn a_port_outside_the_room_is_rejected() {
    let mut room = make_room(H, rigid());
    let e = RoomPort::new(&mut room, [5.0, 0.15, 0.15], None).unwrap_err();
    assert!(port_err(e).contains("outside the room"));
}

/// `test_unresolvable_radius_is_refused` — a ball smaller than the grid **is** a point port; say
/// so rather than pretending otherwise.
#[test]
fn an_unresolvable_radius_is_rejected() {
    let mut room = make_room(H, rigid());
    let e = RoomPort::new(&mut room, INTERIOR, Some(0.01)).unwrap_err();
    assert!(port_err(e).contains("smaller than the grid"));
    let e = RoomPort::new(&mut room, INTERIOR, Some(0.0)).unwrap_err();
    assert!(port_err(e).contains("positive length"));
}

/// `test_forgotten_room_step_raises` — the port does not step the room, and forgetting to is
/// caught loudly rather than silently.
#[test]
fn a_forgotten_room_step_is_rejected_and_recovers() {
    let (mut room, mut inst) = Fixture::default().build();
    inst.step(&mut room, 0.0).unwrap();
    let e = inst.step(&mut room, 0.0).unwrap_err();
    assert!(e.to_string().contains("twice within one room step"));
    room.step();
    // And it recovers the moment the room advances.
    inst.step(&mut room, 0.0).unwrap();
}

/// `test_forgotten_room_step_guard_is_per_port` — with two instruments in one room the second
/// solves while the first's injection is queued, so a global "is anything pending" guard would
/// fire on every scene ever built.
#[test]
fn the_forgotten_room_step_guard_is_per_port() {
    let mut room = make_room(H, rigid());
    let mut a = Fixture::default().mount(&mut room);
    let mut b = Fixture {
        at: [0.35, 0.25, 0.15],
        ..Fixture::default()
    }
    .mount(&mut room);
    for _ in 0..10 {
        a.step(&mut room, 0.0).unwrap();
        b.step(&mut room, 0.0).unwrap(); // must NOT refuse
        room.step();
    }
}

/// `test_sample_rate_mismatch_is_refused`.
#[test]
fn a_sample_rate_mismatch_is_rejected() {
    let mut room = make_room(H, rigid());
    let fs = room.p.fs * 1.5;
    let body = ModalBody::new(
        body::Params::new(
            vec![220.0],
            fs,
            vec![0.0],
            vec![MASS],
            vec![1.0],
            Some(vec![1e-3]),
        )
        .unwrap(),
    );
    let e = RoomLoadedBody::new(body, &mut room, INTERIOR, None).unwrap_err();
    assert!(matches!(e, WrapError::RateMismatch { .. }));
    assert!(e.to_string().contains("sample-rate mismatch"));
}

/// `test_set_state_and_reset_clear_the_coupling_ledger` — in the reference the attribute fallback
/// would delegate both to the bare body and leave a stale ledger behind.
#[test]
fn set_state_and_reset_clear_the_coupling_ledger() {
    let (mut room, mut inst) = Fixture::default().build();
    for _ in 0..20 {
        inst.step(&mut room, 0.0).unwrap();
        room.step();
    }
    assert!(inst.radiated_energy() != 0.0);
    inst.set_state(&[1e-3, 0.0], &[0.0, 0.0]);
    assert_eq!(
        (
            inst.radiated_energy(),
            inst.volume_velocity(),
            inst.port_pressure(),
            inst.n()
        ),
        (0.0, 0.0, 0.0, 0)
    );
    inst.step(&mut room, 0.0).unwrap(); // the pending mark went with it
    room.step();
    inst.reset();
    assert_eq!(inst.radiated_energy(), 0.0);
    assert_eq!(inst.body.q(), &[0.0, 0.0]);
}

/// `test_energy_is_an_override_not_a_delegation` — a delegated `energy()` would return the bare
/// modal energy, the total *without* its coupling channel, which is the number that looks fine and
/// is not conserved.
#[test]
fn the_energy_is_an_override_not_a_delegation() {
    let (mut room, mut inst) = Fixture::default().build();
    for _ in 0..50 {
        inst.step(&mut room, 0.0).unwrap();
        room.step();
    }
    assert!(inst.energy() != inst.body.energy());
    assert_eq!(inst.energy(), inst.body.energy() + inst.radiated_energy());
}

/// `test_room_set_state_unsticks_every_port`.
#[test]
fn a_room_set_state_unsticks_every_port() {
    let (mut room, mut inst) = Fixture::default().build();
    inst.step(&mut room, 0.0).unwrap();
    let zeros = vec![0.0; room.pressure.len()];
    room.set_state(&zeros, None);
    inst.step(&mut room, 0.0).unwrap(); // must not refuse: the room's fresh run owes nothing
}

/// **New** — and it is the bar the native design needs that the reference did not.
///
/// The reference unsticks a port by *writing into it* (`room._ports` then `_queued_at = -1`). A
/// value-typed port cannot be written into, so the room carries an epoch instead. The distinction
/// is invisible to the bar above, because there the port had queued at step 0 and `set_state`
/// returns the room to step 0 — a bare step-count comparison would have refused, and only *does
/// not* because the epoch moved. Assert exactly that: same step count, different run.
#[test]
fn the_epoch_and_not_the_step_count_is_what_unsticks_a_port() {
    let (mut room, mut inst) = Fixture::default().build();
    inst.step(&mut room, 0.0).unwrap();
    assert_eq!(
        room.n, 0,
        "the port queued at step 0 and the room has not advanced"
    );
    let zeros = vec![0.0; room.pressure.len()];
    room.set_state(&zeros, None);
    assert_eq!(room.n, 0, "and set_state puts it back at step 0");
    inst.step(&mut room, 0.0).unwrap();
}
