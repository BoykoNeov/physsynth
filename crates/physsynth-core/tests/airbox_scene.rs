//! What the room can do that no lumped load can: give energy back late, and let two bodies hear
//! each other — carried from `tests/test_airbox_scene.py` (retirement plan §44).
//!
//! `airbox_port.rs` proves the body↔room terminal is exact and passive. Every bar there would still
//! pass if the port's free pressure were always zero, because the port would then degenerate into
//! a constant-`R` load with `R = R_room`: passive, perfectly conservative and **silently
//! reflection-free**. These are the bars that prove the free field is read at all.
//!
//! **Both arrival oracles are discrete:**
//!
//! * the reflection returns after `2d + 1` steps for a port `d` nodes from the wall, the `+1` being
//!   the injection's own step — asserted as **bit-identity** between two rooms differing only in
//!   `Lz`: identical before, different immediately after;
//! * a second body first moves at the **Manhattan** distance in nodes, not at `r/c0`: the 7-point
//!   stencil spreads influence one node along one axis per step, so an off-axis listener gets a
//!   machine-precision precursor before the physical wavefront.
//!
//! The file's three cross-tier bars — what a port *is*, as an equivalent radius — are in
//! `airbox_port_size.rs`, split out because they cost 98 s unoptimised (CI runs them optimised
//! only, the human's call, retirement plan §44.6).

use physsynth_core::airbox::{AirBox, Params, Wall, C0_AIR, RHO0_AIR};
use physsynth_core::airbox_wrap::RoomLoadedBody;
use physsynth_core::body::{self, ModalBody};

/// Acceptance criterion 1 — the same bar as every other resonator (`CLAUDE.md`).
const DRIFT_TOL: f64 = 1e-10;

// -- the fixture, `tests/helpers.py::make_room_loaded_body` (as in `airbox_port.rs`) -------------

/// The scene grid: 5 cm nodes, so a travel time is a countable number of steps.
const H: f64 = 0.05;
const ROOM: [f64; 3] = [0.5, 0.4, 0.3]; // -> N = (10, 8, 6)
const CFL: f64 = 0.9;
const AT: [f64; 3] = [0.15, 0.15, 0.15]; // node (3, 3, 3)
const FREQS: [f64; 2] = [220.0, 337.0];
const MASS: f64 = 0.05;
const RADIATION: f64 = 2e-3;
const Q0: [f64; 2] = [1e-3, 5e-4];

fn make_room(l: [f64; 3]) -> AirBox {
    let fs = C0_AIR * 3.0f64.sqrt() / (CFL * H);
    AirBox::new(Params::new(l, fs, H, [Wall::Rigid; 6], None, RHO0_AIR, C0_AIR).unwrap())
}

/// A lossless two-mode body mounted at `at`, displaced to `q0` (or left at rest).
fn mount(room: &mut AirBox, at: [f64; 3], q0: Option<[f64; 2]>) -> RoomLoadedBody {
    let a: Vec<f64> = (0..FREQS.len())
        .map(|i| RADIATION * 0.65f64.powi(i as i32))
        .collect();
    let body = ModalBody::new(
        body::Params::new(
            FREQS.to_vec(),
            room.p.fs,
            vec![0.0; 2],
            vec![MASS; 2],
            vec![1.0; 2],
            Some(a),
        )
        .unwrap(),
    );
    let mut inst = RoomLoadedBody::new(body, room, at, None).unwrap();
    if let Some(q0) = q0 {
        inst.set_state(&q0, &[0.0, 0.0]);
    }
    inst
}

/// The conserved total of a whole scene; the coupling term cancels identically.
fn scene_energy(instruments: &[&RoomLoadedBody], room: &AirBox) -> f64 {
    instruments.iter().map(|i| i.energy()).sum::<f64>() + room.energy()
}

// -- the room gives energy back, and the delay is right -------------------------------------------

/// A room identical in every way except `Lz = nz h`; the body's modal history.
fn history(nz: usize) -> Vec<Vec<f64>> {
    let mut room = make_room([ROOM[0], ROOM[1], nz as f64 * H]);
    let mut inst = mount(&mut room, AT, Some(Q0));
    (0..40)
        .map(|_| {
            inst.step(&mut room, 0.0).unwrap();
            room.step();
            inst.body.q().to_vec()
        })
        .collect()
}

/// The first step on which two histories differ in any bit.
fn first_difference(a: &[Vec<f64>], b: &[Vec<f64>]) -> Option<usize> {
    a.iter()
        .zip(b)
        .position(|(x, y)| x.iter().zip(y).any(|(u, v)| u.to_bits() != v.to_bits()))
}

/// `test_reflection_returns_at_the_round_trip_time` — bit-identity tests both directions at once:
/// the identical prefix catches a coupling that arrives **early** (or a free pressure that is
/// really a local self-term), and the difference right after it catches one that never arrives.
///
/// The reference room (`Nz = 12`) is deep enough that its own far-wall echo is still in transit
/// throughout: it lands at `2*9 + 1 = 19`, four steps past the largest asserted `t`.
#[test]
fn the_reflection_returns_at_the_round_trip_time() {
    let reference = history(12);
    for (nz, d) in [(6usize, 3usize), (8, 5), (10, 7)] {
        let hist = history(nz);
        let t = first_difference(&hist, &reference);
        assert_eq!(t, Some(2 * d + 1), "Lz = {nz} h");
    }
}

/// `test_reflection_is_not_a_lumped_load` — the delay scales with the distance, which no one-port
/// can do at any order: two nodes further away costs exactly four steps, out and back.
#[test]
fn the_delay_scales_with_the_distance_which_no_lumped_load_can_do() {
    let reference = history(12);
    let steps: Vec<usize> = [6usize, 8, 10]
        .iter()
        .map(|&nz| first_difference(&history(nz), &reference).unwrap())
        .collect();
    assert_eq!(steps, [7, 11, 15]);
    assert!(steps.windows(2).all(|w| w[1] - w[0] == 4));
}

/// `test_the_room_hands_energy_back` — a constant-`R` load books `k R U^2 >= 0`, so energy only
/// leaves. A port books `k pbar U` with `pbar` carrying the room's returning field, so the
/// radiated ledger genuinely **decreases** on some steps. Only the scene total is flat.
#[test]
fn the_room_hands_energy_back() {
    let mut room = make_room(ROOM);
    let mut inst = mount(&mut room, AT, Some(Q0));
    let e0 = scene_energy(&[&inst], &room);
    let mut hist = Vec::with_capacity(400);
    for n in 0..400 {
        inst.step(&mut room, 0.0).unwrap();
        room.step();
        hist.push(inst.radiated_energy());
        let d = (scene_energy(&[&inst], &room) - e0).abs();
        assert!(d < DRIFT_TOL * e0.abs(), "step {n}: {:e}", d / e0.abs());
    }
    let inc: Vec<f64> = hist.windows(2).map(|w| w[1] - w[0]).collect();
    assert!(
        inc.iter().any(|&x| x > 0.0),
        "the body never drove the room"
    );
    assert!(
        inc.iter().any(|&x| x < 0.0),
        "the room never drove the body back"
    );
}

// -- two instruments, one room --------------------------------------------------------------------

/// `test_second_body_moves_only_when_the_sound_arrives` — B is at rest and stays **exactly** zero
/// until A's disturbance can reach it, and the index is Manhattan: a bar written against `r/c0`
/// would fail by up to 2x for a reason that has nothing to do with the back-reaction.
#[test]
fn a_second_body_moves_exactly_when_the_sound_can_reach_it() {
    for (at_a, at_b) in [
        ([0.10, 0.20, 0.15], [0.40, 0.20, 0.15]),
        ([0.10, 0.10, 0.10], [0.35, 0.30, 0.25]),
    ] {
        let mut room = make_room(ROOM);
        let mut a = mount(&mut room, at_a, Some(Q0));
        let mut b = mount(&mut room, at_b, None);
        let (ia, ib) = (a.port.index(), b.port.index());
        let manhattan: usize = (0..3).map(|d| ia[d].abs_diff(ib[d])).sum();
        let euclid_steps = H
            * (0..3)
                .map(|d| (ia[d] as f64 - ib[d] as f64).powi(2))
                .sum::<f64>()
                .sqrt()
            / C0_AIR
            * room.p.fs;

        let e0 = scene_energy(&[&a, &b], &room);
        let mut first = None;
        for n in 0..2 * manhattan + 10 {
            a.step(&mut room, 0.0).unwrap();
            b.step(&mut room, 0.0).unwrap();
            room.step();
            if first.is_none() && b.body.q().iter().any(|&q| q != 0.0) {
                first = Some(n);
            }
            let d = (scene_energy(&[&a, &b], &room) - e0).abs();
            assert!(d < DRIFT_TOL * e0.abs(), "step {n}: {:e}", d / e0.abs());
        }
        assert_eq!(first, Some(manhattan), "{ia:?} -> {ib:?}");
        assert!(
            (manhattan as f64) < euclid_steps,
            "the grid's precursor beats the wavefront"
        );
    }
}

/// The two disjoint instruments of the last two bars, both displaced.
fn two_instruments() -> (AirBox, RoomLoadedBody, RoomLoadedBody) {
    let mut room = make_room(ROOM);
    let a = mount(&mut room, AT, Some([1e-3, 5e-4]));
    let b = mount(&mut room, [0.35, 0.25, 0.15], Some([7e-4, 2e-4]));
    (room, a, b)
}

/// `test_disjoint_ports_are_exactly_independent` — disjointness is exactly the condition that makes
/// each port's scalar solve exact, so two of them conserve the scene total.
#[test]
fn disjoint_ports_conserve_the_scene_total() {
    let (mut room, mut a, mut b) = two_instruments();
    let e0 = scene_energy(&[&a, &b], &room);
    let (mut lo, mut hi) = (e0, e0);
    for _ in 0..400 {
        a.step(&mut room, 0.0).unwrap();
        b.step(&mut room, 0.0).unwrap();
        room.step();
        let e = scene_energy(&[&a, &b], &room);
        assert!(!e.is_nan());
        lo = lo.min(e);
        hi = hi.max(e);
    }
    let d = (hi - lo) / e0.abs();
    assert!(d < DRIFT_TOL, "drift {d:e}");
}

/// `test_port_solve_order_does_not_matter` — with disjoint ports the scene is **bit-identical**
/// whichever order they solve in, which is why a port's free pressure may ignore injections
/// already queued this step.
#[test]
fn the_port_solve_order_does_not_matter() {
    let run = |reverse: bool| {
        let (mut room, mut a, mut b) = two_instruments();
        for _ in 0..120 {
            if reverse {
                b.step(&mut room, 0.0).unwrap();
                a.step(&mut room, 0.0).unwrap();
            } else {
                a.step(&mut room, 0.0).unwrap();
                b.step(&mut room, 0.0).unwrap();
            }
            room.step();
        }
        (
            a.body.q().to_vec(),
            b.body.q().to_vec(),
            room.pressure.clone(),
        )
    };
    let (forward, backward) = (run(false), run(true));
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&forward.0), bits(&backward.0));
    assert_eq!(bits(&forward.1), bits(&backward.1));
    assert_eq!(bits(&forward.2), bits(&backward.2));
}
