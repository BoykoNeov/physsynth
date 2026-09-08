//! Native acceptance bars for the **cut** — a rigid, zero-thickness internal partition on a plane
//! of velocity faces (HANDOFF §12.H, air-box batch 4).
//!
//! These replace `tests/test_airbox_cut.py`, retired in the same commit. The cut primitive came
//! into `physsynth-core` as the callee `InteriorSurfacePort` could not be ported without (plan
//! §14.1); this file is the room-tier file that became portable with it.
//!
//! The cut is batch 4's **only** new machinery. Prescribing a face velocity and injecting a `-q` /
//! `+q` pair on the two node planes that straddle it are the same arithmetic (`A_face w_z == W`
//! identically), so the port, the injection weights and the `injected` ledger are all batch 3's —
//! what is genuinely new is the *obstacle*, and an obstacle is a set of faces where `u` is held at
//! zero.
//!
//! Three things make it worth its own file:
//!
//! * **It costs nothing to book.** A cut face's `u` is identically zero at every half-step, so its
//!   contribution to the kinetic sum is identically zero: the energy identity needs no new term and
//!   no exclusion list. The area-coupling plan anticipated "removing those faces from the
//!   kinetic-energy sum"; they remove themselves.
//! * **It brings an exact modal oracle that is a genuinely new one.** A full cut splits the room
//!   into two independent rooms of length `(m + 1/2) h` and `(N - m - 1/2) h` — summing to `N h`
//!   exactly, because the cut lies on a *face*, half a cell past the last node on each side. That
//!   end is **face-centered**, so the exact discrete eigenvector along the cut axis is
//!   `cos(n pi i / (m + 1/2))` and **not** the room's own `cos(n pi i / N)`.
//! * **It must be additive, and the failure mode of a single-slot cut is silent.** With one slot a
//!   second plate's cut replaces the first's, the first keeps injecting its `-q` / `+q` pair and
//!   stops blocking — it degrades to a transparent doublet — while every ledger stays green. Hence
//!   the boolean union, and hence a port records its own face set.
//!
//! The oracle here exercises the cut **primitive**, never a port: `InteriorSurfacePort`'s rim
//! refusal means a legal port can never span a full cross-section, so a *port* can never seal the
//! room. A hand-placed cut can, and the corollary — a legal interior plate always has a
//! diffraction path around it — is measured in [`a_full_cut_isolates_exactly`].

use physsynth_core::airbox::{
    impedance_from_zeta, node_index, AirBox, CutError, Params, Wall, C0_AIR, RHO0_AIR,
};
use physsynth_core::airbox_port::PLANES;

const ROOM: [f64; 3] = [0.9, 0.7, 0.6];
const H: f64 = 0.1;
const CFL: f64 = 0.9;
/// The scene total, relative to the acoustic scale.
const DRIFT_TOL: f64 = 1e-12;
/// The modal oracle, absolute against an amplitude of 1.
const FIELD_TOL: f64 = 1e-12;

fn rigid() -> [Wall; 6] {
    [Wall::Rigid; 6]
}

fn lossy() -> [Wall; 6] {
    [Wall::Impedance(impedance_from_zeta(4.0, RHO0_AIR, C0_AIR)); 6]
}

fn make_airbox(walls: [Wall; 6]) -> AirBox {
    let fs = C0_AIR * 3.0f64.sqrt() / (CFL * H);
    AirBox::new(Params::new(ROOM, fs, H, walls, None, RHO0_AIR, C0_AIR).unwrap())
}

/// A deterministic broadband field in `[-1, 1]`, standing in for the reference's `standard_normal`.
///
/// Broadband excites **every** mode at once, including the ones a smooth pulse would miss, so an
/// energy identity that survives it has no direction left to hide in. The reference drew from
/// NumPy's PCG64 and nothing here needs the same *numbers* — only the same lack of structure — so
/// this is a splitmix64 hash of the flat index, mapped to a double.
fn noise(seed: u64, i: usize) -> f64 {
    let mut z = seed
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(i as u64)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    // The top 53 bits as a double in [0, 1), then centred.
    2.0 * ((z >> 11) as f64 / (1u64 << 53) as f64) - 1.0
}

fn seed_noise(room: &mut AirBox, seed: u64) {
    let p0: Vec<f64> = (0..room.p.n_nodes()).map(|i| noise(seed, i)).collect();
    room.set_state(&p0, None);
}

/// The **exact** discrete mode of one sub-room of a fully cut room — `(p0, u0, f)`.
///
/// The cut lies on a *face*, half a cell past the last node on each side, so the sub-rooms are
/// `(index + 1/2) h` and `(N - index - 1/2) h` long and the cut end is **face-centered**: the mirror
/// plane sits *between* nodes, ghost condition `p_{m+1} = p_m`. The exact eigenvector along the cut
/// axis is therefore `cos(n pi i / (index + 1/2))` and **not** the room's own `cos(n pi i / N)` — a
/// different oracle, not a restriction of the batch-1 one. Tensored with ordinary node-centered
/// cosines in plane, and `u0` is the omega-free half-step-back, so the evolution is exact rather
/// than nearly so.
fn sub_room_mode(
    room: &AirBox,
    axis: usize,
    index: usize,
    lo_side: bool,
    n: i32,
    transverse: [i32; 2],
) -> (Vec<f64>, [Vec<f64>; 3], f64) {
    let cells = room.p.n;
    let mut mu2 = 0.0f64;
    let mut tr = transverse.iter();
    let mut shapes: Vec<Vec<f64>> = Vec::with_capacity(3);
    for (a, &n_cells) in cells.iter().enumerate() {
        let mut v = vec![0.0f64; n_cells + 1];
        if a == axis {
            if lo_side {
                let theta = n as f64 * std::f64::consts::PI / (index as f64 + 0.5);
                for (i, slot) in v.iter_mut().take(index + 1).enumerate() {
                    *slot = (theta * i as f64).cos();
                }
                mu2 += (4.0 / (H * H)) * (theta / 2.0).sin().powi(2);
            } else {
                let span = n_cells - index - 1;
                let theta = n as f64 * std::f64::consts::PI / (n_cells as f64 - index as f64 - 0.5);
                for (m, slot) in v.iter_mut().skip(index + 1).enumerate() {
                    *slot = (theta * (span - m) as f64).cos();
                }
                mu2 += (4.0 / (H * H)) * (theta / 2.0).sin().powi(2);
            }
        } else {
            let q = *tr.next().expect("two transverse indices") as f64;
            for (i, slot) in v.iter_mut().enumerate() {
                *slot = (q * std::f64::consts::PI * i as f64 / n_cells as f64).cos();
            }
            mu2 += (4.0 / (H * H))
                * (q * std::f64::consts::PI / (2.0 * n_cells as f64))
                    .sin()
                    .powi(2);
        }
        shapes.push(v);
    }
    let shape = room.p.p_shape();
    let mut p0 = vec![0.0f64; room.p.n_nodes()];
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                p0[(i * shape[1] + j) * shape[2] + k] = shapes[0][i] * shapes[1][j] * shapes[2][k];
            }
        }
    }
    let s = room.p.k / (2.0 * room.p.rho0 * H);
    let mut u0: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (a, slot) in u0.iter_mut().enumerate() {
        let us = room.p.u_shape(a);
        let mut out = vec![0.0f64; us[0] * us[1] * us[2]];
        for i in 0..us[0] {
            for j in 0..us[1] {
                for k in 0..us[2] {
                    let mut hi = [i, j, k];
                    hi[a] += 1;
                    let a_flat = (i * shape[1] + j) * shape[2] + k;
                    let b_flat = (hi[0] * shape[1] + hi[1]) * shape[2] + hi[2];
                    out[(i * us[1] + j) * us[2] + k] = s * (p0[b_flat] - p0[a_flat]);
                }
            }
        }
        *slot = out;
    }
    let omega = (2.0 / room.p.k)
        * (C0_AIR * room.p.k * mu2.sqrt() / 2.0)
            .clamp(-1.0, 1.0)
            .asin();
    (p0, u0, omega / (2.0 * std::f64::consts::PI))
}

/// The flat velocity offsets of one full face plane, for the "is it zero there" bars.
fn face_plane(room: &AirBox, axis: usize, index: usize) -> Vec<usize> {
    let us = room.p.u_shape(axis);
    let (t0, t1) = physsynth_core::airbox::other_axes(axis);
    let mut out = Vec::new();
    let mut node = [0usize; 3];
    node[axis] = index;
    for a in 0..us[t0] {
        node[t0] = a;
        for b in 0..us[t1] {
            node[t1] = b;
            out.push((node[0] * us[1] + node[1]) * us[2] + node[2]);
        }
    }
    out
}

// -- the ledger -----------------------------------------------------------------------------------

/// `test_a_cut_room_still_conserves` — energy flat with a cut **and** a driven source: the identity
/// needs no new term.
///
/// The source is there on purpose. A cut room with nothing driving it would leave the `injected`
/// channel out of the statement, and it is the *sum* `acoustic + dissipated - injected` that is
/// flat.
#[test]
fn a_cut_room_still_conserves() {
    for (name, walls) in [("rigid", rigid()), ("lossy", lossy())] {
        for plane in PLANES {
            let mut room = make_airbox(walls);
            room.add_cut(plane, 3, None).expect("a legal cut");
            seed_noise(&mut room, 1);
            let e0 = room.energy();
            let scale = room.acoustic_energy();
            let at = node_index([0.2, 0.2, 0.1], room.p.h, room.p.n).expect("a node in the room");
            let mut worst: f64 = 0.0;
            for step in 0..400 {
                room.inject(1e-4 * noise(7, step), Some(at));
                room.step();
                worst = worst.max((room.energy() - e0).abs());
            }
            assert!(
                worst <= DRIFT_TOL * scale,
                "{name} / {plane}: drift {worst:e} against scale {scale:e}"
            );
        }
    }
}

/// `test_a_cut_face_carries_no_velocity_at_any_half_step` — both stored half-steps are zero on a cut
/// face, **at every entry point**.
///
/// Three of them, and each is a different code path: adding a cut to a room already in motion
/// (`register_cut` clears what is there), the consistent start (`set_state` cuts the caller's
/// `u^{-1/2}` as well as the `u^{+1/2}` it derives), and every subsequent step.
///
/// The `set_state` half is the one an implementation naturally misses, because only the *derived*
/// half-step passes through the momentum update — and the natural `u0`, the omega-free
/// `(k / 2 rho0 h) diff(p0)` the modal oracle uses, is **nonzero across a cut**, since `p0` jumps
/// there. Nothing numerical depends on it (the kinetic term is the cross-time product and its other
/// factor is zero), which is exactly why it needs a bar rather than a measurement: the only thing it
/// would break is the claim.
#[test]
fn a_cut_face_carries_no_velocity_at_any_half_step() {
    let mut room = make_airbox(rigid());
    seed_noise(&mut room, 0);
    for _ in 0..5 {
        room.step();
    }
    let plane = face_plane(&room, 2, 3);
    assert!(
        plane.iter().any(|&f| room.u[2][f] != 0.0),
        "the room must be moving there, or this proves nothing"
    );
    room.add_cut("z", 3, None).expect("a legal cut");
    for _ in 0..5 {
        for &f in &plane {
            assert_eq!(room.u[2][f], 0.0);
            assert_eq!(room.u_prev[2][f], 0.0);
        }
        room.step();
    }

    // ... and the consistent start, whose `u0` is nonzero on the cut before `set_state` touches it.
    let mut fresh = make_airbox(rigid());
    fresh.add_cut("z", 3, None).expect("a legal cut");
    let (p0, u0, _) = sub_room_mode(&fresh, 2, 3, true, 1, [0, 0]);
    let plane = face_plane(&fresh, 2, 3);
    assert!(
        plane.iter().any(|&f| u0[2][f] != 0.0),
        "the seed must be nonzero there, or this proves nothing"
    );
    fresh.set_state(&p0, Some([&u0[0], &u0[1], &u0[2]]));
    for &f in &plane {
        assert_eq!(fresh.u[2][f], 0.0);
        assert_eq!(fresh.u_prev[2][f], 0.0);
    }
}

// -- the oracle -----------------------------------------------------------------------------------

/// `test_the_sub_rooms_have_exact_half_offset_modes` — `cos(n pi i / (m + 1/2))` is exact to
/// machine precision, and it is a new oracle rather than a restriction of the room's own.
///
/// The room's tensor-cosine oracle has a **node-centered** mirror at each wall. The cut end is
/// **face-centered**: the mirror plane sits between nodes, ghost condition `p_{m+1} = p_m`, which
/// moves the eigenvector's denominator by exactly half a cell. Getting that half wrong looks like
/// scheme inaccuracy, so it is asserted at machine precision rather than as a rate.
///
/// The other sub-room stays **exactly** zero throughout, which is the same statement as
/// [`a_full_cut_isolates_exactly`] seen from the initial condition's side.
#[test]
fn the_sub_rooms_have_exact_half_offset_modes() {
    // The reference reported an unreachable index with `pytest.skip`; the `continue` below is
    // silent, so a room that shrank would leave this bar green having tested nothing. Counted.
    let mut cases = 0usize;
    for (axis, plane) in PLANES.iter().enumerate() {
        for index in [1usize, 3, 4] {
            for lo_side in [true, false] {
                let mut room = make_airbox(rigid());
                if index > room.p.n[axis] - 1 {
                    continue; // the room has fewer cells than that along this axis
                }
                room.add_cut(plane, index as i64, None)
                    .expect("a legal cut");
                let (p0, u0, f) = sub_room_mode(&room, axis, index, lo_side, 1, [1, 0]);
                room.set_state(&p0, Some([&u0[0], &u0[1], &u0[2]]));
                let e0 = room.energy();
                let shape = room.p.p_shape();
                // The far sub-room: everything on the other side of the cut.
                let far: Vec<usize> = (0..room.p.n_nodes())
                    .filter(|&m| {
                        let i = [
                            m / (shape[1] * shape[2]),
                            (m / shape[2]) % shape[1],
                            m % shape[2],
                        ][axis];
                        if lo_side {
                            i > index
                        } else {
                            i <= index
                        }
                    })
                    .collect();
                let (mut worst, mut drift) = (0.0f64, 0.0f64);
                for step in 1..=200 {
                    room.step();
                    let phase = (2.0 * std::f64::consts::PI * f * step as f64 * room.p.k).cos();
                    for (m, &want) in p0.iter().enumerate() {
                        worst = worst.max((room.pressure[m] - phase * want).abs());
                    }
                    drift = drift.max((room.energy() - e0).abs());
                    for &m in &far {
                        assert_eq!(
                            room.pressure[m], 0.0,
                            "the far sub-room must stay EXACTLY at rest"
                        );
                    }
                }
                let side = if lo_side { "lo" } else { "hi" };
                assert!(
                    worst <= FIELD_TOL,
                    "{plane} {index} {side}: field error {worst:e}"
                );
                assert!(
                    drift <= DRIFT_TOL * e0,
                    "{plane} {index} {side}: drift {drift:e}"
                );
                cases += 1;
            }
        }
    }
    assert_eq!(cases, 18, "three planes, three indices, two sides");
}

/// `test_the_sub_room_lengths_sum_to_the_room` — `(m + 1/2) h + (N - m - 1/2) h == N h`: no cell is
/// lost, and neither side is the room's own grid restricted.
#[test]
fn the_sub_room_lengths_sum_to_the_room() {
    let mut cases = 0usize;
    for (axis, _) in PLANES.iter().enumerate() {
        for index in [0usize, 2, 5] {
            let room = make_airbox(rigid());
            if index > room.p.n[axis] - 1 {
                continue;
            }
            let lo = (index as f64 + 0.5) * room.p.h;
            let hi = (room.p.n[axis] as f64 - index as f64 - 0.5) * room.p.h;
            let want = room.p.l_actual[axis];
            assert!((lo + hi - want).abs() <= 1e-15 * want);
            // ... and a half cell shifts BOTH ends, so neither sub-room length is a whole number of
            // cells. Asserted as the positive "half a cell past an integer" rather than the
            // reference's "is not an integer": a negative about a floating-point result passes for
            // any wrong value, not only for the right one. It needs a tolerance and the reference's
            // did not, which is the whole difference — `(index + 0.5) h / h` came back as
            // 5.500000000000002 on the first fixture, so an exact `== 0.5` is a claim about the
            // round trip and not about the geometry.
            assert!(((lo / room.p.h).fract() - 0.5).abs() <= 1e-12);
            assert!(((hi / room.p.h).fract() - 0.5).abs() <= 1e-12);
            cases += 1;
        }
    }
    assert_eq!(cases, 9, "three planes, three indices");
}

// -- what a cut does to sound ---------------------------------------------------------------------

/// The peak `|p|` beyond the cut plane, over a driven run.
fn peak_across(cut: Option<Option<[[i64; 2]; 2]>>) -> f64 {
    let mut room = make_airbox(rigid());
    if let Some(extent) = cut {
        room.add_cut("z", 3, extent).expect("a legal cut");
    }
    let at = node_index([0.4, 0.3, 0.1], room.p.h, room.p.n).expect("a node in the room");
    let shape = room.p.p_shape();
    let mut peak = 0.0f64;
    for _ in 0..220 {
        room.inject(1e-3, Some(at));
        room.step();
        for (m, &v) in room.pressure.iter().enumerate() {
            if m % shape[2] >= 4 {
                peak = peak.max(v.abs());
            }
        }
    }
    peak
}

/// `test_a_full_cut_isolates_exactly` — a full cut passes **exactly** `0.0`; a partial one does not,
/// so a port can never seal the room.
///
/// The exact zero is the assertion that a cut is a *rigid* boundary and not a strong impedance. Its
/// corollary is physical rather than a limitation to hide: a legal interior plate can only ever be a
/// partial cut, so it always has a diffraction path around it.
#[test]
fn a_full_cut_isolates_exactly() {
    let open_peak = peak_across(None);
    assert_eq!(peak_across(Some(None)), 0.0);
    let partial = peak_across(Some(Some([[0, 4], [0, 4]])));
    assert!(
        partial > 0.01 * open_peak,
        "partial/open = {:e}",
        partial / open_peak
    );
}

/// `test_cuts_are_additive_so_a_second_one_cannot_un_block_the_first` — two cuts coexist and the
/// first still blocks.
///
/// A single-slot cut would fail this silently: the first plate degrades to the phantom (keeps
/// injecting, stops blocking) while the scene stays perfectly green. Hence the boolean union, and
/// hence a port records its own face set.
#[test]
fn cuts_are_additive_so_a_second_one_cannot_un_block_the_first() {
    let mut room = make_airbox(rigid());
    room.add_cut("z", 3, None).expect("a legal cut");
    let faces_one = room.cut_faces();
    room.add_cut("x", 2, None).expect("a second legal cut");
    assert_eq!(
        room.cut_faces(),
        faces_one + (room.p.n[1] + 1) * (room.p.n[2] + 1)
    );
    // The source at node (4, 3, 1) sits in the sub-room {x >= 3, z <= 3}; both other quadrants must
    // stay at rest, and it is the FIRST cut's quadrant that a single-slot implementation would have
    // re-opened.
    let at = node_index([0.4, 0.3, 0.1], room.p.h, room.p.n).expect("a node in the room");
    for _ in 0..220 {
        room.inject(1e-3, Some(at));
        room.step();
    }
    let shape = room.p.p_shape();
    for (m, &v) in room.pressure.iter().enumerate() {
        let i = m / (shape[1] * shape[2]);
        let k = m % shape[2];
        if k >= 4 {
            assert_eq!(
                v, 0.0,
                "the z cut must still block after the x cut was added"
            );
        }
        if i < 3 {
            assert_eq!(v, 0.0, "and the x cut must block too");
        }
    }
}

/// `test_overlapping_hand_placed_cuts_are_idempotent` — a boolean union: cutting the same faces
/// twice is the same room, and is not refused.
///
/// Only a **port**'s cut is exclusive (its cut and its `-q`/`+q` pair are two halves of one object).
/// Two hand-placed partitions occupying the same faces are simply one partition.
#[test]
fn overlapping_hand_placed_cuts_are_idempotent() {
    let mut room = make_airbox(rigid());
    room.add_cut("z", 3, Some([[0, 5], [0, 5]])).expect("a cut");
    let once = room.cut_faces();
    room.add_cut("z", 3, Some([[2, 4], [1, 3]]))
        .expect("an overlapping hand-placed cut is accepted");
    assert_eq!(room.cut_faces(), once);
}

/// `test_cut_faces_counts_the_full_cross_section` — reported, not tuned.
#[test]
fn cut_faces_counts_the_full_cross_section() {
    let mut room = make_airbox(rigid());
    assert_eq!(room.cut_faces(), 0);
    room.add_cut("z", 3, None).expect("a legal cut");
    assert_eq!(room.cut_faces(), (room.p.n[0] + 1) * (room.p.n[1] + 1));

    let mut room = make_airbox(rigid());
    room.add_cut("y", 3, None).expect("a legal cut");
    assert_eq!(room.cut_faces(), (room.p.n[0] + 1) * (room.p.n[2] + 1));
}

// -- refusals ---------------------------------------------------------------------------------------

/// `test_refuses_an_unknown_plane` — an interior plane is named by its normal axis alone; it has no
/// end, unlike a wall face.
#[test]
fn an_unknown_plane_is_rejected() {
    let mut room = make_airbox(rigid());
    let err = room
        .add_cut("z0", 3, None)
        .expect_err("a face name is not a plane name");
    assert!(err.to_string().contains("unknown plane"));
    assert!(matches!(err, CutError::UnknownPlane(_)));
}

/// `test_refuses_a_cut_index_outside_the_faces` — the room's walls are at NODE planes 0 and N and
/// are already rigid; they are not cut positions.
#[test]
fn a_cut_index_outside_the_faces_is_rejected() {
    for index in [-1i64, 100] {
        let mut room = make_airbox(rigid());
        let err = room
            .add_cut("z", index, None)
            .expect_err("an index off the face range must be refused");
        assert!(err.to_string().contains("out of range"), "index {index}");
    }
}

/// `test_refuses_a_malformed_extent`, minus the half that has no analogue.
///
/// The reference also refuses `extent=(4, 4)` — "extent must be a ((lo0, hi0), (lo1, hi1)) pair" —
/// which is a claim about the shape of a Python argument; `Option<[[i64; 2]; 2]>` is the same claim
/// made by the compiler (plan §14.2). The half that survives is the one about a *value*: a range
/// running backwards, or off the axis.
#[test]
fn a_malformed_extent_is_rejected() {
    let mut room = make_airbox(rigid());
    let err = room
        .add_cut("z", 3, Some([[0, 4], [3, 1]]))
        .expect_err("lo > hi must be refused");
    assert!(err.to_string().contains("inclusive node-index range"));

    let err = room
        .add_cut("z", 3, Some([[0, 4], [0, 999]]))
        .expect_err("a range past the axis must be refused");
    assert!(err.to_string().contains("inclusive node-index range"));

    let err = room
        .add_cut("z", 3, Some([[-1, 4], [0, 4]]))
        .expect_err("a negative low end must be refused");
    assert!(err.to_string().contains("inclusive node-index range"));
    // ... and none of the three left a mark on the room.
    assert_eq!(room.cut_faces(), 0);
    assert!(room.cut_records.is_empty());
}
