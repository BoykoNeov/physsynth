//! The drum in a room through the payload builder — `docs/dev/membrane-mapping-plan.md` §13.
//!
//! The page tunes the drum in vacuo and SHOWS what the air does to it: a bare twin of the same head
//! runs beside the room, and the read-out is the twin's and the room's measured pitch and ring time
//! per mode, and the shift between them. Every bar here listens to the simulated output — the twin
//! must land on the scheme's in-vacuo values, the room must move off them — so a build that
//! reported the right numbers without running them would still go red.
//!
//! No frozen case can name this page: it was born after the Python viewer was deleted
//! (`tests/frozen.rs`, `MODELS_BORN_AFTER_THE_FREEZE`).

mod common;

use common::{f, ok, sim};
use physsynth_core::membrane::{self as mem, Domain};
use physsynth_core::membrane_tuning::{
    head, lowest_modes, mode_frequency, tune_membrane, Geometry, SetBy, Targets,
};
use serde_json::{json, Value};
use std::sync::OnceLock;

// The air, written out rather than imported (a bar about a constant must not import it): the
// room's sound speed, and its Courant number 0.9/sqrt(3).
const C0: f64 = 343.0;
const CFL_AIR: f64 = 0.9;

fn dr(overrides: Value) -> Value {
    let mut p = json!({"model": "drumroom"});
    common::merge(&mut p, overrides);
    p
}

fn baffled() -> &'static Value {
    static P: OnceLock<Value> = OnceLock::new();
    P.get_or_init(|| ok(&dr(json!({}))))
}

fn suspended() -> &'static Value {
    static P: OnceLock<Value> = OnceLock::new();
    P.get_or_init(|| ok(&dr(json!({"domain": "suspended"}))))
}

fn rigid() -> &'static Value {
    static P: OnceLock<Value> = OnceLock::new();
    P.get_or_init(|| ok(&dr(json!({"walls": "rigid"}))))
}

fn modes(d: &Value) -> &Vec<Value> {
    d["meta"]["air"]["modes"].as_array().expect("air modes")
}

/// The fundamental's measured shift, room against twin (cents).
fn shift0(d: &Value) -> f64 {
    f(&modes(d)[0]["shift_cents"])
}

fn cents(a: f64, b: f64) -> f64 {
    1200.0 * (a / b).log2()
}

// -- the musical drum: every key off the fallbacks (head 7.5 mil, f0 220, t60 2 / 0.5, f_ref
//    2000), with physics decoys a musical build must ignore and a sample rate the room overrides.
const HEAD_LIGHT: &str = "single-ply 3 mil";
const HEAD_HEAVY: &str = "two-ply 7 + 7 mil";
const F0: f64 = 190.0;
const T60_LOW: f64 = 1.1;
const T60_HIGH: f64 = 0.35;
const F_REF: f64 = 1100.0;
const RADIUS: f64 = 0.14;
const N: i64 = 20;
const AIR_H: f64 = 0.0155;

fn musical(head: &str, overrides: Value) -> Value {
    let mut p = dr(json!({
        "controls": "musical", "head": head, "set_by": "pitch", "f0": F0,
        "t60_low": T60_LOW, "t60_high": T60_HIGH, "f_ref": F_REF,
        "radius": RADIUS, "N": N, "air_h": AIR_H, "room_size": 0.46,
        // Decoys: the room sets the sample rate, and a musical build reads none of the physics.
        "sample_rate": 22050.0, "T": 777.0, "rho": 0.9, "sigma": 5.0, "tension": 1234.0,
    }));
    common::merge(&mut p, overrides);
    p
}

fn light() -> &'static Value {
    static P: OnceLock<Value> = OnceLock::new();
    P.get_or_init(|| ok(&musical(HEAD_LIGHT, json!({}))))
}

fn heavy() -> &'static Value {
    static P: OnceLock<Value> = OnceLock::new();
    P.get_or_init(|| ok(&musical(HEAD_HEAVY, json!({}))))
}

fn room_fs(h: f64) -> f64 {
    C0 / (CFL_AIR / 3f64.sqrt() * h)
}

/// The bare twin lands on the scheme's in-vacuo frequency for every clear mode, and the room moves
/// every one of them DOWN: the air is a mass load on the head.
#[test]
fn the_twin_sits_on_the_scheme_and_the_air_lowers_every_mode() {
    for d in [baffled(), suspended(), light()] {
        for m in modes(d) {
            let target = f(&m["target"]["f"]);
            let tw = &m["twin"]["peak"];
            match tw["status"].as_str().unwrap() {
                "clear" => assert!(cents(f(&tw["f"]), target).abs() < 0.1, "{m}"),
                // a pair the grid split: each member on its own scheme value
                "grid pair" => {
                    for mem in m["members"].as_array().unwrap() {
                        let t = f(&mem["target"]["f"]);
                        let got = f(&mem["twin"]["peak"]["f"]);
                        assert!(cents(got, t).abs() < 0.1, "{mem}");
                    }
                }
                s => panic!("twin read {s}: {m}"),
            }
            assert!(f(&m["shift_cents"]) < -20.0, "{m}");
        }
    }
}

/// The fundamental's shift is the size the plan's Part 0 measured: a few semitones, hung more.
#[test]
fn the_fundamental_drops_semitones_and_a_hung_drum_drops_further() {
    let (b, s) = (shift0(baffled()), shift0(suspended()));
    assert!((-400.0..-150.0).contains(&b), "baffled {b}");
    assert!((-550.0..-250.0).contains(&s), "suspended {s}");
    assert!(s < b - 50.0, "hung {s} vs baffled {b}");
}

/// A heavier head is moved less by the same air: the load is the air's mass against the head's.
#[test]
fn a_heavier_head_is_shifted_less() {
    let (l, h) = (shift0(light()), shift0(heavy()));
    assert!(l < 1.4 * h, "light {l} vs heavy {h}");
}

/// The walls are built as asked: a rigid room keeps far more of the head's energy than an
/// absorbing one, which the radiated sound leaves.
#[test]
fn rigid_walls_keep_the_energy_absorbing_walls_let_go() {
    let keep = |d: &Value| f(&d["meta"]["air"]["head_energy_kept"]);
    assert!(keep(rigid()) > 0.2, "{}", keep(rigid()));
    assert!(keep(baffled()) < 0.05, "{}", keep(baffled()));
    assert_eq!(rigid()["walls"], "rigid");
    assert_eq!(baffled()["walls"], "absorbing");
}

fn ledger(d: &Value, key: &str) -> Vec<f64> {
    d["meta"]["ledger"][key]
        .as_array()
        .unwrap_or_else(|| panic!("no {key}"))
        .iter()
        .map(f)
        .collect()
}

/// Both ledgers: the power the head sends equals the power the room receives, to round-off. The
/// scene's total counts what the walls absorbed, so a lossless head conserves it in any room, and a
/// lossy head only ever lowers it. Absorbing walls take most of the energy; rigid ones take none.
#[test]
fn the_seam_closes_and_the_scene_is_passive() {
    for d in [baffled(), suspended(), rigid(), light()] {
        let lg = &d["meta"]["ledger"];
        assert!(f(&lg["residual_max"]) < 1e-12, "{lg}");
    }
    for d in [baffled(), suspended(), rigid()] {
        let drift = ledger(d, "total_frac")
            .iter()
            .map(|v| (v - 1.0).abs())
            .fold(0.0, f64::max);
        assert!(drift < 1e-10, "{drift}");
    }
    let t = ledger(light(), "total_frac");
    assert!(t.windows(2).all(|w| w[1] <= w[0] + 1e-12), "not passive");
    assert!(t[t.len() - 1] < 0.99, "the head's own loss shows");
    let absorbed = |d: &Value| *ledger(d, "e_dissipated_frac").last().unwrap();
    assert!(absorbed(baffled()) > 0.5, "{}", absorbed(baffled()));
    assert!(absorbed(suspended()) > 0.5, "{}", absorbed(suspended()));
    assert_eq!(absorbed(rigid()), 0.0);
}

/// The physical head reads every key: its twin lands on the in-vacuo scheme built from the request
/// independently, every key off the page's default (T 3000, rho 0.26, sigma 0, radius 0.12, N 24,
/// air_h 0.0165).
#[test]
fn the_physical_head_reads_every_key() {
    let (t, rho, sigma, radius, n, h) = (2300.0, 0.31, 1.5, 0.13, 22, 0.0150);
    let d = ok(&dr(json!({
        "T": t, "rho": rho, "sigma": sigma, "radius": radius, "N": n, "air_h": h,
        "room_size": 0.44, "audio_duration": 0.3, "sample_rate": 22050.0,
    })));
    let fs = room_fs(h);
    assert!(
        (f(&d["fs_sim"]) - fs).abs() < 1e-3,
        "{} vs {fs}",
        d["fs_sim"]
    );
    assert_eq!(d["controls"], "physical");
    let p = mem::Params::new(
        Some(Domain::Circle),
        t,
        rho,
        fs,
        n,
        None,
        None,
        Some(radius),
        sigma,
    )
    .unwrap();
    let mu = lowest_modes(&p, 1).unwrap()[0];
    let k = 1.0 / fs;
    let want = mode_frequency(sigma * k, p.c * p.c * k * k * mu, 0.0, k).unwrap();
    let got = f(&modes(&d)[0]["twin"]["peak"]["f"]);
    assert!(cents(got, want).abs() < 0.1, "{got} vs {want}");
    assert!((f(&modes(&d)[0]["target"]["t60"]) - 3.0 * 10f64.ln() / sigma).abs() < 1e-3);
}

/// The musical head is tuned at the ROOM's sample rate, from the head and the targets alone: its
/// twin's fundamental is the asked pitch and its ring times are the two asked for, and every
/// mode's twin ring time is the scheme's — which a dropped frequency-dependent loss would break.
#[test]
fn the_musical_head_hits_its_targets_in_vacuo() {
    let d = light();
    assert_eq!(d["controls"], "musical");
    let fs = room_fs(AIR_H);
    assert!((f(&d["fs_sim"]) - fs).abs() < 1e-3);
    let m0 = &modes(d)[0];
    assert!(cents(f(&m0["twin"]["peak"]["f"]), F0).abs() < 0.1, "{m0}");
    assert!(
        (f(&m0["target"]["t60"]) / T60_LOW - 1.0).abs() < 1e-6,
        "{m0}"
    );
    let tuned = tune_membrane(
        &Geometry {
            domain: Domain::Circle,
            lx: 0.0,
            ly: 0.0,
            radius: RADIUS,
            n: N,
            fs,
        },
        head(HEAD_LIGHT).unwrap().areal_density(),
        &Targets {
            set_by: SetBy::Pitch(F0),
            t60_low: T60_LOW,
            t60_high: T60_HIGH,
            f_ref: F_REF,
        },
    )
    .unwrap();
    assert!(tuned.sigma1 > 0.0);
    let head_meta = &d["meta"]["head"];
    assert!(
        (f(&head_meta["T"]) / tuned.tension - 1.0).abs() < 1e-3,
        "{head_meta}"
    );
    let mut rang = 0;
    for m in modes(d) {
        let want = f(&m["target"]["t60"]);
        if let Some(got) = m["twin"]["ring"]["t60"].as_f64() {
            assert!((got / want - 1.0).abs() < 0.02, "{m}");
            rang += 1;
        }
    }
    assert!(rang >= 4, "only {rang} twin ring times were measured");
    // the high modes ring shorter than the fundamental: the loss rises with frequency
    let last = f(&modes(d)[modes(d).len() - 1]["target"]["t60"]);
    assert!(last < 0.8 * T60_LOW, "{last}");
    assert!(d["tuning"].is_object());
}

/// The room takes energy out of the fundamental: its ring time is quoted (the fit stops above the
/// floor of the slower modes' leakage) and is shorter than the twin's.
#[test]
fn the_air_shortens_the_fundamentals_ring() {
    let m0 = &modes(light())[0];
    let room = f(&m0["room"]["ring"]["t60"]);
    let twin = f(&m0["twin"]["ring"]["t60"]);
    assert!(room < 0.5 * twin, "{m0}");
    let b0 = &modes(baffled())[0];
    assert!(b0["room"]["ring"]["t60"].is_f64(), "{b0}");
    assert!(
        b0["target"]["t60"].is_null(),
        "a lossless head has no ring time to aim at"
    );
}

/// A pair the grid splits (by 54 cents at N 12) is measured member by member: each member's twin
/// is its own scheme value and the shift is read, not lost to a "split".
#[test]
fn a_grid_split_pair_is_measured_member_by_member() {
    let d = ok(&dr(json!({"N": 12})));
    let m = modes(&d)
        .iter()
        .find(|m| f(&m["grid_split_cents"]) > 20.0)
        .expect("N 12 splits the (2,1) pair");
    assert_eq!(m["multiplicity"], 2);
    assert_eq!(m["twin"]["peak"]["status"], "grid pair", "{m}");
    assert_eq!(m["room"]["peak"]["status"], "grid pair", "{m}");
    let members = m["members"].as_array().unwrap();
    let shifts: Vec<f64> = members.iter().map(|u| f(&u["shift_cents"])).collect();
    assert!(shifts.iter().all(|s| *s < -50.0), "{shifts:?}");
    assert!((f(&m["shift_cents"]) - 0.5 * (shifts[0] + shifts[1])).abs() < 0.01);
}

/// The hung drum's (0,2) mode drops below its in-vacuo neighbour, the (2,1) pair: it is found,
/// because the window starts where the pair actually landed in the room, not where it sat in vacuo.
#[test]
fn a_mode_the_air_moves_past_its_neighbour_is_still_found() {
    let ms = modes(suspended());
    let (pair, axis) = (&ms[2], &ms[3]);
    assert_eq!(axis["multiplicity"], 1);
    let below = f(&pair["target"]["f"]);
    let room = &axis["room"]["peak"];
    assert_eq!(room["status"], "clear", "{axis}");
    assert!(f(&room["f"]) < below, "{} vs {below}", room["f"]);
    assert!(f(&axis["shift_cents"]) < -100.0, "{axis}");
}

#[test]
fn every_mounting_and_every_wall_builds_and_nothing_else_does() {
    for (tier, walls) in [
        ("baffled", "rigid"),
        ("suspended", "absorbing"),
        ("suspended", "rigid"),
    ] {
        let d = ok(&dr(
            json!({"domain": tier, "walls": walls, "audio_duration": 0.02}),
        ));
        assert_eq!(d["tier"], tier);
        assert_eq!(d["walls"], walls);
    }
    let e = sim(&dr(json!({"domain": "rectangle"})));
    assert!(
        e["error"]["message"].as_str().unwrap().contains("baffled"),
        "{e}"
    );
    let e = sim(&dr(json!({"walls": "open"})));
    assert!(
        e["error"]["message"]
            .as_str()
            .unwrap()
            .contains("absorbing"),
        "{e}"
    );
}

#[test]
fn what_the_room_cannot_hold_is_refused_with_advice() {
    let msg = |p: Value| -> String {
        let e = sim(&dr(p));
        e["error"]["message"]
            .as_str()
            .unwrap_or_else(|| panic!("not refused: {e}"))
            .to_owned()
    };
    // N 9 groups the grid-split pair wrongly; 10 is the floor
    assert!(msg(json!({"N": 9})).contains("N must be in [10, 40]"));
    assert!(msg(json!({"N": 41})).contains("N must be in [10, 40]"));
    // a head too fast for the air's sample rate: the advice is the air's grid
    let m = msg(json!({"T": 1.0e6, "rho": 0.05}));
    assert!(m.contains("CFL") && m.contains("air_h"), "{m}");
    let m = msg(json!({"controls": "musical", "f0": 1500.0, "f_ref": 6000.0, "N": 40}));
    assert!(m.contains("CFL") && m.contains("air_h"), "{m}");
    // the footprint and the budget
    assert!(msg(json!({"room_size": 0.8, "air_h": 0.010})).contains("nodes"));
    let m = msg(json!({"room_size": 0.62, "air_h": 0.010, "audio_duration": 1.0}));
    assert!(m.contains("budget"), "{m}");
    // a head wider than the room
    let m = msg(json!({"radius": 0.30, "room_size": 0.30}));
    assert!(m.contains("does not fit") && m.contains("room_size"), "{m}");
    assert!(msg(json!({"air_h": 0.009})).contains("air_h"));
    assert!(msg(json!({"sigma": -1.0})).contains("sigma"));
}
