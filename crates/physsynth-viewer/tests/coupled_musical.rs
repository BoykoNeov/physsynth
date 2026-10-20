//! The strings on a bridge — the `body`, `radbody`, `sympathetic` and `platebody` pages — through
//! their musical controls (`docs/dev/string-mapping-plan.md` §16–§18).
//!
//! The tuning is held by core's `tests/coupled_tuning.rs`, which reads every tuned note off a run.
//! What is held here is the wiring, three ways:
//!
//! * the physics set is untouched: naming it, with every musical key riding along, changes no byte;
//! * with the ring time off, the musical page IS the physics page fed the tuned tension (the twin);
//! * the tuned note is read off a rig this file builds itself from the payload's tension and string
//!   loss and the request's other settings, all off their defaults — so a page that tuned one rig
//!   and ran another, or dropped a setting from the tuning's build, plays off the pitch here.
//!
//! Then the human's four options (plan §17), and one bar per refusal. The grid is `N = 40`, where a
//! tuning takes about a second (the two-string decay page's ring time, ~16 s, runs once).

mod common;

use std::f64::consts::{LN_10, PI};

use common::{base_params, f, merge, ok, sim};
use physsynth_core::body::{self as body, ModalBody};
use physsynth_core::connection::{StringBodyBridge, StringPlateBridge, SympatheticStrings};
use physsynth_core::coupled_tuning::{damped_mode, Coupled};
use physsynth_core::exciter::triangular_pluck;
use physsynth_core::plate::{self as pl, Boundary as PlateBoundary, Plate, PlateSpec};
use physsynth_core::radiation::RadiatedBody;
use physsynth_core::string_ideal::{Boundary, IdealString, Params};
use serde_json::{json, Value};

/// The rig's settings, every one off the page's default.
const L: f64 = 0.8;
const RHO: f64 = 0.004;
const LAM: f64 = 0.85;
const N: i64 = 40;
/// The bodies' modes, written out rather than imported: a page that swapped them would tune a
/// rig this file does not build.
const BODY: [f64; 4] = [110.0, 196.0, 261.0, 440.0];
const SYMP: [f64; 5] = [137.0, 213.0, 330.0, 471.0, 620.0];

/// The pages, each with its own settings off their defaults.
fn pages() -> Vec<(&'static str, Value)> {
    vec![
        (
            "body",
            json!({"model": "body", "bridge_stiffness": 6000.0, "sigma_body": 4.0}),
        ),
        (
            "radbody",
            json!({"model": "radbody", "bridge_stiffness": 7000.0, "radiation_R": 90.0,
                   "sigma_body": 3.0, "sweep_points": 4, "sweep_cap": 0.1}),
        ),
        // The normal and transfer regimes run a lossless body: their body-loss setting is a decoy.
        (
            "normal",
            json!({"model": "sympathetic", "domain": "normal", "K": 7000.0, "sigma_body": 9.0}),
        ),
        (
            "transfer",
            json!({"model": "sympathetic", "domain": "transfer", "K": 1800.0, "detune": 1.5,
                   "sigma_body": 9.0}),
        ),
        (
            "weinreich",
            json!({"model": "sympathetic", "domain": "weinreich", "K": 5000.0,
                   "sigma_body": 15.0, "detune": 0.2}),
        ),
        // The plate page on both edges, its plate grid and loss off their defaults.
        (
            "platebody",
            json!({"model": "platebody", "domain": "supported", "bridge_stiffness": 4500.0,
                   "sigma_plate": 5.0, "n_plate": 10}),
        ),
        (
            "platefree",
            json!({"model": "platebody", "domain": "free", "bridge_stiffness": 2500.0,
                   "n_plate": 9}),
        ),
    ]
}

/// A request: the shared rig, the page's settings, then `extra`. `T` is a decoy — at 750 N the
/// string sounds 342 Hz — that a musical build must not read.
fn request(page: &Value, extra: Value) -> Value {
    let mut p = base_params(
        json!({"L": L, "rho": RHO, "lambda": LAM, "N": N, "T": 750.0,
                                   "audio_duration": 0.2}),
    );
    merge(&mut p, page.clone());
    merge(&mut p, extra);
    p
}

fn musical(page: &Value, extra: Value) -> Value {
    let mut p = request(
        page,
        json!({"controls": "musical", "f0": 98.0, "t60_low": 1.0, "ring_time": true,
               "split": "play", "detune_tuning": "exact", "sweep_grid": "fixed"}),
    );
    merge(&mut p, extra);
    p
}

fn page(name: &str) -> Value {
    pages()
        .into_iter()
        .find(|(n, _)| *n == name)
        .map(|(_, p)| p)
        .expect("a page")
}

fn num(p: &Value, key: &str, default: f64) -> f64 {
    p.get(key).map_or(default, f)
}

/// The page's rig, built here from the payload's tension and string loss and the request's other
/// settings — not through the viewer.
enum Rig {
    Body(StringBodyBridge<ModalBody>),
    Rad(StringBodyBridge<RadiatedBody>),
    Two(SympatheticStrings),
    Plate(Box<StringPlateBridge<Plate>>),
}

fn string(t: f64, sigma: f64, fs: f64) -> IdealString {
    IdealString::new(
        Params::new(
            L,
            t,
            RHO,
            fs,
            N,
            sigma,
            Some((Boundary::Fixed, Boundary::Free)),
        )
        .expect("a string"),
    )
}

fn modal(freqs: &[f64], fs: f64, sigma: f64) -> ModalBody {
    let m = freqs.len();
    ModalBody::new(
        body::Params::new(
            freqs.to_vec(),
            fs,
            vec![sigma; m],
            vec![0.02; m],
            vec![1.0; m],
            None,
        )
        .expect("a body"),
    )
}

fn rig_of(req: &Value, tuning: &Value) -> Rig {
    let (t, sigma) = (f(&tuning["T"]), f(&tuning["sigma"]));
    let fs = (t / RHO).sqrt() * N as f64 / (L * LAM);
    match req["model"].as_str().unwrap() {
        "body" => Rig::Body(
            StringBodyBridge::new(
                string(t, sigma, fs),
                modal(&BODY, fs, num(req, "sigma_body", 0.0)),
                num(req, "bridge_stiffness", 8000.0),
            )
            .unwrap(),
        ),
        "platebody" => {
            let free = req["domain"].as_str() == Some("free");
            let spec = PlateSpec {
                lx: 1.0,
                ly: 1.0,
                kappa: 20.0,
                rho: 0.005,
                fs,
                n: num(req, "n_plate", 16.0) as i64,
                sigma: num(req, "sigma_plate", 0.0),
                boundary: Some(if free {
                    PlateBoundary::Free
                } else {
                    PlateBoundary::Supported
                }),
                nu: free.then_some(0.3),
                ..PlateSpec::default()
            };
            Rig::Plate(Box::new(
                StringPlateBridge::new(
                    string(t, sigma, fs),
                    Plate::new(pl::Params::new(&spec).unwrap()),
                    num(req, "bridge_stiffness", 3000.0),
                    None,
                )
                .unwrap(),
            ))
        }
        "radbody" => Rig::Rad(
            StringBodyBridge::new(
                string(t, sigma, fs),
                RadiatedBody::new(
                    modal(&BODY, fs, num(req, "sigma_body", 0.0)),
                    num(req, "radiation_R", 133.0),
                )
                .unwrap(),
                num(req, "bridge_stiffness", 8000.0),
            )
            .unwrap(),
        ),
        _ => {
            let domain = req["domain"].as_str().unwrap();
            let (detune, sigma_body) = match domain {
                "normal" => (0.0, 0.0),
                "transfer" => (num(req, "detune", 0.0), 0.0),
                _ => (num(req, "detune", 0.0), num(req, "sigma_body", 20.0)),
            };
            let tb = t * 2f64.powf(-detune / 6.0);
            let k = num(req, "K", 0.0);
            Rig::Two(
                SympatheticStrings::new(
                    vec![string(t, sigma, fs), string(tb, sigma, fs)],
                    modal(&SYMP, fs, sigma_body),
                    vec![k, k],
                )
                .unwrap(),
            )
        }
    }
}

/// `(pitch Hz, ring time s)` of the damped mode nearest the tuning's root, on this file's own rig.
fn note_on<R: Coupled>(r: &mut R, tuning: &Value) -> (f64, f64) {
    let root = (f(&tuning["root"][0]), f(&tuning["root"][1]));
    let ((re, im), _) = damped_mode(r, root).expect("the mode");
    let k = r.k();
    let a = -(re.hypot(im)).ln() / k;
    (im.atan2(re) / (2.0 * PI * k), 3.0 * LN_10 / a)
}

fn played(req: &Value, tuning: &Value) -> (f64, f64) {
    match rig_of(req, tuning) {
        Rig::Body(mut r) => note_on(&mut r, tuning),
        Rig::Rad(mut r) => note_on(&mut r, tuning),
        Rig::Two(mut r) => note_on(&mut r, tuning),
        Rig::Plate(mut r) => note_on(r.as_mut(), tuning),
    }
}

/// The run's last stored energy, on this file's own rig, plucked as the page plucks.
fn last_energy(req: &Value, tuning: &Value, steps: usize) -> f64 {
    let pluck = |x: &[f64]| {
        let u0 = triangular_pluck(x, L, num(req, "pluck_position", 0.3) * L, 1e-3).unwrap();
        let v0 = vec![0.0; u0.len()];
        (u0, v0)
    };
    match rig_of(req, tuning) {
        Rig::Body(mut r) => {
            let (u0, v0) = pluck(&r.string().params().grid());
            r.string_mut().set_state(&u0, &v0);
            for _ in 0..steps {
                let _ = r.step(&mut ());
            }
            r.energy()
        }
        Rig::Two(mut r) => {
            let (u0, v0) = pluck(&r.strings()[0].params().grid());
            r.string_mut(0).set_state(&u0, &v0);
            for _ in 0..steps {
                r.step();
            }
            r.energy()
        }
        Rig::Plate(mut r) => {
            let (u0, v0) = pluck(&r.string().params().grid());
            r.string_mut().set_state(&u0, &v0);
            for _ in 0..steps {
                let _ = r.step(&mut ());
            }
            r.energy()
        }
        Rig::Rad(_) => unreachable!("not asked"),
    }
}

// -- the physics set ---------------------------------------------------------------------------

#[test]
fn the_physics_set_is_untouched_by_the_musical_keys() {
    for (name, pg) in pages() {
        let plain = sim(&request(&pg, json!({})));
        assert!(plain.get("error").is_none(), "{name}: {}", plain["error"]);
        let named = sim(&request(
            &pg,
            json!({"controls": "physical", "f0": 300.0, "t60_low": 0.07, "ring_time": true,
                   "split": "refuse", "detune_tuning": "unison", "sweep_grid": "page"}),
        ));
        assert_eq!(plain, named, "{name}");
    }
}

// -- the twin: ring time off -------------------------------------------------------------------

#[test]
fn a_lossless_musical_page_is_the_physics_page_on_the_tuned_string() {
    for (name, pg) in pages() {
        let req = musical(&pg, json!({"ring_time": false}));
        let mut m = ok(&req);
        let tuning = m
            .as_object_mut()
            .unwrap()
            .remove("tuning")
            .expect("a tuning block");
        assert_eq!(tuning["sigma"], json!(0.0), "{name}");
        assert_eq!(tuning["t60"], Value::Null, "{name}");
        assert_eq!(tuning["ring_time"], json!(false), "{name}");
        let t = f(&tuning["T"]);
        // The control: the decoy tension would have played another string.
        assert!((t - 750.0).abs() > 1.0, "{name}: T = {t}");
        let phys = ok(&request(&pg, json!({"controls": "physical", "T": t})));
        assert_eq!(m, phys, "{name}");
        // And the note the solve put on the pitch plays it on this file's rig.
        let (fp, _) = played(&req, &tuning);
        assert!((fp / 98.0 - 1.0).abs() < 1e-9, "{name}: plays {fp}");
    }
}

// -- the ring time -----------------------------------------------------------------------------

/// Every page with the ring time on, but the two-string decay page (its own bar, below): the note
/// plays the pitch and rings the time on this file's rig, and the page's run is that rig's run.
#[test]
fn the_tuned_note_plays_and_rings_on_the_pages_own_rig() {
    for name in [
        "body",
        "radbody",
        "normal",
        "transfer",
        "platebody",
        "platefree",
    ] {
        let req = musical(&page(name), json!({}));
        let payload = ok(&req);
        let tuning = &payload["tuning"];
        assert!(f(&tuning["sigma"]) > 0.0, "{name}");
        let (fp, t60) = played(&req, tuning);
        assert!((fp / 98.0 - 1.0).abs() < 1e-9, "{name}: plays {fp}");
        assert!((t60 - 1.0).abs() < 1e-6, "{name}: rings {t60}");
        // The energy panel judges a lossy run.
        assert_eq!(payload["energy"]["sigma_is_zero"], json!(false), "{name}");
        if matches!(name, "body" | "transfer" | "platebody") {
            let steps = payload["meta"]["num_steps"].as_i64().unwrap() as usize;
            let values = payload["energy"]["value"].as_array().unwrap();
            let want = last_energy(&req, tuning, steps);
            let got = f(&values[values.len() - 1]);
            assert!((got / want - 1.0).abs() < 1e-12, "{name}: {got} vs {want}");
        }
    }
}

/// The two-string decay page tunes through its own build: the body-loss slider and the detune.
#[test]
fn the_decay_page_tunes_on_its_lossy_body() {
    let req = musical(&page("weinreich"), json!({}));
    let payload = ok(&req);
    let (fp, t60) = played(&req, &payload["tuning"]);
    assert!((fp / 98.0 - 1.0).abs() < 1e-9, "plays {fp}");
    assert!((t60 - 1.0).abs() < 1e-6, "rings {t60}");
}

// -- the human's options -----------------------------------------------------------------------

/// 196 Hz on the body's second mode: the note on the pitch carries a ninth of the string's motion,
/// and louder notes sit beside it (plan §16's probe; at `N = 40` the loudest is the 325 Hz one).
fn split_request(split: &str, f0: f64) -> Value {
    let mut p = base_params(
        json!({"model": "body", "L": 1.0, "rho": 0.005, "lambda": 0.9,
                                   "N": N, "audio_duration": 0.2, "T": 750.0}),
    );
    merge(
        &mut p,
        json!({"controls": "musical", "f0": f0, "ring_time": false, "split": split}),
    );
    p
}

#[test]
fn a_split_band_plays_by_default_and_is_refused_when_asked() {
    let payload = ok(&split_request("play", 196.0));
    let t = &payload["tuning"];
    assert_eq!(t["split"], json!(true));
    assert!(f(&t["louder"]["share"]) > f(&t["note"]["share"]));
    let loud = format!("{} Hz", (f(&t["louder"]["f"]) * 100.0).round() / 100.0);
    let refused = sim(&split_request("refuse", 196.0));
    let msg = refused["error"]["message"].as_str().expect("refused");
    assert!(
        msg.contains("split band") && msg.contains(&loud),
        "{msg} / {loud}"
    );
    // The control: a note with no louder neighbour renders under the same option.
    let clean = ok(&split_request("refuse", 98.0));
    assert_eq!(clean["tuning"]["split"], json!(false));
}

#[test]
fn the_detune_is_solved_in_or_applied_after() {
    let exact_req = musical(&page("transfer"), json!({"ring_time": false}));
    let exact = ok(&exact_req);
    let after_req = musical(
        &page("transfer"),
        json!({"ring_time": false, "detune_tuning": "unison"}),
    );
    let after = ok(&after_req);
    let (te, ta) = (&exact["tuning"], &after["tuning"]);
    assert_eq!(te["off_pitch"], json!(false));
    assert_eq!(ta["off_pitch"], json!(true));
    assert_ne!(te["T"], ta["T"]);
    let (fe, _) = played(&exact_req, te);
    assert!((fe / 98.0 - 1.0).abs() < 1e-9, "solved in: plays {fe}");
    // Tuned as a unison, the detuned pair plays off the pitch — the drift the page warns of.
    let (fa, _) = played(&after_req, ta);
    assert!((fa / 98.0 - 1.0).abs() > 1e-4, "applied after: plays {fa}");
    // With no detune the two settings are one tuning.
    let unison = |mode: &str| {
        ok(&musical(
            &page("transfer"),
            json!({"ring_time": false, "detune": 0.0, "detune_tuning": mode}),
        ))["tuning"]["T"]
            .clone()
    };
    assert_eq!(unison("exact"), unison("unison"));
}

#[test]
fn the_air_sweep_runs_on_the_fixed_grid_or_the_pages() {
    let sweep = |grid: &str| {
        let p = ok(&musical(
            &page("radbody"),
            json!({"ring_time": false, "sweep_grid": grid}),
        ));
        (
            p["tuning"]["sweep_n"].clone(),
            p["meta"]["spectrum"]["sweep"].clone(),
        )
    };
    let (n_fixed, fixed) = sweep("fixed");
    let (n_page, own) = sweep("page");
    assert_eq!(n_fixed, json!(100));
    assert_eq!(fixed["sweep_n"], json!(100));
    assert_eq!(n_page, json!(N));
    assert_eq!(own["sweep_n"], json!(N));
    assert_ne!(fixed["t50_ms"], own["t50_ms"]);
}

// -- the refusals ------------------------------------------------------------------------------

fn refusal(p: &Value) -> String {
    let r = sim(p);
    r["error"]["message"]
        .as_str()
        .unwrap_or_else(|| panic!("not refused: {}", r["tuning"]))
        .to_owned()
}

#[test]
fn a_ring_time_under_the_floor_is_refused() {
    let msg = refusal(&musical(&page("body"), json!({"t60_low": 0.03})));
    assert!(msg.contains("at least 0.05 s"), "{msg}");
}

#[test]
fn a_malformed_option_is_refused() {
    let msg = refusal(&musical(&page("body"), json!({"ring_time": "yes"})));
    assert!(msg.contains("ring_time must be true or false"), "{msg}");
    let msg = refusal(&musical(&page("body"), json!({"split": "maybe"})));
    assert!(msg.contains("split must be one of"), "{msg}");
    let msg = refusal(&musical(&page("transfer"), json!({"detune_tuning": "x"})));
    assert!(msg.contains("detune_tuning must be one of"), "{msg}");
    let msg = refusal(&musical(&page("radbody"), json!({"sweep_grid": "x"})));
    assert!(msg.contains("sweep_grid must be one of"), "{msg}");
}

/// Two runs of 2 s at `N = 100`: even the slackest string that sounds 300 Hz overruns the budget,
/// so the page says so before a tuning that would take minutes.
#[test]
fn a_run_over_budget_at_any_tension_is_refused_before_tuning() {
    let p = musical(
        &page("normal"),
        json!({"N": 100, "f0": 300.0, "audio_duration": 2.0, "lambda": 0.9}),
    );
    let start = std::time::Instant::now();
    let msg = refusal(&p);
    assert!(
        msg.contains("before tuning") && msg.contains("No pitch above"),
        "{msg}"
    );
    assert!(start.elapsed().as_secs_f64() < 1.0, "it tuned first");
}

/// The plate page counts its budget in node-steps (every step updates the whole plate): at its
/// largest grids 500 Hz overruns it at any tension, and the page says so before computing the
/// plate's modes, which take seconds there.
#[test]
fn the_plate_page_refuses_over_budget_before_tuning() {
    let p = musical(
        &page("platefree"),
        json!({"N": 160, "n_plate": 24, "f0": 500.0, "audio_duration": 3.0, "lambda": 0.9}),
    );
    let start = std::time::Instant::now();
    let msg = refusal(&p);
    assert!(
        msg.contains("before tuning") && msg.contains("node-steps") && msg.contains("n_plate"),
        "{msg}"
    );
    assert!(start.elapsed().as_secs_f64() < 2.0, "it tuned first");
}

/// 196 Hz on the body at `N = 100`: the slackest string that could sound it fits 3 s, the tuned
/// one (its clamped note at 392 Hz) does not — refused after the tuning, in the musical controls'
/// terms.
#[test]
fn a_tuned_string_over_budget_is_refused_in_the_musical_terms() {
    let mut p = split_request("play", 196.0);
    merge(&mut p, json!({"N": 100, "audio_duration": 3.0}));
    let msg = refusal(&p);
    assert!(msg.contains("3 s of audio does not fit"), "{msg}");
    assert!(!msg.contains("tension"), "{msg}");
}

/// The ring-time toggle starts off (the human's eleventh answer, plan §17): a musical request that
/// does not name it tunes a lossless string, however its ring-time slider is set.
#[test]
fn the_ring_time_is_off_unless_asked() {
    let mut p = musical(&page("body"), json!({}));
    p.as_object_mut().unwrap().remove("ring_time");
    let t = &ok(&p)["tuning"];
    assert_eq!(t["ring_time"], json!(false));
    assert_eq!(t["sigma"], json!(0.0));
}
