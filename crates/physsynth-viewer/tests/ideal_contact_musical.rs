//! The ideal string's and the barrier pages' musical controls through the payload builder —
//! `docs/dev/string-mapping-plan.md` §13 (slice 2).
//!
//! The mappings are held by core's `tests/string_ideal_tuning.rs`, which runs the simulation. What
//! is held here is the wiring, judged by what each page PLAYS (parameter mapping plan §9): the
//! ideal page's rendered pluck sounds its target, and each barrier page in musical mode is the
//! physics page fed the mapped string — the twin that held the tension page (plan §10).

mod common;

use common::{base_params, f, ok, sim};
use physsynth_core::string_stiff::THETA_DEFAULT;
use physsynth_core::string_tuning::{tune_flexible_string, tune_ideal_string};
use serde_json::{json, Value};

/// Decoys: a musical build that read any of these would play another string — at `L = 1`,
/// `T = 750` and `rho = 0.005` the flexible string sounds 194 Hz.
fn decoys() -> Value {
    json!({"T": 750.0, "sigma": 3.0, "sigma0": 3.0, "lambda": 0.6})
}

fn request(model: &str, overrides: Value) -> Value {
    let mut p = base_params(json!({"model": model, "controls": "musical"}));
    common::merge(&mut p, decoys());
    common::merge(&mut p, overrides);
    p
}

/// Numbers within `rel` of each other, everything else equal, recursively; `b64` strings (float32
/// renderings of numbers that may differ in their last bits) are skipped.
fn assert_close(a: &Value, b: &Value, rel: f64, path: &str) {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            let (kx, ky): (Vec<_>, Vec<_>) = (x.keys().collect(), y.keys().collect());
            assert_eq!(kx, ky, "{path}: keys differ");
            for (k, v) in x {
                if k != "b64" {
                    assert_close(v, &y[k], rel, &format!("{path}.{k}"));
                }
            }
        }
        (Value::Array(x), Value::Array(y)) => {
            assert_eq!(x.len(), y.len(), "{path}: lengths differ");
            for (i, (u, v)) in x.iter().zip(y).enumerate() {
                assert_close(u, v, rel, &format!("{path}[{i}]"));
            }
        }
        (Value::Number(_), Value::Number(_)) => {
            let (u, v) = (f(a), f(b));
            let scale = u.abs().max(v.abs()).max(1e-300);
            assert!((u - v).abs() <= rel * scale, "{path}: {u} vs {v}");
        }
        _ => assert_eq!(a, b, "{path}"),
    }
}

// -- the ideal string -------------------------------------------------------------------------

fn ideal(overrides: Value) -> Value {
    let mut p = request(
        "ideal",
        json!({"N": 64, "L": 0.65, "rho": 0.0012, "f0": 196.0, "t60_low": 2.0,
               "audio_duration": 1.0}),
    );
    common::merge(&mut p, overrides);
    p
}

#[test]
fn the_musical_ideal_string_builds_from_its_targets_at_lambda_one() {
    let payload = ok(&ideal(json!({})));
    let t = &payload["tuning"];
    let want = tune_ideal_string(0.65, 0.0012, 64, 196.0, 2.0).unwrap();
    assert_eq!(t["kind"], "ideal");
    assert_eq!(f(&t["T"]), want.t);
    assert_eq!(f(&t["sigma0"]), want.sigma);
    assert_eq!(f(&t["fs"]), want.fs);
    // The sample rate follows the pitch: 2 f0 N, pulled a hair up by the loss.
    assert!((f(&payload["fs_sim"]) - want.fs).abs() < 1e-3);
    assert!(
        (f(&payload["lambda"]) - 1.0).abs() < 1e-6,
        "{}",
        payload["lambda"]
    );
    assert_eq!(f(&payload["meta"]["f1"]), 196.0);
}

#[test]
fn the_musical_ideal_string_plays_its_partials() {
    // What the page plays: its own partial overlay, against m f0. A build that read the decoy T
    // would sound 1.7 octaves away.
    let payload = ok(&ideal(json!({})));
    let cents = payload["meta"]["partials"]["cents"].as_array().unwrap();
    for (m, c) in cents.iter().enumerate().take(4) {
        assert!(f(c).abs() < 0.5, "partial {}: {} cents", m + 1, c);
    }
    assert_eq!(payload["tuning"]["warn"], false, "{}", payload["tuning"]);
}

#[test]
fn the_musical_ideal_string_rings_its_ring_time() {
    // What the page plays, for the loss: the energy panel's own decay fit. Every partial of the
    // ideal string rings the same T60, so the energy falls at 2 x 3 ln 10 / T60 (amplitude 60 dB in
    // T60). The tuning block cannot stand in for this — it reports what was mapped, whether or not
    // the string was built from it (found 2026-10-19: a build that read the sigma slider passed
    // every other bar here).
    let payload = ok(&ideal(json!({})));
    let measured = f(&payload["energy"]["lossy"]["measured_2sigma"]);
    let want = 6.0 * 10f64.ln() / 2.0;
    println!("ideal 196 Hz, T60 2 s: energy decays at {measured:.6}/s, target {want:.6}/s");
    assert!((measured / want - 1.0).abs() < 1e-3, "{measured} vs {want}");
}

#[test]
fn a_heavily_damped_ideal_string_warns_its_partials_run_sharp() {
    let payload = ok(&ideal(json!({"N": 16, "L": 1.0, "rho": 0.005, "f0": 27.5,
                                  "t60_low": 0.1, "audio_duration": 0.5})));
    let t = &payload["tuning"];
    assert_eq!(t["warn"], true, "{t}");
    let worst = f(&t["worst_drift_cents"]);
    assert!(worst > 90.0 && worst < 140.0, "{worst}");
    assert!(t["message"].as_str().unwrap().contains("run sharp"), "{t}");
}

#[test]
fn musical_keys_do_not_touch_the_physical_ideal_string() {
    let plain = base_params(json!({"model": "ideal"}));
    let mut noisy = plain.clone();
    common::merge(
        &mut noisy,
        json!({"controls": "physical", "f0": 330.0, "t60_low": 0.2, "sample_rate": 96000.0}),
    );
    let (a, b) = (sim(&plain), sim(&noisy));
    assert!(a.get("error").is_none(), "{}", a["error"]);
    assert!(a.get("tuning").is_none());
    assert_eq!(a.to_string(), b.to_string());
}

// -- the barrier pages ------------------------------------------------------------------------

/// Each barrier page at a musical setting it accepts, with a short run.
fn barrier_cases() -> Vec<(&'static str, Value)> {
    vec![
        ("jawari", json!({"N": 64, "audio_duration": 0.2})),
        (
            "juari",
            json!({"N": 64, "audio_duration": 0.2, "thread_position": 0.1}),
        ),
        ("fret", json!({"N": 64, "audio_duration": 0.2})),
    ]
}

fn barrier(model: &str, extra: &Value) -> Value {
    let mut p = request(
        model,
        json!({"L": 1.0, "rho": 0.005, "f0": 110.0, "t60_low": 3.0, "sample_rate": 44100.0}),
    );
    common::merge(&mut p, extra.clone());
    p
}

#[test]
fn each_barrier_page_in_musical_mode_is_its_physics_page_on_the_mapped_string() {
    for (model, extra) in barrier_cases() {
        let m = ok(&barrier(model, &extra));
        let t = &m["tuning"];
        assert_eq!(t["kind"], "flexible", "{model}");
        let want =
            tune_flexible_string(1.0, 0.005, 44100.0, 64, THETA_DEFAULT, 110.0, 3.0).unwrap();
        assert_eq!(f(&t["T"]), want.t, "{model}");
        assert_eq!(f(&t["sigma0"]), want.sigma0, "{model}");
        // The physics page fed the mapped tension and loss, at the lambda that gives 44.1 kHz.
        let mut twin = base_params(json!({"model": model, "L": 1.0, "rho": 0.005, "T": want.t,
                                          "sigma0": want.sigma0, "lambda": want.lam}));
        common::merge(&mut twin, extra.clone());
        let physics = ok(&twin);
        let mut musical = m.clone();
        musical.as_object_mut().unwrap().remove("tuning");
        assert_close(&musical, &physics, 1e-9, model);
        // The control differs from its subject: the decoy sliders alone play another string.
        let mut decoyed = base_params(json!({"model": model}));
        common::merge(&mut decoyed, decoys());
        common::merge(&mut decoyed, extra.clone());
        let d = ok(&decoyed);
        assert_ne!(d["fs_sim"], m["fs_sim"], "{model}");
    }
}

#[test]
fn musical_keys_do_not_touch_the_physical_barrier_pages() {
    for (model, extra) in barrier_cases() {
        // lambda 0.4, the pages' own default: the shared base's 1.0 is refused by the contact
        // solve.
        let mut plain = base_params(json!({"model": model, "lambda": 0.4}));
        common::merge(&mut plain, extra.clone());
        let mut noisy = plain.clone();
        common::merge(
            &mut noisy,
            json!({"controls": "physical", "f0": 330.0, "t60_low": 0.2, "sample_rate": 96000.0}),
        );
        let (a, b) = (sim(&plain), sim(&noisy));
        assert!(a.get("error").is_none(), "{model}: {}", a["error"]);
        assert!(a.get("tuning").is_none(), "{model}");
        assert_eq!(a.to_string(), b.to_string(), "{model}");
    }
}

#[test]
fn a_barrier_pitch_too_high_for_its_grid_says_what_to_change() {
    // 440 Hz on N = 100 at 44.1 kHz needs lambda ~ 2; the contact solve needs lambda < 1.
    for model in ["jawari", "juari", "fret"] {
        let payload = sim(&barrier(model, &json!({"N": 100, "f0": 440.0})));
        assert_eq!(payload["error"]["kind"], "param", "{model}: {payload}");
        let msg = payload["error"]["message"].as_str().unwrap();
        assert!(
            msg.contains("raise the sample rate above"),
            "{model}: {msg}"
        );
    }
}
