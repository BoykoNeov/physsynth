//! The bow's and the tension string's musical controls through the payload builder —
//! `docs/dev/string-mapping-plan.md` §4, §5 and §7 (slice 1).
//!
//! The mappings are held by core's `tests/string_tuning.rs` and `tests/string_tension_tuning.rs`,
//! which run the simulation. What is held here is the wiring: each page builds from the targets
//! and not from the physics sliders the front-end still sends (judged by what the page PLAYS, not
//! by the `tuning` block, which reports what was mapped whether or not it was used — parameter
//! mapping plan §9), the physics controls are untouched by the musical keys, and the refusals
//! reach the payload.

mod common;

use common::{base_params, f, ok, sim};
use physsynth_core::string_tuning::{
    string_material, tune_damped_string, tune_tension_string, Targets,
};
use serde_json::{json, Value};

/// Decoys: a musical build that read any of these would play a different string — at `L = 1`,
/// `T = 750` and `rho = 0.005` the flexible string sounds 194 Hz, far from every target below.
fn decoys() -> Value {
    json!({"T": 750.0, "kappa": 6.0, "sigma0": 3.0, "sigma1": 0.009, "lambda": 0.6, "EA": 5.0})
}

fn request(model: &str, overrides: Value) -> Value {
    let mut p = base_params(json!({"model": model, "controls": "musical"}));
    common::merge(&mut p, decoys());
    common::merge(&mut p, overrides);
    p
}

fn bow(overrides: Value) -> Value {
    let mut p = request(
        "bow",
        json!({
            "N": 100, "L": 1.0, "rho": 0.005, "theta": 0.28,
            "f0": 110.0, "inharmonicity": 1e-4, "t60_low": 3.0, "t60_high": 0.8,
            "f_ref": 2000.0, "sample_rate": 22050.0, "audio_duration": 1.3,
            "pickup_position": 0.33,
        }),
    );
    common::merge(&mut p, overrides);
    p
}

fn tension(overrides: Value) -> Value {
    let mut p = request(
        "tension",
        json!({
            "domain": "duffing", "N": 64, "L": 0.65, "theta": 0.28,
            "string_material": "steel", "diameter_mm": 0.5,
            "f0": 196.0, "inharmonicity": 1e-4, "t60_low": 3.0, "t60_high": 0.8,
            "f_ref": 2000.0, "sample_rate": 44100.0, "amplitude": 0.003,
            "audio_duration": 0.3,
        }),
    );
    common::merge(&mut p, overrides);
    p
}

fn targets(f0: f64) -> Targets {
    Targets {
        f0,
        inharmonicity: 1e-4,
        t60_low: 3.0,
        t60_high: 0.8,
        f_ref: 2000.0,
    }
}

// -- the bow ----------------------------------------------------------------------------------

#[test]
fn the_musical_bow_builds_its_string_from_the_targets() {
    let payload = ok(&bow(json!({})));
    let t = &payload["tuning"];
    let want = tune_damped_string(1.0, 0.005, 22050.0, 100, 0.28, &targets(110.0)).unwrap();
    assert_eq!(f(&t["T"]), want.t);
    assert_eq!(f(&t["kappa"]), want.kappa);
    assert_eq!(f(&t["sigma0"]), want.sigma0);
    assert_eq!(f(&t["sigma1"]), want.sigma1);
    assert_eq!(f(&payload["fs_sim"]), 22050.0);
    assert_eq!(f(&payload["meta"]["f1"]), 110.0);
    assert!(t["bow_note"].as_str().unwrap().contains("moves with N"));
}

#[test]
fn the_musical_bow_plays_near_the_target() {
    // What the page plays, not what it reports. The probe (plan §9) read +3.9 cents for this
    // string; a build that read the decoy T would bow near 194 Hz, ~980 cents away.
    let payload = ok(&bow(json!({})));
    let sp = &payload["meta"]["spectrum"];
    assert_eq!(sp["helmholtz"], true, "{sp}");
    let cents = f(&sp["pitch_cents"]);
    println!("musical bow at 110 Hz, N = 100, 22.05 kHz: {cents:+.2} cents");
    assert!(cents.abs() < 10.0, "bowed {cents} cents off the target");
    assert!((f(&sp["f_detected"]) - 110.0).abs() < 1.0);
}

#[test]
fn the_bow_defaults_to_a_rate_its_work_budget_holds() {
    // No sample_rate in the request: 22.05 kHz, so the page's own 2 s default run fits the
    // 60,000-step budget; 44.1 kHz would be refused.
    let mut p = bow(json!({"audio_duration": 2.0}));
    p.as_object_mut().unwrap().remove("sample_rate");
    assert_eq!(f(&ok(&p)["fs_sim"]), 22050.0);
    let at_44k = sim(&bow(json!({"audio_duration": 2.0, "sample_rate": 44100.0})));
    let msg = at_44k["error"]["message"].as_str().unwrap();
    // and it names the lever this control set has: the sample rate, not the grid
    assert!(
        msg.contains("work budget") && msg.contains("Lower audio_duration or the sample rate"),
        "{msg}"
    );
}

#[test]
fn musical_keys_do_not_touch_the_physical_bow() {
    let plain = base_params(json!({"model": "bow", "N": 64, "audio_duration": 0.3}));
    let mut noisy = plain.clone();
    common::merge(
        &mut noisy,
        json!({"controls": "physical", "f0": 330.0, "inharmonicity": 0.01, "t60_low": 9.0,
               "t60_high": 0.1, "f_ref": 4000.0, "sample_rate": 96000.0}),
    );
    let (a, b) = (sim(&plain), sim(&noisy));
    assert!(a.get("error").is_none(), "{}", a["error"]);
    assert!(a.get("tuning").is_none());
    assert_eq!(a.to_string(), b.to_string());
}

// -- the tension string -------------------------------------------------------------------------

#[test]
fn the_musical_tension_string_builds_from_its_gauge_and_targets() {
    let payload = ok(&tension(json!({})));
    let t = &payload["tuning"];
    let steel = string_material("steel").unwrap();
    let want =
        tune_tension_string(steel, 0.5e-3, 0.65, 44100.0, 64, 0.28, &targets(196.0)).unwrap();
    assert_eq!(f(&t["T"]), want.tuning.t);
    assert_eq!(f(&t["kappa"]), want.tuning.kappa);
    assert_eq!(f(&t["sigma0"]), want.tuning.sigma0);
    let g = &t["gauge"];
    assert_eq!(g["material"], "steel");
    assert_eq!(f(&g["rho"]), want.rho);
    assert_eq!(f(&g["EA"]), want.ea);
    assert!(g["source"].as_str().unwrap().contains("A228"));
    assert_eq!(f(&payload["fs_sim"]), 44100.0);
    // EA/T is the gauge's, not the decoy slider's 5 kN.
    let ratio = f(&payload["meta"]["EA_over_T"]);
    assert!(
        (ratio - want.ea / want.tuning.t).abs() < 1e-2,
        "EA/T {ratio}"
    );
}

#[test]
fn the_musical_tension_string_settles_on_the_target_and_starts_sharp() {
    // The page's gentle reference run (1e-5 m, lossless) is the settled note; the 3 mm run is the
    // start. Both are what the page PLAYS.
    let payload = ok(&tension(json!({})));
    let sp = &payload["meta"]["spectrum"];
    let settled = f(&sp["f_reference"]);
    let off = 1200.0 * (settled / 196.0).log2();
    println!(
        "tension, steel 0.5 mm at 196 Hz: settled {off:+.4} cents, 3 mm start {:+.2} cents",
        f(&sp["shift_cents"])
    );
    assert!(
        off.abs() < 0.02,
        "settled note {settled} Hz, {off} cents off"
    );
    assert!(f(&sp["shift_cents"]) > 5.0, "{sp}");
}

/// The musical tension page measures exactly what the physics page measures on the string the
/// mapping chose: its `T`, `rho`, `EA` and `kappa` typed into the physics sliders, `lambda` set so
/// that `fs` is the musical sample rate. So the musical page's single-mode runs are the physics
/// page's lossless ones (the physics twin zeroes its `sigma` keys; the musical build zeroes its
/// mapped losses), and nothing else differs.
///
/// Found writing this (2026-10-19): at this gentle 3 mm pluck both pages read the start shift
/// 2.7 % below the Duffing closed form, 1.403 Hz against 1.443. The physics page's own bars run
/// a 2 cm pluck, whose shift is ~20 times larger; the gap is the page's 12-period measurement,
/// not the musical wiring, so no musical bar bounds it.
#[test]
fn the_musical_tension_page_is_the_physics_page_on_the_mapped_string() {
    let m = ok(&tension(json!({})));
    let t = &m["tuning"];
    let (tt, rho, ea) = (f(&t["T"]), f(&t["gauge"]["rho"]), f(&t["gauge"]["EA"]));
    let lam = (tt / rho).sqrt() * 64.0 / (0.65 * 44100.0);
    let twin = ok(&base_params(json!({
        "model": "tension", "domain": "duffing", "N": 64, "L": 0.65, "theta": 0.28,
        "T": tt, "rho": rho, "EA": ea, "kappa": f(&t["kappa"]),
        "sigma0": f(&t["sigma0"]), "sigma1": f(&t["sigma1"]), "lambda": lam,
        "amplitude": 0.003, "audio_duration": 0.3,
    })));
    let (a, b) = (&m["meta"]["spectrum"], &twin["meta"]["spectrum"]);
    for key in [
        "f_reference",
        "f_hardened",
        "f_linear",
        "shift_measured",
        "shift_oracle",
        "dT_over_T",
    ] {
        assert!(
            (f(&a[key]) / f(&b[key]) - 1.0).abs() < 1e-9,
            "{key}: musical {} vs physics {}",
            a[key],
            b[key]
        );
    }
    // The control differs from its subject: the decoy sliders alone play another string.
    let decoyed = ok(&base_params(json!({
        "model": "tension", "domain": "duffing", "N": 64, "L": 0.65, "theta": 0.28,
        "T": 750.0, "rho": 0.005, "EA": 5000.0, "kappa": 6.0, "lambda": 0.6,
        "amplitude": 0.003, "audio_duration": 0.3,
    })));
    let d = f(&decoyed["meta"]["spectrum"]["f_reference"]);
    assert!(
        (d / f(&a["f_reference"]) - 1.0).abs() > 0.05,
        "decoy plays {d}"
    );
}

#[test]
fn the_material_moves_the_start_not_the_settled_note() {
    let (steel, gut) = (
        ok(&tension(json!({}))),
        ok(&tension(json!({"string_material": "gut"}))),
    );
    let (ss, sg) = (&steel["meta"]["spectrum"], &gut["meta"]["spectrum"]);
    let cents = |v: &Value| 1200.0 * (f(v) / 196.0).log2();
    assert!(cents(&ss["f_reference"]).abs() < 0.02 && cents(&sg["f_reference"]).abs() < 0.02);
    assert!(
        f(&ss["shift_cents"]) > 3.0 * f(&sg["shift_cents"]),
        "steel {} vs gut {}",
        ss["shift_cents"],
        sg["shift_cents"]
    );
    assert_eq!(gut["tuning"]["gauge"]["material"], "gut");
}

#[test]
fn musical_keys_do_not_touch_the_physical_tension_string() {
    let plain = base_params(json!({"model": "tension", "N": 64, "audio_duration": 0.2,
                                   "amplitude": 0.02, "kappa": 1.0}));
    let mut noisy = plain.clone();
    common::merge(
        &mut noisy,
        json!({"controls": "physical", "f0": 330.0, "inharmonicity": 0.01, "t60_low": 9.0,
               "t60_high": 0.1, "f_ref": 4000.0, "sample_rate": 96000.0,
               "string_material": "nylon", "diameter_mm": 1.0}),
    );
    let (a, b) = (sim(&plain), sim(&noisy));
    assert!(a.get("error").is_none(), "{}", a["error"]);
    assert!(a.get("tuning").is_none());
    assert_eq!(a.to_string(), b.to_string());
}

#[test]
fn stress_past_the_data_or_the_breaking_point_is_warned_and_still_rendered() {
    // The human's call (plan §12): warn, keep running. A nylon string at 1 kHz on 0.65 m runs at
    // about 1,800 MPa — past nylon's ~280 MPa breaking stress and far outside the 30–220 MPa its
    // stiffness was measured at.
    let hot = ok(&tension(
        json!({"string_material": "nylon", "f0": 1000.0, "f_ref": 4000.0,
                                 "amplitude": 0.001}),
    ));
    let g = &hot["tuning"]["gauge"];
    assert_eq!(g["modulus_extrapolated"], true, "{g}");
    assert_eq!(g["past_breaking"], true, "{g}");
    let w: Vec<&str> = g["warnings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(w.len(), 2, "{w:?}");
    assert!(w[0].contains("outside the 30–220 MPa"), "{}", w[0]);
    assert!(
        w[1].contains("would break") && w[1].contains("280 MPa"),
        "{}",
        w[1]
    );
    assert!(hot["audio"]["n"].as_i64().unwrap() > 0);
    // The ordinary steel string of the other bars carries neither.
    let calm = ok(&tension(json!({})));
    assert_eq!(calm["tuning"]["gauge"]["warnings"], json!([]));
}

#[test]
fn unrepresentable_musical_requests_come_back_as_clean_errors() {
    let cases = [
        (
            tension(json!({"diameter_mm": 0.05})),
            "diameter_mm must be in",
        ),
        (
            tension(json!({"string_material": "silk"})),
            "string_material must be one of",
        ),
        (tension(json!({"diameter_mm": 1.5})), "above the page's"),
        (tension(json!({"domain": "parametric"})), "Duffing regime"),
        (tension(json!({"t60_high": 8.0})), "negative loss"),
        (
            bow(json!({"sample_rate": 1000.0})),
            "sample_rate must be in",
        ),
        (bow(json!({"force": 9.0})), "force must be in"),
        // the musical refusals name the controls this set shows, not EA or N
        (
            tension(json!({"amplitude": 0.06})),
            "choose a thinner or softer string",
        ),
        (
            tension(json!({"f0": 55.0, "audio_duration": 1.2})),
            "Lower audio_duration or the sample rate",
        ),
    ];
    for (req, needle) in cases {
        let payload = sim(&req);
        assert_eq!(payload["error"]["kind"], "param", "{needle}: {payload}");
        let msg = payload["error"]["message"].as_str().unwrap();
        assert!(msg.contains(needle), "{needle}: {msg}");
    }
}
