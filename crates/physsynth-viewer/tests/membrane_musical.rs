//! The drum's musical controls through the payload builders — `docs/dev/membrane-mapping-plan.md`
//! §6: the drum page and the mallet page, which builds its drum through the drum page's code.
//!
//! The mapping is held by core's `tests/membrane_tuning.rs`, which runs the simulation. What is
//! held here is the wiring, and every bar that can listens to the OUTPUT rather than comparing the
//! reported `tuning` block with core: a build that ignored the mapping would still report it.

mod common;

use common::{base_params, decode_f32, f, ok, sim};
use physsynth_analysis::spectrum::measure_partials_near;
use physsynth_core::exciter::raised_cosine_2d;
use physsynth_core::membrane::{Domain, Membrane};
use physsynth_core::membrane_tuning::{head, tune_membrane, Geometry, SetBy, Targets, HEADS};
use serde_json::{json, Value};

// Every musical key OFF the server's fallback (`build_musical`'s defaults are the 7.5 mil head,
// f0 220, tension 2000, t60 2 / 0.5, f_ref 2000, sample_rate 44100, custom_rho 0.26, N 80,
// radius 0.5): a misspelled key reads the fallback, and a request that sent it could not tell.
const FS: f64 = 48000.0;
const HEAD: &str = "two-ply 7 + 7 mil";
const F0: f64 = 175.0;
const T60_LOW: f64 = 2.6;
const T60_HIGH: f64 = 0.55;
const F_REF: f64 = 1300.0;
const RADIUS: f64 = 0.21;
const N: i64 = 40;

/// A musical drum request, with the physics keys set to values the mapping must ignore.
fn musical(overrides: Value) -> Value {
    let mut p = base_params(json!({
        "model": "membrane", "domain": "circle", "controls": "musical",
        // Decoys: a musical build that read any of these would be a different drum.
        "T": 650.0, "rho": 0.9, "lambda": 0.31, "sigma": 4.0,
        "radius": RADIUS, "Lx": 0.5, "Ly": 0.37, "N": N,
        "head": HEAD, "set_by": "pitch", "f0": F0, "tension": 1234.0,
        "t60_low": T60_LOW, "t60_high": T60_HIGH, "f_ref": F_REF, "sample_rate": FS,
        "pluck_x": 0.37, "pluck_y": 0.58, "pluck_width": 0.4, "amplitude": 2e-3,
        "pickup_x": 0.62, "pickup_y": 0.66, "audio_duration": 1.2,
    }));
    common::merge(&mut p, overrides);
    p
}

fn circle() -> Geometry {
    Geometry {
        domain: Domain::Circle,
        lx: 0.0,
        ly: 0.0,
        radius: RADIUS,
        n: N,
        fs: FS,
    }
}

fn targets() -> Targets {
    Targets {
        set_by: SetBy::Pitch(F0),
        t60_low: T60_LOW,
        t60_high: T60_HIGH,
        f_ref: F_REF,
    }
}

fn rho() -> f64 {
    head(HEAD).unwrap().areal_density()
}

fn cents(f: f64, target: f64) -> f64 {
    1200.0 * (f / target).log2()
}

fn audio(payload: &Value) -> Vec<f64> {
    decode_f32(payload["audio"]["b64"].as_str().unwrap())
        .into_iter()
        .map(f64::from)
        .collect()
}

#[test]
fn the_musical_drum_builds_from_the_targets_not_the_sliders() {
    let payload = ok(&musical(json!({})));
    let t = &payload["tuning"];
    let want = tune_membrane(&circle(), rho(), &targets()).unwrap();
    assert!((f(&t["tension"]) - want.tension).abs() < 1e-3, "{t}");
    assert!((f(&t["rho"]) - rho()).abs() < 1e-6);
    assert!((f(&t["sigma0"]) - want.sigma0).abs() < 1e-6);
    assert!((f(&t["sigma1"]) - want.sigma1).abs() < 1e-9);
    assert_eq!(t["exact"], true);
    assert_eq!(t["head"], HEAD);
    assert_eq!(t["partials"].as_array().unwrap().len(), 8);
    assert_eq!(f(&payload["fs_sim"]), FS);
    // lambda is a read-out of the MAPPED wave speed, not the slider's 0.31
    let h = 2.0 * RADIUS / N as f64;
    let lam = want.params.c / FS / h;
    assert!(
        (f(&payload["lambda"]) - lam).abs() < 1e-5,
        "{}",
        payload["lambda"]
    );
}

#[test]
fn the_struck_drum_sounds_the_pitch_asked_for() {
    // The rendered audio through the analysis crate's detector — the output, not the report.
    for (domain, f0) in [("circle", 175.0), ("rectangle", 233.0)] {
        let payload = ok(&musical(json!({"domain": domain, "f0": f0})));
        let got = measure_partials_near(
            &audio(&payload),
            f(&payload["audio"]["fs"]),
            &[f0],
            Some(15.0),
        );
        let c = cents(got[0], f0);
        assert!(
            c.abs() < 0.5,
            "{domain}: the strike sounds {c} cents from {f0} Hz"
        );
    }
}

#[test]
fn the_energy_is_the_heads_and_both_losses_act() {
    // A wrong density leaves every pitch right and every energy wrong. Read the run's own first
    // energy against a membrane built here from the mapping and struck the way the payload strikes;
    // then its LAST energy, decayed through both losses at the mapped wave speed.
    let payload = ok(&musical(json!({})));
    let tu = tune_membrane(&circle(), rho(), &targets()).unwrap();
    let mut mem = Membrane::new(tu.params.clone());
    let p = mem.params().clone();
    let (cx, cy) = ((2.0 * 0.37 - 1.0) * RADIUS, (2.0 * 0.58 - 1.0) * RADIUS);
    let field = raised_cosine_2d(&p.x, &p.y, (cx, cy), 0.4 * RADIUS, 2e-3).unwrap();
    mem.set_displacement(&p.to_live(&field));
    let values = payload["energy"]["value"].as_array().unwrap();
    let times = payload["energy"]["time"].as_array().unwrap();
    let e0 = f(&values[0]);
    assert!(
        (e0 / mem.energy() - 1.0).abs() < 1e-12,
        "payload E0 {e0} against {}",
        mem.energy()
    );
    let steps = (f(times.last().unwrap()) * FS).round() as usize;
    for _ in 0..steps {
        mem.step();
    }
    let e_end = f(values.last().unwrap());
    assert!(
        e_end < 0.5 * e0,
        "the run lost too little to tell the losses apart"
    );
    assert!(
        (e_end / mem.energy() - 1.0).abs() < 1e-9,
        "payload E_end {e_end} against {} after {steps} steps",
        mem.energy()
    );
}

#[test]
fn musical_keys_do_not_touch_the_physical_drum() {
    // The front-end sends every slider to every model: a drum request without musical controls
    // must come back byte for byte whatever musical keys ride along.
    for model in ["membrane", "mallet"] {
        let plain = base_params(json!({
            "model": model, "domain": "circle", "T": 200.0, "rho": 0.005, "lambda": 0.6,
            "sigma": 0.5, "N": 24, "radius": 0.5, "audio_duration": 0.1,
        }));
        let mut noisy = plain.clone();
        common::merge(
            &mut noisy,
            json!({"controls": "physical", "head": "single-ply 3 mil", "set_by": "tension",
                   "tension": 900.0, "f0": 330.0, "t60_low": 9.0, "t60_high": 0.1,
                   "f_ref": 4000.0, "sample_rate": 96000.0, "custom_rho": 0.4}),
        );
        let (a, b) = (sim(&plain), sim(&noisy));
        assert!(a.get("error").is_none(), "{model}: {}", a["error"]);
        assert!(a.get("tuning").is_none());
        assert_eq!(a.to_string(), b.to_string(), "{model}");
    }
}

#[test]
fn set_by_tension_reads_the_pitch_out() {
    let payload = ok(&musical(json!({"set_by": "tension", "tension": 3100.0})));
    let t = &payload["tuning"];
    assert_eq!(t["set_by"], "tension");
    assert!((f(&t["tension"]) - 3100.0).abs() < 1e-9);
    assert!(t["f0_target"].is_null());
    // The continuum f01 = j01 c / (2 pi a); the staircased disk rings within a few percent.
    let c = (3100.0 / rho()).sqrt();
    let f01 = 2.404825557695773 * c / (2.0 * std::f64::consts::PI * RADIUS);
    let f0 = f(&t["f0"]);
    assert!(
        (f0 / f01 - 1.0).abs() < 0.03,
        "{f0} against the continuum {f01}"
    );
    assert!(f0 < f01, "the staircased rim rings flat of the continuum");
}

#[test]
fn the_energy_verdict_drops_the_single_rate_oracle_under_a_frequency_dependent_loss() {
    let payload = ok(&musical(json!({"audio_duration": 0.3})));
    let e = &payload["energy"];
    assert_eq!(e["sigma_is_zero"], false);
    assert!(e["lossy"].get("measured_2sigma").is_none(), "{e}");
    assert_eq!(e["lossy"]["monotone"], true);
}

#[test]
fn unreachable_settings_come_back_as_clean_errors() {
    let cases = [
        (json!({"sample_rate": 1000.0}), "sample_rate must be in"),
        (json!({"head": "calfskin"}), "head must be one of"),
        (
            json!({"set_by": "weight"}),
            "set_by must be 'pitch' or 'tension'",
        ),
        (json!({"t60_high": 8.0}), "out of reach"),
        (json!({"f_ref": 150.0}), "picks the fundamental"),
        (json!({"f0": 0.0}), "f0 must be positive"),
        (json!({"f0": 300.0, "sample_rate": 8000.0}), "CFL"),
        (json!({"N": 101}), "N must be in"),
    ];
    for (bad, needle) in cases {
        let payload = sim(&musical(bad.clone()));
        assert_eq!(payload["error"]["kind"], "param", "{bad}: {payload}");
        let msg = payload["error"]["message"].as_str().unwrap();
        assert!(msg.contains(needle), "{bad}: {msg}");
    }
}

#[test]
fn the_head_menu_is_the_cores_head_list() {
    // index.html lists the heads by hand; a head added or renamed in core without the menu
    // following would be unreachable from the browser, or a menu entry the server refuses.
    let html = std::fs::read_to_string(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../web/static/index.html"),
    )
    .unwrap();
    let start = html
        .find("<select id=\"drum-head\">")
        .expect("the menu moved");
    let body = &html[start..start + html[start..].find("</select>").unwrap()];
    let menu: Vec<&str> = body
        .split("<option value=\"")
        .skip(1)
        .map(|s| &s[..s.find('"').unwrap()])
        .collect();
    let mut want: Vec<&str> = HEADS.iter().map(|h| h.name).collect();
    want.push("custom");
    assert_eq!(menu, want);
}

#[test]
fn a_custom_density_builds_the_drum_it_describes() {
    // Off the fallback: 0.41 kg/m² (the server falls back to 0.26).
    let payload = ok(&musical(json!({"head": "custom", "custom_rho": 0.41})));
    let t = &payload["tuning"];
    let want = tune_membrane(&circle(), 0.41, &targets()).unwrap();
    assert_eq!(t["head"], "custom");
    assert!((f(&t["rho"]) - 0.41).abs() < 1e-9, "{t}");
    assert!((f(&t["tension"]) - want.tension).abs() < 1e-3, "{t}");
}

#[test]
fn the_mallet_strikes_the_musical_drum() {
    // The mallet page builds its drum through the drum page's code. The head starts at rest and
    // the mallet leaves after a millisecond or so; the ring-down sounds the mapped pitch.
    let payload = ok(&musical(json!({
        "model": "mallet", "audio_duration": 1.0, "mass": 0.03, "stiffness": 4.0e4,
        "strike_velocity": 2.0,
    })));
    let t = &payload["tuning"];
    assert_eq!(t["head"], HEAD);
    let want = tune_membrane(&circle(), rho(), &targets()).unwrap();
    assert!((f(&t["sigma1"]) - want.sigma1).abs() < 1e-9);
    assert_eq!(payload["energy"]["sigma_is_zero"], false);
    let got = measure_partials_near(
        &audio(&payload),
        f(&payload["audio"]["fs"]),
        &[F0],
        Some(15.0),
    );
    let c = cents(got[0], F0);
    assert!(
        c.abs() < 0.5,
        "the struck drum sounds {c} cents from {F0} Hz"
    );
}
