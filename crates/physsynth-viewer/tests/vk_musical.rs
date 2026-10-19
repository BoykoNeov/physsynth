//! The gong's and the cymbal's musical controls through the payload builder, and the guard both
//! control sets share — `docs/dev/vk-mapping-plan.md` §6, §8.8, §11.
//!
//! The mapping is held by core's `tests/vk_tuning.rs`, which runs the nonlinear gong. What is held
//! here is the wiring, and the bars listen to the OUTPUT — the run's own starting energy and pitch
//! — rather than comparing the reported `tuning` block with core: a build that ignored the mapping
//! would still report it correctly (the string batch's finding).

mod common;

use common::{base_params, f, ok, sim};
use physsynth_core::plate::{Boundary, VkParams, VkPlate, VkSpec};
use physsynth_core::plate_tuning::{preset, tune_vk_plate, SetBy, Targets};
use serde_json::{json, Value};
use std::f64::consts::PI;

// Every musical key OFF the server's fallback (`build_musical`'s defaults are f0 100, t60 2 / 0.5,
// f_ref 2000, fs 32000, N 20, steel, 1 mm, 0.4 m): a misspelled key reads the fallback, and a
// request that sent the fallback could not tell (CLAUDE.md's first rule for a bar).
const FS: f64 = 40_000.0;
const F0: f64 = 130.0;
const T60_LOW: f64 = 2.6;
const T60_HIGH: f64 = 0.45;
const F_REF: f64 = 1700.0;
const LX: f64 = 0.35;
const LY: f64 = 0.27;
const N: i64 = 16;

/// A musical gong request, with the physics keys set to values the mapping must ignore.
fn musical(overrides: Value) -> Value {
    let mut p = base_params(json!({
        "model": "vk", "domain": "supported", "controls": "musical",
        // Decoys: a musical build that read any of these would be a different gong.
        "E": 1.0e11, "e": 0.004, "nu": 0.1, "rho": 3000.0, "sigma": 4.0,
        "Lx": LX, "Ly": LY, "N": N, "fs": FS,
        "material": "brass", "set_by": "pitch", "f0": F0, "thickness_mm": 2.5,
        "t60_low": T60_LOW, "t60_high": T60_HIGH, "f_ref": F_REF,
        "w_over_e": 1.0, "audio_duration": 0.3, "animation_window": 0.01,
    }));
    common::merge(&mut p, overrides);
    p
}

fn geometry(boundary: Boundary) -> VkSpec {
    VkSpec {
        lx: LX,
        ly: LY,
        fs: FS,
        n: N,
        boundary: Some(boundary),
        couple_max_iter: 50,
        ..VkSpec::default()
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

fn cents(f: f64, target: f64) -> f64 {
    1200.0 * (f / target).log2()
}

fn message(payload: &Value) -> String {
    assert_eq!(payload["error"]["kind"], "param", "{payload}");
    payload["error"]["message"].as_str().unwrap().to_owned()
}

#[test]
fn the_musical_gong_is_built_from_the_metal_and_the_targets_not_the_sliders() {
    // The run's own starting energy against a core gong built from the mapping and struck the
    // page's way (mode (1,1), w_over_e thicknesses deep): the energy carries E, ν, ρ_v and the
    // thickness together, so a decoy read or a density in the wrong slot moves it.
    let payload = ok(&musical(json!({})));
    let tu = tune_vk_plate(
        &geometry(Boundary::Supported),
        &preset("brass").unwrap().material,
        &targets(),
    )
    .unwrap();
    let mut spec = geometry(Boundary::Supported);
    tu.apply(&mut spec);
    let mut vk = VkPlate::new(VkParams::new(&spec).unwrap());
    let lin = &vk.p.lin;
    let full: Vec<f64> = lin
        .x
        .iter()
        .zip(&lin.y)
        .map(|(&x, &y)| tu.plate.thickness * (PI * x / lin.lx).sin() * (PI * y / lin.ly).sin())
        .collect();
    let u0 = vk.p.to_live(&full);
    vk.set_state(&u0, &vec![0.0; u0.len()]).unwrap();
    let e0 = f(&payload["energy"]["value"][0]);
    assert!(
        (e0 / vk.energy() - 1.0).abs() < 1e-12,
        "payload E0 {e0} against the mapped gong's {}",
        vk.energy()
    );
    let t = &payload["tuning"];
    assert_eq!(payload["controls"], "musical");
    assert_eq!(t["material"], "brass");
    assert!((f(&t["thickness_mm"]) / (tu.plate.thickness * 1e3) - 1.0).abs() < 1e-3);
    assert!(
        f(&t["rho"]) == 8530.0,
        "the volume density, not the areal one: {t}"
    );
    assert_eq!(t["exact"], true);
    assert_eq!(t["partials"].as_array().unwrap().len(), 8);
}

#[test]
fn a_gentle_strike_rings_at_the_settled_pitch_and_a_hard_one_starts_above_it() {
    // The whole run's zero-crossing pitch (the page's own `f0_detected`): within a cent of the
    // pitch asked for at a twentieth of a thickness (the bend there is ~0.2 cents, plan §8.2), and
    // well above it at two thicknesses — which is what the gentle bar is not passing by accident.
    let gentle = ok(&musical(json!({"w_over_e": 0.05, "audio_duration": 0.5})));
    let fg = f(&gentle["meta"]["spectrum"]["f0_detected"]);
    assert!(cents(fg, F0).abs() < 1.0, "gentle strike reads {fg} Hz");
    let hard = ok(&musical(json!({"w_over_e": 2.0, "audio_duration": 0.5})));
    let fh = f(&hard["meta"]["spectrum"]["f0_detected"]);
    assert!(cents(fh, F0) > 50.0, "a 2 e strike reads only {fh} Hz");
}

#[test]
fn the_gong_reads_where_the_note_starts_and_that_it_settles() {
    // Short ring times so a second of sound reaches the note: plan §8.4's early and late windows.
    let payload = ok(&musical(json!({
        "w_over_e": 2.0, "audio_duration": 1.0, "t60_low": 0.5, "t60_high": 0.25,
    })));
    let r = &payload["strike_readout"];
    assert_eq!(r["kind"], "gong");
    let (start, end) = (f(&r["start_cents"]), f(&r["end_cents"]));
    assert!(start > 100.0, "starts only {start:+.1} cents up: {r}");
    assert!(end.abs() < 10.0, "ends {end:+.2} cents off the note: {r}");
    assert!(
        r["message"].as_str().unwrap().contains("settled onto it"),
        "{r}"
    );
    let c = &payload["cost"];
    assert!(f(&c["solves_per_step"]) > 1.0, "{c}");
    assert!(
        c["predicted_multiple"].is_number(),
        "the gong is priced before it runs: {c}"
    );
}

#[test]
fn the_cymbal_reads_when_its_lowest_tone_settles() {
    // The cymbal has no starting pitch (plan §8.4): its read-out is the time its lowest elastic
    // mode settles within 25 cents of the note — after a crash at a hard strike, from the start at
    // a gentle one.
    let hard = ok(&musical(json!({
        "domain": "free", "material": "bronze", "w_over_e": 2.0, "audio_duration": 0.6,
        "t60_low": 1.0, "t60_high": 0.5, "pluck_width": 0.45, "pluck_x": 0.4, "pluck_y": 0.55,
    })));
    let r = &hard["strike_readout"];
    assert_eq!(r["kind"], "cymbal", "{r}");
    assert_eq!(hard["tuning"]["exact"], false);
    let settle = f(&r["settle_ms"]);
    assert!(settle > 0.0 && settle < 600.0, "{r}");
    let gentle = ok(&musical(json!({
        "domain": "free", "material": "bronze", "w_over_e": 0.05, "audio_duration": 0.3,
        "pluck_width": 0.45, "pluck_x": 0.4, "pluck_y": 0.55,
    })));
    assert_eq!(
        gentle["strike_readout"]["settle_ms"], 0.0,
        "{}",
        gentle["strike_readout"]
    );
}

#[test]
fn musical_keys_do_not_touch_the_physical_gong() {
    // The front-end sends every slider to every model: a gong request on its physical controls
    // must come back byte for byte whatever musical keys ride along, `controls: physical` among
    // them.
    let plain = base_params(json!({
        "model": "vk", "E": 2.0e11, "e": 0.001, "nu": 0.3, "rho": 7800.0,
        "Lx": 0.15, "Ly": 0.15, "N": 14, "fs": 48000.0, "w_over_e": 3.0,
        "audio_duration": 0.05, "animation_window": 0.01,
    }));
    let mut noisy = plain.clone();
    common::merge(
        &mut noisy,
        json!({"controls": "physical", "material": "glass", "set_by": "thickness",
               "thickness_mm": 4.0, "f0": 330.0, "t60_low": 9.0, "t60_high": 0.1,
               "f_ref": 4000.0, "custom_E_gpa": 90.0}),
    );
    let (a, b) = (sim(&plain), sim(&noisy));
    assert!(a.get("error").is_none(), "{}", a["error"]);
    for key in ["tuning", "strike_readout", "cost", "controls"] {
        assert!(a.get(key).is_none(), "a physical gong carries {key}");
    }
    assert_eq!(a.to_string(), b.to_string());
}

#[test]
fn the_energy_verdict_drops_the_single_rate_oracle_under_a_frequency_dependent_loss() {
    let payload = ok(&musical(json!({"audio_duration": 0.1})));
    let e = &payload["energy"];
    assert_eq!(e["sigma_is_zero"], false);
    assert!(e["lossy"].get("measured_2sigma").is_none(), "{e}");
    assert_eq!(e["lossy"]["monotone"], true);
}

#[test]
fn set_by_thickness_reads_the_settled_pitch_out() {
    let payload = ok(&musical(
        json!({"set_by": "thickness", "thickness_mm": 1.5, "material": "steel"}),
    ));
    let t = &payload["tuning"];
    assert_eq!(t["set_by"], "thickness");
    assert!((f(&t["thickness_mm"]) - 1.5).abs() < 1e-9);
    assert!(t["f0_target"].is_null());
    // The continuum f11 = (π/2) κ (1/Lx² + 1/Ly²), κ = e sqrt(E / 12 ρ (1 − ν²)) for steel.
    let (e, nu, rho): (f64, f64, f64) = (200.0e9, 0.303, 7820.0);
    let kappa = 1.5e-3 * (e / (12.0 * rho * (1.0 - nu * nu))).sqrt();
    let ly = f(&payload["grid"]["extent_y"]);
    let f11 = 0.5 * PI * kappa * (1.0 / (LX * LX) + 1.0 / (ly * ly));
    assert!(
        (f(&t["f0"]) / f11 - 1.0).abs() < 0.03,
        "{} against {f11}",
        t["f0"]
    );
}

#[test]
fn what_no_gong_can_be_is_refused_by_name() {
    let cases = [
        (json!({"material": "sitka spruce"}), "isotropic"),
        (
            json!({"material": "unobtainium"}),
            "material must be one of",
        ),
        (
            json!({"set_by": "weight"}),
            "set_by must be 'pitch' or 'thickness'",
        ),
        (json!({"t60_high": 8.0}), "out of reach"),
        (json!({"f_ref": F0}), "picks the fundamental"),
        (json!({"fs": 1000.0}), "fs must be in"),
    ];
    for (bad, needle) in cases {
        let msg = message(&sim(&musical(bad.clone())));
        assert!(msg.contains(needle), "{bad}: {msg}");
    }
}

// -- the guard (plan §11): the gong predicted, the cymbal stopped ---------------------------------

#[test]
fn a_gong_past_the_measured_edge_is_refused_before_it_runs_and_told_what_fits() {
    // f₁·k = 300 / 8000 = 0.0375 at a strike of 6: the table's broken corner. A lower strike or a
    // higher sample rate fits, and the refusal names both — on either control set.
    let msg = message(&sim(&musical(json!({
        "f0": 300.0, "fs": 8000.0, "w_over_e": 6.0, "t60_low": 1.0, "t60_high": 0.5,
        "f_ref": 1200.0,
    }))));
    assert!(msg.contains("past the measured edge"), "{msg}");
    assert!(
        msg.contains("raise fs to") && msg.contains("strike at most 4"),
        "{msg}"
    );
    // The physical page: a 9 mm steel gong 0.4 m square has f₁ ≈ 271 Hz; at 8 kHz (f₁·k ≈ 0.034)
    // that is past the same edge.
    let phys = message(&sim(&base_params(json!({
        "model": "vk", "domain": "supported", "E": 2.0e11, "e": 0.009, "nu": 0.3, "rho": 7800.0,
        "Lx": 0.4, "Ly": 0.4, "N": 16,
        "fs": 8000.0, "w_over_e": 6.0, "audio_duration": 0.1, "animation_window": 0.01,
    }))));
    assert!(phys.contains("past the measured edge"), "{phys}");
}

#[test]
fn a_cymbal_that_does_not_settle_is_stopped_at_its_first_failed_step() {
    // Plan §11.5: a width-0.1 strike four thicknesses deep on a 12-segment cymbal at f₁·k ≈ 0.003
    // broke on its third step. The render stops there and says what to change.
    let msg = message(&sim(&musical(json!({
        "domain": "free", "material": "steel", "N": 12, "fs": 32000.0, "f0": 96.0,
        "Lx": 0.4, "Ly": 0.4, "w_over_e": 4.0, "pluck_width": 0.1,
        "t60_low": 2.0, "t60_high": 1.0, "f_ref": 384.0, "audio_duration": 0.5,
    }))));
    assert!(msg.contains("did not settle"), "{msg}");
    assert!(msg.contains("widen the strike"), "{msg}");
}

#[test]
fn a_centred_cymbal_strike_says_it_cannot_reach_the_lowest_tone() {
    // The free plate's lowest mode is a twist, odd about both centre lines, so a centred strike
    // hands it nothing; reading its "pitch" would read leakage (it came back +54 cents, settled
    // or not). The read-out says so instead.
    let payload = ok(&musical(json!({
        "domain": "free", "material": "bronze", "w_over_e": 2.0, "audio_duration": 0.2,
        "pluck_x": 0.5, "pluck_y": 0.5,
    })));
    let r = &payload["strike_readout"];
    assert!(f(&r["mode1_share"]) < 0.01, "{r}");
    assert!(r["settle_ms"].is_null(), "{r}");
    assert!(
        r["message"]
            .as_str()
            .unwrap()
            .contains("off the centre lines"),
        "{r}"
    );
}
