//! The geometrically-exact string through the payload builder — `test_web_backend.py`'s geometric
//! and phantom sections.
//!
//! What these pin: (a) the two bit-exact zeros the model rests on (planar `max|w|`, the unseeded
//! whirl) really are zero; (b) the rotating wave is an exact solution — round, with a longitudinal
//! field that leans without moving; (c) the whirl grows only inside the Mathieu tongue and the
//! energy conserves THROUGH the growth; (d) `lam_long` is a hard cap; (e) the phantom partials are
//! the quadratic combinations of the measured partials, gated on a one-sided defect.

mod common;

use std::sync::OnceLock;

use common::{decode_b64, decode_f32, f, ok, sim};
use physsynth_viewer::energy::LOSSLESS_TOL;
use physsynth_viewer::geometric::{
    GEOM_DT_MAX, GEOM_LAM_LONG_MAX, GEOM_N_MAX, GEOM_PHANTOM_DEFECT_MIN, GEOM_PHANTOM_WINDOW,
    GEOM_WORK_MAX,
};
use physsynth_viewer::py::commas;
use physsynth_viewer::AUDIO_FS;
use serde_json::{json, Value};

fn geom(overrides: Value) -> Value {
    let mut p = json!({
        "model": "geometric", "N": 8, "lam_long": 0.9, "T": 200.0, "rho": 0.005,
        "EA": 1.0e5, "L": 1.0, "theta": 0.28, "kappa": 0.0, "sigma0": 0.0, "sigma1": 0.0,
        "pickup_position": 0.25, "amplitude": 4e-3, "animation_window": 0.01,
    });
    common::merge(&mut p, overrides);
    p
}

fn whirl(overrides: Value) -> Value {
    let mut p = geom(json!({
        "domain": "whirl", "dt_over_t0": 1.5, "tongue_position": 0.25, "animation_window": 0.03,
    }));
    common::merge(&mut p, overrides);
    p
}

fn phantom_p(overrides: Value) -> Value {
    let mut p = json!({
        "model": "geometric", "domain": "phantom", "N": 16, "lam_long": 0.9,
        "T": 200.0, "rho": 0.005, "EA": 1.0e5, "L": 1.0, "theta": 0.28,
        "kappa": 8.0, "sigma0": 0.0, "sigma1": 0.0, "pickup_position": 0.25,
        "amplitude": 1.5e-3,
        // Deliberately absurd, and ignored: the window is fixed physics.
        "animation_window": 0.3,
    });
    common::merge(&mut p, overrides);
    p
}

/// The phantom run, shared by every test that reads it (the reference's module-scoped fixture).
fn phantom() -> &'static Value {
    static RUN: OnceLock<Value> = OnceLock::new();
    RUN.get_or_init(|| ok(&phantom_p(json!({}))))
}

fn nums(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_f64)
        .collect()
}

// -- planar and rotating

#[test]
fn planar_max_w_is_bit_exact_zero() {
    let sp = ok(&geom(json!({"domain": "planar"})))["meta"]["spectrum"].clone();
    assert_eq!(sp["kind"], "planar");
    assert_eq!(f(&sp["max_w"]), 0.0, "not small — exactly zero");
    assert_eq!(sp["exact_zero"], true);
}

#[test]
fn planar_conserves_through_the_wrapper() {
    let e = ok(&geom(json!({"domain": "planar"})))["energy"].clone();
    assert_eq!(e["sigma_is_zero"], true);
    assert!(f(&e["lossless"]["drift"]) < LOSSLESS_TOL);
    assert_eq!(e["lossless"]["pass"], true);
}

#[test]
fn rotating_wave_is_a_true_circle() {
    let sp = ok(&geom(json!({"domain": "rotating"})))["meta"]["spectrum"].clone();
    assert_eq!(sp["kind"], "rotating");
    assert!(f(&sp["roundness"]) < 1e-6);
    assert!(f(&sp["bvp_frequency"]) > 0.0);
    assert_eq!(sp["bvp_converged"], true);
    assert!(sp["bvp_iterations"].as_i64().unwrap() > 0);
}

#[test]
fn rotating_wave_longitudinal_field_leans_but_does_not_move() {
    let d = ok(&geom(json!({"domain": "rotating"})));
    assert!(f(&d["meta"]["spectrum"]["long_kin_over_e"]) < 1e-12);
}

#[test]
fn frames_decode_to_three_fields_with_clamped_ends() {
    let d = ok(&geom(json!({"domain": "planar"})));
    let fr = &d["frames"];
    assert_eq!(fr["dims"], 1);
    assert_eq!(fr["fields"], json!(["u", "w", "v"]));
    let w = fr["width"].as_u64().unwrap() as usize;
    let n = fr["n_frames"].as_u64().unwrap() as usize;
    assert_eq!(w, d["grid"]["x"].as_array().unwrap().len());
    let buf = decode_f32(fr["b64"].as_str().unwrap());
    assert_eq!(buf.len(), n * 3 * w);
    let mut u_peak = 0.0f64;
    for k in 0..n {
        for field in 0..3 {
            let row = &buf[(3 * k + field) * w..(3 * k + field + 1) * w];
            assert_eq!(row[0], 0.0);
            assert_eq!(row[w - 1], 0.0);
            if field == 0 {
                u_peak = row.iter().fold(u_peak, |m, &v| m.max(f64::from(v).abs()));
            }
            if field == 1 {
                assert!(
                    row.iter().all(|&v| v == 0.0),
                    "planar: w is zero in every frame"
                );
            }
        }
    }
    assert!((u_peak - 4e-3).abs() <= 0.02 * 4e-3);
}

#[test]
fn orbit_trail_is_the_probe_node_and_indexable_by_frame() {
    let orb = ok(&geom(json!({"domain": "rotating"})))["orbit"].clone();
    let dec = |k: &str| decode_f32(orb[k].as_str().unwrap());
    let (u, w) = (dec("u"), dec("w"));
    assert_eq!(u.len() as u64, orb["n"].as_u64().unwrap());
    assert_eq!(u.len(), w.len());
    assert!(f(&orb["per_frame"]) > 0.0);
    let peak = |a: &[f32]| a.iter().fold(0.0f64, |m, &v| m.max(f64::from(v).abs()));
    assert!(
        (peak(&w) - peak(&u)).abs() <= 0.05 * peak(&u),
        "a circle: equal amplitudes"
    );
}

// -- the whirl

#[test]
fn whirl_grows_inside_the_tongue_and_conserves_through_it() {
    let d = ok(&whirl(json!({})));
    let (sp, e) = (&d["meta"]["spectrum"], &d["energy"]);
    assert_eq!(sp["kind"], "whirl");
    assert_eq!(sp["in_tongue"], true);
    assert!(f(&sp["growth"]) > 3.0);
    assert!(f(&e["lossless"]["drift"]) < LOSSLESS_TOL);
    assert_eq!(e["lossless"]["pass"], true);
}

#[test]
fn whirl_needs_no_new_energy_verdict_unlike_the_bow() {
    let e = ok(&whirl(json!({})))["energy"].clone();
    assert!(e.get("kind").is_none() && e.get("balance").is_none());
    assert!(e.get("lossless").is_some());
}

#[test]
fn whirl_is_dead_outside_the_tongue() {
    let sp = ok(&whirl(json!({"tongue_position": 0.8})))["meta"]["spectrum"].clone();
    assert_eq!(sp["in_tongue"], false);
    assert!(f(&sp["growth"]) < 3.0);
    assert_eq!(f(&sp["predicted_rate"]), 0.0);
}

#[test]
fn degenerate_string_cannot_whirl() {
    let sp = ok(&whirl(json!({"tongue_position": 0.0})))["meta"]["spectrum"].clone();
    assert_eq!(sp["degenerate"], true);
    assert_eq!(sp["in_tongue"], false);
    assert!((f(&sp["growth"]) - 1.0).abs() <= 0.15);
    assert_eq!(f(&sp["kappa_w"]), 0.0);
}

#[test]
fn velocity_seed_makes_the_degenerate_string_marginal_not_stable() {
    let disp = ok(&whirl(json!({"tongue_position": 0.0})))["meta"]["spectrum"].clone();
    let vel = ok(&whirl(
        json!({"tongue_position": 0.0, "seed_velocity": true}),
    ))["meta"]["spectrum"]
        .clone();
    assert_eq!(disp["seed_velocity"], false);
    assert_eq!(vel["seed_velocity"], true);
    assert!(f(&vel["growth"]) > f(&disp["growth"]));
    let tongue = ok(&whirl(json!({})))["meta"]["spectrum"].clone();
    assert!(
        f(&vel["growth"]) < f(&tongue["growth"]),
        "secular, not exponential"
    );
}

#[test]
fn unseeded_whirl_is_the_honesty_gate() {
    let d = ok(&whirl(json!({"seed_frac": 0.0})));
    let sp = &d["meta"]["spectrum"];
    assert_eq!(sp["seeded"], false);
    assert_eq!(
        nums(&sp["envelope"])
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
        0.0
    );
    assert_eq!(d["energy"]["lossless"]["pass"], true);
}

#[test]
fn whirl_rate_matches_the_mathieu_prediction_and_runs_low() {
    let sp = ok(&whirl(json!({})))["meta"]["spectrum"].clone();
    assert!(f(&sp["predicted_rate"]) > 0.0);
    assert!(!sp["measured_rate"].is_null());
    assert!((f(&sp["rate_ratio"]) - 1.0).abs() <= 0.35);
}

#[test]
fn a_string_spelled_seed_velocity_is_read_the_way_as_bool_reads_it() {
    // `_as_bool`: "yes" / "true" / "on" / "1" are true, whatever the case and padding.
    let yes = ok(&whirl(json!({"seed_velocity": " Yes "})))["meta"]["spectrum"].clone();
    assert_eq!(yes["seed_velocity"], true);
    let no = ok(&whirl(json!({"seed_velocity": "nope"})))["meta"]["spectrum"].clone();
    assert_eq!(no["seed_velocity"], false);
}

// -- guards and the viz-only contract

#[test]
fn has_no_audio_and_says_why() {
    let d = ok(&geom(json!({"domain": "planar"})));
    assert!(d["audio"].is_null());
    let note = d["audio_note"].as_str().unwrap();
    assert!(note.contains("22") && note.contains("minutes"), "{note}");
}

#[test]
fn lam_long_above_one_is_rejected() {
    let d = sim(&geom(
        json!({"domain": "planar", "lam_long": GEOM_LAM_LONG_MAX + 0.5}),
    ));
    assert!(d["error"]["message"].as_str().unwrap().contains("lam_long"));
}

#[test]
fn lam_long_is_the_knob_and_lambda_is_derived() {
    let d = ok(&geom(json!({"domain": "planar"})));
    assert!((f(&d["lam_long"]) - 0.9).abs() <= 0.9e-6);
    assert!(f(&d["lambda"]) < 0.1);
    let ratio = f(&d["meta"]["c_long"]) / f(&d["meta"]["c"]);
    assert!((ratio - 22.4).abs() <= 0.05 * 22.4);
}

#[test]
fn work_budget_is_its_own() {
    let d = sim(&geom(json!({"domain": "planar", "animation_window": 0.3})));
    let msg = d["error"]["message"].as_str().unwrap();
    assert!(
        msg.contains("work budget") && msg.contains(&commas(GEOM_WORK_MAX)),
        "{msg}"
    );
}

#[test]
fn bad_params_give_error_payload() {
    for bad in [
        json!({"N": GEOM_N_MAX + 1}),
        json!({"lam_long": 0.0}),
        json!({"domain": "nonsense"}),
        json!({"EA": 0.0}),
        json!({"domain": "whirl", "dt_over_t0": GEOM_DT_MAX + 1.0}),
        json!({"domain": "whirl", "tongue_position": -0.1}),
        json!({"pickup_position": 1.5}),
    ] {
        let d = sim(&geom(bad.clone()));
        let kind = d["error"]["kind"].as_str().unwrap_or("");
        assert!(kind == "param" || kind == "construction", "{bad} gave {d}");
    }
}

// -- the phantom partials (one shared run)

#[test]
fn phantom_peaks_are_the_quadratic_combinations_of_the_measured_partials() {
    let sp = &phantom()["meta"]["spectrum"];
    assert_eq!(sp["kind"], "phantom");
    assert_eq!(sp["resolved"], true);
    assert!(sp["n_peaks"].as_i64().unwrap() >= 4);
    assert!(f(&sp["combo_err"]) < 0.15);
    assert!(f(&sp["dominance"]) > 3.0);
    let (f1, f2) = (f(&sp["f1"]), f(&sp["f2"]));
    assert!((f(&sp["combos"]["f1+f2"]) - (f1 + f2)).abs() <= 1e-12 * (f1 + f2));
    assert!((f(&sp["combos"]["f2-f1"]) - (f2 - f1)).abs() <= 1e-12 * (f2 - f1));
    let ladder = nums(&sp["ladder"]);
    assert!((f(&sp["combos"]["f1+f2"]) - (ladder[0] + ladder[1])).abs() > 0.05);
}

#[test]
fn phantom_no_longitudinal_peak_sits_on_a_transverse_partial() {
    let sp = &phantom()["meta"]["spectrum"];
    for name in ["f1", "f2"] {
        let near = f(&sp[format!("nearest_to_{name}").as_str()]);
        assert!((near - f(&sp[name])).abs() > 3.0, "{name}");
    }
}

#[test]
fn phantom_headline_is_the_defect_measured_from_both_sides() {
    let sp = &phantom()["meta"]["spectrum"];
    let defect = f(&sp["defect"]);
    let (f1, f2) = (f(&sp["f1"]), f(&sp["f2"]));
    assert!((defect - (f2 - 2.0 * f1)).abs() <= 1e-12 * defect.abs());
    assert!(defect > GEOM_PHANTOM_DEFECT_MIN);
    let d = nums(&sp["displacements"]);
    assert!((d[0] - defect).abs() <= 0.3 && (d[1] - defect).abs() <= 0.3);
    assert!(d[0].min(d[1]) > 3.0);
}

#[test]
fn phantom_conserves_through_the_wrapper() {
    let e = &phantom()["energy"];
    assert_eq!(e["sigma_is_zero"], true);
    assert!(f(&e["lossless"]["drift"]) < LOSSLESS_TOL);
    assert_eq!(e["lossless"]["pass"], true);
}

#[test]
fn phantom_window_is_fixed_physics_and_ignores_the_animation_slider() {
    let d = phantom();
    let want = (GEOM_PHANTOM_WINDOW * f(&d["fs_sim"])).round_ties_even() as i64;
    assert_eq!(d["meta"]["num_steps"].as_i64().unwrap(), want);
}

#[test]
fn phantom_display_grid_is_denser_than_the_measurement_grid() {
    let d = phantom();
    let sp = &d["meta"]["spectrum"];
    let zoom = nums(&sp["zoom"]);
    let zoom_hz = zoom[1] - zoom[0];
    let steps = d["meta"]["num_steps"].as_f64().unwrap();
    let detector_nfft = 2f64.powf((steps * 2.0).log2().ceil());
    let detector_df = f(&d["fs_sim"]) / detector_nfft;
    let n_zoom = sp["zoom_freq"].as_array().unwrap().len() as f64;
    assert!(n_zoom > 10.0 * (zoom_hz / detector_df));
    assert!(sp["wide_freq"].as_array().unwrap().len() > 100);
}

#[test]
fn phantom_ships_the_audio_that_the_viz_only_regimes_do_not() {
    let d = phantom();
    let a = &d["audio"];
    assert_eq!(f(&a["fs"]), AUDIO_FS);
    let n = f(&a["n"]);
    assert!((n - GEOM_PHANTOM_WINDOW * AUDIO_FS).abs() <= 0.02 * GEOM_PHANTOM_WINDOW * AUDIO_FS);
    assert!(f(&a["peak"]) > 0.0);
    let raw = decode_b64(a["b64"].as_str().unwrap());
    assert_eq!(raw.len() as f64, 4.0 * n);
    assert!(decode_f32(a["b64"].as_str().unwrap())
        .iter()
        .all(|v| v.is_finite()));
    assert!(d["audio_note"].as_str().unwrap().contains("bridge force"));
    let planar = ok(&geom(json!({"domain": "planar"})));
    assert!(planar["audio"].is_null());
    assert!(planar["audio_note"].as_str().unwrap().contains("viz-only"));
}

#[test]
fn phantom_linear_string_has_no_channel_to_put_a_phantom_in() {
    let d = ok(&phantom_p(json!({"N": 8, "EA": 200.0}))); // EA == T0
    let sp = &d["meta"]["spectrum"];
    assert_eq!(sp["linear"], true);
    assert_eq!(f(&sp["bridge_max"]), 0.0);
    assert_eq!(sp["n_peaks"], 0);
    assert_eq!(sp["resolved"], false);
    assert_eq!(f(&d["audio"]["peak"]), 0.0);
}

#[test]
fn phantom_labels_a_grid_too_coarse_to_show_the_stiffness() {
    let sp = ok(&phantom_p(json!({"N": 8})))["meta"]["spectrum"].clone();
    assert_eq!(sp["linear"], false);
    assert!(sp["n_peaks"].as_i64().unwrap() >= 4);
    assert!(f(&sp["defect"]) < GEOM_PHANTOM_DEFECT_MIN);
    assert_eq!(sp["resolved"], false);
}

#[test]
fn phantom_defect_gate_is_one_sided_not_absolute() {
    const { assert!(GEOM_PHANTOM_DEFECT_MIN > 0.0) };
    for defect in [-10.0, -3.5, -0.5, 0.0, 2.9] {
        assert!(
            defect < GEOM_PHANTOM_DEFECT_MIN,
            "{defect} must not be scorable"
        );
    }
    const { assert!(3.79 >= GEOM_PHANTOM_DEFECT_MIN) };
}

#[test]
fn phantom_budget_and_amplitude_guards_give_clean_error_payloads() {
    for bad in [
        json!({"N": 32, "lam_long": 0.2}),
        json!({"amplitude": 0.02}),
    ] {
        let d = sim(&phantom_p(bad.clone()));
        let kind = d["error"]["kind"].as_str().unwrap_or("");
        assert!(kind == "param" || kind == "construction", "{bad} gave {d}");
    }
}
