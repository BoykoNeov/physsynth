//! The air as an impedance through the payload builder — `test_web_backend.py`'s airload section.
//!
//! The booked ledger grows a FIFTH channel (stored), and the damping becomes frequency-dependent —
//! the half a constant `R` cannot do. The anchor runs the other way: pull the corner to 0 and the
//! constant-`R` body comes back bit for bit.

mod common;

use common::{decode_f32, f, ok, sim};
use physsynth_viewer::airload::{
    self as al, AIRLOAD_CORNER_DEFAULT, AIRLOAD_CORNER_MAX, AIRLOAD_K_MAX, AIRLOAD_N_MAX,
    AIRLOAD_R_MAX, AIRLOAD_SWEEP_FS, AIRLOAD_SWEEP_POINTS, AIRLOAD_WEAK_LOADING_MAX,
    AIRLOAD_WEIGHT_MAX, AIRLOAD_WORK_MAX,
};
use physsynth_viewer::energy::LOSSLESS_TOL;
use physsynth_viewer::py::commas;
use serde_json::{json, Value};

/// A short run with a COARSE sweep — the sweep is most of this payload's cost.
fn alp(overrides: Value) -> Value {
    let mut p = json!({"model": "airload", "audio_duration": 0.35, "sweep_points": 3,
                       "sweep_cycles": 6});
    common::merge(&mut p, overrides);
    p
}

/// The non-null entries of a list.
fn nums(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_f64)
        .collect()
}

#[test]
fn conserves_while_the_air_both_stores_and_radiates() {
    let d = ok(&alp(json!({"audio_duration": 0.6})));
    let e = &d["energy"];
    assert_eq!(e["sigma_is_zero"], true);
    assert!(f(&e["lossless"]["drift"]) < LOSSLESS_TOL);
    assert_eq!(e["lossless"]["pass"], true);
    let ex = &d["meta"]["exchange"];
    assert_eq!(ex["kind"], "airload");
    assert!(f(&ex["total_drift"]) < LOSSLESS_TOL);
    assert!(f(&ex["rad_frac_end"]) > 0.5);
    assert!(
        f(&ex["stored_frac_peak"]) > 0.05,
        "the new channel is visible"
    );
    assert!((f(&ex["mech_frac_end"]) - (1.0 - f(&ex["rad_frac_end"]))).abs() <= 1e-9);
}

#[test]
fn a_zero_corner_reproduces_the_constant_r_body_bit_identically() {
    let shared = json!({"audio_duration": 0.35, "bridge_stiffness": 8000, "N": 90,
                        "lambda": 0.85, "pluck_position": 0.27, "amplitude": 1.5e-3,
                        "distance": 1.4, "radiation_R": 133.0});
    let mut a = json!({"model": "airload", "air_corner": 0.0, "radiation_weight": 1.0,
                       "sweep_points": 2, "sweep_cycles": 4});
    common::merge(&mut a, shared.clone());
    let mut b = json!({"model": "radbody", "sweep_points": 2, "sweep_cap": 0.2});
    common::merge(&mut b, shared);
    let (flat, b15) = (ok(&a), ok(&b));
    assert_eq!(flat["energy"], b15["energy"]);
    assert_eq!(flat["audio"]["b64"], b15["audio"]["b64"]);
    assert_eq!(flat["frames"]["b64"], b15["frames"]["b64"]);
    for key in [
        "time",
        "e_string_frac",
        "e_body_frac",
        "e_conn_frac",
        "e_rad_frac",
        "total_frac",
    ] {
        assert_eq!(
            flat["meta"]["exchange"][key], b15["meta"]["exchange"][key],
            "{key}"
        );
    }
    assert!(nums(&flat["meta"]["exchange"]["e_stored_frac"])
        .iter()
        .all(|&v| v == 0.0));
}

#[test]
#[allow(clippy::needless_range_loop)] // parallel channels, one index
fn five_channels_sum_to_the_flat_reference_and_e_conn_stays_signed() {
    let ex = ok(&alp(json!({})))["meta"]["exchange"].clone();
    let n = ex["time"].as_array().unwrap().len();
    let keys = [
        "e_string_frac",
        "e_body_frac",
        "e_conn_frac",
        "e_stored_frac",
        "e_rad_frac",
        "total_frac",
    ];
    let v: Vec<Vec<f64>> = keys.iter().map(|k| nums(&ex[*k])).collect();
    assert!(n > 100 && v.iter().all(|a| a.len() == n));
    for i in 0..n {
        assert!((v[0][i] + v[1][i] + v[2][i] + v[3][i] + v[4][i] - v[5][i]).abs() <= 1e-9);
        assert!((v[5][i] - 1.0).abs() <= 1e-9);
    }
    assert!(v[2].iter().copied().fold(f64::INFINITY, f64::min) < 0.0);
    assert_eq!(v[4][0], 0.0);
    assert!(v[4][n - 1] > v[4][n / 2]);
    let st_max = v[3].iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(st_max > 0.0 && v[3][n - 1] < st_max, "stored is returned");
    assert!(f(&ex["stored_frac_window_peak"]) <= f(&ex["stored_frac_peak"]) + 1e-9);
    assert!((f(&ex["stored_frac_window_peak"]) - st_max).abs() <= 1e-4 * st_max);
}

#[test]
fn the_ledger_residual_catches_what_the_drift_cannot() {
    let ex = ok(&alp(json!({})))["meta"]["exchange"].clone();
    assert!(f(&ex["ledger_residual"]) < 1e-12);
    assert_eq!(ex["ledger_pass"], true);
    let g = |k: &str| nums(&ex[k]);
    let (es, eb, ec, est, er, tot) = (
        g("e_string_frac"),
        g("e_body_frac"),
        g("e_conn_frac"),
        g("e_stored_frac"),
        g("e_rad_frac"),
        g("total_frac"),
    );
    let worst = |sign: f64, keep: f64| {
        (0..tot.len())
            .map(|i| (tot[i] - (es[i] + eb[i] + ec[i] + sign * keep * est[i] + er[i])).abs())
            .fold(0.0f64, f64::max)
    };
    assert!(worst(-1.0, 1.0) > 1e-3, "a sign-flipped channel is LOUD");
    assert!(worst(1.0, 0.0) > 1e-3, "a dropped channel is LOUD");
    assert!(
        f(&ex["total_drift"]) < LOSSLESS_TOL,
        "while the drift stays green"
    );
}

#[test]
fn alpha_climbs_with_frequency_and_the_constant_r_impostor_cannot() {
    let sw = ok(&alp(json!({"sweep_points": 5})))["meta"]["spectrum"]["sweep"].clone();
    let a = nums(&sw["alpha"]);
    assert_eq!(a.len(), 5);
    assert!(a.windows(2).all(|w| w[1] > w[0]), "{a:?}");
    assert!(f(&sw["span"]) > 50.0);
    assert!(f(&sw["flat_ratio_top"]) > 20.0);
    assert!(f(&sw["alpha_flat"]) > 0.0 && f(&sw["r_flat"]) > 0.0);
}

#[test]
fn the_oracle_residual_is_second_order_in_the_loading_ratio() {
    let sw = ok(&alp(json!({"sweep_points": 5})))["meta"]["spectrum"]["sweep"].clone();
    let (a, o, w) = (
        nums(&sw["alpha"]),
        nums(&sw["alpha_oracle"]),
        nums(&sw["alpha_over_omega"]),
    );
    assert!(a.len() == 5 && o.len() == 5 && w.len() == 5);
    let rel: Vec<f64> = (0..5).map(|i| (a[i] - o[i]).abs() / o[i]).collect();
    let weak: Vec<usize> = (0..5)
        .filter(|&i| w[i] <= AIRLOAD_WEAK_LOADING_MAX)
        .collect();
    assert!(!weak.is_empty());
    assert!(weak.iter().all(|&i| rel[i] < 0.02));
    let coef: Vec<f64> = (0..5).map(|i| rel[i] / (w[i] * w[i])).collect();
    assert!(!sw["resid_coef"].is_null());
    let (mn, mx) = (
        coef.iter().copied().fold(f64::INFINITY, f64::min),
        coef.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    );
    assert!(5.0 < mn && mx < 60.0, "{coef:?}");
    assert!(
        mx / mn < 3.0,
        "roughly CONSTANT: that is what makes it 2nd order"
    );
}

#[test]
fn the_pitch_drop_is_the_reactance_and_a_flat_load_has_none() {
    let sw = ok(&alp(json!({"sweep_points": 4})))["meta"]["spectrum"]["sweep"].clone();
    let (df, dfo) = (nums(&sw["df_meas"]), nums(&sw["df_oracle"]));
    assert!(df.len() == 4 && df.iter().all(|&d| d < -1.0), "{df:?}");
    assert!((0..4).all(|i| (df[i] - dfo[i]).abs() < 0.6));
    assert!(df[3] > df[0], "the drop shrinks with frequency");
    let flat = ok(&alp(
        json!({"sweep_points": 4, "air_corner": 0.0, "radiation_R": 133.0}),
    ))["meta"]["spectrum"]["sweep"]
        .clone();
    let (fdf, fa) = (nums(&flat["df_meas"]), nums(&flat["alpha"]));
    assert!(fdf.iter().all(|d| d.abs() < 0.05), "{fdf:?}");
    assert!(
        fa.iter().all(|&x| (x - fa[0]).abs() <= 0.02 * fa[0].abs()),
        "{fa:?}"
    );
}

#[test]
fn the_pitch_shift_needs_the_schemes_own_unloaded_reference() {
    let (f0, fs, steps) = (1760.0, 22222.2, 400);
    let (r, m_a) = (13146.4, 1.9164);
    let (f_load, _) = al::measure_mode(f0, r, m_a, 0.02, 0.02, fs, steps).unwrap();
    let (f_free, _) = al::measure_mode(f0, 0.0, m_a, 0.02, 0.02, fs, steps).unwrap();
    assert!(
        (f_load - f0) / f0 > 0.0,
        "against NOMINAL the scheme's warping wins — the trap"
    );
    assert!(
        (f_load - f_free) / f_free < 0.0,
        "against the scheme's OWN unloaded run, the air wins"
    );
    assert!(f_free > f0);
}

#[test]
fn overdamped_points_are_censored_not_guessed() {
    let sw = ok(&alp(json!({"sweep_points": 3, "air_corner": 0.0})))["meta"]["spectrum"]["sweep"]
        .clone();
    assert_eq!(sw["skipped"], false);
    assert!(sw["n_censored"].as_i64().unwrap() > 0);
    let alpha = sw["alpha"].as_array().unwrap();
    assert_eq!(alpha.len(), 3);
    assert_eq!(sw["f"].as_array().unwrap().len(), 3);
    assert!(alpha.iter().any(Value::is_null));
}

#[test]
fn r_zero_skips_the_sweep_with_a_label_instead_of_a_row_of_nulls() {
    let d = ok(&alp(json!({"radiation_R": 0.0})));
    let sw = &d["meta"]["spectrum"]["sweep"];
    assert_eq!(sw["skipped"], true);
    assert_eq!(sw["steps"], 0);
    assert_eq!(sw["alpha"], json!([]));
    assert!(sw["note"].as_str().unwrap().contains("decoupled"));
    let ex = &d["meta"]["exchange"];
    assert_eq!(f(&ex["rad_frac_end"]), 0.0);
    assert_eq!(f(&ex["stored_frac_peak"]), 0.0);
    assert_eq!(d["energy"]["lossless"]["pass"], true);
}

#[test]
fn sweep_is_pinned_at_48k_and_does_not_follow_the_renders_rate() {
    let a = ok(&alp(json!({"N": 60, "sweep_points": 3})));
    let b = ok(&alp(json!({"N": 120, "sweep_points": 3})));
    for d in [&a, &b] {
        assert_eq!(f(&d["meta"]["spectrum"]["sweep"]["fs"]), AIRLOAD_SWEEP_FS);
    }
    assert_ne!(a["fs_sim"], b["fs_sim"]);
    let (sa, sb) = (
        &a["meta"]["spectrum"]["sweep"],
        &b["meta"]["spectrum"]["sweep"],
    );
    assert_eq!(sa["alpha"], sb["alpha"]);
    assert_eq!(sa["df_meas"], sb["df_meas"]);
}

#[test]
fn sigma_body_gates_the_verdict_and_radiation_never_does() {
    assert_eq!(ok(&alp(json!({})))["energy"]["sigma_is_zero"], true);
    let lossy = ok(&alp(json!({"sigma_body": 20.0})))["energy"].clone();
    assert_eq!(lossy["sigma_is_zero"], false);
    assert_eq!(lossy["lossy"]["monotone"], true);
    assert!(lossy["lossy"].get("measured_2sigma").is_none());
}

#[test]
fn the_radiation_weight_is_not_a_volume_control_and_the_peak_says_so() {
    let peak = |w: f64| f(&ok(&alp(json!({"radiation_weight": w})))["meta"]["pressure_peak"]);
    let (a, b, c) = (peak(0.02), peak(0.05), peak(0.2));
    assert!(a > 0.0 && b > 0.0 && c > 0.0);
    assert!(b > a && b > c, "the match is the loudest: {a} {b} {c}");
}

#[test]
fn the_sphere_readout_says_which_radius_each_coefficient_implies() {
    let sp = ok(&alp(json!({})))["meta"]["spectrum"]["sphere"].clone();
    assert_eq!(sp["consistent"], true);
    assert!(f(&sp["rel_gap"]) < 0.01);
    assert!((f(&sp["a_from_R"]) - 0.05).abs() <= 0.05e-3);
    assert!((f(&sp["a_from_corner"]) - f(&sp["a_from_R"])).abs() <= 1e-3 * f(&sp["a_from_R"]));
    assert!(
        (f(&sp["corner_sphere"]) - AIRLOAD_CORNER_DEFAULT).abs() <= 1e-3 * AIRLOAD_CORNER_DEFAULT
    );
    let off = ok(&alp(json!({"air_corner": 200.0})))["meta"]["spectrum"]["sphere"].clone();
    assert_eq!(off["consistent"], false);
    assert!(f(&off["rel_gap"]) > 0.5);
    assert_eq!(off["a_from_R"], sp["a_from_R"]);
}

#[test]
fn the_exact_load_tracks_the_compact_law_over_the_band() {
    let sp = ok(&alp(json!({})))["meta"]["spectrum"].clone();
    let (exact, compact) = (nums(&sp["re_z_modes"]), nums(&sp["r_compact_modes"]));
    let rel: Vec<f64> = (0..exact.len())
        .map(|i| (exact[i] - compact[i]).abs() / compact[i])
        .collect();
    assert!(rel.iter().all(|&r| r < 0.15), "{rel:?}");
    assert!(rel.windows(2).all(|w| w[1] > w[0]));
    assert!(exact[exact.len() - 1] / exact[0] > 10.0);
}

#[test]
fn shipped_sweep_settings_are_the_measured_ones() {
    // Expensive ON PURPOSE: the only run with the sweep overrides ABSENT, which is what ships.
    let sw = ok(&json!({"model": "airload", "audio_duration": 0.2}))["meta"]["spectrum"]["sweep"]
        .clone();
    assert_eq!(
        sw["f"].as_array().unwrap().len() as i64,
        AIRLOAD_SWEEP_POINTS
    );
    assert_eq!(
        sw["alpha"].as_array().unwrap().len() as i64,
        AIRLOAD_SWEEP_POINTS
    );
    assert_eq!(f(&sw["fs"]), AIRLOAD_SWEEP_FS);
    assert_eq!(sw["truncated"], false);
    assert_eq!(sw["n_censored"], 0);
    assert!(sw["steps"].as_i64().unwrap() < 60_000);
    for bad in [
        json!({"sweep_points": 1}),
        json!({"sweep_points": 99}),
        json!({"sweep_points": "x"}),
    ] {
        let mut p = json!({"model": "airload"});
        common::merge(&mut p, bad.clone());
        assert_eq!(sim(&p)["error"]["kind"], "param", "{bad}");
    }
}

#[test]
fn sweep_truncation_censors_the_tail_rather_than_hanging() {
    // Driven past its budget at 4,000 steps, which the reference reached by patching the constant.
    let d = al::build_payload_with(&alp(json!({"sweep_points": 6})), 4_000).unwrap();
    let sw = &d["meta"]["spectrum"]["sweep"];
    assert_eq!(sw["truncated"], true);
    let alpha = sw["alpha"].as_array().unwrap();
    assert_eq!(alpha.len(), 6);
    assert!(sw["n_censored"].as_i64().unwrap() > 0);
    assert!(alpha[5].is_null() && !alpha[0].is_null());
}

#[test]
fn the_coupled_k_guard_is_the_bare_bodys_and_the_load_does_not_move_it() {
    ok(&alp(
        json!({"N": 100, "bridge_stiffness": AIRLOAD_K_MAX, "sweep_points": 2}),
    ));
    let tight = sim(
        &json!({"model": "airload", "N": 40, "bridge_stiffness": 12000.0,
                            "audio_duration": 0.1, "sweep_points": 2}),
    );
    assert!(tight.get("error").is_some());
    let same = sim(
        &json!({"model": "radbody", "N": 40, "bridge_stiffness": 12000.0,
                           "audio_duration": 0.1, "sweep_points": 2, "sweep_cap": 0.1}),
    );
    assert_eq!(same["error"]["kind"], tight["error"]["kind"]);
}

#[test]
fn frame_and_grid_bookkeeping_line_up_and_the_nut_stays_clamped() {
    let d = ok(&alp(json!({})));
    let fr = &d["frames"];
    let (n, w) = (
        fr["n_frames"].as_u64().unwrap() as usize,
        fr["width"].as_u64().unwrap() as usize,
    );
    assert_eq!(fr["dims"], 1);
    assert_eq!(w, d["grid"]["x"].as_array().unwrap().len());
    assert_eq!(n, d["frame_times"].as_array().unwrap().len());
    assert!(n > 1);
    let buf = decode_f32(fr["b64"].as_str().unwrap());
    assert!((0..n).all(|k| buf[k * w] == 0.0));
    assert!(buf.iter().all(|v| v.is_finite()));
}

#[test]
fn guards_and_budget_are_clean_error_payloads() {
    for bad in [
        json!({"N": AIRLOAD_N_MAX + 1}),
        json!({"lambda": 1.0}),
        json!({"radiation_R": -1.0}),
        json!({"radiation_R": AIRLOAD_R_MAX + 1.0}),
        json!({"air_corner": -1.0}),
        json!({"air_corner": AIRLOAD_CORNER_MAX + 1.0}),
        json!({"radiation_weight": 0.0}),
        json!({"radiation_weight": AIRLOAD_WEIGHT_MAX + 0.1}),
        json!({"bridge_stiffness": AIRLOAD_K_MAX + 1.0}),
        json!({"sigma_body": -1.0}),
        json!({"audio_duration": 0.0}),
        json!({"distance": 0.0}),
        json!({"pluck_position": 1.0}),
    ] {
        let mut p = json!({"model": "airload", "sweep_points": 2});
        common::merge(&mut p, bad.clone());
        assert_eq!(sim(&p)["error"]["kind"], "param", "{bad}");
    }
    let over = sim(
        &json!({"model": "airload", "N": AIRLOAD_N_MAX, "rho": 0.001,
                           "audio_duration": 3.0, "sweep_points": 2}),
    );
    assert!(over["error"]["message"]
        .as_str()
        .unwrap()
        .contains(&commas(AIRLOAD_WORK_MAX)));
}

#[test]
fn ignores_params_that_belong_to_other_models() {
    let base = ok(&alp(json!({"sweep_points": 2})));
    let noisy = ok(&alp(json!({
        "sweep_points": 2, "K": 2.0e6, "alpha": 2.3, "depth": 1e-3, "kappa": 5.0, "EA": 1e4,
        "sigma_plate": 3.0, "n_plate": 12, "sweep_cap": 0.9,
    })));
    assert_eq!(noisy["meta"]["exchange"], base["meta"]["exchange"]);
}
