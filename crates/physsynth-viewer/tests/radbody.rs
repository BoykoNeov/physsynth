//! The radiation-loaded body through the payload builder — `test_web_backend.py`'s radbody section.
//!
//! The claim is the BOOKED channel: `E_string + E_body + E_conn + int P_rad` conserves while the
//! mechanical part drains away as sound, and the drain has an OPTIMUM in `R` — more air is worse.
//! (The `R = 0` anchor against the plain `body` scene lives in `tests/body.rs`.)

mod common;

use common::{decode_f32, f, ok, sim};
use physsynth_core::radiation::monopole_radiation_resistance;
use physsynth_viewer::energy::LOSSLESS_TOL;
use physsynth_viewer::radbody::{
    self as rb, RADBODY_AUDIO_MAX, RADBODY_DISTANCE_MAX, RADBODY_K_MAX, RADBODY_N_MAX,
    RADBODY_R_DEFAULT, RADBODY_R_MAX, RADBODY_SIGMA_BODY_MAX, RADBODY_SWEEP_CAP, RADBODY_SWEEP_N,
    RADBODY_SWEEP_POINTS, RADBODY_WORK_MAX,
};
use physsynth_viewer::reed::{C0_AIR, RHO0_AIR};
use serde_json::{json, Value};

/// A short run with a COARSE sweep — the sweep is the whole cost of this payload.
fn rbp(overrides: Value) -> Value {
    let mut p = json!({"model": "radbody", "audio_duration": 0.5, "sweep_points": 3,
                       "sweep_cap": 0.4});
    common::merge(&mut p, overrides);
    p
}

fn nums(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_f64)
        .collect()
}

#[test]
fn conserves_while_the_air_carries_the_energy_away() {
    let d = ok(&rbp(
        json!({"audio_duration": 0.6, "radiation_R": RADBODY_R_DEFAULT}),
    ));
    let e = &d["energy"];
    assert_eq!(e["sigma_is_zero"], true, "R > 0 must NOT flip the gate");
    assert!(f(&e["lossless"]["drift"]) < LOSSLESS_TOL);
    assert_eq!(e["lossless"]["pass"], true);
    let ex = &d["meta"]["exchange"];
    assert_eq!(ex["kind"], "radbody");
    assert!(f(&ex["total_drift"]) < LOSSLESS_TOL);
    assert!(f(&ex["rad_frac_end"]) > 0.5);
    assert!((f(&ex["mech_frac_end"]) - (1.0 - f(&ex["rad_frac_end"]))).abs() <= 1e-9);
}

#[test]
#[allow(clippy::needless_range_loop)] // parallel channels, one index
fn four_channels_sum_to_the_flat_reference_and_e_conn_stays_signed() {
    let ex = ok(&rbp(json!({"radiation_R": RADBODY_R_DEFAULT})))["meta"]["exchange"].clone();
    let n = ex["time"].as_array().unwrap().len();
    assert!(n > 100);
    let keys = [
        "e_string_frac",
        "e_body_frac",
        "e_conn_frac",
        "e_rad_frac",
        "total_frac",
    ];
    let v: Vec<Vec<f64>> = keys.iter().map(|k| nums(&ex[*k])).collect();
    for (k, a) in keys.iter().zip(&v) {
        assert_eq!(a.len(), n, "{k}");
    }
    for i in 0..n {
        assert!((v[0][i] + v[1][i] + v[2][i] + v[3][i] - v[4][i]).abs() <= 1e-9);
        assert!((v[4][i] - 1.0).abs() <= 1e-9);
    }
    assert!(
        v[2].iter().copied().fold(f64::INFINITY, f64::min) < 0.0,
        "E_conn is signed"
    );
    assert_eq!(v[3][0], 0.0);
    assert!(v[3][n - 1] > v[3][n / 2], "the radiated channel only fills");
}

#[test]
fn t50_optimum_is_physics_not_the_scheme_timestep() {
    let p = json!({"model": "radbody"});
    for r in [3.0, 30.0] {
        let t: Vec<f64> = [50, 100, 160]
            .iter()
            .map(|&n| rb::t50(&p, r, 0.8, n).unwrap().0)
            .collect();
        assert!(t.iter().all(|v| v.is_finite()));
        let (mx, mn) = (
            t.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            t.iter().copied().fold(f64::INFINITY, f64::min),
        );
        assert!(
            (mx - mn) / mn < 0.08,
            "t50 must be N-independent at R={r}: {t:?}"
        );
    }
}

#[test]
fn sweep_is_a_controlled_reference_curve_not_the_render() {
    let a = ok(&rbp(json!({"N": 60})))["meta"]["spectrum"]["sweep"].clone();
    let b = ok(&rbp(json!({"N": 140})))["meta"]["spectrum"]["sweep"].clone();
    let c = ok(&rbp(json!({"N": 60, "sigma_body": 20.0})))["meta"]["spectrum"]["sweep"].clone();
    assert_eq!(a, b);
    assert_eq!(a, c);
    assert_eq!(a["sweep_n"], RADBODY_SWEEP_N);
    ok(
        &json!({"model": "radbody", "N": RADBODY_SWEEP_N, "bridge_stiffness": RADBODY_K_MAX,
               "audio_duration": 0.1, "sweep_points": 2, "sweep_cap": 0.1}),
    );
}

#[test]
fn radiated_fraction_is_amplitude_invariant_bit_exactly() {
    let a = ok(&rbp(
        json!({"amplitude": 1e-3, "sweep_points": 2, "sweep_cap": 0.2}),
    ))["meta"]["exchange"]
        .clone();
    let b = ok(&rbp(
        json!({"amplitude": 2e-3, "sweep_points": 2, "sweep_cap": 0.2}),
    ))["meta"]["exchange"]
        .clone();
    assert_eq!(a["e_rad_frac"], b["e_rad_frac"]);
    assert_eq!(a["rad_frac_end"], b["rad_frac_end"]);
}

#[test]
fn more_air_is_worse_the_optimum_is_a_broad_basin() {
    let sw = ok(&rbp(json!({
        "sweep_points": RADBODY_SWEEP_POINTS, "sweep_cap": RADBODY_SWEEP_CAP,
        "audio_duration": 0.3,
    })))["meta"]["spectrum"]["sweep"]
        .clone();
    assert_eq!(sw["skipped"], false);
    assert_eq!(sw["n_censored"], 0);
    let t = nums(&sw["t50_ms"]);
    let best = f(&sw["best_r"]);
    assert!(1.5 < best && best < 20.0, "{best}");
    let (lo, hi) = (f(&sw["basin"][0]), f(&sw["basin"][1]));
    assert!(lo <= best && best <= hi && hi / lo < 20.0);
    let bt = f(&sw["best_t50_ms"]);
    assert!(t[0] > 3.0 * bt && t[t.len() - 1] > 3.0 * bt);
}

#[test]
fn k_zero_skips_the_sweep_instead_of_drawing_a_row_of_nans() {
    let d = ok(&rbp(json!({"bridge_stiffness": 0.0})));
    let sw = &d["meta"]["spectrum"]["sweep"];
    assert_eq!(sw["skipped"], true);
    assert_eq!(sw["steps"], 0);
    assert_eq!(sw["t50_ms"], json!([]));
    assert!(sw["note"].as_str().unwrap().contains("coupling"));
    let ex = &d["meta"]["exchange"];
    assert_eq!(f(&ex["rad_frac_end"]), 0.0);
    assert_eq!(f(&ex["body_frac_peak"]), 0.0);
    assert_eq!(f(&d["audio"]["peak"]), 0.0);
}

#[test]
fn sigma_body_gates_the_verdict_and_radiation_never_does() {
    let loud = ok(&rbp(
        json!({"radiation_R": RADBODY_R_MAX, "sigma_body": 0.0}),
    ));
    assert_eq!(loud["energy"]["sigma_is_zero"], true);
    assert_eq!(loud["energy"]["lossless"]["pass"], true);
    let lossy = ok(&rbp(
        json!({"radiation_R": RADBODY_R_MAX, "sigma_body": 20.0}),
    ));
    assert_eq!(lossy["energy"]["sigma_is_zero"], false);
    assert_eq!(lossy["energy"]["lossy"]["monotone"], true);
    assert!(lossy["energy"]["lossy"].get("measured_2sigma").is_none());
}

#[test]
fn the_load_turns_the_reservoir_into_a_conduit() {
    let peaks: Vec<f64> = [0.0, 1.0, 10.0, RADBODY_R_DEFAULT]
        .iter()
        .map(|&r| {
            f(&ok(&rbp(
                json!({"radiation_R": r, "sweep_points": 2, "sweep_cap": 0.2}),
            ))["meta"]["exchange"]["body_frac_peak"])
        })
        .collect();
    assert!(peaks[0] > 0.5);
    assert!(peaks[3] < 0.01);
    assert!(peaks.windows(2).all(|w| w[0] > w[1]), "{peaks:?}");
}

#[test]
fn f_match_names_the_one_frequency_where_load_and_readout_agree() {
    let sp = ok(&rbp(
        json!({"radiation_R": RADBODY_R_DEFAULT, "sweep_points": 2,
                            "sweep_cap": 0.2}),
    ))["meta"]["spectrum"]
        .clone();
    let modes = nums(&sp["body_modes"]);
    assert!((f(&sp["f_match"]) - modes[0]).abs() <= 1.0);
    let back = monopole_radiation_resistance(
        2.0 * std::f64::consts::PI * f(&sp["f_match"]),
        RHO0_AIR,
        C0_AIR,
    );
    assert!((back - RADBODY_R_DEFAULT).abs() <= 1e-3 * RADBODY_R_DEFAULT);
    let r_phys = nums(&sp["r_phys"]);
    let want = (modes[3] / modes[0]).powi(2);
    assert!((r_phys[3] / r_phys[0] - want).abs() <= 1e-3 * want);
    assert!(f(&sp["sigma_ratio"]) > 1.0);
}

#[test]
fn shipped_sweep_settings_are_the_measured_ones() {
    // Expensive ON PURPOSE: the only run here with the override keys ABSENT, which is what ships.
    assert_eq!(
        (RADBODY_SWEEP_POINTS, RADBODY_SWEEP_CAP, RADBODY_SWEEP_N),
        (18, 0.8, 100)
    );
    assert_eq!(RADBODY_R_DEFAULT, 133.0);
    let sw = ok(&json!({"model": "radbody", "audio_duration": 0.3}))["meta"]["spectrum"]["sweep"]
        .clone();
    assert_eq!(sw["r"].as_array().unwrap().len(), 18);
    assert_eq!(sw["t50_ms"].as_array().unwrap().len(), 18);
    assert!((f(&sw["cap_ms"]) - 800.0).abs() < 1e-9);
    assert_eq!(sw["truncated"], false);
    for bad in [
        json!({"sweep_points": 999}),
        json!({"sweep_points": "x"}),
        json!({"sweep_cap": 9.0}),
    ] {
        let mut p = json!({"model": "radbody"});
        common::merge(&mut p, bad.clone());
        assert_eq!(sim(&p)["error"]["kind"], "param", "{bad}");
    }
}

#[test]
fn sweep_truncation_censors_the_tail_rather_than_hanging() {
    // The budget cannot trip in the shipped range; driven past itself at 5,000 steps, which the
    // reference reached by patching the constant.
    let p = rbp(
        json!({"sweep_points": RADBODY_SWEEP_POINTS, "audio_duration": 0.2,
                       "sweep_cap": 0.4}),
    );
    let sw = rb::build_payload_with(&p, 5_000).unwrap()["meta"]["spectrum"]["sweep"].clone();
    assert_eq!(sw["truncated"], true);
    assert_eq!(sw["skipped"], false);
    assert!(sw["n_censored"].as_i64().unwrap() > 0);
    let t = sw["t50_ms"].as_array().unwrap();
    assert!(t[t.len() - 1].is_null(), "censored, not extrapolated");
    assert!(sw["steps"].as_i64().unwrap() >= 5_000);
    assert_eq!(t.len(), 18, "censored points keep their place on the axis");
}

#[test]
fn frame_and_grid_bookkeeping_line_up_and_the_nut_stays_clamped() {
    let d = ok(&rbp(json!({"audio_duration": 0.4})));
    let w = d["frames"]["width"].as_u64().unwrap() as usize;
    let n = d["frames"]["n_frames"].as_u64().unwrap() as usize;
    assert_eq!(d["frames"]["dims"], 1);
    assert_eq!(w, d["grid"]["x"].as_array().unwrap().len());
    assert_eq!(n, d["frame_times"].as_array().unwrap().len());
    let fr = decode_f32(d["frames"]["b64"].as_str().unwrap());
    assert_eq!(fr.len(), n * w);
    assert!((0..n).all(|k| fr[k * w] == 0.0), "the nut is fixed");
    assert!(
        (0..n).any(|k| fr[k * w + w - 1] != 0.0),
        "the bridge end moves"
    );
}

#[test]
fn audio_is_the_far_field_pressure_real_and_normalized() {
    let d = ok(&rbp(json!({"audio_duration": 0.4, "distance": 2.0})));
    let a = &d["audio"];
    let sig = decode_f32(a["b64"].as_str().unwrap());
    assert_eq!(sig.len() as u64, a["n"].as_u64().unwrap());
    assert!(sig.iter().all(|v| v.is_finite() && v.abs() <= 1.0 + 1e-6));
    assert!(f(&a["peak"]) > 0.0);
    let sp = &d["meta"]["spectrum"];
    assert!((f(&sp["gain_times_r"]) - f(&sp["gain"]) * 2.0).abs() <= 1e-6);
}

#[test]
fn guards_and_budget_are_clean_error_payloads() {
    for bad in [
        json!({"N": RADBODY_N_MAX + 1}),
        json!({"radiation_R": RADBODY_R_MAX + 1.0}),
        json!({"sigma_body": RADBODY_SIGMA_BODY_MAX + 1.0}),
        json!({"distance": RADBODY_DISTANCE_MAX + 1.0}),
        json!({"audio_duration": RADBODY_AUDIO_MAX + 1.0}),
        json!({"bridge_stiffness": RADBODY_K_MAX + 1.0}),
        json!({"lambda": 1.0}),
        json!({"radiation_R": -1.0}),
    ] {
        let mut p = json!({"model": "radbody", "sweep_points": 2});
        common::merge(&mut p, bad.clone());
        assert_eq!(sim(&p)["error"]["kind"], "param", "{bad}");
    }
    let over = sim(
        &json!({"model": "radbody", "N": RADBODY_N_MAX, "rho": 0.001,
                           "audio_duration": RADBODY_AUDIO_MAX, "sweep_points": 2}),
    );
    assert_eq!(over["error"]["kind"], "param");
    assert!(over["error"]["message"]
        .as_str()
        .unwrap()
        .contains("budget"));
    assert_eq!(RADBODY_WORK_MAX, 200_000);
    let tight = sim(
        &json!({"model": "radbody", "N": 40, "bridge_stiffness": 12000.0,
                            "sweep_points": 2, "audio_duration": 0.1}),
    );
    assert_eq!(tight["error"]["kind"], "construction");
}

#[test]
fn ignores_params_that_belong_to_other_models() {
    let base = ok(&rbp(
        json!({"radiation_R": 50.0, "sweep_points": 2, "sweep_cap": 0.2}),
    ));
    let noisy = ok(&rbp(json!({
        "radiation_R": 50.0, "sweep_points": 2, "sweep_cap": 0.2, "K": 2.0e6, "alpha": 2.3,
        "depth": 1e-3, "kappa": 5.0, "EA": 1e4, "sigma_plate": 3.0, "n_plate": 12,
    })));
    assert_eq!(noisy["meta"]["exchange"], base["meta"]["exchange"]);
    assert_eq!(noisy["meta"]["spectrum"], base["meta"]["spectrum"]);
    assert_eq!(noisy["energy"], base["energy"]);
}
