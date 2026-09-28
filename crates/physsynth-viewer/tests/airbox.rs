//! String -> body -> a 3-D room through the payload builder — `test_web_backend.py`'s airbox
//! section.
//!
//! The claim is an INDEX: a mic's first nonzero sample lands exactly on the Manhattan distance in
//! cells (the lattice light cone), independent of the Courant number. The money test beside it is
//! the cross-ledger residual, because the scene total cannot see a wrong coupling constant.

mod common;

use common::{decode_f32, f, ok, sim};
use physsynth_viewer::airbox::{
    np_percentile, AIRBOX_AUDIO_MAX, AIRBOX_CFL_MAX, AIRBOX_H_MAX, AIRBOX_H_MIN, AIRBOX_MIC_MAX,
    AIRBOX_NODE_MAX, AIRBOX_N_MAX, AIRBOX_SIZE_MAX, AIRBOX_ZETA_MAX,
};
use physsynth_viewer::energy::LOSSLESS_TOL;
use physsynth_viewer::membrane::DISPLAY_MAX;
use physsynth_viewer::py::has_nonfinite;
use serde_json::{json, Value};
use std::collections::BTreeSet;

/// A SHORT room run: the room costs seconds per simulated second.
fn ab(overrides: Value) -> Value {
    let mut p = json!({"model": "airbox", "audio_duration": 0.05});
    common::merge(&mut p, overrides);
    p
}

#[test]
fn the_cone_is_the_manhattan_cell_count_exactly() {
    let a = ok(&ab(json!({})))["meta"]["arrival"].clone();
    assert_eq!(a["measured_cells"], a["manhattan_cells"], "{a}");
    assert_eq!(a["match"], true);
    // meaningful only because the three candidate answers disagree
    assert!(f(&a["euclid_steps"]) > f(&a["manhattan_cells"]));
    assert!(f(&a["manhattan_steps_physical"]) > f(&a["euclid_steps"]));
}

#[test]
fn the_cone_tracks_the_mic_and_never_stops_matching() {
    let mut seen = Vec::new();
    for frac in [0.2, 0.5, 0.8, 1.0] {
        let a = ok(&ab(json!({"mic_position": frac})))["meta"]["arrival"].clone();
        assert_eq!(a["match"], true, "{frac}");
        seen.push(a["manhattan_cells"].as_i64().unwrap());
    }
    assert!(seen.windows(2).all(|w| w[0] < w[1]), "{seen:?}");
}

/// The cone is a property of the STENCIL, so it must not move with the Courant number, while the
/// physical amplitude-threshold arrivals beside it do.
#[test]
fn the_cone_is_lambda_independent_while_the_amplitude_arrivals_are_not() {
    let (mut cones, mut rates, mut thresh) = (BTreeSet::new(), BTreeSet::new(), BTreeSet::new());
    for cfl in [0.4, 0.6, 0.9] {
        let d = ok(&ab(json!({"air_cfl": cfl})));
        let a = &d["meta"]["arrival"];
        assert_eq!(a["match"], true, "{cfl}");
        cones.insert(a["manhattan_cells"].as_i64().unwrap());
        rates.insert(f(&d["fs_sim"]).round_ties_even() as i64);
        let t: Vec<i64> = a["thresholds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|q| q["steps"].as_i64().unwrap())
            .collect();
        thresh.insert(t);
    }
    assert_eq!(cones.len(), 1, "the cone MOVED with lambda: {cones:?}");
    assert_eq!(
        rates.len(),
        3,
        "the sample rates must differ or the test is vacuous"
    );
    assert!(
        thresh.len() > 1,
        "the amplitude arrivals should move with lambda"
    );
}

#[test]
fn the_money_test_is_the_cross_ledger_residual() {
    let g = ok(&ab(json!({})))["meta"]["ledger"].clone();
    assert_eq!(g["kind"], "airbox");
    assert!(f(&g["residual_max"]) < 1e-10, "{}", g["residual_max"]);
    assert!(f(&g["total_drift"]).abs() < LOSSLESS_TOL);
    assert!(g["e_stride"].as_u64().unwrap() >= 1 && g["n_samples"].as_u64().unwrap() > 100);
}

/// The room sets fs, so the string's lambda is derived, and a COARSER room raises fs.
#[test]
fn the_room_sets_fs_and_the_strings_lambda_is_derived() {
    let fine = ok(&ab(json!({"air_h": 0.025})));
    let coarse = ok(&ab(json!({"air_h": 0.05, "N": 60})));
    assert!(f(&fine["fs_sim"]) > f(&coarse["fs_sim"]));
    for d in [&fine, &coarse] {
        let r = &d["meta"]["room"];
        let ls = f(&r["lam_string"]);
        assert!(0.0 < ls && ls < 1.0);
        assert!(f(&r["lam_air"]) <= 1.0 / 3.0f64.sqrt() + 1e-12);
        assert!((f(&d["lambda"]) - ls).abs() < 1e-12);
    }
    let bad = sim(&ab(json!({"air_h": 0.05, "N": 140})));
    assert!(bad["error"]["message"]
        .as_str()
        .unwrap()
        .contains("derived"));
}

#[test]
fn it_ships_three_named_slices_not_a_volume() {
    let d = ok(&ab(json!({})));
    let fr = &d["frames"];
    assert_eq!(fr["dims"], 3);
    assert_eq!(fr["kind"], "slices");
    let planes = fr["planes"].as_array().unwrap();
    let names: Vec<&str> = planes.iter().map(|p| p["name"].as_str().unwrap()).collect();
    assert_eq!(names, ["xy", "xz", "yz"]);
    let per_frame: u64 = planes
        .iter()
        .map(|p| p["nu"].as_u64().unwrap() * p["nv"].as_u64().unwrap())
        .sum();
    assert_eq!(fr["width"], per_frame);
    let nf = fr["n_frames"].as_u64().unwrap();
    let field = decode_f32(fr["b64"].as_str().unwrap());
    assert_eq!(field.len() as u64, per_frame * nf);
    assert!(field.iter().all(|v| v.is_finite()));
    assert_eq!(d["frame_times"].as_array().unwrap().len() as u64, nf);
    assert!(nf > 1);
    let n = &d["meta"]["room"]["N"];
    let vol: u64 = (0..3).map(|i| n[i].as_u64().unwrap() + 1).product();
    assert!((per_frame as f64) < 0.25 * vol as f64);
}

/// A room finer than the display cap shrinks the planes and says so.
#[test]
fn the_slices_decimate_and_report_their_own_stride() {
    let d = ok(&ab(
        json!({"air_h": AIRBOX_H_MIN, "room_size": 1.2, "audio_duration": 0.02}),
    ));
    let planes = d["frames"]["planes"].as_array().unwrap();
    for pl in planes {
        let (nu, nv, s) = (
            pl["nu"].as_u64().unwrap() as usize,
            pl["nv"].as_u64().unwrap() as usize,
            pl["stride"].as_u64().unwrap() as usize,
        );
        assert!(nu <= DISPLAY_MAX && nv <= DISPLAY_MAX && s >= 1);
        assert_eq!(
            nu,
            (0..pl["nu_full"].as_u64().unwrap() as usize)
                .step_by(s)
                .count()
        );
        assert_eq!(
            nv,
            (0..pl["nv_full"].as_u64().unwrap() as usize)
                .step_by(s)
                .count()
        );
    }
    assert!(planes.iter().any(|pl| pl["stride"].as_u64().unwrap() > 1));
}

#[test]
fn the_colour_mapping_travels_with_the_values() {
    let sc = ok(&ab(json!({})))["frames"]["scale"].clone();
    assert_eq!(sc["map"], "asinh");
    assert!(f(&sc["ref"]) > 0.0 && f(&sc["amp"]) >= f(&sc["ref"]));
    let pctl = f(&sc["pctl"]);
    assert!(0.0 < pctl && pctl < 100.0);
}

/// `np.percentile`'s default method, to the last digit (values printed by NumPy 2.4). The
/// reference ships the colour reference unrounded, so its arithmetic is part of the payload.
#[test]
fn the_percentile_is_numpys_linear_method_to_the_last_digit() {
    let a = [0.3, 1.7, 2.2, 0.05, 9.1, 4.4, 3.3];
    for (q, want) in [
        (55.0, 2.530_000_000_000_000_2),
        (50.0, 2.2),
        (12.5, 0.2375),
        (99.0, 8.817_999_999_999_998),
        (0.0, 0.05),
        (100.0, 9.1),
    ] {
        assert_eq!(np_percentile(&a, q), want, "q = {q}");
    }
    // `np.linspace(0.1, 1.0, 11) ** 3`, as NumPy printed it
    let b = [
        0.001_000_000_000_000_000_2,
        0.006_859_000_000_000_000_5,
        0.021_952_000_000_000_006,
        0.050_653,
        0.097_335_999_999_999_98,
        0.166_374_999_999_999_94,
        0.262_144_000_000_000_04,
        0.389_016_999_999_999_95,
        0.551_367_999_999_999_9,
        0.753_570_999_999_999_8,
        1.0,
    ];
    assert_eq!(np_percentile(&b, 55.0), 0.214_259_5);
}

#[test]
fn the_grid_snap_is_reported_never_silently_resampled() {
    let r = ok(&ab(json!({"air_h": 0.03, "room_size": 1.0})))["meta"]["room"].clone();
    assert_ne!(r["L"], r["L_requested"], "this h/L pair should snap");
    for i in 0..3 {
        let (actual, n) = (f(&r["L"][i]), f(&r["N"][i]));
        assert!((actual - n * f(&r["h"])).abs() <= 1e-9 * actual);
    }
}

/// `open` (Z = 0) reflects with inversion and dissipates NOTHING, so it stays on the lossless bar;
/// only `absorbing` is lossy.
#[test]
fn open_walls_are_lossless_and_only_absorbing_is_not() {
    for walls in ["rigid", "open"] {
        let d = ok(&ab(json!({"walls": walls})));
        assert!(d["energy"].get("lossless").is_some(), "{walls}");
        assert_eq!(d["energy"]["lossless"]["pass"], true);
        assert_eq!(f(&d["meta"]["ledger"]["dissipated_frac_end"]), 0.0);
    }
    let lossy = ok(&ab(json!({"walls": "absorbing"})));
    assert!(lossy["energy"].get("lossy").is_some());
    assert!(f(&lossy["meta"]["ledger"]["dissipated_frac_end"]) > 0.0);
    assert_eq!(lossy["energy"]["lossy"]["monotone"], true);
    assert_eq!(lossy["meta"]["room"]["walls"], "absorbing (zeta = 3)");
}

#[test]
fn the_payload_survives_strict_json() {
    for walls in ["rigid", "absorbing", "open"] {
        assert!(!has_nonfinite(&ok(&ab(json!({"walls": walls})))), "{walls}");
    }
    for bad in [
        json!({"air_cfl": 1.0}),
        json!({"N": AIRBOX_N_MAX + 1}),
        json!({"walls": "squishy"}),
    ] {
        assert!(!has_nonfinite(&sim(&ab(bad))));
    }
}

#[test]
fn guards_and_budget_are_clean_error_payloads() {
    for bad in [
        json!({"N": AIRBOX_N_MAX + 1}),
        json!({"air_cfl": AIRBOX_CFL_MAX + 0.01}),
        json!({"air_cfl": 0.1}),
        json!({"air_h": AIRBOX_H_MAX + 0.01}),
        json!({"air_h": 0.001}),
        json!({"room_size": AIRBOX_SIZE_MAX + 0.1}),
        json!({"room_size": 0.1}),
        json!({"walls": "squishy"}),
        json!({"walls": "absorbing", "wall_zeta": 0.0}),
        json!({"walls": "absorbing", "wall_zeta": AIRBOX_ZETA_MAX + 1.0}),
        json!({"mic_position": AIRBOX_MIC_MAX + 0.1}),
        json!({"slice_position": 1.5}),
        json!({"audio_duration": AIRBOX_AUDIO_MAX + 0.1}),
        json!({"audio_duration": 0.0}),
        json!({"sigma_body": -1.0}),
        json!({"pluck_position": 1.0}),
    ] {
        assert_eq!(sim(&ab(bad.clone()))["error"]["kind"], "param", "{bad}");
    }
    let nodes = sim(&ab(
        json!({"air_h": AIRBOX_H_MIN, "room_size": AIRBOX_SIZE_MAX}),
    ));
    let msg = nodes["error"]["message"].as_str().unwrap().replace(',', "");
    assert!(msg.contains(&AIRBOX_NODE_MAX.to_string()), "{msg}");
    let work = sim(&ab(
        json!({"air_h": 0.021, "audio_duration": AIRBOX_AUDIO_MAX}),
    ));
    assert!(work["error"]["message"]
        .as_str()
        .unwrap()
        .contains("work budget"));
    let stiff = sim(&ab(json!({"bridge_stiffness": 5.0e6})));
    assert_eq!(stiff["error"]["kind"], "construction");
}

#[test]
fn it_ignores_params_that_belong_to_other_models() {
    let base = ok(&ab(json!({})));
    let noisy = ok(&ab(
        json!({"K": 2.0e6, "alpha": 2.3, "depth": 1e-3, "kappa": 5.0, "EA": 1e4,
                              "sigma_plate": 3.0, "n_plate": 12, "radiation_R": 133.0,
                              "air_corner": 1091.8, "radiation_weight": 0.5, "distance": 2.0}),
    ));
    assert_eq!(noisy["meta"]["arrival"], base["meta"]["arrival"]);
    assert_eq!(noisy["meta"]["ledger"], base["meta"]["ledger"]);
    assert_eq!(noisy["energy"], base["energy"]);
}
