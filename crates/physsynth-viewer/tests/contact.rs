//! The barrier family (model #8) through the payload builder — `test_web_backend.py`'s jawari,
//! juari and fret sections.
//!
//! Each scene runs the geometry its model's own suite validated, so the suite's numbers come back
//! through the payload as a free end-to-end oracle: the jawari's ~3.4x shimmer, the juari's
//! position-selective tuning curve, the fret's ~1.24 slaps per period at a ~15 % duty.

mod common;

use common::{decode_b64, decode_f32, f, ok, sim};
use physsynth_viewer::contact::{
    FRET_AUDIO_MAX, FRET_BRIGHTNESS_PEAK, FRET_CONTROL_MAX, FRET_DUTY_MAX, FRET_EPISODES_MIN,
    FRET_N_MAX, FRET_RAIL_FRAC_MIN, FRET_RASTER_COLS_PER_PERIOD, FRET_RASTER_MAX_ROWS,
    FRET_WORK_MAX, JAWARI_AMP_MAX, JAWARI_DEPTH_MAX, JAWARI_ELEVATION_GATE, JAWARI_N_MAX,
    JAWARI_RATIO_FLOOR, JAWARI_WORK_MAX, JUARI_AMP_MAX, JUARI_ELEVATION_GATE, JUARI_N_MAX,
    JUARI_SWEEP_DUR, JUARI_WORK_MAX,
};
use physsynth_viewer::energy::LOSSLESS_TOL;
use physsynth_viewer::AUDIO_FS;
use serde_json::{json, Value};

fn with(base: Value, overrides: Value) -> Value {
    let mut p = base;
    common::merge(&mut p, overrides);
    p
}

fn jaw(o: Value) -> Value {
    sim(&with(json!({"model": "jawari", "audio_duration": 0.24}), o))
}

/// Small `N` and a tiny non-canonical sweep: the per-node construction dominates the cost.
fn jua(o: Value) -> Value {
    sim(&with(
        json!({"model": "juari", "N": 40, "sweep_duration": 0.04, "audio_duration": 0.05}),
        o,
    ))
}

fn fret(o: Value) -> Value {
    sim(&with(json!({"model": "fret", "audio_duration": 0.4}), o))
}

fn nums(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_f64)
        .collect()
}

fn is_err(d: &Value) -> bool {
    d.get("error").is_some()
}

fn close(a: f64, b: f64, abs: f64) -> bool {
    (a - b).abs() <= abs
}

fn assert_audio_real(d: &Value) {
    let a = &d["audio"];
    assert_eq!(f(&a["fs"]), AUDIO_FS);
    let sig = decode_f32(a["b64"].as_str().unwrap());
    assert!(sig.iter().all(|v| v.is_finite()));
    assert!(sig.iter().fold(0.0f32, |m, v| m.max(v.abs())) > 0.0);
}

// == jawari ========================================================================================

#[test]
fn jawari_reproduces_the_suites_shimmer_and_wrap_numbers() {
    let sp = jaw(json!({}))["meta"]["spectrum"].clone();
    assert_eq!(sp["kind"], "jawari");
    assert!(
        close(f(&sp["elevation"]), 3.44, 0.05),
        "{}",
        sp["elevation"]
    );
    assert_eq!(sp["shimmering"], true);
    assert!(f(&sp["elevation"]) > JAWARI_ELEVATION_GATE);
    assert!(close(f(&sp["wrap"]["std"]), 4.89, 0.05));
    assert_eq!(sp["wrap"]["min_node"], 0);
    assert_eq!(sp["wrap"]["max_node"], 14);
    // the clean contrast is spectrally pure — what makes the elevation attributable
    let f1 = f(&sp["f1"]);
    assert!((f(&sp["centroid"]["clean_late"]) - f1).abs() <= 0.02 * f1);
}

#[test]
fn jawari_energy_keeps_the_flat_loss_oracle_unlike_the_mallet() {
    let e = jaw(json!({"sigma0": 0.5}))["energy"].clone();
    assert_eq!(e["sigma_is_zero"], false);
    assert_eq!(e["lossy"]["monotone"], true);
    assert!(close(f(&e["lossy"]["oracle_2sigma"]), 1.0, 1e-12));
    assert!((f(&e["lossy"]["measured_2sigma"]) - 1.0).abs() <= 0.02);
}

#[test]
fn jawari_sigma0_gates_the_verdict_and_conserves_through_the_curved_wrap() {
    let d = jaw(json!({"sigma0": 0.0}));
    let e = &d["energy"];
    assert_eq!(e["sigma_is_zero"], true);
    assert!(f(&e["lossless"]["drift"]) < LOSSLESS_TOL);
    assert_eq!(e["lossless"]["pass"], true);
    assert!(
        f(&d["meta"]["spectrum"]["wrap"]["duty"]) > 0.1,
        "must actually be in contact"
    );
}

#[test]
fn jawari_grazing_config_is_labelled_not_failed() {
    let sp = jaw(json!({"depth": 6.0e-3}))["meta"]["spectrum"].clone();
    assert_eq!(sp["grazing"], true);
    assert!(f(&sp["ratio"]) < JAWARI_RATIO_FLOOR);
    assert_eq!(sp["shimmering"], false);
    assert!(f(&sp["elevation"]) < JAWARI_ELEVATION_GATE);
    assert!(
        sp["wrap"]["max_node"].as_i64().unwrap() < 14,
        "the wrap contracts to the crest"
    );
}

#[test]
fn jawari_ratio_is_the_control_and_amplitude_moves_it_as_hard_as_depth() {
    let base = jaw(json!({}))["meta"]["spectrum"].clone();
    let quiet = jaw(json!({"amplitude": 2.0e-3}))["meta"]["spectrum"].clone();
    // the ratio is rounded to 2 dp
    assert!(close(f(&quiet["ratio"]), f(&base["ratio"]) / 4.0, 0.01));
    assert!(f(&quiet["elevation"]) < f(&base["elevation"]));
}

#[test]
fn jawari_ignores_the_mallet_alpha_and_the_sympathetic_k() {
    let base = jaw(json!({}))["meta"]["spectrum"].clone();
    let leaked = jaw(json!({"alpha": 2.3, "K": 8000}))["meta"]["spectrum"].clone();
    assert_eq!(leaked["elevation"], base["elevation"]);
    assert_eq!(leaked["wrap"]["std"], base["wrap"]["std"]);
    let bites = jaw(json!({"bridge_stiffness": 2.0e5}))["meta"]["spectrum"]["elevation"].clone();
    assert_ne!(
        bites, base["elevation"],
        "the name it DOES read still bites"
    );
}

#[test]
fn jawari_payload_carries_the_bridge_profile_and_the_wrap_marker() {
    let d = jaw(json!({}));
    let b = d["grid"]["barrier"].as_array().unwrap();
    assert_eq!(b.len(), d["grid"]["x"].as_array().unwrap().len());
    let finite: Vec<f64> = b.iter().filter_map(Value::as_f64).collect();
    assert_eq!(
        finite.len() as u64,
        d["meta"]["spectrum"]["wrap"]["support"].as_u64().unwrap()
    );
    assert!(
        finite.iter().all(|&v| v <= 0.0),
        "the bridge sits at or below the rest line"
    );
    assert!(
        finite.windows(2).all(|w| w[0] >= w[1]),
        "a parabola falling from the crest"
    );
    let wf: Vec<i64> = d["wrap_frames"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    assert_eq!(wf.len() as u64, d["frames"]["n_frames"].as_u64().unwrap());
    // GRID node indices, not support-relative ones: every marked node is ON the bridge
    assert!(wf.iter().all(|&w| w == -1 || !b[w as usize].is_null()));
    assert!(
        wf.iter().any(|&w| w >= 0),
        "the string must contact during the window"
    );
}

#[test]
fn jawari_late_spectra_share_one_scale() {
    let sp = jaw(json!({}))["meta"]["spectrum"]["spectra"].clone();
    let jm = nums(&sp["jawari"]["mag"]);
    let cm = nums(&sp["clean"]["mag"]);
    let (mj, mc) = (
        jm.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        cm.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    );
    assert!(close(mj.max(mc), 1.0, 1e-12));
    assert!(
        mj < 1.0 || mc < 1.0,
        "one trace is the normalizer, not both"
    );
    assert_eq!(
        sp["jawari"]["f"].as_array().unwrap().len(),
        sp["jawari"]["mag"].as_array().unwrap().len()
    );
}

#[test]
fn jawari_audio_is_real_and_finite() {
    assert_audio_real(&jaw(json!({})));
}

#[test]
fn jawari_work_budget_counts_both_runs_and_the_guards_are_reachable() {
    let r = jaw(json!({"audio_duration": 1.5, "N": 128, "width_frac": 0.4}));
    assert!(r["error"]["message"]
        .as_str()
        .unwrap()
        .contains(&JAWARI_WORK_MAX.to_string()));
    for o in [
        json!({"N": JAWARI_N_MAX + 1}),
        json!({"depth": JAWARI_DEPTH_MAX + 1e-3}),
        json!({"amplitude": JAWARI_AMP_MAX + 1e-3}),
        json!({"bridge_stiffness": 0.0}),
    ] {
        assert!(is_err(&jaw(o.clone())), "{o}");
    }
}

#[test]
fn jawari_sustain_ratio_is_reported_but_never_gates() {
    let sp = jaw(json!({}))["meta"]["spectrum"].clone();
    assert!(!sp["sustain_ratio"].is_null() && !sp["clean_sustain_ratio"].is_null());
    assert_eq!(
        sp["shimmering"].as_bool().unwrap(),
        f(&sp["elevation"]) > JAWARI_ELEVATION_GATE
    );
}

// == juari =========================================================================================

#[test]
fn juari_tuning_curve_is_position_selective() {
    let sp = jua(json!({}))["meta"]["spectrum"].clone();
    assert_eq!(sp["kind"], "juari");
    let tn = &sp["tuning"];
    let n = tn["node"].as_array().unwrap().len();
    for key in ["frac", "x", "elevation"] {
        assert_eq!(tn[key].as_array().unwrap().len(), n, "{key}");
    }
    let elev = nums(&tn["elevation"]);
    let (lo, hi) = elev
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
            (a.min(v), b.max(v))
        });
    assert!(
        hi - lo > 0.6,
        "the tuning curve must be position-selective, not flat"
    );
    let f1 = f(&sp["f1"]);
    assert!((f(&sp["centroid"]["clean_late"]) - f1).abs() <= 0.05 * f1);
}

#[test]
fn juari_thread_marker_sits_on_the_drawn_curve() {
    let sp = jua(json!({"thread_position": 0.1}))["meta"]["spectrum"].clone();
    let node = &sp["thread"]["node"];
    let nodes = sp["tuning"]["node"].as_array().unwrap();
    let i = nodes
        .iter()
        .position(|v| v == node)
        .expect("the selected node is swept");
    assert_eq!(sp["thread"]["elevation"], sp["tuning"]["elevation"][i]);
    assert_eq!(sp["thread"]["elevation"], sp["elevation"]);
}

#[test]
fn juari_energy_keeps_the_flat_loss_oracle_like_the_jawari() {
    let e = jua(json!({"sigma0": 0.5}))["energy"].clone();
    assert_eq!(e["sigma_is_zero"], false);
    assert_eq!(e["lossy"]["monotone"], true);
    let o = f(&e["lossy"]["oracle_2sigma"]);
    assert!((f(&e["lossy"]["measured_2sigma"]) - o).abs() <= 0.05 * o);
}

#[test]
fn juari_sigma0_gates_and_conserves_through_the_point_contact() {
    let d = jua(json!({"sigma0": 0.0}));
    let e = &d["energy"];
    assert_eq!(e["sigma_is_zero"], true);
    assert!(f(&e["lossless"]["drift"]) < LOSSLESS_TOL);
    assert_eq!(e["lossless"]["pass"], true);
    let cf: Vec<i64> = d["contact_frames"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    assert!(
        cf.iter().sum::<i64>() > 0,
        "the string must contact the thread"
    );
    assert_eq!(cf.len() as u64, d["frames"]["n_frames"].as_u64().unwrap());
}

#[test]
fn juari_thread_snaps_to_a_grid_node_and_reports_the_resolution() {
    let d = jua(json!({"N": 40, "thread_position": 0.1}));
    let node = d["grid"]["thread_node"].as_u64().unwrap() as usize;
    assert!((1..=39).contains(&node));
    assert!(close(
        f(&d["grid"]["thread_x"]),
        f(&d["grid"]["x"][node]),
        1e-6
    ));
    let q = &d["meta"]["spectrum"]["quantization"];
    assert!(close(f(&q["h"]), 1.0 / 40.0, 1e-4));
    assert_eq!(
        q["near_nut_nodes"].as_i64().unwrap(),
        (f(&q["near_nut_frac"]) * 40.0).round_ties_even() as i64
    );
}

#[test]
fn juari_reference_lines_are_clean_1x_and_a_flat_jawari() {
    let r = jua(json!({}))["meta"]["spectrum"]["reference"].clone();
    assert_eq!(f(&r["clean"]), 1.0);
    assert!(
        f(&r["jawari"]) > 1.0,
        "the curved bridge must be in reach at these params"
    );
}

#[test]
fn juari_tuning_curve_is_decoupled_from_the_audio_length() {
    let a = jua(json!({"audio_duration": 0.06}))["meta"]["spectrum"]["tuning"]["elevation"].clone();
    let b = jua(json!({"audio_duration": 0.1}))["meta"]["spectrum"]["tuning"]["elevation"].clone();
    assert_eq!(a, b);
}

#[test]
fn juari_ignores_the_mallet_alpha_and_the_sympathetic_k() {
    let base = jua(json!({}))["meta"]["spectrum"]["tuning"]["elevation"].clone();
    let leaked =
        jua(json!({"alpha": 2.3, "K": 8000}))["meta"]["spectrum"]["tuning"]["elevation"].clone();
    assert_eq!(leaked, base);
    let changed =
        jua(json!({"bridge_stiffness": 2.0e5}))["meta"]["spectrum"]["tuning"]["elevation"].clone();
    assert_ne!(changed, base);
}

#[test]
fn juari_audio_is_real_and_finite() {
    assert_audio_real(&jua(json!({})));
}

#[test]
fn juari_below_signal_is_labelled_not_failed() {
    let d = jua(json!({"thread_position": 0.5}));
    assert!(!is_err(&d));
    let sp = &d["meta"]["spectrum"];
    assert_eq!(
        sp["buzzing"].as_bool().unwrap(),
        f(&sp["elevation"]) > JUARI_ELEVATION_GATE
    );
}

#[test]
fn juari_guards_are_clean_error_payloads() {
    for o in [
        json!({"thread_position": 1.4}),
        json!({"thread_position": 0.0}),
        json!({"N": JUARI_N_MAX + 1}),
        json!({"lambda": 1.5}),
        json!({"amplitude": JUARI_AMP_MAX + 1e-3}),
        json!({"bridge_stiffness": 0.0}),
        json!({"sweep_duration": JUARI_SWEEP_DUR + 0.1}),
        json!({"N": 128, "sweep_duration": JUARI_SWEEP_DUR, "audio_duration": JUARI_WORK_MAX}),
    ] {
        assert!(is_err(&jua(o.clone())), "{o}");
    }
}

/// The one claim that needs the SETTLED buzz: the sweet spot sits near the nut, and a
/// well-placed thread genuinely rivals the whole curved bridge. Bracketed, not point-pinned.
#[test]
fn juari_sweet_spot_sits_near_the_nut_settled() {
    let sp = jua(json!({"N": 40, "sweep_duration": 0.16, "audio_duration": 0.06}))["meta"]
        ["spectrum"]
        .clone();
    assert!(
        f(&sp["sweet_spot"]["frac"]) <= 0.25,
        "the peak buzz must sit near the nut"
    );
    let e = f(&sp["sweet_spot"]["elevation"]);
    assert!(2.7 < e && e < 3.5, "{e}");
    let j = f(&sp["reference"]["jawari"]);
    assert!(3.0 < j && j < 3.9, "{j}");
}

/// The reference's `N` pre-read, carried as it is. A digit string is `int()`-ed and agrees with
/// the integer. Anything else (`40.0`, which the front-end never sends) snaps the MAIN run's thread
/// on `N = 100` before the validated `N` re-snaps the label — so the swept curve and the marker are
/// right and the audio run's thread sits elsewhere. Pinned so a fix is a decision, not a drift.
#[test]
fn juari_n_pre_read_matches_the_reference_including_its_float_quirk() {
    let base = jua(json!({"thread_position": 0.1}));
    let as_str = jua(json!({"thread_position": 0.1, "N": "40"}));
    assert_eq!(as_str["meta"]["spectrum"], base["meta"]["spectrum"]);
    let as_float = jua(json!({"thread_position": 0.1, "N": 40.0}));
    assert_eq!(
        as_float["meta"]["spectrum"]["tuning"],
        base["meta"]["spectrum"]["tuning"]
    );
    assert_eq!(as_float["grid"]["thread_node"], base["grid"]["thread_node"]);
    assert_ne!(
        as_float["meta"]["spectrum"]["spectra"], base["meta"]["spectrum"]["spectra"],
        "the main run's thread is at round(0.1 * 100) = 10, not the labelled node 4"
    );
}

// == fret ==========================================================================================

#[test]
fn fret_reproduces_the_probes_intermittency_numbers() {
    let ct = ok(&with(
        json!({"model": "fret", "audio_duration": 0.4}),
        json!({}),
    ))["meta"]["contact"]
        .clone();
    assert_eq!(ct["kind"], "fret");
    assert!(close(f(&ct["duty"]), 0.154, 0.005));
    assert!(close(f(&ct["episodes_per_period"]), 1.24, 0.05));
    assert_eq!(ct["intermittent"], true);
    assert_eq!(ct["out_of_reach"], false);
    assert_eq!(ct["pinned"], false);
}

#[test]
fn fret_active_set_is_a_vector_and_the_newton_is_cheap() {
    let ct = fret(json!({}))["meta"]["contact"].clone();
    assert_eq!(ct["support"], 99);
    let am = ct["active_max"].as_i64().unwrap();
    assert!((am - 69).abs() <= 3 && am > 1, "{am}");
    assert!(ct["iters_max"].as_i64().unwrap() <= 3);
    assert!(close(f(&ct["iters_mean"]), 1.16, 0.05));
}

#[test]
fn fret_sigma0_gates_the_verdict_and_conserves_through_genuine_contact() {
    let d = fret(json!({"sigma0": 0.0}));
    let e = &d["energy"];
    assert_eq!(e["sigma_is_zero"], true);
    assert!(f(&e["lossless"]["drift"]) < LOSSLESS_TOL);
    assert_eq!(e["lossless"]["pass"], true);
    assert!(
        f(&d["meta"]["contact"]["duty"]) > 0.1,
        "must actually be in contact"
    );
}

#[test]
fn fret_drops_the_jawari_decay_oracle_and_ships_the_triple_instead() {
    let e = fret(json!({"sigma0": 0.5}))["energy"].clone();
    assert_eq!(e["sigma_is_zero"], false);
    assert_eq!(e["lossy"]["monotone"], true);
    assert!(
        e["lossy"].get("measured_2sigma").is_none(),
        "must NOT inherit the jawari's oracle"
    );
    assert!(e["lossy"].get("oracle_2sigma").is_none());
    let t = &e["decay_triple"];
    assert!(close(f(&t["oracle_2sigma"]), 1.0, 1e-12));
    assert!(
        f(&t["ratio"]) > 1.03,
        "the naive oracle is measurably WRONG here"
    );
    assert!(f(&t["agreement"]) < 0.01);
    assert!((f(&t["corrected"]) - f(&t["rate"])).abs() <= 0.01 * f(&t["rate"]));
}

#[test]
fn fret_equipartition_correction_tracks_the_rate_across_clearances() {
    let mut ratios = Vec::new();
    for clearance in [4.0e-3, 2.0e-3, 1.0e-3] {
        let t = fret(json!({"audio_duration": 0.25, "clearance": clearance}))["energy"]
            ["decay_triple"]
            .clone();
        assert!(f(&t["agreement"]) < 0.01);
        ratios.push(f(&t["ratio"]));
    }
    assert!(
        ratios[0] < ratios[2],
        "a closer rail breaks equipartition harder"
    );
    assert!(ratios[0] > 1.0);
}

#[test]
fn fret_out_of_reach_rail_is_labelled_not_failed() {
    let ct = fret(json!({"clearance": 6.0e-3}))["meta"]["contact"].clone();
    assert_eq!(ct["out_of_reach"], true);
    assert_eq!(ct["intermittent"], false);
    assert_eq!(f(&ct["duty"]), 0.0);
    assert_eq!(ct["episodes"], 0);
    assert_eq!(ct["active_max"], 0);
}

#[test]
fn fret_rail_frac_floor_is_enforced_server_side() {
    assert!(is_err(&fret(
        json!({"rail_frac": FRET_RAIL_FRAC_MIN - 0.05})
    )));
    let ct = fret(json!({"rail_frac": FRET_RAIL_FRAC_MIN}))["meta"]["contact"].clone();
    assert!(f(&ct["duty"]) > 0.0);
    assert_eq!(
        ct["intermittent"], true,
        "the floor must still reach the rail"
    );
}

/// A lossless one-sided spring always pushes back, so the string never comes to REST on the rail:
/// softening it raises the duty, but toward the free-sinusoid limit 0.5, not toward pinning.
#[test]
fn fret_intermittency_is_structural_not_a_tuned_accident() {
    let mut duties = Vec::new();
    for stiffness in [2.0e6, 2.0e4, 2.0e2] {
        let ct = fret(json!({"audio_duration": 0.08, "rail_stiffness": stiffness,
                             "clearance": 5.0e-4}))["meta"]["contact"]
            .clone();
        assert_eq!(ct["intermittent"], true);
        assert_eq!(ct["pinned"], false);
        duties.push(f(&ct["duty"]));
    }
    assert!(
        duties[0] < duties[2],
        "a softer rail admits the string for longer"
    );
    assert!(duties.iter().all(|&d| d < 0.5) && 0.5 < FRET_DUTY_MAX);
}

#[test]
fn fret_scalars_are_computed_at_full_rate_not_read_off_the_raster() {
    let ct = fret(json!({}))["meta"]["contact"].clone();
    let r = &ct["raster"];
    let img = decode_b64(r["b64"].as_str().unwrap());
    let (rows, cols) = (
        r["n_rows"].as_u64().unwrap() as usize,
        r["n_cols"].as_u64().unwrap() as usize,
    );
    let lit = (0..cols)
        .filter(|&c| (0..rows).any(|rw| img[rw * cols + c] > 0))
        .count();
    let column_duty = lit as f64 / cols as f64;
    assert!(
        f(&ct["duty"]) < column_duty,
        "the raster dilates in time; the scalar must not"
    );
    assert!(close(f(&ct["duty"]), 0.154, 0.005));
}

#[test]
fn fret_raster_resolves_at_least_ten_columns_per_period() {
    for duration in [0.1, 0.4, 0.6] {
        let r = fret(json!({"audio_duration": duration}))["meta"]["contact"]["raster"].clone();
        assert!(
            f(&r["cols_per_period"]) >= FRET_RASTER_COLS_PER_PERIOD - 0.5,
            "{duration}"
        );
        assert!(r["n_rows"].as_u64().unwrap() as usize <= FRET_RASTER_MAX_ROWS);
    }
}

#[test]
fn fret_raster_decodes_to_the_grid_and_greys_without_losing_contacts() {
    let d = fret(json!({}));
    let ct = &d["meta"]["contact"];
    let r = &ct["raster"];
    let img = decode_b64(r["b64"].as_str().unwrap());
    let (rows, cols) = (r["n_rows"].as_u64().unwrap(), r["n_cols"].as_u64().unwrap());
    assert_eq!(img.len() as u64, rows * cols);
    assert_eq!(r["n_rows"], ct["support"]);
    assert_eq!(r["x_binned"], false);
    let mx = *img.iter().max().unwrap();
    assert!(mx > 0 && *img.iter().min().unwrap() == 0);
    assert!(f(&r["force_max"]) > 0.0);
    let x = d["grid"]["x"].as_array().unwrap();
    assert_eq!(d["grid"]["barrier"].as_array().unwrap().len(), x.len());
    assert!(f(&r["x0"]) > 0.0 && f(&r["x1"]) < f(&x[x.len() - 1]));
    assert_eq!(ct["trace"]["active"].as_array().unwrap().len() as u64, cols);
    assert_eq!(ct["trace"]["iters"].as_array().unwrap().len() as u64, cols);
}

#[test]
fn fret_brightness_is_reported_with_its_non_monotonicity_named() {
    let sp = fret(json!({}))["meta"]["spectrum"].clone();
    assert!(close(f(&sp["elevation"]), 4.682, 0.02));
    let f1 = f(&sp["f1"]);
    assert!((f(&sp["centroid_control"]) - f1).abs() <= 0.001 * f1);
    assert_eq!(sp["monotone"], false);
    assert_eq!(f(&sp["peak_clearance"]), FRET_BRIGHTNESS_PEAK);
}

/// Brightness is higher at 2 mm than at either side: 4 mm barely reaches, 1 mm rides the rail.
#[test]
fn fret_brightness_peaks_at_an_intermediate_clearance() {
    let el = |c: f64| {
        f(&fret(json!({"audio_duration": 0.25, "clearance": c}))["meta"]["spectrum"]["elevation"])
    };
    let peak = el(2.0e-3);
    for side in [4.0e-3, 1.0e-3] {
        assert!(
            peak > el(side),
            "brightness must peak at an INTERMEDIATE clearance ({side})"
        );
    }
}

#[test]
fn fret_crossing_rate_is_never_called_pitch() {
    let sp = fret(json!({}))["meta"]["spectrum"].clone();
    assert_eq!(sp["kind"], "fret");
    assert!(sp.get("pitch").is_none() && sp.get("cents").is_none());
    assert_eq!(sp["crossing_is_pitch"], false);
    let f1 = f(&sp["f1"]);
    assert!((f(&sp["crossing_rate_control"]) - f1).abs() <= 0.01 * f1);
    assert!(f(&sp["crossing_rate"]) > f(&sp["crossing_rate_control"]));
    assert!(f(&sp["static_oracle"]["residual"]) < 1e-13);
}

#[test]
fn fret_control_window_is_independent_of_the_fret_window() {
    let long_run = fret(json!({"audio_duration": 0.4}))["meta"]["spectrum"].clone();
    assert!(close(f(&long_run["centroid_control"]), 100.0, 0.1));
    assert!(close(f(&long_run["elevation"]), 4.682, 0.02));
    let short = fret(json!({"audio_duration": 0.15}))["meta"]["spectrum"].clone();
    assert!(close(f(&short["centroid_control"]), 100.0, 0.2));
    assert!(
        f(&short["centroid_fret"]) < f(&long_run["centroid_fret"]),
        "the fret window is NOT invariant"
    );
}

#[test]
fn fret_ignores_the_names_that_belong_to_other_models() {
    let base = fret(json!({"audio_duration": 0.08}));
    let other = fret(
        json!({"audio_duration": 0.08, "alpha": 2.3, "K": 8000, "depth": 1.0e-3,
                            "bridge_stiffness": 1.0, "width_frac": 0.4, "kappa": 4.0,
                            "sigma1": 0.05}),
    );
    assert_eq!(other["fs_sim"], base["fs_sim"]);
    assert_eq!(
        other["meta"]["contact"]["duty"],
        base["meta"]["contact"]["duty"]
    );
    assert_eq!(
        other["meta"]["contact"]["active_max"],
        base["meta"]["contact"]["active_max"]
    );
    assert_eq!(
        other["meta"]["spectrum"]["elevation"],
        base["meta"]["spectrum"]["elevation"]
    );
}

#[test]
fn fret_work_budget_counts_both_runs_and_the_guards_are_reachable() {
    for o in [
        json!({"N": FRET_N_MAX + 1}),
        json!({"audio_duration": FRET_AUDIO_MAX + 0.05}),
        json!({"clearance": 0.0}),
        json!({"rail_stiffness": 0.0}),
    ] {
        assert!(is_err(&fret(o.clone())), "{o}");
    }
    let at_cap = fret(json!({"audio_duration": FRET_AUDIO_MAX}));
    assert!(!is_err(&at_cap));
    let m = &at_cap["meta"];
    assert!(
        m["num_steps"].as_i64().unwrap() + m["n_control_steps"].as_i64().unwrap() <= FRET_WORK_MAX
    );
    let err = fret(json!({"audio_duration": FRET_AUDIO_MAX, "lambda": 0.2}));
    assert!(err["error"]["message"].as_str().unwrap().contains("steps"));
}

#[test]
fn fret_control_is_bounded_and_the_animation_is_a_stride_of_one_run() {
    let d = fret(json!({"audio_duration": 0.4}));
    let fs = f(&d["fs_sim"]);
    let nc = d["meta"]["n_control_steps"].as_i64().unwrap();
    assert!(nc <= (FRET_CONTROL_MAX * fs).round_ties_even() as i64);
    assert!(nc < d["meta"]["num_steps"].as_i64().unwrap());
    assert_eq!(
        f(&d["frame_times"][0]),
        0.0,
        "frames start at t = 0 (no settling window)"
    );
    let n_frames = d["frames"]["n_frames"].as_u64().unwrap() as usize;
    let width = d["frames"]["width"].as_u64().unwrap() as usize;
    assert_eq!(n_frames, d["frame_times"].as_array().unwrap().len());
    assert_eq!(width, d["grid"]["x"].as_array().unwrap().len());
    let field = decode_f32(d["frames"]["b64"].as_str().unwrap());
    assert_eq!(field.len(), n_frames * width);
    assert!(field.iter().all(|v| v.is_finite()));
}

#[test]
fn fret_audio_is_a_near_termination_pickup_and_is_real() {
    let d = fret(json!({}));
    let audio = decode_f32(d["audio"]["b64"].as_str().unwrap());
    assert_eq!(f(&d["audio"]["fs"]), AUDIO_FS);
    assert_eq!(d["audio"]["n"], audio.len());
    assert!(audio.iter().all(|v| v.is_finite()));
    let peak = f(&d["audio"]["peak"]);
    assert!(0.0 < peak && peak <= 1.0);
    let mx = audio.iter().fold(0.0f32, |m, v| m.max(v.abs())) as f64;
    assert!(close(mx, 0.9, 0.05));
    assert!(close(f(&d["meta"]["probe_x"]), 0.05, 0.01));
}

#[test]
fn fret_episode_debounce_merges_chatter_without_inventing_it() {
    let ct = fret(json!({}))["meta"]["contact"].clone();
    let (ep, raw) = (
        ct["episodes"].as_i64().unwrap(),
        ct["raw_onsets"].as_i64().unwrap(),
    );
    assert!(ep <= raw);
    assert!(
        ep as f64 >= 0.9 * raw as f64,
        "the default must not be over-merged"
    );
    assert!(f(&ct["episodes_per_period"]) >= FRET_EPISODES_MIN);
}
