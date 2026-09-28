//! The tension-modulated string through the payload builder — both regimes. Carried from
//! `test_web_backend.py`'s tension and parametric sections.
//!
//! The Duffing regime's headline is the amplitude SHIFT against the exact closed form, gated by
//! purity and by the root-find's convergence. The parametric regime's is that the energy drift
//! stays at machine precision THROUGH a complete disintegration of the driven mode.

mod common;

use common::{decode_f32, f, ok, sim};
use physsynth_analysis::{damping, duffing};
use physsynth_viewer::energy::LOSSLESS_TOL;
use physsynth_viewer::string::build_resonator;
use physsynth_viewer::tension::{
    self as t, amplitude_for, dt_over_t, measure_mode1, mode_frequency, run_parametric,
    spectrum_block, PARAM_SEED_REL, PARAM_SWEEP_CAP_PERIODS, PARAM_SWEEP_CHUNK_PERIODS,
    PARAM_SWEEP_DTS, PARAM_SWEEP_N, TENSION_AMP_MAX, TENSION_DT_MAX, TENSION_N_MAX,
    TENSION_OFFMODE_MAX,
};
use serde_json::{json, Value};

fn tension_p(overrides: Value) -> Value {
    let mut p = json!({"model": "tension", "N": 128, "audio_duration": 0.4, "amplitude": 0.02});
    common::merge(&mut p, overrides);
    p
}

fn param_p(overrides: Value) -> Value {
    let mut p = json!({
        "model": "tension", "domain": "parametric", "N": 128, "EA": 1.0e5, "kappa": 2.0,
        "sweep_points": [1.0, 3.0], "sweep_cap": 25,
    });
    common::merge(&mut p, overrides);
    p
}

// == the Duffing regime ============================================================================

#[test]
fn shift_matches_the_exact_duffing_oracle() {
    let sp = ok(&tension_p(json!({})))["meta"]["spectrum"].clone();
    assert_eq!(sp["kind"], "tension");
    assert!(f(&sp["shift_oracle"]) > 1.0, "a real, audible number of Hz");
    let (meas, orac) = (f(&sp["shift_measured"]), f(&sp["shift_oracle"]));
    assert!((meas - orac).abs() <= 1e-2 * orac, "{meas} vs {orac}");
    assert!(f(&sp["shift_rel_error"]) < 1e-2);
    assert!(f(&sp["f_hardened"]) > f(&sp["f_linear"]), "hardening only");
}

#[test]
fn shift_is_immune_to_loss() {
    // The measurement runs force sigma0 = sigma1 = 0, so the number is bit-identical however lossy
    // the audio run is.
    let quiet = ok(&tension_p(json!({})))["meta"]["spectrum"].clone();
    let lossy = ok(&tension_p(json!({"sigma0": 5.0, "sigma1": 0.002})))["meta"]["spectrum"].clone();
    assert_eq!(
        ok(&tension_p(json!({"sigma0": 5.0})))["energy"]["sigma_is_zero"],
        false
    );
    assert_eq!(lossy["shift_measured"], quiet["shift_measured"]);
    assert_eq!(lossy["f_hardened"], quiet["f_hardened"]);
}

#[test]
fn energy_survives_the_wrapper_and_the_nonlinearity_is_engaged() {
    let r = ok(&tension_p(json!({})));
    assert!(f(&r["energy"]["lossless"]["drift"]) < LOSSLESS_TOL);
    assert_eq!(r["energy"]["lossless"]["pass"], true);
    assert_eq!(r["energy"]["convergence"]["all_converged"], true);
    assert!(
        f(&r["meta"]["nonlinear_fraction"]) > 0.01,
        "not hiding at small amplitude"
    );
    assert!((f(&r["meta"]["EA_over_T"]) - 500.0).abs() < 1e-9);
}

#[test]
fn ea_zero_collapses_to_the_linear_string() {
    let r = ok(&tension_p(json!({"EA": 0.0})));
    let sp = &r["meta"]["spectrum"];
    assert_eq!(f(&r["meta"]["nonlinear_fraction"]), 0.0);
    assert_eq!(f(&sp["shift_oracle"]), 0.0);
    assert!(f(&sp["shift_measured"]).abs() <= 1e-6);
    assert_eq!(f(&sp["dT_over_T"]), 0.0);
}

#[test]
fn dt_over_t_matches_the_closed_form() {
    let sp = ok(&tension_p(json!({})))["meta"]["spectrum"].clone();
    let p2 = damping::spatial_eigenvalue_p2(128, 1.0 / 128.0, 1);
    // abs 1e-4: the payload rounds to 4 places, and reports the run's MEASURED peak
    assert!((f(&sp["dT_over_T"]) - dt_over_t(1e5, 0.02, p2, 200.0)).abs() <= 1e-4);
}

#[test]
fn purity_gate_nulls_the_shift_when_the_mode_breaks_up() {
    // Driven directly, past the dT/T0 guard that makes this unreachable through the public path —
    // with the measurement run at 20 periods, which the reference reached by patching a constant.
    let p = json!({"model": "tension", "N": 64});
    let broke = measure_mode1(&p, 0.20, 20.0).unwrap(); // dT/T0 ~ 49
    assert!(
        broke.off_mode > TENSION_OFFMODE_MAX,
        "expected genuine parametric breakup"
    );
    assert!(measure_mode1(&p, 0.005, 20.0).unwrap().off_mode < TENSION_OFFMODE_MAX);

    let fs = 200.0 * 64.0 / 1.0;
    let p2 = damping::spatial_eigenvalue_p2(64, 1.0 / 64.0, 1);
    let (w0sq, eps) = duffing::kc_mode_coefficients(200.0, 0.0, 1e5, 0.005, p2, 1.0).unwrap();
    let sig: Vec<f64> = (0..4096)
        .map(|i| (2.0 * std::f64::consts::PI * 100.0 * i as f64 / fs).sin())
        .collect();
    let sp = spectrum_block(&p, &sig, fs, w0sq, eps, 0.20, 20.0).unwrap();
    assert_eq!(sp["purity"]["pure"], false);
    assert!(sp["shift_measured"].is_null(), "no shift, not a wrong one");
    assert!(sp["shift_rel_error"].is_null());
    assert!(f(&sp["shift_oracle"]) > 0.0, "the oracle is still reported");
}

#[test]
fn dt_guard_is_not_an_amplitude_proxy() {
    let at_cap = tension_p(json!({"amplitude": TENSION_AMP_MAX, "EA": 1e5, "audio_duration": 0.2}));
    assert!(f(&ok(&at_cap)["meta"]["spectrum"]["dT_over_T"]) <= TENSION_DT_MAX);
    let mut stiffer = at_cap.clone();
    stiffer["EA"] = json!(2e5);
    let err = sim(&stiffer)["error"].clone();
    assert_eq!(err["kind"], "param");
    assert!(err["message"].as_str().unwrap().contains("dT/T0"));
}

#[test]
fn lossy_reports_passivity() {
    let r = ok(&tension_p(json!({"sigma0": 4.0})));
    assert_eq!(r["energy"]["sigma_is_zero"], false);
    assert_eq!(r["energy"]["lossy"]["monotone"], true);
    let m = f(&r["energy"]["lossy"]["measured_2sigma"]);
    assert!((m - 8.0).abs() <= 0.25 * 8.0, "{m}");
}

#[test]
fn frames_decode_to_a_mode1_sine_with_fixed_ends() {
    let r = ok(&tension_p(json!({})));
    let fr = &r["frames"];
    let w = fr["width"].as_u64().unwrap() as usize;
    let field = decode_f32(fr["b64"].as_str().unwrap());
    assert_eq!(fr["dims"], 1);
    assert_eq!(w, 129);
    assert!(f64::from(field[0]).abs() <= 1e-12);
    assert!(f64::from(field[w - 1]).abs() <= 1e-12);
    // frame 0 IS the initial condition: a half sine peaking at mid-span
    assert!((f64::from(field[64]) - 0.02).abs() <= 0.02 * 1e-4);
    let peak = field.iter().fold(0.0f64, |m, &v| m.max(f64::from(v).abs()));
    assert!((peak - f(&r["field_amp"])).abs() <= 1e-5 * peak);
}

#[test]
fn bad_params_give_error_payload() {
    for bad in [
        json!({"amplitude": 0.0}),
        json!({"amplitude": TENSION_AMP_MAX + 0.01}),
        json!({"EA": -1.0}),
        json!({"EA": 5e5}),
        json!({"N": TENSION_N_MAX + 1}),
        json!({"audio_duration": 6.0}),
        json!({"N": 256, "audio_duration": 3.0}),
        json!({"pickup_position": 1.5}),
    ] {
        let r = sim(&tension_p(bad.clone()));
        let kind = r["error"]["kind"].as_str().unwrap_or("");
        assert!(kind == "param" || kind == "construction", "{bad} gave {r}");
        assert!(!r["error"]["message"].as_str().unwrap().is_empty());
    }
}

// == the parametric regime =========================================================================

#[test]
fn parametric_conserves_through_a_complete_disintegration() {
    let d = ok(&param_p(json!({})));
    let sp = &d["meta"]["spectrum"];
    assert_eq!(sp["kind"], "parametric");
    assert_eq!(d["regime"], "parametric");
    assert_eq!(sp["above"]["unstable"], true);
    assert!(f(&sp["above"]["level"]) > 0.1, "a real disintegration");
    assert_eq!(d["energy"]["lossless"]["pass"], true);
    assert!(f(&d["energy"]["lossless"]["drift"]) < 1e-10);
}

#[test]
fn parametric_pair_straddles_and_the_stable_run_stays_on_its_seed() {
    let sp = ok(&param_p(json!({})))["meta"]["spectrum"].clone();
    assert_eq!(sp["straddles"], true);
    assert_eq!(sp["below"]["unstable"], false);
    assert!(f(&sp["below"]["growth"]) < 10.0);
    assert!(f(&sp["above"]["growth"]) > 1e4);
}

#[test]
fn parametric_below_run_is_not_a_linear_control() {
    let sp = ok(&param_p(json!({})))["meta"]["spectrum"].clone();
    assert!(f(&sp["below"]["nl_fraction"]) > 0.3);
    assert!(f(&sp["nl_gap"]) < 0.25);
}

#[test]
fn parametric_drift_is_scored_on_the_full_array_not_the_shipped_trace() {
    let d = ok(&param_p(json!({})));
    let trace: Vec<f64> = d["meta"]["spectrum"]["drift_trace"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_f64)
        .collect();
    assert!(!trace.is_empty());
    let top = trace.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(f(&d["energy"]["lossless"]["drift"]) >= top * (1.0 - 1e-12));
}

#[test]
fn parametric_log_axis_traces_are_not_rounded_into_zero() {
    let sp = ok(&param_p(json!({})))["meta"]["spectrum"].clone();
    let positive: Vec<f64> = sp["drift_trace"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_f64)
        .filter(|&v| v > 0.0)
        .collect();
    assert!(!positive.is_empty());
    assert!(positive.iter().copied().fold(f64::INFINITY, f64::min) < 1e-12);
    let env_min = sp["below"]["env"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(Value::as_f64)
        .filter(|&v| v > 0.0)
        .fold(f64::INFINITY, f64::min);
    assert!(env_min < 1e-5);
}

#[test]
fn parametric_energy_lands_in_low_neighbours_not_at_the_grid_scale() {
    let c = ok(&param_p(json!({})))["meta"]["spectrum"]["cascade"].clone();
    let top = c["top"].as_array().unwrap();
    assert!(!top.is_empty());
    assert!(f(&c["over_grid"]) > 100.0);
    let winner = top[0][0].as_i64().unwrap();
    assert!(winner != c["driven"].as_i64().unwrap() && winner < c["grid_from"].as_i64().unwrap());
    // Every NAMED partner survives the readout's own one-decimal percentage.
    for pair in top {
        assert_ne!(format!("{:.1}", f(&pair[1]) * 100.0), "0.0", "{pair}");
    }
}

#[test]
fn parametric_names_no_partner_below_the_tongue() {
    let sp = ok(&param_p(json!({"dt_over_t0": 1.2, "dt_below": 1.0})))["meta"]["spectrum"].clone();
    assert_eq!(sp["above"]["unstable"], false);
    assert_eq!(sp["cascade"]["top"], json!([]));
    assert!(
        sp["cascade"]["over_grid"].is_null(),
        "no winner is null, never 0.0"
    );
}

#[test]
fn parametric_mode_one_is_the_robust_one() {
    let at_three = ok(&param_p(json!({"mode_number": 3})))["meta"]["spectrum"].clone();
    let at_one = ok(&param_p(json!({"mode_number": 1})))["meta"]["spectrum"].clone();
    assert_eq!(at_three["above"]["unstable"], true);
    assert_eq!(at_one["above"]["unstable"], false);
    assert_eq!(at_one["straddles"], false);
}

#[test]
fn parametric_run_length_is_periods_of_mode_m_not_steps() {
    let coarse = ok(&param_p(json!({"N": 64})))["meta"]["spectrum"].clone();
    let fine = ok(&param_p(json!({"N": 128})))["meta"]["spectrum"].clone();
    assert_eq!(coarse["periods"], fine["periods"]);
    let last = |sp: &Value| f(sp["above"]["t"].as_array().unwrap().last().unwrap());
    assert!((last(&fine) - last(&coarse)).abs() <= 0.05 * last(&coarse));
}

#[test]
fn parametric_self_seeding_still_breaks_up_without_the_explicit_seed() {
    let base = param_p(json!({"sigma0": 0.0, "sigma1": 0.0}));
    let b = build_resonator(&base).unwrap();
    let mut res = b.res.into_tension();
    let (f_m, p2) = mode_frequency(&res, 3);
    let amp = amplitude_for(&res, 3.0, p2);
    let n_steps = physsynth_viewer::py::round_int(40.0 * res.p.fs / f_m) as usize;
    let run = run_parametric(&mut res, 3, amp, n_steps, 0.0, None, 0, 0).unwrap();
    let top = run.off.iter().copied().fold(0.0f64, f64::max);
    assert!(top > 1e-3, "roundoff alone must still seed the instability");
    let e0 = run.energy[0];
    let drift = run
        .energy
        .iter()
        .map(|e| (e - e0).abs())
        .fold(0.0f64, f64::max)
        / e0.abs();
    assert!(drift < 1e-10, "and it must still conserve");
    assert!(amp > 0.0 && PARAM_SEED_REL > 0.0);
}

#[test]
fn parametric_sweep_bins_stable_unstable_and_unsaturated_separately() {
    let sw = ok(&param_p(
        json!({"sweep_points": [1.0, 1.5, 2.0, 3.0, 5.0], "sweep_cap": 40}),
    ))["meta"]["spectrum"]["sweep"]
        .clone();
    let pt = |dt: f64| {
        sw["points"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| f(&p["dt"]) == dt)
            .unwrap()
            .clone()
    };
    assert_eq!(pt(1.0)["unstable"], false);
    assert_eq!(pt(1.5)["unstable"], false);
    assert_eq!(pt(3.0)["unstable"], true);
    assert_eq!(pt(5.0)["unstable"], true);
    assert!(f(&pt(5.0)["level"]) > f(&pt(3.0)["level"]));
    assert!(f(&pt(1.0)["level"]) <= 100.0 * f(&pt(1.0)["floor"]));
    assert!(
        f(&sw["edge_lo"]) < f(&sw["edge_hi"]),
        "the edge is a BRACKET"
    );
}

#[test]
fn parametric_sweep_is_a_controlled_reference_curve() {
    let sw = ok(&param_p(json!({"N": 64, "sigma0": 7.0})))["meta"]["spectrum"]["sweep"].clone();
    assert_eq!(sw["N"], PARAM_SWEEP_N);
    assert_ne!(PARAM_SWEEP_N, 64);
    assert_eq!(f(&sw["seed_rel"]), PARAM_SEED_REL);
}

#[test]
fn parametric_sweep_truncation_is_labelled_never_silent() {
    // The work budget cannot trip on the shipped grid; driven past itself with a budget of 1 step,
    // which the reference reached by patching the constant.
    let p = param_p(json!({"sweep_points": [1.0, 3.0, 5.0]}));
    let d = t::build_payload_parametric_with(&p, 1).unwrap();
    let sw = &d["meta"]["spectrum"]["sweep"];
    assert_eq!(sw["truncated"], true);
    let pts = sw["points"].as_array().unwrap();
    let dropped: Vec<&Value> = pts.iter().filter(|p| p["truncated"] == true).collect();
    assert!(!dropped.is_empty());
    assert!(dropped.iter().all(|p| p["level"].is_null()));
    let dts: Vec<f64> = pts.iter().map(|p| f(&p["dt"])).collect();
    assert_eq!(dts, vec![1.0, 3.0, 5.0], "every point keeps its place");
}

#[test]
fn parametric_shipped_sweep_path_is_pinned_with_the_overrides_absent() {
    let sw = ok(&json!({"model": "tension", "domain": "parametric", "N": 64, "claim_periods": 8}))
        ["meta"]["spectrum"]["sweep"]
        .clone();
    let dts: Vec<f64> = sw["points"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| f(&p["dt"]))
        .collect();
    assert_eq!(dts, PARAM_SWEEP_DTS.to_vec());
    assert_eq!(f(&sw["cap_periods"]), PARAM_SWEEP_CAP_PERIODS);
    assert_eq!(sw["chunk_periods"], PARAM_SWEEP_CHUNK_PERIODS);
}

#[test]
fn parametric_forces_sigma_to_zero() {
    let d = ok(&param_p(json!({"sigma0": 5.0, "sigma1": 0.01})));
    assert_eq!(d["energy"]["sigma_is_zero"], true);
    assert_eq!(d["energy"]["lossless"]["pass"], true);
}

#[test]
fn parametric_regime_leaves_the_duffing_path_bit_for_bit() {
    assert_eq!(
        ok(&tension_p(json!({}))),
        ok(&tension_p(json!({"domain": "duffing"})))
    );
}

#[test]
fn parametric_bad_params_give_error_payload() {
    for bad in [
        json!({"domain": "nonesuch"}),
        json!({"mode_number": 0}),
        json!({"mode_number": 99}),
        json!({"dt_over_t0": 0}),
        json!({"dt_over_t0": 400}),
        json!({"dt_below": -1}),
        json!({"EA": 0}),
        json!({"N": 400}),
        json!({"N": 200, "claim_periods": 180}),
        json!({"claim_periods": 0}),
        json!({"pickup_position": 1.5}),
        json!({"sweep_points": [0.0]}),
        json!({"sweep_points": "nope"}),
    ] {
        let r = sim(&param_p(bad.clone()));
        let kind = r["error"]["kind"].as_str().unwrap_or("");
        assert!(kind == "param" || kind == "construction", "{bad} gave {r}");
        assert!(!r["error"]["message"].as_str().unwrap().is_empty());
    }
}

#[test]
fn an_unreadable_mode_number_gets_the_integer_message() {
    // `ParamError` is a `ValueError`, so the reference's `except (TypeError, ValueError)` around
    // `int(_fnum(...))` caught `_fnum`'s own refusal and re-worded it. Found by the one-time diff.
    let r = sim(&param_p(json!({"mode_number": "abc"})));
    assert_eq!(
        r["error"]["message"],
        "mode_number must be an integer, got 'abc'."
    );
}
