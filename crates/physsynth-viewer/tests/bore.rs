//! The acoustic bore with a radiating bell through the payload builder — `test_web_backend.py`'s
//! bore section.
//!
//! The claim is that the bell's loss is BOOKED: a lossless tube with a radiating end conserves its
//! total while the split shows acoustic energy falling and radiated energy rising.

mod common;

use common::{decode_f32, f, ok, sim};
use physsynth_viewer::bore::{
    BORE_ANIM_MAX, BORE_AUDIO_MAX, BORE_N_MAX, BORE_N_MIN, BORE_ODD_EVEN_GATE,
};
use physsynth_viewer::energy::LOSSLESS_TOL;
use physsynth_viewer::string::ANIM_WIN_MAX;
use physsynth_viewer::AUDIO_FS;
use serde_json::{json, Value};

fn bore(overrides: Value) -> Value {
    let mut p = json!({"model": "bore", "audio_duration": 0.25});
    common::merge(&mut p, overrides);
    p
}

fn nums(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(f).collect()
}

/// Total drift is necessary and NOT sufficient — it would pass with a bell that sheds nothing.
/// The split must drain and book, and sum flat.
#[test]
fn conserves_with_the_bell_radiating_and_the_split_actually_moves() {
    let d = ok(&bore(json!({"bell_ratio_exp": 0.0})));
    let e = &d["energy"];
    assert_eq!(e["sigma_is_zero"], true);
    assert!(f(&e["lossless"]["drift"]) < LOSSLESS_TOL);
    assert_eq!(e["lossless"]["pass"], true);
    let (ac, rad) = (nums(&e["split"]["acoustic"]), nums(&e["split"]["radiated"]));
    assert!(
        ac[ac.len() - 1] < 0.05 * ac[0],
        "the matched bell must drain the tube"
    );
    assert!(
        rad[rad.len() - 1] > 0.95 * ac[0],
        "and the drained energy must be BOOKED"
    );
    assert!(
        rad.windows(2).all(|w| w[1] - w[0] >= -1e-18),
        "radiated energy is cumulative"
    );
    let t0 = ac[0] + rad[0];
    let worst = ac
        .iter()
        .zip(&rad)
        .map(|(a, r)| (a + r - t0).abs())
        .fold(0.0f64, f64::max);
    assert!(worst / t0 < LOSSLESS_TOL);
    assert!((f(&d["meta"]["radiated_frac"]) - 1.0).abs() <= 1e-6);
}

#[test]
fn a_lightly_radiating_clarinet_sheds_a_little_and_still_conserves() {
    let d = ok(&bore(json!({})));
    assert_eq!(d["energy"]["lossless"]["pass"], true);
    let frac = f(&d["meta"]["radiated_frac"]);
    assert!(0.0 < frac && frac < 0.5);
    assert_eq!(d["meta"]["radiating"], true);
    assert_eq!(d["meta"]["ends"], json!(["closed", "radiating"]));
}

/// One bounce sheds `1/2 (1 - r²)`, and at the match exactly half — the curve must contain the
/// null EXACTLY rather than straddle it.
#[test]
fn the_reflection_matches_the_closed_form_including_the_anechoic_null() {
    let rb = ok(&bore(json!({})))["meta"]["reflection"].clone();
    assert_eq!(rb["radiating"], true);
    assert_eq!(rb["pass"], true);
    assert!(f(&rb["abs_error"]) < 1e-12);
    assert!((f(&rb["measured"]) - f(&rb["oracle"])).abs() <= 1e-12);
    let an = ok(&bore(json!({"bell_ratio_exp": 0.0})))["meta"]["reflection"].clone();
    assert_eq!(an["anechoic"], true);
    assert!(f(&an["r"]).abs() <= 1e-12);
    assert!((f(&an["measured"]) - 0.5).abs() <= 1e-12);
    let shed = nums(&an["curve"]["shed"]);
    let top = shed.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!((top - 0.5).abs() <= 1e-15, "{top}");
}

#[test]
fn r_over_z0_is_the_control_and_it_monotonically_buys_loss() {
    let fracs: Vec<f64> = [-3.5, -2.0, -1.0, 0.0]
        .iter()
        .map(|&x| f(&ok(&bore(json!({"bell_ratio_exp": x})))["meta"]["radiated_frac"]))
        .collect();
    assert!(fracs.windows(2).all(|w| w[0] <= w[1]), "{fracs:?}");
    assert!(fracs[0] < 0.5 && 0.5 < fracs[3]);
}

/// The open end is a perfect mirror: nothing radiates, the panel says so, and `R/Z0` does not
/// echo a live-looking value at a control doing nothing.
#[test]
fn the_ideal_open_end_is_the_lossless_contrast_with_no_bell_to_score() {
    let d = ok(&bore(json!({"domain": "open"})));
    assert_eq!(d["energy"]["lossless"]["pass"], true);
    assert!(f(&d["meta"]["radiated_frac"]).abs() <= 1e-15);
    assert_eq!(d["meta"]["radiating"], false);
    assert!(d["meta"]["r_ratio"].is_null());
    assert_eq!(d["meta"]["ends"], json!(["closed", "open"]));
    assert_eq!(d["meta"]["reflection"]["radiating"], false);
    assert!(d["meta"]["reflection"].get("note").is_some());
}

/// The odd/even ratio is a property of the FFT window, so the gate is checked at the SHORTEST
/// allowed render.
#[test]
fn it_is_a_clarinet_odd_harmonics_only_at_the_shortest_allowed_render() {
    let sp = ok(&bore(json!({"audio_duration": 0.25})))["meta"]["spectrum"].clone();
    assert_eq!(sp["applies"], true);
    assert_eq!(sp["odd_even"]["pass"], true);
    assert!(f(&sp["odd_even"]["ratio"]) > BORE_ODD_EVEN_GATE);
    assert!((f(&sp["f1"]) - 343.0 / 2.0).abs() <= 1e-3 * 171.5);
}

/// The eigenvalue oracle — from the new native solver — sits on the continuum at lambda = 1, and
/// the parabolic-refined partials sit on it.
#[test]
fn the_partials_land_on_the_eigenvalue_oracle_which_is_exact_at_lambda_one() {
    for n in [64, 128, 200] {
        let pa = ok(&bore(json!({"N": n})))["meta"]["spectrum"]["partials"].clone();
        let evc = nums(&pa["eigen_vs_continuum"]);
        assert!(evc.iter().all(|v| v.abs() < 1e-3), "N {n}: {evc:?}");
        let cve = nums(&pa["cents_vs_eigen"]);
        assert!(cve.iter().all(|v| v.abs() < 0.05), "N {n}: {cve:?}");
    }
}

/// At the anechoic match there is no standing wave: labelled, never failed, never a silent pass.
#[test]
fn a_heavily_absorbing_bell_is_labelled_not_failed() {
    let d = ok(&bore(json!({"bell_ratio_exp": 0.0})));
    assert_eq!(d["meta"]["spectrum"]["applies"], false);
    assert_eq!(d["meta"]["spectrum"]["odd_even"]["pass"], false);
}

/// The lambda claim is computed from the operator: the departure falls 4x across a 2x refinement
/// and collapses at lambda = 1.
#[test]
fn the_dispersion_is_an_eigenvalue_computation_showing_second_order_departure() {
    let dp = ok(&bore(json!({})))["meta"]["dispersion"].clone();
    let n = dp["lambda"].as_array().unwrap().len();
    for key in ["coarse", "fine", "order"] {
        assert_eq!(dp[key].as_array().unwrap().len(), n, "{key}");
    }
    assert_eq!(f(&dp["lambda"][n - 1]), 1.0);
    assert!(f(&dp["coarse"][n - 1]) < 1e-3 && f(&dp["fine"][n - 1]) < 1e-3);
    assert!(f(&dp["coarse"][0]) > 1.0);
    assert!((f(&dp["order"][0]) - 4.0).abs() <= 0.1);
}

/// THE TRAP: pace on the transit (L/c0), not on f1 = c0/4L, or the bounce aliases.
#[test]
fn the_animation_is_paced_on_the_transit_not_the_fundamental() {
    for n in [64, 128, 256] {
        let m = ok(&bore(json!({"N": n})))["meta"].clone();
        let fpt = f(&m["frames_per_transit"]);
        assert!((10.0..=14.0).contains(&fpt), "N {n}: {fpt}");
        assert!((f(&m["transit"]) - 0.5 / 343.0).abs() <= 1e-4 * 0.5 / 343.0);
    }
}

#[test]
fn frame_and_grid_bookkeeping_line_up() {
    let d = ok(&bore(json!({"N": 64})));
    let x = d["grid"]["x"].as_array().unwrap().len();
    assert_eq!(d["frames"]["dims"], 1);
    assert_eq!(d["frames"]["width"], 65);
    assert_eq!(x, 65);
    assert_eq!(d["grid"]["envelope"].as_array().unwrap().len(), x);
    let nf = d["frames"]["n_frames"].as_u64().unwrap() as usize;
    assert_eq!(nf, d["frame_times"].as_array().unwrap().len());
    assert_eq!(nf, d["radiated_frames"].as_array().unwrap().len());
    let frames = decode_f32(d["frames"]["b64"].as_str().unwrap());
    assert_eq!(frames.len(), nf * 65);
    assert!(frames.iter().all(|v| v.is_finite()));
    // the closed end is a pressure antinode, the open end a node — the odd-harmonic asymmetry
    assert!(f(&d["grid"]["envelope"][0]) > 0.0);
    let open = ok(&bore(json!({"N": 64, "domain": "open"})));
    assert_eq!(f(&open["grid"]["envelope"][64]), 0.0);
}

#[test]
fn radiated_frames_are_cumulative_and_normalized_for_the_mouth_glow() {
    let rf = nums(&ok(&bore(json!({"bell_ratio_exp": 0.0})))["radiated_frames"]);
    assert!(rf[0].abs() <= 1e-12);
    assert!(rf.windows(2).all(|w| w[1] - w[0] >= -1e-15));
    let last = rf[rf.len() - 1];
    assert!(0.0 < last && last <= 1.0);
}

#[test]
fn the_audio_is_real_finite_and_normalized() {
    let a = ok(&bore(json!({})))["audio"].clone();
    assert_eq!(f(&a["fs"]), AUDIO_FS);
    assert!(f(&a["peak"]) > 0.0);
    let sig = decode_f32(a["b64"].as_str().unwrap());
    assert_eq!(a["n"], sig.len());
    assert!(sig.iter().all(|v| v.is_finite() && v.abs() <= 1.0 + 1e-6));
}

#[test]
fn the_work_budget_counts_the_reflection_run_and_the_guards_are_reachable() {
    for bad in [
        json!({"N": BORE_N_MAX, "audio_duration": BORE_AUDIO_MAX + 0.5}),
        json!({"N": BORE_N_MAX + 1}),
        json!({"N": BORE_N_MIN - 1}),
        json!({"audio_duration": BORE_AUDIO_MAX + 0.1}),
        json!({"domain": "reed"}),
        json!({"bell_ratio_exp": 3.0}),
        json!({"pickup_position": 1.0}),
    ] {
        assert!(sim(&bore(bad.clone())).get("error").is_some(), "{bad}");
    }
}

/// A frame-count ceiling is not a cost ceiling: the bore has its own animation cap.
#[test]
fn the_animation_window_has_its_own_cap_not_the_shared_one() {
    const { assert!(BORE_ANIM_MAX < ANIM_WIN_MAX) };
    assert!(
        sim(&bore(json!({"animation_window": BORE_ANIM_MAX + 0.01})))
            .get("error")
            .is_some()
    );
    ok(&bore(json!({"animation_window": BORE_ANIM_MAX})));
}

#[test]
fn it_ignores_params_that_belong_to_other_models() {
    let base = ok(&bore(json!({})));
    let other = ok(&bore(
        json!({"sigma0": 0.9, "sigma1": 0.05, "kappa": 4.0, "depth": 0.002,
                                "T": 500}),
    ));
    assert_eq!(other["energy"]["sigma_is_zero"], true);
    assert_eq!(other["fs_sim"], base["fs_sim"]);
    assert_eq!(
        other["meta"]["radiated_frac"],
        base["meta"]["radiated_frac"]
    );
}

/// The refusal names the ratio the way the reference's `.3e` did: a signed two-digit exponent.
#[test]
fn the_ratio_refusal_formats_its_number_like_the_reference() {
    let d = sim(&bore(json!({"bell_ratio_exp": 3.0})));
    let msg = d["error"]["message"].as_str().unwrap();
    assert!(
        msg.contains("got 1.000e+03 (bell_ratio_exp = 3.0)"),
        "{msg}"
    );
}
