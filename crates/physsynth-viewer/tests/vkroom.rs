//! The gong in the room through the payload builder — `test_web_backend.py`'s vkroom section.
//!
//! The claim is a SEPARATION: a struck von Kármán plate's radiation pattern moves during the strike
//! and its linear twin's does not — one flag apart, the same plate, room and strike. The honesty
//! line is the per-step convergence record, because both energy ledgers read green even on rigs
//! whose claim has already collapsed.

mod common;

use common::{f, ok, sim};
use physsynth_viewer::py::has_nonfinite;
use serde_json::{json, Value};
use std::sync::OnceLock;

/// The reference's module-scoped fixture: one coupled run, shared.
fn fixture() -> &'static Value {
    static P: OnceLock<Value> = OnceLock::new();
    P.get_or_init(|| ok(&json!({"model": "vkroom", "audio_duration": 0.06})))
}

fn vr(overrides: Value) -> Value {
    let mut p = json!({"model": "vkroom", "audio_duration": 0.03});
    common::merge(&mut p, overrides);
    p
}

/// THE CLAIM: the drift separates by more than 20x, and the resolved-band spread by more than 3x.
#[test]
fn the_struck_plates_pattern_moves_and_the_linear_ones_does_not() {
    let c = &fixture()["meta"]["claim"];
    assert!(
        f(&c["modal_drift"]) > 20.0 * f(&c["modal_drift_twin"]),
        "{c}"
    );
    assert!(f(&c["spread_resolved"]) > f(&c["spread_resolved_twin"]));
    let sep = (f(&c["spread_resolved"]) - 1.0) / (f(&c["spread_resolved_twin"]) - 1.0).max(1e-12);
    assert!(sep > 3.0, "{sep}");
}

#[test]
fn the_honesty_line_is_convergence_because_neither_ledger_is() {
    let d = fixture();
    let cv = &d["meta"]["convergence"];
    assert_eq!(cv["all_converged"], true, "{cv}");
    assert_eq!(cv["n_not_converged"], 0);
    let mi = cv["max_iters"].as_i64().unwrap();
    assert!(1 <= mi && mi <= cv["cap"].as_i64().unwrap());
    assert_eq!(
        cv["twin_max_iters"], 1,
        "the linear twin needs no iteration"
    );
    assert_eq!(d["energy"]["convergence"]["all_converged"], true);
}

#[test]
fn both_ledgers_are_green_and_the_scene_is_conserved() {
    let d = fixture();
    let lg = &d["meta"]["ledger"];
    assert_eq!(lg["kind"], "vkroom");
    assert!(f(&lg["residual_max"]) < 1e-10);
    assert_eq!(d["energy"]["lossless"]["pass"], true);
    assert!(f(&d["energy"]["lossless"]["drift"]) < 1e-10);
}

/// The compact monopole is a caption beside the truth, and no far-field pressure is shipped.
#[test]
fn the_compact_monopole_is_shipped_beside_the_truth_not_as_it() {
    let d = fixture();
    let mono = f(&d["meta"]["claim"]["mono_ratio"]);
    assert!(0.0 < mono && mono < 1e-3, "{mono}");
    assert!(d["meta"].get("pressure").is_none());
    assert!(d["meta"].get("t50").is_none());
}

/// Two clocks: the plate pane on its first flexural mode, the slices on acoustic transit.
#[test]
fn it_ships_the_plate_field_and_the_room_slices_on_two_clocks() {
    let d = fixture();
    let (fr, rf) = (&d["frames"], &d["room_frames"]);
    assert_eq!(fr["dims"], 2);
    assert_eq!(rf["dims"], 3);
    assert_eq!(rf["kind"], "slices");
    assert!(fr["n_frames"].as_u64().unwrap() >= 2 && rf["n_frames"].as_u64().unwrap() >= 2);
    assert_ne!(rf["anim_dt"], d["anim_dt"]);
    let planes = rf["planes"].as_array().unwrap();
    assert_eq!(planes.len(), 3);
    let width: u64 = planes
        .iter()
        .map(|p| p["nu"].as_u64().unwrap() * p["nv"].as_u64().unwrap())
        .sum();
    assert_eq!(rf["width"], width);
    assert_eq!(
        d["frame_times"].as_array().unwrap().len() as u64,
        fr["n_frames"].as_u64().unwrap()
    );
    assert_eq!(
        rf["times"].as_array().unwrap().len() as u64,
        rf["n_frames"].as_u64().unwrap()
    );
}

#[test]
fn it_reports_the_snapped_room_and_the_derived_rate() {
    let d = fixture();
    let room = &d["meta"]["room"];
    let (l0, n0, h) = (f(&room["L"][0]), f(&room["N"][0]), f(&room["h"]));
    assert!((l0 - n0 * h).abs() <= 1e-9 * l0);
    assert_eq!(f(&room["cfl"]), 0.9);
    let want = 343.0 * 3.0f64.sqrt() / (0.9 * h);
    assert!((f(&d["fs_sim"]) - want).abs() <= 1e-6 * want);
    assert_eq!(d["boundary"], "free");
    assert!(!has_nonfinite(d));
}

/// With the coupling already off there is no second rig: the run is its own control.
#[test]
fn the_flag_off_makes_the_run_its_own_control() {
    let d = ok(&vr(json!({"nonlinear": false})));
    let c = &d["meta"]["claim"];
    assert_eq!(d["nonlinear"], false);
    assert_eq!(c["spread_resolved"], c["spread_resolved_twin"]);
    assert_eq!(c["modal_drift"], c["modal_drift_twin"]);
    assert_eq!(d["meta"]["convergence"]["max_iters"], 1);
}

/// The coarsest legal plate still carries the claim.
#[test]
fn the_claim_survives_the_coarsest_legal_plate() {
    let d = ok(&vr(json!({"plate_N": 8})));
    let c = &d["meta"]["claim"];
    assert_eq!(d["meta"]["plate"]["N"], 8);
    assert_eq!(c["n_modes"], 81);
    assert_eq!(c["n_resolved"], 19);
    assert!(f(&c["modal_drift"]) > 20.0 * f(&c["modal_drift_twin"]));
    let sep = (f(&c["spread_resolved"]) - 1.0) / (f(&c["spread_resolved_twin"]) - 1.0).max(1e-12);
    assert!(sep > 3.0, "{sep}");
    assert_eq!(d["meta"]["convergence"]["all_converged"], true);
}

#[test]
fn both_tiers_run() {
    for tier in ["baffled", "suspended"] {
        let d = ok(&vr(json!({"domain": tier})));
        assert_eq!(d["tier"], tier);
        assert!(d["meta"]["claim"]["n_resolved"].as_u64().unwrap() >= 1);
    }
}

/// Every guard is a measured limit, and its message says which kind.
#[test]
fn guards_are_clean_error_payloads() {
    for (over, needle) in [
        (json!({"w_over_e": 3.5}), "cliff"),
        (json!({"air_h": 0.02}), "converging"),
        (json!({"air_h": 0.005}), "air_h"),
        (json!({"domain": "supported"}), "domain"),
        (json!({"room_size": 1.5}), "room_size"),
        (json!({"plate_N": 40}), "plate_N"),
        (json!({"audio_duration": 5.0}), "audio_duration"),
        (json!({"wall_zeta": 0.0}), "wall_zeta"),
    ] {
        let d = sim(&vr(over.clone()));
        assert_eq!(d["error"]["kind"], "param", "{over}");
        assert!(
            d["error"]["message"].as_str().unwrap().contains(needle),
            "{over}"
        );
    }
}

/// The plate term is priced at the sweep cap, so a render legal on the mean cannot slip through.
#[test]
fn the_two_term_budget_refuses_what_neither_shipped_cap_would() {
    let d = sim(&json!({"model": "vkroom", "audio_duration": 0.3}));
    assert_eq!(d["error"]["kind"], "param");
    assert!(d["error"]["message"].as_str().unwrap().contains("budget"));
}

#[test]
fn it_ignores_params_that_belong_to_other_models() {
    let d = ok(&vr(
        json!({"kappa": 8.0, "T": 500.0, "bridge_stiffness": 1e6, "air_cfl": 0.45,
                          "radius": 0.9}),
    ));
    assert_eq!(
        f(&d["meta"]["room"]["cfl"]),
        0.9,
        "air_cfl is a CONSTANT here"
    );
}
