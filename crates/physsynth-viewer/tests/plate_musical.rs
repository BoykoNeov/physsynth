//! The plate's musical controls through the payload builder — `docs/dev/plate-mapping-plan.md` §7.
//!
//! The mapping is held by core's `tests/plate_tuning.rs`, which runs the simulation. What is held
//! here is the wiring, and every bar that can listens to the OUTPUT rather than comparing the
//! reported `tuning` block with core: a build that ignored the mapping would still report it
//! correctly (the string batch's finding).

mod common;

use common::{base_params, decode_f32, f, ok, sim};
use physsynth_analysis::spectrum::measure_partials_near;
use physsynth_core::exciter::raised_cosine_2d;
use physsynth_core::plate::{Boundary, Domain, Params, Plate, PlateSpec};
use physsynth_core::plate_tuning::{preset, tune_plate, GrainAlong, SetBy, Targets};
use serde_json::{json, Value};

/// A musical plate request, with the physics keys set to values the mapping must ignore.
fn musical(overrides: Value) -> Value {
    let mut p = base_params(json!({
        "model": "plate", "domain": "supported", "controls": "musical",
        // Decoys: a musical build that read any of these would be a different plate.
        "kappa": 5.0, "rho": 0.9, "mu": 0.7, "sigma": 4.0, "nu": 0.1,
        "Lx": 0.6, "Ly": 0.45, "N": 40, "theta": 0.28,
        "material": "steel", "set_by": "pitch", "f0": 220.0, "thickness_mm": 2.0,
        "t60_low": 2.0, "t60_high": 0.5, "f_ref": 2000.0, "sample_rate": 44100.0,
        "audio_duration": 1.5,
    }));
    common::merge(&mut p, overrides);
    p
}

fn geometry(boundary: Boundary, domain: Domain, lx: f64, ly: f64, n: i64) -> PlateSpec {
    PlateSpec {
        lx,
        ly,
        fs: 44100.0,
        n,
        theta: 0.28,
        boundary: Some(boundary),
        domain: Some(domain),
        ..PlateSpec::default()
    }
}

fn cents(f: f64, target: f64) -> f64 {
    1200.0 * (f / target).log2()
}

#[test]
fn the_musical_plate_builds_from_the_targets_not_the_sliders() {
    let payload = ok(&musical(json!({})));
    let t = &payload["tuning"];
    let want = tune_plate(
        &geometry(Boundary::Supported, Domain::Rectangle, 0.6, 0.45, 40),
        &preset("steel").unwrap().material,
        GrainAlong::X,
        &Targets {
            set_by: SetBy::Pitch(220.0),
            t60_low: 2.0,
            t60_high: 0.5,
            f_ref: 2000.0,
        },
    )
    .unwrap();
    assert!((f(&t["kappa"]) - want.material.kappa).abs() < 1e-6, "{t}");
    assert!((f(&t["rho_s"]) - want.material.rho_s).abs() < 1e-6);
    assert!((f(&t["sigma0"]) - want.sigma0).abs() < 1e-6);
    assert!((f(&t["sigma1"]) - want.sigma1).abs() < 1e-8);
    assert_eq!(t["exact"], true);
    assert_eq!(f(&payload["fs_sim"]), 44100.0);
    // mu is a read-out of the MAPPED stiffness here, not the slider's 0.7
    let h = 0.6 / 40.0;
    let mu = want.material.kappa / 44100.0 / (h * h);
    assert!((f(&payload["mu"]) - mu).abs() < 1e-5, "{}", payload["mu"]);
}

#[test]
fn the_struck_plate_sounds_the_pitch_asked_for() {
    // The rendered audio, through the analysis crate's detector — the output, not the report.
    for (domain, material, n, f0) in [
        ("supported", "steel", 40, 220.0),
        ("guitar", "sitka spruce", 30, 140.0),
    ] {
        let payload = ok(&musical(json!({
            "domain": domain, "material": material, "N": n, "f0": f0,
            "Lx": 0.4, "Ly": 0.5, "f_ref": 1500.0, "t60_high": 0.9,
        })));
        let audio: Vec<f64> = decode_f32(payload["audio"]["b64"].as_str().unwrap())
            .into_iter()
            .map(f64::from)
            .collect();
        let got = measure_partials_near(&audio, f(&payload["audio"]["fs"]), &[f0], Some(15.0));
        let c = cents(got[0], f0);
        // 1.5 s at 48 kHz after resampling; the free plate's mapping is first-order (≤ 0.05 c)
        assert!(
            c.abs() < 0.5,
            "{domain}: the strike sounds {c} cents from {f0} Hz"
        );
    }
}

#[test]
fn the_energy_is_the_areal_densitys() {
    // A volume density passed where the areal one belongs leaves every pitch right and every
    // energy wrong by a factor of the thickness. Read the run's own first energy against a plate
    // built here from the mapping, struck the way the payload strikes.
    let payload = ok(&musical(json!({"material": "aluminium"})));
    let tu = tune_plate(
        &geometry(Boundary::Supported, Domain::Rectangle, 0.6, 0.45, 40),
        &preset("aluminium").unwrap().material,
        GrainAlong::X,
        &Targets {
            set_by: SetBy::Pitch(220.0),
            t60_low: 2.0,
            t60_high: 0.5,
            f_ref: 2000.0,
        },
    )
    .unwrap();
    let mut spec = geometry(Boundary::Supported, Domain::Rectangle, 0.6, 0.45, 40);
    tu.apply(&mut spec);
    let mut pl = Plate::new(Params::new(&spec).unwrap());
    let (lx, ly) = (pl.p.lx, pl.p.ly);
    let field = raised_cosine_2d(
        &pl.p.x,
        &pl.p.y,
        (0.4 * lx, 0.55 * ly),
        0.3 * lx.min(ly),
        1e-3,
    )
    .unwrap();
    let flags = pl.p.mask.flags().to_vec();
    let u0: Vec<f64> = field
        .iter()
        .zip(&flags)
        .filter_map(|(&v, &alive)| alive.then_some(v))
        .collect();
    pl.set_state(&u0, &vec![0.0; u0.len()]);
    let e0 = f(&payload["energy"]["value"][0]);
    assert!(
        (e0 / pl.energy() - 1.0).abs() < 1e-12,
        "payload E0 {e0} against {}",
        pl.energy()
    );
    // And the run's LAST energy: it has decayed through both losses at the mapped stiffness, so a
    // payload that built from the targets but dropped sigma1 (or misplaced sigma0) differs here.
    let values = payload["energy"]["value"].as_array().unwrap();
    let times = payload["energy"]["time"].as_array().unwrap();
    let steps = (f(times.last().unwrap()) * 44100.0).round() as usize;
    for _ in 0..steps {
        pl.step(None);
    }
    let e_end = f(values.last().unwrap());
    assert!(
        e_end < 0.5 * e0,
        "the run lost too little to tell the losses apart"
    );
    assert!(
        (e_end / pl.energy() - 1.0).abs() < 1e-9,
        "payload E_end {e_end} against {} after {steps} steps",
        pl.energy()
    );
}

#[test]
fn musical_keys_do_not_touch_the_physical_plate() {
    // The front-end sends every slider to every model: a plate request without musical controls
    // must come back byte for byte whatever musical keys ride along.
    let plain = base_params(
        json!({"model": "plate", "kappa": 20.0, "sigma": 0.5, "N": 30,
                                   "audio_duration": 0.2}),
    );
    let mut noisy = plain.clone();
    common::merge(
        &mut noisy,
        json!({"controls": "physical", "material": "glass", "set_by": "thickness",
               "thickness_mm": 4.0, "f0": 330.0, "t60_low": 9.0, "t60_high": 0.1,
               "f_ref": 4000.0, "sample_rate": 96000.0, "grain": "y"}),
    );
    let (a, b) = (sim(&plain), sim(&noisy));
    assert!(a.get("error").is_none(), "{}", a["error"]);
    assert!(a.get("tuning").is_none());
    assert_eq!(a.to_string(), b.to_string());
}

#[test]
fn set_by_thickness_reads_the_pitch_out() {
    let payload = ok(&musical(
        json!({"set_by": "thickness", "thickness_mm": 3.0}),
    ));
    let t = &payload["tuning"];
    assert_eq!(t["set_by"], "thickness");
    assert!((f(&t["thickness_mm"]) - 3.0).abs() < 1e-9);
    assert!(t["f0_target"].is_null());
    // The continuum f11 = (π/2) κ (1/Lx² + 1/Ly²), κ = t sqrt(E / 12 ρ (1 − ν²)) for steel's
    // constants; the scheme's fundamental lies within a few percent of it on this grid.
    let (e, nu, rho): (f64, f64, f64) = (200.0e9, 0.303, 7820.0);
    let kappa = 0.003 * (e / (12.0 * rho * (1.0 - nu * nu))).sqrt();
    let ly = f(&payload["grid"]["extent_y"]);
    let f11 = 0.5 * std::f64::consts::PI * kappa * (1.0 / 0.36 + 1.0 / (ly * ly));
    let f0 = f(&t["f0"]);
    assert!(
        (f0 / f11 - 1.0).abs() < 0.03,
        "{f0} against the continuum {f11}"
    );
}

#[test]
fn a_free_plate_says_it_is_not_exact_and_wood_reports_its_grain() {
    let payload = ok(&musical(json!({
        "domain": "guitar", "material": "western redcedar", "N": 30, "Lx": 0.4, "Ly": 0.5,
        "f0": 150.0, "f_ref": 1200.0, "t60_high": 0.9,
    })));
    let t = &payload["tuning"];
    assert_eq!(t["exact"], false);
    assert_eq!(
        t["grain"], "y",
        "a guitar top's grain runs along its length by default"
    );
    assert!(t["message"].as_str().unwrap().contains("0.05 cents"), "{t}");
    assert_eq!(t["partials"].as_array().unwrap().len(), 8);
    let steel = ok(&musical(json!({})));
    assert!(
        steel["tuning"]["grain"].is_null(),
        "isotropic material has no grain"
    );
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
        (
            json!({"material": "unobtainium"}),
            "material must be one of",
        ),
        (json!({"grain": "z"}), "grain must be 'x' or 'y'"),
        (
            json!({"set_by": "weight"}),
            "set_by must be 'pitch' or 'thickness'",
        ),
        (json!({"t60_high": 8.0}), "out of reach"),
        (json!({"f_ref": 150.0}), "picks the fundamental"),
        (json!({"f0": 0.0}), "f0 must be positive"),
        (
            json!({"N": 80, "Lx": 1.0, "Ly": 1.0, "sample_rate": 8000.0, "f0": 300.0}),
            "too coarse in time",
        ),
    ];
    for (bad, needle) in cases {
        let payload = sim(&musical(bad.clone()));
        assert_eq!(payload["error"]["kind"], "param", "{bad}: {payload}");
        let msg = payload["error"]["message"].as_str().unwrap();
        assert!(msg.contains(needle), "{bad}: {msg}");
    }
}
