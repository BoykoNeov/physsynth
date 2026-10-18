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
use physsynth_core::plate_tuning::{preset, tune_plate, GrainAlong, Material, SetBy, Targets};
use serde_json::{json, Value};

// Every musical key OFF the server's fallback (`build_musical`'s defaults are f0 220, t60 2 / 0.5,
// f_ref 2000, sample_rate 44100, theta 0.28, steel, 2 mm): a misspelled key or a hard-wired theta
// reads the fallback, and a request that sent the fallback could not tell (CLAUDE.md's first rule
// for a bar).
const FS: f64 = 48000.0;
const THETA: f64 = 0.5;
const F0: f64 = 196.0;
const T60_LOW: f64 = 2.7;
const T60_HIGH: f64 = 0.65;
const F_REF: f64 = 1700.0;

/// A musical plate request, with the physics keys set to values the mapping must ignore.
fn musical(overrides: Value) -> Value {
    let mut p = base_params(json!({
        "model": "plate", "domain": "supported", "controls": "musical",
        // Decoys: a musical build that read any of these would be a different plate.
        "kappa": 5.0, "rho": 0.9, "mu": 0.7, "sigma": 4.0, "nu": 0.1,
        "Lx": 0.6, "Ly": 0.45, "N": 40, "theta": THETA,
        "material": "brass", "set_by": "pitch", "f0": F0, "thickness_mm": 2.5,
        "t60_low": T60_LOW, "t60_high": T60_HIGH, "f_ref": F_REF, "sample_rate": FS,
        "audio_duration": 1.5,
    }));
    common::merge(&mut p, overrides);
    p
}

fn geometry(boundary: Boundary, domain: Domain, lx: f64, ly: f64, n: i64) -> PlateSpec {
    PlateSpec {
        lx,
        ly,
        fs: FS,
        n,
        theta: THETA,
        boundary: Some(boundary),
        domain: Some(domain),
        ..PlateSpec::default()
    }
}

/// The targets `musical()` sends, set by pitch.
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

#[test]
fn the_musical_plate_builds_from_the_targets_not_the_sliders() {
    let payload = ok(&musical(json!({})));
    let t = &payload["tuning"];
    let want = tune_plate(
        &geometry(Boundary::Supported, Domain::Rectangle, 0.6, 0.45, 40),
        &preset("brass").unwrap().material,
        GrainAlong::X,
        &targets(),
    )
    .unwrap();
    assert!((f(&t["kappa"]) - want.material.kappa).abs() < 1e-6, "{t}");
    assert!((f(&t["rho_s"]) - want.material.rho_s).abs() < 1e-6);
    assert!((f(&t["sigma0"]) - want.sigma0).abs() < 1e-6);
    assert!((f(&t["sigma1"]) - want.sigma1).abs() < 1e-8);
    assert_eq!(t["exact"], true);
    assert_eq!(t["material"], "brass");
    assert_eq!(f(&payload["fs_sim"]), FS);
    // mu is a read-out of the MAPPED stiffness here, not the slider's 0.7
    let h = 0.6 / 40.0;
    let mu = want.material.kappa / FS / (h * h);
    assert!((f(&payload["mu"]) - mu).abs() < 1e-5, "{}", payload["mu"]);
}

#[test]
fn the_struck_plate_sounds_the_pitch_asked_for() {
    // The rendered audio, through the analysis crate's detector — the output, not the report.
    for (domain, material, n, f0) in [
        ("supported", "steel", 40, 233.0),
        ("guitar", "sitka spruce", 30, 140.0),
    ] {
        let payload = ok(&musical(json!({
            "domain": domain, "material": material, "N": n, "f0": f0,
            "Lx": 0.4, "Ly": 0.5, "f_ref": 1500.0, "t60_high": 1.3,
            "waist": 0.35, "asym": 0.2,
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
        &targets(),
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
    let steps = (f(times.last().unwrap()) * FS).round() as usize;
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
        json!({"set_by": "thickness", "thickness_mm": 3.0, "material": "steel"}),
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
            json!({"N": 80, "Lx": 1.0, "Ly": 1.0, "sample_rate": 8000.0, "f0": 300.0, "t60_high": 1.5}),
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

#[test]
fn the_material_menu_is_the_cores_preset_list() {
    // index.html lists the materials by hand; a preset added or renamed in core without the menu
    // following would be unreachable from the browser, or a menu entry the server refuses.
    let html = std::fs::read_to_string(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../web/static/index.html"),
    )
    .unwrap();
    let start = html
        .find("<select id=\"plate-material\">")
        .expect("the menu moved");
    let body = &html[start..start + html[start..].find("</select>").unwrap()];
    let menu: Vec<&str> = body
        .split("<option value=\"")
        .skip(1)
        .map(|s| &s[..s.find('"').unwrap()])
        .collect();
    let mut want: Vec<&str> = physsynth_core::plate_tuning::PRESETS
        .iter()
        .map(|p| p.name)
        .collect();
    want.push("custom");
    assert_eq!(menu, want);
}

#[test]
fn custom_constants_build_the_plate_they_describe() {
    // Off every default: E 150 GPa, nu 0.27, rho 6100 (the server falls back to 100, 0.3, 5000).
    let payload = ok(&musical(json!({
        "material": "custom", "custom_E_gpa": 150.0, "custom_nu": 0.27, "custom_density": 6100.0,
    })));
    let t = &payload["tuning"];
    let want = tune_plate(
        &geometry(Boundary::Supported, Domain::Rectangle, 0.6, 0.45, 40),
        &Material::Isotropic {
            e: 150.0e9,
            nu: 0.27,
            rho: 6100.0,
        },
        GrainAlong::X,
        &targets(),
    )
    .unwrap();
    assert_eq!(t["material"], "custom");
    assert!(
        (f(&t["thickness_mm"]) - want.thickness * 1e3).abs() < 1e-4,
        "{t}"
    );
    assert!((f(&t["rho_s"]) - want.material.rho_s).abs() < 1e-6, "{t}");
}

#[test]
fn an_explicit_grain_is_the_grain_the_plate_gets() {
    // The worked-out thickness depends on which way spruce's grain runs on a 0.6 x 0.45 plate, so
    // a server that ignored `grain` (and took the rectangle's default, x) would be caught.
    let thickness = |grain: &str| {
        let p = ok(&musical(
            json!({"material": "sitka spruce", "grain": grain}),
        ));
        assert_eq!(p["tuning"]["grain"], grain);
        f(&p["tuning"]["thickness_mm"])
    };
    let (along_x, along_y) = (thickness("x"), thickness("y"));
    let core = |along| {
        tune_plate(
            &geometry(Boundary::Supported, Domain::Rectangle, 0.6, 0.45, 40),
            &preset("sitka spruce").unwrap().material,
            along,
            &targets(),
        )
        .unwrap()
        .thickness
            * 1e3
    };
    assert!((along_y - core(GrainAlong::Y)).abs() < 1e-4, "{along_y}");
    assert!((along_x - core(GrainAlong::X)).abs() < 1e-4, "{along_x}");
    assert!(
        (along_x / along_y - 1.0).abs() > 0.05,
        "the grain made no difference"
    );
}

#[test]
fn a_grained_supported_plates_spectrum_markers_are_the_orthotropic_ones() {
    // Before the musical controls the viewer never built a grained plate, and two read-outs
    // assumed it never would: the discrete modes squared the Laplacian's eigenvalues (B = L·L), and
    // the continuum markers used the isotropic Navier law.
    let payload = ok(&musical(json!({"material": "sugar maple", "grain": "y"})));
    let t = &payload["tuning"];
    let s = &payload["meta"]["spectrum"];
    // The lowest discrete mode IS the mapped fundamental (lossless; the loss's pull is ~1e-8).
    let f1 = f(&s["modes_discrete"][0]);
    assert!(
        (f1 / F0 - 1.0).abs() < 1e-5,
        "discrete f1 {f1} against the target {F0}"
    );
    // The continuum markers: the orthotropic Navier law at the mapped kappa and the grain.
    let m = physsynth_core::plate_tuning::plate_material(
        &preset("sugar maple").unwrap().material,
        GrainAlong::Y,
        f(&t["thickness_mm"]) * 1e-3,
    )
    .unwrap();
    let ly = f(&payload["grid"]["extent_y"]);
    let modes: Vec<(i64, i64)> = (1..=6).flat_map(|a| (1..=6).map(move |b| (a, b))).collect();
    let mut want = physsynth_analysis::modal::orthotropic_plate_freqs(
        m.kappa,
        0.6,
        ly,
        &modes,
        m.grain_x,
        m.grain_cross.unwrap(),
        m.grain_y,
    )
    .unwrap();
    want.sort_by(f64::total_cmp);
    let got = s["modes_continuum"].as_array().unwrap();
    assert!(!got.is_empty());
    for (g, w) in got.iter().zip(&want) {
        assert!(
            (f(g) - w).abs() < 1e-3 * w,
            "continuum marker {} against {w}",
            f(g)
        );
    }
}

#[test]
fn a_grained_guitars_waist_sweep_sweeps_the_same_top() {
    // The sweep rebuilds the plate at each waist; it must carry the grain, or a spruce top's
    // crossing panel describes an isotropic one. At a waist the sweep samples, on the shipped grid,
    // the sweep's row is the shipped plate — so its f1 must equal the shipped plate's own.
    let w: f64 = 0.88 * 8.0 / 23.0; // the sweep's ninth waist, linspace(0, 0.88, 24)
    let payload = ok(&musical(json!({
        "domain": "guitar", "material": "western redcedar", "N": 30, "Lx": 0.4, "Ly": 0.5,
        "waist": w, "asym": 0.2, "f0": 150.0, "f_ref": 1200.0, "t60_high": 1.4,
        "audio_duration": 0.2,
    })));
    let claim = &payload["meta"]["claim"];
    let you = f(&claim["you"]["f1"]);
    let row = claim["rows"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| (f(&r["waist"]) - (w * 1e4).round() / 1e4).abs() < 1e-9)
        .expect("the sweep samples this waist");
    assert!(
        (f(&row["f1"]) - you).abs() < 2e-3,
        "sweep f1 {} against the shipped plate's {you}",
        row["f1"]
    );
}
