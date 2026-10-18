//! Musical controls for the plates (`docs/dev/plate-mapping-plan.md`): the materials half.

use physsynth_core::plate::{Boundary, Domain, MaterialError, Params, PlateSpec};
use physsynth_core::plate_tuning::{
    plate_material, preset, thickness_for_kappa, GrainAlong, Material, PRESETS,
};

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs()
}

#[test]
fn every_preset_builds_every_plate_with_its_grain_either_way() {
    for p in PRESETS {
        for along in [GrainAlong::X, GrainAlong::Y] {
            let m = plate_material(&p.material, along, 0.003).expect("a preset is admissible");
            for (boundary, domain) in [
                (Boundary::Supported, Domain::Rectangle),
                (Boundary::Free, Domain::Rectangle),
                (Boundary::Free, Domain::Guitar),
            ] {
                let mut s = PlateSpec {
                    lx: 0.4,
                    ly: 0.5,
                    fs: 44100.0,
                    n: 10,
                    boundary: Some(boundary),
                    domain: Some(domain),
                    ..PlateSpec::default()
                };
                m.apply(&mut s);
                Params::new(&s).unwrap_or_else(|e| {
                    panic!("{} {along:?} {boundary:?}/{domain:?}: {e}", p.name)
                });
            }
        }
    }
}

#[test]
fn the_handbook_footnotes_are_applied_to_sitka_spruce() {
    // Retyped from FPL-GTR-190 here, not read from the preset: bending modulus 10,800 MPa raised
    // 10 % for shear (Table 5-1's footnote), specific gravity 0.40 on volume at 12 % moisture, so
    // 448 kg/m³ with the water (Table 5-3a's footnote b).
    let Material::Wood {
        e_l,
        e_r,
        g_lr,
        nu_lr,
        rho,
    } = preset("sitka spruce").unwrap().material
    else {
        panic!("sitka spruce is a wood");
    };
    assert!(rel(e_l, 11.88e9) < 1e-15);
    assert!(rel(e_r, 0.078 * 11.88e9) < 1e-15);
    assert!(rel(g_lr, 0.064 * 11.88e9) < 1e-15);
    assert_eq!(nu_lr, 0.372);
    assert!(rel(rho, 448.0) < 1e-15);
}

#[test]
fn an_isotropic_plate_gets_the_textbook_stiffness_and_an_areal_density() {
    // κ = sqrt(D / ρ_s) with D = E t³ / 12(1 − ν²) and ρ_s = ρ t, written out here.
    for p in PRESETS {
        let Material::Isotropic { e, nu, rho } = p.material else {
            continue;
        };
        for t in [0.0005, 0.002, 0.011] {
            let m = plate_material(&p.material, GrainAlong::X, t).unwrap();
            let kappa = t * (e / (12.0 * rho * (1.0 - nu * nu))).sqrt();
            assert!(rel(m.kappa, kappa) < 1e-14, "{} t={t}", p.name);
            // the trap: a volume density here leaves every pitch right and every energy wrong
            assert!(
                rel(m.rho_s, rho * t) < 1e-15,
                "{} t={t}: rho_s {}",
                p.name,
                m.rho_s
            );
            assert_eq!(m.nu, Some(nu));
            assert_eq!((m.grain_x, m.grain_y), (1.0, 1.0));
            assert!(m.grain_cross.is_none() && m.grain_coupling.is_none());
        }
    }
}

#[test]
fn turning_a_woods_grain_swaps_its_two_bending_rigidities_and_nothing_else() {
    // Rigidities recovered from what the plate is handed: D_x = κ² ρ_s, the rest as ratios to it.
    // Grain along y must be grain along x seen from the other side — D_x and D_y swap, the cross,
    // coupling and torsion terms stay. An unrotated Poisson ratio breaks D_1 and H (or is refused).
    let rig = |p: &physsynth_core::plate_tuning::PlateMaterial| {
        let dx = p.kappa * p.kappa * p.rho_s;
        (
            dx,
            dx * p.grain_y,
            dx * p.grain_cross.unwrap(),
            dx * p.grain_coupling.unwrap(),
            dx * p.grain_torsion.unwrap(),
        )
    };
    let mut woods = 0;
    for p in PRESETS {
        if !matches!(p.material, Material::Wood { .. }) {
            continue;
        }
        woods += 1;
        let x = rig(&plate_material(&p.material, GrainAlong::X, 0.003).unwrap());
        let y = rig(&plate_material(&p.material, GrainAlong::Y, 0.003).unwrap());
        assert!(
            rel(y.0, x.1) < 1e-12 && rel(y.1, x.0) < 1e-12,
            "{}: D_x, D_y",
            p.name
        );
        assert!(rel(y.2, x.2) < 1e-12, "{}: H", p.name);
        assert!(rel(y.3, x.3) < 1e-12, "{}: D_1", p.name);
        assert!(rel(y.4, x.4) < 1e-12, "{}: D_xy", p.name);
        // the control: a wood is genuinely orthotropic, so the swap moved something
        assert!(
            rel(x.0, x.1) > 1.0,
            "{}: D_x and D_y are not different",
            p.name
        );
    }
    assert!(woods >= 5);
}

#[test]
fn the_thickness_comes_back_from_its_own_stiffness() {
    for p in PRESETS {
        for along in [GrainAlong::X, GrainAlong::Y] {
            for t in [0.0007, 0.0027, 0.019] {
                let kappa = plate_material(&p.material, along, t).unwrap().kappa;
                let back = thickness_for_kappa(&p.material, along, kappa).unwrap();
                assert!(rel(back, t) < 1e-14, "{} {along:?} t={t}: {back}", p.name);
            }
        }
    }
}

#[test]
fn what_no_plate_can_be_is_refused() {
    let steel = preset("steel").unwrap().material;
    for t in [0.0, -1e-3, f64::NAN, f64::INFINITY] {
        assert_eq!(
            plate_material(&steel, GrainAlong::X, t),
            Err(MaterialError::NonPositive),
            "t = {t}"
        );
    }
    for kappa in [0.0, -1.0, f64::NAN] {
        assert!(thickness_for_kappa(&steel, GrainAlong::X, kappa).is_err());
    }
    let rubbery = Material::Isotropic {
        e: 1e9,
        nu: 0.5,
        rho: 1000.0,
    };
    assert!(matches!(
        plate_material(&rubbery, GrainAlong::X, 0.001),
        Err(MaterialError::Inadmissible { .. })
    ));
    // A made-up wood whose Poisson pair breaks thermodynamics: ν_LR ν_RL = ν_LR² E_R/E_L ≥ 1.
    let impossible = Material::Wood {
        e_l: 1e10,
        e_r: 1e10,
        g_lr: 1e9,
        nu_lr: 1.2,
        rho: 500.0,
    };
    for along in [GrainAlong::X, GrainAlong::Y] {
        assert!(matches!(
            plate_material(&impossible, along, 0.001),
            Err(MaterialError::Inadmissible { .. })
        ));
    }
}

#[test]
fn every_preset_names_its_source_and_names_are_unique() {
    let mut names: Vec<&str> = PRESETS.iter().map(|p| p.name).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), PRESETS.len());
    assert!(PRESETS.iter().all(|p| !p.source.is_empty()));
    assert!(preset("unobtainium").is_none());
}
