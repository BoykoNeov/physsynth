//! The geometrically exact string's rotating wave, spun — carried from
//! `tests/test_geometric_rotating_wave.py` (retirement plan §36).
//!
//! The helix `u = φ cos Ωt`, `w = φ sin Ωt`, `v = ψ`. `V` sees the polarizations only through
//! `r² = u_x² + w_x²`, so a circular polarization freezes the whole nonlinearity — stretch, tension
//! field and longitudinal forcing all go static — and the string is bent into a fixed helix and
//! spun. Seeded from the converged BVP, the longitudinal field does not move **at all**
//! (`long_kin/E ~ 1e-26`, against `8e-3` planar).
//!
//! These are the bars that need a **string**; the ones that need only the BVP live with the solver
//! in `crates/physsynth-analysis/tests/rotating_wave.rs`. The fixture is the Python's `_string()`:
//! `N = 32`, `kappa = 0` (bending is irrelevant to the relative equilibrium), `EA = 1e5`, and
//! `lam_long = 0.5`, so a failure is the BVP's and not an under-resolved longitudinal field. Every
//! negative control seeds the same string differently and asks for **orders** of difference: the
//! claim is that each ingredient — the exact history, the converged shape, the static stretch, the
//! degeneracy — is load-bearing.
//!
//! Nothing the Python compared against was NumPy's or SciPy's own number (the BVP, the history and
//! the core Jacobian were all Rust through the binding), so nothing is frozen, as in §35.

mod geometric_fixture;

use geometric_fixture::*;
use physsynth_analysis::damping::spatial_eigenvalue_p2;
use physsynth_analysis::duffing::kc_mode_coefficients;
use physsynth_analysis::rotating_wave::{kc_circular_frequency, planar_hessian_cells};
use physsynth_core::string_geometric::{dg_jacobian, GeometricString};

/// The Python's `AMP`: the shape deformation is ~1e-5 here, so a nonlinearity bug cannot hide,
/// and the continuation stays in Newton's basin at its default eight steps.
const AMP: f64 = 5e-3;

/// `_string(**kw)`: degenerate, flexible, lossless, at the helper's `lam_long = 0.5`.
fn string() -> Geo {
    Geo {
        kappa: 0.0,
        ..Geo::new(32)
    }
}

/// `long_kin / E` after 400 steps of a string seeded by `seed`.
fn spun_400(s: &mut GeometricString) -> f64 {
    let e = s.energy();
    spin(s, 400).0 / e
}

#[test]
fn a_seeded_helix_rotates_rigidly_and_its_longitudinal_field_never_moves() {
    // Carried from `test_seeded_helix_rotates_rigidly_and_the_longitudinal_field_never_moves`: one
    // full revolution. `psi` is NONZERO -- the helix holds a static stretch against its own frozen
    // transverse load -- so `v == 0` would be the wrong test; what is bit-zero is the MOTION.
    let mut s = string().build();
    let wave = helix(&s, AMP);
    seed_helix(&mut s, &wave);
    let e0 = s.energy();
    let steps = (s.p.fs / wave.frequency) as usize; // one revolution
    let (lk, r_dev, _) = spin(&mut s, steps);
    println!(
        "{steps} steps: long_kin/E {:.3e}, radius {r_dev:.3e}",
        lk / e0
    );
    assert!(lk / e0 < 1e-20, "long_kin/E = {:.3e}", lk / e0);
    assert!(
        r_dev < 1e-9,
        "the helix must be rigid: |(u, w)| moved {r_dev:.3e}"
    );
    assert!(max_abs(&wave.psi) > 1e-7, "the static stretch must be real");
    assert!(max_abs(&s.v) > 1e-7, "and the string must still hold it");
}

#[test]
fn a_spinning_helix_conserves_energy_with_the_nonlinearity_engaged() {
    // Carried from `test_rotating_wave_conserves_energy`. A steady state is where a conservation
    // bug hides behind nothing happening, and a nonlinearity bug hides at small amplitude: hence
    // the second assertion.
    let mut s = string().build();
    let wave = helix(&s, AMP);
    seed_helix(&mut s, &wave);
    let e0 = s.energy();
    step_n(&mut s, 400);
    let drift = rel_drift(s.energy(), e0);
    let nl = s.nonlinear_energy().abs() / e0;
    println!("drift {drift:.3e}, nonlinear fraction {nl:.3e}");
    assert!(drift < DRIFT_GATE, "drift {drift:.3e}");
    assert!(nl > 1e-6, "the nonlinearity must be engaged: {nl:.3e}");
}

#[test]
fn a_taylor_start_leaks_orders_of_magnitude_more_than_the_exact_history() {
    // Carried from `test_set_state_seeding_costs_ten_orders`. `set_state`'s `y^{-1}` is a
    // second-order Taylor start: consistent, not exact. The helix sheds that history error into
    // the longitudinal field, and the number it leaves (~6e-18) still LOOKS like machine precision.
    let base = string().build();
    let wave = helix(&base, AMP);

    let mut exact = string().build();
    seed_helix(&mut exact, &wave);
    let lk_exact = spun_400(&mut exact);

    let mut taylor = string().build();
    let z = zeros(&taylor);
    let w_dot: Vec<f64> = wave.phi.iter().map(|p| wave.omega * p).collect();
    taylor.set_state(&wave.phi, &z, &wave.psi, &[z.clone(), w_dot, z.clone()]);
    let lk_taylor = spun_400(&mut taylor);

    println!("exact {lk_exact:.3e}, taylor {lk_taylor:.3e}");
    assert!(
        lk_taylor > 1e6 * lk_exact,
        "{lk_taylor:.3e} vs {lk_exact:.3e}"
    );
}

#[test]
fn a_sine_at_the_kirchhoff_carrier_frequency_is_not_a_relative_equilibrium() {
    // Carried from `test_a_sine_is_not_a_relative_equilibrium`: the obvious guess -- a sine at the
    // KC circular frequency, with the BVP's own `psi` handed to it free -- still pumps the
    // longitudinal field orders harder than the converged shape. What is left is the shape
    // deformation the BVP solved for.
    let base = string().build();
    let wave = helix(&base, AMP);
    let p = &base.p;
    let p2 = spatial_eigenvalue_p2(p.n as i64, p.h, 1);
    let (omega0_sq, eps) = kc_mode_coefficients(p.c, 0.0, p.ea - p.t, p.rho, p2, p.l).unwrap();
    let omega_kc = kc_circular_frequency(omega0_sq, eps, AMP).unwrap();

    let mut exact = string().build();
    seed_helix(&mut exact, &wave);
    let lk_exact = spun_400(&mut exact);

    let mut sine = string().build();
    let shape = mode_ic(&sine, 1, AMP);
    let k = sine.p.k;
    sine.u = shape.clone();
    sine.w = zeros(&sine);
    sine.v = wave.psi.clone();
    sine.u_prev = shape.iter().map(|x| x * (omega_kc * k).cos()).collect();
    sine.w_prev = shape.iter().map(|x| -x * (omega_kc * k).sin()).collect();
    sine.v_prev = wave.psi.clone();
    let lk_sine = spun_400(&mut sine);

    println!("exact {lk_exact:.3e}, sine {lk_sine:.3e}");
    assert!(lk_sine > 1e6 * lk_exact, "{lk_sine:.3e} vs {lk_exact:.3e}");
}

#[test]
fn circular_is_bit_zero_where_planar_is_percent_level() {
    // Carried from `test_circular_is_bit_zero_where_planar_is_percent_level`: same string, same
    // amplitude, same mode, only the polarization differs -- the claim model #9 cannot make.
    let base = string().build();
    let wave = helix(&base, AMP);

    let mut circ = string().build();
    seed_helix(&mut circ, &wave);
    let lk_circ = spun_400(&mut circ);

    let mut planar = string().build();
    let shape = mode_ic(&planar, 1, AMP);
    start_u(&mut planar, &shape);
    let lk_planar = spun_400(&mut planar);

    println!("circular {lk_circ:.3e}, planar {lk_planar:.3e}");
    assert!(lk_circ < 1e-20, "circular {lk_circ:.3e}");
    assert!(lk_planar > 1e-4, "planar {lk_planar:.3e}");
    assert!(lk_planar / lk_circ > 1e15);
}

#[test]
fn the_static_stretch_is_the_ingredient_batch_two_never_varied() {
    // Carried from `test_the_static_stretch_is_the_ingredient_batch_2_never_varied`: every circular
    // start in batch 2 had `v^0 = 0`, so `psi` was pinned at zero and could not appear as a
    // variable -- and it is the biggest lever. Released with `v = 0` the helix is not near its
    // equilibrium; it rings about the stretch it should already be holding. The phi-vs-Omega
    // ordering is NOT asserted: it inverts between metrics, unexplained
    // (`docs/dev/geometrically-exact-string-plan.md`).
    let base = string().build();
    let wave = helix(&base, AMP);
    let seeded = |psi: &[f64]| -> GeometricString {
        let mut s = string().build();
        let k = s.p.k;
        s.u = wave.phi.clone();
        s.w = zeros(&s);
        s.v = psi.to_vec();
        s.u_prev = wave
            .phi
            .iter()
            .map(|x| x * (wave.omega * k).cos())
            .collect();
        s.w_prev = wave
            .phi
            .iter()
            .map(|x| -x * (wave.omega * k).sin())
            .collect();
        s.v_prev = psi.to_vec();
        s
    };
    let lk_full = spun_400(&mut seeded(&wave.psi));
    let lk_none = spun_400(&mut seeded(&zeros(&base)));
    println!("full {lk_full:.3e}, psi = 0 {lk_none:.3e}");
    assert!(lk_none > 1e15 * lk_full, "{lk_none:.3e} vs {lk_full:.3e}");
    assert!(lk_full < 1e-20, "{lk_full:.3e}");
}

#[test]
fn a_helix_does_not_survive_on_a_non_degenerate_string() {
    // Carried from `test_helix_does_not_survive_on_a_non_degenerate_string`. One `phi` must serve
    // both polarizations, so the `u` and `w` rows must be the same equation. A degenerate string
    // cannot whirl but can spin rigidly; a detuned one can whirl but cannot spin: the knob that
    // unlocks one closes the other.
    let base = string().build();
    let wave = helix(&base, AMP);

    let mut degenerate = string().build();
    assert!(degenerate.is_degenerate());
    seed_helix(&mut degenerate, &wave);
    let lk_deg = spun_400(&mut degenerate);

    let mut detuned = Geo {
        kappa_w: Some(0.3),
        ..string()
    }
    .build();
    assert!(!detuned.is_degenerate());
    seed_helix(&mut detuned, &wave);
    let lk_det = spun_400(&mut detuned);

    println!("degenerate {lk_deg:.3e}, detuned {lk_det:.3e}");
    assert!(lk_det > 1e6 * lk_deg, "{lk_det:.3e} vs {lk_deg:.3e}");
}

#[test]
fn the_planar_hessian_is_twice_the_core_discrete_gradient_jacobian() {
    // Carried from `test_planar_hessian_matches_the_core_discrete_gradient_jacobian`: two
    // independent derivations of one Hessian -- the oracle's, in closed form on the planar slice,
    // and the core's, assembled for its Newton solve -- meeting at `q+ == q-`. The 2 is
    // `d(qbar)/d(q+) = 1/2`. The `(v, v)` block was once the loose one (the core cancelled two
    // `O(1)` terms, 7e-11 at strain 1e-3); it is now assembled cancellation-free and held to the
    // same bar as the rest. The draws stand in for NumPy's: the claim is about every state (§15).
    let s = string().build();
    let a = s.p.ea - s.p.t;
    let n = s.p.n;
    for (j, strain) in [1e-3, 1e-2, 0.1].into_iter().enumerate() {
        let p: Vec<f64> = (0..n).map(|i| strain * normal(7, 2 * j * n + i)).collect();
        let z: Vec<f64> = (0..n)
            .map(|i| 0.1 * strain * normal(7, (2 * j + 1) * n + i))
            .collect();
        let mut q = p.clone();
        q.extend(std::iter::repeat_n(0.0, n));
        q.extend(&z);
        let core = dg_jacobian(&q, &q, a);
        let (h_pp, h_pz, h_zz) = planar_hessian_cells(&p, &z, a);
        let worst = |h: &[f64], block: (usize, usize)| -> f64 {
            let gap: Vec<f64> = (0..n)
                .map(|i| h[i] - 2.0 * core.get(block.0 * n + i, block.1 * n + i))
                .collect();
            max_abs(&gap) / max_abs(h)
        };
        let gaps = [
            worst(&h_pp, (0, 0)),
            worst(&h_pz, (0, 2)),
            worst(&h_pz, (2, 0)),
            worst(&h_zz, (2, 2)),
        ];
        println!("strain {strain:e}: (u,u) (u,v) (v,u) (v,v) {gaps:?}");
        for g in gaps {
            assert!(g < 1e-12, "strain {strain:e}: {gaps:?}");
        }
    }
}
