//! Native bars for the Kirchhoff plate — models #5, #5b, #5o, #5of, #5g and #6.
//!
//! The project's acceptance contract is energy, so that is what most of these assert: a lossless
//! run conserves to machine precision and a lossy one decreases monotonically, on every one of the
//! four branches. Two of them are about something else and are the reason this file exists rather
//! than leaning on the Python parity suite:
//!
//! * **the anchor**, asserted here as well as there — a `VkPlate` with the coupling off is
//!   bit-identical to a `Plate`, which in this implementation is structural (they share a
//!   `Params` and a `step_rhs`) rather than a claim about two transcriptions staying in step;
//! * **`linspace0`'s overwritten endpoint**, which is one ulp and decides whether a node on a
//!   curved rim is alive.

use physsynth_core::ops2d::Mask;
use physsynth_core::plate::{
    count_components, energy, grain_ratios_from_material, linspace0, pickup_index_at, vk_newton,
    Boundary, CoupleMethod, CoupleOutcome, Domain, MaterialError, ParamError, Params, Plate,
    PlateSpec, VkCoupledStep, VkParamError, VkParams, VkPlate, VkSpec,
};

const FS: f64 = 20_000.0;

fn spec(boundary: Boundary, domain: Domain, n: i64) -> PlateSpec {
    PlateSpec {
        lx: 0.4,
        ly: 0.4,
        kappa: 1.0,
        rho: 2.0,
        fs: FS,
        n,
        boundary: Some(boundary),
        domain: Some(domain),
        ..PlateSpec::default()
    }
}

fn plate(boundary: Boundary, domain: Domain, n: i64) -> Plate {
    Plate::new(Params::new(&spec(boundary, domain, n)).expect("a valid plate"))
}

/// A centred Gaussian bump over the live nodes, peak `amp`.
fn bump(p: &Params, amp: f64) -> Vec<f64> {
    let (cx, cy) = (0.5 * p.lx, 0.5 * p.ly);
    p.mask
        .flags()
        .iter()
        .enumerate()
        .filter(|(_, &alive)| alive)
        .map(|(idx, _)| {
            let (dx, dy) = (p.x[idx] - cx, p.y[idx] - cy);
            amp * (-((dx * dx + dy * dy) / 0.01)).exp()
        })
        .collect()
}

fn drift(p: &mut Plate, steps: usize) -> f64 {
    let e0 = p.energy();
    let mut worst = 0.0f64;
    for _ in 0..steps {
        p.step(None);
        worst = worst.max((p.energy() / e0 - 1.0).abs());
    }
    worst
}

// -- energy, on all four branches ----------------------------------------------------------

#[test]
fn a_lossless_plate_conserves_its_energy_on_every_branch() {
    // The project's acceptance bar is 1e-10 and is deliberately not tightened (CLAUDE.md).
    for (label, boundary, domain, n) in [
        ("supported", Boundary::Supported, Domain::Rectangle, 12),
        ("free", Boundary::Free, Domain::Rectangle, 12),
        ("circle", Boundary::Free, Domain::Circle, 16),
        ("guitar", Boundary::Free, Domain::Guitar, 20),
    ] {
        let mut p = plate(boundary, domain, n);
        let u0 = bump(&p.p, 1e-4);
        p.set_state(&u0, &vec![0.0; p.p.n_live]);
        let worst = drift(&mut p, 500);
        assert!(worst < 1e-10, "{label}: drift {worst:.3e}");
    }
}

#[test]
fn a_lossy_plate_is_passive() {
    for (boundary, domain) in [
        (Boundary::Supported, Domain::Rectangle),
        (Boundary::Free, Domain::Rectangle),
    ] {
        let mut s = spec(boundary, domain, 12);
        s.sigma = 40.0;
        let mut p = Plate::new(Params::new(&s).expect("a valid plate"));
        let u0 = bump(&p.p, 1e-4);
        p.set_state(&u0, &vec![0.0; p.p.n_live]);
        let e0 = p.energy();
        let mut last = e0;
        for step in 0..300 {
            p.step(None);
            let now = p.energy();
            assert!(now <= last * (1.0 + 1e-12), "step {step}: {now} > {last}");
            last = now;
        }
        // Monotone is only half the claim -- a plate that never moved is monotone too.
        assert!(last < 0.5 * e0, "the loss did no work: {last} against {e0}");
    }
}

/// The Navier modal law is `f_mn = (pi/2) kappa [(m/Lx)^2 + (n/Ly)^2]`, and the *discrete*
/// operator has `sin(m pi x/Lx) sin(n pi y/Ly)` as an exact eigenvector.
#[test]
fn the_supported_plate_has_the_navier_mode_as_an_exact_eigenvector() {
    let p = plate(Boundary::Supported, Domain::Rectangle, 16);
    let (lx, ly) = (p.p.lx, p.p.ly);
    let mode: Vec<f64> =
        p.p.mask
            .flags()
            .iter()
            .enumerate()
            .filter(|(_, &alive)| alive)
            .map(|(idx, _)| {
                (std::f64::consts::PI * p.p.x[idx] / lx).sin()
                    * (2.0 * std::f64::consts::PI * p.p.y[idx] / ly).sin()
            })
            .collect();
    // B is a positive-definite operator; the Rayleigh quotient must reproduce the entrywise ratio.
    let bu = p.p.stiffness.matvec(&mode);
    let mut ratio = f64::NAN;
    for (i, &m) in mode.iter().enumerate() {
        if m.abs() < 1e-6 {
            continue;
        }
        let r = bu[i] / m;
        if ratio.is_nan() {
            ratio = r;
        } else {
            assert!(
                (r / ratio - 1.0).abs() < 1e-11,
                "node {i}: {r} against {ratio}"
            );
        }
    }
    assert!(ratio > 0.0, "B must be positive-definite");
}

/// A free plate's stiffness annihilates `{1, x, y}` — the three rigid-body motions.
#[test]
fn the_free_plate_stiffness_annihilates_its_rigid_body_nullspace() {
    for (label, domain, n) in [
        ("rectangle", Domain::Rectangle, 12),
        ("circle", Domain::Circle, 16),
        ("guitar", Domain::Guitar, 20),
    ] {
        let p = plate(Boundary::Free, domain, n);
        let live: Vec<usize> = (0..p.p.mask.flags().len())
            .filter(|&i| p.p.mask.flags()[i])
            .collect();
        let scale: f64 =
            p.p.stiffness
                .data()
                .iter()
                .fold(0.0f64, |a, b| a.max(b.abs()));
        for (which, f) in [
            ("1", Box::new(|_: usize| 1.0) as Box<dyn Fn(usize) -> f64>),
            ("x", Box::new(|idx: usize| p.p.x[idx])),
            ("y", Box::new(|idx: usize| p.p.y[idx])),
        ] {
            let v: Vec<f64> = live.iter().map(|&idx| f(idx)).collect();
            let kv = p.p.stiffness.matvec(&v);
            let worst = kv.iter().fold(0.0f64, |a, b| a.max(b.abs()));
            assert!(
                worst < 1e-9 * scale,
                "{label}: K annihilates {which} only to {worst:.3e} against a scale of {scale:.3e}"
            );
        }
    }
}

// -- the anchor ------------------------------------------------------------------------------

#[test]
fn a_linear_von_karman_plate_is_bit_identical_to_the_linear_plate() {
    // Structural here, not transcribed: `VkParams` OWNS a `Params` and both classes step through
    // the same `step_rhs`. In the Python original the two are separate spellings of the theta
    // scheme kept in step by a docstring, and the suite pins them with 150 steps of `array_equal`.
    for boundary in [Boundary::Supported, Boundary::Free] {
        let vk_spec = VkSpec {
            lx: 0.4,
            ly: 0.4,
            young: 2.0e11,
            thickness: 1e-3,
            nu: 0.3,
            rho: 7860.0,
            fs: 48_000.0,
            n: 16,
            boundary: Some(boundary),
            nonlinear: false,
            ..VkSpec::default()
        };
        let mut vk = VkPlate::new(VkParams::new(&vk_spec).expect("a valid plate"));
        let mut p5 = Plate::new(
            Params::new(&PlateSpec {
                lx: 0.4,
                ly: 0.4,
                kappa: vk.p.lin.kappa,
                rho: vk.p.rho_s,
                fs: 48_000.0,
                n: 16,
                boundary: Some(boundary),
                ..PlateSpec::default()
            })
            .expect("a valid plate"),
        );
        let u0 = bump(&p5.p, 1e-4);
        let zero = vec![0.0; p5.p.n_live];
        vk.set_state(&u0, &zero).expect("the Airy solve factors");
        p5.set_state(&u0, &zero);
        for step in 0..150 {
            vk.step(None).expect("the solves succeed");
            p5.step(None);
            assert_eq!(vk.u, p5.u, "diverged at step {step}");
        }
        assert_eq!(vk.energy(), p5.energy());
    }
}

#[test]
fn a_nonlinear_plate_conserves_its_total_energy() {
    for boundary in [Boundary::Supported, Boundary::Free] {
        let mut vk = VkPlate::new(
            VkParams::new(&VkSpec {
                lx: 0.4,
                ly: 0.4,
                young: 2.0e11,
                thickness: 1e-3,
                nu: 0.3,
                rho: 7860.0,
                fs: 48_000.0,
                n: 16,
                boundary: Some(boundary),
                ..VkSpec::default()
            })
            .expect("a valid plate"),
        );
        // Three thicknesses of amplitude -- the membrane term is a real fraction of the total.
        let u0 = bump(&vk.p.lin, 3.0 * vk.p.thickness);
        let zero = vec![0.0; vk.p.lin.n_live];
        vk.set_state(&u0, &zero).expect("the Airy solve factors");
        assert!(
            vk.membrane_energy() > 0.0,
            "the coupling must be doing work"
        );
        let e0 = vk.energy();
        let mut worst = 0.0f64;
        for _ in 0..300 {
            vk.step(None).expect("the solves succeed");
            worst = worst.max((vk.energy() / e0 - 1.0).abs());
        }
        assert!(worst < 1e-10, "drift {worst:.3e}");
    }
}

// -- why the sweep loop stopped (plan §5 Part 0) ------------------------------------------------

/// A 40 cm steel square struck in the middle, `amp` thicknesses tall, with a chosen sweep cap.
///
/// **Pinned to `Picard` rather than left on the default.** Everything below is a claim about why
/// the *sweep loop* stopped, and under `Auto` there is no such claim to make: the rescue converges
/// these fixtures, `outcome()` reads `Converged`, and every assertion here passes on nothing. The
/// sixty-thickness strike is the sharpest case -- it reported `Converged` on all twenty steps.
fn struck_vk(amp: f64, cap: i64) -> VkPlate {
    let mut vk = VkPlate::new(
        VkParams::new(&VkSpec {
            lx: 0.4,
            ly: 0.4,
            young: 2.0e11,
            thickness: 1e-3,
            nu: 0.3,
            rho: 7860.0,
            fs: 48_000.0,
            n: 16,
            couple_max_iter: cap,
            couple_method: Some(CoupleMethod::Picard),
            ..VkSpec::default()
        })
        .expect("a valid plate"),
    );
    let u0 = bump(&vk.p.lin, amp * vk.p.thickness);
    let zero = vec![0.0; vk.p.lin.n_live];
    vk.set_state(&u0, &zero).expect("the Airy solve factors");
    vk
}

/// The same strike reports `Converged` or `Capped` purely by how many sweeps it was allowed.
///
/// Half of Part 0's point: at this amplitude `converged == false` carries no information about the
/// physics at all, only about the constructor argument.
#[test]
fn the_same_strike_is_converged_or_capped_by_its_cap_alone() {
    let mut easy = struck_vk(3.0, 50);
    easy.step(None).expect("the solves succeed");
    assert!(easy.converged, "3e converges inside 50 sweeps");
    assert_eq!(easy.outcome(), CoupleOutcome::Converged);

    // The identical strike, starved of sweeps: still contracting when the cap ran out.
    let mut starved = struck_vk(3.0, 3);
    starved.step(None).expect("the solves succeed");
    assert!(!starved.converged, "3 sweeps is not enough for 3e");
    assert_eq!(starved.n_iters, 3);
    assert!(
        starved.residual_ratio < 1.0,
        "ratio {:.3e} should be a contraction",
        starved.residual_ratio
    );
    assert_eq!(starved.outcome(), CoupleOutcome::Capped);

    // One sweep cannot form a ratio, and the outcome says so rather than guessing.
    let mut blind = struck_vk(3.0, 1);
    blind.step(None).expect("the solves succeed");
    assert!(blind.residual_ratio.is_nan());
    assert_eq!(blind.outcome(), CoupleOutcome::Unknown);
}

/// Hit hard enough, the sweeps *expand*, and no cap at any size would have converged that step.
///
/// The other half of Part 0's point, and the one a bigger `couple_max_iter` cannot buy back. The
/// amplitude here is far past anything the model claims to resolve — it is chosen to make the
/// verdict unambiguous, not to be a plate anyone would strike.
#[test]
fn a_plate_hit_far_too_hard_reports_the_wall_rather_than_the_cap() {
    let mut vk = struck_vk(60.0, 50);
    let mut seen = Vec::new();
    for _ in 0..20 {
        vk.step(None).expect("the solves succeed");
        seen.push(vk.outcome());
        if vk.outcome() == CoupleOutcome::Expansive {
            break;
        }
    }
    assert!(
        seen.contains(&CoupleOutcome::Expansive),
        "60 thicknesses should expand, saw {seen:?}"
    );
}

/// The linear path forms no ratio at all, and is `Converged` by construction, not by measurement.
#[test]
fn the_linear_path_reports_a_converged_step_with_no_ratio() {
    let mut vk = VkPlate::new(
        VkParams::new(&VkSpec {
            lx: 0.4,
            ly: 0.4,
            young: 2.0e11,
            thickness: 1e-3,
            nu: 0.3,
            rho: 7860.0,
            fs: 48_000.0,
            n: 12,
            nonlinear: false,
            ..VkSpec::default()
        })
        .expect("a valid plate"),
    );
    let u0 = bump(&vk.p.lin, 3.0 * vk.p.thickness);
    let zero = vec![0.0; vk.p.lin.n_live];
    vk.set_state(&u0, &zero).expect("the Airy solve factors");
    vk.step(None).expect("the solve succeeds");
    assert_eq!(vk.n_iters, 1);
    assert!(vk.residual_ratio.is_nan());
    assert_eq!(vk.outcome(), CoupleOutcome::Converged);
}

/// A blown-up step carries a non-finite residual, and that must not read as "short of sweeps".
///
/// This is the reason the `Capped` test is positive: `NaN < 1.0` is `false`, but so is
/// `NaN >= 1.0`, so a negative test would file an overflow under the outcome a bigger cap fixes.
#[test]
fn a_non_finite_residual_is_expansive_and_not_capped() {
    assert_eq!(
        physsynth_core::plate::couple_outcome(false, 50, f64::NAN, f64::NAN),
        CoupleOutcome::Expansive
    );
    assert_eq!(
        physsynth_core::plate::couple_outcome(false, 50, f64::INFINITY, 2.0),
        CoupleOutcome::Expansive
    );
    assert_eq!(
        physsynth_core::plate::couple_outcome(false, 50, 1e-3, 1.4),
        CoupleOutcome::Expansive
    );
    assert_eq!(
        physsynth_core::plate::couple_outcome(false, 50, 1e-3, 0.6),
        CoupleOutcome::Capped
    );
    // `converged` wins over everything else — a root is a root.
    assert_eq!(
        physsynth_core::plate::couple_outcome(true, 1, 0.0, f64::NAN),
        CoupleOutcome::Converged
    );
}

// -- geometry --------------------------------------------------------------------------------

#[test]
fn linspace_overwrites_its_endpoint_rather_than_computing_it() {
    // NumPy sets the last entry to `stop` outright. For most extents `(num-1) * step` is a
    // different double, and on a curved rim that one ulp is a live node or a dead one.
    // Searched rather than hand-picked: a constant witness lands in the agreeing majority about
    // two times in three, and a pin that finds nothing cannot tell "no difference exists" from
    // "I did not look in the right place" (plan sections 23.5 and 26.6).
    let num = 21usize;
    let div = (num - 1) as f64;
    let mut witnesses = 0usize;
    for i in 1..2000 {
        let stop = (i as f64) * 1e-3;
        let step = stop / div;
        if div * step != stop {
            witnesses += 1;
            let xs = linspace0(stop, num);
            assert_eq!(
                xs[num - 1],
                stop,
                "the endpoint is overwritten, not computed"
            );
            assert_eq!(xs[0], 0.0);
            assert_eq!(xs[7], 7.0 * step);
        }
    }
    assert!(
        witnesses > 100,
        "only {witnesses} extents of 1999 distinguish the two spellings -- the search is wrong"
    );
}

#[test]
fn components_are_counted_four_connected() {
    // Two 2x2 blocks touching only at a corner are two plates, not one: a diagonal neighbour does
    // not carry a bending moment, and the failure mode is a doubled rigid-body nullspace that
    // reads as a suspiciously low fundamental rather than as an error.
    let mut live = vec![false; 25];
    for (j, i) in [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 2),
        (2, 3),
        (3, 2),
        (3, 3),
    ] {
        live[j * 5 + i] = true;
    }
    assert_eq!(count_components(&Mask::new(5, 5, live.clone())), 2);
    live[5 + 2] = true; // bridge them
    assert_eq!(count_components(&Mask::new(5, 5, live)), 1);
    assert_eq!(count_components(&Mask::new(3, 3, vec![false; 9])), 0);
}

#[test]
fn the_pickup_index_is_the_nearest_live_node_and_ties_go_first() {
    let p = plate(Boundary::Free, Domain::Rectangle, 8);
    // A node position must map to its own index.
    let mut live = 0usize;
    for (idx, &alive) in p.p.mask.flags().iter().enumerate() {
        if alive {
            assert_eq!(pickup_index_at(p.p.x[idx], p.p.y[idx], &p.p), live);
            live += 1;
        }
    }
    // Exactly between the first two nodes along x, `np.argmin` takes the first.
    assert_eq!(pickup_index_at(0.5 * p.p.h, 0.0, &p.p), 0);
}

#[test]
fn a_guitar_reports_a_staircase_deficit_and_prunes_only_at_the_rim() {
    let p = plate(Boundary::Free, Domain::Guitar, 24);
    assert!(p.p.area < p.p.outline_area, "the staircase loses area");
    assert!(
        p.p.area_deficit < 0.0 && p.p.area_deficit > -0.2,
        "deficit {:.4}",
        p.p.area_deficit
    );
    // Exposed rather than only compared: a bar that raises only on violation is never observed on
    // a grid that passes, so a sign error here would leave every test green.
    if p.p.n_pruned > 0 {
        assert!(p.p.prune_depth_max > 0.0);
        assert!(p.p.prune_depth_max <= 1.0001 * p.p.h);
    }
}

// -- the refusals ----------------------------------------------------------------------------

#[test]
fn the_grain_guards_are_the_branchs_own() {
    // Supported: definiteness of g_x a^2 + 2 g_h a b + g_y b^2 needs g_h > -sqrt(g_x g_y).
    let mut s = spec(Boundary::Supported, Domain::Rectangle, 8);
    s.grain_cross = Some(-1.01);
    assert!(matches!(
        Params::new(&s),
        Err(ParamError::IndefiniteCross { .. })
    ));

    // Free: a DIFFERENT set, not a tighter one -- the coupling ceiling and positive torsion.
    let mut s = spec(Boundary::Free, Domain::Rectangle, 8);
    s.grain_coupling = Some(1.5);
    s.grain_torsion = Some(0.5);
    assert!(matches!(
        Params::new(&s),
        Err(ParamError::IndefiniteCoupling { .. })
    ));
    s.grain_coupling = Some(0.3);
    s.grain_torsion = Some(0.0);
    assert!(matches!(
        Params::new(&s),
        Err(ParamError::NonPositiveTorsion(_))
    ));

    // Half a split is refused rather than completed from Poisson's ratio.
    let mut s = spec(Boundary::Free, Domain::Rectangle, 8);
    s.grain_coupling = Some(0.3);
    assert!(matches!(Params::new(&s), Err(ParamError::HalfSplit)));

    // A grained free plate with no split at all.
    let mut s = spec(Boundary::Free, Domain::Rectangle, 8);
    s.grain_y = 0.3;
    assert!(matches!(Params::new(&s), Err(ParamError::FreeNeedsSplit)));
}

#[test]
fn a_curved_supported_plate_is_a_refusal_not_a_limitation() {
    let s = spec(Boundary::Supported, Domain::Circle, 12);
    let e = Params::new(&s).expect_err("a supported disk is refused");
    assert!(matches!(e, ParamError::CurvedSupported(Domain::Circle)));
    assert!(e.to_string().contains("offered on boundary='free' only"));
}

#[test]
fn the_split_contradiction_message_prints_the_effective_cross_term() {
    let mut s = spec(Boundary::Free, Domain::Rectangle, 8);
    s.grain_coupling = Some(0.0269);
    s.grain_torsion = Some(0.063);
    s.grain_cross = Some(0.9);
    let e = Params::new(&s).expect_err("a contradicted split is refused");
    let text = e.to_string();
    assert!(text.contains("contradicts the split"), "{text}");
    assert!(text.contains("= 0.1529)"), "{text}");
}

#[test]
fn a_nonlinear_plate_refuses_what_the_original_refuses() {
    let base = VkSpec {
        lx: 0.4,
        ly: 0.4,
        young: 2.0e11,
        thickness: 1e-3,
        nu: 0.3,
        rho: 7860.0,
        fs: 48_000.0,
        n: 8,
        ..VkSpec::default()
    };
    let cases: [(VkSpec, VkParamError); 4] = [
        (
            VkSpec {
                young: 0.0,
                ..base.clone()
            },
            VkParamError::NonPositiveYoung,
        ),
        (
            VkSpec {
                thickness: 0.0,
                ..base.clone()
            },
            VkParamError::NonPositiveThickness,
        ),
        (
            VkSpec {
                couple_tol: 0.0,
                ..base.clone()
            },
            VkParamError::NonPositiveTol,
        ),
        (
            VkSpec {
                couple_max_iter: 0,
                ..base.clone()
            },
            VkParamError::TooFewSweeps,
        ),
    ];
    for (s, want) in cases {
        assert_eq!(VkParams::new(&s).err(), Some(want));
    }
}

// -- the material helper ----------------------------------------------------------------------

#[test]
fn isotropic_material_comes_back_at_exactly_one() {
    // The H convention is the single most transcribable-wrong number in this model, and this is
    // the assertion that pins it: an isotropic sheet must land on `(1, 1, 1)` EXACTLY, or the
    // shipped plate would move the moment anyone routed the default through the helper.
    let nu = 0.3;
    let e = 1.1e10;
    let g = e / (2.0 * (1.0 + nu));
    let s = grain_ratios_from_material(e, e, nu, g, 3e-3, 420.0).expect("admissible");
    assert_eq!(s.grain_x, 1.0);
    assert_eq!(s.grain_y, 1.0);
    assert_eq!(s.grain_cross, 1.0);
    // And the split it implies is the isotropic free plate's own assembly.
    assert!((s.grain_coupling - nu).abs() < 1e-15);
    assert!((s.grain_torsion - 0.5 * (1.0 - nu)).abs() < 1e-15);
    assert_eq!(s.rho_s, 420.0 * 3e-3);
}

#[test]
fn spruce_is_not_an_isotropic_plate_with_one_axis_stretched() {
    let s = grain_ratios_from_material(11e9, 0.8e9, 0.37, 0.7e9, 3e-3, 420.0).expect("admissible");
    assert!((s.grain_y - 0.0727).abs() < 1e-3, "{}", s.grain_y);
    assert!((s.grain_cross - 0.153).abs() < 2e-3, "{}", s.grain_cross);
    // 82% of the cross term is torsional -- which is why a FREE plate, whose fundamental is set by
    // D_xy alone, is the boundary that can measure it.
    let torsional = 2.0 * s.grain_torsion / s.grain_cross;
    assert!(torsional > 0.8 && torsional < 0.85, "{torsional}");
    // H / sqrt(D_x D_y) is nowhere near 1: the cross term is an independent axis.
    let ratio = s.grain_cross / (s.grain_x * s.grain_y).sqrt();
    assert!(ratio > 0.5 && ratio < 0.65, "{ratio}");
}

#[test]
fn an_inadmissible_poisson_pair_is_refused_with_its_own_numbers() {
    let e =
        grain_ratios_from_material(1.0, 40.0, 0.9, 1.0, 1e-3, 400.0).expect_err("nu_xy nu_yx >= 1");
    assert!(matches!(e, MaterialError::Inadmissible { .. }));
    assert!(e.to_string().contains("thermodynamic admissibility"));
}

// -- the external force ------------------------------------------------------------------------

#[test]
fn an_external_force_enters_before_the_solve_and_a_zero_one_changes_nothing() {
    for boundary in [Boundary::Supported, Boundary::Free] {
        let mut a = plate(boundary, Domain::Rectangle, 10);
        let mut b = plate(boundary, Domain::Rectangle, 10);
        let u0 = bump(&a.p, 1e-4);
        let zero = vec![0.0; a.p.n_live];
        a.set_state(&u0, &zero);
        b.set_state(&u0, &zero);
        for _ in 0..20 {
            a.step(None);
            b.step(Some(&zero));
        }
        assert_eq!(a.u, b.u, "a zero force is not the same as no force");

        // A real force moves the plate, and it moves EVERY node -- the A-solve couples them, which
        // is why a post-solve correction would be invalid.
        let mut f = vec![0.0; a.p.n_live];
        f[a.p.n_live / 3] = 1.0;
        b.step(Some(&f));
        a.step(None);
        let moved = a.u.iter().zip(b.u.iter()).filter(|(x, y)| x != y).count();
        // Most of the plate, not just the driven node: a one-node response would mean the force
        // had been applied AFTER the solve, which is the invalid spelling this pins against. Not
        // every node, because a single step's response falls off below the last bit far away.
        assert!(
            moved > a.p.n_live / 2,
            "the force reached only {moved} of {} nodes",
            a.p.n_live
        );
    }
}

#[test]
fn the_energy_form_is_the_updates_own_matrix() {
    // P(f, f) >= 0 for the operator the step applies, on both branches: that is what makes the
    // conservation identity exact rather than approximate.
    for boundary in [Boundary::Supported, Boundary::Free] {
        let p = plate(boundary, Domain::Rectangle, 10);
        let u = bump(&p.p, 1e-3);
        let e = energy(&u, &vec![0.0; p.p.n_live], &p.p);
        assert!(e > 0.0, "a displaced plate stores energy");
    }
}

/// `Ny = max(int(round(Ly / h)), 1)`, and `round` there is **CPython's**, not Rust's.
///
/// Python rounds a tie to even and `f64::round` rounds a tie away from zero, so the two disagree
/// on exactly the geometries where `Ly / h` lands on a half — and that is a plate with a different
/// number of rows, not a rounding: a different mask, a different operator, a different spectrum
/// (§25.2, a discrete output). `py_round` transcribes CPython and is asserted on its own in
/// `tests/radiation.rs`; this asserts that the plate's snap actually goes through it, which is the
/// half a test of the primitive cannot see.
///
/// Moved here from `tests/test_rust_parity_plate.py` when unit 5's Python body was deleted (plan
/// §43). There the claim was "the two implementations snap alike"; here it is stated against the
/// arithmetic itself, so the expected row counts are written out rather than compared.
#[test]
fn the_side_length_snaps_with_pythons_half_to_even_round() {
    let lx = 0.4;
    // (segments along x, the multiple of `h` asked for, the row count half-to-even gives).
    // 2.5 -> 2 and 3.5 -> 4: away-from-zero would give 3 and 4, so the first of each pair is the
    // witness and the second is the control that keeps this from passing on a constant.
    let cases: [(i64, f64, usize); 8] = [
        (8, 2.5, 2),
        (8, 3.5, 4),
        (10, 4.5, 4),
        (10, 5.5, 6),
        (12, 2.5, 2),
        (16, 6.5, 6),
        (20, 3.5, 4),
        (20, 4.5, 4),
    ];
    let mut ties = 0;
    for (n, mult, want_ny) in cases {
        let h = lx / n as f64;
        let ly = h * mult;
        let p = Params::new(&PlateSpec {
            ly,
            n,
            ..spec(Boundary::Supported, Domain::Rectangle, n)
        })
        .expect("a legal plate");
        assert_eq!(
            p.ny, want_ny,
            "N={n} Ly={mult}h: snapped to {} rows, want {want_ny}",
            p.ny
        );
        // The grid really is that many rows, so the claim is about the plate and not a field.
        assert_eq!(p.mask.nrows(), want_ny + 1);
        if (ly / h % 1.0 - 0.5).abs() < 1e-9 {
            ties += 1;
        }
    }
    assert_eq!(
        ties, 8,
        "these geometries no longer land on a tie and the test is vacuous"
    );
}

// -- the Jacobian, asserted before Newton is allowed to use it (plan §5 Part 1) -----------------
//
// `VkCoupledStep::jacobian_vector` is the closed-form derivative of `VkCoupledStep::residual`,
// and the loop in `vk_step` iterates `VkCoupledStep::sweep`, of which the residual is `w - sweep`.
// So there are two independent things to establish and they are established separately:
//
// * the **derivative** is right -- a central finite difference of `residual` in the same
//   direction, which is what these tests do;
// * the **map** is right -- that `sweep` is what the shipped loop actually iterates, which is
//   structural (`vk_step` calls it) and pinned end-to-end by
//   `the_model_step_lands_on_a_root_of_the_residual_the_jacobian_differentiates`.
//
// A finite difference cannot see the second: it would happily verify the derivative of the wrong
// map. Nor can it see `couple_factor`, which `sweep` and `jacobian_vector` **share by
// construction** -- that is how the free edge's extra `h^2` (plan §7 trap 6) is made structural
// rather than remembered in two places.
//
// What the four bars actually catch was measured, by mutating the Jacobian six ways and running
// them (plan §10.3):
//
// * the **finite-difference** bar caught all six, and is the workhorse;
// * the **margin** floor caught the two that pull `J` back toward `I` -- the stub, which makes it
//   `I` exactly, and the flipped sign, which lets the two terms cancel -- i.e. exactly the two a
//   badly chosen fixture lets through;
// * the **linearity** identity caught nothing on its own merits. Its one hit is the `h^2` mutant,
//   which it saw only because that map *diverges* and the point overflows. Every other mutant --
//   including the one that scales the residual by 1.01 -- leaves `J` exactly linear in `d`, which
//   is what a linear operator assembled from bilinear pieces does whichever piece you get wrong.
//   It is a structural check against a `J` that is not an operator at all, not a transcription
//   check, and it should be read that way;
// * the **root** check has the same one hit, for the same reason. The 1.01 mutant *does* change
//   the map and it stayed quiet, because 1% of a residual already at 7e-14 is under any bar this
//   check could carry. What it guards is a context that does not match the model's -- an error of
//   a factor, not of a percent.
//
// The `couple_factor` mutant is worth stating precisely rather than counting: removing the free
// edge's `h^2` was caught by all four, but as **the physics blowing up** -- the map diverges and
// the point overflows -- never as a derivative disagreement. A finite difference is structurally
// blind to `couple_factor` because both sides read the same field. That sharing *is* the guard
// for trap 6. The bar is not, and a Part 2 that believed otherwise would be relying on nothing.

/// A struck plate: `amp` thicknesses tall, a Gaussian `width` wide, its centre `off` from the mid.
fn jac_plate(side: f64, n: i64, amp: f64, width: f64, off: f64, boundary: Boundary) -> VkPlate {
    let mut vk = VkPlate::new(
        VkParams::new(&VkSpec {
            lx: side,
            ly: side,
            young: 2.0e11,
            thickness: 1e-3,
            nu: 0.3,
            rho: 7860.0,
            fs: 48_000.0,
            n,
            boundary: Some(boundary),
            couple_max_iter: 400,
            ..VkSpec::default()
        })
        .expect("a valid plate"),
    );
    let p = &vk.p.lin;
    let (cx, cy) = (0.5 * p.lx + off, 0.5 * p.ly + off);
    let a = amp * vk.p.thickness;
    let u0: Vec<f64> = p
        .mask
        .flags()
        .iter()
        .enumerate()
        .filter(|(_, &alive)| alive)
        .map(|(idx, _)| {
            let (dx, dy) = (p.x[idx] - cx, p.y[idx] - cy);
            a * (-((dx * dx + dy * dy) / (width * width))).exp()
        })
        .collect();
    let zero = vec![0.0; p.n_live];
    vk.set_state(&u0, &zero).expect("the Airy solve factors");
    vk
}

fn l2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// A deterministic unit direction — xorshift64, so it is the same on every platform.
///
/// Rough on purpose: a direction shaped like the strike would excite only the modes the strike
/// already excites, and the bracket's mixed derivatives are exactly what such a direction is
/// blindest to.
fn jac_direction(n: usize, salt: u64) -> Vec<f64> {
    let mut s = 0x2545_f491_4f6c_dd1d ^ salt.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let mut d: Vec<f64> = (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64) / 9_007_199_254_740_992.0 - 0.5
        })
        .collect();
    let norm = l2(&d);
    for v in d.iter_mut() {
        *v /= norm;
    }
    d
}

/// `(label, side, n, amp/thickness, strike width, strike offset, boundary, margin is asserted)`.
///
/// The first five are plan §9.4's converging fixtures and their free-edge counterparts: loud
/// enough that `J` is a long way from `I`. The sixth is `w = e`, where it is **not** — kept and
/// asserted from the other side, because that is the amplitude at which this whole family of
/// checks starts to go vacuous, and a reader should be able to see where.
///
/// A **centred** strike on a 40 cm plate is numerically zero at the rim (`exp(-44)`), so the
/// supported and free arithmetic coincide there to the last digit — measured, not assumed. The
/// off-centre free case is the one that puts ~38% of the peak on a free edge and genuinely
/// exercises those rows.
type JacCase = (&'static str, f64, i64, f64, f64, f64, Boundary, bool);
const JAC_CASES: [JacCase; 6] = [
    (
        "40cm supported, 3cm strike, 6e",
        0.4,
        20,
        6.0,
        0.03,
        0.0,
        Boundary::Supported,
        true,
    ),
    (
        "40cm supported, 8cm strike, 16e",
        0.4,
        20,
        16.0,
        0.08,
        0.0,
        Boundary::Supported,
        true,
    ),
    (
        "16cm supported, 3.2cm strike, 6e",
        0.16,
        8,
        6.0,
        0.032,
        0.0,
        Boundary::Supported,
        true,
    ),
    (
        "40cm free, 8cm strike, 16e",
        0.4,
        20,
        16.0,
        0.08,
        0.0,
        Boundary::Free,
        true,
    ),
    (
        "40cm free, 8cm strike off-centre, 16e",
        0.4,
        20,
        16.0,
        0.08,
        0.12,
        Boundary::Free,
        true,
    ),
    (
        "40cm supported, 10cm strike, 1e",
        0.4,
        16,
        1.0,
        0.1,
        0.0,
        Boundary::Supported,
        false,
    ),
];

/// The two points a Newton step would ever be taken from: the loop's seed and where Picard lands.
fn jac_points(vk: &VkPlate, ctx: &VkCoupledStep) -> [(&'static str, Vec<f64>); 2] {
    let seed = VkCoupledStep::seed(&vk.u, &vk.u_prev);
    let mut w = seed.clone();
    for _ in 0..400 {
        let (next, _) = ctx.sweep(&w).expect("the solves succeed");
        let d: Vec<f64> = (0..w.len()).map(|i| next[i] - w[i]).collect();
        let r = l2(&d) / l2(&next).max(1e-30);
        w = next;
        if r <= vk.p.couple_tol {
            break;
        }
    }
    [("seed", seed), ("converged", w)]
}

/// The finite-difference step, **relative** to `||w||`.
///
/// Measured, not guessed. `G` is exactly cubic, so a central difference has a pure `eps^2`
/// truncation term and the error curve is a clean V: at this fixture family it falls
/// `2e-4 -> 2e-6 -> 2e-8 -> 2e-10` from `1e-1` to `1e-4` and bottoms at `~2e-11` here before
/// rounding lifts it again. Two decades either side still sit three orders under the bar.
const JAC_FD_STEP: f64 = 1e-5;

/// Plan §5 Part 1's gate. Measured across the six fixtures and both linearisation points:
/// **9.5e-12 to 1.1e-10**. The three orders of gap are deliberate headroom of the same kind as the
/// `1e-10` energy bar -- this number has to survive a different machine, and a sparse
/// back-substitution is exactly the kind of arithmetic that moves when it does.
const JAC_FD_BAR: f64 = 1e-7;

/// A floor on `||J d - d|| / ||d||`, without which the finite-difference bar asserts nothing.
///
/// At zero amplitude `J = I` **exactly** (plan §1), so a `jacobian_vector` that dropped the
/// coupling term and returned `d` unchanged is a *good approximation* at small amplitude and
/// passes a finite-difference check there. This is not hypothetical: taking `F'` at `w_bar`
/// instead of the raw `W` -- a real and plausible transcription error -- was measured at 1e-4 to
/// 1e-2 on the loud fixtures and at **6.3e-8 at margin 5.9e-3**, i.e. green under `JAC_FD_BAR`.
///
/// So the floor is placed between two measurements rather than chosen round: the gated fixtures
/// sit at **0.024 to 0.124**, and the bar demonstrably goes soft by **0.0059**.
const JAC_MARGIN_FLOOR: f64 = 1e-2;

#[test]
fn the_jacobian_matches_a_finite_difference_of_the_residual() {
    for (label, side, n, amp, width, off, boundary, _) in JAC_CASES {
        let vk = jac_plate(side, n, amp, width, off, boundary);
        let ctx = VkCoupledStep::new(&vk.u, &vk.u_prev, &vk.f_prev, None, &vk.p);
        for (point, w) in jac_points(&vk, &ctx) {
            let av = ctx.averages(&w).expect("the solves succeed");
            let d = jac_direction(w.len(), 1);
            let jd = ctx.jacobian_vector(&av, &d).expect("the solves succeed");

            // Central difference of the residual in the same direction, at the same frozen
            // context: `rhs_lin`, `w_prev_full`, `f_prev` and `couple_factor` are functions of
            // time-`n` state, not of `w`. Rebuilding the context from a perturbed `u` would
            // differentiate a different function.
            let eps = JAC_FD_STEP * l2(&w);
            let plus: Vec<f64> = (0..w.len()).map(|i| w[i] + eps * d[i]).collect();
            let minus: Vec<f64> = (0..w.len()).map(|i| w[i] - eps * d[i]).collect();
            let (g_plus, _) = ctx.residual(&plus).expect("the solves succeed");
            let (g_minus, _) = ctx.residual(&minus).expect("the solves succeed");
            let fd: Vec<f64> = (0..w.len())
                .map(|i| (g_plus[i] - g_minus[i]) / (2.0 * eps))
                .collect();

            let err: Vec<f64> = (0..w.len()).map(|i| fd[i] - jd[i]).collect();
            let rel = l2(&err) / l2(&jd);
            assert!(
                rel < JAC_FD_BAR,
                "{label} at the {point} point: relative disagreement {rel:.3e}"
            );
        }
    }
}

#[test]
fn the_jacobian_is_linear_in_its_direction_to_rounding() {
    for (label, side, n, amp, width, off, boundary, _) in JAC_CASES {
        let vk = jac_plate(side, n, amp, width, off, boundary);
        let ctx = VkCoupledStep::new(&vk.u, &vk.u_prev, &vk.f_prev, None, &vk.p);
        for (point, w) in jac_points(&vk, &ctx) {
            let av = ctx.averages(&w).expect("the solves succeed");
            let d1 = jac_direction(w.len(), 1);
            let d2 = jac_direction(w.len(), 2);
            let sum: Vec<f64> = (0..w.len()).map(|i| d1[i] + d2[i]).collect();
            let j1 = ctx.jacobian_vector(&av, &d1).expect("the solves succeed");
            let j2 = ctx.jacobian_vector(&av, &d2).expect("the solves succeed");
            let js = ctx.jacobian_vector(&av, &sum).expect("the solves succeed");
            let defect: Vec<f64> = (0..w.len()).map(|i| js[i] - j1[i] - j2[i]).collect();
            let rel = l2(&defect) / l2(&js);
            // Not bitwise: two sparse back-substitutions are floating-point evaluations of a
            // linear map, and are not exactly additive. Measured ~1e-16 here.
            assert!(
                rel < 1e-13,
                "{label} at the {point} point: defect {rel:.3e}"
            );
        }
    }
}

#[test]
fn the_jacobian_is_far_from_the_identity_where_it_is_asserted() {
    for (label, side, n, amp, width, off, boundary, gated) in JAC_CASES {
        let vk = jac_plate(side, n, amp, width, off, boundary);
        let ctx = VkCoupledStep::new(&vk.u, &vk.u_prev, &vk.f_prev, None, &vk.p);
        for (point, w) in jac_points(&vk, &ctx) {
            let av = ctx.averages(&w).expect("the solves succeed");
            let d = jac_direction(w.len(), 1);
            let jd = ctx.jacobian_vector(&av, &d).expect("the solves succeed");
            let dev: Vec<f64> = (0..w.len()).map(|i| jd[i] - d[i]).collect();
            let margin = l2(&dev) / l2(&d);
            if gated {
                assert!(
                    margin > JAC_MARGIN_FLOOR,
                    "{label} at the {point} point: margin {margin:.3e} — the finite-difference \
                     bar is nearly vacuous here"
                );
            } else {
                // The other side of the same claim: at one thickness of amplitude the coupling
                // is four orders down, `J` is the identity to within 2e-4, and no
                // finite-difference check on this fixture would notice a Jacobian that said so.
                assert!(
                    margin < JAC_MARGIN_FLOOR,
                    "{label} at the {point} point: margin {margin:.3e} — this fixture is meant \
                     to be the one where the guard goes soft"
                );
            }
        }
    }
}

#[test]
fn the_model_step_lands_on_a_root_of_the_residual_the_jacobian_differentiates() {
    // The claim a finite difference cannot make: that `VkCoupledStep` is the *same* map
    // `VkPlate::step` iterates -- same `rhs_lin`, same `f_prev` slot, same `couple_factor`. Build
    // the context from the pre-step state, let the model take its own step, and the displacement
    // it returns must be a root of that context's residual.
    for (label, side, n, amp, width, off, boundary, _) in JAC_CASES {
        let mut vk = jac_plate(side, n, amp, width, off, boundary);
        let ctx_u = vk.u.clone();
        let ctx_u_prev = vk.u_prev.clone();
        let ctx_f_prev = vk.f_prev.clone();
        vk.step(None).expect("the solves succeed");
        assert!(vk.converged, "{label}: the baseline step must converge");
        let ctx = VkCoupledStep::new(&ctx_u, &ctx_u_prev, &ctx_f_prev, None, &vk.p);
        let (g, _) = ctx.residual(&vk.u).expect("the solves succeed");
        let rel = l2(&g) / l2(&vk.u);
        // Ten times `couple_tol`, not `couple_tol` itself. Picard's exit test is on the
        // *increment*, so `G` at the returned point is a contraction factor smaller and lands at
        // 3.4e-14 to 7.5e-14 against a 1e-13 tolerance -- 1.3x of headroom, which is not enough
        // for a quantity produced by two sparse back-substitutions on an unknown machine. It
        // costs no discriminating power: a context that did not match the model's would be wrong
        // by a factor, not by a percent.
        assert!(
            rel <= 10.0 * vk.p.couple_tol,
            "{label}: the step's own answer leaves residual {rel:.3e}"
        );
    }
}

// -- Newton behind `couple_method` (plan §5 Part 2) ----------------------------------------------
//
// The gate is that Newton reaches the *same root* on every fixture where Picard reaches one, and
// meets the energy bar on the converged path. Both halves are asserted below, and the first is
// stated as the plan's §5 says it must be -- as a claim about `w`, not about both methods
// reporting `converged`. `G` is cubic, so "both converged" leaves open that they converged to
// different roots, and only comparing the displacements closes it.
//
// What is deliberately *not* here: any claim that Newton moves the wall. That is Part 3, it is
// measured over a grid rather than at six points, and Part 0 already drew the baseline it will be
// measured against so that a cap-50 Picard cannot flatter it.

/// [`jac_plate`] with the coupling iteration chosen -- the same plate, twice, is the whole gate.
fn jac_plate_m(
    side: f64,
    n: i64,
    amp: f64,
    width: f64,
    off: f64,
    boundary: Boundary,
    method: CoupleMethod,
) -> VkPlate {
    let mut vk = VkPlate::new(
        VkParams::new(&VkSpec {
            lx: side,
            ly: side,
            young: 2.0e11,
            thickness: 1e-3,
            nu: 0.3,
            rho: 7860.0,
            fs: 48_000.0,
            n,
            boundary: Some(boundary),
            couple_max_iter: 400,
            couple_method: Some(method),
            ..VkSpec::default()
        })
        .expect("a valid plate"),
    );
    let p = &vk.p.lin;
    let (cx, cy) = (0.5 * p.lx + off, 0.5 * p.ly + off);
    let a = amp * vk.p.thickness;
    let u0: Vec<f64> = p
        .mask
        .flags()
        .iter()
        .enumerate()
        .filter(|(_, &alive)| alive)
        .map(|(idx, _)| {
            let (dx, dy) = (p.x[idx] - cx, p.y[idx] - cy);
            a * (-((dx * dx + dy * dy) / (width * width))).exp()
        })
        .collect();
    let zero = vec![0.0; p.n_live];
    vk.set_state(&u0, &zero).expect("the Airy solve factors");
    vk
}

/// Both methods converge to the same displacement, on the six fixtures Part 1 asserted `J` on.
///
/// The bar is not `couple_tol` itself. Each method stops when `||G||/||w||` is under `1e-13`, and
/// near a root the *displacement* error is that residual divided by the smallest singular value of
/// `J` -- so two solves that both meet a residual bar can sit further apart than the bar in `w`.
/// The number below is measured; the claim it carries is that the two land on one root rather than
/// on two, which at these amplitudes would be a gap of percent, not of parts in a trillion.
#[test]
fn newton_and_picard_land_on_the_same_root() {
    for (label, side, n, amp, width, off, boundary, _) in JAC_CASES {
        let mut pic = jac_plate_m(side, n, amp, width, off, boundary, CoupleMethod::Picard);
        let mut new = jac_plate_m(side, n, amp, width, off, boundary, CoupleMethod::Newton);
        // Identical initial conditions, so any difference downstream is the iteration's.
        assert_eq!(pic.u, new.u, "{label}: the two plates start differently");
        assert_eq!(
            pic.f, new.f,
            "{label}: the two stress caches start differently"
        );

        pic.step(None).expect("the solves succeed");
        new.step(None).expect("the solves succeed");
        assert!(
            pic.converged,
            "{label}: Picard must reach a root for this gate to mean anything"
        );
        assert!(new.converged, "{label}: Newton did not converge");

        let gap: Vec<f64> = (0..pic.u.len()).map(|i| new.u[i] - pic.u[i]).collect();
        let rel = l2(&gap) / l2(&pic.u);
        assert!(
            rel < NEWTON_ROOT_BAR,
            "{label}: the two roots differ by {rel:.3e}"
        );
        // And the energy of the two states agrees, which is the reading that actually matters.
        let de = (new.energy() / pic.energy() - 1.0).abs();
        assert!(de < 1e-12, "{label}: energies differ by {de:.3e}");
    }
}

/// Measured across the six fixtures and both linearisation points. See the test above on why this
/// is not `couple_tol`.
const NEWTON_ROOT_BAR: f64 = 1e-8;

/// The project's acceptance contract, on the Newton path — plan §6's first bar.
///
/// A root of the discrete-gradient equation conserves exactly whichever iteration found it, so
/// this asserts something the theory already promises. It is here because the theory promises it
/// *of a root*, and the thing worth checking is that what Newton returns is one — including that
/// the `F` it hands back (the stress function at the accepted iterate, not at the sweep's incoming
/// one) leaves the energy bookkeeping consistent.
#[test]
fn a_newton_stepped_plate_conserves_its_total_energy() {
    for boundary in [Boundary::Supported, Boundary::Free] {
        let mut vk = VkPlate::new(
            VkParams::new(&VkSpec {
                lx: 0.4,
                ly: 0.4,
                young: 2.0e11,
                thickness: 1e-3,
                nu: 0.3,
                rho: 7860.0,
                fs: 48_000.0,
                n: 16,
                boundary: Some(boundary),
                couple_method: Some(CoupleMethod::Newton),
                ..VkSpec::default()
            })
            .expect("a valid plate"),
        );
        let u0 = bump(&vk.p.lin, 3.0 * vk.p.thickness);
        let zero = vec![0.0; vk.p.lin.n_live];
        vk.set_state(&u0, &zero).expect("the Airy solve factors");
        assert!(
            vk.membrane_energy() > 0.0,
            "the coupling must be doing work"
        );
        let e0 = vk.energy();
        let mut worst = 0.0f64;
        for step in 0..300 {
            vk.step(None).expect("the solves succeed");
            // On the converged path only -- plan §4. An under-relaxed iterate is not a root and
            // the identity says nothing about it, so a drift bar that quietly averaged over
            // unconverged steps would be asserting a different claim than it reads as.
            assert!(
                vk.converged,
                "{boundary:?} step {step}: Newton did not converge, so the bar below would be \
                 measuring something else"
            );
            worst = worst.max((vk.energy() / e0 - 1.0).abs());
        }
        assert!(worst < 1e-10, "{boundary:?}: drift {worst:.3e}");
    }
}

/// `nonlinear = false` never reaches the iteration at all, whichever one is selected.
///
/// Structural rather than measured: [`vk_step`]'s linear early return is *before* the method
/// branch, so the two paths are the same line of code. Asserted anyway, because "before" is a
/// property of the source that a later edit can quietly reverse — and plan §6's second bar is that
/// the linear plate stays bit-identical to model #5.
#[test]
fn the_linear_path_is_bit_identical_whichever_method_is_selected() {
    for boundary in [Boundary::Supported, Boundary::Free] {
        let make = |method: CoupleMethod| {
            VkPlate::new(
                VkParams::new(&VkSpec {
                    lx: 0.4,
                    ly: 0.4,
                    young: 2.0e11,
                    thickness: 1e-3,
                    nu: 0.3,
                    rho: 7860.0,
                    fs: 48_000.0,
                    n: 12,
                    boundary: Some(boundary),
                    nonlinear: false,
                    couple_method: Some(method),
                    ..VkSpec::default()
                })
                .expect("a valid plate"),
            )
        };
        let mut pic = make(CoupleMethod::Picard);
        let mut new = make(CoupleMethod::Newton);
        let mut aut = make(CoupleMethod::Auto);
        let u0 = bump(&pic.p.lin, 1e-3);
        let zero = vec![0.0; pic.p.lin.n_live];
        pic.set_state(&u0, &zero).expect("the Airy solve factors");
        new.set_state(&u0, &zero).expect("the Airy solve factors");
        aut.set_state(&u0, &zero).expect("the Airy solve factors");
        for step in 0..40 {
            pic.step(None).expect("the solves succeed");
            new.step(None).expect("the solves succeed");
            aut.step(None).expect("the solves succeed");
            assert_eq!(
                pic.u, new.u,
                "{boundary:?} step {step}: the linear path forked"
            );
            assert_eq!(
                pic.u, aut.u,
                "{boundary:?} step {step}: the linear path forked under `auto`"
            );
        }
        // And the linear path's cost is one solve, on all three -- it is one back-substitution,
        // and `n_solves` is only worth having if it says so. `auto` never reaches its fallback
        // here, which is the same statement made on the counter.
        assert_eq!(pic.n_solves, 1);
        assert_eq!(new.n_solves, 1);
        assert_eq!(aut.n_solves, 1);
        assert_eq!(aut.n_fallbacks, 0);
    }
}

/// The cost each method reports is its own arithmetic, and the two are on one scale.
///
/// `n_iters` is not comparable across the methods and `n_solves` is — one Picard sweep is two
/// back-substitutions, one Newton iteration is two plus two per Krylov product and two per
/// line-search trial. This checks the accounting adds up on both sides, which is what Part 3's
/// convergence map will be drawn against.
#[test]
fn the_solve_count_is_the_cost_both_methods_can_be_read_on() {
    for (label, side, n, amp, width, off, boundary, _) in JAC_CASES {
        let mut pic = jac_plate_m(side, n, amp, width, off, boundary, CoupleMethod::Picard);
        pic.step(None).expect("the solves succeed");
        assert_eq!(
            pic.n_solves,
            2 * pic.n_iters,
            "{label}: a Picard sweep is exactly one Airy solve and one theta-scheme solve"
        );

        // The Newton side, read off the report rather than the model, so the parts are visible.
        let vk = jac_plate_m(side, n, amp, width, off, boundary, CoupleMethod::Newton);
        let ctx = VkCoupledStep::new(&vk.u, &vk.u_prev, &vk.f_prev, None, &vk.p);
        let r =
            vk_newton(&ctx, VkCoupledStep::seed(&vk.u, &vk.u_prev)).expect("the solves succeed");
        assert!(r.converged, "{label}: Newton did not converge");
        // 2 for the seed's residual, 2 per Krylov product, 2 per line-search trial. Every
        // iteration takes at least one trial, and `n_line_search` counts only the *rejected* ones.
        let trials = r.n_iters + r.n_line_search;
        assert_eq!(
            r.n_solves,
            2 + 2 * r.gmres_products + 2 * trials,
            "{label}: {} solves does not decompose into 1 seed + {} products + {trials} trials",
            r.n_solves,
            r.gmres_products
        );
        assert_eq!(
            r.gmres_stalls, 0,
            "{label}: an inner solve hit its product cap"
        );
    }
}

/// Plan §7 trap 2: "a line search that never fires is untested code" — so record whether it does.
///
/// It does not, on any of Part 1's six fixtures, and this test says so rather than leaving the
/// question open. That is not an argument for deleting it: those six are all fixtures where Picard
/// itself converges, which is the region where the residual is smooth and a full Newton step is
/// the right one. The second half of this test is a fixture where the seed is deliberately far
/// from the root, and there the search *does* fire — which is what makes the code exercised rather
/// than merely present.
#[test]
fn the_line_search_is_recorded_rather_than_assumed() {
    let mut fired_on_a_gate_fixture = 0usize;
    for (label, side, n, amp, width, off, boundary, _) in JAC_CASES {
        let vk = jac_plate_m(side, n, amp, width, off, boundary, CoupleMethod::Newton);
        let ctx = VkCoupledStep::new(&vk.u, &vk.u_prev, &vk.f_prev, None, &vk.p);
        let r =
            vk_newton(&ctx, VkCoupledStep::seed(&vk.u, &vk.u_prev)).expect("the solves succeed");
        assert!(r.converged, "{label}: Newton did not converge");
        fired_on_a_gate_fixture += r.n_line_search;
    }
    assert_eq!(
        fired_on_a_gate_fixture, 0,
        "the line search now fires on a gate fixture; that is a finding, not a failure -- \
         re-measure and rewrite this test's claim rather than raising a number"
    );

    // The same plate, seeded a long way from its root. How far is *measured*, not assumed, and
    // the answer is further than one would guess: at 1x, 10x and 40x the physical seed the full
    // Newton step is accepted every single time (0 halvings), and the search first fires at 200x,
    // where it takes 6. That is worth knowing on its own -- it says the residual stays convex
    // enough for an undamped step across two orders of magnitude of nonsense, and it is why the
    // first half of this test reads as a finding rather than as a gap.
    let vk = jac_plate_m(
        0.4,
        20,
        16.0,
        0.08,
        0.0,
        Boundary::Supported,
        CoupleMethod::Newton,
    );
    let ctx = VkCoupledStep::new(&vk.u, &vk.u_prev, &vk.f_prev, None, &vk.p);
    let seed = VkCoupledStep::seed(&vk.u, &vk.u_prev);
    let quiet: Vec<f64> = seed.iter().map(|v| 40.0 * v).collect();
    let r = vk_newton(&ctx, quiet).expect("the solves succeed");
    assert_eq!(
        r.n_line_search, 0,
        "40x the seed now needs damping; re-measure where the search starts firing"
    );
    let far: Vec<f64> = seed.iter().map(|v| 200.0 * v).collect();
    let r = vk_newton(&ctx, far).expect("the solves succeed");
    assert!(
        r.n_line_search > 0,
        "even from 200x the seed the full step was accepted every time, so the backtracking \
         branch is unexercised"
    );
    // And it recovers: the point of a line search is not that it fires but that the step it
    // salvages still reaches the root. (The inner GMRES *does* hit its product cap here -- 12
    // stalls -- which is the other half of the same picture, and is recorded rather than fixed:
    // an inexact correction handed to a line search is exactly the designed behaviour.)
    assert!(
        r.converged,
        "damped Newton did not recover, residual {:.3e}",
        r.last_residual
    );
    assert!(
        r.gmres_stalls > 0,
        "a 200x seed no longer stresses the inner solve -- if NEWTON_GMRES_MAX_PRODUCTS was          raised, this failing is the improvement showing up, and the fact to re-record is where          the cap now bites; the load-bearing claim on this run is the halving above"
    );
}

/// Newton converges on all three fixtures where Picard cannot — the measurement, stated narrowly.
///
/// Plan §9.4 established that these three are decided on step **zero**: Picard is `expansive` (or,
/// for the small plate, still at a relative residual of 0.57 after 400 sweeps) from the very first
/// step, at every cap. Newton reaches `couple_tol` on each in four or five iterations.
///
/// **What this test claims, and what it does not.** It claims a property of the *iteration*: there
/// is a root there, and Newton finds it while Picard's map is expansive. It does not claim the
/// territory. Plan §7 trap 3 — "a root that is not a plate" — is answered here only as far as this
/// project's primary detector goes: the run conserves energy to `1e-12` over 300 steps with every
/// step converged, which is what a root of the discrete-gradient equation must do. The other half
/// of trap 3, a comparison against a refined-`k` reference, is Part 3's, and until it is done the
/// honest reading of this test is "the solver got there", not "the plate is now audible".
///
/// Part 3 draws this over a grid; six points is not a map, and the boundary's *position* is not
/// what is asserted here.
#[test]
fn newton_converges_where_picard_does_not() {
    // (label, side, N, amplitude/thickness, strike width, fs, Picard's measured verdict at cap 400)
    let walls: [(&str, f64, i64, f64, f64, f64, CoupleOutcome); 3] = [
        (
            "12cm, 2.4cm strike, 6e",
            0.12,
            20,
            6.0,
            0.024,
            48_000.0,
            CoupleOutcome::Expansive,
        ),
        (
            "40cm, 8cm strike, 20e",
            0.4,
            20,
            20.0,
            0.08,
            48_000.0,
            CoupleOutcome::Expansive,
        ),
        (
            "40cm, 8cm strike, 12e, 24kHz",
            0.4,
            20,
            12.0,
            0.08,
            24_000.0,
            CoupleOutcome::Expansive,
        ),
    ];
    for (label, side, n, amp, width, fs, want) in walls {
        let make = |method: CoupleMethod| {
            let mut vk = VkPlate::new(
                VkParams::new(&VkSpec {
                    lx: side,
                    ly: side,
                    young: 2.0e11,
                    thickness: 1e-3,
                    nu: 0.3,
                    // 7800, not Part 1's 7860: these are plan §9.4's fixtures exactly, so that
                    // the Picard column below is the same measurement Part 0 published.
                    rho: 7800.0,
                    fs,
                    n,
                    boundary: Some(Boundary::Supported),
                    couple_max_iter: 400,
                    couple_method: Some(method),
                    ..VkSpec::default()
                })
                .expect("a valid plate"),
            );
            let p = &vk.p.lin;
            let (cx, cy) = (0.5 * p.lx, 0.5 * p.ly);
            let a = amp * vk.p.thickness;
            let u0: Vec<f64> = p
                .mask
                .flags()
                .iter()
                .enumerate()
                .filter(|(_, &alive)| alive)
                .map(|(idx, _)| {
                    let (dx, dy) = (p.x[idx] - cx, p.y[idx] - cy);
                    a * (-((dx * dx + dy * dy) / (width * width))).exp()
                })
                .collect();
            let zero = vec![0.0; p.n_live];
            vk.set_state(&u0, &zero).expect("the Airy solve factors");
            vk
        };

        // Picard, one step from rest, with a cap eight times the shipped default.
        let mut pic = make(CoupleMethod::Picard);
        pic.step(None).expect("the solves succeed");
        assert!(
            !pic.converged,
            "{label}: Picard converged, so this is not a wall fixture"
        );
        assert_eq!(pic.outcome(), want, "{label}: Picard's verdict moved");

        // Newton, the same step, and then a whole run to see whether the root behaves.
        let mut new = make(CoupleMethod::Newton);
        new.step(None).expect("the solves succeed");
        assert!(new.converged, "{label}: Newton did not converge either");
        assert!(
            new.n_iters <= 8,
            "{label}: {} Newton iterations",
            new.n_iters
        );

        let e0 = new.energy();
        let mut worst = 0.0f64;
        for step in 1..300 {
            new.step(None).expect("the solves succeed");
            assert!(new.converged, "{label}: step {step} did not converge");
            worst = worst.max((new.energy() / e0 - 1.0).abs());
        }
        // Measured 6.3e-13, 1.4e-12 and 9.1e-13. The bar is the project's own 1e-10.
        assert!(worst < 1e-10, "{label}: drift {worst:.3e}");
    }
}

/// The default is `Auto`, at both levels, and an unparseable spelling is refused.
///
/// Plan §6's third bar said Picard, and §15 moved it: what that bar was protecting — "the whole
/// existing suite exercises today's code unchanged" — is now protected by
/// [`auto_is_bit_identical_to_picard_wherever_the_sweeps_converge`] instead, which is a stronger
/// statement than a default's name because it is about the numbers rather than about the spelling.
#[test]
fn the_method_defaults_to_auto_and_a_bad_spelling_is_refused() {
    assert_eq!(VkSpec::default().couple_method, Some(CoupleMethod::Auto));
    let p = VkParams::new(&VkSpec {
        lx: 0.4,
        ly: 0.4,
        young: 2.0e11,
        thickness: 1e-3,
        nu: 0.3,
        rho: 7860.0,
        fs: 48_000.0,
        n: 8,
        ..VkSpec::default()
    })
    .expect("a valid plate");
    assert_eq!(p.couple_method, CoupleMethod::Auto);
    assert_eq!(CoupleMethod::Picard.label(), "picard");
    assert_eq!(CoupleMethod::Newton.label(), "newton");
    assert_eq!(CoupleMethod::Auto.label(), "auto");

    // `None` is how the binding says "the caller passed something that is not a method name" --
    // the same shape as `boundary`, and refused the same way rather than defaulted.
    let err = VkParams::new(&VkSpec {
        lx: 0.4,
        ly: 0.4,
        young: 2.0e11,
        thickness: 1e-3,
        nu: 0.3,
        rho: 7860.0,
        fs: 48_000.0,
        n: 8,
        couple_method: None,
        ..VkSpec::default()
    })
    .expect_err("an unparseable method must be refused");
    assert_eq!(err, VkParamError::BadMethod);
}

// -- `Auto`: the sweeps first, Newton only where they fail (plan §5 Part 6) -----------------------
//
// §13.3 is why this exists rather than a flipped default: neither method dominates. Picard is
// cheaper than Newton in six of that table's eleven cells at musical amplitude, and Newton is up
// to 68× cheaper at the wall, so the default has to be able to be both. Three things make that
// safe, and all three are asserted below:
//
// 1. where the sweeps converge, `Auto` *is* Picard — bit for bit, not merely to a tolerance;
// 2. where they do not, the answer is Newton's own — bit for bit, which is the only way to show
//    the rescue re-seeded from `2 w^n - w^{n-1}` and not from the sweeps' exit iterate, where an
//    expansive run parks its overflow;
// 3. the abandoned sweeps are still counted, so `n_solves` does not flatter the new default.

/// A supported 40 cm steel square struck off-centre — plan §13.5's cell, where the outcome is
/// *not* monotone in amplitude and Picard's failures are therefore found rather than arranged.
fn auto_case(amp: f64, off: f64, cap: i64, method: CoupleMethod) -> VkPlate {
    let mut vk = VkPlate::new(
        VkParams::new(&VkSpec {
            lx: 0.4,
            ly: 0.4,
            young: 2.0e11,
            thickness: 1e-3,
            nu: 0.3,
            rho: 7860.0,
            fs: 48_000.0,
            n: 20,
            boundary: Some(Boundary::Supported),
            couple_max_iter: cap,
            couple_method: Some(method),
            ..VkSpec::default()
        })
        .expect("a valid plate"),
    );
    let p = &vk.p.lin;
    let (cx, cy) = (0.5 * p.lx + off, 0.5 * p.ly + off);
    let a = amp * vk.p.thickness;
    let u0: Vec<f64> = p
        .mask
        .flags()
        .iter()
        .enumerate()
        .filter(|(_, &alive)| alive)
        .map(|(idx, _)| {
            let (dx, dy) = (p.x[idx] - cx, p.y[idx] - cy);
            a * (-((dx * dx + dy * dy) / (0.03 * 0.03))).exp()
        })
        .collect();
    let zero = vec![0.0; p.n_live];
    vk.set_state(&u0, &zero).expect("the Airy solve factors");
    vk
}

/// Every number the suite recorded under the old default is the number it still gets.
///
/// The claim the moved default stands on, and it is stated on the *state* rather than on the
/// verdict: `u` and the stress cache compared with `assert_eq!` on every one of twenty steps,
/// across the six fixtures Part 1 asserted `J` on. A tolerance here would let a rescue that fired
/// where it should not have hide inside it — which is exactly the regression this guards.
///
/// The diagnostics are compared too, because they are what the suite's other bars read: a step
/// that agreed in `w` while reporting a different sweep count would still have moved a number.
#[test]
fn auto_is_bit_identical_to_picard_wherever_the_sweeps_converge() {
    for (label, side, n, amp, width, off, boundary, _) in JAC_CASES {
        let mut pic = jac_plate_m(side, n, amp, width, off, boundary, CoupleMethod::Picard);
        let mut aut = jac_plate_m(side, n, amp, width, off, boundary, CoupleMethod::Auto);
        assert_eq!(pic.u, aut.u, "{label}: the two plates start differently");

        for step in 0..20 {
            pic.step(None).expect("the solves succeed");
            aut.step(None).expect("the solves succeed");
            assert!(
                pic.converged,
                "{label} step {step}: Picard must converge for this bar to say anything"
            );
            assert_eq!(pic.u, aut.u, "{label} step {step}: `auto` moved the state");
            assert_eq!(pic.f, aut.f, "{label} step {step}: `auto` moved the cache");
            assert_eq!(
                (pic.n_iters, pic.converged, pic.n_solves),
                (aut.n_iters, aut.converged, aut.n_solves),
                "{label} step {step}: `auto` moved a read-out"
            );
            assert_eq!(pic.last_residual, aut.last_residual);
            assert_eq!(
                aut.n_fallbacks, 0,
                "{label} step {step}: nothing to rescue, so nothing may be spent on Newton"
            );
        }
    }
}

/// Where the sweeps blow up, `auto` returns Newton's own root — and the seed is what proves it.
///
/// **The bar is `assert_eq!` on `u`, deliberately.** Picard's expansive exit leaves an iterate that
/// is enormous or non-finite; a Newton seeded from *that* returns a NaN and reports
/// `converged: false`, which would make `auto` strictly worse than either method alone with
/// nothing in the read-outs to say so. Equality with a solo Newton run is the statement that the
/// rescue started from `2 w^n - w^{n-1}` instead. A relative tolerance could not make it: two
/// Newton runs from *different* seeds land on the same root to about `1e-13`, so any bar loose
/// enough to be safe against that would also pass on the bug.
#[test]
fn auto_rescues_an_expansive_sweep_from_the_sweeps_own_seed() {
    let (amp, off, cap) = (10.0, 0.12, 400);
    let mut pic = auto_case(amp, off, cap, CoupleMethod::Picard);
    let mut new = auto_case(amp, off, cap, CoupleMethod::Newton);
    let mut aut = auto_case(amp, off, cap, CoupleMethod::Auto);

    pic.step(None).expect("the solves succeed");
    assert_eq!(
        pic.outcome(),
        CoupleOutcome::Expansive,
        "the fixture must be one the sweeps genuinely fail on, or this test is vacuous"
    );

    new.step(None).expect("the solves succeed");
    assert!(
        new.converged,
        "Newton must reach a root for the rescue to have one to reach"
    );

    aut.step(None).expect("the solves succeed");
    assert!(
        aut.converged,
        "`auto` did not converge where solo Newton did"
    );
    assert_eq!(aut.n_fallbacks, 1, "the rescue did not fire");
    assert_eq!(
        aut.u, new.u,
        "the rescue did not re-seed from `2 w^n - w^{{n-1}}`"
    );
    assert_eq!(aut.f, new.f, "the rescued stress cache is not Newton's");
    assert_eq!(
        (aut.n_iters, aut.last_residual),
        (new.n_iters, new.last_residual),
        "the rescued read-outs must be Newton's, since Newton produced the answer"
    );

    // And the abandoned sweeps are still paid for. This is the number a cost budget has to see:
    // the default is not free where it rescues, and `n_solves` is where that shows up.
    assert_eq!(
        aut.n_solves,
        pic.n_solves + new.n_solves,
        "a rescued step costs the abandoned sweeps plus Newton"
    );
    assert!(
        aut.n_solves > new.n_solves,
        "a rescue that cost no more than solo Newton did not run the sweeps at all"
    );
}

/// A sweep budget that merely runs out is rescued too — and that is a trade, not a free win.
///
/// `Capped` and `Expansive` are different failures (the first says more sweeps would have worked)
/// but they are the same thing to a caller: a step whose answer is not converged. So the fallback
/// fires on both.
///
/// **What it costs, stated because it is easy to meet by surprise:** a small `couple_max_iter` was
/// a cost *ceiling* under Picard — the step could never spend more than `2 · cap` back-substitutions.
/// Under `auto` it is not one any more; the step pays the capped sweeps and then Newton. `picard`
/// stays reachable precisely so that ceiling stays available to a caller who wants it.
#[test]
fn auto_rescues_a_capped_sweep_and_that_trades_away_the_cost_ceiling() {
    // `couple_max_iter` caps Newton's iterations as well as Picard's sweeps, so the fixture has
    // to be one where the budget is short for the sweeps and sufficient for Newton -- the whole
    // point being that they are not the same currency. At this strike Picard wants ~143 sweeps.
    let (amp, off, cap) = (6.0, 0.0, 8);
    let mut pic = auto_case(amp, off, cap, CoupleMethod::Picard);
    let mut new = auto_case(amp, off, cap, CoupleMethod::Newton);
    let mut aut = auto_case(amp, off, cap, CoupleMethod::Auto);

    pic.step(None).expect("the solves succeed");
    assert_eq!(
        pic.outcome(),
        CoupleOutcome::Capped,
        "the fixture must be one the sweeps run out on while still contracting"
    );
    assert_eq!(
        pic.n_solves,
        2 * cap as usize,
        "Picard's ceiling is `2 · cap` back-substitutions"
    );

    new.step(None).expect("the solves succeed");
    assert!(new.converged, "Newton must reach a root under the same cap");

    aut.step(None).expect("the solves succeed");
    assert!(aut.converged, "a capped sweep was not rescued");
    assert_eq!(aut.n_fallbacks, 1);
    assert_eq!(aut.u, new.u, "the rescued root is not Newton's");
    assert!(
        aut.n_solves > 2 * cap as usize,
        "the ceiling the cap used to impose is gone, and the trade must be visible in the cost"
    );
}

// -- Part 3's population: the line search and the inner solve, over the territory Newton opened --

/// A map point: `(fs, strike width, w/e)`, chosen from Part 3's grid so that Picard is `expansive`
/// on step zero and Newton converges. `rho` is 7800 rather than [`jac_plate_m`]'s 7860 so these are
/// the *same* plates the Python map measured (`M:\claud_projects\temp\vk-newton\part3_map.py`).
const MAP_POINTS: [(f64, f64, f64); 14] = [
    (24_000.0, 0.12, 18.0),
    (24_000.0, 0.08, 16.0),
    (24_000.0, 0.05, 12.0),
    (24_000.0, 0.03, 6.0),
    (24_000.0, 0.02, 3.0),
    (48_000.0, 0.12, 32.0),
    (48_000.0, 0.08, 24.0),
    (48_000.0, 0.05, 16.0),
    (48_000.0, 0.03, 10.0),
    (48_000.0, 0.02, 8.0),
    (96_000.0, 0.08, 40.0),
    (96_000.0, 0.05, 28.0),
    (96_000.0, 0.03, 16.0),
    (96_000.0, 0.02, 12.0),
];

fn map_plate(fs: f64, width: f64, amp: f64, boundary: Boundary, off: f64) -> VkPlate {
    let mut vk = VkPlate::new(
        VkParams::new(&VkSpec {
            lx: 0.4,
            ly: 0.4,
            young: 2.0e11,
            thickness: 1e-3,
            nu: 0.3,
            rho: 7800.0,
            fs,
            n: 20,
            boundary: Some(boundary),
            couple_max_iter: 50,
            couple_method: Some(CoupleMethod::Newton),
            ..VkSpec::default()
        })
        .expect("a valid plate"),
    );
    let p = &vk.p.lin;
    let (cx, cy) = (0.5 * p.lx + off, 0.5 * p.ly);
    let a = amp * vk.p.thickness;
    let u0: Vec<f64> = p
        .mask
        .flags()
        .iter()
        .enumerate()
        .filter(|(_, &alive)| alive)
        .map(|(idx, _)| {
            let (dx, dy) = (p.x[idx] - cx, p.y[idx] - cy);
            a * (-((dx * dx + dy * dy) / (width * width))).exp()
        })
        .collect();
    let zero = vec![0.0; p.n_live];
    vk.set_state(&u0, &zero).expect("the Airy solve factors");
    vk
}

/// The line search and the inner solve, over Part 3's population rather than the gate's six.
///
/// Plan §7 trap 2 asks whether the backtracking branch is live code, and
/// `the_line_search_is_recorded_rather_than_assumed` answered it on the *gate* fixtures: zero
/// halvings, with the search first firing only from a seed 200× the physical one. That is a
/// statement about six comfortable points. This one asks the same question where the map says
/// Newton is doing work Picard cannot — fourteen points spanning three sample rates and five strike
/// widths, every one of them `expansive` under best-effort Picard.
///
/// The three counters this reads (`n_line_search`, `gmres_products`, `gmres_stalls`) live on
/// [`VkNewtonReport`] and are deliberately **not** on `VkStep`, so no Python client can see them
/// (plan §11.4 — a field there costs two hand-copies). A native bar is the only place the question
/// can be asked over a population, which is why it is asked here.
#[test]
fn the_line_search_over_part_threes_population() {
    let mut halvings = 0usize;
    let mut stalls = 0usize;
    let mut worst_products = 0usize;
    let mut worst_iters = 0usize;
    for (fs, width, amp) in MAP_POINTS {
        let vk = map_plate(fs, width, amp, Boundary::Supported, 0.0);
        let ctx = VkCoupledStep::new(&vk.u, &vk.u_prev, &vk.f_prev, None, &vk.p);
        let r =
            vk_newton(&ctx, VkCoupledStep::seed(&vk.u, &vk.u_prev)).expect("the solves succeed");
        assert!(
            r.converged,
            "{fs} Hz, {width} m strike, {amp}e: Newton did not converge (residual {:.3e}) -- \
             the Python map says this point is Newton's, so either the map or this fixture moved",
            r.last_residual
        );
        halvings += r.n_line_search;
        stalls += r.gmres_stalls;
        worst_products = worst_products.max(r.gmres_products);
        worst_iters = worst_iters.max(r.n_iters);
    }

    // Measured, and the shape of the claim matters: this is a *finding*, so a change here is a new
    // measurement to write down rather than a number to raise. The full Newton step is accepted at
    // every one of these points, which says the residual stays convex enough for an undamped step
    // right up to the boundary the map draws -- and it is why the boundary is a cost ramp (the
    // Krylov work climbs) rather than a place the globalisation starts earning its keep.
    assert_eq!(
        halvings, 0,
        "the line search now fires somewhere in Part 3's population; that is a finding, not a \
         failure -- record where, and rewrite this claim rather than raising a number"
    );
    assert_eq!(
        stalls, 0,
        "the inner solve now hits NEWTON_GMRES_MAX_PRODUCTS in Part 3's population; the inexact \
         correction is then handed to the line search by design, but the population's cost claim \
         needs re-measuring"
    );
    // Bounds, not equalities: these are the cost ramp the map reports, and a tighter grid or a
    // different CPU may move the last digit of neither, but a factor here is a real change.
    assert!(
        worst_products <= 60,
        "worst GMRES products over the population is {worst_products}, was 38"
    );
    assert!(
        worst_iters <= 12,
        "worst Newton iterations over the population is {worst_iters}, was 7"
    );
}

/// The same population's free-edge twin, struck **off-centre** because a centred one is not a
/// free-boundary fixture at all (plan §10.4: a 3 cm Gaussian centred on a 40 cm plate is `exp(-44)`
/// at the rim, so free and supported are the same interior problem).
///
/// Part 3's Python slice measured the free edge as slightly *harder* than supported — Newton's
/// step-zero boundary at a 3 cm off-centre strike is `16e` free against `20e` supported. This bar
/// asserts the direction rather than the number: the free plate still converges where the map says
/// it does, and it still does it without damping.
#[test]
fn the_free_edge_holds_the_same_way_off_centre() {
    for (fs, width, amp) in [
        (48_000.0, 0.08, 16.0),
        (48_000.0, 0.03, 12.0),
        (96_000.0, 0.05, 20.0),
    ] {
        let vk = map_plate(fs, width, amp, Boundary::Free, 0.12);
        let ctx = VkCoupledStep::new(&vk.u, &vk.u_prev, &vk.f_prev, None, &vk.p);
        let r =
            vk_newton(&ctx, VkCoupledStep::seed(&vk.u, &vk.u_prev)).expect("the solves succeed");
        assert!(
            r.converged,
            "free edge, {fs} Hz, {width} m, {amp}e: residual {:.3e}",
            r.last_residual
        );
        assert_eq!(
            r.n_line_search, 0,
            "the free edge now needs damping at {fs} Hz, {width} m, {amp}e -- a finding"
        );
    }
}
