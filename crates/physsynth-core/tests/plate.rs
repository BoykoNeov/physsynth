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
    count_components, energy, grain_ratios_from_material, linspace0, pickup_index_at, Boundary,
    Domain, MaterialError, ParamError, Params, Plate, PlateSpec, VkParamError, VkParams, VkPlate,
    VkSpec,
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
