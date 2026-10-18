//! The plate's frequency-dependent loss `2 σ₁ Δ u_t` (`docs/dev/plate-mapping-plan.md` §2).
//!
//! Born after the Python plate, so there is no frozen answer to hold it to; every bar here is
//! physics written out in the test. The loss operator is never taken from the model where it can
//! be avoided: the supported branch's Laplacian eigenvalues are closed form, and the free branch's
//! gradient form is re-assembled here from the mask, so a model that dissipates through the wrong
//! operator cannot agree with a bar that uses its own.

use physsynth_core::eig::generalized_eigen_diag;
use physsynth_core::mallet::{MalletPlate, PlateParams};
use physsynth_core::plate::{
    initial_previous, loss_form, Boundary, Domain, ParamError, Params, Plate, PlateSpec,
};
use std::f64::consts::PI;

const KAPPA: f64 = 20.0;
const RHO: f64 = 0.7;
const FS: f64 = 8000.0;

fn spec(boundary: Boundary, domain: Domain, n: i64, sigma: f64, sigma1: f64) -> PlateSpec {
    let (lx, ly) = if domain == Domain::Guitar {
        (0.4, 0.5)
    } else {
        (0.5, 0.37)
    };
    PlateSpec {
        lx,
        ly,
        kappa: KAPPA,
        rho: RHO,
        fs: FS,
        n,
        sigma,
        sigma1,
        boundary: Some(boundary),
        domain: Some(domain),
        ..PlateSpec::default()
    }
}

fn build(s: &PlateSpec) -> Plate {
    Plate::new(Params::new(s).expect("a valid plate"))
}

/// Live-node coordinates, walked from the mask.
fn live_xy(p: &Params) -> Vec<(f64, f64)> {
    let flags = p.mask.flags();
    (0..flags.len())
        .filter(|&i| flags[i])
        .map(|i| (p.x[i], p.y[i]))
        .collect()
}

/// A smooth off-centre bump.
fn bump(p: &Params, amp: f64) -> Vec<f64> {
    let width = 0.1 * p.lx;
    live_xy(p)
        .into_iter()
        .map(|(x, y)| {
            let (dx, dy) = (x - 0.42 * p.lx, y - 0.38 * p.ly);
            amp * (-((dx * dx + dy * dy) / (width * width))).exp()
        })
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `uᵀ G v` for the free branch's `∫∇u·∇v`, assembled here edge by edge from the live cells (a
/// cell with all four corners live hands each of its edges half a weight) — the bar's own copy.
fn free_gradient(p: &Params, u: &[f64], v: &[f64]) -> f64 {
    let (nr, nc) = (p.mask.nrows(), p.mask.ncols());
    let at = |j: usize, i: usize| p.index_map[j * nc + i];
    let mut s = 0.0;
    for j in 0..nr - 1 {
        for i in 0..nc - 1 {
            let c = [at(j, i), at(j, i + 1), at(j + 1, i), at(j + 1, i + 1)];
            if c.iter().any(|&q| q < 0) {
                continue;
            }
            let c = c.map(|q| q as usize);
            for (a, b) in [(c[0], c[1]), (c[2], c[3]), (c[0], c[2]), (c[1], c[3])] {
                s += 0.5 * (u[a] - u[b]) * (v[a] - v[b]);
            }
        }
    }
    s
}

/// The scalar two-step recurrence of one mode, `A z² + B z + C = 0`, run forward from two values.
fn recurrence(x: f64, y: f64, theta: f64, a0: f64, a1: f64, steps: usize) -> Vec<f64> {
    let (aa, bb, cc) = (
        1.0 + theta * x + y,
        -2.0 + (1.0 - 2.0 * theta) * x,
        1.0 + theta * x - y,
    );
    let mut out = vec![a0, a1];
    for n in 1..steps {
        out.push(-(bb * out[n] + cc * out[n - 1]) / aa);
    }
    out
}

// -- the supported branch: every Navier mode decays at its own closed-form rate -------------------

#[test]
fn each_supported_mode_decays_at_sigma0_plus_sigma1_times_its_laplacian_eigenvalue() {
    // Isotropic and grained (a spruce-like stiffness ratio), three thetas, three modes. The mode
    // is an exact eigenvector of the scheme, so its amplitude obeys the scalar recurrence to
    // rounding — with the loss written here from the 1-D Dirichlet eigenvalues, not taken from L.
    let (sigma, sigma1) = (2.0, 0.05);
    let grains = [(1.0, 1.0, 1.0), (1.0, 0.35, 0.08)];
    for (gx, gc, gy) in grains {
        for theta in [0.28, 0.5, 1.0] {
            let mut s = spec(Boundary::Supported, Domain::Rectangle, 12, sigma, sigma1);
            s.theta = theta;
            if gc != 1.0 {
                s.grain_x = gx;
                s.grain_cross = Some(gc);
                s.grain_y = gy;
            }
            for (pm, qm) in [(1usize, 1usize), (3, 2), (7, 5)] {
                let mut pl = build(&s);
                let (h, k, n, ny) = (pl.p.h, pl.p.k, pl.p.n, pl.p.ny);
                let phi: Vec<f64> = live_xy(&pl.p)
                    .into_iter()
                    .map(|(x, y)| {
                        (pm as f64 * PI * x / pl.p.lx).sin() * (qm as f64 * PI * y / pl.p.ly).sin()
                    })
                    .collect();
                let dp = 4.0 / (h * h) * (pm as f64 * PI / (2.0 * n as f64)).sin().powi(2);
                let eq = 4.0 / (h * h) * (qm as f64 * PI / (2.0 * ny as f64)).sin().powi(2);
                let beta = gx * dp * dp + 2.0 * gc * dp * eq + gy * eq * eq;
                let lam = dp + eq;
                let pp = dot(&phi, &phi);
                pl.set_state(&phi, &vec![0.0; phi.len()]);
                let steps = 400;
                let mut got = vec![dot(&phi, &pl.u_prev) / pp, dot(&phi, &pl.u) / pp];
                for _ in 1..steps {
                    pl.step(None);
                    got.push(dot(&phi, &pl.u) / pp);
                }
                let x = k * k * KAPPA * KAPPA * beta;
                let want = recurrence(x, k * (sigma + sigma1 * lam), theta, got[0], got[1], steps);
                let without = recurrence(x, k * sigma, theta, got[0], got[1], steps);
                let scale = got.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                let err = got
                    .iter()
                    .zip(&want)
                    .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
                let ctl = got
                    .iter()
                    .zip(&without)
                    .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
                let label = format!("grain ({gx},{gc},{gy}) theta {theta} mode ({pm},{qm})");
                assert!(err < 1e-9 * scale, "{label}: {err:.3e} off the recurrence");
                // the control: without sigma1 the same mode is visibly elsewhere
                assert!(
                    ctl > 1e-3 * scale,
                    "{label}: sigma1 made no difference ({ctl:.3e})"
                );
            }
        }
    }
}

// -- every branch: the energy books close ----------------------------------------------------------

#[test]
fn the_energy_drop_is_the_work_both_losses_do_on_every_branch() {
    // E^{n+1/2} − E^{n−1/2} = −2 k ρ [σ₀ vᵀ M v + σ₁ vᵀ G v], v = (u^{n+1} − u^{n−1}) / 2k, with
    // M = h² I and G = −h² L supported, M = W and G the gradient form free. Accumulated over the
    // run against the starting energy at the project's 1e-10.
    let (sigma, sigma1) = (1.0, 0.02);
    for (boundary, domain, n) in [
        (Boundary::Supported, Domain::Rectangle, 12),
        (Boundary::Free, Domain::Rectangle, 12),
        (Boundary::Free, Domain::Guitar, 16),
    ] {
        for theta in [0.28, 0.5] {
            let mut s = spec(boundary, domain, n, sigma, sigma1);
            s.theta = theta;
            let mut pl = build(&s);
            let u0 = bump(&pl.p, 1e-4);
            pl.set_state(&u0, &vec![0.0; u0.len()]);
            let (k, h2) = (pl.p.k, pl.p.h * pl.p.h);
            let e0 = pl.energy();
            let (mut spent, mut spent1, mut worst) = (0.0, 0.0, 0.0f64);
            for _ in 0..400 {
                let older = pl.u_prev.clone();
                pl.step(None);
                let v: Vec<f64> = (0..older.len())
                    .map(|i| (pl.u[i] - older[i]) / (2.0 * k))
                    .collect();
                let (mass, grad) = match boundary {
                    Boundary::Supported => {
                        let lv = pl.p.laplacian.as_ref().unwrap().matvec(&v);
                        (h2 * dot(&v, &v), -h2 * dot(&v, &lv))
                    }
                    Boundary::Free => {
                        let wv: Vec<f64> = v.iter().zip(&pl.p.w).map(|(a, b)| a * b).collect();
                        (dot(&v, &wv), free_gradient(&pl.p, &v, &v))
                    }
                };
                spent += 2.0 * k * RHO * (sigma * mass + sigma1 * grad);
                spent1 += 2.0 * k * RHO * sigma1 * grad;
                worst = worst.max((pl.energy() - e0 + spent).abs() / e0);
            }
            let label = format!("{boundary:?}/{domain:?} theta {theta}");
            assert!(worst < 1e-10, "{label}: the books are open by {worst:.3e}");
            assert!(
                spent1 > 0.05 * e0,
                "{label}: sigma1 did {:.3e} of {e0:.3e} — the bar would not see it missing",
                spent1
            );
        }
    }
}

#[test]
fn the_frequency_dependent_loss_alone_is_passive() {
    for (boundary, domain, n) in [
        (Boundary::Supported, Domain::Rectangle, 12),
        (Boundary::Free, Domain::Rectangle, 12),
        (Boundary::Free, Domain::Guitar, 16),
    ] {
        let mut pl = build(&spec(boundary, domain, n, 0.0, 0.05));
        let u0 = bump(&pl.p, 1e-4);
        pl.set_state(&u0, &vec![0.0; u0.len()]);
        let e0 = pl.energy();
        let mut last = e0;
        for step in 0..400 {
            pl.step(None);
            let now = pl.energy();
            assert!(
                now <= last * (1.0 + 1e-12),
                "{domain:?} step {step}: {now} > {last}"
            );
            last = now;
        }
        // Monotone is only half the claim — a plate that never moved is monotone too.
        assert!(
            last < 0.75 * e0,
            "{boundary:?}/{domain:?}: lost only to {last} of {e0}"
        );
    }
}

// -- the free branch ------------------------------------------------------------------------------

#[test]
fn a_free_translation_is_not_damped_and_a_tilt_is() {
    // `G` annihilates constants: a plate sliding sideways keeps its speed under σ₁ alone. A tilt
    // is not in G's null space (plan §9 measured it; the human accepted it), so it slows — the
    // second half pins the documented behaviour so a change to it is a decision, not an accident.
    let s = spec(Boundary::Free, Domain::Rectangle, 12, 0.0, 0.05);
    let mut slide = build(&s);
    let n = slide.p.n_live;
    slide.set_state(&vec![0.0; n], &vec![1e-3; n]);
    let e0 = slide.energy();
    for _ in 0..300 {
        slide.step(None);
    }
    let k = slide.p.k;
    let worst = (0..n).fold(0.0f64, |m, i| {
        m.max(((slide.u[i] - slide.u_prev[i]) / k - 1e-3).abs())
    });
    // Rounding through the solve reads ~1e-10 of the speed here; a damped slide would lose
    // `σ₁ ε t` of it for a row sum ε, so a 1e-9 bar sees any ε above ~1e-6.
    assert!(worst < 1e-9 * 1e-3, "the slide slowed by {worst:.3e}");
    assert!((slide.energy() / e0 - 1.0).abs() < 1e-10);

    let mut tilt = build(&s);
    let xs: Vec<f64> = live_xy(&tilt.p).into_iter().map(|(x, _)| x).collect();
    let mean = xs.iter().sum::<f64>() / n as f64;
    let v0: Vec<f64> = xs.iter().map(|x| 1e-3 * (x - mean)).collect();
    tilt.set_state(&vec![0.0; n], &v0);
    let e0 = tilt.energy();
    for _ in 0..300 {
        tilt.step(None);
    }
    assert!(
        tilt.energy() < (1.0 - 1e-2) * e0,
        "the tilt kept its energy under sigma1"
    );
}

#[test]
fn a_free_mode_decays_at_its_rayleigh_quotient_to_within_the_coupling() {
    // The free form does not share eigenvectors with K, so a mode is not exactly a mode under the
    // loss; Part 0 measured the miss at ~1e-5 in decay rate. Started in a (K, W) eigenvector, its
    // projection follows the scalar recurrence with λ = φᵀGφ (the bar's own G) to the stated
    // tolerance — and visibly not the recurrence without σ₁.
    let (sigma, sigma1, theta) = (0.5, 0.02, 0.28);
    for (domain, n) in [(Domain::Rectangle, 10), (Domain::Guitar, 12)] {
        let s = spec(Boundary::Free, domain, n, sigma, sigma1);
        let probe = Params::new(&s).unwrap();
        let nl = probe.n_live;
        let mut kd = vec![0.0; nl * nl];
        for i in 0..nl {
            for j in 0..nl {
                kd[i * nl + j] = probe.stiffness.get(i, j);
            }
        }
        let (beta, vecs) = generalized_eigen_diag(&kd, &probe.w, nl).unwrap();
        for j in [3usize, 15] {
            let phi: Vec<f64> = (0..nl).map(|i| vecs[i * nl + j]).collect();
            let lam = free_gradient(&probe, &phi, &phi);
            let wphi: Vec<f64> = phi.iter().zip(&probe.w).map(|(a, b)| a * b).collect();
            let mut pl = Plate::new(probe.clone());
            pl.set_state(&phi, &vec![0.0; nl]);
            let steps = 300;
            let mut got = vec![dot(&wphi, &pl.u_prev), dot(&wphi, &pl.u)];
            for _ in 1..steps {
                pl.step(None);
                got.push(dot(&wphi, &pl.u));
            }
            let k = probe.k;
            let x = k * k * KAPPA * KAPPA * beta[j];
            let want = recurrence(x, k * (sigma + sigma1 * lam), theta, got[0], got[1], steps);
            let without = recurrence(x, k * sigma, theta, got[0], got[1], steps);
            let err = got
                .iter()
                .zip(&want)
                .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
            let ctl = got
                .iter()
                .zip(&without)
                .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
            println!("{domain:?} mode {j}: err {err:.3e}, without sigma1 {ctl:.3e}");
            assert!(
                err < 1e-4,
                "{domain:?} mode {j}: {err:.3e} off the recurrence"
            );
            assert!(
                ctl > 1e-2,
                "{domain:?} mode {j}: sigma1 made no difference ({ctl:.3e})"
            );
        }
    }
}

// -- construction ----------------------------------------------------------------------------------

#[test]
fn a_negative_or_non_finite_sigma1_is_refused() {
    for bad in [-1e-6, f64::NAN, f64::INFINITY] {
        let err = Params::new(&spec(Boundary::Supported, Domain::Rectangle, 8, 0.0, bad))
            .expect_err("refused");
        assert_eq!(err, ParamError::NegativeSigma1, "sigma1 = {bad}");
        assert_eq!(
            err.to_string(),
            "sigma1 (frequency-dependent loss) must be finite and >= 0."
        );
    }
}

#[test]
fn the_loss_form_is_carried_only_when_it_is_used() {
    for boundary in [Boundary::Supported, Boundary::Free] {
        let off = Params::new(&spec(boundary, Domain::Rectangle, 8, 1.0, 0.0)).unwrap();
        assert!(
            off.loss.is_none(),
            "{boundary:?}: sigma1 = 0 must carry no loss form"
        );
        let on = Params::new(&spec(boundary, Domain::Rectangle, 8, 1.0, 0.01)).unwrap();
        let carried = on.loss.as_ref().expect("sigma1 > 0 carries the form");
        assert_eq!(
            carried,
            &loss_form(&off),
            "{boundary:?}: one form, whatever sigma1 is"
        );
    }
}

#[test]
fn the_start_up_is_the_lossless_taylor_step_whatever_sigma1_is() {
    // As with σ₀: the first previous state is `u0 − k v0 + ½ k² a0` with the lossless
    // acceleration. A start-up has its own bar because conservation bars cannot see it.
    for boundary in [Boundary::Supported, Boundary::Free] {
        let off = Params::new(&spec(boundary, Domain::Rectangle, 10, 0.0, 0.0)).unwrap();
        let on = Params::new(&spec(boundary, Domain::Rectangle, 10, 0.0, 0.03)).unwrap();
        let u0 = bump(&off, 1e-3);
        let v0: Vec<f64> = u0.iter().map(|u| 7.0 * u).collect();
        assert_eq!(
            initial_previous(&u0, &v0, &off),
            initial_previous(&u0, &v0, &on)
        );
    }
}

// -- clients that compose the plate's step ---------------------------------------------------------

#[test]
fn the_mallets_influence_column_carries_the_loss() {
    // The mallet steps the plate force-free, then adds `column · f` from the plate's own
    // factorization. With σ₁ > 0 that must still equal a step with the force in the right-hand
    // side — over the whole field.
    for boundary in [Boundary::Supported, Boundary::Free] {
        let mut s = spec(boundary, Domain::Rectangle, 12, 0.5, 0.03);
        s.nu = Some(0.3);
        let pl = build(&s);
        let p = PlateParams::new(
            &pl.p, 0.02, 5.0e4, 2.3, 0.0, 0.3, 0.2, 0.0, 1e-12, 1e-14, 60,
        )
        .expect("valid mallet");
        let mal = MalletPlate::new(p, pl, 0.0, 3.0);
        let (node, influence) = (mal.params().node, &mal.params().influence);
        let force = 7.5;
        let mut forced = build(&s);
        let b = bump(&forced.p, 1e-3);
        forced.set_state(&b, &vec![0.0; b.len()]);
        for _ in 0..5 {
            forced.step(None);
        }
        let mut split = forced.clone();
        let mut f_ext = vec![0.0; forced.p.n_live];
        f_ext[node] = -force;
        forced.step(Some(&f_ext));
        split.step(None);
        let scale = forced.u.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let worst = forced
            .u
            .iter()
            .zip(&split.u)
            .zip(influence)
            .fold(0.0f64, |m, ((f, s), g)| m.max((f - (s - g * force)).abs()))
            / scale;
        assert!(
            worst < 1e-14,
            "{boundary:?}: column vs forced step {worst:.3e}"
        );
    }
}
