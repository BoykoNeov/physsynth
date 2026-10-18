//! The membrane's frequency-dependent loss `2 σ₁ Δ u_t`, explicit backward difference
//! (`docs/dev/membrane-mapping-plan.md` §2).
//!
//! Born after the Python membrane, so there is no frozen answer to hold it to; every bar here is
//! physics written out in the test. The loss operator is never taken from the model: the
//! rectangle's Laplacian eigenvalues are closed form, and `−L` is re-applied here from the mask, so
//! a model that dissipates through the wrong operator cannot agree with a bar that uses its own.

use physsynth_core::airbox::{AirBox, Params as RoomParams, Wall, C0_AIR, RHO0_AIR};
use physsynth_core::airbox_port::Spreading;
use physsynth_core::airbox_wrap::{MembraneSeam, RoomGrid};
use physsynth_core::mallet::{self, MalletMembrane};
use physsynth_core::membrane::{initial_previous, Domain, Membrane, ParamError, Params};
use std::f64::consts::PI;

const T: f64 = 3000.0;
const RHO: f64 = 0.26;
const FS: f64 = 44_100.0;

fn rect(n: i64, sigma: f64) -> Params {
    Params::new(
        Some(Domain::Rectangle),
        T,
        RHO,
        FS,
        n,
        Some(0.30),
        Some(0.22),
        None,
        sigma,
    )
    .expect("a valid rectangle")
}

fn circle(n: i64, sigma: f64) -> Params {
    Params::new(
        Some(Domain::Circle),
        T,
        RHO,
        FS,
        n,
        None,
        None,
        Some(0.15),
        sigma,
    )
    .expect("a valid circle")
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `−L v` from the mask: `(4 v_i − Σ live neighbours) / h²`, a dead neighbour reading zero — the
/// bar's own copy of the clamped 5-point operator.
fn neg_lap(p: &Params, v: &[f64]) -> Vec<f64> {
    let (nr, nc) = (p.mask.nrows(), p.mask.ncols());
    let at = |j: i64, i: i64| -> f64 {
        if j < 0 || i < 0 || j >= nr as i64 || i >= nc as i64 {
            return 0.0;
        }
        let q = p.index_map[j as usize * nc + i as usize];
        if q < 0 {
            0.0
        } else {
            v[q as usize]
        }
    };
    let mut out = vec![0.0; v.len()];
    for j in 0..nr as i64 {
        for i in 0..nc as i64 {
            let q = p.index_map[j as usize * nc + i as usize];
            if q < 0 {
                continue;
            }
            let s = at(j - 1, i) + at(j + 1, i) + at(j, i - 1) + at(j, i + 1);
            out[q as usize] = (4.0 * v[q as usize] - s) / (p.h * p.h);
        }
    }
    out
}

/// The work both losses take out between `E^n` and `E^{n+1}`:
/// `k ρ h² [2σ ‖v‖² + 2σ₁ ⟨−L v, v⟩]`, `v = (u^{n+1} − u^{n−1}) / 2k`.
fn loss_work(p: &Params, u_next: &[f64], u_prev: &[f64]) -> (f64, f64) {
    let v: Vec<f64> = u_next
        .iter()
        .zip(u_prev)
        .map(|(a, b)| (a - b) / (2.0 * p.k))
        .collect();
    let w = p.k * p.rho * p.h * p.h;
    (
        w * 2.0 * p.sigma * dot(&v, &v),
        w * 2.0 * p.sigma1 * dot(&neg_lap(p, &v), &v),
    )
}

fn live(p: &Params, field: &[f64]) -> Vec<f64> {
    p.to_live(field)
}

/// A smooth off-centre bump.
fn bump(p: &Params) -> Vec<f64> {
    let (x, y) = (live(p, &p.x), live(p, &p.y));
    let (cx, cy, w) = match p.domain {
        Domain::Rectangle => (0.11, 0.08, 0.04),
        Domain::Circle => (0.03, -0.02, 0.04),
    };
    x.iter()
        .zip(&y)
        .map(|(x, y)| 1e-3 * (-((x - cx).powi(2) + (y - cy).powi(2)) / (w * w)).exp())
        .collect()
}

/// The rectangle's `(m, n)` Dirichlet mode at the live nodes, unit-normalized, and its `−L`
/// eigenvalue from the 1-D eigenvalues, written here.
fn rect_mode(p: &Params, m: f64, n: f64) -> (Vec<f64>, f64) {
    let (lx, ly, h) = (p.lx.expect("rectangle"), p.ly.expect("rectangle"), p.h);
    let (x, y) = (live(p, &p.x), live(p, &p.y));
    let mut v: Vec<f64> = x
        .iter()
        .zip(&y)
        .map(|(x, y)| (m * PI * x / lx).sin() * (n * PI * y / ly).sin())
        .collect();
    let norm = dot(&v, &v).sqrt();
    v.iter_mut().for_each(|e| *e /= norm);
    let s = |q: f64, l: f64| (q * PI * h / (2.0 * l)).sin().powi(2);
    (v, 4.0 / (h * h) * (s(m, lx) + s(n, ly)))
}

/// One mode's scalar recurrence, `(1 + σk) z² + (−2 + X + Y) z + (1 − σk − Y) = 0`, with
/// `X = c²k²μ`, `Y = 2σ₁kμ`, run forward from two values.
fn recurrence(p: &Params, mu: f64, sigma1: f64, a0: f64, a1: f64, steps: usize) -> Vec<f64> {
    let (sk, x, y) = (
        p.sigma * p.k,
        p.c * p.c * p.k * p.k * mu,
        2.0 * sigma1 * p.k * mu,
    );
    let mut out = vec![a0, a1];
    for n in 1..steps {
        out.push(((2.0 - x - y) * out[n] - (1.0 - sk - y) * out[n - 1]) / (1.0 + sk));
    }
    out
}

// -- every rectangle mode decays at σ + σ₁μ ---------------------------------------------------------

#[test]
fn each_rectangle_mode_follows_its_recurrence_with_sigma0_plus_sigma1_mu() {
    // The mode is an exact eigenvector of the masked L, so its amplitude obeys the scalar
    // recurrence to rounding — with μ from the 1-D Dirichlet eigenvalues, not from the model.
    let (sigma, sigma1) = (1.5, 2e-3);
    let p = rect(40, sigma)
        .with_sigma1(sigma1)
        .expect("inside the bound");
    for (m, n) in [(1.0, 1.0), (3.0, 2.0), (7.0, 5.0)] {
        let (phi, mu) = rect_mode(&p, m, n);
        let mut mem = Membrane::new(p.clone());
        mem.set_displacement(&phi);
        let mut seen = vec![dot(&mem.u_prev, &phi), dot(&mem.u, &phi)];
        let steps = 3000;
        for _ in 1..steps {
            mem.step();
            seen.push(dot(&mem.u, &phi));
        }
        let with = recurrence(&p, mu, sigma1, seen[0], seen[1], steps);
        let without = recurrence(&p, mu, 0.0, seen[0], seen[1], steps);
        let miss = |r: &[f64]| {
            r.iter()
                .zip(&seen)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max)
        };
        assert!(
            miss(&with) < 1e-9 * seen[1].abs(),
            "({m},{n}): {:.2e}",
            miss(&with)
        );
        assert!(
            miss(&without) > 1e-4 * seen[1].abs(),
            "({m},{n}): σ₁ changed nothing"
        );
    }
}

// -- the books close, and only with the correction ------------------------------------------------

#[test]
fn the_energy_books_close_per_step_on_both_shapes() {
    for (name, base) in [("rectangle", rect(40, 1.5)), ("circle", circle(40, 1.5))] {
        let p = base.with_sigma1(2e-3).expect("inside the bound");
        let mut mem = Membrane::new(p.clone());
        mem.set_displacement(&bump(&p));
        let e0 = mem.energy();
        let (mut worst, mut s1_total, mut prev) = (0.0f64, 0.0, e0);
        for _ in 0..3000 {
            let before = mem.u_prev.clone();
            mem.step();
            let (w0, w1) = loss_work(&p, &mem.u, &before);
            let now = mem.energy();
            worst = worst.max((now - prev + w0 + w1).abs());
            assert!(now <= prev, "{name}: the lossy run gained energy");
            s1_total += w1;
            prev = now;
        }
        assert!(
            worst < 1e-10 * e0,
            "{name}: books miss by {:.2e} of E0",
            worst / e0
        );
        assert!(
            s1_total > 0.05 * e0,
            "{name}: σ₁ did {:.3} of E0",
            s1_total / e0
        );
    }
}

#[test]
fn without_the_energy_correction_the_books_do_not_close() {
    // The control for the bar above: the same run judged by the σ₁ = 0 energy formula misses by
    // far more than rounding, so a dropped correction is a red bar, not a silent one.
    let p = rect(40, 1.5).with_sigma1(2e-3).expect("inside the bound");
    let mut plain = p.clone();
    plain.sigma1 = 0.0;
    let mut mem = Membrane::new(p.clone());
    mem.set_displacement(&bump(&p));
    let e0 = mem.energy();
    let energy_plain = |m: &Membrane| physsynth_core::membrane::energy(&m.u, &m.u_prev, &plain);
    let (mut worst, mut prev) = (0.0f64, energy_plain(&mem));
    for _ in 0..2000 {
        let before = mem.u_prev.clone();
        mem.step();
        let (w0, w1) = loss_work(&p, &mem.u, &before);
        let now = energy_plain(&mem);
        worst = worst.max((now - prev + w0 + w1).abs());
        prev = now;
    }
    assert!(
        worst > 1e-7 * e0,
        "the correction is invisible: {:.2e}",
        worst / e0
    );
}

#[test]
fn sigma1_alone_is_passive_on_both_shapes() {
    for (name, base) in [("rectangle", rect(30, 0.0)), ("circle", circle(30, 0.0))] {
        let p = base.with_sigma1(5e-3).expect("inside the bound");
        let mut mem = Membrane::new(p.clone());
        mem.set_displacement(&bump(&p));
        let e0 = mem.energy();
        let mut prev = e0;
        for _ in 0..2000 {
            mem.step();
            let now = mem.energy();
            assert!(now <= prev + 1e-14 * e0, "{name}: σ₁ alone added energy");
            prev = now;
        }
        assert!(
            prev < 0.9 * e0,
            "{name}: σ₁ alone took only {:.3} of E0",
            1.0 - prev / e0
        );
    }
}

// -- σ₁ = 0 changes nothing -------------------------------------------------------------------------

#[test]
fn sigma1_zero_is_the_old_membrane_bit_for_bit() {
    for base in [rect(30, 1.5), circle(30, 1.5)] {
        let zero = base.clone().with_sigma1(0.0).expect("zero is always legal");
        assert_eq!(zero.sigma1.to_bits(), 0.0f64.to_bits());
        let (mut a, mut b) = (Membrane::new(base.clone()), Membrane::new(zero));
        let u0 = bump(&base);
        a.set_displacement(&u0);
        b.set_displacement(&u0);
        for _ in 0..300 {
            a.step();
            b.step();
        }
        assert!(a
            .u
            .iter()
            .zip(&b.u)
            .all(|(x, y)| x.to_bits() == y.to_bits()));
        assert_eq!(a.energy().to_bits(), b.energy().to_bits());
    }
    // A membrane exactly at the plain CFL ceiling stays legal with σ₁ = 0.
    let c = (T / RHO).sqrt();
    let h = 0.30 / 30.0;
    let fs = c / (h / 2.0f64.sqrt());
    let at_ceiling = Params::new(
        Some(Domain::Rectangle),
        T,
        RHO,
        fs,
        30,
        Some(0.30),
        Some(0.22),
        None,
        0.0,
    )
    .expect("the ceiling itself is legal");
    assert!(at_ceiling.clone().with_sigma1(0.0).is_ok());
    assert!(matches!(
        at_ceiling.with_sigma1(1e-9),
        Err(ParamError::LossCflViolated(_))
    ));
}

#[test]
fn the_start_up_is_the_lossless_taylor_step_either_way() {
    // `initial_previous` ignores σ as it always has; σ₁ is treated the same way.
    let base = rect(30, 1.5);
    let with = base.clone().with_sigma1(4e-3).expect("inside the bound");
    let u0 = bump(&base);
    let v0: Vec<f64> = u0.iter().map(|x| 0.3 * x).collect();
    let a = initial_previous(&u0, &v0, &base);
    let b = initial_previous(&u0, &v0, &with);
    assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
}

// -- the bound ------------------------------------------------------------------------------------

#[test]
fn the_tightened_bound_is_refused_past_and_is_the_real_one() {
    let p = rect(40, 0.0);
    let suff = (0.5 - p.lam * p.lam) * p.h * p.h / (4.0 * p.k);
    assert!(p.clone().with_sigma1(0.999 * suff).is_ok());
    match p.clone().with_sigma1(1.001 * suff) {
        Err(ParamError::LossCflViolated(v)) => assert!(v > 0.5),
        other => panic!("expected the loss CFL refusal, got {other:?}"),
    }
    for bad in [-1e-6, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            p.clone().with_sigma1(bad),
            Err(ParamError::NegativeSigma1)
        ));
    }

    // The exact limit, for the grid's top mode: c²k²μ + 4σ₁kμ ≤ 4. Just inside it the top mode
    // stays bounded; just outside — set past the refusal by hand — it grows. So the bound guards a
    // real instability, and the sufficient form sits just below it.
    let (lx, ly) = (p.lx.unwrap(), p.ly.unwrap());
    let (nx, ny) = ((lx / p.h).round(), (ly / p.h).round());
    let (top, mu) = rect_mode(&p, nx - 1.0, ny - 1.0);
    let exact = (4.0 - p.c * p.c * p.k * p.k * mu) / (4.0 * p.k * mu);
    assert!(
        suff < exact && exact < 1.01 * suff,
        "sufficient {suff:.4e}, exact {exact:.4e}"
    );
    for (factor, grows) in [(0.99, false), (1.01, true)] {
        let mut q = p.clone();
        q.sigma1 = factor * exact;
        let mut mem = Membrane::new(q);
        mem.set_displacement(&top);
        for _ in 0..3000 {
            mem.step();
        }
        let amp = dot(&mem.u, &mem.u).sqrt();
        assert_eq!(
            amp > 1e3,
            grows,
            "at {factor} x the exact limit: |u| = {amp:.3e}"
        );
    }
}

// -- the clients carry it -------------------------------------------------------------------------

#[test]
fn the_struck_drum_s_books_close_with_sigma1() {
    // The mallet steps through `Membrane::step`, so it carries σ₁ with no code of its own; its
    // contact uses the bare nodal admittance, which the explicit term leaves true. Elastic felt,
    // σ = 0: the coupled total plus the head's σ₁ work is a constant.
    let n = 24i64;
    let (t, rho) = (100.0, 0.26);
    let c = f64::sqrt(t / rho);
    let fs = c / (0.5 / n as f64);
    let mp = Params::new(
        Some(Domain::Rectangle),
        t,
        rho,
        fs,
        n,
        Some(1.0),
        Some(1.0),
        None,
        0.0,
    )
    .unwrap()
    .with_sigma1(2e-3)
    .expect("inside the bound");
    let mem = Membrane::new(mp.clone());
    let params = mallet::Params::new(
        mem.params(),
        0.02,
        5.0e4,
        2.3,
        0.0,
        0.5,
        0.5,
        0.0,
        1e-12,
        1e-14,
        60,
    )
    .expect("a valid mallet");
    let mut mal = MalletMembrane::new(params, mem, 0.0, 3.0);
    let e0 = mal.energy();
    let (mut worst, mut s1_total, mut deepest) = (0.0f64, 0.0, 0.0f64);
    for _ in 0..1500 {
        let before = mal.membrane.u_prev.clone();
        mal.step().expect("contact solve converged");
        s1_total += loss_work(&mp, &mal.membrane.u, &before).1;
        worst = worst.max((mal.energy() + s1_total - e0).abs());
        deepest = deepest.max(mal.state().penetration);
    }
    assert!(deepest > 0.0, "the mallet never struck");
    assert!(worst < 1e-10 * e0, "books miss by {:.2e} of E0", worst / e0);
    assert!(
        s1_total > 0.01 * e0,
        "σ₁ did only {:.3e} of E0",
        s1_total / e0
    );
}

#[test]
fn the_room_s_books_close_with_sigma1() {
    // The room seam's right-hand side is `step_numerator_into` and its `A` stays `(1 + σk) I`, so
    // σ₁ arrives through the shared numerator. Scene total (head + air + radiated channel) plus the
    // head's loss work is a constant, and the port's two ledgers agree.
    let fs = 40_000.0;
    let h_air = C0_AIR * 3.0f64.sqrt() / (0.9 * fs);
    for suspended in [false, true] {
        let cells = [28.0, 28.0, 9.0];
        let l = [cells[0] * h_air, cells[1] * h_air, cells[2] * h_air];
        let mut room = AirBox::new(
            RoomParams::new(l, fs, h_air, [Wall::Rigid; 6], None, RHO0_AIR, C0_AIR).unwrap(),
        );
        let p = Params::new(
            Some(Domain::Circle),
            T,
            RHO,
            fs,
            16,
            None,
            None,
            Some(0.12),
            0.5,
        )
        .unwrap()
        .with_sigma1(3e-3)
        .expect("inside the bound");
        let seam = MembraneSeam::new(Membrane::new(p.clone()));
        let mut g = if suspended {
            RoomGrid::suspended(seam, &mut room, "z", 4, None, Spreading::Bilinear)
        } else {
            RoomGrid::baffled(seam, &mut room, "z0", None, Spreading::Bilinear)
        }
        .expect("a legal head");
        let u0 = bump(&p);
        g.set_state(&u0, &vec![0.0; u0.len()]);
        let total = |g: &RoomGrid<MembraneSeam>, room: &AirBox| g.energy() + room.energy();
        let e0 = total(&g, &room);
        let (mut worst, mut work, mut s1_total) = (0.0f64, 0.0, 0.0);
        for _ in 0..400 {
            let before = g.seam.membrane.u_prev.clone();
            g.step(&mut room, None).expect("a legal step");
            room.step();
            let (w0, w1) = loss_work(&p, &g.seam.membrane.u, &before);
            work += w0 + w1;
            s1_total += w1;
            worst = worst.max((total(&g, &room) + work - e0).abs());
        }
        assert!(
            worst < 1e-10 * e0,
            "suspended={suspended}: books miss {:.2e}",
            worst / e0
        );
        assert!(
            s1_total > 0.01 * e0,
            "suspended={suspended}: σ₁ did {:.3e}",
            s1_total / e0
        );
        let gap = (g.radiated_energy() - room.injected).abs();
        assert!(
            gap <= 1e-12 * g.radiated_energy().abs(),
            "ledger gap {gap:.3e}"
        );
    }
}
