//! The gong's frequency-dependent loss `σ₁` (`docs/dev/vk-mapping-plan.md` §2, bars §5.1).
//!
//! `VkSpec::sigma1` is forwarded to the linear half untouched, so the term is the plate's own
//! (`tests/plate_loss.rs`) and every coupled step carries it through `p.lin`. Born after the Python
//! gong, so there is no frozen answer: every bar is physics written out here. The loss work is
//! never read off the model — `σ₁` and `ρ_s = ρ_v e` are this file's constants, the supported
//! branch's `−L` and the free branch's gradient form are re-applied here from the mask — so a gong
//! that dissipates through the wrong operator, or not at all, cannot agree with a bar that uses its
//! own.
//!
//! `σ₁ = 0` is not given a trajectory bar: it builds no loss form at all (asserted below), which is
//! the plate's own skip, and every other gong, mallet-gong, bridge and room bar plus the viewer
//! freeze runs at `σ₁ = 0` against frozen or physical answers.

use physsynth_core::airbox::{AirBox, Params as RoomParams, Wall, C0_AIR, RHO0_AIR};
use physsynth_core::airbox_port::Spreading;
use physsynth_core::airbox_wrap::{RoomGrid, VkSeam};
use physsynth_core::connection::StringPlateBridge;
use physsynth_core::exciter::{raised_cosine_2d, triangular_pluck};
use physsynth_core::mallet::{MalletVkPlate, VkPlateParams};
use physsynth_core::plate::{
    Boundary, CoupleMethod, Domain, Params, Plate, PlateSpec, VkParams, VkPlate, VkSpec,
};
use physsynth_core::string_ideal::{Boundary as SB, IdealString, Params as StringParams};
use std::f64::consts::PI;

// Steel, as every von Kármán bar: the areal density is `RHO_V * e`, computed here.
const YOUNG: f64 = 2.0e11;
const NU: f64 = 0.3;
const RHO_V: f64 = 7800.0;
const E: f64 = 1.0e-3;

const BOUNDARIES: [Boundary; 2] = [Boundary::Supported, Boundary::Free];
const METHODS: [CoupleMethod; 3] = [
    CoupleMethod::Picard,
    CoupleMethod::Newton,
    CoupleMethod::Auto,
];

/// The balance rig of plan §8.1: a 0.4 × 0.35 m steel gong at 48 kHz, `N` 20.
fn spec(boundary: Boundary, sigma: f64, sigma1: f64, method: CoupleMethod) -> VkSpec {
    VkSpec {
        lx: 0.4,
        ly: 0.35,
        young: YOUNG,
        thickness: E,
        nu: NU,
        rho: RHO_V,
        fs: 48_000.0,
        n: 20,
        sigma,
        sigma1,
        boundary: Some(boundary),
        couple_method: Some(method),
        ..VkSpec::default()
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `−L v` on the live nodes, the rim held at zero — the supported branch's loss operator.
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

/// `vᵀ G v` for the free branch's `∫∇v·∇v`, assembled edge by edge from the live cells (a cell
/// with all four corners live hands each of its edges half a weight) — the bar's own copy.
fn free_gradient(p: &Params, v: &[f64]) -> f64 {
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
                s += 0.5 * (v[a] - v[b]) * (v[a] - v[b]);
            }
        }
    }
    s
}

/// The work `(σ₀ part, σ₁ part)` the two losses take out between `E^n` and `E^{n+1}`, with
/// `v = (u^{n+1} − u^{n−1}) / 2k`: `k ρ_s h² [2σ ‖v‖² + 2σ₁ ⟨−L v, v⟩]` supported,
/// `k ρ_s [2σ vᵀWv + 2σ₁ vᵀGv]` free. `σ`, `σ₁` and `ρ_s` are the caller's, not the model's.
fn loss_work(
    p: &Params,
    rho_s: f64,
    sigma: f64,
    sigma1: f64,
    u_next: &[f64],
    u_prev: &[f64],
) -> (f64, f64) {
    let v: Vec<f64> = u_next
        .iter()
        .zip(u_prev)
        .map(|(a, b)| (a - b) / (2.0 * p.k))
        .collect();
    match p.boundary {
        Boundary::Supported => {
            let w = p.k * rho_s * p.h * p.h;
            (
                w * 2.0 * sigma * dot(&v, &v),
                w * 2.0 * sigma1 * dot(&neg_lap(p, &v), &v),
            )
        }
        Boundary::Free => {
            let wv: f64 = v.iter().zip(&p.w).map(|(a, w)| w * a * a).sum();
            let w = p.k * rho_s;
            (
                w * 2.0 * sigma * wv,
                w * 2.0 * sigma1 * free_gradient(p, &v),
            )
        }
    }
}

fn unit(v: Vec<f64>) -> Vec<f64> {
    let m = v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    v.into_iter().map(|x| x / m).collect()
}

/// The strike of plan §8.1, peak `amp`: mode (1,1) on the supported gong (the strike that loads
/// the membrane most), an off-centre raised cosine on the free one.
fn strike(vp: &VkParams, amp: f64) -> Vec<f64> {
    let lin = &vp.lin;
    let shape = match lin.boundary {
        Boundary::Supported => {
            let full: Vec<f64> = lin
                .x
                .iter()
                .zip(&lin.y)
                .map(|(&x, &y)| (PI * x / lin.lx).sin() * (PI * y / lin.ly).sin())
                .collect();
            vp.to_live(&full)
        }
        Boundary::Free => {
            let full = raised_cosine_2d(&lin.x, &lin.y, (0.2, 0.17), 0.12, 1.0).expect("inside");
            vp.to_live(&full)
        }
    };
    unit(shape).into_iter().map(|x| amp * x).collect()
}

fn struck(s: &VkSpec, amp: f64) -> VkPlate {
    let vp = VkParams::new(s).expect("a valid gong");
    let u0 = strike(&vp, amp);
    let mut vk = VkPlate::new(vp);
    vk.set_state(&u0, &vec![0.0; u0.len()])
        .expect("a factorable start");
    vk
}

// -- the term reaches the linear half, and only when it is asked for ------------------------------

#[test]
fn sigma1_reaches_the_linear_half_and_zero_builds_no_loss_form() {
    for b in BOUNDARIES {
        let on = VkParams::new(&spec(b, 1.5, 2e-3, CoupleMethod::Auto)).unwrap();
        assert_eq!(on.lin.sigma1.to_bits(), 2e-3f64.to_bits(), "{b:?}");
        assert_eq!(on.lin.sigma.to_bits(), 1.5f64.to_bits(), "{b:?}: σ₀ moved");
        assert!(on.lin.loss.is_some(), "{b:?}: σ₁ > 0 built no loss form");
        let off = VkParams::new(&spec(b, 1.5, 0.0, CoupleMethod::Auto)).unwrap();
        assert!(off.lin.loss.is_none(), "{b:?}: σ₁ = 0 built a loss form");
    }
    let bad = VkSpec {
        sigma1: -1e-3,
        ..spec(Boundary::Supported, 0.0, 0.0, CoupleMethod::Auto)
    };
    assert!(VkParams::new(&bad).is_err(), "a negative σ₁ was accepted");
}

#[test]
fn a_linear_gong_with_sigma1_is_the_plate_with_sigma1_bit_for_bit() {
    // With the coupling off the gong is its linear half by construction, so the claim is about
    // the hand-off: the plate built here from the gong's `κ`, this file's `ρ_s` and the literal
    // `σ₁` steps to the same bits. A `σ₁` dropped, or sent into `σ`, cannot.
    for b in BOUNDARIES {
        let s = VkSpec {
            nonlinear: false,
            ..spec(b, 1.5, 2e-3, CoupleMethod::Auto)
        };
        let mut vk = struck(&s, 0.5 * E);
        let l = &vk.p.lin;
        let mut twin = Plate::new(
            Params::new(&PlateSpec {
                lx: 0.4,
                ly: 0.35,
                kappa: l.kappa,
                rho: RHO_V * E,
                fs: 48_000.0,
                n: 20,
                sigma: 1.5,
                sigma1: 2e-3,
                theta: l.theta,
                boundary: Some(b),
                domain: Some(Domain::Rectangle),
                nu: Some(NU),
                ..PlateSpec::default()
            })
            .unwrap(),
        );
        twin.set_state(&vk.u.clone(), &vec![0.0; vk.u.len()]);
        for _ in 0..200 {
            vk.step(None).unwrap();
            twin.step(None);
        }
        assert!(
            vk.u.iter()
                .zip(&twin.u)
                .all(|(a, c)| a.to_bits() == c.to_bits()),
            "{b:?}: the linear gong left the plate's trajectory"
        );
    }
}

// -- the books close with the membrane carrying over half the energy ------------------------------

#[test]
fn the_books_close_with_sigma1_at_a_high_membrane_share() {
    // Plan §8.1's rig. The energy drop per step must equal the two losses' work, through every
    // coupled solver, with the stress field holding at least half the energy at its peak — where
    // a σ₁ that reached the linear step but not the coupled one would show.
    let (sigma, sigma1) = (1.5, 2e-3);
    for b in BOUNDARIES {
        let amp = match b {
            Boundary::Supported => 4.5 * E,
            Boundary::Free => 3.0 * E,
        };
        for m in METHODS {
            let mut vk = struck(&spec(b, sigma, sigma1, m), amp);
            let e0 = vk.energy();
            let (mut worst, mut share, mut s1_total) = (0.0f64, 0.0f64, 0.0);
            for _ in 0..1500 {
                let before = vk.u_prev.clone();
                let e_before = vk.energy();
                vk.step(None).unwrap();
                assert!(vk.converged, "{b:?}/{m:?}: a step did not settle");
                let (w0, w1) = loss_work(&vk.p.lin, RHO_V * E, sigma, sigma1, &vk.u, &before);
                s1_total += w1;
                worst = worst.max((vk.energy() - e_before + w0 + w1).abs());
                share = share.max(vk.membrane_energy() / vk.energy());
            }
            eprintln!(
                "{b:?}/{m:?}: books {:.2e} of E0, σ₁ work {:.3e}, membrane share {share:.3}",
                worst / e0,
                s1_total / e0
            );
            assert!(share >= 0.5, "{b:?}/{m:?}: membrane share only {share:.3}");
            assert!(
                worst < 1e-10 * e0,
                "{b:?}/{m:?}: books miss by {:.2e} of E0",
                worst / e0
            );
            assert!(
                s1_total > 1e-3 * e0,
                "{b:?}/{m:?}: σ₁ did only {:.3e} of E0",
                s1_total / e0
            );
        }
    }
}

#[test]
fn sigma1_alone_is_passive() {
    for b in BOUNDARIES {
        let mut vk = struck(&spec(b, 0.0, 2e-3, CoupleMethod::Auto), 3.0 * E);
        let e0 = vk.energy();
        let mut prev = e0;
        for i in 0..1500 {
            vk.step(None).unwrap();
            let e = vk.energy();
            assert!(e <= prev + 1e-14 * e0, "{b:?}: E rose at step {i}");
            prev = e;
        }
        eprintln!("passive {b:?}: took {:.3e}", 1.0 - prev / e0);
        assert!(
            prev < 0.999 * e0,
            "{b:?}: σ₁ alone took only {:.2e}",
            1.0 - prev / e0
        );
    }
}

// -- the clients carry it ------------------------------------------------------------------------

#[test]
fn the_mallet_gong_s_books_close_with_sigma1() {
    // `mallet_gong.rs`'s shipped gong and mallet, σ = 0, elastic felt: the coupled total plus the
    // gong's σ₁ work is a constant. The contact column is `lin.lu.solve`, so it carries σ₁ in `A`.
    let sigma1 = 1e-2;
    let s = VkSpec {
        lx: 0.4,
        ly: 0.4,
        young: YOUNG,
        thickness: E,
        nu: NU,
        rho: RHO_V,
        fs: 48_000.0,
        n: 20,
        sigma1,
        ..VkSpec::default()
    };
    let p = VkParams::new(&s).unwrap();
    let mp = VkPlateParams::new(
        &p, 0.05, 5.0e4, 2.3, 0.0, 0.12, 0.16, 6.0, 0.0, 1e-12, 1e-14, 60, 1e-13, 20,
    )
    .expect("a valid mallet");
    let mut m = MalletVkPlate::new(mp, VkPlate::new(p), 0.0, 6.0);
    let e0 = m.energy();
    let (mut worst, mut s1_total, mut share) = (0.0f64, 0.0, 0.0f64);
    for _ in 0..2000 {
        let before = m.plate.u_prev.clone();
        m.step().expect("a gong step");
        let l = m.last.as_ref().expect("stepped");
        assert!(l.outer_converged && l.inner_converged);
        s1_total += loss_work(&m.plate.p.lin, RHO_V * E, 0.0, sigma1, &m.plate.u, &before).1;
        worst = worst.max((m.energy() + s1_total - e0).abs());
        share = share.max(m.plate.membrane_energy() / e0);
    }
    eprintln!(
        "mallet: books {:.2e}, σ₁ work {:.3e}, share {share:.3}",
        worst / e0,
        s1_total / e0
    );
    assert!(share > 0.01, "the strike never left the linear regime");
    assert!(worst < 1e-10 * e0, "books miss by {:.2e} of E0", worst / e0);
    assert!(
        s1_total > 1e-3 * e0,
        "σ₁ did only {:.3e} of E0",
        s1_total / e0
    );
}

#[test]
fn the_string_gong_bridge_s_books_close_with_sigma1() {
    // `connection_plate.rs`'s thin sheet on a plucked string; the bridge steps the gong through
    // `VkPlate::step`, so σ₁ arrives with no code of its own. Lossless string, σ = 0.
    let sigma1 = 2e-3;
    let fs = 200.0 * 100.0 / 0.9;
    let e_sheet = 1.0e-4;
    for b in BOUNDARIES {
        let vk = VkPlate::new(
            VkParams::new(&VkSpec {
                lx: 0.4,
                ly: 0.4,
                young: YOUNG,
                thickness: e_sheet,
                nu: NU,
                rho: RHO_V,
                fs,
                n: 16,
                sigma1,
                boundary: Some(b),
                ..VkSpec::default()
            })
            .unwrap(),
        );
        let sp = StringParams::new(1.0, 200.0, 0.005, fs, 100, 0.0, Some((SB::Fixed, SB::Free)))
            .unwrap();
        let s = IdealString::new(sp.clone());
        let mut br = StringPlateBridge::new(s, vk, 3000.0, None).expect("inside the guard");
        let u0 = triangular_pluck(&sp.grid(), sp.l, 0.137 * sp.l, 1e-3).unwrap();
        br.string_mut().set_state(&u0, &vec![0.0; u0.len()]);
        let e0 = br.energy();
        let (mut worst, mut s1_total, mut peak) = (0.0f64, 0.0, 0.0f64);
        for _ in 0..2500 {
            let before = br.plate().u_prev.clone();
            br.step(&mut ()).unwrap();
            assert!(br.plate().converged, "{b:?}: a step did not settle");
            let pl = br.plate();
            s1_total += loss_work(&pl.p.lin, RHO_V * e_sheet, 0.0, sigma1, &pl.u, &before).1;
            worst = worst.max((br.energy() + s1_total - e0).abs());
            peak = pl.u.iter().fold(peak, |a, v| a.max(v.abs()));
        }
        eprintln!(
            "bridge {b:?}: books {:.2e}, σ₁ work {:.3e}",
            worst / e0,
            s1_total / e0
        );
        assert!(
            peak > e_sheet,
            "{b:?}: the plate never reached its thickness"
        );
        assert!(
            worst < 1e-10 * e0,
            "{b:?}: books miss by {:.2e}",
            worst / e0
        );
        assert!(
            s1_total > 1e-4 * e0,
            "{b:?}: σ₁ did only {:.3e}",
            s1_total / e0
        );
    }
}

#[test]
fn the_gong_in_a_room_s_books_close_with_sigma1() {
    // `airbox_vk.rs`'s room and gong. The seam's operator is `theta_matrix(&plate.p.lin)`, which
    // adds `σ₁kG`, and its right-hand side the plate's own `step_rhs`. Scene total (gong + air +
    // radiated channel) plus the gong's σ₁ work is a constant; the port's two ledgers agree.
    let (fs, sigma1) = (8000.0, 5e-3);
    let h_air = C0_AIR * 3.0f64.sqrt() / (0.9 * fs);
    for b in BOUNDARIES {
        for suspended in [false, true] {
            let l = [12.0 * h_air, 11.0 * h_air, 9.0 * h_air];
            let mut room = AirBox::new(
                RoomParams::new(l, fs, h_air, [Wall::Rigid; 6], None, RHO0_AIR, C0_AIR).unwrap(),
            );
            let vp = VkParams::new(&VkSpec {
                lx: 0.30,
                ly: 0.30,
                young: YOUNG,
                thickness: E,
                nu: NU,
                rho: RHO_V,
                fs,
                n: 8,
                sigma1,
                boundary: Some(b),
                ..VkSpec::default()
            })
            .unwrap();
            let u0: Vec<f64> = {
                let lin = &vp.lin;
                let w = 0.20 * lin.lx;
                let full: Vec<f64> = lin
                    .x
                    .iter()
                    .zip(&lin.y)
                    .map(|(&x, &y)| {
                        let (dx, dy) = (x - 0.5 * lin.lx, y - 0.5 * lin.ly);
                        2.0 * E * (-((dx * dx + dy * dy) / (w * w))).exp()
                    })
                    .collect();
                vp.to_live(&full)
            };
            let seam = VkSeam::new(VkPlate::new(vp));
            let mut g = if suspended {
                RoomGrid::suspended(seam, &mut room, "z", 4, None, Spreading::Bilinear)
            } else {
                RoomGrid::baffled(seam, &mut room, "z0", None, Spreading::Bilinear)
            }
            .expect("a legal mounting");
            g.set_state(&u0, &vec![0.0; u0.len()]);
            let total = |g: &RoomGrid<VkSeam>, room: &AirBox| g.energy() + room.energy();
            let e0 = total(&g, &room);
            let (mut worst, mut s1_total) = (0.0f64, 0.0);
            for _ in 0..300 {
                let before = g.seam.plate.u_prev.clone();
                g.step(&mut room, None).expect("a legal step");
                room.step();
                let pl = &g.seam.plate;
                assert!(
                    pl.converged,
                    "{b:?} suspended={suspended}: a step did not settle"
                );
                s1_total += loss_work(&pl.p.lin, RHO_V * E, 0.0, sigma1, &pl.u, &before).1;
                worst = worst.max((total(&g, &room) + s1_total - e0).abs());
            }
            eprintln!(
                "room {b:?} suspended={suspended}: books {:.2e}, σ₁ work {:.3e}",
                worst / e0,
                s1_total / e0
            );
            assert!(
                worst < 1e-10 * e0,
                "{b:?} suspended={suspended}: books miss {:.2e}",
                worst / e0
            );
            assert!(
                s1_total > 1e-3 * e0,
                "{b:?} suspended={suspended}: σ₁ did {:.3e}",
                s1_total / e0
            );
            let gap = (g.radiated_energy() - room.injected).abs();
            assert!(
                gap <= 1e-12 * g.radiated_energy().abs(),
                "{b:?} suspended={suspended}: ledger gap {gap:.3e}"
            );
        }
    }
}
