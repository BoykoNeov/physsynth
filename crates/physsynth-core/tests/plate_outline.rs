//! Native bars for the curved-outline plate — model #5g, the guitar and the disk.
//!
//! Carried from `tests/test_guitar_plate.py` (retirement plan §28). **Only one detector here can
//! falsify an outline.** Energy conservation is geometry-blind (a wrong outline conserves
//! perfectly); the supported branch's spectrum on any outline is the membrane's squared, an
//! identity; the rigid-body nullspace is necessary, not sufficient. So the anchor is a **derived
//! free-circular-plate oracle**, run through the same staircased-mask machinery the guitar uses.
//!
//! **Three outside referees retire here, with their numbers**, frozen on 2026-09-29 before the
//! Python was deleted (NumPy 2.4.6, SciPy 1.17.1, wheel reinstalled) into
//! `tests/reference/guitar_plate.json`:
//!
//! * the outline's **pre-2026-08-28 NumPy spelling** — the vectorised `sin`/`cos` profile the
//!   shipped geometry was validated against — as a digest of every shipped outline's mask on every
//!   grid the Python used, before and after the prune, and its values for the degenerate lens;
//! * **SciPy's Bessel functions** (Cephes/AMOS) evaluating each derived root's own Rayleigh
//!   quotient — an implementation independent of the analysis crate's `bessel` module;
//! * **LAPACK's dense eigenvalues** of three staircased disks (§24.3's rule), with each `mu_max`.

use physsynth_analysis::bessel::{iv, ivp, jn, jvp};
use physsynth_analysis::modal::{free_circular_plate_lambda_roots, free_circular_plate_lambdas};
use physsynth_core::eigs::eigsh_shift_invert;
use physsynth_core::ops2d::{
    biharmonic_from_mask, cells_per_node, disk_mask, free_plate_stiffness,
    free_plate_stiffness_from_mask, guitar_area, guitar_half_width, guitar_mask,
    laplacian_from_mask, prune_to_area_carrying, Mask,
};
use physsynth_core::plate::{linspace0, Boundary, Domain, ParamError, Params, Plate, PlateSpec};
use physsynth_core::sparse::Csr;
use serde_json::Value;

const NU: f64 = 0.3;
/// The acceptance bar, unchanged — see CLAUDE.md.
const DRIFT_TOL: f64 = 1e-10;
/// The shipped outline's defaults.
const WAIST: f64 = 0.42;
const ASYM: f64 = 0.30;

fn reference() -> Value {
    serde_json::from_str(include_str!("reference/guitar_plate.json")).expect("the frozen record")
}

fn f(v: &Value) -> f64 {
    v.as_f64().expect("a number")
}

fn u(v: &Value) -> u128 {
    v.as_u64().expect("an integer") as u128
}

// -- geometry fixtures -----------------------------------------------------------------------

/// `(count, Σ (i+1), Σ (i+1)²)` over the live flat indices, row-major — the recorded digest.
fn digest(m: &Mask) -> [u128; 3] {
    let mut d = [0u128; 3];
    for (i, _) in m.flags().iter().enumerate().filter(|(_, &b)| b) {
        let k = i as u128 + 1;
        d[0] += 1;
        d[1] += k;
        d[2] += k * k;
    }
    d
}

/// `_outline_grid`: the plate's own node grid, `x` centred, `Ly` snapped to whole cells.
fn outline_grid(lx: f64, ly: f64, n: usize) -> (Vec<f64>, Vec<f64>, f64, usize) {
    let h = lx / n as f64;
    let ny = ((ly / h).round() as usize).max(1);
    let ly_snapped = ny as f64 * h;
    let xs = linspace0(lx, n + 1);
    let ys = linspace0(ly_snapped, ny + 1);
    let mut x = Vec::with_capacity((ny + 1) * (n + 1));
    let mut y = Vec::with_capacity((ny + 1) * (n + 1));
    for &yv in &ys {
        for &xv in &xs {
            x.push(xv - 0.5 * lx);
            y.push(yv);
        }
    }
    (x, y, ly_snapped, ny)
}

/// `_disk`: a pruned staircased disk of radius `a` on an `N`-cell bounding box, plus `h`.
fn disk(n: usize, a: f64) -> (Mask, f64) {
    let h = 2.0 * a / n as f64;
    let xs: Vec<f64> = (0..=n).map(|i| (i as f64 - n as f64 / 2.0) * h).collect();
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for &yv in &xs {
        for &xv in &xs {
            x.push(xv);
            y.push(yv);
        }
    }
    (
        prune_to_area_carrying(&disk_mask(&x, &y, a, n + 1, n + 1)).0,
        h,
    )
}

/// `_guitar`'s raw, unpruned mask and its `h`.
fn guitar_raw(n: usize) -> (Mask, f64) {
    let (length, width) = (0.48, 0.37);
    let h = width / n as f64;
    let (nx, ny) = ((width / h).round() as usize, (length / h).round() as usize);
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for j in 0..=ny {
        for i in 0..=nx {
            x.push((i as f64 - nx as f64 / 2.0) * h);
            y.push(j as f64 * h);
        }
    }
    (
        guitar_mask(&x, &y, length, width, WAIST, ASYM, ny + 1, nx + 1),
        h,
    )
}

fn guitar(n: usize) -> (Mask, f64) {
    let (raw, h) = guitar_raw(n);
    (prune_to_area_carrying(&raw).0, h)
}

fn diag(w: &Csr) -> Vec<f64> {
    (0..w.nrows()).map(|i| w.get(i, i)).collect()
}

/// The shift-invert target for a unit-radius disk: `-1e-3 mu_1`, `mu_1 = (5.358 / a²)² ≈ 28.7` —
/// `free_plate_low_eigenfrequencies`' convention. The Python used ARPACK at `-1e-8`, three rigid
/// modes a hair away; the native Krylov solver hits its 300-iteration cap there — measured on every
/// disk (N = 32, 64, 128; 6 and 11 pairs) and every circle (N = 32, 33, 64, 128) this file solves
/// (retirement plan §28). The shift is the instrument's setting, not the physics.
const SIGMA: f64 = -0.03;

/// The lowest `count` generalized eigenvalues of the free plate on `mask`.
fn lowest_mu(mask: &Mask, h: f64, count: usize) -> Vec<f64> {
    let (k, w, _) = free_plate_stiffness_from_mask(mask, h, NU, 1.0, 1.0, None, None);
    eigsh_shift_invert(&k, Some(&w), SIGMA, count)
        .expect("the shifted pencil factors")
        .values
}

/// `_elastic_lambdas`: the lowest `n` elastic `Λ = ω a²/κ`, rigid modes skipped — and the PSD tier
/// asserted on the way past. The rigid bar is RELATIVE: `mu[0:3]` come out of shift-invert with
/// arbitrary sign and a size that grows with the problem.
fn elastic_lambdas(mu: &[f64], n: usize, a2: f64) -> Vec<f64> {
    assert!(mu[3] > 0.0, "no positive elastic mode: {:e}", mu[3]);
    let rigid = mu[..3].iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(
        rigid < 1e-6 * mu[3],
        "rigid modes not clean: {:e}",
        rigid / mu[3]
    );
    mu[3..3 + n]
        .iter()
        .map(|m| a2 * m.max(0.0).sqrt())
        .collect()
}

/// Native against LAPACK, in units of the dense floor `eps · mu_max`; returns the worst.
fn against_lapack(what: &str, n: i64, got: &[f64]) -> f64 {
    let rec = reference();
    let row = rec["lapack"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["what"] == what && r["N"] == n)
        .expect("a recorded pencil");
    let floor = f64::EPSILON * f(&row["mu_max"]);
    let want: Vec<f64> = row["mu"].as_array().unwrap().iter().map(f).collect();
    let worst = got
        .iter()
        .zip(&want)
        .skip(3)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f64, f64::max)
        / floor;
    // Measured 0.26 (disk 32), 0.44 (disk 64), 0.27 (circle 33).
    assert!(
        worst < 20.0,
        "{what} N={n}: {worst:.2} dense floors from LAPACK"
    );
    worst
}

fn free_spec(lx: f64, ly: f64, n: i64, domain: Domain) -> PlateSpec {
    PlateSpec {
        lx,
        ly,
        kappa: 1.0,
        rho: 2.0,
        fs: 20_000.0,
        n,
        boundary: Some(Boundary::Free),
        domain: Some(domain),
        ..PlateSpec::default()
    }
}

// -- the outline, pinned against its pre-port spelling ------------------------------------------

#[test]
fn the_scalar_libm_spelling_moved_no_node_of_any_shipped_outline() {
    // `guitar_half_width` moved from NumPy's vectorised sin/cos to the scalar libm on 2026-08-28;
    // the OLD spelling's masks were frozen before NumPy left. A transcription slip in the profile
    // would be reproduced by every parity test and seen by no energy, nullspace or spectrum bar —
    // this is where it lands. Both prunes too: the prune iterates, so one moved node can take a
    // neighbour with it. The four shipped outlines clear a rounding difference by ~1.9e7 ulps, so
    // this holds on any libm.
    let rec = reference();
    let rows = rec["outline_masks"].as_array().unwrap();
    assert_eq!(rows.len(), 80, "2 plates x 4 outlines x 10 grids");
    for r in rows {
        let r = r.as_array().unwrap();
        let (lx, ly, waist, asym) = (f(&r[0]), f(&r[1]), f(&r[2]), f(&r[3]));
        let n = r[4].as_u64().unwrap() as usize;
        let (x, y, ly_s, ny) = outline_grid(lx, ly, n);
        let now = guitar_mask(&x, &y, ly_s, lx, waist, asym, ny + 1, n + 1);
        let want_raw = [u(&r[5]), u(&r[6]), u(&r[7])];
        let want_pruned = [u(&r[8]), u(&r[9]), u(&r[10])];
        assert_eq!(
            digest(&now),
            want_raw,
            "Lx={lx} waist={waist} asym={asym} N={n}: the shipped outline is not the validated one"
        );
        assert_eq!(
            digest(&prune_to_area_carrying(&now).0),
            want_pruned,
            "pruned, N={n}"
        );
    }
}

#[test]
fn the_degenerate_lens_agrees_with_its_old_spelling_to_a_few_ulps() {
    // The lens `0.5 Lx sin(pi t)` passes exactly through grid nodes at rational t, so mask
    // equality there would be a claim about a CPU's sin. What the pin is FOR is a transcription
    // slip, which moves the profile by orders of magnitude: so the lens gets the strongest claim it
    // can support, the half-width within a few ulps of NumPy's recorded vectorised values.
    let rec = reference();
    for grid in rec["lens"].as_array().unwrap() {
        let n = grid["N"].as_u64().unwrap();
        let t = grid["t"].as_array().unwrap();
        let half = grid["half"].as_array().unwrap();
        let mut worst = 0.0f64;
        for (tv, hv) in t.iter().zip(half) {
            let (now, before) = (guitar_half_width(f(tv), 0.0, 0.0), f(hv));
            let ulp = (before.abs().next_up() - before.abs()).max(5e-324);
            worst = worst.max((now - before).abs() / ulp);
        }
        // Measured 0 ulps on every grid here (Windows); 4 leaves room for another platform's libm.
        assert!(
            worst <= 4.0,
            "N={n}: the lens differs by {worst:.1} ulps — a different formula"
        );
    }
}

// -- the assembly: the mask generalization is a strict superset --------------------------------

/// The #5o seven-grid survey: some grids where two assemblies agree to the bit, some where they
/// differ by ~2e-16.
const GRIDS: [(usize, usize, f64); 7] = [
    (12, 12, 1.0 / 12.0),
    (24, 24, 1.0 / 24.0),
    (20, 14, 0.05),
    (17, 17, 1.0 / 17.0),
    (13, 9, 0.62 / 13.0),
    (16, 16, 0.7 / 16.0),
    (11, 7, 0.31 / 11.0),
];

#[test]
fn the_rectangle_builder_is_the_masked_builder_on_a_full_mask() {
    // VERDICT carried as a premise. The Python compared a Kronecker assembly with the masked one,
    // bit for bit, over 84 cases — and found the twist coefficient had to be (1/h)*(1/h), not
    // 1/(h*h), on the one grid (h = 0.05) where they differ. In Rust `free_plate_stiffness`
    // DELEGATES to `free_plate_stiffness_from_mask` on an all-live mask, so the equality compares a
    // computation with itself (finding #78). Pinned so a second implementation cannot creep back.
    let splits: [Option<[f64; 4]>; 3] = [
        None,
        Some([1.0, 0.5, 0.35, 0.22]),
        Some([2.4, 0.13, 0.15, 0.30]),
    ];
    for (nx, ny, h) in GRIDS {
        let mask = Mask::new(ny + 1, nx + 1, vec![true; (ny + 1) * (nx + 1)]);
        for nu in [0.3, 0.0, 0.49, -0.5] {
            for s in splits {
                let (gx, gy, c, t) = match s {
                    None => (1.0, 1.0, None, None),
                    Some(g) => (g[0], g[1], Some(g[2]), Some(g[3])),
                };
                let (k1, w1, i1) = free_plate_stiffness(nx, ny, h, nu, gx, gy, c, t);
                let (k2, w2, i2) = free_plate_stiffness_from_mask(&mask, h, nu, gx, gy, c, t);
                assert_eq!(k1.indices(), k2.indices());
                assert_eq!(k1.data(), k2.data(), "{nx}x{ny} h={h} nu={nu} {s:?}");
                assert_eq!(diag(&w1), diag(&w2));
                assert_eq!(i1, i2);
            }
        }
    }
}

#[test]
fn the_area_weight_is_the_trapezoidal_rule_restated() {
    // h² (live cells)/4 is h², h²/2, h²/4 — not a new convention.
    let mask = Mask::new(6, 8, vec![true; 48]);
    let counts = cells_per_node(&mask);
    assert_eq!((counts[3 * 8 + 4], counts[4], counts[0]), (4, 2, 1));
    let (_, w, _) = free_plate_stiffness_from_mask(&mask, 0.05, NU, 1.0, 1.0, None, None);
    let w = diag(&w);
    let h2 = 0.05 * 0.05;
    assert_eq!((w[3 * 8 + 4], w[4], w[0]), (h2, 0.5 * h2, 0.25 * h2));
}

// -- the trap: a curved outline makes nodes that carry no area ---------------------------------

#[test]
fn a_curved_outline_makes_massless_nodes_and_the_prune_removes_them() {
    // Without the prune the mass matrix is SINGULAR: a one-node spike's area weight is exactly 0.
    let (raw, h) = guitar_raw(16);
    let massless = |m: &Mask| {
        let c = cells_per_node(m);
        m.flags()
            .iter()
            .zip(&c)
            .filter(|(&alive, &n)| alive && n == 0)
            .count()
    };
    assert!(
        massless(&raw) > 0,
        "expected massless spikes on a coarse outline"
    );
    let (pruned, dropped) = prune_to_area_carrying(&raw);
    assert!(dropped > 0);
    assert_eq!(massless(&pruned), 0);
    let (_, w, _) = free_plate_stiffness_from_mask(&pruned, h, NU, 1.0, 1.0, None, None);
    let wmin = diag(&w).into_iter().fold(f64::INFINITY, f64::min);
    assert!(
        wmin > 0.0,
        "a pruned mask must have a strictly positive mass diagonal"
    );
}

#[test]
fn the_prune_is_idempotent_and_reaches_a_fixed_point() {
    // Dropping a node can orphan its neighbour, so the rule is a fixed point, not a single pass.
    for n in [16usize, 24, 32, 40, 56] {
        let (mask, _) = guitar(n);
        let (again, dropped) = prune_to_area_carrying(&mask);
        assert_eq!(dropped, 0, "N={n}");
        assert_eq!(again.flags(), mask.flags(), "N={n}");
    }
}

#[test]
fn every_pruned_node_lies_at_the_rim() {
    // The prune rule is TOPOLOGICAL, so it is checked GEOMETRICALLY — and the measured depth is
    // asserted, not merely the absence of a refusal: the lower bound proves the check ran over a
    // non-empty set. Measured 0.750, 0.733, 0.712, 0.704, 0.703 h at N = 20…80, as Python did.
    for n in [20i64, 28, 40, 56, 80] {
        let p = Params::new(&free_spec(0.37, 0.48, n, Domain::Guitar)).unwrap();
        assert!(
            p.n_pruned > 0,
            "N={n}: nothing pruned, the rim check ran over nothing"
        );
        assert!(
            p.prune_depth_max > 0.0,
            "N={n}: a pruned node was not inside the outline"
        );
        assert!(p.prune_depth_max <= 1.0001 * p.h, "N={n}");
        let depth = p.prune_depth_max / p.h;
        assert!(
            0.6 < depth && depth < 0.85,
            "N={n}: depth {depth:.3} h left the 0.70-0.75 band"
        );
        assert_eq!(p.mask.n_live(), p.n_live);
    }
}

#[test]
fn a_pinched_outline_is_refused_rather_than_silently_two_plates() {
    // A deep waist on a coarse grid separates the bouts: two plates, a 6-D nullspace. Pinned to
    // the CONNECTIVITY refusal, so a mid-plate-prune refusal could not stand in for it.
    let mut s = free_spec(0.37, 0.48, 8, Domain::Guitar);
    s.waist = 0.97;
    let e = Params::new(&s).expect_err("a pinched guitar");
    assert!(matches!(e, ParamError::Disconnected { .. }), "{e:?}");
    assert!(e.to_string().contains("disconnected pieces"), "{e}");
}

// -- the oracle: a DERIVED free circular plate, and its self-checks ------------------------------

#[test]
fn the_derived_frequency_equation_admits_the_rigid_body_modes() {
    // W = 1 (n = 0) and W = rho (n = 1) must annihilate both free-edge lines exactly: a
    // translation and a tilt cost nothing. Catches a sign error in either line.
    for (n, w, w1, w2, w3) in [(0.0f64, 1.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 0.0, 0.0)] {
        let moment = w2 + NU * (w1 - n * n * w);
        let shear = w3 + w2 - (1.0 + n * n * (2.0 - NU)) * w1 + n * n * (3.0 - NU) * w;
        assert!(
            moment == 0.0 && shear == 0.0,
            "n={n}: a rigid-body mode is not a root"
        );
    }
}

/// SciPy's Rayleigh quotients, recorded: `(n, lam, quotient)`.
fn recorded_quotients() -> Vec<(i32, f64, f64)> {
    reference()["rayleigh"]
        .as_array()
        .unwrap()
        .iter()
        .map(|q| {
            (
                q["n"].as_i64().unwrap() as i32,
                f(&q["lam"]),
                f(&q["quotient"]),
            )
        })
        .collect()
}

#[test]
fn every_derived_root_returns_lambda_to_the_fourth_in_the_plate_energy() {
    // The decisive check: each root's own eigenfunction, put back into the bending energy, gives
    // P/M = lam⁴ (the FOURTH power: omega = kappa k²). Computed here with the analysis crate's
    // Bessel functions and compared with the same quotient SciPy's recorded — two independent
    // Bessel implementations — as well as with lam⁴.
    let recorded = recorded_quotients();
    let mut k = 0;
    let rho: Vec<f64> = (0..40_000).map(|i| (i as f64 + 0.5) / 40_000.0).collect();
    for n in 0..4i32 {
        let roots = free_circular_plate_lambda_roots(NU, n, 9.0, 20_000).unwrap();
        for &lam in &roots[..2] {
            let nn = f64::from(n * n);
            let mut m = [[0.0f64; 2]; 2];
            for (col, g) in [jvp as fn(i32, f64, u32) -> f64, ivp].iter().enumerate() {
                let (d0, d1) = (g(n, lam, 0), g(n, lam, 1) * lam);
                let (d2, d3) = (g(n, lam, 2) * lam.powi(2), g(n, lam, 3) * lam.powi(3));
                m[0][col] = d2 + NU * (d1 - nn * d0);
                m[1][col] = d3 + d2 - (1.0 + nn * (2.0 - NU)) * d1 + nn * (3.0 - NU) * d0;
            }
            let scale = if lam > 1.0 { (-lam).exp() } else { 1.0 };
            m[0][1] *= scale;
            m[1][1] *= scale;
            // The null vector of a (numerically) singular 2x2, from its larger row.
            let r = if m[0][0].hypot(m[0][1]) >= m[1][0].hypot(m[1][1]) {
                0
            } else {
                1
            };
            let (a, mut b) = (m[r][1], -m[r][0]);
            b *= scale;
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for &rh in &rho {
                let z = lam * rh;
                let w = a * jn(n, z) + b * iv(n, z);
                let w1 = lam * (a * jvp(n, z, 1) + b * ivp(n, z, 1));
                let w2 = lam * lam * (a * jvp(n, z, 2) + b * ivp(n, z, 2));
                let lap = w2 + w1 / rh - nn * w / (rh * rh);
                let gauss =
                    w2 * (w1 / rh - nn * w / (rh * rh)) - nn * (w1 / rh - w / (rh * rh)).powi(2);
                num += (lap * lap - 2.0 * (1.0 - NU) * gauss) * rh;
                den += w * w * rh;
            }
            let quotient = num / den;
            assert!(
                (quotient / lam.powi(4) - 1.0).abs() < 3e-3,
                "n={n} lam={lam:.5}: quotient {quotient:.4}, not lam^4 — this root is not a mode"
            );
            let (rn, rlam, rq) = recorded[k];
            assert_eq!(rn, n);
            assert!(
                (lam / rlam - 1.0).abs() < 1e-12,
                "the root moved: {lam} vs {rlam}"
            );
            // Measured <= 1.4e-14 across the eight roots; |quotient / lam⁴ - 1| was <= 1.8e-9
            // (the 40,000-point quadrature), so the Python's 3e-3 bar had six orders to spare.
            assert!(
                (quotient / rq - 1.0).abs() < 1e-12,
                "n={n}: the crate's Bessel functions disagree with SciPy's: {quotient} vs {rq}"
            );
            k += 1;
        }
    }
    assert_eq!(k, recorded.len());
}

// -- the anchor: the masked assembly against that oracle ---------------------------------------

#[test]
fn a_staircased_disk_matches_the_derived_oracle_and_converges_at_first_order() {
    // Seven modes, converging FROM ABOVE (a staircased disk is smaller) at O(h) — staircasing taxes
    // a 4th-order operator the same first order it taxed the membrane. Every n >= 1 mode is a
    // degenerate PAIR, which the oracle's list carries.
    let (target, _) = free_circular_plate_lambdas(NU, 7, 8).unwrap();
    let mut errs = Vec::new();
    for n in [32usize, 64, 128] {
        let (mask, h) = disk(n, 1.0);
        let mu = lowest_mu(&mask, h, 11);
        if n <= 64 {
            against_lapack("disk", n as i64, &mu);
        }
        let lam = elastic_lambdas(&mu, 7, 1.0);
        assert!(
            lam.iter().zip(&target).all(|(l, t)| l > t),
            "N={n}: must ring sharp"
        );
        errs.push(
            lam.iter()
                .zip(&target)
                .map(|(l, t)| (l / t - 1.0).abs())
                .sum::<f64>()
                / 7.0,
        );
    }
    // Measured 8.5%, 4.0%, 2.0%: ratios 2.12 and 2.00.
    assert!(errs[0] < 0.12, "coarse disk off by {:.1}%", errs[0] * 100.0);
    assert!(errs[2] < 0.03, "fine disk off by {:.1}%", errs[2] * 100.0);
    for w in errs.windows(2) {
        let ratio = w[0] / w[1];
        assert!(
            1.25 < ratio && ratio < 2.6,
            "improved {ratio:.2}x; O(h) ~2, O(h²) ~4"
        );
    }
}

#[test]
fn the_shipped_circle_path_matches_the_derived_oracle() {
    // The oracle must anchor `domain = circle` itself — its centring, snapping and prune — not only
    // the bare mask helper. Odd N is allowed: the centre falls between nodes and the spectrum does
    // not care. The area deficit is the error's leading term, and the two track each other
    // (measured |err + deficit| 4.4e-3, 6.2e-3, 2.3e-3, 1.0e-3 at N = 32, 33, 64, 128).
    let (target, _) = free_circular_plate_lambdas(NU, 7, 8).unwrap();
    let mut prev: Option<f64> = None;
    for n in [32i64, 33, 64, 128] {
        let mut s = free_spec(2.0, 2.0, n, Domain::Circle);
        s.fs = 200_000.0;
        let p = Params::new(&s).unwrap();
        let mu = eigsh_shift_invert(&p.stiffness, p.mass.as_ref(), SIGMA, 11)
            .unwrap()
            .values;
        if n == 33 {
            against_lapack("circle", 33, &mu);
        }
        let a2 = (0.5 * p.lx).powi(2);
        let lam: Vec<f64> = mu[3..10].iter().map(|m| a2 * m.max(0.0).sqrt()).collect();
        assert!(
            lam.iter().zip(&target).all(|(l, t)| l > t),
            "N={n}: must ring sharp"
        );
        let err = lam
            .iter()
            .zip(&target)
            .map(|(l, t)| (l / t - 1.0).abs())
            .sum::<f64>()
            / 7.0;
        assert!(
            (err + p.area_deficit).abs() < 0.012,
            "N={n}: error {:.3}% and deficit {:.3}% came apart",
            err * 100.0,
            p.area_deficit * 100.0
        );
        // 33 refines 32 by only 3%, so it is not a rate step.
        if n != 33 {
            if let Some(pe) = prev {
                assert!(err < pe, "N={n}: not converging");
            }
            prev = Some(err);
        }
    }
    assert!(
        prev.unwrap() < 0.03,
        "the finest disk is off by {:.2}%",
        prev.unwrap() * 100.0
    );
}

#[test]
fn the_degenerate_pairs_split_and_the_exact_answer_is_zero() {
    // A ZERO-valued oracle: a square grid relates a pair's two members by no symmetry. Shrinking
    // but not monotone in N, so a ceiling that shrinks with h, never monotonicity.
    for (n, ceiling) in [(32usize, 0.02), (64, 0.012), (128, 0.006)] {
        let (mask, h) = disk(n, 1.0);
        let lam = elastic_lambdas(&lowest_mu(&mask, h, 6), 2, 1.0);
        let split = 2.0 * (lam[1] - lam[0]).abs() / (lam[1] + lam[0]);
        // Measured 1.0%, 0.52%, 0.013% — as the Python recorded.
        assert!(
            split < ceiling,
            "N={n}: the m=2 pair splits by {:.3}%",
            split * 100.0
        );
    }
}

// -- the supported branch: a NEGATIVE result, stated as the identity it is ----------------------

#[test]
fn a_supported_curved_plate_is_the_membrane_squared_and_therefore_says_nothing() {
    // B = L @ L on any mask, so eig(B) = eig(-L)². This CANNOT fail; it is asserted so nobody
    // mistakes it for evidence, and it is why the supported branch refuses a curved outline.
    let (mask, h) = guitar(24);
    let (l, _) = laplacian_from_mask(&mask, h);
    let (b, _) = biharmonic_from_mask(&mask, h);
    let lam_l = eigsh_shift_invert(&l.scaled(-1.0), None, 0.0, 6)
        .unwrap()
        .values;
    let lam_b = eigsh_shift_invert(&b, None, 0.0, 6).unwrap().values;
    for (lb, ll) in lam_b.iter().zip(&lam_l) {
        let want = ll * ll;
        assert!(
            (lb - want).abs() <= 1e-8 * want,
            "B = L @ L is not an identity: {lb} vs {want}"
        );
    }
    let mut s = free_spec(0.37, 0.48, 16, Domain::Guitar);
    s.boundary = Some(Boundary::Supported);
    let e = Params::new(&s).expect_err("a supported guitar is refused");
    assert!(
        matches!(e, ParamError::CurvedSupported(Domain::Guitar)),
        "{e:?}"
    );
    assert!(
        e.to_string().contains("offered on boundary='free' only"),
        "{e}"
    );
}

// -- tier 1: the ledger, a REGRESSION test here and not evidence -------------------------------

/// A plate on `domain` struck at one node, `(rows / 3, cols / 2)` of its grid.
fn struck(domain: Domain, sigma: f64) -> Plate {
    let ly = if domain == Domain::Guitar { 0.48 } else { 0.37 };
    let mut s = free_spec(0.37, ly, 24, domain);
    s.kappa = 2.0;
    s.sigma = sigma;
    let p = Params::new(&s).unwrap();
    let (rows, cols) = (p.mask.nrows(), p.mask.ncols());
    let hit = (rows / 3) * cols + cols / 2;
    assert!(p.mask.flags()[hit], "the struck node must be live");
    let live = p.index_map[hit] as usize;
    let mut u0 = vec![0.0; p.n_live];
    u0[live] = 1e-3;
    let mut plate = Plate::new(p);
    plate.set_state(&u0, &vec![0.0; u0.len()]);
    plate
}

#[test]
fn a_lossless_outline_plate_conserves_energy() {
    // A masked assembly could plausibly break SYMMETRY, which the ledger does see; it cannot see
    // geometry.
    for domain in [Domain::Guitar, Domain::Circle] {
        let mut p = struck(domain, 0.0);
        let e0 = p.energy();
        let mut drift = 0.0f64;
        for _ in 0..400 {
            p.step(None);
            drift = drift.max((p.energy() - e0).abs() / e0);
        }
        assert!(drift < DRIFT_TOL, "{domain:?}: drift {drift:.3e}");
    }
}

#[test]
fn a_lossy_outline_plate_is_passive() {
    for domain in [Domain::Guitar, Domain::Circle] {
        let mut p = struck(domain, 4.0);
        let e0 = p.energy();
        let mut prev = e0;
        for _ in 0..300 {
            p.step(None);
            let now = p.energy();
            assert!(
                now <= prev * (1.0 + 1e-12),
                "{domain:?}: a lossy plate gained energy"
            );
            prev = now;
        }
        assert!(
            prev < e0,
            "{domain:?}: a plate that never moved is monotone too"
        );
    }
}

// -- what the plate REPORTS about its own outline ------------------------------------------------

#[test]
fn the_area_deficit_is_reported_and_shrinks_under_refinement() {
    // The staircase error's leading term: reported, never applied — dividing it out would make a
    // coarse plate look converged.
    let mut prev: Option<f64> = None;
    for n in [20i64, 40, 80] {
        let p = Params::new(&free_spec(0.37, 0.48, n, Domain::Guitar)).unwrap();
        assert!(
            p.area_deficit < 0.0,
            "N={n}: a staircase cannot carry MORE area"
        );
        // The outline uses the SNAPPED length.
        let want = guitar_area(p.ly, p.lx, WAIST, ASYM);
        assert!((p.outline_area / want - 1.0).abs() < 1e-9, "N={n}");
        if let Some(pd) = prev {
            assert!(
                p.area_deficit.abs() < pd.abs(),
                "N={n}: the deficit must shrink"
            );
        }
        prev = Some(p.area_deficit);
    }
    assert!(prev.unwrap().abs() < 0.06);
}

#[test]
fn a_rectangle_still_prunes_nothing_and_carries_its_whole_area() {
    let p = Params::new(&free_spec(0.4, 0.3, 20, Domain::Rectangle)).unwrap();
    assert_eq!(p.n_pruned, 0);
    // Measured -2.3e-15: the Python's 1e-14 had 4.3x, and sat BELOW the worst-case rounding of the
    // 441-weight sum it checks (n·eps ≈ 1e-13). 1e-12 is that bound with 10x on top.
    assert!(p.area_deficit.abs() <= 1e-12, "{:e}", p.area_deficit);
    assert_eq!(p.domain, Domain::Rectangle);
}
