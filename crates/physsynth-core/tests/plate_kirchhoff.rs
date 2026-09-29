//! Native bars for the plain Kirchhoff plate — model #5: energy, modes, stability.
//!
//! Carried from `tests/test_plate_energy.py`, `tests/test_plate_modal.py` and
//! `tests/test_plate_stability.py` (retirement plan §26). The simply-supported rectangle has
//! **machine-precise** eigenvalues — `sin·sin` is an exact discrete eigenvector of `B = L²` with
//! eigenvalue `Λ²` — so the bars are tight: the operator to machine precision, the continuum at
//! O(h²), low modes within a cent. There is no `kappa = 0` reduction to lean on (it gives
//! `u_tt = 0`), so operator correctness is proved by the `B`-eigenvalue-equals-`Λ²` money bar.
//!
//! **SciPy's sparse product retires as a referee here, with its numbers.** Three Python tests were
//! about the order SciPy's `L @ L` stores each row in, which the Rust port had to sort (plan §26 of
//! the migration). The product is transcribed below (`scipy_csr_matmat`, SciPy's `csr_matmat`
//! kernel), and it is **proved faithful rather than assumed**: it reproduces a digest of SciPy's
//! actual output — row order and values — recorded on 2026-09-29 before the Python was deleted
//! (NumPy 2.4.6, SciPy 1.17.1), at all four grids the Python test used.

use physsynth_analysis::horizon::{mode_block, mode_family, pitch_horizon, sinc_horizon_fraction};
use physsynth_analysis::modal::{
    cents, discrete_plate_eigenfrequency, rectangular_discrete_eigenvalues, rectangular_mode_field,
    rectangular_plate_freqs,
};
use physsynth_analysis::spectrum::measure_partials_near;
use physsynth_core::eigs::eigsh_shift_invert;
use physsynth_core::engine::{simulate, SimResult};
use physsynth_core::exciter::raised_cosine_2d;
use physsynth_core::ops2d::{biharmonic_from_mask, free_plate_stiffness};
use physsynth_core::plate::{
    pickup_index_at, Boundary, ParamError, Params, Plate, PlateSpec, THETA_DEFAULT,
};
use physsynth_core::sparse::Csr;

/// `tests/helpers.py`'s `KAPPA_PLATE_DEFAULT` and `RHO_AREAL_DEFAULT`; `THETA` is what the
/// analytic bars pass, and equals the plate's own `THETA_DEFAULT`, which the fixtures use.
const KAPPA: f64 = 20.0;
const RHO: f64 = 0.005;
const THETA: f64 = 0.28;
/// The acceptance bar, unchanged — see CLAUDE.md.
const DRIFT_TOL: f64 = 1e-10;

// -- fixtures ---------------------------------------------------------------------------------

/// `make_plate`: a supported unit square at plate-Courant number `mu`, `fs = kappa / (mu h²)`.
fn supported(n: i64, mu: f64, sigma: f64, rho: f64) -> Params {
    let h = 1.0 / n as f64;
    Params::new(&PlateSpec {
        lx: 1.0,
        ly: 1.0,
        kappa: KAPPA,
        rho,
        fs: KAPPA / (mu * h * h),
        n,
        sigma,
        theta: THETA_DEFAULT,
        ..PlateSpec::default()
    })
    .expect("an admissible plate")
}

fn plate(n: i64, mu: f64) -> Params {
    supported(n, mu, 0.0, RHO)
}

/// `make_free_plate`: a completely free unit square, `nu = 0.3`.
fn free(n: i64, mu: f64) -> Params {
    let h = 1.0 / n as f64;
    Params::new(&PlateSpec {
        lx: 1.0,
        ly: 1.0,
        kappa: KAPPA,
        rho: RHO,
        fs: KAPPA / (mu * h * h),
        n,
        boundary: Some(Boundary::Free),
        nu: Some(0.3),
        ..PlateSpec::default()
    })
    .expect("an admissible free plate")
}

/// The full-grid field restricted to the live nodes, in the plate's own ordering.
fn to_live(full: &[f64], p: &Params) -> Vec<f64> {
    full.iter()
        .zip(p.mask.flags())
        .filter(|(_, &alive)| alive)
        .map(|(v, _)| *v)
        .collect()
}

/// A raised cosine centred at `(cx Lx, cy Ly)` of radius `width`, on the live nodes.
fn bump(p: &Params, (cx, cy): (f64, f64), width: f64) -> Vec<f64> {
    let full = raised_cosine_2d(&p.x, &p.y, (cx * p.lx, cy * p.ly), width, 1e-3).unwrap();
    to_live(&full, p)
}

/// `test_plate_energy.py`'s `_pluck`.
fn pluck(p: &Params) -> Vec<f64> {
    bump(p, (0.4, 0.55), 0.25 * p.lx.min(p.ly))
}

/// `sin(mπx/Lx) sin(nπy/Ly)` scaled to `1e-3`, on the live nodes.
fn mode(p: &Params, m: i64, n: i64) -> Vec<f64> {
    let full = rectangular_mode_field(&p.x, &p.y, p.lx, p.ly, m, n);
    to_live(&full, p).iter().map(|v| v * 1e-3).collect()
}

fn started(p: Params, u0: &[f64]) -> Plate {
    let mut plate = Plate::new(p);
    plate.set_state(u0, &vec![0.0; u0.len()]);
    plate
}

/// `int(secs * fs)` steps — Python's truncation.
fn run(plate: &mut Plate, secs: f64, pickup: Option<usize>) -> SimResult {
    let steps = (secs * plate.p.fs) as usize;
    simulate(plate, steps, pickup, 0).expect("a linear plate cannot fail a step")
}

fn max_of(a: impl Iterator<Item = f64>) -> f64 {
    a.fold(f64::NEG_INFINITY, f64::max)
}

/// The `n_modes` lowest discrete frequencies — `plate_low_eigenfrequencies`: the smallest
/// eigenvalues of `-L` by shift-invert at 0, each mapped through the theta-scheme's `Λ -> f`.
fn low_eigenfrequencies(p: &Params, n_modes: usize) -> Vec<f64> {
    let neg_l = p
        .laplacian
        .as_ref()
        .expect("a supported plate")
        .scaled(-1.0);
    let mut lam = eigsh_shift_invert(&neg_l, None, 0.0, n_modes)
        .expect("-L is SPD")
        .values;
    lam.sort_by(f64::total_cmp);
    lam.iter()
        .map(|&l| discrete_plate_eigenfrequency(l, p.kappa, p.k, p.theta))
        .collect()
}

// -- tier 1: energy ----------------------------------------------------------------------------

#[test]
fn a_lossless_plate_conserves_energy_across_mu() {
    // mu > 1/4 is unstable for an EXPLICIT plate; the implicit theta-scheme conserves regardless.
    // Measured drift at mu 0.5 / 2 / 8: see §26.2; at worst ~40x under the bar.
    for mu in [0.5, 2.0, 8.0] {
        let p = plate(32, mu);
        let u0 = pluck(&p);
        let mut pl = started(p, &u0);
        let drift = run(&mut pl, 1.0, None).energy_drift();
        assert!(drift < DRIFT_TOL, "drift {drift:.2e} at mu = {mu}");
    }
}

#[test]
fn energy_is_conserved_at_a_timestep_an_explicit_plate_could_not_run() {
    // The unconditional-stability claim as a direct bar: mu = 16, 64x the explicit bound of 1/4.
    let p = plate(32, 16.0);
    assert!(p.mu > 0.25, "would blow up an explicit plate immediately");
    let u0 = pluck(&p);
    let mut pl = started(p, &u0);
    let drift = run(&mut pl, 1.0, None).energy_drift();
    assert!(drift < DRIFT_TOL, "drift {drift:.2e} at mu = 16");
}

#[test]
fn lossless_energy_stays_strictly_positive() {
    let p = plate(32, 2.0);
    let u0 = pluck(&p);
    let mut pl = started(p, &u0);
    let e = run(&mut pl, 0.5, None).energy;
    assert!(e.iter().all(|&v| v > 0.0), "a non-positive energy");
}

#[test]
fn a_lossy_plate_decreases_monotonically() {
    let p = supported(32, 2.0, 8.0, RHO);
    let u0 = pluck(&p);
    let mut pl = started(p, &u0);
    let e = run(&mut pl, 1.0, None).energy;
    let worst = max_of(e.windows(2).map(|w| w[1] - w[0])) / e[0];
    assert!(worst <= 1e-10, "max positive step {worst:.2e} * E0");
    // Monotone is only half the claim: a plate that never moved is monotone too.
    assert!(e[e.len() - 1] < e[0]);
}

#[test]
fn a_single_low_mode_decays_at_two_sigma() {
    // A single LOW mode (Q k² << 1) decays at ~2 sigma; that is where the analytic rate holds.
    let (sigma, secs) = (6.0, 0.5);
    let p = supported(32, 2.0, sigma, RHO);
    let u0 = mode(&p, 1, 1);
    let mut pl = started(p, &u0);
    let e = run(&mut pl, secs, None).energy;
    let measured = e[e.len() - 1] / e[0];
    let expected = (-2.0 * sigma * secs).exp();
    let rel = (measured.ln() - expected.ln()).abs() / expected.ln().abs();
    assert!(rel < 0.02, "low-mode decay off by {:.3}%", rel * 100.0);
}

#[test]
fn a_higher_mode_underdamps_relative_to_a_lower_one() {
    // The damping caveat, pinned: the rate 2σ(1 - θ Q k²) FALLS with mode (Q = κ² Λ²), so a higher
    // mode keeps more energy than a lower one — the opposite of a real plate, and the reason a
    // frequency-dependent loss is a separate model. Coarse timestep so θ Q k² is visible at (8,8).
    let (sigma, secs) = (6.0, 0.3);
    let retained: Vec<f64> = [1, 8]
        .iter()
        .map(|&m| {
            let p = supported(32, 8.0, sigma, RHO);
            let u0 = mode(&p, m, m);
            let mut pl = started(p, &u0);
            let e = run(&mut pl, secs, None).energy;
            e[e.len() - 1] / e[0]
        })
        .collect();
    assert!(
        retained[1] > retained[0],
        "high/low retained energy = {retained:?} (expected high > low)"
    );
}

#[test]
fn energy_is_in_joules_and_scales_with_areal_density() {
    // Doubling rho doubles E for the same field (kappa fixed: same grid, B and frequencies).
    let (p1, p2) = (
        supported(32, 2.0, 0.0, 0.005),
        supported(32, 2.0, 0.0, 0.010),
    );
    let e1 = started(p1.clone(), &pluck(&p1)).energy();
    let e2 = started(p2.clone(), &pluck(&p2)).energy();
    assert!(
        (e2 / e1 - 2.0).abs() <= 1e-12 * 2.0,
        "E(2 rho) / E(rho) = {}",
        e2 / e1
    );
}

// -- tier 2: modes ----------------------------------------------------------------------------

#[test]
fn the_biharmonic_eigenvalues_are_the_squared_laplacian_ones() {
    // Replaces the nonexistent kappa = 0 anchor: B's eigenvalues must equal Λ².
    let n = 24;
    let p = plate(n, 1.0);
    let ny = (p.ly / p.h).round() as i64;
    let modes = [(1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (1, 3)];
    let mut want: Vec<f64> = rectangular_discrete_eigenvalues(p.h, n, ny, &modes)
        .iter()
        .map(|l| l * l)
        .collect();
    want.sort_by(f64::total_cmp);
    let mut got = eigsh_shift_invert(&p.stiffness, None, 0.0, modes.len())
        .expect("B is SPD")
        .values;
    got.sort_by(f64::total_cmp);
    let rel = max_of(got.iter().zip(&want).map(|(g, w)| ((g - w) / w).abs()));
    assert!(
        rel < 1e-10,
        "biharmonic eigenvalue mismatch {rel:.2e} (B is mis-assembled)"
    );

    // The plate builds B through `biharmonic_from_mask`; a fresh build is the same matrix,
    // structure included. (Structural in Rust — `Params` calls it — so this pins the premise.)
    let (b, _) = biharmonic_from_mask(&p.mask, p.h);
    assert_eq!(b.indptr(), p.stiffness.indptr());
    assert_eq!(b.indices(), p.stiffness.indices());
    assert_eq!(b.data(), p.stiffness.data());
}

#[test]
fn the_discrete_law_converges_to_the_continuum_at_second_order() {
    // k ∝ h² at fixed mu, so the temporal error ∝ h⁴ and the spatial O(h²) sets the order.
    let (mu, lx) = (1.0, 1.0);
    let modes = [(1, 1), (2, 1)];
    let f_cont = rectangular_plate_freqs(KAPPA, lx, lx, &modes);
    let (mut hs, mut errs) = (Vec::new(), Vec::new());
    for n in [16i64, 32, 64, 128] {
        let h = lx / n as f64;
        let k = mu * h * h / KAPPA;
        let lam = rectangular_discrete_eigenvalues(h, n, n, &modes);
        let err = max_of(
            lam.iter()
                .zip(&f_cont)
                .map(|(&l, fc)| (discrete_plate_eigenfrequency(l, KAPPA, k, THETA) - fc).abs()),
        );
        hs.push(h);
        errs.push(err);
    }
    assert!(
        errs.windows(2).all(|w| w[1] < w[0]),
        "errors not decreasing: {errs:?}"
    );
    let last = errs.len() - 1;
    let order = (errs[last - 1] / errs[last]).ln() / (hs[last - 1] / hs[last]).ln();
    assert!(order > 1.8, "continuum convergence order {order:.2} < 1.8");
}

#[test]
fn the_low_modes_are_within_one_cent_and_the_band_is_the_measured_horizon() {
    // The band is DERIVED from the plate's own pitch horizon, and it is exactly saturated: a 2x2
    // block's worst mode is its diagonal corner, so the block's horizon is the diagonal family's,
    // which is a prefix. Both families' horizons are 2 at one cent here, and one more index in
    // either direction breaks the bar — this literal has zero headroom by construction.
    let (n, mu, lx, bound) = (96i64, 0.5, 1.0, 1.0);
    let h = lx / n as f64;
    let k = mu * h * h / KAPPA;
    let frequencies = |modes: &[(i64, i64)]| -> (Vec<f64>, Vec<f64>) {
        let f_disc = rectangular_discrete_eigenvalues(h, n, n, modes)
            .iter()
            .map(|&l| discrete_plate_eigenfrequency(l, KAPPA, k, THETA))
            .collect();
        (f_disc, rectangular_plate_freqs(KAPPA, lx, lx, modes))
    };
    let worst_cents = |modes: &[(i64, i64)]| {
        let (d, c) = frequencies(modes);
        max_of(d.iter().zip(&c).map(|(a, b)| cents(*a, *b).abs()))
    };

    let four = worst_cents(&mode_block(2).unwrap());
    assert!(four < 1.0, "max error {four:.3} cents > 1 (tight bar)");

    // The window comes from the closed-form space floor, so `horizon < window` cannot quietly
    // become `horizon == window` on a finer grid.
    let window = (2.0 * sinc_horizon_fraction(bound, 2).unwrap() * n as f64).ceil() as i64;
    let mut horizons = Vec::new();
    for kind in ["diagonal", "axial"] {
        let (d, c) = frequencies(&mode_family(kind, window).unwrap());
        let (horizon, monotone) = pitch_horizon(&d, &c, bound).unwrap();
        assert!(
            monotone,
            "the {kind} error curve is not monotone; the prefix hides a mode"
        );
        assert!(
            (horizon as i64) < window,
            "the {kind} horizon {horizon} filled the window {window}: a lower bound, not a horizon"
        );
        horizons.push(horizon);
    }
    assert_eq!(
        horizons[0], horizons[1],
        "the two families should agree in their own index"
    );
    // The only line a dispersion oracle broken toward flatness could fail.
    assert!(
        horizons[0] >= 2,
        "the one-cent horizon fell to {}",
        horizons[0]
    );

    let hz = horizons[0] as i64;
    let inside = worst_cents(&mode_block(hz).unwrap());
    let outside = worst_cents(&mode_block(hz + 1).unwrap());
    assert!(
        inside < bound && bound <= outside,
        "the block is not saturated: {inside:.3} inside, {outside:.3} one index wider"
    );
}

#[test]
fn the_low_spectrum_of_the_assembled_laplacian_matches_the_oracle() {
    // Shift-invert on the actual assembled L, mapped through the scheme oracle, against the
    // closed-form series (degeneracy handled by sorting).
    let n = 64;
    let p = plate(n, 0.5);
    let ny = (p.ly / p.h).round() as i64;
    let modes: Vec<(i64, i64)> = (1..5).flat_map(|m| (1..5).map(move |q| (m, q))).collect();
    let mut oracle: Vec<f64> = rectangular_discrete_eigenvalues(p.h, n, ny, &modes)
        .iter()
        .map(|&l| discrete_plate_eigenfrequency(l, p.kappa, p.k, p.theta))
        .collect();
    oracle.sort_by(f64::total_cmp);
    let measured = low_eigenfrequencies(&p, 6);
    let worst = max_of(
        measured
            .iter()
            .zip(&oracle)
            .map(|(m, o)| cents(*m, *o).abs()),
    );
    assert!(
        worst < 0.5,
        "worst low-mode error {worst:.3} cents; measured {measured:?}"
    );
}

#[test]
fn the_time_stepper_rings_at_the_discrete_fundamental() {
    let p = plate(48, 1.0);
    let f_disc = low_eigenfrequencies(&p, 1)[0];
    let u0 = bump(&p, (0.35, 0.42), 0.4 * p.lx);
    let pickup = pickup_index_at(0.3 * p.lx, 0.28 * p.ly, &p);
    let mut pl = started(p, &u0);
    let res = run(&mut pl, 0.5, Some(pickup));
    let found =
        measure_partials_near(res.output.as_ref().unwrap(), res.fs, &[f_disc], Some(20.0))[0];
    let err = cents(found, f_disc).abs();
    assert!(
        err < 5.0,
        "FFT fundamental off by {err:.2} cents ({found:.2} vs {f_disc:.2})"
    );
}

// -- SciPy's product, the order it stores, and what sorting it did ------------------------------

/// SciPy's `csr_matmat` (`scipy/sparse/sparsetools/csr.h`), transcribed: each output row is
/// accumulated over `A`'s row in stored order, its columns threaded onto a linked list as they are
/// first touched, then emitted by walking the list from its head — i.e. in REVERSE first-touch
/// order — dropping any entry whose sum is exactly zero. Returns `(indptr, indices, data)`.
fn scipy_csr_matmat(a: &Csr, b: &Csr) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let m = b.ncols();
    let mut next = vec![-1isize; m];
    let mut sums = vec![0.0f64; m];
    let (mut indptr, mut indices, mut data) = (vec![0usize], Vec::new(), Vec::new());
    for i in 0..a.nrows() {
        let mut head: isize = -2;
        let mut length = 0;
        for jj in a.indptr()[i]..a.indptr()[i + 1] {
            let (j, v) = (a.indices()[jj], a.data()[jj]);
            for kk in b.indptr()[j]..b.indptr()[j + 1] {
                let k = b.indices()[kk];
                sums[k] += v * b.data()[kk];
                if next[k] == -1 {
                    next[k] = head;
                    head = k as isize;
                    length += 1;
                }
            }
        }
        for _ in 0..length {
            let hd = head as usize;
            if sums[hd] != 0.0 {
                indices.push(hd);
                data.push(sums[hd]);
            }
            head = next[hd];
            next[hd] = -1;
            sums[hd] = 0.0;
        }
        indptr.push(indices.len());
    }
    (indptr, indices, data)
}

/// SciPy's `L @ L` for `make_plate(N, mu=1)`, recorded 2026-09-29: `(N, nnz, Σ (i+1)·indices[i],
/// Σ data[i]·(i+1) summed left to right)`. The two weighted sums see the stored ORDER as well as
/// the values; every row opens with the reverse-first-touch pattern (`N = 8`: 14, 8, 2, 7, 1, 0).
const SCIPY_LL_DIGESTS: [(i64, usize, u64, f64); 4] = [
    (8, 501, 3_826_310, 43_601_920.0),
    (12, 1_357, 71_820_838, 828_382_464.0),
    (16, 2_629, 507_453_238, 6_465_847_296.0),
    (24, 6_421, 7_184_107_174, 114_212_229_120.0),
];

fn scipy_ll(p: &Params) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let l = p.laplacian.as_ref().expect("a supported plate");
    scipy_csr_matmat(l, l)
}

fn is_sorted_by_row(indptr: &[usize], indices: &[usize]) -> bool {
    indptr
        .windows(2)
        .all(|w| indices[w[0]..w[1]].windows(2).all(|c| c[0] < c[1]))
}

#[test]
fn the_canonical_sort_changed_an_order_and_not_a_value() {
    // The Rust port sorts each row of B; SciPy stored it in kernel order. A CSR matvec sums a row
    // in STORED order, so `B @ u` was a different sum on the two sides, and sorting changed the
    // shipped numbers' last bits. This pins that it changed an ORDER and not a VALUE — against
    // SciPy's own product, reproduced exactly from the recorded digests.
    for (n, nnz, isum, fsum) in SCIPY_LL_DIGESTS {
        let p = plate(n, 1.0);
        let (indptr, indices, data) = scipy_ll(&p);
        assert_eq!(indices.len(), nnz, "N={n}: not SciPy's product");
        let got_isum: u64 = indices
            .iter()
            .enumerate()
            .map(|(k, &i)| (k as u64 + 1) * i as u64)
            .sum();
        let mut got_fsum = 0.0f64;
        for (k, &x) in data.iter().enumerate() {
            got_fsum += x * (k + 1) as f64;
        }
        assert_eq!(
            (got_isum, got_fsum),
            (isum, fsum),
            "N={n}: not SciPy's order or values"
        );
        assert!(
            !is_sorted_by_row(&indptr, &indices),
            "N={n}: the kernel order is sorted, so this compares an operator with itself"
        );

        // Sort each row of SciPy's product: it must BE the shipped operator, to the bit.
        let mut sorted_idx = Vec::with_capacity(nnz);
        let mut sorted_dat = Vec::with_capacity(nnz);
        for w in indptr.windows(2) {
            let mut row: Vec<(usize, f64)> = (w[0]..w[1]).map(|q| (indices[q], data[q])).collect();
            row.sort_by_key(|e| e.0);
            sorted_idx.extend(row.iter().map(|e| e.0));
            sorted_dat.extend(row.iter().map(|e| e.1));
        }
        assert_eq!(indptr, p.stiffness.indptr(), "N={n}");
        assert_eq!(sorted_idx, p.stiffness.indices(), "N={n}");
        assert_eq!(
            sorted_dat,
            p.stiffness.data(),
            "N={n}: sorting SciPy's operator does not give the shipped one — a VALUE changed"
        );
    }
}

/// A structureless field in `[-1, 1)` — splitmix64. Replaces the Python's seeded normal draw; the
/// claim is about a broadband field, which any structureless generator serves (retirement plan
/// §15's rule for tolerances on broadband fields).
fn structureless(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            2.0 * ((z >> 11) as f64 / (1u64 << 53) as f64) - 1.0
        })
        .collect()
}

#[test]
fn the_canonical_sort_left_the_shipped_plate_where_it_was() {
    // Step one plate on the canonical B and one on SciPy's kernel-order B for 2000 steps: the
    // trajectories agree to 1e-11 of their amplitude and both conserve. The kernel-order operator
    // is injected through `Csr::from_arrays_preserving_order`, the one constructor that does not
    // sort — it exists for this bar — and the injected matrix is asserted UNSORTED, or both
    // plates would carry the shipped B and the comparison would assert nothing.
    for n in [12i64, 16] {
        let p_new = plate(n, 1.0);
        let mut p_old = plate(n, 1.0);
        let (indptr, indices, data) = scipy_ll(&p_old);
        let nl = p_old.n_live;
        p_old.stiffness = Csr::from_arrays_preserving_order(nl, nl, indptr, indices, data)
            .expect("SciPy's product is a well-formed CSR");
        assert!(
            !p_old.stiffness.has_sorted_indices(),
            "N={n}: the injected operator is canonical; the two plates are one plate"
        );
        let u0: Vec<f64> = structureless(nl, 20_260_828 + n as u64)
            .iter()
            .map(|v| 1e-4 * v)
            .collect();
        let mut drifts = Vec::new();
        let mut states = Vec::new();
        for p in [p_new, p_old] {
            let mut pl = started(p, &u0);
            let e0 = pl.energy();
            let mut worst = 0.0f64;
            for _ in 0..2000 {
                pl.step(None);
                worst = worst.max((pl.energy() / e0 - 1.0).abs());
            }
            drifts.push(worst);
            states.push(pl.u);
        }
        let amp = max_of(states[0].iter().map(|v| v.abs()));
        let moved = max_of(states[0].iter().zip(&states[1]).map(|(a, b)| (a - b).abs())) / amp;
        assert!(
            moved < 1e-11,
            "N={n}: the shipped plate moved by {moved:.2e} of its amplitude"
        );
        for (drift, which) in drifts.iter().zip(["canonical", "kernel-order"]) {
            assert!(
                *drift < 1e-10,
                "N={n}: the {which} operator drifts {drift:.2e}"
            );
        }
    }
}

#[test]
fn the_free_plates_stiffness_is_canonical_by_construction() {
    // VERDICT carried as a premise. The Python asserted that SciPy returns the free plate's Gram
    // product already sorted, so the canonical sort was a no-op there. Since phase A that builder
    // has been Rust, whose `Csr` sorts every row it assembles — the Python test was already
    // asserting this, not anything about SciPy. Pinned so a future assembly that stops sorting
    // (it would have to go through `from_arrays_preserving_order`) is caught here.
    for n in [8usize, 12, 16] {
        let (k, w, _) = free_plate_stiffness(n, n, 1.0 / n as f64, 0.3, 1.0, 1.0, None, None);
        assert!(k.has_sorted_indices() && w.has_sorted_indices(), "N={n}");
    }
}

// -- tier 3: stability and construction -------------------------------------------------------

/// `test_plate_stability.py`'s `_run`: a wide bump, read at live node 0.
fn stability_run(p: Params, secs: f64) -> SimResult {
    let u0 = bump(&p, (0.4, 0.5), 0.3 * p.lx);
    let mut pl = started(p, &u0);
    run(&mut pl, secs, Some(0))
}

fn all_finite(res: &SimResult) -> bool {
    res.energy.iter().all(|v| v.is_finite())
        && res.output.as_ref().unwrap().iter().all(|v| v.is_finite())
}

#[test]
fn no_nan_across_mu_on_either_boundary() {
    // The implicit theta-scheme (theta >= 1/4) has no CFL ceiling to reject, on either branch.
    for mu in [0.1, 0.5, 2.0, 8.0, 32.0] {
        assert!(
            all_finite(&stability_run(plate(40, mu), 0.3)),
            "supported, mu = {mu}"
        );
        assert!(
            all_finite(&stability_run(free(32, mu), 0.3)),
            "free, mu = {mu}"
        );
    }
}

#[test]
fn a_timestep_200x_past_the_explicit_bound_runs_and_conserves_on_either_boundary() {
    for (label, p) in [("supported", plate(40, 50.0)), ("free", free(32, 50.0))] {
        assert!(p.mu > 0.25);
        let res = stability_run(p, 0.5);
        assert!(res.energy.iter().all(|v| v.is_finite()), "{label}");
        let drift = res.energy_drift();
        assert!(drift < 1e-9, "{label}: drift {drift:.2e} at mu = 50");
    }
}

#[test]
fn non_physical_parameters_are_refused_at_construction() {
    let base = PlateSpec {
        lx: 1.0,
        ly: 1.0,
        kappa: 20.0,
        rho: 0.005,
        fs: 50_000.0,
        n: 40,
        ..PlateSpec::default()
    };
    let cases: [(&str, PlateSpec, ParamError); 15] = [
        (
            "kappa = 0",
            PlateSpec {
                kappa: 0.0,
                ..base.clone()
            },
            ParamError::NonPositiveKappa,
        ),
        (
            "kappa < 0",
            PlateSpec {
                kappa: -1.0,
                ..base.clone()
            },
            ParamError::NonPositiveKappa,
        ),
        (
            "rho = 0",
            PlateSpec {
                rho: 0.0,
                ..base.clone()
            },
            ParamError::NonPositiveRho,
        ),
        (
            "rho < 0",
            PlateSpec {
                rho: -1.0,
                ..base.clone()
            },
            ParamError::NonPositiveRho,
        ),
        (
            "fs = 0",
            PlateSpec {
                fs: 0.0,
                ..base.clone()
            },
            ParamError::NonPositive,
        ),
        (
            "Lx = 0",
            PlateSpec {
                lx: 0.0,
                ..base.clone()
            },
            ParamError::NonPositive,
        ),
        (
            "Ly < 0",
            PlateSpec {
                ly: -1.0,
                ..base.clone()
            },
            ParamError::NonPositive,
        ),
        (
            "sigma < 0",
            PlateSpec {
                sigma: -0.1,
                ..base.clone()
            },
            ParamError::NegativeSigma,
        ),
        (
            "N = 1",
            PlateSpec {
                n: 1,
                ..base.clone()
            },
            ParamError::TooFewSegments,
        ),
        (
            "theta = 0",
            PlateSpec {
                theta: 0.0,
                ..base.clone()
            },
            ParamError::BadTheta(0.0),
        ),
        (
            "theta > 1",
            PlateSpec {
                theta: 1.5,
                ..base.clone()
            },
            ParamError::BadTheta(1.5),
        ),
        // nu must be in (-1, 1/2): the energy's positive-definite, physical range.
        (
            "nu = 1/2",
            PlateSpec {
                nu: Some(0.5),
                ..base.clone()
            },
            ParamError::BadNu(0.5),
        ),
        (
            "nu = 1",
            PlateSpec {
                nu: Some(1.0),
                ..base.clone()
            },
            ParamError::BadNu(1.0),
        ),
        (
            "nu = -1",
            PlateSpec {
                nu: Some(-1.0),
                ..base.clone()
            },
            ParamError::BadNu(-1.0),
        ),
        // Python's `boundary="clamped"`: a string the binding cannot parse arrives as `None`.
        // The native API cannot SPELL a clamped plate at all; a shape refusal became a type.
        (
            "boundary",
            PlateSpec {
                boundary: None,
                ..base.clone()
            },
            ParamError::BadBoundary,
        ),
    ];
    for (label, spec, want) in cases {
        let got = Params::new(&spec).expect_err(label);
        assert_eq!(
            std::mem::discriminant(&got),
            std::mem::discriminant(&want),
            "{label}: {got:?}"
        );
        if let (ParamError::BadTheta(a), ParamError::BadTheta(b))
        | (ParamError::BadNu(a), ParamError::BadNu(b)) = (&got, &want)
        {
            assert_eq!(a, b, "{label}");
        }
    }
}

#[test]
fn the_free_boundary_constructs_with_every_node_a_free_unknown() {
    let p = Params::new(&PlateSpec {
        lx: 1.0,
        ly: 1.0,
        kappa: 20.0,
        rho: 0.005,
        fs: 50_000.0,
        n: 20,
        boundary: Some(Boundary::Free),
        nu: Some(0.3),
        ..PlateSpec::default()
    })
    .expect("boundary = free constructs");
    assert_eq!(p.boundary, Boundary::Free);
    assert_eq!(p.n_live, (p.n + 1) * (p.n + 1), "no Dirichlet rim");
}
