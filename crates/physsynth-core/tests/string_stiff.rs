//! Native bars for the theta-scheme strings, models #2 and #3 (plan §18).
//!
//! What can be asserted here is what does not need Python: the project's acceptance contract
//! (lossless energy conservation, passivity under either loss), the closed-form modal oracle, the
//! construction rejections, and the two reduction anchors that chain the family — `kappa = 0` and
//! `sigma1 = 0` — asserted here as **exact** equality rather than as a tolerance, which is the
//! form the Python suite asserts them in.
//!
//! Both models live in this one file because the reduction anchor spans them: a test that model #3
//! at `sigma1 = 0` *is* model #2 has to see both, and it is the reason the two cores are separate
//! transcriptions in the first place (see `string_stiff`'s header).
//!
//! # One test here must be run in both profiles
//!
//! [`squaring_is_pow_not_multiply`] pins an arithmetic *spelling*, and §17.2 established that such
//! a test asserts nothing in `--release` unless the exponent is opaque to LLVM. It is written
//! against `pyfloat::scalar_pow`'s `#[inline(never)]` rather than against a witness value, so it
//! holds in both; `cargo test --release` is the run that proves it.

use physsynth_core::pyfloat::scalar_pow;
use physsynth_core::sparse::Csr;
use physsynth_core::string_damped as damped;
use physsynth_core::string_stiff as stiff;

const L: f64 = 1.0;
const T: f64 = 200.0;
const RHO: f64 = 0.005;
const KAPPA: f64 = 1.5;

/// A model #2 string at the given loss and stiffness.
fn stiff_params(n: i64, kappa: f64, sigma: f64) -> stiff::Params {
    stiff::Params::new(
        L,
        T,
        RHO,
        44100.0,
        n,
        kappa,
        sigma,
        stiff::THETA_DEFAULT,
        true,
    )
    .expect("fixture must construct")
}

/// A model #3 string at the given losses and stiffness.
fn damped_params(n: i64, kappa: f64, sigma0: f64, sigma1: f64) -> damped::Params {
    damped::Params::new(
        L,
        T,
        RHO,
        44100.0,
        n,
        kappa,
        sigma0,
        sigma1,
        stiff::THETA_DEFAULT,
        true,
    )
    .expect("fixture must construct")
}

/// A triangular pluck on the full grid — the excitation the Python energy tests use.
fn pluck(x: &[f64], length: f64, at: f64, amplitude: f64) -> Vec<f64> {
    x.iter()
        .map(|&xi| {
            if xi <= at {
                amplitude * xi / at
            } else {
                amplitude * (length - xi) / (length - at)
            }
        })
        .collect()
}

/// `sin(m pi x / L)` — an exact discrete eigenvector of the simply-supported operator.
fn mode(x: &[f64], m: usize, amplitude: f64) -> Vec<f64> {
    x.iter()
        .map(|&xi| amplitude * (m as f64 * std::f64::consts::PI * xi / L).sin())
        .collect()
}

// -- the acceptance contract ---------------------------------------------------------------

#[test]
fn lossless_energy_is_conserved() {
    // The project's primary bug detector (CLAUDE.md): a lossless run drifts < 1e-10 relative.
    // The theta-scheme's energy is an exact algebraic identity, so the observed drift is ~1e-13.
    for kappa in [0.0, KAPPA] {
        let p = stiff_params(100, kappa, 0.0);
        let mut s = stiff::StiffString::new(p);
        let x = s.p.grid();
        s.set_state(&pluck(&x, L, 0.137 * L, 1e-3), &vec![0.0; x.len()]);
        let e0 = s.energy();
        assert!(e0 > 0.0, "a plucked string must store energy");
        let mut worst: f64 = 0.0;
        for _ in 0..2000 {
            s.step();
            worst = worst.max((s.energy() - e0).abs() / e0);
        }
        assert!(worst < 1e-10, "kappa = {kappa}: drift {worst:e}");
    }
}

#[test]
fn both_losses_are_passive() {
    // Passivity is unconditional for either loss term: the discrete dissipation is a negative
    // definite quadratic form. Asserted separately so a failure bisects to one term.
    for (sigma0, sigma1) in [(2.0, 0.0), (0.0, 5e-3), (2.0, 5e-3)] {
        let p = damped_params(100, KAPPA, sigma0, sigma1);
        let mut s = damped::DampedStiffString::new(p);
        let x = s.p.grid();
        s.set_state(&pluck(&x, L, 0.137 * L, 1e-3), &vec![0.0; x.len()]);
        let mut prev = s.energy();
        let e0 = prev;
        for n in 0..2000 {
            s.step();
            let e = s.energy();
            assert!(
                e - prev <= 1e-12 * e0,
                "sigma0 = {sigma0}, sigma1 = {sigma1}: energy rose at step {n}"
            );
            assert!(e.is_finite());
            prev = e;
        }
        assert!(prev < e0, "a lossy run must actually lose energy");
    }
}

#[test]
fn a_single_mode_oscillates_at_the_analytic_frequency() {
    // `sin(m pi x / L)` is an exact discrete eigenvector, so the scheme's own dispersion relation
    // is a closed form: the amplitude follows cos(omega_scheme n k) exactly. This checks the
    // operator, the factor and the step together against arithmetic that shares none of them.
    let n = 64;
    let p = stiff_params(n, KAPPA, 0.0);
    let h = p.h;
    let k = p.k;
    let theta = p.theta;
    let m = 3usize;

    // Discrete eigenvalue of -L for mode m: 4/h^2 sin^2(m pi h / 2L) scaled by c^2, plus the
    // biharmonic block, which is that squared times kappa^2 (because L's stiff part is D2 @ D2).
    let s = (m as f64 * std::f64::consts::PI * h / (2.0 * L)).sin();
    let lam_d2 = 4.0 / (h * h) * s * s;
    let lam_l = p.c * p.c * lam_d2 + p.kappa * p.kappa * lam_d2 * lam_d2;
    // Amplification: (1 + sk) z - 2 + z^-1 = -k^2 lam_l (theta z + (1 - 2 theta) + theta z^-1).
    // At sigma = 0 the root is on the unit circle: cos(omega k) = (1 - (1 - 2 theta) q) / (1 + 2
    // theta q) with q = k^2 lam_l / 2.
    let q = k * k * lam_l / 2.0;
    let cos_wk = (1.0 - (1.0 - 2.0 * theta) * q) / (1.0 + 2.0 * theta * q);
    let omega = cos_wk.acos() / k;

    // Seeded with the EXACT two-level cosine rather than through `set_state`: the consistent
    // Taylor start is only second-order accurate against the closed form, so it would put a ~1e-6
    // floor under this test and hide the thing being measured. `set_state` has its own test.
    let mut string = stiff::StiffString::new(p);
    let x = string.p.grid();
    let amp = 1e-3;
    string.u = mode(&x, m, amp);
    string.u_prev = mode(&x, m, amp * (omega * k).cos());
    let probe = (0.241 * n as f64).round() as usize;
    let shape = (m as f64 * std::f64::consts::PI * x[probe] / L).sin();

    for step in 1..=400 {
        string.step();
        let t = step as f64 * k;
        let want = amp * shape * (omega * t).cos();
        assert!(
            (string.u[probe] - want).abs() < 1e-11 * amp,
            "step {step}: got {}, want {want}",
            string.u[probe]
        );
    }
}

// -- the reduction anchors -----------------------------------------------------------------

#[test]
fn sigma1_zero_is_the_stiff_string_exactly() {
    // The anchor `tests/test_damped_string.py` asserts across the two Python classes, here across
    // the two Rust ones. Exact, not close: the sigma1 terms are GUARDED OUT rather than added as
    // zeros, in the matrix and in the right-hand side alike, so the two must do identical
    // arithmetic. This is the test that pays for keeping the two cores separate transcriptions.
    for kappa in [0.0, KAPPA] {
        let mut a = stiff::StiffString::new(stiff_params(100, kappa, 3.0));
        let mut b = damped::DampedStiffString::new(damped_params(100, kappa, 3.0, 0.0));
        let x = a.p.grid();
        let u0 = pluck(&x, L, 0.137 * L, 1e-3);
        let v0 = vec![0.0; x.len()];
        a.set_state(&u0, &v0);
        b.set_state(&u0, &v0);
        assert_eq!(
            a.u_prev, b.u_prev,
            "the consistent start must match exactly"
        );
        for step in 0..1500 {
            a.step();
            b.step();
            assert_eq!(a.u, b.u, "kappa = {kappa}: state diverged at step {step}");
            assert_eq!(a.energy(), b.energy(), "energy diverged at step {step}");
        }
    }
}

#[test]
fn kappa_zero_drops_the_biharmonic_block_structurally() {
    // `kappa = 0` skips the subtraction entirely, so `L` is TRIDIAGONAL rather than pentadiagonal
    // with two zero bands. The distinction is not cosmetic: it is what makes the two spellings
    // bit-identical rather than merely equal, and a `sub` that kept explicit zeros would pass
    // every energy test here while changing `nnz`.
    let p0 = stiff_params(16, 0.0, 0.0);
    let p1 = stiff_params(16, KAPPA, 0.0);
    assert_eq!(p0.op_l.nnz(), 3 * 15 - 2, "kappa = 0 must be tridiagonal");
    assert_eq!(p1.op_l.nnz(), 5 * 15 - 6, "kappa > 0 must be pentadiagonal");
    for i in 0..p0.op_l.nrows() {
        for j in 0..p0.op_l.ncols() {
            if i.abs_diff(j) > 1 {
                assert_eq!(p0.op_l.get(i, j), 0.0);
            }
        }
    }
}

#[test]
fn the_operator_is_symmetric_and_negative_definite() {
    // `-L` is SPD by summation by parts, which is what makes `P(f, f) >= 0` and the energy a norm.
    // Checked as the property rather than through the energy, so a failure here bisects to the
    // operator rather than to the form built on it.
    let p = stiff_params(24, KAPPA, 0.0);
    assert!(p.op_l.is_symmetric());
    for m in 1..12 {
        let x = p.grid();
        let f: Vec<f64> = mode(&x, m, 1.0)[1..x.len() - 1].to_vec();
        let quad = stiff::potential_form(&f, &f, &p);
        assert!(quad > 0.0, "mode {m}: P(f, f) = {quad} must be positive");
    }
}

// -- the sparse subtraction the operator is built with -------------------------------------

#[test]
fn sub_takes_the_union_and_drops_exact_cancellations() {
    let a = Csr::from_rows(2, 3, vec![vec![(0, 1.0), (2, 3.0)], vec![(1, 5.0)]]);
    let b = Csr::from_rows(2, 3, vec![vec![(1, 2.0), (2, 3.0)], vec![(1, 1.0)]]);
    let c = a.sub(&b);
    // Row 0: 1 - 0 = 1 at col 0, 0 - 2 = -2 at col 1, 3 - 3 = 0 at col 2 -> DROPPED.
    assert_eq!(c.indptr(), &[0, 2, 3]);
    assert_eq!(c.indices(), &[0, 1, 1]);
    assert_eq!(c.data(), &[1.0, -2.0, 4.0]);
}

#[test]
fn sub_result_is_canonical_whatever_the_operands_were() {
    // The property §18 needs on this side: the output row is ascending, so a matvec sums it in the
    // one order both languages can express. SciPy's own kernel does not promise this, which is why
    // `portable.canonical` sorts the Python side to meet it.
    let p = stiff_params(32, KAPPA, 0.0);
    for i in 0..p.op_l.nrows() {
        let row = &p.op_l.indices()[p.op_l.indptr()[i]..p.op_l.indptr()[i + 1]];
        assert!(
            row.windows(2).all(|w| w[0] < w[1]),
            "row {i} is not ascending: {row:?}"
        );
    }
}

// -- the spelling that only survives an opaque call ----------------------------------------

#[test]
fn squaring_is_pow_not_multiply() {
    // §17.2/§17.3: every constant these models own is built from `** 2` on a PYTHON float, which
    // is the C library's `pow` and not `x * x`. The claim is structural — `scalar_pow` reaches a
    // real `pow` — rather than a witness value, because HOW OFTEN the two disagree is a property
    // of the platform's libm and not of this port. Run in `--release` too: with the exponent
    // visible LLVM folds the call away and this test asserts nothing.
    for x in [1.0f64, 3.7, 1e-5, 12345.678, 0.1] {
        assert_eq!(x.powf(std::hint::black_box(2.0)), scalar_pow(x, 2.0));
    }
    // And report the disagreement rather than requiring it, for the same reason.
    let mut differ = 0usize;
    let mut v = 1.000_000_1f64;
    for _ in 0..200_000 {
        v = v * 1.000_013 + 1e-9;
        if scalar_pow(v, 2.0) != v * v {
            differ += 1;
        }
    }
    println!("scalar_pow(x, 2.0) != x * x in {differ} of 200000 samples");
}

#[test]
fn the_inharmonicity_constant_uses_the_squaring_path() {
    // `B = pi^2 kappa^2 / (c^2 L^2)`, all four through `pow`. Pinned because `B` reaches the
    // analysis layer and a `c * c` here would be invisible to every physics bar in the project.
    let p = stiff_params(64, KAPPA, 0.0);
    let want = (scalar_pow(std::f64::consts::PI, 2.0) * scalar_pow(KAPPA, 2.0))
        / (scalar_pow(p.c, 2.0) * scalar_pow(L, 2.0));
    assert_eq!(p.b, want);
    assert_eq!(stiff_params(64, 0.0, 0.0).b, 0.0);
}

// -- construction rejections ---------------------------------------------------------------

#[test]
fn construction_rejects_in_pythons_order() {
    // The messages are matched on by `tests/test_stability.py`, and the ORDER matters: a call that
    // is wrong in two ways must report the fault Python reports.
    let bad = |l, t, rho, fs, n, kappa, sigma, theta, ok| {
        stiff::Params::new(l, t, rho, fs, n, kappa, sigma, theta, ok)
            .expect_err("must be rejected")
            .to_string()
    };
    assert_eq!(
        bad(0.0, T, RHO, 44100.0, 64, 0.0, 0.0, 0.28, true),
        "L, T, rho, fs must all be positive."
    );
    assert_eq!(
        bad(L, T, RHO, 44100.0, 1, 0.0, 0.0, 0.28, true),
        "N must be >= 2 (need at least one interior node)."
    );
    assert_eq!(
        bad(L, T, RHO, 44100.0, -3, 0.0, 0.0, 0.28, true),
        "N must be >= 2 (need at least one interior node)."
    );
    assert_eq!(
        bad(L, T, RHO, 44100.0, 64, -1.0, 0.0, 0.28, true),
        "kappa (stiffness) must be >= 0."
    );
    assert_eq!(
        bad(L, T, RHO, 44100.0, 64, 0.0, -1.0, 0.28, true),
        "sigma (loss) must be >= 0."
    );
    assert_eq!(
        bad(L, T, RHO, 44100.0, 64, 0.0, 0.0, 0.0, true),
        "theta must be in (0, 1], got 0.0."
    );
    assert_eq!(
        bad(L, T, RHO, 44100.0, 64, 0.0, 0.0, 1.5, true),
        "theta must be in (0, 1], got 1.5."
    );
    // A negative tension is caught before a negative kappa, as Python's `min(...) <= 0` is.
    assert_eq!(
        bad(L, -1.0, RHO, 44100.0, 64, -1.0, 0.0, 0.28, true),
        "L, T, rho, fs must all be positive."
    );
    // Model #3 splits the loss message in two.
    let bad3 = |sigma0, sigma1| {
        damped::Params::new(L, T, RHO, 44100.0, 64, 0.0, sigma0, sigma1, 0.28, true)
            .expect_err("must be rejected")
            .to_string()
    };
    assert_eq!(
        bad3(-1.0, 0.0),
        "sigma0 (frequency-independent loss) must be >= 0."
    );
    assert_eq!(
        bad3(0.0, -1.0),
        "sigma1 (frequency-dependent loss) must be >= 0."
    );
}

#[test]
fn there_is_no_cfl_limit() {
    // The whole point of the implicit scheme: `lam > 1` is admissible and must construct, run and
    // conserve. The explicit `IdealString` rejects exactly this call.
    // lam = c k / h = 200 / (8000 * 0.005) = 5: five grid cells per timestep, which the explicit
    // `IdealString` rejects at construction and this scheme is required to run.
    let p = stiff::Params::new(
        L,
        T,
        RHO,
        8000.0,
        200,
        KAPPA,
        0.0,
        stiff::THETA_DEFAULT,
        true,
    )
    .expect("a supercritical string must construct");
    assert!(
        p.lam > 1.0,
        "the fixture must actually be supercritical, got {}",
        p.lam
    );
    let mut s = stiff::StiffString::new(p);
    let x = s.p.grid();
    s.set_state(&pluck(&x, L, 0.137 * L, 1e-3), &vec![0.0; x.len()]);
    let e0 = s.energy();
    for _ in 0..2000 {
        s.step();
    }
    assert!((s.energy() - e0).abs() / e0 < 1e-10, "supercritical drift");
    assert!(s.u.iter().all(|v| v.is_finite()));
}

// -- what a coupled model reaches for -------------------------------------------------------

#[test]
fn apply_ainv_inverts_the_matrix_the_step_uses() {
    // `bow`, `collision::BarrierString` and `connection` all build an admittance out of this, and
    // it is exact only if it is the SAME factor. Checked by reconstructing `A x` from the bands
    // and comparing against the right-hand side.
    let p = damped_params(24, KAPPA, 2.0, 5e-3);
    let n = p.interior();
    let ab = damped::update_matrix_bands(
        &p.op_l,
        &p.op_d2,
        p.sigma0 * p.k,
        p.theta * p.k * p.k,
        p.sigma1 * p.k,
        true,
    );
    for col in [0usize, 1, n / 2, n - 1] {
        let mut e = vec![0.0; n];
        e[col] = 1.0;
        let x = damped::apply_ainv(&e, &p);
        // (A x)_i = sum_j A_ij x_j, with A symmetric banded, kd = 2.
        for i in 0..n {
            let mut acc = 0.0;
            for d in 0..=2usize {
                if i + d < n {
                    acc += ab[(2 - d) * n + i + d] * x[i + d];
                }
                if d > 0 && i >= d {
                    acc += ab[(2 - d) * n + i] * x[i - d];
                }
            }
            assert!(
                (acc - e[i]).abs() < 1e-9,
                "column {col}, row {i}: A A^-1 e = {acc}"
            );
        }
    }
}

#[test]
fn set_state_clamps_the_ends_and_seeds_the_history() {
    // The ends are clamped BEFORE `L u0` is formed, so a caller who hands in a non-zero end gets a
    // consistent start rather than one built on a boundary value the scheme never honours.
    let p = stiff_params(32, KAPPA, 0.0);
    let mut s = stiff::StiffString::new(p);
    let x = s.p.grid();
    let mut u0 = mode(&x, 1, 1e-3);
    u0[0] = 7.0;
    *u0.last_mut().unwrap() = -7.0;
    s.set_state(&u0, &vec![0.0; x.len()]);
    assert_eq!(s.u[0], 0.0);
    assert_eq!(*s.u.last().unwrap(), 0.0);
    assert_eq!(s.u_prev[0], 0.0);
    assert_eq!(*s.u_prev.last().unwrap(), 0.0);
    assert_eq!(s.n, 0);

    // A non-zero initial velocity moves `u^{-1}` the other way by exactly `k v`.
    let mut a = stiff::StiffString::new(stiff_params(32, KAPPA, 0.0));
    let mut b = stiff::StiffString::new(stiff_params(32, KAPPA, 0.0));
    let clean = mode(&x, 1, 1e-3);
    a.set_state(&clean, &vec![0.0; x.len()]);
    b.set_state(&clean, &vec![1.0; x.len()]);
    // Not `assert_eq!`: `u^{-1}` is `(u0 - k v) + half_k2 * Lu0`, so the two differ by k only up
    // to the rounding of that final addition. The claim is the coefficient, not the last bit.
    for i in 1..x.len() - 1 {
        let delta = a.u_prev[i] - b.u_prev[i];
        assert!(
            (delta - a.p.k).abs() < 1e-12 * a.p.k,
            "node {i}: {delta} != k"
        );
    }
}
