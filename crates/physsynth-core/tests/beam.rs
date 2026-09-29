//! Native bars for the free-free Euler–Bernoulli beam — model #5b-pre.
//!
//! The beam is the 1D rehearsal of the free-edge plate: the one free-boundary flexural model in
//! this project with a *genuine* closed-form spectrum, `cos(betaL) cosh(betaL) = 1`. So unlike
//! most native bar files this one is not only energy — the modal claim can be checked against
//! mathematics rather than against a second transcription, and that is the point of having it.
//!
//! **It is also the acceptance contract the Python suite was** (retirement plan §30):
//! `tests/test_beam_{energy,modal,stability}.py` retired into this file, at `make_beam`'s own
//! parameters (`L = 1`, `rho = 0.005`, `kappa = 20`, `theta = 0.28`, `fs = kappa / (mu h^2)`).
//! Where a bar here was already at those parameters and at least as strict it was counted as
//! carried; where it was shorter (the one-second runs) or at another grid, it was brought to the
//! Python's.
//!
//! **One outside referee retires here, with its numbers.** Everything else the Python tests
//! touched — the exciter, `simulate`, the spectrum detector, every `modal.*` oracle — already ran
//! through the Rust binding. SciPy's `eigsh` did not: it was frozen on 2026-09-29 (NumPy 2.4.6,
//! SciPy 1.17.1, wheel reinstalled) into `tests/reference/beam.json` at every grid the suite solved
//! on, with LAPACK's dense generalized `eigh` and the **50-digit Rayleigh quotient of LAPACK's
//! eigenvector**, which is exact for that vector and whose error as an eigenvalue is second order
//! in the vector's. Measured against it, ARPACK at the Python's own settings was the least accurate
//! solver in the room again (22.6 `eps mu_max` on the second elastic mode at N = 120; LAPACK ≤ 1.8
//! everywhere) — §24.3's finding, a third time.
//!
//! Four things here are not a translation of the Python suite and are the reason the file is
//! worth its length:
//!
//! * **the oracle is derived in this file too**, by bisecting `cos(x) - sech(x)`, and tied to the
//!   analysis crate's `free_free_beam_beta_l` by its own bar — two derivations that must agree, so
//!   a defect planted in either is seen. The residual is spelled `cos - sech` and not
//!   `cos * cosh - 1` because the second scales like `cosh`, so one tolerance would mean a
//!   different accuracy at every root;
//! * **the eigensolver is built from the crate's own sparse LU** — a shifted inverse iteration
//!   with the rigid-body directions projected out in the `W` inner product. Its shift is what
//!   makes it terminate: iterating on the beam's *own* update matrix `A = W + theta k^2 kappa^2 K`
//!   gives an eigenvalue ratio of `1 - 1.5e-7` at the modal fixture, which is not slow but
//!   impossible. See [`elastic_modes`]. It supplies the *vectors* the single-mode bars start from,
//!   and it is certified against the frozen record like the Krylov solver is;
//! * **the time discretization is pinned pointwise**, at 1e-15, rather than only through an FFT.
//!   With a machine-precision eigenvector in hand `u^{n+1} + u^{n-1} = 2 cos(omega k) u^n` is an
//!   exact algebraic identity of the scheme, so the discrete modal frequency can be asserted
//!   directly. The Python's FFT bar is carried as well — it is an end-to-end claim about the
//!   pickup and the detector, which the identity does not make;
//! * **every worst-case figure propagates NaN** (§29.3): `fold(_, f64::max)` returns the other
//!   operand when one side is NaN, and would report a clean figure for a run that blew up.

use physsynth_analysis::horizon::pitch_horizon;
use physsynth_analysis::modal::{
    cents as analysis_cents, discrete_beam_eigenfrequency, free_free_beam_beta_l,
    free_free_beam_freqs,
};
use physsynth_analysis::spectrum::measure_partials_near;
use physsynth_core::beam::{FreeBeam, ParamError, Params, THETA_DEFAULT};
use physsynth_core::eigs::eigsh_shift_invert;
use physsynth_core::engine::{simulate, SimResult};
use physsynth_core::exciter::raised_cosine;
use physsynth_core::ops::free_beam_stiffness;
use physsynth_core::sparse_lu::SparseLu;
use serde_json::Value;
use std::f64::consts::PI;

const L: f64 = 1.0;
const RHO: f64 = 0.005;
const KAPPA: f64 = 20.0;
/// The acceptance bar, unchanged — see CLAUDE.md.
const DRIFT_TOL: f64 = 1e-10;

/// The continuum fundamental's `betaL`, used only to set the inverse iteration's shift and the
/// "is this rigid-body eigenvalue zero" scale. The bars themselves bisect for their own roots.
const BETA_L1: f64 = 4.730_040_744_862_704;

// -- fixtures ------------------------------------------------------------------------------------

/// Parameters at beam-Courant number `mu = kappa k / h^2`, the Python helper's construction:
/// `h = L/N` is fixed by the geometry, so the sample rate is solved for.
fn params(n: i64, mu: f64, sigma: f64) -> Params {
    let h = L / (n as f64);
    let fs = KAPPA / (mu * h * h);
    Params::new(L, RHO, fs, n, KAPPA, sigma, THETA_DEFAULT, true).expect("a valid beam")
}

fn beam(n: i64, mu: f64, sigma: f64) -> FreeBeam {
    FreeBeam::new(params(n, mu, sigma))
}

/// A smooth off-centre bump — band-limited, and it touches neither end, so nothing is clamped.
fn pluck(p: &Params, center: f64, width: f64) -> Vec<f64> {
    let x = p.grid();
    raised_cosine(&x, p.l, center * p.l, width * p.l, 1e-3).expect("a positive width")
}

/// `_run`: pluck (`0.4 L`, width `0.2 L`), then `simulate` for `int(secs * fs)` steps.
fn run(b: &mut FreeBeam, secs: f64, steps_at_least: usize) -> SimResult {
    let u0 = pluck(&b.p, 0.4, 0.2);
    b.set_state(&u0, &vec![0.0; b.p.nodes()]);
    let steps = ((secs * b.p.fs) as usize).max(steps_at_least);
    simulate(b, steps, None, 0).expect("a beam step cannot fail")
}

/// `np.max`: NaN-propagating, unlike `fold(_, f64::max)` (§29.3).
fn nan_max(it: impl Iterator<Item = f64>) -> f64 {
    let mut m = f64::NEG_INFINITY;
    for v in it {
        if v.is_nan() {
            return f64::NAN;
        }
        m = m.max(v);
    }
    m
}

fn reference() -> Value {
    serde_json::from_str(include_str!("reference/beam.json")).expect("the frozen record")
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("an array")
        .iter()
        .map(|x| x.as_f64().expect("a number"))
        .collect()
}

/// The `count` lowest eigenvalues of `K phi = mu W phi` from the crate's shift-invert Krylov
/// solver, ascending, rigid pair included — `beam_low_eigenfrequencies`' solve, at its shift
/// `-1e-3 mu_1` (which is also the shift §28.2 found the native solver needs next to a nullspace).
fn krylov_lowest(p: &Params, count: usize) -> Vec<f64> {
    let shift = -1e-3 * (BETA_L1 / p.l).powi(4);
    eigsh_shift_invert(&p.stiffness, Some(&p.mass), shift, count)
        .expect("the shifted pencil factors")
        .values
}

/// `f = kappa sqrt(mu) / 2 pi`, with the helper's `clip(mu, 0)`.
fn spatial_hz(p: &Params, mu: f64) -> f64 {
    p.kappa * mu.max(0.0).sqrt() / (2.0 * PI)
}

// -- the eigensolver ------------------------------------------------------------------------------

/// `sum(a[i] * b[i])`, left to right. The crate's `beam::dot` is the same spelling; this is a
/// local copy so the test does not lean on a diagnostic helper it is also measuring through.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = 0.0;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

/// `<a, b>_W` — the mass-weighted inner product the generalized eigenproblem is orthogonal in.
fn w_dot(a: &[f64], b: &[f64], w: &[f64]) -> f64 {
    let mut acc = 0.0;
    for i in 0..a.len() {
        acc += a[i] * (w[i] * b[i]);
    }
    acc
}

/// Remove the `W`-orthogonal projection of `v` onto each of `basis` (each already `W`-normalized).
fn deflate(v: &mut [f64], basis: &[Vec<f64>], w: &[f64]) {
    for b in basis {
        let c = w_dot(b, v, w);
        for i in 0..v.len() {
            v[i] -= c * b[i];
        }
    }
}

/// Scale `v` to `<v, v>_W = 1`.
fn w_normalize(v: &mut [f64], w: &[f64]) {
    let nrm = w_dot(v, v, w).sqrt();
    for x in v.iter_mut() {
        *x /= nrm;
    }
}

/// The `count` lowest **elastic** eigenpairs of `K phi = mu W phi`, ascending, as
/// `(mu, phi)` with `phi` `W`-normalized.
///
/// `K` is only positive-*semi*definite — its nullspace is exactly the rigid-body pair `{1, x}` —
/// so the problem is solved shifted: `B = K + eps W` is SPD for any `eps > 0`, is factored once,
/// and `B^-1 W` is inverse-iterated. Three decisions in that sentence are the whole design:
///
/// * **the shift is a fixed positive number, not the update matrix.** The tempting move is to
///   reuse the factorization the beam already carries, `A = W + theta k^2 kappa^2 K`. Its
///   eigenvalues are `1/(1 + theta k^2 kappa^2 mu)`, and at the modal fixture the coefficient is
///   ~4.4e-11, so consecutive modes are separated by one part in 1e7 — about 1e8 iterations for
///   the digits this file needs. With `eps ~ 1e-3 mu_1` the ratio is ~0.13 instead;
/// * **the rigid-body directions are projected out every iteration**, not once. They are exactly
///   invariant, but roundoff reintroduces them and their shifted eigenvalue `1/eps` is three
///   orders *above* the fundamental's, so a reintroduced component grows fastest of anything;
/// * **the iteration count is fixed, and `mu` is read off as a Rayleigh quotient.** Fixed because
///   a convergence-tolerance stop makes the count a datum, and the count is not a comparable
///   quantity (plan ledger #33). Rayleigh because recovering `mu` through the shift subtracts two
///   nearly equal numbers, while `(K phi . phi) / (W phi . phi)` is accurate to the *square* of
///   the eigenvector error.
fn elastic_modes(p: &Params, count: usize) -> Vec<(f64, Vec<f64>)> {
    let nodes = p.nodes();
    let x = p.grid();
    let w = &p.w;

    // The rigid-body nullspace {1, x}, W-orthonormalized. Deflating against these is what makes
    // the iteration converge to the lowest *elastic* mode rather than to a rigid-body motion.
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for seed in [vec![1.0; nodes], x.clone()] {
        let mut v = seed;
        deflate(&mut v, &basis, w);
        w_normalize(&mut v, w);
        basis.push(v);
    }

    let eps = 1e-3 * (BETA_L1 / p.l).powi(4);
    let shifted = p.stiffness.add(&p.mass.scaled(eps));
    let lu = SparseLu::factor(&shifted).expect("K + eps W is SPD");

    // 80 sweeps. The slowest ratio this file asks for is mode 4 against mode 5, ~0.45, which is
    // 1e-27 in 80 -- far below the roundoff floor, so the count is chosen to be obviously enough
    // rather than to be tight.
    const SWEEPS: usize = 80;

    let mut out = Vec::with_capacity(count);
    for m in 0..count {
        // A deterministic, non-symmetric seed: a symmetric one is orthogonal to every
        // antisymmetric mode and the iteration would never find mode 2.
        let mut v: Vec<f64> = (0..nodes)
            .map(|i| ((i as f64) * 0.37 + (m as f64) * 1.13).sin())
            .collect();
        deflate(&mut v, &basis, w);
        w_normalize(&mut v, w);
        for _ in 0..SWEEPS {
            let rhs: Vec<f64> = (0..nodes).map(|i| w[i] * v[i]).collect();
            v = lu.solve(&rhs).expect("the factor matches the state length");
            deflate(&mut v, &basis, w);
            w_normalize(&mut v, w);
        }
        let kv = p.stiffness.matvec(&v);
        let mu = dot(&kv, &v) / w_dot(&v, &v, w);
        basis.push(v.clone()); // deflate the next mode against this one
        out.push((mu, v));
    }
    out
}

/// The `j`-th root of `cos(x) cosh(x) = 1`, `j` counting from 1.
///
/// Spelled `cos(x) - 1/cosh(x)` and bisected. On `[j pi, (j+1) pi]` the cosine is monotone and
/// `sech` is a small decreasing perturbation, so there is exactly one sign change per bracket and
/// bisection cannot land on the wrong root — which is the reason for bracketing on multiples of
/// `pi` rather than around the asymptotic `(j + 1/2) pi`.
fn beta_l(j: usize) -> f64 {
    let f = |x: f64| x.cos() - 1.0 / x.cosh();
    let (mut lo, mut hi) = (
        (j as f64) * std::f64::consts::PI + 0.05,
        ((j + 1) as f64) * std::f64::consts::PI - 0.05,
    );
    let (flo, fhi) = (f(lo), f(hi));
    assert!(flo * fhi < 0.0, "root {j} is not bracketed: {flo}, {fhi}");
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if f(mid) * flo <= 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

/// The continuum spatial frequency of elastic mode `j` (1-based), Hz.
fn oracle_hz(j: usize, kappa: f64, l: f64) -> f64 {
    let beta = beta_l(j) / l;
    kappa * beta * beta / (2.0 * std::f64::consts::PI)
}

fn cents(measured: f64, reference: f64) -> f64 {
    1200.0 * (measured / reference).log2()
}

// -- the operator ---------------------------------------------------------------------------------

/// `K = h D2^T D2` is symmetric *by construction*, which is what makes the energy identity exact
/// rather than approximate. Asserted as an exact structural property, not a tolerance.
#[test]
fn the_stiffness_is_symmetric_to_the_bit() {
    for n in [8, 40, 100] {
        let p = params(n, 1.0, 0.0);
        assert!(
            p.stiffness.is_symmetric(),
            "N = {n}: K is not its own transpose"
        );
    }
}

/// The money test of the free-edge operator: `K {1, x} = 0` to machine precision, and `K x^2` is
/// emphatically not.
///
/// The tolerance is **relative**, because the absolute residual of `K x` scales as `eps ||K||`
/// and `||K|| ~ 1/h^3` — an absolute bar would be N-dependent and flaky. The discriminating signal
/// is the contrast: the two rigid-body directions sit twelve orders below a quadratic, which has
/// genuine curvature energy. A dropped stiffness term would kill `x^2` too, and that is the 1D
/// rehearsal of the 2D dropped-`(1-nu)` catch.
#[test]
fn the_stiffness_annihilates_its_rigid_body_nullspace_and_nothing_else() {
    let p = params(100, 1.0, 0.0);
    let x = p.grid();
    let k_fro = dot(p.stiffness.data(), p.stiffness.data()).sqrt();

    let rel = |v: &[f64]| {
        let kv = p.stiffness.matvec(v);
        dot(&kv, &kv).sqrt() / (k_fro * dot(v, v).sqrt())
    };
    let one = rel(&vec![1.0; p.nodes()]);
    let lin = rel(&x);
    let quad = rel(&x.iter().map(|v| v * v).collect::<Vec<_>>());

    assert!(one < 1e-12, "K@1 is not in the nullspace: {one:.3e}");
    assert!(lin < 1e-12, "K@x is not in the nullspace: {lin:.3e}");
    assert!(quad > 1e-9, "K@x^2 is spuriously ~0: {quad:.3e}");
    assert!(
        quad > 1e6 * one.max(lin),
        "nullspace contrast too small: {quad:.3e} against {:.3e}",
        one.max(lin)
    );
}

/// The free-end closure is the `h/2` mass cell, not a hand-coded stencil row — so the mass is the
/// trapezoid rule and its total is exactly the length.
#[test]
fn the_mass_is_the_trapezoid_and_its_total_is_the_length() {
    let p = params(64, 1.0, 0.0);
    let last = p.nodes() - 1;
    assert_eq!(p.w[0], 0.5 * p.h, "the left end is not a half cell");
    assert_eq!(p.w[last], 0.5 * p.h, "the right end is not a half cell");
    for i in 1..last {
        assert_eq!(p.w[i], p.h, "node {i} is not a whole cell");
    }
    // Left-to-right, which is what makes this an equality rather than a tolerance at this size.
    let total: f64 = p.w.iter().fold(0.0, |a, b| a + b);
    assert!(
        (total - p.l).abs() < 1e-15,
        "the quadrature weights sum to {total}, not {}",
        p.l
    );
}

/// The resonator builds its operators through the shared helper rather than inlining them — the
/// single-source-of-truth claim, asserted to the bit here because both sides are this crate's.
#[test]
fn the_resonator_uses_the_operator_helper_verbatim() {
    let p = params(40, 1.0, 0.0);
    let (k, w) = free_beam_stiffness(p.n, p.h);
    assert_eq!(k.data(), p.stiffness.data(), "resonator K != helper K");
    assert_eq!(k.indices(), p.stiffness.indices(), "K pattern differs");
    assert_eq!(w.data(), p.mass.data(), "resonator W != helper W");
}

/// `np.linspace` **overwrites** its endpoint rather than computing it, and the two differ in the
/// last bit for most lengths. Same claim as the plate's, one dimension down.
#[test]
fn the_grid_overwrites_its_endpoint_rather_than_computing_it() {
    // `7 * (0.9 / 7)` is 0.9000000000000001, not 0.9; the overwrite is what makes the last node
    // exactly L. Most lengths do this and a few (0.7 among them) do not, so the fixture is
    // searched for rather than assumed -- and the assertion below is what would say it had gone.
    let p = Params::new(0.9, RHO, 40_000.0, 7, KAPPA, 0.0, THETA_DEFAULT, true).expect("valid");
    let x = p.grid();
    let step = p.l / (p.n as f64);
    assert_eq!(*x.last().unwrap(), p.l, "the endpoint was not overwritten");
    assert_ne!(
        (p.n as f64) * step,
        p.l,
        "this fixture no longer exercises the overwrite"
    );
}

// -- energy ---------------------------------------------------------------------------------------

/// The acceptance contract: a lossless run is flat to machine precision, and it stays flat at
/// `mu` far past the explicit bound `1/4` — which is the unconditional-stability claim of the
/// implicit theta-scheme stated as a measurement rather than as an argument.
///
/// The longer of one second — the Python's run (`test_energy_conserved` at 0.5, 2, 8 and the
/// explicit-could-not-run case at 16) — and the 8,000 steps this file ran before it carried them,
/// so neither bar got shorter: `mu = 16` is 5,120 steps a second here, and `mu = 50` is 1,638.
/// Every step's energy must also be positive (`test_energy_strictly_positive_when_lossless`, whose
/// 0.5 s at `mu = 2` is a prefix of this run). At `mu = 0.5` that is 163,840 steps.
///
/// The bar is the project-wide 1e-10 and is deliberately not tightened (`CLAUDE.md`).
#[test]
fn a_lossless_beam_conserves_its_energy_at_every_mu() {
    for mu in [0.5, 2.0, 8.0, 16.0, 50.0] {
        let mut b = beam(64, mu, 0.0);
        assert!(b.p.mu > 0.0);
        let res = run(&mut b, 1.0, 8_000);
        assert!(
            res.energy.iter().all(|&e| e > 0.0),
            "mu = {mu}: the energy was not positive at every step"
        );
        let drift = res.energy_drift();
        assert!(drift < DRIFT_TOL, "mu = {mu}: drift {drift:.3e}");
        assert!(
            b.u.iter().all(|v| v.is_finite()),
            "mu = {mu}: the field went non-finite"
        );
    }
}

/// With loss the energy is monotone non-increasing — and the loss actually did work, which
/// monotonicity alone does not say (a beam that never moved is monotone too).
///
/// The third assertion is the interesting one and it points at the neighbouring test. A *pluck* is
/// broadband, and this scheme's high modes underdamp badly, so a plucked beam decays far more
/// slowly than the analytic `exp(-2 sigma t)` that a single low mode obeys — by three orders here.
/// Asserting that gap rather than tolerating it turns a fixture-sensitive "did work" bar into a
/// statement about the model.
#[test]
fn a_lossy_beam_is_passive() {
    let sigma = 8.0;
    let mut b = beam(64, 2.0, sigma);
    let steps = b.p.fs as usize; // one second, as the Python ran it
    let u0 = pluck(&b.p, 0.4, 0.2);
    b.set_state(&u0, &vec![0.0; b.p.nodes()]);
    let e0 = b.energy();
    let mut last = e0;
    for step in 0..steps {
        b.step();
        let now = b.energy();
        assert!(
            now <= last + 1e-10 * e0,
            "step {step}: energy rose, {now:.6e} > {last:.6e}"
        );
        last = now;
    }
    let retained = last / e0;
    assert!(retained < 0.5, "the loss did no work: {retained:.3e} of E0");
    let single_low_mode = (-2.0 * sigma * (steps as f64) / b.p.fs).exp();
    assert!(
        retained > 10.0 * single_low_mode,
        "a broadband pluck decayed at the low-mode rate ({retained:.3e} against {single_low_mode:.3e}) -- the underdamping caveat has gone"
    );
}

/// A single **low** mode decays at `2 sigma`, where `theta Q k^2 << 1` and the analytic rate
/// holds. Initialised with a machine-precision eigenvector rather than a pluck, because a broad
/// excitation decays at a mixture of rates.
#[test]
fn a_low_mode_decays_at_twice_sigma() {
    let sigma = 6.0;
    let secs = 0.5;
    let mut b = beam(64, 2.0, sigma);
    let phi = elastic_modes(&b.p, 1)[0].1.clone();
    let u0: Vec<f64> = phi.iter().map(|v| v * 1e-3).collect();
    b.set_state(&u0, &vec![0.0; b.p.nodes()]);
    let e0 = b.energy();
    for _ in 0..(secs * b.p.fs) as usize {
        b.step();
    }
    let measured = b.energy() / e0;
    let expected = (-2.0 * sigma * secs).exp();
    let rel = (measured.ln() - expected.ln()).abs() / expected.ln().abs();
    assert!(
        rel < 0.02,
        "low-mode decay off by {:.3}% (got {measured:.3e}, want {expected:.3e})",
        100.0 * rel
    );
}

/// The damping caveat, pinned as a **wrong** behaviour rather than worked around: the theta
/// time-average makes a frequency-independent loss effectively frequency-*dependent*, at rate
/// `2 sigma (1 - theta Q k^2)` with `Q = kappa^2 mu`. So a higher mode retains *more* energy than
/// a lower one after the same time — the opposite of a real beam, and the standing reason a
/// frequency-dependent loss model is owed.
#[test]
fn a_higher_mode_underdamps_relative_to_a_lower_one() {
    let (sigma, secs) = (6.0, 0.3);
    let mut retained = Vec::new();
    for which in [0usize, 6] {
        let mut b = beam(48, 8.0, sigma); // large mu -> large k -> a visible theta Q k^2
        let phi = elastic_modes(&b.p, which + 1)[which].1.clone();
        let u0: Vec<f64> = phi.iter().map(|v| v * 1e-3).collect();
        b.set_state(&u0, &vec![0.0; b.p.nodes()]);
        let e0 = b.energy();
        for _ in 0..(secs * b.p.fs) as usize {
            b.step();
        }
        retained.push(b.energy() / e0);
    }
    assert!(
        retained[1] > retained[0],
        "high/low retained energy = {retained:?} (expected high > low)"
    );
}

/// `energy()` is in Joules and linear in the linear density: `kappa` fixed means the same grid,
/// the same `K` and the same frequencies, so doubling `rho` doubles `E` for the same field.
#[test]
fn the_energy_scales_linearly_with_density() {
    let mut a = FreeBeam::new(
        Params::new(L, 0.005, 40_960.0, 64, KAPPA, 0.0, THETA_DEFAULT, true).unwrap(),
    );
    let mut c = FreeBeam::new(
        Params::new(L, 0.010, 40_960.0, 64, KAPPA, 0.0, THETA_DEFAULT, true).unwrap(),
    );
    for b in [&mut a, &mut c] {
        let u0 = pluck(&b.p, 0.4, 0.2);
        b.set_state(&u0, &vec![0.0; b.p.nodes()]);
    }
    let ratio = c.energy() / a.energy();
    assert!((ratio - 2.0).abs() < 1e-12, "E scaled by {ratio}, not 2");
}

// -- the modal oracle ------------------------------------------------------------------------------

/// The closed-form bar. The low spatial eigenfrequencies of `K phi = mu W phi` match the roots of
/// `cos(betaL) cosh(betaL) = 1` — the free-free beam's genuine analytic spectrum, and the reason
/// this model was built before the free-edge plate.
///
/// The tier is O(h^2), not machine precision: unlike the simply-supported plate there is no exact
/// discrete eigenvector, so a fine grid is used and the bar is in cents.
#[test]
fn the_low_modes_match_the_closed_form_spectrum() {
    let p = params(200, 0.5, 0.0); // fine grid and fine timestep -> the tight regime
    let modes = elastic_modes(&p, 5);
    for (j, (mu, _)) in modes.iter().enumerate() {
        let measured = p.kappa * mu.sqrt() / (2.0 * std::f64::consts::PI);
        let want = oracle_hz(j + 1, p.kappa, p.l);
        let err = cents(measured, want).abs();
        let bar = if j == 0 { 0.5 } else { 2.0 };
        assert!(
            err < bar,
            "mode {} off by {err:.3} cents (measured {measured:.4}, oracle {want:.4})",
            j + 1
        );
    }
}

/// `test_modal_frequencies_match_closed_form`, at its own window: how many *leading* modes are
/// inside 0.5 and 2 cents of `cos(bL) cosh(bL) = 1`, counted by `pitch_horizon`, with the
/// monotone flag asserted (a prefix count is only honest on a monotone error curve) and the
/// window proved wide enough not to truncate either count.
///
/// Both bands were MEASURED on 2026-09-07 rather than picked — 2 modes at 0.5 cents and 6 at 2 —
/// and the old literals (1 and 4) survive as floors, so a regression that shortens a band cannot
/// hide inside it. They are asserted as floors, as the Python did.
///
/// The Python took these from ARPACK with `k = window + 2` and threw the two lowest away. The two
/// discarded values are checked to be genuinely ~0 on the fundamental's scale — which says the
/// solver found the rigid pair *before* the fundamental, something the exact-vector check of
/// `the_rigid_body_eigenvalues_are_zero_on_the_fundamentals_scale` cannot say.
#[test]
fn the_low_modes_sit_inside_their_measured_pitch_horizons() {
    let p = params(200, 0.5, 0.0); // fine grid + fine timestep -> the tight (O(h^2)) regime
    let window = 24;
    let all = krylov_lowest(&p, window + 2);
    let (rigid, elastic) = all.split_at(2);
    let measured: Vec<f64> = elastic.iter().map(|&mu| spatial_hz(&p, mu)).collect();
    let oracle = free_free_beam_freqs(p.kappa, p.l, window).expect("a positive count");

    let mu1_est = (4.730041 / p.l).powi(4); // the Python's literal
    let worst_rigid = nan_max(rigid.iter().map(|m| m.abs()));
    assert!(
        worst_rigid < 1e-6 * mu1_est,
        "rigid-body modes not ~0: {rigid:?}"
    );

    for (bound, floor) in [(0.5, 1usize), (2.0, 4)] {
        let (horizon, monotone) =
            pitch_horizon(&measured, &oracle, bound).expect("same lengths, positive bound");
        let errors: Vec<f64> = measured
            .iter()
            .zip(&oracle)
            .map(|(&m, &o)| analysis_cents(m, o).abs())
            .collect();
        assert!(
            monotone,
            "pitch error not monotone at {bound} cents, so the prefix count hides a mode: {errors:?}"
        );
        assert!(
            horizon < window,
            "horizon {horizon} truncated by the window; widen past {window}"
        );
        assert!(
            horizon >= floor,
            "only the first {horizon} modes are within {bound} cents of cos(bL)cosh(bL)=1 (this \
             fixture claimed {floor} until 2026-09-07)"
        );
        println!("{bound} cents: horizon {horizon}; errors {errors:?}");
    }
}

/// SciPy's eigenvalues, retired with their numbers: at every grid the Python suite solved on, both
/// of this file's eigensolvers reproduce the frozen record to within 20 `eps mu_max` — the unit a
/// backward-stable dense solve is accurate in (§25's rule), written against the 50-digit Rayleigh
/// quotient of LAPACK's eigenvector rather than against LAPACK, because that quotient is the more
/// accurate of the two by construction.
///
/// The record's own consistency is asserted first — `h`, the node count and `K`'s stored-entry
/// count, so a fixture that built a different beam fails loudly instead of comparing two beams —
/// and then that LAPACK itself sits within 3 of the quotient, which is what makes it the referee
/// and ARPACK (22.6 at N = 120, recorded alongside) not.
///
/// The inverse iteration is certified on exactly the modes the single-mode bars start from: the
/// fundamental at N = 64 (the `2 sigma` bar and the discrete cosine), modes 1 and 7 at N = 48 (the
/// underdamping caveat), and the low 5 and 3 of the closed-form and convergence bars.
#[test]
fn both_eigensolvers_reproduce_the_frozen_referee_at_every_grid_the_suite_solved_on() {
    let rec = reference();
    let grids = rec["grids"].as_object().expect("a grid table");
    let eps = f64::EPSILON;
    let mut report = Vec::new();
    for (key, g) in grids {
        let n: i64 = key.parse().expect("a grid size");
        let p = params(n, 1.0, 0.0);
        assert_eq!(g["h"].as_f64(), Some(p.h), "N = {n}: a different h");
        assert_eq!(g["nodes"].as_u64(), Some(p.nodes() as u64), "N = {n}");
        assert_eq!(
            g["k_nnz"].as_u64(),
            Some(p.stiffness.data().len() as u64),
            "N = {n}: K stores a different number of entries"
        );
        let truth = floats(&g["rayleigh_lowest"]);
        let lapack = floats(&g["lapack_lowest"]);
        let unit = eps * g["lapack_max"].as_f64().expect("mu_max");
        let lapack_gap = nan_max(lapack.iter().zip(&truth).map(|(a, t)| (a - t).abs())) / unit;
        // A fact about two frozen numbers, so it cannot drift: measured 1.80 at worst (N = 50).
        assert!(
            lapack_gap < 3.0,
            "N = {n}: the record's LAPACK is {lapack_gap:.2} off"
        );
        // And the finding, pinned: ARPACK at the Python's own settings missed by 22.6 at N = 120,
        // past the bar the native solvers are held to below.
        let arpack = floats(&g["arpack_lowest"]);
        let arpack_gap = nan_max(arpack.iter().zip(&truth).map(|(a, t)| (a - t).abs())) / unit;
        if n == 120 {
            assert!(
                arpack_gap > 20.0,
                "the recorded ARPACK is no longer the outlier: {arpack_gap:.2}"
            );
        }

        // The Krylov solver's error has two terms. A floor every solve through `K` shares, in the
        // dense unit `eps mu_max`; and its stopping rule, which is RELATIVE — a Ritz pair is
        // accepted when the out-of-basis residual is below 1e-10 of `theta` (`eigs::RESIDUAL_TOL`,
        // written here as a literal so a loosened tolerance cannot move this bar with it, §29.3).
        // Neither alone is a bar: the top of N = 200's 24-mode window sits 3e-11 relative off,
        // which is 172 `eps mu_max`, while N = 400's fundamental is 0.04 `eps mu_max` off, which is
        // 7e-9 relative because `mu_1 / mu_max` is 1e-9 there. So each mode is allowed
        // `20 eps mu_max + 1e-9 |mu|`, and the rigid pair, which has no relative scale, the first
        // term alone.
        let krylov = krylov_lowest(&p, truth.len());
        let krylov_frac = nan_max(
            krylov
                .iter()
                .zip(&truth)
                .map(|(a, t)| (a - t).abs() / (20.0 * unit + 1e-9 * t.abs())),
        );
        assert!(
            krylov_frac < 1.0,
            "N = {n}: Krylov is {krylov_frac:.2} of its allowance from the frozen eigenvalues"
        );

        let wanted = match n {
            64 => 1,
            48 => 7,
            200 => 5,
            _ => 3.min(truth.len() - 2),
        };
        let inverse: Vec<f64> = elastic_modes(&p, wanted).into_iter().map(|m| m.0).collect();
        let inverse_gap =
            nan_max(inverse.iter().zip(&truth[2..]).map(|(a, t)| (a - t).abs())) / unit;
        assert!(
            inverse_gap < 20.0,
            "N = {n}: inverse iteration is {inverse_gap:.2} eps mu_max from the frozen eigenvalues"
        );
        report.push((n, lapack_gap, krylov_frac, inverse_gap));
    }
    assert_eq!(report.len(), 7, "the record lost a grid: {report:?}");
    println!("(N, lapack, krylov fraction of allowance, inverse): {report:?}");
}

/// The file's own bisected roots and the analysis crate's `brentq` roots are the same numbers —
/// to `brentq`'s `xtol = 2e-12`, which the crate inherits from SciPy's defaults. The bars that use
/// the crate's oracle (the horizon, the FFT) and the bars that use this file's (the closed form,
/// the convergence order) are then checking against one spectrum, and a defect in either
/// derivation is seen here rather than only through a count that might not move.
#[test]
fn the_local_roots_are_the_analysis_crates_roots() {
    let crate_roots = free_free_beam_beta_l(24).expect("a positive count");
    let crate_hz = free_free_beam_freqs(KAPPA, L, 24).expect("a positive count");
    let mut worst: f64 = 0.0;
    for (j, (&root, &hz)) in crate_roots.iter().zip(&crate_hz).enumerate() {
        let local = beta_l(j + 1);
        let gap = (local - root).abs();
        worst = nan_max([worst, gap].into_iter());
        assert!(
            gap < 1e-11,
            "root {}: {local} here, {root} in the crate",
            j + 1
        );
        let hz_gap = (oracle_hz(j + 1, KAPPA, L) - hz).abs() / hz;
        assert!(
            hz_gap < 1e-11,
            "mode {}: the frequencies differ by {hz_gap:.2e}",
            j + 1
        );
    }
    println!("worst |root gap| = {worst:.3e}");
}

/// `test_fft_rings_at_fundamental`, end to end: the time-stepper actually rings at the discrete
/// fundamental, as the pickup and the spectrum detector see it.
///
/// The spatial fundamental comes from the eigensolver, is mapped to its 4th-power eigenvalue, and
/// through `discrete_beam_eigenfrequency` to the theta-scheme's frequency; an off-centre bump (it
/// breaks the symmetry, so the fundamental is excited) is struck and read near an end, where the
/// free-free modes have antinodes. `a_single_mode_advances_as_the_discrete_cosine` makes the
/// stronger pointwise claim about the scheme; this one is about the path a listener's signal takes.
#[test]
fn a_struck_beam_rings_at_its_discrete_fundamental() {
    let mut b = beam(120, 1.0, 0.0);
    let f_spatial = spatial_hz(&b.p, krylov_lowest(&b.p, 4)[2]);
    let mu = (2.0 * PI * f_spatial / b.p.kappa).powi(2);
    let f_disc = discrete_beam_eigenfrequency(mu, b.p.kappa, b.p.k, b.p.theta);

    let u0 = pluck(&b.p, 0.3, 0.15);
    b.set_state(&u0, &vec![0.0; b.p.nodes()]);
    let pickup = (0.12 * b.p.n as f64) as usize;
    let steps = (0.5 * b.p.fs) as usize;
    let res = simulate(&mut b, steps, Some(pickup), 0).expect("a beam step cannot fail");
    let out = res.output.expect("a pickup was requested");
    let found = measure_partials_near(&out, res.fs, &[f_disc], Some(15.0))[0];
    let err = analysis_cents(found, f_disc).abs();
    assert!(
        err < 5.0,
        "FFT fundamental off by {err:.3} cents (found {found:.3}, want {f_disc:.3})"
    );
    println!("FFT: {err:.4} cents");
}

/// The rigid-body pair really is a *pair*, and its eigenvalues really are zero — the free
/// cross-check that the nullspace is wired right rather than merely small.
///
/// The scale is the fundamental's own `mu ~ (betaL_1 / L)^4`, because zero has no scale of its
/// own and an absolute bar here would be a statement about `L`.
#[test]
fn the_rigid_body_eigenvalues_are_zero_on_the_fundamentals_scale() {
    let p = params(120, 1.0, 0.0);
    let mu1 = (BETA_L1 / p.l).powi(4);
    let x = p.grid();
    for (label, v) in [("1", vec![1.0; p.nodes()]), ("x", x.clone())] {
        let kv = p.stiffness.matvec(&v);
        let rayleigh = dot(&kv, &v) / w_dot(&v, &v, &p.w);
        assert!(
            rayleigh.abs() < 1e-6 * mu1,
            "rigid-body direction {label} has mu = {rayleigh:.3e}, not ~0 against {mu1:.3e}"
        );
    }
}

/// The real rigour, and the one that needs no external table: the operator's eigenvalues converge
/// at **order 2** in `h`. Four grids, and the error must both fall monotonically and fall at the
/// documented rate.
#[test]
fn the_operator_eigenvalues_converge_at_second_order() {
    let ns = [50i64, 100, 200, 400]; // errors stay well above the noise floor at the finest grid
    let mut hs = Vec::new();
    let mut err_fund = Vec::new();
    let mut err_low3 = Vec::new();
    for n in ns {
        let p = params(n, 1.0, 0.0);
        let modes = elastic_modes(&p, 3);
        let f: Vec<f64> = modes
            .iter()
            .map(|(mu, _)| p.kappa * mu.sqrt() / (2.0 * std::f64::consts::PI))
            .collect();
        let oracle: Vec<f64> = (1..=3).map(|j| oracle_hz(j, p.kappa, p.l)).collect();
        hs.push(p.h);
        err_fund.push((f[0] - oracle[0]).abs());
        err_low3.push(nan_max((0..3).map(|i| (f[i] - oracle[i]).abs())));
    }
    for w in err_fund.windows(2) {
        assert!(
            w[1] < w[0],
            "fundamental error not decreasing: {err_fund:?}"
        );
    }
    for (label, err) in [("fundamental", &err_fund), ("low-3", &err_low3)] {
        let i = err.len() - 1;
        let order = (err[i - 1] / err[i]).ln() / (hs[i - 1] / hs[i]).ln();
        assert!(order > 1.9, "{label} convergence order {order:.2} < 1.9");
    }
}

/// The time discretization against its own closed form, **pointwise and at machine precision**.
///
/// For an eigenvector of `K phi = mu W phi` the theta-scheme collapses to a scalar recurrence, and
/// eliminating it gives `u^{n+1} + u^{n-1} = 2 cos(omega k) u^n` with
///
/// ```text
///     cos(omega k) = (2 - c (1 - 2 theta)) / (2 (1 + theta c)),   c = kappa^2 mu k^2
/// ```
///
/// which rearranges to `sin^2(omega k / 2) = c / (4 (1 + theta c))` — the spelling
/// `physsynth-analysis`'s `discrete_beam_eigenfrequency` uses, derived here rather than imported.
/// This is what the Python suite measured off an FFT to within five cents; as an identity of the
/// update it holds to 1e-15, and it holds at *every node*, which a spectrum cannot say.
#[test]
fn a_single_mode_advances_as_the_discrete_cosine() {
    let mut b = beam(64, 2.0, 0.0);
    let (mu, phi) = elastic_modes(&b.p, 1)[0].clone();

    let c = b.p.kappa * b.p.kappa * mu * b.p.k * b.p.k;
    let cos_wk = (2.0 - c * (1.0 - 2.0 * b.p.theta)) / (2.0 * (1.0 + b.p.theta * c));
    // Cross-check against the analysis crate's spelling, `sin^2(omega k / 2) = c / (4 (1 + theta
    // c))` — the function itself, which is what the Python's FFT bar measured against.
    let f_disc = discrete_beam_eigenfrequency(mu, b.p.kappa, b.p.k, b.p.theta);
    let omega_k = 2.0 * PI * f_disc * b.p.k;
    assert!(
        (omega_k.cos() - cos_wk).abs() < 1e-14,
        "the two spellings of the discrete frequency disagree: {} against {cos_wk}",
        omega_k.cos()
    );

    let amp = 1e-3;
    let u0: Vec<f64> = phi.iter().map(|v| v * amp).collect();
    b.set_state(&u0, &vec![0.0; b.p.nodes()]);

    let mut prev = b.u_prev.clone();
    let mut now = b.u.clone();
    let mut worst = 0.0f64;
    for _ in 0..2_000 {
        b.step();
        let next = b.u.clone();
        let here =
            nan_max((0..next.len()).map(|i| (next[i] + prev[i] - 2.0 * cos_wk * now[i]).abs()));
        worst = nan_max([worst, here].into_iter());
        prev = now;
        now = next;
    }
    // Normalised by the amplitude: a pointwise relative error is meaningless where the mode has a
    // node, and the trajectory is bounded by `amp` because the run is lossless.
    let rel = worst / amp;
    assert!(
        rel < 1e-13,
        "the mode does not follow the discrete cosine: {rel:.3e}"
    );
}

/// The start-up is the consistent second-order one, `u^{-1} = u^0 - k v^0 + 1/2 k^2 a^0` — asserted
/// as the two exact identities it implies for an eigenmode, because nothing else in the workspace
/// can see it (retirement plan §30: dropping the `1/2` passed every test in every crate, and the
/// Python suite never looked either). Energy conservation cannot see it, since any `u^{-1}` gives
/// a conserved run; the discrete cosine cannot, since the recurrence holds from every start; and
/// a phase error of order `c` is invisible to a five-cent FFT bar.
///
/// For `u^n = a_n phi` with `K phi = mu W phi` and `c = kappa^2 mu k^2`:
///
/// * **at rest** (`v^0 = 0`): `a_{-1} = (1 - c/2) a_0`, and the recurrence gives exactly
///   `a_1 - a_{-1} = theta c^2 / (1 + theta c) a_0` — second order, as documented, and not zero,
///   because `1 - c/2` is only the Taylor start of the scheme's own `cos(omega k)`. Without the
///   `1/2` the difference is about `c a_0`, four orders larger here;
/// * **launched** (`u^0 = 0`, `v^0 = V phi`): `a_{-1} = -k V` and `a_1 = k V`, so the centred
///   velocity `(u^1 - u^{-1}) / 2k` is `V` exactly. This one pins the velocity term's sign.
#[test]
fn the_start_up_is_the_consistent_second_order_one() {
    let mut b = beam(64, 2.0, 0.0);
    let (mu, phi) = elastic_modes(&b.p, 1)[0].clone();
    let c = b.p.kappa * b.p.kappa * mu * b.p.k * b.p.k;
    let (theta, k) = (b.p.theta, b.p.k);
    let nodes = b.p.nodes();

    let amp = 1e-3;
    let u0: Vec<f64> = phi.iter().map(|v| v * amp).collect();
    b.set_state(&u0, &vec![0.0; nodes]);
    let u_minus = b.u_prev.clone();
    b.step();
    let gain = theta * c * c / (1.0 + theta * c);
    let rest = nan_max((0..nodes).map(|i| (b.u[i] - u_minus[i] - gain * u0[i]).abs())) / amp;
    // Measured 9.5e-15 (about 40 eps, the solve's rounding); the missing 1/2 reads ~1.2e-4.
    assert!(
        rest < 1e-12,
        "a mode at rest does not start symmetric to second order: {rest:.3e} of the amplitude \
         (the expected gap is {gain:.3e})"
    );

    let speed = 1e-2;
    let v0: Vec<f64> = phi.iter().map(|v| v * speed).collect();
    b.set_state(&vec![0.0; nodes], &v0);
    let u_minus = b.u_prev.clone();
    b.step();
    let launched =
        nan_max((0..nodes).map(|i| ((b.u[i] - u_minus[i]) / (2.0 * k) - v0[i]).abs())) / speed;
    assert!(
        launched < 1e-12,
        "a launched mode's centred velocity is not V: {launched:.3e} of it"
    );
    println!("start-up: rest {rest:.3e}, launched {launched:.3e}, c = {c:.3e}");
}

// -- construction -----------------------------------------------------------------------------------

/// The eleven documented refusals, each with the error it is supposed to raise. `boundary_ok` is
/// `false` for the last one because the caller — the binding — is what can `repr()` the object the
/// user passed; the *order* of the checks is what this table pins.
#[test]
fn the_construction_refusals_are_the_documented_ones() {
    struct Case {
        label: &'static str,
        l: f64,
        rho: f64,
        fs: f64,
        n: i64,
        kappa: f64,
        sigma: f64,
        theta: f64,
        boundary_ok: bool,
        want: ParamError,
    }
    let base = |label, want| Case {
        label,
        l: 1.0,
        rho: 0.005,
        fs: 50_000.0,
        n: 40,
        kappa: 20.0,
        sigma: 0.0,
        theta: THETA_DEFAULT,
        boundary_ok: true,
        want,
    };
    let cases = [
        Case {
            l: 0.0,
            ..base("L = 0", ParamError::NonPositive)
        },
        Case {
            rho: 0.0,
            ..base("rho = 0", ParamError::NonPositive)
        },
        Case {
            rho: -1.0,
            ..base("rho < 0", ParamError::NonPositive)
        },
        Case {
            fs: 0.0,
            ..base("fs = 0", ParamError::NonPositive)
        },
        Case {
            kappa: 0.0,
            ..base("kappa = 0", ParamError::NonPositiveKappa)
        },
        Case {
            kappa: -1.0,
            ..base("kappa < 0", ParamError::NonPositiveKappa)
        },
        Case {
            n: 3,
            ..base("N < 4", ParamError::TooFewSegments)
        },
        Case {
            sigma: -0.1,
            ..base("sigma < 0", ParamError::NegativeSigma)
        },
        Case {
            theta: 0.0,
            ..base("theta = 0", ParamError::BadTheta(0.0))
        },
        Case {
            theta: 1.5,
            ..base("theta > 1", ParamError::BadTheta(1.5))
        },
        Case {
            boundary_ok: false,
            ..base("boundary != free", ParamError::BadBoundary)
        },
    ];
    for c in cases {
        let got = Params::new(
            c.l,
            c.rho,
            c.fs,
            c.n,
            c.kappa,
            c.sigma,
            c.theta,
            c.boundary_ok,
        );
        assert_eq!(got.err(), Some(c.want), "{}: wrong refusal", c.label);
    }
}

/// `kappa = 0` is *degenerate* here rather than merely a beam without stiffness (`u_tt = 0`), so
/// it is refused rather than special-cased — unlike the stiff string, where it is a valid limit.
/// Recorded as its own test because it is the one refusal in the table that is a physics decision.
#[test]
fn a_zero_stiffness_beam_is_refused_because_it_is_degenerate_not_because_it_is_a_string() {
    let err = Params::new(L, RHO, 50_000.0, 40, 0.0, 0.0, THETA_DEFAULT, true).unwrap_err();
    assert_eq!(err, ParamError::NonPositiveKappa);
    assert_eq!(err.to_string(), "kappa (stiffness) must be positive.");
}

/// `test_no_nan_across_mu`: no NaN and no blow-up across a sweep of `mu` at N = 40, including
/// values far past the explicit bound — there is no CFL ceiling to reject. Both the pickup at the
/// free end (node 0) and the energy must be finite at every step, over the Python's 0.3 s.
#[test]
fn nothing_blows_up_anywhere_in_the_mu_sweep() {
    for mu in [0.1, 0.5, 2.0, 8.0, 32.0] {
        let mut b = beam(40, mu, 0.0);
        let u0 = pluck(&b.p, 0.4, 0.25);
        b.set_state(&u0, &vec![0.0; b.p.nodes()]);
        let steps = (0.3 * b.p.fs) as usize;
        let res = simulate(&mut b, steps, Some(0), 0).expect("a beam step cannot fail");
        let out = res.output.expect("a pickup was requested");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "mu = {mu}: the pickup went non-finite"
        );
        assert!(
            res.energy.iter().all(|e| e.is_finite()),
            "mu = {mu}: the energy went non-finite"
        );
    }
}

/// `test_explicit_unstable_config_runs_stably`: `mu = 50` is 200 times the explicit beam bound, at
/// the Python's N = 40 for its 0.5 s, and the run still conserves — at a coarse timestep, so to
/// the Python's own `1e-9` rather than the acceptance bar.
#[test]
fn a_courant_number_two_hundred_times_the_explicit_bound_runs_and_conserves() {
    let mut b = beam(40, 50.0, 0.0);
    assert!(
        b.p.mu > 0.25,
        "this fixture no longer exceeds the explicit bound"
    );
    let u0 = pluck(&b.p, 0.4, 0.25);
    b.set_state(&u0, &vec![0.0; b.p.nodes()]);
    let steps = (0.5 * b.p.fs) as usize;
    let res = simulate(&mut b, steps, Some(0), 0).expect("a beam step cannot fail");
    assert!(res.energy.iter().all(|e| e.is_finite()));
    let drift = res.energy_drift();
    assert!(drift < 1e-9, "drift {drift:.3e} at mu = {}", b.p.mu);
    println!("mu = 50, N = 40: drift {drift:.3e}");
}

/// A valid beam reports `mu` and does not gate on it: there is no CFL ceiling to reject, which is
/// the whole reason the scheme is implicit.
#[test]
fn a_courant_number_far_past_the_explicit_bound_is_accepted() {
    let p = params(40, 50.0, 0.0);
    assert!(
        p.mu > 0.25,
        "this fixture no longer exceeds the explicit bound"
    );
    assert!(
        (p.mu - 50.0).abs() < 1e-9,
        "mu was not reproduced: {}",
        p.mu
    );
}
