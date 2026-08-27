//! Native bars for the tension-modulated string, model #9 (plan §19).
//!
//! What can be asserted without Python: the project's acceptance contract (lossless energy
//! conservation, passivity under loss), the physics that only a *nonlinear* string has (tension
//! rises and never falls, the pitch glides up with amplitude), the reduction anchor that chains
//! this model to model #3 — `EA = 0` — asserted as **exact** equality, and the construction
//! rejections.
//!
//! # Three things here exist because no fixture in the Python suite reaches them
//!
//! - [`bracket_doubling_terminates_and_is_counted`]. The bracket-doubling loop is a *dormant*
//!   branch: swept across ten fixtures spanning amplitude, grid, damping and Courant number, it
//!   fired **zero** times, because the `max(I^{n+1}(0), I^{n-1})` seed already brackets. §16.6's
//!   Armijo hazard again — a safety net nothing in the suite exercises is a safety net nothing in
//!   the suite has ever *checked*. So it is driven here directly.
//! - [`stretch_int_association_is_left_to_right`]. `_stretch_int` sums
//!   `(dot + u_0^2) + u_last^2`, and grouping the two end terms first is a different number. That
//!   was the batch's one real porting error, and the trajectory did **not** see it (§19.4), so the
//!   spelling is pinned here rather than left to a parity run.
//! - [`the_stretch_squares_with_pow_and_not_a_multiply`]. §17.2: a test that pins an arithmetic
//!   spelling asserts nothing in `--release` unless the exponent is opaque to LLVM, which is what
//!   `scalar_pow`'s `#[inline(never)]` buys. Both profiles are run in CI for exactly this. Its
//!   first version added a second lesson the hard way: opacity is what makes the difference
//!   *observable*, but whether one exists at all belongs to the C library, so a hardcoded witness
//!   is a claim about the runner. It separates on UCRT, does not on the CI runner, and was red for
//!   two batches. The count is now searched for and reported, and only self-consistency asserted.

use physsynth_core::pyfloat::scalar_pow;
use physsynth_core::string_damped as damped;
use physsynth_core::string_nonlinear as nl;

const L: f64 = 1.0;
const T: f64 = 200.0;
const RHO: f64 = 0.005;
const KAPPA: f64 = 1.5;

/// A model #9 string. `ea = 0.0` is model #3 and every other value is the nonlinearity.
fn params(n: i64, kappa: f64, ea: f64, sigma0: f64, sigma1: f64) -> nl::Params {
    nl::Params::new(
        L,
        T,
        RHO,
        44100.0,
        n,
        kappa,
        ea,
        sigma0,
        sigma1,
        nl::THETA,
        nl::TENSION_TOL_DEFAULT,
        true,
    )
    .expect("fixture must construct")
}

/// The matching model #3 string — the `EA = 0` anchor's other end.
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
        damped::THETA,
        true,
    )
    .expect("fixture must construct")
}

/// `amp * sin(m pi x / L)` on the full grid — an exact discrete eigenvector of both `D2` and the
/// biharmonic, which is what makes the closed-form Duffing collapse exact.
fn mode(p: &nl::Params, m: usize, amp: f64) -> Vec<f64> {
    p.grid()
        .iter()
        .map(|x| amp * (std::f64::consts::PI * (m as f64) * x / L).sin())
        .collect()
}

/// A tension-modulated string started on one mode.
fn started(p: nl::Params, m: usize, amp: f64) -> nl::TensionModulatedString {
    let u0 = mode(&p, m, amp);
    let v0 = vec![0.0; p.nodes()];
    let mut s = nl::TensionModulatedString::new(p);
    s.set_state(&u0, &v0);
    s
}

// == the acceptance contract =====================================================================

#[test]
fn lossless_energy_is_conserved() {
    // The bar is CLAUDE.md's 1e-10 relative drift; the nonlinear term must not leak.
    for &amp in &[1e-3, 5e-3, 1e-2] {
        let mut s = started(params(64, KAPPA, 4.0 * T, 0.0, 0.0), 1, amp);
        let e0 = s.energy();
        let mut worst = 0.0f64;
        for _ in 0..2000 {
            s.step().expect("the tension solve must converge");
            worst = worst.max((s.energy() - e0).abs() / e0);
        }
        assert!(
            worst < 1e-10,
            "drift {worst:.3e} at amplitude {amp} exceeds 1e-10"
        );
    }
}

#[test]
fn the_nonlinear_term_is_a_real_fraction_of_the_energy() {
    // A nonlinearity bug HIDES at small amplitude, where the test merely re-runs the linear
    // scheme (model #6's lesson). So the conservation bar above is only meaningful if the
    // stretch term actually carries weight in these fixtures.
    let mut s = started(params(64, KAPPA, 20.0 * T, 0.0, 0.0), 1, 3e-2);
    let mut worst = 0.0f64;
    for _ in 0..500 {
        s.step().expect("the tension solve must converge");
        worst = worst.max(s.nonlinear_energy() / s.energy());
    }
    assert!(
        worst > 1e-2,
        "the nonlinear term is only {worst:.3e} of E — this fixture is linear"
    );
}

#[test]
fn losses_make_the_energy_monotone() {
    for &(sigma0, sigma1) in &[(2.0, 0.0), (0.0, 1e-4), (1.0, 5e-5)] {
        let mut s = started(params(64, KAPPA, 4.0 * T, sigma0, sigma1), 1, 5e-3);
        let mut prev = s.energy();
        for n in 0..600 {
            s.step().expect("the tension solve must converge");
            let e = s.energy();
            assert!(
                e <= prev * (1.0 + 1e-12),
                "energy rose at step {n} ({prev:.6e} -> {e:.6e}) with sigma0={sigma0}, \
                 sigma1={sigma1}"
            );
            prev = e;
        }
    }
}

#[test]
fn tension_only_ever_rises() {
    // `I >= 0` always, so transverse motion cannot un-stretch a string: hardening, never
    // softening. This is the sign check the whole model rests on.
    let mut s = started(params(48, 0.0, 6.0 * T, 0.0, 0.0), 2, 8e-3);
    for _ in 0..800 {
        s.step().expect("the tension solve must converge");
        assert!(
            s.tension() >= s.p.t,
            "tension {} fell below T0 {}",
            s.tension(),
            s.p.t
        );
        assert!(
            s.delta_tension >= 0.0,
            "dT went negative: {}",
            s.delta_tension
        );
    }
}

#[test]
fn a_string_at_rest_stays_at_rest() {
    // dT_hi <= 0 is a real branch: no stretch means no modulation and nothing to root-find.
    let p = params(32, KAPPA, 4.0 * T, 0.0, 0.0);
    let nodes = p.nodes();
    let mut s = nl::TensionModulatedString::new(p);
    s.set_state(&vec![0.0; nodes], &vec![0.0; nodes]);
    for _ in 0..50 {
        s.step().expect("a string at rest must converge trivially");
        assert!(s.u.iter().all(|&v| v == 0.0), "a string at rest moved");
        assert_eq!(s.delta_tension, 0.0);
        assert!(s.converged);
    }
}

// == the nonlinearity, as physics ================================================================

#[test]
fn pitch_rises_with_amplitude() {
    // Hardening, measured: the mode's period shortens as the amplitude grows. Zero crossings of
    // the modal projection, not a spectrum — the peak moves tens of percent and a window anchored
    // on the linear frequency would simply miss it (model #6's lesson).
    let mut previous = f64::INFINITY;
    for &amp in &[1e-3, 6e-3, 1.2e-2] {
        let p = params(64, 0.0, 8.0 * T, 0.0, 0.0);
        let shape = mode(&p, 1, 1.0);
        let k = p.k;
        let mut s = started(p, 1, amp);
        let period = half_period(&mut s, &shape, k);
        assert!(
            period < previous,
            "amplitude {amp} did not shorten the period ({period:.6e} vs {previous:.6e})"
        );
        previous = period;
    }
}

/// Time (s) to the first descending zero crossing of the modal projection, interpolated.
fn half_period(s: &mut nl::TensionModulatedString, shape: &[f64], k: f64) -> f64 {
    let project = |u: &[f64]| -> f64 {
        let num: f64 = u.iter().zip(shape).map(|(a, b)| a * b).sum();
        let den: f64 = shape.iter().map(|b| b * b).sum();
        num / den
    };
    let mut prev = project(&s.u);
    for n in 1..200_000 {
        s.step().expect("the tension solve must converge");
        let cur = project(&s.u);
        if prev > 0.0 && cur <= 0.0 {
            return ((n - 1) as f64 + prev / (prev - cur)) * k;
        }
        prev = cur;
    }
    panic!("no zero crossing in 200,000 steps");
}

// == the anchor: EA = 0 is model #3, exactly =====================================================

#[test]
fn ea_zero_is_model_three_bit_for_bit() {
    // The chain anchor, in the form the Python suite asserts it: `array_equal`, not a tolerance.
    // It holds because the `EA = 0` branch is model #3's code path — the prefactored matrix and
    // the same right-hand side — not because the tension happens to come out zero.
    for &(kappa, sigma0, sigma1) in &[(0.0, 0.0, 0.0), (KAPPA, 0.0, 0.0), (KAPPA, 1.5, 8e-5)] {
        let pn = params(48, kappa, 0.0, sigma0, sigma1);
        let pd = damped_params(48, kappa, sigma0, sigma1);
        let u0 = mode(&pn, 1, 3e-3);
        let v0 = vec![0.0; pn.nodes()];

        let mut a = nl::TensionModulatedString::new(pn);
        let mut b = damped::DampedStiffString::new(pd);
        a.set_state(&u0, &v0);
        b.set_state(&u0, &v0);
        assert_eq!(
            a.u_prev, b.u_prev,
            "the consistent start must be model #3's"
        );

        for n in 0..1500 {
            a.step().expect("EA = 0 cannot fail to converge");
            b.step();
            assert_eq!(a.u, b.u, "state diverged at step {n} with kappa={kappa}");
            assert_eq!(a.energy(), b.energy(), "energy diverged at step {n}");
            assert_eq!(
                a.nonlinear_energy(),
                0.0,
                "EA = 0 must carry no stretch energy"
            );
            assert_eq!(a.delta_tension, 0.0);
        }
    }
}

// == the branches and spellings no fixture reaches ===============================================

#[test]
fn bracket_doubling_terminates_and_is_counted() {
    // The doubling loop provably terminates (`resid -> +inf`), and no realistic fixture enters it.
    // Driven here by handing `solve_tension` a right-hand side whose free stretch is far SMALLER
    // than the previous step's, which is the "winding back up" case the seed is chosen to survive.
    let p = params(24, 0.0, 40.0 * T, 0.0, 0.0);
    let interior = p.interior();
    // u_prev with a large stretch, and a right-hand side that pulls u^{n+1} nearly flat.
    let mut u_prev = vec![0.0; p.nodes()];
    for (i, v) in u_prev.iter_mut().enumerate().take(p.nodes() - 1).skip(1) {
        *v = 2e-2 * (std::f64::consts::PI * (i as f64) / (p.n as f64)).sin();
    }
    let rhs0 = vec![1e-9; interior];
    let s = nl::solve_tension(&rhs0, &u_prev, &p).expect("the bracket search must terminate");
    assert!(
        s.converged,
        "a bracket exists for any finite state — the loop must find it"
    );
    assert!(
        s.delta_tension > 0.0,
        "a stretched history must raise the tension"
    );
    // The residual really is a root there, to the tolerance the solve was asked for.
    let resid = |d_t: f64| {
        let d2_up = p.op_d2.matvec(&u_prev[1..u_prev.len() - 1]);
        let u = nl::update_for(d_t, &rhs0, &d2_up, &p).expect("A stays SPD for dT >= 0");
        d_t - (p.ea / (4.0 * p.l)) * (nl::stretch_int(&u, &p) + nl::stretch(&u_prev, &p))
    };
    let scale = s.delta_tension.abs().max(1e-30);
    assert!(
        resid(s.delta_tension).abs() / scale < 1e-9,
        "residual {:.3e} at the returned root is not a root",
        resid(s.delta_tension)
    );
}

#[test]
fn stretch_int_association_is_left_to_right() {
    // `(dot + u_0^2) + u_last^2`, not `dot + (u_0^2 + u_last^2)`. The two groupings disagree in the
    // last bit on ordinary vectors, and the STATE does not see the difference (§19.4) — so the
    // spelling is pinned here, against a witness searched for on the Python side and carried over.
    let p = params(4, 0.0, 0.0, 0.0, 0.0);
    let u = [
        -15.143_835_037_313_956,
        3.949_818_627_495_3,
        -6.705_658_236_878_794_5,
    ];
    let du: Vec<f64> = (1..u.len()).map(|i| u[i] - u[i - 1]).collect();
    let d = physsynth_core::string_stiff::dot(&du, &du);
    let left = ((d + scalar_pow(u[0], 2.0)) + scalar_pow(u[2], 2.0)) / p.h;
    let ends_first = (d + (scalar_pow(u[0], 2.0) + scalar_pow(u[2], 2.0))) / p.h;
    assert_ne!(
        left, ends_first,
        "the witness no longer separates the two groupings"
    );
    assert_eq!(
        nl::stretch_int(&u, &p),
        left,
        "stretch_int must sum left to right, as `_stretch_int` does"
    );
}

#[test]
fn the_stretch_squares_with_pow_and_not_a_multiply() {
    // §17.3's spelling, at this module's own call site rather than in the abstract: the two end
    // terms are `** 2` on a Python float, which is the C library's `pow`. §17.2 is why this must
    // also be run in `--release` -- with a visible exponent LLVM folds the call into a multiply
    // and the distinction evaporates. `scalar_pow`'s `#[inline(never)]` is what keeps it alive,
    // and `black_box` on this side's own exponents is the belt to that pair of braces.
    //
    // WHAT MAY AND MAY NOT BE A§ERTED HERE, and the first version of this test had it backwards.
    // It pinned a single witness, searched on Windows/UCRT, and asserted that `pow` and a multiply
    // *disagree* on it. That is a claim about the runner's C library rather than about the port,
    // and it turned CI red for two batches while passing in both profiles on the machine that
    // wrote it -- so §17.2's "run it in release" was necessary and is not sufficient.
    //
    // What is machine-specific is the WITNESS, not the phenomenon, and the runner said so itself.
    // The same red run that failed here had `test_rust_parity_mallet` and `test_rust_parity_collision`
    // pass on that machine, and both of those separate `pow` from a multiply at values of their
    // own -- so witnesses exist on the runner and this hardcoded one merely is not among them.
    // How dense they are, and where, is the C library's business and no port'''s: UCRT and the
    // runner'''s libm round the same call differently, which is §14.2 arriving in a test.
    //
    // So this is `collision`'s rule -- report the count, assert only self-consistency -- arriving
    // in the one place its own comment said it would (§16.2). The teeth are kept where the
    // library grows them: the witness is *searched for* rather than hardcoded, and the multiply is
    // excluded only if the search finds one.
    let p = params(4, 0.0, 0.0, 0.0, 0.0);
    let two = std::hint::black_box(2.0);

    // The claim that holds on every machine: the end terms go through `scalar_pow`, whichever way
    // that rounds. Bit for bit, and regardless of whether a multiply would have agreed. This also
    // catches the fold happening inside the *port* -- there the two sides would part company.
    let x = 1.034_084_420_585_755_3;
    let u = [x, x, x]; // du = 0, so the value is exactly the two end terms
    assert_eq!(
        nl::stretch_int(&u, &p),
        ((0.0 + scalar_pow(x, two)) + scalar_pow(x, two)) / p.h,
        "the end terms must be `pow`, not a multiply"
    );

    // How often the two spellings part company is a property of the C library, so it is counted
    // and reported, never required (§14.2). 225 in 400,000 on Windows/UCRT.
    let mut witnesses = 0usize;
    let mut first: Option<f64> = None;
    for i in 1..200_000 {
        let y = 1.0 + 1e-5 * f64::from(i);
        if scalar_pow(y, two) != y * y {
            witnesses += 1;
            first.get_or_insert(y);
        }
    }
    println!("`pow` and a multiply differ in {witnesses} of 199,999 samples on this C library");
    assert!(
        witnesses < 199_999,
        "the two spellings cannot disagree everywhere"
    );

    // Only where the library actually separates them is the multiply spelling detectable at all --
    // and there, `stretch_int` must not be it.
    if let Some(y) = first {
        let v = [y, y, y];
        assert_ne!(
            nl::stretch_int(&v, &p),
            ((0.0 + y * y) + y * y) / p.h,
            "the stretch squared with a multiply"
        );
    }
}

// == construction ================================================================================

#[test]
fn rejections_carry_the_python_messages() {
    let bad = |args: (f64, f64, f64, f64, i64, f64, f64, f64, f64, f64, f64, bool)| {
        nl::Params::new(
            args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7, args.8, args.9,
            args.10, args.11,
        )
        .expect_err("must be rejected")
        .to_string()
    };
    let ok = (
        L,
        T,
        RHO,
        44100.0,
        32,
        0.0,
        0.0,
        0.0,
        0.0,
        nl::THETA,
        1e-13,
        true,
    );

    let mut a = ok;
    a.1 = -1.0;
    assert_eq!(bad(a), "L, T, rho, fs must all be positive.");
    let mut a = ok;
    a.4 = 1;
    assert_eq!(bad(a), "N must be >= 2 (need at least one interior node).");
    let mut a = ok;
    a.5 = -1.0;
    assert_eq!(bad(a), "kappa (stiffness) must be >= 0.");
    let mut a = ok;
    a.6 = -1.0;
    assert_eq!(bad(a), "EA (axial stiffness) must be >= 0.");
    let mut a = ok;
    a.7 = -1.0;
    assert_eq!(bad(a), "sigma0 (frequency-independent loss) must be >= 0.");
    let mut a = ok;
    a.8 = -1.0;
    assert_eq!(bad(a), "sigma1 (frequency-dependent loss) must be >= 0.");
    let mut a = ok;
    a.9 = 1.5;
    assert_eq!(bad(a), "theta must be in (0, 1], got 1.5.");
    let mut a = ok;
    a.10 = 0.0;
    assert_eq!(bad(a), "tension_tol must be > 0.");
    let mut a = ok;
    a.11 = false;
    assert_eq!(bad(a), "boundary must be 'supported'.");

    // The check ORDER is Python's, so a doubly-invalid set reports the same one on both sides.
    let mut a = ok;
    a.5 = -1.0;
    a.6 = -1.0;
    assert_eq!(bad(a), "kappa (stiffness) must be >= 0.");
}

#[test]
fn derived_quantities_and_the_grid() {
    let p = params(64, KAPPA, 3.0 * T, 0.0, 0.0);
    assert_eq!(p.c, (T / RHO).sqrt());
    assert_eq!(p.h, L / 64.0);
    assert_eq!(p.k, 1.0 / 44100.0);
    assert_eq!(p.lam, p.c * p.k / p.h);
    assert_eq!(p.ea_over_t, 3.0 * T / T);
    assert_eq!(p.nodes(), 65);
    assert_eq!(p.interior(), 63);
    let x = p.grid();
    assert_eq!(x[0], 0.0);
    assert_eq!(x[64], L, "linspace pins its endpoint, and so must this");
    // `_ab_D2` is D2's three upper bands; D2 is tridiagonal, so the second one is empty.
    assert!(p.ab_d2[..p.interior()].iter().all(|&v| v == 0.0));
}

#[test]
fn the_dt_zero_factor_is_model_threes() {
    // `_chol0` is the `EA = 0` path's factor, and it must be the same object model #3 factors —
    // otherwise the anchor above would be comparing two different solves that happen to agree.
    let pn = params(48, KAPPA, 7.0 * T, 1.0, 3e-5);
    let pd = damped_params(48, KAPPA, 1.0, 3e-5);
    // Both models build `A0`'s bands through the SAME function, so this is not a check of that
    // function — it is a check that model #9 derives the same INPUTS to it (`L`, `D2` and the three
    // scaled coefficients) as model #3 does. The independent check on the builder itself is the
    // `sigma1 = 0` anchor in `string_stiff.rs`, where model #2 has its own.
    assert_eq!(
        pn.op_l, pd.op_l,
        "the two models must assemble the same operator"
    );
    assert_eq!(
        pn.ab0,
        physsynth_core::string_damped::update_matrix_bands(
            &pd.op_l,
            &pd.op_d2,
            pd.sigma0 * pd.k,
            pd.theta * scalar_pow(pd.k, 2.0),
            pd.sigma1 * pd.k,
            pd.sigma1 != 0.0,
        )
    );
    assert_eq!(
        pn.chol0, pd.chol,
        "the dT = 0 factor is model #3's, to the bit"
    );
}
