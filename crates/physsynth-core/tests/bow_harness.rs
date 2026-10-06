//! The bowed string's acceptance harness at the Python suite's own rig (retirement plan §41).
//!
//! `tests/test_bow_{energy,modal,stability}.py` were the bow's validation suite; this file is where
//! their assertions went when they were deleted. It runs at `make_bowed_string`'s rig — a flexible
//! (`kappa = 0`), fixed-end 1 m string, `c = 200` m/s, `N = 100`, `lam = 0.9`, `theta = 0.28`,
//! `sigma0 = 0.5`, `sigma1 = 0.05`, bowed at `0.13` of its length at `0.1` m/s with a peak friction
//! of 1 N and a sharpness of 60 — which is **not** `bow.rs`'s rig (it bows at `0.2`, mostly at a
//! sharpness of 100), so that file keeps its own bars and this one is new, as the stiff string's
//! was (§32).
//!
//! Three groups, in the order the Python argued them:
//!
//! - **Energy** — the bow is active, so the claim is a *balance*, not conservation: lossless,
//!   `E - E0` equals the accumulated bow work to machine precision whatever the friction solve's
//!   residual; lossy, the inferred dissipation never runs backwards; a force-free bow is the bare
//!   string bit for bit.
//! - **Helmholtz motion** — what actually exercises the friction coupling: the sustained note sits
//!   at the string's own `f1 = c / 2L`, does not move with the bow speed, grows with it, slips for
//!   a fraction `beta` of each period, slips once per period, and swings to the two-slope slip
//!   velocity `-v_bow (1 - beta) / beta`. Six distinct 2.5 s runs carry all six claims, so each is
//!   simulated once and shared (`steady`).
//! - **Stability** — friction is bounded, so nothing blows up anywhere in the playable space, and
//!   the `helmholtz_number` diagnostic predicts when the bracketed fallback is needed.
//!
//! The signals are measured the way the Python measured them: `np.mean` is NumPy's pairwise sum,
//! which `reduce::sum` reproduces, and the pitch goes through the analysis crate's
//! `measure_partials_near`, which is what the Python's `spectrum` had called since phase 7.

use std::sync::OnceLock;

use physsynth_analysis::spectrum;
use physsynth_core::bow::BowedString;
use physsynth_core::engine::Resonator;
use physsynth_core::reduce;
use physsynth_core::string_damped::{self, DampedStiffString};

const L: f64 = 1.0;
const T: f64 = 200.0;
const RHO: f64 = 0.005; // -> c = 200 m/s, f1 = 100 Hz
const N: i64 = 100;
const LAM: f64 = 0.9;
const THETA: f64 = 0.28; // `string_stiff.THETA_DEFAULT`, the helper's

/// The lossless balance bar — the Python's. Observed ~6e-15.
const BALANCE_TOL: f64 = 1e-11;

/// One bowed string, as `make_bowed_string` built it. `Default` is the helper's defaults.
#[derive(Clone, Copy)]
struct Rig {
    sigma0: f64,
    sigma1: f64,
    position: f64,
    v_bow: f64,
    force: f64,
    sharpness: f64,
    /// The string's time weighting. Every carried bar runs at the default; §41.4's guard does not.
    theta: f64,
    /// The friction solve's tolerance — the binding's default, which the helper never overrode.
    newton_tol: f64,
}

impl Default for Rig {
    fn default() -> Self {
        Rig {
            sigma0: 0.5,
            sigma1: 0.05,
            position: 0.13,
            v_bow: 0.1,
            force: 1.0,
            sharpness: 60.0,
            theta: THETA,
            newton_tol: 1e-13,
        }
    }
}

fn damped(sigma0: f64, sigma1: f64, theta: f64) -> DampedStiffString {
    let c = (T / RHO).sqrt();
    let fs = c * (N as f64) / (L * LAM);
    let p = string_damped::Params::new(L, T, RHO, fs, N, 0.0, sigma0, sigma1, theta, true)
        .expect("valid string");
    DampedStiffString::new(p)
}

impl Rig {
    fn build(self) -> BowedString {
        // `newton_maxiter = 60`: the binding's default, which the helper never overrode.
        BowedString::new(
            damped(self.sigma0, self.sigma1, self.theta),
            self.position,
            self.v_bow,
            self.force,
            self.sharpness,
            self.newton_tol,
            60,
        )
        .expect("valid bow")
    }
}

/// Step `b` for `steps`, returning the energy and the accumulated bow work, step 0 included.
fn run(b: &mut BowedString, steps: usize) -> (Vec<f64>, Vec<f64>) {
    let mut e = vec![b.energy()];
    let mut w = vec![b.s.bow_work];
    for _ in 0..steps {
        b.step().expect("the friction root always exists");
        e.push(b.energy());
        w.push(b.s.bow_work);
    }
    (e, w)
}

// -- energy: the balance, passivity, decoupling -------------------------------------------------

#[test]
fn a_lossless_bowed_string_balances_energy_against_the_bow_work() {
    for (force, sharpness) in [(1.0, 60.0), (2.0, 100.0), (0.5, 40.0)] {
        let mut b = Rig {
            sigma0: 0.0,
            sigma1: 0.0,
            force,
            sharpness,
            ..Rig::default()
        }
        .build();
        let (e, w) = run(&mut b, 4000);
        let worst = e
            .iter()
            .zip(&w)
            .map(|(&ei, &wi)| ((ei - e[0]) - wi).abs() / (ei.abs() + wi.abs() + 1e-30))
            .fold(0.0, f64::max);
        eprintln!(
            "balance force={force} a={sharpness}: rel {worst:e} E {:?} W {:?} fallbacks {}",
            e[4000], w[4000], b.s.fallbacks
        );
        assert!(
            worst < BALANCE_TOL,
            "energy-balance error {worst:.2e} (force={force}, a={sharpness})"
        );
        // The bar is about the solve's whole range: the fallback must have been exercised.
        assert!(b.s.fallbacks > 0, "no slip ever needed the bracket");
    }
}

#[test]
fn the_bow_drives_a_string_up_from_rest() {
    let mut b = Rig {
        sigma0: 0.0,
        sigma1: 0.0,
        ..Rig::default()
    }
    .build();
    assert_eq!(b.energy(), 0.0);
    run(&mut b, 2000);
    eprintln!("injects: work {:?} energy {:?}", b.s.bow_work, b.energy());
    assert!(b.s.bow_work > 0.0, "bow did no net work");
    assert!(b.energy() > 1e-8, "string never gained energy from the bow");
}

#[test]
fn loss_only_ever_removes_energy() {
    for (sigma0, sigma1) in [(0.5, 0.05), (2.0, 0.0), (0.0, 0.1)] {
        let mut b = Rig {
            sigma0,
            sigma1,
            ..Rig::default()
        }
        .build();
        let (e, w) = run(&mut b, 6000);
        assert!(e.iter().all(|x| x.is_finite()), "non-finite energy");
        let d: Vec<f64> = e
            .iter()
            .zip(&w)
            .map(|(&ei, &wi)| wi - (ei - e[0]))
            .collect();
        let min_step = d
            .windows(2)
            .map(|p| p[1] - p[0])
            .fold(f64::INFINITY, f64::min);
        let w_end = w[6000];
        eprintln!(
            "loss s0={sigma0} s1={sigma1}: diss {:?} min step {min_step:?} W {w_end:?}",
            d[6000]
        );
        assert!(
            d[6000] >= -BALANCE_TOL * (w_end.abs() + 1.0),
            "loss added energy overall (sigma0={sigma0}, sigma1={sigma1})"
        );
        assert!(
            min_step >= -1e-9 * (w_end.abs() + 1.0),
            "a loss step added energy ({min_step:.2e}) at sigma0={sigma0}, sigma1={sigma1}"
        );
    }
}

#[test]
fn a_force_free_bow_is_bit_identical_to_the_bare_string() {
    // The Python's start: mode 3 at 1 mm (`modal.mode_shape(x, L, 3) * 1e-3`), released from rest.
    let mut bare = damped(0.5, 0.05, THETA);
    let mut bowed = Rig {
        force: 0.0,
        ..Rig::default()
    }
    .build();
    assert_eq!(bowed.p.node, 13);
    let x = bare.p.grid();
    let phi: Vec<f64> = physsynth_analysis::modal::mode_shape(&x, L, 3)
        .iter()
        .map(|v| v * 1e-3)
        .collect();
    let zero = vec![0.0; phi.len()];
    bare.set_state(&phi, &zero);
    bowed.string.set_state(&phi, &zero);
    for _ in 0..500 {
        bare.step();
        bowed.step().expect("root");
    }
    eprintln!(
        "zero force: sum|u| {:?}",
        bare.u.iter().map(|v| v.abs()).sum::<f64>()
    );
    assert_eq!(bowed.s.bow_force, 0.0);
    assert_eq!(bowed.s.bow_work, 0.0);
    assert_eq!(bowed.string.u, bare.u, "a force-free bow moved the string");
    assert_eq!(bowed.string.u_prev, bare.u_prev);
}

#[test]
fn the_bows_energy_is_the_strings() {
    let mut b = Rig::default().build();
    assert_eq!(b.energy(), b.string.energy());
    for _ in 0..50 {
        b.step().expect("root");
    }
    eprintln!("delegate: E50 {:?}", b.energy());
    assert!(b.energy() > 0.0, "nothing was compared");
    assert_eq!(b.energy(), b.string.energy());
}

/// The engine drives a model through this trait alone, so the claim is checked through it.
fn read_through_the_trait<R: Resonator>(r: &mut R) -> (usize, f64, f64, f64) {
    for _ in 0..50 {
        r.step().expect("root");
    }
    let s = r.state();
    (s.len(), s[10], r.displacement_at(10), r.timestep())
}

#[test]
fn the_bowed_string_is_a_resonator() {
    let mut b = Rig::default().build();
    let (len, from_state, from_pickup, k) = read_through_the_trait(&mut b);
    eprintln!("resonator: disp10 {from_pickup:?}");
    assert_eq!(len, b.string.p.n + 1);
    assert_eq!(from_state, from_pickup);
    assert_ne!(from_pickup, 0.0, "the pickup read a string at rest");
    assert_eq!(k, b.string.p.k);
}

// -- Helmholtz motion ---------------------------------------------------------------------------

const F1: f64 = 100.0; // c / 2L on this rig, exactly

/// The settled tail of one 2.5 s bowed note: the pickup signal at `N / 3` and the bow-point
/// relative velocity, the last 40% of the run — the Python's `_bow_to_steady`.
struct Steady {
    sig: Vec<f64>,
    vrel: Vec<f64>,
    fs: f64,
    beta: f64,
    v_bow: f64,
    fallbacks: usize,
}

/// The six notes the Python bowed. `Slow` is both the bow-speed sweep's first point and the
/// `beta = 0.13` slip rig (`v_bow = 0.1`, `force = 0.4`), so it is played once.
#[derive(Clone, Copy, Debug)]
enum Scene {
    Default,
    Slow,
    Mid,
    Fast,
    Slip20,
    Slip25,
}

fn steady(scene: Scene) -> &'static Steady {
    static RUNS: [OnceLock<Steady>; 6] = [const { OnceLock::new() }; 6];
    RUNS[scene as usize].get_or_init(|| {
        let speed = |v_bow: f64| Rig {
            v_bow,
            force: 4.0 * v_bow,
            ..Rig::default()
        };
        let slip = |position: f64| Rig {
            position,
            force: 0.4,
            ..Rig::default()
        };
        let rig = match scene {
            Scene::Default => Rig::default(),
            Scene::Slow => speed(0.1),
            Scene::Mid => speed(0.15),
            Scene::Fast => speed(0.2),
            Scene::Slip20 => slip(0.2),
            Scene::Slip25 => slip(0.25),
        };
        let mut b = rig.build();
        let steps = (2.5 * b.string.p.fs) as usize;
        let pickup = (N / 3) as usize;
        let mut sig = Vec::with_capacity(steps);
        let mut vrel = Vec::with_capacity(steps);
        for _ in 0..steps {
            b.step().expect("root");
            sig.push(b.string.u[pickup]);
            vrel.push(b.s.v_rel);
        }
        let i0 = ((1.0 - 0.4) * steps as f64) as usize;
        Steady {
            sig: sig[i0..].to_vec(),
            vrel: vrel[i0..].to_vec(),
            fs: b.string.p.fs,
            beta: b.beta,
            v_bow: b.p.v_bow,
            fallbacks: b.s.fallbacks,
        }
    })
}

/// `sig - sig.mean()`, with NumPy's pairwise mean.
fn centred(sig: &[f64]) -> Vec<f64> {
    let mean = reduce::sum(sig) / sig.len() as f64;
    sig.iter().map(|v| v - mean).collect()
}

fn pitch(st: &Steady) -> f64 {
    spectrum::measure_partials_near(&centred(&st.sig), st.fs, &[F1], Some(0.15 * F1))[0]
}

fn amplitude(st: &Steady) -> f64 {
    centred(&st.sig).iter().fold(0.0, |m, v| m.max(v.abs()))
}

/// The two-state split: slipping when the string swings away from the bow by half the bow speed.
fn slipping(st: &Steady) -> Vec<bool> {
    st.vrel.iter().map(|v| v.abs() >= 0.5 * st.v_bow).collect()
}

fn slip_fraction(st: &Steady) -> f64 {
    slipping(st).iter().filter(|&&s| s).count() as f64 / st.vrel.len() as f64
}

/// Stick-to-slip transitions per period of `f1` over the tail.
fn slip_onsets(st: &Steady) -> (usize, f64) {
    let s = slipping(st);
    let onsets = s.windows(2).filter(|p| !p[0] && p[1]).count();
    let periods = st.vrel.len() as f64 * F1 / st.fs;
    (onsets, onsets as f64 / periods)
}

#[test]
fn the_helmholtz_note_sits_at_the_strings_fundamental() {
    let st = steady(Scene::Default);
    let f = pitch(st);
    let cents = 1200.0 * (f / F1).log2();
    eprintln!("pitch: f {f:?} cents {cents:?} fallbacks {}", st.fallbacks);
    assert!(
        cents.abs() < 60.0,
        "bowed pitch is {cents:.1} cents off f1 = {F1}"
    );
}

#[test]
fn the_pitch_does_not_follow_the_bow_speed() {
    let freqs: Vec<f64> = [Scene::Slow, Scene::Mid, Scene::Fast]
        .iter()
        .map(|&s| pitch(steady(s)))
        .collect();
    let (lo, hi) = freqs
        .iter()
        .fold((f64::INFINITY, 0.0f64), |(l, h), &f| (l.min(f), h.max(f)));
    let spread = 1200.0 * (hi / lo).log2();
    let mid = steady(Scene::Mid);
    eprintln!(
        "speeds: freqs {freqs:?} spread {spread:?}; v_bow 0.15: slip fraction {:?} onsets {:?}",
        slip_fraction(mid),
        slip_onsets(mid)
    );
    assert!(
        spread < 25.0,
        "pitch moved {spread:.1} cents across bow speeds {freqs:?}"
    );
}

#[test]
fn the_amplitude_follows_the_bow_speed() {
    // Doubling the bow speed (force ~ v_bow holds the Schelleng window) roughly doubles the note.
    let amps: Vec<f64> = [Scene::Slow, Scene::Mid, Scene::Fast]
        .iter()
        .map(|&s| amplitude(steady(s)))
        .collect();
    let ratio = amps[2] / amps[0];
    eprintln!("amps {amps:?} ratio {ratio:?}");
    assert!(
        1.5 < ratio && ratio < 2.5,
        "amplitude ratio {ratio:.2} for 2x bow speed (expected ~2): {amps:?}"
    );
}

#[test]
fn the_slip_fraction_is_the_bow_position() {
    for scene in [Scene::Slow, Scene::Slip20, Scene::Slip25] {
        let st = steady(scene);
        let frac = slip_fraction(st);
        eprintln!(
            "slip {scene:?}: beta {:?} fraction {frac:?} error {:?} fallbacks {}",
            st.beta,
            (frac - st.beta).abs(),
            st.fallbacks
        );
        assert!(
            (frac - st.beta).abs() < 0.05,
            "slip fraction {frac:.3} != beta {:.3}",
            st.beta
        );
    }
}

#[test]
fn the_string_slips_once_per_period() {
    for scene in [Scene::Slow, Scene::Slip20, Scene::Slip25] {
        let st = steady(scene);
        let (onsets, per_period) = slip_onsets(st);
        eprintln!("slips {scene:?}: onsets {onsets} per period {per_period:?}");
        assert!(
            0.85 < per_period && per_period < 1.25,
            "{per_period:.2} slips/period (expected ~1) at beta = {:.3}",
            st.beta
        );
    }
}

#[test]
fn the_slip_velocity_is_the_helmholtz_two_slope_one() {
    let st = steady(Scene::Slow);
    let measured = st
        .vrel
        .iter()
        .map(|v| v + st.v_bow)
        .fold(f64::INFINITY, f64::min);
    let ideal = -st.v_bow * (1.0 - st.beta) / st.beta;
    eprintln!(
        "slip velocity: measured {measured:?} ideal {ideal:?} rel {:?}",
        (measured - ideal).abs() / ideal.abs()
    );
    assert!(
        (measured - ideal).abs() < 0.4 * ideal.abs(),
        "slip velocity {measured:.3} vs ideal {ideal:.3}"
    );
}

// -- stability ------------------------------------------------------------------------------------

fn settle(rig: Rig, steps: usize) -> BowedString {
    let mut b = rig.build();
    for _ in 0..steps {
        b.step().expect("the friction root always exists");
    }
    b
}

fn max_abs(u: &[f64]) -> f64 {
    u.iter().fold(0.0, |m, v| m.max(v.abs()))
}

#[test]
fn nothing_blows_up_across_force_and_bow_speed() {
    for force in [0.2, 1.0, 3.0, 8.0] {
        for v_bow in [0.02, 0.1, 0.4] {
            let b = settle(
                Rig {
                    force,
                    v_bow,
                    ..Rig::default()
                },
                3000,
            );
            eprintln!(
                "sweep1 force={force} v_bow={v_bow}: max|u| {:?} E {:?} fallbacks {}",
                max_abs(&b.string.u),
                b.energy(),
                b.s.fallbacks
            );
            assert!(
                b.string.u.iter().all(|x| x.is_finite()) && b.energy().is_finite(),
                "non-finite state (force={force}, v_bow={v_bow})"
            );
        }
    }
}

#[test]
fn nothing_blows_up_across_sharpness_and_bow_position() {
    for sharpness in [20.0, 60.0, 150.0, 400.0] {
        for position in [0.08, 0.2, 0.35, 0.5] {
            let b = settle(
                Rig {
                    sharpness,
                    position,
                    ..Rig::default()
                },
                3000,
            );
            eprintln!(
                "sweep2 a={sharpness} pos={position}: max|u| {:?} E {:?} fallbacks {} H {:?}",
                max_abs(&b.string.u),
                b.energy(),
                b.s.fallbacks,
                b.p.helmholtz_number
            );
            assert!(
                b.string.u.iter().all(|x| x.is_finite()),
                "non-finite state (a={sharpness}, position={position})"
            );
        }
    }
}

#[test]
fn the_multivalued_regime_needs_the_fallback_and_the_solve_never_fails() {
    let strong = Rig {
        force: 4.0,
        sharpness: 120.0,
        ..Rig::default()
    };
    assert!(strong.build().p.helmholtz_number > 1.0);
    let b = settle(strong, 4000);
    eprintln!(
        "strong: H {:?} fallbacks {}",
        b.p.helmholtz_number, b.s.fallbacks
    );
    assert!(
        b.s.fallbacks > 0,
        "expected some slip-event fallbacks in the multivalued regime"
    );
}

#[test]
fn the_single_valued_regime_never_needs_the_fallback() {
    let weak = Rig {
        force: 0.02,
        ..Rig::default()
    };
    assert!(weak.build().p.helmholtz_number < 1.0);
    let b = settle(weak, 3000);
    eprintln!(
        "weak: H {:?} fallbacks {}",
        b.p.helmholtz_number, b.s.fallbacks
    );
    assert!(b.s.bow_work != 0.0, "the weak bow did nothing");
    assert_eq!(
        b.s.fallbacks, 0,
        "single-valued regime should need no root fallback"
    );
}

#[test]
fn the_helmholtz_number_is_the_documented_product_at_the_pythons_bow() {
    let b = Rig::default().build();
    let expected = b.p.g * b.p.force * (2.0 * b.p.sharpness).sqrt() * 0.5_f64.exp();
    eprintln!("H {:?} g {:?}", b.p.helmholtz_number, b.p.g);
    assert_eq!(b.p.helmholtz_number, expected);
}

// -- guards the breakage round asked for (§41.4, the human's calls) ------------------------------

/// The lossless balance's worst relative error over `steps`, the Newton phase's total residual
/// evaluations, and the fallback count.
fn balance_and_cost(rig: Rig, steps: usize) -> (f64, usize, usize) {
    let mut b = rig.build();
    let e0 = b.energy();
    let (mut worst, mut evals) = (0.0f64, 0usize);
    for _ in 0..steps {
        evals += b.step().expect("root").newton_evals;
        let (e, w) = (b.energy(), b.s.bow_work);
        worst = worst.max(((e - e0) - w).abs() / (e.abs() + w.abs() + 1e-30));
    }
    (worst, evals, b.s.fallbacks)
}

#[test]
fn the_balance_holds_off_the_default_theta() {
    // The admittance `a = A^{-1} e_i` is solved once, with the string's own `theta`. Built at any
    // other `theta`, the correction and the velocity it reports still agree with each other (both
    // come from the same `a`), but `a` is no longer the string's forced response, so the work the
    // bow reports stops matching the energy the string gained — and every carried bar,
    // run at the default, cannot see it.
    for theta in [0.5, 1.0] {
        let (worst, _, fallbacks) = balance_and_cost(
            Rig {
                sigma0: 0.0,
                sigma1: 0.0,
                theta,
                ..Rig::default()
            },
            4000,
        );
        eprintln!("theta {theta}: balance {worst:e} fallbacks {fallbacks}");
        assert!(
            worst < BALANCE_TOL,
            "energy-balance error {worst:.2e} at theta = {theta}"
        );
        assert!(
            fallbacks > 0,
            "the bracket was never exercised at theta = {theta}"
        );
    }
}

#[test]
fn the_balance_is_exact_at_a_loose_solve_and_a_loose_solve_costs_less() {
    // The power is read from the TRUE post-correction velocity, so the balance is exact whatever
    // the residual — a claim only visible at a tolerance far above the balance bar.
    let lossless = Rig {
        sigma0: 0.0,
        sigma1: 0.0,
        ..Rig::default()
    };
    let (tight, tight_evals, _) = balance_and_cost(lossless, 4000);
    let (loose, loose_evals, _) = balance_and_cost(
        Rig {
            newton_tol: 1e-6,
            ..lossless
        },
        4000,
    );
    eprintln!(
        "tolerance: balance {tight:e} / {loose:e}, Newton evaluations {tight_evals} / {loose_evals}"
    );
    assert!(
        loose < BALANCE_TOL,
        "energy-balance error {loose:.2e} at newton_tol = 1e-6"
    );
    // ... and the tolerance is the one the caller passed.
    assert!(
        (loose_evals as f64) < 0.9 * tight_evals as f64,
        "a looser solve did not do less work ({loose_evals} vs {tight_evals})"
    );
}

/// Every root of the friction residual, by an independent fine scan refined by bisection.
fn all_roots(v_free: f64, p: &physsynth_core::bow::Params) -> Vec<f64> {
    use physsynth_core::bow::residual;
    let span = p.g * p.force + 6.0 / (2.0 * p.sharpness).sqrt();
    let n = 4000;
    let at = |k: usize| v_free - span + 2.0 * span * (k as f64) / (n as f64);
    let mut roots = Vec::new();
    for k in 0..n {
        let (mut lo, mut hi) = (at(k), at(k + 1));
        let rlo = residual(lo, v_free, p);
        if rlo * residual(hi, v_free, p) >= 0.0 {
            continue;
        }
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if (residual(mid, v_free, p) < 0.0) == (rlo < 0.0) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        roots.push(0.5 * (lo + hi));
    }
    roots
}

#[test]
fn the_fallback_takes_the_root_nearest_the_pre_step_velocity() {
    // At a slip the friction residual has several roots, and the physical branch is the one
    // nearest the previous step's relative velocity — not the first in the scan, and not the one
    // nearest wherever Newton wandered before it gave up. Swept over a grid of (v_free, seed) at
    // the Python's bow.
    use physsynth_core::bow::solve_v_rel;
    let p = Rig::default().build().p;
    let (mut fallbacks, mut several, mut wrong) = (0, 0, 0);
    for i in 0..120 {
        for j in 0..120 {
            let v_free = -0.6 + 1.2 * (i as f64) / 119.0;
            let seed = -0.6 + 1.2 * (j as f64) / 119.0;
            let sol = solve_v_rel(v_free, seed, &p).expect("root");
            if !sol.used_fallback {
                continue;
            }
            fallbacks += 1;
            let roots = all_roots(v_free, &p);
            several += usize::from(roots.len() > 1);
            let nearest = roots
                .iter()
                .copied()
                .min_by(|a, b| (a - seed).abs().total_cmp(&(b - seed).abs()))
                .expect("a root");
            if (nearest - sol.v_rel).abs() > 1e-9 {
                wrong += 1;
            }
        }
    }
    eprintln!("root pick: fallbacks {fallbacks} several roots {several} not nearest {wrong}");
    assert_eq!(
        wrong, 0,
        "{wrong} fallbacks took a root other than the nearest"
    );
    assert!(
        several > 100,
        "the sweep never reached the multivalued regime"
    );
}

#[test]
fn the_bow_snaps_to_the_nearest_node_not_the_one_below() {
    // `position / h` lands a hair BELOW the integer for these six (0.29 / 0.01 = 28.999...96), so
    // rounding down would put the bow one cell short.
    let cases = [
        (0.29, 29),
        (0.47, 47),
        (0.57, 57),
        (0.58, 58),
        (0.59, 59),
        (0.94, 94),
    ];
    for (position, node) in cases {
        let b = Rig {
            position,
            ..Rig::default()
        }
        .build();
        assert_eq!(b.p.node, node, "bow at {position} m");
    }
}
