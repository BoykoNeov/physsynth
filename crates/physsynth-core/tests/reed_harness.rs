//! The single reed's acceptance harness at the Python suite's own rig (retirement plan §42).
//!
//! `tests/test_reed_{energy,signature,stability}.py` were the clarinet's validation suite; this
//! file is where their assertions went when they were deleted. It runs at `make_reed_bore`'s rig —
//! a 0.5 m, 8 mm-radius closed-mouthpiece tube on `N = 200` segments at `lam = 1`, with a lightly
//! radiating bell (`R_bell = 650`), blown through the default reed: `f_reed = 2500` Hz,
//! `q_reed = 4`, `mu = 0.03`, `Sr = 1.5e-4`, `width = 1.5e-2`, `H0 = 4e-4`, solved to
//! `newton_tol = 1e-10` in at most 60 Newton steps. **Not** `reed.rs`'s rig (a 0.6 m tube, mostly
//! on 100 segments), so that file keeps its own bars and this one is new, as the stiff string's was.
//!
//! Three groups, in the order the Python argued them:
//!
//! - **Energy** — the reed is active, so the claim is a *balance*: on a lossless bore,
//!   `E^n - E^0 = mouth_work - jet_loss - reed_damp_work` to machine precision every step, with
//!   both dissipation channels sign-definite and monotone, and a radiating bell needing no extra
//!   term because the bore's `energy()` already carries what it shed.
//! - **The signature** — the balance passes on silence, so the independent oracle is that the
//!   clarinet *plays*: it speaks above a blowing threshold and not below it, locks near the bore's
//!   quarter wave whatever the reed's own frequency, is odd-harmonic, and beats shut when blown
//!   hard. Nine distinct notes carry those claims, each played once and shared (`note`).
//! - **Stability** — the reed stays bounded across the blowing range, the bore's hook is inert when
//!   unused, the Bernoulli jet is an odd passive `sqrt` law, and non-physical reeds are refused.
//!
//! The signals are measured the way the Python measured them: `np.mean` is NumPy's pairwise sum
//! (`reduce::sum`), and the spectra go through the analysis crate's `magnitude_spectrum`, which is
//! what the Python's `spectrum` had called since phase 7.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use physsynth_analysis::{modal, spectrum};
use physsynth_core::bore::{self, Bore, End};
use physsynth_core::engine::Resonator;
use physsynth_core::pyfloat::scalar_pow;
use physsynth_core::reduce;
use physsynth_core::reed::{self, ParamError, Params, ReedBore};

const L: f64 = 0.5; // `BORE_LENGTH_DEFAULT`
const RADIUS: f64 = 0.008; // `BORE_RADIUS_DEFAULT`
const RHO0: f64 = 1.2041; // the bore's default `rho0`; `make_reed_bore` never passed one
const C0: f64 = 343.0; // `C0_AIR`
const N: usize = 200;
const R_BELL: f64 = 650.0; // `R_BELL_DEFAULT`

/// The lossless balance bar — the Python's. Observed ~6e-15.
const BALANCE_TOL: f64 = 1e-11;

const CO: (End, End) = (End::Closed, End::Open);
const CR: (End, End) = (End::Closed, End::Radiating);

/// One clarinet as `make_reed_bore` built it. `Default` is the helper's defaults.
#[derive(Clone, Copy)]
struct Rig {
    p_mouth: f64,
    bc: (End, End),
    r_bell: f64,
    f_reed: f64,
    /// The reed's areal mass, jet width, rest opening and solve tolerance, and the air's density —
    /// every carried bar runs at the helper's values; §42.4's guards do not.
    mu: f64,
    width: f64,
    h0: f64,
    newton_tol: f64,
    rho0: f64,
}

impl Default for Rig {
    fn default() -> Self {
        Rig {
            p_mouth: 1500.0, // `REED_P_MOUTH_DEFAULT`
            bc: CR,
            r_bell: R_BELL,
            f_reed: 2500.0,
            mu: 0.03,
            width: 1.5e-2,
            h0: 4.0e-4,
            newton_tol: 1e-10,
            rho0: RHO0,
        }
    }
}

impl Rig {
    fn blown(p_mouth: f64) -> Rig {
        Rig {
            p_mouth,
            ..Rig::default()
        }
    }

    /// The lossless rig of the balance bars: `boundary=("closed", "open"), sigma=0`.
    fn lossless(p_mouth: f64) -> Rig {
        Rig {
            p_mouth,
            bc: CO,
            ..Rig::default()
        }
    }

    fn bore_params(&self) -> bore::Params {
        let h = L / (N as f64);
        let fs = C0 / (1.0 * h);
        bore::Params::new(
            L,
            fs,
            N,
            RADIUS,
            Some(self.bc),
            0.0,
            self.r_bell,
            self.rho0,
            C0,
        )
        .expect("bore parameters should be accepted")
    }

    fn build(&self) -> ReedBore {
        let bp = self.bore_params();
        let rp = Params::new(
            &bp,
            self.f_reed,
            4.0,
            self.mu,
            1.5e-4,
            self.width,
            self.h0,
            self.newton_tol,
            60,
        )
        .expect("reed parameters should be accepted");
        ReedBore::new(rp, Bore::new(bp), self.p_mouth)
    }
}

/// Per-step traces and the cumulative channels, every step from `E^0`.
struct Books {
    e: Vec<f64>,
    mouth: Vec<f64>,
    jet: Vec<f64>,
    damp: Vec<f64>,
    fallback_steps: Vec<usize>,
}

/// The Python's `_run`: step `steps` times, recording the energy and the three work channels.
fn books(reed: &mut ReedBore, steps: usize) -> Books {
    let mut b = Books {
        e: vec![reed.energy()],
        mouth: vec![0.0],
        jet: vec![0.0],
        damp: vec![0.0],
        fallback_steps: Vec::new(),
    };
    for i in 1..=steps {
        let before = reed.state().fallbacks;
        reed.step().expect("the monotone residual always brackets");
        if reed.state().fallbacks != before {
            b.fallback_steps.push(i);
        }
        let s = reed.state();
        b.e.push(reed.energy());
        b.mouth.push(s.mouth_work);
        b.jet.push(s.jet_loss);
        b.damp.push(s.reed_damp_work);
    }
    b
}

/// `np.max(np.abs(lhs - rhs) / scale)` with the Python's per-step scale `|e| + |mouth| + 1e-30`.
/// NaN-propagating, as `np.max` is.
fn balance(b: &Books) -> f64 {
    let e0 = b.e[0];
    let mut worst = 0.0_f64;
    for i in 0..b.e.len() {
        let lhs = b.e[i] - e0;
        let rhs = b.mouth[i] - b.jet[i] - b.damp[i];
        let rel = (lhs - rhs).abs() / (b.e[i].abs() + b.mouth[i].abs() + 1e-30);
        if rel.is_nan() || worst.is_nan() {
            worst = f64::NAN;
        } else {
            worst = worst.max(rel);
        }
    }
    worst
}

// -- energy (`test_reed_energy.py`) -----------------------------------------------------------------

#[test]
fn a_lossless_clarinet_balances_its_books_every_step() {
    // The money test, beating or not. The per-step error is linear in the scalar solve's residual,
    // so it verifies the solve converged as well as the physics; and at the two louder pressures
    // the bracketed fallback carries part of the note, so both branches are on the books.
    for p_mouth in [1000.0, 1500.0, 2500.0] {
        let mut reed = Rig::lossless(p_mouth).build();
        let b = books(&mut reed, 8000);
        let rel = balance(&b);
        println!(
            "p_mouth={p_mouth}: rel {rel:e}, E {:e}, mouth {:e}, jet {:e}, damp {:e}, fallbacks {} {:?}",
            b.e[8000], b.mouth[8000], b.jet[8000], b.damp[8000], b.fallback_steps.len(), b.fallback_steps
        );
        assert!(
            rel < BALANCE_TOL,
            "energy-balance error {rel:.2e} at p_mouth={p_mouth}"
        );
        if p_mouth > 1000.0 {
            assert!(
                !b.fallback_steps.is_empty(),
                "the bracket was never exercised at {p_mouth}"
            );
            // ...but rarely: each Newton solve starts from the previous step's root, so the cusp
            // stalls it only near a sign change of the drop. Measured 49 and 53; seeded from zero
            // instead, the same note took the fallback 1,900 times with the physics unchanged —
            // a cost that only the viewer's Windows-exact freeze could see (§42.4, the human's call).
            assert!(
                b.fallback_steps.len() < 100,
                "{} fallbacks in 8000 steps at {p_mouth}: is the solve still continuation-seeded?",
                b.fallback_steps.len()
            );
        }
    }
}

#[test]
fn the_breath_drives_the_air_column_up_from_rest() {
    let mut reed = Rig::lossless(1500.0).build();
    assert_eq!(reed.energy(), 0.0);
    books(&mut reed, 4000);
    println!("mouth {:e}, E {:e}", reed.state().mouth_work, reed.energy());
    assert!(reed.state().mouth_work > 0.0, "the mouth did no net work");
    assert!(
        reed.energy() > 1e-8,
        "the air column never gained energy from the reed"
    );
}

#[test]
fn both_dissipation_channels_are_passive_and_monotone() {
    // Each accumulator only ever grows: `dp U_B >= 0` (Bernoulli) and `Mr g y'^2 >= 0` every step.
    let mut reed = Rig::lossless(1500.0).build();
    books(&mut reed, 6000);
    println!(
        "jet {:e}, damp {:e}",
        reed.state().jet_loss,
        reed.state().reed_damp_work
    );
    assert!(
        reed.state().jet_loss >= 0.0,
        "the Bernoulli jet returned energy"
    );
    assert!(
        reed.state().reed_damp_work >= 0.0,
        "the reed damping returned energy"
    );

    let mut reed = Rig::lossless(1800.0).build();
    let b = books(&mut reed, 4000);
    let jet_step = b
        .jet
        .windows(2)
        .map(|w| w[1] - w[0])
        .fold(f64::INFINITY, f64::min);
    let damp_step = b
        .damp
        .windows(2)
        .map(|w| w[1] - w[0])
        .fold(f64::INFINITY, f64::min);
    println!("min jet step {jet_step:e}, min damp step {damp_step:e}");
    assert!(jet_step >= -1e-18, "a jet-loss step returned energy");
    assert!(damp_step >= -1e-18, "a reed-damping step returned energy");
}

#[test]
fn the_balance_survives_a_radiating_bell() {
    // `energy()` folds the bore's radiated channel into `E_bore`, so the identity holds verbatim.
    let mut reed = Rig {
        r_bell: 5e4,
        ..Rig::default()
    }
    .build();
    let b = books(&mut reed, 6000);
    let rel = balance(&b);
    println!(
        "rel {rel:e}, radiated {:e}, E {:e}, fallbacks {} {:?}",
        reed.bore().radiated_energy(),
        b.e[6000],
        b.fallback_steps.len(),
        b.fallback_steps
    );
    assert!(
        rel < BALANCE_TOL,
        "balance error with a radiating bell {rel:.2e}"
    );
    assert!(
        reed.bore().radiated_energy() > 0.0,
        "the bell should shed some energy"
    );
}

#[test]
fn the_energy_is_the_bores_plus_the_reeds_and_zero_at_rest() {
    let mut reed = Rig::lossless(1500.0).build();
    assert_eq!(reed.state().reed_energy(reed.params()), 0.0);
    assert_eq!(reed.energy(), 0.0);
    for _ in 0..200 {
        reed.step().unwrap();
    }
    let parts = reed.bore().energy() + reed.state().reed_energy(reed.params());
    println!("E {:e}", reed.energy());
    assert_eq!(reed.energy(), parts);
}

#[test]
fn the_balance_holds_off_the_default_reed_and_air() {
    // §42.4: every carried bar ran at the helper's reed and air, where a reed mass, a jet width or
    // an air density hard-wired to its default changes nothing at all. Off them, each one breaks the
    // books by 0.2% to 5% (planted, measured); built right, the balance is as exact as at home.
    let rigs = [
        (
            "mu = 0.05",
            Rig {
                mu: 0.05,
                ..Rig::lossless(1500.0)
            },
        ),
        (
            "width = 1.2e-2",
            Rig {
                width: 1.2e-2,
                ..Rig::lossless(1500.0)
            },
        ),
        (
            "rho0 = 1.18",
            Rig {
                rho0: 1.18,
                ..Rig::lossless(1500.0)
            },
        ),
    ];
    for (label, rig) in rigs {
        let mut reed = rig.build();
        let b = books(&mut reed, 8000);
        let rel = balance(&b);
        println!(
            "{label}: rel {rel:e}, mouth {:e}, fallbacks {}",
            b.mouth[8000],
            b.fallback_steps.len()
        );
        assert!(
            rel < BALANCE_TOL,
            "energy-balance error {rel:.2e} at {label}"
        );
        assert!(b.mouth[8000] > 0.0, "the breath did no work at {label}");
    }
}

#[test]
fn the_reed_rests_at_its_own_opening() {
    // A rest opening hard-wired to the default 4e-4 conserves and balances perfectly — it only
    // plays a different reed. At rest the channel is open by exactly the H0 the reed was built with.
    let reed = Rig {
        h0: 3.0e-4,
        ..Rig::default()
    }
    .build();
    assert_eq!(reed.state().reed_opening(reed.params()), 3.0e-4);
    assert_eq!(reed.params().h0, 3.0e-4);
}

#[test]
fn a_looser_solve_shows_in_the_books() {
    // The reed's per-step balance error is linear in the scalar residual (unlike the bow's, which
    // reads the true velocity), so the tolerance is visible in the books: at 1e-6 the balance error
    // grows from ~6e-15 to ~4e-11 — past the 1e-11 bar, which is why the carried bars run at
    // 1e-10. A solve that ignored its tolerance would read the same at both (§42.4).
    let mut tight = Rig::lossless(1500.0).build();
    let mut loose = Rig {
        newton_tol: 1e-6,
        ..Rig::lossless(1500.0)
    }
    .build();
    let t = balance(&books(&mut tight, 8000));
    let l = balance(&books(&mut loose, 8000));
    println!("tight {t:e}, loose {l:e}, ratio {:e}", l / t);
    assert!(
        l > 100.0 * t,
        "a 1e-6 solve balanced as well as a 1e-10 one: {l:e} vs {t:e}"
    );
    assert!(
        l < 1e-9,
        "a 1e-6 solve should still balance to ~1e-11, got {l:e}"
    );
}

// -- the signature (`test_reed_signature.py`) -----------------------------------------------------

/// One played note: the per-step mouthpiece pressure, the node-1 pressure, the channel opening and
/// the fallback count, from the first step on.
struct Note {
    mouth: Vec<f64>,
    interior: Vec<f64>,
    opening: Vec<f64>,
    k: f64,
    gamma: f64,
    fallbacks: usize,
}

/// `int(secs / reed.k)` — the Python's step count for a note `secs` long.
fn count(k: f64, secs: f64) -> usize {
    (secs / k) as usize
}

/// The note blown at `p_mouth` through a reed of `f_reed`, played for `secs` and shared: a shorter
/// read of the same note is a prefix of the same deterministic run, which is exactly what the
/// Python's separate, shorter runs computed.
fn note(p_mouth: f64, f_reed: f64, secs: f64) -> &'static Note {
    // One cell per note, handed out under the lock and filled outside it, so tests running in
    // parallel that want the same note wait for one run instead of each playing it.
    type Cells = HashMap<(u64, u64, u64), &'static OnceLock<Note>>;
    static NOTES: OnceLock<Mutex<Cells>> = OnceLock::new();
    let key = (p_mouth.to_bits(), f_reed.to_bits(), secs.to_bits());
    let cell: &'static OnceLock<Note> = NOTES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap()
        .entry(key)
        .or_insert_with(|| Box::leak(Box::new(OnceLock::new())));
    cell.get_or_init(|| play(p_mouth, f_reed, secs))
}

fn play(p_mouth: f64, f_reed: f64, secs: f64) -> Note {
    let mut reed = Rig {
        p_mouth,
        f_reed,
        ..Rig::default()
    }
    .build();
    let k = reed.params().k;
    let steps = count(k, secs);
    let mut n = Note {
        mouth: Vec::with_capacity(steps),
        interior: Vec::with_capacity(steps),
        opening: Vec::with_capacity(steps),
        k,
        gamma: reed.state().gamma(reed.params()),
        fallbacks: 0,
    };
    for _ in 0..steps {
        reed.step().expect("the monotone residual always brackets");
        n.mouth.push(reed.mouthpiece_pressure());
        n.interior.push(reed.bore().displacement_at(1));
        n.opening.push(reed.state().reed_opening(reed.params()));
    }
    n.fallbacks = reed.state().fallbacks;
    n
}

/// `np.sqrt(np.mean(x ** 2))`, NumPy's pairwise mean.
fn rms(x: &[f64]) -> f64 {
    let sq: Vec<f64> = x.iter().map(|v| v * v).collect();
    (reduce::sum(&sq) / x.len() as f64).sqrt()
}

/// The quarter-wave fundamental `c / 4L` = 171.5 Hz.
fn f1() -> f64 {
    modal::bore_resonance_frequencies(C0, L, 1, "closed-open").unwrap()[0]
}

/// The tallest bin above DC in the second half of the node-1 signal, the Python's pitch read.
fn played_pitch(n: &Note, secs: f64) -> (f64, spectrum::SpectrumOut) {
    let sig = &n.interior[..count(n.k, secs)];
    let s = spectrum::magnitude_spectrum(&sig[sig.len() / 2..], 1.0 / n.k, 2);
    let mut best = 1;
    for i in 2..s.mag.len() {
        if s.mag[i] > s.mag[best] {
            best = i;
        }
    }
    (s.freqs[best], s)
}

#[test]
fn the_note_sustains_above_threshold() {
    let n = note(1500.0, 2500.0, 0.8);
    let sig = &n.mouth[..count(n.k, 0.6)];
    let (first, second) = sig.split_at(sig.len() / 2);
    let (rms1, rms2) = (rms(first), rms(second));
    println!(
        "gamma {:e}, rms1 {rms1:e}, rms2 {rms2:e}, fallbacks {}",
        n.gamma, n.fallbacks
    );
    assert!(n.gamma > 0.4);
    assert!(
        rms2 > 100.0,
        "the note did not speak (second-half rms {rms2:.2} Pa)"
    );
    assert!(
        rms2 > 0.5 * rms1,
        "the amplitude is decaying, not a sustained regime"
    );
}

#[test]
fn the_note_is_silent_below_threshold() {
    let n = note(300.0, 2500.0, 0.6);
    let sig = &n.mouth[..count(n.k, 0.6)];
    let rms_end = rms(&sig[sig.len() / 2..]);
    println!(
        "gamma {:e}, rms_end {rms_end:e}, fallbacks {}",
        n.gamma, n.fallbacks
    );
    assert!(n.gamma < 0.2);
    assert!(
        rms_end < 5.0,
        "the note spoke below threshold (rms {rms_end:.3} Pa)"
    );
}

#[test]
fn the_threshold_lies_between_gentle_and_hard_blowing() {
    let quiet = note(600.0, 2500.0, 0.5);
    let loud = note(1500.0, 2500.0, 0.8);
    let tail = |n: &Note| {
        let sig = &n.mouth[..count(n.k, 0.5)];
        rms(&sig[sig.len() - 4000..])
    };
    let (rq, rl) = (tail(quiet), tail(loud));
    println!("rq {rq:e}, rl {rl:e}, ratio {:e}", rl / (rq + 1e-9));
    assert!(
        rl > 50.0 * (rq + 1e-9),
        "no clear speak onset (quiet {rq:.2}, loud {rl:.2})"
    );
}

#[test]
fn the_pitch_locks_near_the_quarter_wave() {
    let n = note(1500.0, 2500.0, 0.8);
    let (f0, _) = played_pitch(n, 0.8);
    let rel = (f0 - f1()).abs() / f1();
    println!("f0 {f0:e}, rel {rel:e}");
    assert!(rel < 0.1, "pitch {f0:.1} Hz not near c/4L = {:.1} Hz", f1());
}

#[test]
fn the_bore_sets_the_pitch_not_the_reed() {
    // The reed frequency swept at a fixed blowing ratio (`p_mouth` scaled with `p_closing`, i.e.
    // `f_reed^2`) so both speak; the pitch barely moves, because the reed is inward-striking.
    let mut f0s = Vec::new();
    for f_reed in [2000.0, 3000.0] {
        // `0.03 * (2.0 * np.pi * f_reed) ** 2 * 4.0e-4` — a Python float's `** 2`, libm's `pow`.
        let p_closing = (0.03 * scalar_pow(2.0 * std::f64::consts::PI * f_reed, 2.0)) * 4.0e-4;
        let n = note(0.5 * p_closing, f_reed, 0.8);
        let (f0, _) = played_pitch(n, 0.8);
        println!(
            "f_reed={f_reed}: p_closing {p_closing:e}, gamma {:e}, f0 {f0:e}, fallbacks {}",
            n.gamma, n.fallbacks
        );
        f0s.push(f0);
    }
    let worst = f0s
        .iter()
        .map(|f| (f - f1()).abs() / f1())
        .fold(0.0, f64::max);
    let spread = (f0s[0] - f0s[1]).abs() / f1();
    println!("worst {worst:e}, spread {spread:e}");
    assert!(worst < 0.1, "the pitch tracks the reed: {f0s:?}");
    assert!(spread < 0.06, "the pitch moved with the reed: {f0s:?}");
}

#[test]
fn the_odd_harmonics_dominate() {
    let n = note(1500.0, 2500.0, 0.8);
    let (_, s) = played_pitch(n, 0.8);
    let df = s.freqs[1] - s.freqs[0];
    // The Python's `peak`: the tallest bin in `[i-3, i+4)`.
    let peak = |f: f64| {
        let i = (f / df).round_ties_even() as usize;
        s.mag[1.max(i - 3)..i + 4]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    };
    let (p1, p2, p3) = (peak(f1()), peak(2.0 * f1()), peak(3.0 * f1()));
    println!("peaks {p1:e} {p2:e} {p3:e}, ratio12 {:e}", p1 / p2);
    assert!(
        p1 > 100.0 * p2,
        "the fundamental does not dwarf the second harmonic"
    );
    assert!(
        p3 > p2,
        "the third (odd) harmonic is weaker than the second (even)"
    );
}

#[test]
fn the_reed_beats_shut_when_blown_hard_and_stays_open_when_not() {
    let hard = note(1800.0, 2500.0, 0.5);
    let shut = hard.opening[..count(hard.k, 0.5)]
        .iter()
        .position(|&h| h <= 0.0);
    let beats = hard.opening.iter().filter(|&&h| h <= 0.0).count();
    println!("first beat {shut:?}, beat steps {beats}");
    assert!(
        shut.is_some(),
        "the reed never beat shut under hard blowing"
    );

    let gentle = note(300.0, 2500.0, 0.6);
    let min_open = gentle.opening[..count(gentle.k, 0.4)]
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    println!("min opening {min_open:e}");
    assert!(min_open > 0.0, "the reed beat shut although it never spoke");
}

// -- stability (`test_reed_stability.py`) ----------------------------------------------------------

/// `Bore(L=0.5, fs=1e6, N=200, radius=0.008, boundary=...)` — the stability file's own bore.
fn stability_bore(bc: (End, End)) -> bore::Params {
    bore::Params::new(L, 1e6, N, RADIUS, Some(bc), 0.0, 0.0, RHO0, C0).unwrap()
}

#[test]
fn an_inert_hook_is_bit_for_bit_the_undriven_bore() {
    // The reed's seam must be no perturbation when unused. The Python asserted two halves:
    // `source=None` against the default — an `Option` has no third spelling here, so that half is a
    // type — and a no-op callable against none, carried here at its rig.
    let p = stability_bore(CO);
    // `1e-3 * np.exp(-((bore.x - 0.15) ** 2) / (2.0 * 0.04**2))`.
    let w2 = scalar_pow(0.04, 2.0);
    let p0: Vec<f64> = p
        .grid()
        .iter()
        .map(|&x| {
            let d = x - 0.15;
            1e-3 * (-(d * d) / (2.0 * w2)).exp()
        })
        .collect();
    let mut a = Bore::new(p.clone());
    let mut b = Bore::new(p.clone());
    a.set_state(&p0, &vec![0.0; p.n]);
    b.set_state(&p0, &vec![0.0; p.n]);
    for _ in 0..3000 {
        a.step(None);
        b.step(Some(&mut |_: &mut [f64]| {}));
    }
    assert_eq!(a.p(), b.p());
    assert_eq!(a.u(), b.u());
    assert_ne!(a.p()[1], 0.0, "the comparison ran on a silent tube");
}

#[test]
fn the_reed_stays_bounded_across_the_blowing_range() {
    // Passive: the jet and the reed damping cap the stored energy, even beating hard.
    for p_mouth in [500.0, 1500.0, 2500.0, 4000.0] {
        let n = note(p_mouth, 2500.0, if p_mouth == 1500.0 { 0.8 } else { 0.4 });
        let mp = &n.mouth[..count(n.k, 0.4)];
        let peak =
            mp.iter().map(|v| v.abs()).fold(
                0.0,
                |m: f64, v| if v.is_nan() { f64::NAN } else { m.max(v) },
            );
        println!("p_mouth={p_mouth}: max/p_mouth {:e}", peak / p_mouth);
        assert!(
            mp.iter().all(|v| v.is_finite()),
            "blew up at p_mouth={p_mouth}"
        );
        assert!(
            peak < 20.0 * p_mouth,
            "the amplitude ran away at p_mouth={p_mouth}"
        );
    }
}

#[test]
fn the_clarinet_is_a_resonator() {
    // Through the engine's trait: the state is the bore's N + 1 pressure nodes, the pickup is the
    // state's entry, and the timestep is the reed's (the bore's). `isinstance` and `callable` are
    // a type, not a value.
    let mut reed = Rig::default().build();
    let k = reed.params().k;
    let r: &mut dyn Resonator = &mut reed;
    assert_eq!(r.state().len(), N + 1);
    assert_eq!(r.displacement_at(10), 0.0);
    assert_eq!(r.timestep(), k);
    for _ in 0..400 {
        r.step().unwrap();
    }
    assert_eq!(r.displacement_at(1), r.state()[1]);
    assert_ne!(r.displacement_at(1), 0.0, "the breath never reached node 1");
}

#[test]
fn the_jet_is_an_odd_passive_square_root_law() {
    // The Python's numbers (`rho = 1.2`): zero at zero drop, zero through a shut reed, odd in the
    // drop, `dp U_B >= 0`, and four times the drop gives twice the flow.
    assert_eq!(reed::bernoulli_flow(0.0, 4e-4, 1.5e-2, 1.2), 0.0);
    assert_eq!(reed::bernoulli_flow(1000.0, 0.0, 1.5e-2, 1.2), 0.0);
    let up = reed::bernoulli_flow(1000.0, 4e-4, 1.5e-2, 1.2);
    let dn = reed::bernoulli_flow(-1000.0, 4e-4, 1.5e-2, 1.2);
    assert!(up > 0.0);
    assert_eq!(dn, -up);
    assert!(1000.0 * up >= 0.0);
    let u1 = reed::bernoulli_flow(500.0, 4e-4, 1.5e-2, 1.2);
    let u4 = reed::bernoulli_flow(2000.0, 4e-4, 1.5e-2, 1.2);
    println!("up {up:e}, u4/u1 {:e}", u4 / u1);
    assert!(
        (u4 - 2.0 * u1).abs() <= 1e-12 * (2.0 * u1),
        "u4 = {u4:e}, 2 u1 = {:e}",
        2.0 * u1
    );
}

#[test]
fn the_reed_refuses_what_the_python_refused() {
    let default = |bp: &bore::Params, f_reed, q_reed, mu, sr, h0| {
        Params::new(bp, f_reed, q_reed, mu, sr, 1.5e-2, h0, 1e-10, 60)
    };
    // A mouthpiece that is not a closed wall.
    let err = default(
        &stability_bore((End::Open, End::Closed)),
        2500.0,
        4.0,
        0.03,
        1.5e-4,
        4e-4,
    )
    .unwrap_err();
    assert_eq!(err, ParamError::MouthpieceNotClosed("open"));
    assert!(err.to_string().contains("closed"));

    // An absurdly stiff reed on a coarse-time bore: `wr k = 2.513 >= 2`.
    let coarse = bore::Params::new(L, 2e4, 20, RADIUS, Some(CO), 0.0, 0.0, RHO0, C0).unwrap();
    let err = default(&coarse, 8000.0, 4.0, 0.03, 1.5e-4, 4e-4).unwrap_err();
    assert!(matches!(err, ParamError::CflViolated(_)));
    assert_eq!(
        err.to_string(),
        "reed CFL violated: wr*k = 2.513 >= 2 (reed too stiff for the timestep). \
         Raise the sample rate (finer bore grid / larger lam) or lower f_reed."
    );

    // Non-physical values: `mu = 0`, `H0 < 0`, `Sr = 0`, `q_reed < 0`.
    let bp = stability_bore(CO);
    for err in [
        default(&bp, 2500.0, 4.0, 0.0, 1.5e-4, 4e-4).unwrap_err(),
        default(&bp, 2500.0, 4.0, 0.03, 1.5e-4, -1e-4).unwrap_err(),
        default(&bp, 2500.0, 4.0, 0.03, 0.0, 4e-4).unwrap_err(),
        default(&bp, 2500.0, -1.0, 0.03, 1.5e-4, 4e-4).unwrap_err(),
    ] {
        assert_eq!(err, ParamError::NonPositiveScalar);
        assert_eq!(
            err.to_string(),
            "f_reed, q_reed, mu, Sr, width, H0 must all be positive."
        );
    }
}

#[test]
fn the_closing_pressure_and_the_blowing_ratio_are_the_documented_ones() {
    // `p_closing = mu wr^2 H0` and `gamma = p_mouth / p_closing`, to the Python's `rel=1e-12`, with
    // the expected values spelled as the Python spelled them (`reed.wr**2` is libm's `pow`).
    let reed = Rig::blown(1000.0).build();
    let p = reed.params();
    let expected = (p.mu * scalar_pow(p.wr, 2.0)) * p.h0;
    println!(
        "p_closing {:e}, expected {expected:e}, gamma {:e}",
        p.p_closing,
        reed.state().gamma(p)
    );
    assert!((p.p_closing - expected).abs() <= 1e-12 * expected);
    let gamma = 1000.0 / expected;
    assert!((reed.state().gamma(p) - gamma).abs() <= 1e-12 * gamma);
}
