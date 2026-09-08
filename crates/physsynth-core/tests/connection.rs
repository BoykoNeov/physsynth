//! Native acceptance bars for `connection::SympatheticStrings` — several strings sharing one
//! bridge point on a common modal body (piano unisons, sitar and tanpura sympathetics;
//! HANDOFF §12.B).
//!
//! These replace `tests/test_sympathetic.py`, retired in the same commit under the retirement
//! plan's naming rule; each bar below names the Python test it carries over, and the two that
//! changed shape say why.
//!
//! Energy conservation and passivity follow automatically from the linear-leapfrog structure, so
//! they are necessary but **not discriminating** — a flipped coupling sign would still conserve
//! energy. The sharp bars are the structural ones the energy report cannot see:
//!
//! * the **antisymmetric normal mode** — two identical strings started `u_B = -u_A` with the body
//!   at rest keep the bridge exactly still, and stay `u_B = -u_A` to machine precision;
//! * its **symmetric** contrast, where the bridge swings and energy floods into the body;
//! * **sympathetic transfer** — plucking one string rings up a second far more when it is tuned to
//!   the first than when it is detuned;
//! * and the coupling's **shape**: one step of the coupled set is one step of each free part plus
//!   exactly one spring force, at one node.

use physsynth_core::body::{self, ModalBody};
use physsynth_core::connection::{ConnectionError, SympatheticStrings};
use physsynth_core::exciter::triangular_pluck;
use physsynth_core::string_ideal::{Boundary, IdealString, Params};

/// The project's lossless acceptance bar (`CLAUDE.md`), deliberately not tightened.
const DRIFT_TOL: f64 = 1e-10;

// -- the suite's fixture, `tests/helpers.py::make_sympathetic` ---------------------------------

const L_DEFAULT: f64 = 1.0;
const T_DEFAULT: f64 = 200.0;
/// `c = sqrt(T / rho) = 200 m/s`, fundamental `f1 = c / 2L = 100 Hz`.
const RHO_DEFAULT: f64 = 0.005;
const BODY_FREQS_DEFAULT: [f64; 4] = [110.0, 196.0, 261.0, 440.0];
const BODY_MASS_DEFAULT: f64 = 0.02;
const K_BRIDGE_DEFAULT: f64 = 8000.0;

/// How a set is built, mirroring the Python helper's keyword defaults one field at a time.
struct Fixture {
    n_strings: usize,
    n: i64,
    lam: f64,
    k: Vec<f64>,
    tensions: Option<Vec<f64>>,
    sigma_string: f64,
    sigma_body: f64,
}

impl Default for Fixture {
    fn default() -> Fixture {
        Fixture {
            n_strings: 2,
            n: 100,
            lam: 0.9,
            k: vec![K_BRIDGE_DEFAULT],
            tensions: None,
            sigma_string: 0.0,
            sigma_body: 0.0,
        }
    }
}

impl Fixture {
    /// One stiffness for every string, the helper's scalar-`K` broadcast.
    fn stiffness(mut self, k: f64) -> Fixture {
        self.k = vec![k];
        self
    }

    fn strings(mut self, n_strings: usize) -> Fixture {
        self.n_strings = n_strings;
        self
    }

    fn lam(mut self, lam: f64) -> Fixture {
        self.lam = lam;
        self
    }

    /// Per-string tensions — how the transfer bars detune a neighbour.
    fn tensions(mut self, tensions: Vec<f64>) -> Fixture {
        self.tensions = Some(tensions);
        self
    }

    /// The `2 sigma u_t` loss coefficient on every string.
    fn string_loss(mut self, sigma: f64) -> Fixture {
        self.sigma_string = sigma;
        self
    }

    /// The per-mode loss on the shared body.
    fn body_loss(mut self, sigma: f64) -> Fixture {
        self.sigma_body = sigma;
        self
    }

    /// The sample rate every part shares: `c0 N / (L lambda)`, with `c0` from the **first**
    /// string's tension so that string 0 sits at the requested `lambda`.
    fn fs(&self) -> f64 {
        let t0 = self.tensions.as_ref().map_or(T_DEFAULT, |t| t[0]);
        let c0 = (t0 / RHO_DEFAULT).sqrt();
        c0 * (self.n as f64) / (L_DEFAULT * self.lam)
    }

    fn tension(&self, j: usize) -> f64 {
        self.tensions.as_ref().map_or(T_DEFAULT, |t| t[j])
    }

    fn string(&self, j: usize) -> IdealString {
        IdealString::new(
            Params::new(
                L_DEFAULT,
                self.tension(j),
                RHO_DEFAULT,
                self.fs(),
                self.n,
                self.sigma_string,
                Some((Boundary::Fixed, Boundary::Free)),
            )
            .expect("string parameters should be accepted"),
        )
    }

    fn body(&self) -> ModalBody {
        ModalBody::new(
            body::Params::new(
                BODY_FREQS_DEFAULT.to_vec(),
                self.fs(),
                vec![self.sigma_body; BODY_FREQS_DEFAULT.len()],
                vec![BODY_MASS_DEFAULT; BODY_FREQS_DEFAULT.len()],
                vec![1.0; BODY_FREQS_DEFAULT.len()],
                None,
            )
            .expect("body parameters should be accepted"),
        )
    }

    /// The stiffness vector as given: one entry broadcast to every string, or the list itself.
    fn ks(&self) -> Vec<f64> {
        if self.k.len() == 1 {
            vec![self.k[0]; self.n_strings]
        } else {
            self.k.clone()
        }
    }

    fn build(&self) -> Result<SympatheticStrings, ConnectionError> {
        let strings: Vec<IdealString> = (0..self.n_strings).map(|j| self.string(j)).collect();
        SympatheticStrings::new(strings, self.body(), self.ks())
    }

    fn make(&self) -> SympatheticStrings {
        self.build().expect("the set should be accepted")
    }
}

/// `triangular_pluck(s.x, L, 0.137 L, amplitude)` on string `j`, the others left at rest.
fn pluck(symp: &mut SympatheticStrings, j: usize, amplitude: f64) {
    let shape = pluck_shape(symp.strings()[j].params(), amplitude);
    let rest = vec![0.0; shape.len()];
    symp.string_mut(j).set_state(&shape, &rest);
}

fn pluck_shape(p: &Params, amplitude: f64) -> Vec<f64> {
    triangular_pluck(&p.grid(), p.l, 0.137 * p.l, amplitude).expect("the pluck should be inside")
}

/// `max |E^n / E^0 - 1|` over the run — `SimResult.energy_drift`, and the same spelling
/// `tests/plate.rs` uses.
fn drift(symp: &mut SympatheticStrings, steps: usize) -> f64 {
    let e0 = symp.energy();
    let mut worst: f64 = 0.0;
    for _ in 0..steps {
        symp.step();
        worst = worst.max((symp.energy() / e0 - 1.0).abs());
    }
    worst
}

/// One second of this fixture's audio, in steps.
fn seconds(symp: &SympatheticStrings, secs: f64) -> usize {
    (secs * symp.strings()[0].params().fs) as usize
}

// -- criterion 1: the TOTAL energy is conserved (no single string's is) --------------------------

/// Replaces `test_total_energy_conserved_across_lambda`.
#[test]
fn the_total_energy_is_conserved_across_lambda() {
    for lam in [0.9, 0.7, 0.5] {
        let mut symp = Fixture::default().lam(lam).make();
        pluck(&mut symp, 0, 1e-3);
        let steps = seconds(&symp, 1.5);
        let worst = drift(&mut symp, steps);
        assert!(worst < DRIFT_TOL, "total drift {worst:.2e} at lam={lam}");
    }
}

/// Replaces `test_total_energy_conserved_across_count`.
#[test]
fn the_total_energy_is_conserved_across_string_count() {
    for n_strings in [2, 3, 4] {
        let mut symp = Fixture::default()
            .strings(n_strings)
            .stiffness(6000.0)
            .make();
        pluck(&mut symp, 0, 1e-3);
        let steps = seconds(&symp, 1.0);
        let worst = drift(&mut symp, steps);
        assert!(
            worst < DRIFT_TOL,
            "drift {worst:.2e} for {n_strings} strings"
        );
    }
}

// -- passivity: with loss anywhere, the total decreases monotonically ----------------------------

/// Replaces `test_passivity_with_body_damping`.
#[test]
fn a_lossy_body_makes_the_total_decrease_monotonically() {
    let mut symp = Fixture::default().body_loss(10.0).make();
    pluck(&mut symp, 0, 1e-3);
    let steps = seconds(&symp, 1.5);
    assert_monotone_decreasing(&mut symp, steps);
}

/// Replaces `test_passivity_with_string_damping`.
#[test]
fn lossy_strings_make_the_total_decrease_monotonically() {
    let mut symp = Fixture::default().string_loss(5.0).make();
    pluck(&mut symp, 0, 1e-3);
    let steps = seconds(&symp, 1.5);
    assert_monotone_decreasing(&mut symp, steps);
}

/// `np.all(np.diff(res.energy) <= 1e-12 * e0)` — a step may not *gain* more than roundoff.
fn assert_monotone_decreasing(symp: &mut SympatheticStrings, steps: usize) {
    let e0 = symp.energy();
    let mut previous = e0;
    for step in 0..steps {
        symp.step();
        let e = symp.energy();
        assert!(
            e - previous <= 1e-12 * e0,
            "energy rose by {:.3e} at step {step} (bar {:.3e})",
            e - previous,
            1e-12 * e0
        );
        previous = e;
    }
}

// -- the shape of the coupling ------------------------------------------------------------------

/// Replaces `test_single_string_bit_identical_to_string_body_bridge`, **restated**.
///
/// The Python bar compared a one-string set against `StringBodyBridge` — a second implementation
/// of the same spring, written to cross-check a transcription. Both are Rust now and the bridge
/// class lives only in the binding crate, so that comparison would be the code agreeing with
/// itself. What it was really protecting is the *shape* of the coupling, and that is assertable
/// against the free parts: one step of the coupled set is one step of each part on its own, plus
/// exactly one spring force `F = K (u_end - w_b)` — subtracted at the string's end node alone,
/// weighted by `2 k^2 / (rho h)`, and handed to the body unscaled. Bit for bit, because every
/// difference here is elementwise arithmetic in the same order.
#[test]
fn one_step_is_the_free_parts_plus_exactly_one_spring_force() {
    let f = Fixture::default().strings(1);
    let mut symp = f.make();
    let shape = pluck_shape(f.string(0).params(), 1e-3);
    let rest = vec![0.0; shape.len()];
    symp.string_mut(0).set_state(&shape, &rest);
    let q0 = [1e-3, -0.5e-3, 0.7e-3, 0.3e-3];
    symp.body_mut().set_state(&q0, &[0.0; 4]);

    // The same two objects, uncoupled, carrying the same state.
    let mut free_string = f.string(0);
    free_string.set_state(&shape, &rest);
    let mut free_body = f.body();
    free_body.set_state(&q0, &[0.0; 4]);

    let p = free_string.params().clone();
    let beta = 2.0 * p.k * p.k / (p.rho * p.h);
    let last = free_string.u.len() - 1;

    for step in 0..500 {
        // The force is read from the state *before* anything moves, and the bridge displacement
        // is the body's own read-out, not a re-derivation.
        let force = K_BRIDGE_DEFAULT * (free_string.u[last] - free_body.bridge_displacement());
        free_string.step();
        free_string.u[last] -= beta * force;
        free_body.step(force);

        symp.step();

        assert_eq!(
            symp.strings()[0].u,
            free_string.u,
            "the string diverged from free-step-plus-one-force at step {step}"
        );
        assert_eq!(
            symp.body().q(),
            free_body.q(),
            "the body diverged from one force at step {step}"
        );
    }
}

/// Replaces `test_K0_bit_identical_to_uncoupled_parts`. Ports unchanged: both of its referents —
/// a string alone and a body alone — exist natively, so this stays a comparison against something
/// other than itself.
#[test]
fn at_zero_stiffness_the_set_is_bit_identical_to_the_uncoupled_parts() {
    let f = Fixture::default()
        .strings(2)
        .stiffness(0.0)
        .tensions(vec![T_DEFAULT, 0.5 * T_DEFAULT]);
    let mut symp = f.make();

    let shape0 = pluck_shape(f.string(0).params(), 1e-3);
    let p1 = f.string(1);
    let shape1 = triangular_pluck(&p1.params().grid(), L_DEFAULT, 0.29 * L_DEFAULT, 7e-4)
        .expect("the pluck should be inside");
    let rest = vec![0.0; shape0.len()];
    let q0 = [1e-3, -0.5e-3, 0.7e-3, 0.3e-3];
    symp.string_mut(0).set_state(&shape0, &rest);
    symp.string_mut(1).set_state(&shape1, &rest);
    symp.body_mut().set_state(&q0, &[0.0; 4]);

    let mut r0 = f.string(0);
    let mut r1 = f.string(1);
    let mut rb = f.body();
    r0.set_state(&shape0, &rest);
    r1.set_state(&shape1, &rest);
    rb.set_state(&q0, &[0.0; 4]);

    for _ in 0..400 {
        symp.step();
        r0.step();
        r1.step();
        rb.step(0.0);
    }
    assert_eq!(symp.strings()[0].u, r0.u, "string 0 coupled at K = 0");
    assert_eq!(symp.strings()[1].u, r1.u, "string 1 coupled at K = 0");
    assert_eq!(symp.body().q(), rb.q(), "body coupled at K = 0");
}

// -- THE discriminating oracle: the antisymmetric mode ------------------------------------------

/// Replaces `test_antisymmetric_mode_keeps_bridge_still`.
///
/// Two identical strings, equal springs, body at rest, started `u_B = -u_A`. By symmetry the total
/// bridge force is `-2 K w_b`, so a bridge at `w_b = 0` feels zero force and stays there forever
/// while `u_B` stays exactly `-u_A`. Energy conservation cannot see this; a flipped sign can.
///
/// The Python original's three thresholds were absolute (`1e-13`, `1e-15`) on a fixture plucked to
/// `1e-3`, which makes them claims about the amplitude someone picked. They are relative here:
/// the bridge against the string's own displacement scale, the body against the initial energy.
#[test]
fn the_antisymmetric_mode_keeps_the_bridge_exactly_still() {
    let amplitude = 1e-3;
    let mut symp = Fixture::default().strings(2).stiffness(8000.0).make();
    let shape = pluck_shape(symp.strings()[0].params(), amplitude);
    let mirrored: Vec<f64> = shape.iter().map(|v| -v).collect();
    let rest = vec![0.0; shape.len()];
    symp.string_mut(0).set_state(&shape, &rest);
    // The exact antisymmetric initial condition: the linear Taylor start negates too.
    symp.string_mut(1).set_state(&mirrored, &rest);

    let e0 = symp.string_energy(0) + symp.string_energy(1);
    let mut max_wb: f64 = 0.0;
    let mut max_ebody: f64 = 0.0;
    for step in 0..4000 {
        symp.step();
        max_wb = max_wb.max(symp.bridge_displacement(false).abs());
        max_ebody = max_ebody.max(symp.body().energy());
        let (a, b) = (&symp.strings()[0].u, &symp.strings()[1].u);
        let worst = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x + y).abs())
            .fold(0.0, f64::max);
        assert!(
            worst <= 1e-12 * amplitude,
            "u_B drifted from -u_A by {worst:.2e} at step {step}"
        );
    }
    assert!(
        max_wb < 1e-10 * amplitude,
        "bridge moved (max |w_b| = {max_wb:.2e}); coupling sign or sum wrong"
    );
    assert!(
        max_ebody < 1e-13 * e0,
        "body gained energy ({max_ebody:.2e}) from a still bridge"
    );
}

/// Replaces `test_symmetric_mode_drives_bridge` — the contrast that makes the zero mean something.
#[test]
fn the_symmetric_mode_drives_the_bridge() {
    let mut symp = Fixture::default().strings(2).stiffness(8000.0).make();
    let shape = pluck_shape(symp.strings()[0].params(), 1e-3);
    let rest = vec![0.0; shape.len()];
    symp.string_mut(0).set_state(&shape, &rest);
    // Symmetric: both ends push the bridge the same way.
    symp.string_mut(1).set_state(&shape, &rest);

    let e0 = symp.string_energy(0) + symp.string_energy(1);
    let mut max_ebody: f64 = 0.0;
    for _ in 0..seconds(&symp, 0.5) {
        symp.step();
        max_ebody = max_ebody.max(symp.body().energy());
    }
    assert!(
        max_ebody > 0.05 * e0,
        "symmetric mode barely moved the body ({:.1}%)",
        100.0 * max_ebody / e0
    );
}

// -- sympathetic transfer -----------------------------------------------------------------------

/// Pluck string 0; return the peak fraction of the *total* energy ever held by string 1.
///
/// A softer bridge (`K = 1500`) makes the coupling frequency-selective — the resonant transfer
/// that is the whole point of sympathetic strings — so a tuned neighbour drains most of the energy
/// while a detuned one barely responds.
fn peak_neighbour_fraction(tensions: Vec<f64>) -> f64 {
    let f = Fixture::default()
        .strings(2)
        .stiffness(1500.0)
        .tensions(tensions);
    let mut symp = f.make();
    pluck(&mut symp, 0, 1e-3);
    let steps = seconds(&symp, 1.5);
    let mut peak: f64 = 0.0;
    for _ in 0..steps {
        symp.step();
        peak = peak.max(symp.string_energy(1) / symp.energy());
    }
    peak
}

/// Replaces `test_sympathetic_transfer_tuned_beats_detuned` — the model's money oracle.
#[test]
fn a_tuned_neighbour_rings_up_far_more_than_a_detuned_one() {
    // Unison: near-complete exchange. Four semitones flat: stays quiet.
    let tuned = peak_neighbour_fraction(vec![200.0, 200.0]);
    let detuned = peak_neighbour_fraction(vec![200.0, 120.0]);
    assert!(
        tuned > 0.5,
        "a tuned unison neighbour should drain most of the energy ({:.1}%)",
        100.0 * tuned
    );
    assert!(
        detuned < 0.25,
        "the detuned neighbour rang too much ({:.1}%)",
        100.0 * detuned
    );
    assert!(
        tuned > 3.0 * detuned,
        "tuned {:.1}% not far above detuned {:.1}%",
        100.0 * tuned,
        100.0 * detuned
    );
}

// -- the exact dense stability guard ------------------------------------------------------------

/// Replaces `test_unstable_stiffness_rejected`.
#[test]
fn an_over_stiff_bridge_is_rejected() {
    let err = Fixture::default().stiffness(1e6).build().unwrap_err();
    match err {
        ConnectionError::Unstable(bound) => assert!(bound >= 4.0, "reported {bound}"),
        other => panic!("expected an instability refusal, got {other:?}"),
    }
    assert!(
        err.to_string()
            .starts_with("connection unstable: k^2 * lambda_max(A) = "),
        "{err}"
    );
}

/// Replaces `test_guard_holds_at_its_boundary`.
#[test]
fn the_guard_holds_just_inside_its_boundary() {
    // Two strings share the bridge, so each contributes to the coupled operator; a stiffness just
    // inside the guard must still conserve energy to machine precision.
    let mut symp = Fixture::default().strings(2).stiffness(10000.0).make();
    pluck(&mut symp, 0, 1e-3);
    let bound = symp.timestep() * symp.timestep() * symp.spectral_radius();
    assert!(bound < 4.0, "the fixture is outside the guard: {bound}");
    let steps = seconds(&symp, 1.0);
    let worst = drift(&mut symp, steps);
    assert!(
        worst < DRIFT_TOL,
        "drift {worst:.2e} near the guard boundary"
    );
}

/// New: the claim that licenses a *symmetric* eigenvalue routine inside the guard.
///
/// `A = M^-1 K` is not symmetric, and the Python guard went through LAPACK's general `eigvals` for
/// that reason. The coupled operator is nevertheless self-adjoint in the energy inner product —
/// `M^1/2 A M^-1/2` is symmetric, with `M` the trapezoidal mass diagonal (`rho h`, halved at the
/// free end that carries the spring, `m_i` per mode). That is a property of the discretization,
/// including the factor of two on the end node's spring term, and if it stopped holding the guard
/// would be answering a different question rather than answering this one imprecisely.
///
/// **The frozen `lambda_max` at the end is what keeps this from being self-consistent.** The
/// symmetry above is built from `mass_diagonal()`, which is *this module's* derivation of the
/// mass; a wrong derivation that still symmetrizes — a uniform scale, or the half cell on the
/// wrong end of two identical strings — would pass it while scaling the whole spectrum. The
/// number below was measured against LAPACK's general `eigvals` on this exact fixture through the
/// binding, before the Python went (plan §12): the two agree to 16 ulps, and any error in the mass
/// diagonal would move it by far more than that. It is written as a value and a tolerance rather
/// than a bit pattern, because 16 ulps of cross-implementation spread is the floor on what is
/// portable here (findings ledger #68).
#[test]
fn the_coupled_operator_is_self_adjoint_in_the_energy_inner_product() {
    let symp = Fixture::default().strings(2).stiffness(8000.0).make();
    let n = symp.dofs();
    let a = symp.symmetrized_operator();
    let mut worst: f64 = 0.0;
    let mut scale: f64 = 0.0;
    for i in 0..n {
        for j in 0..n {
            worst = worst.max((a[i * n + j] - a[j * n + i]).abs());
            scale = scale.max(a[i * n + j].abs());
        }
    }
    // Roundoff in the scaling multiply itself: the two halves are the same number, computed by
    // two different routes through `sqrt`.
    assert!(
        worst <= 1e-15 * scale,
        "asymmetry {worst:.3e} against scale {scale:.3e}"
    );
    // Real, and non-negative: it is a stiffness. The guard reads the top of this spectrum.
    let values = physsynth_core::eig::symmetric_eigenvalues(&a, n).expect("a finite operator");
    assert!(
        values[0] > 0.0,
        "smallest eigenvalue {} is not positive",
        values[0]
    );
    assert_eq!(values[n - 1], symp.spectral_radius());

    // LAPACK's `np.linalg.eigvals` on this fixture, through the binding, 2026-09-08.
    const LAPACK_LAMBDA_MAX: f64 = 1661856272.3158104;
    let measured = symp.spectral_radius();
    assert!(
        (measured / LAPACK_LAMBDA_MAX - 1.0).abs() < 1e-12,
        "lambda_max {measured:.10e} has moved off the LAPACK-verified \
         {LAPACK_LAMBDA_MAX:.10e}; suspect the mass diagonal before the eigensolver"
    );
}

// -- construction guards -------------------------------------------------------------------------

/// Replaces `test_mismatched_timestep_rejected`.
#[test]
fn a_mismatched_timestep_is_rejected() {
    let s = IdealString::new(
        Params::new(
            1.0,
            200.0,
            0.005,
            22000.0,
            100,
            0.0,
            Some((Boundary::Fixed, Boundary::Free)),
        )
        .expect("string parameters should be accepted"),
    );
    let b = ModalBody::new(
        body::Params::new(
            BODY_FREQS_DEFAULT.to_vec(),
            24000.0,
            vec![0.0; 4],
            vec![1.0; 4],
            vec![1.0; 4],
            None,
        )
        .expect("body parameters should be accepted"),
    );
    let err = SympatheticStrings::new(vec![s], b, vec![K_BRIDGE_DEFAULT]).unwrap_err();
    assert!(
        matches!(err, ConnectionError::TimestepMismatch(0, _, _)),
        "{err:?}"
    );
    assert!(err.to_string().contains("must share a timestep"), "{err}");
}

/// Replaces `test_right_end_must_be_free`.
#[test]
fn a_clamped_right_end_is_rejected() {
    let f = Fixture::default().strings(1);
    let s = IdealString::new(
        Params::new(
            L_DEFAULT,
            T_DEFAULT,
            RHO_DEFAULT,
            f.fs(),
            f.n,
            0.0,
            Some((Boundary::Fixed, Boundary::Fixed)),
        )
        .expect("string parameters should be accepted"),
    );
    let err = SympatheticStrings::new(vec![s], f.body(), vec![K_BRIDGE_DEFAULT]).unwrap_err();
    assert_eq!(err, ConnectionError::RightEndNotFree(0));
    assert!(err.to_string().contains("must be 'free'"), "{err}");
}

/// Replaces `test_ks_length_must_match`.
#[test]
fn one_stiffness_per_string_is_required() {
    let f = Fixture::default().strings(2);
    // Given as a list rather than broadcast: one entry for two strings.
    let strings: Vec<IdealString> = (0..2).map(|j| f.string(j)).collect();
    let err = SympatheticStrings::new(strings, f.body(), vec![8000.0]).unwrap_err();
    assert_eq!(err, ConnectionError::StiffnessCount(1, 2));
    assert!(
        err.to_string().contains("one stiffness per string"),
        "{err}"
    );
}

/// Replaces `test_empty_strings_rejected`.
#[test]
fn an_empty_set_is_rejected() {
    let f = Fixture::default();
    let err = SympatheticStrings::new(Vec::new(), f.body(), Vec::new()).unwrap_err();
    assert_eq!(err, ConnectionError::NoStrings);
    assert_eq!(err.to_string(), "need at least one string.");
}

/// New, and the counterpart the Python file never had: a negative spring is refused before the
/// guard runs, because a negative stiffness passes the spectral bound while being unphysical.
#[test]
fn a_negative_stiffness_is_rejected() {
    let f = Fixture::default().strings(2);
    let strings: Vec<IdealString> = (0..2).map(|j| f.string(j)).collect();
    let err = SympatheticStrings::new(strings, f.body(), vec![8000.0, -1.0]).unwrap_err();
    assert_eq!(err, ConnectionError::NegativeStiffness);
    assert_eq!(err.to_string(), "every bridge stiffness K must be >= 0.");
}

/// New: `lambda = 1` is refused, which the guard alone would not do — the string's Nyquist mode is
/// marginal there and the bridge spring pushes it over.
#[test]
fn a_string_at_the_courant_limit_is_rejected() {
    let f = Fixture::default().strings(1).lam(1.0);
    let strings: Vec<IdealString> = (0..1).map(|j| f.string(j)).collect();
    let err = SympatheticStrings::new(strings, f.body(), vec![K_BRIDGE_DEFAULT]).unwrap_err();
    assert_eq!(err, ConnectionError::LambdaAtLimit(0));
    assert!(err.to_string().contains("must run at lambda < 1"), "{err}");
}
