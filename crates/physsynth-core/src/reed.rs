//! Single-reed mouthpiece — the wind leg's continuous **nonlinear exciter**.
//!
//! Port of `physsynth/core/reed.py`, the acoustic dual of the bowed string: a dynamic reed valve
//! driven by a steady mouth pressure self-oscillates against a [`crate::bore::Bore`], turning a
//! constant breath into a clarinet. The reference docstring in `reed.py` is the physics; this
//! module documents only what the translation had to decide.
//!
//! # This is the first ported model whose step is not a formula
//!
//! Everything ported so far evaluates a fixed expression. The reed solves a scalar equation each
//! timestep — a safeguarded Newton continuation-seeded from the previous step's answer, with a
//! bracketed Brent fallback for the `sqrt` cusp at `dp = 0`, where the derivative is infinite and
//! Newton stalls.
//!
//! **The fallback fires.** Measured over the configurations `tests/helpers.py` builds: 4-5 times
//! per 4,000 steps at the flagship `p_mouth = 1500 Pa`, 13 at `1800`, and 219 on a coarse `N = 40`
//! grid. So `crate::root::brentq` exists, transcribed from SciPy rather than reinvented, for the
//! reason written down there. Two consequences worth stating:
//!
//! - **Which branch a step took is part of the trajectory, not a diagnostic.** If Rust takes the
//!   fallback on a different step than Python, the two separate structurally rather than by
//!   rounding, and no energy bar sees it. `fallbacks` is therefore compared step for step in the
//!   parity file, not merely at the end.
//! - **The stall test is `!(|r_new| < |r|)`, not `|r_new| >= |r|`.** The original spells it
//!   `if not (abs(r_new) < abs(r))`, which is *true* when a residual is NaN. Spelling it the other
//!   way round would silently accept a NaN Newton step and keep iterating on it.
//!
//! # The one-ulp compliance, preserved rather than tidied
//!
//! `reed` computes node 0's half-cell capacitance from the bore's **public** geometry — deliberately,
//! to avoid reaching into the bore's private update arrays — and the result is *not* bit-equal to
//! the bore's own `p_pref[0]`. The bore spells the compliance `rho0 * c0**2` (one libm `pow`); the
//! reed spells it `rho0 * c0 * c0` (two multiplies). Measured, they disagree by one ulp in 3,531 of
//! 3,552 tube/grid combinations, worst 4.1e-16 relative.
//!
//! That divergence predates the migration by a long way and its physical consequence is nil — it
//! scales an injection that is itself a correction. It is reproduced here, on both sides, because a
//! port that "fixed" the two spellings into agreement would be changing a number the acceptance
//! runs were taken with. Same class as the plan's `h ** 4` finding (§10.3), in Python-vs-Python
//! form. See `bore::Params::new` for the other half of it.
//!
//! # The coupling is energy-consistent only if one area does two jobs
//!
//! `Sr` is both the area the mouthpiece pressure acts on and the area the reed sweeps as it moves.
//! Because it is the same number, the reed-force work and the bore sweep-flow work are the same
//! reactive term with opposite sign and cancel exactly, which is what makes the balance identity
//! hold to machine precision. Two separate areas would be a different model that still looked
//! plausible.

use crate::bore::{self, Bore};
use crate::pyfloat::scalar_pow;
use crate::root::{self, RootError};

/// A construction-time rejection. Every variant's `Display` is the Python original's message
/// verbatim, because the suite matches on the text.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamError {
    /// One of `f_reed`, `q_reed`, `mu`, `Sr`, `width`, `H0` was not positive.
    NonPositiveScalar,
    /// `newton_maxiter < 1`.
    BadMaxIter,
    /// The bore's left end is not `"closed"`. Carries the token it actually was.
    MouthpieceNotClosed(&'static str),
    /// `wr * k >= 2`. Carries the offending `wr * k`.
    CflViolated(f64),
}

impl std::fmt::Display for ParamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamError::NonPositiveScalar => {
                write!(f, "f_reed, q_reed, mu, Sr, width, H0 must all be positive.")
            }
            ParamError::BadMaxIter => write!(f, "newton_maxiter must be >= 1."),
            ParamError::MouthpieceNotClosed(got) => write!(
                f,
                "the reed rides on the bore's LEFT end, which must be 'closed' (a live half-cell \
                 mouthpiece DOF), got '{got}'. Use boundary=('closed', <far end>)."
            ),
            ParamError::CflViolated(wk) => write!(
                f,
                "reed CFL violated: wr*k = {wk:.3} >= 2 (reed too stiff for the timestep). \
                 Raise the sample rate (finer bore grid / larger lam) or lower f_reed."
            ),
        }
    }
}

impl std::error::Error for ParamError {}

/// The bracket expansion gave up. The original raises `RuntimeError` and says it should be
/// impossible for a monotone residual; carries the step number so the message can match.
#[derive(Debug, Clone, PartialEq)]
pub struct BracketFailure {
    /// The step at which it happened.
    pub step: usize,
    /// `None` if the bracket never changed sign; otherwise how the Brent call itself failed.
    pub cause: Option<RootError>,
}

impl std::fmt::Display for BracketFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.cause {
            None => write!(
                f,
                "reed pressure-drop residual failed to bracket at step {} \
                 (should be impossible for the monotone residual).",
                self.step
            ),
            Some(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for BracketFailure {}

/// SciPy's `brentq` tolerances, as `reed.py` passes them. `rtol` clears SciPy's own `4 * eps`
/// floor, which is why the original may pass it at all.
const BRENT_XTOL: f64 = 1e-13;
const BRENT_RTOL: f64 = 8.9e-16;

/// Quasi-static Bernoulli volume flow `U_B` (m³/s) through the reed channel.
///
/// `U_B = width * opening * sign(dp) * sqrt(2 |dp| / rho)` — the jet velocity times the channel
/// cross-section, signed by the pressure drop. `opening` is the *clamped* height `max(H0 + y, 0)`:
/// a shut reed passes no air. A passive resistor, because `dp * U_B >= 0` always.
pub fn bernoulli_flow(dp: f64, opening: f64, width: f64, rho: f64) -> f64 {
    if opening <= 0.0 {
        return 0.0;
    }
    // `width * opening * math.copysign(math.sqrt(2.0 * abs(dp) / rho), dp)`, left to right.
    (width * opening) * ((2.0 * dp.abs()) / rho).sqrt().copysign(dp)
}

/// The validated, immutable parameter set: everything about a reed that is not its state.
///
/// `p_mouth` is **not** here. The original documents it as mutable between steps (that is how an
/// attack is played), and nothing derived from it is cached — so it belongs with the state.
#[derive(Debug, Clone)]
pub struct Params {
    /// Timestep, taken from the bore.
    pub k: f64,
    /// Reed resonance (Hz).
    pub f_reed: f64,
    /// Reed quality factor.
    pub q_reed: f64,
    /// Reed areal mass (kg/m²).
    pub mu: f64,
    /// Effective reed area (m²) — the one area that both feels the pressure and sweeps volume.
    pub sr: f64,
    /// Effective channel width (m) for the Bernoulli jet.
    pub width: f64,
    /// Reed rest opening (m).
    pub h0: f64,
    /// Air density, taken from the bore.
    pub rho: f64,
    /// Convergence tolerance on the scalar residual (Pa).
    pub newton_tol: f64,
    /// Max safeguarded-Newton iterations before the bracketed fallback.
    pub newton_maxiter: usize,
    /// Angular reed resonance `2 pi f_reed`.
    pub wr: f64,
    /// Reed damping rate `wr / q_reed`.
    pub g: f64,
    /// Lumped reed mass `mu * Sr` (kg).
    pub mr: f64,
    /// Static closing pressure `mu wr^2 H0` (Pa) — the instrument's pressure scale.
    pub p_closing: f64,
    /// Injection prefactor `k / C_0` for node 0's half-cell. See the module header's one-ulp note.
    pub p_pref0: f64,
    /// Denominator of the reed leapfrog, `1 + gk/2`.
    pub den: f64,
    /// Coefficient of `y^n` in `y^{n+1}`.
    pub cy_n: f64,
    /// Coefficient of `y^{n-1}` in `y^{n+1}`.
    pub cy_prev: f64,
    /// Coefficient of `dp_bar` in `y^{n+1}`.
    pub c_reed: f64,
    /// Reactive stiffening of the scalar equation (>= 2), constant across the run.
    pub d: f64,
}

impl Params {
    /// Validate and derive, in the original's check order.
    ///
    /// `bore` is the air column the reed will ride on; its left end must be `"closed"` — a live
    /// half-cell mouthpiece DOF — and its `h`, `S_node[0]`, `rho0`, `c0` and `k` are read here.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        bore: &bore::Params,
        f_reed: f64,
        q_reed: f64,
        mu: f64,
        sr: f64,
        width: f64,
        h0: f64,
        newton_tol: f64,
        newton_maxiter: i64,
    ) -> Result<Params, ParamError> {
        if f_reed <= 0.0 || q_reed <= 0.0 || mu <= 0.0 || sr <= 0.0 || width <= 0.0 || h0 <= 0.0 {
            return Err(ParamError::NonPositiveScalar);
        }
        if newton_maxiter < 1 {
            return Err(ParamError::BadMaxIter);
        }
        if bore.bc_left != bore::End::Closed {
            return Err(ParamError::MouthpieceNotClosed(bore.bc_left.name()));
        }

        let k = bore.k;
        let wr = 2.0 * std::f64::consts::PI * f_reed;
        let g = wr / q_reed;
        if wr * k >= 2.0 {
            return Err(ParamError::CflViolated(wr * k));
        }

        let mr = mu * sr;
        // `self.mu * self.wr * self.wr * self.H0`, left to right.
        let p_closing = ((mu * wr) * wr) * h0;

        // `(0.5 * bore.h) * bore.S_node[0] / (bore.rho0 * bore.c0 * bore.c0)`. The denominator is
        // `(rho0 * c0) * c0` — NOT the bore's own `rho0 * c0**2`. See the module header.
        let c0_cap = ((0.5 * bore.h) * bore.s_node[0]) / ((bore.rho0 * bore.c0) * bore.c0);
        let p_pref0 = k / c0_cap;

        let gk = g * k;
        let den = 1.0 + 0.5 * gk;
        // `(2.0 - (self.wr * self.k) ** 2) / self._den` — `** 2` is libm's `pow`, not a multiply
        // (plan §10.3; measured, `x ** 2 != x * x` in 79 of 200,007 random doubles). Through
        // `scalar_pow` and not a bare `powf(2.0)`, whose literal exponent LLVM folds back into a
        // multiply in release builds only (§17.2).
        let cy_n = (2.0 - scalar_pow(wr * k, 2.0)) / den;
        let cy_prev = (0.5 * gk - 1.0) / den;
        // `(self.k * self.k / self.mu) / self._den`.
        let c_reed = ((k * k) / mu) / den;
        // `2.0 + self._p_pref0 * self.Sr * self._c_reed / (2.0 * self.k)`.
        let d = 2.0 + ((p_pref0 * sr) * c_reed) / (2.0 * k);

        Ok(Params {
            k,
            f_reed,
            q_reed,
            mu,
            sr,
            width,
            h0,
            rho: bore.rho0,
            newton_tol,
            newton_maxiter: newton_maxiter as usize,
            wr,
            g,
            mr,
            p_closing,
            p_pref0,
            den,
            cy_n,
            cy_prev,
            c_reed,
            d,
        })
    }
}

/// Everything about a reed that changes: the leapfrog history, the per-step observables and the
/// cumulative energy channels.
///
/// `p_mouth` lives here rather than in [`Params`] because the original documents it as mutable
/// between steps — that is how an attack is played.
#[derive(Debug, Clone, PartialEq)]
pub struct State {
    /// Steady mouth (blowing) pressure `p_m` (Pa) — the control input.
    pub p_mouth: f64,
    /// Reed tip displacement `y^n` (m).
    pub y: f64,
    /// `y^{n-1}`.
    pub y_prev: f64,
    /// The dp-independent part of `y^{n+1}`, stashed between the injection and the commit.
    pub y_new: f64,
    /// Centered pressure drop `dp_bar` of the last step (Pa); also the Newton continuation seed.
    pub dp: f64,
    /// Centered reed velocity of the last step (m/s).
    pub reed_velocity: f64,
    /// Total volume flow into the bore `U = U_B - Sr y'` (m³/s).
    pub flow: f64,
    /// The Bernoulli jet alone (m³/s).
    pub jet_flow: f64,
    /// Cumulative `sum k p_m U` — the active breath input (J).
    pub mouth_work: f64,
    /// Cumulative `sum k dp_bar U_B` — Bernoulli dissipation, `>= 0` (J).
    pub jet_loss: f64,
    /// Cumulative `sum k Mr g y'^2` — reed damping, `>= 0` (J).
    pub reed_damp_work: f64,
    /// How many steps handed off to the bracketed fallback.
    pub fallbacks: usize,
    /// Completed steps.
    pub n: usize,
}

impl State {
    /// A reed at rest, blown at `p_mouth`: `y = y' = 0`, channel fully open at `H0`.
    pub fn at_rest(p_mouth: f64) -> State {
        State {
            p_mouth,
            y: 0.0,
            y_prev: 0.0,
            y_new: 0.0,
            dp: 0.0,
            reed_velocity: 0.0,
            flow: 0.0,
            jet_flow: 0.0,
            mouth_work: 0.0,
            jet_loss: 0.0,
            reed_damp_work: 0.0,
            fallbacks: 0,
            n: 0,
        }
    }

    /// Current clamped channel opening `H^+ = max(H0 + y, 0)` (m).
    ///
    /// `0` when the reed beats shut. This is the value frozen explicit at `n` in the Bernoulli
    /// conductance each step — it scales the passive jet and never the reactive coupling, which is
    /// what leaves the scalar residual with a single `sqrt` cusp and no clamp kink.
    pub fn reed_opening(&self, p: &Params) -> f64 {
        (p.h0 + self.y).max(0.0)
    }

    /// Stored reed mechanical energy `E_reed` (J): kinetic plus the **cross-time** potential
    /// `1/2 Mr wr^2 y^n y^{n-1}`, matching the explicit spring. Positive-definite while `wr k < 2`.
    pub fn reed_energy(&self, p: &Params) -> f64 {
        let y_dot_back = (self.y - self.y_prev) / p.k;
        // `0.5*Mr*ydb*ydb + 0.5*Mr*wr*wr*(y*y_prev)`, each chain left to right.
        ((0.5 * p.mr) * y_dot_back) * y_dot_back
            + (((0.5 * p.mr) * p.wr) * p.wr) * (self.y * self.y_prev)
    }

    /// Dimensionless blowing pressure `gamma = p_mouth / p_closing` — the clarinet control.
    pub fn gamma(&self, p: &Params) -> f64 {
        self.p_mouth / p.p_closing
    }
}

// -- the scalar coupling solve -----------------------------------------------------------------

/// `R(dp) = D (p_m - dp) - c_const - p_pref0 U_B(dp)` (Pa).
///
/// Strictly decreasing in `dp` — the linear term falls and the passive jet falls too — so the root
/// is unique.
pub fn residual(dp: f64, opening: f64, c_const: f64, p_mouth: f64, p: &Params) -> f64 {
    let u_b = bernoulli_flow(dp, opening, p.width, p.rho);
    (p.d * (p_mouth - dp) - c_const) - p.p_pref0 * u_b
}

/// Solve `R(dp) = 0`, returning `(dp, used_fallback)`.
///
/// Continuation-seeded Newton from `dp_seed` (last step's answer) with a guaranteed bracketing
/// fallback. `U_B ~ sign(dp) sqrt|dp|` has infinite slope at the origin, so a Newton step is
/// accepted only while it strictly shrinks `|R|`; otherwise the bracket takes over. Both give a
/// machine-precision root, so the applied flow — and the energy balance — is exact either way.
pub fn solve_dp(
    opening: f64,
    c_const: f64,
    dp_seed: f64,
    p_mouth: f64,
    step: usize,
    p: &Params,
) -> Result<(f64, bool), BracketFailure> {
    let sq = (2.0 / p.rho).sqrt();
    let mut dp = dp_seed;
    let mut r = residual(dp, opening, c_const, p_mouth, p);
    for _ in 0..p.newton_maxiter {
        if r.abs() <= p.newton_tol {
            return Ok((dp, false));
        }
        // `R'(dp) = -D - p_pref0 * w * opening * sqrt(2/rho) / (2 sqrt|dp|)`.
        let rp = if opening > 0.0 && dp.abs() > 1e-30 {
            let slope = ((p.width * opening) * sq) / (2.0 * dp.abs().sqrt());
            -p.d - p.p_pref0 * slope
        } else {
            -p.d
        };
        let dp_new = dp - r / rp;
        let r_new = residual(dp_new, opening, c_const, p_mouth, p);
        // `if not (abs(r_new) < abs(r))` — NaN-true, deliberately. See the module header.
        //
        // clippy wants `>=` here and it is WRONG about this one: the two spellings differ exactly
        // when a residual is NaN, and `!(a < b)` is the branch that bails out to the bracket
        // instead of iterating on a NaN forever. `tests/reed.rs::the_stall_test_is_nan_true`
        // pins it, and `the reed's own docstring` spells it the same way.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(r_new.abs() < r.abs()) {
            break; // stalled on the sqrt cusp -> robust bracket
        }
        dp = dp_new;
        r = r_new;
    }
    if r.abs() <= p.newton_tol {
        return Ok((dp, false));
    }
    Ok((bracketed_root(opening, c_const, p_mouth, step, p)?, true))
}

/// Bracket the unique root by expanding a window around `p_mouth` until `R` changes sign, then
/// hand it to [`root::brentq`].
///
/// `R(-inf) = +inf` and `R(+inf) = -inf` (the linear `-D dp` dominates the `sqrt` jet), so a
/// sign-changing bracket always exists and the failure below really is unreachable.
pub fn bracketed_root(
    opening: f64,
    c_const: f64,
    p_mouth: f64,
    step: usize,
    p: &Params,
) -> Result<f64, BracketFailure> {
    let mut span = p_mouth.abs().max(p.p_closing).max(1.0);
    let mut lo = p_mouth - span;
    let mut hi = p_mouth + span;
    let mut r_lo = residual(lo, opening, c_const, p_mouth, p);
    let mut r_hi = residual(hi, opening, c_const, p_mouth, p);
    let mut bracketed = false;
    for _ in 0..60 {
        // `if r_lo > 0.0 >= r_hi` — a chained comparison, so both halves must hold.
        if r_lo > 0.0 && 0.0 >= r_hi {
            bracketed = true;
            break;
        }
        span *= 2.0;
        lo = p_mouth - span;
        hi = p_mouth + span;
        r_lo = residual(lo, opening, c_const, p_mouth, p);
        r_hi = residual(hi, opening, c_const, p_mouth, p);
    }
    if !bracketed {
        return Err(BracketFailure { step, cause: None });
    }
    root::brentq(
        |dp| residual(dp, opening, c_const, p_mouth, p),
        lo,
        hi,
        BRENT_XTOL,
        BRENT_RTOL,
        root::DEFAULT_MAXITER,
    )
    .map_err(|e| BracketFailure {
        step,
        cause: Some(e),
    })
}

// -- the injection and the commit ----------------------------------------------------------------

/// The bore's `source` hook: correct the mouthpiece node `p_next[0]` for the reed flow and stash
/// the post-solve quantities for [`commit`] to book.
///
/// `p_next[0]` arrives as the bore's force-free (rigid-wall) half-cell step. With the reed's affine
/// response `y^{n+1} = y_hist + c_reed p_bar` eliminated, the node balance is the scalar
/// `D p_bar = C_const + p_pref0 U_B(dp_bar)` in `p_bar = (p0^{n+1} + p0^n)/2`. Solve it, write
/// `p0^{n+1} = 2 p_bar - p0^n`, and record everything from **post-solve** values — which is what
/// makes the energy balance exact rather than merely close.
///
/// `p_old` is `p0^n`, read before the bore committed anything.
pub fn inject(
    p_next: &mut [f64],
    p_old: f64,
    p: &Params,
    s: &mut State,
) -> Result<(), BracketFailure> {
    let p_rigid = p_next[0];
    let opening = s.reed_opening(p);

    // `y_hist` is the dp-independent part of `y^{n+1}`; `yd_hist` its part of `y' = (y+ - y-)/2k`.
    let y_hist = ((p.cy_n * s.y) + (p.cy_prev * s.y_prev)) - (p.c_reed * s.p_mouth);
    let yd_hist = (y_hist - s.y_prev) / (2.0 * p.k);
    // `C_const` collects everything known: the node balance minus the reed-sweep history term.
    let c_const = (p_rigid + p_old) - ((p.p_pref0 * p.sr) * yd_hist);

    let (dp, used_fallback) = solve_dp(opening, c_const, s.dp, s.p_mouth, s.n, p)?;
    let p_bar = s.p_mouth - dp;
    p_next[0] = 2.0 * p_bar - p_old;

    let y_new = y_hist + p.c_reed * p_bar;
    let y_dot = (y_new - s.y_prev) / (2.0 * p.k);
    let u_b = bernoulli_flow(dp, opening, p.width, p.rho);
    let u_total = u_b - p.sr * y_dot;

    s.y_new = y_new;
    s.dp = dp;
    s.reed_velocity = y_dot;
    s.jet_flow = u_b;
    s.flow = u_total;
    s.fallbacks += usize::from(used_fallback);
    Ok(())
}

/// Commit the reed leapfrog and book the energy channels, from the post-solve values.
///
/// The balance `E - E0 = mouth_work - jet_loss - reed_damp_work` is exact up to
/// `k p_bar R / p_pref0` — **linear in the scalar residual** — so it both requires and verifies a
/// converged solve each step.
pub fn commit(p: &Params, s: &mut State) {
    s.y_prev = s.y;
    s.y = s.y_new;

    let k = p.k;
    s.mouth_work += (k * s.p_mouth) * s.flow;
    s.jet_loss += (k * s.dp) * s.jet_flow;
    // `k * self.Mr * self.g * self.reed_velocity ** 2` — `** 2` is libm's `pow`, and through
    // `scalar_pow` so that a release build does not fold the literal exponent (§17.2).
    s.reed_damp_work += ((k * p.mr) * p.g) * scalar_pow(s.reed_velocity, 2.0);
    s.n += 1;
}

// -- the native owning struct ----------------------------------------------------------------

/// A bore blown through a dynamic single reed.
///
/// The Rust caller's view and what `cargo test` exercises. The Python binding does **not** wrap
/// this: its bore has to be the *Python object* the caller passed, so that `reed.bore.energy()`
/// reaches the same instance. It holds a handle to a `PyBore` and calls the free functions above.
#[derive(Debug, Clone)]
pub struct ReedBore {
    p: Params,
    s: State,
    bore: Bore,
}

impl ReedBore {
    /// Build from a validated parameter set and an air column, at rest.
    pub fn new(p: Params, bore: Bore, p_mouth: f64) -> ReedBore {
        ReedBore {
            p,
            s: State::at_rest(p_mouth),
            bore,
        }
    }

    /// The parameters.
    pub fn params(&self) -> &Params {
        &self.p
    }

    /// The reed state.
    pub fn state(&self) -> &State {
        &self.s
    }

    /// The reed state, mutable — `p_mouth` is meant to be changed between steps.
    pub fn state_mut(&mut self) -> &mut State {
        &mut self.s
    }

    /// The air column.
    pub fn bore(&self) -> &Bore {
        &self.bore
    }

    /// Advance one step: the bore's leapfrog with the reed injecting at node 0, then commit the
    /// reed state and book the energy channels.
    pub fn step(&mut self) -> Result<(), BracketFailure> {
        let p_old = self.bore.p()[0];
        // Destructured so the closure can hold `&mut self.s` while `self.bore` is stepping.
        let ReedBore { p, s, bore } = self;
        let mut failure: Option<BracketFailure> = None;
        {
            let mut hook = |p_next: &mut [f64]| {
                if let Err(e) = inject(p_next, p_old, p, s) {
                    failure = Some(e);
                }
            };
            bore.step(Some(&mut hook));
        }
        if let Some(e) = failure {
            return Err(e);
        }
        commit(&self.p, &mut self.s);
        Ok(())
    }

    /// Total stored energy `E_bore + E_reed` (J) — the quantity the balance identity tracks.
    ///
    /// **Not** conserved: the mouth is active. Assert
    /// `E^n - E^0 == mouth_work - jet_loss - reed_damp_work`, not conservation.
    pub fn energy(&self) -> f64 {
        self.bore.energy() + self.s.reed_energy(&self.p)
    }

    /// Pressure at the mouthpiece node `p0` (Pa) — the natural playing signal.
    pub fn mouthpiece_pressure(&self) -> f64 {
        self.bore.p()[0]
    }

    /// The balance residual `(E^n - E^0) - (mouth_work - jet_loss - reed_damp_work)`.
    ///
    /// The exciter money test. `energy()` already carries the bore's `radiated_energy`, so a
    /// radiating bell needs no extra term — but the bore's **viscous** loss is booked nowhere, so
    /// this only closes for `sigma = 0`.
    pub fn balance_error(&self, e0: f64) -> f64 {
        (self.energy() - e0) - (self.s.mouth_work - self.s.jet_loss - self.s.reed_damp_work)
    }
}
