//! The bowed string — the project's first *continuous nonlinear exciter*, and Phase 3's last model.
//!
//! A bow drawn across a [`crate::string_damped`] string at one interior node applies a friction
//! force set by the **relative** velocity between string and bow. The friction law is evaluated at
//! the *centered* velocity, so it is implicit; because the string's update is linear in the force
//! except through that single node, the whole coupling collapses to one **scalar** root problem
//!
//! ```text
//! v_rel = v_free - g Phi(v_rel),    g = k a_i / (2 rho h),    a = A^{-1} e_i
//! ```
//!
//! solved by a safeguarded Newton seeded from the previous step (*continuation*) with a scanned
//! bracket plus [`crate::root::brentq`] behind it. `physsynth/core/bow.py` is the reference and its
//! docstrings carry the physics; this module carries the arithmetic.
//!
//! # The finding this module exists to record: the residual is spelled TWICE, on purpose
//!
//! The original evaluates the friction residual in two places that look like the same expression
//! and are not:
//!
//! ```text
//! _residual:        v - v_free + g * (((force * sqrt(2a)) * v) * exp(...))
//! _bracketed_root:  v - v_free +     (((g * (force * sqrt(2a))) * v) * exp(...))
//! ```
//!
//! The first goes through `friction_smooth` and multiplies by `g` *last*; the second hoists
//! `g * (force * sqrt(2a))` into a single scalar so NumPy can apply it to the whole scan array at
//! once. Floating-point multiplication is not associative, so these are different doubles —
//! measured 2026-08-27 at the canonical rig's real `g` (0.318) over 20,000 samples per fixture, they
//! disagree in **4,158** of them at the flagship `force = 4, a = 120` and in 568-5,372 across the
//! three the suite builds. It is §16.2's finding wearing a third hat: not a ufunc-versus-scalar this
//! time, and not a compiler fold ([§17.2](crate::pyfloat)), but a *hoist a caller performed by
//! hand*. Both spellings are therefore carried here, as [`residual`] and [`scan_residual`], and
//! they must not be merged: the scan's values decide **which brackets exist**, and one sign that
//! flips is one `brentq` call that does not happen — at a slip event, a different branch.
//!
//! What §16.2 would predict and does *not* happen **on this machine**: `np.exp` on an array and
//! `math.exp` on a scalar returned bit-identical results in 20,000 of 20,000 samples over the
//! argument range this model uses (`-a v^2 + 0.5`, so `(-inf, 0.5]`). That is a claim about a
//! *runner*, in §14.2's sense, and it is one this crate's own CI runner may not satisfy: on Windows
//! all three of NumPy, CPython and Rust reach UCRT's `exp`, while on Linux NumPy uses its own SIMD
//! loop and only CPython and Rust reach glibc. The exposure if they diverge is bounded and does not
//! reach the trajectory — the scan's values decide only whether a bracket *exists*, which needs a
//! sample within an ulp of zero, and `brentq` then re-evaluates through [`residual`], which is the
//! same libm call on both sides.
//!
//! # Where the bow sits in the migration
//!
//! It is the last model of Phase 3 and the one with the fewest new mechanisms: the banded solve it
//! needs is [`crate::banded`] (Phase 3 batch 1), the string it drives is [`crate::string_damped`]
//! (batch 3), the Brent fallback is [`crate::root`] (Phase 2 batch 3) and the scan-and-bracket
//! idiom is [`crate::collision::linspace`]'s (batch 2) — which is why that helper is shared rather
//! than copied. What is genuinely new is the *hand hoist* above, and the question §19.11 left: a
//! Newton **iteration count** and a fallback **branch** are compared by nothing in the repo, so
//! `tests/test_rust_parity_bow.py` compares them step for step.
//!
//! # The admittance is built ONCE
//!
//! `a = A^{-1} e_i` is a construction-time solve, not a per-step one — the update matrix is
//! constant. That is what keeps this model out of §15.4's ~100-step Group A window: under a shared
//! solver the two implementations agree to the bit, and the only solve that differs between them
//! without the flag happens before the run starts.

use crate::collision::linspace;
use crate::root::{brentq, RootError, DEFAULT_MAXITER};

/// Why a bowed string could not be constructed. Each variant's `Display` is the exact text
/// `physsynth.core.bow.BowedString.__init__` raises, so the binding can hand it straight to
/// `ValueError`.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamError {
    /// `force < 0`.
    NegativeForce,
    /// `sharpness <= 0`.
    NonPositiveSharpness,
    /// `newton_maxiter < 1`.
    BadMaxIter,
    /// `bow_position` outside `(0, L)`; carries the message, which quotes both numbers.
    BowPosition(String),
}

impl std::fmt::Display for ParamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamError::NegativeForce => write!(f, "bow force must be >= 0."),
            ParamError::NonPositiveSharpness => write!(f, "sharpness (a) must be > 0."),
            ParamError::BadMaxIter => write!(f, "newton_maxiter must be >= 1."),
            ParamError::BowPosition(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for ParamError {}

/// Why a friction solve failed. Both map to `RuntimeError` on the Python side.
#[derive(Debug, Clone, PartialEq)]
pub enum BowError {
    /// The scanned band held no sign change — the original's loud backstop.
    NoRoot,
    /// `brentq` itself failed inside a bracket.
    Root(RootError),
}

impl std::fmt::Display for BowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BowError::NoRoot => write!(f, "bow friction residual has no root in the bracket"),
            BowError::Root(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for BowError {}

// -- the friction characteristic -----------------------------------------------------------------

/// Smooth single-hump friction `Phi(v) = force sqrt(2a) v exp(-a v^2 + 1/2)` (Newtons).
///
/// Left-to-right exactly as the original spells it: `((force * sqrt(2a)) * v) * exp(...)`.
pub fn friction_smooth(v_rel: f64, force: f64, sharpness: f64) -> f64 {
    let a = sharpness;
    force * (2.0 * a).sqrt() * v_rel * (-a * v_rel * v_rel + 0.5).exp()
}

/// `Phi'(v)` — the derivative of [`friction_smooth`] (N·s/m).
pub fn friction_smooth_deriv(v_rel: f64, force: f64, sharpness: f64) -> f64 {
    let a = sharpness;
    force * (2.0 * a).sqrt() * (-a * v_rel * v_rel + 0.5).exp() * (1.0 - 2.0 * a * v_rel * v_rel)
}

// -- parameters ------------------------------------------------------------------------------

/// A bowed string's constants. Everything the string itself owns (`rho`, `h`, the factor) has
/// already been folded into `g` and `force_pref` by the time one of these is complete.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Params {
    /// Bow surface speed (m/s).
    pub v_bow: f64,
    /// Peak friction magnitude (N).
    pub force: f64,
    /// Friction-curve sharpness `a` (s^2/m^2).
    pub sharpness: f64,
    /// Convergence tolerance on the scalar residual.
    pub newton_tol: f64,
    /// Safeguarded-Newton iteration cap.
    pub newton_maxiter: usize,
    /// Interior grid node the bow is snapped to.
    pub node: usize,
    /// Timestep `k` (s) — the string's, copied so the work integral needs no borrow.
    pub k: f64,
    /// `g = k a_i / (2 rho h)`, the scalar that makes the coupling affine.
    pub g: f64,
    /// `k^2 / (rho h)`, the rank-1 force-injection prefactor.
    pub force_pref: f64,
    /// `g * force * sqrt(2a) * e^{1/2}` — reported, never asserted.
    pub helmholtz_number: f64,
}

impl Params {
    /// Validate and snap the bow node, in the original's order of checks.
    ///
    /// `bow_position` is snapped with `int(round(...))` — Python's rounding, which takes halves to
    /// **even** — then clamped into `1 ..= N - 1`. The three admittance-derived scalars are left
    /// at zero until [`Params::with_admittance`], because solving for `a = A^{-1} e_i` needs the
    /// node this call produces.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        bow_position: f64,
        v_bow: f64,
        force: f64,
        sharpness: f64,
        newton_tol: f64,
        newton_maxiter: i64,
        string_l: f64,
        string_h: f64,
        string_n: usize,
    ) -> Result<Params, ParamError> {
        if force < 0.0 {
            return Err(ParamError::NegativeForce);
        }
        if sharpness <= 0.0 {
            return Err(ParamError::NonPositiveSharpness);
        }
        if newton_maxiter < 1 {
            return Err(ParamError::BadMaxIter);
        }
        if !(bow_position > 0.0 && bow_position < string_l) {
            return Err(ParamError::BowPosition(format!(
                "bow_position must satisfy 0 < x < L (L={}), got {}.",
                crate::fmt::py_float(string_l),
                crate::fmt::py_float(bow_position)
            )));
        }
        let snapped = (bow_position / string_h).round_ties_even() as i64;
        let node = snapped.max(1).min(string_n as i64 - 1) as usize;
        Ok(Params {
            v_bow,
            force,
            sharpness,
            newton_tol,
            newton_maxiter: newton_maxiter as usize,
            node,
            k: 0.0,
            g: 0.0,
            force_pref: 0.0,
            helmholtz_number: 0.0,
        })
    }

    /// Fill in the three derived scalars once the caller has the admittance in hand.
    pub fn with_admittance(mut self, k: f64, rho: f64, h: f64, a_i: f64) -> Self {
        self.k = k;
        self.g = k * a_i / (2.0 * rho * h);
        self.force_pref = k * k / (rho * h);
        self.helmholtz_number = self.g * self.force * (2.0 * self.sharpness).sqrt() * 0.5_f64.exp();
        self
    }
}

// -- the scalar friction solve -----------------------------------------------------------------

/// `r(v) = v - v_free + g Phi(v)`, the spelling **Newton and `brentq` both use**.
///
/// `g` multiplies the assembled friction *last*. Not interchangeable with [`scan_residual`] — see
/// the module header.
pub fn residual(v_rel: f64, v_free: f64, p: &Params) -> f64 {
    v_rel - v_free + p.g * friction_smooth(v_rel, p.force, p.sharpness)
}

/// The same residual as the **bracket scan** spells it, with `g * force * sqrt(2a)` hoisted into
/// one scalar ahead of the array multiply.
///
/// Deliberately a second function rather than a call to [`residual`]. Merging them would move the
/// scan's values by a last bit, which is a different set of sign changes and, at a slip event, a
/// different branch. Measured disagreement at the canonical rig: 4,158 samples in 20,000.
pub fn scan_residual(v_rel: f64, v_free: f64, p: &Params) -> f64 {
    let a = p.sharpness;
    v_rel - v_free + p.g * (p.force * (2.0 * a).sqrt()) * v_rel * (-a * v_rel * v_rel + 0.5).exp()
}

/// Every root of `r` lies within `g |Phi|_max` of `v_free`; scan that band (plus the hump's width)
/// for sign changes, `brentq` each bracket, and return the root nearest `seed`.
///
/// `seed` is the **pre-step** `v_rel`, which is also what Newton was seeded from — the original
/// reads `self.v_rel` here rather than the iterate Newton walked to, and that choice is what keeps
/// the branch pick physical at a slip.
pub fn bracketed_root(v_free: f64, seed: f64, p: &Params) -> Result<f64, BowError> {
    let a = p.sharpness;
    let span = p.g * p.force + 6.0 / (2.0 * a).sqrt();
    let vs = linspace(v_free - span, v_free + span, 512);
    let rs: Vec<f64> = vs.iter().map(|&v| scan_residual(v, v_free, p)).collect();

    let mut best: Option<f64> = None;
    let mut best_dist = f64::INFINITY;
    for j in 0..(vs.len() - 1) {
        if rs[j] * rs[j + 1] < 0.0 {
            let root = brentq(
                |v| residual(v, v_free, p),
                vs[j],
                vs[j + 1],
                1e-15,
                8.9e-16,
                DEFAULT_MAXITER,
            )
            .map_err(BowError::Root)?;
            // `np.argmin` keeps the FIRST minimum, so a later root wins only on a strict
            // improvement.
            let d = (root - seed).abs();
            if d < best_dist {
                best_dist = d;
                best = Some(root);
            }
        }
    }
    best.ok_or(BowError::NoRoot)
}

/// The solve's answer: the relative velocity, whether the bracket was needed, and how much work
/// the Newton phase did.
///
/// `newton_evals` is **not** part of the Python original's interface. It is here because §19.11
/// asked for it: an iteration count is a control-flow decision a last bit can reach, and nothing in
/// the repo compared one until `tests/test_rust_parity_bow.py` did.
///
/// It counts **residual evaluations inside the Newton phase**, seed included, and deliberately not
/// accepted steps. Two reasons, both about being comparable: it is what the Python side can count
/// without being rewritten (patch `_residual`, count calls, mute the bracket), and it separates a
/// *rejected* Newton attempt from a converged one, which an accepted-step count folds together.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrictionSolution {
    /// The root of `r`.
    pub v_rel: f64,
    /// Whether the scanned bracket was used.
    pub used_fallback: bool,
    /// Residual evaluations in the Newton phase, the seed's included. The bracket's are not here.
    pub newton_evals: usize,
}

/// Solve `r(v) = 0` — safeguarded Newton from `seed`, scanned bracket behind it.
pub fn solve_v_rel(v_free: f64, seed: f64, p: &Params) -> Result<FrictionSolution, BowError> {
    let mut v = seed;
    let mut r = residual(v, v_free, p);
    let mut evals = 1usize;
    for _ in 0..p.newton_maxiter {
        if r.abs() <= p.newton_tol {
            return Ok(FrictionSolution {
                v_rel: v,
                used_fallback: false,
                newton_evals: evals,
            });
        }
        let rp = 1.0 + p.g * friction_smooth_deriv(v, p.force, p.sharpness);
        if rp.abs() < 1e-30 {
            break; // flat spot -> hand off to the robust bracket
        }
        let v_new = v - r / rp;
        let r_new = residual(v_new, v_free, p);
        evals += 1;
        // `not (|r_new| < |r|)` in the original — a negation so that a NaN residual, which compares
        // false against everything, breaks out to the bracket rather than being accepted as
        // progress. Kept in that spelling, which is also why clippy's `>=` rewrite is refused.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(r_new.abs() < r.abs()) {
            break;
        }
        v = v_new;
        r = r_new;
    }
    if r.abs() <= p.newton_tol {
        return Ok(FrictionSolution {
            v_rel: v,
            used_fallback: false,
            newton_evals: evals,
        });
    }
    Ok(FrictionSolution {
        v_rel: bracketed_root(v_free, seed, p)?,
        used_fallback: true,
        newton_evals: evals,
    })
}

// -- the step --------------------------------------------------------------------------------

/// The bow's own state — the string holds the field, this holds the telemetry.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct State {
    /// Post-correction relative velocity of the last step.
    pub v_rel: f64,
    /// Friction force applied on the last step (N).
    pub bow_force: f64,
    /// Power delivered to the string on the last step (W).
    pub bow_power: f64,
    /// Accumulated bow work (J) — the energy balance's second curve.
    pub bow_work: f64,
    /// How many steps needed the bracketed fallback.
    pub fallbacks: usize,
    /// Completed steps.
    pub n: usize,
}

/// The force-free relative velocity: centered `delta_t.` at the bow node, minus the bow speed.
///
/// `u_i` is the bow node's displacement **after** the string's force-free advance; `u_prev_i` is
/// the same node **before** it, i.e. `u^{n-1}`.
pub fn v_free(u_i: f64, u_prev_i: f64, k: f64, v_bow: f64) -> f64 {
    (u_i - u_prev_i) / (2.0 * k) - v_bow
}

/// Solve the friction, apply the exact rank-1 correction to `u_full`, and commit the telemetry.
///
/// The correction is `u += (force_pref * f_B) * a_full`, with the scalar formed once, which is what
/// NumPy does with `self._force_pref * f_B * self._a_full`. It runs over the **whole** grid,
/// including the two clamped ends where `a_full` is zero, and is never short-circuited on
/// `f_B == 0` — `test_zero_force_is_bit_identical_to_bare_string` is a cross-class bit-identity
/// anchor (§15.2) and a fast path taken on only one side would empty it.
pub fn apply(
    v_free_val: f64,
    u_full: &mut [f64],
    a_full: &[f64],
    s: &mut State,
    p: &Params,
) -> Result<FrictionSolution, BowError> {
    let sol = solve_v_rel(v_free_val, s.v_rel, p)?;
    let f_b = -friction_smooth(sol.v_rel, p.force, p.sharpness);

    let pref = p.force_pref * f_b;
    for (u, &a) in u_full.iter_mut().zip(a_full.iter()) {
        *u += pref * a;
    }

    // The TRUE post-correction velocity, so the energy balance is exact whatever the residual was.
    let v_true = v_free_val + p.g * f_b;
    s.v_rel = v_true;
    s.bow_force = f_b;
    s.bow_power = f_b * (v_true + p.v_bow);
    s.bow_work += p.k * s.bow_power;
    s.fallbacks += usize::from(sol.used_fallback);
    s.n += 1;
    Ok(sol)
}

// -- the native owning struct ------------------------------------------------------------------

/// A bowed string with its own buffers — for Rust callers and for `cargo test`.
#[derive(Debug, Clone)]
pub struct BowedString {
    /// The resonator.
    pub string: crate::string_damped::DampedStiffString,
    /// The bow's constants.
    pub p: Params,
    /// The bow's telemetry.
    pub s: State,
    /// `a = A^{-1} e_i` embedded on the full grid, zero at the clamped ends.
    pub a_full: Vec<f64>,
    /// Bow contact point in metres (the snapped node's coordinate).
    pub x_bow: f64,
    /// Fractional bow position `x_bow / L` — bookkeeping, not physics.
    pub beta: f64,
}

impl BowedString {
    /// Build a bow on `string`, solving the driving-point admittance once.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        string: crate::string_damped::DampedStiffString,
        bow_position: f64,
        v_bow: f64,
        force: f64,
        sharpness: f64,
        newton_tol: f64,
        newton_maxiter: i64,
    ) -> Result<Self, ParamError> {
        let (p, a_full, x_bow, beta) = {
            let sp = &string.p;
            let p = Params::new(
                bow_position,
                v_bow,
                force,
                sharpness,
                newton_tol,
                newton_maxiter,
                sp.l,
                sp.h,
                sp.n,
            )?;
            let (a_full, a_i) = admittance(sp, p.node);
            let p = p.with_admittance(sp.k, sp.rho, sp.h, a_i);
            let x_bow = sp.grid()[p.node];
            (p, a_full, x_bow, x_bow / sp.l)
        };
        Ok(BowedString {
            string,
            p,
            s: State::default(),
            a_full,
            x_bow,
            beta,
        })
    }

    /// Advance one step: force-free string advance, scalar friction solve, rank-1 correction.
    pub fn step(&mut self) -> Result<FrictionSolution, BowError> {
        let i = self.p.node;
        let u_prev_i = self.string.u_prev[i];
        self.string.step();
        let vf = v_free(self.string.u[i], u_prev_i, self.p.k, self.p.v_bow);
        apply(vf, &mut self.string.u, &self.a_full, &mut self.s, &self.p)
    }

    /// The string's discrete energy — the bow stores none.
    pub fn energy(&self) -> f64 {
        self.string.energy()
    }

    /// String transverse velocity at the bow node for the last step.
    pub fn bow_velocity(&self) -> f64 {
        self.s.v_rel + self.p.v_bow
    }
}

/// The one-step driving-point admittance `a = A^{-1} e_node`, embedded on the full grid.
///
/// Returns `(a_full, a_node)`. Shared by the native struct and the binding because it is the whole
/// of what construction does with the string, and doing it twice is how the two would drift.
pub fn admittance(sp: &crate::string_damped::Params, node: usize) -> (Vec<f64>, f64) {
    let mut e_local = vec![0.0; sp.interior()];
    e_local[node - 1] = 1.0;
    let a_vec = crate::string_damped::apply_ainv(&e_local, sp);
    let a_i = a_vec[node - 1];
    let nodes = sp.nodes();
    let mut a_full = vec![0.0; nodes];
    a_full[1..nodes - 1].copy_from_slice(&a_vec);
    (a_full, a_i)
}
