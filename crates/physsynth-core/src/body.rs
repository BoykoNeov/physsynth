//! Modal body / soundboard — a bank of independent damped harmonic oscillators.
//!
//! Port of `physsynth/core/body.py`, the **body** node of HANDOFF §3.2's
//! `exciter -> resonator -> body/radiation` chain. Each mode is one oscillator
//!
//! ```text
//! q_i'' + 2 sigma_i q_i' + omega_i^2 q_i = phi_i F / m_i,     omega_i = 2 pi f_i
//! ```
//!
//! stepped by the same explicit leapfrog `string_ideal` uses, with the same **cross-time**
//! potential `1/2 omega_i^2 q_i^n q_i^{n-1}` — the two-time-level form is what makes the discrete
//! energy conserved to machine precision rather than merely bounded. Stability is per mode:
//! `omega_i k < 2`, checked at construction.
//!
//! # There is no coupling between modes, and that is the whole model
//!
//! Every array here is length `M` and every kernel is elementwise. There is no matrix, no
//! neighbour stencil and no geometry — which makes this the smallest resonator in the project and
//! the one whose Rust translation is closest to a transcription. What it does have, and what the
//! membrane did not, is **clients that write its state**.
//!
//! # `accel` is not an implementation detail, whatever its Python name says
//!
//! The original stores the modal acceleration as `_accel`, taken from the *actual* second
//! difference of the last step so that the pressure read-out reflects every force — including a
//! bridge force applied from outside. Three separate modules rely on that, and none of them merely
//! reads it: `radiation::RadiatedBody`, `radiation`'s rational air load and
//! `airbox::RoomLoadedBody` all apply a rank-1 correction to `q` *after* `step()` returns and then
//! **overwrite `_accel`** from the corrected second difference. So the field is part of the
//! interface in everything but spelling, it must be settable through the binding, and it belongs
//! to the same class of buffer as `q` and `q_prev` (plan §9.3) rather than to the immutable half.
//!
//! # Operation order
//!
//! As everywhere in this crate, the kernels reproduce NumPy's *evaluation order* and not merely
//! its algebra, so the elementwise updates agree bit-for-bit rather than to a tolerance. The five
//! read-outs (`energy`, `bridge_displacement`, `bridge_velocity`, `pressure`, and the potential
//! half of the energy) go through `np.dot`/BLAS on the Python side and cannot; they are held to
//! the plan's Group A target instead.

/// A construction-time rejection. Every variant's `Display` is the Python original's message
/// verbatim, because the suite matches on the text.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamError {
    /// `freqs` was empty (or not one-dimensional, which the caller checks).
    EmptyFreqs,
    /// A modal frequency was not positive.
    NonPositiveFreq,
    /// `fs` was not positive.
    NonPositiveFs,
    /// A loss coefficient was negative.
    NegativeSigma,
    /// A modal mass was not positive.
    NonPositiveMass,
    /// `omega_i k >= 2` for at least one mode. Carries `(omega_k, mode, freq, fs)` for the worst.
    CflViolated(f64, usize, f64, f64),
}

impl std::fmt::Display for ParamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamError::EmptyFreqs => {
                write!(f, "freqs must be a 1-D array with at least one mode.")
            }
            ParamError::NonPositiveFreq => write!(f, "all modal frequencies must be positive."),
            ParamError::NonPositiveFs => write!(f, "fs must be positive."),
            ParamError::NegativeSigma => write!(f, "sigmas (loss) must all be >= 0."),
            ParamError::NonPositiveMass => write!(f, "masses must all be positive."),
            ParamError::CflViolated(ok, mode, freq, fs) => write!(
                f,
                "CFL violated: omega*k = {ok:.6} >= 2 for mode {mode} \
                 (f = {freq:.3} Hz at fs = {fs:.1}). Raise fs."
            ),
        }
    }
}

impl std::error::Error for ParamError {}

/// A hair of slack so a requested `omega*k == 2` boundary case is not spuriously rejected.
const OMEGA_K_TOL: f64 = 1e-12;

/// The validated, immutable parameter set: everything about a body that is not its state.
///
/// The five per-mode vectors are all length `M`. `sigma`, `m`, `phi` and `a` arrive already
/// broadcast — the caller owns turning a scalar into a vector, because NumPy's broadcasting rules
/// (and its error text when they fail) belong on the Python side of the boundary.
#[derive(Debug, Clone)]
pub struct Params {
    /// Modal frequencies `f_i` (Hz).
    pub freqs: Vec<f64>,
    /// Sample rate (Hz).
    pub fs: f64,
    /// Timestep `1 / fs` (s).
    pub k: f64,
    /// Per-mode loss coefficients in the `2 sigma_i q_i'` term.
    pub sigma: Vec<f64>,
    /// Modal masses `m_i` (kg).
    pub m: Vec<f64>,
    /// Driving-point mode-shape amplitudes `phi_i`.
    pub phi: Vec<f64>,
    /// Radiation weights `a_i` for the pressure read-out.
    pub a: Vec<f64>,
    /// Angular frequencies `2 pi f_i`.
    pub omega: Vec<f64>,
    /// The per-mode CFL number `omega_i k`.
    pub omega_k: Vec<f64>,
}

impl Params {
    /// Validate and derive.
    ///
    /// The checks run in the original's order so that a call with two faults reports the same one.
    /// `a` is `None` when the caller passed no `radiation`, in which case it defaults to `phi` —
    /// a copy, because the original copies and a shared buffer would make a later edit to one show
    /// up in the other.
    pub fn new(
        freqs: Vec<f64>,
        fs: f64,
        sigma: Vec<f64>,
        m: Vec<f64>,
        phi: Vec<f64>,
        a: Option<Vec<f64>>,
    ) -> Result<Params, ParamError> {
        if freqs.is_empty() {
            return Err(ParamError::EmptyFreqs);
        }
        if freqs.iter().any(|&f| f <= 0.0) {
            return Err(ParamError::NonPositiveFreq);
        }
        if fs <= 0.0 {
            return Err(ParamError::NonPositiveFs);
        }

        let n_modes = freqs.len();
        assert_eq!(
            sigma.len(),
            n_modes,
            "sigmas must be broadcast to M entries"
        );
        assert_eq!(m.len(), n_modes, "masses must be broadcast to M entries");
        assert_eq!(phi.len(), n_modes, "phi must be broadcast to M entries");

        if sigma.iter().any(|&s| s < 0.0) {
            return Err(ParamError::NegativeSigma);
        }
        if m.iter().any(|&x| x <= 0.0) {
            return Err(ParamError::NonPositiveMass);
        }

        let a = match a {
            None => phi.clone(),
            Some(v) => {
                assert_eq!(v.len(), n_modes, "radiation must be broadcast to M entries");
                v
            }
        };

        let k = 1.0 / fs;
        // `2.0 * np.pi * f`, in that order.
        let omega: Vec<f64> = freqs
            .iter()
            .map(|&f| 2.0 * std::f64::consts::PI * f)
            .collect();
        let omega_k: Vec<f64> = omega.iter().map(|&w| w * k).collect();

        // `np.any(omega_k >= 2 - tol)`, and the mode reported is `np.argmax(omega_k)` — the
        // *largest* CFL number, which need not be the first offender. Reproducing that choice
        // matters only for the message, and the message is matched on.
        if omega_k.iter().any(|&ok| ok >= 2.0 - OMEGA_K_TOL) {
            let mut worst = 0usize;
            for (i, &ok) in omega_k.iter().enumerate() {
                if ok > omega_k[worst] {
                    worst = i;
                }
            }
            return Err(ParamError::CflViolated(
                omega_k[worst],
                worst,
                freqs[worst],
                fs,
            ));
        }

        Ok(Params {
            freqs,
            fs,
            k,
            sigma,
            m,
            phi,
            a,
            omega,
            omega_k,
        })
    }

    /// Number of modes `M`.
    pub fn n_modes(&self) -> usize {
        self.freqs.len()
    }
}

// -- kernels ---------------------------------------------------------------------------------
//
// Free functions over slices, as in `string_ideal` and `membrane`: they hold no state and write
// only into what they are given, so the native struct below can keep its buffers in `Vec`s while
// the Python binding keeps the same buffers in NumPy arrays. One copy of the physics, two owners.

/// One timestep: write `q^{n+1}` into `out` and the true modal acceleration into `accel`.
///
/// `force` is the scalar bridge force `F`, entering mode `i` as the generalised force `phi_i F`.
/// `accel` is the second difference of the step that was actually taken, so it carries `force` —
/// reconstructing `q'' = -omega^2 q - 2 sigma q'` would silently drop it, which is the bug the
/// original's comment warns about.
///
/// # Panics
/// If any slice does not have `M` entries.
pub fn step_into(
    q: &[f64],
    q_prev: &[f64],
    force: f64,
    out: &mut [f64],
    accel: &mut [f64],
    p: &Params,
) {
    let n = p.n_modes();
    assert_eq!(q.len(), n, "q must have M entries");
    assert_eq!(q_prev.len(), n, "q_prev must have M entries");
    assert_eq!(out.len(), n, "out must have M entries");
    assert_eq!(accel.len(), n, "accel must have M entries");

    let k2 = p.k * p.k;
    for i in 0..n {
        // `sk = self.sigma * self.k` is a VECTOR here — sigma is per mode — so both the numerator
        // coefficient and the denominator vary along the bank.
        let sk = p.sigma[i] * p.k;
        // The original's `rhs` chain, associated exactly as NumPy associates it:
        //   ((2*q - (1-sk)*q_prev) - ((k2*omega)*omega)*q) + ((k2*phi)*force)/m
        let restoring = ((k2 * p.omega[i]) * p.omega[i]) * q[i];
        let driving = ((k2 * p.phi[i]) * force) / p.m[i];
        let rhs = ((2.0 * q[i] - (1.0 - sk) * q_prev[i]) - restoring) + driving;
        out[i] = rhs / (1.0 + sk);
        // `(q_next - 2.0 * q + q_prev) / k2`, same left-to-right grouping.
        accel[i] = ((out[i] - 2.0 * q[i]) + q_prev[i]) / k2;
    }
}

/// The consistent second-order start `q^{-1} = q^0 - k v^0 - 1/2 k^2 omega^2 q^0`.
///
/// The lossless Taylor step, matching `IdealString::set_state`, so a single lossless mode
/// oscillates as a clean discrete cosine with no first-step transient. Also returns the modal
/// acceleration at `n = 0`, which the original sets to the lossless free response `-omega^2 q^0`
/// rather than to zero — a body that has been given an initial displacement is already
/// accelerating, and `pressure()` may be read before the first `step()`.
///
/// # Panics
/// If `q0` or `v0` do not have `M` entries.
pub fn initial_state(q0: &[f64], v0: &[f64], p: &Params) -> (Vec<f64>, Vec<f64>) {
    let n = p.n_modes();
    assert_eq!(q0.len(), n, "q0 must have M entries");
    assert_eq!(v0.len(), n, "v0 must have M entries");

    let mut prev = Vec::with_capacity(n);
    let mut accel = Vec::with_capacity(n);
    for i in 0..n {
        // `q0 - k*v0 - ((((0.5*k)*k)*omega)*omega)*q0`
        let taylor = ((((0.5 * p.k) * p.k) * p.omega[i]) * p.omega[i]) * q0[i];
        prev.push((q0[i] - p.k * v0[i]) - taylor);
        // `-self.omega * self.omega * q0`: the unary minus binds to omega first.
        accel.push(((-p.omega[i]) * p.omega[i]) * q0[i]);
    }
    (prev, accel)
}

/// Discrete modal energy `E^n` (Joules) with the cross-time potential.
///
/// `E^n = sum_i m_i [ 1/2 (delta_t- q_i)^2 + 1/2 omega_i^2 q_i^n q_i^{n-1} ]`. Conserved to machine
/// precision for a lossless body; monotonically decreasing when any `sigma_i > 0`. Both reductions
/// are `np.dot` on the Python side, so this is not bit-identical to it.
///
/// # Panics
/// If `q` or `q_prev` do not have `M` entries.
pub fn energy(q: &[f64], q_prev: &[f64], p: &Params) -> f64 {
    let n = p.n_modes();
    assert_eq!(q.len(), n, "q must have M entries");
    assert_eq!(q_prev.len(), n, "q_prev must have M entries");

    // `0.5 * np.dot(self.m, dt_q * dt_q)` — the elementwise square happens before the reduction.
    let mut kin = 0.0;
    for i in 0..n {
        let dt_q = (q[i] - q_prev[i]) / p.k;
        kin += p.m[i] * (dt_q * dt_q);
    }
    // `0.5 * np.dot(self.m * self.omega * self.omega, self.q * self.q_prev)`.
    let mut pot = 0.0;
    for i in 0..n {
        pot += ((p.m[i] * p.omega[i]) * p.omega[i]) * (q[i] * q_prev[i]);
    }
    0.5 * kin + 0.5 * pot
}

/// Driving-point (bridge) displacement `w_b = sum_i phi_i q_i^n`.
///
/// # Panics
/// If `q` does not have `M` entries.
pub fn bridge_displacement(q: &[f64], p: &Params) -> f64 {
    assert_eq!(q.len(), p.n_modes(), "q must have M entries");
    dot(&p.phi, q)
}

/// Driving-point velocity `sum_i phi_i (delta_t- q_i)`, a backward difference.
///
/// # Panics
/// If `q` or `q_prev` do not have `M` entries.
pub fn bridge_velocity(q: &[f64], q_prev: &[f64], p: &Params) -> f64 {
    let n = p.n_modes();
    assert_eq!(q.len(), n, "q must have M entries");
    assert_eq!(q_prev.len(), n, "q_prev must have M entries");
    let mut acc = 0.0;
    for i in 0..n {
        acc += p.phi[i] * ((q[i] - q_prev[i]) / p.k);
    }
    acc
}

/// Radiated pressure read-out `p = sum_i a_i q_i''` — a monopole radiates as volume acceleration.
///
/// # Panics
/// If `accel` does not have `M` entries.
pub fn pressure(accel: &[f64], p: &Params) -> f64 {
    assert_eq!(accel.len(), p.n_modes(), "accel must have M entries");
    dot(&p.a, accel)
}

/// `np.dot` of two equal-length slices, accumulated in index order.
///
/// BLAS does not promise this order, so every caller of this function is in the tolerance bucket
/// rather than the bit-identical one (plan §2.1).
fn dot(x: &[f64], y: &[f64]) -> f64 {
    let mut acc = 0.0;
    for i in 0..x.len() {
        acc += x[i] * y[i];
    }
    acc
}

// -- the native owning struct ----------------------------------------------------------------

/// A modal body owning its own state.
///
/// The Rust caller's view and what `cargo test` exercises. The Python binding does **not** wrap
/// this — it holds NumPy arrays and calls the kernels above directly, because §9.3's buffer
/// contract requires Python to own anything a client can write to, and here that includes
/// [`Self::accel`].
#[derive(Debug, Clone)]
pub struct ModalBody {
    p: Params,
    q: Vec<f64>,
    q_prev: Vec<f64>,
    accel: Vec<f64>,
    n: usize,
}

impl ModalBody {
    /// Build from a validated parameter set, at rest.
    pub fn new(p: Params) -> ModalBody {
        let n_modes = p.n_modes();
        ModalBody {
            p,
            q: vec![0.0; n_modes],
            q_prev: vec![0.0; n_modes],
            accel: vec![0.0; n_modes],
            n: 0,
        }
    }

    /// The parameters.
    pub fn params(&self) -> &Params {
        &self.p
    }

    /// Current modal displacement `q^n`.
    pub fn q(&self) -> &[f64] {
        &self.q
    }

    /// Previous modal displacement `q^{n-1}`.
    pub fn q_prev(&self) -> &[f64] {
        &self.q_prev
    }

    /// Modal acceleration of the most recent step, `q''`.
    pub fn accel(&self) -> &[f64] {
        &self.accel
    }

    /// Modal acceleration of the most recent step.
    ///
    /// Mutable on purpose: a body-loading wrapper corrects `q` after the step and then rewrites
    /// this from the corrected second difference. See the module header.
    pub fn accel_mut(&mut self) -> &mut Vec<f64> {
        &mut self.accel
    }

    /// Modal displacement, mutable — same reason as [`Self::accel_mut`].
    pub fn q_mut(&mut self) -> &mut Vec<f64> {
        &mut self.q
    }

    /// Completed steps.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Set the initial modal displacement and velocity; resets the step count.
    ///
    /// # Panics
    /// If `q0` or `v0` do not have `M` entries.
    pub fn set_state(&mut self, q0: &[f64], v0: &[f64]) {
        let (prev, accel) = initial_state(q0, v0, &self.p);
        self.q.copy_from_slice(q0);
        self.q_prev = prev;
        self.accel = accel;
        self.n = 0;
    }

    /// Advance one timestep under an optional scalar bridge `force`.
    pub fn step(&mut self, force: f64) {
        let mut next = vec![0.0; self.p.n_modes()];
        step_into(
            &self.q,
            &self.q_prev,
            force,
            &mut next,
            &mut self.accel,
            &self.p,
        );
        std::mem::swap(&mut self.q_prev, &mut self.q);
        self.q = next;
        self.n += 1;
    }

    /// Discrete modal energy `E^n` (Joules).
    pub fn energy(&self) -> f64 {
        energy(&self.q, &self.q_prev, &self.p)
    }

    /// Driving-point displacement `w_b`.
    pub fn bridge_displacement(&self) -> f64 {
        bridge_displacement(&self.q, &self.p)
    }

    /// Driving-point velocity.
    pub fn bridge_velocity(&self) -> f64 {
        bridge_velocity(&self.q, &self.q_prev, &self.p)
    }

    /// Radiated pressure read-out.
    pub fn pressure(&self) -> f64 {
        pressure(&self.accel, &self.p)
    }
}
