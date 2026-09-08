//! `SympatheticStrings`: several strings sharing **one** bridge point on a common `ModalBody`.
//!
//! Piano unisons, the sitar's and tanpura's sympathetic strings — HANDOFF §12.B. Each string ends
//! free and is tied to the shared bridge by its own linear spring `K_j`; the body feels the scalar
//! sum of the spring forces, and every string sees the same bridge displacement
//! `w_b = sum_i phi_i q_i`. That single shared node is the whole model: it is what lets a plucked
//! string ring up a silent neighbour, and it is why no single string's energy is conserved while
//! the total is.
//!
//! # Why this module exists here rather than in the binding crate
//!
//! It did not, until now. `SympatheticStrings` was ported in `docs/dev/rust-migration-plan.md`
//! §34 into `crates/physsynth-py/src/connection.rs` — the crate whose only purpose is letting
//! Python call Rust, and the crate `docs/dev/python-retirement-plan.md` §7 deletes. That port was
//! written *polymorphic over its collaborators through Python duck typing*, which was the right
//! shape for the other three bridges in that file (`StringBodyBridge` accepts eight kinds of body)
//! and is not a shape that survives having no Python. With the interpreter gone the model would go
//! with it, so the model comes here first. The retirement plan's §11 records the hole; this is the
//! first of the modules it names.
//!
//! The generality is not lost, it is *narrowed to what is used*: every construction site in the
//! repository — the suite's `make_sympathetic`, the viewer's sympathetic and Weinreich scenes —
//! passes `IdealString`s and a `ModalBody`, so that is what this takes, by value. A caller plucks
//! through [`SympatheticStrings::string_mut`] rather than by holding its own reference, which is
//! the ownership difference between the two languages and the only API change.
//!
//! # The arithmetic is the binding's, deliberately
//!
//! Expression order, the `2 k^2 / (rho h)` precompute, the end-node correction applied *after* the
//! string's own step, and `reduce::sum` for the body's driving force are all kept exactly as
//! §34's port spells them, even though nothing here has a NumPy twin to agree with any more. The
//! reason is transitional and expires with the binding: while both implementations exist, the two
//! can be run side by side on the same fixture and compared bit for bit, which is the only
//! independent check available on a port whose Python acceptance suite is being retired in the
//! same batch. `reduce::sum` reproduces NumPy's pairwise blocking, and it is the reduction that
//! reaches the next timestep through `body.step(force)`.
//!
//! # The stability guard
//!
//! Construction assembles the coupled leapfrog operator `A` column by column and refuses if
//! `k^2 lambda_max(A) >= 4`. That is the exact dense guard, not a per-part estimate: a bridge
//! spring couples the string's end mass to the body's modal masses, and the coupled spectrum can
//! exceed both parts' separately. `A = M^-1 K` is not symmetric, but it is *similar* to a
//! symmetric matrix under the diagonal mass scaling, which is what lets [`crate::eig`] be a
//! symmetric routine; that module's header carries the argument and `tests/connection.rs` asserts
//! it.

use crate::body::ModalBody;
use crate::eig;
use crate::fmt::py_exp;
use crate::reduce;
use crate::string_ideal::{self, Boundary, IdealString};

/// The slack on both stability comparisons, so that a fixture built to sit exactly at a bound is
/// not refused by its own last bit.
///
/// The *timestep* agreement is a tighter, separate number — `1e-15` absolute, inline below, which
/// is the original's `np.isclose(a, b, rtol=0, atol=1e-15)`. The two are not the same tolerance
/// and must not be unified: one asks whether two sample rates are the same number, the other
/// leaves headroom under a spectral bound.
const CFL_TOL: f64 = 1e-12;

/// Why a coupled set was refused.
///
/// Every variant is a construction-time refusal: this project's rule for an explicit scheme is
/// that an unstable configuration is an error, never a run that quietly overflows.
#[derive(Debug, Clone, PartialEq)]
pub enum ConnectionError {
    /// No strings were given.
    NoStrings,
    /// `Ks` does not have one entry per string.
    StiffnessCount(usize, usize),
    /// A bridge stiffness is negative.
    NegativeStiffness,
    /// String `j` and the body were built at different sample rates.
    TimestepMismatch(usize, f64, f64),
    /// String `j`'s right end is not free, so it cannot carry a bridge spring.
    RightEndNotFree(usize),
    /// String `j` runs at `lambda >= 1`, where the bridge spring pushes its Nyquist mode unstable.
    LambdaAtLimit(usize),
    /// The coupled operator's `k^2 lambda_max` reached 4.
    Unstable(f64),
    /// The guard's eigenvalue computation failed.
    Eigen(eig::EigError),
}

impl std::fmt::Display for ConnectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConnectionError::NoStrings => write!(f, "need at least one string."),
            ConnectionError::StiffnessCount(got, count) => write!(
                f,
                "Ks must have one stiffness per string (got {got} for {count} strings)."
            ),
            ConnectionError::NegativeStiffness => {
                write!(f, "every bridge stiffness K must be >= 0.")
            }
            ConnectionError::TimestepMismatch(j, sk, bk) => write!(
                f,
                "string {} and the body must share a timestep (got k={} vs {}); build them at \
                 the same fs.",
                j,
                py_exp(*sk, 3),
                py_exp(*bk, 3)
            ),
            ConnectionError::RightEndNotFree(j) => write!(
                f,
                "string {j}'s right end must be 'free' to attach to the bridge (build it with \
                 boundary=('fixed', 'free'))."
            ),
            ConnectionError::LambdaAtLimit(j) => write!(
                f,
                "string {j} must run at lambda < 1: its Nyquist mode is marginal at lambda = 1 \
                 and the bridge spring pushes it unstable. Rebuild at lambda < 1."
            ),
            ConnectionError::Unstable(v) => write!(
                f,
                "connection unstable: k^2 * lambda_max(A) = {v:.6} >= 4. Reduce the bridge \
                 stiffnesses, raise fs, or increase the body/string end mass."
            ),
            ConnectionError::Eigen(e) => write!(f, "the coupled stability guard failed: {e}"),
        }
    }
}

impl std::error::Error for ConnectionError {}

/// Several strings sharing one bridge point on a common [`ModalBody`].
#[derive(Debug, Clone)]
pub struct SympatheticStrings {
    strings: Vec<IdealString>,
    body: ModalBody,
    k_springs: Vec<f64>,
    k: f64,
    /// `2 k^2 / (rho_j h_j)` — the end-node correction weight, precomputed per string.
    beta_s: Vec<f64>,
    spectral_radius: f64,
    n: usize,
}

impl SympatheticStrings {
    /// Validate the set, run the exact dense stability guard, and take ownership.
    ///
    /// # Errors
    /// A [`ConnectionError`] for each of the refusals listed there, in the order they are checked:
    /// the string count, then the stiffness vector, then per string the shared timestep, the free
    /// right end and `lambda < 1`, then the coupled spectral bound.
    pub fn new(
        strings: Vec<IdealString>,
        body: ModalBody,
        ks: Vec<f64>,
    ) -> Result<SympatheticStrings, ConnectionError> {
        if strings.is_empty() {
            return Err(ConnectionError::NoStrings);
        }
        let count = strings.len();
        if ks.len() != count {
            return Err(ConnectionError::StiffnessCount(ks.len(), count));
        }
        if ks.iter().any(|&v| v < 0.0) {
            return Err(ConnectionError::NegativeStiffness);
        }

        let kb = body.params().k;
        for (j, s) in strings.iter().enumerate() {
            let p = s.params();
            if (p.k - kb).abs() > 1e-15 {
                return Err(ConnectionError::TimestepMismatch(j, p.k, kb));
            }
            if p.bc_right != Boundary::Free {
                return Err(ConnectionError::RightEndNotFree(j));
            }
            if p.lam >= 1.0 - CFL_TOL {
                return Err(ConnectionError::LambdaAtLimit(j));
            }
        }

        let k = kb;
        let beta_s: Vec<f64> = strings
            .iter()
            .map(|s| {
                let p = s.params();
                2.0 * k * k / (p.rho * p.h)
            })
            .collect();

        let mut me = SympatheticStrings {
            strings,
            body,
            k_springs: ks,
            k,
            beta_s,
            spectral_radius: 0.0,
            n: 0,
        };
        me.spectral_radius = me.max_leapfrog_eigenvalue()?;
        let bound = me.k * me.k * me.spectral_radius;
        if bound >= 4.0 - CFL_TOL {
            return Err(ConnectionError::Unstable(bound));
        }
        Ok(me)
    }

    // -- attributes ----------------------------------------------------------------------------

    /// The strings, in the order they were given.
    pub fn strings(&self) -> &[IdealString] {
        &self.strings
    }

    /// String `j`, mutably — how a caller plucks one.
    ///
    /// # Panics
    /// If `j` is out of range.
    pub fn string_mut(&mut self, j: usize) -> &mut IdealString {
        &mut self.strings[j]
    }

    /// The shared body.
    pub fn body(&self) -> &ModalBody {
        &self.body
    }

    /// The shared body, mutably — how a caller gives it an initial state.
    pub fn body_mut(&mut self) -> &mut ModalBody {
        &mut self.body
    }

    /// The bridge stiffnesses `K_j` (N/m).
    pub fn stiffnesses(&self) -> &[f64] {
        &self.k_springs
    }

    /// The shared timestep `k` (s).
    pub fn timestep(&self) -> f64 {
        self.k
    }

    /// Number of strings `J`.
    pub fn count(&self) -> usize {
        self.strings.len()
    }

    /// `lambda_max(A)` of the coupled leapfrog operator, as measured at construction.
    pub fn spectral_radius(&self) -> f64 {
        self.spectral_radius
    }

    /// Steps taken.
    pub fn n_steps(&self) -> usize {
        self.n
    }

    // -- helpers -------------------------------------------------------------------------------

    /// Shared bridge displacement `w_b = sum_i phi_i q_i`, at step `n` or `n - 1`.
    pub fn bridge_displacement(&self, prev: bool) -> f64 {
        if prev {
            let p = self.body.params();
            let q_prev = self.body.q_prev();
            let mut acc = 0.0;
            for (phi, q) in p.phi.iter().zip(q_prev.iter()) {
                acc += phi * q;
            }
            acc
        } else {
            self.body.bridge_displacement()
        }
    }

    /// Stretch `eta_j = u_{end,j} - w_b` of string `j`'s spring.
    ///
    /// # Panics
    /// If `j` is out of range.
    pub fn stretch(&self, j: usize, prev: bool) -> f64 {
        let w_b = self.bridge_displacement(prev);
        let s = &self.strings[j];
        let u = if prev { &s.u_prev } else { &s.u };
        u[u.len() - 1] - w_b
    }

    /// Per-string bridge forces `F_j = K_j eta_j^n` (explicit; N).
    pub fn connection_forces(&self) -> Vec<f64> {
        let w_b = self.bridge_displacement(false);
        self.k_springs
            .iter()
            .zip(self.strings.iter())
            .map(|(&kj, s)| kj * (s.u[s.u.len() - 1] - w_b))
            .collect()
    }

    // -- time stepping -------------------------------------------------------------------------

    /// Advance one timestep.
    ///
    /// Each explicit spring drives its own string and the shared body, which feels the scalar sum.
    /// The order matters and is the original's: the forces are read from the *current* state
    /// before anything moves, each string takes its own free step and is then corrected at its
    /// end node, and the body steps last.
    pub fn step(&mut self) {
        let forces = self.connection_forces();
        for (j, &f) in forces.iter().enumerate() {
            let s = &mut self.strings[j];
            s.step();
            let last = s.u.len() - 1;
            s.u[last] -= self.beta_s[j] * f;
        }
        // The body feels the scalar sum, and that sum reaches the next timestep — the case where
        // the reduction is part of the trajectory rather than a read-out.
        self.body.step(reduce::sum(&forces));
        self.n += 1;
    }

    // -- diagnostics ---------------------------------------------------------------------------

    /// Total discrete energy `E_body + sum_j (E_{string,j} + E_{conn,j})` (Joules).
    ///
    /// The connection term is the cross-time spring potential `1/2 K_j eta_j^n eta_j^{n-1}`, the
    /// same two-time-level form the string and body energies use — which is what makes the total
    /// conserved to machine precision for a lossless set rather than merely bounded.
    pub fn energy(&self) -> f64 {
        let mut e = self.body.energy();
        for (j, &kj) in self.k_springs.iter().enumerate() {
            e += self.strings[j].energy();
            e += 0.5 * kj * self.stretch(j, false) * self.stretch(j, true);
        }
        e
    }

    /// Energy of string `j` alone — the sympathetic-transfer trace.
    ///
    /// # Panics
    /// If `j` is out of range.
    pub fn string_energy(&self, j: usize) -> f64 {
        self.strings[j].energy()
    }

    /// Radiated pressure from the shared body.
    pub fn pressure(&self) -> f64 {
        self.body.pressure()
    }

    // -- the stability guard ---------------------------------------------------------------------

    /// Apply the stacked coupled leapfrog operator `A` to `x`.
    ///
    /// `x` is the string DOFs (node 0 of each string is clamped and does not appear) followed by
    /// the body's modal coordinates. All strings couple to the *shared* bridge displacement, which
    /// is what makes this operator more than a block diagonal.
    ///
    /// # Panics
    /// If `x` does not have `sum_j N_j + M` entries.
    pub fn apply_a(&self, x: &[f64]) -> Vec<f64> {
        let p = self.body.params();
        let modes = p.n_modes();
        assert_eq!(x.len(), self.dofs(), "x must have sum_j N_j + M entries");
        let split = x.len() - modes;
        let q = &x[split..];

        let mut w_b = 0.0;
        for (phi, qi) in p.phi.iter().zip(q.iter()) {
            w_b += phi * qi;
        }

        let mut out = vec![0.0; x.len()];
        let mut body_force = 0.0;
        let mut off = 0usize;
        for (j, &kj) in self.k_springs.iter().enumerate() {
            let s = &self.strings[j];
            let sp = s.params();
            let n = sp.n;
            let mut out_u = self.string_internal(j, &x[off..off + n]);
            let eta = x[off + n - 1] - w_b;
            let last = out_u.len() - 1;
            out_u[last] += 2.0 * kj * eta / (sp.rho * sp.h);
            out[off..off + n].copy_from_slice(&out_u);
            body_force += kj * eta;
            off += n;
        }

        for i in 0..modes {
            out[split + i] = (p.omega[i] * p.omega[i]) * q[i] - p.phi[i] * body_force / p.m[i];
        }
        out
    }

    /// Total coupled degrees of freedom, `sum_j N_j + M`.
    pub fn dofs(&self) -> usize {
        self.strings.iter().map(|s| s.params().n).sum::<usize>() + self.body.params().n_modes()
    }

    /// String `j`'s `-(c^2/h^2) * second_diff([0, x])[1:]` — the internal block of `apply_a`.
    ///
    /// The string's own stencil is called rather than re-derived, so the stability spectrum cannot
    /// disagree with the integrator that will actually run.
    fn string_internal(&self, j: usize, x: &[f64]) -> Vec<f64> {
        let s = &self.strings[j];
        let p = s.params();
        let mut u = Vec::with_capacity(x.len() + 1);
        u.push(0.0);
        u.extend_from_slice(x);
        let d2 = string_ideal::second_diff(&u, p.bc_left, p.bc_right);
        let scale = -(p.c * p.c / (p.h * p.h));
        d2[1..].iter().map(|v| scale * v).collect()
    }

    /// The diagonal mass `M` of the coupled system, in the DOF order [`apply_a`] uses.
    ///
    /// `rho h` per interior string node, `rho h / 2` at the free end — the half cell the
    /// trapezoidal node weights give it, and the reason the end node's spring term carries a
    /// factor of two — and `m_i` per body mode. Public because the symmetry it induces is a claim
    /// a test asserts rather than an internal of the guard.
    pub fn mass_diagonal(&self) -> Vec<f64> {
        let mut m = Vec::with_capacity(self.dofs());
        for s in &self.strings {
            let p = s.params();
            for i in 0..p.n {
                let cell = if i + 1 == p.n { 0.5 } else { 1.0 };
                m.push(cell * (p.rho * p.h));
            }
        }
        m.extend_from_slice(&self.body.params().m);
        m
    }

    /// The coupled operator, mass-symmetrized: `M^1/2 A M^-1/2`, row-major.
    ///
    /// Built column by column from [`apply_a`], so it is the operator the integrator uses and not
    /// a second derivation of it. Symmetric by construction of the discretization — see
    /// [`crate::eig`]'s header — and asserted so in `tests/connection.rs`.
    pub fn symmetrized_operator(&self) -> Vec<f64> {
        let n = self.dofs();
        let root: Vec<f64> = self.mass_diagonal().iter().map(|v| v.sqrt()).collect();
        let mut a = vec![0.0; n * n];
        let mut e = vec![0.0; n];
        for j in 0..n {
            e[j] = 1.0;
            let col = self.apply_a(&e);
            for (i, &v) in col.iter().enumerate() {
                a[i * n + j] = (root[i] * v) / root[j];
            }
            e[j] = 0.0;
        }
        a
    }

    /// `lambda_max(A)` — the number the stability guard compares against `4 / k^2`.
    ///
    /// # Errors
    /// [`ConnectionError::Eigen`] if the symmetric eigenvalue routine fails, which a finite
    /// operator does not cause.
    pub fn max_leapfrog_eigenvalue(&self) -> Result<f64, ConnectionError> {
        let n = self.dofs();
        let a = self.symmetrized_operator();
        eig::symmetric_max_eigenvalue(&a, n).map_err(ConnectionError::Eigen)
    }
}
