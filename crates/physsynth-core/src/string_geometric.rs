//! Geometrically exact stiff string — two polarizations plus longitudinal motion (model #10).
//!
//! Port of `physsynth/core/string_geometric.py`, the last of the four theta-scheme strings and the
//! last model in `physsynth/core/` outside `connection`, `airbox` and `analysis/`. Three fields
//! `(u, w, v)` share one implicit theta-scheme; what couples them is an **energy-conserving
//! discrete gradient** of the exact stretch potential, solved by a damped Newton iteration on
//! `3(N-1)` unknowns.
//!
//! That module's docstrings were the physics and this header documented only what the translation
//! had to decide. They were cut to a shim by deletion 5 (`f6a3a3c`) and the shim went with the last
//! of the Python (retirement plan §51), so they are carried from `f6a3a3c^`: the module docstring
//! below, under "The physics", verbatim save for its cross-references; the `energy_floor` and
//! `apply_Ainv` docstrings as the two sections after it, since neither has a native item to sit
//! on; and the derivations on the stretch terms, the discrete gradient, its Jacobian, the Newton
//! solve, the energy density, `tension`, `is_degenerate`, `nonlinear_energy`, `set_state` and
//! `allow_softening` on the native items of those names. (Three constants' docstrings had already
//! been carried at §51.)
//!
//! **The measured figures in the carried text are the Python implementation's.** They were
//! measured before the port and have not been re-measured on this one; re-measure, and save
//! the native print, before quoting any of them as a Rust result.
//!
//! # The physics
//!
//! The string family's **exact** geometry, and the model that discharges the two claims model #9
//! ([`crate::string_nonlinear::TensionModulatedString`]) explicitly refuses to make: **piano
//! phantom partials** and **out-of-plane whirling**. Both refusals have the same root cause --
//! model #9 has *one field*. Kirchhoff-Carrier collapses the tension to a **spatial scalar**
//! `T(t) = T0 + (EA/2L) I`, which is precisely what makes it (a) blind to longitudinal dynamics and
//! (b) planar. This model keeps the tension a **local field** and carries all three components:
//!
//! ```text
//! r(x, t) = ( x + v(x,t),  u(x,t),  w(x,t) )     v = longitudinal, u/w = the two polarizations
//! ```
//!
//! with `x` the coordinate in the **rest (already tensioned)** configuration.
//!
//! **The potential is exact, not a truncation.** With `q = (u_x, w_x, v_x)` the local strain and
//!
//! ```text
//! Lambda(q) = |dr/dx| = sqrt( (1 + v_x)^2 + u_x^2 + w_x^2 )     the local STRETCH RATIO
//! ```
//!
//! the stored energy density is (see the plan doc for the pre-stress derivation):
//!
//! ```text
//! V(q) = (T0/2)(u_x^2 + w_x^2) + (EA/2) v_x^2
//!        + a * [ (u_x^2 + w_x^2)/2 + 1 + v_x - Lambda ]              a = EA - T0
//!        \____ LINEAR: model #3's theta-scheme ____/  \__ NONLINEAR EXCESS: DG, theta=1/2 __/
//! ```
//!
//! The **nonlinearity coefficient is** `a = EA - T0`, so `EA = T0` is **exactly linear** -- three
//! decoupled waves, and the `u` polarization is then **bit-for-bit**
//! [`crate::string_damped::DampedStiffString`]. That is this model's regression anchor, the direct
//! analog of model #9's `EA = 0`.
//!
//! Expanding the excess for small slopes (`r^2 = u_x^2 + w_x^2`) shows **the two claims as two
//! terms**:
//!
//! ```text
//! V_nl = a * [ r^2 v_x / 2  +  r^4 / 8  + ... ]
//!             \_________/     \______/
//!              PHANTOM         model #9's KC quartic, recovered LOCALLY (u_x^4) rather
//!              PARTIALS        than spatially averaged (I^2/L). The KC<->GE distinction.
//! ```
//!
//! `r^2 v_x/2` is quadratic in transverse times linear in longitudinal: two transverse partials at
//! `f_i`, `f_j` drive the **longitudinal** field at `f_i +- f_j`. That term *is* phantom partials,
//! and it is **structurally absent** from model #9, which has no `v`.
//!
//! **Parameterisation.** `c = sqrt(T0/rho)` for **both** polarizations (the tension is isotropic by
//! construction); `c_long = sqrt(EA/rho)`. The governing ratio is `EA/T0 = (c_long/c)^2` --
//! *literally* model #9's ratio, so [`crate::string_nonlinear::string_coefficients_from_material`]
//! carries over. **But mind the identification `EA_#9 <-> (EA - T0)_#10`**: cross-model KC checks
//! show a ~0.2-0.7 % offset (`T0/EA ~ 1/150 .. 1/600`). That is the identification, not a
//! discrepancy -- do not chase it.
//!
//! **The discrete gradient -- closed form, and NO 0/0 branch.** `V` is not quadratic (unlike
//! model #9), so a genuine discrete gradient is required: `<gradbar V, q+ - q-> = V(q+) - V(q-)`
//! exactly. It has a closed form, via two exact facts:
//!
//! 1. `g = Lambda^2` is **quadratic** in `q`, so the midpoint rule is exact: `<grad g(qbar), q+ -
//!    q-> = g+ - g-`.
//! 2. The square-root difference quotient **rationalizes**: `(sqrt(g+) - sqrt(g-)) / (g+ - g-) = 1
//!    / (Lambda+ + Lambda-)` -- exactly, no limit, no Taylor.
//!
//! Chaining them collapses the whole thing to one substitution:
//!
//! > **The exact DG is the CONTINUUM gradient at the midpoint strain `qbar`, with the single
//! > replacement `Lambda(qbar) -> Lambdabar = (Lambda+ + Lambda-)/2`. mean(Lambda), NOT
//! > Lambda(mean).**
//!
//! ```text
//! gradbar V_nl = a * ( chi * ubar_x,  chi * wbar_x,  chi * (1 + vbar_x) - vbar_x )
//! chi = 1 - 1/Lambdabar = (Lambdabar - 1)/Lambdabar          the mean strain-ratio
//! ```
//!
//! `chi = 0` at rest, `> 0` stretched (hardening), `< 0` slack. **`mean(Lambda)` vs `Lambda(mean)`
//! is the single highest-risk line in this model**: the naive midpoint's DG error *shrinks with
//! amplitude* (measured 1.4e-2 -> 1.3e-5 -> 1.8e-8 as strain goes 0.1 -> 0.01 -> 0.001), so it
//! looks *nearly right* in every qualitative test -- right glide, right spectrum -- and fails
//! **only** the energy gate.
//!
//! Contrast the family: models #7/#8 need [`crate::collision`]'s `[DG]` 0/0 Taylor branch
//! (power-law potential, genuinely 0/0 in the quiet region); model #9 needs no DG at all (quadratic
//! `V` -> midpoint collapse). Model #10 sits between: a real DG, but **exactly regular** -- the
//! denominator is `Lambda+ + Lambda- ~ 2` for any physical configuration (`Lambda` is a *stretch
//! ratio*; `Lambda -> 0` means an element crushed to zero length). **Do not import `[DG]`** --
//! there is no 0/0 here to protect against.
//!
//! **The split (why `EA = T0` stays bit-identical).** Model #9's trick, verbatim. The excess needs
//! theta=1/2 to telescope, but model #3 averages its operator at theta ~ 0.28. Split, and average
//! only the **excess** at theta=1/2 -- the pieces telescope independently, so it is free:
//!
//! ```text
//! rho delta_tt u = L_u . (theta u+ + (1-2theta) u^n + theta u-)          <- model #3, VERBATIM
//!                  - 2 rho sigma0 delta_t. u + 2 rho sigma1 delta_t.(delta_xx u)
//!                  + delta_x-[ gradbar V_nl ]_u                          <- NEW, theta=1/2 (DG)
//! rho delta_tt w = (same, with kappa_w)
//! rho delta_tt v = EA delta_xx(theta v+ + (1-2theta) v^n + theta v-)
//!                  - 2 rho sigma0_long delta_t. v + 2 rho sigma1_long delta_t.(delta_xx v)
//!                  + delta_x-[ gradbar V_nl ]_v                          <- no kappa: no bending
//! ```
//!
//! `L_u = c^2 delta_xx - kappa_u^2 delta_xxxx`, `L_w = c^2 delta_xx - kappa_w^2 delta_xxxx`. **The
//! detuning lives there and nowhere else**: `c` is shared and `gradbar V_nl` sees the polarizations
//! only through `r^2 = u_x^2 + w_x^2`, so `kappa_u != kappa_w` splits **only the linear operator**
//! and the geometric nonlinearity stays exactly isotropic.
//!
//! **Energy.**
//!
//! ```text
//! E^n = [ model #3's theta-form, per field, with each field's own L ]
//!       + (1/2) ( Vnl(q^n) + Vnl(q^{n-1}) )
//! ```
//!
//! The nonlinear term is the **two-time half-average** -- model #6/#9's odd/even lesson, here
//! *derived*: by the SBP adjoint pair `<delta_x- F, y> = -<F, delta_x+ y>`, the DG identity makes
//! the nonlinear power exactly `-delta_t+[ (Vnl^n + Vnl^{n-1})/2 ]`.
//!
//! **The energy floor is 0, and that is less obvious than it looks.** Measured against the natural
//! (zero-tension) length `Lambda0 = a/EA`, the potential is exactly:
//!
//! ```text
//! V(q) = (EA/2)(Lambda - Lambda0)^2 - T0^2/(2 EA) - T0 v_x
//! ```
//!
//! The `-T0 v_x` null Lagrangian telescopes to `T0 (v_N - v_0) = 0` at fixed ends and drops from
//! the sum, but the pre-stress density `T0^2/(2 EA)` survives on every cell -- which tempts the
//! conclusion `E >= -L T0^2/(2 EA)` (a string relaxed everywhere). **That state is inadmissible**:
//! relaxing every element needs `v_x = -T0/EA` throughout, i.e. `v_N - v_0 != 0`, and both ends are
//! clamped. Imposing the constraint, `Lambda >= 1 + v_x` gives `mean(Lambda) >= 1`, and Jensen
//! makes the square term dominate the pre-stress term exactly:
//!
//! ```text
//! =>  E >= 0,   with equality iff Lambda == 1 everywhere and the string is at rest.
//! ```
//!
//! `-L T0^2/(2 EA)` is the floor for a **free** string -- relevant only if a free end is ever
//! added. See "The energy floor, in full" below. Structural only at theta=1/2 (the split's two
//! averaging rules recombine into `h sum V` only there); at theta ~ 0.28 it is an empirical bar. If
//! it is ever hit, **distinguish bug from physics via `min Lambda`**: a slack element (negative
//! [`tension`] -> buckling) is *real physics* -- measured, the scheme conserves energy to 1e-12
//! straight through local slackness -- not a scheme artifact.
//!
//! **Resolve the longitudinal field: `lam_long = c_long k / h` governs, and nothing enforces it.**
//! The scheme is unconditionally stable, so there is no CFL to violate and no error to raise -- but
//! `c_long/c = sqrt(EA/T0)` is ~22 at realistic `EA/T0`, so the familiar `lam = 0.5` silently means
//! `lam_long = 11`. Measured: `lam_long <= 2` conserves to ~1e-12 on every hard case tried; by
//! `lam_long >= 4` the Newton solve stops converging and drift explodes to 1e+3 .. 1e+5. *Stable is
//! not accurate.* Oversample (CLAUDE.md's rule for nonlinearities) and check [`Params::lam_long`].
//!
//! **Solve.** Vector Newton on `3(N-1)` unknowns `[u+; w+; v+]`. `gradbar V_nl` is **local per
//! cell**, so the Jacobian `J = A0 - (k^2/rho) delta_x- D_cell delta_x+` has `D_cell`
//! block-diagonal (3x3 per cell) and is **sparse**. A sparse LU ([`crate::sparse_lu`]), not the
//! banded Cholesky: the DG Jacobian is **not symmetric** (a discrete gradient is not the gradient
//! of anything -- `D_cell` pairs a midpoint `mbar` against a plus-level `n+`). Damped Newton +
//! Armijo; `newton_tol` is exposed and **drift is proportional to it** -- the self-certification,
//! absent a closed form for general motion. **Uniqueness is deliberately not gated** (bow /
//! model #9 precedent): *any* root conserves energy exactly, since the telescoping needs only the
//! DG identity.
//!
//! **Bending is added linearly** (Euler-Bernoulli, per polarization), *not* geometrically-exact.
//! This is honestly a "stiff geometrically-exact string", not a Timoshenko/Cosserat rod.
//!
//! Headless: the sparse LU is [`crate::sparse_lu`] and the banded Cholesky [`crate::banded`] (in
//! the Python, SciPy's). No I/O, no plotting. See `docs/dev/geometrically-exact-string-plan.md`.
//!
//! ## The energy floor, in full
//!
//! The lower bound on the energy is **zero**, attained only at rest.
//!
//! Written out because the number is not the obvious one and the derivation is worth pinning -- it
//! went wrong here in *both* directions before it went right. Measure the potential against the
//! natural (zero-tension) length `Lambda0 = a/EA`:
//!
//! ```text
//! V(q) = (EA/2)(Lambda - Lambda0)^2 - T0^2/(2 EA) - T0 v_x        (exact, any sign of a)
//! ```
//!
//! The `-T0 v_x` term is a **null Lagrangian**: it sums to `T0 (v_N - v_0) = 0` at fixed ends. The
//! pre-stress density `T0^2/(2 EA)` does *not* vanish -- it survives on every cell, which is what
//! tempts the conclusion `E >= -L T0^2/(2 EA)`, the energy of a string relaxed everywhere. **That
//! state is inadmissible here**: relaxing every element needs `v_x = -T0/EA` throughout, hence
//! `v_N - v_0 = -L T0/EA != 0` -- the string would have to shorten, and both ends are clamped.
//!
//! Imposing the constraint instead: `Lambda >= 1 + v_x` always (the root only *adds* transverse
//! length), so `mean(Lambda) >= 1 + mean(v_x) = 1`. Jensen then gives `mean((Lambda -
//! Lambda0)^2) >= (mean(Lambda) - Lambda0)^2 >= (1 - Lambda0)^2`, whose weight `(EA/2) L (1 -
//! Lambda0)^2` is *exactly* the pre-stress term. They cancel:
//!
//! ```text
//! E >= 0,    equality iff Lambda == 1 everywhere and the string is at rest.
//! ```
//!
//! So `-L T0^2/(2 EA)` is the floor for a **free** string, where the null Lagrangian no longer
//! telescopes -- worth remembering if a free end is ever added. Note this bounds the *continuum*
//! potential; the discrete `E` pairs a cross-time theta-form against a same-time nonlinear
//! half-average, so at `theta ~ 0.28` the bar is empirical, structural only at `theta = 1/2`.
//!
//! ## Why there is no `apply_ainv`
//!
//! There is no `apply_ainv` on this model -- **the one-step response is state-dependent**.
//!
//! Model #3 exposes `A^{-1}`'s action so a coupled element (e.g. [`crate::bow::BowedString`]) can
//! precompute a *constant* driving-point admittance `a = A^{-1} e_i`.
//!
//! The reason it fails here is **not** model #9's. There, `A` itself moves with the tension every
//! step. Here `A3` is genuinely **constant** -- the nonlinearity is a force on the RHS, not a term
//! in the matrix. But the string's one-step response to an injected force is still *not* `A3^{-1}`:
//! the DG force couples all three fields implicitly at `n+1`, so the true admittance is the inverse
//! of the **Newton Jacobian**, which is state-dependent (and not symmetric). A precomputed `A3^{-1}
//! e_i` would be silently wrong by exactly the nonlinearity one came here for. Coupling an exciter
//! needs a joint solve -- deliberately out of scope; see the plan doc.
//!
//! # What is new here, and it is one thing: the ordering in front of the sparse LU
//!
//! Every earlier Group D model factored **once**, at construction, and back-substituted per step.
//! This one factors **inside the Newton loop** — a fresh Jacobian per iteration per timestep — so
//! the cost of a factorization is on the hot path rather than off it, and §24's decision to leave
//! [`crate::sparse_lu`] in the natural column order stops being free.
//!
//! §24 wrote the escape clause itself: *"every Group D matrix in this project is a banded FDTD
//! operator whose natural order already has none [no fill] to speak of … if a later model makes
//! fill the constraint, an ordering goes in front of this, not inside it."* This model is that
//! model, and by a wide margin. Its unknowns are stacked **by field** — all of `u`, then all of
//! `w`, then all of `v` — while the discrete-gradient force couples the three fields *at the same
//! cell*. In that order every coupling sits `N-1` columns off the diagonal and the elimination
//! fills the whole envelope between. Measured at `N = 128` (`n = 381`):
//!
//! | | `nnz(L) + nnz(U)` | factor |
//! |---|---|---|
//! | SciPy (SuperLU + COLAMD) | 2,788 | 156 µs |
//! | this module, natural order | **33,895** | **2,068 µs** |
//! | this module, reordered by node | 2,645 | 58 µs |
//!
//! The reordering is `(u_i, w_i, v_i)` taken together — [`interleave_perm`] — and it is a **closed
//! form in `N`**, so no ordering heuristic is needed. That is §24's own finding about the beam's
//! permutation being a closed form, arriving from the other side: there it meant the reference's
//! choice could be predicted, here it means ours does not have to be searched for.
//!
//! **A permutation of the unknowns cannot move a single sum in this model**, which is what makes it
//! free rather than a trade. Every operator on the update path is block diagonal by field (`A3`,
//! `Gp3`, `Gm3`) or diagonal per cell (the discrete-gradient Jacobian), so each output entry is a
//! reduction over one block's entries and the global index order never enters one. The permutation
//! is applied *inside* [`crate::sparse_lu::SparseLu::factor_permuted`] and the residual, the state
//! and the energy stay in Python's `[u; w; v]` order throughout.
//!
//! # Where the two implementations part company, and where they do not
//!
//! * **`EA == T` is exact and structural.** The nonlinearity coefficient is `a = EA - T0`, so
//!   `EA = T` makes `a` exactly zero and [`GeometricString::step`] takes a branch with no Newton
//!   solve at all: three banded back-substitutions, model #3's expressions in model #3's order.
//!   That is what
//!   `tests/string_geometric_harness.rs::ea_equals_t_is_model_three_bit_for_bit_from_a_pluck`
//!   asserts against `DampedStiffString` (carried from the retired Python test, retirement plan
//!   §35) — an anchor between two model *classes* (§15.2), now between two Rust ones — and it
//!   holds bit-for-bit including `energy()`, because the two spare fields
//!   contribute exact zeros and adding `0.0` changes nothing.
//! * **`EA != T` diverges at the sparse LU**, per §24.2's measured verdict, and nowhere earlier:
//!   every matrix on the update path arrives from SciPy already canonical (measured at four grid
//!   sizes — `D2`, the three `L`s, the three `A`s, `A3`, `Gp`, `Gm`, `Gp3`, `Gm3` and the Jacobian),
//!   so this is the first ported theta-scheme string that needed **no** `portable.canonical` work
//!   on the Python side.
//! * **The Newton iteration count is compared, not assumed.** §19.2's branch rule: the convergence
//!   test is `max|r| <= newton_tol * max|Y_seed|`, and a max is order-independent — unlike the
//!   tension string's `brentq` bracket, which was a *sum*. The Armijo line search does branch on a
//!   reduction (`0.5 r·r`), but on the first trial step it compares a converged residual against
//!   the seed's, orders apart, so §20.3's question — how far the fed quantity sits from the
//!   threshold — answers itself.
//!
//! # Reductions
//!
//! `dot` (left to right, `portable.dot`) is used everywhere the Python original uses it: the
//! kinetic term and the potential form. `_nl_density` sums with `np.sum` — NumPy's pairwise
//! blocking — and this module declines to reproduce it, following `ops2d::guitar_area` and
//! `collision::barrier_energy` and for their reason (§14.2, §22.1: it is a claim about a library
//! internal and about a CPU). It is a read-out; nothing on the update path reads it.

use crate::banded::{self, BandedError};
use crate::fmt::py_float;
use crate::ops::{biharmonic_matrix, second_difference_matrix};
use crate::pyfloat::scalar_pow;
use crate::sparse::Csr;
use crate::sparse_lu::{SparseLu, SparseLuError, DIAG_PIVOT_THRESH};
use crate::string_stiff::dot;

// The three constants' longer docs are the Python module's, carried verbatim when it was deleted
// (retirement plan §51): they are the record of the measurements that set the numbers.

/// **Relative** tolerance on the max-norm of the Newton update residual.
///
/// Unit-free: the bar is `newton_tol * max|Y_seed|`, following model #9's normalized tension solve
/// ("so `tension_tol` is a unit-free relative bar"). An **absolute** bar here is a trap -- the
/// residual scales with the displacement, so a fixed `1e-14` is `1e-11` relative on a 1 mm string
/// and lands the drift at `2e-10`, *over* the `1e-10` gate, while looking like a tight number.
/// Measured: drift is proportional to this over five decades.
pub const NEWTON_TOL_DEFAULT: f64 = 1e-15;

/// Cap on damped-Newton iterations per step. Reaching it warns; it never silently renders.
pub const NEWTON_MAXITER_DEFAULT: i64 = 60;

/// Warn when `lam_long = c_long k / h` exceeds this. **The one guard with no CFL behind it.**
///
/// Every other explicit scheme in this project rejects `lam > 1` because the scheme is *unstable*
/// above it. This one is an **accuracy** bar on an unconditionally stable scheme, which is exactly
/// what makes it dangerous: nothing throws, nothing violates, and the model quietly returns
/// nonsense.
///
/// **There are two edges here, not one, and they are a factor of two apart.** Re-measured
/// 2026-09-03 by `scripts/sweep_geometric_lam_long.py` (since 2026-09-28
/// `crates/physsynth-core/examples/geometric_lam_long.rs`) over nine `(N, amplitude, IC)` cells --
/// plucked and mode-3, `N` 16 to 32, amplitudes to 1e-2 -- with the Newton iteration counter read
/// beside the drift rather than the drift alone:
///
/// ```text
/// lam_long <= 2    conserves ~1e-13 .. 1e-15, no step exhausts newton_maxiter
/// lam_long ~ 4     the CONVERGENCE edge: steps begin to exhaust newton_maxiter (<= 4 in 7 of 9
///                  cells on the Python implementation, 6 of 9 on Rust -- a last-bit event, so
///                  quote it as "about 4", never per cell; scientific-hurdles.md section 6)
/// lam_long  5-10   the ENERGY edge, case-dependent: drift breaks 1e-10 and runs to 1e+5
/// ```
///
/// Between the two the solve stalls on up to a fifth of its steps and the energy *still* conserves
/// to ~1e-15 -- so in that band the model's primary bug detector is blind, and blind in the
/// safe-looking direction (`docs/dev/scientific-hurdles.md` section 8 is the other place in this
/// project where a flat energy is not a stability certificate). An earlier version of this table
/// gave a single row per `lam_long` and so read as one threshold; its `= 4` was the convergence
/// edge and its `1e+3` drift was the energy edge, two cells apart.
///
/// The bar sits at 1 rather than the measured-safe 2 to mirror this project's "tune toward
/// lambda = 1" rule, and to keep 4x of margin on the *convergence* edge -- which, now that the two
/// are separated, is what the 4x is margin against. It **warns rather than rejects**:
/// `lam_long = 2` demonstrably conserves, so a hard bar would forbid working configurations -- and
/// the regime above is worth being able to *study*, just not to trust. Because
/// `c_long/c = sqrt(EA/T0) ~ 22` at realistic stiffness, the familiar transverse `lam = 0.5` lands
/// at `lam_long ~ 11`: this warning will fire on the parameters a reader of models #1-#9 would
/// reach for first. That is its whole purpose.
///
/// **Not raised at `EA == T0`** (`a == 0`), regardless of `lam_long`. There the three fields
/// decouple and the model *is* the damped stiff string three times over -- which does not warn
/// about its own `lambda` either. The exemption is load-bearing rather than tidy: the `EA = T0`
/// bit-identity anchor sits at `lam_long == 1.0` **exactly**, flush against this bar, so without it
/// a float wobble would fire a spurious warning on this model's single most important regression
/// test.
///
/// **The bar is one-sided, and mind the flip.** It reads `lam_long` because the *longitudinal*
/// field is the fast one -- but only while `EA > T0`. Below the anchor `c_long < c`, the
/// **transverse** field becomes the fast one and plain `lam` governs instead; this bar then says
/// nothing useful, and a caller who resolves `lam_long` on a softening string is under-resolving
/// the wave that actually sets the timestep (measured: `EA/T0 = 1/200` at `lam_long = 0.5` means
/// `lam = 7` and the Newton Jacobian goes singular, while the same string at `lam = 0.5` conserves
/// to 1e-13). Softening is a deliberate opt-in (see `allow_softening`), so this is documented
/// rather than branched on.
pub const LAM_LONG_WARN: f64 = 1.0;

/// Armijo's sufficient-decrease constant, and the cap on backtracking halvings.
const ARMIJO_C: f64 = 1e-4;
const ARMIJO_MAXITER: usize = 40;

/// A construction-time rejection. Every variant's `Display` is the Python original's message
/// verbatim; `tests/string_geometric_harness.rs` asserts each variant with the fragment the
/// retired Python test matched on.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamError {
    /// One of `L`, `T`, `rho`, `fs` was not positive.
    NonPositive,
    /// `EA` was not positive.
    NonPositiveEa,
    /// Fewer than two spatial segments.
    TooFewSegments,
    /// Negative transverse stiffness.
    NegativeKappa,
    /// Negative out-of-plane stiffness.
    NegativeKappaW,
    /// Negative `sigma0` or `sigma1`.
    NegativeSigma,
    /// Negative `sigma0_long` or `sigma1_long`.
    NegativeSigmaLong,
    /// `theta` outside `(0, 1]`. Carries the offending value, which the message quotes.
    BadTheta(f64),
    /// `newton_tol` was not positive.
    BadNewtonTol,
    /// `newton_maxiter` was below one.
    BadNewtonMaxiter,
    /// The boundary spec was not `"supported"`; the caller formats the message.
    BadBoundary,
    /// `EA < T` without `allow_softening`. Carries `(EA, T)`, which the message quotes.
    Softening(f64, f64),
    /// One of the three `A`s was not positive definite.
    NotFactorable(BandedError),
}

impl std::fmt::Display for ParamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamError::NonPositive => write!(f, "L, T, rho, fs must all be positive."),
            ParamError::NonPositiveEa => write!(f, "EA (axial stiffness) must be positive."),
            ParamError::TooFewSegments => {
                write!(f, "N must be >= 2 (need at least one interior node).")
            }
            ParamError::NegativeKappa => write!(f, "kappa (stiffness) must be >= 0."),
            ParamError::NegativeKappaW => write!(f, "kappa_w (stiffness) must be >= 0."),
            ParamError::NegativeSigma => write!(f, "sigma0, sigma1 (losses) must be >= 0."),
            ParamError::NegativeSigmaLong => {
                write!(f, "sigma0_long, sigma1_long (losses) must be >= 0.")
            }
            ParamError::BadTheta(t) => {
                write!(f, "theta must be in (0, 1], got {}.", py_float(*t))
            }
            ParamError::BadNewtonTol => write!(f, "newton_tol must be > 0."),
            ParamError::BadNewtonMaxiter => write!(f, "newton_maxiter must be >= 1."),
            ParamError::BadBoundary => write!(f, "boundary must be 'supported'."),
            ParamError::Softening(ea, t) => write!(
                f,
                "EA ({}) < T ({}) makes the natural (unstretched) length ratio Lambda0 = \
                 (EA - T0)/EA = {:.4} NEGATIVE, i.e. a SOFTENING string no real material can be: \
                 at rest every element is already stretched from a natural length below zero. The \
                 model stays well-posed there (energy still conserves, E >= 0 still holds, and the \
                 string cannot go slack -- tension = EA*Lambda + |EA - T0| > 0 always), so this is \
                 hyperreality, not blow-up. Pass allow_softening=True to build it -- and mind that \
                 below EA = T0 the LONGITUDINAL wave is the slow one, so resolve the transverse \
                 lam, not lam_long.",
                py_float(*ea),
                py_float(*t),
                crate::fmt::py_general((ea - t) / ea, 4),
            ),
            ParamError::NotFactorable(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ParamError {}

/// The fill-reducing reordering the Newton Jacobian is factored in: `(u_i, w_i, v_i)` per node.
///
/// `q[factored] = caller`, taking the caller's `[u; w; v]` block order to a node-major one. See
/// the module header for the measurement that makes this worth having.
pub fn interleave_perm(n_int: usize) -> Vec<usize> {
    let mut q = Vec::with_capacity(3 * n_int);
    for i in 0..n_int {
        for f in 0..3 {
            q.push(f * n_int + i);
        }
    }
    q
}

/// The validated parameter set plus every time-constant operator and factor.
#[derive(Debug, Clone, PartialEq)]
pub struct Params {
    /// Length (m).
    pub l: f64,
    /// Tension (N).
    pub t: f64,
    /// Linear density (kg/m).
    pub rho: f64,
    /// Sample rate (Hz).
    pub fs: f64,
    /// Number of spatial segments.
    pub n: usize,
    /// Axial stiffness `EA` (N) — required, and the defining parameter. `EA = T` is exactly
    /// linear (the regression anchor). The governing ratio is `EA/T = (c_long/c)^2`; real strings
    /// sit at `EA/T ~ 150-600` (see [`crate::string_nonlinear::string_coefficients_from_material`],
    /// but mind `EA_#9 <-> (EA - T0)_#10`).
    pub ea: f64,
    /// Transverse stiffness of the `u` polarization.
    pub kappa: f64,
    /// Alias of [`Params::kappa`].
    pub kappa_u: f64,
    /// Stiffness of the `w` polarization; detuning it is what lets the string whirl.
    pub kappa_w: f64,
    /// Frequency-independent transverse loss.
    pub sigma0: f64,
    /// Frequency-dependent transverse loss.
    pub sigma1: f64,
    /// Frequency-independent longitudinal loss. Omitted at construction, both longitudinal losses
    /// inherit the transverse values, so the constructor makes no silent physics claim. **Real
    /// strings damp longitudinal motion far less than transverse** — that is a setting to opt into
    /// (pass `0.0`), not a default.
    pub sigma0_long: f64,
    /// Frequency-dependent longitudinal loss.
    pub sigma1_long: f64,
    /// Time-averaging weight in `(0, 1]`.
    pub theta: f64,
    /// Relative Newton tolerance.
    pub newton_tol: f64,
    /// Newton iteration cap.
    pub newton_maxiter: usize,
    /// Whether `EA < T` was explicitly permitted.
    ///
    /// Permit `EA < T`. **Off by default -- but the line it draws is materials, not stability.**
    ///
    /// `EA < T` makes `Lambda0 = (EA - T0)/EA` negative, and `Lambda0` is *both* the minimizer of
    /// `V` *and* the natural-length ratio of an element (see the module header's energy floor for
    /// why it is the same number). A negative natural length is not a thing a real material has:
    /// the string would be pre-stretched from an unstretched state shorter than nothing.
    ///
    /// **What it is not is unstable, and an earlier version of this docstring said otherwise.** The
    /// claim was that `EA_n = EA - T0 < 0` leaves the potential unbounded below. It does not: the
    /// identity in the module header's energy floor holds for **either sign** of `a`, `EA > 0` is
    /// separately enforced, so `(EA/2)(Lambda - Lambda0)^2 >= 0` regardless -- and the same Jensen
    /// step gives `E >= 0` at `a < 0` too. A softening string cannot even go slack (`tension = EA
    /// Lambda - a = EA Lambda + |a| > 0` for every `Lambda > 0`, where a *hardening* one genuinely
    /// can). Measured, plucked, at `EA/T0` from `1/2` down to `1/200`: drift ~1e-13, `E` positive,
    /// `min(tension) ~ T0`. So the energy floor and the drift gate are **not** void with the hatch
    /// open -- both still hold, and the tests assert it.
    ///
    /// The hatch therefore reads the way the project's rule that unphysical parameter combinations
    /// are a feature asks, from the other side: the surface `(T, rho, kappa, EA)` stays mutually
    /// unconstrained, and the *unphysical* corner is opt-in rather than forbidden. **Resolution
    /// flips here**: below `EA = T0` the longitudinal wave is the slow one, so resolve `lam`, not
    /// [`Params::lam_long`] (see [`LAM_LONG_WARN`]).
    pub allow_softening: bool,
    /// Transverse wave speed `sqrt(T / rho)` (m/s).
    pub c: f64,
    /// Longitudinal wave speed `sqrt(EA / rho)` (m/s).
    pub c_long: f64,
    /// Grid spacing `L / N` (m).
    pub h: f64,
    /// Timestep `1 / fs` (s).
    pub k: f64,
    /// Transverse Courant number — reported only.
    pub lam: f64,
    /// Longitudinal Courant number — the accuracy bar, see [`LAM_LONG_WARN`].
    pub lam_long: f64,
    /// Inharmonicity `pi^2 kappa^2 / (c^2 L^2)`.
    pub b: f64,
    /// `EA / T`, the stiffness ratio.
    pub ea_over_t: f64,
    /// `a = EA - T0`, **the** nonlinearity coefficient. Exactly `0.0` is the linear code path.
    pub a: f64,
    /// Whether construction should raise the `lam_long` warning; the caller does the warning.
    pub warn_lam_long: bool,
    /// Second-difference matrix, canonical CSR.
    pub d2: Csr,
    /// The `u` polarization's wave operator `c^2 D2 - kappa_u^2 D4`.
    pub l_u: Csr,
    /// The `w` polarization's wave operator.
    pub l_w: Csr,
    /// The longitudinal operator `c_long^2 D2` — no bending.
    pub l_v: Csr,
    /// The `u` update matrix.
    pub a_u: Csr,
    /// The `w` update matrix.
    pub a_w: Csr,
    /// The longitudinal update matrix.
    pub a_v: Csr,
    /// Upper-banded Cholesky factor of `A_u`, two superdiagonals, row-major `3 x (n - 1)`.
    pub chol_u: Vec<f64>,
    /// Upper-banded Cholesky factor of `A_w`.
    pub chol_w: Vec<f64>,
    /// Upper-banded Cholesky factor of `A_v`.
    pub chol_v: Vec<f64>,
    /// `block_diag(A_u, A_w, A_v)`.
    pub a3: Csr,
    /// The SBP gradient: interior nodes to cell strains.
    pub gp: Csr,
    /// Its adjoint `-Gp^T`: cell forces back to interior nodes.
    pub gm: Csr,
    /// `block_diag(Gp, Gp, Gp)`.
    pub gp3: Csr,
    /// `block_diag(Gm, Gm, Gm)`.
    pub gm3: Csr,
    /// The Newton Jacobian's fill-reducing reordering — [`interleave_perm`].
    pub perm: Vec<usize>,
}

impl Params {
    /// Number of grid nodes, `N + 1`.
    pub fn nodes(&self) -> usize {
        self.n + 1
    }

    /// Number of interior unknowns per field, `N - 1`.
    pub fn interior(&self) -> usize {
        self.n - 1
    }

    /// Node positions — `np.linspace(0.0, L, N + 1)`, endpoint overwritten as NumPy does.
    pub fn grid(&self) -> Vec<f64> {
        let step = self.l / (self.n as f64);
        let mut x: Vec<f64> = (0..self.nodes()).map(|i| (i as f64) * step).collect();
        x[self.n] = self.l;
        x
    }

    /// Validate, derive, assemble every operator and factor the three banded matrices.
    ///
    /// The check order is the Python original's, because the suite matches on which message comes
    /// out when two arguments are wrong at once. `n` is `i64` so `N = 1` and `N = -3` take the
    /// same documented path.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        l: f64,
        t: f64,
        rho: f64,
        fs: f64,
        n: i64,
        ea: f64,
        kappa: f64,
        kappa_w: Option<f64>,
        sigma0: f64,
        sigma1: f64,
        sigma0_long: Option<f64>,
        sigma1_long: Option<f64>,
        theta: f64,
        boundary_ok: bool,
        newton_tol: f64,
        newton_maxiter: i64,
        allow_softening: bool,
    ) -> Result<Self, ParamError> {
        if l.min(t).min(rho).min(fs) <= 0.0 {
            return Err(ParamError::NonPositive);
        }
        if ea <= 0.0 {
            return Err(ParamError::NonPositiveEa);
        }
        if n < 2 {
            return Err(ParamError::TooFewSegments);
        }
        if kappa < 0.0 {
            return Err(ParamError::NegativeKappa);
        }
        if kappa_w.is_some_and(|kw| kw < 0.0) {
            return Err(ParamError::NegativeKappaW);
        }
        if sigma0 < 0.0 || sigma1 < 0.0 {
            return Err(ParamError::NegativeSigma);
        }
        if sigma0_long.is_some_and(|s| s < 0.0) || sigma1_long.is_some_and(|s| s < 0.0) {
            return Err(ParamError::NegativeSigmaLong);
        }
        if !(theta > 0.0 && theta <= 1.0) {
            return Err(ParamError::BadTheta(theta));
        }
        if newton_tol <= 0.0 {
            return Err(ParamError::BadNewtonTol);
        }
        if newton_maxiter < 1 {
            return Err(ParamError::BadNewtonMaxiter);
        }
        if !boundary_ok {
            return Err(ParamError::BadBoundary);
        }
        if ea < t && !allow_softening {
            return Err(ParamError::Softening(ea, t));
        }

        let n = n as usize;
        let kappa_u = kappa;
        let kappa_w = kappa_w.unwrap_or(kappa);
        let sigma0_long = sigma0_long.unwrap_or(sigma0);
        let sigma1_long = sigma1_long.unwrap_or(sigma1);

        let c = (t / rho).sqrt();
        let c_long = (ea / rho).sqrt();
        let h = l / (n as f64);
        let k = 1.0 / fs;
        let lam = c * k / h;
        let lam_long = c_long * k / h;
        let b = scalar_pow(std::f64::consts::PI, 2.0) * scalar_pow(kappa, 2.0)
            / (scalar_pow(c, 2.0) * scalar_pow(l, 2.0));
        let ea_over_t = ea / t;
        let a = ea - t;
        let warn_lam_long = a != 0.0 && lam_long > LAM_LONG_WARN;

        let n_int = n - 1;
        let d2 = second_difference_matrix(n, h);
        let l_u = wave_operator(&d2, c, kappa_u, n, h);
        let l_w = wave_operator(&d2, c, kappa_w, n, h);
        let l_v = d2.scaled(scalar_pow(c_long, 2.0));

        let a_u = update_matrix(&l_u, &d2, n_int, sigma0, sigma1, theta, k);
        let a_w = update_matrix(&l_w, &d2, n_int, sigma0, sigma1, theta, k);
        let a_v = update_matrix(&l_v, &d2, n_int, sigma0_long, sigma1_long, theta, k);
        let chol_u = factor_banded(&a_u)?;
        let chol_w = factor_banded(&a_w)?;
        let chol_v = factor_banded(&a_v)?;
        let a3 = Csr::block_diag(&[&a_u, &a_w, &a_v]);

        let inv_h = 1.0 / h;
        let gp = Csr::from_rows(
            n,
            n_int,
            (0..n)
                .map(|i| {
                    let mut row = Vec::with_capacity(2);
                    if i < n_int {
                        row.push((i, inv_h));
                    }
                    if i >= 1 {
                        row.push((i - 1, -inv_h));
                    }
                    row
                })
                .collect(),
        );
        let gm = gp.transpose().scaled(-1.0);
        let gp3 = Csr::block_diag(&[&gp, &gp, &gp]);
        let gm3 = Csr::block_diag(&[&gm, &gm, &gm]);

        Ok(Params {
            l,
            t,
            rho,
            fs,
            n,
            ea,
            kappa,
            kappa_u,
            kappa_w,
            sigma0,
            sigma1,
            sigma0_long,
            sigma1_long,
            theta,
            newton_tol,
            newton_maxiter: newton_maxiter as usize,
            allow_softening,
            c,
            c_long,
            h,
            k,
            lam,
            lam_long,
            b,
            ea_over_t,
            a,
            warn_lam_long,
            d2,
            l_u,
            l_w,
            l_v,
            a_u,
            a_w,
            a_v,
            chol_u,
            chol_w,
            chol_v,
            a3,
            gp,
            gm,
            gp3,
            gm3,
            perm: interleave_perm(n_int),
        })
    }
}

/// `L = c^2 D2 - kappa^2 D4` — `_wave_operator`.
///
/// The `kappa != 0.0` guard is the original's and is not an optimization: at `kappa == 0` the
/// biharmonic term is *skipped*, not subtracted as a zero matrix, which is what keeps the
/// `EA = T`, `kappa = 0` case bit-identical to the damped string built the same way.
fn wave_operator(d2: &Csr, c: f64, kappa: f64, n: usize, h: f64) -> Csr {
    let op = d2.scaled(scalar_pow(c, 2.0));
    if kappa != 0.0 {
        op.sub(&biharmonic_matrix(n, h).scaled(scalar_pow(kappa, 2.0)))
    } else {
        op
    }
}

/// `A = (1 + sigma0 k) I - theta k^2 L - sigma1 k D2` — `_update_matrix`, model #3's verbatim.
fn update_matrix(
    op: &Csr,
    d2: &Csr,
    n_int: usize,
    sigma0: f64,
    sigma1: f64,
    theta: f64,
    k: f64,
) -> Csr {
    let ident = Csr::identity(n_int).scaled(1.0 + sigma0 * k);
    let a = ident.sub(&op.scaled(theta * scalar_pow(k, 2.0)));
    if sigma1 != 0.0 {
        a.sub(&d2.scaled(sigma1 * k))
    } else {
        a
    }
}

/// The three upper bands of a symmetric pentadiagonal `A`, then LAPACK's `DPBTF2` — `_banded`.
fn factor_banded(a: &Csr) -> Result<Vec<f64>, ParamError> {
    let n = a.nrows();
    let mut ab = vec![0.0; 3 * n];
    for i in 0..n {
        ab[2 * n + i] = a.get(i, i);
        if i >= 1 {
            ab[n + i] = a.get(i - 1, i);
        }
        if i >= 2 {
            ab[i] = a.get(i - 2, i);
        }
    }
    banded::cholesky_banded_upper(ab, 2, n).map_err(ParamError::NotFactorable)
}

// -- the discrete gradient ----------------------------------------------------------------------

/// Cell strains `q = (u_x, w_x, v_x)` as three rows of `N` from full-grid fields — `_strain`.
pub fn strain(u: &[f64], w: &[f64], v: &[f64], h: f64) -> Vec<f64> {
    let n = u.len() - 1;
    let mut q = vec![0.0; 3 * n];
    for (f, field) in [u, w, v].iter().enumerate() {
        for i in 0..n {
            q[f * n + i] = (field[i + 1] - field[i]) / h;
        }
    }
    q
}

/// `Lambda = sqrt((1 + v_x)^2 + u_x^2 + w_x^2)` per cell — `_stretch_ratio`.
///
/// The squarings are the **array** path, so `x ** 2` is NumPy's ufunc shortcut `x * x` (§16.2) and
/// not `pow` — the opposite of every scalar `** 2` in [`Params::new`].
pub fn stretch_ratio(q: &[f64]) -> Vec<f64> {
    let n = q.len() / 3;
    (0..n)
        .map(|i| {
            let vx = 1.0 + q[2 * n + i];
            (vx * vx + q[i] * q[i] + q[n + i] * q[n + i]).sqrt()
        })
        .collect()
}

/// `(Lambda, Lambda-1, Lambda-(1+v_x), r^2, Lambda+1+v_x)` per cell, free of cancellation.
///
/// `_stretch_terms`. The two rearrangements' bad regions are complementary, so the `denom > 1.0`
/// branch is exact everywhere rather than a tolerance — the derivation follows. This is *not* the
/// mallet's 0/0 Taylor branch: nothing here is genuinely 0/0 in the
/// physical region, and the branch predicate is a comparison against `1.0` on a quantity that sits
/// at `~2` for every physical element.
///
/// **Why this exists (a real defect, measured).** Both `Lambda - 1` and `Lambda - (1+v_x)` are
/// `O(strain)` or `O(strain^2)` quantities assembled from `O(1)` ones, so evaluating them literally
/// destroys relative accuracy exactly where the model is used: musical strings run at strain `~1e-4
/// .. 1e-3`. Measured on the raw forms, the DG identity's relative error degraded `4e-16 -> 7e-8`
/// as strain went `0.5 -> 1e-3` -- i.e. **worse the more realistic the string**, and within one
/// digit of the energy gate.
///
/// The cure is the *same* rationalization the discrete gradient itself rests on:
///
/// ```text
/// Lambda - 1       = (v_x (2 + v_x) + r^2) / (Lambda + 1)
/// Lambda - (1+v_x) = r^2 / (Lambda + 1 + v_x)
/// ```
///
/// The first is **unconditionally safe** (`Lambda + 1 >= 1` always). The second degenerates (0/0)
/// only for an **inverted** element (`1 + v_x < 0` with `r = 0`, i.e. compressed through zero
/// length) -- and that is precisely the region where the plain subtraction has no cancellation at
/// all (there `Lambda ~ -(1+v_x)`, so the difference is `O(1)`). The two forms' bad regions are
/// **complementary**, so the branch below is exact everywhere, not a tolerance. This is *not*
/// model #7/#8's `[DG]` 0/0 Taylor branch: nothing here is genuinely 0/0 in the physical region.
pub struct StretchTerms {
    /// `Lambda`.
    pub lam: Vec<f64>,
    /// `Lambda - 1`, through `(v_x (2 + v_x) + r^2) / (Lambda + 1)`.
    pub lam_m1: Vec<f64>,
    /// `Lambda - (1 + v_x)`, through `r^2 / (Lambda + 1 + v_x)` where that is safe.
    pub d: Vec<f64>,
    /// `r^2 = u_x^2 + w_x^2`.
    pub r2: Vec<f64>,
    /// `Lambda + 1 + v_x`, the second rearrangement's denominator.
    pub denom: Vec<f64>,
}

/// `_stretch_terms`.
pub fn stretch_terms(q: &[f64]) -> StretchTerms {
    let n = q.len() / 3;
    let mut lam = Vec::with_capacity(n);
    let mut lam_m1 = Vec::with_capacity(n);
    let mut d = Vec::with_capacity(n);
    let mut r2 = Vec::with_capacity(n);
    let mut denom = Vec::with_capacity(n);
    for i in 0..n {
        let ux = q[i];
        let wx = q[n + i];
        let vx = q[2 * n + i];
        let r2_i = ux * ux + wx * wx;
        let one_plus = 1.0 + vx;
        let lam_i = (one_plus * one_plus + r2_i).sqrt();
        let lam_m1_i = (vx * (2.0 + vx) + r2_i) / (lam_i + 1.0);
        let denom_i = lam_i + 1.0 + vx;
        let safe = denom_i > 1.0;
        d.push(if safe {
            r2_i / denom_i
        } else {
            lam_i - 1.0 - vx
        });
        lam.push(lam_i);
        lam_m1.push(lam_m1_i);
        r2.push(r2_i);
        denom.push(denom_i);
    }
    StretchTerms {
        lam,
        lam_m1,
        d,
        r2,
        denom,
    }
}

/// The exact discrete gradient `gradbar V_nl` per cell, three rows of `N` — `_dg_force`.
///
/// `<gradbar V_nl, q+ - q-> = V_nl(q+) - V_nl(q-)` **exactly**. It is the continuum gradient at
/// `qbar` with the single replacement `Lambda(qbar) -> Lambdabar = mean(Lambda)`. Passing the same
/// strains twice therefore yields the plain **continuum** gradient -- which is exactly how
/// [`initial_previous`] uses it.
///
/// **`mean(Lambda)`, NOT `Lambda(mean)`.** The naive midpoint's error shrinks with amplitude, so it
/// passes every qualitative test and fails only the energy gate.
///
/// Written in the cancellation-free variables of [`stretch_terms`], using two exact rearrangements
/// of the same expression:
///
/// ```text
/// chi = (Lambdabar - 1)/Lambdabar  with  Lambdabar - 1     = mean(Lambda^m - 1)
/// [.]_v = (Lambdabar - 1 - vbar_x)/Lambdabar   with the numerator = mean(Lambda^m - 1 -
///         v_x^m)   -- both means are exact because Lambdabar and vbar_x are themselves
///         means, so the per-level stable forms carry straight through.
/// ```
pub fn dg_force(q_plus: &[f64], q_minus: &[f64], a: f64) -> Vec<f64> {
    let n = q_plus.len() / 3;
    let sp = stretch_terms(q_plus);
    let sm = stretch_terms(q_minus);
    let mut out = vec![0.0; 3 * n];
    for i in 0..n {
        let lam_bar = 0.5 * (sp.lam[i] + sm.lam[i]);
        let chi = 0.5 * (sp.lam_m1[i] + sm.lam_m1[i]) / lam_bar;
        let u_bar = 0.5 * (q_plus[i] + q_minus[i]);
        let w_bar = 0.5 * (q_plus[n + i] + q_minus[n + i]);
        out[i] = a * chi * u_bar;
        out[n + i] = a * chi * w_bar;
        out[2 * n + i] = a * (0.5 * (sp.d[i] + sm.d[i])) / lam_bar;
    }
    out
}

/// `d(gradbar V_nl)/d q+` as a `3N x 3N` matrix of diagonal blocks — `_dg_jacobian`.
///
/// Per cell this is `a [ (chi/2) I3 - (1/2) e_v e_v^T + (1/(2 Lambdabar^2)) mbar (n+)^T ]` with
/// `mbar = (ubar_x, wbar_x, 1 + vbar_x)` and `n+ = (u_x+, w_x+, 1 + v_x+)/Lambda+`.
///
/// **It is not symmetric** -- `mbar` is a midpoint quantity while `n+` is a plus-level one. A
/// discrete gradient is not the gradient of anything, which is exactly why the Newton solve uses a
/// sparse LU ([`crate::sparse_lu`]) and not the banded Cholesky the rest of this family uses.
///
/// **The `(v,v)` entry is assembled cancellation-free.** Written literally it is `chi/2 - 1/2 +
/// (1+vbar_x)(1+v_x+)/(2 Lambdabar^2 Lambda+)`: two `O(1)` terms that cancel to an `O(strain^2)`
/// remainder, so its relative error grows like `1/strain^2` -- measured against a 60-digit
/// reference, `3.9e-11` at strain `0.1`, `1.2e-6` at `1e-3` and `5.8e-5` at `1e-4`, *worse the more
/// realistic the string* (the pathology [`stretch_terms`] cures for the residual). Using
/// `Lambdabar - 1 = mean(Lambda - 1)` the three terms collapse exactly to
/// `((1+vbar_x)(1+v_x+)/Lambda+ - Lambdabar) / (2 Lambdabar^2)`, and with `d = Lambda - (1 + v_x)`
/// (the stable third output of [`stretch_terms`], `dbar = mean(d)`) the numerator is exactly `-(d+
/// (1+vbar_x) + dbar Lambda+) / Lambda+`, so:
///
/// ```text
/// J_vv = -a (d+ (1 + vbar_x) + dbar Lambda+) / (2 Lambda+ Lambdabar^2)
/// ```
///
/// which holds ~1e-15 relative at every strain. At `q+ == q-` it is `-a r^2/(2 Lambda^3)`, half the
/// continuum Hessian, as the rotating-wave cross-check asserts. Only the Newton iteration is
/// steered by this matrix (the residual defines the root), so the fix changes how fast the solve
/// converges, never where. The Python spelled the `(v,v)` entry identically.
///
/// Not being symmetric also makes this the first Group D matrix in the project that is not SPD,
/// so [`DIAG_PIVOT_THRESH`]'s justification ("every Group D matrix is SPD") no longer covers it.
/// What covers it instead is measured: the diagonal is ~8x the off-diagonal row sum at every grid
/// size and amplitude the suite builds, so the threshold is never close to firing.
pub fn dg_jacobian(q_plus: &[f64], q_minus: &[f64], a: f64) -> Csr {
    let n = q_plus.len() / 3;
    let sp = stretch_terms(q_plus);
    let sm = stretch_terms(q_minus);

    // The three per-cell vectors the block is built from, laid out field-major like `q`.
    let mut chi = vec![0.0; n];
    let mut coef = vec![0.0; n];
    let mut n_p = vec![0.0; 3 * n];
    // `m_bar[2]` below adds `1.0` to a half-sum where the Python writes `1.0 + q_bar[2]` on an
    // array; the two spellings are the same double because `q_bar[2] = 0.5 (v_x+ + v_x-)` is
    // formed first in both.
    let mut m_bar = vec![0.0; 3 * n];
    // The cancellation-free `(v,v)` entry, `-(d+ m_v + dbar Lambda+) / (2 Lambda+ Lambdabar^2)`,
    // in the Python's evaluation order: `lam_bar**2` is NumPy's `x*x` shortcut (§16.2).
    let mut vv = vec![0.0; n];
    for i in 0..n {
        let lam_bar = 0.5 * (sp.lam[i] + sm.lam[i]);
        chi[i] = 0.5 * (sp.lam_m1[i] + sm.lam_m1[i]) / lam_bar;
        coef[i] = 0.5 / (lam_bar * lam_bar);
        let d_bar = 0.5 * (sp.d[i] + sm.d[i]);
        let m_v = 1.0 + 0.5 * (q_plus[2 * n + i] + q_minus[2 * n + i]);
        vv[i] = -(sp.d[i] * m_v + d_bar * sp.lam[i]) / ((2.0 * sp.lam[i]) * (lam_bar * lam_bar));
        n_p[i] = q_plus[i] / sp.lam[i];
        n_p[n + i] = q_plus[n + i] / sp.lam[i];
        n_p[2 * n + i] = (1.0 + q_plus[2 * n + i]) / sp.lam[i];
        m_bar[i] = 0.5 * (q_plus[i] + q_minus[i]);
        m_bar[n + i] = 0.5 * (q_plus[n + i] + q_minus[n + i]);
        m_bar[2 * n + i] = 1.0 + 0.5 * (q_plus[2 * n + i] + q_minus[2 * n + i]);
    }

    let mut rows: Vec<Vec<(usize, f64)>> = Vec::with_capacity(3 * n);
    for ai in 0..3 {
        for i in 0..n {
            let mut row = Vec::with_capacity(3);
            for bi in 0..3 {
                let d = if ai == 2 && bi == 2 {
                    vv[i]
                } else {
                    let mut d = coef[i] * m_bar[ai * n + i] * n_p[bi * n + i];
                    if ai == bi {
                        d += 0.5 * chi[i];
                    }
                    d
                };
                row.push((bi * n + i, a * d));
            }
            rows.push(row);
        }
    }
    Csr::from_rows(3 * n, 3 * n, rows)
}

/// `h sum_c V_nl(q_c)` (J) — the nonlinear **excess** only. `_nl_density`.
///
/// Reduced left to right, declining `np.sum`'s pairwise blocking (see the module header).
///
/// `V_nl(q) = a [ (u_x^2 + w_x^2)/2 + 1 + v_x - Lambda ]`, and `V_nl(0) = 0` exactly (at rest
/// `Lambda = 1`). The linear part lives in the theta-form, per the split.
///
/// Evaluated through the cancellation-free rearrangement (see [`stretch_terms`]):
///
/// ```text
/// V_nl = a r^2 ((Lambda - 1) + v_x) / (2 (Lambda + 1 + v_x))
/// ```
///
/// which is exact and keeps full **relative** accuracy on a quantity that is `O(strain^3)` -- the
/// literal expression is a difference of `O(1)` terms and was the accuracy limiter on the energy
/// gate at musical amplitudes. Leading order this is `a r^2 v_x / 2`: the phantom term, visible
/// directly in the formula.
pub fn nl_density(q: &[f64], a: f64, h: f64) -> f64 {
    if a == 0.0 {
        return 0.0;
    }
    let n = q.len() / 3;
    let s = stretch_terms(q);
    let mut acc = 0.0;
    for i in 0..n {
        let dens = if s.denom[i] > 1.0 {
            s.r2[i] * (s.lam_m1[i] + q[2 * n + i]) / (2.0 * s.denom[i])
        } else {
            0.5 * s.r2[i] - s.d[i]
        };
        acc += dens;
    }
    a * h * acc
}

/// Per-cell axial tension `T(Lambda) = EA Lambda - a` (N) — the `tension` property.
///
/// A **field**, which is the whole point of model #10: model #9's Kirchhoff-Carrier tension is a
/// spatial scalar, and that collapse is exactly what makes it blind to longitudinal dynamics.
///
/// Exactly `T0` at rest (`Lambda = 1`), and **linear in `Lambda`**. Goes **negative** (compression
/// -> buckling) when `Lambda < (EA - T0)/EA = 1/(1 + eps0)`, i.e. when an element is squeezed below
/// its natural length. Unlike model #9 -- where `I >= 0` forces the tension to only ever *rise* --
/// a geometrically-exact string genuinely can go slack, and that is real physics, not a bug.
/// Cross-check it against the energy floor (the module header) before suspecting the scheme.
pub fn tension(q: &[f64], ea: f64, a: f64) -> Vec<f64> {
    stretch_ratio(q).iter().map(|&l| ea * l - a).collect()
}

// -- time stepping ------------------------------------------------------------------------------

/// `L f` on the full grid, zeros at the two clamped nodes — `_apply_full`.
pub fn apply_full(op: &Csr, f_full: &[f64]) -> Vec<f64> {
    let last = f_full.len() - 1;
    let mut out = vec![0.0; f_full.len()];
    let interior = op.matvec(&f_full[1..last]);
    out[1..last].copy_from_slice(&interior);
    out
}

/// The consistent second-order start for all three fields — `set_state`'s tail.
///
/// The three initial fields are clamped in place, and the returned triple is `y^{-1}`. The
/// nonlinear term uses the discrete gradient with `q+ == q-`, which *is* the continuum gradient at
/// `q^0`, and is skipped entirely at `a == 0` — the branch that earns the model-#3 anchor.
pub fn initial_previous(
    u0: &mut [f64],
    w0: &mut [f64],
    v0: &mut [f64],
    dots: &[Vec<f64>; 3],
    p: &Params,
) -> [Vec<f64>; 3] {
    let last = u0.len() - 1;
    for f in [&mut *u0, &mut *w0, &mut *v0] {
        f[0] = 0.0;
        f[last] = 0.0;
    }
    let mut accel = [
        apply_full(&p.l_u, u0),
        apply_full(&p.l_w, w0),
        apply_full(&p.l_v, v0),
    ];
    if p.a != 0.0 {
        let q0 = strain(u0, w0, v0, p.h);
        let force = dg_force(&q0, &q0, p.a);
        let n = p.n;
        for (i, acc) in accel.iter_mut().enumerate() {
            let contrib = p.gm.matvec(&force[i * n..(i + 1) * n]);
            for (j, c) in contrib.iter().enumerate() {
                acc[j + 1] += c / p.rho;
            }
        }
    }
    let half_k2 = 0.5 * scalar_pow(p.k, 2.0);
    let fields = [&*u0, &*w0, &*v0];
    let mut prev: [Vec<f64>; 3] = std::array::from_fn(|i| {
        (0..fields[i].len())
            .map(|j| fields[i][j] - p.k * dots[i][j] + half_k2 * accel[i][j])
            .collect()
    });
    for f in prev.iter_mut() {
        f[0] = 0.0;
        f[last] = 0.0;
    }
    prev
}

/// Model #3's right-hand side, expression for expression — `_rhs0`.
pub fn step_rhs(
    fn_: &[f64],
    fp: &[f64],
    op: &Csr,
    sigma0: f64,
    sigma1: f64,
    p: &Params,
) -> Vec<f64> {
    let last = fn_.len() - 1;
    let k2 = scalar_pow(p.k, 2.0);
    let lu = op.matvec(&fn_[1..last]);
    let lu_prev = op.matvec(&fp[1..last]);
    let a_c = (1.0 - 2.0 * p.theta) * k2;
    let b_c = p.theta * k2;
    let s0k = sigma0 * p.k;
    let mut rhs: Vec<f64> = (0..p.interior())
        .map(|i| {
            let un = fn_[i + 1];
            let up = fp[i + 1];
            (((2.0 * un + a_c * lu[i]) - up) + b_c * lu_prev[i]) + s0k * up
        })
        .collect();
    if sigma1 != 0.0 {
        let s1k = sigma1 * p.k;
        let d2_up = p.d2.matvec(&fp[1..last]);
        for (i, r) in rhs.iter_mut().enumerate() {
            *r -= s1k * d2_up[i];
        }
    }
    rhs
}

/// What one Newton solve reports back — the telemetry the Python class stores on `self`.
#[derive(Debug, Clone, PartialEq)]
pub struct NewtonReport {
    /// The three interior solutions, concatenated `[u; w; v]`.
    pub y: Vec<f64>,
    /// Iterations taken, in the original's counting: the index at which the residual test passed.
    pub iters: usize,
    /// Whether the final residual met the bar.
    pub converged: bool,
    /// `max|r|` at exit — the warning quotes it.
    pub residual: f64,
    /// The bar `newton_tol * max|Y_seed|` — the warning quotes it too.
    pub tol_abs: f64,
}

/// Damped Newton + Armijo on the coupled `3(N-1)` system — `_solve_newton`.
///
/// The Jacobian is refactored every iteration, which is what makes [`interleave_perm`] worth
/// having (module header). The convergence test is on `max|r|`, an order-independent reduction;
/// the Armijo test is on `r . r`, which is a sum, and §20.3's question is answered in the module
/// header.
///
/// Residual `F(Y) = A3 Y - rhs0 - (k^2/rho) delta_x-[ gradbar V_nl(delta_x+ Y, q^{n-1}) ]`. Note
/// the DG depends on `y^{n+1}` and `y^{n-1}` only -- never on `y^n`; that is the discrete-gradient
/// structure, and it is what telescopes.
///
/// Seeded with the **linear** (force-free) solve against the prefactored factors -- `A3` is
/// constant here, unlike model #9's, so that seed is free and is the exact answer whenever the
/// nonlinearity is weak.
///
/// **The convergence bar is RELATIVE** (`newton_tol * max|Y_seed|`), model #9's unit-free idiom.
/// This is not fussiness: the residual scales with the displacement, so an absolute bar silently
/// loosens as the string gets quieter, and it is the **energy drift's floor** -- measured drift
/// tracks it proportionally over five decades (`1e-12 -> 8e-9`, `1e-16 -> 2e-12`), because the DG
/// force is exact only *at* the root. A run that decays under damping keeps its relative bar
/// automatically; at exact rest the residual is exactly `0.0` (`gradbar V_nl(0, 0) == 0`
/// identically) and the seed is the answer.
///
/// **Uniqueness is deliberately not gated** (the [`crate::bow::BowedString`] and model #9
/// precedent): *any* root conserves energy exactly, since the telescoping needs only the DG
/// identity `<gradbar V, q+ - q-> = V(q+) - V(q-)`. Non-uniqueness would be a branch-selection
/// question, not a correctness one.
pub fn solve_newton(
    rhs: &[Vec<f64>; 3],
    u_prev: &[f64],
    w_prev: &[f64],
    v_prev: &[f64],
    p: &Params,
) -> Result<NewtonReport, SparseLuError> {
    let n_int = p.interior();
    let mut rhs3 = Vec::with_capacity(3 * n_int);
    for r in rhs.iter() {
        rhs3.extend_from_slice(r);
    }
    let q_minus = strain(u_prev, w_prev, v_prev, p.h);
    let force_pref = scalar_pow(p.k, 2.0) / p.rho;

    let residual = |y: &[f64]| -> Vec<f64> {
        let q_plus = p.gp3.matvec(y);
        let f = dg_force(&q_plus, &q_minus, p.a);
        let ay = p.a3.matvec(y);
        let gmf = p.gm3.matvec(&f);
        (0..3 * n_int)
            .map(|i| ay[i] - rhs3[i] - force_pref * gmf[i])
            .collect()
    };

    let mut y = Vec::with_capacity(3 * n_int);
    for (r, chol) in rhs.iter().zip([&p.chol_u, &p.chol_w, &p.chol_v]) {
        y.extend(
            banded::cho_solve_banded_upper(chol, 2, n_int, r)
                .expect("the factor and the right-hand side are shaped by construction"),
        );
    }
    let tol_abs = p.newton_tol * max_abs(&y);
    let mut r = residual(&y);
    let mut iters = p.newton_maxiter;
    for it in 0..p.newton_maxiter {
        if max_abs(&r) <= tol_abs {
            iters = it;
            break;
        }
        let q_plus = p.gp3.matvec(&y);
        let jac = p.a3.sub(
            &p.gm3
                .matmul(&dg_jacobian(&q_plus, &q_minus, p.a))
                .matmul(&p.gp3)
                .scaled(force_pref),
        );
        let neg_r: Vec<f64> = r.iter().map(|v| -v).collect();
        let delta = SparseLu::factor_permuted(&jac, &p.perm, DIAG_PIVOT_THRESH)?.solve(&neg_r)?;

        // Armijo backtracking on 0.5||r||^2. The DG is smooth, so full steps are the norm; this
        // only guards the far-from-root transient.
        let f0 = 0.5 * dot(&r, &r);
        let mut t = 1.0;
        for _ls in 0..ARMIJO_MAXITER {
            let y_try: Vec<f64> = (0..y.len()).map(|i| y[i] + t * delta[i]).collect();
            let r_try = residual(&y_try);
            if 0.5 * dot(&r_try, &r_try) < (1.0 - ARMIJO_C * t) * f0 {
                break;
            }
            t *= 0.5;
        }
        for (i, yi) in y.iter_mut().enumerate() {
            *yi += t * delta[i];
        }
        r = residual(&y);
    }

    let residual_max = max_abs(&r);
    Ok(NewtonReport {
        y,
        iters,
        converged: residual_max <= tol_abs,
        residual: residual_max,
        tol_abs,
    })
}

/// `max |v|`, NumPy's `np.max(np.abs(v))` — an order-independent reduction, which is what makes
/// the convergence branch safe to compare across implementations (§19.2).
fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

// -- energy -------------------------------------------------------------------------------------

/// `(1/2) ||delta_t- f^n||^2` on the interior — `_kinetic`.
pub fn kinetic(fn_: &[f64], fp: &[f64], p: &Params) -> f64 {
    let last = fn_.len() - 1;
    let dt: Vec<f64> = (1..last).map(|i| (fn_[i] - fp[i]) / p.k).collect();
    0.5 * p.h * dot(&dt, &dt)
}

/// `P(f, g) = <-L f, g> = -h (L f) . g` on interior vectors — `_P`.
pub fn potential_form(op: &Csr, f: &[f64], g: &[f64], p: &Params) -> f64 {
    -p.h * dot(&op.matvec(f), g)
}

/// Model #3's theta-weighted cross-time linear potential for one field — `_potential`.
pub fn potential(fn_: &[f64], fp: &[f64], op: &Csr, p: &Params) -> f64 {
    let last = fn_.len() - 1;
    let un = &fn_[1..last];
    let up = &fp[1..last];
    let p_nn = potential_form(op, un, un, p);
    let p_pp = potential_form(op, up, up, p);
    let p_np = potential_form(op, un, up, p);
    0.5 * p.theta * (p_nn + p_pp) + (0.5 - p.theta) * p_np
}

/// The nonlinear excess part of `E^n` — the **two-time half-average**. `nonlinear_energy`.
///
/// The **two-time half-average** `(Vnl(q^n) + Vnl(q^{n-1}))/2` -- model #6/#9's odd/even lesson,
/// here *derived* rather than certified after the fact (a single-level `Vnl(q^n)` is a 2-step
/// invariant and oscillates spuriously).
///
/// **Sign-indefinite** (the `r^2 v_x/2` term), unlike model #9's, so report `abs()` of its fraction
/// of the energy in energy tests: a nonlinearity bug *hides* at small amplitude, where the test
/// merely re-runs the linear scheme (model #6's lesson).
pub fn nonlinear_energy(
    u: &[f64],
    w: &[f64],
    v: &[f64],
    u_prev: &[f64],
    w_prev: &[f64],
    v_prev: &[f64],
    p: &Params,
) -> f64 {
    if p.a == 0.0 {
        return 0.0;
    }
    let q_n = strain(u, w, v, p.h);
    let q_p = strain(u_prev, w_prev, v_prev, p.h);
    0.5 * (nl_density(&q_n, p.a, p.h) + nl_density(&q_p, p.a, p.h))
}

/// Model #3's energy form, per field, each with its own operator — `_linear_energy`.
pub fn linear_energy(
    u: &[f64],
    w: &[f64],
    v: &[f64],
    u_prev: &[f64],
    w_prev: &[f64],
    v_prev: &[f64],
    p: &Params,
) -> f64 {
    let kin = kinetic(u, u_prev, p) + kinetic(w, w_prev, p) + kinetic(v, v_prev, p);
    let pot = potential(u, u_prev, &p.l_u, p)
        + potential(w, w_prev, &p.l_w, p)
        + potential(v, v_prev, &p.l_v, p);
    p.rho * (kin + pot)
}

/// The `v` field's kinetic + linear-potential energy alone — `longitudinal_energy`.
pub fn longitudinal_energy(v: &[f64], v_prev: &[f64], p: &Params) -> f64 {
    p.rho * (kinetic(v, v_prev, p) + potential(v, v_prev, &p.l_v, p))
}

// -- the native owning struct ---------------------------------------------------------------

/// A geometrically exact string with its own buffers — for Rust callers and for `cargo test`.
#[derive(Debug, Clone)]
pub struct GeometricString {
    /// The parameters and every constant operator.
    pub p: Params,
    /// Transverse polarization 1 on the full grid.
    pub u: Vec<f64>,
    /// Transverse polarization 2.
    pub w: Vec<f64>,
    /// Longitudinal displacement.
    pub v: Vec<f64>,
    /// `u` one step back.
    pub u_prev: Vec<f64>,
    /// `w` one step back.
    pub w_prev: Vec<f64>,
    /// `v` one step back.
    pub v_prev: Vec<f64>,
    /// Completed steps.
    pub n: usize,
    /// Whether the most recent Newton solve converged.
    pub converged: bool,
    /// Iterations taken by the most recent step.
    pub newton_iters: usize,
    /// Cumulative Newton iterations.
    pub total_newton_iters: usize,
    /// Cumulative steps whose Newton solve stalled.
    pub n_not_converged: usize,
}

impl GeometricString {
    /// A string at rest.
    pub fn new(p: Params) -> Self {
        let nodes = p.nodes();
        GeometricString {
            p,
            u: vec![0.0; nodes],
            w: vec![0.0; nodes],
            v: vec![0.0; nodes],
            u_prev: vec![0.0; nodes],
            w_prev: vec![0.0; nodes],
            v_prev: vec![0.0; nodes],
            n: 0,
            converged: true,
            newton_iters: 0,
            total_newton_iters: 0,
            n_not_converged: 0,
        }
    }

    /// Set the three displacement fields and their optional velocities — `set_state`.
    ///
    /// **Warning — a name clash with models #1-3/#9.** There, `set_state(u0, v0)`'s `v0` is the
    /// initial **velocity**. Here `v` is the **longitudinal displacement field**, so `v0` is a
    /// *displacement* and the velocities are `dots`. Porting a call from model #3 without renaming
    /// silently injects a longitudinal displacement where a transverse velocity was meant.
    ///
    /// Uses the lossless consistent second-order start `y^{-1} = y^0 - k ydot^0 + (k^2/2) ytt^0`,
    /// with `ytt^0` including the **nonlinear** force at `t = 0` (the *continuum* gradient at `q^0`
    /// -- at a single time level there is no DG) so a single eigenmode starts as a clean discrete
    /// Duffing cosine. `EA = T` skips that term -> model #3's start exactly. All three components
    /// are clamped at both ends. Under damping the start is only *consistent* (not exact), as in
    /// model #3.
    pub fn set_state(&mut self, u0: &[f64], w0: &[f64], v0: &[f64], dots: &[Vec<f64>; 3]) {
        let mut u = u0.to_vec();
        let mut w = w0.to_vec();
        let mut v = v0.to_vec();
        let mut dots = dots.clone();
        let last = u.len() - 1;
        for d in dots.iter_mut() {
            d[0] = 0.0;
            d[last] = 0.0;
        }
        let prev = initial_previous(&mut u, &mut w, &mut v, &dots, &self.p);
        self.u = u;
        self.w = w;
        self.v = v;
        let [up, wp, vp] = prev;
        self.u_prev = up;
        self.w_prev = wp;
        self.v_prev = vp;
        self.n = 0;
        self.converged = true;
        self.newton_iters = 0;
    }

    /// Advance one timestep, rolling the history — `step`.
    ///
    /// Returns the Newton report when the nonlinear branch ran, so the caller can raise the
    /// original's `RuntimeWarning` from a Python frame; `None` on the `a == 0` linear path, which
    /// is the branch that keeps the model-#3 anchor exact.
    pub fn step(&mut self) -> Result<Option<NewtonReport>, SparseLuError> {
        let p = &self.p;
        let rhs = [
            step_rhs(&self.u, &self.u_prev, &p.l_u, p.sigma0, p.sigma1, p),
            step_rhs(&self.w, &self.w_prev, &p.l_w, p.sigma0, p.sigma1, p),
            step_rhs(
                &self.v,
                &self.v_prev,
                &p.l_v,
                p.sigma0_long,
                p.sigma1_long,
                p,
            ),
        ];
        let n_int = p.interior();

        let (interiors, report) = if p.a == 0.0 {
            self.converged = true;
            self.newton_iters = 0;
            let mut y = Vec::with_capacity(3 * n_int);
            for (r, chol) in rhs.iter().zip([&p.chol_u, &p.chol_w, &p.chol_v]) {
                y.extend(
                    banded::cho_solve_banded_upper(chol, 2, n_int, r)
                        .expect("the factor and the right-hand side are shaped by construction"),
                );
            }
            (y, None)
        } else {
            let rep = solve_newton(&rhs, &self.u_prev, &self.w_prev, &self.v_prev, p)?;
            self.converged = rep.converged;
            if !rep.converged {
                self.n_not_converged += 1;
            }
            self.newton_iters = rep.iters;
            self.total_newton_iters += rep.iters;
            (rep.y.clone(), Some(rep))
        };

        let nodes = p.nodes();
        let mut rolled: [Vec<f64>; 3] = std::array::from_fn(|f| {
            let mut full = vec![0.0; nodes];
            full[1..nodes - 1].copy_from_slice(&interiors[f * n_int..(f + 1) * n_int]);
            full
        });
        std::mem::swap(&mut self.u_prev, &mut self.u);
        std::mem::swap(&mut self.w_prev, &mut self.w);
        std::mem::swap(&mut self.v_prev, &mut self.v);
        self.u = std::mem::take(&mut rolled[0]);
        self.w = std::mem::take(&mut rolled[1]);
        self.v = std::mem::take(&mut rolled[2]);
        self.n += 1;
        Ok(report)
    }

    /// Per-cell stretch ratio `Lambda^n` — `stretch_ratio`.
    pub fn stretch_ratio(&self) -> Vec<f64> {
        stretch_ratio(&strain(&self.u, &self.w, &self.v, self.p.h))
    }

    /// Per-cell axial tension — `tension`.
    pub fn tension(&self) -> Vec<f64> {
        tension(
            &strain(&self.u, &self.w, &self.v, self.p.h),
            self.p.ea,
            self.p.a,
        )
    }

    /// Discrete energy `E^n` (J) — the linear theta-form plus the nonlinear half-average.
    pub fn energy(&self) -> f64 {
        self.linear_energy() + self.nonlinear_energy()
    }

    /// The linear part alone.
    pub fn linear_energy(&self) -> f64 {
        linear_energy(
            &self.u,
            &self.w,
            &self.v,
            &self.u_prev,
            &self.w_prev,
            &self.v_prev,
            &self.p,
        )
    }

    /// The nonlinear excess alone.
    pub fn nonlinear_energy(&self) -> f64 {
        nonlinear_energy(
            &self.u,
            &self.w,
            &self.v,
            &self.u_prev,
            &self.w_prev,
            &self.v_prev,
            &self.p,
        )
    }

    /// The longitudinal field's own energy.
    pub fn longitudinal_energy(&self) -> f64 {
        longitudinal_energy(&self.v, &self.v_prev, &self.p)
    }

    /// Whether the two polarizations are exactly degenerate — `is_degenerate`.
    ///
    /// A **discrete** output (§25.2), and the comparison is an exact `==` on two constructor
    /// arguments rather than on anything computed, so there is no margin to measure: the two
    /// implementations agree by construction or not at all.
    ///
    /// **A degenerate string provably cannot whirl.** `w -> -w` is a reflection symmetry, so a
    /// planar initial condition stays *bit-exactly* planar forever; and an out-of-plane
    /// perturbation is only **marginal**, never exponential -- `dw = u(t)` is an exact solution of
    /// the out-of-plane variational equation (it is the rotation generator), so one Floquet
    /// multiplier is `+1`, and the conserved Wronskian forces the other to `+1` too. The second
    /// solution grows only *secularly* (linear in `t`), proportional to injected angular momentum.
    /// Exponential whirling (Gough 1984, threshold `eps A^2 ~ dw0^2`) needs `kappa_u != kappa_w`.
    /// There is likewise **no universal precession rate**: rotational symmetry forces `omega_w =
    /// omega_u` exactly, at any amplitude.
    pub fn is_degenerate(&self) -> bool {
        self.p.kappa_u == self.p.kappa_w
    }
}
