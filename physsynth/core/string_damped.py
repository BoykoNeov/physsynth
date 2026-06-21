"""Damped stiff string — implicit theta-scheme FDTD with frequency-dependent loss (model #3).

Implements HANDOFF section 5 model #3 (see ``docs/dev/damped-string-plan.md``). PDE:

    u_tt = c^2 u_xx - kappa^2 u_xxxx - 2 sigma0 u_t + 2 sigma1 u_txx

This is model #2 (the stiff string) plus a **frequency-dependent** loss term ``+2 sigma1 u_txx``.
In the modal domain ``delta_xx -> -p^2``, so the new term damps a mode at a rate proportional to its
wavenumber squared: a mode's **energy** decays as ``exp(-2 sigma_eff t)`` with

    sigma_eff(m) = sigma0 + sigma1 * (wavenumber)^2.

``sigma0`` (model #2's frequency-independent ``sigma``, renamed here) damps every mode equally;
``sigma1`` makes high partials die faster -- the physically-correct ordering real strings have, and
the cure for model #2's *backwards* high-frequency under-damping.

**Scheme.** The conservative operator ``L = c^2 delta_xx - kappa^2 delta_xxxx`` is
theta-time-averaged (as model #2); both loss terms use a centered time difference ``delta_t.``.
Rearranged, each step is one banded SPD solve with the constant matrix

    A = (1 + sigma0 k) I - theta k^2 L - sigma1 k D2     (pentadiagonal, SPD, factored once)

where ``D2`` is the second-difference matrix. Adding ``sigma1`` costs exactly one extra block in
``A`` (and one RHS term); ``cholesky_banded`` carries over unchanged. ``sigma1 = 0`` skips both
terms -> bit-for-bit :class:`~physsynth.core.string_stiff.StiffString`. Unconditionally stable for
``theta >= 1/4`` (so ``lambda = c k / h > 1`` is admissible, like model #2).

**Boundary: simply supported** (``u = 0``, ``u_xx = 0``); the biharmonic block is ``(delta_xx)^2``
so ``sin(m pi x / L)`` stays an exact discrete eigenvector and the modal / decay harness carries on.

**Energy is the model #2 energy, unchanged.** The loss terms never enter the stored mechanical
energy ``E^n`` -- only its rate of change. Both discrete losses are dissipative by SBP
(``-2 sigma0 ||delta_t. u||^2 - 2 sigma1 ||delta_x+ delta_t. u||^2 <= 0``), so ``energy()`` is
reused verbatim and passivity (monotone decrease) is automatic:

    E^n = rho [ 1/2 ||delta_t- u^n||^2
                + (theta/2)(P(u^n,u^n) + P(u^{n-1},u^{n-1})) + (1/2 - theta) P(u^n,u^{n-1}) ]

with ``P(f,g) = <-L f, g> = c^2 <delta_x+ f, delta_x+ g> + kappa^2 <delta_xx f, delta_xx g> >= 0``,
evaluated through the *same* matrix ``L`` used in the update.

Headless: NumPy + SciPy (banded Cholesky). No I/O, no plotting.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.linalg import cho_solve_banded, cholesky_banded

from .operators import biharmonic_matrix, second_difference_matrix
from .string_stiff import THETA_DEFAULT

Boundary = Literal["supported"]


class DampedStiffString:
    """A discretized damped stiff string (implicit theta-scheme, frequency-dependent loss).

    Parameters
    ----------
    L, T, rho : float
        Length (m), tension (N), linear density (kg/m). Wave speed ``c = sqrt(T/rho)``.
    fs : float
        Sample rate (Hz); timestep ``k = 1/fs``. No CFL limit (unconditionally stable for
        ``theta >= 1/4``).
    N : int
        Number of spatial segments; the grid has ``N + 1`` nodes, spacing ``h = L/N``. The two end
        nodes are clamped (``u = 0``); the ``N - 1`` interior nodes are the unknowns.
    kappa : float
        Stiffness coefficient ``kappa = sqrt(E I / rho)`` (m^2/s). ``0`` -> non-stiff (flexible)
        damped string. Larger ``kappa`` -> more inharmonicity ``B = pi^2 kappa^2 / (c^2 L^2)``.
    sigma0 : float
        **Frequency-independent** loss (>= 0) -- the ``-2 sigma0 u_t`` term (model #2's ``sigma``).
        Damps every mode at the same base rate.
    sigma1 : float
        **Frequency-dependent** loss (>= 0) -- the ``+2 sigma1 u_txx`` term. Damps mode ``m`` at an
        extra ``sigma1 * p_m^2``; high partials die faster. ``0`` -> reduces to ``StiffString``.
    theta : float
        Time-averaging weight in ``(0, 1]``; ``>= 1/4`` is unconditionally stable. Default a hair
        above ``1/4`` (accuracy-first), matching :class:`StiffString`.
    boundary : {"supported"}
        Simply-supported ends (the only boundary with a clean closed-form oracle).

    Raises
    ------
    ValueError
        If parameters are non-physical (negative tension/density/stiffness/losses, ``N < 2``,
        ``theta`` outside ``(0, 1]``, unsupported boundary).
    """

    def __init__(
        self,
        *,
        L: float,
        T: float,
        rho: float,
        fs: float,
        N: int,
        kappa: float = 0.0,
        sigma0: float = 0.0,
        sigma1: float = 0.0,
        theta: float = THETA_DEFAULT,
        boundary: Boundary = "supported",
    ) -> None:
        if min(L, T, rho, fs) <= 0:
            raise ValueError("L, T, rho, fs must all be positive.")
        if N < 2:
            raise ValueError("N must be >= 2 (need at least one interior node).")
        if kappa < 0:
            raise ValueError("kappa (stiffness) must be >= 0.")
        if sigma0 < 0:
            raise ValueError("sigma0 (frequency-independent loss) must be >= 0.")
        if sigma1 < 0:
            raise ValueError("sigma1 (frequency-dependent loss) must be >= 0.")
        if not (0.0 < theta <= 1.0):
            raise ValueError(f"theta must be in (0, 1], got {theta}.")
        if boundary != "supported":
            raise ValueError(f"boundary must be 'supported', got {boundary!r}.")

        self.L = float(L)
        self.T = float(T)
        self.rho = float(rho)
        self.fs = float(fs)
        self.N = int(N)
        self.kappa = float(kappa)
        self.sigma0 = float(sigma0)
        self.sigma1 = float(sigma1)
        self.theta = float(theta)
        self.boundary: Boundary = boundary

        self.c = float(np.sqrt(T / rho))
        self.h = self.L / self.N
        self.k = 1.0 / self.fs
        self.lam = self.c * self.k / self.h  # reported only; no CFL limit (unconditional)
        self.B = float((np.pi ** 2) * self.kappa ** 2 / (self.c ** 2 * self.L ** 2))

        self.x: NDArray[np.float64] = np.linspace(0.0, self.L, self.N + 1)

        # Conservative interior operator L = c^2 delta_xx - kappa^2 delta_xxxx, on u[1 .. N-1].
        # D2 is kept separately for the frequency-dependent loss term (sigma1 delta_t. delta_xx u).
        self._D2 = second_difference_matrix(self.N, self.h)
        self._L = (self.c ** 2) * self._D2
        if self.kappa != 0.0:
            self._L = self._L - (self.kappa ** 2) * biharmonic_matrix(self.N, self.h)
        self._L = self._L.tocsr()

        # A = (1 + sigma0 k) I - theta k^2 L - sigma1 k D2  (pentadiagonal SPD; factor once).
        s0k = self.sigma0 * self.k
        n_int = self.N - 1
        ident = sparse.identity(n_int, format="csr")
        A = (1.0 + s0k) * ident - (self.theta * self.k ** 2) * self._L
        if self.sigma1 != 0.0:
            A = A - (self.sigma1 * self.k) * self._D2
        A = A.tocsr()
        # Upper-banded storage for cholesky_banded (2 superdiagonals: pentadiagonal symmetric).
        ab = np.zeros((3, n_int))
        ab[2, :] = A.diagonal(0)
        ab[1, 1:] = A.diagonal(1)
        ab[0, 2:] = A.diagonal(2)
        self._chol = cholesky_banded(ab, lower=False)

        self.u: NDArray[np.float64] = np.zeros(self.N + 1)
        self.u_prev: NDArray[np.float64] = np.zeros(self.N + 1)
        self.n: int = 0  # number of completed steps

    # -- initial conditions -------------------------------------------------------------

    def set_state(
        self,
        u0: NDArray[np.float64],
        v0: NDArray[np.float64] | float = 0.0,
    ) -> None:
        """Set the initial displacement (and optional velocity).

        Uses the lossless consistent second-order start ``u^{-1} = u^0 - k v^0 + 1/2 k^2 L u^0``
        (the Taylor step with ``u_tt = L u`` at ``t = 0``, ignoring loss -- so a single
        eigenmode oscillates as a clean discrete cosine). The end nodes are clamped to 0. Under
        damping this start is only *consistent* (not exact), so the first few steps deviate slightly
        from the asymptotic ``g_m^n`` decay -- decay measurements skip the start (see the plan).
        """
        u0 = np.asarray(u0, dtype=float).copy()
        if u0.shape != (self.N + 1,):
            raise ValueError(f"u0 must have shape {(self.N + 1,)}, got {u0.shape}.")
        v0_arr = np.broadcast_to(np.asarray(v0, dtype=float), (self.N + 1,)).copy()

        u0[0] = u0[-1] = 0.0
        Lu0 = self._apply_L(u0)  # full-grid L u0 (0 at boundary nodes)
        u_prev = u0 - self.k * v0_arr + 0.5 * self.k ** 2 * Lu0
        u_prev[0] = u_prev[-1] = 0.0

        self.u = u0
        self.u_prev = u_prev
        self.n = 0

    # -- time stepping ------------------------------------------------------------------

    def step(self) -> None:
        """Advance one timestep via the banded SPD back-substitution (rolls the history)."""
        s0k = self.sigma0 * self.k
        un = self.u[1:-1]
        up = self.u_prev[1:-1]
        Lu = self._L @ un
        Lu_prev = self._L @ up
        rhs = (
            2.0 * un
            + (1.0 - 2.0 * self.theta) * self.k ** 2 * Lu
            - up
            + self.theta * self.k ** 2 * Lu_prev
            + s0k * up
        )
        if self.sigma1 != 0.0:
            # centered freq-dependent loss: +2 sigma1 delta_t.(delta_xx u) -> -sigma1 k D2 u^{n-1}
            rhs = rhs - (self.sigma1 * self.k) * (self._D2 @ up)
        u_next_int = cho_solve_banded((self._chol, False), rhs)

        u_next = np.zeros(self.N + 1)
        u_next[1:-1] = u_next_int
        self.u_prev = self.u
        self.u = u_next
        self.n += 1

    # -- diagnostics --------------------------------------------------------------------

    @property
    def state(self) -> NDArray[np.float64]:
        """Current displacement field ``u^n`` (a copy, safe to mutate/store for plotting)."""
        return self.u.copy()

    def energy(self) -> float:
        """Discrete energy ``E^n`` (Joules) -- the model #2 form, unchanged.

        The loss terms never enter the stored energy, only its rate of change. Lossless -> conserved
        to machine precision; lossy -> monotone decreasing (passivity, unconditional).
        """
        un = self.u[1:-1]
        up = self.u_prev[1:-1]
        dt_u = (un - up) / self.k  # delta_t- u^n on the interior (boundary velocity is 0)
        kinetic = 0.5 * self.h * float(np.dot(dt_u, dt_u))

        p_nn = self._P(un, un)
        p_pp = self._P(up, up)
        p_np = self._P(un, up)
        potential = 0.5 * self.theta * (p_nn + p_pp) + (0.5 - self.theta) * p_np
        return self.rho * (kinetic + potential)

    def displacement_at(self, index: int) -> float:
        """Displacement at grid node ``index`` -- a pickup for spectral analysis."""
        return float(self.u[index])

    # -- internals ----------------------------------------------------------------------

    def _P(self, f: NDArray[np.float64], g: NDArray[np.float64]) -> float:
        """Potential bilinear form ``P(f,g) = <-L f, g> = h * (-L f) . g`` (interior vectors)."""
        return -self.h * float(np.dot(self._L @ f, g))

    def _apply_L(self, u_full: NDArray[np.float64]) -> NDArray[np.float64]:
        """``L u`` returned on the full grid (zeros at the clamped boundary nodes)."""
        out = np.zeros_like(u_full)
        out[1:-1] = self._L @ u_full[1:-1]
        return out
