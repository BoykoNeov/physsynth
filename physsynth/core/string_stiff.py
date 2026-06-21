"""Stiff (lossless or linearly damped) transverse string — implicit theta-scheme FDTD.

Implements HANDOFF section 5 model #2 (see ``docs/dev/stiff-string-plan.md``). PDE:

    u_tt = c^2 u_xx - kappa^2 u_xxxx - 2 sigma u_t,    c^2 = T/rho,  kappa^2 = E I / rho

The bending term ``-kappa^2 u_xxxx`` stretches the partials off the harmonic series (piano-like
inharmonicity ``f_n = n f0 sqrt(1 + B n^2)``, ``B = pi^2 kappa^2 / (c^2 L^2)``). Treating it
explicitly would force a brutal ``kappa^2 k^2 / h^4`` CFL, so the *whole* spatial operator
``L = c^2 delta_xx - kappa^2 delta_xxxx`` is time-averaged with a theta weight:

    delta_tt u = L (theta u^{n+1} + (1-2 theta) u^n + theta u^{n-1}) - 2 sigma delta_t. u

which is unconditionally stable for ``theta >= 1/4`` (so ``theta`` is a pure accuracy knob, not a
stability one). Rearranged, each step is one banded SPD solve with the constant matrix

    A = (1 + sigma k) I - theta k^2 L      (pentadiagonal, factored once at construction).

**Boundary: simply supported** (``u = 0`` and ``u_xx = 0``). Building ``L``'s biharmonic block as
``(delta_xx)^2`` (see :func:`physsynth.core.operators.biharmonic_matrix`) bakes in both conditions
and keeps ``sin(m pi x / L)`` an *exact* discrete eigenvector, so the modal / dispersion harness
carries over unchanged. ``kappa = 0`` recovers the implicit wave scheme -- which is **not** the
explicit :class:`~physsynth.core.string_ideal.IdealString` (a different, still-validated scheme; it
is not even exact at ``lambda = 1``).

**Energy** (theta-dependent; reduces to the ideal string's cross-time form at ``theta = 0``):

    E^n = rho [ 1/2 ||delta_t- u^n||^2
                + (theta/2)(P(u^n,u^n) + P(u^{n-1},u^{n-1})) + (1/2 - theta) P(u^n,u^{n-1}) ]

with ``P(f,g) = <-L f, g> = c^2 <delta_x+ f, delta_x+ g> + kappa^2 <delta_xx f, delta_xx g> >= 0``
(summation by parts). ``P`` is evaluated through the *same* matrix ``L`` used in the update, so the
conservation ``E^{n+1} = E^n`` is an exact algebraic identity (machine precision lossless; monotone
decreasing at ``e^{-2 sigma t}`` lossy).

Headless: NumPy + SciPy (banded Cholesky). No I/O, no plotting.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.linalg import cho_solve_banded, cholesky_banded

from .operators import biharmonic_matrix, second_difference_matrix

Boundary = Literal["supported"]

# theta below 1/4 is only conditionally stable; theta in (0, 1] keeps A SPD (genuinely implicit).
# Default a hair above 1/4 (accuracy-first per the plan) so the energy has a small positivity margin
# while staying near the minimal-dispersion theta = 1/4.
THETA_DEFAULT = 0.28


class StiffString:
    """A discretized stiff string resonator (implicit theta-scheme, simply-supported ends).

    Parameters
    ----------
    L, T, rho : float
        Length (m), tension (N), linear density (kg/m). Wave speed ``c = sqrt(T/rho)``.
    fs : float
        Sample rate (Hz); timestep ``k = 1/fs``. No CFL limit (unconditionally stable for
        ``theta >= 1/4``) -- coarse grids / large timesteps that the explicit stiff scheme could
        not run are admissible here.
    N : int
        Number of spatial segments; the grid has ``N + 1`` nodes, spacing ``h = L/N``. The two end
        nodes are clamped (``u = 0``); the ``N - 1`` interior nodes are the unknowns.
    kappa : float
        Stiffness coefficient ``kappa = sqrt(E I / rho)`` (units m^2/s). ``0`` -> ideal wave
        equation (implicit scheme). Larger ``kappa`` -> more inharmonicity ``B = pi^2 kappa^2 /
        (c^2 L^2)``.
    sigma : float
        Frequency-independent loss coefficient (>= 0) for the ``-2 sigma u_t`` term. ``0`` ->
        lossless.
    theta : float
        Time-averaging weight in ``(0, 1]``. ``>= 1/4`` is unconditionally stable; smaller theta is
        more accurate (less numerical dispersion) but only conditionally stable. Default a hair
        above ``1/4``.
    boundary : {"supported"}
        Simply-supported ends. The only boundary with a clean closed-form oracle; clamped / free are
        deferred (see the plan).

    Raises
    ------
    ValueError
        If parameters are non-physical (negative tension/density/stiffness/loss, ``N < 2``,
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
        sigma: float = 0.0,
        theta: float = THETA_DEFAULT,
        boundary: Boundary = "supported",
    ) -> None:
        if min(L, T, rho, fs) <= 0:
            raise ValueError("L, T, rho, fs must all be positive.")
        if N < 2:
            raise ValueError("N must be >= 2 (need at least one interior node).")
        if kappa < 0:
            raise ValueError("kappa (stiffness) must be >= 0.")
        if sigma < 0:
            raise ValueError("sigma (loss) must be >= 0.")
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
        self.sigma = float(sigma)
        self.theta = float(theta)
        self.boundary: Boundary = boundary

        self.c = float(np.sqrt(T / rho))
        self.h = self.L / self.N
        self.k = 1.0 / self.fs
        self.lam = self.c * self.k / self.h  # reported only; no CFL limit (unconditional)
        self.B = float((np.pi ** 2) * self.kappa ** 2 / (self.c ** 2 * self.L ** 2))

        self.x: NDArray[np.float64] = np.linspace(0.0, self.L, self.N + 1)

        # Interior spatial operator L = c^2 delta_xx - kappa^2 delta_xxxx, acting on u[1 .. N-1].
        # Building the biharmonic block as (delta_xx)^2 makes the discrete energy exactly the
        # ||delta_xx u||^2 cross-time form (SBP-exact) and bakes in u_xx = 0 at the ends.
        d2 = second_difference_matrix(self.N, self.h)
        self._L = (self.c ** 2) * d2
        if self.kappa != 0.0:
            self._L = self._L - (self.kappa ** 2) * biharmonic_matrix(self.N, self.h)
        self._L = self._L.tocsr()

        # A = (1 + sigma k) I - theta k^2 L  (pentadiagonal SPD, constant in time -> factor once).
        sk = self.sigma * self.k
        n_int = self.N - 1
        A = (1.0 + sk) * sparse.identity(n_int, format="csr") - (self.theta * self.k ** 2) * self._L
        A = A.tocsr()
        # Upper-banded storage for cholesky_banded (2 superdiagonals: pentadiagonal symmetric).
        ab = np.zeros((3, n_int))
        ab[2, :] = A.diagonal(0)
        ab[1, 1:] = A.diagonal(1)
        ab[0, 2:] = A.diagonal(2)
        self._chol = cholesky_banded(ab, lower=False)

        # Full-grid state (boundary nodes stay clamped at 0); the solve touches only the interior.
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

        Uses the consistent second-order start ``u^{-1} = u^0 - k v^0 + 1/2 k^2 L u^0`` (the
        discrete analog of a Taylor step with acceleration ``u_tt = L u`` at ``t = 0``), so a single
        eigenmode oscillates as a clean discrete cosine and zero initial velocity is exact to second
        order. The end nodes are clamped to 0.
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
        sk = self.sigma * self.k
        Lu = self._L @ self.u[1:-1]
        Lu_prev = self._L @ self.u_prev[1:-1]
        rhs = (
            2.0 * self.u[1:-1]
            + (1.0 - 2.0 * self.theta) * self.k ** 2 * Lu
            - self.u_prev[1:-1]
            + self.theta * self.k ** 2 * Lu_prev
            + sk * self.u_prev[1:-1]
        )
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
        """Discrete energy ``E^n`` (Joules) for the implicit theta-scheme.

        ``E^n = rho [ 1/2 ||delta_t- u||^2 + (theta/2)(P_nn + P_pp) + (1/2 - theta) P_np ]`` with
        ``P(f,g) = <-L f, g>``. Lossless -> conserved to machine precision; lossy -> monotone
        decreasing.
        """
        un = self.u[1:-1]
        up = self.u_prev[1:-1]
        dt_u = (un - up) / self.k  # delta_t- u^n on the interior (boundary velocity is 0)
        kinetic = 0.5 * self.h * float(np.dot(dt_u, dt_u))

        p_nn = self._P(un, un)
        p_pp = self._P(up, up)
        p_np = self._P(un, up)
        potential = (
            0.5 * self.theta * (p_nn + p_pp) + (0.5 - self.theta) * p_np
        )
        return self.rho * (kinetic + potential)

    def displacement_at(self, index: int) -> float:
        """Displacement at grid node ``index`` -- a pickup for spectral analysis."""
        return float(self.u[index])

    # -- internals ----------------------------------------------------------------------

    def _P(self, f: NDArray[np.float64], g: NDArray[np.float64]) -> float:
        """Potential bilinear form ``P(f,g) = <-L f, g> = h * (-L f) . g`` (interior vectors).

        Uses the *same* matrix ``L`` as the update, so the energy identity is exact. ``-L`` is
        positive-definite, hence ``P(f,f) >= 0``.
        """
        return -self.h * float(np.dot(self._L @ f, g))

    def _apply_L(self, u_full: NDArray[np.float64]) -> NDArray[np.float64]:
        """``L u`` returned on the full grid (zeros at the clamped boundary nodes)."""
        out = np.zeros_like(u_full)
        out[1:-1] = self._L @ u_full[1:-1]
        return out
