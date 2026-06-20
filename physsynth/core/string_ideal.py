"""Ideal (lossless or linearly damped) transverse string — explicit FDTD scheme.

Implements HANDOFF §4.2-§4.3:

    u_tt = c^2 u_xx - 2 sigma u_t,      c = sqrt(T / rho)

with the explicit second-order scheme

    u[l, n+1] = 2 u[l,n] - u[l,n-1] + lambda^2 (u[l+1,n] - 2 u[l,n] + u[l-1,n]),
    lambda = c k / h   (Courant number; lambda <= 1 required for stability).

The defining feature is :meth:`energy`, which uses the **cross-time** potential term

    E^n = rho [ 1/2 ||delta_t- u^n||_w^2  +  (c^2/2) <delta_x+ u^n, delta_x+ u^{n-1}> ]

(the strain energy is a product of the gradient at steps n and n-1). This two-time-level form is
what makes the discrete energy conserved to machine precision for a lossless run; the intuitive
same-time form ||delta_x+ u^n||^2 drifts at ~1e-3. Do not "simplify" it.

Headless: NumPy only.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .operators import delta_x_forward, inner

Boundary = Literal["fixed", "free"]

# Courant numbers above 1 are forbidden for the explicit scheme; allow a hair of
# floating-point slack so that a requested lambda == 1 is not spuriously rejected.
_LAMBDA_TOL = 1e-12


class IdealString:
    """A discretized ideal string resonator.

    Parameters
    ----------
    L, T, rho : float
        Length (m), tension (N), linear density (kg/m). Wave speed ``c = sqrt(T/rho)``.
    fs : float
        Sample rate (Hz); timestep ``k = 1/fs``.
    N : int
        Number of spatial segments; the grid has ``N + 1`` nodes, spacing ``h = L/N``.
    boundary : {"fixed", "free"}
        ``"fixed"`` (Dirichlet, ``u = 0`` at ends) or ``"free"`` (Neumann, ``u_x = 0`` via a
        reflected stencil). Both conserve energy in the lossless case.
    sigma : float
        Loss coefficient (>= 0) for the ``-2 sigma u_t`` term. ``0`` -> lossless.

    Raises
    ------
    ValueError
        If parameters are non-physical or the Courant number ``lambda = c k / h > 1``
        (the explicit scheme would be unstable).
    """

    def __init__(
        self,
        *,
        L: float,
        T: float,
        rho: float,
        fs: float,
        N: int,
        boundary: Boundary = "fixed",
        sigma: float = 0.0,
    ) -> None:
        if min(L, T, rho, fs) <= 0:
            raise ValueError("L, T, rho, fs must all be positive.")
        if N < 2:
            raise ValueError("N must be >= 2 (need at least one interior node).")
        if sigma < 0:
            raise ValueError("sigma (loss) must be >= 0.")
        if boundary not in ("fixed", "free"):
            raise ValueError(f"boundary must be 'fixed' or 'free', got {boundary!r}.")

        self.L = float(L)
        self.T = float(T)
        self.rho = float(rho)
        self.fs = float(fs)
        self.N = int(N)
        self.boundary: Boundary = boundary
        self.sigma = float(sigma)

        self.c = float(np.sqrt(T / rho))
        self.h = self.L / self.N
        self.k = 1.0 / self.fs
        self.lam = self.c * self.k / self.h

        if self.lam > 1.0 + _LAMBDA_TOL:
            raise ValueError(
                f"CFL violated: lambda = c*k/h = {self.lam:.6f} > 1. "
                "Reduce fs, refine the grid (increase N), or lower the wave speed."
            )

        self.x: NDArray[np.float64] = np.linspace(0.0, self.L, self.N + 1)

        # Trapezoidal node weights for the (kinetic) inner product: h/2 at the two boundary
        # nodes, h in the interior. This is the SBP-consistent weighting that keeps the
        # free-boundary energy exact; for fixed ends the boundary velocity is zero so it is
        # equivalent to uniform weighting.
        self._w: NDArray[np.float64] = np.full(self.N + 1, self.h)
        self._w[0] = self._w[-1] = 0.5 * self.h

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

        Uses the consistent second-order start ``u^{-1} = u^0 - k v^0 + 1/2 * stencil(u^0)`` so
        that a single-mode initial condition oscillates as a clean discrete cosine (no spurious
        first-step transient) and zero initial velocity is exact.
        """
        u0 = np.asarray(u0, dtype=float).copy()
        if u0.shape != (self.N + 1,):
            raise ValueError(f"u0 must have shape {(self.N + 1,)}, got {u0.shape}.")
        v0_arr = np.broadcast_to(np.asarray(v0, dtype=float), (self.N + 1,)).copy()

        self._apply_boundary(u0)
        # lambda^2 * (second difference) — the lossless acceleration * k^2.
        stencil0 = self.lam * self.lam * self._second_diff(u0)
        self.u = u0
        self.u_prev = u0 - self.k * v0_arr + 0.5 * stencil0
        self._apply_boundary(self.u_prev)
        self.n = 0

    # -- time stepping ------------------------------------------------------------------

    def step(self) -> None:
        """Advance one timestep (updates :attr:`u` in place, rolling the history)."""
        sk = self.sigma * self.k
        stencil = self.lam * self.lam * self._second_diff(self.u)
        u_next = (2.0 * self.u - (1.0 - sk) * self.u_prev + stencil) / (1.0 + sk)
        self._apply_boundary(u_next)
        self.u_prev = self.u
        self.u = u_next
        self.n += 1

    # -- diagnostics --------------------------------------------------------------------

    @property
    def state(self) -> NDArray[np.float64]:
        """Current displacement field ``u^n`` (a copy, safe to mutate/store for plotting)."""
        return self.u.copy()

    def energy(self) -> float:
        """Discrete energy ``E^n`` (Joules) using the cross-time potential term.

        For a lossless run this is conserved to machine precision; for ``sigma > 0`` it
        decreases monotonically (passivity).
        """
        dt_u = (self.u - self.u_prev) / self.k  # delta_t- u^n
        kinetic = 0.5 * float(np.dot(self._w, dt_u * dt_u))
        gx_now = delta_x_forward(self.u, self.h)
        gx_prev = delta_x_forward(self.u_prev, self.h)
        potential = 0.5 * self.c * self.c * inner(gx_now, gx_prev, self.h)
        return self.rho * (kinetic + potential)

    def displacement_at(self, index: int) -> float:
        """Displacement at grid node ``index`` — a pickup for spectral analysis."""
        return float(self.u[index])

    # -- internals ----------------------------------------------------------------------

    def _second_diff(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """``u[l+1] - 2u[l] + u[l-1]`` over the whole grid, with the boundary stencil.

        Fixed ends contribute 0 at the boundary nodes (they are held clamped); free ends use the
        reflected (Neumann) stencil ``2(u[1]-u[0])`` and ``2(u[N-1]-u[N])``.
        """
        s = np.empty_like(u)
        s[1:-1] = u[2:] - 2.0 * u[1:-1] + u[:-2]
        if self.boundary == "free":
            s[0] = 2.0 * (u[1] - u[0])
            s[-1] = 2.0 * (u[-2] - u[-1])
        else:  # fixed: boundary nodes are clamped, never updated
            s[0] = 0.0
            s[-1] = 0.0
        return s

    def _apply_boundary(self, u: NDArray[np.float64]) -> None:
        if self.boundary == "fixed":
            u[0] = 0.0
            u[-1] = 0.0
