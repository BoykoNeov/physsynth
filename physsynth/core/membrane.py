"""2D membrane (drumhead) resonator — explicit 5-point FDTD (HANDOFF §5 model #4).

PDE (transverse displacement ``u(x, y, t)``, fixed rim):

    u_tt = c² ∇²u − 2σ u_t,     c² = T/ρ,     ∇² = ∂_xx + ∂_yy

with the explicit scheme

    u^{n+1} = (2 u^n − (1−σk) u^{n-1} + c²k² Δ_h u^n) / (1 + σk),

where ``Δ_h`` is the 5-point Laplacian restricted to the live (interior) nodes of the domain mask
(see :mod:`physsynth.core.operators2d`). Two domains share this one class via the mask: a
**rectangle** (exact ``sin·sin`` modes — the harness unit-test) and a **circle** (the drumhead, a
*staircased* round rim whose Bessel match degrades to ~O(h) while energy stays exact).

The defining feature, as in 1D, is :meth:`energy` — the **cross-time** potential

    E^n = ρ [ ½ ‖δ_t- u^n‖²  +  (c²/2) P(u^n, u^{n-1}) ],   P(f,g) = <−L f, g> = <∇_+ f, ∇_+ g> ≥ 0

evaluated through the *same* masked Laplacian ``L`` used in the update, so ``E^{n+1} = E^n`` is an
exact identity (machine-precision lossless; monotone decreasing lossy). The masked ``L`` is
symmetric ⇒ conservation holds for the staircased circle too. See ``docs/dev/membrane-plan.md``.

**2D CFL: λ = c k / h ≤ 1/√2** (the 5-point Laplacian's spectral radius is 8/h², double the 1D
case). And — unlike 1D — *no* λ is dispersionless: the 5-point scheme is anisotropic.

Headless: NumPy + SciPy. No I/O, no plotting.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .operators2d import (
    disk_mask,
    embed,
    grid_coords,
    laplacian_from_mask,
    rectangle_mask,
)

Domain = Literal["rectangle", "circle"]

# CFL ceiling for the 2D explicit scheme; a hair of slack so a requested lambda == 1/sqrt(2) is not
# spuriously rejected by floating-point round-off.
_LAMBDA_MAX = 1.0 / np.sqrt(2.0)
_LAMBDA_TOL = 1e-12


class Membrane:
    """A discretized membrane resonator (explicit 5-point FDTD, fixed rim).

    Parameters
    ----------
    domain : {"rectangle", "circle"}
        Geometry. ``"rectangle"`` needs ``Lx, Ly``; ``"circle"`` needs ``radius``.
    T, rho : float
        Tension per unit length (N/m) and areal density (kg/m²). Wave speed ``c = sqrt(T/rho)``.
    fs : float
        Sample rate (Hz); timestep ``k = 1/fs``.
    N : int
        Resolution control: number of grid segments along x (rectangle) or across the bounding box
        ``[-radius, radius]`` (circle). Spacing ``h`` is derived; cells are square.
    Lx, Ly : float, optional
        Rectangle side lengths (m). ``Ly`` is snapped to an integer number of cells so cells stay
        square; the snapped value is stored back on :attr:`Ly`.
    radius : float, optional
        Disk radius (m) for ``domain="circle"``.
    sigma : float
        Frequency-independent loss (>= 0) for the ``-2σ u_t`` term. ``0`` -> lossless.

    Raises
    ------
    ValueError
        Non-physical parameters, a missing geometry argument, or a CFL violation
        ``λ = c k / h > 1/√2`` (the explicit scheme would be unstable).
    """

    def __init__(
        self,
        *,
        domain: Domain,
        T: float,
        rho: float,
        fs: float,
        N: int,
        Lx: float | None = None,
        Ly: float | None = None,
        radius: float | None = None,
        sigma: float = 0.0,
    ) -> None:
        if min(T, rho, fs) <= 0:
            raise ValueError("T, rho, fs must all be positive.")
        if N < 2:
            raise ValueError("N must be >= 2.")
        if sigma < 0:
            raise ValueError("sigma (loss) must be >= 0.")

        self.domain: Domain = domain
        self.T = float(T)
        self.rho = float(rho)
        self.fs = float(fs)
        self.N = int(N)
        self.sigma = float(sigma)

        self.c = float(np.sqrt(T / rho))
        self.k = 1.0 / self.fs

        if domain == "rectangle":
            if Lx is None or Ly is None:
                raise ValueError("domain='rectangle' requires Lx and Ly.")
            if min(Lx, Ly) <= 0:
                raise ValueError("Lx, Ly must be positive.")
            self.h = float(Lx) / self.N
            Ny = max(int(round(float(Ly) / self.h)), 1)
            self.Lx = float(Lx)
            self.Ly = Ny * self.h  # snapped so cells are square
            xs = np.linspace(0.0, self.Lx, self.N + 1)
            ys = np.linspace(0.0, self.Ly, Ny + 1)
            self.X, self.Y = np.meshgrid(xs, ys)
            self.mask = rectangle_mask(self.N, Ny)
            self.radius = None
        elif domain == "circle":
            if radius is None:
                raise ValueError("domain='circle' requires radius.")
            if radius <= 0:
                raise ValueError("radius must be positive.")
            self.radius = float(radius)
            self.X, self.Y, self.h = grid_coords(self.N, self.radius)
            self.mask = disk_mask(self.X, self.Y, self.radius)
            self.Lx = self.Ly = None
        else:
            raise ValueError(f"domain must be 'rectangle' or 'circle', got {domain!r}.")

        self.lam = self.c * self.k / self.h
        if self.lam > _LAMBDA_MAX + _LAMBDA_TOL:
            raise ValueError(
                f"CFL violated: lambda = c*k/h = {self.lam:.6f} > 1/sqrt(2) = {_LAMBDA_MAX:.6f}. "
                "Reduce fs, refine the grid (increase N), or lower the wave speed."
            )

        self.L, self.index_map = laplacian_from_mask(self.mask, self.h)
        self.n_live = self.L.shape[0]
        if self.n_live < 1:
            raise ValueError("the domain mask has no interior (live) nodes; refine the grid.")

        self.u: NDArray[np.float64] = np.zeros(self.n_live)
        self.u_prev: NDArray[np.float64] = np.zeros(self.n_live)
        self.n: int = 0  # completed steps

    # -- initial conditions -------------------------------------------------------------

    def to_live(self, field: NDArray[np.float64]) -> NDArray[np.float64]:
        """Select the live-node values from a full 2D ``field`` (shape ``mask.shape``)."""
        field = np.asarray(field, dtype=float)
        if field.shape != self.mask.shape:
            raise ValueError(f"field must have shape {self.mask.shape}, got {field.shape}.")
        return field[self.mask]

    def set_state(
        self,
        u0: NDArray[np.float64],
        v0: NDArray[np.float64] | float = 0.0,
    ) -> None:
        """Set the initial displacement (and optional velocity).

        ``u0`` may be a full 2D field (shape ``mask.shape``) or a flat live-node vector
        (length ``n_live``). Uses the consistent second-order start
        ``u^{-1} = u^0 − k v^0 + ½ c²k² L u^0`` so a single eigenmode oscillates as a clean discrete
        cosine and zero initial velocity is exact to second order. Dead (rim) nodes stay clamped.
        """
        u0 = np.asarray(u0, dtype=float)
        if u0.shape == self.mask.shape:
            u0 = u0[self.mask]
        elif u0.shape != (self.n_live,):
            raise ValueError(
                f"u0 must have shape {self.mask.shape} (full field) or {(self.n_live,)} (live), "
                f"got {u0.shape}."
            )
        u0 = u0.copy()

        if np.isscalar(v0) or np.asarray(v0).shape == ():
            v0_live = np.full(self.n_live, float(v0))
        else:
            v0 = np.asarray(v0, dtype=float)
            v0_live = v0[self.mask] if v0.shape == self.mask.shape else v0

        c2k2 = self.c * self.c * self.k * self.k
        self.u = u0
        self.u_prev = u0 - self.k * v0_live + 0.5 * c2k2 * (self.L @ u0)
        self.n = 0

    # -- time stepping ------------------------------------------------------------------

    def step(self) -> None:
        """Advance one timestep (rolls the history)."""
        sk = self.sigma * self.k
        c2k2 = self.c * self.c * self.k * self.k
        u_next = (2.0 * self.u - (1.0 - sk) * self.u_prev + c2k2 * (self.L @ self.u)) / (1.0 + sk)
        self.u_prev = self.u
        self.u = u_next
        self.n += 1

    # -- diagnostics --------------------------------------------------------------------

    @property
    def state(self) -> NDArray[np.float64]:
        """Current displacement field ``u^n`` as a full 2D array (dead nodes are 0)."""
        return embed(self.u, self.index_map)

    def energy(self) -> float:
        """Discrete energy ``E^n`` (Joules) using the cross-time potential term.

        Lossless -> conserved to machine precision; ``sigma > 0`` -> monotone decreasing (passive).
        """
        h2 = self.h * self.h
        dt_u = (self.u - self.u_prev) / self.k  # delta_t- u^n (boundary velocity is 0)
        kinetic = 0.5 * h2 * float(np.dot(dt_u, dt_u))
        # P(u^n, u^{n-1}) = <-L u^n, u^{n-1}> = -h^2 (L u^n) . u^{n-1}  (>= 0; -L is SPD)
        p_np = -h2 * float(np.dot(self.L @ self.u, self.u_prev))
        potential = 0.5 * self.c * self.c * p_np
        return self.rho * (kinetic + potential)

    def displacement_at(self, index: int) -> float:
        """Displacement at flat live-node ``index`` — a pickup for spectral analysis."""
        return float(self.u[index])

    def pickup_index_at(self, x: float, y: float) -> int:
        """Flat live-node index nearest the physical point ``(x, y)`` (for placing a pickup)."""
        live = self.index_map >= 0
        xs = self.X[live]
        ys = self.Y[live]
        d2 = (xs - x) ** 2 + (ys - y) ** 2
        return int(np.argmin(d2))
