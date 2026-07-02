"""Modal body / soundboard — a bank of damped harmonic oscillators (the "body" node of the
``exciter -> resonator -> body/radiation`` abstraction, HANDOFF §3.2).

Physically this is a lumped **modal** model of a radiating body (guitar top, violin corpus, drum
shell): the body's motion is expanded on its own eigenmodes, each an independent damped oscillator

    q_i'' + 2 sigma_i q_i' + omega_i^2 q_i = phi_i F / m_i ,      omega_i = 2 pi f_i

driven by a scalar force ``F`` applied at the body's **driving point** (the bridge), where the
physical driving-point displacement is ``w_b = sum_i phi_i q_i``. ``phi_i`` is the mode shape
sampled at that point, ``m_i`` the modal mass. Radiated pressure is read out as ``sum_i a_i q_i''``
(a
monopole radiates in proportion to volume *acceleration*).

Discretisation mirrors :class:`physsynth.core.string_ideal.IdealString` exactly — the explicit
leapfrog with a **cross-time** potential term ``1/2 omega_i^2 q_i^n q_i^{n-1}``. That two-time-level
form (not the naive same-time ``1/2 omega_i^2 (q_i^n)^2``) is what makes the discrete energy
conserved to machine precision for a lossless mode; the same warning the string docstring gives
applies here. Each mode is stable under ``omega_i k < 2`` (the modal CFL), asserted at construction.

This module is the standalone body **and** the minimal test-fixture for the energy-conserving bridge
connection (built on top of it): the connection is validated by ``E_string + E_body + E_conn`` being
conserved, so the body must first conserve its own energy in isolation.

Headless: NumPy only.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# omega*k above 2 makes the explicit leapfrog unstable; allow a hair of floating-point slack so a
# requested omega*k == 2 boundary case is not spuriously rejected (mirrors IdealString._LAMBDA_TOL).
_OMEGA_K_TOL = 1e-12


class ModalBody:
    """A radiating body as a bank of independent damped modal oscillators.

    Parameters
    ----------
    freqs : array_like
        Modal frequencies ``f_i`` (Hz), one per mode. Must be positive.
    fs : float
        Sample rate (Hz); timestep ``k = 1/fs``.
    sigmas : array_like or float
        Per-mode loss coefficients ``sigma_i`` (>= 0) in the ``2 sigma_i q_i'`` term. A scalar is
        broadcast to every mode. ``0`` -> lossless (energy-conserving) mode.
    masses : array_like or float
        Modal masses ``m_i`` (> 0). A scalar is broadcast. Together with ``phi`` these set the
        driving-point admittance the string will feel.
    phi : array_like or float
        Driving-point mode-shape amplitudes ``phi_i`` (dimensionless): the value of eigenmode ``i``
        at the bridge. The physical bridge displacement is ``sum_i phi_i q_i`` and a bridge force
        ``F`` enters mode ``i`` as ``phi_i F``. A scalar is broadcast (default 1).
    radiation : array_like or float, optional
        Radiation weights ``a_i`` for the pressure read-out ``p = sum_i a_i q_i''``. Defaults to
        ``phi`` (the driving-point amplitudes) if not given.

    Raises
    ------
    ValueError
        On non-physical parameters or if any mode violates the CFL ``omega_i k < 2`` (the explicit
        leapfrog would be unstable — raise ``fs``).
    """

    def __init__(
        self,
        *,
        freqs: NDArray[np.float64],
        fs: float,
        sigmas: NDArray[np.float64] | float = 0.0,
        masses: NDArray[np.float64] | float = 1.0,
        phi: NDArray[np.float64] | float = 1.0,
        radiation: NDArray[np.float64] | float | None = None,
    ) -> None:
        f = np.atleast_1d(np.asarray(freqs, dtype=float)).copy()
        if f.ndim != 1 or f.size < 1:
            raise ValueError("freqs must be a 1-D array with at least one mode.")
        if np.any(f <= 0.0):
            raise ValueError("all modal frequencies must be positive.")
        if fs <= 0.0:
            raise ValueError("fs must be positive.")

        M = f.size
        sig = np.broadcast_to(np.asarray(sigmas, dtype=float), (M,)).copy()
        m = np.broadcast_to(np.asarray(masses, dtype=float), (M,)).copy()
        ph = np.broadcast_to(np.asarray(phi, dtype=float), (M,)).copy()
        if np.any(sig < 0.0):
            raise ValueError("sigmas (loss) must all be >= 0.")
        if np.any(m <= 0.0):
            raise ValueError("masses must all be positive.")

        self.freqs = f
        self.fs = float(fs)
        self.k = 1.0 / self.fs
        self.sigma = sig
        self.m = m
        self.phi = ph
        self.a = ph.copy() if radiation is None else \
            np.broadcast_to(np.asarray(radiation, dtype=float), (M,)).copy()
        self.M = M

        self.omega = 2.0 * np.pi * f
        self.omega_k = self.omega * self.k
        bad = self.omega_k >= 2.0 - _OMEGA_K_TOL
        if np.any(bad):
            worst = int(np.argmax(self.omega_k))
            raise ValueError(
                f"CFL violated: omega*k = {self.omega_k[worst]:.6f} >= 2 for mode {worst} "
                f"(f = {f[worst]:.3f} Hz at fs = {self.fs:.1f}). Raise fs."
            )

        self.q: NDArray[np.float64] = np.zeros(M)
        self.q_prev: NDArray[np.float64] = np.zeros(M)
        # Modal acceleration q_i'' of the most recent step, taken from the actual second difference
        # so it captures *every* force (internal, damping, and any external bridge force) — the
        # radiation read-out reads this. Initialised to the lossless free response.
        self._accel: NDArray[np.float64] = np.zeros(M)
        self.n: int = 0  # completed steps

    # -- initial conditions -------------------------------------------------------------

    def set_state(
        self,
        q0: NDArray[np.float64] | float,
        v0: NDArray[np.float64] | float = 0.0,
    ) -> None:
        """Set the initial modal displacement (and optional modal velocity).

        Uses the consistent second-order start ``q^{-1} = q^0 - k v^0 - 1/2 k^2 omega^2 q^0`` (the
        lossless Taylor step, matching :meth:`IdealString.set_state`) so a single lossless mode
        oscillates as a clean discrete cosine with no first-step transient.
        """
        q0 = np.broadcast_to(np.asarray(q0, dtype=float), (self.M,)).copy()
        v0 = np.broadcast_to(np.asarray(v0, dtype=float), (self.M,)).copy()
        self.q = q0
        self.q_prev = q0 - self.k * v0 - 0.5 * self.k * self.k * self.omega * self.omega * q0
        self._accel = -self.omega * self.omega * q0  # lossless free response at n = 0
        self.n = 0

    # -- time stepping ------------------------------------------------------------------

    def step(self, force: float = 0.0) -> None:
        """Advance one timestep under an optional scalar bridge ``force`` ``F`` (default 0).

        The force enters mode ``i`` as ``phi_i F`` (the generalised force). With ``force = 0`` this
        is the unforced, energy-conserving (lossless) / passive (lossy) modal update.
        """
        sk = self.sigma * self.k
        k2 = self.k * self.k
        rhs = (
            2.0 * self.q
            - (1.0 - sk) * self.q_prev
            - k2 * self.omega * self.omega * self.q
            + k2 * self.phi * force / self.m
        )
        q_next = rhs / (1.0 + sk)
        self._accel = (q_next - 2.0 * self.q + self.q_prev) / k2  # true q'' (all forces)
        self.q_prev = self.q
        self.q = q_next
        self.n += 1

    # -- diagnostics --------------------------------------------------------------------

    @property
    def state(self) -> NDArray[np.float64]:
        """Current modal-displacement vector ``q^n`` (a copy)."""
        return self.q.copy()

    def energy(self) -> float:
        """Discrete modal energy ``E^n`` (Joules), cross-time potential.

        ``E^n = sum_i m_i [ 1/2 (delta_t- q_i)^2 + 1/2 omega_i^2 q_i^n q_i^{n-1} ]``. Conserved to
        machine precision for a lossless body; monotonically decreasing when any ``sigma_i > 0``.
        """
        dt_q = (self.q - self.q_prev) / self.k
        kinetic = 0.5 * float(np.dot(self.m, dt_q * dt_q))
        potential = 0.5 * float(np.dot(self.m * self.omega * self.omega, self.q * self.q_prev))
        return kinetic + potential

    def bridge_displacement(self) -> float:
        """Physical driving-point (bridge) displacement ``w_b = sum_i phi_i q_i^n``."""
        return float(np.dot(self.phi, self.q))

    def bridge_velocity(self) -> float:
        """Driving-point velocity ``sum_i phi_i (delta_t- q_i)`` (backward difference)."""
        return float(np.dot(self.phi, (self.q - self.q_prev) / self.k))

    def pressure(self) -> float:
        """Radiated pressure read-out ``p = sum_i a_i q_i''`` (monopole ∝ volume acceleration).

        Uses :attr:`_accel`, the modal acceleration taken from the *actual* second difference of the
        last step, so it reflects every force acting on the mode — including a bridge force when the
        body is coupled. (Reconstructing ``q'' = -omega^2 q - 2 sigma q'`` would silently omit that
        coupling term.)
        """
        return float(np.dot(self.a, self._accel))

    def displacement_at(self, index: int) -> float:
        """Modal coordinate ``q_index`` — lets :func:`engine.simulate` tap a single mode."""
        return float(self.q[index])
