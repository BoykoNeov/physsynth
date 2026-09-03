"""Far-field acoustic radiation — the "air" node of the ``exciter -> resonator -> body/radiation``
abstraction (HANDOFF §3.2).

A vibrating body pushes air; at a listening distance ``r`` a microphone hears a **radiated
pressure**. This module is the first, lumped tier of that air node (the human's scope decision:
*lumped passive impedance*, built read-out-first). It turns the body's monopole source strength —
its net **volume acceleration** ``Q'' = d²/dt² ∫ w dA`` (already exposed project-wide as
``body.pressure()``) — into the far-field pressure of a **simple source (monopole)** in free space:

    p_far(t, r) = rho0 / (4 pi r) * Q''(t - r / c0) .

That is the exact free-space Green's-function solution for a compact source whose size is small
against the wavelength: the pressure is the volume *acceleration* scaled by ``rho0 / (4 pi r)`` and
retarded by the travel time ``r / c0`` (the wavefront takes ``r / c0`` seconds to reach the
listener). :class:`AirRadiation` itself carries no new state on the body and **no back-reaction**:
it is a pure output transform, so the body radiates "for free" and its own energy balance is
untouched — hence it deliberately has no ``energy()`` (it stores and dissipates none). The passive
**radiation load** (the air *damping* the body, with radiated power ``P_rad = R U²`` accounted as an
energy channel and a closed-form ``R_a`` oracle) is the separate :class:`RadiatedBody` below —
batch 2, the back-reaction.

**Why this is still validatable without energy** (the project rule is closed-form checks, not ears):
a prescribed sinusoidal volume velocity ``U(t) = U0 sin(omega t)`` has volume acceleration
``Q'' = U0 omega cos(omega t)``, so the monopole radiates a tone of amplitude
``|p_far| = rho0 omega U0 / (4 pi r)`` — an exact amplitude/phase oracle — and the pressure falls
off as ``1 / r`` (the inverse-distance law) with a pure ``r / c0`` delay. Those are the tests.

The retardation is applied as an **integer-sample** delay line (``round(r / c0 * fs)`` samples), so
the amplitude is preserved *exactly* (a fractional/interpolating delay would act as a mild low-pass
and blur the amplitude oracle); the sub-sample residual is exposed on :attr:`retardation_residual`
for honesty. Set ``retarded=False`` for the pure instantaneous gain (the delay is inaudible latency
for a single source, but matters once several sources are summed at one listener).

Headless: NumPy only.
"""

from __future__ import annotations

import math
import os
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.special import j1

from .body import ModalBody

# Air at ~20 °C, 1 atm. Defaults; override for other media (helium, water, altitude).
RHO0_AIR = 1.2041  # kg/m^3, ambient air density
C0_AIR = 343.0     # m/s, speed of sound in air


# -- radiation resistance (the closed-form load oracle) ---------------------------------------

def monopole_radiation_resistance(
    omega: float, *, rho0: float = RHO0_AIR, c0: float = C0_AIR
) -> float:
    """Free-space **acoustic** radiation resistance of a compact monopole (Pa·s/m³ = kg/m⁴·s).

    ``R_a(omega) = rho0 omega^2 / (4 pi c0)``. This is the resistive part of the free-space monopole
    load: a body radiating a volume velocity ``U`` (m³/s) into free space hands the far field
    ``P_rad = R_a U^2`` watts. It is the exact passive dual of the batch-1 free-space (``4 pi``)
    read-out — same geometry, same ``rho0``/``c0``. Acoustic (per-volume-velocity) units, **not**
    the mechanical ``rho0 c pi a^2`` resistance (which multiplies a *surface* velocity and is
    ``S^2`` larger). Note the ``∝ omega^2`` frequency dependence: the constant-``R``
    :class:`RadiatedBody` evaluates this at one reference frequency; true per-mode spectral shaping
    is a later, frequency-dependent batch.
    """
    return rho0 * omega * omega / (4.0 * np.pi * c0)


def piston_radiation_resistance(
    omega: float, radius: float, *, rho0: float = RHO0_AIR, c0: float = C0_AIR
) -> float:
    """Baffled circular-piston (half-space) **acoustic** radiation resistance (Pa·s/m³).

    ``R_a(ka) = (rho0 c0 / S) [1 - J1(2ka)/(ka)]``, ``S = pi a^2``, wavenumber ``k = omega / c0``
    (Rayleigh's classic result; ``J1`` the first-order Bessel function). As ``ka -> 0`` the bracket
    ``-> (ka)^2/2`` so ``R_a -> rho0 omega^2 / (2 pi c0)`` — the half-space monopole, exactly
    **twice** the free-space :func:`monopole_radiation_resistance` (a baffle radiates into ``2 pi``,
    not ``4 pi``). Offered as the closed-form ``R_a(ka)`` modeling oracle; the default load uses the
    free-space monopole to stay consistent with batch 1. Divided by ``S^2`` from the more commonly
    tabulated *mechanical* piston resistance ``rho0 c S [1 - J1(2ka)/(ka)]``.
    """
    ka = omega * radius / c0
    S = np.pi * radius * radius
    # 1 - J1(2ka)/(ka) -> (ka)^2/2 as ka -> 0 (a 0/0 in the direct form); use the series there.
    bracket = 0.5 * ka * ka if ka < 1e-8 else 1.0 - j1(2.0 * ka) / ka
    return rho0 * c0 / S * bracket


class _VolumeAccelerationSource(Protocol):
    """Anything with a monopole read-out ``pressure()`` = net volume acceleration ``Q''``.

    :class:`physsynth.core.body.ModalBody`, :class:`physsynth.core.plate.Plate`, and the bridge
    connections all satisfy this — :meth:`AirRadiation.radiate` duck-types on it.
    """

    def pressure(self) -> float: ...


class AirRadiation:
    """Free-space monopole radiation: body volume acceleration -> far-field pressure at ``r``.

    A pure, passive **output transform** (no back-reaction on the source — that is the next batch).
    Feed it the body's volume acceleration each step, or hand it the body and let it read
    ``body.pressure()``.

    Parameters
    ----------
    fs : float
        Sample rate (Hz); timestep ``k = 1 / fs``. Must match the body's ``fs`` so the retardation
        delay is counted in the right samples.
    distance : float, optional
        Listening distance ``r`` (m) from the (compact) source to the microphone. Default 1 m. Must
        be positive.
    rho0 : float, optional
        Ambient medium density ``rho0`` (kg/m³). Default :data:`RHO0_AIR`.
    c0 : float, optional
        Speed of sound ``c0`` (m/s) in the medium. Default :data:`C0_AIR`. Sets the retardation
        ``r / c0`` and (later) the radiation resistance.
    retarded : bool, optional
        If ``True`` (default) delay the output by the travel time ``round(r / c0 * fs)`` samples
        (integer, amplitude-exact). If ``False`` emit the instantaneous gain ``rho0/(4 pi r) * Q''``
        with no delay.

    Raises
    ------
    ValueError
        On non-physical parameters (``fs``, ``distance``, ``rho0`` or ``c0`` not positive).
    """

    def __init__(
        self,
        *,
        fs: float,
        distance: float = 1.0,
        rho0: float = RHO0_AIR,
        c0: float = C0_AIR,
        retarded: bool = True,
    ) -> None:
        if fs <= 0.0:
            raise ValueError("fs must be positive.")
        if distance <= 0.0:
            raise ValueError("distance (listening radius r) must be positive.")
        if rho0 <= 0.0:
            raise ValueError("rho0 (medium density) must be positive.")
        if c0 <= 0.0:
            raise ValueError("c0 (speed of sound) must be positive.")

        self.fs = float(fs)
        self.k = 1.0 / self.fs
        self.distance = float(distance)
        self.rho0 = float(rho0)
        self.c0 = float(c0)
        self.retarded = bool(retarded)

        # Monopole far-field gain: p_far = gain * Q''  (Q'' = volume acceleration).
        self.gain = self.rho0 / (4.0 * np.pi * self.distance)

        # Retardation: travel time r / c0, quantised to an integer number of samples so the delayed
        # signal is an *exact* copy (amplitude-preserving). The sub-sample residual is exposed.
        self.retardation_seconds = self.distance / self.c0
        delay_exact = self.retardation_seconds * self.fs
        self.latency_samples = int(round(delay_exact)) if self.retarded else 0
        # Signed sub-sample rounding error (samples): +ve => rounded down. Purely diagnostic.
        self.retardation_residual = float(delay_exact - self.latency_samples) if self.retarded \
            else 0.0

        # Integer-sample delay line: _buf[_idx] holds the sample from latency_samples steps ago.
        # Zero-filled, so the first latency_samples outputs are silence (the wavefront in transit).
        self._buf: NDArray[np.float64] = np.zeros(max(self.latency_samples, 0))
        self._idx = 0
        self.n = 0  # samples processed

    # -- processing ---------------------------------------------------------------------

    def process(self, volume_accel: float) -> float:
        """Map one volume-acceleration sample ``Q''`` to the far-field pressure ``p_far`` (Pa).

        Applies the monopole gain ``rho0 / (4 pi r)`` and (if ``retarded``) the integer-sample
        travel delay. During the first ``latency_samples`` calls the output is ``0`` — the wavefront
        has not yet reached the listener.
        """
        p = self.gain * float(volume_accel)
        self.n += 1
        if self.latency_samples == 0:
            return p
        out = self._buf[self._idx]        # sample stored latency_samples steps ago
        self._buf[self._idx] = p          # overwrite with the current sample
        self._idx = (self._idx + 1) % self._buf.size
        return float(out)

    def radiate(self, source: _VolumeAccelerationSource) -> float:
        """Convenience: read ``source.pressure()`` (its volume acceleration) and :meth:`process` it.

        ``source`` is any body/bridge exposing ``pressure()`` — :class:`ModalBody`, :class:`Plate`,
        or a :class:`StringBodyBridge` / :class:`StringPlateBridge`. Call once per timestep, *after*
        stepping the source, so it reads the freshest acceleration.
        """
        return self.process(source.pressure())

    def reset(self) -> None:
        """Clear the delay line (silence in transit) and the sample counter — reuse on a new run."""
        self._buf[:] = 0.0
        self._idx = 0
        self.n = 0


class RadiatedBody:
    """A :class:`ModalBody` loaded by its own radiation — the passive **back-reaction** (batch 2).

    Batch 1 read the body's monopole out "for free"; here the air pushes back. The radiation
    presents a resistance ``R`` (acoustic, Pa·s/m³) to the body's net **volume velocity**
    ``U = sum_i a_i q_i'`` (the same weights ``a_i`` that read the volume acceleration also couple
    the uniform back-pressure ``R U`` into each mode, by reciprocity), so the air removes power
    ``P_rad = R U^2`` and hands it to the far field. That lost energy is tracked in
    :attr:`radiated_energy` (``= integral P_rad dt``), giving the passivity identity

        E_body^n  +  integral_0^n P_rad dt  =  const      (lossless body, any R >= 0),

    the radiation analogue of the ``sigma``-damping energy channel — a closed-form check, not ears.
    ``R = 0`` is **bit-identical** to a bare :class:`ModalBody`.

    **Why this is a rank-1 *implicit* dashpot (and why that matters).** The load force on mode ``i``
    is ``-a_i R U`` with ``U`` the **centered** volume velocity ``a^T (q^{n+1} - q^{n-1}) / (2k)``,
    centered (implicit), exactly like the body's own ``2 sigma q'`` term, which is what makes the
    discrete dissipation ``k R U^2`` provably non-negative (an explicit one-sided ``U`` goes
    unstable at large ``R``). Because the coupling is a single scalar ``U``, the implicit solve
    collapses to one scalar (Sherman–Morrison): take the force-free step
    ``q~^{n+1} = ModalBody.step(force)``, read its free centered velocity ``U_free``, then

        U = U_free / (1 + R G) ,      G = (k/2) sum_i a_i^2 / (m_i (1 + sigma_i k)) ,

    and correct each mode ``q_i^{n+1} = q~_i - [k^2 a_i / (m_i (1 + sigma_i k))] R U``. The
    corrected state's centered volume velocity is exactly this ``U`` (``U_free - R U G = U``), so
    the energy increment ``k R U^2`` telescopes to machine precision. Since ``1 + R G >= 1`` for
    any ``R >= 0`` the solve is never singular: the load is **unconditionally** passive — no CFL, no
    stability guard (unlike the bridge springs, whose rank-1 block can go *negative*). Mirrors the
    bow's rank-1 scalar solve, but linear (no Newton).

    Being a drop-in for :class:`ModalBody` (it delegates every read accessor), it also slots
    straight into a :class:`~physsynth.core.connection.StringBodyBridge` as the body, so the full
    instrument chain conserves ``E_string + E_body + E_conn + integral P_rad`` with no edit to it.

    Parameters
    ----------
    body : ModalBody
        The radiating body to load. Its radiation weights ``body.a`` set the volume-velocity
        coupling. Use ``sigmas = 0`` (lossless modes) to isolate the radiation channel in the energy
        identity — then *all* the lost energy is the radiated energy.
    R : float
        Radiation resistance ``R`` (acoustic, Pa·s/m³), constant across frequency for this batch.
        ``R = 0`` decouples the air (bit-identical to the bare body). A representative value comes
        from :func:`monopole_radiation_resistance` (free space) or
        :func:`piston_radiation_resistance` (baffled) at a reference frequency. Must be ``>= 0``.

    Raises
    ------
    ValueError
        If ``R < 0``.
    """

    def __init__(self, *, body: ModalBody, R: float) -> None:
        if R < 0.0:
            raise ValueError("radiation resistance R must be >= 0.")
        self.body = body
        self.R = float(R)
        self.k = body.k
        # Rank-1 dashpot precomputes (all constant): G for the scalar Sherman-Morrison solve and the
        # per-mode correction prefactor k^2 a_i / (m_i (1 + sigma_i k)). The (1 + sigma_i k) carries
        # the body's implicit damping denominator into the load (dropping it drifts a lossy body).
        one_plus_sk = 1.0 + body.sigma * body.k
        self._G = 0.5 * body.k * float(np.sum(body.a * body.a / (body.m * one_plus_sk)))
        self._corr = body.k * body.k * body.a / (body.m * one_plus_sk)
        self.radiated_energy = 0.0  # integral of P_rad dt: the energy handed to the far field
        self.volume_velocity = 0.0  # last centered volume velocity U^n (diagnostic)
        self.n = 0

    def __getattr__(self, name: str):
        # Delegate read-only body accessors (phi, m, omega, M, q, q_prev, state, bridge_*, ...) so a
        # RadiatedBody is a drop-in wherever a bare ModalBody is expected (e.g. StringBodyBridge).
        # Only reached for names not set on the instance, so the overrides below always win.
        return getattr(self.body, name)

    # -- time stepping ------------------------------------------------------------------

    def step(self, force: float = 0.0) -> None:
        """Advance one step: force-free body advance, scalar volume-velocity solve, rank-1 correct.

        ``force`` is the optional external (bridge) force, forwarded to :meth:`ModalBody.step`; the
        radiation back-reaction is applied on top of it.
        """
        b = self.body
        q_nm1 = b.q_prev.copy()                        # q^{n-1}, before step() rolls history
        b.step(force)                                  # commit the force-free next state q~^{n+1}
        u_free = float(np.dot(b.a, b.q - q_nm1)) / (2.0 * self.k)  # free centered volume velocity
        u = u_free / (1.0 + self.R * self._G)          # Sherman-Morrison scalar solve
        b.q = b.q - (self.R * u) * self._corr          # rank-1 correction of q^{n+1}
        # Refresh q'' from the *corrected* second difference so pressure() carries the load (the
        # same reason ModalBody.pressure reads the true _accel rather than reconstructing it).
        b._accel = (b.q - 2.0 * b.q_prev + q_nm1) / (self.k * self.k)
        self.radiated_energy += self.k * self.R * u * u  # P_rad dt = k R U^2 >= 0
        self.volume_velocity = u
        self.n += 1

    # -- diagnostics --------------------------------------------------------------------

    def energy(self) -> float:
        """Total discrete energy ``E_body + integral P_rad dt`` (Joules).

        Conserved to machine precision for a lossless body (``sigmas = 0``, any ``R >= 0``) — the
        radiation channel :attr:`radiated_energy` exactly captures what the body sheds;
        monotonically decreasing if any body mode is itself lossy. Assert on this total, not
        ``body.energy()`` alone (which decreases as the body radiates)."""
        return self.body.energy() + self.radiated_energy

    def pressure(self) -> float:
        """Radiated pressure read-out ``sum_i a_i q_i''`` (monopole ∝ volume acceleration).

        Delegates to :meth:`ModalBody.pressure`, whose ``_accel`` this class refreshes *after* the
        rank-1 correction — so it reflects the radiation load, not just the force-free acceleration.
        """
        return self.body.pressure()

    def set_state(
        self,
        q0: NDArray[np.float64] | float,
        v0: NDArray[np.float64] | float = 0.0,
    ) -> None:
        """Set the body's initial modal state and reset the radiated-energy channel to zero."""
        self.body.set_state(q0, v0)
        self.radiated_energy = 0.0
        self.volume_velocity = 0.0
        self.n = 0

    def reset(self) -> None:
        """Zero the body state and the radiated-energy channel — reuse on a new run."""
        self.body.set_state(0.0)
        self.radiated_energy = 0.0
        self.volume_velocity = 0.0
        self.n = 0


class RationalAirLoad:
    """The air as a **first-order positive-real impedance** — resistance *and* radiation mass.

    Batch 2's :class:`RadiatedBody` loads a body with a single constant resistance ``R``; real air
    is frequency-dependent (``R_a(omega) = rho0 omega^2 / (4 pi c0)`` for a free-space monopole), so
    a constant over-damps the low modes and under-damps the high ones. The fix needs **no filter
    approximation**: the exact acoustic radiation impedance of a **pulsating sphere** of radius
    ``a`` is already a first-order rational function of ``j omega``,

        Z_a(j omega) = (rho0 c0 / S) * j k a / (1 + j k a) ,    S = 4 pi a^2 ,  k = omega / c0 ,

    which is exactly a resistance in **parallel** with an acoustic inertance:

        Z_a = R * j omega tau / (1 + j omega tau) ,
        R = rho0 c0 / S ,   M_a = rho0 / (4 pi a) ,   tau = M_a / R = a / c0 .

    ``M_a`` is the classic **radiation mass** — air dragged along but not radiated. The circuit
    reading *is* the numerical scheme: the volume velocity splits, ``U = U_R + U_L``, with
    ``p = R U_R = M_a dU_L/dt``. The resistor branch is radiated (lost to the far field), the
    inertance branch is stored (returned later). That split makes the frequency dependence
    **passive by construction** — the network is the proof, so there is no positive-realness check
    to pass and no stability guard to tune.

    Two closed-form anchors bracket it, and the first one is batch 2's own helper:

    ========== ==================================== =========================================
    limit      ``Re Z_a``                           anchor
    ========== ==================================== =========================================
    ``ka -> 0``  ``rho0 omega^2 / (4 pi c0)``        exactly :func:`monopole_radiation_resistance`
    ``ka -> oo`` ``rho0 c0 / S``                     plane-wave (fully loaded) saturation
    ========== ==================================== =========================================

    so this class does not supersede the batch-2 helper — it *contains* it as its low-frequency
    limit. ``Im Z_a`` peaks at ``ka = 1``.

    **Parameters are the effective coefficients** ``(R, M_a)``, independently, **not** a radius.
    Besides being the general first-order load (reusable for a fitted horn/piston termination), it
    is the only parameterisation that can reach ``M_a -> infinity`` at fixed ``R`` — the
    constant-resistance load, i.e. batch 2 exactly. A sphere ties ``M_a`` to ``R`` and cannot.
    :meth:`from_sphere` is the physically-consistent constructor for those who want the radius.

    **Discretisation: trapezoid on the inertance, which is the bilinear transform.** With
    ``L^- = U_L^{n-1/2}`` known,

        M_a (U_L^{n+1/2} - U_L^{n-1/2}) / k = p^n ,   U_L^n = (U_L^{n+1/2} + U_L^{n-1/2}) / 2 ,
        p^n = R (U^n - U_L^n)   ==>   p = R_eff (U - L^-) ,  R_eff = R / (1 + k R / (2 M_a)) .

    So on the body's side the whole batch is ``R -> R_eff`` plus a known offset ``L^-`` — batch 2's
    scalar Sherman-Morrison solve is untouched in shape. Because ``R_eff`` lies in ``[0, R]`` for
    every ``M_a > 0``, ``1 + R_eff G >= 1`` still holds: the solve is never singular, so the load is
    **unconditionally passive** at any ``R``, any ``M_a``, any ``k``. No CFL, no guard.

    The exact discrete impedance is ``Z_a`` evaluated at the **pre-warped** frequency
    ``s = (2j / k) tan(omega k / 2)`` (:meth:`impedance_discrete`), not at ``j omega`` — that is the
    bilinear transform's signature, and comparing a measured sweep to :meth:`impedance` instead
    shows a growing ``O((omega k)^2)`` error that looks like a scheme bug but is not.

    Parameters
    ----------
    fs : float
        Sample rate (Hz); timestep ``k = 1 / fs``. Must match the body's ``fs``.
    R : float
        Acoustic radiation resistance (Pa·s/m³ = kg/m⁴·s). ``0`` decouples the air entirely.
    M_a : float, optional
        Acoustic radiation mass / inertance (kg/m⁴), ``> 0``. Default ``inf`` — the
        constant-resistance load, **bit-identical** to :class:`RadiatedBody` (in IEEE arithmetic
        ``k R / (2 inf) = 0`` and ``p / inf = 0``, so ``R_eff = R`` exactly and the auxiliary state
        stays exactly zero).
    rho0, c0 : float, optional
        Ambient medium density and sound speed — used only for the equivalent-sphere geometry
        behind :meth:`far_field_pressure`, never in the time step.

    Raises
    ------
    ValueError
        If ``R < 0``, ``M_a <= 0``, or ``fs``/``rho0``/``c0`` are not positive.
    """

    def __init__(
        self,
        *,
        fs: float,
        R: float,
        M_a: float = np.inf,
        rho0: float = RHO0_AIR,
        c0: float = C0_AIR,
    ) -> None:
        if fs <= 0.0:
            raise ValueError("fs must be positive.")
        if R < 0.0:
            raise ValueError("radiation resistance R must be >= 0.")
        if not (M_a > 0.0):  # catches 0, negatives and NaN; +inf is allowed (= constant R)
            raise ValueError("radiation mass M_a must be positive (inf = constant-R load).")
        if rho0 <= 0.0 or c0 <= 0.0:
            raise ValueError("rho0 and c0 must be positive.")

        self.fs = float(fs)
        self.k = 1.0 / self.fs
        self.R = float(R)
        self.M_a = float(M_a)
        self.rho0 = float(rho0)
        self.c0 = float(c0)

        # The trapezoid's effective resistance. M_a = inf -> k R / (2 inf) = 0.0 -> R_eff = R
        # exactly (the batch-2 reduction, bit-for-bit).
        self.R_eff = self.R / (1.0 + self.k * self.R / (2.0 * self.M_a))
        # Relaxation time tau = M_a / R (inf if either the mass is infinite or R = 0).
        self.tau = np.inf if (self.R == 0.0 or np.isinf(self.M_a)) else self.M_a / self.R

        # Equivalent pulsating-sphere geometry, for the far-field read-out only. A general (R, M_a)
        # pair need not be sphere-consistent (that permissiveness is deliberate); the radius is
        # None unless 4 pi a_eq^2 == S_eq holds, and only far_field_pressure() insists on it.
        self.sphere_radius: float | None = None
        self.sphere_area: float | None = None
        if self.R > 0.0 and np.isfinite(self.M_a):
            a_eq = self.c0 * self.M_a / self.R          # from tau = a / c0
            s_eq = self.rho0 * self.c0 / self.R         # from R = rho0 c0 / S
            if abs(4.0 * np.pi * a_eq * a_eq - s_eq) <= 1e-9 * s_eq:
                self.sphere_radius = a_eq
                self.sphere_area = s_eq

        self.u_l = 0.0              # auxiliary state U_L^{n-1/2}: the inertance branch's velocity
        self.radiated_energy = 0.0  # integral of R U_R^2 dt — energy handed to the far field
        self.volume_velocity = 0.0  # last centered total U^n (diagnostic)
        self.pressure_load = 0.0    # last load pressure p^n (diagnostic; drives the far field)
        self.n = 0

    @classmethod
    def from_sphere(
        cls,
        *,
        fs: float,
        radius: float,
        rho0: float = RHO0_AIR,
        c0: float = C0_AIR,
    ) -> RationalAirLoad:
        """The physically consistent pulsating sphere of radius ``a``: the *exact* monopole load.

        ``R = rho0 c0 / (4 pi a^2)`` and ``M_a = rho0 / (4 pi a)``, so ``tau = a / c0`` and the
        impedance is exactly ``(rho0 c0 / S) j k a / (1 + j k a)``. Use this when you want physics;
        use the ``(R, M_a)`` constructor when you want to dial the two effects independently
        (including the ``M_a = inf`` reduction to batch 2, which no radius can express).
        """
        if radius <= 0.0:
            raise ValueError("sphere radius must be positive.")
        area = 4.0 * np.pi * radius * radius
        return cls(fs=fs, R=rho0 * c0 / area, M_a=rho0 / (4.0 * np.pi * radius), rho0=rho0, c0=c0)

    # -- the scalar solve ---------------------------------------------------------------

    def solve(self, u_free: float, G: float) -> tuple[float, float]:
        """Load pressure ``p^n`` and centered volume velocity ``U^n`` — *without* committing.

        ``u_free`` is the body's force-free centered volume velocity and ``G`` its scalar
        driving-point factor ``(k/2) sum_i a_i^2 / (m_i (1 + sigma_i k))``; the coupled scalar

            u* = (u_free - L^-) / (1 + R_eff G) ,   p = R_eff u* ,   U = u* + L^-

        is exact (the last identity because ``u*(1 + R_eff G) = u_free - L^-``). ``G = 0`` is the
        **prescribed-velocity drive** — a rigid piston moving at ``U = u_free`` regardless of the
        load — which is how the impedance oracle drives this object standalone.

        The operation order matters: forming ``u*`` first and multiplying by ``R_eff`` (rather than
        the algebraically equal alternatives) is what keeps ``M_a = inf`` bit-identical to
        :class:`RadiatedBody`.
        """
        u_star = (u_free - self.u_l) / (1.0 + self.R_eff * G)
        p = self.R_eff * u_star
        return p, u_star + self.u_l

    def commit(self, p: float, u: float) -> None:
        """Advance the auxiliary state on the accepted ``(p, U)`` and book the energy split.

        The inertance branch takes ``U_L`` (stored, returned later); the resistor branch takes
        ``U_R = U - U_L`` (radiated, gone). The trapezoid makes the stored increment exactly
        ``k p U_L^n`` and the dissipated one exactly ``k R U_R^2 >= 0``, and together they equal the
        ``k p U^n`` the body sheds — so the three-way identity telescopes to machine precision.
        """
        u_l_mid = self.u_l + 0.5 * self.k * p / self.M_a      # U_L^n (the trapezoid midpoint)
        u_r = u - u_l_mid                                     # what actually goes to the far field
        self.u_l = self.u_l + self.k * p / self.M_a           # U_L^{n+1/2}
        self.radiated_energy += self.k * self.R * u_r * u_r   # P_rad dt = k R U_R^2 >= 0
        self.volume_velocity = u
        self.pressure_load = p
        self.n += 1

    def step(self, u_free: float, G: float = 0.0) -> tuple[float, float]:
        """:meth:`solve` then :meth:`commit` — the standalone driven form (``G = 0`` = prescribed
        volume velocity ``u_free``). Returns ``(p, U)``."""
        p, u = self.solve(u_free, G)
        self.commit(p, u)
        return p, u

    # -- energy -------------------------------------------------------------------------

    def stored_energy(self) -> float:
        """Kinetic energy of the radiation mass, ``1/2 M_a (U_L^{n+1/2})^2`` (Joules).

        Zero for the constant-``R`` load (``M_a = inf``, where the auxiliary state is identically
        zero and the product ``inf * 0`` would otherwise be a NaN — special-cased here).
        """
        if np.isinf(self.M_a):
            return 0.0
        return 0.5 * self.M_a * self.u_l * self.u_l

    def energy(self) -> float:
        """The air's whole share: stored (radiation mass) + dissipated (radiated to the far field).

        Unlike batch 2's purely dissipative channel this air can also *give back*, so the stored
        term is genuinely new structure — a wrong reactance shows up as drift in the identity
        ``E_body + air.energy() = const``.
        """
        return self.stored_energy() + self.radiated_energy

    # -- closed-form oracles ------------------------------------------------------------

    def impedance(self, omega: float) -> complex:
        """Continuous acoustic impedance ``Z_a(j omega) = R j omega tau / (1 + j omega tau)``.

        The physics. ``Re Z -> rho0 omega^2 / (4 pi c0)`` as ``omega tau -> 0`` (agreeing with
        :func:`monopole_radiation_resistance` for a sphere-consistent load) and ``-> R`` as
        ``omega tau -> inf``. Constant-``R`` loads (``M_a = inf``) return ``R`` at every frequency —
        which is precisely batch 2's approximation, stated.
        """
        if np.isinf(self.tau):
            return complex(self.R if np.isinf(self.M_a) else 0.0)
        s = 1j * float(omega)
        return self.R * s * self.tau / (1.0 + s * self.tau)

    def impedance_discrete(self, omega: float) -> complex:
        """The **scheme's** impedance: ``Z_a`` at the pre-warped ``s = (2j / k) tan(omega k / 2)``.

        Trapezoid *is* the bilinear transform, so this — not :meth:`impedance` — is what a measured
        sweep matches to machine precision. The gap between the two is the honest discretisation
        error; it falls as ``O((omega k)^2)`` and vanishes as ``k -> 0``. Valid below Nyquist
        (``omega k < pi``), where the tangent is finite and positive.
        """
        if np.isinf(self.tau):
            return complex(self.R if np.isinf(self.M_a) else 0.0)
        # `math.tan`, deliberately, not `np.tan`. NumPy does not call the platform libm for the
        # transcendentals -- it has its own vectorised routines, dispatched by CPU feature -- so
        # `np.tan` here would make this read-out's last bit a property of the machine, and the
        # Rust port (which reaches libm) would match on one CI runner and not the next. That is
        # exactly what happened to `x ** alpha` in `collision`; see the plan's section 22. CPython
        # and Rust both reach the platform libm, so `math.tan` is the portable spelling. Nothing
        # is lost: `omega` is a scalar, so there was no vectorisation here to give up.
        s = 2j * math.tan(0.5 * float(omega) * self.k) / self.k
        return self.R * s * self.tau / (1.0 + s * self.tau)

    def loaded_mode(
        self,
        omega0: float,
        *,
        weight: float,
        mass: float,
        iterations: int = 50,
        tol: float = 1e-14,
    ) -> tuple[float, float]:
        """Closed-form ``(omega_eff, alpha)`` of one weakly loaded mode — **both** parts of ``Z_a``.

        A single mode ``m q'' + m omega0^2 q = -a p`` driven against this load, with
        ``p = Z_a U`` and ``U = a q'``, becomes (at ``q' = j omega q``)

            [m + a^2 Im Z_a(omega) / omega] q''  +  a^2 Re Z_a(omega) q'  +  m omega0^2 q  =  0 ,

        so the air does **two** things and the batch needs both to be right:

        * the **reactance** is an *added mass* ``m_add = a^2 Im Z_a / omega`` — the body gets
          heavier and its pitch drops (a constant-``R`` load cannot: its ``Im Z`` is zero);
        * the **resistance** damps it at ``alpha = a^2 Re Z_a / (2 m_eff)``, evaluated at the
          *shifted* frequency.

        Both depend on the frequency they shift, so this solves the fixed point
        ``omega_eff = omega0 sqrt(m / m_eff(omega_eff))`` by iteration — to ``tol``, and it
        **raises** rather than silently returning the last iterate if that is not reached.

        This is a weak-loading result: valid while ``alpha << omega``, with a residual second order
        in that ratio (~1% at ``alpha / omega ~ 1%``). It is a *modeling* oracle, independent of the
        time-stepping scheme — it reads only :meth:`impedance`.
        """
        if mass <= 0.0:
            raise ValueError("mass must be positive.")
        w0 = float(omega0)
        if w0 <= 0.0:
            raise ValueError("omega0 must be positive.")
        w = w0
        a2 = float(weight) * float(weight)
        for _ in range(int(iterations)):
            w_next = w0 * np.sqrt(mass / (mass + a2 * self.impedance(w).imag / w))
            if abs(w_next - w) <= tol * w_next:
                w = w_next
                break
            w = w_next
        else:
            raise ValueError(
                f"loaded_mode did not converge in {iterations} iterations (last relative step "
                f"{abs(w_next - w) / w_next:.3e} > tol {tol:.1e}); the added mass is comparable to "
                "the modal mass, which is outside this weak-loading formula's range."
            )
        z = self.impedance(w)
        m_eff = mass + a2 * z.imag / w
        return float(w), float(a2 * z.real / (2.0 * m_eff))

    def far_field_pressure(self, distance: float, p_load: float | None = None) -> float:
        """Far-field pressure at ``r`` from the sphere's own surface pressure: ``(a / r) p_load``.

        **Not** the batch-1 read-out, and the difference is a trap worth stating.
        :class:`AirRadiation` is the ``a -> 0`` compact-source limit ``rho0 Q'' / (4 pi r)``; a
        *finite* sphere additionally low-passes the far field by ``1 / (1 + j k a)``, so a naive
        power balance against batch 1 misses by ``1 / (1 + (ka)^2)`` — which reads as a bug and is
        not one. With this form the balance is **exact at every ``ka``**, because
        ``S |Z_a|^2 / (rho0 c0) == Re Z_a`` identically:

            4 pi r^2 <p_far^2> / (rho0 c0)  ==  <R U_R^2>   =   the booked radiated power.

        The pure travel delay ``(r - a) / c0`` is *not* applied (it is latency, not level — hand the
        result to :class:`AirRadiation` if you want the delay line). Requires a sphere-consistent
        load: a general ``(R, M_a)`` pair has no radius, and this read-out is the one place where
        that interpretation is genuinely needed, so it refuses rather than guessing.
        """
        if self.sphere_radius is None:
            raise ValueError(
                "far_field_pressure needs a sphere-consistent load (4 pi a^2 == rho0 c0 / R); "
                "build it with RationalAirLoad.from_sphere(...)."
            )
        if distance <= 0.0:
            raise ValueError("distance must be positive.")
        p = self.pressure_load if p_load is None else float(p_load)
        return self.sphere_radius / float(distance) * p

    def reset(self) -> None:
        """Zero the auxiliary state, the radiated channel and the counters — reuse on a new run."""
        self.u_l = 0.0
        self.radiated_energy = 0.0
        self.volume_velocity = 0.0
        self.pressure_load = 0.0
        self.n = 0


class ReactiveRadiatedBody:
    """A :class:`ModalBody` loaded by a **frequency-dependent** radiation impedance (batch 3).

    Batch 2's :class:`RadiatedBody` is the purely **resistive** load; what is new here is the
    **reactance** — the radiation mass — which is what turns one number into ``Z_a(omega)``. The
    audible consequence is the one a constant ``R`` cannot produce: high partials radiate better and
    die first, at the per-mode rate ``alpha_i = a_i^2 Re Z_a(omega_i) / (2 m_i)``.

    Structurally it is batch 2 with the load pressure generalised from ``p = R U`` to
    ``p = R_eff (U - L^-)`` (see :class:`RationalAirLoad`), so the same rank-1 scalar
    Sherman-Morrison solve carries it:

        q~^{n+1} = ModalBody.step(force)                    (force-free advance)
        U_free   = a^T (q~^{n+1} - q^{n-1}) / (2 k)         (its centered volume velocity)
        p, U     = load.solve(U_free, G) ,   G = (k/2) sum_i a_i^2 / (m_i (1 + sigma_i k))
        q^{n+1}  = q~^{n+1} - p * [k^2 a_i / (m_i (1 + sigma_i k))]

    and the energy identity gains a **stored** term alongside the dissipated one:

        E_body  +  1/2 M_a (U_L^{n+1/2})^2  +  integral R U_R^2 dt  =  const     (lossless modes).

    Passivity is unconditional (``1 + R_eff G >= 1`` for every ``R >= 0``, ``M_a > 0``, ``k``), so
    there is still no CFL and no guard. Two exact reductions pin it: ``R = 0`` is bit-identical to a
    bare :class:`ModalBody`, and ``M_a = inf`` is bit-identical to :class:`RadiatedBody` at the same
    ``R``. Like batch 2 it delegates every read accessor, so it drops straight into a
    :class:`~physsynth.core.connection.StringBodyBridge` as the body with no edit to the bridge.

    Parameters
    ----------
    body : ModalBody
        The radiating body. Its radiation weights ``body.a`` set the volume-velocity coupling; use
        ``sigmas = 0`` to isolate the air channel in the energy identity.
    load : RationalAirLoad
        The air impedance. Its ``fs`` must match the body's.

    Raises
    ------
    ValueError
        If the load's timestep differs from the body's.
    """

    def __init__(self, *, body: ModalBody, load: RationalAirLoad) -> None:
        if not np.isclose(load.k, body.k, rtol=1e-12, atol=0.0):
            raise ValueError(
                f"load fs ({load.fs}) must match the body's ({1.0 / body.k}) — the trapezoid's "
                "R_eff and the body's centered velocity share one timestep."
            )
        self.body = body
        self.load = load
        self.k = body.k
        one_plus_sk = 1.0 + body.sigma * body.k
        self._G = 0.5 * body.k * float(np.sum(body.a * body.a / (body.m * one_plus_sk)))
        self._corr = body.k * body.k * body.a / (body.m * one_plus_sk)
        self.n = 0

    def __getattr__(self, name: str):
        # Delegate read-only body accessors (phi, m, omega, M, q, q_prev, state, bridge_*, ...) so
        # this is a drop-in wherever a bare ModalBody is expected. Only reached for names not set on
        # the instance, so the overrides below always win.
        return getattr(self.body, name)

    # -- time stepping ------------------------------------------------------------------

    def step(self, force: float = 0.0) -> None:
        """Advance one step: force-free body advance, scalar load solve, rank-1 correction.

        ``force`` is the optional external (bridge) force, forwarded to :meth:`ModalBody.step`; the
        radiation back-reaction is applied on top of it.
        """
        b = self.body
        q_nm1 = b.q_prev.copy()                        # q^{n-1}, before step() rolls history
        b.step(force)                                  # commit the force-free next state q~^{n+1}
        u_free = float(np.dot(b.a, b.q - q_nm1)) / (2.0 * self.k)  # free centered volume velocity
        p, u = self.load.solve(u_free, self._G)        # scalar coupled solve (Sherman-Morrison)
        b.q = b.q - p * self._corr                     # rank-1 correction of q^{n+1}
        # Refresh q'' from the *corrected* second difference so pressure() carries the load.
        b._accel = (b.q - 2.0 * b.q_prev + q_nm1) / (self.k * self.k)
        self.load.commit(p, u)
        self.n += 1

    # -- diagnostics --------------------------------------------------------------------

    @property
    def radiated_energy(self) -> float:
        """Energy handed to the far field, ``integral R U_R^2 dt`` (the load's dissipated share)."""
        return self.load.radiated_energy

    @property
    def volume_velocity(self) -> float:
        """Last centered total volume velocity ``U^n`` (diagnostic)."""
        return self.load.volume_velocity

    def energy(self) -> float:
        """Total discrete energy ``E_body + E_air`` (Joules), the air term being *stored plus
        radiated*.

        Conserved to machine precision for a lossless body (``sigmas = 0``, any ``R``, any ``M_a``);
        monotonically decreasing if any body mode is itself lossy. Assert on this total — neither
        ``body.energy()`` (which the air both takes from and returns to) nor the radiated channel
        alone is conserved.
        """
        return self.body.energy() + self.load.energy()

    def pressure(self) -> float:
        """Monopole read-out ``sum_i a_i q_i''`` carrying the load (cf. :meth:`ModalBody.pressure`).

        This is the compact-source (``a -> 0``) volume acceleration that :class:`AirRadiation`
        expects. For the *finite* sphere's own far field — the one whose power balances the booked
        radiated energy exactly — use :meth:`RationalAirLoad.far_field_pressure`.
        """
        return self.body.pressure()

    def far_field_pressure(self, distance: float) -> float:
        """Far-field pressure at ``r`` from the finite sphere: see
        :meth:`RationalAirLoad.far_field_pressure`."""
        return self.load.far_field_pressure(distance)

    def set_state(
        self,
        q0: NDArray[np.float64] | float,
        v0: NDArray[np.float64] | float = 0.0,
    ) -> None:
        """Set the body's initial modal state and reset the air (auxiliary state + channels)."""
        self.body.set_state(q0, v0)
        self.load.reset()
        self.n = 0

    def reset(self) -> None:
        """Zero the body state and the air's auxiliary state and channels — reuse on a new run."""
        self.body.set_state(0.0)
        self.load.reset()
        self.n = 0


# -- Rust swap (docs/dev/rust-migration-plan.md) ----------------------------------------------
#
# Phase 2 batch 4. `PHYSSYNTH_RS=1` replaces the four classes and the monopole helper with the
# `physsynth_rs` implementations; every test in `tests/test_radiation.py` runs unmodified against
# them, and so does the whole body/bridge leg, because `web/serialize.py` hands a `RadiatedBody`
# to `StringBodyBridge` as the body.
#
# Two things to know before editing either side.
#
# **`piston_radiation_resistance` was NOT swapped for four phases, and now is (2026-09-03).**
# It is the only name here that needs a Bessel J1, `scipy.special.j1` is Cephes, and reproducing it
# was a special-function problem rather than a load-batch one -- so the module ported in halves,
# exactly as `operators2d` did for a solver the membrane never called. Phase 7 batch 2 wrote the
# Bessel routine and the half closes here.
#
# Three things about that swap, because the arrangement reads oddly until they are said:
#
# * **The implementation comes from `physsynth-analysis`, not `physsynth-core`.** The core crate's
#   dependency list must stay empty (plan section 2.2), so it cannot reach a Bessel function; and
#   `modal.rs` reaches core's Brent transcription in the other direction, so a core->analysis edge
#   would also be a cycle. `physsynth-py` depends on both, which is what lets a core-flagged name
#   have an analysis-crate implementation.
# * **It still swaps on `PHYSSYNTH_RS`, not `PHYSSYNTH_RS_ANALYSIS`.** `CLAUDE.md`'s rule is that
#   no `core/` module ever reads the analysis flag, and this is a `core/` module. The crate a
#   function lives in and the flag its name is swapped by are separate questions.
# * **What makes that safe is a measurement, not the rule.** `test_bore_radiation.py` uses this
#   resistance to check a bore, so under the model flag a Rust bore would be checked against a
#   Rust-computed R -- the shared-misreading shape. But the same flagged run executes
#   `test_radiation.py::test_piston_resistance_matches_bessel_formula_away_from_the_limit`, which
#   builds its expectation from `scipy.special.j1` inside the test body at `rel = 1e-12`. The ruler
#   is checked against an unmoved reference in the run that uses it. Observed there: 7.9e-16.
#
# **A defect this function has always had, found while porting and deliberately reproduced.** The
# `ka < 1e-8` branch below switches to the series because `1 - J1(2ka)/ka` is a 0/0 -- but just
# *above* that threshold the direct form subtracts two numbers agreeing to sixteen digits, and with
# SciPy's own `j1` doing the work it is 544% wrong at ka = 1e-8, 2.3% at 1e-7, and only reaches
# 1e-6 accuracy around ka = 1e-5. The threshold is about three decades too small. The port
# reproduces it rather than fixing it, because changing a shipped physics number inside a porting
# batch is not a port; it is registered in `docs/dev/scientific-hurdles.md` section 14 with the
# proposed threshold, for the human's call. No caller is in the band -- the suite's two real call
# sites are at ka = 9.2e-5 and ka = 1.83, where the two implementations agree to 5.3e-8 and 0.0.
#
# **The state is NOT bit-identical under the flag, and that is the first time in this migration.**
# `RadiatedBody.step` reads `np.dot(b.a, b.q - q_nm1)`, and unlike `body.pressure()` -- the same
# reduction, since batch 2 -- that number decides `q^{n+1}` rather than being read out. OpenBLAS
# fuses the multiply-add and vectorises past sixteen terms, and which kernel it picks is a
# property of the CPU, so Rust sums plainly and the agreement bar is the plan's Group A (1e-13)
# rather than zero. `tests/test_rust_parity_radiation.py` measures what it actually is. Nothing in
# the physics harness moves: the energy identities are exact in either arithmetic.
#
# Off by default. The Python model is still the reference oracle for every model not yet ported.
AirRadiationPy = AirRadiation
RadiatedBodyPy = RadiatedBody
RationalAirLoadPy = RationalAirLoad
ReactiveRadiatedBodyPy = ReactiveRadiatedBody
monopole_radiation_resistance_py = monopole_radiation_resistance
piston_radiation_resistance_py = piston_radiation_resistance
"""The pure-Python reference implementations, under names the swap below never rebinds."""

_USE_RUST = os.environ.get("PHYSSYNTH_RS", "").strip() not in ("", "0", "false", "False")

if _USE_RUST:  # pragma: no cover - exercised by the dedicated CI job, not the default gate
    from physsynth_rs import (  # type: ignore[assignment]  # noqa: F811
        AirRadiation,
        RadiatedBody,
        RationalAirLoad,
        ReactiveRadiatedBody,
        monopole_radiation_resistance,
        piston_radiation_resistance,
    )
