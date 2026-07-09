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
