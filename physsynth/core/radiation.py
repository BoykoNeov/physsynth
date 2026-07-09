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
listener). No new state on the body and **no back-reaction**: this batch is a pure output transform,
so the body radiates "for free" and its own energy balance is untouched. The passive **radiation
load** — the air damping the body, with radiated power ``P_rad = R_rad U²`` accounted as an energy
channel and a closed-form baffled-piston ``R_rad(ka)`` oracle — is the next batch; until then this
node deliberately has no ``energy()`` (it stores and dissipates none).

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

# Air at ~20 °C, 1 atm. Defaults; override for other media (helium, water, altitude).
RHO0_AIR = 1.2041  # kg/m^3, ambient air density
C0_AIR = 343.0     # m/s, speed of sound in air


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
