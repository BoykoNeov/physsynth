"""The air node in three tiers — radiation as an output transform, a load, and a network.

**The implementations are Rust**: ``crates/physsynth-core/src/radiation.rs``, bound in
``crates/physsynth-py`` and re-exported here (``docs/dev/rust-migration-plan.md`` §39, unit 2).
That module's header is now the only copy of the physics — the three tiers, and in particular that
``RationalAirLoad`` is an **exact rational impedance** whose network *is* the passivity proof, so
air stores as well as dissipates. Note that batch 2's ``R`` and batch 3's ``R`` are different
quantities; the Rust header says which is which.

What stays here is the module's **types and measured constants**:

* ``RHO0_AIR`` / ``C0_AIR`` — two floats, and (with ``bore``'s pair) the whole of the viewer's
  remaining seam per §39.1.
* ``PISTON_SERIES_CUTOFF_KA`` — a *measured* choice, not a round number, and its docstring carries
  the measurement. It is read by ``tests/test_rust_parity_analysis.py``, which is why it is public.
* ``_VolumeAccelerationSource`` — a `Protocol`. ``AirRadiation.radiate`` duck-types on it, and a
  Protocol has no runtime implementation to port.

``piston_radiation_resistance`` is implemented in ``crates/physsynth-analysis`` rather than in the
core crate — the core crate's dependency list must stay empty so it cannot reach a Bessel function
— while its *name* is swapped here because it lives in a ``core/`` module. The crate a function is
implemented in and the flag its name is swapped by are separate questions (§37.7).

Headless: no I/O, no graphics.
"""

from __future__ import annotations

from typing import Protocol

from physsynth_rs import (
    AirRadiation,
    RadiatedBody,
    RationalAirLoad,
    ReactiveRadiatedBody,
    monopole_radiation_resistance,
    piston_radiation_resistance,
)

from .body import ModalBody

# Air at ~20 °C, 1 atm. Defaults; override for other media (helium, water, altitude).
RHO0_AIR = 1.2041  # kg/m^3, ambient air density
C0_AIR = 343.0     # m/s, speed of sound in air

PISTON_SERIES_CUTOFF_KA = 3e-2
"""Below this ``ka``, :func:`piston_radiation_resistance` uses its series rather than ``J1``.

Chosen by measurement against a 60-digit reference (hurdles §14): with the three-term series the
whole function's worst relative error over ``ka`` in ``[1e-10, 10]`` is 6.7e-13 at this cutoff,
against 7.9e-13 at ``2e-2``, 2.8e-12 at ``4e-2`` and 5.0e-12 at ``1e-2``. It sits deliberately
*past* the direct form's own noisy region (~5e-12 around ``ka = 3e-3 .. 1e-2``) rather than at the
crossover, which is what buys the last order of magnitude.
"""


class _VolumeAccelerationSource(Protocol):
    """Anything with a monopole read-out ``pressure()`` = net volume acceleration ``Q''``.

    :class:`physsynth.core.body.ModalBody`, :class:`physsynth.core.plate.Plate`, and the bridge
    connections all satisfy this — :meth:`AirRadiation.radiate` duck-types on it.
    """

    def pressure(self) -> float: ...


__all__ = [
    "AirRadiation",
    "C0_AIR",
    "ModalBody",
    "PISTON_SERIES_CUTOFF_KA",
    "RHO0_AIR",
    "RadiatedBody",
    "RationalAirLoad",
    "ReactiveRadiatedBody",
    "monopole_radiation_resistance",
    "piston_radiation_resistance",
]
