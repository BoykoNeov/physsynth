"""Tension-modulated string — Kirchhoff–Carrier, the string family's nonlinearity (model #9).

**The implementation is Rust**: ``crates/physsynth-core/src/string_nonlinear.rs``, bound in
``crates/physsynth-py`` and re-exported here (``docs/dev/rust-migration-plan.md`` §39, unit 1).
That module's header is now the only copy of the physics — the tension that rises with the string's
own stretch, the *exact* Duffing oracle it is validated against, and the finding that single-mode
motion is parametrically unstable above ``ΔT/T₀ ≈ 3`` (real physics, not a scheme artefact). Energy
is a **structural** bar here and spectral purity a **dynamical** one; they fail differently.

Two things stay because they have no Rust twin and real callers:
``StringCoefficients`` / ``string_coefficients_from_material`` are a **modelling oracle**, not part
of the model. The core deliberately takes effective coefficients ``(T, rho, kappa, EA)`` that are
mutually unconstrained — a string may be given steel's bending stiffness and rubber's axial
stiffness, which is a feature (HANDOFF §12.J) and also the honest surface for wound strings. This
helper *offers* consistency with a real material; it never imposes it, and it is Python arithmetic
over four scalars with nothing to gain from being compiled.

Headless: no I/O, no graphics.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np
from physsynth_rs import TensionModulatedString

from .string_stiff import THETA_DEFAULT

Boundary = Literal["supported"]

TENSION_TOL_DEFAULT = 1e-13
"""Relative tolerance on the tension root-find. Read by ``tests/helpers.py``."""

MAX_BRACKET_EXPANSIONS = 40
"""How far the tension bracket may be widened before the solve gives up."""


class StringCoefficients(NamedTuple):
    """A physically consistent coefficient set for a plain (unwound) cylindrical string."""

    rho: float
    """Linear density (kg/m) = rho_v * pi r^2."""
    kappa: float
    """Stiffness sqrt(E I_area / rho) (m^2/s), I_area = pi r^4 / 4."""
    EA: float
    """Axial stiffness (N) = E pi r^2."""
    c: float
    """Transverse wave speed sqrt(T/rho) (m/s)."""
    c_long: float
    """Longitudinal wave speed sqrt(E/rho_v) (m/s)."""
    EA_over_T: float
    """The governing nonlinearity ratio EA/T0 = (c_long/c)^2 -- radius-independent."""


def string_coefficients_from_material(
    *, E: float, radius: float, rho_v: float, T: float
) -> StringCoefficients:
    """Derive a **consistent** ``(rho, kappa, EA)`` from real material + geometry.

    A *modeling oracle*, not a constraint (cf. radiation's ``R_a`` helpers). The core deliberately
    takes **effective coefficients** ``(T, rho, kappa, EA)`` that are mutually unconstrained -- so a
    string can be given steel's bending stiffness and rubber's axial stiffness. That is a *feature*
    (HANDOFF section 12.J, hyperreal instruments: physics beyond real materials), and it is also the
    honest surface for **wound** strings (a steel core under a bronze overwind has no single ``E``,
    ``radius``, or ``rho_v``; the literature characterizes it by exactly these effective
    coefficients). This helper *offers* realism; it never imposes it.

    Parameters
    ----------
    E : Young's modulus (Pa). radius : string radius (m). rho_v : volumetric density (kg/m^3).
    T : rest tension (N) -- sets the transverse wave speed, hence the ratio below.

    Notes
    -----
    The nonlinearity's governing ratio is **radius-independent**::

        EA/T0 = E pi r^2 / (rho_v pi r^2 c^2) = E / (rho_v c^2) = (c_long / c)^2

    The radius cancels exactly: hardening is set by the ratio of **longitudinal to transverse wave
    speed**. Steel (``c_long ~ 5000 m/s``) at musical ``c ~ 200-400 m/s`` gives ``EA/T0 ~ 150-600``.
    That single number -- not the radius -- predicts whether a string will audibly glide.
    """
    if min(E, radius, rho_v, T) <= 0:
        raise ValueError("E, radius, rho_v, T must all be positive.")
    area = np.pi * radius**2
    second_moment = np.pi * radius**4 / 4.0
    rho = rho_v * area
    c = float(np.sqrt(T / rho))
    return StringCoefficients(
        rho=float(rho),
        kappa=float(np.sqrt(E * second_moment / rho)),
        EA=float(E * area),
        c=c,
        c_long=float(np.sqrt(E / rho_v)),
        EA_over_T=float(E * area / T),
    )


__all__ = [
    "Boundary",
    "MAX_BRACKET_EXPANSIONS",
    "StringCoefficients",
    "TENSION_TOL_DEFAULT",
    "THETA_DEFAULT",
    "TensionModulatedString",
    "string_coefficients_from_material",
]
