"""The rotating-wave BVP — a helix that is a relative equilibrium of the geometric string.

**The implementation is Rust.** ``crates/physsynth-analysis/src/rotating_wave.rs``, re-exported
through the wrappers below (``docs/dev/rust-migration-plan.md`` §44, unit 10). The residual, the
Jacobian, the continuation and the Newton solve all live there; what stays here is the result type
(a ``NamedTuple`` with no runtime implementation), the three measured defaults with their
reasoning, and the non-convergence warning — which must be raised **from this frame**, because
``stacklevel=2`` cannot mean what it means from inside an extension module and
``tests/test_geometric_rotating_wave.py`` matches its text.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
import physsynth_rs as _rs
from numpy.typing import NDArray

__all__ = [
    "RotatingWave",
    "solve_rotating_wave",
    "rotating_wave_history",
    "planar_hessian_cells",
    "kc_circular_frequency",
]

CONTINUATION_STEPS_DEFAULT = 8
"""Amplitude-continuation steps from the linear mode up to the requested amplitude.

The ``R -> 0`` seed (a sine at the theta-scheme's linear frequency) is exact only in the limit, and
the mode shape deforms as ``R`` grows. Stepping the amplitude and reusing each solution as the next
seed is what keeps Newton in its basin at amplitudes where the deformation is large.
"""

NEWTON_TOL_DEFAULT = 1e-14
"""**Relative** convergence bar on the Newton *step* (see :func:`solve_rotating_wave`).

Relative, and on the step rather than the residual, because the residual's three blocks carry
*different units* (transverse force density, longitudinal force density, and a displacement), so a
single max-norm bar on it would be a unit-mixing accident. The step is compared against the
solution's own scale, which is unit-free per block and is what Newton's quadratic convergence makes
a proxy for the error.
"""

NEWTON_MAXITER_DEFAULT = 50
"""Cap on Newton iterations per continuation step. Reaching it warns; it never silently renders."""


class RotatingWave(NamedTuple):
    """A converged rotating-wave relative equilibrium: the helix, its stretch, its frequency."""

    phi: NDArray[np.float64]
    """Transverse profile on the **full grid** ``(N+1,)`` (m), zero at both clamped ends.

    The helix is ``u = phi cos(Omega t)``, ``w = phi sin(Omega t)``. A *deformed* sine -- see
    :attr:`shape_residual`.
    """
    psi: NDArray[np.float64]
    """Longitudinal profile on the **full grid** ``(N+1,)`` (m), zero at both clamped ends.

    **Static, and nonzero.** The helix holds a fixed longitudinal stretch against the frozen
    transverse load. Do not test the rotating wave by asserting ``v == 0``; assert that ``v`` does
    not *move* (zero longitudinal kinetic energy).
    """
    Omega: float
    """Angular frequency (rad/s) of the rotation."""
    frequency: float
    """``Omega / (2 pi)`` (Hz)."""
    s: float
    """The scheme's ``s = (4/k^2) sin^2(Omega k / 2)`` (rad^2/s^2) -- the unknown solved for.

    Equals ``Omega^2`` exactly when ``time_discrete`` is false. Both time-discrete factors are
    linear in this variable, which is why it, and not ``Omega``, is the natural unknown.
    """
    amplitude: float
    """The imposed modal amplitude ``R`` (m) -- ``phi``'s projection onto ``sin(m pi x / L)``."""
    mode: int
    """The mode number ``m`` continued up from."""
    stretch_ratio: NDArray[np.float64]
    """Per-cell ``Lambda`` ``(N,)`` -- the frozen stretch field. ``1`` at rest."""
    tension: NDArray[np.float64]
    """Per-cell axial tension ``EA Lambda - a`` (N) ``(N,)`` -- **non-uniform**, which is the point.

    A rigid helix is stretched most near the nodes, where ``phi'`` is largest. That non-uniformity
    is exactly what a Kirchhoff-Carrier tension (a single scalar) cannot represent, and it is the
    mechanism behind :attr:`shape_residual` and Tier C/8's breakage.
    """
    shape_residual: float
    """``max|phi - R sin(m pi x / L)| / max|phi|`` -- the deviation from a sine.

    **The physics payoff of this module.** ``-> 0`` with amplitude; grows as the helix stretches
    non-uniformly. This is *why* :func:`kc_circular_frequency` is wrong, and reporting it turns Tier
    C/8's frequency residual from a bare number into a mechanism.
    """
    iterations: int
    """Total Newton iterations summed over the continuation."""
    converged: bool
    """Whether every continuation step converged. Never render a false one as physics."""
    time_discrete: bool
    """Whether the scheme's ``(1 - theta k^2 s)`` and ``cos(Omega k)`` factors were included."""


# The Python bodies are gone and this module delegates. Three things follow, and they are the same
# three that follow for every deleted module in this project:
#
#   * `physsynth_rs` is a hard requirement to import this module, and therefore to collect most of
#     the suite. There is no `PHYSSYNTH_RS_ANALYSIS` switch any more: the flag chose between two
#     implementations and there is one.
#   * what these wrappers do beyond delegating is real work, not ceremony -- coercing whatever a
#     caller passed into the contiguous float64 arrays and Python ints the binding requires. That
#     is why this is a shim and not a row of re-exports, the way `airbox.py`'s wrapper tier and
#     `operators2d.py` are shims for their own reasons.
#   * what the Python implementation *said* is not lost. `tests/analysis_frozen_values.py` holds
#     62 fixtures' worth of its answers, recorded to the last digit before it was deleted, and
#     `tests/test_analysis_frozen.py` asserts them on every run. That is the human's condition on
#     this deletion (plan §44) and it is the only remaining check of these oracles against a second
#     implementation.


def _asarray(a: object) -> NDArray[np.float64]:
    """Whatever the caller passed, as the contiguous float64 array the binding requires."""
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64))


def solve_rotating_wave(
    *,
    L,
    T,
    rho,
    EA,
    fs,
    N,
    theta,
    amplitude,
    mode=1,
    kappa=0.0,
    time_discrete=True,
    continuation_steps=CONTINUATION_STEPS_DEFAULT,
    tol=NEWTON_TOL_DEFAULT,
    maxiter=NEWTON_MAXITER_DEFAULT,
):
    phi, psi, stretch_ratio, tension, scalars = _rs.rotating_wave_solve(
        L,
        T,
        rho,
        EA,
        fs,
        int(N),
        theta,
        amplitude,
        int(mode),
        kappa,
        bool(time_discrete),
        int(continuation_steps),
        tol,
        int(maxiter),
    )
    (
        omega,
        frequency,
        s,
        amp,
        mode_out,
        shape_residual,
        iterations,
        converged,
        td,
        failed_step,
        failed_amplitude,
    ) = scalars
    if not converged:
        # Raised here rather than in Rust: the binding has no business owning a Python warning
        # category, and the message has to match character for character because
        # `test_geometric_rotating_wave.py` matches on it.
        warnings.warn(
            f"Rotating-wave BVP did not converge at continuation step {failed_step}/"
            f"{continuation_steps} (amplitude {failed_amplitude:.3e} m) in {maxiter} Newton "
            f"iterations. The helix is not a relative equilibrium; seeding a resonator with it "
            f"will NOT rotate rigidly. Raise continuation_steps, or lower the amplitude.",
            RuntimeWarning,
            stacklevel=2,
        )
    return RotatingWave(
        phi=phi,
        psi=psi,
        Omega=omega,
        frequency=frequency,
        s=s,
        amplitude=amp,
        mode=int(mode_out),
        stretch_ratio=stretch_ratio,
        tension=tension,
        shape_residual=shape_residual,
        iterations=int(iterations),
        converged=bool(converged),
        time_discrete=bool(td),
    )


def rotating_wave_history(wave, *, fs):
    return _rs.rotating_wave_history(_asarray(wave.phi), _asarray(wave.psi), wave.Omega, fs)


def planar_hessian_cells(p, z, a):
    return _rs.rotating_wave_planar_hessian_cells(_asarray(p), _asarray(z), a)


def kc_circular_frequency(*, omega0_sq, eps, amplitude):
    return _rs.rotating_wave_kc_circular_frequency(omega0_sq, eps, amplitude)
