"""String↔barrier collision — the distributed contact model (fret buzz, jawari; model #8).

**The implementation is Rust**: ``crates/physsynth-core/src/collision.rs``, bound in
``crates/physsynth-py`` and re-exported here (``docs/dev/rust-migration-plan.md`` §39, unit 1).
That module's header is now the only copy of the physics — the vector Newton solve with an Armijo
line search, the static-equilibrium magnitude oracle, and the contact primitives that were promoted
here when the barrier became their second consumer (the mallet was the first, and re-exports them
back).

One function is **not** a bare re-export and cannot be, which is why this file is 60 lines rather
than 25. ``solve_contact_vector`` keeps a Python wrapper for two reasons, both deliberate and both
recorded when the model was ported:

* ``np.ascontiguousarray`` is a no-op returning the *same object* for an already-C-contiguous
  float64 array, which every call site's is — so this is a type check, not the second pass over the
  data that a previous batch measured a port's speed away on.
* the non-convergence warning is raised **from this frame**, because ``stacklevel=2`` has to keep
  meaning what it meant before, which it cannot if it is issued from inside an extension module.

Three private helpers that used to sit here — ``_contact_force_dg_deriv``, ``_contact_force_hyst``,
``_contact_force_hyst_deriv`` — went with the body. They had no Rust twin and, measured, no caller
anywhere outside the class that is now gone.

Headless: no I/O, no graphics.
"""

from __future__ import annotations

import numpy as np
import physsynth_rs as _rs
from numpy.typing import NDArray
from physsynth_rs import (
    BarrierString,
    contact_force_dg,
    contact_force_elastic,
    contact_force_total,
    contact_potential,
    contact_stiffness,
    solve_contact,
)

from .string_damped import DampedStiffString

# Reached through the private spelling by the model and by the mallet: the alias namespace was flat
# and this module's is not (a `<name>_py` alias could not carry the leading underscore), which is
# the shape that once made a whole module fall out of a derived guard.
_contact_force_total_deriv = _rs.contact_force_total_deriv
_force_total_vec = _rs.force_total_vec
_deriv_total_vec = _rs.deriv_total_vec

__all__ = [
    "BarrierString",
    "DampedStiffString",
    "contact_potential",
    "contact_force_elastic",
    "contact_stiffness",
    "contact_force_dg",
    "contact_force_total",
    "solve_contact",
    "solve_contact_vector",
]


def solve_contact_vector(
    eta_free: NDArray[np.float64],
    eta_prev: NDArray[np.float64],
    G: NDArray[np.float64],
    K: float,
    alpha: float,
    lam_h: float,
    k: float,
    *,
    tol: float,
    seed: NDArray[np.float64],
    newton_tol: float = 1e-13,
    maxiter: int = 60,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """The vector contact solve. See the module docstring for why this wrapper exists."""
    eta, f, iters, residual, converged = _rs.solve_contact_vector(
        np.ascontiguousarray(eta_free, dtype=float),
        np.ascontiguousarray(eta_prev, dtype=float),
        np.ascontiguousarray(G, dtype=float),
        K, alpha, lam_h, k,
        tol=tol,
        seed=np.ascontiguousarray(seed, dtype=float),
        newton_tol=newton_tol,
        maxiter=maxiter,
    )
    if not converged:
        import warnings

        warnings.warn(
            f"vector contact solve did not converge in {maxiter} iterations "
            f"(residual {residual:.2e} > {newton_tol:.1e}); energy may drift. Raise "
            f"newton_maxiter or oversample the contact.",
            stacklevel=2,
        )
    return eta, f, iters
