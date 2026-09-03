"""Geometrically-exact string — two polarizations plus longitudinal motion (model #10).

**The implementation is Rust**: ``crates/physsynth-core/src/string_geometric.rs``, bound in
``crates/physsynth-py`` and re-exported here (``docs/dev/rust-migration-plan.md`` §39, unit 1).
That module's header is now the only copy of the physics — the exact geometry that discharges model
#9's two refusals (phantom partials and whirling), the fact that ``EA = T`` reduces the ``u``
polarization **bit-for-bit** to :class:`~physsynth.core.string_damped.DampedStiffString`, and the
Mathieu tongue that whirling lives in.

Three things stay, and none of them is a shim:

* the three **measured** constants below — their docstrings are the record of the measurements that
  set them, and ``LAM_LONG_WARN``'s in particular is the longest single piece of reasoning in this
  package. ``LAM_LONG_WARN`` is read by ``web/serialize.py``, ``tests/test_geometric_energy.py`` and
  ``scripts/sweep_geometric_lam_long.py``; ``NEWTON_MAXITER_DEFAULT`` by
  ``physsynth/analysis/rotating_wave.py``.
* ``GeometricState``, and this one is load-bearing in a direction the migration did not expect:
  **the Rust class constructs it.** ``crates/physsynth-py/src/string_geometric.rs`` does
  ``py.import("physsynth.core.string_geometric").getattr("GeometricState")`` to build what
  ``GeometricString.state`` returns, so the binding depends on this module rather than the other way
  round (plan §41.2, findings #39). Deleting it would break the Rust model, not a Python caller.

Headless: no I/O, no graphics.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray
from physsynth_rs import GeometricString

from .string_stiff import THETA_DEFAULT

Boundary = Literal["supported"]

NEWTON_TOL_DEFAULT = 1e-15
"""**Relative** tolerance on the max-norm of the update residual (see :meth:`GeometricString.\
_solve_newton`).

Unit-free: the bar is ``newton_tol * max|Y_seed|``, following model #9's normalized tension solve
("so ``tension_tol`` is a unit-free relative bar"). An **absolute** bar here is a trap -- the
residual scales with the displacement, so a fixed ``1e-14`` is ``1e-11`` relative on a 1 mm string
and lands the drift at ``2e-10``, *over* the ``1e-10`` gate, while looking like a tight number.
Measured: drift is proportional to this over five decades.
"""

NEWTON_MAXITER_DEFAULT = 60
"""Cap on damped-Newton iterations per step. Reaching it warns; it never silently renders."""

LAM_LONG_WARN = 1.0
"""Warn when ``lam_long = c_long k / h`` exceeds this. **The one guard with no CFL behind it.**

Every other explicit scheme in this package rejects ``lam > 1`` because the scheme is *unstable*
above it. This one is an **accuracy** bar on an unconditionally stable scheme, which is exactly what
makes it dangerous: nothing throws, nothing violates, and the model quietly returns nonsense.

**There are two edges here, not one, and they are a factor of two apart.** Re-measured 2026-09-03
by ``scripts/sweep_geometric_lam_long.py`` over nine ``(N, amplitude, IC)`` cells -- plucked and
mode-3, ``N`` 16 to 32, amplitudes to 1e-2 -- with the Newton iteration counter read beside the
drift rather than the drift alone::

    lam_long <= 2    conserves ~1e-13 .. 1e-15, no step exhausts newton_maxiter
    lam_long  = 4    the CONVERGENCE edge: steps begin to exhaust newton_maxiter (7 of 9 cells)
    lam_long  5-10   the ENERGY edge, case-dependent: drift breaks 1e-10 and runs to 1e+5

Between the two the solve stalls on up to a fifth of its steps and the energy *still* conserves to
~1e-15 -- so in that band the model's primary bug detector is blind, and blind in the safe-looking
direction (``docs/dev/scientific-hurdles.md`` section 8 is the other place in this project where a
flat energy is not a stability certificate). An earlier version of this table gave a single row per
``lam_long`` and so read as one threshold; its ``= 4`` was the convergence edge and its ``1e+3``
drift was the energy edge, two cells apart.

The bar sits at 1 rather than the measured-safe 2 to mirror this project's "tune toward lambda = 1"
rule, and to keep 4x of margin on the *convergence* edge -- which, now that the two are separated,
is what the 4x is margin against. It **warns rather than rejects**:
``lam_long = 2`` demonstrably conserves, so a hard bar would forbid working configurations -- and
the regime above is worth being able to *study*, just not to trust. Because
``c_long/c = sqrt(EA/T0) ~ 22`` at realistic stiffness, the familiar transverse ``lam = 0.5`` lands
at ``lam_long ~ 11``: this warning will fire
on the parameters a reader of models #1-#9 would reach for first. That is its whole purpose.

**Not raised at ``EA == T0``** (``a == 0``), regardless of ``lam_long``. There the three fields
decouple and the model *is* :class:`~physsynth.core.string_damped.DampedStiffString` three times
over -- which does not warn about its own ``lambda`` either. The exemption is load-bearing rather
than tidy: the ``EA = T0`` bit-identity anchor sits at ``lam_long == 1.0`` **exactly**, flush
against this bar, so without it a float wobble would fire a spurious warning on this model's
single most important regression test the day CI turns warnings into errors.

**The bar is one-sided, and mind the flip.** It reads ``lam_long`` because the *longitudinal* field
is the fast one -- but only while ``EA > T0``. Below the anchor ``c_long < c``, the **transverse**
field becomes the fast one and plain ``lam`` governs instead; this bar then says nothing useful, and
a caller who resolves ``lam_long`` on a softening string is under-resolving the wave that actually
sets the timestep (measured: ``EA/T0 = 1/200`` at ``lam_long = 0.5`` means ``lam = 7`` and the
Newton Jacobian goes singular, while the same string at ``lam = 0.5`` conserves to 1e-13).
Softening is a deliberate opt-in (see ``allow_softening``), so this is documented rather than
branched on."""


class GeometricState(NamedTuple):
    """A snapshot of the three displacement fields (copies, safe to mutate/store)."""

    u: NDArray[np.float64]
    """Transverse polarization 1 (m) -- the one that reduces to model #3 at ``EA = T0``."""
    w: NDArray[np.float64]
    """Transverse polarization 2 (m) -- the out-of-plane direction."""
    v: NDArray[np.float64]
    """Longitudinal displacement (m) -- where phantom partials live."""


__all__ = [
    "Boundary",
    "GeometricState",
    "GeometricString",
    "LAM_LONG_WARN",
    "NEWTON_MAXITER_DEFAULT",
    "NEWTON_TOL_DEFAULT",
    "THETA_DEFAULT",
]
