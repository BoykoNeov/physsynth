"""Energy-conserving bridge connection: string terminus <-> modal body (the coupling node of the
``exciter -> resonator -> body/radiation`` abstraction, HANDOFF §3.2).

A string clamped at the nut (left, fixed) and **free at the right end** is loaded there by a linear
spring of stiffness ``K`` connecting the string-end displacement ``u_b = u[N]`` to the body's
driving-point displacement ``w_b = sum_i phi_i q_i``. The spring stores ``E_conn = 1/2 K eta^2``
with stretch ``eta = u_b - w_b``; it pulls the string end with ``-F`` and the body with ``+F``,
where ``F = K eta`` (Newton's third law).

**Why explicit is exact here (not a compromise).** The string, the body, and this spring together
form *one linear leapfrog* ``x^{n+1} = 2x^n - x^{n-1} - k^2 M^{-1} S x^n`` (``S`` symmetric, the
spring contributing the rank-1 block ``K a a^T`` with ``a = [.. +1 (end) .. -phi_i (modes) ..]``).
A linear leapfrog conserves the cross-time energy

    H^n = 1/2 (delta_t- x)^T M (delta_t- x)  +  1/2 (x^n)^T S x^{n-1}

*exactly* for any stable ``k``. That ``H`` splits cleanly into ``E_string + E_body + E_conn`` with
``E_conn = 1/2 K eta^n eta^{n-1}`` (the spring block of ``1/2 (x^n)^T S x^{n-1}``) — so the
three-way energy decomposition is a rigorous identity, not a seam. No implicit solve is needed (that
was for the *nonlinear* von Kármán coupling); a linear spring is conserved explicitly.

**Stability (exact, not a heuristic).** Being explicit, the connection has a CFL. The isolated
end-mass + body 2-DOF estimate ``K (beta_s + beta_b) < 4`` (``beta_s = 2 k^2 / (rho h)`` the string
end-node inverse mass from its ``h/2`` half-cell weight, ``beta_b = k^2 sum_i phi_i^2 / m_i``) is
kept only as a diagnostic — it is *necessary but not sufficient*, because the spring is a rank-1
perturbation that lifts the top coupled eigenvalue above both parts' isolated maxima (empirically
the system can blow up at half that bound). The **actual** guard assembles the coupled leapfrog
operator ``A`` (``x^{n+1} = 2x^n - x^{n-1} - k^2 A x^n``) — matrix-free, applying the *same* step
operators so it cannot disagree with the scheme — and requires ``k^2 lambda_max(A) < 4``, the exact
stability bound. The string's own Nyquist mode already sits near this ceiling at ``lambda = 1``, so
coupling has the most headroom below it; run the string at ``lambda < 1``.

Headless: NumPy only.
"""


__all__ = [
    "StringBodyBridge",
    "StringPlateBridge",
    "StringVKPlateBridge",
    "SympatheticStrings",
]

# --- the Python bodies are DELETED (docs/dev/rust-migration-plan.md section 49) -----------------
#
# The last of them in the project. The four bridges above were ported in Phase 5 batch 10 and ran
# behind `PHYSSYNTH_RS` for eleven batches; this module is now the Rust classes under their public
# names, on every path, with no flag to read.
#
# What survives a deletion is section 41.2's three categories, and here it is only the third with a
# twist: `crates/physsynth-py/src/connection.rs` reads `sparse`, `spsolve` and `splu` **off this
# module's namespace at call time** (`py.import("physsynth.core.connection")`), because the
# reference computed its stability guards through those module globals and the transcription is
# faithful. That is what makes section 24.4's manoeuvre available on this file -- replace the name
# here and the guard changes -- and it is why the three imports below are load-bearing with no
# Python caller left in the file. `ruff check --fix` would delete them and nothing would fail until
# the binding raised `AttributeError` three layers from the cause, so they carry `noqa: F401` AND a
# guard: `tests/test_binding_surface.py` asserts each is the SciPy object it should be, and
# `test_the_guard_reads_its_solver_from_the_module_at_call_time` asserts the binding actually
# reaches them.
#
# `_CFL_TOL` went with the bodies -- private, no reader outside them, and `connection.rs` carries
# its own `const CFL_TOL`. So did `ModalBody`, `Plate`, `VKPlate` and `IdealString`: they were
# imported purely as the collaborators the bridges are handed, and a shim hands nothing to anyone.

from physsynth_rs import (  # noqa: E402
    StringBodyBridge,
    StringPlateBridge,
    StringVKPlateBridge,
    SympatheticStrings,
)

# The three SciPy names below have no caller in this file -- the note above says why they are
# here, and `tests/test_binding_surface.py` asserts both that they are SciPy's own and that the
# binding actually reaches them.
from scipy import sparse  # noqa: E402, F401
from scipy.sparse.linalg import splu, spsolve  # noqa: E402, F401

