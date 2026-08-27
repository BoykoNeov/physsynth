"""Banded Cholesky — the one solver the theta-scheme string family shares.

Four models factor a pentadiagonal SPD update matrix and back-substitute every timestep:
:mod:`~physsynth.core.string_stiff`, :mod:`~physsynth.core.string_damped`,
:mod:`~physsynth.core.string_nonlinear` and :mod:`~physsynth.core.string_geometric`. Until this
module existed each of them imported ``cholesky_banded`` / ``cho_solve_banded`` straight from SciPy,
which was fine while there was only one implementation and is exactly wrong once there are two.

**Why this module exists at all** (``docs/dev/rust-migration-plan.md`` section 15). Those four
models are chained together by *bit-identity* reduction anchors -- ``sigma1 = 0`` makes the damped
string ``array_equal`` to the stiff one, ``EA = 0`` makes the tension-modulated string
``array_equal`` to the damped one, ``EA = T`` makes the geometrically exact string ``array_equal``
to it as well. Those tests compare one model against another, not against a closed form, so they
hold only while every model in the chain does the *same arithmetic in the same order*. Swapping any
single model to Rust would turn an intra-Python anchor into a cross-language one, and the banded
solve cannot carry that: OpenBLAS's triangular band kernel is blocked, and no scalar transcription
reproduces it (measured -- see the plan). Porting the **solver** instead of a model keeps all four
in step, because all four call this.

This is Phase 1's move one level down: :mod:`~physsynth.core.operators` is not a model either, and
flipping it swung five models at once.

Headless: NumPy + SciPy. No I/O, no plotting.
"""

from __future__ import annotations

import os

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_solve_banded, cholesky_banded

__all__ = ["cho_solve_banded", "cholesky_banded"]


# --- the Rust swap (docs/dev/rust-migration-plan.md, Phase 3) -----------------------------------
#
# The reference implementation here is SciPy's, i.e. LAPACK's -- unlike every other swap in this
# migration, the `_py` alias is not a Python transcription but the library call the acceptance
# numbers came from. That is the honest labelling: what this module ports is the *choice of
# solver*, and the Python side of the comparison is dpbtrf/dpbtrs.
#
# What the Rust side is, precisely: the reference LAPACK algorithms transcribed plainly -- DPBTF2
# for the factor (which is the path dpbtrf takes at kd = 2, since NB > KD), DTBSV twice for the
# solve. Two details are not cosmetic and are written down where they live, in
# `crates/physsynth-core/src/banded.rs`: DSCAL forms the reciprocal ONCE and multiplies, which is
# the reference algorithm's own behaviour and worth 19/120 -> 120/120 agreement on this family's
# matrices; and the fused multiply-add in DSYR is deliberately NOT reproduced, because fusing is a
# property of the kernel OpenBLAS picks at run time and asserting it would be a claim about a CPU.
#
# So this swap CHANGES THE NUMBERS. It is the first one in the migration that does so by design.
# What it does not change is any physics bar: measured on 2026-08-27, the transcribed solver's
# lossless energy drift is 2.7e-12 against LAPACK's 2.7e-12 on the same string (the bar is 1e-10),
# and the two trajectories agree to ~1e-13 of amplitude over a short run, growing slowly. The four
# models' agreement with each other stays EXACT, which is the property the anchors need.
#
# Off by default. SciPy is still the reference oracle.
cholesky_banded_py = cholesky_banded
cho_solve_banded_py = cho_solve_banded

_USE_RUST = os.environ.get("PHYSSYNTH_RS", "").strip() not in ("", "0", "false", "False")

if _USE_RUST:  # pragma: no cover - exercised by the dedicated CI job, not the default gate
    import physsynth_rs as _rs
    from scipy.linalg import LinAlgError

    def cholesky_banded(  # type: ignore[misc]  # noqa: F811
        ab: NDArray[np.float64],
        overwrite_ab: bool = False,
        lower: bool = False,
        check_finite: bool = True,
    ) -> NDArray[np.float64]:
        """``scipy.linalg.cholesky_banded``, on the Rust transcription of DPBTF2.

        ``overwrite_ab`` is accepted and ignored -- the Rust side always returns a fresh array,
        which is what every caller in this project already assumes. So is ``check_finite``: the
        Rust side always checks, because it does so in the pass that copies the input and a
        separate ``np.isfinite`` here cost more than the whole port saved. ``lower=True`` is
        refused rather than silently transposed: no call site asks for it, so it would be
        untested code.
        """
        if lower:
            raise NotImplementedError(
                "the Rust banded Cholesky implements upper storage only; every call site in "
                "this project passes lower=False"
            )
        try:
            return _rs.cholesky_banded_upper(ab)
        except _rs.NotPositiveDefinite as exc:
            raise LinAlgError(str(exc)) from None

    def cho_solve_banded(  # type: ignore[misc]  # noqa: F811
        cb_and_lower: tuple[NDArray[np.float64], bool],
        b: NDArray[np.float64],
        overwrite_b: bool = False,
        check_finite: bool = True,
    ) -> NDArray[np.float64]:
        """``scipy.linalg.cho_solve_banded``, on the Rust transcription of DPBTRS."""
        cb, lower = cb_and_lower
        if lower:
            raise NotImplementedError(
                "the Rust banded solve implements upper storage only; every call site in this "
                "project passes lower=False"
            )
        return _rs.cho_solve_banded_upper(cb, b)
