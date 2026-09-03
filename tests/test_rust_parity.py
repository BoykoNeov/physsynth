"""What is left of the ideal string's parity file after unit 7's deletion.

This file was the migration's first comparison: ``IdealStringPy`` against ``IdealStringRs``, state
arrays bit-for-bit and energy to the plan's Group A target. **That comparison is gone**, because
``IdealStringPy`` is gone — ``physsynth/core/string_ideal.py`` is now a re-export of the Rust class
and there is no second implementation to hold it against (``docs/dev/rust-migration-plan.md`` §39,
unit 7).

Its 19 tests were not homogeneous, and §39.5 sorted them:

* **Fourteen were two-sided comparisons** of the ideal string. They are deleted here. They retired
  a *diagnostic*, not a physics bar — the string's bars live in ``test_energy.py``,
  ``test_convergence.py``, ``test_modal.py``, ``test_dispersion.py`` and ``test_stability.py``,
  none of which ever named ``IdealStringPy``, and all of which now run against Rust unchanged. The
  native bars in ``crates/physsynth-core/tests/string_ideal.rs`` assert the same physics from the
  other side.
* **Six were properties of the binding** rather than comparisons — buffer lifetime, what ``state``
  hands out, PyO3's argument mapping. They moved to ``tests/test_binding_surface.py``, which that
  file's docstring explains is permanent.
* **One was a cross-model spelling table** over six classes, and it is the only thing left below.

The table is here rather than in ``test_binding_surface.py`` because this half of it *is* still a
two-sided comparison: it asserts that each surviving Python reference implementation refuses
``boundary=None`` the way its Rust twin does. The Rust half is asserted unconditionally next door.
**This list drains one row per deletion, and when it is empty this file goes.** That is why the
rows are written out rather than derived — a derived list would silently shrink to nothing and keep
passing, which is §37.8's finding one level down.
"""

import pytest

physsynth_rs = pytest.importorskip(
    "physsynth_rs", reason="the Rust extension is not built in this environment"
)

# One row per model whose Python reference implementation still exists. `IdealString` went with
# unit 7, `Bore` with unit 3, and the three theta-scheme strings with unit 1. **`FreeBeam` is the
# last one**: when unit 8 lands, this list is empty and this file is deleted in the same commit.
# Written out rather than derived, on purpose -- a derived list would shrink to nothing and keep
# passing, which is section 37.8's finding one level down, and the whole point of this list is that
# reaching zero has to be a visible event.
#
# The `exc` column was not uniform while `Bore` was here: it unpacked its boundary
# (`left, right = boundary`), so Python's refusal was an incidental `TypeError` from the unpacking
# rather than a designed message, while the Rust side raises its own `ValueError` naming what it
# wanted. Reproducing an accident was never worth a special case; silently *accepting* the value
# was, and that half is pinned in `test_binding_surface.py` for all six classes, permanently.
STILL_HAVE_A_PYTHON_TWIN = [
    ("beam", "FreeBeamPy", dict(L=1.0, rho=0.005, fs=48000.0, N=16, kappa=20.0), ValueError),
]


@pytest.mark.parametrize("module,alias,kwargs,exc", STILL_HAVE_A_PYTHON_TWIN)
def test_the_python_twin_also_refuses_an_explicit_none_boundary(module, alias, kwargs, exc):
    import importlib

    cls = getattr(importlib.import_module(f"physsynth.core.{module}"), alias)
    with pytest.raises(exc):
        cls(**kwargs, boundary=None)
