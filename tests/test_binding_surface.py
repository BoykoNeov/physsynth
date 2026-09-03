"""Properties of the **binding**, which no physics bar and no native Rust bar can see.

This file is the residue of ``test_rust_parity.py`` after the migration's first deletion
(``docs/dev/rust-migration-plan.md`` §39.5). It exists because two kinds of test were mixed in
there and only one of them dies when a Python reference implementation is deleted:

* a **comparison** ("Python and Rust agree") has nothing left to compare and goes;
* a **property** ("a reference to ``.u`` held across a step is a snapshot") is a statement about
  the object handed to Python, and is as true and as necessary afterwards as before.

The properties below are permanent. They cannot move to ``crates/physsynth-core/tests/`` — that
crate has no Python objects, so it has no buffer whose lifetime it could check and no PyO3
signature whose argument mapping it could probe. §39.3 says the same of ``connection.py`` and the
airbox wrapper tier for the same underlying reason: what only exists at the binding can only be
checked from Python.

Deliberately **not** named ``test_rust_parity_*``: that prefix means "must run unflagged, because
it builds both sides itself" (``scripts/shard_tests.py``'s ``PARITY_PREFIX``). Nothing here builds
a Python side, so it should run in every shard of every job, flagged or not.
"""

import numpy as np
import pytest

from physsynth.core.exciter import triangular_pluck

physsynth_rs = pytest.importorskip(
    "physsynth_rs", reason="the Rust extension is not built in this environment"
)
IdealString = physsynth_rs.IdealString

L_DEFAULT = 1.0
T_DEFAULT = 200.0
RHO_DEFAULT = 0.005  # -> c = 200 m/s


def _params(*, N=100, lam=1.0, boundary="fixed", sigma=0.0):
    c = np.sqrt(T_DEFAULT / RHO_DEFAULT)
    return {
        "L": L_DEFAULT,
        "T": T_DEFAULT,
        "rho": RHO_DEFAULT,
        "fs": c * N / (L_DEFAULT * lam),
        "N": N,
        "boundary": boundary,
        "sigma": sigma,
    }


# -- what `state` hands out ----------------------------------------------------------------------


def test_state_is_a_copy_not_a_view():
    s = IdealString(**_params(N=8))
    s.set_state(triangular_pluck(s.x, s.L, 0.3 * s.L, amplitude=1e-3))
    snapshot = s.state
    snapshot[3] = 12345.0
    assert s.u[3] != 12345.0, "`state` handed out a live view"


def test_set_state_accepts_a_plain_list():
    # Not a numpy array: `web/serialize.py` deserialises JSON, and JSON has no ndarray.
    s = IdealString(**_params(N=4))
    s.set_state([0.0, 1.0, 2.0, 1.0, 0.0])
    assert np.array_equal(s.u, np.array([0.0, 1.0, 2.0, 1.0, 0.0]))


# -- buffer lifetime -----------------------------------------------------------------------------


def test_a_reference_to_u_held_across_a_step_is_a_snapshot():
    # `step()` rebinds `self.u`; it does not write into it. So a reference taken before the step
    # keeps showing that step's values. The binding reproduces this by owning NumPy arrays rather
    # than Rust `Vec`s — a zero-copy view over a reallocated `Vec` would *look* right here
    # (measured: it still reads the old contents) while being a use-after-free.
    s = IdealString(**_params(N=16))
    s.set_state(triangular_pluck(s.x, s.L, 0.3 * s.L, amplitude=1e-3))
    held = s.u
    before = np.array(held, copy=True)

    s.step()

    assert np.array_equal(held, before), "the held reference changed under a step"
    assert not np.array_equal(s.u, before), "the string did not actually advance"
    assert s.u_prev is held, "u_prev after a step must be the very object u was"


def test_an_in_place_write_through_u_reaches_the_string():
    # This is exactly what a bridge connection does — `self.string.u[-1] -= beta_s * F` — and it
    # is the reason `.u` cannot be handed out as a copy.
    s = IdealString(**_params(N=16, boundary=("fixed", "free")))
    s.set_state(triangular_pluck(s.x, s.L, 0.3 * s.L, amplitude=1e-3))
    before = s.energy()

    s.u[-1] -= 1e-3

    assert s.energy() != before, "a write through `.u` did not reach the string"
    assert s.state[-1] == s.u[-1]
    s.step()  # and the modified state is what gets stepped, without complaint


def test_the_state_arrays_can_be_replaced_wholesale():
    # `tests/test_collision_modal.py` does this to a string it owns (`bar.string.u = uf.copy()`),
    # so assignment has to work, not just mutation.
    s = IdealString(**_params(N=8))
    fresh = np.linspace(0.0, 1.0, 9)
    s.u = fresh.copy()
    s.u_prev = fresh.copy()
    s.n = 0
    assert np.array_equal(s.u, fresh)
    s.step()
    assert s.n == 1


# -- an omitted keyword and an explicit `None` are different arguments (plan §24.7) ---------------
#
# Found while writing the beam's parity file, and it was true of every binding in the crate. PyO3
# maps a Python `None` and a missing argument onto the same Rust `Option::None`, so a binding
# written the obvious way treats `boundary=None` as "not supplied" and quietly builds the DEFAULT
# boundary, while the Python original rejected it. Nothing caught it: no parity file passed `None`
# to a constructor, because `None` is not a plausible boundary and nobody thinks to try it.
#
# The fix is `Option<Option<_>>`, and the arm order is the surprising half -- PyO3 wraps the
# *default expression*, so `Some(None)` means "argument omitted" and a bare `None` is the caller's
# literal. That is exactly the kind of thing that gets silently inverted in a later refactor, so
# both halves are pinned: the default still applies when the argument is absent, and an explicit
# `None` is still refused.
#
# Both halves are binding properties and live here, for all six classes, permanently. What stayed
# behind in `test_rust_parity.py` is only the claim that the *Python* twin refuses it the same way,
# which drains one row per deletion.

BOUNDARY_CASES = [
    ("StiffString", dict(L=1.0, T=200.0, rho=0.005, fs=48000.0, N=16), "supported"),
    ("DampedStiffString", dict(L=1.0, T=200.0, rho=0.005, fs=48000.0, N=16), "supported"),
    ("TensionModulatedString", dict(L=1.0, T=200.0, rho=0.005, fs=48000.0, N=16), "supported"),
    ("IdealString", dict(L=1.0, T=200.0, rho=0.005, fs=48000.0, N=16), "fixed"),
    ("Bore", dict(L=0.5, fs=48000.0, N=64), ("closed", "open")),
    ("FreeBeam", dict(L=1.0, rho=0.005, fs=48000.0, N=16, kappa=20.0), "free"),
]


@pytest.mark.parametrize("name,kwargs,default", BOUNDARY_CASES)
def test_omitting_the_boundary_keeps_the_default(name, kwargs, default):
    assert getattr(physsynth_rs, name)(**kwargs).boundary == default


@pytest.mark.parametrize("name,kwargs,default", BOUNDARY_CASES)
def test_an_explicit_none_boundary_is_refused(name, kwargs, default):
    with pytest.raises(ValueError):
        getattr(physsynth_rs, name)(**kwargs, boundary=None)
