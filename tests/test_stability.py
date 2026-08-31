"""Stability and construction guards (criterion 4) + the headless-core invariant.

- No NaN / blow-up across a sweep of valid lambda in (0, 1].
- A deliberate lambda > 1 is rejected at construction time, not silently run into an overflow.
- Non-physical parameters are rejected.
- physsynth.core imports no plotting/audio library (CLAUDE.md non-negotiable #4).
"""

import os
import subprocess
import sys

import numpy as np
import pytest
from helpers import make_string, wave_speed

from physsynth.core.engine import simulate
from physsynth.core.exciter import triangular_pluck
from physsynth.core.string_ideal import IdealString


@pytest.mark.parametrize("lam", [0.999, 0.95, 0.9, 0.75, 0.5, 0.3, 0.1])
def test_no_nan_across_valid_lambda(lam):
    string = make_string(N=100, lam=lam)
    string.set_state(triangular_pluck(string.x, string.L, 0.3 * string.L, amplitude=1e-3))
    res = simulate(string, num_steps=int(0.5 * string.fs), pickup_index=50)
    assert np.all(np.isfinite(res.output))
    assert np.all(np.isfinite(res.energy))


def test_lambda_above_one_rejected_at_construction():
    c, L, N = wave_speed(), 1.0, 100
    # lambda = c*N / (fs*L): a LOWER fs (coarser time step) raises lambda, so divide by 1.05.
    fs_unstable = c * N / (L * 1.05)  # forces lambda = 1.05 > 1
    with pytest.raises(ValueError, match="CFL"):
        IdealString(L=L, T=200.0, rho=0.005, fs=fs_unstable, N=N)


def test_lambda_exactly_one_is_accepted():
    # The CFL guard must not reject the exact (and most accurate) lambda = 1 case on round-off.
    string = make_string(N=100, lam=1.0)
    assert string.lam == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rho": -1.0},
        {"T": 0.0},
        {"L": -2.0},
        {"sigma": -0.1},
        {"N": 1},
        {"boundary": "clamped"},
    ],
)
def test_invalid_parameters_rejected(kwargs):
    base = {"L": 1.0, "T": 200.0, "rho": 0.005, "fs": 20000.0, "N": 100}
    base.update(kwargs)
    with pytest.raises(ValueError):
        IdealString(**base)


def test_core_is_headless():
    # Import the whole core in a fresh interpreter and assert no plotting/audio library was pulled
    # in (transitively included). The core must stay portable to C++/Rust later.
    code = (
        "import sys;"
        "import physsynth.core.operators, physsynth.core.string_ideal,"
        "       physsynth.core.exciter, physsynth.core.engine;"
        "forbidden={'matplotlib','sounddevice','pyaudio','pygame','PyQt5','PySide6'};"
        "hit=sorted(m for m in sys.modules if m.split('.')[0] in forbidden);"
        "print(','.join(hit));"
        "sys.exit(1 if hit else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, f"core imported forbidden libraries: {result.stdout.strip()}"


# --- portability contract (docs/dev/portability-contract.md) ----------------------------------

# Boilerplate run in a *fresh* interpreter: import every submodule of physsynth.core so these
# guards auto-cover new core modules (e.g. a future string_stiff.py) with no edits here.
_IMPORT_ALL_CORE = (
    "import sys, importlib, pkgutil;"
    "import physsynth.core as _core;"
    "[importlib.import_module(m.name) "
    " for m in pkgutil.iter_modules(_core.__path__, _core.__name__ + '.')];"
)


def _run_core_probe(body: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _IMPORT_ALL_CORE + body], capture_output=True, text=True
    )


# Hardcoded allowlist of top-level third-party packages the core is permitted to pull in: the
# declared numeric stack (numpy + scipy) plus the compiled-extension runtime baggage that stack
# unavoidably drags along. Verified empirically -- importing the numpy/scipy stack alone pulls
# exactly {numpy, scipy, charset_normalizer, cython_runtime, <hash>__mypyc}. Anything outside this
# set (torch, requests, PIL, sounddevice, ...) is a real portability leak and must fail the test.
# The mypyc runtime is named with a per-build hash prefix (e.g. "81d243...__mypyc"), so it is
# matched structurally by its "__mypyc" suffix, not by name.
#
# `physsynth_rs` (docs/dev/rust-migration-plan.md §2.2) is the ONE deliberate, reviewed addition
# this list has ever taken. It is the compiled Rust core, and `sys.modules` cannot tell it apart
# from any other third-party binary -- which is the whole reason it has to be named here rather
# than pattern-matched. Two things follow, and both are the point:
#
#   - It only appears when `PHYSSYNTH_RS` is set, so on the default path this entry is inert. The
#     list did not get looser for the runs that produce the acceptance numbers.
#   - The rule it used to carry alone -- "the core depends on the numeric stack and nothing else"
#     -- now lives on the Rust side too, as `crates/physsynth-core/tests/deps.rs`, which checks
#     the same thing against `cargo metadata`. This test can no longer see inside the extension
#     module, so something has to, and that is it. Same spirit, same visibility, other language.
_CORE_DEP_ALLOWLIST = {
    "numpy",
    "scipy",
    "charset_normalizer",
    "cython_runtime",
    "physsynth",
    "physsynth_rs",
}


def test_core_dependency_allowlist():
    # Stronger than the blocklist above: the core may use ONLY the allowlisted numeric stack and
    # its compiled-runtime baggage -- no third-party dependency of its own. Import every core
    # submodule (auto-discovers new ones, e.g. string_stiff) and assert nothing outside the
    # allowlist appears. We measure the DELTA -- modules pulled *by importing the core*, not the
    # absolute set -- by snapshotting sys.modules first: this excludes interpreter-startup baggage
    # injected via a .pth (e.g. Windows pywin32's pywin32_bootstrap/pywin32_system32), which is
    # present for *any* subprocess and is not something the core pulls. Underscore-private modules
    # (_csparsetools, editable-install finders, ...) are internal plumbing, excluded by the
    # leading-underscore rule; the hash-suffixed mypyc runtime is excluded by its "__mypyc" suffix.
    allowed = sorted(_CORE_DEP_ALLOWLIST)
    probe = (
        "import sys, importlib, pkgutil;"
        "stdlib=set(sys.stdlib_module_names)|set(sys.builtin_module_names);"
        "before=set(sys.modules);"
        "import physsynth.core as _core;"
        "[importlib.import_module(m.name) "
        " for m in pkgutil.iter_modules(_core.__path__, _core.__name__ + '.')];"
        "allowed=set(" + repr(allowed) + ");"
        "tp={n.split('.')[0] for n in set(sys.modules) - before"
        "    if n.split('.')[0] not in stdlib and not n.startswith('_')"
        "    and not n.endswith('__mypyc')};"
        "leaked=sorted(tp - allowed);"
        "print(','.join(leaked));"
        "sys.exit(1 if leaked else 0)"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert result.returncode == 0, (
        f"core pulled third-party module(s) outside the allowlist {allowed}: "
        f"{result.stdout.strip()}"
    )


def test_the_rust_swap_matches_the_environment():
    # The guard without which the whole Rust CI job can be green and mean nothing.
    #
    # `PHYSSYNTH_RS=1 pytest ...` claims to run the existing tests against the Rust model. But
    # nothing in those tests mentions Rust -- the substitution happens at the bottom of
    # `physsynth/core/string_ideal.py` -- so if the variable were mistyped, or the swap block were
    # refactored away, or an import-ordering change defeated it, the suite would run against
    # PYTHON and pass. Green, and asserting nothing about the port. That is verbatim the failure
    # mode `docs/dev/rust-migration-plan.md` §1 exists to prevent, one level up.
    #
    # This cannot live in `tests/test_rust_parity.py`: that file imports both implementations by
    # name and is immune to the swap by design. It belongs here, with the other three portability
    # guards, and it runs on BOTH paths -- the default one included, where it asserts the Rust
    # model is *not* silently in play.
    from physsynth.core import (
        airbox,
        banded,
        beam,
        body,
        bore,
        bow,
        collision,
        exciter,
        mallet,
        membrane,
        operators,
        operators2d,
        plate,
        reed,
        string_damped,
        string_geometric,
        string_ideal,
        string_nonlinear,
        string_stiff,
    )

    expected_rust = os.environ.get("PHYSSYNTH_RS", "").strip() not in ("", "0", "false", "False")
    for module in (
        string_ideal,
        operators,
        # `airbox` has read PHYSSYNTH_RS since Phase 5's fourth batch (it takes the `splu` swap),
        # and it was NOT in this tuple until the fifth ported `AirBox` itself -- a whole batch in
        # which its reading of the flag could have diverged with nothing noticing. The derive is
        # only as wide as the tuple it derives over, for the fourth time.
        airbox,
        membrane,
        operators2d,
        exciter,
        body,
        bore,
        reed,
        banded,
        collision,
        mallet,
        string_stiff,
        string_damped,
        string_nonlinear,
        string_geometric,
        plate,
        bow,
        beam,
    ):
        assert module._USE_RUST is expected_rust, (
            f"{module.__name__}'s reading of PHYSSYNTH_RS disagrees with this test's -- one of "
            "the two changed without the other"
        )

    # The class half of the guard, DERIVED rather than listed -- section 17.6's finding applied to
    # the half it was not applied to. `collision` fell out of the function table below for a whole
    # batch because its names were spelled unexpectedly, and nothing failed; a block of
    # hand-written `assert mallet.MalletWall is physsynth_rs.MalletWall` lines has exactly the same
    # hole, one forgotten paste wide. So the set of swapped CLASSES is read off the `<Name>Py`
    # aliases the modules actually define, and then checked against a written-down expectation --
    # which is what keeps adding a port a reviewed edit rather than a silent one.
    swapped_classes = {}
    for module in (
        airbox,
        string_ideal,
        membrane,
        body,
        bore,
        reed,
        mallet,
        string_stiff,
        string_damped,
        string_nonlinear,
        string_geometric,
        bow,
        collision,
        beam,
        operators2d,
        plate,
    ):
        # `collision` joined this tuple in Phase 3's last batch, and it was ABSENT before -- so for
        # the whole of §16's batch and after, a `BarrierStringPy` alias could have been added with
        # nothing noticing. That is §17.6's finding a second time, in the half of the guard §17.6's
        # own text says it was applying the lesson TO. The derive is only as wide as the tuple it
        # is derived over, which is the one thing about it that still has to be written by hand.
        for alias in dir(module):
            if not alias.endswith("Py") or alias.endswith("_py"):
                continue
            name = alias[:-2]
            reference = getattr(module, alias)
            if isinstance(reference, type) and isinstance(getattr(module, name, None), type):
                swapped_classes[(module.__name__, name)] = (module, name, reference)

    expected_classes = {
        ("physsynth.core.airbox", "AirBox"),
        ("physsynth.core.airbox", "RoomPort"),
        ("physsynth.core.airbox", "SurfacePort"),
        ("physsynth.core.airbox", "InteriorSurfacePort"),
        ("physsynth.core.airbox", "_PlateSurface"),
        ("physsynth.core.airbox", "_VKPlateSurface"),
        ("physsynth.core.airbox", "RoomLoadedBody"),
        ("physsynth.core.airbox", "RoomLoadedPlate"),
        ("physsynth.core.airbox", "RoomSuspendedPlate"),
        ("physsynth.core.airbox", "RoomLoadedVKPlate"),
        ("physsynth.core.airbox", "RoomSuspendedVKPlate"),
        ("physsynth.core.string_ideal", "IdealString"),
        ("physsynth.core.membrane", "Membrane"),
        ("physsynth.core.body", "ModalBody"),
        ("physsynth.core.bore", "Bore"),
        ("physsynth.core.reed", "ReedBore"),
        ("physsynth.core.mallet", "MalletMembrane"),
        ("physsynth.core.mallet", "MalletWall"),
        ("physsynth.core.string_stiff", "StiffString"),
        ("physsynth.core.string_damped", "DampedStiffString"),
        ("physsynth.core.string_nonlinear", "TensionModulatedString"),
        ("physsynth.core.string_geometric", "GeometricString"),
        ("physsynth.core.operators2d", "VonKarmanBracket"),
        ("physsynth.core.operators2d", "AiryStressSolver"),
        ("physsynth.core.bow", "BowedString"),
        ("physsynth.core.collision", "BarrierString"),
        ("physsynth.core.beam", "FreeBeam"),
        ("physsynth.core.plate", "Plate"),
        ("physsynth.core.plate", "VKPlate"),
    }
    assert set(swapped_classes) == expected_classes, (
        f"the swapped classes are {sorted(swapped_classes)}, but this guard expects "
        f"{sorted(expected_classes)} -- a model was ported (or unported) without the guard "
        "being updated"
    )

    if expected_rust:
        import physsynth_rs

        for module, name, _reference in swapped_classes.values():
            assert getattr(module, name) is getattr(physsynth_rs, name), (
                f"PHYSSYNTH_RS is set but `{name}` is still the Python class: this run is NOT "
                "exercising the Rust model, whatever it reports"
            )
    else:
        for module, name, reference in swapped_classes.values():
            assert getattr(module, name) is reference, (
                f"PHYSSYNTH_RS is unset but `{name}` is not the Python class -- the default path "
                "must stay the one the acceptance numbers came from"
            )

    # The operators (plan Phase 1) need the same guard for the same reason, and it has to be
    # spelled differently: they are functions, so there is no class identity to compare. The
    # question that matters is whether the PUBLIC name still refers to the `_py` implementation.
    #
    # This one is wider than the string's. `string_stiff`, `string_damped`, `string_nonlinear`,
    # `string_geometric` and `beam` all do `from .operators import ...` at import time, so the
    # binding they captured is fixed the moment their module was first imported. If the swap ever
    # landed after them -- a lazy import, a reordered `physsynth.core.__init__` -- those five would
    # hold the Python functions while `operators` reported Rust, and the run would be green while
    # testing the wrong thing for five models at once.
    #
    # The set of swapped names is DERIVED from the `_py` aliases the module actually defines, not
    # listed here -- so a function ported without an alias escapes nothing and a function aliased
    # without being swapped fails loudly. It is then checked against a written-down expectation,
    # which is what makes adding a port a reviewed edit rather than a silent one (the same
    # reasoning as the hardcoded dependency allowlist below).
    ported_expected = {
        # `airbox` is ported in HALVES: the room in Phase 5's sixth batch and the port tier in the
        # seventh, with the six `RoomLoaded*` / `RoomSuspended*` wrappers still Python. Its three
        # port CLASSES are checked by identity above; the one free function that moved is the
        # helper the ports share. The alias is `free_pressure_nodes_py` and not
        # `_free_pressure_nodes_py`, because the collection below skips aliases beginning with an
        # underscore -- `collision`'s spelling, for `collision`'s reason.
        airbox: {"free_pressure_nodes"},
        # `plate` is ported in full as of Phase 5's fourth batch. Both CLASSES are checked by
        # identity above; the only free FUNCTION in the module is the material helper. Its private
        # `_count_components` is deliberately NOT aliased: it is not swapped either, and an alias
        # with no swap behind it is exactly what this derive is built to catch.
        plate: {"grain_ratios_from_material"},
        # `operators` is ported in full (plan Phase 1).
        operators: set(operators.__all__),
        # `operators2d` is ported in FULL as of Phase 5's third batch, and it took four of them:
        # the builders the membrane needs (plan Phase 2), the guitar outline's geometry (Phase 5
        # batch 1), the plate's matrices (batch 2) and the nonlinear plate -- the five private 1-D
        # differences, `VonKarmanBracket` and `AiryStressSolver` -- in batch 3. The two classes are
        # checked by identity above rather than here, and `operators2d` had to be ADDED to that
        # derive's tuple to be seen at all, which is section 23.7's finding coming due on schedule.
        #
        # The five 1-D differences are here under their unprefixed spellings for the reason
        # `collision`'s entry records: the module names it `_dirichlet_interior_d2_1d` and the
        # alias namespace is flat, so `_public` below tries the underscored form too. That
        # mismatch is exactly what dropped a whole module out of this table for a batch.
        operators2d: {
            "grid_coords",
            "rectangle_mask",
            "disk_mask",
            "guitar_half_width",
            "guitar_scale",
            "guitar_mask",
            "guitar_area",
            "live_cells",
            "cells_per_node",
            "prune_to_area_carrying",
            "laplacian_from_mask",
            "biharmonic_from_mask",
            "dirichlet_interior_d2_1d",
            "orthotropic_biharmonic",
            "free_plate_stiffness",
            "free_plate_stiffness_from_mask",
            "collocated_d2_1d",
            "forward_d1_1d",
            "centered_d2_1d",
            "clamped_d2_1d",
            "avg_d1_1d",
            "embed",
            "inner2d",
            "norm2_2d",
        },
        exciter: set(exciter.__all__),
        # `banded` is ported in full (plan Phase 3), and it is the odd one out in this table: its
        # `_py` aliases are SciPy's functions, not a Python transcription, because what this
        # module ports is a choice of solver rather than a piece of arithmetic somebody wrote
        # here. It is also the only swap so far that deliberately changes the numbers -- see the
        # module's own header for why that is safe and what it buys.
        banded: set(banded.__all__),
        # `reed` is ported in full, but only ONE of its two public names is a function -- the
        # class is checked by identity above. `bernoulli_flow` needs the swap because
        # `tests/test_reed_stability.py` imports it by name and asserts its oddness and passivity
        # directly, and that file is in the flagged CI step.
        reed: {"bernoulli_flow"},
        # `collision` is ported in full, `BarrierString` included since Phase 3's last batch --
        # the class half of that is checked by identity above. This entry was missing until the
        # mallet's batch, and the reason is worth writing down rather than quietly fixing: three
        # of the module's public
        # names carry a LEADING UNDERSCORE while their aliases do not
        # (`_force_total_vec` <-> `force_total_vec_py`), so the derive below could not find them
        # and the whole module fell out of the table. A guard that silently covers nothing is the
        # same shape as the empty parity job section 16.8 found, reached through a third door --
        # so the lookup now tries the underscored spelling too, which keeps the set DERIVED rather
        # than listed.
        # `bow` is ported in full. Both module-level functions are swapped because
        # `tests/test_bow_stability.py` imports them BY NAME and asserts the friction curve's
        # oddness, peak and derivative directly -- the same reason `reed.bernoulli_flow` is here.
        bow: {"friction_smooth", "friction_smooth_deriv"},
        collision: {
            "contact_potential",
            "contact_force_elastic",
            "contact_stiffness",
            "contact_force_dg",
            "contact_force_total",
            "contact_force_total_deriv",
            "force_total_vec",
            "deriv_total_vec",
            "solve_contact",
            "solve_contact_vector",
        },
    }

    def _public(module, name):
        """The public name an alias `<name>_py` refers to -- underscored or not.

        `collision` exports `_force_total_vec` but aliases it `force_total_vec_py`, because the
        alias namespace is flat and the module's is not.
        """
        if hasattr(module, name):
            return getattr(module, name)
        return getattr(module, f"_{name}")

    for module, expected_names in ported_expected.items():
        aliased = {n[:-3] for n in dir(module) if n.endswith("_py") and not n.startswith("_")}
        assert aliased == expected_names, (
            f"{module.__name__}'s `_py` aliases are {sorted(aliased)}, but this guard expects "
            f"{sorted(expected_names)} -- a port landed (or left) without the guard being updated"
        )
        for name in sorted(expected_names):
            public = _public(module, name)
            reference = getattr(module, f"{name}_py")
            if expected_rust:
                assert public is not reference, (
                    f"PHYSSYNTH_RS is set but `{module.__name__}.{name}` is still the Python "
                    "function: this run is NOT exercising the Rust code, whatever it reports"
                )
            else:
                assert public is reference, (
                    f"PHYSSYNTH_RS is unset but `{module.__name__}.{name}` is not the Python "
                    "function -- the default path must stay the one the numbers came from"
                )

    if expected_rust:
        from physsynth.core import (
            beam,
            connection,
            mallet,
            plate,
            radiation,
            string_damped,
            string_geometric,
            string_nonlinear,
            string_stiff,
        )

        # The banded solver's version of the captured-binding hazard, and it is the widest one in
        # this test. Four models do `from .banded import cho_solve_banded, cholesky_banded` at
        # module scope, and unlike every other entry here the consequence of a mis-ordered swap is
        # not merely "one model runs Python". These four are chained by `array_equal` reduction
        # anchors -- sigma1 = 0, EA = 0, EA = T -- which hold only while all four do the SAME
        # arithmetic. If the swap reached three of them and not the fourth, those anchors would
        # start comparing LAPACK against a transcription and fail with a message about physics.
        for model in (string_stiff, string_damped, string_nonlinear, string_geometric):
            for name in ("cholesky_banded", "cho_solve_banded"):
                assert getattr(model, name) is getattr(banded, name), (
                    f"`{model.__name__}` captured a different `{name}` than `banded` now exposes "
                    "-- the swap landed after that module was imported, so this model is solving "
                    "with LAPACK while the rest of the family solves with Rust"
                )

        assert string_stiff.biharmonic_matrix is operators.biharmonic_matrix, (
            "`string_stiff` captured a different `biharmonic_matrix` than `operators` now "
            "exposes -- the swap landed after that module was imported, so the model is running "
            "on the Python operator while this run claims otherwise"
        )
        assert beam.free_beam_stiffness is operators.free_beam_stiffness, (
            "`beam` captured a different `free_beam_stiffness` than `operators` now exposes"
        )
        # The membrane's version of the same hazard: `mallet` does `from .membrane import
        # Membrane` at import time, so a swap that landed after it would leave the mallet striking
        # a Python drumhead while this run reports Rust.
        assert mallet.Membrane is membrane.Membrane, (
            "`mallet` captured a different `Membrane` than `membrane` now exposes -- the swap "
            "landed after that module was imported"
        )
        assert membrane.laplacian_from_mask is operators2d.laplacian_from_mask, (
            "`membrane` captured a different `laplacian_from_mask` than `operators2d` now exposes"
        )
        # The plate's version, added with Phase 5's first batch. `plate` is not ported, but it
        # does `from .operators2d import guitar_mask, ...` at module scope, so it is a CLIENT of
        # names that are -- and the geometry is the half of the module where a mis-ordered swap
        # would be worst. A Python `guitar_mask` under a run that claims Rust is a plate with a
        # possibly different set of nodes, which every physics bar in the suite would pass.
        # The second batch widens it from the outline to the OPERATORS: `plate` also captures the
        # three matrix builders, and `Plate.step` multiplies by two of them every timestep. A
        # mis-ordered swap there is subtler than a wrong outline and not much better -- the plate
        # would step on a SciPy-ordered operator while the run reported Rust, which is a different
        # trajectory in its last bits and nothing in the suite says so.
        for name in (
            "guitar_half_width",
            "guitar_scale",
            "guitar_mask",
            "guitar_area",
            "prune_to_area_carrying",
            "biharmonic_from_mask",
            "orthotropic_biharmonic",
            "free_plate_stiffness",
            "free_plate_stiffness_from_mask",
        ):
            assert getattr(plate, name) is getattr(operators2d, name), (
                f"`plate` captured a different `{name}` than `operators2d` now exposes -- the "
                "swap landed after that module was imported, so this plate's outline is being "
                "built by the Python code while the run reports Rust"
            )

        # The same assertion one level out, added with Phase 5's fourth batch, which is what
        # creates the exposure: `connection.py` does `from .plate import Plate, VKPlate` at module
        # SCOPE, so it holds whatever those names were bound to when it was imported. That is
        # section 0's named failure mode verbatim -- a lazy import or a reordered
        # `physsynth.core.__init__` and the bridges would build Python plates while the run
        # reported Rust, which is a different trajectory in its last bits and nothing else in the
        # suite would say so.
        #
        # `airbox.py` needs no entry here and that is worth stating rather than leaving to a
        # reader: its `from physsynth.core.plate import Plate, VKPlate` sits under
        # `if TYPE_CHECKING:`, so it captures nothing at runtime. What airbox DOES capture is
        # `splu`, which it re-binds itself (plan section 28.2) and which the bare-vs-loaded
        # reduction tests already pin exactly.
        for name in ("Plate", "VKPlate"):
            assert getattr(connection, name) is getattr(plate, name), (
                f"`connection` captured a different `{name}` than `plate` now exposes -- the swap "
                "landed after that module was imported, so a bridge is driving a Python plate "
                "while the run reports Rust"
            )

        # The body's version, and it has TWO importers rather than one. `connection` and
        # `radiation` both do `from .body import ModalBody` at module scope, so a swap that
        # landed after either of them would leave the whole body/radiation leg on Python
        # while this run reported Rust. (`airbox` imports the name only under TYPE_CHECKING
        # and so captures nothing at runtime -- deliberately, and worth not un-doing.)
        for client in (connection, radiation):
            assert client.ModalBody is body.ModalBody, (
                f"`{client.__name__}` captured a different `ModalBody` than `body` now "
                "exposes -- the swap landed after that module was imported"
            )

        # The string's version, and the one this batch added. `bow.py` does `from .string_damped
        # import DampedStiffString` at module scope, so a swap that landed after it would leave the
        # bow drawing across a PYTHON string while this run reported Rust. Unlike the four
        # `cho_solve_banded` captures above, this one is caught for free at construction -- the
        # Rust `BowedString` raises `TypeError` on a Python string rather than falling back -- so
        # the assertion is here to say WHICH of the two broke, not to be the only thing that
        # notices.
        assert bow.DampedStiffString is string_damped.DampedStiffString, (
            "`bow` captured a different `DampedStiffString` than `string_damped` now exposes -- "
            "the swap landed after that module was imported"
        )

        # The bore's version. `reed.py` does `from .bore import Bore` at module scope, so a swap
        # that landed after it would leave the clarinet blowing a PYTHON air column while this run
        # reported Rust -- and every reed test would still pass, because the reed's own physics is
        # unchanged by which bore it drives. Same hazard as `mallet.Membrane`, one model on.
        assert reed.Bore is bore.Bore, (
            "`reed` captured a different `Bore` than `bore` now exposes -- the swap landed "
            "after that module was imported"
        )
        # And the sharper version of the same question, which only this pair can ask: the Rust
        # reed REFUSES a Python bore rather than falling back to a per-step interpreter crossing,
        # so if the two swaps ever landed out of order the clarinet would not quietly run slow --
        # it would raise. Asserting it here means the failure is one line of test output rather
        # than a TypeError from inside somebody's model.
        rs_bore = bore.Bore(
            L=0.6, fs=343.0 / (0.6 / 60), N=60, boundary=("closed", "open")
        )
        reed.ReedBore(bore=rs_bore, p_mouth=1500.0)


def test_core_does_not_import_sibling_layers():
    # The dependency arrow points one way: analysis/viz/io depend on core, never the reverse.
    body = (
        "bad={'physsynth.viz','physsynth.analysis','physsynth.io'};"
        "hit=sorted(m for m in sys.modules if any(m==b or m.startswith(b+'.') for b in bad));"
        "print(','.join(hit));"
        "sys.exit(1 if hit else 0)"
    )
    result = _run_core_probe(body)
    assert result.returncode == 0, (
        f"core imported a sibling layer (must not): {result.stdout.strip()}"
    )


# -- the ARPACK start vector is pinned everywhere (rust-migration-plan.md Sec 7) ----------------
#
# `eigsh` without `v0` draws a RANDOM start vector, so the oracle it computes is not reproducible
# run to run. Measured on the free-free beam: elastic eigenvalues wobble ~1e-12 relative, their
# eigenvectors ~5e-11, and the two rigid-body modes come back as an arbitrary basis of the {1, x}
# nullspace (~1e-1 apart). An eigenvector fed to `set_state` is an INITIAL CONDITION, so that is a
# different trajectory, not a last-digit difference -- and it would read as a port bug. The first
# test asserts the property; the second asserts it cannot be lost by adding a call site, which is
# the shape of guard Sec 17.6 and Sec 23.7 record going quietly empty.


def test_arpack_oracles_are_bit_reproducible():
    from helpers import beam_low_eigenfrequencies, make_beam, make_free_plate
    from helpers import free_plate_low_eigenfrequencies as fp

    beam = make_beam(N=32)
    plate = make_free_plate(N=12)
    for name, first, second in (
        ("beam", beam_low_eigenfrequencies(beam, 4), beam_low_eigenfrequencies(beam, 4)),
        ("free plate", fp(plate, 3), fp(plate, 3)),
    ):
        assert np.array_equal(first, second), (
            f"{name} oracle is not bit-reproducible: {first} vs {second} -- an eigsh call lost "
            "its pinned v0"
        )


def test_every_eigsh_call_in_the_tests_pins_v0():
    import ast
    import pathlib

    here = pathlib.Path(__file__).parent
    unpinned = []
    for path in sorted(here.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if name != "eigsh":
                continue
            if not any(kw.arg == "v0" for kw in node.keywords):
                unpinned.append(f"{path.name}:{node.lineno}")
    assert not unpinned, (
        "eigsh called without a pinned v0 (use helpers.arpack_v0), so the oracle is not "
        f"reproducible run to run: {', '.join(unpinned)}"
    )
