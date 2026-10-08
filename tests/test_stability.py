"""The headless-core invariant and the Python package's import and dependency guards.

- physsynth.core imports no plotting/audio library (CLAUDE.md non-negotiable #4).
- The core pulls in nothing outside its allowlist, and nothing from a sibling layer.
- No module in physsynth.core chooses between two implementations of a class.

The guard that every ARPACK call in the tests pinned its start vector was deleted with
``tests/helpers.py`` (retirement plan §46): no Python test calls ``eigsh`` any more, so it
scanned an empty population, and the helper its message named was going too.

The ideal string's stability and construction guards (criterion 4) that opened this file went
native with the rest of its harness (retirement plan §31): they are in
``crates/physsynth-core/tests/string_ideal.rs``.
"""

import importlib
import pkgutil
import subprocess
import sys

import pytest

import physsynth.core


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
#   - It was added when it only appeared under `PHYSSYNTH_RS`, and it has been imported on every
#     path since the model bodies were deleted; phase A removed the flag itself (retirement plan
#     §21). It is still the one name here that is not the numeric stack or its baggage.
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


def test_no_module_chooses_between_two_implementations():
    # The guard that stood here for the whole migration was `test_the_rust_swap_matches_the_
    # environment`: `PHYSSYNTH_RS=1 pytest` claimed to run the suite against Rust, nothing in the
    # tests mentioned Rust, so a mistyped variable or a swap landing after its clients' imports
    # would have been green while testing Python. Phase A (python-retirement-plan §21) deleted the
    # last three modules that read the flag -- `operators` and `exciter` lost their Python bodies,
    # `banded` went whole -- so there is no second implementation for the environment to select,
    # and the guard's three flag-shaped halves went with it rather than being left empty:
    #
    #   * the `_USE_RUST` reader tuple (operators, exciter, banded) had nothing left to read;
    #   * `ported_expected`, the `<name>_py` FUNCTION table, had no aliases left to derive over;
    #   * the `if expected_rust:` captured-binding block compared names that are now one object
    #     with nothing to mis-order -- `string_stiff.biharmonic_matrix is operators.biharmonic_
    #     matrix`, `reed.Bore is bore.Bore` and the rest had become `x is x` (§42.4's rule).
    #
    # What survives is the claim those halves were approximating, WIDENED to the package: no
    # module in `physsynth.core` reads the flag or keeps a reference alias, and each public name
    # that is a Rust object is the Rust object.
    physsynth_rs = pytest.importorskip("physsynth_rs")
    from physsynth.core import (
        airbox,
        beam,
        body,
        bore,
        bow,
        collision,
        connection,
        exciter,
        mallet,
        membrane,
        operators,
        operators2d,
        plate,
        radiation,
        reed,
        string_damped,
        string_geometric,
        string_ideal,
        string_nonlinear,
        string_stiff,
    )

    # The names each module must resolve to the Rust object, listed BY HAND: a deletion is a
    # reviewed edit, and a name that quietly stopped being a re-export must fail here.
    deleted_bodies = {
        string_ideal: {"IdealString"},
        membrane: {"Membrane"},
        # `MalletPlate` was never written in Python at all; a model born in Rust must not be able
        # to slip in without the same claim.
        mallet: {"MalletMembrane", "MalletPlate", "MalletWall"},
        bore: {"Bore"},
        reed: {"ReedBore", "bernoulli_flow"},
        body: {"ModalBody"},
        radiation: {
            "AirRadiation",
            "RadiatedBody",
            "RationalAirLoad",
            "ReactiveRadiatedBody",
            "monopole_radiation_resistance",
            "piston_radiation_resistance",
        },
        # The module keeps three SciPy names that this loop cannot speak about, because they are
        # not Rust objects; their guard is in `tests/test_binding_surface.py` (plan section 49.2).
        connection: {
            "StringBodyBridge",
            "StringPlateBridge",
            "StringVKPlateBridge",
            "SympatheticStrings",
        },
        # The seams are included because `airbox_wrap.rs` reads them off this module's namespace
        # by name, and a wrong one there is a silently different seam.
        airbox: {
            "AirBox",
            "InteriorSurfacePort",
            "RoomLoadedBody",
            "RoomLoadedMembrane",
            "RoomLoadedPlate",
            "RoomLoadedVKPlate",
            "RoomPort",
            "RoomSuspendedMembrane",
            "RoomSuspendedPlate",
            "RoomSuspendedVKPlate",
            "SurfacePort",
            "_MembraneSurface",
            "_PlateSurface",
            "_VKPlateSurface",
            "impedance_from_zeta",
        },
        string_stiff: {"StiffString"},
        string_damped: {"DampedStiffString"},
        string_nonlinear: {"TensionModulatedString"},
        string_geometric: {"GeometricString"},
        bow: {"BowedString", "friction_smooth", "friction_smooth_deriv"},
        collision: {
            "BarrierString",
            "contact_potential",
            "contact_force_elastic",
            "contact_stiffness",
            "contact_force_dg",
            "contact_force_total",
            "solve_contact",
        },
        # `GrainSpec` stays Python because the Rust helper CONSTRUCTS it, reaching back through
        # `py.import("physsynth.core.plate")`.
        plate: {"Plate", "VKPlate", "grain_ratios_from_material"},
        beam: {"FreeBeam"},
        operators2d: {"VonKarmanBracket", "AiryStressSolver"},
        # Phase A. The binding's signatures are the deleted body's, keywords and defaults included,
        # so these are bare re-exports.
        exciter: {"triangular_pluck", "raised_cosine", "raised_cosine_2d"},
    }
    for module, names in deleted_bodies.items():
        for name in sorted(names):
            assert getattr(module, name) is getattr(physsynth_rs, name), (
                f"{module.__name__}.{name} must be the Rust object"
            )

    # The DELEGATING modules, where the inverse is the claim. The binding hands every matrix back
    # as CSR triplets -- a Rust crate cannot construct a `scipy.sparse.csr_matrix` -- so each
    # builder stays a Python function that puts a matrix back around the result, and `operators`
    # also coerces its inputs to what NumPy would have accepted. Asserting `is physsynth_rs.<name>`
    # would be asserting that the shim had been bypassed and a caller was being handed a 4-tuple.
    delegating = {
        operators2d: {
            "grid_coords", "rectangle_mask", "disk_mask", "guitar_half_width", "guitar_scale",
            "guitar_mask", "guitar_area", "live_cells", "cells_per_node", "prune_to_area_carrying",
            "laplacian_from_mask", "biharmonic_from_mask", "_dirichlet_interior_d2_1d",
            "orthotropic_biharmonic", "free_plate_stiffness", "free_plate_stiffness_from_mask",
            "_collocated_d2_1d", "_forward_d1_1d", "_centered_d2_1d", "_clamped_d2_1d",
            "_avg_d1_1d", "embed", "inner2d", "norm2_2d",
        },
        operators: set(operators.__all__),
    }
    for module, names in delegating.items():
        for name in sorted(names):
            fn = getattr(module, name)
            assert fn is not getattr(physsynth_rs, name.lstrip("_"), None), (
                f"`{module.__name__}.{name}` is the Rust function itself -- the delegating "
                "wrapper has been bypassed"
            )
            assert getattr(fn, "__module__", None) == module.__name__, (
                f"`{module.__name__}.{name}` is no longer defined in this module ({fn!r}) -- the "
                "wrapper is the module's whole remaining body and cannot be re-exported away"
            )
    # `collision`'s underscored names, and the one that must NOT be a bare re-export.
    for name in ("_contact_force_total_deriv", "_force_total_vec", "_deriv_total_vec"):
        assert getattr(collision, name) is getattr(physsynth_rs, name[1:]), (
            f"`collision.{name}` must be the Rust function -- the underscored spelling is the one "
            "the model and the mallet reach for"
        )
    assert collision.solve_contact_vector is not physsynth_rs.solve_contact_vector, (
        "`collision.solve_contact_vector` must stay a Python wrapper: `stacklevel=2` on its "
        "non-convergence warning cannot mean the same thing from inside an extension module"
    )

    # The package-wide half, DERIVED rather than listed: `pkgutil` reads the directory, so a module
    # added tomorrow is in the population without anyone remembering it (ledger #67). Before phase
    # A the flag check covered a three-module tuple and the alias check covered the modules in
    # `deleted_bodies`; both now cover everything, which is what makes them worth more at zero.
    all_core_modules = [
        importlib.import_module(f"physsynth.core.{info.name}")
        for info in pkgutil.iter_modules(physsynth.core.__path__)
        if not info.ispkg
    ]
    # Named POSITIVE CONTROLS rather than a floor (finding #61): an `iter_modules` over the wrong
    # path yields nothing, and every assertion below would then pass over an empty list.
    for named in (connection, plate, string_ideal, operators):
        assert named in all_core_modules, (
            f"`{named.__name__}` is not in the scanned set, so `pkgutil.iter_modules` is not "
            "reaching `physsynth/core/` and this guard is checking nothing"
        )
    for module in all_core_modules:
        assert not hasattr(module, "_USE_RUST"), (
            f"{module.__name__} reads PHYSSYNTH_RS again -- the flag chose between two "
            "implementations, and phase A left one"
        )
        leftovers = [a for a in dir(module) if a.endswith("Py") or a.endswith("_py")]
        assert not leftovers, (
            f"{module.__name__} defines the reference alias(es) {sorted(leftovers)} -- either a "
            "Python reference implementation came back, or an alias outlived its body"
        )


def test_core_does_not_import_sibling_layers():
    # The dependency arrow points one way: analysis/io depend on core, never the reverse.
    body = (
        "bad={'physsynth.analysis','physsynth.io'};"
        "hit=sorted(m for m in sys.modules if any(m==b or m.startswith(b+'.') for b in bad));"
        "print(','.join(hit));"
        "sys.exit(1 if hit else 0)"
    )
    result = _run_core_probe(body)
    assert result.returncode == 0, (
        f"core imported a sibling layer (must not): {result.stdout.strip()}"
    )
