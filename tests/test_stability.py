"""Stability and construction guards (criterion 4) + the headless-core invariant.

- No NaN / blow-up across a sweep of valid lambda in (0, 1].
- A deliberate lambda > 1 is rejected at construction time, not silently run into an overflow.
- Non-physical parameters are rejected.
- physsynth.core imports no plotting/audio library (CLAUDE.md non-negotiable #4).
"""

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
# exactly {numpy, scipy, charset_normalizer, cython_runtime, <hash>__mypyc}; a bare interpreter
# pulls none of these. Anything outside this set (torch, requests, PIL, sounddevice, ...) is a real
# portability leak and must fail the test. The mypyc runtime is named with a per-build hash prefix
# (e.g. "81d243...__mypyc"), so it is matched structurally by its "__mypyc" suffix, not by name.
_CORE_DEP_ALLOWLIST = {"numpy", "scipy", "charset_normalizer", "cython_runtime", "physsynth"}


def test_core_dependency_allowlist():
    # Stronger than the blocklist above: the core may use ONLY the allowlisted numeric stack and
    # its compiled-runtime baggage -- no third-party dependency of its own. Import every core
    # submodule (auto-discovers new ones, e.g. string_stiff) and assert nothing outside the
    # allowlist appears. Underscore-private modules (_csparsetools, editable-install finders, ...)
    # are internal plumbing, excluded by the leading-underscore rule; the hash-suffixed mypyc
    # runtime is excluded by its "__mypyc" suffix.
    allowed = sorted(_CORE_DEP_ALLOWLIST)
    probe = (
        "import sys, importlib, pkgutil;"
        "stdlib=set(sys.stdlib_module_names)|set(sys.builtin_module_names);"
        "import physsynth.core as _core;"
        "[importlib.import_module(m.name) "
        " for m in pkgutil.iter_modules(_core.__path__, _core.__name__ + '.')];"
        "allowed=set(" + repr(allowed) + ");"
        "tp={n.split('.')[0] for n in list(sys.modules)"
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
