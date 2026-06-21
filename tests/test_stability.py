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


def test_core_dependency_allowlist():
    # Stronger than the blocklist above: after importing the whole core, the ONLY third-party
    # top-level packages allowed in sys.modules are the permitted numeric ones. This catches any
    # future leak (torch, requests, PIL, ...), not just the named offenders.
    body = (
        "allowed={'physsynth','numpy','scipy'};"
        "stdlib=set(sys.stdlib_module_names);"
        "builtin=set(sys.builtin_module_names);"
        "tops={n.split('.')[0] for n in sys.modules};"
        "third=sorted(t for t in tops if t not in allowed and t not in stdlib "
        "and t not in builtin and not t.startswith('_'));"
        "print(','.join(third));"
        "sys.exit(1 if third else 0)"
    )
    result = _run_core_probe(body)
    assert result.returncode == 0, (
        f"core pulled in non-allowlisted third-party module(s): {result.stdout.strip()}"
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
