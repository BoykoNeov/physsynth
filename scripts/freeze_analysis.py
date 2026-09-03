"""Record what ``physsynth/analysis/``'s **Python** implementations measure, as literals.

Run once, before those implementations are deleted (``docs/dev/rust-migration-plan.md`` §44):

    python scripts/freeze_analysis.py

It writes ``tests/analysis_frozen_values.py`` — a generated table of every case in
``tests/analysis_frozen_cases.py``, evaluated through the ``*_py`` reference names, together with
the gap between the Python and Rust answers **as measured on the machine that generated it**.

Why the gap is recorded and not asserted from here: the frozen test compares one implementation
against numbers a *different* implementation produced on a *different machine*, so an equality
would be a claim about two things at once (ledger #28). The recorded gap is what makes the
tolerances in ``tests/test_analysis_frozen.py`` reviewable — a bar with no measurement behind it is
a guess.

Re-running this on an unchanged tree must produce an unchanged file. If it does not, something in
the fixtures is reading the clock or the environment, and that is a bug in the fixtures.
"""

from __future__ import annotations

import datetime
import importlib
import json
import os
import pathlib
import subprocess
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tests"))

import analysis_frozen_cases as C  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "tests" / "analysis_frozen_values.py"


def python_impl(module: str, name: str):
    """The Python reference under the alias the swap never rebinds."""
    mod = importlib.import_module("physsynth.analysis." + module)
    return getattr(mod, name + "_py")


# The Rust side is measured in a SUBPROCESS with `PHYSSYNTH_RS_ANALYSIS=1`, calling the same
# PUBLIC names the callers call. Two reasons, and the second is the important one:
#
#   * the binding's names are not a mechanical transform of the Python ones --
#     `free_free_beam_betaL` is `modal_free_free_beam_beta_l`, `solve_rotating_wave` is
#     `rotating_wave_solve`,
#     `stiff_dispersion_frequencies` is `dispersion_stiff_frequencies` -- so guessing them silently
#     reported "no Rust twin" for fifteen cases that have one;
#   * the swap is not only a rebinding. Several of these public names are thin Python wrappers that
#     adapt arguments before calling the extension, and it is the *wrapper* that callers reach. A
#     gap measured against the raw binding would be measuring something nobody runs.
#
# This is exactly the configuration the deletion retires -- a Rust instrument reached through the
# flag -- so the number recorded is the last honest reading of it.

_PROBE = r"""
import json, sys
sys.path.insert(0, %(root)r)
sys.path.insert(0, %(tests)r)
import analysis_frozen_cases as C
import importlib


def public(module, name):
    return getattr(importlib.import_module("physsynth.analysis." + module), name)


out = {}
for key in C.KEYS:
    try:
        shape, floats, ints = C.flatten(C.call(key, public))
    except Exception as exc:  # recorded, never swallowed
        out[key] = {"error": f"{type(exc).__name__}: {exc}"}
        continue
    out[key] = {"shape": shape, "floats": floats, "ints": ints}
print(json.dumps(out))
"""


def rust_values() -> dict:
    """Every case through the public names, with ``PHYSSYNTH_RS_ANALYSIS=1``."""
    env = dict(os.environ, PHYSSYNTH_RS_ANALYSIS="1")
    src = _PROBE % {"root": str(_ROOT), "tests": str(_ROOT / "tests")}
    done = subprocess.run(
        [sys.executable, "-c", src], capture_output=True, text=True, env=env, cwd=str(_ROOT)
    )
    if done.returncode != 0:
        raise SystemExit(f"the Rust probe failed:\n{done.stderr}")
    return json.loads(done.stdout)


def gap(py_floats: list[float], rs: dict) -> float | str | None:
    """``max|python - rust| / max|python|``, or a string saying why there is no number."""
    if "error" in rs:
        return rs["error"]
    if len(py_floats) != len(rs["floats"]):
        return f"structure differs: {len(py_floats)} floats vs {len(rs['floats'])}"
    scale = max((abs(v) for v in py_floats), default=0.0)
    worst = max((abs(a - b) for a, b in zip(py_floats, rs["floats"], strict=True)), default=0.0)
    return worst if scale == 0.0 else worst / scale


def main() -> None:
    rs_all = rust_values()
    lines = [
        '"""Frozen output of the PYTHON analysis oracles — GENERATED, do not edit by hand.',
        "",
        "Regenerate with ``python scripts/freeze_analysis.py``, which only works while the Python",
        "implementations still exist. After units 10 and 11 are deleted they do not, and that is",
        "the whole point of this file: it is the last thing a second implementation of these",
        "oracles ever said, kept so the Rust one can go on being checked against it.",
        "",
        "``gap`` on each row is ``max|python - rust| / max|python|`` as measured on the machine",
        "that generated the file, with the Rust side reached through ``PHYSSYNTH_RS_ANALYSIS=1``",
        "and the same public names the callers call. It is a **record, not a bar** — the bars live",
        "in ``tests/test_analysis_frozen.py`` and are chosen with these numbers in front of you,",
        "because a bar with no measurement behind it is a guess. A string there means the two",
        "sides could not be compared at all, and says why.",
        '"""',
        "",
        "# ruff: noqa: E501",
        "",
        f"GENERATED = {datetime.date.today().isoformat()!r}",
        "",
        "# key -> (structure, floats, ints, measured gap against Rust at generation time)",
        "FROZEN: dict[str, tuple[str, list[float], list[int], float | str]] = {",
    ]
    n_f = n_i = 0
    report = []
    for key in C.KEYS:
        shape, floats, ints = C.flatten(C.call(key, python_impl))
        g = gap(floats, rs_all[key])
        n_f += len(floats)
        n_i += len(ints)
        report.append((key, g, rs_all[key].get("shape"), shape))
        lines.append(f"    {key!r}: (")
        lines.append(f"        {shape!r},")
        lines.append(f"        {floats!r},")
        lines.append(f"        {ints!r},")
        lines.append(f"        {g!r},")
        lines.append("    ),")
    lines.append("}")
    lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"{OUT}: {len(C.KEYS)} cases, {n_f} floats, {n_i} ints")

    numeric = sorted(((g, k) for k, g, _, _ in report if isinstance(g, float)), reverse=True)
    print("worst measured gaps:")
    for g, k in numeric[:10]:
        print(f"  {g:.3e}  {k}")
    print(f"  ... and {sum(1 for g, _ in numeric if g == 0.0)} of {len(numeric)} are exactly 0.0")
    bad = [(k, g) for k, g, _, _ in report if not isinstance(g, float)]
    for k, g in bad:
        print(f"NOT COMPARABLE  {k}: {g}")
    mism = [(k, a, b) for k, _, a, b in report if a is not None and a != b]
    for k, a, b in mism:
        print(f"STRUCTURE DIFFERS  {k}: rust {a} vs python {b}")


if __name__ == "__main__":
    main()
