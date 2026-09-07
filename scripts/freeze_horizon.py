"""Record what ``tests/helpers.py``'s **Python** resolution-horizon helpers measure, as literals.

Run once, in the batch that promotes them to ``physsynth/analysis/horizon.py``
(``docs/dev/resolution-horizon-plan.md`` §6), and *before* the commit that replaces their bodies::

    python scripts/freeze_horizon.py

The sibling of ``scripts/freeze_analysis.py`` and deliberately a separate file rather than an
extension of it. That one reads ``*_py`` reference aliases and drives the Rust side with
``PHYSSYNTH_RS_ANALYSIS=1``; neither exists any more, so it cannot run at all. This one has a
different Python side (a test-folder module, not a swap alias) and a different Rust side (the shim,
reached directly, because there is no flag to set). Merging them would mean one script with two
dead halves.

# Why this is possible at all, when §6 said it was not

That bullet says freezing a promoted helper "is impossible — there is no Python implementation left
to freeze against". True of the six modules deleted in ``docs/dev/rust-migration-plan.md`` §44, and
false of these seven functions: they were never deleted, they were *promoted*, so their Python
bodies were live in ``tests/helpers.py`` up to the moment this script ran. The window is one commit
wide and this script is what walks through it.

# What that means for re-running

Like ``freeze_analysis.py``, this becomes unrunnable the moment its subject is replaced, and it
says so rather than producing a half-file. It is kept because a generated file must name the thing
that generated it — ``tests/analysis_frozen_values.py`` now has two provenances and its header says
which rows came from where.

The existing (non-horizon) rows are re-emitted verbatim from the current file: their Python side is
gone and cannot be recomputed, so they are carried through rather than regenerated. Python's
``repr`` round-trips a float exactly, so "carried through" is bit-for-bit and not an approximation.
"""

from __future__ import annotations

import datetime
import importlib
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tests"))

import analysis_frozen_cases as C  # noqa: E402
from analysis_frozen_values import FROZEN as EXISTING  # noqa: E402

OUT = _ROOT / "tests" / "analysis_frozen_values.py"

HORIZON_PREFIX = "horizon."


def python_impl(module: str, name: str):
    """The pre-promotion Python body, which lives in the test folder rather than the package."""
    if module != "horizon":
        raise SystemExit(f"this generator only knows the horizon cases, not {module}.{name}")
    helpers = importlib.import_module("helpers")
    fn = getattr(helpers, name, None)
    if fn is None:
        raise SystemExit(
            f"`helpers.{name}` is gone, so there is no second implementation to record. This "
            "script had one commit in which to run and that commit has passed; the numbers it "
            "wrote are in tests/analysis_frozen_values.py and are the only copy."
        )
    if getattr(fn, "__module__", "") != "helpers":
        raise SystemExit(
            f"`helpers.{name}` is no longer defined in helpers.py — it resolves to "
            f"{fn.__module__}.{name}, which is the promoted implementation. Recording that against "
            "itself would freeze a tautology. See the module docstring."
        )
    return fn


def rust_impl(module: str, name: str):
    """The promoted public name — the shim over the binding, which is what callers reach."""
    return getattr(importlib.import_module("physsynth.analysis." + module), name)


def gap(py_floats: list[float], rs_floats: list[float]) -> float | str:
    """``max|python - rust| / max|python|``, or a string saying why there is no number."""
    if len(py_floats) != len(rs_floats):
        return f"structure differs: {len(py_floats)} floats vs {len(rs_floats)}"
    if not py_floats:
        return "no floats in this answer -- the comparison is the exact int/structure one"
    scale = max(abs(v) for v in py_floats)
    worst = max(abs(a - b) for a, b in zip(py_floats, rs_floats, strict=True))
    return worst if scale == 0.0 else worst / scale


HEADER = '''"""Frozen output of the analysis oracles — GENERATED, do not edit by hand.

**Two provenances, and the difference matters.**

* Rows whose key does **not** begin ``horizon.`` were written by ``scripts/freeze_analysis.py``
  from the Python implementations in ``physsynth/analysis/`` before they were deleted
  (``docs/dev/rust-migration-plan.md`` §44). That generator can no longer run — it reads ``*_py``
  aliases and drives the Rust side with ``PHYSSYNTH_RS_ANALYSIS=1``, and neither survives — so
  those rows are carried forward verbatim by whatever regenerates this file. Python's ``repr``
  round-trips a float exactly, so carrying them forward is bit-for-bit.
* Rows beginning ``horizon.`` were written by ``scripts/freeze_horizon.py`` from the Python bodies
  in ``tests/helpers.py``, in the batch that promoted them into ``physsynth/analysis/horizon.py``
  (``docs/dev/resolution-horizon-plan.md`` §6). That generator can no longer run either, and for
  the same reason: its subject was replaced by the shim in the commit after it ran.

``gap`` on each row is ``max|python - rust| / max|python|`` as measured on the machine that
generated it. It is a **record, not a bar** — the bars live in ``tests/test_analysis_frozen.py``
and are chosen with these numbers in front of you, because a bar with no measurement behind it is
a guess. A string there means the two sides could not be compared as floats at all, and says why.
Five of the horizon rows are that case by construction: `pitch_horizon` returns `(int, bool)` and
the two index builders return lists of integer pairs, so there is nothing float to measure and the
whole comparison is the exact one on `ints` and `structure`. Do NOT read those rows as bit-
identity between two float computations.
"""

# ruff: noqa: E501
'''


def main() -> None:
    rows: dict[str, tuple] = {}
    report = []

    for key, value in EXISTING.items():
        if not key.startswith(HORIZON_PREFIX):
            rows[key] = value

    for key in C.KEYS:
        if not key.startswith(HORIZON_PREFIX):
            if key not in rows:
                raise SystemExit(
                    f"{key} is a case with no frozen row and no way to make one: its Python side "
                    "was deleted in rust-migration-plan §44."
                )
            continue
        shape, floats, ints = C.flatten(C.call(key, python_impl))
        rs_shape, rs_floats, rs_ints = C.flatten(C.call(key, rust_impl))
        g = gap(floats, rs_floats)
        rows[key] = (shape, floats, ints, g)
        report.append((key, g, rs_shape, shape, ints == rs_ints))

    lines = [HEADER, f"GENERATED = {datetime.date.today().isoformat()!r}", ""]
    lines.append("# key -> (structure, floats, ints, measured gap against Rust at generation time)")
    lines.append("FROZEN: dict[str, tuple[str, list[float], list[int], float | str]] = {")
    for key in C.KEYS:
        shape, floats, ints, g = rows[key]
        lines.append(f"    {key!r}: (")
        lines.append(f"        {shape!r},")
        lines.append(f"        {floats!r},")
        lines.append(f"        {ints!r},")
        lines.append(f"        {g!r},")
        lines.append("    ),")
    lines.append("}")
    lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")

    print(f"{OUT}: {len(C.KEYS)} cases, {len(report)} of them newly recorded")
    for key, g, rs_shape, py_shape, ints_agree in report:
        flag = "" if ints_agree else "   <-- INTEGERS DISAGREE"
        note = "" if rs_shape == py_shape else f"   <-- STRUCTURE {rs_shape} vs {py_shape}"
        shown = f"{g:.3e}" if isinstance(g, float) else g
        print(f"  {shown:>40}  {key}{flag}{note}")


if __name__ == "__main__":
    main()
