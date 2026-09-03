"""The analysis oracles against numbers a **different implementation** produced.

This file is what replaces ``PHYSSYNTH_RS_ANALYSIS`` when units 10 and 11 are deleted
(``docs/dev/rust-migration-plan.md`` §44), and it is worth being exact about what it does and does
not preserve.

**What §36.4 built the second flag for.** ``physsynth/analysis/`` is the *instrument* — the thing
that measures every model in this project. The acceptance run sets ``PHYSSYNTH_RS`` and not
``PHYSSYNTH_RS_ANALYSIS``, so a Rust model is read by a Python ruler against an unmoved oracle. The
point is that a misreading shared by a model and its detector would **cancel**, and two
implementations cannot easily share one. Widening a single flag over both would have destroyed
that, which is why the two were never merged.

**What deleting the Python instrument costs, precisely.** The check itself has been made and
passed; what goes is the ability to *re-derive* it. After the deletion there is one implementation
of these oracles, so nothing can independently recompute what they should say. This file keeps the
next best thing: 62 fixtures' worth of what the Python implementation actually said, recorded to
the last digit, and asserted against for as long as the project lasts. It catches a transcription
error, a wrong branch, a changed convention and a regression. It cannot catch an error the Python
made too, and neither could the flag — that is what
``crates/physsynth-analysis/tests/`` is for, where the oracles are checked against their
mathematical definitions rather than against a second spelling.

**The bar is one number and it is the plan's Group A target**, 1e-13, on the gap normalised by the
field's own scale. The measured gaps are recorded per case in ``analysis_frozen_values.py`` — 51 of
62 were exactly zero at generation and the worst was 3.5e-15 (``duffing_frequency_shift``, a
difference of two nearly equal frequencies, a scar already on the record). So the bar carries about
28x headroom on the worst case. It is deliberately **not** an equality: this compares a Rust answer
computed here against a Python answer computed on another machine at another time, and ledger #28
is the standing warning that a cross-machine bit-identity claim is a claim about the runner.

Structure and integers **are** compared exactly, and that is where the sharp teeth are. A root
search that returns a different number of roots, a spectrum that comes back with a different
multiplicity, a solver that reports ``converged=False``, a mode-count that shifts — none of those is
a small error and no tolerance can describe one.
"""

from __future__ import annotations

import importlib

import analysis_frozen_cases as cases
import pytest
from analysis_frozen_values import FROZEN

# The plan's Group A agreement target. One bar, applied to every case, on the amplitude-normalised
# gap -- see the module docstring for why it is not an equality and why it is not per-case.
BAR = 1e-13

ANALYSIS_MODULES = ("modal", "damping", "dispersion", "duffing", "spectrum", "rotating_wave")


def _public(module: str, name: str):
    """The name a caller reaches -- whatever it is bound to now."""
    return getattr(importlib.import_module("physsynth.analysis." + module), name)


@pytest.mark.parametrize("key", cases.KEYS)
def test_the_frozen_case_still_reads_the_same(key):
    want_shape, want_floats, want_ints, recorded = FROZEN[key]
    shape, floats, ints = cases.flatten(cases.call(key, _public))

    assert shape == want_shape, (
        f"{key}: the shape of the answer changed, {shape} against the frozen {want_shape}. That is "
        "a different number of roots, partials or fields -- not a numerical drift, and no "
        "tolerance describes it"
    )
    assert ints == want_ints, (
        f"{key}: an integer or boolean in the answer changed -- {ints} against {want_ints}. Mode "
        "counts, multiplicities, iteration flags and `converged` live here"
    )
    scale = max((abs(v) for v in want_floats), default=0.0)
    worst = max((abs(a - b) for a, b in zip(floats, want_floats, strict=True)), default=0.0)
    rel = worst if scale == 0.0 else worst / scale
    assert rel < BAR, (
        f"{key}: {rel:.3e} of the field's own scale away from what the Python oracle said "
        f"(recorded gap at generation: {recorded!r}). The bar is {BAR:.0e}"
    )


def test_every_public_analysis_function_is_frozen():
    """Derived, not listed: a new oracle cannot be added without being frozen at the same time.

    The set comes from each module's ``__all__``, so it stays right when a function is added or
    renamed. This is the guard that keeps the file above from quietly covering less of the
    instrument than it did -- the same reasoning as the swap guard's derived class set, and the
    same reason it is checked against something rather than merely collected.
    """
    frozen = {k.split("#")[0] for k in cases.KEYS}
    missing = []
    for module in ANALYSIS_MODULES:
        mod = importlib.import_module("physsynth.analysis." + module)
        for name in mod.__all__:
            obj = getattr(mod, name)
            # `RotatingWave` is a result type, exercised through `solve_rotating_wave`; a measured
            # constant (`T60_SECONDS_PER_RATE`) is a number in the source, not an answer to freeze.
            if isinstance(obj, type) or not callable(obj):
                continue
            if f"{module}.{name}" not in frozen:
                missing.append(f"{module}.{name}")
    assert not missing, (
        f"these analysis functions have no frozen case: {sorted(missing)}. Add one to "
        "`tests/analysis_frozen_cases.py` and regenerate with `python scripts/freeze_analysis.py` "
        "-- which only works while a second implementation exists, so if the Python bodies are "
        "already gone the honest options are a native bar in `crates/physsynth-analysis/tests/` or "
        "nothing"
    )


def test_every_case_carries_a_measured_gap_rather_than_a_reason_it_could_not_be_measured():
    """The canary on the record itself.

    Each row's fourth entry is the Python-versus-Rust gap measured when the file was generated. A
    *string* there means the two sides could not be compared -- a missing binding, a differing
    structure -- and the generator writes the reason rather than dropping the case. If any appear,
    the freeze covered less than it looks like it covers.
    """
    unmeasured = {k: v[3] for k, v in FROZEN.items() if not isinstance(v[3], float)}
    assert not unmeasured, f"cases frozen without a comparison behind them: {unmeasured}"


def test_the_frozen_file_covers_exactly_the_cases_module():
    """Neither file may drift from the other: a case with no numbers never runs, and numbers with
    no case are a measurement of something that is no longer asked for."""
    assert set(FROZEN) == set(cases.KEYS), (
        f"only in the values file: {sorted(set(FROZEN) - set(cases.KEYS))}; "
        f"only in the cases file: {sorted(set(cases.KEYS) - set(FROZEN))}"
    )


# -- a frozen BOOL is only as safe as its margin --------------------------------------------------


def test_the_underdamped_predicate_has_room_to_spare():
    """A bool is a discrete output, so freezing one is a claim about how far it is from flipping.

    Inherited from ``test_rust_parity_analysis.py``'s
    ``test_the_underdamped_predicate_agrees_and_its_margin_is_reported``, which was deleted with
    unit 10. There the two implementations' verdicts were compared and the margin reported; here
    there is one implementation and a *recorded* verdict, so the margin is the whole of what makes
    the record trustworthy — a frozen `True` next to a discriminant one ulp from zero would be a
    coin toss written down as a fact.

    Observed over these five configurations and modes 1..40: the discriminant ``b² - 4ac`` is never
    within eleven orders of magnitude of zero relative to ``|b²| + |4ac|``, so no rounding can flip
    it. The bar is that measurement, and a future fixture that narrows it fails here rather than
    silently answering differently.
    """
    c, ell, n, kappa, k, theta = 200.0, 0.65, 128, 0.7, 1e-5, 0.5
    worst = float("inf")
    saw_both = set()
    for sigma0, sigma1 in ((0.0, 0.0), (0.5, 1e-5), (50.0, 1e-3), (1e6, 0.0), (1e5, 1.0)):
        for m in range(1, 41):
            saw_both.add(
                _public("damping", "discrete_damped_mode_is_underdamped")(
                    c, ell, n, kappa, k, theta, sigma0, sigma1, m
                )
            )
            p2 = _public("damping", "spatial_eigenvalue_p2")(n, ell / n, m)
            q_mode = c * c * p2 + kappa * kappa * p2 * p2
            base = 1.0 + theta * k * k * q_mode
            aa = base + (sigma0 + sigma1 * p2) * k
            bb = -2.0 + (1.0 - 2.0 * theta) * k * k * q_mode
            cc = base - (sigma0 + sigma1 * p2) * k
            disc = bb * bb - 4.0 * aa * cc
            scale = abs(bb * bb) + abs(4.0 * aa * cc)
            worst = min(worst, abs(disc) / scale)
    assert worst > 1e-11, f"the underdamped margin has narrowed to {worst:.3e}"
    assert saw_both == {True, False}, (
        f"this sweep only ever answers {saw_both} -- it no longer exercises both arms of the "
        "predicate, so the margin it measures is a margin on one of them"
    )
