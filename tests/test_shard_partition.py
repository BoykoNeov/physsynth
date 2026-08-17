"""The CI shard split covers the suite exactly once -- the guard that lets the gate be split at all.

The validation harness is bulk-bound, not bottleneck-bound (measured 2026-08-17: ~80-85% of a
4-core runner's budget is genuinely busy), so the only lever on its wall clock is more cores, and
more cores means several concurrent jobs each running part of the suite. Splitting a gate is exactly
the kind of change that can go *silently* wrong: a file in no shard is a file that never runs, and
every job still reports green. This repo has already paid for that failure shape twice.

:mod:`scripts.shard_tests` is written so the failure is impossible by construction -- it partitions
``glob("tests/test_*.py")`` rather than a list anyone maintains. This file is the assertion that the
construction actually holds, for every shard count the workflow might use, and it is deliberately
about *coverage* and never about *balance*: balance depends on a cost table that is documented as a
hint and allowed to go stale, and a test that asserted balance would fail the first time someone
added an expensive model without re-profiling.

The outer half of the guard cannot live here -- if the partition ever dropped *this* file, this test
would not run to complain. That half is the ``reconcile`` job in ``.github/workflows/ci.yml``, which
compares pytest's own collected count per shard against the count for the whole suite.
"""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "shard_tests", REPO_ROOT / "scripts" / "shard_tests.py"
)
shard_tests = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(shard_tests)

SHARD_COUNTS = [1, 2, 3, 4, 5]
"""Every split worth checking, not just the one the workflow uses today.

Changing the matrix width is a one-character edit in the YAML and nobody would think to come back
here for it, so the property is asserted across a range instead of at a point.
"""


@pytest.mark.parametrize("k", SHARD_COUNTS)
def test_every_test_file_lands_in_exactly_one_shard(k):
    """The whole point: the partition is total and disjoint over the files that actually exist."""
    files = shard_tests.test_files()
    shards = shard_tests.partition(files, k)

    flat = [f for shard in shards for f in shard]
    assert sorted(flat) == sorted(files), "a test file is missing from the split, or duplicated"
    assert len(flat) == len(set(flat)), "a test file appears in more than one shard"


@pytest.mark.parametrize("k", SHARD_COUNTS)
def test_no_shard_is_empty(k):
    """An empty shard is a runner spun up to do nothing -- cheap to check, easy to introduce."""
    shards = shard_tests.partition(shard_tests.test_files(), k)
    assert all(shards), f"an empty shard at k={k}: {[len(s) for s in shards]}"


def test_the_split_is_a_pure_function_of_its_inputs():
    """Each job recomputes the split independently, so two computations MUST agree.

    Nothing passes shard membership between jobs -- there is no artifact and no ordering. That only
    works because the partition is deterministic: LPT over a cost-then-name sort, with ties broken
    on the filename rather than on dict order.
    """
    files = shard_tests.test_files()
    assert shard_tests.partition(files, 3) == shard_tests.partition(files, 3)


def test_a_file_the_cost_table_has_never_seen_is_still_placed():
    """The load-bearing case: a test file added *after* the last profile.

    An unpriced file must be scheduled on a guess, never skipped -- guessing wrong costs wall clock,
    skipping loses coverage. This is the property that makes the stale cost table safe.
    """
    files = shard_tests.test_files() + ["test_a_model_that_does_not_exist_yet.py"]
    shards = shard_tests.partition(files, 3, costs={f: 1.0 for f in shard_tests.test_files()})
    placed = [f for shard in shards for f in shard]
    assert "test_a_model_that_does_not_exist_yet.py" in placed


def test_more_shards_than_files_is_rejected_rather_than_silently_emptied():
    """A k above the file count cannot be honoured; failing loudly beats returning empty shards."""
    with pytest.raises(ValueError):
        shard_tests.partition(["test_one.py", "test_two.py"], 3)


def test_the_heaviest_files_are_spread_and_not_stacked():
    """LPT's one real job: the expensive files must not pile onto the same shard.

    Asserted on a synthetic cost table and not the measured one -- this is a property of the
    *algorithm* (heaviest-first onto the lightest shard), and pinning it to the real profile would
    turn a scheduling hint into a claim that goes stale the next time the suite grows.
    """
    files = [f"test_{i}.py" for i in range(6)]
    costs = {"test_0.py": 100.0, "test_1.py": 100.0, "test_2.py": 100.0}
    shards = shard_tests.partition(files, 3, costs=costs)
    for shard in shards:
        assert sum(1 for f in shard if costs.get(f, 0.0) == 100.0) == 1
