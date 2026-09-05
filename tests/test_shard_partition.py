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


@pytest.mark.parametrize("k", SHARD_COUNTS)
def test_dropping_the_parity_family_does_not_move_anything_between_shards(k):
    """``--exclude-parity`` filters *after* the split, and this is what says so.

    The flagged CI run is the same three shards as the unflagged one with the two-sided comparison
    files removed. If the removal happened before the partition instead, LPT would see a different
    file set and place every remaining file differently -- the two runs would be different
    partitions that happen to cover the same files, and both would be green while a file sat in
    shard 2 of one and shard 3 of the other. Harmless until the two disagree about what exists.

    So the property is *not* "the flagged shards cover the non-parity files", which a
    filter-then-split would also satisfy. It is that each flagged shard is exactly its own
    unflagged shard minus the parity files in it.
    """
    files = shard_tests.test_files()
    shards = shard_tests.partition(files, k)
    # The parity set spelled independently of the script -- by the same glob the CI step names,
    # so this compares two derivations rather than the script against itself.
    parity = {p.name for p in (REPO_ROOT / "tests").glob("test_rust_parity*.py")}
    for shard in shards:
        assert shard_tests.drop_parity(shard) == [f for f in shard if f not in parity]
    covered = sorted(f for shard in shards for f in shard_tests.drop_parity(shard))
    assert covered == sorted(f for f in files if f not in parity)


# The parity family as it stands, written out. Until 2026-09-03 the canary below asserted a
# FLOOR (`len(dropped) >= 20`), which was the right shape while the family only ever grew and the
# wrong one now that it drains: plan section 39's deletions take one file out per unit, so a floor
# has to be lowered by hand on every deletion (which reads as a failure to explain rather than a
# reviewed edit) and, at the end, becomes unsatisfiable -- the family reaches zero, at which point
# "an exclusion that excludes nothing" stops being a bug and starts being the truth.
#
# A written-down set fixes both. It catches drift in BOTH directions (a file that stopped matching
# and a file that started), it makes each deletion a one-line reviewed edit, and it can reach the
# empty set honestly. The canary's actual job -- proving the predicate still fires -- moves to the
# two synthetic probes at the bottom, which keep working after the family is gone.
# `test_rust_parity_connection.py` left with unit 10 (plan section 49), the last file in the
# family that belonged to the deletion graph. The three that remain are not models: two operator
# modules and a solver, none of which has a Python body to delete -- `operators2d` because
# something has to rebuild a matrix from the binding's CSR triplets, `banded` because what it
# ports is a choice of solver rather than a piece of arithmetic. So this set is expected to be
# STABLE now rather than draining, which is a different claim from the one it has carried since
# section 39 and the reason it is worth writing down.
REMAINING_PARITY_FAMILY = {
    "test_rust_parity_banded.py",
    "test_rust_parity_operators.py",
    "test_rust_parity_ops2d.py",
}


def test_the_parity_family_is_exactly_what_is_left():
    """The canary. A scan that silently stopped matching would pass every test above forever.

    Same shape as ``tests/test_ci_workflow.py``'s asserted token count and the ``checks`` job's
    emptiness checks: an exclusion that excludes nothing is indistinguishable from no exclusion,
    and it fails in the direction that stays green -- the flagged run would quietly include the
    parity files again and start comparing Rust against Rust.
    """
    files = shard_tests.test_files()
    dropped = set(files) - set(shard_tests.drop_parity(files))
    assert dropped == REMAINING_PARITY_FAMILY, (
        f"the parity family is {sorted(dropped)}, but this guard expects "
        f"{sorted(REMAINING_PARITY_FAMILY)} -- either the prefix has drifted, or a file was "
        "deleted (plan section 39) without draining this list"
    )
    assert all(f.startswith("test_rust_parity") for f in dropped)

    # The part that outlives the family. `drop_parity`'s predicate is asserted directly on names
    # that are not in `tests/`, so this keeps saying something after the set above is empty --
    # which is the whole reason the floor was replaced rather than lowered.
    assert shard_tests.drop_parity(["test_rust_parity_synthetic.py"]) == []
    assert shard_tests.drop_parity(["test_energy.py"]) == ["test_energy.py"]


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
