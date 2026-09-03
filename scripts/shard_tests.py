"""Split the test suite across concurrent CI jobs -- by *derivation*, never by a hand-written list.

The gate is one job on a 4-core runner, so its wall clock is (total core-seconds)/4 plus packing
slack. Measured on 2026-08-17 that budget is ~80-85% full, which means the two
``xdist_group``-pinned modules everyone points at are *not* what sets the clock -- the bulk is.
The only lever that moves a bulk-bound suite is more cores, and on a public repo more cores are
free: N concurrent jobs on N standard runners. This script decides which files each job runs.

**The whole design constraint is that a new test file must never be able to fall outside the
split.** A file list in the workflow YAML would do exactly that, silently, and the run would be
green -- the same shape as the failure this repo has already paid for twice (20 unread red runs; a
red gate sitting on ``main`` for 90 minutes). So the partition is computed from ``glob`` over
``tests/`` at job time: every file that exists lands in exactly one shard *by construction*, and the
worst a file the cost table has never seen can do is unbalance a shard. Unbalanced is slow;
unbalanced is not untested.

That is also why the cost table below is documented as a **scheduling hint and not a claim**. It is
a measurement, and this repo's rule is that measurements in config go stale -- so nothing here
asserts anything about it. If it drifts, the shards drift out of balance and the wall clock creeps
back up; correctness cannot notice, because coverage does not depend on a single number in it.

Usage (the CI matrix does this)::

    pytest -n auto --dist loadgroup $(python scripts/shard_tests.py --shard 1 --of 3)

``--list`` prints the whole partition with its per-shard cost, for eyeballing the balance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent.parent / "tests"
COSTS_PATH = Path(__file__).resolve().parent / "shard_costs.json"

SHARDS = 3
"""How many concurrent jobs the gate is split across -- **the single place this number lives.**

The workflow reads it (``--matrix`` emits the job list, ``--count`` the divisor) rather than
repeating it, because the failure mode of writing it twice is silent: a matrix listing shards 1 and
2 while the split is computed ``--of 3`` runs two thirds of the suite and reports green. Same shape
as a hand-maintained file list, one level up.

Three, because the split is worth making only until a shard's wall meets the longest single chain
inside it, and past that point more runners buy nothing.
"""

PARITY_PREFIX = "test_rust_parity"
"""Prefix of the files that compare the two implementations against each other.

These are the one family that must **not** run with ``PHYSSYNTH_RS`` set. Each of them builds both
sides itself -- the Python reference and the Rust port -- and asserts something about the pair; with
the flag on, the "Python" half is Rust too, so the file compares Rust against Rust. Most of its
assertions then pass vacuously and at least one fails outright (a negative control, asserting that
a difference *exists*, is the shape that goes red). Neither outcome is information.

So the flagged whole-suite run in CI is the shards **minus this family**, and the family runs once,
unflagged, in its own step. The exclusion lives here rather than as a ``grep -v`` in the workflow
for the same reason the partition does: it is then a function this repo's tests can assert about,
and ``tests/test_shard_partition.py`` asserts both halves of it -- that it removes exactly the
files the glob finds, and that it removes them *after* the split rather than before.

That ordering is the whole hazard and it is worth stating in full. Filtering before partitioning
changes which shard every remaining file lands on, because LPT is a function of the file set; the
flagged shards would then be a different partition from the unflagged ones, and a file could sit in
shard 2 of one and shard 3 of the other. Nothing would notice -- both runs are green, both are
complete -- until the day the two partitions disagree about a file that exists in only one of them.
Split first, drop second, and the flagged run is provably the unflagged run minus a known list.
"""

DEFAULT_COST = 20.0
"""Cost assumed for a file the table has never seen -- i.e. every file added after the last profile.

Deliberately on the high side of the median rather than at it: a *new* test file in this repo is
usually a new model's validation batch, which is nearer the expensive end than the cheap one. Guess
high and a genuinely cheap newcomer costs one shard a few idle seconds; guess low and a 300-second
newcomer lands on whichever shard is already heaviest. Neither guess can drop it.
"""


def _load_costs() -> dict[str, float]:
    """The measured per-file core-seconds, or an empty table if the profile has not been taken."""
    if not COSTS_PATH.exists():
        return {}
    return {str(k): float(v) for k, v in json.loads(COSTS_PATH.read_text()).items()}


def test_files(tests_dir: Path = TESTS_DIR) -> list[str]:
    """Every collectable test file, sorted -- the set the partition is required to cover exactly."""
    return sorted(p.name for p in tests_dir.glob("test_*.py"))


def is_parity(name: str) -> bool:
    """Is this file one of the two-sided comparisons that must run unflagged? See PARITY_PREFIX."""
    return name.startswith(PARITY_PREFIX)


def drop_parity(files: list[str]) -> list[str]:
    """``files`` without the parity family, order preserved."""
    return [f for f in files if not is_parity(f)]


def partition(files: list[str], k: int, costs: dict[str, float] | None = None) -> list[list[str]]:
    """Split ``files`` into ``k`` cost-balanced shards (longest-processing-time greedy).

    LPT: walk the files heaviest-first and drop each on whichever shard is currently lightest. It is
    the standard 4/3-approximation for this, it is two lines, and -- the property that matters here
    -- it is **deterministic**: ties break on the filename, so shard membership is a pure function
    of (the file set, k, the cost table). Two jobs in the same matrix computing it independently
    cannot disagree, which is what lets the split be recomputed per job rather than passed between
    them.

    Guarantees, both asserted by ``tests/test_shard_partition.py``: every file appears in exactly
    one shard, and no shard is empty (which would waste a runner).
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > len(files):
        raise ValueError(f"cannot split {len(files)} files into {k} shards")
    costs = costs if costs is not None else _load_costs()

    order = sorted(files, key=lambda f: (-costs.get(f, DEFAULT_COST), f))
    shards: list[list[str]] = [[] for _ in range(k)]
    loads = [0.0] * k
    for f in order:
        i = min(range(k), key=lambda j: (loads[j], j))
        shards[i].append(f)
        loads[i] += costs.get(f, DEFAULT_COST)
    return [sorted(s) for s in shards]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shard", type=int, help="1-based shard index to print")
    ap.add_argument("--of", type=int, default=SHARDS, help=f"total number of shards ({SHARDS})")
    ap.add_argument("--list", action="store_true", help="print every shard with its cost")
    ap.add_argument("--count", action="store_true", help="print the shard count and exit")
    ap.add_argument("--matrix", action="store_true", help="print the shard indices as JSON")
    ap.add_argument(
        "--exclude-parity",
        action="store_true",
        help="drop the test_rust_parity_* family from the printed shard (for the PHYSSYNTH_RS run)",
    )
    args = ap.parse_args(argv)

    # Both of these exist for the workflow's `setup` job, which turns SHARDS into a matrix so the
    # number is never typed into the YAML. They answer before anything is collected or costed.
    if args.count:
        print(args.of)
        return 0
    if args.matrix:
        print(json.dumps(list(range(1, args.of + 1))))
        return 0

    files = test_files()
    costs = _load_costs()
    shards = partition(files, args.of, costs)

    if args.list:
        for i, shard in enumerate(shards, start=1):
            total = sum(costs.get(f, DEFAULT_COST) for f in shard)
            unknown = sum(1 for f in shard if f not in costs)
            note = f", {unknown} unpriced" if unknown else ""
            print(f"shard {i}/{args.of}: {len(shard):3d} files, {total:8.1f}s{note}")
            for f in shard:
                print(f"    {costs.get(f, DEFAULT_COST):8.1f}  {f}")
        return 0

    if args.shard is None:
        ap.error("--shard is required unless --list is given")
    if not 1 <= args.shard <= args.of:
        ap.error(f"--shard must be in 1..{args.of}, got {args.shard}")

    # Relative, forward-slashed: the output is pasted straight into a shell command line, and an
    # absolute path here would carry this checkout's directory -- which on the dev box contains a
    # space and word-splits into nonsense that pytest reports as "no tests collected", i.e. a green
    # empty run. Relative paths are also what the logs are readable as.
    # Note the order: the shard is selected out of the full partition and only then filtered, so
    # `--exclude-parity` cannot move a file between shards. See PARITY_PREFIX for why that matters.
    shard = shards[args.shard - 1]
    if args.exclude_parity:
        shard = drop_parity(shard)
    print(" ".join(f"tests/{f}" for f in shard))
    return 0


if __name__ == "__main__":
    sys.exit(main())
