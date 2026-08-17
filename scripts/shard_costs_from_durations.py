"""Turn a ``pytest --durations=0`` dump into the per-file cost table the shard split schedules on.

Reads the durations table on stdin (the whole CI log is fine -- anything that is not a duration line
is ignored) and writes ``scripts/shard_costs.json``: one entry per test file, the summed
setup + call + teardown seconds of every test in it.

Two things about the numbers it produces, both of which are the reason the output is a *hint* and
never an assertion:

* They are **core-seconds, not wall-clock** -- the sum over a parallel run's workers. That is the
  right quantity for balancing shards (a shard's wall is its core-seconds over its workers) and
  exactly the wrong quantity to quote as "the suite takes N seconds".
* They come off **one runner**, and GitHub's runner class varies by ~1.6x. So the table is good for
  *relative* weight -- which file is heavy next to which -- and worthless as an absolute. Re-profile
  when the balance visibly drifts, not on a schedule.

Usage::

    gh run view <id> --log | python scripts/shard_costs_from_durations.py
    python scripts/shard_tests.py --of 3 --list      # eyeball the resulting balance
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parent / "shard_costs.json"

# `12.34s call     tests/test_thing.py::test_name[param]` -- with an optional log prefix in front,
# because a `gh run view --log` line carries job/step/timestamp columns before the payload.
DURATION_RE = re.compile(
    r"(?P<seconds>\d+\.\d+)s\s+(?:call|setup|teardown)\s+(?P<path>tests/[\w.]+\.py)::"
)


def parse(lines) -> dict[str, float]:
    """Sum every duration line per test file. Unparseable lines are simply not durations."""
    totals: dict[str, float] = defaultdict(float)
    for line in lines:
        m = DURATION_RE.search(line)
        if m:
            totals[Path(m.group("path")).name] += float(m.group("seconds"))
    return dict(totals)


def main() -> int:
    totals = parse(sys.stdin)
    if not totals:
        print("no duration lines on stdin -- was the run made with --durations=0?", file=sys.stderr)
        return 1

    ordered = dict(sorted(totals.items(), key=lambda kv: (-kv[1], kv[0])))
    OUT_PATH.write_text(json.dumps({k: round(v, 2) for k, v in ordered.items()}, indent=2) + "\n")
    print(f"{len(ordered)} files, {sum(ordered.values()):.1f} core-seconds -> {OUT_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
