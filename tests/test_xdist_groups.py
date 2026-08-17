"""Every consumer of a module-scoped fixture sits in the same xdist group -- or in none at all.

``--dist loadgroup`` builds a module-scoped fixture once **per worker**, so an expensive one is kept
on a single worker by giving its tests a shared ``xdist_group``. The gate used to do that a module
at a time, which is safe but coarse: it turns a whole file into one serial chain, and since the
suite is run in shards a chain longer than its shard's wall is a floor no number of runners can
divide. Two files now group *per fixture* instead, which is finer and faster -- and which
introduces a failure this file exists to catch.

The failure is silent. A test written against the ``tongue`` fixture but carrying the neighbouring
group runs on a different worker, rebuilds 200 s of string simulation there, and **passes**. Nothing
is wrong with the answer; the only symptom is a slower suite, i.e. exactly the thing nobody
attributes correctly two months later. So the invariant is asserted directly: consumers of one
module-scoped fixture must agree on their group, and *no group* is a group for this purpose (mixing
grouped and ungrouped consumers scatters the fixture just as effectively).

Read **statically**, with ``ast``. Importing 70 test modules to introspect their marks would cost
seconds and run their module-level code for no reason; the pattern being checked is a decorator
and a constant, both visible in the source. The cost is that a group assembled dynamically would be
invisible here -- and the answer to that is that this repo has none, and if one ever appears, this
test failing loudly is the right outcome for it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent


def _xdist_group_name(node: ast.AST) -> str | None:
    """``pytest.mark.xdist_group("name")`` -> ``"name"``, anything else -> ``None``."""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return None
    if node.func.attr != "xdist_group" or not node.args:
        return None
    arg = node.args[0]
    return arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str) else None


def _is_module_scoped_fixture(func: ast.FunctionDef) -> bool:
    """True for ``@pytest.fixture(scope="module")`` -- the only scope this invariant is about.

    Session scope would need the same treatment and there are none; function scope rebuilds per test
    anyway, so grouping cannot save it anything.
    """
    for dec in func.decorator_list:
        if not isinstance(dec, ast.Call) or not isinstance(dec.func, ast.Attribute):
            continue
        if dec.func.attr != "fixture":
            continue
        for kw in dec.keywords:
            if kw.arg == "scope" and isinstance(kw.value, ast.Constant):
                if kw.value.value == "module":
                    return True
    return False


def _module_level_groups(tree: ast.Module) -> tuple[dict[str, str], str | None]:
    """``(constants, module_group)``: names bound to a group mark, and the file-wide group if any.

    The constants matter because the readable way to write these decorators is to name the mark once
    (``TONGUE_GROUP = pytest.mark.xdist_group(...)``) and apply the name -- so resolving a decorator
    means resolving one level of assignment.
    """
    constants: dict[str, str] = {}
    module_group: str | None = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id == "pytestmark":
                marks = node.value.elts if isinstance(node.value, ast.List) else [node.value]
                for mark in marks:
                    module_group = _xdist_group_name(mark) or module_group
            elif (name := _xdist_group_name(node.value)) is not None:
                constants[target.id] = name
    return constants, module_group


def _consumers(tree: ast.Module) -> dict[str, set[str | None]]:
    """Map each module-scoped fixture to the set of groups its test functions carry."""
    constants, module_group = _module_level_groups(tree)
    fixtures = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and _is_module_scoped_fixture(node)
    }

    groups: dict[str, set[str | None]] = {f: set() for f in fixtures}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        group = module_group
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name) and dec.id in constants:
                group = constants[dec.id]
            elif (name := _xdist_group_name(dec)) is not None:
                group = name
        for arg in node.args.args:
            if arg.arg in groups:
                groups[arg.arg].add(group)
    return groups


TEST_MODULES = sorted(p.name for p in TESTS_DIR.glob("test_*.py"))


@pytest.mark.parametrize("filename", TEST_MODULES)
def test_a_module_scoped_fixture_is_not_split_across_groups(filename):
    """The invariant, over every test file -- including the ones that use no groups at all.

    Parametrized per file so a failure names the file rather than handing back a list to read.
    """
    tree = ast.parse((TESTS_DIR / filename).read_text(encoding="utf-8"))
    for fixture, groups in _consumers(tree).items():
        assert len(groups) <= 1, (
            f"{filename}::{fixture} is requested from {len(groups)} different xdist groups "
            f"({sorted(g or '<ungrouped>' for g in groups)}), so `--dist loadgroup` will build it "
            f"once per worker instead of once"
        )


def test_the_two_geometric_files_are_grouped_per_fixture_and_not_per_module():
    """The concrete arrangement the split shard depends on, asserted where it can be read.

    The general invariant above is satisfied by grouping a whole module too, which is what these two
    files used to do -- so it cannot notice a revert. This one can: the whirl and phantom fixtures
    are the two longest chains in the suite, and they are only *not* chains because their consumers
    sit in more than one group.
    """
    for filename, expected in (
        ("test_geometric_whirl.py", 3),
        ("test_geometric_phantom.py", 2),
    ):
        tree = ast.parse((TESTS_DIR / filename).read_text(encoding="utf-8"))
        groups = {g for gs in _consumers(tree).values() for g in gs if g is not None}
        assert len(groups) == expected, (
            f"{filename} spreads its module-scoped fixtures over {sorted(groups)}; expected "
            f"{expected} distinct groups. One group means the file is a single serial chain again."
        )
