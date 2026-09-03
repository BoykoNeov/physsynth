"""The workflow file's own guard — shell continuations, and the test files the steps name.

``docs/dev/rust-migration-plan.md`` §19.7 found a CI step that had been failing since the previous
batch for a reason nothing in the repo could see: a ``run:`` block whose line continuation had been
written as the two characters backslash-``n`` instead of a backslash and a newline. The shell then
received that pair as an argument, ``pytest`` reported "file or directory not found", and the job
went red — loudly, and for a reason that reads like a missing test file rather than a typo in YAML.

Two things make that shape recur rather than being a one-off. It is invisible to a YAML parser,
because the sequence is a perfectly valid pair of characters inside a block scalar; and it is
introduced by *tooling* rather than by typing — any editor, script or patch that passes a string
through one round of escaping too few produces exactly it. The same bug was reintroduced while
§20's step was being added, which is what this file is for.

**No YAML parser is used here, deliberately, and not only to avoid a dependency.** After parsing, a
literal backslash-``n`` and a real newline are both just characters in a string and the distinction
this file exists for is gone. The raw text is where the two are still different things.

Nothing here is about the *content* of a step. That belongs to the steps themselves, which are
claims about a batch and are meant to be edited. This file asserts only what no reader would check
by eye and no other tool checks at all.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
BACKSLASH_N = chr(92) + "n"


@pytest.fixture(scope="module")
def workflow_lines() -> list[str]:
    return WORKFLOW.read_text(encoding="utf-8").splitlines()


def test_no_line_contains_a_literal_backslash_n(workflow_lines):
    offenders = [(n, line) for n, line in enumerate(workflow_lines, 1) if BACKSLASH_N in line]
    assert not offenders, (
        "a literal backslash-n in the workflow -- a line continuation that lost a round of "
        f"escaping, so the shell will receive it as an argument: {offenders}"
    )


def test_every_test_file_the_workflow_names_exists(workflow_lines):
    """The general form of the same failure, and the one that catches a rename too.

    §19.7's step was red because it asked for a file whose name was the mangled continuation. Rather
    than special-case that spelling, this asks the broader question: every ``tests/...py`` token
    anywhere in the workflow must be a file in this repo.

    The count is asserted as well as the membership. A scan that silently matched nothing would
    pass this test forever while checking nothing -- §16.8's shape, and the reason the number is
    written down rather than assumed to be large.

    **The threshold dropped from 50 to 20 on 2026-09-03, and that is a real weakening.** It used to
    be met several times over by twenty-one per-batch steps that each named the files that batch's
    port could reach; plan §35.7 collapsed those into one flagged run of the whole suite, computed
    from the glob, and a computed list has no tokens to check. What is left is the parity family,
    which stays spelled out literally in the `rust` job precisely so this canary keeps something to
    count -- folding *that* list into a script too would take the number to zero and leave this
    test green forever while asserting nothing. The floor is set just under the family's size so
    that deleting a parity file (which §35.4 says happens, one per model, when its Python side
    goes) reads as a threshold to lower deliberately rather than as a failure to explain.
    """
    named = [
        token
        for line in workflow_lines
        for token in line.split()
        if token.startswith("tests/") and token.endswith(".py")
    ]
    assert len(named) >= 20, (
        f"only {len(named)} test-file tokens found in the workflow -- the scan has stopped "
        "matching, so this guard is checking nothing"
    )
    missing = sorted({t for t in named if not (REPO_ROOT / t).is_file()})
    assert not missing, f"the workflow names test files that do not exist: {missing}"


# The same escaping failure with the opposite sign, found in §24 and pre-existing on three steps.
#
# §19.7's variant leaves a literal backslash-``n`` behind, and the test above catches it. The
# variant found while adding the beam's step *swallows* the continuation instead: the same tooling
# round-trip that turns ``\`` + newline into two visible characters can also consume both, and what
# is left is two shell lines JOINED into one. Three earlier steps had it, and none of them failed,
# because ``pytest a.py b.py`` runs the same whether the file names were on one line or four.
#
# It is only harmless while what follows a continuation is another *argument*. The moment a
# swallowed continuation sits between two commands -- ``pip install ...`` and ``pytest ...`` --
# the second becomes an argument of the first and the step silently stops doing half its job. That
# is invisible to the backslash-n test (there is no backslash left to find), to a YAML parser (the
# block scalar is valid) and to the eye (the line runs off the screen).
#
# A length limit is what distinguishes the two shapes without asserting anything about content. The
# longest legitimate ``run:`` line in this workflow is well under 120 characters; a joined one is
# 300 to 850.
RUN_LINE_LIMIT = 120


def test_no_run_block_line_is_a_swallowed_continuation(workflow_lines):
    inside, block_indent, offenders = False, 0, []
    for n, line in enumerate(workflow_lines, 1):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if inside and stripped and indent <= block_indent:
            inside = False
        if stripped.endswith("run: |") or stripped.endswith("run: |-"):
            inside, block_indent = True, indent
            continue
        if inside and len(line) > RUN_LINE_LIMIT:
            offenders.append((n, len(line), line.strip()[:80]))
    assert not offenders, (
        "a `run:` line over "
        f"{RUN_LINE_LIMIT} characters -- almost certainly two shell lines joined by a line "
        f"continuation that lost its backslash AND its newline: {offenders}"
    )


def test_every_rust_parity_file_guards_its_extension_import():
    """A parity file must reach the extension through ``importorskip``, never a bare ``import``.

    The default gate does not build ``physsynth_rs``: the sharded validation harness installs only
    the Python package, and the shard-reconciliation step runs a bare ``pytest --collect-only``. A
    module-scope ``import physsynth_rs`` is therefore a **collection error** there, not a skip --
    it fails the shard it lands in and it makes the reconciliation step's ``grep`` find no count at
    all, so two jobs go red for one line.

    None of that is visible on a development machine, where the extension is always installed, and
    ruff's import sorting will happily tidy a bare import into the third-party block where it looks
    exactly like every other line. Fifteen parity files had it right and the sixteenth did not;
    this is that scar as a test rather than as a paragraph, which is the trade plan section 20.7
    argued for and section 27.7 collected on.
    """
    root = Path(__file__).resolve().parent
    files = sorted(root.glob("test_rust_parity*.py"))
    assert len(files) >= 15, f"only found {len(files)} parity files -- the glob is wrong"
    for path in files:
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            assert stripped != "import physsynth_rs", (
                f"{path.name}:{lineno} imports the extension directly. Use "
                'pytest.importorskip("physsynth_rs") -- the default gate does not build it, and a '
                "bare import is a collection error rather than a skip."
            )
        assert 'importorskip(' in text and '"physsynth_rs"' in text, (
            f"{path.name} never reaches the extension through importorskip"
        )
