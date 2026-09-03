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

    That the scan still MATCHES is asserted too. A scan that silently found nothing would pass this
    test forever while checking nothing -- §16.8's shape.

    **How that half is asserted changed on 2026-09-03, twice, and the second change is the one to
    keep.** It was a count: 50 while twenty-one per-batch steps each named the files that batch's
    port could reach, then 20 after plan §35.7 collapsed those into one glob-computed flagged run
    (a computed list has no tokens to check), leaving the parity family as the only literal list.
    But that family **drains** -- one file per deletion (plan §39's units) -- so the count had to be
    hand-lowered on every deletion and was heading for zero, which is findings #40 in the third
    place it has turned up.

    A count on a draining population is replaced by a **named positive control**:
    `tests/test_binding_surface.py` is spelled out literally in the `rust` job and is permanent
    (it holds the binding's own properties, not a comparison, so no deletion removes it). Finding
    that one token proves the scan works, cannot drain, and is a stronger statement than any
    threshold -- the floor below is kept only as a cheap sanity bound.
    """
    named = [
        token
        for line in workflow_lines
        for token in line.split()
        # A token containing a glob metacharacter is not a file name, it is a *query* -- the
        # detector's step passes `tests/test_*.py` to grep on purpose, so that the list of files
        # reaching the detector is derived at job time rather than typed and left to rot. Asking
        # whether that path "exists" is a category error, and the check that matters for it is the
        # floor inside the step itself. Only literal names are the business of this test.
        if token.startswith("tests/") and token.endswith(".py") and "*" not in token
    ]
    # The positive control: a literal, permanent name the scan must find. See the docstring.
    assert "tests/test_binding_surface.py" in named, (
        "the scan did not find `tests/test_binding_surface.py`, which the `rust` job names "
        "literally and permanently -- so the scan has stopped matching and this guard is "
        f"checking nothing. It found: {sorted(set(named))}"
    )
    assert len(named) >= 2, (
        f"only {len(named)} test-file tokens found in the workflow -- see the docstring for why "
        "this bound is a sanity check rather than the real assertion"
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


def test_every_test_file_guards_its_extension_import():
    """A test file must reach the extension through ``importorskip``, never a bare ``import``.

    Fifteen parity files had it right and the sixteenth did not; this is that scar as a test rather
    than as a paragraph, which is the trade plan section 20.7 argued for and section 27.7 collected
    on. None of it is visible on a development machine, where the extension is always installed,
    and ruff's import sorting will happily tidy a bare import into the third-party block where it
    looks exactly like every other line.

    **Two things changed on 2026-09-03 and the rule survives both.** The premise used to be "the
    default gate does not build ``physsynth_rs``"; it does now (plan section 39.6, route 1), and
    the ``checks`` job builds it too for the reason its own comment gives. What still holds is the
    narrower and more durable reason: a module-scope ``import physsynth_rs`` is a **collection
    error** rather than a skip in any environment that lacks the wheel, and a collection error
    fails the shard it lands in *and* makes the reconciliation step's ``grep`` find no count at
    all -- two jobs red for one line. Contributors, forks and a local ``pytest`` before
    ``pip install ./crates/physsynth-py`` are all such environments.

    And the POPULATION changed. It used to be ``test_rust_parity*.py`` with a floor of fifteen --
    which drains, one file per deletion, exactly like the canary in
    ``tests/test_shard_partition.py`` (plan section 41.5, findings #40). It is now every test file
    that mentions the extension at all, a population that grows as the migration proceeds rather
    than shrinking to nothing: ``tests/test_binding_surface.py`` is in it and is permanent.
    """
    root = Path(__file__).resolve().parent
    files = sorted(
        f for f in root.glob("test_*.py") if "physsynth_rs" in f.read_text(encoding="utf-8")
    )
    assert len(files) >= 10, (
        f"only {len(files)} test files mention the extension -- the scan is wrong. This floor is "
        "on a GROWING population, unlike the one it replaced"
    )
    for path in files:
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            # MODULE SCOPE ONLY -- column zero. Widening the population past the parity family
            # found a bare `import physsynth_rs` inside `test_stability.py`'s `if expected_rust:`
            # branch, and turning THAT into an `importorskip` would be wrong in a way the original
            # rule never had to think about: in that branch the flag IS set, so a missing wheel is
            # an error and must not become a skip. The hazard this rule exists for is specifically
            # a collection error, which only a module-scope import can cause -- so the check is on
            # indentation, and a guarded import inside a function is left alone.
            if line[:1] in (" ", chr(9)):
                continue
            stripped = line.strip()
            assert stripped != "import physsynth_rs", (
                f"{path.name}:{lineno} imports the extension at MODULE SCOPE. Use "
                'pytest.importorskip("physsynth_rs") -- in an environment without the wheel a '
                "module-scope import is a collection error rather than a skip, which fails the "
                "shard AND leaves the reconciliation step with no count to subtract."
            )
        assert 'importorskip(' in text and '"physsynth_rs"' in text, (
            f"{path.name} never reaches the extension through importorskip"
        )
