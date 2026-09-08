# Python retirement — the last of it

> **Decision, 2026-09-07 (the human's).** Python goes to **zero** in this repository. Not "the core
> is Rust and the tooling is Python" — zero. That reverses two of the human's own earlier calls,
> both recorded in `rust-migration-plan.md`: §35.4's "the test suite's move to native bars is
> *elective*", and §35.5's "route (b), the viewer backend stays Python — do not re-open this". They
> were the human's to set and they are the human's to unset; this document does not re-argue them
> and neither should anything downstream. `CLAUDE.md`'s non-negotiable #3 is amended in the same
> commit that lands this file.
>
> A second decision, taken and then changed within the hour: a Python **plotting island** was chosen
> and then withdrawn. There is no island. Where a diagnostic used to draw a figure, **Rust emits the
> data** and whatever draws it lives outside this repository.

What is *not* in scope: the browser front-end. `web/static/app.js` is JavaScript, it is the viewer's
whole visual surface, and nothing here proposes moving it. "Get rid of Python" means Python.

---

## 1. The measurement, 2026-09-07

Everything below is counted on the tree at commit `9ad95d1`, not quoted from an earlier note.

| what | files | lines | what it is |
|---|---|---|---|
| `tests/` | 88 | 34,562 | the correctness gate — 2,713 tests |
| `web/serialize.py` + `web/server.py` | 2 | 10,388 | the viewer backend: builds payloads, serves them |
| `scripts/` | 41 | 9,720 | diagnostics, freezers, the shard splitter, a browser harness |
| `physsynth/` | 34 | 3,485 | shims over Rust, three dual modules, one plotting module |
| `conftest.py`, `pyproject.toml` | 2 | ~120 | the packaging and pytest wiring |
| **Python total** | **167** | **~58,300** | |

And the number that is easy to miss:

| also deleted, and it is **Rust** | lines |
|---|---|
| `crates/physsynth-py` — the pyo3 binding | 19,024 |

The binding exists for exactly one purpose: letting Python call Rust. With no Python there is no
binding, and that is not a rounding error — it is a third of all the Rust in the repository, and it
is the most awkward third, because it is where the `&mut self`-cannot-call-back-into-Python rule,
the buffer-ownership rules and the four Rust-to-Python namespace reaches all live.

**The whole operation deletes roughly 77,000 lines across both languages and writes back less.**
That is the argument for doing it, and it is a stronger one than "Python is retired" as a slogan.

### 1.1 What already exists on the far side

| | tests | source lines |
|---|---|---|
| `physsynth-core` (including `mod tests` blocks inside `src/`) | 448 | 32,279 |
| `physsynth-analysis` | 102 | 5,264 |
| **native total** | **550** | |

---

## 2. A third of the suite has nothing to port to

The 2,713 Python tests are not 2,713 physics bars. Sorted by what happens to each:

| category | tests | fate |
|---|---|---|
| Rust-vs-Python parity (`test_rust_parity_{ops2d,operators,banded}.py`) | 275 | **evaporates** — one side of the comparison is being deleted |
| the binding's Python-facing surface (`test_binding_surface.py`) | 60 | **evaporates** — no binding |
| the Python package's shape (`test_stability.py`: dependency allowlist, layering, module scan) | 21 | **evaporates** — claims about a package that will not exist. The *Rust* half of the same contract already lives in `crates/*/tests/deps.rs` and stays |
| the pytest runner's own machinery (`test_shard_partition.py`, `test_xdist_groups.py`) | 106 | **evaporates** — no pytest, no shards |
| the viewer backend (`test_web_backend.py`) | 429 | **ports, as the viewer's specification** (§5) |
| the analysis freeze (`test_analysis_frozen.py`) | 78 | **ports as data** — the 3,708 recorded floats are literals; they become a Rust fixture file and the enforcement moves with them |
| everything else | ~1,744 | **the physics bars.** These are the work |

So the real question is not "rewrite 34,562 lines" — it is "give the physics bars a native
equivalent, against 550 that already exist", plus one viewer port, and then delete.

**Every number in this section is a `pytest` case count, and §9 revises them down.** A case is a
row of a `@parametrize`, not a claim; measured as test *functions* — the unit a Rust `#[test]`
actually corresponds to — the 1,744 is **899** against 475 native ones. And the first
reading of that map was itself wrong in a way §9.3 records: most of the job is *verifying* that an
existing native bar asserts the same thing, not writing one. Read §9 for the numbers that matter.

**The rule that makes this safe is `rust-migration-plan.md` §35.4's, unchanged**: port by
*criterion*, not by file; and for every Python test retired, a native test asserts the same bar at
the same fixture, **named in the retiring commit**. A physics bar deleted without a named
replacement is a silently lost claim, and this repository has twenty-three models' worth of claims
that only the suite holds.

---

## 3. The order is forced, not chosen

The binding can only be deleted when its last caller is gone, and it has exactly three classes of
caller: the **tests**, the **viewer**, and the **scripts**. Everything else follows from that.

```
                      ┌─► A. resolve the three dual modules ─┐
  B. derive the map ──┤                                      ├─► F. delete the binding,
                      └─► C. port the physics bars ──────────┤      the shims, the
                                                             │      packaging, the CI
  D. the viewer becomes a Rust crate ────────────────────────┤
                                                             │
  E. triage the scripts ─────────────────────────────────────┘
```

B comes first and **both A and C hang off it**: C because it needs to know what is missing, A for
the reason in §8.2 — the parity files are the last cross-language check there will ever be, and
they should not go before what they were protecting is known. D and E depend on nothing but F.

### 3.1 The four Rust-to-Python reaches unwind for free

`grep -rn 'import("physsynth' crates/physsynth-py/src/` finds four places where **Rust reaches back
into Python** — `airbox_wrap.rs`, `connection.rs`, `plate.rs` and `string_geometric.rs` read a
Python module's namespace at call time. These look like a blocker and are not: every one of them is
inside `crates/physsynth-py`, the crate being deleted. They cost nothing at step F beyond going away
with their file.

---

## 4. Phase A — the three dual modules, and one of them is a real decision

Three modules still choose between two implementations at import time. They are the last readers of
`PHYSSYNTH_RS`, and each is a Python body that has to go regardless.

| module | the Python side is | what deleting it means |
|---|---|---|
| `physsynth/core/exciter.py` | a NumPy transcription | nothing. Delete it |
| `physsynth/core/operators.py` | a NumPy/SciPy transcription | nothing. Delete it |
| `physsynth/core/banded.py` | **LAPACK**, through SciPy — `dpbtrf`/`dpbtrs` | a **reference oracle** goes |

`banded` is the one to think about. Its "Python implementation" was never a transcription: it is the
library call every acceptance number in the four-string θ-scheme family was originally measured
against, and `banded.py`'s own header says so in those words. Deleting it does not tidy anything up
— it retires the yardstick.

Two facts make that acceptable and both are already measured (2026-08-27, recorded in that header):
the transcribed solver's lossless energy drift is **2.7e-12 against LAPACK's 2.7e-12** on the same
string, with the bar at 1e-10; and the four models' bit-identity with *each other* — the property
the reduction anchors need — is unaffected, because all four call the same solver either way.

**Acceptance for phase A:** the 275 parity tests are deleted in the same commits as the Python
spellings they compare (§1.2's "together" rule), `PHYSSYNTH_RS` is read by nothing, and the
retirement of LAPACK-as-oracle is stated in the commit with those two numbers in it rather than left
to be rediscovered.

---

## 5. Phase D — the viewer, route (a), revived

§35.5 named two routes and took (b). This takes (a): a Rust HTTP server. What that is, precisely:

- **`web/static/` does not change.** `app.js`, `index.html` and `style.css` are the viewer and they
  stay exactly as they are. The contract they speak — `POST /simulate {json} -> {json payload}`,
  errors as `{"error": {...}}` with HTTP 200, never a 500 and never a NaN — is the seam, and it is
  already narrow and already documented in `web/server.py`'s header.
- **`web/server.py` is 125 lines** of routing. It is the trivial half.
- **`web/serialize.py` is 10,263 lines** and it is not trivial: it is where every payload shape,
  every parameter cap, every guard and every refusal lives, accumulated over twenty viewer batches.
  It is a translation of accumulated *judgment*, not of arithmetic.
- **Its specification is the 429 tests in `test_web_backend.py`**, which is why those port rather
  than evaporate. They are the only complete statement of what a payload must contain.

**A fourth crate, `physsynth-viewer`, with its own dependency list.** The precedent is
`physsynth-analysis`, which is a separate crate for exactly this reason. The core's
`[dependencies]` is **empty** and `crates/physsynth-core/tests/deps.rs` enforces it with an empty
`ALLOWED`, plus a `NEVER` list that names `tokio`, `async-std`, `reqwest` and `hyper` by category. A
viewer needs an HTTP server and a JSON serializer; the core must not learn about either.

Recommendation, to be measured at the batch rather than assumed here: a **synchronous** server with
few transitive dependencies (`tiny_http` is the obvious candidate) plus `serde_json` — which is
already a dev-dependency of the core, so it is not a new name in the tree. No async runtime: the
`NEVER` list's spirit is that this project does not acquire an executor in order to serve localhost,
and the Python server it replaces is a threaded blocking one for the same reason.

**Acceptance for phase D:** the browser renders every one of the twenty model/domain scenes with the
front-end untouched; the 429 payload assertions pass as native tests; and the deep-link parameter
set round-trips. That last one is named because viewer batch 19 shipped a link that silently dropped
every parameter but two for a whole batch and nothing failed — it is the specific regression this
phase must not repeat.

---

## 6. Phase E — the scripts, triaged rather than ported

41 files, and they are not one thing:

| group | count | recommendation |
|---|---|---|
| `diagnose_*.py` | ~33 | **triage.** These are investigation records: a script written to answer one question, mostly already answered and written up in `docs/dev/`. Default is **delete as spent**. A survivor becomes a small Rust binary that **emits data** — JSON or CSV — and draws nothing |
| `freeze_analysis.py`, `freeze_horizon.py` | 2 | **delete.** Their output is already frozen in `tests/analysis_frozen_values.py`, which ports as a Rust fixture (§2). A freezer whose source is gone has nothing left to freeze |
| `shard_tests.py`, `shard_costs_from_durations.py`, `shard_costs.json`, `nicepytest.py` | 4 | **delete.** They exist to split and run a pytest suite |
| `verify_web_headless.py` | 1 | **port**, and it is the only script that is really a gate (§6.1) |
| `sweep_geometric_lam_long.py` | 1 | triage with the diagnostics — it backs an open hurdle (`scientific-hurdles.md` §6), so it is a survivor candidate |

**No plotting.** The withdrawn island decision means the ~20 matplotlib scripts and
`physsynth/viz/plots.py` (609 lines) are deleted rather than translated. Where a figure was the
point, the Rust survivor writes the numbers and the figure is drawn outside this repository. The
browser viewer already covers twenty models interactively and is the project's actual
visualization.

**But triage is two questions, not one, and the second is the hard one.** Survival is the first; the
second is *what data shape replaces the figure*, and a figure and a data file are not the same
artifact. A convergence study is a table and ports directly. A Chladni pattern, an airbox field
slice, a whirl orbit are not tables — "emit the array as CSV" is not a substitute for the thing
those scripts existed to show. **A survivor whose output was inherently visual must have its claim
restated as a number before it can be ported at all**, and if the claim cannot be stated as a
number, that is an argument the script was always a *look at this* rather than a measurement —
which makes it a delete, not a port. `diagnose_free_plate.py`, `diagnose_orthotropic_free_plate.py`
and the airbox field-slice scripts are where this bites; decide it in the triage rather than
mid-batch.

### 6.1 The browser harness is the one script that is load-bearing

`scripts/verify_web_headless.py` drives real Chrome over the DevTools protocol and checks what the
page *rendered* — the readouts, the canvas pixels, and (since the horizon batch) `_horizon_mark_ok`,
which cross-checks the strip's number against what each panel decided it drew. No payload test can
replace it: `test_web_backend.py` proves the data is right, and this proves the page drew it.

Porting it needs a WebSocket client in the viewer crate's **dev**-dependencies, since the DevTools
protocol is JSON over a socket. That is a dependency decision to take at the batch, under the same
two-step rule the core uses: add the name to `Cargo.toml` and to the allowlist in the same commit,
with the reason.

---

## 7. Phase F — the end state

What is left when this is done:

```
crates/physsynth-core       the physics.       [dependencies] empty, still enforced
crates/physsynth-analysis   the instruments.   [dependencies] empty, still enforced
crates/physsynth-viewer     the HTTP server and payload builder, with its own small allowlist
web/static/                 the JavaScript front-end, untouched
docs/                       unchanged in kind
```

Deleted: `physsynth/`, `tests/`, `web/*.py`, `scripts/`, `conftest.py`, `pyproject.toml`, and
`crates/physsynth-py/`.

**CI collapses.** Today there are four jobs: `validate` (the Python suite, sharded across machines),
`rust-harness` (the same suite again with `PHYSSYNTH_RS=1`), `rust` (native bars plus parity) and
`checks` (ruff plus shard reconciliation). Three of the four are about Python. The end state is
`cargo fmt --check`, `cargo clippy`, `cargo test --workspace`.

That is also where the schedule payoff lands. The Python suite is **5,027 core-seconds** and needs a
computed three-way shard split — with its own guard, because a test that falls out of the split
fails green. Cargo parallelizes tests in-process by default, so the split, `shard_costs.json`, the
LPT partitioner and the two tests that check the partitioner all disappear as a class of problem.

---

## 8. What could make this not worth it, stated up front

1. **The suite is the acceptance contract, and it is the thing being dismantled.** Every previous
   phase of this migration could be checked by running the Python suite against the Rust. This
   phase deletes the checker. §35.4's naming rule is the only defence, and it is a discipline
   rather than a mechanism — nothing *fails* if a bar is dropped silently.
2. **Phase A removes the cross-language cross-check before phase C needs it.** Once the parity files
   are gone, "the two spellings agree" is no longer assertable about anything. That is correct —
   there is one spelling — but it is why phase A should land *after* the coverage map, so that what
   the parity files were protecting is known before they go.
3. **`serialize.py` is judgment, not arithmetic.** 10,263 lines of caps, guards and refusals, with
   429 tests describing them. It is the single largest piece of translation in the whole migration
   and the one where "it compiles and the tests pass" proves the least.
4. **Rust's test ergonomics are worse for this shape of suite.** No `parametrize`, no fixtures, no
   `-k`. 1,744 bars written in a parametrized style will not transcribe one-to-one; the native files
   already in `crates/*/tests/` show the shape that works — loops over arrays of cases — and it is
   more verbose.
5. **The plotting capability genuinely leaves.** Not deferred — leaves. Anyone wanting a static
   figure gets a data file and draws it elsewhere. That was decided with the alternative on the
   table and then re-decided the same way.

---

## 9. Phase B, done — the coverage map

Derived 2026-09-07. Reproducible from `W:\temp\claude\coverage-map\` (`build_map.py` plus the two
input dumps); the model mapping is hand-written in that script and every one of the 88 Python files
and 41 Rust files is mapped, so nothing falls through unclassified.

### 9.1 The counting unit is a test FUNCTION, and that halves the job

`pytest` reports **2,713** tests. That is cases, not claims: 1,389 of them are further rows of a
`@parametrize`, and the same suite is **1,324 test functions**. One function accounts for 85 of
those cases and another for 80. Rust has no `parametrize` — a native test loops over an array of
cases inside one `#[test]` — so comparing 2,713 against 550 compares two different units and
inflates the Python side by about 2×.

Everything below counts **functions**. Both numbers are worth keeping: the case count is what a CI
log prints, the function count is what has to be written.

| unit | Python | Rust |
|---|---|---|
| cases as reported | 2,713 | 550 |
| test functions | 1,324 | 550 |
| of those, **physics** | **899** | **475** |

### 9.2 The map

Cells are `python/rust`; `.` means neither side says anything under that heading. The model axis is
hand-mapped and reliable. The claim axis is derived from test names by keyword and is **indicative
only** — a third of both sides lands in `other`, which is a fact about naming, not about coverage.

| model | energy | dispersion | convergence | modal | stability | signature | reduction | surface | other | py | rs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `airbox` | 27/6 | 2/1 | 4/2 | 11/5 | 43/3 | 2/0 | 13/2 | 4/1 | 82/17 | **188** | **37** |
| `analysis_freeze` | . | . | . | . | . | . | . | . | 5/0 | **5** | **0** |
| `analysis_horizon` | 1/0 | . | 2/0 | 14/8 | 5/8 | . | 2/1 | 1/1 | 23/9 | **48** | **27** |
| `analysis_modal` | . | . | . | 0/5 | 0/4 | . | . | . | 0/4 | **0** | **13** |
| `analysis_oracles` | 0/1 | 0/2 | 0/1 | 0/2 | . | . | 0/1 | . | 0/9 | **0** | **16** |
| `analysis_rotating_wave` | . | 0/1 | . | 0/2 | 0/2 | . | 0/1 | 0/1 | 0/6 | **0** | **13** |
| `analysis_special` | . | . | . | 0/2 | . | . | 0/1 | 0/1 | 0/15 | **0** | **19** |
| `analysis_spectrum` | . | . | 2/2 | 1/3 | . | . | 1/1 | . | 2/5 | **6** | **11** |
| `beam` | 5/3 | . | 1/1 | 3/5 | 3/2 | . | 0/1 | . | 4/5 | **16** | **17** |
| `body` | 4/4 | . | . | 3/2 | 1/2 | . | . | 0/1 | 1/5 | **9** | **14** |
| `bore` | 8/4 | . | 1/0 | 4/2 | 9/3 | . | 1/1 | . | 8/14 | **31** | **24** |
| `bow` | 4/3 | . | 1/0 | 2/0 | 5/2 | 5/1 | 1/1 | 1/1 | 3/3 | **22** | **11** |
| `collision` | 4/0 | . | 0/1 | . | . | 3/2 | 1/0 | . | 7/6 | **15** | **9** |
| `collision_barrier` | 1/3 | . | . | . | 0/2 | 2/1 | 0/1 | . | 6/5 | **9** | **12** |
| **`connection`** | 6/0 | . | . | . | 4/0 | . | 1/0 | . | 1/0 | **12** | **0** |
| `exciter` | . | . | . | . | 0/3 | 0/2 | . | 0/1 | 0/2 | **0** | **8** |
| **`free_plate`** | 13/0 | . | 1/0 | 4/0 | 7/0 | 1/0 | 2/0 | . | 9/0 | **37** | **0** |
| **`free_plate_orthotropic`** | 4/0 | . | 2/0 | 2/0 | 4/0 | . | 2/0 | . | 15/0 | **29** | **0** |
| **`guitar_plate`** | 3/0 | . | 2/0 | 2/0 | 2/0 | . | 1/0 | . | 9/0 | **19** | **0** |
| `mallet` | 15/8 | . | . | 7/0 | 2/3 | 10/5 | 6/2 | 3/2 | 26/13 | **69** | **33** |
| `membrane` | 5/5 | . | 2/0 | 3/1 | 6/2 | . | . | . | 3/4 | **19** | **12** |
| `operators` | 0/5 | . | . | 1/9 | . | . | 0/1 | . | 3/45 | **4** | **60** |
| `plate` | 11/5 | . | 1/4 | 5/2 | 13/5 | 1/0 | 1/2 | . | 7/24 | **39** | **42** |
| **`plate_orthotropic`** | 3/0 | . | 1/0 | 5/0 | 3/0 | . | . | 1/0 | 6/0 | **19** | **0** |
| `radiation` | 11/4 | . | 2/0 | 9/3 | 6/5 | . | 4/4 | . | 20/8 | **52** | **24** |
| `reed` | 7/5 | . | . | 3/0 | 6/6 | . | 2/0 | . | 7/8 | **25** | **19** |
| **`string_damped`** | 4/0 | . | 1/0 | 3/0 | 4/0 | . | 1/0 | . | 3/0 | **16** | **0** |
| `string_geometric` | 9/3 | 1/0 | 3/0 | 9/0 | 9/1 | 4/1 | 6/1 | 2/0 | 26/8 | **69** | **14** |
| `string_ideal` | 5/3 | 4/1 | 1/0 | 5/2 | 0/4 | . | . | . | 1/3 | **16** | **13** |
| `string_nonlinear` | 6/3 | . | 1/0 | 10/3 | 4/1 | . | . | 1/0 | 8/6 | **30** | **13** |
| `string_stiff` | 4/2 | 2/0 | 3/0 | 6/3 | 3/3 | . | . | . | 2/6 | **20** | **14** |
| **`sympathetic`** | 4/0 | . | . | 2/0 | 5/0 | 1/0 | 2/0 | . | 1/0 | **15** | **0** |
| **`vk_plate`** | 15/0 | . | 8/0 | 8/0 | 8/0 | 1/0 | 2/0 | 1/0 | 17/0 | **60** | **0** |
| **total** | 179/67 | 9/5 | 39/11 | 122/59 | 152/61 | 30/12 | 49/21 | 14/9 | 305/230 | **899** | **475** |

Not physics, and already accounted for in §2 (function counts, so lower than that table's cases):
`!viewer` 338, `!binding` 31, `!parity` 32, `!package` 10, `!runner` 10 and `!ci` 4 on the Python
side; `!numerics` 69 and `!deps` 6 native-only on the Rust side.

### 9.3 The first reading said nine models were at zero. It was wrong

The file-level join above put nine models at zero native tests. **Eight of the nine were an
artifact of the join, not a hole in the suite**, and the reason is structural:

> **The two suites are indexed on different axes.** Python test files are named after a *model and
> a claim* — `test_free_plate_modal.py`, `test_vk_energy.py`. Rust test files are named after the
> *source module* they exercise. In Rust, the supported plate, the free plate, the orthotropic
> plate, the guitar outline and the von Kármán plate are all `plate.rs`, because they are all one
> module. So a join on filename reports five Python-facing models as absent while forty native
> tests sit in the file that covers them.

The tell was visible in the table and was read past: `plate` shows **39/42**, which is suspiciously
balanced for a file that would have to carry five plate variants on the Python side alone. Read by
hand, `crates/physsynth-core/tests/plate.rs` contains:

| Python-facing model | native bars in the shared file | examples |
|---|---|---|
| von Kármán plate | **~24** (`plate.rs` + `plate_theta_solve.rs`) | the four Jacobian bars, `newton_converges_where_picard_does_not`, `auto_is_bit_identical_to_picard_wherever_the_sweeps_converge`, `a_nonlinear_plate_conserves_its_total_energy`, the wall-versus-cap pair |
| orthotropic plate | ~5 | `spruce_is_not_an_isotropic_plate_with_one_axis_stretched`, `isotropic_material_comes_back_at_exactly_one`, `the_split_contradiction_message_prints_the_effective_cross_term` |
| guitar plate | ~4 | `a_guitar_reports_a_staircase_deficit_and_prunes_only_at_the_rim`, `components_are_counted_four_connected`, `a_curved_supported_plate_is_a_refusal_not_a_limitation` |
| free plate | ~3 | `the_free_plate_stiffness_annihilates_its_rigid_body_nullspace`, `the_free_edge_holds_the_same_way_off_centre` |

and the same mistake was made once more, one file over: the **damped string**'s bars are inside
`string_stiff.rs`, which carries a `damped_params` helper and four tests using it, including
`sigma1_zero_is_the_stiff_string_exactly` — the chain anchor whose whole point is that model #3
reduces to model #2.

### 9.4 What is actually at zero, after the correction

| model | py functions | native | what is really true |
|---|---|---|---|
| **sympathetic strings** | 15 | **0** | genuinely absent — no native test anywhere mentions the model |
| orthotropic **free** plate | 29 | **1** | the grained plate and the free plate each have bars; the combination has one |
| the bridge **connection** | 12 | **partial** | `body.rs` has the bridge read-outs and the rank-one correction guard; the coupled-energy bars are still only Python's |
| the analysis **freeze** | 5 | n/a | ports as *data* regardless (§2) |
| von Kármán plate | 60 | ~24 | **not a zero** |
| free plate | 37 | ~3 | **not a zero**, but thin |
| orthotropic plate | 19 | ~5 | **not a zero** |
| guitar plate | 19 | ~4 | **not a zero** |
| damped string | 16 | 4 | **not a zero** |

**So the job is mostly verification, not authorship.** The earlier "212 functions with nothing on
the far side" was wrong by an order of magnitude: what is genuinely unwritten is the sympathetic
strings (15 functions) and most of the orthotropic free plate (29), with the connection's coupled
energy a partial. Everything else is a claim-by-claim reading of whether a native bar that *exists*
asserts the same thing at the same fixture — slower per test than writing from scratch, but a
different and much smaller kind of work.

The one number the correction does not move is the room: **`airbox` is 188 to 37**, and its native
bars are the nineteen `mod tests` blocks inside `src/` plus the eighteen added by migration §46.
That is a real shortfall by volume and it is now the largest single one.

### 9.5 The methodological lesson, because it will recur

**A join between two test suites indexed on different axes measures the indexing, not the
coverage.** This map's model axis was hand-written and complete on both sides — every one of the
88 Python and 41 Rust files is mapped, nothing fell through — and it was still wrong, because
"complete" and "correct" are different properties of a mapping. A file whose name says `plate` is
mapped honestly to `plate`; the error was assuming the Python-facing model taxonomy and the Rust
source-module taxonomy are the same taxonomy. They are not, and the mismatch is exactly where a
Python model was a *configuration* of a Rust type rather than a type of its own.

The cure is cheap and it is what phase C's per-batch reading already has to do: before believing a
zero, read the test *names* in the file that would plausibly hold the bar. Two greps overturned
eight of nine.

---

## 10. Where the work stands

Phase B is done (§9) including its own correction. Phase C's **first batch is done** (§12), and it
found the hole §11 records.

The map's revised recommendation for the next batch was **sympathetic strings**, and §12 did it.
**Batch 2 is done too** (§13), and it changes what "next" means: the ordering is no longer a choice
between models with thin coverage, it is the dependency order §13.1 derives inside §11's hole —
ports, then wrappers, then bridges. **Batch 3 is done** (§14) and finishes the port tier: the hole
is down to **eleven classes across two files**, the six grid wrappers plus three surface adapters,
and the three bridges. The next batch is the wrappers (§14.8).

§11's table is **corrected by §13.1**: the hole is 16 classes across *three* binding files, not 14
across two, and the file it missed is the one underneath the other two.

**D**, the viewer, remains independent of all of this and is still the longest pole.

---

## 11. The hole: model classes with no home outside the binding crate

Found on the first phase-C batch, 2026-09-08, and it is a correction to §1 and §7 rather than a
detail of that batch. **The count and the file list below are superseded by §13.1**, which derived
them mechanically instead of by hand and found sixteen classes across *three* files — this section
missed `airbox_port.rs` entirely, and it is the tier the other two stand on. Everything below about
*why* the classes are there, and what porting them costs, still holds.

§1's table counts `crates/physsynth-py` — 19,024 lines — as something that *goes away*, on the
reasoning that a binding with nothing to bind is dead code. That is true of the binding **as
glue**. It is not true of everything inside it. Two of its files are the only implementation of
models this project ships, because the migration deliberately put them there:

| file | lines | classes with no half in a surviving crate |
|---|---|---|
| `connection.rs` | 1,415 | `StringBodyBridge`, `StringPlateBridge`, `StringVKPlateBridge`, `SympatheticStrings` |
| `airbox_wrap.rs` | 1,816 | `_PlateSurface`, `_MembraneSurface`, `_VKPlateSurface`, `RoomLoadedPlate`, `RoomSuspendedPlate`, `RoomLoadedVKPlate`, `RoomSuspendedVKPlate`, `RoomLoadedMembrane`, `RoomSuspendedMembrane`, `RoomLoadedBody` |

Both were written that way on purpose and the reasons are in their headers: these classes are
**polymorphic over their collaborators through Python duck typing** (`StringBodyBridge`'s `body=`
slot takes eight different types; the airbox wrappers compute *through* SciPy objects the tier
below stores as Python attributes). A downcast to a concrete `#[pyclass]` would have narrowed them,
so `rust-migration-plan.md` §32.2 and §34 accepted "no core half" as the price. That price was
correct while Python existed. With Python gone the classes go with the interpreter unless they are
re-homed first, and §7's end state lists no crate that would hold them.

**So phase F is gated on a phase that did not exist:** every class above has to be ported into
`physsynth-core` (or the viewer crate, for anything only the viewer builds) before the binding can
be deleted. The port is not a transcription — the duck typing has to become something Rust can
express. §12 is the first instance and it is the cheap case: `SympatheticStrings` is *always* built
from `IdealString`s and a `ModalBody` at every call site in the repository, so the general slot
collapses to a concrete type and nothing is lost. `StringBodyBridge` is the expensive case, with
eight body types behind one slot, and it will need an enum or a trait; the airbox wrappers are
worse, because the tier below them stores its matrices as Python objects *by design*.

Two things follow for sequencing. The re-homing is **phase C work by another name** — it lands with
the bars, since a class with no test is not a port — and the estimate in §2 (899 physics functions
against 475) does not include it, because it counted tests rather than implementations. And the
count of what phase F deletes drops: 3,231 of the binding's 19,024 lines are moving, not going.

---

## 12. Phase C batch 1, done — sympathetic strings

`SympatheticStrings` now lives in `crates/physsynth-core/src/connection.rs`, with
`crates/physsynth-core/tests/connection.rs` (18 bars) replacing `tests/test_sympathetic.py`
(15 functions, 59 s of suite time), deleted in the same commit.

**What the port needed that the plan did not predict.** A whole new numerical routine. The model's
constructor refuses an over-stiff bridge by the exact dense guard `k^2 lambda_max(A) < 4`, and the
Python got `lambda_max` from `np.linalg.eigvals` — LAPACK on a general matrix. The core crate's
dependency list is empty and stays empty, so there was nothing to call:
`crates/physsynth-core/src/eig.rs` is Householder tridiagonalization plus implicit-shift QL,
written for this guard, with 10 bars of its own in `crates/physsynth-core/tests/eig.rs`.

It is the **symmetric** routine, and that is a measured claim about the operator rather than a
convenience. `A = M^-1 K` is not symmetric, but `M^1/2 A M^-1/2` is — measured at 3.6e-18 relative
on a two-string fixture, i.e. roundoff in the scaling multiply itself — with `M` the trapezoidal
mass diagonal, halved at the free end that carries the spring. That is asserted as a bar in its own
right (`the_coupled_operator_is_self_adjoint_in_the_energy_inner_product`), because if it stopped
holding, the guard would be answering a different question rather than answering this one
imprecisely. Power iteration was rejected on a measurement too: the top of this spectrum is
clustered (`lambda_2 / lambda_1 = 0.9969`) and two identical strings make it exactly degenerate.

**The one bar that changed shape, and why that is the rule rather than the exception.**
`test_single_string_bit_identical_to_string_body_bridge` compared a one-string set against
`StringBodyBridge` — a second implementation of the same spring, and one this batch does not port.
In a one-implementation world that comparison is the code agreeing with itself, which is
`rust-migration-findings.md`'s ledger #64/#65 exactly. What it was protecting is the *shape* of the
coupling, and that is assertable against the free parts: one step of the coupled set is one step of
each part alone plus exactly one spring force, at one node, weighted by `2 k^2 / (rho h)`, handed
to the body unscaled — bit for bit, since every difference is elementwise arithmetic in the same
order. `test_K0_bit_identical_to_uncoupled_parts` did **not** change, because both of its referents
(a string alone, a body alone) exist natively.

**The retirement rule, discharged.** Every retired Python test is named in the doc comment of the
native bar that carries it, and the three new ones say they are new.

| retired | native replacement |
|---|---|
| `test_total_energy_conserved_across_lambda` | `the_total_energy_is_conserved_across_lambda` |
| `test_total_energy_conserved_across_count` | `the_total_energy_is_conserved_across_string_count` |
| `test_passivity_with_body_damping` | `a_lossy_body_makes_the_total_decrease_monotonically` |
| `test_passivity_with_string_damping` | `lossy_strings_make_the_total_decrease_monotonically` |
| `test_single_string_bit_identical_to_string_body_bridge` | `one_step_is_the_free_parts_plus_exactly_one_spring_force` (**restated**) |
| `test_K0_bit_identical_to_uncoupled_parts` | `at_zero_stiffness_the_set_is_bit_identical_to_the_uncoupled_parts` |
| `test_antisymmetric_mode_keeps_bridge_still` | `the_antisymmetric_mode_keeps_the_bridge_exactly_still` |
| `test_symmetric_mode_drives_bridge` | `the_symmetric_mode_drives_the_bridge` |
| `test_sympathetic_transfer_tuned_beats_detuned` | `a_tuned_neighbour_rings_up_far_more_than_a_detuned_one` |
| `test_unstable_stiffness_rejected` | `an_over_stiff_bridge_is_rejected` |
| `test_guard_holds_at_its_boundary` | `the_guard_holds_just_inside_its_boundary` |
| `test_mismatched_timestep_rejected` | `a_mismatched_timestep_is_rejected` |
| `test_right_end_must_be_free` | `a_clamped_right_end_is_rejected` |
| `test_ks_length_must_match` | `one_stiffness_per_string_is_required` |
| `test_empty_strings_rejected` | `an_empty_set_is_rejected` |
| — | `the_coupled_operator_is_self_adjoint_in_the_energy_inner_product` (new) |
| — | `a_negative_stiffness_is_rejected` (new) |
| — | `a_string_at_the_courant_limit_is_rejected` (new) |

**Three absolute thresholds became relative.** The antisymmetric bar's `1e-13` and `1e-15` were
absolute on a fixture plucked to `1e-3`, which makes them claims about the amplitude someone picked
rather than about the physics. They are scaled by the pluck amplitude and by `E^0` now.

**The port was verified against the implementation it replaces, once, before the Python went.**
Same fixture through both — two strings, `N = 100`, `lambda = 0.9`, `K = 8000`, one plucked, the
body given an initial state, 500 steps: every state array and both energies are **bit-identical**,
and the guard's `lambda_max` differs by **16 ulps (2.3e-15 relative)**, which is this crate's
tridiagonal QL against LAPACK's general `dgeev` and is the only number in the model that could
differ. That check is deliberately *not* a committed test — it is a cross-language parity
comparison of the kind phase A retires — but it is why the arithmetic in `core/connection.rs` keeps
the binding's expression order and its `reduce::sum`, and it stays repeatable until the binding
goes.

**One number from it *is* frozen, and it is what stops the symmetry bar being self-consistent.**
The symmetrized operator is built from `mass_diagonal()`, which is the new module's own derivation
of the mass; a wrong derivation that still symmetrizes — a uniform scale, or the half cell on the
wrong end of two *identical* strings — would satisfy the symmetry check while scaling the entire
spectrum. So `the_coupled_operator_is_self_adjoint_in_the_energy_inner_product` also asserts
`lambda_max = 1661856272.3158104` to 1e-12 relative, measured against LAPACK on that fixture
through the binding. A value and a tolerance rather than a bit pattern, because 16 ulps of
cross-implementation spread is the floor on what is portable (findings ledger #68). The mass
derivation is load-bearing now rather than checked against itself.

**Cost.** The Python file was 59.11 s of suite time; the 18 native bars run in **2.1 s** in the
debug profile `cargo test` uses.

**What is still only in the binding for this model.** `PySympatheticStrings` was left exactly as it
is: the viewer builds it through Python, and `tests/test_web_backend.py`'s 22 sympathetic bars —
including the antisymmetric zero, both transfer arms and the guard refusal — go on covering that
path until phase D moves the viewer. Rewiring the binding class to delegate to `core::connection`
would touch a shipping path to no end; it goes when the crate goes.

---

## 13. Phase C batch 2, done — the lumped port and the body it loads

`RoomPort` and `RoomLoadedBody` now live in `crates/physsynth-core/src/airbox_port.rs` and a new
`crates/physsynth-core/src/airbox_wrap.rs`, with `crates/physsynth-core/tests/airbox_port.rs`
(26 bars) replacing `tests/test_airbox_port.py` (25 functions, 427 lines), deleted in the same
commit.

### 13.1 §11's table was incomplete: the hole is 16 classes across THREE files

§11 was hand-written and it missed a file. The audit is three steps and is worth keeping because it
is reproducible where a hand list is not:

1. every `#[pyclass(name = "...")]` in `crates/physsynth-py/src/` (43 names);
2. every `pub struct` / `pub enum` in `crates/physsynth-core/src/` (88 names);
3. the set difference, then **clear the case-spelling false positives by hand** — the two here are
   `VKPlate` → core's `VkPlate` and `MalletVKPlate` → core's `MalletVkPlate`, both real ports whose
   Python-facing name capitalises an acronym the Rust name does not.

That leaves 18 names, 16 of them real (struck through = re-homed):

| file | classes with no half in a surviving crate |
|---|---|
| `airbox_wrap.rs` | `_PlateSurface`, `_MembraneSurface`, `_VKPlateSurface`, `RoomLoadedPlate`, `RoomSuspendedPlate`, `RoomLoadedVKPlate`, `RoomSuspendedVKPlate`, `RoomLoadedMembrane`, `RoomSuspendedMembrane`, ~~`RoomLoadedBody`~~ |
| `connection.rs` | `StringBodyBridge`, `StringPlateBridge`, `StringVKPlateBridge`, ~~`SympatheticStrings`~~ |
| **`airbox_port.rs`** | `SurfacePort`, `InteriorSurfacePort`, ~~`RoomPort`~~ |

**`airbox_port.rs` is the file §11 does not mention at all**, and it matters more than the count
does, because it is *underneath* the other two. `Patch` (`crates/physsynth-py/src/airbox_port.rs:651`)
holds `room`, `t`, `r` and `load_matrix` as `Py<PyAny>`, and the core's `airbox_port` module exports
free functions only — `build_t`, `load_matrix`, `port_weights`, `r_room`, `free_pressure_nodes`,
`ball_nodes`, `spread` — with no stateful port anywhere. Every wrapper in the tier above calls
`port.require_ready()`, `port.free_pressure()`, `port.inject()` and `port.reset()`. So the ports
must be re-homed **before** the wrappers, and the wrappers before the three bridges (whose `body=`
and `plate=` slots take the room wrappers as arguments). That ordering is read off the source, not
chosen.

The native `AirBox` is a fourth understatement of the same kind. It is a *shell*: its `step` called
`inject_scalar` and never `inject_port` / `booked_port`, which exist in the crate and were reachable
only from the binding. `crates/physsynth-py/src/airbox.rs` does not hold a `core::AirBox` at all —
it holds `core::Params` and re-assembles the step out of the same free kernels — so bringing the
native room up to a full room **cannot** disturb the shipping path, which is what made it safe to do
inside this batch rather than as its own.

### 13.2 The decision that propagates to all sixteen: who owns the room

`test_two_heads_share_one_room` (two instruments, one `AirBox`) rules out a wrapper owning its room.
Of the three shapes left, this batch chose and the rest of the tier inherits:

> **The port is a value the caller owns, and the room is passed at each call.**
> `port.free_pressure(&room)`, `port.inject(&mut room, q)`, `inst.step(&mut room, force)`.

Rejected: `Rc<RefCell<AirBox>>`, a literal transcription of Python's shared mutable object, which
would put interior mutability into the core crate's public API; and "the room owns its ports and a
wrapper holds an index", which works but makes every port method a method on the room and hands out
an unchecked handle.

Two invariants the reference kept **on the room** had to survive the change, and each cost one field
on `AirBox`:

* **Disjointness** — two ports may not share a node. `room.claims: Vec<PortClaim>` is `room._ports`
  reduced to what the refusal actually reads (the flat footprint and the label its message quotes),
  and `RoomPort::new` takes `&mut AirBox` and claims before it returns.
* **Unsticking** — `AirBox.set_state` clears every registered port's pending mark by *writing into
  it* (`port._queued_at = -1`). A value-typed port cannot be written into, so the room carries
  `epoch: u64`, bumped by `set_state`, and the port records the epoch alongside the step it queued
  at. A mark from epoch `e` says nothing about a room in epoch `e + 1`, whatever its step count.

  That distinction is invisible to the retired test, and deliberately has its own new bar. In
  `a_room_set_state_unsticks_every_port` the port queued at step 0 and `set_state` returns the room
  to step 0 — so a bare step-count comparison refuses, and the bar passes only because the epoch
  moved. `the_epoch_and_not_the_step_count_is_what_unsticks_a_port` asserts exactly that, and both
  bars go red when the epoch is dropped from the comparison.

### 13.3 The one-time check, and the one thing it found

Same scene through both implementations, before the Python went, and deliberately not committed:
two instruments in one room — a point port and a 33-node ball — 200 steps, comparing 34 quantities
including all 693 pressure nodes and 1,840 velocity faces. Run on two fixtures: two lossless modes
in a rigid room, and five damped modes in a room with two lossy walls.

It found one real divergence, and it is a **fused multiply-add**.

`u_free = np.dot(a, q^{n+1} - q^{n-1}) / 2k` is the modal read of the body's velocity, and it is the
one `np.dot` in this crate that **reaches the timestep** — it becomes the volume velocity the room is
injected with and the pressure the body is corrected by. The crate's three existing `dot` copies
(`beam`, `plate`, `string_stiff`) spell themselves `s += x * y` and each says in its doc comment that
it is a read-out. Spelled that way here, the native scene diverges from the reference at the **second
step**, by one ulp in the ball port's volume velocity, and 200 steps later 661 of 693 pressure nodes
differ at 1.9e-12 — the room amplifying a single ulp, with the body's own state still bit-identical
throughout. Spelled `x.mul_add(y, s)`, all 34 quantities agree to the bit on both fixtures.

BLAS `ddot` fuses its multiply-add; `f64::mul_add` is single-rounded by definition, so what the
spelling buys is a *portable* native answer, and that it is also `ddot`'s answer is what the check
measured, at two mode counts, while there was still something to measure it against.

**And no native bar can see the difference.** Both spellings are physically correct — they differ in
the last bit — so with the plain loop restored, all 26 bars still pass. Measured, not assumed. That
generalises past this batch: *a last-bit choice invisible to every native bar is not a defect, but it
is unrecoverable once the reference goes*, so the cross-implementation check has to happen in the
batch that ports the class and cannot be deferred. The same is true of one more thing this batch
changed: `AirBox::step` now books its injections as a **per-step subtotal** added once to the running
total, which is the reference's association and differs from the crate's previous
`self.injected += term` only when two injections land in one step — i.e. only in a scene the crate
could not express until this batch. Reverting it breaks nothing native either.

### 13.4 The retirement rule, discharged

| retired | native replacement |
|---|---|
| `test_conserved_rigid_room` | `the_scene_total_is_flat_in_a_rigid_room` |
| `test_conserved_lossy_walls` | `the_scene_total_is_flat_with_lossy_walls` |
| `test_conserved_spread_port` | `a_spread_port_conserves_across_wall_sets` |
| `test_conserved_lossy_body` | `a_lossy_body_makes_the_scene_total_decrease_monotonically` |
| `test_free_pressure_matches_full_array` | `the_local_free_pressure_read_is_the_full_array_update_exactly` |
| `test_free_pressure_matches_full_array_spread` | `a_clipped_ball_reads_the_same_as_the_full_array` |
| `test_R_room_is_what_the_room_does` | `r_room_is_what_the_room_actually_does` |
| `test_R_room_wall_factor_is_not_free` | `the_wall_closure_factor_in_r_room_is_not_free` |
| `test_zero_radiation_is_bit_identical_to_bare_body` | `zero_radiation_is_bit_identical_to_the_bare_body` |
| `test_absurd_coupling_stays_passive` | `an_absurd_coupling_stays_passive` |
| `test_passivity_across_grids_and_walls` | `passivity_holds_across_grids_and_wall_sets` |
| `test_open_face_port_is_refused` | `a_port_on_an_open_face_is_rejected` |
| `test_open_face_reached_by_a_BALL_is_refused` | `an_open_face_reached_by_the_ball_is_rejected` |
| `test_shared_node_is_refused` | `a_shared_node_is_rejected` |
| `test_overlapping_balls_are_refused` | `overlapping_balls_are_rejected` |
| `test_disjoint_ports_are_accepted` | `disjoint_ports_are_accepted` |
| `test_port_outside_the_room_is_refused` | `a_port_outside_the_room_is_rejected` |
| `test_unresolvable_radius_is_refused` | `an_unresolvable_radius_is_rejected` |
| `test_forgotten_room_step_raises` | `a_forgotten_room_step_is_rejected_and_recovers` |
| `test_forgotten_room_step_guard_is_per_port` | `the_forgotten_room_step_guard_is_per_port` |
| `test_sample_rate_mismatch_is_refused` | `a_sample_rate_mismatch_is_rejected` |
| `test_set_state_and_reset_clear_the_coupling_ledger` | `set_state_and_reset_clear_the_coupling_ledger` |
| `test_energy_is_an_override_not_a_delegation` | `the_energy_is_an_override_not_a_delegation` |
| `test_room_set_state_unsticks_every_port` | `a_room_set_state_unsticks_every_port` |
| `test_string_bridge_body_room_chain_conserves` | **moved, not replaced** — see §13.5 |
| — | `the_two_ledgers_agree_across_the_terminal` (new) |
| — | `the_epoch_and_not_the_step_count_is_what_unsticks_a_port` (new) |

### 13.5 One test could not be replaced, and it moved rather than died

`test_string_bridge_body_room_chain_conserves` drives `string -> StringBodyBridge -> RoomLoadedBody
-> AirBox`, and `StringBodyBridge` is one of the three classes still only in the binding. It is now
the last test in `tests/test_connection.py`, which is the bridge's own file and retires with it.
§12's harvesting rule says a survivor goes where its referent is; here the referent is the thing
blocking it.

### 13.6 The new bar's fixture was wrong, and a mutation found it

`the_two_ledgers_agree_across_the_terminal` was written with both instruments at interior nodes.
Deleting the `1 / (1 + beta)` factor from `r_room` — the trap the whole file exists to catch — left
it **green**, because at an interior node `beta` is zero and the factor is exactly 1. Moved to a
corner, it goes red. The bar as first written asserted a true thing about a fixture that could not
exercise it.

The same mutation is worth recording for what it did *not* break: with the factor deleted, all six
conservation and passivity bars stay green — the conserved total is structurally blind to it, exactly
as the retired file's header claimed, and now measured natively rather than quoted. Three bars catch
it and none of them is the drift.

### 13.7 Cost, and what is still only in the binding

The Python file was 3.39 s of suite time (`scripts/shard_costs.json`, entry removed with it); the 26
native bars run in **0.9 s** in the debug profile `cargo test` uses.

`PyRoomPort` and `PyRoomLoadedBody` are left exactly as they are, per §12's precedent. The viewer
builds a `RoomLoadedBody` (`web/serialize.py:9288`, the string→bridge→body→room scene) and
`tests/test_airbox_scene.py` and `tests/test_web_backend.py` both cover that path; rewiring the
binding classes to delegate to the core would touch a shipping path to no end. They go when the crate
goes.

### 13.8 The next batch

The ordering §13.1 derives makes it **`SurfacePort` and `InteriorSurfacePort`** — the distributed
tier of the same module, and the last thing between the retirement and the six grid wrappers. Their
kernels are already in core with bars behind them (`build_t`, `spread`, `patch_resistance`,
`load_matrix`, `footprint_unfed`); what is new is the same room-ownership shape applied to a port
that carries three sparse matrices, and one thing this batch did not have to face — `load_matrix` is
a sparse triple product whose **stored order**, not just its values, has to match, because SciPy's
ordering has caught this project out before.

**And one constraint this batch created for the bridge batch, written down now because §13.3 says
it is the kind that cannot be recovered later.** `RoomLoadedBody` overrides `energy()` and delegates
everything else; in the reference the attribute fallback made that automatic, which is exactly what
lets a bridge take a room-loaded body without knowing a room exists. Whatever trait or enum the
bridge batch puts over `StringBodyBridge`'s `body=` slot must therefore dispatch **`energy()` to the
wrapper** and the modal reads (`q`, `q_prev`, `m`, `omega`, `phi`, `a`, `bridge_displacement`) to the
inner `ModalBody`. Handing a bridge the inner body instead — the shape Rust makes easiest, because
`inst.body` is right there and has the modal surface on it — compiles, conserves nothing, and is
asserted against by exactly one test: the chain test §13.5 moved, which retires with the bridge
itself.

---

## 14. Phase C batch 3, done — the distributed tier, and the room's cut primitive

`SurfacePort` and `InteriorSurfacePort` now live in `crates/physsynth-core/src/airbox_port.rs`,
with `crates/physsynth-core/tests/airbox_surface.rs` (28 bars) replacing the **port-tier half** of
`tests/test_airbox_surface.py` — 15 test functions and five helpers, deleted in the same commit.
The file itself survives; §14.3 is why, and it is a precedent this migration has not set before.

Two of §13.1's three tiers are now home. The hole is down to **eleven classes across two files**:
the six grid wrappers plus their three surface adapters in `airbox_wrap.rs`, and the three bridges
in `connection.rs`.

### 14.1 The cut is not an optional extra, it is inside the port

`InteriorSurfacePort.__init__`'s last-but-one line is `room._register_cut(...)`, and the native
`AirBox` had only `cut_plane(axis, index)` — a full cross-section, no extents, no refusals, no
record of who owns the faces. So the batch could not port one of its two named classes without
first giving the room a **partial** cut, and that is not scope creep: it is the callee, the same
way §13 found the port tier underneath the wrappers.

What landed on `AirBox`:

* `cut_records: Vec<CutRecord>` — `room._cuts` reduced to what the refusal actually reads (the
  owner, the axis, the face set). `cuts` is the union every kernel walks and has lost that.
* `register_cut(owner, axis, index, i0, i1)` — the one writer, additive, with the shared-face
  refusal. Two hand-placed cuts may overlap (the mask is a boolean union); anything sharing faces
  with a **port**'s cut is refused.
* `add_cut(plane, index, extent)` — the public entry with its range refusals. `cut_plane` is now
  its unrestricted case rather than a separate implementation.
* `cut_faces()`.

`add_cut` is the part that was optional, and it is in because the next commit retires
`tests/test_airbox_cut.py` against it.

### 14.2 One refusal has NO analogue, and that is a third verdict

`test_refuses_a_malformed_extent` asserts two things and they translate differently:

| the reference refuses | in Rust |
|---|---|
| `add_cut("z", 3, ((0, 4), (3, 1)))` — `lo > hi` | `CutError::BadExtent`, carried over verbatim |
| `add_cut("z", 3, (4, 4))` — not a pair of pairs | **no analogue**: `Option<[[i64; 2]; 2]>` is the claim, made by the compiler |

The same thing happens three more times in this batch, and it is worth naming as a class rather
than as four incidents. A refusal about the **shape of a Python argument** does not port — it
becomes a type — while a refusal about a **value** does. The four:

* `coords must be an (n_surface, 2) array` → `&[[f64; 2]]`
* `origin must be an (o0, o1) pair` → `Option<(f64, f64)>`
* `unknown spreading 'cubic'` → `Spreading` is an enum
* `extent must be a ((lo0, hi0), (lo1, hi1)) pair` → `Option<[[i64; 2]; 2]>`

So §13.4's retirement table needs a third verdict beside "replaced" and "moved, not replaced": **no
analogue**. It is not a coverage hole, it is coverage that moved from a test to the type system —
but it must be *written down*, because the alternative is a later reader finding a Python refusal
with no native twin and concluding the port dropped it.

The length refusals that remain are the ones about a value: `areas` must be one per surface node,
and `q` must be the per-node (or per-face) vector, both of which a slice can get wrong.

### 14.3 The first PARTIAL retirement of a test file, and the line it draws

Every previous batch deleted a whole file. `tests/test_airbox_surface.py` cannot go: 19 of its 34
test functions drive `RoomLoadedPlate`, and the six wrappers have not been re-homed. So the file was
split, and the rule that split it is:

> A test retires when its referent is the **port** — it constructs one directly, or its every
> assertion reads a port attribute. A test that **drives a wrapper** stays, because the trajectory
> it asserts is the wrapper's even where the claim is named after the port.

That rule is sharper than "which class does it mention", and it had to be: `test_ledgers_agree` is
named after the port's resistance and is a wrapper test; `test_refuses_a_footprint_outside_the_face`
is built through `make_room_loaded_plate` and is a port test. The tell is `inst.step()`.

Two consequences, both stated rather than discovered.

**The core implementation is on no shipping path.** The binding's `PySurfacePort` and
`PyInteriorSurfacePort` are left exactly as they are, per §12's and §13.7's precedent — the wrapper
tier calls them, the viewer reaches them, and rewiring them to delegate would touch a shipping path
to no end. So the 28 native bars are the *only* thing exercising `airbox_port::SurfacePort`, and
their coverage has to be complete rather than representative; there is no Python test that might
happen to hit a branch.

**And the binding's own copies of the retired refusals are now untested.** `PySurfacePort` has its
own `accept_surface`, `check_footprint` and `check_in_plane_rim`, and the fifteen deleted tests were
what covered them. This is the same trade §13 made when it deleted `tests/test_airbox_port.py`
whole, and the argument is the same: the binding is dead code walking, testing it is testing what
phase F removes, and nobody edits a class that is being deleted. Recorded here so it is a decision
rather than an oversight.

One arm of it is worth naming rather than leaving inside that generality, because a later reader can
act on the specific and not on the general. `parse_spreading` distinguishes an **omitted**
`spreading` from an explicit `None`, and the binding's own comment records that the first draft had
the two arms backwards — silently building the bilinear default where the reference raises — and
that eleven tests in this very file caught it. The retired tests included every remaining
construction that omitted the argument (`_disk_port` and the `the_comb_verdict` sweep), and every
builder in `tests/helpers.py` passes `spreading="bilinear"` explicitly. **So the omitted-argument
arm of `parse_spreading` is now unreached by the suite**, measured by grep rather than assumed. The
*unknown*-spreading arm still is: `tests/test_airbox_dipole.py` asks for `spreading="cubic"` and
expects the refusal.

### 14.4 The one-time check: 392 quantities, 11 fixtures, zero differences

Same scenes through both implementations, before the Python went, and deliberately not committed.
Eleven fixtures — seven wall-mounted surfaces across four faces, three area patterns (uniform,
lumped-cell, and one with explicit zeros), two spreadings and three wall sets; four interior
surfaces on three planes. Per fixture, 34 to 42 quantities: `T`'s `indptr` / `indices` / `data`,
`R`, the load matrix's three arrays, `nodes` / `_flat` / `face_coords` and their **ordering**,
`origin`, `net_area`, `footprint_empty`, the interior tier's low/high split and `cut_faces` — then
a **200-step driven trajectory** (`free_pressure` → `q = alpha p` by hand → `inject` →
`room.step()`) compared over every pressure node, every velocity face, and all three room ledgers.

Every one of the 392 keys agreed to the bit. Unlike §13.3, this batch found **no** divergence: the
port tier contains no `np.dot`, so the fused multiply-add that §13.3 had to copy does not arise
here. The reductions that do arise are `np.sum`, which `crate::reduce::sum` already reproduces at
every length.

**The check had to be shown to be sensitive, and was.** A green comparison proves nothing until a
deliberate error turns it red, so two mutations were run against the same 392 keys:

* **The association folded right** (`T_ki (R_k T_kj)` instead of `(T_ki R_k) T_kj`): nine of the
  eleven fixtures went red, and the two that stayed green were exactly the two with
  `spreading = "nearest"`. That is the module docs' blind-fixture warning, measured — under nearest
  every stored entry in a row of `T` is the same uniform node area, and `(x d) x` and `x (d x)` are
  then the same double identically.
* **The divergence accumulated in reverse axis order**: all eleven went red, and the shape of the
  failure is §13.3's. On three fixtures `free_pressure` at step 0 still agreed and only the
  200-step field differed — a last bit at the terminal, amplified by the room. A check that
  compared only the constructed matrices would have passed all three.

The first mutation is now a permanent native bar
(`the_diagonal_folds_left_and_only_nearest_cannot_tell`), because the association is the one
decision here with no referent left after phase F: the bar recomputes the other association and
pins *which fixtures could tell the difference*, so a future reader cannot mistake "the nearest
fixture agrees" for "the association does not matter".

**What the check does NOT cover, and it is the wrapper batch's to measure.** The driven trajectory
never touches `load_matrix`: the solve that uses it belongs to the wrapper, which is not ported, so
the load matrix is verified statically (values, `indptr`, `indices`) and dynamically not at all.
Per §13.3's rule this is exactly the kind of thing that cannot be recovered later — see §14.7.

### 14.5 The retirement rule, discharged

| retired | native replacement |
|---|---|
| `test_T_distributes_every_node_area` | `t_distributes_every_surface_node_area` |
| `test_net_area_is_not_the_bounding_rectangle` | `the_net_area_is_not_the_bounding_rectangle` |
| `test_bilinear_equivariance_needs_centring` | `bilinear_equivariance_needs_centring` |
| `test_nearest_node_equivariance_is_an_accident` | `nearest_node_equivariance_is_an_accident` |
| `test_bilinear_assignment_is_exact_at_an_integral_grid_ratio` | `bilinear_assignment_is_exact_at_an_integral_grid_ratio` |
| `test_bilinear_beats_nearest_node_at_every_refinement` | `bilinear_beats_nearest_node_at_every_refinement` |
| `test_accepts_a_staircased_disk_the_bounding_box_refused` | `a_staircased_disk_the_bounding_box_refused_is_accepted` |
| `test_a_rectangle_is_judged_by_the_identical_required_set` | `a_rectangle_is_judged_by_the_identical_required_set` |
| `test_the_comb_verdict_changes_where_it_always_did` | `the_comb_verdict_changes_where_it_always_did` |
| `test_refuses_a_footprint_reaching_the_face_rim` | `a_footprint_reaching_the_face_rim_is_rejected` |
| `test_refuses_a_footprint_outside_the_face` | `a_footprint_outside_the_face_is_rejected` |
| `test_refuses_a_surface_too_coarse_for_the_air_grid` | `a_surface_too_coarse_for_the_air_grid_is_rejected` |
| `test_refuses_a_surface_on_an_open_face` | `a_surface_on_an_open_face_is_rejected` |
| `test_refuses_overlapping_ports` | `overlapping_surfaces_are_rejected` |
| `test_refuses_unknown_face_and_spreading` (face half) | `an_unknown_face_is_rejected` |
| `test_refuses_unknown_face_and_spreading` (spreading half) | **no analogue** — §14.2 |
| `test_load_matrix_is_symmetric_and_the_cost_is_reported` | **split**: the symmetry half is `the_load_matrix_is_symmetric_but_not_symmetrised`; the `lu_nnz` half is the wrapper's and the Python test **stays** |
| `test_free_pressure_matches_full_array` | **stays** (wrapper-driven), and `the_local_free_pressure_read_is_the_full_array_update_exactly` asserts the same claim natively over six faces and both interior planes |
| `test_two_disjoint_surfaces_share_one_room` | **stays** (wrapper-driven); its port-tier half is `two_disjoint_surfaces_share_one_room` |
| `test_refuses_solving_twice_without_a_room_step` | **stays** (it calls `inst.step()` twice); `solving_twice_without_a_room_step_is_rejected` is the native bar |
| — | `the_diagonal_folds_left_and_only_nearest_cannot_tell` (new — §14.4) |
| — | `the_two_node_planes_carry_one_q_with_opposite_signs` (new) |
| — | `the_interior_surface_cuts_exactly_the_faces_it_covers` (new) |
| — | `a_refused_interior_port_leaves_the_room_untouched` (new — §14.6) |
| — | `an_interior_surface_over_a_hand_placed_cut_is_rejected` (new) |
| — | `an_interior_index_that_reaches_a_wall_is_rejected` (new) |
| — | `a_q_of_the_wrong_length_is_rejected` (new) |
| — | `malformed_areas_are_rejected` (new) |
| — | `the_epoch_and_not_the_step_count_is_what_unsticks_a_surface_port` (new — §13.2's invariant, restated where it can regress independently) |

### 14.6 Refusal ORDER is a bar, and the fixture has to be one that could fail

`InteriorSurfacePort` is the only port that **writes** its room — it registers a cut — and the
reference got the ordering for free: `_cut` and `_register` are the last two lines of `__init__`,
which runs after everything that can raise. In the value-owned shape there is no such split, so
every refusal has to be spelled out ahead of the two writes, and a happy-path test cannot see it
being wrong.

`a_refused_interior_port_leaves_the_room_untouched` is that bar, and per §13.6 its fixture is one
where **the cut would otherwise have registered**: a second surface on the same faces is refused for
disjointness — the last refusal before the cut — so if the cut ran first the room would be left
carrying a partition belonging to an object that does not exist. The bar asserts `room.claims`,
`room.cuts` and `room.cut_records` are all exactly as they were.

The room's own shared-face refusal reaches the port too, and gets its own bar
(`an_interior_surface_over_a_hand_placed_cut_is_rejected`) — it is the one rejection an interior
surface inherits rather than raises, and it also arrives *after* every check the port makes itself,
so it is the case that proves the cut is genuinely last.

### 14.7 Cost, and the constraint this batch creates for the wrapper batch

The Python file was 6.41 s of CI suite time (`scripts/shard_costs.json`) and the deletion **did not
move it**: measured within one machine, the trimmed file runs at 1.04x the original, i.e. inside the
noise. The retired tests were the cheap ones — the expensive tests in that file are the wrapper's
200-step trajectories, and all of them stayed. So the cost entry is left as it is rather than
scaled, and this is worth generalising: **a partial retirement does not reduce a file's cost in
proportion to its test count**, because test count and test cost are uncorrelated within a file.
The 28 native bars run in 0.4 s in the debug profile.

**And one thing the wrapper batch cannot recover, written down now because §13.3 says that is when
it has to be.** The six `splu` calls in `airbox_wrap.rs` are the only consumers of
`port.load_matrix`, and they are plan §4's sparse-LU risk group. The load matrix's **values** and
**stored order** are verified against SciPy by §14.4 and can be verified again as long as Python is
installed, but *how they reach the factorization* — whether `a_loaded = a_bare + load` is formed by
adding two CSR matrices or by assembling one, and in which order — is a decision only the wrapper
batch makes, and the reference for it disappears with the wrapper. That comparison has to happen in
the batch that ports the wrappers, on a scene that actually solves, and it cannot be deferred.

### 14.8 The next batch

The wrappers. §13.1's ordering has nothing left below them: the ports are home, so the six
`RoomLoaded*` / `RoomSuspended*` classes and the three `_*Surface` adapters can take a
`SurfacePort` or an `InteriorSurfacePort` by value. Two things about that tier are known in
advance and neither is small — it owns all six factorizations (plan §4's risk group, and §14.7 is
the measurement it must make), and §13.8's constraint stands: whatever trait or enum covers a
wrapper's `plate=` / `body=` slot must dispatch `energy()` to the wrapper and the modal reads to
the inner model.

Before it, one short commit: `add_cut`'s validation layer is already in, so
`tests/test_airbox_cut.py` (14 functions, 1.37 s) retires against native bars for the cut
primitive — the room-tier file this batch's callee made portable.

---

## 15. The cut's own file, retired against the primitive §14 had to build

`tests/test_airbox_cut.py` (14 functions, 42 parametrized cases, 1.37 s) is gone, replaced by
`crates/physsynth-core/tests/airbox_cut.rs` (11 bars, 0.6 s). It is the **room**-tier file that
§14.1's callee made portable: the cut primitive landed in `physsynth-core` because
`InteriorSurfacePort` could not be ported without it, and once it was there the file testing it had
a native home.

Nothing here is a port of a model — it is the same deletion §39–§49 did twenty-three times, one
commit late, and it is short. Three things worth recording anyway.

**The randomness had to be replaced, and that is fine.** The reference seeds the room from NumPy's
PCG64 (`airbox_noise`) and drives it from another draw. Neither claim is about the *numbers* — both
are about a field with no structure, so that an energy identity has no direction left to hide in —
so the bars use a splitmix64 hash of the flat index instead. That is the opposite of the exactness
discipline everywhere else in this migration and it is correct here: **a bar that asserts a
tolerance on a broadband field is asserting a property of the class of fields, not of one field.**
The tell is that nothing in the retired file compared two runs.

**The modal oracle is the file's whole value, so its half-cell was mutation-tested.** The sub-rooms
of a fully cut room are `(m + 1/2) h` and `(N - m - 1/2) h`, and the cut end is *face*-centered, so
the exact eigenvector along the cut axis is `cos(n pi i / (m + 1/2))` — not the room's own
`cos(n pi i / N)`. Getting that half wrong looks like scheme inaccuracy rather than like a bug,
which is exactly why the reference asserted it at machine precision. Removing the `0.5` from both
sides of the native bar takes the field error from below `1e-12` to **2.49** against an amplitude of
1, so the half-cell is load-bearing and the bar is not passing on a technicality.

**A negative about a floating-point result is not an assertion.** The reference checked that the
sub-room lengths land half a cell off the grid by asserting `(lo / h) % 1 != 0.0` — which passes for
*any* wrong value, not only for the right one, and needs no tolerance precisely because it is
claiming nothing. The native bar asserts the positive instead, `|(lo/h).fract() - 0.5| <= 1e-12`,
and that immediately needed the tolerance: the first fixture returns `5.500000000000002`, so an
exact `== 0.5` would have been a claim about the round trip rather than about the geometry. Read
every `!=` in a retiring file as a candidate for this.

**A `continue` is silent where `pytest.skip` reports.** Two of the bars sweep a nested loop and skip
combinations the fixture room is too small for. In pytest a skipped case is *printed*; in Rust it
vanishes, so a future fixture change could empty the loop and leave the bar green having tested
nothing. Both now count what ran and assert the count (`18` and `9`). Any loop-with-`continue`
carried over from a parametrized test needs one.

**Two helpers lost their only caller and went with the file.** `make_cut_room` and `sub_room_mode`
in `tests/helpers.py` were used by nothing else (`airbox_noise` was, by two other files, and stays).
§12's harvesting rule cuts both ways: a survivor moves to where its referent is, and a helper whose
referent has gone is residue.

The refusal that has **no analogue** is §14.2's: `extent must be a ((lo0, hi0), (lo1, hi1)) pair` is
a claim about the shape of a Python argument and `Option<[[i64; 2]; 2]>` is the same claim made by
the compiler. The half of that test about a *value* — a range running backwards, past the axis, or
starting below zero — is carried over and widened from one case to three, with the room asserted
unmarked afterwards.

### 15.1 The retirement rule, discharged

| retired | native replacement |
|---|---|
| `test_a_cut_room_still_conserves` | `a_cut_room_still_conserves` |
| `test_a_cut_face_carries_no_velocity_at_any_half_step` | `a_cut_face_carries_no_velocity_at_any_half_step` |
| `test_the_sub_rooms_have_exact_half_offset_modes` | `the_sub_rooms_have_exact_half_offset_modes` |
| `test_the_sub_room_lengths_sum_to_the_room` | `the_sub_room_lengths_sum_to_the_room` |
| `test_a_full_cut_isolates_exactly` | `a_full_cut_isolates_exactly` |
| `test_cuts_are_additive_so_a_second_one_cannot_un_block_the_first` | `cuts_are_additive_so_a_second_one_cannot_un_block_the_first` |
| `test_overlapping_hand_placed_cuts_are_idempotent` | `overlapping_hand_placed_cuts_are_idempotent` |
| `test_cut_faces_counts_the_full_cross_section` | `cut_faces_counts_the_full_cross_section` |
| `test_refuses_an_unknown_plane` | `an_unknown_plane_is_rejected` |
| `test_refuses_a_cut_index_outside_the_faces` | `a_cut_index_outside_the_faces_is_rejected` |
| `test_refuses_a_malformed_extent` (the value half) | `a_malformed_extent_is_rejected`, widened to three cases |
| `test_refuses_a_malformed_extent` (the shape half) | **no analogue** — §14.2 |

The next batch is still the wrappers (§14.8).
