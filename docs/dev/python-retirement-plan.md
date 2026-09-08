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

The map's revised recommendation for the next batch: **not** `string_damped`, which the first
reading suggested and which turns out to have four native bars already. The genuinely unwritten
model is **sympathetic strings** — 15 functions, zero native, and structurally simple (N linear
strings sharing one bridge point on a modal body, no nonlinearity and no solver of its own). It is
the smallest complete phase-C batch available and it exercises the whole retirement ritual once:
write the native bars, name them in the commit that retires the Python ones, delete.

**D**, the viewer, remains independent of all of this and is still the longest pole.

---

## 11. The hole: fourteen model classes have no home outside the binding crate

Found on the first phase-C batch, 2026-09-08, and it is a correction to §1 and §7 rather than a
detail of that batch.

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

**Cost.** The Python file was 59.11 s of suite time; the 18 native bars run in **2.1 s** in the
debug profile `cargo test` uses.

**What is still only in the binding for this model.** `PySympatheticStrings` was left exactly as it
is: the viewer builds it through Python, and `tests/test_web_backend.py`'s 22 sympathetic bars —
including the antisymmetric zero, both transfer arms and the guard refusal — go on covering that
path until phase D moves the viewer. Rewiring the binding class to delegate to `core::connection`
would touch a shipping path to no end; it goes when the crate goes.
