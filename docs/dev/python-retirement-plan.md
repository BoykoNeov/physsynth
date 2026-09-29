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

**Done at §22** — and this table's sharding row was wrong about *when*: those scripts run the suite,
so they leave at F with it (§22.4).

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
and the three bridges. **Batch 4 is done** (§16) and takes the *linear* half of the wrapper
tier — the ordinary plate and the drumhead, four wrappers and two seams, re-homed as one generic
`RoomGrid<S>` — leaving **six**: the von Kármán seam with its two wrappers, and the three bridges.
Read §13.1's derive together with §16.7's re-homing table from here on: the derive is a *name join*,
and §16.2 renamed six classes on purpose, so it now over-reports. **Batch 5 is done** (§17) and
finishes the wrapper tier with the von Kármán seam, as a third seam of the same generic. What is
left is the **three bridges and one thing no class count could see**: the mallet's gong-in-a-room
*mode* (§17.5), a composition that lives inside `MalletVKPlate` rather than in a class of its own.
**Batch 6 is done** (§18) and takes that mode as its own type, leaving only the three bridges.
**Batch 7 is done** (§19, §20): `StringBodyBridge` is native, generic over the four bodies its slot
took, and the two plate bridges are one generic `StringPlateBridge`. **The hole is closed** — nothing
is implemented only in the binding.

§11's table is **corrected by §13.1**: the hole is 16 classes across *three* binding files, not 14
across two, and the file it missed is the one underneath the other two.

**Phase A is done** (§21): `PHYSSYNTH_RS` is read by nothing, `banded.py` and its binding are gone, and
the `rust-harness` CI job with them. The default suite is now what the flagged suite was.

**Phase E is done** except what cannot go yet (§22): 36 of `scripts/`'s 41 files and
`physsynth/viz/` are deleted, matplotlib is no longer a dependency, and the one measurement worth
keeping is a Cargo example. The sharding scripts leave with the suite at F; the browser check with D.

**D**, the viewer, remains independent of all of this and is still the longest pole. **Its first
batch is done** (§23): `crates/physsynth-viewer` exists, serves the untouched front-end, and renders
the three linear strings; the other nineteen scenes follow model by model in the order §23.7 derives,
and the Python server stays the live viewer until the last of them lands. **Batch D2 is done**
(§23.8): the tension string, both regimes, and the bow. **Batch D3 is done** (§23.9): sympathetic,
geometric, reed, radbody and airload. **Batch D4 is done** (§23.10): body, jawari, juari and
fret, plus an arbitrary-length `rfft` in the analysis crate. **D5 is under way** (§23.11-§23.13): a
native eigensolver, then membrane, mallet, all three plates, the bore, the von Kármán plate and
the plate body. **D5 and D6 are done** (§23.11-§23.18): **all twenty-two keys are native**. **D7 is
done** (§23.19): the Python viewer's outputs are frozen in `crates/physsynth-viewer/tests/frozen.rs`
(exact on the recording platform, structure elsewhere), the browser check is a Cargo example, and
`web/*.py` and `tests/test_web_backend.py` are deleted. **Phase D is done.**

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

---

## 16. Phase C batch 4, done — the LINEAR wrapper tier, and the audit stops being computable

`PlateSeam`, `MembraneSeam`, `GridPort` and `RoomGrid<S>` now live in
`crates/physsynth-core/src/airbox_wrap.rs` beside §13's `RoomLoadedBody`, with
`crates/physsynth-core/tests/airbox_grid.rs` (45 bars) replacing `tests/test_airbox_membrane.py`
(20 functions, 545 lines, deleted whole) and the wrapper halves of `tests/test_airbox_surface.py`
(19 -> 1) and `tests/test_airbox_dipole.py` (28 -> 1).

**Split, the human's call.** The tier is 9 classes and ~106 test functions, larger than any batch
this migration has done in one commit, so it lands in two: the **linear** pair — the ordinary plate
and the drumhead, four of the six wrappers and two of the three seams — here, and the von Kármán
seam with its two wrappers next. Two things were checked before the split rather than after: §48's
half-state hazard does **not** apply (nothing Python is being *deleted*, only test files retired, so
there is no `deleted_bodies`-shaped identity table to leave half-asserted), and the one bit-identity
tie across the halves — `RoomLoadedVKPlate(nonlinear=False)` must reproduce `RoomLoadedPlate` under
`array_equal` — lives in `test_airbox_vk.py`, which retires in the *second* commit with both sides
already native.

### 16.1 §14.7's unrecoverable measurement, made and discharged

§14.7 said the six `splu` calls are the only consumers of `port.load_matrix`, that *how* `a_loaded`
reaches the factorization is a decision whose reference disappears with the wrapper, and that the
comparison could not be deferred. It was made first, on scenes that actually solve, for all six
wrappers — and it comes out better than §14.7 feared.

The reference's expression is `(a_bare + load_matrix * load_scale).tocsc()` then
`eliminate_zeros()`. Three findings, in order of how much they were worth:

1. **`eliminate_zeros()` never fires.** Measured at every reachable fixture — all six wrappers at
   the shipped defaults, plus `nearest` spreading, a grid-aligned plate, an integral-offset origin,
   and the zero-**area** surface the call's own comment says it exists for: `nnz` before and after
   is the same number every time. The reason is structural rather than lucky. `spread` drops a
   zero weight, so `T` carries no explicit zero; `R` is strictly positive; and `T^T R T` is
   therefore a Gram matrix over positive weights, which cannot cancel to an exact zero. On the
   zero-area path `T` has `nnz = 0` and there is nothing to eliminate at all.
2. **The native assembly reproduces it by construction, not by imitation.** `Csr::add` builds
   through `Csr::from_rows`, which drops an entry whose value is exactly `0.0` — so the elimination
   is *inside* the add. SciPy's `csr_binop_csr` does the same thing (it writes a result only when
   it is nonzero). One line, `a_bare.add(&port.load_matrix().scaled(load_scale))`, and both the
   arithmetic and the zero-dropping match.
3. **So `nnz_growth` and `lu_nnz` are the same quantities, not redefinitions**, and this is worth
   saying explicitly because it was the one place where a plausible number could have been quietly
   wrong in either direction. The native values agree with the reference's exactly:
   `nnz_growth = 2.8642714570858283` and `lu_nnz = 1911` for the plate on both tiers,
   `12.484444444444444` and `9345` for the membrane on both.

One structural fact worth recording alongside: the load's sparsity pattern **subsumes** the plate's
under `bilinear` (501 of 1435 entries) but does **not** under `nearest` (255 load entries, 501 bare,
533 in the sum) — so the assembly genuinely needs a union merge and a bar that only ever ran
`bilinear` would not have discovered that.

### 16.2 Six classes become two generic types, and that is not a simplification

The reference has six wrappers because Python has no generics: `RoomLoadedPlate`,
`RoomSuspendedPlate` and the four others are one body of arithmetic with two enum arms in it (which
face the surface is mounted on, and therefore which port it drives) and one attribute name that
differs. The binding says so in its own comment — "the reference's six wrappers are one class with
two enum arms". Natively that is expressible: `RoomGrid<S>` over the seam, with the tier carried by
`GridPort`, and `baffled` / `suspended` constructors.

It loses no distinction the reference made — `test_airbox_vk.py` anchors the two tiers against each
other with `array_equal`, which is only meaningful because they *are* one transcription — and it
removes five chances to drift. The three decisions that propagate:

* **The seam is a trait, `GridSeam`**, with the load-bearing member being `u_prev`: a live read the
  caller must take *before* `commit` rolls it away, which is why `RoomGrid::prepare` copies it out
  rather than reading it again later in the step.
* **The wrapper owns its resonator and the room is passed at each call**, which is §13.2's decision
  unchanged and what lets two instruments share one room.
* **`RoomGrid::refactor` replaces a manoeuvre the reference did by hand.** Five retired tests
  rebuilt `A_loaded` from scratch in Python to install a deliberately wrong coupling
  (`inst._lu_loaded = splu(...)`), which was a *second spelling* of `build`'s assembly and could
  have drifted from it. `refactor()` rebuilds through the one spelling. The port's four written
  slots — §31.6 of the Rust migration plan found `T`, `load_matrix`, `R` and `areas` are written by
  tests — kept their native setters for the same reason.

### 16.3 The one-time check: 72 quantities over four scenes, and the ONE thing that is not exact

Same scenes through both implementations, before the Python went, deliberately not committed. Four
scenes (plate and membrane, baffled and suspended), 200 hand-driven coupled steps each plus one
`f_ext`-driven step, 18 quantities apiece: `a_bare`, the load matrix and `T` as full
`indptr`/`indices`/`data`, `R`, the derived scalars, and per-step `energy`, `volume_velocity`,
`radiated_energy` and room energy, then the final resonator state, `pbar`, the per-node volume
velocity and the room's **whole** pressure field.

**Sixty-five of the seventy-two are bit-identical.** The seven that are not are the coupling ledger
`radiated_energy` and the `energy()` that contains it, at `1.3e-16` to `1.3e-15` relative.

The initial conditions were built from integers on purpose — `1e-3 ((i mod 7) - 3) / 8` rather than
a Gaussian bump — so that both sides start from the identical doubles instead of from NumPy's `exp`
and Rust's. That is the standing `libm`-dispatch finding applied in advance rather than diagnosed
afterwards.

**The ledger is `k * np.dot(pbar, q)` and `np.dot` is BLAS `ddot`.** Probed at the actual operand
lengths: a serial `s += x*y` reproduces it exactly on two of the four scenes, `f64::mul_add` on
three, and **nothing tried reproduces it on the fourth** (`n = 169`, suspended membrane) — not a
serial fold, not a fused one, not `np.sum(a*b)`, not 2-, 4- or 8-way blocking. So this is the
migration's standing "bit-identity ends at a BLAS reduction" finding, met again.

What is new is the *disposition*, and it is decided by a measurement rather than by preference:
**the ledger does not feed back.** Over 200 steps of all four scenes the resonator's state, the
room's entire pressure field, `pbar` and the per-node volume velocity are bit-identical while this
number is not — which is what "does not reach the timestep" looks like when it is measured instead
of argued. So it takes the crate's documented **read-out** spelling, `plate::dot`'s plain
`s += x * y`, and not this module's `mul_add` one (which exists because `RoomLoadedBody`'s single
`dot` *does* reach the timestep). Two reasons, neither a preference: the ledger reaches nothing, and
the identity it exists for is `radiated == injected`, whose other side the room books with exactly
this spelling. **Matching the room matters more than matching a `ddot` that is about to stop
existing.**

### 16.4 Three mutations, and the one that CANNOT be seen is the finding

A passing comparison proves nothing unless a wrong version fails it (§14.4's discipline).

| mutation | effect |
|---|---|
| re-associate the two room load terms, `base + (-a + b)` for `(base - a) + b` | **all four scenes red**, on every quantity: 49/49 plate nodes, 1559/1560 room nodes, 200/200 volume velocities |
| take the interior port's pressure jump the other way round, `lo - hi` | **exactly the two suspended scenes red**, on everything; the two baffled ones untouched |
| the suspended closure's operand order, `2 (R q)` for `(2 R) q` | **nothing, anywhere** |

The third is the one worth recording. The binding carries a careful comment pinning that operand
order — "the operand order is the reference's, `(2 R) q` rather than `2 (R q)`" — and it **cannot
matter**: multiplying a double by 2 is exact, and rounding is scale-invariant in binary floating
point, so `round(2 r q) = 2 round(r q)` for any operands short of overflow or subnormals. The
general form, and the third entry on this migration's list of operand-order hazards:

> **An operand-order note is load-bearing only when the constant is not a power of two.** A
> transcription that pins `(2 R) q` is pinning nothing; one that pins `(0.5 k) / rho` against
> `0.5 (k / rho)` is pinning a real bit.

A fourth divergence was found by the same route and is *not* a mutation: the two tiers'
**load matrices are bit-identical**, because an `InteriorSurfacePort`'s `R` is exactly half a
`SurfacePort`'s (a face in the interior sees half the node weight of one on a wall) and the
suspended tier doubles it back. So a bar comparing the two tiers' load matrices is not a
discriminating test, and `airbox_grid.rs` says so where the enum is defined rather than shipping
one.

### 16.5 A translation hazard with teeth: `np.allclose`'s default `rtol`

`test_the_default_origin_centres_a_disk` asserted `np.allclose(load[perm][:, perm], load,
atol=1e-18)`, and read as written that is a claim at `1e-18` **absolute**. It is not:
`np.allclose` keeps its default `rtol = 1e-5` unless told otherwise, so the bar that actually ran
was `|a - b| <= 1e-18 + 1e-5 |b|`. Carried over as its `atol` alone the native bar is seven orders
of magnitude stricter than the claim it stands for, and it **failed** — the triple product's
summation order leaves a node and its 180-degree image 2e-15 apart *relatively*.

The rule, which will recur across the remaining files: **read a translated `allclose` for the
tolerance it did not write down.** An `atol=` with no `rtol=` is almost always a bar the author
believed was absolute and that NumPy ran as relative. The native bar keeps the relative form and
sets it at `1e-12`, still seven decades tighter than what shipped.

### 16.6 Two more verdicts of "no analogue", and they are different kinds

§14.2 established a third verdict beside *retire* and *stays*: a claim with **no native analogue**.
Two more, and they fail for different reasons:

* **`test_a_sign_flip_is_invisible_to_every_energy_quantity`, the suspended arm** worked by
  *replacing two of the port's methods on the live instance*
  (`port.free_pressure = lambda: tuple(reversed(free()))`). The binding deliberately keeps that seam
  alive by giving every port class a `dict` (Rust migration plan §31.6). A value-typed native port
  has no such seam and no way to grow one that is not production API written for a test. What
  survives: the *baffled* arm of the same claim, which is expressible through `set_t`, and the
  detector the claim exists to justify — `the_sign_is_readable_on_the_first_step`, on three planes
  and both boundaries. **The general form: a test that works by replacing a live object's methods
  has no analogue once the object becomes a value.**
* **`test_there_is_no_pressure_readout_and_that_is_deliberate`** asserted `not hasattr(inst,
  "pressure")` on the membrane wrapper. Natively `pressure()` is an inherent method on
  `RoomGrid<PlateSeam>` alone, so the membrane wrapper does not have one and *cannot be asked* — a
  bar asserting its absence would not compile. This is §14.2's shape verdict again: **a claim about
  what an object does not have becomes a type, and a type is not a test.** Likewise the delegation
  half of `test_delegates_to_the_membrane_and_overrides_energy`: Python's `__getattr__` forwarding
  has no analogue because Rust has no attribute fallback (§13.8), and the caller reads
  `inst.seam.membrane.u` directly. The `energy()`-override half is a real claim and is carried.

### 16.7 The audit that made the hole computable has stopped being computable

§13.1's three-step derive — every `#[pyclass(name = ...)]` minus every core `pub struct`/`pub enum`,
then clear the case-spelling false positives — is a **name join**, and it was correct exactly as
long as every port was 1:1 and name-preserving. This batch broke that assumption on purpose (§16.2),
so re-running the derive today still lists all twelve remaining names, six of which are re-homed:

| Python-facing class | native home |
|---|---|
| `_PlateSurface` | `airbox_wrap::PlateSeam` |
| `_MembraneSurface` | `airbox_wrap::MembraneSeam` |
| `RoomLoadedPlate` | `airbox_wrap::RoomGrid<PlateSeam>::baffled` |
| `RoomSuspendedPlate` | `airbox_wrap::RoomGrid<PlateSeam>::suspended` |
| `RoomLoadedMembrane` | `airbox_wrap::RoomGrid<MembraneSeam>::baffled` |
| `RoomSuspendedMembrane` | `airbox_wrap::RoomGrid<MembraneSeam>::suspended` |

**The remaining hole is six** — and §10/§14.8's "eleven classes across two files" was itself a
miscount of **twelve** (six wrappers + three surface adapters + three bridges), so do not read
11 − 6 = 5 off it: `_VKPlateSurface`, `RoomLoadedVKPlate`, `RoomSuspendedVKPlate`, and
the three bridges `StringBodyBridge`, `StringPlateBridge`, `StringVKPlateBridge`. Read §13.1's
derive together with this table from here on; it is a *lower* bound on what is done, not a
statement of what is left. The lesson is general and cheap to state: **an audit built on a name
join measures the port's fidelity to a naming convention, not to a contract**, and it expires the
first time a port is allowed to be a better shape than its original.

### 16.8 The retirement rule, discharged

`tests/test_airbox_membrane.py` — **deleted whole**.

| retired | native replacement |
|---|---|
| `test_zero_area_reduces_to_the_bare_membrane` | `zero_area_reduces_to_the_bare_membrane` |
| `test_ledgers_agree_and_the_channel_is_not_vacuous` | `the_heads_two_ledgers_agree_and_the_channel_is_not_vacuous` |
| `test_the_scene_total_is_flat` | `the_heads_scene_total_is_flat` |
| `test_the_channel_shows_the_acoustic_short_circuit` | `the_channel_shows_the_acoustic_short_circuit` |
| `test_the_coupled_residual_catches_both_wrong_2s` | `the_heads_coupled_residual_catches_both_wrong_2s` |
| `test_a_lossy_head_in_a_lossy_room_is_monotone` | `a_lossy_head_in_a_lossy_room_is_monotone` |
| `test_R_is_the_rooms_own_differential_response` | `r_is_the_rooms_own_differential_response` |
| `test_the_f_ext_term_is_pinned_twice` | `the_f_ext_term_is_pinned_twice` |
| `test_the_lagged_explicit_load_is_caught_only_by_the_total` | `the_lagged_explicit_load_is_caught_only_by_the_total` |
| `test_the_cut_follows_the_port_and_the_two_areas_differ` | `the_cut_follows_the_port_and_the_two_areas_differ` |
| `test_the_default_origin_centres_a_disk` | `the_default_origin_centres_a_disk` (§16.5) |
| `test_refuses_a_sample_rate_mismatch_and_names_the_membrane` | `a_head_rate_mismatch_is_rejected_and_names_the_membrane` |
| `test_the_membrane_cfl_is_the_models_own_refusal` | `the_membrane_cfl_is_the_models_own_refusal` |
| `test_delegates_to_the_membrane_and_overrides_energy` (the override) | `the_head_overrides_energy_rather_than_delegating_it` |
| `test_delegates_to_the_membrane_and_overrides_energy` (the delegation) | **no analogue** — §16.6 |
| `test_reset_clears_the_ledger_but_not_the_geometry` | `reset_clears_the_ledger_but_not_the_geometry` |
| `test_two_heads_share_one_room` | `two_heads_share_one_room` |
| `test_there_is_no_pressure_readout_and_that_is_deliberate` | **no analogue** — §16.6 |
| `test_refuses_a_head_that_overruns_the_plane` | already native, §14: `a_footprint_reaching_the_face_rim_is_rejected` |
| `test_refuses_solving_twice_without_a_room_step` | `solving_twice_without_a_room_step_is_rejected` |
| `test_refuses_overlapping_ports` | already native, §14: `overlapping_surfaces_are_rejected` |

`tests/test_airbox_surface.py` — 19 functions to **1**.

| retired | native replacement |
|---|---|
| `test_ledgers_agree` | `the_two_ledgers_agree` (both tiers, 12 cases) |
| `test_conservation_is_blind_to_a_wrong_R` | `conservation_is_blind_to_a_wrong_r` |
| `test_R_j_is_what_the_room_does` | `r_j_is_what_the_room_does` |
| `test_volume_is_conserved_exactly` | `volume_is_conserved_exactly` (both tiers) |
| `test_coupled_scheme_residual` | `the_coupled_scheme_residual_vanishes` |
| `test_no_air_load_reproduces_the_bare_plate` | `no_air_load_reproduces_the_bare_plate` (both tiers) |
| `test_sign_convention_is_uniform_over_faces` | `the_sign_convention_is_uniform_over_all_six_faces` |
| `test_a_sign_flip_is_invisible_to_every_energy_quantity` | `a_sign_flip_is_invisible_to_every_energy_quantity` |
| `test_scene_total_is_flat` | `the_scene_total_is_flat` (both tiers) |
| `test_lossy_plate_scene_is_monotone` | `a_lossy_plate_scene_is_monotone` |
| `test_free_plate_piston_is_fully_radiated` | `a_free_plate_piston_is_fully_radiated` |
| `test_load_matrix_is_symmetric_and_the_cost_is_reported` (cost) | `the_factorization_cost_is_reported` |
| `test_load_matrix_is_symmetric_and_the_cost_is_reported` (symmetry) | already native, §14: `the_load_matrix_is_symmetric_but_not_symmetrised` |
| `test_an_even_mode_is_silent_to_every_one_port_and_still_radiates` | same name — **the headline** |
| `test_the_silence_is_a_property_of_the_whole_scene` | same name |
| `test_free_pressure_matches_full_array` | already native, §14: `the_local_free_pressure_read_is_the_full_array_update_exactly` |
| `test_refuses_a_sample_rate_mismatch` | `a_sample_rate_mismatch_is_rejected` |
| `test_refuses_solving_twice_without_a_room_step` | `solving_twice_without_a_room_step_is_rejected` |
| `test_two_disjoint_surfaces_share_one_room` | `two_disjoint_surfaces_share_one_room` |
| `test_string_bridge_plate_room_chain` | **stays** — drives `StringPlateBridge`, not ported |

`tests/test_airbox_dipole.py` — 28 functions to **1**.

| retired | native replacement |
|---|---|
| `test_ledgers_agree` | `the_two_ledgers_agree`, suspended arm |
| `test_the_piston_is_the_non_vacuous_channel` | `the_piston_is_the_non_vacuous_channel` |
| `test_the_coupled_residual_catches_both_wrong_2s` | `the_coupled_residual_catches_both_wrong_2s` |
| `test_each_ledger_is_blind_to_a_different_wrong_2` | `each_ledger_is_blind_to_a_different_wrong_2` |
| `test_R_j_is_the_same_on_both_planes_with_opposite_signs` | `r_j_is_the_same_on_both_planes_with_opposite_signs` |
| `test_scene_total_is_flat` | `the_scene_total_is_flat`, suspended arm |
| `test_volume_is_conserved_exactly` | `volume_is_conserved_exactly`, suspended arm |
| `test_the_channel_is_a_reservoir_not_a_drain` | `the_channel_is_a_reservoir_not_a_drain` |
| `test_the_sign_is_readable_on_the_first_step` | `the_sign_is_readable_on_the_first_step` |
| `test_a_mirror_symmetric_scene_gives_pbar_lo_equal_to_minus_pbar_hi` | same name |
| `test_the_source_alone_converges_to_silence` | same name — **the headline** |
| `test_the_phantom_is_bit_identically_two_monopoles` | same name |
| `test_no_air_load_reproduces_the_bare_plate` | `no_air_load_reproduces_the_bare_plate`, suspended arm |
| `test_two_suspended_plates_share_one_room` | `two_suspended_plates_share_one_room` |
| `test_surface_port_is_unchanged_...` (construction digests) | `the_baffled_port_construction_digests_are_unchanged` |
| `test_surface_port_is_unchanged_...` (200-step run-end values) | **not re-frozen** — spent, see below |
| `test_the_interior_port_can_never_touch_an_open_face` | `the_interior_port_can_never_touch_an_open_face` |
| `test_a_sign_flip_is_invisible_to_every_energy_quantity` | **no analogue** — §16.6 |
| `test_refuses_a_sample_rate_mismatch` | `a_sample_rate_mismatch_is_rejected` |
| `test_refuses_solving_twice_without_a_room_step` | `solving_twice_without_a_room_step_is_rejected` |
| the eight remaining `test_refuses_*` | already native, §14 (index, rim, outside-plane, too-coarse, overlap, hand-placed cut, unknown plane/spreading, `q` length) |
| `test_string_bridge_plate_room_chain` | **stays** — drives `StringPlateBridge`, not ported |

**The goldens are two thirds carried and one third spent, and the split is not arbitrary.**
`SURFACE_GOLDEN`'s construction digests — `node_count`, `nnz(T)` and the index-weighted sums
`sum_i a_i i`, which move by order unity under *any* permutation of the data — are carried over
**unchanged**, asserted against the reference's own Windows-captured numbers. That is a free
cross-implementation check on top of §16.3: the native `T` and load matrix reproduce digests
measured through SciPy, at six cases including `nearest`, an off-centre origin and a `y1` face. The
200-step run-end values are **not** re-frozen. They had already been downgraded from `==` to a
tolerance because Windows and Linux differ in the last ULP, the refactor they were written to guard
shipped long ago, and re-recording them from this machine would promote a pile of incidental digits
to a cross-machine claim — §15's hazard, ledger #68, exactly.

**Two helpers-file consequences.** Six membrane fixtures (`make_air_membrane`, `make_membrane_room`,
`make_room_loaded_membrane`, `make_suspended_membrane`, `membrane_bump`, `membrane_bulge`) and the
seven `AIRBOX_MEMBRANE_*` constants lost their only caller and went with the file — §15's rule, 117
lines. `make_room_loaded_plate` and `make_suspended_plate` **stay**, each now with exactly one
caller: the bridge test that outlived its file.

### 16.9 Cost

`tests/test_airbox_membrane.py` was 12.9 s of CI suite time and is gone outright. The two partial
retirements take `test_airbox_surface.py` from 6.41 s and `test_airbox_dipole.py` from 7.3 s to
0.65 s each measured locally; against three unchanged airbox files measured the same way
(`scene` 6.78 s local / 13.64 CI, `energy` 0.61 / 1.35, `modal` 0.84 / 2.46, a CI:local ratio of
2.0–2.9) that is **1.4 s** each in `scripts/shard_costs.json`. Total suite saving ~23 s. Unlike
§14.7's partial retirement, this one *does* scale with the test count, because what left is the
whole file except one test rather than the cheap half.

The 45 native bars run in **0.86 s** release, 15.3 s debug. The headline pair — the phantom's `t50`
at two air-grid refinements, four runs of up to 4000 steps in a 25x23x19 room — is inside that.

### 16.10 The next batch

The von Kármán seam and its two wrappers: `_VKPlateSurface`, `RoomLoadedVKPlate`,
`RoomSuspendedVKPlate`, retiring `tests/test_airbox_vk.py` (26 functions, 65.2 s — five times any
other file in this family) and whatever part of `tests/test_mallet_room_gong.py` has no unported
caller. Four things are known in advance:

1. **The seam is a different shape.** The linear seams' `commit` takes one array and their solve is
   a back-substitution; the VK seam's `solve` *iterates* (Picard or Newton) and its `commit` takes
   `(w, F)`. `GridSeam` as it stands cannot express that, so the trait grows an arm or the VK
   wrapper takes a different path through `RoomGrid::step` — decide it before writing bars.
2. **The bit-identity anchor is the batch's own gate.** `RoomLoadedVKPlate(nonlinear=false)` must
   reproduce `RoomLoadedPlate` under `array_equal` on both stored levels, on the coupling ledger and
   on the Airy roll. Both sides will be native, so it becomes a native bar — and §16.2's
   single-transcription design is what makes it cheap.
3. **`LoadedLu` is the seam the binding built for exactly this.** `physsynth_core::plate::ThetaSolve`
   exists because the von Kármán kernel had to invert a matrix that lived in SciPy. Natively the
   loaded factorization is a `SparseLu` and implements the trait directly, so that adapter has no
   native counterpart — a third "no analogue", of the §16.6 second kind.
4. **`test_mallet_room_gong.py` drives a client, not the wrapper.** `MalletVKPlate` in a room is
   `VkRoom` in `crates/physsynth-py/src/mallet.rs`, which composes `prepare` / `loaded_rhs` /
   `finish`. `physsynth_core::mallet` already has room-aware machinery
   (`VkCoupledStep::with_rhs`), so how much of that file retires is a question about the *mallet*
   tier and is worth answering before the batch rather than during it.

---

## 17. Phase C batch 5, done — the von Kármán seam, and the wrapper tier is finished

Commit 2 of §16's split (the human's: linear first, gong second). The nonlinear plate's room
wrappers — the reference's `_VKPlateSurface`, `RoomLoadedVKPlate` and `RoomSuspendedVKPlate` —
re-home into `crates/physsynth-core/src/airbox_wrap.rs` as a third seam, `VkSeam`, of the same
generic `RoomGrid<S>`. Seventeen native bars in `crates/physsynth-core/tests/airbox_vk.rs`;
seventeen of `tests/test_airbox_vk.py`'s twenty-six functions retire. With this the **wrapper tier
is finished**: every class the reference's `airbox.py` had is native.

### 17.1 §16.10's first question: the seam is a different shape, and one provided method absorbs it

The linear seams solve once and commit one array; the von Kármán seam *iterates* and commits two
histories plus six read-outs. §16.10 said to decide before writing bars whether the trait grows an
arm or the wrapper takes a different path through `RoomGrid::step`. It grew **one provided
method**, `GridSeam::advance(&mut self, lu, rhs) -> u_next`. Its default is the linear seams' "one
back-substitution, then `commit`", so `PlateSeam` and `MembraneSeam` did not change and their 45
bars are untouched. `VkSeam` overrides it with the model's own coupled step, `plate::vk_step_with`,
against `VkCoupledStep::with_rhs(rhs, …, &lu_loaded)`. `RoomGrid::step` borrows the seam mutably and
the factorization immutably; they are disjoint fields.

Two structural decisions ride on it:

* **One commit path.** `VkPlate::step`'s six assignments moved into `VkPlate::record(VkStep)`, and
  the seam commits through it. The binding's seam once kept its own list and missed two
  (`n_solves`, `residual_ratio` — the air-box Newton plan's §2.4); natively "the room-driven plate
  writes every read-out the bare step writes" is true by construction, and a bar still says so.
* **`theta_matrix` and `plate_areas` are shared** between `PlateSeam` and `VkSeam` rather than
  written twice, so the `nonlinear = false` anchor cannot fail on the matrix for a spelling reason.
  The **force path is deliberately not shared**: the linear seam adds `f_ext` inside
  `plate::step_rhs`, the gong's through `add_f_ext` on `VkPlate::linear_rhs`, so their agreement is
  a measurement the anchor makes with `sigma > 0` and a live force, not a consequence of the code.

### 17.2 One deliberate departure: the stress cache on the linear path

With `nonlinear = false` the model's step returns no stress function. The binding's seam handed the
model's own `F` back and rolled it anyway (`F_prev <- F`); `VkSeam` follows the **bare plate** and
leaves both levels alone. The two agree whenever the cache holds what the model put there (zeros on
that path) and differ only for a caller who wrote `F` by hand — and there, "a room-loaded plate
reduces to the bare one" requires the bare plate's behaviour. Documented on `VkSeam` itself.

### 17.3 The one-time check: 8 scenes, 120 steps, every state quantity and every iteration count exact

The concern going in: batch 4 compared one solve per step, but this seam back-substitutes **many**
times per step and the sweep count branches on a norm, so a last-bit difference between SciPy's
`splu` (the binding's `LoadedLu`) and the native `SparseLu` could flip an iteration count and send
the two trajectories apart. Measured, with the inputs recorded from the binding as raw doubles so
both sides start from identical numbers (no transcendental evaluated on either side):

| scenes | compared every step | differing |
|---|---|---|
| supported / free × baffled / suspended, `sigma = 2`, lossy mounting wall, live `f_ext`, `auto` | `w`, `F`, the room's whole pressure field, volume velocity, `last_residual`, `(n_iters, n_solves, n_fallbacks, converged)`, `room.injected` | **0** |
| supported baffled, lossless, rigid room | same | **0** |
| free suspended under **Newton** (3–4 iterations, GMRES inside) | same | **0** |
| both `nonlinear = false` arms | same | **0** |
| all eight | `radiated_energy` and the wrapper's `energy()` | 10–115 of 120 steps, ≤ 2.0e-16 relative |

Identical in the debug and the release profile. The only difference is the coupling ledger's BLAS
`ddot`, which §16.3 already showed does not feed back — and the room's *own* booking of the same
identity, `injected`, is exact. The iteration counts span 6–19 sweeps per step, so the concern was
exercised hard and did not materialise: the native factorization reproduces SuperLU's solves to the
bit on these matrices, which batch 4's exact state arrays implied for a single solve and which this
shows survives a loop that branches on the result.

### 17.4 Three deliberate breakages, three different sets of bars

| breakage | bars that fail |
|---|---|
| the seam iterates against the plate's **own** operator (`VkCoupledStep::new` for `with_rhs`) — the plausible slip | 6: the `nonlinear = false` anchor, the coupled residual, flat and monotone totals, the `couple_tol` split, Newton past the wall |
| `rho_v` where `rho_s` belongs in the denominator | 7: the same six plus the factorization-inputs bar |
| commit by hand instead of through `VkPlate::record`, dropping three read-outs | 4: the read-outs bar, the capped-step verdict, `couple_method`, the zero-area reduction |

Note what passes under the first: **`radiated == injected`**. A gong that is not in the room at all
still books its ledger consistently. That is the family's standing rule — no single detector is
sufficient — for the fifth batch running.

### 17.5 §16.10's fourth question: the mallet, and a piece of the hole no class count could see

`tests/test_mallet_room_gong.py` does not drive a wrapper. It drives `MalletVKPlate`, and when that
class is handed a room wrapper instead of a bare plate it switches to a different step
(`step_in_room` in `crates/physsynth-py/src/mallet.rs`) that composes the wrapper's
`prepare`/`loaded_rhs`/`finish` around its own outer iteration. That composition exists **only in
the binding** — `physsynth_core::mallet` has the pieces (`vk_plate_step_with`, `retarget_column`)
but nothing that assembles them around a room — so it is part of the hole. It is a *mode* of an
existing class rather than a class, so neither §11's hand count nor §13.1's derive could list it.

To make sure it is the only one, every binding site that recognises a room wrapper was grepped
(`VkRoom::of`, casts to the `PyRoom*` classes, `in_room`, `airbox_wrap::` outside that file). One
site: the mallet. (The bridges accept room wrappers too, but they are already on the list.) So the
hole after this batch is **the three bridges and the mallet's room mode**. The mallet file was not
retired here: the plan this batch ran on did not list it. Asked afterwards, **the human chose to
port it next, as its own batch (2026-09-23)**, ahead of the bridges. When it is ported, `RoomGrid` already has everything it composes, and the binding's
pointer-identity check for a swapped factorization has a clean native analogue — a generation
counter that `RoomGrid::refactor` bumps.

### 17.6 The retirement rule, discharged

`tests/test_airbox_vk.py` 26 -> 9. The seventeen that went are every test whose referent is the
wrapper; the nine that stay drive `StringVKPlateBridge` (§14.3's rule). The file's docstring is
rewritten to say so, and its helpers and imports shrink to what the chain uses.

| retired | native bar |
|---|---|
| `test_nonlinear_false_is_the_linear_room_loaded_plate_bit_identical` (8 cases) | same name, all 8 cases, plus the room's pressure field |
| `test_the_loaded_factorization_matches_the_linear_one` | same name |
| `test_the_loaded_factorization_is_never_assigned_to_the_model` | same name — **behavioural**, see below |
| `test_the_nonlinear_path_runs_inside_the_load` | same name |
| `test_zero_area_reduces_to_the_bare_vk_plate` | same name, plus `n_solves` |
| `test_ledgers_agree_and_the_channel_is_not_vacuous` (12) | same name |
| `test_the_piston_is_the_free_plate_s_fat_channel` | same name |
| `test_the_scene_total_is_flat` | same name |
| `test_the_lossy_scene_total_is_monotone` | same name |
| `test_energy_is_an_override_and_not_a_delegation` | same name |
| `test_the_coupled_residual_at_two_timesteps` (8, three controls each) | same name |
| `test_couple_tol_moves_the_total_and_not_the_money_test` (`slow`) | same name — **not** ignored natively |
| `test_the_room_seam_writes_every_read_out_the_bare_step_writes` | same name, plus `n_fallbacks` |
| `test_a_short_capped_room_step_reports_capped_and_not_expansive` | same name |
| `test_the_loaded_step_honours_couple_method` | same name |
| `test_newton_carries_a_strike_that_kills_the_loaded_picard_loop` | same name |
| `test_refuses_a_sample_rate_mismatch` | same name, plus "a refused wrapper claims nothing" |

Two carried differently, and neither is a weakening:

* **"Never assigned to the model"** asserted `plate._lu is not inst._lu_loaded`, an identity.
  Natively the loaded factorization is an *argument* to `advance` and the plate's own is a field of
  its parameters, so identity is structural. What is left to assert is the behaviour the identity
  protected: after a run in the room, the plate's own operator is still the one a fresh plate
  builds, and it differs from the loaded one.
* **`_refactor`**, the reference's hand re-derivation of `A_loaded` for the half-load control,
  becomes `RoomGrid::refactor` after rescaling the port's load — the wrapper's one assembly
  spelling, as §16.8 did for the linear tier.

The `slow` bar runs in ordinary CI natively: the whole file is 0.7 s in release and 9 s in debug.

### 17.7 Cost

`tests/test_airbox_vk.py` was the most expensive file in the airbox family (65.2 s in CI). The nine
chain tests are the larger half of it — measured in one process against the old file, the remainder
costs 0.52 of the whole — so `scripts/shard_costs.json` carries **33.9 s**, an estimate for the next
CI durations run to replace.

### 17.8 The next batch

**The mallet's room mode, on its own** (the human's call, 2026-09-23): it is small, stands on
nothing unported, and retires most of `tests/test_mallet_room_gong.py`. After it, the hole is only
the three bridges in `crates/physsynth-py/src/connection.rs` (`StringBodyBridge`,
`StringPlateBridge`, `StringVKPlateBridge`) — what every remaining chain test in
`test_airbox_vk.py`, `test_airbox_surface.py` and `test_airbox_dipole.py` waits on.

## 18. Phase C batch 6, done — the mallet's room mode, and the hole is the three bridges

The batch §17.8 named, on its own (the human's call). The gong-in-a-room step that lived only in
the binding (`step_in_room` in `crates/physsynth-py/src/mallet.rs`) re-homes as its own native type,
`MalletVkRoom` in `crates/physsynth-core/src/mallet.rs`, holding a `RoomGrid<VkSeam>`. Ten native
bars in `crates/physsynth-core/tests/mallet_room_gong.rs`; `tests/test_mallet_room_gong.py` retires
**whole** — §17.8 said "most", and every one of its thirteen functions (21 collected tests) turned
out to have either a native bar or a recorded verdict (§18.5). Its helpers, `make_mallet_room_gong`
and the `MALLET_ROOM_*` constants, had no other caller and went with it.

### 18.1 A type, not a mode

The binding recognised a room wrapper at construction and branched inside one class. Natively the
two are different types with different step signatures — the bare gong owns its plate and steps with
no argument, the room gong holds a wrapper that never owns its room and so steps with
`step(&mut AirBox)`, as `RoomGrid::step` does. One class with a branch would have had to carry an
`Option<&mut AirBox>` whose `None` is an error in one mode and required in the other. The step is the
binding's five phases unchanged: refresh the column if stale, `prepare` once, the chord (whose trial
solver is `loaded_rhs` + `VkCoupledStep::with_rhs` against `lu_loaded`, the same pair
`VkSeam::advance` uses), one commit, `finish` once from the field read back off the plate.

Two pieces of the binding had no clean native shape and each got one:

* **The swapped-factorization check** compared the Python factorization object's pointer each
  step, because a caller could assign `_lu_loaded`. Natively `RoomGrid::refactor` is the only way the
  factorization changes, so it now bumps `RoomGrid::generation()` and the mallet compares stamps —
  §17.5's prediction, taken as written.
* **The error channel** needed `ParkedErr` in the binding because two failures inside the trial
  solver were Python exceptions. Natively `loaded_rhs` cannot fail, so the error is a plain two-arm
  enum, `RoomGongError { Room(WrapError), Gong(VkContactError) }`.

The bare mallet was **not** made generic over "a plate or a room". Its twelve bars
(`mallet_gong.rs`) did not move, and the one piece the two must share — the commit — is a function
both call (§18.3).

### 18.2 The one-time check: six scenes, every state quantity and every count exact

The binding's side was dumped from Python as raw doubles and replayed natively in a scratch crate
under `W:\temp\claude\mallet_room`, with the wheel reinstalled first. Compared every step: the
plate's `w` and `F`, the room's whole pressure field, the per-node volume velocity, the outer
residual, contact force, `z_H`, penetration, `room.injected`, `last_residual`, `g_s`, `mal.energy()`,
and eleven counts and flags (`n_outer`, `n_solves`, `inner_iters`, `n_fallbacks`, the three
convergence flags, `in_contact`, and the plate's `n_iters`, `n_fallbacks`, `converged`). The
constructor's whole influence column was compared too.

| scene | steps | differing |
|---|---|---|
| baffled, supported, lossless, `auto` | 200 | **0** |
| suspended, same | 200 | **0** |
| baffled, `nonlinear = false` | 200 | **0** |
| suspended, a miss (`v0 = -1`, `gap = 0.01`, 2e initial deflection) | 120 | **0** |
| suspended, **free** edge, `sigma = 2`, **Newton** | 150 | **0** |
| baffled, factorization swapped to the plate's own at step 60 | 200 | **0** — `g_s` refreshed on the same step, to the bit |
| all six | `radiated_energy` only | 20–129 steps, ≤ 3.2e-16 of the ledger's size |

Identical in the debug and release profiles. The one difference is §16.3's coupling-ledger `ddot`
again, which does not feed back; the room's own booking of the same identity was exact. (So was
`mal.energy()`, which contains the ledger — but that is rounding absorbing a last-bit difference in
a term far smaller than the plate's energy, not independent evidence.) Measured relative to the value
instead of to its size, the suspended scene reads 2.5e-14 — because the ledger passes near zero
there, which is the findings ledger's "what is the bar divided by" (#30), not a larger error.

### 18.3 A second hand-written commit, found in the native bare mallet

`MalletVkPlate::step` in the core crate rolled the plate's histories and wrote **three** of its six
read-outs by hand (`n_iters`, `converged`, `n_solves`), leaving `last_residual`, `residual_ratio` and
`n_fallbacks` at whatever the previous bare step had left. The binding's commit wrote all six — the
outer residual as `last_residual`, `NaN` as `residual_ratio`, the summed `n_fallbacks`. It is
§17.1's drift exactly, one type over, and invisible to every bar the bare gong had because none of
them reads those three after a mallet step. Both mallets now convert their `VkContactStep` with one
function (`plate_step_of`) and commit through `VkPlate::record`, and
`both_mallets_write_every_read_out_through_the_one_commit_path` seeds all six with garbage first so
an untouched field cannot pass by coincidence.

### 18.4 Five deliberate breakages — and the retired file's tangent bar could not see its own defect

| breakage | bars that fail |
|---|---|
| no retarget at construction (the chord frozen on the bare column) | 4: the loaded column, `n_outer == 1`, the rebuild, the tangent |
| no generation check (a rebuilt factorization never noticed) | 1: the rebuild |
| `finish` handed the field from **before** the commit | 3: the committed field, the miss, the scene total |
| the tangent's operator left on the plate's own factorization | 1: the tangent — **after a rewrite; 0 as carried** |
| the bare mallet's old three-field hand commit put back | 1: the one-commit-path bar (reads the seeded `-7.0`) |

The last row is the batch's finding (#78). The Python bar for the exact tangent compared the room's
tangent with a bare gong's and asserted they **differ**. They do — but the retargeted *column* alone
makes them differ, so the bar passed with the *operator* unrouted, which is the precise defect the
mallet-room batch recorded as its own scar (a loaded right-hand side inverted against the bare
Jacobian). Carried over faithfully, the native bar was green under that mutation too. It is now a
**positive** oracle: `response = -d w_node / d f` at the root, so a central difference of the room's
own trial solve from the pre-step state has to match it. At the step of largest force, with
`delta = 1e-4 f`, routed agrees to 9.5e-11 (baffled) and 2.6e-12 (suspended); unrouted misses by
9.1e-6 in both. The bar is 1e-8, about a hundred times clear of each. The by-difference assertion
is kept beside it.

A fifth breakage was not run, because it cannot be expressed: driving the wrapper once per trial now
**refuses** at the second `inject` (`require_ready`), so the miscount the Python file was designed
around is a runtime error natively. The injection-count bar is kept as a statement of the design,
and its second half asserts the refusal and that a refused step mutates nothing.

### 18.5 The retirement rule, discharged

| retired (`tests/test_mallet_room_gong.py`) | native bar, or verdict |
|---|---|
| `test_the_wrapper_is_what_mal_plate_returns` (2) | **type** — `MalletVkRoom` holds a `RoomGrid<VkSeam>`; there is no second object to hand back (§14.2) |
| `test_a_bare_gong_is_unchanged_by_the_widening` | **type** — two types, no widened cast |
| `test_a_linear_plate_still_gets_the_message_that_names_MalletPlate` | **type** here (a `RoomGrid<PlateSeam>` does not compile); the message itself is still asserted on the bare class by `tests/test_mallet_gong.py` |
| `test_the_chord_freezes_the_LOADED_column_not_the_plates_own` (2) | `the_chord_freezes_the_loaded_column_not_the_plates_own` |
| `test_nonlinear_false_in_a_room_exits_the_chord_at_one_iteration` (2) | same name, with the rate arithmetic |
| `test_a_swapped_factorization_refreshes_the_frozen_column` | `a_rebuilt_factorization_refreshes_the_frozen_column` — through `refactor` and the stamp, plus "no rebuild, no refresh" and "stale until the next step" |
| `test_the_exact_tangent_is_taken_against_the_LOADED_operator` | same name — **sharpened** to a finite-difference oracle, both mounts (§18.4) |
| `test_a_mallet_that_never_lands_leaves_the_room_scene_bit_identical` (2) | same name, plus `F` |
| `test_the_port_is_injected_once_per_step_however_long_the_chord_runs` (2) | same name — counted in `room.pending_ports`, whose per-injection size is read off the bare wrapper (a suspended port queues a `-q`/`+q` pair); plus the refusal |
| `test_the_room_is_driven_by_the_COMMITTED_field_not_by_a_trial` (2) | same name |
| `test_the_rooms_load_terms_do_not_move_across_the_chord` (2) | **no analogue** — it counted calls by replacing the port's methods (§16.6's first kind); `step` calls `prepare` once and hands the chord the half by reference |
| `test_the_scene_total_is_conserved_through_the_strike` (2) | same name |
| `test_the_delegated_plate_energy_is_the_wrong_number_and_the_wrapper_knows_it` | `the_plates_own_energy_is_the_wrong_number_and_the_wrapper_knows_it` |
| — | **new**: `both_mallets_write_every_read_out_through_the_one_commit_path` (§18.3) |

The whole native file runs in 3.9 s in debug. The Python file was never in `scripts/shard_costs.json`,
so nothing there moves.

### 18.6 What stays in the binding

`PyMalletVKPlate`'s room arm — `VkRoom::of` in the constructor, `step_in_room`, `loaded_influence`,
`ParkedErr` and the routed tangent — is left as it is. After this batch nothing in Python drives it
and nothing tests it; it goes when `crates/physsynth-py` does. Deleting it now would change what the
Python class accepts, which is not this batch's call, and would buy nothing the crate's deletion
does not.

### 18.7 The next batch

**The three bridges** in `crates/physsynth-py/src/connection.rs` — `StringBodyBridge`,
`StringPlateBridge`, `StringVKPlateBridge` — and with them the chain tests still in
`tests/test_airbox_vk.py` (9), `tests/test_airbox_surface.py` and `tests/test_airbox_dipole.py`.
That is the whole hole: after the bridges, nothing is implemented only in the binding.

## 19. Phase C batch 7, first half — the body bridge

§18.7 named all three bridges as the next batch. They split in two along the only exact anchor
that ties any of them together: `StringVKPlateBridge`'s guard must equal `StringPlateBridge`'s to
the last digit when the plate is linear, so those two are **one unit**; nothing binds
`StringBodyBridge` to either (the anchor that bound it to `SympatheticStrings` was restated away in
§12). So this half is `StringBodyBridge` alone, and the plate pair is §20.

`StringBodyBridge<B>` is in `crates/physsynth-core/src/connection.rs`, with 22 native bars in
`crates/physsynth-core/tests/connection_body.rs`. `tests/test_connection.py` retires **whole**
(13 functions), with the four chain tests that put the bridge in front of a loaded body: three in
`tests/test_radiation.py` and one in `tests/test_airbox_freefield.py`. `make_bridge` and its two
constants had no other caller and went with them, and so did `make_sympathetic`, which had had no
caller since §12 retired its file and was left behind then.

### 19.1 Four bodies, one slot, and the room as an argument

The binding's `body=` slot took a `ModalBody`, a `RadiatedBody`, a `ReactiveRadiatedBody` and a
`RoomLoadedBody`, and read `phi`, `m`, `omega`, `q_prev` and three methods off whatever it was
given. That is now the trait `BridgeBody`: `modal()` for the bare body underneath, `step`,
`energy` and `pressure`. Two choices inside it are worth stating.

* **The room is an associated type, not a member.** A mounted body steps as
  `RoomLoadedBody::step(&mut AirBox, F)`, because a port never owns the room it drives (§13.2). So
  `BridgeBody::Room` is `()` for the three free-air bodies and `AirBox` for the mounted one, and
  `StringBodyBridge::step(&mut B::Room)` passes it straight through. The bridge never steps the room
  itself; the caller does, afterwards, exactly as every Python chain's loop read
  `bridge.step(); room.step()`. The cost is `br.step(&mut ())` at a free-air call site.
* **The guard reads only `modal()`.** The radiation load — resistive, reactive or a room — is
  dissipative or a separate storage channel and was never in the leapfrog operator the binding
  assembled. Folding it in would change which configurations are refused.

Two things were deliberately **not** carried across from the sympathetic set, which is otherwise
the same spring on one string. The set refuses `lambda >= 1`; this bridge never did, and at
`lambda = 1` the binding accepts `K = 100` (`k^2 lambda_max = 3.99986`) and refuses `K = 1000`
through the *spectral* guard. And the body's row of the coupled operator keeps the binding's two
statements (`omega^2 q`, then `- K phi eta / m`) rather than the set's summed force, so the guard
stays comparable to the binding's while both exist.

### 19.2 The one-time check

Six scenes dumped from the binding (wheel reinstalled first) and replayed natively, under
`W:\temp\claude\bridges`:

| scene | steps | differing |
|---|---|---|
| bare body, the suite's default fixture, body given an initial state | 500 | **0** |
| lossy string and body, `lambda = 0.7`, `K = 15000` | 500 | **0** |
| non-unit mode weights and masses | 500 | **0** in state and energy; 45 read-outs of the previous-step stretch, ≤ 4.6e-14 relative |
| `RadiatedBody`, `R = 1500` | 500 | **0** |
| `ReactiveRadiatedBody`, `R = 1500`, `M_a = 0.05` | 500 | **0** |
| `RoomLoadedBody` in a matched-wall room, room stepped after the bridge | 300 | **0**, including the room's whole pressure field and `injected` |
| guard `lambda_max`, all six | — | ≤ 4.8e-15 relative |

Compared every step: the string's `u`, the body's `q`, the total energy, the pressure read-out,
the bridge force, the previous-step stretch, the loaded body's radiated energy. Identical in the
debug and release profiles.

The one difference is the previous-step stretch, and only where the mode weights are not all 1.
The binding forms `phi . q_prev` with `np.dot`, which is BLAS `ddot` and fuses its multiply-add
(§14.2); natively it is a plain index-order sum. With every weight 1.0 the products are exact and
the two agree; with `phi = [1, 0.6, -0.4, 0.8]` they do not. It is a read-out only — the energy's
cross-time factor — and the energy that contains it still matched to the bit, which is rounding
absorbing a last-bit difference, not independent evidence (§18.2's note, again). The stretch *now*,
which is the force and does reach the trajectory, goes through the body's own
`bridge_displacement()` on both sides and was exact.

### 19.3 Six deliberate breakages

| breakage | bars that fail |
|---|---|
| `energy()` reads the bare modal body instead of the loaded body's override | 3: the resistive, reactive and room chains |
| no end-node correction on the string | 10, including the shape bar |
| the body handed `-F` | 10, including the shape bar and the pressure bar |
| the guard's spring term without the half-cell factor 2 | 2: the frozen `lambda_max` and the self-adjointness bar |
| the sympathetic set's `lambda < 1` refusal copied in | 1: the Courant-limit bar |
| the energy's cross-time factor read at the same time level | 8, every conservation and passivity bar |

The first row is the one a single-body test file could never have produced: it is invisible on a
bare `ModalBody`, whose override *is* the delegated number, and only a chain with a sink sees it.

### 19.4 The retirement rule, discharged

| retired | native bar, or verdict |
|---|---|
| `test_total_energy_conserved_across_lambda` (3) | `the_total_energy_is_conserved_across_lambda` |
| `test_total_energy_conserved_across_stiffness` (4) | `the_total_energy_is_conserved_across_stiffness` |
| `test_string_energy_alone_is_not_conserved` | `the_string_energy_alone_is_not_conserved` |
| `test_energy_flows_string_to_body` | `energy_flows_from_the_string_into_the_body` |
| `test_passivity_with_body_damping` | `a_lossy_body_makes_the_total_decrease_monotonically` |
| `test_passivity_with_string_damping` | `a_lossy_string_makes_the_total_decrease_monotonically` |
| `test_K0_bit_identical_to_uncoupled_parts` | `at_zero_stiffness_the_bridge_is_bit_identical_to_the_uncoupled_parts` |
| `test_pressure_includes_coupling_term` | `the_pressure_read_out_carries_the_coupling_term` |
| `test_unstable_stiffness_rejected` | `an_over_stiff_spring_is_rejected` |
| `test_guard_holds_at_its_boundary` | `the_guard_holds_just_inside_its_boundary` |
| `test_mismatched_timestep_rejected` | `a_mismatched_timestep_is_rejected` |
| `test_right_end_must_be_free` | `a_clamped_right_end_is_rejected` |
| `test_string_bridge_body_room_chain_conserves` | `the_string_bridge_body_room_chain_conserves` |
| `test_radiation.py::test_full_chain_radiates_with_retardation` | `the_full_chain_radiates_with_retardation` |
| `test_radiation.py::test_full_chain_with_radiation_conserves_the_total` | `the_full_chain_with_radiation_conserves_the_total` |
| `test_radiation.py::test_full_chain_with_reactive_radiation_conserves_the_total` | `the_full_chain_with_reactive_radiation_conserves_the_total` |
| `test_airbox_freefield.py::test_string_bridge_body_air_load_drives_the_room` | `the_string_bridge_body_air_load_drives_the_room` |
| — | **new**: `one_step_is_the_free_parts_plus_exactly_one_spring_force` — §12's restated shape bar, on the bridge it was restated away from |
| — | **new**: `the_guard_reproduces_the_binding_measured_spectral_radius` — `1.6617411698938603e9` to 1e-12, which makes the mass derivation load-bearing (§12's argument) |
| — | **new**: `the_coupled_operator_is_self_adjoint_in_the_energy_inner_product` |
| — | **new**: `a_string_at_the_courant_limit_is_refused_only_by_the_spectral_guard` |
| — | **new**: `a_negative_stiffness_is_rejected` |

The other 49 functions in `test_radiation.py` and five in `test_airbox_freefield.py` stay: their
referents are the radiation tiers and the room, not the bridge (§14.3's referent rule).
`tests/test_connection.py` was 39.67 s in `scripts/shard_costs.json` and its line is gone; the 22
native bars run in 2.1 s in debug.

### 19.5 What stays in the binding

`PyStringBodyBridge` is untouched. The viewer builds it — on a bare body and on three loaded
ones — and `tests/test_binding_surface.py`'s rows for it are claims about the binding that go when
the crate does. Same reasoning as §12 and §18.6.

### 19.6 The next batch

**The two plate bridges, as one unit** (§20): `StringPlateBridge` and `StringVKPlateBridge`, with
`tests/test_plate_connection.py`, `tests/test_free_plate_connection.py`,
`tests/test_vk_connection.py` and the chain tests in `tests/test_airbox_vk.py` (9),
`tests/test_airbox_surface.py` (1) and `tests/test_airbox_dipole.py` (1). After it nothing is
implemented only in the binding.

## 20. Phase C batch 7, second half — the plate bridges, and the hole is closed

`StringPlateBridge` and `StringVKPlateBridge` re-home as **one** generic type,
`StringPlateBridge<P: BridgePlate>` in `crates/physsynth-core/src/connection.rs`, with 48 native
bars in `crates/physsynth-core/tests/connection_plate.rs`. Six Python files retire **whole**:
`tests/test_plate_connection.py` (18 functions), `tests/test_free_plate_connection.py` (18),
`tests/test_vk_connection.py` (16), and the three that had been kept only for their chain tests,
`tests/test_airbox_vk.py` (9), `tests/test_airbox_surface.py` (1) and `tests/test_airbox_dipole.py`
(1) — 63 functions, 268 s of `scripts/shard_costs.json`. Thirty-four helpers left
`tests/helpers.py` with them; `plate_bump`, `plate_mode_shape` and `vk_strike`, which sat in the
same section, stay because other files use them.

**With this, nothing is implemented only in the binding** — derived at the class level and one level
down (§20.5), because §17.5 showed a class count alone cannot see a mode living inside a method.
Every model class the binding holds has a native implementation. What remains in `crates/physsynth-py` is glue, plus the binding's own copies
of the four bridges, which the viewer still builds (§19.5's reasoning, unchanged).

### 20.1 Two classes, one type, and where the density lives

The binding's two classes differed in exactly one attribute name, the plate's areal density —
`rho` on a `Plate`, `rho_s` on a `VKPlate` — and an exact anchor required their guards to agree to
the bit, so they already had to be one piece of arithmetic. Natively the difference is a trait
method, `BridgePlate::linear()`, returning the plate's linear parameters. For a `VkPlate` that is
`p.lin`, the plate it reduces to with the coupling off, whose `rho` is already `rho_v e`. So there
is one margin function for all four plate collaborators — `Plate`, `VkPlate`,
`RoomGrid<PlateSeam>`, `RoomGrid<VkSeam>` — and the factor-of-1000 trap the Python policed with a
cross-class anchor lives in one place, `VkParams`'s construction of `lin`.

The room is an associated type, as in §19.1. Three other shape decisions:

* **The pressure read-out is an impl, not a method on the trait.** It exists for
  `StringPlateBridge<Plate>` and `StringPlateBridge<RoomGrid<PlateSeam>>` only. The binding's von
  Kármán bridge had no `pressure` on purpose (the compact monopole reads 3e-7 of the truth for a
  gong), and a Python test asserted the attribute was absent; natively that is a type fact
  (§16.6's second kind).
* **Two setters, both found by grepping every write to a bridge attribute across the suite.**
  `set_beta_s` (the wrong-reaction test injects a fault) and `set_stiffness_unguarded` (the
  true-onset tests raise `K` past the ceiling after construction). The first grep matched only a
  variable named `bridge` and missed `over.K = 1.05 * Kc`; the second matched any name.
* **The boundary refusal has no analogue.** The binding refused a plate whose `boundary` was neither
  `"supported"` nor `"free"`; the native boundary is a two-variant enum.

The guard itself is the binding's expression order: the string block is tridiagonal and solved
directly (forward elimination, one back-substitution for the last unknown), and the plate block is
assembled as `rho h^2 (I + c B)` or `rho (W + c K)` and factored with the crate's `SparseLu`.

### 20.2 The one-time check

Ten scenes dumped from the binding (wheel reinstalled) and replayed natively under
`W:\temp\claude\bridges`:

| scene | steps | differing |
|---|---|---|
| supported plate, `K = 3000` | 400 | **0**; margin exact |
| free plate, `K = 6000` | 400 | **0**; margin exact |
| supported, lossy string and plate, `lambda = 0.7` | 400 | **0**; margin 4.3e-16 |
| von Kármán, supported and free | 300 each | **0**, including every step's sweep count and convergence flag; margins exact |
| von Kármán, `nonlinear = false`, free, `sigma = 2` | 300 | **0**; margin exact |
| linear plate in a room, baffled supported and suspended free | 300 each | **0** in state, room field and `injected`; margins exact |
| the gong chain, suspended free and baffled supported | 300 each | **0** in state, sweep counts, room field and `injected`; margins exact |
| the four room scenes | — | `radiated_energy` only: ≤ 1.1e-15 relative, plus one last-bit total energy that contains it |

The default drive node matched in every scene (79 supported and 107 free on the suite's plate,
15 and 29 on the room's), as did the string's reaction weight. The only difference is §16.3's
coupling-ledger `ddot`, which does not feed back.

One thing looked wrong and was not. The supported and free von Kármán margins agree to the last
digit. That could have meant the plate block was below the string block's rounding, and then the
Python bar "the margin uses the areal density" would have been blind to the density. Measured through
the binding: with the volume density the margin is 0.1394 against 0.1425, so the bar is live. The two
boundaries agree because the drive node is interior, where both lumped masses are `rho_s h^2`, and
the theta-excess stiffness is negligible at this timestep. The bar is now sharpened to assert the
volume-density twin **differs** — finding #78's pattern, an equality that could not rule out the
defect replaced by a positive claim — so the check that was made here stays made.

### 20.3 Nine deliberate breakages, and one that no bar could see

| breakage | bars that fail |
|---|---|
| a room-mounted **linear** plate's energy read from the bare plate, not the wrapper's override | 1: the linear chain |
| the same for the room-mounted **gong** | 4: the gong chain, the lossy chain, and two detector bars |
| the plate handed `-F` | 18 |
| no `lambda < 1` refusal | 1 |
| the free plate block given an extra `h^2` | 26 (the margin crosses 1, so most constructions refuse) |
| the string block solved without its off-diagonal | 5: both frozen-margin bars and three ceiling bars |
| the energy's cross-time factor read at the current level | 14 |
| the gong's linear parameters carrying the volume density | 3, including the sharpened density bar |
| **the force entry not zeroed after the step** | **0** |

The last row is finding #79. The binding zeroed the drive entry after every step because there a
caller could replace `_f_ext` or move `drive_index`. Natively neither can change after construction,
so the only nonzero entry is always the one about to be overwritten, and the reset is unobservable.
The shape bar's comment claimed a leak would be caught; that was false. The reset and the claim are
both gone. The mutation was planned before the port as a check on the reset; running it is what
showed there was nothing for the reset to do.

The two energy-override rows were run separately (#73, #78's "mutate each half"). A first attempt
at them did not compile — the trait method was not in scope — and a row reading "0 fail" from a
mutant that never built would have been a false green. The sweep script now reports a mutant that
did not run as such.

### 20.4 The retirement rule, discharged

Where a bar loops over both boundaries it carries one test from each linear file.

| retired | native bar, or verdict |
|---|---|
| `test_total_energy_conserved_across_lambda` (plate + free, 6) | `the_total_energy_is_conserved_across_lambda` |
| `test_total_energy_conserved_across_stiffness` (plate + free, 8) | `the_total_energy_is_conserved_across_stiffness` |
| `test_string_energy_alone_is_not_conserved` (plate + free) | `the_string_energy_alone_is_not_conserved` |
| `test_energy_flows_string_to_plate` (plate + free) | `energy_flows_from_the_string_into_the_plate` |
| `test_passivity_with_plate_damping` (plate + free) | `a_lossy_plate_makes_the_total_decrease_monotonically` |
| `test_passivity_with_string_damping` (plate + free) | `a_lossy_string_makes_the_total_decrease_monotonically` |
| `test_K0_bit_identical_to_uncoupled_parts` (plate + free) | `at_zero_stiffness_the_bridge_is_bit_identical_to_the_uncoupled_parts` |
| `test_pressure_includes_coupling_term` (plate + free) | `the_pressure_read_out_carries_the_coupling_term` |
| `test_no_rigid_body_drift` (free) | `a_free_plate_does_not_drift_rigidly` |
| `test_unstable_stiffness_rejected` (plate + free) | `an_over_stiff_spring_is_rejected` |
| `test_margin_is_linear_in_stiffness` (plate + free) | `the_margin_is_linear_in_the_stiffness` |
| `test_guard_holds_at_its_boundary` (plate + free) | `the_guard_holds_just_inside_its_boundary` |
| `test_just_over_the_ceiling_is_rejected` (plate + free) | `just_over_the_ceiling_is_rejected` |
| `test_ceiling_is_the_true_instability_onset` (plate + free) | `the_ceiling_is_the_true_instability_onset`, through `set_stiffness_unguarded` |
| `test_string_at_lambda_one_rejected` (plate + free) | `a_string_at_the_courant_limit_is_rejected` |
| `test_mismatched_timestep_rejected` (plate + free) | `a_mismatched_timestep_is_rejected` |
| `test_right_end_must_be_free` (plate + free) | `a_clamped_right_end_is_rejected` |
| `test_drive_index_out_of_range_rejected` (plate + free) | `a_drive_index_off_the_plate_is_rejected` |
| `test_free_plate_is_accepted` | **type** — the boundary is an enum, and every bar that loops over both boundaries builds a free bridge |
| `test_nonlinear_false_is_the_linear_bridge_bit_identical` (4) | `nonlinear_false_is_the_linear_bridge_bit_identical` — still a real comparison: a linear `VkPlate` against an independently built `Plate`, two different step paths |
| `test_the_margin_ignores_the_nonlinearity_and_uses_the_areal_density` (2) | same name, **sharpened**: the volume-density twin must differ (§20.2) |
| `test_total_energy_conserved_lossless` (2) | `the_von_karman_total_is_conserved_lossless` |
| `test_nonlinearity_is_genuinely_engaged` (2) | `the_string_drives_the_plate_past_its_thickness` |
| `test_passivity_with_loss` (4) | `a_lossy_von_karman_chain_decreases_monotonically` |
| `test_zero_stiffness_decouples_bit_identically` (2) | `at_zero_stiffness_the_von_karman_bridge_decouples_bit_identically` |
| `test_vk_connection.py::test_string_energy_alone_is_not_conserved` | `the_string_energy_alone_is_not_conserved_on_the_gong` |
| `test_a_linear_body_scales_bit_exactly_with_the_pluck` (2) | same name |
| `test_departure_from_a_linear_body_is_second_order_in_the_pluck` (2) | `the_departure_from_a_linear_body_is_second_order_in_the_pluck` |
| `test_linear_energy_share_is_amplitude_invariant_and_the_gong_s_is_not` (2) | `the_linear_energy_share_is_amplitude_invariant_and_the_gongs_is_not` |
| `test_rigid_modes_are_immune_to_the_nonlinearity` | same name — a claim about the von Kármán plate itself that had no other native bar, so it lands here |
| `test_guard_rejects_an_overstiff_spring` (2) | `the_von_karman_guard_rejects_an_over_stiff_spring` |
| `test_the_linear_margin_survives_a_strongly_nonlinear_run` | same name |
| `test_the_failure_mode_migrates_to_non_convergence` | same name |
| `test_construction_rejects_mismatched_and_malformed_inputs` | `the_von_karman_bridge_refuses_malformed_inputs` |
| `test_bridge_exposes_no_pressure_readout` | **type** — `pressure()` is implemented only for the two linear plates |
| `test_airbox_vk.py::test_the_chain_composes_and_the_guard_is_bit_identical` (4) | `the_gong_chain_composes_and_the_guard_ignores_the_room` |
| `test_the_money_test_holds_with_the_string_as_the_only_excitation` (4) | same name |
| `test_the_room_adds_no_outer_iteration` (4) | same name |
| `test_zero_bridge_stiffness_decouples_the_chain` (4) | same name |
| `test_the_nonlinear_false_chain_is_the_linear_bridge_bit_identical` (4) | `the_nonlinear_false_chain_is_the_linear_chain_bit_identical` |
| `test_the_lossy_chain_is_monotone` (4) | same name |
| `test_band_overlap_decides_the_rigid_share_not_the_pluck` (2) | same name |
| `test_a_wrong_string_reaction_is_seen_by_the_total_and_not_the_money_test` (2) | same name, through `set_beta_s` |
| `test_every_detector_is_blind_to_a_drive_index_that_differs_between_two_runs` (2) | same name |
| `test_airbox_surface.py::test_string_bridge_plate_room_chain` (2), `test_airbox_dipole.py::test_string_bridge_plate_room_chain` (2) | `the_linear_plate_chain_conserves_and_the_guard_ignores_the_room` |
| — | **new**: `one_step_is_the_free_parts_plus_exactly_one_spring_force` |
| — | **new**: `the_guard_reproduces_the_binding_measured_margins` and `the_chain_margins_reproduce_the_binding` — six margins frozen to 1e-12 |
| — | **new**: `a_negative_stiffness_is_rejected`, `the_default_drive_node_is_the_corner_offset_point` |

**The two cross-class equalities**, which are now one code path, each got a verdict rather than being
carried as tautologies. The von Kármán margin equalling the linear one is kept, because the twin is an
independently built `Plate` and the equality asserts that `lin` carries the areal density. It is now
paired with the volume-density twin that must differ. The room-loaded margin equalling the bare one
is kept as the statement that the guard reads the plate's parameters and never the loaded
factorization.

The 48 bars run in 198 s in debug locally (7 s in release); the six Python files were 268 s of
shard cost. **On CI that trade is not neutral, and it is recorded here rather than tuned away.** The
Rust job's native step went from 6 min 56 s on §19's run to **18 min 46 s** on this one
(runner-class variance on this project is ~1.6x, and the run before §19 took 12 min 37 s, so the
jump is not the runner). The Python bars were spread over three concurrent shards; the native ones
all land in the one Rust job, in the debug profile `cargo test --workspace` uses, where the two
long conservation bars (1.2–1.5 s of audio each, at up to 40 kHz, over both boundaries and several
lambdas or stiffnesses) dominate. The Rust job was already the slowest in the run; it is now the
whole run's critical path, at ~20 minutes against ~3 per Python shard. Whether to answer that — the
native step in release, a split of the Rust job, or fewer audio-seconds in the two long bars — is
the human's call, and none of it is done here.

### 20.5 The closing claim, derived one level down

§13.1's derive is a name join and has expired as a measure (§16.7), so "the hole is closed" was
checked three ways rather than one:

1. **Classes.** Every `#[pyclass]` name in `crates/physsynth-py` against every public native type:
   43 classes, 10 without a same-named native type, and all 10 are recorded re-homings — the six
   room wrappers and three surface adapters (`RoomGrid<S>` and its seams, §16.7) and
   `StringVKPlateBridge` (`StringPlateBridge<VkPlate>`, here).
2. **Methods that compute.** Every `#[pymethods]` method and `#[pyfunction]` whose body has a loop
   or three or more arithmetic operations — 34. Each was read: the bridges, wrappers and loaded
   bodies are ported (§13–§20); the rest are constructors and argument parsing around a native
   type that does the work (`AirBox::add_cut`, `GeometricString::step` with both of its branches,
   `TensionModulatedString::tension`, `parabolic_refine` with its edge case, and so on).
3. **Methods that orchestrate.** The second pass exists because the first is blind to exactly the
   thing this batch ported: the bridges' `step` did one multiplication, and its content was the
   *sequence* of calls on duck-typed collaborators. Every call on a Python object outside the
   ported files is NumPy marshalling, the room reading its own attributes, or the bore's `source`
   hook, which the native `Bore::step` takes too.

The only type-dispatched mode in the binding is `MalletVKPlate`'s room arm, which is §18's.

### 20.6 What is next

Phase C's hole work is done. What stands between here and deleting `crates/physsynth-py` is the
rest of the plan as §7 laid it out: the viewer (phase D, still the longest pole), the scripts
(phase E), and the remaining Python physics files whose bars have not been carried yet (§9's map,
read with §16.7's table).

---

## 21. Phase A, done — the flag is gone

`PHYSSYNTH_RS` is read by nothing. The last three modules that chose between two implementations at
import time are resolved, and the choice they offered went with them:

| module | before | after |
|---|---|---|
| `physsynth/core/exciter.py` | 116 lines: a NumPy body, `_py` aliases, a swap block | 34: three re-exports, since the binding's signatures are the body's, keywords and defaults included |
| `physsynth/core/operators.py` | 253 lines: a NumPy/SciPy body, aliases, a swap block | 106: the swap block's two seams **as the whole body** — `_csr` rebuilds a `csr_matrix` from the binding's triplets, `_asarray` re-widens the input to what NumPy accepted. It delegates rather than re-exports, the same shape as `operators2d` |
| `physsynth/core/banded.py` | 106 lines, LAPACK behind a swap | **deleted whole.** No model has called it since the string family's bodies went (unit 1); its only importer was its own parity file |
| `crates/physsynth-py/src/banded.rs` | 113 lines of **Rust** | **deleted**, with its registration and the `NotPositiveDefinite` exception type. The native solver in `physsynth-core` is untouched and is what the four strings factor with |

Also gone: `tests/test_rust_parity_banded.py` and `tests/test_rust_parity_operators.py` (116
cases), the `rust-harness` CI job, and `scripts/shard_tests.py --exclude-parity` with its two guard
tests. That last one existed only so the flagged run could skip the two-sided files, and the flagged
run is gone.

### 21.1 The acceptance run: the default suite IS the old flagged suite

Measured on a freshly installed wheel, before any edit: the whole suite minus the three parity files,
with `PHYSSYNTH_RS=1`, **1,920 passed**. After the deletion, the same suite with no flag at all (and
minus the one remaining parity file) **1,940 passed**. The 20 is reconciled case by case against a
collection of the pre-deletion tree, not by arithmetic on totals:

- +28 in `tests/test_binding_surface.py`, the harvest (§21.3);
- −6 in `tests/test_shard_partition.py`, the `--exclude-parity` guard (5 shard counts) and its canary;
- −2 in `tests/test_xdist_groups.py`, whose module-fixture scan is parametrized over test files and
  lost exactly the two deleted parity files.

Nothing else moved, so the property the plan wanted is met: the default run now exercises what the
flagged run did, and a module that grew a second implementation again would fail
`test_no_module_chooses_between_two_implementations` (§21.4).

### 21.2 LAPACK retires as the oracle, with its numbers

`banded.py`'s Python side was never a transcription. It was `dpbtrf`/`dpbtrs` through SciPy, the
library call every θ-scheme string acceptance number was first measured against. Deleting it retires
that yardstick, and §4 required the retirement be stated with the numbers rather than rediscovered:

- **lossless energy drift, transcribed solver 2.7e-12 against LAPACK's 2.7e-12** on the same string,
  against a bar of 1e-10 (measured 2026-08-27);
- **the four models' agreement with each other stays exact** — `sigma1 = 0`, `EA = 0` and `EA = T`
  are still `array_equal`, because all four call the same native solver.

Two measurements lived only in the deleted file's comments and are kept here so they are not lost:

- **The two solvers separate like a square root, not a saturation.** On N = 128, kappa = 2.7,
  sigma = 3, worst state difference as a fraction of the run's amplitude: 100 steps 1.1e-13, 500
  2.9e-13, 1000 4.1e-13, 2000 9.7e-13, 5000 2.0e-12, 20000 3.2e-12. The Group A target of ~1e-13 was
  a *hundred-step* claim for a fed-back solve. Normalise by the run's amplitude, never pointwise: a
  damped string decays by orders of magnitude.
- **The factor matched OpenBLAS on 120/120 of the family's matrices with a fused multiply-add in the
  rank-1 update and 82/120 without.** The native solver deliberately does not fuse, since fusing is a
  property of the kernel OpenBLAS picks at run time. The worst relative difference stayed under
  1e-13.

### 21.3 The retirement rule, discharged

**`tests/test_rust_parity_banded.py`** (46 cases):

| retired | native bar, or verdict |
|---|---|
| `test_factor_agrees_with_lapack` (18) | LAPACK as oracle retired (§21.2); `the_factor_reconstructs_the_matrix` asserts what the factor is *for* |
| `test_solve_agrees_with_lapack` (18), `test_the_solve_actually_inverts_the_matrix` | `the_solve_inverts_the_matrix` |
| `test_how_often_the_factor_is_exact_is_measured_not_asserted` | a measurement against LAPACK, recorded in §21.2; no subject left |
| `test_a_non_spd_band_is_refused_with_lapacks_own_message` | `a_non_positive_definite_band_names_the_minor_that_failed` and `a_zero_diagonal_is_refused_and_so_is_a_nan`. The message *text* was LAPACK's and retires with it |
| `test_a_shape_that_is_not_a_band_is_refused` (3), `test_a_right_hand_side_of_the_wrong_shape_is_refused` | `a_shape_that_is_not_a_band_is_refused`, whose third assertion is the right-hand side's length. The 1-D refusal was the binding's and went with it |
| `test_the_shim_keeps_the_exception_type_scipy_raises`, `test_lower_storage_is_refused_rather_than_silently_transposed` | **no referent**: both were about the Python shim, which is deleted, and nothing in Python calls a banded solve |
| `test_the_family_still_reduces_to_itself_exactly` | `sigma1_zero_is_the_stiff_string_exactly`, `ea_zero_is_model_three_bit_for_bit`, `ea_equals_t_is_bit_identical_to_the_damped_string`; each anchor is also still asserted in Python in `test_damped_string.py`, `test_tension_string.py` and `test_geometric_energy.py` |

**`tests/test_rust_parity_operators.py`** (70 cases). Twelve of its fourteen tests compared an
operator with its Python transcription. The ones with a referent that is not being deleted were
**harvested** into `tests/test_binding_surface.py`, and two are sharper for being re-aimed at it:

| retired | native bar, or where it went |
|---|---|
| `test_pointwise_differences_are_bit_identical` (20) | `forward_difference_is_exact_on_a_linear_ramp`, `second_difference_is_exact_on_a_quadratic`, `second_difference_has_the_exact_discrete_eigenvector`, `fourth_difference_is_exact_on_a_quartic` |
| `test_the_fourth_difference_divisor_survives_a_sweep_of_h` (7) | twin only: it measured whether two libms agree on `h**4` (they disagree with `h*h*h*h` on 1,400 of N = 2..3999). `fourth_difference_is_exact_on_a_quartic` |
| `test_the_two_first_differences_are_the_same_function_on_both_sides` | the Rust half kept: `test_the_two_first_differences_are_the_same_function` |
| `test_a_too_short_field_yields_an_empty_array_on_both_sides` | **re-aimed at NumPy's slicing**: `test_a_too_short_field_yields_numpys_empty_slice_rather_than_a_panic` |
| `test_the_inner_product_agrees_to_the_group_a_target` (4) | **re-aimed at `h * np.dot`**, the thing the transcription wrapped: `test_the_inner_product_agrees_with_numpys_dot_to_the_group_a_target`; natively `the_inner_product_is_the_energy_bookkeeping_it_claims_to_be` |
| `test_inner_is_exactly_norm2_when_the_operands_coincide` | kept, same name |
| `test_second_difference_matrix_is_bit_identical` (8) | `the_second_difference_matrix_is_the_pointwise_operator_with_dirichlet_ghosts`, `the_second_difference_matrix_has_the_exact_discrete_eigenpair` |
| `test_biharmonic_matrix_is_bit_identical` (8) + the `B` half of `test_the_matrices_agree_on_a_non_unit_grid_spacing` | **re-aimed at SciPy**: `test_the_biharmonic_is_scipys_own_product_of_the_second_difference` (16 — both grids, eight sizes), exact in values. Natively `the_biharmonic_is_the_second_difference_squared_and_conserves_the_energy_identity` and two more |
| `test_free_beam_stiffness_is_bit_identical` (8) + the `D2`, `K`, `W` half of the non-unit test | `the_free_beam_mass_is_trapezoidal`, `the_free_beam_stiffness_annihilates_exactly_the_rigid_body_space`, `the_mass_normalised_free_beam_is_bilbaos_energy_conserving_bar`, `the_free_beam_bending_energy_is_the_curvature_norm` |
| `test_both_sides_reject_a_grid_too_coarse_to_have_an_interior` (3) | the Rust half kept: `test_a_grid_too_coarse_to_have_an_interior_is_refused` |
| `test_the_binding_hands_back_triplets_not_a_matrix` | kept, same name |
| — | **new**: `test_the_operator_shim_takes_what_numpy_would`, the reason `operators` delegates rather than re-exports (an int array, a strided view, a Fortran-ordered one) |

**`tests/test_rust_parity_ops2d.py` is not part of phase A**, and §2's count of 275 parity tests
misleads on this point. Its Python side went with unit 5 (rust-migration-plan §43) and it was
harvested then. What it asserts now is against SciPy and against itself: the Gram-association
witness, the guitar outline's margin, the Airy solve against SuperLU. Its 159 cases stay until phase
F or until each claim has a native home. It keeps its old prefix, which no longer means anything.

### 21.4 The guards: two tables deleted, one widened, per ledger #52 and #67

`tests/test_stability.py`'s `test_the_rust_swap_matches_the_environment` became
`test_no_module_chooses_between_two_implementations`:

- **the `_USE_RUST` reader tuple** (operators, exciter, banded) had nothing left to read, and it is
  **deleted**;
- **`ported_expected`**, the `<name>_py` function table, had no aliases left to derive over, and it
  is **deleted**;
- **the `if expected_rust:` captured-binding block** is **deleted**, because every assertion in it
  had become `x is x`, for example `string_stiff.biharmonic_matrix is operators.biharmonic_matrix`
  and `reed.Bore is bore.Bore`;
- **the flag and alias checks are WIDENED** from hand-written tuples to every module `pkgutil`
  finds in `physsynth/core/`: no module has `_USE_RUST`, and none defines a name ending in `Py` or
  `_py`. Named positive controls keep an empty scan from passing;
- `deleted_bodies` gains `exciter`'s three names, and `operators` joins `operators2d` under the
  *inverted* claim: each function is a wrapper defined in its module, never the Rust function
  itself.

In CI, the `rust` job's comparison step is renamed for what is left in it. It is
`test_binding_surface.py` plus the SciPy-facing 2-D file, run with `-rP` for their reports. The
`validate` job's own comment had said the flag's end was the moment "these two jobs are the same run
and one of them should be removed", and `rust-harness` is the one removed.

### 21.5 What is next

Phase A is the first phase of this plan to finish outright. What stands between here and deleting
`crates/physsynth-py` is unchanged in kind: the viewer (phase D, still the longest pole), the scripts
(phase E), and the Python physics files whose bars have not yet been carried natively (§9's map, read
with §16.7's table). `test_rust_parity_ops2d.py` joins that last list.

---

## 22. Phase E, done except what cannot go yet — the scripts

Thirty-six of `scripts/`'s forty-one files are gone, with `physsynth/viz/` and the `viz` extra:
**9,590 lines**, and not one test with them. The collected suite is the same size before and after
(2,099 test ids, measured on a fresh wheel), because no test imported a deleted file.

| what | lines | verdict |
|---|---|---|
| `diagnose_*.py`, all 33 | 8,414 | **deleted as spent** (§22.1) |
| `physsynth/viz/` (`plots.py` + `__init__`) | 610 | **deleted** — its only importers were the 33 |
| `freeze_analysis.py`, `freeze_horizon.py` | 332 | **deleted**; provenance by commit (§22.3) |
| `sweep_geometric_lam_long.py` | 234 | **replaced** by a native example (§22.2) |
| `shard_tests.py`, `shard_costs.json`, `shard_costs_from_durations.py`, `nicepytest.py` | — | **stay until F** — §6 was wrong about the timing (§22.4) |
| `verify_web_headless.py` | — | **stays until D**, which it ports with (§6.1) |

`matplotlib` has left `pyproject.toml` entirely (the `viz` extra and the `dev` pin), so CI no longer
installs it.

### 22.1 The triage bar, and why it is lower than §6 made it sound

§6 framed triage as a coverage question. It is not one: **CI never ran any of these scripts**, so
deleting one removes no enforced check. The question per script is narrower — *does it compute a
number that is recorded nowhere else, in no test and no doc?* Every `print` literal of the 33 was
extracted by AST and sorted into three bins:

- **validation reports** (24 scripts — ideal, stiff and damped strings, membrane, the plates, bore,
  bell, reed, bow, the collisions, the bridges, sympathetic, tension, both von Kármán plates, the
  three radiation scripts): the printed numbers *are* the suite's bars — drift against 1e-10,
  partials against the oracle, convergence order;
- **investigation records** (9 — the six air-box scripts, the string→gong→room chain, the geometric
  string, both orthotropic plates): each finding is written into the plan that cites the script.
  Twenty numbers and phrases were grepped for in `docs/` and every one was found. That is a spot
  check, not a proof: a few of the tokens (`5.3`, `8.9`, `cn`) are common enough to match unrelated
  text, and the evidence is carried by the specific ones — `57.9 kHz`, `2.35 M`, `0.9998`, `5a/6`,
  `0.5625`, `781`, `46.0%`, `anticlastic`, `0.567`;
- **recorded nowhere**: none.

`diagnose_mallet_plate.py` was the one script no doc names; its findings are in
`docs/dev/mallet-plate-plan.md`, which describes them without the filename.

**The picture-output scripts §6 flagged** (the Chladni figures, the air-box field slices, the
Schelleng diagram, the whirl orbits) got the verdict §6 asked for: each was a *look at this*, and
the viewer shows all of them interactively. The Schelleng diagram's one number, its clean-Helmholtz cell
count, is *recorded* (23 of 48, `docs/memory/bow-state.md`) but is **not** a bar:
`tests/test_bow_modal.py` only picks points known to lie inside the window. That was true before this
deletion too, and the web viewer plan already says why no bar is attempted — the window has no
closed form in the core.

**The scripts' own verdicts were never gates**, and one run shows it: `diagnose_membrane.py` prints
`energy monotone non-increasing = False` on today's tree, over a single rise of 6.0e-19 J at step 0
of 17,066 (1.6e-15 relative). `tests/test_membrane_energy.py` asserts the same property with a
`1e-12 * E0` round-off allowance and passes. A script that printed a verdict with no tolerance was
a reader's aid, and deleting it loses nothing the suite asserts.

### 22.2 The sweep was already dead, and its successor is half of it

`scripts/sweep_geometric_lam_long.py` backs `docs/dev/scientific-hurdles.md` §6, which is still open,
so §6 of this plan named it a survivor candidate. It turned out to be **unimportable on the tree it
was being triaged on**: its A/B control was `class OldJacobianString(GeometricString)`, and since the
geometric string's body was deleted the base is a pyo3 class that is not subclassable —
`TypeError: type 'physsynth_rs.GeometricString' is not an acceptable base type`. A Python override of
`_dg_jacobian` would have done nothing anyway, because the Rust `step` never calls back into Python.
`tests/test_geometric_energy.py` meanwhile told a reader whose bar failed to re-run it.

Its successor is `crates/physsynth-core/examples/geometric_lam_long.rs` — the project's first Cargo
example, no dependencies, built by `cargo test --workspace` and linted by `clippy --all-targets`, so
it cannot rot silently the way the script did. It carries the half of the script that measures
**the model as it is**: the two-edge table (default) and the full drift/iterations/stalls sweep
(`--full`). The A/B half is spent — its answer, 9 cells of 9 unmoved, is §6's record — and the
amplitude table existed only to explain that answer.

One deliberate difference: a non-finite drift now counts as past the energy gate. The script's
`drift > gate` is false for NaN, so a run that went to NaN would have read as conserving.

**The port was checked against the original, and the check found something.** The script cannot
run on today's tree, so it was run at its own commit (`305661f`) in a `git worktree`, where the
string was still Python. The energy edge agreed in **9 cells of 9**. The convergence edge differed
in **4 of 9**, in both directions — by one grid point in three cells and by two in the fourth, the
pinned test's. Driving the same sweep through today's binding
reproduced the example to the cell, so the port is faithful and the difference is the Python-to-Rust
move: a stall is a Newton residual failing a `1e-15` tolerance, which sits at round-off, so which
step fails first is a last-bit event. `scientific-hurdles.md` §6 now says to quote the edge as
"about 4" and never per cell. Two consequences, both recorded where they live:

- the "7 of 9" in three places counted cells **at or below** 4 on Python; on Rust it is 6. The hurdle
  doc, the model's docstring table and the pinned test's docstring now say so;
- the pinned test (`test_a_flat_energy_is_not_a_convergence_certificate_in_the_under_resolved_band`)
  asserts stalls at `λ_long = 6`, and its cell is the one that moved from 4 to 6. It sees 2 stalls in
  ~95 steps at 6 and none at 5 — **half a grid step of margin, where it had two**. It passes; it was
  not changed in this batch, because the margin was lost at the deletion of the Python body, not here.
  A future move of the solver's last bits could flip it, and its message now names the example to
  re-run.

### 22.3 The freezers: deleted, provenance by commit

Both freezers' docstrings argued for keeping them unrunnable, on the principle that a generated file
must name its generator. The plan (§6) said delete, and the human's call was to delete and keep the
principle by **commit**: `tests/analysis_frozen_values.py`'s header now says nothing regenerates it,
that `17efb1e` is the last commit containing both generators, and that a new oracle gets a native
bar rather than a row. `test_analysis_frozen.py`'s "regenerate with `freeze_analysis.py`" message is
replaced by the only remedy left.

### 22.4 What §6 had wrong: the sharding scripts go at F, not E

§6's table filed `shard_tests.py`, `shard_costs*` and `nicepytest.py` under **delete**, and that is
the end state, but not this phase's: CI calls `shard_tests.py` in the `setup` and `validate` jobs,
`tests/test_shard_partition.py` imports it, and the README's local run uses `nicepytest.py`. They
exist to split and run the pytest suite, so they leave in the commit that deletes the suite (§7). The
same is true of `verify_web_headless.py` relative to D.

### 22.5 What is next

Unchanged in kind from §21.5: D (the viewer, the longest pole, and now also the owner of the one
script left that is not about running pytest), and the physics files whose bars are not yet native
(§9.4's list; the air box is still the largest shortfall).

---

## 23. Phase D, batch 1 — the crate, the server, and the three linear strings

The viewer's backend has a Rust half. `crates/physsynth-viewer` is a library (`simulate_to_payload`,
the whole contract the front-end speaks) plus a binary: `physsynth-viewer serve` is `python
web/server.py`'s replacement, and `physsynth-viewer payload < params.json` runs one request with no
socket. It serves `web/static/` unchanged. Of the twenty-two model keys the front-end offers, three are
ported — `ideal`, `stiff`, `damped` — and the rest are refused with a new error kind, `unported`, so a
scene can never render from a half-built payload. **The Python server is still the viewer anyone
should use** until §23.7's list is finished; nothing about it changed.

What landed:

| where | what |
|---|---|
| `crates/physsynth-core/src/engine.rs` | `physsynth/core/engine.py`'s `simulate` and its `Resonator` protocol, native, with `tests/engine.rs` |
| `crates/physsynth-viewer/src/lib.rs` | the dispatch, the three-arm refusal, the horizon fallback |
| `…/py.rs` | Python's coercions, rounding and reprs (§23.3), NumPy's `linspace` index, base64 |
| `…/resample.rs` | `scipy.signal.resample_poly` and `Fraction.limit_denominator`, transcribed |
| `…/energy.rs`, `…/horizon.rs` | the energy panel and the resolution read-out, whole — every verdict branch, not only the strings' |
| `…/string.rs` | `_build_resonator` and `_build_payload_string` |
| `…/server.rs`, `…/main.rs` | the HTTP shell and the CLI |
| `crates/physsynth-viewer/tests/` | `deps`, `strings`, `horizon`, `server` — 36 tests, plus 5 unit tests in `resample.rs` |

`cargo test --workspace --release`: **848 passed**, the 801 before this batch plus 6 engine and 41
viewer tests.

### 23.1 No HTTP crate, measured rather than assumed

§5 named `tiny_http` as the obvious candidate and said to measure it at the batch. Measured on
2026-09-28, `tiny_http 0.12` pulls `ascii`, `chunked_transfer`, `httpdate` and **`log`**, and `log` is
on the core's NEVER list by category. The server is `std::net` plus one thread per connection, which is
all the Python `ThreadingHTTPServer` was: HTTP/1.0, one request per connection, localhost. The
viewer's only outside dependency is `serde_json`, with what it pulls (`serde_core`, `itoa`, `zmij`,
`memchr`), and `tests/deps.rs` holds that list — the third copy of the rule, and the first non-empty
one. Its NEVER list names the binding (`physsynth-py`, `pyo3`, `numpy`): the viewer is what makes the
binding deletable, so it must never come to depend on it.

**The allowlist test found a hole in its own walker on its first run.** `serde_json` declares `serde`
under `[target.'cfg(any())'.dependencies]` purely to constrain which `serde` version may coexist with
it. That cfg holds on no platform, so nothing is compiled, and `cargo tree` shows nothing, but `cargo
metadata`'s resolve lists the edge with the cfg as data. The walk reported `serde`, `serde_derive`,
`syn`, `quote`, `proc-macro2` and `unicode-ident` as shipped. The viewer's copy now skips exactly the
literal `cfg(any())` and nothing else, so a real platform cfg is still walked and over-reports on the
safe side. The two physics crates' copies have the same blind spot but cannot hit it: their normal
dependency lists are empty.

`serde_json` takes the `float_roundtrip` feature. Without it the parser is not correctly rounded, and
a request's `L` or `T` one ulp off would move the last bits of every step of the run it
parameterizes.

### 23.2 "Never a NaN" had to be made true again

The Python server dumped with `json.dumps(..., allow_nan=False)`, so a non-finite float anywhere in a
payload was a loud failure. `serde_json` does the opposite, and silently: `Value::from(f64::NAN)` is
`null`. A test asserting "no NaN in the payload" would then pass while checking nothing. So every float
enters a payload through `py::num`, which turns a non-finite value into a marker string, and
`simulate_to_payload` walks the finished tree for it. If it finds one, the whole payload is refused
with kind `internal`. The one field that *should* carry `null` for a non-finite value (`_finite_list`)
does that on purpose and is the only path that can. A test pins both halves, including the
`serde_json` behaviour that makes the marker necessary.

### 23.3 Python's semantics, where the payload depended on them

The reference read its request with `float()`, `int()` and `str()`, and those accept more than a
JSON number: a numeric string, a bool (`float(True) == 1.0`), a float for an integer (`int(64.7) ==
64`). Its refusal messages quote the offending value in Python's `repr`. All of that is reproduced in
`py.rs`, along with four spellings that decide numbers rather than messages:

- **`round(x)` is ties-to-even**, and the index arithmetic inherits it (`pickup_idx`, step counts,
  strides). `f64::round` rounds ties away from zero; `round_ties_even` is used instead.
- **`round(x, n)`** goes through Rust's fixed-precision formatting and back. Rust's formatting is
  correctly rounded on the exact binary value with ties to even, which is CPython's
  `float.__round__`. That was checked against Python on halfway values (`0.5`, `2.5`, `0.125`,
  `0.375`), on `2.675`, on a negative zero and on `1e300` before any code relied on it.
- **Python's `min(a, b, c)`** returns the first element unless a later one is strictly smaller, so
  a NaN in first position passes a `<= 0` guard. Transcribed rather than "fixed", because the guard
  order is part of which refusal a request gets.
- **NumPy's `linspace(...).astype(int)`** overwrites its last entry with the endpoint, so the
  energy trace's decimation index is reproduced exactly.

Six differences are **deliberate**, and they are the only ones:

1. `int(float("inf"))` raised an uncaught `OverflowError` in Python, and the request died with a
   500. Here it is refused the way a NaN is.
2. A dict quoted in a refusal message prints its keys sorted, because `serde_json`'s map is ordered by
   key. Only a malformed request reaches this.
3. A request body using the `NaN` / `Infinity` JSON literals, which Python's `json.loads` accepted,
   is a 400. The browser cannot send them (`JSON.stringify` writes `null`).
4. `HEAD` is served. The Python handler had no `do_HEAD`, so it answered 501, despite a `_send` that
   was written to support it.
5. A panic inside a render is a 500 with a JSON body. The Python server dropped the connection.
6. Path traversal is refused by construction: only plain path components are accepted. The Python
   server compared strings after `normpath`.

### 23.4 The one-time check: 60 requests against the live Python serializer

The same discipline as every port in this migration: run both implementations on one corpus while
the reference still exists, then compare parsed JSON trees rather than text. Key sets must match
exactly, and so must int-versus-float type, strings and bools. Floats are exact unless a named
tolerance class says otherwise, with the reason written next to it. The corpus has 60 requests:

- defaults and the canonical short run;
- all three models, lossless and lossy;
- the horizon fixtures from `test_web_backend.py`;
- an odd-rate grid and `N = 2`;
- 14 refusals;
- 17 coercion cases (strings, bools, nulls, lists and dicts where numbers belong).

The Python side ran on a freshly reinstalled wheel. The harness is `W:\temp\claude\viewer-port\`
(`dump.py`, `compare.py`, `corpus_strings.py`) and is scratch, not repo: once the reference is deleted
there is nothing left for it to compare against.

**Result: 56 identical to the bit, 4 within a stated tolerance, 0 failing.**

- **The decay-rate fit** (3 lossy cases): at most **1.9e-14 relative**. `np.polyfit` solves the
  least-squares line through LAPACK's SVD; this uses the closed form about the mean. The class is set
  at 1e-12.
- **The `N = 2` audio**: 52 of 14,520 float32 samples differ, by at most **1.9e-33 against a peak of
  0.9**. Those are samples that should cancel to exactly zero and carry rounding noise in both
  implementations. A per-sample ulp count put the difference at 19 million ulps, because that is
  the wrong scale for a value near zero. The class is 2 float32 ulps of the buffer's *peak*.

**The audio was expected to be the weak point and was not.** `resample.rs`'s header argues that the
filter taps cannot be bit-identical: they call `sin`, which NumPy computes with its own CPU-dispatched
routine (findings §22.1), and a Kaiser window, whose Bessel `I0` is Cephes' Chebyshev expansion in
SciPy and a power series here. That argument is correct about what *can* differ. On this machine,
nothing did: 59 of the 60 audio buffers are identical to the bit. The tolerance class stays in place,
because the header's reason is still true on another CPU.

**In the browser.** The existing headless check (`scripts/verify_web_headless.py`), pointed at the
Rust server with `VIEWER_BASE`, plus a scratch copy that adds `stiff`, `damped` and a lossy `damped`,
passed all four string scenes on its own gates. The rendered readouts checked were:

- the energy verdict;
- the partials panel;
- the horizon strip, and the horizon mark cross-checked against it;
- the deep link applying every parameter;
- the canvas actually painted.

The three unported scenes in its list (`tension`, the parametric regime, `bow`) show the `unported`
message and fail, as they must.

### 23.5 The retirement rule, discharged for this batch

`test_web_backend.py` is the specification (§5), so its string tests are carried as native tests
rather than dropped:

- the eleven string tests in its opening section are all in `tests/strings.rs`, with the
  parametrized ones as loops;
- `tests/horizon.rs` carries every horizon test that a string scene can answer.

The horizon tests are split where a test mixes a string fixture with a 2-D one, and the 2-D half is
still owed:

- `test_horizon_is_built_from_the_SCHEME_...` has a plate half;
- `test_horizon_a_tighter_bound_...` has membrane and plate halves;
- `test_horizon_the_hertz_ceiling_...` has a membrane half;
- `test_horizon_survives_the_servers_strict_json` has 2-D halves.

Each 2-D half lands with its scene's batch.

Some tests are new, because the port created something to test:

- **the refusal messages, word for word.** The reference asserted substrings, and a port is exactly
  where a message drifts.
- **the coercions.** A string-spelled request must be the same scene as the numeric one.
- **params for other models are ignored.**
- **the non-finite marker** (§23.2).
- **the dispatch table against the markup.** This extends
  `test_horizon_every_model_the_viewer_OFFERS_is_classified`, so a model offered in the `<select>`
  can never fall through to the string builder's "unknown model".

**No Python test is deleted in this batch.** They test the Python serializer, which is still the
live viewer. They go in the batch that deletes it, and so does
`test_horizon_the_canvas_MARK_vocabulary_...`, which reads the headless harness that goes then too.

### 23.6 Cost

The Rust payloads are fast enough that no budget moved: the full string test file runs in 0.1 s in
release and 1.7 s unoptimized. The default ideal-string render took 0.05 s in the browser run. The
viewer's tests add about 13 s to CI's debug pass.

### 23.7 The order of the remaining nineteen

This order is derived rather than chosen. The derivation takes the transitive closure of each
payload builder over `serialize.py`'s helpers, then records which SciPy or NumPy tool the closure
needs that is not native yet. Three tools are missing:

- **`np.fft.rfft` at an arbitrary length.** The analysis crate's FFT pads to a power of two, and
  `np.fft.rfft(sig)` does not.
- **`scipy.sparse.linalg.eigsh`** in shift-invert mode, including the generalized form with a mass
  matrix and eigenvectors (the guitar's modes).
- **`scipy.linalg.eigh`** with eigenvectors. `physsynth-core::eig` returns eigenvalues only.

Each is a new numerical routine, so each gets native bars against a closed form before a payload
uses it. That is the krylov precedent.

| batch | scenes | new numerics |
|---|---|---|
| D2 | `tension` (both regimes), `bow` | none |
| D3 | `geometric`, `sympathetic`, `reed`, `radbody`, `airload` | none new (the rotating-wave oracle is already in `physsynth-analysis`; `uniform_filter1d` and a median are a few lines each) |
| D4 | `body`, `jawari`, `juari`, `fret` | arbitrary-length `rfft` (`np.hanning` is already `spectrum::hann`) |
| D5 | `membrane`, `mallet`, `plate` (all three outlines), `vk`, `bore`, `platebody` | sparse shift-invert eigensolver |
| D6 | `airbox`, `vkroom` | dense symmetric eigenvectors |
| D7 | **freeze first**, then: the headless check ported to Rust (a WebSocket client as a **dev**-dependency, allowlisted with its reason), the servers switched, `web/*.py` and `test_web_backend.py` deleted | — |

**The exact-agreement evidence has to outlive the reference, and nothing yet makes it.** The scratch
diff is the only thing that saw "56 of 60 identical". The native tests are *property* tests, so a
changed rounding, decimation index or band field could pass all of them. The analysis port set the
precedent (`tests/analysis_frozen_values.py`, the human's condition for deleting that Python):
freeze the reference's numbers before the reference goes. So D7 does not delete `serialize.py` until
a native fixture holds each batch's corpus and the Python's payloads, compared with `compare.py`'s
tolerance classes. The audio is size-bounded by short durations, or frozen as a digest plus
spot samples. One caution transfers from findings §22.1: Rust's own `sin`, `ln` and `log2` come
from the platform libm, so the Linux CI runner may differ from this machine in the last bits. Any
frozen field that passes through them needs a peak- or relative-class, never exact.

**Each batch's corpus also takes the browser's own requests**, captured during the headless run.
`gatherParams` sends every slider, hidden ones included, and a scene reading another model's
parameter is a failure this viewer has shipped before (the leak family, batches 2/3/7/8/12). The
test-derived corpus does not send those extra parameters.

Four constants in `serialize.py` are lowered by `monkeypatch` in its tests so that a guard which
cannot fire in the shipped range fires somewhere real: `PARAM_SWEEP_WORK_MAX`,
`RADBODY_SWEEP_WORK_MAX`, `AIRLOAD_SWEEP_WORK_MAX` and `TENSION_MEASURE_PERIODS`. A Rust constant
cannot be patched. The batch that ports each one decides whether it becomes a parameter of an
internal function or whether the test is rewritten against a fixture that reaches the guard. That
is §16.6's "no analogue" verdict (a test that works by replacing part of a live module), arriving
at the viewer.

### 23.8 Batch D2, done — the tension string (both regimes) and the bow

Five of the twenty-two keys are now native: `ideal`, `stiff`, `damped`, **`tension`** (the Duffing
regime and the parametric one) and **`bow`**. `crates/physsynth-viewer/src/tension.rs` and
`…/bow.rs` carry them, and `string.rs`'s `_build_resonator` now builds all five kinds it builds in
the reference.

**The check, on two corpora.**

| corpus | requests | identical | within a class | failing |
|---|---|---|---|---|
| test-derived (`corpus_d2.py`) | 60 | 41 | 19 | 0 |
| the browser's own requests, logged | 7 | 3 | 4 | 0 |
| batch 1's strings, re-run after this batch | 60 | 56 | 4 | 0 |

The browser corpus is the advisor's point from §23.7 put into practice. The server gained
`PHYSSYNTH_VIEWER_REQUEST_LOG`: set it to a path and every accepted `/simulate` body is appended as
one JSON line. The headless check was then run over all seven string-family scenes against the Rust
server, and the seven logged bodies each carry about sixty keys, most of them other models' sliders.
All seven matched.

**Every simulated trajectory matched to the bit** in all 60 test-derived cases: the energies, the
frames and the audio. That includes the parametric regime's seeded start, which is the one place a
stray last bit would be amplified exponentially. The 19 tolerance cases all come from quantities
computed *after* a run:

- **the off-mode fractions.** These are the `off`, `env` and `level` traces, `purity.off_mode`,
  `cascade.grid_scale`, and the seed floors. Each is a norm of `u - q·shape`, which cancels to about
  1e-7 of the amplitude (or about 1e-14 for a pure mode). BLAS `ddot` fuses its multiply-adds
  (findings §14.2), so its last bit differs, and the cancellation amplifies that difference in
  *relative* terms up to 2.7e-4. In the units the fractions are defined in (the driven amplitude)
  the largest difference is 3e-17. The class is absolute, 1e-15. No gate moves: the purity gate is at
  1e-6, the instability gate at 100 times the floor, and every derived boolean, every growth factor
  and every cascade partner matched exactly.
- **the decay-rate fit**, as in batch 1: 2.8e-15 relative.

**The comparison found one bug** that no test was set up to find. In the parametric regime a
non-numeric `mode_number` got `_fnum`'s message ("'mode_number' must be a number"). The reference
says "mode_number must be an integer". Its `int(_fnum(...))` sits inside `except (TypeError,
ValueError)`, and **`ParamError` subclasses `ValueError`**, so the outer handler catches the inner
refusal and rewords it. Grepped: it is the only `int(_fnum(...))` wrapped that way. A native test
now pins the message.

**The random seed is frozen, not reimplemented.** The parametric regime seeds its run from
`np.random.default_rng(12345).standard_normal(25)`. Those 25 numbers are in `tension.rs` as
`PARAM_SEED_COEF`, with the NumPy version (2.4.6) they came from. Reproducing PCG64 plus NumPy's
ziggurat in Rust would mean keeping a second copy of NumPy's internals in step, and the point of a
fixed seed is only that the perturbation is the same every time.

**`engine::Resonator::step` is now fallible.** The tension string's root-find and the bow's
friction solve can fail inside a step, which in Python was an exception unwinding out of
`simulate`. It is now an `Err` carrying the model's own message, `simulate` stops there, and the
payload is an `internal` error. A step that merely did not *converge* is still not an error: the
models count those themselves, and the payload's convergence gate reports them. Two more of §23.3's
deliberate differences come with this batch, both cases where the reference gave the browser a
dropped connection:

- a step failure is an `internal` error payload;
- a `mode_number` above the number of modes tracked at a small `N`, which made the reference index
  past its array, is an `internal` error payload.

**The two patched constants became arguments.** `measure_mode1` takes the measurement length in
periods, and `build_payload_parametric_with` takes the sweep's work budget. The two tests that
patched `TENSION_MEASURE_PERIODS` and `PARAM_SWEEP_WORK_MAX` pass the lowered value instead. Both
lowered values *replaced a number*, not a function, so an argument covers them exactly.

**Tests carried:**

- `tests/tension.rs`, 28 tests: all 10 of the Duffing section, all 17 of the parametric section,
  and the new `mode_number` message test.
- `tests/bow.rs`, 15 tests: all 14 of the bow section, plus the horizon section's "an exciter
  inherits the horizon of what it drives", which has a bow scene to answer it now.
- Unoptimized, the tension file takes 24 s. That is inside what CI's debug pass tolerates, so it is
  not added to `release_only`.

**Found along the way: the headless check leaks its Chrome.** The second browser run reported
"attaching to the Chrome already listening on :9333". That was the first run's headless Chrome,
still alive under its own temporary profile. It was closed through its own DevTools port
(`Browser.close`) and not by any kill. Its command line carried `--do-not-de-elevate`, which Chrome
adds when it relaunches itself. That makes it likely that the process ID `verify_web_headless.py`
holds is a launcher that has already exited, so its `proc.terminate()` misses the browser. The
likely cause is recorded here rather than fixed, because the script is ported at D7 and the port
should shut the browser down through its port.

### 23.9 Batch D3, done — five scenes with no new numerics

Ten of the twenty-two keys are now native. The five added here are **`sympathetic`** (normal,
transfer and weinreich), **`geometric`** (planar, rotating, whirl and phantom), **`reed`**,
**`radbody`** and **`airload`**.

| corpus | requests | identical | within a class | failing |
|---|---|---|---|---|
| sympathetic | 22 | 22 | 0 | 0 |
| geometric | 24 | 23 | 1 (phantom audio) | 0 |
| reed | 27 | 27 | 0 | 0 |
| radbody | 30 | 30 | 0 | 0 |
| airload | 36 | 36 | 0 | 0 |
| the browser's own requests, all eleven scene/regime pairs | 11 | 11 | 0 | 0 |

The headless check passed all eleven of this batch's cases on the Rust server.

Four things the batch settled:

- **`uniform_filter1d` had one right order out of three plausible ones.** SciPy keeps a running
  sum and divides on output (`tmp += new - old; out = tmp / size`). Dividing inside the update,
  or compensating the sum, each missed tens of thousands of samples against the installed SciPy.
  The matching order was found by testing all three on random data before any of them was used,
  and it matched every sample (`py::uniform_filter1d_nearest`). The weinreich envelopes are
  bit-identical because of it.
- **`np.geomspace` is three steps, and two of them overwrite.** It computes the log-spaced
  `linspace`, raises it as `10 ** y`, and then puts `start` and `stop` back exactly, since
  `10 ** log10(x)` need not be `x`. Transcribed as such (`py::geomspace`), both sweeps' `R` and
  `f` grids are bit-identical.
- **The reed's sweep cache is gone, deliberately.** `_REED_SWEEP_MEMO` existed because the
  threshold-plus-pitch sweep cost ~3.5 s in Python. Natively it is well under a second, so it is
  recomputed. The reference's own docstring called its key "the trap", since a key that misses an
  input returns stale numbers, and a cache that is not needed cannot be keyed wrong. The test that
  guarded the key keeps its assertion: the sweep must move with `f_reed`, `q_reed`, the bell and
  `L`.
- **A step failure's error kind follows the binding's exception type.** The tension string's and
  the bow's steps raised `RuntimeError`, which the reference never caught, so they are `internal`
  here. The geometric string's raised `ValueError`, which the reference reported as a construction
  error, so it is `construction` here.

**The headless check leaked its Chrome a second time**, so this is now a confirmed pattern and not
a one-off. The fix belongs in D7's port of the check: shut the browser down through its DevTools
port and wait for it to exit. The next batch's runs close it by hand the same way. `Browser.close`
takes about two seconds to exit, so a check made immediately afterwards still sees it running.

**Tests carried:**

- `tests/sympathetic.rs` (22), `tests/geometric.rs` (30), `tests/reed.rs` (17),
  `tests/radbody.rs` (16) and `tests/airload.rs` (21): every test in the reference's five sections.
- The one exception is radbody's `R = 0` anchor against the `body` scene, which lands with that
  scene in D4. Airload's anchor against radbody is carried, since both sides are native now.
- The two remaining patched constants (`RADBODY_SWEEP_WORK_MAX`, `AIRLOAD_SWEEP_WORK_MAX`) became
  `build_payload_with(p, work_max)`, the same move as D2's.
- The phantom tests share one run through a `OnceLock`, the reference's module-scoped fixture.
- Unoptimized, the slowest new file is the reed's at about 9 s.

### 23.10 Batch D4, done — the lumped body and the three barrier scenes

Fourteen of the twenty-two keys are now native. The four added here are **`body`** (a string on a
lumped modal body, radiating), and the three configurations of the barrier model: **`jawari`** (a
curved bridge), **`juari`** (a single-node thread) and **`fret`** (a flat rail). They live in
`crates/physsynth-viewer/src/body.rs` and `…/contact.rs`. The three barrier scenes share one
contact runner, one centroid and one band spectrum, where the reference had three near-copies.

**The one new numerical routine is an `rfft` at any length** (`physsynth_analysis::spectrum::rfft`,
and `rfft_mag` beside it). A power-of-two length goes through the existing FFT, and any other length
through Bluestein's chirp-z transform. The chirp's angle is taken from `k² mod 2n` in 128-bit
integers, so it does not lose precision at long records. It got its three native bars before any
payload used it: a direct DFT at twenty lengths (relative error under 1e-13), Parseval's identity
with purely real edge bins, and a pure tone at `n = 1001` landing on its bin. Planting a sign error
in the chirp made all three fail.

| corpus | requests | identical | within a class | failing |
|---|---|---|---|---|
| body | 22 | 8 | 14 | 0 |
| jawari, juari, fret | 51 | 23 | 28 | 0 |
| the browser's own requests (all four scenes, plus radbody again) | 8 | 1 | 7 | 0 |
| every earlier corpus, re-run after this batch | 259 | 235 | 24 | 0 |

**Every trajectory matched to the bit** again, including every contact statistic: the fret's
duty, episode count, active-set sizes, Newton iteration counts and its whole contact raster, the
jawari's wrap edge, and the juari's tuning-curve positions. The spectral centroids, which the
panels round to 0.1 Hz, matched exactly after rounding. The tolerance cases all come from reading a
finished run:

- **the normalized magnitude spectra** (`meta.spectrum.mag`, and now `meta.spectrum.spectra.*.mag`
  too, the nested form the barrier scenes use). Bluestein is not pocketfft's algorithm, so bins
  differ in the last bit. The largest difference is 4.4e-16 against a peak of 1.0, inside the
  existing absolute class of 1e-12.
- **the fret's decay-rate triple.** `rate` is the same least-squares fit as the other scenes'
  `measured_2sigma`, but it needs an ABSOLUTE class: at `sigma0 = 0` it is a fit to conserved
  energy, a rate of about 6e-12 made of rounding noise, and the two fits disagree there by 0.5 %
  relative while agreeing to 3e-14 absolutely. The class is 1e-12 absolute (largest seen 4.3e-14).
  `corrected` is `2 sigma0 <2KE/E>`, where `KE` is a BLAS `ddot` in the reference: relative 1e-13
  (largest seen 2.1e-16).

**One deliberate difference.** At `bridge_stiffness = 0` the body never moves, and the reference
computed its `omega2_consistency` read-out as 0/0 = NaN. The Python server serializes with
`allow_nan=False`, so it would have answered that request with a 500. The Rust payload carries
`null`, which the front-end already renders as "—" (`app.js`: `o2 == null ? "—" : o2.toFixed(2)`,
read rather than assumed). The recorder now marks every such reference
payload, and the comparison treats NaN-versus-null as its own named class.

**One reference quirk carried, and pinned.** The juari reads `N` twice. The first read is a
pre-read that `int()`s it only when `str(N)` is all digits, and otherwise snaps the thread on
`N = 100`. The second read re-snaps the label on the validated `N`. For `N = 40.0`, which the
front-end never sends, the main audio run's thread therefore sits at node `round(0.1 · 100) = 10`,
while the curve and the marker report node 4. It is ported exactly, because this batch's job is
agreement, and a test (`juari_n_pre_read_matches_the_reference_including_its_float_quirk`) pins
it, so that fixing it after the switch is a recorded decision rather than a silent change.

**The headless check passed all four scenes** on the Rust server: shimmer 3.44x, the juari buzzing
2.84x at 0.1 L, the fret's brightness 4.682x, and the body's terminus at 90.8 Hz. **It leaked its
Chrome a third time**, and it was closed through its own DevTools port again. The running viewer
server also has to be stopped (by the process ID recorded at launch) before `cargo test`, because
Windows will not replace a running executable.

**Tests carried:**

- `tests/body.rs`, 14 tests: all 13 of the reference's body section, including the `K = 0`
  bit-identity against a bare fixed/free string, which now asserts the `null` read-out as well.
  The fourteenth is radbody's `R = 0` anchor, deferred from D3: at `R = 0` the radiation-loaded
  body reproduces this scene's energy report, audio bytes, frames and exchange channels exactly.
- `tests/contact.rs`, 44 tests: 11 for the jawari, 13 for the juari (the reference's 12 plus the
  pre-read quirk) and 20 for the fret. The fret's two-sided brightness-peak test, parametrized in
  the reference for wall-clock, is one test here, since the whole file runs in 10 s. Its
  dispatch-`kind` test folds into the two blocks that already assert `kind`.
- Unoptimized, `tests/contact.rs` takes 145 s on this machine (10 s in release), so it joins CI's
  `release_only` list, the first viewer file to do so. It pins no arithmetic spelling, which is
  that list's condition. `tests/body.rs` takes 1 s unoptimized and stays in both profiles.
- The whole workspace: 1,059 Rust tests pass.

### 23.11 Before D5: which eigen-solves reach a payload, and what each needs

An eigen-solver fixes each eigenvector only up to its sign, and a repeated eigenvalue only up to a
rotation inside its group. So a native solver cannot be expected to reproduce SciPy's *vectors* to
the bit, and each use has to be classified before the solver is written, not after a diff fails.
Seven calls, found by grepping `serialize.py` for `eigsh` and `eigh`:

| scene (batch) | call | what reaches the payload | sign | repeated eigenvalues |
|---|---|---|---|---|
| `membrane` (D5) | `eigsh(-L)`, values only | frequencies | — | invariant |
| `plate` and `vk`, supported / free (D5; one helper serves both) | `eigsh(-L)` and `eigsh(K, M)`, values only | frequencies | — | invariant |
| `bore` (D5) | `eigsh(L, M)`, values only | frequencies | — | invariant |
| `plate` guitar (D5) | `eigsh(K, M)` with vectors | mirror parity of mode 1; `abs` of each of the first four modes' overlap with the strike | invariant (a quadratic, and an `abs`) | a guitar outline has none; the bending/twist crossing is resolved by the mirror symmetry, so it bites only if a sweep point sits exactly on the crossing |
| `vkroom` (D6) | dense `eigh(K, M)` with vectors | per-mode energy SHARES `c_j²`, and a projection onto the resolved modes | invariant (squares, and `V Vᵀ`) | **basis-dependent.** The plate is square and free, so it has exactly repeated pairs, and one pair's share can split any way between its two members |

So D5 needs no sign rule at all, only a named tolerance class for the eigenvalues, which come from
an iteration rather than a closed form. The guitar's parity and strike overlaps are sign-invariant
by construction. They get a small absolute class, and the corpus includes a waist point next to the
crossing, so that the class is measured where it is weakest. D6 needs a real decision before its
diff means anything. The choices are to compare the shares summed over each repeated group, or to
fix the basis inside each group by the plate's x↔y mirror (symmetric and antisymmetric members).
The second makes the Rust output well-defined where SciPy's is not. That decision is D6's.

**The order inside D5 follows the reviewer's advice** (the D3 and airbox-halves precedent). The
solver goes first, shift-invert Lanczos in the mass inner product on the existing `SparseLu`, with
its native bars against closed forms (a 1-D Laplacian, a diagonal generalized pencil, and a
rectangle's Navier modes). Then one scene. Only after that do the other five follow, so that a
surprise in the solver blocks one scene rather than six.

**The solver is done** (`physsynth_core::eigs::eigsh_shift_invert`, with `eig::symmetric_eigen`
under it for the small projected problem). It factors `K - sigma M` once with the crate's
`SparseLu`, grows an `M`-orthonormal Krylov basis three vectors at a time with full
reorthogonalization (twice), and accepts a Ritz pair when the part of `Op y` outside the basis
is below `1e-10 |theta|` (ARPACK's test; the first version used the full residual, which §23.13
found to stall on a free plate).
There are no restarts, and the start vectors come from a fixed SplitMix sequence, so a run is
reproducible to the bit. Its bars in `crates/physsynth-core/tests/eigs.rs` all check closed
forms or a dense reference: the 1-D Dirichlet and reflected (singular) operators, the 2-D grid
with its repeated pairs, a mass matrix that varies, a generalized pencil with a triple and the
shift inside the spectrum, and the viewer's largest membrane size (9,801 unknowns, 12 values:
0.66 s in release). `tests/eig.rs` gained five bars for the vector path, including that its
eigenvalues are the value path's to the bit.

Two planted faults, both on a saved copy restored afterwards. A wrong sign in the QL
eigenvector rotation fails three of the vector bars. Block size 1 fails the triple, but the 2-D
grid's pairs still come back doubled at block 1: by the time twelve values converge, rounding has
fed each pair's second direction into the basis. The grid is therefore a closed-form bar and not
the multiplicity bar, and its test says so. That is the reason the diagonal triple exists, since a
diagonal operator gives rounding nothing to mix.

### 23.12 Batch D5, first scene — the membrane

Fifteen of the twenty-two keys are now native. **`membrane`** (disk and rectangle) is in
`crates/physsynth-viewer/src/membrane.rs`, and it is the first scene whose payload runs through
the new solver. Its spectrum markers are the operator's lowest twelve eigenfrequencies.
`horizon.rs` gained `grid2d_block`, the rectangular 2-D read-out, written with a scheme enum so
the plate can reuse it.

| corpus | requests | identical | within a class | failing |
|---|---|---|---|---|
| membrane (test-derived, both domains, the horizon section's rectangles, a 99 x 99 square) | 30 | 28 | 2 | 0 |
| the browser's own requests (disk and rectangle) | 2 | 2 | 0 | 0 |

**The eigenvalues needed no tolerance class after all.** §23.11 expected one, because the markers
come from an iteration. But the payload rounds them to four decimals, and in all seventeen
successful cases every marker came out identical to SciPy's after rounding, as did the fundamental
that sets the animation stride. The two cases within a class are the fitted decay rate, the
existing class. The solver's own bars (1e-11 to 1e-12 relative against closed forms) are what
certifies the unrounded values. The rounding hides a difference at that level except when a value
sits within that distance of a rounding boundary, which a future diff could hit. If one does, the
class to add is relative 1e-10 on `modes_discrete`, and it is not a regression.

**A finding the membrane test now pins: a grid disk has the square's symmetry, not the circle's.**
Its cos/sin pairs of odd angular order come back exactly repeated, since they belong to the
square group's two-dimensional representation. The even-order ones split: at `N = 48` the (2, 1)
pair is 323.32 / 323.47 Hz. The markers carry the exact pairs doubled and the split ones as two
lines, identically to the reference. This is the first scene where the solver's block design is
visible in a payload.

**The headless check passed both domains** on the Rust server (drift 9.4e-15 and 1.1e-14; the
rectangle's read-out "trustworthy to 866 Hz, 40 modes"). Its Chrome leaked again and was closed
through its DevTools port.

**Tests carried:**

- `tests/membrane.rs`, 12 tests: the reference's membrane section (its parametrized cases
  folded into loops), plus one new test, the disk's repeated and split pairs above. Unoptimized it
  takes 48 s. It stays in both profiles: the `release_only` list holds files well past a minute,
  and the name `membrane` would also exclude the core crate's own `tests/membrane.rs`, which must
  run in both.
- `tests/horizon.rs`, 16 tests (+3): the 2-D halves that were waiting for a 2-D scene. They
  check that the index reading is the conservative one, that the worst corner is axial at the
  CFL ceiling and tied below it, and that a disk is refused as a staircase. The membrane was
  added to the monotone-bound, hertz-ceiling and strict-JSON loops.
- The whole workspace: 1,088 Rust tests pass.

### 23.13 Batch D5, continued — the mallet and all three plates, and a solver bug the plate found

Seventeen of the twenty-two keys are now native. **`mallet`** (`crates/physsynth-viewer/src/mallet.rs`)
reuses the membrane's machinery with a hand-stepped contact loop. **`plate`** (`…/src/plate.rs`)
covers the simply-supported rectangle, the free rectangle and the guitar outline. The guitar
brings its waist sweep, which is the first payload to read eigenVECTORS, its outline report, and
the pooled display decimation that cannot split a concave outline.

| corpus | requests | identical | within a class | failing |
|---|---|---|---|---|
| mallet | 24 | 24 | 0 | 0 |
| plate (all three plates, the guitar's edge cases, the refusals) | 40 | 38 | 2 | 0 |
| membrane, re-run after the solver fix below | 30 | 28 | 2 | 0 |
| the browser's own requests (two mallets, three plates) | 5 | 5 | 0 | 0 |

The four in-class cases are all the fitted decay rate. **Every guitar sweep row matched SciPy
exactly**: every parity (to six places), both strike overlaps and both frequencies, over 24 waists
in each of eleven guitars, including the rows either side of the crossing. §23.11's claim holds:
a quadratic form and an absolute value cannot see an eigenvector's sign, and nothing else of the
vector reaches the payload.

**The solver's convergence test was wrong, and the free plate showed it.** The first plate
comparison stalled for more than ten minutes on one request. It was the free plate: its shifted
matrix `K - sigma W` is nearly singular, because the three rigid-body modes sit right beside the
negative shift, so every linear solve carries an error of about 1e-5 relative. The test accepted a
Ritz pair on the FULL residual `||Op y - theta y||`, which includes that solve error. The solve
error does not shrink as the basis grows, so it floored above the bar, and the basis grew to the
whole space. The test is now ARPACK's: the part of `Op y` outside the basis. Each earlier block's
image was orthogonalized into the basis when the next block was built, so only the newest block
contributes, and the solve error is absorbed rather than measured. The free plate now takes 1.8 s
and the guitar 0.9 s, on a par with Python, and the 9,801-unknown membrane bar fell from 0.66 s to
0.43 s. A new bar in `crates/physsynth-core/tests/eigs.rs` builds that free plate and asserts
three things: the basis stays under 120, the rigid trio comes back at zero, and every pair
satisfies `K x = lambda W x`. The old criterion was not planted back to watch that bar fail,
because under it the bar runs for minutes rather than failing. The evidence that it catches the
fault is the viewer run that stalled.

**A finding that is not a porting one: the free plate at the browser's defaults fails its own
lossless bar.** At `N = 60`, `mu = 1` and one second of audio, the drift is 1.3189e-10 against
the 1e-10 bar, and the Python reference reports the identical number and `pass: false`. It is the
model's behaviour at that setting, so the port reproduces it. It is recorded here and not
touched. The bar is not to be tightened or loosened (`CLAUDE.md`), and whether that default
should move is the human's question.

**Tests carried:**

- `tests/mallet.rs`, 10 tests. Nine are the reference's section. The tenth is the "an exciter
  inherits the horizon of what it drives" half that the reference's docstring described and never
  asserted: a struck rectangle reports exactly the plain membrane's read-out. One more pins the
  reference's two `N` defaults: the pre-check reads 60 and the drum is built at 80.
- `tests/plate.rs`, 21 tests: the rectangle and guitar sections, with their parametrized cases
  folded into loops, including the two pooling unit tests on synthetic masks.
- `tests/horizon.rs`, 17 tests (+1, plus the plate added to three loops): one plate key, three
  plates, one horizon.
- Unoptimized, `tests/plate.rs` takes 102 s. It is already excluded from CI's debug pass, because
  `release_only` excludes by file name and `plate` is on the list for the core crate's own
  `tests/plate.rs`. That coincidence is recorded here so that it is not mistaken for a decision.
  `tests/mallet.rs` takes 36 s and runs in both profiles.
- The whole workspace: 1,121 Rust tests pass.

**Two guards added after the review of this batch, and one re-measurement.**
- `eigsh_shift_invert` now refuses a `K` or `M` that is not symmetric to `1e-12` of its largest
  entry. The method would not fail on one: it symmetrizes the projected problem, so it would
  return plausible wrong frequencies, and SciPy would be wrong differently. That diff would read as
  a porting bug. The bore's operator is a slice of a staggered pressure/flow operator, the first
  one this solver sees that is not symmetric by construction, so the check comes before that scene.
- It gives up at a basis of 300 (`MAX_BASIS`) with a named `NotConverged`, as ARPACK gives up at
  its restart cap, instead of growing to the whole space at cubic cost per block. Both guards have
  native bars, and every membrane, mallet and plate payload is unchanged by them. Any solver
  failure in a scene is now `internal` in both the membrane and the plate. The reference's ARPACK
  error was an uncaught `RuntimeError`; the membrane had said `construction`.
- The block-of-one fault was re-planted after the convergence fix, since the new test accepts
  pairs sooner, which is the direction that could let a single-vector space pass. The triple
  still fails at block 1, so the multiplicity bar stands.

### 23.14 Batch D5, continued — the bore

Eighteen of the twenty-two keys are now native. **`bore`** (`crates/physsynth-viewer/src/bore.rs`)
is the first wind scene. Its field is pressure, and its loss is booked: a lossless tube with a
radiating bell conserves `acoustic + radiated`, and the energy panel plots the split. Three of its
panels read the operator instead of a render. The resonances come from a generalized solve on the
free pressure nodes. The dispersion-versus-lambda curve is eighteen such solves and no stepping.
The one-bounce reflection is checked against `r = (R - Z0)/(R + Z0)`.

| corpus | requests | identical | within a class | failing |
|---|---|---|---|---|
| bore (both ends, five `R/Z0`, four `N`, the refusals) | 24 | 9 | 15 | 0 |
| the browser's own requests (radiating, open) | 2 | 0 | 2 | 0 |

Every trajectory, energy, reflection number and eigenfrequency matched exactly. The two classes
this scene adds are both about what a panel does to a number after the run:

- **The band spectrum** (`meta.spectrum.spectrum.mag`), absolute 1e-9. An FFT's rounding is
  relative to the whole RECORD. At the anechoic bell the pulse leaves the tube, and the plotted
  low band holds only 1.06e-6 of the record's peak, so normalizing to that band amplifies the
  rounding about a millionfold. Measured: 2.8e-11. The contact and body spectra, whose band holds
  most of the record, stay in the 1e-12 class.
- **`dispersion.order[8]`, no tolerance.** At lambda = 1 the scheme is dispersionless, both
  departures round to 0.0 cents, and "order" is the ratio of two rounding residues: 2.667 in the
  reference, 1.2 here, for the same request. It is meaningless in both, and the front-end reads
  only `order[0]`. Shipping `null` there would be the honest payload. That is a change to the
  reference's output, so it is recorded here for after the switch rather than made now.

**Before this scene the solver got the reviewer's guard** (§23.11's addendum): it refuses a
non-symmetric pencil. A new native bar in `crates/physsynth-core/tests/eigs.rs` checks the bore's
own free-node pencil: `L` exactly symmetric, `C` a positive diagonal, and the first five
resonances on `(2n-1) c0 / 4L` to 1e-10 at lambda = 1. The bore is the first operator the solver
sees that is not symmetric by construction, and this bar asserts it rather than assuming it.

Also carried: the reference formats the ratio refusal with Python's `{:.3e}` (`1.000e+03`), which
Rust's `{:.3e}` spells `1.000e3`. A local `sci3` restores the signed two-digit exponent, and a
test pins the message.

**Tests carried:** `tests/bore.rs`, 17 tests: the reference's 16, plus the refusal's format.
Unoptimized they take 2 s. The whole workspace: 1,141 Rust tests pass.

### 23.15 Batch D5, continued — the von Kármán plate

Nineteen of the twenty-two keys are now native. **`vk`** (`crates/physsynth-viewer/src/vk.rs`) is
the gong (supported) and the cymbal (free). The native `VkParams` carries a complete linear plate
(`lin`), so the plate scene's marker solve and horizon block serve it unchanged. The run loop is
hand-written for the convergence record the energy verdict is gated on.

| corpus | requests | identical | within a class | failing |
|---|---|---|---|---|
| vk (both boundaries, linear toggle as bool and string, hardest strike, the refusals) | 20 | 19 | 1 | 0 |
| the browser's own requests (gong, cymbal) | 2 | 2 | 0 | 0 |

**The nonlinear trajectories match to the bit**, including every per-step iteration count behind
`max_iters` and every fixed-point residual behind `worst_residual`. That is expected, not lucky:
the reference steps the same native plate through the binding. The one case in a class is the
fitted decay rate. One spelling mattered: this scene's time axis is `np.arange(n) * k`, not
`i / fs`, and the two can differ in the last digit, so it is transcribed as the reference wrote it.

**Tests carried:** `tests/vk.rs`, 5 tests (the reference's section; the linear toggle is also sent
as the string `"false"`), plus two horizon tests the vk scene unblocked: a zero horizon ships
`null`, and the nonlinear plate is refused while its linear twin is measured. Unoptimized,
`tests/vk.rs` takes 46 s and stays in both profiles.

### 23.16 Batch D5, done — the plate body

Twenty of the twenty-two keys are now native. **`platebody`**
(`crates/physsynth-viewer/src/platebody.rs`) is the lumped body scene with the modal body swapped
for a grid plate: the supported soundboard or the free cymbal. The coupling and its exact stability
guard are the core's `StringPlateBridge<Plate>`. It reuses the body scene's spectrum, terminus and
consistency helpers, and the plate scene's marker solve.

| corpus | requests | identical | within a class | failing |
|---|---|---|---|---|
| platebody (both boundaries, the soft and decoupled bridges, both guards, the refusals) | 26 | 13 | 13 | 0 |
| the browser's own requests (free, supported) | 2 | 0 | 2 | 0 |

Every in-class difference is the pooled spectrum's last digits, plus one expected null.
**`K = 0` repeats the body scene's finding**: the plate never moves, the omega² read-out is 0/0,
and the reference's own server would answer that request with a 500. The port ships `null`, as
§23.10 decided for the lumped body.

**Tests carried:** `tests/platebody.rs`, 12 tests: the reference's section, with the `K = 0`
bit-identity against a bare string also asserting the `null`. Unoptimized they take 85 s, so
`platebody` joins CI's `release_only` list. The name is unique across the crates, unlike `plate`
(§23.13). The whole workspace: 1,160 Rust tests pass.

**D5 is complete**: a native eigensolver, then membrane, mallet, the three plates, the bore, the
von Kármán plate and the plate body. Two keys remain, `airbox` and `vkroom`, and that is D6. D6
must settle §23.11's basis decision before its first diff: vkroom's per-mode energy shares on a
square plate depend on the basis chosen inside each repeated pair.

**A counting error, corrected here.** From §23.13 on, each batch's opening count was one too high,
because the mallet and the plate were counted as reaching eighteen when they reached seventeen. The
three commit messages that repeated the numbers (mallet and plates, bore, von Kármán plate) carry
the same error, and pushed history is not rewritten for it. The dispatch table is the source of
truth, and it reads twenty `true` and two `false` (`airbox`, `vkroom`).

### 23.17 Batch D6, first half — the air box

Twenty-one of the twenty-two keys are now native. **`airbox`** (`crates/physsynth-viewer/src/airbox.rs`)
is string -> body -> a 3-D room: the first `dims: 3` payload, sent as a set of three decimated
orthogonal slices. It runs on the core's `StringBodyBridge<RoomLoadedBody>` stepping into a
native `AirBox`, with the step order the reference's contract: the bridge queues the port's
injection, then the room steps once.

| corpus | requests | identical | within a class | failing |
|---|---|---|---|---|
| airbox (three wall kinds, four mic positions, three Courant numbers, the snap, the refusals) | 37 | 37 | 0 | 0 |
| the browser's own requests (rigid, absorbing) | 2 | 2 | 0 | 0 |

**Every field matched to the bit**, including the lattice light-cone integers, the five booked
channels, the cross-ledger residual, the slice frames and the colour-scale reference. That last one
took one correction. The reference is `np.percentile(live, 55)`, shipped unrounded, and the first
transcription used the virtual index of NumPy's general quantile method, `n q + (alpha + q (1 -
alpha - beta)) - 1`. The default `linear` method overrides it with `(n - 1) q`. The two are
algebraically equal and differ in the last bit, and the diff caught it in 10 of 37 cases.
`airbox::np_percentile` is now pinned by a test against values NumPy printed.

**Two Python number formats became shared helpers**, `py::sci` (`f"{x:.2e}"` is `1.20e+09`, where
Rust writes `1.2e9`) and `py::fmt_g` (`f"{x:g}"`), with `tests/py_format.rs` pinning both
against CPython's output. The bore's local copy was folded into `py::sci`.

**Tests carried:** `tests/airbox.rs`, 14 tests: the reference's 13 plus the percentile's pin.
Unoptimized they take 51 s and stay in both profiles. The whole workspace: 1,176 Rust tests pass.

**What remains is `vkroom`, and it needs a decision first.** §23.11 found that its per-mode
energy shares depend on the basis chosen inside each repeated pair. On reading the scene, the
exposure is narrower than feared: the shares reach the payload only through `modal_drift` and
`modal_drift_twin`, each a total-variation distance between the first and last window's share
vectors. The efficiency figures use the modes only through a projection that does not depend
on the basis. So one pair of scalars is basis-dependent, in the reference too. How to treat
it is the human's call.

**Measured, and the decision is moot.** The reviewer's point was that the dependence needs the
energy's direction inside a repeated pair to ROTATE during the run: a fixed direction contributes
`|g - g'|` in every basis. So this was measured on the reference rather than argued
(`W:\temp\claude\viewer-port\vkroom_basis_probe.py`). Every exactly repeated pair of SciPy's
eigenvectors was rotated (71 pairs on the shipped plate), the payload was recomputed, and the
result was compared. Five configurations were used (default, short, baffled, linear, harder
strike) at four angles (0.3, 0.7, 1.1 and 2.0 rad). `modal_drift`, `modal_drift_twin` and every
efficiency figure were identical to all six shipped decimals in every case. Any orthonormal basis
of a pair is such a rotation (or a sign flip, which a square cannot see), so no solver's choice
can move the payload. `vkroom` is ported as the reference defines it, and the only new numerical
obligation is the one the reviewer named: the vectors must be `W`-orthonormal (`V^T W V = I`), as
`scipy.linalg.eigh(A, B)` returns them.

### 23.18 Batch D6 done — the gong in the room; every scene is native

**All twenty-two keys are now native**, and no model is refused as `unported`. **`vkroom`**
(`crates/physsynth-viewer/src/vkroom.rs`) is the von Kármán cymbal as a radiating surface in the
3-D room, baffled in a wall or suspended mid-room, on the core's `RoomGrid<VkSeam>`. The claim is a
separation: the struck plate's radiation pattern moves during the strike, and its linear twin's does
not.

The one new numerical routine is `eig::generalized_eigen_diag`, the dense `A x = lambda D x` for a
positive diagonal mass, with `D`-orthonormal vectors as `scipy.linalg.eigh(A, B)` returns them.
Its bar in `crates/physsynth-core/tests/eig.rs` uses a free square plate, whose spectrum has the
rigid trio and exactly repeated pairs. It asserts `V^T W V = I`, `K V = W V Lambda`, the trio, a
repeated pair, and agreement with the independent sparse shift-invert route. A planted fault that
skips the back-transform fails it.

| corpus | requests | identical | within a class | failing |
|---|---|---|---|---|
| vkroom (both tiers, linear, quiet, coarsest plate, an off-default rig, the guards) | 18 | 9 | 9 | 0 |
| the browser's own requests (suspended twice, baffled) | 3 | 0 | 3 | 0 |

**Every state array, every claim figure and every convergence count matched to the bit.** The
in-class differences are the radiated-energy ledger's last bit, which §16 decided: the binding
accumulates it through NumPy's dot, the core through its own read-out sum, and the ledger never
feeds back. That reaches the scene total and every fraction of it (at most 1.6e-15 relative), and
the drift and residual figures that are differences of two nearly equal ledgers (at most 1.7e-16
absolute). These classes are the first ones restricted to ONE model (`compare.py`'s
`MODEL_CLASSES`), so the loosening cannot hide a regression in another scene.

**The final regression**, every corpus re-run after the last scene: 17 corpora, 561 requests,
0 failing.

**Tests carried:** `tests/vkroom.rs`, 12 tests: the reference's section, with the module-scoped
fixture as a `OnceLock`. Unoptimized they take 145 s, so `vkroom` joins `release_only`. The whole
workspace: 1,189 Rust tests pass.

**What is left is D7**, and it is not only engineering. Freeze the reference's outputs as a native
fixture while the Python still exists. Port the headless check, and make it shut its Chrome down
through the DevTools port: it leaked every run of this phase. Then switch the served viewer to the
Rust binary, and delete `web/serialize.py`, `web/server.py` and `tests/test_web_backend.py`. That
last step removes the reference the whole of phase D was checked against, so it waits for the
human's go-ahead.

### 23.19 Batch D7 done — the freeze, the browser check in Rust, and the Python viewer deleted

The human's go-ahead came 2026-09-28 ("do all of D7"), together with a second decision: the free
plate's default render fails its own lossless bar (drift 1.32e-10 against 1e-10, the Python
identically), and the answer was to **change the scene's opening settings, not the bar**.

**The free plate opens green.** A regime-only override (`"plate:free"` in `MODEL_RANGES`) was the
obvious edit and the wrong one: `regimeSwitchRerangs` exists so that a supported → free switch keeps
the user's sliders, and a regime key would reset them on every switch. So the plate's base defaults
moved for both rectangles, from `N = 60, mu = 1` to `N = 48, mu = 2` (the guitar outline keeps its
own `N = 40, mu = 2`). Measured on the 1 s default render:

| | old (N 60, mu 1) | new (N 48, mu 2) |
|---|---|---|
| supported: drift, render time | 1.35e-11, 76 s | 1.60e-11, 13 s |
| free: drift, render time | **1.32e-10 (fails)**, 86 s | 1.49e-11, 15 s |

The supported plate's fundamental moves from 62.817 to 62.808 Hz, and its horizon read-out now
quotes 251 Hz, because the grid is coarser. Both open green in a real browser. The bar is
untouched: a user who drags the free plate back to `N = 60, mu = 1` sees it fail, honestly.

#### 23.19.1 Every Python test accounted for, before the file went

`tests/test_web_backend.py` was the viewer's specification (§5): 338 functions, 429 once
parametrized. A count comparison (about 390 Rust viewer tests) proves nothing, so each function was
mapped to the Rust test that carries it. The pairs were proposed by name and every weak pair and
every miss was read by hand. The table is `docs/dev/viewer-test-accounting.md`.

**338 of 338 carried; none dies.** Two had not been carried by D1–D6 and were added now: the phantom
regime's own work budget, and the bridge-coupled string refusing a horizon **through a real payload**
(the table was asserted, the path that reads it was not). The horizon-mark vocabulary check moved to
`tests/front_end.rs` and now reads the Rust harness. The deletion took pytest from 2,099 to 1,669:
the file's 429, plus one parametrization of `test_xdist_groups.py`, which derives its population from
the test files with module-scoped fixtures.

#### 23.19.2 The freeze

`crates/physsynth-viewer/tests/frozen.rs`, fixtures in `tests/frozen/`: **588 requests over 26
corpora**, the phase's whole comparison set, recorded from the Python on a freshly reinstalled wheel.
The payloads were 180 MB; the fixture is 3.3 MB, and it is a digest that gives up no exactness:

- keys, strings, bools, nulls, ints, and int-versus-float type are kept as they were;
- numbers are kept, except inside a numeric list longer than 16, which becomes its length and an
  FNV-1a hash of each element's type tag and bits;
- a base64 buffer becomes its byte length and an FNV-1a hash;
- a buffer or long list under a tolerance class keeps its hash, 64 samples plus its argmax, and the
  max and min of 64 blocks. The class's bar bounds these soundly: an extreme cannot move further than
  the worst element does.

`compare.py`'s tolerance classes are carried as they were, reasons included, none widened; the
model-restricted ones stay restricted. The coverage is **derived**: every key in `MODELS` must have a
successful frozen case, and every option in the front-end's own `DOMAIN_OPTS` must be named by one.
The first run of that guard found the air box's `open` walls missing, which turned out to be the
guard's error: the room's select is sent as `walls`, not `domain` (`gatherParams`).

**Proved to bite while the Python still existed.** The Rust comparator's verdict, run over all 26
corpora, matched `compare.py`'s count for count: the same number of identical, in-class and failing
cases in every corpus. A test corrupts one thing at a time — one ulp, one buffer byte, one audio
sample, an int, an int turned float, a key, a length, a string — and each must fail. The value
corruptions need exact mode (below), so off the recording platform that half is reported as
*ignored*, not as a pass that asserted nothing.

#### 23.19.3 Exactness is a claim about the platform — measured

§23.7 warned that Rust's `sin`, `ln` and `log2` come from the platform's C library. The freeze was
run on three machines before landing:

| machine | result |
|---|---|
| the dev machine (`x86_64-pc-windows-msvc`, the recording platform) | 588 of 588 |
| GitHub `windows-latest` | **588 of 588, to the bit** |
| GitHub `ubuntu-latest` | 430 of 588; the 158 others differ |

On Linux, 157 of the 158 differ only in float values: last-bit drift through a transcendental,
which moves figures that are themselves rounding noise by tens of percent (a lossless drift of
6.1e-15 against 8.3e-15). The parametric tension scene's own instability carries the last bit to
about 1e-6 in its audio. **No tolerance that admits that asserts anything anywhere else**, so none
was attempted. The 158th was not a float at all, or rather was one in disguise: the geometric string's
`orbit.u` is a float32 buffer under a key that does not end in `b64`, so the freezer had kept it
as a string. A scan of every recorded payload for long base64 strings found it and `orbit.w`
and nothing else; frozen as buffers, **no int, string, key or length differs on Linux in any of
the 588 cases.**

So the freeze has two modes. On the recording platform, everything above is compared. Everywhere
else only what cannot pass through a transcendental is: keys, types, lengths, ints, strings, bools.
CI runs both: the Linux `rust` job in structure mode, and a new `frozen-windows` job exactly. This is
ledger #28 and findings §22.1 again, now for a whole payload rather than one reduction.

In release the freeze takes about 3.5 minutes on this machine (4.5 on the Windows runner); the three
default plate renders are four of the ten single-thread minutes. It joins `release_only`.

#### 23.19.4 The browser check, in Rust

`crates/physsynth-viewer/examples/verify_headless.rs` replaces `scripts/verify_web_headless.py`:
the same probe, the same verdict (status, painted pixels, deep-link notes, the horizon strip and the
panel's horizon mark), and three more cases (stiff, damped, damped-lossy), so every model is rendered
at least once — a guard in `tests/front_end.rs` derives that from `MODELS`.

The WebSocket client is **hand-written on `std::net`**, not a crate. `tests/deps.rs` walks only what
the viewer ships, so a dev-dependency's whole tree would go unreviewed, and the protocol needs little:
one handshake, masked text frames out, fragmented text frames in, a pong for a ping. Its first run
found its own bug: the DevTools endpoint ignores `Connection: close`, so a read-to-end only ended at
the timeout. It reads by `Content-Length` now.

**It shuts its browser down through the browser's own port.** The Python harness ended with
`proc.terminate()`, which on Windows stops only the launcher stub; the real browser survived every
run of this phase. Measured again here: the spawned PID was 24812 and the browser that answered the
port was 31880, relaunched by the stub. So the harness gives the browser its own profile directory
and port, sends `Browser.close`, and waits for the port to go quiet. Only if the browser outlives
that does it kill anything, and only the process it spawned. A browser it attached to is never
closed. Full run against the Rust server: **40 of 40 pass, and nothing is left on the port.**

#### 23.19.5 Deleted, switched and left

Deleted: `web/__init__.py`, `web/serialize.py` (10,263 lines), `web/server.py`,
`tests/test_web_backend.py` (5,524) and `scripts/verify_web_headless.py`, with the shard-cost entry.
The README launches `cargo run --release -p physsynth-viewer -- serve`. Comments in five Python
modules and two tests that named `web/serialize.py` as a live reader now say it was. `web/` holds
only `static/`.

**Left for later, recorded rather than done:** the binding's names that only the viewer reached
(`_stretch`, `_bridge_displacement`, `_support`, `_b`, `_open_left`, `_open_right`, and the six state
arrays written on the geometric string) are now dead code, and their doc comments still name the
viewer as the reason they exist. They go with the binding.

**Phase D is done.** The served viewer is the Rust binary, and nothing in the viewer path imports
Python.

---

## 24. Phase C, carrying batch 1 — the free orthotropic plate

Done 2026-09-29. **The first batch that retires a Python file without porting anything first.** Every
function `tests/test_free_plate_orthotropic.py` called — the four-constant free assembly, the
plate's split API, the free beam, both eigensolvers, the two closed-form oracles — was already
native. So this batch is the shape the rest of phase C takes: read each Python test's assertions,
find or write the native bar, prove the new bars are live, delete. It sets the pattern, and its
accounting is written out in full for that reason.

The new file is `crates/physsynth-core/tests/plate_free_grain.rs`: 30 `#[test]`s for the file's 29
functions (31 pytest cases).

### 24.1 §9.4's map had gone stale, and was re-measured

The batch was first scoped from §9.4, which lists the **sympathetic strings** as the one model with
genuinely zero native bars. That was true on 2026-09-07 and has not been since §12 (2026-09-08),
which ported the class and retired its file; its bars are in `connection.rs` and
`connection_body.rs`. A map is dated, and §9's is three weeks and twenty batches old. Re-measured
after this batch: **62 Python physics files, 625 test functions** — the seven non-physics files
(`test_binding_surface`, `test_ci_workflow`, `test_rust_parity_ops2d`, `test_shard_partition`,
`test_stability`, `test_xdist_groups`, `test_analysis_frozen`) excluded, since §2 already accounts
for them.

### 24.2 Where a bar that needs both crates lives — the human's call

Three bars check the plate's operator against the analysis crate's closed forms
(`free_plate_twist_bound`, `free_plate_coupling_form`). No native test used both crates before.
Three homes were put to the human: a **test-only dependency** of `physsynth-core` on
`physsynth-analysis`, copying the two formulas into the test, or the viewer crate (which already
links both). **Chosen: the test-only dependency.** `tests/deps.rs` walks normal and build edges
only, so the shipped core stays dependency-free exactly as it is for `serde_json`, and the analysis
crate depends on nothing, so there is no cycle. Copying the formulas was the option that looked
safest and was not: it would leave the analysis crate's own versions unchecked against the plate,
which is the claim the Python made.

### 24.3 SciPy retires as the oracle, and ARPACK was the least accurate solver in the room

Seven of the Python tests took their truth from `scipy.linalg.eigh` (LAPACK) or
`scipy.sparse.linalg.eigsh` (ARPACK). Carrying only their inequalities would leave the native
eigensolvers as the sole referee of their own answers, so SciPy's values at the file's exact
fixtures were recorded before deletion (with the wheel freshly reinstalled) and are asserted as
`SCIPY_*` constants, alongside the inequalities rather than instead of them.

Recording them found something. At `test_free_plate_is_not`'s shift of `-1e-4` — three orders
closer to the rigid trio than `free_plate_low_eigenfrequencies` goes — **ARPACK was off by up to
7.2e-7 in the eigenvalue** (`g_1 = -0.1`: ARPACK `31.9445589`, LAPACK dense `31.94453600587`).
The native shift-invert at the same shift lands within 1.7e-11 of LAPACK, and the native dense
solver within 4.8e-11. The Python bar was `ratio > 4` and could never have noticed. So where the
Python used ARPACK, the recorded referee is **LAPACK's dense solve of the same pencil**, and the
ARPACK values are kept in the doc comments as the finding.

At `N = 80` (6,561 unknowns) the referees swap. A dense solve's error is absolute, about
`eps · mu_max`, and `mu_max` grows like `h^-4`, so relative to the low modes LAPACK's floor there is
~1e-9. The native shift-invert sits 2.1e-9 from LAPACK and 2.1e-10 from ARPACK: the two *shifted*
solvers agree and the dense one is the outlier.

**Every recorded margin was then measured, not only the ones that failed** (findings #80). Passing
is not headroom: three bars passed within 1.7x–4.7x of their tolerance and were widened in a
follow-up commit, each with its measurement written beside it. All three gaps are the same cause,
LAPACK's `eps·mu_max/mu` floor on the larger pencils:

| comparison | measured | bar | headroom |
|---|---|---|---|
| zero-torsion scale / fifth eigenvalue | 5.6e-16 / 9.2e-14 | 1e-12 / 1e-9 | > 1,000x |
| guard pencils, scale / first elastic | 1.2e-15 / 3.3e-11 | 1e-12 / 1e-9 | ≥ 30x |
| free beam | 1.1e-12 | 1e-10 | 94x |
| mode reordering (625 unknowns) | 6.0e-11 | 1e-10 → **1e-9** | 1.7x → 16x |
| the two fundamentals | 7.9e-12 | 1e-9 | 127x |
| split lambdas | 3.0e-10 | 1e-9 → **1e-8** | 3.4x → 34x |
| convergence N = 20 / 40 / 80 | 5.3e-12 / 1.2e-10 / 2.1e-9 | 1e-8 → **2e-8** | 4.7x → 9x at N = 80 |

**The lesson generalises past this file:** a recorded oracle is only as good as the configuration it
was run in, and a test whose assertion is loose never exercised that. Record with an accurate
solver, not with the one the test happened to call.

### 24.4 Six deliberate breakages

Each was planted in `crates/physsynth-core/src` (snapshot, one change, release build of the new
file, restore from the snapshot, byte-compare). All six were caught.

| breakage | red | what catches it |
|---|---|---|
| `grain_x` / `grain_y` swapped in the assembly | 6 | the direct build, the beam reduction (both), the mode reordering, the guard pencil, the zero-torsion pencil |
| coupling 10% low | 8 | the coupling probe, the direct build, every recorded eigenvalue; the twist probe does NOT (it is blind by design) |
| torsion's factor 4 → 2 | 9 | the twist numerator and its convergence, the direct build, every recorded eigenvalue; the coupling probe does NOT |
| coupling not symmetrized (`cross` doubled, not `cross + crossᵀ`) | 12 | symmetry, and the only breakage the **energy ledger** sees — conservation and passivity go red here and nowhere else |
| the constructor swaps coupling and torsion | 8 | the two seam bars, the von Kármán anchor, the implied-Poisson bar, the eigenvalues |
| the constructor swaps `grain_x` and `grain_y` | 2 | **only** the two non-square seam bars |

The last row is the Python docstring's warning made measurable: on a square an x/y transposition is
invisible, so every square-plate bar in the file passes it. The fourth row is the file's other
warning: any symmetric `K` conserves exactly, so the ledger validates the time-stepper, not the
material.

### 24.5 The retirement rule, discharged

| retired | native bar, or verdict |
|---|---|
| `test_isotropic_split_is_bit_identical_on_every_grid` | **verdict: one code path.** `free_plate_stiffness_from_mask` fills a missing half-split with `unwrap_or(nu)` and `unwrap_or(0.5 * (1.0 - nu))`, the same expressions a caller passes, into the same assembly; a carried equality compares a computation with itself (finding #78). `the_isotropic_split_is_the_isotropic_plate_by_construction` pins the premise over the same seven grids and four `nu` |
| `test_resonator_default_free_plate_is_bit_identical_to_the_helper` | `the_default_free_plate_is_the_isotropic_operator_and_its_split_is_nus` |
| `test_resonator_builds_the_operator_its_own_parameters_imply` | `a_grained_plate_builds_the_operator_its_own_parameters_imply` — **sharpened**: asserts the fixture is non-square, which is its whole point (§24.4, last row) |
| `test_matches_direct_assembly_with_four_distinct_constants` | `the_assembly_matches_a_direct_per_node_build_with_four_distinct_constants` |
| `test_operator_stays_symmetric_with_a_grain` | `the_grained_operator_stays_symmetric` |
| `test_rigid_body_nullspace_survives_the_grain` | same name |
| `test_zero_torsion_puts_the_saddle_into_the_nullspace` | same name, plus LAPACK's scale and fifth eigenvalue |
| `test_twist_quotient_is_blind_to_the_other_three_constants` | `the_twist_quotient_is_blind_to_the_other_three_constants` |
| `test_twist_numerator_is_the_closed_form_and_exact_without_cancellation` | same name |
| `test_twist_quotient_converges_to_the_continuum_bound_at_h2` | same name |
| `test_fundamental_is_below_the_twist_bound_and_the_bound_is_informative` | same name, through the analysis crate's `free_plate_twist_bound`, plus LAPACK's two fundamentals |
| `test_transverse_independent_spectrum_is_the_free_beam_exactly` (x, y) | `a_field_constant_along_one_axis_has_the_free_beams_spectrum_exactly`, both axes, plus LAPACK's beam eigenvalues |
| `test_coupling_breaks_the_beam_reduction_without_changing_its_energy` | same name |
| `test_coupling_probe_hits_its_exact_discrete_value` | `the_coupling_probe_hits_its_exact_discrete_value`, through `free_plate_coupling_form` |
| `test_only_the_coupling_probe_sees_the_coupling_rigidity` | same name |
| `test_free_grain_admissibility_is_rejected_at_construction` | same name — `plate.rs`'s `the_grain_guards_are_the_branchs_own` had only one value per side; this carries all five and the just-inside case |
| `test_the_two_boundaries_admissible_sets_differ` | same name |
| `test_the_guard_is_conservative_and_the_claim_is_one_sided` | same name, plus LAPACK's scale and first elastic eigenvalue for all six pencils |
| `test_the_split_api_refuses_every_ambiguous_call` | same name — **sharpened**: each refusal asserts its exact variant as well as the Python's `match=` words |
| `test_supported_is_blind_to_the_split` | `the_supported_plate_is_blind_to_the_split` |
| `test_free_plate_is_not` | `the_free_plate_is_not_blind_to_the_split`, plus LAPACK's five lambdas (§24.3) |
| `test_the_grain_reorders_the_free_plates_modes` | same name, plus LAPACK's eigenvalues for both plates |
| `test_the_grain_is_worth_more_here_than_the_supported_branch_suggested` | same name |
| `test_energy_conserved_with_a_grain` (mu 0.5, 4.0) | `a_grained_free_plate_conserves_its_energy`, both |
| `test_passivity_with_a_grain` | `a_lossy_grained_free_plate_is_passive` |
| `test_self_convergence_order_h2_with_a_grain` | `a_grained_free_plate_self_converges_at_second_order`, plus LAPACK's nine eigenvalues |
| `test_material_chain_returns_a_consistent_split` | `the_material_split_adds_back_to_its_cross_term`, with `plate.rs`'s `isotropic_material_comes_back_at_exactly_one` (the isotropic split) and `spruce_is_not_an_isotropic_plate_with_one_axis_stretched` (the 82% torsional share), which already asserted the other clauses |
| `test_von_karman_plate_is_untouched` | `the_von_karman_plates_bending_operator_is_the_isotropic_free_plate` |
| `test_an_implied_poisson_ratio_above_one_half_is_admissible` | same name, and `a_plain_isotropic_call_still_refuses_that_poisson_ratio` (a `should_panic`, since the refusal is an `assert!`) |

Also deleted: `tests/helpers.py`'s `make_orthotropic_free_plate` and `spruce_free_grain`, which had
no other caller, with the helper module's now-unused import of `grain_ratios_from_material`, and
the file's `scripts/shard_costs.json` entry. The model's design record,
`docs/dev/orthotropic-free-plate-plan.md`, now points at the native file.

### 24.6 Cost and counts

- **pytest 1,669 → 1,637.** The file was 31 cases; the 32nd is
  `test_xdist_groups.py::test_a_module_scoped_fixture_is_not_split_across_groups`, which is
  parametrized over every test file.
- **Native: 1,224 → 1,254 in release.** The new file runs in 1.5 s in release and ~30 s in debug
  on the dev machine; the Python file was 5.7 s of shard cost. It is **not** on the CI job's
  `release_only` list, so it runs in both profiles by default — 30 s is small against the 64–495 s
  files that list exists for, and several of these bars assert bit-identity, which is the kind the
  both-profiles rule protects. Whether to add it is the human's call.

### 24.7 What is next

The next carrying batch is `tests/test_plate_orthotropic.py` (19 functions), the *supported*
grained plate — the other half of the material story, and held back from this batch so that this
one's pattern could be checked first. After it, §24.1's 625 functions are what is left of phase C.

## 25. Phase C, carrying batch 2 — the supported orthotropic plate

Done 2026-09-29. The other half of the material story: `tests/test_plate_orthotropic.py` (19
functions, 22 pytest cases), the *simply-supported* grained plate, whose three ratios enter as
`B = g_x δ_xx² + 2 g_h δ_xx δ_yy + g_y δ_yy²`. Every function it called was already native, so this
is §24's shape again: read the assertions, write the bars, plant breakages, delete. The new file is
`crates/physsynth-core/tests/plate_grain.rs`, 19 `#[test]`s (the two parametrized Python tests,
over two grains and three `mu`, loop inside one bar each).

### 25.1 What came from outside the project's code, and was recorded

Everything the Python file asserted already ran through the Rust binding — the model, the operator
builders, the analysis oracles — so almost every number in it was Rust checking Rust. Exactly two
referees were independent, and both were recorded before deletion (wheel reinstalled first; NumPy
2.4.6, SciPy 1.17.1):

- **LAPACK's eigenvalues** of the guard's N = 8 operators (`np.linalg.eigvalsh`): smallest
  `56.543774203884176` just inside the cross-term floor, `-1639.6437722933872` just outside it.
  The native dense solver sits 1.9e-11 and 1.8e-10 away. A dense solve's error is absolute, about
  `eps · lambda_max` (~1.4e-10 here), so the bar is written in that unit — 20 of them, ~2.9e-9 —
  rather than §24's bare `1e-9`, which the outside value would have met with only 5.7x to spare.
- **SciPy's sparse `L @ L`**, which the isotropic-default bar compares the plate against. Natively
  that product cannot be `biharmonic_from_mask` — it is what `Params` itself calls, and comparing a
  builder with itself proves nothing — so the bar builds `L @ L` by hand (dense, `k` ascending).
  The hand loop is **proved faithful** rather than assumed: its gap to the general assembly is
  SciPy's recorded `1.70601310856e-16` exactly, over the same 195 entries, and both are asserted.
  That is multiply-and-add with no transcendental, so it is a cross-platform claim.

### 25.2 Every margin measured; two were thin

| bar | measured | bar | headroom |
|---|---|---|---|
| sine residual (strong / wild) | 4.5e-12 / 3.0e-12 | 1e-11 → **1e-10** | 2.2x → 22x |
| LAPACK, outside the floor | 1.8e-10 | 1e-9 → **20·eps·λ_max ≈ 2.9e-9** | 5.7x → 16x |
| LAPACK, inside the floor | 1.9e-11 | same | 150x |
| uniform grain vs isotropic, spectrum / 200 steps | 3.6e-16 / 1.9e-12 | 1e-14 / 1e-11 | 28x / 5x |
| convergence order | 2.003 | > 1.8 | — |
| FFT fundamental | 0.023 cents | 5 cents | 200x |
| drift at mu 0.5 / 2 / 8 | 1.3e-13 / 4.4e-12 / 1.8e-13 | 1e-10 | ≥ 23x |
| lossy fundamental vs 2σ | 0.45% | 2% | 4.4x |
| mutation separation (swap / factor 2 / D_1) | 1.25 / 0.095 / 0.136 | > 0.05 | factor 2 only 1.9x |
| damping split, grained | 22.3% | > 15% | — |
| detune: (3,1) / (2,4) / spread | 2.3% / 29.0% / 21.7x | < 5% / > 25% / > 15x | — |
| level ratios | 0.811 – 1.117 | straddle 1, within 30% | — |

The sine residual is §24.3's cause in a different place: `B`'s entries are ~1e8 (`g · 64/h⁴`) and
the (1,1) eigenvalue ~1e3, so `eps · |B| / q` is ~1e-11 by itself. Any wiring error puts the
residual near 1, so the wider bar loses nothing. The deterministic physics margins (the 4.4x on the
decay rate, the 1.9x on the dropped factor of 2) are not rounding and were left as the Python set
them, with the measurement written beside each.

### 25.3 Seven deliberate breakages

Planted one at a time in `crates/physsynth-core/src` and `crates/physsynth-analysis/src` (snapshot,
one change, release run of the new file, restore by copy, byte-compare). All seven caught.

| breakage | red | what catches it |
|---|---|---|
| assembly swaps `grain_x` / `grain_y` | 1 | **only** the sine residual — the one bar on a rectangle |
| assembly's cross factor 2 → 1 | 5 | residual, uniform twin, guard, isotropic default, the FFT |
| `grain_is_isotropic` forced false | 1 | only the isotropic-default bar, as designed |
| constructor feeds `grain_cross` where `grain_y` belongs | 3 | residual, guard, the FFT |
| constructor swaps `grain_x` / `grain_y` | 1 | **only** the sine residual |
| the floor's `<=` → `<` | 1 | the exactly-at-the-floor refusal |
| the **oracle** drops its factor 2 | 4 | the FFT and the uniform twin — the two bars where the frequency formula meets something it did not compute — plus convergence and the detune |

Two things here are the file's own warnings made measurable. A transposed grain is invisible on a
square, and every bar but one is on a square — so the one rectangular bar is load-bearing, and an
edit that squared its fixture would leave the transposition uncaught. And the energy ledger went red
for none of the seven.

The last row is why it was planted: several bars (the ledger's modal detector, the diagonal
blindness, the detune) compare the oracle with itself and pass a broken oracle. It has independent
witnesses, so it is not self-certifying.

### 25.4 The retirement rule, discharged

| retired | native bar, or verdict |
|---|---|
| `test_operator_eigenvalue_is_the_closed_form` (strong, wild) | `the_analytic_sine_is_an_exact_eigenvector_of_the_grained_operator`, both grains, bar widened (§25.2) |
| `test_the_isotropic_default_stays_on_the_untouched_squaring_path` | same name — **sharpened**: the reference `L @ L` is proved to be SciPy's by reproducing its gap exactly |
| `test_uniform_grain_collapses_to_an_isotropic_plate` | `a_uniform_grain_is_an_isotropic_plate_of_stiffness_kappa_sqrt_r` — **sharpened**: asserts the plate took the general path, which the Python's "through the *new* code path" only claimed |
| `test_continuum_oracle_and_its_isotropic_reduction` | `the_continuum_law_reduces_to_the_isotropic_one_and_the_grain_is_not_vacuous` (the analysis crate's `the_orthotropic_oracles_reduce_to_the_isotropic_ones` asserts the reduction at 1e-9; this is 1e-14) |
| `test_discrete_converges_to_the_continuum_at_second_order` | same name |
| `test_the_time_stepper_actually_rings_at_the_grained_frequency` | `the_time_stepper_rings_at_the_grained_frequency` |
| `test_energy_is_conserved_with_grain` (mu 0.5, 2, 8) | `a_grained_plate_conserves_its_energy`, all three |
| `test_lossy_grained_plate_is_passive_and_a_low_mode_decays_at_2sigma` | `a_lossy_grained_plate_is_passive_and_its_fundamental_decays_at_two_sigma` |
| `test_the_cross_term_guard_is_sharp_and_rejected_at_construction` | same name, plus LAPACK's two eigenvalues; each refusal asserts its variant as well as the word |
| `test_degenerate_grain_arguments_are_rejected` | `degenerate_grain_ratios_are_refused` |
| `test_grain_on_the_free_boundary_needs_the_split_and_says_so` | `a_grain_on_the_free_boundary_needs_the_split_and_says_so` |
| `test_an_isotropic_material_returns_exactly_no_grain` | `an_isotropic_material_returns_exactly_no_grain_and_an_areal_density_by_name`, with `plate.rs`'s `isotropic_material_comes_back_at_exactly_one` |
| `test_the_two_rival_cross_term_packagings_are_measurably_wrong` | same name |
| `test_spruce_is_not_a_stretched_isotropic_plate` | same name, with `plate.rs`'s `spruce_is_not_an_isotropic_plate_with_one_axis_stretched` |
| `test_the_energy_ledger_cannot_see_a_wrongly_wired_grain` | same name — **sharpened, and the Python control was a verdict**: described as "the same operator, reassociated", it built the *identical* plate and asserted nothing. The native control scales every modulus by 1.1, which lands the ratios on the same values in exact arithmetic and a different last bit in doubles (`cross` and `y` differ), and **asserts that it differs** so it cannot silently go vacuous again |
| `test_a_square_plates_diagonal_modes_are_blind_to_the_grain_running_the_wrong_way` | same name |
| `test_grain_makes_the_theta_damping_anisotropic_and_the_ledger_stays_green` | same name; the rates reproduce the Python docstring's 5.857 / 5.857 and 6.024 / 7.751 |
| `test_the_cross_term_detunes_selectively_without_reordering_anything` | same name — **sharpened**: asserts no exact frequency ties before comparing orderings, since `np.argsort` leaves a tie's order unspecified |
| `test_the_grain_is_in_the_partial_series_and_not_in_the_level` | same name; the five level ratios reproduce the Python's 0.811–1.117 |

Also deleted: `tests/helpers.py`'s `SPRUCE`, `make_orthotropic_plate` and `orthotropic_mode_freqs`,
which had no other caller. `docs/dev/orthotropic-plate-plan.md` now points at the native file.
`ops2d.rs`'s two orthotropic bars stay; they probe the builder on other fixtures.

### 25.5 Cost and counts

- **pytest 1,622 → 1,599**, measured against a worktree at the previous commit: the file's 22 cases
  plus its `test_xdist_groups` parametrization. (§24.6's 1,637 predates `687003c`, which removed the
  shard-partition tests; this batch accounts for 23 of the 38.)
- **Native: +19 in both profiles.** 8 s in release, **240 s in debug** on the dev machine — eight
  times §24's file, because several bars carry the Python's long trajectories (23,040 steps of a
  2,209-node plate for the FFT bar, three 20k-step conservation runs). The file is **not** on the
  `rust-debug` job's `release_only` list, per the default that a new core file lands in both
  profiles, and it asserts bit-identity in two places (the squaring path, the reproduced SciPy gap)
  — the kind the both-profiles rule protects. Whether to add it is the human's call. **On CI
  (`679a9ad`)** the file took 304 s in `rust-debug`, which went from ~3.5 min to 8 min 20 s; it
  is not the gate's critical path, which is still the release `rust` job at ~10 min. Everything
  passed on Linux, including the ledger control's last-bit difference. **Then added to
  `release_only` (the human's call, same day).** What the unoptimised pass would have protected is
  the two exact checks, and neither contains a transcendental or a constant exponent for LLVM to
  fold; both had passed unoptimised locally and on CI before the file left that pass.

### 25.6 What is next

61 physics files, 606 test functions (re-derived after this batch, with §24.1's exclusions). The
smallest are `test_convergence` (2), then `test_beam_stability`, `test_geometric_limits`,
`test_membrane_dispersion`, `test_modal` and `test_vk_modal` (3 each). A batch is better scoped by
model than by size, so the next step is to pick a model family and check its native bars first.

## 26. Phase C, carrying batch 3 — the plain plate

Done 2026-09-29; the human chose the model family. Model #5's three files —
`tests/test_plate_energy.py` (7 functions), `tests/test_plate_modal.py` (8) and
`tests/test_plate_stability.py` (6), 21 functions and 45 pytest cases — retire into one native
file, `crates/physsynth-core/tests/plate_kirchhoff.rs` (19 `#[test]`s; parametrized cases loop
inside a bar, and the two free/supported stability pairs share one bar each).

### 26.1 Three tests were about SciPy, and SciPy was transcribed rather than dropped

`test_plate_modal.py` held three tests whose subject was SciPy's sparse product itself: the Rust
port of `biharmonic_from_mask` sorts each row, where SciPy's `L @ L` had stored it in kernel order,
and a CSR matvec sums a row in stored order — so the port changed the shipped plate's last bits
(migration plan §26). With SciPy gone those tests have no referee unless one is written.

So SciPy's `csr_matmat` kernel is **transcribed** into the test (`scipy_csr_matmat`, ~30 lines:
accumulate over `A`'s row, thread each column onto a linked list on first touch, emit from the
head — reverse first-touch order — dropping exact-zero sums). And the transcription is **proved
faithful**, not assumed: before deletion, SciPy's actual product was recorded for all four grids
the Python used (`N` = 8, 12, 16, 24; NumPy 2.4.6, SciPy 1.17.1) as `nnz`, `Σ (i+1)·indices[i]`
and `Σ data[i]·(i+1)` summed left to right. The two weighted sums see the stored ORDER as well as
the values, and the transcription reproduces all four digests exactly — an independent stand-in,
certified by reproducing the retiring referee's output. **This certificate is weaker than §25.1's
in one respect, measured:** on these four grids `1/h²` is an exact integer (64, 144, 256, 576), so
every entry of `L @ L` is an exact integer and any accumulation order gives the same values. The
value half of the digest therefore cannot see accumulation ORDER; what certifies the kernel is the
stored column order (the index-weighted sum) and the sparsity (`nnz`, including SciPy's dropped
zeros). That is exactly what the three tests are about, so it suffices here; accumulation order is
certified by §25.1's non-dyadic grid, where SciPy's gap was reproduced to the bit.

| retired | native bar, or verdict |
|---|---|
| `test_the_canonical_sort_changed_an_order_and_not_a_value` | `the_canonical_sort_changed_an_order_and_not_a_value` — against SciPy's product reproduced to the digest: unsorted, and row-sorted it IS the shipped operator to the bit |
| `test_the_canonical_sort_left_the_shipped_plate_where_it_was` | same name — SciPy's kernel-order operator injected through `Csr::from_arrays_preserving_order` (the constructor that exists for this test) and asserted unsorted; the Python's seeded normal draw is replaced by a structureless splitmix64 field (§15's rule for broadband fields) |
| `test_the_free_plate_was_not_touched_at_all` | **verdict**: it asserted that SciPy returns the free Gram product sorted, but since phase A that builder has been Rust, whose `Csr` sorts every row it assembles — so it was already asserting Rust, not SciPy. Carried as a premise, `the_free_plates_stiffness_is_canonical_by_construction` |

### 26.2 Every margin measured; one is the acceptance bar itself

| bar | measured | bar |
|---|---|---|
| lossless drift, mu 0.5 / 2 / 8 (1 s, N = 32) | **3.5e-11** / 1.5e-12 / 4.0e-13 | 1e-10 |
| drift at mu 16 / mu 50 supported / mu 50 free | 2.8e-13 / 8.1e-14 / 2.3e-14 | 1e-10 / 1e-9 / 1e-9 |
| lossy, worst step / final energy | every step negative (max -2.0e-5·E0) / 0.54 E0 | ≤ 1e-10·E0 |
| low-mode decay vs 2σ | 0.20% | 2% |
| retained energy, (1,1) vs (8,8) | 0.028 vs 0.869 | high > low |
| E(2ρ) / E(ρ) − 2 | 0 exactly | 2e-12 (the Python's `np.isclose(rtol=1e-12)` kept its default `atol=1e-8`, finding (d), so its bar was effectively 1e-8) |
| B eigenvalues vs Λ² | 2.6e-13 | 1e-10 |
| convergence order | 2.02 | > 1.8 |
| one-cent block / horizons / inside / outside | 0.621 / 2 and 2 (window 6) / 0.621 / 1.406 | < 1 |
| low spectrum vs oracle | 9.6e-12 cents | 0.5 cents |
| FFT fundamental | 0.030 cents | 5 cents |
| kernel-order vs canonical, 2000 steps (N 12 / 16) | 1.0e-13 / 1.3e-13; drifts ≤ 3.7e-13 | 1e-11 / 1e-10 |

**One margin is thin, and it is not ours to move**: the lossless drift at `mu = 0.5` — 40,960 steps
— is 3.5e-11 against 1e-10, **2.9x**. That bar is the project's acceptance contract (CLAUDE.md:
not tightened, and by the same reasoning not loosened), the Python test ran the same Rust model at
the same margin, and it is recorded here rather than adjusted. It is the thinnest acceptance margin
any carrying batch has measured so far. It is **not** a random walk in the step count: the three
runs are 2,560, 10,240 and 40,960 steps, and the drift goes 4.0e-13 → 1.5e-12 → 3.5e-11, x3.75 and
then x23 for each x4 in steps, where a random walk would give x2. What drives it is unmeasured
(cancellation in `(u − u_prev)/k` as `k` shrinks is one candidate, and a shorter run would not
cure that). If a platform ever fails it, that is a question for the human, not a bar to move.

### 26.3 Eight deliberate breakages

Planted one at a time in `crates/physsynth-core/src` (snapshot, one change, release run, restore by
copy, byte-compare). All eight caught.

| breakage | red | what catches it |
|---|---|---|
| `B` scaled 0.1% high | 3 | the eigenvalue bar and both SciPy bars — **not** the FFT: 0.1% in `B` is 0.9 cents, inside its 5-cent bar |
| Laplacian diagonal −4 → −4.01 | 3 | eigenvalues, low spectrum, the SciPy digest |
| loss term dropped from the RHS | 2 | passivity, the 2σ decay |
| loss doubled in `A` | 1 | only the 2σ decay |
| kinetic energy halved | 6 | every conservation bar, passivity, decay |
| theta 1% off in the RHS | 7 | conservation, decay, the SciPy trajectory, the FFT |
| negative sigma accepted | 1 | the refusal table |
| `nu = 1/2` accepted | 1 | the refusal table |

### 26.4 The retirement rule, discharged

| retired | native bar, or verdict |
|---|---|
| `test_energy_conserved` (mu 0.5, 2, 8) | `a_lossless_plate_conserves_energy_across_mu` |
| `test_energy_conserved_with_timestep_explicit_could_not_run` | `energy_is_conserved_at_a_timestep_an_explicit_plate_could_not_run` |
| `test_energy_strictly_positive_when_lossless` | `lossless_energy_stays_strictly_positive` |
| `test_passivity_monotonic_decrease` | `a_lossy_plate_decreases_monotonically` — **sharpened**: also asserts the energy fell, since a plate that never moved is monotone too |
| `test_decay_rate_matches_2sigma_low_mode` | `a_single_low_mode_decays_at_two_sigma` |
| `test_higher_mode_underdamps_relative_to_lower` | `a_higher_mode_underdamps_relative_to_a_lower_one` |
| `test_energy_units_scale_with_density` | `energy_is_in_joules_and_scales_with_areal_density` |
| `test_biharmonic_eigenvalues_are_squared_laplacian` | `the_biharmonic_eigenvalues_are_the_squared_laplacian_ones` — ARPACK there, native shift-invert here; the truth is the closed form `Λ²`, so nothing of SciPy's needed recording. The builder-equals-plate half is structural in Rust and kept as a premise |
| `test_rectangle_continuum_convergence_order` | `the_discrete_law_converges_to_the_continuum_at_second_order` |
| `test_low_modes_within_one_cent` | `the_low_modes_are_within_one_cent_and_the_band_is_the_measured_horizon`, through the analysis crate's `horizon` module |
| `test_low_spectrum_via_eigsh_matches_oracle` | `the_low_spectrum_of_the_assembled_laplacian_matches_the_oracle` (truth: the closed form) |
| `test_fft_peak_at_fundamental` | `the_time_stepper_rings_at_the_discrete_fundamental` |
| the three SciPy-order tests | §26.1 |
| `test_no_nan_across_mu` + `test_free_no_nan_across_mu` | `no_nan_across_mu_on_either_boundary`, all ten |
| `test_explicit_unstable_config_runs_stably` + `test_free_explicit_unstable_config_runs_stably` | `a_timestep_200x_past_the_explicit_bound_runs_and_conserves_on_either_boundary` |
| `test_invalid_parameters_rejected` (15) | `non_physical_parameters_are_refused_at_construction` — **sharpened**: each case asserts its variant (and the offending value), where the Python accepted any `ValueError`. `boundary="clamped"` is a **shape** refusal (§14's rule): the native API cannot spell a clamped plate, so the case is the binding's `None` → `BadBoundary` |
| `test_free_boundary_is_accepted` | `the_free_boundary_constructs_with_every_node_a_free_unknown` |

Also removed from `tests/helpers.py`: `plate_low_eigenfrequencies` (no other caller) and
`plate_kwargs`, which existed only for the column-order test and is folded back into `make_plate`.
`docs/dev/plate-plan.md` now points at the native file.

### 26.5 Cost and counts

- **pytest 1,599 → 1,551**: 45 cases plus one `test_xdist_groups` parametrization per file.
- **Native +19.** 37–56 s in release on the dev machine (the conservation bars carry the Python's
  1-second runs: 40,960 steps at `mu = 0.5`). **1,072 s in debug** — all 19 pass there too. Left
  in both profiles it would have made `rust-debug` the gate's slowest job by ~10 minutes, so it is
  on `release_only` from the start (the human's call, asked before pushing). Its exact checks —
  SciPy's digests, the builder-equals-plate premise — are multiply-and-add with no transcendental
  or constant exponent, and they passed unoptimised before leaving that pass.

  **On CI (`dfe5e4e`), all green on Linux**, including the 2.9x drift margin. The file costs
  **54 s** in the release `rust` job, which IS the gate's critical path; the job's own wall-clock is
  too noisy to isolate that (10.3, 5.6 and 10.8 min over the last three runs, mostly build cache),
  so the file's own time is the number to quote. `rust-debug` stayed at ~3 min.

### 26.6 What is next

58 physics files, 585 test functions. The nearest families to what the three plate batches built
are the free plate (`test_free_plate_energy`, `test_free_plate_modal`) and the guitar plate
(`test_guitar_plate`), which reuse this file's fixtures; the human picks.

## 27. Phase C, carrying batch 4 — the free plate

Done 2026-09-29; the human chose the family. Model #5b's two files — `tests/test_free_plate_energy.py`
(7 functions) and `tests/test_free_plate_modal.py` (12), 19 functions and 21 pytest cases — retire
into `crates/physsynth-core/tests/plate_free.rs` (18 `#[test]`s).

### 27.1 SciPy's eigenvalues, recorded — and ARPACK was the least accurate solver again

Every eigenvalue the Python file used came from ARPACK. Per §24.3 the recorded referee is LAPACK's
dense solve of the same pencil, symmetrized as `W^{-1/2} K W^{-1/2}` (`dsyevr`), for every grid the
file used (N = 20, 32, 40, 64, 80; NumPy 2.4.6, SciPy 1.17.1): three rigid and five elastic
eigenvalues each, plus each grid's largest, `mu_max`. A dense solve's error is absolute, about
`eps · mu_max` — 2.2e-9 at N = 20, 1.5e-8 at 32, 3.6e-8 at 40, 2.4e-7 at 64, 5.8e-7 at 80 — so
every comparison is written **in units of that floor** (bar: 20). The native shift-invert sat at
most **0.38** floors from LAPACK anywhere.

ARPACK, as the Python called it (shift `-1e-3 (13/a²)²`), was not so close. At N = 40 its fourth
elastic eigenvalue came back `1208.1428639773883` against LAPACK's `1208.14286648793` — 2.5e-6 off,
**~70 LAPACK floors** — and further up the recorded list it was 1.4e-5 off (`3700.50185267` against
`3700.50186634`). No Python bar was tight enough to notice (the nearest was Leissa's 0.6%), and the
Python never used those modes at N = 40, but it is §24.3's finding a second time on a second pencil:
record with the dense solver, not with the one the test called.

### 27.2 Every margin measured

| bar | measured | bar |
|---|---|---|
| lossless drift, mu 0.5 / 2 / 8 (1 s, N = 32) | **2.3e-11** / 8.8e-13 / 4.8e-13 | 1e-10 |
| drift at mu 16 | 2.8e-13 | 1e-10 |
| lossy, worst step / final | every step negative / 0.54 E0 | ≤ 1e-10·E0 |
| saddle decay vs 2σ | 0.33% | 3% |
| retained, fundamental vs 13th elastic | 0.027 vs 0.053 | high > low |
| E(2ρ)/E(ρ) − 2 | 0 exactly | 2e-12 |
| symmetry | 0 exactly | 1e-12 |
| production vs dense per-node assembly | 2.4e-16 **relative** | 1e-12 absolute → **1e-14 relative** |
| nullspace {1, x, y} / saddle xy | 5e-19 – 4e-18 / 1.1e-5 | 1e-12 / > 1e-9 and 1e6x contrast |
| xy energy ∝ (1 − nu) | 5.7e-15 | 1e-12 |
| bending diagonal = beam | 5.7e-14 | 1e-12 |
| self-convergence orders | 2.15, 2.36, 2.26, 2.34 | > 1.8 / > 1.6 |
| Leissa at N = 32 / 64 | 0.25% / 0.026% | 0.6%, decreasing |
| saddle corners / centre | ±1.000 / 2.5e-12 | ±0.5 / 0.1 |
| FFT fundamental | 0.080 cents | 8 cents |
| supported operators through the generalized map | 8.6e-13 | 1e-9 |

Two notes. The direct-assembly bar was the Python's **absolute** 1e-12 on entries of ~2e3, met with
2.2x to spare; it is one ulp of rounding measured on the wrong scale, so it is relative now (42x).
And the lossless drift at `mu = 0.5`, 2.3e-11, is 4.4x under the acceptance bar — thin, as §26.2's
3.5e-11 on the supported plate was, and recorded rather than moved for the same reason.

### 27.3 Seven deliberate breakages — and one the carried bar could not see

Planted one at a time in `crates/physsynth-core/src` (snapshot, one change, release run, restore by
copy, byte-compare).

| breakage | red | what catches it |
|---|---|---|
| torsion factor 4 → 2 | 5 | direct assembly, zero modes, convergence, Leissa, the FFT |
| coupling not symmetrized (`2·cross`, not `cross + crossᵀ`) | 12 | symmetry, every energy bar, the spectrum — the only one the ledger sees |
| coupling 10% low | 5 | direct assembly, zero modes, convergence, Leissa, the FFT |
| the plate's own curvature `-2 → -2.01` | 8 | the bending diagonal, direct assembly, the nullspace, the (1 − nu) scaling, the spectrum |
| loss term dropped from the free RHS | 2 | passivity, the 2σ decay |
| free kinetic energy halved | 4 | conservation (both), passivity, decay |
| `collocated_d2_1d`'s `-2 → -2.01` | — | see below |

The last row is why the bending-diagonal bar was **rewritten rather than carried**. The Python
rebuilt the plate's bending diagonal from `_collocated_d2_1d` and compared it with the free beam.
Transcribed literally, a planted error in `collocated_d2_1d` turned that bar — and only that bar —
red, and no plate bar noticed: the Rust plate assembles its curvatures from the mask
(`free_plate_stiffness_from_mask`) and **never calls `collocated_d2_1d`**. The carried bar would
have certified a building block the plate does not use. The native bar reads the diagonal off the
real builder instead (`free_plate_stiffness` with coupling and torsion set to zero, which leaves
exactly the two bending terms), and the plate's-own-curvature row above is the proof that it now
watches the plate. `collocated_d2_1d` keeps its own bars in `ops2d.rs`.

### 27.4 The retirement rule, discharged

| retired | native bar, or verdict |
|---|---|
| `test_energy_conserved` (mu 0.5, 2, 8) | `a_lossless_free_plate_conserves_energy_across_mu` |
| `test_energy_conserved_with_timestep_explicit_could_not_run` | `energy_is_conserved_at_a_timestep_an_explicit_plate_could_not_run` |
| `test_energy_strictly_positive_when_lossless` | `lossless_energy_stays_strictly_positive` |
| `test_passivity_monotonic_decrease` | `a_lossy_free_plate_decreases_monotonically` — also asserts the energy fell |
| `test_decay_rate_matches_2sigma_low_mode` | `the_saddle_fundamental_decays_at_two_sigma` |
| `test_higher_mode_underdamps_relative_to_lower` | `a_higher_mode_underdamps_relative_to_the_fundamental` |
| `test_energy_units_scale_with_density` | `energy_is_in_joules_and_scales_with_areal_density` |
| `test_operator_symmetric` | `the_energy_first_operator_is_symmetric` |
| `test_matches_direct_assembly` | `the_assembly_matches_a_dense_per_node_build` — bar made relative (§27.2), and renamed: the Python called the production build a Kronecker one, the Rust builder assembles row by row from the mask; the corner-weight clause's `np.allclose` defaults (~4e-6 relative) tightened to 1e-14 |
| `test_rigid_body_nullspace` | `the_rigid_body_nullspace_is_exact_and_the_saddle_is_not_in_it` |
| `test_xy_energy_scales_with_one_minus_nu` | `the_saddles_energy_scales_exactly_with_one_minus_nu` (the Python's `np.allclose` kept `atol=1e-8`, finding (d); this is the relative claim alone) |
| `test_exactly_three_zero_modes` | `exactly_three_modes_are_rigid`, plus LAPACK |
| `test_bending_diagonal_is_beam_operator` | `the_plates_bending_diagonal_is_the_free_beam_operator_along_each_axis` — **rewritten against the plate's own assembly** (§27.3) |
| `test_self_convergence_order_h2` | `the_low_eigenvalues_self_converge_at_second_order`, plus LAPACK at all three grids |
| `test_leissa_ffff_square_anchor` | `the_low_modes_match_leissas_ffff_square_and_improve_with_refinement`, plus LAPACK |
| `test_fundamental_is_saddle` | `the_fundamental_is_the_saddle_not_a_bulge` |
| `test_fft_rings_at_fundamental` | `the_time_stepper_rings_at_the_discrete_fundamental`, plus LAPACK |
| `test_resonator_uses_operator_helper` | already carried: §24's `the_default_free_plate_is_the_isotropic_operator_and_its_split_is_nus` asserts the resonator's `K` and `W` are the helper's **bit for bit** on the same `N = 16` grid, stricter than this test's 1e-12 |
| `test_ss_operators_through_generalized_map_match_model5` | `the_generalized_map_on_the_supported_operators_is_model_5` |

No helper was orphaned: `make_free_plate`, `free_plate_low_eigenfrequencies` and `arpack_v0` all
have other callers. `docs/dev/plate-free-edge-plan.md` now points at the native file.

### 27.5 Cost and counts

- **pytest 1,551 → 1,528**: 21 cases plus one `test_xdist_groups` parametrization per file.
- **Native +18.** 9–18 s in release on the dev machine, **213 s in debug** (all 18 pass there).
  It would not have become the gate's slowest job, but it is on `release_only` for consistency with
  §25's file in the same position (the human's call, asked before pushing).
  **On CI (`f6115a3`)**, all green on Linux: the file took **6.5 s** in the release `rust` job, and
  `rust-debug` stayed at ~2.6 min.

### 27.6 What is next

56 physics files, 566 test functions. The guitar plate (`test_guitar_plate`) is the last plate
family and the nearest to these four batches; the human picks.

## 28. Phase C, carrying batch 5 — the curved-outline plate

Done 2026-09-29; the human chose the family. `tests/test_guitar_plate.py` (21 functions, **113**
pytest cases — one parametrization alone is 2 plates × 4 outlines × 10 grids) retires into
`crates/physsynth-core/tests/plate_outline.rs` (18 `#[test]`s). With it **every plate family's
Python suite is gone** (§24–§28).

### 28.1 Three outside referees, frozen into a reference file

This file had more independent referees than any before it, and they were frozen — not merely
summarised — into `crates/physsynth-core/tests/reference/guitar_plate.json` (37 kB; wheel reinstalled;
NumPy 2.4.6, SciPy 1.17.1), the first frozen reference in the core crate:

- **The outline's pre-2026-08-28 NumPy spelling.** The shipped geometry was validated against the
  vectorised `sin`/`cos` profile; on 2026-08-28 it moved to the scalar libm so the port could match
  it to the bit, and the Python kept the old expression to pin that no node moved. Frozen as a digest
  (`count`, `Σ(i+1)`, `Σ(i+1)²` over the live flat indices) of all 80 old-spelling masks, before and
  after the prune — the Rust masks reproduce every one. For the degenerate lens, where mask equality
  would be a claim about a CPU's `sin`, NumPy's half-width values themselves are frozen and the Rust
  profile matches them to **0 ulps** (bar: the Python's 4, kept for another platform's libm).
- **SciPy's Bessel functions** (Cephes/AMOS), which evaluated each derived circular-plate root's own
  Rayleigh quotient. Natively the quotient is recomputed with the analysis crate's `bessel` module —
  a second, independent implementation — and agrees with SciPy's recorded value to **≤ 1.4e-14** on
  all eight roots — a comparison the Python never made, now asserted at 1e-12.
  Against `lam⁴` itself the quotient is within 1.8e-9 (the 40,000-point quadrature); the Python's bar
  there was 3e-3.
- **LAPACK's dense eigenvalues** of the staircased disks at N = 32 and 64 and the shipped circle at
  N = 33, with each `mu_max` (§24.3). The native solver sits 0.26–0.44 dense floors away (bar 20).

**Recording it found that the core crate could not read it exactly.** The first run failed the lens
bar at 31 ulps: `serde_json` without its `float_roundtrip` feature parses a decimal to the
nearest-but-one double in rare cases, and one ulp of a recorded `t` near `t = 1` — where `sin(πt)` is
small and steep — is 31 ulps of the profile. The viewer crate had already learned this (its
`Cargo.toml` calls the feature load-bearing); the core crate's test-only `serde_json` had not, because
nothing in it had read a float from JSON before. The feature is now on there too, with the reason.
**Any frozen float read by a crate without `float_roundtrip` is a one-ulp question.**

### 28.2 The native eigensolver cannot run at the Python's shift

The Python asked ARPACK for the eigenvalues nearest `-1e-8`, a hair below the plate's three rigid
modes. The native shift-invert Krylov solver hits its 300-iteration cap there — measured on every
disk (N = 32, 64, 128; 6 and 11 pairs) and every circle (N = 32, 33, 64, 128) this file solves. At
`-1e-3 mu_1` — `free_plate_low_eigenfrequencies`' own convention, `-0.03` for a unit disk — it
converges everywhere, the rigid modes come out clean to 1e-12…1e-9 of the first elastic one (bar
1e-6), and it matches LAPACK. The shift is an instrument setting, so the bars use it; the
solver's behaviour that close to a singular pencil is recorded here rather than fixed, because **no
shipped path asks for it** — checked, every `eigsh_shift_invert` caller in `crates/*/src` is in the
viewer: the free plate at `-1e-3 (13/(Lx·Ly))²` (the same convention), the closed bore at
`-1e-3 (π c/L)²` below its zero mode, and the supported plate, membrane and open bore at `0.0` on
pencils with no nullspace. Remaining Python files that call ARPACK at a tiny shift next to a
nullspace will need the same change when they are carried.

### 28.3 Every margin measured

| bar | measured | bar |
|---|---|---|
| the 80 outline digests, raw and pruned | all equal | equal |
| lens vs NumPy | 0 ulps | 4 ulps |
| pruned-node depth, N = 20…80 | 0.750, 0.733, 0.712, 0.704, 0.703 h (= Python's) | (0.6, 0.85) h |
| crate vs SciPy Bessel quotient | ≤ 1.4e-14 | 1e-12 |
| quotient vs lam⁴ | ≤ 1.8e-9 | 3e-3 |
| disk vs oracle, N = 32 / 64 / 128 | 8.5% / 4.0% / 2.0%; rate 2.12, 2.00 | < 12% / < 3%; (1.25, 2.6) |
| circle, abs(error + deficit), N = 32 / 33 / 64 / 128 | 4.4e-3 / 6.2e-3 / 2.3e-3 / 1.0e-3 | 0.012 |
| degenerate pair split, N = 32 / 64 / 128 | 1.0% / 0.52% / 0.013% (= Python's) | 2% / 1.2% / 0.6% |
| eig(B) vs eig(−L)² on the guitar | ≤ 6.4e-13 | 1e-8 |
| drift, guitar / circle | 5.0e-15 / 2.2e-14 | 1e-10 |
| outline area vs `guitar_area` | 0 | 1e-9 |
| rectangle's area deficit | 2.3e-15 | 1e-14 → **1e-12** |

The last row: the Python's absolute 1e-14 had 4.3x and sat **below** the worst-case rounding of the
336-weight sum it checks (21 × 16 nodes; `n·eps ≈ 7.5e-14`); the bar is ~13x that bound.

### 28.4 Seven deliberate breakages — two caught only elsewhere, on purpose

| breakage | red here | notes |
|---|---|---|
| outline profile mis-parenthesised (`4π t − 0.5`) | 2 | the digests and the rim depth — the slip the NumPy pin exists for |
| prune keeps only nodes touching 2+ cells | 12 | nearly everything |
| area weight `¼ → 0.3` per cell | 6 | the trapezoid bar, the deficits, the whole disk anchor |
| connectivity refusal off | 1 | the pinch bar |
| rim depth `min → max` | 2 | the rim bar and the pinch |
| disk rim made inclusive (`<` → `<=`) | **0** | the only on-rim nodes are one-node spikes the prune removes; `ops2d.rs`'s `a_node_on_the_rim_is_dead` catches it |
| twist `(1/h)(1/h)` → `1/(h·h)` | **0** | the last bit of the operator; `ops2d.rs`'s `the_twist_coefficient_is_two_reciprocals_and_not_one` pins it on every platform, and six viewer freezes (`plate`, `vk`, `vkroom`, `browser5b/5d/6b`) catch it too — but those compare exactly only on the Windows CI job |

The last row is the retired file's headline finding (its rectangle bar): the masked and
Kronecker assemblies differed on exactly one grid until the twist was spelled as two reciprocals.
With the Kronecker assembly gone that bar compares a computation with itself and is a **verdict**
(finding #78) — and the planted breakage shows the spelling is still pinned, just not here.

### 28.5 The retirement rule, discharged

| retired | native bar, or verdict |
|---|---|
| `test_the_scalar_libm_spelling_moved_no_node_of_any_shipped_outline` (80) | same name, against the frozen digests (§28.1) |
| `test_the_degenerate_lens_gets_the_weaker_claim_it_can_actually_support` (10) | `the_degenerate_lens_agrees_with_its_old_spelling_to_a_few_ulps`, against the frozen values |
| `test_masked_assembly_reproduces_the_rectangle_bit_for_bit` | **verdict** (§28.4): `free_plate_stiffness` delegates to the masked builder; kept as the premise bar `the_rectangle_builder_is_the_masked_builder_on_a_full_mask` over the same 84 cases |
| `test_area_weight_is_the_trapezoidal_rule_restated` | `the_area_weight_is_the_trapezoidal_rule_restated` |
| `test_a_curved_outline_produces_massless_nodes_and_the_prune_removes_them` | `a_curved_outline_makes_massless_nodes_and_the_prune_removes_them` |
| `test_prune_is_idempotent_and_reaches_a_fixed_point` | same name |
| `test_every_pruned_node_lies_at_the_rim` (5) | same name, all five grids |
| `test_a_pinched_outline_is_refused_rather_than_silently_two_plates` | same name — asserts the `Disconnected` variant as well as the words |
| `test_derived_frequency_equation_admits_the_rigid_body_modes` | `the_derived_frequency_equation_admits_the_rigid_body_modes` |
| `test_saddle_bound_brackets_the_derived_fundamental` | already carried, stricter: the analysis crate's `the_free_disk_reproduces_its_derived_lambdas_and_respects_its_own_bound` asserts the bound, and the overshoot at 8.18 ± 0.05% against this test's (5%, 12%) |
| `test_every_derived_root_returns_lambda_to_the_fourth_in_the_plate_energy` | same name — **sharpened** with the SciPy comparison (§28.1) |
| `test_staircased_disk_matches_the_derived_oracle_and_converges` | `a_staircased_disk_matches_the_derived_oracle_and_converges_at_first_order`, plus LAPACK |
| `test_the_shipped_circle_path_matches_the_derived_oracle` | same name, plus LAPACK at N = 33 |
| `test_the_degenerate_pairs_split_and_the_exact_answer_is_zero` | same name. The Python's `_fingerprint` diagnostic (SHA-256 of the mask and `K`, printed on failure) is not carried: it asserted nothing |
| `test_a_supported_curved_plate_is_the_membrane_squared_and_therefore_says_nothing` | same name — the refusal asserts `CurvedSupported(Guitar)` as well as the words |
| `test_lossless_outline_plate_conserves_energy` (guitar, circle) | `a_lossless_outline_plate_conserves_energy`, both |
| `test_lossy_outline_plate_is_passive` (guitar, circle) | `a_lossy_outline_plate_is_passive`, both — also asserts the energy fell |
| `test_the_area_deficit_is_reported_and_shrinks_under_refinement` | same name |
| `test_a_rectangle_still_prunes_nothing_and_carries_its_whole_area` | same name, bar re-derived (§28.3) |

No helper was orphaned (`arpack_v0` has other callers). `docs/dev/guitar-plate-plan.md` now points
at the native file.

### 28.6 Cost and counts

- **pytest 1,528 → 1,414**: 113 cases plus one `test_xdist_groups` parametrization.
- **Native +18**, 2.7 s in release and **45 s in debug** on the dev machine — in both CI profiles by
  the default, like §24's 30 s file.
- **On CI (run 36538027804, `b7b06aa`, all green)**: 5.3 s in the release job and **75.7 s in the
  debug job**, which went from ~2.6 to ~4.6 minutes. That job still ends ~7 minutes before the
  release job, so the run's wall clock did not move; if the debug job ever becomes the long pole,
  this file is the first candidate for `release_only`. The lens bar passed on Linux, so the
  `float_roundtrip` fix holds across platforms.

### 28.7 What is next

55 physics files, 545 test functions. With the plates done, the natural next families are the
membrane (`test_membrane_{energy,modal,stability,dispersion}`) or the beam
(`test_beam_{energy,modal,stability}`); the human picks.

## 29. Phase C, carrying batch 6 — the membrane

Done 2026-09-29; the human chose the membrane on the recommendation (the plates' method carries
straight over, and it closes the 2-D grid models). Four files — `tests/test_membrane_{energy,modal,
stability,dispersion}.py`, **19 functions, 42 pytest cases** — retire into
`crates/physsynth-core/tests/membrane_harness.rs` (14 `#[test]`s) and four bars appended to
`crates/physsynth-analysis/tests/modal.rs`. The dispersion file touched no model — its only
subject was `discrete_membrane_eigenfrequency` and a symbol written inside the test — so its bars
live with the oracle, not the core.

`membrane.rs` already existed and stays: it is the port's own floor at a drumhead's `T` and `rho`.
The new file is the acceptance contract at `make_membrane`'s parameters (`c = 200 m/s`, unit
square, disk of radius 0.5, `fs = c/(lambda h)`). An existing bar counted as carried only where it
asserted the same claim at the same or a stricter bar; none did exactly, so every Python test has
a bar in the new file at its own parameters.

### 29.1 Two outside referees, frozen

Everything else the four files touched — `raised_cosine_2d`, `simulate`, `measure_partials_near`,
every `modal.*` oracle — already ran through the binding (§25's rule). What did not, recorded into
`crates/physsynth-core/tests/reference/membrane.json` (wheel reinstalled; NumPy 2.4.6, SciPy 1.17.1):

- **LAPACK's dense eigenvalues of `-L`**, the lowest 8 on the staircased disk at N = 32, 64, 128
  and the lowest 6 on the unit square at N = 24, with the mask digest (`count`, `Σ(i+1)`,
  `Σ(i+1)²`) and `h` of each grid. N = 128 is 12,849 unknowns, 1.3 GB dense and 62 s in `evr`.
  **ARPACK agreed with LAPACK to ≤ 4.8e-16 of `8/h²` everywhere** — unlike §24.3 and §28.2, the
  Python's shift of 0 sits on a pencil with no nullspace, so the dense rule cost nothing here. ARPACK's
  values are in the file too, for the comparison.
- **SciPy's Bessel zeros** (Cephes) for the eight lowest circular modes with their `(m, n)` and
  degeneracy. `crates/physsynth-analysis/tests/bessel.rs` pinned the crate's zeros on their own terms
  (vanishing, interlacing, three published digits), never against SciPy; now all eight are, at 8 eps.

The disk keeps its **exact degenerate pairs** (4-fold symmetry: `m = 1` and `m = 3` stay paired,
`m = 2` splits — visible in the record), the square has two more; the native single-start Krylov
solve returned every copy of every pair.

### 29.2 Every margin measured

| bar | measured | bar |
|---|---|---|
| native eigenvalues vs LAPACK, 30 values on 4 grids | ≤ 2.1 eps·8/h² (disk N = 64, 7th) | 20 |
| crate's Bessel zeros vs SciPy (through the frequency round trip) | ≤ 1.9 eps | 8 eps |
| lossless drift, 2 domains × λ ∈ {0.7071, 0.6, 0.4}, 1 s | ≤ 8.5e-15 | 1e-10 |
| passivity, worst step / E⁰ (σ = 8) | −4.2e-16 (never rose) | ≤ 1e-12 |
| decay rate vs `2σ` | 9.6e-5 relative | 2% |
| `E(2ρ)/E(ρ) − 2` | 0 | 2e-12 |
| square eigenvalues vs closed form | 1.6e-15 | 1e-10 |
| square continuum order, N = 16…128 | 2.0003 | > 1.8 |
| disk Bessel order, N = 32 / 64 / 128 | 0.818, 0.955 | (0.5, 1.5) |
| disk fundamental at N = 128 | **8.75 cents** | 12 |
| disk low 8 vs sorted Bessel, N = 128 | **9.11 cents** | 20 |
| FFT peak vs discrete fundamental | 0.0067 cents | 5 |
| λ reported at the ceiling | exact | 1e-9 |
| isotropy at κh = 0.02 | 8.3e-6 | 1e-3 |
| axial–diagonal gap at κh = 0.6π, 4 λ | ≥ 0.070 c | > 1e-3 c |
| diagonal speed at the ceiling, `v/c − 1` | **+2.2e-16** | ≤ 1e-9 |

The two bold cents rows are the Python's own staircase bars at 1.4x and 2.2x; they measure a
geometry error (the "not a horizon" note is carried verbatim) and were not moved. The last row is
the finding: **at `λ = 1/√2` the 5-point scheme is exact along the diagonal** — the 2-D echo of the
1-D string at `λ = 1` — so the "subluminal" bar's 1e-9 slack is what lets an exact identity through
round-off, and it is now commented as such so nobody tightens it to zero.

### 29.3 Nine deliberate breakages

| breakage | red | caught by |
|---|---|---|
| Laplacian drops one neighbour (`ops2d`) | 11 | energy, eigenvalues, Bessel, FFT — here and in `membrane.rs` |
| disk rim inclusive (`<` → `<=`) | 2 | **the mask digest**: 4 extra rim nodes, `[797, …]` vs `[793, …]` |
| energy drops `rho` | 1 | only the density-ratio bar — every other energy check is a ratio to itself |
| loss sign in the denominator | 3 | passivity (both files) and the `2σ` rate |
| CFL ceiling `1.1/√2` | 1 | the 5%-past bar — **after** a fix, below |
| start-up drops the `½` | 1 | **only `membrane.rs`'s eigenmode bar** — pinned elsewhere, as §28.4 |
| rectangle oracle's `y` factor uses `Nx` | 1 | **only the new symbol bar** in `modal.rs` — below |
| Bessel oracle drops the `2` | 3 | both disk bars and the analysis crate's first-zero bar |
| `discrete_membrane_eigenfrequency` `½` → `0.45` | 6 | across both crates |

Two rows needed the bars changed first:

- **The CFL bars first read the ceiling from the model** (`membrane::lambda_max`). A planted
  `1.1/√2` would then pass both: `1.05×` the moved ceiling is still past it, and the moved ceiling
  is still accepted. The Python wrote its own `LAMBDA_MAX`; the native file now does too. A bar
  about a constant must not import the constant.
- **A rectangle oracle that used `Nx` for both axes is invisible on a square**, and every rectangle
  bar here runs on the unit square, as the Python's did. The new
  `the_plane_wave_symbol_is_the_rectangle_eigenvalue_at_a_standing_wave` evaluates the dispersion
  bars' symbol at `m π/Lx, n π/Ly` on a 24 × 18 grid and requires it to equal
  `rectangular_discrete_eigenvalues` — which catches the defect and also ties the dispersion bars'
  hand-written symbol to the operator the core bars check against LAPACK.

A tenth, found in review after the batch commit: **the passivity bar folded its worst step with
`f64::max`, which drops a NaN** (it returns the other operand), so a lossy run whose energy went NaN
after step 0 folded to `-inf` and passed — where the Python's `np.all(steps <= 1e-12 E⁰)` fails.
Planted (the step writes NaN whenever `sigma > 0`): the committed bar **passed**, the fixed one
fails. It now asserts every step, and every worst-case figure in the file goes through a
NaN-propagating `nan_max` — the trap `engine.rs`'s `energy_drift` already documents. None of the
nine plants reached it: each broke a *value*, and a NaN is not a value. **A carried "worst of"
must propagate NaN, or it is weaker than the `np.max` / `np.all` it replaces.**

### 29.4 The retirement rule, discharged

| retired | native bar |
|---|---|
| `test_energy_conserved` (6) | `lossless_energy_is_flat_and_positive_on_both_domains_at_every_courant_number` |
| `test_circle_conserves_like_rectangle` | the same runs (its two are two of the six) |
| `test_energy_strictly_positive_when_lossless` | the same runs — its circle at 0.6 for 0.5 s is a prefix of the 1 s run |
| `test_passivity_monotonic_decrease` | `loss_makes_the_energy_fall_at_every_step`, its parameters and slack |
| `test_decay_rate_matches_2sigma` | `a_uniformly_damped_membrane_loses_energy_at_two_sigma` |
| `test_energy_units_scale_with_density` | `energy_is_in_joules_and_linear_in_the_areal_density` |
| `test_rectangle_eigenvalues_match_closed_form` | `the_square_eigenvalues_are_the_closed_form_and_lapacks` — plus LAPACK |
| `test_rectangle_continuum_convergence_order` | `the_square_continuum_error_converges_at_second_order` |
| `test_circle_bessel_convergence_rate` | `the_disk_fundamental_converges_to_bessel_at_the_staircase_rate` — plus LAPACK at all three N |
| `test_circle_low_spectrum_tracks_bessel` | `the_disk_low_spectrum_tracks_the_sorted_bessel_series` — plus LAPACK and SciPy's zeros |
| `test_circle_fft_peak_at_fundamental` | `a_struck_disk_rings_at_its_discrete_fundamental` |
| `test_no_nan_across_valid_lambda` (10) | `nothing_blows_up_anywhere_in_the_admissible_courant_range` |
| `test_lambda_above_cfl_rejected_at_construction` | `a_courant_number_just_past_the_ceiling_is_refused_at_construction` — also the variant |
| `test_lambda_at_cfl_ceiling_accepted` | `the_ceiling_itself_is_accepted_and_reported` |
| `test_invalid_parameters_rejected` (7) | `non_physical_or_missing_parameters_are_refused` — each case's variant, not just "an error" |
| `test_rectangle_requires_sides` | `a_rectangle_without_sides_says_rectangle` |
| `test_isotropic_in_continuum_limit` | `modal.rs::the_membrane_is_isotropic_only_in_the_continuum_limit` |
| `test_anisotropic_at_every_lambda` (4) | `modal.rs::the_membrane_is_anisotropic_at_every_admissible_courant_number` |
| `test_both_directions_subluminal` | `modal.rs::short_membrane_waves_are_subluminal_in_both_directions` |

One helper was orphaned and deleted: `membrane_low_eigenfrequencies` in `tests/helpers.py`.
`make_membrane` keeps three callers (`test_mallet_energy`, `test_mallet_signature`,
`test_resolution_horizon`). `docs/dev/membrane-plan.md` now points at the native files, and the
hand-picked-band audit row in `docs/dev/resolution-horizon-plan.md` names the native bar.

### 29.5 Cost and counts

- **pytest 1,414 → 1,368**: 42 cases plus four `test_xdist_groups` parametrizations (one per file).
- **Native +18** (14 core, 4 analysis). The core file is 3.4 s in release and **59 s in debug** on
  the dev machine — most of it the six 1-second conservation runs (up to 24,000 steps each). It stays
  in both CI profiles by the default.
- **On CI (run 36546609975, `1be106d`, all green)**: 3.9 s in the release job and **37.4 s in the
  debug job**, which took 3.0 minutes against the release job's 11.2 — nowhere near the long pole.
  First run on Linux (glibc's `sin`/`cos`/`acos`): every bar passed, including the Bessel-zero bar
  whose 4x headroom was the thinnest cross-platform margin in the file.

### 29.6 What is next

51 physics files, 526 test functions. The beam (`test_beam_{energy,modal,stability}`, 16
functions) is the other family §28.7 named; after it the string families; the human picks.

## 30. Phase C, carrying batch 7 — the free-free beam

Done 2026-09-29; the human named the beam as the natural next batch. Three files —
`tests/test_beam_{energy,modal,stability}.py`, **16 functions, 32 pytest cases** — retire into
`crates/physsynth-core/tests/beam.rs`, which goes from 17 `#[test]`s to 24.

Unlike §29 there is no new harness file. `beam.rs` was already written at `make_beam`'s own
parameters (`L = 1`, `rho = 0.005`, `kappa = 20`, `theta = 0.28`, `fs = kappa / (mu h^2)`) and
carried about half the claims at the same or a stricter bar, so it was extended rather than
duplicated. Where an existing bar was at the Python's parameters it was counted as carried; where
it was shorter or on another grid it was brought to the Python's. `theta` was checked first:
`make_beam` took its default from `string_stiff`, not from the beam module, and both are 0.28.

One core source change: `impl Resonator for FreeBeam` in `engine.rs`, so the carried bars run
through the same `simulate` and NaN-propagating `energy_drift` the Python did. The binding does not
use it (it keeps the beam's state in NumPy arrays, per Phase 0).

### 30.1 One outside referee, frozen — and a stronger truth than LAPACK

The only SciPy numbers the three files touched were ARPACK's eigenpairs of `K phi = mu W phi`
(`beam_low_eigenfrequencies`, `_elastic_eigenvector`). Everything else ran through the binding.
Recorded into `crates/physsynth-core/tests/reference/beam.json` (wheel reinstalled; NumPy 2.4.6,
SciPy 1.17.1) at all seven grids the suite solved on (N = 48, 50, 64, 100, 120, 200, 400), with
`h`, the node count and `K`'s stored-entry count so a fixture that built a different beam fails
loudly:

- **LAPACK's dense generalized `eigh`** and its largest eigenvalue, for the `eps mu_max` unit;
- **ARPACK's values at the Python's own settings** (shift `-1e-3 mu_1`, `v0 = arpack_v0`);
- **the 50-digit Rayleigh quotient of LAPACK's eigenvector** (mpmath), rounded once. It is exact
  for that double vector, and its error as an eigenvalue is second order in the vector's error. So
  it is a better truth than either solver, and it is what the native bars are measured against.

Against it, **ARPACK was the least accurate solver in the room for the third time** (§24.3,
§27.1): **22.6 `eps mu_max` on the second elastic mode at N = 120**, and 7.4 at N = 100. LAPACK was
within 1.8 everywhere. The bar asserts the finding, so a re-recorded file that lost it would fail.

Also recorded: **LAPACK's eigenvalues-only path is a different algorithm** from the path that also
computes eigenvectors. The rigid pair at N = 200 moved from −1.4e-6 to −6.3e-6 between the two. The
record says which path produced it.

### 30.2 The native Krylov solver's error has two terms, and neither alone is a bar

`eigs::eigsh_shift_invert` was held to §25's `20 eps mu_max` and failed at N = 200: **172 `eps
mu_max` at the top of the 24-mode window**. That is 3e-11 relative, which is what its stopping rule
promises. It accepts a Ritz pair when the out-of-basis residual is below `RESIDUAL_TOL = 1e-10` of
`theta`, and that is a *relative* promise. A relative bar then failed the other way. At N = 400,
`mu_1 / mu_max` is about 1e-9, so the fundamental's 0.04 `eps mu_max` reads as 7e-9 relative.

The bar is therefore **`20 eps mu_max + 1e-9 |mu|` per mode**: a floor every solve through `K`
shares, plus the solver's relative tolerance, written as a literal so a loosened `RESIDUAL_TOL`
cannot move it (§29.3). The rigid pair has no relative scale and gets the first term alone. Worst
use of the allowance: **0.072** (N = 200). The file's own inverse iteration, which supplies the
eigenvectors the single-mode bars start from, is held to the plain dense unit on exactly those modes.
Its worst is **0.034 `eps mu_max`**.

### 30.3 Every margin measured

| bar | measured | bar |
|---|---|---|
| lossless drift, N = 64, `mu` ∈ {0.5, 2, 8, 16, 50}, the longer of 1 s and 8,000 steps | ≤ 5.5e-12 (`mu` = 2) | 1e-10 |
| passivity, worst step / E⁰, all 40,960 steps of 1 s (σ = 8) | **−1.7e-6** (every step fell) | ≤ 1e-10 |
| passivity, retained energy after 1 s | 0.141 (the single-mode rate would leave 1.1e-7) | < 0.5, and > 10× that |
| low-mode decay vs `2σ` | 1.7e-3 relative | 2% |
| high/low retained (underdamping caveat) | 0.164 vs 0.027 | high > low |
| `E(2ρ)/E(ρ) − 2` | 0 | 1e-12 |
| `K 1`, `K x` / `K x²` relative residual | 0, 4.0e-18 / 1.06e-6 | < 1e-12 / > 1e-9 |
| closed form, modes 1 / 5 at N = 200 | 0.18 / **1.45 cents** | 0.5 / 2 |
| pitch horizon at 0.5 / 2 cents | 2 / 6, monotone | ≥ 1 / ≥ 4, < 24 |
| rigid pair from the solver, / (1e-6 μ₁) | 7.0e-4 | < 1 |
| convergence order, fundamental / low 3 | 1.9999 / 1.9996 | > 1.9 |
| discrete cosine, pointwise / amplitude | **3.6e-14** | 1e-13 |
| start-up at rest / launched (new, §30.4) | 9.5e-15 / 6.9e-16 | 1e-12 |
| FFT fundamental vs discrete, N = 120 | 0.010 cents | 5 |
| local roots vs the crate's `brentq` roots | 3.7e-13 | 1e-11 |
| `mu = 50`, N = 40, 0.5 s drift | 3.4e-14 | 1e-9 |
| Krylov / inverse iteration vs the frozen truth | 0.072 of allowance / 0.034 `eps mu_max` | 1 / 20 |
| the record's LAPACK vs its own Rayleigh quotients | 1.80 (N = 50) | 3 |

The two bold rows are existing bars. The cents row measures an O(h²) discretization error that is
deterministic to ~1e-12, so its 1.4x cannot flake. The discrete cosine's 2.8x is round-off; it has
passed on Linux CI since the port. The start-up bar's first draft was 1e-14 against a measured
9.5e-15. It was widened to 1e-12 before landing, 10⁸ below what the missing ½ reads.

### 30.4 Twelve deliberate breakages — one that nothing in the workspace could see

| breakage | red | caught by |
|---|---|---|
| end mass cell `h/2` → `h` | 5 | trapezoid, frozen referee, closed form, horizon, order |
| `K` scaled by `1 + 1e-7` | 1 | **only the frozen referee** — energy, cents and FFT cannot see 1e-7 |
| last curvature row dropped | 6 | referee, closed form, horizon, order, decay, FFT |
| `theta` in the update matrix only (×0.9) | 8 | every trajectory bar |
| loss sign in the update matrix | 3 | passivity, decay, underdamping |
| **start-up drops the ½** | **1** | **only the new start-up bar** — below |
| start-up velocity sign flipped | 1 | only the new start-up bar |
| energy drops `rho` | 1 | only the density bar, as §29.3 |
| `N ≥ 4` guard → `N ≥ 3` | 1 | the refusal table |
| `free_free_beam_beta_l`: `sech` × 1.01 | 2 | the new roots bar, and the analysis crate's own |
| `discrete_beam_eigenfrequency`: `4θ` → `3.6θ` | 1 | the discrete-cosine cross-check, now calling the function |
| the step writes a NaN into node 1 | 8 | every trajectory bar — **4 against the committed file** |

**Dropping the ½ in `u^{-1} = u^0 - k v^0 + ½ k² a^0` passed every test in every crate** (checked
workspace-wide), and nothing in the Python suite looked either. Energy cannot see it, because any
`u^{-1}` gives a conserved run. The discrete cosine cannot, because the recurrence holds from every
start. A phase error of order `c` is invisible to a five-cent FFT. The new
`the_start_up_is_the_consistent_second_order_one` asserts the two exact identities the start implies
for an eigenmode:

- at rest, `u^1 - u^{-1} = theta c² / (1 + theta c) u^0`, which is not zero, because `1 - c/2` is
  only the Taylor start of the scheme's own `cos(omega k)`;
- launched from rest, the centred velocity is exactly `V`.

The NaN row is §29.3's rule proven on this file. Its convergence bar folded `err_low3` with
`f64::max`, and its discrete-cosine bar folded the residual the same way. **The committed
discrete-cosine bar passed a run that was NaN from the first step**: every residual was NaN, so the
worst stayed 0. Both folds now go through `nan_max`.

### 30.5 The retirement rule, discharged

| retired | native bar |
|---|---|
| `test_energy_conserved` (3) | `a_lossless_beam_conserves_its_energy_at_every_mu` — now the longer of 1 s and the 8,000 steps it ran before, so no run got shorter |
| `test_energy_conserved_with_timestep_explicit_could_not_run` | the same run at `mu = 16` |
| `test_energy_strictly_positive_when_lossless` | the same runs; its 0.5 s at `mu = 2` is a prefix |
| `test_passivity_monotonic_decrease` | `a_lossy_beam_is_passive` — now 1 s, its slack |
| `test_decay_rate_matches_2sigma_low_mode` | `a_low_mode_decays_at_twice_sigma` |
| `test_higher_mode_underdamps_relative_to_lower` | `a_higher_mode_underdamps_relative_to_a_lower_one` |
| `test_energy_units_scale_with_density` | `the_energy_scales_linearly_with_density` — stricter: `np.isclose(rtol=1e-12)` kept `atol = 1e-8` (§16's (d)) |
| `test_operator_symmetric` | `the_stiffness_is_symmetric_to_the_bit` |
| `test_rigid_body_nullspace` | `the_stiffness_annihilates_its_rigid_body_nullspace_and_nothing_else` |
| `test_modal_frequencies_match_closed_form` | `the_low_modes_sit_inside_their_measured_pitch_horizons` (new) — plus the frozen referee |
| `test_convergence_order_h2` | `the_operator_eigenvalues_converge_at_second_order` — NaN fold fixed |
| `test_fft_rings_at_fundamental` | `a_struck_beam_rings_at_its_discrete_fundamental` (new) |
| `test_resonator_uses_operator_helper` | `the_resonator_uses_the_operator_helper_verbatim` — to the bit |
| `test_no_nan_across_mu` (5) | `nothing_blows_up_anywhere_in_the_mu_sweep` (new) |
| `test_explicit_unstable_config_runs_stably` | `a_courant_number_two_hundred_times_the_explicit_bound_runs_and_conserves` (new) |
| `test_invalid_parameters_rejected` (11) | `the_construction_refusals_are_the_documented_ones` — each variant |

Three new bars carry no Python test:

- `both_eigensolvers_reproduce_the_frozen_referee_at_every_grid_the_suite_solved_on` (§30.1–30.2);
- `the_local_roots_are_the_analysis_crates_roots`, which ties the file's own bisected oracle to the
  crate's `brentq` one, so a defect in either is seen;
- the start-up bar.

The file's header no longer says the core crate has no path to the analysis crate. That stopped
being true at §24.

Orphans removed from `tests/helpers.py`: `make_beam`, `beam_low_eigenfrequencies`,
`KAPPA_BEAM_DEFAULT`, `MU_BEAM_DEFAULT` and the `FreeBeam` import. `tests/test_stability.py`'s
`test_arpack_oracles_are_bit_reproducible` lost its beam half. `docs/dev/plate-free-edge-plan.md`
points at the native file. The beam row of `docs/dev/resolution-horizon-plan.md`'s hand-picked-band
audit names the native bar.

### 30.6 Cost and counts

- **pytest 1,368 → 1,333**: 32 cases plus three `test_xdist_groups` parametrizations, reconciled
  by collecting before and after. Full run: 1,333 passed.
- **Native +7**, all in `beam.rs`. The file takes 0.74 s in release and **14.5 s in debug** on the
  dev machine (the one-second runs are 220k steps of a 65-node beam), so it stays in both CI
  profiles by the default. Workspace: 1,353 passed in release.
- **On CI (run 36553822722, `358e004`, all green)**: 1.1 s in the release job and **9.5 s in the
  debug job**, which took 3.2 minutes against the release job's 11.6. This was the first Linux run of
  the frozen-referee, roots and start-up bars (glibc's `cos`/`cosh` under the roots, and the
  frozen LAPACK and Rayleigh numbers read back on a second platform); every bar passed.

### 30.7 What is next

48 physics files, 510 test functions. The string families are next by §28.7's order; the human
picks.

One loose end, from §27 rather than this batch: the surviving half of
`test_arpack_oracles_are_bit_reproducible` guards `free_plate_low_eigenfrequencies`, and that
helper's only remaining caller is the guard itself. The guard still asserts something true, but
about a helper nothing uses. It is the drained-table shape of ledger #52, one level up. The second
guard there, "every `eigsh` call pins `v0`", is still live: the bore and the von Kármán free plate
call `eigsh`.

## 31. Phase C, carrying batch 8 — the ideal string

Done 2026-09-29. The human took the recommendation: the ideal string first among the string
families, because it is the base the stiff, damped and tension strings build on. Its harness was
never one file. The Python suite was organised by acceptance criterion rather than by model, so
four files retire whole:

- `tests/test_energy.py` (6 functions, 10 cases);
- `tests/test_modal.py` (3);
- `tests/test_convergence.py` (2, both `slow`);
- `tests/test_dispersion.py` (5, three `slow`).

A fifth, `tests/test_stability.py`, loses its first four functions (15 cases), the string's
stability and construction guards, and keeps the rest. What stays in it is about the Python
package: the headless-core import check, the dependency allowlist, the sibling-layer check, the
one-implementation guard and the two ARPACK guards. **20 functions, 35 pytest cases** in all.

As in §30 there is no new harness file. `crates/physsynth-core/tests/string_ideal.rs` already built
its string exactly as `make_string` does (`fs = c N / (L lam)`, `c = 200 m/s`), so it was extended,
from 13 `#[test]`s to 21. The claims that touch no model went to
`crates/physsynth-analysis/tests/oracles.rs`, as one new bar. There was no core source change,
because `impl Resonator for IdealString` has been in `engine.rs` since Phase 0.

### 31.1 No outside referee

This is the first carrying batch with nothing to freeze. Every number the retired tests compared
against was one of two things:

- **a closed form**, written in the test: `n c / 2L`, `exp(-2 sigma t)`, the ratio 2 and the
  one-cent bar;
- **already Rust through the binding**, which makes it Rust checking Rust:
  - `triangular_pluck`, `simulate` and `modal.mode_shape`;
  - `spectrum.measure_partials_near` and `spectrum.detect_peaks`;
  - `modal.discrete_mode_frequency`, `dispersion.dispersion_frequencies` and
    `dispersion.phase_velocity`.

NumPy's only role was arithmetic on those results (`np.diff`, `np.log`, `np.round`, `states @ phi`),
and none of it is a referee. The one reduction transcribed by hand, the modal projection
`q = <u, phi_m>`, is a sum whose ordering cannot move a frequency measured to 1e-4 or 1e-7. The
native twin of the Python's `measure_mode_frequencies` reproduces the Python's own recorded worst
cases: 1.35e-5 at λ = 0.8, where the Python's comment says "~1.3e-5", and 8.1e-10 at λ = 1, where
it says "~8e-10".

### 31.2 What the existing native bars could not see

Before the carry, `string_ideal.rs` checked the claims the Python made, but mostly at other
parameters. Three differences mattered.

- **Every energy bar in `string_ideal.rs` ran at λ = 1.** The Python's
  `test_energy_conserved_across_lambda` exists because the conservation identity is algebraic, not
  a λ = 1 accident. The planted breakage B below proves the gap. An energy that uses `(h/k)²` where
  `c²` belongs is exact at λ = 1, so **it passed every bar the string's own file had before this
  batch.** Inside that file it is caught only by the carried λ sweep and the carried λ = 0.9
  free-end runs.

  The workspace as a whole did see it (§31.4, re-planted workspace-wide after review). It saw it
  only through models *built on* the string (the bridges, the sympathetic strings) and through the
  viewer, never through the string's own harness. A defect in the string found only by a chain's
  ledger reads as a chain failure, which is the wrong place to start looking.
- **Two energy bars folded with `worst.max(..)`**, which drops a NaN (§29.3). Both now go through
  the driver's `energy_drift`, which `engine.rs` proves propagates one. The passivity bar was a
  per-step comparison and could not pass a NaN, but its carried twin goes through `nan_max` anyway.
- **The decay-rate bar was at 5%, the Python's at 2%.** It now carries 2%, against a measured 8e-6.

No run got shorter (§30's review rule).

- The old λ = 1 lossless run (pluck at 0.3, 10,000 steps) stays beside the carried sweep.
- The free-end test runs each of its three end combinations at both the old λ = 1 and the
  Python's λ = 0.9, each for the longer of 2 s and 5,000 steps.
- The passivity test keeps the old σ = 3 run and adds the Python's σ = 5 over 2 s.

`boundary="clamped"` was the one construction refusal with no native line. It is a refusal of a
*value*, so by §14's rule it has an analogue. `Boundary::parse("clamped")` must return `None`, with
the two known spellings as its control.

### 31.3 Every margin measured

| bar | measured | bar |
|---|---|---|
| lossless drift, λ ∈ {1, 0.99, 0.9, 0.7, 0.5}, 2 s | ≤ 7.0e-14 (λ = 0.5, 80,000 steps) | 1e-10 |
| lossless drift, three free-end combinations × λ ∈ {1, 0.9} | ≤ 5.1e-14 (free–free, 0.9) | 1e-10 |
| passivity, worst step rise / E⁰ (σ = 3 / σ = 5) | 5.8e-16 / −4.0e-16 (every step fell) | ≤ 1e-12 |
| decay vs `exp(-2σt)`, log space | 8.1e-6 | 2% |
| `E(2ρ, 2T) / E(ρ, T)` | exactly 2 | **exact** (below) |
| ten partials vs `n c / 2L`, λ = 1 | 2.8e-3 cents | 1 cent |
| blind detector, six peaks vs nearest harmonic | 2.8e-3 cents; harmonics {1, 2, 3, 5, 6, 10} | 1 cent |
| mode 8 at λ = 0.9: errors at N = 64 / 128 / 256 | 0.989 / 0.245 / 0.061 Hz | strictly shrinking |
| its orders / mean order | 2.015, 2.006 / 2.010 | > 1.7 / (1.85, 2.15) |
| its N = 128 run vs the dispersion oracle | 2.9e-4 Hz | 3.4e-3 Hz |
| nine modes vs the oracle, λ = 0.8 | 1.35e-5 relative | 1e-4 |
| nine modes vs the continuum, λ = 1 | 8.1e-10 relative | 1e-7 |
| `|v_p / c − 1|` at λ = 1 | 8.1e-10 | 1e-7 |

The density bar is now an exact equality, and that is a structural claim, not a measured one.
Doubling both `ρ` and `T` leaves `c` bit-identical, and with it the grid, `fs` and the displacement
field; the test asserts both of those as its control. `ρ` itself was doubled, which is exact in
binary floating point on any IEEE platform. The Python's `np.isclose(rtol=1e-12)` carried
`atol = 1e-8` (§16's (d)).

The phase-velocity bar at λ = 1 is asserted as the bare `1e-7` the Python wrote. Its
`np.allclose` had kept `rtol = 1e-5`.

The three oracle claims moved to the analysis crate, at the Python's own `c = 200`, `L = 1` and
N ∈ {100, 128}, at 1e-12 relative. The existing oracle bar made the same claims at L = 0.65 with a
1e-9 *absolute* bound, and the Python's `np.isclose` at 2,500 Hz had been governed by its hidden
`atol`.

### 31.4 Ten deliberate breakages

Planted one at a time with `W:\temp\claude\ideal-string\mutate.py`. Each source file was
snapshotted first and restored by copy. Every restore was byte-compared, and `git status` showed
only the test files afterwards.

| breakage | red | caught by |
|---|---|---|
| **A** update uses λ for λ² (identical at λ = 1) | 5 | λ sweep, free end at 0.9, dispersion at 0.8, convergence, the old discrete cosine |
| **B** energy uses `(h/k)²` for `c²` (identical at λ = 1) | 2 | in `string_ideal.rs`, **only the carried λ sweep and free-end λ = 0.9 runs**; workspace-wide, 48 tests in 11 files (below) |
| **C** energy drops `ρ` | 1 | in `string_ideal.rs`, only the density bar; workspace-wide, 59 tests in the same 11 files |
| **D** update doubles σ | 1 | the decay rate |
| **E** loss sign flipped | 2 | passivity, decay rate |
| **F** free-end weight `h/2` → `h` | 1 | the free-end bar |
| **G** CFL test loosened to `λ > 1.1` | 1 | the construction refusal |
| **H** `Boundary::parse` accepts `"clamped"` | 1 | the new `parse` line |
| **I** oracle: λ moved outside the `asin` (identical at λ = 1) | 4 | dispersion at 0.8, convergence vs oracle, both analysis dispersion bars |
| **J** `parabolic_refine`'s sign flipped | 4 | ten partials, blind detector, dispersion at 0.8, convergence |

The "red" column counts `string_ideal.rs` (plus `oracles.rs` for I and J). B and C were then
re-planted against the **whole workspace**, `cargo test --workspace --release --no-fail-fast`, with
`W:\temp\claude\ideal-string\mutate_ws.py`. That was done after review, because §30.4's
"nothing in the workspace" claims had been made that way and this section's first draft had only
run the one file. Both went red in the same eleven test binaries:

- `physsynth-core`: `connection`, `connection_body`, `connection_plate`, `string_ideal`;
- `physsynth-viewer`: `airbox`, `airload`, `body`, `frozen`, `platebody`, `radbody`, `sympathetic`.

Every chain that carries an ideal string runs it inside a total-energy ledger, often at λ < 1. The
Windows-exact viewer freeze pins the energy's bits. So neither defect could have shipped. What the
carry fixes is *where* it would have been found.

Two things the table shows beyond "all red":

- **B is the batch's reason to exist.** Energy bars at λ = 1 alone cannot see a defect that
  vanishes at λ = 1, and the string's own harness had nothing else. The same holds for A and I at
  the other two layers: the model's update and the oracle.
- **J left the λ = 1 continuum bar green, and that is not a gap.** At λ = 1 with N = 128 the
  window is 0.5 s at `fs = 25,600`, zero-padded to 32,768 points. The bin spacing is then exactly
  0.78125 Hz, and every swept mode `m · 100 Hz` lands on bin `128 m`. The peak is centred on its
  bin, so the refiner's correction is zero whatever its sign. That bar asserts the continuum, and
  the refiner is pinned by the four bars that went red.

### 31.5 The retirement rule, discharged

| retired | native bar |
|---|---|
| `test_energy_conserved_across_lambda` (5) | `lossless_energy_is_conserved_and_positive_at_every_courant_number` |
| `test_energy_conserved_free_boundary` | `a_free_end_conserves_energy_too` — its λ = 0.9 run, both ends free |
| `test_energy_strictly_positive_when_lossless` | the λ sweep — every step of all five 2 s runs, a superset of its 1 s at 0.9 |
| `test_passivity_monotonic_decrease` | `loss_makes_the_energy_decrease_monotonically` — its σ = 5, 2 s run |
| `test_decay_rate_matches_2sigma` | `the_decay_rate_matches_the_analytic_two_sigma` — its 2% |
| `test_energy_units_scale_with_density` | `the_energy_is_in_joules_and_scales_with_density` — exact |
| `test_partials_within_one_cent_at_lambda_one` | `a_plucked_string_sounds_its_harmonic_series_within_a_cent` |
| `test_blind_detection_finds_harmonic_series` | `the_blind_detector_finds_the_harmonic_series_on_its_own` |
| `test_discrete_oracle_matches_continuous_at_lambda_one` | `oracles.rs::the_string_harness_oracles_hold_at_its_own_parameters` |
| `test_second_order_convergence_at_fixed_lambda` | `a_dispersive_mode_converges_at_second_order_onto_the_dispersion_oracle` |
| `test_detected_frequency_tracks_dispersion_oracle` | the same test, reading its N = 128 run against the oracle |
| `test_dispersion_matches_oracle_below_lambda_one` | `every_swept_mode_lands_on_the_dispersion_oracle_and_droops_below_c` |
| `test_phase_velocity_droops_with_mode_below_lambda_one` | the same test, on the same measurement |
| `test_dispersion_flat_at_lambda_one` | `at_courant_one_every_swept_mode_is_on_the_continuum` |
| `test_phase_velocity_recovers_c_for_continuum` | `oracles.rs::the_string_harness_oracles_hold_at_its_own_parameters` |
| `test_dispersion_frequencies_match_scalar_oracle_and_droop` | the same analysis bar — the vector–scalar agreement exact (`assert_eq!`) |
| `test_no_nan_across_valid_lambda` (7) | `no_admissible_courant_number_produces_a_nan` |
| `test_lambda_above_one_rejected_at_construction` | `courant_above_one_is_rejected_at_construction` (existing, same λ = 1.05 and the "CFL" text) |
| `test_lambda_exactly_one_is_accepted` | `courant_exactly_one_is_accepted` (existing, same 1e-12) |
| `test_invalid_parameters_rejected` (6) | `non_physical_parameters_are_rejected` (existing, each variant); `"clamped"` in `an_unparseable_boundary_is_rejected_after_the_scalar_checks` |

The file's header no longer says the Python tests "are still the authority". It names what was
carried and why nothing was frozen.

Orphans removed from `tests/helpers.py`:

- `make_string`, whose last callers were the deleted files and the carried half of
  `test_stability.py`. That removal took the module's `string_ideal` import with it.
- `measure_mode_frequencies`, whose only caller was `test_dispersion.py`. Its stiff-string twin,
  `measure_stiff_mode_frequencies`, had defined itself in its docstring as "identical to
  `measure_mode_frequencies`". That docstring now stands alone and points at the native twin.

Each name was grepped on its own. The first scoping grep OR'd the string helpers together, and it
reported callers that belonged to `wave_speed` and `convergence_orders`, not to `make_string`.
Those two stay: eight files call the first and two the second. `tests/test_stability.py` dropped its string
imports and says in its docstring where its first four tests went. `docs/dev/resolution-horizon-plan.md`'s
hand-picked-band audit names the native bar for the ten-partial test.

### 31.6 Cost and counts

- **pytest 1,333 → 1,294**: 35 cases plus four `test_xdist_groups` parametrizations, one per
  deleted file. The stability file stays, so it keeps its parametrization. Reconciled by
  collecting before and after. Full run: 1,294 passed.
- **Native +9**: eight in `string_ideal.rs` (13 → 21) and one in `oracles.rs`. The whole file
  takes 0.16 s in release and **3.3 s in debug** on the dev machine, so it stays in both CI
  profiles by the default. Workspace: 1,362 passed in release.
- **On CI (run 36578409910, `9ab14d0`, all green)**: 0.36 s in the release job and **6.2 s in the
  debug job**. The debug job took 6.0 minutes against the release job's 11.3, the exact Windows
  viewer freeze 5.8 and the Python suite 2.6. This was the first Linux run of the carried
  frequency bars at the Python's parameters, and every bar passed.
- **44 physics files remain** (§24.1's exclusions). Their test functions come to **496** by a
  re-count. The same count run over §30's commit gives 512, not §30.7's 510, so the two-function
  gap is a difference in how the count was taken, not in the files. The carried 16 physics
  functions are the difference either way.

### 31.7 What is next

The three remaining string families, all built on this one:

- the stiff string (`test_stiff_string`, 20 functions);
- the damped string (`test_damped_string`, 16);
- the tension-modulated string (`test_tension_string`, 30).

The human picks. §27's loose end, the free-plate half of `test_arpack_oracles_are_bit_reproducible`,
is still open.
