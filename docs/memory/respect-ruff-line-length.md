---
name: respect-ruff-line-length
description: "Keep every line ≤100 chars from the first draft — was ruff E501 on Python (gone 2026-10-17, §51); the same width is the Rust source convention. Never reflow after"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 2cabff3c-9fbb-420e-9f6c-1bbbf8a6d6a5
---

**UPDATE 2026-10-17 (retirement plan §51):** there is no Python and no ruff job any more. The width rule
stands for Rust: the crates keep code and `//!`/`///` comments at ≤ 100 characters (rustfmt does not
wrap comments, so nothing enforces it — draft within it). The history below is about Python.

When writing or editing ANY Python in this repo, keep every line ≤ 100 characters in the **first draft** — including docstrings and comments. This codebase is deliberately docstring-heavy, which is exactly where over-length lines slip in. The ruff config (`pyproject.toml`) selects `E` (incl. E501) at `line-length = 100`, and CI runs `ruff check .` as a hard gate that fails fast, *before* the test step.

**Why:** On 2026-06-23, building the free-free beam (model #5b-pre), I drafted `beam.py` + its tests with many >100-char lines, tripping 79 E501 violations, then spent a large, wasteful reflow pass (5 subagents) fixing my own mess. The user flagged this explicitly as wasted tokens. Composing within the limit costs nothing; reflowing afterward is pure churn and produces ugly wraps. (Separately discovered the same neglect had left CI red for the two prior commits — plate #5 and web-viewer Phase B — so the gate matters.)

**How to apply:** Budget for the 100-char limit *while* composing — mentally wrap long prose, and break long calls/asserts inside existing parens as you write them, not afterward. More generally: learn a project's lint/format rules before generating a lot of code, and self-check (`ruff check <files>`) before declaring code done. The target is **zero reflow churn**, not "fix it later." Related: [[commit-push-at-batch-end]] — CI must be green for the push rule to mean anything.