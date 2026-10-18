---
name: gate-commit-on-lint-exit
description: "Never chain clippy/fmt | grep ; git commit — a grep swallows the lint's exit status, and a red commit gets pushed"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 34ac3fbb-bad2-4c84-9cfb-b3de45c0935d
  modified: 2026-10-18T05:57:17.883Z
---

Run `cargo clippy ... -- -D warnings` and `cargo fmt --check` as their OWN step and read the result
before `git commit`; never put them in the same command line as the commit behind `| grep` or `;`.

**Why:** 2026-09-28, commit 18798b4 was pushed with a clippy error because the command was
`cargo clippy ... | grep ...; git add ... && git commit ... && git push`. The grep printed the
error and returned success, so the commit and push ran anyway; a follow-up fix commit (86a8c69) was
needed, and CI ran red in between.

**How to apply:** lint first, look at the output, then commit in a separate tool call. If chaining
is wanted, gate on the lint's own exit status (`cargo clippy ... -- -D warnings && git commit`),
never on a pipeline's.

**Since 2026-10-18 CI's lint step also runs `cargo doc` with warnings denied** (a broken or
private doc link is red). Locally, in the same lint step, below-normal and exit-code-true:
`RUSTDOCFLAGS="-D warnings" cmd //v:on //c "start /belownormal /b /wait cargo doc --workspace --no-deps & exit !errorlevel!"`.
The analysis crate `#[path]`-includes core's `root.rs`/`sparse.rs`/`sparse_lu.rs`, so a doc link
in those files must resolve in BOTH crates (a link to a core-only item breaks analysis's doc).

Related: [[respect-ruff-line-length]], [[commit-push-at-batch-end]].
