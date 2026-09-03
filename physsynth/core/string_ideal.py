"""Ideal (lossless or linearly damped) transverse string — explicit FDTD scheme.

**The implementation is Rust.** This module is what is left of the Python one after the migration's
first deletion (``docs/dev/rust-migration-plan.md`` §39, unit 7): the class lives in
``crates/physsynth-core/src/string_ideal.rs``, is bound in ``crates/physsynth-py``, and is
re-exported here under the name every caller already imports. The physics — HANDOFF §4.2–§4.3, the
explicit second-order scheme, and the **cross-time** potential term that is the reason
:meth:`energy` conserves to machine precision rather than drifting at ~1e-3 — is documented at
length in that Rust module's header, which is now the only copy of it.

What this file is *not* is a compatibility shim that could be removed. Under §35.5's route (b) the
viewer stays Python, so ``physsynth.core.string_ideal`` remains the import path for
``web/serialize.py``, ``connection.py``, the diagnostic scripts and the whole test suite. The name
is the interface; only the body moved.

Two consequences worth stating here rather than rediscovering:

* **``physsynth_rs`` is now a hard requirement to import this module**, and therefore to collect
  most of the suite. There is no longer a pure-Python path to fall back to and no
  ``PHYSSYNTH_RS`` switch on this model — the switch existed to choose between two implementations
  and there is one. The ``validate`` CI job installs the wheel for this reason (the human's call,
  §39.6, route 1).
* **The bit-identity bar on this model is now historical.** ``string_ideal.rs``'s header explains
  that its elementwise kernels are written in NumPy's evaluation order so the two implementations
  agree bit-for-bit. That was checked by ``tests/test_rust_parity.py`` against a Python oracle that
  no longer exists. The ordering is kept because it is correct and free, not because anything still
  measures it; what measures the model now are the physics bars in ``tests/`` and the native bars in
  ``crates/physsynth-core/tests/string_ideal.rs``.

Headless: no I/O, no graphics.
"""

from __future__ import annotations

from typing import Literal

from physsynth_rs import IdealString

Boundary = Literal["fixed", "free"]
# Per-end boundary: a single value applies to both ends; a (left, right) tuple sets them
# independently (e.g. ("fixed", "free") for a string clamped at the nut and free at a body bridge).
BoundarySpec = Boundary | tuple[Boundary, Boundary]

__all__ = ["Boundary", "BoundarySpec", "IdealString"]
