"""Modal body — a bank of damped harmonic oscillators driven at a point (the instrument body).

**The implementation is Rust**: ``crates/physsynth-core/src/body.rs``, bound in
``crates/physsynth-py`` and re-exported here (``docs/dev/rust-migration-plan.md`` §39, unit 2).
That module's header is now the only copy of the derivation, including the fact that the
discretisation mirrors the ideal string's explicit scheme exactly and the ``omega*k == 2`` boundary
case that needs the same floating-point slack the string's CFL check needs.

Nothing but the class survives here. ``_accel`` is *not* private in practice — three modules write
it every step (findings #4) — and it is part of the binding's surface for that reason.

Headless: no I/O, no graphics.
"""

from __future__ import annotations

from physsynth_rs import ModalBody

__all__ = ["ModalBody"]
