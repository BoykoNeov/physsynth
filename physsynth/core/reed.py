"""Dynamic single reed on a bore — the wind leg's self-oscillating exciter (the clarinet).

**The implementation is Rust**: ``crates/physsynth-core/src/reed.rs``, bound in
``crates/physsynth-py`` and re-exported here (``docs/dev/rust-migration-plan.md`` §39, unit 3).
That module's header is now the only copy of the physics — the reed as the acoustic dual of the
bow, the Bernoulli flow through the aperture, and the reason an energy *balance* is not sufficient
here: a dead reed passes a balance test on silence, so the model's money bar is a signature oracle.

Note what the port had to reproduce and what therefore cannot be simplified: the solve's Brent
fallback genuinely fires, so `scipy.optimize.brentq` was transcribed rather than replaced — a
branch choice is part of the trajectory (findings #12). That transcription is in the Rust.

Headless: no I/O, no graphics.
"""

from __future__ import annotations

from physsynth_rs import ReedBore, bernoulli_flow

from .bore import Bore

__all__ = ["Bore", "ReedBore", "bernoulli_flow"]
