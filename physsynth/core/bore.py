"""Linear acoustic bore — staggered pressure/flow leapfrog on the Webster horn equation.

**The implementation is Rust**: ``crates/physsynth-core/src/bore.rs``, bound in
``crates/physsynth-py`` and re-exported here (``docs/dev/rust-migration-plan.md`` §39, unit 3).
That module's header is now the only copy of the physics — the staggered ``p``/``U`` leapfrog, the
area profile ``S(x)`` carried as node/segment arrays so that a cylinder is just the constant case
and a bell is a different profile rather than a different solver, and the three end conditions
including the passively-lossy radiating bell.

What stays here is the module's **types and constants**. ``SourceHook`` is the seam through which
an implicit boundary exciter (the reed) injects into a half-cell node — a `Callable` alias with no
runtime existence to port. ``RHO0_AIR`` and ``C0_AIR`` are two floats; §39.1 counted them as part
of the viewer's seam and they are the whole of it.

Headless: no I/O, no graphics.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from physsynth_rs import Bore

# An in-place pressure-field corrector applied inside step() between the pressure and momentum
# sub-steps: the seam through which an implicit boundary exciter (the reed) injects into a half-cell
# node. It receives the freshly-updated p^{n+1} and mutates it. See Bore.step / ReedBore.
SourceHook = Callable[[NDArray[np.float64]], None]

# Ambient air (matches physsynth.core.radiation so the bore and its radiation load agree).
RHO0_AIR = 1.2041  # kg/m^3
C0_AIR = 343.0     # m/s

End = Literal["closed", "open", "radiating"]
# Per-end boundary: one value applies to both ends; a (left, right) tuple sets them independently.
# A clarinet is ("closed", "open") — rigid mouthpiece wall, open bell; ("closed", "radiating") gives
# it a passively-lossy, sound-shedding bell (batch 2).
BoundarySpec = End | tuple[End, End]

__all__ = ["Bore", "BoundarySpec", "C0_AIR", "End", "RHO0_AIR", "SourceHook"]
