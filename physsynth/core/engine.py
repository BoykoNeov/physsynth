"""Timestep loop: drive a resonator and record diagnostics.

The engine is intentionally thin — it owns iteration and data capture, not physics. It records the
energy at every step (the primary bug detector) and, optionally, a pickup signal for spectral
analysis and state snapshots for animation. Pure NumPy; no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Resonator(Protocol):
    """The minimal interface the engine drives (see HANDOFF §3.2)."""

    k: float

    def step(self) -> None: ...
    def energy(self) -> float: ...
    @property
    def state(self) -> NDArray[np.float64]: ...
    def displacement_at(self, index: int) -> float: ...


@dataclass
class SimResult:
    """Captured diagnostics from a run."""

    time: NDArray[np.float64]               # shape (num_steps+1,), seconds
    energy: NDArray[np.float64]             # E^n at each step
    output: NDArray[np.float64] | None      # pickup displacement, if requested
    fs: float
    snapshots: list[tuple[int, NDArray[np.float64]]] = field(default_factory=list)

    @property
    def energy_drift(self) -> float:
        """``max|E^n - E^0| / |E^0|`` — the lossless-conservation figure of merit."""
        e0 = self.energy[0]
        if e0 == 0.0:
            return float(np.max(np.abs(self.energy)))
        return float(np.max(np.abs(self.energy - e0)) / abs(e0))


def simulate(
    resonator: Resonator,
    *,
    num_steps: int,
    pickup_index: int | None = None,
    snapshot_stride: int = 0,
) -> SimResult:
    """Run ``resonator`` for ``num_steps`` steps, capturing energy (always) and optionally a
    pickup signal and periodic state snapshots.

    The arrays include the initial state at index 0, so they have length ``num_steps + 1``.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1.")

    n = num_steps + 1
    energy = np.empty(n)
    output = np.empty(n) if pickup_index is not None else None
    snapshots: list[tuple[int, NDArray[np.float64]]] = []

    energy[0] = resonator.energy()
    if output is not None:
        output[0] = resonator.displacement_at(pickup_index)  # type: ignore[arg-type]
    if snapshot_stride:
        snapshots.append((0, resonator.state))

    for i in range(1, n):
        resonator.step()
        energy[i] = resonator.energy()
        if output is not None:
            output[i] = resonator.displacement_at(pickup_index)  # type: ignore[arg-type]
        if snapshot_stride and (i % snapshot_stride == 0):
            snapshots.append((i, resonator.state))

    time = np.arange(n) * resonator.k
    fs = 1.0 / resonator.k
    return SimResult(time=time, energy=energy, output=output, fs=fs, snapshots=snapshots)
