"""Kirchhoff plate (simply-supported **or** free rectangle) — implicit theta-scheme FDTD.

**The implementation is Rust.** This module is what is left of the Python one after the
migration's sixth deletion (``docs/dev/rust-migration-plan.md`` §43, unit 5): both classes live in
``crates/physsynth-core/src/plate.rs`` — models #5 (supported), #5b (free), #5o and #5of (the
grain), #5g (the guitar outline) and #6 (von Kármán) are all branches of them — bound in
``crates/physsynth-py``, and re-exported here under the names every caller already imports. The
physics is documented at length in that Rust module's header and in ``docs/dev/plate-plan.md``,
``docs/dev/plate-free-edge-plan.md``, ``docs/dev/orthotropic-free-plate-plan.md``,
``docs/dev/guitar-plate-plan.md`` and ``docs/dev/von-karman-plate-plan.md``.

# Why this deletion needed a change to the Rust side first

Unit 5 was blocked for three batches on something small: ``Plate.B`` was a ``#[getter]`` with no
setter, and ``tests/test_plate_modal.py`` assigns to it on purpose. That test is the only pin on
the 2026-08-28 sparse-assembly finding — it steps one plate on the shipped, canonically sorted
biharmonic and another on the pre-fix assembly, the *same numbers in SciPy's kernel order*, and
asserts the trajectory did not move. Deleting the Python plate would have retired a claim about a
summation order because the Rust plate had nowhere to put the other order.

The human's call on 2026-09-03 was to open the door rather than retire the test, and it took two
pieces: ``Csr::from_arrays_preserving_order``, the one constructor in the core crate that does
**not** sort a row it is handed, and a ``#[setter]`` for ``B`` that uses it. Both are documented as
existing for that test, and the three merge kernels (``add``, ``sub``, ``matmul``)
``debug_assert`` the invariant the constructor relaxes, so a non-canonical operator cannot leak
into an assembly unnoticed. Plan §40.5 posed it; §43 records what it cost.

# What is left here, and why each thing had to stay

* ``Boundary`` and ``Domain`` — type aliases with no runtime implementation.
* ``THETA_DEFAULT`` — a measured default with its reasoning, read by ``tests/helpers.py``. The
  number is also ``physsynth_core::plate::THETA_DEFAULT``; the Rust carries the value, this
  carries why it is not 1/4.
* ``GrainSpec`` — kept because **the Rust side constructs it**. ``grain_ratios_from_material`` is
  implemented in the crate and reaches back through
  ``py.import("physsynth.core.plate").getattr("GrainSpec")``, which is a dependency running Rust →
  Python that no audit of Python's own imports can find (§41.2 — the grep is
  ``grep -rn 'import("physsynth' crates/physsynth-py/src/``).

Headless: no I/O, no plotting.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

from physsynth_rs import Plate, VKPlate, grain_ratios_from_material

__all__ = [
    "Boundary",
    "Domain",
    "THETA_DEFAULT",
    "GrainSpec",
    "grain_ratios_from_material",
    "Plate",
    "VKPlate",
]


Boundary = Literal["supported", "free"]
# The plate's OUTLINE, orthogonal to its boundary condition. "rectangle" is every plate the core
# built before model #5g. "circle" and "guitar" are staircased onto the Cartesian grid and are
# offered on the FREE branch only -- a *supported* curved plate would be free to add (B = L @ L
# already takes any mask) and would be a model surface with no content, since its spectrum is
# exactly the membrane's squared on the same outline. See docs/dev/guitar-plate-plan.md §2.
Domain = Literal["rectangle", "circle", "guitar"]

# theta below 1/4 is only conditionally stable; theta in (0, 1] keeps A SPD (genuinely implicit).
# Default a hair above 1/4 (accuracy-first per the plan) so the energy has a small positivity margin
# while staying near the minimal-dispersion theta = 1/4 -- inherited from the stiff string.
THETA_DEFAULT = 0.28


class GrainSpec(NamedTuple):
    """Everything :class:`Plate` needs from a material: ``kappa``, ``rho_s`` and the grain ratios.

    A named tuple rather than a bare one on purpose. ``rho_s`` is the **areal** density
    (``rho * thickness``) and :class:`Plate`'s ``rho`` argument is areal too, but the *material*
    density fed to :func:`grain_ratios_from_material` is a **volume** density — so the one thing a
    caller could plausibly do is pass the volume density straight through to ``Plate``. That slip
    leaves every frequency correct (``kappa`` carries them) and every energy wrong by a factor of
    the thickness, which is precisely the class of error this batch found no ledger and no modal
    oracle can catch. Returning ``rho_s`` explicitly, named, removes the opportunity.

    **Five ratios, not three, since model #5of.** ``grain_cross`` (``H/D_ref``) is what the
    *supported* branch needs; ``grain_coupling`` and ``grain_torsion`` (``D_1/D_ref``,
    ``D_xy/D_ref``) are the two halves of it that a *free* edge can tell apart, and are what the
    free branch needs. `grain_cross == grain_coupling + 2*grain_torsion` by construction — the
    two are not independent, they are the same material seen through two boundaries. Use attribute
    access (``spec.grain_y``): the field list has grown once already and positional unpacking is
    what broke when it did.
    """

    kappa: float
    rho_s: float
    grain_x: float
    grain_cross: float
    grain_y: float
    grain_coupling: float
    grain_torsion: float
