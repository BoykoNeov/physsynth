"""The resolution horizon — where a scheme's answer stops being in tune.

**The implementation is Rust.** ``crates/physsynth-analysis/src/horizon.rs``, re-exported through
the wrappers below.

Unlike every other module in this package, this one was never a Python body that got deleted. It
was *written* in ``tests/helpers.py``, used by five test files, and promoted here by
``docs/dev/resolution-horizon-plan.md`` §6 because it is an instrument like the rest of this
package rather than a fixture. What that promotion had to answer, it answered:

* **The dependency question is closed, and §6 was wrong about it.** That bullet says
  ``scipy.optimize.brentq`` "would be a dependency decision in the analysis crate, whose allowlist
  is deliberately narrow". It is not one. ``crates/physsynth-analysis/src/lib.rs`` already
  ``#[path]``-includes ``physsynth-core/src/root.rs``, a line-for-line transcription of SciPy's
  ``brentq.c``, and argues out at length why that is an include and not a Cargo edge. ``ALLOWED``
  in ``crates/physsynth-analysis/tests/deps.rs`` stays empty; not one line of ``Cargo.toml`` moved.
* **The freeze question is closed, and §6 was wrong about that too.** It says satisfying
  ``tests/test_analysis_frozen.py``'s derived guard "is impossible — there is no Python
  implementation left to freeze against". True of the modules deleted in
  ``docs/dev/rust-migration-plan.md`` §44, and false of these seven functions: their Python bodies
  were live in ``tests/helpers.py`` right up to the commit that replaced them, so they were
  recorded first — the identical move §44 made. The enforcement contract covering the other 62
  fixtures is untouched, and no amendment to it was needed.

What §6 got right is that a native bar is wanted *as well*: the frozen record catches a
transcription error, a wrong branch and a regression, and it cannot catch an error the Python made
too. ``crates/physsynth-analysis/tests/horizon.rs`` is where the identities in the docstrings below
are checked against their own mathematics rather than against a second spelling.

The prose here is the long form on purpose. It is what ``help()`` reaches, and the numbers in it —
5.925% against 8.378%, the ``-1/m_max**2`` cross-term bound, "the CFL ceiling **is** the spectrum's
minimum cancellation Courant number" — are measured claims someone will want to quote.
"""

from __future__ import annotations

import numpy as np
import physsynth_rs as _rs
from numpy.typing import NDArray

__all__ = [
    "block_weight",
    "cancellation_courant",
    "mode_block",
    "mode_family",
    "pitch_error_cents",
    "pitch_horizon",
    "sinc_horizon_fraction",
]


def pitch_error_cents(
    f_discrete: NDArray[np.float64] | list[float],
    f_continuum: NDArray[np.float64] | list[float],
) -> NDArray[np.float64]:
    """Signed per-mode pitch error in cents — negative means the scheme is FLAT.

    Cents rather than a relative frequency because every threshold worth arguing about is a
    perceptual one, and because it makes the two mechanisms comparable: the theta-scheme's rate
    suppression ``S`` and its pitch error are the *same* quantity through
    ``cents = 600 log2(S)`` (``docs/dev/theta-loss-compensation-plan.md`` section 2).
    """
    fd = np.asarray(f_discrete, dtype=float)
    fc = np.asarray(f_continuum, dtype=float)
    if fd.shape != fc.shape:
        raise ValueError(f"shape mismatch: discrete {fd.shape} against continuum {fc.shape}")
    flat = _rs.horizon_pitch_error_cents(np.ascontiguousarray(fd.ravel()),
                                         np.ascontiguousarray(fc.ravel()))
    return flat.reshape(fd.shape)


def pitch_horizon(
    f_discrete: NDArray[np.float64] | list[float],
    f_continuum: NDArray[np.float64] | list[float],
    cents: float = 5.0,
) -> tuple[int, bool]:
    """``(horizon, monotone)`` — how many leading modes are within ``cents`` of the continuum.

    ``horizon`` counts the **leading prefix**: the number of modes from the first such that every
    one of them is inside the bound. That is deliberately the conservative reading rather than
    "the last mode that happens to be inside", and the two differ exactly when the error curve is
    not monotone. So the predicate is returned alongside it: a caller that sees ``monotone=False``
    knows the single number is hiding something, the same way ``VKPlate`` reports *which* failure
    it had rather than only that it failed. Never collapse the pair back to the integer without
    looking at the flag.

    ``f_discrete`` is what the scheme's own dispersion relation says mode *m* will ring at;
    ``f_continuum`` is the closed-form physical answer. Both must be ordered by mode index and
    must be the same family — mixing an axial family into a diagonal one makes the prefix
    meaningless, which matters because the membrane's two families have horizons a factor of nine
    apart at the same Courant number (``docs/dev/resolution-horizon-plan.md`` section 4).
    """
    fd = np.ascontiguousarray(np.asarray(f_discrete, dtype=float).ravel())
    fc = np.ascontiguousarray(np.asarray(f_continuum, dtype=float).ravel())
    horizon, monotone = _rs.horizon_pitch_horizon(fd, fc, cents)
    return int(horizon), bool(monotone)


def sinc_horizon_fraction(cents: float, power: int = 1) -> float:
    """``m*/N`` — the closed-form space floor, as a fraction of the grid, with no fixture in it.

    The discrete axis eigenvalue is ``(2/h) sin(m pi h / 2L)`` against the continuum ``m pi / L``,
    so ``p_disc / p_cont = sinc(u)`` with ``u = m pi / 2N``. ``power`` is how many factors of that
    the model's *frequency* carries, and it is read off the dispersion relation rather than chosen:

    * ``power = 1`` — a wave or a string, ``omega ~ c p``, so the frequency ratio is ``sinc(u)``.
    * ``power = 2`` — a plate or a beam, ``omega ~ kappa p^2``, so it is ``sinc(u)^2``.

    Hence the exact identity ``sinc_horizon_fraction(c, 2) == sinc_horizon_fraction(c / 2, 1)``:
    **a plate resolves the same share of its grid as a string given half the cents budget.** That
    is what ``docs/dev/resolution-horizon-plan.md`` section 3.2's "twice the droop" means as a
    number — 5.925% of the grid at 5 cents against a string's 8.378%.

    Exact in exact arithmetic, and the two sides are two separate root finds, so assert it on a
    measured tolerance rather than on ``==``.

    Independent of ``c``, ``L``, ``N``, ``k`` and ``fs``, which is the claim; the caller supplies
    only the bound and the power.
    """
    return float(_rs.horizon_sinc_horizon_fraction(float(cents), int(power)))


def mode_family(kind: str, count: int) -> list[tuple[int, int]]:
    """The leading ``count`` ``(m, n)`` index pairs of one 2-D mode family.

    A *family* here is a sequence of index pairs along which the pitch error is monotone, so that
    ``pitch_horizon``'s leading-prefix reading means something. The canonical ones:

    * ``"axial"`` — ``(m, 1)``, one half-wave across the other axis. On an isotropic square this
      is the only axial family there is, because ``(1, n)`` is its exact degenerate twin.
    * ``"axial_y"`` — ``(1, n)``, the transpose. A **grain destroys that degeneracy**: the
      orthotropic plate weights the two axes differently, so the two stop being the same
      measurement, and the soft axis is the one that leaves the isotropic closed form first
      (``docs/dev/resolution-horizon-plan.md`` section 9). Deliberately not spelled ``"axial_x"``
      alongside it: one concept, one spelling, and every existing caller means the x-family.
    * ``"diagonal"`` — ``(m, m)``.

    This is deliberately **index-side only**: it returns mode numbers and nothing else, so the
    caller still builds its own discrete and continuum frequencies from its own fixture. That is
    the shape ``spatial_operator_horizon`` got wrong (section 7.7 of the plan) — a helper that
    takes ``(N, mu)`` and hands back "the horizon" hides the geometry and the boundary condition
    in its body, and answers about the wrong model without saying so. That one function is the
    reason this module has seven members and not eight: it stayed in ``tests/helpers.py``, where a
    hardcoded default string is a fixture rather than a lie.

    Assumes a **square** domain: on ``Lx != Ly`` the ``(m, 1)`` and ``(1, n)`` families stop being
    degenerate and are two different measurements, so ask for each separately.
    """
    return [(int(m), int(n)) for m, n in _rs.horizon_mode_family(str(kind), int(count))]


def mode_block(m_max: int) -> list[tuple[int, int]]:
    """Every ``(m, n)`` with ``1 <= m, n <= m_max``, ordered by continuum frequency (``m^2 + n^2``).

    A *block* is the 2-D shape a "the first few modes are in tune" claim actually asserts, and it
    is not a family: the error is not monotone along it (section 8 of the plan), so a prefix over
    it is not a horizon. What makes a block readable is that its worst mode is a **corner** — the
    error weight ``w(m, n) = (m^4 + n^4) / (m^2 + n^2)`` has an interior minimum in ``n``, so the
    maximum over a block never sits inside it. A block's horizon is then some corner family's
    horizon, and a family does have a prefix.

    **Which corner is not a property of the block — it is a property of the scheme**, and getting
    this backwards is a real hazard because the two answers are opposite:

    * on an **implicit** plate or beam, whose time error flattens like its space error, the worst
      mode is the **diagonal** corner ``(m_max, m_max)`` for every ``m_max >= 2`` (section 8.7);
    * on an **explicit** membrane, whose time error is *sharp*, the diagonal corner is worst only
      below ``lambda = 1 / sqrt(m_max^2 + 1)``, which is beneath the 2-D CFL ceiling for every
      ``m_max >= 2``. At every Courant number a membrane is actually run at, the worst mode is an
      **axial** corner — ``(m_max, 1)`` and its exact degenerate twin ``(1, m_max)`` (section 10).

    So ask :func:`cancellation_courant` which regime the caller is in rather than assuming the
    plate's answer.

    **Two things here are isotropic and do not survive a grain** (section 9 of the plan):

    * the **ordering key** is ``m^2 + n^2``, which is the continuum frequency of an *isotropic*
      plate. An orthotropic plate orders its modes by ``g_x a^2 + 2 g_h a b + g_y b^2`` instead,
      so on a grained plate the returned *set* is still the block and the *order* is no longer its
      spectrum. Sort by ``modal.orthotropic_plate_freqs`` if the order matters, or use the set
      alone;
    * the **corner argument** holds exactly while ``grain_cross > -1 / m_max^2`` and fails below
      it, where an off-diagonal mode becomes the block's worst. That bound is measured and exact,
      and it *tightens* as the block grows, so a large enough block breaks it for any negative
      cross term at all. Every real wood has a positive one.
    """
    return [(int(m), int(n)) for m, n in _rs.horizon_mode_block(int(m_max))]


def cancellation_courant(m: int, n: int) -> float:
    """The Courant number at which an explicit scheme's mode ``(m, n)`` is exactly in tune.

    An explicit leapfrog has two pitch errors of **opposite sign**: the spatial operator droops
    the frequency by ``sinc(u)`` and the time discretisation sharpens it. To leading order in
    ``1 / N`` the two combine as ``omega_disc / omega_cont = 1 + (a^2 / 6)(lambda^2 rho^2 - w)``
    with ``a = pi / 2N``, ``rho^2 = m^2 + n^2`` and the block weight ``w = (m^4 + n^4) / rho^2``,
    so they cancel at ``lambda^2 = (m^4 + n^4) / (m^2 + n^2)^2`` — which is what this returns.

    Three consequences, and none of them is a fixture (section 10 of
    ``docs/dev/resolution-horizon-plan.md``):

    * it is ``1 / sqrt(2)`` on the **diagonal** ``m = n`` for every ``m``, and that is exactly the
      2-D CFL ceiling. The membrane's "magic Courant number" is not a coincidence of the stability
      bound: ``t^2 + (1 - t)^2`` with ``t = m^2 / rho^2`` is minimised at ``t = 1/2``, so the
      ceiling **is** the minimum of this function over the whole spectrum;
    * hence ``lambda <= 1 / sqrt(2) <= cancellation_courant(m, n)`` for every mode, so on a stable
      membrane **no mode is ever sharp** — every one of them is flat, or (the diagonal, at the
      ceiling) exact;
    * it rises toward ``1`` along the **axial** family ``(m, 1)``, which is above the ceiling and
      therefore unreachable. That is why the ceiling buys the diagonal family the entire grid and
      the axial family nothing (section 4.1).

    ``n = 0`` spells the **1-D degenerate case** — no second axis, so ``rho^2 = m^2`` and
    ``w = m^2`` — and the formula returns exactly ``1.0`` for every ``m``. That is the 1-D CFL
    limit, and it is the whole reason an ideal string at ``lambda = 1`` resolves its entire grid
    while a membrane at its own ceiling resolves only one family: in 1-D *every* mode attains the
    stability limit at once, in 2-D only the diagonal does.

    Leading order in ``1 / N^2``, so a measured crossing approaches this rather than sitting on
    it. The diagonal value is the exception and is exact at every ``N``: there
    ``lambda sqrt(S) = sin(u)`` identically, and ``arcsin`` undoes it.
    """
    return float(_rs.horizon_cancellation_courant(int(m), int(n)))


def block_weight(m: int, n: int) -> float:
    """``w(m, n) = (m^4 + n^4) / (m^2 + n^2)`` — the space droop's weight, shared by both schemes.

    This is the *only* thing a mode's spatial pitch error depends on, up to a factor set by the
    grid: a plate's frequency ratio is ``1 - a^2 w / 3`` and a membrane's is its square root,
    ``1 - a^2 w / 6``. Square root is monotone, so **the two models order a block identically**
    and the plate's corner argument transfers to the membrane unchanged — in space. What breaks it
    is the term the implicit plate does not have (:func:`cancellation_courant`).
    """
    return float(_rs.horizon_block_weight(int(m), int(n)))
