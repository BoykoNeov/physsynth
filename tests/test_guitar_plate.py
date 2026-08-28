"""Model #5g — the plate's outline stops being a rectangle (HANDOFF §12B, the last named refusal).

The tiers, and the reason this file leans so hard on one of them: **only one detector here can
falsify anything.**

- Energy conservation is *geometry-blind*. The membrane batch recorded it (``energy ⊥ geometry``)
  and #5of recorded it again — three visibly different plates conserving at 1.5–2.1e-13 with
  fundamentals of 5.68, 3.79 and 5.99. A wrong outline conserves perfectly.
- The simply-supported branch's spectrum is the membrane's **squared** on any outline, because
  ``B = L @ L`` and ``eig(L²) = eig(L)²``. That is an identity; it cannot fail; it is asserted here
  as a *negative result*, not as a passing check.
- The rigid-body nullspace ``{1, x, y}`` is necessary, not sufficient — it looks identical for an
  assembly with the wrong boundary treatment.

So the batch's real anchor is a **derived free-circular-plate oracle**
(:func:`physsynth.analysis.modal.free_circular_plate_lambdas`), run through the *same* staircased
mask machinery the guitar uses. See ``docs/dev/guitar-plate-plan.md``.
"""

from __future__ import annotations

import math
import os
from hashlib import sha256

import numpy as np
import pytest
from helpers import arpack_v0
from scipy.sparse.linalg import eigsh

from physsynth.analysis.modal import (
    free_circular_plate_lambda_roots,
    free_circular_plate_lambdas,
    free_circular_plate_saddle_bound,
)
from physsynth.core.operators2d import (
    biharmonic_from_mask,
    cells_per_node,
    disk_mask,
    free_plate_stiffness,
    free_plate_stiffness_from_mask,
    guitar_area,
    guitar_mask,
    laplacian_from_mask,
    prune_to_area_carrying,
)
from physsynth.core.plate import Plate

NU = 0.3


# -- the outline the plate family shipped with, pinned against a change of spelling ---------------
#
# `guitar_half_width` was moved from NumPy's vectorised `sin`/`cos` to the scalar libm on
# 2026-08-28, so that the Rust port and the Python reference reach the *same* implementation and
# the mask can be compared to the bit (`docs/dev/rust-migration-plan.md` section 25.4). NumPy
# picks its transcendental routines by CPU feature, so it was never a spelling this code could
# share -- and it is the one place in the repo where a last bit is not a rounding but a live node.
#
# Everything downstream of that change is asserted against the NEW spelling, which is exactly the
# hole this test fills: a transcription slip in the new profile would be reproduced faithfully by
# the port and agreed on by every parity test. So the OLD expression is written out here, once,
# and the mask it produces is compared node for node. It is not a claim about a CPU -- if the two
# spellings ever disagree in a way that moves a node, that is the news, and this is where it lands.


def _outline_half_width_before_2026_08_28(t, waist, asym):
    """The pre-change vectorised profile, kept verbatim as the geometry's reference."""
    t = np.asarray(t, dtype=float)
    return (
        np.sin(np.pi * t)
        * (1.0 - waist * np.cos(4.0 * np.pi * (t - 0.5)))
        * (1.0 + asym * (t - 0.5))
    )


def _outline_mask_before_2026_08_28(X, Y, length, width, waist, asym):
    peak = float(
        _outline_half_width_before_2026_08_28(np.linspace(0.0, 1.0, 20001), waist, asym).max()
    )
    scale = 0.5 * float(width) / peak
    t = np.asarray(Y, dtype=float) / float(length)
    half = scale * _outline_half_width_before_2026_08_28(t, waist, asym)
    return (t > 0.0) & (t < 1.0) & (np.abs(X) < half)


@pytest.mark.parametrize("N", [8, 16, 20, 24, 32, 33, 40, 48, 64, 96])
@pytest.mark.parametrize(
    "waist,asym",
    # The shipped defaults, the two fixtures below, the viewer's long narrow plate, the first
    # point of its waist sweep (`waist = 0.0`, the outline with a node ONE ulp from the rim) and a
    # negative asymmetry.
    [(0.42, 0.30), (0.97, 0.30), (0.88, 0.0), (0.0, 0.0), (0.60, -0.30)],
)
@pytest.mark.parametrize("Lx,Ly", [(0.37, 0.48), (0.15, 0.70)])
def test_the_scalar_libm_spelling_moved_no_node_of_any_shipped_outline(Lx, Ly, waist, asym, N):
    h = Lx / N
    Ny = max(int(round(Ly / h)), 1)
    Ly_snapped = Ny * h
    X, Y = np.meshgrid(np.linspace(0.0, Lx, N + 1), np.linspace(0.0, Ly_snapped, Ny + 1))
    X = X - 0.5 * Lx

    now = guitar_mask(X, Y, Ly_snapped, Lx, waist, asym)
    before = _outline_mask_before_2026_08_28(X, Y, Ly_snapped, Lx, waist, asym)
    moved = int(np.count_nonzero(now != before))
    assert moved == 0, (
        f"Lx={Lx} waist={waist} asym={asym} N={N}: {moved} node(s) changed when the outline "
        "profile moved to the scalar libm. The plate that ships is no longer the plate that was "
        "validated -- and no energy, nullspace or spectrum bar can see it."
    )
    # Both prunes too: a node dropped for carrying no area is a geometry decision as much as a
    # node that failed the outline test, and `prune_to_area_carrying` iterates, so one moved node
    # can take a neighbour with it.
    assert np.array_equal(
        prune_to_area_carrying(now)[0], prune_to_area_carrying(before)[0]
    )
DRIFT_TOL = 1e-10  # tier 1: the same acceptance bar as every other resonator

# The #5o 7-grid survey, reused verbatim. It deliberately contains grids where the *supported*
# branch's two assemblies agree bit-for-bit and grids where they differ by ~2e-16, so asserting a
# collapse only on friendly grids would pass while proving nothing.
GRIDS = [
    (12, 12, 1.0 / 12), (24, 24, 1.0 / 24), (20, 14, 0.05), (17, 17, 1.0 / 17),
    (13, 9, 0.62 / 13), (16, 16, 0.7 / 16), (11, 7, 0.31 / 11),
]
# nu-derived split, then two genuine grain splits (a mild one and a strongly grained one).
SPLITS = [None, (1.0, 0.5, 0.35, 0.22), (2.4, 0.13, 0.15, 0.30)]


def _disk(N: int, a: float = 1.0):
    """Pruned staircased disk mask of radius ``a`` on an ``N``-cell bounding box, plus ``h``."""
    h = 2.0 * a / N
    xs = (np.arange(N + 1) - N / 2.0) * h
    X, Y = np.meshgrid(xs, xs)
    mask, _ = prune_to_area_carrying(disk_mask(X, Y, a))
    return mask, h


def _guitar(N: int, length: float = 0.48, width: float = 0.37):
    h = width / N
    nx, ny = int(round(width / h)), int(round(length / h))
    xs = (np.arange(nx + 1) - nx / 2.0) * h
    ys = np.arange(ny + 1) * h
    X, Y = np.meshgrid(xs, ys)
    mask, dropped = prune_to_area_carrying(guitar_mask(X, Y, length, width))
    return mask, h, dropped


def _fingerprint(mask, h):
    """Everything needed to tell "the geometry differed" from "the eigensolve did".

    Added 2026-08-27 because this file's degenerate-pair bar failed twice on the CI runner, with a
    *different* wrong value each time, while the identical invocation on the development machine
    was stable and correct and the two geometry paths hashed the same there. A bar that fails has
    to say what it saw, or the next reading costs another run to obtain.
    """
    K, W, _ = free_plate_stiffness_from_mask(mask, h, NU)
    Kc = K.tocsc()
    mu = np.sort(
        eigsh(Kc, k=8, M=W.tocsc(), sigma=-1e-8, which="LM", return_eigenvectors=False,
              v0=arpack_v0(Kc))
    )
    sym = float(abs(Kc - Kc.T).max()) if Kc.nnz else 0.0
    return "".join(
        "\n    " + line
        for line in (
            f"live nodes {int(mask.sum())}, mask sha {sha256(mask.tobytes()).hexdigest()[:16]}",
            f"K nnz {K.nnz}, max abs(K - K.T) {sym:.3e}, "
            f"indices sha {sha256(Kc.indices.tobytes()).hexdigest()[:16]}",
            f"mu[0:8] {np.array2string(mu, precision=6, max_line_width=200)}",
            f"PHYSSYNTH_RS={os.environ.get('PHYSSYNTH_RS', 'unset')}",
        )
    )


def _elastic_lambdas(mask, h, n_modes, a2, nu=NU, **grain):
    """Lowest ``n_modes`` elastic ``Lambda = omega a²/kappa``, with the 3 rigid-body modes skipped.

    Also asserts the PSD tier on the way past, because the eigensolve is already paid for.
    """
    K, W, _ = free_plate_stiffness_from_mask(mask, h, nu, **grain)
    mu = np.sort(
        eigsh(K.tocsc(), k=n_modes + 4, M=W.tocsc(), sigma=-1e-8, which="LM",
              return_eigenvectors=False, v0=arpack_v0(K))
    )
    # The bar is RELATIVE and it has to be. K is PSD with a 3-D nullspace, so mu[0:3] come out of
    # shift-invert with arbitrary sign, and their size grows with the problem: measured
    # max|mu[0:3]|/mu[3] = 1.6e-10 at N=80, 6.3e-11 at N=120, 9.4e-9 at N=180 on the guitar. An
    # absolute tolerance, or a 1e-9 relative one, looks safe and fails at the finest grid.
    assert mu[3] > 0.0, f"no positive elastic mode: mu[3] = {mu[3]:.3e}"
    assert np.abs(mu[:3]).max() < 1e-6 * mu[3], (
        f"the rigid-body modes are not clean relative to the first elastic one: "
        f"max|mu[0:3]|/mu[3] = {np.abs(mu[:3]).max() / mu[3]:.3e}"
    )
    return a2 * np.sqrt(np.clip(mu[3:3 + n_modes], 0.0, None))


# =====================================================================================
# The assembly: the mask generalisation is a strict SUPERSET, measured bit-for-bit
# =====================================================================================


def test_masked_assembly_reproduces_the_rectangle_bit_for_bit():
    """84 cases: 7 grids x 4 values of ``nu`` x 3 grain splits, and every one is a printed ``0.0``.

    This is what licenses ``free_plate_stiffness`` to be a wrapper over the masked routine rather
    than a second implementation — every shipped free-plate number (#5b's Leissa match at 0.01%,
    #5of's four probes, the cymbal in the room) is preserved *by construction*.

    It did not hold on the first attempt, and the way it failed is the reusable part: the twist
    operator is a **product of two forward differences**, so its coefficient is ``(1/h)*(1/h)`` and
    **not** ``1/(h*h)``. Those differ in the last digit whenever ``h`` is not exactly representable,
    which showed up on exactly one grid of this survey (``h = 0.05``), for every ``nu`` and every
    split, and on none of the other six. A single-grid check reports success.
    """
    for Nx, Ny, h in GRIDS:
        mask = np.ones((Ny + 1, Nx + 1), dtype=bool)
        for nu in (0.3, 0.0, 0.49, -0.5):
            for split in SPLITS:
                kw = (
                    {}
                    if split is None
                    else dict(grain_x=split[0], grain_y=split[1],
                              grain_coupling=split[2], grain_torsion=split[3])
                )
                K_kron, W_kron, im_kron = free_plate_stiffness(Nx, Ny, h, nu, **kw)
                K_mask, W_mask, im_mask = free_plate_stiffness_from_mask(mask, h, nu, **kw)
                dK = abs(K_kron - K_mask).max()
                assert dK == 0.0, (
                    f"not bit-identical at {Nx}x{Ny} h={h!r} nu={nu} split={split}: "
                    f"max|dK| = {dK:.3e}"
                )
                assert np.array_equal(W_kron.diagonal(), W_mask.diagonal())
                assert np.array_equal(im_kron, im_mask), "the rectangle's index_map moved"


def test_area_weight_is_the_trapezoidal_rule_restated():
    """``h²·(live cells)/4`` is not a new convention — it is ``h²``, ``h²/2``, ``h²/4`` restated."""
    mask = np.ones((6, 8), dtype=bool)
    counts = cells_per_node(mask)
    assert counts[3, 4] == 4 and counts[0, 4] == 2 and counts[0, 0] == 1
    _, W, _ = free_plate_stiffness_from_mask(mask, 0.05, NU)
    w = W.diagonal().reshape(mask.shape)
    h2 = 0.05 * 0.05
    assert w[3, 4] == h2 and w[0, 4] == 0.5 * h2 and w[0, 0] == 0.25 * h2


# =====================================================================================
# The trap: a curved outline makes nodes that carry no area
# =====================================================================================


def test_a_curved_outline_produces_massless_nodes_and_the_prune_removes_them():
    """Without the prune the free plate's mass matrix is **singular** and ``A`` cannot be factored.

    Not a tolerance question — the area weight of a one-node spike is exactly ``0``. Two such nodes
    are enough. The guitar outline produces them at every grid coarse enough to be interesting.
    """
    h = 0.37 / 16
    nx, ny = int(round(0.37 / h)), int(round(0.48 / h))
    xs = (np.arange(nx + 1) - nx / 2.0) * h
    ys = np.arange(ny + 1) * h
    X, Y = np.meshgrid(xs, ys)
    raw = guitar_mask(X, Y, 0.48, 0.37)
    assert (cells_per_node(raw)[raw] == 0).sum() > 0, "expected massless spikes on a coarse outline"

    pruned, dropped = prune_to_area_carrying(raw)
    assert dropped > 0
    assert (cells_per_node(pruned)[pruned] == 0).sum() == 0
    _, W, _ = free_plate_stiffness_from_mask(pruned, h, NU)
    assert W.diagonal().min() > 0.0, "a pruned mask must have a strictly positive mass diagonal"


def test_prune_is_idempotent_and_reaches_a_fixed_point():
    """Dropping a node can orphan its neighbour, so the rule is a fixed point, not a single pass."""
    for N in (16, 24, 32, 40, 56):
        mask, _, _ = _guitar(N)
        again, dropped_again = prune_to_area_carrying(mask)
        assert dropped_again == 0 and np.array_equal(again, mask)


@pytest.mark.parametrize("N", [20, 28, 40, 56, 80])
def test_every_pruned_node_lies_at_the_rim(N):
    """The prune rule is *topological*, so it must be checked *geometrically*.

    "Touches no live cell" says nothing about where the node is: on a coarse grid with a deep waist
    the same condition can fire in the middle of the plate. That is a silent geometry change, and
    energy, nullspace and spectrum all survive it — which is why this assert exists at all.

    **The measured depth is asserted here, not merely the absence of an exception.** ``Plate``
    raises only when the bar is *violated*, so on a passing grid nothing about depth is observed: a
    sign error in ``_depth_inside_outline``, or an early return in the checker, would leave every
    grid green. Both bounds matter — the upper one is the bar, and the **lower** one proves the
    check ran over a non-empty set rather than vacuously.

    Measured on the shipped path: 0.750, 0.733, 0.712, 0.704, 0.703 ``h`` at N = 20…80. (The plan's
    probe reported a wider 0.53–0.99 ``h`` because it built the outline on the *nominal* length;
    ``Plate`` snaps ``Ly`` to whole cells first, which moves which nodes are spikes.)
    """
    p = Plate(
        Lx=0.37, Ly=0.48, kappa=1.0, rho=2.0, fs=20000.0, N=N, boundary="free", domain="guitar"
    )
    assert p.n_pruned > 0, "nothing was pruned, so the rim check ran over an empty set"
    assert p.prune_depth_max > 0.0, "a pruned node was not inside the outline at all"
    assert p.prune_depth_max <= 1.0001 * p.h, (
        f"deepest pruned node sat {p.prune_depth_max / p.h:.3f} h inside the outline"
    )
    assert 0.6 < p.prune_depth_max / p.h < 0.85, (
        f"depth {p.prune_depth_max / p.h:.3f} h is outside the measured 0.70-0.75 band; the "
        f"outline or the prune rule has moved"
    )
    assert p.mask.sum() == p.n_live


def test_a_pinched_outline_is_refused_rather_than_silently_two_plates():
    """A deep waist on a coarse grid separates the bouts — two plates, a 6-D nullspace, no error.

    The match is pinned to the **connectivity** message specifically. A looser alternation would
    also accept the mid-plate-prune refusal, and then this test would pass without the pinch path
    ever running — measured, it is the connectivity check that fires here, at every waist in
    0.90–0.99 and every N in 8–12.
    """
    with pytest.raises(ValueError, match="disconnected pieces"):
        Plate(
            Lx=0.37, Ly=0.48, kappa=1.0, rho=2.0, fs=20000.0, N=8,
            boundary="free", domain="guitar", waist=0.97,
        )


# =====================================================================================
# The oracle: a DERIVED free circular plate, and its three self-checks
# =====================================================================================


def test_derived_frequency_equation_admits_the_rigid_body_modes():
    """``W = 1`` (n=0) and ``W = rho`` (n=1) must annihilate both free-edge lines exactly.

    A translation and a tilt cost a free plate nothing. This is the cheapest check on the derivation
    and it catches a sign error in either line.
    """
    for n, W, W1, W2, W3 in ((0, 1.0, 0.0, 0.0, 0.0), (1, 1.0, 1.0, 0.0, 0.0)):
        moment = W2 + NU * (W1 - n * n * W)
        shear = W3 + W2 - (1.0 + n * n * (2.0 - NU)) * W1 + n * n * (3.0 - NU) * W
        assert moment == 0.0 and shear == 0.0, f"n={n} rigid-body mode is not a root"


def test_saddle_bound_brackets_the_derived_fundamental():
    """A one-term Rayleigh bound containing **no Bessel function** — so it cannot share a bug.

    ``w = xy`` on a disk is orthogonal to ``{1, x, y}`` and has ``w_xx = w_yy = 0``, giving
    ``Lambda_1 <= sqrt(48(1-nu))``. The overshoot lands at 8.18% against the derived fundamental —
    the same character as the free *rectangle*'s 5.4% twist bound, on the other outline.
    """
    lam, _ = free_circular_plate_lambdas(NU, 1)
    bound = free_circular_plate_saddle_bound(NU)
    assert lam[0] < bound, (
        f"the fundamental {lam[0]:.5f} exceeds its own Rayleigh bound {bound:.5f}"
    )
    overshoot = bound / lam[0] - 1.0
    assert 0.05 < overshoot < 0.12, f"one-term overshoot {100 * overshoot:.2f}% is out of character"


def test_every_derived_root_returns_lambda_to_the_fourth_in_the_plate_energy():
    """The decisive check: each root's own eigenfunction, put back into the bending energy.

    A genuine mode satisfies ``P/M = lam⁴`` — note the **fourth** power, since ``omega = kappa k²``.
    Checking against ``lam²`` instead makes every correct root look like a 2.3x error, which is
    exactly what it did the first time. Every root passes, so the equation has no spurious roots.
    """
    from scipy.special import iv as iv_
    from scipy.special import ivp, jv, jvp

    rho = (np.arange(40000) + 0.5) / 40000
    for n in (0, 1, 2, 3):
        for lam in free_circular_plate_lambda_roots(NU, n, lam_max=9.0)[:2]:
            m = np.empty((2, 2))
            for col, f in enumerate((jvp, ivp)):
                d0, d1 = f(n, lam, 0), f(n, lam, 1) * lam
                d2, d3 = f(n, lam, 2) * lam**2, f(n, lam, 3) * lam**3
                m[0, col] = d2 + NU * (d1 - n * n * d0)
                m[1, col] = (
                    d3 + d2 - (1.0 + n * n * (2.0 - NU)) * d1 + n * n * (3.0 - NU) * d0
                )
            m[:, 1] *= math.exp(-lam) if lam > 1.0 else 1.0
            A, B = np.linalg.svd(m)[2][-1]
            B *= math.exp(-lam) if lam > 1.0 else 1.0

            z = lam * rho
            W = A * jv(n, z) + B * iv_(n, z)
            W1 = lam * (A * jvp(n, z, 1) + B * ivp(n, z, 1))
            W2 = lam**2 * (A * jvp(n, z, 2) + B * ivp(n, z, 2))
            lap = W2 + W1 / rho - n * n * W / rho**2
            gauss = (
                W2 * (W1 / rho - n * n * W / rho**2)
                - n * n * (W1 / rho - W / rho**2) ** 2
            )
            quotient = float(
                np.sum((lap**2 - 2.0 * (1.0 - NU) * gauss) * rho) / np.sum(W * W * rho)
            )
            assert abs(quotient / lam**4 - 1.0) < 3e-3, (
                f"n={n} lam={lam:.5f}: the Rayleigh quotient is {quotient:.4f}, not lam^4 = "
                f"{lam ** 4:.4f} — this root is not a mode"
            )


# =====================================================================================
# The anchor: the masked assembly against that oracle, on a staircased curved rim
# =====================================================================================


def test_staircased_disk_matches_the_derived_oracle_and_converges():
    """The one tier that can falsify the assembly. Seven modes, converging **from above** at O(h).

    Two things this pins, both of which would otherwise be guessed wrong:

    - The rate is **O(h)**, not O(h²). Staircasing taxes a 4th-order operator the same first order
      it taxed the membrane's Bessel match.
    - The multiplicity matters. Every ``n >= 1`` mode is a degenerate *pair*; comparing the discrete
      spectrum against a list naming each pair once misaligns everything past the first entry and
      reads as 26–42% errors that grow *worse* under refinement. That is a comparison failure with
      no bug behind it.
    """
    target, _ = free_circular_plate_lambdas(NU, 7)
    errs = {}
    for N in (32, 64, 128):
        mask, h = _disk(N)
        lam = _elastic_lambdas(mask, h, 7, a2=1.0)
        assert np.all(lam > target), "a staircased disk is SMALLER, so it must ring sharp"
        errs[N] = float(np.abs(lam / target - 1.0).mean())

    assert errs[32] < 0.12, f"coarse disk is off by {100 * errs[32]:.1f}%"
    assert errs[128] < 0.03, f"fine disk is off by {100 * errs[128]:.1f}%"
    # O(h): halving h must buy a factor clearly better than 1 and nowhere near the 4 of O(h²).
    for coarse, fine in ((32, 64), (64, 128)):
        ratio = errs[coarse] / errs[fine]
        assert 1.25 < ratio < 2.6, (
            f"N {coarse}->{fine} improved by {ratio:.2f}x; O(h) predicts ~2 and O(h²) ~4"
        )


def test_the_shipped_circle_path_matches_the_derived_oracle():
    """The oracle must anchor ``Plate(domain="circle")`` itself, not only the bare mask helper.

    The test above validates the *assembly* through a locally-built mask. That leaves the shipped
    constructor — its own grid centring, its own snapping, its own prune — anchored by nothing but
    energy and passivity, which this module's docstring already calls geometry-blind. So the
    constructor is tied to the oracle directly.

    It also settles **odd** ``N``: the disk's centre falls between nodes there rather than on one,
    and the spectrum is unaffected (7.60% at N=33 against 8.55% at N=32), so odd grids are allowed
    rather than rejected.
    """
    target, _ = free_circular_plate_lambdas(NU, 7)
    prev = None
    for N in (32, 33, 64, 128):
        p = Plate(
            Lx=2.0, Ly=2.0, kappa=1.0, rho=2.0, fs=200000.0, N=N, boundary="free", domain="circle"
        )
        mu = np.sort(
            eigsh(p.K.tocsc(), k=11, M=p.W.tocsc(), sigma=-1e-8, which="LM",
                  return_eigenvectors=False, v0=arpack_v0(p.K))
        )
        lam = (0.5 * p.Lx) ** 2 * np.sqrt(np.clip(mu[3:10], 0.0, None))
        assert np.all(lam > target), "a staircased disk is SMALLER, so it must ring sharp"
        err = float(np.abs(lam / target - 1.0).mean())
        # The area deficit is the leading term, and it shows: the two track each other closely.
        assert abs(err + p.area_deficit) < 0.012, (
            f"N={N}: mean error {100 * err:.3f}% and area deficit {100 * p.area_deficit:.3f}% "
            f"have come apart"
        )
        if prev is not None and N != 33:  # 33 refines 32 by only 3%, so it is not a rate step
            assert err < prev
        if N != 33:
            prev = err
    assert prev < 0.03, f"the finest disk is off by {100 * prev:.2f}%"


def test_the_degenerate_pairs_split_and_the_exact_answer_is_zero():
    """A **zero-valued** oracle: a square grid relates a pair's two members by no symmetry.

    Worth more than an absolute one, because the target needs no table. Measured 0.69%, 1.01%,
    0.06%, 0.52%, 0.17%, 0.013% at N = 24…128 — shrinking but **not monotone**, so this asserts a
    ceiling that shrinks with ``h`` and never monotonicity.
    """
    for N, ceiling in ((32, 0.02), (64, 0.012), (128, 0.006)):
        mask, h = _disk(N)
        lam = _elastic_lambdas(mask, h, 2, a2=1.0)
        split = 2.0 * abs(lam[1] - lam[0]) / (lam[1] + lam[0])
        assert split < ceiling, (
            f"N={N}: the m=2 pair splits by {100 * split:.3f}%" + _fingerprint(mask, h)
        )


# =====================================================================================
# The supported branch: a NEGATIVE result, stated as the identity it is
# =====================================================================================


def test_a_supported_curved_plate_is_the_membrane_squared_and_therefore_says_nothing():
    """``B = L @ L`` on any mask, so ``eig(B) = eig(-L)²``. This **cannot fail**.

    It is asserted precisely so the batch does not mistake it for evidence. The guitar outline tells
    the simply-supported plate nothing the drumhead did not already know, which is why ``Plate``
    refuses ``domain != 'rectangle'`` on the supported branch rather than offering a model surface
    with no content.
    """
    mask, h, _ = _guitar(24)
    L, _ = laplacian_from_mask(mask, h)
    B, _ = biharmonic_from_mask(mask, h)
    lam_L = np.sort(-eigsh(L, k=6, which="SM", return_eigenvectors=False, v0=arpack_v0(L)))
    lam_B = np.sort(eigsh(B, k=6, which="SM", return_eigenvectors=False, v0=arpack_v0(B)))
    assert np.allclose(lam_B, lam_L**2, rtol=1e-8), "B = L @ L is not an identity here"

    with pytest.raises(ValueError, match="offered on boundary='free' only"):
        Plate(
            Lx=0.37, Ly=0.48, kappa=1.0, rho=2.0, fs=20000.0, N=16,
            boundary="supported", domain="guitar",
        )


# =====================================================================================
# Tier 1: the energy ledger, which is here to be a REGRESSION test, not evidence
# =====================================================================================


@pytest.mark.parametrize("domain", ["guitar", "circle"])
def test_lossless_outline_plate_conserves_energy(domain):
    """The same 1e-10 acceptance bar as every other resonator — and it proves nothing about shape.

    Included because a masked assembly could plausibly break *symmetry*, which the ledger does see.
    It does not see geometry: ``docs/dev/guitar-plate-plan.md`` §5 is explicit that a wrong outline
    conserves perfectly, so this is a regression tier and not the batch's evidence.
    """
    p = Plate(
        Lx=0.37, Ly=0.48 if domain == "guitar" else 0.37, kappa=2.0, rho=2.0,
        fs=20000.0, N=24, boundary="free", domain=domain,
    )
    field = np.zeros(p.mask.shape)
    ny, nx = field.shape
    field[ny // 3, nx // 2] = 1e-3
    p.set_state(field)
    e0 = p.energy()
    drift = 0.0
    for _ in range(400):
        p.step()
        drift = max(drift, abs(p.energy() - e0) / e0)
    assert drift < DRIFT_TOL, f"{domain}: energy drifted {drift:.3e}"


@pytest.mark.parametrize("domain", ["guitar", "circle"])
def test_lossy_outline_plate_is_passive(domain):
    p = Plate(
        Lx=0.37, Ly=0.48 if domain == "guitar" else 0.37, kappa=2.0, rho=2.0,
        fs=20000.0, N=24, sigma=4.0, boundary="free", domain=domain,
    )
    field = np.zeros(p.mask.shape)
    ny, nx = field.shape
    field[ny // 3, nx // 2] = 1e-3
    p.set_state(field)
    prev = p.energy()
    for _ in range(300):
        p.step()
        now = p.energy()
        assert now <= prev * (1.0 + 1e-12), "a lossy plate gained energy"
        prev = now


# =====================================================================================
# What the plate REPORTS about its own outline
# =====================================================================================


def test_the_area_deficit_is_reported_and_shrinks_under_refinement():
    """The plate states how much of its outline the mask actually carries. Reported, never applied.

    This is the leading term of the staircase error: the frequency error tracks it
    mode-independently, so a staircased plate behaves like a well-modelled slightly *smaller* one.
    Dividing it out silently would make a coarse plate look converged, so ``Plate`` exposes it and
    changes nothing.
    """
    prev = None
    for N in (20, 40, 80):
        p = Plate(
            Lx=0.37, Ly=0.48, kappa=1.0, rho=2.0, fs=20000.0, N=N,
            boundary="free", domain="guitar",
        )
        assert p.area_deficit < 0.0, "a staircased mask cannot carry MORE area than its outline"
        # the plate's outline uses its SNAPPED length, not the requested one
        assert p.outline_area == pytest.approx(guitar_area(p.Ly, p.Lx), rel=1e-9)
        if prev is not None:
            assert abs(p.area_deficit) < abs(prev), "the deficit must shrink under refinement"
        prev = p.area_deficit
    assert abs(prev) < 0.06


def test_a_rectangle_still_prunes_nothing_and_carries_its_whole_area():
    p = Plate(Lx=0.4, Ly=0.3, kappa=1.0, rho=2.0, fs=20000.0, N=20, boundary="free")
    assert p.n_pruned == 0
    assert p.area_deficit == pytest.approx(0.0, abs=1e-14)
    assert p.domain == "rectangle"
