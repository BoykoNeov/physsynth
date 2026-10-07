"""Shared test fixtures and physical parameters for the validation harness.

A single canonical string (L=1 m, T=200 N, rho=0.005 kg/m -> c=200 m/s, f1=100 Hz) is reused so the
expected frequencies are easy to reason about.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from physsynth.analysis import modal
from physsynth.analysis.horizon import pitch_horizon
from physsynth.core.membrane import Domain, Membrane
from physsynth.core.plate import THETA_DEFAULT as PLATE_THETA_DEFAULT  # noqa: F401 (re-export)
from physsynth.core.plate import Plate
from physsynth.core.string_damped import DampedStiffString
from physsynth.core.string_stiff import THETA_DEFAULT

L_DEFAULT = 1.0
T_DEFAULT = 200.0
RHO_DEFAULT = 0.005  # -> c = sqrt(T/rho) = 200 m/s, fundamental f1 = c/(2L) = 100 Hz

# Membrane (model #4): areal density chosen so c = sqrt(T/rho) = 200 m/s again (same nice numbers).
RHO_AREAL_DEFAULT = 0.005  # kg/m^2  -> c = 200 m/s with T_DEFAULT = 200 N/m
RADIUS_DEFAULT = 0.5  # m  -> circular fundamental f_01 = c*j_{0,1}/(2*pi*a) ~ 153.1 Hz
KAPPA_DEFAULT = 2.0  # -> B = pi^2 kappa^2 / (c^2 L^2) ~ 9.87e-4, a piano-ish inharmonicity

# Plate (model #5): stiffness kappa = sqrt(D/rho_s) (m^2/s). On the 1x1 m plate the fundamental is
# f_11 = (pi/2) kappa [1 + 1] = pi kappa ~ 62.8 Hz at kappa = 20, with modes spreading
# quadratically.
KAPPA_PLATE_DEFAULT = 20.0
# Plate "Courant" number mu = kappa k / h^2: the EXPLICIT-scheme stability parameter (explicit needs
# mu <= 1/4). The implicit theta-scheme has no limit, so the default sits well past 1/4 -- a regime
# the explicit plate could not run. make_plate solves fs from mu: fs = kappa / (mu h^2).
MU_PLATE_DEFAULT = 2.0


# ARPACK start vector. `eigsh` without `v0` draws a RANDOM one, so every eigensolve in the suite is
# non-reproducible run to run -- measured on the free-free beam: the elastic eigenvalues wobble by
# ~1e-12 relative and their eigenvectors by ~5e-11, and the two RIGID-BODY modes (mu ~ 0, exactly
# degenerate) come back as an arbitrary basis of the {1, x} nullspace, differing by ~1e-1. That is
# harmless against a 5-cent bar, but an oracle that is not bit-reproducible reads as a port bug the
# first time a comparison is tightened, and an eigenvector fed to `set_state` makes a genuinely
# different trajectory (rust-migration-plan.md Sec 7). Pinned here so it cannot.
ARPACK_SEED = 20260828


def arpack_v0(op) -> NDArray[np.float64]:
    """A fixed ARPACK start vector sized for ``op`` (a square matrix, or its dimension).

    Uniform doubles from a seeded PCG64 stream: deterministic across platforms and NumPy versions,
    generic (no symmetry, so it is orthogonal to no eigenvector -- unlike the obvious ``arange``,
    which is antisymmetric about the centre of a symmetric operator and would starve every
    symmetric mode), and free of any transcendental, whose CPU-dispatched NumPy loop would put the
    start vector itself back in the class of things that vary by machine (Sec 22.1).
    """
    n = int(op) if isinstance(op, (int, np.integer)) else int(op.shape[0])
    return np.random.default_rng(ARPACK_SEED).random(n)


def wave_speed(T: float = T_DEFAULT, rho: float = RHO_DEFAULT) -> float:
    return float(np.sqrt(T / rho))


def make_damped_string(
    *,
    N: int = 100,
    lam: float = 1.0,
    kappa: float = KAPPA_DEFAULT,
    sigma0: float = 0.0,
    sigma1: float = 0.0,
    theta: float = THETA_DEFAULT,
    L: float = L_DEFAULT,
    T: float = T_DEFAULT,
    rho: float = RHO_DEFAULT,
) -> DampedStiffString:
    """Build a damped stiff string (model #3) at Courant number ``lam`` via ``fs = c N / (L lam)``.

    ``sigma0`` is the frequency-independent loss (model #2's ``sigma``), ``sigma1`` the
    frequency-dependent loss (the new term). ``lam > 1`` is allowed (unconditionally stable).
    """
    c = wave_speed(T, rho)
    fs = c * N / (L * lam)
    return DampedStiffString(
        L=L, T=T, rho=rho, fs=fs, N=N, kappa=kappa, sigma0=sigma0, sigma1=sigma1, theta=theta
    )


# =====================================================================================
# The resolution horizon: what stayed here, and why only this
# =====================================================================================
#
# The seven primitives that used to live here -- `pitch_error_cents`, `pitch_horizon`,
# `sinc_horizon_fraction`, `mode_family`, `mode_block`, `cancellation_courant`, `block_weight` --
# were promoted into `physsynth/analysis/horizon.py` and are now Rust, per
# `docs/dev/resolution-horizon-plan.md` section 11. Callers import them from the package
# directly;
# there is deliberately NO re-export here. A re-export would be an unused import that only
# `__all__` keeps alive (a `ruff --fix` hazard the migration hit as ledger #66), and worse, it
# would leave two routes to one name with nothing saying which is canonical.
#
# `spatial_operator_horizon` did NOT go with them, and section 7.7 is the reason. It takes `(N,
# kappa)` and hands back "the horizon", which reads like a general answer and is not one: it
# hardcodes `L_DEFAULT`, `wave_speed()` and a 1-D Dirichlet second difference, so it silently
# answers about the default string whatever the caller had in mind. That is exactly the shape
# `mode_family` was designed against -- index-side only, geometry left to the caller. A fixture
# that assumes the canonical string is honest in the test folder and would be a lie in the library,
# so it stays on this side of the line and builds its frequencies from the promoted `pitch_horizon`
# like any other caller.


def spatial_operator_horizon(N: int, kappa: float, cents: float = 5.0) -> tuple[int, bool]:
    """The horizon of the *spatial* operator alone, with no timestep in it at all.

    ``k`` appears nowhere here: this is the eigenvalue error of the second difference (and of the
    biharmonic built from it), so it is what remains when the timestep is refined to nothing. For
    the implicit theta-scheme family that makes it a genuine **floor** -- the time error and the
    space error both flatten the pitch, so they compound and no sample rate passes this line. For
    the explicit family it is not a floor at all: there the time error is *sharp* and cancels the
    space droop exactly at the magic Courant number.

    About the canonical string, always. See the note above for why that keeps it here.
    """
    h = L_DEFAULT / N
    modes = np.arange(1, N)
    p2_disc = np.array([modal.dirichlet_axis_eigenvalue(int(m), L_DEFAULT, h) for m in modes])
    p2_cont = (modes * np.pi / L_DEFAULT) ** 2
    c = wave_speed()
    w_disc = np.sqrt(c * c * p2_disc + kappa * kappa * p2_disc * p2_disc)
    w_cont = np.sqrt(c * c * p2_cont + kappa * kappa * p2_cont * p2_cont)
    return pitch_horizon(w_disc, w_cont, cents)


def make_membrane(
    *,
    domain: Domain,
    N: int,
    lam: float = 0.7,
    sigma: float = 0.0,
    T: float = T_DEFAULT,
    rho: float = RHO_AREAL_DEFAULT,
    Lx: float = 1.0,
    Ly: float = 1.0,
    radius: float = RADIUS_DEFAULT,
) -> Membrane:
    """Build a membrane whose Courant number is exactly ``lam`` by choosing ``fs = c / (lam h)``.

    ``h`` is fixed by the geometry and ``N`` (``Lx/N`` for a rectangle, ``2 radius/N`` for a disk),
    so the sample rate is solved for to hit the target ``lam``. The 2D CFL ceiling is ``1/sqrt(2)``.
    """
    c = float(np.sqrt(T / rho))
    h = Lx / N if domain == "rectangle" else 2.0 * radius / N
    fs = c / (lam * h)
    if domain == "rectangle":
        return Membrane(domain=domain, T=T, rho=rho, fs=fs, N=N, Lx=Lx, Ly=Ly, sigma=sigma)
    return Membrane(domain=domain, T=T, rho=rho, fs=fs, N=N, radius=radius, sigma=sigma)


def make_plate(
    *,
    N: int,
    mu: float = MU_PLATE_DEFAULT,
    kappa: float = KAPPA_PLATE_DEFAULT,
    sigma: float = 0.0,
    theta: float = THETA_DEFAULT,
    Lx: float = 1.0,
    Ly: float = 1.0,
    rho: float = RHO_AREAL_DEFAULT,
) -> Plate:
    """Build a simply-supported plate at plate-Courant number ``mu = kappa k / h^2``.

    ``h = Lx/N`` is fixed by the geometry, so the sample rate is solved for to hit the target
    ``mu``: ``fs = kappa / (mu h^2)``. There is no CFL ceiling (the implicit theta-scheme is
    unconditionally stable for ``theta >= 1/4``), so ``mu`` above the explicit bound ``1/4`` is a
    feature -- a coarse-grid / large-timestep regime the explicit plate could not run. Smaller
    ``mu`` means a finer timestep (less numerical dispersion), used for the tight modal tests.
    """
    h = Lx / N
    fs = kappa / (mu * h * h)
    return Plate(Lx=Lx, Ly=Ly, kappa=kappa, rho=rho, fs=fs, N=N, sigma=sigma, theta=theta)
