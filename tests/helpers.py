"""Shared test fixtures and physical parameters for the validation harness.

A single canonical string (L=1 m, T=200 N, rho=0.005 kg/m -> c=200 m/s, f1=100 Hz) is reused so the
expected frequencies are easy to reason about.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import eigsh

from physsynth.analysis import modal, spectrum
from physsynth.core.beam import FreeBeam
from physsynth.core.body import ModalBody
from physsynth.core.bore import C0_AIR, RHO0_AIR, Bore
from physsynth.core.bow import BowedString
from physsynth.core.collision import BarrierString
from physsynth.core.connection import (
    StringBodyBridge,
    StringPlateBridge,
    SympatheticStrings,
)
from physsynth.core.engine import simulate
from physsynth.core.mallet import MalletMembrane, MalletWall
from physsynth.core.membrane import Domain, Membrane
from physsynth.core.plate import Plate
from physsynth.core.radiation import AirRadiation, RadiatedBody
from physsynth.core.reed import ReedBore
from physsynth.core.string_damped import DampedStiffString
from physsynth.core.string_geometric import GeometricString
from physsynth.core.string_ideal import Boundary, IdealString
from physsynth.core.string_nonlinear import TENSION_TOL_DEFAULT, TensionModulatedString
from physsynth.core.string_stiff import THETA_DEFAULT, StiffString

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

# Free-free beam (model #5b-pre): stiffness kappa = sqrt(E I / (rho A)) (m^2/s). Same plate-Courant
# parameter mu = kappa k / h^2 (no CFL limit -- implicit theta-scheme). rho is the *linear* density
# (kg/m), so RHO_DEFAULT carries over. The fundamental is f_1 = kappa (beta_1 L)^2 / (2 pi L^2) with
# beta_1 L ~ 4.730, i.e. ~ 71.3 Hz at kappa = 20, L = 1.
KAPPA_BEAM_DEFAULT = 20.0
MU_BEAM_DEFAULT = 2.0


def wave_speed(T: float = T_DEFAULT, rho: float = RHO_DEFAULT) -> float:
    return float(np.sqrt(T / rho))


def make_string(
    *,
    N: int = 100,
    lam: float = 1.0,
    boundary: Boundary = "fixed",
    sigma: float = 0.0,
    L: float = L_DEFAULT,
    T: float = T_DEFAULT,
    rho: float = RHO_DEFAULT,
) -> IdealString:
    """Build a string whose Courant number is exactly ``lam`` by choosing ``fs = c N / (L lam)``."""
    c = wave_speed(T, rho)
    fs = c * N / (L * lam)
    return IdealString(L=L, T=T, rho=rho, fs=fs, N=N, boundary=boundary, sigma=sigma)


def make_stiff_string(
    *,
    N: int = 100,
    lam: float = 1.0,
    kappa: float = KAPPA_DEFAULT,
    sigma: float = 0.0,
    theta: float = THETA_DEFAULT,
    L: float = L_DEFAULT,
    T: float = T_DEFAULT,
    rho: float = RHO_DEFAULT,
) -> StiffString:
    """Build a stiff string at Courant number ``lam`` via ``fs = c N / (L lam)``.

    Unlike :func:`make_string`, ``lam > 1`` is allowed (the implicit scheme is unconditionally
    stable) -- a coarse-grid / large-timestep regime the explicit stiff scheme could not run.
    """
    c = wave_speed(T, rho)
    fs = c * N / (L * lam)
    return StiffString(
        L=L, T=T, rho=rho, fs=fs, N=N, kappa=kappa, sigma=sigma, theta=theta
    )


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


def measure_mode_decay_factor(
    m: int,
    *,
    N: int,
    lam: float,
    kappa: float = KAPPA_DEFAULT,
    sigma0: float = 0.0,
    sigma1: float = 0.0,
    theta: float = THETA_DEFAULT,
    steps: int = 4000,
    window: tuple[float, float] = (0.1, 0.7),
    amplitude: float = 1e-3,
) -> float:
    """Measure the per-step **energy** decay factor ``g_m`` of a single mode ``m`` (model #3).

    Initialises the string with one exact discrete eigenvector ``sin(m pi x / L)``; the field stays
    ``u(t) = q(t) phi_m`` so the energy decays (essentially ripple-free, like the lossless
    cross-time energy) as ``g_m^n``. A log-linear least-squares fit of ``E^n`` over an **interior**
    window (``window`` fractions of the run -- skipping the start, where the lossless Taylor
    ``u^{-1}`` is slightly inconsistent under damping) recovers ``ln g_m`` robustly; the fit
    averages out any small ripple. Returns ``g_m = exp(slope per step)``, comparable directly to
    :func:`physsynth.analysis.damping.discrete_damped_mode_decay`.
    """
    L = L_DEFAULT
    s = make_damped_string(
        N=N, lam=lam, kappa=kappa, sigma0=sigma0, sigma1=sigma1, theta=theta
    )
    phi = modal.mode_shape(s.x, L, m)
    s.set_state(phi * amplitude)
    res = simulate(s, num_steps=steps)
    e = res.energy

    i0 = int(window[0] * steps)
    i1 = int(window[1] * steps)
    # Clip the window to where energy is still well above the roundoff floor (avoid log of noise).
    floor = e[0] * 1e-13
    good = np.where(e[i0 : i1 + 1] > floor)[0]
    if len(good) < 8:
        raise ValueError("decay window too short / energy hit the roundoff floor; raise amplitude "
                         "or lower steps/sigma.")
    idx = np.arange(i0, i1 + 1)[good]
    slope = np.polyfit(idx, np.log(e[idx]), 1)[0]  # per-step ln-energy slope
    return float(np.exp(slope))


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


def make_free_plate(
    *,
    N: int,
    mu: float = MU_PLATE_DEFAULT,
    kappa: float = KAPPA_PLATE_DEFAULT,
    nu: float = 0.3,
    sigma: float = 0.0,
    theta: float = THETA_DEFAULT,
    a: float = 1.0,
    rho: float = RHO_AREAL_DEFAULT,
) -> Plate:
    """Build a **completely free** square plate (side ``a``) at plate-Courant number ``mu``.

    The free-edge (FFFF, curved-Chladni) counterpart of :func:`make_plate` (model #5b). Square by
    construction (``Lx = Ly = a``, no ``Ly``-snapping), which is the geometry of the Leissa anchor.
    ``h = a/N`` is fixed, so ``fs = kappa/(mu h²)`` hits the target ``mu`` (no CFL ceiling — the
    implicit theta-scheme is unconditionally stable for ``theta >= 1/4``). ``nu`` (Poisson's ratio,
    default 0.3) re-enters for free edges.
    """
    h = a / N
    fs = kappa / (mu * h * h)
    return Plate(
        Lx=a, Ly=a, kappa=kappa, rho=rho, fs=fs, N=N, sigma=sigma, theta=theta,
        boundary="free", nu=nu,
    )


def free_plate_low_eigenfrequencies(
    plate: Plate, n_modes: int, *, return_rigid: bool = False
):
    """The ``n_modes`` lowest **elastic** eigenfrequencies (Hz) of a free ``plate`` (ascending).

    Solves the generalized eigenproblem ``K φ = mu W φ`` (``mu = ω²/κ²``, the 4th-power spatial
    eigenvalue). ``K`` is only **positive-semidefinite** (the 3-dim ``{1, x, y}`` rigid-body
    nullspace), so shift-invert at ``sigma = 0`` is singular -- a small **negative** shift is used
    (``K - sigma W`` SPD). The **3** rigid-body modes (``mu ≈ 0``) are discarded; each remaining
    ``mu`` maps to ``f = kappa·sqrt(mu)/(2π)``. Degeneracy-robust (returns sorted values, so the
    square plate's degenerate pairs appear as the near-equal entries they are). With
    ``return_rigid=True`` the 3 discarded near-zero ``mu`` are also returned (a free cross-check of
    the nullspace). The free-plate analogue of :func:`beam_low_eigenfrequencies`.
    """
    n_total = n_modes + 3  # + the 3 rigid-body modes to discard
    a = plate.Lx
    mu1_est = (13.0 / (a * a)) ** 2  # lambda_1 ~ 13.47 -> a safe (< mu_1) negative shift scale
    sigma = -1e-3 * mu1_est
    mu = eigsh(
        plate.K, k=n_total, M=plate.W, sigma=sigma, which="LM", return_eigenvectors=False
    )
    mu = np.sort(mu)
    rigid, elastic = mu[:3], mu[3:n_total]
    freqs = plate.kappa * np.sqrt(np.clip(elastic, 0.0, None)) / (2.0 * np.pi)
    if return_rigid:
        return freqs, rigid
    return freqs


def make_beam(
    *,
    N: int,
    mu: float = MU_BEAM_DEFAULT,
    kappa: float = KAPPA_BEAM_DEFAULT,
    sigma: float = 0.0,
    theta: float = THETA_DEFAULT,
    L: float = L_DEFAULT,
    rho: float = RHO_DEFAULT,
) -> FreeBeam:
    """Build a free-free beam at beam-Courant number ``mu = kappa k / h^2``.

    ``h = L/N`` is fixed by the geometry, so the sample rate is solved for to hit the target ``mu``:
    ``fs = kappa / (mu h^2)`` (the 1D analogue of :func:`make_plate`). There is no CFL ceiling (the
    implicit theta-scheme is unconditionally stable for ``theta >= 1/4``), so ``mu`` above the
    explicit bound ``1/4`` is a feature. Smaller ``mu`` -> finer timestep (less dispersion), for the
    tight modal tests.
    """
    h = L / N
    fs = kappa / (mu * h * h)
    return FreeBeam(L=L, rho=rho, fs=fs, N=N, kappa=kappa, sigma=sigma, theta=theta)


def beam_low_eigenfrequencies(
    beam: FreeBeam, n_modes: int, *, return_rigid: bool = False
):
    """The ``n_modes`` lowest **elastic** spatial eigenfrequencies (Hz) of ``beam`` (ascending).

    Solves the generalized eigenproblem ``K φ = mu W φ`` (``mu = ω²/κ²``, the 4th-power spatial
    eigenvalue). ``K`` is only **positive-semidefinite** (the ``{1, x}`` rigid-body nullspace), so
    shift-invert at ``sigma = 0`` is singular -- a small **negative** shift is used instead
    (``K - sigma W`` SPD). The two ``mu ≈ 0`` rigid-body modes are discarded; each remaining ``mu``
    is mapped to the spatial frequency ``f = kappa·sqrt(mu)/(2π)``. With ``return_rigid=True`` the
    two discarded near-zero ``mu`` are also returned (a free cross-check that the nullspace is wired
    right).
    """
    n_total = n_modes + 2  # + the 2 rigid-body modes to discard
    mu1_est = (4.730041 / beam.L) ** 4  # continuum fundamental scale -> a safe (< mu_1) shift
    sigma = -1e-3 * mu1_est
    mu = eigsh(
        beam.K, k=n_total, M=beam.W, sigma=sigma, which="LM", return_eigenvectors=False
    )
    mu = np.sort(mu)
    rigid, elastic = mu[:2], mu[2:n_total]
    freqs = beam.kappa * np.sqrt(np.clip(elastic, 0.0, None)) / (2.0 * np.pi)
    if return_rigid:
        return freqs, rigid
    return freqs


def plate_low_eigenfrequencies(plate: Plate, n_modes: int) -> np.ndarray:
    """The ``n_modes`` lowest discrete modal frequencies (Hz) of ``plate`` (ascending).

    Uses shift-invert ``eigsh`` (around 0) on the SPD operator ``-L`` to get the smallest
    **Laplacian** eigenvalues ``Λ`` robustly, then maps each through
    :func:`modal.discrete_plate_eigenfrequency` (the theta-scheme's exact ``Λ -> f`` relation, with
    biharmonic stiffness ``Q = kappa² Λ²``). Degeneracy-robust: it returns sorted values, so the
    square plate's ``(m,n)<->(n,m)`` pairs appear as the near-equal entries they are.
    """
    lam_vals = eigsh(-plate.L, k=n_modes, sigma=0.0, which="LM", return_eigenvectors=False)
    lam_vals = np.sort(lam_vals)
    return np.asarray(
        modal.discrete_plate_eigenfrequency(lam_vals, plate.kappa, plate.k, plate.theta)
    )


def membrane_low_eigenfrequencies(membrane: Membrane, n_modes: int) -> np.ndarray:
    """The ``n_modes`` lowest discrete modal frequencies (Hz) of ``membrane`` (ascending).

    Uses shift-invert ``eigsh`` (around 0) on the SPD operator ``-L`` to get the smallest
    eigenvalues ``Λ`` robustly, then maps each through
    :func:`modal.discrete_membrane_eigenfrequency` (the scheme's exact ``Λ -> f`` relation).
    Degeneracy-robust: it just returns sorted values, so
    cos/sin (and square ``(m,n)<->(n,m)``) pairs appear as the near-equal entries they are.
    """
    lam_vals = eigsh(-membrane.L, k=n_modes, sigma=0.0, which="LM", return_eigenvectors=False)
    lam_vals = np.sort(lam_vals)
    return np.asarray(modal.discrete_membrane_eigenfrequency(lam_vals, membrane.c, membrane.k))


# Modal body (body/radiation node): a few guitar-top-ish modes. fs is high (audio rate) so every
# mode sits well under the modal CFL omega*k < 2.
BODY_FREQS_DEFAULT = np.array([110.0, 196.0, 261.0, 440.0])  # Hz


def make_body(
    *,
    freqs: np.ndarray = BODY_FREQS_DEFAULT,
    fs: float = 48000.0,
    sigmas: np.ndarray | float = 0.0,
    masses: np.ndarray | float = 1.0,
    phi: np.ndarray | float = 1.0,
) -> ModalBody:
    """Build a modal body (soundboard) at sample rate ``fs`` with the given modal set."""
    return ModalBody(freqs=freqs, fs=fs, sigmas=sigmas, masses=masses, phi=phi)


# Bridge connection (string terminus <-> body). Modal masses ~0.02 kg (comparable to the string's
# rho*L = 0.005 kg) so the body genuinely loads the string; K well under the exact stability guard.
BODY_MASS_DEFAULT = 0.02
K_BRIDGE_DEFAULT = 8000.0


def make_bridge(
    *,
    N: int = 100,
    lam: float = 0.9,
    K: float = K_BRIDGE_DEFAULT,
    sigma_string: float = 0.0,
    sigma_body: np.ndarray | float = 0.0,
    body_freqs: np.ndarray = BODY_FREQS_DEFAULT,
    masses: np.ndarray | float = BODY_MASS_DEFAULT,
    phi: np.ndarray | float = 1.0,
    L: float = L_DEFAULT,
    T: float = T_DEFAULT,
    rho: float = RHO_DEFAULT,
) -> StringBodyBridge:
    """Build a fixed/free string terminated on a modal body through a bridge spring ``K``.

    Both parts share ``fs = c N / (L lam)`` (``lam < 1`` gives the coupled system headroom below the
    string's Nyquist mode). The default ``K`` sits well inside the exact leapfrog guard.
    """
    c = wave_speed(T, rho)
    fs = c * N / (L * lam)
    string = IdealString(
        L=L, T=T, rho=rho, fs=fs, N=N, boundary=("fixed", "free"), sigma=sigma_string
    )
    body = ModalBody(freqs=body_freqs, fs=fs, sigmas=sigma_body, masses=masses, phi=phi)
    return StringBodyBridge(string=string, body=body, K=K)


# Sympathetic / coupled strings: several fixed/free strings sharing one bridge point on a common
# modal body (piano unisons, sitar sympathetics). Same rig as make_bridge; the strings talk only
# through the shared driving-point displacement w_b.
def make_sympathetic(
    *,
    n_strings: int = 2,
    N: int = 100,
    lam: float = 0.9,
    K: float | np.ndarray = K_BRIDGE_DEFAULT,
    Ts: np.ndarray | None = None,
    sigma_string: float | np.ndarray = 0.0,
    sigma_body: np.ndarray | float = 0.0,
    body_freqs: np.ndarray = BODY_FREQS_DEFAULT,
    masses: np.ndarray | float = BODY_MASS_DEFAULT,
    phi: np.ndarray | float = 1.0,
    L: float = L_DEFAULT,
    rho: float = RHO_DEFAULT,
) -> SympatheticStrings:
    """Build ``n_strings`` fixed/free strings sharing one bridge point on a modal body.

    All strings and the body share ``fs = c0 N / (L lam)`` (``c0`` from the *first* string's
    tension, so string 0 sits at the target ``lam``); ``lam < 1`` gives the coupled system headroom
    below the string Nyquist mode. Pass ``Ts`` (one tension per string) to **detune** the strings
    for the sympathetic-transfer test — detune *downward* (lower tension) to keep every string at
    ``lambda < 1``. ``K`` is a scalar (broadcast to every spring) or a per-string array; all-zero
    decouples (bit-identical to the uncoupled parts). ``sigma_body > 0`` bleeds the body (the loss
    channel that makes a plucked string decay through the bridge — the Weinreich two-stage decay).
    """
    if Ts is None:
        Ts = np.full(n_strings, T_DEFAULT)
    Ts = np.atleast_1d(np.asarray(Ts, dtype=float))
    sig = np.broadcast_to(np.asarray(sigma_string, dtype=float), (n_strings,))
    c0 = wave_speed(float(Ts[0]), rho)
    fs = c0 * N / (L * lam)
    strings = [
        IdealString(
            L=L, T=float(Ts[i]), rho=rho, fs=fs, N=N,
            boundary=("fixed", "free"), sigma=float(sig[i]),
        )
        for i in range(n_strings)
    ]
    body = ModalBody(freqs=body_freqs, fs=fs, sigmas=sigma_body, masses=masses, phi=phi)
    Ks = np.full(n_strings, K) if np.isscalar(K) else np.asarray(K, dtype=float)
    return SympatheticStrings(strings=strings, body=body, Ks=Ks)


# Plate bridge (string terminus <-> grid plate body). A coarse plate (N_plate small) so the per-step
# sparse solve is cheap; its per-node mass rho_s h^2 is comparable to the string's end half-cell, so
# the body genuinely loads the string. K well under the exact Sherman-Morrison guard.
KAPPA_PLATE_BRIDGE = 20.0
K_PLATE_BRIDGE_DEFAULT = 3000.0


def make_plate_bridge(
    *,
    N_string: int = 100,
    N_plate: int = 16,
    lam: float = 0.9,
    K: float = K_PLATE_BRIDGE_DEFAULT,
    sigma_string: float = 0.0,
    sigma_plate: float = 0.0,
    kappa: float = KAPPA_PLATE_BRIDGE,
    drive_index: int | None = None,
    L: float = L_DEFAULT,
    T: float = T_DEFAULT,
    rho: float = RHO_DEFAULT,
    Lx: float = 1.0,
    Ly: float = 1.0,
    rho_plate: float = RHO_AREAL_DEFAULT,
) -> StringPlateBridge:
    """Build a fixed/free string terminated on a simply-supported grid :class:`Plate` body.

    Both parts share ``fs = c N_string / (L lam)`` (``lam < 1`` gives the coupled system headroom
    below the string's Nyquist mode). The plate's sample rate is fixed by the string, so its
    plate-Courant ``mu = kappa k / h^2`` follows from ``N_plate`` (the implicit theta-scheme has no
    CFL limit, so any ``mu`` is fine). The default ``K`` sits well inside the exact stability guard.
    """
    c = wave_speed(T, rho)
    fs = c * N_string / (L * lam)
    string = IdealString(
        L=L, T=T, rho=rho, fs=fs, N=N_string, boundary=("fixed", "free"), sigma=sigma_string
    )
    plate = Plate(
        Lx=Lx, Ly=Ly, kappa=kappa, rho=rho_plate, fs=fs, N=N_plate, sigma=sigma_plate
    )
    return StringPlateBridge(string=string, plate=plate, K=K, drive_index=drive_index)


def make_free_plate_bridge(
    *,
    N_string: int = 100,
    N_plate: int = 16,
    lam: float = 0.9,
    K: float = K_PLATE_BRIDGE_DEFAULT,
    sigma_string: float = 0.0,
    sigma_plate: float = 0.0,
    kappa: float = KAPPA_PLATE_BRIDGE,
    nu: float = 0.3,
    drive_index: int | None = None,
    L: float = L_DEFAULT,
    T: float = T_DEFAULT,
    rho: float = RHO_DEFAULT,
    a: float = 1.0,
    rho_plate: float = RHO_AREAL_DEFAULT,
) -> StringPlateBridge:
    """Build a fixed/free string terminated on a **free-edge** (FFFF) grid :class:`Plate` body.

    The Step-5 counterpart of :func:`make_plate_bridge`: a suspended cymbal/gong (model #5b) as the
    distributed body. Same construction (shared ``fs = c N_string / (L lam)``, plate sample rate
    fixed by the string), but ``boundary="free"`` and the Poisson ratio ``nu`` re-enters (it drops
    out for simply-supported edges). Square by construction (``Lx = Ly = a``, no ``Ly``-snapping).
    The free driving point at ``(0.3a, 0.4a)`` is an interior node (lumped mass ``rho_s h^2``, like
    the supported bridge), so the same default ``K`` sits well inside the free stability guard.
    """
    c = wave_speed(T, rho)
    fs = c * N_string / (L * lam)
    string = IdealString(
        L=L, T=T, rho=rho, fs=fs, N=N_string, boundary=("fixed", "free"), sigma=sigma_string
    )
    plate = Plate(
        Lx=a, Ly=a, kappa=kappa, rho=rho_plate, fs=fs, N=N_plate, sigma=sigma_plate,
        boundary="free", nu=nu,
    )
    return StringPlateBridge(string=string, plate=plate, K=K, drive_index=drive_index)


# Bowed string (nonlinear friction exciter). A flexible (kappa=0), fixed-end damped string bowed
# near the nut. The defaults sit in the multivalued Helmholtz regime (helmholtz_number > 1) so a
# note self-sustains, and carry a little frequency-dependent loss (sigma1 > 0) which rounds the
# Helmholtz corner into clean single-slip motion (else the sharp corner excites raucous multi-slip).
BOW_POSITION_DEFAULT = 0.13    # m -> beta ~ 0.13 (bow near the nut; slip fraction of period ~ beta)
V_BOW_DEFAULT = 0.1            # m/s
BOW_FORCE_DEFAULT = 1.0        # N  (peak friction)
BOW_SHARPNESS_DEFAULT = 60.0   # s^2/m^2
BOW_SIGMA0_DEFAULT = 0.5       # frequency-independent loss
BOW_SIGMA1_DEFAULT = 0.05      # frequency-dependent loss (rounds the corner -> clean Helmholtz)


def make_bowed_string(
    *,
    N: int = 100,
    lam: float = 0.9,
    sigma0: float = BOW_SIGMA0_DEFAULT,
    sigma1: float = BOW_SIGMA1_DEFAULT,
    kappa: float = 0.0,
    bow_position: float = BOW_POSITION_DEFAULT,
    v_bow: float = V_BOW_DEFAULT,
    force: float = BOW_FORCE_DEFAULT,
    sharpness: float = BOW_SHARPNESS_DEFAULT,
    theta: float = THETA_DEFAULT,
    L: float = L_DEFAULT,
    T: float = T_DEFAULT,
    rho: float = RHO_DEFAULT,
) -> BowedString:
    """Build a bowed string (nonlinear friction exciter on a :class:`DampedStiffString`).

    ``kappa = 0`` gives a flexible fixed-end string (``f_1 = c/2L = 100 Hz`` on the canonical rig)
    so the bow physics is isolated. ``lam < 1`` because the friction couples through the string's
    dynamics much like a bridge spring; a hair of headroom below the Nyquist mode keeps the coupled
    solve clean. ``sigma0 > 0`` lets the note reach a steady Helmholtz amplitude instead of growing
    without bound; ``sigma1 > 0`` damps the high partials so the Helmholtz corner stays sharp-but-
    clean (one slip per period) rather than raucous. Pass ``sigma0 = sigma1 = 0`` for the lossless
    energy-balance test, or ``force = 0`` to decouple the bow entirely (the string just decays).
    """
    c = wave_speed(T, rho)
    fs = c * N / (L * lam)
    string = DampedStiffString(
        L=L, T=T, rho=rho, fs=fs, N=N, kappa=kappa, sigma0=sigma0, sigma1=sigma1, theta=theta
    )
    return BowedString(
        string=string, bow_position=bow_position, v_bow=v_bow, force=force, sharpness=sharpness
    )


# Air radiation (the "air" node): a listener 1 m away in ambient air. fs matches the body defaults.
RADIATION_DISTANCE_DEFAULT = 1.0  # m


def make_radiation(
    *,
    fs: float = 48000.0,
    distance: float = RADIATION_DISTANCE_DEFAULT,
    retarded: bool = True,
) -> AirRadiation:
    """Build a monopole far-field radiation node at sample rate ``fs`` for a listener ``distance`` m
    away (ambient-air ``rho0``/``c0`` defaults)."""
    return AirRadiation(fs=fs, distance=distance, retarded=retarded)


# Radiation load (batch 2): a modal body loaded by its own far-field radiation resistance. R here is
# an ACOUSTIC resistance (Pa·s/m^3), sized so the body audibly sheds energy over a couple thousand
# steps (moderate per-step R*G ~ 0.05) — the exact energy identity holds for any R, this just makes
# the decay easy to see. The stability test overrides it with a deliberately enormous value.
R_RADIATION_DEFAULT = 2000.0  # Pa·s/m^3


def make_radiated_body(
    *,
    freqs: np.ndarray = BODY_FREQS_DEFAULT,
    fs: float = 48000.0,
    sigmas: np.ndarray | float = 0.0,
    masses: np.ndarray | float = 1.0,
    phi: np.ndarray | float = 1.0,
    radiation: np.ndarray | float | None = None,
    R: float = R_RADIATION_DEFAULT,
) -> RadiatedBody:
    """Build a modal body loaded by its own radiation resistance ``R`` (the back-reaction).

    ``sigmas = 0`` (default) keeps the modes lossless so the *only* energy sink is the radiation
    channel — then ``body.energy() + radiated_energy`` is conserved and ``radiated_energy`` accounts
    for all the shed energy. ``R = 0`` decouples the air (bit-identical to :func:`make_body`)."""
    body = ModalBody(
        freqs=freqs, fs=fs, sigmas=sigmas, masses=masses, phi=phi, radiation=radiation
    )
    return RadiatedBody(body=body, R=R)


# Acoustic bore (wind leg). A closed-open cylinder ~0.5 m long (clarinet-ish): the odd-harmonic
# fundamental is f1 = c0/(4L) = 171.5 Hz on ambient air. radius small (8 mm) — it only scales the
# absolute energy, not the resonances (which depend on L and c0 alone).
BORE_LENGTH_DEFAULT = 0.5    # m
BORE_RADIUS_DEFAULT = 0.008  # m


def make_bore(
    *,
    N: int = 200,
    lam: float = 1.0,
    boundary=("closed", "open"),
    sigma: float = 0.0,
    L: float = BORE_LENGTH_DEFAULT,
    radius: float = BORE_RADIUS_DEFAULT,
    rho0: float = RHO0_AIR,
    c0: float = C0_AIR,
) -> Bore:
    """Build an acoustic bore whose Courant number is exactly ``lam`` via ``fs = c0 / (lam h)``.

    ``h = L/N`` is fixed by the geometry, so the sample rate is solved for to hit the target ``lam``
    (the 1D-wave CFL ceiling is ``lam <= 1``; ``lam = 1`` is dispersionless). Default is the
    clarinet (closed-open) cylinder. Pass ``boundary=("open", "open")`` for the full-harmonic pipe,
    or ``sigma > 0`` for the passivity test.
    """
    h = L / N
    fs = c0 / (lam * h)
    return Bore(
        L=L, fs=fs, N=N, radius=radius, boundary=boundary, sigma=sigma, rho0=rho0, c0=c0
    )


# Radiating bell (wind leg, batch 2): a closed-open clarinet whose open end is a passively-lossy
# bell of acoustic resistance R (Pa·s/m^3). R_BELL_DEFAULT ~ the piston radiation resistance at the
# fundamental (a realistic, lightly-radiating clarinet bell: R << Z0, high reflection, slow leak).
# The characteristic impedance Z0 = rho0 c0 / S dwarfs it (~2e6 here), so the tube stays
# odd-harmonic and only acquires finite-Q resonances. A specific reflection sweep passes R directly.
R_BELL_DEFAULT = 650.0  # Pa·s/m^3


def make_radiating_bore(
    *,
    N: int = 200,
    lam: float = 1.0,
    boundary=("closed", "radiating"),
    R_bell: float = R_BELL_DEFAULT,
    sigma: float = 0.0,
    L: float = BORE_LENGTH_DEFAULT,
    radius: float = BORE_RADIUS_DEFAULT,
    rho0: float = RHO0_AIR,
    c0: float = C0_AIR,
) -> Bore:
    """Build a clarinet with a **radiating** (passively-lossy) bell at Courant number ``lam``.

    The batch-2 counterpart of :func:`make_bore`: identical geometry/rig (``fs = c0 / (lam h)``),
    but the open end is replaced by a radiation resistance ``R_bell`` that sheds sound to the field.
    Default ``R_bell`` is a realistic lightly-radiating bell (``R << Z0``); pass a larger ``R_bell``
    (toward ``Z0``) for a heavily-absorbing / anechoic termination, or ``boundary`` to place the
    radiating end differently. ``sigma > 0`` adds the interior viscous loss on top of the radiation.
    """
    h = L / N
    fs = c0 / (lam * h)
    return Bore(
        L=L, fs=fs, N=N, radius=radius, boundary=boundary, R_bell=R_bell, sigma=sigma,
        rho0=rho0, c0=c0,
    )


# Single-reed mouthpiece (wind leg, batch 3): a dynamic reed blowing a clarinet air column. The
# defaults are a clarinet-plausible reed (f_reed ~ 2.5 kHz, heavily lip-damped) whose closing
# pressure p_closing = mu wr^2 H0 ~ 3 kPa; the control is gamma = p_mouth / p_closing (the note
# speaks around gamma ~ 1/3). Default bore is a radiating clarinet so the note settles into a steady
# regime; pass boundary=("closed", "open") + sigma=0 for the LOSSLESS energy-balance money test.
REED_P_MOUTH_DEFAULT = 1500.0  # Pa (gamma ~ 0.5, comfortably above threshold)


def make_reed_bore(
    *,
    N: int = 200,
    lam: float = 1.0,
    p_mouth: float = REED_P_MOUTH_DEFAULT,
    boundary=("closed", "radiating"),
    R_bell: float = R_BELL_DEFAULT,
    sigma: float = 0.0,
    f_reed: float = 2500.0,
    q_reed: float = 4.0,
    L: float = BORE_LENGTH_DEFAULT,
    radius: float = BORE_RADIUS_DEFAULT,
) -> ReedBore:
    """Build a dynamic-reed clarinet (a :class:`ReedBore` on a :class:`Bore`) at Courant ``lam``.

    The bore is the batch-1/2 clarinet (``fs = c0 / (lam h)``, left end ``"closed"`` for the
    mouthpiece); the reed self-oscillates it under a steady mouth pressure ``p_mouth``. Default is a
    lightly-radiating bell (``R_bell``) so the tone reaches a steady amplitude. For the lossless
    energy-balance test pass ``boundary=("closed", "open"), sigma=0`` (then ``E = E_bore + E_reed``
    changes only by ``mouth_work - jet_loss - reed_damp_work``). Lower ``p_mouth`` below threshold
    to watch the note fail to speak.
    """
    h = L / N
    fs = C0_AIR / (lam * h)
    bore = Bore(
        L=L, fs=fs, N=N, radius=radius, boundary=boundary, R_bell=R_bell, sigma=sigma
    )
    return ReedBore(bore=bore, p_mouth=p_mouth, f_reed=f_reed, q_reed=q_reed)


# Mallet-membrane collision (model #7, first contact model). A soft mallet strikes a square
# drumhead at the centre. The defaults keep the felt half-period well-resolved (~32 steps at K=5e4,
# M=0.02) and hand the head ~two thirds of the strike energy at peak, so the conservation money test
# genuinely exercises the nonlinear coupling (a bracket bug can't hide behind a linear scheme).
MALLET_MASS_DEFAULT = 0.02      # kg
MALLET_K_DEFAULT = 5.0e4        # N/m^alpha  (felt stiffness)
MALLET_ALPHA_DEFAULT = 2.3      # felt exponent (piano-ish)
MALLET_VELOCITY_DEFAULT = 3.0   # m/s impact speed toward the head


def make_mallet(
    *,
    N: int = 40,
    lam: float = 0.5,
    K: float = MALLET_K_DEFAULT,
    mass: float = MALLET_MASS_DEFAULT,
    alpha: float = MALLET_ALPHA_DEFAULT,
    hysteresis: float = 0.0,
    strike_x: float = 0.5,
    strike_y: float = 0.5,
    strike_velocity: float = MALLET_VELOCITY_DEFAULT,
    gap: float = 0.0,
    sigma: float = 0.0,
    domain: Domain = "rectangle",
    Lx: float = 1.0,
    Ly: float = 1.0,
    radius: float = RADIUS_DEFAULT,
    T: float = T_DEFAULT,
    rho: float = RHO_AREAL_DEFAULT,
) -> MalletMembrane:
    """Build a mallet striking a membrane (model #7). ``lam < 1/sqrt(2)`` (default 0.5) oversamples
    the stiff contact; ``sigma = 0`` and ``hysteresis = 0`` give the lossless conservation money
    test, ``sigma > 0`` or ``hysteresis > 0`` the passivity test. ``K = 0`` is not allowed (a
    massless felt) — pass ``strike_velocity = 0`` or a large ``gap`` to keep the mallet clear."""
    membrane = make_membrane(
        domain=domain, N=N, lam=lam, sigma=sigma, T=T, rho=rho, Lx=Lx, Ly=Ly, radius=radius
    )
    return MalletMembrane(
        membrane=membrane, mass=mass, stiffness=K, alpha=alpha, hysteresis=hysteresis,
        strike_x=strike_x, strike_y=strike_y, strike_velocity=strike_velocity, gap=gap,
    )


def make_mallet_wall(
    *,
    K: float = MALLET_K_DEFAULT,
    mass: float = MALLET_MASS_DEFAULT,
    alpha: float = 1.0,
    hysteresis: float = 0.0,
    fs: float = 96000.0,
    strike_velocity: float = 2.0,
    gap: float = 0.0,
) -> MalletWall:
    """Build the standalone mass-vs-fixed-wall rig (model #7 closed-form oracle). ``alpha = 1``,
    ``hysteresis = 0`` gives the analytic half-period ``pi*sqrt(M/K)`` and exact velocity reversal;
    ``hysteresis > 0`` makes the felt lossy (restitution < 1)."""
    return MalletWall(
        mass=mass, stiffness=K, fs=fs, alpha=alpha, hysteresis=hysteresis,
        strike_velocity=strike_velocity, gap=gap,
    )


# Barrier-string collision (model #8, first *distributed* contact model). A stiff/flexible string
# vibrating against a one-sided nonlinear barrier below it (fret buzz / tanpura jawari). The default
# is a flexible fixed-end string and a flat rail 2 mm below rest; K is a stiff felt/wood contact.
# lam < 1 keeps the coupled solve clear of the string's Nyquist mode. A big-negative barrier (out of
# reach) is the K=0 analog (bit-identical to the bare string).
BARRIER_K_DEFAULT = 1.0e6      # N/m^alpha  (contact stiffness density)
BARRIER_ALPHA_DEFAULT = 1.5    # contact exponent (Hertzian-ish)
BARRIER_HEIGHT_DEFAULT = -2.0e-3  # m  (flat rail below the string's rest line)


def make_barrier_string(
    *,
    N: int = 80,
    lam: float = 0.9,
    K: float = BARRIER_K_DEFAULT,
    alpha: float = BARRIER_ALPHA_DEFAULT,
    barrier=BARRIER_HEIGHT_DEFAULT,
    hysteresis: float = 0.0,
    kappa: float = 0.0,
    sigma0: float = 0.0,
    sigma1: float = 0.0,
    theta: float = THETA_DEFAULT,
    newton_tol: float = 1e-13,
    L: float = L_DEFAULT,
    T: float = T_DEFAULT,
    rho: float = RHO_DEFAULT,
) -> BarrierString:
    """Build a string against a one-sided distributed barrier (model #8) at Courant number ``lam``.

    ``fs = c N / (L lam)``; ``lam < 1`` gives the coupled contact solve headroom below the string's
    Nyquist mode. ``sigma0 = sigma1 = 0`` and ``hysteresis = 0`` give the lossless conservation
    money test; ``sigma > 0`` or ``hysteresis > 0`` the passivity test. ``barrier`` is a scalar flat
    rail or an ``(N+1,)`` profile (use ``-inf`` off-support for a point fret). A big-negative
    ``barrier`` keeps the string clear (the ``K = 0`` analog)."""
    c = wave_speed(T, rho)
    fs = c * N / (L * lam)
    string = DampedStiffString(
        L=L, T=T, rho=rho, fs=fs, N=N, kappa=kappa, sigma0=sigma0, sigma1=sigma1, theta=theta
    )
    return BarrierString(
        string=string, barrier=barrier, stiffness=K, alpha=alpha, hysteresis=hysteresis,
        newton_tol=newton_tol,
    )


# Jawari / buzzing bridge (composes model #8, no new core physics): a *curved* barrier at the string
# termination. The bridge is a downward-opening parabola tangent to the rest line at the fixed end;
# the string wraps onto it on the downswing, its departure point travelling along the curve — the
# "life"/shimmer of the sitar & tanpura. `clearance` is the crest's drop below rest: >0 grazes,
# <0 preloads (the whole span contacts at rest — the static-equilibrium-oracle case). `depth` is the
# curve's total drop over the bridge span; keep it comparable to the near-termination downswing so
# the string wraps a wide span (too deep -> it only grazes the crest, acting like a point contact).
JAWARI_WIDTH_FRAC_DEFAULT = 0.15   # bridge span as a fraction of L (near the termination)
JAWARI_DEPTH_DEFAULT = 1.0e-3      # m   (crest-to-far-edge drop of the parabola)
JAWARI_K_DEFAULT = 2.0e6           # N/m^alpha  (stiff wood/bone bridge)


def jawari_barrier(
    x: np.ndarray,
    L: float,
    *,
    width_frac: float = JAWARI_WIDTH_FRAC_DEFAULT,
    depth: float = JAWARI_DEPTH_DEFAULT,
    clearance: float = 0.0,
) -> np.ndarray:
    """Parabolic jawari-bridge profile on the grid ``x`` (length ``N+1``): a curved barrier hugging
    the ``x = 0`` termination, ``-inf`` (out of support) beyond the bridge span.

    ``b(x) = -clearance - depth·(x/d)²`` for ``0 < x ≤ d = width_frac·L``. The crest (nearest the
    string) is at the termination side and the surface curves away by ``depth`` at the far edge.
    """
    d = width_frac * L
    b = np.full_like(np.asarray(x, dtype=float), -np.inf)
    on = (x > 0.0) & (x <= d)
    b[on] = -clearance - depth * (x[on] / d) ** 2
    return b


def make_jawari_string(
    *,
    N: int = 100,
    lam: float = 0.4,
    K: float = JAWARI_K_DEFAULT,
    alpha: float = BARRIER_ALPHA_DEFAULT,
    width_frac: float = JAWARI_WIDTH_FRAC_DEFAULT,
    depth: float = JAWARI_DEPTH_DEFAULT,
    clearance: float = 0.0,
    hysteresis: float = 0.0,
    kappa: float = 0.0,
    sigma0: float = 0.0,
    sigma1: float = 0.0,
    theta: float = THETA_DEFAULT,
    newton_tol: float = 1e-13,
    L: float = L_DEFAULT,
    T: float = T_DEFAULT,
    rho: float = RHO_DEFAULT,
) -> BarrierString:
    """Build a sitar/tanpura *jawari* string: a :class:`BarrierString` (model #8) whose barrier is
    the curved bridge of :func:`jawari_barrier`. ``N = 100`` resolves the wrap (support ~15 nodes,
    well under the dense-solve cliff). ``sigma0 = sigma1 = hysteresis = 0`` gives the lossless
    conservation gate; ``clearance < 0`` seats the whole bridge in contact for the static oracle."""
    c = wave_speed(T, rho)
    fs = c * N / (L * lam)
    string = DampedStiffString(
        L=L, T=T, rho=rho, fs=fs, N=N, kappa=kappa, sigma0=sigma0, sigma1=sigma1, theta=theta
    )
    barrier = jawari_barrier(string.x, L, width_frac=width_frac, depth=depth, clearance=clearance)
    return BarrierString(
        string=string, barrier=barrier, stiffness=K, alpha=alpha, hysteresis=hysteresis,
        newton_tol=newton_tol,
    )


def bore_low_eigenfrequencies(bore: Bore, n_modes: int) -> np.ndarray:
    """The ``n_modes`` lowest discrete resonance frequencies (Hz) of ``bore`` (ascending).

    Solves the generalized eigenproblem ``L φ = ω² C φ`` on the **free** (non-open) pressure nodes —
    ``L = Gᵀ M⁻¹ G`` (pressure stiffness) and ``C`` (compliance mass), both exposed by the bore,
    for the smallest ``ω²``, then maps each through :func:`modal.discrete_bore_eigenfrequency` (the
    leapfrog dispersion). A closed-open or open-open tube is positive-definite (an open end pins a
    node), so plain shift-invert at ``σ = 0`` works; a fully closed tube has a constant-pressure
    nullspace (``ω = 0``), handled with a small negative shift and dropping that mode.
    """
    dof = bore.dof
    Lfree = bore.Lop[dof][:, dof]
    Cfree = bore.Cmat[dof][:, dof]
    n_open = int(bore._open_left) + int(bore._open_right)
    if n_open == 0:
        w1_scale = (np.pi * bore.c0 / bore.L) ** 2  # ~ first resonance ω² -> a safe negative shift
        shift = -1e-3 * w1_scale
        w2 = eigsh(
            Lfree, k=n_modes + 1, M=Cfree, sigma=shift, which="LM", return_eigenvectors=False
        )
        w2 = np.sort(w2)[1 : n_modes + 1]  # drop the ω≈0 constant-pressure mode
    else:
        w2 = eigsh(Lfree, k=n_modes, M=Cfree, sigma=0.0, which="LM", return_eigenvectors=False)
        w2 = np.sort(w2)
    return np.asarray(modal.discrete_bore_eigenfrequency(w2, bore.k))


def discrete_sho_frequency(f: float, k: float) -> float:
    """Exact discrete oscillation frequency (Hz) of the leapfrog SHO for a mode of ``f`` Hz.

    The scheme ``q^{n+1} - 2q^n + q^{n-1} = -k^2 omega^2 q^n`` has solutions ``cos(Omega n k)`` with
    ``sin(Omega k / 2) = omega k / 2``, i.e. ``Omega = (2/k) arcsin(omega k / 2)``. Approaches the
    continuum ``f`` as ``omega k -> 0``; used as the modal oracle.
    """
    omega = 2.0 * np.pi * f
    return float(np.arcsin(0.5 * omega * k) / (np.pi * k))


def convergence_orders(errors: np.ndarray, step_sizes: np.ndarray) -> np.ndarray:
    """Empirical orders ``p`` between consecutive (h, error) pairs: ``error ~ C h^p``."""
    errors = np.asarray(errors, dtype=float)
    step_sizes = np.asarray(step_sizes, dtype=float)
    return np.log(errors[:-1] / errors[1:]) / np.log(step_sizes[:-1] / step_sizes[1:])


def measure_mode_frequencies(
    modes: np.ndarray,
    *,
    N: int,
    lam: float,
    secs: float = 0.5,
    amplitude: float = 1e-3,
) -> np.ndarray:
    """Measure the FDTD oscillation frequency (Hz) of each single spatial mode in ``modes``.

    Each run is initialised with one exact discrete eigenvector ``phi_m = sin(m pi x / L)``. Because
    that eigenvector is the *only* thing present, the field stays ``u(t) = q(t) phi_m``, so the
    modal coordinate ``q(t) = <u(t), phi_m>`` is a pure tone at the mode's discrete frequency. We
    measure that projection rather than a single grid pickup: a point pickup can land on a node of
    the mode (e.g. mode N/2 vanishes at every even grid node), whereas the projection has maximal
    SNR for every mode and any N. The peak search is anchored at the closed-form dispersion oracle.
    Returns one frequency per mode (the dispersion curve, measured).
    """
    c, L = wave_speed(), L_DEFAULT
    out = []
    for m in np.atleast_1d(modes):
        m = int(m)
        s = make_string(N=N, lam=lam)
        phi = modal.mode_shape(s.x, L, m)
        s.set_state(phi * amplitude)
        res = simulate(s, num_steps=int(secs * s.fs), snapshot_stride=1)
        states = np.array([state for _, state in res.snapshots])  # (T+1, N+1)
        q = states @ phi  # modal coordinate time series (proportional to a pure cosine)
        f_oracle = modal.discrete_mode_frequency(c, L, N, lam, m)
        out.append(spectrum.measure_partials_near(q, res.fs, np.array([f_oracle]))[0])
    return np.array(out)


def measure_stiff_mode_frequencies(
    modes: np.ndarray,
    *,
    N: int,
    lam: float,
    kappa: float = KAPPA_DEFAULT,
    theta: float = THETA_DEFAULT,
    secs: float = 0.5,
    amplitude: float = 1e-3,
) -> np.ndarray:
    """Measure the stiff-string FDTD frequency (Hz) of each single spatial mode in ``modes``.

    Identical to :func:`measure_mode_frequencies` but for :class:`StiffString`: ``sin(m pi x / L)``
    is still an *exact* discrete eigenvector (the biharmonic block is ``(delta_xx)^2``), so a
    single-mode initial condition stays a pure tone and its modal-coordinate projection is a clean
    cosine at the stiff discrete frequency. The peak search is anchored at the closed-form stiff
    dispersion oracle (which depends on ``theta``).
    """
    c, L = wave_speed(), L_DEFAULT
    out = []
    for m in np.atleast_1d(modes):
        m = int(m)
        s = make_stiff_string(N=N, lam=lam, kappa=kappa, theta=theta)
        phi = modal.mode_shape(s.x, L, m)
        s.set_state(phi * amplitude)
        res = simulate(s, num_steps=int(secs * s.fs), snapshot_stride=1)
        states = np.array([state for _, state in res.snapshots])
        q = states @ phi
        f_oracle = modal.discrete_stiff_mode_frequency(c, L, N, kappa, s.k, m, theta)
        out.append(spectrum.measure_partials_near(q, res.fs, np.array([f_oracle]))[0])
    return np.array(out)


# -- model #9: tension-modulated string -------------------------------------------------

EA_DEFAULT = 1.0e5
"""Axial stiffness (N) for model #9 tests. With T_DEFAULT = 200 N this is EA/T = 500 -- squarely in
a real steel string's range (the governing ratio is EA/T = (c_long/c)^2 ~ 150-600; see
:func:`~physsynth.core.string_nonlinear.string_coefficients_from_material`). **Reused by model
#10**, so the cross-model KC check compares like with like."""

GEO_NEWTON_TOL = 1e-15
"""Relative Newton tolerance for model #10 tests -- **tighter than the class default on purpose**.

Energy drift is *proportional* to it (measured over five decades), so the drift gate is really a
statement about this number. The tests pin it rather than inherit, so that a future change to the
class default cannot silently loosen the 1e-10 bar.

It is only *reachable* because :data:`GEO_LAM_LONG_DEFAULT` is small: the residual's round-off floor
scales with the operator norm, which grows like ``lam_long^2``. At ``lam_long ~ 11`` the floor sits
*above* this bar, so every step exhausts ``newton_maxiter`` and stalls -- 60 iterations per step,
and a tolerance that can never be met. The two constants are a pair."""

GEO_LAM_LONG_DEFAULT = 0.5
"""Default **longitudinal** Courant number for model #10 tests -- the fast field sets the timestep.

Measured: ``lam_long <= 2`` conserves energy to ~1e-12 across every hard case tried (plucked and
mode-3 ICs, amplitudes to 1e-2); at ``lam_long >= 4`` the Newton solve fails and drift explodes to
1e+3 .. 1e+5. There is no CFL to catch this -- the theta-scheme is unconditionally stable and
reports nothing -- so 0.5 buys a 4x margin on a measured cliff. Tests that *want* the unresolved
regime pass ``lam=`` explicitly."""


def make_tension_string(
    *,
    N: int = 100,
    lam: float = 1.0,
    kappa: float = KAPPA_DEFAULT,
    EA: float = 0.0,
    sigma0: float = 0.0,
    sigma1: float = 0.0,
    theta: float = THETA_DEFAULT,
    tension_tol: float = TENSION_TOL_DEFAULT,
    L: float = L_DEFAULT,
    T: float = T_DEFAULT,
    rho: float = RHO_DEFAULT,
) -> TensionModulatedString:
    """Build a tension-modulated string (model #9) at Courant ``lam`` via ``fs = c N/(L lam)``.

    ``EA = 0`` (the default) is model #3 bit-for-bit -- pass ``EA=EA_DEFAULT`` for the nonlinearity.
    ``lam > 1`` is allowed (the linear part is unconditionally stable), though a nonlinearity always
    wants oversampling on its own account.
    """
    c = wave_speed(T, rho)
    fs = c * N / (L * lam)
    return TensionModulatedString(
        L=L, T=T, rho=rho, fs=fs, N=N, kappa=kappa, EA=EA, sigma0=sigma0, sigma1=sigma1,
        theta=theta, tension_tol=tension_tol,
    )


# -- model #10: geometrically-exact string ----------------------------------------------


def make_geometric_string(
    *,
    N: int = 64,
    lam: float | None = None,
    lam_long: float | None = None,
    kappa: float = KAPPA_DEFAULT,
    kappa_w: float | None = None,
    EA: float = EA_DEFAULT,
    sigma0: float = 0.0,
    sigma1: float = 0.0,
    sigma0_long: float | None = None,
    sigma1_long: float | None = None,
    theta: float = THETA_DEFAULT,
    newton_tol: float = GEO_NEWTON_TOL,
    L: float = L_DEFAULT,
    T: float = T_DEFAULT,
    rho: float = RHO_DEFAULT,
    **kwargs,
) -> GeometricString:
    """Build a geometrically-exact string (model #10), timestep set from ``lam_long`` by default.

    ``EA = T`` is model #3 bit-for-bit (the nonlinearity coefficient is ``EA - T0``); the default
    ``EA = EA_DEFAULT`` is **the same number model #9's tests use**, so the batch-2 cross-model KC
    check is apples-to-apples -- modulo the identification ``EA_#9 <-> (EA - T0)_#10``, a 0.2 %
    offset at these values that is *the identification, not a discrepancy*.

    **The fast field sets the timestep, so ``lam_long`` is the default knob** (see
    ``GEO_LAM_LONG_DEFAULT``). Pass ``lam=`` to set ``fs`` from the **transverse** wave instead;
    the two are mutually exclusive.

    Why the default is this way round, and not model #1-#9's ``lam``: the longitudinal field runs at
    ``lam_long = sqrt(EA/T) * lam``, about **22x larger** at the default ``EA/T = 500``. So the
    familiar ``lam=0.5`` silently means ``lam_long = 11`` -- eleven cells of longitudinal travel per
    timestep. The implicit scheme is unconditionally *stable* there and reports no CFL violation,
    but **stable is not accurate**, and past ``lam_long ~ 4`` the Newton solve stops converging and
    the energy gate fails by *fourteen orders of magnitude* rather than a little. ``lam=`` is
    therefore a deliberate opt-in, not the path of least resistance. Phantom *frequencies* ride on
    the well-resolved transverse partials and are safe either way.
    """
    if lam is not None and lam_long is not None:
        raise ValueError("pass lam= or lam_long=, not both — they both set fs")
    c = wave_speed(T, rho)
    if lam is not None:
        fs = c * N / (L * lam)
    else:
        lam_long = GEO_LAM_LONG_DEFAULT if lam_long is None else lam_long
        fs = float(np.sqrt(EA / rho)) * N / (L * lam_long)
    return GeometricString(
        L=L, T=T, rho=rho, fs=fs, N=N, EA=EA, kappa=kappa, kappa_w=kappa_w, sigma0=sigma0,
        sigma1=sigma1, sigma0_long=sigma0_long, sigma1_long=sigma1_long, theta=theta,
        newton_tol=newton_tol, **kwargs,
    )


def geometric_mode_ic(N: int, m: int = 1, amp: float = 1e-3, L: float = L_DEFAULT) -> np.ndarray:
    """A single simply-supported eigenmode ``amp * sin(m pi x / L)`` on the full ``N+1`` grid."""
    return amp * np.sin(m * np.pi * np.linspace(0.0, L, N + 1) / L)


def geometric_pluck_ic(
    N: int, amp: float = 1e-3, at: float = 0.2, L: float = L_DEFAULT
) -> np.ndarray:
    """A triangular pluck of height ``amp`` at fractional position ``at`` (full ``N+1`` grid).

    **The general-case IC, and the one that matters.** A single-mode IC is secretly a scalar Duffing
    test: it keeps the strain field nearly uniform and never exercises the *local* tension that
    distinguishes model #10 from model #9's spatial scalar. The pluck's corner is broadband, so the
    stretch varies cell-to-cell -- which is the thing under test.
    """
    x = np.linspace(0.0, 1.0, N + 1)
    return amp * np.where(x <= at, x / at, (1.0 - x) / (1.0 - at))


def mode_off_fraction(u: np.ndarray, shape: np.ndarray, scale: float) -> float:
    """Off-mode content of ``u``, as a fraction of the **fixed** amplitude ``scale``.

    Never normalize by the instantaneous ``||u||``: a single mode passes through ``u ~ 0`` twice a
    period, where roundoff dominates and the ratio reports a spurious ``1.0`` that looks exactly
    like a catastrophic bug. ``scale`` should be ``||u_0||`` (see the model #9 plan doc).
    """
    proj = np.dot(u, shape) / np.dot(shape, shape) * shape
    return float(np.linalg.norm(u - proj) / scale)


def measure_tension_mode_frequency(
    s: TensionModulatedString,
    shape: np.ndarray,
    *,
    n_crossings: int = 10,
    max_steps: int = 400_000,
) -> float:
    """Measure a mode's **nonlinear** frequency (Hz) from descending zero crossings of ``q(t)``.

    Steps ``s`` in place. Uses the modal projection ``q = <u, shape>/<shape, shape>`` (maximal SNR
    for any mode and any N, unlike a point pickup that may sit on a node) and linearly interpolates
    each crossing. Zero crossings -- **not** ``spectrum.measure_partials_near``: its search window
    is anchored on the *linear* frequency and simply misses a peak shifted tens of percent by
    hardening (model #6's lesson, the same trap in a new model).
    """
    denom = float(np.dot(shape, shape))
    prev = float(np.dot(s.state, shape)) / denom
    times: list[float] = []
    for n in range(1, max_steps + 1):
        s.step()
        cur = float(np.dot(s.state, shape)) / denom
        if prev > 0.0 >= cur:
            times.append((n - 1 + prev / (prev - cur)) * s.k)
            if len(times) >= n_crossings:
                break
        prev = cur
    if len(times) < 2:
        raise RuntimeError(f"only {len(times)} crossings in {max_steps} steps -- can't measure")
    return 1.0 / float(np.mean(np.diff(times)))
