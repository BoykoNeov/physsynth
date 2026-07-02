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
from physsynth.core.connection import StringBodyBridge, StringPlateBridge
from physsynth.core.engine import simulate
from physsynth.core.membrane import Domain, Membrane
from physsynth.core.plate import Plate
from physsynth.core.string_damped import DampedStiffString
from physsynth.core.string_ideal import Boundary, IdealString
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
