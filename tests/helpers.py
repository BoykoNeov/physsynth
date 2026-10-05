"""Shared test fixtures and physical parameters for the validation harness.

A single canonical string (L=1 m, T=200 N, rho=0.005 kg/m -> c=200 m/s, f1=100 Hz) is reused so the
expected frequencies are easy to reason about.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import eigsh

from physsynth.analysis import modal
from physsynth.analysis.horizon import pitch_horizon
from physsynth.core.airbox import (
    AirBox,
    RoomLoadedBody,
)
from physsynth.core.body import ModalBody
from physsynth.core.bore import C0_AIR, RHO0_AIR, Bore
from physsynth.core.bow import BowedString
from physsynth.core.collision import BarrierString
from physsynth.core.mallet import MalletMembrane, MalletPlate, MalletVKPlate, MalletWall
from physsynth.core.membrane import Domain, Membrane
from physsynth.core.plate import THETA_DEFAULT as PLATE_THETA_DEFAULT
from physsynth.core.plate import Plate, VKPlate
from physsynth.core.radiation import (
    AirRadiation,
    RadiatedBody,
    RationalAirLoad,
    ReactiveRadiatedBody,
)
from physsynth.core.reed import ReedBore
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


# Frequency-dependent radiation load (batch 3): resistance R in PARALLEL with the radiation mass
# M_a, the exact first-order (pulsating-sphere) impedance. The default M_a puts the trapezoid's
# k R / (2 M_a) around 0.1 — the reactance is genuinely in play without being stiff. M_a = inf is
# the constant-R load (batch 2, bit-identical); RationalAirLoad.from_sphere gives the physically
# consistent pair for a given radius.
M_A_RADIATION_DEFAULT = 0.2  # kg/m^4
SPHERE_RADIUS_DEFAULT = 0.05  # m — a 5 cm pulsating sphere (ka = 1 at ~1.1 kHz)


def make_reactive_body(
    *,
    freqs: np.ndarray = BODY_FREQS_DEFAULT,
    fs: float = 48000.0,
    sigmas: np.ndarray | float = 0.0,
    masses: np.ndarray | float = 1.0,
    phi: np.ndarray | float = 1.0,
    radiation: np.ndarray | float | None = None,
    R: float = R_RADIATION_DEFAULT,
    M_a: float = M_A_RADIATION_DEFAULT,
) -> ReactiveRadiatedBody:
    """Build a modal body loaded by the rational (frequency-dependent) radiation impedance.

    ``sigmas = 0`` (default) keeps the modes lossless so the air is the only channel and
    ``body.energy() + stored + radiated`` is conserved. ``R = 0`` decouples the air entirely
    (bit-identical to :func:`make_body`); ``M_a = inf`` collapses to :func:`make_radiated_body`."""
    body = ModalBody(
        freqs=freqs, fs=fs, sigmas=sigmas, masses=masses, phi=phi, radiation=radiation
    )
    return ReactiveRadiatedBody(body=body, load=RationalAirLoad(fs=fs, R=R, M_a=M_a))


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


# Mallet on a PLATE (model #7p). The same felt and the same mallet as `make_mallet`, against an
# *implicit* resonator instead of an explicit one -- which is the whole difference, and the reason
# the default `mu` here is 1.0 rather than the plate suite's 2.0. `mu` sets the timestep
# (`fs = kappa / (mu h^2)`), and the felt does not care about the plate's Courant number, it cares
# about its own half-period `pi*sqrt(M/K)` ~ 1.99 ms: at `mu = 1, N = 24, Lx = 1` that is
# `fs = 11520 Hz` and ~23 steps through the contact, against the 8 below which the model warns.
# Raising `mu` or coarsening `N` lowers `fs` and will eventually under-resolve the strike, so
# `test_mallet_plate.py` asserts `steps_per_contact` on the shipped defaults rather than trusting
# this comment.
MALLET_PLATE_MU_DEFAULT = 1.0
MALLET_PLATE_N_DEFAULT = 24


def make_mallet_plate(
    *,
    N: int = MALLET_PLATE_N_DEFAULT,
    mu: float = MALLET_PLATE_MU_DEFAULT,
    kappa: float = KAPPA_PLATE_DEFAULT,
    K: float = MALLET_K_DEFAULT,
    mass: float = MALLET_MASS_DEFAULT,
    alpha: float = MALLET_ALPHA_DEFAULT,
    hysteresis: float = 0.0,
    strike_x: float = 0.3,
    strike_y: float = 0.4,
    strike_velocity: float = MALLET_VELOCITY_DEFAULT,
    gap: float = 0.0,
    sigma: float = 0.0,
    boundary: str = "supported",
    domain: str = "rectangle",
    Lx: float = 1.0,
    Ly: float = 1.0,
    nu: float = 0.3,
    theta: float = PLATE_THETA_DEFAULT,
    rho: float = RHO_AREAL_DEFAULT,
) -> MalletPlate:
    """Build a mallet striking a Kirchhoff plate (model #7p).

    ``boundary="supported"`` is a struck soundboard, ``boundary="free"`` a suspended cymbal --
    which **recoils**, because a point strike feeds the free plate's ``{1, x, y}`` rigid nullspace
    and nothing holds the mean. ``sigma = 0`` with ``hysteresis = 0`` gives the lossless
    conservation money test, either one positive the passivity test.

    The strike defaults to ``(0.3 Lx, 0.4 Ly)``: off every low mode's symmetry axis, so no partial
    is nulled by accident. Move it to the centre deliberately when that is the point.
    """
    h = Lx / N
    fs = kappa / (mu * h * h)
    plate = Plate(
        Lx=Lx, Ly=Ly, kappa=kappa, rho=rho, fs=fs, N=N, sigma=sigma, theta=theta,
        boundary=boundary, domain=domain, nu=nu,
    )
    return MalletPlate(
        plate=plate, mass=mass, stiffness=K, alpha=alpha, hysteresis=hysteresis,
        strike_x=strike_x * Lx, strike_y=strike_y * Ly, strike_velocity=strike_velocity, gap=gap,
    )

# Mallet on a GONG (model #7g) -- the nested solve. The same felt again, now against a *nonlinear*
# resonator, so there is no drive-point influence column and the contact force is found by an outer
# iteration wrapped around a full von Karman plate solve. See docs/dev/mallet-gong-plan.md.
#
# The plate is `tests/test_vk_energy.py`'s: a 0.4 m square of 1 mm steel, whose fundamental is near
# 30 Hz and whose thickness is the amplitude scale the nonlinearity switches on at. `fs` is a free
# parameter here and NOT derived from a Courant number, because the theta-scheme is unconditionally
# stable -- 48 kHz gives 151 steps through the felt's half-period, well clear of the 8 below which
# the model warns.
#
# `strike_velocity` is the knob that matters: 1 m/s stays essentially linear (`w/e ~ 0.5`), 6 m/s
# reaches `w/e ~ 2.8` with the membrane term at 3% of the total, and 12 m/s reaches 4.4. A test
# whose subject is the nonlinearity must check the membrane share rather than trusting the velocity.
GONG_MATERIAL = dict(E=2.0e11, e=1.0e-3, nu=0.3, rho=7800.0)  # rho is VOLUMETRIC (kg/m^3)
GONG_SIDE = 0.4                # m
GONG_FS = 48000.0              # Hz -- free, not a CFL
GONG_N = 20                    # 361 live nodes on the supported branch
GONG_VELOCITY_DEFAULT = 6.0    # m/s -- w/e ~ 2.8, membrane energy ~3% of the total
# A HEAVIER head than the membrane and linear-plate models' 20 g, and it is not a style choice: a
# 20 g head at 3 m/s leaves the membrane term at 0.6% of the total, which is a test of the linear
# theta-scheme wearing a nonlinear plate's name. 50 g at the same speed reaches 1.8%. The native
# fixture in `crates/physsynth-core/tests/mallet_gong.rs` is the same 50 g for the same reason.
GONG_MASS_DEFAULT = 0.05       # kg


def make_mallet_gong(
    *,
    N: int = GONG_N,
    fs: float = GONG_FS,
    a: float = GONG_SIDE,
    mass: float = GONG_MASS_DEFAULT,
    K: float = MALLET_K_DEFAULT,
    alpha: float = MALLET_ALPHA_DEFAULT,
    hysteresis: float = 0.0,
    strike_x: float = 0.3,
    strike_y: float = 0.4,
    strike_velocity: float = GONG_VELOCITY_DEFAULT,
    gap: float = 0.0,
    sigma: float = 0.0,
    boundary: str = "supported",
    nonlinear: bool = True,
    couple_tol: float = 1e-13,
    couple_max_iter: int = 50,
    couple_method: str = "picard",
    outer_tol: float = 1e-13,
    outer_max_iter: int = 20,
    material: dict | None = None,
) -> MalletVKPlate:
    """Build a mallet striking a von Karman plate (model #7g, the gong).

    ``boundary="supported"`` is a gong and ``"free"`` a suspended cymbal -- which **recoils**, and
    whose energy read-out is therefore a read-out bar rather than an energy bar, exactly as it is
    for the linear :func:`make_mallet_plate`. ``nonlinear=False`` is the regression path: the plate
    is then affine in the contact force, the outer loop exits at one iteration, and the whole model
    reduces to :func:`make_mallet_plate` on :func:`gong_linear_twin`.

    ``rho`` inside ``material`` is the plate's **volumetric** density, as :class:`VKPlate` spells
    it. Rectangles only: :class:`VKPlate` has no outline argument, so #7p's "the outline is free"
    does not carry over.
    """
    mat = dict(GONG_MATERIAL)
    if material:
        mat.update(material)
    plate = VKPlate(
        Lx=a, Ly=a, fs=fs, N=N, sigma=sigma, boundary=boundary, nonlinear=nonlinear,
        couple_tol=couple_tol, couple_max_iter=couple_max_iter, couple_method=couple_method,
        **mat,
    )
    return MalletVKPlate(
        plate=plate, mass=mass, stiffness=K, alpha=alpha, hysteresis=hysteresis,
        strike_x=strike_x * a, strike_y=strike_y * plate.Ly,
        strike_velocity=strike_velocity, gap=gap,
        outer_tol=outer_tol, outer_max_iter=outer_max_iter,
    )


def gong_linear_twin(gong: MalletVKPlate) -> MalletPlate:
    """The :class:`MalletPlate` that ``make_mallet_gong(nonlinear=False)`` must reduce to.

    Every number is read back off the gong itself rather than off the arguments that built it --
    :func:`vk_linear_twin`'s four traps apply unchanged (areal density, the *snapped* ``Ly``,
    ``kappa``, and ``nu``), and to them this adds the mallet's own five plus the strike point,
    which is taken as the **snapped** ``(x_strike, y_strike)`` the gong reports. Re-deriving the
    strike from the fractions would let a snapping difference masquerade as a coupling bug.
    """
    vk = gong.plate
    twin = Plate(
        Lx=vk.Lx, Ly=vk.Ly, kappa=vk.kappa, rho=vk.rho_s, fs=vk.fs, N=vk.N,
        sigma=vk.sigma, theta=vk.theta, boundary=vk.boundary, nu=vk.nu,
    )
    return MalletPlate(
        plate=twin, mass=gong.M, stiffness=gong.K, alpha=gong.alpha, hysteresis=gong.lam_h,
        strike_x=gong.x_strike, strike_y=gong.y_strike,
        strike_velocity=gong.strike_velocity, gap=gong.z_H,
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
            Lfree, k=n_modes + 1, M=Cfree, sigma=shift, which="LM", return_eigenvectors=False,
            v0=arpack_v0(Lfree),
        )
        w2 = np.sort(w2)[1 : n_modes + 1]  # drop the ω≈0 constant-pressure mode
    else:
        w2 = eigsh(
            Lfree, k=n_modes, M=Cfree, sigma=0.0, which="LM", return_eigenvectors=False,
            v0=arpack_v0(Lfree),
        )
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


# -- the 3-D air box (HANDOFF §12.H): the distributed tier of the air node ---------------
#
# A small, ordinary room. The default grid is deliberately tiny (0.9 x 0.7 x 0.6 m at h = 10 cm,
# i.e. 10 x 8 x 7 = 560 nodes): 3-D is the first model here where grid cost is a design constraint,
# and every structural/modal oracle is grid-size-independent, so they run where they are free.
# The sample rate is *solved for* from the requested Courant number, exactly as make_bore does --
# but the 3-D ceiling is lambda <= 1/sqrt(3) ~ 0.577, and unlike the 1-D string NO lambda is
# dispersionless, so 0.9 of the ceiling is a default, never a sweet spot.
AIRBOX_ROOM_DEFAULT = (0.9, 0.7, 0.6)  # m
AIRBOX_H_DEFAULT = 0.1                 # m
AIRBOX_CFL_FRACTION = 0.9              # lambda = fraction / sqrt(3)


def make_airbox(
    *,
    L: tuple[float, float, float] = AIRBOX_ROOM_DEFAULT,
    h: float = AIRBOX_H_DEFAULT,
    cfl: float = AIRBOX_CFL_FRACTION,
    walls="rigid",
    source: tuple[float, float, float] | None = None,
    rho0: float = RHO0_AIR,
    c0: float = C0_AIR,
) -> AirBox:
    """Build an :class:`AirBox` whose Courant number is exactly ``cfl / sqrt(3)``.

    ``h`` fixes the grid, so the sample rate is solved for: ``fs = c0 sqrt(3) / (cfl h)``. ``cfl``
    is the *fraction of the 3-D CFL ceiling*, not lambda itself — pass ``cfl=1.0`` to sit exactly
    on ``lambda = 1/sqrt(3)`` (where the corner mode reaches the arcsin's argument of 1) and
    ``cfl > 1`` to check that construction is refused. ``walls`` takes the same token / float /
    per-face mapping the class does.
    """
    fs = c0 * np.sqrt(3.0) / (cfl * h)
    return AirBox(L=L, fs=fs, h=h, walls=walls, source=source, rho0=rho0, c0=c0)


def airbox_noise(box: AirBox, seed: int = 0, amplitude: float = 1.0) -> AirBox:
    """Seed the room with a random pressure field (and rest velocity) — the structural-test IC.

    Broadband noise excites **every** mode at once, including the ones a smooth pulse would miss,
    so an energy identity that survives it has no direction left to hide in.
    """
    rng = np.random.default_rng(seed)
    box.set_state(amplitude * rng.standard_normal(box.p.shape))
    return box


# The two-way port (batch 2): a modal body loaded BY the room rather than merely heard in it.
# A slightly larger default room than make_airbox's, at a coarser h, so a port can sit at an
# interior node, on a wall and in a corner of the same geometry, and so the two-instrument scene
# has room for a measurable travel time -- 11 x 9 x 7 = 693 nodes, still free.
AIRBOX_PORT_ROOM_DEFAULT = (0.5, 0.4, 0.3)  # m -> N = (10, 8, 6) at h = 5 cm
AIRBOX_PORT_H_DEFAULT = 0.05                # m
AIRBOX_PORT_AT_DEFAULT = (0.15, 0.15, 0.15)  # m -> node (3, 3, 3), interior on every axis
# Body: inharmonic pair, a soundboard-scale modal mass, and radiation weights (m^2) big enough
# that the room genuinely loads it within a few hundred steps.
AIRBOX_PORT_FREQS = np.array([220.0, 337.0])
AIRBOX_PORT_MASS = 0.05          # kg
AIRBOX_PORT_RADIATION = 2e-3     # m^2 (a_1; a_2 = 0.65 a_1)
AIRBOX_PORT_Q0 = np.array([1e-3, 5e-4])  # m


def make_room_loaded_body(
    *,
    room: AirBox | None = None,
    L: tuple[float, float, float] = AIRBOX_PORT_ROOM_DEFAULT,
    h: float = AIRBOX_PORT_H_DEFAULT,
    cfl: float = AIRBOX_CFL_FRACTION,
    walls="rigid",
    at: tuple[float, float, float] = AIRBOX_PORT_AT_DEFAULT,
    radius: float | None = None,
    freqs: np.ndarray = AIRBOX_PORT_FREQS,
    sigmas: np.ndarray | float = 0.0,
    masses: np.ndarray | float = AIRBOX_PORT_MASS,
    phi: np.ndarray | float = 1.0,
    radiation: np.ndarray | float | None = None,
    q0: np.ndarray | float | None = AIRBOX_PORT_Q0,
) -> RoomLoadedBody:
    """A modal body loaded by an :class:`AirBox` through a port — the two-way coupling fixture.

    Returns the :class:`RoomLoadedBody`; its room is on ``.room`` and its port on ``.port``. Pass
    ``room=`` to put a **second** instrument in an existing room (their ports must be disjoint) —
    the sample rate then comes from that room, so ``h`` and ``cfl`` are ignored.

    ``radius=None`` (the default) is a point port: exact, cheap, and with a *grid-dependent* load
    magnitude, which is right for every structural oracle and wrong for a physical one. ``sigmas=0``
    keeps the modes lossless so the **only** energy channel is the room, and the conserved statement
    is ``inst.energy() + inst.room.energy()``.
    """
    if room is None:
        room = make_airbox(L=L, h=h, cfl=cfl, walls=walls)
    if radiation is None:
        radiation = AIRBOX_PORT_RADIATION * 0.65 ** np.arange(np.size(freqs))
    body = ModalBody(
        freqs=freqs, fs=room.fs, sigmas=sigmas, masses=masses, phi=phi, radiation=radiation
    )
    inst = RoomLoadedBody(body=body, room=room, at=at, radius=radius)
    if q0 is not None:
        inst.set_state(q0)
    return inst


def room_scene_energy(*instruments: RoomLoadedBody) -> float:
    """The conserved total of a whole scene: ``sum_j inst_j.energy() + room.energy()``.

    The coupling term cancels identically — each port's ``radiated_energy`` *is* the room's
    ``injected``, seen from the other side of the same terminal — so this statement contains no
    coupling term at all, which is why a drift in it is unambiguously a bug rather than accounting.
    Every instrument must share one room.
    """
    room = instruments[0].room
    if any(inst.room is not room for inst in instruments):
        raise ValueError("room_scene_energy expects instruments sharing a single room.")
    return sum(inst.energy() for inst in instruments) + room.energy()


def gaussian_pulse(fs: float, f0: float, *, amplitude: float = 1e-3, widths: float = 4.0):
    """A Gaussian volume-velocity pulse ``q(t)`` and its derivative ``qdot(t)``, centred at
    ``widths`` standard deviations in so it starts (and ends) at numerical zero.

    ``sigma = 1/(2 pi f0)`` puts the spectral peak near ``f0``. Returned as a pair of callables of
    *time in seconds*, plus the effective duration ``2 * widths * sigma`` — which is what sizes the
    reflection-free window in the free-field oracle.
    """
    sigma = 1.0 / (2.0 * np.pi * f0)
    t0 = widths * sigma

    def q(t):
        return amplitude * np.exp(-((t - t0) ** 2) / (2.0 * sigma * sigma))

    def qdot(t):
        return -amplitude * (t - t0) / (sigma * sigma) * np.exp(
            -((t - t0) ** 2) / (2.0 * sigma * sigma)
        )

    return q, qdot, 2.0 * widths * sigma


def plate_bump(plate: Plate, amplitude: float = 1e-3) -> np.ndarray:
    """A smooth off-centre bump on ``plate``'s live nodes — the generic struck initial condition.

    Off-centre so it is not orthogonal to the antisymmetric modes. The free plate gets its mean
    removed: a net piston is the most efficient radiator the geometry has (see §7.8), so leaving it
    in would drown every other channel within a few hundred steps.
    """
    x, y = plate.X[plate.mask], plate.Y[plate.mask]
    width = 0.08 * plate.Lx
    u0 = amplitude * np.exp(
        -(((x - 0.42 * plate.Lx) ** 2 + (y - 0.38 * plate.Ly) ** 2) / (width * width))
    )
    return u0 if plate.boundary == "supported" else u0 - u0.mean()


def plate_mode_shape(plate: Plate, m: int, n: int) -> np.ndarray:
    """The **exact** discrete mode ``sin(m pi x/Lx) sin(n pi y/Ly)`` of a supported plate, rms 1.

    Exact because ``B = L^2`` keeps the sine product an eigenvector of the scheme, which is what
    makes ``sum_i sin(m pi i/N) = 0`` for even ``m`` an *identity* rather than an approximation —
    i.e. what makes an even-index mode's net volume displacement exactly zero.
    """
    if plate.boundary != "supported":
        raise ValueError(
            "the closed-form sine mode is the supported plate's; #5b has no such form."
        )
    x, y = plate.X[plate.mask], plate.Y[plate.mask]
    shape = np.sin(m * np.pi * x / plate.Lx) * np.sin(n * np.pi * y / plate.Ly)
    return shape / np.sqrt(np.mean(shape * shape))


def vk_strike(vk: VKPlate, amplitude: float | None = None, width: float = 0.20) -> np.ndarray:
    """A centred raised-Gaussian strike on ``vk``'s live nodes, peak ``amplitude`` (default ``e``).

    The default amplitude is the plate thickness, i.e. the onset of the nonlinearity — scale it by
    ``w/e`` to move between the frozen control (``w << e``) and the drifting regime (``w ~ 3e``).
    ``width`` is a fraction of ``Lx``, and it is not a free parameter: a *narrow* strike is what
    fails to converge, hitting the Picard cap and producing NaN, so the broad strike is the only
    one that runs at large amplitude.
    """
    amp = vk.e if amplitude is None else amplitude
    w = width * vk.Lx
    dx = vk.X - 0.5 * vk.Lx
    dy = vk.Y - 0.5 * vk.Ly
    field = amp * np.exp(-((dx * dx + dy * dy) / (w * w)))
    return field[vk.mask]
