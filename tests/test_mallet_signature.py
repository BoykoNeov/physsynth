"""Physical-signature validation for the mallet -> membrane strike (model #7).

Energy conservation proves the scheme is *consistent*; these tests prove it is the *right physics*.
A struck drumhead spreads energy across many modes; a harder mallet (stiffer felt) makes briefer
contact and a brighter spectrum; the mallet **bounces** (one contact, then it flies clear); and the
strike position sets which modes speak — a strike on a node line of a mode cannot excite it (the
mode-comb, here the (2,1) mode nulled by a centre strike of a square head).
"""

import numpy as np
from helpers import make_mallet, make_membrane

from physsynth.core.mallet import MalletMembrane


def _spectral_centroid(sig, fs):
    """Amplitude-weighted mean frequency (Hz) of ``sig`` — a brightness proxy."""
    sig = sig - np.mean(sig)
    mag = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)
    return float(np.sum(freqs * mag) / (np.sum(mag) + 1e-30))


def _mode_basis(mem, modes):
    """Normalized analytic eigenmodes ``sin(m*pi*x/Lx) sin(n*pi*y/Ly)`` at the live nodes."""
    live = mem.index_map >= 0
    xs, ys = mem.X[live], mem.Y[live]
    basis = []
    for m, n in modes:
        phi = np.sin(m * np.pi * xs / mem.Lx) * np.sin(n * np.pi * ys / mem.Ly)
        basis.append(phi / np.linalg.norm(phi))
    return np.asarray(basis)  # (n_modes, n_live)


def test_strike_excites_multiple_modes():
    # A single strike spreads energy across several membrane modes (not one pure tone). Measured by
    # projecting onto the analytic eigenmodes — more robust than FFT peak-counting.
    mal = make_mallet(N=40, K=8.0e4, strike_x=0.3, strike_y=0.4)
    mem = mal.membrane
    modes = [(1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (1, 3), (3, 3)]
    basis = _mode_basis(mem, modes)
    peak_e = 0.0
    max_proj = np.zeros(len(modes))
    for _ in range(6000):
        mal.step()
        peak_e = max(peak_e, mem.energy())
        max_proj = np.maximum(max_proj, np.abs(basis @ mem.u))
    # The head takes a real share of the strike energy at the peak of contact (the elastic mallet
    # then bounces off carrying much of it back, so the *instantaneous* end-energy is smaller).
    assert peak_e > 1e-2 * mal.energy(), "the strike deposited almost no energy in the head"
    n_modes = int(np.sum(max_proj > 0.1 * max_proj.max()))
    assert n_modes >= 3, f"only {n_modes} modes meaningfully excited — strike not broadband"


def test_harder_mallet_is_shorter_and_brighter():
    # A stiffer felt makes briefer contact and a brighter (higher-centroid) spectrum.
    soft = make_mallet(K=1.0e4, strike_x=0.3, strike_y=0.4)
    hard = make_mallet(K=2.0e5, strike_x=0.3, strike_y=0.4)
    idx_s = soft.membrane.pickup_index_at(0.6, 0.55)
    idx_h = hard.membrane.pickup_index_at(0.6, 0.55)

    contact_soft = contact_hard = 0
    sig_s = np.empty(6000)
    sig_h = np.empty(6000)
    for i in range(6000):
        soft.step()
        hard.step()
        contact_soft += int(soft.in_contact)
        contact_hard += int(hard.in_contact)
        sig_s[i] = soft.displacement_at(idx_s)
        sig_h[i] = hard.displacement_at(idx_h)

    assert contact_hard < contact_soft, (
        f"stiffer felt should shorten contact: hard={contact_hard} soft={contact_soft} steps"
    )
    fs = soft.membrane.fs
    assert _spectral_centroid(sig_h, fs) > _spectral_centroid(sig_s, fs), (
        "stiffer felt should brighten the spectrum"
    )


def test_mallet_bounces_off():
    # The mallet arrives moving into the head and leaves moving away: one contact, then separation.
    mal = make_mallet(N=40, K=1.0e5, strike_x=0.3, strike_y=0.4, strike_velocity=3.0)
    contacts = []
    for _ in range(2500):
        mal.step()
        contacts.append(mal.in_contact)
    contacts = np.asarray(contacts)
    # Exactly one contact episode within the window (bounces clear, no immediate re-strike).
    intervals = int(np.sum((~contacts[:-1]) & contacts[1:])) + int(contacts[0])
    assert intervals == 1, f"expected a single bounce, saw {intervals} contact episodes"
    assert not contacts[-1], "mallet never separated from the head"
    assert mal.mallet_velocity() > 0.0, "mallet did not rebound (still moving into the head)"


def test_strike_position_mode_comb():
    # A centre strike of a SQUARE head sits on the node line of the (2,1) mode (sin(2*pi*x) = 0 at
    # x = L/2), so it cannot excite it; an off-centre strike does. The mode-comb signature.
    def max_mode21_projection(strike_x, strike_y):
        mal = make_mallet(N=40, lam=0.5, strike_x=strike_x, strike_y=strike_y)
        mem = mal.membrane
        live = mem.index_map >= 0
        xs, ys = mem.X[live], mem.Y[live]
        phi = np.sin(2.0 * np.pi * xs / mem.Lx) * np.sin(np.pi * ys / mem.Ly)
        phi = phi / np.linalg.norm(phi)
        proj = 0.0
        for _ in range(4000):
            mal.step()
            proj = max(proj, abs(float(mem.u @ phi)))
        return proj

    centre = max_mode21_projection(0.5, 0.5)
    offset = max_mode21_projection(0.3, 0.4)
    assert offset > 50.0 * (centre + 1e-30), (
        f"(2,1) mode should be nulled by a centre strike: centre={centre:.2e} offset={offset:.2e}"
    )


def test_membrane_and_mallet_share_a_timestep():
    # The wrapper exposes the membrane's timestep and field unchanged (Resonator protocol).
    mem = make_membrane(domain="rectangle", N=32, lam=0.5)
    mal = MalletMembrane(
        membrane=mem, mass=0.02, stiffness=5.0e4, alpha=2.3,
        strike_x=0.4, strike_y=0.4, strike_velocity=2.0,
    )
    assert mal.k == mem.k
    assert mal.state.shape == mem.mask.shape
