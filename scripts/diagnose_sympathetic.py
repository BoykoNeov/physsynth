"""Diagnostics for sympathetic / coupled strings: several fixed/free strings sharing one bridge
point on a common modal body (models: IdealString x N + ModalBody + SympatheticStrings).

Renders the diagnostics-as-visuals and prints the real validation numbers:

  1. **Normal-mode structure (the discriminating oracle).** Two identical strings started in the
     **antisymmetric** shape (u_B = -u_A) keep the bridge displacement ``w_b`` pinned at zero and
     the body's energy at zero forever — the string pair rings on a mode the bridge cannot feel.
     The **symmetric** start (u_B = +u_A) is its opposite: the bridge swings and energy floods into
     the body. Energy conservation cannot tell these apart; the bridge trace can.
  2. **Sympathetic transfer.** Pluck one string; the energy sloshes into a neighbour tuned to it
     (near-complete exchange, the classic coupled-oscillator beating) but barely touches a detuned
     one. This is why an undamped sympathetic string "lights up" only for the right note.
  3. **Weinreich two-stage decay.** Two slightly-mistuned unison strings over a *lossy* bridge,
     plucked together: the in-phase (symmetric) motion couples strongly to the lossy bridge and dies
     fast (the "prompt" sound); the out-of-phase motion couples weakly and lingers (the
     "aftersound"). The total string energy shows the tell-tale fast-then-slow two-slope decay of a
     real piano unison.

Run:  python scripts/diagnose_sympathetic.py
Figures land in ./out/.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from physsynth.core.body import ModalBody  # noqa: E402
from physsynth.core.connection import SympatheticStrings  # noqa: E402
from physsynth.core.exciter import triangular_pluck  # noqa: E402
from physsynth.core.string_ideal import IdealString  # noqa: E402

# Canonical strings (c = 200 m/s) below lambda = 1 for coupling headroom; a light modal body
# (masses ~0.02 kg, comparable to the string's rho*L = 0.005 kg) so it genuinely reacts.
L, RHO = 1.0, 0.005
N, LAM = 100, 0.9
BODY_FREQS = np.array([137.0, 213.0, 330.0, 471.0, 620.0])  # off-harmonic body resonances
BODY_MASS = 0.02
OUT = "out"


def build(Ts, K, *, sigma_body=0.0):
    """A :class:`SympatheticStrings` of strings with tensions ``Ts`` on a shared body.

    ``fs`` is set from the first tension so string 0 sits at ``lambda = LAM``; detune *downward*
    (lower tension) to keep every string below the string Nyquist mode.
    """
    Ts = np.atleast_1d(np.asarray(Ts, dtype=float))
    c0 = float(np.sqrt(Ts[0] / RHO))
    fs = c0 * N / (L * LAM)
    strings = [
        IdealString(L=L, T=float(t), rho=RHO, fs=fs, N=N, boundary=("fixed", "free"))
        for t in Ts
    ]
    body = ModalBody(freqs=BODY_FREQS, fs=fs, masses=BODY_MASS, sigmas=sigma_body, phi=1.0)
    Ks = np.full(len(strings), K, dtype=float)
    return SympatheticStrings(strings=strings, body=body, Ks=Ks), fs


# -- 1. normal-mode structure: antisymmetric bridge stays still, symmetric drives it ----------
def normal_modes(K=8000.0, secs=0.25):
    out = {}
    for name, sign in (("antisymmetric", -1.0), ("symmetric", +1.0)):
        symp, fs = build([200.0, 200.0], K)
        a, b = symp.strings
        pluck = triangular_pluck(a.x, a.L, 0.137 * a.L, amplitude=1e-3)
        a.set_state(pluck)
        b.set_state(sign * pluck)
        steps = int(secs * fs)
        e0 = symp.energy()
        t = np.arange(steps + 1) / fs
        wb = np.empty(steps + 1)
        ebody = np.empty(steps + 1)
        wb[0], ebody[0] = symp._bridge_displacement(), symp.body.energy()
        for i in range(1, steps + 1):
            symp.step()
            wb[i], ebody[i] = symp._bridge_displacement(), symp.body.energy()
        out[name] = (t, wb, ebody / e0)
    return out


# -- 2. sympathetic transfer: tuned neighbour drains the energy, detuned stays quiet ----------
def transfer(K=1500.0, secs=1.5):
    out = {}
    for name, Ts in (("unison (tuned)", [200.0, 200.0]), ("detuned ~4 semis", [200.0, 120.0])):
        symp, fs = build(Ts, K)
        s0 = symp.strings[0]
        s0.set_state(triangular_pluck(s0.x, s0.L, 0.137 * s0.L, amplitude=1e-3))
        steps = int(secs * fs)
        t = np.arange(steps + 1) / fs
        e0 = np.empty(steps + 1)
        e1 = np.empty(steps + 1)
        tot = symp.energy()
        e0[0], e1[0] = symp.string_energy(0) / tot, symp.string_energy(1) / tot
        for i in range(1, steps + 1):
            symp.step()
            e0[i] = symp.string_energy(0) / tot
            e1[i] = symp.string_energy(1) / tot
        out[name] = (t, e0, e1)
    return out


# -- 3. Weinreich two-stage decay: prompt (symmetric, lossy) + aftersound (antisymmetric) -----
def two_stage_decay(K=6000.0, sigma_body=10.0, secs=4.0):
    # Two slightly-mistuned unison strings (a real piano unison is never perfectly in tune) over a
    # lossy bridge. Striking *one* string excites the symmetric and antisymmetric normal modes in
    # equal measure: the symmetric motion loads the lossy bridge and decays fast (the "prompt"
    # sound); the antisymmetric motion barely loads it and lingers (the "aftersound"). The total
    # string energy therefore shows the tell-tale fast-then-slow knee.
    symp, fs = build([200.0, 200.0 * (1.0 + 0.004)], K, sigma_body=sigma_body)
    s0 = symp.strings[0]
    s0.set_state(triangular_pluck(s0.x, s0.L, 0.137 * s0.L, amplitude=1e-3))
    steps = int(secs * fs)
    t = np.arange(steps + 1) / fs
    e_str = np.empty(steps + 1)
    e_str[0] = symp.string_energy(0) + symp.string_energy(1)
    for i in range(1, steps + 1):
        symp.step()
        e_str[i] = symp.string_energy(0) + symp.string_energy(1)
    return t, e_str / e_str[0]


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (1) normal modes
    nm = normal_modes()
    ax = axes[0, 0]
    for name, (t, wb, _) in nm.items():
        ax.plot(t * 1e3, wb * 1e3, label=name, lw=1.2)
    ax.set_title("1. Bridge motion by normal mode\n(antisymmetric ≡ 0: the bridge cannot feel it)")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("bridge displacement $w_b$ (mm)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for name, (t, _, ebody) in nm.items():
        ax.plot(t * 1e3, ebody, label=name, lw=1.2)
    ax.set_title("1b. Body energy fraction by normal mode")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("$E_{body}/E_0$")
    ax.legend()
    ax.grid(alpha=0.3)

    # (2) sympathetic transfer
    tr = transfer()
    ax = axes[1, 0]
    colours = {"unison (tuned)": "C2", "detuned ~4 semis": "C3"}
    for name, (t, _e0, e1) in tr.items():
        ax.plot(t, e1, color=colours[name], lw=1.3, label=f"neighbour — {name}")
    ax.set_title("2. Sympathetic transfer\n(energy that reaches the un-plucked neighbour)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("$E_{string\\,1}/E_{tot}$")
    ax.legend()
    ax.grid(alpha=0.3)

    # (3) Weinreich two-stage decay
    t, e = two_stage_decay()
    ax = axes[1, 1]
    ax.semilogy(t, e, "C0", lw=1.0)
    ax.set_title("3. Weinreich two-stage decay\n(fast prompt + slow aftersound, a piano unison)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("total string energy (norm., log)")
    ax.set_ylim(1e-4, 2.0)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    path = os.path.join(OUT, "sympathetic.png")
    fig.savefig(path, dpi=120)
    print(f"wrote {path}")

    # -- printed validation numbers ---------------------------------------------------------
    (_, wb_anti, eb_anti) = nm["antisymmetric"]
    (_, wb_sym, eb_sym) = nm["symmetric"]
    print("\nnormal-mode oracle:")
    print(f"  antisymmetric  max|w_b| = {np.max(np.abs(wb_anti)):.2e} m   "
          f"max E_body/E0 = {np.max(eb_anti):.2e}   (both ~0: bridge stays still)")
    print(f"  symmetric      max|w_b| = {np.max(np.abs(wb_sym)):.2e} m   "
          f"max E_body/E0 = {np.max(eb_sym):.2e}   (bridge swings, body loads up)")

    peak_tuned = np.max(tr["unison (tuned)"][2])
    peak_detuned = np.max(tr["detuned ~4 semis"][2])
    print("\nsympathetic transfer (peak neighbour energy fraction):")
    print(f"  tuned unison   {peak_tuned:.1%}    detuned {peak_detuned:.1%}    "
          f"ratio {peak_tuned / max(peak_detuned, 1e-9):.1f}x")


if __name__ == "__main__":
    main()
