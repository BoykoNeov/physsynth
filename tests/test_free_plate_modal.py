"""Free-edge (FFFF) plate (model #5b): the energy-first operator + its three independent oracles.

The completely free plate is the **curved-Chladni** resonator (``docs/dev/plate-free-edge-plan.md``
Part 1). Unlike the simply-supported plate it has **no closed-form modal oracle** (free edges have
no ``sin·sin`` eigenvector), so the tight ~1-cent bar is gone. Validation rests on three independent
anchors of increasing externality:

1. the **rigid-body nullspace** ``{1, x, y}`` (machine precision) with the ``K(xy)≠0`` twist
   counter-check (scaling with ``(1-ν)`` — the dropped-Poisson catch);
2. **O(h²) self-convergence** of the low generalized eigenvalues (Richardson, no external data);
3. **Leissa's FFFF-square frequency parameters** (percent-level absolute anchor), with the
   fundamental being the **saddle/twist**, not a drum bulge.
"""

import numpy as np
from helpers import (
    KAPPA_PLATE_DEFAULT,
    free_plate_low_eigenfrequencies,
    make_free_plate,
)
from scipy import sparse
from scipy.sparse.linalg import eigsh

from physsynth.analysis import modal, spectrum
from physsynth.core.engine import simulate
from physsynth.core.exciter import raised_cosine_2d
from physsynth.core.operators import free_beam_stiffness
from physsynth.core.operators2d import biharmonic_from_mask, free_plate_stiffness, rectangle_mask

KAPPA = KAPPA_PLATE_DEFAULT


# -- A fully independent (explicit-loop, no Kronecker) reference assembly of the energy form. ------
def _dense_free_plate_K(Nx, Ny, h, nu):
    """Assemble ``K`` by explicit per-node loops — the cross-check on the kron ordering."""
    nn = (Nx + 1) * (Ny + 1)

    def idx(i, j):  # node (i in x, j in y) -> C-order flat index (j outer, i inner)
        return j * (Nx + 1) + i

    inv_h2 = 1.0 / (h * h)
    Dxx = np.zeros((nn, nn))
    Dyy = np.zeros((nn, nn))
    for j in range(Ny + 1):  # u_xx at interior-x nodes (zero at the x free edges)
        for i in range(1, Nx):
            r = idx(i, j)
            Dxx[r, idx(i - 1, j)] += inv_h2
            Dxx[r, idx(i, j)] += -2.0 * inv_h2
            Dxx[r, idx(i + 1, j)] += inv_h2
    for j in range(1, Ny):  # u_yy at interior-y nodes
        for i in range(Nx + 1):
            r = idx(i, j)
            Dyy[r, idx(i, j - 1)] += inv_h2
            Dyy[r, idx(i, j)] += -2.0 * inv_h2
            Dyy[r, idx(i, j + 1)] += inv_h2

    wa = np.zeros(nn)
    for j in range(Ny + 1):
        wy = h if 0 < j < Ny else 0.5 * h
        for i in range(Nx + 1):
            wx = h if 0 < i < Nx else 0.5 * h
            wa[idx(i, j)] = wx * wy
    Wa = np.diag(wa)

    cross = Dxx.T @ Wa @ Dyy
    K = Dxx.T @ Wa @ Dxx + Dyy.T @ Wa @ Dyy + nu * (cross + cross.T)

    ncell = Nx * Ny  # cell-centered twist u_xy on the Nx*Ny cells
    Dxy = np.zeros((ncell, nn))
    for j in range(Ny):
        for i in range(Nx):
            c = j * Nx + i
            Dxy[c, idx(i, j)] += inv_h2
            Dxy[c, idx(i + 1, j)] += -inv_h2
            Dxy[c, idx(i, j + 1)] += -inv_h2
            Dxy[c, idx(i + 1, j + 1)] += inv_h2
    K += 2.0 * (1.0 - nu) * (h * h) * (Dxy.T @ Dxy)
    return K


# -- Operator symmetry: the energy-first Gram form is symmetric by construction (=> conservation). -
def test_operator_symmetric():
    K, _, _ = free_plate_stiffness(12, 10, 0.1, 0.3)
    asym = abs(K - K.T).max()
    assert asym < 1e-12, f"K not symmetric: ||K-K^T||_max = {asym:.3e}"


# -- Ordering / assembly: production kron build == an independent explicit-loop assembly. ----------
def test_matches_direct_assembly():
    """The Kronecker ordering must match the C-order (embed/index_map) flattening. A non-square
    grid (Nx != Ny) is used so any x<->y axis swap would show up."""
    for Nx, Ny, nu in [(4, 4, 0.3), (5, 3, 0.3), (4, 5, 0.0), (3, 4, 0.49)]:
        h = 0.1
        K, W, index_map = free_plate_stiffness(Nx, Ny, h, nu)
        K_dense = _dense_free_plate_K(Nx, Ny, h, nu)
        diff = np.abs(K.toarray() - K_dense).max()
        assert diff < 1e-12, f"kron K != direct assembly at (Nx,Ny,nu)=({Nx},{Ny},{nu}): {diff:.2e}"
        # index_map is the trivial full-grid C-order map (every node live).
        assert np.array_equal(index_map.ravel(), np.arange((Nx + 1) * (Ny + 1)))
        assert np.allclose(W.diagonal().min(), 0.25 * h * h)  # corner weight h^2/4


# -- Money test: rigid-body nullspace {1,x,y} (machine) + the saddle xy counter-check (nonzero). ---
def test_rigid_body_nullspace():
    """``K{1,x,y}=0`` to machine precision; ``K(xy)≠0``. Tolerances are RELATIVE (the absolute
    residual of ``K@x`` scales as ``eps·‖K‖``, growing as h shrinks), so the discriminating signal
    is the *contrast*: the rigid modes sit ~10+ orders below the saddle ``xy``."""
    Nx, Ny, h = 14, 12, 0.1
    K, _, _ = free_plate_stiffness(Nx, Ny, h, 0.3)
    xs = np.arange(Nx + 1) * h
    ys = np.arange(Ny + 1) * h
    X, Y = np.meshgrid(xs, ys)  # (Ny+1, Nx+1), C-order matches K
    k_fro = np.sqrt((K.toarray() ** 2).sum())

    fields = {
        "1": np.ones_like(X).ravel(),
        "x": X.ravel(),
        "y": Y.ravel(),
        "xy": (X * Y).ravel(),
    }
    rel = {name: np.linalg.norm(K @ v) / (k_fro * np.linalg.norm(v)) for name, v in fields.items()}
    for name in ("1", "x", "y"):
        assert rel[name] < 1e-12, f"K@{name} not in nullspace: rel residual {rel[name]:.2e}"
    # The saddle xy carries real twist energy (only the (1-ν) term feeds it); a dropped-ν operator
    # would spuriously kill it. Demand many orders of contrast against the rigid-body residuals.
    assert rel["xy"] > 1e-9, f"K@xy spuriously ~0 (dropped-ν?): rel {rel['xy']:.2e}"
    assert rel["xy"] > 1e6 * max(rel["1"], rel["x"], rel["y"]), "nullspace contrast too small"


# -- The saddle xy energy scales exactly with (1-ν): the quantitative Poisson check. ---------------
def test_xy_energy_scales_with_one_minus_nu():
    Nx, Ny, h = 12, 10, 0.1
    xs = np.arange(Nx + 1) * h
    ys = np.arange(Ny + 1) * h
    xy = (np.meshgrid(xs, ys)[0] * np.meshgrid(xs, ys)[1]).ravel()
    scaled = []
    for nu in (0.0, 0.3, 0.49, -0.5):
        K, _, _ = free_plate_stiffness(Nx, Ny, h, nu)
        scaled.append(np.linalg.norm(K @ xy) / (1.0 - nu))
    assert np.allclose(scaled, scaled[0], rtol=1e-12), f"K@xy not ∝ (1-ν): {scaled}"


# -- Exactly 3 zero modes: the cell-centered twist has NO checkerboard nullspace to pollute them. --
def test_exactly_three_zero_modes():
    """A collocated centred ``u_xy`` would add a spurious ``(-1)^{i+j}`` near-zero mode (a 4th). The
    cell-centered twist must leave exactly the 3 rigid-body modes ``{1, x, y}`` near zero."""
    p = make_free_plate(N=20)
    a = p.Lx
    mu1 = (13.0 / (a * a)) ** 2
    vals = np.sort(eigsh(p.K, k=8, M=p.W, sigma=-1e-3 * mu1, which="LM", return_eigenvectors=False))
    n_zero = int(np.sum(np.abs(vals) < 1e-3 * abs(vals[3])))
    assert n_zero == 3, f"expected exactly 3 near-zero modes, got {n_zero}: {vals}"


# -- Single source of truth: the bending-diagonal is the validated free-beam operator in each axis.-
def test_bending_diagonal_is_beam_operator():
    """The plate's bending-diagonal equals the **validated free-beam stiffness** applied along each
    axis: ``C2xᵀWaC2x + C2yᵀWaC2y == kron(M_y, S_x) + kron(S_y, M_x)`` with ``S, M =
    free_beam_stiffness``. So the plate's bending inherits the beam's proven symmetry, per-line
    ``{1, x}`` nullspace and O(h²) — re-derived nowhere. (White-box: compares the internal
    collocated build against the public beam operator.)"""
    from physsynth.core.operators2d import _collocated_d2_1d

    Nx, Ny, h = 9, 7, 0.1
    C2x = sparse.kron(sparse.identity(Ny + 1), _collocated_d2_1d(Nx, h))
    C2y = sparse.kron(_collocated_d2_1d(Ny, h), sparse.identity(Nx + 1))
    mx = np.full(Nx + 1, h)
    mx[0] = mx[-1] = 0.5 * h
    my = np.full(Ny + 1, h)
    my[0] = my[-1] = 0.5 * h
    Wa = sparse.diags(np.kron(my, mx))
    bend_collocated = (C2x.T @ Wa @ C2x + C2y.T @ Wa @ C2y).tocsr()

    Sx, Mx = free_beam_stiffness(Nx, h)
    Sy, My = free_beam_stiffness(Ny, h)
    bend_beam = (sparse.kron(My, Sx) + sparse.kron(Sy, Mx)).tocsr()
    assert abs(bend_collocated - bend_beam).max() < 1e-12, "bending-diagonal != free-beam operator"


# -- O(h²) self-convergence of the low eigenvalues (Richardson; needs no external table). ---------
def test_self_convergence_order_h2():
    Ns = [20, 40, 80]  # h, h/2, h/4 -> Richardson order = log2((mu_h - mu_h/2)/(mu_h/2 - mu_h/4))
    mus = []
    for N in Ns:
        p = make_free_plate(N=N)
        # generalized low eigenvalues mu = (2π f / κ)² (undo the f-mapping to compare raw mu)
        f = free_plate_low_eigenfrequencies(p, 4)
        mus.append((2.0 * np.pi * f / p.kappa) ** 2)
    mus = np.array(mus)  # shape (3, 4)
    d1 = mus[0] - mus[1]
    d2 = mus[1] - mus[2]
    orders = np.log2(np.abs(d1) / np.abs(d2))
    assert np.all(np.abs(d2) < np.abs(d1)), f"eigenvalues not converging: {mus}"
    assert orders[0] > 1.8, f"fundamental self-convergence order {orders[0]:.2f} < 1.8 (want ~2)"
    assert np.min(orders) > 1.6, f"low-mode convergence orders {orders} (min should be ~2)"


# -- Leissa absolute anchor: lowest 5 elastic modes of the FFFF square match the tabulated λ. ------
def test_leissa_ffff_square_anchor():
    """Percent-level absolute anchor (no closed form). Match by **sorted eigenvalue** (modes 4,5 are
    a degenerate pair) and confirm the error shrinks as the grid refines."""
    lambdas = modal.free_plate_ffff_square_lambdas()
    errs = {}
    for N in (32, 64):
        p = make_free_plate(N=N)
        a = p.Lx
        f_oracle = modal.free_plate_freq_from_lambda(lambdas, p.kappa, a)
        f_meas = free_plate_low_eigenfrequencies(p, len(lambdas))
        errs[N] = np.abs(f_meas - f_oracle) / f_oracle
    assert np.max(errs[64]) < 0.006, f"Leissa off {np.max(errs[64]):.3%} at N=64 (want <0.6%)"
    assert np.max(errs[64]) < np.max(errs[32]), "error not decreasing with refinement"


# -- The fundamental is the SADDLE/twist (diagonal nodal lines), not a drum-like bulge. -----------
def test_fundamental_is_saddle():
    """The single best qualitative catch for a wrong (ν-dropped) operator: a free plate's lowest
    elastic mode is the twisting saddle — corners alternate sign, the centre is a node."""
    p = make_free_plate(N=40)
    a = p.Lx
    mu1 = (13.0 / (a * a)) ** 2
    vals, vecs = eigsh(p.K, k=6, M=p.W, sigma=-1e-3 * mu1, which="LM")
    order = np.argsort(vals)
    phi = vecs[:, order[3]]  # first elastic mode (after the 3 rigid)
    field = phi.reshape(p.N + 1, p.N + 1)
    field = field / np.abs(field).max()
    c00, c0N, cN0, cNN = field[0, 0], field[0, -1], field[-1, 0], field[-1, -1]
    center = field[p.N // 2, p.N // 2]
    # Saddle: the two diagonals carry opposite sign; the centre is ~0 (a nodal point).
    assert c00 * cNN > 0.5, f"corners (0,0)&(N,N) should share sign (saddle): {c00:.2f},{cNN:.2f}"
    assert c0N * cN0 > 0.5, f"corners (0,N)&(N,0) should share sign (saddle): {c0N:.2f},{cN0:.2f}"
    assert c00 * c0N < -0.5, "adjacent corners should be opposite sign (diagonal nodal lines)"
    assert abs(center) < 0.1, f"centre should be a node for the saddle, got {center:.3f} (bulge?)"


# -- End-to-end: the time-stepper actually rings at the discrete fundamental (FFT sanity). --------
def test_fft_rings_at_fundamental():
    p = make_free_plate(N=40, mu=1.0)
    a = p.Lx
    f_spatial = free_plate_low_eigenfrequencies(p, 2)
    mu = (2.0 * np.pi * f_spatial / p.kappa) ** 2
    f_disc = modal.discrete_beam_eigenfrequency(mu, p.kappa, p.k, p.theta)  # same θ-scheme W/K map
    # Off-centre bump breaks symmetry -> excites the low modes; pick up away from nodal lines.
    field = raised_cosine_2d(p.X, p.Y, (0.3 * a, 0.62 * a), 0.3 * a, amplitude=1e-3)
    p.set_state(field)
    pickup = p.pickup_index_at(0.18 * a, 0.22 * a)
    res = simulate(p, num_steps=int(0.5 * p.fs), pickup_index=pickup)
    found = spectrum.measure_partials_near(
        res.output, res.fs, np.array([f_disc[0]]), search_hz=20.0
    )[0]
    cents = abs(modal.cents(found, f_disc[0]))
    assert cents < 8.0, f"FFT fundamental off {cents:.2f} cents ({found:.2f} vs {f_disc[0]:.2f})"


# -- The resonator builds exactly what the standalone operator helper returns (single truth). ------
def test_resonator_uses_operator_helper():
    p = make_free_plate(N=16)
    Ny = round(p.Ly / p.h)
    K, W, _ = free_plate_stiffness(p.N, Ny, p.h, p.nu)
    assert abs(K - p.K).max() < 1e-12, "resonator K != free_plate_stiffness K"
    assert abs(W - p.W).max() < 1e-12, "resonator W != free_plate_stiffness W"


# -- SS regression (plan test #8): the generalized W/K eigen-map fed the SS operators (W=h²I,K=h²B)
# reproduces model #5's frequencies -- i.e. the free machinery is a correct generalization. --------
def test_ss_operators_through_generalized_map_match_model5():
    N, h = 32, 1.0 / 32
    mask = rectangle_mask(N, N)
    B, _ = biharmonic_from_mask(mask, h)
    n = B.shape[0]
    K_ss = (h * h) * B  # the unified scheme's SS special case: K = h²B, W = h²I
    W_ss = sparse.identity(n, format="csr") * (h * h)
    # Generalized K φ = μ W φ -> μ = Λ² (the biharmonic eigenvalue); f = κ√μ/2π = κΛ/2π.
    mu = np.sort(eigsh(K_ss, k=6, M=W_ss, sigma=0.0, which="LM", return_eigenvectors=False))
    kappa = KAPPA
    f_gen = kappa * np.sqrt(mu) / (2.0 * np.pi)
    # Model #5's own oracle (squared-Laplacian eigenvalues mapped to continuum plate frequencies).
    modes = [(m, nn) for m in range(1, 4) for nn in range(1, 4)]
    Lam = np.sort(modal.rectangular_discrete_eigenvalues(h, N, N, modes))[:6]
    f_ss = kappa * Lam / (2.0 * np.pi)  # f = κ·Λ/2π (Λ = Laplacian eigenvalue, biharmonic = Λ²)
    rel = np.max(np.abs(f_gen - f_ss) / f_ss)
    assert rel < 1e-9, f"generalized W/K map on SS operators != model #5 spectrum: rel {rel:.2e}"
