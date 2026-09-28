//! Sparse symmetric eigenpairs nearest a shift — what the viewer asked of
//! `scipy.sparse.linalg.eigsh(K, k, M, sigma=..., which="LM")`.
//!
//! Solves `K x = lambda M x` for the `nev` eigenvalues nearest `sigma`, with `K` symmetric and `M`
//! symmetric positive definite (the identity when `None`). Every call site in the viewer is one of
//! the grid operators — a membrane's Laplacian, a plate's biharmonic with its lumped areas, a
//! bore's pressure Laplacian with its compliances — and asks for the lowest few tens of modes of a
//! system of a few thousand unknowns, so this is written for that and nothing larger.
//!
//! # The method: shift-invert, a block Krylov space, Rayleigh-Ritz
//!
//! `A = K - sigma M` is factored once with [`SparseLu`], and the iteration runs on
//! `Op = A^-1 M`, whose eigenvalues `theta = 1 / (lambda - sigma)` are largest exactly where
//! `lambda` is nearest the shift. `Op` is self-adjoint in the `M` inner product, so a basis `Q`
//! that is `M`-orthonormal turns the problem into the small symmetric one `H = Q^T M Op Q`, solved
//! by [`crate::eig::symmetric_eigen`]. That is ARPACK's mode 3 in outline; what differs is how the
//! space grows.
//!
//! **Blocks of three, because the operators have repeated eigenvalues.** A square membrane or
//! plate has exact `(m, n)` / `(n, m)` pairs, and a grid disk has the cos/sin pairs of its
//! symmetry group. A single-vector Krylov space holds only ONE direction of each eigenspace in
//! exact arithmetic, and finds the second only as rounding error slowly feeds it in: the classic
//! way a Lanczos run silently reports a double eigenvalue once. A block of `b` random vectors
//! finds multiplicities up to `b` by construction. Three covers every symmetry group these grids
//! have, where the largest irreducible representation is two-dimensional, with one to spare.
//!
//! **Full reorthogonalization, twice, and no restarts.** The space only grows; every new vector is
//! orthogonalized against the whole basis twice (classical Gram-Schmidt run twice is as good as
//! modified Gram-Schmidt, and "twice is enough" is Kahan's), and the basis stays small because the
//! wanted end converges long before it would matter. A vector that orthogonalizes away to nothing
//! is replaced by a fresh random one, so the space keeps growing and reaches the whole space if it
//! has to, where the answer is exact.
//!
//! **Converged means the residual, not the eigenvalue.** A Ritz pair `(theta, y)` is accepted when
//! `||Op y - theta y||_M <= RESIDUAL_TOL |theta|`. The eigenvalue error is then of the order of the
//! square of that over the gap to the next eigenvalue, far below what any payload rounds to.
//!
//! # What is NOT promised
//!
//! Agreement with SciPy to the bit. ARPACK's implicit restarts, its start vector and LAPACK's
//! arithmetic all differ from this, so eigenvalues agree to a tolerance (retirement plan §23.11),
//! and each eigenvector's sign and the basis inside a repeated eigenvalue are arbitrary here as
//! they are there. The start vectors are a fixed deterministic sequence, so THIS routine is
//! reproducible run to run.

use crate::eig::{symmetric_eigen, EigError};
use crate::sparse::Csr;
use crate::sparse_lu::{SparseLu, SparseLuError};

/// Vectors added per step: the largest multiplicity found by construction.
pub const BLOCK: usize = 3;

/// Accept a Ritz pair when `||Op y - theta y||_M <= RESIDUAL_TOL * |theta|`.
pub const RESIDUAL_TOL: f64 = 1e-10;

/// A new vector that keeps less than this fraction of its `M`-norm through orthogonalization is
/// treated as already in the space, and replaced by a fresh random one.
const DEPENDENT_TOL: f64 = 1e-10;

/// The eigenpairs nearest the shift.
#[derive(Debug, Clone)]
pub struct Eigenpairs {
    /// Eigenvalues, ascending.
    pub values: Vec<f64>,
    /// `vectors[j]` goes with `values[j]`; the set is `M`-orthonormal. Sign and the basis inside
    /// a repeated eigenvalue are arbitrary.
    pub vectors: Vec<Vec<f64>>,
    /// Size of the Krylov basis when the wanted pairs converged — a cost diagnostic.
    pub basis_size: usize,
}

/// Why the eigenpairs could not be computed.
#[derive(Debug, Clone, PartialEq)]
pub enum EigsError {
    /// `K` is not square, `M` is not its size, or `nev` is not in `[1, n)`.
    BadShape(String),
    /// `K - sigma M` could not be factored (the shift sits on an eigenvalue).
    Factor(SparseLuError),
    /// The small projected problem failed, which a finite symmetric one does not.
    Projected(EigError),
}

impl std::fmt::Display for EigsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EigsError::BadShape(s) => write!(f, "{s}"),
            EigsError::Factor(e) => write!(f, "factor of K - sigma M failed: {e}"),
            EigsError::Projected(e) => write!(f, "projected eigenproblem failed: {e}"),
        }
    }
}

impl std::error::Error for EigsError {}

/// `a . b`, left to right.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for (x, y) in a.iter().zip(b) {
        s += x * y;
    }
    s
}

/// SplitMix64: a fixed, structureless sequence of start vectors. A structured start (all ones,
/// say) is orthogonal to every antisymmetric mode of a symmetric grid and would never find one.
struct SplitMix(u64);

impl SplitMix {
    fn next_unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // 53 random bits onto [-1, 1)
        (z >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }

    fn vector(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.next_unit()).collect()
    }
}

/// The growing `M`-orthonormal basis, with `M q` kept beside each `q`.
struct Basis<'a> {
    m: Option<&'a Csr>,
    q: Vec<Vec<f64>>,
    mq: Vec<Vec<f64>>,
}

impl Basis<'_> {
    fn apply_m(&self, x: &[f64]) -> Vec<f64> {
        match self.m {
            Some(m) => m.matvec(x),
            None => x.to_vec(),
        }
    }

    /// Orthogonalize `x` against the basis twice and append it normalized. Returns `false` when
    /// nothing survived, so the caller can offer another vector.
    fn push(&mut self, mut x: Vec<f64>) -> bool {
        let before = dot(&x, &self.apply_m(&x)).sqrt();
        if before == 0.0 || !before.is_finite() {
            return false;
        }
        for _ in 0..2 {
            let coeffs: Vec<f64> = self.mq.iter().map(|mqi| dot(mqi, &x)).collect();
            for (c, qi) in coeffs.iter().zip(&self.q) {
                for (xv, qv) in x.iter_mut().zip(qi) {
                    *xv -= c * qv;
                }
            }
        }
        let mx = self.apply_m(&x);
        let after = dot(&x, &mx).sqrt();
        if after <= DEPENDENT_TOL * before {
            return false;
        }
        let inv = 1.0 / after;
        self.q.push(x.iter().map(|v| v * inv).collect());
        self.mq.push(mx.iter().map(|v| v * inv).collect());
        true
    }
}

/// The `nev` eigenpairs of `K x = lambda M x` nearest `sigma` (see the module header).
///
/// # Errors
/// [`EigsError::BadShape`] for mismatched sizes or `nev` outside `[1, n)` (ARPACK's own range);
/// [`EigsError::Factor`] when `K - sigma M` is singular to the factorization.
pub fn eigsh_shift_invert(
    k: &Csr,
    m: Option<&Csr>,
    sigma: f64,
    nev: usize,
) -> Result<Eigenpairs, EigsError> {
    let n = k.nrows();
    if k.ncols() != n {
        return Err(EigsError::BadShape(format!(
            "K must be square, got {n} x {}.",
            k.ncols()
        )));
    }
    if let Some(m) = m {
        if m.nrows() != n || m.ncols() != n {
            return Err(EigsError::BadShape(format!(
                "M must be {n} x {n}, got {} x {}.",
                m.nrows(),
                m.ncols()
            )));
        }
    }
    if nev == 0 || nev >= n {
        return Err(EigsError::BadShape(format!(
            "nev must be in [1, {n}) for a {n} x {n} problem, got {nev}."
        )));
    }

    let shifted = match m {
        Some(m) => k.sub(&m.scaled(sigma)),
        None => k.sub(&Csr::identity(n).scaled(sigma)),
    };
    let lu = SparseLu::factor(&shifted).map_err(EigsError::Factor)?;
    let mut basis = Basis {
        m,
        q: Vec::new(),
        mq: Vec::new(),
    };
    let op = |x: &[f64], basis: &Basis| -> Result<Vec<f64>, EigsError> {
        lu.solve(&basis.apply_m(x)).map_err(EigsError::Factor)
    };
    let mut rng = SplitMix(0x00C0_FFEE_D15E_A5E5);
    let mut w: Vec<Vec<f64>> = Vec::new(); // Op q, column by column
                                           // `(M q_i) . w_j`, row i. Entries depend only on their two vectors, so a grown basis only
                                           // adds a border — recomputing the whole projection each block was most of a run's cost.
    let mut proj: Vec<Vec<f64>> = Vec::new();

    // The start block.
    let b = BLOCK.min(n);
    while basis.q.len() < b {
        let x = rng.vector(n);
        basis.push(x);
    }

    loop {
        // Apply Op to every basis vector that does not have its image yet.
        while w.len() < basis.q.len() {
            let wi = op(&basis.q[w.len()], &basis)?;
            w.push(wi);
        }
        let dim = basis.q.len();

        if dim >= nev + b || dim == n {
            for i in 0..dim {
                if i == proj.len() {
                    proj.push(Vec::with_capacity(dim));
                }
                let have = proj[i].len();
                for wj in &w[have..dim] {
                    let v = dot(&basis.mq[i], wj);
                    proj[i].push(v);
                }
            }
            if let Some(found) = rayleigh_ritz(&basis, &w, &proj, nev, sigma, dim == n)? {
                return Ok(Eigenpairs {
                    basis_size: dim,
                    ..found
                });
            }
        }

        // Grow by one block: the images of the newest block, each orthogonalized in.
        let start = dim.saturating_sub(b);
        let candidates: Vec<Vec<f64>> = w[start..dim].to_vec();
        for x in candidates {
            if basis.q.len() == n {
                break;
            }
            if !basis.push(x) {
                // Already in the space: keep the space growing with fresh directions.
                while basis.q.len() < n && !basis.push(rng.vector(n)) {}
            }
        }
    }
}

/// Project, solve the small problem, and return the wanted pairs if every one has converged (or
/// unconditionally when the basis spans the whole space, where the projection is exact).
fn rayleigh_ritz(
    basis: &Basis,
    w: &[Vec<f64>],
    proj: &[Vec<f64>],
    nev: usize,
    sigma: f64,
    exact: bool,
) -> Result<Option<Eigenpairs>, EigsError> {
    let dim = basis.q.len();
    let n = basis.q[0].len();
    let mut h = vec![0.0; dim * dim];
    for i in 0..dim {
        h[i * dim..i * dim + dim].copy_from_slice(&proj[i][..dim]);
    }
    // Op is M-self-adjoint, so H is symmetric up to rounding; average the two halves.
    for i in 0..dim {
        for j in 0..i {
            let s = 0.5 * (h[i * dim + j] + h[j * dim + i]);
            h[i * dim + j] = s;
            h[j * dim + i] = s;
        }
    }
    let (theta, s) = symmetric_eigen(&h, dim).map_err(EigsError::Projected)?;

    // The wanted end: the largest |theta|, i.e. lambda nearest sigma.
    let mut order: Vec<usize> = (0..dim).collect();
    order.sort_by(|&a, &b| {
        theta[b]
            .abs()
            .partial_cmp(&theta[a].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut pairs: Vec<(f64, Vec<f64>)> = Vec::with_capacity(nev);
    for &j in order.iter().take(nev) {
        let t = theta[j];
        let mut y = vec![0.0; n];
        let mut wy = vec![0.0; n];
        for l in 0..dim {
            let c = s[l * dim + j];
            for i in 0..n {
                y[i] += c * basis.q[l][i];
                wy[i] += c * w[l][i];
            }
        }
        if !exact {
            let r: Vec<f64> = wy.iter().zip(&y).map(|(a, b)| a - t * b).collect();
            let rn = dot(&r, &basis.apply_m(&r)).sqrt();
            // A NaN residual is not converged either.
            if rn > RESIDUAL_TOL * t.abs() || rn.is_nan() {
                return Ok(None);
            }
        }
        pairs.push((sigma + 1.0 / t, y));
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let (values, vectors) = pairs.into_iter().unzip();
    Ok(Some(Eigenpairs {
        values,
        vectors,
        basis_size: dim,
    }))
}
