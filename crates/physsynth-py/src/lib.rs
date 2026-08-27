//! `physsynth_rs` — the PyO3 binding over `physsynth-core`.
//!
//! **Temporary by construction.** `docs/dev/rust-migration-plan.md` §1 deletes this crate when the
//! last model finishes its port; until then it has two live consumers, not one:
//!
//! 1. the existing Python validation suite, which is how a ported model is proved to reproduce the
//!    physics contract (the suite runs *unmodified* — the swap happens inside
//!    `physsynth/core/string_ideal.py`, gated on `PHYSSYNTH_RS`), and
//! 2. the web viewer backend (`web/serialize.py`), which imports the same names and would break
//!    the moment a Python model were deleted (plan §1.1).
//!
//! # The one thing this layer exists to get right: who owns the buffers
//!
//! Python's `step()` **rebinds** `self.u`; it does not overwrite it. A reference taken before a
//! step therefore stays valid and keeps showing that step's values — a snapshot. `connection.py`
//! depends on the other half of the same property, writing *through* `.u` with
//! `self.string.u[-1] -= beta_s * F` and expecting the string to see it.
//!
//! A Rust struct holding `Vec<f64>` state cannot honour both. Handing out a copy loses the write;
//! handing out a zero-copy view over a `Vec` that a later step reallocates is worse than wrong, it
//! is a use-after-free that *reads plausibly* — measured on 2026-08-26, a held view over a
//! reassigned `Vec` still returned the old contents, which is exactly what a correct snapshot
//! looks like right up until the allocator reuses the page.
//!
//! So the buffers here are **NumPy arrays owned by Python**, and this type holds `Py<PyArray1>`
//! handles to them. Lifetime is then refcounted by the interpreter, which is the only mechanism
//! that actually knows who is still holding what. `step()` allocates a fresh array and rebinds —
//! the same allocation pattern the Python original already has, so there is no cost to it — and
//! the physics runs on slices borrowed from those arrays via the kernels in `physsynth-core`.
//! Note that this is also why the binding does **not** wrap `physsynth_core::IdealString`: that
//! struct owns `Vec`s, which is right for a native caller and wrong here.
//!
//! # Private names are part of the surface
//!
//! `_bc_left`, `_bc_right` and `_second_diff` are exposed on purpose. `physsynth/core/connection.py`
//! reaches for the last two, and connection is a Phase 5 model — so for the whole migration a
//! Python module is a client of this binding's *private* names. Measured, not guessed; see the
//! plan §3.1.

#![allow(non_snake_case)] // The Python API spells them `L`, `T`, `N`; the binding must match.

mod banded;
mod body;
mod bore;
mod collision;
mod exciter;
mod mallet;
mod membrane;
mod ops2d;
mod radiation;
mod reed;
mod shape;

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyUntypedArrayMethods};
use physsynth_core::ops;
use physsynth_core::sparse::Csr;
use physsynth_core::string_ideal as core;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyString, PyTuple};

/// Translate a core rejection into the `ValueError` the Python original raises.
///
/// `ParamError::BadBoundary` never reaches here — its message quotes the object the caller passed,
/// which only the caller can `repr()`, so it is formatted at the parse site instead.
fn param_err(e: core::ParamError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Parse the `boundary=` argument into a `(left, right)` pair.
///
/// Mirrors `(boundary, boundary) if isinstance(boundary, str) else boundary`: a bare string sets
/// both ends, a 2-sequence sets them independently. Anything else — a bad spelling, the wrong
/// length, a non-sequence — yields `None`, which `Params::new` turns into the rejection *at the
/// position in the check order where Python raises it*, after the scalar checks and before CFL.
fn parse_boundary(obj: &Bound<'_, PyAny>) -> Option<(core::Boundary, core::Boundary)> {
    if let Ok(s) = obj.cast::<PyString>() {
        let b = core::Boundary::parse(&s.to_cow().ok()?)?;
        return Some((b, b));
    }
    let seq = obj.cast::<PyTuple>().ok()?;
    if seq.len() != 2 {
        return None;
    }
    let left = core::Boundary::parse(
        &seq.get_item(0)
            .ok()?
            .cast::<PyString>()
            .ok()?
            .to_cow()
            .ok()?,
    )?;
    let right = core::Boundary::parse(
        &seq.get_item(1)
            .ok()?
            .cast::<PyString>()
            .ok()?
            .to_cow()
            .ok()?,
    )?;
    Some((left, right))
}

/// `np.asarray(obj, dtype=float)` followed by a 1-D length check, with the Python error text.
///
/// Going through NumPy rather than a direct downcast is what makes a *list* an acceptable `u0`,
/// which the original accepts and some callers use. `want` is `N + 1`.
pub(crate) fn as_1d_f64(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    name: &str,
    want: usize,
) -> PyResult<Vec<f64>> {
    let np = py.import("numpy")?;
    let arr = np.call_method1("asarray", (obj, np.getattr("float64")?))?;
    let arr = np.call_method1("ascontiguousarray", (arr,))?;
    let arr: Bound<'_, PyArray1<f64>> = arr.cast_into().map_err(|_| {
        PyValueError::new_err(format!("{name} must be a 1-D array of shape ({want},)."))
    })?;
    let ro = arr.readonly();
    let got = ro.len();
    if got != want {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape ({want},), got ({got},)."
        )));
    }
    Ok(ro.as_slice()?.to_vec())
}

/// Take the slice out of a stored state array, refusing a non-contiguous one with a clear error.
pub(crate) fn state_slice<'a>(
    ro: &'a PyReadonlyArray1<'_, f64>,
    name: &str,
) -> PyResult<&'a [f64]> {
    ro.as_slice().map_err(|_| {
        PyValueError::new_err(format!("{name} must be a contiguous 1-D float64 array."))
    })
}

/// A discretized ideal string resonator — the Rust implementation, wearing the Python interface.
///
/// Attribute-for-attribute and method-for-method compatible with
/// `physsynth.core.string_ideal.IdealString`; the docstring on that class is the reference.
#[pyclass(name = "IdealString", module = "physsynth_rs")]
pub struct PyIdealString {
    p: core::Params,
    w: Vec<f64>,
    /// The `boundary=` argument exactly as passed, so `.boundary` echoes a str or a tuple the way
    /// the Python original's does.
    boundary: Py<PyAny>,
    x: Py<PyArray1<f64>>,
    u: Py<PyArray1<f64>>,
    u_prev: Py<PyArray1<f64>>,
    n: usize,
}

impl PyIdealString {
    /// Rebind `self.u` to `values`, returning the array object that was there before.
    fn swap_u(&mut self, py: Python<'_>, values: Vec<f64>) -> Py<PyArray1<f64>> {
        let fresh = PyArray1::from_vec(py, values).unbind();
        std::mem::replace(&mut self.u, fresh)
    }

    /// Validate an array being assigned to `.u` or `.u_prev` and take ownership of it.
    ///
    /// Python would accept literally any object here; this accepts any contiguous 1-D float64
    /// array of the right length and rejects the rest loudly. The narrowing is deliberate — a
    /// migration wants a wrong assignment to fail at the assignment, not three models downstream.
    fn adopt_state(&self, value: &Bound<'_, PyAny>, name: &str) -> PyResult<Py<PyArray1<f64>>> {
        let arr: Bound<'_, PyArray1<f64>> = value.clone().cast_into().map_err(|_| {
            PyValueError::new_err(format!("{name} must be a 1-D float64 numpy array."))
        })?;
        let ro = arr.readonly();
        if ro.len() != self.p.nodes() {
            return Err(PyValueError::new_err(format!(
                "{name} must have shape ({},), got ({},).",
                self.p.nodes(),
                ro.len()
            )));
        }
        state_slice(&ro, name)?;
        Ok(arr.unbind())
    }
}

#[pymethods]
impl PyIdealString {
    // Seven keyword arguments plus the GIL token. Clippy's limit is seven, and the shape is not
    // negotiable: this signature IS `IdealString.__init__`, and every call site in `tests/` and
    // `web/serialize.py` spells it out. Bundling them into a struct would be a nicer Rust API and
    // a different Python one.
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (*, L, T, rho, fs, N, boundary=None, sigma=0.0))]
    fn new(
        py: Python<'_>,
        L: f64,
        T: f64,
        rho: f64,
        fs: f64,
        N: i64,
        boundary: Option<Bound<'_, PyAny>>,
        sigma: f64,
    ) -> PyResult<Self> {
        let boundary = match boundary {
            Some(b) => b,
            None => PyString::new(py, "fixed").into_any(),
        };
        let bc = parse_boundary(&boundary);
        let p = core::Params::new(L, T, rho, fs, N, sigma, bc).map_err(|e| match e {
            core::ParamError::BadBoundary => {
                let shown = boundary
                    .repr()
                    .map(|r| r.to_string_lossy().into_owned())
                    .unwrap_or_else(|_| "<unrepresentable>".to_owned());
                PyValueError::new_err(format!(
                    "each boundary end must be 'fixed' or 'free', got {shown}."
                ))
            }
            other => param_err(other),
        })?;

        let w = p.node_weights();
        let x = PyArray1::from_vec(py, p.grid()).unbind();
        let u = PyArray1::from_vec(py, vec![0.0; p.nodes()]).unbind();
        let u_prev = PyArray1::from_vec(py, vec![0.0; p.nodes()]).unbind();
        Ok(PyIdealString {
            p,
            w,
            boundary: boundary.unbind(),
            x,
            u,
            u_prev,
            n: 0,
        })
    }

    // -- parameters ------------------------------------------------------------------------

    #[getter]
    fn L(&self) -> f64 {
        self.p.l
    }
    #[getter]
    fn T(&self) -> f64 {
        self.p.t
    }
    #[getter]
    fn rho(&self) -> f64 {
        self.p.rho
    }
    #[getter]
    fn fs(&self) -> f64 {
        self.p.fs
    }
    #[getter]
    fn N(&self) -> usize {
        self.p.n
    }
    #[getter]
    fn sigma(&self) -> f64 {
        self.p.sigma
    }
    #[getter]
    fn c(&self) -> f64 {
        self.p.c
    }
    #[getter]
    fn h(&self) -> f64 {
        self.p.h
    }
    #[getter]
    fn k(&self) -> f64 {
        self.p.k
    }
    #[getter]
    fn lam(&self) -> f64 {
        self.p.lam
    }
    #[getter]
    fn boundary(&self, py: Python<'_>) -> Py<PyAny> {
        self.boundary.clone_ref(py)
    }
    #[getter]
    fn x(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.x.clone_ref(py)
    }

    /// The per-end conditions, as `connection.py` reads them.
    #[getter]
    fn _bc_left(&self) -> &'static str {
        match self.p.bc_left {
            core::Boundary::Fixed => "fixed",
            core::Boundary::Free => "free",
        }
    }
    #[getter]
    fn _bc_right(&self) -> &'static str {
        match self.p.bc_right {
            core::Boundary::Fixed => "fixed",
            core::Boundary::Free => "free",
        }
    }

    // -- state -----------------------------------------------------------------------------

    /// Current displacement field `u^n` — the live array, writable in place.
    #[getter]
    fn u(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.u.clone_ref(py)
    }
    #[setter]
    fn set_u(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.u = self.adopt_state(value, "u")?;
        Ok(())
    }

    /// Previous displacement field `u^{n-1}` — after a step this *is* the object `.u` was.
    #[getter]
    fn u_prev(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.u_prev.clone_ref(py)
    }
    #[setter]
    fn set_u_prev(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.u_prev = self.adopt_state(value, "u_prev")?;
        Ok(())
    }

    /// Number of completed steps.
    #[getter]
    fn n(&self) -> usize {
        self.n
    }
    #[setter]
    fn set_n(&mut self, value: usize) {
        self.n = value;
    }

    /// Current displacement field (a copy, safe to mutate/store for plotting).
    #[getter]
    fn state(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f64>>> {
        let bound = self.u.bind(py);
        let ro = bound.readonly();
        Ok(PyArray1::from_slice(py, state_slice(&ro, "u")?).unbind())
    }

    // -- initial conditions ----------------------------------------------------------------

    /// Set the initial displacement (and optional velocity).
    ///
    /// Uses the consistent second-order start `u^{-1} = u^0 - k v^0 + 1/2 * stencil(u^0)`.
    #[pyo3(signature = (u0, v0=None))]
    fn set_state(
        &mut self,
        py: Python<'_>,
        u0: &Bound<'_, PyAny>,
        v0: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let nodes = self.p.nodes();
        let mut u = as_1d_f64(py, u0, "u0", nodes)?;

        // `np.broadcast_to(np.asarray(v0, float), (N+1,))`: a scalar (the default) fills, and a
        // full-length array is taken as is. Anything in between is the caller's mistake.
        let v = match v0 {
            None => vec![0.0; nodes],
            Some(obj) => match obj.extract::<f64>() {
                Ok(scalar) => vec![scalar; nodes],
                Err(_) => as_1d_f64(py, obj, "v0", nodes)?,
            },
        };

        core::apply_boundary(&mut u, self.p.bc_left, self.p.bc_right);
        let prev = core::initial_previous(&u, &v, &self.p);
        self.u_prev = PyArray1::from_vec(py, prev).unbind();
        self.swap_u(py, u);
        self.n = 0;
        Ok(())
    }

    // -- time stepping ---------------------------------------------------------------------

    /// Advance one timestep, rolling the history.
    fn step(&mut self, py: Python<'_>) -> PyResult<()> {
        let nodes = self.p.nodes();
        let mut next = vec![0.0; nodes];
        {
            let u_bound = self.u.bind(py);
            let up_bound = self.u_prev.bind(py);
            let u_ro = u_bound.readonly();
            let up_ro = up_bound.readonly();
            core::step_into(
                state_slice(&u_ro, "u")?,
                state_slice(&up_ro, "u_prev")?,
                &mut next,
                &self.p,
            );
        }
        self.u_prev = self.swap_u(py, next);
        self.n += 1;
        Ok(())
    }

    // -- diagnostics -----------------------------------------------------------------------

    /// Discrete energy `E^n` (Joules) using the cross-time potential term.
    fn energy(&self, py: Python<'_>) -> PyResult<f64> {
        let u_bound = self.u.bind(py);
        let up_bound = self.u_prev.bind(py);
        let u_ro = u_bound.readonly();
        let up_ro = up_bound.readonly();
        Ok(core::energy(
            state_slice(&u_ro, "u")?,
            state_slice(&up_ro, "u_prev")?,
            &self.w,
            &self.p,
        ))
    }

    /// Displacement at grid node `index` — a pickup for spectral analysis.
    ///
    /// Negative indices count from the end, as they do on the NumPy array this replaces;
    /// `displacement_at(-1)` is the terminus a body connection loads.
    fn displacement_at(&self, py: Python<'_>, index: i64) -> PyResult<f64> {
        let nodes = self.p.nodes() as i64;
        let idx = if index < 0 { index + nodes } else { index };
        if idx < 0 || idx >= nodes {
            return Err(PyIndexError::new_err(format!(
                "index {index} is out of bounds for a grid of {nodes} nodes"
            )));
        }
        let bound = self.u.bind(py);
        let ro = bound.readonly();
        Ok(state_slice(&ro, "u")?[idx as usize])
    }

    // -- internals, exposed because `connection.py` uses them ------------------------------

    /// `u[l+1] - 2u[l] + u[l-1]` over the whole grid, with this string's boundary stencil.
    fn _second_diff(
        &self,
        py: Python<'_>,
        u: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let s = state_slice(&u, "u")?;
        if s.len() != self.p.nodes() {
            return Err(PyValueError::new_err(format!(
                "u must have shape ({},), got ({},).",
                self.p.nodes(),
                s.len()
            )));
        }
        let out = core::second_diff(s, self.p.bc_left, self.p.bc_right);
        Ok(PyArray1::from_vec(py, out).unbind())
    }

    /// Clamp this string's fixed ends on `u`, in place.
    fn _apply_boundary(&self, u: &Bound<'_, PyArray1<f64>>) -> PyResult<()> {
        let mut rw = u.readwrite();
        let s = rw
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("u must be a contiguous 1-D float64 array."))?;
        if s.len() != self.p.nodes() {
            return Err(PyValueError::new_err(format!(
                "u must have shape ({},), got ({},).",
                self.p.nodes(),
                s.len()
            )));
        }
        core::apply_boundary(s, self.p.bc_left, self.p.bc_right);
        Ok(())
    }
}

// ==== operators (plan Phase 1) ===================================================================
//
// `physsynth/core/operators.py` is a module of free functions, not a class, so the binding is a
// set of free functions too. Two shapes come back, and the difference is the whole design:
//
//   * the four pointwise differences return a fresh NumPy array, exactly as the original does;
//   * the three matrix builders return **CSR triplets** — `(data, indices, indptr, shape)` — and
//     *not* a matrix.
//
// The second is deliberate (plan §3.1). `physsynth-core` must not learn what SciPy is, and this
// crate must not decide what a sparse matrix means to the caller. So the Rust side hands back the
// four arrays and the shim at the bottom of `physsynth/core/operators.py` rebuilds a
// `scipy.sparse.csr_matrix` from them. That keeps the swap invisible to `string_stiff.py`,
// `string_damped.py`, `string_nonlinear.py`, `string_geometric.py` and `beam.py`, all five of
// which do `from .operators import ...` and expect a SciPy object back — the "switch is the lever,
// zero client edits" property Phase 0 established, now carrying five models instead of one.
//
// Index arrays come back as **int32**, which is what SciPy picks for every matrix this project
// builds (measured at N = 4, 64 and 1000). Handing back int64 would work and would silently change
// `.indices.dtype`, so the width is asserted rather than left to the rebuild.

/// Borrow a contiguous 1-D float64 slice for an operator argument.
fn op_slice<'a>(ro: &'a PyReadonlyArray1<'_, f64>, name: &str) -> PyResult<&'a [f64]> {
    state_slice(ro, name)
}

/// Validate the interval count shared by the three builders.
///
/// The Python original rejects `N < 2` in two different voices: `free_beam_stiffness` raises its
/// own message, while the other two fall through to NumPy's `negative dimensions are not allowed`.
/// Both are `ValueError`, which is the part callers can depend on; the text here is the clearer of
/// the two in all three cases. The divergence is noted in the swap block that installs these.
fn n_intervals(n: i64) -> PyResult<usize> {
    if n < 2 {
        return Err(PyValueError::new_err(format!(
            "N must be >= 2 (need at least one interior node); got {n}."
        )));
    }
    Ok(n as usize)
}

/// A CSR matrix as `(data, indices, indptr, (nrows, ncols))`, ready for `scipy.sparse.csr_matrix`.
pub(crate) type CsrTriplets = (
    Py<PyArray1<f64>>,
    Py<PyArray1<i32>>,
    Py<PyArray1<i32>>,
    (usize, usize),
);

pub(crate) fn csr_triplets(py: Python<'_>, m: &Csr) -> PyResult<CsrTriplets> {
    // SciPy switches its index width above 2^31 and so would this, but nothing in this project is
    // within three orders of magnitude of that. Refuse rather than wrap: `as i32` is a silent
    // truncation, and a truncated index array builds a *plausible* wrong matrix.
    let limit = i32::MAX as usize;
    if m.nnz() > limit || m.nrows() > limit || m.ncols() > limit {
        return Err(PyValueError::new_err(
            "matrix is too large for 32-bit CSR indices; SciPy would widen to int64 here and this \
             binding has never needed to",
        ));
    }
    Ok((
        PyArray1::from_slice(py, m.data()).unbind(),
        PyArray1::from_vec(py, m.indices().iter().map(|&j| j as i32).collect()).unbind(),
        PyArray1::from_vec(py, m.indptr().iter().map(|&j| j as i32).collect()).unbind(),
        (m.nrows(), m.ncols()),
    ))
}

/// Forward spatial difference `(u[l+1] - u[l]) / h`; `len(u) - 1` inter-node strains.
#[pyfunction]
#[pyo3(name = "delta_x_forward")]
fn op_delta_x_forward(
    py: Python<'_>,
    u: PyReadonlyArray1<'_, f64>,
    h: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let s = op_slice(&u, "u")?;
    // NumPy's `u[1:] - u[:-1]` yields an empty array rather than raising when `u` is too short.
    // The core kernel documents a precondition and panics instead, which is right for a Rust
    // caller and wrong at an interpreter boundary — a panic here would surface as PanicException.
    let out = if s.len() < 2 {
        Vec::new()
    } else {
        ops::delta_x_forward(s, h)
    };
    Ok(PyArray1::from_vec(py, out).unbind())
}

/// Backward spatial difference — the same numbers as `delta_x_forward`, for notational symmetry.
#[pyfunction]
#[pyo3(name = "delta_x_backward")]
fn op_delta_x_backward(
    py: Python<'_>,
    u: PyReadonlyArray1<'_, f64>,
    h: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    op_delta_x_forward(py, u, h)
}

/// Second spatial difference at the `len(u) - 2` interior nodes.
#[pyfunction]
#[pyo3(name = "delta_xx")]
fn op_delta_xx(
    py: Python<'_>,
    u: PyReadonlyArray1<'_, f64>,
    h: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let s = op_slice(&u, "u")?;
    let out = if s.len() < 3 {
        Vec::new()
    } else {
        ops::delta_xx(s, h)
    };
    Ok(PyArray1::from_vec(py, out).unbind())
}

/// Fourth spatial difference at the `len(u) - 4` nodes where the 5-point stencil fits.
#[pyfunction]
#[pyo3(name = "delta_xxxx")]
fn op_delta_xxxx(
    py: Python<'_>,
    u: PyReadonlyArray1<'_, f64>,
    h: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    let s = op_slice(&u, "u")?;
    let out = if s.len() < 5 {
        Vec::new()
    } else {
        ops::delta_xxxx(s, h)
    };
    Ok(PyArray1::from_vec(py, out).unbind())
}

/// Discrete inner product `<f, g> = h * sum_l f[l] g[l]`.
#[pyfunction]
#[pyo3(name = "inner")]
fn op_inner(f: PyReadonlyArray1<'_, f64>, g: PyReadonlyArray1<'_, f64>, h: f64) -> PyResult<f64> {
    let a = op_slice(&f, "f")?;
    let b = op_slice(&g, "g")?;
    if a.len() != b.len() {
        return Err(PyValueError::new_err(format!(
            "inner() operands must have equal length; got ({},) and ({},).",
            a.len(),
            b.len()
        )));
    }
    Ok(ops::inner(a, b, h))
}

/// Squared discrete norm `||f||^2 = <f, f>`.
#[pyfunction]
#[pyo3(name = "norm2")]
fn op_norm2(f: PyReadonlyArray1<'_, f64>, h: f64) -> PyResult<f64> {
    Ok(ops::norm2(op_slice(&f, "f")?, h))
}

/// `(N-1) x (N-1)` Dirichlet second-difference operator, as CSR triplets.
#[pyfunction]
#[pyo3(name = "second_difference_matrix_csr")]
fn op_second_difference_matrix(py: Python<'_>, N: i64, h: f64) -> PyResult<CsrTriplets> {
    csr_triplets(py, &ops::second_difference_matrix(n_intervals(N)?, h))
}

/// `(N-1) x (N-1)` simply-supported biharmonic operator `D2 @ D2`, as CSR triplets.
#[pyfunction]
#[pyo3(name = "biharmonic_matrix_csr")]
fn op_biharmonic_matrix(py: Python<'_>, N: i64, h: f64) -> PyResult<CsrTriplets> {
    csr_triplets(py, &ops::biharmonic_matrix(n_intervals(N)?, h))
}

/// Free-free Euler–Bernoulli `(K, W)` on the `N+1` nodes, as a pair of CSR triplets.
#[pyfunction]
#[pyo3(name = "free_beam_stiffness_csr")]
fn op_free_beam_stiffness(py: Python<'_>, N: i64, h: f64) -> PyResult<(CsrTriplets, CsrTriplets)> {
    let (k, w) = ops::free_beam_stiffness(n_intervals(N)?, h);
    Ok((csr_triplets(py, &k)?, csr_triplets(py, &w)?))
}

/// The extension module. Two models, the operators, the 2-D builders and the excitations today;
/// later phases add in place.
#[pymodule]
fn physsynth_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIdealString>()?;
    m.add_class::<membrane::PyMembrane>()?;
    m.add_class::<body::PyModalBody>()?;
    m.add_class::<bore::PyBore>()?;
    m.add_class::<reed::PyReedBore>()?;
    m.add_class::<mallet::PyMalletMembrane>()?;
    m.add_class::<mallet::PyMalletWall>()?;
    m.add_class::<radiation::PyAirRadiation>()?;
    m.add_class::<radiation::PyRadiatedBody>()?;
    m.add_class::<radiation::PyRationalAirLoad>()?;
    m.add_class::<radiation::PyReactiveRadiatedBody>()?;
    m.add_function(wrap_pyfunction!(
        radiation::py_monopole_radiation_resistance,
        m
    )?)?;
    m.add("RHO0_AIR", physsynth_core::radiation::RHO0_AIR)?;
    m.add("C0_AIR", physsynth_core::radiation::C0_AIR)?;
    m.add_function(wrap_pyfunction!(reed::py_bernoulli_flow, m)?)?;
    m.add_function(wrap_pyfunction!(collision::py_contact_potential, m)?)?;
    m.add_function(wrap_pyfunction!(collision::py_contact_force_elastic, m)?)?;
    m.add_function(wrap_pyfunction!(collision::py_contact_stiffness, m)?)?;
    m.add_function(wrap_pyfunction!(collision::py_contact_force_dg, m)?)?;
    m.add_function(wrap_pyfunction!(collision::py_contact_force_total, m)?)?;
    m.add_function(wrap_pyfunction!(
        collision::py_contact_force_total_deriv,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(collision::py_force_total_vec, m)?)?;
    m.add_function(wrap_pyfunction!(collision::py_deriv_total_vec, m)?)?;
    m.add_function(wrap_pyfunction!(collision::py_solve_contact, m)?)?;
    m.add_function(wrap_pyfunction!(collision::py_solve_contact_vector, m)?)?;
    m.add_function(wrap_pyfunction!(collision::py_lu_factor, m)?)?;
    m.add_function(wrap_pyfunction!(collision::py_lu_solve, m)?)?;
    m.add_function(wrap_pyfunction!(banded::py_cholesky_banded_upper, m)?)?;
    m.add_function(wrap_pyfunction!(banded::py_cho_solve_banded_upper, m)?)?;
    m.add(
        "NotPositiveDefinite",
        m.py().get_type::<banded::NotPositiveDefinite>(),
    )?;
    m.add_function(wrap_pyfunction!(op_delta_x_forward, m)?)?;
    m.add_function(wrap_pyfunction!(op_delta_x_backward, m)?)?;
    m.add_function(wrap_pyfunction!(op_delta_xx, m)?)?;
    m.add_function(wrap_pyfunction!(op_delta_xxxx, m)?)?;
    m.add_function(wrap_pyfunction!(op_inner, m)?)?;
    m.add_function(wrap_pyfunction!(op_norm2, m)?)?;
    m.add_function(wrap_pyfunction!(op_second_difference_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(op_biharmonic_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(op_free_beam_stiffness, m)?)?;
    m.add_function(wrap_pyfunction!(ops2d::py_grid_coords, m)?)?;
    m.add_function(wrap_pyfunction!(ops2d::py_rectangle_mask, m)?)?;
    m.add_function(wrap_pyfunction!(ops2d::py_disk_mask, m)?)?;
    m.add_function(wrap_pyfunction!(ops2d::py_laplacian_from_mask, m)?)?;
    m.add_function(wrap_pyfunction!(ops2d::py_embed, m)?)?;
    m.add_function(wrap_pyfunction!(ops2d::py_inner2d, m)?)?;
    m.add_function(wrap_pyfunction!(ops2d::py_norm2_2d, m)?)?;
    m.add_function(wrap_pyfunction!(exciter::py_triangular_pluck, m)?)?;
    m.add_function(wrap_pyfunction!(exciter::py_raised_cosine, m)?)?;
    m.add_function(wrap_pyfunction!(exciter::py_raised_cosine_2d, m)?)?;
    Ok(())
}
