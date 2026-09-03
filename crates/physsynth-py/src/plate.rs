//! The binding over `physsynth_core::plate` — models #5 and #6 wearing the Python interface.
//!
//! Buffer ownership follows §9.3: `u`, `u_prev`, `_accel`, `F` and `F_prev` are **Python-owned**
//! `PyArray1`s, rebound rather than overwritten by a step, and settable.
//!
//! # Five of those setters are not a convenience
//!
//! `airbox.py`'s `_PlateSurface` and `_VKPlateSurface` extract one step of each model so the room
//! can put its load *inside* the solve. They read `u`, `u_prev`, `B`/`K`/`W`, `theta`, `kappa`,
//! `sigma`, `h`, `rho`/`rho_s`, `X`, `Y`, `mask` — and they **write** `u`, `u_prev`, `n`, `_accel`,
//! `F`, `F_prev`, `n_iters`, `converged` and `last_residual`. Four of those names begin with an
//! underscore or read like diagnostics, and §12.2 is the standing warning: *a leading underscore
//! is not a statement about the interface.* The same seam calls `_linear_rhs`, `_to_full`,
//! `_to_live` and `_airy_F` as methods, so those are `pymethods` here under their private names.
//!
//! `connection.py`, by contrast, touches nothing private at all — it reads `n_live`, `u`,
//! `u_prev`, `converged`, `n_iters`, `last_residual` and calls `step(f_ext=...)`, `energy()` and
//! `pressure()`. §0's prediction that it would be a Phase 5 client of a model's private names came
//! true for the string and **not** for the plate.
//!
//! # Where this binding is not the model, exactly
//!
//! Two argument shapes differ, in opposite directions, and both are recorded rather than hidden.
//! `v0` as a wrongly-sized live vector is **refused** here where the original lets it through to
//! fail later inside NumPy — the alternative would be a panic rather than a raise. And `f_ext` as
//! a *list* is **accepted** here where the original raises `TypeError` from a NumPy operator,
//! because it goes through `as_1d_f64`, the helper every model's binding uses and whose whole
//! reason for existing is that a list is an acceptable array. Neither is a designed refusal on
//! either side, and no test depends on one.
//!
//! # `B`, `K`, `W` and `L` are built once
//!
//! They are the documented interface, not a parity convenience: `tests/helpers.py` hands `K` and
//! `W` straight to a generalized `eigsh` for the free plate's modal oracle, `test_plate_modal.py`
//! compares `B` against the shared builder, and `airbox`'s `a_bare()` reassembles the system matrix
//! from them. A getter that rebuilt them per access would refactor on every read.

use crate::ops2d::{PyAiryStressSolver, PyVonKarmanBracket};
use crate::shape::{as_f64_field, shape_repr, to_2d_bool, to_2d_f64, to_2d_i64};
use crate::sparse_lu::PySparseLu;
use crate::state_slice;
use crate::string_stiff::{csr_object, node_value};
use numpy::{PyArray1, PyArrayMethods, PyUntypedArrayMethods};
use physsynth_core::plate as core;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyString;

/// Translate a core rejection into the `ValueError` the Python original raises.
fn param_err(e: core::ParamError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// `repr(obj)`, for the two refusals that quote the argument they were handed.
fn shown(obj: &Bound<'_, PyAny>) -> String {
    obj.repr()
        .map(|r| r.to_string_lossy().into_owned())
        .unwrap_or_else(|_| "<unrepresentable>".to_owned())
}

/// Resolve an omitted-or-explicit `boundary` argument into the object and the parsed value.
///
/// `Option<Option<_>>` so that an omitted argument and an explicit `None` stay distinguishable —
/// §24.7, where collapsing them made `boundary=None` silently build the default. PyO3 wraps the
/// *default* expression, so `Some(None)` is "omitted" and a bare `None` is the caller's literal.
fn resolve_boundary<'py>(
    py: Python<'py>,
    arg: Option<Option<Py<PyAny>>>,
    default: &str,
) -> (Bound<'py, PyAny>, Option<core::Boundary>) {
    let obj = match arg {
        Some(None) => PyString::new(py, default).into_any(),
        None => py.None().into_bound(py),
        Some(Some(b)) => b.into_bound(py),
    };
    let parsed = obj
        .cast::<PyString>()
        .ok()
        .and_then(|s| s.to_cow().ok())
        .and_then(|c| match &*c {
            "supported" => Some(core::Boundary::Supported),
            "free" => Some(core::Boundary::Free),
            _ => None,
        });
    (obj, parsed)
}

/// The same for `domain`, whose three spellings the original also quotes back.
fn resolve_domain<'py>(
    py: Python<'py>,
    arg: Option<Option<Py<PyAny>>>,
) -> (Bound<'py, PyAny>, Option<core::Domain>) {
    let obj = match arg {
        Some(None) => PyString::new(py, "rectangle").into_any(),
        None => py.None().into_bound(py),
        Some(Some(d)) => d.into_bound(py),
    };
    let parsed = obj
        .cast::<PyString>()
        .ok()
        .and_then(|s| s.to_cow().ok())
        .and_then(|c| match &*c {
            "rectangle" => Some(core::Domain::Rectangle),
            "circle" => Some(core::Domain::Circle),
            "guitar" => Some(core::Domain::Guitar),
            _ => None,
        });
    (obj, parsed)
}

/// Read a SciPy sparse matrix into a core `Csr` **keeping its stored row order**.
///
/// The mirror of `csr_object`, and the direction that only exists because of one test. Every
/// matrix this binding hands *out* was assembled in canonical order; a matrix handed *in* through
/// `Plate.B`'s setter may deliberately not be, because the order is what that test is about
/// (`Csr::from_arrays_preserving_order` carries the argument at length).
///
/// `indptr`/`indices` are read as `int64` whatever width SciPy chose — it picks `int32` or
/// `int64` by the matrix's size, and both are the same integers.
fn csr_from_scipy(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    name: &str,
    n: usize,
) -> PyResult<physsynth_core::sparse::Csr> {
    let np = py.import("numpy")?;
    let shape = obj
        .getattr("shape")
        .and_then(|s| s.extract::<(usize, usize)>())
        .map_err(|_| {
            PyValueError::new_err(format!(
                "{name} must be a sparse matrix of shape ({n}, {n}); got {}.",
                shown(obj)
            ))
        })?;
    if shape != (n, n) {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape ({n}, {n}), got {:?}.",
            shape
        )));
    }
    let ints = |attr: &str| -> PyResult<Vec<usize>> {
        let a = obj.getattr(attr).map_err(|_| {
            PyValueError::new_err(format!(
                "{name} must be in CSR format (no `{attr}`); call `.tocsr()` on it first."
            ))
        })?;
        let a = np.call_method1("asarray", (a, np.getattr("int64")?))?;
        let a = np.call_method1("ascontiguousarray", (a,))?;
        let a: Bound<'_, numpy::PyArray1<i64>> = a.cast_into().map_err(|_| {
            PyValueError::new_err(format!("{name}.{attr} must be a 1-D integer array."))
        })?;
        let ro = a.readonly();
        let s = ro
            .as_slice()
            .map_err(|_| PyValueError::new_err(format!("{name}.{attr} must be contiguous.")))?;
        if let Some(&bad) = s.iter().find(|&&v| v < 0) {
            return Err(PyValueError::new_err(format!(
                "{name}.{attr} holds a negative entry ({bad})."
            )));
        }
        Ok(s.iter().map(|&v| v as usize).collect())
    };
    let indptr = ints("indptr")?;
    let indices = ints("indices")?;
    let data = {
        let a = obj.getattr("data")?;
        let a = np.call_method1("asarray", (a, np.getattr("float64")?))?;
        let a = np.call_method1("ascontiguousarray", (a,))?;
        let a: Bound<'_, PyArray1<f64>> = a.cast_into().map_err(|_| {
            PyValueError::new_err(format!("{name}.data must be a 1-D float array."))
        })?;
        let ro = a.readonly();
        ro.as_slice()
            .map_err(|_| PyValueError::new_err(format!("{name}.data must be contiguous.")))?
            .to_vec()
    };
    physsynth_core::sparse::Csr::from_arrays_preserving_order(n, n, indptr, indices, data)
        .map_err(|e| PyValueError::new_err(format!("{name} is not a well-formed CSR matrix: {e}")))
}

/// `u0` as a live vector, accepting either the full 2-D field or the live vector itself.
fn live_arg(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    name: &str,
    mask: &[bool],
    nrows: usize,
    ncols: usize,
    n_live: usize,
) -> PyResult<Vec<f64>> {
    let (shape, values) = as_f64_field(py, obj, name)?;
    if shape == [nrows, ncols] {
        return Ok(values
            .iter()
            .zip(mask.iter())
            .filter(|(_, &alive)| alive)
            .map(|(&v, _)| v)
            .collect());
    }
    if shape == [n_live] {
        return Ok(values);
    }
    Err(PyValueError::new_err(format!(
        "{name} must have shape {} (full field) or {} (live), got {}.",
        shape_repr(&[nrows, ncols]),
        shape_repr(&[n_live]),
        shape_repr(&shape)
    )))
}

/// `v0` as a live vector: a scalar broadcasts, a full field is masked, a live vector passes.
fn velocity_arg(
    py: Python<'_>,
    obj: Option<&Bound<'_, PyAny>>,
    mask: &[bool],
    nrows: usize,
    ncols: usize,
    n_live: usize,
) -> PyResult<Vec<f64>> {
    let Some(obj) = obj else {
        return Ok(vec![0.0; n_live]);
    };
    let (shape, values) = as_f64_field(py, obj, "v0")?;
    if shape.is_empty() {
        return Ok(vec![values[0]; n_live]);
    }
    if shape == [nrows, ncols] {
        return Ok(values
            .iter()
            .zip(mask.iter())
            .filter(|(_, &alive)| alive)
            .map(|(&v, _)| v)
            .collect());
    }
    // The original does not length-check this branch -- it lets a wrong-length vector through to
    // fail later, in NumPy, on a broadcast. Refusing here instead is the one place this binding is
    // deliberately stricter than the model, because the alternative is a panic rather than a raise.
    if shape == [n_live] {
        return Ok(values);
    }
    Err(PyValueError::new_err(format!(
        "v0 must be a scalar, a full field of shape {}, or a live vector of shape {}, got {}.",
        shape_repr(&[nrows, ncols]),
        shape_repr(&[n_live]),
        shape_repr(&shape)
    )))
}

/// A discretized Kirchhoff plate — the Rust implementation, wearing the Python interface.
///
/// Attribute-for-attribute and method-for-method compatible with `physsynth.core.plate.Plate`;
/// the docstring on that class is the reference.
#[pyclass(name = "Plate", module = "physsynth_rs")]
pub struct PyPlate {
    p: core::Params,
    boundary: Py<PyAny>,
    domain: Py<PyAny>,
    grid_x: Py<PyAny>,
    grid_y: Py<PyAny>,
    mask: Py<PyAny>,
    index_map: Py<PyAny>,
    laplacian: Option<Py<PyAny>>,
    stiffness: Py<PyAny>,
    mass: Option<Py<PyAny>>,
    w: Option<Py<PyArray1<f64>>>,
    u: Py<PyArray1<f64>>,
    u_prev: Py<PyArray1<f64>>,
    accel: Py<PyArray1<f64>>,
    n: usize,
}

impl PyPlate {
    /// Validate an array being assigned to a state attribute and take ownership of it.
    fn adopt(&self, value: &Bound<'_, PyAny>, name: &str) -> PyResult<Py<PyArray1<f64>>> {
        let arr: Bound<'_, PyArray1<f64>> = value.clone().cast_into().map_err(|_| {
            PyValueError::new_err(format!("{name} must be a 1-D float64 numpy array."))
        })?;
        let ro = arr.readonly();
        if ro.len() != self.p.n_live {
            return Err(PyValueError::new_err(format!(
                "{name} must have shape ({},), got ({},).",
                self.p.n_live,
                ro.len()
            )));
        }
        state_slice(&ro, name)?;
        Ok(arr.unbind())
    }

    /// The two state buffers as slices, for a kernel that reads both.
    fn with_state<T>(&self, py: Python<'_>, f: impl FnOnce(&[f64], &[f64]) -> T) -> PyResult<T> {
        let ub = self.u.bind(py);
        let pb = self.u_prev.bind(py);
        let uro = ub.readonly();
        let pro = pb.readonly();
        Ok(f(state_slice(&uro, "u")?, state_slice(&pro, "u_prev")?))
    }
}

#[pymethods]
impl PyPlate {
    // Twenty keyword arguments plus the GIL token: this signature IS `Plate.__init__`.
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (*, Lx, Ly, kappa, rho, fs, N, sigma=0.0, theta=core::THETA_DEFAULT,
                        boundary=None::<Py<PyAny>>, domain=None::<Py<PyAny>>, waist=0.42,
                        asym=0.30, nu=None, grain_x=1.0, grain_cross=None, grain_y=1.0,
                        grain_coupling=None, grain_torsion=None))]
    fn new(
        py: Python<'_>,
        Lx: f64,
        Ly: f64,
        kappa: f64,
        rho: f64,
        fs: f64,
        N: i64,
        sigma: f64,
        theta: f64,
        boundary: Option<Option<Py<PyAny>>>,
        domain: Option<Option<Py<PyAny>>>,
        waist: f64,
        asym: f64,
        nu: Option<f64>,
        grain_x: f64,
        grain_cross: Option<f64>,
        grain_y: f64,
        grain_coupling: Option<f64>,
        grain_torsion: Option<f64>,
    ) -> PyResult<Self> {
        let (boundary_obj, boundary) = resolve_boundary(py, boundary, "supported");
        let (domain_obj, domain) = resolve_domain(py, domain);
        let spec = core::PlateSpec {
            lx: Lx,
            ly: Ly,
            kappa,
            rho,
            fs,
            n: N,
            sigma,
            theta,
            boundary,
            domain,
            waist,
            asym,
            nu,
            grain_x,
            grain_cross,
            grain_y,
            grain_coupling,
            grain_torsion,
        };
        let p = core::Params::new(&spec).map_err(|e| match e {
            core::ParamError::BadBoundary => PyValueError::new_err(format!(
                "boundary must be 'supported' or 'free', got {}.",
                shown(&boundary_obj)
            )),
            core::ParamError::BadDomain => PyValueError::new_err(format!(
                "domain must be 'rectangle', 'circle' or 'guitar'; got {}.",
                shown(&domain_obj)
            )),
            other => param_err(other),
        })?;

        let (nrows, ncols) = (p.mask.nrows(), p.mask.ncols());
        let grid_x = to_2d_f64(py, p.x.clone(), nrows, ncols)?;
        let grid_y = to_2d_f64(py, p.y.clone(), nrows, ncols)?;
        let mask = to_2d_bool(py, p.mask.flags().to_vec(), nrows, ncols)?;
        let index_map = to_2d_i64(py, p.index_map.clone(), nrows, ncols)?;
        let laplacian = match &p.laplacian {
            Some(l) => Some(csr_object(py, l)?),
            None => None,
        };
        let stiffness = csr_object(py, &p.stiffness)?;
        let mass = match &p.mass {
            Some(m) => Some(csr_object(py, m)?),
            None => None,
        };
        let w = if p.w.is_empty() {
            None
        } else {
            Some(PyArray1::from_slice(py, &p.w).unbind())
        };
        let n_live = p.n_live;
        Ok(PyPlate {
            p,
            boundary: boundary_obj.unbind(),
            domain: domain_obj.unbind(),
            grid_x,
            grid_y,
            mask,
            index_map,
            laplacian,
            stiffness,
            mass,
            w,
            u: PyArray1::from_vec(py, vec![0.0; n_live]).unbind(),
            u_prev: PyArray1::from_vec(py, vec![0.0; n_live]).unbind(),
            accel: PyArray1::from_vec(py, vec![0.0; n_live]).unbind(),
            n: 0,
        })
    }

    // -- parameters ------------------------------------------------------------------------

    #[getter]
    fn Lx(&self) -> f64 {
        self.p.lx
    }
    #[getter]
    fn Ly(&self) -> f64 {
        self.p.ly
    }
    #[getter]
    fn kappa(&self) -> f64 {
        self.p.kappa
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
    fn theta(&self) -> f64 {
        self.p.theta
    }
    #[getter]
    fn nu(&self) -> f64 {
        self.p.nu
    }
    #[getter]
    fn waist(&self) -> f64 {
        self.p.waist
    }
    #[getter]
    fn asym(&self) -> f64 {
        self.p.asym
    }
    #[getter]
    fn grain_x(&self) -> f64 {
        self.p.grain_x
    }
    #[getter]
    fn grain_cross(&self) -> f64 {
        self.p.grain_cross
    }
    #[getter]
    fn grain_y(&self) -> f64 {
        self.p.grain_y
    }
    #[getter]
    fn grain_coupling(&self) -> f64 {
        self.p.grain_coupling
    }
    #[getter]
    fn grain_torsion(&self) -> f64 {
        self.p.grain_torsion
    }
    #[getter]
    fn grain_is_isotropic(&self) -> bool {
        self.p.grain_is_isotropic
    }
    #[getter]
    fn k(&self) -> f64 {
        self.p.k
    }
    #[getter]
    fn h(&self) -> f64 {
        self.p.h
    }
    #[getter]
    fn mu(&self) -> f64 {
        self.p.mu
    }
    #[getter]
    fn n_live(&self) -> usize {
        self.p.n_live
    }
    #[getter]
    fn n_pruned(&self) -> usize {
        self.p.n_pruned
    }
    #[getter]
    fn prune_depth_max(&self) -> f64 {
        self.p.prune_depth_max
    }
    #[getter]
    fn outline_area(&self) -> f64 {
        self.p.outline_area
    }
    #[getter]
    fn area(&self) -> f64 {
        self.p.area
    }
    #[getter]
    fn area_deficit(&self) -> f64 {
        self.p.area_deficit
    }
    #[getter]
    fn boundary(&self, py: Python<'_>) -> Py<PyAny> {
        self.boundary.clone_ref(py)
    }
    #[getter]
    fn domain(&self, py: Python<'_>) -> Py<PyAny> {
        self.domain.clone_ref(py)
    }
    #[getter]
    fn X(&self, py: Python<'_>) -> Py<PyAny> {
        self.grid_x.clone_ref(py)
    }
    #[getter]
    fn Y(&self, py: Python<'_>) -> Py<PyAny> {
        self.grid_y.clone_ref(py)
    }
    #[getter]
    fn mask(&self, py: Python<'_>) -> Py<PyAny> {
        self.mask.clone_ref(py)
    }
    #[getter]
    fn index_map(&self, py: Python<'_>) -> Py<PyAny> {
        self.index_map.clone_ref(py)
    }

    /// The masked Dirichlet Laplacian — **supported branch only**, as in the original, where a
    /// free plate has no `L` attribute at all.
    #[getter]
    fn L(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.laplacian {
            Some(l) => Ok(l.clone_ref(py)),
            None => Err(pyo3::exceptions::PyAttributeError::new_err(
                "'Plate' object has no attribute 'L'",
            )),
        }
    }

    /// The biharmonic `B` — supported branch only.
    #[getter]
    fn B(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if self.p.boundary != core::Boundary::Supported {
            return Err(pyo3::exceptions::PyAttributeError::new_err(
                "'Plate' object has no attribute 'B'",
            ));
        }
        Ok(self.stiffness.clone_ref(py))
    }

    /// Replace the biharmonic the step applies — and **only** that.
    ///
    /// The one door in this binding that lets a caller put an operator of their own into a model,
    /// and it is here for one test. `tests/test_plate_modal.py` holds the pin on the
    /// 2026-08-28 sparse-assembly finding: it steps one plate on the shipped, canonically sorted
    /// `B` and another on the pre-fix assembly — the same numbers in the order SciPy's sparse
    /// product emits them — and asserts the trajectory did not move and neither drifts. That is a
    /// claim about a *summation order*, so it needs the other order in a plate, and a read-only
    /// getter made it inexpressible once the Python plate was deleted (plan §40.5, the human's
    /// call on 2026-09-03).
    ///
    /// **The factorization is deliberately not rebuilt**, because the Python original does not
    /// rebuild it either: `A = (1 + σk) I + θ k² κ² B` is factored in `__init__` and
    /// `plate.B = X` there rebinds one attribute, leaving `_lu` as it was. Assigning a *different*
    /// operator therefore gives an inconsistent plate on both sides equally, which is the fidelity
    /// this setter is for; assigning a reordering of the same operator — the only use — leaves the
    /// factorization correct, since the sort moved no value.
    ///
    /// Refused on the free branch, where `Plate` has no `B` at all. That is a divergence and it is
    /// deliberate: Python would happily attach an unused attribute to the instance, and silently
    /// accepting a stiffness the free step never reads is worse than a raise.
    #[setter]
    fn set_B(&mut self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        if self.p.boundary != core::Boundary::Supported {
            return Err(pyo3::exceptions::PyAttributeError::new_err(
                "'Plate' object has no attribute 'B'",
            ));
        }
        self.p.stiffness = csr_from_scipy(py, value, "B", self.p.n_live)?;
        self.stiffness = value.clone().unbind();
        Ok(())
    }

    /// The energy-first stiffness `K` — free branch only.
    #[getter]
    fn K(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if self.p.boundary != core::Boundary::Free {
            return Err(pyo3::exceptions::PyAttributeError::new_err(
                "'Plate' object has no attribute 'K'",
            ));
        }
        Ok(self.stiffness.clone_ref(py))
    }

    /// The diagonal lumped mass `W` — free branch only.
    #[getter]
    fn W(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.mass {
            Some(m) => Ok(m.clone_ref(py)),
            None => Err(pyo3::exceptions::PyAttributeError::new_err(
                "'Plate' object has no attribute 'W'",
            )),
        }
    }

    /// `W.diagonal()` — the lumped cell areas. Free branch only.
    #[getter]
    fn w(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f64>>> {
        match &self.w {
            Some(w) => Ok(w.clone_ref(py)),
            None => Err(pyo3::exceptions::PyAttributeError::new_err(
                "'Plate' object has no attribute 'w'",
            )),
        }
    }

    // -- state -----------------------------------------------------------------------------

    #[getter]
    fn u(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.u.clone_ref(py)
    }
    #[setter]
    fn set_u(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.u = self.adopt(value, "u")?;
        Ok(())
    }
    #[getter]
    fn u_prev(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.u_prev.clone_ref(py)
    }
    #[setter]
    fn set_u_prev(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.u_prev = self.adopt(value, "u_prev")?;
        Ok(())
    }

    /// The acceleration cache. Private by name and public by use — `airbox`'s `_PlateSurface`
    /// writes it on every commit, which is §12.2 for a third model.
    #[getter]
    fn _accel(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.accel.clone_ref(py)
    }
    #[setter]
    fn set__accel(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.accel = self.adopt(value, "_accel")?;
        Ok(())
    }

    /// The prefactored system matrix. Private by name and public by use — see `PySparseLu`.
    #[getter]
    fn _lu(&self) -> PySparseLu {
        PySparseLu::from_core(self.p.lu.clone())
    }
    #[getter]
    fn n(&self) -> usize {
        self.n
    }
    #[setter]
    fn set_n(&mut self, value: usize) {
        self.n = value;
    }

    /// Current displacement as a full 2-D field, dead nodes zero.
    #[getter]
    fn state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let full = self.with_state(py, |u, _| {
            physsynth_core::ops2d::embed(u, &self.p.index_map)
        })?;
        to_2d_f64(py, full, self.p.mask.nrows(), self.p.mask.ncols())
    }

    // -- initial conditions ----------------------------------------------------------------

    /// Select the live-node values from a full 2-D `field`.
    fn to_live(&self, py: Python<'_>, field: &Bound<'_, PyAny>) -> PyResult<Py<PyArray1<f64>>> {
        let (nrows, ncols) = (self.p.mask.nrows(), self.p.mask.ncols());
        let (shape, values) = as_f64_field(py, field, "field")?;
        if shape != [nrows, ncols] {
            return Err(PyValueError::new_err(format!(
                "field must have shape {}, got {}.",
                shape_repr(&[nrows, ncols]),
                shape_repr(&shape)
            )));
        }
        let live: Vec<f64> = values
            .iter()
            .zip(self.p.mask.flags().iter())
            .filter(|(_, &alive)| alive)
            .map(|(&v, _)| v)
            .collect();
        Ok(PyArray1::from_vec(py, live).unbind())
    }

    /// Set the initial displacement (and optional velocity), consistent to second order.
    #[pyo3(signature = (u0, v0=None))]
    fn set_state(
        &mut self,
        py: Python<'_>,
        u0: &Bound<'_, PyAny>,
        v0: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let (nrows, ncols) = (self.p.mask.nrows(), self.p.mask.ncols());
        let flags = self.p.mask.flags().to_vec();
        let u = live_arg(py, u0, "u0", &flags, nrows, ncols, self.p.n_live)?;
        let v = velocity_arg(py, v0, &flags, nrows, ncols, self.p.n_live)?;
        let prev = core::initial_previous(&u, &v, &self.p);
        let accel = core::initial_accel(&u, &self.p);
        self.u = PyArray1::from_vec(py, u).unbind();
        self.u_prev = PyArray1::from_vec(py, prev).unbind();
        self.accel = PyArray1::from_vec(py, accel).unbind();
        self.n = 0;
        Ok(())
    }

    // -- time stepping ---------------------------------------------------------------------

    /// Advance one timestep via the prefactored sparse solve, rolling the history.
    #[pyo3(signature = (f_ext=None))]
    fn step(&mut self, py: Python<'_>, f_ext: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let force = match f_ext {
            Some(obj) => Some(crate::as_1d_f64(py, obj, "f_ext", self.p.n_live)?),
            None => None,
        };
        let (rhs, u_now, u_was) = self.with_state(py, |u, up| {
            (
                core::step_rhs(u, up, force.as_deref(), &self.p),
                u.to_vec(),
                up.to_vec(),
            )
        })?;
        let next = self
            .p
            .lu
            .solve(&rhs)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let k2 = self.p.k * self.p.k;
        let accel: Vec<f64> = (0..self.p.n_live)
            .map(|i| (next[i] - 2.0 * u_now[i] + u_was[i]) / k2)
            .collect();
        self.accel = PyArray1::from_vec(py, accel).unbind();
        let fresh = PyArray1::from_vec(py, next).unbind();
        self.u_prev = std::mem::replace(&mut self.u, fresh);
        self.n += 1;
        Ok(())
    }

    // -- diagnostics -----------------------------------------------------------------------

    /// Discrete energy `E^n` (Joules).
    fn energy(&self, py: Python<'_>) -> PyResult<f64> {
        self.with_state(py, |u, up| core::energy(u, up, &self.p))
    }

    /// Displacement at flat live-node `index` — a pickup for spectral analysis.
    fn displacement_at(&self, py: Python<'_>, index: i64) -> PyResult<f64> {
        let bound = self.u.bind(py);
        let ro = bound.readonly();
        node_value(state_slice(&ro, "u")?, index)
    }

    /// Flat live-node index nearest the physical point `(x, y)`.
    fn pickup_index_at(&self, x: f64, y: f64) -> usize {
        core::pickup_index_at(x, y, &self.p)
    }

    /// Radiated pressure read-out — the monopole, proportional to volume acceleration.
    fn pressure(&self, py: Python<'_>) -> PyResult<f64> {
        let bound = self.accel.bind(py);
        let ro = bound.readonly();
        Ok(core::pressure(state_slice(&ro, "_accel")?, &self.p))
    }
}

/// Real orthotropic material to the ratios `Plate` takes — the optional realism layer.
///
/// Returns the same seven-field named tuple the original does, built through
/// `physsynth.core.plate.GrainSpec` so `isinstance` and attribute access are unchanged.
#[pyfunction]
#[pyo3(signature = (*, E_x, E_y, nu_xy, G_xy, thickness, rho))]
pub fn grain_ratios_from_material(
    py: Python<'_>,
    E_x: f64,
    E_y: f64,
    nu_xy: f64,
    G_xy: f64,
    thickness: f64,
    rho: f64,
) -> PyResult<Py<PyAny>> {
    let s = core::grain_ratios_from_material(E_x, E_y, nu_xy, G_xy, thickness, rho)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let spec = py
        .import("physsynth.core.plate")?
        .getattr("GrainSpec")?
        .call1((
            s.kappa,
            s.rho_s,
            s.grain_x,
            s.grain_cross,
            s.grain_y,
            s.grain_coupling,
            s.grain_torsion,
        ))?;
    Ok(spec.unbind())
}

/// A von Karman **nonlinear** plate — the Rust implementation, wearing the Python interface.
///
/// Attribute-for-attribute and method-for-method compatible with `physsynth.core.plate.VKPlate`;
/// the docstring on that class is the reference.
///
/// The four `_`-prefixed methods below are `airbox.py`'s, not this class's own convenience:
/// `_VKPlateSurface.solve` runs the Picard loop itself against the **loaded** factorization and
/// calls every one of them per sweep.
#[pyclass(name = "VKPlate", module = "physsynth_rs")]
pub struct PyVKPlate {
    p: core::VkParams,
    boundary: Py<PyAny>,
    grid_x: Py<PyAny>,
    grid_y: Py<PyAny>,
    mask: Py<PyAny>,
    index_map: Py<PyAny>,
    laplacian: Option<Py<PyAny>>,
    stiffness: Py<PyAny>,
    mass: Option<Py<PyAny>>,
    wdiag: Option<Py<PyArray1<f64>>>,
    bracket: Py<PyVonKarmanBracket>,
    airy: Py<PyAiryStressSolver>,
    u: Py<PyArray1<f64>>,
    u_prev: Py<PyArray1<f64>>,
    f: Py<PyArray1<f64>>,
    f_prev: Py<PyArray1<f64>>,
    n: usize,
    n_iters: usize,
    converged: bool,
    last_residual: f64,
}

impl PyVKPlate {
    /// Validate an array being assigned to a state attribute and take ownership of it.
    fn adopt(
        &self,
        value: &Bound<'_, PyAny>,
        name: &str,
        want: usize,
    ) -> PyResult<Py<PyArray1<f64>>> {
        let arr: Bound<'_, PyArray1<f64>> = value.clone().cast_into().map_err(|_| {
            PyValueError::new_err(format!("{name} must be a 1-D float64 numpy array."))
        })?;
        let ro = arr.readonly();
        if ro.len() != want {
            return Err(PyValueError::new_err(format!(
                "{name} must have shape ({want},), got ({},).",
                ro.len()
            )));
        }
        state_slice(&ro, name)?;
        Ok(arr.unbind())
    }

    /// A stored buffer as an owned `Vec`, since the kernels take slices and the GIL borrow ends.
    fn buffer(&self, py: Python<'_>, which: &Py<PyArray1<f64>>, name: &str) -> PyResult<Vec<f64>> {
        let bound = which.bind(py);
        let ro = bound.readonly();
        Ok(state_slice(&ro, name)?.to_vec())
    }
}

#[pymethods]
impl PyVKPlate {
    #[allow(clippy::too_many_arguments)]
    #[new]
    #[pyo3(signature = (*, Lx, Ly, E, e, nu, rho, fs, N, sigma=0.0, theta=core::THETA_DEFAULT,
                        boundary=None::<Py<PyAny>>, nonlinear=true, couple_tol=1e-13,
                        couple_max_iter=50))]
    fn new(
        py: Python<'_>,
        Lx: f64,
        Ly: f64,
        E: f64,
        e: f64,
        nu: f64,
        rho: f64,
        fs: f64,
        N: i64,
        sigma: f64,
        theta: f64,
        boundary: Option<Option<Py<PyAny>>>,
        nonlinear: bool,
        couple_tol: f64,
        couple_max_iter: i64,
    ) -> PyResult<Self> {
        let (boundary_obj, parsed) = resolve_boundary(py, boundary, "supported");
        let spec = core::VkSpec {
            lx: Lx,
            ly: Ly,
            young: E,
            thickness: e,
            nu,
            rho,
            fs,
            n: N,
            sigma,
            theta,
            boundary: parsed,
            nonlinear,
            couple_tol,
            couple_max_iter,
        };
        let p = core::VkParams::new(&spec).map_err(|err| match err {
            core::VkParamError::BadBoundary => PyValueError::new_err(format!(
                "boundary must be 'supported' or 'free', got {}.",
                shown(&boundary_obj)
            )),
            other => PyValueError::new_err(other.to_string()),
        })?;

        let lin = &p.lin;
        let (nrows, ncols) = (lin.mask.nrows(), lin.mask.ncols());
        let grid_x = to_2d_f64(py, lin.x.clone(), nrows, ncols)?;
        let grid_y = to_2d_f64(py, lin.y.clone(), nrows, ncols)?;
        let mask = to_2d_bool(py, lin.mask.flags().to_vec(), nrows, ncols)?;
        let index_map = to_2d_i64(py, lin.index_map.clone(), nrows, ncols)?;
        let laplacian = match &lin.laplacian {
            Some(l) => Some(csr_object(py, l)?),
            None => None,
        };
        let stiffness = csr_object(py, &lin.stiffness)?;
        let mass = match &lin.mass {
            Some(m) => Some(csr_object(py, m)?),
            None => None,
        };
        let wdiag = if lin.w.is_empty() {
            None
        } else {
            Some(PyArray1::from_slice(py, &lin.w).unbind())
        };
        // Cloned, not rebuilt: `AiryStressSolver::new` factors `B_F`, and the plate already owns a
        // factored one. A getter that assembled a second would double construction.
        let bracket = Py::new(py, PyVonKarmanBracket::from_core(py, p.bracket.clone())?)?;
        let airy = Py::new(py, PyAiryStressSolver::from_core(py, p.airy.clone())?)?;
        let (n_live, n_nodes) = (lin.n_live, p.n_nodes);
        Ok(PyVKPlate {
            p,
            boundary: boundary_obj.unbind(),
            grid_x,
            grid_y,
            mask,
            index_map,
            laplacian,
            stiffness,
            mass,
            wdiag,
            bracket,
            airy,
            u: PyArray1::from_vec(py, vec![0.0; n_live]).unbind(),
            u_prev: PyArray1::from_vec(py, vec![0.0; n_live]).unbind(),
            f: PyArray1::from_vec(py, vec![0.0; n_nodes]).unbind(),
            f_prev: PyArray1::from_vec(py, vec![0.0; n_nodes]).unbind(),
            n: 0,
            n_iters: 0,
            converged: true,
            last_residual: 0.0,
        })
    }

    // -- parameters ------------------------------------------------------------------------

    #[getter]
    fn E(&self) -> f64 {
        self.p.young
    }
    #[getter]
    fn e(&self) -> f64 {
        self.p.thickness
    }
    #[getter]
    fn nu(&self) -> f64 {
        self.p.lin.nu
    }
    #[getter]
    fn rho_v(&self) -> f64 {
        self.p.rho_v
    }
    #[getter]
    fn rho_s(&self) -> f64 {
        self.p.rho_s
    }
    #[getter]
    fn D(&self) -> f64 {
        self.p.d
    }
    #[getter]
    fn kappa(&self) -> f64 {
        self.p.lin.kappa
    }
    #[getter]
    fn Y_mem(&self) -> f64 {
        self.p.y_mem
    }
    #[getter]
    fn fs(&self) -> f64 {
        self.p.lin.fs
    }
    #[getter]
    fn N(&self) -> usize {
        self.p.lin.n
    }
    #[getter]
    fn Ny(&self) -> usize {
        self.p.lin.ny
    }
    #[getter]
    fn Lx(&self) -> f64 {
        self.p.lin.lx
    }
    #[getter]
    fn Ly(&self) -> f64 {
        self.p.lin.ly
    }
    #[getter]
    fn sigma(&self) -> f64 {
        self.p.lin.sigma
    }
    #[getter]
    fn theta(&self) -> f64 {
        self.p.lin.theta
    }
    #[getter]
    fn nonlinear(&self) -> bool {
        self.p.nonlinear
    }
    #[getter]
    fn couple_tol(&self) -> f64 {
        self.p.couple_tol
    }
    #[getter]
    fn couple_max_iter(&self) -> usize {
        self.p.couple_max_iter
    }
    #[getter]
    fn k(&self) -> f64 {
        self.p.lin.k
    }
    #[getter]
    fn h(&self) -> f64 {
        self.p.lin.h
    }
    #[getter]
    fn mu(&self) -> f64 {
        self.p.lin.mu
    }
    #[getter]
    fn n_live(&self) -> usize {
        self.p.lin.n_live
    }
    #[getter]
    fn n_nodes(&self) -> usize {
        self.p.n_nodes
    }
    #[getter]
    fn force_denominator(&self) -> f64 {
        self.p.force_denominator
    }
    #[getter]
    fn boundary(&self, py: Python<'_>) -> Py<PyAny> {
        self.boundary.clone_ref(py)
    }
    #[getter]
    fn X(&self, py: Python<'_>) -> Py<PyAny> {
        self.grid_x.clone_ref(py)
    }
    #[getter]
    fn Y(&self, py: Python<'_>) -> Py<PyAny> {
        self.grid_y.clone_ref(py)
    }
    #[getter]
    fn mask(&self, py: Python<'_>) -> Py<PyAny> {
        self.mask.clone_ref(py)
    }
    #[getter]
    fn index_map(&self, py: Python<'_>) -> Py<PyAny> {
        self.index_map.clone_ref(py)
    }
    #[getter]
    fn L(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.laplacian {
            Some(l) => Ok(l.clone_ref(py)),
            None => Err(pyo3::exceptions::PyAttributeError::new_err(
                "'VKPlate' object has no attribute 'L'",
            )),
        }
    }
    #[getter]
    fn B(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if self.p.lin.boundary != core::Boundary::Supported {
            return Err(pyo3::exceptions::PyAttributeError::new_err(
                "'VKPlate' object has no attribute 'B'",
            ));
        }
        Ok(self.stiffness.clone_ref(py))
    }
    #[getter]
    fn K(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if self.p.lin.boundary != core::Boundary::Free {
            return Err(pyo3::exceptions::PyAttributeError::new_err(
                "'VKPlate' object has no attribute 'K'",
            ));
        }
        Ok(self.stiffness.clone_ref(py))
    }
    #[getter]
    fn W(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.mass {
            Some(m) => Ok(m.clone_ref(py)),
            None => Err(pyo3::exceptions::PyAttributeError::new_err(
                "'VKPlate' object has no attribute 'W'",
            )),
        }
    }
    #[getter]
    fn wdiag(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f64>>> {
        match &self.wdiag {
            Some(w) => Ok(w.clone_ref(py)),
            None => Err(pyo3::exceptions::PyAttributeError::new_err(
                "'VKPlate' object has no attribute 'wdiag'",
            )),
        }
    }

    /// The shared Monge-Ampere bracket. `airbox.py` calls this object once per Picard sweep.
    #[getter]
    fn bracket(&self, py: Python<'_>) -> Py<PyVonKarmanBracket> {
        self.bracket.clone_ref(py)
    }
    /// The clamped Airy stress solve.
    #[getter]
    fn airy(&self, py: Python<'_>) -> Py<PyAiryStressSolver> {
        self.airy.clone_ref(py)
    }

    // -- state -----------------------------------------------------------------------------

    #[getter]
    fn u(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.u.clone_ref(py)
    }
    #[setter]
    fn set_u(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.u = self.adopt(value, "u", self.p.lin.n_live)?;
        Ok(())
    }
    #[getter]
    fn u_prev(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.u_prev.clone_ref(py)
    }
    #[setter]
    fn set_u_prev(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.u_prev = self.adopt(value, "u_prev", self.p.lin.n_live)?;
        Ok(())
    }

    /// `F(w^n)` on the full grid. Written by `airbox`'s commit, which rolls **two** histories.
    #[getter]
    fn F(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.f.clone_ref(py)
    }
    #[setter]
    fn set_F(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.f = self.adopt(value, "F", self.p.n_nodes)?;
        Ok(())
    }
    #[getter]
    fn F_prev(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.f_prev.clone_ref(py)
    }
    #[setter]
    fn set_F_prev(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.f_prev = self.adopt(value, "F_prev", self.p.n_nodes)?;
        Ok(())
    }

    /// The prefactored theta-scheme matrix. `airbox.py` deliberately never assigns its *loaded*
    /// factorization here, and a test asserts that; this is the bare one.
    #[getter]
    fn _lu(&self) -> PySparseLu {
        PySparseLu::from_core(self.p.lin.lu.clone())
    }
    #[getter]
    fn n(&self) -> usize {
        self.n
    }
    #[setter]
    fn set_n(&mut self, value: usize) {
        self.n = value;
    }
    #[getter]
    fn n_iters(&self) -> usize {
        self.n_iters
    }
    #[setter]
    fn set_n_iters(&mut self, value: usize) {
        self.n_iters = value;
    }
    #[getter]
    fn converged(&self) -> bool {
        self.converged
    }
    #[setter]
    fn set_converged(&mut self, value: bool) {
        self.converged = value;
    }
    #[getter]
    fn last_residual(&self) -> f64 {
        self.last_residual
    }
    #[setter]
    fn set_last_residual(&mut self, value: f64) {
        self.last_residual = value;
    }

    /// Current displacement as a full 2-D field, rim zero.
    #[getter]
    fn state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let u = self.buffer(py, &self.u, "u")?;
        let full = physsynth_core::ops2d::embed(&u, &self.p.lin.index_map);
        to_2d_f64(py, full, self.p.lin.mask.nrows(), self.p.lin.mask.ncols())
    }

    /// The current Airy stress function as a full 2-D field.
    #[getter]
    fn stress_field(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let f = self.buffer(py, &self.f, "F")?;
        to_2d_f64(py, f, self.p.lin.mask.nrows(), self.p.lin.mask.ncols())
    }

    // -- the seam `airbox.py` steps through ------------------------------------------------

    /// Scatter a live-node vector to a full-grid vector (rim held at 0).
    #[pyo3(name = "_to_full")]
    fn to_full_py(&self, py: Python<'_>, u_live: &Bound<'_, PyAny>) -> PyResult<Py<PyArray1<f64>>> {
        let u = crate::as_1d_f64(py, u_live, "u_live", self.p.lin.n_live)?;
        Ok(PyArray1::from_vec(py, self.p.to_full(&u)).unbind())
    }

    /// Restrict a full-grid vector to the live nodes.
    #[pyo3(name = "_to_live")]
    fn to_live_py(&self, py: Python<'_>, full: &Bound<'_, PyAny>) -> PyResult<Py<PyArray1<f64>>> {
        let v = crate::as_1d_f64(py, full, "full_vec", self.p.n_nodes)?;
        Ok(PyArray1::from_vec(py, self.p.to_live(&v)).unbind())
    }

    /// Solve for the stress function from a full-grid `w`.
    #[pyo3(name = "_airy_F")]
    fn airy_f_py(&self, py: Python<'_>, w_full: &Bound<'_, PyAny>) -> PyResult<Py<PyArray1<f64>>> {
        let w = crate::as_1d_f64(py, w_full, "w_full", self.p.n_nodes)?;
        let f = self
            .p
            .airy_f(&w)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyArray1::from_vec(py, f).unbind())
    }

    /// The linear theta-scheme right-hand side — `Plate`'s own, by construction.
    #[pyo3(name = "_linear_rhs")]
    fn linear_rhs_py(&self, py: Python<'_>) -> PyResult<Py<PyArray1<f64>>> {
        let u = self.buffer(py, &self.u, "u")?;
        let up = self.buffer(py, &self.u_prev, "u_prev")?;
        Ok(PyArray1::from_vec(py, core::step_rhs(&u, &up, None, &self.p.lin)).unbind())
    }

    // -- initial conditions ----------------------------------------------------------------

    /// Select the live-node values from a full 2-D `field`.
    fn to_live(&self, py: Python<'_>, field: &Bound<'_, PyAny>) -> PyResult<Py<PyArray1<f64>>> {
        let (nrows, ncols) = (self.p.lin.mask.nrows(), self.p.lin.mask.ncols());
        let (shape, values) = as_f64_field(py, field, "field")?;
        if shape != [nrows, ncols] {
            return Err(PyValueError::new_err(format!(
                "field must have shape {}, got {}.",
                shape_repr(&[nrows, ncols]),
                shape_repr(&shape)
            )));
        }
        Ok(PyArray1::from_vec(py, self.p.to_live(&values)).unbind())
    }

    /// Set the initial displacement (and optional velocity), seeding both cached `F`s.
    #[pyo3(signature = (u0, v0=None))]
    fn set_state(
        &mut self,
        py: Python<'_>,
        u0: &Bound<'_, PyAny>,
        v0: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let lin = &self.p.lin;
        let (nrows, ncols) = (lin.mask.nrows(), lin.mask.ncols());
        let flags = lin.mask.flags().to_vec();
        let u = live_arg(py, u0, "u0", &flags, nrows, ncols, lin.n_live)?;
        let v = velocity_arg(py, v0, &flags, nrows, ncols, lin.n_live)?;
        let start = core::vk_initial_state(&u, &v, &self.p)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        self.u = PyArray1::from_vec(py, u).unbind();
        self.u_prev = PyArray1::from_vec(py, start.u_prev).unbind();
        self.f = PyArray1::from_vec(py, start.f).unbind();
        self.f_prev = PyArray1::from_vec(py, start.f_prev).unbind();
        self.n = 0;
        Ok(())
    }

    // -- time stepping ---------------------------------------------------------------------

    /// Advance one timestep: one solve when linear, a Picard loop when not.
    #[pyo3(signature = (f_ext=None))]
    fn step(&mut self, py: Python<'_>, f_ext: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let force = match f_ext {
            Some(obj) => Some(crate::as_1d_f64(py, obj, "f_ext", self.p.lin.n_live)?),
            None => None,
        };
        let u = self.buffer(py, &self.u, "u")?;
        let up = self.buffer(py, &self.u_prev, "u_prev")?;
        let fc = self.buffer(py, &self.f, "F")?;
        let fp = self.buffer(py, &self.f_prev, "F_prev")?;
        let out = core::vk_step(&u, &up, &fc, &fp, force.as_deref(), &self.p)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let fresh = PyArray1::from_vec(py, out.u).unbind();
        self.u_prev = std::mem::replace(&mut self.u, fresh);
        // The linear path returns early in the original and does NOT roll the F cache.
        if let Some(f_new) = out.f {
            let fresh_f = PyArray1::from_vec(py, f_new).unbind();
            self.f_prev = std::mem::replace(&mut self.f, fresh_f);
        }
        self.n += 1;
        self.n_iters = out.n_iters;
        self.converged = out.converged;
        self.last_residual = out.last_residual;
        Ok(())
    }

    // -- diagnostics -----------------------------------------------------------------------

    /// Kinetic plus bending energy — the linear theta-scheme energy.
    fn linear_energy(&self, py: Python<'_>) -> PyResult<f64> {
        let u = self.buffer(py, &self.u, "u")?;
        let up = self.buffer(py, &self.u_prev, "u_prev")?;
        Ok(core::energy(&u, &up, &self.p.lin))
    }

    /// Half-step membrane energy; zero when the coupling is off.
    fn membrane_energy(&self, py: Python<'_>) -> PyResult<f64> {
        if !self.p.nonlinear {
            return Ok(0.0);
        }
        let f = self.buffer(py, &self.f, "F")?;
        let fp = self.buffer(py, &self.f_prev, "F_prev")?;
        Ok(0.5 * (self.p.membrane_energy_of(&f) + self.p.membrane_energy_of(&fp)))
    }

    /// Total discrete energy.
    fn energy(&self, py: Python<'_>) -> PyResult<f64> {
        Ok(self.linear_energy(py)? + self.membrane_energy(py)?)
    }

    /// Displacement at flat live-node `index`.
    fn displacement_at(&self, py: Python<'_>, index: i64) -> PyResult<f64> {
        let bound = self.u.bind(py);
        let ro = bound.readonly();
        node_value(state_slice(&ro, "u")?, index)
    }

    /// Flat live-node index nearest the physical point `(x, y)`.
    fn pickup_index_at(&self, x: f64, y: f64) -> usize {
        core::pickup_index_at(x, y, &self.p.lin)
    }
}
