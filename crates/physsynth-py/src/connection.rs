//! `connection.py`: the four bridges — the coupling node of `exciter -> resonator -> body`.
//!
//! Plan section 34. `StringBodyBridge`, `StringPlateBridge`, `StringVKPlateBridge` and
//! `SympatheticStrings`, ported as one unit because two exact anchors bind them together
//! (section 15.2): `tests/test_sympathetic.py` asserts a one-string `SympatheticStrings` is
//! `array_equal` to a `StringBodyBridge`, and
//! `tests/test_airbox_vk.py::test_the_nonlinear_false_chain_is_the_linear_bridge_bit_identical`
//! asserts `StringVKPlateBridge.stability_margin == StringPlateBridge.stability_margin` to the
//! last digit.
//!
//! ## The second module with no core half
//!
//! `airbox_wrap.rs` is the precedent and the reasoning is section 32.2's, reached by a different
//! road. There the tier below had *promised* its clients they could replace a port's matrices; here
//! the class itself is **polymorphic over its collaborators' types** and always was. The `body=`
//! slot takes a `ModalBody` of either language, a `RadiatedBody`, a `ReactiveRadiatedBody` and a
//! `RoomLoadedBody`; the `plate=` slot takes a `Plate`, a `VKPlate` and four room wrappers. There
//! is no `isinstance`, `hasattr`, `getattr` or `type(` anywhere in the reference — it is pure duck
//! typing (section 31.11) — so this port reads every collaborator attribute by name and calls
//! every collaborator method by name. A downcast to a concrete `#[pyclass]` would turn a class
//! that handles eight kinds of body into one that handles two.
//!
//! The line drawn, therefore:
//!
//! * **Through Python:** `scipy.sparse`'s assembly, `spsolve`, `splu(...).solve`,
//!   `np.linalg.eigvals`, `np.dot` (BLAS fuses its multiply-add — section 14.2), the string's
//!   `_second_diff`, and every `step`/`energy`/`pressure` on a collaborator.
//! * **In Rust:** the four validation chains, the `beta` precomputes, `_apply_A`'s elementwise
//!   arithmetic, the stretch/force/energy ledgers and `step`'s sequencing. Elementwise `+ - * /`
//!   on f64 admits no reassociation, so that half is exact by construction.
//!
//! `np.sum` is the reduction that transcribes: section 31.2 established that NumPy's pairwise
//! blocking is an algorithm rather than a CPU-dispatched kernel, not a claim about the machine. Of
//! the two sums here, `SympatheticStrings.step`'s `sum(forces)` answers section 14.2's question
//! **yes** — it reaches the next timestep through `body.step(force=)` — while `beta_b` answers it
//! **no**: its only consumer is `cfl_2dof`, which the reference itself calls a cheap diagnostic and
//! not the real guard. It is transcribed anyway because here that costs one `reduce::sum`, where in
//! section 30's room ledgers the same exactness would have cost a refusal.
//!
//! ## The private names, which section 33.11 said were not there
//!
//! `connection.py` reaches `string._bc_right` (four times) and `string._second_diff` (twice). The
//! plan called this file private-name-free as recently as section 33.11; it is not, and the only
//! reason it never blocked anything is that Phase 0 predicted exactly these two names three phases
//! early and `lib.rs` has exposed them ever since. See section 34.4.
//!
//! ## SciPy is looked up as a module global
//!
//! The reference's `_stability_margin` calls the *module-global* `sparse`, `spsolve` and `splu` in
//! `physsynth.core.connection`, so this file reads the same three names at call time. That is the
//! faithful transcription, and it is what lets a parity file put both languages' guards on one
//! factorization (section 24.4's manoeuvre, sixth use).

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use physsynth_core::fmt::py_exp;
use physsynth_core::reduce;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyList, PyTuple};

/// The reference's `_CFL_TOL`.
const CFL_TOL: f64 = 1e-12;

/// `connection.py`'s own module namespace — for `sparse`, `spsolve` and `splu`.
fn connection_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("physsynth.core.connection")
}

/// A 1-D float64 array as a `Vec<f64>`. The fast arm is a downcast, which costs no Python call;
/// the slow one is `np.ascontiguousarray(obj, float64)` for a list or a strided view.
fn vec1(py: Python<'_>, obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<f64>> {
    if let Ok(arr) = obj.cast::<PyArray1<f64>>() {
        if let Ok(slice) = arr.readonly().as_slice() {
            return Ok(slice.to_vec());
        }
    }
    let np = py.import("numpy")?;
    let arr = np.call_method1("ascontiguousarray", (obj, np.getattr("float64")?))?;
    let arr: Bound<'_, PyArray1<f64>> = arr
        .cast_into()
        .map_err(|_| PyValueError::new_err(format!("{name} must be a 1-D float64 array.")))?;
    let ro = arr.readonly();
    Ok(ro.as_slice()?.to_vec())
}

/// A fresh NumPy array from a `Vec`, as the object the next Python call will take.
fn pyarr(py: Python<'_>, v: Vec<f64>) -> Py<PyAny> {
    PyArray1::from_vec(py, v).into_any().unbind()
}

/// `float(np.dot(a, b))` — BLAS, and deliberately not transcribed: `ddot` fuses its multiply-add
/// and admits no scalar recipe (plan section 14.2).
fn np_dot(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<f64> {
    py.import("numpy")?.call_method1("dot", (a, b))?.extract()
}

/// `x * obj` with the float on the **left**, so the full number protocol runs exactly as it does
/// in the reference (`(s.T / s.h) * dtd` reaches `sparse.__rmul__`, not `sparse.__mul__`).
fn scalar_mul<'py>(
    py: Python<'py>,
    x: f64,
    obj: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    PyFloat::new(py, x).as_any().mul(obj)
}

/// A float attribute off a collaborator, read live rather than cached — the reference reads
/// `p.theta`, `p.kappa`, `s.rho` at call time, so a caller who mutates one sees the same thing
/// through either implementation.
fn f64_attr(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<f64> {
    obj.getattr(name)?.extract()
}

/// A `usize` attribute off a collaborator.
fn usize_attr(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<usize> {
    obj.getattr(name)?.extract()
}

/// `np.isclose(a, b, rtol=0, atol=1e-15)` — `|a - b| <= 1e-15`, NaN false either way.
///
/// **Not** `airbox_wrap::same_rate`, which looks the same and is not: that one is the reference's
/// `rtol=1e-12, atol=0.0`, a *relative* bound. The two modules refuse a mismatched timestep at
/// different tolerances because their references do; they must not be unified.
fn same_step(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-15
}

/// The reference's right-end refusal, shared by all four constructors. `_bc_right` is a private
/// name on the string and has been part of this binding's surface since Phase 0.
fn require_free_right(string: &Bound<'_, PyAny>, message: &str) -> PyResult<()> {
    if string.getattr("_bc_right")?.extract::<String>()? != "free" {
        return Err(PyValueError::new_err(message.to_owned()));
    }
    Ok(())
}

/// `string.u[-1] -= delta`, written **in place** on the live array object.
///
/// The one line `lib.rs`'s module docstring names as the reason the string's buffers are
/// Python-owned. Reading `u` into a `Vec` and rebinding through a setter would replace the array
/// object — losing any snapshot a caller holds, and impossible on a Python string, which has no
/// setter at all.
fn subtract_at_end(string: &Bound<'_, PyAny>, delta: f64) -> PyResult<()> {
    let u = string.getattr("u")?;
    let last: f64 = u.get_item(-1i32)?.extract()?;
    u.set_item(-1i32, last - delta)
}

/// `body.step(force=F)` — by **keyword**, as the reference spells it, through a keyword dict this
/// instance owns.
///
/// The keyword is not decoration: `radiation`'s `AirLoad.step` takes `(u_free, G)`, so a
/// positional call would silently mean something else on a mis-wired collaborator. Rebuilding the
/// dict per step is measurable at this altitude (section 34.5), so it is built once and its one
/// value replaced.
fn step_body(body: &Bound<'_, PyAny>, kwargs: &Bound<'_, PyDict>, force: f64) -> PyResult<()> {
    kwargs.set_item("force", force)?;
    body.call_method("step", (), Some(kwargs))?;
    Ok(())
}

/// `float(np.max(np.linalg.eigvals(A).real))` for a dense matrix given row by row.
fn max_real_eigenvalue(py: Python<'_>, rows: &[Vec<f64>]) -> PyResult<f64> {
    let a = PyArray2::from_vec2(py, rows)?;
    let np = py.import("numpy")?;
    let eig = np.getattr("linalg")?.call_method1("eigvals", (a,))?;
    np.call_method1("max", (eig.getattr("real")?,))?.extract()
}

/// The string's `-(c^2/h^2) * _second_diff(u)[1:]` — the internal block of `_apply_A`.
///
/// `_second_diff` is the string's own stencil, called so the stability spectrum cannot disagree
/// with the integrator. `x` is the string's `N` free DOFs; node 0 stays clamped.
fn string_internal(py: Python<'_>, s: &Bound<'_, PyAny>, x: &[f64]) -> PyResult<Vec<f64>> {
    let mut u = Vec::with_capacity(x.len() + 1);
    u.push(0.0);
    u.extend_from_slice(x);
    let d2 = s.call_method1("_second_diff", (pyarr(py, u),))?;
    let d2 = vec1(py, &d2, "_second_diff")?;
    let c = f64_attr(s, "c")?;
    let h = f64_attr(s, "h")?;
    let scale = -(c * c / (h * h));
    Ok(d2[1..].iter().map(|v| scale * v).collect())
}

// == the modal bridge ==========================================================================

/// A string terminated on a `ModalBody` through a linear bridge spring.
#[pyclass(dict, name = "StringBodyBridge", module = "physsynth_rs")]
pub struct PyStringBodyBridge {
    string: Py<PyAny>,
    body: Py<PyAny>,
    k_spring: f64,
    k: f64,
    beta_s: f64,
    beta_b: f64,
    cfl_2dof: f64,
    spectral_radius: f64,
    n: i64,
    /// `{"force": F}`, reused across steps — see [`step_body`].
    force_kw: Py<PyDict>,
}

#[pymethods]
impl PyStringBodyBridge {
    #[new]
    #[pyo3(signature = (*, string, body, K))]
    fn new(py: Python<'_>, string: Py<PyAny>, body: Py<PyAny>, K: f64) -> PyResult<Self> {
        let s = string.bind(py);
        let b = body.bind(py);
        let ks = f64_attr(s, "k")?;
        let kb = f64_attr(b, "k")?;
        if !same_step(ks, kb) {
            return Err(PyValueError::new_err(format!(
                "string and body must share a timestep (got k={} vs {}); build them at the same \
                 fs.",
                py_exp(ks, 3),
                py_exp(kb, 3)
            )));
        }
        require_free_right(
            s,
            "the string's right end must be 'free' to attach a body bridge (build it with \
             boundary=('fixed', 'free')).",
        )?;
        if K < 0.0 {
            return Err(PyValueError::new_err("bridge stiffness K must be >= 0."));
        }

        let k = ks;
        // beta_s: inverse of the string end-node mass rho*(h/2). beta_b: the body driving-point
        // inverse modal mass sum, transcribed rather than called because it feeds the guard.
        let beta_s = 2.0 * k * k / (f64_attr(s, "rho")? * f64_attr(s, "h")?);
        let phi = vec1(py, &b.getattr("phi")?, "phi")?;
        let m = vec1(py, &b.getattr("m")?, "m")?;
        let terms: Vec<f64> = phi.iter().zip(m.iter()).map(|(p, mi)| p * p / mi).collect();
        let beta_b = k * k * reduce::sum(&terms);

        let mut me = Self {
            string,
            body,
            k_spring: K,
            k,
            beta_s,
            beta_b,
            cfl_2dof: K * (beta_s + beta_b),
            spectral_radius: 0.0,
            n: 0,
            force_kw: PyDict::new(py).unbind(),
        };
        me.spectral_radius = me.eigenvalue(py)?;
        if me.k * me.k * me.spectral_radius >= 4.0 - CFL_TOL {
            return Err(PyValueError::new_err(format!(
                "connection unstable: k^2 * lambda_max(A) = {:.6} >= 4. Reduce K, raise fs, or \
                 increase the body/string end mass.",
                me.k * me.k * me.spectral_radius
            )));
        }
        Ok(me)
    }

    // -- attributes. Section 33.2's default: writable unless there is a reason not to be. ------

    #[getter]
    fn string(&self, py: Python<'_>) -> Py<PyAny> {
        self.string.clone_ref(py)
    }
    #[setter]
    fn set_string(&mut self, value: Py<PyAny>) {
        self.string = value;
    }
    #[getter]
    fn body(&self, py: Python<'_>) -> Py<PyAny> {
        self.body.clone_ref(py)
    }
    #[setter]
    fn set_body(&mut self, value: Py<PyAny>) {
        self.body = value;
    }
    #[getter]
    fn K(&self) -> f64 {
        self.k_spring
    }
    #[setter]
    fn set_K(&mut self, value: f64) {
        self.k_spring = value;
    }
    #[getter]
    fn k(&self) -> f64 {
        self.k
    }
    #[setter]
    fn set_k(&mut self, value: f64) {
        self.k = value;
    }
    #[getter]
    fn beta_s(&self) -> f64 {
        self.beta_s
    }
    #[setter]
    fn set_beta_s(&mut self, value: f64) {
        self.beta_s = value;
    }
    #[getter]
    fn beta_b(&self) -> f64 {
        self.beta_b
    }
    #[setter]
    fn set_beta_b(&mut self, value: f64) {
        self.beta_b = value;
    }
    #[getter]
    fn cfl_2dof(&self) -> f64 {
        self.cfl_2dof
    }
    #[setter]
    fn set_cfl_2dof(&mut self, value: f64) {
        self.cfl_2dof = value;
    }
    #[getter]
    fn spectral_radius(&self) -> f64 {
        self.spectral_radius
    }
    #[setter]
    fn set_spectral_radius(&mut self, value: f64) {
        self.spectral_radius = value;
    }
    #[getter]
    fn n(&self) -> i64 {
        self.n
    }
    #[setter]
    fn set_n(&mut self, value: i64) {
        self.n = value;
    }

    // -- stability (assembled once at construction, off the hot loop) ----------------------

    /// Apply the coupled leapfrog operator `A` to the stacked `[string DOFs, modal coords]`.
    fn _apply_A(&self, py: Python<'_>, x: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let x = vec1(py, x.bind(py), "x")?;
        Ok(pyarr(py, self.apply_a(py, &x)?))
    }

    /// Largest eigenvalue of `A` — assembled densely, one basis vector per column.
    fn _max_leapfrog_eigenvalue(&self, py: Python<'_>) -> PyResult<f64> {
        self.eigenvalue(py)
    }

    // -- helpers ---------------------------------------------------------------------------

    /// Spring stretch `eta = u_b - w_b` now (`prev=True` -> the previous step).
    #[pyo3(signature = (*, prev=false))]
    fn _stretch(&self, py: Python<'_>, prev: bool) -> PyResult<f64> {
        self.stretch(py, prev)
    }

    /// Current bridge force `F = K eta^n` (explicit; N).
    fn connection_force(&self, py: Python<'_>) -> PyResult<f64> {
        Ok(self.k_spring * self.stretch(py, false)?)
    }

    // -- time stepping ---------------------------------------------------------------------

    /// Advance one timestep. Explicit spring `F = K eta^n` drives both parts at time `n`.
    fn step(&mut self, py: Python<'_>) -> PyResult<()> {
        let f = self.connection_force(py)?;
        let s = self.string.bind(py);
        s.call_method0("step")?;
        subtract_at_end(s, self.beta_s * f)?;
        step_body(self.body.bind(py), self.force_kw.bind(py), f)?;
        self.n += 1;
        Ok(())
    }

    // -- diagnostics -------------------------------------------------------------------------

    /// Total discrete energy `E_string + E_body + E_conn` (Joules).
    fn energy(&self, py: Python<'_>) -> PyResult<f64> {
        let e_conn = 0.5 * self.k_spring * self.stretch(py, false)? * self.stretch(py, true)?;
        let e_s: f64 = self.string.bind(py).call_method0("energy")?.extract()?;
        let e_b: f64 = self.body.bind(py).call_method0("energy")?.extract()?;
        Ok(e_s + e_b + e_conn)
    }

    /// The string displacement field (the visible resonator, for animation snapshots).
    #[getter]
    fn state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self.string.bind(py).getattr("state")?.unbind())
    }

    /// String pickup at node `index`.
    fn displacement_at(&self, py: Python<'_>, index: i64) -> PyResult<f64> {
        self.string
            .bind(py)
            .call_method1("displacement_at", (index,))?
            .extract()
    }

    /// Radiated pressure from the body, `sum_i a_i q_i''`.
    fn pressure(&self, py: Python<'_>) -> PyResult<f64> {
        self.body.bind(py).call_method0("pressure")?.extract()
    }
}

impl PyStringBodyBridge {
    fn apply_a(&self, py: Python<'_>, x: &[f64]) -> PyResult<Vec<f64>> {
        let s = self.string.bind(py);
        let b = self.body.bind(py);
        let n = usize_attr(s, "N")?;
        let mut out_u = string_internal(py, s, &x[..n])?;

        let phi = vec1(py, &b.getattr("phi")?, "phi")?;
        let m = vec1(py, &b.getattr("m")?, "m")?;
        let omega = vec1(py, &b.getattr("omega")?, "omega")?;
        let q: Vec<f64> = x[n..].to_vec();
        let mut out_q: Vec<f64> = omega
            .iter()
            .zip(q.iter())
            .map(|(w, qi)| (w * w) * qi)
            .collect();

        // eta = u_N - sum phi_i q_i, through BLAS exactly as the reference does.
        let u_end = if n == 0 { 0.0 } else { x[n - 1] };
        let eta = u_end - np_dot(py, &b.getattr("phi")?, pyarr(py, q).bind(py))?;

        if let Some(last) = out_u.last_mut() {
            *last += 2.0 * self.k_spring * eta / (f64_attr(s, "rho")? * f64_attr(s, "h")?);
        }
        for (i, o) in out_q.iter_mut().enumerate() {
            *o += -self.k_spring * phi[i] * eta / m[i];
        }
        out_u.extend_from_slice(&out_q);
        Ok(out_u)
    }

    fn eigenvalue(&self, py: Python<'_>) -> PyResult<f64> {
        let n = usize_attr(self.string.bind(py), "N")? + usize_attr(self.body.bind(py), "M")?;
        let mut rows = vec![vec![0.0f64; n]; n];
        let mut e = vec![0.0f64; n];
        for j in 0..n {
            e[j] = 1.0;
            for (i, v) in self.apply_a(py, &e)?.iter().enumerate() {
                rows[i][j] = *v;
            }
            e[j] = 0.0;
        }
        max_real_eigenvalue(py, &rows)
    }

    fn stretch(&self, py: Python<'_>, prev: bool) -> PyResult<f64> {
        let s = self.string.bind(py);
        let b = self.body.bind(py);
        if prev {
            let u_b: f64 = s.getattr("u_prev")?.get_item(-1i32)?.extract()?;
            let w_b = np_dot(py, &b.getattr("phi")?, &b.getattr("q_prev")?)?;
            Ok(u_b - w_b)
        } else {
            let u_b: f64 = s.getattr("u")?.get_item(-1i32)?.extract()?;
            let w_b: f64 = b.call_method0("bridge_displacement")?.extract()?;
            Ok(u_b - w_b)
        }
    }
}

// == the plate bridges =========================================================================

/// The shared body of `StringPlateBridge` and `StringVKPlateBridge`.
///
/// The two classes differ in exactly one name — the plate's areal density is `rho` on a `Plate`
/// and `rho_s` on a `VKPlate` — and an exact anchor says so: with `nonlinear=False` the two guards
/// must agree to the last digit, so the arithmetic below has to be literally one piece of code.
struct PlateBridge {
    string: Py<PyAny>,
    plate: Py<PyAny>,
    k_spring: f64,
    k: f64,
    drive_index: usize,
    beta_s: f64,
    f_ext: Py<PyAny>,
    /// The keyword dict `plate.step(f_ext=...)` is called with. Reused across steps to skip the
    /// per-step allocation (section 34.5), but its **entry is re-set from `self.f_ext` every
    /// step** — a cached array object would ignore a caller who replaced `_f_ext`, which is
    /// section 32.2's hazard one tier up.
    f_ext_kw: Py<PyDict>,
    stability_margin: f64,
    n: i64,
}

impl PlateBridge {
    /// The five refusals both constructors share, then the precomputes and the exact guard.
    fn build(
        py: Python<'_>,
        string: Py<PyAny>,
        plate: Py<PyAny>,
        k_spring: f64,
        drive_index: Option<Py<PyAny>>,
        rho_name: &str,
    ) -> PyResult<Self> {
        let s = string.bind(py);
        let p = plate.bind(py);
        let ks = f64_attr(s, "k")?;
        let kp = f64_attr(p, "k")?;
        if !same_step(ks, kp) {
            return Err(PyValueError::new_err(format!(
                "string and plate must share a timestep (got k={} vs {}); build them at the same \
                 fs.",
                py_exp(ks, 3),
                py_exp(kp, 3)
            )));
        }
        require_free_right(
            s,
            "the string's right end must be 'free' to attach a plate bridge (build it with \
             boundary=('fixed', 'free')).",
        )?;
        let boundary = p.getattr("boundary")?;
        let known = matches!(
            boundary.extract::<String>().as_deref(),
            Ok("supported") | Ok("free")
        );
        if !known {
            return Err(PyValueError::new_err(format!(
                "the plate body must be 'supported' or 'free', got {}.",
                boundary.repr()?
            )));
        }
        if f64_attr(s, "lam")? >= 1.0 - CFL_TOL {
            return Err(PyValueError::new_err(
                "the string must run at lambda < 1: its Nyquist mode is marginal at lambda = 1 \
                 and the bridge spring pushes it unstable (the guard's string block G0_str is \
                 singular there). Rebuild the string at lambda < 1 (0.9 is a good default).",
            ));
        }
        if k_spring < 0.0 {
            return Err(PyValueError::new_err("bridge stiffness K must be >= 0."));
        }

        let k = ks;
        let n_live = usize_attr(p, "n_live")?;
        let chosen = match drive_index {
            Some(d) => d,
            // A corner-offset point: off every low mode's symmetry axis.
            None => p
                .call_method1(
                    "pickup_index_at",
                    (0.3 * f64_attr(p, "Lx")?, 0.4 * f64_attr(p, "Ly")?),
                )?
                .unbind(),
        };
        let as_int: i64 = py
            .import("builtins")?
            .getattr("int")?
            .call1((chosen.bind(py),))?
            .extract()?;
        if as_int < 0 || as_int as usize >= n_live {
            return Err(PyValueError::new_err(format!(
                "drive_index {} out of range [0, {}).",
                chosen.bind(py).str()?,
                n_live
            )));
        }

        // String end-node inverse mass (its h/2 half-cell).
        let beta_s = 2.0 * k * k / (f64_attr(s, "rho")? * f64_attr(s, "h")?);
        let f_ext = pyarr(py, vec![0.0; n_live]);
        let mut me = Self {
            string,
            plate,
            k_spring,
            k,
            drive_index: as_int as usize,
            beta_s,
            // Preallocated live-node force vector (only the drive node is ever nonzero).
            f_ext,
            f_ext_kw: PyDict::new(py).unbind(),
            stability_margin: 0.0,
            n: 0,
        };
        me.stability_margin = me.margin(py, rho_name)?;
        if me.stability_margin >= 1.0 - CFL_TOL {
            return Err(PyValueError::new_err(format!(
                "connection unstable: stability margin = {:.6} >= 1. Reduce K, raise fs, or \
                 increase the string/plate node mass.",
                me.stability_margin
            )));
        }
        Ok(me)
    }

    /// `(k^2/4) K [ (G0_str^-1)_end + (G0_plate^-1)_dp ]`, assembled through SciPy.
    ///
    /// Every sparse product, the two solves and the factorization stay where the reference put
    /// them; what is Rust is the two diagonals and the final three-term product, whose bracketing
    /// is the reference's verbatim because three `test_airbox_*` files compare this number across
    /// a Python/Rust boundary — and one of them across two *classes* — to the bit.
    fn margin(&self, py: Python<'_>, rho_name: &str) -> PyResult<f64> {
        let s = self.string.bind(py);
        let p = self.plate.bind(py);
        let conn = connection_module(py)?;
        let sparse = conn.getattr("sparse")?;
        let csc = PyDict::new(py);
        csc.set_item("format", "csc")?;
        let quarter_k2 = 0.25 * self.k * self.k;

        // String block: G0_str = M_str - (k^2/4) S_str, unknowns = nodes 1..N (node 0 clamped).
        let n = usize_attr(s, "N")?;
        let rho_h = f64_attr(s, "rho")? * f64_attr(s, "h")?;
        let mut m_diag = vec![rho_h; n];
        if let Some(last) = m_diag.last_mut() {
            *last = 0.5 * rho_h;
        }
        let mut main = vec![2.0f64; n];
        if let Some(last) = main.last_mut() {
            *last = 1.0;
        }
        let ones = vec![-1.0f64; n.saturating_sub(1)];
        let diagonals = PyList::new(
            py,
            [pyarr(py, ones.clone()), pyarr(py, main), pyarr(py, ones)],
        )?;
        let dtd = sparse.call_method(
            "diags",
            (diagonals, PyList::new(py, [-1i32, 0, 1])?),
            Some(&csc),
        )?;
        let s_str = scalar_mul(py, f64_attr(s, "T")? / f64_attr(s, "h")?, &dtd)?;
        let mass = sparse.call_method("diags", (pyarr(py, m_diag),), Some(&csc))?;
        let g_str = mass.sub(scalar_mul(py, quarter_k2, &s_str)?)?;
        let mut e_end = vec![0.0; n];
        if let Some(last) = e_end.last_mut() {
            *last = 1.0;
        }
        let g_str_inv_end: f64 = conn
            .getattr("spsolve")?
            .call1((g_str, pyarr(py, e_end)))?
            .get_item(-1i32)?
            .extract()?;

        // Plate block: G0_plate = M + (theta - 1/4) k^2 S (PD for theta >= 1/4).
        //   "supported": rho h^2 [I + (theta-1/4) k^2 kappa^2 B]
        //   "free":      rho [W + (theta-1/4) k^2 kappa^2 K] -- W already carries the h^2 weight.
        let kappa = f64_attr(p, "kappa")?;
        let coeff = (f64_attr(p, "theta")? - 0.25) * self.k * self.k * kappa * kappa;
        let n_live = usize_attr(p, "n_live")?;
        let rho = f64_attr(p, rho_name)?;
        let g_plate = if p.getattr("boundary")?.extract::<String>()? == "supported" {
            let h = f64_attr(p, "h")?;
            let ident = sparse.call_method("identity", (n_live,), Some(&csc))?;
            let inner = ident.add(scalar_mul(py, coeff, &p.getattr("B")?)?)?;
            scalar_mul(py, rho * h * h, &inner)?
        } else {
            let inner = p
                .getattr("W")?
                .add(scalar_mul(py, coeff, &p.getattr("K")?)?)?;
            scalar_mul(py, rho, &inner)?
        };
        let mut e_dp = vec![0.0; n_live];
        e_dp[self.drive_index] = 1.0;
        let g_plate_inv_dp: f64 = conn
            .getattr("splu")?
            .call1((g_plate.call_method0("tocsc")?,))?
            .call_method1("solve", (pyarr(py, e_dp),))?
            .get_item(self.drive_index)?
            .extract()?;

        Ok(quarter_k2 * self.k_spring * (g_str_inv_end + g_plate_inv_dp))
    }

    /// Spring stretch `eta = u_end - w_dp` now (`prev` -> the previous step).
    fn stretch(&self, py: Python<'_>, prev: bool) -> PyResult<f64> {
        let name = if prev { "u_prev" } else { "u" };
        let u_end: f64 = self
            .string
            .bind(py)
            .getattr(name)?
            .get_item(-1i32)?
            .extract()?;
        let w_dp: f64 = self
            .plate
            .bind(py)
            .getattr(name)?
            .get_item(self.drive_index)?
            .extract()?;
        Ok(u_end - w_dp)
    }

    fn force(&self, py: Python<'_>) -> PyResult<f64> {
        Ok(self.k_spring * self.stretch(py, false)?)
    }

    /// One step. The string takes its reaction as a post-step impulse; the plate takes `+F` into
    /// the implicit right-hand side, before the solve (and, for a von Karman plate, outside the
    /// Picard loop).
    fn advance(&mut self, py: Python<'_>) -> PyResult<()> {
        let f = self.force(py)?;
        let s = self.string.bind(py);
        s.call_method0("step")?;
        subtract_at_end(s, self.beta_s * f)?;
        let f_ext = self.f_ext.bind(py);
        f_ext.set_item(self.drive_index, f)?;
        let kwargs = self.f_ext_kw.bind(py);
        kwargs.set_item("f_ext", f_ext)?;
        self.plate.bind(py).call_method("step", (), Some(kwargs))?;
        f_ext.set_item(self.drive_index, 0.0)?;
        self.n += 1;
        Ok(())
    }

    fn total_energy(&self, py: Python<'_>) -> PyResult<f64> {
        let e_conn = 0.5 * self.k_spring * self.stretch(py, false)? * self.stretch(py, true)?;
        let e_s: f64 = self.string.bind(py).call_method0("energy")?.extract()?;
        let e_p: f64 = self.plate.bind(py).call_method0("energy")?.extract()?;
        Ok(e_s + e_p + e_conn)
    }
}

/// A string terminated on a **grid** `Plate` (supported soundboard or free-edge cymbal).
#[pyclass(dict, name = "StringPlateBridge", module = "physsynth_rs")]
pub struct PyStringPlateBridge {
    b: PlateBridge,
}

#[pymethods]
impl PyStringPlateBridge {
    #[new]
    #[pyo3(signature = (*, string, plate, K, drive_index=None::<Py<PyAny>>))]
    fn new(
        py: Python<'_>,
        string: Py<PyAny>,
        plate: Py<PyAny>,
        K: f64,
        drive_index: Option<Option<Py<PyAny>>>,
    ) -> PyResult<Self> {
        // Sections 24.7 and 31.7: `Some(None)` is the omitted keyword and a bare `None` is the
        // caller's literal — and here the reference treats both as "take the corner offset".
        Ok(Self {
            b: PlateBridge::build(py, string, plate, K, drive_index.flatten(), "rho")?,
        })
    }

    #[getter]
    fn string(&self, py: Python<'_>) -> Py<PyAny> {
        self.b.string.clone_ref(py)
    }
    #[setter]
    fn set_string(&mut self, value: Py<PyAny>) {
        self.b.string = value;
    }
    #[getter]
    fn plate(&self, py: Python<'_>) -> Py<PyAny> {
        self.b.plate.clone_ref(py)
    }
    #[setter]
    fn set_plate(&mut self, value: Py<PyAny>) {
        self.b.plate = value;
    }
    #[getter]
    fn K(&self) -> f64 {
        self.b.k_spring
    }
    #[setter]
    fn set_K(&mut self, value: f64) {
        self.b.k_spring = value;
    }
    #[getter]
    fn k(&self) -> f64 {
        self.b.k
    }
    #[setter]
    fn set_k(&mut self, value: f64) {
        self.b.k = value;
    }
    #[getter]
    fn drive_index(&self) -> usize {
        self.b.drive_index
    }
    #[setter]
    fn set_drive_index(&mut self, value: usize) {
        self.b.drive_index = value;
    }
    #[getter]
    fn beta_s(&self) -> f64 {
        self.b.beta_s
    }
    #[setter]
    fn set_beta_s(&mut self, value: f64) {
        self.b.beta_s = value;
    }
    #[getter]
    fn _f_ext(&self, py: Python<'_>) -> Py<PyAny> {
        self.b.f_ext.clone_ref(py)
    }
    #[setter]
    fn set__f_ext(&mut self, value: Py<PyAny>) {
        self.b.f_ext = value;
    }
    #[getter]
    fn stability_margin(&self) -> f64 {
        self.b.stability_margin
    }
    #[setter]
    fn set_stability_margin(&mut self, value: f64) {
        self.b.stability_margin = value;
    }
    #[getter]
    fn n(&self) -> i64 {
        self.b.n
    }
    #[setter]
    fn set_n(&mut self, value: i64) {
        self.b.n = value;
    }

    fn _stability_margin(&self, py: Python<'_>) -> PyResult<f64> {
        self.b.margin(py, "rho")
    }

    #[pyo3(signature = (*, prev=false))]
    fn _stretch(&self, py: Python<'_>, prev: bool) -> PyResult<f64> {
        self.b.stretch(py, prev)
    }

    /// Current bridge force `F = K eta^n` (explicit; N).
    fn connection_force(&self, py: Python<'_>) -> PyResult<f64> {
        self.b.force(py)
    }

    /// Plate displacement `w_dp` at the driving point.
    fn driving_point_displacement(&self, py: Python<'_>) -> PyResult<f64> {
        self.b
            .plate
            .bind(py)
            .getattr("u")?
            .get_item(self.b.drive_index)?
            .extract()
    }

    fn step(&mut self, py: Python<'_>) -> PyResult<()> {
        self.b.advance(py)
    }

    /// Total discrete energy `E_string + E_plate + E_conn` (Joules).
    fn energy(&self, py: Python<'_>) -> PyResult<f64> {
        self.b.total_energy(py)
    }

    #[getter]
    fn state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self.b.string.bind(py).getattr("state")?.unbind())
    }

    fn displacement_at(&self, py: Python<'_>, index: i64) -> PyResult<f64> {
        self.b
            .string
            .bind(py)
            .call_method1("displacement_at", (index,))?
            .extract()
    }

    /// Radiated pressure from the plate (monopole proportional to volume acceleration).
    fn pressure(&self, py: Python<'_>) -> PyResult<f64> {
        self.b.plate.bind(py).call_method0("pressure")?.extract()
    }
}

/// A string terminated on a **von Karman** plate (model #6) — the gong, played through a bridge.
#[pyclass(dict, name = "StringVKPlateBridge", module = "physsynth_rs")]
pub struct PyStringVKPlateBridge {
    b: PlateBridge,
}

#[pymethods]
impl PyStringVKPlateBridge {
    #[new]
    #[pyo3(signature = (*, string, plate, K, drive_index=None::<Py<PyAny>>))]
    fn new(
        py: Python<'_>,
        string: Py<PyAny>,
        plate: Py<PyAny>,
        K: f64,
        drive_index: Option<Option<Py<PyAny>>>,
    ) -> PyResult<Self> {
        Ok(Self {
            // `rho_s`, not `rho_v`. The reference's docstring calls the wrong one this class's
            // silent-failure trap: a margin 1000x too small that passes every construction and
            // leaves every energy ledger green, because each side telescopes against the force it
            // used. The `nonlinear=False` regression is what catches it.
            b: PlateBridge::build(py, string, plate, K, drive_index.flatten(), "rho_s")?,
        })
    }

    #[getter]
    fn string(&self, py: Python<'_>) -> Py<PyAny> {
        self.b.string.clone_ref(py)
    }
    #[setter]
    fn set_string(&mut self, value: Py<PyAny>) {
        self.b.string = value;
    }
    #[getter]
    fn plate(&self, py: Python<'_>) -> Py<PyAny> {
        self.b.plate.clone_ref(py)
    }
    #[setter]
    fn set_plate(&mut self, value: Py<PyAny>) {
        self.b.plate = value;
    }
    #[getter]
    fn K(&self) -> f64 {
        self.b.k_spring
    }
    #[setter]
    fn set_K(&mut self, value: f64) {
        self.b.k_spring = value;
    }
    #[getter]
    fn k(&self) -> f64 {
        self.b.k
    }
    #[setter]
    fn set_k(&mut self, value: f64) {
        self.b.k = value;
    }
    #[getter]
    fn drive_index(&self) -> usize {
        self.b.drive_index
    }
    #[setter]
    fn set_drive_index(&mut self, value: usize) {
        self.b.drive_index = value;
    }
    #[getter]
    fn beta_s(&self) -> f64 {
        self.b.beta_s
    }
    #[setter]
    fn set_beta_s(&mut self, value: f64) {
        self.b.beta_s = value;
    }
    #[getter]
    fn _f_ext(&self, py: Python<'_>) -> Py<PyAny> {
        self.b.f_ext.clone_ref(py)
    }
    #[setter]
    fn set__f_ext(&mut self, value: Py<PyAny>) {
        self.b.f_ext = value;
    }
    #[getter]
    fn stability_margin(&self) -> f64 {
        self.b.stability_margin
    }
    #[setter]
    fn set_stability_margin(&mut self, value: f64) {
        self.b.stability_margin = value;
    }
    #[getter]
    fn n(&self) -> i64 {
        self.b.n
    }
    #[setter]
    fn set_n(&mut self, value: i64) {
        self.b.n = value;
    }

    fn _stability_margin(&self, py: Python<'_>) -> PyResult<f64> {
        self.b.margin(py, "rho_s")
    }

    #[pyo3(signature = (*, prev=false))]
    fn _stretch(&self, py: Python<'_>, prev: bool) -> PyResult<f64> {
        self.b.stretch(py, prev)
    }

    fn connection_force(&self, py: Python<'_>) -> PyResult<f64> {
        self.b.force(py)
    }

    fn driving_point_displacement(&self, py: Python<'_>) -> PyResult<f64> {
        self.b
            .plate
            .bind(py)
            .getattr("u")?
            .get_item(self.b.drive_index)?
            .extract()
    }

    fn step(&mut self, py: Python<'_>) -> PyResult<()> {
        self.b.advance(py)
    }

    /// Total discrete energy `E_string + E_plate + E_conn` — conserved **at the Picard fixed
    /// point**, which is why `converged` is part of the diagnostic surface.
    fn energy(&self, py: Python<'_>) -> PyResult<f64> {
        self.b.total_energy(py)
    }

    #[getter]
    fn state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self.b.string.bind(py).getattr("state")?.unbind())
    }

    fn displacement_at(&self, py: Python<'_>, index: i64) -> PyResult<f64> {
        self.b
            .string
            .bind(py)
            .call_method1("displacement_at", (index,))?
            .extract()
    }

    /// Did the **last step's** Picard iteration reach `couple_tol`? Read it every step.
    #[getter]
    fn converged(&self, py: Python<'_>) -> PyResult<bool> {
        self.b.plate.bind(py).getattr("converged")?.extract()
    }

    /// Picard sweeps the last step used (rises with amplitude; the cap is the failure mode).
    #[getter]
    fn n_iters(&self, py: Python<'_>) -> PyResult<i64> {
        self.b.plate.bind(py).getattr("n_iters")?.extract()
    }

    /// The last step's final `||dw||/||w||`.
    #[getter]
    fn last_residual(&self, py: Python<'_>) -> PyResult<f64> {
        self.b.plate.bind(py).getattr("last_residual")?.extract()
    }
}

// == the shared bridge point ===================================================================

/// Several strings sharing **one** bridge point on a common `ModalBody`.
#[pyclass(dict, name = "SympatheticStrings", module = "physsynth_rs")]
pub struct PySympatheticStrings {
    strings: Py<PyAny>,
    body: Py<PyAny>,
    k_springs: Py<PyAny>,
    k: f64,
    j: usize,
    beta_s: Py<PyAny>,
    offsets: Py<PyAny>,
    spectral_radius: f64,
    n: i64,
    /// `{"force": sum_j F_j}`, reused across steps — see [`step_body`].
    force_kw: Py<PyDict>,
}

#[pymethods]
impl PySympatheticStrings {
    #[new]
    #[pyo3(signature = (*, strings, body, Ks))]
    fn new(py: Python<'_>, strings: Py<PyAny>, body: Py<PyAny>, Ks: Py<PyAny>) -> PyResult<Self> {
        let np = py.import("numpy")?;
        let items: Vec<Bound<'_, PyAny>> =
            strings.bind(py).try_iter()?.collect::<PyResult<Vec<_>>>()?;
        let strings = PyList::new(py, items)?;
        let count = strings.len();
        if count < 1 {
            return Err(PyValueError::new_err("need at least one string."));
        }
        let dtype = PyDict::new(py);
        dtype.set_item("dtype", py.import("builtins")?.getattr("float")?)?;
        let arr = np.call_method1(
            "atleast_1d",
            (np.call_method("asarray", (Ks,), Some(&dtype))?,),
        )?;
        let shape = arr.getattr("shape")?;
        if !shape.eq(PyTuple::new(py, [count])?)? {
            return Err(PyValueError::new_err(format!(
                "Ks must have one stiffness per string (got {} for {} strings).",
                shape.repr()?,
                count
            )));
        }
        let ks = vec1(py, &arr, "Ks")?;
        if ks.iter().any(|v| *v < 0.0) {
            return Err(PyValueError::new_err(
                "every bridge stiffness K must be >= 0.",
            ));
        }

        let b = body.bind(py);
        let kb = f64_attr(b, "k")?;
        for (j, item) in strings.iter().enumerate() {
            let sk = f64_attr(&item, "k")?;
            if !same_step(sk, kb) {
                return Err(PyValueError::new_err(format!(
                    "string {} and the body must share a timestep (got k={} vs {}); build them \
                     at the same fs.",
                    j,
                    py_exp(sk, 3),
                    py_exp(kb, 3)
                )));
            }
            require_free_right(
                &item,
                &format!(
                    "string {j}'s right end must be 'free' to attach to the bridge (build it \
                     with boundary=('fixed', 'free'))."
                ),
            )?;
            if f64_attr(&item, "lam")? >= 1.0 - CFL_TOL {
                return Err(PyValueError::new_err(format!(
                    "string {j} must run at lambda < 1: its Nyquist mode is marginal at \
                     lambda = 1 and the bridge spring pushes it unstable. Rebuild at lambda < 1."
                )));
            }
        }

        let k = kb;
        let mut beta = Vec::with_capacity(count);
        let mut cum = vec![0i64];
        for item in strings.iter() {
            beta.push(2.0 * k * k / (f64_attr(&item, "rho")? * f64_attr(&item, "h")?));
            cum.push(usize_attr(&item, "N")? as i64);
        }
        // `np.cumsum([0] + [s.N for s in strings])`, built through NumPy so the dtype is the
        // reference's platform integer rather than a guess.
        let offsets = np
            .call_method1("cumsum", (PyList::new(py, cum)?,))?
            .unbind();

        let mut me = Self {
            strings: strings.into_any().unbind(),
            body,
            k_springs: arr.unbind(),
            k,
            j: count,
            beta_s: pyarr(py, beta),
            offsets,
            spectral_radius: 0.0,
            n: 0,
            force_kw: PyDict::new(py).unbind(),
        };
        me.spectral_radius = me.eigenvalue(py)?;
        if me.k * me.k * me.spectral_radius >= 4.0 - CFL_TOL {
            return Err(PyValueError::new_err(format!(
                "connection unstable: k^2 * lambda_max(A) = {:.6} >= 4. Reduce the bridge \
                 stiffnesses, raise fs, or increase the body/string end mass.",
                me.k * me.k * me.spectral_radius
            )));
        }
        Ok(me)
    }

    // -- attributes ------------------------------------------------------------------------

    #[getter]
    fn strings(&self, py: Python<'_>) -> Py<PyAny> {
        self.strings.clone_ref(py)
    }
    #[setter]
    fn set_strings(&mut self, value: Py<PyAny>) {
        self.strings = value;
    }
    #[getter]
    fn body(&self, py: Python<'_>) -> Py<PyAny> {
        self.body.clone_ref(py)
    }
    #[setter]
    fn set_body(&mut self, value: Py<PyAny>) {
        self.body = value;
    }
    #[getter]
    fn K(&self, py: Python<'_>) -> Py<PyAny> {
        self.k_springs.clone_ref(py)
    }
    #[setter]
    fn set_K(&mut self, value: Py<PyAny>) {
        self.k_springs = value;
    }
    #[getter]
    fn k(&self) -> f64 {
        self.k
    }
    #[setter]
    fn set_k(&mut self, value: f64) {
        self.k = value;
    }
    #[getter]
    fn J(&self) -> usize {
        self.j
    }
    #[setter]
    fn set_J(&mut self, value: usize) {
        self.j = value;
    }
    #[getter]
    fn beta_s(&self, py: Python<'_>) -> Py<PyAny> {
        self.beta_s.clone_ref(py)
    }
    #[setter]
    fn set_beta_s(&mut self, value: Py<PyAny>) {
        self.beta_s = value;
    }
    #[getter]
    fn _offsets(&self, py: Python<'_>) -> Py<PyAny> {
        self.offsets.clone_ref(py)
    }
    #[setter]
    fn set__offsets(&mut self, value: Py<PyAny>) {
        self.offsets = value;
    }
    #[getter]
    fn spectral_radius(&self) -> f64 {
        self.spectral_radius
    }
    #[setter]
    fn set_spectral_radius(&mut self, value: f64) {
        self.spectral_radius = value;
    }
    #[getter]
    fn n(&self) -> i64 {
        self.n
    }
    #[setter]
    fn set_n(&mut self, value: i64) {
        self.n = value;
    }

    // -- stability ---------------------------------------------------------------------------

    /// Apply the stacked coupled leapfrog operator `A`. All strings couple to the *shared* bridge
    /// displacement `w_b = sum_i phi_i q_i`.
    fn _apply_A(&self, py: Python<'_>, x: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let x = vec1(py, x.bind(py), "x")?;
        Ok(pyarr(py, self.apply_a(py, &x)?))
    }

    fn _max_leapfrog_eigenvalue(&self, py: Python<'_>) -> PyResult<f64> {
        self.eigenvalue(py)
    }

    // -- helpers -----------------------------------------------------------------------------

    /// Shared bridge displacement `w_b = sum_i phi_i q_i`.
    #[pyo3(signature = (*, prev=false))]
    fn _bridge_displacement(&self, py: Python<'_>, prev: bool) -> PyResult<f64> {
        self.bridge_displacement(py, prev)
    }

    /// Stretch `eta_j = u_{end,j} - w_b` of string `j`'s spring.
    #[pyo3(signature = (j, *, prev=false))]
    fn _stretch(&self, py: Python<'_>, j: usize, prev: bool) -> PyResult<f64> {
        let w_b = self.bridge_displacement(py, prev)?;
        let s = self.strings.bind(py).get_item(j)?;
        let name = if prev { "u_prev" } else { "u" };
        let u_end: f64 = s.getattr(name)?.get_item(-1i32)?.extract()?;
        Ok(u_end - w_b)
    }

    /// Per-string bridge forces `F_j = K_j eta_j^n` (explicit; N).
    fn connection_forces(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(pyarr(py, self.forces(py)?))
    }

    // -- time stepping -------------------------------------------------------------------------

    /// Advance one timestep. Each explicit spring drives its string and the shared body, which
    /// feels the scalar sum.
    fn step(&mut self, py: Python<'_>) -> PyResult<()> {
        let forces = self.forces(py)?;
        let beta = vec1(py, self.beta_s.bind(py), "beta_s")?;
        let strings = self.strings.bind(py);
        for (j, f) in forces.iter().enumerate() {
            let s = strings.get_item(j)?;
            s.call_method0("step")?;
            subtract_at_end(&s, beta[j] * f)?;
        }
        // The body feels the scalar sum, which reaches the next timestep — so this reduction is
        // transcribed rather than called (section 31.2 answering section 14.2's question `yes`).
        step_body(
            self.body.bind(py),
            self.force_kw.bind(py),
            reduce::sum(&forces),
        )?;
        self.n += 1;
        Ok(())
    }

    // -- diagnostics ---------------------------------------------------------------------------

    /// Total discrete energy `sum_j E_{string,j} + E_body + sum_j E_{conn,j}` (Joules).
    fn energy(&self, py: Python<'_>) -> PyResult<f64> {
        let ks = vec1(py, self.k_springs.bind(py), "K")?;
        let strings = self.strings.bind(py);
        let mut e: f64 = self.body.bind(py).call_method0("energy")?.extract()?;
        for (j, kj) in ks.iter().enumerate().take(self.j) {
            let s = strings.get_item(j)?;
            e += s.call_method0("energy")?.extract::<f64>()?;
            e += 0.5 * kj * self._stretch(py, j, false)? * self._stretch(py, j, true)?;
        }
        Ok(e)
    }

    /// Energy of string `j` alone — the sympathetic-transfer trace.
    fn string_energy(&self, py: Python<'_>, j: usize) -> PyResult<f64> {
        self.strings
            .bind(py)
            .get_item(j)?
            .call_method0("energy")?
            .extract()
    }

    /// The first string's displacement field (a representative resonator for snapshots).
    #[getter]
    fn state(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self
            .strings
            .bind(py)
            .get_item(0)?
            .getattr("state")?
            .unbind())
    }

    #[pyo3(signature = (index, *, string=0))]
    fn displacement_at(&self, py: Python<'_>, index: i64, string: usize) -> PyResult<f64> {
        self.strings
            .bind(py)
            .get_item(string)?
            .call_method1("displacement_at", (index,))?
            .extract()
    }

    /// Radiated pressure from the shared body.
    fn pressure(&self, py: Python<'_>) -> PyResult<f64> {
        self.body.bind(py).call_method0("pressure")?.extract()
    }
}

impl PySympatheticStrings {
    fn bridge_displacement(&self, py: Python<'_>, prev: bool) -> PyResult<f64> {
        let b = self.body.bind(py);
        if prev {
            np_dot(py, &b.getattr("phi")?, &b.getattr("q_prev")?)
        } else {
            b.call_method0("bridge_displacement")?.extract()
        }
    }

    fn forces(&self, py: Python<'_>) -> PyResult<Vec<f64>> {
        let w_b = self.bridge_displacement(py, false)?;
        let ks = vec1(py, self.k_springs.bind(py), "K")?;
        let strings = self.strings.bind(py);
        let mut out = Vec::with_capacity(self.j);
        for (j, kj) in ks.iter().enumerate().take(self.j) {
            let u_end: f64 = strings
                .get_item(j)?
                .getattr("u")?
                .get_item(-1i32)?
                .extract()?;
            out.push(kj * (u_end - w_b));
        }
        Ok(out)
    }

    fn apply_a(&self, py: Python<'_>, x: &[f64]) -> PyResult<Vec<f64>> {
        let b = self.body.bind(py);
        let modes = usize_attr(b, "M")?;
        let ks = vec1(py, self.k_springs.bind(py), "K")?;
        let strings = self.strings.bind(py);
        let split = x.len() - modes;
        let q: Vec<f64> = x[split..].to_vec();
        let w_b = if modes > 0 {
            np_dot(py, &b.getattr("phi")?, pyarr(py, q.clone()).bind(py))?
        } else {
            0.0
        };

        let mut out = vec![0.0f64; x.len()];
        let mut body_force = 0.0f64;
        let mut off = 0usize;
        for (j, kj) in ks.iter().enumerate().take(self.j) {
            let s = strings.get_item(j)?;
            let n = usize_attr(&s, "N")?;
            let mut out_u = string_internal(py, &s, &x[off..off + n])?;
            let eta = x[off + n - 1] - w_b;
            if let Some(last) = out_u.last_mut() {
                *last += 2.0 * kj * eta / (f64_attr(&s, "rho")? * f64_attr(&s, "h")?);
            }
            out[off..off + n].copy_from_slice(&out_u);
            body_force += kj * eta;
            off += n;
        }

        if modes > 0 {
            let phi = vec1(py, &b.getattr("phi")?, "phi")?;
            let m = vec1(py, &b.getattr("m")?, "m")?;
            let omega = vec1(py, &b.getattr("omega")?, "omega")?;
            for i in 0..modes {
                out[split + i] = (omega[i] * omega[i]) * q[i] - phi[i] * body_force / m[i];
            }
        }
        Ok(out)
    }

    fn eigenvalue(&self, py: Python<'_>) -> PyResult<f64> {
        let strings = self.strings.bind(py);
        let mut total = 0usize;
        for j in 0..self.j {
            total += usize_attr(&strings.get_item(j)?, "N")?;
        }
        let n = total + usize_attr(self.body.bind(py), "M")?;
        let mut rows = vec![vec![0.0f64; n]; n];
        let mut e = vec![0.0f64; n];
        for j in 0..n {
            e[j] = 1.0;
            for (i, v) in self.apply_a(py, &e)?.iter().enumerate() {
                rows[i][j] = *v;
            }
            e[j] = 0.0;
        }
        max_real_eigenvalue(py, &rows)
    }
}
