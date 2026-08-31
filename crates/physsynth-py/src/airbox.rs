//! The binding over `physsynth_core::airbox` — the 3-D room wearing the Python interface.
//!
//! # The private surface is the interface here, and it is fourteen names wide
//!
//! `airbox.py` keeps three tiers above `AirBox`: the ports (`RoomPort`, `SurfacePort`,
//! `InteriorSurfacePort`) and the six `RoomLoaded*` / `RoomSuspended*` wrappers. None of them is
//! ported, and all of them reach into the room through names the reference spells with a leading
//! underscore. Enumerated from the source and the suite before a line of this file was written:
//!
//! | name | who reads it | who **writes** it |
//! | --- | --- | --- |
//! | `_w` | `_free_pressure_nodes` | — |
//! | `_W` | ports, `test_airbox_modal`, `test_airbox_port` | — |
//! | `_beta`, `_open` | `_free_pressure_nodes`, `test_airbox_port`, `test_airbox_surface` | — |
//! | `_has_walls` | `_free_pressure_nodes`, `test_airbox_port` | — |
//! | `_pending` | `test_airbox_dipole`, `test_airbox_port`, `test_airbox_surface` | the same |
//! | `_pending_ports` | every port | every port, and `test_airbox_dipole` |
//! | `_ports` | the disjointness check | every port |
//! | `_cut_mask`, `_cut_index`, `_cuts` | `cut_faces`, `step` | `test_airbox_dipole::_uncut` |
//! | `_register_cut`, `_plane_axis` | `InteriorSurfacePort` | — |
//! | `_divergence` | `test_airbox_port`, `test_airbox_surface` | — |
//!
//! So none of the six containers can be mirrored in Rust: a client appends to `_pending_ports` and
//! a test assigns a fresh `[None, None, None]` over `_cut_mask`. They are Python objects the class
//! merely holds, and every step reads them back. That is §12.2's finding ("a leading underscore is
//! not a statement about the interface") at its widest — three phases after `body` found three
//! modules writing to a `_accel`.
//!
//! Five private names are deliberately **not** exposed, because nothing outside `airbox.py` reads
//! them and they would each cost a fabricated Python object: `_lossy` (a list of slice tuples),
//! `_momentum`, `_apply_cut`, `_normalize_walls` and `_build_wall_closure`. `_Wx` / `_Wy` / `_Wz`
//! *are* exposed even though nothing outside reads them, because they are plain arrays built once
//! and the convention for a swapped class is attribute-for-attribute compatibility.
//!
//! # The cut index is converted once per shape, not once per step
//!
//! `_momentum` zeroes the cut faces every half-step, and `_cut_index[axis]` is a Python tuple of
//! three index arrays. Converting it to flat offsets on every `step()` would put an `O(cut faces)`
//! Python round-trip inside the hot loop, so it is cached against the **identity** of the three
//! entries — with a strong reference held, so a freed object cannot have its address reused under
//! the cache. `_register_cut` rebinds the entry (a fresh tuple) and `_uncut` rebinds the whole
//! list, so both invalidate it for free.
//!
//! # Where the walls are parsed
//!
//! `_normalize_walls` accepts a token, a float, or a per-face mapping, and three of its rejections
//! quote `repr()` of the caller's own object. That cannot be done from the core crate, which has
//! no Python, so the parsing lives here and the core takes six already-classified [`Wall`]s.

use crate::shape::shape_repr;
use numpy::{PyArray1, PyArray3, PyArrayMethods, PyUntypedArrayMethods};
use physsynth_core::airbox as core;
use physsynth_core::airbox::Wall;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

/// The flat cut-face offsets for one `_cut_index`, kept alive alongside the objects they came from.
struct CutCache {
    /// A strong reference to each `_cut_index[axis]`, so the pointer comparison below is sound.
    keys: [Py<PyAny>; 3],
    /// The flat face offsets, per axis.
    flat: [Vec<usize>; 3],
}

/// A rectangular room of air on a 3-D Yee grid — the Rust implementation, wearing the Python
/// interface.
///
/// Attribute-for-attribute and method-for-method compatible with `physsynth.core.airbox.AirBox`;
/// the docstring on that class is the reference.
#[pyclass(name = "AirBox", module = "physsynth_rs")]
pub struct PyAirBox {
    params: core::Params,
    // -- state: Python-owned, rebound (not overwritten) every step ------------------------------
    pressure: Py<PyArray3<f64>>,
    u: [Py<PyArray3<f64>>; 3],
    u_prev: [Py<PyArray3<f64>>; 3],
    // -- immutable after construction ------------------------------------------------------------
    w: Py<PyAny>,
    wv: Py<PyAny>,
    wf: [Py<PyAny>; 3],
    beta: Py<PyAny>,
    open: Py<PyAny>,
    walls: Py<PyAny>,
    // -- the six containers a client writes ------------------------------------------------------
    pending: Py<PyAny>,
    pending_ports: Py<PyAny>,
    ports: Py<PyAny>,
    cut_mask: Py<PyAny>,
    cut_index: Py<PyAny>,
    cuts: Py<PyAny>,
    cut_cache: Option<CutCache>,
    /// The default injection node. It lives here rather than in `Params` because it is **written**
    /// from outside: `tests/test_airbox_freefield.py` relocates the source with a plain
    /// `box.source_index = box.node_index(centre)`. It is the fifteenth name across the seam and
    /// the only *public* one — the private-name grep that found the other fourteen could not see
    /// it, which is §29.1's "ask what a client does, not only what it reads" one door further on.
    source_index: [usize; 3],
    // -- books -----------------------------------------------------------------------------------
    dissipated: f64,
    injected: f64,
    n: usize,
}

/// Read a 3-D float64 array's contents in C order, as a copy.
///
/// Only for the read-outs that want one anyway (`state`, `pressure_at`). The hot paths — `step`,
/// `acoustic_energy`, `_divergence` — take [`borrow3`] instead: copying `p`, `ux`, `uy` and `uz`
/// into fresh `Vec`s every step is four array-sized memcpys of pure overhead, and it is *visible*.
/// Measured on a 41x33x25 room it made the Rust step **3.2x slower than NumPy's** (1,161 us against
/// 362) while every answer stayed bit-identical — plan §29.2's rule arriving in the batch that
/// cites it.
fn read3(py: Python<'_>, arr: &Py<PyArray3<f64>>) -> PyResult<Vec<f64>> {
    Ok(borrow3(py, arr)?.0.as_slice().unwrap().to_vec())
}

/// A read-only borrow of a 3-D float64 array, with the contiguity check done once.
///
/// The returned guard must outlive every use of the slice, which is why it is handed back rather
/// than dropped here.
type Borrowed<'py> = (
    numpy::PyReadonlyArray3<'py, f64>,
    std::marker::PhantomData<&'py ()>,
);

fn borrow3<'py>(py: Python<'py>, arr: &Py<PyArray3<f64>>) -> PyResult<Borrowed<'py>> {
    let ro = arr.bind(py).readonly();
    ro.as_slice()
        .map_err(|_| PyValueError::new_err("state arrays must be contiguous."))?;
    Ok((ro, std::marker::PhantomData))
}

/// A flat row-major `Vec<f64>` as a fresh 3-D NumPy array.
fn to_3d(py: Python<'_>, values: Vec<f64>, shape: [usize; 3]) -> PyResult<Py<PyArray3<f64>>> {
    Ok(PyArray1::from_vec(py, values).reshape(shape)?.unbind())
}

/// The same, handed back as a plain object (for the weight fields, which are never rebound).
fn to_3d_any(py: Python<'_>, values: Vec<f64>, shape: [usize; 3]) -> PyResult<Py<PyAny>> {
    Ok(PyArray1::from_vec(py, values)
        .reshape(shape)?
        .into_any()
        .unbind())
}

/// A flat row-major `Vec<bool>` as a fresh 3-D NumPy array.
fn to_3d_bool(py: Python<'_>, values: Vec<bool>, shape: [usize; 3]) -> PyResult<Py<PyAny>> {
    Ok(PyArray1::from_vec(py, values)
        .reshape(shape)?
        .into_any()
        .unbind())
}

/// `np.asarray(obj, dtype=np.intp)` reduced to a `Vec<i64>`.
fn as_intp(py: Python<'_>, obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<i64>> {
    let np = py.import("numpy")?;
    let arr = np.call_method1("asarray", (obj, np.getattr("int64")?))?;
    let arr = np.call_method1("ascontiguousarray", (arr,))?;
    let arr: Bound<'_, numpy::PyArrayDyn<i64>> = arr
        .cast_into()
        .map_err(|_| PyValueError::new_err(format!("{name} must be an integer index array.")))?;
    let ro = arr.readonly();
    Ok(ro
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be contiguous.")))?
        .to_vec())
}

/// `np.asarray(obj, dtype=float)` reduced to a `Vec<f64>`.
fn as_f64(py: Python<'_>, obj: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<f64>> {
    let np = py.import("numpy")?;
    let arr = np.call_method1("asarray", (obj, np.getattr("float64")?))?;
    let arr = np.call_method1("ascontiguousarray", (arr,))?;
    let arr: Bound<'_, numpy::PyArrayDyn<f64>> = arr
        .cast_into()
        .map_err(|_| PyValueError::new_err(format!("{name} must be an array of floats.")))?;
    let ro = arr.readonly();
    Ok(ro
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be contiguous.")))?
        .to_vec())
}

/// Turn a `(ix, iy, iz)` fancy-index triple into flat offsets into an array of `shape`.
fn fancy_to_flat(
    py: Python<'_>,
    nodes: &Bound<'_, PyAny>,
    shape: [usize; 3],
    name: &str,
) -> PyResult<Vec<usize>> {
    let triple: Vec<Bound<'_, PyAny>> = nodes.try_iter()?.collect::<PyResult<_>>()?;
    if triple.len() != 3 {
        return Err(PyValueError::new_err(format!(
            "{name} must be a triple of index arrays, got {} component(s).",
            triple.len()
        )));
    }
    let cols: Vec<Vec<i64>> = triple
        .iter()
        .map(|a| as_intp(py, a, name))
        .collect::<PyResult<_>>()?;
    let n = cols[0].len();
    if cols[1].len() != n || cols[2].len() != n {
        return Err(PyValueError::new_err(format!(
            "{name}'s three index arrays must have equal length."
        )));
    }
    let mut out = Vec::with_capacity(n);
    for ((&a, &b), &c) in cols[0].iter().zip(cols[1].iter()).zip(cols[2].iter()) {
        if a < 0 || b < 0 || c < 0 {
            return Err(PyValueError::new_err(format!(
                "{name} has a negative index."
            )));
        }
        let (a, b, c) = (a as usize, b as usize, c as usize);
        if a >= shape[0] || b >= shape[1] || c >= shape[2] {
            return Err(PyValueError::new_err(format!(
                "{name} indexes outside {}.",
                shape_repr(&shape)
            )));
        }
        out.push(core::flat(shape, a, b, c));
    }
    Ok(out)
}

impl PyAirBox {
    /// The flat cut-face offsets, converted from `_cut_index` at most once per rebinding.
    fn cut_flat(&mut self, py: Python<'_>) -> PyResult<[Vec<usize>; 3]> {
        let list = self.cut_index.bind(py);
        let entries: [Bound<'_, PyAny>; 3] =
            [list.get_item(0)?, list.get_item(1)?, list.get_item(2)?];
        if let Some(cache) = &self.cut_cache {
            let hit = (0..3).all(|a| cache.keys[a].bind(py).is(&entries[a]));
            if hit {
                return Ok(cache.flat.clone());
            }
        }
        let mut flat: [Vec<usize>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for (axis, item) in entries.iter().enumerate() {
            if item.is_none() {
                continue;
            }
            flat[axis] = fancy_to_flat(py, item, self.params.u_shape(axis), "_cut_index")?;
        }
        self.cut_cache = Some(CutCache {
            keys: [
                entries[0].clone().unbind(),
                entries[1].clone().unbind(),
                entries[2].clone().unbind(),
            ],
            flat: flat.clone(),
        });
        Ok(flat)
    }

    /// The queued scalar injections, as the core spells them.
    fn read_pending(&self, py: Python<'_>) -> PyResult<Vec<core::Injection>> {
        let mut out = Vec::new();
        for item in self.pending.bind(py).try_iter()? {
            let item = item?;
            let node: [usize; 3] = item.get_item(0)?.extract()?;
            let q: f64 = item.get_item(1)?.extract()?;
            out.push((node, q));
        }
        Ok(out)
    }

    /// The queued port injections: flat node offsets, per-node weights, and the volume velocity.
    #[allow(clippy::type_complexity)]
    fn read_pending_ports(&self, py: Python<'_>) -> PyResult<Vec<(Vec<usize>, Vec<f64>, f64)>> {
        let shape = self.params.p_shape();
        let mut out = Vec::new();
        for item in self.pending_ports.bind(py).try_iter()? {
            let item = item?;
            let nodes = fancy_to_flat(py, &item.get_item(0)?, shape, "_pending_ports entry")?;
            let w = as_f64(py, &item.get_item(1)?, "_pending_ports weights")?;
            let q: f64 = item.get_item(2)?.extract()?;
            out.push((nodes, w, q));
        }
        Ok(out)
    }

    /// Rebind the three velocity arrays, keeping `u_prev` as the pair `energy()` needs.
    fn commit_velocity(&mut self, py: Python<'_>, next: [Vec<f64>; 3]) -> PyResult<()> {
        let [nx, ny, nz] = next;
        let fresh = [
            to_3d(py, nx, self.params.u_shape(0))?,
            to_3d(py, ny, self.params.u_shape(1))?,
            to_3d(py, nz, self.params.u_shape(2))?,
        ];
        let old = std::mem::replace(&mut self.u, fresh);
        self.u_prev = old;
        Ok(())
    }

    /// Borrow `p` and the three velocity components at once — no copy.
    #[allow(clippy::type_complexity)]
    fn borrow_state<'py>(&self, py: Python<'py>) -> PyResult<(Borrowed<'py>, [Borrowed<'py>; 3])> {
        Ok((
            borrow3(py, &self.pressure)?,
            [
                borrow3(py, &self.u[0])?,
                borrow3(py, &self.u[1])?,
                borrow3(py, &self.u[2])?,
            ],
        ))
    }
}

/// Parse the `walls=` argument the way `_normalize_walls` does, quoting the caller's objects.
fn parse_walls(py: Python<'_>, walls: &Bound<'_, PyAny>) -> PyResult<[Wall; 6]> {
    let abc = py.import("collections.abc")?;
    let mapping = abc.getattr("Mapping")?;
    let mut spec: Vec<Py<PyAny>> = Vec::with_capacity(6);
    if walls.is_instance(&mapping)? {
        let keys: Vec<String> = walls
            .try_iter()?
            .map(|k| k?.extract())
            .collect::<PyResult<_>>()?;
        let mut unknown: Vec<String> = keys
            .iter()
            .filter(|k| !core::FACES.contains(&k.as_str()))
            .cloned()
            .collect();
        unknown.sort();
        unknown.dedup();
        if !unknown.is_empty() {
            let listed: Vec<String> = unknown.iter().map(|k| format!("'{k}'")).collect();
            return Err(PyValueError::new_err(format!(
                "unknown face name(s) [{}]; expected {}.",
                listed.join(", "),
                faces_repr()
            )));
        }
        for face in core::FACES {
            let got = walls.call_method1("get", (face, "rigid"))?;
            spec.push(got.unbind());
        }
    } else {
        for _ in core::FACES {
            spec.push(walls.clone().unbind());
        }
    }

    let mut out = [Wall::Rigid; 6];
    for (i, value) in spec.iter().enumerate() {
        let face = core::FACES[i];
        let value = value.bind(py);
        if let Ok(token) = value.extract::<String>() {
            out[i] = match token.as_str() {
                "rigid" => Wall::Rigid,
                "open" => Wall::Open,
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "wall '{face}': unknown token {}; expected 'rigid', 'open', or a float \
                         specific acoustic impedance Z (Pa*s/m).",
                        value.repr()?
                    )))
                }
            };
        } else {
            let z: f64 = value.extract()?;
            if z < 0.0 || z.is_nan() {
                return Err(PyValueError::new_err(format!(
                    "wall '{face}': impedance Z must be >= 0, got {}.",
                    physsynth_core::fmt::py_float(z)
                )));
            }
            out[i] = Wall::from_z(z);
        }
    }
    Ok(out)
}

/// `repr(FACES)`, the tuple the unknown-face message quotes.
fn faces_repr() -> String {
    let inner: Vec<String> = core::FACES.iter().map(|f| format!("'{f}'")).collect();
    format!("({})", inner.join(", "))
}

#[pymethods]
impl PyAirBox {
    #[new]
    #[pyo3(signature = (*, L, fs, h, walls = None, source = None, rho0 = core::RHO0_AIR, c0 = core::C0_AIR))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        L: &Bound<'_, PyAny>,
        fs: f64,
        h: f64,
        walls: Option<&Bound<'_, PyAny>>,
        source: Option<&Bound<'_, PyAny>>,
        rho0: f64,
        c0: f64,
    ) -> PyResult<PyAirBox> {
        // `L = tuple(float(v) for v in L)` first, so the shape rejection quotes the *converted*
        // tuple exactly as the reference does.
        let dims: Vec<f64> = L
            .try_iter()?
            .map(|v| v?.extract::<f64>())
            .collect::<PyResult<_>>()?;
        if dims.len() != 3 {
            let inner: Vec<String> = dims
                .iter()
                .map(|v| physsynth_core::fmt::py_float(*v))
                .collect();
            let got = if dims.len() == 1 {
                format!("({},)", inner[0])
            } else {
                format!("({})", inner.join(", "))
            };
            return Err(PyValueError::new_err(format!(
                "L must be a (Lx, Ly, Lz) triple, got {got}."
            )));
        }
        let l = [dims[0], dims[1], dims[2]];

        let rigid = py.None();
        let walls_obj = match walls {
            Some(w) => w.clone(),
            None => rigid.bind(py).clone(),
        };
        let parsed = if walls_obj.is_none() {
            [Wall::Rigid; 6]
        } else {
            parse_walls(py, &walls_obj)?
        };

        let src = match source {
            Some(s) if !s.is_none() => {
                let point: Vec<f64> = s
                    .try_iter()?
                    .map(|v| v?.extract::<f64>())
                    .collect::<PyResult<_>>()?;
                if point.len() != 3 {
                    return Err(PyValueError::new_err(format!(
                        "point must be an (x, y, z) triple, got {}.",
                        s.repr()?
                    )));
                }
                Some([point[0], point[1], point[2]])
            }
            _ => None,
        };

        let params = core::Params::new(l, fs, h, parsed, src, rho0, c0)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let shape = params.p_shape();
        let pressure = to_3d(py, vec![0.0; params.n_nodes()], shape)?;
        let u = [
            to_3d(py, vec![0.0; params.n_faces(0)], params.u_shape(0))?,
            to_3d(py, vec![0.0; params.n_faces(1)], params.u_shape(1))?,
            to_3d(py, vec![0.0; params.n_faces(2)], params.u_shape(2))?,
        ];
        let u_prev = [
            to_3d(py, vec![0.0; params.n_faces(0)], params.u_shape(0))?,
            to_3d(py, vec![0.0; params.n_faces(1)], params.u_shape(1))?,
            to_3d(py, vec![0.0; params.n_faces(2)], params.u_shape(2))?,
        ];

        let w = PyTuple::new(
            py,
            params
                .w
                .iter()
                .map(|axis| PyArray1::from_slice(py, axis).into_any()),
        )?
        .into_any()
        .unbind();
        let wv = to_3d_any(py, params.wv.clone(), shape)?;
        let wf = [
            to_3d_any(py, params.wf[0].clone(), params.u_shape(0))?,
            to_3d_any(py, params.wf[1].clone(), params.u_shape(1))?,
            to_3d_any(py, params.wf[2].clone(), params.u_shape(2))?,
        ];
        let beta = to_3d_any(py, params.beta.clone(), shape)?;
        let open = to_3d_bool(py, params.open.clone(), shape)?;

        let source_index = params.source_index;
        let walls_dict = PyDict::new(py);
        for (i, face) in core::FACES.iter().enumerate() {
            walls_dict.set_item(face, params.walls[i])?;
        }

        // Two SEPARATE lists: `Bound::clone` is a reference clone, so building both from one
        // `PyList` would make `_cut_mask` and `_cut_index` the same object, and `_register_cut`
        // would overwrite each with the other. (Measured: `cut_faces` came back 375 instead of
        // 136, because the mask slot was holding an index tuple.)
        Ok(PyAirBox {
            params,
            pressure,
            u,
            u_prev,
            w,
            wv,
            wf,
            beta,
            open,
            walls: walls_dict.into_any().unbind(),
            pending: PyList::empty(py).into_any().unbind(),
            pending_ports: PyList::empty(py).into_any().unbind(),
            ports: PyList::empty(py).into_any().unbind(),
            cut_mask: PyList::new(py, [py.None(), py.None(), py.None()])?
                .into_any()
                .unbind(),
            cut_index: PyList::new(py, [py.None(), py.None(), py.None()])?
                .into_any()
                .unbind(),
            cuts: PyList::empty(py).into_any().unbind(),
            cut_cache: None,
            source_index,
            dissipated: 0.0,
            injected: 0.0,
            n: 0,
        })
    }

    // -- the parameters -------------------------------------------------------------------------

    #[getter]
    #[allow(non_snake_case)]
    fn L(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyTuple::new(py, self.params.l)?.into_any().unbind())
    }

    #[getter]
    fn fs(&self) -> f64 {
        self.params.fs
    }

    #[getter]
    fn h(&self) -> f64 {
        self.params.h
    }

    #[getter]
    fn rho0(&self) -> f64 {
        self.params.rho0
    }

    #[getter]
    fn c0(&self) -> f64 {
        self.params.c0
    }

    #[getter]
    fn k(&self) -> f64 {
        self.params.k
    }

    #[getter]
    #[allow(non_snake_case)]
    fn N(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyTuple::new(py, self.params.n)?.into_any().unbind())
    }

    #[getter]
    #[allow(non_snake_case)]
    fn L_actual(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyTuple::new(py, self.params.l_actual)?.into_any().unbind())
    }

    #[getter]
    fn lam(&self) -> f64 {
        self.params.lam
    }

    #[getter]
    fn walls(&self, py: Python<'_>) -> Py<PyAny> {
        self.walls.clone_ref(py)
    }

    #[getter]
    fn source_index(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(PyTuple::new(py, self.source_index)?.into_any().unbind())
    }

    #[setter]
    fn set_source_index(&mut self, value: [i64; 3]) -> PyResult<()> {
        for (axis, &i) in value.iter().enumerate() {
            if !(0..=self.params.n[axis] as i64).contains(&i) {
                return Err(PyValueError::new_err(format!(
                    "source_index[{axis}] = {i} is outside 0..{} for this room.",
                    self.params.n[axis]
                )));
            }
        }
        self.source_index = [value[0] as usize, value[1] as usize, value[2] as usize];
        Ok(())
    }

    // -- the state ------------------------------------------------------------------------------

    #[getter]
    fn p(&self, py: Python<'_>) -> Py<PyArray3<f64>> {
        self.pressure.clone_ref(py)
    }

    #[setter]
    fn set_p(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.pressure = self.adopt(value, self.params.p_shape(), "p")?;
        Ok(())
    }

    #[getter]
    fn ux(&self, py: Python<'_>) -> Py<PyArray3<f64>> {
        self.u[0].clone_ref(py)
    }

    #[getter]
    fn uy(&self, py: Python<'_>) -> Py<PyArray3<f64>> {
        self.u[1].clone_ref(py)
    }

    #[getter]
    fn uz(&self, py: Python<'_>) -> Py<PyArray3<f64>> {
        self.u[2].clone_ref(py)
    }

    #[setter]
    fn set_ux(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.u[0] = self.adopt(value, self.params.u_shape(0), "ux")?;
        Ok(())
    }

    #[setter]
    fn set_uy(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.u[1] = self.adopt(value, self.params.u_shape(1), "uy")?;
        Ok(())
    }

    #[setter]
    fn set_uz(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.u[2] = self.adopt(value, self.params.u_shape(2), "uz")?;
        Ok(())
    }

    #[getter]
    fn ux_prev(&self, py: Python<'_>) -> Py<PyArray3<f64>> {
        self.u_prev[0].clone_ref(py)
    }

    #[getter]
    fn uy_prev(&self, py: Python<'_>) -> Py<PyArray3<f64>> {
        self.u_prev[1].clone_ref(py)
    }

    #[getter]
    fn uz_prev(&self, py: Python<'_>) -> Py<PyArray3<f64>> {
        self.u_prev[2].clone_ref(py)
    }

    #[setter]
    fn set_ux_prev(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.u_prev[0] = self.adopt(value, self.params.u_shape(0), "ux_prev")?;
        Ok(())
    }

    #[setter]
    fn set_uy_prev(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.u_prev[1] = self.adopt(value, self.params.u_shape(1), "uy_prev")?;
        Ok(())
    }

    #[setter]
    fn set_uz_prev(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.u_prev[2] = self.adopt(value, self.params.u_shape(2), "uz_prev")?;
        Ok(())
    }

    #[getter]
    fn dissipated(&self) -> f64 {
        self.dissipated
    }

    #[setter]
    fn set_dissipated(&mut self, value: f64) {
        self.dissipated = value;
    }

    #[getter]
    fn injected(&self) -> f64 {
        self.injected
    }

    #[setter]
    fn set_injected(&mut self, value: f64) {
        self.injected = value;
    }

    #[getter]
    fn n(&self) -> usize {
        self.n
    }

    #[setter]
    fn set_n(&mut self, value: usize) {
        self.n = value;
    }

    // -- the private surface --------------------------------------------------------------------

    #[getter(_w)]
    fn get_w(&self, py: Python<'_>) -> Py<PyAny> {
        self.w.clone_ref(py)
    }

    #[getter(_W)]
    fn get_wv(&self, py: Python<'_>) -> Py<PyAny> {
        self.wv.clone_ref(py)
    }

    #[getter(_Wx)]
    fn get_wx(&self, py: Python<'_>) -> Py<PyAny> {
        self.wf[0].clone_ref(py)
    }

    #[getter(_Wy)]
    fn get_wy(&self, py: Python<'_>) -> Py<PyAny> {
        self.wf[1].clone_ref(py)
    }

    #[getter(_Wz)]
    fn get_wz(&self, py: Python<'_>) -> Py<PyAny> {
        self.wf[2].clone_ref(py)
    }

    #[getter(_beta)]
    fn get_beta(&self, py: Python<'_>) -> Py<PyAny> {
        self.beta.clone_ref(py)
    }

    #[getter(_open)]
    fn get_open(&self, py: Python<'_>) -> Py<PyAny> {
        self.open.clone_ref(py)
    }

    #[getter(_has_walls)]
    fn get_has_walls(&self) -> bool {
        self.params.has_walls
    }

    #[getter(_pending)]
    fn get_pending(&self, py: Python<'_>) -> Py<PyAny> {
        self.pending.clone_ref(py)
    }

    #[setter(_pending)]
    fn set_pending(&mut self, value: Py<PyAny>) {
        self.pending = value;
    }

    #[getter(_pending_ports)]
    fn get_pending_ports(&self, py: Python<'_>) -> Py<PyAny> {
        self.pending_ports.clone_ref(py)
    }

    #[setter(_pending_ports)]
    fn set_pending_ports(&mut self, value: Py<PyAny>) {
        self.pending_ports = value;
    }

    #[getter(_ports)]
    fn get_ports(&self, py: Python<'_>) -> Py<PyAny> {
        self.ports.clone_ref(py)
    }

    #[setter(_ports)]
    fn set_ports(&mut self, value: Py<PyAny>) {
        self.ports = value;
    }

    #[getter(_cut_mask)]
    fn get_cut_mask(&self, py: Python<'_>) -> Py<PyAny> {
        self.cut_mask.clone_ref(py)
    }

    #[setter(_cut_mask)]
    fn set_cut_mask(&mut self, value: Py<PyAny>) {
        self.cut_mask = value;
    }

    #[getter(_cut_index)]
    fn get_cut_index(&self, py: Python<'_>) -> Py<PyAny> {
        self.cut_index.clone_ref(py)
    }

    #[setter(_cut_index)]
    fn set_cut_index(&mut self, value: Py<PyAny>) {
        self.cut_index = value;
        self.cut_cache = None;
    }

    #[getter(_cuts)]
    fn get_cuts(&self, py: Python<'_>) -> Py<PyAny> {
        self.cuts.clone_ref(py)
    }

    #[setter(_cuts)]
    fn set_cuts(&mut self, value: Py<PyAny>) {
        self.cuts = value;
    }

    /// Discrete divergence at every node — the transpose of the momentum gradient.
    #[pyo3(name = "_divergence")]
    fn divergence(&self, py: Python<'_>) -> PyResult<Py<PyArray3<f64>>> {
        let div = {
            let (_, u) = self.borrow_state(py)?;
            core::divergence(
                &self.params,
                u[0].0.as_slice().unwrap(),
                u[1].0.as_slice().unwrap(),
                u[2].0.as_slice().unwrap(),
            )
        };
        to_3d(py, div, self.params.p_shape())
    }

    /// The axis a plane name refers to.
    #[pyo3(name = "_plane_axis")]
    fn plane_axis(&self, plane: &str) -> PyResult<usize> {
        core::PLANES
            .iter()
            .position(|p| *p == plane)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "unknown plane '{plane}'; expected {}. An interior plane is named by its \
                     normal axis alone — it has no end, unlike a wall face {}.",
                    planes_repr(),
                    faces_repr()
                ))
            })
    }

    // -- initial conditions ---------------------------------------------------------------------

    #[pyo3(signature = (p0, u0 = None))]
    fn set_state(
        &mut self,
        py: Python<'_>,
        p0: &Bound<'_, PyAny>,
        u0: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let np = py.import("numpy")?;
        let shape = self.params.p_shape();
        let (got, mut p0v) = shaped_f64(py, p0, "p0")?;
        if got != shape.to_vec() {
            return Err(PyValueError::new_err(format!(
                "p0 must have shape {}, got {}.",
                shape_repr(&shape),
                shape_repr(&got)
            )));
        }
        for (i, &o) in self.params.open.iter().enumerate() {
            if o {
                p0v[i] = 0.0;
            }
        }

        let want: Vec<Vec<usize>> = (0..3).map(|a| self.params.u_shape(a).to_vec()).collect();
        let scalar = match u0 {
            None => Some(0.0),
            Some(v) => {
                if np.call_method1("isscalar", (v,))?.extract::<bool>()? {
                    Some(v.extract::<f64>()?)
                } else {
                    None
                }
            }
        };
        let mut prev: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        match scalar {
            Some(value) => {
                for (axis, slot) in prev.iter_mut().enumerate() {
                    *slot = vec![value; self.params.n_faces(axis)];
                }
            }
            None => {
                let parts: Vec<Bound<'_, PyAny>> =
                    u0.unwrap().try_iter()?.collect::<PyResult<_>>()?;
                let mut shapes: Vec<Vec<usize>> = Vec::new();
                let mut values: Vec<Vec<f64>> = Vec::new();
                for part in &parts {
                    let (s, v) = shaped_f64(py, part, "u0")?;
                    shapes.push(s);
                    values.push(v);
                }
                if shapes.len() != 3 || shapes != want {
                    return Err(PyValueError::new_err(format!(
                        "u0 components must have shapes {}, got {}.",
                        shape_list(&want),
                        shape_list(&shapes)
                    )));
                }
                for (axis, v) in values.into_iter().enumerate() {
                    prev[axis] = v;
                }
            }
        }

        let cuts = self.cut_flat(py)?;
        core::apply_cut(&mut prev, &cuts);
        let next = core::momentum(&self.params, &p0v, [&prev[0], &prev[1], &prev[2]], &cuts);
        self.pressure = to_3d(py, p0v, shape)?;
        for (axis, slot) in prev.iter_mut().enumerate() {
            self.u_prev[axis] = to_3d(py, std::mem::take(slot), self.params.u_shape(axis))?;
        }
        let [nx, ny, nz] = next;
        self.u = [
            to_3d(py, nx, self.params.u_shape(0))?,
            to_3d(py, ny, self.params.u_shape(1))?,
            to_3d(py, nz, self.params.u_shape(2))?,
        ];
        self.dissipated = 0.0;
        self.injected = 0.0;
        self.pending.bind(py).call_method0("clear")?;
        self.pending_ports.bind(py).call_method0("clear")?;
        for port in self.ports.bind(py).try_iter()? {
            port?.setattr("_queued_at", -1)?;
        }
        self.n = 0;
        Ok(())
    }

    // -- time stepping --------------------------------------------------------------------------

    /// Queue a soft point injection of volume velocity `q` for the next [`PyAirBox::step`].
    #[pyo3(signature = (q, at = None))]
    fn inject(&mut self, py: Python<'_>, q: f64, at: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        let index: Py<PyAny> = match at {
            Some(point) if !point.is_none() => self.node_index(py, point)?,
            _ => PyTuple::new(py, self.source_index)?.into_any().unbind(),
        };
        let entry = PyTuple::new(py, [index, q.into_pyobject(py)?.into_any().unbind()])?;
        self.pending.bind(py).call_method1("append", (entry,))?;
        Ok(())
    }

    /// Advance one timestep: pressure (plus source and walls) first, then velocity.
    ///
    /// The ordering is the bore's and it is load-bearing: afterwards the object holds `p^{n+1}`
    /// alongside `u^{n+3/2}` and `u^{n+1/2}`, so `energy()` is a pure function of the stored state.
    fn step(&mut self, py: Python<'_>) -> PyResult<()> {
        // Everything that needs `&mut self` or reaches back into Python happens first, so the
        // arithmetic below can hold four read borrows of the state arrays and copy none of them.
        let pending = self.read_pending(py)?;
        let ports = self.read_pending_ports(py)?;
        let cuts = self.cut_flat(py)?;
        let queued = !pending.is_empty() || !ports.is_empty();

        // `dissipated` is carried in and out rather than accumulated per step: the reference
        // books each lossy face onto the running total separately, and a per-step subtotal would
        // be a different association on that accumulator.
        let mut dissipated = self.dissipated;
        let (p_next, u_next, injected) = {
            let (p_guard, u_guard) = self.borrow_state(py)?;
            let p_old = p_guard.0.as_slice().unwrap();
            let u = [
                u_guard[0].0.as_slice().unwrap(),
                u_guard[1].0.as_slice().unwrap(),
                u_guard[2].0.as_slice().unwrap(),
            ];
            let div = core::divergence(&self.params, u[0], u[1], u[2]);
            let mut p_next = core::pressure_step(&self.params, p_old, &div);

            if queued {
                core::inject_scalar(&self.params, &mut p_next, &pending);
                for (nodes, w, q) in &ports {
                    core::inject_port(&self.params, &mut p_next, nodes, w, *q);
                }
            }

            if self.params.has_walls {
                core::apply_walls(&self.params, &mut p_next, p_old, &mut dissipated);
            }

            let mut injected = 0.0;
            if queued {
                for &(node, q) in &pending {
                    injected += core::booked_scalar(&self.params, &p_next, p_old, node, q);
                }
                for (nodes, w, q) in &ports {
                    injected += core::booked_port(&self.params, &p_next, p_old, nodes, w, *q);
                }
            }

            let u_next = core::momentum(&self.params, &p_next, u, &cuts);
            (p_next, u_next, injected)
        };

        self.dissipated = dissipated;
        // `injected` may be a per-step subtotal because the reference adds exactly one term per
        // queued injection, in this order, and the block above did the same additions.
        if queued {
            self.injected += injected;
            self.pending.bind(py).call_method0("clear")?;
            self.pending_ports.bind(py).call_method0("clear")?;
        }
        self.pressure = to_3d(py, p_next, self.params.p_shape())?;
        self.commit_velocity(py, u_next)?;
        self.n += 1;
        Ok(())
    }

    // -- internal boundaries: the cut -----------------------------------------------------------

    /// Add a rigid, zero-thickness internal partition on a plane of velocity **faces**.
    #[pyo3(signature = (plane, index, extent = None))]
    fn add_cut(
        &mut self,
        py: Python<'_>,
        plane: &str,
        index: &Bound<'_, PyAny>,
        extent: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let axis = self.plane_axis(plane)?;
        let n_face = self.params.n[axis] as i64;
        let idx: i64 = index
            .call_method0("__int__")
            .and_then(|v| v.extract())
            .or_else(|_| index.extract::<i64>())?;
        if !(0..=n_face - 1).contains(&idx) {
            return Err(PyValueError::new_err(format!(
                "cut index {} is out of range for plane '{plane}': the room has {n_face} face(s) \
                 there, so a cut sits at index 0..{} (face i lies between node planes i and i+1). \
                 The room's own walls are at NODE planes 0 and {n_face} and are already rigid — \
                 they are not cut positions.",
                index.str()?,
                n_face - 1
            )));
        }
        let (t0, t1) = core::other_axes(axis);
        let mut sel: Vec<Vec<i64>> = Vec::with_capacity(2);
        for (d, ax) in [t0, t1].into_iter().enumerate() {
            let n_node = self.params.n[ax] as i64;
            match extent {
                None => sel.push((0..=n_node).collect()),
                Some(e) => {
                    let bad = || {
                        PyValueError::new_err(format!(
                            "extent must be a ((lo0, hi0), (lo1, hi1)) pair of inclusive \
                             node-index ranges on the in-plane axes '{}' and '{}', got {}.",
                            core::AXES[t0],
                            core::AXES[t1],
                            e.repr()
                                .map(|r| r.to_string())
                                .unwrap_or_else(|_| "<unprintable>".to_owned())
                        ))
                    };
                    let pair = e.get_item(d).map_err(|_| bad())?;
                    let values: Vec<i64> = pair
                        .try_iter()
                        .map_err(|_| bad())?
                        .map(|v| v.and_then(|v| v.extract::<i64>()))
                        .collect::<PyResult<_>>()
                        .map_err(|_| bad())?;
                    if values.len() != 2 {
                        return Err(bad());
                    }
                    let (lo, hi) = (values[0], values[1]);
                    if !(0 <= lo && lo <= hi && hi <= n_node) {
                        return Err(PyValueError::new_err(format!(
                            "cut extent {lo}..{hi} on axis '{}' is not an inclusive node-index \
                             range inside 0..{n_node}.",
                            core::AXES[ax]
                        )));
                    }
                    sel.push((lo..=hi).collect());
                }
            }
        }
        // `np.meshgrid(sel0, sel1, indexing="ij")` then ravel: sel0 repeats, sel1 tiles.
        let mut i0 = Vec::with_capacity(sel[0].len() * sel[1].len());
        let mut i1 = Vec::with_capacity(sel[0].len() * sel[1].len());
        for &a in &sel[0] {
            for &b in &sel[1] {
                i0.push(a);
                i1.push(b);
            }
        }
        self.register_cut_inner(py, py.None(), axis, idx, &i0, &i1)
    }

    /// Cut the faces `(i0, i1)` on plane `axis` at `index`, additively — the one writer.
    #[pyo3(name = "_register_cut")]
    fn register_cut(
        &mut self,
        py: Python<'_>,
        owner: Py<PyAny>,
        axis: usize,
        index: i64,
        i0: &Bound<'_, PyAny>,
        i1: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let a = as_intp(py, i0, "i0")?;
        let b = as_intp(py, i1, "i1")?;
        self.register_cut_inner(py, owner, axis, index, &a, &b)
    }

    /// How many velocity faces are currently cut — **reported, not tuned**.
    #[getter]
    fn cut_faces(&self, py: Python<'_>) -> PyResult<usize> {
        let np = py.import("numpy")?;
        let mut total = 0usize;
        for m in self.cut_mask.bind(py).try_iter()? {
            let m = m?;
            if m.is_none() {
                continue;
            }
            total += np.call_method1("count_nonzero", (m,))?.extract::<usize>()?;
        }
        Ok(total)
    }

    // -- geometry / read-out --------------------------------------------------------------------

    /// Index of the grid node nearest `point` (m).
    fn node_index(&self, py: Python<'_>, point: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let coords: Vec<f64> = point
            .try_iter()?
            .map(|v| v?.extract::<f64>())
            .collect::<PyResult<_>>()?;
        if coords.len() != 3 {
            let inner: Vec<String> = coords
                .iter()
                .map(|v| physsynth_core::fmt::py_float(*v))
                .collect();
            let got = if coords.len() == 1 {
                format!("({},)", inner[0])
            } else {
                format!("({})", inner.join(", "))
            };
            return Err(PyValueError::new_err(format!(
                "point must be an (x, y, z) triple, got {got}."
            )));
        }
        let p = [coords[0], coords[1], coords[2]];
        match core::node_index(p, self.params.h, self.params.n) {
            Some(idx) => Ok(PyTuple::new(py, idx)?.into_any().unbind()),
            None => Err(PyValueError::new_err(
                core::ParamError::OutsideRoom {
                    point: p,
                    index: core::node_index_raw(p, self.params.h),
                    l_actual: self.params.l_actual,
                    n: self.params.n,
                }
                .to_string(),
            )),
        }
    }

    /// The node coordinate `point` actually lands on (m) — the snap, made visible.
    fn snapped(&self, py: Python<'_>, point: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let idx: [usize; 3] = self.node_index(py, point)?.extract(py)?;
        let coords: Vec<f64> = idx.iter().map(|i| *i as f64 * self.params.h).collect();
        Ok(PyTuple::new(py, coords)?.into_any().unbind())
    }

    /// Pressure (Pa) at the node nearest `point` — a microphone.
    fn pressure_at(&self, py: Python<'_>, point: &Bound<'_, PyAny>) -> PyResult<f64> {
        let idx: [usize; 3] = self.node_index(py, point)?.extract(py)?;
        let field = read3(py, &self.pressure)?;
        Ok(field[core::flat(self.params.p_shape(), idx[0], idx[1], idx[2])])
    }

    /// Current pressure field `p^n` (a copy, safe to store for plotting).
    #[getter]
    fn state(&self, py: Python<'_>) -> PyResult<Py<PyArray3<f64>>> {
        let field = read3(py, &self.pressure)?;
        to_3d(py, field, self.params.p_shape())
    }

    // -- energy ---------------------------------------------------------------------------------

    /// Energy **stored in the air** (Joules).
    fn acoustic_energy(&self, py: Python<'_>) -> PyResult<f64> {
        let (p_guard, u_guard) = self.borrow_state(py)?;
        let up = [
            borrow3(py, &self.u_prev[0])?,
            borrow3(py, &self.u_prev[1])?,
            borrow3(py, &self.u_prev[2])?,
        ];
        Ok(core::acoustic_energy(
            &self.params,
            p_guard.0.as_slice().unwrap(),
            [
                u_guard[0].0.as_slice().unwrap(),
                u_guard[1].0.as_slice().unwrap(),
                u_guard[2].0.as_slice().unwrap(),
            ],
            [
                up[0].0.as_slice().unwrap(),
                up[1].0.as_slice().unwrap(),
                up[2].0.as_slice().unwrap(),
            ],
        ))
    }

    /// Cumulative energy absorbed by the impedance walls (Joules, monotone non-decreasing).
    fn dissipated_energy(&self) -> f64 {
        self.dissipated
    }

    /// Cumulative work done on the room by the soft source (Joules).
    fn injected_energy(&self) -> f64 {
        self.injected
    }

    /// The **conserved** total `acoustic + dissipated - injected` (Joules).
    fn energy(&self, py: Python<'_>) -> PyResult<f64> {
        Ok(self.acoustic_energy(py)? + self.dissipated - self.injected)
    }

    // -- the modal oracle -----------------------------------------------------------------------

    /// The rigid-room mode `cos(l pi i/Nx) cos(m pi j/Ny) cos(n pi k/Nz)` on the grid.
    #[allow(non_snake_case)]
    fn mode_shape(&self, py: Python<'_>, l: i64, m: i64, n: i64) -> PyResult<Py<PyArray3<f64>>> {
        let idx = self.mode_indices(l, m, n)?;
        to_3d(
            py,
            core::mode_shape(&self.params, idx),
            self.params.p_shape(),
        )
    }

    /// The **exact discrete** frequency (Hz) of mode `(l, m, n)`.
    fn mode_frequency(&self, l: i64, m: i64, n: i64) -> PyResult<f64> {
        Ok(core::mode_frequency(
            &self.params,
            self.mode_indices(l, m, n)?,
        ))
    }

    /// The **textbook** rigid rectangular-room frequency (Hz).
    fn continuum_mode_frequency(&self, l: i64, m: i64, n: i64) -> PyResult<f64> {
        Ok(core::continuum_mode_frequency(
            &self.params,
            self.mode_indices(l, m, n)?,
        ))
    }

    /// Initialise the room in the exact discrete mode `(l, m, n)`; return its frequency (Hz).
    #[pyo3(signature = (l, m, n, amplitude = 1.0))]
    fn set_mode(
        &mut self,
        py: Python<'_>,
        l: i64,
        m: i64,
        n: i64,
        amplitude: f64,
    ) -> PyResult<f64> {
        let idx = self.mode_indices(l, m, n)?;
        let shape_vec: Vec<f64> = core::mode_shape(&self.params, idx)
            .into_iter()
            .map(|v| amplitude * v)
            .collect();
        let u0 = core::mode_velocity(&self.params, &shape_vec);
        let p0 = to_3d(py, shape_vec, self.params.p_shape())?;
        let parts = PyTuple::new(
            py,
            [
                to_3d(py, u0[0].clone(), self.params.u_shape(0))?.into_any(),
                to_3d(py, u0[1].clone(), self.params.u_shape(1))?.into_any(),
                to_3d(py, u0[2].clone(), self.params.u_shape(2))?.into_any(),
            ],
        )?;
        self.set_state(py, p0.bind(py).as_any(), Some(parts.as_any()))?;
        Ok(core::mode_frequency(&self.params, idx))
    }

    #[pyo3(name = "_mu_squared")]
    fn mu_squared(&self, l: i64, m: i64, n: i64) -> PyResult<f64> {
        Ok(core::mu_squared(&self.params, self.mode_indices(l, m, n)?))
    }

    #[pyo3(name = "_mode_indices")]
    fn py_mode_indices(&self, py: Python<'_>, l: i64, m: i64, n: i64) -> PyResult<Py<PyAny>> {
        let idx = self.mode_indices(l, m, n)?;
        Ok(PyTuple::new(py, idx)?.into_any().unbind())
    }
}

impl PyAirBox {
    /// Validate an array being assigned to a state attribute and take ownership of it.
    fn adopt(
        &self,
        value: &Bound<'_, PyAny>,
        shape: [usize; 3],
        name: &str,
    ) -> PyResult<Py<PyArray3<f64>>> {
        let arr: Bound<'_, PyArray3<f64>> = value.clone().cast_into().map_err(|_| {
            PyValueError::new_err(format!("{name} must be a 3-D float64 numpy array."))
        })?;
        let got = arr.shape().to_vec();
        if got != shape.to_vec() {
            return Err(PyValueError::new_err(format!(
                "{name} must have shape {}, got {}.",
                shape_repr(&shape),
                shape_repr(&got)
            )));
        }
        Ok(arr.unbind())
    }

    /// `_mode_indices`: the triple, range-checked against `N` per axis.
    fn mode_indices(&self, l: i64, m: i64, n: i64) -> PyResult<[usize; 3]> {
        let idx = [l, m, n];
        for (axis, &q) in idx.iter().enumerate() {
            let limit = self.params.n[axis] as i64;
            if !(0..=limit).contains(&q) {
                return Err(PyValueError::new_err(format!(
                    "mode index {}={q} out of range 0..{limit}.",
                    core::AXES[axis]
                )));
            }
        }
        Ok([idx[0] as usize, idx[1] as usize, idx[2] as usize])
    }

    /// The shared body of `add_cut` and `_register_cut`.
    ///
    /// `owner` is the port that owns these faces, or `None` for a hand-placed cut. Two hand-placed
    /// cuts may overlap (the mask is a boolean union); anything sharing faces with a **port**'s cut
    /// is refused, because there the cut and the `-q`/`+q` pair are two halves of one object.
    fn register_cut_inner(
        &mut self,
        py: Python<'_>,
        owner: Py<PyAny>,
        axis: usize,
        index: i64,
        i0: &[i64],
        i1: &[i64],
    ) -> PyResult<()> {
        let (t0, t1) = core::other_axes(axis);
        let shape = self.params.u_shape(axis);
        let mut flat: Vec<i64> = Vec::with_capacity(i0.len());
        for (a, b) in i0.iter().zip(i1.iter()) {
            let mut node = [0usize; 3];
            node[axis] = index as usize;
            node[t0] = *a as usize;
            node[t1] = *b as usize;
            flat.push(core::flat(shape, node[0], node[1], node[2]) as i64);
        }

        let owner_is_none = owner.bind(py).is_none();
        for entry in self.cuts.bind(py).try_iter()? {
            let entry = entry?;
            let other_owner = entry.get_item(0)?;
            let other_axis: usize = entry.get_item(1)?.extract()?;
            if other_axis != axis || (owner_is_none && other_owner.is_none()) {
                continue;
            }
            let other_flat = as_intp(py, &entry.get_item(2)?, "_cuts entry")?;
            let mut shared: Vec<i64> = flat
                .iter()
                .filter(|f| other_flat.contains(f))
                .copied()
                .collect();
            shared.sort_unstable();
            shared.dedup();
            if !shared.is_empty() {
                // `np.unravel_index` so the tuple prints exactly as the reference's does — the
                // repr of a NumPy scalar is not the repr of an int.
                let np = py.import("numpy")?;
                let face = np.call_method1("unravel_index", (shared[0], shape))?;
                return Err(PyValueError::new_err(format!(
                    "the cut on plane '{}' at index {index} shares face {face} with an existing \
                     cut ({} face(s) in common). A port's cut and its -q/+q pair are two halves \
                     of one object, so sharing faces makes the pairing ambiguous: the blocked \
                     path belongs to one plate and the injection to another, and every ledger \
                     stays green while one of them silently stops blocking.",
                    core::PLANES[axis],
                    shared.len()
                )));
            }
        }

        // The mask is the bookkeeping; the fancy index it holds is the hot path.
        let n_faces = shape[0] * shape[1] * shape[2];
        let mask_list = self.cut_mask.bind(py);
        let existing = mask_list.get_item(axis)?;
        let mut mask = if existing.is_none() {
            vec![false; n_faces]
        } else {
            let np = py.import("numpy")?;
            let arr = np.call_method1("ravel", (existing,))?;
            let flags: Vec<bool> = arr.extract()?;
            flags
        };
        for &f in &flat {
            mask[f as usize] = true;
        }
        let mask_arr = PyArray1::from_slice(py, &mask).reshape(shape)?;
        mask_list.set_item(axis, mask_arr)?;

        let mut ii: [Vec<i64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        for (f, &on) in mask.iter().enumerate() {
            if !on {
                continue;
            }
            let c = f % shape[2];
            let b = (f / shape[2]) % shape[1];
            let a = f / (shape[1] * shape[2]);
            ii[0].push(a as i64);
            ii[1].push(b as i64);
            ii[2].push(c as i64);
        }
        let np = py.import("numpy")?;
        let intp = np.getattr("intp")?;
        let idx_tuple = PyTuple::new(
            py,
            ii.iter()
                .map(|v| {
                    np.call_method1("asarray", (PyArray1::from_slice(py, v), &intp))
                        .map(|a| a.into_any())
                })
                .collect::<PyResult<Vec<_>>>()?,
        )?;
        self.cut_index.bind(py).set_item(axis, idx_tuple)?;
        self.cut_cache = None;

        let flat_arr = np.call_method1("asarray", (PyArray1::from_slice(py, &flat), &intp))?;
        let record = PyTuple::new(
            py,
            [
                owner,
                axis.into_pyobject(py)?.into_any().unbind(),
                flat_arr.unbind(),
            ],
        )?;
        self.cuts.bind(py).call_method1("append", (record,))?;

        // A cut face carries no velocity at ANY half-step, so clear the stored pair too.
        for slot in [&self.u[axis], &self.u_prev[axis]] {
            let bound = slot.bind(py);
            let mut rw = bound.readwrite();
            let s = rw
                .as_slice_mut()
                .map_err(|_| PyValueError::new_err("velocity arrays must be contiguous."))?;
            for &f in &flat {
                s[f as usize] = 0.0;
            }
        }
        Ok(())
    }
}

/// `repr(PLANES)`.
fn planes_repr() -> String {
    let inner: Vec<String> = core::PLANES.iter().map(|p| format!("'{p}'")).collect();
    format!("({})", inner.join(", "))
}

/// `np.asarray(obj, dtype=float)` with its shape, for the two `set_state` arguments.
fn shaped_f64(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<(Vec<usize>, Vec<f64>)> {
    let np = py.import("numpy")?;
    let arr = np.call_method1("asarray", (obj, np.getattr("float64")?))?;
    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    let values = as_f64(py, &arr, name)?;
    Ok((shape, values))
}

/// `str()` of a Python list of shape tuples, as the `u0` message prints it.
fn shape_list(shapes: &[Vec<usize>]) -> String {
    let inner: Vec<String> = shapes.iter().map(|s| shape_repr(s)).collect();
    format!("[{}]", inner.join(", "))
}

/// `impedance_from_zeta(zeta, rho0=RHO0_AIR, c0=C0_AIR)`.
#[pyfunction]
#[pyo3(name = "impedance_from_zeta", signature = (zeta, *, rho0 = core::RHO0_AIR, c0 = core::C0_AIR))]
pub fn py_impedance_from_zeta(zeta: f64, rho0: f64, c0: f64) -> f64 {
    core::impedance_from_zeta(zeta, rho0, c0)
}
