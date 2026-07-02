"use strict";
/* Web viewer front-end (architecture B). Talks to the local Python backend: POST /simulate -> the
 * core recomputes offline -> we animate the string (slow-mo, decoupled from audio per catch #2),
 * play the sound at its true rate (48 kHz, catch #1), and draw the energy + partials diagnostics
 * gated by loss (catch #4). Vanilla JS + Canvas2D, no framework. */

// ── element handles ─────────────────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const modelSel = $("model");
const domainSel = $("domain");
const renderBtn = $("render");
const autoRender = $("autorender");
const statusEl = $("status");
const stringCv = $("string");
const energyCv = $("energy");
const partialsCv = $("partials");
const partialsTitle = $("partials-title");
const partialsSub = $("partials-sub");
const scrub = $("scrub");
const speedInput = $("speed");
const speedVal = $("speed-val");
const animToggle = $("anim-toggle");
const playAudioBtn = $("play-audio");
const loopAudio = $("loop-audio");
const canvasMsg = $("canvas-msg");

const LABELS = {
  T: "tension T", rho: "density ρ", L: "length L", N: "grid N", lambda: "Courant λ",
  kappa: "stiffness κ", theta: "θ (time-avg)", sigma: "loss σ", sigma0: "loss σ₀ (flat)",
  sigma1: "loss σ₁ (HF)", pluck_position: "pluck pos", pickup_position: "pickup pos",
  audio_duration: "audio length", animation_window: "anim window",
  radius: "radius a", Lx: "width Lx", Ly: "height Ly",
  pluck_x: "strike x", pluck_y: "strike y", pluck_width: "strike width",
  pickup_x: "pickup x", pickup_y: "pickup y",
  mu: "plate Courant μ", fs: "sample rate fs", nu: "Poisson ν",
  E: "Young's E", e: "thickness e", w_over_e: "strike w/e",
};

// Per-model slider re-ranging (min/max/step/fixed/val) applied on model switch. The backend mirrors
// these caps. Two params carry DIFFERENT physical meaning per model and must be re-ranged + reset:
// κ (string stiffness ~1 vs plate bending speed ~20) and ρ (areal kg/m² for string/membrane/plate
// vs volumetric kg/m³ for the von Kármán plate). _default restores the string family's values so a
// switch back off plate/vk resets them. A spec's `val` is (re)applied on every model switch; params
// with no `val` in the spec persist and are only clamped into range.
const MODEL_RANGES = {
  membrane: { N: { max: 100, val: 80 }, lambda: { max: 0.7, val: 0.6 },
              audio_duration: { max: 2, val: 1.5 } },
  plate: { N: { max: 80, val: 60 }, kappa: { min: 2, max: 80, step: 0.5, fixed: 1, val: 20 },
           rho: { min: 0.001, max: 0.02, step: 0.0005, fixed: 4, val: 0.005, unit: "kg/m²" },
           Lx: { val: 1.0 }, Ly: { val: 1.0 }, audio_duration: { max: 2, val: 1 } },
  vk: { N: { min: 8, max: 32, val: 20 },
        rho: { min: 2000, max: 12000, step: 100, fixed: 0, val: 7800, unit: "kg/m³" },
        Lx: { val: 0.3 }, Ly: { val: 0.3 }, audio_duration: { max: 1, val: 0.5 } },
  _default: { N: { min: 16, max: 512 }, lambda: { max: 2.0, val: 1.0 },
              kappa: { min: 0, max: 8, step: 0.05, fixed: 2, val: 1.0 },
              rho: { min: 0.001, max: 0.02, step: 0.0005, fixed: 4, val: 0.005, unit: "kg/m²" },
              audio_duration: { max: 6, val: 2 } },
};

// Secondary select repurposed per model: geometry (membrane) vs boundary (plate / von Kármán).
const DOMAIN_MODELS = ["membrane", "plate", "vk"];
const DOMAIN_OPTS = {
  membrane: [["circle", "Circle (drumhead)"], ["rectangle", "Rectangle"]],
  plate: [["supported", "Simply-supported (#5)"], ["free", "Free edge — Chladni (#5b)"]],
  vk: [["supported", "Supported gong (#6)"], ["free", "Free-edge cymbal (#6)"]],
};

const sliders = {};      // param -> <input>
const updaters = {};     // param -> fn() that refreshes its value label
const fixedOf = {};      // param -> decimal places for the value label (re-rangeable)
const scaleOf = {};      // param -> multiplier applied in gatherParams (E in GPa, e in mm)
const unitOf = {};       // param -> value-label unit suffix (re-rangeable: ρ is areal vs volumetric)
let payload = null;
let dims = 1;            // 1 = string polyline, 2 = membrane heatmap
let frames = null, nFrames = 0, width = 0, fieldAmp = 1, animDt = 1e-3;
let gridNx = 0, gridNy = 0, maskData = null, gridMeta = null, heatCv = null;
let audioSamples = null, audioFs = 48000, audioBuf = null, audioCtx = null, audioSrc = null;
let speed = 0.02, animPlaying = true, scrubbing = false, currentFrame = 0, animStart = 0;
let autoTimer = null;

// ── slider construction ─────────────────────────────────────────────────────────────────────
function guessFixed(step) {
  if (step >= 1) return 0;
  if (step >= 0.1) return 1;
  if (step >= 0.01) return 2;
  if (step >= 0.001) return 3;
  return 4;
}

function buildSliders() {
  document.querySelectorAll(".slider").forEach((el) => {
    const d = el.dataset;
    fixedOf[d.param] = d.fixed !== undefined ? +d.fixed : guessFixed(+d.step);
    scaleOf[d.param] = d.scale !== undefined ? +d.scale : 1;
    unitOf[d.param] = d.unit || "";
    el.innerHTML =
      `<div class="row"><span class="name">${LABELS[d.param] || d.param}</span>` +
      `<span class="value" id="v-${d.param}"></span></div>` +
      `<input type="range" id="s-${d.param}" min="${d.min}" max="${d.max}" ` +
      `step="${d.step}" value="${d.val}">`;
    const input = el.querySelector("input");
    const valEl = el.querySelector(".value");
    const update = () => {
      const u = unitOf[d.param] ? " " + unitOf[d.param] : "";
      valEl.textContent = (+input.value).toFixed(fixedOf[d.param]) + u;
    };
    input.addEventListener("input", () => { update(); onControlChange(d.param); });
    update();
    sliders[d.param] = input;
    updaters[d.param] = update;
  });
}

function param(name) { return sliders[name] ? +sliders[name].value : undefined; }

function setSlider(name, val) {
  if (!sliders[name]) return;
  sliders[name].value = val;
  if (updaters[name]) updaters[name]();
}

// ── model-dependent visibility, ranges + hints ───────────────────────────────────────────────
// Repopulate the secondary select for the current model (geometry for the membrane, boundary for
// the plate family). Preserves the current value if still valid; sets the label text.
function populateDomain(model) {
  const opts = DOMAIN_OPTS[model];
  if (!opts) return;                       // hidden for string models; leave as-is
  const prev = domainSel.value;
  domainSel.innerHTML = opts.map(([v, l]) => `<option value="${v}">${l}</option>`).join("");
  if (opts.some(([v]) => v === prev)) domainSel.value = prev;
  const lbl = $("domain-label");
  if (lbl) lbl.textContent = model === "membrane" ? "Domain" : "Boundary";
}

function updateVisibility() {
  const m = modelSel.value;
  const usesDomain = DOMAIN_MODELS.includes(m);
  const d = usesDomain ? domainSel.value : "";
  document.querySelectorAll("[data-show]").forEach((el) => {
    el.hidden = !el.dataset.show.split(" ").includes(m);
  });
  // data-domain sliders (radius / Lx / Ly) are gated by the secondary select; the multi-value list
  // lets Lx/Ly show for both the membrane rectangle AND either plate boundary. Model gating is by
  // the parent fieldset's data-show, so here we only test the domain membership.
  document.querySelectorAll("[data-domain]").forEach((el) => {
    el.hidden = !usesDomain || !el.dataset.domain.split(" ").includes(d);
  });
}

// Re-range sliders to the current model's caps/defaults (see MODEL_RANGES). Merges _default with the
// model spec (model wins); applies min/max/step/fixed, (re)sets `val` when the spec gives one, and
// always clamps the current value into range. Run on model switch only, so resets are intentional.
function applyModelRanges() {
  const spec = Object.assign({}, MODEL_RANGES._default, MODEL_RANGES[modelSel.value] || {});
  for (const pkey in spec) {
    const inp = sliders[pkey];
    if (!inp) continue;
    const s = spec[pkey];
    if (s.min !== undefined) inp.min = String(s.min);
    if (s.max !== undefined) inp.max = String(s.max);
    if (s.step !== undefined) inp.step = String(s.step);
    if (s.fixed !== undefined) fixedOf[pkey] = s.fixed;
    if (s.unit !== undefined) unitOf[pkey] = s.unit;
    if (s.val !== undefined) inp.value = String(s.val);
    const lo = +inp.min, hi = +inp.max;
    if (+inp.value > hi) inp.value = String(hi);
    if (+inp.value < lo) inp.value = String(lo);
    if (updaters[pkey]) updaters[pkey]();
  }
}

function updateLambdaHint() {
  const m = modelSel.value;
  const hint = $("lambda-hint");
  if (m === "plate") {
    const mu = param("mu");
    hint.textContent = `μ = κ·k/h² = ${mu.toFixed(2)}  (implicit → no CFL; large μ is coarse-but-`
      + `stable, cost rises at LOW μ)`;
    hint.style.color = "var(--muted)";
  } else if (m === "vk") {
    const fs = param("fs");
    hint.textContent = `fs = ${Math.round(fs)} Hz  (oversample the nonlinearity: higher = truer, `
      + `more cost; κ is derived from E, e, ν, ρ)`;
    hint.style.color = "var(--muted)";
  } else if (m === "membrane") {
    const lam = param("lambda");
    hint.textContent = `λ = c·k/h = ${lam.toFixed(2)}  (2D CFL: λ ≤ 1/√2 ≈ 0.71; no λ is `
      + `dispersionless)`;
    hint.style.color = lam > 0.708 ? "var(--bad)" : "var(--muted)";
  } else if (m === "ideal" && param("lambda") > 1.0) {
    hint.textContent = "λ>1 breaks the explicit ideal string's CFL (will error). Stiff/damped allow"
      + " it.";
    hint.style.color = "var(--bad)";
  } else {
    const lam = param("lambda");
    hint.textContent = m === "ideal"
      ? `λ = c·k/h = ${lam.toFixed(2)}  (1.0 = exact, dispersionless)`
      : `λ = ${lam.toFixed(2)}  (implicit scheme — no CFL limit)`;
    hint.style.color = "var(--muted)";
  }
  // von Kármán: on the supported gong the strike is the (1,1) eigenmode (a clean glide), so the
  // strike-position sliders are ignored; the free cymbal takes a positioned crash.
  const vkHint = $("vk-strike-hint");
  if (vkHint) {
    vkHint.textContent = (m === "vk" && domainSel.value === "supported")
      ? "supported gong: struck in its (1,1) mode — strike x/y/width are ignored (pickup still used)"
      : (m === "vk" ? "free cymbal: a positioned crash (multi-mode wash)" : "");
  }
}

function onControlChange(name) {
  if (name === "lambda" || name === "mu" || name === "fs") updateLambdaHint();
  scheduleAuto();
}

modelSel.addEventListener("change", () => {
  populateDomain(modelSel.value);
  applyModelRanges();
  updateVisibility();
  updateLambdaHint();
  scheduleAuto();
});
if (domainSel) domainSel.addEventListener("change", () => {
  updateVisibility();
  updateLambdaHint();
  scheduleAuto();
});
const nonlinearChk = $("nonlinear");
if (nonlinearChk) nonlinearChk.addEventListener("change", scheduleAuto);

function scheduleAuto() {
  if (!autoRender.checked) return;
  clearTimeout(autoTimer);
  setStatus("queued…", "busy");
  autoTimer = setTimeout(render, 400);
}

// ── networking ──────────────────────────────────────────────────────────────────────────────
function gatherParams() {
  const p = { model: modelSel.value };
  if (domainSel) p.domain = domainSel.value;
  for (const k in sliders) p[k] = +sliders[k].value * (scaleOf[k] || 1);
  if (nonlinearChk) p.nonlinear = nonlinearChk.checked;
  return p;
}

function setStatus(text, cls) {
  statusEl.textContent = text;
  statusEl.className = "status" + (cls ? " " + cls : "");
}

async function render() {
  clearTimeout(autoTimer);
  setStatus("computing…", "busy");
  renderBtn.disabled = true;
  try {
    const resp = await fetch("/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(gatherParams()),
    });
    const data = await resp.json();
    if (data.error) {
      setStatus("error: " + data.error.message, "error");
      showOverlay("⚠ " + data.error.message);
      return;
    }
    applyPayload(data);
    const scheme = data.lambda !== undefined ? `λ ${data.lambda}`
      : data.mu !== undefined ? `μ ${data.mu}`
      : (data.nonlinear === false ? "linear" : "nonlinear");
    setStatus(
      `ok — fs_sim ${fmt(data.fs_sim)} Hz · ${scheme} · ` +
      `${data.frames.n_frames} frames · ${data.audio.n} audio samples`, "");
  } catch (err) {
    setStatus("network error: " + err, "error");
  } finally {
    renderBtn.disabled = false;
  }
}

// ── payload handling ────────────────────────────────────────────────────────────────────────
function b64ToFloat32(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

function b64ToUint8(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

function applyPayload(data) {
  payload = data;
  dims = (data.frames && data.frames.dims) || 1;
  frames = b64ToFloat32(data.frames.b64);
  nFrames = data.frames.n_frames;
  width = data.frames.width;
  fieldAmp = data.field_amp || 1;
  animDt = data.anim_dt || 1e-3;
  if (dims === 2) {
    gridNx = data.frames.nx; gridNy = data.frames.ny;
    maskData = data.mask ? b64ToUint8(data.mask.b64) : null;
    gridMeta = data.grid;
    heatCv = document.createElement("canvas");   // offscreen field-resolution buffer
    heatCv.width = gridNx; heatCv.height = gridNy;
  }
  audioSamples = b64ToFloat32(data.audio.b64);
  audioFs = data.audio.fs;
  audioBuf = null;                       // rebuilt lazily on first Play
  currentFrame = 0; animStart = 0;
  scrub.max = Math.max(0, nFrames - 1);
  scrub.value = 0;
  hideOverlay();
  drawEnergy();
  drawDiagnostics();
}

// ── string animation ────────────────────────────────────────────────────────────────────────
function drawString(idx) {
  const g = stringCv.getContext("2d");
  const W = stringCv.width, H = stringCv.height, midY = H / 2, margin = 22;
  g.clearRect(0, 0, W, H);
  g.strokeStyle = "#2a3340"; g.lineWidth = 1;
  g.beginPath(); g.moveTo(0, midY); g.lineTo(W, midY); g.stroke();
  if (!frames || nFrames === 0) return;

  const amp = fieldAmp > 0 ? fieldAmp : 1;
  const sx = (W - 2 * margin) / (width - 1);
  const sy = (H / 2 - margin) / amp * 0.92;
  const base = idx * width;

  // pickup marker
  if (payload) {
    const px = margin + Math.round(param("pickup_position") * (width - 1)) * sx;
    g.strokeStyle = "rgba(255,207,92,.35)"; g.setLineDash([4, 4]); g.lineWidth = 1;
    g.beginPath(); g.moveTo(px, 8); g.lineTo(px, H - 8); g.stroke(); g.setLineDash([]);
  }

  g.strokeStyle = "#4cc2ff"; g.lineWidth = 2.5; g.lineJoin = "round";
  g.beginPath();
  for (let i = 0; i < width; i++) {
    const x = margin + i * sx;
    const y = midY - frames[base + i] * sy;
    if (i === 0) g.moveTo(x, y); else g.lineTo(x, y);
  }
  g.stroke();

  // endpoints
  g.fillStyle = "#8b98a8";
  g.beginPath(); g.arc(margin, midY, 3.5, 0, 7); g.fill();
  g.beginPath(); g.arc(W - margin, midY, 3.5, 0, 7); g.fill();
}

function tick(ts) {
  if (frames && nFrames > 0) {
    if (animPlaying && !scrubbing) {
      if (animStart === 0) animStart = ts - (currentFrame * animDt / speed) * 1000;
      const physElapsed = ((ts - animStart) / 1000) * speed;
      currentFrame = Math.floor(physElapsed / animDt) % nFrames;
      scrub.value = currentFrame;
    }
    (dims === 2 ? drawHeatmap : drawString)(currentFrame);
  }
  requestAnimationFrame(tick);
}

// ── membrane heatmap ─────────────────────────────────────────────────────────────────────────
// Diverging colormap centred at 0: cool (cyan/blue) for displacement < 0, warm (orange/red) for
// > 0, near-black at rest. t is signed displacement / field_amp, clamped to [-1, 1].
function divColor(t) {
  const a = Math.min(1, Math.abs(t));
  if (t >= 0) return [20 + a * 235, 24 + a * 96, 30 + a * 6];     // dark → orange/red
  return [20 + a * 24, 24 + a * 162, 30 + a * 225];               // dark → cyan/blue
}

function drawHeatmap(idx) {
  const g = stringCv.getContext("2d");
  const W = stringCv.width, H = stringCv.height;
  g.clearRect(0, 0, W, H);
  if (!frames || nFrames === 0 || !heatCv) return;

  // Paint the decimated field into the offscreen buffer (one device pixel per node), masking
  // the exterior to the panel background so the domain shape (incl. the staircased rim) reads.
  const hctx = heatCv.getContext("2d");
  const img = hctx.createImageData(gridNx, gridNy);
  const amp = fieldAmp > 0 ? fieldAmp : 1;
  const base = idx * gridNx * gridNy;
  for (let p = 0; p < gridNx * gridNy; p++) {
    const o = p * 4;
    if (maskData && maskData[p] === 0) {            // outside the membrane
      img.data[o] = 22; img.data[o + 1] = 27; img.data[o + 2] = 34; img.data[o + 3] = 255;
      continue;
    }
    const c = divColor(frames[base + p] / amp);
    img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2]; img.data[o + 3] = 255;
  }
  hctx.putImageData(img, 0, 0);

  // Blit to the main canvas preserving the physical aspect ratio (snapped Ly for a rectangle).
  const extX = (gridMeta && gridMeta.extent_x) || 1, extY = (gridMeta && gridMeta.extent_y) || 1;
  const pad = 14, availW = W - 2 * pad, availH = H - 2 * pad;
  const scale = Math.min(availW / extX, availH / extY);
  const dw = extX * scale, dh = extY * scale;
  const dx = (W - dw) / 2, dy = (H - dh) / 2;
  g.imageSmoothingEnabled = true;
  g.drawImage(heatCv, dx, dy, dw, dh);

  // Pickup marker (x, y) in domain fractions → screen.
  const pkx = param("pickup_x"), pky = param("pickup_y");
  if (pkx !== undefined && pky !== undefined) {
    const mx = dx + pkx * dw, my = dy + pky * dh;
    g.strokeStyle = "rgba(255,207,92,.9)"; g.lineWidth = 1.5;
    g.beginPath(); g.arc(mx, my, 5, 0, 7); g.stroke();
    g.beginPath(); g.moveTo(mx - 8, my); g.lineTo(mx + 8, my);
    g.moveTo(mx, my - 8); g.lineTo(mx, my + 8); g.stroke();
  }
}

// ── energy diagnostic ───────────────────────────────────────────────────────────────────────
function drawEnergy() {
  const g = energyCv.getContext("2d");
  const W = energyCv.width, H = energyCv.height, pad = 24;
  g.clearRect(0, 0, W, H);
  if (!payload) return;
  const e = payload.energy;
  const t = e.time, v = e.value.map((x) => (x == null ? 0 : x));
  const tmax = t[t.length - 1] || 1;
  const vmax = Math.max(...v) || 1;

  g.strokeStyle = "#2a3340"; g.lineWidth = 1;
  g.strokeRect(pad, 8, W - pad - 8, H - pad - 8);
  g.strokeStyle = "#5ad17a"; g.lineWidth = 2; g.beginPath();
  for (let i = 0; i < v.length; i++) {
    const x = pad + (t[i] / tmax) * (W - pad - 8);
    const y = (H - pad) - (v[i] / vmax) * (H - pad - 8);
    if (i === 0) g.moveTo(x, y); else g.lineTo(x, y);
  }
  g.stroke();
  g.fillStyle = "#8b98a8"; g.font = "10px ui-monospace, monospace";
  g.fillText("E", 6, 16); g.fillText(`${tmax.toFixed(2)} s`, W - 48, H - 8);

  const badge = $("energy-verdict"), out = $("energy-readout");
  // von Kármán convergence gate (catch): the energy identity telescopes ONLY at the Picard fixed
  // point, so a run with any non-converged step has a drift number that is iteration noise, not
  // physics. Overrides the pass/fail verdict in BOTH the lossless and lossy branches.
  const conv = e.convergence;
  if (conv && !conv.all_converged) {
    badge.textContent = "NOT CONVERGED";
    badge.className = "badge bad";
    out.textContent =
      `Picard did not converge: ${conv.n_not_converged} step(s), worst residual ` +
      `${conv.worst_residual.toExponential(1)} > tol ${conv.couple_tol.toExponential(0)}\n` +
      `energy verdict N/A — lower the strike amplitude (w/e) or raise fs`;
    return;
  }
  const convNote = conv ? `  ·  Picard converged (≤ ${conv.max_iters} sweeps)` : "";
  if (e.sigma_is_zero) {
    const ok = e.lossless.pass;
    badge.textContent = ok ? "conserved" : "DRIFT";
    badge.className = "badge " + (ok ? "good" : "bad");
    out.textContent =
      `lossless · drift max|Eⁿ−E⁰|/E⁰ = ${e.lossless.drift.toExponential(2)}${convNote}\n` +
      `tol ${e.lossless.tol.toExponential(0)}  →  ${ok ? "PASS ✓" : "FAIL ✗"}`;
  } else {
    const mono = e.lossy.monotone;
    badge.textContent = mono ? "passive" : "NON-MONOTONE";
    badge.className = "badge " + (mono ? "good" : "bad");
    const meas = e.lossy.measured_2sigma;
    out.textContent =
      `lossy · energy monotone decrease: ${mono ? "yes ✓" : "NO ✗"}${convNote}\n` +
      `measured 2σ = ${meas == null ? "—" : meas.toFixed(3)} s⁻¹` +
      `  (flat-loss oracle ${e.lossy.oracle_2sigma.toFixed(3)})`;
  }
}

// ── second diagnostic panel: partials (string) | mode spectrum (2D) ──────────────────────────
function drawDiagnostics() {
  if (dims !== 2) {
    partialsTitle.firstChild.textContent = "Partials ";
    partialsSub.textContent = "detected vs analytic";
    drawPartials();
    return;
  }
  const sp = payload && payload.meta && payload.meta.spectrum;
  const kind = sp && sp.kind;
  if (kind === "vk") {
    partialsTitle.firstChild.textContent = "Spectrum (nonlinear) ";
    partialsSub.textContent = "FFT vs linear modes + hardening";
    drawVkSpectrum();
  } else {
    partialsTitle.firstChild.textContent = "Mode spectrum ";
    partialsSub.textContent = kind === "plate" ? "FFT vs plate eigenmodes"
      : "FFT vs discrete eigenmodes";
    drawSpectrum();
  }
}

// Membrane: pickup magnitude spectrum with vertical markers at the discrete eigenfreqs (where the
// time-stepper actually rings — peaks landing on these = self-consistency) and fainter markers at
// the continuum oracle (the O(h) staircase offset; shown, NOT scored). Cf. advisor review 3.
function drawSpectrum() {
  const g = partialsCv.getContext("2d");
  const W = partialsCv.width, H = partialsCv.height, padL = 26, padB = 16, top = 8;
  g.clearRect(0, 0, W, H);
  const out = $("partials-readout");
  const sp = payload && payload.meta && payload.meta.spectrum;
  if (!sp) { out.textContent = "no spectrum"; return; }

  const plotW = W - padL - 8, plotH = H - padB - top;
  const x0 = padL, y0 = top + plotH;
  const fmax = sp.fmax || (sp.freq[sp.freq.length - 1] || 1);
  const fx = (f) => x0 + (f / fmax) * plotW;

  g.strokeStyle = "#2a3340"; g.lineWidth = 1; g.strokeRect(x0, top, plotW, plotH);

  // continuum (Bessel / rectangular) markers — faint, the geometry-tier reference
  g.strokeStyle = "rgba(139,152,168,.35)"; g.lineWidth = 1; g.setLineDash([3, 3]);
  (sp.modes_continuum || []).forEach((f) => {
    if (f == null || f > fmax) return;
    g.beginPath(); g.moveTo(fx(f), top); g.lineTo(fx(f), y0); g.stroke();
  });
  g.setLineDash([]);
  // discrete eigenmode markers — the honest lines the FFT peaks should land on
  g.strokeStyle = "rgba(76,194,255,.55)"; g.lineWidth = 1;
  (sp.modes_discrete || []).forEach((f) => {
    if (f == null || f > fmax) return;
    g.beginPath(); g.moveTo(fx(f), top); g.lineTo(fx(f), y0); g.stroke();
  });

  // FFT magnitude (normalized 0..1)
  g.strokeStyle = "#5ad17a"; g.lineWidth = 1.5; g.beginPath();
  for (let i = 0; i < sp.freq.length; i++) {
    const m = sp.mag[i] == null ? 0 : sp.mag[i];
    const x = fx(sp.freq[i]), y = y0 - m * (plotH - 4);
    if (i === 0) g.moveTo(x, y); else g.lineTo(x, y);
  }
  g.stroke();

  g.fillStyle = "#8b98a8"; g.font = "10px ui-monospace, monospace";
  g.fillText("|X(f)|", 3, top + 10);
  g.fillText(`${Math.round(fmax)} Hz`, W - 54, H - 4);

  const cf = sp.cents_fundamental, cg = sp.cents_geometry;
  // The geometry-tier gap means different things per model: the membrane's O(h) staircase (~9
  // cents by design), the SS plate's tight Navier tier (~1 cent), the free plate's Leissa anchor.
  const tierLabel = sp.kind === "plate" ? "geometry tier (continuum/Leissa)"
    : "geometry tier (O(h) staircase)";
  out.textContent =
    `f₁ = ${sp.f1_discrete.toFixed(2)} Hz (discrete)   peaks on blue lines = self-consistent\n` +
    `fundamental detected vs discrete: ${cf == null ? "—" : cf.toFixed(3) + " cents"}` +
    `   ·   ${tierLabel}: ${cg == null ? "—" : cg.toFixed(2) + " cents"}`;
}

// von Kármán: the marker lines are the LINEAR (w→0) modes; the real peaks sit ABOVE them by the
// amplitude hardening. So this reads the gap as a *hardening shift*, never a cents error. On the
// free cymbal the fundamental is a mode wash (no clean f0) — report the cascade, not a percentage.
function drawVkSpectrum() {
  const g = partialsCv.getContext("2d");
  const W = partialsCv.width, H = partialsCv.height, padL = 26, padB = 16, top = 8;
  g.clearRect(0, 0, W, H);
  const out = $("partials-readout");
  const sp = payload && payload.meta && payload.meta.spectrum;
  if (!sp) { out.textContent = "no spectrum"; return; }

  const plotW = W - padL - 8, plotH = H - padB - top;
  const x0 = padL, y0 = top + plotH;
  const fmax = sp.fmax || (sp.freq[sp.freq.length - 1] || 1);
  const fx = (f) => x0 + (f / fmax) * plotW;

  g.strokeStyle = "#2a3340"; g.lineWidth = 1; g.strokeRect(x0, top, plotW, plotH);

  // linear (w→0) eigenmode markers — the peaks harden ABOVE these, so they are a reference floor
  g.strokeStyle = "rgba(139,152,168,.5)"; g.lineWidth = 1; g.setLineDash([3, 3]);
  (sp.modes_linear || []).forEach((f) => {
    if (f == null || f > fmax) return;
    g.beginPath(); g.moveTo(fx(f), top); g.lineTo(fx(f), y0); g.stroke();
  });
  g.setLineDash([]);
  // the detected hardened fundamental (supported gong only)
  if (sp.f0_detected != null && sp.f0_detected <= fmax) {
    g.strokeStyle = "#ffcf5c"; g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(fx(sp.f0_detected), top); g.lineTo(fx(sp.f0_detected), y0); g.stroke();
  }

  // FFT magnitude (normalized 0..1)
  g.strokeStyle = "#5ad17a"; g.lineWidth = 1.5; g.beginPath();
  for (let i = 0; i < sp.freq.length; i++) {
    const m = sp.mag[i] == null ? 0 : sp.mag[i];
    const x = fx(sp.freq[i]), y = y0 - m * (plotH - 4);
    if (i === 0) g.moveTo(x, y); else g.lineTo(x, y);
  }
  g.stroke();

  g.fillStyle = "#8b98a8"; g.font = "10px ui-monospace, monospace";
  g.fillText("|X(f)|", 3, top + 10);
  g.fillText(`${Math.round(fmax)} Hz`, W - 54, H - 4);

  if (sp.shift_pct != null) {
    out.textContent =
      `linear f₁ = ${sp.f1_linear.toFixed(1)} Hz (grey)   ·   hardened f₀ = `
      + `${sp.f0_detected.toFixed(1)} Hz (yellow)\n`
      + `amplitude hardening: ${sp.shift_pct >= 0 ? "+" : ""}${sp.shift_pct.toFixed(1)}% — the peak `
      + `rides ABOVE the linear mode (not a cents error)`;
  } else {
    out.textContent =
      `linear modes (grey) at ${sp.f1_linear.toFixed(1)} Hz and up\n`
      + `free-edge crash: a multi-mode wash — no single hardened fundamental to report`;
  }
}

// ── partials diagnostic ─────────────────────────────────────────────────────────────────────
function drawPartials() {
  const g = partialsCv.getContext("2d");
  const W = partialsCv.width, H = partialsCv.height, pad = 24, mid = H / 2;
  g.clearRect(0, 0, W, H);
  const out = $("partials-readout");
  const pa = payload && payload.meta.partials;
  if (!pa) { out.textContent = "no partials in band"; return; }

  const cents = pa.cents;
  let worst = 0;
  cents.forEach((c) => { if (c != null && Math.abs(c) > worst) worst = Math.abs(c); });
  const scale = Math.max(worst, 1);

  g.strokeStyle = "#2a3340"; g.lineWidth = 1;
  g.beginPath(); g.moveTo(pad, mid); g.lineTo(W - 8, mid); g.stroke();   // 0-cent line
  const n = cents.length, bw = (W - pad - 16) / n;
  for (let i = 0; i < n; i++) {
    const c = cents[i];
    if (c == null) continue;
    const h = (c / scale) * (H / 2 - 14);
    const x = pad + i * bw + 2;
    g.fillStyle = Math.abs(c) < 1 ? "#4cc2ff" : "#ffcf5c";
    g.fillRect(x, mid - Math.max(h, 0), bw - 4, Math.abs(h) || 1);
    if (h < 0) g.fillRect(x, mid, bw - 4, -h);
  }
  g.fillStyle = "#8b98a8"; g.font = "10px ui-monospace, monospace";
  g.fillText("+cents", 2, 14); g.fillText("−cents", 2, H - 6);

  out.textContent =
    `f₁ = ${payload.meta.f1.toFixed(2)} Hz   ·   partials shown: ${n}\n` +
    `worst error = ${worst.toFixed(3)} cents`;
}

// ── audio ───────────────────────────────────────────────────────────────────────────────────
function playAudio() {
  if (!audioSamples) return;
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === "suspended") audioCtx.resume();
  if (!audioBuf) {
    audioBuf = audioCtx.createBuffer(1, audioSamples.length, audioFs);
    audioBuf.copyToChannel(audioSamples, 0);
  }
  if (audioSrc) { try { audioSrc.stop(); } catch (e) { /* already stopped */ } }
  audioSrc = audioCtx.createBufferSource();
  audioSrc.buffer = audioBuf;
  audioSrc.loop = loopAudio.checked;
  audioSrc.connect(audioCtx.destination);
  audioSrc.start();
}

// ── overlay + misc ──────────────────────────────────────────────────────────────────────────
function showOverlay(msg) { canvasMsg.textContent = msg; canvasMsg.hidden = false; }
function hideOverlay() { canvasMsg.hidden = true; }
function fmt(x) { return Math.round(x).toLocaleString(); }

// ── transport wiring ────────────────────────────────────────────────────────────────────────
animToggle.addEventListener("click", () => {
  animPlaying = !animPlaying;
  animToggle.textContent = animPlaying ? "⏸ Pause" : "▶ Play";
  if (animPlaying) animStart = 0;     // re-anchor on resume
});
speedInput.addEventListener("input", () => {
  speed = +speedInput.value;
  speedVal.textContent = speed.toFixed(3) + "×";
  animStart = 0;                       // re-anchor so the new speed is continuous
});
scrub.addEventListener("input", () => { scrubbing = true; currentFrame = +scrub.value; });
scrub.addEventListener("change", () => { scrubbing = false; animStart = 0; });
playAudioBtn.addEventListener("click", playAudio);
renderBtn.addEventListener("click", render);

// ── boot ────────────────────────────────────────────────────────────────────────────────────
// Optional deep-link: ?model=membrane&domain=circle preselects the model/domain before the first
// render (also what the headless-browser verification drives).
function applyUrlParams() {
  const q = new URLSearchParams(location.search);
  const m = q.get("model");
  if (m && [...modelSel.options].some((o) => o.value === m)) modelSel.value = m;
  populateDomain(modelSel.value);          // options depend on the (possibly just-set) model
  const d = q.get("domain");
  if (d && domainSel && [...domainSel.options].some((o) => o.value === d)) domainSel.value = d;
}

buildSliders();
applyUrlParams();
applyModelRanges();
updateVisibility();
updateLambdaHint();
speedVal.textContent = speed.toFixed(3) + "×";
requestAnimationFrame(tick);
render();                              // auto-render the defaults on load
