"""Headless-browser smoke check for the web viewer (Phase A string + Phase B membrane).

Drives a real Chrome over the DevTools protocol (no Selenium/Puppeteer needed — just the
``websocket-client`` package): launch headless Chrome, navigate to the running viewer with a
``?model=…&domain=…`` deep-link, wait for the *real* "ok" render status (not a virtual-time
guess), then (a) read the energy/spectrum readouts the page actually rendered and (b) sample the
main canvas
pixels to prove the field painted (a byte-order bug would paint background-only or uniform garbage).
A screenshot is saved per case for eyeballing.

Prereq: the server must already be running (``python web/server.py``). Run::

    python scripts/verify_web_headless.py

Exits non-zero if any case fails its assertions. This is a dev harness, not part of the pytest
suite (it needs Chrome + a live server); the data-pipeline invariants are pinned headlessly in
``tests/test_web_backend.py``.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

try:
    from websocket import create_connection
except ImportError:  # pragma: no cover - optional dev dep
    print("need websocket-client:  pip install websocket-client")
    sys.exit(2)

PORT = 9333
BASE = os.environ.get("VIEWER_BASE", "http://localhost:8000")  # override if the server moved ports
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "google-chrome",
    "chromium",
]

# JS run after each render: returns the rendered readouts + a histogram of canvas pixel
# "temperature"
# (warm = positive displacement, cool = negative, bg = exterior/at-rest), so we can assert the field
# actually painted into the interior rather than leaving a flat background.
PROBE = r"""
(function () {
  const cv = document.getElementById('string');
  const g = cv.getContext('2d');
  const d = g.getImageData(0, 0, cv.width, cv.height).data;
  let warm = 0, cool = 0, bg = 0, other = 0;
  for (let i = 0; i < d.length; i += 4) {
    const r = d[i], gg = d[i + 1], b = d[i + 2];
    if (Math.abs(r - 22) < 8 && Math.abs(gg - 27) < 8 && Math.abs(b - 34) < 8) { bg++; continue; }
    if (r > b + 10) warm++; else if (b > r + 10) cool++; else other++;
  }
  return JSON.stringify({
    status: document.getElementById('status').textContent,
    energy: document.getElementById('energy-readout').textContent,
    diag2: document.getElementById('partials-readout').textContent,
    warm: warm, cool: cool, bg: bg, other: other, total: d.length / 4,
  });
})()
"""


def _find_chrome() -> str | None:
    for c in CHROME_CANDIDATES:
        if os.path.sep in c:
            if os.path.exists(c):
                return c
        else:
            from shutil import which
            if which(c):
                return c
    return None


class CDP:
    def __init__(self, ws_url: str) -> None:
        self.ws = create_connection(ws_url, max_size=None)
        self._id = 0

    def cmd(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        mid = self._id
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == mid:
                return msg


def _evaluate(cdp: CDP, expr: str) -> object:
    r = cdp.cmd("Runtime.evaluate", {"expression": expr, "returnByValue": True})
    return r.get("result", {}).get("result", {}).get("value")


def run_case(cdp: CDP, name: str, query: str) -> bool:
    cdp.cmd("Page.navigate", {"url": f"{BASE}/?{query}"})
    status = ""
    # Up to ~90 s. It used to be ~40 s, which was ample when a worst-case membrane render was ~5 s
    # and the slowest case (the whirl) was ~25 s — but the geometric string's PHANTOM regime
    # measures 0.10 s of bridge force at fs ≈ 159 kHz: ~15,900 vector Newton steps ≈ 45 s. That
    # window is fixed physics, not a budget choice (halving it mislocates the weakest phantom by
    # 0.52 Hz), so the wait had to grow instead. NOTE: a 40 s window would time out mid-"computing…"
    # and report a FALSE FAIL — which is exactly what a loaded machine did to the plate cases once
    # before; if a case fails on "computing…", check the render time before suspecting the code.
    for _ in range(450):
        status = str(_evaluate(cdp, "document.getElementById('status').textContent") or "")
        if status.startswith("ok") or status.startswith("error") or status.startswith("network"):
            break
        time.sleep(0.2)

    probe = json.loads(str(_evaluate(cdp, PROBE)))
    shot = cdp.cmd("Page.captureScreenshot", {})
    png = shot.get("result", {}).get("data")
    if png:
        with open(os.path.join(OUT, f"viewer_{name}.png"), "wb") as fh:
            fh.write(base64.b64decode(png))

    painted = probe["warm"] + probe["cool"] + probe["other"]
    ok = probe["status"].startswith("ok") and painted > 2000
    print(f"\n=== {name} ({query}) ===")
    print(f"  status   : {probe['status']}")
    print(f"  energy   : {probe['energy'].splitlines()[0] if probe['energy'] else '(none)'}")
    print(f"  diag2    : {probe['diag2'].splitlines()[0] if probe['diag2'] else '(none)'}")
    print(f"  painted  : warm={probe['warm']} cool={probe['cool']} other={probe['other']} "
          f"bg={probe['bg']} / {probe['total']}  -> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # readouts contain λ, σ, etc.
    except Exception:
        pass
    os.makedirs(OUT, exist_ok=True)
    chrome = _find_chrome()
    if not chrome:
        print("Chrome not found; set a path in CHROME_CANDIDATES.")
        return 2
    try:
        urllib.request.urlopen(BASE, timeout=2).read(1)
    except Exception:
        print(f"server not reachable at {BASE}; start it with `python web/server.py`.")
        return 2

    tmp = tempfile.mkdtemp(prefix="chrome-verify-")
    proc = subprocess.Popen(
        [chrome, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
         f"--remote-debugging-port={PORT}", f"--user-data-dir={tmp}", "--remote-allow-origins=*",
         "--window-size=1400,1000", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        page = None
        for _ in range(60):
            try:
                targets = json.load(urllib.request.urlopen(f"http://localhost:{PORT}/json"))
                pages = [
                    t for t in targets if t.get("type") == "page" and t.get("webSocketDebuggerUrl")
                ]
                if pages:
                    page = pages[0]
                    break
            except Exception:
                pass
            time.sleep(0.25)
        if not page:
            print("could not reach Chrome DevTools endpoint.")
            return 2

        cdp = CDP(page["webSocketDebuggerUrl"])
        cdp.cmd("Page.enable")
        cdp.cmd("Runtime.enable")

        cases = [
            ("membrane_circle", "model=membrane&domain=circle"),
            ("membrane_rect", "model=membrane&domain=rectangle"),
            ("mallet_circle", "model=mallet&domain=circle"),
            ("mallet_rect", "model=mallet&domain=rectangle"),
            ("string_ideal", "model=ideal"),
            ("string_tension", "model=tension"),
            ("string_bow", "model=bow"),
            ("plate_supported", "model=plate&domain=supported"),
            ("plate_free", "model=plate&domain=free"),
            ("vk_supported", "model=vk&domain=supported"),
            ("vk_free", "model=vk&domain=free"),
            # The geometric string's four regimes. Slowest cases in the set: every step is a 3-field
            # Newton solve at ~22x a normal string's fs. The first three are viz-only (the only
            # cases with no audio, so `painted` is the whole verdict); `phantom` is the exception —
            # it measures the bridge force, so it has both a spectrum AND 0.1 s of sound, and at
            # ~45 s it is the slowest render in the viewer (see the wait loop in run_case).
            ("geom_rotating", "model=geometric&domain=rotating"),
            ("geom_planar", "model=geometric&domain=planar"),
            ("geom_whirl", "model=geometric&domain=whirl"),
            ("geom_phantom", "model=geometric&domain=phantom"),
            # Sympathetic / coupled strings: the two-string oracles. normal runs twice (antisym +
            # symmetric contrast) so its render is ~2x a single string's; transfer animates the full
            # slosh; weinreich runs twice too (strike-one + strike-both) over a lossy body.
            # All ship real audio (a string pickup), unlike the geometric string.
            ("symp_normal", "model=sympathetic&domain=normal"),
            ("symp_transfer", "model=sympathetic&domain=transfer"),
            ("symp_weinreich", "model=sympathetic&domain=weinreich"),
            # Jawari: also a two-run model (the string over its bridge, plus the clean contrast that
            # turns "bright" into a number). At the defaults ~5 s; the guards admit up to ~34 s, so
            # it sits comfortably inside the 90 s window above.
            ("jawari", "model=jawari"),
            # The fret / flat rail: model #8's OTHER configuration, and the most expensive model in
            # the viewer per second of audio (~13 s of wall clock per second of sound at N = 100).
            # Also a two-run model — the out-of-reach brightness control costs 0.95x the fret run,
            # since the string step and the rank-m correction dominate either way, not the contact
            # solve. FRET_N_MAX / FRET_AUDIO_MAX / FRET_WORK_MAX are sized so the worst passing
            # render stays inside the 90 s window; at the defaults it is ~11 s.
            ("fret", "model=fret"),
            # The acoustic bore: the first WIND model and a new field type (pressure along a tube,
            # with the two ends drawn differently — the closed end is a pressure ANTINODE). Both
            # far-end regimes are covered because they exercise different halves of drawBore: the
            # radiating case paints the flared mouth and its glow, the open case the p = 0 node.
            # The cheapest recent model — the worst passing render is ~3 s.
            ("bore_radiating", "model=bore&domain=radiating"),
            ("bore_open", "model=bore&domain=open"),
            # The dynamic reed on that same tube: the wind leg's exciter, and the third case of the
            # meta.ends switch (a "reed" mouth, drawn as a moving flap on its own scale — to scale
            # it is 2.5 % of the bore diameter). Both far ends again, because the reed's energy
            # curve differs between them: with a bell, radiated energy accumulates inside the book
            # so dE RAMPS; on an ideal open end the limit cycle flattens it. Cold renders pay the
            # fixed-N threshold sweep (~3.5 s) on top of the render, and it is memoized after.
            ("reed_radiating", "model=reed&domain=radiating"),
            ("reed_open", "model=reed&domain=open"),
            # String → modal body + radiation (batch 12): the coupling/radiation leg, shown for
            # the first time. ONE instrumented run; the Energy card is the string ⇄ body exchange
            # (not the bare conservation line) and the second panel is the radiated spectrum. Cheap
            # (fs ≈ 22 kHz, ~2 s of audio). A fresh ?model=body deep-link initialises
            # bridge_stiffness to 8k via applyModelRanges, well under the ~21.5k exact-guard
            # ceiling, so it renders "ok" (the jawari→body IN-PLACE leak, which this fresh-load
            # cannot exercise, is guarded separately by the _default reset of bridge_stiffness).
            ("body", "model=body"),
        ]
        # Optional name filters, so a single-model batch can re-check its own case without paying
        # for the whole sweep (the geometric regimes alone are ~2 minutes).
        if len(sys.argv) > 1:
            cases = [(n, q) for n, q in cases if any(a in n for a in sys.argv[1:])]
            if not cases:
                print(f"no case matches {sys.argv[1:]}")
                return 2
        results = [run_case(cdp, n, q) for n, q in cases]
        print(f"\n{sum(results)}/{len(results)} cases passed; screenshots in out/viewer_*.png")
        return 0 if all(results) else 1
    finally:
        proc.terminate()


if __name__ == "__main__":
    sys.exit(main())
