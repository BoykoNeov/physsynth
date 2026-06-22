"""Headless-browser smoke check for the web viewer (Phase A string + Phase B membrane).

Drives a real Chrome over the DevTools protocol (no Selenium/Puppeteer needed — just the
``websocket-client`` package): launch headless Chrome, navigate to the running viewer with a
``?model=…&domain=…`` deep-link, wait for the *real* "ok" render status (not a virtual-time guess),
then (a) read the energy/spectrum readouts the page actually rendered and (b) sample the main canvas
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
BASE = "http://localhost:8000"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "google-chrome",
    "chromium",
]

# JS run after each render: returns the rendered readouts + a histogram of canvas pixel "temperature"
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
    for _ in range(200):  # up to ~40 s (a worst-case membrane render is ~5 s)
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
                pages = [t for t in targets if t.get("type") == "page" and t.get("webSocketDebuggerUrl")]
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
            ("string_ideal", "model=ideal"),
        ]
        results = [run_case(cdp, n, q) for n, q in cases]
        print(f"\n{sum(results)}/{len(results)} cases passed; screenshots in out/viewer_*.png")
        return 0 if all(results) else 1
    finally:
        proc.terminate()


if __name__ == "__main__":
    sys.exit(main())
