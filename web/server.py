"""Local HTTP shell for the web viewer — a thin transport over :func:`web.serialize`.

Run:  python web/server.py            # then open http://localhost:8000
      python web/server.py --port 9000

``ThreadingHTTPServer`` (not the single-threaded ``HTTPServer``) so a multi-second offline recompute
on one connection never blocks static-asset serving on another (advisor catch #3). This is a *local
dev tool* (architecture B): no auth, bound to localhost. No physics here — it only routes:

    GET  /                 -> static/index.html          GET /app.js, /style.css -> static assets
    POST /simulate  {json} -> serialize.simulate_to_payload(json) -> json

All the work (and all error handling) lives in the pure serializer; bad params come back as a
``{"error": {...}}`` payload with HTTP 200, never a 500 or a NaN.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Put the repo root on sys.path so `python web/server.py` can import the package without an editable
# install (mirrors the scripts/diagnose_*.py shim).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from web.serialize import simulate_to_payload  # noqa: E402

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".ico": "image/x-icon",
}
_MAX_BODY = 1 << 20  # 1 MiB request cap (params are tiny; reject anything absurd)


class Handler(BaseHTTPRequestHandler):
    """Routes static GETs and the /simulate POST. Stateless; one instance per request."""

    server_version = "PhysSynthViewer/0.1"

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj, allow_nan=False).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    # -- routing --------------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 (stdlib-mandated name)
        path = self.path.split("?", 1)[0]
        rel = "index.html" if path == "/" else path.lstrip("/")
        # Resolve under the static dir and reject any traversal outside it.
        target = os.path.normpath(os.path.join(_STATIC_DIR, rel))
        if not target.startswith(_STATIC_DIR) or not os.path.isfile(target):
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        ext = os.path.splitext(target)[1].lower()
        with open(target, "rb") as fh:
            body = fh.read()
        self._send(200, body, _CONTENT_TYPES.get(ext, "application/octet-stream"))

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] != "/simulate":
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            length = 0
        if length <= 0 or length > _MAX_BODY:
            self._send_json(
                {"error": {"kind": "request", "message": "missing/oversized body"}}, 400
            )
            return
        raw = self.rfile.read(length)
        try:
            params = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            self._send_json({"error": {"kind": "request", "message": f"bad JSON: {exc}"}}, 400)
            return
        if not isinstance(params, dict):
            self._send_json(
                {"error": {"kind": "request", "message": "body must be a JSON object"}}, 400
            )
            return
        # The serializer never raises; it returns its own {"error": ...} on bad params.
        self._send_json(simulate_to_payload(params))

    def log_message(self, fmt: str, *args) -> None:
        # One concise line per request (BaseHTTPRequestHandler's default is noisier).
        sys.stderr.write(f"  {self.address_string()} {fmt % args}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Local web viewer for the physical-synthesis core.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"physical-synthesis viewer -> {url}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
