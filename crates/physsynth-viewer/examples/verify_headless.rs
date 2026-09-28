//! Headless-browser check of the web viewer: what the page RENDERED, not what the server sent.
//!
//! The payload tests prove the data is right; this proves the page drew it. It drives a real Chrome
//! over the DevTools protocol, loads the running viewer with a `?model=…&domain=…` deep link per
//! case, waits for the page's own "ok" status, then reads the rendered read-outs and samples the
//! main canvas's pixels (a byte-order bug paints background only, or uniform garbage). A screenshot
//! is saved per case.
//!
//! The server must already be running (`cargo run --release -p physsynth-viewer -- serve`). Then:
//!
//! ```text
//! cargo run --release -p physsynth-viewer --example verify_headless -- [--out DIR] [--profile DIR] [FILTER...]
//! ```
//!
//! A FILTER keeps the cases whose name contains it. `VIEWER_BASE` overrides the viewer's address
//! (default `http://127.0.0.1:8000`), `PHYSSYNTH_CHROME` the browser's path. Exits 1 if any case
//! fails, 2 if the check could not run at all.
//!
//! # The browser is shut down through its own port, and nothing else is touched
//!
//! The Python harness this replaces ended with `proc.terminate()`, which on Windows stops only the
//! launcher stub: the real browser survived every run of phase D and held its port. This one gives
//! the browser its own profile directory and its own debugging port, asks it to close over that
//! port (`Browser.close`), then waits for the port to go quiet. Only if it is still up does it kill
//! a process, and only the one it spawned, by that process's own handle. A Chrome it did not launch
//! (one already listening on the port, which it attaches to) is never closed.
//!
//! # Why the WebSocket client is written here
//!
//! The DevTools protocol is JSON over a WebSocket. A client crate would be a dev-dependency, and
//! `tests/deps.rs` walks only the dependencies the viewer SHIPS, so its whole tree would go
//! unreviewed. What this needs is small: one handshake, masked text frames out, text frames in
//! (possibly fragmented), and a pong for a ping. The same judgement as the server's: `std::net`,
//! no framework (retirement plan §6.1, §23.19).

use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const PORT: u16 = 9333;

const CHROME_CANDIDATES: &[&str] = &[
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
];

/// Run after each render: the rendered read-outs, plus a histogram of the main canvas's pixel
/// "temperature" (warm = positive displacement, cool = negative, bg = exterior or at rest), so a
/// case can assert that the field actually painted into the interior.
const PROBE: &str = r"
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
    horizon: (document.getElementById('horizon').hidden ? '(hidden)'
              : document.getElementById('horizon-badge').textContent + ' — '
                + document.getElementById('horizon-line').textContent),
    urlNotes: (window.__urlParamNotes || []).join(' | '),
    hzMark: window.__horizonMark || null,
    warm: warm, cool: cool, bg: bg, other: other, total: d.length / 4,
  });
})()
";

/// What the second diagnostic panel may have decided about the horizon. The vocabulary lives in
/// two files — `markHorizon` in `web/static/app.js` emits it, this grades it — so
/// `tests/front_end.rs` asserts the two agree; a reason this set does not know fails LOUDLY.
const HORIZON_MARK_REASONS: &[&str] =
    &["refused", "whole-grid", "off-panel", "zero-modes", "drawn"];

/// `(name, deep-link query)`. Every model the viewer offers has at least one (`tests/front_end.rs`).
const CASES: &[(&str, &str)] = &[
    ("membrane_circle", "model=membrane&domain=circle"),
    ("membrane_rect", "model=membrane&domain=rectangle"),
    ("mallet_circle", "model=mallet&domain=circle"),
    ("mallet_rect", "model=mallet&domain=rectangle"),
    ("string_ideal", "model=ideal"),
    ("string_stiff", "model=stiff"),
    ("string_damped", "model=damped"),
    (
        "string_damped_lossy",
        "model=damped&sigma0=2&sigma1=0.0001&kappa=1",
    ),
    ("string_tension", "model=tension"),
    // The tension string's second regime: the same string past its parametric threshold, where the
    // Duffing panel refuses to go. Two claim runs plus the tongue sweep.
    ("string_parametric", "model=tension&domain=parametric"),
    ("string_bow", "model=bow"),
    ("plate_supported", "model=plate&domain=supported"),
    ("plate_free", "model=plate&domain=free"),
    // The only 2-D field with a NON-CONVEX mask: what a headless render sees and a payload test
    // cannot is the outline actually painted — masked exterior, both bouts, the waist between.
    // N and mu are pinned low because the claim sweep is ~24 eigensolves on top of the audio run.
    (
        "plate_guitar",
        "model=plate&domain=guitar&N=24&mu=8&audio_duration=0.3",
    ),
    ("vk_supported", "model=vk&domain=supported"),
    ("vk_free", "model=vk&domain=free"),
    // The geometric string's four regimes: every step a 3-field Newton solve at ~22x a normal
    // string's fs. The first three ship no audio, so `painted` is their whole verdict; `phantom`
    // measures the bridge force, and at ~45 s it is the slowest render in the viewer.
    ("geom_rotating", "model=geometric&domain=rotating"),
    ("geom_planar", "model=geometric&domain=planar"),
    ("geom_whirl", "model=geometric&domain=whirl"),
    ("geom_phantom", "model=geometric&domain=phantom"),
    ("symp_normal", "model=sympathetic&domain=normal"),
    ("symp_transfer", "model=sympathetic&domain=transfer"),
    ("symp_weinreich", "model=sympathetic&domain=weinreich"),
    ("jawari", "model=jawari"),
    // Its cost is the tuning-curve sweep (~11 thread positions), not the audio.
    ("juari", "model=juari"),
    // The most expensive model per second of audio; also a two-run model (the brightness control).
    ("fret", "model=fret"),
    // Both far ends, because they exercise different halves of drawBore: the flared mouth and its
    // glow, and the p = 0 node.
    ("bore_radiating", "model=bore&domain=radiating"),
    ("bore_open", "model=bore&domain=open"),
    ("reed_radiating", "model=reed&domain=radiating"),
    ("reed_open", "model=reed&domain=open"),
    ("body", "model=body"),
    ("platebody_free", "model=platebody&domain=free"),
    ("platebody_supported", "model=platebody&domain=supported"),
    ("radbody", "model=radbody"),
    ("airload", "model=airload"),
    // The first `dims: 3` field, shipped as a slice set. Plane buffers are C order (nv fastest), and
    // a front-end decoding them u-fastest TRANSPOSES the picture — plausible banding, not an error.
    // Only pixels catch that.
    ("airbox", "model=airbox"),
    // The only lossy wall, and the case that proves the zeta slider is reachable.
    ("airbox_absorbing", "model=airbox&domain=absorbing"),
    // A DUAL field pane, reusing the slice decode above in a pane no payload test looks at. Found
    // the hard way: `requestAnimationFrame` does not fire in a background tab, so a correct field
    // reads as a blank canvas — this driver's own foregrounded load is what makes pixels mean
    // anything. Kept short: the default 0.12 s render is ~23 s.
    ("vkroom", "model=vkroom&audio_duration=0.02"),
    (
        "vkroom_baffled",
        "model=vkroom&domain=baffled&audio_duration=0.02",
    ),
    // The coarsest legal plate, where the claim still separates (63x against a 20x bar).
    (
        "vkroom_coarse",
        "model=vkroom&plate_N=8&audio_duration=0.02",
    ),
];

// -- a minimal HTTP GET and WebSocket client over std::net ----------------------------------------

/// `host:port` and path of an `http://` or `ws://` URL.
fn split_url(url: &str) -> (String, String) {
    let rest = url.split_once("://").map_or(url, |(_, r)| r);
    match rest.find('/') {
        Some(i) => (rest[..i].to_owned(), rest[i..].to_owned()),
        None => (rest.to_owned(), "/".to_owned()),
    }
}

/// The body of a `GET`, or `None` if nothing answers.
fn http_get(url: &str, timeout: Duration) -> Option<Vec<u8>> {
    let (host, path) = split_url(url);
    let addr = std::net::ToSocketAddrs::to_socket_addrs(&host)
        .ok()?
        .next()?;
    let mut s = TcpStream::connect_timeout(&addr, timeout).ok()?;
    s.set_read_timeout(Some(timeout)).ok()?;
    write!(
        s,
        "GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n"
    )
    .ok()?;
    // Read by the stated length, never to end of stream: the DevTools endpoint ignores
    // `Connection: close` and holds the socket open, so a read-to-end only ends at the timeout.
    let mut r = BufReader::new(s);
    let mut head = String::new();
    loop {
        let mut line = String::new();
        if r.read_line(&mut line).ok()? == 0 || line == "\r\n" {
            break;
        }
        head.push_str(&line.to_ascii_lowercase());
    }
    if !head.starts_with("http/1.1 200") && !head.starts_with("http/1.0 200") {
        return None;
    }
    let length = head
        .lines()
        .find_map(|l| l.strip_prefix("content-length:"))
        .and_then(|v| v.trim().parse::<usize>().ok());
    if let Some(n) = length {
        let mut body = vec![0u8; n];
        r.read_exact(&mut body).ok()?;
        return Some(body);
    }
    if !head.contains("transfer-encoding: chunked") {
        let mut body = Vec::new();
        r.read_to_end(&mut body).ok()?;
        return Some(body);
    }
    let mut out = Vec::new();
    loop {
        let mut line = String::new();
        r.read_line(&mut line).ok()?;
        let size = usize::from_str_radix(line.trim(), 16).ok()?;
        if size == 0 {
            return Some(out);
        }
        let start = out.len();
        out.resize(start + size + 2, 0);
        r.read_exact(&mut out[start..]).ok()?;
        out.truncate(start + size);
    }
}

fn http_json(url: &str, timeout: Duration) -> Option<Value> {
    serde_json::from_slice(&http_get(url, timeout)?).ok()
}

struct WebSocket {
    reader: BufReader<TcpStream>,
    writer: TcpStream,
}

impl WebSocket {
    fn connect(url: &str) -> std::io::Result<Self> {
        let (host, path) = split_url(url);
        let writer = TcpStream::connect(&host)?;
        let mut reader = BufReader::new(writer.try_clone()?);
        // The key only has to be 16 bytes in base64; the server's accept hash is not checked,
        // because this client talks to one local browser it launched or was pointed at.
        let key = b64_encode(&nonce());
        let mut w = &writer;
        write!(
            w,
            "GET {path} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\
             Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
        )?;
        let mut status = String::new();
        reader.read_line(&mut status)?;
        if !status.contains(" 101 ") {
            return Err(std::io::Error::other(format!(
                "handshake refused: {status:?}"
            )));
        }
        loop {
            let mut line = String::new();
            reader.read_line(&mut line)?;
            if line == "\r\n" || line.is_empty() {
                break;
            }
        }
        Ok(Self { reader, writer })
    }

    /// One masked frame, as a client must send.
    fn send_frame(&mut self, opcode: u8, payload: &[u8]) -> std::io::Result<()> {
        let mut f = vec![0x80 | opcode];
        let n = payload.len();
        if n < 126 {
            f.push(0x80 | n as u8);
        } else if n <= 0xffff {
            f.push(0x80 | 126);
            f.extend_from_slice(&(n as u16).to_be_bytes());
        } else {
            f.push(0x80 | 127);
            f.extend_from_slice(&(n as u64).to_be_bytes());
        }
        let mask = nonce();
        f.extend_from_slice(&mask[..4]);
        f.extend(payload.iter().enumerate().map(|(i, b)| b ^ mask[i % 4]));
        self.writer.write_all(&f)
    }

    fn send_text(&mut self, text: &str) -> std::io::Result<()> {
        self.send_frame(0x1, text.as_bytes())
    }

    /// The next whole text message, reassembling fragments and answering pings on the way.
    fn recv_text(&mut self) -> std::io::Result<String> {
        let mut msg = Vec::new();
        loop {
            let mut h = [0u8; 2];
            self.reader.read_exact(&mut h)?;
            let (fin, opcode) = (h[0] & 0x80 != 0, h[0] & 0x0f);
            let mut n = u64::from(h[1] & 0x7f);
            if n == 126 {
                let mut b = [0u8; 2];
                self.reader.read_exact(&mut b)?;
                n = u64::from(u16::from_be_bytes(b));
            } else if n == 127 {
                let mut b = [0u8; 8];
                self.reader.read_exact(&mut b)?;
                n = u64::from_be_bytes(b);
            }
            let mask = if h[1] & 0x80 != 0 {
                let mut m = [0u8; 4];
                self.reader.read_exact(&mut m)?;
                Some(m)
            } else {
                None
            };
            let mut payload = vec![0u8; n as usize];
            self.reader.read_exact(&mut payload)?;
            if let Some(m) = mask {
                payload
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, b)| *b ^= m[i % 4]);
            }
            match opcode {
                0x0..=0x2 => {
                    msg.extend_from_slice(&payload);
                    if fin {
                        return String::from_utf8(msg).map_err(std::io::Error::other);
                    }
                }
                0x8 => return Err(std::io::Error::other("the browser closed the socket")),
                0x9 => self.send_frame(0xA, &payload)?,
                _ => {}
            }
        }
    }
}

/// 16 bytes that differ between calls — a WebSocket key and mask need no more than that.
fn nonce() -> [u8; 16] {
    use std::hash::{BuildHasher, Hasher};
    let mut out = [0u8; 16];
    for half in out.chunks_mut(8) {
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u128(Instant::now().elapsed().as_nanos());
        half.copy_from_slice(&h.finish().to_le_bytes());
    }
    out
}

const B64: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn b64_encode(b: &[u8]) -> String {
    let mut s = String::new();
    for c in b.chunks(3) {
        let n = (u32::from(c[0]) << 16)
            | (u32::from(*c.get(1).unwrap_or(&0)) << 8)
            | u32::from(*c.get(2).unwrap_or(&0));
        for k in 0..4 {
            s.push(if k <= c.len() {
                B64[(n >> (18 - 6 * k)) as usize & 63] as char
            } else {
                '='
            });
        }
    }
    s
}

fn b64_decode(s: &str) -> Vec<u8> {
    let val = |c: u8| B64.iter().position(|&x| x == c).unwrap_or(0) as u32;
    let mut out = Vec::new();
    for chunk in s.as_bytes().chunks(4) {
        let pad = chunk.iter().filter(|&&c| c == b'=').count();
        let mut n = 0u32;
        for (i, &c) in chunk.iter().enumerate() {
            n |= if c == b'=' { 0 } else { val(c) } << (18 - 6 * i);
        }
        out.extend_from_slice(&[(n >> 16) as u8, (n >> 8) as u8, n as u8][..3 - pad]);
    }
    out
}

// -- the DevTools protocol ------------------------------------------------------------------------

struct Cdp {
    ws: WebSocket,
    id: u64,
}

impl Cdp {
    fn cmd(&mut self, method: &str, params: Value) -> std::io::Result<Value> {
        self.id += 1;
        let id = self.id;
        self.ws
            .send_text(&json!({"id": id, "method": method, "params": params}).to_string())?;
        loop {
            let msg: Value = serde_json::from_str(&self.ws.recv_text()?)?;
            if msg["id"].as_u64() == Some(id) {
                return Ok(msg);
            }
        }
    }

    fn evaluate(&mut self, expr: &str) -> std::io::Result<Value> {
        let r = self.cmd(
            "Runtime.evaluate",
            json!({"expression": expr, "returnByValue": true}),
        )?;
        Ok(r["result"]["result"]["value"].clone())
    }
}

/// The first `page` target on the port, polling until `timeout` is up. Chrome on Windows can
/// launch through a stub that hands off and exits, so the endpoint is the only thing that means
/// the browser is up — never the launched process's exit.
fn devtools_page(timeout: Duration) -> Option<Value> {
    let end = Instant::now() + timeout;
    loop {
        if let Some(Value::Array(ts)) = http_json(
            &format!("http://127.0.0.1:{PORT}/json"),
            Duration::from_secs(2),
        ) {
            if let Some(p) = ts
                .into_iter()
                .find(|t| t["type"] == "page" && t["webSocketDebuggerUrl"].is_string())
            {
                return Some(p);
            }
        }
        if Instant::now() >= end {
            return None;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

fn devtools_up() -> bool {
    http_get(
        &format!("http://127.0.0.1:{PORT}/json/version"),
        Duration::from_millis(500),
    )
    .is_some()
}

/// Close the browser this run launched: `Browser.close` over its own port, then wait for the port
/// to go quiet. Only if it outlives that is a process killed, and only the one this run spawned.
fn shut_down(mut child: Child) {
    if let Some(v) = http_json(
        &format!("http://127.0.0.1:{PORT}/json/version"),
        Duration::from_secs(2),
    ) {
        if let Some(url) = v["webSocketDebuggerUrl"].as_str() {
            if let Ok(ws) = WebSocket::connect(url) {
                let mut cdp = Cdp { ws, id: 0 };
                let _ = cdp.cmd("Browser.close", json!({}));
            }
        }
    }
    let end = Instant::now() + Duration::from_secs(15);
    while devtools_up() && Instant::now() < end {
        std::thread::sleep(Duration::from_millis(250));
    }
    if !devtools_up() {
        let _ = child.wait();
        println!("browser closed through its own port.");
        return;
    }
    match child.try_wait() {
        Ok(None) => {
            println!("browser outlived Browser.close; killing the process this run spawned");
            let _ = child.kill();
            let _ = child.wait();
        }
        _ => println!(
            "WARNING: the DevTools port {PORT} is still open and the process this run spawned has \
             exited, so the browser listening there is not one this run can name. Nothing killed."
        ),
    }
}

// -- the verdict ----------------------------------------------------------------------------------

/// Does what the PANEL drew agree with what the STRIP says? `window.__horizonMark` is the page's own
/// record of what the panel concluded; a strip quoting a number while the panel decided the scene
/// was refused (or the reverse) is a read-out contradicting itself on screen, which a screenshot
/// would not catch. A null mark is a pass: most panels have no frequency axis to draw a horizon on.
fn horizon_mark_ok(mark: &Value, horizon: &str) -> (bool, String) {
    if mark.is_null() {
        return (true, "no markable axis".into());
    }
    let Some(reason) = mark["reason"]
        .as_str()
        .filter(|r| HORIZON_MARK_REASONS.contains(r))
    else {
        return (false, format!("malformed mark {mark}"));
    };
    let strip_refused = horizon.starts_with("not quoted");
    if strip_refused != (reason == "refused") {
        let quotes = if strip_refused {
            "no number"
        } else {
            "a number"
        };
        return (
            false,
            format!("panel says {reason:?} but the strip quotes {quotes}"),
        );
    }
    let at = &mark["at"];
    if mark["drawn"].as_bool() == Some(true) && !at.is_number() {
        return (false, format!("drawn with no position ({at})"));
    }
    let units = mark["units"].as_str().unwrap_or("");
    let note = if at.is_null() {
        reason.to_owned()
    } else {
        format!("{reason} at {at} {units}")
    };
    (true, note)
}

fn first_line(s: &Value) -> String {
    s.as_str()
        .and_then(|t| t.lines().next())
        .unwrap_or("(none)")
        .to_owned()
}

fn run_case(
    cdp: &mut Cdp,
    base: &str,
    out: &Path,
    name: &str,
    query: &str,
) -> std::io::Result<bool> {
    cdp.cmd("Page.navigate", json!({"url": format!("{base}/?{query}")}))?;
    // Up to ~90 s: the phantom regime's fixed 0.10 s window at fs ~ 159 kHz is ~45 s of Newton
    // steps. A case failing on "computing…" on a loaded machine is the wait, not the code — check
    // the render time before suspecting anything else.
    for _ in 0..450 {
        let status = cdp.evaluate("document.getElementById('status').textContent")?;
        let s = status.as_str().unwrap_or("");
        if s.starts_with("ok") || s.starts_with("error") || s.starts_with("network") {
            break;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    let probe: Value = serde_json::from_str(cdp.evaluate(PROBE)?.as_str().unwrap_or("{}"))?;
    let shot = cdp.cmd("Page.captureScreenshot", json!({}))?;
    if let Some(png) = shot["result"]["data"].as_str() {
        std::fs::write(out.join(format!("viewer_{name}.png")), b64_decode(png))?;
    }
    let n = |k: &str| probe[k].as_u64().unwrap_or(0);
    let painted = n("warm") + n("cool") + n("other");
    // A deep-link parameter dropped on the floor is a pass that proves nothing, so the page's own
    // record of every name it did not know and every value it clamped is part of the verdict.
    let notes = probe["urlNotes"].as_str().unwrap_or("");
    // The resolution strip must be visible and say something; its content is a two-value union,
    // so a case can be asked for an answer, not a number.
    let horizon = probe["horizon"].as_str().unwrap_or("(hidden)");
    let (mark_ok, mark_note) = horizon_mark_ok(&probe["hzMark"], horizon);
    let status = probe["status"].as_str().unwrap_or("");
    let ok = status.starts_with("ok")
        && painted > 2000
        && notes.is_empty()
        && horizon != "(hidden)"
        && horizon.chars().count() > 20
        && mark_ok;
    println!("\n=== {name} ({query}) ===");
    println!("  status   : {status}");
    println!("  energy   : {}", first_line(&probe["energy"]));
    println!("  diag2    : {}", first_line(&probe["diag2"]));
    println!("  horizon  : {horizon}");
    println!(
        "  hz-mark  : {mark_note}{}",
        if mark_ok { "" } else { "   <- MISMATCH" }
    );
    println!(
        "  url      : {}",
        if notes.is_empty() {
            "every parameter applied as given"
        } else {
            notes
        }
    );
    println!(
        "  painted  : warm={} cool={} other={} bg={} / {}  -> {}",
        n("warm"),
        n("cool"),
        n("other"),
        n("bg"),
        n("total"),
        if ok { "PASS" } else { "FAIL" }
    );
    Ok(ok)
}

fn find_chrome() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("PHYSSYNTH_CHROME") {
        return Some(PathBuf::from(p));
    }
    CHROME_CANDIDATES
        .iter()
        .map(PathBuf::from)
        .find(|p| p.exists())
}

fn main() {
    std::process::exit(run());
}

fn run() -> i32 {
    let mut args = std::env::args().skip(1);
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut out = repo.join("out");
    let mut profile: Option<PathBuf> = None;
    let mut filters = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--out" => out = PathBuf::from(args.next().expect("--out needs a directory")),
            "--profile" => {
                profile = Some(PathBuf::from(
                    args.next().expect("--profile needs a directory"),
                ))
            }
            _ => filters.push(a),
        }
    }
    let base = std::env::var("VIEWER_BASE").unwrap_or_else(|_| "http://127.0.0.1:8000".into());
    std::fs::create_dir_all(&out).expect("the screenshot directory");
    let cases: Vec<(&str, &str)> = CASES
        .iter()
        .copied()
        .filter(|(n, _)| filters.is_empty() || filters.iter().any(|f| n.contains(f.as_str())))
        .collect();
    if cases.is_empty() {
        println!("no case matches {filters:?}");
        return 2;
    }
    if http_get(&format!("{base}/"), Duration::from_secs(2)).is_none() {
        println!(
            "server not reachable at {base}; start it with \
             `cargo run --release -p physsynth-viewer -- serve`."
        );
        return 2;
    }

    // ATTACH if a browser already listens on the port, LAUNCH otherwise. An attached browser is
    // never closed. Pass a FRESH profile when web/static has just been edited, or an attached
    // browser may serve its cached copy and hand back a pass that proves nothing.
    let mut child = None;
    let page = match devtools_page(Duration::from_millis(500)) {
        Some(p) => {
            println!("attaching to the browser already listening on :{PORT} (nothing launched).");
            Some(p)
        }
        None => {
            let Some(chrome) = find_chrome() else {
                println!("Chrome not found; set PHYSSYNTH_CHROME to its path.");
                return 2;
            };
            let dir = profile.unwrap_or_else(|| {
                std::env::temp_dir().join(format!("physsynth-chrome-verify-{}", std::process::id()))
            });
            std::fs::create_dir_all(&dir).expect("the browser profile directory");
            let spawned = Command::new(&chrome)
                .args([
                    "--headless=new",
                    "--disable-gpu",
                    "--no-first-run",
                    "--no-default-browser-check",
                    &format!("--remote-debugging-port={PORT}"),
                    &format!("--user-data-dir={}", dir.display()),
                    "--remote-allow-origins=*",
                    "--window-size=1400,1000",
                    "about:blank",
                ])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn();
            match spawned {
                Ok(c) => {
                    println!(
                        "launched {} as pid {} (profile {})",
                        chrome.display(),
                        c.id(),
                        dir.display()
                    );
                    child = Some(c);
                }
                Err(e) => {
                    println!("could not launch {}: {e}", chrome.display());
                    return 2;
                }
            }
            devtools_page(Duration::from_secs(30))
        }
    };

    let code = match page {
        None => {
            println!(
                "could not reach a DevTools endpoint on :{PORT}. Start a browser yourself and \
                 re-run — this will attach to it:\n  chrome --headless=new \
                 --remote-debugging-port={PORT} --user-data-dir=<a fresh dir> about:blank"
            );
            2
        }
        Some(p) => {
            let url = p["webSocketDebuggerUrl"].as_str().unwrap().to_owned();
            let result = (|| -> std::io::Result<Vec<bool>> {
                let mut cdp = Cdp {
                    ws: WebSocket::connect(&url)?,
                    id: 0,
                };
                cdp.cmd("Page.enable", json!({}))?;
                cdp.cmd("Runtime.enable", json!({}))?;
                cases
                    .iter()
                    .map(|(n, q)| run_case(&mut cdp, &base, &out, n, q))
                    .collect()
            })();
            match result {
                Ok(r) => {
                    let passed = r.iter().filter(|&&x| x).count();
                    println!(
                        "\n{passed}/{} cases passed; screenshots in {}",
                        r.len(),
                        out.display()
                    );
                    i32::from(passed != r.len())
                }
                Err(e) => {
                    println!("the DevTools session failed: {e}");
                    2
                }
            }
        }
    };
    if let Some(c) = child {
        shut_down(c);
    }
    code
}
