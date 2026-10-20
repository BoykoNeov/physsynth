//! Headless-browser check of the web viewer: what the page RENDERED, not what the server sent.
//!
//! The payload tests prove the data is right; this proves the page drew it. It drives a real Chrome
//! over the DevTools protocol, loads the running viewer with a `?model=â¦&domain=â¦` deep link per
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
//! no framework (retirement plan Â§6.1, Â§23.19).

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
              : document.getElementById('horizon-badge').textContent + ' â '
                + document.getElementById('horizon-line').textContent),
    urlNotes: (window.__urlParamNotes || []).join(' | '),
    tuning: (document.getElementById('tuning-hint') || {}).textContent || '',
    plateTuning: (document.getElementById('plate-tuning-hint') || {}).textContent || '',
    drumTuning: (document.getElementById('drum-tuning-hint') || {}).textContent || '',
    sliders: [...document.querySelectorAll('.slider')].filter((e) => e.offsetParent !== null)
      .map((e) => e.dataset.param).join(' '),
    hzMark: window.__horizonMark || null,
    warm: warm, cool: cool, bg: bg, other: other, total: d.length / 4,
  });
})()
";

/// What the second diagnostic panel may have decided about the horizon. The vocabulary lives in
/// two files â `markHorizon` in `web/static/app.js` emits it, this grades it â so
/// `tests/front_end.rs` asserts the two agree; a reason this set does not know fails LOUDLY.
const HORIZON_MARK_REASONS: &[&str] =
    &["refused", "whole-grid", "off-panel", "zero-modes", "drawn"];

/// `(name, deep-link query)`. Every model the viewer offers has at least one (`tests/front_end.rs`).
const CASES: &[(&str, &str)] = &[
    ("membrane_circle", "model=membrane&domain=circle"),
    ("membrane_rect", "model=membrane&domain=rectangle"),
    ("mallet_circle", "model=mallet&domain=circle"),
    ("mallet_rect", "model=mallet&domain=rectangle"),
    // The drum's musical controls (membrane mapping plan §6): a cited head set by pitch on the
    // round drum, and the mallet striking a drum set by tension.
    (
        "membrane_musical",
        "model=membrane&domain=circle&controls=musical&head=single-ply%2010%20mil&N=48\
         &f0=160&f_ref=900&audio_duration=0.5",
    ),
    (
        "mallet_musical",
        "model=mallet&domain=circle&controls=musical&head=two-ply%207%20%2B%207%20mil\
         &set_by=tension&tension=3000&N=40&f_ref=900&audio_duration=0.4",
    ),
    ("string_ideal", "model=ideal"),
    ("string_stiff", "model=stiff"),
    ("string_damped", "model=damped"),
    (
        "string_damped_lossy",
        "model=damped&sigma0=2&sigma1=0.0001&kappa=1",
    ),
    // The musical controls (parameter-mapping plan §6), on the coarse low stiff string that
    // WARNS: the read-out must name the mapped physics and the drift.
    (
        "string_damped_musical",
        "model=damped&domain=musical&N=64&f0=82.4&inharmonicity=0.01&audio_duration=0.5",
    ),
    ("string_tension", "model=tension"),
    // The tension string's second regime: the same string past its parametric threshold, where the
    // Duffing panel refuses to go. Two claim runs plus the tongue sweep.
    ("string_parametric", "model=tension&domain=parametric"),
    ("string_bow", "model=bow"),
    // The bow's and the tension string's musical controls (string mapping plan §7): the bow on a
    // tuned string, whose read-out names the bowed pitch; the tension string from a steel gauge,
    // whose read-out names what the gauge gave and where the pluck started.
    (
        "string_bow_musical",
        "model=bow&controls=musical&N=100&f0=110&audio_duration=1",
    ),
    // Both pages exactly as a user first meets them: every slider at its default. The tension
    // string's defaults sit near its 60,000-step budget, so a default that drifts past it shows
    // here as a refusal rather than in a user's first click.
    ("string_bow_musical_defaults", "model=bow&controls=musical"),
    // The ideal string and a barrier page with musical controls (string mapping plan §13): the
    // ideal one at its defaults (its sample rate follows the pitch), the jawari on its own grid.
    ("string_ideal_musical", "model=ideal&controls=musical"),
    ("string_stiff_musical", "model=stiff&controls=musical"),
    (
        "jawari_musical",
        "model=jawari&controls=musical&N=64&f0=110&audio_duration=0.24",
    ),
    (
        "string_tension_musical_defaults",
        "model=tension&controls=musical",
    ),
    (
        "string_tension_musical",
        "model=tension&domain=duffing&controls=musical&string_material=steel&N=64&L=0.65\
         &f0=196&amplitude=0.003&audio_duration=0.3",
    ),
    ("plate_supported", "model=plate&domain=supported"),
    // The plate's musical controls (plate mapping plan §7): a cited metal set by pitch on the
    // supported plate, and a spruce guitar top — a free plate, so the read-out says it is not exact.
    (
        "plate_musical",
        "model=plate&domain=supported&controls=musical&material=brass&N=40&audio_duration=0.5",
    ),
    (
        "plate_musical_guitar",
        "model=plate&domain=guitar&controls=musical&material=sitka%20spruce&N=24&f0=140\
         &f_ref=1200&t60_high=1.5&audio_duration=0.3",
    ),
    ("plate_free", "model=plate&domain=free"),
    // The only 2-D field with a NON-CONVEX mask: what a headless render sees and a payload test
    // cannot is the outline actually painted â masked exterior, both bouts, the waist between.
    // N and mu are pinned low because the claim sweep is ~24 eigensolves on top of the audio run.
    (
        "plate_guitar",
        "model=plate&domain=guitar&N=24&mu=8&audio_duration=0.3",
    ),
    ("vk_supported", "model=vk&domain=supported"),
    ("vk_free", "model=vk&domain=free"),
    // The gong's musical controls (vk mapping plan §6): a brass gong set by its settled pitch,
    // whose read-out names the thickness, what the strike did to the note and what the run cost;
    // and a bronze cymbal set by thickness, struck off its centre lines and read from its lowest
    // tone.
    (
        "vk_musical",
        "model=vk&domain=supported&controls=musical&material=brass&f0=120&N=16\
         &audio_duration=0.5",
    ),
    (
        "vk_musical_cymbal",
        "model=vk&domain=free&controls=musical&material=bronze&set_by=thickness\
         &thickness_mm=1.5&N=16&audio_duration=0.3&pluck_x=0.3&pluck_y=0.3",
    ),
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
    // The bore's musical controls (wind mapping plan §2): the page's defaults through the bell, and
    // the open end, which takes the pitch alone and says why.
    ("bore_musical", "model=bore&domain=radiating&controls=musical"),
    (
        "bore_musical_open",
        "model=bore&domain=open&controls=musical&f0=220",
    ),
    ("reed_radiating", "model=reed&domain=radiating"),
    ("reed_open", "model=reed&domain=open"),
    ("body", "model=body"),
    ("platebody_free", "model=platebody&domain=free"),
    ("platebody_supported", "model=platebody&domain=supported"),
    ("radbody", "model=radbody"),
    // The strings on a bridge with musical controls (string mapping plan §16–§17): the pitch solved
    // through the coupling, the ring time off (seconds) and on (the slow path, on a coarse grid), a
    // split band named, and the two-string page with its detune solved in.
    (
        "body_musical",
        "model=body&controls=musical&N=48&ring_time=false&audio_duration=0.5",
    ),
    (
        "body_musical_wolf",
        "model=body&controls=musical&N=40&f0=196&ring_time=false&audio_duration=0.5",
    ),
    (
        "radbody_musical",
        "model=radbody&controls=musical&N=40&ring_time=true&t60_low=1&audio_duration=0.5",
    ),
    (
        "symp_musical_transfer",
        "model=sympathetic&domain=transfer&controls=musical&N=40&detune=1.5&ring_time=false",
    ),
    // The string on a plate (plan §18): read off the plate's modes, both edges, the ring time
    // off at the page's default and on, on a coarser grid.
    ("platebody_musical", "model=platebody&controls=musical&ring_time=false"),
    (
        "platebody_musical_ring",
        "model=platebody&domain=supported&controls=musical&N=48&n_plate=10&ring_time=true&t60_low=1",
    ),
    ("airload", "model=airload"),
    // The first `dims: 3` field, shipped as a slice set. Plane buffers are C order (nv fastest), and
    // a front-end decoding them u-fastest TRANSPOSES the picture â plausible banding, not an error.
    // Only pixels catch that.
    ("airbox", "model=airbox"),
    // The only lossy wall, and the case that proves the zeta slider is reachable.
    ("airbox_absorbing", "model=airbox&domain=absorbing"),
    // A DUAL field pane, reusing the slice decode above in a pane no payload test looks at. Found
    // the hard way: `requestAnimationFrame` does not fire in a background tab, so a correct field
    // reads as a blank canvas â this driver's own foregrounded load is what makes pixels mean
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
    // The drum in a room (membrane mapping plan §14): the gong's dual pane with a ROUND masked
    // head, and the air read-out panel. The physical head set in a wall at the default 0.35 s
    // (~5 s), and a musical head hung in absorbing air on a coarser room.
    ("drumroom", "model=drumroom"),
    (
        "drumroom_musical_hung",
        "model=drumroom&domain=suspended&controls=musical&head=two-ply%207%20%2B%207%20mil\
         &f0=180&f_ref=900&t60_low=1.5&t60_high=0.5&air_h=0.02&walls=rigid",
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

/// 16 bytes that differ between calls â a WebSocket key and mask need no more than that.
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
/// the browser is up â never the launched process's exit.
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

/// A musical request on one of the strings on a bridge.
fn musical_coupled(query: &str) -> bool {
    query.contains("controls=musical")
        && [
            "model=body&",
            "model=radbody&",
            "model=sympathetic&",
            "model=platebody&",
        ]
        .iter()
        .any(|m| query.starts_with(m))
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
    // steps. A case failing on "computingâ¦" on a loaded machine is the wait, not the code â check
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
    // The musical domain's read-out must say what the targets mapped to; elsewhere it is empty.
    let tuning = probe["tuning"].as_str().unwrap_or("");
    let tuning_ok = if query.contains("domain=musical") {
        tuning.contains("Exact at f₀") && tuning.contains("cents")
    } else if query.contains("controls=musical") && query.contains("model=bow") {
        tuning.contains("Exact at f₀") && tuning.contains("moves with N")
            // the bow's 22.05 kHz default must sit on the sample-rate slider's grid, or the page
            // quietly runs another rate (it ran 22,100 Hz on a 100 Hz step, 2026-10-19)
            && probe["status"].as_str().unwrap_or("").contains("fs_sim 22,050 Hz")
    } else if query.contains("controls=musical") && query.contains("model=stiff") {
        tuning.contains("Exact at f₀") && tuning.contains("One loss")
    } else if query.contains("controls=musical") && query.contains("model=ideal") {
        tuning.contains("λ = 1") && tuning.contains("EVERY partial")
    } else if query.contains("controls=musical") && query.contains("model=jawari") {
        tuning.contains("exact at f₀") && tuning.contains("σ =")
    } else if query.contains("controls=musical") && query.contains("model=tension") {
        tuning.contains("Exact at f₀")
            && tuning.contains("EA =")
            && tuning.contains("settles on f₀")
    } else if query.contains("controls=musical") && query.contains("model=bore") {
        tuning.contains("L = ")
            && if query.contains("domain=open") {
                tuning.contains("no bell") && tuning.contains("rings forever")
            } else {
                tuning.contains("R/Z₀") && tuning.contains("exact at λ = 1")
            }
    } else if musical_coupled(query) {
        tuning.contains("the string alone would sound")
            && tuning.contains("On the bridge it plays")
            && (!query.contains("f0=196") || tuning.contains("Split band"))
            && (!query.contains("model=radbody") || tuning.contains("air sweep"))
    } else {
        true
    };
    // The plate's musical read-out names the thickness, the mapped stiffness and the partials'
    // ring times; a free plate's also says how far from exact it is.
    let musical = query.contains("controls=musical");
    let plate_tuning = probe["plateTuning"].as_str().unwrap_or("");
    // `model=plate&`, not `model=plate`: the latter also names the string on a plate (`platebody`).
    let plate_tuning_ok = if musical && query.contains("model=plate&") {
        plate_tuning.contains(" mm")
            && plate_tuning.contains("κ =")
            && plate_tuning.contains("Ring times")
            && (!query.contains("domain=guitar") || plate_tuning.contains("0.05 cents"))
    } else if musical && query.contains("model=vk") {
        // The gong's: the thickness and its metal's constants, what the strike did, what it cost.
        plate_tuning.contains(" mm")
            && plate_tuning.contains("E =")
            && plate_tuning.contains("The strike")
            && plate_tuning.contains("Cost:")
            && plate_tuning.contains("Ring times")
    } else {
        plate_tuning.is_empty()
    };
    // The drum's names the tension, the wave speed and the partials' ring times, and says the
    // simulated drum meets the targets exactly (membrane mapping plan §6) — the mallet's too.
    let drum_tuning = probe["drumTuning"].as_str().unwrap_or("");
    let drum = query.contains("model=membrane")
        || query.contains("model=mallet")
        || query.contains("model=drumroom");
    let drum_tuning_ok = if musical && drum {
        drum_tuning.contains("N/m")
            && drum_tuning.contains("c =")
            && drum_tuning.contains("Ring times")
            && drum_tuning.contains("exactly")
    } else {
        drum_tuning.is_empty()
    };
    // The drum in a room's panel says, per mode, where the head sat in vacuo and where the room
    // put it; a panel left blank or showing another model's read-out is a failure.
    let diag2 = probe["diag2"].as_str().unwrap_or("");
    let sliders = probe["sliders"].as_str().unwrap_or("");
    let air_ok = !query.contains("model=drumroom")
        || (diag2.contains("in vacuo") && diag2.contains("in the room")
            // ζ is fixed at 1 here; a visible ζ slider would be a control the server never reads
            && !sliders.split(' ').any(|s| s == "wall_zeta"));
    let tuning_ok = tuning_ok && plate_tuning_ok && drum_tuning_ok && air_ok;
    let tuning = [plate_tuning, drum_tuning]
        .into_iter()
        .find(|s| !s.is_empty())
        .unwrap_or(tuning);
    let status = probe["status"].as_str().unwrap_or("");
    let ok = status.starts_with("ok")
        && tuning_ok
        && painted > 2000
        && notes.is_empty()
        && horizon != "(hidden)"
        && horizon.chars().count() > 20
        && mark_ok;
    println!("\n=== {name} ({query}) ===");
    println!("  status   : {status}");
    println!("  energy   : {}", first_line(&probe["energy"]));
    if query.contains("model=drumroom") {
        println!("  sliders  : {sliders}");
    }
    println!("  diag2    : {}", first_line(&probe["diag2"]));
    println!("  horizon  : {horizon}");
    if !tuning.is_empty() {
        println!(
            "  tuning   : {tuning}{}",
            if tuning_ok { "" } else { "   <- MISSING" }
        );
    }
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
                 re-run â this will attach to it:\n  chrome --headless=new \
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
