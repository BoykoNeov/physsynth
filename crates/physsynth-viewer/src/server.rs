//! The localhost HTTP shell — what `web/server.py` was, on `std::net`.
//!
//! A local dev tool: no auth, bound to `127.0.0.1` by default, HTTP/1.0 with one request per
//! connection, and a thread per connection so a multi-second render on one never blocks the
//! static files on another (the reason the reference used `ThreadingHTTPServer`, advisor catch #3).
//! No physics here — it routes:
//!
//! ```text
//! GET  /                 -> static/index.html        GET /app.js, /style.css -> static assets
//! POST /simulate  {json} -> simulate_to_payload(json) -> json
//! ```
//!
//! Status codes are the reference's: a malformed *request* (no body, over 1 MiB, not JSON, not an
//! object) is a 400 with `{"error": {"kind": "request", ...}}`; a bad *parameter* is a 200 whose
//! payload says so, because that is a scene the front-end renders an explanation for. Unknown
//! paths are a plain-text 404. A panic inside a render is a 500 with a JSON body rather than a
//! dropped connection, which is the one place this is kinder than the original.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Component, Path, PathBuf};

use serde_json::{json, Value};

/// Request body cap (params are tiny; reject anything absurd).
pub const MAX_BODY: usize = 1 << 20;

/// Environment variable naming a file every accepted `/simulate` body is appended to, one JSON
/// object per line. Off unless set.
///
/// A development aid for the port: the front-end's `gatherParams` sends *every* slider, hidden
/// ones included, and a scene that reads another model's parameter is a failure this viewer has
/// shipped before. A corpus written from the test suite never sends those extra keys; a log of
/// what the browser actually sent does (retirement plan §23.7).
pub const REQUEST_LOG_ENV: &str = "PHYSSYNTH_VIEWER_REQUEST_LOG";

/// Serializes appends from the connection threads, so two bodies never interleave on one line.
static REQUEST_LOG: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn log_request(params: &Value) {
    let Some(path) = std::env::var_os(REQUEST_LOG_ENV) else {
        return;
    };
    let _guard = REQUEST_LOG.lock().unwrap_or_else(|e| e.into_inner());
    let line = format!("{params}\n");
    let written = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .and_then(|mut f| f.write_all(line.as_bytes()));
    if let Err(e) = written {
        eprintln!("  request log {}: {e}", PathBuf::from(path).display());
    }
}

/// What the handler needs.
#[derive(Clone)]
pub struct Config {
    /// The front-end's directory (`web/static`).
    pub static_dir: PathBuf,
}

/// Serve forever on `listener`.
pub fn serve(listener: TcpListener, cfg: Config) {
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let cfg = cfg.clone();
        std::thread::spawn(move || {
            let peer = stream
                .peer_addr()
                .map_or_else(|_| "?".to_owned(), |a| a.to_string());
            match handle(stream, &cfg) {
                Ok(line) => eprintln!("  {peer} {line}"),
                Err(e) => eprintln!("  {peer} connection error: {e}"),
            }
        });
    }
}

/// A parsed request.
struct Request {
    method: String,
    path: String,
    content_length: Option<usize>,
}

fn read_head(reader: &mut BufReader<&TcpStream>) -> std::io::Result<Option<Request>> {
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_owned();
    let path = parts.next().unwrap_or("/").to_owned();
    let mut content_length = None;
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h)? == 0 {
            break;
        }
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        if let Some((name, value)) = h.split_once(':') {
            if name.trim().eq_ignore_ascii_case("content-length") {
                // `int(self.headers.get("Content-Length", 0))`, a ValueError reading as 0.
                content_length = Some(value.trim().parse().unwrap_or(0));
            }
        }
    }
    Ok(Some(Request {
        method,
        path,
        content_length,
    }))
}

fn send(
    mut stream: &TcpStream,
    code: u16,
    reason: &str,
    body: &[u8],
    content_type: &str,
    head_only: bool,
) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.0 {code} {reason}\r\nServer: PhysSynthViewer/0.2\r\nContent-Type: {content_type}\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    if !head_only {
        stream.write_all(body)?;
    }
    stream.flush()
}

fn send_json(stream: &TcpStream, code: u16, reason: &str, v: &Value) -> std::io::Result<()> {
    let body = serde_json::to_vec(v).expect("a Value always serializes");
    send(
        stream,
        code,
        reason,
        &body,
        "application/json; charset=utf-8",
        false,
    )
}

fn request_error(stream: &TcpStream, message: &str) -> std::io::Result<()> {
    send_json(
        stream,
        400,
        "Bad Request",
        &json!({"error": {"kind": "request", "message": message}}),
    )
}

/// The content-type table, by extension.
fn content_type(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "application/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("ico") => "image/x-icon",
        _ => "application/octet-stream",
    }
}

/// Resolve a request path under the static dir, refusing anything that could leave it.
///
/// Only plain name components are accepted — no `..`, no root, no drive prefix — so a traversal
/// is refused by construction rather than by comparing strings after the fact.
pub fn resolve_static(static_dir: &Path, path: &str) -> Option<PathBuf> {
    let path = path.split('?').next().unwrap_or("");
    let rel = if path == "/" {
        "index.html"
    } else {
        path.trim_start_matches('/')
    };
    if rel.is_empty() || rel.contains('\\') {
        return None;
    }
    let rel = Path::new(rel);
    if !rel.components().all(|c| matches!(c, Component::Normal(_))) {
        return None;
    }
    let target = static_dir.join(rel);
    target.is_file().then_some(target)
}

fn handle(stream: TcpStream, cfg: &Config) -> std::io::Result<String> {
    let mut reader = BufReader::new(&stream);
    let Some(req) = read_head(&mut reader)? else {
        return Ok("(empty request)".into());
    };
    let route = req.path.split('?').next().unwrap_or("").to_owned();
    let summary = format!("\"{} {}\"", req.method, req.path);

    match req.method.as_str() {
        "GET" | "HEAD" => {
            let head_only = req.method == "HEAD";
            match resolve_static(&cfg.static_dir, &req.path) {
                Some(target) => {
                    let body = std::fs::read(&target)?;
                    send(&stream, 200, "OK", &body, content_type(&target), head_only)?;
                    Ok(format!("{summary} 200"))
                }
                None => {
                    send(
                        &stream,
                        404,
                        "Not Found",
                        b"not found",
                        "text/plain; charset=utf-8",
                        head_only,
                    )?;
                    Ok(format!("{summary} 404"))
                }
            }
        }
        "POST" => {
            if route != "/simulate" {
                send(
                    &stream,
                    404,
                    "Not Found",
                    b"not found",
                    "text/plain; charset=utf-8",
                    false,
                )?;
                return Ok(format!("{summary} 404"));
            }
            let length = req.content_length.unwrap_or(0);
            if length == 0 || length > MAX_BODY {
                request_error(&stream, "missing/oversized body")?;
                return Ok(format!("{summary} 400"));
            }
            let mut raw = vec![0u8; length];
            reader.read_exact(&mut raw)?;
            let params: Value = match std::str::from_utf8(&raw)
                .map_err(|e| e.to_string())
                .and_then(|s| serde_json::from_str(s).map_err(|e| e.to_string()))
            {
                Ok(v) => v,
                Err(e) => {
                    request_error(&stream, &format!("bad JSON: {e}"))?;
                    return Ok(format!("{summary} 400"));
                }
            };
            if !params.is_object() {
                request_error(&stream, "body must be a JSON object")?;
                return Ok(format!("{summary} 400"));
            }
            log_request(&params);
            let t0 = std::time::Instant::now();
            let result = std::panic::catch_unwind(|| crate::simulate_to_payload(&params));
            match result {
                Ok(payload) => {
                    send_json(&stream, 200, "OK", &payload)?;
                    Ok(format!(
                        "{summary} 200 ({:.2} s)",
                        t0.elapsed().as_secs_f64()
                    ))
                }
                Err(_) => {
                    send_json(
                        &stream,
                        500,
                        "Internal Server Error",
                        &json!({"error": {"kind": "internal",
                                          "message": "the render panicked; see the server log."}}),
                    )?;
                    Ok(format!("{summary} 500"))
                }
            }
        }
        _ => {
            send(
                &stream,
                501,
                "Not Implemented",
                b"unsupported method",
                "text/plain; charset=utf-8",
                false,
            )?;
            Ok(format!("{summary} 501"))
        }
    }
}
