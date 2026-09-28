//! The HTTP shell over a real socket: routing, status codes, the body cap, traversal.
//!
//! `web/server.py` had no test of its own — it was 125 lines, "trivially" a router — and the one
//! thing that went wrong with it (a traversal check written as a string prefix) is exactly what a
//! trivial router gets wrong. These run the server on an ephemeral port and speak raw HTTP to it.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;

use physsynth_viewer::server::{resolve_static, serve, Config};
use serde_json::Value;

fn static_dir() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/static"))
        .canonicalize()
        .expect("web/static exists")
}

/// Start a server on an ephemeral port and return its address.
fn start() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let cfg = Config {
        static_dir: static_dir(),
    };
    std::thread::spawn(move || serve(listener, cfg));
    addr
}

/// Send raw bytes, return (status code, headers, body).
fn raw(addr: &str, request: &[u8]) -> (u16, String, Vec<u8>) {
    let mut s = TcpStream::connect(addr).unwrap();
    s.write_all(request).unwrap();
    let mut buf = Vec::new();
    s.read_to_end(&mut buf).unwrap();
    let split = buf
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .expect("a header block");
    let head = String::from_utf8_lossy(&buf[..split]).into_owned();
    let code = head.split_whitespace().nth(1).unwrap().parse().unwrap();
    (code, head, buf[split + 4..].to_vec())
}

fn post(addr: &str, body: &str) -> (u16, Value) {
    let req = format!(
        "POST /simulate HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\n\r\n{body}",
        body.len()
    );
    let (code, _, body) = raw(addr, req.as_bytes());
    (code, serde_json::from_slice(&body).unwrap())
}

#[test]
fn get_root_serves_the_front_end() {
    let addr = start();
    let (code, head, body) = raw(&addr, b"GET / HTTP/1.1\r\nHost: x\r\n\r\n");
    assert_eq!(code, 200);
    assert!(head.contains("text/html"));
    assert!(String::from_utf8_lossy(&body).contains("<select id=\"model\">"));
    let (code, head, _) = raw(&addr, b"GET /app.js?v=3 HTTP/1.1\r\n\r\n");
    assert_eq!(code, 200);
    assert!(head.contains("application/javascript"));
}

#[test]
fn head_sends_headers_without_a_body() {
    let addr = start();
    let (code, head, body) = raw(&addr, b"HEAD /style.css HTTP/1.1\r\n\r\n");
    assert_eq!(code, 200);
    assert!(head.contains("text/css"));
    assert!(body.is_empty());
}

#[test]
fn unknown_paths_and_traversals_are_404() {
    let addr = start();
    for path in [
        "/nope.js",
        "/../Cargo.toml",
        "/..%2fCargo.toml",
        "/static/../../Cargo.toml",
        "/..\\Cargo.toml",
        "//etc/passwd",
        "/C:/Windows/win.ini",
    ] {
        let req = format!("GET {path} HTTP/1.1\r\n\r\n");
        let (code, _, _) = raw(&addr, req.as_bytes());
        assert_eq!(code, 404, "{path}");
    }
    let (code, _, _) = raw(
        &addr,
        b"POST /elsewhere HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}",
    );
    assert_eq!(code, 404);
}

#[test]
fn traversal_is_refused_by_construction() {
    let dir = static_dir();
    assert!(resolve_static(&dir, "/").is_some());
    assert!(resolve_static(&dir, "/index.html").is_some());
    for bad in [
        "/../web/static/index.html",
        "/./index.html",
        "/a/../index.html",
        "",
    ] {
        assert!(resolve_static(&dir, bad).is_none(), "{bad}");
    }
}

#[test]
fn simulate_returns_a_payload_and_a_bad_param_is_still_200() {
    let addr = start();
    let (code, payload) = post(
        &addr,
        r#"{"model": "ideal", "N": 32, "audio_duration": 0.05}"#,
    );
    assert_eq!(code, 200);
    assert!(payload.get("error").is_none(), "{payload}");
    assert!(payload.get("horizon").is_some());
    let (code, payload) = post(&addr, r#"{"N": 1}"#);
    assert_eq!(code, 200); // a refused SCENE is a payload the front-end explains
    assert_eq!(payload["error"]["kind"], "param");
}

#[test]
fn malformed_requests_are_400_with_kind_request() {
    let addr = start();
    for body in ["not json", "[1, 2]", "3"] {
        let (code, payload) = post(&addr, body);
        assert_eq!(code, 400, "{body}");
        assert_eq!(payload["error"]["kind"], "request", "{body}");
    }
    let (code, _, body) = raw(&addr, b"POST /simulate HTTP/1.1\r\n\r\n");
    assert_eq!(code, 400); // no body
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["error"]["message"], "missing/oversized body");
    let big = format!(
        "POST /simulate HTTP/1.1\r\nContent-Length: {}\r\n\r\n",
        physsynth_viewer::server::MAX_BODY + 1
    );
    let (code, _, _) = raw(&addr, big.as_bytes());
    assert_eq!(code, 400); // refused on the header, before reading a byte of it
}
