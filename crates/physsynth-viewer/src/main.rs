//! `physsynth-viewer` — the web viewer's server, and a one-shot payload mode.
//!
//! ```text
//! physsynth-viewer [serve] [--port 8000] [--host 127.0.0.1] [--static DIR]
//! physsynth-viewer payload < params.json > payload.json
//! ```
//!
//! `serve` is `python web/server.py`'s replacement; then open `http://127.0.0.1:8000`. `payload`
//! runs one request through the payload builder with no socket, which is what the one-time
//! comparison against the Python reference drives (retirement plan §23), and what a script that
//! wants a scene's numbers without a browser should use.

use std::io::Read;
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::ExitCode;

use physsynth_viewer::server::{serve, Config};
use physsynth_viewer::simulate_to_payload;

/// The front-end's files, found relative to this crate's source so `cargo run` works from anywhere
/// in the tree. `--static` overrides it for an installed binary.
fn default_static_dir() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/static"))
}

fn usage() -> ExitCode {
    eprintln!(
        "usage: physsynth-viewer [serve] [--port N] [--host ADDR] [--static DIR]\n       \
         physsynth-viewer payload < params.json"
    );
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("payload") {
        let mut input = String::new();
        if let Err(e) = std::io::stdin().read_to_string(&mut input) {
            eprintln!("reading stdin: {e}");
            return ExitCode::FAILURE;
        }
        let params: serde_json::Value = match serde_json::from_str(&input) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("bad JSON on stdin: {e}");
                return ExitCode::FAILURE;
            }
        };
        println!("{}", simulate_to_payload(&params));
        return ExitCode::SUCCESS;
    }
    if args.first().map(String::as_str) == Some("serve") {
        args.remove(0);
    }

    let (mut port, mut host, mut static_dir) =
        (8000u16, "127.0.0.1".to_owned(), default_static_dir());
    let mut it = args.into_iter();
    while let Some(flag) = it.next() {
        let Some(value) = it.next() else {
            return usage();
        };
        match flag.as_str() {
            "--port" => match value.parse() {
                Ok(p) => port = p,
                Err(_) => return usage(),
            },
            "--host" => host = value,
            "--static" => static_dir = PathBuf::from(value),
            _ => return usage(),
        }
    }
    let static_dir = match static_dir.canonicalize() {
        Ok(d) if d.join("index.html").is_file() => d,
        _ => {
            eprintln!(
                "no index.html under {} — pass --static DIR",
                static_dir.display()
            );
            return ExitCode::FAILURE;
        }
    };
    let listener = match TcpListener::bind((host.as_str(), port)) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("cannot bind {host}:{port}: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!("physical-synthesis viewer -> http://{host}:{port}  (Ctrl-C to stop)");
    serve(listener, Config { static_dir });
    ExitCode::SUCCESS
}
