use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

use coffer_cas::{Cas, SqliteCas};
use serde_json::{Value, json};

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/tmp")
            .join(format!("coffer-mcp-stdio-{}-{tag}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    fn db(&self) -> PathBuf {
        self.0.join("cas.db")
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn write_message(stdin: &mut ChildStdin, value: Value) {
    writeln!(stdin, "{value}").unwrap();
    stdin.flush().unwrap();
}

fn read_response(rx: &mpsc::Receiver<String>, id: u64) -> Value {
    for _ in 0..20 {
        let line = rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let value: Value = serde_json::from_str(&line).unwrap();
        if value.get("id").and_then(Value::as_u64) == Some(id) {
            return value;
        }
    }
    panic!("no JSON-RPC response for id {id}");
}

#[test]
fn mcp_stdio_unfold_reads_shared_sqlite_cas() {
    let dir = TempDir::new("unfold");
    let db = dir.db();
    let hash = {
        let cas = SqliteCas::open(&db).unwrap();
        let hash = cas.put(b"0123456789");
        cas.flush();
        hash.short().to_string()
    };

    let mut child = Command::new(env!("CARGO_BIN_EXE_coffer-mcp"))
        .env("COFFER_CAS_DB", &db)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let _child = ChildGuard(child);
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else {
                break;
            };
            let _ = tx.send(line);
        }
    });

    write_message(
        &mut stdin,
        json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "coffer-stdio-test", "version": "0"}
            }
        }),
    );
    let init = read_response(&rx, 1);
    assert!(init.get("result").is_some(), "{init}");

    write_message(
        &mut stdin,
        json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }),
    );

    write_message(
        &mut stdin,
        json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "coffer_unfold",
                "arguments": {
                    "hash": hash,
                    "start": 3,
                    "max_bytes": 4
                }
            }
        }),
    );
    let unfold = read_response(&rx, 2);
    let text = unfold["result"]["content"][0]["text"].as_str().unwrap();
    assert_eq!(text, "bytes 3..7 of 10 (3 before, 3 after)\n3456");
    assert_eq!(unfold["result"]["isError"], false);
}
