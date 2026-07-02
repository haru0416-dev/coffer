use std::convert::Infallible;
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

use bytes::Bytes;
use coffer_cas::read_blob;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/tmp")
            .join(format!(
                "coffer-proxy-process-{}-{tag}-{n}",
                std::process::id()
            ));
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

fn unused_addr() -> SocketAddr {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
}

fn wait_for_tcp(addr: SocketAddr) {
    for _ in 0..100 {
        if TcpStream::connect_timeout(&addr, Duration::from_millis(20)).is_ok() {
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    panic!("proxy did not listen on {addr}");
}

fn messages_request_with_tool_result(tool_text: &str) -> Vec<u8> {
    let req = serde_json::json!({
        "model": "claude-opus-4-8",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "how many records?"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "reading the data"},
                {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "cat d.json"}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": tool_text}]}
            ]}
        ]
    });
    serde_json::to_vec(&req).unwrap()
}

fn big_json_array() -> String {
    let items: Vec<String> = (0..400)
        .map(|i| format!(r#"{{"id":{i},"sub":"drivers"}}"#))
        .collect();
    format!("[{}]", items.join(","))
}

fn tool_result_text(body: &[u8]) -> String {
    let v: serde_json::Value = serde_json::from_slice(body).unwrap();
    v["messages"][2]["content"][0]["content"][0]["text"]
        .as_str()
        .unwrap()
        .to_string()
}

fn sentinel_span(rendered: &str) -> std::ops::Range<usize> {
    let start = rendered.find("<<cof:").unwrap();
    let end = start + rendered[start..].find(">>").unwrap() + ">>".len();
    start..end
}

fn sentinel_hash(sentinel: &str) -> &str {
    let rest = sentinel.strip_prefix("<<cof:").unwrap();
    let end = rest.find(|c: char| c.is_whitespace() || c == '>').unwrap();
    &rest[..end]
}

async fn spawn_recording_upstream() -> (String, mpsc::Receiver<Vec<u8>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = mpsc::channel();

    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let io = TokioIo::new(stream);
        let service = service_fn(move |req: Request<Incoming>| {
            let tx = tx.clone();
            async move {
                let body = req.into_body().collect().await.unwrap().to_bytes();
                let _ = tx.send(body.to_vec());
                Ok::<_, Infallible>(Response::new(Full::new(Bytes::from_static(
                    br#"{"ok":true}"#,
                ))))
            }
        });
        let _ = hyper::server::conn::http1::Builder::new()
            .serve_connection(io, service)
            .await;
    });

    (format!("http://{addr}"), rx)
}

#[tokio::test]
async fn proxy_binary_writes_shared_cas_recoverable_from_another_process() {
    let dir = TempDir::new("shared-cas");
    let db = dir.db();
    let listen = unused_addr();
    let (upstream, rx) = spawn_recording_upstream().await;
    let proxy = Command::new(env!("CARGO_BIN_EXE_coffer-proxy"))
        .env("COFFER_PROXY_LISTEN", listen.to_string())
        .env("COFFER_PROXY_UPSTREAM", upstream)
        .env("COFFER_PROXY_MIN", "1024")
        .env("COFFER_PROXY_MAX_BODY_MB", "64")
        .env("COFFER_CAS_DB", &db)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let _proxy = ChildGuard(proxy);
    wait_for_tcp(listen);

    let original_tool_result = big_json_array();
    let original = messages_request_with_tool_result(&original_tool_result);
    let response = reqwest::Client::new()
        .post(format!("http://{listen}/v1/messages"))
        .header("content-type", "application/json")
        .body(original)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let forwarded = rx.recv_timeout(Duration::from_secs(2)).unwrap();
    let rendered = tool_result_text(&forwarded);
    // The rewritten block carries the in-band sentinel explainer; the reversible splice
    // applies to the render that follows it (stripping doubles as the presence assert).
    let rendered = rendered
        .strip_prefix(coffer_proxy::SENTINEL_EXPLAINER)
        .expect("rewritten block must start with the sentinel explainer")
        .to_string();
    let span = sentinel_span(&rendered);
    let hash = sentinel_hash(&rendered[span.clone()]);
    let recovered = read_blob(&db, hash).unwrap().unwrap();
    let mut reconstructed = String::new();
    reconstructed.push_str(&rendered[..span.start]);
    reconstructed.push_str(std::str::from_utf8(&recovered).unwrap());
    reconstructed.push_str(&rendered[span.end..]);

    assert_eq!(
        reconstructed, original_tool_result,
        "a separate process should be able to recover the proxy binary's elided bytes from COFFER_CAS_DB"
    );
}
