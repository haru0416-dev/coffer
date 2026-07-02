//! coffer-proxy server: a local HTTP proxy. Point an agent at it
//! (`ANTHROPIC_BASE_URL=http://127.0.0.1:8788`); it compresses the `tool_result` blocks of every
//! `POST /v1/messages` request via coffer and forwards to the real API, streaming the response
//! back unchanged. Inbound is HTTP/1.1 via **hyper** (keep-alive, robust parsing); the upstream leg
//! uses reqwest, so the same path forwards to `http://` (a mock, for tests) and
//! `https://api.anthropic.com` (production, via rustls).
//!
//! Env: `COFFER_PROXY_LISTEN` (default `127.0.0.1:8788`; a non-loopback bind is refused unless
//! `COFFER_PROXY_ALLOW_PUBLIC` is set, since the proxy has no auth and replays client API keys —
//!), `COFFER_PROXY_UPSTREAM`
//! (default `https://api.anthropic.com`), `COFFER_PROXY_MIN` (min tool_result bytes to compress),
//! `COFFER_PROXY_MAX_BODY_MB` (max inbound request body MiB, default 64),
//! `COFFER_CAS_DB` (shared `SqliteCas` for the proxy + MCP-unfold reversible loop; unset -> in-memory),
//! `COFFER_CAS_SOFT_CAP_MB` (optional resident-cache warning threshold),
//! `COFFER_CAS_RESIDENT_CAP_MB` (optional hard cap on the HISTORICAL read cache only — it does NOT
//! bound this run's own offload `put`s, so proxy RSS still grows with offload volume within a run;
//! recycle the process / cap `COFFER_PROXY_MAX_BODY_MB` to bound write-path memory),
//! `COFFER_CAS_WARM_BYTES_ON_OPEN` (optional eager warm of historical CAS bytes),
//! `COFFER_CAS_TRUST_HASHES_ON_OPEN` (optional hash-only fast open),
//! `COFFER_CAS_CHECKPOINT_EVERY` (optional WAL checkpoint cadence, in blob commits),
//! `COFFER_PROXY_CAPTURE_DIR` (optional raw/proxied request artifact directory for audits).
//!
//! Fail-open: a malformed JSON payload is forwarded unchanged (the transform never panics);
//! an unreadable inbound body becomes `400`, and an upstream error becomes a `502` rather than a
//! dropped connection.

use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;
use coffer_cas::{Cas, MemoryCas, SqliteCas, SqliteConfig};
use coffer_proxy::{
    TransformKind, compress_ollama_body_kind, compress_request_body_kind,
    compress_responses_body_kind,
};
use futures_util::StreamExt;
use http_body_util::{BodyExt, Full, LengthLimitError, Limited, StreamBody, combinators::BoxBody};
use hyper::body::{Frame, Incoming};
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;
use tokio::sync::Semaphore;

const MIB: usize = 1024 * 1024;
const DEFAULT_PROXY_MAX_BODY_MB: usize = 64;
/// Target ceiling on concurrently-buffered raw request bodies; the default concurrency cap
/// is derived from this and `COFFER_PROXY_MAX_BODY_MB`.
const DEFAULT_CONCURRENCY_BUDGET_MB: usize = 1024;

/// The offload store: `SqliteCas` (shared, persistent — enables MCP unfold) or in-process `MemoryCas`.
enum Store {
    Memory(MemoryCas),
    Sqlite(Box<SqliteCas>),
}

impl Store {
    fn as_cas(&self) -> &dyn Cas {
        match self {
            Store::Memory(c) => c,
            Store::Sqlite(c) => c.as_ref(),
        }
    }
    fn kind(&self) -> &'static str {
        match self {
            Store::Memory(_) => "memory",
            Store::Sqlite(_) => "sqlite",
        }
    }
    /// Make this request's offloaded bytes durable/visible to a separate unfold process.
    fn flush(&self) {
        if let Store::Sqlite(c) = self {
            c.flush();
        }
    }

    /// A payload-free snapshot of CAS storage volume for the metrics endpoint: object count,
    /// resident/on-disk byte totals, and the durability backlog. The proxy is the *writer* in the
    /// shared-CAS write-through path, so without this an operator scraping the proxy is blind to
    /// store growth and to silent durability loss (`dropped_writes`) until the disk fills.
    /// Aggregate counts and sizes only — never any stored bytes or filesystem paths.
    fn volume_snapshot(&self) -> serde_json::Value {
        match self {
            Store::Memory(c) => serde_json::json!({
                "kind": "memory",
                "objects": c.len(),
            }),
            Store::Sqlite(c) => {
                let disk = c.disk_usage();
                serde_json::json!({
                    "kind": "sqlite",
                    "objects": c.len(),
                    "resident_bytes": c.mem_bytes(),
                    "resident_evictions": c.resident_evictions(),
                    "disk_db_bytes": disk.db_bytes,
                    "disk_wal_bytes": disk.wal_bytes,
                    "disk_shm_bytes": disk.shm_bytes,
                    "disk_total_bytes": disk.total_bytes(),
                    "durability_lag": c.durability_lag(),
                    "dropped_writes": c.dropped_writes(),
                    "persisted_blobs_this_run": c.persisted(),
                })
            }
        }
    }
}

struct Config {
    upstream: String,
    min: usize,
    max_body_bytes: usize,
    store: Store,
    client: reqwest::Client,
    capture_dir: Option<PathBuf>,
    capture_seq: AtomicU64,
    metrics: ProxyMetrics,
    /// Process-wide admission control: caps concurrent in-flight requests so a burst of
    /// large bodies cannot multiply (each ~5-8x its body as a serde DOM) into multi-GB RSS. Excess
    /// requests are shed with 503 rather than queued unboundedly.
    concurrency: Arc<Semaphore>,
    max_concurrent: usize,
}

struct ProxyMetrics {
    requests_total: AtomicU64,
    supported_requests_total: AtomicU64,
    compressed_requests_total: AtomicU64,
    passthrough_requests_total: AtomicU64,
    /// Subset of `passthrough_requests_total`: a supported endpoint whose body WAS the expected
    /// shape but no block shrank (benign — incompressible content).
    noshrink_requests_total: AtomicU64,
    /// Subset of `passthrough_requests_total`: a supported endpoint whose body was NOT the expected
    /// shape (non-JSON / missing marker / no array) — fail-open. A spike signals a shape regression.
    failopen_requests_total: AtomicU64,
    oversized_requests_total: AtomicU64,
    shed_requests_total: AtomicU64,
    body_read_errors_total: AtomicU64,
    upstream_errors_total: AtomicU64,
    raw_request_body_bytes_total: AtomicU64,
    forwarded_request_body_bytes_total: AtomicU64,
    saved_request_body_bytes_total: AtomicU64,
    // Latency attribution: the two synchronous-vs-network legs an operator needs to tell apart
    // (is the proxy slow, or is the upstream slow?). Nanosecond sums + counts so a scraper can
    // derive means; never wall-clock-sensitive in a way that affects forwarded bytes.
    compress_nanos_total: AtomicU64,
    upstream_responses_total: AtomicU64,
    upstream_nanos_total: AtomicU64,
    // Upstream status classes: a 4xx/5xx is forwarded to the client as-is (still a success at the
    // transport layer), so without these an Anthropic incident leaves `upstream_errors_total` at 0.
    upstream_4xx_total: AtomicU64,
    upstream_5xx_total: AtomicU64,
}

impl ProxyMetrics {
    fn new() -> Self {
        Self {
            requests_total: AtomicU64::new(0),
            supported_requests_total: AtomicU64::new(0),
            compressed_requests_total: AtomicU64::new(0),
            passthrough_requests_total: AtomicU64::new(0),
            noshrink_requests_total: AtomicU64::new(0),
            failopen_requests_total: AtomicU64::new(0),
            oversized_requests_total: AtomicU64::new(0),
            shed_requests_total: AtomicU64::new(0),
            body_read_errors_total: AtomicU64::new(0),
            upstream_errors_total: AtomicU64::new(0),
            raw_request_body_bytes_total: AtomicU64::new(0),
            forwarded_request_body_bytes_total: AtomicU64::new(0),
            saved_request_body_bytes_total: AtomicU64::new(0),
            compress_nanos_total: AtomicU64::new(0),
            upstream_responses_total: AtomicU64::new(0),
            upstream_nanos_total: AtomicU64::new(0),
            upstream_4xx_total: AtomicU64::new(0),
            upstream_5xx_total: AtomicU64::new(0),
        }
    }

    fn record_oversized(&self) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.oversized_requests_total
            .fetch_add(1, Ordering::Relaxed);
    }

    fn record_body_read_error(&self) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.body_read_errors_total.fetch_add(1, Ordering::Relaxed);
    }

    fn record_shed(&self) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.shed_requests_total.fetch_add(1, Ordering::Relaxed);
    }

    fn record_forwarded(
        &self,
        supported_endpoint: bool,
        raw_body_bytes: usize,
        forwarded_body_bytes: usize,
        transform: Option<TransformKind>,
    ) {
        let raw_body_bytes = raw_body_bytes as u64;
        let forwarded_body_bytes = forwarded_body_bytes as u64;
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.raw_request_body_bytes_total
            .fetch_add(raw_body_bytes, Ordering::Relaxed);
        self.forwarded_request_body_bytes_total
            .fetch_add(forwarded_body_bytes, Ordering::Relaxed);
        self.saved_request_body_bytes_total.fetch_add(
            raw_body_bytes.saturating_sub(forwarded_body_bytes),
            Ordering::Relaxed,
        );
        if supported_endpoint {
            self.supported_requests_total
                .fetch_add(1, Ordering::Relaxed);
        }
        // `compressed`/`passthrough` keep their aggregate meaning (shrank vs not); `noshrink` and
        // `failopen` subdivide passthrough so an operator can tell incompressible content (benign)
        // from an unrecognized request shape (fail-open — a likely regression). An unsupported
        // endpoint ran no transform (`None`) and is a plain passthrough.
        match transform {
            Some(TransformKind::Compressed) => {
                self.compressed_requests_total
                    .fetch_add(1, Ordering::Relaxed);
            }
            Some(TransformKind::NoShrink) => {
                self.passthrough_requests_total
                    .fetch_add(1, Ordering::Relaxed);
                self.noshrink_requests_total.fetch_add(1, Ordering::Relaxed);
            }
            Some(TransformKind::Passthrough) => {
                self.passthrough_requests_total
                    .fetch_add(1, Ordering::Relaxed);
                self.failopen_requests_total.fetch_add(1, Ordering::Relaxed);
            }
            None => {
                self.passthrough_requests_total
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn record_upstream_error(&self) {
        self.upstream_errors_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Time spent in the synchronous compression leg (parse + budget render + offload) for one
    /// request — the part attributable to coffer rather than the network.
    fn record_compress_nanos(&self, nanos: u64) {
        self.compress_nanos_total
            .fetch_add(nanos, Ordering::Relaxed);
    }

    /// An upstream response arrived (headers received). Records time-to-response-headers — the
    /// network/provider leg, excluding the streamed body — and the response's status class.
    fn record_upstream_response(&self, status: reqwest::StatusCode, nanos: u64) {
        self.upstream_responses_total
            .fetch_add(1, Ordering::Relaxed);
        self.upstream_nanos_total
            .fetch_add(nanos, Ordering::Relaxed);
        if status.is_client_error() {
            self.upstream_4xx_total.fetch_add(1, Ordering::Relaxed);
        } else if status.is_server_error() {
            self.upstream_5xx_total.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "requests_total": self.requests_total.load(Ordering::Relaxed),
            "supported_requests_total": self.supported_requests_total.load(Ordering::Relaxed),
            "compressed_requests_total": self.compressed_requests_total.load(Ordering::Relaxed),
            "passthrough_requests_total": self.passthrough_requests_total.load(Ordering::Relaxed),
            "noshrink_requests_total": self.noshrink_requests_total.load(Ordering::Relaxed),
            "failopen_requests_total": self.failopen_requests_total.load(Ordering::Relaxed),
            "oversized_requests_total": self.oversized_requests_total.load(Ordering::Relaxed),
            "shed_requests_total": self.shed_requests_total.load(Ordering::Relaxed),
            "body_read_errors_total": self.body_read_errors_total.load(Ordering::Relaxed),
            "upstream_errors_total": self.upstream_errors_total.load(Ordering::Relaxed),
            "raw_request_body_bytes_total": self.raw_request_body_bytes_total.load(Ordering::Relaxed),
            "forwarded_request_body_bytes_total": self.forwarded_request_body_bytes_total.load(Ordering::Relaxed),
            "saved_request_body_bytes_total": self.saved_request_body_bytes_total.load(Ordering::Relaxed),
            "compress_nanos_total": self.compress_nanos_total.load(Ordering::Relaxed),
            "upstream_responses_total": self.upstream_responses_total.load(Ordering::Relaxed),
            "upstream_nanos_total": self.upstream_nanos_total.load(Ordering::Relaxed),
            "upstream_4xx_total": self.upstream_4xx_total.load(Ordering::Relaxed),
            "upstream_5xx_total": self.upstream_5xx_total.load(Ordering::Relaxed),
        })
    }
}

/// Install a stderr-only, env-filtered tracing subscriber so coffer-cas durability and
/// corruption warnings reach operators. Filter precedence: `RUST_LOG`, then `COFFER_LOG`, then
/// `default_directives`. Fail-open: a bad filter falls back to the default, and `try_init` is a
/// no-op if a subscriber is already installed.
fn init_tracing(default_directives: &str) {
    let directives = std::env::var("RUST_LOG")
        .or_else(|_| std::env::var("COFFER_LOG"))
        .unwrap_or_else(|_| default_directives.to_string());
    let filter = tracing_subscriber::EnvFilter::try_new(&directives)
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default_directives));
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(filter)
        .with_target(false)
        .try_init();
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Surface coffer-cas durability warnings (the proxy is the writer in the shared-CAS
    // write-through path, so a silent persist failure can orphan an injected sentinel).
    init_tracing("info");
    let listen = env_or("COFFER_PROXY_LISTEN", "127.0.0.1:8788");
    let upstream = env_or("COFFER_PROXY_UPSTREAM", "https://api.anthropic.com");
    let min: usize = std::env::var("COFFER_PROXY_MIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1024);
    let max_body_bytes = proxy_max_body_bytes_from_env();

    let store = match std::env::var("COFFER_CAS_DB") {
        Ok(path) => {
            if let Some(parent) = std::path::Path::new(&path).parent() {
                create_private_dir_all(parent)?;
            }
            eprintln!("coffer-proxy: persisting offloads to shared CAS at {path}");
            Store::Sqlite(Box::new(SqliteCas::open_with_config(
                &path,
                &SqliteConfig {
                    soft_cap_bytes: sqlite_soft_cap_bytes_from_env(),
                    warm_bytes_on_open: sqlite_warm_bytes_on_open_from_env(),
                    trust_hashes_on_open: sqlite_trust_hashes_on_open_from_env(),
                    resident_cap_bytes: sqlite_resident_cap_bytes_from_env(),
                    checkpoint_every_blobs: sqlite_checkpoint_every_blobs_from_env(),
                },
            )?))
        }
        Err(_) => Store::Memory(MemoryCas::new()),
    };
    let capture_dir = std::env::var_os("COFFER_PROXY_CAPTURE_DIR").map(PathBuf::from);
    if let Some(dir) = &capture_dir {
        create_private_dir_all(dir)?;
        eprintln!(
            "coffer-proxy: capturing raw/proxied request artifacts under {}",
            dir.display()
        );
    }

    // A connect_timeout converts the dominant slow/unreachable-upstream failure (a black-holed
    // upstream that accepts the socket but never responds) into the existing 502 path, instead of
    // a request that awaits forever holding the inbound connection task and its buffered body.
    // Deliberately NO total `.timeout()`: it bounds the whole request through the end of the
    // response-body read (unchanged through reqwest 0.13) and would abort legitimate long-lived SSE
    // streams. `read_timeout` (a per-read idle bound, split out since 0.12.4) is intentionally unset
    // too — the connect_timeout above is the only bound we want.
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .build()?;

    let max_concurrent = max_concurrent_from_env(max_body_bytes);

    let cfg = Arc::new(Config {
        upstream,
        min,
        max_body_bytes,
        store,
        client,
        capture_dir,
        capture_seq: AtomicU64::new(0),
        metrics: ProxyMetrics::new(),
        concurrency: Arc::new(Semaphore::new(max_concurrent)),
        max_concurrent,
    });

    // Safe-by-default: the proxy has no auth and replays the client's upstream API key, so refuse a
    // non-loopback bind unless explicitly opted in.
    match classify_listen(&listen, proxy_allow_public_from_env()) {
        ListenExposure::Loopback => {}
        ListenExposure::PublicAllowed => tracing::warn!(
            listen = %listen,
            "coffer-proxy: binding a non-loopback address; the proxy has no authentication and replays client API keys upstream"
        ),
        ListenExposure::PublicRefused => anyhow::bail!(
            "refusing to bind non-loopback address {listen}: coffer-proxy has no authentication and \
             replays client API keys to the upstream. Set COFFER_PROXY_ALLOW_PUBLIC=1 to override \
             (only behind your own auth / network controls)."
        ),
    }

    let listener = TcpListener::bind(&listen).await?;
    eprintln!(
        "coffer-proxy: {listen} -> {} (compress tool_result >= {min} bytes, max request body {} bytes, max concurrent {})",
        cfg.upstream, cfg.max_body_bytes, cfg.max_concurrent
    );

    loop {
        // A per-connection accept() error (ECONNABORTED when a client drops mid-handshake, or
        // EMFILE/ENFILE under fd pressure) is recoverable: the listener stays valid. Log and keep
        // serving instead of propagating `?` out of main() and crashing every in-flight connection.
        // The short backoff avoids a busy-spin while fds are exhausted; the listener recovers as
        // existing connections close.
        let (stream, _) = match listener.accept().await {
            Ok(conn) => conn,
            Err(e) => {
                eprintln!("coffer-proxy: accept error (continuing): {e}");
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                continue;
            }
        };
        let io = TokioIo::new(stream);
        let cfg = Arc::clone(&cfg);
        tokio::spawn(async move {
            let service = service_fn(move |req| proxy(req, Arc::clone(&cfg)));
            // http1 keep-alive: serve_connection handles many requests per connection.
            if let Err(e) = hyper::server::conn::http1::Builder::new()
                .serve_connection(io, service)
                .await
            {
                eprintln!("conn error: {e}");
            }
        });
    }
}

fn create_private_dir_all(path: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

type ProxyBody = BoxBody<Bytes, std::io::Error>;

#[derive(Debug, PartialEq, Eq)]
enum BodyReadError {
    TooLarge,
    ReadFailed,
}

async fn proxy(
    req: Request<Incoming>,
    cfg: Arc<Config>,
) -> Result<Response<ProxyBody>, Infallible> {
    let (parts, body) = req.into_parts();
    if parts.method == Method::GET && parts.uri.path() == "/_coffer/health" {
        return Ok(health_response(&cfg));
    }
    if parts.method == Method::GET && parts.uri.path() == "/_coffer/metrics" {
        return Ok(metrics_response(&cfg));
    }

    // Admission control: bound concurrent in-flight requests so a burst of large bodies
    // cannot multiply into multi-GB RSS. Shed with 503 (rather than queue unboundedly) on
    // contention; the permit is held across the body buffer + compression + upstream send, then
    // released when this function returns (before the response body streams, which holds no buffers).
    let _permit = match cfg.concurrency.try_acquire() {
        Ok(permit) => permit,
        Err(_) => {
            cfg.metrics.record_shed();
            return Ok(shed_response());
        }
    };

    let body = body.map_err(std::io::Error::other).boxed();
    let body_bytes = match collect_limited_body(body, cfg.max_body_bytes).await {
        Ok(bytes) => bytes,
        Err(BodyReadError::TooLarge) => {
            cfg.metrics.record_oversized();
            return Ok(status_response(
                StatusCode::PAYLOAD_TOO_LARGE,
                &format!(
                    "coffer-proxy request body exceeds COFFER_PROXY_MAX_BODY_MB limit ({} bytes)",
                    cfg.max_body_bytes
                ),
            ));
        }
        Err(BodyReadError::ReadFailed) => {
            cfg.metrics.record_body_read_error();
            return Ok(status_response(
                StatusCode::BAD_REQUEST,
                "coffer-proxy could not read the inbound request body",
            ));
        }
    };

    // Compress tool output on the two supported endpoints, fail-open everywhere else:
    //  - Anthropic Messages  (`POST /v1/messages`)        → tool_result blocks
    //  - OpenAI  Responses   (`POST …/responses`, codex)  → *_call_output items
    // (count_tokens etc. are left alone.)
    let path = parts.uri.path();
    let supported_endpoint =
        path == "/v1/messages" || path.ends_with("/responses") || path == "/api/chat";
    let compress_start = std::time::Instant::now();
    let (new_body, transform_kind): (Bytes, Option<TransformKind>) = if path == "/v1/messages" {
        let (out, kind) = compress_then_flush(&body_bytes, &cfg, compress_request_body_kind);
        (Bytes::from(out), Some(kind))
    } else if path.ends_with("/responses") {
        let (out, kind) = compress_then_flush(&body_bytes, &cfg, compress_responses_body_kind);
        (Bytes::from(out), Some(kind))
    } else if path == "/api/chat" {
        let (out, kind) = compress_then_flush(&body_bytes, &cfg, compress_ollama_body_kind);
        (Bytes::from(out), Some(kind))
    } else {
        // Passthrough: forward the original bytes without a deep copy — a `Bytes` clone is a
        // refcount bump, not a memcpy of up to COFFER_PROXY_MAX_BODY_MB. Byte-identical to the
        // inbound body, so fail-open behavior is unchanged. No transform ran (`None`).
        (body_bytes.clone(), None)
    };
    cfg.metrics
        .record_compress_nanos(nanos_since(compress_start));
    cfg.metrics.record_forwarded(
        supported_endpoint,
        body_bytes.len(),
        new_body.len(),
        transform_kind,
    );
    capture_request_artifacts(&cfg, path, &body_bytes, &new_body);

    // Forward to the upstream (reqwest handles http and https uniformly).
    let pq = parts.uri.path_and_query().map_or("/", |pq| pq.as_str());
    let url = format!("{}{}", cfg.upstream.trim_end_matches('/'), pq);
    let mut headers = parts.headers.clone();
    for h in [
        hyper::header::HOST,
        hyper::header::CONTENT_LENGTH,
        hyper::header::CONNECTION,
        hyper::header::ACCEPT_ENCODING,
    ] {
        headers.remove(h);
    }

    let upstream_start = std::time::Instant::now();
    match cfg
        .client
        .request(parts.method, &url)
        .headers(headers)
        .body(new_body)
        .send()
        .await
    {
        Ok(resp) => {
            // Time-to-response-headers (not the streamed body) is the upstream/provider leg. A 4xx
            // or 5xx is still forwarded to the client verbatim, but counted here so an upstream
            // incident is visible rather than indistinguishable from a healthy success.
            cfg.metrics
                .record_upstream_response(resp.status(), nanos_since(upstream_start));
            Ok(forward_response(resp))
        }
        Err(e) => {
            cfg.metrics.record_upstream_error();
            Ok(error_response(&format!("coffer-proxy upstream error: {e}")))
        }
    }
}

/// Saturating nanoseconds elapsed since `start`, for the latency counters. `u64` nanos overflow
/// only after ~584 years, so the clamp is purely defensive.
fn nanos_since(start: std::time::Instant) -> u64 {
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

async fn collect_limited_body(
    body: ProxyBody,
    max_body_bytes: usize,
) -> Result<Bytes, BodyReadError> {
    match Limited::new(body, max_body_bytes).collect().await {
        Ok(collected) => Ok(collected.to_bytes()),
        Err(error) if error.downcast_ref::<LengthLimitError>().is_some() => {
            Err(BodyReadError::TooLarge)
        }
        Err(_) => Err(BodyReadError::ReadFailed),
    }
}

/// Turn the upstream reqwest response into a streamed hyper response (status + headers preserved;
/// body streamed chunk-by-chunk so SSE passes through live).
fn forward_response(resp: reqwest::Response) -> Response<ProxyBody> {
    let mut builder = Response::builder().status(resp.status());
    for (k, v) in resp.headers() {
        // hop-by-hop / framing headers are re-derived by hyper.
        if matches!(
            k,
            &hyper::header::CONTENT_LENGTH
                | &hyper::header::TRANSFER_ENCODING
                | &hyper::header::CONNECTION
        ) {
            continue;
        }
        builder = builder.header(k, v);
    }
    let stream = resp
        .bytes_stream()
        .map(|res| res.map(Frame::data).map_err(std::io::Error::other));
    let body = BodyExt::boxed(StreamBody::new(stream));
    builder.body(body).expect("response builds")
}

fn error_response(msg: &str) -> Response<ProxyBody> {
    status_response(StatusCode::BAD_GATEWAY, msg)
}

/// Load-shed response when the concurrency cap is saturated: a clean 503 + `Retry-After`
/// so the client retries, never a corrupted or dropped forward.
fn shed_response() -> Response<ProxyBody> {
    let body = Full::new(Bytes::from_static(
        b"coffer-proxy: too many concurrent requests (COFFER_PROXY_MAX_CONCURRENT_REQUESTS); retry shortly",
    ))
    .map_err(|never| match never {})
    .boxed();
    Response::builder()
        .status(StatusCode::SERVICE_UNAVAILABLE)
        .header(hyper::header::RETRY_AFTER, "1")
        .body(body)
        .expect("shed response builds")
}

fn status_response(status: StatusCode, msg: &str) -> Response<ProxyBody> {
    let body = Full::new(Bytes::from(msg.to_owned()))
        .map_err(|never| match never {})
        .boxed();
    Response::builder()
        .status(status)
        .body(body)
        .expect("status response builds")
}

fn health_response(cfg: &Config) -> Response<ProxyBody> {
    let mut body = serde_json::to_vec(&serde_json::json!({
        "ok": true,
        "service": "coffer-proxy",
        "version": env!("CARGO_PKG_VERSION"),
        "store": cfg.store.kind(),
        "compress_min_bytes": cfg.min,
        "max_body_bytes": cfg.max_body_bytes,
        "max_concurrent_requests": cfg.max_concurrent,
        "capture_enabled": cfg.capture_dir.is_some(),
        "upstream_configured": !cfg.upstream.trim().is_empty(),
    }))
    .expect("health JSON renders");
    body.push(b'\n');
    let body = Full::new(Bytes::from(body))
        .map_err(|never| match never {})
        .boxed();
    Response::builder()
        .status(StatusCode::OK)
        .header(hyper::header::CONTENT_TYPE, "application/json")
        .body(body)
        .expect("health response builds")
}

fn metrics_response(cfg: &Config) -> Response<ProxyBody> {
    let mut body = serde_json::to_vec(&serde_json::json!({
        "ok": true,
        "service": "coffer-proxy",
        "version": env!("CARGO_PKG_VERSION"),
        "store": cfg.store.kind(),
        "counters": cfg.metrics.snapshot(),
        "store_volume": cfg.store.volume_snapshot(),
        "in_flight_requests": cfg.max_concurrent.saturating_sub(cfg.concurrency.available_permits()),
        "config": {
            "compress_min_bytes": cfg.min,
            "max_body_bytes": cfg.max_body_bytes,
            "max_concurrent_requests": cfg.max_concurrent,
            "capture_enabled": cfg.capture_dir.is_some(),
            "upstream_configured": !cfg.upstream.trim().is_empty(),
        },
        "claim_boundary": [
            "This endpoint exposes aggregate proxy counters only.",
            "It does not include request bodies, response bodies, auth headers, CAS contents, or local filesystem paths.",
            "Counters reset when the proxy process restarts."
        ],
    }))
    .expect("metrics JSON renders");
    body.push(b'\n');
    let body = Full::new(Bytes::from(body))
        .map_err(|never| match never {})
        .boxed();
    Response::builder()
        .status(StatusCode::OK)
        .header(hyper::header::CONTENT_TYPE, "application/json")
        .body(body)
        .expect("metrics response builds")
}

fn capture_request_artifacts(cfg: &Config, path: &str, raw: &[u8], proxied: &[u8]) {
    let Some(dir) = &cfg.capture_dir else {
        return;
    };
    let seq = cfg.capture_seq.fetch_add(1, Ordering::Relaxed) + 1;
    let stem = format!("{seq:06}-{}", capture_path_slug(path));
    let raw_path = dir.join(format!("{stem}.raw-request.json"));
    let proxied_path = dir.join(format!("{stem}.proxied-request.json"));
    let metadata_path = dir.join(format!("{stem}.metadata.json"));

    let metadata = serde_json::json!({
        "sequence": seq,
        "path": path,
        "raw_request": raw_path.file_name().and_then(|s| s.to_str()).unwrap_or(""),
        "proxied_request": proxied_path.file_name().and_then(|s| s.to_str()).unwrap_or(""),
        "raw_body_bytes": raw.len(),
        "proxied_body_bytes": proxied.len(),
        "request_body_delta": (proxied.len() as i64) - (raw.len() as i64),
        "proxied_smaller": proxied.len() < raw.len(),
        "unix_time_seconds": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs()),
        "claim_boundary": [
            "This is an opt-in local request artifact capture.",
            "Files may contain sensitive provider payloads; secure or delete them after analysis.",
            "A smaller proxied request proves transport shrinkage, not provider billing savings."
        ],
    });

    if let Err(error) = std::fs::write(&raw_path, raw) {
        eprintln!(
            "coffer-proxy capture: write {} failed: {error}",
            raw_path.display()
        );
        return;
    }
    if let Err(error) = std::fs::write(&proxied_path, proxied) {
        eprintln!(
            "coffer-proxy capture: write {} failed: {error}",
            proxied_path.display()
        );
        return;
    }
    match serde_json::to_vec_pretty(&metadata) {
        Ok(mut bytes) => {
            bytes.push(b'\n');
            if let Err(error) = std::fs::write(&metadata_path, bytes) {
                eprintln!(
                    "coffer-proxy capture: write {} failed: {error}",
                    metadata_path.display()
                );
            }
        }
        Err(error) => eprintln!("coffer-proxy capture: metadata render failed: {error}"),
    }
}

fn capture_path_slug(path: &str) -> String {
    let slug: String = path
        .trim_matches('/')
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if slug.is_empty() {
        "root".to_string()
    } else {
        slug
    }
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

/// How a configured listen address exposes the (unauthenticated, key-replaying) proxy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ListenExposure {
    /// A loopback address — safe to bind without an opt-in.
    Loopback,
    /// A non-loopback address WITH the explicit public opt-in set (bind, but warn).
    PublicAllowed,
    /// A non-loopback address WITHOUT the opt-in — refuse to bind.
    PublicRefused,
}

/// The host portion of a `host:port` listen string, handling bracketed IPv6 (`[::1]:8788`).
fn listen_host(listen: &str) -> &str {
    if let Some(rest) = listen.strip_prefix('[') {
        return rest.split(']').next().unwrap_or(rest);
    }
    listen.rsplit_once(':').map_or(listen, |(host, _)| host)
}

/// Classify a listen address: loopback is always allowed; a non-loopback bind (routable IP,
/// `0.0.0.0`/`[::]`, or an unrecognized host) is refused unless `allow_public` is set, because the
/// proxy has no auth and replays client API keys upstream.
fn classify_listen(listen: &str, allow_public: bool) -> ListenExposure {
    let host = listen_host(listen);
    let is_loopback = host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback());
    if is_loopback {
        ListenExposure::Loopback
    } else if allow_public {
        ListenExposure::PublicAllowed
    } else {
        ListenExposure::PublicRefused
    }
}

fn proxy_allow_public_from_env() -> bool {
    matches!(
        std::env::var("COFFER_PROXY_ALLOW_PUBLIC")
            .ok()
            .map(|v| v.trim().to_ascii_lowercase())
            .as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

fn proxy_max_body_bytes_from_env() -> usize {
    let raw = std::env::var("COFFER_PROXY_MAX_BODY_MB").ok();
    proxy_max_body_bytes_from_value(raw.as_deref())
        .unwrap_or_else(|| DEFAULT_PROXY_MAX_BODY_MB.saturating_mul(MIB))
}

fn proxy_max_body_bytes_from_value(raw: Option<&str>) -> Option<usize> {
    let mb = raw?.trim().parse::<usize>().ok()?;
    (mb > 0).then(|| mb.saturating_mul(MIB))
}

fn max_concurrent_from_env(max_body_bytes: usize) -> usize {
    std::env::var("COFFER_PROXY_MAX_CONCURRENT_REQUESTS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| default_max_concurrent(max_body_bytes))
}

/// Default concurrency cap tied to the body cap so the worst-case *raw* buffered bytes stay near
/// `DEFAULT_CONCURRENCY_BUDGET_MB` regardless of `COFFER_PROXY_MAX_BODY_MB` (the transient serde DOM
/// on top is caught by the systemd `MemoryMax` backstop). Clamped to a usable range.
fn default_max_concurrent(max_body_bytes: usize) -> usize {
    let body_mb = max_body_bytes.div_ceil(MIB).max(1);
    (DEFAULT_CONCURRENCY_BUDGET_MB / body_mb).clamp(1, 256)
}

fn sqlite_soft_cap_bytes_from_env() -> Option<usize> {
    let raw = std::env::var("COFFER_CAS_SOFT_CAP_MB").ok();
    sqlite_soft_cap_bytes_from_value(raw.as_deref())
}

fn sqlite_soft_cap_bytes_from_value(raw: Option<&str>) -> Option<usize> {
    let mb = raw?.trim().parse::<usize>().ok()?;
    (mb > 0).then(|| mb.saturating_mul(1024 * 1024))
}

fn sqlite_resident_cap_bytes_from_env() -> Option<usize> {
    let raw = std::env::var("COFFER_CAS_RESIDENT_CAP_MB").ok();
    sqlite_resident_cap_bytes_from_value(raw.as_deref())
}

fn sqlite_resident_cap_bytes_from_value(raw: Option<&str>) -> Option<usize> {
    let mb = raw?.trim().parse::<usize>().ok()?;
    (mb > 0).then(|| mb.saturating_mul(1024 * 1024))
}

fn sqlite_warm_bytes_on_open_from_env() -> bool {
    let raw = std::env::var("COFFER_CAS_WARM_BYTES_ON_OPEN").ok();
    sqlite_warm_bytes_on_open_from_value(raw.as_deref())
}

fn sqlite_warm_bytes_on_open_from_value(raw: Option<&str>) -> bool {
    matches!(
        raw.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

fn sqlite_trust_hashes_on_open_from_env() -> bool {
    let raw = std::env::var("COFFER_CAS_TRUST_HASHES_ON_OPEN").ok();
    sqlite_trust_hashes_on_open_from_value(raw.as_deref())
}

fn sqlite_trust_hashes_on_open_from_value(raw: Option<&str>) -> bool {
    matches!(
        raw.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

fn sqlite_checkpoint_every_blobs_from_env() -> Option<usize> {
    let raw = std::env::var("COFFER_CAS_CHECKPOINT_EVERY").ok();
    sqlite_checkpoint_every_blobs_from_value(raw.as_deref())
}

fn sqlite_checkpoint_every_blobs_from_value(raw: Option<&str>) -> Option<usize> {
    let every = raw?.trim().parse::<usize>().ok()?;
    (every > 0).then_some(every)
}

/// Apply a tool-output transform, flush offloads for cross-process unfold, and (under
/// `COFFER_PROXY_DEBUG`) log a content-free structural summary.
/// One of the two body transforms: returns the (possibly unchanged) bytes and why (see
/// [`TransformKind`]).
type TransformFn = fn(&[u8], &dyn Cas, usize) -> (Vec<u8>, TransformKind);

fn compress_then_flush(
    body: &[u8],
    cfg: &Config,
    transform: TransformFn,
) -> (Vec<u8>, TransformKind) {
    let (out, kind) = transform(body, cfg.store.as_cas(), cfg.min);
    cfg.store.flush();
    if std::env::var_os("COFFER_PROXY_DEBUG").is_some() {
        eprintln!(
            "[debug] {} -> {} bytes ({kind:?}) | {}",
            body.len(),
            out.len(),
            summarize_structure(body)
        );
    }
    (out, kind)
}

/// A content-free structural summary of a Messages body for `COFFER_PROXY_DEBUG` (block types and
/// byte lengths only — never any message text), to diagnose whether a client's `tool_result` shape
/// matches what the transform compresses.
fn summarize_structure(body: &[u8]) -> String {
    use serde_json::Value;
    let Ok(v) = serde_json::from_slice::<Value>(body) else {
        return "not-json".into();
    };
    let Some(msgs) = v.get("messages").and_then(Value::as_array) else {
        return summarize_responses(&v);
    };
    let mut parts = Vec::new();
    for m in msgs {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("?");
        match m.get("content") {
            Some(Value::String(s)) => parts.push(format!("{role}:str({})", s.len())),
            Some(Value::Array(a)) => {
                let blocks: Vec<String> = a
                    .iter()
                    .map(|b| {
                        let t = b.get("type").and_then(Value::as_str).unwrap_or("?");
                        if t != "tool_result" {
                            return t.to_string();
                        }
                        let form = match b.get("content") {
                            Some(Value::String(s)) => format!("str({})", s.len()),
                            Some(Value::Array(ca)) => {
                                let inner: Vec<String> = ca
                                    .iter()
                                    .map(|c| {
                                        let ct =
                                            c.get("type").and_then(Value::as_str).unwrap_or("?");
                                        let len = c
                                            .get("text")
                                            .and_then(Value::as_str)
                                            .map_or(0, str::len);
                                        format!("{ct}({len})")
                                    })
                                    .collect();
                                format!("arr[{}]", inner.join(","))
                            }
                            _ => "?".into(),
                        };
                        format!("tool_result:{form}")
                    })
                    .collect();
                parts.push(format!("{role}:[{}]", blocks.join(",")));
            }
            _ => parts.push(format!("{role}:?")),
        }
    }
    format!("msgs={} {}", msgs.len(), parts.join(" | "))
}

/// Responses-format counterpart of [`summarize_structure`]: input-item types + tool-output byte
/// lengths only (never content).
fn summarize_responses(v: &serde_json::Value) -> String {
    use serde_json::Value;
    let Some(items) = v.get("input").and_then(Value::as_array) else {
        return "no-messages/input".into();
    };
    let parts: Vec<String> = items
        .iter()
        .map(|it| {
            let t = it.get("type").and_then(Value::as_str).unwrap_or("?");
            if t == "message" {
                let role = it.get("role").and_then(Value::as_str).unwrap_or("?");
                let lens: Vec<String> = it
                    .get("content")
                    .and_then(Value::as_array)
                    .map(|cs| {
                        cs.iter()
                            .map(|c| {
                                let ct = c.get("type").and_then(Value::as_str).unwrap_or("?");
                                format!(
                                    "{ct}({})",
                                    c.get("text").and_then(Value::as_str).map_or(0, str::len)
                                )
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                format!("msg:{role}[{}]", lens.join(","))
            } else if t.ends_with("_call_output") {
                format!(
                    "{t}({})",
                    it.get("output").and_then(Value::as_str).map_or(0, str::len)
                )
            } else {
                t.to_string()
            }
        })
        .collect();
    format!("responses input={} {}", items.len(), parts.join(" | "))
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use bytes::Bytes;
    use http_body_util::{BodyExt, Full, StreamBody};
    use hyper::body::Frame;
    use hyper::service::service_fn;
    use hyper::{Request, Response, StatusCode};
    use hyper_util::rt::TokioIo;
    use tokio::net::TcpListener;

    #[test]
    fn classify_listen_allows_loopback_and_gates_public() {
        use super::ListenExposure::{Loopback, PublicAllowed, PublicRefused};
        use super::classify_listen;
        // loopback hosts/IPs (v4 + v6) are always safe to bind
        assert_eq!(classify_listen("127.0.0.1:8788", false), Loopback);
        assert_eq!(classify_listen("localhost:8788", false), Loopback);
        assert_eq!(classify_listen("[::1]:8788", false), Loopback);
        // non-loopback (routable IP, all-interfaces, or unknown host) is refused without opt-in
        assert_eq!(classify_listen("0.0.0.0:8788", false), PublicRefused);
        assert_eq!(classify_listen("192.168.1.5:8788", false), PublicRefused);
        assert_eq!(classify_listen("[::]:8788", false), PublicRefused);
        assert_eq!(
            classify_listen("registry.internal:8788", false),
            PublicRefused
        );
        // explicit opt-in turns refusal into an allowed (warned) public bind; loopback never needs it
        assert_eq!(classify_listen("0.0.0.0:8788", true), PublicAllowed);
        assert_eq!(classify_listen("127.0.0.1:8788", true), Loopback);
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

    fn boxed_test_body(bytes: &'static [u8]) -> super::ProxyBody {
        Full::new(Bytes::from_static(bytes))
            .map_err(|never| -> std::io::Error { match never {} })
            .boxed()
    }

    fn failing_test_body() -> super::ProxyBody {
        let stream = futures_util::stream::once(async {
            Err::<Frame<Bytes>, std::io::Error>(std::io::Error::other("body read failed"))
        });
        StreamBody::new(stream).boxed()
    }

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("coffer-proxy-{}-{tag}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    async fn spawn_proxy_once(upstream: String, max_body_bytes: usize) -> String {
        spawn_proxy_once_with_capture(upstream, max_body_bytes, None).await
    }

    async fn spawn_proxy_once_with_capture(
        upstream: String,
        max_body_bytes: usize,
        capture_dir: Option<std::path::PathBuf>,
    ) -> String {
        spawn_proxy_cfg(upstream, max_body_bytes, capture_dir, 1024).await
    }

    async fn spawn_proxy_cfg(
        upstream: String,
        max_body_bytes: usize,
        capture_dir: Option<std::path::PathBuf>,
        max_concurrent: usize,
    ) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let cfg = Arc::new(super::Config {
            upstream,
            min: 1024,
            max_body_bytes,
            store: super::Store::Memory(coffer_cas::MemoryCas::new()),
            client: reqwest::Client::new(),
            capture_dir,
            capture_seq: std::sync::atomic::AtomicU64::new(0),
            metrics: super::ProxyMetrics::new(),
            concurrency: std::sync::Arc::new(tokio::sync::Semaphore::new(max_concurrent)),
            max_concurrent,
        });

        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let io = TokioIo::new(stream);
                let cfg = Arc::clone(&cfg);
                tokio::spawn(async move {
                    let service = service_fn(move |req| super::proxy(req, Arc::clone(&cfg)));
                    let _ = hyper::server::conn::http1::Builder::new()
                        .serve_connection(io, service)
                        .await;
                });
            }
        });

        format!("http://{addr}")
    }

    async fn spawn_recording_upstream() -> (String, std::sync::mpsc::Receiver<Vec<u8>>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = std::sync::mpsc::channel();

        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let io = TokioIo::new(stream);
            let service = service_fn(move |req: Request<super::Incoming>| {
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

    async fn spawn_counting_upstream(count: Arc<AtomicUsize>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                count.fetch_add(1, Ordering::Relaxed);
                let io = TokioIo::new(stream);
                tokio::spawn(async move {
                    let service = service_fn(|_req: Request<super::Incoming>| async {
                        Ok::<_, Infallible>(Response::new(Full::new(Bytes::from_static(
                            br#"{"ok":true}"#,
                        ))))
                    });
                    let _ = hyper::server::conn::http1::Builder::new()
                        .serve_connection(io, service)
                        .await;
                });
            }
        });

        format!("http://{addr}")
    }

    #[tokio::test]
    async fn http_proxy_compresses_messages_request_before_forwarding() {
        let (upstream, rx) = spawn_recording_upstream().await;
        let proxy = spawn_proxy_once(upstream, 64 * super::MIB).await;
        let original = messages_request_with_tool_result(&big_json_array());

        let response = reqwest::Client::new()
            .post(format!("{proxy}/v1/messages"))
            .header("content-type", "application/json")
            .body(original.clone())
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let forwarded = rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(
            forwarded.len() < original.len(),
            "forwarded request should be compressed: {} vs {}",
            forwarded.len(),
            original.len()
        );
        let v: serde_json::Value = serde_json::from_slice(&forwarded).unwrap();
        let text = v["messages"][2]["content"][0]["content"][0]["text"]
            .as_str()
            .unwrap();
        assert!(text.contains("<<cof:"), "{text}");
    }

    #[tokio::test]
    async fn http_proxy_rejects_oversized_body_before_forwarding() {
        let forwards = Arc::new(AtomicUsize::new(0));
        let upstream = spawn_counting_upstream(Arc::clone(&forwards)).await;
        let proxy = spawn_proxy_once(upstream, 5).await;

        let response = reqwest::Client::new()
            .post(format!("{proxy}/v1/messages"))
            .body("abcdef")
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let text = response.text().await.unwrap();
        assert!(text.contains("COFFER_PROXY_MAX_BODY_MB"), "{text}");
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(
            forwards.load(Ordering::Relaxed),
            0,
            "oversized request must not reach upstream"
        );

        let metrics = reqwest::Client::new()
            .get(format!("{proxy}/_coffer/metrics"))
            .send()
            .await
            .unwrap();
        assert_eq!(metrics.status(), StatusCode::OK);
        let body: serde_json::Value = metrics.json().await.unwrap();
        assert_eq!(body["counters"]["requests_total"], 1);
        assert_eq!(body["counters"]["oversized_requests_total"], 1);
        assert_eq!(body["counters"]["body_read_errors_total"], 0);
        assert_eq!(body["counters"]["forwarded_request_body_bytes_total"], 0);
    }

    #[tokio::test]
    async fn health_endpoint_reports_local_config_without_forwarding() {
        let forwards = Arc::new(AtomicUsize::new(0));
        let upstream = spawn_counting_upstream(Arc::clone(&forwards)).await;
        let proxy = spawn_proxy_once(upstream, 64 * super::MIB).await;

        let response = reqwest::Client::new()
            .get(format!("{proxy}/_coffer/health"))
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get(hyper::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("application/json")
        );
        let body: serde_json::Value = response.json().await.unwrap();
        assert_eq!(body["ok"], true);
        assert_eq!(body["service"], "coffer-proxy");
        assert_eq!(body["store"], "memory");
        assert_eq!(body["compress_min_bytes"], 1024);
        assert_eq!(body["max_body_bytes"], 64 * super::MIB);
        assert_eq!(body["capture_enabled"], false);
        assert_eq!(body["upstream_configured"], true);
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(
            forwards.load(Ordering::Relaxed),
            0,
            "health endpoint must not reach upstream"
        );
    }

    #[tokio::test]
    async fn metrics_endpoint_reports_payload_free_runtime_counters_without_forwarding() {
        let forwards = Arc::new(AtomicUsize::new(0));
        let upstream = spawn_counting_upstream(Arc::clone(&forwards)).await;
        let proxy = spawn_proxy_once(upstream, 64 * super::MIB).await;
        let original = messages_request_with_tool_result(&big_json_array());

        let initial = reqwest::Client::new()
            .get(format!("{proxy}/_coffer/metrics"))
            .send()
            .await
            .unwrap();
        assert_eq!(initial.status(), StatusCode::OK);
        let body: serde_json::Value = initial.json().await.unwrap();
        assert_eq!(body["ok"], true);
        assert_eq!(body["service"], "coffer-proxy");
        assert_eq!(body["counters"]["requests_total"], 0);
        assert_eq!(body["counters"]["compressed_requests_total"], 0);
        assert_eq!(body["counters"]["body_read_errors_total"], 0);
        assert!(
            body["claim_boundary"]
                .as_array()
                .is_some_and(|items| !items.is_empty()),
            "{body}"
        );
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(
            forwards.load(Ordering::Relaxed),
            0,
            "metrics endpoint must not reach upstream"
        );

        let response = reqwest::Client::new()
            .post(format!("{proxy}/v1/messages"))
            .header("content-type", "application/json")
            .body(original.clone())
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let metrics = reqwest::Client::new()
            .get(format!("{proxy}/_coffer/metrics"))
            .send()
            .await
            .unwrap();
        assert_eq!(metrics.status(), StatusCode::OK);
        let body: serde_json::Value = metrics.json().await.unwrap();
        let counters = &body["counters"];
        assert_eq!(counters["requests_total"], 1);
        assert_eq!(counters["supported_requests_total"], 1);
        assert_eq!(counters["compressed_requests_total"], 1);
        assert_eq!(counters["passthrough_requests_total"], 0);
        assert_eq!(counters["oversized_requests_total"], 0);
        assert_eq!(counters["body_read_errors_total"], 0);
        assert_eq!(counters["upstream_errors_total"], 0);
        assert_eq!(counters["raw_request_body_bytes_total"], original.len());
        assert!(
            counters["forwarded_request_body_bytes_total"]
                .as_u64()
                .is_some_and(|n| n < original.len() as u64),
            "{body}"
        );
        assert!(
            counters["saved_request_body_bytes_total"]
                .as_u64()
                .is_some_and(|n| n > 0),
            "{body}"
        );
        // Latency + upstream-status counters: one 200 response was received and timed.
        assert_eq!(counters["upstream_responses_total"], 1);
        assert_eq!(counters["upstream_4xx_total"], 0);
        assert_eq!(counters["upstream_5xx_total"], 0);
        assert!(
            counters["compress_nanos_total"].as_u64().is_some(),
            "{body}"
        );
        assert!(
            counters["upstream_nanos_total"].as_u64().is_some(),
            "{body}"
        );
        // Payload-free store-volume snapshot is present (in-memory store in this test).
        assert_eq!(body["store_volume"]["kind"], "memory");
        assert!(
            body["store_volume"]["objects"]
                .as_u64()
                .is_some_and(|n| n >= 1),
            "{body}"
        );
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(
            forwards.load(Ordering::Relaxed),
            1,
            "only the proxied model request should reach upstream"
        );
    }

    #[tokio::test]
    async fn saturated_concurrency_sheds_with_503_without_forwarding() {
        let forwards = Arc::new(AtomicUsize::new(0));
        let upstream = spawn_counting_upstream(Arc::clone(&forwards)).await;
        // Zero permits: every request is shed before the body is read or forwarded.
        let proxy = spawn_proxy_cfg(upstream, 64 * super::MIB, None, 0).await;

        let response = reqwest::Client::new()
            .post(format!("{proxy}/v1/messages"))
            .header("content-type", "application/json")
            .body(messages_request_with_tool_result(&big_json_array()))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(response.headers().contains_key(hyper::header::RETRY_AFTER));

        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(
            forwards.load(Ordering::Relaxed),
            0,
            "a shed request must not reach upstream"
        );

        // Health/metrics bypass admission control and still work; the shed is counted.
        let metrics: serde_json::Value = reqwest::Client::new()
            .get(format!("{proxy}/_coffer/metrics"))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(metrics["counters"]["shed_requests_total"], 1);
        assert_eq!(metrics["config"]["max_concurrent_requests"], 0);
    }

    #[tokio::test]
    async fn http_proxy_capture_dir_writes_raw_proxied_and_metadata_artifacts() {
        let capture_dir = temp_dir("capture-dir");
        let (upstream, rx) = spawn_recording_upstream().await;
        let proxy =
            spawn_proxy_once_with_capture(upstream, 64 * super::MIB, Some(capture_dir.clone()))
                .await;
        let original = messages_request_with_tool_result(&big_json_array());

        let response = reqwest::Client::new()
            .post(format!("{proxy}/v1/messages"))
            .header("content-type", "application/json")
            .body(original.clone())
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let forwarded = rx.recv_timeout(Duration::from_secs(2)).unwrap();
        let raw_path = capture_dir.join("000001-v1_messages.raw-request.json");
        let proxied_path = capture_dir.join("000001-v1_messages.proxied-request.json");
        let metadata_path = capture_dir.join("000001-v1_messages.metadata.json");

        assert_eq!(std::fs::read(&raw_path).unwrap(), original);
        assert_eq!(std::fs::read(&proxied_path).unwrap(), forwarded);
        let metadata: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&metadata_path).unwrap()).unwrap();
        assert_eq!(metadata["path"], "/v1/messages");
        assert_eq!(metadata["raw_body_bytes"], original.len());
        assert_eq!(metadata["proxied_body_bytes"], forwarded.len());
        assert_eq!(metadata["proxied_smaller"], true);
        assert!(
            metadata["claim_boundary"]
                .as_array()
                .is_some_and(|items| !items.is_empty()),
            "{metadata}"
        );

        let _ = std::fs::remove_dir_all(capture_dir);
    }

    #[tokio::test]
    async fn collect_limited_body_accepts_body_at_limit() {
        let got = super::collect_limited_body(boxed_test_body(b"abcdef"), 6)
            .await
            .unwrap();
        assert_eq!(got, Bytes::from_static(b"abcdef"));
    }

    #[tokio::test]
    async fn collect_limited_body_rejects_oversized_body() {
        let err = super::collect_limited_body(boxed_test_body(b"abcdef"), 5)
            .await
            .unwrap_err();
        assert_eq!(err, super::BodyReadError::TooLarge);
    }

    #[tokio::test]
    async fn collect_limited_body_rejects_unreadable_body() {
        let err = super::collect_limited_body(failing_test_body(), 64)
            .await
            .unwrap_err();
        assert_eq!(err, super::BodyReadError::ReadFailed);
    }

    #[test]
    fn proxy_max_body_parser_uses_mebibytes_and_ignores_zero_or_bad_values() {
        assert_eq!(
            super::proxy_max_body_bytes_from_value(Some("2")),
            Some(2 * super::MIB)
        );
        assert_eq!(
            super::proxy_max_body_bytes_from_value(Some(" 3 ")),
            Some(3 * super::MIB)
        );
        assert_eq!(super::proxy_max_body_bytes_from_value(Some("0")), None);
        assert_eq!(super::proxy_max_body_bytes_from_value(Some("bad")), None);
        assert_eq!(super::proxy_max_body_bytes_from_value(None), None);
    }

    #[test]
    fn capture_path_slug_sanitizes_paths_for_file_names() {
        assert_eq!(super::capture_path_slug("/v1/messages"), "v1_messages");
        assert_eq!(super::capture_path_slug("/"), "root");
        assert_eq!(
            super::capture_path_slug("/openai/v1/responses?x=1"),
            "openai_v1_responses_x_1"
        );
    }

    #[test]
    fn sqlite_soft_cap_parser_uses_mebibytes_and_ignores_zero_or_bad_values() {
        assert_eq!(
            super::sqlite_soft_cap_bytes_from_value(Some("2")),
            Some(2 * 1024 * 1024)
        );
        assert_eq!(
            super::sqlite_soft_cap_bytes_from_value(Some(" 3 ")),
            Some(3 * 1024 * 1024)
        );
        assert_eq!(super::sqlite_soft_cap_bytes_from_value(Some("0")), None);
        assert_eq!(super::sqlite_soft_cap_bytes_from_value(Some("bad")), None);
        assert_eq!(super::sqlite_soft_cap_bytes_from_value(None), None);
    }

    #[test]
    fn sqlite_resident_cap_parser_uses_mebibytes_and_ignores_zero_or_bad_values() {
        assert_eq!(
            super::sqlite_resident_cap_bytes_from_value(Some("2")),
            Some(2 * 1024 * 1024)
        );
        assert_eq!(
            super::sqlite_resident_cap_bytes_from_value(Some(" 3 ")),
            Some(3 * 1024 * 1024)
        );
        assert_eq!(super::sqlite_resident_cap_bytes_from_value(Some("0")), None);
        assert_eq!(
            super::sqlite_resident_cap_bytes_from_value(Some("bad")),
            None
        );
        assert_eq!(super::sqlite_resident_cap_bytes_from_value(None), None);
    }

    #[test]
    fn sqlite_warm_bytes_parser_accepts_common_true_values_only() {
        assert!(super::sqlite_warm_bytes_on_open_from_value(Some("1")));
        assert!(super::sqlite_warm_bytes_on_open_from_value(Some("true")));
        assert!(super::sqlite_warm_bytes_on_open_from_value(Some(" YES ")));
        assert!(super::sqlite_warm_bytes_on_open_from_value(Some("on")));
        assert!(!super::sqlite_warm_bytes_on_open_from_value(Some("0")));
        assert!(!super::sqlite_warm_bytes_on_open_from_value(Some("false")));
        assert!(!super::sqlite_warm_bytes_on_open_from_value(None));
    }

    #[test]
    fn sqlite_trust_hashes_parser_accepts_common_true_values_only() {
        assert!(super::sqlite_trust_hashes_on_open_from_value(Some("1")));
        assert!(super::sqlite_trust_hashes_on_open_from_value(Some("true")));
        assert!(super::sqlite_trust_hashes_on_open_from_value(Some(" YES ")));
        assert!(super::sqlite_trust_hashes_on_open_from_value(Some("on")));
        assert!(!super::sqlite_trust_hashes_on_open_from_value(Some("0")));
        assert!(!super::sqlite_trust_hashes_on_open_from_value(Some(
            "false"
        )));
        assert!(!super::sqlite_trust_hashes_on_open_from_value(None));
    }

    #[test]
    fn sqlite_checkpoint_every_parser_uses_positive_blob_count_only() {
        assert_eq!(
            super::sqlite_checkpoint_every_blobs_from_value(Some("1")),
            Some(1)
        );
        assert_eq!(
            super::sqlite_checkpoint_every_blobs_from_value(Some(" 25 ")),
            Some(25)
        );
        assert_eq!(
            super::sqlite_checkpoint_every_blobs_from_value(Some("0")),
            None
        );
        assert_eq!(
            super::sqlite_checkpoint_every_blobs_from_value(Some("bad")),
            None
        );
        assert_eq!(super::sqlite_checkpoint_every_blobs_from_value(None), None);
    }
}
