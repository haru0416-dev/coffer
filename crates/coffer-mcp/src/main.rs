//! coffer MCP server: hold large agent tool-output **server-side** and let the model
//! direct coffer at it via opaque handles — exact `digest` aggregates, predicate `query`,
//! targeted `search`/`lines`, and byte-exact `retrieve` — so huge tool-output never enters the
//! model's context.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Stdio};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use coffer_cas::{Cas, ContentHash, SqliteCas, SqliteConfig};
use coffer_core::{
    Agg, ContentType, Op, Predicate, compress_json_where, compress_structural_code_to_budget,
    describe, detect, digest, query_aggregate, query_subset,
};
use coffer_tokenizer::{HeuristicCounter, TokenCounter};
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, Content};
use rmcp::{ErrorData, ServerHandler, ServiceExt, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;
use tokio::io::AsyncReadExt;

type HeldBytes = Arc<[u8]>;
const MIB: usize = 1024 * 1024;
const DEFAULT_RETRIEVE_BYTES: usize = 64 * 1024;
const DEFAULT_MAX_RETRIEVE_BYTES: usize = 1024 * 1024;
const DEFAULT_STRUCTURAL_CODE_TARGET_TOKENS: usize = 1024;
const DEFAULT_RUN_TIMEOUT_SECONDS: u64 = 300;
const DEFAULT_MAX_RUN_OUTPUT_BYTES: usize = 64 * MIB;
/// Hard ceiling on rows pretty-printed by `coffer_rows`, overridable via `COFFER_MCP_MAX_ROWS`.
/// Bounds the model-facing response so a large `limit` cannot re-bloat the context that the
/// server-side hold exists to keep out of it (the original stays reachable via coffer_retrieve).
const DEFAULT_MAX_ROWS: usize = 200;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RetrieveLimits {
    default_bytes: usize,
    max_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RunLimits {
    timeout_seconds: u64,
    max_output_bytes: usize,
}

#[derive(Debug)]
struct RunCapture {
    bytes: Vec<u8>,
    status: Option<ExitStatus>,
    timed_out: bool,
    output_truncated: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunReadOutcome {
    Exited(ExitStatus),
    OutputLimit,
}

enum HandleStore {
    Memory(Mutex<HashMap<String, HeldBytes>>),
    Sqlite {
        path: PathBuf,
        cas: Box<SqliteCas>,
        soft_cap_bytes: Option<usize>,
        warm_bytes_on_open: bool,
        trust_hashes_on_open: bool,
        resident_cap_bytes: Option<usize>,
        checkpoint_every_blobs: Option<usize>,
    },
}

#[derive(Clone)]
struct Coffer {
    store: Arc<HandleStore>,
}

impl Coffer {
    fn new() -> anyhow::Result<Self> {
        match std::env::var("COFFER_CAS_DB") {
            Ok(path) if !path.trim().is_empty() => Self::with_sqlite(path),
            _ => Ok(Self::in_memory()),
        }
    }

    fn in_memory() -> Self {
        Self {
            store: Arc::new(HandleStore::Memory(Mutex::new(HashMap::new()))),
        }
    }

    fn with_sqlite(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            create_private_dir_all(parent)?;
        }
        let soft_cap_bytes = sqlite_soft_cap_bytes_from_env();
        let resident_cap_bytes = sqlite_resident_cap_bytes_from_env();
        let warm_bytes_on_open = sqlite_warm_bytes_on_open_from_env();
        let trust_hashes_on_open = sqlite_trust_hashes_on_open_from_env();
        let checkpoint_every_blobs = sqlite_checkpoint_every_blobs_from_env();
        Ok(Self {
            store: Arc::new(HandleStore::Sqlite {
                path: path.to_path_buf(),
                cas: Box::new(SqliteCas::open_with_config(
                    path,
                    &SqliteConfig {
                        soft_cap_bytes,
                        warm_bytes_on_open,
                        trust_hashes_on_open,
                        resident_cap_bytes,
                        checkpoint_every_blobs,
                    },
                )?),
                soft_cap_bytes,
                warm_bytes_on_open,
                trust_hashes_on_open,
                resident_cap_bytes,
                checkpoint_every_blobs,
            }),
        })
    }

    fn put_bytes(&self, bytes: &[u8]) -> ContentHash {
        match self.store.as_ref() {
            HandleStore::Memory(map) => {
                let hash = ContentHash::of(bytes);
                map.lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .entry(hash.as_str().to_string())
                    .or_insert_with(|| Arc::from(bytes));
                hash
            }
            HandleStore::Sqlite { cas, .. } => {
                let hash = cas.put(bytes);
                cas.flush();
                hash
            }
        }
    }

    fn get_handle(&self, handle: &str) -> Option<HeldBytes> {
        match self.store.as_ref() {
            HandleStore::Memory(map) => map
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .get(handle)
                .cloned(),
            HandleStore::Sqlite { path, cas, .. } => {
                if let Some(hash) = ContentHash::from_hex(handle) {
                    return cas.get(&hash);
                }
                coffer_cas::read_blob(path, handle)
                    .ok()
                    .flatten()
                    .map(Arc::from)
            }
        }
    }

    fn handle_count(&self) -> usize {
        match self.store.as_ref() {
            HandleStore::Memory(map) => map.lock().unwrap_or_else(PoisonError::into_inner).len(),
            HandleStore::Sqlite { cas, .. } => cas.len(),
        }
    }

    fn status_text(&self) -> String {
        let retrieve_limits = retrieve_limits_from_env();
        match self.store.as_ref() {
            HandleStore::Memory(map) => {
                let map = map.lock().unwrap_or_else(PoisonError::into_inner);
                let bytes: usize = map.values().map(|b| b.len()).sum();
                format!(
                    "store: memory\nhandles: {}\nresident_bytes: {bytes}\nretrieve_default_bytes: {}\nretrieve_max_bytes: {}",
                    map.len(),
                    retrieve_limits.default_bytes,
                    retrieve_limits.max_bytes
                )
            }
            HandleStore::Sqlite {
                path: _,
                cas,
                soft_cap_bytes,
                warm_bytes_on_open,
                trust_hashes_on_open,
                resident_cap_bytes,
                checkpoint_every_blobs,
            } => {
                let disk = cas.disk_usage();
                format!(
                    "store: sqlite\nhandles: {}\nresident_bytes: {}\nsqlite_db_bytes: {}\nsqlite_wal_bytes: {}\nsqlite_shm_bytes: {}\nsqlite_total_bytes: {}\nsoft_cap_bytes: {}\nresident_cap_bytes: {}\nresident_evictions: {}\nwarm_bytes_on_open: {}\ntrust_hashes_on_open: {}\ncheckpoint_every_blobs: {}\nwal_checkpoints: {}\nwal_checkpoint_failures: {}\ndurability_lag: {}\ndropped_writes: {}\npersisted_blobs_this_run: {}\nretrieve_default_bytes: {}\nretrieve_max_bytes: {}",
                    cas.len(),
                    cas.mem_bytes(),
                    disk.db_bytes,
                    disk.wal_bytes,
                    disk.shm_bytes,
                    disk.total_bytes(),
                    soft_cap_bytes.map_or_else(|| "unset".to_string(), |n| n.to_string()),
                    resident_cap_bytes.map_or_else(|| "unset".to_string(), |n| n.to_string()),
                    cas.resident_evictions(),
                    warm_bytes_on_open,
                    trust_hashes_on_open,
                    checkpoint_every_blobs.map_or_else(|| "unset".to_string(), |n| n.to_string()),
                    cas.wal_checkpoints(),
                    cas.wal_checkpoint_failures(),
                    cas.durability_lag(),
                    cas.dropped_writes(),
                    cas.persisted(),
                    retrieve_limits.default_bytes,
                    retrieve_limits.max_bytes
                )
            }
        }
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

impl Cas for Coffer {
    fn put(&self, bytes: &[u8]) -> ContentHash {
        self.put_bytes(bytes)
    }

    fn get(&self, hash: &ContentHash) -> Option<HeldBytes> {
        self.get_handle(hash.as_str())
    }

    fn len(&self) -> usize {
        self.handle_count()
    }
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

fn retrieve_limits_from_env() -> RetrieveLimits {
    let default_raw = std::env::var("COFFER_MCP_DEFAULT_RETRIEVE_BYTES").ok();
    let max_raw = std::env::var("COFFER_MCP_MAX_RETRIEVE_BYTES").ok();
    retrieve_limits_from_values(default_raw.as_deref(), max_raw.as_deref())
}

fn max_rows_from_env() -> usize {
    positive_usize_from_value(std::env::var("COFFER_MCP_MAX_ROWS").ok().as_deref())
        .unwrap_or(DEFAULT_MAX_ROWS)
}

fn retrieve_limits_from_values(default_raw: Option<&str>, max_raw: Option<&str>) -> RetrieveLimits {
    let max_bytes = positive_usize_from_value(max_raw).unwrap_or(DEFAULT_MAX_RETRIEVE_BYTES);
    let default_bytes = positive_usize_from_value(default_raw)
        .unwrap_or(DEFAULT_RETRIEVE_BYTES)
        .min(max_bytes);
    RetrieveLimits {
        default_bytes,
        max_bytes,
    }
}

fn run_limits_from_env() -> RunLimits {
    let timeout_raw = std::env::var("COFFER_MCP_RUN_TIMEOUT_SECONDS").ok();
    let output_raw = std::env::var("COFFER_MCP_MAX_RUN_OUTPUT_MB").ok();
    run_limits_from_values(timeout_raw.as_deref(), output_raw.as_deref())
}

fn run_limits_from_values(timeout_raw: Option<&str>, output_mb_raw: Option<&str>) -> RunLimits {
    let timeout_seconds =
        positive_u64_from_value(timeout_raw).unwrap_or(DEFAULT_RUN_TIMEOUT_SECONDS);
    let max_output_bytes = positive_usize_from_value(output_mb_raw)
        .map(|mb| mb.saturating_mul(MIB))
        .unwrap_or(DEFAULT_MAX_RUN_OUTPUT_BYTES);
    RunLimits {
        timeout_seconds,
        max_output_bytes,
    }
}

/// Whether `coffer_run` may execute, and any command allowlist. Off by default: arbitrary
/// `sh -c` execution is opt-in via `COFFER_MCP_ENABLE_RUN`, optionally narrowed by a comma-separated
/// `COFFER_MCP_RUN_ALLOWLIST` of command prefixes. When an allowlist is configured, shell control
/// syntax is refused before spawning.
struct RunPolicy {
    enabled: bool,
    allowlist: Vec<String>,
}

impl RunPolicy {
    /// `Ok(())` if `command` may run under this policy, else an explanatory message the caller returns
    /// to the client. Never spawns anything itself.
    fn permits(&self, command: &str) -> Result<(), String> {
        if !self.enabled {
            return Err(
                "coffer_run is disabled by default; set COFFER_MCP_ENABLE_RUN=1 to enable it \
                 (optionally COFFER_MCP_RUN_ALLOWLIST=\"prog1,prog2\" to restrict commands)"
                    .to_string(),
            );
        }
        let trimmed = command.trim_start();
        if !self.allowlist.is_empty() {
            if contains_shell_control(trimmed) {
                return Err(
                    "command refused: COFFER_MCP_RUN_ALLOWLIST accepts only simple command lines; shell control syntax is not allowed"
                        .to_string(),
                );
            }
            if !self
                .allowlist
                .iter()
                .any(|prefix| allowlist_prefix_matches(trimmed, prefix))
            {
                return Err(format!(
                    "command refused: COFFER_MCP_RUN_ALLOWLIST permits only commands beginning with one of [{}]",
                    self.allowlist.join(", ")
                ));
            }
        }
        Ok(())
    }
}

fn contains_shell_control(command: &str) -> bool {
    command.bytes().any(|b| {
        matches!(
            b,
            b';' | b'|'
                | b'&'
                | b'<'
                | b'>'
                | b'`'
                | b'$'
                | b'('
                | b')'
                | b'{'
                | b'}'
                | b'\n'
                | b'\r'
        )
    })
}

fn allowlist_prefix_matches(command: &str, prefix: &str) -> bool {
    command.starts_with(prefix)
        && command
            .as_bytes()
            .get(prefix.len())
            .is_none_or(u8::is_ascii_whitespace)
}

fn truthy(raw: Option<&str>) -> bool {
    raw.is_some_and(|v| {
        matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn run_policy_from_env() -> RunPolicy {
    run_policy_from_values(
        std::env::var("COFFER_MCP_ENABLE_RUN").ok().as_deref(),
        std::env::var("COFFER_MCP_RUN_ALLOWLIST").ok().as_deref(),
    )
}

fn run_policy_from_values(enable_raw: Option<&str>, allowlist_raw: Option<&str>) -> RunPolicy {
    let allowlist = allowlist_raw
        .map(|raw| {
            raw.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();
    RunPolicy {
        enabled: truthy(enable_raw),
        allowlist,
    }
}

fn positive_u64_from_value(raw: Option<&str>) -> Option<u64> {
    let value = raw?.trim().parse::<u64>().ok()?;
    (value > 0).then_some(value)
}

fn positive_usize_from_value(raw: Option<&str>) -> Option<usize> {
    let value = raw?.trim().parse::<usize>().ok()?;
    (value > 0).then_some(value)
}

async fn run_shell_command(command: &str, limits: RunLimits) -> std::io::Result<RunCapture> {
    let shell_command = format!("exec 2>&1; {command}");
    let mut child = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(shell_command)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()?;
    let mut stdout = child
        .stdout
        .take()
        .expect("stdout is piped before spawning the shell");
    let mut bytes = Vec::new();
    let mut buf = [0_u8; 8192];
    let timeout = Duration::from_secs(limits.timeout_seconds);
    let read = async {
        loop {
            let read = stdout.read(&mut buf).await?;
            if read == 0 {
                break;
            }
            let remaining = limits.max_output_bytes.saturating_sub(bytes.len());
            if read > remaining {
                bytes.extend_from_slice(&buf[..remaining]);
                return Ok(RunReadOutcome::OutputLimit);
            }
            bytes.extend_from_slice(&buf[..read]);
        }
        child.wait().await.map(RunReadOutcome::Exited)
    };

    match tokio::time::timeout(timeout, read).await {
        Ok(Ok(RunReadOutcome::Exited(status))) => Ok(RunCapture {
            bytes,
            status: Some(status),
            timed_out: false,
            output_truncated: false,
        }),
        Ok(Ok(RunReadOutcome::OutputLimit)) => {
            let _ = child.start_kill();
            let _ = child.wait().await;
            Ok(RunCapture {
                bytes,
                status: None,
                timed_out: false,
                output_truncated: true,
            })
        }
        Ok(Err(error)) => Err(error),
        Err(_) => {
            let _ = child.start_kill();
            let _ = child.wait().await;
            Ok(RunCapture {
                bytes,
                status: None,
                timed_out: true,
                output_truncated: false,
            })
        }
    }
}

fn summarize(bytes: &[u8]) -> String {
    match detect(bytes) {
        ContentType::Json => match serde_json::from_slice::<Value>(bytes) {
            Ok(Value::Array(a)) => {
                let keys = a
                    .first()
                    .and_then(Value::as_object)
                    .map(|o| o.keys().cloned().collect::<Vec<_>>().join(","))
                    .unwrap_or_default();
                format!("json array, {} items, fields: [{keys}]", a.len())
            }
            Ok(Value::Object(o)) => {
                format!(
                    "json object, keys: [{}]",
                    o.keys().cloned().collect::<Vec<_>>().join(",")
                )
            }
            _ => format!("json, {} bytes", bytes.len()),
        },
        ContentType::Log => format!(
            "log, {} lines",
            String::from_utf8_lossy(bytes).lines().count()
        ),
        ContentType::Text => format!("text, {} bytes", bytes.len()),
    }
}

/// A query-free **fact card**: per-field basic stats over a JSON array, so even un-queried data
/// arrives carrying computable facts that hint what `coffer_digest` can answer exactly. Numeric
/// columns (every value f64-representable) report min/max/mean; everything else reports distinct
/// count. Defensive on purpose: a non-f64 "number" (e.g. `1e400`) falls to the distinct branch
/// rather than risk a wrong stat — the authoritative aggregate is still `coffer_digest`.
fn fact_card(bytes: &[u8]) -> Option<String> {
    let Ok(Value::Array(rows)) = serde_json::from_slice::<Value>(bytes) else {
        return None;
    };
    let mut keys = Vec::new();
    for object in rows.iter().filter_map(Value::as_object) {
        for key in object.keys() {
            if !keys.contains(key) {
                keys.push(key.clone());
            }
        }
    }
    if keys.is_empty() {
        return None;
    }
    let mut lines = Vec::new();
    for key in keys {
        let vals: Vec<&Value> = rows
            .iter()
            .filter_map(Value::as_object)
            .filter_map(|o| o.get(&key))
            .collect();
        if vals.is_empty() {
            continue;
        }
        let nums: Vec<f64> = vals.iter().filter_map(|v| v.as_f64()).collect();
        if nums.len() == vals.len() {
            let min = nums.iter().copied().fold(f64::INFINITY, f64::min);
            let max = nums.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            #[allow(clippy::cast_precision_loss)]
            let mean = nums.iter().sum::<f64>() / nums.len() as f64;
            lines.push(format!(
                "  {key}: numeric present={}/{} min={min} max={max} mean={mean:.4}",
                vals.len(),
                rows.len()
            ));
        } else {
            let distinct: std::collections::HashSet<String> =
                vals.iter().map(ToString::to_string).collect();
            lines.push(format!(
                "  {key}: present={}/{} {} distinct",
                vals.len(),
                rows.len(),
                distinct.len()
            ));
        }
    }
    (!lines.is_empty()).then(|| format!("field stats:\n{}", lines.join("\n")))
}

struct ByteWindow<'a> {
    start: usize,
    end: usize,
    total: usize,
    bytes: &'a [u8],
}

fn byte_window(bytes: &[u8], start: Option<usize>, max_bytes: Option<usize>) -> ByteWindow<'_> {
    let total = bytes.len();
    let start = start.unwrap_or(0).min(total);
    let end = max_bytes.map_or(total, |n| start.saturating_add(n).min(total));
    ByteWindow {
        start,
        end,
        total,
        bytes: &bytes[start..end],
    }
}

fn render_retrieved_bytes(
    bytes: &[u8],
    start: Option<usize>,
    max_bytes: Option<usize>,
    full: bool,
    limits: RetrieveLimits,
) -> Result<String, String> {
    if full {
        if start.is_some() || max_bytes.is_some() {
            return Err("full=true cannot be combined with start or max_bytes".to_string());
        }
        if bytes.len() > limits.max_bytes {
            return Err(format!(
                "full retrieval would return {} bytes, exceeding COFFER_MCP_MAX_RETRIEVE_BYTES={} \
                 (use start/max_bytes for a bounded window, or raise the limit intentionally)",
                bytes.len(),
                limits.max_bytes
            ));
        }
        return Ok(String::from_utf8_lossy(bytes).into_owned());
    }

    let window_len = max_bytes.unwrap_or(limits.default_bytes);
    if window_len > limits.max_bytes {
        return Err(format!(
            "requested max_bytes={} exceeds COFFER_MCP_MAX_RETRIEVE_BYTES={}",
            window_len, limits.max_bytes
        ));
    }
    let window = byte_window(bytes, start, Some(window_len));
    let text = String::from_utf8_lossy(window.bytes).into_owned();
    if start.is_some() || max_bytes.is_some() || window.end < window.total {
        Ok(format!(
            "bytes {}..{} of {} ({} before, {} after)\n{}",
            window.start,
            window.end,
            window.total,
            window.start,
            window.total - window.end,
            text
        ))
    } else {
        Ok(text)
    }
}

fn render_json_rows(
    bytes: &[u8],
    start: Option<usize>,
    limit: Option<usize>,
    max_rows: usize,
) -> Result<String, String> {
    let value: Value = serde_json::from_slice(bytes).map_err(|e| format!("not valid JSON: {e}"))?;
    let Value::Array(rows) = value else {
        return Err("held data is JSON but not an array".to_string());
    };
    let total = rows.len();
    let start = start.unwrap_or(0).min(total);
    // Clamp the page size so coffer_rows(limit=usize::MAX) cannot pretty-print the entire held
    // array back into the model's context — the exact bloat these window tools exist to avoid.
    // `total` is still reported so the model sees there are more rows; the original is reachable
    // via coffer_retrieve / coffer_digest.
    let limit = limit.unwrap_or(20).min(max_rows.max(1));
    let end = start.saturating_add(limit).min(total);
    let body =
        serde_json::to_string_pretty(&rows[start..end]).map_err(|e| format!("render rows: {e}"))?;
    Ok(format!(
        "rows {start}..{end} of {total} ({start} before, {} after)\n{body}",
        total - end
    ))
}

fn render_json_path(bytes: &[u8], path: &str) -> Result<String, String> {
    let value: Value = serde_json::from_slice(bytes).map_err(|e| format!("not valid JSON: {e}"))?;
    let selected = select_json_path(&value, path)?;
    let body =
        serde_json::to_string_pretty(selected).map_err(|e| format!("render JSON path: {e}"))?;
    Ok(format!("json path {path}\n{body}"))
}

fn render_text_lines(
    bytes: &[u8],
    start_line: Option<usize>,
    end_line: Option<usize>,
    head: Option<usize>,
    tail: Option<usize>,
) -> Result<String, String> {
    if head.is_some() && tail.is_some() {
        return Err("choose either head or tail, not both".to_string());
    }
    if (head.is_some() || tail.is_some()) && (start_line.is_some() || end_line.is_some()) {
        return Err("choose head/tail or start_line/end_line, not both".to_string());
    }

    let text = String::from_utf8_lossy(bytes);
    let lines: Vec<&str> = text.lines().collect();
    let total = lines.len();

    let (start_idx, end_idx) = if let Some(n) = head {
        (0, n.min(total))
    } else if let Some(n) = tail {
        (total.saturating_sub(n), total)
    } else {
        let start = start_line.unwrap_or(1).max(1);
        let end = end_line.unwrap_or_else(|| start.saturating_add(79));
        let start_idx = start.saturating_sub(1).min(total);
        let end_idx = if end < start {
            start_idx
        } else {
            end.min(total)
        };
        (start_idx, end_idx)
    };

    let shown_start = if start_idx < end_idx {
        start_idx + 1
    } else {
        start_idx.min(total)
    };
    let body = lines[start_idx..end_idx]
        .iter()
        .enumerate()
        .map(|(i, line)| format!("{:>6}|{line}", start_idx + i + 1))
        .collect::<Vec<_>>()
        .join("\n");

    Ok(format!(
        "lines {shown_start}..{end_idx} of {total} ({start_idx} before, {} after)\n{body}",
        total - end_idx
    ))
}

fn render_text_search(bytes: &[u8], pattern: &str, limit: Option<usize>) -> Result<String, String> {
    if pattern.is_empty() {
        return Err("pattern must not be empty".to_string());
    }

    let text = String::from_utf8_lossy(bytes);
    let lines: Vec<&str> = text.lines().collect();
    let pattern_lower = pattern.to_lowercase();
    let limit = limit.unwrap_or(20);
    let mut total_matches = 0usize;
    let mut shown = Vec::new();

    for (i, line) in lines.iter().enumerate() {
        if line.to_lowercase().contains(&pattern_lower) {
            total_matches += 1;
            if shown.len() < limit {
                shown.push(format!("{:>6}|{line}", i + 1));
            }
        }
    }

    if total_matches == 0 {
        return Ok(format!(
            "0 matches for \"{pattern}\" in {} lines",
            lines.len()
        ));
    }

    let omitted = total_matches.saturating_sub(shown.len());
    Ok(format!(
        "{total_matches} matches for \"{pattern}\" in {} lines (showing {}, {omitted} omitted)\n{}",
        lines.len(),
        shown.len(),
        shown.join("\n")
    ))
}

fn select_json_path<'a>(value: &'a Value, path: &str) -> Result<&'a Value, String> {
    let mut rest = path.trim();
    if rest.is_empty() || rest == "$" {
        return Ok(value);
    }
    if let Some(stripped) = rest.strip_prefix('$') {
        rest = stripped;
    }
    let mut current = value;
    while !rest.is_empty() {
        if let Some(after_dot) = rest.strip_prefix('.') {
            let end = after_dot.find(['.', '[']).unwrap_or(after_dot.len());
            if end == 0 {
                return Err("empty field in JSON path".to_string());
            }
            let key = &after_dot[..end];
            let Some(object) = current.as_object() else {
                return Err(format!("path field {key} needs an object"));
            };
            let Some(next) = object.get(key) else {
                return Err(format!("path field not found: {key}"));
            };
            current = next;
            rest = &after_dot[end..];
        } else if let Some(after_open) = rest.strip_prefix('[') {
            let Some(close) = after_open.find(']') else {
                return Err("unterminated array index in JSON path".to_string());
            };
            let index_text = &after_open[..close];
            let index: usize = index_text
                .parse()
                .map_err(|_| format!("invalid array index: {index_text}"))?;
            let Some(array) = current.as_array() else {
                return Err(format!("array index {index} needs an array"));
            };
            let Some(next) = array.get(index) else {
                return Err(format!("array index out of range: {index}"));
            };
            current = next;
            rest = &after_open[close + 1..];
        } else {
            return Err("JSON path must use .field and [index] steps".to_string());
        }
    }
    Ok(current)
}

fn parse_op(op: &str) -> Op {
    match op {
        "ne" | "!=" => Op::Ne,
        "gt" | ">" => Op::Gt,
        "ge" | ">=" => Op::Ge,
        "lt" | "<" => Op::Lt,
        "le" | "<=" => Op::Le,
        _ => Op::Eq,
    }
}

/// Parse an argument value as JSON, falling back to treating it as a bare string (so callers can pass
/// `error` instead of `"error"`). Shared by the predicate-taking tools.
fn parse_value_arg(raw: &str) -> Value {
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
}

/// Build a conjunction of typed predicates from the wire form. Shared by coffer_select / coffer_aggregate.
fn predicates_from_args(args: &[PredicateArg]) -> Vec<Predicate> {
    args.iter()
        .map(|p| Predicate {
            field: p.field.clone(),
            op: parse_op(&p.op),
            value: parse_value_arg(&p.value),
        })
        .collect()
}

/// Resolve an aggregate name (+ optional field) to an [`Agg`]. `count` needs no field; the rest require one.
fn parse_agg(agg: &str, field: Option<&str>) -> Option<Agg> {
    match agg {
        "count" => Some(Agg::Count),
        "sum" => field.map(|f| Agg::Sum(f.to_string())),
        "mean" | "avg" => field.map(|f| Agg::Mean(f.to_string())),
        "min" => field.map(|f| Agg::Min(f.to_string())),
        "max" => field.map(|f| Agg::Max(f.to_string())),
        _ => None,
    }
}

fn ingested_text(handle: &str, bytes: &[u8]) -> String {
    let card = fact_card(bytes)
        .map(|c| format!("\n{c}"))
        .unwrap_or_default();
    format!(
        "handle: {handle}\nsummary: {}\nbytes: {}{card}\n\nThe full output is held server-side and is NOT in your context. \
         Use coffer_digest(handle, query) for EXACT aggregates over ALL of it (count/sum/mean/median/percentile/\
         group-by/argmax/filter-aggregate), coffer_query(handle, field, op, value) to keep only matching rows, \
         coffer_select(handle, where) to filter by a conjunction and get the matches as a new handle to narrow again, or \
         coffer_search(handle, pattern) / coffer_lines(handle, start_line, end_line) for logs/text, \
         coffer_json(handle, path) / coffer_rows(handle, start, limit) / coffer_retrieve(handle, start, max_bytes) for a small window; \
         set full=true only for small payloads you truly need raw, subject to the configured hard cap.",
        summarize(bytes),
        bytes.len()
    )
}

#[derive(Debug)]
enum IngestView {
    Summary,
    StructuralCode,
}

fn ingest_view(view: Option<&str>) -> Result<IngestView, String> {
    match view.map(str::trim).filter(|v| !v.is_empty()) {
        None => Ok(IngestView::Summary),
        Some("structural_code") => Ok(IngestView::StructuralCode),
        Some(other) => Err(format!(
            "unknown ingest view {other:?}; supported values: structural_code"
        )),
    }
}

fn ingested_text_with_view(
    handle: &str,
    bytes: &[u8],
    view: IngestView,
    target_tokens: Option<usize>,
    cas: &dyn Cas,
) -> String {
    match view {
        IngestView::Summary => ingested_text(handle, bytes),
        IngestView::StructuralCode => {
            let target = target_tokens.unwrap_or(DEFAULT_STRUCTURAL_CODE_TARGET_TOKENS);
            let counter = HeuristicCounter;
            let compact = compress_structural_code_to_budget(bytes, cas, target, &counter);
            let tokens = counter.count(&compact.model_text);
            format!(
                "{}\n\nview: structural_code\nview_target_tokens: {target}\nview_tokens: {tokens}\n{}\n\nThis is a compact code outline. Retrieve the original handle before exact edits.",
                ingested_text(handle, bytes),
                compact.model_text.trim_end()
            )
        }
    }
}

fn unknown_handle() -> CallToolResult {
    CallToolResult::error(vec![Content::text(
        "unknown handle or sentinel hash prefix (ingest/run it first, or use a prefix from a shared-CAS <<cof:...>> sentinel)",
    )])
}

fn unfold_shared_cas_result(
    path: impl AsRef<Path>,
    args: &UnfoldArgs,
    limits: RetrieveLimits,
) -> CallToolResult {
    match coffer_cas::read_blob(path, &args.hash) {
        Ok(Some(bytes)) => match render_retrieved_bytes(
            &bytes,
            args.start,
            args.max_bytes,
            args.full.unwrap_or(false),
            limits,
        ) {
            Ok(text) => CallToolResult::success(vec![Content::text(text)]),
            Err(e) => CallToolResult::error(vec![Content::text(e)]),
        },
        Ok(None) => CallToolResult::error(vec![Content::text(
            "no bytes for that sentinel hash in the shared CAS (wrong hash, or it was never offloaded here)",
        )]),
        Err(e) => {
            CallToolResult::error(vec![Content::text(format!("shared CAS read failed: {e}"))])
        }
    }
}

#[tool_router]
impl Coffer {
    /// Run a shell command and hold its stdout/stderr server-side; returns a handle + summary.
    #[tool(
        description = "Run a shell command and hold its stdout/stderr SERVER-SIDE; returns a handle + a summary. \
        The output never enters your context — interrogate it with coffer_digest / coffer_query instead of reading it."
    )]
    async fn coffer_run(
        &self,
        Parameters(a): Parameters<RunArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        // Off by default: refuse before spawning unless explicitly enabled / allowlisted.
        if let Err(message) = run_policy_from_env().permits(&a.command) {
            return Ok(CallToolResult::error(vec![Content::text(message)]));
        }
        let limits = run_limits_from_env();
        match run_shell_command(&a.command, limits).await {
            Ok(capture) => {
                let h = self.put_bytes(&capture.bytes).as_str().to_string();
                let ingested = ingested_text(&h, &capture.bytes);
                let text = if capture.timed_out {
                    format!(
                        "command timed out after {} seconds; captured {} bytes before termination (partial output)\n{}",
                        limits.timeout_seconds,
                        capture.bytes.len(),
                        ingested
                    )
                } else if capture.output_truncated {
                    format!(
                        "command output exceeded COFFER_MCP_MAX_RUN_OUTPUT_MB-derived limit ({} bytes); captured first {} bytes and terminated the process (partial output)\n{}",
                        limits.max_output_bytes,
                        capture.bytes.len(),
                        ingested
                    )
                } else if capture.status.is_some_and(|status| status.success()) {
                    ingested
                } else if let Some(status) = capture.status {
                    format!("command exited {status}\n{ingested}")
                } else {
                    format!("command ended without an exit status\n{ingested}")
                };
                Ok(CallToolResult::success(vec![Content::text(text)]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "spawn failed: {e}"
            ))])),
        }
    }

    /// Ingest a file from disk and hold it server-side; returns a handle + summary.
    #[tool(
        description = "Ingest a file from disk and hold it server-side; returns a handle + summary. \
        Optional view=\"structural_code\" returns an explicit compact code outline backed by the same retrievable original."
    )]
    async fn coffer_ingest(
        &self,
        Parameters(a): Parameters<IngestArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let view = match ingest_view(a.view.as_deref()) {
            Ok(view) => view,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };
        match std::fs::read(&a.path) {
            Ok(bytes) => {
                let h = self.put_bytes(&bytes).as_str().to_string();
                Ok(CallToolResult::success(vec![Content::text(
                    ingested_text_with_view(&h, &bytes, view, a.target_tokens, self),
                )]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "read {}: {e}",
                a.path
            ))])),
        }
    }

    /// Compute an EXACT deterministic aggregate over ALL of the held data.
    #[tool(
        description = "Compute an EXACT deterministic aggregate over ALL of the held data \
        (count/sum/mean/median/percentile/range/distinct/group-by/argmax/filter-aggregate). \
        In SQLite/shared-CAS deployments, handle may be a full handle or a unique <<cof:HASH>> \
        sentinel prefix. The result is COMPUTED, not estimated — trust it over your own count. \
        Returns nothing if no aggregate matches (it refuses rather than guess)."
    )]
    async fn coffer_digest(
        &self,
        Parameters(a): Parameters<DigestArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let Some(bytes) = self.get_handle(&a.handle) else {
            return Ok(unknown_handle());
        };
        match digest(&bytes, &a.query) {
            Some(fact) => Ok(CallToolResult::success(vec![Content::text(format!(
                "{fact}  (computed exactly over the whole dataset)"
            ))])),
            None => Ok(CallToolResult::error(vec![Content::text(
                "no deterministic aggregate matched this query over this data",
            )])),
        }
    }

    /// Shape-generic exact summary of a held JSON array of records (schema + per-field stats / count-by).
    #[tool(
        description = "Summarize a held JSON array of records GENERICALLY and exactly: row count, and per \
        field its present/distinct counts plus either numeric stats (min/max/mean/sum) or a \
        count-by-value breakdown for low-cardinality categoricals. No per-tool/per-format code — point it \
        at any record set (ideally a tool's structured --json/-o json output) for an RTK-style decision \
        summary that is exact and recoverable. Use coffer_aggregate for a specific number. handle may be a \
        full handle or a unique <<cof:HASH>> sentinel prefix over a JSON array."
    )]
    async fn coffer_describe(
        &self,
        Parameters(a): Parameters<DescribeArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let Some(bytes) = self.get_handle(&a.handle) else {
            return Ok(unknown_handle());
        };
        match describe(&bytes) {
            Some(card) => Ok(CallToolResult::success(vec![Content::text(card)])),
            None => Ok(CallToolResult::error(vec![Content::text(
                "coffer_describe needs a handle over a JSON array of records",
            )])),
        }
    }

    /// Keep only the rows where `field <op> value`; the rest are elided (still retrievable).
    #[tool(
        description = "Keep only the rows where field <op> value (op: eq|ne|gt|ge|lt|le) and return them; \
        non-matching rows are elided as placeholders. In SQLite/shared-CAS deployments, handle may be \
        a full handle or a unique <<cof:HASH>> sentinel prefix when that blob is valid JSON. \
        Byte-exact reversible — the originals stay retrievable."
    )]
    async fn coffer_query(
        &self,
        Parameters(a): Parameters<QueryArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let Some(bytes) = self.get_handle(&a.handle) else {
            return Ok(unknown_handle());
        };
        let op = parse_op(&a.op);
        let value = parse_value_arg(&a.value);
        let doc = compress_json_where(&bytes, self, &a.field, op, value);
        Ok(CallToolResult::success(vec![Content::text(
            doc.render_for_model(),
        )]))
    }

    /// Filter a held JSON array by a CONJUNCTION of predicates and hold the matching rows as a NEW handle.
    #[tool(
        description = "Filter a held JSON array by a conjunction of predicates (where: a list of \
        {field, op, value}; op is eq|ne|gt|ge|lt|le; a row is kept only if it passes ALL of them) and \
        hold the matching rows as a NEW handle, returned with a fact card. The result is itself a \
        dataset: feed its handle back into coffer_select / coffer_digest / coffer_query / coffer_rows to \
        narrow further — all server-side, the rows never entering your context. Each kept row is byte-exact. \
        handle may be a full handle or a unique <<cof:HASH>> sentinel prefix over a JSON array."
    )]
    async fn coffer_select(
        &self,
        Parameters(a): Parameters<SelectArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let Some(bytes) = self.get_handle(&a.handle) else {
            return Ok(unknown_handle());
        };
        let predicates = predicates_from_args(&a.predicates);
        let Some(subset) = query_subset(&bytes, &predicates) else {
            return Ok(CallToolResult::error(vec![Content::text(
                "coffer_select needs a handle over a top-level JSON array",
            )]));
        };
        let handle = self.put_bytes(&subset).as_str().to_string();
        Ok(CallToolResult::success(vec![Content::text(ingested_text(
            &handle, &subset,
        ))]))
    }

    /// Exact typed aggregate over a held JSON array, returning the number AND its backing row indices.
    #[tool(
        description = "Exact aggregate over a held JSON array, typed and unambiguous: filter by a \
        conjunction of predicates (where: list of {field, op, value}; op eq|ne|gt|ge|lt|le; an ordering \
        op gt/ge/lt/le matches only when field and value are the same type, so compare a numeric field \
        with a number, not a quoted string) and \
        compute agg = count|sum|mean|min|max (field required for all but count). Computed over ALL \
        rows including offloaded ones — trust it over your own count. Returns the value AND the 0-based \
        indices of the backing records (provenance); feed those into coffer_pick(handle, indices) to \
        fetch exactly those rows and re-verify byte-for-byte. Refuses (no guess) when the aggregated \
        field is present-but-non-numeric, or the handle is not a JSON array. handle may be a full \
        handle or a unique <<cof:HASH>> sentinel prefix."
    )]
    async fn coffer_aggregate(
        &self,
        Parameters(a): Parameters<AggregateArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let Some(bytes) = self.get_handle(&a.handle) else {
            return Ok(unknown_handle());
        };
        let Some(agg) = parse_agg(&a.agg, a.field.as_deref()) else {
            return Ok(CallToolResult::error(vec![Content::text(
                "agg must be count|sum|mean|min|max, and sum/mean/min/max require a field",
            )]));
        };
        let predicates = predicates_from_args(&a.predicates);
        match query_aggregate(&bytes, &predicates, &agg) {
            Some(r) => {
                const SHOWN: usize = 64;
                let idx = if r.matched.len() <= SHOWN {
                    format!("{:?}", r.matched)
                } else {
                    let head: Vec<usize> = r.matched.iter().take(SHOWN).copied().collect();
                    format!("{head:?} … ({} total)", r.matched.len())
                };
                Ok(CallToolResult::success(vec![Content::text(format!(
                    "{}\nprovenance row indices: {idx}\nFetch them with coffer_pick(handle, indices) to re-verify.",
                    r.display
                ))]))
            }
            None => Ok(CallToolResult::error(vec![Content::text(
                "no exact aggregate: the handle is not a JSON array, or the aggregated field is present-but-non-numeric",
            )])),
        }
    }

    /// Pull the rows at an explicit set of indices (e.g. a digest's provenance) as a NEW handle.
    #[tool(
        description = "Pull the rows at an explicit set of indices from a held JSON array and hold them \
        as a NEW handle, returned with a fact card. Use it to AUDIT an aggregate: coffer_digest / a typed \
        query reports which record indices back its number, and coffer_pick(handle, indices) fetches \
        exactly those rows so you can re-verify byte-for-byte (e.g. coffer_digest the result to recount). \
        Rows keep the order given; an out-of-range index is refused. handle may be a full handle or a \
        unique <<cof:HASH>> sentinel prefix over a JSON array."
    )]
    async fn coffer_pick(
        &self,
        Parameters(a): Parameters<PickArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let Some(bytes) = self.get_handle(&a.handle) else {
            return Ok(unknown_handle());
        };
        let Some(picked) = coffer_core::pick_rows(&bytes, &a.indices) else {
            return Ok(CallToolResult::error(vec![Content::text(
                "coffer_pick needs a JSON-array handle and in-range indices",
            )]));
        };
        let handle = self.put_bytes(&picked).as_str().to_string();
        Ok(CallToolResult::success(vec![Content::text(ingested_text(
            &handle, &picked,
        ))]))
    }

    /// Return a small row window from a held JSON array.
    #[tool(
        description = "Return rows start..start+limit from a held JSON array. Defaults to start=0, limit=20. \
        In SQLite/shared-CAS deployments, handle may be a full handle or a unique <<cof:HASH>> \
        sentinel prefix when that blob is a JSON array. Use before coffer_retrieve when you only \
        need local examples or a page of rows."
    )]
    async fn coffer_rows(
        &self,
        Parameters(a): Parameters<RowsArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let Some(bytes) = self.get_handle(&a.handle) else {
            return Ok(unknown_handle());
        };
        match render_json_rows(&bytes, a.start, a.limit, max_rows_from_env()) {
            Ok(rows) => Ok(CallToolResult::success(vec![Content::text(rows)])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e)])),
        }
    }

    /// Return one value from held JSON by a small path syntax: `$.field[0].field`.
    #[tool(
        description = "Return one value from held JSON by path. Supported syntax: $, .field, and [index], e.g. $.items[0].name. \
        In SQLite/shared-CAS deployments, handle may be a full handle or a unique <<cof:HASH>> \
        sentinel prefix when that blob is valid JSON."
    )]
    async fn coffer_json(
        &self,
        Parameters(a): Parameters<JsonPathArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let Some(bytes) = self.get_handle(&a.handle) else {
            return Ok(unknown_handle());
        };
        match render_json_path(&bytes, &a.path) {
            Ok(value) => Ok(CallToolResult::success(vec![Content::text(value)])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e)])),
        }
    }

    /// Return line-numbered text/log lines from a held payload.
    #[tool(
        description = "Return line-numbered text/log lines from a held payload. \
        Uses 1-based inclusive start_line/end_line, or head/tail for first/last N lines. \
        Defaults to the first 80 lines. In SQLite/shared-CAS deployments, handle may be a full \
        handle or a unique <<cof:HASH>> sentinel prefix."
    )]
    async fn coffer_lines(
        &self,
        Parameters(a): Parameters<LinesArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let Some(bytes) = self.get_handle(&a.handle) else {
            return Ok(unknown_handle());
        };
        match render_text_lines(&bytes, a.start_line, a.end_line, a.head, a.tail) {
            Ok(lines) => Ok(CallToolResult::success(vec![Content::text(lines)])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e)])),
        }
    }

    /// Search held text/log output and return matching line numbers.
    #[tool(
        description = "Search held text/log output and return matching line numbers. \
        Case-insensitive substring search; limit defaults to 20. In SQLite/shared-CAS deployments, \
        handle may be a full handle or a unique <<cof:HASH>> sentinel prefix."
    )]
    async fn coffer_search(
        &self,
        Parameters(a): Parameters<SearchArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let Some(bytes) = self.get_handle(&a.handle) else {
            return Ok(unknown_handle());
        };
        match render_text_search(&bytes, &a.pattern, a.limit) {
            Ok(lines) => Ok(CallToolResult::success(vec![Content::text(lines)])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e)])),
        }
    }

    /// Return a byte-exact window from the held original payload.
    #[tool(
        description = "Return a byte-exact window from the original bytes held under the handle. \
        By default returns only a bounded head window. Optional start/max_bytes returns that byte window. \
        In SQLite/shared-CAS deployments, handle may be a full handle or a unique <<cof:HASH>> \
        sentinel prefix. Set full=true only for small payloads you truly need raw; full retrieval is capped by COFFER_MCP_MAX_RETRIEVE_BYTES."
    )]
    async fn coffer_retrieve(
        &self,
        Parameters(a): Parameters<RetrieveArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        match self.get_handle(&a.handle) {
            Some(bytes) => match render_retrieved_bytes(
                &bytes,
                a.start,
                a.max_bytes,
                a.full.unwrap_or(false),
                retrieve_limits_from_env(),
            ) {
                Ok(text) => Ok(CallToolResult::success(vec![Content::text(text)])),
                Err(e) => Ok(CallToolResult::error(vec![Content::text(e)])),
            },
            None => Ok(unknown_handle()),
        }
    }

    /// Recover a byte-exact window from bytes elided behind a `<<cof:HASH …>>` compression sentinel.
    #[tool(
        description = "Recover a byte-exact window from bytes elided behind a <<cof:HASH …>> sentinel \
        (e.g. in a coffer-compressed tool output that a proxy shrank). Pass the HASH shown in the \
        sentinel; returns a bounded byte window from the shared CAS by default. Optional start/max_bytes chooses the window. Set full=true only for small payloads; full retrieval is capped by COFFER_MCP_MAX_RETRIEVE_BYTES."
    )]
    async fn coffer_unfold(
        &self,
        Parameters(a): Parameters<UnfoldArgs>,
    ) -> Result<CallToolResult, ErrorData> {
        let Ok(path) = std::env::var("COFFER_CAS_DB") else {
            return Ok(CallToolResult::error(vec![Content::text(
                "no shared CAS configured (set COFFER_CAS_DB to the proxy's database path)",
            )]));
        };
        Ok(unfold_shared_cas_result(
            path,
            &a,
            retrieve_limits_from_env(),
        ))
    }

    /// Report handle-store backend and durability metrics.
    #[tool(
        description = "Report coffer handle-store backend, resident bytes, handle count, and SQLite durability metrics."
    )]
    async fn coffer_status(&self) -> Result<CallToolResult, ErrorData> {
        Ok(CallToolResult::success(vec![Content::text(
            self.status_text(),
        )]))
    }
}

#[tool_handler(
    name = "coffer",
    instructions = "coffer holds large tool-output SERVER-SIDE so it never fills your context. \
        Direct it at data instead of reading it: coffer_run / coffer_ingest return an opaque handle \
        plus a fact card; then coffer_digest gives EXACT aggregates over ALL of it (count/sum/mean/median/\
        percentile/group-by/argmax — trust these over your own count), coffer_describe gives a generic \
        exact summary of any record set (schema + per-field stats / count-by), coffer_aggregate gives a typed \
        count|sum|mean|min|max over a predicate conjunction and returns the answer WITH its backing row \
        indices, coffer_query keeps only the rows \
        matching field <op> value, coffer_select filters by a conjunction of predicates and returns the \
        matches as a NEW handle you can narrow again server-side, coffer_pick pulls rows at explicit indices \
        (a digest's provenance) so you can audit an aggregate, coffer_search/coffer_lines drill into logs and text by line number, \
        coffer_json returns one JSON value, coffer_rows returns a small JSON row window, and coffer_retrieve \
        returns bounded byte windows. In SQLite/shared-CAS deployments those handle-taking tools can also \
        accept a unique hash prefix from a <<cof:...>> proxy sentinel; use coffer_unfold for explicit \
        sentinel byte windows. Whole-payload retrieval requires full=true and is capped. Use coffer_status for backend/durability diagnostics."
)]
impl ServerHandler for Coffer {}

#[derive(Deserialize, JsonSchema)]
struct RunArgs {
    /// Shell command whose stdout/stderr to capture and hold server-side.
    command: String,
}
#[derive(Deserialize, JsonSchema)]
struct IngestArgs {
    /// Path to a file to ingest.
    path: String,
    /// Optional compact view to return with the handle. Supported: structural_code.
    view: Option<String>,
    /// Optional token target for the compact view. Defaults to 1024 for structural_code.
    target_tokens: Option<usize>,
}
#[derive(Deserialize, JsonSchema)]
struct DigestArgs {
    /// The handle returned by coffer_run / coffer_ingest, or a unique shared-CAS sentinel prefix.
    handle: String,
    /// A natural-language aggregate, e.g. "how many commits changed more than 50 files".
    query: String,
}
#[derive(Deserialize, JsonSchema)]
struct QueryArgs {
    /// The handle returned by coffer_run / coffer_ingest, or a unique shared-CAS sentinel prefix.
    handle: String,
    /// The JSON field to filter on.
    field: String,
    /// Comparison operator: eq | ne | gt | ge | lt | le.
    op: String,
    /// The value to compare against (a number or string).
    value: String,
}
#[derive(Deserialize, JsonSchema)]
struct SelectArgs {
    /// The handle returned by coffer_run / coffer_ingest / coffer_select, or a unique shared-CAS sentinel prefix.
    handle: String,
    /// Conjunctive predicates; a row is kept only if it passes ALL of them. An empty list keeps every row.
    #[serde(rename = "where", default)]
    predicates: Vec<PredicateArg>,
}
#[derive(Deserialize, JsonSchema)]
struct DescribeArgs {
    /// The handle returned by coffer_run / coffer_ingest / coffer_select, or a unique shared-CAS sentinel prefix.
    handle: String,
}
#[derive(Deserialize, JsonSchema)]
struct AggregateArgs {
    /// The handle returned by coffer_run / coffer_ingest / coffer_select, or a unique shared-CAS sentinel prefix.
    handle: String,
    /// Conjunctive filter predicates; a row is counted only if it passes ALL of them. Empty = all rows.
    #[serde(rename = "where", default)]
    predicates: Vec<PredicateArg>,
    /// Aggregate to compute: count | sum | mean | min | max.
    agg: String,
    /// Field to aggregate over. Required for sum/mean/min/max; ignored for count.
    field: Option<String>,
}
#[derive(Deserialize, JsonSchema)]
struct PredicateArg {
    /// The JSON field to filter on.
    field: String,
    /// Comparison operator: eq | ne | gt | ge | lt | le.
    op: String,
    /// The value to compare against (a number or string).
    value: String,
}
#[derive(Deserialize, JsonSchema)]
struct PickArgs {
    /// The handle returned by coffer_run / coffer_ingest / coffer_select, or a unique shared-CAS sentinel prefix.
    handle: String,
    /// Zero-based record indices to pull (e.g. the provenance indices reported by a typed query). Order is preserved.
    indices: Vec<usize>,
}
#[derive(Deserialize, JsonSchema)]
struct RowsArgs {
    /// The handle returned by coffer_run / coffer_ingest, or a unique shared-CAS sentinel prefix.
    handle: String,
    /// Optional zero-based row offset.
    start: Option<usize>,
    /// Optional maximum number of rows to return. Defaults to 20.
    limit: Option<usize>,
}
#[derive(Deserialize, JsonSchema)]
struct JsonPathArgs {
    /// The handle returned by coffer_run / coffer_ingest, or a unique shared-CAS sentinel prefix.
    handle: String,
    /// JSON path using `$`, `.field`, and `[index]`, e.g. `$.items[0].name`.
    path: String,
}
#[derive(Deserialize, JsonSchema)]
struct LinesArgs {
    /// The handle returned by coffer_run / coffer_ingest, or a unique shared-CAS sentinel prefix.
    handle: String,
    /// Optional 1-based first line to return.
    start_line: Option<usize>,
    /// Optional 1-based inclusive last line to return.
    end_line: Option<usize>,
    /// Optional number of first lines to return.
    head: Option<usize>,
    /// Optional number of last lines to return.
    tail: Option<usize>,
}
#[derive(Deserialize, JsonSchema)]
struct SearchArgs {
    /// The handle returned by coffer_run / coffer_ingest, or a unique shared-CAS sentinel prefix.
    handle: String,
    /// Case-insensitive substring to search for.
    pattern: String,
    /// Optional maximum matching lines to return. Defaults to 20.
    limit: Option<usize>,
}
#[derive(Deserialize, JsonSchema)]
struct RetrieveArgs {
    /// The handle returned by coffer_run / coffer_ingest, or a unique shared-CAS sentinel prefix.
    handle: String,
    /// Optional byte offset to start returning from.
    start: Option<usize>,
    /// Optional maximum number of bytes to return.
    max_bytes: Option<usize>,
    /// Return the entire payload only when it is under the configured hard cap.
    full: Option<bool>,
}
#[derive(Deserialize, JsonSchema)]
struct UnfoldArgs {
    /// The hash shown inside a `<<cof:HASH …>>` sentinel (a hex prefix).
    hash: String,
    /// Optional byte offset to start returning from.
    start: Option<usize>,
    /// Optional maximum number of bytes to return.
    max_bytes: Option<usize>,
    /// Return the entire payload only when it is under the configured hard cap.
    full: Option<bool>,
}

/// Install a stderr-only, env-filtered tracing subscriber so coffer-cas durability and
/// corruption warnings reach operators. **Stderr only**: stdout is the MCP JSON-RPC channel and any
/// stray byte on it corrupts the protocol. Filter precedence: `RUST_LOG`, then `COFFER_LOG`, then
/// `default_directives`. Fail-open: a bad filter falls back to the default and `try_init` is a no-op
/// if a subscriber is already installed (e.g. under a test harness).
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
    // Default to warn: keep the JSON-RPC peer's stderr quiet, but surface coffer-cas durability /
    // corruption warnings (e.g. an offloaded original that failed to persist) at failure time.
    init_tracing("warn");
    let service = Coffer::new()?.serve(rmcp::transport::stdio()).await?;
    service.waiting().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        AggregateArgs, Coffer, DEFAULT_MAX_ROWS, DescribeArgs, IngestView, PickArgs, PredicateArg,
        QueryArgs, RetrieveLimits, RunLimits, SelectArgs, UnfoldArgs, byte_window, fact_card,
        ingest_view, ingested_text_with_view, positive_usize_from_value, render_json_path,
        render_json_rows, render_retrieved_bytes, render_text_lines, render_text_search,
        retrieve_limits_from_values, run_limits_from_values, run_policy_from_values,
        run_shell_command, unfold_shared_cas_result,
    };
    use coffer_cas::{Cas, ContentHash, MemoryCas, SqliteCas};
    use rmcp::handler::server::wrapper::Parameters;
    use rmcp::model::CallToolResult;

    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            use std::sync::atomic::{AtomicU32, Ordering};
            static N: AtomicU32 = AtomicU32::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            let dir =
                std::env::temp_dir().join(format!("coffer-mcp-{}-{tag}-{n}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn db(&self) -> std::path::PathBuf {
            self.0.join("cas.db")
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn tool_text(result: &CallToolResult) -> &str {
        result.content[0].as_text().unwrap().text.as_str()
    }

    #[test]
    fn ingest_view_rejects_unknown_values() {
        let err = ingest_view(Some("whole_repo_map")).unwrap_err();
        assert!(err.contains("supported values: structural_code"), "{err}");
    }

    #[test]
    fn structural_code_ingest_view_keeps_outline_and_retrieval_sentinel() {
        let input = br#"
use std::collections::HashMap;

pub struct Widget {
    value: usize,
}

impl Widget {
    pub fn compute(&self, items: &[usize]) -> usize {
        let mut total = self.value;
        for item in items {
            total += item;
        }
        total
    }
}
"#;
        let handle = ContentHash::of(input).as_str().to_string();
        let cas = MemoryCas::new();
        let text =
            ingested_text_with_view(&handle, input, IngestView::StructuralCode, Some(120), &cas);
        assert!(text.contains(&format!("handle: {handle}")), "{text}");
        assert!(text.contains(&format!("<<cof:{}", &handle[..12])), "{text}");
        assert!(text.contains("view: structural_code"), "{text}");
        assert!(text.contains("pub struct Widget"), "{text}");
        assert!(text.contains("pub fn compute"), "{text}");
        assert!(!text.contains("total += item"), "{text}");
    }

    #[tokio::test]
    async fn query_sentinel_is_retrievable_from_sqlite_store() {
        let dir = TempDir::new("query-sentinel");
        let server = Coffer::with_sqlite(dir.db()).unwrap();
        let rows = (0..30)
            .map(|i| {
                serde_json::json!({
                    "id": i,
                    "status": if i == 17 { "open" } else { "closed" },
                    "payload": format!("row-{i:02}-{}", "x".repeat(200)),
                })
            })
            .collect::<Vec<_>>();
        let input = serde_json::to_vec(&rows).unwrap();
        let handle = server.put_bytes(&input).as_str().to_string();

        let result = server
            .coffer_query(Parameters(QueryArgs {
                handle,
                field: "id".to_string(),
                op: "eq".to_string(),
                value: "17".to_string(),
            }))
            .await
            .unwrap();
        let text = tool_text(&result);
        let prefix = text
            .split_once("<<cof:")
            .and_then(|(_, rest)| rest.split_whitespace().next())
            .expect("query should elide non-matching rows");

        let result = unfold_shared_cas_result(
            dir.db(),
            &UnfoldArgs {
                hash: prefix.to_string(),
                start: Some(0),
                max_bytes: Some(80),
                full: None,
            },
            RetrieveLimits {
                default_bytes: 80,
                max_bytes: 256,
            },
        );

        assert_eq!(result.is_error, Some(false), "{}", tool_text(&result));
        assert!(
            tool_text(&result).contains("\"id\":0"),
            "{}",
            tool_text(&result)
        );
    }

    fn handle_of(card: &str) -> String {
        card.lines()
            .next()
            .unwrap()
            .strip_prefix("handle: ")
            .unwrap()
            .to_string()
    }

    #[tokio::test]
    async fn coffer_select_chains_handles_server_side() {
        let server = Coffer::in_memory();
        let rows = (0..20)
            .map(|i| {
                serde_json::json!({
                    "a": i,
                    "status": if i % 2 == 0 { "error" } else { "ok" },
                })
            })
            .collect::<Vec<_>>();
        let input = serde_json::to_vec(&rows).unwrap();
        let handle = server.put_bytes(&input).as_str().to_string();

        // select a > 10 → a NEW handle (the derived subset lives in the shared CAS).
        let r1 = server
            .coffer_select(Parameters(SelectArgs {
                handle,
                predicates: vec![PredicateArg {
                    field: "a".into(),
                    op: "gt".into(),
                    value: "10".into(),
                }],
            }))
            .await
            .unwrap();
        assert_eq!(r1.is_error, Some(false), "{}", tool_text(&r1));
        let h1 = handle_of(tool_text(&r1));

        // chain: select status == "error" on the DERIVED handle, server-side.
        let r2 = server
            .coffer_select(Parameters(SelectArgs {
                handle: h1,
                predicates: vec![PredicateArg {
                    field: "status".into(),
                    op: "eq".into(),
                    value: "error".into(),
                }],
            }))
            .await
            .unwrap();
        assert_eq!(r2.is_error, Some(false), "{}", tool_text(&r2));
        let h2 = handle_of(tool_text(&r2));

        // the chained handle equals applying both predicates at once (composition holds across handles).
        let got = server.get_handle(&h2).unwrap();
        let parsed: Vec<serde_json::Value> = serde_json::from_slice(&got).unwrap();
        let want: Vec<serde_json::Value> = rows
            .iter()
            .filter(|r| r["a"].as_i64().unwrap() > 10 && r["status"].as_str() == Some("error"))
            .cloned()
            .collect();
        assert_eq!(parsed, want);
        // a > 10 and even: 12, 14, 16, 18 → 4 rows.
        assert_eq!(parsed.len(), 4);
    }

    #[tokio::test]
    async fn coffer_describe_summarizes_a_records_handle() {
        let server = Coffer::in_memory();
        let rows = (0..20)
            .map(|i| serde_json::json!({ "v": i, "status": if i % 4 == 0 { "error" } else { "ok" } }))
            .collect::<Vec<_>>();
        let input = serde_json::to_vec(&rows).unwrap();
        let handle = server.put_bytes(&input).as_str().to_string();

        let r = server
            .coffer_describe(Parameters(DescribeArgs { handle }))
            .await
            .unwrap();
        assert_eq!(r.is_error, Some(false), "{}", tool_text(&r));
        let t = tool_text(&r);
        assert!(t.contains("20 records"), "{t}");
        assert!(t.contains("sum=190"), "{t}"); // 0+1+..+19
        assert!(
            t.contains(r#""error":5"#) && t.contains(r#""ok":15"#),
            "{t}"
        ); // exact count-by

        // non-array handle is refused.
        let h2 = server.put_bytes(b"not json").as_str().to_string();
        let bad = server
            .coffer_describe(Parameters(DescribeArgs { handle: h2 }))
            .await
            .unwrap();
        assert_eq!(bad.is_error, Some(true), "{}", tool_text(&bad));
    }

    #[tokio::test]
    async fn coffer_aggregate_reports_value_and_provenance() {
        let server = Coffer::in_memory();
        let rows = (0..10)
            .map(|i| serde_json::json!({ "a": i, "b": i * 2 }))
            .collect::<Vec<_>>();
        let input = serde_json::to_vec(&rows).unwrap();
        let handle = server.put_bytes(&input).as_str().to_string();

        // sum b where a >= 7  -> rows 7,8,9 -> b = 14+16+18 = 48
        let r = server
            .coffer_aggregate(Parameters(AggregateArgs {
                handle,
                predicates: vec![PredicateArg {
                    field: "a".into(),
                    op: "ge".into(),
                    value: "7".into(),
                }],
                agg: "sum".into(),
                field: Some("b".into()),
            }))
            .await
            .unwrap();
        assert_eq!(r.is_error, Some(false), "{}", tool_text(&r));
        let text = tool_text(&r);
        assert!(text.contains("= 48"), "{text}");
        // provenance indices (7,8,9) are reported so they can feed coffer_pick.
        assert!(text.contains("provenance row indices"), "{text}");
        assert!(
            text.contains('7') && text.contains('8') && text.contains('9'),
            "{text}"
        );

        // a non-numeric aggregated field refuses rather than guessing.
        let strs = serde_json::to_vec(&serde_json::json!([{ "a": 1, "b": "x" }])).unwrap();
        let h2 = server.put_bytes(&strs).as_str().to_string();
        let bad = server
            .coffer_aggregate(Parameters(AggregateArgs {
                handle: h2,
                predicates: vec![],
                agg: "sum".into(),
                field: Some("b".into()),
            }))
            .await
            .unwrap();
        assert_eq!(bad.is_error, Some(true), "{}", tool_text(&bad));
    }

    #[tokio::test]
    async fn coffer_aggregate_wiring_edges_and_pick_roundtrip() {
        let server = Coffer::in_memory();
        let rows = (0..6)
            .map(|i| serde_json::json!({ "a": i }))
            .collect::<Vec<_>>();
        let input = serde_json::to_vec(&rows).unwrap();
        let handle = server.put_bytes(&input).as_str().to_string();
        let agg = |op: &str, field: Option<&str>| AggregateArgs {
            handle: handle.clone(),
            predicates: vec![PredicateArg {
                field: "a".into(),
                op: "ge".into(),
                value: "3".into(),
            }],
            agg: op.to_string(),
            field: field.map(str::to_string),
        };

        // sum/mean/min/max WITHOUT a field error (never silently count); unknown agg errors.
        for op in ["sum", "mean", "min", "max", "bogus"] {
            let r = server
                .coffer_aggregate(Parameters(agg(op, None)))
                .await
                .unwrap();
            assert_eq!(r.is_error, Some(true), "{op} without field must error");
        }
        // count needs no field.
        let c = server
            .coffer_aggregate(Parameters(agg("count", None)))
            .await
            .unwrap();
        assert_eq!(c.is_error, Some(false), "{}", tool_text(&c));

        // "avg" aliases mean: mean of a in {3,4,5} = 4.
        let m = server
            .coffer_aggregate(Parameters(agg("avg", Some("a"))))
            .await
            .unwrap();
        assert!(tool_text(&m).contains("= 4"), "{}", tool_text(&m));

        // provenance round-trip: the reported indices, fed to coffer_pick, return exactly those rows.
        let prov = server
            .coffer_aggregate(Parameters(agg("count", None)))
            .await
            .unwrap();
        let text = tool_text(&prov);
        assert!(text.contains("provenance row indices: [3, 4, 5]"), "{text}");
        let picked = server
            .coffer_pick(Parameters(PickArgs {
                handle: handle.clone(),
                indices: vec![3, 4, 5],
            }))
            .await
            .unwrap();
        let bytes = server.get_handle(&handle_of(tool_text(&picked))).unwrap();
        let parsed: Vec<serde_json::Value> = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            parsed,
            vec![rows[3].clone(), rows[4].clone(), rows[5].clone()]
        );
    }

    #[tokio::test]
    async fn coffer_pick_pulls_provenance_rows() {
        let server = Coffer::in_memory();
        let rows = (0..6)
            .map(|i| serde_json::json!({ "id": i, "v": i * 10 }))
            .collect::<Vec<_>>();
        let input = serde_json::to_vec(&rows).unwrap();
        let handle = server.put_bytes(&input).as_str().to_string();

        // pull records 1 and 4 (e.g. a digest's provenance), as a NEW handle.
        let r = server
            .coffer_pick(Parameters(PickArgs {
                handle,
                indices: vec![1, 4],
            }))
            .await
            .unwrap();
        assert_eq!(r.is_error, Some(false), "{}", tool_text(&r));
        let picked = server.get_handle(&handle_of(tool_text(&r))).unwrap();
        let parsed: Vec<serde_json::Value> = serde_json::from_slice(&picked).unwrap();
        assert_eq!(parsed, vec![rows[1].clone(), rows[4].clone()]);
    }

    #[test]
    fn fact_card_includes_fields_absent_from_first_row() {
        let bytes = br#"[
            {"id": 1, "status": "seed"},
            {"id": 2, "latency_ms": 120, "status": "ok"},
            {"id": 3, "latency_ms": 240, "error_code": "E42"}
        ]"#;

        let card = fact_card(bytes).expect("JSON rows should produce field stats");

        assert!(
            card.contains("id: numeric present=3/3 min=1 max=3 mean=2.0000"),
            "{card}"
        );
        assert!(card.contains("status: present=2/3 2 distinct"), "{card}");
        assert!(
            card.contains("latency_ms: numeric present=2/3 min=120 max=240 mean=180.0000"),
            "{card}"
        );
        assert!(
            card.contains("error_code: present=1/3 1 distinct"),
            "{card}"
        );
    }

    #[test]
    fn byte_window_defaults_to_full_payload() {
        let window = byte_window(b"abcdef", None, None);
        assert_eq!(window.start, 0);
        assert_eq!(window.end, 6);
        assert_eq!(window.total, 6);
        assert_eq!(window.bytes, b"abcdef");
    }

    #[test]
    fn byte_window_applies_start_and_max_bytes() {
        let window = byte_window(b"abcdef", Some(2), Some(3));
        assert_eq!(window.start, 2);
        assert_eq!(window.end, 5);
        assert_eq!(window.total, 6);
        assert_eq!(window.bytes, b"cde");
    }

    #[test]
    fn byte_window_clamps_out_of_range_start() {
        let window = byte_window(b"abc", Some(99), Some(10));
        assert_eq!(window.start, 3);
        assert_eq!(window.end, 3);
        assert_eq!(window.total, 3);
        assert_eq!(window.bytes, b"");
    }

    #[test]
    fn render_full_payload_preserves_legacy_no_header_shape() {
        let limits = RetrieveLimits {
            default_bytes: 64,
            max_bytes: 64,
        };
        assert_eq!(
            render_retrieved_bytes(b"abcdef", None, None, false, limits).unwrap(),
            "abcdef"
        );
    }

    #[test]
    fn render_window_includes_offsets_and_omission_counts() {
        let limits = RetrieveLimits {
            default_bytes: 64,
            max_bytes: 64,
        };
        assert_eq!(
            render_retrieved_bytes(b"abcdef", Some(1), Some(2), false, limits).unwrap(),
            "bytes 1..3 of 6 (1 before, 3 after)\nbc"
        );
    }

    #[test]
    fn render_default_retrieve_is_bounded_window() {
        let limits = RetrieveLimits {
            default_bytes: 3,
            max_bytes: 10,
        };
        assert_eq!(
            render_retrieved_bytes(b"abcdef", None, None, false, limits).unwrap(),
            "bytes 0..3 of 6 (0 before, 3 after)\nabc"
        );
    }

    #[test]
    fn render_full_retrieve_requires_payload_under_hard_cap() {
        let limits = RetrieveLimits {
            default_bytes: 3,
            max_bytes: 6,
        };
        assert_eq!(
            render_retrieved_bytes(b"abcdef", None, None, true, limits).unwrap(),
            "abcdef"
        );
        let err = render_retrieved_bytes(b"abcdefg", None, None, true, limits).unwrap_err();
        assert!(err.contains("COFFER_MCP_MAX_RETRIEVE_BYTES=6"), "{err}");
    }

    #[test]
    fn render_rejects_window_over_hard_cap() {
        let limits = RetrieveLimits {
            default_bytes: 3,
            max_bytes: 4,
        };
        let err = render_retrieved_bytes(b"abcdef", Some(0), Some(5), false, limits).unwrap_err();
        assert_eq!(
            err,
            "requested max_bytes=5 exceeds COFFER_MCP_MAX_RETRIEVE_BYTES=4"
        );
    }

    #[test]
    fn render_full_rejects_window_arguments() {
        let limits = RetrieveLimits {
            default_bytes: 3,
            max_bytes: 10,
        };
        let err = render_retrieved_bytes(b"abcdef", Some(1), None, true, limits).unwrap_err();
        assert_eq!(err, "full=true cannot be combined with start or max_bytes");
    }

    #[test]
    fn unfold_shared_cas_returns_bounded_window_from_sentinel_hash() {
        let dir = TempDir::new("unfold-window");
        let cas = SqliteCas::open(dir.db()).unwrap();
        let hash = cas.put(b"0123456789");
        cas.flush();

        let result = unfold_shared_cas_result(
            dir.db(),
            &UnfoldArgs {
                hash: hash.short().to_string(),
                start: Some(3),
                max_bytes: Some(4),
                full: None,
            },
            RetrieveLimits {
                default_bytes: 3,
                max_bytes: 8,
            },
        );

        assert_eq!(result.is_error, Some(false));
        assert_eq!(
            tool_text(&result),
            "bytes 3..7 of 10 (3 before, 3 after)\n3456"
        );
    }

    #[test]
    fn unfold_shared_cas_enforces_full_retrieve_cap() {
        let dir = TempDir::new("unfold-full-cap");
        let cas = SqliteCas::open(dir.db()).unwrap();
        let hash = cas.put(b"0123456789");
        cas.flush();

        let result = unfold_shared_cas_result(
            dir.db(),
            &UnfoldArgs {
                hash: hash.short().to_string(),
                start: None,
                max_bytes: None,
                full: Some(true),
            },
            RetrieveLimits {
                default_bytes: 3,
                max_bytes: 8,
            },
        );

        assert_eq!(result.is_error, Some(true));
        assert!(
            tool_text(&result).contains("COFFER_MCP_MAX_RETRIEVE_BYTES=8"),
            "{}",
            tool_text(&result)
        );
    }

    #[test]
    fn sqlite_store_accepts_unique_sentinel_prefix_as_handle() {
        let dir = TempDir::new("sqlite-prefix-handle");
        let hash = {
            let cas = SqliteCas::open(dir.db()).unwrap();
            let hash = cas.put(br#"[{"id":0},{"id":1},{"id":2}]"#);
            cas.flush();
            hash
        };
        let server = Coffer::with_sqlite(dir.db()).unwrap();

        let bytes = server
            .get_handle(hash.short())
            .expect("unique sentinel prefix should resolve in shared SQLite CAS");
        assert_eq!(&bytes[..], br#"[{"id":0},{"id":1},{"id":2}]"#);
        let rows = render_json_rows(&bytes, Some(1), Some(1), DEFAULT_MAX_ROWS).unwrap();
        assert!(rows.contains(r#""id": 1"#), "{rows}");

        assert!(
            server.get_handle(&hash.as_str()[..7]).is_none(),
            "prefixes shorter than the shared-CAS sentinel floor must not resolve"
        );
    }

    #[test]
    fn render_json_rows_returns_a_small_window_with_offsets() {
        let rows = br#"[{"id":0},{"id":1},{"id":2},{"id":3}]"#;
        let got = render_json_rows(rows, Some(1), Some(2), DEFAULT_MAX_ROWS).unwrap();
        assert!(got.starts_with("rows 1..3 of 4 (1 before, 1 after)\n"));
        assert!(got.contains(r#""id": 1"#), "{got}");
        assert!(got.contains(r#""id": 2"#), "{got}");
        assert!(!got.contains(r#""id": 0"#), "{got}");
        assert!(!got.contains(r#""id": 3"#), "{got}");
    }

    #[test]
    fn render_json_rows_rejects_non_arrays() {
        let err = render_json_rows(br#"{"id":1}"#, None, None, DEFAULT_MAX_ROWS).unwrap_err();
        assert_eq!(err, "held data is JSON but not an array");
    }

    #[test]
    fn render_json_rows_clamps_oversized_limit_to_max_rows() {
        // A 50-element array with a deliberately huge requested limit: the page must be clamped to
        // max_rows so coffer_rows cannot re-bloat the model context, while `total` still reports 50.
        let items: Vec<String> = (0..50).map(|i| format!(r#"{{"id":{i}}}"#)).collect();
        let array = format!("[{}]", items.join(","));
        let got = render_json_rows(array.as_bytes(), None, Some(usize::MAX), 10).unwrap();
        assert!(
            got.starts_with("rows 0..10 of 50 (0 before, 40 after)\n"),
            "{got}"
        );
        assert!(got.contains(r#""id": 9"#), "{got}");
        assert!(!got.contains(r#""id": 10"#), "{got}");
    }

    #[test]
    fn render_json_path_selects_nested_value() {
        let json = br#"{"items":[{"name":"alpha"},{"name":"beta"}],"count":2}"#;
        let got = render_json_path(json, "$.items[1].name").unwrap();
        assert_eq!(got, "json path $.items[1].name\n\"beta\"");
    }

    #[test]
    fn render_json_path_reports_missing_field() {
        let err = render_json_path(br#"{"items":[]}"#, "$.missing").unwrap_err();
        assert_eq!(err, "path field not found: missing");
    }

    #[test]
    fn render_text_lines_returns_numbered_range() {
        let got =
            render_text_lines(b"one\ntwo\nthree\nfour\n", Some(2), Some(3), None, None).unwrap();
        assert_eq!(
            got,
            "lines 2..3 of 4 (1 before, 1 after)\n     2|two\n     3|three"
        );
    }

    #[test]
    fn render_text_lines_returns_tail() {
        let got = render_text_lines(b"one\ntwo\nthree\nfour\n", None, None, None, Some(2)).unwrap();
        assert_eq!(
            got,
            "lines 3..4 of 4 (2 before, 0 after)\n     3|three\n     4|four"
        );
    }

    #[test]
    fn render_text_lines_rejects_conflicting_selectors() {
        let err = render_text_lines(b"one\n", Some(1), None, Some(1), None).unwrap_err();
        assert_eq!(err, "choose head/tail or start_line/end_line, not both");
    }

    #[test]
    fn render_text_search_returns_matching_line_numbers() {
        let got =
            render_text_search(b"INFO start\nError failed\ninfo retry\n", "error", None).unwrap();
        assert_eq!(
            got,
            "1 matches for \"error\" in 3 lines (showing 1, 0 omitted)\n     2|Error failed"
        );
    }

    #[test]
    fn render_text_search_limits_and_reports_omitted_matches() {
        let got = render_text_search(b"error a\nok\nerror b\nerror c\n", "error", Some(2)).unwrap();
        assert_eq!(
            got,
            "3 matches for \"error\" in 4 lines (showing 2, 1 omitted)\n     1|error a\n     3|error b"
        );
    }

    #[test]
    fn render_text_search_reports_no_matches() {
        let got = render_text_search(b"alpha\nbeta\n", "gamma", None).unwrap();
        assert_eq!(got, "0 matches for \"gamma\" in 2 lines");
    }

    #[test]
    fn sqlite_handle_store_survives_new_server_instance() {
        let dir = TempDir::new("persistent-handles");
        let handle = {
            let server = Coffer::with_sqlite(dir.db()).unwrap();
            let handle = server
                .put_bytes(b"persistent stderr\nerror[E0308]\n")
                .as_str()
                .to_string();
            assert_eq!(
                server.get_handle(&handle).as_deref(),
                Some(&b"persistent stderr\nerror[E0308]\n"[..])
            );
            handle
        };

        let reopened = Coffer::with_sqlite(dir.db()).unwrap();
        assert_eq!(
            reopened.get_handle(&handle).as_deref(),
            Some(&b"persistent stderr\nerror[E0308]\n"[..])
        );
    }

    #[test]
    fn sqlite_handle_store_creates_parent_directory() {
        let dir = TempDir::new("sqlite-parent");
        let db = dir.0.join("nested").join("session.db");

        let server = Coffer::with_sqlite(&db).unwrap();
        let handle = server.put_bytes(b"parent dir created").as_str().to_string();

        assert!(db.exists());
        assert_eq!(
            server.get_handle(&handle).as_deref(),
            Some(&b"parent dir created"[..])
        );
    }

    #[test]
    fn status_reports_memory_store_metrics() {
        let server = Coffer::in_memory();
        server.put_bytes(b"abc");
        server.put_bytes(b"defg");

        let status = server.status_text();

        assert!(status.contains("store: memory"), "{status}");
        assert!(status.contains("handles: 2"), "{status}");
        assert!(status.contains("resident_bytes: 7"), "{status}");
        assert!(status.contains("retrieve_default_bytes:"), "{status}");
        assert!(status.contains("retrieve_max_bytes:"), "{status}");
    }

    #[test]
    fn status_reports_sqlite_durability_metrics() {
        let dir = TempDir::new("sqlite-status");
        let server = Coffer::with_sqlite(dir.db()).unwrap();
        server.put_bytes(b"sqlite status");

        let status = server.status_text();

        assert!(status.contains("store: sqlite"), "{status}");
        assert!(status.contains("handles: 1"), "{status}");
        assert!(status.contains("resident_bytes: 13"), "{status}");
        assert!(status.contains("sqlite_db_bytes:"), "{status}");
        assert!(status.contains("sqlite_wal_bytes:"), "{status}");
        assert!(status.contains("sqlite_shm_bytes:"), "{status}");
        assert!(status.contains("sqlite_total_bytes:"), "{status}");
        assert!(status.contains("soft_cap_bytes:"), "{status}");
        assert!(status.contains("resident_cap_bytes:"), "{status}");
        assert!(status.contains("resident_evictions: 0"), "{status}");
        assert!(status.contains("warm_bytes_on_open: false"), "{status}");
        assert!(status.contains("trust_hashes_on_open: false"), "{status}");
        assert!(status.contains("checkpoint_every_blobs:"), "{status}");
        assert!(status.contains("wal_checkpoints:"), "{status}");
        assert!(status.contains("wal_checkpoint_failures: 0"), "{status}");
        assert!(status.contains("durability_lag: 0"), "{status}");
        assert!(status.contains("dropped_writes: 0"), "{status}");
        assert!(status.contains("persisted_blobs_this_run: 1"), "{status}");
        assert!(status.contains("retrieve_default_bytes:"), "{status}");
        assert!(status.contains("retrieve_max_bytes:"), "{status}");
    }

    #[test]
    fn sqlite_soft_cap_parser_uses_mebibytes_and_ignores_zero_or_bad_values() {
        assert_eq!(
            super::sqlite_soft_cap_bytes_from_value(Some("2")),
            Some(2 * 1024 * 1024)
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

    #[test]
    fn positive_usize_parser_ignores_zero_or_bad_values() {
        assert_eq!(positive_usize_from_value(Some("1")), Some(1));
        assert_eq!(positive_usize_from_value(Some(" 4096 ")), Some(4096));
        assert_eq!(positive_usize_from_value(Some("0")), None);
        assert_eq!(positive_usize_from_value(Some("bad")), None);
        assert_eq!(positive_usize_from_value(None), None);
    }

    #[test]
    fn run_policy_disabled_by_default_and_enables_on_truthy() {
        // Off unless explicitly enabled — the kill-switch.
        assert!(run_policy_from_values(None, None).permits("ls").is_err());
        assert!(
            run_policy_from_values(Some("0"), None)
                .permits("ls")
                .is_err()
        );
        assert!(
            run_policy_from_values(Some("false"), None)
                .permits("ls")
                .is_err()
        );
        assert!(
            run_policy_from_values(Some("1"), None)
                .permits("ls")
                .is_ok()
        );
        assert!(
            run_policy_from_values(Some("YES"), None)
                .permits("anything goes")
                .is_ok()
        );
    }

    #[test]
    fn run_policy_allowlist_restricts_to_prefixes() {
        let p = run_policy_from_values(Some("1"), Some("kubectl, git "));
        assert!(p.permits("kubectl get pods").is_ok());
        assert!(p.permits("  git status").is_ok()); // leading whitespace trimmed before matching
        assert!(p.permits("rm -rf /").is_err()); // not on the allowlist
        assert!(p.permits("github --help").is_err()); // prefix must end at a word boundary
        assert!(p.permits("git status; cat ~/.ssh/id_rsa").is_err());
        assert!(p.permits("git status && cat ~/.ssh/id_rsa").is_err());
        assert!(p.permits("git status | cat").is_err());
        let subcommand = run_policy_from_values(Some("1"), Some("git status"));
        assert!(subcommand.permits("git status --short").is_ok());
        assert!(subcommand.permits("git statusx").is_err());
        // an allowlist that is only separators/space imposes no restriction (enabled still required)
        let empty = run_policy_from_values(Some("1"), Some(" , "));
        assert!(empty.permits("rm -rf /").is_ok());
        // allowlist without enable is still refused (enable is the primary guard)
        assert!(
            run_policy_from_values(None, Some("kubectl"))
                .permits("kubectl get pods")
                .is_err()
        );
    }

    #[test]
    fn run_limit_parser_uses_defaults_and_positive_overrides() {
        assert_eq!(
            run_limits_from_values(Some("7"), Some("2")),
            RunLimits {
                timeout_seconds: 7,
                max_output_bytes: 2 * super::MIB,
            }
        );
        assert_eq!(
            run_limits_from_values(Some("0"), Some("bad")),
            RunLimits {
                timeout_seconds: super::DEFAULT_RUN_TIMEOUT_SECONDS,
                max_output_bytes: super::DEFAULT_MAX_RUN_OUTPUT_BYTES,
            }
        );
    }

    #[tokio::test]
    async fn run_shell_command_times_out_and_returns_partial_capture() {
        let capture = run_shell_command(
            "printf before; sleep 2; printf after",
            RunLimits {
                timeout_seconds: 1,
                max_output_bytes: 1024,
            },
        )
        .await
        .unwrap();

        assert!(capture.timed_out);
        assert!(!capture.output_truncated);
        assert_eq!(capture.status, None);
        assert_eq!(String::from_utf8_lossy(&capture.bytes), "before");
    }

    #[tokio::test]
    async fn run_shell_command_enforces_output_cap() {
        let capture = run_shell_command(
            "i=0; while [ $i -lt 20 ]; do printf 0123456789; i=$((i + 1)); done",
            RunLimits {
                timeout_seconds: 10,
                max_output_bytes: 64,
            },
        )
        .await
        .unwrap();

        assert!(!capture.timed_out);
        assert!(capture.output_truncated);
        assert_eq!(capture.status, None);
        assert_eq!(capture.bytes.len(), 64);
    }

    #[test]
    fn retrieve_limits_parser_uses_defaults_and_caps_default_to_max() {
        assert_eq!(
            retrieve_limits_from_values(None, None),
            RetrieveLimits {
                default_bytes: 64 * 1024,
                max_bytes: 1024 * 1024
            }
        );
        assert_eq!(
            retrieve_limits_from_values(Some("20"), Some("10")),
            RetrieveLimits {
                default_bytes: 10,
                max_bytes: 10
            }
        );
        assert_eq!(
            retrieve_limits_from_values(Some("bad"), Some("30")),
            RetrieveLimits {
                default_bytes: 30,
                max_bytes: 30
            }
        );
    }
}
