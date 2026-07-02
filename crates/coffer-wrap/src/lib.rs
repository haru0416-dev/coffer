//! coffer-wrap: wrap any stdio MCP server so oversized tool results become verified,
//! queryable CAS handles instead of context floods or hard client-side failures.
//!
//! The relay forwards JSON-RPC traffic between an MCP client (our stdin/stdout) and a
//! wrapped downstream server (a child process) with exactly two interventions:
//!
//! 1. `tools/list` responses gain a small set of injected query tools. Injection is
//!    collision-aware: a downstream tool with the same name is never shadowed — the
//!    injected tool is renamed (`wrap_` prefix), or skipped if renaming still collides.
//! 2. `tools/call` responses whose text content exceeds a token threshold have that
//!    content stored byte-exact in a SHA-256 content-addressed store and replaced with a
//!    compact fact card carrying the handle. The injected tools then answer
//!    describe/digest/aggregate/rows/search/lines/retrieve queries against the stored
//!    bytes without the payload ever entering the model context (search is row-aware on
//!    JSON arrays and its matches feed straight into `rows` for verbatim fetches).
//!
//! Everything else — notifications, server-initiated requests, unknown methods, error
//! results, malformed lines — is forwarded verbatim (fail-open). The wrap layer never
//! invents data: query tools reuse coffer-core's refuse-rather-than-guess digest and
//! aggregation, and `retrieve` returns the stored bytes, which the CAS layer re-verifies
//! against their hash on read.
//!
//! Known v1 limits, chosen deliberately: `structuredContent` is passed through untouched
//! (rewriting it would violate the tool's declared `outputSchema`), `isError` results are
//! never offloaded (a large error must stay visible), and collision renames take effect
//! after the first `tools/list` response (clients list before they call).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use coffer_cas::{Cas, ContentHash, MemoryCas, SqliteCas};
use coffer_core::dataset::DatasetCache;
use coffer_core::{Agg, Op, Predicate, pick_rows};
use coffer_tokenizer::{HeuristicCounter, TokenCounter};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};

/// Relay configuration. Every knob has an environment-variable override in the binary.
#[derive(Clone, Debug)]
pub struct WrapConfig {
    /// Offload a text content block when its heuristic token estimate exceeds this.
    /// Default 10,000 — the MCP hosts' warning band, well under common 25k hard caps.
    pub threshold_tokens: usize,
    /// Name prefix for injected tools. Default `coffer_`.
    pub prefix: String,
    /// Characters of raw preview kept in the fact card. Default 400.
    pub preview_chars: usize,
    /// Default byte window for `retrieve`. Default 64 KiB.
    pub retrieve_default_bytes: usize,
    /// Hard cap for a single `retrieve` window. Default 1 MiB.
    pub retrieve_max_bytes: usize,
}

impl Default for WrapConfig {
    fn default() -> Self {
        Self {
            threshold_tokens: 10_000,
            prefix: "coffer_".to_string(),
            preview_chars: 400,
            retrieve_default_bytes: 64 * 1024,
            retrieve_max_bytes: 1024 * 1024,
        }
    }
}

impl WrapConfig {
    /// Build a config from `COFFER_WRAP_*` environment variables, falling back to defaults.
    #[must_use]
    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        if let Some(v) = env_usize("COFFER_WRAP_THRESHOLD_TOKENS") {
            cfg.threshold_tokens = v;
        }
        if let Ok(p) = std::env::var("COFFER_WRAP_PREFIX")
            && !p.trim().is_empty()
        {
            cfg.prefix = p.trim().to_string();
        }
        if let Some(v) = env_usize("COFFER_WRAP_PREVIEW_CHARS") {
            cfg.preview_chars = v;
        }
        if let Some(v) = env_usize("COFFER_WRAP_RETRIEVE_MAX_BYTES") {
            cfg.retrieve_max_bytes = v;
        }
        cfg
    }
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.trim().parse().ok()
}

/// Where offloaded originals live. SQLite (via `COFFER_CAS_DB`) shares the store with
/// coffer-proxy / coffer-mcp across processes; memory is the zero-setup default.
pub enum HandleStore {
    /// In-memory store — handles live only as long as this wrap process.
    Memory(MemoryCas),
    /// Persistent write-through store, flushed after every put so a sibling process
    /// (or a wrap restart) can resolve the handle immediately. Boxed: `SqliteCas` is a
    /// large struct and would dominate the enum's size.
    Sqlite(Box<SqliteCas>),
}

impl HandleStore {
    fn put(&self, bytes: &[u8]) -> ContentHash {
        match self {
            Self::Memory(m) => m.put(bytes),
            Self::Sqlite(s) => {
                let hash = s.put(bytes);
                s.flush();
                hash
            }
        }
    }

    fn get(&self, hash: &ContentHash) -> Option<Arc<[u8]>> {
        match self {
            Self::Memory(m) => m.get(hash),
            Self::Sqlite(s) => s.get(hash),
        }
    }
}

/// The injected query tools, in advertisement order.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ToolKind {
    Describe,
    Digest,
    Aggregate,
    Rows,
    Search,
    Lines,
    Retrieve,
}

const BASE_TOOLS: [(ToolKind, &str); 7] = [
    (ToolKind::Describe, "describe"),
    (ToolKind::Digest, "digest"),
    (ToolKind::Aggregate, "aggregate"),
    (ToolKind::Rows, "rows"),
    (ToolKind::Search, "search"),
    (ToolKind::Lines, "lines"),
    (ToolKind::Retrieve, "retrieve"),
];

/// What we owe the downstream server an intercept for, keyed by request id.
enum Pending {
    ToolsList,
    ToolCall,
}

struct RelayState {
    pending: Mutex<HashMap<String, Pending>>,
    /// Currently advertised (kind, name) pairs — defaults at startup, refreshed with
    /// collision-resolved names on every final `tools/list` page.
    names: Mutex<Vec<(ToolKind, String)>>,
    store: HandleStore,
    /// Parsed-dataset LRU keyed by handle: the offload's fact card pre-warms it, so the
    /// first follow-up query already skips the parse. Sound: handles are content-addressed.
    datasets: DatasetCache,
    cfg: WrapConfig,
}

impl RelayState {
    fn new(store: HandleStore, cfg: WrapConfig) -> Self {
        let names = BASE_TOOLS
            .iter()
            .map(|(kind, base)| (*kind, format!("{}{base}", cfg.prefix)))
            .collect();
        Self {
            pending: Mutex::new(HashMap::new()),
            names: Mutex::new(names),
            store,
            datasets: DatasetCache::new(4),
            cfg,
        }
    }

    fn kind_for(&self, name: &str) -> Option<ToolKind> {
        self.names
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .find(|(_, n)| n == name)
            .map(|(kind, _)| *kind)
    }

    fn name_of(&self, kind: ToolKind) -> String {
        self.names
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .find(|(k, _)| *k == kind)
            .map_or_else(String::new, |(_, n)| n.clone())
    }

    fn usage_line(&self) -> String {
        format!(
            "Query it without loading it into context: {digest}(handle, query) for \
             natural-language exact stats; {aggregate}(handle, where, agg, field) for filtered \
             exact aggregation with row provenance; {rows}(handle, start, limit) to fetch \
             specific JSON rows verbatim (feed it an index from aggregate's provenance); \
             {search}(handle, pattern) to find lines; \
             {lines}(handle, start, end) to page lines; {retrieve}(handle, start, len) for raw \
             bytes; {describe}(handle) for schema and field stats.",
            digest = self.name_of(ToolKind::Digest),
            aggregate = self.name_of(ToolKind::Aggregate),
            rows = self.name_of(ToolKind::Rows),
            search = self.name_of(ToolKind::Search),
            lines = self.name_of(ToolKind::Lines),
            retrieve = self.name_of(ToolKind::Retrieve),
            describe = self.name_of(ToolKind::Describe),
        )
    }
}

/// Run the relay until the client closes its input (then the downstream stdin is closed
/// so the child can exit) and the downstream closes its output.
///
/// Generic over the four byte streams so tests can drive it with in-memory duplex pipes
/// while the binary passes stdin/stdout and the child's pipes.
///
/// # Errors
///
/// Returns any I/O error from the client-side streams; downstream write failures end the
/// relay without error (the child exiting first is a normal shutdown order).
pub async fn run_relay<CI, CO, DI, DO>(
    client_in: CI,
    client_out: CO,
    down_in: DI,
    down_out: DO,
    store: HandleStore,
    cfg: WrapConfig,
) -> std::io::Result<()>
where
    CI: AsyncRead + Unpin + Send + 'static,
    CO: AsyncWrite + Unpin + Send + 'static,
    DI: AsyncWrite + Unpin + Send + 'static,
    DO: AsyncRead + Unpin + Send + 'static,
{
    let state = Arc::new(RelayState::new(store, cfg));
    let client_out = Arc::new(tokio::sync::Mutex::new(client_out));

    let pump = tokio::spawn(pump_downstream(
        down_out,
        Arc::clone(&client_out),
        Arc::clone(&state),
    ));

    let mut down_writer = down_in;
    let mut lines = BufReader::new(client_in).lines();
    while let Some(line) = lines.next_line().await? {
        if handle_client_line(&line, &mut down_writer, &client_out, &state)
            .await
            .is_err()
        {
            break;
        }
    }
    // Client EOF (or downstream write failure): close the child's stdin so it can exit.
    let _ = down_writer.shutdown().await;
    drop(down_writer);
    let _ = pump.await;
    Ok(())
}

async fn handle_client_line<DW, CW>(
    line: &str,
    down: &mut DW,
    client_out: &Arc<tokio::sync::Mutex<CW>>,
    state: &Arc<RelayState>,
) -> std::io::Result<()>
where
    DW: AsyncWrite + Unpin,
    CW: AsyncWrite + Unpin,
{
    let Ok(msg) = serde_json::from_str::<Value>(line) else {
        // Not JSON we understand — forward verbatim, fail-open.
        return write_line(down, line).await;
    };
    let method = msg.get("method").and_then(Value::as_str);
    let id = msg.get("id").filter(|v| !v.is_null());
    match (method, id) {
        (Some("tools/call"), Some(id)) => {
            let name = msg
                .pointer("/params/name")
                .and_then(Value::as_str)
                .unwrap_or("");
            if let Some(kind) = state.kind_for(name) {
                let args = msg
                    .pointer("/params/arguments")
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                let result = handle_injected(kind, &args, state);
                let resp = json!({"jsonrpc": "2.0", "id": id, "result": result});
                return write_locked_line(client_out, &resp.to_string()).await;
            }
            insert_pending(state, id, Pending::ToolCall);
            write_line(down, line).await
        }
        (Some("tools/list"), Some(id)) => {
            insert_pending(state, id, Pending::ToolsList);
            write_line(down, line).await
        }
        _ => write_line(down, line).await,
    }
}

fn insert_pending(state: &RelayState, id: &Value, pending: Pending) {
    state
        .pending
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(id_key(id), pending);
}

async fn pump_downstream<R, W>(
    down_out: R,
    client_out: Arc<tokio::sync::Mutex<W>>,
    state: Arc<RelayState>,
) where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let mut lines = BufReader::new(down_out).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        let out = process_downstream_line(&line, &state);
        if write_locked_line(&client_out, &out).await.is_err() {
            break;
        }
    }
}

/// Intercept a downstream stdout line: rewrite pending `tools/list` / `tools/call`
/// responses, forward everything else (notifications, server-initiated requests,
/// unmatched responses, non-JSON) verbatim.
fn process_downstream_line(line: &str, state: &RelayState) -> String {
    let Ok(mut msg) = serde_json::from_str::<Value>(line) else {
        return line.to_string();
    };
    let is_response = msg.get("method").is_none() && msg.get("id").is_some();
    if !is_response {
        return line.to_string();
    }
    let key = id_key(msg.get("id").unwrap_or(&Value::Null));
    let pending = state
        .pending
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .remove(&key);
    match pending {
        Some(Pending::ToolsList) => {
            inject_tools(&mut msg, state);
            msg.to_string()
        }
        Some(Pending::ToolCall) => {
            offload_large_content(&mut msg, state);
            msg.to_string()
        }
        None => line.to_string(),
    }
}

/// A JSON-RPC id (number or string) as a stable map key.
fn id_key(id: &Value) -> String {
    id.to_string()
}

async fn write_line<W: AsyncWrite + Unpin>(w: &mut W, line: &str) -> std::io::Result<()> {
    w.write_all(line.as_bytes()).await?;
    w.write_all(b"\n").await?;
    w.flush().await
}

async fn write_locked_line<W: AsyncWrite + Unpin>(
    w: &Arc<tokio::sync::Mutex<W>>,
    line: &str,
) -> std::io::Result<()> {
    let mut w = w.lock().await;
    write_line(&mut *w, line).await
}

// ---------------------------------------------------------------------------
// tools/list injection
// ---------------------------------------------------------------------------

fn inject_tools(msg: &mut Value, state: &RelayState) {
    let Some(result) = msg.get_mut("result") else {
        return; // error response — forward as-is
    };
    if result.get("nextCursor").and_then(Value::as_str).is_some() {
        return; // not the final page; inject once, at the end
    }
    let Some(tools) = result.get_mut("tools").and_then(Value::as_array_mut) else {
        return;
    };
    let downstream: Vec<String> = tools
        .iter()
        .filter_map(|t| t.get("name").and_then(Value::as_str))
        .map(str::to_string)
        .collect();
    let mut advertised = Vec::with_capacity(BASE_TOOLS.len());
    for (kind, base) in BASE_TOOLS {
        let default = format!("{}{base}", state.cfg.prefix);
        let name = if downstream.contains(&default) {
            // Never shadow a downstream tool: rename ours, and if even the renamed
            // form collides, drop the injection rather than break the wrapped server.
            let renamed = format!("wrap_{default}");
            if downstream.contains(&renamed) {
                tracing::warn!(tool = %default, "skipping injected tool: both names collide with downstream");
                continue;
            }
            renamed
        } else {
            default
        };
        tools.push(tool_def(kind, &name));
        advertised.push((kind, name));
    }
    *state.names.lock().unwrap_or_else(PoisonError::into_inner) = advertised;
}

fn tool_def(kind: ToolKind, name: &str) -> Value {
    let handle_prop = json!({
        "type": "string",
        "description": "The 64-hex SHA-256 handle from a coffer-wrap fact card."
    });
    match kind {
        ToolKind::Describe => json!({
            "name": name,
            "description": "Schema and per-field stats for an offloaded tool result (exact, computed over ALL rows, never a sample).",
            "inputSchema": {
                "type": "object",
                "properties": {"handle": handle_prop},
                "required": ["handle"]
            }
        }),
        ToolKind::Digest => json!({
            "name": name,
            "description": "Ask a natural-language stats question (count/sum/mean/median/percentile/max/group-by) over an offloaded result. Answers are computed exactly over ALL bytes; refuses rather than guesses.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "handle": handle_prop,
                    "query": {"type": "string", "description": "e.g. \"how many rows have status Error\", \"max restarts\""}
                },
                "required": ["handle", "query"]
            }
        }),
        ToolKind::Aggregate => json!({
            "name": name,
            "description": "Exact filtered aggregation over an offloaded JSON-array result, with matched row indices as provenance. Refuses on mixed-type fields rather than guessing.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "handle": handle_prop,
                    "where": {
                        "type": "array",
                        "description": "Conjunctive predicates over object fields.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "field": {"type": "string"},
                                "op": {"type": "string", "enum": ["eq", "ne", "gt", "ge", "lt", "le"]},
                                "value": {"description": "JSON string or number to compare against."}
                            },
                            "required": ["field", "op", "value"]
                        }
                    },
                    "agg": {"type": "string", "enum": ["count", "sum", "mean", "min", "max"]},
                    "field": {"type": "string", "description": "Numeric field for sum/mean/min/max."}
                },
                "required": ["handle", "agg"]
            }
        }),
        ToolKind::Rows => json!({
            "name": name,
            "description": "Fetch specific rows of an offloaded JSON-array result VERBATIM (byte-exact copies). Use an index from aggregate's provenance, or page with start/limit.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "handle": handle_prop,
                    "start": {"type": "integer", "description": "0-based row index to start at."},
                    "limit": {"type": "integer", "description": "Rows to return (default 10, max 50)."}
                },
                "required": ["handle", "start"]
            }
        }),
        ToolKind::Search => json!({
            "name": name,
            "description": "Case-insensitive substring search over the lines of an offloaded result; returns matching line numbers and text.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "handle": handle_prop,
                    "pattern": {"type": "string"},
                    "max_matches": {"type": "integer", "description": "Cap on returned matches (default 50, max 500)."}
                },
                "required": ["handle", "pattern"]
            }
        }),
        ToolKind::Lines => json!({
            "name": name,
            "description": "Return a 1-based inclusive line range of an offloaded result (bounded page).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "handle": handle_prop,
                    "start": {"type": "integer"},
                    "end": {"type": "integer"}
                },
                "required": ["handle", "start", "end"]
            }
        }),
        ToolKind::Retrieve => json!({
            "name": name,
            "description": "Return a raw byte window of the offloaded original (SHA-256 verified on read). Bounded; use start/len to page.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "handle": handle_prop,
                    "start": {"type": "integer", "description": "Byte offset (default 0)."},
                    "len": {"type": "integer", "description": "Window size in bytes (default 65536, capped)."}
                },
                "required": ["handle"]
            }
        }),
    }
}

// ---------------------------------------------------------------------------
// tools/call offload
// ---------------------------------------------------------------------------

fn offload_large_content(msg: &mut Value, state: &RelayState) {
    let usage = state.usage_line();
    let threshold = state.cfg.threshold_tokens;
    let Some(result) = msg.get_mut("result") else {
        return; // error response — forward as-is
    };
    if result.get("isError").and_then(Value::as_bool) == Some(true) {
        return; // a large error must stay visible, not vanish behind a handle
    }
    let Some(content) = result.get_mut("content").and_then(Value::as_array_mut) else {
        return;
    };
    for block in content.iter_mut() {
        if block.get("type").and_then(Value::as_str) != Some("text") {
            continue;
        }
        let Some(text) = block.get("text").and_then(Value::as_str) else {
            continue;
        };
        let tokens = HeuristicCounter.count(text);
        if tokens <= threshold {
            continue;
        }
        let hash = state.store.put(text.as_bytes());
        tracing::info!(
            bytes = text.len(),
            est_tokens = tokens,
            handle = %hash.short(),
            "offloaded oversized tool result block"
        );
        // Parsing for the fact card doubles as pre-warming the dataset cache: the first
        // follow-up query against this handle skips its parse entirely.
        let described = state
            .datasets
            .get_or_parse(hash.as_str(), text.as_bytes())
            .and_then(|ds| ds.describe());
        let card = fact_card(text, tokens, &hash, described, &state.cfg, &usage);
        if let Some(obj) = block.as_object_mut() {
            obj.insert("text".to_string(), Value::String(card));
        }
    }
}

fn fact_card(
    text: &str,
    est_tokens: usize,
    hash: &ContentHash,
    described: Option<String>,
    cfg: &WrapConfig,
    usage: &str,
) -> String {
    let bytes = text.as_bytes();
    let mut card = format!(
        "[coffer-wrap] tool result offloaded: {} bytes \u{2248} {est_tokens} tokens (> threshold {}). \
         Original preserved byte-exact; SHA-256 verified on read.\nhandle: {}\n",
        bytes.len(),
        cfg.threshold_tokens,
        hash.as_str()
    );
    if let Some(described) = described {
        card.push_str(&clip_chars(&described, 1200));
        card.push('\n');
    } else {
        card.push_str(&shape_line(bytes));
    }
    card.push_str("preview:\n");
    card.push_str(&clip_chars(text, cfg.preview_chars));
    card.push('\n');
    card.push_str(usage);
    card
}

fn shape_line(bytes: &[u8]) -> String {
    if let Ok(v) = serde_json::from_slice::<Value>(bytes) {
        match v {
            Value::Array(items) => {
                let fields = items
                    .first()
                    .and_then(Value::as_object)
                    .map(|o| o.keys().take(12).cloned().collect::<Vec<_>>().join(", "))
                    .unwrap_or_default();
                if fields.is_empty() {
                    format!("shape: JSON array, {} elements\n", items.len())
                } else {
                    format!(
                        "shape: JSON array, {} rows; fields: {fields}\n",
                        items.len()
                    )
                }
            }
            Value::Object(o) => format!("shape: JSON object, {} top-level keys\n", o.len()),
            _ => "shape: JSON scalar\n".to_string(),
        }
    } else {
        format!(
            "shape: text, {} lines\n",
            bytes.split(|b| *b == b'\n').count()
        )
    }
}

fn clip_chars(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let clipped: String = s.chars().take(max_chars).collect();
    format!("{clipped}\u{2026}")
}

// ---------------------------------------------------------------------------
// injected tool handlers
// ---------------------------------------------------------------------------

fn ok_result(text: String) -> Value {
    json!({"content": [{"type": "text", "text": text}], "isError": false})
}

fn error_result(msg: &str) -> Value {
    json!({"content": [{"type": "text", "text": format!("[coffer-wrap] error: {msg}")}], "isError": true})
}

fn handle_injected(kind: ToolKind, args: &Value, state: &RelayState) -> Value {
    let Some(handle) = args.get("handle").and_then(Value::as_str) else {
        return error_result("missing required argument: handle");
    };
    let bytes = match resolve_bytes(handle, state) {
        Ok(b) => b,
        Err(e) => return error_result(&e),
    };
    match kind {
        ToolKind::Describe => handle_describe(state, handle, &bytes),
        ToolKind::Digest => handle_digest(state, handle, &bytes, args),
        ToolKind::Aggregate => handle_aggregate(state, handle, &bytes, args),
        ToolKind::Rows => handle_rows(state, handle, &bytes, args),
        ToolKind::Search => handle_search(state, handle, &bytes, args),
        ToolKind::Lines => handle_lines(&bytes, args),
        ToolKind::Retrieve => handle_retrieve(&bytes, args, &state.cfg),
    }
}

fn resolve_bytes(handle: &str, state: &RelayState) -> Result<Arc<[u8]>, String> {
    let hash = ContentHash::from_hex(handle)
        .ok_or_else(|| format!("invalid handle (expected 64-hex SHA-256): {handle}"))?;
    state
        .store
        .get(&hash)
        .ok_or_else(|| format!("unknown handle (not in this store, or evicted): {handle}"))
}

fn handle_describe(state: &RelayState, handle: &str, bytes: &[u8]) -> Value {
    match state
        .datasets
        .get_or_parse(handle, bytes)
        .and_then(|ds| ds.describe())
    {
        Some(card) => ok_result(card),
        None => ok_result(format!(
            "not a JSON array of objects; {}no field stats available. {}",
            shape_line(bytes),
            "Use search/lines/retrieve to inspect it."
        )),
    }
}

fn handle_digest(state: &RelayState, handle: &str, bytes: &[u8], args: &Value) -> Value {
    let Some(query) = args.get("query").and_then(Value::as_str) else {
        return error_result("missing required argument: query");
    };
    match state
        .datasets
        .get_or_parse(handle, bytes)
        .and_then(|ds| ds.digest(query))
    {
        Some(answer) => ok_result(answer),
        None => ok_result(format!(
            "digest refused: no exact answer for this query over this data \
             (exact-or-nothing, never a guess). Try {}(handle) to see the fields, or \
             {}(handle, where, agg, field) for a typed query.",
            state.name_of(ToolKind::Describe),
            state.name_of(ToolKind::Aggregate),
        )),
    }
}

fn parse_op(op: &str) -> Option<Op> {
    match op {
        "eq" => Some(Op::Eq),
        "ne" => Some(Op::Ne),
        "gt" => Some(Op::Gt),
        "ge" => Some(Op::Ge),
        "lt" => Some(Op::Lt),
        "le" => Some(Op::Le),
        _ => None,
    }
}

fn parse_predicates(args: &Value) -> Result<Vec<Predicate>, String> {
    let Some(where_) = args.get("where") else {
        return Ok(Vec::new());
    };
    let Some(items) = where_.as_array() else {
        return Err("`where` must be an array of {field, op, value}".to_string());
    };
    let mut predicates = Vec::with_capacity(items.len());
    for item in items {
        let field = item
            .get("field")
            .and_then(Value::as_str)
            .ok_or("each predicate needs a string `field`")?;
        let op_str = item
            .get("op")
            .and_then(Value::as_str)
            .ok_or("each predicate needs an `op`")?;
        // Refuse unknown operators outright — a typo like "gte" must never silently
        // become an equality filter.
        let op = parse_op(op_str)
            .ok_or_else(|| format!("unknown op `{op_str}` (expected eq|ne|gt|ge|lt|le)"))?;
        let value = item
            .get("value")
            .cloned()
            .ok_or("each predicate needs a `value`")?;
        predicates.push(Predicate {
            field: field.to_string(),
            op,
            value,
        });
    }
    Ok(predicates)
}

fn parse_agg(args: &Value) -> Result<Agg, String> {
    let agg = args
        .get("agg")
        .and_then(Value::as_str)
        .ok_or("missing required argument: agg (count|sum|mean|min|max)")?;
    if agg == "count" {
        return Ok(Agg::Count);
    }
    let field = args
        .get("field")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("agg `{agg}` needs a numeric `field`"))?
        .to_string();
    match agg {
        "sum" => Ok(Agg::Sum(field)),
        "mean" => Ok(Agg::Mean(field)),
        "min" => Ok(Agg::Min(field)),
        "max" => Ok(Agg::Max(field)),
        other => Err(format!(
            "unknown agg `{other}` (expected count|sum|mean|min|max)"
        )),
    }
}

fn handle_aggregate(state: &RelayState, handle: &str, bytes: &[u8], args: &Value) -> Value {
    let predicates = match parse_predicates(args) {
        Ok(p) => p,
        Err(e) => return error_result(&e),
    };
    let agg = match parse_agg(args) {
        Ok(a) => a,
        Err(e) => return error_result(&e),
    };
    match state
        .datasets
        .get_or_parse(handle, bytes)
        .and_then(|ds| ds.query_aggregate(&predicates, &agg))
    {
        Some(qr) => {
            let shown = qr
                .matched
                .iter()
                .take(100)
                .map(usize::to_string)
                .collect::<Vec<_>>();
            let ellipsis = if qr.matched.len() > shown.len() {
                ", \u{2026}"
            } else {
                ""
            };
            ok_result(format!(
                "{}\nprovenance: {} matched row(s), indices [{}{}]",
                qr.display,
                qr.matched.len(),
                shown.join(", "),
                ellipsis
            ))
        }
        None => ok_result(
            "aggregation refused: input is not a JSON array of objects, or the aggregated \
             field is missing / non-numeric / mixed-type over the matched rows \
             (exact-or-nothing, never a guess)."
                .to_string(),
        ),
    }
}

fn handle_rows(state: &RelayState, handle: &str, bytes: &[u8], args: &Value) -> Value {
    let Some(start) = args.get("start").and_then(Value::as_u64) else {
        return error_result("missing required argument: start (0-based row index)");
    };
    let limit = args
        .get("limit")
        .and_then(Value::as_u64)
        .map_or(10, |v| (v as usize).clamp(1, 50));
    // Row count via the cached dataset (parse-once); the bytes themselves are copied
    // verbatim from the original by pick_rows, so what comes back is byte-auditable.
    let Some(ds) = state.datasets.get_or_parse(handle, bytes) else {
        return error_result("not a JSON array of records — use lines/retrieve for text");
    };
    let total = ds.len();
    let start = start as usize;
    if start >= total {
        return error_result(&format!(
            "start {start} is past the last row ({total} rows)"
        ));
    }
    let end = (start + limit).min(total);
    let indices: Vec<usize> = (start..end).collect();
    match pick_rows(bytes, &indices) {
        Some(rows_bytes) => ok_result(format!(
            "rows {start}..{} of {total} (verbatim):\n{}",
            end - 1,
            String::from_utf8_lossy(&rows_bytes)
        )),
        None => error_result("row extraction refused (scanner/parser disagreement)"),
    }
}

fn handle_search(state: &RelayState, handle: &str, bytes: &[u8], args: &Value) -> Value {
    let Some(pattern) = args.get("pattern").and_then(Value::as_str) else {
        return error_result("missing required argument: pattern");
    };
    let max_matches = args
        .get("max_matches")
        .and_then(Value::as_u64)
        .map_or(50, |v| (v as usize).clamp(1, 500));
    let needle = pattern.to_lowercase();

    // JSON arrays are usually ONE physical line, which makes a line search a trap (it
    // matches "line 1" and shows the head of the file). Search rows instead: return the
    // matching row indices — which feed straight into rows(handle, start) — and the rows.
    if let Some(ds) = state.datasets.get_or_parse(handle, bytes) {
        let mut shown = Vec::new();
        let mut total = 0usize;
        for (i, row) in ds.rows().iter().enumerate() {
            let text = row.to_string();
            if text.to_lowercase().contains(&needle) {
                total += 1;
                if shown.len() < max_matches {
                    shown.push(format!("row {i}: {}", clip_chars(&text, 240)));
                }
            }
        }
        return ok_result(format!(
            "{total} matching row(s) for {pattern:?}; showing {}:\n{}\n(fetch any row verbatim with {}(handle, start=<row index>))",
            shown.len(),
            shown.join("\n"),
            state.name_of(ToolKind::Rows),
        ));
    }

    let text = String::from_utf8_lossy(bytes);
    let mut shown = Vec::new();
    let mut total = 0usize;
    for (i, line) in text.lines().enumerate() {
        if line.to_lowercase().contains(&needle) {
            total += 1;
            if shown.len() < max_matches {
                shown.push(format!("{}: {}", i + 1, clip_chars(line, 240)));
            }
        }
    }
    ok_result(format!(
        "{total} matching line(s) for {pattern:?}; showing {}:\n{}",
        shown.len(),
        shown.join("\n")
    ))
}

fn handle_lines(bytes: &[u8], args: &Value) -> Value {
    let (Some(start), Some(end)) = (
        args.get("start").and_then(Value::as_u64),
        args.get("end").and_then(Value::as_u64),
    ) else {
        return error_result("missing required arguments: start, end (1-based, inclusive)");
    };
    if start == 0 || end < start {
        return error_result("start must be >= 1 and end >= start");
    }
    const MAX_SPAN: u64 = 500;
    let clamped_end = end.min(start + MAX_SPAN - 1);
    let text = String::from_utf8_lossy(bytes);
    let slice: Vec<&str> = text
        .lines()
        .skip(start as usize - 1)
        .take((clamped_end - start + 1) as usize)
        .collect();
    if slice.is_empty() {
        return error_result(&format!("start {start} is past the last line"));
    }
    let note = if clamped_end < end {
        format!("\n[coffer-wrap] span clamped to {MAX_SPAN} lines (requested {start}..{end})")
    } else {
        String::new()
    };
    ok_result(format!(
        "lines {start}..{} of the offloaded result:\n{}{note}",
        start + slice.len() as u64 - 1,
        slice.join("\n")
    ))
}

fn handle_retrieve(bytes: &[u8], args: &Value, cfg: &WrapConfig) -> Value {
    let start = args.get("start").and_then(Value::as_u64).unwrap_or(0) as usize;
    let len = args
        .get("len")
        .and_then(Value::as_u64)
        .map_or(cfg.retrieve_default_bytes, |v| v as usize)
        .min(cfg.retrieve_max_bytes);
    if start >= bytes.len() {
        return error_result(&format!(
            "start {start} is past the end ({} bytes total)",
            bytes.len()
        ));
    }
    let end = (start + len).min(bytes.len());
    let window = &bytes[start..end];
    let body = String::from_utf8_lossy(window);
    if start == 0 && end == bytes.len() {
        ok_result(body.into_owned())
    } else {
        ok_result(format!(
            "[coffer-wrap] bytes {start}..{end} of {} total \
             (page with start/len; max window {} bytes)\n{body}",
            bytes.len(),
            cfg.retrieve_max_bytes
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> RelayState {
        RelayState::new(HandleStore::Memory(MemoryCas::new()), WrapConfig::default())
    }

    #[test]
    fn unknown_op_is_refused_not_eq() {
        let args = json!({"where": [{"field": "s", "op": "gte", "value": 1}], "agg": "count"});
        let err = parse_predicates(&args).unwrap_err();
        assert!(err.contains("gte"), "error should name the bad op: {err}");
    }

    #[test]
    fn id_key_distinguishes_number_and_string() {
        assert_ne!(id_key(&json!(1)), id_key(&json!("1")));
    }

    #[test]
    fn small_results_pass_through_untouched() {
        let st = state();
        let line = r#"{"jsonrpc":"2.0","id":7,"result":{"content":[{"type":"text","text":"ok"}],"isError":false}}"#;
        insert_pending(&st, &json!(7), Pending::ToolCall);
        let out = process_downstream_line(line, &st);
        let msg: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(msg.pointer("/result/content/0/text").unwrap(), "ok");
    }

    #[test]
    fn error_results_are_never_offloaded() {
        let st = state();
        let big = "x".repeat(200_000);
        let line = json!({"jsonrpc": "2.0", "id": 8, "result": {
            "content": [{"type": "text", "text": big}], "isError": true
        }})
        .to_string();
        insert_pending(&st, &json!(8), Pending::ToolCall);
        let out = process_downstream_line(&line, &st);
        assert!(
            out.contains(&"x".repeat(1000)),
            "large error text must stay visible"
        );
    }

    #[test]
    fn collision_renames_never_shadow_downstream() {
        let st = state();
        let mut msg = json!({"jsonrpc": "2.0", "id": 1, "result": {"tools": [
            {"name": "coffer_digest", "description": "downstream's own", "inputSchema": {"type": "object"}}
        ]}});
        inject_tools(&mut msg, &st);
        let names: Vec<&str> = msg
            .pointer("/result/tools")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.get("name").unwrap().as_str().unwrap())
            .collect();
        assert_eq!(
            names.iter().filter(|n| **n == "coffer_digest").count(),
            1,
            "downstream name must appear exactly once: {names:?}"
        );
        assert!(names.contains(&"wrap_coffer_digest"), "{names:?}");
        // Routing follows the rename: the downstream name is forwarded, not intercepted.
        assert!(st.kind_for("coffer_digest").is_none());
        assert_eq!(st.kind_for("wrap_coffer_digest"), Some(ToolKind::Digest));
    }
}
