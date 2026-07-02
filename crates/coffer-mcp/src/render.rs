//! Pure response shaping: summaries, fact cards, byte/line/row windows, JSON-path
//! selection, aggregate parsing, and shared-CAS unfold. Everything here takes bytes in
//! and returns strings/results out — unit-testable without a server.
// Split out of main.rs for maintainability; behavior is unchanged. main.rs glob-imports
// this module, so the original single-file scope (and its test suite) still sees every name.

use coffer_cas::Cas;
use coffer_core::{Agg, ContentType, Op, compress_structural_code_to_budget, detect};
use coffer_tokenizer::{HeuristicCounter, TokenCounter};
use rmcp::model::{CallToolResult, Content};
use serde_json::Value;

use crate::limits::{DEFAULT_STRUCTURAL_CODE_TARGET_TOKENS, RetrieveLimits};

pub(crate) fn summarize(bytes: &[u8]) -> String {
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
pub(crate) fn fact_card(bytes: &[u8]) -> Option<String> {
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

pub(crate) struct ByteWindow<'a> {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) total: usize,
    pub(crate) bytes: &'a [u8],
}

pub(crate) fn byte_window(
    bytes: &[u8],
    start: Option<usize>,
    max_bytes: Option<usize>,
) -> ByteWindow<'_> {
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

pub(crate) fn render_retrieved_bytes(
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

pub(crate) fn render_json_rows(
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

pub(crate) fn render_json_path(bytes: &[u8], path: &str) -> Result<String, String> {
    let value: Value = serde_json::from_slice(bytes).map_err(|e| format!("not valid JSON: {e}"))?;
    let selected = select_json_path(&value, path)?;
    let body =
        serde_json::to_string_pretty(selected).map_err(|e| format!("render JSON path: {e}"))?;
    Ok(format!("json path {path}\n{body}"))
}

pub(crate) fn render_text_lines(
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

pub(crate) fn render_text_search(
    bytes: &[u8],
    pattern: &str,
    limit: Option<usize>,
) -> Result<String, String> {
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

pub(crate) fn select_json_path<'a>(value: &'a Value, path: &str) -> Result<&'a Value, String> {
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

pub(crate) fn parse_op(op: &str) -> Op {
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
pub(crate) fn parse_value_arg(raw: &str) -> Value {
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
}

/// Resolve an aggregate name (+ optional field) to an [`Agg`]. `count` needs no field; the rest require one.
pub(crate) fn parse_agg(agg: &str, field: Option<&str>) -> Option<Agg> {
    match agg {
        "count" => Some(Agg::Count),
        "sum" => field.map(|f| Agg::Sum(f.to_string())),
        "mean" | "avg" => field.map(|f| Agg::Mean(f.to_string())),
        "min" => field.map(|f| Agg::Min(f.to_string())),
        "max" => field.map(|f| Agg::Max(f.to_string())),
        _ => None,
    }
}

/// Format a numeric aggregate value: a whole number prints without a fractional part (so a `count`
/// reads `42`), anything else keeps full precision (so a `mean` reads `12.3456`).
pub(crate) fn fmt_group_value(v: f64) -> String {
    if v.is_finite() && v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{v:.0}")
    } else {
        format!("{v}")
    }
}

/// Render a [`coffer_core::GroupAggregate`] (numeric bucketing, windowed log histogram, or
/// project-join group-by) as model-facing text: the one-line summary, then one line per group with
/// its value and its backing-row count (provenance), capped so a pathological group count cannot
/// flood the context.
pub(crate) fn group_aggregate_text(g: &coffer_core::GroupAggregate) -> String {
    use std::fmt::Write as _;
    const SHOWN: usize = 64;
    let mut s = String::new();
    let _ = writeln!(s, "{}", g.display);
    for b in g.groups.iter().take(SHOWN) {
        let _ = writeln!(
            s,
            "  {} → {} ({} rows)",
            b.key,
            fmt_group_value(b.value),
            b.matched.len()
        );
    }
    if g.groups.len() > SHOWN {
        let _ = writeln!(s, "  … ({} groups total)", g.groups.len());
    }
    s.push_str(
        "Computed over ALL rows including offloaded ones; each group lists its backing-row count.",
    );
    s
}

pub(crate) fn ingested_text(handle: &str, bytes: &[u8]) -> String {
    let card = fact_card(bytes)
        .map(|c| format!("\n{c}"))
        .unwrap_or_default();
    format!(
        "handle: {handle}\nsummary: {}\nbytes: {}{card}\n\nThe full output is held server-side and is NOT in your context. \
         Use coffer_digest(handle, query) for EXACT aggregates over ALL of it (count/sum/mean/median/percentile/\
         group-by/argmax/filter-aggregate), coffer_query(handle, field, op, value) to keep only matching rows, \
         coffer_select(handle, where) to filter by a conjunction and get the matches as a new handle to narrow again, \
         coffer_bucket(handle, field, width) for a numeric-band histogram and coffer_window(handle, pattern, window) for a per-block log histogram, \
         coffer_join(left, right, left_key, right_key, agg) to correlate two handles exactly, or \
         coffer_search(handle, pattern) / coffer_lines(handle, start_line, end_line) for logs/text, \
         coffer_json(handle, path) / coffer_rows(handle, start, limit) / coffer_retrieve(handle, start, max_bytes) for a small window; \
         set full=true only for small payloads you truly need raw, subject to the configured hard cap.",
        summarize(bytes),
        bytes.len()
    )
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum IngestView {
    Summary,
    StructuralCode,
}

pub(crate) fn ingest_view(view: Option<&str>) -> Result<IngestView, String> {
    match view.map(str::trim).filter(|v| !v.is_empty()) {
        None => Ok(IngestView::Summary),
        Some("structural_code") => Ok(IngestView::StructuralCode),
        Some(other) => Err(format!(
            "unknown ingest view {other:?}; supported values: structural_code"
        )),
    }
}

pub(crate) fn ingested_text_with_view(
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

pub(crate) fn unknown_handle() -> CallToolResult {
    CallToolResult::error(vec![Content::text(
        "unknown handle or sentinel hash prefix (ingest/run it first, or use a prefix from a shared-CAS <<cof:...>> sentinel)",
    )])
}
