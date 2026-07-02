//! coffer-proxy: a transparent-compression proxy. This library half is the **pure
//! request transform** — compress the large `tool_result` blocks of an Anthropic Messages request so
//! huge tool output never costs the model context; the binary half ([`main`](../main.rs)) is the HTTP
//! server that applies it in the request path.
//!
//! The transform is **fail-open**: a body that is not the expected JSON shape is returned unchanged,
//! and a request too small to contain a compressible `tool_result` skips JSON parsing entirely.
//! A `tool_result` that does not shrink leaves the whole request byte-for-byte unchanged.
//! Offloaded bytes go into the caller's
//! [`coffer_cas::Cas`], so a retrieve tool sharing that CAS can recover them (proxy compresses ⊕
//! MCP retrieves — the reversible loop). Only `tool_result` content is touched; the system prompt,
//! user, and assistant text are never compressed.

use coffer_cas::Cas;
use coffer_core::{Budget, Compressor};
use coffer_tokenizer::{HeuristicCounter, TokenCounter};
use serde_json::Value;

/// Default fraction of a compressible `tool_result`'s tokens to cut (`COFFER_PROXY_REDUCTION`).
pub const DEFAULT_REDUCTION: f32 = 0.8;

/// Default absolute ceiling on the tokens a rewritten block may keep
/// (`COFFER_PROXY_MAX_KEPT_TOKENS`, heuristic chars/4 count; 0 disables). Proportional
/// reduction alone has an unbounded remainder — measured on the eval dump, an input of
/// ~915k o200k tokens still kept ~183k at the default reduction, defeating the rewrite on
/// exactly the oversized results that motivate it. The ceiling makes the kept size
/// scale-invariant: keep = min(raw × (1 − reduction), ceiling). Note the chars/4 search
/// lands ~33% above its target in real o200k terms on dense JSON, so 4,000 here is ~5.3k
/// o200k tokens kept.
pub const DEFAULT_MAX_KEPT_TOKENS: usize = 4_000;
const MESSAGES_TOOL_RESULT_MARKER: &[u8] = b"tool_result";
const RESPONSES_CALL_OUTPUT_MARKER: &[u8] = b"_call_output";
const OLLAMA_TOOL_MARKER: &[u8] = b"\"tool\"";

/// What the model reads at the top of a rewritten tool-output block, so a `<<cof:…>>`
/// sentinel is never mistaken for truncation, corruption, or "weird log noise". The model
/// only ever learns about coffer in-band, inside tool output — authored system/user/
/// assistant text is never touched — so this line IS the awareness mechanism: it says what
/// the markers mean, that nothing is lost, how to query exactly when coffer tools are
/// registered, and (crucially) not to guess elided content when they are not.
pub const SENTINEL_EXPLAINER: &str = "[coffer] Long runs in this tool result were elided to save context. A marker like \
     <<cof:HASH +N items>> stands for N elided items whose exact bytes are preserved \
     server-side — this is NOT truncation or corruption, and elided content must never be \
     guessed. If coffer_* tools are available, they answer questions over the FULL data \
     exactly (use the marker's HASH as the handle); otherwise rely only on the visible rows \
     and the marker counts.\n";

/// How the tool-output rewrite behaves. `From<usize>` keeps the common "just a
/// min-compress threshold" call shape working, with the explainer on (the default).
#[derive(Clone, Copy, Debug)]
pub struct RewriteOptions {
    /// Bodies (and per-block text) below this many bytes pass through untouched.
    pub min_compress: usize,
    /// Prepend [`SENTINEL_EXPLAINER`] to every rewritten block (`COFFER_PROXY_EXPLAIN`).
    pub explain: bool,
    /// Fraction of a block's tokens to cut (see [`DEFAULT_REDUCTION`]). Non-finite or
    /// out-of-range values fall back to the default.
    pub reduction: f32,
    /// Absolute ceiling on kept tokens per block, heuristic count; 0 disables (see
    /// [`DEFAULT_MAX_KEPT_TOKENS`]).
    pub max_kept_tokens: usize,
}

impl From<usize> for RewriteOptions {
    fn from(min_compress: usize) -> Self {
        Self {
            min_compress,
            explain: true,
            reduction: DEFAULT_REDUCTION,
            max_kept_tokens: DEFAULT_MAX_KEPT_TOKENS,
        }
    }
}

/// Compress every `tool_result` block's text in an Anthropic Messages request `body`, offloading the
/// elided bytes into `cas`. `tool_result` text below `min_compress` bytes is left as-is. Returns the
/// original body unchanged if it is below the compression threshold, is not the expected JSON
/// shape, or no block actually shrinks (fail-open). The render is byte-exact reconstructable from
/// `cas` (the engine's Stage-0 invariant).
/// Why a transform left the body unchanged (or did not), so the proxy can distinguish a degraded
/// fail-open from a benign no-shrink in its metrics (production observability).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransformKind {
    /// At least one tool_result / tool-output block shrank — coffer compressed the body.
    Compressed,
    /// The body WAS the expected shape, but no block shrank (incompressible / below per-block
    /// threshold). Benign — the request is forwarded unchanged.
    NoShrink,
    /// The body was NOT the expected shape (below the overall threshold, missing the marker, not
    /// JSON, or no messages/input array) — fail-open passthrough. A spike on a supported endpoint
    /// signals a request-shape change the transform no longer recognizes.
    Passthrough,
}

#[must_use]
pub fn compress_request_body(
    body: &[u8],
    cas: &dyn Cas,
    opts: impl Into<RewriteOptions>,
) -> Vec<u8> {
    compress_request_body_kind(body, cas, opts).0
}

/// Like [`compress_request_body`], also returning WHY the body was/wasn't changed (see
/// [`TransformKind`]). The returned bytes are identical to [`compress_request_body`].
#[must_use]
pub fn compress_request_body_kind(
    body: &[u8],
    cas: &dyn Cas,
    opts: impl Into<RewriteOptions>,
) -> (Vec<u8>, TransformKind) {
    use TransformKind::{Compressed, NoShrink, Passthrough};
    let opts = opts.into();
    if body.len() < opts.min_compress {
        return (body.to_vec(), Passthrough);
    }
    if !contains_bytes(body, MESSAGES_TOOL_RESULT_MARKER) {
        return (body.to_vec(), Passthrough);
    }
    let Ok(mut v) = serde_json::from_slice::<Value>(body) else {
        return (body.to_vec(), Passthrough);
    };
    let Some(msgs) = v.get_mut("messages").and_then(Value::as_array_mut) else {
        return (body.to_vec(), Passthrough);
    };
    let counter = HeuristicCounter;
    let mut changed = false;
    for m in msgs.iter_mut() {
        let Some(content) = m.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        for block in content.iter_mut() {
            if block.get("type").and_then(Value::as_str) != Some("tool_result") {
                continue;
            }
            match block.get_mut("content") {
                // tool_result content is a bare string …
                Some(Value::String(s)) => changed |= squash(s, &counter, cas, opts),
                // … or an array of content blocks; compress only the text ones.
                Some(Value::Array(arr)) => {
                    for c in arr.iter_mut() {
                        if c.get("type").and_then(Value::as_str) == Some("text") {
                            if let Some(Value::String(t)) = c.get_mut("text") {
                                changed |= squash(t, &counter, cas, opts);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    if !changed {
        return (body.to_vec(), NoShrink);
    }
    match serde_json::to_vec(&v) {
        Ok(out) => (out, Compressed),
        Err(_) => (body.to_vec(), Passthrough), // re-serialization failed → fail open
    }
}

/// Compress the **tool-output** items of an OpenAI **Responses** request `body` (the shape Codex and
/// other Responses-API clients send to `/responses`): for each `input[]` item whose type ends in
/// `_call_output` (`function_call_output`, `local_shell_call_output`, `custom_tool_call_output`),
/// compress its `output` (a string, or an array of content blocks). Authored `message` content
/// (`input_text`/`output_text`) is never touched — the model reads its own instructions/prompts
/// whole, exactly as the Anthropic transform leaves system/user/assistant text. Fail-open: a body
/// that is below the compression threshold, is not the expected shape, or has no shrinking output
/// is returned byte-for-byte unchanged. Offloaded bytes go into `cas` for unfold.
#[must_use]
pub fn compress_responses_body(
    body: &[u8],
    cas: &dyn Cas,
    opts: impl Into<RewriteOptions>,
) -> Vec<u8> {
    compress_responses_body_kind(body, cas, opts).0
}

/// Like [`compress_responses_body`], also returning WHY the body was/wasn't changed (see
/// [`TransformKind`]). The returned bytes are identical to [`compress_responses_body`].
#[must_use]
pub fn compress_responses_body_kind(
    body: &[u8],
    cas: &dyn Cas,
    opts: impl Into<RewriteOptions>,
) -> (Vec<u8>, TransformKind) {
    use TransformKind::{Compressed, NoShrink, Passthrough};
    let opts = opts.into();
    if body.len() < opts.min_compress {
        return (body.to_vec(), Passthrough);
    }
    if !contains_bytes(body, RESPONSES_CALL_OUTPUT_MARKER) {
        return (body.to_vec(), Passthrough);
    }
    let Ok(mut v) = serde_json::from_slice::<Value>(body) else {
        return (body.to_vec(), Passthrough);
    };
    let Some(items) = v.get_mut("input").and_then(Value::as_array_mut) else {
        return (body.to_vec(), Passthrough);
    };
    let counter = HeuristicCounter;
    let mut changed = false;
    for item in items.iter_mut() {
        let is_tool_output = item
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|t| t.ends_with("_call_output"));
        if !is_tool_output {
            continue;
        }
        match item.get_mut("output") {
            Some(Value::String(s)) => changed |= squash(s, &counter, cas, opts),
            Some(Value::Array(arr)) => {
                for c in arr.iter_mut() {
                    if let Some(Value::String(t)) = c.get_mut("text") {
                        changed |= squash(t, &counter, cas, opts);
                    }
                }
            }
            _ => {}
        }
    }
    if !changed {
        return (body.to_vec(), NoShrink);
    }
    match serde_json::to_vec(&v) {
        Ok(out) => (out, Compressed),
        Err(_) => (body.to_vec(), Passthrough), // re-serialization failed → fail open
    }
}

/// Compress the **tool-output** messages of an **Ollama** `/api/chat` request `body`: each
/// `messages[]` entry with `role == "tool"` carries a tool result in its `content` string, which is
/// compressed like an Anthropic `tool_result` / OpenAI `*_call_output`. Authored user/assistant/system
/// messages are never touched. Fail-open and byte-exact recoverable, identical contract to the other
/// transforms. Bedrock and OpenRouter need no new transform — they use the
/// Anthropic Messages / OpenAI Responses shapes already handled, so they fan out by upstream routing.
#[must_use]
pub fn compress_ollama_body(
    body: &[u8],
    cas: &dyn Cas,
    opts: impl Into<RewriteOptions>,
) -> Vec<u8> {
    compress_ollama_body_kind(body, cas, opts).0
}

/// Like [`compress_ollama_body`], also returning WHY the body was/wasn't changed ([`TransformKind`]).
#[must_use]
pub fn compress_ollama_body_kind(
    body: &[u8],
    cas: &dyn Cas,
    opts: impl Into<RewriteOptions>,
) -> (Vec<u8>, TransformKind) {
    use TransformKind::{Compressed, NoShrink, Passthrough};
    let opts = opts.into();
    if body.len() < opts.min_compress {
        return (body.to_vec(), Passthrough);
    }
    if !contains_bytes(body, OLLAMA_TOOL_MARKER) {
        return (body.to_vec(), Passthrough);
    }
    let Ok(mut v) = serde_json::from_slice::<Value>(body) else {
        return (body.to_vec(), Passthrough);
    };
    let Some(msgs) = v.get_mut("messages").and_then(Value::as_array_mut) else {
        return (body.to_vec(), Passthrough);
    };
    let counter = HeuristicCounter;
    let mut changed = false;
    for m in msgs.iter_mut() {
        if m.get("role").and_then(Value::as_str) != Some("tool") {
            continue;
        }
        if let Some(Value::String(s)) = m.get_mut("content") {
            changed |= squash(s, &counter, cas, opts);
        }
    }
    if !changed {
        return (body.to_vec(), NoShrink);
    }
    match serde_json::to_vec(&v) {
        Ok(out) => (out, Compressed),
        Err(_) => (body.to_vec(), Passthrough),
    }
}

fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    let Some((&first, rest)) = needle.split_first() else {
        return false;
    };
    if rest.len() >= haystack.len() {
        return needle.len() == haystack.len() && haystack == needle;
    }
    let mut start = 0;
    while let Some(offset) = haystack[start..].iter().position(|&byte| byte == first) {
        let idx = start + offset;
        let end = idx + needle.len();
        if end <= haystack.len() && &haystack[idx + 1..end] == rest {
            return true;
        }
        start = idx + 1;
    }
    false
}

/// Replace `text` in place with its coffer-compressed render, iff it is large enough and the
/// render actually shrinks (fail-open on small/incompressible content).
fn squash(
    text: &mut String,
    counter: &HeuristicCounter,
    cas: &dyn Cas,
    opts: RewriteOptions,
) -> bool {
    if text.len() < opts.min_compress {
        return false;
    }
    // Hybrid budget: proportional cut with an absolute ceiling, so the kept size cannot
    // scale with the input (a 915k-token block would otherwise still keep ~183k at 0.8).
    let reduction = if opts.reduction.is_finite() && (0.0..1.0).contains(&opts.reduction) {
        opts.reduction
    } else {
        DEFAULT_REDUCTION
    };
    let raw_tokens = counter.count(text);
    let mut target = (raw_tokens as f32 * (1.0 - reduction)) as usize;
    if opts.max_kept_tokens > 0 {
        target = target.min(opts.max_kept_tokens);
    }
    let target = target.max(1);
    let Ok(doc) = Compressor::new()
        .budget(Budget::Tokens(target))
        .counter(counter)
        .min_bytes(0)
        .compress(text.as_bytes(), cas)
    else {
        return false;
    };
    let rendered = doc.render_for_model();
    // The shrink gate counts the explainer too: the rewrite must never GROW the block.
    // A smaller render implies at least one elided run (a verbatim-only render equals the
    // original text), so the explainer never appears without a sentinel to explain.
    let total = if opts.explain {
        SENTINEL_EXPLAINER.len() + rendered.len()
    } else {
        rendered.len()
    };
    if total < text.len() {
        *text = if opts.explain {
            let mut out = String::with_capacity(total);
            out.push_str(SENTINEL_EXPLAINER);
            out.push_str(&rendered);
            out
        } else {
            rendered
        };
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coffer_cas::{MemoryCas, SqliteCas, read_blob};
    use proptest::prelude::*;

    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            use std::sync::atomic::{AtomicU32, Ordering};
            static N: AtomicU32 = AtomicU32::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            let dir =
                std::env::temp_dir().join(format!("coffer-proxy-{}-{tag}-{n}", std::process::id()));
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

    #[test]
    fn explainer_present_by_default_and_absent_when_disabled() {
        let cas = MemoryCas::new();
        let big = big_json_array();

        let on = tool_result_text(&compress_request_body(
            &request_with_tool_result(&big),
            &cas,
            1024,
        ));
        assert!(
            on.starts_with(SENTINEL_EXPLAINER),
            "default rewrite must lead with the in-band explainer"
        );

        let off = tool_result_text(&compress_request_body(
            &request_with_tool_result(&big),
            &cas,
            RewriteOptions {
                min_compress: 1024,
                explain: false,
                reduction: DEFAULT_REDUCTION,
                max_kept_tokens: DEFAULT_MAX_KEPT_TOKENS,
            },
        ));
        assert!(
            !off.contains("[coffer]"),
            "explain=false must omit the explainer"
        );
        assert!(off.contains("<<cof:"), "the render itself is unchanged");
        assert_eq!(
            on.strip_prefix(SENTINEL_EXPLAINER).unwrap(),
            off,
            "the explainer is a pure prefix — the render is identical either way"
        );
    }

    #[test]
    fn absolute_ceiling_bounds_kept_tokens_on_huge_blocks() {
        // Proportional reduction alone keeps 20% of ANY input — unbounded. The ceiling
        // must make the kept size scale-invariant.
        let cas = MemoryCas::new();
        let huge: String = {
            let items: Vec<String> = (0..4000)
                .map(|i| format!(r#"{{"id":{i},"sub":"drivers"}}"#))
                .collect();
            format!("[{}]", items.join(","))
        };
        let counter = HeuristicCounter;
        let raw_tokens = counter.count(&huge);
        assert!(raw_tokens > 20_000, "fixture must dwarf the ceiling");

        let out = compress_request_body(
            &request_with_tool_result(&huge),
            &cas,
            RewriteOptions {
                min_compress: 1024,
                explain: false,
                reduction: DEFAULT_REDUCTION,
                max_kept_tokens: 500,
            },
        );
        let kept = counter.count(&tool_result_text(&out));
        assert!(
            kept <= 500,
            "ceiling must bound the kept size (kept {kept} heuristic tokens)"
        );
    }

    #[test]
    fn proportional_cut_governs_when_under_the_ceiling() {
        // Mid-size input: raw*0.2 is far below the default ceiling, so the proportional
        // target applies — today's behavior for modest blocks is unchanged.
        let cas = MemoryCas::new();
        let big = big_json_array();
        let counter = HeuristicCounter;
        let raw_tokens = counter.count(&big);
        assert!(raw_tokens / 5 < DEFAULT_MAX_KEPT_TOKENS);

        let out = compress_request_body(&request_with_tool_result(&big), &cas, 1024);
        let rendered = tool_result_text(&out);
        let rendered = strip_explainer(&rendered);
        let kept = counter.count(rendered);
        assert!(
            kept <= raw_tokens / 4 && kept > 0,
            "expected ~20% of {raw_tokens}, kept {kept}"
        );
    }

    #[test]
    fn reduction_knob_changes_kept_size() {
        let big = big_json_array();
        let counter = HeuristicCounter;
        let kept_at = |reduction: f32| {
            let cas = MemoryCas::new();
            let out = compress_request_body(
                &request_with_tool_result(&big),
                &cas,
                RewriteOptions {
                    min_compress: 1024,
                    explain: false,
                    reduction,
                    max_kept_tokens: 0, // ceiling off: isolate the knob under test
                },
            );
            counter.count(&tool_result_text(&out))
        };
        assert!(
            kept_at(0.5) > kept_at(0.95),
            "a gentler reduction must keep more"
        );
    }

    fn request_with_tool_result(tool_text: &str) -> Vec<u8> {
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

    fn verbose_git_status() -> String {
        let mut s = String::from(
            "On branch main\n\
             Changes not staged for commit:\n\
               (use \"git add <file>...\" to update what will be committed)\n\
               (use \"git restore <file>...\" to discard changes in working directory)\n",
        );
        for i in 0..80 {
            s.push_str(&format!("\tmodified:   crates/example_{i}/src/lib.rs\n"));
        }
        s.push_str(
            "\n\
             Untracked files:\n\
               (use \"git add <file>...\" to include in what will be committed)\n",
        );
        for i in 0..30 {
            s.push_str(&format!("\tdocs/generated-{i}.md\n"));
        }
        s.push_str("\nno changes added to commit (use \"git add\" and/or \"git commit -a\")\n");
        s
    }

    fn tool_result_text(body: &[u8]) -> String {
        let v: Value = serde_json::from_slice(body).unwrap();
        v["messages"][2]["content"][0]["content"][0]["text"]
            .as_str()
            .unwrap()
            .to_string()
    }

    /// Rewritten blocks carry the in-band explainer; the reversible-splice checks below
    /// operate on the render that follows it (stripping doubles as the presence assert).
    fn strip_explainer(rendered: &str) -> &str {
        rendered
            .strip_prefix(SENTINEL_EXPLAINER)
            .expect("rewritten block must start with the sentinel explainer")
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

    #[test]
    fn reserialization_compacts_authored_whitespace() {
        // Contract evidence: when a tool_result compresses, the proxy re-serializes the WHOLE
        // body via serde_json::to_vec -> COMPACT output. preserve_order + arbitrary_precision keep key
        // order and numeric precision, but a pretty-printed authored envelope is normalized to compact,
        // so the transform is NOT byte-for-byte for non-compact senders (content is preserved).
        let req = serde_json::json!({
            "model": "claude-opus-4-8",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1",
                     "content": [{"type": "text", "text": big_json_array()}]}
                ]}
            ]
        });
        let pretty = serde_json::to_vec_pretty(&req).unwrap();
        assert!(
            pretty.windows(3).any(|w| w == b"\n  "),
            "input is pretty-printed"
        );
        let cas = MemoryCas::new();
        let out = compress_request_body(&pretty, &cas, 1024);
        assert!(out.len() < pretty.len(), "tool_result compressed");
        assert!(
            !out.windows(3).any(|w| w == b"\n  "),
            "output is compact — authored whitespace normalized by re-serialization"
        );
        // content + key order still survive (preserve_order / arbitrary_precision).
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["model"], "claude-opus-4-8");
        assert_eq!(v["max_tokens"], 1024);
    }

    #[test]
    fn marker_scan_handles_boundaries() {
        assert!(contains_bytes(b"abc tool_result xyz", b"tool_result"));
        assert!(contains_bytes(b"_call_output", b"_call_output"));
        assert!(!contains_bytes(b"tool_resul", b"tool_result"));
        assert!(!contains_bytes(b"", b"tool_result"));
        assert!(!contains_bytes(b"anything", b""));
    }

    #[test]
    fn compresses_tool_result_and_preserves_structure() {
        let big = big_json_array();
        let body = request_with_tool_result(&big);
        let cas = MemoryCas::new();
        let out = compress_request_body(&body, &cas, 1024);

        assert!(out.len() < body.len(), "request shrinks");
        let v: Value = serde_json::from_slice(&out).unwrap();
        // structure preserved
        assert_eq!(v["model"], "claude-opus-4-8");
        assert_eq!(v["messages"].as_array().unwrap().len(), 3);
        assert_eq!(v["messages"][1]["content"][1]["id"], "toolu_1");
        // tool_result text was compressed and offloaded bytes are retained for retrieve
        let t = tool_result_text(&out);
        assert!(t.contains("<<cof:"), "tool_result carries a sentinel: {t}");
        assert!(
            !cas.is_empty(),
            "offloaded bytes retained in the shared CAS for retrieval"
        );
        // head AND tail survive
        assert!(t.contains(r#"{"id":0,"#), "head survives");
        assert!(t.contains(r#"{"id":399,"#), "tail survives");
    }

    #[test]
    fn passthrough_prefix_byte_identical_under_compression_h5_cache_parity() {
        // H5 cache parity, verified at the byte level (the decisive invariant — independent of any
        // live caching backend): coffer rewrites ONLY the tool_result text, so the authored
        // passthrough prefix (model, max_tokens, messages[0] user, messages[1] assistant, and
        // messages[2] up to the tool_result content) is byte-for-byte identical to a plain
        // serialization. A provider's prompt cache keyed on that prefix therefore hits identically
        // whether the tool_result is compressed or not — cache parity holds by construction.
        let big = big_json_array();
        let body = request_with_tool_result(&big);
        let cas = MemoryCas::new();

        // Baseline: a plain serde round-trip (no compression) — the same deterministic serializer.
        let uncompressed =
            serde_json::to_vec(&serde_json::from_slice::<Value>(&body).unwrap()).unwrap();
        let compressed = compress_request_body(&body, &cas, 1024);
        assert!(
            compressed.len() < uncompressed.len(),
            "tool_result compressed"
        );

        // Longest common prefix of the compressed vs uncompressed serializations.
        let lcp = uncompressed
            .iter()
            .zip(&compressed)
            .take_while(|(a, b)| a == b)
            .count();
        // The first divergence MUST fall inside the tool_result block — i.e. after its marker.
        let tr_pos = String::from_utf8_lossy(&uncompressed)
            .find("tool_result")
            .expect("request has a tool_result");
        assert!(
            lcp > tr_pos,
            "passthrough prefix mutated: first divergence at byte {lcp} precedes the tool_result marker at {tr_pos}"
        );
        // And that prefix is substantial (it contains the authored user+assistant turns).
        assert!(
            tr_pos > 120,
            "prefix should include the authored messages before the tool_result"
        );
        // The bytes before the tool_result content are identical in both serializations.
        assert_eq!(
            &uncompressed[..tr_pos],
            &compressed[..tr_pos],
            "every authored byte before the tool_result is preserved (0 passthrough-prefix mutation)"
        );
    }

    #[test]
    fn proxy_sentinel_recovers_exact_original_from_shared_sqlite_cas() {
        let dir = TempDir::new("sqlite-unfold");
        let big = big_json_array();
        let body = request_with_tool_result(&big);
        let cas = SqliteCas::open(dir.db()).unwrap();

        let out = compress_request_body(&body, &cas, 1024);
        cas.flush();

        let rendered = tool_result_text(&out);
        let rendered = strip_explainer(&rendered).to_string();
        let span = sentinel_span(&rendered);
        let hash = sentinel_hash(&rendered[span.clone()]);
        let recovered = read_blob(dir.db(), hash).unwrap().unwrap();
        let mut reconstructed = String::new();
        reconstructed.push_str(&rendered[..span.start]);
        reconstructed.push_str(std::str::from_utf8(&recovered).unwrap());
        reconstructed.push_str(&rendered[span.end..]);

        assert_eq!(
            reconstructed, big,
            "compact render plus a fresh shared-CAS read by the sentinel hash must reconstruct the exact tool_result bytes"
        );
    }

    #[test]
    fn compresses_verbose_git_status_tool_result_and_recovers_it() {
        let dir = TempDir::new("git-status");
        let status = verbose_git_status();
        let body = request_with_tool_result(&status);
        let cas = SqliteCas::open(dir.db()).unwrap();

        let out = compress_request_body(&body, &cas, 1024);
        cas.flush();

        assert!(out.len() < body.len(), "status request shrinks");
        let rendered = tool_result_text(&out);
        assert!(
            rendered.contains("<<cof:"),
            "git status output carries a sentinel: {rendered}"
        );
        assert!(
            rendered.contains("On branch main"),
            "status head context survives"
        );
        assert!(
            rendered.contains("no changes added"),
            "status tail summary survives"
        );

        let rendered = strip_explainer(&rendered).to_string();
        let span = sentinel_span(&rendered);
        let hash = sentinel_hash(&rendered[span.clone()]);
        let recovered = read_blob(dir.db(), hash).unwrap().unwrap();
        let reconstructed = format!(
            "{}{}{}",
            &rendered[..span.start],
            std::str::from_utf8(&recovered).unwrap(),
            &rendered[span.end..]
        );

        assert_eq!(
            reconstructed, status,
            "proxy-compressed git status must remain byte-exactly recoverable"
        );
    }

    #[test]
    fn leaves_small_tool_result_and_authored_text_untouched() {
        let body = request_with_tool_result("tiny output");
        let cas = MemoryCas::new();
        let out = compress_request_body(&body, &cas, 1024);
        // small tool_result unchanged; authored user/assistant text never touched
        assert_eq!(tool_result_text(&out), "tiny output");
        let v: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["messages"][0]["content"], "how many records?");
        assert_eq!(v["messages"][1]["content"][0]["text"], "reading the data");
        assert!(cas.is_empty(), "nothing offloaded");
    }

    #[test]
    fn leaves_small_supported_messages_request_byte_exact() {
        let body = br#"{
  "model": "claude-opus-4-8",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "tool_result",
          "tool_use_id": "toolu_1",
          "content": "tiny output"
        }
      ]
    }
  ]
}"#;
        let cas = MemoryCas::new();
        assert_eq!(compress_request_body(body, &cas, 1024), body);
        assert!(cas.is_empty(), "nothing offloaded");
    }

    #[test]
    fn compresses_ollama_tool_messages_and_leaves_authored_untouched() {
        let cas = MemoryCas::new();
        // the tool result is a big JSON array (coffer offloads it); authored turns are plain text.
        let tool_output = big_json_array();
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "llama3.2",
            "messages": [
                {"role": "user", "content": "how many records?"},
                {"role": "tool", "content": tool_output},
                {"role": "assistant", "content": "thinking"}
            ]
        }))
        .unwrap();
        let (out, kind) = compress_ollama_body_kind(&body, &cas, 1024);
        assert_eq!(kind, TransformKind::Compressed);
        assert!(out.len() < body.len(), "the ollama tool message shrank");
        let v: Value = serde_json::from_slice(&out).unwrap();
        // the tool content became a sentinel render; authored user/assistant text is untouched.
        assert!(
            v["messages"][1]["content"]
                .as_str()
                .unwrap()
                .contains("<<cof:")
        );
        assert_eq!(v["messages"][0]["content"], "how many records?");
        assert_eq!(v["messages"][2]["content"], "thinking");
        // a body with no tool message is fail-open passthrough.
        let no_tool =
            serde_json::to_vec(&serde_json::json!({"messages":[{"role":"user","content":"hi"}]}))
                .unwrap();
        assert_eq!(compress_ollama_body(&no_tool, &cas, 1), no_tool);
    }

    /// Build an Ollama body and recover the tool message's original content by reading the
    /// offloaded blob from the shared SQLite CAS by the sentinel's short hash, then splicing it
    /// back into the compact render. Returns `(recovered_tool_content, parsed_output_value)`.
    fn ollama_compress_and_recover(
        tool_content: &str,
        user_text: &str,
        assistant_text: &str,
    ) -> (String, Value) {
        let dir = TempDir::new("ollama-prop");
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "llama3.2",
            "messages": [
                {"role": "user", "content": user_text},
                {"role": "tool", "content": tool_content},
                {"role": "assistant", "content": assistant_text}
            ]
        }))
        .unwrap();
        let cas = SqliteCas::open(dir.db()).unwrap();
        let out = compress_ollama_body(&body, &cas, 1024);
        cas.flush();

        let v: Value = serde_json::from_slice(&out).unwrap();
        let rendered = v["messages"][1]["content"].as_str().unwrap().to_string();
        let rendered = strip_explainer(&rendered).to_string();
        let span = sentinel_span(&rendered);
        let hash = sentinel_hash(&rendered[span.clone()]);
        let recovered = read_blob(dir.db(), hash).unwrap().unwrap();
        let reconstructed = format!(
            "{}{}{}",
            &rendered[..span.start],
            std::str::from_utf8(&recovered).unwrap(),
            &rendered[span.end..]
        );
        (reconstructed, v)
    }

    #[test]
    fn ollama_falsification_control_catches_a_wrong_recovery() {
        // Show the recovery check is load-bearing: if the offloaded bytes were corrupted (one byte
        // flipped) the byte-exact reconstruction would DIFFER from the original — the property's
        // assertion must reject that. We mutate the recovered bytes by hand to simulate a broken
        // feature and confirm inequality, then confirm the honest path is equal.
        let big = big_json_array();
        let (reconstructed, _v) = ollama_compress_and_recover(&big, "how many?", "thinking");
        assert_eq!(reconstructed, big, "honest recovery must be byte-exact");

        let mut corrupted = reconstructed.clone().into_bytes();
        let i = corrupted.iter().position(|&b| b == b'd').unwrap_or(0);
        corrupted[i] = corrupted[i].wrapping_add(1);
        assert_ne!(
            String::from_utf8(corrupted).unwrap(),
            big,
            "a single mutated byte must make the byte-exact check FAIL — the check is not vacuous"
        );
    }

    proptest! {
        // The CAS is durable on disk per case, so keep the case count modest (each case opens a
        // fresh SQLite file). 64 cases is plenty to fuzz array width, authored text, and unicode.
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// REVERSIBILITY over fuzzed Ollama bodies: the offloaded tool content recovered from the
        /// shared CAS by the rendered short hash, spliced back into the compact render, equals the
        /// original tool content byte-for-byte; authored turns are untouched.
        #[test]
        fn ollama_tool_message_reverses_byte_exact(
            rows in proptest::collection::vec(
                proptest::collection::btree_map("[a-z]{1,6}", "[a-zA-Z0-9 ._/-]{0,40}", 1..6),
                30..120,
            ),
            user_text in "[a-zA-Z0-9 ?.!]{0,80}",
            assistant_text in "[a-zA-Z0-9 ?.!]{0,80}",
        ) {
            use serde_json::{Map, Value as JVal};
            // Independent reference: a big JSON array of objects, large enough to clear the 1024B
            // proxy threshold and drive the offload (Ref) path inside `squash`.
            let array: Vec<JVal> = rows
                .into_iter()
                .map(|m| {
                    let obj: Map<String, JVal> =
                        m.into_iter().map(|(k, val)| (k, JVal::String(val))).collect();
                    JVal::Object(obj)
                })
                .collect();
            let tool_content = serde_json::to_string(&array).unwrap();
            prop_assume!(tool_content.len() >= 1024);

            let (reconstructed, v) =
                ollama_compress_and_recover(&tool_content, &user_text, &assistant_text);

            // (c) byte-exact reconstruct round-trip.
            prop_assert_eq!(
                &reconstructed,
                &tool_content,
                "recovered tool content must equal the original byte-for-byte"
            );
            // The compact render actually offloaded (sentinel present, render strictly smaller).
            let render = v["messages"][1]["content"].as_str().unwrap();
            prop_assert!(render.contains("<<cof:"), "tool message carries a sentinel");
            prop_assert!(
                render.len() < tool_content.len(),
                "tool message render must shrink (offload path was driven, not passthrough)"
            );
            // Authored turns are byte-identical to what we authored (only role:tool is touched).
            prop_assert_eq!(v["messages"][0]["content"].as_str().unwrap(), user_text.as_str());
            prop_assert_eq!(v["messages"][2]["content"].as_str().unwrap(), assistant_text.as_str());
            prop_assert_eq!(v["messages"][0]["role"].as_str().unwrap(), "user");
            prop_assert_eq!(v["messages"][2]["role"].as_str().unwrap(), "assistant");
        }

        /// Fail-open invariant under arbitrary bytes: any body that is NOT a well-formed Ollama
        /// /api/chat object with a role:"tool" string message must be returned UNCHANGED.
        #[test]
        fn ollama_fail_open_on_arbitrary_bytes(garbage in proptest::collection::vec(any::<u8>(), 0..4096)) {
            let cas = MemoryCas::new();
            // Exclude inputs that ARE a valid Ollama tool body (those legitimately transform).
            let parses_as_ollama_tool = serde_json::from_slice::<Value>(&garbage)
                .ok()
                .and_then(|v| v.get("messages").and_then(Value::as_array).map(|m| {
                    m.iter().any(|x| {
                        x.get("role").and_then(Value::as_str) == Some("tool")
                            && x.get("content").and_then(Value::as_str).is_some()
                    })
                }))
                .unwrap_or(false);
            prop_assume!(!parses_as_ollama_tool);
            prop_assert_eq!(compress_ollama_body(&garbage, &cas, 1024), garbage);
            prop_assert!(cas.is_empty(), "fail-open path must not offload anything");
        }
    }

    #[test]
    fn ollama_garbage_body_is_fail_open_unchanged() {
        let cas = MemoryCas::new();
        let garbage = b"\x00not an ollama body at all \xff{";
        assert_eq!(compress_ollama_body(garbage, &cas, 1024), garbage);
        assert!(cas.is_empty(), "garbage offloads nothing");
    }

    #[test]
    fn fail_open_on_non_json_body() {
        let cas = MemoryCas::new();
        let garbage = b"not json at all";
        assert_eq!(compress_request_body(garbage, &cas, 1024), garbage);
        // the Responses transform is fail-open on the same garbage.
        assert_eq!(compress_responses_body(garbage, &cas, 1024), garbage);
    }

    #[test]
    fn transform_kind_distinguishes_compressed_noshrink_passthrough() {
        use super::TransformKind::{Compressed, NoShrink, Passthrough};
        use super::compress_request_body_kind;
        let cas = MemoryCas::new();
        // a big compressible tool_result → Compressed
        let big = big_json_array();
        let (_, kind) = compress_request_body_kind(&request_with_tool_result(&big), &cas, 1024);
        assert_eq!(kind, Compressed);
        // recognized Messages shape, but the block is below the per-block threshold → benign NoShrink
        let (out, kind) =
            compress_request_body_kind(&request_with_tool_result("short tool text"), &cas, 20);
        assert_eq!(kind, NoShrink);
        assert_eq!(
            tool_result_text(&out),
            "short tool text",
            "unchanged on no-shrink"
        );
        // not the expected shape → fail-open Passthrough (non-JSON, and missing the marker)
        assert_eq!(
            compress_request_body_kind(b"not json at all", &cas, 1).1,
            Passthrough
        );
        let no_marker = serde_json::to_vec(
            &serde_json::json!({"messages": [{"role": "user", "content": "hi"}]}),
        )
        .unwrap();
        assert_eq!(
            compress_request_body_kind(&no_marker, &cas, 1).1,
            Passthrough
        );
    }

    #[test]
    fn leaves_small_supported_responses_request_byte_exact() {
        let body = br#"{
  "model": "gpt-5.5",
  "input": [
    {
      "type": "local_shell_call_output",
      "call_id": "c1",
      "output": "tiny output"
    }
  ]
}"#;
        let cas = MemoryCas::new();
        assert_eq!(compress_responses_body(body, &cas, 1024), body);
        assert!(cas.is_empty(), "nothing offloaded");
    }

    #[test]
    fn large_messages_request_without_tool_result_skips_transform_byte_exact() {
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "claude-opus-4-8",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "large authored text ".repeat(400)}
            ]
        }))
        .unwrap();
        assert!(body.len() > 1024);
        let cas = MemoryCas::new();

        assert_eq!(compress_request_body(&body, &cas, 1024), body);
        assert!(cas.is_empty(), "nothing offloaded");
    }

    #[test]
    fn large_responses_request_without_call_output_skips_transform_byte_exact() {
        let body = serde_json::to_vec(&serde_json::json!({
            "model": "gpt-5.5",
            "input": [
                {"type": "message", "role": "developer", "content": [
                    {"type": "input_text", "text": "large authored text ".repeat(400)}
                ]}
            ]
        }))
        .unwrap();
        assert!(body.len() > 1024);
        let cas = MemoryCas::new();

        assert_eq!(compress_responses_body(&body, &cas, 1024), body);
        assert!(cas.is_empty(), "nothing offloaded");
    }

    #[test]
    fn responses_compresses_tool_output_only() {
        // The codex/Responses shape: authored `message` content + a `*_call_output` item carrying
        // the shell tool's output. Only the latter is compressed.
        let log = (0..400)
            .map(|i| format!("2026-06-07 INFO event {i} ok"))
            .collect::<Vec<_>>()
            .join("\n");
        let authored = "BIG AUTHORED DEVELOPER CONTEXT ".repeat(100);
        let req = serde_json::json!({
            "model": "gpt-5.5",
            "instructions": "you are codex",
            "input": [
                {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": authored}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "summarize the log"}]},
                {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": "{}"},
                {"type": "local_shell_call_output", "call_id": "c1", "output": log}
            ],
            "tools": []
        });
        let body = serde_json::to_vec(&req).unwrap();
        let cas = MemoryCas::new();
        let out = compress_responses_body(&body, &cas, 1024);

        assert!(out.len() < body.len(), "request shrinks");
        let v: Value = serde_json::from_slice(&out).unwrap();
        // the *_call_output tool output is compressed and offloaded
        let tool_out = v["input"][3]["output"].as_str().unwrap();
        assert!(
            tool_out.contains("<<cof:"),
            "tool output carries a sentinel: {tool_out}"
        );
        assert!(!cas.is_empty(), "offloaded bytes retained for retrieval");
        // authored developer / user message content is untouched
        let dev = v["input"][0]["content"][0]["text"].as_str().unwrap();
        assert!(
            dev.starts_with("BIG AUTHORED DEVELOPER CONTEXT") && !dev.contains("<<cof:"),
            "authored text verbatim"
        );
        assert_eq!(v["input"][1]["content"][0]["text"], "summarize the log");
    }
}
