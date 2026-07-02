//! The Stage 0 compressor.
//!
//! It classifies the input and, for compressible types big enough to be worth it,
//! offloads the **whole** input to the CAS and renders a one-line structural summary.
//! This is intentionally the extreme CCR case (fully offloaded, retrieve-on-demand) —
//! it proves the reversible mechanism and the losslessness gate without yet doing
//! selective intra-document pruning (keep keys / drop values), which is a later stage.
//! The [`Segment`] model already supports multiple/partial segments, so finer-grained
//! codecs slot in without changing the invariant.

use coffer_cas::Cas;

use crate::detect::{ContentType, detect_with_value};
use crate::doc::{CompressedDoc, Segment};

/// Inputs smaller than this are passed through unchanged — not worth a CAS round-trip
/// or the sentinel overhead.
pub const MIN_COMPRESS_BYTES: usize = 256;

/// Compress `input`, storing any offloaded original bytes in `cas`.
///
/// Guarantees `doc.reconstruct(cas) == input` byte-for-byte (see crate docs). Takes
/// `&dyn Cas` (interior mutability) so a shared store can back concurrent harness tasks.
/// This is the Stage-0 whole-input behavior; [`crate::compress_to_budget`] adds a budget
/// knob for the Stage-1 tradeoff curve.
#[must_use]
pub fn compress(input: &[u8], cas: &dyn Cas) -> CompressedDoc {
    offload_whole(input, cas)
}

/// Stage-0 whole-input behavior: offload the entire input as one `Ref` when it is a
/// compressible type at/above the size threshold, else pass through verbatim. Reused by
/// `compress_to_budget(.., MaxReduction, ..)` and as the safe fallback when an input
/// cannot be partitioned into droppable units.
pub(crate) fn offload_whole(input: &[u8], cas: &dyn Cas) -> CompressedDoc {
    offload_whole_min(input, cas, MIN_COMPRESS_BYTES)
}

/// [`offload_whole_min`] without the compressible-type gate: offloads ANY content type at or
/// above `min_bytes`. Reserved for the explicit-budget path — the caller asked for a token
/// target, so even an unpartitionable text blob must approach it rather than pass through.
pub(crate) fn offload_whole_any_min(
    input: &[u8],
    cas: &dyn Cas,
    min_bytes: usize,
) -> CompressedDoc {
    if input.len() < min_bytes {
        return CompressedDoc {
            segments: vec![Segment::Verbatim(input.to_vec())],
        };
    }
    let (content_type, parsed) = detect_with_value(input);
    let summary = summarize(content_type, input, parsed.as_ref());
    let hash = cas.put(input); // store the EXACT original bytes — never a re-serialization
    CompressedDoc {
        segments: vec![Segment::Ref {
            hash,
            summary,
            original_len: input.len(),
        }],
    }
}

/// [`offload_whole`] with a caller-chosen minimum-size threshold (the `Compressor` `min_bytes`
/// knob): a compressible input below `min_bytes` is passed through verbatim.
pub(crate) fn offload_whole_min(input: &[u8], cas: &dyn Cas, min_bytes: usize) -> CompressedDoc {
    // Parse once: `detect_with_value` hands back the JSON value it already parsed, so the
    // summarizer does not re-parse the same bytes (was the `NOTE(stage1)` double-parse).
    let (content_type, parsed) = detect_with_value(input);
    let worth_compressing =
        matches!(content_type, ContentType::Json | ContentType::Log) && input.len() >= min_bytes;

    if !worth_compressing {
        return CompressedDoc {
            segments: vec![Segment::Verbatim(input.to_vec())],
        };
    }

    let summary = summarize(content_type, input, parsed.as_ref());
    let hash = cas.put(input); // store the EXACT original bytes — never a re-serialization
    CompressedDoc {
        segments: vec![Segment::Ref {
            hash,
            summary,
            original_len: input.len(),
        }],
    }
}

fn summarize(
    content_type: ContentType,
    input: &[u8],
    parsed: Option<&serde_json::Value>,
) -> String {
    match content_type {
        ContentType::Json => summarize_json(parsed, input.len()),
        ContentType::Log => summarize_log(input),
        ContentType::Text => format!("text, {} bytes", input.len()),
    }
}

/// Summarize the already-parsed JSON `value`. Key order is the input's (the workspace enables
/// `serde_json`'s `preserve_order`), so the summary is byte-identical to parsing here directly.
fn summarize_json(value: Option<&serde_json::Value>, byte_len: usize) -> String {
    match value {
        Some(serde_json::Value::Array(items)) => {
            let keys = items
                .first()
                .and_then(serde_json::Value::as_object)
                .map(|obj| obj.keys().cloned().collect::<Vec<_>>().join(","))
                .unwrap_or_default();
            if keys.is_empty() {
                format!("json array, {} items", items.len())
            } else {
                format!("json array, {} items, keys: {keys}", items.len())
            }
        }
        Some(serde_json::Value::Object(obj)) => {
            format!(
                "json object, keys: {}",
                obj.keys().cloned().collect::<Vec<_>>().join(",")
            )
        }
        _ => format!("json, {byte_len} bytes"),
    }
}

fn summarize_log(input: &[u8]) -> String {
    let lines = String::from_utf8_lossy(input).lines().count();
    format!("log, {lines} lines")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::detect_with_value;

    /// Parse-once must not reorder keys: with `preserve_order`, the summary lists keys in the
    /// input's order (here deliberately non-alphabetical), proving byte-identity with the old
    /// re-parse-in-summarize path.
    #[test]
    fn summarize_json_array_preserves_input_key_order() {
        let input = br#"[{"zebra":1,"apple":2,"mango":3}]"#;
        let (ct, parsed) = detect_with_value(input);
        assert_eq!(ct, ContentType::Json);
        assert_eq!(
            summarize_json(parsed.as_ref(), input.len()),
            "json array, 1 items, keys: zebra,apple,mango"
        );
    }

    #[test]
    fn summarize_json_object_preserves_input_key_order() {
        let input = br#"{"gamma":1,"alpha":2}"#;
        let (_ct, parsed) = detect_with_value(input);
        assert_eq!(
            summarize_json(parsed.as_ref(), input.len()),
            "json object, keys: gamma,alpha"
        );
    }
}
