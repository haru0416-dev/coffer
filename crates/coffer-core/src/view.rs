//! Explicit compact views backed by a byte-exact CAS original.
//!
//! These are intentionally not part of the transparent [`CompressedDoc`](crate::CompressedDoc)
//! render path. The model sees a compact, lossy view plus a `<<cof:...>>` retrieval sentinel;
//! reconstruction still uses the exact original bytes stored in the CAS.

use std::fmt::Write as _;

use coffer_cas::Cas;
use coffer_tokenizer::TokenCounter;

use crate::doc::{CompressedDoc, ReconstructError, Segment};

/// A lossy model-facing view with a byte-exact reconstruction document.
#[derive(Clone, Debug, PartialEq)]
pub struct CompactViewDoc {
    /// The text intended for the model: a retrieval sentinel plus a compact view.
    pub model_text: String,
    /// A document that reconstructs the exact original bytes from the CAS.
    pub compressed: CompressedDoc,
}

impl CompactViewDoc {
    /// Byte-exact reconstruction of the original input from `cas`.
    ///
    /// # Errors
    /// Returns [`ReconstructError`] if the CAS is missing or returns corrupt bytes for the
    /// offloaded original.
    pub fn reconstruct(&self, cas: &dyn Cas) -> Result<Vec<u8>, ReconstructError> {
        self.compressed.reconstruct(cas)
    }
}

/// Build a Repomix-style structural code view and store the exact source bytes in `cas`.
///
/// This is an explicit file-read/code-map affordance, not the default transparent compressor:
/// it keeps signatures and outlines in `model_text`, while the original file remains retrievable
/// through the leading `<<cof:...>>` sentinel and reconstructable through [`CompactViewDoc`].
#[must_use]
pub fn compress_structural_code_to_budget(
    input: &[u8],
    cas: &dyn Cas,
    target_tokens: usize,
    counter: &dyn TokenCounter,
) -> CompactViewDoc {
    let view = structural_code_view(input);
    compact_view_to_budget(
        input,
        &view,
        target_tokens,
        counter,
        cas,
        "structural code view",
    )
}

/// Return a heuristic structural view of code-like text: imports, declarations, attributes,
/// signatures, and omission markers for non-structural bodies.
///
/// This is parser-free by design. It is useful as a cheap read-mode outline, but callers should
/// treat it as a navigation surface and retrieve the original for exact edits.
#[must_use]
pub fn structural_code_view(input: &[u8]) -> String {
    let text = String::from_utf8_lossy(input);
    let mut out = String::new();
    let mut omitted = 0usize;
    for line in text.lines() {
        if is_structural_code_line(line) {
            flush_omitted_code_lines(&mut out, &mut omitted);
            out.push_str(line.trim_end());
            out.push('\n');
        } else if !line.trim().is_empty() {
            omitted += 1;
        }
    }
    flush_omitted_code_lines(&mut out, &mut omitted);
    if out.is_empty() {
        text.into_owned()
    } else {
        out
    }
}

fn compact_view_to_budget(
    input: &[u8],
    view: &str,
    target_tokens: usize,
    counter: &dyn TokenCounter,
    cas: &dyn Cas,
    summary: &str,
) -> CompactViewDoc {
    let hash = cas.put(input);
    let compressed = CompressedDoc {
        segments: vec![Segment::Ref {
            hash,
            summary: format!("{summary}, {} bytes", input.len()),
            original_len: input.len(),
        }],
    };
    let sentinel = compressed.render_for_model();
    let model_text = fit_compact_view_to_target(&sentinel, view, target_tokens, counter);
    CompactViewDoc {
        model_text,
        compressed,
    }
}

fn fit_compact_view_to_target(
    sentinel: &str,
    view: &str,
    target_tokens: usize,
    counter: &dyn TokenCounter,
) -> String {
    let header = format!("{sentinel}\n");
    let full = format!("{header}{view}");
    if counter.count(&full) <= target_tokens {
        return full;
    }
    if counter.count(&header) >= target_tokens {
        return header;
    }
    let lines: Vec<&str> = view.lines().collect();
    if lines.is_empty() {
        return header;
    }

    let mut best = header.clone();
    let (mut lo, mut hi) = (0usize, lines.len());
    while lo <= hi {
        let mid = lo + (hi - lo) / 2;
        let candidate = format!("{header}{}", render_head_tail_lines(&lines, mid));
        if counter.count(&candidate) <= target_tokens {
            best = candidate;
            lo = mid + 1;
        } else if mid == 0 {
            break;
        } else {
            hi = mid - 1;
        }
    }
    best
}

fn render_head_tail_lines(lines: &[&str], kept: usize) -> String {
    if kept >= lines.len() {
        return lines.join("\n") + "\n";
    }
    if kept == 0 {
        return String::new();
    }
    let head = kept.div_ceil(2);
    let tail = kept / 2;
    let omitted = lines.len().saturating_sub(head + tail);
    let mut out = String::new();
    for line in &lines[..head] {
        out.push_str(line);
        out.push('\n');
    }
    if omitted > 0 {
        let _ = writeln!(
            out,
            "... {omitted} compact-view lines omitted to fit budget ..."
        );
    }
    if tail > 0 {
        for line in &lines[lines.len() - tail..] {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

fn flush_omitted_code_lines(out: &mut String, omitted: &mut usize) {
    if *omitted > 0 {
        let _ = writeln!(out, "// ... {} non-structural lines omitted ...", *omitted);
        *omitted = 0;
    }
}

fn is_structural_code_line(line: &str) -> bool {
    let t = line.trim_start();
    if t.starts_with("#[") || t.starts_with('@') {
        return true;
    }
    let prefixes = [
        "use ",
        "pub use ",
        "mod ",
        "pub mod ",
        "struct ",
        "pub struct ",
        "enum ",
        "pub enum ",
        "trait ",
        "pub trait ",
        "impl ",
        "type ",
        "pub type ",
        "const ",
        "pub const ",
        "static ",
        "pub static ",
        "class ",
        "export class ",
        "interface ",
        "export interface ",
        "export type ",
        "function ",
        "export function ",
    ];
    prefixes.iter().any(|p| t.starts_with(p))
        || t.starts_with("fn ")
        || t.starts_with("pub fn ")
        || t.starts_with("async fn ")
        || t.starts_with("pub async fn ")
        || t.contains(" fn ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use coffer_cas::MemoryCas;
    use coffer_tokenizer::{HeuristicCounter, TokenCounter};

    fn code_fixture() -> Vec<u8> {
        br"
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
"
        .to_vec()
    }

    #[test]
    fn structural_code_view_keeps_signatures_not_bodies() {
        let view = structural_code_view(&code_fixture());
        assert!(view.contains("use std::collections::HashMap;"));
        assert!(view.contains("pub struct Widget"));
        assert!(view.contains("impl Widget"));
        assert!(view.contains("pub fn compute"));
        assert!(!view.contains("total += item"));
        assert!(view.contains("non-structural lines omitted"));
    }

    #[test]
    fn structural_code_compact_view_has_sentinel_and_reconstructs() {
        let input = code_fixture();
        let cas = MemoryCas::new();
        let counter = HeuristicCounter;
        let doc = compress_structural_code_to_budget(&input, &cas, 120, &counter);
        assert!(doc.model_text.contains("<<cof:"));
        assert!(doc.model_text.contains("pub struct Widget"));
        assert!(doc.model_text.contains("pub fn compute"));
        assert!(!doc.model_text.contains("total += item"));
        assert_eq!(doc.reconstruct(&cas).unwrap(), input);
    }

    #[test]
    fn structural_code_compact_view_fits_when_above_sentinel_floor() {
        let input = code_fixture();
        let cas = MemoryCas::new();
        let counter = HeuristicCounter;
        let target = 40;
        let doc = compress_structural_code_to_budget(&input, &cas, target, &counter);
        let tokens = counter.count(&doc.model_text);
        assert!(tokens <= target, "{tokens} > {target}\n{}", doc.model_text);
        assert!(doc.model_text.contains("<<cof:"));
        assert_eq!(doc.reconstruct(&cas).unwrap(), input);
    }
}
