//! Token counting with a seam for **per-model parity**.
//!
//! Token *savings* are only real when counted with the target model's own tokenizer.
//! Stage 0 ships the [`TokenCounter`] trait plus a cheap heuristic so the rest of the
//! system can be wired up; the real counters land later (see `docs/PREREGISTRATION.md`
//! §4 and `docs/REFERENCES.md`):
//!
//! - **OpenAI/GPT** — offline, byte-exact via tiktoken-rs (`TiktokenCounter`, behind
//!   the `tiktoken` feature).
//! - **Claude** — no usable offline tokenizer for Claude 3+; use the Anthropic
//!   `count_tokens` API, reconciled against the billed `usage.input_tokens` (Opus 4.7
//!   changed the tokenizer, so counts are model-versioned).
//!
//! Every counter reports a [`TokenCounter::model_label`] so any number we publish can
//! be attributed to a specific tokenizer version. Never cross-apply one model's
//! tokenizer to another.

#![warn(clippy::pedantic, missing_docs)]

/// Counts tokens for a specific model/tokenizer.
pub trait TokenCounter {
    /// The number of tokens `text` encodes to.
    fn count(&self, text: &str) -> usize;

    /// A stable label identifying which tokenizer/model this counter represents.
    fn model_label(&self) -> &'static str;

    /// If this counter's token count is a pure function of the *character* count (e.g. the
    /// chars/4 heuristic), return the token count a render of `char_count` characters would
    /// produce. This lets a caller score a candidate render without materializing the string —
    /// the budget search uses it to estimate a probe in O(units) instead of rendering, hashing,
    /// and token-scanning the whole document per probe.
    ///
    /// Real subword tokenizers (BPE) are **not** character-linear — a token can span a variable
    /// number of characters and merges depend on the actual bytes — so they return `None` and the
    /// caller falls back to counting the real render. Default: `None`.
    fn count_for_char_count(&self, _char_count: usize) -> Option<usize> {
        None
    }
}

/// Cheap, model-agnostic approximation (~4 chars/token). For rough internal budgeting
/// only — **never** for headline numbers, which must use a real per-model counter.
#[derive(Debug, Clone, Copy, Default)]
pub struct HeuristicCounter;

impl TokenCounter for HeuristicCounter {
    fn count(&self, text: &str) -> usize {
        // ceil(chars / 4)
        text.chars().count().div_ceil(4)
    }

    fn model_label(&self) -> &'static str {
        "heuristic-chars/4"
    }

    fn count_for_char_count(&self, char_count: usize) -> Option<usize> {
        // Character-linear by construction: count() depends only on the character count, so a
        // caller can score a render of `char_count` chars without building the string.
        Some(char_count.div_ceil(4))
    }
}

#[cfg(feature = "tiktoken")]
pub use tiktoken::TiktokenCounter;

#[cfg(feature = "tiktoken")]
mod tiktoken {
    use tiktoken_rs::{CoreBPE, o200k_base};

    use super::TokenCounter;

    /// Offline, byte-exact parity with `OpenAI`'s `o200k_base` (`GPT-4o` / `GPT-5` family).
    pub struct TiktokenCounter {
        bpe: CoreBPE,
        label: &'static str,
    }

    // `CoreBPE` is not `Debug`; print the label only.
    impl std::fmt::Debug for TiktokenCounter {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("TiktokenCounter")
                .field("label", &self.label)
                .finish_non_exhaustive()
        }
    }

    impl TiktokenCounter {
        /// Build a counter for `o200k_base`.
        ///
        /// # Panics
        /// Panics only if the bundled `o200k_base` table fails to load — a build/packaging
        /// defect, never a runtime/input condition.
        #[must_use]
        pub fn o200k() -> Self {
            Self {
                bpe: o200k_base().expect("bundled o200k_base data should load"),
                label: "openai-o200k_base",
            }
        }
    }

    impl TokenCounter for TiktokenCounter {
        fn count(&self, text: &str) -> usize {
            // `encode_ordinary` skips the special-token scan that `encode_with_special_tokens`
            // runs every call. For content without literal `<|...|>` markers (JSON, logs, RAG —
            // everything coffer compresses) the token sequence is IDENTICAL, so counts are unchanged
            // (budget-matching is unaffected) — it is just faster on the hot compression path.
            self.bpe.encode_ordinary(text).len()
        }

        fn model_label(&self) -> &'static str {
            self.label
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heuristic_is_ceil_quarter() {
        let c = HeuristicCounter;
        assert_eq!(c.count(""), 0);
        assert_eq!(c.count("abcd"), 1);
        assert_eq!(c.count("abcde"), 2);
        assert_eq!(c.model_label(), "heuristic-chars/4");
    }

    #[test]
    fn heuristic_char_count_estimate_agrees_with_count() {
        let c = HeuristicCounter;
        for s in ["", "abcd", "abcde", "a longer string with spaces"] {
            assert_eq!(
                c.count_for_char_count(s.chars().count()),
                Some(c.count(s)),
                "char-count estimate must equal the real count for the heuristic: {s:?}"
            );
        }
    }

    // o200k_base is a fixed, published vocabulary, so these counts are stable across tiktoken-rs
    // versions. The anchors lock coffer's headline property — byte-exact OpenAI token parity — so a
    // future tokenizer bump that silently shifts counts (e.g. a merge-algorithm change) fails here
    // instead of corrupting budget matching. Counts were taken from the o200k_base encoder itself.
    #[cfg(feature = "tiktoken")]
    #[test]
    fn tiktoken_o200k_parity_anchors() {
        let c = TiktokenCounter::o200k();
        assert_eq!(c.model_label(), "openai-o200k_base");
        assert_eq!(c.count(""), 0, "empty input must be zero tokens");

        for (text, expected) in [
            ("hello world", 2usize),
            (r#"{"name":"coffer","count":42}"#, 10),
            ("the quick brown fox jumps over the lazy dog", 9),
        ] {
            assert_eq!(
                c.count(text),
                expected,
                "o200k_base count drift for {text:?}"
            );
            // Encoding is deterministic: the same input always yields the same count.
            assert_eq!(c.count(text), c.count(text));
        }
    }
}
