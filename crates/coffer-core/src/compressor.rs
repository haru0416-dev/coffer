//! The blessed front door for compression.
//!
//! [`Compressor`] is a small builder over the free functions ([`crate::compress`],
//! [`crate::compress_to_budget`]): it carries the three orthogonal knobs — `budget`, `counter`,
//! `min_bytes` — with sensible defaults, and lets the type system reject the one invalid
//! combination (a token/reduction budget with no tokenizer) as a [`CompressError`] instead of a
//! silent fallback.
//!
//! ```
//! use coffer_core::{Budget, Compressor};
//! use coffer_cas::MemoryCas;
//! use coffer_tokenizer::HeuristicCounter;
//!
//! let cas = MemoryCas::new();
//! let counter = HeuristicCounter;
//! let input = br#"[{"id":1},{"id":2}]"#;
//!
//! let doc = Compressor::new()
//!     .budget(Budget::Tokens(8))
//!     .counter(&counter)
//!     .compress(input, &cas)
//!     .expect("a counter is set");
//!
//! assert_eq!(doc.reconstruct(&cas).unwrap(), input); // byte-exact, as always
//! ```

use coffer_cas::Cas;
use coffer_tokenizer::TokenCounter;

use crate::budget::{Budget, compress_to_budget};
use crate::compress::{MIN_COMPRESS_BYTES, offload_whole_min};
use crate::doc::CompressedDoc;

/// Why a [`Compressor`] could not run. Distinct from [`crate::ReconstructError`], which is
/// about recovery, not configuration.
#[derive(Debug, thiserror::Error)]
pub enum CompressError {
    /// A [`Budget::Tokens`] or [`Budget::Reduction`] budget was set without a token counter.
    /// Both need to measure the render in the target model's tokens; set one with
    /// [`Compressor::counter`].
    #[error("a {budget:?} budget needs a token counter; set one with Compressor::counter")]
    CounterRequired {
        /// The budget that required a counter.
        budget: Budget,
    },
}

/// A configurable front door for byte-exact reversible compression.
///
/// Build with [`Compressor::new`], set knobs, then call [`Compressor::compress`]. Defaults to
/// [`Budget::MaxReduction`] (the Stage-0 whole-input behavior) with the standard
/// `MIN_COMPRESS_BYTES` threshold and no counter. The direct free functions
/// ([`crate::compress`], [`crate::compress_to_budget`]) remain available for callers that do
/// not need configuration validation.
#[derive(Clone, Copy)]
pub struct Compressor<'a> {
    budget: Budget,
    counter: Option<&'a dyn TokenCounter>,
    min_bytes: usize,
}

// `&dyn TokenCounter` is not `Debug`; show the counter by its model label instead.
impl std::fmt::Debug for Compressor<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Compressor")
            .field("budget", &self.budget)
            .field("min_bytes", &self.min_bytes)
            .field("counter", &self.counter.map(TokenCounter::model_label))
            .finish()
    }
}

impl Default for Compressor<'_> {
    fn default() -> Self {
        Self {
            budget: Budget::MaxReduction,
            counter: None,
            min_bytes: MIN_COMPRESS_BYTES,
        }
    }
}

impl<'a> Compressor<'a> {
    /// A compressor with default knobs ([`Budget::MaxReduction`], `MIN_COMPRESS_BYTES`, no
    /// counter).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the compression target. A [`Budget::Tokens`]/[`Budget::Reduction`] target also
    /// requires [`Compressor::counter`].
    #[must_use]
    pub fn budget(mut self, budget: Budget) -> Self {
        self.budget = budget;
        self
    }

    /// Set the token counter used to measure the model-facing render against a token budget.
    #[must_use]
    pub fn counter(mut self, counter: &'a dyn TokenCounter) -> Self {
        self.counter = Some(counter);
        self
    }

    /// Override the minimum input size below which a compressible input is passed through
    /// verbatim. Only affects the [`Budget::MaxReduction`] path (a token/reduction budget
    /// honors its target regardless of size).
    #[must_use]
    pub fn min_bytes(mut self, min_bytes: usize) -> Self {
        self.min_bytes = min_bytes;
        self
    }

    /// Compress `input`, offloading any original bytes into `cas`. The result is byte-exactly
    /// reconstructable (`doc.reconstruct(cas) == input`).
    ///
    /// # Errors
    /// [`CompressError::CounterRequired`] if the budget is [`Budget::Tokens`] or
    /// [`Budget::Reduction`] but no counter was set.
    pub fn compress(&self, input: &[u8], cas: &dyn Cas) -> Result<CompressedDoc, CompressError> {
        match self.budget {
            Budget::MaxReduction => Ok(offload_whole_min(input, cas, self.min_bytes)),
            Budget::Tokens(_) | Budget::Reduction(_) => {
                let counter = self.counter.ok_or(CompressError::CounterRequired {
                    budget: self.budget,
                })?;
                Ok(compress_to_budget(input, cas, self.budget, counter))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coffer_cas::MemoryCas;
    use coffer_tokenizer::HeuristicCounter;

    fn big_json() -> Vec<u8> {
        let items: Vec<String> = (0..40)
            .map(|i| format!(r#"{{"id":{i},"v":"xxxxxxxx"}}"#))
            .collect();
        format!("[{}]", items.join(",")).into_bytes()
    }

    #[test]
    fn default_is_max_reduction_and_round_trips() {
        let cas = MemoryCas::new();
        let input = big_json();
        let doc = Compressor::new().compress(&input, &cas).unwrap();
        // big compressible JSON → offloaded to a single Ref, byte-exact on reconstruct.
        assert_eq!(doc.reconstruct(&cas).unwrap(), input);
        assert!(doc.rendered_len() < input.len());
    }

    #[test]
    fn token_budget_without_counter_is_an_error() {
        let cas = MemoryCas::new();
        let err = Compressor::new()
            .budget(Budget::Tokens(50))
            .compress(&big_json(), &cas)
            .unwrap_err();
        assert!(matches!(
            err,
            CompressError::CounterRequired {
                budget: Budget::Tokens(50)
            }
        ));
    }

    #[test]
    fn reduction_budget_without_counter_is_an_error() {
        let cas = MemoryCas::new();
        let err = Compressor::new()
            .budget(Budget::Reduction(0.8))
            .compress(&big_json(), &cas)
            .unwrap_err();
        assert!(matches!(err, CompressError::CounterRequired { .. }));
    }

    #[test]
    fn token_budget_with_counter_round_trips() {
        let cas = MemoryCas::new();
        let counter = HeuristicCounter;
        let input = big_json();
        let doc = Compressor::new()
            .budget(Budget::Tokens(20))
            .counter(&counter)
            .compress(&input, &cas)
            .unwrap();
        assert_eq!(doc.reconstruct(&cas).unwrap(), input);
    }

    #[test]
    fn min_bytes_controls_the_passthrough_threshold() {
        let cas = MemoryCas::new();
        let small = br#"[{"id":1},{"id":2}]"#; // below MIN_COMPRESS_BYTES

        // Default threshold: too small to be worth offloading → passthrough (one verbatim seg).
        let passthrough = Compressor::new().compress(small, &cas).unwrap();
        assert_eq!(passthrough.segments.len(), 1);
        assert_eq!(passthrough.reconstruct(&cas).unwrap(), &small[..]);

        // Threshold lowered to 0: now offloaded to a Ref, still byte-exact.
        let offloaded = Compressor::new()
            .min_bytes(0)
            .compress(small, &cas)
            .unwrap();
        assert_eq!(offloaded.reconstruct(&cas).unwrap(), &small[..]);
    }
}
