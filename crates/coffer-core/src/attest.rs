//! Reversibility attestation: a one-call, falsifiable replay of the round-trip over a
//! corpus — compress each sample, reconstruct it, and assert byte-for-byte equality — turning the
//! Stage-0 invariant `reconstruct(compress(x)) == x` from an asserted property into a demonstrable
//! artifact. Reconstruction re-verifies each offloaded run's SHA-256 hash, so a passing report
//! proves the stored bytes are intact, not merely present.

use coffer_cas::Cas;
use coffer_tokenizer::TokenCounter;

use crate::{Budget, compress_to_budget};

/// The result of attesting a corpus.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttestReport {
    /// Number of samples attested.
    pub samples: usize,
    /// Samples that round-tripped byte-for-byte.
    pub passed: usize,
    /// Total original bytes round-tripped across the passing samples.
    pub bytes_round_tripped: usize,
    /// Index of the first sample that failed to reconstruct byte-exact, if any.
    pub first_failure: Option<usize>,
}

impl AttestReport {
    /// True iff every sample reconstructed byte-for-byte.
    #[must_use]
    pub fn ok(&self) -> bool {
        self.first_failure.is_none() && self.passed == self.samples
    }
}

/// Replay the reversibility round-trip over `samples`: compress each to a maximal-offload budget (so
/// the offload + recovery path is exercised, not passthrough), reconstruct it from `cas`, and check
/// it byte-for-byte against the original. Returns a falsifiable [`AttestReport`].
#[must_use]
pub fn attest(samples: &[&[u8]], cas: &dyn Cas, counter: &dyn TokenCounter) -> AttestReport {
    let mut passed = 0usize;
    let mut bytes_round_tripped = 0usize;
    let mut first_failure = None;
    for (i, sample) in samples.iter().enumerate() {
        let doc = compress_to_budget(sample, cas, Budget::Tokens(0), counter);
        if doc
            .reconstruct(cas)
            .is_ok_and(|out| out.as_slice() == *sample)
        {
            passed += 1;
            bytes_round_tripped += sample.len();
        } else if first_failure.is_none() {
            first_failure = Some(i);
        }
    }
    AttestReport {
        samples: samples.len(),
        passed,
        bytes_round_tripped,
        first_failure,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::attest;
    use coffer_cas::{Cas, ContentHash, MemoryCas};
    use coffer_tokenizer::HeuristicCounter;
    use proptest::prelude::*;

    #[test]
    fn attest_round_trips_a_corpus_byte_exact() {
        let cas = MemoryCas::new();
        let tok = HeuristicCounter;
        let json =
            br#"[{"name":"a","value":1},{"name":"b","value":2},{"name":"c","value":3},{"name":"d","value":4}]"#
                .as_slice();
        let log = b"2026-01-01T00:00:00 INFO boot\n2026-01-01T00:00:01 ERROR x\n2026-01-01T00:00:02 INFO boot\n2026-01-01T00:00:03 WARN y\n2026-01-01T00:00:04 INFO z\n2026-01-01T00:00:05 ERROR x\n2026-01-01T00:00:06 INFO boot\n2026-01-01T00:00:07 INFO done\n".as_slice();
        let text = b"plain prose that simply passes through unchanged".as_slice();
        let report = attest(&[json, log, text], &cas, &tok);
        assert!(report.ok(), "{report:?}");
        assert_eq!(report.passed, 3);
        assert_eq!(report.first_failure, None);
        assert_eq!(
            report.bytes_round_tripped,
            json.len() + log.len() + text.len()
        );
    }

    #[test]
    fn attest_empty_corpus_is_trivially_ok() {
        let report = attest(&[], &MemoryCas::new(), &HeuristicCounter);
        assert!(report.ok());
        assert_eq!(report.samples, 0);
    }

    /// A CAS wrapper that returns CORRUPTED bytes on `get`: it flips the high bit of the first byte
    /// of every retrieved payload while preserving its length. Length-preserving corruption slips
    /// past `reconstruct`'s length guard, so only the SHA-256 hash re-check can catch it.
    struct CorruptingCas {
        inner: MemoryCas,
    }

    impl CorruptingCas {
        fn new() -> Self {
            Self {
                inner: MemoryCas::new(),
            }
        }
    }

    impl Cas for CorruptingCas {
        fn put(&self, bytes: &[u8]) -> ContentHash {
            self.inner.put(bytes)
        }

        fn get(&self, hash: &ContentHash) -> Option<Arc<[u8]>> {
            self.inner.get(hash).map(|bytes| {
                let mut corrupted = bytes.to_vec();
                if let Some(first) = corrupted.first_mut() {
                    *first ^= 0x80; // same length, different bytes
                }
                Arc::<[u8]>::from(corrupted)
            })
        }

        fn contains(&self, hash: &ContentHash) -> bool {
            self.inner.contains(hash)
        }

        fn len(&self) -> usize {
            self.inner.len()
        }
    }

    /// A many-element JSON array reliably classifies as JSON and, under a zero-token budget, offloads
    /// its elements to the CAS — exercising the offload + recover path a corrupting backend can break.
    fn json_array_bytes(values: &[i64]) -> Vec<u8> {
        let body: Vec<String> = values
            .iter()
            .enumerate()
            .map(|(i, v)| format!(r#"{{"k{i}":{v}}}"#))
            .collect();
        format!("[{}]", body.join(",")).into_bytes()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]
        /// CORRECTNESS over a fuzzed corpus, cross-checked against an INDEPENDENT reference: every
        /// sample is a byte-exact-recoverable input, so the report must be all-pass and the byte
        /// total must equal the directly-summed input lengths.
        #[test]
        fn attest_round_trips_a_fuzzed_corpus_byte_exact(
            arr in proptest::collection::vec(any::<i64>(), 1..12),
            raw in proptest::collection::vec(any::<u8>(), 0..64),
            log_lines in proptest::collection::vec("[a-z ]{0,20}", 1..10),
        ) {
            let json = json_array_bytes(&arr);
            let parsed: serde_json::Value = serde_json::from_slice(&json).unwrap();
            prop_assert_eq!(parsed.as_array().map(Vec::len), Some(arr.len()));

            let mut log_s = String::new();
            for (i, l) in log_lines.iter().enumerate() {
                use std::fmt::Write as _;
                writeln!(log_s, "2026-01-01T00:00:{i:02} INFO {l}").unwrap();
            }
            let log: Vec<u8> = log_s.into_bytes();

            let samples: Vec<&[u8]> = vec![json.as_slice(), raw.as_slice(), log.as_slice()];
            let expected_bytes: usize = samples.iter().map(|s| s.len()).sum();

            let cas = MemoryCas::new();
            let report = attest(&samples, &cas, &HeuristicCounter);

            prop_assert!(report.ok(), "{report:?}");
            prop_assert_eq!(report.samples, 3);
            prop_assert_eq!(report.passed, 3);
            prop_assert_eq!(report.first_failure, None);
            prop_assert_eq!(report.bytes_round_tripped, expected_bytes);
        }
    }

    /// FALSIFICATION control: a clean CAS passes an offloading sample (positive control), but a
    /// length-preserving byte flip on `get` makes attest FAIL with `first_failure == Some(0)` —
    /// proving reconstruct's hash re-check is load-bearing, not decorative.
    #[test]
    fn attest_detects_corrupted_offloaded_bytes() {
        let tok = HeuristicCounter;
        let json = json_array_bytes(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let sample: &[u8] = json.as_slice();

        let clean = MemoryCas::new();
        let good = attest(&[sample], &clean, &tok);
        assert!(good.ok(), "clean CAS must pass: {good:?}");
        assert!(
            clean.len() > 0,
            "sample must offload to the CAS to be a real control"
        );

        let evil = CorruptingCas::new();
        let report = attest(&[sample], &evil, &tok);
        assert!(
            evil.len() > 0,
            "sample must have offloaded into the corrupting CAS"
        );
        assert!(!report.ok(), "corruption must be detected: {report:?}");
        assert_eq!(report.first_failure, Some(0), "{report:?}");
        assert_eq!(report.passed, 0, "{report:?}");
        assert_eq!(report.bytes_round_tripped, 0, "{report:?}");
    }
}
