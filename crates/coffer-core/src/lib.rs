//! coffer-core: byte-exact, reversible compression of agent tool-output, plus an exact
//! compute-digest over the offloaded bytes.
//!
//! # The Stage 0 invariant
//!
//! `reconstruct(compress(x)) == x` byte-for-byte, for **any** input.
//!
//! Reversibility is *structural*, not algorithmic: whenever the compressor decides a
//! region is worth offloading, it stores that region's **original bytes** verbatim in a
//! [`coffer_cas::Cas`] and emits a [`Segment::Ref`]. Reconstruction splices the original
//! bytes back, so byte-exact recovery holds regardless of how cleverly the region is
//! summarized for the model — or even if content detection misclassifies it.
//!
//! Compression *effectiveness* (how much [`CompressedDoc::render_for_model`] shrinks
//! vs the input) is a **separate axis**, measured later under the protocol in
//! `docs/PREREGISTRATION.md`. It is never blended into the fidelity guarantee: a codec
//! can be a poor compressor and still be perfectly lossless.

#![warn(clippy::pedantic, missing_docs)]
// All float<->integer casts in this crate operate on values bounded by the input length or an
// array index (token counts, percentile ranks, byte offsets) — far below 2^52 / usize::MAX on any
// realistic input — so precision/truncation/sign loss cannot occur. The arithmetic is in
// `apply_reduction` (budget %) and the `index` digests (mean/percentile).
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
// Test fixtures favor readability over the perf/style nits that matter only in the shipped path.
#![cfg_attr(
    test,
    allow(clippy::format_push_string, clippy::redundant_closure_for_method_calls)
)]

mod budget;
mod compress;
mod compressor;
mod view;

pub mod attest;
pub mod detect;
pub mod doc;
pub mod index;
pub mod receipt;
pub mod redact;

/// Lower-level building blocks for composing custom unit selections — the affordance the
/// coffer-eval / coffer-lens arms tile over. Most callers want [`Compressor`], [`compress`], or
/// [`compress_to_budget`] instead; these are here for building bespoke keep/drop strategies.
pub mod low_level {
    pub use crate::budget::{emit_kept, target_for_reduction, units_for};
}

pub use attest::{AttestReport, attest};
pub use budget::{
    Budget, Op, compress_by_predicate, compress_dedup, compress_json_where, compress_to_budget,
};
pub use compress::{MIN_COMPRESS_BYTES, compress};
pub use compressor::{CompressError, Compressor};
pub use detect::{ContentType, detect};
pub use doc::{CompressedDoc, ReconstructError, Segment};
pub use index::{
    Agg, GroupAggregate, GroupBucket, Predicate, QueryResult, aggregate_index, bucket_aggregate,
    count_matches_per_window, describe, digest, digest_across, digest_ndjson, join_aggregate,
    join_group_aggregate, pick_rows, query_aggregate, query_subset, superlative_rows,
};
pub use receipt::{Receipt, ReceiptPredicate, ReceiptVerdict, issue_receipt, verify_receipt};
pub use redact::{Redacted, Secret, redact_secrets};
pub use view::{CompactViewDoc, compress_structural_code_to_budget, structural_code_view};
