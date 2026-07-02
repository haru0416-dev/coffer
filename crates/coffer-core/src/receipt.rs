//! A re-executable **exactness receipt** for a typed aggregate.
//!
//! An [`issue_receipt`] call records a small, serializable proof that, over a given input, a typed
//! predicate-conjunction + aggregate yields a specific value, backed by a specific set of rows whose
//! bytes hash to a recorded digest. [`verify_receipt`] re-executes that query over a candidate input
//! and reports whether the candidate reproduces the attested answer **byte-identically**, or carries
//! a tamper signal — with no model call and no network.
//!
//! This is the "cost-of-being-wrong" surface: the value of an exact aggregate is not just that it was
//! right when computed, but that a third party can re-run it later — after a process restart, on a
//! different machine — and get the same number or a precise mismatch. The receipt binds the answer to
//! the bytes that produced it through the same SHA-256 content hashing the CAS already trusts, so the
//! guarantee rests only on coffer's two mechanical invariants (byte-exact recovery + exact compute).
//!
//! The receipt stores the query as a self-contained serde projection (`op`/`agg` as strings) so it
//! never depends on the serialization of the engine's internal [`Op`]/[`Agg`] enums.

use coffer_cas::ContentHash;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::budget::Op;
use crate::index::{Agg, Predicate, pick_rows, query_aggregate};

/// One predicate of a receipt's query, projected to a serde-stable form (`op` is a lowercase string).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReceiptPredicate {
    /// The object field tested.
    pub field: String,
    /// The comparison operator, one of `eq|ne|gt|ge|lt|le`.
    pub op: String,
    /// The value compared against (a JSON number or string).
    pub value: Value,
}

/// A persisted, re-executable proof that a typed aggregate yields a specific value over specific rows.
///
/// Serialize it with `serde_json::to_string`, hand it to anyone, and they can [`verify_receipt`] it
/// against the data later. Equality is structural, so two receipts for the same query+answer compare
/// equal.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Receipt {
    /// The conjunctive filter (a row backs the answer only if it passes ALL of these). Empty = all rows.
    pub query: Vec<ReceiptPredicate>,
    /// The aggregate kind: `count|sum|mean|min|max`.
    pub agg: String,
    /// The aggregated field; present for all aggregates except `count`.
    pub agg_field: Option<String>,
    /// The attested numeric answer.
    pub value: f64,
    /// 0-based indices of the rows that back the answer (its provenance).
    pub matched: Vec<usize>,
    /// SHA-256 (64-hex) of the canonical bytes of the backing rows — the tamper anchor.
    pub backing_sha256: String,
    /// Byte length of the input the receipt was issued over.
    pub input_len: usize,
    /// SHA-256 (64-hex) of the full input — identifies which dataset the receipt attests.
    pub input_sha256: String,
    /// Optional issue timestamp (Unix seconds). The engine never reads the clock; a binary may set it.
    pub issued_unix: Option<u64>,
}

/// The outcome of verifying a [`Receipt`] against a candidate input.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum ReceiptVerdict {
    /// The input reproduces the attested answer AND backing rows byte-identically.
    Valid,
    /// The re-executed aggregate differs from the attested value — the data does not yield this answer.
    ValueMismatch {
        /// The value the receipt attests.
        expected: f64,
        /// The value the candidate input actually produces.
        actual: f64,
    },
    /// The value matches but the backing rows changed (different provenance set or different bytes).
    BackingTampered,
    /// The query could not be run over this input (not a JSON array, or a non-numeric aggregated field).
    Refused,
    /// The receipt itself carries an unknown `op`/`agg` string and cannot be re-executed.
    MalformedReceipt,
}

/// Issue a receipt for `agg` filtered by `predicates` over `input`.
///
/// Returns `None` exactly when [`query_aggregate`] refuses (input is not a JSON array, or the
/// aggregated field is present-but-non-numeric) — never a guessed receipt.
#[must_use]
pub fn issue_receipt(input: &[u8], predicates: &[Predicate], agg: &Agg) -> Option<Receipt> {
    let result = query_aggregate(input, predicates, agg)?;
    let backing = pick_rows(input, &result.matched)?;
    let (agg_kind, agg_field) = agg_to_parts(agg);
    Some(Receipt {
        query: predicates
            .iter()
            .map(|p| ReceiptPredicate {
                field: p.field.clone(),
                op: op_to_str(p.op).to_string(),
                value: p.value.clone(),
            })
            .collect(),
        agg: agg_kind,
        agg_field,
        value: result.value,
        matched: result.matched,
        backing_sha256: ContentHash::of(&backing).as_str().to_string(),
        input_len: input.len(),
        input_sha256: ContentHash::of(input).as_str().to_string(),
        issued_unix: None,
    })
}

/// Re-execute `receipt`'s query over `input` and report whether `input` reproduces the attested
/// answer byte-identically, or carries a tamper signal. No model call, no network — pure re-derivation.
#[must_use]
pub fn verify_receipt(receipt: &Receipt, input: &[u8]) -> ReceiptVerdict {
    let mut predicates = Vec::with_capacity(receipt.query.len());
    for rp in &receipt.query {
        let Some(op) = op_from_str(&rp.op) else {
            return ReceiptVerdict::MalformedReceipt;
        };
        predicates.push(Predicate {
            field: rp.field.clone(),
            op,
            value: rp.value.clone(),
        });
    }
    let Some(agg) = agg_from_parts(&receipt.agg, receipt.agg_field.as_deref()) else {
        return ReceiptVerdict::MalformedReceipt;
    };

    let Some(result) = query_aggregate(input, &predicates, &agg) else {
        return ReceiptVerdict::Refused;
    };
    if !approx_eq(result.value, receipt.value) {
        return ReceiptVerdict::ValueMismatch {
            expected: receipt.value,
            actual: result.value,
        };
    }
    let Some(backing) = pick_rows(input, &result.matched) else {
        return ReceiptVerdict::Refused;
    };
    if result.matched != receipt.matched
        || ContentHash::of(&backing).as_str() != receipt.backing_sha256
    {
        return ReceiptVerdict::BackingTampered;
    }
    ReceiptVerdict::Valid
}

/// Relative-tolerance float equality, so an exact integer count/sum compares equal across a
/// serialize/deserialize round-trip without being defeated by float formatting.
fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 + 1e-9 * b.abs()
}

/// The lowercase wire string for an [`Op`].
fn op_to_str(op: Op) -> &'static str {
    match op {
        Op::Eq => "eq",
        Op::Ne => "ne",
        Op::Gt => "gt",
        Op::Ge => "ge",
        Op::Lt => "lt",
        Op::Le => "le",
    }
}

/// Parse a lowercase wire string back to an [`Op`]; `None` for anything else.
fn op_from_str(s: &str) -> Option<Op> {
    Some(match s {
        "eq" => Op::Eq,
        "ne" => Op::Ne,
        "gt" => Op::Gt,
        "ge" => Op::Ge,
        "lt" => Op::Lt,
        "le" => Op::Le,
        _ => return None,
    })
}

/// Split an [`Agg`] into its `(kind, field)` wire parts.
fn agg_to_parts(agg: &Agg) -> (String, Option<String>) {
    match agg {
        Agg::Count => ("count".to_string(), None),
        Agg::Sum(f) => ("sum".to_string(), Some(f.clone())),
        Agg::Mean(f) => ("mean".to_string(), Some(f.clone())),
        Agg::Min(f) => ("min".to_string(), Some(f.clone())),
        Agg::Max(f) => ("max".to_string(), Some(f.clone())),
    }
}

/// Rebuild an [`Agg`] from its `(kind, field)` wire parts; `None` for an unknown kind or a missing
/// field where one is required.
fn agg_from_parts(kind: &str, field: Option<&str>) -> Option<Agg> {
    Some(match kind {
        "count" => Agg::Count,
        "sum" => Agg::Sum(field?.to_string()),
        "mean" => Agg::Mean(field?.to_string()),
        "min" => Agg::Min(field?.to_string()),
        "max" => Agg::Max(field?.to_string()),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::{Receipt, ReceiptVerdict, issue_receipt, verify_receipt};
    use crate::budget::Op;
    use crate::index::{Agg, Predicate};
    use serde_json::json;

    fn dataset() -> Vec<u8> {
        serde_json::to_vec(&json!([
            { "status": "ok",    "cost": 10 },
            { "status": "error", "cost": 40 },
            { "status": "ok",    "cost": 20 },
            { "status": "error", "cost": 60 }
        ]))
        .unwrap()
    }

    fn err_sum() -> (Vec<Predicate>, Agg) {
        (
            vec![Predicate {
                field: "status".into(),
                op: Op::Eq,
                value: json!("error"),
            }],
            Agg::Sum("cost".into()),
        )
    }

    #[test]
    fn receipt_survives_serialize_then_fresh_verify() {
        let data = dataset();
        let (preds, agg) = err_sum();
        let receipt = issue_receipt(&data, &preds, &agg).expect("issue");
        assert!(
            (receipt.value - 100.0).abs() < 1e-9,
            "sum cost where error = 100"
        );

        // Simulate a fresh process: the receipt crosses a serialize/deserialize boundary, the
        // in-memory query objects are gone, and only the bytes + the JSON receipt remain.
        let wire = serde_json::to_string(&receipt).expect("serialize");
        let reloaded: Receipt = serde_json::from_str(&wire).expect("deserialize");
        assert_eq!(reloaded, receipt);

        assert_eq!(verify_receipt(&reloaded, &data), ReceiptVerdict::Valid);
    }

    #[test]
    fn tamper_to_a_backing_row_is_caught() {
        let data = dataset();
        let (preds, agg) = err_sum();
        let receipt = issue_receipt(&data, &preds, &agg).expect("issue");

        // Inflate a backing row's cost: the re-executed sum no longer matches the attested value.
        let tampered = serde_json::to_vec(&json!([
            { "status": "ok",    "cost": 10 },
            { "status": "error", "cost": 999 },
            { "status": "ok",    "cost": 20 },
            { "status": "error", "cost": 60 }
        ]))
        .unwrap();
        match verify_receipt(&receipt, &tampered) {
            ReceiptVerdict::ValueMismatch { expected, actual } => {
                assert!((expected - 100.0).abs() < 1e-9);
                assert!((actual - 1059.0).abs() < 1e-9);
            }
            other => panic!("expected ValueMismatch, got {other:?}"),
        }
    }

    #[test]
    fn tamper_that_changes_backing_rows_without_changing_value_is_caught() {
        let data = dataset();
        let (preds, agg) = err_sum();
        let receipt = issue_receipt(&data, &preds, &agg).expect("issue");

        // Swap which error rows hold the cost so the SUM is still 100 but the backing bytes differ.
        let reshaped = serde_json::to_vec(&json!([
            { "status": "ok",    "cost": 10 },
            { "status": "error", "cost": 1 },
            { "status": "ok",    "cost": 20 },
            { "status": "error", "cost": 99 }
        ]))
        .unwrap();
        assert_eq!(
            verify_receipt(&receipt, &reshaped),
            ReceiptVerdict::BackingTampered
        );
    }

    #[test]
    fn non_array_input_refuses_to_issue() {
        let (preds, agg) = err_sum();
        assert!(issue_receipt(b"not json", &preds, &agg).is_none());
    }
}
