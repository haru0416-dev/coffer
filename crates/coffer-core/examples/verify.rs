//! Runnable proof of the two top "verifier" strategies, with NO model call and NO network:
//!
//! 1. claim-check — recompute any number an agent reports over the held bytes and return
//!    AGREE / DISAGREE + the exact answer + provenance (the "lie-detector").
//! 2. exactness receipt — issue a small portable proof, hand it across a fresh process, and
//!    re-verify it byte-identically later; a tampered dataset trips a mismatch.
//! 3. consensus — when two agents report DIFFERENT numbers, coffer holds ground truth and
//!    adjudicates which one is right.
//!
//! Everything here is exact and deterministic — the same output on any machine.
//!
//! Run:  cargo run --release -p coffer-core --example verify

use coffer_core::{
    Agg, Op, Predicate, Receipt, ReceiptVerdict, issue_receipt, query_aggregate, verify_receipt,
};
use serde_json::{Value, json};

/// A small, fixed `kubectl get pods`-shaped dataset with one pod crash-looping far beyond the rest —
/// the kind of buried needle a head/tail glance misses and an agent then miscounts.
fn pods() -> Vec<Value> {
    vec![
        json!({ "name": "api-0",   "status": "Running",          "restarts": 0 }),
        json!({ "name": "api-1",   "status": "Running",          "restarts": 1 }),
        json!({ "name": "worker-0","status": "CrashLoopBackOff", "restarts": 7 }),
        json!({ "name": "worker-1","status": "Running",          "restarts": 0 }),
        json!({ "name": "cache-0", "status": "CrashLoopBackOff", "restarts": 4 }),
        json!({ "name": "db-0",    "status": "Running",          "restarts": 2 }),
        json!({ "name": "job-7",   "status": "CrashLoopBackOff", "restarts": 938 }), // the needle
        json!({ "name": "api-2",   "status": "Running",          "restarts": 0 }),
        json!({ "name": "cache-1", "status": "Pending",          "restarts": 0 }),
        json!({ "name": "db-1",    "status": "Running",          "restarts": 1 }),
    ]
}

fn eq(field: &str, value: &str) -> Predicate {
    Predicate {
        field: field.into(),
        op: Op::Eq,
        value: json!(value),
    }
}

/// Recompute `agg` over the held bytes and judge an agent's `claimed` number against the exact truth.
fn check_claim(input: &[u8], predicates: &[Predicate], agg: &Agg, claimed: f64) {
    let r = query_aggregate(input, predicates, agg).expect("aggregate");
    let agree = (r.value - claimed).abs() <= 1e-9 + 1e-9 * r.value.abs();
    let verdict = if agree { "✅ AGREE" } else { "❌ DISAGREE" };
    let shown: Vec<usize> = r.matched.iter().take(8).copied().collect();
    println!(
        "  agent claims {claimed:>5} → {verdict}  (exact answer {:>5}, backed by rows {shown:?})",
        r.value,
    );
}

fn main() {
    let rows = pods();
    let held = serde_json::to_vec(&Value::Array(rows.clone())).expect("serialize");

    // Ground truth, computed in Rust over ALL rows.
    let crash = query_aggregate(&held, &[eq("status", "CrashLoopBackOff")], &Agg::Count)
        .expect("count")
        .value;
    let total_restarts = query_aggregate(
        &held,
        &[Predicate {
            field: "restarts".into(),
            op: Op::Ge,
            value: json!(0),
        }],
        &Agg::Sum("restarts".into()),
    )
    .expect("sum")
    .value;

    println!(
        "# coffer verify — exact claim-check + re-executable receipt (no model, no network)\n"
    );
    println!(
        "dataset: {} pods held server-side. ground truth over all rows: CrashLoopBackOff = {crash:.0}, total restarts = {total_restarts:.0}\n",
        rows.len(),
    );

    // 1. Claim-check: a confidently-wrong number is caught; the right one is confirmed.
    println!("## 1. Claim-check — recompute the number an agent reports (the lie-detector)\n");
    let crash_q = [eq("status", "CrashLoopBackOff")];
    check_claim(&held, &crash_q, &Agg::Count, 2.0); // agent eyeballed the visible rows, missed the needle's row
    check_claim(&held, &crash_q, &Agg::Count, crash); // agent got it right
    println!(
        "  → a wrong count is caught against the held bytes, with the backing rows to audit.\n"
    );

    // 2. Re-executable exactness receipt: issue, cross a fresh process, re-verify, then tamper.
    println!("## 2. Re-executable exactness receipt — survives a fresh process, trips on tamper\n");
    let receipt = issue_receipt(&held, &crash_q, &Agg::Count).expect("receipt");
    let wire = serde_json::to_string(&receipt).expect("serialize receipt");
    println!("  issued a portable receipt (hand it to anyone):\n    {wire}\n");

    // Model a fresh process / a third party weeks later: nothing survives but the receipt JSON and
    // the held bytes. Rebuild the receipt purely from its serialized form.
    let reloaded: Receipt = serde_json::from_str(&wire).expect("deserialize receipt");
    println!(
        "  --- fresh process: only the receipt JSON + the held bytes remain ---\n  re-verify → {}",
        verdict_str(&verify_receipt(&reloaded, &held)),
    );

    // Tamper the dataset: flip the needle's restarts so it no longer reads as CrashLoopBackOff-by-cost.
    let mut tampered_rows = rows.clone();
    tampered_rows[6]["status"] = json!("Running"); // hide the needle
    let tampered = serde_json::to_vec(&Value::Array(tampered_rows)).expect("serialize");
    println!(
        "  tamper the held bytes (hide one CrashLoopBackOff pod), re-verify → {}",
        verdict_str(&verify_receipt(&reloaded, &tampered)),
    );
    println!(
        "  → the receipt re-derives the answer from the bytes; doctoring the data is detected.\n"
    );

    // 3. Consensus: two agents disagree; coffer is the only party holding ground truth.
    println!("## 3. Consensus — coffer adjudicates a disagreement no single agent can settle\n");
    let restarts_q = [Predicate {
        field: "restarts".into(),
        op: Op::Ge,
        value: json!(0),
    }];
    println!("  question: total restarts across all pods");
    check_claim(
        &held,
        &restarts_q,
        &Agg::Sum("restarts".into()),
        total_restarts,
    ); // agent A
    check_claim(
        &held,
        &restarts_q,
        &Agg::Sum("restarts".into()),
        total_restarts - 938.0,
    ); // agent B missed the needle
    println!(
        "  → agent A and agent B disagree; coffer holds the ground truth, so A is right and B is wrong.\n    A single agent computing its OWN number cannot settle which of two disagreeing agents is\n    correct — only a neutral oracle over the shared held bytes can.",
    );
}

fn verdict_str(v: &ReceiptVerdict) -> String {
    match v {
        ReceiptVerdict::Valid => {
            "✅ VALID (re-derived the attested answer byte-identically)".to_string()
        }
        ReceiptVerdict::ValueMismatch { expected, actual } => {
            format!("❌ VALUE_MISMATCH (receipt attests {expected}, data now yields {actual})")
        }
        ReceiptVerdict::BackingTampered => {
            "❌ BACKING_TAMPERED (value held but the backing rows changed)".to_string()
        }
        ReceiptVerdict::Refused => "⚠ REFUSED (query no longer runs over this input)".to_string(),
        ReceiptVerdict::MalformedReceipt => "⚠ MALFORMED_RECEIPT".to_string(),
    }
}
