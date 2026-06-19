//! Kill-probe benchmark (NOT a correctness test; `#[ignore]` so CI never runs it).
//!
//! QUESTION: for REPEATED typed queries over the SAME held JSON-array handle, does a per-blob
//! COLUMNAR sidecar (parse once into typed `Vec`s, then scan the columns with no JSON parse) beat
//! the current full-parse-per-call behavior of [`coffer_core::query_aggregate`]?
//!
//! BASELINE per call == exactly today's behavior: `query_aggregate(blob, [a > THRESH], Sum("b"))`,
//! which does `serde_json::from_slice(blob)` (full parse of the whole blob) + filter + aggregate
//! on EVERY call. We call the real public function, so the baseline is not strawmanned.
//!
//! SIDECAR: parse the blob ONCE into `{ a: Vec<i64>, b: Vec<i64>, status: Vec<u8> }`, then answer
//! the SAME query (sum b where a > THRESH) by scanning those `Vec`s — no serde parse per call. The
//! one-time build cost is measured separately from the per-call cost.
//!
//! BREAKEVEN: `sidecar_build_us / (baseline_per_call_us - sidecar_per_call_us)` = the number of
//! re-queries of the same handle after which paying the build cost once is net cheaper.
//!
//! Run (RELEASE, required for meaningful numbers):
//!   cargo test -p coffer-core --release --test sidecar_bench -- --ignored --nocapture
//!
//! Honest scope: this measures IN-MEMORY cost only. A real SqliteCas deployment must STORE the
//! sidecar blob and LOAD it back to reuse it across requests/processes — serialization + a CAS
//! round-trip per re-query that is NOT modeled here. Breakeven also assumes the same handle is
//! actually re-queried; a one-shot query never amortizes the build.

use std::hint::black_box;
use std::time::Instant;

use coffer_core::{Agg, Op, Predicate, query_aggregate};
use serde_json::{Value, json};

/// Status code, kept as a `u8` column to mirror a realistic typed-categorical sidecar encoding.
const STATUS: [&str; 3] = ["ok", "warn", "error"];

/// `a` is a value the predicate filters on; `THRESH` keeps roughly half the rows so the filter is
/// neither trivially empty nor trivially universal (a realistic selectivity).
const A_MAX: i64 = 1000;
const THRESH: i64 = 500;

/// Build a synthetic JSON array of `rows` records:
/// `{"a": i64, "b": i64, "status": "ok"|"warn"|"error", "msg": <~60-char string>}`.
/// Values are deterministic (a cheap LCG) so every size/iteration sees the same blob.
fn make_blob(rows: usize) -> Vec<u8> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        // SplitMix64-ish step — deterministic, decent spread, no external dep.
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut arr: Vec<Value> = Vec::with_capacity(rows);
    for i in 0..rows {
        let a = (next() % (A_MAX as u64)) as i64;
        let b = (next() % 10_000) as i64;
        let status = STATUS[(next() % 3) as usize];
        // ~60-char message to give each row a realistic byte width (not a toy {"a":1} row).
        let msg = format!(
            "record {i:07} event={status} detail=lorem-ipsum-dolor-sit-amet-{:04}",
            next() % 10_000
        );
        arr.push(json!({ "a": a, "b": b, "status": status, "msg": msg }));
    }
    serde_json::to_vec(&Value::Array(arr)).expect("serialize blob")
}

/// The columnar sidecar: typed `Vec`s for the fields a query touches, plus the row count. Built by
/// parsing the blob exactly ONCE. `msg` is intentionally NOT columnized — a query that filters on
/// `a` and sums `b` never needs it, which is the whole point of a query-specific sidecar.
struct Sidecar {
    a: Vec<i64>,
    b: Vec<i64>,
    #[allow(dead_code)] // built to mirror a realistic sidecar; unused by this particular query.
    status: Vec<u8>,
}

impl Sidecar {
    /// Parse the blob ONCE into typed columns. This is the cost amortized across re-queries.
    fn build(blob: &[u8]) -> Self {
        let v: Value = serde_json::from_slice(blob).expect("parse blob");
        let arr = v.as_array().expect("array");
        let mut a = Vec::with_capacity(arr.len());
        let mut b = Vec::with_capacity(arr.len());
        let mut status = Vec::with_capacity(arr.len());
        for rec in arr {
            let o = rec.as_object().expect("object");
            a.push(o["a"].as_i64().expect("a is i64"));
            b.push(o["b"].as_i64().expect("b is i64"));
            let s = o["status"].as_str().expect("status is str");
            let code = STATUS.iter().position(|&x| x == s).expect("known status") as u8;
            status.push(code);
        }
        Self { a, b, status }
    }

    /// Answer "sum b where a > THRESH" by scanning the typed columns — NO JSON parse. This is the
    /// per-call cost the baseline's full parse is being raced against.
    fn sum_b_where_a_gt(&self, thresh: i64) -> i64 {
        let mut sum = 0i64;
        for i in 0..self.a.len() {
            if self.a[i] > thresh {
                sum += self.b[i];
            }
        }
        sum
    }
}

/// The CHEAPER alternative to a columnar sidecar: a transparent parsed-`Value` cache. Parse the blob
/// into a `Value` array ONCE (the cache-populate cost on a miss), then answer repeated queries over
/// the cached `&[Value]` with NO `from_slice` per call. This mirrors `query_aggregate`'s post-parse
/// work EXACTLY (build `matched: Vec<usize>`, then `vals: Vec<f64>`, then sum) — minus only the parse
/// and the display string — so it is the per-call cost a Value-cache would pay. Crucially it needs NO
/// typed columns and NO re-implementation of the refuse-on-non-f64 contract: it reuses the same
/// `Value` path the real code already trusts.
fn value_cache_call(arr: &[Value], thresh: i64) -> i64 {
    let matched: Vec<usize> = arr
        .iter()
        .enumerate()
        .filter(|(_, rec)| {
            rec.as_object()
                .and_then(|o| o.get("a"))
                .and_then(Value::as_f64)
                .is_some_and(|a| a > thresh as f64)
        })
        .map(|(i, _)| i)
        .collect();
    let mut vals: Vec<f64> = Vec::new();
    for &i in &matched {
        if let Some(bv) = arr[i]
            .as_object()
            .and_then(|o| o.get("b"))
            .and_then(Value::as_f64)
        {
            vals.push(bv);
        }
    }
    vals.iter().sum::<f64>() as i64
}

/// Median of `Instant` deltas in microseconds (f64). Sorts a copy; sample sizes here are small.
fn median_us(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let n = samples.len();
    if n % 2 == 1 {
        samples[n / 2]
    } else {
        (samples[n / 2 - 1] + samples[n / 2]) / 2.0
    }
}

const WARMUP: usize = 30;
const ITERS: usize = 300;

#[test]
#[ignore = "benchmark, not a CI test; run with --release --ignored --nocapture"]
fn sidecar_vs_full_parse() {
    let sizes = [1_000usize, 5_000, 50_000, 200_000];

    // Accumulator summed into across every measured call, then printed, so the optimizer cannot
    // discard the query work as dead code (also black_box on each result).
    let mut acc: i64 = 0;

    let pred = Predicate {
        field: "a".into(),
        op: Op::Gt,
        value: json!(THRESH),
    };
    let agg = Agg::Sum("b".into());

    println!("\n=== sidecar vs full-parse: sum(b) where a > {THRESH} ===");
    println!("warmup={WARMUP} measured_iters={ITERS} timer=Instant unit=microseconds stat=median");
    println!(
        "{:>8}  {:>10}  {:>14}  {:>17}  {:>15}  {:>12}  {:>13}  {:>13}",
        "rows",
        "blob_KiB",
        "baseline_us",
        "sidecar_build_us",
        "sidecar_call_us",
        "breakeven",
        "valcache_us",
        "valbuild_us"
    );

    for &rows in &sizes {
        let blob = make_blob(rows);
        let blob_kib = blob.len() as f64 / 1024.0;

        // Sanity: the sidecar answer must equal the baseline answer (no strawman, no wrong column).
        let baseline_ref = query_aggregate(&blob, std::slice::from_ref(&pred), &agg)
            .expect("baseline result")
            .value as i64;
        let sidecar_ref = Sidecar::build(&blob).sum_b_where_a_gt(THRESH);
        assert_eq!(
            baseline_ref, sidecar_ref,
            "sidecar and baseline disagree at rows={rows}"
        );

        // ---- BASELINE: full parse + filter + aggregate, every call (today's behavior) ----
        for _ in 0..WARMUP {
            let r = query_aggregate(&blob, std::slice::from_ref(&pred), &agg).unwrap();
            acc = acc.wrapping_add(black_box(r.value as i64));
        }
        let mut baseline_samples = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let t = Instant::now();
            let r = query_aggregate(&blob, std::slice::from_ref(&pred), &agg).unwrap();
            let v = black_box(r.value as i64);
            baseline_samples.push(t.elapsed().as_nanos() as f64 / 1000.0);
            acc = acc.wrapping_add(v);
        }
        let baseline_us = median_us(baseline_samples);

        // ---- SIDECAR BUILD: parse once into columns (the amortized one-time cost) ----
        for _ in 0..WARMUP {
            let sc = Sidecar::build(&blob);
            acc = acc.wrapping_add(black_box(sc.a.len() as i64));
        }
        let mut build_samples = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let t = Instant::now();
            let sc = Sidecar::build(&blob);
            let n = black_box(sc.a.len() as i64);
            build_samples.push(t.elapsed().as_nanos() as f64 / 1000.0);
            acc = acc.wrapping_add(n);
        }
        let build_us = median_us(build_samples);

        // ---- SIDECAR PER-CALL: scan typed columns, no JSON parse ----
        let sc = Sidecar::build(&blob);
        for _ in 0..WARMUP {
            acc = acc.wrapping_add(black_box(sc.sum_b_where_a_gt(THRESH)));
        }
        let mut call_samples = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let t = Instant::now();
            let v = black_box(sc.sum_b_where_a_gt(THRESH));
            call_samples.push(t.elapsed().as_nanos() as f64 / 1000.0);
            acc = acc.wrapping_add(v);
        }
        let call_us = median_us(call_samples);

        let breakeven = build_us / (baseline_us - call_us);

        // ---- VALUE-CACHE: parse once into a Value array, then filter+aggregate over it per call ----
        // valbuild == populate-on-miss cost (just the parse); valcache_us == per-call cost on a HIT.
        let value_ref = {
            let v: Value = serde_json::from_slice(&blob).unwrap();
            value_cache_call(v.as_array().unwrap(), THRESH)
        };
        assert_eq!(
            value_ref, baseline_ref,
            "value-cache and baseline disagree at rows={rows}"
        );
        for _ in 0..WARMUP {
            let v: Value = serde_json::from_slice(&blob).unwrap();
            acc = acc.wrapping_add(black_box(v.as_array().unwrap().len() as i64));
        }
        let mut valbuild_samples = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let t = Instant::now();
            let v: Value = serde_json::from_slice(&blob).unwrap();
            let n = black_box(v.as_array().unwrap().len() as i64);
            valbuild_samples.push(t.elapsed().as_nanos() as f64 / 1000.0);
            acc = acc.wrapping_add(n);
        }
        let valbuild_us = median_us(valbuild_samples);

        let cached: Value = serde_json::from_slice(&blob).unwrap();
        let cached_arr = cached.as_array().unwrap();
        for _ in 0..WARMUP {
            acc = acc.wrapping_add(black_box(value_cache_call(cached_arr, THRESH)));
        }
        let mut valcall_samples = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let t = Instant::now();
            let v = black_box(value_cache_call(cached_arr, THRESH));
            valcall_samples.push(t.elapsed().as_nanos() as f64 / 1000.0);
            acc = acc.wrapping_add(v);
        }
        let valcall_us = median_us(valcall_samples);

        println!(
            "{rows:>8}  {blob_kib:>10.1}  {baseline_us:>14.3}  {build_us:>17.3}  {call_us:>15.4}  {breakeven:>12.2}  {valcall_us:>13.4}  {valbuild_us:>13.3}"
        );
    }

    // Print the accumulator so all the black_box'd work is observably used (defeats DCE).
    println!("\n(acc checksum, ignore value: {acc})");
}
