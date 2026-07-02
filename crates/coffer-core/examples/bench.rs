//! Repeatable perf baseline for the engine's hot paths (NOT a correctness test).
//!
//! Run (RELEASE, required for meaningful numbers):
//!   cargo run --release -p coffer-core --example bench
//!
//! Deterministic inputs (SplitMix64, fixed seed), medians over repeated calls, `black_box`
//! against dead-code elimination. Covers the paths a surface hits per request: detect,
//! Stage-0 compress, budget search (heuristic counter), reconstruct, and the read-side
//! query family (describe / digest / query_aggregate) that MCP/wrap answer tools with.

use std::hint::black_box;
use std::time::Instant;

use coffer_cas::MemoryCas;
use coffer_core::{
    Agg, Budget, Dataset, Op, Predicate, compress_to_budget, describe, detect, digest,
    query_aggregate,
};
use coffer_tokenizer::HeuristicCounter;
use serde_json::{Value, json};

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// kubectl-shaped JSON array, ~55-70 bytes per row.
fn make_json(rows: usize) -> Vec<u8> {
    let mut s: u64 = 42;
    let arr: Vec<Value> = (0..rows)
        .map(|i| {
            let r = splitmix(&mut s);
            json!({
                "name": format!("pod-{i:06}"),
                "status": if r % 10 == 0 { "Error" } else { "Running" },
                "restarts": r % 7,
                "cpu_m": r % 4000,
            })
        })
        .collect();
    serde_json::to_vec(&Value::Array(arr)).unwrap()
}

/// Timestamped log lines, ~80 bytes per line.
fn make_log(lines: usize) -> Vec<u8> {
    let mut s: u64 = 7;
    let mut out = String::new();
    for i in 0..lines {
        let r = splitmix(&mut s);
        out.push_str(&format!(
            "2026-07-02T08:{:02}:{:02}Z {} worker-{:03} request completed in {}ms code={}\n",
            (i / 60) % 60,
            i % 60,
            if r % 50 == 0 { "ERROR" } else { "INFO" },
            r % 100,
            r % 900,
            if r % 50 == 0 { 500 } else { 200 },
        ));
    }
    out.into_bytes()
}

fn median_us(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// Median microseconds of `f` over adaptive iterations (fewer for slow calls).
fn bench<T>(label: &str, iters: usize, mut f: impl FnMut() -> T) -> f64 {
    for _ in 0..(iters / 5).max(2) {
        black_box(f());
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        black_box(f());
        samples.push(t.elapsed().as_nanos() as f64 / 1000.0);
    }
    let us = median_us(samples);
    println!("{label:<58} {us:>12.1} us");
    us
}

fn main() {
    let json_5k = make_json(5_000);
    let json_40k = make_json(40_000);
    let log_100k = make_log(100_000);
    println!(
        "inputs: json_5k={} KiB, json_40k={} KiB, log_100k={} KiB\n",
        json_5k.len() / 1024,
        json_40k.len() / 1024,
        log_100k.len() / 1024
    );

    // --- detect ---
    bench("detect(json_40k)", 50, || detect(&json_40k));
    bench("detect(log_100k)", 50, || detect(&log_100k));

    // --- Stage-0 compress (MaxReduction) + reconstruct ---
    let cas = MemoryCas::new();
    bench("compress MaxReduction (json_40k)", 30, || {
        compress_to_budget(&json_40k, &cas, Budget::MaxReduction, &HeuristicCounter)
    });
    let doc = compress_to_budget(&json_40k, &cas, Budget::MaxReduction, &HeuristicCounter);
    bench("reconstruct (json_40k)", 30, || doc.reconstruct(&cas));

    // --- budget search (heuristic char-model fast path) ---
    bench("compress_to_budget 90% (json_40k)", 30, || {
        compress_to_budget(&json_40k, &cas, Budget::Reduction(0.9), &HeuristicCounter)
    });
    bench("compress_to_budget 50% (json_40k)", 30, || {
        compress_to_budget(&json_40k, &cas, Budget::Reduction(0.5), &HeuristicCounter)
    });
    bench("compress_to_budget 90% (log_100k)", 30, || {
        compress_to_budget(&log_100k, &cas, Budget::Reduction(0.9), &HeuristicCounter)
    });

    // --- read-side query family (what MCP / wrap answer tools with) ---
    let pred = [Predicate {
        field: "status".into(),
        op: Op::Eq,
        value: json!("Error"),
    }];
    bench("describe (json_5k)", 50, || describe(&json_5k));
    bench("describe (json_40k)", 20, || describe(&json_40k));
    bench("digest max restarts (json_40k)", 20, || {
        digest(&json_40k, "max restarts")
    });
    bench("query_aggregate count Error (json_5k)", 50, || {
        query_aggregate(&json_5k, &pred, &Agg::Count)
    });
    bench("query_aggregate count Error (json_40k)", 20, || {
        query_aggregate(&json_40k, &pred, &Agg::Count)
    });
    bench("query_aggregate sum restarts Error (json_40k)", 20, || {
        query_aggregate(&json_40k, &pred, &Agg::Sum("restarts".into()))
    });

    // --- Dataset: parse once per handle, then query repeatedly (the surface cache path) ---
    println!();
    bench("Dataset::parse (json_40k)", 20, || {
        Dataset::parse(&json_40k)
    });
    let ds = Dataset::parse(&json_40k).unwrap();
    bench(
        "Dataset aggregate count Error, cold columns (json_40k)",
        20,
        || {
            // fresh Dataset each call: pays the lazy column build, not the parse
            let d = Dataset::from_rows(serde_json::from_slice::<Vec<Value>>(&json_40k).unwrap());
            d.query_aggregate(&pred, &Agg::Count)
        },
    );
    bench(
        "Dataset aggregate count Error, warm (json_40k)",
        200,
        || ds.query_aggregate(&pred, &Agg::Count),
    );
    bench(
        "Dataset aggregate sum restarts Error, warm (json_40k)",
        200,
        || ds.query_aggregate(&pred, &Agg::Sum("restarts".into())),
    );
    bench("Dataset digest max restarts, warm (json_40k)", 20, || {
        ds.digest("max restarts")
    });
    // The one-time stats pass a NEW handle pays for describe, parse excluded (single-shot
    // timings: the memo makes repeated calls on one dataset meaningless to sample).
    {
        let mut singles = Vec::new();
        for _ in 0..5 {
            let d = Dataset::from_rows(serde_json::from_slice::<Vec<Value>>(&json_40k).unwrap());
            let t = Instant::now();
            black_box(d.describe());
            singles.push(t.elapsed().as_nanos() as f64 / 1000.0);
        }
        println!(
            "{:<58} {:>12.1} us",
            "Dataset describe, cold stats pass (json_40k)",
            median_us(singles)
        );
    }
    bench("Dataset describe, warm (json_40k)", 20, || ds.describe());
}
