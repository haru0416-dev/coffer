//! Reproducible, API-key-free benchmark for coffer's two mechanical properties.
//!
//! It answers, deterministically and with NO model call, the question the README's "Honesty, up
//! front" section commits to on the dimension coffer actually wins: **at the same token budget, does
//! computing an aggregate over the offloaded bytes beat reading a head/tail-truncated window?**
//!
//! What it shows, at each compression level over one synthetic-but-realistic `kubectl`-shaped dump:
//!   1. compression % counted with the model's OWN tokenizer (`OpenAI` `o200k_base`, exact offline),
//!   2. a byte-exact `reconstruct(compress(x)) == x` check (the Stage-0 invariant) — at EVERY level,
//!   3. coffer's exact answer to four aggregation questions, asserted against ground truth computed
//!      independently in this file (so the harness also self-checks coffer's exactness),
//!   4. the SAME questions answered by an idealized head/tail truncation baseline at the SAME token
//!      budget — modeled GENEROUSLY as a perfect aggregate over exactly the rows the window shows
//!      (a real LLM sees no more rows and is worse at arithmetic, so this is an upper bound for it),
//!   5. the retrieval round-trip token cost of coffer's exact answer (a short line, ~constant in N).
//!
//! It is honest about where coffer does NOT win: the final in-window-retrieval check is a question
//! whose answer sits in the kept head, where truncation that keeps that row answers correctly too —
//! a tie, reported as a tie.
//!
//! Run:
//!   cargo run --release -p coffer-eval            # defaults: 5000 rows, fixed seed
//!   cargo run --release -p coffer-eval -- 20000 7 # rows seed
//!
//! Determinism: a fixed-seed `SplitMix64` generator and a published, frozen tokenizer vocabulary mean
//! the printed numbers reproduce byte-for-byte on any machine.

#![warn(clippy::pedantic)]
// Cast lints: percentages/means over row counts and rng-derived bounded values —
// all far below any lossy threshold (same rationale as coffer-core).
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use coffer_cas::MemoryCas;
use coffer_core::{Agg, Budget, Op, Predicate, compress_to_budget, query_aggregate};
use coffer_tokenizer::{HeuristicCounter, TiktokenCounter, TokenCounter};
use serde_json::{Value, json};

/// Deterministic `SplitMix64` — same sequence on every machine, no external dependency.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Realistic pod-phase mix: most pods are healthy, a long tail is broken. The skew matters — a
/// uniform mix would make "count the broken pods" trivial; here the broken ones are a minority that
/// a head/tail window mostly misses.
const STATUSES: [(&str, u32); 5] = [
    ("Running", 70),
    ("Completed", 12),
    ("Pending", 9),
    ("CrashLoopBackOff", 6),
    ("Error", 3),
];

fn pick_status(rng: &mut Rng) -> &'static str {
    let roll = rng.below(100) as u32;
    let mut acc = 0;
    for (name, weight) in STATUSES {
        acc += weight;
        if roll < acc {
            return name;
        }
    }
    "Running"
}

/// Build a `kubectl get pods -o json`-shaped array of `n` records. One extreme restarts outlier is
/// injected at the midpoint: it is the global argmax, and a head/tail window never contains the
/// middle, so it is exactly the "needle buried in a big dump" that truncation loses and an exact
/// digest keeps. Returns the records (for independent ground-truth) and their compact JSON bytes
/// (what coffer holds).
fn make_dump(n: usize, seed: u64) -> (Vec<Value>, Vec<u8>) {
    let mut rng = Rng(seed ^ 0x1234_5678_9ABC_DEF0);
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let status = pick_status(&mut rng);
        // Healthy pods rarely restart; broken pods restart a handful of times.
        let restarts = match status {
            "CrashLoopBackOff" => 3 + rng.below(28) as i64,
            "Error" => 1 + rng.below(12) as i64,
            _ => rng.below(3) as i64,
        };
        let cpu_m = 5 + rng.below(900) as i64;
        let mem_mib = 16 + rng.below(2048) as i64;
        let namespace = ["default", "kube-system", "prod", "staging"][rng.below(4) as usize];
        rows.push(json!({
            "name": format!("svc-{:05}-{:04x}", i, rng.below(0x1_0000)),
            "namespace": namespace,
            "status": status,
            "restarts": restarts,
            "cpu_m": cpu_m,
            "mem_mib": mem_mib,
            "node": format!("node-{:02}", rng.below(24)),
            "ready": status == "Running" || status == "Completed",
        }));
    }
    // The buried needle: one pod crash-looping far beyond any other, at the midpoint.
    if n > 0 {
        let mid = n / 2;
        rows[mid]["status"] = json!("CrashLoopBackOff");
        rows[mid]["restarts"] = json!(99_999);
        rows[mid]["ready"] = json!(false);
    }
    let bytes = serde_json::to_vec(&Value::Array(rows.clone())).expect("serialize dump");
    (rows, bytes)
}

/// A `field == "value"` string predicate (the common selector).
fn eq(field: &str, value: &str) -> Predicate {
    Predicate {
        field: field.into(),
        op: Op::Eq,
        value: json!(value),
    }
}

/// Numeric helper: read an integer field, or 0 if absent/non-numeric (the synthetic data is clean,
/// so this only guards against a typo, never silently drops real rows the way a fuzzy reader would).
fn num(rec: &Value, field: &str) -> f64 {
    rec.get(field).and_then(Value::as_f64).unwrap_or(0.0)
}

/// Relative error in percent of `got` against `truth` (absolute difference over |truth|).
fn rel_err_pct(got: f64, truth: f64) -> f64 {
    if truth == 0.0 {
        if got == 0.0 { 0.0 } else { 100.0 }
    } else {
        100.0 * (got - truth).abs() / truth.abs()
    }
}

/// The set of record indices a head/tail truncation keeps within `budget_tok` tokens. Records are
/// added alternately from the front and the back (a balanced "keep the start AND the end" window),
/// each costing its own compact-JSON token count plus one for the separator, until the next record
/// would overflow the budget. This is the payload a same-budget `head`+`tail` truncation would feed
/// a model — and we then let that model be PERFECT over it, which only helps the baseline.
fn truncation_window(per_row_tok: &[usize], budget_tok: usize) -> Vec<usize> {
    let n = per_row_tok.len();
    let mut visible = Vec::new();
    let mut used = 2; // the enclosing `[` `]`
    let (mut lo, mut hi) = (0usize, n);
    let mut take_front = true;
    while lo < hi {
        let idx = if take_front { lo } else { hi - 1 };
        let cost = per_row_tok[idx] + 1; // record + one separator
        if used + cost > budget_tok {
            break;
        }
        used += cost;
        visible.push(idx);
        if take_front {
            lo += 1;
        } else {
            hi -= 1;
        }
        take_front = !take_front;
    }
    visible
}

// A linear benchmark script: setup, five budget levels, four questions, one table —
// splitting it would hide the protocol it exists to make readable.
#[allow(clippy::too_many_lines)]
fn main() {
    let mut args = std::env::args().skip(1);
    let n: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(5_000);
    let seed: u64 = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0x_C0FF_EE42);

    let (rows, bytes) = make_dump(n, seed);
    let cas = MemoryCas::new();
    // Budget search runs the FAST char-linear heuristic (a real BPE re-encode per probe would be
    // orders of magnitude slower); every HEADLINE number below is then counted once with the real
    // o200k_base encoder, so the reported tokens are the model's own, not the search estimate.
    let search = HeuristicCounter;
    let tok = TiktokenCounter::o200k();

    let raw_text = String::from_utf8_lossy(&bytes);
    let raw_tok = tok.count(&raw_text);

    // ---- ground truth, computed here independently of coffer ----
    // Four deliberately different shapes so neither column is "juiced": a minority-class COUNT, a
    // SUM over a class that does NOT contain the injected needle (so its error is pure sampling
    // loss, not one outlier), a sampling-robust MEAN, and the buried-needle MAX. The needle's effect
    // is isolated to the MAX/argmax column on purpose.
    let truth_count = rows
        .iter()
        .filter(|r| r["status"] == json!("CrashLoopBackOff"))
        .count() as f64;
    let truth_sum: f64 = rows
        .iter()
        .filter(|r| r["status"] == json!("Error"))
        .map(|r| num(r, "restarts"))
        .sum();
    let running: Vec<&Value> = rows
        .iter()
        .filter(|r| r["status"] == json!("Running"))
        .collect();
    let truth_mean = if running.is_empty() {
        0.0
    } else {
        running.iter().map(|r| num(r, "cpu_m")).sum::<f64>() / running.len() as f64
    };
    let truth_max = rows
        .iter()
        .map(|r| num(r, "restarts"))
        .fold(f64::MIN, f64::max);

    // ---- coffer's exact answers (computed over ALL bytes, as the MCP digest does over the CAS) ----
    let c_count = query_aggregate(&bytes, &[eq("status", "CrashLoopBackOff")], &Agg::Count)
        .expect("count aggregate");
    let c_sum = query_aggregate(
        &bytes,
        &[eq("status", "Error")],
        &Agg::Sum("restarts".into()),
    )
    .expect("sum aggregate");
    let c_mean = query_aggregate(
        &bytes,
        &[eq("status", "Running")],
        &Agg::Mean("cpu_m".into()),
    )
    .expect("mean aggregate");
    let c_max = query_aggregate(
        &bytes,
        &[Predicate {
            field: "restarts".into(),
            op: Op::Ge,
            value: json!(0),
        }],
        &Agg::Max("restarts".into()),
    )
    .expect("max aggregate");

    // Self-check: coffer must equal the independently-computed truth, exactly. If this ever fails it
    // is a real coffer bug, not a benchmark artifact — so it is an assert, not a printed row.
    assert!(
        rel_err_pct(c_count.value, truth_count) < 1e-9,
        "coffer count != truth"
    );
    assert!(
        rel_err_pct(c_sum.value, truth_sum) < 1e-9,
        "coffer sum != truth"
    );
    assert!(
        rel_err_pct(c_mean.value, truth_mean) < 1e-6,
        "coffer mean != truth"
    );
    assert!(
        (c_max.value - truth_max).abs() < 1e-9,
        "coffer max != truth"
    );

    // The retrieval round-trip cost of an exact answer: the model reads back this one line, whose
    // size does not grow with the dump. We report the count answer's line as the representative.
    let answer_tok = tok.count(&c_count.display);

    // Per-row token costs, measured once with the real tokenizer, for the truncation window model.
    let per_row_tok: Vec<usize> = rows
        .iter()
        .map(|r| tok.count(&serde_json::to_string(r).expect("row json")))
        .collect();

    println!("# coffer-eval — exact compute-digest vs head/tail truncation\n");
    println!(
        "dataset: {n} kubectl-shaped pod records · {} raw bytes · {raw_tok} tokens ({}) · seed {seed:#x}",
        bytes.len(),
        tok.model_label(),
    );
    println!(
        "questions (ground truth over full data): **count** CrashLoopBackOff pods = {truth_count:.0} · **sum** restarts over Error pods = {truth_sum:.0} · **mean** cpu_m over Running pods = {truth_mean:.1} · **max** restarts = {truth_max:.0} (the buried needle)\n",
    );

    // ---- Table 1: the two mechanical invariants across compression levels ----
    println!("## Mechanical invariants (counted with the model's own tokenizer)\n");
    println!("| target | kept tokens | compression | byte-exact round-trip |");
    println!("|-------:|------------:|------------:|:---------------------:|");

    let fracs = [0.5_f64, 0.25, 0.10, 0.05, 0.02];
    // Capture per-level facts for table 2 so each level is compressed exactly once.
    let mut levels: Vec<(f64, usize, Vec<usize>)> = Vec::new();
    for &f in &fracs {
        let target = ((raw_tok as f64) * f).round() as usize;
        let doc = compress_to_budget(&bytes, &cas, Budget::Tokens(target), &search);
        let render = doc.render_for_model();
        let kept = tok.count(&render);
        let recovered = doc.reconstruct(&cas).expect("reconstruct");
        let exact = recovered == bytes;
        let compression = 100.0 * (1.0 - kept as f64 / raw_tok as f64);
        let visible = truncation_window(&per_row_tok, kept);
        println!(
            "| {:.0}% | {kept} | {compression:.1}% | {} |",
            f * 100.0,
            if exact { "✅ pass" } else { "❌ FAIL" },
        );
        assert!(
            exact,
            "byte-exact reversibility violated at target {target}"
        );
        levels.push((compression, visible.len(), visible));
    }

    // ---- Table 2: the decisive test — same token budget, exact digest vs ideal truncation ----
    println!(
        "\n## Same-budget answer quality: exact digest vs ideal head/tail truncation\n\nFor every row, **coffer's error is 0.00%** — its answers are computed over all bytes (including the offloaded ones) and asserted against independently-computed ground truth. The columns below are the *truncation* baseline's error at the same token budget, modeled generously as a perfect aggregate over exactly the rows its window shows.\n",
    );
    println!(
        "| compression | rows truncation sees | count err | sum err | mean err | max (argmax) | exact-answer tokens |",
    );
    println!(
        "|------------:|---------------------:|----------:|--------:|---------:|:------------:|--------------------:|",
    );
    for (compression, seen, visible) in &levels {
        let t_count = visible
            .iter()
            .filter(|&&i| rows[i]["status"] == json!("CrashLoopBackOff"))
            .count() as f64;
        let t_sum: f64 = visible
            .iter()
            .filter(|&&i| rows[i]["status"] == json!("Error"))
            .map(|&i| num(&rows[i], "restarts"))
            .sum();
        let t_running: Vec<usize> = visible
            .iter()
            .copied()
            .filter(|&i| rows[i]["status"] == json!("Running"))
            .collect();
        let t_mean = if t_running.is_empty() {
            0.0
        } else {
            t_running
                .iter()
                .map(|&i| num(&rows[i], "cpu_m"))
                .sum::<f64>()
                / t_running.len() as f64
        };
        let t_max = visible
            .iter()
            .map(|&i| num(&rows[i], "restarts"))
            .fold(f64::MIN, f64::max);
        let max_ok = (t_max - truth_max).abs() < 1e-9;
        println!(
            "| {compression:.1}% | {seen} / {n} | {:.1}% | {:.1}% | {:.1}% | {} | {answer_tok} |",
            rel_err_pct(t_count, truth_count),
            rel_err_pct(t_sum, truth_sum),
            rel_err_pct(t_mean, truth_mean),
            if max_ok { "✅ found" } else { "❌ missed" },
        );
    }

    // ---- The honest tie: a question whose answer is in the kept head ----
    // Pod 0 is always in the head window, so a head/tail truncation that keeps it answers a
    // single-record lookup correctly — exactly as coffer does. We report this as a tie, not a win.
    let head_name = rows[0]["name"].as_str().unwrap_or("");
    let aggressive = levels.last().expect("at least one level");
    let head_visible = aggressive.2.contains(&0);
    println!("\n## Where coffer does NOT win (reported, not hidden)\n");
    println!(
        "- **In-window retrieval** — \"what is the status of pod `{head_name}`?\" (record 0, in the kept head). At the most aggressive {:.1}% level the truncation window still {} this row, so it answers correctly — a **tie** with coffer, not a win. Compressing input the model's context already fits does not beat feeding it raw.",
        aggressive.0,
        if head_visible {
            "contains"
        } else {
            "would need"
        },
    );
    println!(
        "- **End-task accuracy with a real model** is a separate, still-open question, measured elsewhere against this same protocol — this harness proves the two *mechanical* properties (byte-exact round-trip and exact aggregation), not that an LLM answers better.",
    );
    println!(
        "\n_Reproduce: `cargo run --release -p coffer-eval`. Determinism: fixed-seed SplitMix64 data + the frozen o200k_base vocabulary._",
    );
}

#[cfg(test)]
mod tests {
    use super::{
        Agg, Budget, HeuristicCounter, MemoryCas, Op, Predicate, TiktokenCounter, TokenCounter,
        compress_to_budget, eq, json, make_dump, num, query_aggregate, truncation_window,
    };

    /// CI smoke test of the harness's three load-bearing claims on a small deterministic dump:
    /// (1) coffer's digest equals independently-computed ground truth, (2) `reconstruct` is
    /// byte-exact at several budgets, and (3) head/tail truncation genuinely MISSES the buried
    /// needle — so the benchmark's win is not an artifact of a window that happens to include it.
    #[test]
    fn digest_is_exact_roundtrip_holds_and_truncation_misses_the_needle() {
        let (rows, bytes) = make_dump(400, 0xABCD);
        let cas = MemoryCas::new();
        let search = HeuristicCounter;
        let tok = TiktokenCounter::o200k();

        let truth_count = rows
            .iter()
            .filter(|r| r["status"] == json!("CrashLoopBackOff"))
            .count() as f64;
        let truth_max = rows
            .iter()
            .map(|r| num(r, "restarts"))
            .fold(f64::MIN, f64::max);
        // Exact literal comparison intended: the needle is an integer-valued f64.
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(truth_max, 99_999.0, "the buried needle must be present");
        }

        // (1) coffer is exact.
        let c = query_aggregate(&bytes, &[eq("status", "CrashLoopBackOff")], &Agg::Count)
            .expect("count");
        assert!((c.value - truth_count).abs() < 1e-9);
        let m = query_aggregate(
            &bytes,
            &[Predicate {
                field: "restarts".into(),
                op: Op::Ge,
                value: json!(0),
            }],
            &Agg::Max("restarts".into()),
        )
        .expect("max");
        assert!((m.value - truth_max).abs() < 1e-9);

        // (2) byte-exact reversibility at several budgets.
        let raw_tok = tok.count(&String::from_utf8_lossy(&bytes));
        for f in [0.5_f64, 0.1, 0.02] {
            let target = ((raw_tok as f64) * f) as usize;
            let doc = compress_to_budget(&bytes, &cas, Budget::Tokens(target), &search);
            assert_eq!(
                doc.reconstruct(&cas).expect("reconstruct"),
                bytes,
                "byte-exact at {f}"
            );
        }

        // (3) truncation misses the needle at an aggressive budget.
        let per_row: Vec<usize> = rows
            .iter()
            .map(|r| tok.count(&serde_json::to_string(r).expect("row")))
            .collect();
        let visible = truncation_window(&per_row, raw_tok / 20);
        let t_max = visible
            .iter()
            .map(|&i| num(&rows[i], "restarts"))
            .fold(f64::MIN, f64::max);
        assert!(
            t_max < truth_max,
            "head/tail truncation must miss the buried needle"
        );
    }
}
