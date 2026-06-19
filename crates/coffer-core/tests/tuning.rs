//! Compression tuning harness + locked baseline.
//!
//! These are the regression gates for the tuning program: the stable properties every tuning step
//! must preserve — compressible classes (JSON arrays, logs) shrink and stay byte-exact reversible,
//! while content the model reads whole (prose/code) is passed through verbatim (the deliberate
//! restraint, not a miss). Individual tuning steps ADD assertions here (e.g. T-FIDELITY's head+tail
//! survival); they must not weaken these.
//!
//! Manual measurement across a wider real corpus lives in `examples/filter.rs`.

use coffer_cas::MemoryCas;
use coffer_core::{Budget, Compressor};
use coffer_tokenizer::{HeuristicCounter, TokenCounter};

/// Reduction budget used across the baseline (aim to cut 80%).
const FRAC: f32 = 0.8;

fn savings_pct(raw: &[u8], rendered: &str, counter: &dyn TokenCounter) -> f64 {
    let r = counter.count(&String::from_utf8_lossy(raw));
    let o = counter.count(rendered);
    if r == 0 {
        0.0
    } else {
        100.0 * (r as f64 - o as f64) / r as f64
    }
}

#[test]
fn json_array_compresses_and_round_trips() {
    let items: Vec<String> = (0..200)
        .map(|i| format!(r#"{{"id":{i},"sub":"drivers","files":{}}}"#, i % 97))
        .collect();
    let input = format!("[{}]", items.join(",")).into_bytes();

    let cas = MemoryCas::new();
    let counter = HeuristicCounter;
    let doc = Compressor::new()
        .budget(Budget::Reduction(FRAC))
        .counter(&counter)
        .compress(&input, &cas)
        .unwrap();

    assert_eq!(
        doc.reconstruct(&cas).unwrap(),
        input,
        "byte-exact reversible"
    );
    let saved = savings_pct(&input, &doc.render_for_model(), &counter);
    assert!(
        saved > 50.0,
        "JSON array should compress substantially, got {saved:.1}%"
    );
}

#[test]
fn log_compresses_and_round_trips() {
    let input = "2026-06-07 12:00:00 INFO request handled ok\n"
        .repeat(40)
        .into_bytes();

    let cas = MemoryCas::new();
    let counter = HeuristicCounter;
    let doc = Compressor::new()
        .budget(Budget::Reduction(FRAC))
        .counter(&counter)
        .compress(&input, &cas)
        .unwrap();

    assert_eq!(
        doc.reconstruct(&cas).unwrap(),
        input,
        "byte-exact reversible"
    );
    let saved = savings_pct(&input, &doc.render_for_model(), &counter);
    assert!(
        saved > 50.0,
        "a repetitive log should compress substantially, got {saved:.1}%"
    );
}

#[test]
fn window_keeps_head_and_tail() {
    // T-FIDELITY: a tight budget keeps BOTH ends, not just the leading prefix — so the
    // tail (errors, summaries, most-recent rows) survives. The middle run is offloaded reversibly.
    let items: Vec<String> = (0..60).map(|i| format!(r#"{{"id":{i}}}"#)).collect();
    let input = format!("[{}]", items.join(",")).into_bytes();

    let cas = MemoryCas::new();
    let counter = HeuristicCounter;
    let doc = Compressor::new()
        .budget(Budget::Reduction(0.7))
        .counter(&counter)
        .compress(&input, &cas)
        .unwrap();

    assert_eq!(
        doc.reconstruct(&cas).unwrap(),
        input,
        "byte-exact reversible"
    );
    let render = doc.render_for_model();
    assert!(render.contains(r#""id":0"#), "head survives: {render}");
    assert!(
        render.contains(r#""id":59"#),
        "tail survives (the leading-keep regression): {render}"
    );
    assert!(
        render.contains("cof:"),
        "the middle run is offloaded to a sentinel: {render}"
    );
}

#[test]
fn build_log_compresses_and_round_trips() {
    // T-COVERAGE: repetitive build/test output (low first-token diversity) now compresses
    // — it previously classified as Text (0%). Code/prose still pass through (other tests).
    let mut s = String::new();
    for i in 0..120 {
        s.push_str(&format!("   Compiling crate_{i} v0.1.{i}\n"));
    }
    s.push_str("    Finished `dev` profile [unoptimized] target(s) in 3.2s\n");
    let input = s.into_bytes();

    let cas = MemoryCas::new();
    let counter = HeuristicCounter;
    let doc = Compressor::new()
        .budget(Budget::Reduction(FRAC))
        .counter(&counter)
        .compress(&input, &cas)
        .unwrap();

    assert_eq!(
        doc.reconstruct(&cas).unwrap(),
        input,
        "byte-exact reversible"
    );
    let saved = savings_pct(&input, &doc.render_for_model(), &counter);
    assert!(
        saved > 50.0,
        "repetitive build output should compress, got {saved:.1}%"
    );
    assert!(
        doc.render_for_model().contains("Finished"),
        "the tail summary survives the window"
    );
}

#[test]
fn verbose_git_status_compresses_and_round_trips() {
    // The long `git status` form has many distinct path-leading tokens under the untracked section,
    // so the generic low-first-token-diversity log detector does not catch it. It is still
    // machine-generated status output and should use the reversible log window instead of passing
    // thousands of tokens through verbatim.
    let mut s = String::from(
        "On branch main\n\
         Changes not staged for commit:\n\
           (use \"git add <file>...\" to update what will be committed)\n\
           (use \"git restore <file>...\" to discard changes in working directory)\n",
    );
    for i in 0..60 {
        s.push_str(&format!("\tmodified:   crates/example_{i}/src/lib.rs\n"));
    }
    s.push_str(
        "\n\
         Untracked files:\n\
           (use \"git add <file>...\" to include in what will be committed)\n",
    );
    for i in 0..20 {
        s.push_str(&format!("\tdocs/generated-{i}.md\n"));
    }
    s.push_str("\nno changes added to commit (use \"git add\" and/or \"git commit -a\")\n");
    let input = s.into_bytes();

    let cas = MemoryCas::new();
    let counter = HeuristicCounter;
    let doc = Compressor::new()
        .budget(Budget::Reduction(FRAC))
        .counter(&counter)
        .compress(&input, &cas)
        .unwrap();

    assert_eq!(
        doc.reconstruct(&cas).unwrap(),
        input,
        "byte-exact reversible"
    );
    let render = doc.render_for_model();
    let saved = savings_pct(&input, &render, &counter);
    assert!(
        saved > 50.0,
        "verbose git status should compress substantially, got {saved:.1}%"
    );
    assert!(
        render.contains("On branch main"),
        "the head status context survives: {render}"
    );
    assert!(
        render.contains("no changes added"),
        "the tail summary survives: {render}"
    );
}

#[test]
fn budget_dedups_logs_so_distinct_middle_events_survive() {
    // T-DEDUP ∘ T-FIDELITY: in a duplicate-heavy log, dedup runs FIRST so the budget is
    // spent on distinct lines — distinct events in the MIDDLE survive a duplicate flood that a pure
    // positional head+tail window would bury under repeated heartbeats.
    let mut s = String::new();
    for _ in 0..50 {
        s.push_str("heartbeat ping ok\n");
    }
    s.push_str("EVENT_X the important middle thing\n");
    for _ in 0..50 {
        s.push_str("heartbeat ping ok\n");
    }
    s.push_str("EVENT_Y another middle thing\n");
    for _ in 0..50 {
        s.push_str("heartbeat ping ok\n");
    }
    let input = s.into_bytes();

    let cas = MemoryCas::new();
    let counter = HeuristicCounter;
    // aggressive budget: a pure positional window would keep only head/tail heartbeats.
    let doc = Compressor::new()
        .budget(Budget::Reduction(0.9))
        .counter(&counter)
        .compress(&input, &cas)
        .unwrap();

    assert_eq!(
        doc.reconstruct(&cas).unwrap(),
        input,
        "byte-exact reversible"
    );
    let render = doc.render_for_model();
    assert!(
        render.contains("EVENT_X"),
        "dedup preserves the distinct middle event: {render}"
    );
    assert!(
        render.contains("EVENT_Y"),
        "and the second one a positional window would bury: {render}"
    );
}

#[test]
fn prose_passes_through_verbatim() {
    // Content the model reads whole: a blind transparent layer must NOT compress it (compressing it
    // would only force retrieve round-trips). Restraint is the correct behavior here.
    let input =
        b"The quick brown fox jumps over the lazy dog. This is ordinary prose, not a log or a JSON \
          array, so the engine should leave it exactly as-is rather than offload any of it."
            .to_vec();

    let cas = MemoryCas::new();
    let counter = HeuristicCounter;
    let doc = Compressor::new()
        .budget(Budget::Reduction(FRAC))
        .counter(&counter)
        .compress(&input, &cas)
        .unwrap();

    assert_eq!(
        doc.reconstruct(&cas).unwrap(),
        input,
        "byte-exact reversible"
    );
    assert_eq!(
        doc.render_for_model().as_bytes(),
        &input[..],
        "prose is passed through verbatim"
    );
}
