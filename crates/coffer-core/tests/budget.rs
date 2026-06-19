//! Stage 1 budget-knob gate: `reconstruct(compress_to_budget(x)) == x` at every budget,
//! plus the effectiveness/fairness properties (lands ≤ target, monotone, one coalesced
//! Ref, H5 prefix exactness, p=0 passthrough, object whole-offload, MaxReduction == Stage 0).

use coffer_cas::{Cas, MemoryCas};
use coffer_core::{Budget, Segment, compress, compress_to_budget};
use coffer_tokenizer::{HeuristicCounter, TokenCounter};
use proptest::prelude::*;
use std::cell::Cell;

fn bloated_array(n: usize) -> Vec<u8> {
    let mut s = String::from("[");
    for i in 0..n {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!(
            r#"{{"id":{i},"name":"item-{i}","status":"active"}}"#
        ));
    }
    s.push(']');
    s.into_bytes()
}

fn bloated_log(n: usize) -> Vec<u8> {
    (0..n)
        .map(|i| {
            format!(
                "2026-06-02T00:00:{:02} INFO event number {i} happened\n",
                i % 60
            )
        })
        .collect::<String>()
        .into_bytes()
}

proptest! {
    /// THE Stage-1 reversibility gate: byte-exact at any reduction, for arrays/logs/text.
    #[test]
    fn reconstruct_exact_at_any_budget(
        kind in 0u8..3,
        n in 0usize..40,
        pct in 0u8..=100,
    ) {
        let input = match kind {
            0 => bloated_array(n),
            1 => bloated_log(n),
            _ => format!("just some prose, item {n}, nothing structured").into_bytes(),
        };
        let cas = MemoryCas::new();
        let tok = HeuristicCounter;
        let doc = compress_to_budget(&input, &cas, Budget::Reduction(pct as f32 / 100.0), &tok);
        prop_assert_eq!(doc.reconstruct(&cas).expect("reconstruct"), input);
    }
}

#[test]
fn lands_under_target_and_is_monotone() {
    let input = bloated_array(400);
    let cas = MemoryCas::new();
    let tok = HeuristicCounter;
    let raw = tok.count(&String::from_utf8_lossy(&input));

    let mut prev = usize::MAX;
    for pct in [0u8, 20, 40, 60, 80] {
        let target = (raw as f64 * (1.0 - pct as f64 / 100.0)).round() as usize;
        let doc = compress_to_budget(&input, &cas, Budget::Reduction(pct as f32 / 100.0), &tok);
        let got = tok.count(&doc.render_for_model());
        assert!(
            got <= target,
            "pct {pct}: render {got} tokens exceeded target {target}"
        );
        assert!(
            got <= prev,
            "pct {pct}: not monotone non-increasing ({got} > {prev})"
        );
        assert_eq!(
            doc.reconstruct(&cas).unwrap(),
            input,
            "must stay reversible"
        );
        prev = got;
    }
}

#[test]
fn zero_reduction_is_passthrough() {
    let input = bloated_array(100);
    let cas = MemoryCas::new();
    let tok = HeuristicCounter;
    let doc = compress_to_budget(&input, &cas, Budget::Reduction(0.0), &tok);
    assert!(cas.is_empty(), "0% reduction must not offload anything");
    assert_eq!(doc.segments.len(), 1);
    assert!(matches!(doc.segments[0], Segment::Verbatim(_)));
    assert_eq!(doc.reconstruct(&cas).unwrap(), input);
}

#[test]
fn rounded_noop_reduction_skips_budget_probes() {
    struct CountingOneTokenCounter {
        calls: Cell<usize>,
    }

    impl TokenCounter for CountingOneTokenCounter {
        fn count(&self, _text: &str) -> usize {
            self.calls.set(self.calls.get() + 1);
            1
        }

        fn model_label(&self) -> &'static str {
            "counting-one-token"
        }
    }

    let input = br#"[{"id":1},{"id":2}]"#;
    let cas = MemoryCas::new();
    let tok = CountingOneTokenCounter {
        calls: Cell::new(0),
    };
    let doc = compress_to_budget(input, &cas, Budget::Reduction(0.1), &tok);

    assert_eq!(tok.calls.get(), 1, "raw count only; no budget probes");
    assert!(cas.is_empty(), "no-op reduction must not offload anything");
    assert_eq!(doc.segments.len(), 1);
    assert!(matches!(doc.segments[0], Segment::Verbatim(_)));
    assert_eq!(doc.reconstruct(&cas).unwrap(), input);
}

#[test]
fn non_positive_reduction_skips_token_counting() {
    struct PanicCounter;
    impl TokenCounter for PanicCounter {
        fn count(&self, _text: &str) -> usize {
            panic!("non-positive reduction should not count tokens")
        }

        fn model_label(&self) -> &'static str {
            "panic-counter"
        }
    }

    let input = bloated_array(100);
    for budget in [
        Budget::Reduction(0.0),
        Budget::Reduction(-0.5),
        Budget::Reduction(f32::NAN),
    ] {
        let cas = MemoryCas::new();
        let doc = compress_to_budget(&input, &cas, budget, &PanicCounter);
        assert!(cas.is_empty(), "{budget:?} must not offload anything");
        assert_eq!(doc.segments.len(), 1);
        assert!(matches!(doc.segments[0], Segment::Verbatim(_)));
        assert_eq!(doc.reconstruct(&cas).unwrap(), input);
    }
}

#[test]
fn text_reduction_skips_token_counting() {
    struct PanicCounter;
    impl TokenCounter for PanicCounter {
        fn count(&self, _text: &str) -> usize {
            panic!("text inputs cannot be budget-compressed and should not count tokens")
        }

        fn model_label(&self) -> &'static str {
            "panic-counter"
        }
    }

    let input = b"plain prose without enough line structure to be treated as logs";
    let cas = MemoryCas::new();
    let doc = compress_to_budget(input, &cas, Budget::Reduction(0.8), &PanicCounter);
    assert!(cas.is_empty(), "text reduction must not offload anything");
    assert_eq!(doc.segments.len(), 1);
    assert!(matches!(doc.segments[0], Segment::Verbatim(_)));
    assert_eq!(doc.reconstruct(&cas).unwrap(), input);
}

#[test]
fn leading_keep_coalesces_to_one_ref() {
    let input = bloated_array(300);
    let cas = MemoryCas::new();
    let tok = HeuristicCounter;
    let doc = compress_to_budget(&input, &cas, Budget::Reduction(0.5), &tok);
    let refs = doc
        .segments
        .iter()
        .filter(|s| matches!(s, Segment::Ref { .. }))
        .count();
    assert_eq!(
        refs, 1,
        "the dropped tail must coalesce into exactly one Ref"
    );
    assert_eq!(doc.reconstruct(&cas).unwrap(), input);
}

#[test]
fn kept_prefix_is_byte_identical() {
    let input = bloated_array(200);
    let cas = MemoryCas::new();
    let tok = HeuristicCounter;
    let doc = compress_to_budget(&input, &cas, Budget::Reduction(0.6), &tok);
    match &doc.segments[0] {
        Segment::Verbatim(prefix) => {
            assert_eq!(
                prefix.as_slice(),
                &input[..prefix.len()],
                "H5: kept prefix mutated"
            );
        }
        other => panic!("expected a verbatim prefix, got {other:?}"),
    }
}

#[test]
fn json_object_offloads_whole_and_reconstructs() {
    let mut s = String::from("{");
    for i in 0..100 {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!(r#""k{i}":"value number {i}""#));
    }
    s.push('}');
    let input = s.into_bytes();

    let cas = MemoryCas::new();
    let tok = HeuristicCounter;
    let doc = compress_to_budget(&input, &cas, Budget::Reduction(0.8), &tok);
    assert_eq!(doc.reconstruct(&cas).unwrap(), input);
}

#[test]
fn max_reduction_matches_stage0_compress() {
    let input = bloated_array(100);
    let tok = HeuristicCounter;
    let cas1 = MemoryCas::new();
    let cas2 = MemoryCas::new();
    let budgeted = compress_to_budget(&input, &cas1, Budget::MaxReduction, &tok);
    let stage0 = compress(&input, &cas2);
    assert_eq!(budgeted, stage0, "MaxReduction must equal Stage-0 compress");
    assert_eq!(budgeted.reconstruct(&cas1).unwrap(), input);
}

proptest! {
    /// The strengthened reversibility gate: `emit_kept` is byte-exact for ANY kept-set,
    /// including scattered selections that produce MULTIPLE coalesced Refs.
    #[test]
    fn emit_kept_reconstructs_for_any_selection(
        kind in 0u8..2,
        n in 0usize..40,
        keep_bits in any::<u64>(),
    ) {
        let input = if kind == 0 { bloated_array(n) } else { bloated_log(n) };
        let (units, noun) = coffer_core::low_level::units_for(&input);
        let kept: Vec<bool> = (0..units.len()).map(|i| (keep_bits >> (i % 64)) & 1 == 1).collect();
        let cas = MemoryCas::new();
        let doc = coffer_core::low_level::emit_kept(&input, &units, &kept, &cas, noun);
        prop_assert_eq!(doc.reconstruct(&cas).expect("reconstruct"), input);
    }
}

#[test]
fn scattered_selection_coalesces_per_run() {
    // Keep every other element: dropped units are singletons separated by kept ones,
    // so each becomes its own Ref. 6 elements, drop {1,3,5} → 3 Refs.
    let input = format!(
        "[{}]",
        (0..6)
            .map(|i| format!("{{\"id\":{i}}}"))
            .collect::<Vec<_>>()
            .join(",")
    )
    .into_bytes();
    let (units, noun) = coffer_core::low_level::units_for(&input);
    assert_eq!(units.len(), 6);
    let kept: Vec<bool> = (0..6).map(|i| i % 2 == 0).collect();
    let cas = MemoryCas::new();
    let doc = coffer_core::low_level::emit_kept(&input, &units, &kept, &cas, noun);
    let refs = doc
        .segments
        .iter()
        .filter(|s| matches!(s, Segment::Ref { .. }))
        .count();
    assert_eq!(refs, 3, "three isolated dropped units → three Refs");
    assert_eq!(doc.reconstruct(&cas).unwrap(), input);
}

#[test]
fn tiny_target_below_floor_stays_reversible_and_stable() {
    // Below the structural floor (kept '[' + one sentinel + ']'), the codec cannot reach
    // the target; it emits its minimum render (which exceeds target) and stays reversible.
    let input = bloated_array(200);
    let cas = MemoryCas::new();
    let tok = HeuristicCounter;
    for t in [0usize, 1, 5, 8] {
        let doc = compress_to_budget(&input, &cas, Budget::Tokens(t), &tok);
        assert_eq!(
            doc.reconstruct(&cas).unwrap(),
            input,
            "reversible at tiny target {t}"
        );
    }
    // All sub-floor targets collapse to the same minimum render.
    let a = compress_to_budget(&input, &cas, Budget::Tokens(0), &tok);
    let b = compress_to_budget(&input, &cas, Budget::Tokens(1), &tok);
    assert_eq!(
        a, b,
        "below the floor, tiny targets yield the same minimum doc"
    );
}

#[test]
fn reduction_fraction_and_percent_helper_agree() {
    // Budget::Reduction(0.4) and target_for_reduction(raw, 40) share one rounding rule.
    let input = bloated_array(120);
    let cas = MemoryCas::new();
    let tok = HeuristicCounter;
    let raw = tok.count(&String::from_utf8_lossy(&input));
    let by_frac = compress_to_budget(&input, &cas, Budget::Reduction(0.4), &tok);
    let by_pct = compress_to_budget(
        &input,
        &cas,
        Budget::Tokens(coffer_core::low_level::target_for_reduction(raw, 40)),
        &tok,
    );
    assert_eq!(by_frac, by_pct);
}

#[test]
fn log_budget_reconstructs_and_offloads() {
    let input = bloated_log(200);
    let cas = MemoryCas::new();
    let tok = HeuristicCounter;
    let doc = compress_to_budget(&input, &cas, Budget::Reduction(0.7), &tok);
    assert!(!cas.is_empty(), "a 70%-reduced log should offload");
    assert_eq!(doc.reconstruct(&cas).unwrap(), input);
}
