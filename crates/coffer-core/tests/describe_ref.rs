//! `describe` output must be BYTE-IDENTICAL to the original straightforward
//! implementation. The optimized single-pass version (per-type borrowed-key counting,
//! canonical-text materialization deferred to the categorical branch) is an optimization,
//! never a semantic fork — this file pins that with the original algorithm as a test-only
//! reference, fed adversarial values (escape-needing strings, unicode, quotes, control
//! chars, floats, non-f64 numbers, -0.0, bools, null, containers, absent fields).

use coffer_core::describe;
use proptest::prelude::*;
use serde_json::{Value, json};

/// The ORIGINAL describe algorithm, verbatim in shape: count-by `val.to_string()` into a
/// `BTreeMap` per field, `numeric_vals` as two extra passes, `MAX_CATEGORICAL` = 12.
fn reference_describe(input: &[u8]) -> Option<String> {
    const MAX_CATEGORICAL: usize = 12;
    let v: Value = serde_json::from_slice(input).ok()?;
    let arr = v.as_array()?;
    let n = arr.len();
    if n == 0 {
        return Some("[describe] 0 records".to_string());
    }
    if !arr.iter().any(Value::is_object) {
        return None;
    }
    let mut fields: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for rec in arr {
        if let Some(o) = rec.as_object() {
            for k in o.keys() {
                if seen.insert(k.clone()) {
                    fields.push(k.clone());
                }
            }
        }
    }
    fn field_is_clean_numeric(arr: &[Value], field: &str) -> bool {
        let mut seen = false;
        for obj in arr.iter().filter_map(Value::as_object) {
            if let Some(v) = obj.get(field) {
                if v.as_f64().is_none() {
                    return false;
                }
                seen = true;
            }
        }
        seen
    }
    let mut lines = vec![format!("[describe] {n} records, {} fields", fields.len())];
    for f in &fields {
        let mut counts: std::collections::BTreeMap<String, usize> =
            std::collections::BTreeMap::new();
        for rec in arr {
            if let Some(val) = rec.as_object().and_then(|o| o.get(f)) {
                if !val.is_null() {
                    *counts.entry(val.to_string()).or_default() += 1;
                }
            }
        }
        let present: usize = counts.values().sum();
        let distinct = counts.len();
        let numeric_vals: Option<Vec<f64>> = if field_is_clean_numeric(arr, f) {
            let vals: Vec<f64> = arr
                .iter()
                .filter_map(Value::as_object)
                .filter_map(|o| o.get(f).and_then(Value::as_f64))
                .collect();
            if vals.is_empty() { None } else { Some(vals) }
        } else {
            None
        };
        if let Some(vals) = numeric_vals {
            let sum: f64 = vals.iter().sum();
            let mean = sum / vals.len() as f64;
            let min = vals.iter().copied().fold(f64::INFINITY, f64::min);
            let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            lines.push(format!(
                "  {f}: number present={present} distinct={distinct} min={min} max={max} mean={mean} sum={sum}"
            ));
        } else if distinct <= MAX_CATEGORICAL && distinct < present {
            let mut pairs: Vec<(String, usize)> = counts.into_iter().collect();
            pairs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let breakdown = pairs
                .iter()
                .map(|(k, c)| format!("{k}:{c}"))
                .collect::<Vec<_>>()
                .join(", ");
            lines.push(format!(
                "  {f}: present={present} distinct={distinct} — {breakdown}"
            ));
        } else {
            lines.push(format!("  {f}: present={present} distinct={distinct}"));
        }
    }
    Some(lines.join("\n"))
}

/// Adversarial field values: strings that DO need JSON escaping, unicode, floats whose
/// text matters, bools, null, containers — and absent.
fn hostile_value() -> impl Strategy<Value = Option<Value>> {
    prop_oneof![
        3 => "[a-c]{0,3}".prop_map(|s| Some(json!(s))),
        2 => prop_oneof![
            Just(r#"say "hi""#.to_string()),
            Just("back\\slash".to_string()),
            Just("tab\there".to_string()),
            Just("newline\nhere".to_string()),
            Just("naïve — 日本語 ✓".to_string()),
            Just(String::new()),
        ].prop_map(|s| Some(json!(s))),
        3 => (-50i64..50).prop_map(|v| Some(json!(v))),
        2 => prop_oneof![
            Just(0.5f64), Just(-0.0), Just(1e-9), Just(123_456.789),
        ].prop_map(|v| Some(json!(v))),
        1 => any::<bool>().prop_map(|b| Some(json!(b))),
        1 => Just(Some(Value::Null)),
        1 => Just(Some(json!([1, "two"]))),
        1 => Just(Some(json!({"nested": true}))),
        1 => Just(None),
    ]
}

fn rows() -> impl Strategy<Value = Vec<Value>> {
    proptest::collection::vec(
        (hostile_value(), hostile_value(), hostile_value()).prop_map(|(a, b, c)| {
            let mut obj = serde_json::Map::new();
            if let Some(v) = a {
                obj.insert("a".to_string(), v);
            }
            if let Some(v) = b {
                obj.insert("b".to_string(), v);
            }
            if let Some(v) = c {
                obj.insert("c".to_string(), v);
            }
            Value::Object(obj)
        }),
        0..30,
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(768))]

    #[test]
    fn describe_matches_original_reference(rows in rows()) {
        let bytes = serde_json::to_vec(&Value::Array(rows)).unwrap();
        prop_assert_eq!(reference_describe(&bytes), describe(&bytes));
    }
}

/// Non-f64 numbers (`arbitrary_precision`) and mixed numeric/null fields hit the
/// clean-numeric gate; a >12-distinct field hits the plain present/distinct branch.
#[test]
fn describe_matches_reference_on_edge_shapes() {
    let big: serde_json::Number = "1e400".parse().unwrap();
    let cases: Vec<Value> = vec![
        json!([{"x": 1}, {"x": Value::Number(big.clone())}]),
        json!([{"x": 1}, {"x": null}, {"x": 3}]),
        json!([{"x": 1.0}, {"x": 1}]), // "1.0" and "1" are DISTINCT canonical texts
        json!([{"s": "a"}, {"s": "a"}, {"s": "b"}, 42, "not-an-object"]),
        (0..40)
            .map(|i| json!({"id": format!("u-{i}")}))
            .collect::<Vec<_>>()
            .into(),
        json!([{}]),
    ];
    for case in cases {
        let bytes = serde_json::to_vec(&case).unwrap();
        assert_eq!(
            reference_describe(&bytes),
            describe(&bytes),
            "diverged on {case}"
        );
    }
}
