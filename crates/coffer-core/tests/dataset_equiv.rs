//! Equivalence: `Dataset` (parse-once + lazy column cache) must agree with the byte-slice
//! entry points on EVERY input and query — including mixed-type fields, missing fields,
//! cross-type predicates, and non-`f64` numbers under `arbitrary_precision`. The column
//! fast path is an optimization, never a semantic fork.

use coffer_core::{
    Agg, Dataset, Op, Predicate, bucket_aggregate, describe, digest, query_aggregate,
};
use proptest::prelude::*;
use serde_json::{Value, json};

/// A field value drawn from every class the column cells distinguish: f64 numbers,
/// strings, bools, null, nested containers — plus ABSENT (the generator skips the field).
fn field_value() -> impl Strategy<Value = Option<Value>> {
    prop_oneof![
        3 => (-1000i64..1000).prop_map(|n| Some(json!(n))),
        2 => (-100.0f64..100.0).prop_map(|f| Some(json!(f))),
        3 => "[a-c]{1,3}".prop_map(|s| Some(json!(s))),
        1 => any::<bool>().prop_map(|b| Some(json!(b))),
        1 => Just(Some(Value::Null)),
        1 => Just(Some(json!([1, 2]))),
        1 => Just(None), // field absent from this record
    ]
}

/// Rows over a tiny field pool so predicates actually hit the generated fields.
fn rows() -> impl Strategy<Value = Vec<Value>> {
    proptest::collection::vec(
        (field_value(), field_value(), field_value()).prop_map(|(x, y, z)| {
            let mut obj = serde_json::Map::new();
            if let Some(v) = x {
                obj.insert("x".to_string(), v);
            }
            if let Some(v) = y {
                obj.insert("y".to_string(), v);
            }
            if let Some(v) = z {
                obj.insert("z".to_string(), v);
            }
            Value::Object(obj)
        }),
        0..40,
    )
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        Just(Op::Eq),
        Just(Op::Ne),
        Just(Op::Gt),
        Just(Op::Ge),
        Just(Op::Lt),
        Just(Op::Le),
    ]
}

fn predicate() -> impl Strategy<Value = Predicate> {
    (
        prop_oneof![Just("x"), Just("y"), Just("z"), Just("w")],
        op(),
        prop_oneof![
            2 => (-1000i64..1000).prop_map(|n| json!(n)),
            2 => "[a-c]{1,3}".prop_map(|s| json!(s)),
            1 => any::<bool>().prop_map(|b| json!(b)),
            1 => Just(Value::Null),
        ],
    )
        .prop_map(|(field, op, value)| Predicate {
            field: field.to_string(),
            op,
            value,
        })
}

fn agg() -> impl Strategy<Value = Agg> {
    prop_oneof![
        Just(Agg::Count),
        prop_oneof![Just("x"), Just("y"), Just("z")].prop_map(|f| Agg::Sum(f.to_string())),
        prop_oneof![Just("x"), Just("y"), Just("z")].prop_map(|f| Agg::Mean(f.to_string())),
        prop_oneof![Just("x"), Just("y"), Just("z")].prop_map(|f| Agg::Min(f.to_string())),
        prop_oneof![Just("x"), Just("y"), Just("z")].prop_map(|f| Agg::Max(f.to_string())),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn query_aggregate_equivalence(
        rows in rows(),
        preds in proptest::collection::vec(predicate(), 0..3),
        agg in agg(),
    ) {
        let bytes = serde_json::to_vec(&Value::Array(rows)).unwrap();
        let via_bytes = query_aggregate(&bytes, &preds, &agg);
        let ds = Dataset::parse(&bytes).expect("top-level array parses");
        // Run twice: the first call builds columns, the second exercises the cache.
        let via_ds_cold = ds.query_aggregate(&preds, &agg);
        let via_ds_warm = ds.query_aggregate(&preds, &agg);
        prop_assert_eq!(&via_bytes, &via_ds_cold);
        prop_assert_eq!(&via_bytes, &via_ds_warm);
    }

    #[test]
    fn describe_and_digest_equivalence(rows in rows()) {
        let bytes = serde_json::to_vec(&Value::Array(rows)).unwrap();
        let ds = Dataset::parse(&bytes).expect("top-level array parses");
        prop_assert_eq!(describe(&bytes), ds.describe());
        for q in ["max x", "how many rows have y", "mean z", "count distinct x"] {
            prop_assert_eq!(digest(&bytes, q), ds.digest(q));
        }
    }

    #[test]
    fn bucket_equivalence(rows in rows(), width in prop_oneof![Just(1.0f64), Just(10.0), Just(0.5)]) {
        let bytes = serde_json::to_vec(&Value::Array(rows)).unwrap();
        let ds = Dataset::parse(&bytes).expect("top-level array parses");
        prop_assert_eq!(
            bucket_aggregate(&bytes, "x", width, &Agg::Count),
            ds.bucket_aggregate("x", width, &Agg::Count)
        );
    }
}

/// `arbitrary_precision` edge: a number that is `is_number()` but NOT `f64`-representable
/// must refuse aggregation on both paths, and behave identically under predicates.
#[test]
fn non_f64_number_refuses_on_both_paths() {
    let big: serde_json::Number = "1e400".parse().expect("arbitrary-precision number");
    let bytes = serde_json::to_vec(&json!([
        {"x": 1, "s": "a"},
        {"x": Value::Number(big), "s": "b"},
        {"x": 3, "s": "a"},
    ]))
    .unwrap();
    let ds = Dataset::parse(&bytes).unwrap();
    let sum = Agg::Sum("x".to_string());

    // Aggregating over a matched set containing the non-f64 value: refuse, both paths.
    assert_eq!(query_aggregate(&bytes, &[], &sum), None);
    assert_eq!(ds.query_aggregate(&[], &sum), None);

    // A predicate that excludes the bad row: exact answer, both paths, identical.
    let pred = [Predicate {
        field: "s".to_string(),
        op: Op::Eq,
        value: json!("a"),
    }];
    let via_bytes = query_aggregate(&bytes, &pred, &sum).expect("clean subset aggregates");
    let via_ds = ds
        .query_aggregate(&pred, &sum)
        .expect("clean subset aggregates");
    assert_eq!(via_bytes, via_ds);
    assert_eq!(via_bytes.value, 4.0);

    // Predicate ON the mixed field: the non-f64 row is an Other cell — cross-type rules
    // must match json_cmp on both paths (Ne matches, ordering does not).
    for op in [Op::Eq, Op::Ne, Op::Gt, Op::Le] {
        let p = [Predicate {
            field: "x".to_string(),
            op,
            value: json!(1),
        }];
        assert_eq!(
            query_aggregate(&bytes, &p, &Agg::Count),
            ds.query_aggregate(&p, &Agg::Count),
            "op {op:?} diverged"
        );
    }
}

/// Dataset::parse must accept exactly what the byte-slice entry points accept: a
/// top-level array (empty included), nothing else.
#[test]
fn parse_acceptance_matches_entry_points() {
    assert!(Dataset::parse(b"[]").is_some());
    assert!(Dataset::parse(br#"[{"a":1}]"#).is_some());
    assert!(Dataset::parse(br#"{"a":1}"#).is_none());
    assert!(Dataset::parse(b"42").is_none());
    assert!(Dataset::parse(b"not json").is_none());
}
