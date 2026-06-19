//! The Stage 0 gate: `reconstruct(compress(x)) == x` byte-for-byte, with the Ref/offload
//! path *actually driven* (not silently degraded to passthrough), plus effectiveness and
//! boundary checks. Fidelity (byte-exactness) and effectiveness (size reduction) are
//! asserted on separate axes — see crate docs.

use coffer_cas::{Cas, MemoryCas};
use coffer_core::{CompressedDoc, ContentType, MIN_COMPRESS_BYTES, Segment, compress, detect};
use proptest::prelude::*;
use proptest::test_runner::TestCaseError;

fn took_ref(doc: &CompressedDoc) -> bool {
    matches!(doc.segments.as_slice(), [Segment::Ref { .. }])
}

/// Round-trips `bytes` and asserts the Ref branch is taken *exactly when* the input is a
/// compressible type at/above the size threshold — so the test cannot silently degrade to
/// only exercising the trivial passthrough path.
fn round_trip_with_branch_check(bytes: &[u8]) -> Result<(), TestCaseError> {
    let cas = MemoryCas::new();
    let doc = compress(bytes, &cas);
    prop_assert_eq!(doc.reconstruct(&cas).expect("reconstruct"), bytes.to_vec());

    let compressible = matches!(detect(bytes), ContentType::Json | ContentType::Log)
        && bytes.len() >= MIN_COMPRESS_BYTES;
    prop_assert_eq!(took_ref(&doc), compressible);
    if compressible {
        prop_assert!(
            !cas.is_empty(),
            "Ref path must store the original in the CAS"
        );
    }
    Ok(())
}

proptest! {
    /// Arbitrary bytes (dominated by the passthrough path) reconstruct exactly.
    #[test]
    fn arbitrary_bytes_round_trip(bytes in proptest::collection::vec(any::<u8>(), 0..8192)) {
        let cas = MemoryCas::new();
        let doc = compress(&bytes, &cas);
        prop_assert_eq!(doc.reconstruct(&cas).expect("reconstruct"), bytes);
    }

    /// JSON arrays of objects, inflated so many cases clear the 256-byte threshold and
    /// drive the Ref path. Reconstruct must be byte-exact AND the right branch taken.
    #[test]
    fn json_array_round_trip(
        rows in proptest::collection::vec(
            proptest::collection::btree_map("[a-z]{1,6}", "[a-zA-Z0-9 ]{0,40}", 1..6),
            0..40,
        )
    ) {
        use serde_json::{Map, Value};
        let array: Vec<Value> = rows
            .into_iter()
            .map(|m| {
                let obj: Map<String, Value> =
                    m.into_iter().map(|(k, v)| (k, Value::String(v))).collect();
                Value::Object(obj)
            })
            .collect();
        let bytes = serde_json::to_vec(&Value::Array(array)).unwrap();
        round_trip_with_branch_check(&bytes)?;
    }

    /// JSON objects (the `Value::Object` summarize branch), inflated past the threshold.
    #[test]
    fn json_object_round_trip(
        fields in proptest::collection::btree_map("[a-z]{1,8}", "[a-zA-Z0-9 ]{0,60}", 1..30)
    ) {
        use serde_json::{Map, Value};
        let obj: Map<String, Value> =
            fields.into_iter().map(|(k, v)| (k, Value::String(v))).collect();
        let bytes = serde_json::to_vec(&Value::Object(obj)).unwrap();
        round_trip_with_branch_check(&bytes)?;
    }

    /// Synthetic logs (both LF and CRLF) drive the Log Ref branch.
    #[test]
    fn log_round_trip(
        n in 8usize..60,
        msg in "[a-zA-Z0-9 ._-]{1,40}",
        crlf in any::<bool>(),
    ) {
        let sep = if crlf { "\r\n" } else { "\n" };
        let mut s = String::new();
        for i in 0..n {
            s.push_str(&format!("2026-06-02T00:00:{:02} INFO {}", i % 60, msg));
            s.push_str(sep);
        }
        round_trip_with_branch_check(s.as_bytes())?;
    }
}

/// Boundary: 255 bytes stays verbatim, 256 offloads — pinning `MIN_COMPRESS_BYTES`.
#[test]
fn threshold_boundary_255_passthrough_256_offload() {
    // `["aaa…aa"]` has total len = 4 (brackets+quotes) + pad.
    let make = |total: usize| format!("[\"{}\"]", "a".repeat(total - 4)).into_bytes();

    let b255 = make(255);
    assert_eq!(b255.len(), 255);
    let cas = MemoryCas::new();
    let doc = compress(&b255, &cas);
    assert!(matches!(doc.segments.as_slice(), [Segment::Verbatim(_)]));
    assert!(cas.is_empty());
    assert_eq!(doc.reconstruct(&cas).unwrap(), b255);

    let b256 = make(256);
    assert_eq!(b256.len(), 256);
    let cas2 = MemoryCas::new();
    let doc2 = compress(&b256, &cas2);
    assert!(matches!(doc2.segments.as_slice(), [Segment::Ref { .. }]));
    assert_eq!(doc2.reconstruct(&cas2).unwrap(), b256);
}

/// Two distinct inputs offloaded into one shared CAS each reconstruct to their own bytes
/// (the Stage-1 multi-document scenario).
#[test]
fn two_inputs_share_one_cas() {
    let a = format!(
        "[{}]",
        (0..60)
            .map(|i| format!("{{\"id\":{i},\"v\":\"x\"}}"))
            .collect::<Vec<_>>()
            .join(",")
    )
    .into_bytes();
    let b = "2026-06-02 ERROR boom\n".repeat(20).into_bytes();

    let cas = MemoryCas::new();
    let da = compress(&a, &cas);
    let db = compress(&b, &cas);
    assert!(took_ref(&da) && took_ref(&db), "both inputs should offload");
    assert_eq!(da.reconstruct(&cas).unwrap(), a);
    assert_eq!(db.reconstruct(&cas).unwrap(), b);
}

/// The "store original bytes, never re-serialize" honesty rail: the CAS holds the exact
/// input slice that produced the Ref, byte for byte.
#[test]
fn ref_stores_exact_original_bytes() {
    let bytes = format!(
        "[{}]",
        (0..50)
            .map(|i| format!("{{\"id\":{i},\"v\":\"x\"}}"))
            .collect::<Vec<_>>()
            .join(",")
    )
    .into_bytes();
    let cas = MemoryCas::new();
    let doc = compress(&bytes, &cas);
    let Segment::Ref { hash, .. } = &doc.segments[0] else {
        panic!("expected a Ref segment");
    };
    let got = cas.get(hash);
    assert_eq!(
        got.as_deref(),
        Some(bytes.as_slice()),
        "CAS must hold the raw input, not a re-serialization"
    );
}

/// Effectiveness (separate from fidelity): a big, redundant JSON array collapses to a tiny
/// model-facing render while STILL reconstructing exactly.
#[test]
fn bloated_json_shrinks_model_view_but_stays_lossless() {
    let mut json = String::from("[");
    for i in 0..500 {
        if i > 0 {
            json.push(',');
        }
        json.push_str(&format!(
            r#"{{"id":{i},"name":"item-{i}","status":"active","score":{}}}"#,
            i * 7
        ));
    }
    json.push(']');
    let bytes = json.into_bytes();

    let cas = MemoryCas::new();
    let doc = compress(&bytes, &cas);
    assert!(
        doc.rendered_len() < bytes.len() / 10,
        "expected a >10x reduction in model-facing size, got {} vs {}",
        doc.rendered_len(),
        bytes.len()
    );
    assert_eq!(doc.reconstruct(&cas).expect("reconstruct"), bytes);
}

/// `rendered_len` is faithful: non-UTF-8 passthrough is counted by true byte length, not
/// inflated by lossy U+FFFD expansion.
#[test]
fn rendered_len_is_faithful_for_non_utf8_passthrough() {
    let bytes = vec![0xff, 0xfe, b'h', b'i', 0x80, 0x81];
    let cas = MemoryCas::new();
    let doc = compress(&bytes, &cas);
    assert_eq!(doc.rendered_len(), bytes.len());
}

/// A short JSON input stays verbatim (below the threshold) and is unchanged.
#[test]
fn small_input_passes_through() {
    let bytes = br#"[{"a":1}]"#.to_vec();
    let cas = MemoryCas::new();
    let doc = compress(&bytes, &cas);
    assert!(cas.is_empty(), "small input should not be offloaded");
    assert_eq!(doc.reconstruct(&cas).unwrap(), bytes);
}
