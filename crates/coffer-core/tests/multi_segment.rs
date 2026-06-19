//! H5 — "passthrough is sacred". Verbatim bytes adjacent to a compressed Ref must survive
//! both reconstruction and model-rendering completely unmutated. Stage 0's compressor only
//! ever emits a single segment, but the `CompressedDoc` model already supports the
//! multi-segment documents Stage 1 will produce — so we lock the property in *now*, before
//! the codebase grows interleaved Verbatim/Ref documents the rest of the suite wouldn't police.

use coffer_cas::{Cas, MemoryCas};
use coffer_core::{CompressedDoc, Segment};

#[test]
fn verbatim_around_ref_survives_reconstruct_and_render() {
    let prefix = b"PREFIX-keep-exact\n".to_vec();
    let original = br#"{"big":"original tool output that got offloaded"}"#.to_vec();
    let suffix = b"\nSUFFIX-keep-exact".to_vec();

    let cas = MemoryCas::new();
    let hash = cas.put(&original);
    let doc = CompressedDoc {
        segments: vec![
            Segment::Verbatim(prefix.clone()),
            Segment::Ref {
                hash,
                summary: "json object".into(),
                original_len: original.len(),
            },
            Segment::Verbatim(suffix.clone()),
        ],
    };

    // (1) reconstruct splices prefix + original + suffix, byte for byte.
    let mut expected = Vec::new();
    expected.extend_from_slice(&prefix);
    expected.extend_from_slice(&original);
    expected.extend_from_slice(&suffix);
    assert_eq!(doc.reconstruct(&cas).unwrap(), expected);

    // (2) the model-facing render keeps the verbatim prefix/suffix unmutated around the sentinel.
    let rendered = doc.render_for_model();
    assert!(
        rendered.starts_with("PREFIX-keep-exact\n"),
        "prefix mutated: {rendered:?}"
    );
    assert!(
        rendered.ends_with("\nSUFFIX-keep-exact"),
        "suffix mutated: {rendered:?}"
    );
    assert!(
        rendered.contains("<<cof:"),
        "sentinel missing: {rendered:?}"
    );
}
