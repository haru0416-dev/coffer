//! `reconstruct` must reject a CAS that returns bytes whose hash != the key — a corrupt,
//! buggy, or poisoned backend — rather than silently emitting wrong bytes. The
//! content-hash check is what makes byte-faithfulness self-verifying once shared/persistent
//! CAS backends land (the length guard alone would miss a same-length payload).

use std::sync::Arc;

use coffer_cas::{Cas, ContentHash};
use coffer_core::{CompressedDoc, ReconstructError, Segment};

/// A deliberately dishonest CAS: `get` returns same-length-but-different bytes for a key.
struct LyingCas {
    key: ContentHash,
    wrong: Vec<u8>,
}

impl Cas for LyingCas {
    fn put(&self, bytes: &[u8]) -> ContentHash {
        ContentHash::of(bytes)
    }
    fn get(&self, hash: &ContentHash) -> Option<Arc<[u8]>> {
        (hash == &self.key).then(|| Arc::from(self.wrong.as_slice()))
    }
    fn len(&self) -> usize {
        1
    }
}

#[test]
fn reconstruct_rejects_hash_mismatch() {
    let original = b"the real original bytes".to_vec();
    let key = ContentHash::of(&original);
    let wrong = vec![b'X'; original.len()]; // same length, different content

    let doc = CompressedDoc {
        segments: vec![Segment::Ref {
            hash: key.clone(),
            summary: "x".into(),
            original_len: original.len(),
        }],
    };
    let cas = LyingCas { key, wrong };

    match doc.reconstruct(&cas) {
        Err(ReconstructError::HashMismatch(_)) => {}
        other => panic!("expected HashMismatch, got {other:?}"),
    }
}

#[test]
fn reconstruct_rejects_missing_original() {
    let doc = CompressedDoc {
        segments: vec![Segment::Ref {
            hash: ContentHash::of(b"never stored"),
            summary: "x".into(),
            original_len: 12,
        }],
    };
    // A store that holds a different key: nothing to retrieve for ours.
    let cas = LyingCas {
        key: ContentHash::of(b"other"),
        wrong: Vec::new(),
    };
    match doc.reconstruct(&cas) {
        Err(ReconstructError::MissingOriginal(_)) => {}
        other => panic!("expected MissingOriginal, got {other:?}"),
    }
}
