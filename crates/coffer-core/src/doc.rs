//! The compressed-document model: a sequence of [`Segment`]s that renders one way for
//! the model and reconstructs byte-exactly from the CAS.

use coffer_cas::{Cas, ContentHash};

/// One piece of a compressed document.
#[derive(Clone, Debug, PartialEq)]
pub enum Segment {
    /// Passthrough bytes — emitted verbatim both to the model and on reconstruction.
    Verbatim(Vec<u8>),
    /// A compressed reference. The model sees `summary`; the original bytes live in the
    /// CAS under `hash` and are spliced back on reconstruction. `original_len` is a cheap
    /// pre-check; the authoritative integrity check is the content hash itself.
    Ref {
        /// CAS key for the exact original bytes of this offloaded region.
        hash: ContentHash,
        /// Short, model-facing description shown in place of the bytes.
        summary: String,
        /// Length of the original region — a cheap pre-check before the hash verification.
        original_len: usize,
    },
}

/// A compressed view of some input: byte-faithfully reconstructable, while rendering
/// compactly for the model.
#[derive(Clone, Debug, PartialEq)]
pub struct CompressedDoc {
    /// The ordered segments that tile the original input exactly.
    pub segments: Vec<Segment>,
}

/// Why reconstruction failed. Each variant is a hard bug (a stored original went missing,
/// changed length, or no longer hashes to its key), never an expected outcome.
#[derive(Debug, thiserror::Error)]
pub enum ReconstructError {
    /// The CAS no longer holds the original bytes for a referenced hash.
    #[error("CAS is missing the original bytes for hash {0}")]
    MissingOriginal(String),
    /// Recovered bytes whose length disagrees with the segment's recorded length.
    #[error("recovered {got} bytes but the segment recorded {expected}")]
    LengthMismatch {
        /// Length actually returned by the CAS.
        got: usize,
        /// Length the segment recorded at compression time.
        expected: usize,
    },
    /// The CAS returned bytes that do not hash to the requested key (corrupt/dishonest backend).
    #[error("CAS returned bytes whose hash does not match key {0} (corrupt or dishonest backend)")]
    HashMismatch(String),
}

/// The model-facing sentinel for a compressed region. Display only — see
/// [`CompressedDoc::render_for_model`].
fn sentinel(hash: &ContentHash, summary: &str) -> String {
    format!("<<cof:{} {}>>", hash.short(), summary)
}

impl CompressedDoc {
    /// What the model sees: verbatim text interleaved with short `<<cof:…>>` sentinels
    /// standing in for compressed regions. Non-UTF-8 verbatim bytes are rendered lossily
    /// (display only — reconstruction always uses the raw bytes).
    ///
    /// This is a **display rendering, not a reversible wire format**: it is never parsed back
    /// (reconstruction walks `segments`). The sentinel shows [`ContentHash::short`] (kept short so it
    /// stays cheap even when many regions are offloaded); the proxy ⊕ MCP unfold loop
    /// resolves it by hex **prefix** against the shared CAS. Sentinels are not escaped against
    /// verbatim content that happens to look like one (a known gap — an escaping scheme is future work).
    #[must_use]
    pub fn render_for_model(&self) -> String {
        let mut out = String::new();
        for seg in &self.segments {
            match seg {
                Segment::Verbatim(bytes) => out.push_str(&String::from_utf8_lossy(bytes)),
                Segment::Ref { hash, summary, .. } => out.push_str(&sentinel(hash, summary)),
            }
        }
        out
    }

    /// Byte-exact reconstruction of the original input from `cas`, with an integrity
    /// check: every offloaded region is verified to hash back to its key, so a corrupt or
    /// dishonest CAS yields an error rather than silently-wrong bytes.
    ///
    /// # Errors
    /// Returns [`ReconstructError`] if a referenced original is missing from the CAS
    /// ([`ReconstructError::MissingOriginal`]), has the wrong length
    /// ([`ReconstructError::LengthMismatch`]), or no longer hashes to its key
    /// ([`ReconstructError::HashMismatch`]) — each a hard integrity failure, never expected.
    pub fn reconstruct(&self, cas: &dyn Cas) -> Result<Vec<u8>, ReconstructError> {
        let mut out = Vec::new();
        for seg in &self.segments {
            match seg {
                Segment::Verbatim(bytes) => out.extend_from_slice(bytes),
                Segment::Ref {
                    hash, original_len, ..
                } => {
                    let original = cas.get(hash).ok_or_else(|| {
                        ReconstructError::MissingOriginal(hash.as_str().to_string())
                    })?;
                    if original.len() != *original_len {
                        return Err(ReconstructError::LengthMismatch {
                            got: original.len(),
                            expected: *original_len,
                        });
                    }
                    // Defense-in-depth: a correct CAS returns the bytes that produced the
                    // key. Verifying it makes byte-faithfulness self-checking against a
                    // corrupt / shared / poisoned backend, where the length guard above is
                    // not enough (a same-length payload would slip through).
                    if ContentHash::of(&original) != *hash {
                        return Err(ReconstructError::HashMismatch(hash.as_str().to_string()));
                    }
                    out.extend_from_slice(&original);
                }
            }
        }
        Ok(out)
    }

    /// A faithful byte size of the model-facing document: verbatim segments by their true
    /// byte length, ref segments by their sentinel string. Unlike `render_for_model().len()`,
    /// this does not inflate non-UTF-8 verbatim bytes via lossy U+FFFD expansion, so
    /// `input.len() - rendered_len()` is a sound size delta. (Real token counts use the
    /// target model's tokenizer via `coffer-tokenizer`; this is a byte proxy for quick checks.)
    #[must_use]
    pub fn rendered_len(&self) -> usize {
        self.segments
            .iter()
            .map(|seg| match seg {
                Segment::Verbatim(bytes) => bytes.len(),
                Segment::Ref { hash, summary, .. } => sentinel(hash, summary).len(),
            })
            .sum()
    }
}
