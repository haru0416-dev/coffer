//! Content-addressed store: byte-faithful `put`/`get` keyed by SHA-256.
//!
//! This is the persistence half of coffer's reversibility guarantee. [`Cas::put`] stores
//! the *exact original bytes*; [`Cas::get`] returns them unchanged. coffer-core stores the
//! original of any region it compresses here, so reconstruction is byte-exact regardless
//! of how the compressor rendered that region for the model. (Reversibility comes from
//! storing original bytes, never re-serializing a value — see `docs/DESIGN.md` §1.)
//!
//! `Cas` uses interior mutability (`put(&self)`) and is `Send + Sync`, so the Stage 1
//! budget-matched harness can share one store across concurrent tasks without serializing
//! every put behind an exclusive borrow.

#![warn(clippy::pedantic, missing_docs)]

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use sha2::{Digest, Sha256};

#[cfg(feature = "sqlite")]
mod sqlite;
#[cfg(feature = "sqlite")]
pub use sqlite::{SqliteCas, SqliteConfig, SqliteDiskUsage, read_blob};

/// A content hash: the hex-encoded SHA-256 of the stored bytes.
///
/// Constructed only via [`ContentHash::of`], so the inner string is always a
/// 64-character lowercase hex digest.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ContentHash(String);

impl ContentHash {
    /// Compute the content hash of `bytes`.
    #[must_use]
    pub fn of(bytes: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        ContentHash(hex_encode(&hasher.finalize()))
    }

    /// Compute the same 12-hex-character display prefix as [`ContentHash::short`]
    /// without allocating the full 64-character storage key. Use this only for
    /// model-facing sentinel probes; storage and retrieval must use [`ContentHash::of`].
    #[must_use]
    pub fn short_of(bytes: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        let mut out = String::with_capacity(12);
        push_hex_prefix(&mut out, &hasher.finalize(), 6);
        out
    }

    /// Append the same 12-hex-character display prefix as [`ContentHash::short`]
    /// directly into `out`, avoiding a temporary string allocation.
    pub fn push_short_of(bytes: &[u8], out: &mut String) {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        push_hex_prefix(out, &hasher.finalize(), 6);
    }

    /// The full 64-character hex digest — the canonical storage key.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// A short prefix for human-facing sentinels. NOT a storage key — never retrieve by
    /// this; collisions are possible at 12 hex chars (48 bits).
    #[must_use]
    pub fn short(&self) -> &str {
        &self.0[..12]
    }

    /// Parse a full 64-character lowercase-hex digest back into a `ContentHash` — the inverse of
    /// [`ContentHash::as_str`], for retrieve-by-key (e.g. unfolding a proxy sentinel's hash).
    /// Returns `None` for anything that is not exactly 64 lowercase hex chars, so it can never key
    /// on a [`ContentHash::short`] prefix or arbitrary text.
    #[must_use]
    pub fn from_hex(s: &str) -> Option<Self> {
        if s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
            Some(ContentHash(s.to_string()))
        } else {
            None
        }
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    push_hex_prefix(&mut out, bytes, bytes.len());
    out
}

fn push_hex_prefix(out: &mut String, bytes: &[u8], take: usize) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let take = take.min(bytes.len());
    for b in bytes.iter().take(take) {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
}

/// A byte-faithful content-addressed store. `Send + Sync` so it can be shared (`&dyn Cas`)
/// across concurrent harness tasks.
pub trait Cas: Send + Sync {
    /// Store `bytes`, returning their content hash. Idempotent: storing identical bytes
    /// yields the same hash and does not duplicate the payload.
    fn put(&self, bytes: &[u8]) -> ContentHash;

    /// Retrieve the exact bytes for `hash`, if present. The returned bytes equal what was
    /// passed to [`Cas::put`], byte for byte. Returns an `Arc<[u8]>` rather than a borrow
    /// because backends behind a lock cannot hand out an internal reference.
    fn get(&self, hash: &ContentHash) -> Option<Arc<[u8]>>;

    /// Whether `hash` is present.
    fn contains(&self, hash: &ContentHash) -> bool {
        self.get(hash).is_some()
    }

    /// Number of distinct objects stored.
    fn len(&self) -> usize;

    /// Whether the store holds no objects.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// In-memory CAS backend — the zero-dependency default. A persistent, write-through
/// `SqliteCas` is available behind the `sqlite` feature; on-disk blob and
/// `TTL` eviction may follow behind the same trait.
#[derive(Default)]
pub struct MemoryCas {
    map: RwLock<HashMap<ContentHash, Arc<[u8]>>>,
}

impl MemoryCas {
    /// Create an empty in-memory store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl Cas for MemoryCas {
    fn put(&self, bytes: &[u8]) -> ContentHash {
        let hash = ContentHash::of(bytes);
        // Recover from a poisoned lock instead of panicking: the data under the lock is always in a
        // consistent state (insert/get/len are individually safe), so a panic elsewhere must not
        // permanently brick the store — that would defeat the whole reversibility guarantee.
        let mut map = self
            .map
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        map.entry(hash.clone()).or_insert_with(|| Arc::from(bytes));
        hash
    }

    fn get(&self, hash: &ContentHash) -> Option<Arc<[u8]>> {
        self.map
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(hash)
            .cloned()
    }

    fn len(&self) -> usize {
        self.map
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_then_get_is_exact() {
        let cas = MemoryCas::new();
        let h = cas.put(b"hello");
        let got = cas.get(&h);
        assert_eq!(got.as_deref(), Some(&b"hello"[..]));
    }

    #[test]
    fn hash_is_full_hex_and_short_is_prefix() {
        let h = ContentHash::of(b"x");
        assert_eq!(h.as_str().len(), 64);
        assert_eq!(h.short(), &h.as_str()[..12]);
        assert_eq!(ContentHash::short_of(b"x"), h.short());
        let mut out = String::from("prefix:");
        ContentHash::push_short_of(b"x", &mut out);
        assert_eq!(out, format!("prefix:{}", h.short()));
    }

    #[test]
    fn from_hex_roundtrips_and_rejects_bad_input() {
        let h = ContentHash::of(b"payload");
        assert_eq!(ContentHash::from_hex(h.as_str()), Some(h.clone()));
        assert_eq!(
            ContentHash::from_hex(h.short()),
            None,
            "a 12-char prefix is not a key"
        );
        assert_eq!(ContentHash::from_hex("zz"), None);
        assert_eq!(
            ContentHash::from_hex(&"A".repeat(64)),
            None,
            "uppercase is not our digest form"
        );
    }

    #[test]
    fn put_is_idempotent() {
        let cas = MemoryCas::new();
        cas.put(b"same");
        cas.put(b"same");
        assert_eq!(cas.len(), 1);
    }

    #[test]
    fn concurrent_put_get_is_consistent() {
        use std::sync::Arc;
        use std::thread;
        let cas = Arc::new(MemoryCas::new());
        let handles: Vec<_> = (0..8)
            .map(|t| {
                let c = Arc::clone(&cas);
                thread::spawn(move || {
                    for i in 0..100 {
                        let h = c.put(format!("item-{t}-{i}").into_bytes().as_slice());
                        assert!(c.get(&h).is_some());
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(cas.len(), 800);
    }

    #[test]
    fn survives_a_poisoned_lock() {
        use std::sync::Arc;
        let cas = Arc::new(MemoryCas::new());
        let h = cas.put(b"before");
        // GENUINELY poison the lock: panic while holding the write guard (same-crate test reaches
        // the private field). Recovery must keep the store working — never permanently brick it.
        let c = Arc::clone(&cas);
        let _ = std::thread::spawn(move || {
            let _guard = c.map.write().unwrap();
            panic!("intentional poison while holding the write lock");
        })
        .join();
        assert!(cas.get(&h).is_some(), "CAS must survive a poisoned lock");
        let h2 = cas.put(b"after");
        assert!(
            cas.get(&h2).is_some(),
            "put must work after poison recovery"
        );
    }

    #[test]
    fn empty_bytes_round_trip() {
        let cas = MemoryCas::new();
        let h = cas.put(b"");
        let got = cas.get(&h);
        assert_eq!(got.as_deref(), Some(&b""[..]));
    }
}
