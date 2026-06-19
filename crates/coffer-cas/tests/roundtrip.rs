//! Property tests for the byte-faithful storage guarantee.

use coffer_cas::{Cas, ContentHash, MemoryCas};
use proptest::prelude::*;

proptest! {
    /// `get(put(bytes))` returns exactly `bytes`, for any byte sequence.
    #[test]
    fn get_returns_exact_bytes(bytes in proptest::collection::vec(any::<u8>(), 0..4096)) {
        let cas = MemoryCas::new();
        let h = cas.put(&bytes);
        let got = cas.get(&h);
        prop_assert_eq!(got.as_deref(), Some(bytes.as_slice()));
    }

    /// The hash is a pure function of the bytes (deterministic across stores).
    #[test]
    fn put_is_deterministic(bytes in proptest::collection::vec(any::<u8>(), 0..4096)) {
        let a = MemoryCas::new();
        let b = MemoryCas::new();
        prop_assert_eq!(a.put(&bytes), b.put(&bytes));
    }

    /// Distinct inputs hash to distinct keys (no collisions at this scale).
    #[test]
    fn distinct_bytes_distinct_hash(
        x in proptest::collection::vec(any::<u8>(), 0..512),
        y in proptest::collection::vec(any::<u8>(), 0..512),
    ) {
        prop_assume!(x != y);
        prop_assert_ne!(ContentHash::of(&x), ContentHash::of(&y));
    }
}
