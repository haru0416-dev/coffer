//! Handle storage: the in-memory/SQLite `HandleStore`, the `Coffer` service type it backs,
//! and the parsed-dataset cache. The MCP tool surface itself lives in `main.rs`.
// Split out of main.rs for maintainability; behavior is unchanged. main.rs glob-imports
// this module, so the original single-file scope (and its test suite) still sees every name.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

use coffer_cas::{Cas, ContentHash, SqliteCas, SqliteConfig, create_private_dir_all};
use coffer_core::Dataset;

use crate::limits::retrieve_limits_from_env;
use coffer_core::dataset::DatasetCache;

pub(crate) type HeldBytes = Arc<[u8]>;

pub(crate) enum HandleStore {
    Memory(Mutex<HashMap<String, HeldBytes>>),
    Sqlite {
        path: PathBuf,
        cas: Box<SqliteCas>,
        soft_cap_bytes: Option<usize>,
        warm_bytes_on_open: bool,
        trust_hashes_on_open: bool,
        resident_cap_bytes: Option<usize>,
        checkpoint_every_blobs: Option<usize>,
    },
}

#[derive(Clone)]
pub(crate) struct Coffer {
    pub(crate) store: Arc<HandleStore>,
    /// Parsed-dataset LRU: N query tools against one held handle parse its bytes once.
    /// Sound because handles are content-addressed (bytes never change under a key).
    pub(crate) datasets: Arc<DatasetCache>,
}

impl Coffer {
    pub(crate) fn new() -> anyhow::Result<Self> {
        match std::env::var("COFFER_CAS_DB") {
            Ok(path) if !path.trim().is_empty() => Self::with_sqlite(path),
            _ => Ok(Self::in_memory()),
        }
    }

    pub(crate) fn in_memory() -> Self {
        Self {
            store: Arc::new(HandleStore::Memory(Mutex::new(HashMap::new()))),
            datasets: Arc::new(DatasetCache::new(dataset_cache_entries_from_env())),
        }
    }

    pub(crate) fn with_sqlite(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            create_private_dir_all(parent)?;
        }
        let cfg = SqliteConfig::from_env();
        Ok(Self {
            store: Arc::new(HandleStore::Sqlite {
                path: path.to_path_buf(),
                cas: Box::new(SqliteCas::open_with_config(path, &cfg)?),
                soft_cap_bytes: cfg.soft_cap_bytes,
                warm_bytes_on_open: cfg.warm_bytes_on_open,
                trust_hashes_on_open: cfg.trust_hashes_on_open,
                resident_cap_bytes: cfg.resident_cap_bytes,
                checkpoint_every_blobs: cfg.checkpoint_every_blobs,
            }),
            datasets: Arc::new(DatasetCache::new(dataset_cache_entries_from_env())),
        })
    }

    /// The parsed dataset for a handle's bytes (cache hit or parse) — `None` iff the bytes
    /// are not a top-level JSON array, exactly like the byte-slice query entry points.
    pub(crate) fn dataset(&self, key: &str, bytes: &[u8]) -> Option<Arc<Dataset>> {
        self.datasets.get_or_parse(key, bytes)
    }

    pub(crate) fn put_bytes(&self, bytes: &[u8]) -> ContentHash {
        match self.store.as_ref() {
            HandleStore::Memory(map) => {
                let hash = ContentHash::of(bytes);
                map.lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .entry(hash.as_str().to_string())
                    .or_insert_with(|| Arc::from(bytes));
                hash
            }
            HandleStore::Sqlite { cas, .. } => {
                let hash = cas.put(bytes);
                cas.flush();
                hash
            }
        }
    }

    pub(crate) fn get_handle(&self, handle: &str) -> Option<HeldBytes> {
        match self.store.as_ref() {
            HandleStore::Memory(map) => map
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .get(handle)
                .cloned(),
            HandleStore::Sqlite { path, cas, .. } => {
                if let Some(hash) = ContentHash::from_hex(handle) {
                    return cas.get(&hash);
                }
                coffer_cas::read_blob(path, handle)
                    .ok()
                    .flatten()
                    .map(Arc::from)
            }
        }
    }

    pub(crate) fn handle_count(&self) -> usize {
        match self.store.as_ref() {
            HandleStore::Memory(map) => map.lock().unwrap_or_else(PoisonError::into_inner).len(),
            HandleStore::Sqlite { cas, .. } => cas.len(),
        }
    }

    pub(crate) fn status_text(&self) -> String {
        let retrieve_limits = retrieve_limits_from_env();
        match self.store.as_ref() {
            HandleStore::Memory(map) => {
                let map = map.lock().unwrap_or_else(PoisonError::into_inner);
                let bytes: usize = map.values().map(|b| b.len()).sum();
                format!(
                    "store: memory\nhandles: {}\nresident_bytes: {bytes}\nretrieve_default_bytes: {}\nretrieve_max_bytes: {}",
                    map.len(),
                    retrieve_limits.default_bytes,
                    retrieve_limits.max_bytes
                )
            }
            HandleStore::Sqlite {
                path: _,
                cas,
                soft_cap_bytes,
                warm_bytes_on_open,
                trust_hashes_on_open,
                resident_cap_bytes,
                checkpoint_every_blobs,
            } => {
                let disk = cas.disk_usage();
                format!(
                    "store: sqlite\nhandles: {}\nresident_bytes: {}\nsqlite_db_bytes: {}\nsqlite_wal_bytes: {}\nsqlite_shm_bytes: {}\nsqlite_total_bytes: {}\nsoft_cap_bytes: {}\nresident_cap_bytes: {}\nresident_evictions: {}\nwarm_bytes_on_open: {}\ntrust_hashes_on_open: {}\ncheckpoint_every_blobs: {}\nwal_checkpoints: {}\nwal_checkpoint_failures: {}\ndurability_lag: {}\ndropped_writes: {}\npersisted_blobs_this_run: {}\nretrieve_default_bytes: {}\nretrieve_max_bytes: {}",
                    cas.len(),
                    cas.mem_bytes(),
                    disk.db_bytes,
                    disk.wal_bytes,
                    disk.shm_bytes,
                    disk.total_bytes(),
                    soft_cap_bytes.map_or_else(|| "unset".to_string(), |n| n.to_string()),
                    resident_cap_bytes.map_or_else(|| "unset".to_string(), |n| n.to_string()),
                    cas.resident_evictions(),
                    warm_bytes_on_open,
                    trust_hashes_on_open,
                    checkpoint_every_blobs.map_or_else(|| "unset".to_string(), |n| n.to_string()),
                    cas.wal_checkpoints(),
                    cas.wal_checkpoint_failures(),
                    cas.durability_lag(),
                    cas.dropped_writes(),
                    cas.persisted(),
                    retrieve_limits.default_bytes,
                    retrieve_limits.max_bytes
                )
            }
        }
    }
}

impl Cas for Coffer {
    fn put(&self, bytes: &[u8]) -> ContentHash {
        self.put_bytes(bytes)
    }

    fn get(&self, hash: &ContentHash) -> Option<HeldBytes> {
        self.get_handle(hash.as_str())
    }

    fn len(&self) -> usize {
        self.handle_count()
    }
}

/// How many parsed datasets the query tools keep hot (`COFFER_MCP_DATASET_CACHE_ENTRIES`,
/// default 8; 0 disables retention — every query parses fresh). Read once at startup: the
/// cache lives for the process, unlike the per-call limits above.
pub(crate) fn dataset_cache_entries_from_env() -> usize {
    std::env::var("COFFER_MCP_DATASET_CACHE_ENTRIES")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(8)
}
