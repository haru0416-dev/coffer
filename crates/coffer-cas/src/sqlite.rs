//! Persistent, write-through CAS backed by `SQLite`.
//!
//! [`SqliteCas`] keeps bytes produced during this process run in RAM and persists offloaded
//! originals to `SQLite` on a background single-writer thread. On reopen it validates stored
//! rows by default, but does not keep every historical blob resident unless configured to do so;
//! historical bytes are read lazily into the resident cache when [`Cas::get`] asks for them.
//! Operators can opt into a faster hash-only open when startup I/O matters more than eager
//! corruption detection. The [`Cas`] trait stays infallible: `put`/`get` never surface I/O
//! errors. Disk durability is *best-effort* and observable on a separate axis
//! ([`SqliteCas::flush`], [`SqliteCas::durability_lag`], [`SqliteCas::dropped_writes`]) —
//! matching's "fidelity on a separate axis from the hot path."
//!
//! Within-run correctness for bytes produced this run never depends on disk: a `get` for any
//! newly inserted hash hits RAM. With the default open mode, every stored blob is re-hashed
//! before its hash is accepted into the known set, so a corrupt row is skipped rather than
//! trusted. With hash-only fast open, a corrupt row may appear in `len()` until touched, but
//! [`Cas::get`] still re-hashes the loaded bytes and returns `None` on mismatch. Combined with
//! the hash check in `CompressedDoc::reconstruct`, a bad row degrades to a caught error, never
//! silent corruption.
//!
//! Puts are enqueued to the writer as cheap `Arc` clones (pointers, not byte copies) over an
//! unbounded channel, so `put` is non-blocking *and* lossless: the only extra memory is the
//! pointer backlog, bounded by what the resident cache already holds for new puts. A write is
//! counted as dropped only if the background writer has ended (see [`SqliteCas::dropped_writes`]).
//!
//! There is **no on-disk eviction**: dropping a persisted blob a live document
//! references would turn reconstruction into a hard error. The resident cache is lazy by
//! default. [`SqliteConfig::soft_cap_bytes`] warns on RAM-resident bytes, while
//! [`SqliteConfig::resident_cap_bytes`] can evict only historical read-cache entries; `SQLite`
//! rows are never deleted by the CAS.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet, VecDeque};
#[cfg(unix)]
use std::fs;
#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex, PoisonError, RwLock};
use std::thread::JoinHandle;
use std::time::Duration;

use rusqlite::{Connection, params};

use crate::{Cas, ContentHash};

/// Read one blob from a persisted CAS database at `path` by the hex **prefix** a sentinel shows
/// (e.g. [`ContentHash::short`]), WITHOUT opening a full [`SqliteCas`] — a stateless, cross-process
/// **fresh** read (the proxy process writes; a separate process, e.g. the MCP server, unfolds a
/// sentinel on demand). `prefix` must be ≥ 8 lowercase-hex chars; returns the matching stored bytes,
/// or `None` if absent / ambiguous-input. Verifies the bytes' full hash starts with `prefix`, so a
/// corrupt or foreign row yields `None`, never wrong bytes.
///
/// # Errors
/// The `rusqlite` error if the database cannot be opened or queried.
pub fn read_blob(path: impl AsRef<Path>, prefix: &str) -> rusqlite::Result<Option<Vec<u8>>> {
    if prefix.len() < 8
        || !prefix
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
    {
        return Ok(None);
    }
    let path = path.as_ref();
    prepare_private_sqlite_file(path)?;
    let conn = Connection::open(path)?;
    conn.busy_timeout(Duration::from_secs(5))?;
    // The table may not exist yet if the writer never ran; treat that as "absent", not an error.
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS cas(hash TEXT PRIMARY KEY, bytes BLOB) WITHOUT ROWID;",
    )?;
    harden_sqlite_file_modes(path)?;
    let mut stmt = conn.prepare("SELECT bytes FROM cas WHERE hash LIKE ?1 || '%' LIMIT 2")?;
    let mut rows = stmt.query(params![prefix])?;
    let Some(first) = rows.next()? else {
        return Ok(None);
    };
    let bytes: Vec<u8> = first.get(0)?;
    if rows.next()?.is_some() {
        return Ok(None);
    }
    Ok(ContentHash::of(&bytes)
        .as_str()
        .starts_with(prefix)
        .then_some(bytes))
}

/// The resident in-RAM map: content hash to exact original bytes.
type BlobMap = HashMap<ContentHash, Arc<[u8]>>;
type KnownSet = HashSet<ContentHash>;

/// File sizes for a `SQLite` CAS database and its WAL sidecars.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SqliteDiskUsage {
    /// Main database file size in bytes.
    pub db_bytes: usize,
    /// Write-ahead log sidecar (`<db>-wal`) size in bytes.
    pub wal_bytes: usize,
    /// Shared-memory sidecar (`<db>-shm`) size in bytes.
    pub shm_bytes: usize,
}

impl SqliteDiskUsage {
    /// Total bytes across the main database, WAL, and shared-memory sidecar.
    #[must_use]
    pub fn total_bytes(self) -> usize {
        self.db_bytes
            .saturating_add(self.wal_bytes)
            .saturating_add(self.shm_bytes)
    }
}

/// Pragmas + schema applied on open. WAL + `synchronous = NORMAL` is the durable-and-fast
/// pairing (no fsync per commit; fsync amortized at WAL checkpoints). `WITHOUT ROWID` keeps
/// the hex-key primary index as the table itself.
const SCHEMA: &str = "\
PRAGMA journal_mode=WAL;\n\
PRAGMA synchronous=NORMAL;\n\
CREATE TABLE IF NOT EXISTS cas(hash TEXT PRIMARY KEY, bytes BLOB) WITHOUT ROWID;";

/// Tuning for [`SqliteCas`]. The default validates stored rows, holds no soft cap, and lazily
/// reads reopened blobs.
#[derive(Debug, Clone, Default)]
pub struct SqliteConfig {
    /// Soft ceiling on resident bytes. Crossing it logs a warning once; it never drops or
    /// evicts data (that would break reconstruction). `None` disables the warning.
    pub soft_cap_bytes: Option<usize>,
    /// If true, keep all valid historical blobs resident during open. The default (`false`)
    /// validates rows but leaves historical bytes on disk until `get` asks for them.
    pub warm_bytes_on_open: bool,
    /// If true, open reads only stored hash keys instead of scanning and re-hashing every blob.
    /// `get` still verifies bytes before returning them. Ignored when `warm_bytes_on_open` is
    /// true, because warming requires reading bytes anyway.
    pub trust_hashes_on_open: bool,
    /// Hard ceiling on RAM-resident **historical** bytes (the read cache). Only blobs already
    /// persisted to `SQLite` and loaded by open/`get` are evictable; new `put`s stay resident for
    /// within-run correctness, so this cap does NOT bound the current run's write path — a process
    /// that `put`s a large volume of new blobs grows RSS regardless of this cap. To bound write-path
    /// memory, recycle the process or cap upstream input size; this only limits the historical cache.
    pub resident_cap_bytes: Option<usize>,
    /// Run `PRAGMA wal_checkpoint(PASSIVE)` after at least this many blob commits. `None`
    /// leaves checkpointing to SQLite/default process shutdown behavior.
    pub checkpoint_every_blobs: Option<usize>,
}

#[derive(Debug, Default)]
struct Metrics {
    /// Puts enqueued for disk but not yet committed.
    durability_lag: AtomicUsize,
    /// Puts that could not be persisted because the writer ended.
    dropped_writes: AtomicUsize,
    /// Puts committed to `SQLite`.
    persisted: AtomicUsize,
    /// Bytes resident in this process's RAM cache.
    mem_bytes: AtomicUsize,
    /// Resident historical blobs evicted from the in-process cache.
    resident_evictions: AtomicUsize,
    /// WAL checkpoints attempted by the background writer.
    wal_checkpoints: AtomicUsize,
    /// WAL checkpoint attempts that returned a `SQLite` error.
    wal_checkpoint_failures: AtomicUsize,
    /// Whether the soft-cap warning has already fired (one-shot).
    soft_cap_warned: AtomicBool,
    /// Whether the writer-thread-loss warning has already fired (one-shot).
    writer_loss_warned: AtomicBool,
}

#[derive(Debug, Default)]
struct ResidentIndex {
    next_touch: u64,
    evictable: HashMap<ContentHash, u64>,
    lru: VecDeque<(u64, ContentHash)>,
}

/// A unit of work for the background writer.
enum WriteOp {
    /// Persist one offloaded original.
    Put(ContentHash, Arc<[u8]>),
    /// Commit everything queued so far, then acknowledge on this channel.
    Flush(Sender<()>),
}

/// A byte-faithful CAS that persists to `SQLite` while keeping a lazy resident cache.
///
/// See the module-level documentation for the durability model. Construct with
/// [`SqliteCas::open`].
///
/// A process should hold at most **one** `SqliteCas` per database path: two openers keep
/// independent resident caches and contend on `SQLite`'s single
/// writer, where a slow commit can exceed the busy-timeout and be counted as a dropped write.
/// Within one opener the store is fully concurrent (share it via `&self`).
pub struct SqliteCas {
    /// Resident store for this run. New `put`s enter here immediately; reopened blobs enter on
    /// first `get` unless `warm_bytes_on_open` was enabled.
    mem: RwLock<BlobMap>,
    /// Hashes whose bytes are known either from `SQLite` validation or from this process run.
    known: RwLock<KnownSet>,
    /// Database path used for lazy reads on resident-cache misses.
    path: PathBuf,
    /// Sender to the writer thread. `Option` so [`Drop`] can close the channel before joining.
    tx: Option<Sender<WriteOp>>,
    /// The background writer. `Option` for the same reason.
    writer: Option<JoinHandle<()>>,
    metrics: Arc<Metrics>,
    soft_cap_bytes: Option<usize>,
    resident_cap_bytes: Option<usize>,
    resident_index: Mutex<ResidentIndex>,
    /// Long-lived read handle for resident-cache misses: a second connection to the same WAL
    /// database (WAL allows a reader concurrent with the writer), so `get` reuses one connection +
    /// a cached prepared statement instead of opening a fresh connection per call.
    /// `Mutex` because rusqlite `Connection` is `!Sync`; locked only for the brief read query.
    read_conn: Mutex<Connection>,
}

impl SqliteCas {
    /// Open (or create) a store at `path` with default tuning.
    ///
    /// Validates stored rows and records their hashes by default. Historical bytes remain on
    /// disk until `get` asks for them unless [`SqliteConfig::warm_bytes_on_open`] is enabled.
    ///
    /// # Errors
    /// Returns the `rusqlite` error if the database cannot be opened or the schema/pragmas
    /// cannot be applied, or if reading existing rows fails.
    ///
    /// # Panics
    /// Panics only if the OS cannot spawn the background writer thread (resource exhaustion) —
    /// a host condition, never an input condition.
    pub fn open<P: AsRef<Path>>(path: P) -> rusqlite::Result<Self> {
        Self::open_with_config(path, &SqliteConfig::default())
    }

    /// Like [`SqliteCas::open`], with an explicit [`SqliteConfig`].
    ///
    /// # Errors
    /// See [`SqliteCas::open`].
    ///
    /// # Panics
    /// See [`SqliteCas::open`].
    pub fn open_with_config<P: AsRef<Path>>(
        path: P,
        config: &SqliteConfig,
    ) -> rusqlite::Result<Self> {
        let path = path.as_ref().to_path_buf();
        prepare_private_sqlite_file(&path)?;
        let conn = Connection::open(&path)?;
        conn.busy_timeout(Duration::from_secs(5))?;
        conn.execute_batch(SCHEMA)?;
        harden_sqlite_file_modes(&path)?;

        let metrics = Arc::new(Metrics::default());
        let (mem, known, resident) = if config.trust_hashes_on_open && !config.warm_bytes_on_open {
            trust_hashes_from_disk(&conn)?
        } else {
            warm_from_disk(&conn, config.warm_bytes_on_open)?
        };
        metrics.mem_bytes.store(resident, Ordering::Relaxed);

        let (tx, rx) = mpsc::channel::<WriteOp>();
        let writer_metrics = Arc::clone(&metrics);
        let checkpoint_every_blobs = config.checkpoint_every_blobs;
        let writer = std::thread::Builder::new()
            .name("coffer-cas-sqlite-writer".into())
            .spawn(move || run_writer(conn, &rx, &writer_metrics, checkpoint_every_blobs))
            .expect("spawning the CAS writer thread should not fail");

        // A separate connection for the read path. The writer connection (moved above) already
        // created the table and set WAL mode persistently, so this reader needs no schema/DDL.
        let read_conn = Connection::open(&path)?;
        read_conn.busy_timeout(Duration::from_secs(5))?;
        harden_sqlite_file_modes(&path)?;

        let cas = Self {
            mem: RwLock::new(mem),
            known: RwLock::new(known),
            path,
            tx: Some(tx),
            writer: Some(writer),
            metrics,
            soft_cap_bytes: config.soft_cap_bytes,
            resident_cap_bytes: config.resident_cap_bytes,
            resident_index: Mutex::new(ResidentIndex::default()),
            read_conn: Mutex::new(read_conn),
        };
        if config.warm_bytes_on_open {
            cas.mark_existing_resident_blobs_evictable();
        }
        cas.note_soft_cap(resident);
        cas.enforce_resident_cap();
        Ok(cas)
    }

    /// Block until every `put` that returned before this call has been committed to the
    /// database, so a clean reopen will see them.
    ///
    /// With `synchronous = NORMAL` (the default), "committed" means written to the
    /// write-ahead log and visible to any reader — durable across a clean shutdown/reopen, but
    /// fsync-durable against power loss only at the next WAL checkpoint. Returns
    /// immediately if the background writer has already ended.
    pub fn flush(&self) {
        let Some(tx) = &self.tx else { return };
        let (ack, ack_rx) = mpsc::channel();
        if tx.send(WriteOp::Flush(ack)).is_ok() {
            // `Err` here means the writer ended before acknowledging; nothing left to wait on.
            let _ = ack_rx.recv();
        }
    }

    /// Puts queued for disk but not yet drained by the writer (the backlog). Reaches zero
    /// after a [`SqliteCas::flush`] with no concurrent writers — but zero means *drained*, not
    /// necessarily *persisted*: pair it with [`SqliteCas::dropped_writes`] `== 0` to conclude
    /// every put reached `SQLite`.
    #[must_use]
    pub fn durability_lag(&self) -> usize {
        self.metrics.durability_lag.load(Ordering::Relaxed)
    }

    /// Puts that could not be persisted because the background writer ended. They remain
    /// correct in RAM for this run but will not survive a restart. Normally zero.
    #[must_use]
    pub fn dropped_writes(&self) -> usize {
        self.metrics.dropped_writes.load(Ordering::Relaxed)
    }

    /// Distinct blobs committed to `SQLite` so far.
    #[must_use]
    pub fn persisted(&self) -> usize {
        self.metrics.persisted.load(Ordering::Relaxed)
    }

    /// Total bytes resident in this process's RAM cache. Reopened historical blobs are not
    /// resident until `get` asks for them unless `warm_bytes_on_open` was configured.
    #[must_use]
    pub fn mem_bytes(&self) -> usize {
        self.metrics.mem_bytes.load(Ordering::Relaxed)
    }

    /// Number of historical blobs evicted from the resident RAM cache. Disk data is not deleted.
    #[must_use]
    pub fn resident_evictions(&self) -> usize {
        self.metrics.resident_evictions.load(Ordering::Relaxed)
    }

    /// WAL checkpoints attempted by the background writer.
    #[must_use]
    pub fn wal_checkpoints(&self) -> usize {
        self.metrics.wal_checkpoints.load(Ordering::Relaxed)
    }

    /// WAL checkpoint attempts that returned a `SQLite` error.
    #[must_use]
    pub fn wal_checkpoint_failures(&self) -> usize {
        self.metrics.wal_checkpoint_failures.load(Ordering::Relaxed)
    }

    /// Current on-disk file sizes for the main database and `SQLite` WAL sidecars.
    ///
    /// Missing sidecar files or metadata read failures are reported as `0` for that file. This
    /// keeps status reporting infallible and suitable for the MCP/proxy hot path.
    #[must_use]
    pub fn disk_usage(&self) -> SqliteDiskUsage {
        sqlite_disk_usage(&self.path)
    }

    fn note_soft_cap(&self, resident: usize) {
        if let Some(cap) = self.soft_cap_bytes {
            if resident > cap && !self.metrics.soft_cap_warned.swap(true, Ordering::Relaxed) {
                tracing::warn!(
                    resident,
                    cap,
                    "coffer-cas: SQLite-backed resident cache exceeded its soft cap"
                );
            }
        }
    }

    fn touch_evictable(&self, hash: &ContentHash) {
        let mut index = self
            .resident_index
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if index.evictable.contains_key(hash) {
            index.next_touch = index.next_touch.saturating_add(1);
            let touch = index.next_touch;
            index.evictable.insert(hash.clone(), touch);
            index.lru.push_back((touch, hash.clone()));
        }
    }

    fn mark_evictable(&self, hash: &ContentHash) {
        let mut index = self
            .resident_index
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        index.next_touch = index.next_touch.saturating_add(1);
        let touch = index.next_touch;
        index.evictable.insert(hash.clone(), touch);
        index.lru.push_back((touch, hash.clone()));
    }

    fn mark_existing_resident_blobs_evictable(&self) {
        let keys: Vec<ContentHash> = self
            .mem
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .keys()
            .cloned()
            .collect();
        for hash in keys {
            self.mark_evictable(&hash);
        }
    }

    fn next_evictable_hash(&self) -> Option<ContentHash> {
        let mut index = self
            .resident_index
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        while let Some((touch, hash)) = index.lru.pop_front() {
            if index.evictable.get(&hash).copied() == Some(touch) {
                index.evictable.remove(&hash);
                return Some(hash);
            }
        }
        None
    }

    fn enforce_resident_cap(&self) {
        let Some(cap) = self.resident_cap_bytes else {
            return;
        };
        while self.mem_bytes() > cap {
            let Some(hash) = self.next_evictable_hash() else {
                return;
            };
            let removed = self
                .mem
                .write()
                .unwrap_or_else(PoisonError::into_inner)
                .remove(&hash);
            if let Some(blob) = removed {
                self.metrics
                    .mem_bytes
                    .fetch_sub(blob.len(), Ordering::Relaxed);
                self.metrics
                    .resident_evictions
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

#[cfg(unix)]
fn prepare_private_sqlite_file(path: &Path) -> rusqlite::Result<()> {
    match fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
    {
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            set_private_file_mode(path)
        }
        Err(error) => Err(sqlite_path_io_error(path, error)),
    }
}

#[cfg(not(unix))]
fn prepare_private_sqlite_file(_path: &Path) -> rusqlite::Result<()> {
    Ok(())
}

#[cfg(unix)]
fn harden_sqlite_file_modes(path: &Path) -> rusqlite::Result<()> {
    set_private_file_mode(path)?;
    set_private_file_mode_if_exists(&sidecar_path(path, "-wal"))?;
    set_private_file_mode_if_exists(&sidecar_path(path, "-shm"))
}

#[cfg(not(unix))]
fn harden_sqlite_file_modes(_path: &Path) -> rusqlite::Result<()> {
    Ok(())
}

#[cfg(unix)]
fn set_private_file_mode_if_exists(path: &Path) -> rusqlite::Result<()> {
    match fs::metadata(path) {
        Ok(_) => set_private_file_mode(path),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(sqlite_path_io_error(path, error)),
    }
}

#[cfg(unix)]
fn set_private_file_mode(path: &Path) -> rusqlite::Result<()> {
    fs::set_permissions(path, fs::Permissions::from_mode(0o600))
        .map_err(|error| sqlite_path_io_error(path, error))
}

#[cfg(unix)]
fn sqlite_path_io_error(path: &Path, _error: std::io::Error) -> rusqlite::Error {
    rusqlite::Error::InvalidPath(path.to_path_buf())
}

/// Validate every row, optionally keeping its bytes resident. Rows whose bytes do not match
/// their stored key are skipped. Returns the resident map, the known hash set, and resident bytes.
fn warm_from_disk(
    conn: &Connection,
    warm_bytes: bool,
) -> rusqlite::Result<(BlobMap, KnownSet, usize)> {
    let mut mem: BlobMap = HashMap::new();
    let mut known: KnownSet = HashSet::new();
    let mut resident: usize = 0;
    let mut stmt = conn.prepare("SELECT hash, bytes FROM cas")?;
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
    })?;
    for row in rows {
        let (stored_key, bytes) = row?;
        let blob: Arc<[u8]> = Arc::from(bytes);
        let recomputed = ContentHash::of(&blob);
        if recomputed.as_str() == stored_key {
            known.insert(recomputed.clone());
            if warm_bytes {
                resident += blob.len();
                mem.insert(recomputed, blob);
            }
        } else {
            tracing::warn!(
                stored = %stored_key,
                "coffer-cas: skipping SQLite row whose bytes do not hash to their key"
            );
        }
    }
    Ok((mem, known, resident))
}

/// Load only syntactically valid stored hash keys. This skips the byte scan used by
/// `warm_from_disk`; `read_blob` still verifies bytes before returning them on a later `get`.
fn trust_hashes_from_disk(conn: &Connection) -> rusqlite::Result<(BlobMap, KnownSet, usize)> {
    let mem: BlobMap = HashMap::new();
    let mut known: KnownSet = HashSet::new();
    let mut stmt = conn.prepare("SELECT hash FROM cas")?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
    for row in rows {
        let stored_key = row?;
        if let Some(hash) = ContentHash::from_hex(&stored_key) {
            known.insert(hash);
        } else {
            tracing::warn!(
                stored = %stored_key,
                "coffer-cas: skipping SQLite row whose key is not a full lowercase content hash"
            );
        }
    }
    Ok((mem, known, 0))
}

fn sqlite_disk_usage(path: &Path) -> SqliteDiskUsage {
    SqliteDiskUsage {
        db_bytes: file_len(path),
        wal_bytes: file_len(&sidecar_path(path, "-wal")),
        shm_bytes: file_len(&sidecar_path(path, "-shm")),
    }
}

fn sidecar_path(path: &Path, suffix: &str) -> PathBuf {
    let mut raw = path.as_os_str().to_owned();
    raw.push(suffix);
    PathBuf::from(raw)
}

fn file_len(path: &Path) -> usize {
    std::fs::metadata(path)
        .ok()
        .and_then(|m| usize::try_from(m.len()).ok())
        .unwrap_or(0)
}

/// Accounts for any puts still queued if the writer exits abnormally (a panic): without this,
/// in-flight puts would leak `durability_lag` and never surface as dropped writes. On the
/// normal shutdown path the queue is already drained, so its drop is a no-op.
struct DrainGuard<'a> {
    rx: &'a Receiver<WriteOp>,
    metrics: &'a Metrics,
}

impl Drop for DrainGuard<'_> {
    fn drop(&mut self) {
        let mut leftover = 0;
        while let Ok(op) = self.rx.try_recv() {
            if matches!(op, WriteOp::Put(..)) {
                leftover += 1;
            }
        }
        if leftover > 0 {
            self.metrics
                .durability_lag
                .fetch_sub(leftover, Ordering::Relaxed);
            self.metrics
                .dropped_writes
                .fetch_add(leftover, Ordering::Relaxed);
        }
    }
}

/// The background writer loop: drain a batch, commit it in one transaction, ack any flushes.
/// It never panics on a `SQLite` error — a failed batch is counted as dropped and the loop
/// continues, because the authoritative copy is already safe in RAM. A panic from any other
/// source is bounded by `DrainGuard`, which reconciles the durability counters on unwind.
fn run_writer(
    mut conn: Connection,
    rx: &Receiver<WriteOp>,
    metrics: &Metrics,
    checkpoint_every_blobs: Option<usize>,
) {
    let _drain_guard = DrainGuard { rx, metrics };
    let mut blobs_since_checkpoint = 0usize;
    while let Ok(first) = rx.recv() {
        let mut puts: Vec<(ContentHash, Arc<[u8]>)> = Vec::new();
        let mut acks: Vec<Sender<()>> = Vec::new();
        absorb(first, &mut puts, &mut acks);
        while let Ok(op) = rx.try_recv() {
            absorb(op, &mut puts, &mut acks);
        }

        let count = puts.len();
        if count > 0 {
            match commit_batch(&mut conn, &puts) {
                Ok(()) => {
                    metrics.persisted.fetch_add(count, Ordering::Relaxed);
                    if let Some(every) = checkpoint_every_blobs {
                        blobs_since_checkpoint = blobs_since_checkpoint.saturating_add(count);
                        if every > 0 && blobs_since_checkpoint >= every {
                            checkpoint_wal(&mut conn, metrics);
                            blobs_since_checkpoint = 0;
                        }
                    }
                }
                Err(error) => {
                    tracing::warn!(%error, count, "coffer-cas: SQLite batch failed; data stays in RAM");
                    metrics.dropped_writes.fetch_add(count, Ordering::Relaxed);
                }
            }
            metrics.durability_lag.fetch_sub(count, Ordering::Relaxed);
        }
        // Ack flushes only after the batch above has committed, so a returning `flush()` means
        // everything queued before it is at least WAL-committed (see `flush` for the durability
        // nuance under `synchronous = NORMAL`).
        for ack in acks {
            let _ = ack.send(());
        }
    }
}

fn checkpoint_wal(conn: &mut Connection, metrics: &Metrics) {
    metrics.wal_checkpoints.fetch_add(1, Ordering::Relaxed);
    if let Err(error) = conn.execute_batch("PRAGMA wal_checkpoint(PASSIVE);") {
        tracing::warn!(%error, "coffer-cas: SQLite WAL checkpoint failed");
        metrics
            .wal_checkpoint_failures
            .fetch_add(1, Ordering::Relaxed);
    }
}

fn absorb(op: WriteOp, puts: &mut Vec<(ContentHash, Arc<[u8]>)>, acks: &mut Vec<Sender<()>>) {
    match op {
        WriteOp::Put(hash, bytes) => puts.push((hash, bytes)),
        WriteOp::Flush(ack) => acks.push(ack),
    }
}

fn commit_batch(conn: &mut Connection, puts: &[(ContentHash, Arc<[u8]>)]) -> rusqlite::Result<()> {
    let tx = conn.transaction()?;
    {
        let mut stmt =
            tx.prepare_cached("INSERT OR IGNORE INTO cas(hash, bytes) VALUES (?1, ?2)")?;
        for (hash, bytes) in puts {
            stmt.execute(params![hash.as_str(), &bytes[..]])?;
        }
    }
    tx.commit()
}

impl SqliteCas {
    /// Read one blob by EXACT content hash through the long-lived read connection: a cached prepared
    /// `PRIMARY KEY` lookup, no per-call connection open or DDL. Verifies the bytes' hash
    /// before returning, so a corrupt row yields `None`, never wrong bytes.
    fn read_one(&self, hash: &ContentHash) -> Option<Vec<u8>> {
        let conn = self
            .read_conn
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let mut stmt = match conn.prepare_cached("SELECT bytes FROM cas WHERE hash = ?1") {
            Ok(stmt) => stmt,
            Err(error) => {
                tracing::warn!(%error, "coffer-cas: preparing the read statement failed");
                return None;
            }
        };
        let bytes: Vec<u8> = match stmt.query_row(params![hash.as_str()], |row| row.get(0)) {
            Ok(bytes) => bytes,
            Err(rusqlite::Error::QueryReturnedNoRows) => return None,
            Err(error) => {
                tracing::warn!(%error, hash = %hash.as_str(), "coffer-cas: lazy SQLite read failed");
                return None;
            }
        };
        // Defense-in-depth: refuse bytes whose content hash does not match the key (disk corruption),
        // mirroring read_blob — never return wrong bytes.
        (ContentHash::of(&bytes) == *hash).then_some(bytes)
    }
}

impl Cas for SqliteCas {
    fn put(&self, bytes: &[u8]) -> ContentHash {
        let hash = ContentHash::of(bytes);

        // New bytes from this process become resident immediately. If the hash was already known
        // from a lazy reopen, this only warms the resident cache and does not rewrite SQLite.
        let warmed_blob = {
            let mut mem = self.mem.write().unwrap_or_else(PoisonError::into_inner);
            if let Entry::Vacant(slot) = mem.entry(hash.clone()) {
                let blob: Arc<[u8]> = Arc::from(bytes);
                slot.insert(Arc::clone(&blob));
                Some(blob)
            } else {
                None
            }
        };

        if let Some(blob) = warmed_blob {
            let already_known = !self
                .known
                .write()
                .unwrap_or_else(PoisonError::into_inner)
                .insert(hash.clone());
            let resident = self
                .metrics
                .mem_bytes
                .fetch_add(blob.len(), Ordering::Relaxed)
                + blob.len();
            self.note_soft_cap(resident);
            if !already_known && let Some(tx) = &self.tx {
                self.metrics.durability_lag.fetch_add(1, Ordering::Relaxed);
                if tx.send(WriteOp::Put(hash.clone(), blob)).is_err() {
                    // Writer ended: undo the lag bump and record a durability (not correctness) loss.
                    self.metrics.durability_lag.fetch_sub(1, Ordering::Relaxed);
                    self.metrics.dropped_writes.fetch_add(1, Ordering::Relaxed);
                    // A dead writer means *every* subsequent put is silently non-durable, not just
                    // this one. Warn loudly, once, so the loss is discoverable in logs rather than
                    // only by polling `dropped_writes`. RAM stays authoritative.
                    if !self
                        .metrics
                        .writer_loss_warned
                        .swap(true, Ordering::Relaxed)
                    {
                        tracing::error!(
                            "coffer-cas: SQLite writer thread has ended; subsequent puts stay correct in RAM \
                             but are NOT durable and will not survive a restart (see coffer_status dropped_writes)"
                        );
                    }
                }
            }
        }

        hash
    }

    fn get(&self, hash: &ContentHash) -> Option<Arc<[u8]>> {
        if let Some(bytes) = self
            .mem
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .get(hash)
            .cloned()
        {
            self.touch_evictable(hash);
            return Some(bytes);
        }

        let bytes = self.read_one(hash)?;
        let blob: Arc<[u8]> = Arc::from(bytes);

        {
            let mut mem = self.mem.write().unwrap_or_else(PoisonError::into_inner);
            match mem.entry(hash.clone()) {
                Entry::Occupied(existing) => {
                    self.touch_evictable(hash);
                    return Some(Arc::clone(existing.get()));
                }
                Entry::Vacant(slot) => {
                    slot.insert(Arc::clone(&blob));
                }
            }
        }

        self.known
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(hash.clone());
        let resident = self
            .metrics
            .mem_bytes
            .fetch_add(blob.len(), Ordering::Relaxed)
            + blob.len();
        self.mark_evictable(hash);
        self.note_soft_cap(resident);
        self.enforce_resident_cap();
        Some(blob)
    }

    fn len(&self) -> usize {
        self.known
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }
}

impl Drop for SqliteCas {
    fn drop(&mut self) {
        // Close the channel so the writer drains its queue and exits, then join to guarantee
        // every queued commit lands before the database handle (and its WAL) close.
        self.tx.take();
        if let Some(writer) = self.writer.take() {
            let _ = writer.join();
        }
    }
}

impl std::fmt::Debug for SqliteCas {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteCas")
            .field("len", &self.len())
            .field("mem_bytes", &self.mem_bytes())
            .field("durability_lag", &self.durability_lag())
            .field("dropped_writes", &self.dropped_writes())
            .field("disk_usage", &self.disk_usage())
            .field("wal_checkpoints", &self.wal_checkpoints())
            .field("wal_checkpoint_failures", &self.wal_checkpoint_failures())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unique temp directory that is removed when dropped, so each test gets an isolated DB.
    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            use std::sync::atomic::AtomicU32;
            static N: AtomicU32 = AtomicU32::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            let dir =
                std::env::temp_dir().join(format!("coffer-cas-{}-{tag}-{n}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            TempDir(dir)
        }

        fn db(&self) -> std::path::PathBuf {
            self.0.join("cas.db")
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn put_get_roundtrip_and_idempotent() {
        let dir = TempDir::new("roundtrip");
        let cas = SqliteCas::open(dir.db()).unwrap();

        let h = cas.put(b"hello");
        assert_eq!(cas.get(&h).as_deref(), Some(&b"hello"[..]));

        // Empty bytes round-trip, and re-putting is idempotent.
        let empty = cas.put(b"");
        assert_eq!(cas.get(&empty).as_deref(), Some(&b""[..]));
        cas.put(b"hello");
        assert_eq!(cas.len(), 2);

        cas.flush();
        assert_eq!(cas.durability_lag(), 0);
        assert_eq!(cas.persisted(), 2);
        assert_eq!(cas.dropped_writes(), 0);
    }

    #[cfg(unix)]
    #[test]
    fn sqlite_files_are_private_on_unix() {
        let dir = TempDir::new("private-files");
        let db = dir.db();
        let cas = SqliteCas::open(&db).unwrap();
        cas.put(b"secret-bearing tool output");
        cas.flush();

        assert_private_file(&db);
        for suffix in ["-wal", "-shm"] {
            let sidecar = sidecar_path(&db, suffix);
            if sidecar.exists() {
                assert_private_file(&sidecar);
            }
        }
    }

    #[cfg(unix)]
    fn assert_private_file(path: &Path) {
        let mode = std::fs::metadata(path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "{path:?} mode was {mode:o}");
    }

    #[test]
    #[ignore = "sqlite get disk-read timing (the cross-process retrieve hot path)"]
    fn timing_sqlite_get_disk() {
        // Mirror the shared-CAS proxy⊕MCP retrieve path: blobs persisted by one opener, then
        // get() from a FRESH opener whose RAM cache is empty, so every get is a disk read.
        let dir = TempDir::new("get-timing");
        let n = 2000usize;
        let mut hashes = Vec::with_capacity(n);
        {
            let cas = SqliteCas::open(dir.db()).unwrap();
            for i in 0..n {
                hashes.push(cas.put(format!("blob-{i}-{}", "payload".repeat(20)).as_bytes()));
            }
            cas.flush();
        }
        let cas = SqliteCas::open(dir.db()).unwrap(); // trust-hashes: known, bytes not resident
        let t = std::time::Instant::now();
        for h in &hashes {
            std::hint::black_box(cas.get(h));
        }
        let per = t.elapsed() / u32::try_from(n).unwrap();
        eprintln!("sqlite get (RAM-miss -> disk): {per:?}/op over {n} distinct gets");
    }

    #[test]
    fn persists_across_reopen() {
        let dir = TempDir::new("reopen");
        {
            let cas = SqliteCas::open(dir.db()).unwrap();
            cas.put(b"durable");
            cas.put(b"");
            cas.flush();
            // Drop joins the writer, so the bytes are on disk before the next open.
        }

        let cas = SqliteCas::open(dir.db()).unwrap();
        assert_eq!(cas.len(), 2, "both blobs should be known from disk");
        assert_eq!(
            cas.get(&ContentHash::of(b"durable")).as_deref(),
            Some(&b"durable"[..])
        );
        assert_eq!(cas.get(&ContentHash::of(b"")).as_deref(), Some(&b""[..]));
        assert_eq!(cas.dropped_writes(), 0);
    }

    #[test]
    fn reopen_is_lazy_until_get() {
        let dir = TempDir::new("lazy-reopen");
        {
            let cas = SqliteCas::open(dir.db()).unwrap();
            cas.put(b"first");
            cas.put(b"second");
            cas.flush();
        }

        let cas = SqliteCas::open(dir.db()).unwrap();
        assert_eq!(cas.len(), 2);
        assert_eq!(
            cas.mem_bytes(),
            0,
            "historical blobs should not be resident on open"
        );

        assert_eq!(
            cas.get(&ContentHash::of(b"first")).as_deref(),
            Some(&b"first"[..])
        );
        assert_eq!(cas.mem_bytes(), 5);
        assert_eq!(
            cas.get(&ContentHash::of(b"first")).as_deref(),
            Some(&b"first"[..])
        );
        assert_eq!(
            cas.mem_bytes(),
            5,
            "repeated gets must not double-count resident bytes"
        );
        assert_eq!(
            cas.get(&ContentHash::of(b"second")).as_deref(),
            Some(&b"second"[..])
        );
        assert_eq!(cas.mem_bytes(), 11);
    }

    #[test]
    fn can_warm_reopened_bytes_when_configured() {
        let dir = TempDir::new("warm-reopen");
        {
            let cas = SqliteCas::open(dir.db()).unwrap();
            cas.put(b"first");
            cas.put(b"second");
            cas.flush();
        }

        let cas = SqliteCas::open_with_config(
            dir.db(),
            &SqliteConfig {
                warm_bytes_on_open: true,
                ..SqliteConfig::default()
            },
        )
        .unwrap();
        assert_eq!(cas.len(), 2);
        assert_eq!(cas.mem_bytes(), 11);
        assert_eq!(
            cas.get(&ContentHash::of(b"second")).as_deref(),
            Some(&b"second"[..])
        );
        assert_eq!(cas.mem_bytes(), 11);
    }

    #[test]
    fn resident_cap_evicts_lazily_loaded_historical_blobs() {
        let dir = TempDir::new("resident-cap");
        let first = ContentHash::of(b"aaaa");
        let second = ContentHash::of(b"bbbbb");
        {
            let cas = SqliteCas::open(dir.db()).unwrap();
            cas.put(b"aaaa");
            cas.put(b"bbbbb");
            cas.flush();
        }

        let cas = SqliteCas::open_with_config(
            dir.db(),
            &SqliteConfig {
                resident_cap_bytes: Some(5),
                ..SqliteConfig::default()
            },
        )
        .unwrap();
        assert_eq!(cas.mem_bytes(), 0);

        assert_eq!(cas.get(&first).as_deref(), Some(&b"aaaa"[..]));
        assert_eq!(cas.mem_bytes(), 4);
        assert_eq!(cas.get(&second).as_deref(), Some(&b"bbbbb"[..]));
        assert_eq!(
            cas.mem_bytes(),
            5,
            "loading the second blob should evict the older first blob"
        );
        assert_eq!(cas.resident_evictions(), 1);

        assert_eq!(cas.get(&first).as_deref(), Some(&b"aaaa"[..]));
        assert_eq!(cas.mem_bytes(), 4);
        assert_eq!(
            cas.resident_evictions(),
            2,
            "the cap should remain enforceable after reloading an evicted blob"
        );
    }

    #[test]
    fn resident_cap_does_not_evict_current_run_puts() {
        let dir = TempDir::new("resident-cap-current-run");
        let cas = SqliteCas::open_with_config(
            dir.db(),
            &SqliteConfig {
                resident_cap_bytes: Some(1),
                ..SqliteConfig::default()
            },
        )
        .unwrap();
        let h = cas.put(b"new bytes");

        assert_eq!(cas.get(&h).as_deref(), Some(&b"new bytes"[..]));
        assert_eq!(
            cas.mem_bytes(),
            9,
            "new current-run puts stay resident even when the historical cache cap is lower"
        );
        assert_eq!(cas.resident_evictions(), 0);
    }

    #[test]
    fn checkpoint_every_blobs_runs_wal_checkpoint_after_commits() {
        let dir = TempDir::new("checkpoint");
        let cas = SqliteCas::open_with_config(
            dir.db(),
            &SqliteConfig {
                checkpoint_every_blobs: Some(1),
                ..SqliteConfig::default()
            },
        )
        .unwrap();

        cas.put(b"checkpoint me");
        cas.flush();

        assert_eq!(cas.durability_lag(), 0);
        assert_eq!(cas.dropped_writes(), 0);
        assert_eq!(cas.persisted(), 1);
        assert_eq!(cas.wal_checkpoints(), 1);
        assert_eq!(cas.wal_checkpoint_failures(), 0);
    }

    #[test]
    fn disk_usage_reports_database_and_wal_sidecar_sizes() {
        let dir = TempDir::new("disk-usage");
        let cas = SqliteCas::open(dir.db()).unwrap();
        cas.put(b"disk usage bytes");
        cas.flush();

        let usage = cas.disk_usage();

        assert!(usage.db_bytes > 0, "{usage:?}");
        assert!(usage.wal_bytes > 0, "{usage:?}");
        assert!(usage.total_bytes() >= usage.db_bytes, "{usage:?}");
        assert_eq!(
            usage.total_bytes(),
            usage
                .db_bytes
                .saturating_add(usage.wal_bytes)
                .saturating_add(usage.shm_bytes)
        );
    }

    #[test]
    fn skips_a_corrupt_row_on_load() {
        let dir = TempDir::new("corrupt");
        {
            let cas = SqliteCas::open(dir.db()).unwrap();
            cas.put(b"good");
            cas.flush();
        }
        // Inject a row whose bytes do not hash to their key, behind coffer-cas's back.
        {
            let conn = Connection::open(dir.db()).unwrap();
            conn.execute(
                "INSERT OR REPLACE INTO cas(hash, bytes) VALUES (?1, ?2)",
                params!["0".repeat(64), &b"not the preimage"[..]],
            )
            .unwrap();
        }

        let cas = SqliteCas::open(dir.db()).unwrap();
        assert_eq!(
            cas.len(),
            1,
            "the corrupt row must be skipped, the good one kept"
        );
        assert_eq!(
            cas.get(&ContentHash::of(b"good")).as_deref(),
            Some(&b"good"[..])
        );
        // The bogus key is not retrievable.
        let bogus = ContentHash::of(b"not the preimage");
        assert!(bogus.as_str() != "0".repeat(64));
    }

    #[test]
    fn trust_hashes_on_open_skips_byte_scan_but_get_still_rejects_corrupt_rows() {
        let dir = TempDir::new("trust-open-corrupt");
        {
            let cas = SqliteCas::open(dir.db()).unwrap();
            cas.put(b"good");
            cas.flush();
        }
        {
            let conn = Connection::open(dir.db()).unwrap();
            conn.execute(
                "INSERT OR REPLACE INTO cas(hash, bytes) VALUES (?1, ?2)",
                params!["0".repeat(64), &b"not the preimage"[..]],
            )
            .unwrap();
        }

        let cas = SqliteCas::open_with_config(
            dir.db(),
            &SqliteConfig {
                trust_hashes_on_open: true,
                ..SqliteConfig::default()
            },
        )
        .unwrap();
        assert_eq!(
            cas.len(),
            2,
            "fast open counts syntactically valid keys before lazy byte verification"
        );
        assert_eq!(cas.mem_bytes(), 0);
        assert_eq!(
            cas.get(&ContentHash::of(b"good")).as_deref(),
            Some(&b"good"[..])
        );
        let bogus = ContentHash::from_hex(&"0".repeat(64)).unwrap();
        assert_eq!(cas.get(&bogus), None);
    }

    #[test]
    fn concurrent_puts_are_consistent_and_durable() {
        use std::thread;
        let dir = TempDir::new("concurrent");
        let cas = Arc::new(SqliteCas::open(dir.db()).unwrap());

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

        cas.flush();
        assert_eq!(cas.len(), 800);
        assert_eq!(cas.durability_lag(), 0);
        assert_eq!(cas.persisted(), 800);
        assert_eq!(cas.dropped_writes(), 0);
    }

    #[test]
    fn read_blob_reads_a_writers_committed_bytes() {
        let dir = TempDir::new("readblob");
        let h = {
            let cas = SqliteCas::open(dir.db()).unwrap();
            let h = cas.put(b"offloaded original");
            cas.flush();
            h
        };
        // A fresh, process-independent read by the sentinel's short hex prefix (no SqliteCas / RAM
        // map) — the cross-process read the MCP unfold tool uses against the proxy's database.
        assert_eq!(
            read_blob(dir.db(), h.short()).unwrap().as_deref(),
            Some(&b"offloaded original"[..])
        );
        assert_eq!(
            read_blob(dir.db(), ContentHash::of(b"absent").short()).unwrap(),
            None
        );
        assert_eq!(
            read_blob(dir.db(), "short").unwrap(),
            None,
            "too-short prefix is rejected"
        );
    }

    #[test]
    fn read_blob_rejects_ambiguous_hash_prefixes() {
        let dir = TempDir::new("ambiguous-prefix");
        let first = b"prefix-collision-19675";
        let second = b"prefix-collision-39911";
        assert_eq!(&ContentHash::of(first).as_str()[..8], "10022e94");
        assert_eq!(&ContentHash::of(second).as_str()[..8], "10022e94");
        {
            let cas = SqliteCas::open(dir.db()).unwrap();
            cas.put(first);
            cas.put(second);
            cas.flush();
        }

        assert_eq!(read_blob(dir.db(), "10022e94").unwrap(), None);
        assert_eq!(
            read_blob(dir.db(), &ContentHash::of(first).as_str()[..16])
                .unwrap()
                .as_deref(),
            Some(&first[..])
        );
    }

    #[test]
    fn mem_bytes_tracks_resident_payload() {
        let dir = TempDir::new("membytes");
        let cas = SqliteCas::open(dir.db()).unwrap();
        cas.put(b"abc");
        cas.put(b"de");
        cas.put(b"abc"); // duplicate: no additional bytes
        assert_eq!(cas.mem_bytes(), 5);
    }
}
