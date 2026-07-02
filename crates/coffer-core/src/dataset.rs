//! A JSON array parsed **once**, so repeated queries against the same held bytes skip the
//! per-call parse that otherwise dominates every read-side operation (parsing a multi-MB
//! handle costs tens of milliseconds; the query over the parsed rows costs micro- to
//! low-milliseconds).
//!
//! Surfaces that answer many queries against one content-addressed handle (the MCP server,
//! the wrap gateway) hold a [`Dataset`] per handle instead of re-parsing per tool call.
//! Content addressing makes the cache trivially sound: a handle's bytes never change, so a
//! parsed `Dataset` never goes stale.
//!
//! On top of parse-once, [`Dataset::query_aggregate`] lazily builds a per-field **column
//! cache** (typed cells) the first time a field is touched, so repeated filters and
//! aggregates scan a flat `Vec` instead of doing a map lookup per row per call. The column
//! path preserves the row path's semantics *exactly* — including the
//! refuse-rather-than-guess contract for present-but-non-`f64` aggregation fields and
//! [`json_cmp`]'s cross-type comparison rules — and falls back to the raw row for any cell
//! or predicate shape outside its fast path. Equivalence with the byte-slice entry points
//! is property-tested (`tests/dataset_equiv.rs`).

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock, PoisonError, RwLock};

use serde_json::Value;

use crate::Op;
use crate::budget::json_cmp;
use crate::index::{
    Agg, GroupAggregate, Predicate, QueryResult, bucket_aggregate_rows, describe_rows, digest_rows,
    join_aggregate_rows, join_group_aggregate_rows, query_display,
};

/// A parsed JSON array of records with a lazy per-field column cache. See the module docs.
pub struct Dataset {
    rows: Vec<Value>,
    columns: RwLock<HashMap<String, Arc<Column>>>,
    /// [`Dataset::describe`] is a pure function of the rows — computed once, cloned after.
    describe_memo: OnceLock<Option<String>>,
    /// [`Dataset::digest`] memo, keyed by the query string (bounded; see `DIGEST_MEMO_CAP`).
    digest_memo: Mutex<HashMap<String, Option<String>>>,
}

/// Digest answers memoized per dataset. Queries come from a model, so their cardinality is
/// tiny in practice; past the cap, extra queries are computed but not remembered.
const DIGEST_MEMO_CAP: usize = 64;

/// One row's value of one field, typed for the fast paths. `Other` covers everything the
/// fast paths must not reason about themselves (bool, null, nested containers, and numbers
/// that are not `f64`-representable under `arbitrary_precision`); those cells re-consult
/// the raw row through [`json_cmp`], so their semantics cannot drift.
#[derive(Clone, Copy)]
enum Cell {
    /// Field absent (or the row is not an object) — a predicate on it fails, an aggregate skips it.
    Missing,
    /// A JSON number that is `f64`-representable.
    Num(f64),
    /// A JSON string, interned to a symbol.
    Sym(u32),
    /// Present but neither of the above — compared via the raw row, refused by aggregates.
    Other,
}

struct Column {
    cells: Vec<Cell>,
    /// Symbol table for `Cell::Sym` (index = symbol).
    interner: Vec<String>,
    /// Reverse lookup: string -> symbol, used to resolve an equality target once per query.
    syms: HashMap<String, u32>,
}

impl Column {
    fn build(rows: &[Value], field: &str) -> Self {
        let mut cells = Vec::with_capacity(rows.len());
        let mut interner: Vec<String> = Vec::new();
        let mut syms: HashMap<String, u32> = HashMap::new();
        for row in rows {
            let cell = match row.as_object().and_then(|o| o.get(field)) {
                None => Cell::Missing,
                Some(v) => {
                    if let Some(n) = v.as_f64() {
                        Cell::Num(n)
                    } else if let Some(s) = v.as_str() {
                        let sym = *syms.entry(s.to_string()).or_insert_with(|| {
                            interner.push(s.to_string());
                            (interner.len() - 1) as u32
                        });
                        Cell::Sym(sym)
                    } else {
                        Cell::Other
                    }
                }
            };
            cells.push(cell);
        }
        Self {
            cells,
            interner,
            syms,
        }
    }
}

/// `json_cmp`'s numeric branch, extracted for the column fast path.
// float_cmp allow: WHERE-equality of two directly-parsed JSON numbers is the intended
// semantics (mirrors `json_cmp`), not an accumulated-error comparison.
#[allow(clippy::float_cmp)]
fn cmp_f64(a: f64, op: Op, b: f64) -> bool {
    match op {
        Op::Eq => a == b,
        Op::Ne => a != b,
        Op::Gt => a > b,
        Op::Ge => a >= b,
        Op::Lt => a < b,
        Op::Le => a <= b,
    }
}

/// `json_cmp`'s string branch, extracted for the column fast path.
fn cmp_str(a: &str, op: Op, b: &str) -> bool {
    match op {
        Op::Eq => a == b,
        Op::Ne => a != b,
        Op::Gt => a > b,
        Op::Ge => a >= b,
        Op::Lt => a < b,
        Op::Le => a <= b,
    }
}

impl Dataset {
    /// Parse `input` as a top-level JSON array — the same acceptance rule as every
    /// byte-slice query entry point (`None` for anything else, including JSON objects).
    #[must_use]
    pub fn parse(input: &[u8]) -> Option<Self> {
        let v: Value = serde_json::from_slice(input).ok()?;
        match v {
            Value::Array(rows) => Some(Self::from_rows(rows)),
            _ => None,
        }
    }

    /// Wrap already-parsed records.
    #[must_use]
    pub fn from_rows(rows: Vec<Value>) -> Self {
        Self {
            rows,
            columns: RwLock::new(HashMap::new()),
            describe_memo: OnceLock::new(),
            digest_memo: Mutex::new(HashMap::new()),
        }
    }

    /// Number of records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Read-only view of the parsed records (for row-wise scans by surfaces).
    #[must_use]
    pub fn rows(&self) -> &[Value] {
        &self.rows
    }

    /// Whether the array is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// The column for `field`, built on first touch and cached for every later query.
    fn column(&self, field: &str) -> Arc<Column> {
        if let Some(c) = self
            .columns
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .get(field)
        {
            return Arc::clone(c);
        }
        let built = Arc::new(Column::build(&self.rows, field));
        Arc::clone(
            self.columns
                .write()
                .unwrap_or_else(PoisonError::into_inner)
                .entry(field.to_string())
                .or_insert(built),
        )
    }

    /// The raw field value of row `i` — the fallback the fast paths use for `Cell::Other`.
    fn raw_field<'a>(&'a self, i: usize, field: &str) -> Option<&'a Value> {
        self.rows[i].as_object().and_then(|o| o.get(field))
    }

    /// Exactly [`crate::query_aggregate`], minus the parse: same matching, same
    /// refuse-rather-than-guess aggregation, byte-identical display string.
    #[must_use]
    pub fn query_aggregate(&self, predicates: &[Predicate], agg: &Agg) -> Option<QueryResult> {
        let mut mask = vec![true; self.rows.len()];
        for p in predicates {
            self.apply_predicate(p, &mut mask);
        }
        let matched: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, keep)| keep.then_some(i))
            .collect();
        let value = self.agg_over(&matched, agg)?;
        Some(QueryResult {
            display: query_display(predicates, agg, matched.len(), value),
            value,
            matched,
        })
    }

    /// AND `p` into `mask`, using the column fast path where the target's type allows and
    /// deferring to [`json_cmp`] on the raw row for `Other` cells and non-string/non-number
    /// targets — so every cross-type rule stays centralized in `json_cmp`.
    fn apply_predicate(&self, p: &Predicate, mask: &mut [bool]) {
        // Neither a number nor a string target: no fast path, evaluate rows directly.
        if p.value.as_f64().is_none() && p.value.as_str().is_none() {
            for (i, keep) in mask.iter_mut().enumerate() {
                if *keep {
                    *keep = self
                        .raw_field(i, &p.field)
                        .is_some_and(|fv| json_cmp(fv, p.op, &p.value));
                }
            }
            return;
        }
        let col = self.column(&p.field);
        if let Some(target) = p.value.as_f64() {
            for (i, keep) in mask.iter_mut().enumerate() {
                if *keep {
                    *keep = match col.cells[i] {
                        Cell::Num(a) => cmp_f64(a, p.op, target),
                        // string vs number: `json_cmp` matches only `Ne` (full-value inequality)
                        Cell::Sym(_) => p.op == Op::Ne,
                        Cell::Other => self
                            .raw_field(i, &p.field)
                            .is_some_and(|fv| json_cmp(fv, p.op, &p.value)),
                        Cell::Missing => false,
                    };
                }
            }
            return;
        }
        let target = p.value.as_str().expect("checked above");
        let target_sym = col.syms.get(target).copied();
        let ordered = !matches!(p.op, Op::Eq | Op::Ne);
        for (i, keep) in mask.iter_mut().enumerate() {
            if *keep {
                *keep = match col.cells[i] {
                    Cell::Sym(s) if ordered => cmp_str(&col.interner[s as usize], p.op, target),
                    Cell::Sym(s) => (Some(s) == target_sym) == (p.op == Op::Eq),
                    // number vs string: `json_cmp` matches only `Ne`
                    Cell::Num(_) => p.op == Op::Ne,
                    Cell::Other => self
                        .raw_field(i, &p.field)
                        .is_some_and(|fv| json_cmp(fv, p.op, &p.value)),
                    Cell::Missing => false,
                };
            }
        }
    }

    /// Exactly `apply_agg` over the column: missing fields are skipped, any matched
    /// present-but-non-`f64` value refuses the whole query, empty refuses.
    fn agg_over(&self, matched: &[usize], agg: &Agg) -> Option<f64> {
        let field = match agg {
            Agg::Count => return Some(matched.len() as f64),
            Agg::Sum(f) | Agg::Mean(f) | Agg::Min(f) | Agg::Max(f) => f,
        };
        let col = self.column(field);
        let mut vals: Vec<f64> = Vec::new();
        for &i in matched {
            match col.cells[i] {
                Cell::Num(v) => vals.push(v),
                Cell::Missing => {}
                Cell::Sym(_) | Cell::Other => return None,
            }
        }
        if vals.is_empty() {
            return None;
        }
        Some(match agg {
            Agg::Sum(_) => vals.iter().sum(),
            Agg::Mean(_) => vals.iter().sum::<f64>() / vals.len() as f64,
            Agg::Min(_) => vals.iter().copied().fold(f64::INFINITY, f64::min),
            Agg::Max(_) => vals.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            Agg::Count => unreachable!("count handled above"),
        })
    }

    /// Exactly [`crate::describe`], minus the parse — and memoized: describe is a pure
    /// function of the rows, so repeat calls (every fact card, every schema question)
    /// return the first computation.
    #[must_use]
    pub fn describe(&self) -> Option<String> {
        self.describe_memo
            .get_or_init(|| describe_rows(&self.rows))
            .clone()
    }

    /// Exactly [`crate::digest`], minus the parse — memoized per query string.
    #[must_use]
    pub fn digest(&self, query: &str) -> Option<String> {
        if let Some(hit) = self
            .digest_memo
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(query)
        {
            return hit.clone();
        }
        let computed = digest_rows(&self.rows, query);
        let mut memo = self
            .digest_memo
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if memo.len() < DIGEST_MEMO_CAP {
            memo.insert(query.to_string(), computed.clone());
        }
        computed
    }

    /// Exactly [`crate::bucket_aggregate`], minus the parse.
    #[must_use]
    pub fn bucket_aggregate(
        &self,
        bucket_field: &str,
        width: f64,
        agg: &Agg,
    ) -> Option<GroupAggregate> {
        bucket_aggregate_rows(&self.rows, bucket_field, width, agg)
    }

    /// Exactly [`crate::join_aggregate`], minus the two parses.
    #[must_use]
    pub fn join_aggregate(
        &self,
        right: &Dataset,
        left_key: &str,
        right_key: &str,
        right_where: &[Predicate],
        agg: &Agg,
    ) -> Option<QueryResult> {
        join_aggregate_rows(
            &self.rows,
            &right.rows,
            left_key,
            right_key,
            right_where,
            agg,
        )
    }

    /// Exactly [`crate::join_group_aggregate`], minus the two parses.
    #[must_use]
    pub fn join_group_aggregate(
        &self,
        right: &Dataset,
        left_key: &str,
        right_key: &str,
        group_field: &str,
        agg: &Agg,
    ) -> Option<GroupAggregate> {
        join_group_aggregate_rows(
            &self.rows,
            &right.rows,
            left_key,
            right_key,
            group_field,
            agg,
        )
    }
}

/// A small LRU of parsed [`Dataset`]s keyed by handle, shared by the surfaces (MCP server,
/// wrap gateway) so N tool calls against one held blob parse it once instead of N times.
/// Content addressing makes entries immutable — eviction is purely a memory cap, never an
/// invalidation concern. A capacity of 0 disables retention (each lookup parses fresh,
/// still returning a column-accelerated `Dataset`).
pub struct DatasetCache {
    cap: usize,
    inner: Mutex<CacheInner>,
}

#[derive(Default)]
struct CacheInner {
    map: HashMap<String, Arc<Dataset>>,
    /// Recency order, least-recently-used at the front.
    order: VecDeque<String>,
}

impl DatasetCache {
    /// A cache holding at most `cap` parsed datasets.
    #[must_use]
    pub fn new(cap: usize) -> Self {
        Self {
            cap,
            inner: Mutex::new(CacheInner::default()),
        }
    }

    /// The dataset for `key`, parsing `bytes` on a miss. `None` iff `bytes` is not a
    /// top-level JSON array — the same acceptance as [`Dataset::parse`], never cached.
    #[must_use]
    pub fn get_or_parse(&self, key: &str, bytes: &[u8]) -> Option<Arc<Dataset>> {
        if self.cap == 0 {
            return Dataset::parse(bytes).map(Arc::new);
        }
        {
            let mut inner = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
            if let Some(ds) = inner.map.get(key).map(Arc::clone) {
                if let Some(pos) = inner.order.iter().position(|k| k == key) {
                    inner.order.remove(pos);
                }
                inner.order.push_back(key.to_string());
                return Some(ds);
            }
        }
        // Parse outside the lock: a concurrent duplicate parse is benign (last insert wins);
        // holding the lock across a multi-MB parse would serialize every other handle.
        let ds = Arc::new(Dataset::parse(bytes)?);
        let mut inner = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        if !inner.map.contains_key(key) {
            inner.map.insert(key.to_string(), Arc::clone(&ds));
            inner.order.push_back(key.to_string());
            while inner.map.len() > self.cap {
                let Some(evicted) = inner.order.pop_front() else {
                    break;
                };
                inner.map.remove(&evicted);
            }
        }
        Some(ds)
    }
}
