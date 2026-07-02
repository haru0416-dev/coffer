//! Deterministic cross-unit aggregation index.
//!
//! Relevance selection (BM25 or embeddings) — keeping the units that score highest — can only KEEP
//! a unit that already contains the answer. But some queries have no answer in any single unit:
//! "the record with the largest value", "how many errors", "the most recent". The answer is
//! *emergent over the whole set*, so no selection can retrieve it — coffer's own negative
//! control (`gen_json_no_handle`) documents this loss. This module computes such facts
//! deterministically in Rust over ALL units (including the ones coffer offloads) and emits a
//! one-line index the model can read, with NO model call and NO loss of reversibility (it is
//! additive display text alongside the byte-exact compressed doc).
//!
//! Scope: a focused demonstrator for superlative (argmax/argmin) queries over a numeric
//! field of a JSON array of objects — the class coffer structurally cannot answer by selection.

use serde_json::Value;

use crate::budget::{Op, json_cmp, scan_top_level_array_elements};

/// True iff every record that carries `field` carries it as a JSON number that is **`f64`-representable**
/// (and at least one does) — so a numeric reduction (sum/mean/argmax/argmin) is well-defined and will
/// not silently skip a value, which would yield a confidently wrong digest over a surviving subset.
///
/// The gate is `as_f64().is_some()`, not merely `is_number()`: every reduction below extracts via
/// `as_f64`, and the workspace enables `serde_json`'s `arbitrary_precision`, so a value like `1e400`
/// is a number (`is_number() == true`) whose `as_f64()` is `None`. Requiring `f64`-representability
/// refuses such out-of-range numbers (and string-encoded / mixed-type values) instead of dropping
/// them and reporting a wrong aggregate.
fn field_is_clean_numeric(arr: &[Value], field: &str) -> bool {
    let mut seen = false;
    for obj in arr.iter().filter_map(Value::as_object) {
        if let Some(v) = obj.get(field) {
            if v.as_f64().is_none() {
                return false;
            }
            seen = true;
        }
    }
    seen
}

/// Clean numeric values of `field` over all records, or `None` if the field is mixed-type/empty.
fn numeric_vals(arr: &[Value], field: &str) -> Option<Vec<f64>> {
    if !field_is_clean_numeric(arr, field) {
        return None;
    }
    let vals: Vec<f64> = arr
        .iter()
        .filter_map(Value::as_object)
        .filter_map(|o| o.get(field).and_then(Value::as_f64))
        .collect();
    if vals.is_empty() { None } else { Some(vals) }
}

/// Parse a percentile intent: `median` → 50; `pNN` or `NN percentile` → NN (1..=99). Else `None`.
fn percentile_of(q: &str) -> Option<f64> {
    if q.contains("median") {
        return Some(50.0);
    }
    for tok in q.split(|c: char| !c.is_ascii_alphanumeric()) {
        if let Some(rest) = tok.strip_prefix('p') {
            if !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()) {
                if let Ok(v) = rest.parse::<f64>() {
                    if (1.0..=99.0).contains(&v) {
                        return Some(v);
                    }
                }
            }
        }
    }
    if q.contains("percentile") {
        for tok in q.split(|c: char| !c.is_ascii_digit()) {
            if let Ok(v) = tok.parse::<f64>() {
                if (1.0..=99.0).contains(&v) {
                    return Some(v);
                }
            }
        }
    }
    None
}

/// Parse a numeric threshold comparison: returns a comparator code (`>` `<` `g`=≥ `l`=≤) and the
/// number, e.g. "greater than 500" → (`>`, 500). `None` if no clear comparison.
fn threshold_of(q: &str) -> Option<(char, f64)> {
    let cmp = if q.contains("at least") || q.contains("greater than or equal") {
        'g'
    } else if q.contains("at most") || q.contains("less than or equal") {
        'l'
    } else if q.contains("greater than") || q.contains("more than") || q.contains("above") {
        '>'
    } else if q.contains("less than")
        || q.contains("below")
        || q.contains("under")
        || q.contains("fewer than")
    {
        '<'
    } else {
        return None;
    };
    let digits: String = q
        .chars()
        .map(|c| {
            if c.is_ascii_digit() || c == '.' {
                c
            } else {
                ' '
            }
        })
        .collect();
    digits
        .split_whitespace()
        .find_map(|t| t.parse::<f64>().ok())
        .map(|v| (cmp, v))
}

/// GROUP BY a category field, aggregate (sum or mean) a numeric field per group, and return the
/// group with the extreme aggregate — e.g. "which status has the highest total value". `None` unless
/// the query clearly asks for a per-group extreme and names both a category and a numeric field.
fn group_extreme(arr: &[Value], q: &str) -> Option<String> {
    let want_hi = ["highest", "largest", "most", "greatest", "maximum", "top"]
        .iter()
        .any(|w| q.contains(w));
    let want_lo = ["lowest", "smallest", "least", "minimum", "fewest"]
        .iter()
        .any(|w| q.contains(w));
    if want_hi == want_lo {
        return None;
    }
    let grouped = q.contains("total")
        || q.contains("sum")
        || q.contains("average")
        || q.contains("mean")
        || q.contains("per ")
        || q.contains("each")
        || q.contains(" by ");
    if !grouped {
        return None;
    }
    let first = arr.iter().find_map(Value::as_object)?;
    // category fields = the named string fields that are REAL categories (distinct < n, so not an id
    // field like "name" whose distinct count equals n). MULTIPLE named categories compose into a
    // single group-by (e.g. "by region,status"); fewest-distinct-then-name order keeps the composite
    // key stable and deterministic regardless of object field order.
    let mut cats: Vec<(String, usize)> = first
        .iter()
        .filter(|(_, v)| v.is_string())
        .map(|(k, _)| k.clone())
        .filter(|k| q.contains(&k.to_ascii_lowercase()))
        .map(|f| {
            let d = arr
                .iter()
                .filter_map(Value::as_object)
                .filter_map(|o| o.get(&f).and_then(Value::as_str))
                .collect::<std::collections::HashSet<_>>()
                .len();
            (f, d)
        })
        .filter(|(_, d)| *d < arr.len())
        .collect();
    if cats.is_empty() {
        return None;
    }
    cats.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    let cat_fields: Vec<String> = cats.into_iter().map(|(f, _)| f).collect();
    let cat = cat_fields.join(","); // display label; the grouping uses the field list itself
    let num = first
        .iter()
        .filter(|(_, v)| v.is_number())
        .map(|(k, _)| k.clone())
        .find(|k| q.contains(&k.to_ascii_lowercase()))?;
    if !field_is_clean_numeric(arr, &num) {
        return None;
    }
    let mean = q.contains("average") || q.contains("mean");
    let mut groups: std::collections::HashMap<String, (f64, usize)> =
        std::collections::HashMap::new();
    for o in arr.iter().filter_map(Value::as_object) {
        // composite key over EVERY category field; a record missing any category value is skipped
        // (as the single-field path already did) so a group key is always fully specified.
        let key: Option<Vec<&str>> = cat_fields
            .iter()
            .map(|f| o.get(f).and_then(Value::as_str))
            .collect();
        if let (Some(parts), Some(v)) = (key, o.get(&num).and_then(Value::as_f64)) {
            let e = groups.entry(parts.join(" × ")).or_insert((0.0, 0));
            e.0 += v;
            e.1 += 1;
        }
    }
    if groups.is_empty() {
        return None;
    }
    let agg = |s: &(f64, usize)| if mean { s.0 / s.1 as f64 } else { s.0 };
    // Extreme aggregate value, then EVERY group achieving it — report a tie rather than letting
    // HashMap iteration order silently (and non-deterministically) pick one winner.
    let extreme = groups
        .values()
        .map(agg)
        .reduce(|a, b| if want_hi { a.max(b) } else { a.min(b) })?;
    let mut winners: Vec<&String> = groups
        .iter()
        .filter(|(_, s)| (agg(s) - extreme).abs() <= f64::EPSILON * agg(s).abs().max(1.0))
        .map(|(k, _)| k)
        .collect();
    winners.sort(); // deterministic order for both the single-winner and tie renderings
    let aggname = if mean { "mean" } else { "sum" };
    let dir = if want_hi { "max" } else { "min" };
    if winners.len() == 1 {
        return Some(format!(
            "[index] {dir} {aggname}({num}) by {cat} = \"{}\" ({extreme})",
            winners[0]
        ));
    }
    let shown: Vec<String> = winners.iter().take(6).map(|w| format!("\"{w}\"")).collect();
    let more = if winners.len() > 6 {
        format!(", +{} more", winners.len() - 6)
    } else {
        String::new()
    };
    Some(format!(
        "[index] {dir} {aggname}({num}) by {cat} = {}-way TIE: {}{} (each {extreme})",
        winners.len(),
        shown.join(", "),
        more
    ))
}

/// If `query` asks for a superlative over a numeric field of a JSON array of objects, return a
/// one-line index naming the winning record. `None` if the intent/shape doesn't match (so the
/// arm adds nothing and cannot fabricate a wrong index).
#[must_use]
pub fn aggregate_index(input: &[u8], query: &str) -> Option<String> {
    let v: Value = serde_json::from_slice(input).ok()?;
    superlative_on(v.as_array()?, &query.to_ascii_lowercase()).map(|(line, _)| line)
}

/// Like [`aggregate_index`], but also returns the **0-based array indices of the record(s) that back
/// the answer** — the provenance of the computed superlative. A caller can fetch exactly those records
/// (e.g. `coffer_rows` / `coffer_json`) and re-verify the number against the original bytes, so the
/// computed answer is byte-auditable, not just asserted.
#[must_use]
pub fn superlative_rows(input: &[u8], query: &str) -> Option<(String, Vec<usize>)> {
    let v: Value = serde_json::from_slice(input).ok()?;
    superlative_on(v.as_array()?, &query.to_ascii_lowercase())
}

/// Superlative over an already-parsed array — shared by [`aggregate_index`] and [`digest`] so the
/// input JSON is parsed exactly once per request. Returns the one-line index AND the 0-based indices
/// of the record(s) at the extreme.
fn superlative_on(arr: &[Value], q: &str) -> Option<(String, Vec<usize>)> {
    // strip threshold phrases so "at least"/"at most" don't read as min/max superlatives
    // (those are numeric comparisons handled by the digest's threshold-count path).
    let qs = q.replace("at least", "  ").replace("at most", "  ");
    let want_max = ["largest", "highest", "max", "most", "greatest", "biggest"]
        .iter()
        .any(|w| qs.contains(w));
    let want_min = ["smallest", "lowest", "min", "least", "fewest"]
        .iter()
        .any(|w| qs.contains(w));
    if want_max == want_min || arr.is_empty() {
        return None; // need exactly one direction, over a non-empty array
    }
    let first = arr.iter().find_map(Value::as_object)?;

    // numeric field, preferring one named in the query, else the first numeric field (no Vec).
    let field = first
        .iter()
        .filter(|(_, v)| v.is_number())
        .map(|(k, _)| k)
        .find(|k| q.contains(&k.to_ascii_lowercase()))
        .or_else(|| first.iter().find(|(_, v)| v.is_number()).map(|(k, _)| k))?
        .clone();
    // No fabrication on mixed-type fields: if any record carries `field` as a non-number
    // (e.g. a string-encoded "9999999999"), `as_f64` would silently skip it and we'd report a
    // WRONG max/min. Refuse instead — the arm adds nothing rather than a confident falsehood.
    if !field_is_clean_numeric(arr, &field) {
        return None;
    }
    // identifier field = the first string field (e.g. "name").
    let id_field = first
        .iter()
        .find(|(_, v)| v.is_string())
        .map(|(k, _)| k.clone());

    // Extreme value.
    let mut extreme: Option<f64> = None;
    for n in arr
        .iter()
        .filter_map(Value::as_object)
        .filter_map(|o| o.get(&field).and_then(Value::as_f64))
    {
        extreme = Some(match extreme {
            None => n,
            Some(b) if want_max => b.max(n),
            Some(b) => b.min(n),
        });
    }
    let val = extreme?;
    // Provenance: 0-based indices of every record at the extreme value — the records that BACK the
    // answer, so a caller can fetch exactly those and verify the number byte-for-byte.
    let winner_idxs: Vec<usize> = arr
        .iter()
        .enumerate()
        .filter(|(_, e)| {
            e.as_object()
                .and_then(|o| o.get(&field))
                .and_then(Value::as_f64)
                == Some(val)
        })
        .map(|(i, _)| i)
        .collect();
    let dir = if want_max { "max" } else { "min" };
    // With no string id field, the extreme VALUE is the answer (ties are indistinguishable anyway).
    let Some(idf) = &id_field else {
        return Some((
            format!("[index] {dir}({field}) over {} records = {val}", arr.len()),
            winner_idxs,
        ));
    };
    // Every record achieving the extreme — report a tie rather than arbitrarily picking one winner.
    let winners: Vec<String> = arr
        .iter()
        .filter_map(Value::as_object)
        .filter(|o| o.get(&field).and_then(Value::as_f64) == Some(val))
        .map(|o| {
            o.get(idf)
                .and_then(Value::as_str)
                .map_or_else(|| "?".to_string(), str::to_string)
        })
        .collect();
    if winners.len() == 1 {
        return Some((
            format!(
                "[index] {dir}({field}) over {} records: {idf}=\"{}\" ({field}={val})",
                arr.len(),
                winners[0]
            ),
            winner_idxs,
        ));
    }
    let shown: Vec<String> = winners.iter().take(6).map(|w| format!("\"{w}\"")).collect();
    let more = if winners.len() > 6 {
        format!(", +{} more", winners.len() - 6)
    } else {
        String::new()
    };
    Some((
        format!(
            "[index] {dir}({field})={val} over {} records — {}-way TIE on {idf}: {}{} (no unique {dir})",
            arr.len(),
            winners.len(),
            shown.join(", "),
            more
        ),
        winner_idxs,
    ))
}

/// The string/numeric field whose name appears in the (lowercased) query, or `None`.
fn named_field(first: &serde_json::Map<String, Value>, q: &str, is_num: bool) -> Option<String> {
    first
        .iter()
        .filter(|(_, v)| if is_num { v.is_number() } else { v.is_string() })
        .map(|(k, _)| k.clone())
        .find(|k| q.contains(&k.to_ascii_lowercase()))
}

/// The `how many` / `number of` family — threshold count over a numeric field, distinct or
/// group-by count over a string field, or the plain record count. Split out of [`digest`].
fn digest_count(arr: &[Value], q: &str, n: usize) -> Option<String> {
    if !(q.contains("how many") || q.contains("number of")) {
        return None;
    }
    let first = arr.iter().find_map(Value::as_object)?;
    // THRESHOLD count over a numeric field, e.g. "how many records have value greater than 500".
    if let Some(field) = named_field(first, q, true) {
        if let (Some((cmp, thr)), Some(vals)) = (threshold_of(q), numeric_vals(arr, &field)) {
            let c = vals
                .iter()
                .filter(|&&x| match cmp {
                    '>' => x > thr,
                    '<' => x < thr,
                    'g' => x >= thr,
                    _ => x <= thr,
                })
                .count();
            let op = match cmp {
                '>' => ">",
                '<' => "<",
                'g' => ">=",
                _ => "<=",
            };
            return Some(format!(
                "[index] count({field} {op} {thr}) over {n} records = {c}"
            ));
        }
    }
    if let Some(field) = named_field(first, q, false) {
        let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for o in arr.iter().filter_map(Value::as_object) {
            if let Some(s) = o.get(&field).and_then(Value::as_str) {
                *counts.entry(s.to_string()).or_insert(0) += 1;
            }
        }
        // DISTINCT count of a string field.
        if q.contains("distinct") || q.contains("unique") || q.contains("different") {
            return Some(format!(
                "[index] distinct({field}) over {n} records = {} values",
                counts.len()
            ));
        }
        // GROUP-BY count: a field value named in the query.
        if let Some((val, c)) = counts
            .iter()
            .find(|(k, _)| q.contains(&k.to_ascii_lowercase()))
        {
            return Some(format!(
                "[index] count({field}=\"{val}\") over {n} records = {c}"
            ));
        }
    }
    // plain record COUNT.
    if [
        "record",
        "item",
        "entr",
        "element",
        "object",
        "array",
        "in total",
        "are there",
    ]
    .iter()
    .any(|w| q.contains(w))
    {
        return Some(format!("[index] count over {n} records = {n}"));
    }
    None
}

/// Shape-generic, exact, lossless DESCRIBE of a JSON array of objects: row count, field count,
/// and per field — present/distinct counts, and either numeric stats (min/max/mean/sum) for a clean
/// numeric field or a count-by-value breakdown for a low-cardinality categorical (most frequent first).
/// This is the RTK-style decision-relevant summary produced GENERICALLY — no per-tool/per-format code —
/// computed exactly over every record while the full bytes stay recoverable. `None` if the input is not a
/// JSON array containing at least one object.
#[must_use]
pub fn describe(input: &[u8]) -> Option<String> {
    let v: Value = serde_json::from_slice(input).ok()?;
    let arr = v.as_array()?;
    describe_rows(arr)
}

/// The canonical JSON text of a string value, byte-identical to serializing it as a
/// `Value`: `serde_json` escapes only `"`, `\` and control bytes (< 0x20) — non-ASCII
/// passes through raw — so a string with none of those serializes as exactly `"<s>"`.
fn canonical_string_text(s: &str) -> String {
    if s.bytes().all(|b| b >= 0x20 && b != b'"' && b != b'\\') {
        let mut out = String::with_capacity(s.len() + 2);
        out.push('"');
        out.push_str(s);
        out.push('"');
        out
    } else {
        Value::String(s.to_owned()).to_string()
    }
}

/// `Value::to_string()` with allocation fast paths for the two shapes that dominate real record
/// sets (plain strings and numbers). Output is byte-identical to `to_string()`: see
/// [`canonical_string_text`]; a `Number`'s `Display` prints its canonical JSON text.
fn canonical_text(val: &Value) -> String {
    match val {
        Value::String(s) => canonical_string_text(s),
        Value::Number(n) => n.to_string(),
        _ => val.to_string(),
    }
}

/// Per-field accumulator for [`describe_rows`]'s single pass. The count-by-value map is
/// split per JSON type so the hot types count on BORROWED keys — `&str` for strings and
/// `&Number` for numbers (`Number`'s `Eq`/`Hash` compare its repr, which is exactly its
/// canonical text) — instead of allocating a canonical-text `String` per row-value.
/// Canonical texts of distinct types can never collide (strings are quoted, numbers start
/// with a digit or `-`, bools with `t`/`f`, containers with `[`/`{`), so the per-type
/// counts merge losslessly into the single count-by-text map they replace.
#[derive(Default)]
struct FieldAcc<'a> {
    strings: std::collections::HashMap<&'a str, usize>,
    numbers: std::collections::HashMap<&'a serde_json::Number, usize>,
    trues: usize,
    falses: usize,
    /// Containers — rare as field values; keyed by canonical text like the original.
    other: std::collections::HashMap<String, usize>,
    /// `f64` values in row order — used only when the field stays clean-numeric, in which
    /// case it equals `numeric_vals`' collection exactly (same values, same order).
    vals: Vec<f64>,
    /// Starts `true`; cleared when ANY present value (null included) is not an
    /// `f64`-representable number — `field_is_clean_numeric`'s contract.
    clean: bool,
}

/// [`describe`] over already-parsed records — the parse-once path used by [`crate::Dataset`].
///
/// One pass over every record's own key/value pairs discovers fields (first-seen order) and
/// accumulates all per-field statistics at once; the previous shape re-walked the whole
/// array once per field per statistic.
pub(crate) fn describe_rows(arr: &[Value]) -> Option<String> {
    const MAX_CATEGORICAL: usize = 12;
    let n = arr.len();
    if n == 0 {
        return Some("[describe] 0 records".to_string());
    }
    if !arr.iter().any(Value::is_object) {
        return None; // not a record set
    }

    let mut fields: Vec<&str> = Vec::new();
    let mut accs: std::collections::HashMap<&str, FieldAcc> = std::collections::HashMap::new();
    for rec in arr {
        let Some(o) = rec.as_object() else { continue };
        for (k, v) in o {
            // Single hash lookup per (row, field): vacant discovers the field in
            // first-seen order, occupied hands back the accumulator.
            let acc = match accs.entry(k.as_str()) {
                std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                std::collections::hash_map::Entry::Vacant(e) => {
                    fields.push(k.as_str());
                    e.insert(FieldAcc {
                        clean: true,
                        ..FieldAcc::default()
                    })
                }
            };
            match v {
                Value::Null => acc.clean = false, // present but excluded from count-by
                Value::Number(num) => {
                    match num.as_f64() {
                        Some(f) => acc.vals.push(f),
                        None => acc.clean = false, // arbitrary_precision non-f64: refuse stats
                    }
                    *acc.numbers.entry(num).or_default() += 1;
                }
                Value::String(s) => {
                    acc.clean = false;
                    *acc.strings.entry(s.as_str()).or_default() += 1;
                }
                Value::Bool(b) => {
                    acc.clean = false;
                    if *b {
                        acc.trues += 1;
                    } else {
                        acc.falses += 1;
                    }
                }
                other => {
                    acc.clean = false;
                    *acc.other.entry(canonical_text(other)).or_default() += 1;
                }
            }
        }
    }

    let mut lines = vec![format!("[describe] {n} records, {} fields", fields.len())];
    for f in fields {
        lines.push(field_line(f, &accs[f], MAX_CATEGORICAL));
    }
    Some(lines.join("\n"))
}

/// One `describe` output line for a field, from its accumulated stats.
fn field_line(f: &str, acc: &FieldAcc, max_categorical: usize) -> String {
    let present = acc.strings.values().sum::<usize>()
        + acc.numbers.values().sum::<usize>()
        + acc.trues
        + acc.falses
        + acc.other.values().sum::<usize>();
    let distinct = acc.strings.len()
        + acc.numbers.len()
        + usize::from(acc.trues > 0)
        + usize::from(acc.falses > 0)
        + acc.other.len();
    if acc.clean && !acc.vals.is_empty() {
        let vals = &acc.vals;
        let sum: f64 = vals.iter().sum();
        let mean = sum / vals.len() as f64;
        let min = vals.iter().copied().fold(f64::INFINITY, f64::min);
        let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        format!(
            "  {f}: number present={present} distinct={distinct} min={min} max={max} mean={mean} sum={sum}"
        )
    } else if distinct <= max_categorical && distinct < present {
        // low-cardinality categorical: count-by-value, most frequent first (ties broken
        // by key). Canonical texts materialize HERE — at most `max_categorical` of them —
        // not once per row.
        let mut pairs: Vec<(String, usize)> = Vec::with_capacity(distinct);
        pairs.extend(
            acc.strings
                .iter()
                .map(|(s, c)| (canonical_string_text(s), *c)),
        );
        pairs.extend(acc.numbers.iter().map(|(num, c)| (num.to_string(), *c)));
        if acc.trues > 0 {
            pairs.push(("true".to_string(), acc.trues));
        }
        if acc.falses > 0 {
            pairs.push(("false".to_string(), acc.falses));
        }
        pairs.extend(acc.other.iter().map(|(t, c)| (t.clone(), *c)));
        pairs.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let breakdown = pairs
            .iter()
            .map(|(k, c)| format!("{k}:{c}"))
            .collect::<Vec<_>>()
            .join(", ");
        format!("  {f}: present={present} distinct={distinct} — {breakdown}")
    } else {
        format!("  {f}: present={present} distinct={distinct}")
    }
}

/// Generalized deterministic digest over a JSON array of objects — the class of facts emergent
/// over the WHOLE set that no relevance selection (BM25 or embeddings) can recover from a budget-truncated
/// view: `count`, `sum`/`mean`, `median`/`percentile`, `range`, `distinct`, `group-by`, and
/// `threshold` count (e.g. value > N).
/// Computed over ALL records (incl. offloaded); additive display text, raw stays byte-exact in CAS.
/// Returns `None` unless query intent AND array shape match — like [`aggregate_index`], it never
/// fabricates. Superlatives delegate to `superlative_on`. Record- and group-level argmax/argmin
/// report **ties explicitly** (every key at the extreme, deterministically sorted) rather than
/// silently picking one — there is no unique answer to invent.
#[must_use]
pub fn digest(input: &[u8], query: &str) -> Option<String> {
    let v: Value = serde_json::from_slice(input).ok()?;
    let arr = v.as_array()?;
    digest_rows(arr, query)
}

/// [`digest`] over already-parsed records — the parse-once path used by [`crate::Dataset`].
pub(crate) fn digest_rows(arr: &[Value], query: &str) -> Option<String> {
    let q = query.to_ascii_lowercase();
    if arr.is_empty() {
        return None;
    }
    let first = arr.iter().find_map(Value::as_object)?;
    let n = arr.len();

    // GROUP-ARGMAX ("which <category> has the highest total/avg <numeric>") — a GROUP BY + aggregate
    // + argmax over the whole set, BEFORE the record-level superlative (so it isn't mistaken for one).
    if let Some(s) = group_extreme(arr, &q) {
        return Some(s);
    }
    // superlative over a single RECORD (e.g. "the record with the largest value") — shares the
    // already-parsed array, so digest never re-parses the input.
    if let Some((s, _)) = superlative_on(arr, &q) {
        return Some(s);
    }

    // SUM / MEAN over a numeric field, optionally FILTERED by a category value named in the query
    // ("sum of value where status is error") — a SQL WHERE+aggregate no selection can recover.
    if q.contains("sum") || q.contains("total") || q.contains("average") || q.contains("mean") {
        let field = named_field(first, &q, true)?;
        if !field_is_clean_numeric(arr, &field) {
            return None; // mixed-type field → refuse rather than mis-sum
        }
        let filter: Option<(String, String)> = named_field(first, &q, false).and_then(|sf| {
            let values: std::collections::HashSet<String> = arr
                .iter()
                .filter_map(Value::as_object)
                .filter_map(|o| o.get(&sf).and_then(Value::as_str).map(str::to_string))
                .collect();
            values
                .into_iter()
                .find(|val| q.contains(&val.to_ascii_lowercase()))
                .map(|val| (sf, val))
        });
        let vals: Vec<f64> = arr
            .iter()
            .filter_map(Value::as_object)
            .filter(|o| match &filter {
                Some((sf, fv)) => o.get(sf).and_then(Value::as_str) == Some(fv.as_str()),
                None => true,
            })
            .filter_map(|o| o.get(&field).and_then(Value::as_f64))
            .collect();
        if vals.is_empty() {
            return None;
        }
        let m = vals.len();
        let sum: f64 = vals.iter().sum();
        let whr = filter
            .as_ref()
            .map(|(sf, fv)| format!(" where {sf}={fv}"))
            .unwrap_or_default();
        if q.contains("average") || q.contains("mean") {
            return Some(format!(
                "[index] mean({field}{whr}) over {m} records = {}",
                sum / m as f64
            ));
        }
        return Some(format!(
            "[index] sum({field}{whr}) over {m} records = {sum}"
        ));
    }

    // PERCENTILE / MEDIAN over a numeric field (nearest-rank) — LLMs can't compute these over many
    // records even with full context, and no selection can recover them from a subset.
    if let Some(p) = percentile_of(&q) {
        let field = named_field(first, &q, true)?;
        let mut vals = numeric_vals(arr, &field)?;
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = (((p / 100.0) * vals.len() as f64).ceil() as usize).clamp(1, vals.len()) - 1;
        let label = if (p - 50.0).abs() < f64::EPSILON {
            "median".to_string()
        } else {
            format!("p{}", p as u64)
        };
        return Some(format!(
            "[index] {label}({field}) over {n} records = {}",
            vals[idx]
        ));
    }

    // RANGE (max − min) of a numeric field.
    if q.contains("range") {
        let field = named_field(first, &q, true)?;
        let vals = numeric_vals(arr, &field)?;
        let mn = vals.iter().copied().fold(f64::INFINITY, f64::min);
        let mx = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        return Some(format!(
            "[index] range({field}) over {n} records = {} (min {mn}, max {mx})",
            mx - mn
        ));
    }

    // count / distinct / group-by / threshold-count.
    digest_count(arr, &q, n)
}

/// Like [`digest`], but over the **concatenation of several JSON-array payloads** — an exact aggregate
/// across multiple offloaded handles (e.g. a kubectl dump and a log, or paged search results held under
/// separate handles), computed over the UNION of all their records. Inputs that are not a JSON array (or
/// do not parse) are skipped; `None` if nothing usable remains or the query/shape doesn't match. The
/// records are combined and then run through the exact same `digest` path, so the contract (never
/// fabricates, refuses mixed-type fields, reports ties) is identical.
#[must_use]
pub fn digest_across(inputs: &[&[u8]], query: &str) -> Option<String> {
    let mut combined: Vec<Value> = Vec::new();
    for bytes in inputs {
        if let Ok(Value::Array(elems)) = serde_json::from_slice::<Value>(bytes) {
            combined.extend(elems);
        }
    }
    if combined.is_empty() {
        return None;
    }
    // Re-serialize the union and reuse `digest` verbatim — cross-handle aggregation is a deliberate,
    // non-hot operation, so one extra serialize+parse is fine and keeps a single source of truth.
    let bytes = serde_json::to_vec(&Value::Array(combined)).ok()?;
    digest(&bytes, query)
}

/// Like [`digest`], but over **NDJSON** — newline-delimited JSON, one value per line (common tool
/// output from `jq`, structured-log dumps, export streams). Each non-blank line is parsed and the
/// records are aggregated together. Returns `None` unless the input is *predominantly* JSON lines, so
/// prose or ordinary logs are never mistaken for NDJSON and aggregated over a coincidental subset
///.
#[must_use]
pub fn digest_ndjson(input: &[u8], query: &str) -> Option<String> {
    let mut records: Vec<Value> = Vec::new();
    let mut nonblank = 0usize;
    for line in input.split(|&b| b == b'\n') {
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        nonblank += 1;
        if let Ok(v) = serde_json::from_slice::<Value>(line) {
            records.push(v);
        }
    }
    // Require ≥ 80% of non-blank lines to parse as JSON; otherwise this is not NDJSON — refuse.
    if records.is_empty() || records.len() * 5 < nonblank * 4 {
        return None;
    }
    let bytes = serde_json::to_vec(&Value::Array(records)).ok()?;
    digest(&bytes, query)
}

/// A conjunctive filter predicate over a record field: `field <op> value`.
#[derive(Clone, Debug)]
pub struct Predicate {
    /// The object field to test.
    pub field: String,
    /// The comparison operator.
    pub op: Op,
    /// The value to compare against (a JSON number or string).
    pub value: Value,
}

/// The aggregate to compute over the records that pass ALL predicates.
#[derive(Clone, Debug)]
pub enum Agg {
    /// Count of matching records.
    Count,
    /// Sum of a numeric field over matching records.
    Sum(String),
    /// Mean of a numeric field over matching records.
    Mean(String),
    /// Minimum of a numeric field over matching records.
    Min(String),
    /// Maximum of a numeric field over matching records.
    Max(String),
}

/// The exact result of a typed query-aggregate, with the backing record indices (provenance).
#[derive(Clone, Debug, PartialEq)]
pub struct QueryResult {
    /// One-line, model-facing summary.
    pub display: String,
    /// The numeric result (count, sum, mean, min, or max).
    pub value: f64,
    /// 0-based indices of the records that passed ALL predicates — the answer's provenance.
    pub matched: Vec<usize>,
}

/// One group of a grouped aggregate: its key, the aggregate over its members, and their provenance.
#[derive(Clone, Debug, PartialEq)]
pub struct GroupBucket {
    /// The group key (a right-side attribute value, or a numeric bucket's lower bound).
    pub key: String,
    /// The aggregate over the records in this group.
    pub value: f64,
    /// 0-based indices of the records in this group — the group's provenance.
    pub matched: Vec<usize>,
}

/// The exact result of a grouped aggregate (group-by join or numeric bucketing): one entry per group,
/// ordered deterministically by key, each carrying its own provenance.
#[derive(Clone, Debug, PartialEq)]
pub struct GroupAggregate {
    /// One-line, model-facing summary.
    pub display: String,
    /// The per-group results, ordered by key.
    pub groups: Vec<GroupBucket>,
}

fn op_str(op: Op) -> &'static str {
    match op {
        Op::Eq => "==",
        Op::Ne => "!=",
        Op::Gt => ">",
        Op::Ge => ">=",
        Op::Lt => "<",
        Op::Le => "<=",
    }
}

fn agg_label(agg: &Agg) -> String {
    match agg {
        Agg::Count => "count".to_string(),
        Agg::Sum(f) => format!("sum({f})"),
        Agg::Mean(f) => format!("mean({f})"),
        Agg::Min(f) => format!("min({f})"),
        Agg::Max(f) => format!("max({f})"),
    }
}

/// Compute `agg` over the records of `arr` at the given `matched` indices. Returns `None` (refuse)
/// when an aggregated field is present-but-non-`f64`, or when no matched record carries the field.
/// Shared by [`query_aggregate`] and [`join_aggregate`] so both honor the same refuse-rather-than-guess
/// contract.
fn apply_agg(arr: &[Value], matched: &[usize], agg: &Agg) -> Option<f64> {
    match agg {
        Agg::Count => Some(matched.len() as f64),
        Agg::Sum(field) | Agg::Mean(field) | Agg::Min(field) | Agg::Max(field) => {
            // Gather the field over the matched records; a present-but-non-f64 value refuses the
            // whole query (never aggregate over a silently-skipped subset). Missing field is skipped.
            let mut vals: Vec<f64> = Vec::new();
            for &i in matched {
                if let Some(fv) = arr[i].as_object().and_then(|o| o.get(field)) {
                    vals.push(fv.as_f64()?);
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
    }
}

/// Exact aggregate over the records of a JSON array passing a CONJUNCTION of typed predicates — the
/// structured, unambiguous form of [`digest`] (no English intent parsing), supporting MULTIPLE
/// predicates (`value > 10 AND status == "error"`) and returning the backing record indices so the
/// answer is byte-auditable. Computed over ALL records (incl. offloaded). Refuses (returns `None`)
/// rather than guess when the aggregated field is present-but-non-`f64` over the matched set, or when
/// the input is not a JSON array — never a confident wrong number.
#[must_use]
pub fn query_aggregate(input: &[u8], predicates: &[Predicate], agg: &Agg) -> Option<QueryResult> {
    let v: Value = serde_json::from_slice(input).ok()?;
    let arr = v.as_array()?;

    // matched = indices where EVERY predicate holds (a record missing a predicate's field fails it).
    let matched: Vec<usize> = arr
        .iter()
        .enumerate()
        .filter(|(_, rec)| {
            predicates.iter().all(|p| {
                rec.as_object()
                    .and_then(|o| o.get(&p.field))
                    .is_some_and(|fv| json_cmp(fv, p.op, &p.value))
            })
        })
        .map(|(i, _)| i)
        .collect();

    let value = apply_agg(arr, &matched, agg)?;
    Some(QueryResult {
        display: query_display(predicates, agg, matched.len(), value),
        value,
        matched,
    })
}

/// The one-line, model-facing summary of a query-aggregate — shared with [`crate::Dataset`]'s
/// column-accelerated path so both produce byte-identical display strings.
pub(crate) fn query_display(
    predicates: &[Predicate],
    agg: &Agg,
    matched_len: usize,
    value: f64,
) -> String {
    let whr = if predicates.is_empty() {
        String::new()
    } else {
        let clauses: Vec<String> = predicates
            .iter()
            .map(|p| format!("{} {} {}", p.field, op_str(p.op), p.value))
            .collect();
        format!(" where {}", clauses.join(" AND "))
    };
    format!(
        "[index] {}{whr} over {matched_len} matched records = {value}",
        agg_label(agg),
    )
}

/// Cross-handle semi-join aggregate: aggregate over the records of a LEFT JSON array that
/// have at least one matching record in a RIGHT JSON array (equi-join on `left[left_key] ==
/// right[right_key]`) where the right record additionally passes a CONJUNCTION of predicates. This
/// correlates TWO held datasets — "sum order.amount for orders whose customer is gold-tier" — which
/// neither a single-array query nor an English digest can express, and the rows of neither dataset
/// ever enter the context window. Join keys compare by canonical JSON text, so equality is
/// type-sensitive (the number `5` does not join the string `"5"`). Bounded: the output is one scalar
/// plus the LEFT-row indices that joined (provenance), never the cartesian product of rows. Refuses
/// (`None`) if either input is not a JSON array, or the aggregated field is present-but-non-numeric.
#[must_use]
pub fn join_aggregate(
    left: &[u8],
    right: &[u8],
    left_key: &str,
    right_key: &str,
    right_where: &[Predicate],
    agg: &Agg,
) -> Option<QueryResult> {
    let lv: Value = serde_json::from_slice(left).ok()?;
    let larr = lv.as_array()?;
    let rv: Value = serde_json::from_slice(right).ok()?;
    let rarr = rv.as_array()?;
    join_aggregate_rows(larr, rarr, left_key, right_key, right_where, agg)
}

/// [`join_aggregate`] over already-parsed records — the parse-once path used by [`crate::Dataset`].
pub(crate) fn join_aggregate_rows(
    larr: &[Value],
    rarr: &[Value],
    left_key: &str,
    right_key: &str,
    right_where: &[Predicate],
    agg: &Agg,
) -> Option<QueryResult> {
    // Canonical join keys of the right records that pass ALL right predicates. `to_string` canonicalizes
    // a scalar to its JSON text, so two equal scalars collapse to one key and equality stays type-aware.
    let mut allowed: std::collections::HashSet<String> = std::collections::HashSet::new();
    for rec in rarr {
        let passes = right_where.iter().all(|p| {
            rec.as_object()
                .and_then(|o| o.get(&p.field))
                .is_some_and(|fv| json_cmp(fv, p.op, &p.value))
        });
        if passes {
            if let Some(kv) = rec.as_object().and_then(|o| o.get(right_key)) {
                allowed.insert(kv.to_string());
            }
        }
    }

    // Inner semi-join: left rows whose join key matches at least one qualifying right key. A right key
    // appearing many times never double-counts a left row — existence, not multiplicity, gates the row.
    let matched: Vec<usize> = larr
        .iter()
        .enumerate()
        .filter(|(_, rec)| {
            rec.as_object()
                .and_then(|o| o.get(left_key))
                .is_some_and(|kv| allowed.contains(&kv.to_string()))
        })
        .map(|(i, _)| i)
        .collect();

    let value = apply_agg(larr, &matched, agg)?;

    let whr = if right_where.is_empty() {
        String::new()
    } else {
        let clauses: Vec<String> = right_where
            .iter()
            .map(|p| format!("right.{} {} {}", p.field, op_str(p.op), p.value))
            .collect();
        format!(" where {}", clauses.join(" AND "))
    };
    let display = format!(
        "[index] {} over left ⋉ right on {left_key}={right_key}{whr} ({} matched left records) = {value}",
        agg_label(agg),
        matched.len()
    );
    Some(QueryResult {
        display,
        value,
        matched,
    })
}

/// Helper: assemble a [`GroupAggregate`] from a left array, a map of group-key -> member left indices,
/// and an aggregate. Groups are ordered by key; a present-but-non-numeric aggregated field in ANY group
/// refuses the whole result (`None`).
fn finish_groups(
    larr: &[Value],
    mut groups: Vec<(String, Vec<usize>)>,
    agg: &Agg,
    header: &str,
) -> Option<GroupAggregate> {
    groups.sort_by(|a, b| a.0.cmp(&b.0));
    let mut out = Vec::with_capacity(groups.len());
    for (key, matched) in groups {
        let value = apply_agg(larr, &matched, agg)?;
        out.push(GroupBucket {
            key,
            value,
            matched,
        });
    }
    let display = format!("[index] {header} -> {} groups", out.len());
    Some(GroupAggregate {
        display,
        groups: out,
    })
}

/// Project-join group-by: correlate a LEFT array with a RIGHT array on an equi-join key, then
/// group the matched LEFT rows by a RIGHT-side attribute and aggregate each group — "sum order.amount
/// grouped by customer.region". This is the projecting complement to [`join_aggregate`]'s semi-join: it
/// brings a right column onto the grouping, which neither a single-array group-by nor the semi-join can.
/// Keys and group values compare by canonical JSON text. Each group carries the LEFT-row provenance.
/// Refuses (`None`) if either input is not a JSON array, if a join key maps to MORE THAN ONE distinct
/// group value on the right (ambiguous — never guess), or if an aggregated field is non-numeric.
#[must_use]
pub fn join_group_aggregate(
    left: &[u8],
    right: &[u8],
    left_key: &str,
    right_key: &str,
    group_field: &str,
    agg: &Agg,
) -> Option<GroupAggregate> {
    let lv: Value = serde_json::from_slice(left).ok()?;
    let larr = lv.as_array()?;
    let rv: Value = serde_json::from_slice(right).ok()?;
    let rarr = rv.as_array()?;
    join_group_aggregate_rows(larr, rarr, left_key, right_key, group_field, agg)
}

/// [`join_group_aggregate`] over already-parsed records — the parse-once path used by
/// [`crate::Dataset`].
pub(crate) fn join_group_aggregate_rows(
    larr: &[Value],
    rarr: &[Value],
    left_key: &str,
    right_key: &str,
    group_field: &str,
    agg: &Agg,
) -> Option<GroupAggregate> {
    // right join-key -> group value (canonical JSON text). A key mapping to two DIFFERENT group values
    // is ambiguous: refuse rather than pick one.
    let mut key_to_group: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for rec in rarr {
        let o = rec.as_object()?;
        let (Some(kv), Some(gv)) = (o.get(right_key), o.get(group_field)) else {
            continue;
        };
        let k = kv.to_string();
        let g = gv.to_string();
        match key_to_group.get(&k) {
            Some(existing) if existing != &g => return None,
            Some(_) => {}
            None => {
                key_to_group.insert(k, g);
            }
        }
    }

    // group matched left rows by their key's right-side group value (inner join: unmatched left dropped).
    let mut groups: std::collections::HashMap<String, Vec<usize>> =
        std::collections::HashMap::new();
    for (i, rec) in larr.iter().enumerate() {
        if let Some(kv) = rec.as_object().and_then(|o| o.get(left_key)) {
            if let Some(g) = key_to_group.get(&kv.to_string()) {
                groups.entry(g.clone()).or_default().push(i);
            }
        }
    }

    let header = format!(
        "{} over left ⋈ right on {left_key}={right_key} grouped by right.{group_field}",
        agg_label(agg)
    );
    finish_groups(larr, groups.into_iter().collect(), agg, &header)
}

/// Numeric bucketing aggregate: group the records of a JSON array into fixed-width buckets of
/// a numeric `bucket_field` (`floor(v / width) * width`), then aggregate each bucket — a histogram /
/// windowed distribution over ALL held rows ("count per 100-unit latency bucket", "sum amount per bucket").
/// Each bucket carries its members' provenance and is keyed by its lower bound. Refuses (`None`) if the
/// input is not a JSON array, if `width <= 0`, or if `bucket_field` is present-but-non-numeric on any row
/// (mixed type — never bucket a guessed value); rows missing `bucket_field` are excluded.
#[must_use]
pub fn bucket_aggregate(
    input: &[u8],
    bucket_field: &str,
    width: f64,
    agg: &Agg,
) -> Option<GroupAggregate> {
    let v: Value = serde_json::from_slice(input).ok()?;
    let arr = v.as_array()?;
    bucket_aggregate_rows(arr, bucket_field, width, agg)
}

/// [`bucket_aggregate`] over already-parsed records — the parse-once path used by
/// [`crate::Dataset`].
pub(crate) fn bucket_aggregate_rows(
    arr: &[Value],
    bucket_field: &str,
    width: f64,
    agg: &Agg,
) -> Option<GroupAggregate> {
    if !width.is_finite() || width <= 0.0 {
        return None;
    }
    let mut groups: std::collections::HashMap<String, Vec<usize>> =
        std::collections::HashMap::new();
    for (i, rec) in arr.iter().enumerate() {
        let Some(fv) = rec.as_object().and_then(|o| o.get(bucket_field)) else {
            continue; // missing bucket field: excluded, like a predicate miss
        };
        let x = fv.as_f64()?; // present-but-non-numeric: refuse the whole call
        // `floor()*width` can produce -0.0 (a JSON `-0.0`, or a tiny-negative quotient that underflows);
        // adding 0.0 normalizes -0.0 to +0.0 (IEEE-754) so the zero bucket has a single canonical "0" key
        // instead of splitting into "0" and "-0".
        let lo = (x / width).floor() * width + 0.0;
        groups.entry(format!("{lo}")).or_default().push(i);
    }

    // order buckets by numeric lower bound, not lexical key
    let mut pairs: Vec<(String, Vec<usize>)> = groups.into_iter().collect();
    pairs.sort_by(|a, b| {
        let (x, y) = (
            a.0.parse::<f64>().unwrap_or(0.0),
            b.0.parse::<f64>().unwrap_or(0.0),
        );
        x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut out = Vec::with_capacity(pairs.len());
    for (key, matched) in pairs {
        let value = apply_agg(arr, &matched, agg)?;
        out.push(GroupBucket {
            key,
            value,
            matched,
        });
    }
    let display = format!(
        "[index] {} bucketed by {bucket_field} width {width} -> {} buckets",
        agg_label(agg),
        out.len()
    );
    Some(GroupAggregate {
        display,
        groups: out,
    })
}

/// Windowed log histogram: count case-insensitive substring matches of `pattern` per
/// `window`-line block of a held text/log payload — "how many ERROR lines per 1000-line block" — so an
/// agent sees WHERE an event clusters across a large log without reading it. Each block carries the
/// 1-based line numbers that matched (provenance, ready for `coffer_lines`/`coffer_search`). Only blocks
/// with at least one match appear, ordered by line range. Computed over ALL lines. Refuses (`None`) if
/// `window == 0` or `pattern` is empty.
#[must_use]
pub fn count_matches_per_window(
    text: &[u8],
    pattern: &str,
    window: usize,
) -> Option<GroupAggregate> {
    if window == 0 || pattern.is_empty() {
        return None;
    }
    let hay = String::from_utf8_lossy(text);
    let needle = pattern.to_lowercase();
    // bucket index (0-based block of `window` lines) -> matching 1-based line numbers. BTreeMap keeps
    // blocks in ascending line order without a separate sort.
    let mut groups: std::collections::BTreeMap<usize, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (idx, line) in hay.lines().enumerate() {
        if line.to_lowercase().contains(&needle) {
            groups.entry(idx / window).or_default().push(idx + 1);
        }
    }
    let groups: Vec<GroupBucket> = groups
        .into_iter()
        .map(|(bucket, matched)| {
            let lo = bucket * window + 1;
            let hi = lo + window - 1;
            GroupBucket {
                key: format!("lines {lo}-{hi}"),
                value: matched.len() as f64,
                matched,
            }
        })
        .collect();
    let display = format!(
        "[lines] '{pattern}' per {window}-line window -> {} non-empty windows",
        groups.len()
    );
    Some(GroupAggregate { display, groups })
}

/// Derived-handle algebra: return the records of a JSON array passing a CONJUNCTION of
/// typed predicates as a NEW JSON array, copying each kept row's bytes VERBATIM from the input (the
/// rows are byte-exact). The result is itself a valid payload, so [`digest`], [`query_aggregate`], and
/// `query_subset` compose over it — an agent can pipe `query_subset(query_subset(h, p1), p2)` and
/// aggregate the residual without ever materializing rows into the context window. Chaining equals the
/// conjunction byte-for-byte. `None` if the input is not a well-formed top-level JSON array, or if the
/// element scanner and the parser disagree on element count (refuse rather than mis-slice).
#[must_use]
pub fn query_subset(input: &[u8], predicates: &[Predicate]) -> Option<Vec<u8>> {
    let v: Value = serde_json::from_slice(input).ok()?;
    let arr = v.as_array()?;
    let ranges = scan_top_level_array_elements(input)?;
    if ranges.len() != arr.len() {
        return None;
    }
    let mut out = Vec::with_capacity(input.len());
    out.push(b'[');
    let mut first = true;
    for (rec, range) in arr.iter().zip(ranges.iter()) {
        let keep = predicates.iter().all(|p| {
            rec.as_object()
                .and_then(|o| o.get(&p.field))
                .is_some_and(|fv| json_cmp(fv, p.op, &p.value))
        });
        if keep {
            if !first {
                out.push(b',');
            }
            out.extend_from_slice(&input[range.clone()]);
            first = false;
        }
    }
    out.push(b']');
    Some(out)
}

/// Fetch the records at an explicit set of `indices` from a JSON array as a NEW array, copying each
/// row's bytes VERBATIM (byte-exact). This is the executable companion to the provenance returned by
/// [`query_aggregate`] / [`join_aggregate`]: feed the `matched` indices back to pull exactly the rows
/// behind an aggregate and re-verify the number byte-for-byte. Rows appear in the order given; an
/// out-of-range index refuses the whole call (`None`), as does a non-array input.
#[must_use]
pub fn pick_rows(input: &[u8], indices: &[usize]) -> Option<Vec<u8>> {
    let v: Value = serde_json::from_slice(input).ok()?;
    let arr = v.as_array()?;
    let ranges = scan_top_level_array_elements(input)?;
    if ranges.len() != arr.len() {
        return None;
    }
    let mut out = Vec::new();
    out.push(b'[');
    let mut first = true;
    for &idx in indices {
        let range = ranges.get(idx)?;
        if !first {
            out.push(b',');
        }
        out.extend_from_slice(&input[range.clone()]);
        first = false;
    }
    out.push(b']');
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn argmax_over_value_finds_the_record() {
        // The no_handle shape: fillers + one record with a huge value.
        let mut s = String::from("[");
        for i in 0..50 {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!(r#"{{"name":"item-{i}","value":{i}}}"#));
        }
        s.push_str(r#",{"name":"item-99","value":1000000000}]"#);
        let idx = aggregate_index(
            s.as_bytes(),
            "Return the name of the record with the largest value.",
        )
        .expect("index");
        assert!(idx.contains("item-99"), "got: {idx}");
        assert!(idx.contains("1000000000"));
    }

    #[test]
    fn smallest_works_and_non_superlative_returns_none() {
        let input = br#"[{"name":"a","value":5},{"name":"b","value":2},{"name":"c","value":9}]"#;
        let idx = aggregate_index(input, "which record has the smallest value?").unwrap();
        assert!(idx.contains('b'), "got: {idx}");
        assert!(aggregate_index(input, "what is the token of record a").is_none());
    }

    fn fixture() -> Vec<u8> {
        // 6 records: value sums to 5+2+9+4+10+1 = 31; status distinct = {ok, error} (2);
        // status=error appears 2×.
        br#"[
          {"name":"a","value":5,"status":"ok"},
          {"name":"b","value":2,"status":"error"},
          {"name":"c","value":9,"status":"ok"},
          {"name":"d","value":4,"status":"ok"},
          {"name":"e","value":10,"status":"error"},
          {"name":"f","value":1,"status":"ok"}
        ]"#
        .to_vec()
    }

    #[test]
    fn digest_count_sum_distinct_group_are_exact() {
        let x = fixture();
        assert!(
            digest(&x, "How many records are there in the array?")
                .unwrap()
                .contains("= 6")
        );
        assert!(
            digest(&x, "What is the sum of the value field?")
                .unwrap()
                .contains("= 31")
        );
        assert!(
            digest(&x, "What is the average of the value field?")
                .unwrap()
                .contains("5.16")
        );
        assert!(
            digest(&x, "How many distinct status values are there?")
                .unwrap()
                .contains("= 2 values")
        );
        let g = digest(&x, "How many records have status error?").unwrap();
        assert!(g.contains("error") && g.contains("= 2"), "got: {g}");
    }

    #[test]
    fn digest_refuses_unsupported_and_superlative_delegates() {
        let x = fixture();
        // unsupported intent → None (no fabrication)
        assert!(digest(&x, "what is the capital of France").is_none());
        // superlative still works via aggregate_index
        assert!(
            digest(&x, "which record has the largest value?")
                .unwrap()
                .contains("max(value)")
        );
    }

    #[test]
    fn digest_filter_aggregate_and_group_extreme() {
        // fixture: ok={5,9,4,1} sum19 mean4.75; error={2,10} sum12 mean6.
        let x = fixture();
        assert!(
            digest(
                &x,
                "what is the sum of the value field where status is error?"
            )
            .unwrap()
            .contains("= 12"),
            "{:?}",
            digest(
                &x,
                "what is the sum of the value field where status is error?"
            )
        );
        assert!(
            digest(&x, "which status has the highest total value?")
                .unwrap()
                .contains("\"ok\""),
            "{:?}",
            digest(&x, "which status has the highest total value?")
        );
        assert!(
            digest(&x, "which status has the highest average value?")
                .unwrap()
                .contains("\"error\"")
        );
        // a record-level superlative is NOT mistaken for a group query
        assert!(
            digest(&x, "which record has the largest value?")
                .unwrap()
                .contains("name=")
        );
        // group-by ignores the id field even if "name" is mentioned — picks the real category.
        assert!(
            digest(
                &x,
                "across each name, which status has the highest total value?"
            )
            .unwrap()
            .contains("by status")
        );
        // unfiltered sum still works
        assert!(
            digest(&x, "what is the sum of the value field?")
                .unwrap()
                .contains("= 31")
        );
    }

    #[test]
    fn digest_percentile_range_threshold_are_exact() {
        // values 1..=10 (sorted): median=5 (nearest-rank ceil(0.5*10)=5 → idx4 → 5);
        // p90 → ceil(0.9*10)=9 → idx8 → 9; range = 9 (10-1); count(value > 7) = 3 (8,9,10).
        let mut s = String::from("[");
        for i in 1..=10 {
            if i > 1 {
                s.push(',');
            }
            s.push_str(&format!(r#"{{"name":"r{i}","value":{i}}}"#));
        }
        s.push(']');
        let x = s.as_bytes();
        assert!(
            digest(x, "what is the median value field?")
                .unwrap()
                .contains("median(value) over 10 records = 5"),
            "{:?}",
            digest(x, "what is the median value field?")
        );
        assert!(
            digest(x, "what is the p90 of the value field?")
                .unwrap()
                .contains("= 9")
        );
        assert!(
            digest(x, "what is the range of the value field?")
                .unwrap()
                .contains("range(value) over 10 records = 9")
        );
        assert!(
            digest(x, "how many records have value greater than 7?")
                .unwrap()
                .contains("= 3")
        );
        assert!(
            digest(x, "how many records have value at least 9?")
                .unwrap()
                .contains("= 2")
        );
    }

    #[test]
    fn reports_ties_instead_of_silently_breaking_them() {
        // Two records share the max value 10 — no UNIQUE argmax exists, so report the tie
        // rather than arbitrarily naming one (a confident answer to an ambiguous question).
        let x = br#"[{"name":"a","value":3},{"name":"b","value":10},{"name":"c","value":10},{"name":"d","value":1}]"#;
        let s = aggregate_index(x, "which record has the largest value?").unwrap();
        assert!(s.contains("TIE"), "got: {s}");
        assert!(s.contains("\"b\"") && s.contains("\"c\""), "tied ids: {s}");
        assert!(!s.contains("\"a\""), "non-winners must not appear: {s}");

        // GROUP-BY tie: status=ok sum 5+5=10, status=err sum 4+6=10 → both groups tie.
        let g = br#"[{"name":"a","value":5,"status":"ok"},{"name":"b","value":5,"status":"ok"},{"name":"c","value":4,"status":"err"},{"name":"d","value":6,"status":"err"}]"#;
        let gd = digest(g, "which status has the highest total value?").unwrap();
        assert!(gd.contains("TIE"), "got: {gd}");
        assert!(gd.contains("\"err\"") && gd.contains("\"ok\""), "got: {gd}");
        // Determinism: the tie renders identically across runs (sorted, not HashMap order).
        for _ in 0..8 {
            assert_eq!(
                digest(g, "which status has the highest total value?").unwrap(),
                gd
            );
        }

        // A clean unique extreme still renders the single-winner format (no regression).
        let uniq = br#"[{"name":"a","value":3},{"name":"b","value":10},{"name":"c","value":1}]"#;
        let u = aggregate_index(uniq, "which record has the largest value?").unwrap();
        assert!(u.contains("name=\"b\"") && !u.contains("TIE"), "got: {u}");
    }

    #[test]
    fn join_group_aggregate_groups_by_right_attribute_and_refuses_ambiguous() {
        let customers =
            br#"[{"id":1,"region":"us"},{"id":2,"region":"eu"},{"id":3,"region":"us"}]"#;
        let orders =
            br#"[{"cid":1,"amt":100},{"cid":2,"amt":50},{"cid":3,"amt":30},{"cid":1,"amt":20}]"#;
        let r = join_group_aggregate(
            orders,
            customers,
            "cid",
            "id",
            "region",
            &Agg::Sum("amt".into()),
        )
        .unwrap();
        // eu: cid 2 -> 50 ; us: cid 1,3,1 -> 100+30+20 = 150. Ordered by key ("eu" < "us").
        assert_eq!(r.groups.len(), 2);
        assert_eq!(r.groups[0].key, "\"eu\"");
        assert_eq!(r.groups[0].value as i64, 50);
        assert_eq!(r.groups[1].key, "\"us\"");
        assert_eq!(r.groups[1].value as i64, 150);
        assert_eq!(r.groups[1].matched, vec![0, 2, 3]);

        // an ambiguous key (id 1 maps to two different regions) refuses rather than guessing.
        let ambiguous = br#"[{"id":1,"region":"us"},{"id":1,"region":"eu"}]"#;
        assert!(
            join_group_aggregate(orders, ambiguous, "cid", "id", "region", &Agg::Count).is_none()
        );
    }

    #[test]
    fn bucket_aggregate_histograms_a_numeric_field() {
        let input = br#"[{"v":5},{"v":15},{"v":25},{"v":105},{"v":12}]"#;
        // width 10: bucket 0 -> {5} , 10 -> {15,12}, 20 -> {25}, 100 -> {105}
        let r = bucket_aggregate(input, "v", 10.0, &Agg::Count).unwrap();
        assert_eq!(r.groups.len(), 4);
        assert_eq!(r.groups[0].key, "0");
        assert_eq!(r.groups[0].value as i64, 1);
        assert_eq!(r.groups[1].key, "10");
        assert_eq!(r.groups[1].value as i64, 2);
        assert_eq!(r.groups[1].matched, vec![1, 4]);
        assert_eq!(r.groups[3].key, "100");
        // width <= 0 refuses; a non-numeric bucket field refuses.
        assert!(bucket_aggregate(input, "v", 0.0, &Agg::Count).is_none());
        assert!(bucket_aggregate(br#"[{"v":"x"}]"#, "v", 10.0, &Agg::Count).is_none());
    }

    #[test]
    fn bucket_aggregate_signed_zero_is_one_bucket() {
        // Regression: a JSON `-0.0` (and any tiny-negative value whose quotient underflows to -0.0)
        // must NOT split the zero bucket into "0" and "-0". Both belong to a single "0" bucket.
        let r =
            bucket_aggregate(br#"[{"v":0},{"v":-0.0},{"v":5}]"#, "v", 10.0, &Agg::Count).unwrap();
        assert_eq!(r.groups.len(), 1);
        assert_eq!(r.groups[0].key, "0");
        assert_eq!(r.groups[0].value as i64, 3);
        // underflow path: -1e-300 / 1e300 floors to a -0.0 lower bound; still the single "0" bucket.
        let u =
            bucket_aggregate(br#"[{"v":1e-300},{"v":-1e-300}]"#, "v", 1e300, &Agg::Count).unwrap();
        assert_eq!(u.groups.len(), 1);
        assert_eq!(u.groups[0].key, "0");
    }

    #[test]
    fn join_group_aggregate_single_group_equals_semi_join() {
        // Cross-invariant: when every right row shares ONE group value, the lone group's aggregate
        // equals the semi-join aggregate (join_aggregate) over the same left/right/keys.
        let right = br#"[{"id":1,"region":"us"},{"id":2,"region":"us"},{"id":3,"region":"us"}]"#;
        let left = br#"[{"cid":1,"amt":100},{"cid":2,"amt":50},{"cid":9,"amt":999}]"#;
        let grouped =
            join_group_aggregate(left, right, "cid", "id", "region", &Agg::Sum("amt".into()))
                .unwrap();
        assert_eq!(grouped.groups.len(), 1);
        let semi = join_aggregate(left, right, "cid", "id", &[], &Agg::Sum("amt".into())).unwrap();
        // cid 9 has no right match → excluded by both; us group = 100 + 50 = 150.
        assert_eq!(grouped.groups[0].value as i64, semi.value as i64);
        assert_eq!(grouped.groups[0].value as i64, 150);
    }

    #[test]
    fn join_group_aggregate_one_to_many_same_group_does_not_refuse() {
        // A join key appearing on MULTIPLE right rows with the SAME group value is not ambiguous.
        let right = br#"[{"id":1,"region":"us"},{"id":1,"region":"us"}]"#;
        let left = br#"[{"cid":1,"amt":100},{"cid":1,"amt":40}]"#;
        let r = join_group_aggregate(left, right, "cid", "id", "region", &Agg::Sum("amt".into()))
            .unwrap();
        assert_eq!(r.groups.len(), 1);
        assert_eq!(r.groups[0].value as i64, 140); // each LEFT row counted once
        assert_eq!(r.groups[0].matched, vec![0, 1]);
    }

    proptest! {
        /// Project-join group-by equals a brute-force reference: map each right key to its group value
        /// (unique by construction here), group the left rows by that value, sum the agg field.
        #[test]
        fn join_group_aggregate_matches_brute_force(
            regions in proptest::collection::vec(0usize..3, 1..12),
            lefts in proptest::collection::vec((0usize..1000, 0i64..1000), 1..25),
        ) {
            const REGION: [&str; 3] = ["us", "eu", "ap"];
            let n = regions.len();
            // right[i] = {id:i, region:REGION[regions[i]]} — ids unique by construction.
            let rarr: Vec<Value> = regions
                .iter()
                .enumerate()
                .map(|(i, &r)| serde_json::json!({ "id": i, "region": REGION[r] }))
                .collect();
            let larr: Vec<Value> = lefts
                .iter()
                .map(|&(cid, amt)| serde_json::json!({ "cid": cid % n, "amt": amt }))
                .collect();
            let right = serde_json::to_vec(&Value::Array(rarr)).unwrap();
            let left = serde_json::to_vec(&Value::Array(larr)).unwrap();

            // brute force: group left.amt by REGION of its (always valid) cid.
            let mut want: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
            for &(cid, amt) in &lefts {
                let region = REGION[regions[cid % n]];
                *want.entry(format!("\"{region}\"")).or_default() += amt;
            }

            let got = join_group_aggregate(
                &left, &right, "cid", "id", "region", &Agg::Sum("amt".into()),
            )
            .unwrap();
            let got_map: std::collections::HashMap<String, i64> =
                got.groups.iter().map(|g| (g.key.clone(), g.value as i64)).collect();
            prop_assert_eq!(got_map, want);
            // groups arrive ordered by key.
            let mut keys: Vec<String> = got.groups.iter().map(|g| g.key.clone()).collect();
            let sorted = {
                let mut k = keys.clone();
                k.sort();
                k
            };
            prop_assert_eq!(&keys, &sorted);
            keys.dedup();
            prop_assert_eq!(keys.len(), got.groups.len()); // no duplicate group keys
        }

        /// Bucketing equals a brute-force reference: floor(v/width)*width groups, count per bucket, and
        /// every row lands in exactly one bucket (provenance partitions the input).
        #[test]
        fn bucket_aggregate_matches_brute_force(
            vs in proptest::collection::vec(-500i64..500, 1..30),
            width in 1i64..50,
        ) {
            let arr: Vec<Value> = vs.iter().map(|&v| serde_json::json!({ "v": v })).collect();
            let input = serde_json::to_vec(&Value::Array(arr)).unwrap();
            let r = bucket_aggregate(&input, "v", width as f64, &Agg::Count).unwrap();

            let mut want: std::collections::HashMap<i64, i64> = std::collections::HashMap::new();
            for &v in &vs {
                let lo = (v as f64 / width as f64).floor() as i64 * width;
                *want.entry(lo).or_default() += 1;
            }
            let got: std::collections::HashMap<i64, i64> = r
                .groups
                .iter()
                .map(|g| (g.key.parse::<f64>().unwrap() as i64, g.value as i64))
                .collect();
            prop_assert_eq!(&got, &want);
            // every row is in exactly one bucket: provenance counts sum to the row total.
            let total: usize = r.groups.iter().map(|g| g.matched.len()).sum();
            prop_assert_eq!(total, vs.len());
        }
    }

    #[test]
    fn describe_summarizes_records_generically() {
        let input = br#"[{"a":10,"status":"ok"},{"a":20,"status":"error"},{"a":30,"status":"error"},{"a":40,"status":"error"}]"#;
        let d = describe(input).unwrap();
        assert!(d.contains("4 records"), "{d}");
        // numeric field a: exact stats, no per-tool code.
        assert!(
            d.contains("sum=100") && d.contains("min=10") && d.contains("max=40"),
            "{d}"
        );
        // categorical status: count-by-value, most frequent first (error:3 before ok:1).
        let s = d.lines().find(|l| l.contains("status:")).unwrap();
        assert!(s.contains(r#""error":3"#) && s.contains(r#""ok":1"#), "{s}");
        assert!(
            s.find(r#""error":3"#).unwrap() < s.find(r#""ok":1"#).unwrap(),
            "most-frequent first: {s}"
        );
        // non-array / non-record inputs are refused.
        assert!(describe(b"{\"a\":1}").is_none());
        assert!(describe(b"[1,2,3]").is_none());
    }

    proptest! {
        /// The shape-generic describe's numeric sum and categorical count-by are EXACT vs brute force —
        /// the RTK-style summary generated with no per-tool code, computed over every record.
        #[test]
        fn describe_counts_and_sums_are_exact(
            recs in proptest::collection::vec((0i64..100, 0usize..3), 5..40),
        ) {
            const ST: [&str; 3] = ["ok", "warn", "error"];
            let arr: Vec<Value> =
                recs.iter().map(|&(a, s)| serde_json::json!({ "a": a, "status": ST[s] })).collect();
            let input = serde_json::to_vec(&Value::Array(arr)).unwrap();
            let d = describe(&input).unwrap();

            let sum: i64 = recs.iter().map(|&(a, _)| a).sum();
            prop_assert!(d.contains(&format!("sum={sum}")), "want sum={sum} in:\n{d}");
            // present=n records (status always present), distinct<=3<present (n>=5) → count-by renders.
            for (i, name) in ST.iter().enumerate() {
                let c = recs.iter().filter(|&&(_, s)| s == i).count();
                if c > 0 {
                    prop_assert!(d.contains(&format!("\"{name}\":{c}")), "want {name}:{c} in:\n{d}");
                }
            }
        }
    }

    #[test]
    fn count_matches_per_window_clusters_log_events() {
        let log = b"INFO start\nERROR a\nINFO ok\nERROR b\nINFO ok\nINFO ok\nerror c\nINFO done";
        // window 3: lines 1-3 -> ERROR a (1 match, line 2); 4-6 -> ERROR b (line 4); 7-9 -> error c (line 7).
        let r = count_matches_per_window(log, "error", 3).unwrap();
        assert_eq!(r.groups.len(), 3);
        assert_eq!(r.groups[0].key, "lines 1-3");
        assert_eq!(r.groups[0].value as i64, 1);
        assert_eq!(r.groups[0].matched, vec![2]); // 1-based line number, case-insensitive
        assert_eq!(r.groups[1].matched, vec![4]);
        assert_eq!(r.groups[2].matched, vec![7]); // "error c" matched case-insensitively
        // empty pattern / zero window refuse.
        assert!(count_matches_per_window(log, "", 3).is_none());
        assert!(count_matches_per_window(log, "error", 0).is_none());
    }

    proptest! {
        /// Windowed log histogram equals a brute-force reference: count case-insensitive matches per
        /// window-line block; only non-empty blocks appear, ordered by line, and provenance line numbers
        /// are correct (1-based) and sum to the total match count.
        #[test]
        fn count_matches_per_window_matches_brute_force(
            flags in proptest::collection::vec(any::<bool>(), 1..60),
            window in 1usize..8,
        ) {
            // build a log where line i contains "ERR" iff flags[i].
            let text: String = flags
                .iter()
                .enumerate()
                .map(|(i, &f)| if f { format!("line {i} ERR\n") } else { format!("line {i} ok\n") })
                .collect();
            let r = count_matches_per_window(text.as_bytes(), "err", window).unwrap();

            // brute force: bucket -> count.
            let mut want: std::collections::BTreeMap<usize, i64> = std::collections::BTreeMap::new();
            for (i, &f) in flags.iter().enumerate() {
                if f {
                    *want.entry(i / window).or_default() += 1;
                }
            }
            let got: std::collections::BTreeMap<usize, i64> = r
                .groups
                .iter()
                .map(|g| {
                    // recover bucket index from "lines {lo}-{hi}" : (lo-1)/window
                    let lo: usize =
                        g.key.trim_start_matches("lines ").split('-').next().unwrap().parse().unwrap();
                    ((lo - 1) / window, g.value as i64)
                })
                .collect();
            prop_assert_eq!(&got, &want);
            // provenance: total matched line numbers == total matches, and each is a real ERR line.
            let total: usize = r.groups.iter().map(|g| g.matched.len()).sum();
            prop_assert_eq!(total, flags.iter().filter(|&&f| f).count());
            for g in &r.groups {
                for &ln in &g.matched {
                    prop_assert!(flags[ln - 1]); // 1-based line ln really contained ERR
                }
            }
        }
    }

    #[test]
    fn join_aggregate_semi_joins_two_handles() {
        let customers =
            br#"[{"id":1,"tier":"gold"},{"id":2,"tier":"silver"},{"id":3,"tier":"gold"}]"#;
        let orders =
            br#"[{"cid":1,"amount":100},{"cid":2,"amount":50},{"cid":3,"amount":30},{"cid":1,"amount":20}]"#;
        let gold = Predicate {
            field: "tier".into(),
            op: Op::Eq,
            value: serde_json::json!("gold"),
        };

        // sum order.amount for orders whose customer is gold-tier (ids 1, 3): 100 + 30 + 20 = 150.
        let sum = join_aggregate(
            orders,
            customers,
            "cid",
            "id",
            std::slice::from_ref(&gold),
            &Agg::Sum("amount".into()),
        )
        .unwrap();
        assert_eq!(sum.value as i64, 150);
        assert_eq!(sum.matched, vec![0, 2, 3]);

        // count of qualifying orders = 3 (a right key seen twice does not double-count a left row).
        let count = join_aggregate(
            orders,
            customers,
            "cid",
            "id",
            std::slice::from_ref(&gold),
            &Agg::Count,
        )
        .unwrap();
        assert_eq!(count.value as i64, 3);

        // type-sensitive keys: the string "1" must not join the numeric id 1.
        let str_orders = br#"[{"cid":"1","amount":100}]"#;
        let r = join_aggregate(
            str_orders,
            customers,
            "cid",
            "id",
            std::slice::from_ref(&gold),
            &Agg::Count,
        )
        .unwrap();
        assert_eq!(r.value as i64, 0);
    }

    proptest! {
        /// Cross-handle semi-join aggregate equals a brute-force reference: build the set of right keys
        /// passing the predicate, then aggregate the left rows whose key is in that set. Duplicate right
        /// keys (one-to-many) must not inflate the count — semi-join gates on existence.
        #[test]
        fn join_aggregate_matches_brute_force_semi_join(
            rights in proptest::collection::vec((0i64..5, 0usize..2), 1..15),
            lefts in proptest::collection::vec((0i64..5, 0i64..1000), 1..20),
        ) {
            const TIER: [&str; 2] = ["gold", "silver"];
            let rarr: Vec<Value> =
                rights.iter().map(|&(id, t)| serde_json::json!({ "id": id, "tier": TIER[t] })).collect();
            let larr: Vec<Value> =
                lefts.iter().map(|&(cid, amt)| serde_json::json!({ "cid": cid, "amount": amt })).collect();
            let right = serde_json::to_vec(&Value::Array(rarr)).unwrap();
            let left = serde_json::to_vec(&Value::Array(larr)).unwrap();
            let gold = Predicate { field: "tier".into(), op: Op::Eq, value: serde_json::json!("gold") };

            // brute-force reference
            let gold_ids: std::collections::HashSet<i64> =
                rights.iter().filter(|&&(_, t)| t == 0).map(|&(id, _)| id).collect();
            let want_matched: Vec<usize> = lefts
                .iter()
                .enumerate()
                .filter(|(_, pair)| gold_ids.contains(&pair.0))
                .map(|(i, _)| i)
                .collect();
            let want_sum: i64 = want_matched.iter().map(|&i| lefts[i].1).sum();

            let count =
                join_aggregate(&left, &right, "cid", "id", std::slice::from_ref(&gold), &Agg::Count)
                    .unwrap();
            prop_assert_eq!(count.value as usize, want_matched.len());
            prop_assert_eq!(&count.matched, &want_matched);

            if !want_matched.is_empty() {
                let sum = join_aggregate(
                    &left, &right, "cid", "id", std::slice::from_ref(&gold), &Agg::Sum("amount".into()),
                )
                .unwrap();
                prop_assert_eq!(sum.value as i64, want_sum);
                prop_assert_eq!(&sum.matched, &want_matched);
            }
        }
    }

    #[test]
    fn pick_rows_preserves_order_and_refuses_out_of_range() {
        let input = br#"[{"a":1},{"a":2},{"a":3}]"#;
        assert_eq!(
            pick_rows(input, &[1, 0]).unwrap(),
            br#"[{"a":2},{"a":1}]"#.to_vec()
        );
        assert_eq!(pick_rows(input, &[]).unwrap(), b"[]".to_vec());
        assert!(pick_rows(input, &[0, 3]).is_none()); // index 3 is out of range → refuse
    }

    proptest! {
        /// `pick_rows` over the provenance of a `query_aggregate` reconstructs the predicate-filtered
        /// subset byte-for-byte, and re-aggregating the picked rows reproduces the original count — so the
        /// `matched` indices are an executable, byte-auditable receipt for the aggregate.
        #[test]
        fn pick_rows_reconstructs_aggregate_provenance(
            recs in proptest::collection::vec((-1000i64..1000, 0usize..3), 1..30),
            thr in -1000i64..1000,
        ) {
            const STATUS: [&str; 3] = ["ok", "warn", "error"];
            let arr: Vec<Value> = recs
                .iter()
                .map(|&(a, st)| serde_json::json!({ "a": a, "status": STATUS[st] }))
                .collect();
            let input = serde_json::to_vec(&Value::Array(arr)).unwrap();
            let p = Predicate { field: "a".into(), op: Op::Gt, value: serde_json::json!(thr) };

            let agg = query_aggregate(&input, std::slice::from_ref(&p), &Agg::Count).unwrap();
            // pulling the provenance rows == filtering by the predicate (the indices ARE the subset).
            let picked = pick_rows(&input, &agg.matched).unwrap();
            let subset = query_subset(&input, std::slice::from_ref(&p)).unwrap();
            prop_assert_eq!(&picked, &subset);
            // re-counting the pulled rows reproduces the aggregate.
            let recount = query_aggregate(&picked, &[], &Agg::Count).unwrap();
            prop_assert_eq!(recount.value as usize, agg.value as usize);
        }
    }

    #[test]
    fn query_subset_is_byte_exact_and_chains() {
        let input =
            br#"[{"a":5,"status":"ok"},{"a":20,"status":"error"},{"a":30,"status":"error"}]"#;
        let p_a = Predicate {
            field: "a".into(),
            op: Op::Gt,
            value: serde_json::json!(10),
        };
        let p_s = Predicate {
            field: "status".into(),
            op: Op::Eq,
            value: serde_json::json!("error"),
        };
        // a > 10 → rows 1,2; the kept rows are byte-identical to the original element bytes.
        let sub = query_subset(input, std::slice::from_ref(&p_a)).unwrap();
        assert_eq!(
            sub,
            br#"[{"a":20,"status":"error"},{"a":30,"status":"error"}]"#.to_vec()
        );
        // chaining (status==error on the subset) EQUALS the conjunction, byte-for-byte.
        let chained = query_subset(&sub, std::slice::from_ref(&p_s)).unwrap();
        let conj = query_subset(input, &[p_a, p_s]).unwrap();
        assert_eq!(chained, conj);
        // aggregates compose over the derived subset.
        assert_eq!(
            query_aggregate(&sub, &[], &Agg::Sum("a".into()))
                .unwrap()
                .value as i64,
            50
        );
    }

    proptest! {
        /// Derived-subset algebra: query_subset rows are byte-exact slices of the input, the subset
        /// parses to exactly the matched records, aggregates COMPOSE (aggregate over the subset ==
        /// aggregate with the predicate), and CHAINING two subsets equals the conjunction byte-for-byte.
        #[test]
        fn query_subset_composes_and_rows_are_byte_exact(
            recs in proptest::collection::vec((-1000i64..1000, 0usize..3), 1..30),
            thr in -1000i64..1000,
        ) {
            const STATUS: [&str; 3] = ["ok", "warn", "error"];
            let arr: Vec<Value> = recs
                .iter()
                .map(|&(a, st)| serde_json::json!({ "a": a, "status": STATUS[st] }))
                .collect();
            let input = serde_json::to_vec(&Value::Array(arr.clone())).unwrap();

            let p_a = Predicate { field: "a".into(), op: Op::Gt, value: serde_json::json!(thr) };
            let p_s = Predicate { field: "status".into(), op: Op::Eq, value: serde_json::json!("error") };

            let sub_a = query_subset(&input, std::slice::from_ref(&p_a)).unwrap();

            // the subset parses to exactly the matched records, in order.
            let parsed: Vec<Value> = serde_json::from_slice(&sub_a).unwrap();
            let want_a: Vec<Value> =
                arr.iter().filter(|r| r["a"].as_i64().unwrap() > thr).cloned().collect();
            prop_assert_eq!(&parsed, &want_a);

            // byte-exact: each subset element's bytes equal the original matched element's bytes.
            let in_ranges = scan_top_level_array_elements(&input).unwrap();
            let sub_ranges = scan_top_level_array_elements(&sub_a).unwrap();
            let matched_in: Vec<&[u8]> = (0..arr.len())
                .filter(|&i| arr[i]["a"].as_i64().unwrap() > thr)
                .map(|i| &input[in_ranges[i].clone()])
                .collect();
            prop_assert_eq!(sub_ranges.len(), matched_in.len());
            for (sr, orig) in sub_ranges.iter().zip(matched_in.iter()) {
                prop_assert_eq!(&sub_a[sr.clone()], *orig);
            }

            // composition identity: aggregate over the subset (no predicate) == aggregate with the predicate.
            let direct = query_aggregate(&input, std::slice::from_ref(&p_a), &Agg::Count).unwrap();
            let via = query_aggregate(&sub_a, &[], &Agg::Count).unwrap();
            prop_assert_eq!(via.value as i64, direct.value as i64);
            if direct.value as i64 > 0 {
                let ds = query_aggregate(&input, std::slice::from_ref(&p_a), &Agg::Sum("a".into())).unwrap();
                let vs = query_aggregate(&sub_a, &[], &Agg::Sum("a".into())).unwrap();
                prop_assert_eq!(vs.value as i64, ds.value as i64);
            }

            // chaining EQUALS the conjunction, byte-for-byte.
            let chained = query_subset(&sub_a, std::slice::from_ref(&p_s)).unwrap();
            let conj = query_subset(&input, &[p_a.clone(), p_s.clone()]).unwrap();
            prop_assert_eq!(&chained, &conj);
        }
    }

    #[test]
    fn query_aggregate_conjunction_count_sum_and_refusal() {
        let input = br#"[{"a":5,"status":"ok"},{"a":20,"status":"error"},{"a":30,"status":"error"},{"a":1,"status":"error"}]"#;
        // a > 10 AND status == "error" → rows 1,2 (a=20,30): the 2-predicate conjunction the NL digest cannot express.
        let preds = vec![
            Predicate {
                field: "a".into(),
                op: Op::Gt,
                value: serde_json::json!(10),
            },
            Predicate {
                field: "status".into(),
                op: Op::Eq,
                value: serde_json::json!("error"),
            },
        ];
        let s = query_aggregate(input, &preds, &Agg::Sum("a".into())).unwrap();
        assert_eq!(s.value as i64, 50);
        assert_eq!(s.matched, vec![1, 2]); // provenance: exact backing rows
        assert_eq!(
            query_aggregate(input, &preds, &Agg::Count).unwrap().value as i64,
            2
        );
        assert_eq!(
            query_aggregate(input, &preds, &Agg::Max("a".into()))
                .unwrap()
                .value as i64,
            30
        );
        // refuse-rather-than-guess: a present-but-non-numeric aggregated value over the matched set.
        let mixed = br#"[{"a":5,"status":"error"},{"a":"NaN","status":"error"}]"#;
        let only_err = vec![Predicate {
            field: "status".into(),
            op: Op::Eq,
            value: serde_json::json!("error"),
        }];
        assert!(query_aggregate(mixed, &only_err, &Agg::Sum("a".into())).is_none());
        // non-array input → None (never fabricates).
        assert!(query_aggregate(b"{}", &[], &Agg::Count).is_none());
    }

    proptest! {
        /// query_aggregate over a CONJUNCTION of typed predicates matches an INDEPENDENT brute-force
        /// reference (matched set + count/sum/mean/min/max) computed directly over the same records,
        /// with byte-auditable provenance (every matched index actually satisfies both predicates).
        #[test]
        fn query_aggregate_matches_brute_force_reference(
            recs in proptest::collection::vec((-1000i64..1000, 0usize..3), 1..30),
            thr in -1000i64..1000,
        ) {
            const STATUS: [&str; 3] = ["ok", "warn", "error"];
            let arr: Vec<Value> = recs
                .iter()
                .map(|&(a, st)| serde_json::json!({ "a": a, "status": STATUS[st] }))
                .collect();
            let input = serde_json::to_vec(&Value::Array(arr)).unwrap();

            // predicate conjunction: a > thr AND status == "error".
            let preds = vec![
                Predicate { field: "a".into(), op: Op::Gt, value: serde_json::json!(thr) },
                Predicate { field: "status".into(), op: Op::Eq, value: serde_json::json!("error") },
            ];
            let want_idx: Vec<usize> = (0..recs.len())
                .filter(|&i| {
                    let (a, st) = recs[i];
                    a > thr && STATUS[st] == "error"
                })
                .collect();
            let want_vals: Vec<i64> = want_idx.iter().map(|&i| recs[i].0).collect();

            let c = query_aggregate(&input, &preds, &Agg::Count).unwrap();
            prop_assert_eq!(&c.matched, &want_idx);
            prop_assert_eq!(c.value as usize, want_idx.len());

            if want_idx.is_empty() {
                // no matched numeric values → the aggregate refuses rather than inventing 0.
                prop_assert!(query_aggregate(&input, &preds, &Agg::Sum("a".into())).is_none());
                return Ok(());
            }
            let sum: i64 = want_vals.iter().sum();
            let s = query_aggregate(&input, &preds, &Agg::Sum("a".into())).unwrap();
            prop_assert_eq!(s.value as i64, sum);
            prop_assert_eq!(&s.matched, &want_idx);
            let mean = query_aggregate(&input, &preds, &Agg::Mean("a".into())).unwrap();
            prop_assert!((mean.value - sum as f64 / want_vals.len() as f64).abs() < 1e-9);
            prop_assert_eq!(
                query_aggregate(&input, &preds, &Agg::Min("a".into())).unwrap().value as i64,
                *want_vals.iter().min().unwrap()
            );
            prop_assert_eq!(
                query_aggregate(&input, &preds, &Agg::Max("a".into())).unwrap().value as i64,
                *want_vals.iter().max().unwrap()
            );
            // provenance: every matched record actually satisfies BOTH predicates.
            for &i in &s.matched {
                prop_assert!(recs[i].0 > thr && STATUS[recs[i].1] == "error");
            }
        }
    }

    #[test]
    fn digest_across_aggregates_the_union_of_payloads() {
        // two separate JSON-array handles; answers are emergent over their UNION.
        let a =
            br#"[{"name":"a","value":5,"status":"ok"},{"name":"b","value":2,"status":"error"}]"#;
        let b =
            br#"[{"name":"c","value":10,"status":"error"},{"name":"d","value":1,"status":"ok"}]"#;
        // union values 5,2,10,1 → sum 18; status=error {2,10} count 2; max value = c (10).
        assert!(
            digest_across(&[a, b], "what is the sum of the value field?")
                .unwrap()
                .contains("= 18")
        );
        assert!(
            digest_across(&[a, b], "how many records have status error?")
                .unwrap()
                .contains("= 2")
        );
        assert!(
            digest_across(&[a, b], "which record has the largest value?")
                .unwrap()
                .contains("name=\"c\"")
        );
        // a single handle matches digest() exactly.
        assert_eq!(
            digest_across(&[a], "what is the sum of the value field?"),
            digest(a, "what is the sum of the value field?")
        );
        // nothing usable → None (never fabricates).
        assert!(digest_across(&[b"not json", b"{}"], "how many records are there").is_none());
    }

    #[test]
    fn digest_ndjson_aggregates_line_delimited_json() {
        let nd = b"{\"name\":\"a\",\"value\":5,\"status\":\"ok\"}\n{\"name\":\"b\",\"value\":2,\"status\":\"error\"}\n{\"name\":\"c\",\"value\":10,\"status\":\"error\"}\n";
        // values 5,2,10 → sum 17; status=error {2,10} count 2; max value = c.
        assert!(
            digest_ndjson(nd, "what is the sum of the value field?")
                .unwrap()
                .contains("= 17")
        );
        assert!(
            digest_ndjson(nd, "how many records have status error?")
                .unwrap()
                .contains("= 2")
        );
        assert!(
            digest_ndjson(nd, "which record has the largest value?")
                .unwrap()
                .contains("name=\"c\"")
        );
        // blank lines tolerated.
        assert!(
            digest_ndjson(b"{\"v\":1}\n\n{\"v\":2}\n", "how many records are there")
                .unwrap()
                .contains("= 2")
        );
        // prose is not NDJSON → None (no fabrication over a coincidental subset).
        assert!(
            digest_ndjson(
                b"hello world\nthis is prose\nnot json\n",
                "how many records"
            )
            .is_none()
        );
    }

    #[test]
    fn superlative_rows_returns_backing_record_indices() {
        // provenance: the answer points to the exact record index, so it is byte-auditable.
        let x = br#"[{"name":"a","value":5},{"name":"b","value":10},{"name":"c","value":3}]"#;
        let (line, rows) = superlative_rows(x, "which record has the largest value?").unwrap();
        assert!(line.contains("name=\"b\""), "got: {line}");
        assert_eq!(rows, vec![1], "b is at index 1");
        // a tie returns ALL backing indices.
        let t = br#"[{"name":"a","value":10},{"name":"b","value":3},{"name":"c","value":10}]"#;
        let (line, rows) = superlative_rows(t, "the largest value?").unwrap();
        assert!(line.contains("TIE"), "got: {line}");
        assert_eq!(rows, vec![0, 2], "a and c tie at 10");
        // non-superlative intent → None (no fabricated provenance).
        assert!(superlative_rows(x, "what is the capital of France").is_none());
    }

    #[test]
    fn multi_column_group_by_composes_named_categories() {
        // status × region totals: us+ok=5, eu+ok=3, us+err=10, eu+err=2 → max is (us, err)=10.
        let x = br#"[
          {"name":"a","value":5,"status":"ok","region":"us"},
          {"name":"b","value":3,"status":"ok","region":"eu"},
          {"name":"c","value":10,"status":"err","region":"us"},
          {"name":"d","value":2,"status":"err","region":"eu"}
        ]"#;
        let d = digest(x, "which status and region has the highest total value?").unwrap();
        assert!(d.contains("by region,status"), "composite key fields: {d}");
        assert!(
            d.contains("us") && d.contains("err") && d.contains("10"),
            "winner group + total: {d}"
        );
        // determinism across runs (composite key is order-stable, not HashMap-order dependent).
        for _ in 0..8 {
            assert_eq!(
                digest(x, "which status and region has the highest total value?").unwrap(),
                d
            );
        }
        // naming ONE category still groups by one — no regression to the single-column path.
        let one = digest(x, "which status has the highest total value?").unwrap();
        assert!(
            one.contains("by status") && !one.contains("region"),
            "single column: {one}"
        );
    }

    #[test]
    fn digest_edge_cases_never_panic() {
        // empty / non-array / non-JSON / single-record / empty-object — all must be graceful None
        // or a correct value, never a panic.
        assert!(digest(b"[]", "how many records").is_none());
        assert!(digest(b"{}", "how many records").is_none());
        assert!(digest(b"not json", "how many records").is_none());
        assert!(digest(b"[1,2,3]", "what is the sum of the value field").is_none()); // not objects
        assert!(
            digest(br#"[{"name":"a","value":7}]"#, "how many records are there")
                .unwrap()
                .contains("= 1")
        );
        assert!(digest(br#"[{"x":1}]"#, "how many distinct status values").is_none()); // no such field
    }

    #[test]
    fn refuses_mixed_type_field_instead_of_fabricating() {
        // the true max is STRING-encoded — the falsification case. We must REFUSE, not skip it
        // and report a wrong numeric max (was: a confident falsehood with no recovery).
        let mixed =
            br#"[{"name":"a","value":5},{"name":"b","value":"9999999999"},{"name":"c","value":3}]"#;
        assert!(aggregate_index(mixed, "which record has the largest value?").is_none());
        assert!(digest(mixed, "what is the sum of the value field?").is_none());
        // a clean numeric field still works.
        let clean = br#"[{"name":"a","value":5},{"name":"b","value":10},{"name":"c","value":3}]"#;
        assert!(
            digest(clean, "what is the sum of the value field?")
                .unwrap()
                .contains("= 18")
        );
    }

    #[test]
    fn refuses_out_of_f64_range_number_instead_of_fabricating() {
        // `arbitrary_precision` keeps 1e400 as a JSON *number* (is_number()==true) whose `as_f64()`
        // is None. Silently dropping it would report a max/sum over the surviving subset under an
        // all-records label — exactly the fabrication this module promises not to do.
        let oob = br#"[{"name":"a","value":1},{"name":"b","value":1e400},{"name":"c","value":3}]"#;
        assert!(aggregate_index(oob, "which record has the largest value?").is_none());
        assert!(digest(oob, "what is the sum of the value field?").is_none());
        assert!(digest(oob, "what is the max of the value field?").is_none());
        // in-f64-range values are still accepted — the guard must not over-refuse.
        let ok = br#"[{"name":"a","value":5},{"name":"b","value":10},{"name":"c","value":3}]"#;
        assert!(
            digest(ok, "what is the sum of the value field?")
                .unwrap()
                .contains("= 18")
        );
    }

    // ---- cross-handle (`digest_across`) property + falsification tests -------------
    /// One `{name, value}` record. `value` is bounded to `±10_000_000` so the i64 sum of up to a
    /// few hundred records stays exactly f64-representable and matches integer Display formatting.
    fn record_strategy() -> impl Strategy<Value = (String, i64)> {
        ("[a-z]{1,6}", -10_000_000_i64..=10_000_000_i64)
    }

    /// A small JSON array payload of `{name, value}` objects, returned as (bytes, parsed records).
    fn part_strategy() -> impl Strategy<Value = (Vec<u8>, Vec<(String, i64)>)> {
        proptest::collection::vec(record_strategy(), 0..8).prop_map(|recs| {
            let arr = Value::Array(
                recs.iter()
                    .map(|(n, v)| {
                        let mut o = serde_json::Map::new();
                        o.insert("name".into(), Value::String(n.clone()));
                        o.insert("value".into(), Value::Number((*v).into()));
                        Value::Object(o)
                    })
                    .collect(),
            );
            (serde_json::to_vec(&arr).unwrap(), recs)
        })
    }

    proptest! {
        /// `digest_across` over arbitrary vectors of `{name,value}` JSON-array payloads must match a
        /// brute-force reference computed DIRECTLY over the UNION of all records (sum / count / argmax),
        /// equal `digest(single)` for a single handle, and the aggregated union round-trips through serde.
        #[test]
        fn digest_across_matches_brute_force_reference(
            parts in proptest::collection::vec(part_strategy(), 0..6)
        ) {
            let byte_parts: Vec<Vec<u8>> = parts.iter().map(|(b, _)| b.clone()).collect();
            let refs: Vec<&[u8]> = byte_parts.iter().map(Vec::as_slice).collect();

            let all: Vec<(String, i64)> =
                parts.iter().flat_map(|(_, recs)| recs.iter().cloned()).collect();
            let total = all.len();
            let sum: i64 = all.iter().map(|(_, v)| *v).sum();

            let count_out = digest_across(&refs, "how many records are there?");
            if total == 0 {
                prop_assert!(count_out.is_none(), "empty union must yield None, got {count_out:?}");
                prop_assert!(digest_across(&refs, "what is the sum of the value field?").is_none());
                return Ok(());
            }
            let count_out = count_out.expect("non-empty union has a record count");
            prop_assert!(count_out.contains(&format!("= {total}")), "count ref={total}, got: {count_out}");

            let sum_out = digest_across(&refs, "what is the sum of the value field?")
                .expect("non-empty numeric union has a sum");
            prop_assert!(sum_out.contains(&format!("= {sum}")), "sum ref={sum}, got: {sum_out}");

            let max_val = all.iter().map(|(_, v)| *v).max().unwrap();
            let winners: Vec<&String> =
                all.iter().filter(|(_, v)| *v == max_val).map(|(n, _)| n).collect();
            let argmax_out = digest_across(&refs, "which record has the largest value?")
                .expect("non-empty union has an argmax");
            prop_assert!(
                argmax_out.contains(&format!("value={max_val}")) || argmax_out.contains(&format!("= {max_val}")),
                "argmax value ref={max_val}, got: {argmax_out}"
            );
            let distinct_winner = {
                let s: std::collections::HashSet<&String> = winners.iter().copied().collect();
                s.len() == 1
            };
            if distinct_winner {
                prop_assert!(
                    argmax_out.contains(&format!("name=\"{}\"", winners[0])),
                    "unique argmax name ref={}, got: {argmax_out}", winners[0]
                );
            }

            for (b, recs) in &parts {
                if recs.is_empty() {
                    continue;
                }
                let q = "what is the sum of the value field?";
                prop_assert_eq!(digest_across(&[b.as_slice()], q), digest(b, q));
            }

            let union = Value::Array(
                all.iter()
                    .map(|(n, v)| {
                        let mut o = serde_json::Map::new();
                        o.insert("name".into(), Value::String(n.clone()));
                        o.insert("value".into(), Value::Number((*v).into()));
                        Value::Object(o)
                    })
                    .collect(),
            );
            let union_bytes = serde_json::to_vec(&union).unwrap();
            let reparsed: Value = serde_json::from_slice(&union_bytes).unwrap();
            prop_assert_eq!(reparsed, union, "union must reconstruct byte-exactly");
        }
    }

    /// FALSIFICATION control: the correct reference matches `digest_across`, but an off-by-one
    /// reference (sum+1 / count+1 / max+1) does NOT — proving the property above has teeth.
    #[test]
    fn off_by_one_reference_would_fail() {
        let a = br#"[{"name":"a","value":5},{"name":"b","value":2}]"#;
        let b = br#"[{"name":"c","value":10},{"name":"d","value":1}]"#;
        let parts: &[&[u8]] = &[a, b];
        let sum_out = digest_across(parts, "what is the sum of the value field?").unwrap();
        assert!(sum_out.contains("= 18"), "true sum is 18: {sum_out}");
        let count_out = digest_across(parts, "how many records are there?").unwrap();
        assert!(count_out.contains("= 4"), "true count is 4: {count_out}");
        let max_out = digest_across(parts, "which record has the largest value?").unwrap();
        assert!(max_out.contains("name=\"c\"") && max_out.contains("value=10"));
        assert!(
            !sum_out.contains("= 19"),
            "off-by-one sum must not match: {sum_out}"
        );
        assert!(
            !count_out.contains("= 5"),
            "off-by-one count must not match: {count_out}"
        );
        assert!(
            !max_out.contains("value=11"),
            "off-by-one max must not match: {max_out}"
        );
    }

    // ---- provenance (`superlative_rows`) property test ------------------------------
    /// Build a JSON array of `{"name","value"}` records, byte-for-byte what a tool would emit.
    fn build_records(rows: &[(String, i64)]) -> Vec<u8> {
        let arr: Vec<Value> = rows
            .iter()
            .map(|(name, value)| serde_json::json!({ "name": name, "value": value }))
            .collect();
        serde_json::to_vec(&Value::Array(arr)).expect("serialize records")
    }

    proptest! {
        /// PROVENANCE is EXACTLY the argmax/argmin set — cross-checked against an independent
        /// reference: every returned index is at the extreme, every non-returned index is not, and
        /// the backing record (re-parsed from the ORIGINAL bytes) holds the claimed extreme value.
        #[test]
        fn superlative_rows_provenance_is_exactly_the_argmax_set(
            rows in proptest::collection::vec(
                ("[a-z]{1,6}", prop_oneof![any::<i64>(), -5i64..5i64]),
                1..24usize,
            ),
        ) {
            let input = build_records(&rows);
            let values: Vec<i64> = rows.iter().map(|(_, v)| *v).collect();
            let ref_max = *values.iter().max().unwrap();
            let ref_min = *values.iter().min().unwrap();
            let ref_max_idxs: Vec<usize> =
                values.iter().enumerate().filter(|&(_, &v)| v == ref_max).map(|(i, _)| i).collect();
            let ref_min_idxs: Vec<usize> =
                values.iter().enumerate().filter(|&(_, &v)| v == ref_min).map(|(i, _)| i).collect();

            for (query, ref_idxs, ref_val, dir) in [
                ("which record has the largest value?", &ref_max_idxs, ref_max, "max"),
                ("which record has the smallest value?", &ref_min_idxs, ref_min, "min"),
            ] {
                let (line, mut rows_out) = superlative_rows(input.as_slice(), query)
                    .expect("a single-record superlative over a clean numeric field must resolve");
                prop_assert!(line.contains(dir), "dir {dir} not in line: {line}");

                rows_out.sort_unstable();
                let mut want = ref_idxs.clone();
                want.sort_unstable();
                prop_assert_eq!(&rows_out, &want,
                    "provenance != argmax/argmin set for {}: got {:?}, want {:?}", dir, rows_out, want);

                for &i in &rows_out {
                    prop_assert_eq!(values[i], ref_val,
                        "returned index {} has value {} != extreme {} ({})", i, values[i], ref_val, dir);
                }
                for (i, &v) in values.iter().enumerate() {
                    if !rows_out.contains(&i) {
                        prop_assert_ne!(v, ref_val,
                            "index {} has extreme value {} but was NOT returned ({})", i, ref_val, dir);
                    }
                }

                let reparsed: Value = serde_json::from_slice(&input).unwrap();
                let backing = reparsed.as_array().unwrap();
                prop_assert!(!rows_out.is_empty(), "argmax set is never empty for non-empty input");
                for &i in &rows_out {
                    let v = backing[i].get("value").and_then(Value::as_i64).unwrap();
                    prop_assert_eq!(v, ref_val,
                        "backing record at row {} has value {} != claimed extreme {} ({})", i, v, ref_val, dir);
                }
            }
        }
    }

    // ---- content-types / NDJSON property tests -------------------------------------
    mod ndjson_props {
        use super::*;

        /// One arbitrary record: an i64 value bounded so its sum stays exactly f64-representable,
        /// and a status drawn from a tiny categorical set.
        fn record() -> impl Strategy<Value = (i64, usize)> {
            (-1_000_000i64..=1_000_000i64, 0usize..3)
        }

        /// Mirror `digest`'s f64 `Display` of integral aggregates (prints "31", not "31.0").
        fn fmt_num(x: i64) -> String {
            format!("{}", x as f64)
        }

        /// Build the NDJSON byte string for `recs`, interspersing blank lines per `blanks`.
        fn build_ndjson(recs: &[(i64, usize)], blanks: &[bool]) -> Vec<u8> {
            const STATUS: [&str; 3] = ["ok", "warn", "error"];
            let mut s = String::new();
            for (i, &(value, st)) in recs.iter().enumerate() {
                if blanks.get(i).copied().unwrap_or(false) {
                    s.push('\n');
                }
                s.push_str(&format!(
                    r#"{{"name":"item-{i}","value":{value},"status":"{}"}}"#,
                    STATUS[st]
                ));
                s.push('\n');
            }
            s.into_bytes()
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(400))]

            /// sum / count / argmax over fuzzed NDJSON each match an independent brute-force reference.
            #[test]
            fn ndjson_aggregates_match_brute_force_reference(
                recs in proptest::collection::vec(record(), 1..24),
                blanks in proptest::collection::vec(any::<bool>(), 0..24),
            ) {
                const STATUS: [&str; 3] = ["ok", "warn", "error"];
                let nd = build_ndjson(&recs, &blanks);

                let n = recs.len();
                let ref_sum: i64 = recs.iter().map(|&(v, _)| v).sum();
                let ref_max: i64 = recs.iter().map(|&(v, _)| v).max().unwrap();
                let winners: Vec<usize> = recs
                    .iter()
                    .enumerate()
                    .filter(|&(_, &(v, _))| v == ref_max)
                    .map(|(i, _)| i)
                    .collect();
                let ref_err_count = recs.iter().filter(|&&(_, st)| STATUS[st] == "error").count();

                let c = digest_ndjson(&nd, "how many records are there?")
                    .expect("predominantly-JSON NDJSON must aggregate");
                prop_assert!(c.contains(&format!("= {n}")), "count: `= {n}` in {c:?} (recs={recs:?})");

                let sum_line = digest_ndjson(&nd, "what is the sum of the value field?")
                    .expect("sum over numeric field");
                prop_assert!(
                    sum_line.contains(&format!("= {}", fmt_num(ref_sum))),
                    "sum: `= {}` in {sum_line:?}", fmt_num(ref_sum)
                );

                let argmax = digest_ndjson(&nd, "which record has the largest value?")
                    .expect("record-level argmax");
                if winners.len() == 1 {
                    let w = format!("item-{}", winners[0]);
                    prop_assert!(argmax.contains(&format!("name=\"{w}\"")), "argmax {w:?} in {argmax:?}");
                } else {
                    prop_assert!(argmax.contains("TIE"), "expected TIE among {winners:?} in {argmax:?}");
                    for &i in &winners {
                        prop_assert!(argmax.contains(&format!("\"item-{i}\"")), "tied item-{i} in {argmax:?}");
                    }
                }

                if ref_err_count > 0 {
                    let g = digest_ndjson(&nd, "how many records have status error?")
                        .expect("group-by count");
                    prop_assert!(
                        g.contains("count(status=\"error\")") && g.contains(&format!("= {ref_err_count}")),
                        "status=error count `= {ref_err_count}` in {g:?}"
                    );
                }

                let wrong = fmt_num(ref_sum.wrapping_add(1));
                prop_assert!(
                    !sum_line.contains(&format!("= {wrong}")),
                    "falsification: wrong sum `= {wrong}` appeared in {sum_line:?}"
                );
            }

            /// BOUNDARY: the 80% JSON-line gate — aggregate iff `j*5 >= (j+p)*4`, else refuse.
            #[test]
            fn ndjson_eighty_percent_boundary(
                j in 1usize..20,
                p in 0usize..20,
                base in -1000i64..1000,
            ) {
                let mut s = String::new();
                for i in 0..j {
                    let offset = i64::try_from(i).expect("i < 20 fits i64");
                    s.push_str(&format!(r#"{{"name":"r{i}","value":{}}}"#, base + offset));
                    s.push('\n');
                }
                for k in 0..p {
                    s.push_str(&format!("this is prose line {k}, not json at all\n"));
                }
                let nd = s.into_bytes();
                let parses = j * 5 >= (j + p) * 4;
                let got = digest_ndjson(&nd, "how many records are there?");
                if parses {
                    let line = got.expect("at >=80% JSON the corpus must aggregate");
                    prop_assert!(line.contains(&format!("= {j}")), "boundary count `= {j}` (j={j}, p={p}) in {line:?}");
                } else {
                    prop_assert!(got.is_none(), "boundary: <80% JSON (j={j}, p={p}) must be None, got {got:?}");
                }
            }

            /// FALSIFICATION control: prose-only input is never mistaken for NDJSON.
            #[test]
            fn prose_only_is_never_ndjson(
                lines in proptest::collection::vec("[a-z][a-z ]{0,40}", 1..12),
            ) {
                let mut s = String::new();
                for l in &lines {
                    s.push_str(l);
                    s.push('\n');
                }
                let nd = s.into_bytes();
                prop_assert!(
                    digest_ndjson(&nd, "how many records are there?").is_none(),
                    "prose-only corpus must be None: {:?}", String::from_utf8_lossy(&nd)
                );
            }
        }
    }
}
