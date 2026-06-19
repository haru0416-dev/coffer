//! Stage 1: a budget knob over the Stage 0 reversible model.
//!
//! [`compress_to_budget`] offloads *whole structural units* (JSON array elements / log
//! lines) until the model-facing render fits a target token count, keeping a **head+tail
//! window** verbatim so the start AND the end — errors, summaries, most-recent
//! rows — survive. Reversibility stays **structural and unchanged**: every offloaded run
//! stores the exact original byte slice in the CAS, and the emitted segments tile the input
//! exactly, so `reconstruct(compress_to_budget(x)) == x` for every budget — the same
//! guarantee Stage 0 has, now driven to intermediate operating points so coffer can be
//! placed on the accuracy-vs-compression curve.
//!
//! What it does NOT do yet (deferred): relevance/anomaly unit selection (the window is the
//! query-free approximation); JSON *objects* still offload whole (no safe ordered truncation
//! of unordered keys); and the budget search re-renders per probe (fine at Stage-1a sizes).

use std::ops::Range;

use coffer_cas::{Cas, ContentHash};
use coffer_tokenizer::TokenCounter;
use serde_json::Value;

use crate::compress::offload_whole;
use crate::detect::{ContentType, detect};
use crate::doc::{CompressedDoc, Segment};

/// Target for budget-driven compression.
#[derive(Clone, Copy, Debug)]
pub enum Budget {
    /// Aim to keep the model-facing render at/under this many tokens. The codec lands
    /// ≤ target whenever `target` ≥ the structural floor (kept skeleton + one sentinel);
    /// below that floor it emits its minimum achievable render (which exceeds `target`),
    /// and the caller's ±band / sub-unit trim must absorb the residual.
    Tokens(usize),
    /// Fractional reduction vs the raw render: `0.0` = passthrough, `0.8` = 80% off.
    Reduction(f32),
    /// Offload as much as the codec safely can — the Stage-0 whole-input behavior.
    MaxReduction,
}

/// Map an operating point (% reduction) to an absolute token target from the raw count.
#[must_use]
pub fn target_for_reduction(raw_tokens: usize, reduction_pct: u8) -> usize {
    apply_reduction(raw_tokens, f32::from(reduction_pct.min(100)) / 100.0)
}

/// `round(raw * (1 - clamp(frac, 0, 1)))`. A NaN fraction is treated as 0 (passthrough).
/// Single source of truth for both [`target_for_reduction`] and [`Budget::Reduction`].
fn apply_reduction(raw_tokens: usize, frac: f32) -> usize {
    let frac = if frac.is_nan() {
        0.0
    } else {
        frac.clamp(0.0, 1.0)
    };
    (raw_tokens as f64 * (1.0 - f64::from(frac))).round() as usize
}

/// Compress `input` toward `budget`, byte-exact reversible. Keeps a head+tail window of whole JSON
/// array elements / log lines (dropped runs coalesced into `Ref`s) to approach the token target;
/// for logs, identical consecutive lines are deduplicated first so the budget is spent on distinct
/// content. Falls back to whole-input behavior when the input cannot be partitioned.
///
/// Unlike [`compress`](crate::compress), this honors the requested target regardless of
/// `MIN_COMPRESS_BYTES`: the caller asked for a specific budget, so even small inputs may
/// be offloaded.
#[must_use]
pub fn compress_to_budget(
    input: &[u8],
    cas: &dyn Cas,
    budget: Budget,
    counter: &dyn TokenCounter,
) -> CompressedDoc {
    match budget {
        Budget::MaxReduction => return offload_whole(input, cas),
        Budget::Reduction(p) if p.is_nan() || p <= 0.0 => return verbatim(input),
        Budget::Tokens(_) | Budget::Reduction(_) => {}
    }

    let content_type = detect(input);
    if content_type == ContentType::Text {
        return verbatim(input);
    }

    let target = match budget {
        Budget::Tokens(t) => t,
        Budget::Reduction(p) => {
            let raw = counter.count(&String::from_utf8_lossy(input));
            let target = apply_reduction(raw, p);
            if target >= raw {
                return verbatim(input);
            }
            target
        }
        Budget::MaxReduction => unreachable!("max reduction returned before detection"),
    };

    let (units, noun): (Vec<Range<usize>>, &str) = match content_type {
        ContentType::Json => match scan_top_level_array_elements(input) {
            Some(u) if !u.is_empty() => (u, "items"),
            // valid JSON but not a partitionable array (object, or scanner declined):
            // approach the budget with whole-input offload instead of guessing a tiling.
            _ => return whole_or_passthrough_to_budget(input, cas, target, counter),
        },
        ContentType::Log => {
            let u = scan_log_lines(input);
            if u.is_empty() {
                return verbatim(input);
            }
            (u, "lines")
        }
        ContentType::Text => unreachable!("text returned before budget target calculation"),
    };

    // For logs, dedup is free information-preserving compression applied first: only DISTINCT lines
    // are window candidates, so identical consecutive duplicates ALWAYS offload and the head+tail
    // window spends its budget on distinct content — distinct middle events survive a duplicate
    // flood a pure positional window would bury. JSON keeps every
    // element as a candidate (value-dedup stays the caller's choice via compress_dedup).
    let candidates: Vec<usize> = if content_type == ContentType::Log {
        dedup_survivor_mask(input, &units)
            .into_iter()
            .enumerate()
            .filter_map(|(i, survives)| survives.then_some(i))
            .collect()
    } else {
        (0..units.len()).collect()
    };

    // Largest M (kept candidates, split as a head+tail window) whose render fits `target`.
    // render_tokens(M) is monotone non-decreasing in M for any monotone tokenizer (one more kept
    // candidate ⇒ more kept verbatim bytes ⇒ ≥ as many tokens), so binary-search the largest fitting
    // M. A pathological non-monotone counter only costs optimality here — never reversibility.
    let n = candidates.len();
    // Fast path: when the counter's token count is a pure function of character count (the chars/4
    // heuristic — the proxy default), score each binary-search probe analytically in O(units) integer
    // work via a precomputed CharModel, instead of rendering + SHA-hashing + token-scanning the whole
    // document per probe. Subword tokenizers are not char-linear, so they keep the exact
    // render-per-probe path. Either way the final emit below is identical, so the bytes are unchanged.
    // Build the (tokenizer-independent) character curve once. For a char-linear counter it IS the
    // analytic token count; for a subword counter it guides probe placement.
    let char_model = CharModel::new(input, &units, noun);
    let analytic = counter.count_for_char_count(0).is_some();
    let chars = |m: usize| char_model.render_chars(&candidates, m);
    let render_tokens = |m: usize| -> usize {
        // Analytic fast path for a char-linear counter; otherwise (and as a defensive fallback if a
        // counter reports char-linearity inconsistently) count the real render exactly.
        if analytic
            && let Some(tokens) =
                counter.count_for_char_count(char_model.render_chars(&candidates, m))
        {
            return tokens;
        }
        counter.count(&render_keep_window_for_count(
            input,
            &units,
            &candidates,
            m,
            noun,
        ))
    };
    // Largest M whose render fits `target`. Cheap analytic probes → plain bisection (the proxy
    // default, unchanged). Expensive exact-tokenizer probes → char-curve interpolation that jumps
    // near the budget crossover, spending far fewer tokenizations. Both return the same
    // M for a monotone counter (proptested), so the chosen operating point is identical either way.
    let best = if analytic {
        search_keep_count_bisection(n, target, &render_tokens)
    } else {
        search_keep_count_interpolated(n, target, &chars, &render_tokens)
    };

    let doc = build_keep_window_over(input, &units, &candidates, best, noun, |b| cas.put(b));
    // emit_kept_with tiles the input exactly for any kept-set (verbatim runs ∪ coalesced Refs).
    debug_assert_eq!(
        tiled_len(&doc),
        input.len(),
        "segments must tile the input exactly"
    );
    doc
}

/// Largest `m` in `0..=n` with `tokens(m) <= target`, else `0` — plain bisection over the monotone
/// `tokens` curve. The reference the interpolated search below must agree with.
fn search_keep_count_bisection(n: usize, target: usize, tokens: &dyn Fn(usize) -> usize) -> usize {
    let (mut lo, mut hi, mut best) = (0usize, n, 0usize);
    while lo <= hi {
        let mid = lo + (hi - lo) / 2;
        if tokens(mid) <= target {
            best = mid;
            lo = mid + 1;
        } else if mid == 0 {
            break;
        } else {
            hi = mid - 1;
        }
    }
    best
}

/// Same answer as [`search_keep_count_bisection`] (largest `m` with `tokens(m) <= target`), but uses
/// the cheap monotone `chars` curve to INTERPOLATE the probe point — jumping near the token-budget
/// crossover so the EXPENSIVE exact `tokens` runs far fewer times. It only changes WHICH
/// `m` each step probes; every accept/reject still uses `tokens`, the bracket update is identical to
/// bisection, and a midpoint fallback whenever the interpolated point lands on a bracket edge keeps
/// the bracket strictly shrinking (worst case O(log n)). Both `chars` and `tokens` must be monotone
/// non-decreasing in `m`; for a monotone counter the result is exactly the bisection result.
fn search_keep_count_interpolated(
    n: usize,
    target: usize,
    chars: &dyn Fn(usize) -> usize,
    tokens: &dyn Fn(usize) -> usize,
) -> usize {
    let (mut lo, mut hi, mut best) = (0usize, n, 0usize);
    let mut sample: Option<(usize, usize)> = None; // (chars, tokens) of the last exact probe
    while lo <= hi {
        let mid = lo + (hi - lo) / 2;
        // Character count at which `tokens` is predicted to reach `target`: extrapolate from the
        // last exact (chars, tokens) sample, else assume ~4 chars/token for a good first jump.
        let c_star = match sample {
            Some((cs, ts)) if ts > 0 => {
                // widen to u128 (lossless from usize) so the product cannot overflow, then clamp back.
                usize::try_from(cs as u128 * target as u128 / ts as u128).unwrap_or(usize::MAX)
            }
            _ => target.saturating_mul(4),
        };
        let guess = largest_index_with_value_le(lo, hi, c_star, chars);
        // Trust only a strictly-interior guess; on a bracket edge, bisect so progress is guaranteed.
        let probe = if guess > lo && guess < hi { guess } else { mid };
        let t = tokens(probe);
        sample = Some((chars(probe), t));
        if t <= target {
            best = probe;
            lo = probe + 1;
        } else if probe == 0 {
            break;
        } else {
            hi = probe - 1;
        }
    }
    best
}

/// Largest `i` in `[lo, hi]` with `f(i) <= v` for monotone non-decreasing `f`, or `lo` if none —
/// a bisection on the cheap char curve, used to place the interpolated budget-search probe.
fn largest_index_with_value_le(
    lo: usize,
    hi: usize,
    v: usize,
    f: &dyn Fn(usize) -> usize,
) -> usize {
    let (mut a, mut b, mut best) = (lo, hi, lo);
    while a <= b {
        let mid = a + (b - a) / 2;
        if f(mid) <= v {
            best = mid;
            a = mid + 1;
        } else if mid == 0 {
            break;
        } else {
            b = mid - 1;
        }
    }
    best
}

fn verbatim(input: &[u8]) -> CompressedDoc {
    CompressedDoc {
        segments: vec![Segment::Verbatim(input.to_vec())],
    }
}

/// Pass through if the raw render already fits, else offload the whole input.
fn whole_or_passthrough_to_budget(
    input: &[u8],
    cas: &dyn Cas,
    target: usize,
    counter: &dyn TokenCounter,
) -> CompressedDoc {
    let raw = counter.count(&String::from_utf8_lossy(input));
    if raw <= target {
        verbatim(input)
    } else {
        offload_whole(input, cas)
    }
}

/// The droppable structural units of `input` (JSON array elements / log lines) plus a
/// noun for the offload summary. Text (and non-partitionable JSON) yields no units, which
/// callers treat as passthrough. coffer-eval's arms select over these same units, so every
/// arm shares one unit model (the basis for affordance parity).
#[must_use]
pub fn units_for(input: &[u8]) -> (Vec<Range<usize>>, &'static str) {
    match detect(input) {
        ContentType::Json => (
            scan_top_level_array_elements(input).unwrap_or_default(),
            "items",
        ),
        ContentType::Log => (scan_log_lines(input), "lines"),
        ContentType::Text => (Vec::new(), "bytes"),
    }
}

/// Keep the units marked `true`; offload each maximal run of dropped units as one
/// coalesced `Ref` into `cas`. Byte-exact reversible for ANY kept-set (proptested):
/// `reconstruct(emit_kept(..)) == input`. This is the shared mechanism behind every arm —
/// arms differ only in WHICH units they keep.
#[must_use]
pub fn emit_kept(
    input: &[u8],
    units: &[Range<usize>],
    kept: &[bool],
    cas: &dyn Cas,
    noun: &str,
) -> CompressedDoc {
    emit_kept_with(input, units, kept, noun, |b| cas.put(b))
}

/// Keep the units (JSON array elements / log lines) for which `keep` returns true and offload
/// the rest as coalesced `Ref`s — selection over the WHOLE set by predicate, byte-exact
/// reversible. `keep` receives each unit's raw bytes; parse JSON in the closure for
/// field predicates (e.g. keep the rows `WHERE phase = "X"`). `Text` (no units) passes through.
///
/// Unlike [`compress_to_budget`]'s positional leading-K, the kept-set is arbitrary/scattered.
/// Reversibility is inherited from the shared `emit_kept` tiler:
/// `reconstruct(compress_by_predicate(..)) == input` for any predicate.
#[must_use]
pub fn compress_by_predicate<P: Fn(&[u8]) -> bool>(
    input: &[u8],
    cas: &dyn Cas,
    keep: P,
) -> CompressedDoc {
    let (units, noun) = units_for(input);
    let kept: Vec<bool> = units.iter().map(|u| keep(&input[u.start..u.end])).collect();
    let doc = emit_kept(input, &units, &kept, cas, noun);
    // First public entry that routinely produces scattered, non-leading, multi-Ref kept-sets;
    // the segments must still tile the input exactly (reconstruct also re-verifies in release).
    debug_assert_eq!(
        tiled_len(&doc),
        input.len(),
        "segments must tile the input exactly"
    );
    doc
}

/// Collapse runs of identical consecutive units (log lines / JSON array elements): keep the first
/// of each run and offload the rest as a coalesced `Ref` (`uniq -c` semantics), byte-exact
/// reversible. For duplicate-heavy output — heartbeats, retry/stack-trace spam, repeated warnings —
/// this preserves every DISTINCT line while eliding the repeats, independent of any token budget;
/// it composes with detection (such logs now classify as `Log` via low first-token diversity).
/// `Text` (no units) passes through. Reversibility is inherited from `emit_kept`.
#[must_use]
pub fn compress_dedup(input: &[u8], cas: &dyn Cas) -> CompressedDoc {
    let (units, noun) = units_for(input);
    let kept = dedup_survivor_mask(input, &units);
    let doc = emit_kept(input, &units, &kept, cas, noun);
    debug_assert_eq!(
        tiled_len(&doc),
        input.len(),
        "segments must tile the input exactly"
    );
    doc
}

/// A comparison operator for [`compress_json_where`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
    /// Greater than (numeric, or lexical for strings).
    Gt,
    /// Greater than or equal.
    Ge,
    /// Less than.
    Lt,
    /// Less than or equal.
    Le,
}

/// Compare a record's field value to `target` under `op`. Numbers compare numerically (when both
/// are `f64`-representable), strings compare by value (`Eq`/`Ne`) and lexical order; any other type
/// pairing matches only `Eq`/`Ne` via full-value equality.
// `float_cmp` allow: WHERE-equality of two directly-parsed JSON numbers — exact equality is the
// intended semantics here, not an accumulated-float-error comparison.
#[allow(clippy::float_cmp)]
pub(crate) fn json_cmp(field_val: &Value, op: Op, target: &Value) -> bool {
    if let (Some(a), Some(b)) = (field_val.as_f64(), target.as_f64()) {
        return match op {
            Op::Eq => a == b,
            Op::Ne => a != b,
            Op::Gt => a > b,
            Op::Ge => a >= b,
            Op::Lt => a < b,
            Op::Le => a <= b,
        };
    }
    if let (Some(a), Some(b)) = (field_val.as_str(), target.as_str()) {
        return match op {
            Op::Eq => a == b,
            Op::Ne => a != b,
            Op::Gt => a > b,
            Op::Ge => a >= b,
            Op::Lt => a < b,
            Op::Le => a <= b,
        };
    }
    match op {
        Op::Eq => field_val == target,
        Op::Ne => field_val != target,
        _ => false,
    }
}

/// Convenience over [`compress_by_predicate`]: keep the JSON-array rows where
/// `field <op> value` and offload the rest byte-exact — e.g. `compress_json_where(input, cas,
/// "restarts", Op::Gt, 10)` or `(.., "phase", Op::Eq, "CrashLoopBackOff")`. A row that is not a
/// JSON object, or that lacks `field`, does not match. Reversibility is inherited unchanged.
#[must_use]
pub fn compress_json_where<V: Into<Value>>(
    input: &[u8],
    cas: &dyn Cas,
    field: &str,
    op: Op,
    value: V,
) -> CompressedDoc {
    let target = value.into();
    compress_by_predicate(input, cas, |u| {
        serde_json::from_slice::<Value>(u)
            .ok()
            .and_then(|v| v.as_object().and_then(|o| o.get(field).cloned()))
            .is_some_and(|fv| json_cmp(&fv, op, &target))
    })
}

/// General byte-exact tiling over an arbitrary kept-set. `put` stores (real CAS) or merely
/// hashes (`ContentHash::of`, during the budget search) the offloaded bytes. Bytes outside
/// any unit (structural glue) and kept units are emitted verbatim; the segments tile
/// `input` exactly regardless of the kept-set.
fn emit_kept_with<F: Fn(&[u8]) -> ContentHash>(
    input: &[u8],
    units: &[Range<usize>],
    kept: &[bool],
    noun: &str,
    put: F,
) -> CompressedDoc {
    debug_assert_eq!(units.len(), kept.len());
    let mut segments = Vec::new();
    let mut cursor = 0usize;
    let mut i = 0usize;
    while i < units.len() {
        if kept[i] {
            i += 1;
            continue;
        }
        // A maximal run of consecutive dropped units → one coalesced Ref.
        let run_start = units[i].start;
        let mut j = i;
        while j < units.len() && !kept[j] {
            j += 1;
        }
        let run_end = units[j - 1].end;
        if cursor < run_start {
            segments.push(Segment::Verbatim(input[cursor..run_start].to_vec()));
        }
        let original = &input[run_start..run_end];
        segments.push(Segment::Ref {
            hash: put(original),
            summary: format!("+{} {}", j - i, noun),
            original_len: original.len(),
        });
        cursor = run_end;
        i = j;
    }
    if cursor < input.len() {
        segments.push(Segment::Verbatim(input[cursor..].to_vec()));
    }
    if segments.is_empty() {
        segments.push(Segment::Verbatim(input.to_vec()));
    }
    CompressedDoc { segments }
}

/// Keep `m` of the `candidates` (the unit indices eligible to survive) as a **head+tail window** —
/// the first `ceil(m/2)` and last `floor(m/2)` candidates verbatim — and offload everything else:
/// windowed-out candidates AND every non-candidate (e.g. a log's deduplicated duplicate lines), each
/// maximal dropped run coalesced into one `Ref`. This is the window [`compress_to_budget`] sweeps
///: unlike the older leading-only keep, the tail (errors, summaries, most-recent rows)
/// survives; for logs the candidates are the dedup survivors, so the budget is spent on distinct
/// content. Delegates to the general [`emit_kept_with`]; reversibility is inherited unchanged.
fn build_keep_window_over<F: Fn(&[u8]) -> ContentHash>(
    input: &[u8],
    units: &[Range<usize>],
    candidates: &[usize],
    m: usize,
    noun: &str,
    put: F,
) -> CompressedDoc {
    let kept = keep_window_mask(units.len(), candidates, m);
    emit_kept_with(input, units, &kept, noun, put)
}

fn keep_window_mask(unit_count: usize, candidates: &[usize], m: usize) -> Vec<bool> {
    let (head_kept, tail_kept) = keep_window_slices(candidates, m);
    let mut kept = vec![false; unit_count];
    for &idx in head_kept.iter().chain(tail_kept) {
        kept[idx] = true;
    }
    kept
}

fn keep_window_slices(candidates: &[usize], m: usize) -> (&[usize], &[usize]) {
    let c = candidates.len();
    let m = m.min(c);
    let head = m.div_ceil(2);
    let tail = m / 2;
    if head + tail >= c {
        (candidates, &[])
    } else {
        (&candidates[..head], &candidates[c - tail..])
    }
}

fn render_keep_window_for_count(
    input: &[u8],
    units: &[Range<usize>],
    candidates: &[usize],
    m: usize,
    noun: &str,
) -> String {
    let (head_kept, tail_kept) = keep_window_slices(candidates, m);
    render_keep_slices_with(input, units, head_kept, tail_kept, noun)
}

fn render_keep_slices_with(
    input: &[u8],
    units: &[Range<usize>],
    head_kept: &[usize],
    tail_kept: &[usize],
    noun: &str,
) -> String {
    debug_assert!(
        head_kept
            .iter()
            .chain(tail_kept)
            .all(|&idx| idx < units.len())
    );
    debug_assert!(head_kept.windows(2).all(|w| w[0] < w[1]));
    debug_assert!(tail_kept.windows(2).all(|w| w[0] < w[1]));
    // The budget binary search calls this render O(log n) times per compressed tool_result; sizing
    // the buffer to the input up front avoids repeated grow-and-memcpy reallocations. Capacity is a
    // hint only — the rendered bytes are unchanged, so byte-exact round-trip is preserved.
    let mut out = String::with_capacity(input.len());
    let mut cursor = 0usize;
    let mut i = 0usize;
    let mut head_pos = 0usize;
    let mut tail_pos = 0usize;
    while i < units.len() {
        skip_stale_kept(i, head_kept, &mut head_pos);
        skip_stale_kept(i, tail_kept, &mut tail_pos);
        let next_kept = next_kept_index(head_kept, head_pos, tail_kept, tail_pos);
        if next_kept == Some(i) {
            skip_current_kept(i, head_kept, &mut head_pos);
            skip_current_kept(i, tail_kept, &mut tail_pos);
            i += 1;
            continue;
        }

        let run_start = units[i].start;
        let run_stop = next_kept.unwrap_or(units.len());
        debug_assert!(run_stop > i && run_stop <= units.len());
        let run_end = units[run_stop - 1].end;
        if cursor < run_start {
            push_lossy(&mut out, &input[cursor..run_start]);
        }
        let original = &input[run_start..run_end];
        out.push_str("<<cof:");
        ContentHash::push_short_of(original, &mut out);
        out.push(' ');
        out.push('+');
        out.push_str(&(run_stop - i).to_string());
        out.push(' ');
        out.push_str(noun);
        out.push_str(">>");
        cursor = run_end;
        i = run_stop;
    }
    if cursor < input.len() {
        push_lossy(&mut out, &input[cursor..]);
    }
    if out.is_empty() {
        push_lossy(&mut out, input);
    }
    out
}

fn next_kept_index(
    head_kept: &[usize],
    head_pos: usize,
    tail_kept: &[usize],
    tail_pos: usize,
) -> Option<usize> {
    match (head_kept.get(head_pos), tail_kept.get(tail_pos)) {
        (Some(&head), Some(&tail)) => Some(head.min(tail)),
        (Some(&head), None) => Some(head),
        (None, Some(&tail)) => Some(tail),
        (None, None) => None,
    }
}

fn skip_stale_kept(i: usize, kept: &[usize], pos: &mut usize) {
    while kept.get(*pos).is_some_and(|&idx| idx < i) {
        *pos += 1;
    }
}

fn skip_current_kept(i: usize, kept: &[usize], pos: &mut usize) {
    while kept.get(*pos) == Some(&i) {
        *pos += 1;
    }
}

fn push_lossy(out: &mut String, bytes: &[u8]) {
    out.push_str(&String::from_utf8_lossy(bytes));
}

/// Decimal digit count of `n` (`digit_count(0) == 1`, `digit_count(42) == 2`).
fn digit_count(mut n: usize) -> usize {
    let mut digits = 1;
    while n >= 10 {
        n /= 10;
        digits += 1;
    }
    digits
}

/// Character count of one offload sentinel exactly as [`render_keep_slices_with`] emits it:
/// `"<<cof:"` (6) + 12 hex hash + `" "` + `"+"` + decimal `units_in_run` + `" "` + noun + `">>"`.
fn sentinel_char_count(units_in_run: usize, noun_len: usize) -> usize {
    // 6 + 12 + 1 + 1 + digits + 1 + noun + 2 = 23 + digits + noun
    23 + digit_count(units_in_run) + noun_len
}

/// Analytic model of the budget render's CHARACTER count, so a character-linear token counter (the
/// chars/4 heuristic, the proxy default) can score a binary-search probe in O(units) integer work
/// instead of materializing + hashing + token-scanning the whole render per probe.
///
/// It reproduces `render_keep_window_for_count(..).chars().count()` exactly for JSON arrays and
/// logs: their unit boundaries always fall on ASCII delimiters (a value starts/ends on `"`, a
/// digit, `[`/`{`/`]`/`}`, and is surrounded by `,`/whitespace; log lines split on `\n`), so
/// `from_utf8_lossy` never straddles a boundary and per-segment character counts are additive.
/// Only the *measurement* changes — [`build_keep_window_over`] still emits the same bytes, so the
/// byte-exact round-trip invariant is untouched.
struct CharModel {
    /// Character count of the full render with nothing dropped (all units verbatim).
    all_chars: usize,
    /// Prefix sums of per-unit character counts; `unit_chars_pfx[k]` = chars in units `0..k`.
    unit_chars_pfx: Vec<usize>,
    /// Prefix sums of inter-unit gap character counts; `gap_chars_pfx[k]` = chars in gaps `0..k`,
    /// where gap `u` is the glue between unit `u` and unit `u+1`.
    gap_chars_pfx: Vec<usize>,
    noun_len: usize,
    unit_count: usize,
}

impl CharModel {
    fn new(input: &[u8], units: &[Range<usize>], noun: &str) -> Self {
        let n = units.len();
        let chars = |a: usize, b: usize| String::from_utf8_lossy(&input[a..b]).chars().count();

        let head = if n == 0 { 0 } else { chars(0, units[0].start) };
        let tail = if n == 0 {
            0
        } else {
            chars(units[n - 1].end, input.len())
        };

        let mut unit_chars_pfx = Vec::with_capacity(n + 1);
        unit_chars_pfx.push(0);
        let mut unit_acc = 0usize;
        for u in units {
            unit_acc += chars(u.start, u.end);
            unit_chars_pfx.push(unit_acc);
        }

        let mut gap_chars_pfx = Vec::with_capacity(n.max(1));
        gap_chars_pfx.push(0);
        let mut gap_acc = 0usize;
        for w in units.windows(2) {
            gap_acc += chars(w[0].end, w[1].start);
            gap_chars_pfx.push(gap_acc);
        }

        Self {
            all_chars: head + tail + unit_acc + gap_acc,
            unit_chars_pfx,
            gap_chars_pfx,
            noun_len: noun.len(),
            unit_count: n,
        }
    }

    /// Character count of the render that keeps `m` candidates as a head+tail window. Equals
    /// `render_keep_window_for_count(input, units, candidates, m, noun).chars().count()`.
    fn render_chars(&self, candidates: &[usize], m: usize) -> usize {
        // Same kept-set the renderer derives, so dropped runs coalesce identically.
        let kept = keep_window_mask(self.unit_count, candidates, m);
        let mut total = self.all_chars;
        let mut i = 0usize;
        while i < self.unit_count {
            if kept[i] {
                i += 1;
                continue;
            }
            let run_start = i;
            let mut j = i;
            while j < self.unit_count && !kept[j] {
                j += 1;
            }
            // The dropped run is units [run_start, j): replace its verbatim characters (its own
            // units plus the interior gaps between them) with one sentinel's characters.
            let run_chars = (self.unit_chars_pfx[j] - self.unit_chars_pfx[run_start])
                + (self.gap_chars_pfx[j - 1] - self.gap_chars_pfx[run_start]);
            total =
                total.saturating_sub(run_chars) + sentinel_char_count(j - run_start, self.noun_len);
            i = j;
        }
        total
    }
}

/// Mask marking the first unit of each run of identical consecutive units — the `uniq` survivors.
fn dedup_survivor_mask(input: &[u8], units: &[Range<usize>]) -> Vec<bool> {
    units
        .iter()
        .enumerate()
        .map(|(i, u)| {
            i == 0 || input[u.start..u.end] != input[units[i - 1].start..units[i - 1].end]
        })
        .collect()
}

fn tiled_len(doc: &CompressedDoc) -> usize {
    doc.segments
        .iter()
        .map(|s| match s {
            Segment::Verbatim(b) => b.len(),
            Segment::Ref { original_len, .. } => *original_len,
        })
        .sum()
}

/// Top-level element value byte-ranges of a JSON array, or `None` if `input` is not a
/// well-formed top-level array. String/escape/depth aware, so a comma or bracket inside a
/// string never splits an element. Total by design: any ambiguity returns `None` and the
/// caller falls back to whole-input — a wrong partial tiling is never produced.
///
/// (Even if this were buggy, reversibility would hold: the emitted `Ref` is an exact byte
/// slice and verbatim fills cover the rest. A bad range only hurts *effectiveness*.)
pub(crate) fn scan_top_level_array_elements(input: &[u8]) -> Option<Vec<Range<usize>>> {
    let n = input.len();
    let mut i = 0usize;
    while i < n && input[i].is_ascii_whitespace() {
        i += 1;
    }
    if i >= n || input[i] != b'[' {
        return None;
    }
    i += 1; // past '['

    let mut elems: Vec<Range<usize>> = Vec::new();
    loop {
        while i < n && input[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= n {
            return None; // unterminated
        }
        if input[i] == b']' {
            i += 1;
            while i < n && input[i].is_ascii_whitespace() {
                i += 1;
            }
            return if i == n { Some(elems) } else { None };
        }

        // Scan one element value until a top-level ',' or ']'.
        let start = i;
        let mut depth = 0i32;
        let mut in_str = false;
        let mut esc = false;
        while i < n {
            let c = input[i];
            if in_str {
                if esc {
                    esc = false;
                } else if c == b'\\' {
                    esc = true;
                } else if c == b'"' {
                    in_str = false;
                }
                i += 1;
                continue;
            }
            match c {
                b'"' => in_str = true,
                b'[' | b'{' => depth += 1,
                // top-level ']' / '}' / ',' ends this element's value
                b']' | b'}' | b',' if depth == 0 => break,
                b']' | b'}' => depth -= 1,
                _ => {}
            }
            i += 1;
        }

        // Trim trailing whitespace off the value; surrounding glue stays in verbatim fills.
        let mut end = i;
        while end > start && input[end - 1].is_ascii_whitespace() {
            end -= 1;
        }
        if end <= start {
            return None; // empty element, e.g. "[,]" — treat as malformed, fall back
        }
        elems.push(start..end);

        if i >= n {
            return None;
        }
        match input[i] {
            b',' => i += 1,
            b']' => {
                i += 1;
                while i < n && input[i].is_ascii_whitespace() {
                    i += 1;
                }
                return if i == n { Some(elems) } else { None };
            }
            _ => return None,
        }
    }
}

/// Newline-delimited line ranges; each range includes its terminator so concatenation is
/// byte-exact. A trailing line without a final newline is included.
pub(crate) fn scan_log_lines(input: &[u8]) -> Vec<Range<usize>> {
    let mut lines = Vec::new();
    let mut start = 0usize;
    for (i, &b) in input.iter().enumerate() {
        if b == b'\n' {
            lines.push(start..i + 1);
            start = i + 1;
        }
    }
    if start < input.len() {
        lines.push(start..input.len());
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use serde_json::Value;

    /// Arbitrary JSON values, including NESTED arrays/objects, so the scanner's depth
    /// tracking (not just string handling) is fuzzed against serde.
    fn json_value() -> impl Strategy<Value = Value> {
        let leaf = prop_oneof![
            any::<i64>().prop_map(Value::from),
            "[a-z, \\[\\]{}\"]{0,12}".prop_map(Value::String),
        ];
        leaf.prop_recursive(3, 32, 4, |inner| {
            prop_oneof![
                proptest::collection::vec(inner.clone(), 0..4).prop_map(Value::Array),
                proptest::collection::vec(("[a-z]{1,4}", inner), 0..4)
                    .prop_map(|kvs| Value::Object(kvs.into_iter().collect())),
            ]
        })
    }

    proptest! {
        /// The JSON scanner agrees with serde on element count, and every claimed span
        /// re-parses as a valid JSON value — even with commas/brackets/quotes inside strings
        /// and with nested arrays/objects as elements.
        #[test]
        fn scanner_matches_serde(elems in proptest::collection::vec(json_value(), 0..20)) {
            let array = Value::Array(elems);
            let bytes = serde_json::to_vec(&array).unwrap();
            let spans = scan_top_level_array_elements(&bytes).expect("valid array scans");
            prop_assert_eq!(spans.len(), array.as_array().unwrap().len());
            for span in &spans {
                prop_assert!(span.start < span.end && span.end <= bytes.len());
                prop_assert!(serde_json::from_slice::<Value>(&bytes[span.clone()]).is_ok());
            }
        }
    }

    #[test]
    fn scanner_handles_pretty_and_in_string_delimiters() {
        let input = br#"[
            {"a": "x,y", "b": "]["},
            42,
            ["nested", "[,]"]
        ]"#;
        let spans = scan_top_level_array_elements(input).expect("scans");
        assert_eq!(spans.len(), 3);
    }

    #[test]
    fn scanner_rejects_non_array() {
        assert!(scan_top_level_array_elements(br#"{"a":1}"#).is_none());
        assert!(scan_top_level_array_elements(b"42").is_none());
    }

    #[test]
    fn log_lines_tile_exactly() {
        for input in [&b"a\nb\nc\n"[..], &b"a\nb\nc"[..], &b""[..], &b"\n"[..]] {
            let lines = scan_log_lines(input);
            let total: usize = lines.iter().map(|r| r.len()).sum();
            assert_eq!(total, input.len());
        }
    }

    #[test]
    fn budget_probe_render_matches_doc_render() {
        let json = br#"[{"id":0},{"id":1},{"id":2},{"id":3},{"id":4}]"#;
        let json_units = scan_top_level_array_elements(json).expect("json array scans");
        let json_candidates: Vec<usize> = (0..json_units.len()).collect();
        assert_probe_render_matches_doc_render(json, &json_units, &json_candidates, "items");

        let log = b"INFO boot\nINFO boot\nWARN retry\nERROR fail\nERROR fail\n";
        let log_units = scan_log_lines(log);
        let log_candidates: Vec<usize> = dedup_survivor_mask(log, &log_units)
            .into_iter()
            .enumerate()
            .filter_map(|(i, survives)| survives.then_some(i))
            .collect();
        assert_probe_render_matches_doc_render(log, &log_units, &log_candidates, "lines");
    }

    fn assert_probe_render_matches_doc_render(
        input: &[u8],
        units: &[Range<usize>],
        candidates: &[usize],
        noun: &str,
    ) {
        for m in 0..=candidates.len() + 2 {
            let probe = render_keep_window_for_count(input, units, candidates, m, noun);
            let doc = build_keep_window_over(input, units, candidates, m, noun, ContentHash::of);
            assert_eq!(probe, doc.render_for_model(), "m={m}");
        }
    }

    fn assert_char_model_matches_render(
        input: &[u8],
        units: &[Range<usize>],
        candidates: &[usize],
        noun: &str,
    ) {
        let model = CharModel::new(input, units, noun);
        for m in 0..=candidates.len() + 1 {
            let real = render_keep_window_for_count(input, units, candidates, m, noun)
                .chars()
                .count();
            assert_eq!(model.render_chars(candidates, m), real, "m={m}");
        }
    }

    proptest! {
        /// The analytic CharModel reproduces the real render's character count exactly for every
        /// head+tail window size — so the char-linear budget search selects the same keep-count it
        /// would by rendering each probe. Fuzzed over arbitrary JSON arrays, including
        /// nested elements and in-string delimiters.
        #[test]
        fn analytic_char_count_matches_render_for_json(elems in proptest::collection::vec(json_value(), 1..16)) {
            let input = serde_json::to_vec(&Value::Array(elems)).unwrap();
            let units = scan_top_level_array_elements(&input).expect("valid array scans");
            prop_assume!(!units.is_empty());
            let candidates: Vec<usize> = (0..units.len()).collect();
            assert_char_model_matches_render(&input, &units, &candidates, "items");
        }
    }

    #[test]
    fn analytic_char_count_matches_render_for_logs() {
        // Includes consecutive duplicates so the dedup-survivor candidate set (a subset of units)
        // and the non-candidate dropped units are both exercised.
        let log = b"INFO boot\nINFO boot\nWARN retry 3 times\nERROR fail\nERROR fail\nDEBUG ok\nINFO done\n";
        let units = scan_log_lines(log);
        let candidates: Vec<usize> = dedup_survivor_mask(log, &units)
            .into_iter()
            .enumerate()
            .filter_map(|(i, survives)| survives.then_some(i))
            .collect();
        assert_char_model_matches_render(log, &units, &candidates, "lines");
    }

    #[test]
    fn budget_heuristic_fast_path_stays_byte_exact_across_targets() {
        // The heuristic-counter fast path drives the search analytically; the chosen operating
        // point must still reconstruct byte-for-byte and never exceed the requested target above
        // the structural floor (selection equivalence vs the renderer is covered by the analytic
        // char-count proptests above).
        let cas = coffer_cas::MemoryCas::new();
        let counter = coffer_tokenizer::HeuristicCounter;
        let input =
            br#"[{"id":0,"v":"alpha"},{"id":1,"v":"bravo"},{"id":2,"v":"charlie"},{"id":3,"v":"delta"},{"id":4,"v":"echo"},{"id":5,"v":"foxtrot"}]"#;
        let floor = counter.count(
            &compress_to_budget(input, &cas, Budget::Tokens(0), &counter).render_for_model(),
        );
        for target in [1usize, 4, 8, 12, 20, 40, 80, 200] {
            let doc = compress_to_budget(input, &cas, Budget::Tokens(target), &counter);
            assert_eq!(
                doc.reconstruct(&cas).unwrap(),
                &input[..],
                "target={target}"
            );
            if target >= floor {
                assert!(
                    counter.count(&doc.render_for_model()) <= target,
                    "target={target} exceeded by render of {} tokens",
                    counter.count(&doc.render_for_model())
                );
            }
        }
    }

    /// A non-char-linear (subword-like) counter for exercising the interpolation path:
    /// `count_for_char_count` defaults to `None`, so `compress_to_budget` takes the exact-probe path.
    struct Subwordish;
    impl coffer_tokenizer::TokenCounter for Subwordish {
        fn count(&self, s: &str) -> usize {
            s.split(|c: char| !c.is_ascii_alphanumeric())
                .filter(|w| !w.is_empty())
                .count()
                + s.len() / 4
        }
        fn model_label(&self) -> &'static str {
            "subwordish-test"
        }
    }

    proptest! {
        ///: the interpolation-guided search returns the IDENTICAL keep-count plain bisection
        /// finds, for arbitrary monotone (chars, tokens) curves and targets — so swapping it in changes
        /// only how many probes run, never the chosen operating point. A non-zero floor at m=0 exercises
        /// the "even keeping nothing exceeds the target" break path.
        #[test]
        fn interpolated_search_matches_bisection(
            deltas in proptest::collection::vec(0usize..30, 0..48),
            target in 0usize..700,
        ) {
            let n = deltas.len();
            let (mut tok_cum, mut chr_cum) = (vec![3usize], vec![12usize]); // m=0 sentinel floor
            for (i, d) in deltas.iter().enumerate() {
                let last_t = *tok_cum.last().unwrap();
                let last_c = *chr_cum.last().unwrap();
                tok_cum.push(last_t + d);
                chr_cum.push(last_c + d * 4 + (i % 3)); // ~4 chars/token, slightly nonlinear
            }
            let tokens = |m: usize| tok_cum[m.min(n)];
            let chars = |m: usize| chr_cum[m.min(n)];
            let a = search_keep_count_bisection(n, target, &tokens);
            let b = search_keep_count_interpolated(n, target, &chars, &tokens);
            prop_assert_eq!(a, b, "target={}, n={}", target, n);
        }

        /// The same equivalence over the REAL head+tail render token curve (a subword counter) for
        /// arbitrary JSON arrays and targets — the interpolated search and bisection pick the same M,
        /// so the emitted bytes (and thus reversibility) are unchanged by.
        #[test]
        fn interpolated_matches_bisection_on_real_render(
            elems in proptest::collection::vec(json_value(), 1..20),
            target in 0usize..400,
        ) {
            let input = serde_json::to_vec(&Value::Array(elems)).unwrap();
            let units = scan_top_level_array_elements(&input).expect("array scans");
            prop_assume!(!units.is_empty());
            let candidates: Vec<usize> = (0..units.len()).collect();
            let n = candidates.len();
            let char_model = CharModel::new(&input, &units, "items");
            let counter = Subwordish;
            let chars = |m: usize| char_model.render_chars(&candidates, m);
            let tokens = |m: usize| {
                counter.count(&render_keep_window_for_count(&input, &units, &candidates, m, "items"))
            };
            prop_assert_eq!(
                search_keep_count_bisection(n, target, &tokens),
                search_keep_count_interpolated(n, target, &chars, &tokens)
            );
        }
    }

    #[test]
    fn budget_subword_path_byte_exact_and_in_target() {
        // The exact-tokenizer path must still reconstruct byte-for-byte and
        // land within target above the structural floor — same guarantees as the char-linear path.
        let cas = coffer_cas::MemoryCas::new();
        let counter = Subwordish;
        let mut s = String::from("[");
        for i in 0..200 {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!(r#"{{"id":{i},"v":"item-{i}-payload"}}"#));
        }
        s.push(']');
        let input = s.as_bytes();
        let floor = counter.count(
            &compress_to_budget(input, &cas, Budget::Tokens(0), &counter).render_for_model(),
        );
        for target in [1usize, 10, 50, 100, 300, 1000] {
            let doc = compress_to_budget(input, &cas, Budget::Tokens(target), &counter);
            assert_eq!(doc.reconstruct(&cas).unwrap(), input, "target={target}");
            if target >= floor {
                assert!(
                    counter.count(&doc.render_for_model()) <= target,
                    "target={target} exceeded: {} tokens",
                    counter.count(&doc.render_for_model())
                );
            }
        }
    }

    #[test]
    fn predicate_keeps_matching_and_reconstructs() {
        let cas = coffer_cas::MemoryCas::new();
        let input = br#"[{"id":1,"keep":true},{"id":2,"keep":false},{"id":3,"keep":true}]"#;
        // keep the rows where the `keep` field is true; the middle row is offloaded.
        let doc = compress_by_predicate(input, &cas, |u| {
            serde_json::from_slice::<Value>(u)
                .ok()
                .and_then(|v| v.get("keep").and_then(Value::as_bool))
                .unwrap_or(false)
        });
        assert_eq!(
            doc.reconstruct(&cas).unwrap(),
            input,
            "byte-exact reversible"
        );
        assert!(
            doc.segments
                .iter()
                .any(|s| matches!(s, Segment::Ref { .. }))
        );
        let render = doc.render_for_model();
        assert!(render.contains(r#""id":1"#) && render.contains(r#""id":3"#));
        assert!(
            !render.contains(r#""id":2"#),
            "the non-matching row is offloaded, not rendered"
        );
    }

    #[test]
    fn dedup_collapses_consecutive_duplicate_units() {
        let cas = coffer_cas::MemoryCas::new();
        // three identical leading items collapse to the first + a coalesced Ref; distinct items stay.
        let input = br#"[{"x":1},{"x":1},{"x":1},{"x":2},{"x":1}]"#;
        let doc = compress_dedup(input, &cas);
        assert_eq!(
            doc.reconstruct(&cas).unwrap(),
            &input[..],
            "byte-exact reversible"
        );
        let r = doc.render_for_model();
        assert!(r.contains(r#"{"x":2}"#), "the distinct item survives: {r}");
        assert!(
            r.contains("cof:"),
            "the duplicate run is offloaded to a sentinel: {r}"
        );
        assert!(
            doc.segments
                .iter()
                .any(|s| matches!(s, Segment::Ref { .. }))
        );
    }

    #[test]
    fn json_where_numeric_and_string_predicates() {
        let cas = coffer_cas::MemoryCas::new();
        let input = br#"[{"name":"a","phase":"Running","restarts":0},{"name":"b","phase":"CrashLoopBackOff","restarts":12},{"name":"c","phase":"Running","restarts":3}]"#;
        // numeric: restarts > 10 keeps only b
        let hi = compress_json_where(input, &cas, "restarts", Op::Gt, 10);
        assert_eq!(hi.reconstruct(&cas).unwrap(), input);
        let r = hi.render_for_model();
        assert!(r.contains(r#""name":"b""#));
        assert!(!r.contains(r#""name":"a""#) && !r.contains(r#""name":"c""#));
        // string: phase == "Running" keeps a and c, offloads b
        let cas2 = coffer_cas::MemoryCas::new();
        let running = compress_json_where(input, &cas2, "phase", Op::Eq, "Running");
        assert_eq!(running.reconstruct(&cas2).unwrap(), input);
        let r2 = running.render_for_model();
        assert!(r2.contains(r#""name":"a""#) && r2.contains(r#""name":"c""#));
        assert!(!r2.contains(r#""name":"b""#));
    }

    proptest! {
        /// reconstruct == input for an ARBITRARY (scattered) predicate over arbitrary JSON arrays —
        /// predicate-retrieve inherits emit_kept's byte-exact tiling.
        #[test]
        fn predicate_round_trips_for_any_predicate(
            elems in proptest::collection::vec(json_value(), 0..16),
            bits in any::<u64>(),
        ) {
            let input = serde_json::to_vec(&Value::Array(elems)).unwrap();
            let cas = coffer_cas::MemoryCas::new();
            let idx = std::cell::Cell::new(0u32);
            let doc = compress_by_predicate(&input, &cas, |_unit| {
                let i = idx.get();
                idx.set(i + 1);
                (bits >> (i % 64)) & 1 == 1
            });
            prop_assert_eq!(doc.reconstruct(&cas).expect("reconstruct"), input);
        }
    }
}
