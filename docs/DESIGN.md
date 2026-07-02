# coffer — Design & Specification

This document specifies what coffer does and how it works. It describes **shipped behavior only**; it is
not a roadmap. coffer is a Rust engine that compresses LLM tool-output before it enters the model's
context window, while keeping the original bytes recoverable exactly.

Two properties are mechanical and testable, independent of any accuracy claim:

1. **Byte-exact reversibility** — the original bytes are never lost, only relocated.
2. **An exact compute-digest** — exact aggregates computed over *all* of the data, including the bytes
   that were offloaded out of context.

## 1. The reversibility invariant

For every input `x` and budget,

```
reconstruct(compress(x)) == x        // byte-for-byte
```

This is a property-tested invariant, not a goal. It holds because compression never discards bytes: it
**relocates** them. `compress` partitions the input into structural units, chooses a kept set, and emits
each maximal run of dropped units as a single reference into a content-addressed store (CAS). The emitted
segments **tile the input exactly**, so any kept-set reconstructs through one shared, fuzz-tested tiler.

`reconstruct` walks the segments back into the original byte stream. On every reference it re-checks the
recorded length and **re-verifies the SHA-256 hash of the bytes it read from the CAS**, returning a typed
error (`MissingOriginal` / `LengthMismatch` / `HashMismatch`) rather than ever returning wrong bytes.

Reversibility is independent of what the model sees: the **model-facing render** may window, deduplicate,
and replace runs with short sentinels, but recovery always returns the original bytes from the CAS.

## 2. Data model

A compressed document is a sequence of segments:

```
Segment::Verbatim(bytes)                                  // kept inline
Segment::Ref { hash: ContentHash, summary, original_len } // offloaded to the CAS
```

The **content-addressed store** keys blobs by their SHA-256 hash:

- **Verified on read.** Stored rows are re-hashed before entering the in-RAM cache; a corrupt row is
  skipped, not trusted. The get path re-verifies `ContentHash::of(bytes) == hash` before returning.
- **Durable.** The SQLite backend uses `cas(hash TEXT PRIMARY KEY, bytes BLOB) WITHOUT ROWID`, WAL
  journaling with `synchronous = NORMAL`, and an idempotent `INSERT OR IGNORE` put. Get-misses are served
  by a long-lived WAL reader using a cached primary-key lookup. An in-memory store is also available with
  no dependencies.
- **Bounded RAM.** Historical bytes load lazily on first get; `COFFER_CAS_RESIDENT_CAP_MB` caps the
  read-cache RAM by evicting only cached buffers — the durable rows (and thus reversibility) are untouched.

The model sees a short **sentinel** in place of an offloaded run:

```
<<cof:HASH summary>>     // HASH is a 12-char hex prefix of the content hash
```

## 3. Compression pipeline

1. **Detect content type** — `Json`, `Log`, or `Text`. Text passes through verbatim at Stage 0;
   under an **explicit budget** it partitions into physical lines like a log (a single unbroken
   line approaches the budget via whole-input offload instead) — for an oversized diff, file dump,
   or prose blob, "the model reads it whole" is no longer an option, and reversibility is
   structural either way.
2. **Partition into units** — JSON: top-level array elements; logs: lines (detected by a high proportion
   of timestamp/bracket-prefixed lines, or low first-token diversity).
3. **Window** — keep a head+tail window of whole units so the start *and* end of the payload survive; the
   middle is the candidate for offload.
4. **Dedup (logs)** — identical log lines collapse first (timestamp-insensitive key, first occurrence
   kept, `(xN)` annotated), so the budget is spent on distinct events. This stays byte-exact reversible.
5. **Offload** — each maximal run of dropped units becomes one coalesced reference into the CAS.
6. **Render** — emit the kept units plus sentinels for offloaded runs. The render is a *display*
   projection; it is never the source of truth for recovery.

JSON is parsed once and re-serialized with preserved key order, so a passed-through document is byte-stable.

## 4. Budget search

To hit a token budget, coffer keeps the **largest number of units whose rendered token count fits the
target** (more kept units never decrease the token count, so the fit point is found by search).

- **Analytic fast path.** When the token counter is a pure function of character count, each probe is
  scored in O(kept-units) integer arithmetic from precomputed prefix sums — no re-render per probe. The
  chosen window is rendered exactly once at the end.
- **Interpolation for subword tokenizers.** For real BPE counters (where token cost is not character-linear)
  coffer builds a cheap character curve once and uses it to *interpolate* the next probe toward the budget
  crossover, calling the expensive tokenizer only near the answer, with a bisection fallback. It returns the
  **same** keep-count a plain bisection would — fewer tokenizer calls, identical result.

Optimality assumes the token count is monotone in the keep-count; a pathological non-monotone counter still
reconstructs byte-exact, it only loses optimality of the operating point.

## 5. The compute-digest

Some questions have no answer in any single surviving unit — "how many errors?", "the record with the
largest value", "the p95". coffer answers them **exactly, in Rust, over all units including the offloaded
ones** (computed over the CAS-stored originals, never over the lossy render), with no model call.

Operators: `count` (plain / threshold / group / distinct), `sum`, `mean`, `median`, `percentile`, `range`,
`argmax`/`argmin`, and group-by with a `sum`/`mean` aggregate. Contract:

- **Refuse rather than guess.** If a field is mixed-type, or carries a number outside the representable
  range, the digest returns *nothing* for that query instead of computing over a silently-dropped subset.
- **Ties are reported, never broken.** An argmax/argmin or group extreme with more than one winner reports
  every winning key ("no unique maximum"), deterministically; it never picks one arbitrarily.
- **Composite group-by.** Grouping can compose several named category fields into one stable key.

Beyond the natural-language form, a **typed query-aggregate** takes a conjunction of structured
predicates (`field <op> value`, AND-ed) and an aggregate (`count` / `sum` / `mean` / `min` / `max`),
so filters like `value > 10 AND status == "error"` — which a single English query cannot express —
are exact. It returns the answer **and the indices of the backing records** (provenance), so a caller
can fetch exactly those rows and re-verify the number byte-for-byte. The same refuse-rather-than-guess
contract holds: a present-but-non-numeric aggregated value returns nothing.

A query can also **produce held data instead of a number**. A typed predicate conjunction can be applied
to a held JSON array to yield a *new array of just the matching records*, each row copied byte-for-byte
from the original. That derived array is itself a payload: another query, digest, or subset composes over
it. So a caller can narrow `errors → slow errors → group by region` in steps, none of the rows ever
entering the context window, and the algebra is exact — chaining two filters equals the single AND-ed
filter byte-for-byte, and aggregating a subset equals aggregating the original under the same predicate.
Per-row bytes are preserved; the union array is a fresh document, not a slice of the original.

Questions can also span **two held datasets**. A semi-join aggregate correlates a left array with a right
array on an equi-join key and aggregates the left rows that have a qualifying match — "sum order.amount
for orders whose customer is gold-tier" — entirely in Rust, over every row of both, returning one scalar
plus the joined left indices. Neither dataset enters the context window, join keys compare by value with
type sensitivity, and a duplicated key never double-counts a row. This is the exact, bounded form of a
correlation an agent would otherwise have to pull both tables into the prompt to attempt.

Two shapes of **grouped** aggregate extend this. A project-join group-by groups the joined left rows by a
right-side attribute — "revenue **by** customer.region" — bringing a dimension column onto the grouping,
and refusing rather than guessing if a join key maps to conflicting group values. A numeric bucketing
aggregate summarizes a *distribution* — "count **per** 100 ms latency band" — by `floor(v / width)`
buckets ordered by lower bound. Both return one entry per group, each with its own provenance, computed
over every held row.

The same grouping applies to unstructured **logs**: a windowed histogram counts case-insensitive matches
of a pattern per N-line block — "how many ERROR lines per 1000-line block" — so an agent sees where an
event clusters across a long log without reading it, with the matching line numbers as each block's
provenance.

The digest is additive display text alongside the byte-exact compressed document, so it never costs
reversibility.

## 6. Surfaces

coffer exposes the engine three ways, over **one shared content store**, so a payload compressed on one
side is recoverable on the other.

- **Transparent HTTP proxy.** Point an agent's base URL at it. It rewrites only the large tool-output
  values of each request — `tool_result` blocks (Anthropic Messages) and `*_call_output` items (OpenAI
  Responses) — offloads their bytes to the CAS, injects sentinels, and leaves every other value untouched,
  streaming the response back. **Only tool-output values are changed**; authored system/user/assistant text
  is preserved exactly, object key order and numeric precision are kept (the JSON model uses
  `preserve_order` + `arbitrary_precision`), and the body is re-emitted as compact JSON — so a request that
  is already compact round-trips byte-for-byte, while a pretty-printed or `\u`-escaped sender is normalized
  to compact form (semantically identical, not byte-identical). When nothing compresses, the original bytes
  are forwarded verbatim. The transform is **fail-open**: any unexpected body shape or error forwards the
  original bytes unchanged. The per-block budget is a **hybrid**: cut a fraction of the block's tokens
  (`COFFER_PROXY_REDUCTION`, default 0.8) but never keep more than an absolute ceiling
  (`COFFER_PROXY_MAX_KEPT_TOKENS`, default 4000 heuristic tokens; 0 disables) — proportional reduction
  alone has an unbounded remainder (a ~915k-token block would still keep ~183k at 0.8), so the ceiling
  makes the kept size scale-invariant on exactly the oversized results that motivate the rewrite.
  Plain-text blocks (prose, code, diffs, file dumps) keep a stronger restraint: they are rewritten
  only once they exceed the ceiling — below it the model reads them whole — and then with pure
  ceiling semantics, no proportional cut. Every rewritten block leads with a one-line **sentinel explainer** — the model
  only ever learns about coffer in-band, inside tool output, so this line is what keeps a `<<cof:…>>`
  marker from reading as truncation or corruption: it states that elided bytes are preserved and
  recoverable, how to query them exactly when coffer tools are registered, and that elided content must
  never be guessed (`COFFER_PROXY_EXPLAIN=0` disables it; the shrink gate counts the explainer, so a
  rewrite never grows a block). (A byte-exact span patcher that rewrites only the tool-output byte ranges
  is a deferred option.)
- **MCP server.** Tools to direct at server-held output instead of reading it: `coffer_digest` (exact
  aggregates from a natural-language ask), `coffer_describe` (a shape-generic exact summary of any record
  set — row count, per-field stats, count-by-value — with no per-tool/per-format code, the lossless
  counterpart to a lossy trimmer's hand-tuned summary), `coffer_aggregate` (the typed, unambiguous
  counterpart — a
  `count|sum|mean|min|max` over a predicate conjunction, returned with the backing row indices so the
  number can be audited via `coffer_pick`), `coffer_query` (keep rows matching `field op value`),
  `coffer_select` (filter by a
  conjunction of predicates and hold the matches as a **new handle**, so an agent narrows a dataset in
  composable steps entirely server-side), `coffer_pick` (pull the rows at explicit indices — e.g. a
  digest's provenance — as a new handle, so an exact aggregate can be audited by re-fetching and recounting
  its backing rows), `coffer_bucket` (a numeric bucketing histogram — `agg` per `floor(value / width)`
  band, §5), `coffer_window` (a windowed log histogram — case-insensitive matches per N-line block, §5),
  `coffer_join` (the two-dataset semi-join / project-join group-by of §5 — aggregate the left rows whose
  join key has a qualifying match in the right, optionally grouped by a right attribute),
  `coffer_check_claim` (recompute a claimed number over the held bytes and answer AGREE/DISAGREE with the
  matched row indices — a lie detector for agent-reported aggregates), `coffer_receipt` /
  `coffer_verify_receipt` (issue and later re-execute a portable exactness receipt — the typed query, the
  value, the backing row indices, and SHA-256 over the backing rows and the input — with verdicts
  `VALID` / `VALUE_MISMATCH` / `BACKING_TAMPERED` / `REFUSED` / `MALFORMED_RECEIPT`; verification is pure
  re-derivation, no model or network),
  `coffer_search` / `coffer_lines` (drill into logs), `coffer_rows`
  / `coffer_json` (windowed JSON), `coffer_retrieve` / `coffer_unfold` (bounded byte windows),
  `coffer_ingest` (hold a file), `coffer_run` (capture a shell command's output server-side; disabled
  unless `COFFER_MCP_ENABLE_RUN=1`), `coffer_status` (diagnostics). Ingest and select return a content-free
  fact card (type, size, row count, field stats).
- **MCP gateway (`coffer-wrap`).** Wraps any stdio MCP server as a child process (`coffer-wrap --
  <command>`) and relays JSON-RPC with exactly two interventions, both fail-open: `tools/list` responses
  gain a small set of injected query tools (collision-aware — a downstream tool with the same name is never
  shadowed; the injected tool is renamed or skipped), and `tools/call` text content over a token threshold
  (default 10k, the hosts' warning band) is stored byte-exact in the CAS and replaced with a fact card
  carrying the handle. The injected tools answer describe/digest/aggregate/rows/search/lines/retrieve
  against the held bytes — search is row-aware on JSON arrays (a single-line array would otherwise make a
  line search a trap: it matches "line 1" and shows the head of the file) and its matches feed straight
  into `rows` for verbatim fetches — so a result that would fail a host's output cap becomes a queryable
  handle instead.
  `structuredContent` is never rewritten (it would violate the tool's declared `outputSchema`), and
  `isError` results are never offloaded (a large error must stay visible).

## 7. Token accounting

- Token savings are counted with the **target model's own tokenizer**, reconciled against the provider's
  reported usage — not an offline proxy count.
- **Retrieval round-trip tokens are counted**: if the model later fetches an offloaded chunk back, that
  cost is part of the measured net, not assumed to be zero.
- Byte-faithful round-trip fidelity is reported separately from any end-task accuracy result.

## 8. Safety & limits

- **Safe by default.** The shell tool `coffer_run` is disabled unless `COFFER_MCP_ENABLE_RUN=1` (with an
  optional command-prefix allowlist that rejects shell control syntax), and is bounded by a timeout and an
  output cap. The proxy refuses a non-loopback bind unless `COFFER_PROXY_ALLOW_PUBLIC=1`, because it has no
  authentication and replays the client's upstream key.
- **Backpressure.** The proxy admits requests under a concurrency budget and **sheds** with `503` +
  `Retry-After` when saturated, rather than queueing unbounded work; inbound bodies are size-capped.
- **Bounded retrieval.** Byte-window retrieval is capped by default; whole-payload retrieval is explicit
  and separately capped.

## 9. Non-goals

- coffer is a **lossless, byte-exact reversible** engine. Maximizing a lossy compression ratio is not the
  objective; nothing is summarized away irrecoverably.
- Whether compressing tool-output **preserves or improves end-task accuracy** is an experimental question,
  measured separately at multiple compression levels against cheap baselines. A compression percentage is
  never presented as an accuracy result.
