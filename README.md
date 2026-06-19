# coffer

> Byte-exact, **reversible** compression of LLM tool-output — plus an exact **compute-digest** that
> answers questions over data too big to read back into the context window.

**Status: experimental.** The engine, the CAS-backed MCP server and transparent proxy, and the
production smoke gates are real and tested. Two properties are mechanical and verifiable today —
byte-exact recovery and exact aggregation. End-task *accuracy* claims (does compressing tool-output
preserve or improve what the model answers?) are governed by a pre-registered protocol, never by a
compression percentage.

## The idea

When an agent dumps a 100-result code search, a 65k-line log, or a noisy RAG payload into the model's
context, most of it is noise that both costs tokens and degrades the model's answers ("context rot").
coffer compresses that tool-output **before it enters the context window**, keeps the **original bytes**
in a SHA-256 content-addressed store, and shows the model a short `<<cof:HASH>>` sentinel. The model
recovers the exact byte window (or a capped full payload) on demand, and the parts of the prompt the
provider is caching are never touched.

Two things make this more than a trimmer:

- **Byte-exact reversibility.** Nothing is summarized away. `reconstruct(compress(x)) == x` byte-for-byte
  (a Stage-0 invariant, property-tested), backed by a durable, hash-verified content store. A dropped
  needle is recoverable, not lost.
- **A compute-digest.** Some questions have no answer in any single surviving line — "how many errors?",
  "the record with the largest value", "the p95 latency". coffer computes those **exactly, in Rust, over
  ALL of the data including the offloaded bytes** (count / sum / mean / median / percentile / group-by /
  argmax / threshold-count), with a refuse-rather-than-guess contract. No model call, no loss of
  reversibility — the answer is exact even when the raw data was far too big to read. A frontier model
  asked to count or sum a few thousand rows sitting in its own context gets the number wrong; computing
  it in Rust over the held bytes does not.

## Surfaces

- **Transparent proxy.** Point an agent at it (`ANTHROPIC_BASE_URL=http://127.0.0.1:8788`); it compresses
  the `tool_result` blocks of each request and forwards the rest unchanged, streaming the response back.
  OpenAI Responses tool-output is handled the same way.
- **MCP server.** Tools to point at server-held output instead of reading it: `coffer_describe` (a
  generic exact summary of any record set — row count, per-field stats, count-by-value), `coffer_digest`
  and `coffer_aggregate` (exact aggregates from a plain-English ask or a typed `count|sum|mean|min|max`
  over a predicate, returned with the indices of the rows behind the number), `coffer_query` and
  `coffer_select` (keep the matching rows, or hand back a new handle to narrow again), `coffer_pick`
  (pull specific rows back to re-check a number), `coffer_search`/`coffer_lines` (drill into logs),
  `coffer_rows`/`coffer_json`/`coffer_retrieve`/`coffer_unfold` (windowed JSON and bounded byte windows),
  and `coffer_ingest` (hold a file).

Both share one content store, so the proxy can compress and the MCP server can recover the same bytes.

## Honesty, up front

Context-compression tools commonly measure compression % and accuracy on different datasets and report
only their best regime. coffer commits to the opposite, against a protocol fixed before the results:

- Measure **end-task accuracy on the same workload**, at multiple compression levels → an
  accuracy-vs-compression curve, per regime and content type.
- The decisive test is **coffer vs naive head/tail truncation at the same token budget.** If a cheap
  baseline matches us on a workload, we say so.
- Report the **typical regime where we may lose**, not just the favorable tail.
- Count tokens with the **target model's own tokenizer**, and **count retrieval round-trip tokens.**
  Byte-faithful round-trip fidelity is reported separately from accuracy.

Where coffer does **not** win is just as clear. On plain retrieval that a frontier model's context window
already handles, compressing the input does not beat feeding it raw — coffer matches it, no more. And a
code-execution agent can compute the same exact aggregate by writing its own code; on accuracy that is a
tie, not a coffer win. The difference worth stating plainly is narrower: coffer runs at the transport
layer before the bytes ever reach the model, needs no code sandbox or codegen round-trip, and keeps every
original byte recoverable.

If the accuracy thesis fails its kill-probe, the failed curve is still a useful public result.

## Quickstart

```sh
# Proxy: compress tool_result blocks transparently.
cargo run --release -p coffer-proxy
# then point your agent at it:
ANTHROPIC_BASE_URL=http://127.0.0.1:8788  # COFFER_PROXY_UPSTREAM defaults to api.anthropic.com

# MCP server (stdio): register coffer-mcp with your agent, then direct tools at held output.
cargo run --release -p coffer-mcp
```

For agents that install MCP servers over npm, an npm launcher is provided — `npx coffer coffer-mcp`
(or `npm i -g coffer`), which runs the prebuilt native binary for your platform. See [`npm/`](npm/).

Safe by default: the proxy refuses a non-loopback bind unless `COFFER_PROXY_ALLOW_PUBLIC=1` (it has no
auth and replays your upstream key); the MCP `coffer_run` shell tool is disabled unless
`COFFER_MCP_ENABLE_RUN=1`. See [`docs/deployment.md`](docs/deployment.md) for production wiring.

## Layout

- `crates/` — the engine (`coffer-core`), content store (`coffer-cas`), tokenizer-parity counting
  (`coffer-tokenizer`), MCP server (`coffer-mcp`), and transparent proxy (`coffer-proxy`).
- [`docs/DESIGN.md`](docs/DESIGN.md) — design & specification: the reversibility invariant, data model,
  compression pipeline, budget search, the compute-digest, surfaces, and non-goals.
- [`docs/deployment.md`](docs/deployment.md) — MCP/proxy deployment, shared-CAS wiring, and limits.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
