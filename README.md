# coffer

[![ci](https://github.com/haru0416-dev/coffer/actions/workflows/ci.yml/badge.svg)](https://github.com/haru0416-dev/coffer/actions/workflows/ci.yml)
[![license: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![status: experimental](https://img.shields.io/badge/status-experimental-orange.svg)

> Byte-exact, **reversible** compression of LLM tool-output — plus an exact **compute-digest** that
> answers questions over data too big to read back into the context window.

![coffer compressing a 5000-pod kubectl dump ~90%, still byte-for-byte reversible](demo/demo.gif)

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
  (pull specific rows back to re-check a number), `coffer_bucket`/`coffer_window` (an exact numeric-band
  or per-block-of-lines histogram), `coffer_join` (correlate two held datasets — "sum order.amount for
  orders whose customer is gold-tier" — entirely server-side), `coffer_check_claim` (recompute a number an
  agent claims against the held bytes — AGREE/DISAGREE, with the backing rows), `coffer_receipt`/
  `coffer_verify_receipt` (issue a portable, re-executable exactness receipt for an aggregate and verify
  one later with no model in the loop — a runnable demo is `cargo run -p coffer-core --example verify`),
  `coffer_search`/`coffer_lines` (drill into
  logs), `coffer_rows`/`coffer_json`/`coffer_retrieve`/`coffer_unfold` (windowed JSON and bounded byte
  windows), and `coffer_ingest` (hold a file).
- **MCP gateway.** Wrap any existing stdio MCP server (`coffer-wrap -- <command>`): oversized tool
  results are offloaded byte-exact into the store and replaced with a fact card + handle, and the query
  tools above are injected alongside the wrapped server's own (collision-safe — a downstream tool is
  never shadowed). A result that would fail a host's output cap becomes a queryable handle instead
  ([30-second demo](demo/wrap.gif)).

All three share one content store, so the proxy can compress, the gateway can offload, and the MCP
server can recover the same bytes.

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

Two of those commitments are **mechanical, and reproducible with no API key.** `cargo run --release -p
coffer-eval` regenerates the table below over a 5,000-pod `kubectl` dump (~229k o200k tokens), counting
with the model's own tokenizer:

| compression | byte-exact round-trip | coffer answer error | head/tail truncation error at the same budget (count · sum · argmax) |
|------------:|:---------------------:|:-------------------:|:---------------------------------------------------------------------|
| 33% | ✅ | **0.00%** | 33% · 34% · ❌ buried needle missed |
| 67% | ✅ | **0.00%** | 68% · 61% · ❌ buried needle missed |
| 87% | ✅ | **0.00%** | 87% · 83% · ❌ buried needle missed |
| 93% | ✅ | **0.00%** | 93% · 91% · ❌ buried needle missed |
| 97% | ✅ | **0.00%** | 98% · 96% · ❌ buried needle missed |

coffer's answers are computed over **all** the bytes (including the offloaded ones) and asserted against
independently-computed ground truth, so its error is 0.00% at every level. The figures are the
*truncation* baseline's error at the **same token budget**, modeled generously as a perfect aggregate
over exactly the rows its window shows — a real model sees no more rows and is worse at arithmetic, so
this is an upper bound for it. An exact answer costs ~20 retrieval tokens regardless of dump size.
Truncation is **not** uniformly bad: on a sampling-robust statistic like a mean it stays within a few
percent — coffer's win is on the answers that depend on rows the window drops (counts, sums, the buried
extremum), and the harness reports the mean column where it doesn't.

The fourth commitment — **end-task accuracy with a real model**, at multiple compression levels against
this same truncation baseline — is the open experimental question the harness above does **not** settle:
it proves the two mechanical properties (byte-exact round-trip, exact aggregation), not that an LLM
answers better. If that accuracy thesis fails its kill-probe, the failed curve is still a useful public
result.

Where coffer does **not** win is just as clear. On plain retrieval that a frontier model's context window
already handles, compressing the input does not beat feeding it raw — coffer matches it, no more. And a
code-execution agent can compute the same exact aggregate by writing its own code; on accuracy that is a
tie, not a coffer win. The difference worth stating plainly is narrower: coffer runs at the transport
layer before the bytes ever reach the model, needs no code sandbox or codegen round-trip, and keeps every
original byte recoverable.

## Quickstart

```sh
# Proxy: compress tool_result blocks transparently.
cargo run --release -p coffer-proxy
# then point your agent at it:
ANTHROPIC_BASE_URL=http://127.0.0.1:8788  # COFFER_PROXY_UPSTREAM defaults to api.anthropic.com

# MCP server (stdio): register coffer-mcp with your agent, then direct tools at held output.
cargo run --release -p coffer-mcp
```

An npm launcher is scaffolded under [`npm/`](npm/): once published it will run the prebuilt native
binary for your platform (`npx coffer coffer-mcp`). It is **not on the npm registry yet** — for now,
build from source as above.

Safe by default: the proxy refuses a non-loopback bind unless `COFFER_PROXY_ALLOW_PUBLIC=1` (it has no
auth and replays your upstream key); the MCP `coffer_run` shell tool is disabled unless
`COFFER_MCP_ENABLE_RUN=1`. See [`docs/deployment.md`](docs/deployment.md) for production wiring.

## Layout

- `crates/` — the engine (`coffer-core`), content store (`coffer-cas`), tokenizer-parity counting
  (`coffer-tokenizer`), MCP server (`coffer-mcp`), transparent proxy (`coffer-proxy`), MCP gateway
  (`coffer-wrap`), and the reproducible benchmark (`coffer-eval`, the table above — `cargo run
  --release -p coffer-eval`).
- [`docs/DESIGN.md`](docs/DESIGN.md) — design & specification: the reversibility invariant, data model,
  compression pipeline, budget search, the compute-digest, surfaces, and non-goals.
- [`docs/deployment.md`](docs/deployment.md) — MCP/proxy deployment, shared-CAS wiring, and limits.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
