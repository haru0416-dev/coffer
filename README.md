# coffer

[![ci](https://github.com/haru0416-dev/coffer/actions/workflows/ci.yml/badge.svg)](https://github.com/haru0416-dev/coffer/actions/workflows/ci.yml)
[![license: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![status: experimental](https://img.shields.io/badge/status-experimental-orange.svg)

> When a tool result is too big for the model to read, don't truncate it — **hold it**.
> coffer keeps the exact bytes in a content store, hands the model a verified handle, and
> answers count / sum / max / group-by / join questions over **all** of the data, exactly.

![coffer-wrap turning a 20,000-pod tool result into a queryable handle: exact needle-in-the-elided-middle answer, provenance-backed count, sha256-verified retrieve](demo/wrap.gif)

**Status: experimental.** The engine, the three surfaces (MCP gateway, MCP server, transparent
proxy), and the production smoke gates are real and tested. Two properties are mechanical and
verifiable today — byte-exact recovery and exact aggregation. End-task *accuracy* claims (does
compressing tool-output preserve or improve what the model answers?) are governed by a protocol
fixed before the results, never by a compression percentage.

## The problem

One `kubectl get pods -o json`, one CI job log, one default-settings issue list from an API — a
single tool call routinely returns tens of thousands to hundreds of thousands of tokens. MCP hosts
cap tool output (commonly around 25k tokens): the call **fails outright**, gets silently
truncated, or floods the context and degrades every answer after it ("context rot"). And the
usual fixes throw information away — a head/tail window drops exactly the rows a buried answer
lives in, and a lossy summary can never give the bytes back.

## What coffer does instead

coffer moves the payload out of the context **without losing a byte**, and moves the *computation*
to where the bytes are:

- **Byte-exact reversibility.** Offloaded bytes live in a SHA-256 content-addressed store; the
  model sees a short handle / fact card. `reconstruct(compress(x)) == x` byte-for-byte — a Stage-0
  invariant, property-tested, and **re-verified against the hash on every read**, so a corrupt
  store is a hard error, never silently-wrong bytes. A dropped needle is recoverable, not lost.
- **An exact compute-digest.** Some questions have no answer in any surviving line — "how many
  errors?", "the pod with the most restarts", "the p95 latency". coffer computes those **exactly,
  in Rust, over ALL of the data including the offloaded bytes**, and returns the backing row
  indices (provenance) with every number. Refuse-rather-than-guess: a mixed-type field refuses the
  query instead of skipping values. A frontier model asked to count a few thousand rows sitting in
  its own context gets the number wrong; computing it over the held bytes does not.
- **A verifier.** `coffer_check_claim` recomputes a number an agent *claims* and answers
  AGREE/DISAGREE with the backing rows; `coffer_receipt` / `coffer_verify_receipt` issue and
  re-execute a portable exactness receipt (query + value + row indices + SHA-256 of the backing
  rows) with no model in the loop — a runnable demo is `cargo run -p coffer-core --example verify`.

## Surfaces — one content store, three doors

- **MCP gateway (`coffer-wrap`) — the one-line adoption path.** Wrap any existing stdio MCP
  server: `coffer-wrap -- <command>`. Oversized tool results are offloaded byte-exact and replaced
  with a fact card (handle + exact per-field stats + preview + how to query); the query tools below
  are injected alongside the wrapped server's own, collision-safe — a downstream tool is never
  shadowed. A result that would fail your host's output cap becomes a queryable handle instead.
  The gif above is this surface, end to end.
- **MCP server (`coffer-mcp`).** Tools to point at server-held output instead of reading it,
  grouped by what you're doing:
  - *understand exactly*: `coffer_describe` (row count + per-field stats/count-by for any record
    set), `coffer_digest` (plain-English exact stats), `coffer_aggregate` (typed
    `count|sum|mean|min|max` over predicates, with provenance indices), `coffer_bucket` /
    `coffer_window` (numeric-band / per-block-of-lines histograms), `coffer_join` (semi-join or
    grouped join across two held datasets — entirely server-side);
  - *narrow without reading*: `coffer_query` / `coffer_select` (filter rows, get a new handle
    back — composable), `coffer_pick` (fetch exactly the provenance rows to re-verify a number),
    `coffer_search` / `coffer_lines` / `coffer_rows` / `coffer_json` (bounded windows into logs
    and JSON);
  - *verify*: `coffer_check_claim`, `coffer_receipt`, `coffer_verify_receipt` (above);
  - *recover and hold*: `coffer_retrieve` / `coffer_unfold` (bounded byte windows, hash-verified),
    `coffer_ingest` (hold a file), `coffer_run` (capture a shell command server-side; disabled
    unless `COFFER_MCP_ENABLE_RUN=1`), `coffer_status`.
- **Transparent proxy (`coffer-proxy`).** Point an agent's base URL at it; it rewrites only the
  large tool-output values in flight — Anthropic Messages `tool_result`, OpenAI Responses
  `*_call_output`, and Ollama `/api/chat` tool messages — and never touches authored
  system/user/assistant text or the prompt-cache prefix. Every rewritten block leads with a
  one-line explainer so a `<<cof:…>>` marker is never mistaken for corruption. Fail-open: anything
  unexpected passes through unchanged.

All three share one content store, so the proxy can compress, the gateway can offload, and the MCP
server can recover and query the same bytes.

## Honesty, up front

Context-compression tools commonly measure compression % and accuracy on different datasets and
report only their best regime. coffer commits to the opposite, against a protocol fixed before the
results:

- Measure **end-task accuracy on the same workload**, at multiple compression levels → an
  accuracy-vs-compression curve, per regime and content type.
- The decisive test is **coffer vs naive head/tail truncation at the same token budget.** If a
  cheap baseline matches us on a workload, we say so.
- Report the **typical regime where we may lose**, not just the favorable tail.
- Count tokens with the **target model's own tokenizer**, and **count retrieval round-trip
  tokens.** Byte-faithful round-trip fidelity is reported separately from accuracy.

Two of those commitments are **mechanical, and reproducible with no API key.** `cargo run
--release -p coffer-eval` regenerates the table below over a 5,000-pod `kubectl` dump (~229k o200k
tokens), counting with the model's own tokenizer:

| compression | byte-exact round-trip | coffer answer error | head/tail truncation error at the same budget (count · sum · argmax) |
|------------:|:---------------------:|:-------------------:|:---------------------------------------------------------------------|
| 33% | ✅ | **0.00%** | 33% · 34% · ❌ buried needle missed |
| 67% | ✅ | **0.00%** | 68% · 61% · ❌ buried needle missed |
| 87% | ✅ | **0.00%** | 87% · 83% · ❌ buried needle missed |
| 93% | ✅ | **0.00%** | 93% · 91% · ❌ buried needle missed |
| 97% | ✅ | **0.00%** | 98% · 96% · ❌ buried needle missed |

coffer's answers are computed over **all** the bytes (including the offloaded ones) and asserted
against independently-computed ground truth, so its error is 0.00% at every level. The figures are
the *truncation* baseline's error at the **same token budget**, modeled generously as a perfect
aggregate over exactly the rows its window shows — a real model sees no more rows and is worse at
arithmetic, so this is an upper bound for it. An exact answer costs ~20 retrieval tokens
regardless of dump size. Truncation is **not** uniformly bad: on a sampling-robust statistic like
a mean it stays within a few percent — coffer's win is on the answers that depend on rows the
window drops (counts, sums, the buried extremum), and the harness reports the mean column where it
doesn't.

The engine itself is demoed the same way — a 5.7 MB kubectl-shaped dump cut from 2,170,329 to
216,857 real `o200k_base` tokens and rebuilt byte-for-byte (`cmp` on screen):

![coffer compressing a 5000-pod kubectl dump ~90%, still byte-for-byte reversible](demo/demo.gif)

The remaining commitment — **end-task accuracy with a real model**, at multiple compression levels
against this same truncation baseline — is the open experimental question the harness above does
**not** settle: it proves the two mechanical properties (byte-exact round-trip, exact
aggregation), not that an LLM answers better. If that accuracy thesis fails its kill-probe, the
failed curve is still a useful public result.

Where coffer does **not** win is just as clear. On plain retrieval that a frontier model's context
window already handles, compressing the input does not beat feeding it raw — coffer matches it, no
more. And a code-execution agent can compute the same exact aggregate by writing its own code; on
accuracy that is a tie, not a coffer win. The difference worth stating plainly is narrower: coffer
runs at the transport layer before the bytes ever reach the model, needs no code sandbox or
codegen round-trip, and keeps every original byte recoverable.

## Quickstart

```sh
# MCP gateway — wrap the server that floods your context (any stdio MCP server, any MCP host):
cargo build --release -p coffer-wrap
#   in your host's MCP config:  "command": ".../coffer-wrap", "args": ["<your-server-cmd>", "<args>…"]

# MCP server — hold outputs by handle, query them exactly:
cargo run --release -p coffer-mcp

# Transparent proxy — compress tool_result blocks in flight:
cargo run --release -p coffer-proxy
ANTHROPIC_BASE_URL=http://127.0.0.1:8788   # COFFER_PROXY_UPSTREAM defaults to api.anthropic.com
```

Set `COFFER_CAS_DB=/path/to/cas.db` on any of them to share one persistent store across
processes — offload in the gateway or proxy, recover and query in the MCP server. See
[`docs/deployment.md`](docs/deployment.md) for production wiring.

An npm launcher is scaffolded under [`npm/`](npm/): once published it will run the prebuilt native
binary for your platform (`npx coffer coffer-mcp`). It is **not on the npm registry yet** — for
now, build from source as above.

Safe by default: the proxy refuses a non-loopback bind unless `COFFER_PROXY_ALLOW_PUBLIC=1` (it
has no auth and replays your upstream key); the MCP `coffer_run` shell tool is disabled unless
`COFFER_MCP_ENABLE_RUN=1`.

## Layout

- `crates/` — the engine (`coffer-core`), content store (`coffer-cas`), tokenizer-parity counting
  (`coffer-tokenizer`), MCP server (`coffer-mcp`), MCP gateway (`coffer-wrap`), transparent proxy
  (`coffer-proxy`), and the reproducible benchmark (`coffer-eval`, the table above).
- [`docs/DESIGN.md`](docs/DESIGN.md) — design & specification: the reversibility invariant, data
  model, compression pipeline, budget search, the compute-digest, surfaces, and non-goals.
- [`docs/deployment.md`](docs/deployment.md) — MCP/proxy deployment, shared-CAS wiring, and limits.
- [`demo/`](demo/README.md) — both gifs above, recorded reproducibly from the real commands.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
