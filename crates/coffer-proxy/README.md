# coffer-proxy

A transparent-compression proxy. Point an MCP/Anthropic agent at it and it compresses the
large `tool_result` blocks of every `POST /v1/messages` request with the coffer engine before
forwarding to the real API — so huge tool output never costs the model context. It modifies only the
request; the response streams back unchanged.

This is the **cost-effective transparent shape**: unlike a model-invoked tool, there is no "model
chooses coffer vs `jq`" decision, so it just shrinks what would otherwise be paid for in full.

## Run

```sh
cargo build --release -p coffer-proxy -p coffer-mcp

# shared CAS so elided bytes and MCP handles are recoverable across processes
export COFFER_CAS_DB="$HOME/.coffer/session.db"
# optional: warn when the resident CAS cache exceeds this many MiB
export COFFER_CAS_SOFT_CAP_MB="512"
# optional: hard-cap historical read cache in RAM; does not delete SQLite data
# export COFFER_CAS_RESIDENT_CAP_MB="512"
# optional: set to 1/true/yes/on to eager-load historical CAS bytes at startup
# export COFFER_CAS_WARM_BYTES_ON_OPEN="1"
# optional: set to 1/true/yes/on to skip open-time byte scans on large trusted DBs
# export COFFER_CAS_TRUST_HASHES_ON_OPEN="1"
# optional: checkpoint SQLite WAL after this many blob commits
# export COFFER_CAS_CHECKPOINT_EVERY="1000"

# start the proxy (defaults: listen 127.0.0.1:8788, upstream https://api.anthropic.com)
export COFFER_PROXY_MAX_BODY_MB="64"
./target/release/coffer-proxy &

# local readiness check; does not contact the upstream provider
curl -fsS http://127.0.0.1:8788/_coffer/health

# point the agent at it
export ANTHROPIC_BASE_URL="http://127.0.0.1:8788"
```

Register `coffer-mcp` with the **same** `COFFER_CAS_DB` so the model can recover elided content:

```jsonc
// .mcp.json
{ "mcpServers": { "coffer": {
    "command": "cargo", "args": ["run","--quiet","-p","coffer-mcp"],
    "env": {
      "COFFER_CAS_DB": "/home/you/.coffer/session.db",
      "COFFER_CAS_SOFT_CAP_MB": "512",
      "COFFER_CAS_RESIDENT_CAP_MB": "512",
      "COFFER_CAS_WARM_BYTES_ON_OPEN": "0",
      "COFFER_CAS_TRUST_HASHES_ON_OPEN": "0",
      "COFFER_CAS_CHECKPOINT_EVERY": "1000",
      "COFFER_MCP_DEFAULT_RETRIEVE_BYTES": "65536",
      "COFFER_MCP_MAX_RETRIEVE_BYTES": "1048576",
      "COFFER_MCP_RUN_TIMEOUT_SECONDS": "300",
      "COFFER_MCP_MAX_RUN_OUTPUT_MB": "64"
    }
} } }
```

## The reversible loop

1. The proxy compresses a `tool_result`, keeping a head+tail window and replacing the dropped middle
   with a `<<cof:HASH +N items>>` sentinel; the elided bytes go into the shared CAS.
2. The model sees the compact render. When it needs the elided detail, it calls
   `coffer_unfold(HASH, start, max_bytes)` (the MCP tool) — a fresh cross-process read of the
   shared CAS returns the exact requested byte window. For output captured directly through MCP,
   the same `COFFER_CAS_DB` makes full-hash `run`/`ingest` handles survive MCP process restart; use
   the handle tools:
   `coffer_describe` (generic exact summary), `coffer_digest` / `coffer_aggregate` (exact aggregates,
   with the backing row indices), `coffer_query` / `coffer_select` (keep or re-handle matching rows),
   `coffer_pick` (pull specific rows to re-verify), `coffer_search`, `coffer_lines`, `coffer_json`,
   `coffer_rows`, and `coffer_retrieve`. `full=true` returns an entire raw payload only when it is under the
   configured MCP hard cap. Use `coffer_status` to confirm backend, SQLite file sizes, checkpoint
   counters, durability counters, and retrieve caps.

So the proxy compresses ⊕ the MCP retrieves: byte-exact, lossless from the model's point of view.

## Configuration (env)

| var | default | meaning |
|---|---|---|
| `COFFER_PROXY_LISTEN` | `127.0.0.1:8788` | inbound address |
| `COFFER_PROXY_UPSTREAM` | `https://api.anthropic.com` | real API base (http for a mock) |
| `COFFER_PROXY_MIN` | `1024` | min `tool_result` bytes to compress |
| `COFFER_PROXY_MAX_BODY_MB` | `64` | max inbound request body size in MiB; larger requests return `413 Payload Too Large` |
| `COFFER_CAS_DB` | _(unset → in-memory)_ | shared SqliteCas path; required for proxy unfold and persistent MCP handles |
| `COFFER_CAS_SOFT_CAP_MB` | _(unset)_ | optional resident-cache warning threshold, in MiB; this is not eviction |
| `COFFER_CAS_RESIDENT_CAP_MB` | _(unset)_ | optional RAM hard cap for historical read cache, in MiB; SQLite data is never deleted |
| `COFFER_CAS_WARM_BYTES_ON_OPEN` | _(unset → lazy)_ | set `1`/`true`/`yes`/`on` to eager-load historical CAS bytes on startup |
| `COFFER_CAS_TRUST_HASHES_ON_OPEN` | _(unset → validate)_ | set `1`/`true`/`yes`/`on` to skip open-time byte scans on large trusted DBs |
| `COFFER_CAS_CHECKPOINT_EVERY` | _(unset)_ | optional positive blob count for passive SQLite WAL checkpoints |
| `COFFER_MCP_DEFAULT_RETRIEVE_BYTES` | `65536` | default byte window for `coffer_retrieve` / `coffer_unfold` when `max_bytes` is omitted |
| `COFFER_MCP_MAX_RETRIEVE_BYTES` | `1048576` | hard cap for one raw byte-returning MCP call, including `full=true` |
| `COFFER_MCP_RUN_TIMEOUT_SECONDS` | `300` | max runtime for one `coffer_run` shell command |
| `COFFER_MCP_MAX_RUN_OUTPUT_MB` | `64` | max captured stdout/stderr for one `coffer_run` shell command |

## Guarantees & limits

- **Fail-open**: a body that is not the expected JSON is forwarded unchanged; the transform never
  panics. Only `tool_result` text is touched — never the system prompt, user, or assistant text.
- **Local health**: `GET /_coffer/health` returns payload-free proxy status/configuration and never
  contacts the upstream provider.
- **Inbound body cap**: request bodies over `COFFER_PROXY_MAX_BODY_MB` are rejected before parsing or
  compression. This bounds proxy RAM even when a client sends an unexpectedly large request.
- **Cache-friendly**: compression is deterministic, so a given tool_result always renders identically
  and the upstream prompt-cache prefix stays stable.
- **Bounded guarantee**: the proxy/MCP loop guarantees byte-exact recovery of elided bytes from CAS,
  not that the compact render alone preserves every semantic detail. Use targeted MCP drilldowns
  (`search`, `lines`, `json`, `rows`, `digest`, `retrieve`) when the omitted region could matter.
  The `coffer-proxy` test suite pins this loop by compressing a request, extracting the rendered
  sentinel hash, recovering the elided bytes through a fresh SQLite CAS read, and reconstructing the
  exact original `tool_result` from the compact render plus those bytes.
- **Bounded retrieval by default**: `retrieve` and `unfold` return a configured byte window unless
  the caller provides `start`/`max_bytes`. `full=true` is explicit and capped so a recovery action
  does not accidentally refill the model context with the whole payload.
- **Resident cache**: persisted blobs survive restart, but historical bytes are lazy-loaded into RAM
  by default. `coffer_status` reports known handles separately from resident bytes.
- **Resident cap**: `COFFER_CAS_RESIDENT_CAP_MB` evicts historical read-cache blobs from RAM only.
  Current-process puts stay resident until their durability path is settled.
- **Fast open**: `COFFER_CAS_TRUST_HASHES_ON_OPEN=1` reads only hash keys during startup.
  Bytes are still re-hashed before any retrieval returns them.
- **WAL checkpointing**: `COFFER_CAS_CHECKPOINT_EVERY=N` runs passive checkpoints after
  successful blob commits cross `N`. It helps with WAL growth; byte recovery still depends on
  CAS hashes and durability counters.
- **Disk footprint**: `coffer_status` reports SQLite main/WAL/SHM byte sizes for retention
  planning. The proxy does not delete CAS rows; rotate session databases outside the live handle
  lifetime.
- **Transport**: inbound is Hyper HTTP/1.x with keep-alive; upstream is Reqwest over HTTP or HTTPS.
  HTTP/2 inbound and provider-specific hardening remain follow-ups.
- **Provider**: Anthropic `/v1/messages` only. The codex/ChatGPT OAuth backend is hard to proxy, so
  there the MCP server stays the integration shape.

For production-oriented setup, smoke tests, and failure modes, see
[`docs/deployment.md`](../../docs/deployment.md). Copy-editable deployment
templates live under [`deploy/`](../../deploy/).

After `cargo build --release -p coffer-proxy`, run
[`scripts/proxy-smoke.sh`](../../scripts/proxy-smoke.sh) to exercise the release
binary against a local mock upstream and verify shared-CAS recovery.
