# Deployment

This page covers the public source snapshot: the reusable engine, the MCP
server, and the transparent proxy. Internal release-packaging, eval, and
operator handoff scripts are intentionally not part of this snapshot.

## Build

```sh
cargo build --release -p coffer-mcp -p coffer-proxy
```

## Run the Proxy

The proxy listens on loopback by default and forwards Anthropic-compatible
`/v1/messages` requests to the upstream API. It has no authentication layer of
its own, so keep it local unless you are adding your own network boundary.

```sh
mkdir -p "$HOME/.coffer"
chmod 700 "$HOME/.coffer"

export COFFER_CAS_DB="$HOME/.coffer/session.db"
export COFFER_PROXY_LISTEN="127.0.0.1:8788"
export COFFER_PROXY_UPSTREAM="https://api.anthropic.com"

./target/release/coffer-proxy
```

Then point a compatible client at it:

```sh
export ANTHROPIC_BASE_URL="http://127.0.0.1:8788"
```

Tuning: `COFFER_PROXY_REDUCTION` (fraction of each block's tokens to cut, default `0.8`) and
`COFFER_PROXY_MAX_KEPT_TOKENS` (absolute per-block ceiling on kept tokens, heuristic count,
default `4000`, `0` disables) — the effective keep is `min(raw × (1 − reduction), ceiling)`.
`COFFER_PROXY_EXPLAIN=0` drops the in-band sentinel explainer. Both budget knobs are echoed
back by `/_coffer/health` and `/_coffer/metrics` for verification.

The proxy refuses a non-loopback bind unless `COFFER_PROXY_ALLOW_PUBLIC=1` is
set. If you use that override, put the proxy behind your own authentication and
transport security.

## Run the MCP Server

Register the MCP server with the same `COFFER_CAS_DB` when you want MCP tools to
retrieve bytes that the proxy externalized:

```jsonc
{
  "mcpServers": {
    "coffer": {
      "command": "/absolute/path/to/target/release/coffer-mcp",
      "env": {
        "COFFER_CAS_DB": "/home/you/.coffer/session.db"
      }
    }
  }
}
```

Without `COFFER_CAS_DB`, MCP handles are in-process and disappear when the
server exits. With `COFFER_CAS_DB`, stored bytes are keyed by SHA-256 and can be
retrieved after restart.

### Headless agents: allow the MCP tool calls

A headless agent only *uses* coffer if its MCP client is allowed to call the
coffer tools without an interactive approval prompt. Verified end-to-end: with
the server registered, an agent recognizes and calls `coffer_ingest` /
`coffer_aggregate` and trusts the exact result over reading the held bytes. But
under a default "never approve" policy the native MCP calls are cancelled — in
which case the agent fails **safe** (it answers "unavailable" rather than reading
the raw bytes or guessing), it does not silently fall back. So enable approvals
for the coffer tools in the client: Claude Code allows them via the `mcpServers`
entry plus its permission settings; codex runs them in an externally-sandboxed
environment with `--dangerously-bypass-approvals-and-sandbox`.

## Safety Defaults

- `coffer_run` is disabled unless `COFFER_MCP_ENABLE_RUN=1`.
- If `COFFER_MCP_RUN_ALLOWLIST` is set, `coffer_run` only accepts matching
  command prefixes and rejects shell control syntax such as `;`, `&&`, pipes,
  redirects, and command substitution.
- SQLite CAS files may contain original tool-output bytes. Keep the CAS
  directory private (`0700`); coffer creates SQLite database files with private
  file permissions on Unix.

## Useful Limits

- `COFFER_MCP_DEFAULT_RETRIEVE_BYTES`: default byte window for retrieve/unfold.
  Default: `65536`.
- `COFFER_MCP_MAX_RETRIEVE_BYTES`: hard cap for one raw byte-returning call.
  Default: `1048576`.
- `COFFER_MCP_RUN_TIMEOUT_SECONDS`: maximum runtime for one enabled
  `coffer_run` call. Default: `300`.
- `COFFER_MCP_MAX_RUN_OUTPUT_MB`: maximum captured output for one enabled
  `coffer_run` call. Default: `64`.
- `COFFER_CAS_RESIDENT_CAP_MB`: cap historical read-cache bytes in RAM. It does
  not delete SQLite data and does not cap bytes newly written during the current
  process run.

## Claim Boundary

The public snapshot demonstrates byte-exact externalization, bounded retrieval,
and exact digest/query operations over held data. It does not by itself prove
provider billing savings, model accuracy improvements, or that a future model
session will choose the right drilldown call.
