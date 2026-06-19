# coffer

Byte-exact, **reversible** compression of LLM tool-output, plus an exact **compute-digest** that answers
questions over data too big to read back into the context window. Ships an MCP server (`coffer-mcp`) and a
transparent HTTP proxy (`coffer-proxy`).

This npm package is a launcher: it runs a prebuilt native binary for your platform (installed as an
optional dependency — `@coffer/linux-x64`, `@coffer/darwin-arm64`, `@coffer/win32-x64`, …). No build step,
no install scripts.

## Use as an MCP server

Point your agent at it with `npx` (no global install needed):

```jsonc
{
  "mcpServers": {
    "coffer": {
      "command": "npx",
      "args": ["-y", "coffer", "coffer-mcp"],
      "env": { "COFFER_CAS_DB": "/home/you/.coffer/session.db" }
    }
  }
}
```

Or install the commands globally:

```sh
npm install -g coffer
coffer-mcp     # stdio MCP server
coffer-proxy   # transparent HTTP proxy (loopback by default)
```

Each binary is the same one documented in the main repository; the launcher just execs it and passes
stdio/arguments/signals straight through.

## Binary resolution

The launcher finds the binary in this order:

1. `COFFER_COFFER_MCP_BIN` / `COFFER_COFFER_PROXY_BIN` — an explicit path you set.
2. The installed `@coffer/<os>-<arch>` platform package.
3. `target/release/<tool>` relative to a source checkout (so `node bin/coffer-mcp.js` works after
   `cargo build --release` while developing).

If none resolve, it prints how to install the platform package, set the env override, or build from source.

## Safety defaults

The proxy refuses a non-loopback bind unless `COFFER_PROXY_ALLOW_PUBLIC=1`; the MCP `coffer_run` shell tool
is disabled unless `COFFER_MCP_ENABLE_RUN=1`. Headless agents must allow the coffer MCP tools in their
client (otherwise the calls are cancelled and the agent falls back safely without using coffer).

## License

Apache-2.0.
