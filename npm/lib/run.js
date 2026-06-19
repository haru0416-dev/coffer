// Resolve and launch a coffer binary. Shared by the `coffer-mcp` and `coffer-proxy` bin launchers.
//
// Resolution order:
//   1. an explicit COFFER_<TOOL>_BIN env override,
//   2. the installed platform package @coffer/<os>-<arch> (the optionalDependency npm picked for this host),
//   3. a local target/release/<tool> binary (source-checkout dev fallback).
// Then spawn it with stdio inherited and forward termination signals, so an MCP stdio server is transparent.

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

function platformKey() {
  return `${process.platform}-${process.arch}`; // e.g. linux-x64, darwin-arm64, win32-x64
}

function binFile(tool) {
  return process.platform === 'win32' ? `${tool}.exe` : tool;
}

function envOverride(tool) {
  return `COFFER_${tool.toUpperCase().replace(/-/g, '_')}_BIN`; // coffer-mcp -> COFFER_COFFER_MCP_BIN
}

function resolveBinary(tool) {
  const override = process.env[envOverride(tool)];
  if (override && fs.existsSync(override)) return override;

  // The platform package ships its binary under bin/. require.resolve finds it via node_modules.
  try {
    const pkgJson = require.resolve(`@coffer/${platformKey()}/package.json`);
    const candidate = path.join(path.dirname(pkgJson), 'bin', binFile(tool));
    if (fs.existsSync(candidate)) return candidate;
  } catch (_) {
    // platform package not installed (e.g. unsupported host, or running from a source checkout)
  }

  // Dev fallback: npm/lib/run.js -> repo root is two levels up.
  const dev = path.join(__dirname, '..', '..', 'target', 'release', binFile(tool));
  if (fs.existsSync(dev)) return dev;

  return null;
}

function run(tool) {
  const bin = resolveBinary(tool);
  if (!bin) {
    process.stderr.write(
      `coffer: no prebuilt ${tool} binary for ${platformKey()}.\n` +
        `  Install the platform package, set ${envOverride(tool)}=/path/to/${tool}, ` +
        `or build from source: cargo build --release -p ${tool}\n`,
    );
    process.exit(1);
  }

  const child = spawn(bin, process.argv.slice(2), { stdio: 'inherit' });
  for (const sig of ['SIGINT', 'SIGTERM', 'SIGHUP']) {
    process.on(sig, () => {
      try {
        child.kill(sig);
      } catch (_) {
        /* child already gone */
      }
    });
  }
  child.on('error', (err) => {
    process.stderr.write(`coffer: failed to launch ${bin}: ${err.message}\n`);
    process.exit(1);
  });
  child.on('exit', (code, signal) => {
    if (signal) process.kill(process.pid, signal);
    else process.exit(code === null ? 1 : code);
  });
}

module.exports = { run, resolveBinary, platformKey };
