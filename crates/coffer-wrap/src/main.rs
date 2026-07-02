//! coffer-wrap binary: `coffer-wrap [--] <downstream-mcp-command> [args...]`
//!
//! Spawns the downstream stdio MCP server as a child process and relays JSON-RPC
//! between it and our own stdin/stdout, offloading oversized tool results into a CAS
//! (in-memory by default; `SQLite` via `COFFER_CAS_DB`, shared with coffer-proxy /
//! coffer-mcp). stdout carries JSON-RPC only; all logging goes to stderr.

#![warn(clippy::pedantic)]
// Cast lints: env-derived sizes bounded by config validation.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use std::process::{ExitCode, Stdio};

use coffer_wrap::{HandleStore, WrapConfig, run_relay};
use tokio::process::Command;

fn init_tracing() {
    use tracing_subscriber::EnvFilter;
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();
}

fn build_store() -> Result<HandleStore, String> {
    match std::env::var("COFFER_CAS_DB") {
        Ok(path) if !path.trim().is_empty() => {
            let path = path.trim().to_string();
            if let Some(parent) = std::path::Path::new(&path).parent()
                && !parent.as_os_str().is_empty()
            {
                coffer_cas::create_private_dir_all(parent)
                    .map_err(|e| format!("create {}: {e}", parent.display()))?;
            }
            // Honor the same COFFER_CAS_* tuning as the proxy and the MCP server.
            let cas = coffer_cas::SqliteCas::open_with_config(
                &path,
                &coffer_cas::SqliteConfig::from_env(),
            )
            .map_err(|e| format!("open SQLite CAS at {path}: {e}"))?;
            tracing::info!(%path, "using shared SQLite CAS");
            Ok(HandleStore::Sqlite(Box::new(cas)))
        }
        _ => Ok(HandleStore::Memory(coffer_cas::MemoryCas::new())),
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    init_tracing();

    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--") {
        args.remove(0);
    }
    if args.is_empty() {
        eprintln!("usage: coffer-wrap [--] <downstream-mcp-command> [args...]");
        eprintln!(
            "env: COFFER_WRAP_THRESHOLD_TOKENS (default 10000), COFFER_WRAP_PREFIX (default \
             coffer_), COFFER_CAS_DB (SQLite path; default in-memory), \
             COFFER_WRAP_PREVIEW_CHARS, COFFER_WRAP_RETRIEVE_MAX_BYTES"
        );
        return ExitCode::FAILURE;
    }

    let cfg = WrapConfig::from_env();
    let store = match build_store() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("coffer-wrap: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut child = match Command::new(&args[0])
        .args(&args[1..])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        // The child's stderr is its operator/log channel — pass it straight through.
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("coffer-wrap: failed to spawn `{}`: {e}", args.join(" "));
            return ExitCode::FAILURE;
        }
    };
    let down_in = child.stdin.take().expect("piped stdin");
    let down_out = child.stdout.take().expect("piped stdout");
    tracing::info!(downstream = %args.join(" "), "coffer-wrap relay started");

    if let Err(e) = run_relay(
        tokio::io::stdin(),
        tokio::io::stdout(),
        down_in,
        down_out,
        store,
        cfg,
    )
    .await
    {
        tracing::error!("relay I/O error: {e}");
    }

    match child.wait().await {
        Ok(status) if status.success() => ExitCode::SUCCESS,
        Ok(status) => {
            tracing::warn!(%status, "downstream exited non-zero");
            ExitCode::FAILURE
        }
        Err(e) => {
            tracing::error!("wait on downstream: {e}");
            ExitCode::FAILURE
        }
    }
}
