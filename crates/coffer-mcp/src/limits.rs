//! Request/response size limits and their `COFFER_MCP_*` environment parsing.
// Split out of main.rs for maintainability; behavior is unchanged. main.rs glob-imports
// this module, so the original single-file scope (and its test suite) still sees every name.

pub(crate) const MIB: usize = 1024 * 1024;
pub(crate) const DEFAULT_RETRIEVE_BYTES: usize = 64 * 1024;
pub(crate) const DEFAULT_MAX_RETRIEVE_BYTES: usize = 1024 * 1024;
pub(crate) const DEFAULT_STRUCTURAL_CODE_TARGET_TOKENS: usize = 1024;
pub(crate) const DEFAULT_RUN_TIMEOUT_SECONDS: u64 = 300;
pub(crate) const DEFAULT_MAX_RUN_OUTPUT_BYTES: usize = 64 * MIB;
/// Hard ceiling on rows pretty-printed by `coffer_rows`, overridable via `COFFER_MCP_MAX_ROWS`.
/// Bounds the model-facing response so a large `limit` cannot re-bloat the context that the
/// server-side hold exists to keep out of it (the original stays reachable via `coffer_retrieve`).
pub(crate) const DEFAULT_MAX_ROWS: usize = 200;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RetrieveLimits {
    pub(crate) default_bytes: usize,
    pub(crate) max_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RunLimits {
    pub(crate) timeout_seconds: u64,
    pub(crate) max_output_bytes: usize,
}

pub(crate) fn retrieve_limits_from_env() -> RetrieveLimits {
    let default_raw = std::env::var("COFFER_MCP_DEFAULT_RETRIEVE_BYTES").ok();
    let max_raw = std::env::var("COFFER_MCP_MAX_RETRIEVE_BYTES").ok();
    retrieve_limits_from_values(default_raw.as_deref(), max_raw.as_deref())
}

pub(crate) fn max_rows_from_env() -> usize {
    positive_usize_from_value(std::env::var("COFFER_MCP_MAX_ROWS").ok().as_deref())
        .unwrap_or(DEFAULT_MAX_ROWS)
}

pub(crate) fn retrieve_limits_from_values(
    default_raw: Option<&str>,
    max_raw: Option<&str>,
) -> RetrieveLimits {
    let max_bytes = positive_usize_from_value(max_raw).unwrap_or(DEFAULT_MAX_RETRIEVE_BYTES);
    let default_bytes = positive_usize_from_value(default_raw)
        .unwrap_or(DEFAULT_RETRIEVE_BYTES)
        .min(max_bytes);
    RetrieveLimits {
        default_bytes,
        max_bytes,
    }
}

pub(crate) fn run_limits_from_env() -> RunLimits {
    let timeout_raw = std::env::var("COFFER_MCP_RUN_TIMEOUT_SECONDS").ok();
    let output_raw = std::env::var("COFFER_MCP_MAX_RUN_OUTPUT_MB").ok();
    run_limits_from_values(timeout_raw.as_deref(), output_raw.as_deref())
}

pub(crate) fn run_limits_from_values(
    timeout_raw: Option<&str>,
    output_mb_raw: Option<&str>,
) -> RunLimits {
    let timeout_seconds =
        positive_u64_from_value(timeout_raw).unwrap_or(DEFAULT_RUN_TIMEOUT_SECONDS);
    let max_output_bytes = positive_usize_from_value(output_mb_raw)
        .map_or(DEFAULT_MAX_RUN_OUTPUT_BYTES, |mb| mb.saturating_mul(MIB));
    RunLimits {
        timeout_seconds,
        max_output_bytes,
    }
}

pub(crate) fn positive_u64_from_value(raw: Option<&str>) -> Option<u64> {
    let value = raw?.trim().parse::<u64>().ok()?;
    (value > 0).then_some(value)
}

pub(crate) fn positive_usize_from_value(raw: Option<&str>) -> Option<usize> {
    let value = raw?.trim().parse::<usize>().ok()?;
    (value > 0).then_some(value)
}
