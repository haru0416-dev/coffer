//! The gated `coffer_run` shell capture: policy (default-off + allowlist), limits, and
//! the bounded child-process runner.
// Split out of main.rs for maintainability; behavior is unchanged. main.rs glob-imports
// this module, so the original single-file scope (and its test suite) still sees every name.

use std::process::{ExitStatus, Stdio};
use std::time::Duration;

use tokio::io::AsyncReadExt;

use crate::limits::RunLimits;

#[derive(Debug)]
pub(crate) struct RunCapture {
    pub(crate) bytes: Vec<u8>,
    pub(crate) status: Option<ExitStatus>,
    pub(crate) timed_out: bool,
    pub(crate) output_truncated: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RunReadOutcome {
    Exited(ExitStatus),
    OutputLimit,
}

/// Whether `coffer_run` may execute, and any command allowlist. Off by default: arbitrary
/// `sh -c` execution is opt-in via `COFFER_MCP_ENABLE_RUN`, optionally narrowed by a comma-separated
/// `COFFER_MCP_RUN_ALLOWLIST` of command prefixes. When an allowlist is configured, shell control
/// syntax is refused before spawning.
pub(crate) struct RunPolicy {
    pub(crate) enabled: bool,
    pub(crate) allowlist: Vec<String>,
}

impl RunPolicy {
    /// `Ok(())` if `command` may run under this policy, else an explanatory message the caller returns
    /// to the client. Never spawns anything itself.
    pub(crate) fn permits(&self, command: &str) -> Result<(), String> {
        if !self.enabled {
            return Err(
                "coffer_run is disabled by default; set COFFER_MCP_ENABLE_RUN=1 to enable it \
                 (optionally COFFER_MCP_RUN_ALLOWLIST=\"prog1,prog2\" to restrict commands)"
                    .to_string(),
            );
        }
        let trimmed = command.trim_start();
        if !self.allowlist.is_empty() {
            if contains_shell_control(trimmed) {
                return Err(
                    "command refused: COFFER_MCP_RUN_ALLOWLIST accepts only simple command lines; shell control syntax is not allowed"
                        .to_string(),
                );
            }
            if !self
                .allowlist
                .iter()
                .any(|prefix| allowlist_prefix_matches(trimmed, prefix))
            {
                return Err(format!(
                    "command refused: COFFER_MCP_RUN_ALLOWLIST permits only commands beginning with one of [{}]",
                    self.allowlist.join(", ")
                ));
            }
        }
        Ok(())
    }
}

pub(crate) fn contains_shell_control(command: &str) -> bool {
    command.bytes().any(|b| {
        matches!(
            b,
            b';' | b'|'
                | b'&'
                | b'<'
                | b'>'
                | b'`'
                | b'$'
                | b'('
                | b')'
                | b'{'
                | b'}'
                | b'\n'
                | b'\r'
        )
    })
}

pub(crate) fn allowlist_prefix_matches(command: &str, prefix: &str) -> bool {
    command.starts_with(prefix)
        && command
            .as_bytes()
            .get(prefix.len())
            .is_none_or(u8::is_ascii_whitespace)
}

pub(crate) fn truthy(raw: Option<&str>) -> bool {
    raw.is_some_and(|v| {
        matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

pub(crate) fn run_policy_from_env() -> RunPolicy {
    run_policy_from_values(
        std::env::var("COFFER_MCP_ENABLE_RUN").ok().as_deref(),
        std::env::var("COFFER_MCP_RUN_ALLOWLIST").ok().as_deref(),
    )
}

pub(crate) fn run_policy_from_values(
    enable_raw: Option<&str>,
    allowlist_raw: Option<&str>,
) -> RunPolicy {
    let allowlist = allowlist_raw
        .map(|raw| {
            raw.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();
    RunPolicy {
        enabled: truthy(enable_raw),
        allowlist,
    }
}

pub(crate) async fn run_shell_command(
    command: &str,
    limits: RunLimits,
) -> std::io::Result<RunCapture> {
    let shell_command = format!("exec 2>&1; {command}");
    let mut child = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(shell_command)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()?;
    let mut stdout = child
        .stdout
        .take()
        .expect("stdout is piped before spawning the shell");
    let mut bytes = Vec::new();
    let mut buf = [0_u8; 8192];
    let timeout = Duration::from_secs(limits.timeout_seconds);
    let read = async {
        loop {
            let read = stdout.read(&mut buf).await?;
            if read == 0 {
                break;
            }
            let remaining = limits.max_output_bytes.saturating_sub(bytes.len());
            if read > remaining {
                bytes.extend_from_slice(&buf[..remaining]);
                return Ok(RunReadOutcome::OutputLimit);
            }
            bytes.extend_from_slice(&buf[..read]);
        }
        child.wait().await.map(RunReadOutcome::Exited)
    };

    match tokio::time::timeout(timeout, read).await {
        Ok(Ok(RunReadOutcome::Exited(status))) => Ok(RunCapture {
            bytes,
            status: Some(status),
            timed_out: false,
            output_truncated: false,
        }),
        Ok(Ok(RunReadOutcome::OutputLimit)) => {
            let _ = child.start_kill();
            let _ = child.wait().await;
            Ok(RunCapture {
                bytes,
                status: None,
                timed_out: false,
                output_truncated: true,
            })
        }
        Ok(Err(error)) => Err(error),
        Err(_) => {
            let _ = child.start_kill();
            let _ = child.wait().await;
            Ok(RunCapture {
                bytes,
                status: None,
                timed_out: true,
                output_truncated: false,
            })
        }
    }
}
