//! Cheap content-type detection. Stage 0 covers the high-signal cases; this is
//! deliberately a fast heuristic, not the final router. Misclassification never
//! threatens correctness — only how well a region compresses (see crate docs).

/// The content types Stage 0 distinguishes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContentType {
    /// A JSON array or object (the `SmartCrusher` target).
    Json,
    /// A repetitive, line-oriented log.
    Log,
    /// Anything else — passed through verbatim at Stage 0.
    Text,
}

/// Classify `input`. Cheap by design: at most one JSON parse plus a line scan.
///
/// Note: JSON nested deeper than `serde_json`'s default recursion limit (~128) fails to
/// parse and is classified `Text` (passed through, not offloaded). That is safe for the
/// reversibility invariant — only an effectiveness gap on pathologically deep documents.
#[must_use]
pub fn detect(input: &[u8]) -> ContentType {
    if starts_like_json_container(input)
        && serde_json::from_slice::<serde::de::IgnoredAny>(input).is_ok()
    {
        // A document whose first non-whitespace byte is '[' or '{' and that parses as JSON is
        // necessarily a container, so syntax validation alone classifies it — no `Value` is
        // materialized. `detect_with_value` keeps the materializing parse for callers that
        // need the parsed document itself.
        return ContentType::Json;
    }
    classify_non_json(input)
}

/// Like [`detect`], but also returns the parsed JSON value for a JSON container, so a caller
/// that needs both the classification and the value (e.g. the Stage-0 summarizer) parses the
/// bytes only once. The value is `None` for `Log`/`Text` — and for JSON scalars, which
/// classify as `Text`.
pub(crate) fn detect_with_value(input: &[u8]) -> (ContentType, Option<serde_json::Value>) {
    if starts_like_json_container(input) {
        if let Ok(value) = serde_json::from_slice::<serde_json::Value>(input) {
            if value.is_array() || value.is_object() {
                return (ContentType::Json, Some(value));
            }
        }
    }
    (classify_non_json(input), None)
}

/// First non-whitespace byte is `[` or `{` — the precondition under which "parses as JSON"
/// implies "is a JSON container" (a scalar cannot start with either byte).
fn starts_like_json_container(input: &[u8]) -> bool {
    let first_non_ws = input
        .iter()
        .copied()
        .find(|byte| !byte.is_ascii_whitespace());
    matches!(first_non_ws, Some(b'[' | b'{'))
}

/// The non-JSON half of the classifier: log heuristics, else text.
fn classify_non_json(input: &[u8]) -> ContentType {
    if let Ok(text) = std::str::from_utf8(input) {
        let lines: Vec<&str> = text.lines().collect();
        if looks_like_log(&lines) || looks_like_git_status(&lines) {
            return ContentType::Log;
        }
    }
    ContentType::Text
}

/// Heuristic for line-oriented, **samplable bulk** output — the classes a head+tail window
/// compresses safely (the middle is usually noise; errors/summaries sit at the ends). Either signal
/// suffices:
/// - (a) ≥8 lines, a majority beginning with a digit (timestamp/epoch) or `[` (a bracketed level
///   like `[INFO]`);
/// - (b) ≥16 lines with very low **first-token diversity** (≤10% distinct) — machine-generated bulk
///   whose lines repeat a small set of leading tokens (e.g. `Compiling`/`Running`/`Finished` build
///   output, or grep `path:line:` prefixes). Measured to separate build/log output (≈4–8% distinct)
///   from source code (≥20%), prose, and file listings (≥70%) on the tuning corpus.
///
/// This deliberately does NOT match prose or source code, which the model reads whole. A rare
/// false-positive (e.g. a file of repeated `import` lines) stays byte-exact recoverable — never
/// wrong, only a wasted offload.
fn looks_like_log(lines: &[&str]) -> bool {
    if lines.len() >= 8 {
        let logish = lines
            .iter()
            .filter(|line| {
                let t = line.trim_start();
                t.starts_with(|c: char| c.is_ascii_digit()) || t.starts_with('[')
            })
            .count();
        if logish * 2 >= lines.len() {
            return true;
        }
    }
    lines.len() >= 16 && low_first_token_diversity(lines)
}

/// True for verbose `git status` output: high-value machine status, but not a file the model needs
/// to read verbatim. The short form (`git status --short`) is already covered by low first-token
/// diversity; the long form has many distinct path tokens under `Untracked files`, so it needs its
/// own command-shape signal.
fn looks_like_git_status(lines: &[&str]) -> bool {
    if lines.len() < 8 {
        return false;
    }

    let has_branch_header = lines.iter().take(4).any(|line| {
        line.starts_with("On branch ")
            || line.starts_with("HEAD detached")
            || line.starts_with("Not currently on any branch")
    });
    if !has_branch_header {
        return false;
    }

    let has_status_section = lines.iter().any(|line| {
        matches!(
            line.trim_end(),
            "Changes to be committed:"
                | "Changes not staged for commit:"
                | "Untracked files:"
                | "Unmerged paths:"
        )
    });
    if !has_status_section {
        return false;
    }

    lines.iter().any(|line| {
        let trimmed = line.trim_start();
        trimmed.starts_with("modified:")
            || trimmed.starts_with("new file:")
            || trimmed.starts_with("deleted:")
            || trimmed.starts_with("renamed:")
            || trimmed.starts_with("copied:")
            || trimmed.starts_with("both modified:")
            || trimmed.starts_with("both deleted:")
            || trimmed.starts_with("use \"git add")
    })
}

/// True when the non-blank lines' first tokens (leading whitespace trimmed, then up to the first
/// whitespace or `:`) are ≤10% distinct — the signature of repetitive machine-generated bulk output.
fn low_first_token_diversity(lines: &[&str]) -> bool {
    let mut token_count = 0usize;
    let mut distinct = std::collections::HashSet::new();
    for line in lines {
        let trimmed = line.trim_start();
        if trimmed.is_empty() {
            continue;
        }
        let end = trimmed
            .find(|c: char| c.is_whitespace() || c == ':')
            .unwrap_or(trimmed.len());
        token_count += 1;
        distinct.insert(&trimmed[..end]);
    }
    if token_count < 16 {
        return false;
    }
    distinct.len() * 10 <= token_count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_json_array() {
        assert_eq!(detect(br#"[{"a":1},{"a":2}]"#), ContentType::Json);
    }

    #[test]
    fn detects_json_object() {
        assert_eq!(detect(br#"{"a":1}"#), ContentType::Json);
    }

    #[test]
    fn detects_json_object_after_leading_whitespace() {
        assert_eq!(detect(b" \n\t{\"a\":1}"), ContentType::Json);
    }

    #[test]
    fn bare_scalar_is_not_json_type() {
        // valid JSON, but not a container we compress
        assert_eq!(detect(b"42"), ContentType::Text);
    }

    #[test]
    fn detects_log() {
        let log = "2026-06-02 INFO start\n".repeat(10);
        assert_eq!(detect(log.as_bytes()), ContentType::Log);
    }

    #[test]
    fn prose_is_text() {
        assert_eq!(detect(b"just some prose here"), ContentType::Text);
    }

    #[test]
    fn non_utf8_is_text() {
        assert_eq!(detect(&[0xff, 0xfe, 0x00, 0x01]), ContentType::Text);
    }

    #[test]
    fn detects_crlf_log() {
        let log = "2026-06-02 INFO start\r\n".repeat(10);
        assert_eq!(detect(log.as_bytes()), ContentType::Log);
    }

    #[test]
    fn seven_lines_is_below_the_floor() {
        let log = "2026-06-02 INFO x\n".repeat(7);
        assert_eq!(detect(log.as_bytes()), ContentType::Text);
    }

    #[test]
    fn detects_build_output_by_low_first_token_diversity() {
        // T-COVERAGE: repetitive build output (few distinct first tokens) → Log, even though
        // no line starts with a digit or `[`.
        let mut s = String::new();
        for i in 0..40 {
            s.push_str(&format!("   Compiling crate_{i} v0.1.0\n"));
        }
        for i in 0..6 {
            s.push_str(&format!("   Running test_{i}\n"));
        }
        s.push_str("    Finished dev profile\n");
        assert_eq!(detect(s.as_bytes()), ContentType::Log);
    }

    #[test]
    fn detects_verbose_git_status() {
        let status = "\
On branch main
Changes not staged for commit:
  (use \"git add <file>...\" to update what will be committed)
  (use \"git restore <file>...\" to discard changes in working directory)
\tmodified:   README.md
\tmodified:   crates/coffer-core/src/detect.rs
\tmodified:   crates/coffer-core/src/compress.rs

Untracked files:
  (use \"git add <file>...\" to include in what will be committed)
\tdocs/deployment.md
\tscripts/package-smoke.sh
\tscripts/production-gate.sh

no changes added to commit (use \"git add\" and/or \"git commit -a\")
";
        assert_eq!(detect(status.as_bytes()), ContentType::Log);
    }

    #[test]
    fn source_code_is_not_log() {
        // Varied leading tokens (use/fn/let/if/struct/...) keep first-token diversity high → Text,
        // so the model still reads code whole.
        let code = "use std::io;\nfn main() {\n    let x = 1;\n    let y = 2;\n    if x > y {\n        \
                    foo(x);\n    } else {\n        bar(y);\n    }\n    for i in 0..10 {\n        \
                    step(i);\n    }\n    return;\n}\nstruct Foo { a: u8 }\nimpl Foo {\n    fn m(&self) {}\n}\n\
                    enum E { A, B }\ntype T = u32;\n";
        assert_eq!(detect(code.as_bytes()), ContentType::Text);
    }
}
