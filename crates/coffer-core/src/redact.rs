//! Detect-and-redact secrets before they reach the model.
//!
//! coffer's content store keeps the original bytes, so a secret in tool output is recoverable — which
//! is exactly why it must not reach the model in the first place. This module masks secret tokens out
//! of the model-facing text and returns them separately, so an authorized (non-model) retrieve can
//! still recover them: redact-yet-recover, which only a reversible store can offer. Detection is
//! conservative — known secret-token prefixes only, to keep false positives near zero.

use coffer_cas::ContentHash;

/// A secret found and masked out of the model-facing text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Secret {
    /// The sentinel that replaced the secret in the masked text (`<<cof-redacted:HASH>>`).
    pub marker: String,
    /// The original secret value (to be stored access-gated, never shown to the model).
    pub value: String,
    /// Content hash of the secret value.
    pub hash: ContentHash,
}

/// The result of redacting text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Redacted {
    /// Model-facing text with every detected secret replaced by its redaction sentinel.
    pub masked: String,
    /// The secrets removed, in first-seen order — recoverable only by an authorized retrieve.
    pub secrets: Vec<Secret>,
}

/// Known secret-token prefixes (provider API keys / tokens). Conservative on purpose.
const SECRET_PREFIXES: &[&str] = &[
    "sk-",
    "sk_live_",
    "pk_live_",
    "rk_live_",
    "ghp_",
    "gho_",
    "ghs_",
    "ghu_",
    "github_pat_",
    "glpat-",
    "xoxb-",
    "xoxp-",
    "xoxa-",
    "AKIA",
    "ASIA",
    "AIza",
    "ya29.",
    "AGPA",
];

/// Whether `token` looks like a secret: a known prefix and enough length to be a real key
/// (so a bare `sk-` or `AKIA` word is not masked).
fn looks_secret(token: &str) -> bool {
    token.len() >= 12 && SECRET_PREFIXES.iter().any(|p| token.starts_with(p))
}

fn is_sep(c: char) -> bool {
    c.is_whitespace()
        || matches!(
            c,
            '"' | '\''
                | ','
                | ':'
                | ';'
                | '='
                | '('
                | ')'
                | '{'
                | '}'
                | '['
                | ']'
                | '<'
                | '>'
                | '&'
                | '?'
                | '/'
                | '\\'
                | '`'
                | '|'
        )
}

fn flush(token: &mut String, masked: &mut String, secrets: &mut Vec<Secret>) {
    if token.is_empty() {
        return;
    }
    if looks_secret(token) {
        let hash = ContentHash::of(token.as_bytes());
        let marker = format!("<<cof-redacted:{}>>", hash.short());
        masked.push_str(&marker);
        secrets.push(Secret {
            marker,
            value: std::mem::take(token),
            hash,
        });
    } else {
        masked.push_str(token);
    }
    token.clear();
}

/// Mask secret tokens out of `text` for the model, returning the masked text and the secrets removed.
/// The masked text is `text` with each secret token replaced by a `<<cof-redacted:HASH>>` sentinel;
/// substituting each secret's value back for its marker reproduces `text` exactly (the recovery an
/// authorized party performs).
#[must_use]
pub fn redact_secrets(text: &str) -> Redacted {
    let mut masked = String::with_capacity(text.len());
    let mut secrets = Vec::new();
    let mut token = String::new();
    for c in text.chars() {
        if is_sep(c) {
            flush(&mut token, &mut masked, &mut secrets);
            masked.push(c);
        } else {
            token.push(c);
        }
    }
    flush(&mut token, &mut masked, &mut secrets);
    Redacted { masked, secrets }
}

#[cfg(test)]
mod tests {
    use super::{SECRET_PREFIXES, looks_secret, redact_secrets};
    use proptest::prelude::*;

    #[test]
    fn redacts_known_secret_tokens_and_round_trips() {
        let text = "export OPENAI_API_KEY=sk-abc123def456ghi789 and token ghp_0123456789abcdefABCDEF; normal=value";
        let r = redact_secrets(text);
        // secrets are gone from the model-facing text, replaced by sentinels.
        assert!(!r.masked.contains("sk-abc123def456ghi789"), "{}", r.masked);
        assert!(!r.masked.contains("ghp_0123456789abcdefABCDEF"));
        assert!(r.masked.contains("<<cof-redacted:"));
        assert_eq!(r.secrets.len(), 2);
        // non-secret tokens untouched.
        assert!(r.masked.contains("normal=value") && r.masked.contains("export"));
        // authorized recovery: substitute the secrets back → original, byte-for-byte.
        let mut restored = r.masked.clone();
        for s in &r.secrets {
            restored = restored.replace(&s.marker, &s.value);
        }
        assert_eq!(restored, text);
    }

    #[test]
    fn leaves_secret_free_text_unchanged() {
        let text = "just some normal prose with numbers 12345 and ordinary-words";
        let r = redact_secrets(text);
        assert_eq!(r.masked, text);
        assert!(r.secrets.is_empty());
    }

    /// One real-shaped token per known prefix, each >= 12 chars (the length gate), built from the
    /// prefix + separator-free filler so each stays a single token.
    fn corpus_tokens() -> Vec<String> {
        let body = "abcdefABCDEF0123456789-_.";
        SECRET_PREFIXES
            .iter()
            .map(|p| {
                let mut t = (*p).to_string();
                let mut i = 0;
                while t.len() < 24 {
                    let b = body.as_bytes()[i % body.len()] as char;
                    t.push(b);
                    i += 1;
                }
                t
            })
            .collect()
    }

    /// RECALL: a corpus embedding one real-shaped token per known prefix is fully masked.
    #[test]
    fn recall_every_known_prefix_is_masked() {
        let tokens = corpus_tokens();
        for t in &tokens {
            assert!(looks_secret(t), "constructed token not seen as secret: {t}");
        }
        let text = format!(
            "line0: secret={} end\nexport KEY=\"{}\" ; another({}) [{}] <{}> {}/{}|{}&{}?{}\n{} {} {} {} {} {} {} {}",
            tokens[0],
            tokens[1],
            tokens[2],
            tokens[3],
            tokens[4],
            tokens[5],
            tokens[6],
            tokens[7],
            tokens[8],
            tokens[9],
            tokens[10],
            tokens[11],
            tokens[12],
            tokens[13],
            tokens[14],
            tokens[15],
            tokens[16],
            tokens[17],
        );
        let r = redact_secrets(&text);
        for t in &tokens {
            assert!(
                !r.masked.contains(t.as_str()),
                "secret survived in masked text: {t}\nmasked: {}",
                r.masked
            );
        }
        assert_eq!(r.secrets.len(), tokens.len(), "one secret per prefix");
        let mut got: Vec<&str> = r.secrets.iter().map(|s| s.value.as_str()).collect();
        let mut want: Vec<&str> = tokens.iter().map(String::as_str).collect();
        got.sort_unstable();
        want.sort_unstable();
        assert_eq!(got, want, "recovered secret set differs from corpus");
        let mut restored = r.masked.clone();
        for s in &r.secrets {
            restored = restored.replacen(&s.marker, &s.value, 1);
        }
        assert_eq!(restored, text, "recall corpus failed byte-exact round-trip");
    }

    /// PRECISION: benign tokens that *look* key-ish must NOT be masked.
    #[test]
    fn precision_benign_tokens_not_masked() {
        let benign = [
            "550e8400-e29b-41d4-a716-446655440000",
            "da39a3ee5e6b4b0d3255bfef95601890afd80709",
            "/usr/local/share/some/config/path.toml",
            "supercalifragilisticexpialidocious",
            "Authorization",
            "antidisestablishmentarianism",
            "0123456789abcdef0123456789abcdef",
            "package-lock-resolver-internal-token",
        ];
        for b in &benign {
            assert!(!looks_secret(b), "benign token misclassified: {b}");
        }
        let text = format!(
            "uuid={} sha={} path={} word={} hdr={} word2={} hex={} name={}",
            benign[0], benign[1], benign[2], benign[3], benign[4], benign[5], benign[6], benign[7],
        );
        let r = redact_secrets(&text);
        assert!(r.secrets.is_empty(), "false positives: {:?}", r.secrets);
        assert_eq!(r.masked, text, "benign text must pass through unchanged");
    }

    /// FALSIFICATION control: an unknown-prefix token (and a too-short known-prefix one) are left
    /// alone, proving the recall predicate above is sensitive and the length gate is enforced.
    #[test]
    fn falsification_unknown_prefix_is_not_masked() {
        let unknown = "notakey_0123456789abcdefABCDEF";
        assert!(!looks_secret(unknown));
        let r = redact_secrets(&format!("token={unknown} done"));
        assert!(
            r.masked.contains(unknown),
            "control: unknown-prefix unexpectedly masked"
        );
        assert!(r.secrets.is_empty());

        let short = "sk-abc"; // known prefix but < 12 chars
        assert!(!looks_secret(short));
        let r2 = redact_secrets(&format!("k={short} x"));
        assert!(
            r2.secrets.is_empty(),
            "short known-prefix token must not mask"
        );
    }

    fn sep_strategy() -> impl Strategy<Value = char> {
        prop_oneof![
            Just(' '),
            Just('\n'),
            Just('='),
            Just(','),
            Just(';'),
            Just(':'),
            Just('"'),
            Just('('),
            Just(')'),
            Just('/'),
            Just('|'),
        ]
    }

    fn benign_token() -> impl Strategy<Value = String> {
        "[a-zA-Z0-9._-]{1,20}".prop_filter("must not look like a secret", |s| !looks_secret(s))
    }

    fn secret_token() -> impl Strategy<Value = String> {
        (0..SECRET_PREFIXES.len(), "[a-zA-Z0-9]{12,24}")
            .prop_map(|(i, tail)| format!("{}{}", SECRET_PREFIXES[i], tail))
            .prop_filter("padded to pass length gate", |s| looks_secret(s))
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]

        /// ROUND-TRIP over arbitrary text with secrets spliced in: substituting each secret value
        /// back for its marker (first-seen order) reproduces the input byte-for-byte, and the
        /// detected secret set/order matches an independent reference.
        #[test]
        fn round_trip_and_count_match_reference(
            frags in proptest::collection::vec(
                (prop_oneof![benign_token().prop_map(|t| (false, t)),
                             secret_token().prop_map(|t| (true, t))],
                 sep_strategy()),
                0..24),
            lead in sep_strategy(),
        ) {
            let mut text = String::new();
            text.push(lead);
            let mut expected_secret_values: Vec<String> = Vec::new();
            for (is_secret, tok, sep) in frags.iter().map(|((b, t), s)| (*b, t.clone(), *s)) {
                text.push_str(&tok);
                if is_secret {
                    expected_secret_values.push(tok);
                }
                text.push(sep);
            }

            let r = redact_secrets(&text);

            prop_assert_eq!(r.secrets.len(), expected_secret_values.len(),
                "secret count mismatch; masked={:?}", r.masked);
            let got: Vec<&str> = r.secrets.iter().map(|s| s.value.as_str()).collect();
            let want: Vec<&str> = expected_secret_values.iter().map(String::as_str).collect();
            prop_assert_eq!(&got, &want, "secret values/order mismatch");

            for v in &expected_secret_values {
                prop_assert!(!r.masked.contains(v.as_str()),
                    "secret survived masking: {} in {:?}", v, r.masked);
            }

            let mut restored = r.masked.clone();
            for s in &r.secrets {
                let found = restored.replacen(&s.marker, &s.value, 1);
                prop_assert_ne!(&found, &restored, "marker not found for {:?}", s.marker);
                restored = found;
            }
            prop_assert_eq!(restored, text, "round-trip not byte-exact");
        }
    }
}
