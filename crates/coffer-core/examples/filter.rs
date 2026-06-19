//! PoC (NOT shipped): a transparent-layer filter — read tool-output bytes on stdin, emit the
//! coffer-compressed `<<cof:…>>` render on stdout, and print savings + a reversibility check to
//! stderr. Models the transformation a proxy/hook would apply. Usage:
//!   cargo run -q -p coffer-core --example filter -- [reduction_frac]   < tool_output > compressed
//!   cargo run -q -p coffer-core --example filter -- --structural-code [target_tokens] < source > outline
//! reduction_frac defaults to 0.8 (aim to cut 80%).

use std::io::{Read, Write};

use coffer_cas::MemoryCas;
use coffer_core::{Budget, Compressor, compress_structural_code_to_budget, detect};
use coffer_tokenizer::{HeuristicCounter, TokenCounter};

fn main() {
    let mut args = std::env::args().skip(1);
    let first_arg = args.next();
    let mut input = Vec::new();
    std::io::stdin()
        .read_to_end(&mut input)
        .expect("read stdin");

    let cas = MemoryCas::new();
    let counter = HeuristicCounter;
    if matches!(
        first_arg.as_deref(),
        Some("--structural-code" | "structural_code")
    ) {
        let target_tokens = args.next().and_then(|s| s.parse().ok()).unwrap_or(1024);
        let compact = compress_structural_code_to_budget(&input, &cas, target_tokens, &counter);
        let reversible = compact
            .reconstruct(&cas)
            .map(|b| b == input)
            .unwrap_or(false);
        report_stats(
            "view=structural_code",
            &input,
            &compact.model_text,
            reversible,
            &counter,
        );
        std::io::stdout()
            .write_all(compact.model_text.as_bytes())
            .expect("write stdout");
        return;
    }

    let frac: f32 = first_arg.and_then(|s| s.parse().ok()).unwrap_or(0.8);
    let content_type = detect(&input);

    let doc = Compressor::new()
        .budget(Budget::Reduction(frac))
        .counter(&counter)
        .min_bytes(0)
        .compress(&input, &cas)
        .expect("a counter is set");

    let rendered = doc.render_for_model();
    let reversible = doc.reconstruct(&cas).map(|b| b == input).unwrap_or(false);

    report_stats(
        &format!("type={content_type:?}"),
        &input,
        &rendered,
        reversible,
        &counter,
    );
    std::io::stdout()
        .write_all(rendered.as_bytes())
        .expect("write stdout");
}

fn report_stats(
    label: &str,
    input: &[u8],
    rendered: &str,
    reversible: bool,
    counter: &impl TokenCounter,
) {
    let raw_tok = counter.count(&String::from_utf8_lossy(input));
    let out_tok = counter.count(rendered);
    let saved = if raw_tok > 0 {
        100.0 * (raw_tok as f64 - out_tok as f64) / raw_tok as f64
    } else {
        0.0
    };

    eprintln!(
        "{label}  raw_bytes={}  out_bytes={}  raw_tok={raw_tok}  out_tok={out_tok}  saved={saved:.1}%  reversible={reversible}",
        input.len(),
        rendered.len(),
    );
}
