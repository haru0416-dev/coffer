#!/usr/bin/env python3
"""Reproducibly record the main engine demo (compression + byte-exact rebuild) as an
asciicast. Runs the REAL pipeline — gen.py data, the `filter` example built with the real
o200k tokenizer, grep, cmp — captures the real output, and scripts only the typing cadence.
Every number shown in the narration comments is parsed from the actual run.

    python3 demo/record_demo.py          # builds + runs everything, writes demo/demo.cast
    agg --theme dracula --font-size 16 demo/demo.cast demo/demo.gif
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from castlib import Cast


def sh(cmd):
    """Run `cmd` with bash in demo/ (the dir the typed commands run in)."""
    r = subprocess.run(
        ["bash", "-c", cmd], capture_output=True, text=True, cwd=HERE, check=True
    )
    return r.stdout.rstrip("\n"), r.stderr.rstrip("\n")


def main():
    print("preparing data + release build (real o200k tokenizer) …", file=sys.stderr)
    sh("python3 gen.py > pods.json")
    subprocess.run(
        ["cargo", "build", "-q", "--release", "-p", "coffer-core",
         "--example", "filter", "--features", "tiktoken"],
        cwd=ROOT, check=True,
    )
    sh("ln -sf ../target/release/examples/filter ./coffer")

    print("running the real pipeline …", file=sys.stderr)
    ls_out, _ = sh("ls -lh pods.json")
    _, metrics = sh("COFFER_ROUNDTRIP_OUT=back.json ./coffer 0.9 < pods.json > packed.json")
    head_out, _ = sh("head -c 220 packed.json")
    sentinel, _ = sh("grep -o '<<cof:[^>]*>>' packed.json")
    cmp_out, _ = sh(
        "cmp pods.json back.json && echo 'identical: every offloaded byte recovered, nothing summarized'"
    )

    m = re.search(r"raw_tok=(\d+)\s+out_tok=(\d+)\s+saved=([\d.]+%)", metrics)
    raw_tok, out_tok, saved = int(m.group(1)), int(m.group(2)), m.group(3)
    items = re.search(r"\+(\d+) items", sentinel).group(1)

    cast = Cast(title="coffer: compress tool-output ~90%, rebuild it byte-for-byte")
    cast.comment(
        "a real `kubectl get pods -o json` shape: 5,000 nested Pod objects. can a model read this?",
        tail=1.2,
    )
    cast.command("ls -lh pods.json")
    cast.output(ls_out, pause_after=1.2)
    cast.comment(
        f"as context that is {raw_tok:,} tokens — counted with OpenAI's o200k_base, and no window fits it.",
        tail=1.8,
    )

    cast.comment(
        "./coffer = coffer-core's `filter` example. cut ~90% AND rebuild the original, in one pass:",
        tail=1.0,
    )
    cast.command("COFFER_ROUNDTRIP_OUT=back.json ./coffer 0.9 < pods.json > packed.json")
    cast.output(metrics, pause_after=1.6)
    cast.comment(
        f"↑ {raw_tok:,} → {out_tok:,} tokens ({saved} saved) — measured by the tokenizer, not the knob echoed back.",
        tail=2.4,
    )

    cast.comment("packed.json is still ordinary JSON — head and tail pods are untouched:", tail=1.0)
    cast.command("head -c 220 packed.json")
    cast.output(head_out, pause_after=2.6)
    cast.comment(
        f"…and the {int(items):,} MIDDLE pods collapsed into one {len(sentinel)}-byte marker"
        " (their bytes sit in a content store):",
        tail=1.0,
    )
    cast.command("grep -o '<<cof:[^>]*>>' packed.json")
    cast.output(sentinel, pause_after=2.8)

    cast.comment("'reversible' is not a claim — back.json was rebuilt from the store. byte-compare:", tail=1.0)
    cast.command("cmp pods.json back.json && echo 'identical: every offloaded byte recovered, nothing summarized'")
    cast.output(cmp_out, pause_after=3.2)

    cast.comment("exact answers OVER the elided middle (count/max/join, with provenance) → demo/wrap.gif", tail=1.6)
    cast.comment("a PoC example; coffer ships as an MCP server + gateway + transparent proxy. status: experimental.", tail=2.6)

    secs, events = cast.write(os.path.join(HERE, "demo.cast"))
    print(f"wrote demo/demo.cast ({secs:.1f}s, {events} events)", file=sys.stderr)


if __name__ == "__main__":
    main()
