#!/usr/bin/env python3
"""Reproducibly record the coffer-wrap demo as an asciicast (no terminal recorder, no
hand-editing): run the real driver commands, capture their REAL output, and compose the
typing/timing script around it. Render the gif with agg:

    cargo build --release -p coffer-wrap
    python3 demo/wrap/record.py          # writes demo/wrap.cast (runs the real commands)
    agg --theme dracula --font-size 16 demo/wrap.cast demo/wrap.gif

Every byte of command output in the cast comes from the actual run — the only synthetic
part is the typing cadence.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "demo"))
from castlib import Cast

CAST = os.path.join(ROOT, "demo", "wrap.cast")


def run(*args):
    r = subprocess.run(
        [sys.executable, os.path.join(HERE, "wrap_demo.py"), *args],
        capture_output=True,
        text=True,
        cwd=HERE,
        check=True,
    )
    return r.stdout.rstrip("\n")


def main():
    # fresh state so the run in the gif is the run you can reproduce
    for f in ("wrap-demo-cas.db", ".last-handle"):
        try:
            os.remove(os.path.join(HERE, f))
        except FileNotFoundError:
            pass

    print("running the real demo commands …", file=sys.stderr)
    card = run("get_pods")
    ask = run("ask", "which", "pod", "has", "the", "most", "restarts")
    errors = "\n".join(line[:118] for line in run("errors").splitlines()[:2])
    verify = run("verify")

    cast = Cast(title="coffer-wrap: oversized tool results become verified, queryable handles")
    cast.comment(
        "an MCP server's get_pods returns 20,000 pods as ONE tool result: ~1.1 MB ≈ 269k tokens.",
        tail=1.2,
    )
    cast.comment(
        "hosts cap tool output around 25k tokens — unwrapped, this call fails or floods the context.",
        tail=1.6,
    )
    cast.comment(
        "wrap the SAME server (coffer-wrap -- python3 pods_server.py). now the model sees:",
        tail=1.0,
    )
    cast.command("python3 wrap_demo.py get_pods")
    cast.output(card, pause_after=6.5)

    cast.clear()
    cast.comment(
        "'which pod restarts most?' — that answer sits in the ELIDED middle. ask the injected tools:",
        tail=1.2,
    )
    cast.command("python3 wrap_demo.py ask which pod has the most restarts")
    cast.output(ask, pause_after=3.2)

    cast.command("python3 wrap_demo.py errors | cut -c1-118 | head -2")
    cast.output(errors, pause_after=3.2)

    cast.comment("'preserved byte-exact' is not a promise — recompute it:", tail=1.0)
    cast.command("python3 wrap_demo.py verify")
    cast.output(verify, pause_after=4.0)

    cast.comment(
        "every answer was computed over all 20,000 rows server-side; the context got ~1 KB. status: experimental.",
        tail=2.6,
    )

    secs, events = cast.write(CAST)
    print(f"wrote {CAST} ({secs:.1f}s, {events} events)", file=sys.stderr)


if __name__ == "__main__":
    main()
