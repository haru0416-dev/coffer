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
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CAST = os.path.join(ROOT, "demo", "wrap.cast")

COLS, ROWS = 132, 42
TYPE_MS = 0.014  # per typed character
PROMPT = "\x1b[1;32m$\x1b[0m "

events = []
now = 0.0


def emit(text, dt=0.0):
    global now
    now += dt
    events.append([round(now, 4), "o", text])


def type_line(line, lead=0.15, tail=0.25, color=None):
    """Simulate typing one shell line (comment or command) at the prompt."""
    emit(PROMPT, lead)
    shown = f"\x1b[{color}m{line}\x1b[0m" if color else line
    # type per character over len*TYPE_MS, as a few chunks so the cast stays small
    chunk = max(1, len(shown) // 24)
    for i in range(0, len(shown), chunk):
        emit(shown[i : i + chunk], TYPE_MS * chunk)
    emit("\r\n", tail)


def output(text, pause_after=2.0):
    emit(text.replace("\n", "\r\n") + "\r\n", 0.12)
    global now
    now += pause_after


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
    errors = "\n".join(
        line[:118] for line in run("errors").splitlines()[:2]
    )
    verify = run("verify")

    # --- compose the cast ---
    grey, cyan = "90", "36"
    type_line(
        "# an MCP server's get_pods returns 20,000 pods as ONE tool result: ~1.1 MB ≈ 269k tokens.",
        tail=1.2, color=grey,
    )
    type_line(
        "# hosts cap tool output around 25k tokens — unwrapped, this call fails or floods the context.",
        tail=1.6, color=grey,
    )
    type_line(
        "# wrap the SAME server (coffer-wrap -- python3 pods_server.py). now the model sees:",
        tail=1.0, color=grey,
    )
    type_line("python3 wrap_demo.py get_pods", color=cyan)
    output(card, pause_after=6.5)

    emit("\x1b[2J\x1b[H", 0.3)  # clear for the query beats
    type_line(
        "# 'which pod restarts most?' — that answer sits in the ELIDED middle. ask the injected tools:",
        tail=1.2, color=grey,
    )
    type_line("python3 wrap_demo.py ask which pod has the most restarts", color=cyan)
    output(ask, pause_after=3.2)

    type_line("python3 wrap_demo.py errors | cut -c1-118 | head -2", color=cyan)
    output(errors, pause_after=3.2)

    type_line("# 'preserved byte-exact' is not a promise — recompute it:", tail=1.0, color=grey)
    type_line("python3 wrap_demo.py verify", color=cyan)
    output(verify, pause_after=4.0)

    type_line(
        "# every answer was computed over all 20,000 rows server-side; the context got ~1 KB. status: experimental.",
        tail=2.6, color=grey,
    )

    header = {
        "version": 2,
        "width": COLS,
        "height": ROWS,
        "title": "coffer-wrap: oversized tool results become verified, queryable handles",
        "env": {"TERM": "xterm-256color", "SHELL": "/bin/bash"},
    }
    with open(CAST, "w") as f:
        f.write(json.dumps(header) + "\n")
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    print(f"wrote {CAST} ({now:.1f}s, {len(events)} events)", file=sys.stderr)


if __name__ == "__main__":
    main()
