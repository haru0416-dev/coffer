# demo

A short terminal demo of coffer's reversible compression, recorded reproducibly with
[VHS](https://github.com/charmbracelet/vhs) — the `.tape` is the script, so there is no hand-editing.

It compresses a `kubectl get pods -o json`-shaped dump (5000 Pod objects, ~5.9 MB) and shows, on
screen: the real OpenAI `o200k_base` token count dropping from 2.17M to 217K, the middle 4500 pods
collapsing to a single `<<cof:…>>` handle (head and tail kept verbatim), and a `cmp` proving the
original rebuilds byte-for-byte.

## Render the GIF

VHS needs `ttyd` and `ffmpeg`. Install it — `brew install vhs` (macOS),
`sudo pacman -S vhs ttyd ffmpeg` (Arch), `apt install ffmpeg` + the VHS release binary (Debian/Ubuntu),
or see <https://github.com/charmbracelet/vhs#installation>. Then, from the repository root:

```sh
vhs demo/demo.tape        # writes demo/demo.gif
```

The tape builds the example with the real tokenizer (`--release --features tiktoken`), so the token
numbers on screen come from OpenAI's `o200k_base`, not the chars/4 estimate.

## What it runs

`./coffer` in the tape is just `coffer-core`'s `filter` example (a PoC, not a shipped CLI):

```sh
python3 demo/gen.py > pods.json                                        # 5000 kubectl-shaped pods, ~5.9 MB
cargo build --release -p coffer-core --example filter --features tiktoken
cp target/release/examples/filter ./coffer

COFFER_ROUNDTRIP_OUT=back.json ./coffer 0.9 < pods.json > packed.json  # compress (keep head+tail) + rebuild
#   stderr: raw_tok=2170329  out_tok=216857  saved=90.0%  tok=openai-o200k_base  reversible=true
grep -o '<<cof:[^>]*>>' packed.json                                    # <<cof:HASH +4500 items>>
cmp pods.json back.json                                                # exits 0: byte-for-byte identical
```

`0.9` is the reduction target (cut ~90%). The budget search runs on a fast heuristic; the reported
token counts use the real `o200k_base` tokenizer, so `saved` is a measurement, not the knob echoed back.

The GIF is embedded at the top of the [main README](../README.md) via `![coffer demo](demo/demo.gif)`.

## coffer-wrap demo (`wrap.gif`)

The second gif shows the MCP gateway: a server whose `get_pods` returns 20,000 pods as ONE
~1.1 MB tool result (≈ 269k estimated tokens — far past the ~25k-token caps MCP hosts put on
tool output). Wrapped with `coffer-wrap`, the model receives a ~1 KB fact card instead; the
injected tools then answer exactly over ALL rows: `coffer_digest` finds the needle buried in
the elided middle (`pod-010000`, restarts=99999), `coffer_aggregate` counts the 2,000 Error
pods with row-index provenance, and `coffer_retrieve` pages the original back out —
`sha256(bytes) == handle`, byte-identical to what the server produced.

It is recorded without a terminal recorder at all: `demo/wrap/record.py` runs the real
commands (real binary, real MCP JSON-RPC, real output — only the typing cadence is
synthetic) and writes an [asciicast](https://docs.asciinema.org/manual/asciicast/v2/), which
[agg](https://github.com/asciinema/agg) renders to a gif:

```sh
cargo build --release -p coffer-wrap
python3 demo/wrap/record.py                                  # runs the demo, writes demo/wrap.cast
agg --theme dracula --font-size 16 demo/wrap.cast demo/wrap.gif
```

What the driver subcommands actually do (`demo/wrap/wrap_demo.py` is presentation sugar over
real MCP calls; state persists between invocations through the shared SQLite CAS, which is
itself the deployment story):

```sh
python3 wrap_demo.py get_pods    # tools/call through coffer-wrap → prints the fact card
python3 wrap_demo.py ask …       # coffer_digest: exact NL stats over all rows
python3 wrap_demo.py errors      # coffer_aggregate: exact count + provenance indices
python3 wrap_demo.py verify      # coffer_retrieve pages + sha256 == handle + byte compare
```
