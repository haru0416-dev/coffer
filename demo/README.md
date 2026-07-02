# demo

Two short terminal demos, both recorded **reproducibly and without a terminal recorder**:
each `record` script runs the real commands, captures their real output, and composes an
[asciicast v2](https://docs.asciinema.org/manual/asciicast/v2/) around it — the only
synthetic part is the typing cadence. [agg](https://github.com/asciinema/agg) renders the
gifs, so there is no hand-editing anywhere and every number on screen comes from the run.

```sh
# engine demo → demo/demo.gif  (embedded at the top of the main README)
python3 demo/record_demo.py          # builds the example, runs the pipeline, writes demo/demo.cast
agg --theme dracula --font-size 16 demo/demo.cast demo/demo.gif

# gateway demo → demo/wrap.gif
cargo build --release -p coffer-wrap
python3 demo/wrap/record.py          # runs the real MCP session, writes demo/wrap.cast
agg --theme dracula --font-size 16 demo/wrap.cast demo/wrap.gif
```

The `.cast` files are checked in too — play them in a terminal with `asciinema play`.

## engine demo (`demo.gif`) — compress ~90%, rebuild byte-for-byte

A `kubectl get pods -o json`-shaped dump (5,000 nested Pod objects, ~5.7 MB ≈ 2.17M real
`o200k_base` tokens) is cut to ~217K tokens in one pass that ALSO rebuilds the original;
the gif then shows that `packed.json` is still ordinary JSON (head and tail pods verbatim),
that the 4,500 middle pods collapsed into one `<<cof:…>>` marker, and a `cmp` proving the
rebuilt file is byte-identical. `./coffer` in the gif is `coffer-core`'s `filter` example
(a PoC, not a shipped CLI), built with `--features tiktoken` so the token numbers are
OpenAI's own count, not the chars/4 estimate:

```sh
python3 demo/gen.py > pods.json                                        # 5000 kubectl-shaped pods
cargo build --release -p coffer-core --example filter --features tiktoken
ln -sf ../target/release/examples/filter demo/coffer

COFFER_ROUNDTRIP_OUT=back.json ./coffer 0.9 < pods.json > packed.json  # compress + rebuild
#   stderr: raw_tok=2170329  out_tok=216857  saved=90.0%  tok=openai-o200k_base  reversible=true
head -c 220 packed.json                                                # still ordinary JSON
grep -o '<<cof:[^>]*>>' packed.json                                    # <<cof:HASH +4500 items>>
cmp pods.json back.json                                                # exits 0: byte-identical
```

`0.9` is the reduction target (cut ~90%). The budget search runs on a fast heuristic; the
reported token counts use the real `o200k_base` tokenizer, so `saved` is a measurement, not
the knob echoed back.

## gateway demo (`wrap.gif`) — a flood-sized tool result becomes a queryable handle

An MCP server whose `get_pods` returns 20,000 pods as ONE ~1.1 MB tool result (≈ 269k
estimated tokens — far past the ~25k-token caps MCP hosts put on tool output). Wrapped with
`coffer-wrap`, the model receives a ~1 KB fact card instead; the injected tools then answer
exactly over ALL rows: `coffer_digest` finds the needle buried in the elided middle
(`pod-010000`, restarts=99999), `coffer_aggregate` counts the 2,000 Error pods with
row-index provenance, and `coffer_retrieve` pages the original back out —
`sha256(bytes) == handle`, byte-identical to what the server produced.

`demo/wrap/wrap_demo.py` is presentation sugar over real MCP JSON-RPC to the real
`coffer-wrap` binary wrapping `demo/wrap/pods_server.py`; state persists between
invocations through the shared SQLite CAS, which is itself the deployment story:

```sh
python3 wrap_demo.py get_pods    # tools/call through coffer-wrap → prints the fact card
python3 wrap_demo.py ask …       # coffer_digest: exact NL stats over all rows
python3 wrap_demo.py errors      # coffer_aggregate: exact count + provenance indices
python3 wrap_demo.py verify      # coffer_retrieve pages + sha256 == handle + byte compare
```
