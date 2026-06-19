# coffer demo — render with VHS:  vhs demo/demo.tape   →   demo/demo.gif
# VHS: https://github.com/charmbracelet/vhs   (needs ttyd + ffmpeg)
# Run from the repository root.

Output demo/demo.gif

Set Shell "bash"
Set FontSize 18
Set Width 1400
Set Height 360
Set Padding 20
Set Theme "Dracula"
Set TypingSpeed 18ms

# --- setup (hidden so the gif isn't a compile log) ---
# realistic kubectl-shaped data; build the example with the REAL o200k tokenizer; expose it as ./coffer.
Hide
Type "python3 demo/gen.py > pods.json" Enter
Type "cargo build -q --release -p coffer-core --example filter --features tiktoken" Enter
Sleep 6s
Type "cp target/release/examples/filter ./coffer" Enter
Type "clear" Enter
Show

# 1. the input — and why it is a problem for an LLM
Type "# a 'kubectl get pods -o json' dump (.items): 5000 pods. As context that's ~2.17M tokens — fits in nothing."
Sleep 200ms  Enter  Sleep 1700ms
Type "ls -lh pods.json"
Sleep 400ms  Enter  Sleep 1800ms

# 2. compress (keep head+tail, offload the middle) AND rebuild the original, in one pass
Type "# ./coffer = coffer-core's 'filter' example. 0.9 = reduce ~90%. compress, and reconstruct -> back.json:"
Sleep 200ms  Enter  Sleep 1600ms
Type "COFFER_ROUNDTRIP_OUT=back.json ./coffer 0.9 < pods.json > packed.json"
Sleep 400ms  Enter  Sleep 3400ms

# 3. the middle 4500 pods are now one handle; head & tail stay verbatim JSON
Type "grep -o '<<cof:[^>]*>>' packed.json   # head+tail kept verbatim; the middle 4500 pods -> one handle"
Sleep 400ms  Enter  Sleep 2600ms

# 4. the point: 'reversible=true' is not a guess — rebuild and byte-compare
Type "cmp pods.json back.json && echo 'identical: every offloaded byte recovered, nothing summarized'"
Sleep 400ms  Enter  Sleep 3000ms

# 5. honest framing for when the gif travels without the README
Type "# a PoC example; coffer ships as an MCP server + transparent proxy. status: experimental."
Sleep 200ms  Enter  Sleep 2400ms

Sleep 1200ms
