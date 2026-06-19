# coffer demo — render with VHS:  vhs demo/demo.tape   →   demo/demo.gif
# VHS: https://github.com/charmbracelet/vhs   (needs ttyd + ffmpeg)
# Run from the repository root.

Output demo/demo.gif

Set Shell "bash"
Set FontSize 18
Set Width 1200
Set Height 600
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 38ms

# --- setup (hidden so the gif isn't a compile log): sample data + prebuild ---
Hide
Type "python3 demo/gen.py > /tmp/pods.json" Enter
Type "cargo build -q -p coffer-core --example filter" Enter
Sleep 3s
Type "clear" Enter
Show

# --- the demo ---
Type "du -h /tmp/pods.json   # a kubectl dump: 5000 pods"
Sleep 700ms  Enter  Sleep 1800ms

Type "cargo run -q -p coffer-core --example filter 0.9 < /tmp/pods.json > /tmp/out.json"
Sleep 700ms  Enter  Sleep 2800ms

Type "grep -o '<<cof:[^>]*>>' /tmp/out.json   # the elided middle = one recoverable handle"
Sleep 700ms  Enter  Sleep 2400ms

Type "du -h /tmp/out.json   # lossless: every byte is still recoverable"
Sleep 700ms  Enter  Sleep 2800ms

Sleep 1500ms
