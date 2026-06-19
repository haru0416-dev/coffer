# demo

A short terminal demo of coffer's reversible compression, recorded reproducibly with
[VHS](https://github.com/charmbracelet/vhs) (no manual screen recording — the `.tape` is the script).

## Render the GIF

VHS needs `ttyd` and `ffmpeg`. Install it — `brew install vhs` (macOS),
`sudo pacman -S vhs ttyd ffmpeg` (Arch), `apt install ffmpeg` + the VHS release binary (Debian/Ubuntu),
or see <https://github.com/charmbracelet/vhs#installation>. Then, from the repository root:

```sh
vhs demo/demo.tape        # writes demo/demo.gif
```

The tape generates a 5000-record `kubectl`-style dump, compresses it through the
`coffer-core` filter example, shows the elided middle collapsed to a single `<<cof:…>>` handle, and the
shrunken output — all of it still byte-for-byte recoverable.

## What it runs

```sh
python3 demo/gen.py > pods.json                                   # 5000 pods, ~285 KB
cargo run -q -p coffer-core --example filter 0.9 < pods.json      # stderr: saved ~90%  reversible=true
grep -o '<<cof:[^>]*>>' out.json                                  # <<cof:HASH +N items>>
```

To embed it in the top-level README once rendered:

```md
![coffer demo](demo/demo.gif)
```
