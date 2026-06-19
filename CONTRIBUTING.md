# Contributing

Thanks for looking. coffer is early and experimental — issues, reproductions, and focused PRs are welcome.

## Build and test

```sh
cargo build --release
cargo test --workspace
cargo test -p coffer-cas --features sqlite   # the on-disk content store
```

Before opening a PR, run the same gates CI does:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## The one rule that can't bend

coffer's whole point is **byte-exact reversibility**: `reconstruct(compress(x)) == x` for any input. That
is a property-tested invariant, not a goal — a change that can lose or alter offloaded bytes is a bug, even
if it compresses better. Compression *effectiveness* is a separate axis from fidelity; never trade fidelity
for it.

If you touch the compressor, the CAS, or the proxy/MCP transforms, keep (and extend) the reversibility and
integrity tests.

## PRs

- One logical change per PR; keep the diff reviewable.
- A clear title and a sentence on *why* beats a long description.
- New behavior needs a test; the gates above must pass.

Not sure if something fits? Open an issue first and ask — that's cheaper than a large PR going the wrong way.
