# Security Policy

coffer is **experimental** software. There are no released versions yet; fixes land on `main`.

## Reporting a vulnerability

Please report security issues **privately**, not in public issues or pull requests.

Use GitHub's private reporting: **Security → Report a vulnerability** on
<https://github.com/haru0416-dev/coffer/security/advisories>. Include what you found, how to reproduce
it, and the impact. We aim to acknowledge within a few days.

## What to look at

coffer sits in the request path and holds tool-output bytes, so the security-relevant surface is:

- **The proxy replays your upstream API key.** It refuses a non-loopback bind unless
  `COFFER_PROXY_ALLOW_PUBLIC=1`, and it has no auth of its own — do not expose it to an untrusted network.
- **The MCP `coffer_run` shell tool is disabled** unless `COFFER_MCP_ENABLE_RUN=1` (and is further gated by
  `COFFER_MCP_RUN_ALLOWLIST`). Treat enabling it as granting command execution.
- **The content store keeps original tool-output bytes** (in SQLite when `COFFER_CAS_DB` is set). Those
  bytes may be sensitive; protect the file like any data store.
- **Recovery is integrity-checked.** Every read re-verifies the SHA-256 content hash, so a corrupted or
  substituted blob fails closed (returns nothing) rather than handing back wrong bytes — reports that defeat
  this are especially welcome.

Out of scope: issues that require already having privileged local access the threat model grants (e.g. read
access to the CAS file you configured).
