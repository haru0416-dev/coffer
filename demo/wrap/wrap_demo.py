#!/usr/bin/env python3
"""Thin presentation driver for the coffer-wrap demo tape.

Every subcommand talks REAL MCP JSON-RPC to the REAL `coffer-wrap` binary wrapping
`pods_server.py` — the driver only types the JSON-RPC for you and prints the results.
State between invocations is the shared SQLite CAS (`COFFER_CAS_DB`), which is itself
part of the story: the handle minted in one process is resolved in the next.

    get_pods   call the wrapped tool; print the fact card the model would see
    ask <q>    coffer_digest: exact natural-language stats over ALL rows
    errors     coffer_aggregate: exact filtered count with row provenance
    verify     coffer_retrieve pages + sha256: prove the handle is byte-exact
"""
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CAS = os.environ.get("COFFER_CAS_DB", os.path.join(HERE, "wrap-demo-cas.db"))
HANDLE_FILE = os.path.join(HERE, ".last-handle")


class Wrap:
    def __init__(self):
        self.p = subprocess.Popen(
            [os.path.join(ROOT, "target/release/coffer-wrap"),
             "python3", os.path.join(HERE, "pods_server.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True,
            env={**os.environ, "COFFER_CAS_DB": CAS},
        )
        self.n = 0
        self.request("initialize", {"protocolVersion": "2025-06-18",
                                    "capabilities": {},
                                    "clientInfo": {"name": "demo", "version": "0"}})

    def request(self, method, params):
        self.n += 1
        self.p.stdin.write(json.dumps(
            {"jsonrpc": "2.0", "id": self.n, "method": method, "params": params}) + "\n")
        self.p.stdin.flush()
        while True:
            m = json.loads(self.p.stdout.readline())
            if m.get("id") == self.n and "method" not in m:
                return m["result"]

    def call(self, name, args):
        return self.request("tools/call", {"name": name, "arguments": args})["content"][0]["text"]

    def close(self):
        self.p.stdin.close()
        self.p.wait(timeout=10)


def last_handle():
    with open(HANDLE_FILE) as f:
        return f.read().strip()


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "get_pods"
    w = Wrap()
    try:
        if cmd == "get_pods":
            card = w.call("get_pods", {})
            print(card)
            for line in card.splitlines():
                if line.startswith("handle: "):
                    with open(HANDLE_FILE, "w") as f:
                        f.write(line.split()[1])
        elif cmd == "ask":
            print(w.call("coffer_digest", {"handle": last_handle(),
                                           "query": " ".join(sys.argv[2:])}))
        elif cmd == "errors":
            print(w.call("coffer_aggregate", {
                "handle": last_handle(), "agg": "count",
                "where": [{"field": "status", "op": "eq", "value": "Error"}]}))
        elif cmd == "verify":
            handle = last_handle()
            data, start, total, pages = b"", 0, None, 0
            while total is None or start < total:
                page = w.call("coffer_retrieve",
                              {"handle": handle, "start": start, "len": 1 << 20})
                if page.startswith("[coffer-wrap] bytes "):
                    # "[coffer-wrap] bytes S..E of T total (…)\n<body>"
                    header, page = page.split("\n", 1)
                    total = int(header.split(" of ")[1].split()[0])
                else:
                    total = len(page.encode())  # whole blob fit in one window
                data += page.encode()
                start += 1 << 20
                pages += 1
            digest = hashlib.sha256(data).hexdigest()
            sys.path.insert(0, HERE)
            import pods_server
            original = pods_server.dump().encode()
            print(f"retrieved      : {len(data)} bytes over {pages} page(s)")
            print(f"sha256(bytes)  : {digest}")
            print(f"handle         : {handle}")
            print(f"sha256==handle : {digest == handle}")
            print(f"byte-identical to the server's original: {data == original}")
        else:
            print(__doc__)
    finally:
        w.close()


if __name__ == "__main__":
    main()
