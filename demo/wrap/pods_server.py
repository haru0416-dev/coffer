#!/usr/bin/env python3
"""A minimal stdio MCP server for the coffer-wrap demo.

One tool, `get_pods`, returns 20,000 kubectl-shaped pod records as a single JSON tool
result (~1.4 MB). Deterministic: every 10th pod is Error, and the answer to "which pod
restarts most?" (pod-010000, restarts=99999) is buried in the middle — exactly the region
a head+tail window drops.
"""
import json
import sys

ROWS = 20_000
NEEDLE = ROWS // 2


def dump():
    return json.dumps(
        [
            {
                "name": f"pod-{i:06d}",
                "status": "Error" if i % 10 == 0 else "Running",
                "restarts": 99_999 if i == NEEDLE else i % 7,
            }
            for i in range(ROWS)
        ],
        separators=(",", ":"),
    )


def main():
    for line in sys.stdin:
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        mid, meth = m.get("id"), m.get("method")
        if mid is None:
            continue  # notification
        if meth == "initialize":
            res = {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "pods", "version": "0"},
            }
        elif meth == "tools/list":
            res = {
                "tools": [
                    {
                        "name": "get_pods",
                        "description": "List every pod as one JSON array.",
                        "inputSchema": {"type": "object", "properties": {}},
                    }
                ]
            }
        elif meth == "tools/call":
            res = {"content": [{"type": "text", "text": dump()}], "isError": False}
        else:
            res = {}
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": mid, "result": res}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
