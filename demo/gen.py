#!/usr/bin/env python3
# Deterministic sample for the demo: a `kubectl get pods -o json`-style dump of 5000 pods.
import json, random

random.seed(7)
rows = [
    {
        "name": f"svc-{i:04}",
        "phase": "Running" if random.random() > 0.04 else "CrashLoopBackOff",
        "restarts": (11 + i % 8) if random.random() < 0.05 else i % 9,
    }
    for i in range(5000)
]
print(json.dumps(rows))
