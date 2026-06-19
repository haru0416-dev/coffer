#!/usr/bin/env python3
# Deterministic sample for the demo: a `kubectl get pods -o json` capture reduced to its `.items`
# array (coffer windows top-level JSON arrays). 5000 pods, each the usual nested Pod object with
# metadata / spec / status — uids, resourceVersions, image digests, conditions, containerStatuses.
import hashlib
import json
import random

random.seed(7)

APPS = ["checkout", "payments", "web", "search", "ingest", "worker"]
NAMESPACES = ["prod", "prod", "prod", "staging", "kube-system"]
PHASE_POOL = ["Running"] * 92 + ["Pending"] * 3 + ["CrashLoopBackOff"] * 3 + ["Succeeded"] * 2


def digest(*parts, n=12):
    return hashlib.sha256("/".join(str(p) for p in parts).encode()).hexdigest()[:n]


def uid(i):
    s = hashlib.sha256(f"uid/{i}".encode()).hexdigest()
    return f"{s[:8]}-{s[8:12]}-4{s[13:16]}-{s[16:20]}-{s[20:32]}"


pods = []
for i in range(5000):
    app = APPS[i % len(APPS)]
    namespace = NAMESPACES[i % len(NAMESPACES)]
    phase = random.choice(PHASE_POOL)
    running = phase == "Running"
    restarts = random.randint(7, 240) if phase == "CrashLoopBackOff" else random.choice([0, 0, 0, 0, 1, 2])
    image = f"registry.internal/{app}:1.{i % 40}.{i % 7}"
    image_id = f"docker-pullable://{image}@sha256:{digest('img', image, n=64)}"
    ts = f"2026-06-{1 + i % 27:02d}T{i % 24:02d}:{i % 60:02d}:{(i * 7) % 60:02d}Z"
    pods.append(
        {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": f"{app}-{digest(app, i // 7)}-{digest(app, i, n=5)}",
                "namespace": namespace,
                "uid": uid(i),
                "resourceVersion": str(73_000_000 + i * 7),
                "creationTimestamp": ts,
                "labels": {"app": app, "pod-template-hash": digest(app, i // 7, n=10)},
            },
            "spec": {
                "nodeName": f"ip-10-0-{i % 200}-{(i * 3) % 250}.ec2.internal",
                "serviceAccountName": app,
                "containers": [
                    {
                        "name": app,
                        "image": image,
                        "ports": [{"containerPort": 8080, "protocol": "TCP"}],
                        "resources": {"requests": {"cpu": "100m", "memory": "128Mi"}},
                    }
                ],
            },
            "status": {
                "phase": phase,
                "podIP": f"10.{i % 240}.{(i * 5) % 250}.{(i * 11) % 250}",
                "startTime": ts,
                "containerStatuses": [
                    {
                        "name": app,
                        "ready": running,
                        "restartCount": restarts,
                        "image": image,
                        "imageID": image_id,
                        "state": ({"running": {"startedAt": ts}} if running else {"waiting": {"reason": phase}}),
                    }
                ],
                "conditions": [
                    {"type": "Initialized", "status": "True"},
                    {"type": "Ready", "status": "True" if running else "False"},
                    {"type": "ContainersReady", "status": "True" if running else "False"},
                    {"type": "PodScheduled", "status": "True"},
                ],
            },
        }
    )

print(json.dumps(pods))
