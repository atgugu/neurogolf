#!/usr/bin/env python3
"""Read-only access to the exact-reconciled Kaggle-paying design corpus."""
import hashlib
import json
import os

from config import CLEAN
CORPUS = os.path.join(CLEAN, "knowledge_proven_paying")
MANIFEST = os.path.join(CORPUS, "manifest.json")

WARNINGS = {
    396: "DIVERGENCE WARNING: the >=17.05-cost family pays; the ~16.6 family diverges.",
    109: "DIVERGENCE WARNING: v2 att-family is a hidden silent-zero; the proven pin design pays.",
    23: "DIVERGENCE WARNING: only exact-logic (mincover/And/Not/Or/Equal) designs pay.",
    192: "DIVERGENCE WARNING: the direct_conv3 family (lenient ~18.19) is Kaggle-divergent (#p788 shorted exactly its lenient).",
    153: "DIVERGENCE WARNING: historically Kaggle-divergent (confirmed via discriminator-probe subset-sum); novel families are solo-probe only.",
}


def load_manifest(path=MANIFEST):
    try:
        with open(path, errors="replace") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def corpus_artifact(task, manifest_sha, corpus=CORPUS):
    return os.path.join(corpus, f"task{int(task):03d}", f"{manifest_sha}.onnx")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def payment_proof_for_path(task, path, manifest_path=MANIFEST, corpus=CORPUS):
    """Return payment proof only when candidate bytes exactly match a corpus artifact.

    Legacy manifest keys are eight hex characters, so comparing only the key would not be
    exact.  We hash both files in full; changing even one byte therefore revokes proof.
    """
    try:
        candidate_sha = sha256_file(path)
    except OSError:
        return None
    for manifest_sha, rec in load_manifest(manifest_path).items():
        try:
            if int(rec.get("task", -1)) != int(task):
                continue
        except (TypeError, ValueError):
            continue
        proven_path = corpus_artifact(task, manifest_sha, corpus)
        try:
            proven_sha = sha256_file(proven_path)
        except OSError:
            continue
        declared = rec.get("sha256")
        if declared and declared != proven_sha:
            continue  # corrupt/stale manifest entry cannot confer proof
        if candidate_sha == proven_sha:
            return {**rec, "manifest_sha": manifest_sha, "sha256": candidate_sha,
                    "path": proven_path}
    return None


def proven_for(task, limit=5):
    manifest = load_manifest()
    rows = [(sha, rec) for sha, rec in manifest.items() if int(rec.get("task", -1)) == int(task)]
    rows.sort(key=lambda item: (-float(item[1].get("pts", 0)), item[0]))
    return [{"sha8": sha, **rec, "path": os.path.join(CORPUS, f"task{int(task):03d}", f"{sha}.onnx")}
            for sha, rec in rows[:limit]]


def render(task, limit=5):
    rows = proven_for(task, limit)
    if not rows:
        return "## Proven Kaggle corpus\n(no exact-reconciled paying design is available for this task)"
    lines = ["## Proven Kaggle corpus (exact per-member payment proof)",
             "These are safe same-task starting points. Do not widen this list with merely gate-passing artifacts."]
    if int(task) in WARNINGS:
        lines.append(f"- **{WARNINGS[int(task)]}**")
    for r in rows:
        proof = r.get("source", "proof source unavailable")
        lines.append(f"- `{r['sha8']}` → **{float(r.get('pts', 0)):.4f} pts**, proof: {proof}")
        lines.append(f"  iterate from: `{r['path']}`")
    lines.append("Worker rule: iterate FROM a proven design of this task; novel op families with no proven ancestor go to solo probe.")
    return "\n".join(lines)
