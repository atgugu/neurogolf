#!/usr/bin/env python3
"""Stage an exact Kaggle-proven same-task graph for future model calls.

This module is deliberately not called by make_pack.py.  The orchestrator invokes it at
spawn time, which lets code changes roll out on the next orchestrator process without
mutating packs used by agents that were already running when the change was installed.
"""
from __future__ import annotations

import collections
import json
import os
import re
import shutil

from proven_corpus import CORPUS, MANIFEST, load_manifest, sha256_file


REFERENCE_NAME = "kaggle_proven_best.onnx"
CONTEXT_NAME = "GRAPH_CONTEXT.md"


def _probe_number(source):
    match = re.search(r"(?:probe|submission)#?(\d+)", str(source or ""), re.I)
    return int(match.group(1)) if match else -1


def best_verified_proven(task, manifest_path=MANIFEST, corpus=CORPUS):
    """Highest-paying intact manifest artifact with exact full-byte identity.

    Legacy manifests use an eight-character SHA key.  Requiring the artifact's full SHA to
    begin with that key catches a corpus file changed after reconciliation; a declared full
    SHA, when present, is also mandatory.  A proof source is required so mere corpus
    presence cannot silently become Kaggle evidence.
    """
    candidates = []
    for manifest_sha, record in load_manifest(manifest_path).items():
        try:
            if int(record.get("task", -1)) != int(task):
                continue
            points = float(record["pts"])
        except (KeyError, TypeError, ValueError):
            continue
        source = str(record.get("source") or "").strip()
        if not source:
            continue
        path = os.path.join(corpus, f"task{int(task):03d}", f"{manifest_sha}.onnx")
        try:
            full_sha = sha256_file(path)
        except OSError:
            continue
        key = str(manifest_sha).lower()
        declared = str(record.get("sha256") or "").lower()
        if not full_sha.startswith(key) or (declared and declared != full_sha):
            continue
        candidates.append({**record, "manifest_sha": manifest_sha, "sha256": full_sha,
                           "path": path, "pts": points, "source": source})
    if not candidates:
        return None
    # Best Kaggle payment is the baseline; newest proof breaks exact-score ties.
    return max(candidates, key=lambda row: (row["pts"], _probe_number(row["source"]),
                                            row["sha256"]))


def _tensor_text(value):
    try:
        tensor = value.type.tensor_type
        dims = [dim.dim_value if dim.HasField("dim_value") else "?"
                for dim in tensor.shape.dim]
        return f"{value.name}: dtype={tensor.elem_type} shape={dims}"
    except Exception:
        return getattr(value, "name", "?")


def graph_summary(path, max_nodes=80):
    """Compact, non-authoritative disassembly so the graph is visible before tool use."""
    try:
        import onnx
        model = onnx.load(path, load_external_data=False)
        graph = model.graph
    except Exception as exc:
        return f"(ONNX disassembly unavailable: {type(exc).__name__}: {str(exc)[:120]})"
    ops = collections.Counter(node.op_type for node in graph.node)
    init_elements = 0
    for initializer in graph.initializer:
        n = 1
        for dim in initializer.dims:
            n *= int(dim)
        init_elements += n
    lines = [
        f"nodes={len(graph.node)} initializers={len(graph.initializer)} "
        f"initializer_elements={init_elements} file_bytes={os.path.getsize(path)}",
        "ops=" + ", ".join(f"{name}:{count}" for name, count in ops.most_common()),
        "inputs=" + "; ".join(_tensor_text(value) for value in graph.input),
        "outputs=" + "; ".join(_tensor_text(value) for value in graph.output),
        "node listing:",
    ]
    for index, node in enumerate(graph.node[:max_nodes]):
        lines.append(f"  {index:03d} {node.op_type} "
                     f"{','.join(node.input)} -> {','.join(node.output)}")
    if len(graph.node) > max_nodes:
        lines.append(f"  … {len(graph.node) - max_nodes} more nodes; inspect the staged ONNX")
    return "\n".join(lines)


def _atomic_text(path, text):
    tmp = path + f".tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def stage_kaggle_reference(task, pack_dir, manifest_path=MANIFEST, corpus=CORPUS):
    """Stage a read-only proven graph and return the context text shown to models."""
    refs = os.path.join(pack_dir, "refs")
    os.makedirs(refs, exist_ok=True)
    destination = os.path.join(refs, REFERENCE_NAME)
    context_path = os.path.join(pack_dir, CONTEXT_NAME)
    proven = best_verified_proven(task, manifest_path, corpus)
    if proven is None:
        archived = ""
        if os.path.exists(destination):
            quarantine = os.path.join(refs, "quarantine_unproven")
            os.makedirs(quarantine, exist_ok=True)
            try:
                old_sha = sha256_file(destination)
            except OSError:
                old_sha = "unreadable"
            archived_path = os.path.join(quarantine, f"{REFERENCE_NAME}.{old_sha[:12]}")
            serial = 1
            while os.path.exists(archived_path):
                archived_path = os.path.join(
                    quarantine, f"{REFERENCE_NAME}.{old_sha[:12]}.{serial}")
                serial += 1
            os.replace(destination, archived_path)
            archived = (f"\nA formerly staged reference lost proof and was archived at "
                        f"`{archived_path}`; it must not steer a model.\n")
        text = (f"# Trusted graph context — task{int(task):03d}\n\n"
                "Status: **NO EXACT PER-MEMBER KAGGLE-PROVEN GRAPH AVAILABLE**.\n\n"
                "Do not promote a merely local-gate-passing or batch-included graph to a "
                f"Kaggle correctness oracle.\n{archived}")
        _atomic_text(context_path, text)
        return text

    if not os.path.exists(destination) or sha256_file(destination) != proven["sha256"]:
        tmp = destination + f".tmp.{os.getpid()}"
        shutil.copy2(proven["path"], tmp)
        if sha256_file(tmp) != proven["sha256"]:
            os.remove(tmp)
            raise RuntimeError("staged Kaggle reference failed exact-SHA verification")
        os.chmod(tmp, 0o444)
        os.replace(tmp, destination)
    else:
        os.chmod(destination, 0o444)

    warning = ""
    try:
        from proven_corpus import WARNINGS
        warning = WARNINGS.get(int(task), "")
    except Exception:
        pass
    text = f"""# Trusted graph context — task{int(task):03d}

Status: **EXACT PER-MEMBER KAGGLE-PAYMENT PROOF**

- staged read-only graph: `./refs/{REFERENCE_NAME}`
- full SHA-256: `{proven['sha256']}`
- manifest key: `{proven['manifest_sha']}`
- Kaggle-paid points: **{proven['pts']:.4f}**
- reconciliation proof: {proven['source']}
- source corpus path: `{proven['path']}`
{('- task warning: **' + warning + '**') if warning else ''}

## How to use this reference

Inspect it before designing the build. It is the trusted correctness oracle and proven
behavioral baseline, but its architecture is **not** assumed optimal. Preserve the exact
reference; write experiments elsewhere. A novel family may beat it, but local gates cannot
upgrade novel bytes to Kaggle-proven status.

## Compact disassembly

```text
{graph_summary(destination)}
```
"""
    _atomic_text(context_path, text)
    return text


def read_staged_context(task, root, limit=7000):
    path = os.path.join(root, "packs", f"task{int(task):03d}", CONTEXT_NAME)
    try:
        with open(path, errors="replace") as f:
            return f.read()[:limit]
    except OSError:
        return "(trusted graph context has not been staged yet)"
