#!/usr/bin/env python3
"""Advisory hidden-set routing; this tool never rejects a candidate itself.

It implements the wave policy: proven same-task ancestry is required for normal batching;
novel op families, fuzz exemptions, and oracle-dominated designs route to a solo probe.
"""
import argparse, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
from config import CLEAN as _CLEAN
CLEAN = Path(_CLEAN)
CORPUS = CLEAN / "knowledge_proven_paying"
ORACLE = CLEAN / "candidates" / "hidden_oracle"
SATFAM = CLEAN / "candidates" / "satfam"


def proven_hashes(task):
    try:
        manifest = json.load(open(CORPUS / "manifest.json"))
    except (OSError, ValueError):
        return set()
    return {sha for sha, rec in manifest.items() if int(rec.get("task", -1)) == task}


def static_flags(model):
    import onnx
    try:
        matrix = json.load(open(CLEAN / "candidates" / "ort124_matrix.json"))["ops"]
    except (OSError, ValueError, KeyError):
        matrix = {}
    m = onnx.load(model)
    return sorted({n.op_type for n in m.graph.node if n.op_type not in matrix})


def route(task, model, provenance=None, fuzz_failed=False, gate_failed=False):
    flags = []
    novel = static_flags(model)
    if novel:
        flags.append("unmeasured-op:" + ",".join(novel))
    proven = proven_hashes(task)
    if not proven:
        flags.append("no-proven-same-task-ancestor")
    elif provenance and provenance not in proven:
        flags.append("not-identified-as-proven-ancestor")
    oracle_file = ORACLE / f"task{task:03d}.json"
    if fuzz_failed:
        if oracle_file.exists():
            flags.append("oracle-excused-fuzz-fail")
        else:
            sat = SATFAM / f"task{task:03d}.json"
            try:
                if json.load(open(sat)).get("meta", {}).get("growth_next_1000") == 0:
                    flags.append("saturated-family-fuzz-override")
                else:
                    return "NO", ["fuzz-fail-without-oracle-or-saturation"]
            except (OSError, ValueError):
                return "NO", ["fuzz-fail-without-oracle-or-saturation"]
    if gate_failed and not flags:
        return "NO", ["genuine-gate-regression"]
    return ("SOLO_PROBE" if flags else "BATCH"), flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", type=int)
    ap.add_argument("model")
    ap.add_argument("--proven-sha")
    ap.add_argument("--fuzz-failed", action="store_true")
    ap.add_argument("--gate-failed", action="store_true")
    a = ap.parse_args()
    verdict, reasons = route(a.task, a.model, a.proven_sha, a.fuzz_failed, a.gate_failed)
    print(f"ROUTE {verdict}" + (" — " + "; ".join(reasons) if reasons else ""))
    # Routing is advisory. Even NO is a recommendation for the caller, not a gate verdict.


if __name__ == "__main__":
    main()
