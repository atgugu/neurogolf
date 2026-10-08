#!/usr/bin/env python3
"""Promote one exact candidate SHA after a reconciled Kaggle solo probe.

The proof must be a completed probe with exactly one changed member, a surviving logged
zip whose member bytes equal the candidate, and actual-vs-predicted agreement.  Batch/local
gates, ancestry, novelty, and oracle analysis are intentionally not accepted as proof.
"""
import argparse
import hashlib
import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
from config import CLEAN
DEFAULT_CORPUS = os.path.join(CLEAN, "knowledge_proven_paying")
DEFAULT_LOG = os.path.join(CLEAN, "logs", "probe_log.jsonl")
DEFAULT_OUTCOMES = os.path.join(CLEAN, "logs", "probe_outcomes.json")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _probe_entry(probe_id, log_path):
    found = None
    try:
        with open(log_path, errors="replace") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                if entry.get("id") == probe_id:
                    found = entry
    except OSError as exc:
        raise ValueError(f"probe log unreadable: {exc}") from exc
    if found is None:
        raise ValueError(f"probe #{probe_id} is not in the probe log")
    return found


def reconcile_solo_payment(task, candidate_path, probe_id, log_path=DEFAULT_LOG,
                           outcomes_path=DEFAULT_OUTCOMES, tolerance=0.05):
    """Validate and return exact candidate/payment metadata, or raise ValueError."""
    entry = _probe_entry(probe_id, log_path)
    if not entry.get("ok"):
        raise ValueError("probe submission was not logged as successful")
    swaps = entry.get("swaps")
    if not isinstance(swaps, list) or len(swaps) != 1:
        raise ValueError("payment proof must come from a solo probe (exactly one swap)")
    swap = swaps[0]
    if int(swap.get("task", -1)) != int(task):
        raise ValueError(f"solo probe paid task{int(swap.get('task', -1)):03d}, not task{task:03d}")
    try:
        with open(outcomes_path, errors="replace") as f:
            outcomes = json.load(f)
        outcome = outcomes[str(probe_id)]
        actual = float(outcome["actual"])
        predicted = float(entry["predicted"])
        base_score = float(entry["base_score"])
        old_pts = float(swap["old"])
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("probe has not been reconciled to a completed Kaggle outcome") from exc
    gap = actual - predicted
    if abs(gap) > tolerance + 1e-9:
        raise ValueError(f"solo probe did not reconcile: actual-predicted gap {gap:+.4f} "
                         f"exceeds {tolerance:.4f}")
    observed_pts = old_pts + (actual - base_score)
    if observed_pts <= 0:
        raise ValueError(f"solo probe proves no positive Kaggle payment ({observed_pts:.4f})")

    try:
        with open(candidate_path, "rb") as f:
            candidate = f.read()
    except OSError as exc:
        raise ValueError(f"candidate unreadable: {exc}") from exc
    probe_zip = entry.get("zip")
    if not probe_zip or not os.path.exists(probe_zip):
        raise ValueError("logged solo-probe zip is missing; exact submitted bytes cannot be proven")
    member = f"task{task:03d}.onnx"
    try:
        with zipfile.ZipFile(probe_zip) as z:
            submitted = z.read(member)
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise ValueError(f"cannot recover {member} from logged probe zip") from exc
    candidate_sha = sha256_bytes(candidate)
    if candidate_sha != sha256_bytes(submitted):
        raise ValueError("candidate bytes differ from the individually probed member")
    return {"sha256": candidate_sha, "sha8": candidate_sha[:8],
            "pts": round(observed_pts, 4), "gap": round(gap, 4),
            "source": f"probe#{probe_id} solo gap{gap:+.2f}", "bytes": candidate}


def promote(task, candidate_path, probe_id, corpus=DEFAULT_CORPUS, log_path=DEFAULT_LOG,
            outcomes_path=DEFAULT_OUTCOMES, tolerance=0.05):
    proof = reconcile_solo_payment(task, candidate_path, probe_id, log_path,
                                   outcomes_path, tolerance)
    manifest_path = os.path.join(corpus, "manifest.json")
    try:
        with open(manifest_path, errors="replace") as f:
            manifest = json.load(f)
    except FileNotFoundError:
        manifest = {}
    except ValueError as exc:
        raise ValueError("proven manifest is malformed; refusing to overwrite it") from exc
    key = proof["sha8"]
    task_dir = os.path.join(corpus, f"task{task:03d}")
    target = os.path.join(task_dir, f"{key}.onnx")
    if key in manifest:
        try:
            with open(target, "rb") as f:
                existing = f.read()
        except OSError as exc:
            raise ValueError("manifest SHA exists but its corpus artifact is missing") from exc
        if sha256_bytes(existing) != proof["sha256"]:
            raise ValueError(f"sha8 collision at {key}; refusing ambiguous promotion")
    os.makedirs(task_dir, exist_ok=True)
    tmp_model = target + ".tmp"
    with open(tmp_model, "wb") as f:
        f.write(proof["bytes"])
    os.replace(tmp_model, target)
    manifest[key] = {"task": int(task), "pts": proof["pts"],
                     "source": proof["source"], "sha256": proof["sha256"]}
    tmp_manifest = manifest_path + ".tmp"
    with open(tmp_manifest, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp_manifest, manifest_path)
    return key, target, manifest[key]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task", type=int)
    ap.add_argument("candidate")
    ap.add_argument("--probe-id", type=int, required=True)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--probe-log", default=DEFAULT_LOG)
    ap.add_argument("--outcomes", default=DEFAULT_OUTCOMES)
    ap.add_argument("--tolerance", type=float, default=0.05)
    args = ap.parse_args()
    try:
        key, target, rec = promote(args.task, args.candidate, args.probe_id,
                                   args.corpus, args.probe_log, args.outcomes,
                                   args.tolerance)
    except ValueError as exc:
        sys.exit(f"REFUSE: {exc}")
    print(f"PROMOTED task{args.task:03d} {key} → {target} ({rec['source']})")


if __name__ == "__main__":
    main()
