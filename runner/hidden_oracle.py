#!/usr/bin/env python3
"""Read-only access to the mined hidden-set exclusion oracle (2026-07-10).

Facts behind this module (Kaggle-proven):
- For a Kaggle-PAYING graph m, hidden ⊆ passes(m): every fresh generator draw m fails is
  CERTIFIED absent from the hidden set. Excluders are per-member-proven paying graphs that
  also pass ALL published splits locally (anchor filters t144-class runtime quirks).
- Fuzz-fail ≠ Kaggle-fail: members 036/105 failed 2000-draw fuzz yet PAID (hidden is a
  small fixed sample). A candidate failing ONLY certified-excluded draws is live-equivalent.
- Sampling-blind cases exist: task109's paying graphs exclude nothing in 6000 draws, yet a
  candidate that passed thousands of draws paid 0 — hidden contains curated/rare cases.
"""
import json
import os

from config import CLEAN
ORACLE_DIR = os.path.join(CLEAN, "candidates", "hidden_oracle")
SEED_SCHEME = "random.seed(104729*k+11); gen()   # k = draw index below"


def load(task):
    path = os.path.join(ORACLE_DIR, f"task{int(task):03d}.json")
    try:
        with open(path, errors="replace") as f:
            d = json.load(f)
        # only trust the 2026-07-10+ format (anchored excluders, reproducible seeds)
        if not isinstance(d, dict) or "excluded_union" not in d or "n_kept" not in d:
            return None
        d["_path"] = path
        return d
    except (OSError, ValueError):
        return None


def render(task):
    d = load(task)
    header = "## Hidden-set intel (exclusion oracle, Kaggle-proof-backed)"
    if d is None:
        return (f"{header}\n(not mined for this task yet — so treat fuzz verdicts carefully: "
                "fuzz-FAIL is NOT Kaggle-fail (036/105 paid after failing 2000-draw fuzz). "
                "A fuzz-fail alone must not kill an otherwise strong design; flag it for a "
                "solo/mega-batch probe instead.)")
    n, ex = d["n_kept"], len(d.get("excluded_union", []))
    rate = ex / max(1, n)
    lines = [header,
             f"- {ex} of {n} fresh draws are CERTIFIED ABSENT from hidden "
             f"({rate*100:.2f}%), proof: a Kaggle-paying anchored graph fails them.",
             f"- Excluded draw indices: `{d['_path']}` → key `excluded_union`; "
             f"reproduce any draw with `{SEED_SCHEME}`."]
    if rate >= 0.01:
        lines += [
            "- **S-CONTRACT BUILD ALLOWED**: your graph must be exact on published splits and "
            "on fresh draws EXCEPT the certified-excluded list. You may drop logic that only "
            "serves excluded draws — that behavior provably cannot be graded. This is the "
            "cheapest legitimate cost lever on this task.",
            "- Relaxed gate: a candidate whose only failures are certified-excluded draws is "
            "LIVE-EQUIVALENT — register it; do not 'fix' it back into a bigger graph."]
    elif rate > 0:
        lines += [
            "- Mild oracle: use as a RELAXED GATE only — a candidate failing ONLY the excluded "
            "draws is live-equivalent. Not enough exclusion mass for an S-contract build."]
    else:
        lines += [
            "- **0% exclusions**: the paying designs are fully general AND hidden may contain "
            "curated cases invisible to sampling (task109 precedent: passed 1000s of draws, "
            "paid 0). STAY WITHIN the proven design family above; a novel op family here must "
            "go to a solo probe, never straight into a mega-batch."]
    return "\n".join(lines)
