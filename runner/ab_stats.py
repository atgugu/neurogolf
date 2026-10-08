#!/usr/bin/env python3
"""ab_stats — the wave-5 A/B scoreboard (gpt-5.5/xhigh flat vs the effort-floored 5.6 ladder).

Reads every results/task*/ATTEMPTS.jsonl row with ts >= --since, drops infra deaths,
groups by arm and by model_id×effort, and writes results/AB_STATS.md:
N · DONE-rate with Wilson 90% CI · mean banked Δ/attempt · median secs (burn proxy).
Weak-screen rows are reported separately (neither arm).

  python3 runner/ab_stats.py [--since 20260711T120000] [--quiet]

Decision rule (operator, plan 2026-07-11): first checkpoint at N>=30 real attempts/arm;
if the 90% CIs separate, flip GPT55_SHARE to 0.7 toward the winner at the next restart;
kill an arm only at N>=40 and WR<5%.
"""
import argparse
import glob
import json
import math
import os
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RESULTS = os.path.join(ROOT, "results")

WIN = {"DONE", "DONE_SALVAGED"}
INFRA_RE = re.compile(r"Settings fetch failed|Couldn't set model|stream disconnected|"
                      r"error sending request|failed to refresh available models|"
                      r"Reconnecting\.\.\. \d+/\d+|Selected model is at capacity")


def arm_of(rec):
    mid = rec.get("model_id") or ""
    if not mid.startswith("gpt"):
        return "other"
    if mid == "gpt-5.5":
        return "B: gpt-5.5/xhigh"
    if mid.startswith("gpt-5.6"):
        return "A: 5.6 ladder"
    return "other"


def wilson(w, n, z=1.6449):  # 90%
    if not n:
        return (0.0, 0.0)
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="20260711T120000")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    rows = []
    for p in glob.glob(os.path.join(RESULTS, "task*", "ATTEMPTS.jsonl")):
        try:
            for line in open(p, errors="replace"):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(r, dict):
                    continue  # valid JSON that isn't an attempt record
                if str(r.get("ts") or "") < a.since:
                    continue
                if INFRA_RE.search(str(r.get("fail_reason") or "")):
                    continue
                # coerce numeric fields defensively — one bad row must not kill the report
                try:
                    r["secs"] = int(r.get("secs") or 0)
                except (TypeError, ValueError):
                    r["secs"] = 0
                try:
                    r["delta"] = float(r.get("delta")) if r.get("delta") is not None else None
                except (TypeError, ValueError):
                    r["delta"] = None
                rows.append(r)
        except OSError:
            continue

    groups = {}
    for r in rows:
        for key in (("ARM", arm_of(r)),
                    ("MODEL", f"{r.get('model_id') or r.get('model')}/{r.get('effort')}")):
            groups.setdefault(key, []).append(r)

    def fmt(recs):
        n = len(recs)
        w = sum(1 for r in recs if r.get("outcome") in WIN)
        lo, hi = wilson(w, n)
        deltas = [r.get("delta") or 0.0 for r in recs if r.get("outcome") in WIN]
        secs = sorted(r.get("secs") or 0 for r in recs)
        med = secs[len(secs) // 2] if secs else 0
        return (f"| {n:4d} | {w:3d} | {100*w/max(1,n):5.1f}% | "
                f"[{100*lo:4.1f}%, {100*hi:4.1f}%] | "
                f"{sum(deltas):+.3f} | {med:5d}s |")

    lines = [f"# A/B scoreboard — attempts since {a.since} (infra deaths excluded)",
             "",
             "## Arms",
             "| arm | N | DONE | WR | Wilson 90% CI | Σ banked Δ | med secs |",
             "|---|---|---|---|---|---|---|"]
    for (kind, name), recs in sorted(groups.items()):
        if kind == "ARM":
            lines.append(f"| {name} " + fmt(recs))
    lines += ["", "## By model×effort",
              "| config | N | DONE | WR | Wilson 90% CI | Σ banked Δ | med secs |",
              "|---|---|---|---|---|---|---|"]
    for (kind, name), recs in sorted(groups.items()):
        if kind == "MODEL":
            lines.append(f"| {name} " + fmt(recs))
    arm_a = groups.get(("ARM", "A: 5.6 ladder"), [])
    arm_b = groups.get(("ARM", "B: gpt-5.5/xhigh"), [])
    # NOTE: CI non-overlap is a CONSERVATIVE screen, not a calibrated two-sample test, and
    # repeated attempts on one task aren't fully independent — treat "decisive" as an
    # operator hint to rebalance shares, never as proof.
    verdict = "insufficient data (need N>=30/arm)"
    if len(arm_a) >= 30 and len(arm_b) >= 30:
        wa = sum(1 for r in arm_a if r.get("outcome") in WIN) / len(arm_a)
        wb = sum(1 for r in arm_b if r.get("outcome") in WIN) / len(arm_b)
        la, ha = wilson(sum(1 for r in arm_a if r.get("outcome") in WIN), len(arm_a))
        lb, hb = wilson(sum(1 for r in arm_b if r.get("outcome") in WIN), len(arm_b))
        if la > hb:
            verdict = f"ARM A leads decisively (CI-separated): consider GPT55_SHARE=0.3"
        elif lb > ha:
            verdict = f"ARM B leads decisively (CI-separated): consider GPT55_SHARE=0.7"
        else:
            verdict = f"no CI separation yet (A {wa:.1%} vs B {wb:.1%}) — hold 50/50"
    lines += ["", f"**Checkpoint verdict:** {verdict}", ""]

    out = os.path.join(RESULTS, "AB_STATS.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    if not a.quiet:
        print("\n".join(lines))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
