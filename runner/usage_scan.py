#!/usr/bin/env python3
"""Passive codex quota/cache scanner (usage_optimization, 2026-07-11).

Reads the codex CLI's own session rollout files (~/.codex/sessions/**/rollout-*.jsonl) —
each carries per-turn `token_count` events with `cached_input_tokens` AND a live
`rate_limits` snapshot (used_percent for the 5h/weekly windows). That closes two gaps
with ZERO orchestrator coupling: per-attempt cache-hit instrumentation (the fleet never
passes --json), and the "no CLI exposes a proactive quota query" gap (orchestrate.py
_bench_secs) — read-only, after the fact.

Outputs:
  results/USAGE.jsonl — one row per finished session (append; idempotent via state cursor)
  results/USAGE.md    — operator scoreboard: cache-hit % per model×role, quota %
State:
  results/usage_scan_state.json — per-file mtimes already ingested

A session still being written (mtime younger than --settle seconds) is skipped this run
and picked up next run, so rows always reflect FINAL usage.
"""
import argparse
import glob
import json
import os
import re
import statistics
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RESULTS = os.path.join(ROOT, "results")
STATE = os.path.join(RESULTS, "usage_scan_state.json")
USAGE_JSONL = os.path.join(RESULTS, "USAGE.jsonl")
USAGE_MD = os.path.join(RESULTS, "USAGE.md")

# builder marker: new tail form "…(task 46). Attempt #2." on one line; legacy head form had
# an escaped newline between them ("task 46).\nAttempt #13" inside the raw JSON string)
BUILD_RE = re.compile(r"BUILD THIS TASK NOW \(task (\d+)\)\.(?:\s|\\n)*Attempt #(\d+)")
DECIDE_RE = re.compile(r"(?:DECIDE FOR|You are the strategy DECIDER for) NeuroGolf task (\d+)\."
                       r"(?:\s|\\n)*Attempt (\d+)")
CWD_TASK_RE = re.compile(r"packs/task(\d+)")


def parse_session(path):
    """One pass over a rollout file → a summary row (or None if unparseable)."""
    row = {"file": os.path.basename(path), "session_id": None, "ts": None, "cwd": None,
           "model": None, "effort": None, "role": "other", "task": None, "attempt": None}
    last_usage = last_limits = None
    with open(path, errors="replace") as f:
        for line in f:
            if row["session_id"] is None and '"session_meta"' in line:
                try:
                    p = json.loads(line).get("payload", {})
                    row["session_id"] = p.get("session_id") or p.get("id")
                    row["ts"] = p.get("timestamp")
                    row["cwd"] = p.get("cwd")
                except ValueError:
                    pass
            if row["model"] is None and '"turn_context"' in line:
                try:
                    p = json.loads(line).get("payload", {})
                    row["model"] = p.get("model")
                    row["effort"] = p.get("effort")
                except ValueError:
                    pass
            if row["task"] is None:
                m = BUILD_RE.search(line)
                if m:
                    row["role"], row["task"], row["attempt"] = \
                        "build", int(m.group(1)), int(m.group(2))
                else:
                    m = DECIDE_RE.search(line)
                    if m:
                        row["role"], row["task"], row["attempt"] = \
                            "decide", int(m.group(1)), int(m.group(2))
            if '"token_count"' in line:
                try:
                    p = json.loads(line).get("payload", {})
                    if p.get("type") == "token_count":
                        info = p.get("info") or {}
                        if info.get("total_token_usage"):
                            last_usage = info["total_token_usage"]
                        if p.get("rate_limits"):
                            last_limits = p["rate_limits"]
                except ValueError:
                    pass
    if row["task"] is None and row["cwd"]:
        m = CWD_TASK_RE.search(row["cwd"])
        if m:  # builders run with cwd=packs/taskNNN; deciders inherit the orchestrator cwd
            row["task"] = int(m.group(1))
            if row["role"] == "other":
                row["role"] = "build"
    if not last_usage:
        return None
    inp = last_usage.get("input_tokens", 0)
    cached = last_usage.get("cached_input_tokens", 0)
    row.update({
        "input_tokens": inp, "cached_input_tokens": cached,
        "output_tokens": last_usage.get("output_tokens", 0),
        "reasoning_output_tokens": last_usage.get("reasoning_output_tokens", 0),
        "cache_hit_pct": round(100.0 * cached / inp, 1) if inp else None,
        # cache reads bill ~10% of input price: this is the price-weighted input volume
        "effective_input_units": int((inp - cached) + 0.1 * cached),
    })
    if last_limits:
        for k in ("primary", "secondary"):
            w = last_limits.get(k) or {}
            row[f"quota_{k}_pct"] = w.get("used_percent")
            row[f"quota_{k}_resets_at"] = w.get("resets_at")
        row["plan_type"] = last_limits.get("plan_type")
    return row


def scoreboard(rows, live_skipped):
    groups = {}
    for r in rows:
        groups.setdefault((r.get("model") or "?", r.get("role") or "?"), []).append(r)
    lines = ["# Codex usage / prompt-cache scoreboard (passive scan of ~/.codex/sessions)",
             "", f"sessions ingested: {len(rows)}"
             + (f" · {live_skipped} still-running session(s) deferred to the next scan"
                if live_skipped else "")]
    latest = max((r for r in rows if r.get("quota_primary_pct") is not None),
                 key=lambda r: r.get("ts") or "", default=None)
    if latest:
        lines += ["", f"**Quota (from newest session, {latest.get('ts')}):** "
                  f"5h window {latest['quota_primary_pct']}% used · "
                  f"weekly {latest.get('quota_secondary_pct')}% used · "
                  f"plan={latest.get('plan_type')}"]
    lines += ["", "| model | role | N | med cache-hit % | mean cache-hit % | Σ input | "
              "Σ cached | Σ output | Σ effective input |",
              "|---|---|---|---|---|---|---|---|---|"]
    for (model, role), rs in sorted(groups.items()):
        hits = [r["cache_hit_pct"] for r in rs if r.get("cache_hit_pct") is not None]
        lines.append(
            f"| {model} | {role} | {len(rs)} | "
            f"{statistics.median(hits):.1f} | {statistics.mean(hits):.1f} | "
            f"{sum(r['input_tokens'] for r in rs):,} | "
            f"{sum(r['cached_input_tokens'] for r in rs):,} | "
            f"{sum(r['output_tokens'] for r in rs):,} | "
            f"{sum(r['effective_input_units'] for r in rs):,} |"
            if hits else
            f"| {model} | {role} | {len(rs)} | – | – | "
            f"{sum(r['input_tokens'] for r in rs):,} | "
            f"{sum(r['cached_input_tokens'] for r in rs):,} | "
            f"{sum(r['output_tokens'] for r in rs):,} | "
            f"{sum(r['effective_input_units'] for r in rs):,} |")
    recent = sorted((r for r in rows if r.get("cache_hit_pct") is not None),
                    key=lambda r: r.get("ts") or "")[-50:]
    if recent:
        lines += ["", f"**cache-hit % over the last {len(recent)} sessions:** "
                  f"median {statistics.median(r['cache_hit_pct'] for r in recent):.1f}, "
                  f"mean {statistics.mean(r['cache_hit_pct'] for r in recent):.1f}"]
    retry = [r for r in rows if r.get("role") == "build" and (r.get("attempt") or 1) >= 2
             and r.get("cache_hit_pct") is not None]
    first = [r for r in rows if r.get("role") == "build" and r.get("attempt") == 1
             and r.get("cache_hit_pct") is not None]
    if retry and first:
        lines += ["", f"**retry (att≥2) vs first-attempt cache-hit %:** "
                  f"retries median {statistics.median(r['cache_hit_pct'] for r in retry):.1f} "
                  f"(N={len(retry)}) vs att1 median "
                  f"{statistics.median(r['cache_hit_pct'] for r in first):.1f} (N={len(first)})"
                  " — the HOT_RETRY / prompt-reorder levers act on this gap"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions-dir", default=os.path.expanduser("~/.codex/sessions"))
    ap.add_argument("--settle", type=int, default=120,
                    help="skip files modified fewer than this many seconds ago (still live)")
    ap.add_argument("--rescan", action="store_true", help="ignore the state cursor")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    seen = {}
    if not a.rescan and os.path.exists(STATE):
        try:
            seen = json.load(open(STATE)).get("seen", {})
        except ValueError:
            seen = {}
    new_rows, live_skipped = [], 0
    for path in sorted(glob.glob(os.path.join(a.sessions_dir, "**", "rollout-*.jsonl"),
                                 recursive=True)):
        name = os.path.basename(path)
        try:
            mt = os.path.getmtime(path)
        except OSError:
            continue
        if seen.get(name) == mt:
            continue
        if time.time() - mt < a.settle:
            live_skipped += 1
            continue
        row = parse_session(path)
        seen[name] = mt
        if row:
            new_rows.append(row)

    os.makedirs(RESULTS, exist_ok=True)
    if new_rows:
        with open(USAGE_JSONL, "a") as f:
            for r in new_rows:
                f.write(json.dumps(r) + "\n")
    if a.rescan and os.path.exists(USAGE_JSONL):
        # a --rescan re-parses everything; dedupe the jsonl by session file, newest wins
        rows = {}
        for line in open(USAGE_JSONL, errors="replace"):
            try:
                r = json.loads(line)
                rows[r.get("file")] = r
            except ValueError:
                continue
        all_rows = list(rows.values())
        with open(USAGE_JSONL, "w") as f:
            for r in all_rows:
                f.write(json.dumps(r) + "\n")
    else:
        all_rows = []
        if os.path.exists(USAGE_JSONL):
            for line in open(USAGE_JSONL, errors="replace"):
                try:
                    all_rows.append(json.loads(line))
                except ValueError:
                    continue
    tmp = STATE + ".tmp"
    json.dump({"seen": seen}, open(tmp, "w"))
    os.replace(tmp, STATE)
    if all_rows:
        md = scoreboard(all_rows, live_skipped)
        open(USAGE_MD, "w").write(md)
        if not a.quiet:
            print(md)
    if not a.quiet:
        print(f"ingested {len(new_rows)} new session(s); "
              f"{live_skipped} live session(s) deferred; wrote {USAGE_MD}")


if __name__ == "__main__":
    main()
