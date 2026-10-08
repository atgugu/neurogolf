#!/usr/bin/env python3
"""Read-only, live expected-yield estimates for scheduler ordering.

This module deliberately never rewrites QUEUE.csv, DIFFICULTY.csv, packs, or pin data.
The signed/static difficulty table supplies structural priors; current clean evidence and
observed attempt durations turn those priors into expected banked points per seat-hour.
"""
import csv
import json
import math
import os
import statistics
import time

from evidence import load_era, record_is_current
from task_difficulty import is_infra_attempt


PWIN = {"EASY": 0.14, "MEDIUM": 0.12, "HARD": 0.06, "VHARD": 0.025}
DEFAULT_SECONDS = {
    "codex": {"EASY": 3600, "MEDIUM": 4800, "HARD": 6300, "VHARD": 7800},
}
CACHE_SECONDS = max(30, int(os.environ.get("RUNTIME_ROI_CACHE_SECS", "120")))
_CACHE = {}


def _number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _static_rows(results):
    path = os.path.join(results, "DIFFICULTY.csv")
    try:
        signature = (os.path.getmtime(path), os.path.getsize(path))
    except OSError:
        signature = None
    key = ("static", path)
    cached = _CACHE.get(key)
    if cached and cached[0] == signature:
        return cached[1]
    rows = {}
    try:
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                task = str(row.get("task", ""))
                if task.startswith("task"):
                    rows[task[4:].zfill(3)] = row
    except OSError:
        pass
    _CACHE[key] = (signature, rows)
    return rows


def _provider_for_attempt(attempt):
    provider = str(attempt.get("model") or "").lower()
    if provider not in DEFAULT_SECONDS:
        return None
    return provider


def _duration_samples(results):
    """Return current-era duration samples, cached briefly to keep scheduling cheap."""
    key = ("durations", results)
    now = time.time()
    cached = _CACHE.get(key)
    if cached and now - cached[0] < CACHE_SECONDS:
        return cached[1]
    static = _static_rows(results)
    era = load_era()
    samples = {}
    try:
        entries = list(os.scandir(results))
    except OSError:
        entries = []
    for entry in entries:
        if not entry.is_dir() or not entry.name.startswith("task"):
            continue
        task = entry.name[4:].zfill(3)
        srow = static.get(task, {})
        tier = str(srow.get("tier") or "MEDIUM")
        if tier not in PWIN:
            tier = "VHARD"
        family = str(srow.get("fam") or "?")
        path = os.path.join(entry.path, "ATTEMPTS.jsonl")
        try:
            with open(path, errors="replace") as handle:
                records = handle.readlines()
        except OSError:
            continue
        for line in records:
            try:
                attempt = json.loads(line)
            except (TypeError, ValueError):
                continue
            provider = _provider_for_attempt(attempt)
            seconds = _number(attempt.get("secs"))
            if (not provider or not record_is_current(attempt, era)
                    or is_infra_attempt(attempt) or seconds < 60 or seconds > 18000):
                continue
            samples.setdefault((provider, tier, family), []).append(seconds)
            samples.setdefault((provider, tier, None), []).append(seconds)
            samples.setdefault((provider, None, None), []).append(seconds)
    _CACHE[key] = (now, samples)
    return samples


def expected_seconds(results, provider, tier, family="?"):
    """Estimate occupied builder seconds with conservative, shrinkage-safe fallbacks."""
    provider = provider if provider in DEFAULT_SECONDS else "codex"
    tier = tier if tier in PWIN else "VHARD"
    samples = _duration_samples(results)
    choices = [
        ((provider, tier, family), 5, "provider+tier+family"),
        ((provider, tier, None), 8, "provider+tier"),
        ((provider, None, None), 15, "provider"),
    ]
    for key, minimum, source in choices:
        values = samples.get(key, [])
        if len(values) >= minimum:
            return float(statistics.median(values)), source, len(values)
    return float(DEFAULT_SECONDS[provider][tier]), "conservative-prior", 0


def estimate(task, row, provider, results, difficulty):
    """Estimate next-attempt points and points/hour from current, read-only evidence."""
    task_num = f"{int(task):03d}"
    static = _static_rows(results).get(task_num, {})
    signals = difficulty.get("signals") or {}
    tier = str(difficulty.get("tier") or static.get("tier") or "MEDIUM")
    if tier not in PWIN:
        tier = "VHARD"
    attempts = int(_number(signals.get("clean_attempts")))
    nos = int(_number(signals.get("post_win_nos", signals.get("clean_nos"))))
    no_providers = int(_number(signals.get(
        "post_win_no_providers", signals.get("failed_providers"))))
    banked = max(0.0, _number(signals.get("banked_delta")))
    proven = int(_number(static.get("proven")))

    pin_cost = _number(row.get("pin_cost"))
    target_cost = _number(static.get("target_cost"))
    if pin_cost > 0 and target_cost > 0:
        remaining = max(0.0, math.log(max(1.0, pin_cost / target_cost)) - banked)
    else:
        # The static edelta already deducted the bank present when it was generated.
        remaining = max(0.0, _number(static.get("edelta"))
                        + _number(static.get("banked")) - banked)

    pwin = PWIN[tier] * (1.1 if proven >= 2 else 1.0)
    pwin *= max(0.3, 1 - 0.12 * nos)
    pwin *= max(0.3, 1 - 0.22 * max(0, attempts - 2))
    if banked > 0:
        pwin *= 0.6
    floored = nos >= 3 or (nos >= 2 and no_providers >= 2)
    if floored:
        # Endless mode still attempts every task; a floor is a strong penalty, not zero.
        pwin *= 0.25

    expected_points = remaining * pwin
    family = str(static.get("fam") or "?")
    seconds, runtime_source, runtime_samples = expected_seconds(
        results, provider, tier, family)
    points_per_hour = expected_points * 3600.0 / max(60.0, seconds)
    return {
        "task": "task" + task_num,
        "provider": provider,
        "tier": tier,
        "pwin": round(pwin, 6),
        "remaining_delta": round(remaining, 6),
        "expected_points": round(expected_points, 6),
        "expected_seconds": int(round(seconds)),
        "points_per_hour": round(points_per_hour, 6),
        "runtime_source": runtime_source,
        "runtime_samples": runtime_samples,
        "floored": floored,
    }


def clear_cache():
    """Test/operator hook; production refreshes automatically on its short TTL."""
    _CACHE.clear()
