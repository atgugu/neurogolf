#!/usr/bin/env python3
"""Clean-era task difficulty used by both builder and decider routing.

The static score in ``results/DIFFICULTY.csv`` captures task structure and transfer
evidence.  This module replaces its history component with live, post-scorer-fix evidence
so old NOs/floors cannot buy a stronger model.  The result is intentionally explainable:
the web console can show the same signals that caused a routing decision.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re

from evidence import (banked_delta, current_attempt_pairs, file_is_current, load_era,
                      record_is_current)


INFRA_RE = re.compile(
    r"Settings fetch failed|Couldn't set model|stream disconnected|error sending request|"
    r"failed to refresh available models|Reconnecting\.\.\. \d+/\d+|"
    r"Selected model is at capacity", re.I
)
TIER_LIMITS = ((28, "EASY"), (50, "MEDIUM"), (72, "HARD"))
_STATIC_CACHE = {}
_UNSET = object()


def _number(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def tier_for_score(score):
    for ceiling, tier in TIER_LIMITS:
        if score < ceiling:
            return tier
    return "VHARD"


def is_infra_attempt(attempt):
    reason = str(attempt.get("fail_reason") or "")
    if INFRA_RE.search(reason):
        return True
    return (attempt.get("outcome") == "NONE" and (attempt.get("secs") or 9999) < 400
            and reason.startswith(("POST-MORTEM", "RAW TAIL")))


WEAK_SCREEN_MODEL_IDS = {m.strip() for m in
                         os.environ.get("WEAK_MODELS", "").split(",") if m.strip()}


def is_weak_screen_attempt(attempt):
    """Weak first-pass screens are context, never task-hardness/floor evidence."""
    return (attempt.get("model_id") or "") in WEAK_SCREEN_MODEL_IDS


def current_real_attempts(results, task, era=None):
    """Return only fixed-scorer-era, non-infrastructure attempts for ``task``."""
    era = era or load_era()
    task = f"task{int(task):03d}"
    path = os.path.join(results, task, "ATTEMPTS.jsonl")
    attempts = []
    try:
        with open(path, errors="replace") as f:
            for line in f:
                try:
                    attempts.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return []
    return [attempt for _, attempt in current_attempt_pairs(attempts, era)
            if not is_infra_attempt(attempt)]


def current_strong_attempts(results, task, era=None):
    return [attempt for attempt in current_real_attempts(results, task, era)
            if not is_weak_screen_attempt(attempt)]


def _static_row(results, task):
    path = os.path.join(results, "DIFFICULTY.csv")
    wanted = f"task{int(task):03d}"
    try:
        stamp = os.path.getmtime(path)
        key = (path, stamp)
        rows = _STATIC_CACHE.get(key)
        if rows is None:
            with open(path, errors="replace") as f:
                rows = {row.get("task"): row for row in csv.DictReader(f)}
            _STATIC_CACHE.clear()
            _STATIC_CACHE[key] = rows
        return rows.get(wanted, {})
    except OSError:
        return {}


def _fallback_base(row):
    """A useful score even before rank_difficulty.py has produced its richer CSV."""
    pin_points = _number(row.get("pin_pts") or row.get("pin_points"), 17.5)
    if pin_points >= 18.5:
        score = 12
    elif pin_points >= 17.5:
        score = 25
    elif pin_points >= 16.5:
        score = 42
    elif pin_points >= 15.0:
        score = 58
    else:
        score = 68
    if row.get("lane") == "D":
        score += 8
    pin_cost = _number(row.get("pin_cost"))
    bar_cost = _number(row.get("bar_cost"))
    if pin_cost > 0 and bar_cost > 0:
        cut = 1 - bar_cost / pin_cost
        score += 3 if cut >= .35 else 1 if cut >= .20 else 0
    return score


def _failure_class(attempt):
    outcome = str(attempt.get("outcome") or "NONE").upper()
    reason = str(attempt.get("fail_reason") or "").lower()
    if outcome == "NO":
        return "floor/NO"
    if outcome == "TIMEOUT" or "watchdog" in reason or "timed out" in reason:
        return "stalled build"
    if "cost-reject" in reason or "does not close" in reason or "micro-trim" in reason:
        return "cost floor"
    if "wrong(" in reason or "fail wrong" in reason:
        return "semantic failure"
    if outcome.startswith("DONE"):
        return "banked win"
    return "build failure"


def _current_files(results, task, pattern, era):
    task_dir = os.path.join(results, f"task{int(task):03d}")
    return [path for path in glob.glob(os.path.join(task_dir, pattern))
            if file_is_current(path, era)]


def _decision_families(results, task, era):
    families = set()
    for path in _current_files(results, task, "DECISION_att*.md", era):
        if path.endswith("_input.md"):
            continue
        try:
            with open(path, errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        match = re.search(r"(?im)^FAMILY:\s*([^\n]+)", text)
        if match:
            # The first phrase is stable enough to identify repeated family choices while
            # avoiding tiny prose changes turning one family into many.
            families.add(re.split(r"[;:.]", match.group(1).strip().lower(), 1)[0][:80])
    return families


def _rule_signal(results, task, era):
    paths = _current_files(results, task, "POSTMORTEM_att*.md", era)
    if not paths:
        return None
    try:
        with open(max(paths, key=os.path.getmtime), errors="replace") as f:
            text = f.read()[:5000]
    except OSError:
        return None
    line = next((line for line in text.splitlines() if "RULE-STATUS" in line.upper()), "")
    low = line.lower()
    if re.search(r"\b(unverified|uncertain|partial|provisional|ambig|wrong)\b", low):
        return "uncertain"
    if re.search(r"\bverified|all passed|solid\b", low):
        return "verified"
    return None


def _has_closing_price(results, task, bar_cost, era):
    if not bar_cost:
        return False
    for path in _current_files(results, task, "PRICES_att*.md", era):
        try:
            with open(path, errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        if any(int(value) <= bar_cost for value in re.findall(r"(?m)^\s*(\d+)\s+B\s+", text)):
            return True
    return False


def _rejection_count(results, task, era):
    path = os.path.join(results, f"task{int(task):03d}", "REJECTIONS.jsonl")
    count = 0
    try:
        with open(path, errors="replace") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if record_is_current(record, era):
                    count += 1
    except OSError:
        pass
    return count


def profile(task, row, results, ledger_path=None, era=None, static_row=None,
            attempts=None, banked_value=_UNSET):
    """Return ``score``, ``tier``, and human-readable clean-evidence reasons.

    Scores are routing estimates, not claims that a task is intrinsically difficult.  They
    answer the operational question: how much reasoning should the *next* decision/build
    receive given what is known right now?
    """
    era = era or load_era()
    task_num = f"{int(task):03d}"
    static = static_row if static_row is not None else _static_row(results, task_num)
    if static:
        # rank_difficulty's history is a snapshot. Replace it rather than double-count it.
        score = _number(static.get("score")) - _number(static.get("c_history"))
    else:
        score = _fallback_base(row)
    reasons = []
    pin_points = _number(static.get("pin_pts") or row.get("pin_pts") or row.get("pin_points"), 0)
    pin_cost = _number(row.get("pin_cost"))
    bar_cost = int(_number(row.get("bar_cost")))
    if pin_points and pin_points < 17:
        reasons.append(f"low pin score {pin_points:.2f} implies structural headroom")
    if row.get("lane") == "D":
        reasons.append("curated Lane D task")
    if pin_cost and bar_cost:
        cut = max(0, 1 - bar_cost / pin_cost)
        if cut >= .20:
            reasons.append(f"target needs a {cut:.0%} cost cut")

    attempts = (current_strong_attempts(results, task_num, era) if attempts is None else
                [attempt for attempt in attempts
                 if not is_infra_attempt(attempt) and not is_weak_screen_attempt(attempt)])
    nos = [a for a in attempts if str(a.get("outcome")).upper() == "NO"]
    failed = [a for a in attempts if not str(a.get("outcome") or "").upper().startswith("DONE")]
    failed_models = {a.get("model") for a in failed if a.get("model")}
    failure_classes = {_failure_class(a) for a in failed}
    # A banked win refutes every earlier floor hypothesis. Keep the full history for the
    # routing-difficulty score, but expose post-win NO evidence separately to live EV/ROI.
    last_win = max((i for i, attempt in enumerate(attempts)
                    if str(attempt.get("outcome") or "").upper()
                    in ("DONE", "DONE_SALVAGED")), default=-1)
    post_win_attempts = attempts[last_win + 1:]
    post_win_nos = [a for a in post_win_attempts
                    if str(a.get("outcome") or "").upper() == "NO"]
    post_win_no_providers = {a.get("model") for a in post_win_nos if a.get("model")}
    history_score = min(30, len(attempts) * 2 + len(nos) * 5
                        + max(0, len(failed_models) - 1) * 3
                        + max(0, len(failure_classes) - 1) * 2)
    score += history_score
    if nos:
        reasons.append(f"{len(nos)} clean NO{'s' if len(nos) != 1 else ''} across "
                       f"{max(1, len(failed_models))} provider{'s' if len(failed_models) != 1 else ''}")
    if len(failure_classes) >= 2:
        reasons.append("multiple failure surfaces: " + ", ".join(sorted(failure_classes)))

    rejections = _rejection_count(results, task_num, era)
    # Virgin clean-era tasks cannot have a meaningful current plan/postmortem/price table.
    # Avoid three directory scans per idle task in the 4-second web-console refresh path.
    inspect_files = bool(attempts or rejections)
    families = _decision_families(results, task_num, era) if inspect_files else set()
    if len(families) >= 2:
        bump = min(6, (len(families) - 1) * 2)
        score += bump
        reasons.append(f"{len(families)} decision families already exercised")
    if rejections:
        score += min(9, rejections * 3)
        reasons.append(f"{rejections} clean rejected/non-closing decision budget"
                       + ("s" if rejections != 1 else ""))

    rule = _rule_signal(results, task_num, era) if inspect_files else None
    if rule == "verified":
        score -= 4
        reasons.append("rule verified; representation is the remaining problem")
    elif rule == "uncertain":
        score += 6
        reasons.append("rule remains uncertain")

    closing = _has_closing_price(results, task_num, bar_cost, era) if inspect_files else False
    if closing:
        score -= 8
        reasons.append("current harness-priced closing family is available")
    banked = (banked_delta(task_num, ledger_path, era)
              if banked_value is _UNSET else banked_value)
    if banked is not None and banked > 0:
        score += 4
        reasons.append(f"Δ+{banked:.2f} already banked; next gain is marginal optimization")

    donor = _number(static.get("donor"))
    proven = int(_number(static.get("proven")))
    if donor >= .7 or proven >= 2:
        reasons.append(f"transfer support: donor {donor:.2f}, {proven} proven design(s)")
    elif static and proven == 0:
        reasons.append("no exact same-task proven design")

    score = int(round(max(0, min(100, score))))
    tier = tier_for_score(score)
    return {
        "task": "task" + task_num,
        "score": score,
        "tier": tier,
        "reasons": reasons[:6],
        "signals": {
            "clean_attempts": len(attempts),
            "clean_nos": len(nos),
            "failed_providers": len(failed_models),
            "post_win_nos": len(post_win_nos),
            "post_win_no_providers": len(post_win_no_providers),
            "failure_modes": len(failure_classes),
            "families_tried": len(families),
            "decision_rejections": rejections,
            "rule": rule,
            "closing_price": closing,
            "banked_delta": banked,
        },
    }
