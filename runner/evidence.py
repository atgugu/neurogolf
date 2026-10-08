#!/usr/bin/env python3
"""Post-scorer-fix evidence boundary shared by prompts and prioritization.

Historical files are deliberately left in place.  This module only controls which
records may steer a new worker/decider and which reset ledger may establish house money.
"""
import csv
import datetime
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ERA_FILE = os.path.join(ROOT, "data", "EVIDENCE_ERA.json")


def load_era(path=ERA_FILE):
    with open(path, errors="replace") as f:
        era = json.load(f)
    cutoff = str(era["cutoff_utc"])
    era["cutoff_ts"] = datetime.datetime.fromisoformat(
        cutoff.replace("Z", "+00:00")
    ).timestamp()
    return era


def _record_timestamp(value):
    if not value:
        return None
    value = str(value).strip()
    for fmt in ("%Y%m%dT%H%M%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.datetime.strptime(value, fmt).replace(
                tzinfo=datetime.timezone.utc
            ).timestamp()
        except ValueError:
            pass
    try:
        return datetime.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def record_is_current(record, era=None):
    """True only for timestamped records at/after the explicit clean-era cutoff."""
    era = era or load_era()
    ts = _record_timestamp(record.get("ts"))
    return ts is not None and ts >= era["cutoff_ts"]


def current_attempt_pairs(attempts, era=None):
    """Return (original one-based attempt number, record) for clean-era attempts."""
    era = era or load_era()
    return [(i, rec) for i, rec in enumerate(attempts, 1)
            if record_is_current(rec, era)]


def file_is_current(path, era=None):
    """Use file creation/update time only as an inclusion gate; never mutate old files."""
    era = era or load_era()
    try:
        return os.path.getmtime(path) >= era["cutoff_ts"]
    except OSError:
        return False


def clean_ledger_path(era=None):
    era = era or load_era()
    path = era.get("clean_ledger", "results/LEDGER.csv")
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def pending_ledger_rows(ledger_path=None, era=None):
    """Return only ledger rows not already absorbed/acknowledged on the current pin.

    ``AUTOSUBMIT_MARKER.json`` records a data-row cursor and the pin SHA at which that
    cursor was committed.  Rows before it are historical proof, not current house money.
    If no marker exists (legacy/test setup), preserve the old all-rows behavior.  A marker
    for a different pin establishes no current frontier: treating old deltas as pending is
    the more damaging error because it suppresses high-EV tasks and imposes the ≥2× rule.
    """
    path = ledger_path or clean_ledger_path(era)
    try:
        with open(path, errors="replace") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return []

    marker_path = os.path.join(os.path.dirname(path), "AUTOSUBMIT_MARKER.json")
    try:
        with open(marker_path, errors="replace") as f:
            marker = json.load(f)
    except (OSError, ValueError, TypeError):
        return rows

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    pin_path = os.path.join(repo_root, "data", "PIN_SHA.txt")
    try:
        current_pin = open(pin_path, errors="replace").read().strip()
    except OSError:
        current_pin = ""
    marker_pin = str(marker.get("pin_sha") or "").strip()
    if current_pin and marker_pin and marker_pin != current_pin:
        return []
    try:
        consumed = int(marker.get("ledger_lines", 0))
    except (TypeError, ValueError):
        return []
    if consumed < 0 or consumed > len(rows):
        return []
    return rows[consumed:]


def banked_delta(task, ledger_path=None, era=None):
    """Best still-pending delta on this pin, never absorbed rows or attempt history."""
    path = ledger_path or clean_ledger_path(era)
    wanted = f"task{int(task):03d}"
    best = None
    for row in pending_ledger_rows(path, era):
        if row.get("task") != wanted:
            continue
        try:
            value = float(row["delta"])
        except (KeyError, TypeError, ValueError):
            continue
        best = value if best is None else max(best, value)
    return best
