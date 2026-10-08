#!/usr/bin/env python3
"""Ramp a drained-start wave only after a healthy one-hour/three-attempt canary."""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(HERE))
from integrity import integrity_errors


def attempt_count():
    total = 0
    for path in glob.glob(str(RESULTS / "task*" / "ATTEMPTS.jsonl")):
        try:
            with open(path) as fh:
                total += sum(1 for line in fh if line.strip())
        except OSError:
            pass
    return total


def health(unit, pid_file, since, heartbeat_max_age):
    if (RESULTS / "STOP_WAVE").exists():
        return False, "STOP_WAVE requested"
    if (RESULTS / "REPIN_INCOMPLETE.json").exists():
        return False, "repin became incomplete"
    errs = integrity_errors()
    if errs:
        return False, "integrity drift: " + "; ".join(errs[:2])
    try:
        state = json.loads((RESULTS / "RUN_STATE.json").read_text())
        heartbeat = float(state.get("heartbeat_at", 0))
        # A detached launch inherits RUN_STATE from the preceding wave.  The
        # orchestrator can take a tick to publish its first new heartbeat, so
        # do not abort the canary merely because that inherited timestamp is
        # old.  It must become current within a bounded initialization window.
        if heartbeat < since:
            if time.time() - since < 300:
                return True, "orchestrator initializing"
            return False, "orchestrator did not publish a new heartbeat"
        # A normal scheduler cycle can synchronously reap several completed workers,
        # price salvage artifacts, and seed deciders before it reaches its next telemetry
        # write.  Three minutes was below that observed legitimate cycle time and could
        # abort a healthy ramp.  This remains well below the 15-minute supervisor/janitor
        # cadence while admitting those bounded work bursts.
        if time.time() - heartbeat > heartbeat_max_age:
            return False, "orchestrator heartbeat stale"
    except (OSError, ValueError, TypeError):
        return False, "RUN_STATE.json missing or unreadable"
    if unit:
        active = subprocess.run(
            ["systemctl", "--user", "is-active", "--quiet", unit]).returncode == 0
        if not active:
            return False, f"{unit} is not active"
        journal = subprocess.run(
            ["journalctl", "--user", "-u", unit, f"--since=@{int(since)}", "--no-pager"],
            capture_output=True, text=True, timeout=30)
        low = (journal.stdout + journal.stderr).lower()
        if any(term in low for term in ("out of memory", "oom-kill", "killed process")):
            return False, "OOM evidence in unit journal"
    else:
        try:
            pid = int(Path(pid_file).read_text().strip())
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
            if "run_wave.sh" not in cmd:
                return False, f"keepalive pid {pid} is not run_wave.sh"
        except (OSError, ValueError):
            return False, "detached keepalive is not active"
    return True, "healthy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit")
    ap.add_argument("--pid-file")
    ap.add_argument("--initial", type=int, default=6)
    ap.add_argument("--target", type=int, default=12)
    ap.add_argument("--seconds", type=int, default=3600)
    ap.add_argument("--min-attempts", type=int, default=3)
    ap.add_argument("--max-wait", type=int, default=21600)
    ap.add_argument("--heartbeat-max-age", type=int,
                    default=int(os.environ.get("CANARY_HEARTBEAT_MAX_AGE", "900")))
    a = ap.parse_args()
    if bool(a.unit) == bool(a.pid_file):
        ap.error("exactly one of --unit or --pid-file is required")
    started, baseline = time.time(), attempt_count()
    while time.time() - started <= a.max_wait:
        ok, reason = health(a.unit, a.pid_file, started, a.heartbeat_max_age)
        if not ok:
            print(f"CANARY ABORT: {reason}; fleet remains at {a.initial}", flush=True)
            return 1
        completed = attempt_count() - baseline
        if time.time() - started >= a.seconds and completed >= a.min_attempts:
            r = subprocess.run([sys.executable, str(HERE / "fleet.py"), "set",
                                f"codex={a.target}"])
            if r.returncode == 0:
                print(f"CANARY PASS: {completed} attempts, {int(time.time()-started)}s; "
                      f"ramped codex {a.initial}->{a.target}", flush=True)
            return r.returncode
        time.sleep(60)
    print(f"CANARY EXPIRED: fewer than {a.min_attempts} attempts in {a.max_wait}s; "
          f"fleet remains at {a.initial}", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
