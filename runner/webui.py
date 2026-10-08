#!/usr/bin/env python3
"""NeuroGolf live operator console.

Pure-stdlib, read-only HTTP server.  The console consumes atomic RUN_STATE snapshots from
orchestrate.py and falls back to the legacy STATUS files for older runs.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
STATIC = os.path.join(HERE, "webui_static")

sys.path.insert(0, HERE)
from evidence import current_attempt_pairs, load_era  # noqa: E402
from integrity import integrity_errors  # noqa: E402
from task_difficulty import profile as difficulty_profile  # noqa: E402

PRICE_RE = re.compile(
    r"PRICE\s+task(?P<task>\d{3}):\s*cost=(?P<cost>\d+).*?"
    r"pts=(?P<pts>-?[\d.]+)\s+pin=(?P<pin>-?[\d.]+).*?"
    r"(?:Δ|delta)=(?P<delta>[+-]?[\d.]+)", re.I
)
STATUS_FIELDS = ("running", "queue", "done", "cooling", "analysts", "deciders",
                 "infra_requeues")
RUN_EVENT_RE = re.compile(r"^\[(\d{8}T\d{6})\]\s*(.*)$")
TASK_RE = re.compile(r"task(\d{3})")
DEFAULT_WORKER_SHARES = {"codex": 1.0}   # wave-6: builds codex-only (gpt-5.5/xhigh)
DEFAULT_DECIDER_SHARES = {"codex": 1.0}  # wave-6: deciders codex-only
CAPABILITY_BANDS = ((10, 38), (20, 46), (30, 49), (40, 52),
                    (50, 54), (60, 55), (75, 56), (101, 58))
# 2026-07-11 REVERT: all codex traffic back on flat gpt-5.5/xhigh (wave-5 A/B:
# 17.2% WR vs the 5.6 ladder's 6.9%). Keep in sync with orchestrate.GPT_CAPABILITY_ROUTES.
GPT_CAPABILITY_ROUTES = {
    38: ("gpt-5.5", "xhigh"), 46: ("gpt-5.5", "xhigh"),
    49: ("gpt-5.5", "xhigh"), 52: ("gpt-5.5", "xhigh"),
    54: ("gpt-5.5", "xhigh"), 55: ("gpt-5.5", "xhigh"),
    56: ("gpt-5.5", "xhigh"), 58: ("gpt-5.5", "xhigh"),
}


def required_capability(score: int, role: str = "builder") -> int:
    target = next(capability for ceiling, capability in CAPABILITY_BANDS if score < ceiling)
    return max(49, target) if role == "decider" else target


def decider_ladder(score: int) -> dict:
    """Human-facing model match: task difficulty → required capability → route."""
    target = required_capability(score, "decider")
    model, effort = GPT_CAPABILITY_ROUTES[target]
    return {"required_capability": target,
            "gpt": {"model": model, "effort": effort, "benchmark_score": target,
                    "capability_gap": 0}}


def read_tail(path: str, limit: int = 65536) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - limit))
            data = f.read()
        return data.decode("utf-8", "replace")
    except OSError:
        return ""


def utc_iso(value: float | None = None) -> str:
    return dt.datetime.fromtimestamp(value or time.time(), dt.timezone.utc).isoformat().replace(
        "+00:00", "Z")


def parse_stamp(value: str) -> float | None:
    try:
        return dt.datetime.strptime(value, "%Y%m%dT%H%M%S").replace(
            tzinfo=dt.timezone.utc).timestamp()
    except (TypeError, ValueError):
        return None


def task_number(value: str | int) -> str:
    m = re.search(r"(\d+)", str(value))
    return (m.group(1) if m else "0").zfill(3)[-3:]


def parse_legacy_status(text: str) -> dict:
    """Parse STATUS.txt without depending on spacing inside blocked=[...]."""
    first = text.strip().split(None, 1)[0] if text.strip() else ""
    state = {"heartbeat_at": parse_stamp(first), "counts": {}}
    for key in STATUS_FIELDS:
        m = re.search(rf"\b{key}=(\d+)", text)
        if m:
            state["counts"]["queued" if key == "queue" else key] = int(m.group(1))
    m = re.search(r"builds\[([^]]+)\]", text)
    state["spawn_counts"] = ({name: int(value) for name, value in
                              re.findall(r"(codex)=(\d+)", m.group(1))}
                             if m else {})
    m = re.search(r"ledger_sum=\+?(-?[\d.]+)", text)
    state["ledger_sum"] = float(m.group(1)) if m else 0.0
    m = re.search(r"blocked=\[([^]]*)\]", text)
    state["blocked_models"] = {
        name: None for name in re.findall(r"['\"]([^'\"]+)['\"]", m.group(1) if m else "")
    }
    m = re.search(r"\bstop=(\d+)", text)
    state["stop_requested"] = bool(int(m.group(1))) if m else False
    return state


def price_signals(text: str) -> list[dict]:
    """Extract priced candidates and qualify them with the next local gate verdict."""
    matches = list(PRICE_RE.finditer(text))
    out = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else min(len(text), match.end() + 900)
        after = text[match.end():end]
        if re.search(r"\b(?:ALL PASSED|PASS(?:ED)?(?: full)? gate|DONE task)\b", after, re.I):
            qualification = "passing"
        elif re.search(r"\bFAIL\s+wrong|wrong\(", after, re.I):
            qualification = "wrong"
        elif re.search(r"\b(?:UNSCORABLE|COST-REJECT|cost-rejected)\b", after, re.I):
            qualification = "rejected"
        else:
            qualification = "priced"
        d = match.groupdict()
        out.append({
            "task": "task" + d["task"],
            "cost": int(d["cost"]),
            "points": float(d["pts"]),
            "pin_points": float(d["pin"]),
            "delta": float(d["delta"]),
            "qualification": qualification,
            "source": "live-price",
        })
    return out


def model_provider(value: str | None) -> str | None:
    value = str(value or "").lower()
    if any(name in value for name in ("codex", "gpt", "terra", "sol", "luna")):
        return "codex"
    return None


def pid_is_orchestrator(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        with open(f"/proc/{int(pid)}/stat") as f:
            stat = f.read().split()
        if len(stat) > 2 and stat[2] == "Z":
            return False
        with open(f"/proc/{int(pid)}/cmdline", "rb") as f:
            cmd = f.read().replace(b"\0", b" ").decode("utf-8", "replace")
        return "orchestrate.py" in cmd
    except (OSError, ValueError):
        return False


class Monitor:
    def __init__(self, root: str = ROOT):
        self.root = os.path.abspath(root)
        self.results = os.path.join(self.root, "results")
        self._lock = threading.Lock()
        self._cache: tuple[float, dict] | None = None
        self._integrity_cache: tuple[float, list[str]] | None = None
        try:
            self.era = load_era(os.path.join(self.root, "data", "EVIDENCE_ERA.json"))
        except (OSError, ValueError, KeyError):
            self.era = {"era": "unknown", "cutoff_ts": 0, "cutoff_utc": "unknown"}

    def _json(self, path: str, default=None):
        try:
            with open(path, errors="replace") as f:
                return json.load(f)
        except (OSError, ValueError):
            return default

    def _csv(self, path: str) -> list[dict]:
        try:
            with open(path, errors="replace") as f:
                return list(csv.DictReader(f))
        except OSError:
            return []

    def _pid_for_run(self, run_id: str) -> int | None:
        name = "orchestrator.pid" if run_id == "primary" else f"orchestrator.{run_id}.pid"
        try:
            with open(os.path.join(self.results, name)) as f:
                return int(f.read().strip())
        except (OSError, ValueError):
            return None

    def runs(self) -> list[dict]:
        selected: dict[str, dict] = {}
        for path in glob.glob(os.path.join(self.results, "RUN_STATE*.json")):
            state = self._json(path)
            if not isinstance(state, dict) or state.get("schema") != 1:
                continue
            run_id = state.get("run_id") or "primary"
            state["source"] = "telemetry"
            state["path"] = os.path.relpath(path, self.root)
            selected[run_id] = state

        for path in glob.glob(os.path.join(self.results, "STATUS*.txt")):
            name = os.path.basename(path)
            run_id = "primary" if name == "STATUS.txt" else name[len("STATUS."):-len(".txt")]
            parsed = parse_legacy_status(read_tail(path, 4096))
            parsed.update({
                "schema": 0,
                "run_id": run_id,
                "source": "legacy",
                "path": os.path.relpath(path, self.root),
                "phase": "unknown",
                "pid": self._pid_for_run(run_id),
                "workers": [],
                "assistants": [],
                "queue_tasks": [],
                "done_tasks": [],
                "cooling": {},
                "decider_queue": [],
                "config": {},
            })
            old = selected.get(run_id)
            if not old or (parsed.get("heartbeat_at") or 0) > (old.get("heartbeat_at") or 0):
                selected[run_id] = parsed

        now = time.time()
        for run in selected.values():
            run["pid"] = run.get("pid") or self._pid_for_run(run["run_id"])
            run["alive"] = pid_is_orchestrator(run.get("pid"))
            run["heartbeat_age_s"] = max(0, int(now - (run.get("heartbeat_at") or 0)))
            if run.get("phase") == "stopped" or not run["alive"]:
                run["health"] = "offline"
            elif run["heartbeat_age_s"] > (45 if run.get("schema") == 1 else 100):
                run["health"] = "stale"
            elif run.get("blocked_models"):
                run["health"] = "degraded"
            else:
                run["health"] = "healthy"
            if run.get("started_at"):
                run["uptime_s"] = max(0, int(now - run["started_at"]))
        return sorted(selected.values(), key=lambda r: r.get("heartbeat_at") or 0, reverse=True)

    def _fallback_claim_workers(self, runs: list[dict]) -> list[dict]:
        alive_pids = {r.get("pid") for r in runs if r.get("alive")}
        workers = []
        for claim in glob.glob(os.path.join(self.root, "claims", "task*.claim")):
            try:
                with open(claim) as f:
                    bits = f.read().split()
                owner = int(bits[0])
            except (OSError, ValueError, IndexError):
                continue
            if owner not in alive_pids:
                continue
            task = "task" + task_number(os.path.basename(claim))
            logs = glob.glob(os.path.join(self.results, task, "attempts", "att*.log"))
            latest = max(logs, key=os.path.getmtime) if logs else None
            model = effort = None
            attempt = None
            if latest:
                m = re.search(r"att(\d+)_([a-z]+)-([a-z]+)_", os.path.basename(latest), re.I)
                if m:
                    attempt, model, effort = int(m.group(1)), m.group(2), m.group(3)
            workers.append({"task": task, "model": model or "unknown", "model_id": None,
                            "effort": effort, "attempt": attempt, "log": os.path.relpath(
                                latest, self.root) if latest else None, "run_id": "legacy",
                            "elapsed_s": int(time.time() - os.path.getmtime(claim)),
                            "timeout_s": None, "watch_strikes": [], "telemetry": "claim"})
        return workers

    def workers(self, runs: list[dict]) -> list[dict]:
        out = []
        for run in runs:
            if not run.get("alive") or run.get("schema") != 1:
                continue
            for raw in run.get("workers", []):
                worker = dict(raw)
                worker["run_id"] = run["run_id"]
                worker["telemetry"] = "snapshot"
                log = worker.get("log")
                signals = price_signals(read_tail(os.path.join(self.root, log), 160000)) if log else []
                worker["delta_signal"] = signals[-1] if signals else None
                worker["log_tail"] = self._meaningful_tail(
                    read_tail(os.path.join(self.root, log), 18000)) if log else ""
                out.append(worker)
        if not out:
            out = self._fallback_claim_workers(runs)
            for worker in out:
                log = worker.get("log")
                signals = price_signals(read_tail(os.path.join(self.root, log), 160000)) if log else []
                worker["delta_signal"] = signals[-1] if signals else None
                worker["log_tail"] = self._meaningful_tail(
                    read_tail(os.path.join(self.root, log), 18000)) if log else ""
        return out

    @staticmethod
    def _meaningful_tail(text: str) -> str:
        body = text.split("=== WORKER OUTPUT ===", 1)[-1]
        lines = [line.strip() for line in body.splitlines() if line.strip()]
        noise = re.compile(r"^(tokens used|hook:|diff --git|index [0-9a-f]+|--- a/|\+\+\+ b/)", re.I)
        lines = [line for line in lines if not noise.search(line)]
        return "\n".join(lines[-5:])[-900:]

    def _pin_rows(self) -> dict[str, dict]:
        return {row["task"]: row for row in self._csv(os.path.join(
            self.root, "data", "pin_pv2_70_263_pertask.csv")) if row.get("task")}

    def _queue_rows(self) -> list[dict]:
        rows = self._csv(os.path.join(self.root, "data", "lanes", "QUEUE.csv"))
        return sorted(rows, key=lambda r: int(r.get("priority") or 9999))

    def _difficulty(self) -> dict[str, dict]:
        return {row["task"]: row for row in self._csv(os.path.join(
            self.results, "DIFFICULTY.csv")) if row.get("task")}

    def _ledger(self) -> dict[str, dict]:
        best = {}
        for row in self._csv(os.path.join(self.results, "LEDGER.csv")):
            try:
                delta = float(row["delta"])
            except (KeyError, TypeError, ValueError):
                continue
            if row.get("task") not in best or delta > best[row["task"]]["delta"]:
                best[row["task"]] = {**row, "delta": delta}
        return best

    def _repriced(self) -> dict[str, dict]:
        best = {}
        paths = glob.glob(os.path.join(self.results, "REPRICE_AUDIT*.csv"))
        if not paths:
            return best
        latest = max(paths, key=os.path.getmtime)
        for row in self._csv(latest):
            try:
                delta = float(row["true_delta"])
            except (KeyError, TypeError, ValueError):
                continue
            task = row.get("task")
            if task and (task not in best or delta > best[task]["delta"]):
                best[task] = {"delta": delta, "points": self._float(row.get("true_pts")),
                              "cost": self._int(row.get("true_cost")), "file": row.get("file"),
                              "kind": row.get("kind"), "verdict": row.get("verdict"),
                              "source": "repriced"}
        return best

    @staticmethod
    def _float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _attempts(self, task: str) -> tuple[list[dict], int]:
        path = os.path.join(self.results, task, "ATTEMPTS.jsonl")
        all_attempts = []
        try:
            with open(path, errors="replace") as f:
                for line in f:
                    try:
                        all_attempts.append(json.loads(line))
                    except ValueError:
                        continue
        except OSError:
            pass
        return [dict(row, ordinal=i) for i, row in current_attempt_pairs(all_attempts, self.era)], len(all_attempts)

    def tasks(self, runs: list[dict], workers: list[dict]) -> list[dict]:
        pins, difficulty = self._pin_rows(), self._difficulty()
        ledger, repriced = self._ledger(), self._repriced()
        task_map = self._json(os.path.join(self.root, "data", "task_map.json"), {}) or {}
        worker_by = {w["task"]: w for w in workers}
        helper_by = {}
        queue_set, done_set, cooling, decider_queue = set(), set(), {}, set()
        for run in runs:
            if not run.get("alive"):
                continue
            queue_set.update(run.get("queue_tasks") or [])
            done_set.update(run.get("done_tasks") or [])
            cooling.update(run.get("cooling") or {})
            decider_queue.update(run.get("decider_queue") or [])
            for helper in run.get("assistants") or []:
                helper_by[helper.get("task")] = helper

        out = []
        for q in self._queue_rows():
            task = q["task"]
            pin = pins.get(task, {})
            diff = difficulty.get(task, {})
            attempts, historical_count = self._attempts(task)
            diff_profile = difficulty_profile(task_number(task), q, self.results,
                                              ledger_path=os.path.join(self.results, "LEDGER.csv"),
                                              era=self.era, static_row=diff,
                                              attempts=attempts,
                                              banked_value=(ledger.get(task) or {}).get("delta"))
            worker = worker_by.get(task)
            helper = helper_by.get(task)
            if worker:
                state = "building"
            elif helper:
                state = helper.get("stage")
            elif task in decider_queue:
                state = "decider-queued"
            elif task in cooling:
                state = "cooling"
            elif task in done_set:
                state = "done"
            elif task in queue_set:
                state = "queued"
            else:
                state = "idle"

            delta = None
            if task in ledger:
                delta = {"delta": ledger[task]["delta"], "source": "banked",
                         "qualification": "verified"}
            elif worker and worker.get("delta_signal"):
                delta = worker["delta_signal"]
            elif task in repriced:
                delta = repriced[task]
            latest = attempts[-1] if attempts else None
            out.append({
                "task": task,
                "number": task_number(task),
                "arc_id": task_map.get(task),
                "lane": q.get("lane"),
                "priority": self._int(q.get("priority")),
                "state": state,
                "holder": worker.get("model") if worker else ((helper or {}).get("model") or None),
                "model_id": worker.get("model_id") if worker else (helper or {}).get("model_id"),
                "effort": worker.get("effort") if worker else (helper or {}).get("effort"),
                "attempt": worker.get("attempt") if worker else (latest or {}).get("att"),
                "pin_points": self._float(pin.get("score") or q.get("pin_pts")),
                "pin_cost": self._int(pin.get("cost") or q.get("pin_cost")),
                "bar_cost": self._int(q.get("bar_cost")),
                "delta": delta,
                "clean_attempts": len(attempts),
                "historical_attempts": historical_count,
                "latest_outcome": (latest or {}).get("outcome"),
                "difficulty": diff_profile["tier"],
                "difficulty_score": diff_profile["score"],
                "difficulty_reasons": diff_profile["reasons"],
                "difficulty_signals": diff_profile["signals"],
                "decider_ladder": decider_ladder(diff_profile["score"]),
                "ev": self._float(diff.get("ev")),
                "family": diff.get("fam"),
                "cooldown_s": cooling.get(task),
            })
        return out

    def _integrity(self) -> list[str]:
        now = time.time()
        if self.root != ROOT:
            return []
        if not self._integrity_cache or now - self._integrity_cache[0] > 30:
            self._integrity_cache = (now, integrity_errors())
        return self._integrity_cache[1]

    def health(self, runs: list[dict]) -> tuple[list[dict], list[dict]]:
        live = [r for r in runs if r.get("alive")]
        integrity = self._integrity()
        stop_path = os.path.join(self.results, "STOP_WAVE")
        stop_marker = os.path.exists(stop_path)
        try:
            free = shutil.disk_usage(self.root).free
        except OSError:
            free = 0
        ledger_ok = bool(self._csv(os.path.join(self.results, "LEDGER.csv")) or
                         os.path.exists(os.path.join(self.results, "LEDGER.csv")))
        checks = [
            {"id": "orchestrator", "label": "Orchestrator", "status": "ok" if live else
             ("paused" if stop_marker else "error"), "detail": f"{len(live)} live run{'s' if len(live) != 1 else ''}"},
            {"id": "heartbeat", "label": "Heartbeat", "status": "ok" if live and all(
                r.get("health") != "stale" for r in live) else ("idle" if not live else "error"),
             "detail": "fresh" if live else "no live heartbeat"},
            {"id": "scorer", "label": "Scorer SHA", "status": "ok" if not integrity else "error",
             "detail": "locked" if not integrity else f"{len(integrity)} mismatch(es)"},
            {"id": "ledger", "label": "Clean ledger", "status": "ok" if ledger_ok else "error",
             "detail": self.era.get("era", "unknown")},
            {"id": "disk", "label": "Disk", "status": "ok" if free > 20 * 1024**3 else "warn",
             "detail": f"{free / 1024**3:.0f} GB free"},
            {"id": "stop", "label": "Stop marker", "status": "paused" if stop_marker else "ok",
             "detail": "present" if stop_marker else "clear"},
        ]
        attention = []
        if integrity:
            attention.append({"severity": "critical", "title": "Pricing authority drift",
                              "detail": integrity[0], "task": None})
        if stop_marker:
            attention.append({"severity": "info", "title": "Wave is operator-paused",
                              "detail": "results/STOP_WAVE is present; keepalive should not relaunch.", "task": None})
        elif not live:
            attention.append({"severity": "critical", "title": "No live orchestrator",
                              "detail": "Recorded run heartbeats are offline.", "task": None})
        stale = [r for r in live if r.get("health") == "stale"]
        if stale:
            attention.append({"severity": "critical", "title": "Heartbeat stalled",
                              "detail": ", ".join(f"{r['run_id']} {r['heartbeat_age_s']}s" for r in stale),
                              "task": None})
        for run in live:
            for model, seconds in (run.get("blocked_models") or {}).items():
                attention.append({"severity": "warn", "title": f"{model} is benched",
                                  "detail": f"{run['run_id']} · {seconds or 0}s until retry", "task": None})
            active = len(run.get("workers") or [])
            target = (run.get("config") or {}).get("workers")
            ready = (run.get("counts") or {}).get("queued", 0)
            if target and active < target and ready and not run.get("blocked_models"):
                attention.append({"severity": "warn", "title": "Worker slots underfilled",
                                  "detail": f"{run['run_id']} · {active}/{target} active with {ready} queued",
                                  "task": None})
            dec_active = (run.get("counts") or {}).get("deciders_active", 0)
            dec_wait = (run.get("counts") or {}).get("deciders_queued", 0)
            cap = (run.get("config") or {}).get("decider_cap")
            if cap and dec_active >= cap and dec_wait:
                attention.append({"severity": "warn", "title": "Decider backpressure",
                                  "detail": f"{run['run_id']} · {dec_active} active / {dec_wait} waiting",
                                  "task": None})
            for worker in run.get("workers") or []:
                gap = worker.get("capability_gap")
                if gap is not None and gap < 0:
                    attention.append({"severity": "warn", "title": "Capability below target",
                                      "detail": f"{worker['task']} · B{worker.get('benchmark_score')} "
                                                f"vs required B{worker.get('required_capability')}",
                                      "task": worker["task"]})
                strikes = worker.get("watch_strikes") or []
                if strikes:
                    attention.append({"severity": "warn", "title": "Watchdog attention",
                                      "detail": f"{worker['task']} · {' + '.join(strikes[-2:])}",
                                      "task": worker["task"]})
            for helper in run.get("assistants") or []:
                gap = helper.get("capability_gap")
                if helper.get("stage") == "decider" and gap is not None and gap < 0:
                    attention.append({"severity": "warn", "title": "Decider capability gap",
                                      "detail": f"{helper.get('task')} · B{helper.get('benchmark_score')} "
                                                f"vs required B{helper.get('required_capability')}",
                                      "task": helper.get("task")})
        return checks, attention[:12]

    def activity(self, limit: int = 36) -> list[dict]:
        events = []
        cutoff = self.era.get("cutoff_ts", 0)
        for path in glob.glob(os.path.join(self.results, "ORCHESTRATOR*.log")):
            for line in read_tail(path, 220000).splitlines():
                m = RUN_EVENT_RE.match(line)
                if not m:
                    continue
                ts = parse_stamp(m.group(1))
                if not ts or ts < cutoff:
                    continue
                message = m.group(2).strip()
                tm = TASK_RE.search(message)
                severity = "win" if "→ DONE" in message or "BANKED" in message else (
                    "warn" if any(x in message for x in ("ALERT", "EXCEPTION", "blocked", "REJECT")) else "info")
                events.append({"ts": ts, "time": utc_iso(ts), "message": message[:320],
                               "task": "task" + tm.group(1) if tm else None,
                               "severity": severity, "run": os.path.basename(path)})
        seen, unique = set(), []
        for event in sorted(events, key=lambda e: e["ts"], reverse=True):
            key = (event["ts"], event["message"])
            if key in seen:
                continue
            seen.add(key); unique.append(event)
        return unique[:limit]

    def snapshot(self, force: bool = False) -> dict:
        now = time.time()
        with self._lock:
            if not force and self._cache and now - self._cache[0] < 2:
                return self._cache[1]
            runs = self.runs()
            workers = self.workers(runs)
            tasks = self.tasks(runs, workers)
            checks, attention = self.health(runs)
            live_runs = [r for r in runs if r.get("alive")]
            worker_target = sum((r.get("config") or {}).get("workers", 0) for r in live_runs)
            assistants = [a for r in live_runs for a in (r.get("assistants") or [])]
            deciders = [a for a in assistants if a.get("stage") == "decider"]
            decider_wait = sum((r.get("counts") or {}).get("deciders_queued", 0) for r in live_runs)
            worker_shares = dict(DEFAULT_WORKER_SHARES)
            decider_shares = dict(DEFAULT_DECIDER_SHARES)
            if live_runs:
                config = live_runs[0].get("config") or {}
                worker_shares.update(config.get("worker_shares") or {})
                decider_shares.update(config.get("decider_shares") or {})
            worker_active = {provider: 0 for provider in worker_shares}
            for worker in workers:
                provider = worker.get("model") or model_provider(worker.get("model_id"))
                if provider in worker_active:
                    worker_active[provider] += 1
            decider_active = {provider: 0 for provider in decider_shares}
            for helper in deciders:
                provider = helper.get("provider") or model_provider(
                    helper.get("model_id") or helper.get("model"))
                if provider in decider_active:
                    decider_active[provider] += 1
            worker_spawns = {provider: sum((run.get("spawn_counts") or {}).get(provider, 0)
                                           for run in live_runs)
                             for provider in worker_shares}
            decider_spawns = {provider: sum(((run.get("allocation") or {}).get(
                "decider_spawn_counts") or {}).get(provider, 0) for run in live_runs)
                              for provider in decider_shares}
            pin_path = os.path.join(self.root, "data", "PIN_SHA.txt")
            try:
                with open(pin_path) as f:
                    pin_sha = f.read().strip()
            except OSError:
                pin_sha = "unknown"
            # Every orchestrator reports the same shared ledger. Read it once here: summing
            # per-run heartbeat values would multiply house money when side-by-side waves run.
            banked = sum(row.get("delta", 0) for row in self._ledger().values())
            payload = {
                "generated_at": utc_iso(now),
                "generated_ts": now,
                "era": {k: self.era.get(k) for k in ("era", "cutoff_utc")},
                "pin_sha": pin_sha,
                "status": "live" if live_runs else ("paused" if os.path.exists(
                    os.path.join(self.results, "STOP_WAVE")) else "offline"),
                "summary": {
                    "banked_delta": round(banked, 4),
                    "workers": len(workers),
                    "worker_target": worker_target,
                    "deciders": len(deciders),
                    "decider_wait": decider_wait,
                    "analysts": sum(1 for a in assistants if a.get("stage") == "analyst"),
                    "price_searches": sum(1 for a in assistants if a.get("stage") == "price-search"),
                    "queue": sum((r.get("counts") or {}).get("queued", 0) for r in live_runs),
                    "cooling": sum((r.get("counts") or {}).get("cooling", 0) for r in live_runs),
                    "live_runs": len(live_runs),
                },
                "allocation": {
                    "workers": {"shares": worker_shares, "active": worker_active,
                                "spawns": worker_spawns,
                                "bands": [
                                    {"range": "0–9", "target": 38, "gpt": "Luna / medium"},
                                    {"range": "10–19", "target": 46, "gpt": "Terra / medium"},
                                    {"range": "20–29", "target": 49, "gpt": "Terra / high"},
                                    {"range": "30–39", "target": 52, "gpt": "Terra / xhigh"},
                                    {"range": "40–49", "target": 54, "gpt": "Sol / medium"},
                                    {"range": "50–59", "target": 55, "gpt": "Terra / max"},
                                    {"range": "60–74", "target": 56, "gpt": "Sol / high"},
                                    {"range": "75–100", "target": 58, "gpt": "Sol / xhigh"},
                                ]},
                    "deciders": {"shares": decider_shares, "active": decider_active,
                                 "spawns": decider_spawns,
                                 "bands": [
                                     {"range": "0–29", "target": 49,
                                      "gpt": "Terra / high · B49"},
                                     {"range": "30–39", "target": 52,
                                      "gpt": "Terra / xhigh · B52"},
                                     {"range": "40–49", "target": 54,
                                      "gpt": "Sol / medium · B54"},
                                     {"range": "50–59", "target": 55,
                                      "gpt": "Terra / max · B55"},
                                     {"range": "60–74", "target": 56,
                                      "gpt": "Sol / high · B56"},
                                     {"range": "75–100", "target": 58,
                                      "gpt": "Sol / xhigh · B58"},
                                 ]},
                },
                "health": checks,
                "attention": attention,
                "runs": runs,
                "workers": workers,
                "tasks": tasks,
                "activity": self.activity(),
            }
            self._cache = (now, payload)
            return payload

    def task_detail(self, task_value: str) -> dict | None:
        task = "task" + task_number(task_value)
        snap = self.snapshot()
        summary = next((row for row in snap["tasks"] if row["task"] == task), None)
        if not summary:
            return None
        attempts, historical_count = self._attempts(task)
        task_dir = os.path.join(self.results, task)
        decisions = [p for p in glob.glob(os.path.join(task_dir, "DECISION_att*.md"))
                     if not p.endswith("_input.md") and os.path.getmtime(p) >= self.era.get("cutoff_ts", 0)]
        decision = max(decisions, key=os.path.getmtime) if decisions else None
        worker = next((w for w in snap["workers"] if w["task"] == task), None)
        logs = [p for p in glob.glob(os.path.join(task_dir, "attempts", "att*.log"))
                if os.path.getmtime(p) >= self.era.get("cutoff_ts", 0)]
        latest_log = max(logs, key=os.path.getmtime) if logs else None
        log_text = read_tail(latest_log, 50000) if latest_log else ""
        return {
            "summary": summary,
            "worker": worker,
            "attempts": attempts[-12:],
            "historical_attempts": historical_count,
            "decision": {
                "path": os.path.relpath(decision, self.root),
                "text": read_tail(decision, 18000)[:18000],
            } if decision else None,
            "latest_log": {
                "path": os.path.relpath(latest_log, self.root),
                "text": log_text.split("=== WORKER OUTPUT ===", 1)[-1][-18000:],
                "prices": price_signals(log_text),
            } if latest_log else None,
        }


class Handler(BaseHTTPRequestHandler):
    monitor: Monitor
    server_version = "NeuroGolfOps/1"

    def log_message(self, fmt, *args):
        sys.stderr.write("[webui] " + fmt % args + "\n")

    def _send(self, status: int, content_type: str, body: bytes):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store" if content_type.startswith("application/json") else "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; connect-src 'self'")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path == "/api/state":
            body = json.dumps(self.monitor.snapshot(), separators=(",", ":")).encode()
            return self._send(200, "application/json; charset=utf-8", body)
        if path.startswith("/api/task/"):
            detail = self.monitor.task_detail(path.rsplit("/", 1)[-1])
            if detail is None:
                return self._send(404, "application/json", b'{"error":"task not found"}')
            return self._send(200, "application/json; charset=utf-8",
                              json.dumps(detail, separators=(",", ":")).encode())
        assets = {"/": ("index.html", "text/html; charset=utf-8"),
                  "/index.html": ("index.html", "text/html; charset=utf-8"),
                  "/app.css": ("app.css", "text/css; charset=utf-8"),
                  "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                  "/favicon.svg": ("favicon.svg", "image/svg+xml")}
        if path not in assets:
            return self._send(404, "text/plain; charset=utf-8", b"not found\n")
        name, content_type = assets[path]
        try:
            with open(os.path.join(STATIC, name), "rb") as f:
                body = f.read()
        except OSError:
            return self._send(500, "text/plain; charset=utf-8", b"asset missing\n")
        self._send(200, content_type, body)


def main():
    ap = argparse.ArgumentParser(description="read-only NeuroGolf live operator console")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18765)
    args = ap.parse_args()
    Handler.monitor = Monitor(ROOT)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"NeuroGolf operator console: http://{args.host}:{args.port}")
    print("read-only · Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
