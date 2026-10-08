#!/usr/bin/env python3
"""orchestrate — one CLI session runs the whole overnight wave.

Spawns a GPT-5.5/xhigh Codex BUILD fleet on tasks
from lanes/QUEUE.csv, feeds each a self-contained prompt
(RULES + pack + lane thesis + ATTEMPT HISTORY + global LESSONS), enforces claims, records
every attempt, and re-queues failures. Codex Spark handles bounded administrative analysis,
price-search, and watchdog work, with GPT-5.5 as its infrastructure fallback. Builders and
strategy deciders use separate routes.
Self-improving: attempt N's prompt contains everything attempts 1..N-1 learned, plus
fleet-wide lessons.

Usage:
  python3 orchestrate.py                     # run the wave (defaults below)
  python3 orchestrate.py --lanes A,B --workers 6 --max-attempts 3
  python3 orchestrate.py --tasks 133,118     # only these tasks
  python3 orchestrate.py --dry-run           # print prompts+commands, spawn nothing
  python3 orchestrate.py --status            # ledger/attempt summary and exit

Prereqs: the `codex` CLI logged in; neurogolf_clean gates working
(runner/fast_verify.py smoke-tested). Keep --workers ≤ nproc/2 (gates are CPU-heavy).
"""
import argparse, csv, hashlib, json, os, re, shlex, signal, subprocess, sys, time, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
QUEUE = os.path.join(ROOT, "data", "lanes", "QUEUE.csv")
RESULTS = os.path.join(ROOT, "results")
CLAIMS = os.path.join(ROOT, "claims")
LESSONS = os.path.join(RESULTS, "LESSONS.md")
LEDGER = os.path.join(RESULTS, "LEDGER.csv")
ORCH_NAME = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.environ.get("ORCH_NAME", "")).strip("._-")
RUN_STARTED_AT = time.time()

sys.path.insert(0, HERE)
from config import CLEAN
# workers and admin calls inherit this env — $NEUROGOLF_CLEAN in prompts resolves for them
os.environ.setdefault("NEUROGOLF_CLEAN", CLEAN)
from integrity import integrity_errors, lint_file, recipe_errors
from evidence import ERA_FILE, banked_delta, current_attempt_pairs, file_is_current, record_is_current
from task_difficulty import profile as difficulty_profile
from graph_context import read_staged_context, stage_kaggle_reference
from runtime_roi import estimate as runtime_roi_estimate

SCORER_POLICY = """You may NEVER read, edit, or add matchers/tags/hooks to any scorer/pricer
(neurogolf_utils.py, scoring_v2.py, ngolf.py, fast_verify.py). Any plan or build that does,
or that treats a scorer-recognized/native-priced/tagged/skip tensor as free, is an automatic
NO. Only graph input, graph output, and one terminal node's internal workspace are free; all
other node outputs are charged by the fixed scorer. Iterate from an exact-reconciled proven
same-task design when one is supplied; a novel op family routes to solo probe."""


def result_name(base):
    if not ORCH_NAME:
        return os.path.join(RESULTS, base)
    stem, ext = os.path.splitext(base)
    return os.path.join(RESULTS, f"{stem}.{ORCH_NAME}{ext}")


def _atomic_json(path, payload):
    """Publish operator telemetry without exposing a partially-written snapshot."""
    tmp = path + f".tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(payload, f, separators=(",", ":"), sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def write_run_state(args, queue, running, done_tasks, not_before, stop, phase="running"):
    """Write the live, machine-readable view consumed by the read-only web console.

    STATUS.txt remains the stable human heartbeat.  This richer snapshot deliberately
    contains no prompts or commands (which can be enormous); it only describes ownership,
    pipeline position, health, and queue pressure.
    """
    ts = time.time()
    best = {}
    try:
        with open(LEDGER) as f:
            for row in csv.DictReader(f):
                if row.get("delta"):
                    task = row.get("task", "")
                    best[task] = max(best.get(task, -9), float(row["delta"]))
    except (OSError, ValueError):
        pass

    workers = []
    for job in running:
        row = job["row"]
        task = row["task"]
        difficulty = _task_difficulty(task[4:], row)
        rung = _decider_rung(task[4:]) or 0
        effective_score = max(difficulty["score"], RUNG_SCORE_FLOOR[rung])
        required = required_capability(effective_score, "builder")
        measured = model_benchmark(job.get("model_id"), job.get("effort"))
        workers.append({
            "task": task,
            "lane": row.get("lane"),
            "priority": row.get("priority"),
            "model": job.get("model"),
            "model_id": job.get("model_id"),
            "effort": job.get("effort"),
            "difficulty": difficulty["tier"],
            "difficulty_score": difficulty["score"],
            "required_capability": required,
            "benchmark_score": measured,
            "capability_gap": None if measured is None else measured - required,
            "attempt": job.get("att"),
            "pid": job["proc"].pid,
            "started_at": job.get("t0"),
            "elapsed_s": max(0, int(ts - job.get("t0", ts))),
            "timeout_s": job.get("timeout"),
            "log": os.path.relpath(job["log"].name, ROOT),
            "watchdog": (job.get("watch_log") or [None])[-1],
            "watch_strikes": job.get("watch_strikes", []),
            "terminating": "term_at" in job,
            "runtime_roi": job.get("runtime_yield"),
        })

    def helper(j, stage, model=None):
        return {
            "task": "task" + str(j.get("t", "")).zfill(3),
            "stage": stage,
            "model": j.get("label") or model,
            "provider": j.get("provider") or model,
            "model_id": j.get("model_id"),
            "effort": j.get("effort"),
            "difficulty": j.get("difficulty"),
            "difficulty_score": j.get("difficulty_score"),
            "difficulty_reasons": j.get("difficulty_reasons", []),
            "required_capability": j.get("required_capability"),
            "benchmark_score": j.get("benchmark_score"),
            "capability_gap": j.get("capability_gap"),
            "pid": j["p"].pid,
            "started_at": j.get("t0"),
            "elapsed_s": max(0, int(ts - j.get("t0", ts))),
        }

    assistants = [helper(j, "analyst", "admin-chain") for j in PM_JOBS]
    assistants += [helper(j, "price-search", "admin-chain") for j in PS_JOBS]
    for j in DECIDE_JOBS:
        label = "unknown"
        try:
            with open(j["tmp"], errors="replace") as f:
                head = f.readline()
            m = re.search(r"\[([^]]+)\]", head)
            label = m.group(1) if m else label
        except OSError:
            pass
        assistants.append(helper(j, "decider", label))

    cooling = {"task" + t.zfill(3): max(0, int(until - ts))
               for t, until in not_before.items() if until > ts}
    blocked = {model: max(0, int(until - ts))
               for model, until in MODEL_BLOCKED_UNTIL.items() if until > ts}
    worker_active = {provider: sum(1 for job in running if job.get("model") == provider)
                     for provider in ("codex",)}
    decider_active = {provider: sum(1 for job in DECIDE_JOBS
                                    if job.get("provider") == provider)
                      for provider in DECIDER_SHARES}
    decider_model_active = {lane: sum(1 for job in DECIDE_JOBS
                                      if job.get("model_lane") == lane)
                            for lane in DECIDER_MODEL_SHARES}
    fleet = fleet_desired()
    codex_builders = worker_active.get("codex", 0)
    codex_deciders = decider_active.get("codex", 0)
    payload = {
        "schema": 1,
        "run_id": ORCH_NAME or "primary",
        "pid": os.getpid(),
        "heartbeat_at": ts,
        "started_at": RUN_STARTED_AT,
        "phase": phase,
        "config": {
            "workers": args.workers,
            "lanes": args.lanes,
            "finite": args.finite,
            "improve_below": args.improve_below,
            "watchdog": not args.no_watchdog,
            "decider_cap": DECIDER_MAX_CONC,
            "worker_shares": WORKER_SHARES,
            "decider_shares": DECIDER_SHARES,
            "decider_model_shares": DECIDER_MODEL_SHARES,
            "runtime_roi": {
                "enabled": RUNTIME_ROI_ENABLED,
                "coverage_every": RUNTIME_ROI_COVERAGE_EVERY,
            },
            "shared_codex_pool": True,
        },
        "counts": {
            "workers": len(workers),
            "queued": len(queue),
            "done": len(done_tasks),
            "cooling": len(cooling),
            "analysts": len(PM_JOBS),
            "deciders_active": len(DECIDE_JOBS),
            "deciders_queued": len(DECIDE_WAIT),
            "price_searches": len(PS_JOBS),
        },
        "workers": workers,
        "assistants": assistants,
        "queue_tasks": [row.get("task") for row in queue],
        "done_tasks": ["task" + t.zfill(3) for t in sorted(done_tasks)],
        "cooling": cooling,
        "decider_queue": ["task" + t.zfill(3) for t in DECIDE_WAIT],
        "blocked_models": blocked,
        # absolute epochs so a keepalive restart can restore the bench (blocked_models
        # holds remaining-seconds for the console and is useless for restore)
        "blocked_until": {m: until for m, until in MODEL_BLOCKED_UNTIL.items() if until > ts},
        "spawn_counts": SPAWN_COUNTS,
        "allocation": {
            "worker_shares": WORKER_SHARES,
            "worker_active": worker_active,
            "worker_spawn_counts": SPAWN_COUNTS,
            "decider_shares": DECIDER_SHARES,
            "decider_active": decider_active,
            "decider_spawn_counts": DECIDER_SPAWN_COUNTS,
            "decider_model_shares": DECIDER_MODEL_SHARES,
            "decider_model_active": decider_model_active,
            "decider_model_accepted": DECIDER_MODEL_ACCEPT_COUNTS,
            "decider_model_failures": DECIDER_MODEL_FAILURES,
            "decider_sol_fallbacks": DECIDER_SOL_FALLBACKS["n"],
            "fleet_desired": fleet,
            "codex_pool": {
                "desired": (fleet or {}).get("codex"),
                "builders": codex_builders,
                "deciders": codex_deciders,
                "active": codex_builders + codex_deciders,
            },
            "gpt55_share": GPT55_SHARE,
            "model_build_stats": {k: list(v) for k, v in MODEL_BUILD_STATS.items()},
            "fleet_off": sorted(fleet_models_off()),
        },
        "infra_requeues": INFRA_REQUEUES,
        "ledger_sum": round(sum(best.values()), 6),
        "stop_requested": bool(stop.get("n")),
    }
    _atomic_json(result_name("RUN_STATE.json"), payload)

# ---- model allocation & routing ---------------------------------------------
# Provider share controls capacity; granular task difficulty controls the exact model and
# reasoning effort within that provider.
CODEX_EFFORT = os.environ.get("CODEX_EFFORT", "xhigh")
# Timeouts are GENEROUS on purpose: interrupting a reasoning model mid-build loses the
# whole attempt's context. Env-overridable. On expiry we SIGTERM first (the CLI gets
# GRACE_SECS to flush its final message / partial summary), SIGKILL only after that.
# Whatever remains on disk is salvage-priced, and the administrative analyst distills the
# partial transcript into the next attempt's brief — an interrupted run still teaches.
CODEX_TIMEOUT = int(os.environ.get("CODEX_TIMEOUT", 9000))     # 2.5 h
GRACE_SECS = int(os.environ.get("GRACE_SECS", 180))
# RETRY_RESUME (2026-07-11, flag-gated experiment, default OFF): retry attempts re-enter
# the PRIOR attempt's codex session via `codex exec resume` instead of a fresh spawn — the
# whole prior transcript becomes reusable context and (when recent) a warm cache prefix.
# Only same-model retries within MAX_AGE qualify; every guard failure falls back to a
# normal fresh spawn. Measured via runner/usage_scan.py before any default flip.
RETRY_RESUME = os.environ.get("RETRY_RESUME", "0") == "1"
RETRY_RESUME_MAX_AGE = int(os.environ.get("RETRY_RESUME_MAX_AGE", 3600))
# FAST/priority service tier (codex 0.144.4 models_cache: service_tiers ["priority"] =
# "Fast — 1.5x speed, increased usage"). Applies to BUILD spawns only: deciders assemble
# their codex command inline in decider_async and deliberately stay on the default tier.
CODEX_FAST_MODELS = {m.strip() for m in
                     os.environ.get("CODEX_FAST_MODELS", "").split(",") if m.strip()}
MODELS = {
    "codex": {"cmd": lambda p, cwd, eff, mid=None: ["codex", "exec", "-m", mid or os.environ.get("CODEX_MODEL", "gpt-5.5"),
                                     "-c", f'model_reasoning_effort="{eff}"',
                                     *(["-c", 'service_tier="priority"']
                                       if (mid or os.environ.get("CODEX_MODEL", "gpt-5.5"))
                                       in CODEX_FAST_MODELS else []),
                                     "--dangerously-bypass-approvals-and-sandbox", p],
              "timeout": CODEX_TIMEOUT, "max_conc": int(os.environ.get("CODEX_MAX_CONC", 6))},
}
DEFAULT_EFFORT = {"codex": CODEX_EFFORT}

# ---- provider share + difficulty → model×effort -----------------------------
# Weighted-fair selection keeps the long-run mix meaningful even when the worker
# count cannot represent those percentages as integer seats. Max/ultra stay out of BUILD:
# tight ONNX execution benefits from discipline more than open-ended reflection.
# BUILDS ARE CODEX-ONLY (flat gpt-5.5/xhigh — the measured best build config).
# Codex Spark handles bounded administrative roles (analyst / price-search / watchdog).
# NOTE: when results/FLEET.json exists it OVERRIDES these shares — start the wave with
# `python3 runner/fleet.py set codex=6` (or your seat count) to match.
WORKER_SHARES = {"codex": 1.0}
TIER_RUNG = {"EASY": 0, "MEDIUM": 1, "HARD": 2, "VHARD": 3, "SKIP-FLOORED": 3}
# Operator-supplied ARC capability prior. It calibrates which model×effort
# clears a difficulty band; it is not mixed into the task score itself.
MODEL_BENCHMARKS = {
    "gpt-5.6-sol": {"none": 41, "low": 49, "medium": 54, "high": 56, "xhigh": 58},
    "gpt-5.6-terra": {"none": 34, "low": 40, "medium": 46, "high": 49,
                      "xhigh": 52, "max": 55},
    "gpt-5.6-luna": {"none": 27, "low": 33, "medium": 38, "high": 46,
                     "xhigh": 49, "max": 51},
    "gpt-5.5": {"none": 35, "low": 43, "medium": 50, "high": 53, "xhigh": 55},
}

# Task difficulty and model benchmark are different scales. Convert clean task difficulty
# into the minimum benchmark capability wanted from the next model call.
TASK_CAPABILITY_BANDS = [
    (10, 38), (20, 46), (30, 49), (40, 52),
    (50, 54), (60, 55), (75, 56), (101, 58),
]
# Preferred GPT config at each measured capability. Equal-score alternatives remain in
# MODEL_BENCHMARKS for operator inspection; these ties favor the requested Terra/Sol ladder.
# 2026-07-10: entire gpt-5.6 ladder BUMPED UP one notch (operator: raise the floor toward
# gpt-5.5/xhigh capability). Each difficulty band now routes to the next-stronger rung than
# before; luna drops out of builds (floor is now terra/medium). gpt-5.6 max is NOT used
# anywhere (terra/max removed → sol/high).
# 2026-07-11 EFFORT FLOORS (wave-5): fixed-scorer-era win rates by rung — sol/xhigh 22.9%,
# terra/xhigh 20.6%, sol/high 15.0% vs terra/high 5.6%, terra/medium 5.4%, sol/medium 5.3%.
# The medium/low-high rungs burn a worker-slot-hour for a ~5% conversion; only three rungs
# with measured WR ≥ 15% remain. No max/ultra for builds (overthinks structured output).
# 2026-07-11 REVERT to gpt-5.5/xhigh (operator, wave-5 A/B evidence): arm B gpt-5.5/xhigh
# 17.2% WR / +0.610 banked vs arm A 5.6 ladder 6.9% / +0.510 (N=29/arm; sol/xhigh 0/13),
# and 5.6 burned quota faster for no measured gain. All codex traffic (builders AND
# deciders — the routes below serve both) runs the proven flat config again; a single
# model also maximizes prompt-cache prefix reuse (caching is per-model).
GPT_CAPABILITY_ROUTES = {
    38: (os.environ.get("ROUTE_GPT_MODEL", "gpt-5.5"), "xhigh"),
    46: (os.environ.get("ROUTE_GPT_MODEL", "gpt-5.5"), "xhigh"),
    49: (os.environ.get("ROUTE_GPT_MODEL", "gpt-5.5"), "xhigh"),
    52: (os.environ.get("ROUTE_GPT_MODEL", "gpt-5.5"), "xhigh"),
    54: (os.environ.get("ROUTE_GPT_MODEL", "gpt-5.5"), "xhigh"),
    55: (os.environ.get("ROUTE_GPT_MODEL", "gpt-5.5"), "xhigh"),
    56: (os.environ.get("ROUTE_GPT_MODEL", "gpt-5.5"), "xhigh"),
    58: (os.environ.get("ROUTE_GPT_MODEL", "gpt-5.5"), "xhigh"),
}
# A/B (wave-5): GPT55_SHARE of codex builds go to the proven legacy config gpt-5.5/xhigh
# (arm B); the rest ride the effort-floored 5.6 ladder (arm A). Deterministic per-
# (task, attempt) hash → both arms sample the same difficulty mix by construction; the
# comparison is read from ATTEMPTS.jsonl model_id via runner/ab_stats.py.
# 2026-07-11: default 0.5 → 0 — the ladder itself now routes to gpt-5.5/xhigh, so the arm
# split is a no-op; 0 keeps the mechanism dormant (and honest) for future A/Bs.
GPT55_SHARE = float(os.environ.get("GPT55_SHARE", "0"))


def _ab_arm_is_gpt55(task, att_no):
    # Operator wind-down overrides (FLEET.json "off" list, ~10s pickup): "sol" collapses
    # the codex arm split to gpt-5.5; "g55" collapses it to the routed (sol) config.
    off = fleet_models_off()
    if "sol" in off:
        return True
    if "g55" in off:
        return False
    if GPT55_SHARE <= 0:
        return False
    h = int(hashlib.md5(f"{task}:{att_no}".encode()).hexdigest(), 16) % 1000
    return h < int(GPT55_SHARE * 1000)
RUNG_SCORE_FLOOR = {0: 0, 1: 30, 2: 50, 3: 75}


def model_benchmark(model_id, effort):
    return MODEL_BENCHMARKS.get(str(model_id), {}).get(str(effort))


def required_capability(difficulty_score, role="builder"):
    target = next(capability for ceiling, capability in TASK_CAPABILITY_BANDS
                  if difficulty_score < ceiling)
    # A decision steers hours of build work, so even an easy decider gets a capability
    # floor; builders may use the cheaper 38/46 rungs for mechanical work.
    # 2026-07-11: decider floor 49→52 — terra/high-class output (5.6% WR as builder)
    # isn't good enough to steer hours of build work.
    return max(52, target) if role == "decider" else target


def _builder_route(provider, difficulty_score):
    target = required_capability(difficulty_score, "builder")
    model_id, effort = GPT_CAPABILITY_ROUTES[target]
    measured = model_benchmark(model_id, effort)
    return {"provider": provider, "model_id": model_id, "effort": effort,
            "required_capability": target, "benchmark_score": measured,
            "capability_gap": None if measured is None else measured - target}


def _share_order(shares, active, spawned, available):
    """Providers ordered by long-run volume deficit, then active-seat deficit."""
    providers = [provider for provider in shares if available(provider)]
    total_share = sum(shares[provider] for provider in providers)
    if not providers or total_share <= 0:
        return []
    active_slot = sum(active.get(provider, 0) for provider in providers) + 1
    spawn_slot = sum(spawned.get(provider, 0) for provider in providers) + 1
    preference = {"codex": 0}
    return sorted(providers, key=lambda provider: (
        spawn_slot * shares[provider] / total_share - spawned.get(provider, 0),
        active_slot * shares[provider] / total_share - active.get(provider, 0),
        preference.get(provider, -1),
    ), reverse=True)


# A quota-blocked provider is removed from the denominator; live providers absorb its seats
# and the weighted deficit repairs the requested mix once it returns.
MODEL_BLOCKED_UNTIL = {}
SPAWN_FAILS = {}
QUOTA_RE = re.compile(r"usage limit|purchase more credits|rate.?limit|quota exceeded|"
                      r"429 |too many requests|overloaded_error|"
                      # CLI service outages: settings/model registry unreachable makes
                      # EVERY -m call fail "unknown model id" (observed 2026-07-08 01:29)
                      r"unknown model id|Settings fetch failed|Couldn't set model|"
                      # codex network deaths (audit 2026-07-08: the 01:00-07:00 outage;
                      # codex died "Reconnecting... 5/5" + "stream disconnected".
                      # Bare "stream disconnected" (no suffix) is now matched so the write-time
                      # bench is a superset of the read-time _INFRA_ATT_RE heal (T4 audit).
                      r"stream disconnected|error sending request|"
                      r"failed to refresh available models|Reconnecting\.\.\. \d+/\d+|"
                      r"Selected model is at capacity", re.I)


def _bench_secs(txt, default=1800):
    """Quota-aware bench duration. No CLI exposes a proactive quota query, but the exhaustion
    error carries the reset time — parse it. 'try again at 4:20 PM' → seconds until that clock
    time; 'try again in N min/hr/sec' → that duration. Returns (bench_seconds, until_str).
    Falls back to `default` (30 min); +60s margin; capped 24 h."""
    import datetime
    m = re.search(r"try again at (\d{1,2}):(\d{2})\s*([AP]M)", txt, re.I)
    if m:
        h = (int(m.group(1)) % 12) + (12 if m.group(3).upper() == "PM" else 0)
        nowdt = datetime.datetime.now()
        tgt = nowdt.replace(hour=h % 24, minute=int(m.group(2)), second=0, microsecond=0)
        if tgt <= nowdt:
            tgt += datetime.timedelta(days=1)
        return min((tgt - nowdt).total_seconds() + 60, 86400), f"{m.group(1)}:{m.group(2)} {m.group(3)}"
    m = re.search(r"try again in (\d+)\s*([smh])[a-z]*", txt, re.I)
    if m:
        u = m.group(2).lower()
        mult = 1 if u == "s" else 3600 if u == "h" else 60
        return min(int(m.group(1)) * mult + 60, 86400), f"in {m.group(1)}{u}"
    return default, ""


# Builder accounting drives weighted-fair placement. The administrative analyst
# separately distills every attempt, runs price_search, and drives the watchdog.
# Every retry prompt carries the clean attempt history + analyst brief.
SPAWN_COUNTS = {"codex": 0}
INFRA_REQUEUES = {"codex": 0}  # quota/outage re-queues (heartbeat)
MODEL_BUILD_STATS = {}   # model_id -> [spawned, done] — per-model live WR for STATUS/steering
VERIFIED_SHAS = set()    # receipt sha256s that already passed finish()'s full re-gate this run
WAVE_COLD_STARTED = set()  # tasks granted a one-shot decision-free launch spawn this run
# SPRINT_COLD_START (final wave): at launch there are usually 0 CURRENT decisions (a repin
# bumps the pin-dep mtimes and staleness is mtime-based), so all att>=1 tasks would wait
# ~15-20 min for the decider pool to warm before any of the 24 seats can fill. When set,
# each task gets ONE decision-free big-swing (its analyst brief + FAMILY_ROTATION drive it,
# exactly like a virgin first attempt) so seats fill in seconds; a decider is queued for the
# NEXT attempt. A REVISE decision is still respected (never bypassed).
SPRINT_COLD_START = os.environ.get("SPRINT_COLD_START", "0") == "1"
AUTOSUBMIT_TICK = {"last": 0.0}  # main-loop throttle for the fire-and-forget auto_submit child

# Idle-work fallback (wave7 audit 2026-07-14): when every seated provider is quota-benched
# and running=0, the 46-core cgroup quota sits idle for the whole bench. Spend it on the
# deterministic Channel-B sweep over pending artifacts (near-bar ones only need a small
# shave to clear +0.15) + a pending-report refresh. One spawn per bench period.
IDLE_WORK_ENABLED = os.environ.get("IDLE_WORK", "1") == "1"
IDLE_SWEEP_JOBS = os.environ.get("IDLE_SWEEP_JOBS", "10")
IDLE_WORK = {"key": None, "procs": []}
QUOTA_CACHE = {"mtime": 0.0, "txt": ""}


def _latest_quota():
    """Trailing quota fields from results/USAGE.jsonl for the STATUS line ('' if stale/absent).
    Mtime-memoized; reads only the last ~8 KB."""
    path = os.path.join(RESULTS, "USAGE.jsonl")
    try:
        mt = os.path.getmtime(path)
    except OSError:
        return ""
    if mt == QUOTA_CACHE["mtime"]:
        return QUOTA_CACHE["txt"]
    txt = ""
    try:
        with open(path, "rb") as f:
            f.seek(max(0, os.path.getsize(path) - 8192))
            rows = [json.loads(l) for l in f.read().decode("utf-8", "replace").splitlines()[1:]
                    if l.strip().startswith("{")]
        for row in reversed(rows):
            p, s = row.get("quota_primary_pct"), row.get("quota_secondary_pct")
            if p is not None:
                txt = f"codex5h={p:g}%" + (f" wk={s:g}%" if s is not None else "")
                break
    except (OSError, ValueError):
        pass
    QUOTA_CACHE.update(mtime=mt, txt=txt)
    return txt



# retry portfolio: attempt N is ASSIGNED a different family than what failed before.
# Attempts 1–2 are RECONSTRUCTIONS (≥2× cost cut, designed from the rule, never from the
# old graph); only the final fallback is conservative.
FAMILY_ROTATION = [
    "COMPILE THE RULE — re-derive the task rule from the generator source and rebuild "
    "the graph FROM SCRATCH as a cheaper structural family (terminal Einsum contraction, "
    "Gather-LUT, crop-native, ConvInteger stencil…) priced ≥2× below the pin. Design "
    "from the RULE, not from the old graph — trimming the existing design is not a plan",
    "a DIFFERENT reconstruction family than attempt 1 — if a renderer rebuild failed, try "
    "Gather-LUT or crop-native; if a LUT failed, try terminal-Einsum or ConvInteger stencil",
    "FALLBACK (small-certain-win mode, explicitly permitted on this attempt only, after "
    "two full reconstruction attempts): keep "
    "the pin's algorithm, crop to the certificate box, relower dtypes value-exactly, "
    "suppress optional outputs — bank what clears the bar. No stubs: produce a working "
    "complete solution and run the full score_v2 gate on the win.",
]

DONE_RE = re.compile(r"DONE task(\d{3})\s+\+?([-\d.]+)")
NO_RE = re.compile(r"\bNO task(\d{3})\b")
# submit_result's byte-identical dedup (exit 4) tells the worker to end with this:
# an attempt was spent but nothing new was learned — never a NO (must not advance the
# floor/stop-loss counters), never a DONE (no new points).
DUP_RE = re.compile(r"\bDUPLICATE task(\d{3})\b")


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")


def read_queue(lanes, only_tasks):
    q = []
    with open(QUEUE) as f:
        for r in csv.DictReader(f):
            t = r["task"][4:]
            if lanes and r["lane"] not in lanes:
                continue
            if only_tasks and t not in only_tasks:
                continue
            q.append(r)
    # MAX-ROI ORDER (2026-07-11): `priority` IS the global expected-value rank written by
    # build_queue.py --reprioritize (E[Δ]·P(win), banked-aware, attempt-decayed) — honor it
    # as-is. The old lane-first sort + round-robin interleave existed so slow lanes
    # couldn't monopolize slots; the fleet is codex-only (one model, one speed),
    # so interleaving only diluted the queue front with lower-EV tasks.
    q.sort(key=lambda r: int(r["priority"]))
    return q


_QUEUE_STATE = {"mtime": None}


def maybe_reload_queue(queue, lanes, only_tasks):
    """Close the loop: after a repin's `build_queue --reprioritize` rewrites QUEUE.csv, apply
    the new priority order (and fresh bars) to the PENDING in-memory queue live — no restart.

    Only pending tasks live in `queue` (spawned tasks are popped, done tasks tracked
    separately), and refresh_basis keeps membership stable, so re-pointing each pending row
    to its fresh CSV row and re-sorting by the new `priority` never disrupts in-flight work,
    cooldowns, or claims. Returns True if a reprioritize happened."""
    try:
        mt = os.path.getmtime(QUEUE)
    except OSError:
        return False
    if _QUEUE_STATE["mtime"] is None:
        _QUEUE_STATE["mtime"] = mt
        return False
    if mt == _QUEUE_STATE["mtime"]:
        return False
    _QUEUE_STATE["mtime"] = mt
    try:
        fresh = {r["task"]: r for r in read_queue(lanes, only_tasks)}
    except (OSError, KeyError, ValueError):
        return False
    kept = [fresh[r["task"]] for r in queue if r["task"] in fresh]
    kept.sort(key=lambda r: int(r["priority"]))
    queue[:] = kept
    print(f"[{now()}] QUEUE.csv changed → reprioritized {len(kept)} pending tasks live "
          f"(repin ingested; no restart)")
    return True


def attempts_path(t):
    d = os.path.join(RESULTS, f"task{t}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "ATTEMPTS.jsonl")


def load_attempts(t):
    p = attempts_path(t)
    if not os.path.exists(p):
        return []
    out = []
    with open(p, errors="replace") as f:
        for l in f:
            if l.strip():
                try:
                    out.append(json.loads(l))
                except ValueError:
                    pass  # a partially-written line (crash mid-append) must never kill the wave
    return out


def record_attempt(t, rec):
    with open(attempts_path(t), "a") as f:
        f.write(json.dumps(rec) + "\n")


def acquire_claim(t):
    """Atomically claim a task for side-by-side orchestrators.

    Older claims are treated as stale after the longest worker timeout plus grace.
    """
    claim = os.path.join(CLAIMS, f"task{t}.claim")
    stale_after = CODEX_TIMEOUT + GRACE_SECS + 600
    try:
        fd = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        try:
            if time.time() - os.path.getmtime(claim) < stale_after:
                return False
            os.remove(claim)
            fd = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            return False
    with os.fdopen(fd, "w") as f:
        f.write(f"{os.getpid()} {now()}")
    return True


# ---- legacy infra-noise filter (READ-time only; ATTEMPTS.jsonl is never rewritten) ----
# The 2026-07-08 01:00-07:00 network outage recorded 658 CLI-death "attempts" (51% of all
# history): CLI deaths like "Reconnecting... 5/5" / "stream disconnected". They are NOT
# evidence about any task, but they inflate attempt numbers (pushing virgin tasks into the
# conservative-fallback rotation slot), max out cooldowns, and make deciders audit phantom
# failures (task019: 5/5 infra attempts, decider wrote 5 "WRONG" audit lines about them).
# New infra deaths are no longer recorded at all (QUOTA_RE above); this filter heals the
# READING of what's already on disk.
_INFRA_ATT_RE = re.compile(r"Settings fetch failed|Couldn't set model|"
                           r"stream disconnected|error sending request|"
                           r"failed to refresh available models|Reconnecting\.\.\. \d+/\d+")


def _infra_att(a):
    fr = str(a.get("fail_reason") or "")
    if _INFRA_ATT_RE.search(fr):
        return True
    # echo-header NONEs: fail_reason grabbed the prompt echo ("POST-MORTEM OF A PRIOR
    # FAILED ATTEMPT..."), all measured 227-263 s — network deaths, not work (audit 07-08)
    return (a.get("outcome") == "NONE" and (a.get("secs") or 9999) < 400
            and fr.startswith(("POST-MORTEM", "RAW TAIL")))


def real_atts(atts):
    return [a for a in atts if not _infra_att(a)]


# ---- weak-model screen lane ---------------------------------------------------
# A cheap model may run as a FIRST-PASS screen on the easiest virgin tasks. Its attempts
# are recorded and its transcripts seed the strong retry — but a weak model's failure is
# WEAK EVIDENCE: it must never advance the family rotation, count toward stop-loss/floor
# verdicts, or anchor a stronger model's approach. Empty by default; list model ids in
# WEAK_MODELS (comma-separated env) to enable the tier.
WEAK_MODELS = {m.strip() for m in os.environ.get("WEAK_MODELS", "").split(",") if m.strip()}


def _weak_att(a):
    """A build attempt by the weak screening tier."""
    return (a.get("model_id") or "") in WEAK_MODELS


def strong_atts(atts):
    """Real attempts by full-strength builders — the only evidence that parks a task."""
    return [a for a in atts if not _weak_att(a)]


def history_digest(t):
    all_atts = load_attempts(t)
    clean_pairs = current_attempt_pairs(all_atts)
    atts = [a for _, a in clean_pairs]
    out = []
    # Attempt deltas before the scorer fix are untrusted.  House money comes only from
    # the explicitly reset clean ledger (submit_result reprices before appending it).
    best = banked_delta(t, LEDGER)
    if best is not None and best > 0:
        out.append(f"HOUSE MONEY: a win of Δ+{best} is ALREADY BANKED for this task — the "
                   f"floor is locked and this attempt risks NOTHING. A design in the same "
                   f"cost class as the banked one is worthless; only a family ≥2× cheaper "
                   f"than the banked member is worth building. Do not rebuild the banked "
                   f"design.")
    if not clean_pairs:
        out.append("(no post-scorer-fix attempt evidence; older task history is archived "
                   "for forensics and cannot steer this worker. Treat this as a FIRST "
                   "clean-era attempt.)")
        return "\n".join(out)
    real = real_atts(atts)
    n_infra = len(atts) - len(real)
    if not real:
        out.append(f"(no REAL prior attempts — all {n_infra} recorded slots were CLI/"
                   f"network-outage deaths, zero evidence about this task. Treat this as "
                   f"a FIRST attempt.)")
        return "\n".join(out)
    n_weak = sum(1 for a in real if _weak_att(a))
    out.append(f"{len(real)} REAL PRIOR ATTEMPT(S) — do not repeat their failures:"
               + (f" ({n_infra} additional slots lost to CLI/network outages — "
                  f"infrastructure noise, NOT task evidence; omitted)" if n_infra else ""))
    if n_weak:
        out.append(f"⚠ {n_weak} of these were by a WEAKER SCREENING MODEL ("
                   f"marked WEAK-SCREEN below) — treat its failure analysis and family "
                   f"choices as weak evidence: do NOT inherit its approach, do NOT treat "
                   f"its NO as a floor claim; re-derive your attack from the pack. Its "
                   f"transcript is useful only for what it VERIFIED (prices, gate lines).")
    for i, a in clean_pairs:
        if _infra_att(a):
            continue
        tag = " ⚠WEAK-SCREEN" if _weak_att(a) else ""
        out.append(f"  att{i} [{a['model']}{tag}] → {a['outcome']}"
                   + (f" Δ{a.get('delta')}" if a.get("delta") is not None else "")
                   + (f" | {a.get('fail_reason','')[:200]}" if a.get("fail_reason") else ""))
    import glob
    real_nos = {i for i, a in clean_pairs if not _infra_att(a)}
    pms = [p for p in sorted(glob.glob(os.path.join(
        RESULTS, f"task{t}", "POSTMORTEM_att*.md")))
        if file_is_current(p) and (lambda m: m and int(m.group(1)) in real_nos)(
            re.match(r"POSTMORTEM_att(\d+)", os.path.basename(p)))]
    for pm in pms[-2:]:
        with open(pm, errors="replace") as f:
            body = f.read()
        if len(body) > 150:
            out.append("\nPOST-MORTEM OF A PRIOR FAILED ATTEMPT (fleet analyst — attack "
                       "DIFFERENTLY, do not repeat LOOPED items):\n" + body[:2200])
    if not pms:  # analyst not done yet (or crashed) — pass the raw evidence instead
        # only logs of REAL attempts: an infra-death log is 40 KB of reconnect spam
        logs = [l for l in sorted(glob.glob(os.path.join(RESULTS, f"task{t}", "attempts", "att*.log")))
                if file_is_current(l) and (lambda m: m and int(m.group(1)) in real_nos)(
                    re.match(r"att(\d+)", os.path.basename(l)))]
        if logs:
            tail = open(logs[-1], errors="replace").read().split("=== WORKER OUTPUT ===", 1)[-1][-1500:]
            out.append("\nRAW TAIL OF THE LAST FAILED ATTEMPT:\n" + tail)
    notes = os.path.join(RESULTS, f"task{t}", "NOTES.md")
    if file_is_current(notes):
        out.append("\nPRIOR WORKER NOTES (verbatim):\n" + open(notes, errors="replace").read()[-3000:])
    no = os.path.join(RESULTS, f"task{t}", "NO.md")
    if file_is_current(no):
        out.append("\nPRIOR FLOOR-CLAIM (challenge it only with a NEW structural family):\n"
                   + open(no, errors="replace").read()[-1500:])
    return "\n".join(out)


# Static prompt blocks are cached on (path, mtime): server-side prefix caching only matches
# byte-identical prefixes, so every spawn must render these blocks identically — and not
# re-read them from disk each time (usage_optimization/codex_prompt_caching_optimization.md).
_STATIC_MEMO = {}


def _memo_static(key, path, render):
    try:
        mt = os.path.getmtime(path)
    except OSError:
        mt = None
    ent = _STATIC_MEMO.get(key)
    if ent and ent[0] == mt:
        return ent[1]
    txt = render()
    _STATIC_MEMO[key] = (mt, txt)
    return txt


def _rules_block():
    path = os.path.join(ROOT, "data", "RULES.md")

    def render():
        rules = open(path, errors="replace").read()
        # inline only the LAW (cost model, ratios, bans, ORT traps, acceptance, idiom menu);
        # §6+ (never-touch — queued tasks are never on it; discipline — stated once in the
        # mission; toolbelt — digest below) stay on disk in ../../data/RULES.md
        cut = rules.find("## 6.")
        if cut > 0:
            rules = rules[:cut] + """## Toolbelt digest (full commands + policies: ../../data/RULES.md §8)
hypo.py NNN rule.py (numpy rule vs ALL examples — before any rebuild ONNX) · tricks.py
<keywords> (prior art) · diff_one.py NNN model (failing grids) · anatomy.py budget (price
a plan) · polish.py in out --task NNN (post-win, value-exact) · extracted/taskNNN.json
(raw examples). All run from $NEUROGOLF_CLEAN. NEVER touch
probe_submit.py or gate083/gate118.
CONTEXT SAVER: `python3 ../../runner/ask_reader.py "<precise question>" <bigfile>` — the
helper uses read-only Codex Spark to offload
token-heavy READING (dumps, dossiers, foreign scripts) to the cheap model instead of
filling your own context; hints only — verify anything load-bearing; never delegate ONNX
writing or shape math.
DESIGN SEARCH (use BEFORE giving up on cost — 40% of fleet failures were 'budget never
closed' with a verified rule): `python3 ../../runner/price_search.py NNN "<one-sentence
rule>"` — 15+ design budgets across ≥5 families, every price RECOMPUTED by the harness
(trustworthy); sketches are hypotheses, the rule must still fit."""
        return rules

    return _memo_static("rules", path, render)


def _lessons_brief(max_chars=8000, max_nonwin=25):
    """Epoch-frozen lessons block (usage_optimization, 2026-07-13). LESSONS.md gains a
    bullet with nearly every postmortem, and this block sits near the TOP of the
    cache-first prompt — re-rendering it on every mtime change invalidated the shared
    server-side prompt-cache prefix for every subsequent spawn (~600/night), leaving
    only the rules block cached. Refreshing on a 30-min epoch keeps the prefix
    byte-stable within the epoch; a new lesson still propagates the same night, ≤30 min
    later. Content and selection logic unchanged."""
    try:
        epoch = int(os.path.getmtime(LESSONS) // 1800)
    except OSError:
        epoch = None  # missing file renders "(none yet)"; re-checked next epoch key hit
    key = (epoch, max_chars, max_nonwin)
    if key not in _LESSONS_EPOCH_MEMO:
        _LESSONS_EPOCH_MEMO.clear()  # keep exactly one epoch's render
        _LESSONS_EPOCH_MEMO[key] = _lessons_brief_uncached(max_chars, max_nonwin)
    return _LESSONS_EPOCH_MEMO[key]


_LESSONS_EPOCH_MEMO = {}


def _lessons_brief_uncached(max_chars=8000, max_nonwin=25):
    """Fleet-lessons injection. The old raw `read()[-2500:]` byte tail kept only ~8 of 143
    WIN bullets (94% scrolled out) and truncated the first line mid-word — breaking the
    same-night win-propagation engine (audit 2026-07-08 / T12). This keeps EVERY distinct
    WIN bullet plus the newest distinct non-win bullets, deduped, newest-first, line-boundary
    safe, and size-bounded (wins are dropped only if the total still exceeds max_chars)."""
    if not os.path.exists(LESSONS):
        return "(none yet)"
    lines = [l.rstrip("\n") for l in open(LESSONS, errors="replace")
             if l.strip().startswith("- ")]
    wins, nonwins, seen, seen_wintask = [], [], set(), set()
    for l in reversed(lines):  # newest first
        k = l.strip()
        if k in seen:
            continue
        seen.add(k)
        if "WIN +" in l:  # WIN marker is "WIN +", not "DONE +"
            # Legacy WIN bullets have no provenance date and include scorer-hack wins. They
            # remain archived on disk but never re-enter a worker prompt.
            if not re.search(r"\[(?:2026-07-10|2026-07-1[1-9]|2026-[0-9]{2}-[0-9]{2})\]", l):
                continue
            m = re.search(r"\[task(\d+)", l)
            tk = m.group(1) if m else k
            if tk in seen_wintask:  # one (newest) win bullet per task → maximize breadth
                continue
            seen_wintask.add(tk)
            wins.append(l)
        else:
            nonwins.append(l)
    kept = list(reversed(wins)) + list(reversed(nonwins[:max_nonwin]))  # oldest→newest
    out = "\n".join(kept)
    while len(out) > max_chars and kept:  # over budget → drop the OLDEST non-win first
        drop_i = next((i for i, l in enumerate(kept) if "WIN +" not in l), 0)
        kept.pop(drop_i)
        out = "\n".join(kept)
    return out or "(none yet)"


def build_prompt(row, attempt_no, model='codex'):
    t = row["task"][4:]
    lane = row["lane"]
    pack = os.path.join(ROOT, "packs", f"task{t}", "ATTACK.md")
    sweep = os.path.join(ROOT, "results", "sweep", f"task{t}.json")
    pin_deps = [os.path.join(ROOT, "data", "pin_pv2_70_263_pertask.csv"),
                os.path.join(ROOT, "data", "PIN_SHA.txt"),
                ERA_FILE, os.path.join(HERE, "make_pack.py"),
                # a card's cached target_cost + the donor block come from these; treat them as
                # pin-deps so a repin-driven regen rebuilds an otherwise-stale pack (T13 GAP-B)
                os.path.join(ROOT, "data", "strategies.json"),
                os.path.join(ROOT, "data", "donors.json")]
    stale = (os.path.exists(pack) and (
             (os.path.exists(sweep) and os.path.getmtime(sweep) > os.path.getmtime(pack)) or
             any(os.path.exists(dep) and os.path.getmtime(dep) > os.path.getmtime(pack)
                 for dep in pin_deps)))
    stale_warn = ""
    if not os.path.exists(pack) or stale:  # (re)build so Lane E sweep facts reach the pack
        _r = subprocess.run([sys.executable, os.path.join(HERE, "make_pack.py"), str(int(t))],
                            timeout=200, capture_output=True, text=True)
        if _r.returncode != 0 and os.path.exists(pack):
            # a non-zero rebuild leaves the OLD pack on disk — warn loudly instead of silently
            # serving stale numbers as if fresh (T13 GAP-A)
            stale_warn = (f"⚠ PACK MAY BE STALE: regen against the current pin FAILED "
                          f"(rc={_r.returncode}) — re-price every number before trusting it.\n\n")
    pack_txt = (stale_warn + open(pack, errors="replace").read()) if os.path.exists(pack) \
        else "(pack missing — run make_pack)"
    graph_context = read_staged_context(t, ROOT)
    # ATTENTION BUDGET: the pack also sits on disk in the worker's cwd — inline only the
    # decision-critical core (numbers, strategy card, cert, anatomy, graph-to-beat verdict)
    # and point to ./ATTACK.md for the evidence half. Double delivery wastes attention.
    MARK = "<!-- ═══ DETAIL SECTIONS"
    if MARK in pack_txt:
        core = pack_txt.split(MARK)[0]
        pack_txt = core + """
── EVIDENCE SECTIONS ARE IN ./ATTACK.md (same folder) — READ IT BEFORE PRICING ──
It contains: the Hodel rule-as-code + verdict (+ rendered divergence examples), human rule
descriptions, this task's clean-era fleet attempts + newest clean-era build script, DONOR
tasks with cheap solved recipes, best catalog designs, the generator digest, and clean-era
dossier excerpts."""
    rules = _rules_block()  # memoized on RULES.md mtime — byte-identical across spawns
    lessons = _lessons_brief()  # win-retaining, deduped, line-safe (was raw [-2500:], T12)
    lane_file = {"A": "LANE_A_cert_shrink.md", "B": "LANE_B_midband_17_18.md",
                 "C": "LANE_C_band_16_17.md", "D": "LANE_D_named.md",
                 "G": "LANE_G_band_18.md", "H": "LANE_H_reopened_band.md"}[lane]
    lane_path = os.path.join(ROOT, "data", "lanes", lane_file)
    lane_txt = _memo_static(("lane", lane_file), lane_path,
                            lambda: open(lane_path, errors="replace").read())
    # lane thesis = header up to first task block; plus this task's block if present
    head = lane_txt.split("### task")[0][:1200]
    blk = ""
    m = re.search(rf"## task{t}.*?(?=\n## |\Z)", lane_txt, re.S) or \
        re.search(rf"### task{t}.*?(?=\n### |\Z)", lane_txt, re.S)
    if m:
        blk = m.group(0)[:2500]
    bar = "0.15"
    # rotation is indexed by REAL STRONG attempts: legacy infra-death slots must not push
    # virgin tasks into the conservative-fallback directive, and a weak-screen failure must
    # not advance the rotation either — the first strong attempt after a screen is still
    # attempt #1 strategically
    _atts = load_attempts(t)
    real_no = len(strong_atts(real_atts(_atts))) + 1
    fam = FAMILY_ROTATION[min(real_no - 1, len(FAMILY_ROTATION) - 1)]
    banked = banked_delta(t, LEDGER)
    banked = banked if banked is not None else -9
    if banked > 0:
        # a banked win makes the conservative fallback POINTLESS — the floor is locked;
        # only a ≥2× cheaper family can add points, so the directive is always big-swing
        fam = (f"HOUSE-MONEY BIG SWING: Δ+{banked} is already banked — attack with a "
               f"STRUCTURALLY DIFFERENT family ≥2× cheaper than the banked member; never "
               f"rebuild or shave the banked design (a same-cost-class success adds "
               f"NOTHING; 'trim a bit more off' is not an attack)")
    import glob as _g
    fam_line = f"FAMILY DIRECTIVE for this attempt: {fam}"
    if attempt_no >= 2:
        # exclude the "_input.md" save-everything twin: the glob DECISION_att*.md matches
        # both DECISION_attNN.md (the real decision) and DECISION_attNN_input.md (the raw
        # decider prompt), and '.'(0x2E) < '_'(0x5F) so ds[-1] would ALWAYS be the input
        # echo (blank template menu) — silently blackholing the decider (audit 2026-07-08)
        ds = sorted(d for d in _g.glob(os.path.join(RESULTS, f"task{t}", "DECISION_att*.md"))
                    if not d.endswith("_input.md"))
        ds = [d for d in ds if file_is_current(d) and not any(os.path.exists(dep) and
              os.path.getmtime(dep) > os.path.getmtime(d) for dep in pin_deps)]
        if ds:
            txt = open(ds[-1], errors="replace").read()
            # a banked task's ONLY worthwhile move is a ≥2×-cheaper family — keep that lead
            # directive even when a decision exists (don't let it be shadowed)
            hm = (f"HOUSE-MONEY: Δ+{banked} already banked — only a structurally different "
                  f"family ≥2× cheaper than the banked member adds points; never rebuild "
                  f"or shave the banked design.\n"
                  if banked > 0 else "")
            # Binding scope calibrated to measured behavior (wave6 audit: 7/10 winning
            # attempts kept the decided FAMILY but swapped a tail/op the plan got wrong,
            # while family-churn won 4%): the family is binding, the node-level recipe is
            # the default path — tactical substitution within the family is legitimate.
            fam_line = (hm + "STRATEGY DECISION (fleet decider audited ALL prior attempts against "
                        "their plans). The FAMILY choice is BINDING — never churn to another "
                        "family (measured: one family executed wins 43%, churning wins 4%). "
                        "The node-level recipe is your default path: substitute a step WITHIN "
                        "the family when the planned op is infeasible/illegal/mispriced — note "
                        "each substitution and why in NOTES.md:\n" + _plan_only(txt)[:4000])
            pr = sorted(_g.glob(os.path.join(RESULTS, f"task{t}", "PRICES_att*.md")))
            if pr:
                fam_line += ("\nHARNESS-PRICED design table (arithmetic verified; the "
                             "decision chose from it):\n"
                             + open(pr[-1], errors="replace").read()[:1400])
    if lane == "A":
        approach = f"""MISSION (Lane A — certified replacement): the generator source PROVES the pin
computes on oversized planes. The pin's ALGORITHM is your SPEC (keep its output dtype) —
REBUILD its implementation from scratch inside the certified box as a cheaper structural
family (crop-native compute, leaner renderer tail), priced ≥2× below the pin where the
certificate allows. No rule discovery needed — the certificate hands you the spec; the
win comes from a smaller GRAPH, not from shaving the existing one. Do not generate
stubs. Bar: +{bar}. Still required before code: ONE cost-budget table (every planned
tensor: shape·dtype·bytes + params) proving the paper total beats the bar.
Run full gate on the win."""
    elif lane == "G":
        approach = f"""MISSION (Lane G — short rebuild): this graph is 600–1100 B; one killed tensor family or
a leaner renderer tail wins. Aim for cost ≤ 403 B (score 19) where the rule allows; accept
anything above the +{bar} bar. Do not generate stubs. STEP ZERO (measured 10× on P(win)): read the generator
source and write ONE sentence stating the true rule. Then price ONE design (cost-budget
table: every planned tensor shape·dtype·bytes + params) before writing code. Run full gate on win."""
    else:
        approach = f"""MISSION: produce task{t}.onnx that passes the full gate. Your FIRST move is always
RECONSTRUCTION: re-derive the rule and price a from-scratch graph family ≥2× cheaper than
the pin (or the next score band) — points buy RATIOS (RULES §1b); the +{bar} bar is only
the registration floor, and trimming an existing graph is a fallback, not a plan. Every
historical Δ≥0.8 win replaced the representation; 35/35 NO→DONE
conversions came from a FAMILY change, and "correct but expensive" is the #1 historical
failure. Do not generate stubs or minimal solutions — only complete working graphs.
Protocol, in order:
1. STEP ZERO (measured 10× on P(win): generator-readers went 6/16 DONE vs 1/24 without):
   read the generator source; write ONE sentence stating the true rule.
2. If your strategy is rebuild/distill: write `rule.py` (numpy `def rule(grid)`) and make
   `cd {CLEAN} && python3 candidates/hypo.py {int(t)} <abs
   path to rule.py>` pass 100% on every split BEFORE any ONNX — a verified rule compiles
   in one pass; an unverified one wastes six.
3. Price TWO designs from different families (cost-budget table each: every planned
   tensor shape·dtype·bytes + params), at least one targeting ≥2×. Build the most
   ambitious design whose paper total closes. Never build unpriced. Run full score_v2
   gate (--full) on any candidate that clears the bar before claiming DONE."""
    # CACHE-FIRST ORDERING (usage_optimization, 2026-07-11): server-side prompt caching
    # matches only an EXACT prefix, so blocks are ordered by descending stability — global
    # statics first (identical for every spawn), per-lane, per-task, and only then the
    # per-attempt directive. The binding task directive sits at the END on purpose: it is
    # both the most volatile block and the recency-attention sweet spot. Content unchanged.
    return f"""You are a NeuroGolf build agent. The GLOBAL RULES, INTEGRITY LOCK, and FLEET LESSONS
below are the standing law for every task; YOUR SPECIFIC TASK AND ITS BINDING DIRECTIVES
ARE AT THE END of this prompt — read everything, then execute the tail mission.

=== GLOBAL RULES (the law) ===
{rules}

=== INTEGRITY LOCK (automatic NO for violations) ===
{SCORER_POLICY}

=== FLEET LESSONS (from all agents so far tonight) ===
{lessons}

=== YOUR LANE ({lane}) THESIS ===
{head}
{blk}

=== TRUSTED KAGGLE-PROVEN GRAPH REFERENCE ===
{graph_context}

=== ATTACK PACK ===
{pack_txt}

=== ATTEMPT HISTORY FOR THIS TASK ===
{history_digest(t)}

=== YOUR TASK (binding) ===
**Do not generate stubs, placeholder, or minimal/incomplete solutions.** Every implementation must be complete, correct, and pass real verification. Run the full score_v2 gate (fast_verify --full, not just --price) on every win before claiming DONE.
BUILD THIS TASK NOW (task {int(t)}). Attempt #{attempt_no}.
Work ONLY inside {ROOT}/packs/task{t}/ (cwd). Non-interactive: never ask questions.
Prior attempts may have left build.py / *.onnx drafts in your cwd — read them before starting.
{fam_line}
{approach}
Loop: edit build.py → `python3 {ROOT}/runner/fast_verify.py {int(t)} task{t}.onnx --bar {bar}`
(≤6 rounds/family, ONE family pivot allowed — fleet data: one well-chosen family executed
won 43%, churning ≥3 families in a sitting won 4%; the retry with a fresh brief rescues a
task, wandering doesn't; never re-verify an unchanged .onnx) → on PASS run the same with
`--full` (score_v2 gate on the win). On FULL PASS only then submit.
On FULL PASS (after you yourself ran fast_verify --full / score_v2 gate and it passed) run ONE command (it gates again, reprices, copies artifacts, and writes the
ledger atomically — never hand-edit LEDGER.csv):
`python3 {ROOT}/runner/submit_result.py {int(t)} task{t}.onnx --gated --lane {lane} --notes "<method>" --lesson "<one-line fleet lesson>"`
then write {RESULTS}/task{t}/NOTES.md (what worked, what to try next) and END your final
message with the exact `DONE task{t} +<Δ>` line submit_result prints.
If you exhaust the budget: write NOTES.md + (if you believe it's floored) NO.md, append
your lesson bullet, and end with: `NO task{t}`
NO.md format (mandatory, or the claim will be challenged): the families you tried as
`tried{{A,B}}`, each with its budget table or last fast_verify line, then the floor
argument. A NO with zero fast_verify runs in your transcript is invalid.
"""


class PolicyRejected(RuntimeError):
    """A rejected file was quarantined, so the unchanged input must not be requeued."""


class DecisionRevisionRequired(RuntimeError):
    """The current decision was archived and must be freshly regenerated."""


def _rejections_path(t):
    return os.path.join(RESULTS, f"task{t}", "REJECTIONS.jsonl")


def _record_rejection(t, original, quarantined, reason, kind):
    os.makedirs(os.path.dirname(_rejections_path(t)), exist_ok=True)
    with open(_rejections_path(t), "a") as f:
        f.write(json.dumps({"ts": now(), "kind": kind, "original": original,
                            "quarantined": quarantined, "reason": reason}) + "\n")


def _current_rejections(t):
    out = []
    try:
        with open(_rejections_path(t), errors="replace") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if record_is_current(rec):
                    out.append(rec)
    except OSError:
        pass
    return out


def _quarantine_file(t, path, reason, kind="policy"):
    """Move a rejected active input once and append an immutable rejection record."""
    if not os.path.exists(path):
        return None
    qdir = os.path.join(RESULTS, f"task{t}", "quarantine")
    os.makedirs(qdir, exist_ok=True)
    stem = f"{os.path.basename(path)}.{now()}.{kind}.rejected"
    dest = os.path.join(qdir, stem)
    serial = 1
    while os.path.exists(dest):
        dest = os.path.join(qdir, f"{stem}.{serial}")
        serial += 1
    os.replace(path, dest)
    _record_rejection(t, path, dest, reason, kind)
    return dest


def _policy_violations(paths):
    violations = []
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, errors="replace") as f:
            text = f.read()
        bad = lint_file(path) + recipe_errors(text)
        if bad:
            violations.append((path, bad))
    return violations


def _enforce_policy_inputs(t, paths):
    violations = _policy_violations(paths)
    if not violations:
        return
    details = "; ".join(f"{p}: line {n}: {line}"
                        for p, items in violations for n, line in items)
    for path, items in violations:
        reason = "; ".join(f"line {n}: {line}" for n, line in items)
        _quarantine_file(t, path, reason, "policy")
    raise PolicyRejected("INTEGRITY POLICY REJECT — quarantined rejected input; " + details)


def spawn(row, model, attempt_no, dry, effort=None, model_id=None):
    t = row["task"][4:]
    effort = effort or DEFAULT_EFFORT.get(model, "xhigh")
    cwd = os.path.join(ROOT, "packs", f"task{t}")
    os.makedirs(cwd, exist_ok=True)
    # Reject contaminated build/decider inputs before paying for a worker process. Explicit
    # prohibitions are ignored by lint_file; actionable scorer-hack instructions are not.
    check_paths = [os.path.join(cwd, "build.py")]
    pin_deps = [os.path.join(ROOT, "data", "pin_pv2_70_263_pertask.csv"),
                os.path.join(ROOT, "data", "PIN_SHA.txt"), ERA_FILE]
    expected_decision = os.path.join(
        RESULTS, f"task{t}", f"DECISION_att{attempt_no:02d}.md")
    decision_paths = [expected_decision] if (file_is_current(expected_decision) and not any(
        os.path.exists(dep) and os.path.getmtime(dep) > os.path.getmtime(expected_decision)
        for dep in pin_deps)) else []
    check_paths.extend(decision_paths)
    _enforce_policy_inputs(t, check_paths)
    needs_decision = bool(current_attempt_pairs(load_attempts(t)) or _current_rejections(t))
    if needs_decision and not decision_paths:
        # SPRINT cold-start: one decision-free big-swing per task so all seats fill at t=0
        # instead of idling until the decider pool warms. Queue a decider for the next
        # attempt so retries are decision-backed. Never bypasses a REVISE (that path below).
        # Cold-start does NOT consult rejection history: a cold-start has no decision file,
        # so it can't inherit a rejected plan — it's a fresh FAMILY_ROTATION big-swing built
        # from the pack. The real contamination gate is _enforce_policy_inputs(check_paths)
        # above, which re-lints the LIVE inputs at spawn time and raises PolicyRejected
        # (routed out, no attempt spent) if build.py is dirty. History records (mostly stale
        # pre-07-14 lint false-positives here) must not strand 90% of the queue behind the
        # decider warmup — that would defeat the whole point of the cold-start.
        if SPRINT_COLD_START and not dry and t not in WAVE_COLD_STARTED:
            WAVE_COLD_STARTED.add(t)
            DECIDE_WAIT[t] = row
            print(f"[{now()}] task{t}: SPRINT cold-start big-swing (no current decision; "
                  f"decider queued for att{attempt_no + 1})")
        else:
            raise DecisionRevisionRequired(
                f"no current authorized DECISION_att{attempt_no:02d}; fresh decision required")
    if decision_paths:
        status, reason = _decision_budget_status(decision_paths[0])
        if status not in ("READY", "EXPLORATORY"):
            _quarantine_file(t, decision_paths[0], reason, "budget-revise")
            raise DecisionRevisionRequired(
                f"decision budget requires REVISE; quarantined input: {reason}")
    # Future-run rollout: stage only after policy/budget authorization. Existing Python
    # orchestrator processes loaded before this hook was installed never execute it, so
    # their active packs and agents remain untouched.
    stage_kaggle_reference(t, cwd)
    SPAWN_COUNTS[model] = SPAWN_COUNTS.get(model, 0) + 1  # only authorized build tallies
    MODEL_BUILD_STATS.setdefault(model_id or model, [0, 0])[0] += 1
    prompt = build_prompt(row, attempt_no, model=model)
    cmd = MODELS[model]["cmd"](prompt, cwd, effort, model_id)
    tag = f"{model}:{model_id or '-'}/{effort}"
    hot = bool(HOT_RETRY and HOT_READY.pop(t, None))
    resumed_sid = None
    if RETRY_RESUME and model == "codex" and attempt_no >= 2:
        # resume the immediately-prior attempt's session: the prior transcript replays as
        # context (and, if recent, as a warm cache prefix). Guards: the LAST attempt must
        # carry a session_id, be the same model, and have ended < MAX_AGE ago; the delta
        # prompt is only the per-attempt tail — statics/pack already live in the session.
        _atts_r = load_attempts(t)
        _last = _atts_r[-1] if _atts_r else None
        _mid = model_id or os.environ.get("CODEX_MODEL", "gpt-5.5")
        marker = "=== YOUR TASK (binding) ==="
        mi = prompt.rfind(marker)
        if (_last and _last.get("session_id") and _last.get("model_id") == _mid
                and time.time() - _last.get("end_epoch", 0) < RETRY_RESUME_MAX_AGE
                and mi >= 0):
            prompt = ("RETRY IN THE SAME SESSION. Disregard the prior attempt's family "
                      "choice — the directive below SUPERSEDES it; the global rules, "
                      "integrity lock, and attack pack from before still apply.\n\n"
                      + prompt[mi:])
            cmd = ["codex", "exec", "resume", _last["session_id"], "-m", _mid,
                   "-c", f'model_reasoning_effort="{effort}"',
                   "--dangerously-bypass-approvals-and-sandbox", prompt]
            resumed_sid = _last["session_id"]
            tag += "+resume"
    if dry:
        print(f"--- DRY task{t} att{attempt_no} model={tag} cwd={cwd}")
        print("    cmd:", " ".join(shlex.quote(c) for c in cmd[:7]), f"…(+prompt {len(prompt)} chars)")
        return None
    logdir = os.path.join(RESULTS, f"task{t}", "attempts")
    os.makedirs(logdir, exist_ok=True)
    log = open(os.path.join(logdir, f"att{attempt_no:02d}_{model}-{effort}_{now()}.log"), "w")
    # echo the prompt TAIL: after the cache-first reorder the head is RULES boilerplate;
    # the task directive (the part a human wants to see) is at the end. Classification and
    # the watchdog split on the sentinel below and never read this echo.
    log.write("(prompt tail — full prompt in the _prompt.txt twin)\n…"
              + prompt[-2000:] + "\n=== WORKER OUTPUT ===\n"); log.flush()
    # save-everything: the full prompt (log keeps only a 2KB head for readability)
    open(log.name.replace(".log", "_prompt.txt"), "w").write(prompt)
    p = subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, text=True)
    return {"proc": p, "row": row, "model": model, "model_id": model_id,
            "effort": effort, "att": attempt_no, "log": log, "t0": time.time(),
            "timeout": MODELS[model]["timeout"], "hot": hot, "resumed_sid": resumed_sid}


def _latest_decision(t):
    import glob
    ds = sorted(d for d in glob.glob(os.path.join(RESULTS, f"task{t}", "DECISION_att*.md"))
                if not d.endswith("_input.md") and file_is_current(d))
    return ds[-1] if ds else None


def _decision_revision_reason(t):
    """Return why the latest decision must be replaced before any worker may run."""
    path = _latest_decision(t)
    if not path:
        return None
    status, reason = _decision_budget_status(path)
    if status in ("READY", "EXPLORATORY"):
        return None
    return f"decision budget requires REVISE ({reason})"


# ---- attempt analyst (tool-disabled read-only Codex Spark) --------------------------
# Failures become a ≤8-bullet "attack differently" brief embedded in the next (stronger)
# model's retry prompt; wins become a transfer debrief for the morning + sibling tasks.
# Fire-and-forget: never blocks the wave; reaped opportunistically in the main loop.
PM_JOBS = []
ADMIN_PRIMARY_PROVIDER = os.environ.get("ADMIN_PRIMARY_PROVIDER", "codex")
ADMIN_MODEL = os.environ.get("ADMIN_SPARK_MODEL", "gpt-5.3-codex-spark")


def _admin_cmd(prompt, effort="low", timeout=300):
    """Bounded administrative route; read-only Codex Spark -> GPT-5.5."""
    return [sys.executable, os.path.join(HERE, "admin_model.py"),
            "--prompt", prompt, "--effort", effort, "--timeout", str(timeout)]

PM_HEAD = """SINGLE-RESPONSE TASK. You have NO tools; your FIRST message IS the final answer.
You are the fleet attempt analyst for NeuroGolf ONNX graph-golf.
Below is the transcript tail of attempt #{att} on task {t} by [{model}] -> outcome {outcome}.
{weak_note}"""
PM_FAIL = PM_HEAD + """Write AT MOST 11 short bullets for the DECIDER and the next builder — no preamble, no code:
- DECISION-AUDIT (only if a decider plan is shown below): was the plan followed? did its
  FAMILY/budget claims hold in practice? one line: what the decision got right / wrong.
- FAMILY: structural family/design tried (name the charged tensors if visible)
- PROGRESS: furthest VERIFIED step (budget closed on paper? priced Δ vs bar? fast gate? full gate?) — quote the last PRICE/VERDICT line
- RULE-STATUS: was the RULE itself verified (hypo.py pass / gate draws correct) or still uncertain? Cost problems and rule problems need opposite next moves.
- ROOT CAUSE: the exact blocking error (quote it)
- FAILDIFF (only if the transcript shows a gate FAIL with rendered expected/got grids):
  name the failing DRAW-CLASS pattern in one line — e.g. "fails only on grids with
  tied-size objects: the graph tie-breaks leftmost, the rule wants topmost"
- LOOPED: what was retried without progress — the retry must NOT repeat this
- ARTIFACTS: files left in the pack dir worth resuming from (build.py state, rule.py, drafts) — name them
- KEEP: what verifiably worked and is reusable
- UB-RISK: flag any data-dependent indexing, dynamic modulus, gather-by-computed-index, or
  input-distribution assumption visible in the build as `UB-RISK: <op/line>` (the hidden-set
  silent-zero class static razors can't see); write "UB-RISK: none seen" if clean
- NEXT: ONE concrete DIFFERENT-family attack consistent with the evidence above
Only transcript facts; write "unknown" where the transcript is silent.
=== TRANSCRIPT ===
{excerpt}"""
PM_WIN = PM_HEAD + """Write AT MOST 7 short bullets distilling the WIN — no preamble, no code:
- DECISION-AUDIT (only if a decider plan is shown below): did the win follow the plan or
  deviate? one line on the decision's calibration.
- FAMILY: the winning family/design (name the tensors that replaced the pin's cost)
- COST: final cost and Δ (quote the last PRICE/VERDICT line)
- KEY MOVE: the single decision that made the budget close
- WASTE: detours to skip when repeating this
- TRANSFER: which idioms/sibling-task shapes this generalizes to
Only transcript facts; write "unknown" where the transcript is silent.
=== TRANSCRIPT ===
{excerpt}"""


def postmortem_async(t, model, att_no, outcome, log_path, interrupted=False, model_id=None):
    try:
        txt = open(log_path, errors="replace").read().split("=== WORKER OUTPUT ===", 1)[-1]
    except OSError:
        return
    if len(txt.strip()) < 200:  # instant crash — nothing to distill
        return
    if _blocked("admin"):  # administrative CLI benched: raw transcript remains available
        return            # history_digest falls back to the raw transcript tail
    # Keep the evidence compact so an administrative summary stays fast and factual.
    excerpt = txt[:2500] + ("\n[...snip...]\n" + txt[-6500:] if len(txt) > 9000 else txt[2500:])
    # decision accountability: show the analyst the plan that steered this attempt
    dp = os.path.join(RESULTS, f"task{t}", f"DECISION_att{att_no:02d}.md")
    if file_is_current(dp):
        excerpt = ("=== THE DECIDER'S PLAN FOR THIS ATTEMPT (audit it) ===\n"
                   + open(dp, errors="replace").read()[:2000] + "\n\n" + excerpt)
    # WIN template ONLY for verified wins — DONE_NO_ARTIFACT/UNVERIFIED are failures and
    # must get the failure debrief (first run: WIN-framing corrupted 20 failure briefs)
    tpl = PM_WIN if outcome in ("DONE", "DONE_SALVAGED") else PM_FAIL
    weak = (model_id or "") in WEAK_MODELS
    weak_note = ("NOTE: this attempt was by the WEAK SCREENING model tier. Open your "
                 "brief with the line 'WEAK-MODEL ATTEMPT — weak evidence'. Its failure is "
                 "NOT a floor claim and its family choice must not anchor the stronger "
                 "successor; report only what it VERIFIED (prices, gate lines, artifacts).\n"
                 if weak else "")
    prompt = tpl.format(att=att_no, t=t, model=model, outcome=outcome, excerpt=excerpt,
                        weak_note=weak_note)
    try:  # save-everything: analyst input
        open(os.path.join(RESULTS, f"task{t}", "attempts",
                          f"att{att_no:02d}_pm_input.txt"), "w").write(prompt)
    except OSError:
        pass
    if interrupted:
        prompt = prompt.replace(
            "=== TRANSCRIPT ===",
            "IMPORTANT: this attempt was INTERRUPTED at its time limit — the transcript ends "
            "mid-work and the outcome is NOT a verdict on the approach. Judge only what was "
            "verified before the cut. In NEXT, say what the successor should CONTINUE from "
            "(drafts named in the transcript, last verified budget/gate state) versus redo.\n"
            "=== TRANSCRIPT ===")
    final = os.path.join(RESULTS, f"task{t}", f"POSTMORTEM_att{att_no:02d}.md")
    tmp = final + ".tmp"
    out = open(tmp, "w")
    out.write(f"# post-mortem att{att_no:02d} [{model}] outcome={outcome} "
              "(bounded administrative analyst)\n")
    out.flush()
    cmd = _admin_cmd(prompt, os.environ.get("ADMIN_PM_EFFORT", "low"), 300)
    p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.DEVNULL,
                         stdin=subprocess.DEVNULL, text=True)
    PM_JOBS.append({"p": p, "f": out, "tmp": tmp, "final": final, "t0": time.time(),
                    "t": t, "provider": "admin", "model_id": ADMIN_MODEL,
                    "effort": os.environ.get("ADMIN_PM_EFFORT", "low")})


def pm_pending(t):
    """True while this task's analyst brief is still being written — gates its retry."""
    return any(j.get("t") == t for j in PM_JOBS)


PM_FAIL_STREAK = {"n": 0}


def reap_postmortems():
    for j in PM_JOBS[:]:
        rc = j["p"].poll()
        # 600 s: xhigh analysts need more than the old 300 s low-effort budget — a kill
        # here counts toward PM_FAIL_STREAK, so the window must fit normal xhigh runs
        if rc is None and time.time() - j["t0"] > int(os.environ.get("PM_KILL_SECS", 600)):
            j["p"].kill(); rc = -9
        if rc is None:
            continue
        PM_JOBS.remove(j); j["f"].close()
        ok = False
        try:
            body = open(j["tmp"], errors="replace").read()
            # A refusal/offload stub would gate the retry on garbage; drop it so
            # history_digest falls back to the raw tail.
            stub = len(body) < 400 and re.search(r"offload|reading the (full )?prompt",
                                                 body, re.I)
            if rc == 0 and os.path.getsize(j["tmp"]) > 150 and not stub:
                os.replace(j["tmp"], j["final"])
                ok = True
            else:
                if stub:
                    print(f"[{now()}] task{j['t']}: analyst brief was an offload stub — "
                          f"dropped (raw-tail fallback)")
                os.remove(j["tmp"])
        except OSError:
            pass
        if ok:
            PM_FAIL_STREAK["n"] = 0
        else:
            PM_FAIL_STREAK["n"] += 1
            if PM_FAIL_STREAK["n"] >= 3 and not _blocked("admin"):
                MODEL_BLOCKED_UNTIL["admin"] = time.time() + 1800
                PM_FAIL_STREAK["n"] = 0
                print(f"[{now()}] administrative analyst failed 3× — admin route benched 30 min "
                      f"(admin roles skip-spawn; briefs fall back to raw transcript tails)")


# ---- strategy DECIDER (runs between attempts, AFTER the analyst brief) -------------
# Consumes the strategy card + all clean chronological evidence and emits a binding decision.
# The 0–100 score selects the reasoning depth; the retry is budget-gated on it.
DECIDE_WAIT = {}   # task -> queue row, waiting for its analyst brief before deciding
DECIDE_JOBS = []

# CACHE-FIRST ORDERING (usage_optimization, 2026-07-11): the rubric below is byte-identical
# for every decider spawn — it leads so the server-side prompt cache can match it as a
# shared prefix; the task id / attempt / target / card / record are all in the tail.
DECIDER_TPL = """SINGLE-RESPONSE TASK. You have NO tools; your FIRST message IS the final answer.
You are the strategy DECIDER for a NeuroGolf task. THE TASK ID, ATTEMPT NUMBER, TARGET
COST, STRATEGY CARD, AND COMPLETE CHRONOLOGICAL RECORD — every attempt AND every plan you
issued — ARE AT THE END of this prompt; you choose what the coming attempt does.

DO ALL THREE STEPS, IN ORDER:
STEP 1 — AUDIT EVERY PAST DECISION against its result. One line per attempt:
  "att N: plan said <X> → result was <Y> → RIGHT/WRONG because <Z>".
  Judge the PLAN, not the builder: a good plan executed badly ≠ a bad plan.
STEP 2 — EXTRACT what the whole record now proves: rule status (verified where/how?),
  per-family cost floors observed, and any repeated misjudgment in YOUR OWN past plans
  (over-optimistic budgets? re-picking a dead family? ignoring RULE-STATUS?).
STEP 3 — DECIDE, strictly improving on the audited mistakes. Heuristics: rule verified
  but expensive ⇒ keep rule, change representation; rule never verified ⇒ rule-first-reset;
  a family that failed twice is DEAD here; a banked small win means the floor is locked —
  free risk, never rebuild the banked family, target ≥2× below its cost.

INTEGRITY LOCK: You may NEVER read, edit, or add matchers/tags/hooks to any scorer/pricer
(neurogolf_utils.py, scoring_v2.py, ngolf.py, fast_verify.py). A scorer-recognized, tagged,
native-priced, or skip tensor is NOT free. Such a plan is an automatic NO. Only graph input,
graph output, and one terminal node's internal workspace are free; all other node outputs are
charged. Prefer an exact-reconciled proven same-task design; novel families route to solo probe.
Do NOT restate this prohibition in your output: lines naming scorer/pricer/tag/hook/skip
vocabulary are integrity-linted and can void the decision — simply never discuss such plans.

Output EXACTLY these sections (≤2800 chars total, no preamble):
REFLECTION: the STEP-1 audit lines + STEP-2 patterns (≤120 words total)
DECISION: continue-same-family | pivot-family | rule-first-reset | conservative-fallback
FAMILY: the chosen family + design sketch with a cost budget that closes vs register
target cost ≤ the TARGET COST stated at the end of this prompt (list the planned charged
tensors: shape·dtype·bytes)
PRICED-EXPLORATORY-BUILD: NO, or YES — <why this deliberately non-closing but fully priced
build is worth one solo-probe-oriented experiment>. YES cannot excuse missing/malformed math.
BUILD-RECIPE: numbered concrete steps the builder executes — name the ngolf
calls / ops, the exact tensor shapes+dtypes, the crop bounds, the terminal renderer.
ROUTING-DIFFICULTY: one word — EASY|MEDIUM|HARD|VHARD — how hard is the design YOU chose
to actually build correctly and cheaply? EASY = a known replacement family closes the
budget (donor transfer / re-run a
proven design); MEDIUM = standard structural build (plane-kill, terminal renderer, einsum);
HARD = nonstandard structure or tight multi-constraint budget; VHARD = needs genuine
invention / no proven path. This SETS the model tier the harness routes this retry to, so be
honest — over-rating burns tokens, under-rating routes a hard build to a light model.
FAMILY-BUDGET-JSON: one line, machine-parsable, your chosen design's exact budget:
{{"tensors":[{{"shape":[1,1,10,10],"dtype":"u8","count":2}}],"params":123}}
(the harness reprices this and stamps the decision ✓/✗ — dtypes: u8/i8/b=1B, f16=2,
f32/i32/u32=4, i64=8; params = initializer ELEMENT count; graph output tensor is free)
BUDGET COMPLETENESS CHECK (mandatory, the measured #1 failure mode): walk your
BUILD-RECIPE line by line and put EVERY node output it creates into tensors[] — only the
graph output and ONE terminal node's internal workspace are free; every Cast, mask,
crop, index, scratch plane is charged. A recipe step whose output is missing from
tensors[] means your budget is fiction (audited misses ran 2-8x over their paper price).
REUSE: artifacts/insights from prior attempts to start from (name the files)
AVOID: dead ends that must not be repeated (cite the audit evidence)
FIRST-3-ACTIONS: the first three concrete commands/edits of the attempt

=== YOUR TASK (binding) ===
DECIDE FOR NeuroGolf task {t}. Attempt {natt} is about to start; register target cost ≤ {target}.
=== STRATEGY CARD ===
{card}
=== EXACT KAGGLE-PROVEN GRAPH CONTEXT ===
{graph_context}
=== COMPLETE CHRONOLOGICAL RECORD (attempts + your plans) ===
{summaries}"""


def _plan_only(text):
    """A stored decision file = REFLECTION + decision sections. Past reflections are NOT
    re-fed (self-reinforcing narrative risk) — the decider re-reflects from ground truth
    each cycle. Slice from the DECISION: section; fall back to the whole text."""
    i = text.find("\nDECISION:")
    return text[i + 1:] if i >= 0 else text


def build_decider_prompt(t, row, budget=60000):
    """The COMPLETE chronological record: every attempt + every plan + every analyst
    summary. Only if the record physically exceeds the budget are the OLDEST attempts'
    bodies dropped (their one-line records always stay)."""
    all_atts = load_attempts(t)
    natt = len(all_atts) + 1
    # infra deaths (CLI/network outages) are excluded from the audited record: the decider
    # must not spend STEP-1 auditing phantom failures (audit 2026-07-08: task019's decider
    # wrote 5 "WRONG" lines about 5 CLI deaths on a task no builder ever touched)
    clean_pairs = current_attempt_pairs(all_atts)
    pairs = [(i, a) for i, a in clean_pairs if not _infra_att(a)]
    n_infra = len(clean_pairs) - len(pairs)
    n_archived = len(all_atts) - len(clean_pairs)
    rejections = _current_rejections(t)
    if not pairs and not rejections:
        return None, 0  # nothing real happened yet — nothing to decide on
    heads, bodies = [], []
    for i, a in pairs:
        heads.append(f"--- attempt {i} [{a['model']}] outcome={a['outcome']}"
                     + (f" Δ{a.get('delta')}" if a.get("delta") is not None else "")
                     + (f" | {a.get('fail_reason','')[:200]}" if a.get("fail_reason") else ""))
        b = []
        dp = os.path.join(RESULTS, f"task{t}", f"DECISION_att{i:02d}.md")
        if file_is_current(dp):
            b.append("YOUR PLAN for this attempt was:\n"
                     + _plan_only(open(dp, errors="replace").read())[:1600])
        pm = os.path.join(RESULTS, f"task{t}", f"POSTMORTEM_att{i:02d}.md")
        if file_is_current(pm):
            with open(pm, errors="replace") as f:
                b.append(f.read()[:2400])
        bodies.append("\n".join(b))
    # graceful degradation: drop oldest BODIES first, never the record lines
    drop = 0
    def render(k):
        out = []
        if n_archived:
            out.append(f"({n_archived} pre-fix attempt records archived for forensics and "
                       f"excluded from this decision; their deltas and conclusions are "
                       f"not valid under the fixed scorer)")
        if n_infra:
            out.append(f"({n_infra} attempt slots omitted: CLI/network-outage deaths — "
                       f"infrastructure noise, NOT evidence about this task; that is why "
                       f"attempt numbering has gaps)")
        if rejections:
            out.append("--- CLEAN-ERA REJECTED INPUTS (no worker attempt was spent) ---")
            for rec in rejections[-4:]:
                out.append(f"{rec.get('kind', 'policy')} rejection: "
                           f"{os.path.basename(rec.get('original', 'input'))} — "
                           f"{str(rec.get('reason', ''))[:500]}")
        if k:
            out.append(f"({k} oldest attempts shown as one-line records only — the record "
                       f"physically exceeds the context budget; their verdicts are above)")
        for j in range(len(pairs)):
            out.append(heads[j])
            if j >= k and bodies[j]:
                out.append(bodies[j])
        return "\n".join(out)
    summaries = render(0)
    while len(summaries) > budget and drop < len(pairs):
        drop += 1
        summaries = render(drop)
    for extra in ("NOTES.md", "NO.md"):
        p = os.path.join(RESULTS, f"task{t}", extra)
        if file_is_current(p):
            summaries += f"\n--- {extra}:\n" + open(p, errors="replace").read()[-1500:]
    card = "(no card)"
    sp = os.path.join(ROOT, "data", "strategies.json")
    if os.path.exists(sp):
        with open(sp) as f:
            c = json.load(f).get(t)
        if c and "primary" in c:
            card = json.dumps(c, indent=0)[:1500]
    # exact-attempt table first, else the NEWEST still-current one: wave6 audit found 476
    # priced tables on disk while 0/8 sampled decisions cited any — the exact-natt match
    # almost never exists (price_search only fires on the COST-REJECT class), so deciders
    # self-priced from prose and under-priced budgets up to 8x (10/10 sampled decision-
    # backed NOs were paper-budget misses, zero execution failures).
    prices = os.path.join(RESULTS, f"task{t}", f"PRICES_att{natt:02d}.md")
    if not file_is_current(prices):
        import glob as _g
        cands = [p for p in sorted(_g.glob(
            os.path.join(RESULTS, f"task{t}", "PRICES_att*.md"))) if file_is_current(p)]
        prices = cands[-1] if cands else prices
    if file_is_current(prices):
        summaries += ("\n=== HARNESS-PRICED KNOWN-FAMILY DESIGNS (arithmetic verified) ===\n"
                      "These PROVE the bar is reachable and give you safe fallbacks. They are "
                      "NOT a menu limit: inventing a BETTER or novel design is explicitly "
                      "encouraged — the fleet's biggest wins came from families no enumerator "
                      "proposed. Whatever you choose, your FAMILY-BUDGET-JSON below gets "
                      "harness-priced, so novelty costs nothing in safety.\n"
                      "RECONCILE: if your chosen family (or a close variant) appears in this "
                      "table, your FAMILY-BUDGET-JSON may undercut the table's price ONLY if "
                      "you name the exact charged tensor the table includes that your design "
                      "eliminates — otherwise adopt the table's number. Self-priced prose "
                      "budgets are the #1 measured failure mode (omitted charged "
                      "intermediates, up to 8x under-priced).\n"
                      + open(prices, errors="replace").read()[:2200])
    fa = os.path.join(RESULTS, f"task{t}", "FLOOR_AUDIT.md")
    if file_is_current(fa):
        # sidecar's advisory audit of the standing floor claim (attempted-vs-mentioned
        # families, infra-tainted NOs, harness-feasible designs never tried) — evidence
        # for the continue/park judgment; the decider remains the judge.
        fa_txt = open(fa, errors="replace").read()
        fa_verdict = re.search(r"FLOOR-CONFIDENCE:.*", fa_txt)
        summaries += ("\n=== FLOOR-CLAIM AUDIT (UNVERIFIED analyst advisory — hypotheses "
                      "only; the harness-priced table and your own audit rule) ===\n"
                      + (fa_verdict.group(0) + "\n" if fa_verdict else "")
                      + fa_txt[:1600])
    tt = os.path.join(ROOT, "packs", f"task{t}", "TRANSFER_TICKET.md")
    if file_is_current(tt):
        # a same-night win on a taxonomy-sibling task; hypotheses only — any family the
        # decider picks from it still gets harness-priced like every other plan.
        summaries += ("\n=== TRANSFER HYPOTHESIS (sibling task won tonight — verify) ===\n"
                      + open(tt, errors="replace").read()[:1400])
    bs = os.path.join(RESULTS, f"task{t}", "BANKED_SHRINK.md")
    if file_is_current(bs):
        # house-money evidence: where the banked member's remaining fat is (advisory —
        # the >=2x-cheaper policy and the harness pricer still gate any plan from it)
        summaries += ("\n=== BANKED-MEMBER SHRINK HYPOTHESES (house-money re-attack) ===\n"
                      + open(bs, errors="replace").read()[:1400])
    ww = os.path.join(RESULTS, "WHAT_WORKS_NOW.md")
    if os.path.exists(ww):
        # fleet-wide measured evidence (analyst distillation of clean-era postmortems +
        # ledger): which families CONVERT under the fixed scorer, which keep dying.
        # Factual only — feeds family choice the same way LESSONS feeds workers.
        # 2026-07-11: cap 2200→5000 — the old cap cut the file mid-"FAILING", so deciders
        # saw what converts but LOST the failing-families and dead-ends sections, the
        # highest-value evidence for auditing a plan (STEP 1-3 rubric).
        summaries += ("\n=== WHAT CONVERTS FLEET-WIDE (fixed-scorer era, measured) ===\n"
                      + open(ww, errors="replace").read()[:5000])
    if os.path.exists(os.path.join(ROOT, "packs", f"task{t}", "rule.py")):
        # ruleforge/compiled-circuit reminder (wave7 audit 2026-07-14: 0 uses despite the
        # engine's 3 measured wins being wave6's biggest; deciders never saw the citations
        # and WHAT_WORKS' anti-Einsum lines — which are about template-FITTING, a different
        # failure mode — plausibly steered them away). Per-task tail: cache prefix intact.
        summaries += (
            "\n=== COMPILED-CIRCUIT FAMILY (measured: task303 +1.1575, task202 +1.1533, "
            "task350 +0.9362 — the biggest wins of wave6, 3-for-3 on rule-like tasks) ===\n"
            "This task has a verified rule.py. For boolean/finite-domain cellwise rules, "
            "compiling the WHOLE rule into one terminal Einsum via runner/compile_circuit.py "
            "(worked byte-verified exemplars: data/intel/compile_exemplars/) is an available "
            "family. Note: WHAT-WORKS' 'closed-form Einsum fitter' dead-end is about "
            "TEMPLATE-FITTING residual rules — circuit COMPILATION of a verified rule is a "
            "different, measured-winning mechanism. It is a hypothesis like any other: your "
            "FAMILY-BUDGET-JSON still gets harness-priced.\n")
    leg = os.path.join(ROOT, "data", "ORT124_LEGALITY.md")
    if os.path.exists(leg):
        # measured op-legality sheet (2026-07-13): workers always had this via their pack,
        # but the DECIDER did not — so a binding plan could route the attempt into a
        # known-DEAD op×dtype combo and burn the whole attempt at the lint (legality was
        # ~20% of wave6 NO verdicts). Same evidence-not-spec caveat as in the pack.
        summaries += ("\n=== MEASURED ORT-1.24 LEGALITY (ops the gate WILL reject — do "
                      "not plan them) ===\n" + open(leg, errors="replace").read()[:2500])
    graph_context = read_staged_context(t, ROOT, limit=6000)
    return DECIDER_TPL.format(t=t, natt=natt, target=row.get("bar_cost") or "your pack's target",
                              card=card, graph_context=graph_context,
                              summaries=summaries), natt


# Decider allocation: provider is Codex-only; model-level fair allocation sends half of
# accepted decisions to GPT-5.6 Sol/xhigh and half to GPT-5.5/xhigh. Sol receives the
# hardest currently-ready task. Required_capability remains visible in telemetry.
# The provider stays Codex-only; model allocation is independent of builder routing.
DECIDER_SHARES = {"codex": 1.0}
DECIDER_SPAWN_COUNTS = {"codex": 0}
DECIDER_SOL_SHARE = min(1.0, max(0.0, float(os.environ.get("DECIDER_SOL_SHARE", ".50"))))
DECIDER_MODEL_SHARES = {"base": 1.0 - DECIDER_SOL_SHARE, "sol": DECIDER_SOL_SHARE}
DECIDER_MODEL_ACCEPT_COUNTS = {"base": 0, "sol": 0}
DECIDER_MODEL_FAILURES = {"base": 0, "sol": 0}
DECIDER_MODEL_BLOCKED_UNTIL = {"base": 0.0, "sol": 0.0}
DECIDER_FORCE_BASE = set()
DECIDER_SOL_FALLBACKS = {"n": 0}


def _task_difficulty(t, row):
    return difficulty_profile(t, row, RESULTS, ledger_path=LEDGER)


RUNTIME_ROI_SCORE_CACHE_SECS = max(
    15, int(os.environ.get("RUNTIME_ROI_SCORE_CACHE_SECS", "60")))
_RUNTIME_YIELD_CACHE = {}


def _runtime_yield(t, row, provider="codex", difficulty=None):
    """Read-only live yield used solely for in-memory scheduler ordering/telemetry."""
    key = (str(t).zfill(3), provider)
    nowt = time.time()
    if difficulty is None:
        cached = _RUNTIME_YIELD_CACHE.get(key)
        if cached and nowt - cached[0] < RUNTIME_ROI_SCORE_CACHE_SECS:
            return cached[1]
    difficulty = difficulty or _task_difficulty(t, row)
    live = runtime_roi_estimate(t, row, provider, RESULTS, difficulty)
    _RUNTIME_YIELD_CACHE[key] = (nowt, live)
    return live


def _decider_available(provider):
    return provider == "codex" and not _blocked("codex")


def _decider_provider_order(limit=None):
    limit = DECIDER_MAX_CONC if limit is None else limit
    active = {provider: sum(1 for job in DECIDE_JOBS
                            if job.get("provider") == provider)
              for provider in DECIDER_SHARES}
    eligible = [provider for provider in DECIDER_SHARES if _decider_available(provider)]
    total_share = sum(DECIDER_SHARES[p] for p in eligible)
    live = fleet_deciders()

    def available(provider):
        # Operator-set live counts win outright (fleet.py deciders codex=N) — an
        # explicit 0 drains that lane. Otherwise re-normalize over healthy providers so an
        # unavailable lane never leaves shared Codex seats idle. Long-run accepted-call
        # deficits restore the requested mix when it returns; the active cap prevents it
        # from flooding the whole lane at once.
        if live is not None and provider in live:
            cap = live[provider]
        else:
            cap = max(1, int(limit * DECIDER_SHARES[provider]
                             / max(total_share, 1e-9) + .999))
        return active[provider] < cap and _decider_available(provider)
    return _share_order(DECIDER_SHARES, active, DECIDER_SPAWN_COUNTS,
                        available)


# Vendor-group wind-down mapping for decider lanes: a lane is OFF when any of its groups
# is in the FLEET.json "off" list (runner/fleet.py off sol|g55|openai).
_DECIDER_LANE_GROUPS = {"sol": ("sol", "openai"), "base": ("g55", "openai")}


def _decider_lane_off(lane):
    off = fleet_models_off()
    return any(g in off for g in _DECIDER_LANE_GROUPS.get(lane, ()))


def _decider_model_order(limit=None):
    """Weighted-fair configured model allocation based on accepted calls, not task count."""
    limit = DECIDER_MAX_CONC if limit is None else limit
    active = {lane: sum(1 for job in DECIDE_JOBS if job.get("model_lane") == lane)
              for lane in DECIDER_MODEL_SHARES}
    nowt = time.time()
    eligible = [lane for lane in DECIDER_MODEL_SHARES
                if DECIDER_MODEL_SHARES[lane] > 0
                and DECIDER_MODEL_BLOCKED_UNTIL[lane] <= nowt
                and not _decider_lane_off(lane)]
    total_share = sum(DECIDER_MODEL_SHARES[lane] for lane in eligible)
    def available(lane):
        if DECIDER_MODEL_SHARES[lane] <= 0 or DECIDER_MODEL_BLOCKED_UNTIL[lane] > nowt \
                or _decider_lane_off(lane):
            return False
        cap = max(1, int(limit * DECIDER_MODEL_SHARES[lane]
                         / max(total_share, 1e-9) + .999))
        return active[lane] < cap
    return _share_order(DECIDER_MODEL_SHARES, active, DECIDER_MODEL_ACCEPT_COUNTS,
                        available)


def _decider_route(t, row, provider, difficulty=None, natt=None, model_lane="base"):
    difficulty = difficulty or _task_difficulty(t, row)
    target = required_capability(difficulty["score"], "decider")
    if model_lane == "sol":
        model_id = os.environ.get("DECIDER_SOL_MODEL", "gpt-5.6-sol")
        effort = os.environ.get("DECIDER_SOL_EFFORT", "xhigh")
    else:
        model_id = os.environ.get("DECIDER_BASE_MODEL", "gpt-5.5")
        effort = os.environ.get("DECIDER_BASE_EFFORT", "xhigh")
    display = model_id
    measured = model_benchmark(model_id, effort)
    return {"provider": provider, "model_id": model_id, "effort": effort,
            "label": f"{display}/{effort}", "difficulty": difficulty,
            "required_capability": target, "benchmark_score": measured,
            "capability_gap": None if measured is None else measured - target}


def decider_async(t, row, provider=None, difficulty=None, model_lane="base"):
    pack_dir = os.path.join(ROOT, "packs", f"task{t}")
    stage_kaggle_reference(t, pack_dir)
    prompt, natt = build_decider_prompt(t, row)
    if not prompt:
        return
    try:  # save-everything: the decider's complete evidence record
        open(os.path.join(RESULTS, f"task{t}",
                          f"DECISION_att{natt:02d}_input.md"), "w").write(prompt)
    except OSError:
        pass
    provider = provider or next(iter(_decider_provider_order()), None)
    if provider is None:
        DECIDE_WAIT[t] = row
        return
    route = _decider_route(t, row, provider, difficulty=difficulty, natt=natt,
                           model_lane=model_lane)
    cmd = ["codex", "exec", "-m", route["model_id"],
           "-c", f'model_reasoning_effort="{route["effort"]}"',
           # deciders ride the same FAST/priority tier opt-in as builders
           *(["-c", 'service_tier="priority"']
             if route["model_id"] in CODEX_FAST_MODELS else []),
           "--dangerously-bypass-approvals-and-sandbox", prompt]
    label = route["label"]
    final = os.path.join(RESULTS, f"task{t}", f"DECISION_att{natt:02d}.md")
    tmp = final + ".tmp"
    out = open(tmp, "w")
    out.write(f"# decider for att{natt:02d} [{label}] difficulty="
              f"{route['difficulty']['score']}/{route['difficulty']['tier']} "
              f"required-capability={route['required_capability']} "
              f"model-benchmark={route['benchmark_score'] or 'unrated'}\n")
    out.flush()
    try:
        p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.DEVNULL,
                             stdin=subprocess.DEVNULL, text=True)
    except Exception:
        out.close()
        try:
            os.remove(tmp)
        except OSError:
            pass
        if model_lane == "sol":
            DECIDER_MODEL_BLOCKED_UNTIL["sol"] = time.time() + 1800
            DECIDER_FORCE_BASE.add(t)
            DECIDER_SOL_FALLBACKS["n"] += 1
        DECIDE_WAIT[t] = row
        raise
    DECIDER_SPAWN_COUNTS[provider] += 1
    diff = route["difficulty"]
    live_yield = _runtime_yield(t, row, "codex", diff)
    DECIDE_JOBS.append({"p": p, "f": out, "tmp": tmp, "final": final,
                        "t0": time.time(), "t": t, "row": row,
                        "provider": provider, "model_id": route["model_id"],
                        "model_lane": model_lane,
                        "effort": route["effort"], "label": label,
                        "difficulty": diff["tier"], "difficulty_score": diff["score"],
                        "difficulty_reasons": diff["reasons"],
                        "required_capability": route["required_capability"],
                        "benchmark_score": route["benchmark_score"],
                        "capability_gap": route["capability_gap"],
                        "runtime_yield": live_yield})
    print(f"[{now()}] decider spawned for task{t} att{natt} ({label}; "
          f"difficulty={diff['score']}/{diff['tier']} → capability "
          f"{route['required_capability']}, benchmark={route['benchmark_score'] or 'unrated'}, "
          f"yield={live_yield['points_per_hour']:.4f} pts/h)")


PS_JOBS = []  # price_search stage: COST-failed tasks get a harness-priced design table
              # BEFORE the decider, so FAMILY choices are grounded in verified arithmetic


def decider_pending(t):
    return (t in DECIDE_WAIT or any(j["t"] == t for j in DECIDE_JOBS)
            or any(j["t"] == t for j in PS_JOBS))


# spawn caps (audit 2026-07-08): the 07:21 resume seeded 199 deciders and run_deciders
# spawned them ALL simultaneously — 199 concurrent `codex exec` processes; the orchestrator
# died silently 25 min later. DECIDE_WAIT is a queue now, drained a few at a time.
DECIDER_MAX_CONC = int(os.environ.get("DECIDER_MAX_CONC", 6))
PS_MAX_CONC = int(os.environ.get("PS_MAX_CONC", 3))
# STOP-LOSS (audit 2026-07-08): real-attempt conversion peaks at att2-3 (~27%) and craters
# to <9% at att5+; 37 unbanked tasks × ≥4 attempts burned ~63 h/run for zero. A task with
# >= STOP_LOSS_ATTS REAL attempts and no banked win is PARKED (long cooldown) rather than
# re-ground every few hours. Park, not abandon — the wave stays resumable and a rare deep
# att5-6 win is only deferred, never lost. Tunable; 0 disables.
STOP_LOSS_ATTS = int(os.environ.get("STOP_LOSS_ATTS", 5))
STOP_LOSS_PARK = int(os.environ.get("STOP_LOSS_PARK", 86400))  # 24 h
# High-headroom exemption (wave7 audit 2026-07-14): 146 tasks were parked 24 h on the
# "<9% at att5+" premise while the measured wave7 att7-10/att11+ hit rates were
# 20.0%/19.7%, and the parked set included edelta 4.5/3.1/3.0 tasks — multi-point
# modeled headroom idle for a day. Tasks whose DIFFICULTY edelta ≥ EXEMPT threshold
# park for HIGHEV_PARK (4 h) instead. STOP_LOSS_EXEMPT_EDELTA=999 restores old behavior.
STOP_LOSS_EXEMPT_EDELTA = float(os.environ.get("STOP_LOSS_EXEMPT_EDELTA", 2.0))
STOP_LOSS_HIGHEV_PARK = int(os.environ.get("STOP_LOSS_HIGHEV_PARK", 14400))  # 4 h


def _stop_loss_park_secs(t):
    """Park duration for a stop-lossed task: shorter when big modeled headroom remains."""
    try:
        from task_difficulty import _static_row
        edelta = float(_static_row(RESULTS, t).get("edelta") or 0)
        if edelta >= STOP_LOSS_EXEMPT_EDELTA:
            return STOP_LOSS_HIGHEV_PARK, edelta
    except Exception:
        pass
    return STOP_LOSS_PARK, None
# HOT RETRY (2026-07-11, flag-gated A/B, default OFF): on the hot arm a failed attempt
# skips the retry cooldown so decider→retry lands inside the prompt-cache retention window
# (~5–10 min) — with the cache-first prompt ordering, the static+per-task prefix is then a
# warm ~10%-price cache read instead of a cold full prefill. Slot economics cap the burst:
# att2-3 convert ~27% but att5+ <9%, so only attempts up to HOT_RETRY_MAX_ATT qualify (the
# analyst+decider gates still apply — a hot retry is early, never unbriefed). Arms are a
# deterministic task hash; finish() stamps `hot` on each attempt rec for the comparison.
HOT_RETRY = os.environ.get("HOT_RETRY", "0") == "1"
HOT_RETRY_MAX_ATT = int(os.environ.get("HOT_RETRY_MAX_ATT", 3))
HOT_RETRY_WINDOW = int(os.environ.get("HOT_RETRY_WINDOW", 1200))
HOT_RETRY_SHARE = float(os.environ.get("HOT_RETRY_SHARE", "0.5"))
HOT_READY = {}  # task -> epoch it became hot-eligible; drives the codex pick preference


def _hot_arm(task):
    if not HOT_RETRY or HOT_RETRY_SHARE <= 0:
        return False
    h = int(hashlib.md5(f"hot:{task}".encode()).hexdigest(), 16) % 1000
    return h < int(HOT_RETRY_SHARE * 1000)


def _pick_decider_index(ready, provider):
    """Spend Sol on the hardest task; baseline GPT on best live seat-hour ROI."""
    if provider != "sol":
        return max(range(len(ready)), key=lambda i: (
            _runtime_yield(ready[i][0], ready[i][1], "codex", ready[i][2])[
                "points_per_hour"],
            -i,
        ))
    return max(range(len(ready)), key=lambda i: ready[i][2]["score"])


def _decider_concurrency_limit(running):
    """Codex decisions borrow only seats not occupied by Codex builders — unless the
    operator runs the decider pool independent of builder seats (DECIDER_POOL_INDEPENDENT=1,
    final-sprint mode: 24 always-full builder seats would otherwise starve deciders to 0)."""
    # Live per-provider counts (fleet.py deciders …) also set the TOTAL pool size, so the
    # launch-time DECIDER_MAX_CONC never caps an operator who scales the pool up mid-run.
    live = fleet_deciders()
    live_total = sum(live.values()) if live else None
    if os.environ.get("DECIDER_POOL_INDEPENDENT", "") == "1":
        return live_total if live_total is not None else DECIDER_MAX_CONC
    desired = fleet_desired()
    if desired is None:
        return live_total if live_total is not None else DECIDER_MAX_CONC
    builders = sum(1 for job in running if job.get("model") == "codex")
    return min(DECIDER_MAX_CONC, max(0, desired.get("codex", 0) - builders))


def run_deciders(running=None):
    running = running or []
    decider_limit = _decider_concurrency_limit(running)
    if len(DECIDE_JOBS) >= decider_limit:
        return
    if not _decider_provider_order(decider_limit):
        return
    ready = []
    for t, row in list(DECIDE_WAIT.items()):
        if pm_pending(t):
            continue  # the decider consumes the analyst brief — wait for it
        all_atts = load_attempts(t)
        natt = len(all_atts) + 1          # file numbering stays on the FULL count
        clean_atts = [a for _, a in current_attempt_pairs(all_atts)]
        atts = real_atts(clean_atts)      # decider evidence is fixed-scorer-era only
        if not atts and not _current_rejections(t):
            # every recorded slot was an infra death — nothing to audit or decide;
            # the build prompt already frames this as a FIRST attempt (big swing)
            DECIDE_WAIT.pop(t)
            continue
        last_fr = (atts[-1].get("fail_reason") or "") if atts else ""
        prices = os.path.join(RESULTS, f"task{t}", f"PRICES_att{natt:02d}.md")
        cost_class = bool(re.search(r"COST-REJECT|Do not gate|< \+0\.\d+|micro-trim", last_fr))
        if cost_class and not os.path.exists(prices) and not _blocked("admin"):
            if len(PS_JOBS) >= PS_MAX_CONC:
                continue  # stays in DECIDE_WAIT; retried next tick
            DECIDE_WAIT.pop(t)
            # 40% failure class: budget never closed — get 15+ REPRICED designs first
            out = open(prices + ".tmp", "w")
            p = subprocess.Popen([sys.executable, os.path.join(HERE, "price_search.py"),
                                  str(int(t)), "(see verified rule.py / strategy card in inputs)"],
                                 stdout=out, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, text=True)
            PS_JOBS.append({"p": p, "f": out, "tmp": prices + ".tmp", "final": prices,
                            "t0": time.time(), "t": t, "row": row})
            print(f"[{now()}] price_search spawned for task{t} (COST-failed — grounding "
                  f"the decider in harness-priced designs)")
            continue
        ready.append((t, row, _task_difficulty(t, row)))

    # Provider is Codex-only. Model-level weighted fairness sends exactly the Sol share
    # toward the hardest currently-ready work; GPT-5.5 preserves FIFO for the remainder.
    while ready and len(DECIDE_JOBS) < decider_limit:
        providers = _decider_provider_order(decider_limit)
        if not providers:
            break
        provider = providers[0]
        forced = next((i for i, item in enumerate(ready) if item[0] in DECIDER_FORCE_BASE), None)
        if forced is not None:
            model_lane, pick = "base", forced
        else:
            lanes = _decider_model_order(decider_limit)
            if not lanes:
                break
            model_lane = lanes[0]
            pick = _pick_decider_index(ready, model_lane)
        t, row, difficulty = ready.pop(pick)
        DECIDER_FORCE_BASE.discard(t)
        DECIDE_WAIT.pop(t, None)
        decider_async(t, row, provider=provider, difficulty=difficulty,
                      model_lane=model_lane)


def reap_price_searches():
    for j in PS_JOBS[:]:
        rc = j["p"].poll()
        # 720 s: price_search's internal admin call runs xhigh (600 s inner timeout)
        if rc is None and time.time() - j["t0"] > int(os.environ.get("PS_KILL_SECS", 720)):
            j["p"].kill(); rc = -9
        if rc is None:
            continue
        PS_JOBS.remove(j); j["f"].close()
        try:
            # keep a marker file EITHER way (real table >100B, or the tiny failure output):
            # run_deciders re-triggers a fresh price_search only when `not exists(prices)`,
            # so on a re-queue (cap full) an empty marker prevents a re-drain loop (T10)
            os.replace(j["tmp"], j["final"])
        except OSError:
            pass
        # Re-enter the shared queue so the allocator can route the hardest ready task to
        # the strongest lane. Direct spawning here used to bypass both share and priority.
        DECIDE_WAIT[j["t"]] = j["row"]


_BYTES = {"u8": 1, "uint8": 1, "i8": 1, "int8": 1, "b": 1, "bool": 1, "f16": 2,
          "float16": 2, "f32": 4, "float32": 4, "i32": 4, "int32": 4, "u32": 4,
          "uint32": 4, "i64": 8, "int64": 8}


_EXPLORATORY_RE = re.compile(r"(?im)^PRICED-EXPLORATORY-BUILD:\s*YES\b")


def _price_decision_budget(text, row):
    """Return (READY|EXPLORATORY|REVISE, cost, reason) from a decision's exact budget."""
    m = re.search(r"FAMILY-BUDGET-JSON:\s*(\{.*?\})\s*(?:\n|$)", text, re.S)
    if not m:
        return "REVISE", None, "missing parsable FAMILY-BUDGET-JSON"
    try:
        data = json.loads(m.group(1))
        if not isinstance(data, dict) or not isinstance(data.get("tensors", []), list):
            raise ValueError("budget must be an object with a tensors list")
        params = int(data.get("params", 0))
        if params < 0:
            raise ValueError("params must be non-negative")
        cost = params
        for tensor in data.get("tensors", []):
            if not isinstance(tensor, dict):
                raise ValueError("each tensor must be an object")
            count = int(tensor.get("count", 1))
            if count <= 0:
                raise ValueError("tensor count must be positive")
            n = 1
            shape = tensor.get("shape", [])
            if not isinstance(shape, list):
                raise ValueError("tensor shape must be a list")
            for dim in shape:
                dim = int(dim)
                if dim <= 0:
                    raise ValueError("tensor dimensions must be positive")
                n *= dim
            dtype = str(tensor.get("dtype", "")).lower()
            cost += n * _BYTES[dtype] * count
        bar = int(row.get("bar_cost") or 0)
        if bar <= 0:
            raise ValueError("decision target bar is missing")
    except (KeyError, TypeError, ValueError) as exc:
        return "REVISE", None, f"malformed FAMILY-BUDGET-JSON ({str(exc)[:80]})"
    if cost <= bar:
        return "READY", cost, f"priced at {cost} B vs bar {bar}; closes"
    if _EXPLORATORY_RE.search(text):
        return "EXPLORATORY", cost, (f"priced at {cost} B vs bar {bar}; explicit priced "
                                      "exploratory build")
    return "REVISE", cost, f"priced at {cost} B vs bar {bar}; does not close"


def _observed_min_cost(t):
    """Minimum MEASURED cost for this task from prior attempt evidence: `cost=N` lines the
    pricer printed in attempt transcripts (worker-output section only — the prompt echo
    contains historical numbers) plus ACCEPTANCE.jsonl receipts. Min is deliberate: cheap
    wrong graphs drag the floor DOWN, so the observed-floor gate fires less (fail-open)."""
    import glob as _g
    costs = []
    for p in _g.glob(os.path.join(RESULTS, f"task{t}", "attempts", "att*_*.log")):
        try:
            if not file_is_current(p):
                continue
            txt = open(p, errors="replace").read().split("=== WORKER OUTPUT ===", 1)[-1]
            costs += [int(c) for c in re.findall(r"\bcost=(\d+)\b", txt)]
        except (OSError, ValueError):
            continue
    acc = os.path.join(RESULTS, f"task{t}", "ACCEPTANCE.jsonl")
    try:
        for ln in open(acc, errors="replace"):
            try:
                c = json.loads(ln).get("cost")
                if c:
                    costs.append(int(c))
            except (ValueError, TypeError):
                continue
    except OSError:
        pass
    costs = [c for c in costs if c > 0]
    return min(costs) if costs else None


_RECONCILE_NAMES_RE = re.compile(r"(?i)\b(?:eliminat\w+|remov\w+|drops?|kills?|delet\w+)\b")


def _stamp_decision_budget(path, row):
    """Harness-price the decider's FAMILY-BUDGET-JSON and stamp ✓/✗ on the decision file —
    novel designs stay unrestricted, but their arithmetic is verified before a worker
    spends hours on them."""
    with open(path, errors="replace") as f:
        text = f.read()
    status, cost, reason = _price_decision_budget(text, row)
    # Observed-floor gate (wave7 audit 2026-07-14): deciders still stamped CLOSES ✓ on
    # same-family plans that repriced 1.5-5.9× over paper (task090 SWAR: 1793 B claimed,
    # 2754 measured). When a CONTINUE-same-family plan prices below HALF the cheapest
    # cost this task has ever MEASURED, without naming what it eliminates (the RECONCILE
    # contract), the CLOSES endorsement is almost certainly optimistic enumeration —
    # downgrade to EXPLORATORY. Non-blocking by construction: EXPLORATORY still spawns
    # the worker; pivot/reset/fallback (novel families) are never compared to old floors.
    if status == "READY" and cost:
        mdec = re.search(r"DECISION:\s*continue-same-family", text)
        mtask = re.search(r"task(\d{3})", os.path.basename(os.path.dirname(path)) or "")
        fam_section = text.split("FAMILY:", 1)[-1].split("PRICED-EXPLORATORY-BUILD:", 1)[0]
        if mdec and mtask and not _RECONCILE_NAMES_RE.search(fam_section):
            floor = _observed_min_cost(mtask.group(1))
            if floor and cost < floor / 2:
                status = "EXPLORATORY"
                reason += (f"; paper {cost} B is <½ the observed measured floor {floor} B "
                           f"for this same-family plan with no eliminated tensor named — "
                           f"auto-downgraded CLOSES→EXPLORATORY")
    if status == "READY":
        verdict = "CLOSES ✓"
    elif status == "EXPLORATORY":
        verdict = "DOES NOT CLOSE ✗; PRICED EXPLORATORY BUILD ✓"
    else:
        verdict = "REVISE — no worker authorized"
    with open(path, "a") as f:
        f.write(f"\n[harness: decision budget {reason} — {verdict}; status={status}]")
    return status, cost, reason


def _decision_budget_status(path):
    try:
        with open(path, errors="replace") as f:
            text = f.read()
    except OSError:
        return "REVISE", "decision file is unreadable"
    statuses = re.findall(r"status=(READY|EXPLORATORY|REVISE)", text)
    if not statuses:
        return "REVISE", "decision has no harness budget authorization stamp"
    status = statuses[-1]
    stamp = text.rsplit("[harness:", 1)[-1].split("]", 1)[0]
    return status, stamp[:300]


def reap_deciders():
    for j in DECIDE_JOBS[:]:
        rc = j["p"].poll()
        # 1200 s: deciders at high effort need headroom beyond the old 900 s budget
        if rc is None and time.time() - j["t0"] > int(os.environ.get("DECIDER_KILL_SECS", 1200)):
            j["p"].kill(); rc = -9
        if rc is None:
            continue
        DECIDE_JOBS.remove(j); j["f"].close()
        lane = j.get("model_lane", "base")
        accepted = False
        try:
            if rc == 0 and os.path.getsize(j["tmp"]) > 150:
                accepted = True
                DECIDER_MODEL_ACCEPT_COUNTS[lane] += 1
                os.replace(j["tmp"], j["final"])
                status, _, budget_reason = _stamp_decision_budget(
                    j["final"], j.get("row", {}))
                violations = _policy_violations([j["final"]])
                if violations:
                    bad = violations[0][1]
                    reason = "; ".join(f"line {n}: {line}" for n, line in bad)
                    print(f"[{now()}] INTEGRITY POLICY: rejecting decider for task{j['t']} — "
                          + reason)
                    _quarantine_file(j["t"], j["final"], reason, "policy")
                    DECIDE_WAIT[j["t"]] = j["row"]
                elif status == "REVISE":
                    print(f"[{now()}] decision for task{j['t']} requires REVISE — "
                          f"{budget_reason}; no worker attempt spent")
                    _quarantine_file(j["t"], j["final"], budget_reason,
                                     "budget-revise")
                    DECIDE_WAIT[j["t"]] = j["row"]
            else:
                os.remove(j["tmp"])
        except OSError:
            pass
        if not accepted:
            DECIDER_MODEL_FAILURES[lane] += 1
            if lane == "sol":
                # Isolate Sol infrastructure trouble. The same task is retried immediately
                # on GPT-5.5 and cannot reach a builder without a valid decision.
                DECIDER_MODEL_BLOCKED_UNTIL["sol"] = time.time() + 1800
                DECIDER_FORCE_BASE.add(j["t"])
                DECIDER_SOL_FALLBACKS["n"] += 1
                print(f"[{now()}] Sol decider failed for task{j['t']} — Sol benched 30 min; "
                      "same decision requeued on gpt-5.5/xhigh")
            DECIDE_WAIT[j["t"]] = j["row"]


# ---- watchdog (Codex Spark): mid-attempt progress classification -------------------
# Everything else in the system fires AFTER an attempt ends; a worker stuck in a loop
# burns its whole 1-3.5h timeout unobserved. Every WATCH_INTERVAL the watchdog compares
# consecutive transcript tails: unchanged tail = SILENT strike (free); changed tail goes
# to a cheap admin-model classification (PROGRESSING/LOOPING/OFF-TASK/BLOCKED). Two consecutive
# non-progressing strikes ⇒ early SIGTERM through the normal interruption path (salvage +
# interruption-aware analyst + decider) — the wasted hours become extra attempts.
WATCH_INTERVAL = int(os.environ.get("WATCH_INTERVAL", 600))
WATCH_WARMUP = int(os.environ.get("WATCH_WARMUP", 900))
WATCH_JOBS = []

WATCH_TPL = """SINGLE-RESPONSE TASK. You have NO tools; your FIRST message IS the final answer.
You are a run watchdog. Below are two consecutive tail excerpts (~10 min apart) of a LIVE
build-agent transcript working on NeuroGolf task {t}. Classify the current state:
PROGRESSING — new meaningful actions/results since the previous tail
LOOPING — the same command/error/design repeating with no material change
OFF-TASK — working on something other than task {t}'s build/verify loop
BLOCKED — an environment/tool failure repeating (gate crash, missing file, auth error)
COST-DOOMED — the transcript EXPLICITLY shows the design's priced/measured total is
already ≥2× the register bar of {bar} B and the agent keeps building it anyway (never
guess a cost yourself; only quote a total the transcript states)
Reply with EXACTLY one line: VERDICT: <word> — <reason, ≤15 words>.
=== PREVIOUS TAIL ===
{prev}
=== CURRENT TAIL ===
{cur}"""


WATCH_MAX_LOOPING = int(os.environ.get("WATCH_MAX_LOOPING", 4))


def _activity_evidence(job):
    """Reason the attempt is demonstrably ACTIVE despite a LOOPING/BLOCKED verdict, or None.

    Wave7 audit (2026-07-14): content-verdict LOOPING had no activity cross-check, so slow
    gates were false-flagged (task009/022/346 all won minutes after a LOOPING verdict)
    while genuinely looping attempts survived via interleaved PROGRESSING resets (task233,
    ~140 min). Evidence checked: a live gate/pricer descendant process, or fresh writes in
    the task's pack/results dirs within the last watch interval. stdout growth is
    deliberately NOT evidence — a loop also prints."""
    t = job["row"]["task"][4:]
    try:  # 2-level child walk via /proc (ps-free; the box is Linux)
        pids = [str(job["proc"].pid)]
        for _depth in range(2):
            kids = []
            for pid in pids:
                try:
                    kids += open(f"/proc/{pid}/task/{pid}/children").read().split()
                except OSError:
                    pass
            for kid in kids:
                try:
                    cmd = open(f"/proc/{kid}/cmdline", "rb").read().decode(
                        "utf-8", "replace").replace("\x00", " ")
                except OSError:
                    continue
                if re.search(r"fast_verify|submit_result|gate_vs_pin|ngolf\b|scan_razors",
                             cmd):
                    return f"live gate/pricer child ({cmd.strip()[:60]})"
            pids = kids
            if not pids:
                break
    except Exception:
        pass
    nowt = time.time()
    for d in (os.path.join(ROOT, "packs", f"task{t}"),
              os.path.join(RESULTS, f"task{t}")):
        try:
            mt = max((os.path.getmtime(os.path.join(d, f)) for f in os.listdir(d)),
                     default=0)
        except OSError:
            continue
        if nowt - mt < WATCH_INTERVAL:
            return f"fresh writes in {os.path.basename(d)} ({int(nowt - mt)}s ago)"
    return None


def _watch_policy(job):
    s = job.get("watch_strikes", [])
    t = job["row"]["task"][4:]
    # cumulative UNVETOED LOOPING cap per attempt: PROGRESSING resets the consecutive
    # strikes but not this counter, so a task233-class slow loop (LOOPING → PROGRESSING →
    # LOOPING …) dies at ~4 intervals instead of running 140 min. Vetoed verdicts never
    # count (they are the protected slow-gate class).
    if (job.get("watch_loops", 0) >= WATCH_MAX_LOOPING and "term_at" not in job
            and WATCH_MAX_LOOPING > 0):
        print(f"[{now()}] watchdog: task{t} [{job['model']}] {job['watch_loops']} unvetoed "
              f"LOOPING verdicts this attempt — hard kill (LOOPING-cap)")
        job["term_at"] = time.time() - GRACE_SECS  # no grace: ≥40 min of loop evidence
        job["interrupted"] = True
        job["watch_kill"] = "LOOPING-cap"
        job["proc"].kill()
        return
    need = 2
    if len(s) >= need and "term_at" not in job:
        print(f"[{now()}] watchdog: task{t} [{job['model']}] {'+'.join(s[-2:])} twice — "
              f"early terminate (salvage + analyst + decider take over)")
        job["term_at"] = time.time()
        job["interrupted"] = True
        job["watch_kill"] = "+".join(s[-2:])
        job["proc"].terminate()


BLOCKED_LOG = []  # (ts, task) — fleet-wide environment-failure detector


def watchdog_tick(running):
    nowt = time.time()
    for job in running:
        if nowt - job["t0"] < WATCH_WARMUP or job.get("term_at"):
            continue
        if nowt - job.get("watch_t", job["t0"]) < WATCH_INTERVAL:
            continue
        try:
            txt = open(job["log"].name, errors="replace").read().split(
                "=== WORKER OUTPUT ===", 1)[-1]
        except OSError:
            continue
        cur = txt[-3500:]
        prev = job.get("watch_tail")
        job["watch_t"] = nowt
        job["watch_tail"] = cur
        if prev is None:
            continue
        if cur == prev:  # zero new stdout for a whole interval — check DISK before judging:
            # a buffered-stdout worker is often busily writing build.py/rule.py
            d = os.path.join(ROOT, "packs", f"task{job['row']['task'][4:]}")
            try:
                mt = max((os.path.getmtime(os.path.join(d, f)) for f in os.listdir(d)),
                         default=0)
            except OSError:
                mt = 0
            if mt > job.get("watch_disk", 0):
                job["watch_disk"] = mt
                job["watch_strikes"] = []  # silent on stdout but productive on disk
                continue
            job["watch_disk"] = max(mt, job.get("watch_disk", 0))
            job.setdefault("watch_strikes", []).append("SILENT")
            _watch_policy(job)
            continue
        if _blocked("admin"):  # admin route benched — local SILENT/disk checks still work
            continue          # local SILENT/disk-mtime strike logic above keeps working
        prompt = WATCH_TPL.format(t=job["row"]["task"][4:], prev=prev[-1800:], cur=cur[-1800:],
                                  bar=job["row"].get("bar_cost") or "?")
        cmd = _admin_cmd(prompt, os.environ.get("ADMIN_WATCH_EFFORT", "low"), 180)
        pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              stdin=subprocess.DEVNULL, text=True)
        WATCH_JOBS.append({"p": pr, "job": job, "t0": nowt})


def reap_watchdog(running):
    for w in WATCH_JOBS[:]:
        rc = w["p"].poll()
        # Administrative classification must complete well inside the 10-minute cadence.
        if rc is None and time.time() - w["t0"] > int(os.environ.get("WATCH_KILL_SECS", 360)):
            w["p"].kill(); rc = -9
        if rc is None:
            continue
        WATCH_JOBS.remove(w)
        out = (w["p"].stdout.read() or "") if w["p"].stdout else ""
        m = re.search(r"VERDICT:\s*([A-Z-]+)", out.upper())
        v = m.group(1) if m else None
        job = w["job"]
        if job not in running:
            continue  # attempt already finished — verdict moot
        if v == "PROGRESSING":
            job["watch_strikes"] = []
            try:  # log resets too — wave7's task233 audit was unreadable without them
                with open(os.path.join(RESULTS, f"task{job['row']['task'][4:]}",
                                       "attempts", "watchdog.jsonl"), "a") as wf:
                    wf.write(json.dumps({"ts": now(), "att": job["att"],
                                         "model": job["model"],
                                         "verdict": out.strip()[:200]}) + "\n")
            except OSError:
                pass
        elif v == "COST-DOOMED":
            # LOG-ONLY (no strike, no interrupt): the verdict lands in watchdog.jsonl and
            # watch_log, flows into the attempt's fail_reason quotes, and warns the
            # decider — but a misread cost table must never kill a live worker.
            job.setdefault("watch_log", []).append(out.strip()[:140])
            try:
                with open(os.path.join(RESULTS, f"task{job['row']['task'][4:]}",
                                       "attempts", "watchdog.jsonl"), "a") as wf:
                    wf.write(json.dumps({"ts": now(), "att": job["att"],
                                         "model": job["model"],
                                         "verdict": out.strip()[:200]}) + "\n")
            except OSError:
                pass
            print(f"[{now()}] watchdog task{job['row']['task'][4:]} (advisory): "
                  f"{out.strip()[:110]}")
        elif v:
            veto = _activity_evidence(job) if v in ("LOOPING", "BLOCKED") else None
            if veto:
                # slow gate / live pricer misread as a loop: no strike, no loop count —
                # the protected class (task009/022/346 won minutes after such verdicts)
                job.setdefault("watch_log", []).append(f"{v}(vetoed:{veto})"[:140])
                job["watch_loops_vetoed"] = job.get("watch_loops_vetoed", 0) + 1
            else:
                job.setdefault("watch_strikes", []).append(v)
                job.setdefault("watch_log", []).append(out.strip()[:140])
                if v == "LOOPING":
                    job["watch_loops"] = job.get("watch_loops", 0) + 1
            try:  # save-everything: watchdog verdict stream (vetoes included)
                with open(os.path.join(RESULTS, f"task{job['row']['task'][4:]}",
                                       "attempts", "watchdog.jsonl"), "a") as wf:
                    wf.write(json.dumps({"ts": now(), "att": job["att"],
                                         "model": job["model"],
                                         "vetoed": veto or None,
                                         "verdict": out.strip()[:200]}) + "\n")
            except OSError:
                pass
            print(f"[{now()}] watchdog task{job['row']['task'][4:]}: "
                  + (f"{v} VETOED ({veto})" if veto else out.strip()[:110]))
            if veto:
                continue
            _watch_policy(job)
            if v == "BLOCKED":
                BLOCKED_LOG.append((time.time(), job["row"]["task"][4:]))
                recent = {tk for ts, tk in BLOCKED_LOG if time.time() - ts < 1800}
                if len(recent) >= 3:
                    print(f"[{now()}] WATCHDOG ALERT: {len(recent)} distinct workers "
                          f"BLOCKED within 30 min ({' '.join(sorted(recent))}) — possible "
                          f"fleet-wide environment failure (gate venv? disk?). Investigate.")


def _acceptance_for_attempt(t, started_at):
    """Newest exact artifact receipt created by this attempt, plus its on-disk path."""
    import glob as _g
    path = os.path.join(RESULTS, f"task{t}", "ACCEPTANCE.jsonl")
    receipts = []
    try:
        for line in open(path, errors="replace"):
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if (int(rec.get("task", -1)) == int(t)
                    and float(rec.get("accepted_epoch", 0)) >= started_at):
                receipts.append(rec)
    except OSError:
        return None, None
    for rec in sorted(receipts, key=lambda x: float(x.get("accepted_epoch", 0)), reverse=True):
        for artifact in ([os.path.join(RESULTS, f"task{t}", f"task{t}.onnx")]
                         + _g.glob(os.path.join(RESULTS, f"task{t}", f"task{t}__*.onnx"))):
            try:
                if hashlib.sha256(open(artifact, "rb").read()).hexdigest() == rec.get("sha256"):
                    return rec, artifact
            except OSError:
                continue
    return None, None


def finish(job):
    t = job["row"]["task"][4:]
    job["log"].close()
    txt = open(job["log"].name, errors="replace").read()
    # the log echoes the prompt header, which contains the literal sentinels
    # "DONE taskNNN"/"NO taskNNN" — classify on WORKER OUTPUT only, never the echo
    txt = txt.split("=== WORKER OUTPUT ===", 1)[-1]
    outcome, delta, fail = "NONE", None, ""
    accepted_artifact = None
    # sentinels count ONLY in the transcript's final chars: workers grep the June trace
    # archive, whose files CONTAIN literal "DONE taskNNN +Δ" lines — first run: 20/20
    # sampled DONE_NO_ARTIFACT were such mid-transcript echoes, zero genuine claims
    tail600 = txt[-600:]
    m = DONE_RE.search(tail600)
    if m:
        receipt, onnx_p = _acceptance_for_attempt(t, job["t0"])
        if m.group(1) != t:
            outcome, fail = "DONE_UNVERIFIED", f"DONE sentinel named task{m.group(1)}, expected task{t}"
        elif receipt is None:
            outcome, fail = "DONE_UNVERIFIED", "no exact acceptance receipt created by this attempt"
        else:
            outcome, delta = "DONE", float(receipt["delta"])
            accepted_artifact = onnx_p
            # run gate on every win (score_v2 gate): full gate (not just price) for the
            # exact receipt-bound delivered ONNX. This enforces "run gate on every win".
            # NOTE: LEDGER banking already happened inside the worker's submit_result run
            # behind submit_result's OWN full gate — this inline re-check classifies the
            # ATTEMPT (DONE vs DONE_UNVERIFIED) and gates exemplar/LESSONS propagation.
            _sha_now = receipt.get("sha256")
            if _sha_now and _sha_now in VERIFIED_SHAS:
                pass  # byte-identical artifact already re-gated this run (house-money re-win)
            else:
                try:  # 330s: score_v2 inside fast_verify has its own 240s cap; margin for
                      # lint+price. (finish() runs synchronously in the reap loop — a tight
                      # cap bounds loop stalls at 24 seats; median observed gate ≈60s.)
                    v = subprocess.run([sys.executable, os.path.join(HERE, "fast_verify.py"),
                                        str(int(t)), onnx_p, "--full", "--bar",
                                        "0.15"],
                                       capture_output=True, text=True, timeout=330)
                    if v.returncode != 0:
                        outcome, fail = "DONE_UNVERIFIED", v.stdout.strip()[-300:]
                    elif _sha_now:
                        VERIFIED_SHAS.add(_sha_now)
                except subprocess.TimeoutExpired:
                    outcome, fail = "DONE_UNVERIFIED", "full gate timed out (330s) — unverified"
    elif DUP_RE.search(tail600):
        outcome, fail = "DUPLICATE", "byte-identical to an already-banked member"
    elif NO_RE.search(tail600):
        outcome = "NO"
    if outcome == "NONE" and job["proc"] and (time.time() - job["t0"]) < 90 \
            and len(txt.strip()) < 200 and not QUOTA_RE.search(txt[-1500:]):
        # instant startup crash with no quota signature (CLI/env hiccup): re-queue,
        # record nothing — but only 3 free passes per task, then it counts as a failure
        SPAWN_FAILS[t] = SPAWN_FAILS.get(t, 0) + 1
        INFRA_REQUEUES[job["model"]] = INFRA_REQUEUES.get(job["model"], 0) + 1
        if SPAWN_FAILS[t] <= 3:
            print(f"[{now()}] task{t} att{job['att']} {job['model']} → SPAWN_FAIL "
                  f"#{SPAWN_FAILS[t]} ({int(time.time()-job['t0'])}s, no output) — "
                  f"re-queued, no attempt recorded")
            return "QUOTA_BLOCKED"  # same re-queue path (no bench: model not blamed yet)
    if outcome == "NONE" and QUOTA_RE.search(txt[-1500:]):
        # model quota/usage-limit hit — NOT a task failure. Bench the model, re-queue the
        # task unchanged, record NOTHING (no attempt burned, no history polluted). The ladder
        # routes this tier's alt (a DIFFERENT-provider quota pool) while this one is benched.
        # No CLI exposes a proactive quota query, but the exhaustion error carries the EXACT
        # reset time ("...try again at 4:20 PM") — bench until THEN, not a blind 30 min, so a
        # provider is skipped for exactly as long as its quota is actually out (2026-07-10).
        secs, until = _bench_secs(txt)
        MODEL_BLOCKED_UNTIL[job["model"]] = time.time() + secs
        INFRA_REQUEUES[job["model"]] = INFRA_REQUEUES.get(job["model"], 0) + 1
        print(f"[{now()}] task{t} att{job['att']} {job['model']} → QUOTA_BLOCKED "
              f"(benched {int(secs/60)} min{' until '+until if until else ''}; task re-queued, "
              f"no attempt recorded) — will retry {job['model']} when the bench lifts")
        return "QUOTA_BLOCKED"
    if outcome == "NONE" and job.get("interrupted"):
        outcome = "TIMEOUT"
        if job.get("watch_kill"):
            # LOOPING kills record as NO, not TIMEOUT: a TIMEOUT triggers the same-config
            # provider resume (re-entering the looping session), and the WATCHDOG: prefix
            # excludes these from cost-floor evidence downstream (wave7 watchdog-v2).
            if "LOOPING" in str(job["watch_kill"]):
                outcome = "NO"
            fail = (f"WATCHDOG early-stop ({job['watch_kill']}): the run stopped making "
                    f"progress — verdicts: {'; '.join(job.get('watch_log', [])[-2:])}. "
                    f"Partial work remains in packs/task{t}/; the loop it was stuck in "
                    f"must NOT be repeated")
        else:
            fail = (f"interrupted at the {job['timeout']}s limit — partial work remains in "
                    f"packs/task{t}/ (drafts + transcript); CONTINUE from it, don't restart")
    if outcome not in ("DONE", "TIMEOUT") and not fail:
        # scan only the transcript TAIL with word-boundary FAIL: codex echoes the whole
        # prompt (which contains "...FAILED ATTEMPT..." headers) and workers cat archive
        # files — substring "FAIL" matched those, recording 217 garbage fail_reasons that
        # then poisoned every retry prompt and the price_search COST trigger (audit 07-08)
        tail = [l for l in txt[-4000:].splitlines()
                if re.search(r"COST-REJECT|\bFAIL\b|UNSCORABLE", l)
                and "FAILED ATTEMPT" not in l]
        fail = (tail[-1] if tail else txt[-300:])[:300]
    if outcome != "DONE":
        # SALVAGE: a worker killed at timeout (or one that forgot to submit) may have left
        # a PASSING artifact in its pack dir. Price it; if above bar, gate+record it now.
        import glob as _g
        cands = [c for c in [os.path.join(ROOT, "packs", f"task{t}", f"task{t}.onnx")]
                 + sorted(_g.glob(os.path.join(ROOT, "packs", f"task{t}", "*.onnx")),
                          key=lambda p: -os.path.getmtime(p))[:2]
                 if os.path.exists(c)]
        seen = set()
        cands = [c for c in cands if not (c in seen or seen.add(c))]
        cand = cands[0] if cands else os.path.join(ROOT, "packs", f"task{t}", f"task{t}.onnx")
        for c in cands:  # canonical first, then the 2 newest variants (workers strand them)
            if os.path.getmtime(c) >= job["t0"] - 60:
                cand = c
                break
        if os.path.exists(cand) and os.path.getmtime(cand) >= job["t0"] - 60:
            # Async salvage (wave7 audit 2026-07-14): the inline --price (600 s) ran on
            # EVERY non-DONE attempt with a fresh artifact, and the blocking full gate
            # yielded 3 banks for 112 gates across waves 6-7 — the reap loop paid minutes
            # per attempt for near-zero yield. The candidate is COPIED out of the pack
            # (a later attempt overwrites packs/) and handed to salvage_worker.py:
            # flock single-flight, nice -n 10, price → bar+headroom check → normal
            # submit_result bank, entirely off-loop. LEDGER is the source of truth for
            # any salvage win (no DONE_SALVAGED attempt record).
            try:
                import shutil as _sh
                _sha8 = hashlib.sha256(open(cand, "rb").read()).hexdigest()[:8]
                _sdir = os.path.join(RESULTS, f"task{t}", "attempts")
                os.makedirs(_sdir, exist_ok=True)
                _scopy = os.path.join(_sdir, f"salvage_att{job['att']}_{_sha8}.onnx")
                _sh.copy2(cand, _scopy)
                subprocess.Popen(["nice", "-n", "10", sys.executable,
                                  os.path.join(HERE, "salvage_worker.py"), str(int(t)),
                                  _scopy, str(job["row"]["lane"]), str(job["att"]),
                                  str(job["model"])],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 cwd=ROOT, start_new_session=True)
                print(f"[{now()}] task{t}: stranded artifact handed to async salvage "
                      f"({os.path.basename(_scopy)})")
            except OSError as exc:
                print(f"[{now()}] task{t}: salvage handoff failed ({exc}) — skipped")
    rec = {"ts": now(), "model": job["model"],
           "model_id": job.get("model_id"), "effort": job.get("effort"),
           "att": job["att"],
           "outcome": outcome, "delta": delta, "fail_reason": fail,
           "secs": int(time.time() - job["t0"])}
    if HOT_RETRY:  # A/B provenance: was THIS attempt spawned on the hot (no-cooldown) arm?
        rec["hot"] = bool(job.get("hot"))
    if RETRY_RESUME and job["model"] == "codex":
        sid = _find_build_session(t, job["t0"])
        if sid:  # lets the NEXT attempt resume this session (and MAX_AGE-gate on end time)
            rec["session_id"], rec["end_epoch"] = sid, int(time.time())
        if job.get("resumed_sid"):
            rec["resumed_sid"] = job["resumed_sid"]
    record_attempt(t, rec)
    # analyst on EVERY attempt, all models — INCLUDING interrupted/unfinished ones: the
    # partial transcript still teaches, and the brief gates + feeds the next attempt
    postmortem_async(t, job["model"], job["att"], outcome, job["log"].name,
                     interrupted=job.get("interrupted", False),
                     model_id=job.get("model_id"))
    if outcome in ("DONE", "DONE_SALVAGED"):
        # Fleet-wide propagation requires exact Kaggle payment proof. A local full gate is
        # valuable same-task evidence, but hidden-zero candidates must not train every task.
        from proven_corpus import payment_proof_for_path
        proof = (payment_proof_for_path(int(t), accepted_artifact)
                 if accepted_artifact and os.path.exists(accepted_artifact) else None)
        if proof:
            import shutil
            bp = os.path.join(ROOT, "packs", f"task{t}", "build.py")
            if os.path.exists(bp):
                shutil.copy(bp, os.path.join(ROOT, "data", "exemplars", f"won_task{t}_{job['model']}.py"))
            with open(LESSONS, "a") as f:
                f.write(f"- [{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d}]"
                        f"[task{t}/{job['model']}] KAGGLE-CONFIRMED +{delta}: see "
                        f"data/exemplars/won_task{t}_{job['model']}.py\n")
        else:
            print(f"[{now()}] task{t}: local PASS held task-local pending a solo Kaggle proof; "
                  "not propagated as fleet truth")
    if outcome in ("DONE", "DONE_SALVAGED"):
        MODEL_BUILD_STATS.setdefault(job.get("model_id") or job["model"], [0, 0])[1] += 1
    print(f"[{now()}] task{t} att{job['att']} {job['model']} → {outcome}"
          + (f" +{delta}" if delta else "") + (f" | {fail[:120]}" if fail else ""))
    return outcome


def _find_build_session(t, t0):
    """Newest codex session rollout for a build of task t started at/after t0, matched by
    the session's recorded cwd (builders run in packs/taskNNN — unique per task). Only
    called under RETRY_RESUME; returns a session id or None."""
    import glob as _g
    base = os.path.expanduser("~/.codex/sessions")
    best = (0, None)
    for day_off in (0, 86400):  # today + yesterday (builds can span midnight)
        day = time.strftime("%Y/%m/%d", time.localtime(time.time() - day_off))
        for p in _g.glob(os.path.join(base, day, "rollout-*.jsonl")):
            try:
                mt = os.path.getmtime(p)
                if mt < t0 or mt <= best[0]:
                    continue
                with open(p, errors="replace") as f:
                    meta = json.loads(f.readline())
                pay = meta.get("payload", {})
                if str(pay.get("cwd", "")).endswith(f"packs/task{t}"):
                    best = (mt, pay.get("session_id") or pay.get("id"))
            except (OSError, ValueError):
                continue
    return best[1]


def _blocked(m):
    return time.time() < MODEL_BLOCKED_UNTIL.get(m, 0)


def integrity_tick():
    """Return false and create the operator stop marker on any pricing-file drift."""
    errs = integrity_errors()
    if not errs:
        return True
    stop = os.path.join(RESULTS, "STOP_WAVE")
    try:
        open(stop, "a").write(f"{now()} integrity failure: {'; '.join(errs)}\n")
    except OSError:
        pass
    print(f"[{now()}] INTEGRITY ALERT — STOP_WAVE touched: {'; '.join(errs)}")
    return False


def _decider_rung(t):
    """The decider, having chosen the concrete attack family, can set the routing tier — it
    sees the specific structural challenge the static score can't. Read from the latest
    DECISION file's `ROUTING-DIFFICULTY: EASY|MEDIUM|HARD|VHARD` line; overrides the static
    tier (a mechanical pivot → down; a novel-Einsum invention → up)."""
    try:
        import glob as _g
        ds = sorted(d for d in _g.glob(os.path.join(RESULTS, f"task{t}", "DECISION_att*.md"))
                    if not d.endswith("_input.md") and file_is_current(d))
        if not ds:
            return None
        m = re.search(r"ROUTING-DIFFICULTY:\s*(EASY|MEDIUM|HARD|VHARD)",
                      open(ds[-1], errors="replace").read())
        return TIER_RUNG.get(m.group(1)) if m else None
    except Exception:
        return None


def _clean_real_attempts(atts):
    return real_atts([attempt for _, attempt in current_attempt_pairs(atts)])


def _retry_cooldown(natts, floored):
    """Cooldown policy shared by live requeues and restart-state reconstruction.
    RETRY_COOLDOWN_CAP (default 21600 = the old ceiling, i.e. no behavior change) lets a
    final-sprint wave clamp every cooldown so seats never wait hours on a parked pool."""
    cd = min(natts * 900, 10800) * (2 if floored else 1)
    return min(cd, int(os.environ.get("RETRY_COOLDOWN_CAP", 21600)))


# Operator escape hatch for a single graceful backlog release.  It applies only while
# reconstructing state at process start: every released task still must obtain a fresh
# decision and pass the normal claim, price, and full-gate checks.  Live retries retain
# the ordinary cooldown policy, so this cannot silently turn into an unbounded hot loop.
RELEASE_ALL_IDLE_ON_START = os.environ.get("RELEASE_ALL_IDLE_ON_START", "0") == "1"
# FINAL_SPRINT (one-shot max-capacity wave): enables the never-idle park waiver in the
# spawn loop. Cooldown/park DURATIONS are tuned via their own envs (RETRY_COOLDOWN_CAP,
# STOP_LOSS_PARK, STOP_LOSS_HIGHEV_PARK) so each lever stays independently testable.
FINAL_SPRINT = os.environ.get("FINAL_SPRINT", "0") == "1"


def _restart_task_state(t, improve_below, now_epoch=None):
    """Reconstruct scheduler state that is otherwise lost when the process restarts.

    A sufficiently large still-pending accepted win is complete.  Other tasks retain the
    same strong-attempt cooldown/floor/stop-loss policy used by the live reap path.  The
    timestamp is the attempt completion timestamp, matching when the live cooldown starts.
    """
    best = banked_delta(t, LEDGER)
    if best is not None and best >= improve_below:
        return True, 0.0

    atts = strong_atts(_clean_real_attempts(load_attempts(t)))
    if not atts:
        return False, 0.0

    latest = atts[-1]
    latest_won = latest.get("outcome") in ("DONE", "DONE_SALVAGED")
    nos = [x for x in atts if x.get("outcome") == "NO"
           and not str(x.get("fail_reason", "")).startswith("WATCHDOG")]
    floored = (not latest_won and
               ((len(nos) >= 2 and len({x.get("model") for x in nos}) >= 2)
                or len(nos) >= 3))
    stop_loss = bool(not latest_won and STOP_LOSS_ATTS
                     and len(atts) >= STOP_LOSS_ATTS)
    cd = _retry_cooldown(len(atts), floored)
    if stop_loss:
        cd = max(cd, _stop_loss_park_secs(t)[0])

    value = str(latest.get("ts") or "").strip()
    completed_at = None
    for fmt in ("%Y%m%dT%H%M%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            completed_at = datetime.datetime.strptime(value, fmt).replace(
                tzinfo=datetime.timezone.utc).timestamp()
            break
        except ValueError:
            pass
    if completed_at is None:
        try:
            parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=datetime.timezone.utc)
            completed_at = parsed.timestamp()
        except ValueError:
            return False, 0.0

    not_before = completed_at + cd
    current_time = time.time() if now_epoch is None else now_epoch
    return False, not_before if not_before > current_time else 0.0


def _resume_provider(atts):
    """One interrupted attempt may resume its exact provider before share routing resumes."""
    atts = _clean_real_attempts(atts)
    if not atts or atts[-1].get("outcome") != "TIMEOUT":
        return None
    last = atts[-1]
    twice = (len(atts) >= 2 and atts[-2].get("outcome") == "TIMEOUT"
             and atts[-2].get("model") == last.get("model"))
    return None if twice else last.get("model")


def _worker_available(provider, dry=False):
    if not dry:
        desired = fleet_desired()
        if desired is not None and desired.get(provider, 0) <= 0:
            return False
    return dry or not _blocked(provider)


# ---- dynamic fleet control (2026-07-11) --------------------------------------
# results/FLEET.json = {"codex": N} — desired build seats per
# provider, re-read at the top of every spawn pass so the operator can resize a LIVE wave
# (runner/fleet.py add/remove/set). Increase → next loop tick spawns more from the same
# queue. Decrease → drain: no kills, the gate just stops spawning for that provider until
# active ≤ desired. Missing/malformed file → last good value, else legacy --workers mode.
FLEET_FILE = os.path.join(RESULTS, "FLEET.json")
_FLEET_CACHE = {"mtime": None, "desired": None}


def fleet_desired():
    """The file is ~200 B and read at most every spawn pass — always re-read (no mtime
    cache: coarse filesystem timestamps can hide a same-size rewrite). Malformed content
    keeps the last good value; a file that never existed means legacy --workers mode.
    The optional "off" list (runner/fleet.py off openai|sol|g55) masks groups at READ
    time — seat counts in the file are never destroyed, so `fleet.py on <group>` restores
    them exactly. "openai" masks codex seats; "sol"/"g55" steer the codex model arm +
    decider lanes (see _ab_arm_is_gpt55/_decider_lane_off).
    The optional "deciders" map ({"codex": N}, runner/fleet.py deciders codex=N)
    live-sizes the DECIDER pool — absent means fall back to the DECIDER_MAX_CONC x
    DECIDER_SHARES split frozen at launch (see fleet_deciders)."""
    try:
        with open(FLEET_FILE) as f:
            raw = json.load(f)
        desired = {p: max(0, int(raw.get(p, 0))) for p in ("codex",)}
        off = frozenset(str(x).strip().lower() for x in (raw.get("off") or [])
                        if isinstance(x, str) and str(x).strip())
        dec_raw = raw.get("deciders")
        deciders = ({p: max(0, int(v)) for p, v in dec_raw.items()
                     if p in ("codex",) and str(v).strip() != ""}
                    if isinstance(dec_raw, dict) else None)
        if deciders is not None and "openai" in off:
            deciders["codex"] = 0
        if "openai" in off:
            desired["codex"] = 0
    except OSError:
        return _FLEET_CACHE["desired"]
    except (ValueError, TypeError):
        if _FLEET_CACHE.get("warned") != True:
            print(f"[{now()}] FLEET.json unreadable — keeping "
                  f"{_FLEET_CACHE['desired'] or 'legacy --workers mode'}")
            _FLEET_CACHE["warned"] = True
        return _FLEET_CACHE["desired"]
    _FLEET_CACHE["warned"] = False
    if off != _FLEET_CACHE.get("off", frozenset()):
        print(f"[{now()}] FLEET wind-down update: off={sorted(off)}")
    _FLEET_CACHE["off"] = off
    if deciders != _FLEET_CACHE.get("deciders"):
        print(f"[{now()}] DECIDER resize: {_FLEET_CACHE.get('deciders')} -> {deciders} "
              f"(increase spawns next tick; decrease drains, no kills)")
    _FLEET_CACHE["deciders"] = deciders
    if desired != _FLEET_CACHE["desired"]:
        print(f"[{now()}] FLEET resize: {_FLEET_CACHE['desired']} -> {desired} "
              f"(increase spawns next tick; decrease drains, no kills)")
    _FLEET_CACHE["desired"] = desired
    return desired


def fleet_models_off():
    """Vendor groups currently wound down (FLEET.json "off" list); empty when none."""
    return _FLEET_CACHE.get("off") or frozenset()


def fleet_deciders():
    """Live DECIDER pool size from FLEET.json "deciders" ({"codex": N}), or None when
    the operator never set it (then the launch-time DECIDER_MAX_CONC x DECIDER_SHARES
    split applies). Re-read every spawn pass by fleet_desired(), so `runner/fleet.py
    deciders codex=N` resizes a LIVE wave with no restart — the same drain-not-kill
    semantics as builder seats."""
    fleet_desired()  # refresh the shared cache (cheap; same ~200 B read)
    return _FLEET_CACHE.get("deciders")


def _wave_deadline():
    """Absolute epoch in results/WAVE_DEADLINE (written by the launch script). File-based
    so it survives keepalive restarts — the in-process RUN_STARTED_AT resets per restart
    and deliberately cannot express a wave-level deadline. Missing/garbled = no deadline."""
    try:
        return float(open(os.path.join(RESULTS, "WAVE_DEADLINE")).read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _worker_provider_order(running, dry=False, decider_jobs=None):
    """Fleet mode: providers with free desired seats, lowest fractional fill first.
    Legacy mode (no FLEET.json ever seen): weighted-share deficit order."""
    active = {p: sum(1 for job in running if job.get("model") == p)
              for p in ("codex",)}
    decider_jobs = DECIDE_JOBS if decider_jobs is None else decider_jobs
    # FLEET.codex is a shared Codex-agent budget. Decisions consume the same seats as
    # builds. Under
    # DECIDER_POOL_INDEPENDENT=1 deciders have their own budget and don't occupy seats.
    if os.environ.get("DECIDER_POOL_INDEPENDENT", "") != "1":
        active["codex"] += sum(1 for job in decider_jobs if job.get("provider") == "codex")
    desired = fleet_desired()
    if desired is not None:
        # Fleet counts are simultaneous capacity targets, not a serial priority:
        # normalize by each provider's requested size.
        preference = {"codex": 0}
        return [p for p in sorted(
                    desired,
                    key=lambda p: (active.get(p, 0) / max(1, desired[p]),
                                   preference.get(p, 9), p))
                if active.get(p, 0) < desired[p] and _worker_available(p, dry)]
    return _share_order(WORKER_SHARES, active, SPAWN_COUNTS,
                        lambda provider: _worker_available(provider, dry))


RUNTIME_ROI_ENABLED = os.environ.get("RUNTIME_ROI", "1") == "1"
RUNTIME_ROI_COVERAGE_EVERY = max(0, int(os.environ.get("RUNTIME_ROI_COVERAGE_EVERY", "10")))


def _priority(row, fallback):
    try:
        return int(row.get("priority") or fallback + 1)
    except (TypeError, ValueError):
        return fallback + 1


def _roi_order(queue, provider, indexes=None):
    indexes = list(range(len(queue))) if indexes is None else list(indexes)
    scored = []
    for index in indexes:
        row = queue[index]
        t = row["task"][4:]
        live = _runtime_yield(t, row, provider)
        scored.append(((-live["points_per_hour"], _priority(row, index), index), index))
    return [index for _, index in sorted(scored)]


def _coverage_turn(provider):
    """Bound starvation: one of every N provider spawns follows signed original priority."""
    return (RUNTIME_ROI_COVERAGE_EVERY > 0
            and (SPAWN_COUNTS.get(provider, 0) + 1) % RUNTIME_ROI_COVERAGE_EVERY == 0)


def _worker_task_order(queue, provider, running=None, reserved=None):
    """Order ready work by live points/seat-hour without mutating signed queue inputs."""
    # HOT-RETRY preference: a failed task re-enters at the BACK of the FIFO, so without
    # this a "hot" retry would wait behind the whole queue and its cache prefix would go
    # cold — hot-eligible tasks (still inside the warmth window) are offered first.
    if HOT_RETRY and HOT_READY:
        nowt = time.time()
        for k in [k for k, ts in HOT_READY.items() if nowt - ts > HOT_RETRY_WINDOW]:
            HOT_READY.pop(k, None)
        if HOT_READY:
            base = (_roi_order(queue, provider) if RUNTIME_ROI_ENABLED
                    else list(range(len(queue))))
            return sorted(base, key=lambda i: queue[i]["task"][4:] not in HOT_READY)
    if not RUNTIME_ROI_ENABLED or _coverage_turn(provider):
        return sorted(range(len(queue)), key=lambda i: (_priority(queue[i], i), i))
    return _roi_order(queue, provider)


def next_model(row, atts, provider=None):
    """Return ``(provider, effort, model_id)`` for a share-assigned build slot.

    The share allocator chooses provider. The clean-evidence 0–100 score, optionally raised
    by the latest decider's concrete-design assessment, chooses model and effort. A single
    timeout resumes the exact recorded config because interruption is not a refutation.
    """
    clean_atts = _clean_real_attempts(atts)
    resume = _resume_provider(atts)
    if provider is None:
        provider = resume or next(iter(_worker_provider_order([], dry=True)), "codex")
    if resume == provider and clean_atts:
        last = clean_atts[-1]
        difficulty = _task_difficulty(row["task"][4:], row)
        rung = _decider_rung(row["task"][4:]) or 0
        effective_score = max(difficulty["score"], RUNG_SCORE_FLOOR[rung])
        fallback = _builder_route(provider, effective_score)
        fallback_model, fallback_effort = fallback["model_id"], fallback["effort"]
        return provider, last.get("effort") or fallback_effort, last.get("model_id") or fallback_model

    difficulty = _task_difficulty(row["task"][4:], row)
    rung = _decider_rung(row["task"][4:]) or 0
    effective_score = max(difficulty["score"], RUNG_SCORE_FLOOR[rung])
    route = _builder_route(provider, effective_score)
    # Wave-5 A/B: a GPT55_SHARE slice of codex builds runs the proven legacy config flat
    # (gpt-5.5/xhigh was the sole builder before 07-10, never measured on the fixed scorer).
    # Hash on (task, clean-attempt-number) so each task's attempt stream interleaves arms.
    if provider == "codex" and _ab_arm_is_gpt55(row["task"][4:], len(clean_atts) + 1):
        return provider, "xhigh", "gpt-5.5"
    return provider, route["effort"], route["model_id"]



def status():
    for t in sorted(os.listdir(RESULTS)):
        if not t.startswith("task"):
            continue
        atts = load_attempts(t[4:])
        if atts:
            print(t, "→", " | ".join(f"{a['model']}:{a['outcome']}" for a in atts))
    if os.path.exists(LEDGER):
        rows = list(csv.DictReader(open(LEDGER)))
        best = {}
        for r in rows:
            if r.get("delta"):
                best[r["task"]] = max(best.get(r["task"], -9), float(r["delta"]))
        print(f"\nLEDGER: {len(rows)} rows / {len(best)} tasks, bankable Σ (best-per-task) "
              f"= +{sum(best.values()):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lanes", default="D,A,B,C,G,H")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-attempts", type=int, default=5,
                    help="attempt cap in --finite mode (endless mode ignores it)")
    ap.add_argument("--improve-below", type=float, default=0.25,
                    help="a DONE with Δ below this is banked but re-attempted for a bigger win "
                         "(0.15→0.25 on 2026-07-09: the 07-08 wave's house-money re-attacks on "
                         "locked floors yielded +1.72 from 5 tasks — 295 +0.05→+0.84, 392 "
                         "+0.05→+0.60 — while 46 bar-scraper wins at +0.15–0.20 never re-entered)")
    ap.add_argument("--no-watchdog", action="store_true",
                    help="disable the mid-attempt progress watchdog")
    ap.add_argument("--finite", action="store_true",
                    help="old behavior: stop at --max-attempts / floor agreement. Default "
                         "is ENDLESS: tasks cycle with growing cooldowns until Ctrl-C "
                         "(1st = graceful drain, 2nd = kill workers)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.status:
        return status()
    os.makedirs(CLAIMS, exist_ok=True); os.makedirs(RESULTS, exist_ok=True)
    if not os.path.exists(LESSONS):
        open(LESSONS, "w").write("# Fleet lessons (one bullet per attempt — auto-injected into every prompt)\n")
    lanes = set(a.lanes.split(",")) if a.lanes else None
    only = set(x.zfill(3) for x in a.tasks.split(",") if x) or None
    queue = read_queue(lanes, only)
    _fd0 = fleet_desired()
    print(f"queue: {len(queue)} tasks, lanes={sorted(lanes or [])}, "
          + (f"fleet={_fd0} (live-resizable via runner/fleet.py)" if _fd0 is not None
             else f"workers={a.workers} (legacy)")
          + f" · codex A/B: GPT55_SHARE={GPT55_SHARE} (arm B = gpt-5.5/xhigh flat, arm A = "
          f"{os.environ.get('ROUTE_GPT_MODEL', 'gpt-5.5')}/xhigh"
          f"{' FAST-tier' if os.environ.get('ROUTE_GPT_MODEL', 'gpt-5.5') in CODEX_FAST_MODELS else ''}"
          f") · deciders "
          f"{int((1-DECIDER_SOL_SHARE)*100)}% gpt-5.5/{os.environ.get('DECIDER_BASE_EFFORT', 'xhigh')} / "
          f"{int(DECIDER_SOL_SHARE*100)}% gpt-5.6-sol/{os.environ.get('DECIDER_SOL_EFFORT', 'xhigh')} "
          f"(Sol takes hardest)")

    # tee everything to a persistent log — the night must be reconstructable
    class _Tee:
        def __init__(self, *s): self.s = s
        def write(self, x):
            for f in self.s:
                f.write(x); f.flush()
        def flush(self):
            for f in self.s:
                f.flush()
    if not a.dry_run and not sys.__stdout__.isatty():
        try:
            sys.stdout.reconfigure(line_buffering=True)  # live lines when redirected to log
        except Exception:
            pass
    if not a.dry_run and sys.__stdout__.isatty():
        # tee only when stdout is a terminal — when launched with `>> ORCHESTRATOR.log`
        # the tee would double every line (first run's log was fully duplicated)
        sys.stdout = _Tee(sys.__stdout__, open(os.path.join(RESULTS, "ORCHESTRATOR.log"), "a"))
    open(result_name("orchestrator.pid"), "w").write(str(os.getpid()))
    WAVE_TS = now()  # A/B scoreboard window: only THIS wave's attempts are compared
    print(f"[{now()}] === WAVE START === mode={'finite' if a.finite else 'ENDLESS'} "
          f"workers={a.workers} improve_below={a.improve_below} "
          f"watchdog={'off' if a.no_watchdog else 'on'} "
          f"lanes={a.lanes} tasks={len(queue)}")

    running, done_tasks = [], set()
    NOT_BEFORE = {}   # task -> earliest respawn time (endless-mode cooldowns)
    STOP = {"n": 0}
    last_hb = 0.0
    last_integrity = 0.0

    # Restore provider benches across restarts (wave7 audit: bench state was in-process
    # only, so a keepalive relaunch during a quota window re-spawned 12 workers into a
    # dead provider — each burning a spawn, dying, re-benching). blocked_until carries
    # absolute epochs; the legacy blocked_models field is remaining-seconds and is NOT
    # used here. Conservative caps: snapshot must be <6 h old, restored bench never
    # exceeds now+1800 (a corrupt value can't bench longer than one standard bench).
    if not a.dry_run:
        try:
            _rs = json.load(open(result_name("RUN_STATE.json")))
            if time.time() - float(_rs.get("heartbeat_at", 0)) < 6 * 3600:
                for _m, _until in (_rs.get("blocked_until") or {}).items():
                    _until = min(float(_until), time.time() + 1800)
                    if _m in SPAWN_COUNTS and _until > time.time() > MODEL_BLOCKED_UNTIL.get(_m, 0):
                        MODEL_BLOCKED_UNTIL[_m] = _until
                        print(f"[{now()}] restored bench: {_m} blocked "
                              f"{int((_until - time.time()) / 60)}m (from RUN_STATE)")
        except (OSError, ValueError, TypeError):
            pass

    if not a.dry_run:
        restored_cooling = 0
        released_cooling = 0
        for r in queue:
            t = r["task"][4:]
            restored_done, restored_not_before = _restart_task_state(
                t, a.improve_below)
            if restored_done:
                done_tasks.add(t)
            elif restored_not_before and not RELEASE_ALL_IDLE_ON_START:
                NOT_BEFORE[t] = restored_not_before
                restored_cooling += 1
            elif restored_not_before:
                released_cooling += 1
        if done_tasks:
            queue[:] = [r for r in queue if r["task"][4:] not in done_tasks]
        if RELEASE_ALL_IDLE_ON_START:
            print(f"[{now()}] operator one-shot release: {released_cooling} idle cooldown(s) "
                  f"made eligible for fresh decisions; normal retry cooldowns remain enabled")
        if done_tasks or restored_cooling:
            print(f"[{now()}] restart restored {len(done_tasks)} completed task(s) and "
                  f"{restored_cooling} active cooldown(s)")

    pin_deps = [os.path.join(ROOT, "data", "pin_pv2_70_263_pertask.csv"),
                os.path.join(ROOT, "data", "PIN_SHA.txt"), ERA_FILE]
    for r in queue if not a.dry_run else []:  # dry-run spawns NOTHING, deciders included
        t = r["task"][4:]
        if t in done_tasks:
            continue
        atts = load_attempts(t)
        if not atts:
            continue
        last_out = atts[-1].get("outcome", "")
        if last_out in ("DONE", "DONE_SALVAGED"):
            continue
        dp = os.path.join(RESULTS, f"task{t}", f"DECISION_att{len(atts) + 1:02d}.md")
        stale_decision = os.path.exists(dp) and (not file_is_current(dp) or any(
            os.path.exists(dep) and os.path.getmtime(dep) > os.path.getmtime(dp)
            for dep in pin_deps))
        revision_reason = _decision_revision_reason(t)
        if revision_reason:
            bad_dp = _latest_decision(t)
            if bad_dp:
                _quarantine_file(t, bad_dp, revision_reason, "budget-revise")
        if not os.path.exists(dp) or stale_decision:
            DECIDE_WAIT[t] = r
        elif revision_reason:
            DECIDE_WAIT[t] = r
    if DECIDE_WAIT:
        print(f"[{now()}] resume seeded {len(DECIDE_WAIT)} retry decider(s) before spawning")

    def on_sigint(sig, frame):
        STOP["n"] += 1
        if STOP["n"] == 1:
            print(f"\n[{now()}] STOP requested — no new spawns; draining "
                  f"{len(running)} running worker(s) + analysts/deciders. "
                  f"Ctrl-C again to kill workers.")
        else:
            print(f"\n[{now()}] hard stop — killing workers")
            for j in running:
                j["proc"].kill()
    signal.signal(signal.SIGINT, on_sigint)

    import traceback
    WAVE_DEADLINE = None if a.dry_run else _wave_deadline()
    if WAVE_DEADLINE:
        print(f"[{now()}] wave deadline armed: {int(max(0, WAVE_DEADLINE - time.time()) // 60)} min "
              f"remaining (spawns stop 20 min before; running attempts drain at T)")
    while True:
        if WAVE_DEADLINE and not STOP["n"] and time.time() >= WAVE_DEADLINE:
            STOP["n"] = 1
            try:
                open(os.path.join(RESULTS, "STOP_WAVE"), "a").write(
                    f"{now()} wave deadline reached — graceful drain\n")
            except OSError:
                pass
            print(f"[{now()}] WAVE DEADLINE — no new spawns/decisions; draining; "
                  f"keepalive will not resurrect")
        if not a.dry_run and not STOP["n"]:
            # a repin (auto-submit loop) rewrites QUEUE.csv with a fresh priority order;
            # pick it up live so the ingest→reprioritize→continue loop needs no restart.
            maybe_reload_queue(queue, lanes, only)
        if not a.dry_run and time.time() - last_integrity >= 60:
            last_integrity = time.time()
            if not integrity_tick():
                STOP["n"] = max(STOP["n"], 1)
        try:
            reap_postmortems(); reap_price_searches(); reap_deciders()
            if not a.no_watchdog:
                watchdog_tick(running); reap_watchdog(running)
        except Exception:
            print(f"[{now()}] helper-phase EXCEPTION (wave continues):\n"
                  + traceback.format_exc()[-1500:])
        # auto-submit tick (2026-07-11, operator-sanctioned): fire-and-forget child; the
        # child itself does the cheap Σ-trigger check and exits fast when below threshold,
        # and a flock in auto_submit.py guarantees single-flight. Never blocks the loop.
        if (not a.dry_run and os.environ.get("KAGGLE_AUTOSUBMIT", "") == "1"
                and time.time() - AUTOSUBMIT_TICK["last"] > 300):
            AUTOSUBMIT_TICK["last"] = time.time()
            try:
                subprocess.Popen(
                    [sys.executable, os.path.join(HERE, "auto_submit.py"), "--once"],
                    stdout=open(os.path.join(RESULTS, "AUTOSUBMIT.log"), "a"),
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            except Exception as exc:
                print(f"[{now()}] auto_submit spawn failed (wave continues): {exc}")
        # WHAT-WORKS refresh tick (2026-07-11): every decider prompt embeds
        # results/WHAT_WORKS_NOW.md (verified cross-task strategy evidence — the safe form
        # of "deciders share impactful strategies": only harness-verified LEDGER wins and
        # analyst postmortems feed it, never raw decider opinions, so no herding on
        # unproven families and no LLM-writable injection channel). Previously only the
        # admin script refreshed it, so mid-wave deciders read a frozen snapshot.
        # Trigger: LEDGER changed (a win/salvage banked) + 15 min throttle; fire-and-forget
        # via the bounded Codex Spark admin role; the miner keeps the old file on any
        # failure, so this can only improve decider evidence, never degrade it.
        if (not a.dry_run and os.environ.get("WHATWORKS_REFRESH", "1") == "1"
                and not _blocked("admin")
                and time.time() - AUTOSUBMIT_TICK.get("ww_last", 0) > 900):
            try:
                lm = os.path.getmtime(LEDGER) if os.path.exists(LEDGER) else 0
            except OSError:
                lm = 0
            if lm and lm != AUTOSUBMIT_TICK.get("ww_ledger_mtime"):
                AUTOSUBMIT_TICK["ww_last"] = time.time()
                AUTOSUBMIT_TICK["ww_ledger_mtime"] = lm
                try:
                    subprocess.Popen(
                        [sys.executable, os.path.join(HERE, "whatworks_miner.py")],
                        stdout=open(os.path.join(RESULTS, "WHATWORKS.log"), "a"),
                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                    print(f"[{now()}] LEDGER changed — refreshing WHAT_WORKS_NOW.md "
                          f"(decider strategy board; fire-and-forget)")
                    # accumulate-mode (2026-07-13): keep the operator's manual-submission
                    # shopping list fresh vs the LIVE pin. Fire-and-forget; the script is
                    # flock-single-flight and rate-limits its own heavy reprices.
                    subprocess.Popen(
                        [sys.executable, os.path.join(HERE, "pending_report.py")],
                        stdout=open(os.path.join(RESULTS, "PENDING_REPORT.log"), "a"),
                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                except Exception as exc:
                    print(f"[{now()}] whatworks refresh spawn failed (wave continues): {exc}")
        # A/B scoreboard tick: keep results/AB_STATS.md fresh for the operator (read-only
        # aggregation over ATTEMPTS.jsonl since THIS wave's start; fire-and-forget)
        if not a.dry_run and time.time() - AUTOSUBMIT_TICK.get("ab_last", 0) > 900:
            AUTOSUBMIT_TICK["ab_last"] = time.time()
            try:
                subprocess.Popen(
                    [sys.executable, os.path.join(HERE, "ab_stats.py"), "--quiet",
                     "--since", WAVE_TS],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL)
            except Exception:
                pass
        # quota telemetry tick (wave7 audit 2026-07-14: codex quota exhaustion killed the
        # wave with zero warning — USAGE.jsonl had no scheduled writer). usage_scan.py is a
        # passive idempotent scanner of ~/.codex session rollouts (state-cursor'd, skips
        # live sessions); its quota_primary_pct lands in the STATUS line via _latest_quota.
        if (not a.dry_run and os.environ.get("USAGE_SCAN", "1") == "1"
                and time.time() - AUTOSUBMIT_TICK.get("usage_last", 0) > 900):
            AUTOSUBMIT_TICK["usage_last"] = time.time()
            try:
                subprocess.Popen(
                    [sys.executable, os.path.join(HERE, "usage_scan.py"), "--quiet"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL)
            except Exception:
                pass
        # reap (two-stage timeout: SIGTERM + grace so the CLI can flush its final message,
        # then SIGKILL — an interrupted attempt keeps its partial transcript and artifacts)
        for job in running[:]:
            rc = job["proc"].poll()
            if rc is None and "term_at" in job:
                # grace applies to ANY termination (timeout OR watchdog): SIGTERM was
                # sent when term_at was set; enforce SIGKILL after the grace window
                if time.time() - job["term_at"] > GRACE_SECS:
                    job["proc"].kill(); rc = -9
            elif rc is None and time.time() - job["t0"] > job["timeout"]:
                job["term_at"] = time.time()
                job["interrupted"] = True
                job["proc"].terminate()
                print(f"[{now()}] task{job['row']['task'][4:]} hit its time limit — "
                      f"SIGTERM sent, {GRACE_SECS}s grace to flush")
            if rc is not None:
                running.remove(job)
                try:
                    out = finish(job)
                except Exception:
                    t = job["row"]["task"][4:]
                    print(f"[{now()}] finish() EXCEPTION task{t} — task re-queued, wave "
                          f"continues:\n" + traceback.format_exc()[-1200:])
                    c = os.path.join(CLAIMS, f"task{t}.claim")
                    if os.path.exists(c):
                        os.remove(c)
                    queue.append(job["row"])
                    NOT_BEFORE[t] = time.time() + 600
                    continue
                t, row = job["row"]["task"][4:], job["row"]
                if out == "QUOTA_BLOCKED":
                    # model benched — re-queue the task unchanged (no attempt burned, no
                    # analyst/decider); a fresh worker on a live model picks it up.
                    queue.append(row)
                    NOT_BEFORE[t] = time.time() + 90
                    c = os.path.join(CLAIMS, f"task{t}.claim")
                    if os.path.exists(c):
                        os.remove(c)
                    continue
                atts = load_attempts(t)
                if out in ("DONE", "DONE_SALVAGED"):  # DONE_UNVERIFIED/DONE_NO_ARTIFACT retry
                    last = atts[-1] if atts else {}
                    d = last.get("delta")
                    small = d is not None and d < a.improve_below
                    clean_real = _clean_real_attempts(atts)
                    if small and (not a.finite or len(clean_real) < a.max_attempts):
                        # SMALL WIN = house money: the Δ is banked (LEDGER keeps best-per-
                        # task; submit_result never clobbers a better artifact). Re-queue
                        # through the same analyst→decider path to swing for a big gain.
                        print(f"[{now()}] task{t}: small win Δ+{d} BANKED — re-queuing "
                              f"for a bigger swing (floor locked)")
                        queue.append(row)
                        DECIDE_WAIT[t] = row
                        NOT_BEFORE[t] = time.time() + _retry_cooldown(len(clean_real), False)
                    else:
                        done_tasks.add(t)
                else:
                    # STOP-LOSS / FLOOR / COOLDOWN count STRONG attempts only: a weak-screen
                    # failure is not evidence a task is hard — it re-enters the
                    # strong-model flow immediately with no cooldown step.
                    clean_real = strong_atts(_clean_real_attempts(atts))
                    # watchdog kills are recorded as NO for retry semantics but are NOT
                    # cost-floor evidence — exclude them so 2 watchdog kills + 1 real NO
                    # can't fake a floor claim (wave7 watchdog-v2)
                    nos = [x for x in clean_real if x["outcome"] == "NO"
                           and not str(x.get("fail_reason", "")).startswith("WATCHDOG")]
                    floored = (len(nos) >= 2 and len({x["model"] for x in nos}) >= 2) or len(nos) >= 3
                    n_real = len(clean_real)
                    stop_loss = STOP_LOSS_ATTS and n_real >= STOP_LOSS_ATTS
                    if a.finite:
                        if n_real < a.max_attempts and not floored:
                            queue.append(row)
                            DECIDE_WAIT[t] = row
                        elif floored:
                            print(f"[{now()}] task{t} floored — accepting NO")
                    else:
                        # ENDLESS: nothing is abandoned — floored tasks cool down 2× longer;
                        # a STOP-LOSS task (>= STOP_LOSS_ATTS real attempts, no win) is PARKED.
                        queue.append(row)
                        DECIDE_WAIT[t] = row
                        cd = _retry_cooldown(n_real, floored)
                        _park_edelta = None
                        if stop_loss:
                            _park_secs, _park_edelta = _stop_loss_park_secs(t)
                            cd = max(cd, _park_secs)
                        if (_hot_arm(t) and not stop_loss and not floored
                                and n_real + 1 <= HOT_RETRY_MAX_ATT):
                            # hot arm: no cooldown — the analyst+decider gates (2-10 min)
                            # are the only wait, keeping the cache prefix warm for the retry
                            cd = 0
                            HOT_READY[t] = time.time()
                            print(f"[{now()}] task{t} HOT-RETRY armed (att{n_real + 1} next; "
                                  f"cooldown skipped, decider+analyst gates still apply)")
                        NOT_BEFORE[t] = time.time() + cd
                        if stop_loss:
                            print(f"[{now()}] task{t} STOP-LOSS: {n_real} real attempts, no win "
                                  f"— parked {cd//3600}h"
                                  + (f" (high-headroom exemption: edelta {_park_edelta:.2f})"
                                     if _park_edelta is not None else ""))
                        elif floored:
                            print(f"[{now()}] task{t} floored — endless mode: cooldown "
                                  f"{cd//60} min, then the decider must pick a NEVER-tried family")
                os.remove(os.path.join(CLAIMS, f"task{t}.claim")) if \
                    os.path.exists(os.path.join(CLAIMS, f"task{t}.claim")) else None
        # FINAL_SPRINT never-idle waiver: when free seats exceed the eligible pool, the
        # cooling map (NOT_BEFORE) is the only thing idling capacity — release the best
        # parked task (live yield) once per 10s pass. Tasks whose decision/brief is still
        # in flight are NOT released (there is nothing to build yet).
        if FINAL_SPRINT and not STOP["n"] and queue:
            _fw = fleet_desired()
            _free = (sum(_fw.values()) if _fw is not None else a.workers) - len(running)
            if _free > 0:
                _nowt = time.time()
                def _open_row(r):
                    _tt = r["task"][4:]
                    return (_tt not in done_tasks and not pm_pending(_tt)
                            and not decider_pending(_tt))
                _elig = sum(1 for r in queue if _open_row(r)
                            and _nowt >= NOT_BEFORE.get(r["task"][4:], 0))
                if _free > _elig:
                    _cool = [r for r in queue if _open_row(r)
                             and _nowt < NOT_BEFORE.get(r["task"][4:], 0)]
                    if _cool:
                        _b = max(_cool, key=lambda r: _runtime_yield(
                            r["task"][4:], r, "codex")["points_per_hour"])
                        _bt = _b["task"][4:]
                        _early = int(NOT_BEFORE.get(_bt, _nowt) - _nowt)
                        NOT_BEFORE.pop(_bt, None)
                        print(f"[{now()}] SPRINT-WAIVER: {_free} free seats > {_elig} "
                              f"eligible — task{_bt} unparked {_early}s early (best live yield)")
        # Spawn: fleet seats (or legacy capacity share) choose provider; difficulty chooses
        # model×effort. A retry still waits for its analyst+decider, and per-provider
        # MAX_CONC caps remain hard ceilings in both modes.
        while queue and not STOP["n"] and (
                WAVE_DEADLINE is None or time.time() < WAVE_DEADLINE - 1200):
            _fleet = fleet_desired()
            if _fleet is None:
                _cap = a.workers
            else:
                _act = {}
                for j in running:
                    _act[j["model"]] = _act.get(j["model"], 0) + 1
                # max(desired, active) per provider: a wound-down provider's draining
                # overhang must never block another provider's free seats
                _cap = sum(max(n, _act.get(p, 0)) for p, n in _fleet.items())
                if os.environ.get("DECIDER_POOL_INDEPENDENT", "") != "1":
                    _cap -= sum(1 for j in DECIDE_JOBS if j.get("provider") == "codex")
            if len(running) >= _cap:
                break
            conc = {}
            for j in running:
                conc[j["model"]] = conc.get(j["model"], 0) + 1
            idx = model = None
            for assigned_provider in _worker_provider_order(running, dry=a.dry_run):
                if (not a.dry_run and
                        conc.get(assigned_provider, 0) >= MODELS[assigned_provider].get("max_conc", 99)):
                    continue
                for i in _worker_task_order(queue, assigned_provider, running=running):
                    r = queue[i]
                    tt = r["task"][4:]
                    # SPRINT cold-start (final wave): the independent decider pool queues a
                    # decision for almost every task, so decider_pending would starve the
                    # builder lanes. A task
                    # not yet cold-started this run may take its one decision-free big-swing
                    # NOW even while its decider is queued/running; that decider still lands
                    # the att+1 decision for the next retry. queue.pop + the once-per-task
                    # WAVE_COLD_STARTED guard + claims keep this from double-spawning.
                    _cold_swing = SPRINT_COLD_START and tt not in WAVE_COLD_STARTED
                    if tt in done_tasks or pm_pending(tt) \
                            or (decider_pending(tt) and not _cold_swing) \
                            or time.time() < NOT_BEFORE.get(tt, 0):
                        continue
                    revision_reason = _decision_revision_reason(tt)
                    if revision_reason:
                        bad_dp = _latest_decision(tt)
                        if bad_dp:
                            _quarantine_file(tt, bad_dp, revision_reason, "budget-revise")
                        DECIDE_WAIT[tt] = r
                        print(f"[{now()}] task{tt}: {revision_reason} — fresh decision requested; "
                              f"no worker attempt spent")
                        continue
                    task_atts = load_attempts(tt)
                    resume = _resume_provider(task_atts)
                    if (resume and _worker_available(resume, dry=a.dry_run)
                            and resume != assigned_provider):
                        continue
                    m_, eff_, mid_ = next_model(r, task_atts, provider=assigned_provider)
                    idx, model, effort, model_id = i, m_, eff_, mid_
                    break
                if idx is not None:
                    break
            if idx is None:
                break  # everything queued is helper-pending/cooling or providers are capped
            row = queue.pop(idx)
            t = row["task"][4:]
            if t in done_tasks:
                continue
            claim = os.path.join(CLAIMS, f"task{t}.claim")
            if not a.dry_run:  # dry-run must be side-effect free and print every task
                if not acquire_claim(t):
                    # Another orchestrator owns it right now.  Popping must not silently
                    # delete the task from this scheduler; retry after a short backoff.
                    queue.append(row)
                    NOT_BEFORE[t] = time.time() + 60
                    continue
            atts = load_attempts(t)
            try:
                job = spawn(row, model, len(atts) + 1, a.dry_run, effort=effort, model_id=model_id)
            except (PolicyRejected, DecisionRevisionRequired) as exc:
                print(f"[{now()}] task{t}: {exc} — fresh decision requested; rejected "
                      f"input will not be requeued unchanged")
                if os.path.exists(claim):
                    os.remove(claim)
                DECIDE_WAIT[t] = row
                queue.append(row)
                continue
            except Exception:
                print(f"[{now()}] spawn EXCEPTION task{t} — re-queued, wave continues:\n"
                      + traceback.format_exc()[-1200:])
                if os.path.exists(claim):
                    os.remove(claim)
                queue.append(row)
                NOT_BEFORE[t] = time.time() + 300
                continue
            if job:
                live_yield = _runtime_yield(t, row, model)
                job["runtime_yield"] = live_yield
                if WAVE_DEADLINE:
                    # taper: no attempt may run past the deadline by more than the grace
                    job["timeout"] = min(job["timeout"],
                                         max(900, int(WAVE_DEADLINE - time.time())))
                running.append(job)
                bench = model_benchmark(model_id, effort)
                print(f"[{now()}] spawn task{t} att{len(atts)+1} → "
                      f"{model}:{model_id}/{effort} B{bench or '?'} "
                      f"yield={live_yield['points_per_hour']:.4f} pts/h "
                      f"({len(running)}/{a.workers} slots)")
            elif a.dry_run:
                done_tasks.add(t)
        # Builders consume ready decisions first; deciders immediately borrow every
        # remaining Codex seat. On the next tick, completed decisions become build-ready
        # before the pool is backfilled again. Past the deadline's spawn cutoff, new
        # decisions would only feed builds that can never start — skip them.
        try:
            if WAVE_DEADLINE is None or time.time() < WAVE_DEADLINE - 1200:
                run_deciders(running)
        except Exception:
            print(f"[{now()}] decider-fill EXCEPTION (wave continues):\n"
                  + traceback.format_exc()[-1500:])
        # idle-work fallback: all seated providers benched AND nothing running → spend the
        # bench on the deterministic pending-artifact sweep. Keyed on the bench horizon so
        # each bench period spawns exactly once; never killed on bench-lift (nice 10 keeps
        # gates fast, and every verified candidate is pure upside for the shopping list).
        if IDLE_WORK_ENABLED and not a.dry_run and not running and queue and not STOP["n"]:
            _fleet_i = fleet_desired()
            _seatp = ([p for p, n in _fleet_i.items() if n > 0] if _fleet_i is not None
                      else ["codex"])
            if _seatp and all(_blocked(p) for p in _seatp):
                _key = int(max(MODEL_BLOCKED_UNTIL.get(p, 0) for p in _seatp))
                if (_key and _key != IDLE_WORK["key"]
                        and not any(pr.poll() is None for pr in IDLE_WORK["procs"])):
                    IDLE_WORK["key"] = _key
                    IDLE_WORK["procs"] = []
                    try:
                        _ilog = open(os.path.join(RESULTS, "IDLE_SWEEP.log"), "a")
                        IDLE_WORK["procs"].append(subprocess.Popen(
                            ["nice", "-n", "10", sys.executable,
                             os.path.join(HERE, "bool_collapse_sweep.py"),
                             "--pending", "--jobs", IDLE_SWEEP_JOBS],
                            stdout=_ilog, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, cwd=ROOT))
                        IDLE_WORK["procs"].append(subprocess.Popen(
                            [sys.executable, os.path.join(HERE, "pending_report.py")],
                            stdout=_ilog, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, cwd=ROOT))
                        print(f"[{now()}] IDLE-WORK: all seated providers benched, "
                              f"running=0 — spawned pending sweep (bench key {_key})")
                    except Exception as exc:
                        print(f"[{now()}] IDLE-WORK spawn failed (wave continues): {exc}")
        if not a.dry_run:
            try:
                write_run_state(a, queue, running, done_tasks, NOT_BEFORE, STOP)
            except OSError as exc:
                print(f"[{now()}] telemetry write failed (wave continues): {exc}")
        if a.dry_run and not queue:
            break
        if a.finite and not queue and not running:
            break
        if STOP["n"] and not running:
            print(f"[{now()}] drained — exiting on operator stop")
            break
        if not a.finite and not queue and not running and not a.dry_run:
            break  # endless but every task hit a big win — nothing left to improve
        if time.time() - last_hb > 60 and not a.dry_run:
            last_hb = time.time()
            try:
                _b = {}
                if os.path.exists(LEDGER):
                    for r in csv.DictReader(open(LEDGER)):
                        if r.get("delta"):
                            _b[r["task"]] = max(_b.get(r["task"], -9), float(r["delta"]))
                led = sum(_b.values())  # bankable = best-per-task (endless mode re-wins)
                _fd = fleet_desired()
                _fa = {p: sum(1 for j in running if j.get("model") == p)
                       for p in ("codex",)}
                fleet_txt = (" ".join(f"{p}={_fa[p]}/{_fd[p]}" for p in _fd)
                             if _fd is not None else f"legacy workers={a.workers}")
                open(result_name("STATUS.txt"), "w").write(
                    f"{now()} running={len(running)} queue={len(queue)} "
                    f"done={len(done_tasks)} "
                    f"cooling={sum(1 for v in NOT_BEFORE.values() if v > time.time())} "
                    f"analysts={len(PM_JOBS)} deciders={len(DECIDE_JOBS) + len(DECIDE_WAIT)} "
                    f"fleet[{fleet_txt}] "
                    f"builds[codex={SPAWN_COUNTS.get('codex',0)}] "
                    + ("models[" + " ".join(
                        f"{m}={d}/{s}" for m, (s, d) in sorted(MODEL_BUILD_STATS.items()))
                        + "] " if MODEL_BUILD_STATS else "")
                    + (f"off=[{','.join(sorted(fleet_models_off()))}] "
                       if fleet_models_off() else "")
                    + (f"deadline={max(0, int((WAVE_DEADLINE - time.time()) // 60))}m "
                       if WAVE_DEADLINE else "")
                    + f"infra_requeues={sum(INFRA_REQUEUES.values())} "
                    f"blocked={[m for m in MODEL_BLOCKED_UNTIL if time.time()<MODEL_BLOCKED_UNTIL[m]]} "
                    f"ledger_sum=+{led:.2f} stop={STOP['n']}"
                    + (f" {_latest_quota()}" if _latest_quota() else "") + "\n")
            except Exception:
                pass
        time.sleep(10)
    t_end = time.time()  # drain in-flight analysts + deciders (they inform the next wave)
    _past_deadline = bool(WAVE_DEADLINE and time.time() >= WAVE_DEADLINE - 1200)
    while ((PM_JOBS or DECIDE_JOBS or PS_JOBS or (DECIDE_WAIT and not _past_deadline))
           and time.time() - t_end < 900):
        reap_postmortems()
        if not _past_deadline:  # final-wave deadline: fresh decisions feed nothing now
            run_deciders(running)
        reap_price_searches(); reap_deciders()
        if not a.dry_run:
            try:
                write_run_state(a, queue, running, done_tasks, NOT_BEFORE, STOP,
                                phase="draining")
            except OSError:
                pass
        time.sleep(5)
    reap_postmortems(); reap_price_searches(); reap_deciders()
    if not a.dry_run:
        try:
            write_run_state(a, queue, running, done_tasks, NOT_BEFORE, STOP, phase="stopped")
        except OSError:
            pass
    print("wave complete."); status()


if __name__ == "__main__":
    main()
