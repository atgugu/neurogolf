#!/usr/bin/env python3
"""auto_submit — operator-sanctioned automatic Kaggle submission + repin pipeline.

Spawned by the orchestrator every ~300s (fire-and-forget with KAGGLE_AUTOSUBMIT=1) or run
by hand. Single-flight via flock. The cheap trigger check exits in milliseconds; the full
pipeline (bank --regate → submit → recon → promote → repin) runs only when triggered.

Policy (point-max, 2026-07-11): exact-byte bank confirmation of any positive unseen
candidate may ship; unproven/canary artifacts are attributable solos. Probes preserve
final=best; the LIVE daily limit (SDK, cached; static-100 floor) retains a hard reserve.

Safety rails, in order — all BEFORE anything is submitted:
  flock single-flight → KAGGLE_AUTOSUBMIT=1 kill-switch → integrity_errors() empty →
  pin triple-consistency (PIN_SHA.txt == sha8(submission.zip) == kaggle_pin.json.sha) →
  exact live pin razor-clean → quota guard (n_today + planned + 2 ≤ live_limit−reserve AND auto-cap).
Plus bank.py's own interlocks (mandatory reprice w/ PIN-MOVED hard stop, razor scans,
payment-proof chunking, zeros-probe canaries) and probe_submit's (freshness gate,
member-set equality, unscorable refusal). Never uses --force.

Flags: --once (default behavior) · --dry-run (stop after the confirm stage) ·
--force (skip the cheap trigger) · --threshold X.
"""
import argparse
import csv
import datetime
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
from config import CLEAN as _CLEAN
CLEAN = Path(_CLEAN)
PIN_JSON = CLEAN / "logs" / "kaggle_pin.json"
PROBE_LOG = CLEAN / "logs" / "probe_log.jsonl"
OUTCOMES = CLEAN / "logs" / "probe_outcomes.json"
LEDGER = RESULTS / "LEDGER.csv"
MARKER = RESULTS / "AUTOSUBMIT_MARKER.json"
EVENTS = RESULTS / "AUTOSUBMIT_EVENTS.jsonl"
LOCK = RESULTS / "AUTOSUBMIT.lock"
REJECTED = RESULTS / "REJECTED_ARTIFACTS.jsonl"
CYCLE = RESULTS / "AUTOSUBMIT_CYCLE.json"
COMP = "neurogolf-2026"

THRESHOLD = float(os.environ.get("AUTOSUBMIT_DELTA", 10))
FLUSH_HRS = float(os.environ.get("AUTOSUBMIT_FLUSH_HRS", 12))
FLUSH_MIN_SUM = float(os.environ.get("AUTOSUBMIT_FLUSH_MIN", 2))
SOLO_MIN_SUM = float(os.environ.get("AUTOSUBMIT_SOLO_MIN", 0.0))
AUTO_MAX_SUBS_PER_DAY = int(os.environ.get("AUTO_MAX_SUBS_PER_DAY", 20))
QUOTA_RESERVE = int(os.environ.get("AUTOSUBMIT_QUOTA_RESERVE", 10))
GAP_ALARM = -0.3
REPIN_MAX_PER_DAY = 2
REPIN_FREEZE_AT = os.environ.get("REPIN_FREEZE_AT", "2026-07-14T12:00")


def log(msg):
    print(f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ} {msg}",
          flush=True)


def event(kind, **kw):
    with EVENTS.open("a") as f:
        f.write(json.dumps({"ts": f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ}",
                            "kind": kind, **kw}) + "\n")


def sha8(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:8]


def ledger_best_sum(after_lines=0):
    """Best-per-task raw Σ after a data-row cursor (cheap trigger; bank reprices).

    The cursor is essential after a repin: a newly accepted +0.3 artifact must remain
    visible even when an absorbed historical row for the same task was +1.0.
    """
    rows = []
    try:
        with LEDGER.open() as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        pass
    try:
        cursor = max(0, min(len(rows), int(after_lines)))
    except (TypeError, ValueError):
        cursor = 0
    best, solos = {}, []
    for r in rows[cursor:]:
        try:
            d = float(r["delta"])
        except (KeyError, TypeError, ValueError):
            continue
        if d > best.get(r.get("task"), -9):
            best[r.get("task")] = d
        if "ORACLE-SOLO" in (r.get("notes") or "") or "CANARY" in (r.get("notes") or ""):
            solos.append(r.get("task"))
    return round(sum(best.values()), 4), len(rows), sorted(set(solos))


def read_marker():
    try:
        return json.loads(MARKER.read_text())
    except (OSError, ValueError):
        return {"sum_best": 0.0, "ledger_lines": 0, "pin_sha": "", "ts": "",
                "last_flush_ts": 0.0}


def write_marker(sum_best, lines, pin_sha, flushed, cycle_id=None):
    m = read_marker()
    record = {
        "sum_best": sum_best, "ledger_lines": lines, "pin_sha": pin_sha,
        "ts": f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ}",
        "last_flush_ts": time.time() if flushed else m.get("last_flush_ts", 0.0)}
    if cycle_id:
        record["cycle_id"] = cycle_id
    payload = json.dumps(record, indent=1)
    tmp = MARKER.with_suffix(MARKER.suffix + ".tmp")
    with tmp.open("w") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, MARKER)


def read_cycle():
    """Read the durable in-flight submission intent; corruption fails closed."""
    try:
        raw = json.loads(CYCLE.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"unreadable autosubmit cycle journal: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("version") != 1 \
            or not raw.get("cycle_id") or not isinstance(raw.get("ships"), list):
        raise RuntimeError("invalid autosubmit cycle journal schema")
    return raw


def write_cycle(cycle):
    """Atomically checkpoint before/after every externally visible submit action."""
    tmp = CYCLE.with_suffix(CYCLE.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(cycle, f, indent=1, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, CYCLE)


def clear_cycle():
    try:
        CYCLE.unlink()
    except FileNotFoundError:
        pass


def make_cycle(sum_best, lines, pin_fence, ship_paths, planned_sum=0.0):
    """Bind one ledger snapshot to every immutable zip that must be reconciled."""
    ships = []
    seen = set()
    for value in ship_paths:
        path = Path(value).resolve()
        digest = sha8(path)
        if digest in seen:
            continue
        seen.add(digest)
        ships.append({"path": str(path), "zip_sha": digest, "submitted": False,
                      "pid": None})
    nonce = f"{time.time_ns()}:{os.getpid()}:{pin_fence}:{lines}"
    return {
        "version": 1,
        "cycle_id": hashlib.sha256(nonce.encode()).hexdigest()[:16],
        "created_ts": f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ}",
        "stage": "planned",
        "pin_fence": pin_fence,
        "sum_best": float(sum_best),
        "ledger_lines": int(lines),
        "planned_sum": float(planned_sum),
        "ships": ships,
    }


def recover_cycle_pids(cycle, entries):
    """Recover submit IDs by exact zip bytes, including after a process crash.

    A matching base SHA is required when probe_log recorded one. Reusing an older result
    for the exact same immutable zip on the exact same pin is safe and avoids a duplicate.
    """
    changed = False
    pin = cycle.get("pin_fence")
    records = list(cycle.get("ships", []))
    if isinstance(cycle.get("union"), dict):
        records.append(cycle["union"])
    for ship in records:
        if ship.get("pid") is not None:
            if not ship.get("submitted"):
                changed = True
            ship["submitted"] = True
            continue
        matches = [e for e in entries
                   if e.get("zip_sha") == ship.get("zip_sha") and e.get("id") is not None
                   and e.get("base_sha") in (None, pin)]
        if not matches:
            continue
        try:
            ship["pid"] = int(matches[-1]["id"])
        except (TypeError, ValueError):
            continue
        ship["submitted"] = True
        changed = True
    return changed


def cycle_file_errors(cycle):
    errors = []
    records = list(cycle.get("ships", []))
    if isinstance(cycle.get("union"), dict):
        records.append(cycle["union"])
    for ship in records:
        path = Path(ship.get("path") or "")
        try:
            actual = sha8(path)
        except OSError:
            errors.append(f"missing {path}")
            continue
        if actual != ship.get("zip_sha"):
            errors.append(f"mutated {path} ({actual} != {ship.get('zip_sha')})")
    return errors


def cycle_unresolved(cycle, outcomes):
    """List every reason the ledger cursor is not yet safe to advance."""
    unresolved = []
    for ship in cycle.get("ships", []):
        label = ship.get("zip_sha") or ship.get("path") or "unknown"
        if not ship.get("submitted"):
            unresolved.append(f"{label}:not-submitted")
            continue
        if ship.get("pid") is None:
            unresolved.append(f"{label}:pid-unattributed")
            continue
        pid = int(ship["pid"])
        if pid not in outcomes and str(pid) not in outcomes:
            unresolved.append(f"{label}:outcome-pending")
    return unresolved


def kaggle_rows():
    r = subprocess.run(["kaggle", "competitions", "submissions", "--csv",
                        "--page-size", "200", COMP],
                       capture_output=True, text=True, timeout=120)
    import io
    rows = list(csv.DictReader(io.StringIO(r.stdout)))
    if r.returncode != 0 or not rows:
        # an empty/failed listing must NEVER read as "0 submissions today" — that would
        # disarm the quota guard exactly when the API is flaky
        raise RuntimeError(f"kaggle submissions listing failed "
                           f"(rc={r.returncode}): {(r.stderr or r.stdout)[:200]}")
    return rows


def probe_log_entries(limit=None):
    """Tolerant line-by-line parse of probe_log.jsonl (skip malformed / non-dict rows)."""
    out = []
    try:
        for line in PROBE_LOG.read_text().splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if isinstance(e, dict):
                out.append(e)
    except OSError:
        pass
    return out[-limit:] if limit else out


def submitted_shas_today():
    today = f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d}"
    shas = set()
    try:
        for line in EVENTS.open():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("kind") == "submitted" and e.get("ts", "").startswith(today) \
                    and e.get("zip_sha"):
                shas.add(e["zip_sha"])
    except OSError:
        pass
    return shas


def quota_today(rows):
    today = f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d}"
    return sum(1 for x in rows if (x.get("date") or "").startswith(today))


LIMIT_CACHE = RESULTS / "KAGGLE_LIMIT.json"
STATIC_DAILY_LIMIT = 100   # last operator-confirmed value; floor of last resort


def daily_limit():
    """Live max_daily_submissions for COMP (repo-audit P0, 2026-07-12): the guard math
    was hard-coded to 100 — wrong in the dangerous direction if Kaggle lowers the limit
    mid-endgame. Read via the SDK at cycle start; persist the last good read; on failure
    fall back to min(cached, static) so a hiccup can never RAISE the assumed limit."""
    limit = None
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        resp = api.competitions_list(search=COMP)
        comps = getattr(resp, "competitions", None) or resp
        for comp in comps or []:
            if str(getattr(comp, "ref", "")).rstrip("/").endswith(COMP):
                v = int(getattr(comp, "max_daily_submissions", 0) or 0)
                limit = v if v > 0 else None
                break
    except Exception as exc:
        log(f"WARN: live daily-limit read failed ({exc}); using cached/static")
    if limit is not None:
        try:
            LIMIT_CACHE.write_text(json.dumps({"limit": limit, "ts": time.time()}))
        except OSError:
            pass
        return limit
    try:
        cached = int(json.loads(LIMIT_CACHE.read_text())["limit"])
        return min(cached, STATIC_DAILY_LIMIT)
    except (OSError, ValueError, KeyError, TypeError):
        return STATIC_DAILY_LIMIT


def auto_subs_today():
    today = f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d}"
    n = 0
    try:
        for line in EVENTS.open():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("kind") == "submitted" and e.get("ts", "").startswith(today):
                n += 1
    except OSError:
        pass
    return n


def repins_today():
    today = f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d}"
    n = 0
    try:
        for line in EVENTS.open():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("kind") == "repin" and e.get("ts", "").startswith(today):
                n += 1
    except OSError:
        pass
    return n


def probe_submit(*args, timeout=300):
    return subprocess.run([sys.executable, str(CLEAN / "candidates" / "probe_submit.py"),
                           *map(str, args)], cwd=CLEAN, capture_output=True, text=True,
                          timeout=timeout)


def planned_delta(plan_lines):
    """Sum unique task deltas from bank plan lines (chunks/aggregates may overlap)."""
    best = {}
    for line in plan_lines:
        for task, value in re.findall(r"(?<!\d)(\d{3})\+([\d.]+)", line):
            best[task] = max(best.get(task, 0.0), float(value))
    return round(sum(best.values()), 4)


def reject_nonpaying_solo(pid, reason):
    """Persist exact bytes only when a reconciled one-swap probe failed to improve."""
    entries = [e for e in probe_log_entries() if e.get("id") == pid]
    if not entries:
        return False
    entry = entries[-1]
    swaps = entry.get("swaps") or []
    zp = Path(entry.get("zip") or "")
    if len(swaps) != 1 or not zp.is_file() or sha8(zp) != entry.get("zip_sha"):
        return False
    task = int(swaps[0]["task"])
    try:
        import zipfile
        data = zipfile.ZipFile(zp).read(f"task{task:03d}.onnx")
    except (OSError, KeyError, zipfile.BadZipFile):
        return False
    rec = {"ts": f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ}",
           "task": task, "sha256": hashlib.sha256(data).hexdigest(), "probe_id": pid,
           "reason": reason}
    with REJECTED.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    event("artifact-rejected", **rec)
    return True


def promote_paying_solo(pid):
    """Promote exact one-swap submitted bytes into the paying corpus after reconciliation."""
    entries = [e for e in probe_log_entries() if e.get("id") == pid]
    if not entries:
        return False
    entry = entries[-1]
    swaps = entry.get("swaps") or []
    zp = Path(entry.get("zip") or "")
    if len(swaps) != 1 or not zp.is_file() or sha8(zp) != entry.get("zip_sha"):
        return False
    task = int(swaps[0]["task"])
    try:
        import zipfile
        data = zipfile.ZipFile(zp).read(f"task{task:03d}.onnx")
    except (OSError, KeyError, zipfile.BadZipFile):
        return False
    full = hashlib.sha256(data).hexdigest()
    stage = RESULTS / "proven_staging" / f"task{task:03d}_{full}.onnx"
    stage.parent.mkdir(parents=True, exist_ok=True)
    if not stage.exists():
        tmp = stage.with_suffix(stage.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, stage)
        os.chmod(stage, 0o444)
    r = subprocess.run([sys.executable, str(HERE / "promote_proven.py"), str(task), str(stage),
                        "--probe-id", str(pid)], capture_output=True, text=True, timeout=300)
    if r.returncode == 0:
        log(r.stdout.strip())
        event("artifact-proven", pid=pid, task=task, sha256=full)
        return True
    log(f"solo #{pid} paid but exact-proof promotion deferred: {(r.stdout + r.stderr)[-240:]}")
    event("alert", reason="proof-promotion-deferred", pid=pid, task=task)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="single pass (default behavior)")
    ap.add_argument("--dry-run", action="store_true", help="stop after the confirm stage")
    ap.add_argument("--force", action="store_true", help="skip the cheap trigger check")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    a = ap.parse_args()

    # ---- rails ----
    lock = open(LOCK, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return  # another auto_submit is mid-flight — silently yield
    if os.environ.get("KAGGLE_AUTOSUBMIT", "") != "1" and not a.dry_run:
        return  # kill-switch

    # A durable in-flight cycle takes precedence over new trigger discovery. Its immutable
    # zips may already exist on Kaggle, so silently starting a new cycle would orphan their
    # PIDs and could consume ledger rows that were never reconciled.
    marker = read_marker()
    try:
        cycle = read_cycle()
    except RuntimeError as exc:
        log(f"ABORT: {exc} — refusing new submissions until the journal is repaired")
        event("abort", reason="cycle-journal-invalid", detail=str(exc))
        return
    if cycle and marker.get("cycle_id") == cycle.get("cycle_id"):
        # Crash after the atomic marker commit but before journal cleanup: the marker is
        # the commit record, so cleanup is idempotent and a new frontier may proceed.
        log(f"recovering completed cycle {cycle['cycle_id']} — marker already committed")
        clear_cycle()
        cycle = None

    # cheap trigger (before any heavy work)
    sum_best, lines, _ = ledger_best_sum()
    # Trigger on rows unseen by the marker, not a subtraction of historical maxima. The
    # latter misses a real current-pin improvement whenever an older absorbed row for the
    # same task had a larger delta on its older basis.
    gained, _, solos = ledger_best_sum(marker.get("ledger_lines", 0))
    flush_due = time.time() - marker.get("last_flush_ts", 0.0) >= FLUSH_HRS * 3600
    trig = f"resume durable cycle {cycle['cycle_id']}" if cycle else None
    if not cycle and gained >= a.threshold:
        trig = f"Σ trigger: +{gained} ≥ {a.threshold}"
    elif not cycle and flush_due and gained >= FLUSH_MIN_SUM:
        trig = f"flush trigger: +{gained} ≥ {FLUSH_MIN_SUM} and {FLUSH_HRS}h elapsed"
    elif (not cycle and gained > SOLO_MIN_SUM
          and lines > int(marker.get("ledger_lines", 0) or 0)):
        # Any unseen local win may recompute as solo-only even if its bounded ledger note
        # predates structured risk tagging. Bank is the exact-byte routing authority.
        trig = f"new-candidate trigger: +{gained} across {lines - int(marker.get('ledger_lines', 0) or 0)} row(s)"
    if not trig and not a.force:
        return  # below threshold — the 300s tick costs nothing
    log(f"TRIGGERED ({trig or 'forced'}); raw ledger Σ={sum_best} marker={marker.get('sum_best')}")

    sys.path.insert(0, str(HERE))
    from integrity import integrity_errors
    errs = integrity_errors()
    if errs:
        log(f"ABORT: integrity errors — never submit on a drifted scorer: {errs}")
        event("abort", reason="integrity", detail=str(errs))
        return
    pin = json.loads(PIN_JSON.read_text())
    live = CLEAN / "submission.zip"
    rec = (ROOT / "data" / "PIN_SHA.txt").read_text().strip()
    if not (sha8(live) == pin["sha"] == rec):
        log(f"ABORT: pin triple-check failed (zip {sha8(live)} / kaggle_pin {pin['sha']} / "
            f"PIN_SHA.txt {rec}) — run runner/repin.py first")
        event("abort", reason="pin-inconsistent")
        return
    pin_fence = pin["sha"]  # captured NOW; re-checked before every promotion step below

    if cycle and cycle.get("stage") == "committing":
        # A crash between marker replace and journal unlink is handled above. A crash just
        # before marker replace resumes the commit here without resubmitting or repinning.
        if cycle.get("commit_pin_sha") != pin_fence:
            log(f"ABORT: committing cycle expected pin {cycle.get('commit_pin_sha')} but "
                f"live pin is {pin_fence}; marker remains untouched")
            event("abort", reason="cycle-commit-pin-mismatch",
                  cycle_id=cycle.get("cycle_id"), pin=pin_fence)
            return
        write_marker(cycle["sum_best"], cycle["ledger_lines"], pin_fence,
                     flushed=True, cycle_id=cycle["cycle_id"])
        clear_cycle()
        log(f"recovered marker commit for cycle {cycle['cycle_id']}")
        event("cycle-commit-recovered", cycle_id=cycle["cycle_id"])
        return

    if cycle and cycle.get("stage") == "repinning" \
            and pin_fence == cycle.get("expected_pin_sha"):
        # repin.py completed and refreshed the pin, then auto_submit died before recording
        # the marker. The exact promoted zip hash proves which transaction completed.
        try:
            paid = float(pin.get("score")) >= float(cycle.get("best_actual")) - 0.005
        except (TypeError, ValueError):
            paid = False
        if not paid:
            log("ABORT: repin hash matches the journal but its recorded score does not; "
                "marker remains untouched")
            event("abort", reason="cycle-repin-score-mismatch",
                  cycle_id=cycle.get("cycle_id"))
            return
        cycle["stage"] = "committing"
        cycle["commit_pin_sha"] = pin_fence
        write_cycle(cycle)
        write_marker(cycle["sum_best"], cycle["ledger_lines"], pin_fence,
                     flushed=True, cycle_id=cycle["cycle_id"])
        clear_cycle()
        log(f"recovered completed repin and marker for cycle {cycle['cycle_id']}")
        event("cycle-repin-recovered", cycle_id=cycle["cycle_id"], pin=pin_fence)
        return

    if cycle and cycle.get("pin_fence") != pin_fence:
        # Never submit an old-base zip or pretend its unresolved probes belong to the new
        # frontier. Keeping the journal is deliberately fail-closed and leaves every row
        # pending for explicit reconciliation.
        log(f"ABORT: pending cycle {cycle['cycle_id']} was built on pin "
            f"{cycle.get('pin_fence')} but live pin is {pin_fence}; journal and marker kept")
        event("abort", reason="cycle-pin-moved", cycle_id=cycle.get("cycle_id"),
              was=cycle.get("pin_fence"), now=pin_fence)
        return

    bank_base = None
    if cycle is None:
        # Submission attribution is valid only when bank zips are based on the exact
        # Kaggle prediction pin. A freshly generated "hardened" derivative is not proof:
        # it adds a hidden extra swap to every nominal solo. The exact confirmed live pin
        # remains the base even if its inherited task144 razor is noisy; final=best protects
        # the standing score, while exact attribution is preserved.
        razor = subprocess.run([sys.executable, str(HERE / "scan_razors.py"),
                                "--base", str(live)], capture_output=True, text=True,
                               timeout=300)
        if razor.returncode != 0:
            log("WARN: exact Kaggle-confirmed live pin carries an inherited razor; using "
                "those exact bytes rather than an unproven hardened derivative.\n" +
                (razor.stdout + razor.stderr).strip()[-300:])
            event("warning", reason="confirmed-live-pin-razor")
        bank_base = live

    quota_known = True
    try:
        rows = kaggle_rows()
    except Exception as exc:
        if cycle is None:
            log(f"ABORT: kaggle submissions unreachable: {exc}")
            event("abort", reason="kaggle-unreachable")
            return
        # Reconciliation itself spends no quota. Continue for already-submitted PIDs, but
        # refuse missing submissions and the optional union while the quota is unknown.
        quota_known, rows = False, []
        log(f"Kaggle listing unavailable during recovery ({exc}); polling only")
    limit = daily_limit()
    n_today = quota_today(rows) if quota_known else limit
    n_auto = auto_subs_today()

    if cycle is None:
        # ---- confirm: bank --dry-run gives the REPRICED truth (raw ledger deltas lie) ----
        log(f"confirm stage: bank --dry-run (base={bank_base.name}; repricing; minutes)…")
        b = subprocess.run([sys.executable, str(HERE / "bank.py"), "--dry-run",
                            "--base", str(bank_base)], capture_output=True, text=True,
                           timeout=7200, cwd=ROOT)
        plan = [l for l in b.stdout.splitlines()
                if re.match(r"wave_(\d+|ALL)", l.strip())]
        planned_sum = planned_delta(plan)
        log(f"bank plan: {len(plan)} zip(s), repriced Σ=+{planned_sum} "
            f"(raw ledger claimed +{gained})")
        for line in plan:
            log("  " + line.strip())
        if b.returncode != 0:
            if not plan:
                # log the tails (wave6 audit: 5 such events were undiagnosable because
                # bank's sys.exit() message goes to stderr, which this branch discarded)
                log(f"nothing survives repricing (bank rc={b.returncode}) — raw +{gained} "
                    "may be phantom OR a transient bank failure; marker kept fail-closed")
                log(f"  bank stdout tail: {b.stdout[-300:].strip()!r}")
                log(f"  bank stderr tail: {b.stderr[-300:].strip()!r}")
                event("abort", reason="bank-empty-unstructured", raw=gained,
                      bank_rc=b.returncode, stderr_tail=b.stderr[-300:])
                return
            log(f"ABORT: bank --dry-run rc={b.returncode}: {b.stdout[-400:]}")
            event("abort", reason="bank-dryrun-failed", rc=b.returncode)
            return
        if planned_sum <= 0:
            log(f"repriced Σ +{planned_sum} does not confirm the trigger, but bank emits no "
                "structured all-rows-absorbed proof — marker kept fail-closed")
            event("abort", reason="bank-zero-unstructured", raw=gained,
                  repriced=planned_sum)
            return
        if a.dry_run:
            log(f"DRY-RUN: would ship {len(plan)} zip(s), Σ+{planned_sum}; quota "
                f"n_today={n_today} auto_today={n_auto}")
            return
        planned_n = len(plan) + 2  # + union + margin
        if (n_today + planned_n > limit - QUOTA_RESERVE
                or n_auto + planned_n > AUTO_MAX_SUBS_PER_DAY):
            log(f"ABORT: quota guard (today {n_today}/{limit}, "
                f"auto {n_auto}/{AUTO_MAX_SUBS_PER_DAY}, planned {planned_n}, "
                f"reserve {QUOTA_RESERVE})")
            event("abort", reason="quota-guard", n_today=n_today, n_auto=n_auto)
            return

        # ---- ship: bank --regate emits razor-clean zips + exact submit commands ----
        log(f"ship stage: bank --regate (base={bank_base.name}; 2nd secret stream)…")
        b = subprocess.run([sys.executable, str(HERE / "bank.py"), "--regate",
                            "--base", str(bank_base)], capture_output=True, text=True,
                           timeout=14400, cwd=ROOT)
        ship = re.findall(r"probe_submit\.py submit (\S+) --lane", b.stdout)
        if b.returncode != 0:
            log(f"ship-stage bank failed rc={b.returncode}; refusing its potentially partial "
                f"{len(ship)}-zip output and retaining the full ledger frontier")
            event("abort", reason="bank-ship-partial", rc=b.returncode, emitted=len(ship))
            return
        if not ship:
            log(f"nothing razor-clean to ship (bank rc={b.returncode}); "
                f"tail:\n{b.stdout[-400:]}")
            event("abort", reason="nothing-shippable", rc=b.returncode)
            return
        ship.sort(key=lambda p: ("SOLO" not in p, "wave_ALL" in os.path.basename(p), p))
        cycle = make_cycle(sum_best, lines, pin_fence, ship, planned_sum)
        write_cycle(cycle)  # durable intent exists before the first external submit
        event("cycle-started", cycle_id=cycle["cycle_id"],
              ledger_lines=cycle["ledger_lines"], ships=len(cycle["ships"]))
    else:
        errors = cycle_file_errors(cycle)
        if errors:
            log(f"ABORT: durable cycle zip integrity failed: {errors}; marker untouched")
            event("abort", reason="cycle-zip-integrity", detail=errors,
                  cycle_id=cycle["cycle_id"])
            return
        if a.dry_run:
            log(f"DRY-RUN: would resume cycle {cycle['cycle_id']} with "
                f"{len(cycle['ships'])} immutable zip(s)")
            return

    if recover_cycle_pids(cycle, probe_log_entries()):
        write_cycle(cycle)
    missing = sum(1 for s in cycle["ships"] if not s.get("submitted"))
    if missing and (not quota_known
                    or n_today + missing + 2 > limit - QUOTA_RESERVE
                    or n_auto + missing + 2 > AUTO_MAX_SUBS_PER_DAY):
        log(f"ABORT: recovery quota guard leaves {missing} zip(s) unsubmitted; durable "
            "cycle retained for the next quota window")
        event("abort", reason="cycle-quota-guard", missing=missing,
              cycle_id=cycle["cycle_id"])
        return

    already = submitted_shas_today()
    submitted_now = 0
    for ship_rec in cycle["ships"]:
        zp, zsha = ship_rec["path"], ship_rec["zip_sha"]
        if ship_rec.get("pid") is not None:
            continue
        if ship_rec.get("submitted"):
            log(f"waiting to recover PID attribution for already-submitted {os.path.basename(zp)}")
            continue
        if zsha in already:
            ship_rec["submitted"] = True
            write_cycle(cycle)
            log(f"identical zip {os.path.basename(zp)} was submitted today but its PID is "
                "not yet attributable; journal retained")
            continue
        t_submit = f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ}"
        r = probe_submit("submit", zp, "--lane", "auto-w5",
                         "--note", f"auto_submit {datetime.datetime.now(datetime.timezone.utc):%m%dT%H%M}")
        out = (r.stdout + r.stderr).strip()
        if r.returncode != 0:
            log(f"SUBMIT REFUSED for {zp} (rc={r.returncode}): {out[-300:]}")
            event("alert", reason="submit-refused", zip=zp, detail=out[-200:])
            log("aborting the remaining batch — likely a stale pin (another lane banked); "
                  "operator: re-pin then re-run")
            break
        ship_rec["submitted"] = True
        ship_rec["submitted_ts"] = t_submit
        write_cycle(cycle)
        submitted_now += 1
        # attribute OUR entry by zip_sha (+ recency), never by "last line" — a concurrent
        # manual probe_submit could interleave, and malformed lines must not crash us
        mine = [e for e in probe_log_entries(limit=25)
                if e.get("zip_sha") == zsha and (e.get("ts") or "") >= t_submit[:16]]
        if mine:
            pid = int(mine[-1]["id"])
            ship_rec["pid"] = pid
            write_cycle(cycle)
            log(f"submitted #{pid:03d} {os.path.basename(zp)} pred={mine[-1].get('predicted')}")
            event("submitted", pid=pid, zip=zp, zip_sha=zsha,
                  predicted=mine[-1].get("predicted"))
        else:
            # the CLI succeeded but we can't find our record: do NOT resubmit — record the
            # sha so the next cycle's double-submit guard holds, and alert
            log(f"submitted {zp} but no matching probe_log record (sha {zsha}) — "
                f"recorded as submitted; recon will pick its result up by description")
            event("submitted", pid=None, zip=zp, zip_sha=zsha, predicted=None)
            event("alert", reason="pid-attribution-failed", zip=zp)

    if recover_cycle_pids(cycle, probe_log_entries()):
        write_cycle(cycle)
    my_pids = sorted({int(s["pid"]) for s in cycle["ships"] if s.get("pid") is not None})
    if not my_pids:
        log(f"cycle {cycle['cycle_id']} has no attributable PIDs yet; journal retained")
        return

    # ---- poll + recon ----
    log(f"polling recon for pids {my_pids} (≤45 min)…")
    cycle["stage"] = "polling"
    write_cycle(cycle)
    t0 = time.time()
    results = {}

    def collect_outcomes():
        try:
            oc = json.loads(OUTCOMES.read_text())
        except (OSError, ValueError):
            return
        for pid in my_pids:
            result = oc.get(str(pid), oc.get(pid))
            if result is not None and pid not in results:
                results[pid] = result
                log(f"#{pid:03d} → actual={result.get('actual')} gap={result.get('gap')}")

    collect_outcomes()  # restart fast path: do not sleep when Kaggle already reconciled
    while time.time() - t0 < 2700 and len(results) < len(my_pids):
        time.sleep(120)
        probe_submit("recon", timeout=300)
        collect_outcomes()

    unresolved = cycle_unresolved(cycle, results)
    if unresolved:
        cycle["stage"] = "awaiting-outcomes"
        cycle["unresolved"] = unresolved
        write_cycle(cycle)
        log(f"cycle {cycle['cycle_id']} incomplete after polling: {unresolved}; marker "
            "untouched and exact PIDs retained for the next recovery tick")
        event("cycle-pending", cycle_id=cycle["cycle_id"], unresolved=unresolved)
        return

    cycle["stage"] = "reconciled"
    cycle.pop("unresolved", None)
    write_cycle(cycle)
    entries_by_pid = {}
    for entry in probe_log_entries():
        try:
            entries_by_pid[int(entry.get("id"))] = entry
        except (TypeError, ValueError):
            continue
    transferred = {}
    for pid, result in results.items():
        entry = entries_by_pid.get(pid) or {}
        try:
            improved = float(result.get("actual")) > float(entry.get("base_score")) + 0.005
        except (TypeError, ValueError):
            improved = False
        if improved:
            transferred[pid] = result
        else:
            reject_nonpaying_solo(pid, "reconciled probe did not improve its base")
    shorts = {p: r for p, r in results.items() if p not in transferred}
    for pid, r in shorts.items():
        log(f"⚠ NON-PAYING #{pid:03d}: actual={r.get('actual')} gap={r.get('gap')} — "
            f"exact solo bytes quarantined when attributable")
        event("alert", reason="non-paying", pid=pid, gap=r.get("gap"))

    if not transferred:
        # Every planned immutable zip has an attributable outcome. Repeating exact
        # non-paying bytes cannot gain points; solos were quarantined above. Consume only
        # this cycle's captured cursor, never rows appended while it was in flight.
        cycle["stage"] = "committing"
        cycle["commit_pin_sha"] = sha8(live)
        write_cycle(cycle)
        write_marker(cycle["sum_best"], cycle["ledger_lines"], sha8(live),
                     flushed=True, cycle_id=cycle["cycle_id"])
        clear_cycle()
        log("all planned probes reconciled non-paying — exact bytes quarantined where "
            "attributable; captured ledger cursor committed")
        event("cycle-complete", best=None, marker_sum=cycle["sum_best"],
              ledger_lines=cycle["ledger_lines"], cycle_id=cycle["cycle_id"])
        return

    for pid in transferred:
        promote_paying_solo(pid)

    # ---- promote: best actual (single) or union of transferred zips ----
    # PIN FENCE: another lane may have repinned during the ≤45-min poll. Promoting a
    # probe built on the OLD pin would silently revert their wins — verify the live pin
    # is unchanged AND every promoted probe was predicted against it.
    def fence_ok(stage):
        try:
            cur = sha8(live)
            reg = json.loads(PIN_JSON.read_text()).get("sha")
        except (OSError, ValueError):
            cur = reg = None
        if cur != pin_fence or reg != pin_fence:
            log(f"PIN FENCE at {stage}: pin moved {pin_fence} -> {cur}/{reg} mid-cycle — "
                f"promotion aborted (wins stay pending; next cycle re-banks on the new pin)")
            event("alert", reason="pin-fence", stage=stage, was=pin_fence, now=cur)
            return False
        return True

    plog = {}
    for entry in probe_log_entries():
        try:
            pid = int(entry.get("id"))
        except (TypeError, ValueError):
            continue
        if pid in transferred:
            plog[pid] = entry
    stale_base = [p for p in list(transferred)
                  if plog.get(p, {}).get("base_sha") not in (pin_fence, None)]
    for p in stale_base:
        log(f"#{p:03d} predicted against base {plog[p].get('base_sha')} ≠ fence {pin_fence} — "
            f"excluded from promotion")
        transferred.pop(p)
    if not transferred or not fence_ok("promote"):
        if transferred:
            log("transferred wins recorded; promotion deferred")
        return
    best_pid = max(transferred, key=lambda p: transferred[p].get("actual") or 0)
    best_actual = transferred[best_pid]["actual"]
    winner_zip = Path(plog[best_pid]["zip"])
    if len(transferred) > 1:
        if not fence_ok("union-build"):
            return
        import zipfile
        union_rec = cycle.get("union")
        if not isinstance(union_rec, dict):
            # Build once, atomically, then bind its exact bytes into the same journal before
            # the external submit. A restart always reuses this immutable union.
            outdir = RESULTS / "bank"
            outdir.mkdir(parents=True, exist_ok=True)
            tmp = outdir / f"union_{cycle['cycle_id']}.zip.tmp"
            union = outdir / f"union_{cycle['cycle_id']}.zip"
            changed = {}
            for pid in transferred:
                for swap in plog[pid]["swaps"]:
                    member = f"task{int(swap['task']):03d}.onnx"
                    changed[member] = (plog[pid]["zip"], member)
            try:
                with zipfile.ZipFile(live) as base, \
                        zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
                    for name in base.namelist():
                        if name in changed:
                            src, member = changed[name]
                            with zipfile.ZipFile(src) as source:
                                z.writestr(name, source.read(member))
                        else:
                            z.writestr(name, base.read(name))
                os.replace(tmp, union)
                os.chmod(union, 0o444)
            finally:
                try:
                    tmp.unlink()
                except FileNotFoundError:
                    pass
            union_rec = {"path": str(union.resolve()), "zip_sha": sha8(union),
                         "submitted": False, "pid": None, "razor_clean": False}
            cycle["union"] = union_rec
            cycle["stage"] = "union-planned"
            write_cycle(cycle)
        else:
            union = Path(union_rec["path"])

        # The journal may have been checkpointed immediately before this scan. Never infer
        # scan success merely from file existence after a restart.
        union_rec = cycle["union"]
        if not union_rec.get("razor_clean"):
            r = subprocess.run([sys.executable, str(HERE / "scan_razors.py"),
                                "--base", str(union)], capture_output=True, text=True)
            if r.returncode != 0:
                cycle["stage"] = "union-razor-failed"
                write_cycle(cycle)
                log("union razor-scan FAILED — marker untouched; all paying solos remain "
                    "journaled for diagnosis")
                event("abort", reason="union-razor-failed", cycle_id=cycle["cycle_id"])
                return
            union_rec["razor_clean"] = True
            cycle["stage"] = "union-ready"
            write_cycle(cycle)

        if recover_cycle_pids(cycle, probe_log_entries()):
            write_cycle(cycle)
        union_rec = cycle["union"]
        if not union_rec.get("submitted"):
            if (not quota_known
                    or n_today + submitted_now + 1 > limit - QUOTA_RESERVE
                    or n_auto + submitted_now + 1 > AUTO_MAX_SUBS_PER_DAY):
                cycle["stage"] = "union-quota-wait"
                write_cycle(cycle)
                log("union deferred by quota guard — marker untouched; durable cycle retained")
                event("abort", reason="union-quota-guard", cycle_id=cycle["cycle_id"])
                return
            r = probe_submit("submit", union, "--lane", "auto-union",
                             "--note", f"union of {sorted(transferred)}")
            if r.returncode != 0:
                log(f"union submit refused: {(r.stdout + r.stderr)[-240:]}; journal retained")
                event("alert", reason="union-submit-refused", cycle_id=cycle["cycle_id"])
                return
            union_rec["submitted"] = True
            union_rec["submitted_ts"] = (
                f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ}")
            write_cycle(cycle)
            submitted_now += 1
            if recover_cycle_pids(cycle, probe_log_entries()):
                write_cycle(cycle)
            union_rec = cycle["union"]
            event("submitted", pid=union_rec.get("pid"), zip=str(union),
                  zip_sha=union_rec["zip_sha"], predicted=None)

        upid = union_rec.get("pid")
        if upid is None:
            cycle["stage"] = "union-pid-unattributed"
            write_cycle(cycle)
            log("union was submitted but its PID is not attributable yet — marker untouched")
            return
        upid = int(upid)
        u = union_rec.get("outcome")
        t1 = time.time()
        while u is None and time.time() - t1 < 2700:
            try:
                oc = json.loads(OUTCOMES.read_text())
            except (OSError, ValueError):
                oc = {}
            u = oc.get(str(upid), oc.get(upid))
            if u is not None:
                break
            time.sleep(120)
            probe_submit("recon", timeout=300)
        if u is None:
            cycle["stage"] = "union-awaiting-outcome"
            write_cycle(cycle)
            log(f"union #{upid:03d} still pending — marker untouched; PID retained")
            event("cycle-pending", cycle_id=cycle["cycle_id"],
                  unresolved=[f"union:{upid}:outcome-pending"])
            return
        union_rec["outcome"] = u
        cycle["stage"] = "union-reconciled"
        write_cycle(cycle)
        log(f"union #{upid:03d} → actual={u.get('actual')} gap={u.get('gap')}")
        try:
            union_actual = float(u.get("actual"))
        except (TypeError, ValueError):
            union_actual = float("-inf")
        if union_actual <= float(best_actual) + 0.005:
            cycle["stage"] = "union-nonpaying"
            write_cycle(cycle)
            log("union did not combine the individually paying probes — refusing to consume "
                "their ledger rows; operator reconciliation required")
            event("alert", reason="union-nonpaying", pid=upid,
                  actual=u.get("actual"), best_single=best_actual)
            return
        best_actual, winner_zip = union_actual, union

    # ---- repin (guarded; a FAILED repin must NOT absorb the trigger) ----
    now_iso = f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M}"
    repin_needed = best_actual > float(pin["score"]) + 0.005
    repin_ok = True
    if not repin_needed:
        log(f"no improvement over pin ({best_actual} vs {pin['score']}) — no repin")
    elif now_iso >= REPIN_FREEZE_AT:
        log(f"REPIN FROZEN (≥ {REPIN_FREEZE_AT}) — new best {best_actual} recorded, pin kept")
        cycle["stage"] = "repin-frozen"
        cycle["best_actual"] = best_actual
        write_cycle(cycle)
        repin_ok = False
        event("repin-frozen", best=best_actual)
    elif repins_today() >= REPIN_MAX_PER_DAY:
        log(f"repin cap reached ({REPIN_MAX_PER_DAY}/day) — new best {best_actual} recorded")
        cycle["stage"] = "repin-capped"
        cycle["best_actual"] = best_actual
        write_cycle(cycle)
        repin_ok = False
        event("repin-capped", best=best_actual)
    elif not fence_ok("repin"):
        repin_ok = False
    else:
        log(f"PROMOTING {winner_zip.name} @ {best_actual} → repin + full refresh")
        cycle["stage"] = "repinning"
        cycle["expected_pin_sha"] = sha8(winner_zip)
        cycle["best_actual"] = best_actual
        write_cycle(cycle)
        event("repin", zip=str(winner_zip), score=best_actual)
        r = subprocess.run([sys.executable, str(HERE / "repin.py"), "--zip", str(winner_zip),
                            "--score", str(best_actual)],
                           capture_output=True, text=True, timeout=10800, cwd=ROOT)
        log(f"repin rc={r.returncode}; tail:\n" + "\n".join(r.stdout.splitlines()[-6:]))
        repin_ok = r.returncode == 0
        if repin_ok:
            rec2 = (ROOT / "data" / "PIN_SHA.txt").read_text().strip()
            pin2 = json.loads(PIN_JSON.read_text())
            repin_ok = sha8(live) == pin2.get("sha") == rec2
        if not repin_ok:
            log("⚠ repin FAILED or left the pin inconsistent — marker NOT updated (trigger "
                "stays armed); Kaggle best is ahead of the local pin, so probe_submit's "
                "freshness gate blocks further submits until the operator re-runs repin.py")
            event("alert", reason="promotion-pending", zip=str(winner_zip), score=best_actual)

    if repin_ok:
        # Commit only the ledger cursor captured before planning. Wins appended during this
        # potentially hours-long cycle were not in its immutable intent and must stay pending.
        # Journal "committing" first so a crash on either side of marker replace is recoverable.
        cycle["stage"] = "committing"
        cycle["commit_pin_sha"] = sha8(live)
        cycle["best_actual"] = best_actual
        write_cycle(cycle)
        write_marker(cycle["sum_best"], cycle["ledger_lines"], sha8(live),
                     flushed=True, cycle_id=cycle["cycle_id"])
        clear_cycle()
        log(f"cycle complete: marker Σ={cycle['sum_best']} rows={cycle['ledger_lines']}; "
            f"kaggle best now {best_actual}")
        event("cycle-complete", best=best_actual, marker_sum=cycle["sum_best"],
              ledger_lines=cycle["ledger_lines"], cycle_id=cycle["cycle_id"])


if __name__ == "__main__":
    main()
