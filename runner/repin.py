#!/usr/bin/env python3
"""repin — ONE command that rotates the pin and refreshes every artifact derived from it.

The pin (CLEAN/submission.zip, mirrored in logs/kaggle_pin.json) is the single source of
truth; bars, deltas, strategies, donors, queue order, difficulty and packs all derive from
it. Historically the refresh ritual was five separate commands plus printed reminders —
this chains them, logs each step, and prints a before/after summary.

Usage (repo root or anywhere):
  python3 runner/repin.py                                # refresh against the CURRENT pin
  python3 runner/repin.py --zip <new_best.zip> --score 7375.33   # rotate pin THEN refresh
  python3 runner/repin.py --dry-run                      # print the step plan, touch nothing
  python3 runner/repin.py --fast                         # skip packs + reprice + harden (data only)

Steps: 0 stage zip → submission.zip + pinned_bases + kaggle_pin.json · 1 rescore_pin ·
2 refresh_basis (QUEUE numbers, PIN_SHA, strategies, donors) · 3 rank_difficulty ·
4 build_queue --reprioritize · 5 make_pack × every QUEUE task · 6 factor_sweep tickets ·
7 reprice_artifacts (phantom audit) · 8 pad_biases re-harden + full-gate padded members ·
9 publish a generation manifest.

The default is deliberately offline-only. A pin rotation exposes mixed derived state until
the final manifest is published; preflight and banking reject that window.
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
from config import CLEAN as _CLEAN
CLEAN = Path(_CLEAN)
PIN_JSON = CLEAN / "logs" / "kaggle_pin.json"
PINNED_BASES = CLEAN / "logs" / "pinned_bases"
LEDGER = ROOT / "results" / "LEDGER.csv"
LOG = ROOT / "results" / "REPIN.log"
INCOMPLETE = ROOT / "results" / "REPIN_INCOMPLETE.json"
MANIFEST = ROOT / "results" / "REPIN_MANIFEST.json"
(ROOT / "results").mkdir(parents=True, exist_ok=True)  # runtime dir, created on demand
BASIS_FILES = (
    "data/PIN_SHA.txt",
    "data/PIN_CSV_SHA.txt",
    "data/strategies.json",
    "data/donors.json",
    "data/pin_pv2_70_263_pertask.csv",
    "data/lanes/QUEUE.csv",
    "data/lanes/STRATEGY_INDEX.md",
    "results/DIFFICULTY.csv",
)


def sha8(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:8]


def basis_snapshot() -> tuple[dict[str, str], float]:
    """Return an exact content fingerprint and newest mtime for pack inputs."""
    hashes = {}
    mtimes = []
    for rel in BASIS_FILES:
        path = ROOT / rel
        hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
        mtimes.append(path.stat().st_mtime)
    return hashes, max(mtimes)


def log(msg: str):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    print(line, flush=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


def atomic_json(path: Path, payload: dict):
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def wave_activity() -> list[str]:
    reasons = []
    import fcntl
    launch_lock = open(ROOT / "results" / "LAUNCH.lock", "a+")
    try:
        fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(launch_lock, fcntl.LOCK_UN)
    except OSError:
        reasons.append("launch preflight is active")
    pidfile = ROOT / "results" / "orchestrator.pid"
    try:
        pid = int(pidfile.read_text().strip())
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        if "orchestrate.py" in cmd:
            reasons.append(f"orchestrator pid {pid} is active")
    except (OSError, ValueError):
        pass
    try:
        fleet = json.loads((ROOT / "results" / "FLEET.json").read_text())
        live = {p: int(fleet.get(p, 0)) for p in ("codex",)}
        if any(live.values()):
            reasons.append(f"fleet is not drained: {live}")
    except (OSError, ValueError, TypeError):
        reasons.append("FLEET.json is missing or unreadable")
    writers = ("runner/make_pack.py", "runner/reprice_artifacts.py",
               "runner/pending_report.py", "runner/factor_sweep.py",
               "runner/pad_biases.py", "runner/refresh_basis.py",
               "runner/rank_difficulty.py", "runner/build_queue.py")
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except OSError:
            continue
        if any(writer in cmd for writer in writers):
            found.append(proc.name)
    if found:
        reasons.append("derived-state writers are active: " + ",".join(found[:12]))
    return reasons


def run(cmd, step, cwd=ROOT, timeout=3600, env=None):
    log(f"[{step}] $ {' '.join(map(str, cmd))}")
    r = subprocess.run(list(map(str, cmd)), cwd=cwd, timeout=timeout,
                       capture_output=True, text=True,
                       env={**os.environ, **(env or {})})
    tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-8:])
    for ln in tail.splitlines():
        log(f"[{step}]   {ln}")
    if r.returncode != 0:
        log(f"[{step}] FAILED rc={r.returncode} — repin aborted")
        sys.exit(r.returncode)
    return r.stdout


def csv_sum(path: Path) -> float:
    try:
        with path.open() as fh:
            return round(sum(float(r["score"]) for r in csv.DictReader(fh) if r.get("score")), 2)
    except OSError:
        return 0.0


def ledger_lines() -> int:
    try:
        return sum(1 for _ in LEDGER.open()) - 1
    except OSError:
        return 0


def one_pack(task: int) -> tuple:
    r = subprocess.run([sys.executable, str(HERE / "make_pack.py"), str(task)],
                       cwd=ROOT, capture_output=True, text=True, timeout=600)
    return task, r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", help="new best zip to rotate the pin to (default: current pin)")
    ap.add_argument("--score", type=float, help="Kaggle-confirmed score of --zip")
    ap.add_argument("--workers", type=int, default=4, help="rescore_pin workers")
    ap.add_argument("--pack-procs", type=int, default=4, help="make_pack parallelism")
    ap.add_argument("--allow-live", action="store_true",
                    help="expert override: permit refresh with a live orchestrator/fleet")
    ap.add_argument("--skip-rescore", action="store_true",
                    help="resume after a separately completed rescore_pin run")
    ap.add_argument("--resume-after-basis", action="store_true",
                    help="reuse an already-completed basis/difficulty/queue stage")
    ap.add_argument("--resume-packs-since", type=float,
                    help="regenerate only packs older than this epoch (requires --resume-after-basis)")
    ap.add_argument("--force-pack", default="",
                    help="comma-separated tasks to regenerate even when newer than --resume-packs-since")
    ap.add_argument("--fast", action="store_true", help="data refresh only (skip packs/reprice/harden)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.resume_packs_since is not None and not a.resume_after_basis:
        sys.exit("--resume-packs-since requires --resume-after-basis")

    pin_csv = ROOT / "data" / "pin_pv2_70_263_pertask.csv"
    old_sha = (ROOT / "data" / "PIN_SHA.txt").read_text().strip()
    old_sum = csv_sum(pin_csv)
    old_pin = json.loads(PIN_JSON.read_text()) if PIN_JSON.exists() else {}
    ledger_before = ledger_lines()

    if a.zip:
        if a.score is None:
            sys.exit("--zip requires --score (the Kaggle-confirmed score)")
        zp = Path(a.zip).resolve()
        if not zp.exists():
            sys.exit(f"zip not found: {zp}")
        new_sha = sha8(zp)
    else:
        zp = CLEAN / "submission.zip"
        new_sha = sha8(zp)
        a.score = a.score or old_pin.get("score")

    tasks = []
    qcsv = ROOT / "data" / "lanes" / "QUEUE.csv"
    with qcsv.open() as fh:
        for r in csv.DictReader(fh):
            t = r.get("task") or ""
            digits = "".join(c for c in t if c.isdigit())
            if digits:
                tasks.append(int(digits))
    tasks = sorted(set(tasks))

    plan = [
        f"0. stage {zp.name} (sha {new_sha}) -> submission.zip + pinned_bases + kaggle_pin.json @ {a.score}",
        f"1. rescore_pin --workers {a.workers}" + (" [SKIP --skip-rescore]" if a.skip_rescore else ""),
        "2. refresh_basis (QUEUE numbers, PIN_SHA, compile_strategies, find_donors)" +
        (" [REUSE --resume-after-basis]" if a.resume_after_basis else ""),
        "3. rank_difficulty" + (" [REUSE]" if a.resume_after_basis else ""),
        "4. build_queue --reprioritize" + (" [REUSE]" if a.resume_after_basis else ""),
        f"5. make_pack x {len(tasks)} queue tasks ({a.pack_procs} procs)" + (" [SKIP --fast]" if a.fast else ""),
        "6. factor_sweep tickets" + (" [SKIP --fast]" if a.fast else ""),
        "7. reprice_artifacts (phantom audit)" + (" [SKIP --fast]" if a.fast else ""),
        "8. pad_biases re-harden + full-gate padded members" + (" [SKIP --fast]" if a.fast else ""),
        "9. publish generation manifest",
    ]
    print(f"REPIN PLAN  pin {old_sha}@{old_pin.get('score')} -> {new_sha}@{a.score}")
    for p in plan:
        print("  " + p)
    if a.dry_run:
        print("(dry-run — nothing executed)")
        return

    import fcntl
    lock = open(ROOT / "results" / "REPIN.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("another repin is mid-flight (results/REPIN.lock held) — not starting")

    activity = wave_activity()
    if activity and not a.allow_live:
        sys.exit("refusing to repin while the wave is active: " + "; ".join(activity))

    generation = {
        "schema": 1, "status": "incomplete", "pid": os.getpid(),
        "generated_by": "runner/repin.py",
        "run_nonce": hashlib.sha256(os.urandom(32)).hexdigest()[:20],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_pin": old_sha, "target_pin": new_sha, "score": a.score,
        "step": "locked", "fast": bool(a.fast),
    }
    atomic_json(INCOMPLETE, generation)

    def progress(step):
        generation["step"] = step
        generation["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        atomic_json(INCOMPLETE, generation)

    log(f"=== REPIN start: {old_sha}@{old_pin.get('score')} -> {new_sha}@{a.score} ===")

    # 0. stage — validate BEFORE touching the live pin; install atomically (os.replace)
    live = CLEAN / "submission.zip"
    progress("0-stage")
    if a.zip and sha8(live) != new_sha:
        import zipfile
        try:
            new_names = set(zipfile.ZipFile(zp).namelist())
            live_names = set(zipfile.ZipFile(live).namelist())
        except zipfile.BadZipFile as exc:
            sys.exit(f"candidate zip unreadable ({exc}) — live pin untouched")
        if new_names != live_names:
            sys.exit(f"candidate member set differs from the live pin "
                     f"(+{len(new_names - live_names)}/-{len(live_names - new_names)}) — "
                     f"refusing to stage; live pin untouched")
        bak = PINNED_BASES / f"_prev_live_submission_{sha8(live)}.zip.bak"
        PINNED_BASES.mkdir(parents=True, exist_ok=True)
        shutil.copy2(live, bak)
        log(f"[0] archived previous live pin -> {bak.name}")
        tmp_live = live.parent / (live.name + ".staging")
        shutil.copy2(zp, tmp_live)
        os.replace(tmp_live, live)  # atomic for concurrent readers
        log(f"[0] installed {zp.name} as submission.zip (atomic)")
    archive = PINNED_BASES / f"submission_{new_sha}_{str(a.score).replace('.', 'p')}.zip"
    if not archive.exists():
        PINNED_BASES.mkdir(parents=True, exist_ok=True)
        shutil.copy2(live, archive)
    run([sys.executable, CLEAN / "candidates" / "probe_submit.py", "pin", archive, a.score],
        "0-pin", cwd=CLEAN)

    # 1..4 — the data basis
    def rescore_bad_rows():
        bad = []
        with pin_csv.open() as fh:
            for r in csv.DictReader(fh):
                if r.get("status") != "ok" or not (r.get("score") or "").strip():
                    bad.append(r["task"])
        return bad

    if not a.skip_rescore:
        progress("1-rescore")
        run([sys.executable, HERE / "rescore_pin.py", "--workers", a.workers],
            "1-rescore", timeout=7200)
    else:
        csv_sha_path = ROOT / "data" / "PIN_CSV_SHA.txt"
        csv_sha = csv_sha_path.read_text().strip() if csv_sha_path.exists() else old_sha
        if csv_sha != new_sha:
            sys.exit(f"--skip-rescore refused: pin CSV provenance is {csv_sha}, target is "
                     f"{new_sha}; run the rescore")
    bad = rescore_bad_rows()
    if bad:  # scoring subprocesses can time out under load / slow members can exceed the
        # 240s default — one retry, fewer workers, tripled per-member timeout
        log(f"[1-rescore] {len(bad)} member(s) failed to score ({', '.join(bad[:8])}…) — "
            f"retrying with fewer workers and RESCORE_TIMEOUT=720")
        run([sys.executable, HERE / "rescore_pin.py", "--workers", max(4, a.workers // 3)],
            "1-rescore-retry", timeout=14400, env={"RESCORE_TIMEOUT": "720"})
        bad = rescore_bad_rows()
        if bad:
            log(f"[1-rescore] STILL {len(bad)} unscored member(s): {', '.join(bad)} — repin "
                f"aborted (a pin CSV with SCORE_ERROR rows corrupts every downstream bar)")
            sys.exit(3)
    if not a.resume_after_basis:
        progress("2-basis")
        run([sys.executable, HERE / "refresh_basis.py"], "2-basis")
        progress("3-difficulty")
        run([sys.executable, HERE / "rank_difficulty.py"], "3-difficulty")
        progress("4-queue")
        run([sys.executable, HERE / "build_queue.py", "--reprioritize"], "4-queue")
    else:
        if (ROOT / "data" / "PIN_SHA.txt").read_text().strip() != new_sha:
            sys.exit("--resume-after-basis refused: basis pin does not match target")
        log("[2-4] reusing completed basis/difficulty/queue from interrupted refresh")

    try:
        basis_sha256, basis_epoch = basis_snapshot()
    except OSError as exc:
        sys.exit(f"basis snapshot failed: {exc}")
    generation["basis_sha256"] = basis_sha256
    generation["basis_epoch"] = basis_epoch
    atomic_json(INCOMPLETE, generation)

    packs_ok = packs_fail = 0
    phantoms = "skipped"
    if not a.fast:
        # 5. packs — child exceptions must not crash the chain; failures retry once
        # serially, and persistent failures ABORT before the audit/harden steps (a
        # "full refresh" that leaves stale packs is worse than a loud stop).
        progress("5-packs")
        pack_tasks = list(tasks)
        forced = {int(x) for x in a.force_pack.split(",") if x.strip()}
        if a.resume_packs_since is not None:
            expected_pin = sha8(live)
            reuse_epoch = max(a.resume_packs_since, basis_epoch)
            def reusable(t):
                if t in forced:
                    return False
                attack = ROOT / "packs" / f"task{t:03d}" / "ATTACK.md"
                try:
                    return (attack.stat().st_mtime >= reuse_epoch
                            and f"LIVE pin {expected_pin}" in attack.read_text(errors="replace"))
                except OSError:
                    return False
            pack_tasks = [t for t in tasks if not reusable(t)]
        log(f"[5-packs] regenerating {len(pack_tasks)}/{len(tasks)} packs with "
            f"{a.pack_procs} procs")
        failed = []
        with ProcessPoolExecutor(max_workers=a.pack_procs) as ex:
            futs = {ex.submit(one_pack, t): t for t in pack_tasks}
            for f in as_completed(futs):
                try:
                    t, rc = f.result()
                except Exception as exc:
                    t, rc = futs[f], -1
                    log(f"[5-packs] task{t:03d} EXCEPTION: {exc}")
                if rc == 0:
                    packs_ok += 1
                else:
                    failed.append(t)
        for t in failed[:]:
            try:
                t2, rc = one_pack(t)
                if rc == 0:
                    failed.remove(t)
                    packs_ok += 1
                    continue
            except Exception as exc:
                log(f"[5-packs] task{t:03d} retry EXCEPTION: {exc}")
            log(f"[5-packs] task{t:03d} FAILED after retry")
        packs_fail = len(failed)
        log(f"[5-packs] done: {packs_ok} ok, {packs_fail} failed")
        if packs_fail:
            log(f"[5-packs] ABORT: {packs_fail} pack(s) failed after retry: "
                f"{[f'task{t:03d}' for t in failed]} — fix make_pack for these, then re-run "
                f"repin.py (idempotent). Pin/CSV/queue are already refreshed.")
            sys.exit(4)

        expected_pin = sha8(live)
        stale = []
        for t in tasks:
            attack = ROOT / "packs" / f"task{t:03d}" / "ATTACK.md"
            try:
                if (attack.stat().st_mtime < basis_epoch
                        or f"LIVE pin {expected_pin}" not in
                        attack.read_text(errors="replace")):
                    stale.append(t)
            except OSError:
                stale.append(t)
        if stale:
            log(f"[5-packs] ABORT: {len(stale)} packs lack target pin {expected_pin}: "
                f"{[f'task{t:03d}' for t in stale[:20]]}")
            sys.exit(5)

        # 6. deterministic tickets. These are advisory, but stale pin provenance can
        # anchor builders to absorbed savings, so a full repin refreshes them atomically.
        progress("6-tickets")
        run([sys.executable, HERE / "factor_sweep.py"], "6-tickets", timeout=3600)

        # 7. phantom audit
        progress("7-reprice")
        out = run([sys.executable, HERE / "reprice_artifacts.py"], "7-reprice", timeout=7200)
        phantoms = sum(1 for ln in out.splitlines() if "PHANTOM" in ln)

        # 8. re-harden
        progress("8-harden")
        out = run([sys.executable, HERE / "pad_biases.py", "--base", live], "8-harden")
        for ln in out.splitlines():
            ln = ln.strip()
            if ln.startswith("python3 ") and "fast_verify.py" in ln:
                # replace the printed `python3` with THIS interpreter (fast_verify.py has no
                # exec bit — execing it directly would PermissionError before the gate runs)
                run([sys.executable, *ln.split()[1:]], "8-harden-gate", timeout=3600)
        # provenance manifest: which live pin this hardened base was built from —
        # auto_submit refuses a hardened base whose source sha != the current pin
        hb = ROOT / "results" / "hardened" / "hardened_base.zip"
        if hb.exists():
            (ROOT / "results" / "hardened" / "hardened_base.src_sha").write_text(sha8(live))

    new_sum = csv_sum(pin_csv)
    ledger_after = ledger_lines()
    mixed = ledger_after - ledger_before
    log("=== REPIN summary ===")
    log(f"pin:      {old_sha}@{old_pin.get('score')} -> {sha8(live)}@{a.score}")
    log(f"CSV sum:  {old_sum} -> {new_sum}  (LB {a.score}, drift {round(a.score - new_sum, 3) if a.score else 'n/a'})")
    log(f"packs:    {packs_ok} regenerated, {packs_fail} failed" if not a.fast else "packs:    skipped (--fast)")
    log(f"phantoms: {phantoms} flagged by reprice_artifacts (see results/REPRICE_AUDIT*.csv)")
    if mixed:
        log(f"NOTE: {mixed} LEDGER rows appended during the refresh window — their raw deltas are "
            f"basis-mixed; bank.py reprices, nothing to do")
    progress("9-manifest")
    try:
        final_basis_sha256, final_basis_epoch = basis_snapshot()
    except OSError as exc:
        sys.exit(f"final basis snapshot failed: {exc}")
    if final_basis_sha256 != basis_sha256:
        log("[9-manifest] ABORT: a pack basis file changed during refresh; packs are no "
            "longer provably current")
        sys.exit(6)
    with pin_csv.open() as fh:
        pin_rows = list(csv.DictReader(fh))
    live_sha = sha8(live)
    manifest = {
        **generation,
        "status": "partial" if a.fast else "complete",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "step": "complete", "live_pin": live_sha, "pin_csv_rows": len(pin_rows),
        "pin_csv_bad_rows": sum(1 for r in pin_rows
                                if r.get("status") != "ok" or not r.get("score", "").strip()),
        "queue_tasks": len(tasks), "packs_regenerated": packs_ok,
        "packs_verified": len(tasks),
        "basis_sha256": final_basis_sha256,
        "basis_epoch": final_basis_epoch,
        "tickets_summary": str(ROOT / "data" / "tickets" / "SUMMARY.md"),
        "hardened_source_pin": ((ROOT / "results" / "hardened" /
                                 "hardened_base.src_sha").read_text().strip()
                                if (ROOT / "results" / "hardened" /
                                    "hardened_base.src_sha").exists() else None),
    }
    csv_sha_path = ROOT / "data" / "PIN_CSV_SHA.txt"
    if not csv_sha_path.exists() or csv_sha_path.read_text().strip() != live_sha:
        csv_sha_tmp = csv_sha_path.with_name(csv_sha_path.name + f".tmp.{os.getpid()}")
        csv_sha_tmp.write_text(live_sha + "\n")
        os.replace(csv_sha_tmp, csv_sha_path)
    atomic_json(MANIFEST, manifest)
    INCOMPLETE.unlink(missing_ok=True)
    log(f"=== REPIN done nonce={generation['run_nonce']} ===")


if __name__ == "__main__":
    main()
