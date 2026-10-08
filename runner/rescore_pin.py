#!/usr/bin/env python3
"""rescore_pin — regenerate data/pin_pv2_70_263_pertask.csv from the LIVE pin zip.

The missing first step of the refresh ritual (README: "rescore + refresh_basis.py +
build_queue.py --reprioritize"). Run after ANY repin, BEFORE refresh_basis.py.

- Scores every member of neurogolf_clean/submission.zip with the server-exact scorer
  (neurogolf.scoring_v2 — the same path fast_verify prices with).
- Backs up the current CSV as .bak_<old_pin_sha8> (matching the existing convention),
  writes the new CSV atomically, prints the per-task sum for sanity vs the LB score.
- Refuses to run if submission.zip's sha does not match logs/kaggle_pin.json (you'd be
  rescoring a zip that is not the recorded pin).

Usage: python3 runner/rescore_pin.py [--workers 6]
"""
import argparse, contextlib, csv, glob, hashlib, io, json, os, sys, tempfile, zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
from config import CLEAN
CSV_PATH = os.path.join(ROOT, "data", "pin_pv2_70_263_pertask.csv")
ZIP_PATH = os.path.join(CLEAN, "submission.zip")
sys.path.insert(0, CLEAN)


def score_one(args):
    """Runs in a worker process: extract member -> score_v2 -> row dict."""
    name, data = args
    task = int(name[4:7])  # taskNNN.onnx
    from pathlib import Path
    from neurogolf.scoring_v2 import score_v2  # noqa: import inside worker
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        f.write(data); p = f.name
    try:
        with contextlib.redirect_stdout(io.StringIO()):  # swallow the PAYLOAD debug line
            cost, _val = score_v2(Path(p), Path(CLEAN) / "extracted" / f"task{task:03d}.json",
                                  task_num=task,
                                  timeout_sec=int(os.environ.get("RESCORE_TIMEOUT", 240)))
    finally:
        os.unlink(p)
    if not cost.ok:
        err = cost.profile_error or cost.onnx_check_error or "score_error"
        return {"task": f"task{task:03d}", "params": "", "memory": "", "cost": "",
                "score": "", "status": f"SCORE_ERROR:{err[:60]}", "method": "ort"}
    import math
    pts = max(1.0, 25.0 - math.log(max(1, cost.cost)))
    return {"task": f"task{task:03d}", "params": cost.params, "memory": cost.memory_bytes,
            "cost": cost.cost, "score": f"{pts:.6f}", "status": "ok", "method": "ort"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    zsha = hashlib.sha256(open(ZIP_PATH, "rb").read()).hexdigest()[:8]
    pin = json.load(open(os.path.join(CLEAN, "logs", "kaggle_pin.json")))
    if zsha != pin.get("sha"):
        sys.exit(f"REFUSE: submission.zip sha {zsha} != recorded pin {pin.get('sha')} "
                 f"({pin.get('path')}). Point submission.zip at the recorded pin first.")
    old_sha = open(os.path.join(ROOT, "data", "PIN_SHA.txt")).read().strip()
    old_rows = {}
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, errors="replace") as f:
            old_rows = {r.get("task"): r for r in csv.DictReader(f) if r.get("task")}
    if old_sha == zsha:
        print(f"NOTE: PIN_SHA.txt already {zsha} — rescoring anyway (idempotent).")
    elif os.path.exists(CSV_PATH):
        bak = CSV_PATH + f".bak_{old_sha}"
        if not os.path.exists(bak):
            import shutil; shutil.copy(CSV_PATH, bak)
            print(f"backed up stale CSV -> {os.path.basename(bak)}")

    z = zipfile.ZipFile(ZIP_PATH)
    members = [(n, z.read(n)) for n in sorted(z.namelist()) if n.endswith(".onnx")]

    # Exact incremental refresh: score_v2 is per-task, so an unchanged member's old row is
    # mathematically identical. On a repin this avoids rescoring hundreds of byte-identical
    # members (p820 changed 95/400). Reuse requires the prior zip's full bytes and a healthy
    # prior row; any missing/mismatched evidence falls back to a fresh score.
    reused, to_score = [], members
    if old_sha and old_sha != zsha and old_rows:
        old_zip = None
        candidates = glob.glob(os.path.join(CLEAN, "logs", "pinned_bases", f"*{old_sha}*"))
        for candidate in candidates:
            try:
                if hashlib.sha256(open(candidate, "rb").read()).hexdigest()[:8] == old_sha:
                    old_zip = zipfile.ZipFile(candidate)
                    break
            except (OSError, zipfile.BadZipFile):
                continue
        if old_zip is not None:
            old_names = set(old_zip.namelist())
            fresh = []
            for name, data in members:
                task = f"task{int(name[4:7]):03d}"
                row = old_rows.get(task)
                if (name in old_names and old_zip.read(name) == data and row
                        and row.get("status") == "ok" and row.get("score")):
                    reused.append({k: row.get(k, "") for k in
                                   ("task", "params", "memory", "cost", "score",
                                    "status", "method")})
                else:
                    fresh.append((name, data))
            to_score = fresh
    print(f"rescoring {len(to_score)}/{len(members)} changed members of pin {zsha} "
          f"@ {pin.get('score')} with {a.workers} workers; reused {len(reused)} exact rows…")
    rows, errs = list(reused), []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(score_one, m): m[0] for m in to_score}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows.append(r)
            if r["status"] != "ok":
                errs.append((r["task"], r["status"]))
            if i % 25 == 0:
                print(f"  {i}/{len(to_score)} changed")
    rows.sort(key=lambda r: r["task"])
    tmp = CSV_PATH + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "params", "memory", "cost", "score",
                                          "status", "method"])
        w.writeheader(); w.writerows(rows)
    os.replace(tmp, CSV_PATH)
    csv_sha_path = os.path.join(ROOT, "data", "PIN_CSV_SHA.txt")
    csv_sha_tmp = csv_sha_path + f".tmp.{os.getpid()}"
    open(csv_sha_tmp, "w").write(zsha + "\n")
    os.replace(csv_sha_tmp, csv_sha_path)
    total = sum(float(r["score"]) for r in rows if r["score"])
    print(f"\nwrote {CSV_PATH}: {len(rows)} rows, Σscore={total:.2f} "
          f"(recorded LB {pin.get('score')}; local replica historically ~0.5 below)")
    if errs:
        print(f"{len(errs)} SCORE_ERROR rows (investigate before trusting bars on them):")
        for t, e in errs[:10]:
            print(f"  {t}: {e}")
    print("next: python3 runner/refresh_basis.py && python3 runner/build_queue.py --reprioritize")


if __name__ == "__main__":
    main()
