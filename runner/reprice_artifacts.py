#!/usr/bin/env python3
"""reprice_artifacts — post-scorer-fix decontamination audit (2026-07-10).

Re-prices EVERY results/taskNNN/*.onnx under the FIXED scorer (scoring_v2 -> the
grader-fidelity-override neurogolf_utils.py) and joins against the freshly-rescored
per-task pin CSV to compute the TRUE delta. Flags PHANTOM artifacts: the LEDGER
claimed a win but the true delta is below the lane bar (the member was priced on a
memory-skip hack and is worthless / worse-than-pin on Kaggle).

Output CSV: results/REPRICE_AUDIT_2026-07-10.csv
  task,file,kind,true_cost,true_pts,pin_pts,true_delta,ledger_delta,verdict
Read-only w.r.t. every artifact; writes only the audit CSV.
"""
import csv, glob, math, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
from config import CLEAN
sys.path.insert(0, CLEAN)


def price_one(path):
    from pathlib import Path
    from neurogolf.scoring_v2 import points_from_cost_v2, score_v2
    import contextlib, io
    t = int(os.path.basename(path)[4:7])
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            cost, _ = score_v2(Path(path), Path(CLEAN) / "extracted" / f"task{t:03d}.json",
                               task_num=t, timeout_sec=240)
        if not cost.ok:
            return (path, t, None, None, (cost.profile_error or cost.onnx_check_error or "err")[:50])
        pts = points_from_cost_v2(cost.memory_bytes, cost.params)
        return (path, t, cost.cost, round(pts, 4), "")
    except Exception as e:
        return (path, t, None, None, str(e)[:50])


def main():
    pin = {}
    with open(os.path.join(ROOT, "data", "pin_pv2_70_263_pertask.csv")) as f:
        for r in csv.DictReader(f):
            if r["score"]:
                pin[r["task"]] = float(r["score"])
    ledger_best = {}
    lane = {}
    with open(os.path.join(ROOT, "results", "LEDGER.csv")) as f:
        for r in csv.DictReader(f):
            if not r.get("task"):
                continue
            try:
                d = float(r["delta"])
            except (ValueError, KeyError):
                continue
            if r["task"] not in ledger_best or d > ledger_best[r["task"]]:
                ledger_best[r["task"]] = d
            lane[r["task"]] = r.get("lane", "")

    files = sorted(glob.glob(os.path.join(ROOT, "results", "task*", "task*.onnx")))
    print(f"repricing {len(files)} artifacts under the FIXED scorer…")
    rows = []
    with ProcessPoolExecutor(max_workers=int(os.environ.get("REPRICE_WORKERS", "6"))) as ex:
        futs = {ex.submit(price_one, p): p for p in files}
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 40 == 0:
                print(f"  {i}/{len(files)}")

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out = os.path.join(ROOT, "results", "REPRICE_AUDIT_2026-07-10.csv")
    phantom = below = ok = errs = 0
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "file", "kind", "true_cost", "true_pts", "pin_pts",
                    "true_delta", "ledger_delta", "verdict"])
        # keep the best true artifact per task for the summary
        best_true = {}
        for path, t, cost, pts, err in sorted(rows):
            tk = f"task{t:03d}"
            base = os.path.basename(path)
            kind = "canonical" if base == f"{tk}.onnx" else "archive"
            pin_pts = pin.get(tk)
            led = ledger_best.get(tk)
            bar = 0.05 if lane.get(tk) == "A" else 0.15
            if pts is None:
                verdict = f"UNSCORABLE:{err}"; errs += 1
                td = None
            else:
                td = round(pts - pin_pts, 4) if pin_pts is not None else None
                if td is None:
                    verdict = "no_pin"
                elif led is not None and led >= bar and td < bar:
                    verdict = "PHANTOM"; phantom += 1
                elif td < bar:
                    verdict = "below_bar"; below += 1
                else:
                    verdict = "REAL_WIN"; ok += 1
                if tk not in best_true or (td is not None and td > (best_true[tk][1] or -99)):
                    best_true[tk] = (pts, td, verdict, base)
            w.writerow([tk, base, kind, cost, pts, pin_pts, td, led, verdict])
    print(f"\nwrote {out}")
    print(f"artifacts: {len(rows)} | REAL_WIN {ok} · below_bar {below} · PHANTOM {phantom} · "
          f"UNSCORABLE {errs}")
    # per-task headline (best artifact)
    ph_tasks = sorted(tk for tk, (p, td, v, b) in best_true.items()
                      if v in ("PHANTOM", "below_bar"))
    real_tasks = sorted(tk for tk, (p, td, v, b) in best_true.items() if v == "REAL_WIN")
    print(f"\nper-task (best artifact): REAL_WIN {len(real_tasks)} · "
          f"phantom/below {len(ph_tasks)}")
    print("REAL survivors:", ", ".join(real_tasks[:60]))
    print("\nphantom/below tasks (ledger 'wins' that DIE under the true scorer):")
    print(", ".join(ph_tasks))


if __name__ == "__main__":
    main()
