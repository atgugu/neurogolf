#!/usr/bin/env python3
"""refresh_basis — run after ANY pin change: new per-task CSV is in place; this updates
QUEUE.csv numeric columns (membership/priority STAY — attempt history continuity),
re-records PIN_SHA, recompiles strategy cards and donors. Then regenerate hot packs.
Usage: python3 refresh_basis.py"""
import csv, hashlib, json, math, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
from config import CLEAN
rows = {}
with open(os.path.join(ROOT, "data", "pin_pv2_70_263_pertask.csv")) as f:
    for r in csv.DictReader(f):
        if r["cost"]:
            rows[r["task"]] = (int(r["cost"]), float(r["score"]))
q = list(csv.DictReader(open(os.path.join(ROOT, "data", "lanes", "QUEUE.csv"))))
for r in q:
    if r["task"] in rows:
        c, s = rows[r["task"]]
        r["pin_cost"], r["pin_pts"] = str(c), f"{s:.3f}"
        r["bar_cost"] = str(int(c * math.exp(-(0.05 if r["lane"] == "A" else 0.15))))
qp = os.path.join(ROOT, "data", "lanes", "QUEUE.csv")
with open(qp + ".tmp", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=q[0].keys()); w.writeheader(); w.writerows(q)
os.replace(qp + ".tmp", qp)
open(os.path.join(ROOT, "data", "PIN_SHA.txt"), "w").write(
    hashlib.sha256(open(f"{CLEAN}/submission.zip", "rb").read()).hexdigest()[:8])
print("QUEUE numbers + PIN_SHA refreshed; recompiling strategies + donors…")
subprocess.run([sys.executable, os.path.join(HERE, "compile_strategies.py")], check=True)
subprocess.run([sys.executable, os.path.join(HERE, "find_donors.py")], check=True)
print("refresh_basis complete — regenerate hot packs next (make_pack per task)")
