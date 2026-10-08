#!/usr/bin/env python3
"""fast_verify — the ONLY verdict a worker needs during iteration.

Usage:
  python3 fast_verify.py NNN candidate.onnx            # price + gate124 --draws 30
  python3 fast_verify.py NNN candidate.onnx --price    # price only (fast pre-check)
  python3 fast_verify.py NNN candidate.onnx --full     # gate_pair (gate124 + secret, 300 draws)

Exit codes: 0 = PASS at this level · 2 = COST-REJECT (don't bother gating) · 1 = gate FAIL.

Pricing uses the framework scorer (`neurogolf.scoring_v2`), which calls the extracted
Kaggle `score_network(model, trace_path)` over ORT profiling traces. Correctness is
delegated to the project's trusted gates (gate124 / gate_pair) which run real ARC-GEN
draws inside the grader venv. This script adds NO new correctness logic on purpose.
"""
import argparse, csv, contextlib, io, math, os, subprocess, sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
from config import CLEAN
PIN_CSV = os.path.join(HERE, "..", "data", "pin_pv2_70_263_pertask.csv")

# Prefer the vendored grader (runner/grader/) so pricing works without NEUROGOLF_CLEAN;
# CLEAN stays on the path for the ground-truth grids and the correctness gates.
sys.path.insert(0, CLEAN)
sys.path.insert(0, os.path.join(HERE, "grader"))


def real_demo(task):
    """Grader-faithful probe input: the task's first train grid, one-hot. The zeros probe
    misses data-dependent runtime tensors (EXPERIMENT 2026-07-07 20:45: zeros predicted
    7338.21, real-input predicted 7323.45, Kaggle said 7324.20). Fallback: zeros."""
    import numpy as np, json as _json
    try:
        d = _json.load(open(f"{CLEAN}/extracted/task{task:03d}.json"))
        exs = (d.get("train") or []) + (d.get("test") or [])
        g = np.array(max((e["input"] for e in exs),
                         key=lambda x: len(x) * len(x[0])), np.int8)
        a = np.zeros((30, 30), np.int8); a[:g.shape[0], :g.shape[1]] = g
        return (np.arange(10)[:, None, None] == a[None]).astype(np.float32)[None]
    except Exception:
        demo = np.zeros((1, 10, 30, 30), np.float32); demo[0, 0, 0, 0] = 1.0
        return demo


def check_pin_fresh():
    """Bars are computed from the pin CSV — if the live pin moved, every number is stale."""
    import hashlib
    rec = os.path.join(os.path.dirname(PIN_CSV), "PIN_SHA.txt")
    zp = os.path.join(CLEAN, "submission.zip")
    try:
        want = open(rec).read().strip()
        have = hashlib.sha256(open(zp, "rb").read()).hexdigest()[:8]
        if want != have:
            print(f"WARNING: PIN MOVED ({want} -> {have}) — the CSV/bars are STALE. "
                  f"Operator: rescore the new pin and refresh data/ (see README).")
    except OSError:
        pass


def pin_row(task: int):
    check_pin_fresh()
    with open(PIN_CSV) as f:
        for r in csv.DictReader(f):
            if r["task"] == f"task{task:03d}" and r["cost"]:
                return int(r["cost"]), float(r["score"])
    sys.exit(f"no pin row for task{task:03d}")


def price(path: str, task: int = None):
    from neurogolf.scoring_v2 import points_from_cost_v2, score_v2
    if task is None:
        sys.exit("task id required for framework-exact scoring_v2")
    task_json = Path(CLEAN) / "extracted" / f"task{task:03d}.json"
    # scoring_v2 currently prints a debug PAYLOAD line; keep fast_verify parse-stable.
    with contextlib.redirect_stdout(io.StringIO()):
        cost, val = score_v2(Path(path), task_json, task_num=task, timeout_sec=240)
    if not cost.ok:
        err = cost.profile_error or cost.onnx_check_error or (
            "disallowed_ops=" + ",".join(cost.disallowed_ops) if cost.disallowed_ops else "score_error")
        return {"cost": None, "params": cost.params, "memory": cost.memory_bytes,
                "points": 0.0, "status": "SCORE_ERROR:" + err, "method": "score_v2"}
    return {"cost": cost.cost, "params": cost.params, "memory": cost.memory_bytes,
            "points": points_from_cost_v2(cost.memory_bytes, cost.params),
            "status": "ok", "method": "score_v2"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", type=int)
    ap.add_argument("model")
    ap.add_argument("--bar", type=float, default=0.15)
    ap.add_argument("--draws", type=int, default=30)
    ap.add_argument("--price", action="store_true", help="price only, skip gate")
    ap.add_argument("--full", action="store_true", help="run gate_pair (300 draws + secret)")
    a = ap.parse_args()
    model = os.path.abspath(a.model)

    r = price(model, a.task)
    if r["cost"] is None:
        print(f"UNSCORABLE: {r['status']}  <- fix this first (banned op / dynamic shape / "
              f"multi-IO / bad names). See data/RULES.md §2.")
        sys.exit(1)
    pin_cost, pin_pts = pin_row(a.task)
    pts = r["points"]; delta = pts - pin_pts
    bar_cost = int(pin_cost * math.exp(-a.bar))
    print(f"PRICE task{a.task:03d}: cost={r['cost']} (params={r['params']} mem={r['memory']}) "
          f"pts={pts:.4f}  pin={pin_pts:.4f}(cost {pin_cost})  Δ={delta:+.4f}  "
          f"bar=+{a.bar} (need cost ≤ {bar_cost})  [{r['status']}/{r['method']}]")
    if delta < a.bar:
        print(f"COST-REJECT: Δ{delta:+.3f} < +{a.bar}. Do not gate. Delete a charged tensor "
              f"family (see your ATTACK.md targets), don't micro-trim.")
        print(f"Budget won't close in your current family? Get 15+ REPRICED alternatives "
              f"across families:\n  python3 {os.path.join(HERE, 'price_search.py')} "
              f"{a.task} \"<your one-sentence rule>\"")
        sys.exit(2)
    # local frontier: our banked-but-unsubmitted artifact may already beat the pin
    can = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results",
                       f"task{a.task:03d}", f"task{a.task:03d}.onnx")
    if os.path.exists(can) and os.path.getsize(can) and os.path.abspath(can) != model:
        rl = price(os.path.abspath(can), a.task)
        if rl.get("points") and rl["points"] > pts:
            print(f"NOTE: LOCAL BEST already at cost={rl['cost']} pts={rl['points']:.4f} "
                  f"(Δ{rl['points']-pin_pts:+.3f} vs pin, unbanked) — beat THAT, not the pin, "
                  f"or your build changes nothing.")
    if a.price:
        sys.exit(0)

    if a.full:
        cmd = ["python3", "candidates/gate_pair.py", str(a.task), model, "--draws", "300"]
    else:
        cmd = ["python3", "candidates/gate124.py", str(a.task), model, "--draws", str(a.draws)]
    p = subprocess.run(cmd, cwd=CLEAN, capture_output=True, text=True, timeout=1800)
    out = (p.stdout + p.stderr).strip()
    print(out[-4000:])
    if p.returncode == 0 and a.full:
        # integrity marker: submit_result --gated only trusts a marker matching the file sha
        import hashlib
        sha8 = hashlib.sha256(open(model, "rb").read()).hexdigest()[:8]
        open(os.path.join(os.path.dirname(model), f".fullgate_{sha8}"), "w").write(model)
    if p.returncode != 0 and a.task in (2, 118, 187, 255):
        print(f"NOTE task{a.task:03d} is in the measured-AMBIG set — the SHIPPED PIN ITSELF "
              f"fails gate124 here (verified: pin118 fails draw8 yet earns on Kaggle), so "
              f"agree=1.0 is the WRONG target. The winning route: make your cheaper build "
              f"BEHAVIORALLY IDENTICAL to the pin, then\n"
              f"  python3 {os.path.join(HERE, 'gate_vs_pin.py')} {a.task} <model>\n"
              f"  python3 {os.path.join(HERE, 'submit_result.py')} {a.task} <model> "
              f"--vs-pin --lane <L> --notes \"...\"")
    if p.returncode != 0 and "wrong" in out.lower():
        # show the actual failing grids — one look beats three blind retries
        d = subprocess.run(["python3", "candidates/diff_one.py", str(a.task), model, "--max", "2"],
                           cwd=CLEAN, capture_output=True, text=True, timeout=300)
        print("--- failing examples (diff_one) ---")
        print(d.stdout[-3500:])
    if p.returncode == 0:
        lvl = "FULL (gate124+secret) — REGISTERABLE" if a.full else f"FAST ({a.draws} draws)"
        print(f"VERDICT: PASS {lvl}  Δ={delta:+.4f}")
        if not a.full:
            print("Next: rerun with --full before recording in LEDGER.csv.")
    sys.exit(p.returncode)


if __name__ == "__main__":
    main()
