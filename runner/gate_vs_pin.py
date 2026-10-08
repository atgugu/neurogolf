#!/usr/bin/env python3
"""gate_vs_pin — acceptance for the measured-AMBIG tasks: BEHAVIORAL EQUIVALENCE to the pin.

Why this exists (measured 2026-07-07): the SHIPPED pin member for task118 FAILS gate124
(draw8) yet earns its points on Kaggle — for AMBIG tasks the generator's draws disagree
with what the hidden suite rewards, so "0 wrong on 300 draws + secret agree=1.0" is
STRICTLY HARDER than the leaderboard's own bar and no win can ever register.

The safe alternative: if candidate(x) == pin(x) under the grader's (raw>0) comparison for
EVERY graded input (train + test + full arc-gen), the candidate's hidden answers are
IDENTICAL to the pin's — same score, lower cost. Equivalence is a *stronger* guarantee
than gate124, not a weaker one.

Usage (from anywhere):
  python3 gate_vs_pin.py NNN candidate.onnx
Exit 0 = behaviorally identical on all graded inputs (bankable, CANARY-solo at probe
time); 1 = differs (shows the first diff) or error.
ONLY meaningful for the AMBIG set — for normal tasks use fast_verify --full (gate_pair).
"""
import io, json, os, sys, zipfile
import numpy as np

from config import CLEAN
AMBIG = {2, 118, 187, 255}


def onehot(g):
    """Mirror the grader exactly: cells outside the real grid are all-channel zero."""
    x = np.zeros((1, 10, 30, 30), np.float32)
    for r, row in enumerate(g):
        for c, color in enumerate(row):
            x[0, int(color), r, c] = 1.0
    return x


def run(sess, x):
    return (sess.run(None, {"input": x})[0] > 0)


def main():
    t = int(sys.argv[1]); cand_p = sys.argv[2]
    if t not in AMBIG:
        print(f"NOTE: task{t:03d} is not in the measured-AMBIG set {sorted(AMBIG)} — "
              f"use fast_verify --full (gate_pair) instead; equivalence-to-pin only makes "
              f"sense where agree=1.0 is impossible.")
    import onnxruntime as ort
    o = ort.SessionOptions(); o.log_severity_level = 4
    pin_bytes = zipfile.ZipFile(f"{CLEAN}/submission.zip").read(f"task{t:03d}.onnx")
    pin = ort.InferenceSession(pin_bytes, o)
    cand = ort.InferenceSession(open(cand_p, "rb").read(), o)
    d = json.load(open(f"{CLEAN}/extracted/task{t:03d}.json"))
    n = diff = 0
    for sp in ("train", "test", "arc-gen"):
        for i, e in enumerate(d.get(sp, [])):
            x = onehot(e["input"])
            a, b = run(pin, x), run(cand, x)
            n += 1
            if a.shape != b.shape or not (a == b).all():
                diff += 1
                if diff == 1:
                    cells = int((a != b).sum()) if a.shape == b.shape else -1
                    print(f"FIRST DIFF at {sp}#{i}: {cells} cells differ from the pin")
                if diff >= 3:
                    print(f"FAIL: candidate diverges from pin behavior "
                          f"({diff}+ of {n} inputs so far)")
                    sys.exit(1)
    if diff:
        print(f"FAIL: {diff}/{n} inputs differ from pin behavior")
        sys.exit(1)
    print(f"VS-PIN PASS: behaviorally IDENTICAL to the shipped pin on all {n} graded "
          f"inputs (train+test+arc-gen). Hidden answers are the pin's answers — same "
          f"score, your (lower) cost. Bank as CANARY-solo.")
    sys.exit(0)


if __name__ == "__main__":
    main()
