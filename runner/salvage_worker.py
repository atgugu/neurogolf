#!/usr/bin/env python3
"""salvage_worker — standalone async salvage of a stranded passing artifact.

The orchestrator's finish() used to run the salvage gate INLINE: a blocking 600s
`fast_verify --price` plus up to 2400s of `submit_result --gated` froze the reap loop
(wave6 audit 2026-07-13: 54 synchronous gates banked 2, cost 54.6 min of frozen loop).
This worker takes that job off the critical path: the orchestrator copies the candidate
aside and fire-and-forgets this script; we single-flight via a non-blocking flock so
concurrent salvages simply drop (salvage is opportunistic, never load-bearing).

  python3 runner/salvage_worker.py <task NNN> <artifact_path> <lane> <att> <model>

Exit 0 in all handled cases. Writes ONLY results/SALVAGE.log (and deletes its artifact
copy on a below-bar drop) — never ATTEMPTS.jsonl or any orchestrator state; a banked
win still lands in LEDGER.csv/ACCEPTANCE.jsonl through submit_result itself.
"""
import datetime
import fcntl
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RESULTS = os.path.join(ROOT, "results")
LOCK = os.path.join(RESULTS, "salvage.lock")
LOG = os.path.join(RESULTS, "SALVAGE.log")

# Salvage economics: demand real headroom over the lane bar before spending a full gate.
MIN_HEADROOM = float(os.environ.get("SALVAGE_MIN_HEADROOM", "0.15"))
LANE_BARS = {"A": 0.05, "H": 0.05, "F": 0.0}
DELTA_RE = re.compile(r"Δ=?([+-]?\d+(?:\.\d+)?)")        # matches fast_verify's "Δ=+0.1234"
SUBMIT_DELTA_RE = re.compile(r"Δ([+-][\d.]+)")           # matches submit_result's "Δ+0.1234"


def log(msg):
    line = f"[{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}] {msg}\n"
    with open(LOG, "a") as f:  # single-flight (flock'd main) makes plain append safe
        f.write(line)


def main():
    if len(sys.argv) != 6:
        sys.stderr.write("usage: salvage_worker.py <task NNN> <artifact_path> <lane> "
                         "<att> <model>\n")
        return 2
    t = int(sys.argv[1])
    art = os.path.abspath(sys.argv[2])
    lane, att, model = sys.argv[3], sys.argv[4], sys.argv[5]

    os.makedirs(RESULTS, exist_ok=True)
    lf = open(LOCK, "w")
    try:
        fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log(f"salvage busy, dropping task{t:03d} (att{att} {model}, {art})")
        return 0

    if not os.path.exists(art):
        log(f"task{t:03d} att{att} {model}: artifact missing ({art}) — dropping")
        return 0

    bar = LANE_BARS.get(lane, 0.15)
    log(f"task{t:03d} att{att} {model}: pricing {art} (lane {lane} bar +{bar} "
        f"headroom +{MIN_HEADROOM})")
    try:
        v = subprocess.run([sys.executable, os.path.join(HERE, "fast_verify.py"),
                            str(t), art, "--price", "--bar", str(bar)],
                           capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        log(f"task{t:03d} att{att} {model}: fast_verify --price timed out (600s) — dropping")
        return 0
    m = DELTA_RE.search(v.stdout or "")
    if m is None:
        log(f"task{t:03d} att{att} {model}: no Δ in fast_verify output (rc={v.returncode}) "
            f"— dropping: {(v.stdout or '')[-200:].strip()!r}")
        return 0
    delta = float(m.group(1))
    if delta < bar + MIN_HEADROOM:
        log(f"task{t:03d} att{att} {model}: Δ{delta:+.4f} below bar+headroom "
            f"(+{bar + MIN_HEADROOM:.2f}), dropping — artifact copy deleted")
        try:
            os.remove(art)
        except OSError:
            pass
        return 0

    log(f"task{t:03d} att{att} {model}: Δ{delta:+.4f} ≥ bar+headroom — full gate "
        f"(may take minutes)")
    try:
        s = subprocess.run([sys.executable, os.path.join(HERE, "submit_result.py"),
                            str(t), art, "--gated", "--lane", lane,
                            "--notes", f"salvaged att{att} {model} (async)"],
                           capture_output=True, text=True, timeout=2400)
    except subprocess.TimeoutExpired:
        log(f"task{t:03d} att{att} {model}: submit_result timed out (2400s) — dropping")
        return 0
    if s.returncode == 0:
        dm = SUBMIT_DELTA_RE.search(s.stdout or "")
        log(f"task{t:03d} att{att} {model}: BANKED "
            f"Δ{dm.group(1) if dm else f'{delta:+.4f}(priced)'}")
    elif s.returncode == 4:
        log(f"task{t:03d} att{att} {model}: duplicate (byte-identical already banked) — no-op")
    else:
        tail = ((s.stdout or "") + (s.stderr or ""))[-400:].strip()
        log(f"task{t:03d} att{att} {model}: submit_result FAILED rc={s.returncode}: {tail!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
