#!/usr/bin/env python3
"""submit_result — atomic completion. Replaces the 5-step manual ritual with ONE command.

Usage (from your pack dir, after fast_verify PASS):
  python3 ../../runner/submit_result.py NNN taskNNN.onnx --lane B --notes "einsum row-match"

Does, in order (stops at the first failure):
  1. full gate (gate_pair: gate124 300 draws + secret) unless --gated (you already ran --full)
  2. reprice with the replica; enforce the lane bar
  3. copy .onnx + build.py to results/taskNNN/
  4. flock-append LEDGER.csv (correct format, sha8 computed for you)
  5. append your lesson bullet to LESSONS.md
  6. print the exact `DONE taskNNN +Δ` line to end your message with
Exit 0 = banked-ready. Any other exit: fix and retry — do NOT hand-edit the ledger.
"""
from config import CLEAN
import argparse, csv, datetime, fcntl, hashlib, json, math, os, re, shutil, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
LEDGER = os.path.join(ROOT, "results", "LEDGER.csv")
LESSONS = os.path.join(ROOT, "results", "LESSONS.md")
AMBIG_TASKS = {2, 118, 187, 255}


def locked_append(path, line):
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(line)
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


def atomic_copy(src, dst, expected_sha256=None):
    """Publish an artifact atomically so a concurrent bank never reads a partial ONNX."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(dst) + ".", suffix=".tmp",
                               dir=os.path.dirname(dst))
    os.close(fd)
    try:
        shutil.copy2(src, tmp)
        if expected_sha256:
            copied_sha = hashlib.sha256(open(tmp, "rb").read()).hexdigest()
            if copied_sha != expected_sha256:
                raise SystemExit("REJECT: candidate bytes changed after validation — "
                                 "nothing recorded; retry from a stable artifact.")
        os.replace(tmp, dst)
    finally:
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass


def register_ticket(task, model_path, note=""):
    """Engine A intake (study7300 M3): a verified-correct graph that missed its bar /
    class budget becomes a factorization-ticket DONOR instead of a NO — the old stop
    condition threw these away and selected for +0.17 wins. Appends a row in the
    factor_sweep TICKETS.csv schema and archives the donor bytes."""
    tdir = os.path.join(ROOT, "data", "tickets")
    os.makedirs(os.path.join(tdir, "donors"), exist_ok=True)
    sha8 = hashlib.sha256(open(model_path, "rb").read()).hexdigest()[:8]
    donor = os.path.join(tdir, "donors", f"task{task:03d}_{sha8}.onnx")
    atomic_copy(model_path, donor)
    csvp = os.path.join(tdir, "TICKETS.csv")
    if not os.path.exists(csvp):
        locked_append(csvp, "task,flag,recipe,est_els_saved,est_pts,value_exact,"
                            "evidence,exemplars\n")
    ev = ("MISSED_TARGET donor; " + note).replace(",", ";")[:140]
    locked_append(csvp, f"task{task:03d},MISSED_TARGET,factor the donor graph into the "
                        f"terminal contraction,,,N,{ev},{donor}\n")


def verify_acceptance(task, model_path, *, bar, vs_pin=False, gated=False):
    """Run exactly one valid acceptance route for this candidate.

    AMBIG tasks deliberately use behavioral equivalence to the shipped pin instead of
    gate_pair.  Keeping this as one explicit state machine prevents a successful vs-pin
    proof from falling through to the incompatible full gate.
    """
    if vs_pin:
        if task not in AMBIG_TASKS:
            raise SystemExit("REJECT: --vs-pin is only valid for the measured-AMBIG set "
                             "002/118/187/255.")
        g = subprocess.run([sys.executable, os.path.join(HERE, "gate_vs_pin.py"),
                            str(task), model_path], text=True)
        if g.returncode != 0:
            raise SystemExit("REJECT: not behaviorally identical to the pin — nothing recorded.")
        return "VSPIN-CANARY"

    sha8_now = hashlib.sha256(open(model_path, "rb").read()).hexdigest()[:8]
    marker = os.path.join(os.path.dirname(model_path), f".fullgate_{sha8_now}")
    if gated and os.path.exists(marker):
        return "FULL-GATE"
    if gated:
        print("NOTE: --gated given but no full-gate marker for this exact file — gating now.")
    g = subprocess.run([sys.executable, os.path.join(HERE, "fast_verify.py"),
                        str(task), model_path, "--full", "--bar", str(bar)],
                       text=True, capture_output=True)
    print(g.stdout, end="")
    if g.stderr:
        print(g.stderr, end="", file=sys.stderr)
    if g.returncode == 2:
        # fast_verify prices BEFORE gating, so rc=2 means "above bar-cost, correctness
        # unproven". Study7300 M3: correct-but-expensive is Engine-A food, not a NO —
        # gate at a permissive bar; only a build that truly gates earns a ticket.
        g2 = subprocess.run([sys.executable, os.path.join(HERE, "fast_verify.py"),
                             str(task), model_path, "--full", "--bar", "-9"],
                            text=True, capture_output=True)
        print(g2.stdout, end="")
        if g2.stderr:
            print(g2.stderr, end="", file=sys.stderr)
        if g2.returncode == 0:
            register_ticket(task, model_path,
                            note=f"gate PASS at permissive bar; missed lane bar +{bar}")
            print(f"TICKET task{task:03d}: verified CORRECT but above the lane bar — "
                  f"registered as a factorization-ticket donor (tickets/). Not banked; "
                  f"NOT a NO. End your message with: TICKET task{task:03d}")
            raise SystemExit(3)
        _maybe_canary(task, model_path, sha8_now, g2.stdout + g.stdout)
        raise SystemExit("REJECT: above bar AND does not gate — nothing recorded.")
    if g.returncode != 0:
        _maybe_canary(task, model_path, sha8_now, g.stdout)
        raise SystemExit("REJECT: full gate did not pass — nothing recorded.")
    return "FULL-GATE"


def _maybe_canary(task, model_path, sha8_now, gate_out):
    """Queue a positive-delta candidate whose ONLY gate failure is a fresh-draw
    disagreement (`wrong(drawN)`) for a MANUAL Kaggle solo canary probe.

    Canary-probed evidence (GATE_FALSE_REJECTS_2026-07-13.md): gate124's synthetic
    fresh draws can fall outside the real task distribution — 2 of 3 probed
    `wrong(drawN)` rejects were VALID on the LB (+0.154 recovered), while task023
    proves the same symptom can also be a true silent-zero. The LB is the only
    arbiter, so these are never auto-banked and never auto-submitted — they are
    listed in results/CANARY_QUEUE.csv (surfaced by PENDING_WINS.md) for the
    operator to probe solo under final=best, which makes a wrong probe free."""
    if "lint: OK" not in gate_out or "wrong(draw" not in gate_out:
        return
    m = re.search(r"Δ=([+-]?\d+(?:\.\d+)?)", gate_out)
    mc = re.search(r"cost=(\d+)", gate_out)
    if not m or float(m.group(1)) <= 0:
        return
    md = re.search(r"wrong\(draw\d+[^)]*\)", gate_out)
    path = os.path.join(ROOT, "results", "CANARY_QUEUE.csv")
    line = (f"{time.strftime('%Y%m%dT%H%M%S')},task{task:03d},{float(m.group(1)):+.4f},"
            f"{mc.group(1) if mc else '?'},{sha8_now},{os.path.abspath(model_path)},"
            f"{md.group(0) if md else 'wrong(draw?)'}\n")
    if not os.path.exists(path):
        line = ("ts,task,delta,cost,sha8,artifact,fail_reason\n"
                "# MANUAL canary probes only (operator decision): build a solo zip on the "
                "current pin and probe via neurogolf_clean/candidates/probe_submit.py; "
                "final=best makes a zero free. Never auto-submit; never bank without LB "
                "confirmation.\n") + line
    locked_append(path, line)
    print(f"CANARY-QUEUED task{task:03d}: gate FAIL is wrong(drawN)-only with positive "
          f"Δ{float(m.group(1)):+.4f} — added to results/CANARY_QUEUE.csv for a manual "
          f"solo probe (NOT banked, NOT submitted).")


def current_pin_basis():
    """Fail closed while repin has split the live zip from the derived scoring basis."""
    rec = os.path.join(ROOT, "data", "PIN_SHA.txt")
    live = os.path.join(CLEAN, "submission.zip")
    try:
        recorded = open(rec, errors="replace").read().strip()
        actual = hashlib.sha256(open(live, "rb").read()).hexdigest()[:8]
    except OSError as exc:
        raise SystemExit(f"REJECT: pin basis unreadable ({exc}) — nothing recorded.")
    if not recorded or recorded != actual:
        raise SystemExit(f"REJECT: pin refresh in progress/inconsistent ({recorded} != {actual}) — "
                         "nothing recorded; retry after repin completes.")
    return recorded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", type=int)
    ap.add_argument("model_path")
    ap.add_argument("--lane", required=True)
    ap.add_argument("--notes", default="")
    ap.add_argument("--lesson", default="", help="one-line lesson for the fleet")
    ap.add_argument("--gated", action="store_true", help="skip gate (you already passed --full)")
    ap.add_argument("--vs-pin", action="store_true",
                    help="AMBIG tasks (002/118/187/255) ONLY: accept on behavioral "
                         "equivalence to the pin (gate_vs_pin) instead of gate_pair — "
                         "identical hidden answers at lower cost; banked as CANARY-solo")
    a = ap.parse_args()
    t, mp = a.task, os.path.abspath(a.model_path)
    # F = mechanical value-exact scripts (0.0); every LLM lane registers at +0.15 —
    # the floor is structural-replacement-sized on purpose (no shaving lane).
    bar = {"F": "0.0"}.get(a.lane, "0.15")
    basis_pin_sha = current_pin_basis()
    try:
        source_sha256 = hashlib.sha256(open(mp, "rb").read()).hexdigest()
    except OSError as exc:
        raise SystemExit(f"REJECT: candidate unreadable ({exc}) — nothing recorded.")

    # Byte-identical dedup BEFORE the expensive gate (wave6 audit 2026-07-13: six members
    # — 069/136/287/317/391/354 — were re-gated and re-banked with the exact same sha8,
    # each burning a full attempt + gate for zero new points). Same bytes ⇒ same score;
    # a duplicate is a no-op, not a win and not a NO.
    try:
        with open(LEDGER, errors="replace") as _lf:
            for _row in csv.DictReader(_lf):
                if (_row.get("task") == f"task{t:03d}"
                        and _row.get("sha8") == source_sha256[:8]):
                    print(f"DUPLICATE task{t:03d}: candidate sha8 {source_sha256[:8]} is "
                          f"already banked (ledger Δ{_row.get('delta')}) — byte-identical, "
                          f"nothing to gain; skipping gate and ledger. "
                          f"End your message with: DUPLICATE task{t:03d}")
                    raise SystemExit(4)
    except OSError:
        pass

    # ACCEPT-time razor scan (corpus §0.4 / audit T5): catch proven-UB members at the ledger
    # door, not only at ship time in bank.py. HARD classes (initializer short-bias, constant
    # OOB index) have zero false positives → REJECT; WARN classes are advisory → tagged into
    # notes so bank-review sees them. Static + instant, runs before the expensive gate.
    sys.path.insert(0, HERE)
    from scan_razors import scan as _razor_scan
    _findings = []
    _razor_scan(os.path.basename(mp), open(mp, "rb").read(), _findings)
    _hard = [f for f in _findings if f[3] == "HARD"]
    _warn = [f for f in _findings if f[3] == "WARN"]
    if _hard:
        for n, kind, det, sev in _hard:
            print(f"  [HARD] {kind}  {det}")
        sys.exit("REJECT: proven-UB razor (see above) — this member is a Kaggle coin-flip; "
                 "nothing recorded. Fix the graph (pad the bias / clamp the index).")
    if _warn:
        for n, kind, det, sev in _warn:
            print(f"  [WARN] {kind}  {det}  (advisory — not blocking)")
        a.notes = (a.notes + " RAZOR-WARN:" + ";".join(sorted({f[1] for f in _warn}))).strip()

    acceptance = verify_acceptance(t, mp, bar=bar, vs_pin=a.vs_pin, gated=a.gated)
    if acceptance == "VSPIN-CANARY":
        a.notes = ("VSPIN-CANARY " + a.notes).strip()
    route = subprocess.run([sys.executable, os.path.join(HERE, "oracle_route.py"),
                            str(t), mp], capture_output=True, text=True)
    route_line = next((x for x in route.stdout.splitlines() if x.startswith("ROUTE ")), "")
    if "ROUTE SOLO_PROBE" in route_line:
        # Safety/provenance tags MUST lead: the ledger's bounded prose field truncates the
        # tail, and bank policy must never lose a solo-only tag behind a long method note.
        a.notes = ("ORACLE-SOLO " + route_line[:100] + " " + a.notes).strip()
    v = subprocess.run([sys.executable, os.path.join(HERE, "fast_verify.py"),
                        str(t), mp, "--price", "--bar", bar],
                       capture_output=True, text=True)
    m = re.search(r"cost=(\d+).*?pts=([\d.]+).*?Δ=([+-][\d.]+)", v.stdout)
    if v.returncode != 0 or not m:
        print(v.stdout)
        sys.exit("REJECT: repricing failed or below bar — nothing recorded.")
    cost, pts, delta = int(m.group(1)), float(m.group(2)), float(m.group(3))
    # memory-dominated pricing is RUN-SENSITIVE (study7300: task216 priced 4108 vs 7960
    # across runs). For any candidate with charged memory > 1 kB, reprice a second time
    # and accept only a reproducing cost — a non-reproducing price is UNKNOWN, not a win.
    mm = re.search(r"params=(\d+) mem=(\d+)", v.stdout)
    mem_b = int(mm.group(2)) if mm else 0
    if mem_b > 1024:
        v2 = subprocess.run([sys.executable, os.path.join(HERE, "fast_verify.py"),
                             str(t), mp, "--price", "--bar", bar],
                            capture_output=True, text=True)
        m2 = re.search(r"cost=(\d+)", v2.stdout)
        if v2.returncode != 0 or not m2 or int(m2.group(1)) != cost:
            print(v.stdout)
            print(v2.stdout)
            sys.exit(f"REJECT: memory-dominated price did not reproduce "
                     f"({cost} vs {m2.group(1) if m2 else 'FAIL'}) — status UNKNOWN, "
                     f"nothing recorded. Re-run when the box is quiet; never book a "
                     f"non-reproducing delta.")
        a.notes = (a.notes + " MEM-PAIRED-X2").strip()
    OVERSIZE = {21, 55, 80, 184, 202, 366}
    if t in OVERSIZE:
        # local scoring_v2 counts >30-dim grids wrong while Kaggle skips them (study7300):
        # the gates handle draws correctly, but pre-ship validation must use scoring_v4.
        a.notes = ("V4-VALIDATE " + a.notes).strip()
        print(f"NOTE task{t:03d} is on the oversize-grid list — tagged V4-VALIDATE; "
              f"bank must validate via neurogolf/scoring_v4.py before shipping.")

    outdir = os.path.join(ROOT, "results", f"task{t:03d}")
    os.makedirs(outdir, exist_ok=True)
    # never clobber a BETTER banked artifact: the canonical task{t}.onnx (what bank.py
    # ships) only advances; a worse-but-above-bar re-attempt is archived alongside
    # prev_best = REPRICE the existing canonical artifact NOW (ledger deltas go stale
    # after repins — audit 2026-07-07: a stale +0.156 row nearly outranked a real win)
    prev_best = None
    incumbent_unpriced = False
    can = os.path.join(outdir, f"task{t:03d}.onnx")
    if os.path.exists(can):
        try:
            v0 = subprocess.run([sys.executable, os.path.join(HERE, "fast_verify.py"),
                                 str(t), can, "--price", "--bar", "-9"],
                                capture_output=True, text=True, timeout=300)
            m0 = re.search(r"Δ=([+-][\d.]+)", v0.stdout)
            if m0:
                prev_best = float(m0.group(1))
            else:  # exists but reprice gave no Δ (crash / PIN MOVED / below -9 bar)
                incumbent_unpriced = True
        except subprocess.TimeoutExpired:
            incumbent_unpriced = True
    sha8_tmp = hashlib.sha256(open(mp, "rb").read()).hexdigest()[:8]
    # NEVER overwrite an incumbent we could not evaluate: prev_best=None previously conflated
    # "no incumbent" with "unpriceable incumbent" → a hiccup silently clobbered a better win
    # (audit 2026-07-08 / T3). Archive the candidate instead; bank.py re-globs+reprices __*.onnx.
    if incumbent_unpriced or (prev_best is not None and delta <= prev_best):
        atomic_copy(mp, os.path.join(outdir, f"task{t:03d}__{sha8_tmp}.onnx"),
                    source_sha256)
        why = ("incumbent exists but could NOT be repriced — kept; candidate archived"
               if incumbent_unpriced else
               f"banked best Δ{prev_best:+.4f} ≥ this Δ{delta:+.4f} — canonical kept; archived")
        print(f"NOTE: {why} as task{t:03d}__{sha8_tmp}.onnx")
    else:
        atomic_copy(mp, os.path.join(outdir, f"task{t:03d}.onnx"), source_sha256)
    bp = os.path.join(os.path.dirname(mp), "build.py")
    if os.path.exists(bp):
        atomic_copy(bp, os.path.join(outdir, "build.py"))
    sha8 = source_sha256[:8]
    notes = (a.notes or "").replace(",", ";")[:120]
    locked_append(LEDGER, f"task{t:03d},{cost},{pts:.4f},{delta:+.4f},{a.lane},{sha8},{notes}\n")
    receipt = {
        "ts": f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ}",
        "accepted_epoch": time.time(), "task": t,
        "sha256": source_sha256,
        "sha8": sha8, "basis_pin_sha": basis_pin_sha, "delta": delta,
        "cost": cost, "points": pts, "lane": a.lane, "validation": acceptance,
    }
    locked_append(os.path.join(outdir, "ACCEPTANCE.jsonl"), json.dumps(receipt) + "\n")
    if a.lesson:
        locked_append(LESSONS, f"- [{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d}][task{t:03d}] {a.lesson[:200]}\n")
    print(f"RECORDED task{t:03d}: cost={cost} pts={pts:.4f} Δ{delta:+.4f} sha8={sha8}")
    print(f"\nEnd your final message with exactly:\nDONE task{t:03d} {delta:+.2f}")


if __name__ == "__main__":
    main()
