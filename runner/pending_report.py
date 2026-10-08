#!/usr/bin/env python3
"""pending_report — results/PENDING_WINS.md: the operator's manual-submission shopping
list, ALWAYS repriced against the LATEST pin (accumulate-mode, operator ships from
another session; wave6 retro 2026-07-13).

Reads results/REPRICE_AUDIT_2026-07-10.csv (rewritten by reprice_artifacts.py; the
filename is historical). If that audit is STALE — older than LEDGER.csv or than
data/PIN_SHA.txt — it re-runs reprice_artifacts.py first (minutes; single-flight via
flock, ≥15 min between reruns so the orchestrate ledger-change hook can spawn this
fire-and-forget without stacking heavy reprices).

Sections: pin header · best-per-task REAL_WIN table (razor findings advisory, never
blocking — solos self-attribute and final=best makes a zero probe free) · canary queue
(results/CANARY_QUEUE.csv, manual probes only) · launch notes pointer.

  python3 runner/pending_report.py [--no-reprice]   # --no-reprice: render from the
                                                    # existing audit even if stale
"""
import csv
import datetime
import fcntl
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RESULTS = os.path.join(ROOT, "results")
AUDIT = os.path.join(RESULTS, "REPRICE_AUDIT_2026-07-10.csv")
LEDGER = os.path.join(RESULTS, "LEDGER.csv")
PIN_SHA = os.path.join(ROOT, "data", "PIN_SHA.txt")
from config import CLEAN
PIN_JSON = os.path.join(CLEAN, "logs/kaggle_pin.json")
CANARY = os.path.join(RESULTS, "CANARY_QUEUE.csv")
OUT = os.path.join(RESULTS, "PENDING_WINS.md")
LOCK = os.path.join(RESULTS, "pending_report.lock")
MIN_REPRICE_GAP = 900  # s


def _mtime(p):
    try:
        return os.path.getmtime(p)
    except OSError:
        return 0.0


def _pin_header():
    sha = ""
    try:
        sha = open(PIN_SHA, errors="replace").read().strip()
    except OSError:
        pass
    rec = {}
    try:
        rec = json.load(open(PIN_JSON))
    except (OSError, ValueError):
        pass
    consistent = bool(sha) and rec.get("sha") == sha
    return sha, rec, consistent


def _razor_notes(path):
    try:
        sys.path.insert(0, HERE)
        from scan_razors import scan
        findings = []
        scan(os.path.basename(path), open(path, "rb").read(), findings)
        return "; ".join(f"[{sev}] {kind}" for _, kind, _, sev in findings) or ""
    except Exception as exc:  # advisory only — a razor crash must not kill the report
        return f"razor-scan-error:{type(exc).__name__}"


def _maybe_reprice(no_reprice):
    audit_mt = _mtime(AUDIT)
    stale = audit_mt < _mtime(LEDGER) or audit_mt < _mtime(PIN_SHA)
    if not stale or no_reprice:
        return stale
    marker = os.path.join(RESULTS, ".pending_report_last_reprice")
    if os.path.exists(marker) and (os.path.getmtime(marker) + MIN_REPRICE_GAP
                                   > datetime.datetime.now().timestamp()):
        return True  # too soon; render from the stale audit, flagged in the header
    open(marker, "w").close()
    r = subprocess.run([sys.executable, os.path.join(HERE, "reprice_artifacts.py")],
                       capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        print(f"reprice_artifacts failed rc={r.returncode}: {r.stdout[-300:]}",
              file=sys.stderr)
        return True
    return False


def main():
    no_reprice = "--no-reprice" in sys.argv
    lf = open(LOCK, "w")
    try:
        fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return  # another render in flight
    stale = _maybe_reprice(no_reprice)
    sha, rec, consistent = _pin_header()

    best = {}
    counts = {"REAL_WIN": 0, "below_bar": 0, "PHANTOM": 0, "other": 0}
    try:
        for r in csv.DictReader(open(AUDIT, errors="replace")):
            v = r.get("verdict", "")
            counts["REAL_WIN" if v == "REAL_WIN" else
                   "below_bar" if v == "below_bar" else
                   "PHANTOM" if v == "PHANTOM" else "other"] += 1
            if v != "REAL_WIN":
                continue
            t = r["task"]
            td = float(r["true_delta"])
            if t not in best or td > float(best[t]["true_delta"]):
                best[t] = r
    except OSError:
        pass

    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "# PENDING WINS — repriced vs the LIVE pin",
        "",
        f"generated {now} · pin `{sha or '?'}` @ **{rec.get('score', '?')}** "
        f"(kaggle_pin.json ts {rec.get('ts', '?')})",
    ]
    if not consistent:
        lines.append("")
        lines.append("**⚠ PIN INCONSISTENT/ROTATING** — PIN_SHA.txt and kaggle_pin.json "
                     "disagree (repin in flight?). Numbers below may be stale; re-run "
                     "after the repin completes.")
    if stale:
        lines.append("")
        lines.append("**⚠ AUDIT STALE** — the reprice audit predates the ledger or the "
                     "pin; deltas are best-effort until reprice_artifacts.py reruns.")
    def _provenance(t, art):
        """(needs_verify, outcome, reason): match the artifact's sha8 to an
        ACCEPTANCE.jsonl receipt, then that receipt to its ATTEMPTS.jsonl outcome
        (attempt records are appended at finish(), just after acceptance — take the
        earliest attempt ending at/after accepted_epoch). DONE_UNVERIFIED, or no
        receipt matching the sha8, means the win has no verified provenance."""
        try:
            sha8 = hashlib.sha256(open(art, "rb").read()).hexdigest()[:8]
        except OSError:
            return True, "MISSING", "artifact unreadable"
        rec = None
        try:
            for line in open(os.path.join(RESULTS, t, "ACCEPTANCE.jsonl"),
                             errors="replace"):
                try:
                    rr = json.loads(line)
                except ValueError:
                    continue
                if rr.get("sha8") == sha8 and (
                        rec is None or float(rr.get("accepted_epoch", 0))
                        > float(rec.get("accepted_epoch", 0))):
                    rec = rr
        except OSError:
            pass
        if rec is None:
            return True, "NO-RECEIPT", f"no acceptance receipt matches sha8 {sha8}"
        acc = float(rec.get("accepted_epoch", 0))
        att = None
        try:
            for line in open(os.path.join(RESULTS, t, "ATTEMPTS.jsonl"),
                             errors="replace"):
                try:
                    rr = json.loads(line)
                except ValueError:
                    continue
                try:
                    ts = datetime.datetime.strptime(
                        str(rr.get("ts")), "%Y%m%dT%H%M%S").replace(
                        tzinfo=datetime.timezone.utc).timestamp()
                except ValueError:
                    continue
                if ts >= acc - 5 and (att is None or ts < att[0]):
                    att = (ts, rr)
        except OSError:
            pass
        if att is None:  # receipt exists but the attempt log is gone — trust the receipt
            return False, str(rec.get("validation") or "?"), ""
        outcome = str(att[1].get("outcome") or "?")
        reason = " ".join(str(att[1].get("fail_reason") or "").split())[:160]
        return outcome == "DONE_UNVERIFIED", outcome, reason

    shippable, needs = [], []
    for t, r in sorted(best.items(), key=lambda kv: -float(kv[1]["true_delta"])):
        art = os.path.join(RESULTS, t, r["file"])
        nv, outcome, reason = _provenance(t, art)
        (needs if nv else shippable).append((t, r, art, outcome, reason))
    tot = sum(float(r["true_delta"]) for _, r, _, _, _ in shippable)
    pending = f" (+{len(needs)} pending verify)" if needs else ""
    lines += [
        "",
        f"**{len(shippable)} tasks · best-per-task Σ = {tot:+.3f}**{pending} "
        f"(artifact verdicts: {counts['REAL_WIN']} REAL_WIN · "
        f"{counts['below_bar']} below_bar · {counts['PHANTOM']} phantom/absorbed)",
        "",
        "| task | Δ vs pin | cost | artifact | razor (advisory) |",
        "|---|---|---|---|---|",
    ]
    for t, r, art, _, _ in shippable:
        razor = _razor_notes(art) if os.path.exists(art) else "missing?"
        lines.append(f"| {t} | {float(r['true_delta']):+.4f} | {r['true_cost']} "
                     f"| `{art}` | {razor} |")
    lines += [
        "",
        "Razor findings are ADVISORY here: ship solos (self-attributing; final=best "
        "makes a zero free). HARD razor classes still block the multi-member bundle "
        "path in bank.py.",
    ]
    if needs:
        lines += [
            "",
            "## Needs verify before shipping",
            "",
            "Provenance check failed for these REAL_WIN rows: no ACCEPTANCE.jsonl "
            "receipt matches the artifact sha8, or the producing attempt ended "
            "DONE_UNVERIFIED. EXCLUDED from the Σ above — run the verify command, "
            "then re-render.",
            "",
            "| task | Δ vs pin | cost | artifact | razor (advisory) "
            "| outcome | fail_reason | verify |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for t, r, art, outcome, reason in needs:
            razor = _razor_notes(art) if os.path.exists(art) else "missing?"
            reason_md = (reason or "—").replace("|", "\\|").replace("`", "'")
            cmd = f"python3 runner/fast_verify.py {t[4:]} results/{t}/{r['file']} --full"
            lines.append(f"| {t} | {float(r['true_delta']):+.4f} | {r['true_cost']} "
                         f"| `{art}` | {razor} | {outcome} | {reason_md} | `{cmd}` |")
    lines += [
        "",
        "## Canary queue (manual probes only)",
        "",
    ]
    try:
        rows = [l for l in open(CANARY, errors="replace").read().splitlines()
                if l and not l.startswith(("ts,", "#"))]
        if rows:
            lines.append("| ts | task | Δ | cost | sha8 | artifact | gate fail |")
            lines.append("|---|---|---|---|---|---|---|")
            for l in rows:
                lines.append("| " + " | ".join(l.split(",", 6)) + " |")
            lines.append("")
            lines.append("Probe solo on the current pin via "
                         "`neurogolf_clean/candidates/probe_submit.py`; LB outcome feeds "
                         "`neurogolf_clean/candidates/kaggle_proven_ops.json`.")
        else:
            lines.append("(empty)")
    except OSError:
        lines.append("(empty)")
    lines += ["", "Launch notes for the next wave: see `results/LAUNCH_NOTES.md`.", ""]

    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines))
    os.replace(tmp, OUT)
    print(f"wrote {OUT}: {len(shippable)} tasks Σ{tot:+.3f}"
          f"{pending} vs pin {sha}")


if __name__ == "__main__":
    main()
