#!/usr/bin/env python3
"""bank — the morning mile: turn LEDGER wins into ≤12-swap submission zips.

Reads results/LEDGER.csv, razor-scans every member, and splits it into artifacts. Every
SHA without exact per-member Kaggle payment proof is solo-probe-only; every member that
survives its lane's positive registration bar is eligible. Multi-member chunks contain
proven SHAs only. Assembles
base+swaps zips into results/bank/ and prints predicted scores and submission commands.

Usage:
  python3 bank.py                             # uses hardened base if present
  python3 bank.py --base <zip> --chunk 12
  python3 bank.py --dry-run                   # plan only

Canary rule: notes containing 'CANARY' or tasks in the canary set never batch.
"""
import argparse, csv, hashlib, os, re, subprocess, sys, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
LEDGER = os.path.join(ROOT, "results", "LEDGER.csv")
BANK = os.path.join(ROOT, "results", "bank")
HARDENED = os.path.join(ROOT, "results", "hardened", "hardened_base.zip")
REJECTED = os.path.join(ROOT, "results", "REJECTED_ARTIFACTS.jsonl")
CANARY_TASKS = {"233", "285", "101", "219", "392", "002", "118", "187", "255",
                "350"}  # risk classes; 350 = Kaggle-tanker history + novel tensor-network family (2026-07-12)
AMBIG_TASKS = {"002", "118", "187", "255"}
MIN_SOLO_DELTA = 0.0  # point-max policy: every locally positive unproven member gets attribution
BASE_SCORE = 7336.46  # fallback ONLY (overwritten from logs/kaggle_pin.json at runtime, :44-49);
                      # refreshed to the live pin. HARDENED_SCORE was dead code — removed (T14).

sys.path.insert(0, HERE)
from config import CLEAN
from proven_corpus import payment_proof_for_path
from oracle_route import route as hidden_oracle_route


def regate_command(member, bar):
    """Use the acceptance contract appropriate for this task class."""
    if str(member["t"]).zfill(3) in AMBIG_TASKS:
        return [sys.executable, os.path.join(HERE, "gate_vs_pin.py"),
                str(int(member["t"])), member["path"]]
    return [sys.executable, os.path.join(HERE, "fast_verify.py"),
            str(int(member["t"])), member["path"], "--full", "--bar", bar]


def classify_artifact_risk(member):
    """Attach exact payment proof and recompute solo routing from the artifact bytes.

    Ledger notes are bounded human prose and cannot be a safety authority.  Exact Kaggle
    proof overrides advisory oracle routing; static task canaries never get overridden.
    """
    member["payment_proof"] = payment_proof_for_path(int(member["t"]), member["path"])
    try:
        verdict, reasons = hidden_oracle_route(int(member["t"]), member["path"])
    except Exception as exc:
        verdict, reasons = "SOLO_PROBE", [f"route-error:{type(exc).__name__}"]
    member["oracle_route"] = verdict
    member["oracle_reasons"] = reasons
    # non-finite weights are the environment-sensitive coin-flip class (study7300 §0.3 +
    # LB 2026-07-12: same bytes scored 0/266 in one local env and FULL on the LB) — local
    # validity does not transfer either way, so such bytes never batch without exact
    # payment proof; they ship SOLO so a flip stays attributable. Decided from the
    # artifact BYTES (notes are not a safety authority). A razor-scan failure is UNKNOWN,
    # not clean — fail toward solo.
    try:
        from scan_razors import scan as _razor_scan
        _f = []
        with open(member["path"], "rb") as fh:
            _razor_scan(os.path.basename(member["path"]), fh.read(), _f)
        if any(k == "NONFINITE_WEIGHT" for _n, k, _d, _s in _f):
            member["oracle_reasons"] = list(reasons) + ["nonfinite-weights"]
            if not member["payment_proof"]:
                member["canary"] = True
    except Exception as exc:
        member["oracle_reasons"] = list(reasons) + [f"razor-scan-error:{type(exc).__name__}"]
        if not member["payment_proof"]:
            member["canary"] = True
    if verdict != "BATCH" and not member["payment_proof"]:
        member["canary"] = True
    return member


def bundle_id(base, members):
    """Content identity for an immutable submission bundle (base + exact swap bytes)."""
    h = hashlib.sha256()
    with open(base, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    for member in sorted(members, key=lambda m: str(m["t"])):
        h.update(f"task{member['t']}".encode())
        with open(member["path"], "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
    return h.hexdigest()[:12]


def write_bundle(base, out, members):
    """Atomically publish once; never overwrite bytes referenced by a submitted path."""
    if os.path.exists(out):
        return
    swap = {f"task{m['t']}.onnx": m["path"] for m in members}
    tmp = f"{out}.tmp.{os.getpid()}"
    try:
        with zipfile.ZipFile(base) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
            for name in zin.namelist():
                if name in swap:
                    z.write(swap[name], name)
                else:
                    z.writestr(name, zin.read(name))
        os.replace(tmp, out)
        os.chmod(out, 0o444)
    finally:
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass


def rejected_artifacts():
    """Return task-scoped rejections; identical generic bytes may behave differently by task."""
    out = set()
    try:
        for line in open(REJECTED, errors="replace"):
            try:
                rec = __import__("json").loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict) and rec.get("sha256") and rec.get("task") is not None:
                try:
                    task = f"{int(rec['task']):03d}"
                except (TypeError, ValueError):
                    continue
                out.add((task, str(rec["sha256"])))
    except OSError:
        pass
    return out


def validate_chunks(chunks, min_solo_delta=None):
    """Refuse mixed proof and, when requested, low-value single-member artifacts."""
    for chunk in chunks:
        if len(chunk) > 1 and any(not member.get("payment_proof") for member in chunk):
            tasks = ",".join(str(member.get("t", "?")) for member in chunk)
            raise ValueError(f"REFUSE mixed/unproven multi-member chunk: {tasks}")
        if (min_solo_delta is not None and len(chunk) == 1
                and not float(chunk[0].get("delta", float("-inf"))) > min_solo_delta):
            raise ValueError(f"REFUSE low-value solo task{chunk[0].get('t', '?')}: "
                             f"delta must be > {min_solo_delta}")


def _proven_groups_without_low_value_solos(members, chunk_size, min_solo_delta):
    """Partition proven members without a singleton tail where the chunk size permits."""
    if not members:
        return [], [], []
    if chunk_size == 1:
        groups, orphans = [], list(members)
    else:
        groups = [members[i:i + chunk_size]
                  for i in range(0, len(members), chunk_size)]
        orphans = []
        if len(groups[-1]) == 1:
            if len(groups) > 1 and len(groups[-2]) >= 3:
                groups[-1].insert(0, groups[-2].pop())
            else:
                orphans.append(groups.pop()[0])
    high = [m for m in orphans
            if float(m.get("delta", float("-inf"))) > min_solo_delta]
    held = [m for m in orphans if id(m) not in {id(x) for x in high}]
    return groups, high, held


def shipping_chunks(members, chunk_size, min_solo_delta=MIN_SOLO_DELTA):
    """Spend solos only above threshold; only exact proven non-canaries may group."""
    if chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    if min_solo_delta < MIN_SOLO_DELTA:
        raise ValueError(f"solo threshold cannot be lowered below {MIN_SOLO_DELTA}")
    risky = [m for m in members if m.get("canary") or not m.get("payment_proof")]
    solos = [m for m in risky if float(m.get("delta", float("-inf"))) > min_solo_delta]
    solo_ids = {id(m) for m in solos}
    held = [m for m in risky if id(m) not in solo_ids]
    risky_ids = {id(m) for m in risky}
    groupable = [m for m in members if id(m) not in risky_ids]
    groups, proven_solos, proven_held = _proven_groups_without_low_value_solos(
        groupable, chunk_size, min_solo_delta)
    solos += proven_solos
    held += proven_held
    chunks = [[m] for m in solos] + groups
    validate_chunks(chunks, min_solo_delta)
    return chunks, solos, groupable, held


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",
                    default=os.path.join(CLEAN, "submission.zip"),
                    help="base zip to swap members into — DEFAULT IS THE LIVE PIN "
                         "(neurogolf_clean/submission.zip); pass the hardened variant "
                         "once pad_biases has been run on the CURRENT pin")
    ap.add_argument("--chunk", type=int, default=12)
    ap.add_argument("--min-solo-delta", type=float, default=MIN_SOLO_DELTA,
                    help="emit a solo only when repriced delta is strictly greater than this; "
                         "default 0 because bank already enforces each lane's positive bar")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-reprice", action="store_true", help="trust LEDGER deltas (NOT recommended)")
    ap.add_argument("--regate", action="store_true",
                    help="re-run the full gate (2nd independent secret stream) on every member — "
                         "the historical registerable contract was secret×3; workers ran it once")
    ap.add_argument("--aggregate", action="store_true",
                    help="ALSO emit one wave_ALL zip = base + every repriced/regated non-canary "
                         "member (incl. HELD unproven small wins — the p811/p812 full-bank "
                         "pattern; final=best + exact predictions make it one risk-free probe, "
                         "and a short names its culprits via recon bisect suspects)")
    a = ap.parse_args()
    # base score comes from the LIVE pin record, never a hardcoded constant
    import hashlib, json as _json
    base_score = BASE_SCORE
    try:
        pin = _json.load(open(os.path.join(CLEAN, "logs/kaggle_pin.json")))
        base_score = float(pin["score"])
        if hashlib.sha256(open(a.base, "rb").read()).hexdigest()[:8] != pin.get("sha"):
            print(f"NOTE: base zip is not the recorded pin ({pin.get('sha')}) — predicted "
                  f"scores are approximate (pin score {base_score} used as baseline).")
    except Exception as e:
        print(f"WARNING: could not read kaggle_pin.json ({e}) — using {BASE_SCORE}")
    r0 = subprocess.run([sys.executable, os.path.join(HERE, "scan_razors.py"),
                         "--base", a.base], capture_output=True, text=True)
    if r0.returncode != 0:
        live = os.path.join(CLEAN, "submission.zip")
        try:
            base_sha = hashlib.sha256(open(a.base, "rb").read()).hexdigest()[:8]
            live_sha = hashlib.sha256(open(live, "rb").read()).hexdigest()[:8]
            recorded = open(os.path.join(ROOT, "data", "PIN_SHA.txt")).read().strip()
        except OSError:
            base_sha = live_sha = recorded = ""
        if not (base_sha and base_sha == live_sha == recorded):
            sys.exit("HARD FAIL: a non-pin BASE zip carries razors; refusing an unproven "
                     "implicit swap:\n" + r0.stdout.strip()[-300:])
        print("WARN: exact Kaggle-confirmed base carries inherited razor; preserving its "
              "bytes for attributable probes (final=best protects the standing score).")
    rows = [r for r in csv.DictReader(open(LEDGER)) if r.get("task")]
    meta = {}
    for r in rows:
        t = r["task"][4:]
        m = meta.setdefault(t, {"lane": r.get("lane", ""), "canary": False})
        # Explicit CANARY is task-class evidence. ORACLE-SOLO is artifact-specific and is
        # recomputed from exact bytes below; never trust/truncate free-form prose for it.
        m["canary"] |= (t in CANARY_TASKS or "CANARY" in (r.get("notes") or "").upper())
    # ARTIFACT-CENTRIC: consider canonical AND archived variants (taskNNN__sha8.onnx) —
    # after a repin the best current-pin artifact is often an archive (audit 2026-07-07:
    # +0.32 was invisible). Ledger deltas are NOT trusted (stale vs repins) — everything
    # is repriced below.
    import glob as _g
    members = []
    transient_failures = []
    rejected = rejected_artifacts()
    for t, m in meta.items():
        d = os.path.join(ROOT, "results", f"task{t}")
        arts = ([os.path.join(d, f"task{t}.onnx")] if os.path.exists(os.path.join(d, f"task{t}.onnx")) else []) \
               + sorted(_g.glob(os.path.join(d, f"task{t}__*.onnx")))
        if not arts:
            print(f"skip task{t}: no artifact")
            transient_failures.append(f"task{t}:missing-artifact")
            continue
        for p in arts:
            full_sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
            if (t, full_sha) in rejected:
                print(f"DROP task{t}: exact artifact SHA is persistently quarantined")
                continue
            members.append({"t": t, "path": p, "delta": 0.0, "lane": m["lane"],
                            "canary": m["canary"], "sha256": full_sha})
    if not members:
        sys.exit("no bankable members in LEDGER")
    # trust-but-verify: LEDGER deltas are agent-written — reprice (and optionally regate) each
    if a.no_reprice:
        sys.exit("REFUSED: --no-reprice banks stale deltas (task363 ledger said +0.156, "
                 "real Δ was 0.000 after a repin). Repricing is mandatory.")
    if True:
        kept = []
        for m in members:
            bar = "0.15"
            try:
                v = subprocess.run([sys.executable, os.path.join(HERE, "fast_verify.py"),
                                    str(int(m["t"])), m["path"], "--price", "--bar", bar],
                                   capture_output=True, text=True, timeout=600)
            except (subprocess.TimeoutExpired, OSError) as exc:
                print(f"DROP task{m['t']}: reprice isolated failure ({type(exc).__name__})")
                transient_failures.append(f"task{m['t']}:reprice-{type(exc).__name__}")
                continue
            if "PIN MOVED" in v.stdout:
                # fast_verify warns when PIN_SHA.txt ≠ live zip sha — every delta below would
                # be computed against a STALE per-task basis (audit F4; live incident
                # 2026-07-09: repin 07:21, 33 members changed, CSV unrefreshed).
                sys.exit("HARD STOP: PIN MOVED — per-task CSV is stale vs the live pin; all "
                         "deltas would be wrong. Run the refresh ritual first:\n"
                         "  python3 runner/rescore_pin.py && python3 runner/refresh_basis.py "
                         "&& python3 runner/build_queue.py --reprioritize")
            d = re.search(r"Δ=([+-][\d.]+)", v.stdout)
            if v.returncode != 0 or not d:
                detail = (v.stdout + v.stderr).strip()
                print(f"DROP task{m['t']}: reprice failed ({detail[-120:]})")
                if re.search(r"ModuleNotFoundError|No space left|temporar|resource unavailable|"
                             r"timed?\s*out|capacity", detail, re.I):
                    transient_failures.append(f"task{m['t']}:reprice-infrastructure")
                continue
            rp = float(d.group(1))
            if abs(rp - m["delta"]) > 0.02:
                print(f"task{m['t']}: LEDGER Δ{m['delta']:+.3f} ≠ repriced Δ{rp:+.3f} — using repriced")
                m["delta"] = rp
            if a.regate:
                try:
                    g = subprocess.run(regate_command(m, bar), capture_output=True, text=True,
                                       timeout=2400)
                except (subprocess.TimeoutExpired, OSError) as exc:
                    print(f"DROP task{m['t']}: regate isolated failure ({type(exc).__name__})")
                    transient_failures.append(f"task{m['t']}:regate-{type(exc).__name__}")
                    continue
                if g.returncode != 0:
                    mode = "vs-pin equivalence" if m["t"] in AMBIG_TASKS else "2nd secret-stream gate"
                    print(f"DROP task{m['t']}: {mode} FAILED — do not bank"); continue
            kept.append(m)
        if transient_failures:
            sys.exit("TRANSIENT BANK FAILURE: refusing a partial plan that could consume "
                     "unresolved ledger members: " + ", ".join(transient_failures[:12]))
        members = kept
        if not members:
            sys.exit("no members survived repricing")
    # dedupe: keep best delta per task
    best = {}
    for m in members:
        if m["t"] not in best or m["delta"] > best[m["t"]]["delta"]:
            best[m["t"]] = m
    members = sorted(best.values(), key=lambda m: -m["delta"])

    # Batching authority is ONLY exact per-member Kaggle payment proof.  Full-file hashes
    # are compared against corpus artifacts (not just the legacy sha8 manifest key), so a
    # one-byte edit immediately becomes unproven and returns to solo-probe-only.
    for m in members:
        classify_artifact_risk(m)
        if not m["payment_proof"]:
            print(f"UNPROVEN task{m['t']}: exact SHA has no reconciled per-member Kaggle "
                  f"proof; never batch it")
        if m.get("oracle_route") != "BATCH" and not m["payment_proof"]:
            print(f"CANARY task{m['t']}: recomputed artifact route "
                  f"{m.get('oracle_route')} ({'; '.join(m.get('oracle_reasons') or [])})")

    # zeros-probe the candidate set (new members only — the base is Kaggle-proven): a crash
    # on the degenerate all-zeros grid is a HEURISTIC (not proof of grader failure), so it
    # never batches a member — it promotes it to CANARY status.  Point-max policy then
    # emits every strictly positive canary as an attributable solo.
    import tempfile as _tf
    for m in members:
        try:
            r = subprocess.run([sys.executable, "-c",
                                "import sys,numpy as np,onnxruntime as ort;"
                                "o=ort.SessionOptions();o.log_severity_level=4;"
                                f"s=ort.InferenceSession(open({m['path']!r},'rb').read(),o);"
                                "d=np.zeros((1,10,30,30),np.float32);d[0,0,0,0]=1;"
                                "s.run(None,{'input':d})"], capture_output=True, timeout=120)
            probe_failed = r.returncode != 0
            detail = f"rc={r.returncode}"
        except (subprocess.TimeoutExpired, OSError) as exc:
            probe_failed = True
            detail = type(exc).__name__
        if probe_failed:
            print(f"WARN task{m['t']}: zeros-probe failure ({detail}) — marked "
                  f"CANARY (solo above threshold, otherwise held)")
            m["canary"] = True

    chunks, solos, groupable, held = shipping_chunks(members, a.chunk, a.min_solo_delta)
    os.makedirs(BANK, exist_ok=True)
    n_unproven = sum(1 for m in solos if not m.get("payment_proof"))
    n_canary = sum(1 for m in solos if m.get("canary"))
    for m in held:
        why = ("canary" if m.get("canary") else
               "unproven" if not m.get("payment_proof") else "proven singleton")
        print(f"HOLD task{m['t']}: {why} requires solo but repriced Δ{m['delta']:+.3f} "
              f"is not > +{a.min_solo_delta:.3f}")
    print(f"{len(members)} members → {len(solos)} quota-worthy solos "
          f"({n_unproven} unproven, {n_canary} canary flags) + "
          f"{len(chunks)-len(solos)} proven-only chunks (≤{a.chunk}) + "
          f"{len(held)} held\n")
    for i, ch in enumerate(chunks, 1):
        validate_chunks([ch], a.min_solo_delta)  # defense in depth
        pred = base_score + sum(m["delta"] for m in ch)
        kind = "SOLO" if len(ch) == 1 else f"{len(ch)}sw"
        ident = bundle_id(a.base, ch)
        name = f"wave_{i:02d}_{kind}_id{ident}_pred{pred:.2f}.zip"
        out = os.path.join(BANK, name)
        desc = " ".join(f"{m['t']}+{m['delta']:.4f}" for m in ch)
        print(f"{name}: {desc}")
        if a.dry_run:
            continue
        write_bundle(a.base, out, ch)
        r = subprocess.run([sys.executable, os.path.join(HERE, "scan_razors.py"), "--base", out],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  RAZOR SCAN FAILED — do not submit:\n{r.stdout}")
            continue
        print(f"  cd {CLEAN} && python3 candidates/probe_submit.py "
              f"submit {out} --lane wave1 --note '{desc[:120]}'")
    if a.aggregate:
        # the full-bank pattern (p811 banked27 → 7374.92 exact, p812 → 7375.33 exact):
        # Aggregate only exact per-member Kaggle-proven bytes. Unproven members now ship as
        # attributable positive solos (quota guard lives in auto_submit); duplicating them
        # inside a mega-batch loses causal attribution and can strand every gain behind one
        # silent-zero member.
        agg = [m for m in members if not m.get("canary") and m.get("payment_proof")]
        if agg:
            pred = base_score + sum(m["delta"] for m in agg)
            ident = bundle_id(a.base, agg)
            name = f"wave_ALL_{len(agg)}m_id{ident}_pred{pred:.2f}.zip"
            out = os.path.join(BANK, name)
            desc = " ".join(f"{m['t']}+{m['delta']:.4f}" for m in agg)
            print(f"{name}: {desc}")
            if not a.dry_run:
                write_bundle(a.base, out, agg)
                r = subprocess.run([sys.executable, os.path.join(HERE, "scan_razors.py"),
                                    "--base", out], capture_output=True, text=True)
                if r.returncode != 0:
                    print(f"  AGGREGATE RAZOR SCAN FAILED — do not submit:\n{r.stdout}")
                else:
                    print(f"  cd {CLEAN} && python3 "
                          f"candidates/probe_submit.py submit {out} --lane waveALL "
                          f"--note '{desc[:120]}'")
        else:
            print("(aggregate: no non-canary members — nothing to emit)")
    print("\nSubmit only the emitted >threshold solos, then proven-only chunks; watch each "
          "result vs prediction. A ≈−18.8 short = razor "
          "regression (investigate base), NOT member failure; a ≈−22.7 short = the task312 "
          "fp32-margin sentinel flipped (base member, not ours). final=best — probes are free.")


if __name__ == "__main__":
    main()
