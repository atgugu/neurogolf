#!/usr/bin/env python3
"""preflight — GO/NO-GO for the wave. Run this BEFORE launching orchestrate.

A single broken CLI flag, expired login, or missing venv turns the whole night into zero
points silently. This tests every live dependency in ~2 minutes:
  1. the codex CLI with the EXACT command shapes orchestrate uses (1-token echo)
  2. the postmortem-analyst administrative command shape
  3. grader venv + gate124 on a known-good pin member (3 draws)
  4. ngolf import + micro-build + replica pricing agreement
  5. pack generation (make_pack on a queued task)
  6. data files, drive writability, pin consistency, and a razor-clean live base
Exit 0 = GO. Any FAIL = fix before launching.
"""
import csv, hashlib, json, os, subprocess, sys, tempfile, time, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
from config import CLEAN, RESOURCES, VENV_PY
VENV = VENV_PY or sys.executable
FAILS = []
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

sys.path.insert(0, HERE)
from integrity import integrity_errors


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def cli_echo(name, cmd, timeout=120):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           stdin=subprocess.DEVNULL)
        ok = p.returncode == 0 and "OK" in (p.stdout + p.stderr)
        check(f"CLI {name}", ok, f"{time.time()-t0:.0f}s"
              if ok else (p.stdout + p.stderr).strip()[-160:] or f"rc={p.returncode}")
    except Exception as e:
        check(f"CLI {name}", False, str(e)[:120])


def lock_is_free(path):
    import fcntl
    try:
        fh = open(path, "a+")
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()
        return True
    except OSError:
        return False


def basis_snapshot():
    hashes = {}
    mtimes = []
    for rel in BASIS_FILES:
        path = os.path.join(ROOT, rel)
        hashes[rel] = hashlib.sha256(open(path, "rb").read()).hexdigest()
        mtimes.append(os.path.getmtime(path))
    return hashes, max(mtimes)


def active_pipeline_processes():
    needles = ("runner/orchestrate.py", "runner/repin.py", "runner/make_pack.py",
               "runner/exp_ruleforge.py", "runner/exp_floorbuster.py",
               "runner/reprice_artifacts.py", "runner/pending_report.py",
               "runner/factor_sweep.py", "runner/pad_biases.py",
               "runner/refresh_basis.py", "runner/rank_difficulty.py",
               "runner/build_queue.py", "spec_from_file_location('neurogolf_utils'")
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == os.getpid():
            continue
        try:
            cmd = open(f"/proc/{entry}/cmdline", "rb").read().replace(b"\0", b" ").decode()
        except OSError:
            continue
        if any(n in cmd for n in needles):
            found.append(f"{entry}:{cmd[:100]}")
    return found


def main():
    print("== preflight ==")
    # Fail closed if a worker or operator changed a pricing authority.
    errs = integrity_errors()
    check("pricing-authority SHA256", not errs, "; ".join(errs)[:500])
    resources = subprocess.run([sys.executable, os.path.join(HERE, "resource_check.py")],
                               capture_output=True, text=True, timeout=120)
    check("host resources", resources.returncode == 0,
          "; ".join(resources.stdout.strip().splitlines())[-800:])
    # Readiness fence: launch only from one fully-published pin generation.
    active = active_pipeline_processes()
    check("no active wave/repin/pack writers", not active, "; ".join(active[:4]))
    # Codex-only fleet contract — a mis-set env fails closed instead of
    # launching a degraded wave.
    check("builder route is GPT-5.5/xhigh",
          os.environ.get("ROUTE_GPT_MODEL", "gpt-5.5") == "gpt-5.5")
    check("runtime ROI routing guards enabled",
          os.environ.get("RUNTIME_ROI", "1") == "1" and
          int(os.environ.get("RUNTIME_ROI_COVERAGE_EVERY", "10")) == 10)
    check("decider split is 50/50",
          abs(float(os.environ.get("DECIDER_SOL_SHARE", ".50")) - .5) < 1e-9)
    check("Sol decider route is GPT-5.6 Sol/xhigh",
          os.environ.get("DECIDER_SOL_MODEL", "gpt-5.6-sol") == "gpt-5.6-sol")
    check("administrative route is Codex Spark primary",
          os.environ.get("ADMIN_PRIMARY_PROVIDER", "codex") == "codex" and
          os.environ.get("ADMIN_SPARK_MODEL", "gpt-5.3-codex-spark") ==
          "gpt-5.3-codex-spark")
    # Submission mode must be an explicit 0/1 — a typo must never silently arm (or disarm)
    # the automatic loop. 0 = manual banking (default); 1 = auto_submit.py runs the
    # bank→submit→recon→repin cycle behind its own rails.
    _autosubmit = os.environ.get("KAGGLE_AUTOSUBMIT", "0")
    check("Kaggle submission mode is explicit", _autosubmit in ("0", "1"),
          "" if _autosubmit in ("0", "1") else
          f"KAGGLE_AUTOSUBMIT={_autosubmit!r} — expected '0' (manual) or '1' (automatic)")
    print(f"  MODE  Kaggle submission: "
          f"{'AUTOMATIC — auto_submit.py is armed' if _autosubmit == '1' else 'manual banking'}")
    for lock_name in ("run_wave.lock", "REPIN.lock"):
        check(f"{lock_name} is free", lock_is_free(os.path.join(ROOT, "results", lock_name)))
    fleet_path = os.path.join(ROOT, "results", "FLEET.json")
    if os.path.exists(fleet_path):
        try:
            fleet = json.load(open(fleet_path))
            desired = {p: int(fleet.get(p, 0)) for p in ("codex",)}
            check("fleet drained before launch", not any(desired.values()), str(desired))
        except (ValueError, TypeError, OSError) as exc:
            check("fleet drained before launch", False, str(exc))
    else:
        print("  PASS  fleet drained before launch — no FLEET.json (fresh repo)")
    incomplete = os.path.join(ROOT, "results", "REPIN_INCOMPLETE.json")
    check("no incomplete repin marker", not os.path.exists(incomplete), incomplete)
    manifest_path = os.path.join(ROOT, "results", "REPIN_MANIFEST.json")
    basis_epoch = float("inf")  # fail packs closed if manifest/basis cannot be read
    try:
        manifest = json.load(open(manifest_path))
        live_sha = hashlib.sha256(open(f"{CLEAN}/submission.zip", "rb").read()).hexdigest()[:8]
        check("repin manifest complete", manifest.get("status") == "complete",
              f"status={manifest.get('status')}")
        check("repin manifest matches live pin", manifest.get("live_pin") == live_sha,
              f"manifest={manifest.get('live_pin')} live={live_sha}")
        nonce = str(manifest.get("run_nonce", ""))
        manifest_shape = (manifest.get("generated_by") == "runner/repin.py"
                          and len(nonce) >= 16
                          and manifest.get("pin_csv_rows") == 400
                          and manifest.get("pin_csv_bad_rows") == 0
                          and manifest.get("packs_verified") == manifest.get("queue_tasks")
                          and manifest.get("packs_verified", 0) > 0
                          and manifest.get("hardened_source_pin") == live_sha)
        check("repin manifest has verified provenance", manifest_shape,
              f"producer={manifest.get('generated_by')} nonce={nonce[:8]} "
              f"packs={manifest.get('packs_verified')}/{manifest.get('queue_tasks')}")
        repin_log = open(os.path.join(ROOT, "results", "REPIN.log"),
                         errors="replace").read()
        check("repin completion receipt", bool(nonce) and
              f"=== REPIN done nonce={nonce} ===" in repin_log,
              "receipt present" if f"=== REPIN done nonce={nonce} ===" in repin_log
              else "matching completion nonce absent from REPIN.log")
        current_basis, _ = basis_snapshot()
        basis_matches = manifest.get("basis_sha256") == current_basis
        check("repin basis fingerprint current", basis_matches,
              "exact basis hashes match" if basis_matches else
              "queue/strategy/donor/difficulty basis changed after refresh")
        if basis_matches:
            basis_epoch = float(manifest.get("basis_epoch", float("inf")))
    except (OSError, ValueError) as exc:
        manifest, live_sha = {}, "?"
        check("repin manifest complete", False, str(exc))
    # 1. model CLIs — exact shapes from orchestrate.MODELS (cheap effort for the echo)
    P = "Reply with exactly: OK"
    cli_echo("codex (worker shape)",
             ["codex", "exec", "-m", os.environ.get("CODEX_MODEL", "gpt-5.5"),
              "-c", 'model_reasoning_effort="low"',
              "--dangerously-bypass-approvals-and-sandbox", P], timeout=240)
    if os.environ.get("CODEX_FAST_MODELS", "").strip():
        _fast = os.environ["CODEX_FAST_MODELS"].split(",")[0].strip()
        cli_echo(f"codex FAST tier ({_fast}, service_tier=priority)",
                 ["codex", "exec", "-m", _fast,
                  "-c", 'model_reasoning_effort="low"', "-c", 'service_tier="priority"',
                  "--dangerously-bypass-approvals-and-sandbox", P], timeout=240)
    cli_echo("codex Sol (decider availability)",
             ["codex", "exec", "-m", os.environ.get("DECIDER_SOL_MODEL", "gpt-5.6-sol"),
              "-c", 'model_reasoning_effort="low"', "--sandbox", "read-only",
              "--ephemeral", "--ignore-rules", "--skip-git-repo-check", P], timeout=240)
    cli_echo("administrative route (Codex Spark with GPT-5.5 backup)",
             [sys.executable, os.path.join(HERE, "admin_model.py"), "--prompt", P,
              "--effort", "low", "--timeout", "180"], timeout=240)

    # 2. grader venv + gate on a known-good member
    check("grader venv", os.path.exists(VENV), VENV)
    with tempfile.TemporaryDirectory() as td:
        m = os.path.join(td, "task179.onnx")
        open(m, "wb").write(zipfile.ZipFile(f"{CLEAN}/submission.zip").read("task179.onnx"))
        p = subprocess.run(["python3", "candidates/gate124.py", "179", m, "--draws", "3"],
                           cwd=CLEAN, capture_output=True, text=True, timeout=600)
        check("gate124 (3 draws, task179)", p.returncode == 0,
              (p.stdout + p.stderr).strip().splitlines()[-1][:120] if (p.stdout + p.stderr).strip() else "")

    # 3. ngolf micro-build + replica pricing agreement
    try:
        sys.path.insert(0, HERE)
        from ngolf import G
        g = G()
        st = g.color_plane(0, cast="u8")
        g.out_equal_arange(st, K=10)
        with tempfile.TemporaryDirectory() as td:
            mp = os.path.join(td, "t.onnx")
            g.save(mp, ort_check=True)
            sys.path.insert(0, os.path.join(RESOURCES, "audit_2026-07-06"))
            import numpy as np
            from score_bundle import score_onnx
            demo = np.zeros((1, 10, 30, 30), np.float32); demo[0, 0, 0, 0] = 1
            r = score_onnx(mp, demo)
            check("ngolf build + replica agreement", r["cost"] == g.cost(),
                  f"ngolf={g.cost()} replica={r['cost']}")
    except Exception as e:
        check("ngolf build", False, str(e)[:150])

    # 4. pack generation in a temporary root: preflight must never rewrite a canonical pack.
    with tempfile.TemporaryDirectory() as pack_root:
        env = dict(os.environ, PACKS_ROOT=pack_root)
        p = subprocess.run([sys.executable, os.path.join(HERE, "make_pack.py"), "363",
                            "--no-context"], capture_output=True, text=True, timeout=200,
                           env=env)
        check("make_pack temporary (task363)", p.returncode == 0 and
              os.path.exists(os.path.join(pack_root, "task363", "ATTACK.md")),
              p.stderr.strip()[-120:])

    # 5. data + writability
    for f in ("data/strategies.json", "data/donors.json", "data/source_certs.json",
              "data/pin_pv2_70_263_pertask.csv", "data/lanes/QUEUE.csv", "data/RULES.md",
              "data/DSL_TO_ONNX.md"):
        check(f, os.path.exists(os.path.join(ROOT, f)))
    try:
        t = os.path.join(ROOT, "results", ".write_test")
        open(t, "w").write("x"); os.remove(t)
        check("results/ writable", True)
    except OSError as e:
        check("results/ writable", False, str(e))
    try:
        pin_rows = list(csv.DictReader(open(os.path.join(ROOT, "data",
                                                          "pin_pv2_70_263_pertask.csv"))))
        bad = [r.get("task", "?") for r in pin_rows
               if r.get("status") != "ok" or not r.get("score", "").strip()]
        check("pin CSV complete", len(pin_rows) == 400 and not bad,
              f"rows={len(pin_rows)} bad={bad[:8]}")
        queued = []
        for row in csv.DictReader(open(os.path.join(ROOT, "data", "lanes", "QUEUE.csv"))):
            task = row.get("task", "")
            if task:
                queued.append(task)
        stale_packs = []
        for task in queued:
            attack = os.path.join(ROOT, "packs", task, "ATTACK.md")
            try:
                if (os.path.getmtime(attack) < basis_epoch or
                        f"LIVE pin {live_sha}" not in open(attack, errors="replace").read()):
                    stale_packs.append(task)
            except OSError:
                stale_packs.append(task)
        check("all queued packs match live pin", not stale_packs,
              f"queued={len(queued)} stale={stale_packs[:12]}")
        ticket_summary = open(os.path.join(ROOT, "data", "tickets", "SUMMARY.md"),
                              errors="replace").read()
        check("factor tickets match live pin", f"Pin: {live_sha}" in ticket_summary,
              f"expected Pin: {live_sha}")
        harden_sha = open(os.path.join(ROOT, "results", "hardened",
                                       "hardened_base.src_sha")).read().strip()
        check("hardened base matches live pin", harden_sha == live_sha,
              f"hardened={harden_sha} live={live_sha}")
        csv_sha = open(os.path.join(ROOT, "data", "PIN_CSV_SHA.txt")).read().strip()
        check("pin CSV provenance matches live pin", csv_sha == live_sha,
              f"csv={csv_sha} live={live_sha}")
    except OSError as exc:
        check("derived generation consistency", False, str(exc))
    # pin freshness: every bar/target flows from the pin CSV — a moved pin = stale numbers
    try:
        want = open(os.path.join(ROOT, "data", "PIN_SHA.txt")).read().strip()
        live = f"{CLEAN}/submission.zip"
        have = hashlib.sha256(open(live, "rb").read()).hexdigest()[:8]
        check("pin CSV matches live pin", want == have,
              f"recorded {want} vs live {have}" + ("" if want == have else
              " — RESCORE the pin and refresh data/ before launching"))
        pin_meta = json.load(open(f"{CLEAN}/logs/kaggle_pin.json"))
        check("Kaggle pin metadata matches live pin", pin_meta.get("sha") == have,
              f"metadata={pin_meta.get('sha')} live={have}")
        razor = subprocess.run([sys.executable, os.path.join(HERE, "scan_razors.py"),
                                "--base", live], capture_output=True, text=True, timeout=300)
        if razor.returncode == 0:
            check("live Kaggle pin razor scan", True)
        else:
            # An inherited finding in the exact Kaggle-confirmed base is noisy but does not
            # invalidate candidate attribution. Banking must use these exact bytes; it may
            # never silently substitute an unproven hardened derivative.
            print("  WARN  exact confirmed live pin carries inherited razor — final=best "
                  "protects standing score; keep banking on the exact pin bytes")
    except OSError as e:
        check("pin CSV matches live pin", False, str(e))

    print(f"\n{'GO — launch the wave' if not FAILS else 'NO-GO — fix: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
