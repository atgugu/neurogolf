#!/usr/bin/env python3
"""whatworks_miner — Codex Spark distills the fixed-scorer-era postmortems into
results/WHAT_WORKS_NOW.md: which families converted since the scorer fix, by difficulty
band, plus known dead ends (incl. the fitter's 131-task low-degree infeasibility map).
Factual distillation only — injected as decider evidence, never gates anything.

  python3 runner/whatworks_miner.py
"""
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(HERE))
from evidence import pending_ledger_rows
from proven_corpus import payment_proof_for_path

PROMPT = """SINGLE-RESPONSE TASK. You have NO tools; your FIRST message IS the final answer.
You are distilling fleet evidence for NeuroGolf ONNX graph-golf deciders.
Below: (1) exact Kaggle-confirmed artifacts only, (2) excerpts from recent post-mortems of
wins AND failures, (3) the closed-form fitter's infeasibility result.
Write results/WHAT_WORKS_NOW.md content — AT MOST 25 bullets, grouped under these headers:
## Families that CONVERT now (cite the tasks + deltas from the ledger)
## LOCAL-BANKED, unshipped (families winning locally; NOT Kaggle-confirmed yet)
## Families that keep FAILING (cite the repeated root causes)
## Dead ends (do not spend attempts)
Facts only from the material below; no invention; name tasks and families precisely.
STRICT: the CONVERT section may cite ONLY tasks from the KAGGLE-CONFIRMED block; every
locally-banked/pending win belongs under LOCAL-BANKED (this is the accumulate-mode
norm: wins pile up locally between manual submission windows, and deciders still need
the family signal the same night).

=== EXACT KAGGLE-CONFIRMED ARTIFACTS (the ONLY conversion evidence) ===
{ledger}

=== LOCAL-PASS / PENDING CANDIDATES (cite ONLY under the LOCAL-BANKED header) ===
{pending}

=== POSTMORTEM EXCERPTS ===
{pms}

=== FITTER EVIDENCE ===
The closed-form single-Einsum fitter swept 131 queued tasks with a 9-template zoo:
134/139 INFEASIBLE_ALL_TEMPLATES, 0 banked. Conclusion for deciders: the residual queue's
rules are NOT low-degree multilinear — 'fit a small Einsum' framings are a dead end;
algorithmic structure (flood/CC/sort/scan-free equivalents) is where the wins are.
"""


def main():
    # Keep the prompt compact enough for a fast administrative distillation.
    import csv
    confirmed, pending = [], []
    for row in pending_ledger_rows(str(RESULTS / "LEDGER.csv")):
        task = int(row["task"][4:])
        paths = ([RESULTS / row["task"] / f"{row['task']}.onnx"]
                 + list((RESULTS / row["task"]).glob(f"{row['task']}__*.onnx")))
        exact = next((p for p in paths if p.is_file()
                      and __import__("hashlib").sha256(p.read_bytes()).hexdigest()[:8]
                      == row.get("sha8")), None)
        proof = payment_proof_for_path(task, str(exact)) if exact else None
        line = ",".join(str(row.get(k, "")) for k in
                        ("task", "cost", "pts", "delta", "lane", "sha8", "notes"))
        (confirmed if proof else pending).append(line)
    ledger = "\n".join(confirmed) or "(none yet in the current pending frontier)"
    pending_text = "\n".join(pending)[:2500] or "(none)"
    pms = []
    files = sorted(glob.glob(str(RESULTS / "task*" / "POSTMORTEM_att*.md")),
                   key=os.path.getmtime, reverse=True)[:24]
    for p in files:
        try:
            body = open(p, errors="replace").read()
        except OSError:
            continue
        head = f"--- {Path(p).parent.name}/{Path(p).name} ---\n"
        pms.append(head + body[:450])
    blob = "\n".join(pms)[:9500]
    prompt = PROMPT.format(ledger=ledger, pending=pending_text, pms=blob)
    r = subprocess.run([sys.executable, str(HERE / "admin_model.py"), "--prompt", prompt,
                        "--effort", os.environ.get("ADMIN_WHATWORKS_EFFORT", "low"),
                        "--timeout", "300"],
                       capture_output=True, text=True, timeout=600)
    out = RESULTS / "WHAT_WORKS_NOW.md"
    body = (r.stdout or "").strip()
    source_tasks = {int(x) for x in re.findall(r"task(\d{1,3})", prompt, re.I)}
    cited_tasks = {int(x) for x in re.findall(r"task(\d+)", body, re.I)}
    invented = sorted(t for t in cited_tasks if t not in source_tasks or not 1 <= t <= 400)
    if invented:
        print(f"admin output invented/uncited task ids {invented} — keeping existing {out}")
        return 1
    convert_text = body.split("## Families that CONVERT now", 1)[-1].split("## ", 1)[0]
    convert_tasks = {int(x) for x in re.findall(r"task(\d+)", convert_text, re.I)}
    confirmed_tasks = {int(x) for x in re.findall(r"task(\d+)", ledger, re.I)}
    if convert_tasks - confirmed_tasks:
        # 2026-07-13: local wins now have their own legal section (LOCAL-BANKED), so a
        # pending task under CONVERT is a genuine prompt violation, and dropping the
        # whole output is no longer the common case (wave6: this guard discarded ~28/35
        # refreshes because pending wins had nowhere legal to go).
        print(f"admin model promoted local-only tasks {sorted(convert_tasks-confirmed_tasks)} as converts "
              f"— keeping existing {out}")
        return 1
    if len(body) < 400 or "## " not in body:
        # NEVER clobber the last good distillation with an admin outage/refusal — every
        # decider prompt reads this file; a failed refresh must be a no-op, not a wipe.
        print(f"admin output unusable ({len(body)} chars) — keeping existing {out}")
        return 1
    tmp = out.with_suffix(".tmp")
    tmp.write_text("# What works now (fixed-scorer era) — Codex Spark distillation\n\n" + body + "\n")
    os.replace(tmp, out)  # atomic: deciders read this file concurrently mid-wave
    print(f"wrote {out} ({len(body)} chars from {len(files)} postmortems)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
