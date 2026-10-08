#!/usr/bin/env python3
"""build_queue — FRESH-START queue from the CURRENT pin scoring + accumulated learning.

Rebuilds lane membership from the new per-task scores (bands move when pins absorb wins)
and orders work to maximize expected points:
  priority 1: HISTORY-RICH tasks (attempts + verified rule.py / postmortems exist) —
              measured att≥2 conversion 35% vs 24% cold — sorted by headroom;
  priority 2: cold tasks by headroom.
Lanes: D = named-brief set (score < 19.15) ∪ measured-AMBIG {002,118,187,255} (vs-pin
route); B = score < 18 (monsters included at top); C = <17; G = 18–18.6;
H = ≥18.6 REOPENED band (2026-07-12 study7300: the previously-banned band converted as
well as the queued one — 52 verified wins/+22.8 pts landed there — and holds the
cheapest attempts; ratio-sweep discipline, bar 0.15, cheap models best-of-N). Only
tasks at/inside the 8000 goal line (≥20.4 pts ⇒ cost ≤ ~100) stay out. Lane A is
assigned in a SECOND pass (flip_crop_lane) after compile_strategies marks CROP-NATIVE
primaries.
Run order: build_queue.py → compile_strategies.py → find_donors.py →
           build_queue.py --flip-crop → regenerate packs.

Third mode (--reprioritize, safe anytime between waves): keeps lane membership, bars and
columns EXACTLY as they are, and only re-ranks `priority` within each lane by REMAINING
value: rem = (19 − pin_pts) − banked_best_delta(LEDGER). Tiers: (0) unbanked history-rich
(measured 35% vs 24% conversion), (1) unbanked cold, (2) banked house-money re-attacks
last — a task already milked for +Δ must not outrank an untouched monster.
"""
import argparse, csv, glob, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from evidence import load_era, pending_ledger_rows
from task_difficulty import current_strong_attempts
# 2026-07-11 SPEARHEAD UNLOCK (operator-approved): 018/023 out of NEVER and 286/366 out of
# FLOORS — they hold the largest honest headroom on the board (023: known ~+1.5 with a
# 5.68% certified hidden-set exclusion) and their old NO evidence predates the scorer fix.
# S-contract/oracle framing + solo-probe-only discipline for novel families applies.
NEVER = set("209 076 080 350 338 204 191 216 128 179 241 144 230".split())
FLOORS = set("054 064 074 077 145 158 173 324 349 364".split())
# never-won Lane-D monsters that ate 7-12 attempts each for 0 progress (audit 2026-07-08 /
# C-F6): park them so the codex workers stop re-grinding proven-dead ground.
FLOORS |= set("219".split())
AMBIG_D = ["002", "118", "187", "255"]
NAMED_D = "133 118 286 366 187 219 243 002 370 157 101 255 285 233 198 138 205 066 025".split()


def history_rich(t):
    return (os.path.exists(os.path.join(ROOT, "results", f"task{t}", "ATTEMPTS.jsonl"))
            and (os.path.exists(os.path.join(ROOT, "packs", f"task{t}", "rule.py"))
                 or glob.glob(os.path.join(ROOT, "results", f"task{t}", "POSTMORTEM_att*.md"))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flip-crop", action="store_true",
                    help="second pass: move CROP-NATIVE-primary tasks into Lane A (bar 0.15)")
    ap.add_argument("--reprioritize", action="store_true",
                    help="re-rank priorities within lanes by remaining headroom minus "
                         "banked LEDGER delta; membership/bars untouched")
    a = ap.parse_args()
    qp = os.path.join(ROOT, "data", "lanes", "QUEUE.csv")
    if a.reprioritize:
        banked = {}
        lp = os.path.join(ROOT, "results", "LEDGER.csv")
        if os.path.exists(lp):
            for r in pending_ledger_rows(lp):
                try:
                    banked[r["task"][4:]] = max(banked.get(r["task"][4:], 0.0),
                                                float(r["delta"]))
                except (ValueError, KeyError):
                    pass
        rows = list(csv.DictReader(open(qp)))
        donors = {}
        donp = os.path.join(ROOT, "data", "donors.json")
        if os.path.exists(donp):
            try:
                donors = json.load(open(donp))
            except ValueError:
                donors = {}
        def _natt(t):
            return len(current_strong_attempts(os.path.join(ROOT, "results"), t))
        def key(r):
            t = r["task"][4:]
            b = banked.get(t, 0.0)
            rem = max(0.0, (19 - float(r["pin_pts"])) - b)
            best_donor = max((float(d.get("sim", 0)) for d in donors.get(t, [])), default=0.0)
            # Textual donor similarity had no measured conversion lift in the clean era;
            # never let it override observed task outcomes. Current strong attempt evidence
            # breaks EV ties ahead of genuinely cold tasks; absorbed historical files do not.
            tier = 2 if b > 0 else (0 if _natt(t) > 0 else 1)
            return (tier, -rem)
        # refresh difficulty+EV off the new pin FIRST (rank_difficulty reads this QUEUE + pin
        # CSV → DIFFICULTY.csv with per-task ev). The QUEUE is then ordered GLOBALLY by expected
        # value E[Δ]·P(win): the router handles cost (difficulty→model), so the queue purely
        # maximizes banked points. Floored/EV=0 sink to the bottom; ties fall back to the
        # lane/headroom key. (2026-07-10)
        import subprocess
        rc = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__),
                             "rank_difficulty.py")], capture_output=True, text=True)
        ev = {}
        try:
            for d in csv.DictReader(open(os.path.join(ROOT, "results", "DIFFICULTY.csv"))):
                ev[d["task"][4:]] = float(d.get("ev", 0) or 0)
        except OSError:
            pass
        print(rc.stdout.strip().split("\n")[0] if rc.returncode == 0 else
              f"WARN: rank_difficulty failed ({rc.stderr.strip()[-160:]}) — EV order unavailable, "
              f"falling back to lane/headroom; router defaults tasks to MEDIUM")
        by_lane = {}
        for r in rows:
            by_lane.setdefault(r["lane"], []).append(r)
            r["_key"] = key(r)
        # global EV-first order (desc); lane/headroom key breaks ties and drives the fallback
        out = sorted(rows, key=lambda r: (-ev.get(r["task"][4:], 0.0), r["_key"]))
        for i, r in enumerate(out, 1):
            r["priority"] = str(i)
            r.pop("_key", None)
        with open(qp + ".tmp", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[c for c in rows[0].keys() if c != "_key"],
                               lineterminator="\n")
            w.writeheader(); w.writerows(out)
        os.replace(qp + ".tmp", qp)
        nb = sum(1 for r in out if banked.get(r["task"][4:], 0) > 0)
        print(f"reprioritized {len(out)} tasks by EXPECTED VALUE ({nb} banked in house-money "
              f"tier); ΣEV={sum(ev.values()):.2f}. Queue front (task/lane/EV):")
        for r in out[:12]:
            print(f"  {r['task']}/{r['lane']} EV={ev.get(r['task'][4:],0):.3f}")
        return
    if a.flip_crop:
        strat = json.load(open(os.path.join(ROOT, "data", "strategies.json")))
        rows = list(csv.DictReader(open(qp)))
        n = 0
        for r in rows:
            t = r["task"][4:]
            c = strat.get(t, {})
            if (r["lane"] in ("B", "C", "G", "H")
                    and c.get("primary", [None])[0] == "CROP-NATIVE (certified)"):
                r["lane"] = "A"
                r["bar_cost"] = str(int(int(r["pin_cost"]) * math.exp(-0.15)))
                n += 1
        with open(qp + ".tmp", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
            w.writeheader(); w.writerows(rows)
        os.replace(qp + ".tmp", qp)
        print(f"flip-crop: {n} tasks moved to Lane A")
        return

    hexmap = json.load(open(os.path.join(ROOT, "data", "task_map.json")))
    certs = json.load(open(os.path.join(ROOT, "data", "source_certs.json")))
    rows = {}
    with open(os.path.join(ROOT, "data", "pin_pv2_70_263_pertask.csv")) as f:
        for r in csv.DictReader(f):
            if r["cost"]:
                rows[r["task"][4:]] = (int(r["cost"]), float(r["score"]))
    lanes = {"D": [], "B": [], "C": [], "G": [], "H": []}
    for t, (c, s) in rows.items():
        if t in NEVER or t in FLOORS:
            continue  # hard-excluded — NEVER/FLOORS take precedence over any lane exemption
        # 2026-07-12 (study7300/STRATEGY_8000 M1): the ≥18.6 exclusion is GONE — it dropped
        # 41% of the board while that band converted at least as well as the queued one.
        # Only tasks already at/inside the 8000 goal line (avg 20.4 pts ⇒ cost ≤ ~100,
        # i.e. at their rule-class compile target) leave the queue.
        if s >= 20.4:
            continue
        if (t in NAMED_D or t in AMBIG_D) and s < 19.15:
            lanes["D"].append(t)
        elif s < 17:
            lanes["C"].append(t)
        elif s < 18:
            lanes["B"].append(t)
        elif s < 18.6:
            lanes["G"].append(t)
        else:
            lanes["H"].append(t)
    out = []
    for L, ts in lanes.items():
        ts.sort(key=lambda t: (0 if history_rich(t) else 1, -(19 - rows[t][1])))
        for i, t in enumerate(ts, 1):
            c, s = rows[t]
            # Lane H is the reopened-band ratio sweep: attempts are minutes (tiny graphs,
            # fast gates) and the study shows the band dense with 1.2–6× factorizations —
            # Lane-A-style registration floor (0.05) so small exact cuts still register.
            bar = int(c * math.exp(-0.15))
            out.append({"task": f"task{t}", "lane": L, "priority": i, "pin_cost": c,
                        "pin_pts": f"{s:.3f}", "bar_cost": bar,
                        "gen_hex": hexmap[f"task{t}"],
                        "M_proof": certs.get(t, {}).get("M_proof")})
    with open(qp + ".tmp", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()), lineterminator="\n")
        w.writeheader(); w.writerows(out)
    os.replace(qp + ".tmp", qp)
    rich = sum(1 for r in out if history_rich(r["task"][4:]))
    print(f"fresh QUEUE: {len(out)} tasks — D:{len(lanes['D'])} B:{len(lanes['B'])} "
          f"C:{len(lanes['C'])} G:{len(lanes['G'])} H:{len(lanes['H'])} · "
          f"history-rich prioritized: {rich}")


if __name__ == "__main__":
    main()
