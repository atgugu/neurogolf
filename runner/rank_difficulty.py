#!/usr/bin/env python3
"""rank_difficulty — score each queued task's DIFFICULTY (how much reasoning depth a
Kaggle-real cost win needs) → drives the model×effort routing tier.

Difficulty predicts "expected reasoning to produce a bankable structural win", from
transparent on-disk signals. Output: results/DIFFICULTY.csv (task,score,tier,components…)
plus a routing tier per task. READ-ONLY; writes only the CSV.

Signals (higher score = harder):
  STRUCT (0-40)   how big a cost cut is needed — from pin_pts (low pts = memory-dominated
                  = needs plane-kill) and cost-ratio to bar.
  HISTORY(0-25)   attempt hardness — #real NOs, floored-by-2 (→ excluded), att-depth.
  FAMILY (0-20)   primary strategy family — mechanical(relower/init-pack/crop) < einsum <
                  rebuild-from-rule/rare/morphology.
  RULE   (0-10)   Hodel TRUE-RULE (rule known) < PARTIAL/PROVISIONAL/AMBIG (rule uncertain).
  NOVELTY(0-23)   curated Lane D + very low points + no exact proven design/strong donor.
  TRANSFER(-20-0) reduces difficulty — strong donor (sim≥0.7) and #diverse proven designs
                  make a win a transfer, not an invention.
"""
import csv, glob, json, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
from config import CLEAN
sys.path.insert(0, HERE)
from evidence import load_era, pending_ledger_rows
from task_difficulty import current_strong_attempts
ERA = load_era()

FAM_HARD = {  # primary family → family-hardness (0-20)
    "CROP-NATIVE (certified)": 4, "INIT-PACK": 4, "PLANE-KILL + u8 RELOWER": 8,
    "LUT-REFINE": 8, "TERMINAL-RENDERER FIX": 8, "COMPRESSED SCHEDULE": 10,
    "EINSUM-COMPRESS": 12, "MORPHOLOGY ROUND-CAP": 16, "REBUILD-FROM-RULE": 20,
}


def load():
    strat = json.load(open(f"{ROOT}/data/strategies.json"))
    donors = json.load(open(f"{ROOT}/data/donors.json"))
    proven = json.load(open(f"{CLEAN}/knowledge_proven_paying/manifest.json"))
    proven_n = {}
    for v in proven.values():
        proven_n[f"{v['task']:03d}"] = proven_n.get(f"{v['task']:03d}", 0) + 1
    pin = {}
    for r in csv.DictReader(open(f"{ROOT}/data/pin_pv2_70_263_pertask.csv")):
        if r["score"]:
            pin[r["task"][4:]] = (int(r["cost"]), float(r["score"]))
    return strat, donors, proven_n, pin


def real_no_count(t):
    """Count only fixed-scorer-era real NOs; old floors never set routing difficulty.
    Also returns the total real-attempt count (drives the EV conversion decay).

    NOs are counted only AFTER the task's most recent banked win (DONE/DONE_SALVAGED):
    a win refutes the floor hypothesis outright — wave6 audit 2026-07-13 found task198
    (won 5×, 3.76 ln-pts headroom left) parked at ev=0/SKIP-FLOORED by two NOs that
    predated three subsequent DONEs. Post-win NOs re-arm the floor normally."""
    attempts = current_strong_attempts(f"{ROOT}/results", t, ERA)
    last_win = -1
    for i, a in enumerate(attempts):
        if a.get("outcome") in ("DONE", "DONE_SALVAGED"):
            last_win = i
    no_attempts = [a for a in attempts[last_win + 1:] if a.get("outcome") == "NO"]
    nos = len(no_attempts)
    models_no = {a.get("model") for a in no_attempts if a.get("model")}
    return nos, (len(models_no) >= 2 and nos >= 2) or nos >= 3, len(attempts)


def banked_best():
    """Best banked delta per task from the LEDGER — already-banked points are no longer
    available, so they must not keep a task at the queue front (max-ROI, 2026-07-11)."""
    best = {}
    try:
        for r in pending_ledger_rows(f"{ROOT}/results/LEDGER.csv", ERA):
            try:
                t = r["task"][4:]
                best[t] = max(best.get(t, 0.0), float(r["delta"]))
            except (ValueError, KeyError):
                continue
    except OSError:
        pass
    return best


def score_task(t, strat, donors, proven_n, pin, queue_row=None):
    c = {}
    pc, pp = pin.get(t, (None, None))
    # STRUCT (0-40): low points imply a structural problem; a deep cost cut adds pressure.
    if pp is None:
        point_hard = 18
    elif pp >= 18.5: point_hard = 4
    elif pp >= 18:   point_hard = 7
    elif pp >= 17:   point_hard = 14
    elif pp >= 16:   point_hard = 22
    elif pp >= 15:   point_hard = 27
    else:            point_hard = 30
    bar = int((queue_row or {}).get("bar_cost") or 0)
    cut = max(0.0, 1 - bar / pc) if pc and bar else 0.0
    cut_hard = 10 if cut >= .40 else 6 if cut >= .25 else 3 if cut >= .15 else 0
    c["struct"] = point_hard + cut_hard
    # HISTORY (0-25)
    nos, floored, _ = real_no_count(t)
    attempts = current_strong_attempts(f"{ROOT}/results", t, ERA)
    failed_models = {a.get("model") for a in attempts
                     if a.get("outcome") not in ("DONE", "DONE_SALVAGED") and a.get("model")}
    c["history"] = min(25, len(attempts) * 2 + nos * 4
                       + max(0, len(failed_models) - 1) * 3)
    # FAMILY (0-20)
    card = strat.get(t) or strat.get(f"task{t}")
    fam = (card["primary"][0] if card and isinstance(card.get("primary"), list) else "?") if card else "?"
    c["family"] = FAM_HARD.get(fam, 12)
    # RULE (0-10)
    hod = (card.get("hodel", "") if card else "") or ""
    c["rule"] = 0 if "TRUE-RULE" in hod else (10 if ("PARTIAL" in hod or "PROVISIONAL" in hod or "AMBIG" in hod) else 5)
    # TRANSFER (-20..0)
    ds = donors.get(t) or donors.get(f"task{t}") or []
    best_sim = max((d.get("sim", 0) for d in ds), default=0.0)
    tr = 0
    if best_sim >= 0.7: tr -= 12
    elif best_sim >= 0.5: tr -= 6
    npv = proven_n.get(t, 0)
    if npv >= 3: tr -= 8
    elif npv >= 2: tr -= 4
    c["transfer"] = tr
    novelty = 0
    if (queue_row or {}).get("lane") == "D": novelty += 8
    if pp is not None and pp < 16: novelty += 6
    if npv == 0: novelty += 5
    if best_sim < .3: novelty += 4
    c["novelty"] = novelty
    total = max(0, min(100, sum(c.values())))
    tier = ("SKIP-FLOORED" if floored else
            "EASY" if total < 28 else "MEDIUM" if total < 50 else
            "HARD" if total < 72 else "VHARD")
    return total, tier, c, fam, pp, best_sim, npv, nos, floored, cut, len(attempts)


# EXPECTED-VALUE model (2026-07-12 study7300 rewrite): the queue maximizes
# P(win|tier,history) × remaining ln(pin_cost/target_cost); the model ROUTER separately
# handles cost (difficulty→cheap/expensive), so the two compose — no cost term here.
# The old E[Δ] capped headroom at 19−pts, which (a) ceilinged the whole board at ≈7770
# (M1) and (b) selected for the historical +0.17 median instead of the study's verified
# +0.44 mean. The prize is now the task's distance to its RULE-CLASS compile target —
# the score law is 25−ln(cost), so points-to-target IS ln(pin_cost/target_cost).
# P(win) priors stay the measured fixed-scorer-era conversions (2026-07-11): EASY 13.7%,
# MEDIUM 11.9%, HARD 5.1%, VHARD 0 observed (smoothed).
_PWIN   = {"EASY": 0.14, "MEDIUM": 0.12, "HARD": 0.06, "VHARD": 0.025}
_COST   = {"EASY": 1, "MEDIUM": 2, "HARD": 4, "VHARD": 5}              # token/quota weight (~model tier)

# Rule-class compile targets (elements), from the verified study7300 corpus (30 members
# ≤150 els; mechanisms cluster tightly): recolor/count/parity compile at 30–150,
# selector+render 100–400, crop/stamp 200–700, multi-object/relational 400–1500,
# flood/NMS monsters pin×0.5 until someone compiles the rounds away. Keyed by the
# strategy card's primary family (the best per-task rule-class signal on disk).
_FAMILY_TARGET = {
    "REBUILD-FROM-RULE": 150, "EINSUM-COMPRESS": 150,
    "LUT-REFINE": 300, "TERMINAL-RENDERER FIX": 300, "COMPRESSED SCHEDULE": 300,
    "CROP-NATIVE (certified)": 450, "INIT-PACK": 450,
    "PLANE-KILL + u8 RELOWER": 700,
    "MORPHOLOGY ROUND-CAP": None,          # None → pin×0.5 (uncompiled flood class)
}
_DEFAULT_TARGET = 400


def target_cost(fam, pc):
    t = _FAMILY_TARGET.get(fam, _DEFAULT_TARGET)
    if t is None:
        t = max(1, int((pc or 2 * _DEFAULT_TARGET) * 0.5))
    return t


def expected_value(tier, pp, sim, npv, nos, floored, natt=0, banked=0.0,
                   pc=None, fam="?"):
    if floored or tier == "SKIP-FLOORED":
        return 0.0, 0.0, 0.0, target_cost(fam, pc)
    tgt = target_cost(fam, pc)
    # remaining log-gain: a banked Δ (= ln(old/new) by the score law) comes straight off
    # the ratio — a +0.9-banked task must not front-run a virgin one (max-ROI kept).
    # Tasks already at/below their class target have nothing left and sink to EV 0.
    ln_gain = 0.0
    if pc:
        ln_gain = max(0.0, math.log(max(1.0, pc / max(1, tgt))) - max(0.0, banked))
    # Textual donor similarity had no measured lift (8.7% attempt conversion both above
    # 0.7 and below 0.3), so it must not boost queue EV. Multiple exact proven designs get
    # only a small, shrunk transfer prior until task-stratified evidence supports more.
    pwin = _PWIN[tier] * (1 + 0.1 * (npv >= 2)) * max(0.3, 1 - 0.12 * nos)
    # measured conversion decays with attempt depth (att2-3 ~27% → att5+ <9%): the NEXT
    # attempt is natt+1, so decay starts once 2 real attempts exist. NOs decay on top —
    # deliberate: a NO is stronger negative evidence than a mere unfinished failure.
    pwin *= max(0.3, 1 - 0.22 * max(0, natt - 2))
    if banked > 0:
        # house-money: only a ≥2×-cheaper family still banks points — a strictly harder
        # win condition than the first bank; big-swing conversions are real but rarer.
        pwin *= 0.6
    ev = ln_gain * pwin                     # expected banked points (queue sort key)
    roi = ev / _COST[tier]                  # points per token-unit (diagnostic)
    return round(ev, 4), round(roi, 4), round(ln_gain, 3), tgt


def main():
    strat, donors, proven_n, pin = load()
    q = list(csv.DictReader(open(f"{ROOT}/data/lanes/QUEUE.csv")))
    rows = []
    banked = banked_best()
    for r in q:
        t = r["task"][4:]
        total, tier, c, fam, pp, sim, npv, nos, floored, cut, natt = score_task(
            t, strat, donors, proven_n, pin, r)
        pc = pin.get(t, (None, None))[0]
        ev, roi, edelta, tgt = expected_value(tier, pp, sim, npv, nos, floored,
                                              natt=natt, banked=banked.get(t, 0.0),
                                              pc=pc, fam=fam)
        rows.append({"task": r["task"], "lane": r["lane"], "score": total, "tier": tier,
                     "ev": ev, "roi": roi, "edelta": edelta, "target_cost": tgt,
                     "pin_pts": f"{pp:.2f}" if pp else "", "fam": fam, "nos": nos,
                     "natt": natt, "banked": f"{banked.get(t, 0.0):.3f}",
                     "compression": f"{cut:.3f}", "donor": f"{sim:.2f}",
                     "proven": npv, **{f"c_{k}": v for k, v in c.items()}})
    rows.sort(key=lambda x: -x["ev"])   # EXPECTED-VALUE order (front-load the highest banked-points tasks)
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    out = f"{ROOT}/results/DIFFICULTY.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    import collections
    tc = collections.Counter(x["tier"] for x in rows)
    print(f"wrote {out}: {len(rows)} tasks (EV-ordered)")
    print("tiers:", dict(tc))
    active = [r for r in rows if r["tier"] != "SKIP-FLOORED"]
    print(f"Σ expected banked points (all active) = {sum(r['ev'] for r in active):.2f}")
    print("\nTOP-15 by EXPECTED VALUE (queue front — P(win)·ln(pin/target); router picks the model tier):")
    for x in rows[:15]:
        print(f"  {x['task']} {x['tier']:6} EV={x['ev']:.3f} lnGain={x['edelta']:.2f} "
              f"tgt={x['target_cost']:>4} pts={x['pin_pts']:>5} fam={x['fam'][:20]:20} "
              f"nos={x['nos']} proven={x['proven']}")


if __name__ == "__main__":
    main()
