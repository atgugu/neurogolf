# LANE H — the reopened ≥18.6 band (124 tasks · bar +0.15 · cheap models, best-of-N)

**Why this lane exists (2026-07-12, study7300/STRATEGY_8000 M1):** the old queue dropped
every task scoring ≥18.6 — 41% of the board. The verified study corpus shows that band
converted *at least as well* as the queued one (52 verified wins / +22.8 pts landed
there), with the **cheapest attempts on the board**: tiny graphs (600–1100 B), fast
gates, minutes per attempt. Σ headroom to the 8000 goal line (20.4 pts/task ⇒ cost
≤ ~100) lives disproportionately here.

## Discipline (Lane-A style, ratio-first)
- **Bar +0.15** (registration floor only — `submit_result --lane H` applies it). The WIN
  condition is the task's rule-class compile target printed in the pack ("rule-class
  compile target: cost ≤ N").
- **RATIO LAW:** these pins are already smallish, so percentage cuts are what pay:
  −50% = +0.69 · −25% = +0.29 · −10% = +0.11. Micro-trims below −5% are noise — don't
  register them unless they are value-exact byte-identical rewrites (those compound).
- **Best-of-N cheap attempts** beat one expensive attempt here. The band is measured-
  dense with 1.2–6× factorizations (study7300 §1–2): recolor/count/parity rules compile
  at 30–150 els; most of this band still spends 400–1100.
- **COMPILE FIRST:** the pack's "COMPILE THE RULE" section is the default family. A
  correct build that misses the class target is a TICKET (exit 3), never a NO — its
  graph becomes an Engine-A donor.
- Value-exact mechanical tickets for many of these tasks already exist in
  `../tickets/TICKETS.csv` (factor_sweep) — check for your task before designing:
  a CONV_SHRINK/FACTOR_PAIR ticket is a free head start.

## Worker protocol
Identical to every other lane: claim → read `packs/taskNNN/ATTACK.md` →
`fast_verify.py NNN cand.onnx` loop → `submit_result.py NNN cand.onnx --lane H`.
