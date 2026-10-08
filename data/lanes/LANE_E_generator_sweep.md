# LANE E — Generator-source sweep (codex-spark · run FIRST · pure extraction)

**Purpose:** multiply every other lane. You read generator SOURCE files and emit
structured facts. You do NOT build ONNX. One task per call, high parallelism.

**Input per task:** `neurogolf_clean/external/ARC-GEN/tasks/task_<hex>.py`
(hex from `lanes/QUEUE.csv` or `data/task_map.json`). Usually < 120 lines of Python.

**Emit exactly this JSON to `./taskNNN.json` (your cwd IS `results/sweep/` — do not create
nested dirs). Builders' packs pick it up automatically on their next rebuild:**

```json
{
  "task": "taskNNN",
  "grid_bounds": {"in_h": [min,max], "in_w": [min,max], "out_h": [min,max], "out_w": [min,max], "how_proven": "quote the source lines"},
  "palette": {"in": [...], "out": [...], "fixed": true},
  "discrete_params": [{"name": "...", "range": "...", "count": N}],
  "output_space": {"enumerable": true, "size_estimate": N, "argument": "why — e.g. output depends only on (color, count) with ≤ 40 combos"},
  "invariants": ["every output row r equals input row r+1 shifted …", "…"],
  "traps": ["parameter X is filtered when …", "ambiguity: two valid outputs when …"],
  "cheap_form_hint": "one sentence: the cheapest ONNX shape of this rule you can see"
}
```

**Rules:** every claim must be provable from the SOURCE (quote line numbers). If you
cannot prove a bound, write `null` — a wrong certificate is worse than none (sample-
inferred invariants are the historical tank class). `enumerable: true` requires the whole
output to be a function of a small discrete parameter tuple — if true, the task is a
**Gather-LUT candidate** (flag it; a builder agent will take it: LUT initializer of ≤ ~40
uint8 output templates + a small computed key + terminal Gather ⇒ near-19 scores).

**Coverage:** all Lane A, B, C tasks (QUEUE.csv order). ~150 files, this is one night of
cheap calls. Builders consume your JSON as the "certificate" section of their packs.
