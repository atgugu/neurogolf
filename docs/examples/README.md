# A real task, end to end: task014

Frozen artifacts from a real wave (paths and historical fleet labels neutralized).
task014 is a *crop-select minform*: find the rarest color's bounding box, crop, recolor.

## `pack_task014/` — what the builder received

- **`ATTACK.md`** — the complete generated prompt pack: tailored strategy card, per-tensor
  cost anatomy of the incumbent graph, source certificate, Hodel rule verdict, donor
  designs, past fleet record, and the exact register-cost target. This is the heart of the
  system: everything a builder needs to attack one task, in one file.
- **`PIN_PSEUDOCODE.md`** — advisory disassembly of the incumbent graph.
- **`GRAPH_CONTEXT.md`** — the Kaggle-proven reference member staged for this task.
- **`RULE_SENTENCE.md`**, **`rule.py`** — the decoded rule, as prose and as the numpy
  function that must pass `hypo.py` on 100% of visible examples before any ONNX.

## `results_task014/` — what happened

- **`ATTEMPTS.jsonl`** — the attempt record: 8 failures and timeouts across model
  configurations… then attempt 9 lands **+0.89** (a 2.4× cheaper graph) in 403 seconds.
- **`POSTMORTEM_att09.md`** — the analyst's transfer debrief of the win.
- **`DECISION_att09.md`** — the decider's binding strategy decision that set it up.
- **`build.py`** — the winning builder script.
- **`NOTES.md`** — the worker's own working notes.
