#!/usr/bin/env python3
"""price — score any ONNX graph with the competition's own grader, standalone.

    python3 runner/price.py <model.onnx> [task.json]

Runs the vendored Kaggle scorer (runner/grader/, the IJCAI-ECAI 2026 NeuroGolf framework
verbatim) and prints the compression score. No NEUROGOLF_CLEAN, no competition data — a
synthetic demo grid (runner/grader/demo_task.json) drives the forward pass, because the
score is params + memory of the graph, independent of correctness.

    score = 25 − ln(params + memory)

Needs only numpy + onnx + onnxruntime (pip install -r requirements.txt). onnxruntime==1.24.4
gives exact Kaggle parity — the memory measurement reads that runtime's profiler.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GRADER = HERE.parent / "runner" / "grader"
sys.path.insert(0, str(GRADER))

from neurogolf.scoring_v2 import points_from_cost_v2, score_v2  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    model = Path(sys.argv[1])
    task_json = Path(sys.argv[2]) if len(sys.argv) > 2 else GRADER / "demo_task.json"
    if not model.is_file():
        sys.exit(f"no such file: {model}")

    cost, _val = score_v2(model, task_json, task_num=0, timeout_sec=120)
    if not cost.ok:
        err = (cost.profile_error or cost.onnx_check_error
               or ("disallowed ops: " + ", ".join(cost.disallowed_ops)
                   if cost.disallowed_ops else "unscorable"))
        print(f"UNSCORABLE: {err}")
        return 1

    pts = points_from_cost_v2(cost.memory_bytes, cost.params)
    print(f"model    {model.name}")
    print(f"params   {cost.params:>10,}   (initializer + Constant elements)")
    print(f"memory   {cost.memory_bytes:>10,}   bytes (node-output tensors)")
    print(f"cost     {cost.cost:>10,}   = params + memory")
    print(f"points   {pts:>10.4f}   = 25 − ln(cost)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
