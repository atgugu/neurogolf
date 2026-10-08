"""May-2026 scorer using the post-May-4 / post-May-6 framework.

Replaces `neurogolf/scoring.py` for the new objective. Key differences:
  - `score_network(model, trace_path)` returns (memory, params), no MACs.
  - Memory is computed by ORT runtime profiling (not onnx_tool static).
  - Need to run all examples (train + test + arc-gen) to generate the profile.
  - Cost = max(1, memory + params); points = max(1, 25 - ln(cost)).

Architecture:
  - One subprocess per (onnx, task_data) pair.
  - Subprocess loads model + sanitizes node names + runs ORT session with
    profiling enabled, runs every example, then calls score_network.
  - Returns memory, params, per-split right/wrong counts, plus errors.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Optional

import onnx

from neurogolf.paths import (
    EXCLUDED_OPS,
    FILESIZE_LIMIT_BYTES,
    FRAMEWORK_FILE,
)
from neurogolf.report import CostReport, ScoreReport, ValidationReport


# Subprocess template: runs the May-2026 scoring pipeline on (onnx_path, task_data_path).
_BOOTSTRAP_V2 = r"""
import importlib.util, json, sys, traceback, io, contextlib, os, tempfile, glob, types
# The competition framework (neurogolf_utils) imports IPython / matplotlib / onnx_tool at
# module level (and binds display = IPython.display.display etc.), but the pricing path
# (check_network / score_network) never calls them. Stub with no-op attributes so pricing
# needs only numpy + onnx + onnxruntime.
def _noop(*a, **k):
    return None
class _StubMod(types.ModuleType):
    def __getattr__(self, name):
        return _noop
for _m in ("IPython", "IPython.display", "matplotlib", "matplotlib.pyplot", "onnx_tool"):
    sys.modules.setdefault(_m, _StubMod(_m))
sys.modules["IPython"].display = sys.modules["IPython.display"]
sys.modules["matplotlib"].pyplot = sys.modules["matplotlib.pyplot"]
spec = importlib.util.spec_from_file_location('neurogolf_utils', {framework!r})
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except Exception:
    print(json.dumps({{"error": "framework_load", "traceback": traceback.format_exc()}}))
    sys.exit(0)

import onnx, numpy as np, onnxruntime
buf = io.StringIO()
result = {{"memory": None, "params": None,
           "per_split": {{}}, "valid": False, "stdout": ""}}

def _convert_to_numpy(example):
    grid_in = mod._GRID_SHAPE
    grid_out = mod._GRID_SHAPE
    dtype_in = np.float32
    try:
        if candidate.graph.input[0].type.tensor_type.elem_type == 2:
            dtype_in = np.uint8
    except: pass
    inp_arr = np.zeros(grid_in, dtype=dtype_in)
    out_arr = np.zeros(grid_out, dtype=np.float32)
    inp = example["input"]; out = example["output"]
    H_in = len(inp); W_in = len(inp[0]) if H_in > 0 else 0
    H_out = len(out); W_out = len(out[0]) if H_out > 0 else 0
    if H_in > grid_in[2] or W_in > grid_in[3]:
        return None
    if H_out > grid_out[2] or W_out > grid_out[3]:
        return None
    for i in range(H_in):
        for j in range(W_in):
            v = inp[i][j]
            inp_arr[0, v, i, j] = 1
    for i in range(H_out):
        for j in range(W_out):
            v = out[i][j]
            out_arr[0, v, i, j] = 1.0
    return {{"input": inp_arr, "output": out_arr}}


try:
    with contextlib.redirect_stdout(buf):
        # Phase 1: load the candidate ONNX as written
        candidate = onnx.load({onnx_path!r})

        # Phase 2: write to a fresh on-disk filename matching task_num convention.
        # The new framework expects task num for trace prefix.
        task_num = {task_num}
        with tempfile.TemporaryDirectory() as tdir:
            os.chdir(tdir)
            fname = f"task{{task_num:03d}}.onnx"
            onnx.save(candidate, fname)

            # Phase 3: run check_network (file size + banned ops)
            ok = mod.check_network(fname)
            if not ok:
                result["error"] = "check_network_failed"
                raise SystemExit(0)

            # Phase 4: sanitize node names (set name = output[0])
            sanitized = onnx.load(fname)
            kernel_time_seen = False
            for node in sanitized.graph.node:
                node.name = node.output[0]
                if "kernel_time" in node.name:
                    kernel_time_seen = True
            if kernel_time_seen:
                result["error"] = "kernel_time_in_node_name"
                raise SystemExit(0)

            # Phase 5: build ORT session with profiling
            options = onnxruntime.SessionOptions()
            options.enable_profiling = True
            options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
            options.profile_file_prefix = f"{{task_num:03d}}"
            try:
                session = onnxruntime.InferenceSession(sanitized.SerializeToString(), options)
            except Exception as e:
                result["error"] = f"ort_init: {{type(e).__name__}}: {{str(e)[:200]}}"
                raise SystemExit(0)

            # Phase 6: load task data and run every example
            with open({task_data!r}, "r") as f:
                task = json.load(f)
            per_split = {{}}
            valid = True
            for split in ("train", "test", "arc-gen"):
                examples = task.get(split, [])
                right = wrong = 0
                for ex in examples:
                    bench = _convert_to_numpy(ex)
                    if bench is None:
                        wrong += 1
                        continue
                    try:
                        out = session.run(None, {{"input": bench["input"]}})
                    except Exception as e:
                        with open("/tmp/err.log", "a") as f: f.write(str(e) + "\n")
                        print(e, file=sys.stderr)
                        wrong += 1
                        continue
                    raw = out[0]
                    if not np.all(np.isfinite(raw)):
                        wrong += 1
                        continue
                    pred = (raw > 0).astype(np.float32)
                    if np.array_equal(pred, bench["output"]):
                        right += 1
                    else:
                        wrong += 1
                if wrong > 0:
                    valid = False
                per_split[split] = [right, wrong]
            result["per_split"] = per_split
            result["valid"] = valid

            # Phase 7: score
            trace_path = session.end_profiling()
            try:
                memory, params = mod.score_network(sanitized, trace_path)
            except Exception as e:
                result["error"] = f"score_network: {{type(e).__name__}}: {{str(e)[:200]}}"
                raise SystemExit(0)
            result["memory"] = memory
            result["params"] = params

except SystemExit:
    pass
except Exception:
    result["error"] = "exception"
    result["traceback"] = traceback.format_exc()

result["stdout"] = buf.getvalue()[-2000:]
print(json.dumps(result))
"""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _opset_version(model: onnx.ModelProto) -> Optional[int]:
    for imp in model.opset_import:
        if imp.domain in ("", "ai.onnx"):
            return imp.version
    return None


def points_from_cost_v2(memory: Optional[int], params: Optional[int]) -> float:
    """New formula: points = max(1, 25 - ln(max(1, memory + params)))."""
    if memory is None or params is None:
        return 0.0
    cost = max(1, memory + params)
    return max(1.0, 25.0 - math.log(cost))


def score_v2(
    onnx_path: Path,
    task_data_path: Path,
    *,
    task_num: int,
    timeout_sec: float = 60.0,
) -> tuple[CostReport, ValidationReport]:
    """Run the May-2026 scorer in a subprocess. Returns (cost_report, validation_report)."""
    path = Path(onnx_path)
    cost = CostReport()
    val = ValidationReport()

    if not path.is_file():
        cost.onnx_check_error = f"file not found: {path}"
        return cost, val
    size = path.stat().st_size
    cost.file_size_bytes = size
    if size > FILESIZE_LIMIT_BYTES:
        cost.onnx_check_error = f"filesize {size} > limit"
        return cost, val
    try:
        model = onnx.load(str(path))
    except Exception as e:
        cost.onnx_check_error = f"onnx.load: {e}"
        return cost, val
    cost.opset_version = _opset_version(model)
    found = []
    for n in model.graph.node:
        if n.op_type.upper() in EXCLUDED_OPS:
            found.append(n.op_type)
    if found:
        cost.disallowed_ops = found
        return cost, val

    bootstrap = _BOOTSTRAP_V2.format(
        framework=str(FRAMEWORK_FILE),
        onnx_path=str(path),
        task_data=str(task_data_path),
        task_num=task_num,
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", bootstrap],
            capture_output=True,
            timeout=timeout_sec,
            check=False,
            text=True,
        )
    except subprocess.TimeoutExpired:
        cost.profile_error = f"subprocess timeout {timeout_sec}s"
        return cost, val

    if proc.returncode != 0:
        cost.profile_error = f"rc={proc.returncode}; stderr={proc.stderr.strip()[:300]}"
        return cost, val

    try:
        last = proc.stdout.strip().splitlines()[-1]
        payload = json.loads(last)
    except Exception as e:
        cost.profile_error = f"json parse: {e}; stdout={proc.stdout[:300]}"
        return cost, val

    if "error" in payload:
        err = payload["error"]
        tb = payload.get("traceback", "") or ""
        cost.profile_error = f"{err}: {tb.splitlines()[-1] if tb else ''}"
        # may still have valid per_split
        per_split = payload.get("per_split", {})
        if per_split:
            val.per_split = {k: tuple(v) for k, v in per_split.items()}
            val.valid = bool(payload.get("valid", False))
        return cost, val

    cost.memory_bytes = payload.get("memory")
    cost.params = payload.get("params")
    cost.macs = 0  # back-compat field, no longer counted in cost
    val.per_split = {k: tuple(v) for k, v in payload.get("per_split", {}).items()}
    val.valid = bool(payload.get("valid", False))
    return cost, val
