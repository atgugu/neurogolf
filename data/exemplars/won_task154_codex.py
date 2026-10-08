import os
#!/usr/bin/env python3
"""Task 154: exact crop-native narrow QLinearConv build.

The trusted donor already computes a 10x10 class crop.  Its quadratic feature tail is
replaced by two saturated quantized features.  A direct scalar tail is exhaustively
rejected: three ordered codes cannot all be isolated by affine positive half-lines.
Uint8 saturation bends the scalar into a two-dimensional code that also distinguishes
the 15x15 background border from the all-zero 30x30 transport exterior.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parent
DONOR = ROOT / "refs" / "kaggle_proven_best.onnx"
EXAMPLES = Path((os.environ.get("NEUROGOLF_CLEAN", "") + "/extracted/task154.json"))

# The requested helper is absent from this checkout's stripped runner module.  Keep the
# same call surface and fall back to ONNX's constructor, which emits identical nodes.
sys.path.insert(0, "../../runner")
import ngolf  # noqa: E402

make_node = getattr(ngolf, "make_node", helper.make_node)


def arr(name: str, value, dtype=None):
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name)


def qcode(value: float, scale: float, zp: int = 0) -> int:
    """QuantizeLinear scalar semantics for the exact integer ratios searched here."""
    return int(np.clip(np.rint(value / scale) + zp, 0, 255))


def search_scalar_constants():
    """Exhaust the deliberately tiny quantizer and affine integer vocabulary.

    The Max assembly makes background the lowest in-crop code.  A cumulative pair of
    logits is therefore used: red's line is positive at red and gray, while gray's line
    overtakes it only at gray.  Channel 0 remains zero, so padding/background argmax to 0.
    """
    solutions = []
    for bg in range(1, 5):
        for red_den in range(1, 9):
            red_scale = 1.0 / red_den
            red = max(bg, qcode(1.0, red_scale))
            for gray_den in range(1, 9):
                gray_scale = 1.0 / gray_den
                gray = qcode(1.0, gray_scale, bg)
                if not (0 < bg < red < gray):
                    continue
                xs = np.array([0, bg, red, gray], dtype=np.int32)
                for wr in range(1, 9):
                    for br in range(-8, 1):
                        yr = np.maximum(0, wr * xs + br)
                        if yr[0] or yr[1] or yr[2] <= 0:
                            continue
                        for wg in range(1, 9):
                            for bg_bias in range(-16, 1):
                                yg = np.maximum(0, wg * xs + bg_bias)
                                if yg[0] or yg[1]:
                                    continue
                                if not (yr[2] > yg[2] and yg[3] > yr[3]):
                                    continue
                                solutions.append((
                                    max(abs(wr), abs(br), abs(wg), abs(bg_bias)),
                                    red_den + gray_den + bg,
                                    bg, red_den, gray_den, wr, br, wg, bg_bias,
                                ))
    if not solutions:
        raise RuntimeError("no scalar QLinearConv code solution in exhaustive vocabulary")

    # Prefer the donor's upstream codes (bg/red/gray = 1/2/4), then small coefficients.
    donor = [s for s in solutions if s[2:5] == (1, 2, 3)]
    pool = donor or solutions
    best = min(pool)
    _, _, bg, red_den, gray_den, wr, br, wg, bg_bias = best
    codes = (0, bg, max(bg, red_den), bg + gray_den)
    logits = np.zeros((4, 10), dtype=np.int32)
    logits[:, 2] = np.maximum(0, wr * np.asarray(codes) + br)
    logits[:, 5] = np.maximum(0, wg * np.asarray(codes) + bg_bias)
    decoded = logits.argmax(axis=1).tolist()
    if decoded != [0, 0, 2, 5]:
        raise AssertionError((codes, logits.tolist(), decoded))
    return {
        "bg": bg,
        "red_scale": np.float32(1.0 / red_den),
        "gray_scale": np.float32(1.0 / gray_den),
        "weights": (wr, wg),
        "biases": (br, bg_bias),
        "codes": codes,
        "logits": logits,
        "solution_count": len(solutions),
    }


def prove_direct_scalar_impossible(codes) -> None:
    """Exhaust the small affine vocabulary under the grader's raw>0 semantics."""
    xs = np.asarray(codes, dtype=np.int32)
    targets = np.asarray([0, 0, 2, 5], dtype=np.int32)
    candidates = []
    for w in range(-16, 17):
        for b in range(-32, 33):
            candidates.append(np.maximum(0, w * xs + b) > 0)
    for cls in (0, 2, 5):
        want = targets == cls
        if any(np.array_equal(got, want) for got in candidates):
            continue
        if cls != 2:
            raise AssertionError(f"unexpected scalar failure for endpoint class {cls}")
        print("direct scalar proof: red is the ordered middle code; no affine positive half-line isolates it")
        return
    raise AssertionError("direct scalar unexpectedly separates all three classes")


def scalar_q(path: Path) -> dict:
    cfg = search_scalar_constants()
    prove_direct_scalar_impossible(cfg["codes"])
    m = onnx.load(DONOR)
    g = m.graph

    # Preserve the donor exactly through class_crop; replace only the quadratic tail.
    kept = []
    for n in g.node:
        kept.append(n)
        if list(n.output) == ["class_crop"]:
            break
    del g.node[:]
    g.node.extend(kept)

    vals = {x.name: x for x in g.initializer}
    vals["red_q_scale"].CopyFrom(arr("red_q_scale", cfg["red_scale"], np.float32))
    vals["gray_q_scale"].CopyFrom(arr("gray_q_scale", cfg["gray_scale"], np.float32))
    vals["bg_u8"].CopyFrom(arr("bg_u8", cfg["bg"], np.uint8))

    # Saturated features for x=(0 border,1 bg,2 red,4 gray):
    #   f1=clip(120*x-20), f2=clip(-120*x+360)
    #   -> (0,255), (100,240), (220,120), (255,0).
    hinge_w = np.asarray([120, -120], dtype=np.int8).reshape(2, 1, 1, 1)
    hinge_b = np.asarray([-20, 360], dtype=np.int32)
    w = np.zeros((10, 2, 1, 1), dtype=np.int8)
    b = np.full((10,), -1, dtype=np.int32)
    # Background: f2-180.  Red: 2*f1+f2-525.  Gray: f1-f2-150.
    w[0, :, 0, 0] = [0, 1]
    w[2, :, 0, 0] = [2, 1]
    w[5, :, 0, 0] = [1, -1]
    b[0], b[2], b[5] = -180, -525, -150

    features = np.stack([
        np.clip(120 * np.asarray(cfg["codes"]) - 20, 0, 255),
        np.clip(-120 * np.asarray(cfg["codes"]) + 360, 0, 255),
    ], axis=1)
    raw = features @ w[:, :, 0, 0].T + b
    decoded = raw > 0
    expected = np.zeros((4, 10), dtype=bool)
    expected[0, 0] = expected[1, 0] = True
    expected[2, 2] = expected[3, 5] = True
    if not np.array_equal(decoded, expected):
        raise AssertionError((features.tolist(), raw.tolist()))

    keep_init = {
        name for n in g.node for name in n.input
    }
    del g.initializer[:]
    for name, t in vals.items():
        if name in keep_init:
            g.initializer.append(t)
    g.initializer.extend([
        arr("x_scale", np.float32(1.0), np.float32),
        arr("x_zp", np.uint8(0), np.uint8),
        arr("hinge_w", hinge_w),
        arr("hinge_b", hinge_b),
        arr("ql_w", w),
        arr("w_scale", np.float32(1.0), np.float32),
        arr("w_zp", np.int8(0), np.int8),
        arr("y_scale", np.float32(1.0), np.float32),
        arr("y_zp", np.uint8(0), np.uint8),
        arr("ql_b", b),
    ])
    g.node.append(make_node(
        "QLinearConv",
        ["class_crop", "x_scale", "x_zp", "hinge_w", "w_scale", "w_zp",
         "y_scale", "y_zp", "hinge_b"],
        ["class_hinge"],
        pads=[2, 2, 3, 3],
    ))
    # class_hinge is now the complete 15x15 logical grid.  Terminal padding is all-zero,
    # matching the trusted donor's transport exterior exactly.
    g.node.append(make_node(
        "QLinearConv",
        ["class_hinge", "x_scale", "x_zp", "ql_w", "w_scale", "w_zp",
         "y_scale", "y_zp", "ql_b"],
        ["output"],
        pads=[0, 0, 15, 15],
    ))
    del g.value_info[:]
    onnx.checker.check_model(m, full_check=True)
    m = onnx.shape_inference.infer_shapes(m, strict_mode=True)
    inferred = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
                for v in list(m.graph.value_info) + list(m.graph.output)}
    if inferred.get("output") != [1, 10, 30, 30]:
        raise AssertionError(inferred.get("output"))
    onnx.save(m, path)
    return cfg


def encode(grid) -> np.ndarray:
    a = np.asarray(grid, dtype=np.int64)
    x = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            x[0, int(a[r, c]), r, c] = 1.0
    return x


def iter_grids(obj):
    if isinstance(obj, dict):
        if "input" in obj and isinstance(obj["input"], list):
            yield obj["input"]
        for value in obj.values():
            yield from iter_grids(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from iter_grids(value)


def diff_donor(candidate: Path) -> None:
    if not EXAMPLES.exists():
        print(f"donor differential skipped: {EXAMPLES} is absent")
        return
    data = json.loads(EXAMPLES.read_text())
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    ref = ort.InferenceSession(str(DONOR), so, providers=["CPUExecutionProvider"])
    got = ort.InferenceSession(str(candidate), so, providers=["CPUExecutionProvider"])
    count = 0
    for grid in iter_grids(data):
        x = encode(grid)
        a = ref.run(None, {"input": x})[0] > 0
        b = got.run(None, {"input": x})[0] > 0
        if not np.array_equal(a, b):
            where = np.argwhere(a != b)[0].tolist()
            raise AssertionError(f"donor differential failed at example {count}, index {where}")
        count += 1
    print(f"donor differential: {count}/{count} decoded examples equal")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scalar-q", action="store_true")
    ap.add_argument("--diff-donor", action="store_true")
    args = ap.parse_args()
    if not args.scalar_q:
        ap.error("this attempt only supports the binding --scalar-q build")
    path = ROOT / "task154.onnx"
    cfg = scalar_q(path)
    print(
        "scalar search:", cfg["solution_count"], "solutions;",
        "codes(pad,bg,red,gray)=", cfg["codes"],
        "weights=", cfg["weights"], "biases=", cfg["biases"],
    )
    if args.diff_donor:
        diff_donor(path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
