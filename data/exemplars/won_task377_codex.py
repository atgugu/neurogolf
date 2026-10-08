#!/usr/bin/env python3
"""Build task377.onnx.

Rule: nested filled rectangles become a compact top-left square of concentric
one-cell rings, preserving the input colors from outside to inside.

The requested 10x10 transition-bitmap extractor is not source-valid: the
generator may place a child rectangle start after row/column 10. This build
keeps the verified row-moment extractor and uses the cheaper factored terminal
from the audited recipe.

Priced before build:

Design A, slim 10x10 transition bitmap + factored terminal:
  vertical transitions          bool [1,1,9,10]              90 B
  horizontal transitions        bool [1,1,10,9]              90 B
  row/col edge vectors          2 x uint8 [1,1,9]            18 B
  slots/depth                   uint8 [1,1,6], [1,1,1]        7 B
  color eq/coefs                bool [1,10,6] + 2xf16       300 B
  dynamic upper profile         bool/f16 [1,6,30]           540 B
  lower profile params          f16 [6,30]                  180 p
  other params                  mixed                       ~967 p
  paper total                                               ~2198
  Verdict: physically invalid on raw generator grids; child offsets can exceed
  the first 10 rows/columns.

Design B, source-valid row moments + static-lower factored terminal:
  row moments                   2 x f32 [30]                240 B
  f16 row slices/diffs          mixed [30]/[29]             468 B
  transition rank/depth         bool/int32/TopK/gathers     307 B
  scalar/vector color slots     f16/u8 small                 60 B
  color eq/coefs                bool [1,10,6] + 2xf16       300 B
  dynamic upper profile         bool/f16 [1,6,30]           540 B
  lower profile params          f16 [6,30]                  180 p
  other params                  mixed                       ~116 p
  paper total                                               ~2211
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper as nh


N = 30
C = 10
DEPTH = 6
OUT = Path(__file__).with_name("task377.onnx")


def init(name: str, arr) -> onnx.TensorProto:
    return nh.from_array(np.asarray(arr), name)


def build() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    def add(name: str, arr) -> str:
        inits.append(init(name, arr))
        return name

    add("coef1", np.arange(C, dtype=np.float32))
    add("coef2", (np.arange(C, dtype=np.float32) ** 2))
    add("half", np.array(0.5, dtype=np.float16))
    add("two_i32", np.array(2, dtype=np.int32))
    add("color_ids_u8", np.arange(C, dtype=np.uint8).reshape(1, C, 1))

    diff = np.zeros((DEPTH, DEPTH), dtype=np.float16)
    for k in range(DEPTH):
        diff[k, k] = 1
        if k + 1 < DEPTH:
            diff[k, k + 1] = -1
    add("diff_f16", diff)

    lower = np.zeros((DEPTH, N), dtype=np.float16)
    for k in range(DEPTH):
        lower[k, k:] = 1
    add("lower_f16", lower)

    add("ring_ids_i8", np.arange(DEPTH, dtype=np.int8).reshape(1, DEPTH, 1))
    add("coord_i8", np.arange(N, dtype=np.int8).reshape(1, 1, N))
    add("axes0", np.array([0], dtype=np.int64))
    add("topk", np.array([5], dtype=np.int64))
    for i in range(0, 6):
        add(f"s{i}", np.array([i], dtype=np.int64))
    add("s29", np.array([N - 1], dtype=np.int64))
    add("s30", np.array([N], dtype=np.int64))
    add("ax0", np.array([0], dtype=np.int64))
    add("step1", np.array([1], dtype=np.int64))

    nodes.append(h.make_node("Einsum", ["input", "coef1"], ["m1"], equation="nchw,c->h", name="m1"))
    nodes.append(h.make_node("Einsum", ["input", "coef2"], ["m2"], equation="nchw,c->h", name="m2"))
    nodes.append(h.make_node("Cast", ["m1"], ["m1h"], to=T.FLOAT16, name="m1h"))
    nodes.append(h.make_node("Cast", ["m2"], ["m2h"], to=T.FLOAT16, name="m2h"))

    nodes.append(h.make_node("Slice", ["m1h", "s1", "s30", "ax0", "step1"], ["m1_hi"], name="m1_hi"))
    nodes.append(h.make_node("Slice", ["m1h", "s0", "s29", "ax0", "step1"], ["m1_lo"], name="m1_lo"))
    nodes.append(h.make_node("Sub", ["m1_hi", "m1_lo"], ["d1"], name="d1"))
    nodes.append(h.make_node("Slice", ["m2h", "s1", "s30", "ax0", "step1"], ["m2_hi"], name="m2_hi"))
    nodes.append(h.make_node("Slice", ["m2h", "s0", "s29", "ax0", "step1"], ["m2_lo"], name="m2_lo"))
    nodes.append(h.make_node("Sub", ["m2_hi", "m2_lo"], ["d2"], name="d2"))

    nodes.append(h.make_node("Cast", ["d1"], ["trans"], to=T.BOOL, name="trans"))
    nodes.append(h.make_node("Cast", ["trans"], ["trans_i"], to=T.INT32, name="trans_i"))
    nodes.append(h.make_node("ReduceSum", ["trans_i", "axes0"], ["nchange"], keepdims=1, name="nchange"))
    nodes.append(h.make_node("Div", ["nchange", "two_i32"], ["k1"], name="k1"))
    nodes.append(
        h.make_node("TopK", ["trans_i", "topk"], ["top_vals", "top_idx"], axis=0, largest=1, sorted=1, name="top_idx")
    )
    nodes.append(h.make_node("Gather", ["d1", "top_idx"], ["d1_top"], axis=0, name="d1_top"))
    nodes.append(h.make_node("Gather", ["d2", "top_idx"], ["d2_top"], axis=0, name="d2_top"))

    nodes.append(h.make_node("Slice", ["m1h", "s0", "s1", "ax0", "step1"], ["m1_0"], name="m1_0"))
    nodes.append(h.make_node("Slice", ["m2h", "s0", "s1", "ax0", "step1"], ["m2_0"], name="m2_0"))
    nodes.append(h.make_node("Div", ["m2_0", "m1_0"], ["outer_f"], name="outer_f"))
    nodes.append(h.make_node("Add", ["outer_f", "half"], ["outer_r"], name="outer_r"))
    nodes.append(h.make_node("Cast", ["outer_r"], ["outer"], to=T.UINT8, name="outer_u8"))
    nodes.append(h.make_node("Div", ["d2_top", "d1_top"], ["sum_top"], name="sum_top"))

    child_names: list[str] = []
    parent = "outer_f"
    for i in range(5):
        nodes.append(h.make_node("Slice", ["sum_top", f"s{i}", f"s{i+1}", "ax0", "step1"], [f"sum{i+1}_f"], name=f"sum{i+1}_f"))
        nodes.append(h.make_node("Sub", [f"sum{i+1}_f", parent], [f"child{i+1}_raw"], name=f"child{i+1}_raw"))
        nodes.append(h.make_node("Add", [f"child{i+1}_raw", "half"], [f"child{i+1}_r"], name=f"child{i+1}_r"))
        nodes.append(h.make_node("Cast", [f"child{i+1}_r"], [f"child{i+1}_u8"], to=T.UINT8, name=f"child{i+1}_u8"))
        child_names.append(f"child{i+1}_u8")
        parent = f"child{i+1}_raw"

    nodes.append(h.make_node("Concat", ["outer", *child_names], ["slot_color_u8"], axis=0, name="slot_color_u8"))
    nodes.append(h.make_node("Equal", ["slot_color_u8", "color_ids_u8"], ["eq_b"], name="eq_b"))
    nodes.append(h.make_node("Cast", ["eq_b"], ["eq_f16"], to=T.FLOAT16, name="eq_f16"))
    nodes.append(h.make_node("Einsum", ["eq_f16", "diff_f16"], ["coef_f16"], equation="bck,kl->bcl", name="coef_f16"))

    nodes.append(h.make_node("Mul", ["k1", "two_i32"], ["maxcoord_i32"], name="maxcoord_i32"))
    nodes.append(h.make_node("Cast", ["maxcoord_i32"], ["maxcoord_i8"], to=T.INT8, name="maxcoord_i8"))
    nodes.append(h.make_node("Sub", ["maxcoord_i8", "ring_ids_i8"], ["limit_i8"], name="limit_i8"))
    nodes.append(h.make_node("LessOrEqual", ["coord_i8", "limit_i8"], ["upper_b"], name="upper_b"))
    nodes.append(h.make_node("Cast", ["upper_b"], ["upper_f16"], to=T.FLOAT16, name="upper_f16"))
    nodes.append(
        h.make_node(
            "Einsum",
            ["coef_f16", "lower_f16", "upper_f16", "lower_f16", "upper_f16"],
            ["output"],
            equation="bck,ky,bky,kx,bkx->bcyx",
            name="output",
        )
    )

    graph = h.make_graph(
        nodes,
        "task377_static_lower_factored_box",
        [h.make_tensor_value_info("input", T.FLOAT, [1, C, N, N])],
        [h.make_tensor_value_info("output", T.FLOAT16, [1, C, N, N])],
        inits,
    )
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 14)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return onnx.shape_inference.infer_shapes(model, strict_mode=True)


def main() -> None:
    onnx.save(build(), OUT)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
