#!/usr/bin/env python3
"""task132: W=15 uint8 moment span + Einsum as free output (Pad terms only).

True rule: same-color diagonal corner pairs define rectangles; fill with that color.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "candidates/task132_w15u8.onnx"
W = 15


def init(name: str, arr) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr), name=name)


def build() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    idx15 = np.arange(W, dtype=np.float32)
    inits = [
        init("idx15", idx15),
        init("idx_sq15", idx15 * idx15),
        init("row_twice15", (2 * idx15).astype(np.float16).reshape(1, 1, W, 1)),
        init("col_twice15", (2 * idx15).astype(np.float16).reshape(1, 1, 1, W)),
        init("pads_row", np.array([0, 0, 0, 0, 0, 0, 30 - W, 0], np.int64)),
        init("pads_col", np.array([0, 0, 0, 0, 0, 0, 0, 30 - W], np.int64)),
        init("palette_u8", np.arange(10, dtype=np.uint8).reshape(1, 1, 10)),
        init(
            "bg_vec_f16",
            np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float16).reshape(1, 1, 10),
        ),
        init("fg0", np.array([1], np.int64)),
        init("fg9", np.array([10], np.int64)),
        init("axis_c", np.array([1], np.int64)),
        init("one_i64", np.array(1, np.int64)),
        init("pair4_shape", np.array([1, 2, 1, 1], np.int64)),
        init("color_shape", np.array([1, 2, 1], np.int64)),
        init("one_f32", np.array(1.0, np.float32)),
        init("zero_f32", np.array(0.0, np.float32)),
        init("two_f16", np.array(2.0, np.float16)),
        init("zero_bool", np.array(False, np.bool_)),
        init("w0", np.array([0], np.int64)),
        init("w15", np.array([W], np.int64)),
        init("ax_row", np.array([2], np.int64)),
        init("ax_col", np.array([3], np.int64)),
    ]

    def n(op: str, ins: list[str], outs: list[str], **kw) -> None:
        nodes.append(helper.make_node(op, ins, outs, **kw))

    n("ReduceMax", ["input"], ["row_any30"], axes=[1, 3], keepdims=1)
    n("Slice", ["row_any30", "w0", "w15", "ax_row"], ["row_any"])
    n("Greater", ["row_any", "zero_f32"], ["row_valid"])
    n("ReduceMax", ["input"], ["col_any30"], axes=[1, 2], keepdims=1)
    n("Slice", ["col_any30", "w0", "w15", "ax_col"], ["col_any"])
    n("Greater", ["col_any", "zero_f32"], ["col_valid"])

    n("Einsum", ["input", "idx15"], ["row_sum"], equation="nchw,h->nc")
    n("Einsum", ["input", "idx_sq15"], ["row_sq_sum"], equation="nchw,h->nc")
    n("Einsum", ["input", "idx15"], ["col_sum"], equation="nchw,w->nc")
    n("Einsum", ["input", "idx_sq15"], ["col_sq_sum"], equation="nchw,w->nc")

    n("Slice", ["row_sum", "fg0", "fg9", "axis_c"], ["fg_present_raw"])
    n("Min", ["fg_present_raw", "one_f32"], ["fg_present_f"])
    n("ArgMax", ["fg_present_f"], ["fg_first0"], axis=1, keepdims=1, select_last_index=0)
    n("ArgMax", ["fg_present_f"], ["fg_last0"], axis=1, keepdims=1, select_last_index=1)
    n("Add", ["fg_first0", "one_i64"], ["fg_first"])
    n("Add", ["fg_last0", "one_i64"], ["fg_last"])
    n("Concat", ["fg_first", "fg_last"], ["fg_pair4"], axis=1)
    for base in ("row_sum", "row_sq_sum", "col_sum", "col_sq_sum"):
        n("GatherElements", [base, "fg_pair4"], [f"{base}_sel"], axis=1)
        n("Cast", [f"{base}_sel"], [f"{base}_sel_f16"], to=TensorProto.FLOAT16)

    def span(prefix: str, twice15: str, rects: str) -> None:
        n("Mul", [f"{prefix}_sum_sel_f16", f"{prefix}_sum_sel_f16"], [f"{prefix}_s2"])
        n("Mul", [f"{prefix}_sq_sum_sel_f16", "two_f16"], [f"{prefix}_q2"])
        n("Sub", [f"{prefix}_q2", f"{prefix}_s2"], [f"{prefix}_d2"])
        n("Reshape", [f"{prefix}_sum_sel_f16", "pair4_shape"], [f"{prefix}_s4"])
        n("Reshape", [f"{prefix}_d2", "pair4_shape"], [f"{prefix}_d24"])
        n("Sub", [twice15, f"{prefix}_s4"], [f"{prefix}_diff15"])
        n("Mul", [f"{prefix}_diff15", f"{prefix}_diff15"], [f"{prefix}_diff215"])
        n("LessOrEqual", [f"{prefix}_diff215", f"{prefix}_d24"], [rects])

    span("row", "row_twice15", "row_rects15")
    span("col", "col_twice15", "col_rects15")

    n("Pad", ["row_valid", "pads_row", "zero_bool"], ["row_valid30"], mode="constant")
    n("Pad", ["col_valid", "pads_col", "zero_bool"], ["col_valid30"], mode="constant")
    n("Pad", ["row_rects15", "pads_row", "zero_bool"], ["row_rects"], mode="constant")
    n("Pad", ["col_rects15", "pads_col", "zero_bool"], ["col_rects"], mode="constant")

    n("Concat", ["row_valid30", "row_rects"], ["row_terms_b"], axis=1)
    n("Concat", ["col_valid30", "col_rects"], ["col_terms_b"], axis=1)
    n("Cast", ["row_terms_b"], ["row_terms"], to=TensorProto.FLOAT16)
    n("Cast", ["col_terms_b"], ["col_terms"], to=TensorProto.FLOAT16)

    n("Cast", ["fg_pair4"], ["fg_pair_u8"], to=TensorProto.UINT8)
    n("Reshape", ["fg_pair_u8", "color_shape"], ["fg_pair_color"])
    n("Equal", ["fg_pair_color", "palette_u8"], ["color_b"])
    n("Cast", ["color_b"], ["color_f16"], to=TensorProto.FLOAT16)
    n("Sub", ["color_f16", "bg_vec_f16"], ["color_delta"])
    n("Concat", ["bg_vec_f16", "color_delta"], ["color_terms"], axis=1)

    n("Einsum", ["row_terms", "col_terms", "color_terms"], ["output"], equation="ntro,ntoc,ntk->nkrc")

    graph = helper.make_graph(
        nodes,
        "task132_w15u8",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 14)])
    model.ir_version = 10
    del model.graph.value_info[:]
    return model


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    model = build()
    onnx.checker.check_model(model)
    onnx.save(model, OUT)
    print(OUT)