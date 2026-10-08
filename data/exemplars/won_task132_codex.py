#!/usr/bin/env python3
"""task132 scalar-extent fp16 terminal contraction.

True rule: same-color corner pairs are opposite rectangle corners; fill each
rectangle with that color while preserving the native input extent.

Priced before build:
- W15 u8 spatial-code + final Equal channelizer: supplied paper budget 1461 B,
  measured 3269 B because ONNX charges all SSA-visible 15x15 scratch tensors.
- This pivot keeps the legal final fp16 Einsum renderer, but removes the
  row_any30/col_any30 validity reductions by deriving H and W from scalar
  coordinate moments already needed by the extractor.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


OUT = Path(__file__).resolve().parent / "task132.onnx"
W = 15


def init(name: str, arr) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr), name=name)


def build() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    idx30 = np.arange(30, dtype=np.float32)
    idx15 = np.arange(W, dtype=np.float16)
    inits = [
        init("idx", idx30),
        init("idx_sq", idx30 * idx30),
        init("row_twice15", (2 * idx15).reshape(1, 1, W, 1)),
        init("col_twice15", (2 * idx15).reshape(1, 1, 1, W)),
        init("pads_row", np.array([0, 0, 0, 0, 0, 0, 30 - W, 0], np.int64)),
        init("pads_col", np.array([0, 0, 0, 0, 0, 0, 0, 30 - W], np.int64)),
        init(
            "bg_vec_f16",
            np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float16).reshape(1, 1, 10),
        ),
        init("pair4_shape", np.array([1, 2, 1, 1], np.int64)),
        init("zero_f32", np.array(0.0, np.float32)),
        init("zero_bool", np.array(False, np.bool_)),
        init("zero_u8", np.array(0, np.uint8)),
        init("one_f32", np.array(1.0, np.float32)),
        init("two_f32", np.array(2.0, np.float32)),
        init("two_f16", np.array(2.0, np.float16)),
        init("fg_keep_u8", np.arange(10, dtype=np.uint8).clip(0, 1).reshape(1, 10)),
        init("axes_all", np.array([1, 2, 3], np.int64)),
        init("axes_chan", np.array([1], np.int64)),
        init("depth10_i64", np.array(10, np.int64)),
        init("onehot_values_f16", np.array([-1, 1], dtype=np.float16)),
    ]

    def n(op: str, ins: list[str], outs: list[str], **attrs) -> None:
        nodes.append(helper.make_node(op, ins, outs, **attrs))

    # Per-color coordinate moments for the two marked cells of each foreground
    # color. These also let us recover the native H/W without row_any30/col_any30.
    n("Einsum", ["input", "idx"], ["row_sum"], equation="nchw,h->nc")
    n("Einsum", ["input", "idx_sq"], ["row_sq_sum"], equation="nchw,h->nc")
    n("Einsum", ["input", "idx"], ["col_sum"], equation="nchw,w->nc")
    n("Einsum", ["input", "idx_sq"], ["col_sq_sum"], equation="nchw,w->nc")

    # input is one-hot on the native HxW grid, so sum(input)=H*W,
    # sum(row_sum)=W*H*(H-1)/2, and sum(col_sum)=H*W*(W-1)/2.
    n("ReduceSum", ["input", "axes_all"], ["cell_count"], keepdims=1)
    n("ReduceSum", ["row_sum", "axes_chan"], ["row_total"], keepdims=1)
    n("Mul", ["row_total", "two_f32"], ["row_total2"])
    n("Div", ["row_total2", "cell_count"], ["height_m1"])
    n("Add", ["height_m1", "one_f32"], ["height_f32"])
    n("Div", ["cell_count", "height_f32"], ["width_f32"])
    n("Cast", ["height_f32"], ["height_f16"], to=TensorProto.FLOAT16)
    n("Cast", ["width_f32"], ["width_f16"], to=TensorProto.FLOAT16)
    n("Mul", ["height_f16", "two_f16"], ["height_twice"])
    n("Mul", ["width_f16", "two_f16"], ["width_twice"])
    n("Less", ["row_twice15", "height_twice"], ["row_valid15"])
    n("Less", ["col_twice15", "width_twice"], ["col_valid15"])

    n("Greater", ["row_sum", "zero_f32"], ["fg_present_all_b"])
    n("Where", ["fg_present_all_b", "fg_keep_u8", "zero_u8"], ["fg_present_u8"])
    n("ArgMax", ["fg_present_u8"], ["fg_first"], axis=1, keepdims=1, select_last_index=0)
    n("ArgMax", ["fg_present_u8"], ["fg_last"], axis=1, keepdims=1, select_last_index=1)
    n("Concat", ["fg_first", "fg_last"], ["fg_pair4"], axis=1)

    for base in ("row_sum", "row_sq_sum", "col_sum", "col_sq_sum"):
        n("GatherElements", [base, "fg_pair4"], [f"{base}_sel_f"], axis=1)
        n("Cast", [f"{base}_sel_f"], [f"{base}_sel"], to=TensorProto.FLOAT16)

    def span(axis: str, twice15: str, out: str) -> None:
        n("Mul", [f"{axis}_sum_sel", f"{axis}_sum_sel"], [f"{axis}_s2"])
        n("Mul", [f"{axis}_sq_sum_sel", "two_f16"], [f"{axis}_q2"])
        n("Sub", [f"{axis}_q2", f"{axis}_s2"], [f"{axis}_d2"])
        n("Sqrt", [f"{axis}_d2"], [f"{axis}_abs"])
        n("Sub", [f"{axis}_sum_sel", f"{axis}_abs"], [f"{axis}_lo"])
        n("Add", [f"{axis}_sum_sel", f"{axis}_abs"], [f"{axis}_hi"])
        n("Reshape", [f"{axis}_lo", "pair4_shape"], [f"{axis}_lo4"])
        n("Reshape", [f"{axis}_hi", "pair4_shape"], [f"{axis}_hi4"])
        n("LessOrEqual", [f"{axis}_lo4", twice15], [f"{axis}_ge"])
        n("LessOrEqual", [twice15, f"{axis}_hi4"], [f"{axis}_le"])
        n("And", [f"{axis}_ge", f"{axis}_le"], [f"{axis}_rects15"])

    span("row", "row_twice15", "row_rects15")
    span("col", "col_twice15", "col_rects15")

    n("Concat", ["row_valid15", "row_rects15"], ["row_terms15_b"], axis=1)
    n("Concat", ["col_valid15", "col_rects15"], ["col_terms15_b"], axis=1)
    n("Pad", ["row_terms15_b", "pads_row", "zero_bool"], ["row_terms_b"], mode="constant")
    n("Pad", ["col_terms15_b", "pads_col", "zero_bool"], ["col_terms_b"], mode="constant")
    n("Cast", ["row_terms_b"], ["row_terms"], to=TensorProto.FLOAT16)
    n("Cast", ["col_terms_b"], ["col_terms"], to=TensorProto.FLOAT16)
    n("OneHot", ["fg_pair4", "depth10_i64", "onehot_values_f16"], ["color_delta"], axis=-1)
    n("Concat", ["bg_vec_f16", "color_delta"], ["color_terms"], axis=1)
    n("Einsum", ["row_terms", "col_terms", "color_terms"], ["output"], equation="ntro,ntoc,ntk->nkrc")

    graph = helper.make_graph(
        nodes,
        "task132_scalar_extent_fp16_tail",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 14)])
    model.ir_version = 10
    del model.graph.value_info[:]
    return model


if __name__ == "__main__":
    model = build()
    onnx.checker.check_model(model)
    onnx.save(model, OUT)
    print(OUT)
