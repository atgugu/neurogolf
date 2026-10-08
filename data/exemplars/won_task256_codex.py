#!/usr/bin/env python3
"""task256 raw threshold renderer.

True rule: red prefix row R length L gives T=R+L; green/red/blue are
prefixes by row, and black is the valid input rectangle minus that prefix.

Two priced families before building:

1. Exact bool color-id+Pad family (pin-like)
   params 113, memory 1552, cost 1665:
   scalars 15 B; canvas extent 250 B; row/base/color masks 611 B;
   native 4x13x13 bool renderer 676 B. Correct, but not a replacement.

2. Row/column threshold renderer (this file)
   planned params about 193, memory about 1200, cost about 1393:
   scalar decode 149 B; row logits [1,10,30,1] and builders 661 B;
   column logits [1,10,1,30] and builders 390 B; final Greater is free.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def init(name: str, arr: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(np.ascontiguousarray(arr), name=name)


def node(nodes: list[onnx.NodeProto], op: str, ins: list[str], out: str, **attrs) -> str:
    nodes.append(helper.make_node(op, ins, [out], **attrs))
    return out


def build() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    # Scalar selectors for the red prefix and for input row widths.
    ch2 = np.zeros(10, np.float32)
    ch2[2] = 1.0
    inits += [
        init("ch2", ch2),
        init("ones_c", np.ones(10, np.float32)),
        init("ones_w", np.ones(30, np.float32)),
        init("row_w", np.arange(30, dtype=np.float32)),
    ]

    # Coordinate vectors and reusable constants for the terminal threshold form.
    inits += [
        init("row", np.arange(30, dtype=np.uint8).reshape(1, 1, 30, 1)),
        init("col", np.arange(30, dtype=np.uint8).reshape(1, 1, 1, 30)),
        init("zero", np.array([0], dtype=np.uint8)),
        init("thirty", np.array([30], dtype=np.uint8)),
        init("twentynine", np.array([29], dtype=np.uint8)),
        init("fifty", np.array([50], dtype=np.uint8)),
        init("zero_row", np.zeros((1, 1, 30, 1), dtype=np.uint8)),
        init("zero_col", np.zeros((1, 1, 1, 30), dtype=np.uint8)),
    ]

    # Decode L and R from red cells. The generator guarantees one red prefix.
    node(nodes, "Einsum", ["input", "ch2", "ones_w"], "len_f", equation="nchw,c,w->n")
    node(
        nodes,
        "Einsum",
        ["input", "ch2", "row_w", "ones_w"],
        "row_sum_f",
        equation="nchw,c,h,w->n",
    )
    node(nodes, "Div", ["row_sum_f", "len_f"], "R_f")
    node(nodes, "Cast", ["len_f"], "L", to=TensorProto.UINT8)
    node(nodes, "Cast", ["R_f"], "R", to=TensorProto.UINT8)
    node(nodes, "Add", ["R", "L"], "T")

    # Decode width and height from row sums: area = H*W, width = max row sum,
    # height = area / width. This replaces the pin's separate row/column scans.
    node(nodes, "Einsum", ["input", "ones_c", "ones_w"], "grid_rows", equation="nchw,c,w->nh")
    node(nodes, "ReduceMax", ["grid_rows"], "width_f", axes=[1], keepdims=1)
    node(nodes, "ReduceSum", ["grid_rows"], "area_f", keepdims=1)
    node(nodes, "Div", ["area_f", "width_f"], "height_f")
    node(nodes, "Cast", ["width_f"], "width", to=TensorProto.UINT8)
    node(nodes, "Cast", ["height_f"], "height", to=TensorProto.UINT8)

    # Left side of the final Greater: per-channel row thresholds.
    # Channels: 0 black, 1 blue, 2 red, 3 green, 4..9 zero.
    node(nodes, "Sub", ["T", "row"], "T_minus_row")
    node(nodes, "Sub", ["thirty", "T"], "thirty_minus_T")
    node(nodes, "Add", ["row", "thirty_minus_T"], "black_left_base")
    node(nodes, "Less", ["row", "R"], "above")
    node(nodes, "Greater", ["row", "R"], "below")
    node(nodes, "Less", ["row", "T"], "row_lt_T")
    node(nodes, "And", ["below", "row_lt_T"], "blue_row_ok")
    node(nodes, "Equal", ["row", "R"], "same_row")
    node(nodes, "Less", ["row", "height"], "row_valid")
    node(nodes, "Where", ["row_valid", "black_left_base", "zero"], "black_left")
    node(nodes, "Where", ["blue_row_ok", "T_minus_row", "zero"], "blue_left")
    node(nodes, "Where", ["same_row", "L", "zero"], "red_left")
    node(nodes, "Where", ["above", "T_minus_row", "zero"], "green_left")
    node(
        nodes,
        "Concat",
        [
            "black_left",
            "blue_left",
            "red_left",
            "green_left",
            "zero_row",
            "zero_row",
            "zero_row",
            "zero_row",
            "zero_row",
            "zero_row",
        ],
        "row_terms",
        axis=1,
    )

    # Right side of the final Greater: per-channel column thresholds.
    # Colored channels use c. Black uses 29-c for valid columns, and 50 outside
    # the input width so the comparison is forced false.
    node(nodes, "Sub", ["twentynine", "col"], "col_rev")
    node(nodes, "Less", ["col", "width"], "col_valid")
    node(nodes, "Where", ["col_valid", "col_rev", "fifty"], "black_right")
    node(
        nodes,
        "Concat",
        [
            "black_right",
            "col",
            "col",
            "col",
            "zero_col",
            "zero_col",
            "zero_col",
            "zero_col",
            "zero_col",
            "zero_col",
        ],
        "col_terms",
        axis=1,
    )

    # output[o,r,c] is true iff the row threshold beats the column threshold.
    node(nodes, "Greater", ["row_terms", "col_terms"], "output")

    graph = helper.make_graph(
        nodes,
        "task256_rowcol_threshold",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


if __name__ == "__main__":
    onnx.save(build(), "task256.onnx")
    print("saved task256.onnx")
