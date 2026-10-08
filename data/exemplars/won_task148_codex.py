#!/usr/bin/env python3
"""task148 row-event shift renderer.

This keeps the verified terminal input-transform renderer, but removes the
clamped/padded target-row gather.  The generator bounds make the row offset
small enough that negative GatherElements indices land only in lower non-marker
rows, so the shifted marker vector can be gathered directly:

  targetrow[h] = markerrow[h - abs(left_top - right_top)]

The source-span separator remains the att8 signed row-code polynomial because
ONNX Einsum cannot consume uint8/int8 event vectors together with the float
one-hot input.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


OUT = Path(__file__).resolve().parent / "task148.onnx"


def init(name: str, arr) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr), name=name)


def node(op: str, inputs: list[str], output: str, **attrs) -> onnx.NodeProto:
    return helper.make_node(op, inputs, [output], name=output, **attrs)


def build() -> onnx.ModelProto:
    row_ch = np.zeros((3, 10), dtype=np.float32)
    row_w = np.zeros((3, 30), dtype=np.float32)
    row_ch[0, 2] = 1.0
    row_w[0, :] = 1.0
    row_ch[1, 0] = 1.0
    row_w[1, 0] = 12.0
    row_ch[2, 8] = 1.0
    row_w[2, :] = np.arange(30, dtype=np.float32)

    cols = np.arange(30, dtype=np.float32)
    col_feat = np.vstack([np.ones(30, dtype=np.float32), cols, cols * cols])

    # row basis: [1, x, x*x, markerrow, targetrow]
    # col basis: [1, col, col*col]
    # terms: bg base, red base, cyan->yellow, source span, target fill
    terms = 5
    row_coeff = np.zeros((terms, 5, 3), dtype=np.float32)
    in_coeff = np.zeros((terms, 10), dtype=np.float32)
    out_coeff = np.zeros((terms, 10), dtype=np.float32)

    # Base mapping: background and red stay fixed; cyan marker becomes yellow.
    for t, c, k in ((0, 0, 0), (1, 2, 2), (2, 8, 4)):
        row_coeff[t, 0, 0] = 1.0
        in_coeff[t, c] = 1.0
        out_coeff[t, k] = 1.0

    # Positive exactly for source-row background cells between the red portal
    # and the cyan marker.  This row/column polynomial was solved with a unit
    # margin over every generator-legal width/orientation, while x=0/1/12/13
    # non-marker rows stay negative.
    row_coeff[3, :, :] = 0.01 * np.array(
        [
            [-9.266861711282, -20.90435861185, 1.968717408609],
            [8.955766853889, 1.147443950749, -0.179693586996],
            [-0.688905142607, 0.044768084651, 0.001728674166],
            [6.050432592268, 4.382259378996, -0.484882382977],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    in_coeff[3, 0] = 1.0
    out_coeff[3, 0] = -200.0
    out_coeff[3, 8] = 1.0

    # Target rows can have large negative source-separator values; use a strong
    # overwrite margin so raw>0 has only cyan positive on background cells.
    row_coeff[4, 4, 0] = 1.0
    in_coeff[4, 0] = 1.0
    out_coeff[4, 0] = -10000.0
    out_coeff[4, 8] = 10000.0

    inits = [
        init("row_ch", row_ch),
        init("row_w", row_w),
        init("f0", np.array(0.0, dtype=np.float32)),
        init("f2", np.array(2.0, dtype=np.float32)),
        init("f12", np.array(12.0, dtype=np.float32)),
        init("f13", np.array(13.0, dtype=np.float32)),
        init("ones30", np.ones((1, 30), dtype=np.float32)),
        init("iota30", np.arange(30, dtype=np.int32).reshape(1, 30)),
        init("col_feat", col_feat),
        init("row_coeff", row_coeff),
        init("in_coeff", in_coeff),
        init("out_coeff", out_coeff),
    ]

    nodes = [
        node(
            "Einsum",
            ["input", "row_ch", "row_w"],
            "row_code_f",
            equation="nchw,pc,pw->nh",
        ),
        node("Greater", ["row_code_f", "f0"], "code_gt0"),
        node("Less", ["row_code_f", "f12"], "code_lt12"),
        node("And", ["code_gt0", "code_lt12"], "leftrow"),
        node("Greater", ["row_code_f", "f12"], "rightrow"),
        node("Greater", ["row_code_f", "f2"], "code_gt2"),
        node("And", ["leftrow", "code_gt2"], "marker_left"),
        node("Greater", ["row_code_f", "f13"], "marker_right"),
        node("Or", ["marker_left", "marker_right"], "markerrow"),
        node("Cast", ["leftrow"], "leftrow_u8", to=TensorProto.UINT8),
        node("Cast", ["rightrow"], "rightrow_u8", to=TensorProto.UINT8),
        node("ArgMax", ["leftrow_u8"], "left_top", axis=1, keepdims=1),
        node("ArgMax", ["rightrow_u8"], "right_top", axis=1, keepdims=1),
        node("Sub", ["left_top", "right_top"], "top_sub"),
        node("Abs", ["top_sub"], "delta_i64"),
        node("Cast", ["delta_i64"], "delta_i32", to=TensorProto.INT32),
        node("Sub", ["iota30", "delta_i32"], "shift_idx_raw"),
        node("GatherElements", ["markerrow", "shift_idx_raw"], "targetrow", axis=1),
        node("Cast", ["targetrow"], "target_f", to=TensorProto.FLOAT),
        node("Mul", ["row_code_f", "row_code_f"], "row_code_sq"),
        node("Cast", ["markerrow"], "markerrow_f", to=TensorProto.FLOAT),
        node(
            "Concat",
            ["ones30", "row_code_f", "row_code_sq", "markerrow_f", "target_f"],
            "row_basis",
            axis=0,
        ),
        node(
            "Einsum",
            ["input", "row_basis", "col_feat", "row_coeff", "in_coeff", "out_coeff"],
            "output",
            equation="nchw,ah,bw,tab,tc,tk->nkhw",
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "task148_row_poly",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(
        graph, ir_version=10, opset_imports=[helper.make_opsetid("", 14)]
    )
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


def main() -> None:
    model = build()
    onnx.save(model, OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
