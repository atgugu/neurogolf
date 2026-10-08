import os
#!/usr/bin/env python3
"""task246 W18 scalar HPWL rule with strict-between column coding.

The audited single-channel u8 terminal cannot satisfy the NeuroGolf one-hot
output contract without another 10-channel renderer.  This keeps the verified
scalar rule but changes the column code: horizontal cells are strictly between
the red and green columns, while the green-column vertical leg paints the corner.
That preserves common.hpwl and deletes one 18-cell equality/Where pair.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

sys.path.insert(0, (os.environ.get("NEUROGOLF_CLEAN", "") + ""))
from neurogolf.builders.common import fp32_input, fp32_output, make_model


OUT = Path(__file__).resolve().parent / "task246.onnx"
W = 18


def init(name: str, arr, dtype=None) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr, dtype=dtype), name=name)


def build() -> onnx.ModelProto:
    ch_red = np.zeros((10,), dtype=np.float32)
    ch_red[2] = 1.0
    ch_green = np.zeros((10,), dtype=np.float32)
    ch_green[3] = 1.0

    ch8 = np.zeros((1, 10, 1, 1), dtype=np.float32)
    ch8[0, 8, 0, 0] = 1.0

    nodes = [
        helper.make_node("Einsum", ["input", "ch_red", "idx30"], ["r2_f"], equation="nchw,c,h->n"),
        helper.make_node("Einsum", ["input", "ch_green", "idx30"], ["r3_f"], equation="nchw,c,h->n"),
        helper.make_node("Einsum", ["input", "ch_red", "idx30"], ["c2_f"], equation="nchw,c,w->n"),
        helper.make_node("Einsum", ["input", "ch_green", "idx30"], ["c3_f"], equation="nchw,c,w->n"),
        helper.make_node("Cast", ["r2_f"], ["r2"], to=onnx.TensorProto.UINT8),
        helper.make_node("Cast", ["r3_f"], ["r3"], to=onnx.TensorProto.UINT8),
        helper.make_node("Cast", ["c2_f"], ["c2"], to=onnx.TensorProto.UINT8),
        helper.make_node("Cast", ["c3_f"], ["c3"], to=onnx.TensorProto.UINT8),
        helper.make_node("Min", ["r2", "r3"], ["min_r"]),
        helper.make_node("Max", ["r2", "r3"], ["max_r"]),
        helper.make_node("Min", ["c2", "c3"], ["min_c"]),
        helper.make_node("Max", ["c2", "c3"], ["max_c"]),
        helper.make_node("Less", ["r_range", "max_r"], ["r_less_max"]),
        helper.make_node("Greater", ["r_range", "min_r"], ["r_gt_min"]),
        helper.make_node("And", ["r_less_max", "r_gt_min"], ["bet_r_b"]),
        helper.make_node("Greater", ["c_range", "min_c"], ["c_gt_min"]),
        helper.make_node("Less", ["c_range", "max_c"], ["c_lt_max"]),
        helper.make_node("And", ["c_gt_min", "c_lt_max"], ["c_strict_between"]),
        helper.make_node("Equal", ["r_range", "r2"], ["rh2_b"]),
        helper.make_node("Equal", ["c_range", "c3"], ["c_eq_c3"]),
        helper.make_node("Where", ["bet_r_b", "one_u8", "two_u8"], ["row_tail_u8"]),
        helper.make_node("Where", ["rh2_b", "zero_u8", "row_tail_u8"], ["row18_u8"]),
        helper.make_node("Where", ["c_strict_between", "one_u8", "zero_u8"], ["col_between_u8"]),
        helper.make_node("Where", ["c_eq_c3", "two_u8", "col_between_u8"], ["col18_u8"]),
        helper.make_node("Pad", ["row18_u8", "row_pads", "two_u8"], ["row_u8"], mode="constant"),
        helper.make_node("Pad", ["col18_u8", "col_pads"], ["col_u8"], mode="constant"),
        helper.make_node("Greater", ["col_u8", "row_u8"], ["seg_b"]),
        helper.make_node("Where", ["seg_b", "ch8_onehot", "input"], ["output"]),
    ]

    inits = [
        init("ch_red", ch_red),
        init("ch_green", ch_green),
        init("idx30", np.arange(30, dtype=np.float32)),
        init("r_range", np.arange(W, dtype=np.uint8).reshape(W, 1)),
        init("c_range", np.arange(W, dtype=np.uint8).reshape(1, W)),
        init("zero_u8", [0], np.uint8),
        init("one_u8", [1], np.uint8),
        init("two_u8", [2], np.uint8),
        init("ch8_onehot", ch8),
        init("row_pads", [0, 0, 30 - W, 0], np.int64),
        init("col_pads", [0, 0, 0, 30 - W], np.int64),
    ]
    model = make_model(nodes, [fp32_input()], [fp32_output()], inits, opset=14, name="task246_w18_strict_between")
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    onnx.save(build(), OUT)
    print(f"saved {OUT}")
