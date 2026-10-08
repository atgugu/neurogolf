#!/usr/bin/env python3
"""task013 scalar-broadcast stripe solver.

The verified rule is the alternating-stripe task from ARC-GEN 0a938d79.  The
graph keeps the scalar moment decoder, but replaces the class-expanded column
side with scalar row/column codes and a terminal broadcast Equal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


W = 30
OPSET = 14
F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
BOOL = TensorProto.BOOL


def init(name: str, arr) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr), name=name)


def build() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = [
        init("fg_f", np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)),
        init("color_ids_f", np.arange(10, dtype=np.float32)),
        init("pos_f", np.arange(W, dtype=np.float32)),
        init("pos2_f", np.arange(W, dtype=np.float32) ** 2),
        init("line_ids", np.arange(W, dtype=np.uint8).reshape(1, 1, 1, W)),
        init("row_ids", np.arange(W, dtype=np.uint8).reshape(1, 1, W, 1)),
        init("channel_ids_u8", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)),
        init("one_f", np.array([1.0], dtype=np.float32)),
        init("two_f", np.array([2.0], dtype=np.float32)),
        init("zero_u8", np.array([0], dtype=np.uint8)),
        init("row_pad_u8", np.array([20], dtype=np.uint8)),
        init("col_pad_u8", np.array([128], dtype=np.uint8)),
        init("thirteen_u8", np.array([13], dtype=np.uint8)),
    ]

    def node(op: str, ins: list[str], outs: str | list[str], **kw) -> str:
        if isinstance(outs, str):
            outs = [outs]
        nodes.append(helper.make_node(op, ins, outs, **kw))
        return outs[0]

    x = "input"
    area = node("ReduceSum", [x], "area", keepdims=0)
    row_area_s1 = node("Einsum", [x, "pos_f"], "row_area_s1", equation="nchw,h->n")
    height_minus1 = node(
        "Div",
        [node("Mul", ["row_area_s1", "two_f"], "twice_row_area_s1"), "area"],
        "height_minus1",
    )
    height_f = node("Add", ["height_minus1", "one_f"], "height_f")
    width_f = node("Div", ["area", height_f], "width_f")
    height_u8 = node("Cast", ["height_f"], "height_u8", to=U8)
    width_u8 = node("Cast", ["width_f"], "width_u8", to=U8)
    valid_rows = node("Less", ["row_ids", height_u8], "valid_rows")
    valid_cols = node("Less", ["line_ids", width_u8], "valid_cols")
    hmode = node("Less", ["height_u8", "thirteen_u8"], "hmode1")
    not_hmode = node("Not", [hmode], "not_hmode")

    col_s1 = node("Einsum", [x, "fg_f", "pos_f"], "col_s1", equation="nchw,c,w->n")
    row_s1 = node("Einsum", [x, "fg_f", "pos_f"], "row_s1", equation="nchw,c,h->n")
    col_s2 = node("Einsum", [x, "fg_f", "pos2_f"], "col_s2", equation="nchw,c,w->n")
    row_s2 = node("Einsum", [x, "fg_f", "pos2_f"], "row_s2", equation="nchw,c,h->n")
    color_sum = node("Einsum", [x, "color_ids_f"], "color_sum", equation="nchw,c->n")
    col_color_pos = node("Einsum", [x, "color_ids_f", "pos_f"], "col_color_pos", equation="nchw,c,w->n")
    row_color_pos = node("Einsum", [x, "color_ids_f", "pos_f"], "row_color_pos", equation="nchw,c,h->n")

    axis_s1 = node("Where", [hmode, col_s1, row_s1], "axis_s1")
    axis_s2 = node("Where", [hmode, col_s2, row_s2], "axis_s2")
    axis_color_pos = node("Where", [hmode, col_color_pos, row_color_pos], "axis_color_pos")

    diff_sq = node(
        "Sub",
        [node("Mul", [axis_s2, "two_f"], "two_s2"), node("Mul", [axis_s1, axis_s1], "s1_sq")],
        "diff_sq",
    )
    period_f = node("Sqrt", [diff_sq], "period_f")
    period_u8 = node("Cast", [period_f], "period_u8", to=U8)
    double_period = node("Add", [period_u8, period_u8], "double_period")

    lo_f = node("Div", [node("Sub", [axis_s1, period_f], "lo_num"), "two_f"], "lo_f")
    line_lo = node("Cast", [lo_f], "line_lo", to=U8)
    line_hi = node("Add", [line_lo, period_u8], "line_hi")
    hi_f = node("Cast", [line_hi], "hi_f", to=F32)

    c0_num = node("Sub", [node("Mul", [hi_f, color_sum], "hi_color_sum"), axis_color_pos], "c0_num")
    c0_f = node("Div", [c0_num, period_f], "c0_f")
    c1_f = node("Sub", [color_sum, c0_f], "c1_f")
    line_color0 = node("Cast", [c0_f], "line_color0", to=U8)
    line_color1 = node("Cast", [c1_f], "line_color1", to=U8)

    coord_mod = node("Mod", ["line_ids", double_period], "coord_mod", fmod=0)
    lo_mod = node("Mod", [line_lo, double_period], "lo_mod", fmod=0)
    hi_mod = node("Mod", [line_hi, double_period], "hi_mod", fmod=0)
    ge_start = node("GreaterOrEqual", ["line_ids", line_lo], "ge_start")
    first_phase = node("Equal", [coord_mod, lo_mod], "first_phase")
    second_phase = node("Equal", [coord_mod, hi_mod], "second_phase")
    first_colors = node("Where", [first_phase, line_color0, "zero_u8"], "first_colors")
    phase_colors = node("Where", [second_phase, line_color1, first_colors], "phase_colors")
    line_pattern = node("Where", [ge_start, phase_colors, "zero_u8"], "line_pattern")
    row_pattern = node("Transpose", [line_pattern], "row_pattern", perm=[0, 1, 3, 2])

    h_row_value = node("Where", [hmode, "zero_u8", row_pattern], "h_row_value")
    row_sub_code = node("Where", [valid_rows, h_row_value, "row_pad_u8"], "row_sub_code")
    row_side = node("Sub", ["channel_ids_u8", row_sub_code], "row_side")
    h_col_value = node("Where", [hmode, line_pattern, "zero_u8"], "h_col_value")
    col_side = node("Where", [valid_cols, h_col_value, "col_pad_u8"], "col_side")
    node("Equal", [row_side, col_side], "output")

    graph = helper.make_graph(
        nodes,
        "task013_scalar_broadcast",
        [helper.make_tensor_value_info("input", F32, [1, 10, W, W])],
        [helper.make_tensor_value_info("output", BOOL, [1, 10, W, W])],
        inits,
    )
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    return model


if __name__ == "__main__":
    out = Path(__file__).with_name("task013.onnx")
    onnx.save(build(), out)
    print(out)
