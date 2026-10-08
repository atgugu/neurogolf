#!/usr/bin/env python3
"""task400 att12 - structured-palette color-id renderer.

The old pin copies a full one-hot 5x5 patch, which charges f32[1,10,5,5].
This graph keeps only the black mask from the mirrored patch and reconstructs
nonblack cells from the generator's three fixed D4 palette regions:

  A = corner regions, B = edge bands, C = center.

For each region, a scalar weighted average over visible nonblack cells recovers
the region's color id. The final charged one-hot patch is bool[1,10,5,5].
Compared with att10, the category renderer uses row/column broadcast choices
instead of materializing two 5x5 region masks.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def init(name: str, arr, dtype=None):
    arr = np.asarray(arr, dtype=dtype)
    return numpy_helper.from_array(arr, name=name)


def vi(name: str, elem_type: int, shape):
    return helper.make_tensor_value_info(name, elem_type, shape)


def build():
    blue = np.zeros((1, 10, 1, 1), dtype=np.float32)
    blue[0, 1, 0, 0] = 1.0

    edge = np.zeros(30, dtype=np.float32)
    edge[:6] = 1.0
    edge[18:24] = 1.0
    center = np.zeros(30, dtype=np.float32)
    center[6:18] = 1.0

    ch_count = np.zeros(10, dtype=np.float32)
    ch_count[2:] = 1.0
    ch_weight = np.arange(10, dtype=np.float32)
    ch_weight[:2] = 0.0

    inits = [
        init("blue", blue),
        init("neg_pos25", -np.arange(30, dtype=np.float32) / 25.0),
        init("base25", np.array([25.0], dtype=np.float32)),
        init("lo5", np.array([6, 7, 8, 9, 10], dtype=np.int32)),
        init("hi5", np.array([17, 18, 19, 20, 21], dtype=np.int32)),
        init("row_axes", np.array([0, 1, 3], dtype=np.int64)),
        init("col_axes", np.array([0, 1, 2], dtype=np.int64)),
        init("ch_count", ch_count),
        init("ch_weight", ch_weight),
        init("edge", edge),
        init("center", center),
        init("one_f", np.array([1.0], dtype=np.float32)),
        init("ch0_start", np.array([0], dtype=np.int32)),
        init("end_delta", np.array([1, -35, -35], dtype=np.int32)),
        init("axes_chw", np.array([1, 2, 3], dtype=np.int32)),
        init("steps_chw", np.array([1, -1, -1], dtype=np.int32)),
        init("zero_u8", np.array([0], dtype=np.uint8)),
        init("arange10", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)),
        init("pads", np.array([0, 0, 0, 0, 0, 0, 25, 25], dtype=np.int64)),
    ]

    nodes = [
        # Locate the 5x5 blue cutout by its centroid and derive the 180-degree
        # partner start coordinate used by the old pin's negative-step Slice.
        helper.make_node("Einsum", ["input", "blue", "neg_pos25"], ["yneg"],
                         equation="nchw,zcij,h->n"),
        helper.make_node("Einsum", ["input", "blue", "neg_pos25"], ["xneg"],
                         equation="nchw,zcij,w->n"),
        helper.make_node("Add", ["yneg", "base25"], ["sy_f"]),
        helper.make_node("Add", ["xneg", "base25"], ["sx_f"]),
        helper.make_node("Cast", ["sy_f"], ["sy_i"], to=TensorProto.INT32),
        helper.make_node("Cast", ["sx_f"], ["sx_i"], to=TensorProto.INT32),
        # Dynamic A/B/C category vectors for the 5x5 patch.  For output
        # relative index j, the mirrored coordinate is start - j, so the
        # edge test can compare the scalar start against per-j thresholds.
        helper.make_node("Less", ["sy_i", "lo5"], ["row_lo"]),
        helper.make_node("Greater", ["sy_i", "hi5"], ["row_hi"]),
        helper.make_node("Or", ["row_lo", "row_hi"], ["row_edge"]),
        helper.make_node("Less", ["sx_i", "lo5"], ["col_lo"]),
        helper.make_node("Greater", ["sx_i", "hi5"], ["col_hi"]),
        helper.make_node("Or", ["col_lo", "col_hi"], ["col_edge"]),
        helper.make_node("Unsqueeze", ["row_edge", "row_axes"], ["row_e4"]),
        helper.make_node("Unsqueeze", ["col_edge", "col_axes"], ["col_e4"]),

        # Recover one scalar color id per generator region. The count masks
        # exclude black and blue; the weighted masks encode color ids 2..9.
        helper.make_node("Einsum", ["input", "ch_weight", "edge", "edge"], ["A_sum"],
                         equation="nchw,c,h,w->n"),
        helper.make_node("Einsum", ["input", "ch_count", "edge", "edge"], ["A_cnt"],
                         equation="nchw,c,h,w->n"),
        helper.make_node("Einsum", ["input", "ch_weight", "center", "center"], ["C_sum"],
                         equation="nchw,c,h,w->n"),
        helper.make_node("Einsum", ["input", "ch_count", "center", "center"], ["C_cnt"],
                         equation="nchw,c,h,w->n"),
        helper.make_node("Einsum", ["input", "ch_weight"], ["T_sum"],
                         equation="nchw,c->n"),
        helper.make_node("Einsum", ["input", "ch_count"], ["T_cnt"],
                         equation="nchw,c->n"),
        helper.make_node("Sub", ["T_sum", "A_sum"], ["B_sum0"]),
        helper.make_node("Sub", ["B_sum0", "C_sum"], ["B_sum"]),
        helper.make_node("Sub", ["T_cnt", "A_cnt"], ["B_cnt0"]),
        helper.make_node("Sub", ["B_cnt0", "C_cnt"], ["B_cnt"]),
        helper.make_node("Max", ["A_cnt", "one_f"], ["A_den"]),
        helper.make_node("Max", ["B_cnt", "one_f"], ["B_den"]),
        helper.make_node("Max", ["C_cnt", "one_f"], ["C_den"]),
        helper.make_node("Div", ["A_sum", "A_den"], ["A_id_f"]),
        helper.make_node("Div", ["B_sum", "B_den"], ["B_id_f"]),
        helper.make_node("Div", ["C_sum", "C_den"], ["C_id_f"]),
        helper.make_node("Cast", ["A_id_f"], ["A_id"], to=TensorProto.UINT8),
        helper.make_node("Cast", ["B_id_f"], ["B_id"], to=TensorProto.UINT8),
        helper.make_node("Cast", ["C_id_f"], ["C_id"], to=TensorProto.UINT8),

        # Slice only channel 0 from the mirrored 5x5 patch: this is the black
        # mask. It is 100 B instead of the old 1000 B full one-hot crop.
        helper.make_node("Concat", ["ch0_start", "sy_i", "sx_i"], ["starts3"], axis=0),
        helper.make_node("Add", ["starts3", "end_delta"], ["ends3"]),
        helper.make_node("Slice", ["input", "starts3", "ends3", "axes_chw", "steps_chw"],
                         ["black_f"]),
        helper.make_node("Cast", ["black_f"], ["black"], to=TensorProto.BOOL),

        # Region id patch -> bool one-hot 5x5 patch -> final free Pad.
        # row edge + col edge => A; row/col mixed => B; row center + col
        # center => C.  The first two choices are only [1,1,1,5].
        helper.make_node("Where", ["col_e4", "A_id", "B_id"], ["edge_choice"]),
        helper.make_node("Where", ["col_e4", "B_id", "C_id"], ["center_choice"]),
        helper.make_node("Where", ["row_e4", "edge_choice", "center_choice"], ["id_cat"]),
        helper.make_node("Where", ["black", "zero_u8", "id_cat"], ["id_patch"]),
        helper.make_node("Equal", ["id_patch", "arange10"], ["patch_oh"]),
        helper.make_node("Pad", ["patch_oh", "pads"], ["output"], mode="constant"),
    ]

    value_info = [
        vi("yneg", TensorProto.FLOAT, [1]),
        vi("xneg", TensorProto.FLOAT, [1]),
        vi("sy_f", TensorProto.FLOAT, [1]),
        vi("sx_f", TensorProto.FLOAT, [1]),
        vi("sy_i", TensorProto.INT32, [1]),
        vi("sx_i", TensorProto.INT32, [1]),
        vi("row_lo", TensorProto.BOOL, [5]),
        vi("row_hi", TensorProto.BOOL, [5]),
        vi("row_edge", TensorProto.BOOL, [5]),
        vi("col_lo", TensorProto.BOOL, [5]),
        vi("col_hi", TensorProto.BOOL, [5]),
        vi("col_edge", TensorProto.BOOL, [5]),
        vi("row_e4", TensorProto.BOOL, [1, 1, 5, 1]),
        vi("col_e4", TensorProto.BOOL, [1, 1, 1, 5]),
        vi("A_sum", TensorProto.FLOAT, [1]),
        vi("A_cnt", TensorProto.FLOAT, [1]),
        vi("C_sum", TensorProto.FLOAT, [1]),
        vi("C_cnt", TensorProto.FLOAT, [1]),
        vi("T_sum", TensorProto.FLOAT, [1]),
        vi("T_cnt", TensorProto.FLOAT, [1]),
        vi("B_sum0", TensorProto.FLOAT, [1]),
        vi("B_sum", TensorProto.FLOAT, [1]),
        vi("B_cnt0", TensorProto.FLOAT, [1]),
        vi("B_cnt", TensorProto.FLOAT, [1]),
        vi("A_den", TensorProto.FLOAT, [1]),
        vi("B_den", TensorProto.FLOAT, [1]),
        vi("C_den", TensorProto.FLOAT, [1]),
        vi("A_id_f", TensorProto.FLOAT, [1]),
        vi("B_id_f", TensorProto.FLOAT, [1]),
        vi("C_id_f", TensorProto.FLOAT, [1]),
        vi("A_id", TensorProto.UINT8, [1]),
        vi("B_id", TensorProto.UINT8, [1]),
        vi("C_id", TensorProto.UINT8, [1]),
        vi("starts3", TensorProto.INT32, [3]),
        vi("ends3", TensorProto.INT32, [3]),
        vi("black_f", TensorProto.FLOAT, [1, 1, 5, 5]),
        vi("black", TensorProto.BOOL, [1, 1, 5, 5]),
        vi("edge_choice", TensorProto.UINT8, [1, 1, 1, 5]),
        vi("center_choice", TensorProto.UINT8, [1, 1, 1, 5]),
        vi("id_cat", TensorProto.UINT8, [1, 1, 5, 5]),
        vi("id_patch", TensorProto.UINT8, [1, 1, 5, 5]),
        vi("patch_oh", TensorProto.BOOL, [1, 10, 5, 5]),
    ]

    graph = helper.make_graph(
        nodes,
        "task400_structured_palette",
        [vi("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [vi("output", TensorProto.BOOL, [1, 10, 30, 30])],
        inits,
        value_info=value_info,
    )
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 14)])
    onnx.checker.check_model(model, full_check=True)
    return model


if __name__ == "__main__":
    onnx.save(build(), "task400.onnx")
    print("wrote task400.onnx")
