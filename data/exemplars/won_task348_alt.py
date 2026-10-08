#!/usr/bin/env python3
"""task348 — terminal bool emitter from class-specific row/column inequalities."""
import sys
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

W = 10
FULL = 30
SEED_COL0 = 2
SEED_COL1 = 8


def init(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name=name)


def build():
    nodes = []
    sel_c7 = np.zeros((10,), dtype=np.float32)
    sel_c7[7] = 1.0
    row0_sel = np.zeros((FULL,), dtype=np.float32)
    row0_sel[0] = 1.0
    col0_sel = np.zeros((FULL,), dtype=np.float32)
    col0_sel[0] = 1.0
    inits = [
        init("c7_top_st", np.array([7, 0, SEED_COL0], dtype=np.int64)),
        init("c7_top_en", np.array([8, 1, SEED_COL1], dtype=np.int64)),
        init("axes_c_hw", np.array([1, 2, 3], dtype=np.int64)),
        init("pos_c_i8", (np.arange(FULL) - SEED_COL0).astype(np.int8).reshape(1, 1, 1, FULL)),
        init("pos_c_parity_i8", (np.arange(FULL) % 2).astype(np.int8).reshape(1, 1, 1, FULL)),
        init("pos_r1_i8", np.arange(1, FULL + 1, dtype=np.int8).reshape(1, 1, FULL, 1)),
        init("two_i8", np.array([2], np.int8)),
        init("three_i8", np.array([3], np.int8)),
        init("sel_c7", sel_c7),
        init("row0_sel", row0_sel),
        init("col0_sel", col0_sel),
        init("zero_i8", np.array([0], np.int8)),
        init("off100_i8", np.array([100], np.int8)),
        init("off101_i8", np.array([101], np.int8)),
        init("low_u8", np.array([0], np.uint8)),
        init("high_u8", np.array([200], np.uint8)),
        init("col_high_u8", np.full((1, 1, 1, FULL), 200, dtype=np.uint8)),
    ]

    def add(op, ins, outs, **attrs):
        nodes.append(helper.make_node(op, ins, outs, **attrs))

    add("Slice", ["input", "c7_top_st", "c7_top_en", "axes_c_hw"], ["ch7_top"])
    add("ArgMax", ["ch7_top"], ["col_i64"], axis=3, keepdims=1)
    add("Cast", ["col_i64"], ["col_i8"], to=TensorProto.INT8)
    add("Einsum", ["input", "row0_sel"], ["width_f"], equation="nchw,h->n")
    add("Einsum", ["input", "col0_sel"], ["height_f"], equation="nchw,w->n")
    add("Cast", ["width_f"], ["width_i8"], to=TensorProto.INT8)
    add("Cast", ["height_f"], ["height_i8"], to=TensorProto.INT8)
    add("Sub", ["width_i8", "three_i8"], ["width_m3_i8"])
    add("GreaterOrEqual", ["height_i8", "pos_r1_i8"], ["row_active_b"])
    add("GreaterOrEqual", ["width_m3_i8", "pos_c_i8"], ["col_active_b"])
    add("Einsum", ["input", "sel_c7"], ["length_f"], equation="nchw,c->n")
    add("Cast", ["length_f"], ["length_i8"], to=TensorProto.INT8)
    add("Sub", ["length_i8", "pos_r1_i8"], ["half_width_i8"])
    add("Sub", ["zero_i8", "half_width_i8"], ["neg_half_width_i8"])
    add("Sub", ["off100_i8", "half_width_i8"], ["row_bg_raw_i8"])
    add("Cast", ["row_bg_raw_i8"], ["row_bg_raw_u8"], to=TensorProto.UINT8)
    add("Where", ["row_active_b", "row_bg_raw_u8", "low_u8"], ["row_bg_u8"])
    add("Sub", ["off100_i8", "neg_half_width_i8"], ["row_stripe_raw_i8"])
    add("Cast", ["row_stripe_raw_i8"], ["row_stripe_u8"], to=TensorProto.UINT8)
    add("Sub", ["pos_c_i8", "col_i8"], ["dx_i8"])
    add("Abs", ["dx_i8"], ["dist_i8"])
    add("Sub", ["off101_i8", "dist_i8"], ["col_bg_raw_i8"])
    add("Cast", ["col_bg_raw_i8"], ["col_bg_raw_u8"], to=TensorProto.UINT8)
    add("Where", ["col_active_b", "col_bg_raw_u8", "high_u8"], ["col_bg_u8"])
    add("Sub", ["zero_i8", "dist_i8"], ["neg_dist_i8"])
    add("Sub", ["off100_i8", "neg_dist_i8"], ["col_stripe_raw_i8"])
    add("Cast", ["col_stripe_raw_i8"], ["col_stripe_u8"], to=TensorProto.UINT8)
    add("Mod", ["col_i8", "two_i8"], ["col_mod2_i8"], fmod=1)
    add("Equal", ["pos_c_parity_i8", "col_mod2_i8"], ["even_b"])
    add("Where", ["col_active_b", "col_stripe_u8", "high_u8"], ["col_stripe_active_u8"])
    add("Where", ["even_b", "col_stripe_active_u8", "high_u8"], ["col_7_u8"])
    add("Where", ["even_b", "high_u8", "col_stripe_active_u8"], ["col_8_u8"])
    add(
        "Concat",
        [
            "row_bg_u8",
            "row_bg_u8",
            "row_bg_u8",
            "row_bg_u8",
            "row_bg_u8",
            "row_bg_u8",
            "row_bg_u8",
            "row_stripe_u8",
            "row_stripe_u8",
            "row_bg_u8",
        ],
        ["row_score_u8"],
        axis=1,
    )
    add(
        "Concat",
        [
            "col_bg_u8",
            "col_high_u8",
            "col_high_u8",
            "col_high_u8",
            "col_high_u8",
            "col_high_u8",
            "col_high_u8",
            "col_7_u8",
            "col_8_u8",
            "col_high_u8",
        ],
        ["col_score_u8"],
        axis=1,
    )
    add("GreaterOrEqual", ["row_score_u8", "col_score_u8"], ["output"])

    graph = helper.make_graph(
        nodes,
        "task348_terminal_ge",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 14)])
    model.ir_version = 10
    del model.graph.value_info[:]
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    out = Path("task348.onnx")
    onnx.save(build(), out)
    print(f"saved {out}")
