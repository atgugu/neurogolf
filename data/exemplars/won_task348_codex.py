#!/usr/bin/env python3
"""task348 v10 — dynamic ConvInteger terminal renderer.

The source rule is a <=10x10 V fan.  This build keeps only row features as the
ConvInteger input and builds the column-dependent coefficients as the dynamic
ConvInteger weight tensor.  uint8 zero-points encode signed scores, avoiding the
ORT 1.24 int8-Where trap while Conv padding embeds the crop into the 30x30 output.
"""
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

FULL = 30
CROP = 10
SEED_COL0 = 2
SEED_COL1 = 8
ZP = 100


def init(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name=name)


def build():
    nodes = []
    sel_c7 = np.zeros((10,), dtype=np.float32)
    sel_c7[7] = 1.0
    sel0 = np.zeros((FULL,), dtype=np.float32)
    sel0[0] = 1.0

    # Kernel column j maps to output column 9-j with pads=[0,9,20,29].
    rev_cols = np.arange(CROP - 1, -1, -1)
    pos_c_rev = (rev_cols - SEED_COL0).astype(np.int8).reshape(1, 1, 1, CROP)
    parity_rev = (rev_cols % 2).astype(np.int8).reshape(1, 1, 1, CROP)

    inits = [
        init("c7_top_st", np.array([7, 0, SEED_COL0], dtype=np.int64)),
        init("c7_top_en", np.array([8, 1, SEED_COL1], dtype=np.int64)),
        init("axes_c_hw", np.array([1, 2, 3], dtype=np.int64)),
        init("pos_c_rev_i8", pos_c_rev),
        init("pos_c_parity_rev_i8", parity_rev),
        init("pos_r0_p100_u8", (ZP + np.arange(CROP, dtype=np.uint8)).reshape(1, 1, CROP, 1)),
        init("ones_row_enc_u8", np.full((1, 1, CROP, 1), ZP + 1, dtype=np.uint8)),
        init("coef_one_col_u8", np.full((1, 1, 1, CROP), ZP + 1, dtype=np.uint8)),
        init("zero_kernel_u8", np.full((1, 2, 1, CROP), ZP, dtype=np.uint8)),
        init("sel_c7", sel_c7),
        init("sel0", sel0),
        init("two_i8", np.array([2], np.int8)),
        init("three_i8", np.array([3], np.int8)),
        init("hundred_u8", np.array([ZP], np.uint8)),
        init("c99_u8", np.array([ZP - 1], np.uint8)),
        init("c101_u8", np.array([ZP + 1], np.uint8)),
        init("c199_u8", np.array([2 * ZP - 1], np.uint8)),
        init("zero_u8", np.array([0], np.uint8)),
    ]

    def add(op, ins, outs, **attrs):
        nodes.append(helper.make_node(op, ins, outs, **attrs))

    # Geometry scalars: seed column is the shifted index in [0, 5].
    add("Slice", ["input", "c7_top_st", "c7_top_en", "axes_c_hw"], ["ch7_top"])
    add("ArgMax", ["ch7_top"], ["col_i64"], axis=3, keepdims=1)
    add("Cast", ["col_i64"], ["col_i8"], to=TensorProto.INT8)
    add("Mod", ["col_i8", "two_i8"], ["col_mod2_i8"], fmod=1)
    add("Einsum", ["input", "sel0"], ["width_f"], equation="nchw,h->n")
    add("Einsum", ["input", "sel0"], ["height_f"], equation="nchw,w->n")
    add("Einsum", ["input", "sel_c7"], ["length_f"], equation="nchw,c->n")
    add("Cast", ["width_f"], ["width_i8"], to=TensorProto.INT8)
    add("Cast", ["height_f"], ["height_u8"], to=TensorProto.UINT8)
    add("Cast", ["length_f"], ["length_u8"], to=TensorProto.UINT8)
    add("Sub", ["width_i8", "three_i8"], ["width_m3_i8"])

    # Row features, encoded with zero-point 100:
    #   bg     = r - length, disabled to -100 outside the input height
    #   stripe = length - 1 - r
    #   one    = 1
    add("Add", ["height_u8", "c99_u8"], ["height_p99_u8"])
    add("GreaterOrEqual", ["height_p99_u8", "pos_r0_p100_u8"], ["row_active_b"])
    add("Sub", ["pos_r0_p100_u8", "length_u8"], ["row_bg_raw_u8"])
    add("Sub", ["c199_u8", "row_bg_raw_u8"], ["row_stripe_u8"])
    add("Where", ["row_active_b", "row_bg_raw_u8", "zero_u8"], ["row_bg_u8"])
    add(
        "Concat",
        [
            "row_bg_u8",
            "ones_row_enc_u8",
            "row_bg_u8",
            "row_bg_u8",
            "row_bg_u8",
            "row_bg_u8",
            "row_stripe_u8",
            "ones_row_enc_u8",
            "row_stripe_u8",
            "ones_row_enc_u8",
        ],
        ["row_basis_u8"],
        axis=1,
    )

    # Reversed column coefficients, encoded with the same zero-point:
    #   channel 0 one-coeff = dist + 1
    #   channels 7/8 one-coeff = 1 - dist on the matching parity, else -100
    # Other row-feature coefficients are static 0 or 1.
    add("Sub", ["pos_c_rev_i8", "col_i8"], ["dx_i8"])
    add("Abs", ["dx_i8"], ["dist_i8"])
    add("Cast", ["dist_i8"], ["dist_u8"], to=TensorProto.UINT8)
    add("GreaterOrEqual", ["width_m3_i8", "pos_c_rev_i8"], ["col_active_b"])
    add("Equal", ["pos_c_parity_rev_i8", "col_mod2_i8"], ["even_b"])
    add("Add", ["c101_u8", "dist_u8"], ["w0_one_raw_u8"])
    add("Where", ["col_active_b", "w0_one_raw_u8", "zero_u8"], ["w0_one_u8"])
    add("Sub", ["c101_u8", "dist_u8"], ["w_stripe_raw_u8"])
    add("Where", ["col_active_b", "w_stripe_raw_u8", "zero_u8"], ["w_stripe_active_u8"])
    add("Where", ["even_b", "w_stripe_active_u8", "zero_u8"], ["w7_one_u8"])
    add("Where", ["even_b", "zero_u8", "w_stripe_active_u8"], ["w8_one_u8"])

    add("Concat", ["coef_one_col_u8", "w0_one_u8"], ["w0_u8"], axis=1)
    add("Concat", ["coef_one_col_u8", "w7_one_u8"], ["w7_u8"], axis=1)
    add("Concat", ["coef_one_col_u8", "w8_one_u8"], ["w8_u8"], axis=1)
    add(
        "Concat",
        [
            "w0_u8",
            "zero_kernel_u8",
            "zero_kernel_u8",
            "zero_kernel_u8",
            "zero_kernel_u8",
            "zero_kernel_u8",
            "zero_kernel_u8",
            "w7_u8",
            "w8_u8",
            "zero_kernel_u8",
        ],
        ["kernel_u8"],
        axis=0,
    )

    # Free terminal renderer.  For output column y in 0..9, kernel column 9-y is
    # used; columns 10..29 and rows 10..29 see only zero-padded row features.
    add(
        "ConvInteger",
        ["row_basis_u8", "kernel_u8", "hundred_u8", "hundred_u8"],
        ["output"],
        pads=[0, 9, 20, 29],
        kernel_shape=[1, CROP],
        group=5,
    )

    graph = helper.make_graph(
        nodes,
        "task348_convinteger_terminal_v10",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, FULL, FULL])],
        [helper.make_tensor_value_info("output", TensorProto.INT32, [1, 10, FULL, FULL])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 14)])
    model.ir_version = 10
    del model.graph.value_info[:]
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "task348.onnx")
    onnx.save(build(), out)
    print(f"saved {out}")
