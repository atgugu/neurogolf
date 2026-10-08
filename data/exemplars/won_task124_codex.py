#!/usr/bin/env python3
"""task124: relowered native u8 code plane + terminal ConvInteger renderer.

Priced families before build:

Family A, native Gather-LUT/code core:
  bg5_u8 [1,1,5,10] u8 50; fg5_u8 [1,1,5,10] u8 50;
  fg_source_map [1,1,5,10] i32 200; code rows/pad/bottom/final u8
  20+20+10+30+46+50+50+100; color/weights 89; params about 75.
  Paper target: about 900-950 cost, registerable but not a 2x cut.

Family B, packed bitset core:
  bg/fg crop 100; row byte packs about 50; scalar candidate/bottom bytes about
  150; byte unpack and code plane about 400; color/weights about 89; params
  about 120. Paper target: about 850-1000, with more ORT risk from bit ops.

The chosen family keeps the source-map branch but replaces the charged fp32
Slice/Less crop and the old bool out_fg -> Cast -> Add tail with a native u8
0/2 code plane. The final ConvInteger still pads and expands to int32 output.
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto as TP, helper as H, numpy_helper as NH

sys.path.insert(0, "../../runner")
from ngolf import relower_onehot_plane  # noqa: E402


OPSET, IR = 14, 10


def build():
    inits = []

    def init(arr, name):
        inits.append(NH.from_array(np.asarray(arr), name))
        return name

    nodes, relower_inits, _ = relower_onehot_plane(
        "input",
        "bg5_u8",
        channel=0,
        crop=((0, 1), (0, 1), (0, 5), (0, 10)),
        dtype="u8",
        starts_name="bg5_starts",
        ends_name="bg5_ends",
    )
    for name, arr in relower_inits:
        init(arr, name)

    one_u8 = init(np.array(1, np.uint8), "one_u8")
    row02_idx = init(np.array([0, 2], np.int64), "row02_idx")
    shift_kernel = init(np.array([-1, 1], np.int64), "shift_kernel")
    p3a_idx = init(np.array([1], np.int64), "p3a_idx")
    p3b_idx = init(np.array([4], np.int64), "p3b_idx")
    three_i64 = init(np.array(3, np.int64), "three_i64")
    source_offsets = init(
        np.array(
            [
                [[26], [8], [26], [8], [26]],
                [[24], [5], [23], [4], [22]],
                [[22], [2], [20], [0], [18]],
                [[36], [8], [26], [36], [8]],
            ],
            dtype=np.int32,
        ),
        "source_offsets",
    )
    col_idx = init(np.arange(10, dtype=np.int32), "col_idx")
    row_flat_shape = init(np.array([10], np.int64), "row_flat_shape")
    split2 = init(np.array([1, 1], np.int64), "split2")
    false8 = init(np.zeros(8, dtype=np.uint8), "false8")
    weight_base = init(
        np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.uint8).reshape(10, 1, 1, 1),
        "weight_base",
    )
    x_zero_point = init(np.array(1, dtype=np.uint8), "x_zero_point")
    w_zero_point = init(np.array(1, dtype=np.uint8), "w_zero_point")
    channel_idx = init(np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), "channel_idx")
    fg_w_shape = init(np.array([10, 1, 1, 1], np.int64), "fg_w_shape")

    nodes.extend(
        [
            H.make_node("Sub", [one_u8, "bg5_u8"], ["fg5_u8"]),
            H.make_node("Gather", ["fg5_u8", row02_idx], ["fg02_u8"], axis=2),
            H.make_node("ArgMax", ["fg02_u8"], ["left_cols"], axis=3, keepdims=0),
            H.make_node("MatMul", ["left_cols", shift_kernel], ["shift"]),
            H.make_node("Gather", ["fg5_u8", p3a_idx], ["p3_rows_a"], axis=2),
            H.make_node("Gather", ["fg5_u8", p3b_idx], ["p3_rows_b"], axis=2),
            H.make_node("Equal", ["p3_rows_a", "p3_rows_b"], ["p3_equal"]),
            H.make_node("Cast", ["p3_equal"], ["p3_equal_u8"], to=TP.UINT8),
            H.make_node("ReduceMin", ["p3_equal_u8"], ["p3_all_equal"], axes=[0, 1, 2, 3], keepdims=0),
            H.make_node("Cast", ["p3_all_equal"], ["is_p3"], to=TP.BOOL),
            H.make_node("Where", ["is_p3", three_i64, "shift"], ["candidate"]),
            H.make_node("Gather", [source_offsets, "candidate"], ["source_offset"], axis=0),
            H.make_node("Add", ["source_offset", col_idx], ["fg_source_map"]),
            H.make_node("Add", ["fg5_u8", "fg5_u8"], ["fg5_code"]),
            H.make_node("Add", ["fg02_u8", "fg02_u8"], ["fg02_code"]),
            H.make_node("Split", ["fg02_code", split2], ["row0_4d", "row2_4d"], axis=2),
            H.make_node("Add", ["p3_rows_a", "p3_rows_a"], ["row1_4d"]),
            H.make_node("Reshape", ["row0_4d", row_flat_shape], ["row0_flat"]),
            H.make_node("Reshape", ["row1_4d", row_flat_shape], ["row1_flat"]),
            H.make_node("Reshape", ["row2_4d", row_flat_shape], ["row2_flat"]),
            H.make_node("Concat", [false8, "row0_flat", false8, "row1_flat", "row2_flat"], ["fg_pad_flat"], axis=0),
            H.make_node("Gather", ["fg_pad_flat", "fg_source_map"], ["bottom_code"], axis=0),
            H.make_node("Concat", ["fg5_code", "bottom_code"], ["fg_code_u8"], axis=2),
            H.make_node("ReduceMax", ["input"], ["present_f"], axes=[2, 3], keepdims=1),
            H.make_node("ArgMax", ["present_f"], ["color_idx_i64"], axis=1, keepdims=1, select_last_index=1),
            H.make_node("Cast", ["color_idx_i64"], ["color_idx"], to=TP.UINT8),
            H.make_node("Equal", ["color_idx", channel_idx], ["fg_weight_b"]),
            H.make_node("Cast", ["fg_weight_b"], ["fg_weight_u8"], to=TP.UINT8),
            H.make_node("Reshape", ["fg_weight_u8", fg_w_shape], ["fg_weight"]),
            H.make_node("Add", [weight_base, "fg_weight"], ["weights_u8"]),
            H.make_node(
                "ConvInteger",
                ["fg_code_u8", "weights_u8", x_zero_point, w_zero_point],
                ["output"],
                pads=[0, 0, 20, 20],
                kernel_shape=[1, 1],
            ),
        ]
    )

    graph = H.make_graph(
        nodes,
        "task124_relowered_u8_code_convinteger",
        [H.make_tensor_value_info("input", TP.FLOAT, [1, 10, 30, 30])],
        [H.make_tensor_value_info("output", TP.INT32, [1, 10, 30, 30])],
        initializer=inits,
    )
    model = H.make_model(graph, opset_imports=[H.make_operatorsetid("", OPSET)], ir_version=IR)
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    onnx.save(build(), "task124.onnx")
    print("saved task124.onnx")
