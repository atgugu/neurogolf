#!/usr/bin/env python3
"""task177 packed-column rebuild.

Rule: crop the full nonzero rectangle and mirror it horizontally.

The current pin materializes an 18x18 f32 color-id plane, casts it to u8,
then gathers the mirrored 8x8 sample.  This candidate packs four adjacent
inner columns into a uint16 lane per row, computes bbox state on the packed
strip, decodes only the 8x8 mirrored sample, then uses a final QLinearConv
over masked low-nibble features to classify/pad in the free output op.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def init(name: str, arr: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(arr, name=name)


def build() -> onnx.ModelProto:
    # Output [1,1,18,5].  Lane j packs input columns 4*j .. 4*j+3
    # as four 4-bit color ids, so the generator-bounded columns 1..18
    # cover the certified active range 1..18; lane 0 includes the blank
    # border column 0, and lane 4 reaches the certified right blank border.
    # This absolute-column packing keeps the Conv kernel one column smaller
    # than the shifted inner-column version.  The
    # height uses the same dilated strip as the pin:
    # output row r samples input row r+1 and preserves the certified border.
    w = np.zeros((1, 10, 3, 11), dtype=np.float32)
    for color in range(1, 10):
        for k, scale in enumerate([1, 16, 256, 4096]):
            w[0, color, 1, k] = color * scale

    inits = [
        init("w_pack4", w),
        init("zero_f32", np.array(0, dtype=np.float32)),
        init("thr15_f32", np.array(15, dtype=np.float32)),
        init("thr255_f32", np.array(255, dtype=np.float32)),
        init("thr4095_f32", np.array(4095, dtype=np.float32)),
        init("pow16_u16", np.array([1, 16, 256, 4096], dtype=np.uint16)),
        init("bitmask_u8", np.array([1, 2, 4, 8], dtype=np.uint8).reshape(1, 4, 1, 1)),
        init("one_f32", np.array(1, dtype=np.float32)),
        init("zero_u8", np.array(0, dtype=np.uint8)),
        init("zero_i8", np.array(0, dtype=np.int8)),
        init(
            "w_term_i8",
            np.array(
                [
                    [0, 0, 0, 0],
                    [8, -4, -2, -1],
                    [-8, 4, -2, 0],
                    [8, 4, -2, 0],
                    [-8, -4, 2, 0],
                    [8, -4, 2, 0],
                    [-8, 4, 2, 1],
                    [8, 4, 2, 1],
                    [-8, 0, 0, 1],
                    [8, 0, 0, 1],
                ],
                dtype=np.int8,
            ).reshape(10, 4, 1, 1),
        ),
        init("b_term_i32", np.array([0, 0, 0, -8, 0, -8, -8, -16, 0, -8], dtype=np.int32)),
        init("idx8_i32", np.arange(8, dtype=np.int32)),
        init("four_i32", np.array(4, dtype=np.int32)),
        init("eighteen_i32", np.array(18, dtype=np.int32)),
        init("twenty_i32", np.array(20, dtype=np.int32)),
        init("axes_rows", np.array([3], dtype=np.int64)),
        init("axes_groups", np.array([2], dtype=np.int64)),
        init("shape_1", np.array([1], dtype=np.int64)),
    ]

    n = helper.make_node
    nodes = [
        n(
            "Conv",
            ["input", "w_pack4"],
            ["pack_f32"],
            dilations=[11, 1],
            pads=[10, 0, 0, 0],
            strides=[1, 4],
        ),
        n("ReduceMax", ["pack_f32", "axes_rows"], ["row_m"], keepdims=0),
        n("Greater", ["row_m", "zero_f32"], ["row_b"]),
        n("Cast", ["row_b"], ["row_u8"], to=TensorProto.UINT8),
        n("ArgMax", ["row_u8"], ["rmin_i64"], axis=2, keepdims=0),
        n("Cast", ["rmin_i64"], ["rmin_raw"], to=TensorProto.INT32),
        n("Reshape", ["rmin_raw", "shape_1"], ["rmin"]),
        n("ReduceMax", ["pack_f32", "axes_groups"], ["grp_m"], keepdims=0),
        n("Greater", ["grp_m", "zero_f32"], ["grp_b"]),
        n("Cast", ["grp_b"], ["grp_u8"], to=TensorProto.UINT8),
        n("ArgMax", ["grp_u8"], ["gmax_i64"], axis=2, keepdims=0, select_last_index=1),
        n("Cast", ["gmax_i64"], ["gmax_raw"], to=TensorProto.INT32),
        n("Reshape", ["gmax_raw", "shape_1"], ["gmax"]),
        n("Gather", ["grp_m", "gmax"], ["gval"], axis=2),
        n("Greater", ["gval", "thr15_f32"], ["off1_b"]),
        n("Greater", ["gval", "thr255_f32"], ["off2_b"]),
        n("Greater", ["gval", "thr4095_f32"], ["off3_b"]),
        n("Cast", ["off1_b"], ["off1_u8"], to=TensorProto.UINT8),
        n("Cast", ["off2_b"], ["off2_u8"], to=TensorProto.UINT8),
        n("Cast", ["off3_b"], ["off3_u8"], to=TensorProto.UINT8),
        n("Add", ["off1_u8", "off2_u8"], ["off12_u8"]),
        n("Add", ["off12_u8", "off3_u8"], ["off_u8"]),
        n("Cast", ["off_u8"], ["off_raw"], to=TensorProto.INT32),
        n("Reshape", ["off_raw", "shape_1"], ["off"]),
        n("Mul", ["gmax", "four_i32"], ["gbase"]),
        n("Add", ["gbase", "off"], ["cmax"]),
        n("Add", ["rmin", "idx8_i32"], ["out_rows"]),
        n("Mod", ["out_rows", "eighteen_i32"], ["safe_rows"]),
        n("Sub", ["cmax", "idx8_i32"], ["out_cols"]),
        n("Mod", ["out_cols", "twenty_i32"], ["safe_cols"]),
        n("Div", ["safe_cols", "four_i32"], ["grp_idx"]),
        n("Mod", ["safe_cols", "four_i32"], ["rem4"]),
        n("Gather", ["pow16_u16", "rem4"], ["divisor_u16"], axis=0),
        n("Gather", ["pack_f32", "safe_rows"], ["crop_r_f32"], axis=2),
        n("Cast", ["crop_r_f32"], ["crop_r"], to=TensorProto.UINT16),
        n("Gather", ["crop_r", "grp_idx"], ["pack_sample"], axis=3),
        n("Div", ["pack_sample", "divisor_u16"], ["quot_u16"]),
        n("Cast", ["quot_u16"], ["quot_u8"], to=TensorProto.UINT8),
        n("BitwiseAnd", ["quot_u8", "bitmask_u8"], ["bitvals"]),
        n(
            "QLinearConv",
            [
                "bitvals",
                "one_f32",
                "zero_u8",
                "w_term_i8",
                "one_f32",
                "zero_i8",
                "one_f32",
                "zero_u8",
                "b_term_i32",
            ],
            ["output"],
            pads=[0, 0, 22, 22],
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "task177_pack4",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.UINT8, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    return model


if __name__ == "__main__":
    onnx.save(build(), "task177.onnx")
    print("wrote task177.onnx")
