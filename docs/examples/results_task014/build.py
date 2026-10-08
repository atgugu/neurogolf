#!/usr/bin/env python3
"""task014: rarest nonzero color bbox crop via sentinel canvas.

True rule: choose the least frequent nonzero color, take that color's tight
bounding box, and return the original pixels inside that box.

The generator guarantees the selected bbox is binary: background zero plus the
selected color. This graph crops only that selected one-hot channel, pads it to
a uint8 sentinel canvas, and lets the final Equal expand to 10 output channels.
"""
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


OUT = Path(__file__).with_name("task014.onnx")


def c(name: str, arr) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr), name=name)


def build() -> onnx.ModelProto:
    inits = [
        c("axes_counts", np.array([0, 2, 3], dtype=np.int64)),
        c("one_i64", np.array([1], dtype=np.int64)),
        c("zero_i64", np.array([0], dtype=np.int64)),
        c("thirty_i64", np.array([30], dtype=np.int64)),
        c("zero_f16", np.array(0.0, dtype=np.float16)),
        c("big_f16", np.array(60000.0, dtype=np.float16)),
        c("arange10", np.arange(10, dtype=np.int64)),
        c("zero_f32", np.array(0.0, dtype=np.float32)),
        c("pad_value", np.array(255, dtype=np.uint8)),
        c("pal_base", np.array([0, 101, 102, 103, 104, 105, 106, 107, 108, 109], dtype=np.uint8).reshape(1, 10, 1, 1)),
        c("pal_one", np.array(1, dtype=np.uint8)),
        c("pal_shape", np.array([1, 10, 1, 1], dtype=np.int64)),
    ]

    nodes = [
        helper.make_node("ReduceSum", ["input", "axes_counts"], ["counts10_f32"], keepdims=0),
        helper.make_node("Cast", ["counts10_f32"], ["counts10_f16"], to=TensorProto.FLOAT16),
        helper.make_node("Slice", ["counts10_f16", "one_i64", "thirty_i64"], ["counts9_f16"]),
        helper.make_node("LessOrEqual", ["counts9_f16", "zero_f16"], ["no_color"]),
        helper.make_node("Where", ["no_color", "big_f16", "counts9_f16"], ["counts_safe"]),
        helper.make_node("ArgMin", ["counts_safe"], ["tgt_idx0"], axis=0, keepdims=1),
        helper.make_node("Add", ["tgt_idx0", "one_i64"], ["tgt_color"]),
        helper.make_node("Equal", ["arange10", "tgt_color"], ["sel_bool"]),
        helper.make_node("Cast", ["sel_bool"], ["sel_f32"], to=TensorProto.FLOAT),
        helper.make_node("Einsum", ["input", "sel_f32"], ["row_counts"], equation="nchw,c->nh"),
        helper.make_node("Einsum", ["input", "sel_f32"], ["col_counts"], equation="nchw,c->nw"),
        helper.make_node("Greater", ["row_counts", "zero_f32"], ["pres_row"]),
        helper.make_node("Cast", ["pres_row"], ["pres_row_u8"], to=TensorProto.UINT8),
        helper.make_node("Greater", ["col_counts", "zero_f32"], ["pres_col"]),
        helper.make_node("Cast", ["pres_col"], ["pres_col_u8"], to=TensorProto.UINT8),
        helper.make_node("ArgMax", ["pres_row_u8"], ["first_row"], axis=1, keepdims=0),
        helper.make_node("ArgMax", ["pres_row_u8"], ["last_row"], axis=1, keepdims=0, select_last_index=1),
        helper.make_node("ArgMax", ["pres_col_u8"], ["first_col"], axis=1, keepdims=0),
        helper.make_node("ArgMax", ["pres_col_u8"], ["last_col"], axis=1, keepdims=0, select_last_index=1),
        helper.make_node("Add", ["tgt_color", "one_i64"], ["tgt_color_end"]),
        helper.make_node("Add", ["last_row", "one_i64"], ["row_end"]),
        helper.make_node("Add", ["last_col", "one_i64"], ["col_end"]),
        helper.make_node("Concat", ["zero_i64", "tgt_color", "first_row", "first_col"], ["slice_starts"], axis=0),
        helper.make_node("Concat", ["one_i64", "tgt_color_end", "row_end", "col_end"], ["slice_ends"], axis=0),
        helper.make_node("Slice", ["input", "slice_starts", "slice_ends"], ["crop_f32"]),
        helper.make_node("Cast", ["crop_f32"], ["crop_u8"], to=TensorProto.UINT8),
        helper.make_node("Sub", ["row_end", "first_row"], ["crop_h"]),
        helper.make_node("Sub", ["col_end", "first_col"], ["crop_w"]),
        helper.make_node("Sub", ["thirty_i64", "crop_h"], ["pad_h"]),
        helper.make_node("Sub", ["thirty_i64", "crop_w"], ["pad_w"]),
        helper.make_node(
            "Concat",
            [
                "zero_i64",
                "zero_i64",
                "zero_i64",
                "zero_i64",
                "zero_i64",
                "zero_i64",
                "pad_h",
                "pad_w",
            ],
            ["pads"],
            axis=0,
        ),
        helper.make_node("Pad", ["crop_u8", "pads", "pad_value"], ["canvas01"]),
        helper.make_node("Reshape", ["sel_bool", "pal_shape"], ["sel4d"]),
        helper.make_node("Where", ["sel4d", "pal_one", "pal_base"], ["pal"]),
        helper.make_node("Equal", ["canvas01", "pal"], ["output"]),
    ]

    value_info = [
        helper.make_tensor_value_info("crop_f32", TensorProto.FLOAT, [1, 1, 1, 1]),
        helper.make_tensor_value_info("crop_u8", TensorProto.UINT8, [1, 1, 1, 1]),
        helper.make_tensor_value_info("canvas01", TensorProto.UINT8, [1, 1, 30, 30]),
    ]
    graph = helper.make_graph(
        nodes,
        "task014_sentinel_canvas_dyn_palette",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
        inits,
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    onnx.save(build(), OUT)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
