#!/usr/bin/env python3
"""Build task199.onnx with a uint8 bitset terminal renderer."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


OUT = Path(__file__).with_name("task199.onnx")


def c(name: str, arr) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr), name)


def build() -> onnx.ModelProto:
    color_ids = np.arange(10, dtype=np.float32)
    pos30 = np.arange(30, dtype=np.float32)
    pos_r = np.arange(15, dtype=np.float32).reshape(1, 1, 15, 1)
    pos_c = np.arange(15, dtype=np.float32).reshape(1, 1, 1, 15)
    sq_r = ((np.arange(15, dtype=np.float32) + 1.0) ** 2).reshape(1, 1, 15, 1)
    sq_c = ((np.arange(15, dtype=np.float32) + 1.0) ** 2).reshape(1, 1, 1, 15)
    parity_c = (np.arange(15, dtype=np.uint8) % np.uint8(2)).reshape(1, 1, 1, 15)
    color_ids4 = color_ids.reshape(1, 10, 1, 1)
    channel_base = np.zeros((1, 10, 1, 1), dtype=np.uint8)
    channel_base[0, 0, 0, 0] = 1 | 2 | 4
    channel_base[0, 4, 0, 0] = 8

    inits = [
        c("axes_sum_all", np.array([1, 2, 3], dtype=np.int64)),
        c("color_ids", color_ids),
        c("color_ids4", color_ids4),
        c("pos30", pos30),
        c("pos_r", pos_r),
        c("pos_c", pos_c),
        c("sq_r", sq_r),
        c("sq_c", sq_c),
        c("parity_c", parity_c),
        c("one_f", np.array(1.0, dtype=np.float32)),
        c("two_u8", np.array(2, dtype=np.uint8)),
        c("pads_r", np.array([0, 0, 0, 0, 0, 0, 15, 0], dtype=np.int64)),
        c("pads_c", np.array([0, 0, 0, 0, 0, 0, 0, 15], dtype=np.int64)),
        c("u8_0", np.array(0, dtype=np.uint8)),
        c("u8_4", np.array(4, dtype=np.uint8)),
        c("u8_7", np.array(7, dtype=np.uint8)),
        c("u8_9", np.array(9, dtype=np.uint8)),
        c("u8_14", np.array(14, dtype=np.uint8)),
        c("u8_16", np.array(16, dtype=np.uint8)),
        c("u8_18", np.array(18, dtype=np.uint8)),
        c("u8_28", np.array(28, dtype=np.uint8)),
        c("channel_base", channel_base),
    ]

    N = helper.make_node
    nodes = [
        N("ReduceSum", ["input", "axes_sum_all"], ["cell_count"], keepdims=0),
        N("Einsum", ["input", "color_ids"], ["color_value"], equation="nchw,c->n"),
        N("Einsum", ["input", "color_ids", "pos30"], ["row_weighted"], equation="nchw,c,h->n"),
        N("Einsum", ["input", "color_ids", "pos30"], ["col_weighted"], equation="nchw,c,w->n"),
        N("Div", ["row_weighted", "color_value"], ["row_idx_f"]),
        N("Add", ["row_idx_f", "one_f"], ["row_next_f"]),
        N("Div", ["col_weighted", "color_value"], ["col_idx_f"]),
        N("Mul", ["row_next_f", "row_next_f"], ["row_prefix_limit"]),
        N("LessOrEqual", ["sq_r", "cell_count"], ["valid_r_b"]),
        N("LessOrEqual", ["sq_c", "cell_count"], ["valid_c_b"]),
        N("LessOrEqual", ["sq_r", "row_prefix_limit"], ["prefix_b"]),
        N("Equal", ["pos_r", "row_next_f"], ["row_marker_b"]),
        N("Cast", ["col_idx_f"], ["col_idx_u8"], to=TensorProto.UINT8),
        N("Mod", ["col_idx_u8", "two_u8"], ["col_mod_u8"]),
        N("Equal", ["parity_c", "col_mod_u8"], ["col_parity_b"]),
        N("Equal", ["pos_c", "col_idx_f"], ["col_marker_b"]),
        # Row rectangle bits: prefix has bits 1|8, marker row 2|16, lower valid rows 4.
        N("Where", ["valid_r_b", "u8_4", "u8_0"], ["row_valid_bits"]),
        N("Where", ["row_marker_b", "u8_18", "row_valid_bits"], ["row_marker_bits"]),
        N("Where", ["prefix_b", "u8_9", "row_marker_bits"], ["row_bits15"]),
        N("Pad", ["row_bits15", "pads_r"], ["row_bits30"], mode="constant"),
        # Column rectangle bits: nonparity 1|2|4, parity 2|4|8, marker 4|8|16.
        N("Where", ["col_parity_b", "u8_14", "u8_7"], ["col_parity_bits"]),
        N("Where", ["col_marker_b", "u8_28", "col_parity_bits"], ["col_marker_bits"]),
        N("Where", ["valid_c_b", "col_marker_bits", "u8_0"], ["col_bits15"]),
        N("Pad", ["col_bits15", "pads_c"], ["col_bits30"], mode="constant"),
        # Channel masks map rectangle bits to output channels; marker color gets bit 16.
        N("Equal", ["color_ids4", "color_value"], ["color_eq_b"]),
        N("Cast", ["color_eq_b"], ["color_eq_u8"], to=TensorProto.UINT8),
        N("Mul", ["color_eq_u8", "u8_16"], ["marker_channel_bits"]),
        N("Add", ["channel_base", "marker_channel_bits"], ["channel_bits"]),
        N("BitwiseAnd", ["row_bits30", "channel_bits"], ["row_channel_bits"]),
        N("BitwiseAnd", ["row_channel_bits", "col_bits30"], ["output"]),
    ]

    graph = helper.make_graph(
        nodes,
        "task199_bitset",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.UINT8, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


def main() -> None:
    onnx.save(build(), OUT)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
