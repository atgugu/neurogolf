#!/usr/bin/env python3
"""Task 225: axon color LUT with terminal ConvInteger renderer."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = Path(__file__).resolve().parent / "task225.onnx"

F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
I8 = TensorProto.INT8
I32 = TensorProto.INT32
BOOL = TensorProto.BOOL


def _c(name: str, arr: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr), name=name)


def _vi(name: str, dtype: int, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape)


def _label_for_rel(rr: int, cc: int) -> int:
    if rr == 0 and cc == 0:
        return 1
    if rr == 0 and cc == 1:
        return 2
    if rr == 1 and cc == 0:
        return 3
    if rr == 1 and cc == 1:
        return 4
    if rr in (-2, -1) and cc in (-2, -1):
        return 1
    if rr in (-2, -1) and cc in (2, 3):
        return 2
    if rr in (2, 3) and cc in (-2, -1):
        return 3
    if rr in (2, 3) and cc in (2, 3):
        return 4
    return 0


def _selector_bank() -> np.ndarray:
    bank = np.zeros((3, 3, 6, 6), dtype=np.uint8)
    for row0 in range(3):
        for col0 in range(3):
            row = row0 + 1
            col = col0 + 1
            for rr in range(6):
                for cc in range(6):
                    bank[row0, col0, rr, cc] = _label_for_rel(rr - row, cc - col)
    return bank


def _selector_base() -> np.ndarray:
    base = np.zeros((8, 8), dtype=np.uint8)
    for rr in range(-3, 5):
        for cc in range(-3, 5):
            label = 0
            label = _label_for_rel(rr, cc)
            base[rr + 3, cc + 3] = label
    return base


def _conv_integer_weights() -> np.ndarray:
    # color_grid codes are: 10 for background, 1..9 for output channels 1..9.
    target_code = [10, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    weights = np.zeros((10, 3, 1, 1), dtype=np.int8)
    for ch, t in enumerate(target_code):
        weights[ch, 0, 0, 0] = 1 - t * t
        weights[ch, 1, 0, 0] = 2 * t
        weights[ch, 2, 0, 0] = -1
    return weights


def _qlinear_weights() -> np.ndarray:
    # QLinearConv has a per-output bias, so [color, color^2] suffices:
    # bias + 2*t*color - color^2 = 1 - (color-t)^2.
    # Reserve code 0 for terminal zero-padding: every bias is then non-positive.
    target_code = [10, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    weights = np.zeros((10, 2, 1, 1), dtype=np.int8)
    for ch, t in enumerate(target_code):
        weights[ch, 0, 0, 0] = 2 * t
        weights[ch, 1, 0, 0] = -1
    return weights


def build() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = [
        _c("bg_starts", np.array([0, 0, 1, 1], dtype=np.int32)),
        _c("bg_ends", np.array([1, 1, 4, 4], dtype=np.int32)),
        _c("one_i32", np.int32(1)),
        _c("three_i32", np.int32(3)),
        _c("squeeze_axes_0", np.array([0], dtype=np.int64)),
        _c("slice_prefix_start", np.array([0, 1], dtype=np.int32)),
        _c("slice_extents", np.array([1, 9, 2, 2], dtype=np.int32)),
        _c("zero_color", np.array([[[10]]], dtype=np.uint8)),
        _c("source_kernel", np.arange(1, 10, dtype=np.float32).reshape(1, 9, 1, 1)),
        _c("unsqueeze_axes_1", np.array([1], dtype=np.int64)),
        _c("row_idx_bank", np.array(
            [
                [1, 2, 3, 4, 4, 0],
                [1, 1, 2, 3, 4, 4],
                [0, 1, 1, 2, 3, 4],
            ],
            dtype=np.int32,
        )),
        _c("source_code_table2d", np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 4, 0, 0, 3],
                [0, 0, 1, 2, 0],
                [0, 0, 3, 4, 0],
                [0, 2, 0, 0, 1],
            ],
            dtype=np.int32,
        )),
        _c("x_scale", np.float32(1)),
        _c("x_zero", np.uint8(0)),
        _c("w_q", _qlinear_weights()),
        _c("w_scale", np.float32(1)),
        _c("w_zero", np.int8(0)),
        _c("y_scale", np.float32(1)),
        _c("y_zero", np.uint8(0)),
        _c("q_bias", np.array([1 - t * t for t in [10, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.int32)),
    ]

    # Verified axon anchor: find the top-left of the 2x2 non-background block.
    nodes.append(helper.make_node("Slice", ["input", "bg_starts", "bg_ends"], ["anchor_bg"]))
    # This crop is exactly one-hot background bits, so its byte relower is exact.
    nodes.append(helper.make_node("Cast", ["anchor_bg"], ["anchor_bg_u8"], to=U8))
    nodes.append(helper.make_node("Flatten", ["anchor_bg_u8"], ["anchor_flat"], axis=1))
    nodes.append(helper.make_node("ArgMin", ["anchor_flat"], ["anchor_raw"], axis=1, keepdims=0))
    nodes.append(helper.make_node("Cast", ["anchor_raw"], ["anchor_i32"], to=I32))
    nodes.append(helper.make_node("Div", ["anchor_i32", "three_i32"], ["anchor_row0"]))
    nodes.append(helper.make_node("Mod", ["anchor_i32", "three_i32"], ["anchor_col0"]))
    nodes.append(helper.make_node("Squeeze", ["anchor_row0", "squeeze_axes_0"], ["anchor_row0_scalar"]))
    nodes.append(helper.make_node("Squeeze", ["anchor_col0", "squeeze_axes_0"], ["anchor_col0_scalar"]))
    nodes.append(helper.make_node("Add", ["anchor_row0", "one_i32"], ["anchor_row"]))
    nodes.append(helper.make_node("Add", ["anchor_col0", "one_i32"], ["anchor_col"]))
    nodes.append(helper.make_node("Concat", ["slice_prefix_start", "anchor_row", "anchor_col"], ["block_starts"], axis=0))
    nodes.append(helper.make_node("Add", ["block_starts", "slice_extents"], ["block_ends"]))

    # Source colors are still forced through a charged f32 crop by ONNX's input dtype.
    nodes.append(helper.make_node("Slice", ["input", "block_starts", "block_ends"], ["source_block"]))
    nodes.append(helper.make_node("Conv", ["source_block", "source_kernel"], ["source_code_f"]))
    nodes.append(helper.make_node("Cast", ["source_code_f"], ["source_argmax_u8"], to=U8))
    nodes.append(helper.make_node("Flatten", ["source_argmax_u8"], ["source_color_block"], axis=1))
    nodes.append(helper.make_node("Unsqueeze", ["source_color_block", "unsqueeze_axes_1"], ["source_color_block_ch"]))
    nodes.append(helper.make_node("Concat", ["zero_color", "source_color_block_ch"], ["source_colors"], axis=2))

    nodes.append(helper.make_node("Gather", ["row_idx_bank", "anchor_row0_scalar"], ["row_idx"], axis=0))
    nodes.append(helper.make_node("Gather", ["row_idx_bank", "anchor_col0_scalar"], ["col_idx"], axis=0))
    nodes.append(helper.make_node("Gather", ["source_colors", "source_code_table2d"], ["color_table2d"], axis=2))
    nodes.append(helper.make_node("Gather", ["color_table2d", "row_idx"], ["row_color_table"], axis=2))
    nodes.append(helper.make_node("Gather", ["row_color_table", "col_idx"], ["color_grid"], axis=3))
    nodes.append(helper.make_node("Mul", ["color_grid", "color_grid"], ["color_sq_ch"]))
    nodes.append(helper.make_node("Concat", ["color_grid", "color_sq_ch"], ["features_u8"], axis=1))
    nodes.append(helper.make_node("QLinearConv", ["features_u8", "x_scale", "x_zero", "w_q", "w_scale", "w_zero", "y_scale", "y_zero", "q_bias"], ["output"], pads=[0, 0, 24, 24]))

    value_info = [
        _vi("anchor_bg", F32, [1, 1, 3, 3]),
        _vi("anchor_bg_u8", U8, [1, 1, 3, 3]),
        _vi("anchor_flat", U8, [1, 9]),
        _vi("anchor_raw", TensorProto.INT64, [1]),
        _vi("anchor_i32", I32, [1]),
        _vi("anchor_row0", I32, [1]),
        _vi("anchor_col0", I32, [1]),
        _vi("anchor_row0_scalar", I32, []),
        _vi("anchor_col0_scalar", I32, []),
        _vi("anchor_row", I32, [1]),
        _vi("anchor_col", I32, [1]),
        _vi("block_starts", I32, [4]),
        _vi("block_ends", I32, [4]),
        _vi("source_block", F32, [1, 9, 2, 2]),
        _vi("source_code_f", F32, [1, 1, 2, 2]),
        _vi("source_argmax_u8", U8, [1, 1, 2, 2]),
        _vi("source_color_block", U8, [1, 4]),
        _vi("source_color_block_ch", U8, [1, 1, 4]),
        _vi("source_colors", U8, [1, 1, 5]),
        _vi("row_idx", I32, [6]),
        _vi("col_idx", I32, [6]),
        _vi("color_table2d", U8, [1, 1, 5, 5]),
        _vi("row_color_table", U8, [1, 1, 6, 5]),
        _vi("color_grid", U8, [1, 1, 6, 6]),
        _vi("color_sq_ch", U8, [1, 1, 6, 6]),
        _vi("features_u8", U8, [1, 2, 6, 6]),
    ]

    graph = helper.make_graph(
        nodes,
        "task225_selector_renderer_probe",
        [helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", U8, [1, 10, 30, 30])],
        initializer=inits,
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 10
    return model


def main() -> None:
    model = build()
    onnx.checker.check_model(model)
    onnx.save(model, OUT)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
