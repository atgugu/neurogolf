#!/usr/bin/env python3
"""task035 native-code quadratic terminal renderer.

The live pin builds a native scalar grid, pads it to 30x30, then uses terminal
Equal to expand color IDs to channels.  This variant keeps the same native
projection algorithm but encodes the native grid so Max can merge markers
without a high-bit cleanup pass, then lets a terminal quantized 1x1 convolution
both pad to 30x30 and decode quadratic code features to the 10 output channels.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = Path(__file__).resolve().parent / "task035.onnx"


def init(name: str, arr) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr), name=name)


def onehot(length: int, idx: int) -> np.ndarray:
    arr = np.zeros((length,), dtype=np.float32)
    arr[idx] = 1.0
    return arr


def col_window(start: int, width: int) -> np.ndarray:
    arr = np.zeros((30, width), dtype=np.float32)
    for j in range(width):
        arr[start + j, j] = 1.0
    return arr


def row_window(start: int, height: int) -> np.ndarray:
    arr = np.zeros((30, height), dtype=np.float32)
    for j in range(height):
        arr[start + j, j] = 1.0
    return arr


def pack_window(start: int, length: int, base: int = 16) -> np.ndarray:
    arr = np.zeros((30,), dtype=np.float32)
    v = 1
    for j in range(length):
        arr[start + j] = float(v)
        v *= base
    return arr


CODE_MAP = np.array([1, 3, 4, 5, 6, 7, 8, 9, 2, 10], dtype=np.int64)


def qconv_weights() -> tuple[np.ndarray, np.ndarray]:
    """Weights/bias for quadratic code match.

    Features are [x, x*x].  For output color c with native code k:
    score = -x*x + 2*k*x + (1-k*k), so the matching integer code is +1 and
    every other code is <= 0 after uint8 clipping.
    """
    weights = np.empty((10, 2, 1, 1), dtype=np.int8)
    bias = np.empty((10,), dtype=np.int32)
    for color in range(10):
        code = int(CODE_MAP[color])
        weights[color, 0, 0, 0] = 2 * code
        weights[color, 1, 0, 0] = -1
        bias[color] = 1 - code * code
    return weights, bias


def build() -> onnx.ModelProto:
    idx = np.int64
    qw, qb = qconv_weights()
    inits = [
        # Codes are ordered so movable marker colors are greater than cyan.
        # That lets Max perform the marker overwrite directly.
        init("marker_w", CODE_MAP.astype(np.float32)),
        init("top_cyan_s", np.array([8, 3, 2], dtype=idx)),
        init("top_cyan_e", np.array([9, 4, 6], dtype=idx)),
        init("right_cyan_s", np.array([8, 3, 5], dtype=idx)),
        init("right_cyan_e", np.array([9, 8, 6], dtype=idx)),
        init("slice_axes", np.array([1, 2, 3], dtype=idx)),
        init("arange_5_v", np.arange(5, dtype=idx).reshape(1, 1, 5, 1)),
        init("arange_4_h", np.arange(4, dtype=idx).reshape(1, 1, 1, 4)),
        init("core_top_row", np.array([1, 0, 0, 0, 0], dtype=np.bool_).reshape(1, 1, 5, 1)),
        init("core_right_col", np.array([0, 0, 0, 1], dtype=np.bool_).reshape(1, 1, 1, 4)),
        init("z_row_l", np.ones((1, 1, 1, 2), dtype=np.uint8)),
        init("z_mid_c1", np.ones((1, 1, 5, 1), dtype=np.uint8)),
        init("z_row8", np.ones((1, 1, 1, 10), dtype=np.uint8)),
        init("code_one", np.array(1, dtype=np.uint8)),
        init("q_x_scale", np.array(1.0, dtype=np.float32)),
        init("q_x_zp", np.array(0, dtype=np.uint8)),
        init("q_w", qw),
        init("q_w_zp", np.array(0, dtype=np.int8)),
        init("q_bias", qb),
        init("row0_f", onehot(30, 0).reshape(1, 30)),
        init("row9_f", onehot(30, 9).reshape(1, 30)),
        init("col0_f", onehot(30, 0).reshape(30, 1)),
        init("col9_f", onehot(30, 9).reshape(30, 1)),
        init("cols_2_5_pack", pack_window(2, 4)),
        init("rows_3_7_pack", pack_window(3, 5)),
        init("pack_pow4", (16.0 ** np.arange(4, dtype=np.float32)).reshape(1, 1, 1, 4)),
        init("pack_pow5", (16.0 ** np.arange(5, dtype=np.float32)).reshape(1, 1, 5, 1)),
        init("nibble_mask", np.array(15, dtype=np.uint8)),
    ]
    nodes = [
        helper.make_node("Slice", ["input", "top_cyan_s", "top_cyan_e", "slice_axes"], ["top_cyan_f"]),
        helper.make_node("Slice", ["input", "right_cyan_s", "right_cyan_e", "slice_axes"], ["right_cyan_f"]),
        helper.make_node("Cast", ["top_cyan_f"], ["top_cyan_u8"], to=TensorProto.UINT8),
        helper.make_node("Cast", ["right_cyan_f"], ["right_cyan_u8"], to=TensorProto.UINT8),
        helper.make_node("ArgMax", ["top_cyan_u8"], ["p_left"], axis=3, keepdims=1, select_last_index=0),
        helper.make_node("ArgMax", ["right_cyan_u8"], ["p_bot"], axis=2, keepdims=1, select_last_index=1),
        helper.make_node("Equal", ["arange_4_h", "p_left"], ["c_left_oh"]),
        helper.make_node("Equal", ["arange_5_v", "p_bot"], ["r_bot_oh"]),
        helper.make_node("Mul", ["right_cyan_u8", "top_cyan_u8"], ["pool_mask"]),
        helper.make_node("Add", ["pool_mask", "code_one"], ["pool_core"]),
        helper.make_node("Einsum", ["input", "marker_w", "row0_f", "cols_2_5_pack"], ["top_pack"], equation="nkhw,k,ah,w->na"),
        helper.make_node("Einsum", ["input", "marker_w", "row9_f", "cols_2_5_pack"], ["bot_pack"], equation="nkhw,k,ah,w->na"),
        helper.make_node("Einsum", ["input", "marker_w", "rows_3_7_pack", "col0_f"], ["left_pack"], equation="nkhw,k,h,wb->nb"),
        helper.make_node("Einsum", ["input", "marker_w", "rows_3_7_pack", "col9_f"], ["right_pack"], equation="nkhw,k,h,wb->nb"),
        helper.make_node("Div", ["top_pack", "pack_pow4"], ["top_shift"]),
        helper.make_node("Div", ["bot_pack", "pack_pow4"], ["bot_shift"]),
        helper.make_node("Div", ["left_pack", "pack_pow5"], ["left_shift"]),
        helper.make_node("Div", ["right_pack", "pack_pow5"], ["right_shift"]),
        helper.make_node("Cast", ["top_shift"], ["top_byte"], to=TensorProto.UINT8),
        helper.make_node("Cast", ["bot_shift"], ["bot_byte"], to=TensorProto.UINT8),
        helper.make_node("Cast", ["left_shift"], ["left_byte"], to=TensorProto.UINT8),
        helper.make_node("Cast", ["right_shift"], ["right_byte"], to=TensorProto.UINT8),
        helper.make_node("BitwiseAnd", ["top_byte", "nibble_mask"], ["top_m"]),
        helper.make_node("BitwiseAnd", ["bot_byte", "nibble_mask"], ["bot_m"]),
        helper.make_node("BitwiseAnd", ["left_byte", "nibble_mask"], ["left_m"]),
        helper.make_node("BitwiseAnd", ["right_byte", "nibble_mask"], ["right_m"]),
        helper.make_node("Where", ["core_top_row", "top_m", "z_mid_c1"], ["core_top"]),
        helper.make_node("Where", ["r_bot_oh", "bot_m", "z_mid_c1"], ["core_bot"]),
        helper.make_node("Where", ["c_left_oh", "left_m", "z_mid_c1"], ["core_left"]),
        helper.make_node("Where", ["core_right_col", "right_m", "z_mid_c1"], ["core_right"]),
        helper.make_node("Max", ["pool_core", "core_top", "core_bot", "core_left", "core_right"], ["core_final"]),
        helper.make_node("Concat", ["z_row_l", "top_m", "z_row_l", "z_row_l"], ["row_top"], axis=3),
        helper.make_node("Concat", ["z_row_l", "bot_m", "z_row_l", "z_row_l"], ["row_bot"], axis=3),
        helper.make_node("Concat", ["left_m", "z_mid_c1", "core_final", "z_mid_c1", "z_mid_c1", "z_mid_c1", "right_m"], ["mid_rows"], axis=3),
        helper.make_node("Concat", ["row_top", "z_row8", "z_row8", "mid_rows", "z_row8", "row_bot"], ["final_grid_u8"], axis=2),
        helper.make_node("Mul", ["final_grid_u8", "final_grid_u8"], ["grid_sq"]),
        helper.make_node("Concat", ["final_grid_u8", "grid_sq"], ["poly_features"], axis=1),
        helper.make_node(
            "QLinearConv",
            [
                "poly_features",
                "q_x_scale",
                "q_x_zp",
                "q_w",
                "q_x_scale",
                "q_w_zp",
                "q_x_scale",
                "q_x_zp",
                "q_bias",
            ],
            ["output"],
            pads=[0, 0, 20, 20],
        ),
    ]
    g = helper.make_graph(
        nodes,
        "task035_native_poly_qconv_terminal",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.UINT8, [1, 10, 30, 30])],
        inits,
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)])
    m.ir_version = 8
    onnx.checker.check_model(m, full_check=True)
    onnx.save(m, str(OUT))
    print(f"saved {OUT}")
    return m


if __name__ == "__main__":
    build()
