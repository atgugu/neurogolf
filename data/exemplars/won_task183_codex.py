#!/usr/bin/env python3
"""task183: W6 u8 color plane with terminal ConvInteger renderer.

Rule: for N in {2,4,6}, output the full inner N x N square.  Cyan pixels are
recolored from the corner color of their quadrant; other in-output cells are
black; cells outside the N x N output are absent.

The working state is a [1,1,6,6] uint8 code grid.  Real output colors are
remapped to compact codes 1..7, black is encoded as 0, and out-of-output cells
as 8.  The final ConvInteger sees [code, code^2, one] and emits one-hot logits
over the full 30x30 canvas; its output tensor is uncharged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

try:
    import onnxoptimizer
except ImportError:  # pragma: no cover
    onnxoptimizer = None


OUT = Path("task183.onnx")


def t_i64(name: str, arr) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr, dtype=np.int64), name=name)


def t_f32(name: str, arr) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name=name)


def t_f16(name: str, arr) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr, dtype=np.float16), name=name)


def t_i8(name: str, arr) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr, dtype=np.int8), name=name)


def t_u8(name: str, arr) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr, dtype=np.uint8), name=name)


def corner_selector() -> np.ndarray:
    out = np.zeros((4, 30), dtype=np.float32)
    for i, p in enumerate((0, 5, 7, 9)):
        out[i, p] = 1.0
    return out


def color_code_weights() -> np.ndarray:
    # Corner colors are sampled from {2,3,4,5,6,7,9}; compact codes keep the
    # quadratic classifier weights inside int8.
    out = np.zeros(10, dtype=np.float32)
    for code, color in enumerate((2, 3, 4, 5, 6, 7, 9), start=1):
        out[color] = code
    return out


def convinteger_weights() -> np.ndarray:
    # Input channels are [x, x^2, one].  For target t, logit is
    # 1 - (x - t)^2, positive only at the exact compact code.  Channels 1 and 8
    # are impossible output colors and stay negative inside the 6x6 field.
    targets = {0: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 9: 7}
    w = np.zeros((10, 3, 1, 1), dtype=np.int8)
    for cls in (1, 8):
        w[cls, 2, 0, 0] = -1
    for cls, target in targets.items():
        w[cls, 0, 0, 0] = 2 * target
        w[cls, 1, 0, 0] = -1
        w[cls, 2, 0, 0] = 1 - target * target
    return w


def build() -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = [
        t_i64("axes_chw", [1, 2, 3]),
        t_i64("axes_hw", [2, 3]),
        t_i64("starts_cyan6", [8, 2, 2]),
        t_i64("ends_cyan6", [9, 8, 8]),
        t_f32("code_weights", color_code_weights()),
        t_f32("corner_selector", corner_selector()),
        t_u8("half2_u8", [[[[1]]]]),
        t_u8("half4_u8", [[[[2]]]]),
        t_u8("half6_u8", [[[[3]]]]),
        t_u8("row6", np.arange(6, dtype=np.uint8).reshape(1, 1, 6, 1)),
        t_u8("col6", np.arange(6, dtype=np.uint8).reshape(1, 1, 1, 6)),
        t_u8("zero_code", [[[[0]]]]),
        t_u8("outside_code", [[[[8]]]]),
        t_u8("one_plane", np.ones((1, 1, 6, 6), dtype=np.uint8)),
        t_i8("ci_w", convinteger_weights()),
    ]

    nodes: list[onnx.NodeProto] = [
        helper.make_node("Slice", ["input", "starts_cyan6", "ends_cyan6", "axes_chw"], ["cyan6_f"]),
        helper.make_node("Cast", ["cyan6_f"], ["cyan6"], to=TensorProto.UINT8),
        helper.make_node(
            "Einsum",
            ["input", "code_weights", "corner_selector", "corner_selector"],
            ["corner_code_f"],
            equation="nchw,c,rh,sw->nrs",
        ),
        helper.make_node("Cast", ["corner_code_f"], ["corner_code_grid"], to=TensorProto.UINT8),
        helper.make_node("Slice", ["corner_code_grid", "s00", "e11", "axes_corner"], ["tl_code"]),
    ]

    # Add compact corner-grid slice constants after the common constants so the
    # graph keeps the static low-memory corner path from the current pin.
    inits.extend(
        [
            t_i64("axes_corner", [1, 2]),
            t_i64("s00", [0, 0]),
            t_i64("e11", [1, 1]),
            t_i64("s01", [0, 1]),
            t_i64("e12", [1, 2]),
            t_i64("s02", [0, 2]),
            t_i64("e13", [1, 3]),
            t_i64("s03", [0, 3]),
            t_i64("e14", [1, 4]),
            t_i64("s10", [1, 0]),
            t_i64("e21", [2, 1]),
            t_i64("s20", [2, 0]),
            t_i64("e31", [3, 1]),
            t_i64("s30", [3, 0]),
            t_i64("e41", [4, 1]),
            t_i64("s11", [1, 1]),
            t_i64("e22", [2, 2]),
            t_i64("s22", [2, 2]),
            t_i64("e33", [3, 3]),
            t_i64("s33", [3, 3]),
            t_i64("e44", [4, 4]),
            t_i64("feat_axis", [1]),
        ]
    )

    for name, starts, ends in [
        ("tr2_code", "s01", "e12"),
        ("tr4_code", "s02", "e13"),
        ("tr6_code", "s03", "e14"),
        ("bl2_code", "s10", "e21"),
        ("bl4_code", "s20", "e31"),
        ("bl6_code", "s30", "e41"),
        ("br2_code", "s11", "e22"),
        ("br4_code", "s22", "e33"),
        ("br6_code", "s33", "e44"),
    ]:
        nodes.append(helper.make_node("Slice", ["corner_code_grid", starts, ends, "axes_corner"], [name]))

    nodes.extend(
        [
            helper.make_node("Cast", ["tr2_code"], ["is2"], to=TensorProto.BOOL),
            helper.make_node("Cast", ["tr4_code"], ["is4"], to=TensorProto.BOOL),
            helper.make_node("Where", ["is4", "half4_u8", "half6_u8"], ["half_inner"]),
            helper.make_node("Where", ["is2", "half2_u8", "half_inner"], ["half"]),
            helper.make_node("Mul", ["half", "half4_u8"], ["side"]),
            helper.make_node("Less", ["row6", "side"], ["inrow"]),
            helper.make_node("Less", ["col6", "side"], ["incol"]),
            helper.make_node("Less", ["row6", "half"], ["top"]),
            helper.make_node("Less", ["col6", "half"], ["left"]),
            helper.make_node("Where", ["is4", "tr4_code", "tr6_code"], ["tr_inner"]),
            helper.make_node("Where", ["is2", "tr2_code", "tr_inner"], ["tr_code"]),
            helper.make_node("Where", ["is4", "bl4_code", "bl6_code"], ["bl_inner"]),
            helper.make_node("Where", ["is2", "bl2_code", "bl_inner"], ["bl_code"]),
            helper.make_node("Where", ["is4", "br4_code", "br6_code"], ["br_inner"]),
            helper.make_node("Where", ["is2", "br2_code", "br_inner"], ["br_code"]),
            helper.make_node("Where", ["left", "tl_code", "tr_code"], ["toprow"]),
            helper.make_node("Where", ["left", "bl_code", "br_code"], ["botrow"]),
            helper.make_node("Where", ["top", "toprow", "botrow"], ["color_grid_all"]),
            helper.make_node("Mul", ["color_grid_all", "cyan6"], ["inner_masked"]),
            helper.make_node("Where", ["inrow", "zero_code", "outside_code"], ["row_masked"]),
            helper.make_node("Where", ["incol", "zero_code", "outside_code"], ["col_masked"]),
            helper.make_node("Max", ["inner_masked", "row_masked", "col_masked"], ["color_grid"]),
            helper.make_node("Mul", ["color_grid", "color_grid"], ["color_grid_sq"]),
            helper.make_node("Concat", ["color_grid", "color_grid_sq", "one_plane"], ["features"], axis=1),
            helper.make_node(
                "ConvInteger",
                ["features", "ci_w"],
                ["output"],
                kernel_shape=[1, 1],
                pads=[0, 0, 24, 24],
            ),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task183_u8_convinteger_tail",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.INT32, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    return model


def main() -> None:
    model = build()
    if onnxoptimizer is not None:
        model = onnxoptimizer.optimize(model, onnxoptimizer.get_fuse_and_elimination_passes())
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, OUT)
    print(f"saved {OUT} ({len(model.SerializeToString())} bytes)")


if __name__ == "__main__":
    main()
