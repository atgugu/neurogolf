#!/usr/bin/env python3
"""task012: dynamic center-channel slice with direct feature stamp.

The graph avoids the banked 12x12 fp32 color-id plane. It derives the two
active colors from global counts, dynamically slices only the center channel
inside the certified 8x8 center window, stamps two terminal-render features
directly, and expands to the 30x30 one-hot output in the final free op.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runner"))
from ngolf import G  # noqa: E402

OUT = Path(__file__).with_name("task012.onnx")


def qconv(g: G, x, w, out_shape, name_dt="u8", pads=None, y_zp=None):
    attrs = {}
    if pads is not None:
        attrs["pads"] = pads
    if y_zp is None:
        y_zp = qconv.zp_u8
    return g.n(
        "QLinearConv",
        [x, qconv.scale, qconv.zp_u8, w, qconv.scale, qconv.zp_u8, qconv.scale, y_zp],
        out_shape,
        name_dt,
        **attrs,
    )


def build() -> G:
    g = G(task=12)
    qconv.scale = g.init(np.array(1.0, dtype=np.float32))
    qconv.zp_u8 = g.init(np.array(0, dtype=np.uint8))
    qconv.one_u8 = g.init(np.array(1, dtype=np.uint8))

    two_f = g.init(np.array([2.0], dtype=np.float32).reshape(1, 1, 1, 1))
    eight_f = g.init(np.array([8.0], dtype=np.float32).reshape(1, 1, 1, 1))
    sum_hw = g.init(np.array([2, 3], dtype=np.int64))
    crange = g.init(np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1))

    counts = g.n("ReduceSum", [g.input, sum_hw], [1, 10, 1, 1], "f32", keepdims=1)
    center_ch_u8 = g.cast(g.eq(counts, two_f), "u8")
    arm_ch_u8 = g.cast(g.eq(counts, eight_f), "u8")
    center_color = qconv(g, center_ch_u8, crange, [1, 1, 1, 1])
    arm_color = qconv(g, arm_ch_u8, crange, [1, 1, 1, 1])

    center_i32_4d = g.cast(center_color, "i32")
    center_i32 = g.reshape(center_i32_4d, [1])
    center_end = g.add(center_i32, g.init(np.array([1], dtype=np.int32)))
    z_i32 = g.init(np.array([0], dtype=np.int32))
    one_i32 = g.init(np.array([1], dtype=np.int32))
    two_i32 = g.init(np.array([2], dtype=np.int32))
    ten_i32 = g.init(np.array([10], dtype=np.int32))
    starts = g.concat([z_i32, center_i32, two_i32, two_i32], axis=0)
    ends = g.concat([one_i32, center_end, ten_i32, ten_i32], axis=0)
    center_mask_f = g.n("Slice", [g.input, starts, ends], [1, 1, 8, 8], "f32")
    center_mask = g.cast(center_mask_f, "u8")

    two_u8 = g.init(np.array(2, dtype=np.uint8).reshape(1, 1, 1, 1))
    center_plus2 = g.add(center_color, two_u8)
    arm_plus2 = g.add(arm_color, two_u8)
    center_sq_minus1 = g.mul(center_color, center_plus2)
    arm_sq_minus1 = g.mul(arm_color, arm_plus2)
    kernel_src = g.concat([center_color, arm_color, center_sq_minus1, arm_sq_minus1], axis=1)

    diag = np.array(
        [
            [1, 0, 0, 0, 1],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 0, 0, 0, 1],
        ],
        dtype=np.uint8,
    )
    cross = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 0, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=np.uint8,
    )
    feature_masks = np.zeros((2, 4, 5, 5), dtype=np.uint8)
    feature_masks[0, 0] = diag
    feature_masks[0, 1] = cross
    feature_masks[1, 2] = diag
    feature_masks[1, 3] = cross
    kernel_1x2 = qconv(g, kernel_src, g.init(feature_masks), [1, 2, 5, 5], pads=[4, 4, 4, 4])
    kernel_2x1 = g.transpose(kernel_1x2, [1, 0, 2, 3])
    features = qconv(
        g,
        center_mask,
        kernel_2x1,
        [1, 2, 12, 12],
        pads=[4, 4, 4, 4],
        y_zp=qconv.one_u8,
    )
    g.static_value_info = [
        (center_mask_f.name, TensorProto.FLOAT, [1, 1, 8, 8]),
        (center_mask.name, TensorProto.UINT8, [1, 1, 8, 8]),
        (features.name, TensorProto.UINT8, [1, 2, 12, 12]),
    ]

    term_weights = np.zeros((10, 2, 1, 1), dtype=np.int8)
    term_bias = np.zeros((10,), dtype=np.int32)
    for color in range(10):
        shifted = color + 1
        term_weights[color, 0, 0, 0] = 2 * shifted
        term_weights[color, 1, 0, 0] = -1
        term_bias[color] = 1 - shifted * shifted
    term_w = g.init(term_weights)
    term_b = g.init(term_bias)
    term_w_zp = g.init(np.array(0, dtype=np.int8))
    g.n(
        "QLinearConv",
        [features, qconv.scale, qconv.zp_u8, term_w, qconv.scale, term_w_zp, qconv.scale, qconv.zp_u8, term_b],
        [1, 10, 30, 30],
        "u8",
        is_output=True,
        pads=[0, 0, 18, 18],
    )
    return g


if __name__ == "__main__":
    graph = build()
    print(graph.budget())
    graph.save(str(OUT))
    model = onnx.load(OUT)
    for name, dtype, shape in graph.static_value_info:
        model.graph.value_info.append(helper.make_tensor_value_info(name, dtype, shape))
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, OUT)
    print("added static value_info for dynamic Slice/QLinearConv tensors")
