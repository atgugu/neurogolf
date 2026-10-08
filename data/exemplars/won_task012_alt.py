#!/usr/bin/env python3
"""task012 Lane A att02: crop-native ConvInteger terminal renderer.

Certified 12x12 native stamp (dynamic QLinearConv kernel) with a fused terminal
ConvInteger one-hot renderer.  Drops the charged 30x30 sentinel Pad: unstamped
native zeros score only channel 0 via valid/code/code^2 features.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runner"))
from ngolf import G  # noqa: E402

OUT = Path(__file__).with_name("task012.onnx")


def qconv(g: G, x, w, out_shape, name_dt="u8", pads=None):
    attrs = {}
    if pads is not None:
        attrs["pads"] = pads
    return g.n(
        "QLinearConv",
        [x, qconv.q_scale, qconv.q_zp, w, qconv.q_scale, qconv.q_zp, qconv.q_scale, qconv.q_zp],
        out_shape,
        name_dt,
        **attrs,
    )


def build() -> G:
    g = G(task=12)
    qconv.q_scale = g.init(np.array(1.0, dtype=np.float32))
    qconv.q_zp = g.init(np.array(0, dtype=np.uint8))

    w_color = g.init(np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1))
    crange = g.init(np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1))
    two_f = g.init(np.array([2.0], dtype=np.float32).reshape(1, 1, 1, 1))
    eight_f = g.init(np.array([8.0], dtype=np.float32).reshape(1, 1, 1, 1))
    sum_hw = g.init(np.array([2, 3], dtype=np.int64))
    valid_plane = g.init(np.ones((1, 1, 12, 12), dtype=np.uint8))

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
    stamp_masks = np.zeros((1, 2, 5, 5), dtype=np.uint8)
    stamp_masks[0, 0] = diag
    stamp_masks[0, 1] = cross
    stamp_masks_t = g.init(stamp_masks)

    eq_weights = np.zeros((10, 3, 1, 1), dtype=np.int8)
    for color in range(10):
        eq_weights[color, 0, 0, 0] = 1 - color * color
        eq_weights[color, 1, 0, 0] = 2 * color
        eq_weights[color, 2, 0, 0] = -1
    eq_w = g.init(eq_weights)

    cid_a = g.n("Conv", [g.input, w_color], [1, 1, 12, 12], "f32", pads=[0, 0, -18, -18])
    counts = g.n("ReduceSum", [g.input, sum_hw], [1, 10, 1, 1], "f32", keepdims=1)
    center_ch_u8 = g.cast(g.eq(counts, two_f), "u8")
    center_color_u8 = qconv(g, center_ch_u8, crange, [1, 1, 1, 1])
    center_color_f32 = g.cast(center_color_u8, "f32")
    center_mask_u8 = g.cast(g.eq(cid_a, center_color_f32), "u8")

    color_sum = g.n("ReduceSum", [cid_a, sum_hw], [1, 1, 1, 1], "f32", keepdims=1)
    arm_color_u8 = g.cast(
        g.n("Div", [g.sub(color_sum, g.mul(center_color_f32, two_f)), eight_f], [1, 1, 1, 1], "f32"),
        "u8",
    )
    color_pair = g.concat([center_color_u8, arm_color_u8], axis=1)

    kernel_u8 = qconv(g, color_pair, stamp_masks_t, [1, 1, 5, 5], pads=[4, 4, 4, 4])
    result12_u8 = qconv(g, center_mask_u8, kernel_u8, [1, 1, 12, 12], pads=[2, 2, 2, 2])

    code_sq = g.mul(result12_u8, result12_u8)
    features = g.concat([valid_plane, result12_u8, code_sq], axis=1)

    g.n(
        "ConvInteger",
        [features, eq_w],
        [1, 10, 30, 30],
        "i32",
        is_output=True,
        pads=[0, 0, 18, 18],
    )
    return g


if __name__ == "__main__":
    g = build()
    print(g.budget())
    g.save(str(OUT))