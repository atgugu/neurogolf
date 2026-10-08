#!/usr/bin/env python3
"""task190: tagged u8 crop + QLinearConv ray detect + dynamic ConvInteger terminal."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runner"))
from ngolf import G  # noqa: E402

W_ZP = 128
KERNEL = np.array(
    [
        [0, 33, -17, -8, -6, -4, -1, 10, -12, 22, 28, 10, 0, -7, -4, -6, -16, 15, -6],
        [3, 0, 24, -3, -10, -6, -10, 0, 5, 20, -35, 0, -3, 0, -12, -4, 0, 0, 8],
        [0, 7, 0, 11, 7, -8, 11, -5, 0, 7, 0, 29, 23, -14, 7, 19, 0, 15, -35],
        [-41, 21, -11, 0, 2, 6, -25, 19, -6, 34, -41, 26, -20, -2, 9, 0, -13, 7, -6],
        [15, -16, -1, -20, 0, -16, 15, -26, 28, 18, 25, -26, 8, -18, 0, -17, 3, -20, 28],
        [-51, 42, 0, -7, 15, 0, -11, 42, -44, 14, -51, 46, -9, 0, 12, -5, 1, 36, -44],
        [0, 2, 0, 0, 2, 11, 0, 10, -26, -95, 0, 0, 0, 10, -6, 7, 0, 0, 0],
        [-44, 40, 5, -9, 0, 0, -7, 52, -50, 14, -44, 42, 0, 0, 5, -11, -3, 42, -44],
        [13, -22, -8, -14, 0, -12, 13, -30, 17, 34, 21, -30, 3, -17, 0, -15, 10, -18, 5],
        [-32, 15, -2, 0, 3, -2, -14, 19, -25, 45, -40, 24, -14, 2, 0, 0, -5, 7, -13],
        [0, -2, -1, 13, 11, -14, 21, -11, 0, 0, 0, 3, 11, -8, 16, 14, 0, 0, 0],
        [23, 0, 20, -11, -15, -8, 0, 0, 0, 15, -15, 0, 1, -18, -16, -18, 22, 0, 3],
        [0, 12, -18, -1, 2, -5, 0, 0, 35, 0, 33, 7, 0, 0, 8, 0, -17, 14, 0],
    ],
    dtype=np.int16,
)


def build() -> G:
    g = G(task=190)

    # Dynamic terminal weights: input background minus ray -> output background;
    # ray -> selected color. Padding stays zero because both features pad with zero.
    bg_w = np.zeros((10, 1, 1, 1), dtype=np.int8)
    bg_w[0, 0, 0, 0] = 1
    bg_twice = np.zeros((1, 10, 1, 1), dtype=np.int8)
    bg_twice[0, 0, 0, 0] = 2

    scale1 = g.init(np.array(1.0, np.float32))
    zp0_u = g.init(np.array(0, np.uint8))
    w_zp = g.init(np.array(W_ZP, np.uint8))
    ray_w = g.init((KERNEL + W_ZP).astype(np.uint8).reshape(1, 1, 13, 19))
    bg10_u = g.relower_onehot_plane(channel=0, crop=((0, 1), (0, 1), (0, 10), (0, 10)), dtype="u8")
    ray_score = g.n(
        "QLinearConv",
        [bg10_u, scale1, zp0_u, ray_w, scale1, w_zp, scale1, zp0_u],
        [1, 1, 10, 10],
        "u8",
        pads=[6, 9, 6, 9],
    )
    pooled = g.n("MaxPool", [g.input], [1, 10, 1, 1], "f32", kernel_shape=[30, 30], pads=[0, 0, 0, 0])
    pooled_i = g.cast(pooled, "i8")
    ray_row = g.sub(pooled_i, g.init(bg_twice))
    ray_dyn = g.transpose(ray_row, [1, 0, 2, 3])
    dyn_w = g.concat([g.init(bg_w), ray_dyn], axis=1)
    features = g.concat([bg10_u, ray_score], axis=1)
    g.n(
        "ConvInteger",
        [features, dyn_w],
        [1, 10, 30, 30],
        "i32",
        pads=[0, 0, 20, 20],
        is_output=True,
    )
    return g


if __name__ == "__main__":
    g = build()
    print(g.budget())
    g.save("task190.onnx")
