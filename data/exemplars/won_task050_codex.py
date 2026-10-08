#!/usr/bin/env python3
"""task050: relowered att8 MaxPool equality-fold fallback.

True rule: cyan endpoints stay cyan, and cells strictly between two cyan
endpoints sharing a row or column become green.
"""

import sys

import numpy as np

sys.path.insert(0, "../../runner")
from ngolf import G  # noqa: E402


def maxpool(g, x, kernel_shape, pads):
    return g.n(
        "MaxPool",
        [x],
        x.shape,
        x.dt,
        kernel_shape=kernel_shape,
        pads=pads,
    )


def main():
    g = G(task=50)
    g.opset = 18

    cyan = g.relower_onehot_plane(
        channel=8,
        crop=((0, 1), (8, 9), (0, 15), (0, 15)),
        dtype="u8",
    )

    left = maxpool(g, cyan, [1, 15], [0, 14, 0, 0])
    right = maxpool(g, cyan, [1, 15], [0, 0, 0, 14])
    up = maxpool(g, cyan, [15, 1], [14, 0, 0, 0])
    down = maxpool(g, cyan, [15, 1], [0, 0, 14, 0])

    row_between = g.min_(left, right)
    col_between = g.min_(up, down)
    keep15 = g.eq(row_between, col_between)

    pads = g.init(np.array([0, 0, 15, 15], dtype=np.int64))
    axes = g.init(np.array([2, 3], dtype=np.int64))
    keep30 = g.n(
        "Pad",
        [keep15, pads, g.init(np.array(True, dtype=np.bool_)), axes],
        [1, 1, 30, 30],
        "b",
        mode="constant",
    )

    green = np.zeros((1, 10, 1, 1), dtype=np.float32)
    green[0, 3, 0, 0] = 1.0
    g.n(
        "Where",
        [keep30, g.input, g.init(green)],
        [1, 10, 30, 30],
        "f32",
        is_output=True,
    )

    g.save("task050.onnx")


if __name__ == "__main__":
    main()
