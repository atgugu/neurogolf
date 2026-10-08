#!/usr/bin/env python3
"""Build task098: hollow filled rectangles with one terminal grouped Conv."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runner"))
from ngolf import G

OUT = Path(__file__).resolve().parent / "task098.onnx"


def kernel() -> tuple[np.ndarray, np.ndarray]:
    w = np.zeros((10, 1, 15, 3), dtype=np.float32)
    b = np.zeros((10,), dtype=np.float32)

    # Black output: a source-certified max box height of 8 means every strict
    # interior has black below it within this vertical window, while padded
    # cells below/right of the true grid do not.
    w[0, 0] = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-7.556, 5.111, -7.556],
            [3.778, 16.444, 3.778],
            [-8.222, 4.444, -8.222],
            [0.667, 0.667, 0.667],
            [0.667, 0.667, 0.667],
            [0.667, 0.667, 0.667],
            [0.667, 0.667, 0.667],
            [0.667, 0.667, 0.667],
            [0.667, 0.667, 0.667],
        ],
        dtype=np.float32,
    )
    b[0] = -1.0

    # Color outputs: keep a colored cell only when it is not surrounded by the
    # same color on all four cardinal sides.
    for color in range(1, 10):
        w[color, 0, 7, 1] = 7.0
        for rr, cc in ((6, 1), (7, 0), (7, 2), (8, 1)):
            w[color, 0, rr, cc] = -2.0

    return w, b


def build() -> None:
    g = G(task=98)
    w, b = kernel()
    g.n(
        "Conv",
        [g.input, g.init(w), g.init(b)],
        [1, 10, 30, 30],
        "f32",
        is_output=True,
        group=10,
        kernel_shape=[15, 3],
        pads=[7, 1, 7, 1],
    )
    g.save(str(OUT))


if __name__ == "__main__":
    build()
