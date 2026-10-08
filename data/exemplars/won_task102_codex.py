#!/usr/bin/env python3
"""task102 native-state renderer.

Keeps the gated QLinearConv square-interior detector, but replaces the charged
30x30 bool Pad + final Where with a native 12x12 three-plane state.  The final
ConvInteger is the graph output, so its 10-channel 30x30 expansion is free.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runner"))
from ngolf import G


OUT = Path(__file__).resolve().parent / "task102.onnx"


def frame_kernel(k: int) -> np.ndarray:
    if k == 3:
        return np.array(
            [[3, 3, 3], [3, -64, 2], [2, 2, 2]], dtype=np.int8
        ).reshape(1, 1, 3, 3)
    if k == 4:
        return np.array(
            [
                [2, 2, 2, 2],
                [2, -64, -64, 2],
                [2, -64, -64, 2],
                [1, 1, 1, 1],
            ],
            dtype=np.int8,
        ).reshape(1, 1, 4, 4)
    if k == 5:
        return np.array(
            [
                [2, 2, 2, 2, 1],
                [1, -64, -64, -64, 1],
                [1, -64, -64, -64, 1],
                [1, -64, -64, -64, 1],
                [1, 1, 1, 1, 1],
            ],
            dtype=np.int8,
        ).reshape(1, 1, 5, 5)
    if k == 6:
        return np.array(
            [
                [1, 1, 1, 1, 1, 1],
                [1, -64, -64, -64, -64, 1],
                [1, -64, -64, -64, -64, 1],
                [1, -64, -64, -64, -64, 1],
                [1, -64, -64, -64, -64, 1],
                [1, 1, 1, 1, 1, 1],
            ],
            dtype=np.int8,
        ).reshape(1, 1, 6, 6)
    raise ValueError(k)


def terminal_weights() -> np.ndarray:
    w = np.zeros((10, 2, 1, 1), dtype=np.int8)
    # ConvInteger uses x_zero_point=1.  State channels are [gray5 + 1, fill2].
    # Features after zero-point subtraction:
    #   black=[0,-1], gray=[1,-1], red=[0,1], outside=[0,0].
    w[0, :, 0, 0] = [-1, -1]
    w[2, 1, 0, 0] = 1
    w[5, 0, 0, 0] = 1
    return w


def qconv(g: G, x, k: int, out_hw: int):
    q_scale = qconv.q_scale
    x_zero = qconv.x_zero
    w_zero = qconv.w_zero
    return g.n(
        "QLinearConv",
        [
            x,
            q_scale,
            x_zero,
            g.init(frame_kernel(k)),
            q_scale,
            w_zero,
            qconv.y_scale,
            x_zero,
        ],
        [1, 1, out_hw, out_hw],
        "u8",
        kernel_shape=[k, k],
    )


def build() -> None:
    g = G(task=102)
    g.opset = 18

    starts = g.init(np.array([0, 5, 0, 0], dtype=np.int64))
    ends = g.init(np.array([1, 6, 12, 12], dtype=np.int64))
    x5 = g.n("Slice", [g.input, starts, ends], [1, 1, 12, 12], "f32")
    x5u = g.cast(x5, "u8")

    qconv.q_scale = g.init(np.array([0.025316456332802773], dtype=np.float32))
    qconv.y_scale = qconv.q_scale
    qconv.x_zero = g.init(np.array([0], dtype=np.uint8))
    qconv.w_zero = g.init(np.array([0], dtype=np.int8))

    cand1 = qconv(g, x5u, 3, 10)
    cand2 = qconv(g, x5u, 4, 9)
    fill2 = g.n(
        "MaxPool",
        [cand2],
        [1, 1, 10, 10],
        "u8",
        kernel_shape=[2, 2],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
    )
    cand3 = qconv(g, x5u, 5, 8)
    fill3 = g.n(
        "MaxPool",
        [cand3],
        [1, 1, 10, 10],
        "u8",
        kernel_shape=[3, 3],
        pads=[2, 2, 2, 2],
        strides=[1, 1],
    )
    cand4 = qconv(g, x5u, 6, 7)
    fill4 = g.n(
        "MaxPool",
        [cand4],
        [1, 1, 10, 10],
        "u8",
        kernel_shape=[4, 4],
        pads=[3, 3, 3, 3],
        strides=[1, 1],
    )
    fill = g.n("Max", [cand1, fill2, fill3, fill4], [1, 1, 10, 10], "u8")
    fill_x2 = g.add(fill, fill)

    fill12 = g.n(
        "Pad",
        [
            fill_x2,
            g.init(np.array([1, 1, 1, 1], dtype=np.int64)),
            "",
            g.init(np.array([2, 3], dtype=np.int64)),
        ],
        [1, 1, 12, 12],
        "u8",
        mode="constant",
    )
    one_u8 = g.init(np.array([1], dtype=np.uint8))
    x5p1 = g.add(x5u, one_u8)
    state = g.concat([x5p1, fill12], axis=1)
    g.n(
        "ConvInteger",
        [state, g.init(terminal_weights()), one_u8],
        [1, 10, 30, 30],
        "i32",
        is_output=True,
        pads=[0, 0, 18, 18],
    )
    g.save(str(OUT))


if __name__ == "__main__":
    build()
