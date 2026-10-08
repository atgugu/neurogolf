#!/usr/bin/env python3
"""Task336: tagged u8 relower + score-direct terminal QLinearConv renderer.

The score-direct renderer is the current full-gated behavior oracle. This
attempt changes the representation family by using ngolf's tagged one-hot
relower for the fixed 10x10 gray crop, so the transient fp32 Slice is not a
charged tensor under the grader's native relower rule.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runner"))
from ngolf import G

PIN = Path(__file__).resolve().parent / "pin_ref.onnx"
OUT = Path(__file__).resolve().parent / "task336.onnx"
KERNEL = 11


def load_pin_weights() -> tuple[np.ndarray, np.ndarray]:
    weights = {t.name: numpy_helper.to_array(t) for t in onnx.load(str(PIN)).graph.initializer}
    return weights["q_w"], weights["q_b"]


def terminal_weights() -> np.ndarray:
    eff = np.zeros((10, 2, 1, 1), dtype=np.int16)
    eff[0, :, 0, 0] = [-1, -1]
    eff[5, :, 0, 0] = [1, -1]
    eff[8, :, 0, 0] = [-1, 1]
    return (eff + 128).astype(np.uint8)


def terminal_bias() -> np.ndarray:
    b = np.zeros((10,), dtype=np.int32)
    b[0] = -3
    return b


def build() -> None:
    q_w, q_b = load_pin_weights()
    g = G(task=336)
    g.opset = 18

    c5u = g.relower_onehot_plane(
        channel=5,
        crop=((0, 1), (5, 6), (0, 10), (0, 10)),
        dtype="u8",
    )

    one = g.init(1.0, "f32")
    x_zp = g.init(0, "u8")
    w_zp = g.init(128, "u8")
    y_zp = g.init(0, "u8")

    score = g.n(
        "QLinearConv",
        [c5u, one, x_zp, g.init(q_w), one, w_zp, one, y_zp, g.init(q_b)],
        [1, 1, 10, 10],
        "u8",
        pads=[KERNEL // 2] * 4,
    )
    state = g.concat([c5u, score], axis=1)

    g.n(
        "QLinearConv",
        [
            state,
            one,
            g.init(2, "u8"),
            g.init(terminal_weights()),
            one,
            w_zp,
            one,
            y_zp,
            g.init(terminal_bias()),
        ],
        [1, 10, 30, 30],
        "u8",
        is_output=True,
        pads=[0, 0, 20, 20],
    )
    print(g.budget())
    g.save(str(OUT))


if __name__ == "__main__":
    build()
