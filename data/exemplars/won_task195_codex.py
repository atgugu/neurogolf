#!/usr/bin/env python3
import sys

import numpy as np

sys.path.insert(0, "../../runner")
from ngolf import G  # noqa: E402


def main():
    g = G(task=195)
    g.opset = 12

    # Sample one pixel from each possible 3x3 block phase bucket.
    starts = g.init(np.array([0, 5, 3, 3], dtype=np.int64))
    ends = g.init(np.array([1, 6, 16, 16], dtype=np.int64))
    steps = g.init(np.array([1, 1, 3, 3], dtype=np.int64))
    x5f = g.n("Slice", [g.input, starts, ends, "", steps], [1, 1, 5, 5], "f32")
    x = g.cast(x5f, "u8")

    row_has = g.reduce(x, "Max", [0, 1, 3], keepdims=0)
    col_has = g.reduce(x, "Max", [0, 1, 2], keepdims=0)
    top_i64 = g.argmax(row_has, 0, keepdims=0)
    left_i64 = g.argmax(col_has, 0, keepdims=0)
    top = g.cast(top_i64, "i32")
    left = g.cast(left_i64, "i32")
    offs = g.init(np.array([0, 1, 2], dtype=np.int32))
    row_idx = g.add(top, offs)
    col_idx = g.add(left, offs)
    sample_cols = g.n("Gather", [x, col_idx], [1, 1, 5, 3], "u8", axis=3)
    sprite = g.n("Gather", [sample_cols, row_idx], [1, 1, 3, 3], "u8", axis=2)

    # P⊗P in uint8.  Convert the native mask to signed codes: black=+1,
    # gray=-1, while terminal padding contributes 0.  The final signed QLinearConv
    # thresholds by sign: channel 0 sees +code, channel 5 sees -code.
    a1 = g.n("Unsqueeze", [sprite], [1, 1, 3, 1, 3, 1], "u8", axes=[3, 5])
    a2 = g.n("Unsqueeze", [sprite], [1, 1, 1, 3, 1, 3], "u8", axes=[2, 4])
    kron = g.min_(a1, a2)
    mask = g.reshape(kron, [1, 1, 9, 9])
    mask_i8 = g.cast(mask, "i8")

    scale = g.init(np.array(1.0, dtype=np.float32))
    zp0 = g.init(np.array(0, dtype=np.int8))
    zp1 = g.init(np.array(1, dtype=np.int8))
    wpre = g.init(np.array([[[[-2]]]], dtype=np.int8))
    code = g.n(
        "QLinearConv",
        [mask_i8, scale, zp0, wpre, scale, zp0, scale, zp1],
        [1, 1, 9, 9],
        "i8",
    )

    w = np.zeros((10, 1, 1, 1), dtype=np.int8)
    w[0, 0, 0, 0] = 1
    w[5, 0, 0, 0] = -1
    wt = g.init(w)
    g.n(
        "QLinearConv",
        [code, scale, zp0, wt, scale, zp0, scale, zp0],
        [1, 10, 30, 30],
        "i8",
        is_output=True,
        pads=[0, 0, 21, 21],
    )

    print(g.budget())
    g.save("task195.onnx")


if __name__ == "__main__":
    main()
