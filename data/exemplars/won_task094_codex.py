#!/usr/bin/env python3
import sys

import numpy as np

sys.path.insert(0, "../../runner")
from ngolf import G


def qconv(g, x, weight, out_shape, kernel_shape, y_scale=None):
    """Uint8 QLinearConv with shared zero-points; y_scale=43 binarizes <=20 vs >=22."""
    if not hasattr(g, "_q_scale1"):
        g._q_scale1 = g.init(np.array([1.0], np.float32))
        g._q_zp = g.init(np.array([0], np.uint8))
        g._q_scale43 = g.init(np.array([43.0], np.float32))
    w = g.init(np.asarray(weight, np.uint8))
    scale_y = g._q_scale43 if y_scale == 43 else g._q_scale1
    return g.n(
        "QLinearConv",
        [x, g._q_scale1, g._q_zp, w, g._q_scale1, g._q_zp, scale_y, g._q_zp],
        out_shape,
        "u8",
        kernel_shape=kernel_shape,
    )


def main():
    g = G(task=94)

    # The generator keeps every blue 5x5 outline inside rows/cols 1..13; center
    # rows/cols are 3..11. Relower only that certified crop.
    blue = g.relower_onehot_plane(
        channel=1,
        crop=((0, 1), (1, 2), (1, 14), (1, 14)),
        dtype="u8",
    )

    ones_row = np.ones((1, 1, 1, 13), np.uint8)
    ones_col = np.ones((1, 1, 13, 1), np.uint8)
    row_counts = qconv(g, blue, ones_row, [1, 1, 13, 1], [1, 13])
    col_counts = qconv(g, blue, ones_col, [1, 1, 1, 13], [13, 1])

    # Existing source-safe profile: 2*edge_row + center_row + 2*edge_row == 22
    # at a hollow-square center and <=20 elsewhere, even when two row bands merge.
    row_w = np.array([2, 0, 1, 0, 2], np.uint8).reshape(1, 1, 5, 1)
    col_w = np.array([2, 0, 1, 0, 2], np.uint8).reshape(1, 1, 1, 5)
    row_score = qconv(g, row_counts, row_w, [1, 1, 9, 1], [5, 1], y_scale=43)
    col_score = qconv(g, col_counts, col_w, [1, 1, 1, 9], [1, 5], y_scale=43)

    row_hit = g.cast(row_score, "b")
    col_hit = g.cast(col_score, "b")

    row30 = g.n(
        "Pad",
        [row_hit, g.init([0, 0, 3, 0, 0, 0, 18, 0], "i64")],
        [1, 1, 30, 1],
        "b",
        mode="constant",
    )
    col30 = g.n(
        "Pad",
        [col_hit, g.init([0, 0, 0, 3, 0, 0, 0, 18], "i64")],
        [1, 1, 1, 30],
        "b",
        mode="constant",
    )

    row_state_f = g.cast(g.concat([g.not_(row30), row30], axis=1), "f32")
    col_state_f = g.cast(g.concat([g.not_(col30), col30], axis=1), "f32")

    # Source palette is only blue/cyan. Use a semantic 3-color tail
    # (blue, pink, cyan) and a final selector to avoid a dense 10-channel table.
    source_sel = np.zeros((2, 10), np.float32)
    source_sel[0, 1] = 1.0
    source_sel[1, 8] = 1.0

    trans = np.zeros((2, 2, 3, 2), np.float32)
    for a in range(2):
        for b in range(2):
            line = bool(a or b)
            trans[a, b, 0, 0] = 1.0
            trans[a, b, 1 if line else 2, 1] = 1.0

    out_sel = np.zeros((10, 3), np.float32)
    out_sel[1, 0] = 1.0
    out_sel[6, 1] = 1.0
    out_sel[8, 2] = 1.0

    g.n(
        "Einsum",
        [
            g.input,
            g.init(source_sel),
            row_state_f,
            col_state_f,
            g.init(trans),
            g.init(out_sel),
        ],
        [1, 10, 30, 30],
        "f32",
        is_output=True,
        equation="nchw,dc,nahr,nbsw,abed,ke->nkhw",
    )

    print(g.budget())
    g.save("task094.onnx")


if __name__ == "__main__":
    main()
