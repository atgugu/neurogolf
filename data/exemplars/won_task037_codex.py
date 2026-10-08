#!/usr/bin/env python3
"""task037 - terminal bilinear Einsum diagonal connector.

The generator gives a certified 10x10 grid.  Each nonzero color has two
endpoints on one 45-degree diagonal; the output paints the closed segment.

This build replaces the TopK/scatter/QLinearConv family with a zero-memory
terminal Einsum.  The Einsum pairs same-color input cells, tests whether each
native output coordinate lies between them along either diagonal orientation,
then pads to the 30x30 contract inside the same final op.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runner"))
from ngolf import G  # noqa: E402

OUT = Path(__file__).with_name("task037.onnx")
W, PAD, MAX_STEP = 10, 30, 7


def build() -> G:
    g = G(task=37)

    # P maps the full 30x30 one-hot contract to the certified native 10x10
    # coordinates, and maps native output coordinates back into the top-left
    # 10x10 region of the terminal 30x30 output.
    project = np.zeros((W, PAD), dtype=np.float32)
    for i in range(W):
        project[i, i] = 1.0
    p = g.init(project)

    # shift[d, x, y] is true when y = x + d for d in 0..6.  Reusing this table
    # on rows and transformed columns is the whole compression.
    shift = np.zeros((MAX_STEP, W, W), dtype=np.float32)
    for d in range(MAX_STEP):
        for x in range(W - d):
            shift[d, x, x + d] = 1.0
    s = g.init(shift)

    # Orientation 0 keeps columns unchanged (down-right).  Orientation 1 flips
    # columns so the same positive-shift test also covers down-left diagonals.
    orient = np.zeros((2, W, W), dtype=np.float32)
    for x in range(W):
        orient[0, x, x] = 1.0
        orient[1, x, W - 1 - x] = 1.0
    m = g.init(orient)

    # Channel map: color k paints output channel k.  Background channel 0 is
    # produced by background self-pairs, while colored line terms strongly
    # suppress it wherever a nonzero segment is present.
    chan = np.zeros((10, 10), dtype=np.float32)
    chan[0, 0] = 1.0
    for k in range(1, 10):
        chan[k, k] = 1.0
        chan[0, k] = -1000.0
    c = g.init(chan)

    # Indices:
    # input endpoints: (a,b) and (e,f), native projections x/y/z/w
    # native output: (u,v), padded output: (r,c)
    # d/g are distances from top endpoint to output and output to bottom endpoint
    # orientation maps columns y/v/w into p/q/t so both diagonal slopes share shift.
    # Operand order matters for ORT's Einsum planner.  The 30x30 inputs are
    # immediately projected to native row/col coordinates before endpoint pairs
    # are combined; the algebra is identical to the compact formula above.
    eq = "nkab,xa,yb,nkef,ze,wf,dxu,syp,dpq,guz,swt,gqt,ok,svq,vc,ur->norc"
    g.n(
        "Einsum",
        [g.input, p, p, g.input, p, p, s, m, s, s, m, s, c, m, p, p],
        [1, 10, PAD, PAD],
        "f32",
        is_output=True,
        equation=eq,
    )
    return g


def main() -> None:
    g = build()
    print("Paper budget (terminal bilinear Einsum):")
    print("  P[10,30]=300, shift[7,10,10]=700, orient[2,10,10]=200, channel[10,10]=100")
    print("  charged node outputs: none; final Einsum emits the full 10-channel 30x30 output")
    print(g.budget())
    g.save(str(OUT))


if __name__ == "__main__":
    main()
