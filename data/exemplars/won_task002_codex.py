#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "../../runner")
from ngolf import G


R = 18
W = 20
ALL = (1 << W) - 1


def main() -> None:
    g = G(task=2)

    chan_sel = g.init(np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], np.float32))
    poww = np.zeros(30, dtype=np.float32)
    poww[:W] = (2 ** np.arange(W)).astype(np.float32)
    poww_t = g.init(poww)

    pg = g.einsum("bchw,c,w->bh", [g.input, chan_sel, poww_t], [1, 30], "f32")
    pg20 = g.slice(pg, [0], [W], [1])
    wall20 = g.cast(pg20, "u32")

    wall = g.slice(wall20, [1], [19], [1])
    allones = g.init(np.array([[ALL]], np.uint32))
    walk = g.n("BitwiseXor", [wall, allones], [1, R], "u32")

    side_mask = g.init(np.array([[1 | (1 << (W - 1))]], np.uint32))
    side_seed = g.n("BitwiseAnd", [walk, side_mask], [1, R], "u32")

    tb_idx = g.init(np.array([0, W - 1], np.int32))
    topbot_wall = g.gather(wall20, tb_idx, axis=1, clamp=False)
    topbot_open = g.n("BitwiseXor", [topbot_wall, allones], [1, 2], "u32")
    tb_expand = g.init(np.array([0] * (R - 1) + [1], np.int32))
    topbot_rows = g.gather(topbot_open, tb_expand, axis=1, clamp=False)
    tb_selector = np.zeros((1, R), dtype=np.uint32)
    tb_selector[0, 0] = ALL
    tb_selector[0, -1] = ALL
    topbot_seed = g.n("BitwiseAnd", [topbot_rows, g.init(tb_selector)], [1, R], "u32")
    seed = g.n("BitwiseOr", [side_seed, topbot_seed], [1, R], "u32")
    x = g.n("BitwiseAnd", [walk, seed], [1, R], "u32")

    idx = {
        "up1": g.init(np.array([max(0, i - 1) for i in range(R)], np.int32)),
        "up2": g.init(np.array([max(0, i - 2) for i in range(R)], np.int32)),
        "up4": g.init(np.array([max(0, i - 4) for i in range(R)], np.int32)),
        "dn1": g.init(np.array([min(R - 1, i + 1) for i in range(R)], np.int32)),
        "dn2": g.init(np.array([min(R - 1, i + 2) for i in range(R)], np.int32)),
        "dn4": g.init(np.array([min(R - 1, i + 4) for i in range(R)], np.int32)),
    }
    sh1 = g.init(np.array(1, np.uint32))
    sh2 = g.init(np.array(2, np.uint32))
    sh4 = g.init(np.array(4, np.uint32))

    def band(a, b):
        return g.n("BitwiseAnd", [a, b], [1, R], "u32")

    def bor(a, b):
        return g.n("BitwiseOr", [a, b], [1, R], "u32")

    def gather_rows(t, key):
        return g.gather(t, idx[key], axis=1, clamp=False)

    def shift(t, k, direction):
        return g.n("BitShift", [t, {1: sh1, 2: sh2, 4: sh4}[k]], [1, R], "u32", direction=direction)

    p_u1 = band(walk, gather_rows(walk, "up1"))
    p_u2 = band(p_u1, gather_rows(p_u1, "up2"))
    p_d1 = band(walk, gather_rows(walk, "dn1"))
    p_d2 = band(p_d1, gather_rows(p_d1, "dn2"))
    p_l1 = band(walk, shift(walk, 1, "LEFT"))
    p_l2 = band(p_l1, shift(p_l1, 2, "LEFT"))
    p_r1 = band(walk, shift(walk, 1, "RIGHT"))
    p_r2 = band(p_r1, shift(p_r1, 2, "RIGHT"))

    def vclose(cur, omit=()):
        omit = set(omit)
        for key, prop in (("up1", walk), ("up2", p_u1), ("up4", p_u2)):
            if key not in omit:
                cur = bor(cur, band(prop, gather_rows(cur, key)))
        for key, prop in (("dn1", walk), ("dn2", p_d1), ("dn4", p_d2)):
            if key not in omit:
                cur = bor(cur, band(prop, gather_rows(cur, key)))
        return cur

    def hclose(cur, left=True, omit=()):
        omit = set(omit)
        if left:
            for k, prop in ((1, walk), (2, p_l1), (4, p_l2)):
                if f"l{k}" not in omit:
                    cur = bor(cur, band(prop, shift(cur, k, "LEFT")))
        for k, prop in ((1, walk), (2, p_r1), (4, p_r2)):
            if f"r{k}" not in omit:
                cur = bor(cur, band(prop, shift(cur, k, "RIGHT")))
        return cur

    x = vclose(x, omit={"up1", "dn1"})
    x = hclose(x, left=False)
    x = vclose(x, omit={"up1"})
    x = hclose(x, omit={"r2"})
    x = vclose(x)
    x = hclose(x, omit={"r4"})

    filled = g.n("BitwiseXor", [walk, x], [1, R], "u32")
    sh8 = g.init(np.array(8, np.uint32))
    sh16 = g.init(np.array(16, np.uint32))

    def byte_chunk(src, shift, width):
        if shift:
            src = g.n("BitShift", [src, {8: sh8, 16: sh16}[shift]], [1, R], "u32", direction="RIGHT")
        b = g.cast(src, "u8")
        br = g.reshape(b, [R, 1])
        powb = g.init((2 ** np.arange(width)).astype(np.uint8).reshape(1, width))
        return g.n("BitwiseAnd", [br, powb], [R, width], "u8")

    low = byte_chunk(filled, 0, 8)
    mid = byte_chunk(filled, 8, 8)
    high = byte_chunk(filled, 16, 4)
    unpack_u8 = g.concat([low, mid, high], axis=1)
    fbool = g.cast(unpack_u8, "b")
    pads = g.init(np.array([1, 0, 11, 10], np.int64))
    fpad = g.n("Pad", [fbool, pads], [30, 30], "b")
    yellow = g.init(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], np.float32).reshape(1, 10, 1, 1))
    g.n("Where", [fpad, yellow, g.input], [1, 10, 30, 30], "f32", is_output=True)

    out = Path("task002.onnx")
    g.save(out)


if __name__ == "__main__":
    main()
