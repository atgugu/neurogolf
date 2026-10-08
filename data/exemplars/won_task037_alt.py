#!/usr/bin/env python3
"""task037 — Lane A fallback: native 10x10 projections + ConvInteger terminal.

Keep the banked covariance scatter writer on the certified 10x10 box, then render
one-hot output with a single free ConvInteger (no charged 30x30 Pad).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from onnx import helper as H, TensorProto as TP

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runner"))
from ngolf import G, T  # noqa: E402

PAD, W, SLOTS, OFFSETS = 30, 10, 6, 7
OUT = Path(__file__).with_name("task037.onnx")


def topk(g: G, x: T, k: int, axis: int) -> tuple[T, T]:
    k_t = g.init(np.array([k], dtype=np.int64))
    vals = T(g._nm(), [1, k, 1], x.dt)
    idx = T(g._nm(), [1, k, 1], "i64")
    g.nodes.append(
        H.make_node(
            "TopK",
            [x.name, k_t.name],
            [vals.name, idx.name],
            axis=axis,
            largest=1,
            sorted=0,
        )
    )
    g.charged.extend([vals, idx])
    return vals, idx


def build() -> G:
    g = G(task=37)

    idx_t = g.init(np.arange(PAD, dtype=np.float32).reshape(PAD, 1))
    zero_f = g.init(np.array(0.0, np.float32))
    zero_h = g.init(np.array(0.0, np.float16))
    zero_u8 = g.init(np.array(0, np.uint8))
    two_u8 = g.init(np.array(2, np.uint8))
    nine_u8 = g.init(np.array(9, np.uint8))
    ten_u8 = g.init(np.array(10, np.uint8))
    eleven_u8 = g.init(np.array(11, np.uint8))
    offsets = g.init(np.arange(OFFSETS, dtype=np.uint8).reshape(1, 1, OFFSETS))
    count_bias_arr = np.zeros((1, 10, 1), dtype=np.float32)
    count_bias_arr[0, 0, 0] = -1000.0
    count_bias = g.init(count_bias_arr)
    base_grid = g.init(np.zeros((W * W,), dtype=np.uint8))
    valid_plane = g.init(np.ones((1, 1, W, W), dtype=np.uint8))
    eq_weights = np.zeros((10, 3, 1, 1), dtype=np.int8)
    for color in range(10):
        eq_weights[color, 0, 0, 0] = 1 - color * color
        eq_weights[color, 1, 0, 0] = 2 * color
        eq_weights[color, 2, 0, 0] = -1
    eq_w = g.init(eq_weights)

    rs_all = g.einsum("nchw,hq->ncq", [g.input, idx_t], [1, 10, 1], "f32")
    counts_ranked = g.add(rs_all, count_bias)
    row_vals, color_idx = topk(g, counts_ranked, SLOTS, axis=1)
    cs_all = g.einsum("nchw,wq->ncq", [g.input, idx_t], [1, 10, 1], "f32")
    rcs_all = g.einsum("nchw,hq,wq->ncq", [g.input, idx_t, idx_t], [1, 10, 1], "f32")
    rs_f = row_vals
    cs_f = g.n("GatherElements", [cs_all, color_idx], [1, SLOTS, 1], "f32", axis=1)
    rcs_f = g.n("GatherElements", [rcs_all, color_idx], [1, SLOTS, 1], "f32", axis=1)

    rs = g.cast(rs_f, "f16")
    cs = g.cast(cs_f, "f16")
    rcs = g.cast(rcs_f, "f16")
    two_rcs = g.add(rcs, rcs)
    rs_cs = g.mul(rs, cs)
    cov = g.sub(two_rcs, rs_cs)
    main = g.greater(cov, zero_h)
    abs_cov = g.n("Abs", [cov], cov.shape, "f16")
    span_f = g.n("Sqrt", [abs_cov], abs_cov.shape, "f16")
    span = g.cast(span_f, "u8")
    rs_u8 = g.cast(rs, "u8")
    cs_u8 = g.cast(cs, "u8")
    r0_num = g.sub(rs_u8, span)
    r0 = g.n("Div", [r0_num, two_u8], r0_num.shape, "u8")
    minc_num = g.sub(cs_u8, span)
    maxc_num = g.add(cs_u8, span)
    topc_num = g.where(main, minc_num, maxc_num)
    topc = g.n("Div", [topc_num, two_u8], topc_num.shape, "u8")
    present = g.greater(row_vals, zero_f)
    slot_color = g.cast(color_idx, "u8")
    slot_color_present = g.where(present, slot_color, zero_u8)
    r0x10 = g.mul(r0, ten_u8)
    base_linear = g.add(r0x10, topc)
    diag_step = g.where(main, eleven_u8, nine_u8)
    offset_clamped = g.min_(offsets, span)
    offset_delta = g.mul(offset_clamped, diag_step)
    linear_cols = g.add(base_linear, offset_delta)
    updates_3d = g.tile(slot_color_present, [1, 1, OFFSETS])
    linear_idx_flat = g.reshape(linear_cols, [SLOTS * OFFSETS])
    linear_idx = g.cast(linear_idx_flat, "i32")
    updates = g.reshape(updates_3d, [SLOTS * OFFSETS])
    scalar_flat = g.n(
        "ScatterElements",
        [base_grid, linear_idx, updates],
        [W * W],
        "u8",
        axis=0,
        reduction="max",
    )
    scalar_small = g.reshape(scalar_flat, [W, W])
    code_sq_small = g.mul(scalar_small, scalar_small)
    code = g.reshape(scalar_small, [1, 1, W, W])
    code_sq = g.reshape(code_sq_small, [1, 1, W, W])
    features = g.concat([valid_plane, code, code_sq], axis=1)
    g.n(
        "ConvInteger",
        [features, eq_w],
        [1, 10, PAD, PAD],
        "i32",
        is_output=True,
        pads=[0, 0, PAD - W, PAD - W],
    )
    return g


def main() -> None:
    g = build()
    print("Paper budget (native idx + ConvInteger terminal):")
    print("  params: idx[10,1]=10, constants~120, eq_weights[10,3,1,1]=30 => ~160")
    print("  memory: features[1,3,10,10] u8=300, scatter idx i32[42]=168, scatter out=100,")
    print("          reshape[10,10]=100, TopK i64[1,6,1]=48, slot planes 6x7 u8 x5=210,")
    print("          einsum f32[1,10,1]x3=120, gather f32[1,6,1]x2=48, moment f16 ~72")
    print("  paper total ~160 + ~1166 = ~1326 (vs pin 2169)")
    print(g.budget())
    g.save(str(OUT))


if __name__ == "__main__":
    main()