#!/usr/bin/env python3
"""task281 Lane A: certified W=13 shrink of pin fp16 Einsum stamp.

W=13 coord masks, Pad f16 separable terms to 30, per-branch f16 cast (no bool
[3,30] concat). Terminal Einsum unchanged.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runner"))
from ngolf import G, T
from onnx import helper as H

W = 13
M = 30


def build():
    g = G(task=281)
    g.opset = 14

    nonbg = g.init(np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1], np.float32))
    color_m = g.init(np.array([0, 1, 1, 1, 1, 1, 1, 1, 0, 1], np.float32))
    zero_f = g.init(np.array(0.0, np.float32))
    coords = g.init(np.arange(W, dtype=np.int64).reshape(1, W))
    ch_ids = g.init(np.arange(10, dtype=np.uint8).reshape(1, 10))
    topk_k = g.init(np.array([2], np.int64))
    bg_vec = g.init(np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], np.float16))
    batch1 = g.init(np.array([1], np.float16))
    i0 = g.init(np.array([0], np.int64))
    i1 = g.init(np.array([1], np.int64))
    rs_ax = g.init(np.array([0, 1, 3], np.int64))
    pad13_30 = g.init(np.array([0, 0, 0, M - W], np.int64))

    row_nb = g.einsum("nchw,c->h", [g.input, nonbg], [M], "f32")
    col_nb = g.einsum("nchw,c->w", [g.input, nonbg], [M], "f32")
    row_sp = g.greater(row_nb, zero_f)
    col_sp = g.greater(col_nb, zero_f)
    row_u8 = g.cast(row_sp, "u8")
    col_u8 = g.cast(col_sp, "u8")
    new_top = g.argmax(row_u8, 0, keepdims=1)
    new_bot = g.n("ArgMax", [row_u8], [1], "i64", axis=0, keepdims=1, select_last_index=1)
    new_left = g.argmax(col_u8, 0, keepdims=1)
    new_right = g.n("ArgMax", [col_u8], [1], "i64", axis=0, keepdims=1, select_last_index=1)

    row_all = g.n("ReduceSum", [g.input, rs_ax], [M], "f32", keepdims=0)
    row_aw = g.slice(row_all, [0], [W], [0])
    active_w = g.greater(row_aw, zero_f)
    width_f = g.reduce(row_aw, "Max", [0], keepdims=1)
    width_i = g.cast(width_f, "i64")
    active_cols = g.less(coords, width_i)

    counts = g.einsum("nchw,c->c", [g.input, color_m], [10], "f32")
    g.nodes.append(H.make_node(
        "TopK", [counts.name, topk_k.name], ["_top_vals", "top_idx"],
        axis=0, largest=1, sorted=1))
    g.charged.append(T("_top_vals", [2], "f32"))
    top_idx = T("top_idx", [2], "i64")
    g.charged.append(top_idx)
    top_u8 = g.cast(top_idx, "u8")
    outer_u8 = g.gather(top_u8, i0, axis=0)
    inner_u8 = g.gather(top_u8, i1, axis=0)
    outer_eq = g.eq(ch_ids, outer_u8)
    inner_eq = g.eq(ch_ids, inner_u8)
    outer_f = g.cast(outer_eq, "f16")
    inner_f = g.cast(inner_eq, "f16")
    bbox_ch = g.sub(outer_f, bg_vec)
    inner_ch = g.sub(inner_f, outer_f)
    chan_terms = g.concat([bg_vec, bbox_ch, inner_ch], 0)

    active_rs = g.n("Unsqueeze", [active_w, i0], [1, W], "b")
    row_ge = g.n("GreaterOrEqual", [coords, new_top], [1, W], "b")
    row_le = g.n("LessOrEqual", [coords, new_bot], [1, W], "b")
    bbox_rows = g.and_(row_ge, row_le)
    row_gt = g.greater(coords, new_top)
    row_lt = g.less(coords, new_bot)
    inner_rows = g.and_(row_gt, row_lt)
    row13 = g.cast(g.concat([active_rs, bbox_rows, inner_rows], 0), "f16")
    row_terms = g.n("Pad", [row13, pad13_30], [3, M], "f16")

    col_ge = g.n("GreaterOrEqual", [coords, new_left], [1, W], "b")
    col_le = g.n("LessOrEqual", [coords, new_right], [1, W], "b")
    bbox_cols = g.and_(col_ge, col_le)
    col_gt = g.greater(coords, new_left)
    col_lt = g.less(coords, new_right)
    inner_cols = g.and_(col_gt, col_lt)
    col13 = g.cast(g.concat([active_cols, bbox_cols, inner_cols], 0), "f16")
    col_terms = g.n("Pad", [col13, pad13_30], [3, M], "f16")

    g.n("Einsum", [chan_terms, row_terms, col_terms, batch1], [1, 10, M, M], "f16",
        is_output=True, equation="tk,tr,tc,n->nkrc")
    return g


if __name__ == "__main__":
    import onnx
    from onnxsim import simplify

    g = build()
    print(g.budget())
    g.save("task281.onnx", ort_check=False)
    m, _ = simplify(onnx.load("task281.onnx"))
    onnx.save(m, "task281.onnx")