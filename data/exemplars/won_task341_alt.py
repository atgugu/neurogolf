#!/usr/bin/env python3
"""task341 — certified bridge with sparse two-channel terminal update."""
import sys
sys.path.insert(0, "../../runner")
from ngolf import G
import numpy as np

g = G(task=341)
g.opset = 16


def reduce_sum_in(x, axis_init, ax):
    sh = list(x.shape)
    sh[ax] = 1
    return g.n("ReduceSum", [x, axis_init], sh, x.dt, keepdims=1)


bg = g.slice(g.input, [0, 1, 1], [1, 9, 9], [1, 2, 3])
eight = g.init(np.array(8.0, np.float32))
axis2 = g.init(np.array([2], np.int64))
axis3 = g.init(np.array([3], np.int64))

row_bg = reduce_sum_in(bg, axis3, 3)
col_bg = reduce_sum_in(bg, axis2, 2)
row_empty = g.eq(row_bg, eight)
col_empty = g.eq(col_bg, eight)
row_min = g.reduce(row_bg, "Min", [2])
col_min = g.reduce(col_bg, "Min", [3])
row_dual = g.eq(row_bg, row_min)
col_dual = g.eq(col_bg, col_min)

s0 = g.init(np.array([0], np.int64))
s1 = g.init(np.array([1], np.int64))
s7 = g.init(np.array([7], np.int64))
e6 = g.init(np.array([6], np.int64))
e8 = g.init(np.array([8], np.int64))
ax2 = axis2
ax3 = axis3

row_d0 = g.n("Slice", [row_dual, s0, e6, ax2], [1, 1, 6, 1], "b")
row_d2 = g.n("Slice", [row_dual, ax2, e8, ax2], [1, 1, 6, 1], "b")
row_inner6 = g.and_(row_d0, row_d2)
col_d0 = g.n("Slice", [col_dual, s0, e6, ax3], [1, 1, 1, 6], "b")
col_d2 = g.n("Slice", [col_dual, ax2, e8, ax3], [1, 1, 1, 6], "b")
col_inner6 = g.and_(col_d0, col_d2)

row_empty6 = g.n("Slice", [row_empty, s1, s7, ax2], [1, 1, 6, 1], "b")
row_empty6_u8 = g.cast(row_empty6, "u8")
any_row_empty_u8 = g.reduce(row_empty6_u8, "Max", [2])
any_row_empty = g.cast(any_row_empty_u8, "b")
row_edge0 = g.n("Slice", [row_empty, s0, s1, ax2], [1, 1, 1, 1], "b")
row_edge7 = g.n("Slice", [row_empty, s7, e8, ax2], [1, 1, 1, 1], "b")
row_edge_empty = g.or_(row_edge0, row_edge7)
row_edges_clear = g.not_(row_edge_empty)
vertical = g.and_(any_row_empty, row_edges_clear)
horizontal = g.not_(vertical)

col_empty6 = g.n("Slice", [col_empty, s1, s7, ax3], [1, 1, 1, 6], "b")
bridge_rows_v = g.and_(vertical, row_empty6)
bridge_rows_h = g.and_(horizontal, row_inner6)
bridge_rows6 = g.or_(bridge_rows_v, bridge_rows_h)
bridge_cols_v = g.and_(vertical, col_inner6)
bridge_cols_h = g.and_(horizontal, col_empty6)
bridge_cols6 = g.or_(bridge_cols_v, bridge_cols_h)
bridge6 = g.and_(bridge_rows6, bridge_cols6)

bridge_f32 = g.cast(bridge6, "f32")
delta_weights = g.init(np.array([[[[-1.0]]], [[[1.0]]]], np.float32))
updates = g.mul(bridge_f32, delta_weights)

idx = np.zeros((2, 1, 6, 6, 4), dtype=np.int64)
for k, ch in enumerate([0, 8]):
    idx[k, 0, :, :, 1] = ch
    for r in range(6):
        for c in range(6):
            idx[k, 0, r, c, 2] = r + 2
            idx[k, 0, r, c, 3] = c + 2
scatter_idx = g.init(idx)
g.n("ScatterND", [g.input, scatter_idx, updates], [1, 10, 30, 30], "f32",
    is_output=True, reduction="add")

print(g.budget())
g.save("task341.onnx")
