"""task303 — paired row/column frontier features with factored terminal Einsum.

Rule: all-black active rows/columns are frontiers; recolor their union to red and
preserve every other input cell.
"""
import sys

import numpy as np

sys.path.insert(0, "../../runner")
from ngolf import G


g = G(task=303)

# p=0: count non-black cells in the row/column.
# p=1: count active cells in the row/column, zero in padding.
feat = np.zeros((2, 10, 1), np.float32)
feat[0, 1:, 0] = 1.0
feat[1, :, 0] = 1.0
feat_t = g.init(feat)

# Pair the row/column feature index in the terminal renderer:
#   p=0 gives nr*nc, positive only away from full-black frontiers.
#   p=1 gives active-row * active-column, used to light valid frontier cells red.
coeff = np.zeros((2, 10, 10), np.float32)
for c in range(10):
    coeff[0, c, c] = 1.0
    coeff[0, 2, c] -= 1000.0
    coeff[1, 2, c] = 1.0
coeff_t = g.init(coeff)

row_feat = g.einsum("nchw,pcl->nphl", [g.input, feat_t], [1, 2, 30, 1])
col_feat = g.einsum("nchw,pcl->nplw", [g.input, feat_t], [1, 2, 1, 30])
g.n(
    "Einsum",
    [g.input, row_feat, col_feat, coeff_t],
    [1, 10, 30, 30],
    "f32",
    is_output=True,
    equation="nchw,nphx,npxw,poc->nohw",
)

print(g.budget())
g.save("task303.onnx")
