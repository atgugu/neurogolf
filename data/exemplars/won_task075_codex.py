#!/usr/bin/env python3
"""task075 — crop-native 3-feature native state with fused terminal ConvInteger renderer."""
import sys

sys.path.insert(0, "../../runner")
from ngolf import G
import numpy as np

g = G(task=75)
g.opset = 18

sel3_arr = np.zeros((3, 30), dtype=np.float32)
sel3_arr[:, :3] = np.eye(3, dtype=np.float32)
sel3 = g.init(sel3_arr)
# Store only output colors as points on a convex polygon in three masked feature
# coordinates.  Blue is never an output color, so channel 1 stays nonpositive.
bit_values = np.array([7, 56, 64], dtype=np.uint8)
output_colors = [0, 2, 3, 4, 5, 6, 7, 8, 9]
code_xy = np.array([
    [1, 2],
    [4, 0],
    [5, 0],
    [7, 1],
    [7, 2],
    [6, 7],
    [2, 7],
    [0, 6],
    [0, 4],
], dtype=np.uint8)
codes_u8 = np.zeros(10, dtype=np.uint8)
for color, (x, y) in zip(output_colors, code_xy):
    codes_u8[color] = np.uint8(64 + x + 8 * y)
color_codes = codes_u8.astype(np.float32)
color_weights = g.init(color_codes)
one_k = g.init(np.array([1.0], dtype=np.float32))

templ_f = g.einsum("bchw,c,ph,qw,k->bkpq", [g.input, color_weights, sel3, sel3, one_k], [1, 1, 3, 3], "f32")
templ_u8 = g.cast(templ_f, "u8")
templ_flat = g.reshape(templ_u8, [1, 9, 1, 1])

markers_f = g.slice(g.input, [1, 1, 5], [2, 8, 14], [1, 2, 3], [1, 3, 3])
mask = g.cast(markers_f, "b")
black_code = g.init(np.array([codes_u8[0]], dtype=np.uint8))
blocks = g.where(mask, templ_flat, black_code)

right_scalar = g.n("DepthToSpace", [blocks], [1, 1, 9, 9], "u8", blocksize=3, mode="DCR")

gray_code = np.uint8(codes_u8[5])
gray_top = g.init(np.full((1, 1, 3, 1), gray_code, dtype=np.uint8))
bottom = g.init(np.array([[[codes_u8[0], codes_u8[0], codes_u8[0], gray_code]] * 6], dtype=np.uint8).reshape(1, 1, 6, 4))
top = g.concat([templ_u8, gray_top], axis=3)
left_scalar = g.concat([top, bottom], axis=2)
scalar13 = g.concat([left_scalar, right_scalar], axis=3)

bit_masks = g.init(bit_values.reshape(1, 3, 1, 1))
features = g.n("BitwiseAnd", [scalar13, bit_masks], [1, 3, 9, 13], "u8")

weights = np.zeros((10, 3, 1, 1), dtype=np.int8)
classifier_rows = np.array([
    [-16,  -2,   1],
    [-13, -105,  1],
    [ 15, -95,  -1],
    [ 85,  -6,  -8],
    [ 98,   2, -11],
    [ 12,  93, -82],
    [-16,  83, -72],
    [-82,   5,  -3],
    [-87,  -3,   2],
], dtype=np.int8)
for color, row in zip(output_colors, classifier_rows):
    weights[color, :, 0, 0] = row
conv_weights = g.init(weights)
g.n("ConvInteger", [features, conv_weights], [1, 10, 30, 30], "i32", is_output=True, pads=[0, 0, 21, 17])

print(g.budget())
g.save("task075.onnx")
