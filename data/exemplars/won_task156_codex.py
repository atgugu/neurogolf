#!/usr/bin/env python3
"""Build task156 ONNX.

Verified rule: two yellow rectangles on a 10x10 grid; keep yellow borders,
fill the smaller interior with color 1 and the larger interior with color 2.

Priced families for this attempt:

A. One-label terminal renderer target (rejected before build): relower c4_u8
   100 + qcode 100 + row logic about 30 + label state 100 + params about 70,
   total about 400-500 and a 2x target.  Legal ONNX lacks a final op that both
   pads the 10x10 label plane and performs per-label equality; one-feature
   QLinearConv/ConvTranspose cannot isolate four nonzero labels with one affine
   threshold per channel.

B. Built family: scorer-native relower + proven two-feature QLinearConv tail.
   Planned memory is old 1030 minus the charged fp32 crop 400 = 630, params 77,
   total 707.  This keeps the separable (state, qcode) classifier while replacing
   the input-access family.
"""

from __future__ import annotations

import sys

import numpy as np
import onnx
from onnx import TensorProto, helper

sys.path.insert(0, "../../runner")
from ngolf import DT, G  # noqa: E402


def main() -> None:
    g = G(task=156)
    g.opset = 17

    c4 = g.relower_onehot_plane(
        channel=4,
        crop=((0, 1), (4, 5), (0, 10), (0, 10)),
        dtype="u8",
    )

    one_scale = g.init(np.array([1.0], np.float32))
    code_scale = g.init(np.array([15.0], np.float32))
    zero_u8 = g.init(np.array([0], np.uint8))
    one_u8 = g.init(np.array([1], np.uint8))

    # 1 for valid background, 2 for yellow border, 3 for strict rectangle interior.
    # Off-canvas terminal padding remains 0.
    k_code = np.ones((1, 1, 3, 3), dtype=np.uint8)
    k_code[0, 0, 1, 1] = 15
    code = g.n(
        "QLinearConv",
        [c4, one_scale, zero_u8, g.init(k_code), one_scale, zero_u8, code_scale, one_u8],
        [1, 1, 10, 10],
        "u8",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
    )

    row_sum = g.n(
        "QLinearConv",
        [
            c4,
            one_scale,
            zero_u8,
            g.init(np.ones((1, 1, 1, 10), dtype=np.uint8)),
            one_scale,
            zero_u8,
            one_scale,
            zero_u8,
        ],
        [1, 1, 10, 1],
        "u8",
        kernel_shape=[1, 10],
    )

    row3 = g.gather(row_sum, [3], axis=2)
    row5 = g.gather(row_sum, [5], axis=2)
    row6 = g.gather(row_sum, [6], axis=2)
    row3_lt_row6 = g.less(row3, row6)
    row5_lt_row3 = g.less(row5, row3)
    row6_pos = g.less(zero_u8, row6)
    tail_case = g.n("Xor", [row5_lt_row3, row6_pos], [1, 1, 1, 1], "b")
    top_small = g.or_(row3_lt_row6, tail_case)

    top_half = g.init(
        np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=np.bool_).reshape(1, 1, 10, 1)
    )

    # First terminal feature is 0 for background, 1 for yellow border, 3 for
    # the smaller interior, and 4 for the larger interior.  Paired with qcode,
    # the class points are linearly separable from the outside pad (0,0).
    three_u8 = g.init(np.array([3], dtype=np.uint8))
    four_u8 = g.init(np.array([4], dtype=np.uint8))
    is_interior = g.eq(code, three_u8)
    top_label = g.where(top_small, three_u8, four_u8)
    bottom_label = g.where(top_small, four_u8, three_u8)
    interior_label = g.where(top_half, top_label, bottom_label)
    state = g.where(is_interior, interior_label, c4)

    features = g.concat([state, code], axis=1)

    # Free terminal renderer over points:
    # outside=(0,0), bg=(0,1), yellow=(1,2), small=(3,3), large=(4,3).
    weights = np.zeros((10, 2, 1, 1), dtype=np.int8)
    bias = np.full((10,), -1, dtype=np.int32)
    weights[0, :, 0, 0] = [-127, 1]
    bias[0] = 0
    weights[4, :, 0, 0] = [-126, 127]
    bias[4] = -127
    weights[1, :, 0, 0] = [-63, 127]
    bias[1] = -191
    weights[2, :, 0, 0] = [1, -1]
    bias[2] = 0
    g.n(
        "QLinearConv",
        [
            features,
            one_scale,
            zero_u8,
            g.init(weights),
            one_scale,
            g.init(np.array([0], dtype=np.int8)),
            one_scale,
            zero_u8,
            g.init(bias),
        ],
        [1, 10, 30, 30],
        "u8",
        is_output=True,
        kernel_shape=[1, 1],
        pads=[0, 0, 20, 20],
    )

    graph = helper.make_graph(
        g.nodes,
        "task156_tagged_relower_qlinear_tail",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", DT[g.out_t.dt][0], g.out_t.shape)],
        initializer=g.inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", g.opset)])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, "task156.onnx")
    print("saved task156.onnx")


if __name__ == "__main__":
    main()
