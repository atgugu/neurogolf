#!/usr/bin/env python3
"""task093: QLinearMatMul half-count solver with zero-point ConvInteger tail.

Rule: orient to the gray separator, count background cells in each fixed
seven-cell half, pack all non-background cells toward the separator, and emit
only colors 0 and 5.

Priced replacement families:

Family A, direct terminal Einsum target (paper <=858 but rejected before build):
  tensor/param budget:
    gate/orientation scalars                 f32/b   <=16 B
    direct output contraction                output  free
    required dense orientation selectors      params  >= 2*2*14*30*2 = 3360
    channel/rank features                     params  >= 60
  total: >=3420 params before output, so the advertised <=858 cannot close in
  legal ONNX unless the horizontal/vertical conditional transpose is solved
  without dense selectors.

Family B, built QLinearMatMul/relower pivot:
  planned charged tensors:
    ch0_u8                       uint8 [1,1,14,14] = 196 B
    row_bg2_u8                   uint8 [1,1,14,2]  = 28 B
    col_bg2_raw_u8               uint8 [1,1,2,14]  = 28 B
    col_bg2_u8                   uint8 [1,1,14,2]  = 28 B
    bg2_u8                       uint8 [1,1,14,2]  = 28 B
    bg2_cell_u8                  uint8 [1,1,14,14] = 196 B
    rank_score_u8                uint8 [1,1,14,14] = 196 B
    outT_u8                      uint8 [1,1,14,14] = 196 B
    state2_u8                    uint8 [1,1,14,14] = 196 B
    orientation probe tensors    mixed              = 12 B
    params: relower 8 + top probe 8 + QMM 64 + side/threshold 28
            + tail 11 = 119
  paper total: about 1211 B, below the 1476 registration cutoff and a different
  representation from the banked selector-Einsum body.
"""

import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, "../../runner")
from ngolf import relower_onehot_plane  # scorer-recognized one-hot crop relower


H = W = 14
HALF = 7
ZP = 14


def init(name, arr, dtype=None):
    return numpy_helper.from_array(np.asarray(arr, dtype=dtype), name=name)


def add_relower_ch0(nodes, inits):
    rel_nodes, rel_inits, _ = relower_onehot_plane(
        "input",
        "ch0_u8",
        channel=0,
        crop=((0, 1), (0, 1), (0, H), (0, W)),
        dtype="u8",
        starts_name="s_ch0",
        ends_name="e_ch0",
    )
    nodes.extend(rel_nodes)
    for name, arr in rel_inits:
        inits.append(init(name, arr, np.int64))


def build(out_path="task093.onnx"):
    qmm_b = np.zeros((H, 2), dtype=np.uint8)
    qmm_b[:HALF, 0] = 2
    qmm_b[HALF:W, 1] = 2
    qmm_a = qmm_b.T.copy()

    side_idx = np.zeros((W,), dtype=np.int64)
    side_idx[HALF:] = 1

    thresholds = np.zeros((1, 1, 1, W), dtype=np.uint8)
    thresholds[0, 0, 0, :HALF] = ZP - 1 + 2 * np.arange(1, HALF + 1, dtype=np.uint8)
    thresholds[0, 0, 0, HALF:] = ZP - 1 + 2 * np.arange(HALF, 0, -1, dtype=np.uint8)

    tail_w = np.zeros((10, 1, 1, 1), dtype=np.int8)
    tail_w[0, 0, 0, 0] = -1
    tail_w[5, 0, 0, 0] = 1

    inits = [
        init("s_top5", [0, 5, 0, 6], np.int64),
        init("e_top5", [1, 6, 1, 8], np.int64),
        init("qmm_b", qmm_b),
        init("qmm_a", qmm_a),
        init("q_scale", np.array(1.0, dtype=np.float32)),
        init("q_zp", np.array(0, dtype=np.uint8)),
        init("side_idx", side_idx),
        init("thresh_zp_u8", thresholds),
        init("x_zp", np.array(ZP, dtype=np.uint8)),
        init("tail_w", tail_w),
    ]

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    out = helper.make_tensor_value_info("output", TensorProto.INT32, [1, 10, 30, 30])
    nodes = []
    n = nodes.append

    # Vertical-band probe: after transpose, gray appears at top row columns 6/7.
    n(helper.make_node("Slice", ["input", "s_top5", "e_top5"], ["top5_f"]))
    n(helper.make_node("Cast", ["top5_f"], ["top5_u8"], to=TensorProto.UINT8))
    n(helper.make_node("ReduceMax", ["top5_u8"], ["vert_u8"], axes=[2, 3], keepdims=1))
    n(helper.make_node("Cast", ["vert_u8"], ["vert_b"], to=TensorProto.BOOL))

    # Native one-hot channel-0 crop, scorer-tagged so the transient f32 Slice is free.
    add_relower_ch0(nodes, inits)

    # Background counts times two in each fixed half. QLinearMatMul keeps the
    # count tensors uint8 and avoids the int32 MatMulInteger activation bill.
    qmm_common = ["q_scale", "q_zp", "q_scale", "q_zp", "q_scale", "q_zp"]
    n(helper.make_node(
        "QLinearMatMul",
        ["ch0_u8", *qmm_common[:2], "qmm_b", *qmm_common[2:]],
        ["row_bg2_u8"],
    ))
    n(helper.make_node(
        "QLinearMatMul",
        ["qmm_a", *qmm_common[:2], "ch0_u8", *qmm_common[2:]],
        ["col_bg2_raw_u8"],
    ))
    n(helper.make_node("Transpose", ["col_bg2_raw_u8"], ["col_bg2_u8"], perm=[0, 1, 3, 2]))
    n(helper.make_node("Where", ["vert_b", "row_bg2_u8", "col_bg2_u8"], ["bg2_u8"]))

    # Expand the selected side count across the seven cells of each half, then
    # subtract from the zero-point shifted rank thresholds.
    n(helper.make_node("Gather", ["bg2_u8", "side_idx"], ["bg2_cell_u8"], axis=3))
    n(helper.make_node("Sub", ["thresh_zp_u8", "bg2_cell_u8"], ["rank_score_u8"]))
    n(helper.make_node("Transpose", ["rank_score_u8"], ["outT_u8"], perm=[0, 1, 3, 2]))
    n(helper.make_node("Where", ["vert_b", "rank_score_u8", "outT_u8"], ["state2_u8"]))
    n(helper.make_node(
        "ConvInteger",
        ["state2_u8", "tail_w", "x_zp"],
        ["output"],
        pads=[0, 0, 30 - H, 30 - W],
    ))

    graph = helper.make_graph(nodes, "task093_qlinear_half_counts", [inp], [out], inits)
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 14)])
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, out_path)
    print(out_path)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "task093.onnx")
