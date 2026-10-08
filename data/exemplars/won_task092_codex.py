#!/usr/bin/env python3
"""task092 attempt11 compressed valid-base terminal.

TRUE RULE: connect same-color endpoint pairs horizontally or vertically, with
vertical segments taking precedence at crossings.

Attempt #11 prices before build:

1. Pairwise terminal contraction, targeting the 2x band:
   input-pair final Einsum would avoid endpoint tensors, but exact intervals
   need equality/lower/upper coordinate factors. Even shared E/L/R factors are
   3 * [30,30] = 2700 params before color coefficients, above the 1437 target
   and above the 2473 registration ceiling. Not built.

2. Compressed valid-base terminal, built here:
   keep the verified moment/bbox endpoint extractor, prepend a rank-1 valid
   rectangle slot, and render all six slots in one terminal fp16 Einsum.
   Paper budget:
     extractor and five active stick metadata          ~1410 B
     six-slot interval factors [1,6,30] bool/f16       ~1440 B
     scalar h/w from coordinate moments                 ~84 B
     unified six-slot signed coeff path                ~214 B
     params                                            ~125 B
   The margin is tight; direct six-slot intervals avoid rebuilding att09's
   full row/column validity scans.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper as H
from onnx import numpy_helper, shape_inference


HERE = Path(__file__).resolve().parent
SRC = HERE / "pin_task092.onnx"
OUT = HERE / "task092.onnx"


def arr(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(value, name=name)


def n(op: str, inputs: list[str], output: str, **attrs) -> onnx.NodeProto:
    return H.make_node(op, inputs, [output], **attrs)


def build() -> onnx.ModelProto:
    model = onnx.load(SRC)

    keep_outputs = {
        "sum_x_f32",
        "sum_x",
        "sum_y_f32",
        "sum_y",
        "sum_x2_f32",
        "sum_x2",
        "sum_y2_f32",
        "sum_y2",
        "two_sum_x2",
        "sum_x_sq",
        "diff_x_sq",
        "two_sum_y2",
        "sum_y_sq",
        "diff_y_sq",
    }

    nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if not node.output or node.output[0] not in keep_outputs:
            continue
        nodes.append(node)

    nodes.extend(
        [
            n("Equal", ["diff_y_sq", "zero_h"], "h_same_y"),
            n("Equal", ["diff_x_sq", "zero_h"], "v_same_x"),
            n("Xor", ["h_same_y", "v_same_x"], "stick_b"),
            n("Cast", ["stick_b"], "stick_score", to=TP.FLOAT16),
            H.make_node("TopK", ["stick_score", "k_five"], ["stick_score_top", "stick_color_i64"]),
            n("Cast", ["stick_color_i64"], "stick_color_u8_flat", to=TP.UINT8),
            n("GatherElements", ["sum_x", "stick_color_i64"], "sum_x5", axis=1),
            n("GatherElements", ["sum_y", "stick_color_i64"], "sum_y5", axis=1),
            n("GatherElements", ["diff_x_sq", "stick_color_i64"], "diff_x_sq5", axis=1),
            n("GatherElements", ["diff_y_sq", "stick_color_i64"], "diff_y_sq5", axis=1),
            n("Sqrt", ["diff_x_sq5"], "diff_x5"),
            n("Sqrt", ["diff_y_sq5"], "diff_y5"),
            n("Sub", ["sum_x5", "diff_x5"], "left_twice5"),
            n("Sub", ["sum_y5", "diff_y5"], "top_twice5"),
            n("Mul", ["left_twice5", "half_h"], "left_f5"),
            n("Mul", ["top_twice5", "half_h"], "top_f5"),
            n("Add", ["left_f5", "half_h"], "left_f_round5"),
            n("Add", ["top_f5", "half_h"], "top_f_round5"),
            n("Cast", ["left_f_round5"], "left5_u8", to=TP.UINT8),
            n("Cast", ["top_f_round5"], "top5_u8", to=TP.UINT8),
            n("Cast", ["sum_x5"], "sum_x5_u8", to=TP.UINT8),
            n("Cast", ["sum_y5"], "sum_y5_u8", to=TP.UINT8),
            n("Sub", ["sum_x5_u8", "left5_u8"], "right5_u8"),
            n("Sub", ["sum_y5_u8", "top5_u8"], "bottom5_u8"),
            n("Reshape", ["left5_u8", "shape_1_5_1"], "left_151"),
            n("Reshape", ["right5_u8", "shape_1_5_1"], "right_151"),
            n("Reshape", ["top5_u8", "shape_1_5_1"], "top_151"),
            n("Reshape", ["bottom5_u8", "shape_1_5_1"], "bottom_151"),
            n("Sub", ["bottom_151", "top_151"], "r_wid"),
            n("Sub", ["right_151", "left_151"], "c_wid"),
            n("Equal", ["left_151", "right_151"], "is_vert_b"),
            n("ReduceSum", ["input", "axes_all_i64"], "n_cells_f32", keepdims=0),
            n("Cast", ["n_cells_f32"], "n_cells_h", to=TP.FLOAT16),
            n("ReduceSum", ["sum_x", "axes_bc_i64"], "sx_all_h", keepdims=0),
            n("ReduceSum", ["sum_y", "axes_bc_i64"], "sy_all_h", keepdims=0),
            n("Add", ["sx_all_h", "sx_all_h"], "sx2_h"),
            n("Add", ["sy_all_h", "sy_all_h"], "sy2_h"),
            n("Div", ["sx2_h", "n_cells_h"], "w_minus_one_h"),
            n("Div", ["sy2_h", "n_cells_h"], "h_minus_one_h"),
            n("Cast", ["w_minus_one_h"], "w_last_u8", to=TP.UINT8),
            n("Cast", ["h_minus_one_h"], "h_last_u8", to=TP.UINT8),
            n("Reshape", ["w_last_u8", "shape_1_1_1"], "w_last_111"),
            n("Reshape", ["h_last_u8", "shape_1_1_1"], "h_last_111"),
            n("Concat", ["zero_111_u8", "top_151"], "top6", axis=1),
            n("Concat", ["h_last_111", "r_wid"], "r_wid6", axis=1),
            n("Sub", ["coord30_u8", "top6"], "r_off6"),
            n("LessOrEqual", ["r_off6", "r_wid6"], "row6_b"),
            n("Concat", ["zero_111_u8", "left_151"], "left6", axis=1),
            n("Concat", ["w_last_111", "c_wid"], "c_wid6", axis=1),
            n("Sub", ["coord30_u8", "left6"], "c_off6"),
            n("LessOrEqual", ["c_off6", "c_wid6"], "col6_b"),
            n("Cast", ["row6_b"], "row6_f", to=TP.FLOAT16),
            n("Cast", ["col6_b"], "col6_f", to=TP.FLOAT16),
            n("Reshape", ["stick_color_u8_flat", "shape_1_5_1"], "stick_color_151"),
            n("Concat", ["zero_111_u8", "stick_color_151"], "color_161", axis=1),
            n("Equal", ["color_161", "palette_1_1_10"], "color_oh6"),
            n("Where", ["is_vert_b", "three_f16", "two_f16"], "stick_own5"),
            n("Concat", ["one_111_f16", "stick_own5"], "stick_own6", axis=1),
            n("Where", ["color_oh6", "stick_own6", "neg_one_f16"], "coeff_f"),
            H.make_node(
                "Einsum",
                ["row6_f", "col6_f", "coeff_f"],
                ["output"],
                equation="bsy,bsx,bsc->bcyx",
            ),
        ]
    )

    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.value_info[:]

    used = {inp for node in nodes for inp in node.input if inp}
    kept = [init for init in model.graph.initializer if init.name in used]
    extras = [
        arr("two_f16", np.array(2, dtype=np.float16)),
        arr("three_f16", np.array(3, dtype=np.float16)),
        arr("one_111_f16", np.ones((1, 1, 1), dtype=np.float16)),
        arr("zero_111_u8", np.zeros((1, 1, 1), dtype=np.uint8)),
        arr("axes_all_i64", np.array([0, 1, 2, 3], dtype=np.int64)),
        arr("axes_bc_i64", np.array([0, 1], dtype=np.int64)),
        arr("shape_1_1_1", np.array([1, 1, 1], dtype=np.int64)),
    ]
    kept.extend(init for init in extras if init.name in used)
    del model.graph.initializer[:]
    model.graph.initializer.extend([init for init in kept if init.name in used])

    del model.graph.output[:]
    model.graph.output.extend([H.make_tensor_value_info("output", TP.FLOAT16, [1, 10, 30, 30])])

    model.ir_version = 10
    model = shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> int:
    onnx.save(build(), OUT)
    print(f"saved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
