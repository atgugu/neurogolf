#!/usr/bin/env python3
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def init(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name=name)


def main():
    inits = [
        init("coord", np.arange(30, dtype=np.float32)),
        init("one_f", np.array([[1.0]], dtype=np.float32)),
        init("ch8", np.array([8], dtype=np.int64)),
        init("two_u8", np.array([[2]], dtype=np.uint8)),
        init("four_u8", np.array([[4]], dtype=np.uint8)),
        init("shape_1122", np.array([1, 1, 2, 2], dtype=np.int64)),
        init("shape_1", np.array([1], dtype=np.int64)),
        init("zero2_u8", np.array([0, 0], dtype=np.uint8)),
        init("eight_u8", np.array([8], dtype=np.uint8)),
        init("ten_u8", np.array(10, dtype=np.uint8)),
    ]

    w = np.zeros((10, 2, 1, 1), dtype=np.int8)
    b = np.zeros((10,), dtype=np.int32)
    for k in range(10):
        target = 10 if k == 0 else k
        w[k, 0, 0, 0] = 2 * target
        w[k, 1, 0, 0] = -1
        b[k] = 1 - target * target
    inits.extend(
        [
            init("q_w", w),
            init("q_b", b),
            init("x_scale", np.array(1.0, dtype=np.float32)),
            init("w_scale", np.array(1.0, dtype=np.float32)),
            init("y_scale", np.array(1.0, dtype=np.float32)),
            init("x_zp", np.array(0, dtype=np.uint8)),
            init("w_zp", np.array(0, dtype=np.int8)),
            init("y_zp", np.array(0, dtype=np.uint8)),
        ]
    )

    n = []
    add = n.append
    add(helper.make_node("Einsum", ["input", "coord"], ["row_pos"], equation="nchw,h->nc"))
    add(helper.make_node("Einsum", ["input", "coord"], ["col_pos"], equation="nchw,w->nc"))
    add(helper.make_node("Einsum", ["input"], ["total"], equation="nchw->nc"))
    add(helper.make_node("Equal", ["total", "one_f"], ["present"]))
    add(helper.make_node("Cast", ["row_pos"], ["row_u8"], to=TensorProto.UINT8))
    add(helper.make_node("Cast", ["col_pos"], ["col_u8"], to=TensorProto.UINT8))
    add(helper.make_node("Gather", ["row_u8", "ch8"], ["cyan_row_sum"], axis=1))
    add(helper.make_node("Gather", ["col_u8", "ch8"], ["cyan_col_sum"], axis=1))
    add(helper.make_node("Max", ["cyan_row_sum", "two_u8"], ["cyan_row_safe"]))
    add(helper.make_node("Max", ["cyan_col_sum", "two_u8"], ["cyan_col_safe"]))
    add(helper.make_node("Sub", ["cyan_row_safe", "two_u8"], ["row_minus2"]))
    add(helper.make_node("Sub", ["cyan_col_safe", "two_u8"], ["col_minus2"]))
    add(helper.make_node("Div", ["row_minus2", "four_u8"], ["row0"]))
    add(helper.make_node("Div", ["col_minus2", "four_u8"], ["col0"]))
    # Fold absent colors to the cyan origin so quadrant ArgMax cannot choose them.
    add(helper.make_node("Where", ["present", "row_u8", "row0"], ["row_fold"]))
    add(helper.make_node("Where", ["present", "col_u8", "col0"], ["col_fold"]))
    add(helper.make_node("Less", ["row_fold", "row0"], ["row_lt"]))
    add(helper.make_node("Greater", ["row_fold", "row0"], ["row_gt"]))
    add(helper.make_node("Less", ["col_fold", "col0"], ["col_lt"]))
    add(helper.make_node("Greater", ["col_fold", "col0"], ["col_gt"]))

    for name, a, b_ in [
        ("nw", "row_lt", "col_lt"),
        ("ne", "row_lt", "col_gt"),
        ("sw", "row_gt", "col_lt"),
        ("se", "row_gt", "col_gt"),
    ]:
        add(helper.make_node("And", [a, b_], [f"{name}_mask"]))
        add(helper.make_node("Cast", [f"{name}_mask"], [f"{name}_u8mask"], to=TensorProto.UINT8))
        add(helper.make_node("ArgMax", [f"{name}_u8mask"], [f"{name}_color_i64"], axis=1, keepdims=1))
        add(helper.make_node("Cast", [f"{name}_color_i64"], [f"{name}_color"], to=TensorProto.UINT8))

    add(helper.make_node("Concat", ["nw_color", "ne_color", "sw_color", "se_color"], ["colors14"], axis=1))
    add(helper.make_node("Reshape", ["colors14", "shape_1122"], ["block2"]))

    add(helper.make_node("Reshape", ["row0", "shape_1"], ["row_pad_u8"]))
    add(helper.make_node("Reshape", ["col0", "shape_1"], ["col_pad_u8"]))
    add(helper.make_node("Sub", ["eight_u8", "row_pad_u8"], ["bottom_pad_u8"]))
    add(helper.make_node("Sub", ["eight_u8", "col_pad_u8"], ["right_pad_u8"]))
    add(
        helper.make_node(
            "Concat",
            ["zero2_u8", "row_pad_u8", "col_pad_u8", "zero2_u8", "bottom_pad_u8", "right_pad_u8"],
            ["pads_u8"],
            axis=0,
        )
    )
    add(helper.make_node("Cast", ["pads_u8"], ["pads"], to=TensorProto.INT64))
    add(helper.make_node("Pad", ["block2", "pads", "ten_u8"], ["color10"]))
    add(helper.make_node("Mul", ["color10", "color10"], ["color10_sq"]))
    add(helper.make_node("Concat", ["color10", "color10_sq"], ["features"], axis=1))
    add(
        helper.make_node(
            "QLinearConv",
            ["features", "x_scale", "x_zp", "q_w", "w_scale", "w_zp", "y_scale", "y_zp", "q_b"],
            ["output"],
            pads=[0, 0, 20, 20],
        )
    )

    value_info = [
        helper.make_tensor_value_info("color10", TensorProto.UINT8, [1, 1, 10, 10]),
        helper.make_tensor_value_info("color10_sq", TensorProto.UINT8, [1, 1, 10, 10]),
        helper.make_tensor_value_info("features", TensorProto.UINT8, [1, 2, 10, 10]),
    ]
    graph = helper.make_graph(
        n,
        "task342_qconv_colorcode",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.UINT8, [1, 10, 30, 30])],
        inits,
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, "task342.onnx")


if __name__ == "__main__":
    main()
