#!/usr/bin/env python3
"""task294 scalar QLinearConv tail.

The input is one-hot. Slice the 10x10 gray mask, cast it to uint8, and double
it so QLinearConv sees values:
  background: 0 - x_zero_point(1) = -1
  gray:       2 - x_zero_point(1) = +1
  off-canvas padding: x_zero_point padding = 0 contribution

The terminal quantized 3x3 classifier then emits channel 0 for background,
channel 2 for strict gray interiors, and channel 5 for gray borders.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def init(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name=name)


def vi(name, dtype, shape):
    return helper.make_tensor_value_info(name, dtype, shape)


def qweights():
    w = np.zeros((10, 1, 3, 3), dtype=np.int8)
    b = np.full((10,), -1, dtype=np.int32)

    # ch0: background has signed center value -1; gray is +1; off-canvas is 0.
    w[0, 0, 1, 1] = -2
    b[0] = -1

    # ch2/ch5 are the minimum-margin integer separators over the full
    # source-generated set for this signed scalar representation.
    w[2, 0] = np.array(
        [
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
        ],
        dtype=np.int8,
    )
    b[2] = -4

    # ch5: gray center, but not strict interior.
    w[5, 0] = np.array(
        [
            [0, 0, -2],
            [0, 6, 0],
            [-2, 0, 0],
        ],
        dtype=np.int8,
    )
    b[5] = -3
    return w, b


def build(path="task294.onnx"):
    w, b = qweights()
    nodes = [
        helper.make_node("Slice", ["input", "starts", "ends", "axes"], ["gray_f"]),
        helper.make_node("Cast", ["gray_f"], ["gray_u8"], to=TensorProto.UINT8),
        helper.make_node("Add", ["gray_u8", "gray_u8"], ["gray2"]),
        helper.make_node(
            "QLinearConv",
            [
                "gray2",
                "scale",
                "x_zp",
                "qw",
                "scale",
                "w_zp",
                "scale",
                "y_zp",
                "qb",
            ],
            ["output"],
            kernel_shape=[3, 3],
            pads=[1, 1, 21, 21],
        ),
    ]
    inits = [
        init("starts", np.array([5, 0, 0], dtype=np.int64)),
        init("ends", np.array([6, 10, 10], dtype=np.int64)),
        init("axes", np.array([1, 2, 3], dtype=np.int64)),
        init("scale", np.array(1.0, dtype=np.float32)),
        init("x_zp", np.array(1, dtype=np.uint8)),
        init("w_zp", np.array(0, dtype=np.int8)),
        init("y_zp", np.array(0, dtype=np.uint8)),
        init("qw", w),
        init("qb", b),
    ]
    graph = helper.make_graph(
        nodes,
        "task294_scalar_qconv",
        [vi("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [vi("output", TensorProto.UINT8, [1, 10, 30, 30])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, path)


if __name__ == "__main__":
    build()
