#!/usr/bin/env python3
"""task368 certified native-size color-id core + quadratic terminal renderer.

The pin's rule core is already compact: decode the 10x10 color-id plane, find
sprite anchors, extract the colored 4x4 template, and paste it at every anchor.
This build keeps that verified color-id algorithm, but replaces the charged
30x30 sentinel Pad + Equal renderer with a free final ConvInteger classifier.

For an integer color id x, each output channel k uses:
    -x*x + 2*k*x + (1 - k*k)*valid
which is positive exactly when x == k.  The valid plane is an initializer in the
10x10 native frame, so the final ConvInteger's bottom/right padding produces
zeros outside the real canvas instead of channel-0 activations.
"""
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = Path(__file__).resolve().parent / "task368.onnx"


def init(name: str, arr) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr), name=name)


def build() -> onnx.ModelProto:
    ramp_w = np.zeros((1, 10, 2, 2), dtype=np.float32)
    ramp_w[0, :, 0, 0] = np.arange(10, dtype=np.float32)

    quad_w = np.zeros((10, 3, 1, 1), dtype=np.int8)
    for k in range(10):
        quad_w[k, :, 0, 0] = [-1, 2 * k, 1 - k * k]

    inits = [
        init("ramp_w", ramp_w),
        init("anchor_w", np.array([[[[0, -10], [-10, 1]]]], dtype=np.int8)),
        init("q_scale", np.array(1.0, dtype=np.float32)),
        init("zero", np.array(0, dtype=np.uint8)),
        init("zero_i8", np.array(0, dtype=np.int8)),
        init("one", np.array(1, dtype=np.uint8)),
        init("five", np.array(5, dtype=np.uint8)),
        init("valid10", np.ones((1, 1, 10, 10), dtype=np.uint8)),
        init("quad_w", quad_w),
    ]

    nodes = [
        helper.make_node("Conv", ["input", "ramp_w"], ["idx_full"], dilations=[20, 20]),
        helper.make_node("Cast", ["idx_full"], ["idx_u8"], to=TensorProto.UINT8),
        helper.make_node(
            "QLinearConv",
            ["idx_u8", "q_scale", "zero", "anchor_w", "q_scale", "zero_i8", "q_scale", "zero"],
            ["anchor_score_u8"],
            pads=[1, 1, 0, 0],
        ),
        helper.make_node("Min", ["anchor_score_u8", "one"], ["anchor_u8"]),
        helper.make_node(
            "QLinearConv",
            ["anchor_u8", "q_scale", "zero", "idx_u8", "q_scale", "five", "q_scale", "five"],
            ["dyn_kernel"],
            kernel_shape=[10, 10],
            pads=[3, 3, 0, 0],
        ),
        helper.make_node(
            "QLinearConv",
            ["anchor_u8", "q_scale", "zero", "dyn_kernel", "q_scale", "zero", "q_scale", "zero"],
            ["paint_u8"],
            kernel_shape=[4, 4],
            pads=[3, 3, 0, 0],
        ),
        helper.make_node("Mul", ["paint_u8", "paint_u8"], ["paint_sq"]),
        helper.make_node("Concat", ["paint_sq", "paint_u8", "valid10"], ["quad_feat"], axis=1),
        helper.make_node(
            "ConvInteger",
            ["quad_feat", "quad_w", "zero", "zero_i8"],
            ["output"],
            kernel_shape=[1, 1],
            pads=[0, 0, 20, 20],
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "task368_quad",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.INT32, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 14)])
    model.ir_version = 10
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    onnx.save(build(), OUT)
    print(f"saved {OUT}")
