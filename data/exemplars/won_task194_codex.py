#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper as H
from onnx import numpy_helper


OUT = "task194.onnx"


def init(name: str, arr: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr), name=name)


def vi(name: str, dtype: int, shape: list[int]) -> onnx.ValueInfoProto:
    return H.make_tensor_value_info(name, dtype, shape)


def build() -> onnx.ModelProto:
    # ConvTranspose with asymmetric crop pads reads only the top-left 3x3
    # while summing the one-hot color channel into its native color id.
    color_w = np.arange(10, dtype=np.float32).reshape(10, 1, 1, 1)

    block_idx = np.array(
        [
            [0, 1, 2, 6, 3, 0],
            [3, 4, 5, 7, 4, 1],
            [6, 7, 8, 8, 5, 2],
            [2, 5, 8, 8, 7, 6],
            [1, 4, 7, 5, 4, 3],
            [0, 3, 6, 2, 1, 0],
        ],
        dtype=np.int32,
    )

    nodes = [
        H.make_node(
            "ConvTranspose",
            ["input", "color_w"],
            ["color_f"],
            kernel_shape=[1, 1],
            pads=[0, 0, 27, 27],
            strides=[1, 1],
        ),
        H.make_node("Cast", ["color_f"], ["color_u8"], to=TP.UINT8),
        H.make_node("Reshape", ["color_u8", "flat_shape"], ["flat"]),
        H.make_node("Gather", ["flat", "block_idx"], ["tile6"], axis=2),
        H.make_node("Equal", ["tile6", "arange10"], ["hot6"]),
        H.make_node("Pad", ["hot6", "pads_hw", "", "axes_hw"], ["output"], mode="constant"),
    ]

    inits = [
        init("color_w", color_w),
        init("flat_shape", np.array([1, 1, 9], dtype=np.int64)),
        init("block_idx", block_idx),
        init("arange10", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)),
        init("pads_hw", np.array([0, 0, 24, 24], dtype=np.int64)),
        init("axes_hw", np.array([2, 3], dtype=np.int64)),
    ]

    graph = H.make_graph(
        nodes,
        "task194_color_id",
        [vi("input", TP.FLOAT, [1, 10, 30, 30])],
        [vi("output", TP.BOOL, [1, 10, 30, 30])],
        initializer=inits,
        value_info=[
            vi("color_f", TP.FLOAT, [1, 1, 3, 3]),
            vi("color_u8", TP.UINT8, [1, 1, 3, 3]),
            vi("flat", TP.UINT8, [1, 1, 9]),
            vi("tile6", TP.UINT8, [1, 1, 6, 6]),
            vi("hot6", TP.BOOL, [1, 10, 6, 6]),
        ],
    )
    model = H.make_model(graph, opset_imports=[H.make_operatorsetid("", 18)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


if __name__ == "__main__":
    onnx.save(build(), OUT)
    print(f"saved {OUT}")
