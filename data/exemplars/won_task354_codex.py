#!/usr/bin/env python3
"""Build task354 row-band color-id flood model.

Rule from ARC-GEN ddf7fa4f: three row-0 colored lights recolor the gray
rectangle whose horizontal span contains that light.  The generator fixes the
real grid to 10x10, with gray rectangles only on rows 2..9.

Tail strategy: keep the exact native color-id flood, then use a terminal
ConvInteger to both pad 10x10 to 30x30 and expand color ids to thresholdable
10-channel output.  For id x, channel k receives
    -x^2 + 2*k*x + (1 - k*k)
which is positive only when x == k for x in 0..9.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


FP = TensorProto.FLOAT
U8 = TensorProto.UINT8
BOOL = TensorProto.BOOL
I32 = TensorProto.INT32
OPSET = 14
IR = 10


def init(name: str, arr, dtype) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr, dtype=dtype), name=name)


def terminal_w() -> np.ndarray:
    w = np.zeros((10, 3, 1, 1), dtype=np.int8)
    for k in range(10):
        w[k, 0, 0, 0] = 2 * k
        w[k, 1, 0, 0] = -1
        w[k, 2, 0, 0] = 1 - k * k
    return w


def add_flood(nodes: list[onnx.NodeProto], gray: str, keyrow: str) -> str:
    """Four exact radius-1 geodesic horizontal flood steps, width max is 5."""
    nodes.append(helper.make_node("Mul", [keyrow, gray], ["seed"]))
    prev = "seed"
    for i in range(4):
        pooled = f"mp{i + 1}"
        masked = f"f{i + 1}"
        nodes.append(
            helper.make_node(
                "MaxPool",
                [prev],
                [pooled],
                kernel_shape=[1, 3],
                pads=[0, 1, 0, 1],
                strides=[1, 1],
            )
        )
        nodes.append(helper.make_node("Mul", [pooled, gray], [masked]))
        prev = masked
    return prev


def make_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits = [
        init("gray_st", [5, 2, 0], np.int64),
        init("gray_en", [6, 10, 10], np.int64),
        init("chw_axes", [1, 2, 3], np.int64),
        init("left_st", [1, 0, 0], np.int64),
        init("left_en", [10, 1, 3], np.int64),
        init("mid_st", [1, 0, 4], np.int64),
        init("mid_en", [10, 1, 6], np.int64),
        init("right_st", [1, 0, 7], np.int64),
        init("right_en", [10, 1, 10], np.int64),
        init("conv_w", np.arange(1, 10, dtype=np.float32).reshape(1, 9, 1, 1), np.float32),
        init("zgap", np.zeros((1, 1, 1, 1), dtype=np.uint8), np.uint8),
        init("zero_row", np.zeros((1, 1, 1, 10), dtype=np.uint8), np.uint8),
        init("one_u8", np.uint8(1), np.uint8),
        init("term_w", terminal_w(), np.int8),
    ]

    nodes.append(helper.make_node("Slice", ["input", "gray_st", "gray_en", "chw_axes"], ["gray_f"]))
    nodes.append(helper.make_node("Cast", ["gray_f"], ["gray"], to=U8))

    for name in ("left", "mid", "right"):
        nodes.append(
            helper.make_node(
                "Slice",
                ["input", f"{name}_st", f"{name}_en", "chw_axes"],
                [f"{name}_oh"],
            )
        )
        nodes.append(helper.make_node("Conv", [f"{name}_oh", "conv_w"], [f"{name}_cid_f"]))
        nodes.append(helper.make_node("Cast", [f"{name}_cid_f"], [f"{name}_cid"], to=U8))

    nodes.append(
        helper.make_node(
            "Concat",
            ["left_cid", "zgap", "mid_cid", "zgap", "right_cid"],
            ["row0_cid"],
            axis=3,
        )
    )
    flood = add_flood(nodes, "gray", "row0_cid")
    nodes.append(helper.make_node("Concat", ["row0_cid", "zero_row", flood], ["cidx10"], axis=2))
    nodes.append(helper.make_node("Mul", ["cidx10", "cidx10"], ["cidx10_sq"]))
    nodes.append(helper.make_node("Clip", ["cidx10", "one_u8", "one_u8"], ["ones10"]))
    nodes.append(helper.make_node("Concat", ["cidx10", "cidx10_sq", "ones10"], ["poly_state"], axis=1))
    nodes.append(helper.make_node("ConvInteger", ["poly_state", "term_w"], ["output"], pads=[0, 0, 20, 20]))

    graph = helper.make_graph(
        nodes,
        "task354_rowbands",
        [helper.make_tensor_value_info("input", FP, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", I32, [1, 10, 30, 30])],
        initializer=inits,
    )
    model = helper.make_model(graph, ir_version=IR, opset_imports=[helper.make_opsetid("", OPSET)])
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


def main() -> None:
    onnx.save(make_model(), "task354.onnx")
    print("wrote task354.onnx")
    print(
        'FAMILY-BUDGET-JSON: {"tensors":[{"shape":[1,1,8,10],"dtype":"f32","count":1},'
        '{"shape":[1,9,1,3],"dtype":"f32","count":2},'
        '{"shape":[1,9,1,2],"dtype":"f32","count":1},'
        '{"shape":[1,1,10,10],"dtype":"u8","count":3},'
        '{"shape":[1,3,10,10],"dtype":"u8","count":1},'
        '{"shape":[1,1,8,10],"dtype":"u8","count":10},'
        '{"shape":[1,1,1,10],"dtype":"u8","count":1},'
        '{"shape":[1,1,1,3],"dtype":"f32","count":2},'
        '{"shape":[1,1,1,2],"dtype":"f32","count":1},'
        '{"shape":[1,1,1,3],"dtype":"u8","count":2},'
        '{"shape":[1,1,1,2],"dtype":"u8","count":1}],'
        '"params":88}'
    )


if __name__ == "__main__":
    main()
