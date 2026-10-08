#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


sys.path.insert(0, "../../runner")
from ngolf import relower_onehot_plane  # noqa: E402


OUT = Path(__file__).resolve().parent / "task273.onnx"


def init(name: str, arr) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr), name=name)


def node(op: str, inputs: list[str], outputs: list[str], **attrs) -> onnx.NodeProto:
    return helper.make_node(op, inputs, outputs, **attrs)


def main() -> None:
    relower_nodes, relower_inits, _ = relower_onehot_plane(
        "input",
        "yellow10",
        channel=4,
        crop=((0, 1), (4, 5), (0, 10), (0, 10)),
        dtype="u8",
        starts_name="yellow_starts",
        ends_name="yellow_ends",
    )

    h_w = np.zeros((1, 1, 1, 9), dtype=np.uint8)
    h_w[0, 0, 0, 0:4] = 1
    h_w[0, 0, 0, 5:9] = 2

    v_w = np.zeros((1, 1, 9, 1), dtype=np.uint8)
    v_w[0, 0, 0:4, 0] = 1
    v_w[0, 0, 5:9, 0] = 2

    render_w = np.zeros((10, 3, 1, 1), dtype=np.int8)
    render_b = np.zeros((10,), dtype=np.int32)
    render_w[0, :, 0, 0] = [-2, -2, 1]
    render_w[2, :, 0, 0] = [1, 0, 0]
    render_w[4, :, 0, 0] = [0, 1, 0]

    inits = [
        *(init(name, arr) for name, arr in relower_inits),
        init("q_scale", np.array(1.0, dtype=np.float32)),
        init("u8_zero", np.array(0, dtype=np.uint8)),
        init("i8_zero", np.array(0, dtype=np.int8)),
        init("u8_three", np.array(3, dtype=np.uint8)),
        init("h_w", h_w),
        init("v_w", v_w),
        init("one10", np.ones((1, 1, 10, 10), dtype=np.uint8)),
        init("render_w", render_w),
        init("render_b", render_b),
    ]

    q_inputs_h = [
        "yellow10",
        "q_scale",
        "u8_zero",
        "h_w",
        "q_scale",
        "u8_zero",
        "q_scale",
        "u8_zero",
    ]
    q_inputs_v = [
        "hspan10",
        "q_scale",
        "u8_zero",
        "v_w",
        "q_scale",
        "u8_zero",
        "q_scale",
        "u8_zero",
    ]
    q_inputs_render = [
        "feat10",
        "q_scale",
        "u8_zero",
        "render_w",
        "q_scale",
        "i8_zero",
        "q_scale",
        "u8_zero",
        "render_b",
    ]

    nodes = [
        *relower_nodes,
        node(
            "QLinearConv",
            q_inputs_h,
            ["hbits10"],
            kernel_shape=[1, 9],
            pads=[0, 4, 0, 4],
        ),
        node("Equal", ["hbits10", "u8_three"], ["hspan_bool10"]),
        node("Cast", ["hspan_bool10"], ["hspan10"], to=TensorProto.UINT8),
        node(
            "QLinearConv",
            q_inputs_v,
            ["vbits10"],
            kernel_shape=[9, 1],
            pads=[4, 0, 4, 0],
        ),
        node("Equal", ["vbits10", "u8_three"], ["inside_bool10"]),
        node("Cast", ["inside_bool10"], ["inside10"], to=TensorProto.UINT8),
        node("Concat", ["inside10", "yellow10", "one10"], ["feat10"], axis=1),
        node(
            "QLinearConv",
            q_inputs_render,
            ["output"],
            kernel_shape=[1, 1],
            pads=[0, 0, 20, 20],
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "task273_separable_span_u8_terminal_qconv",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.UINT8, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, OUT)
    params = sum(int(np.prod(t.dims)) if t.dims else 1 for t in model.graph.initializer)
    print(f"saved {OUT} params={params}")


if __name__ == "__main__":
    main()
