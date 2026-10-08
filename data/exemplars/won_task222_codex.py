#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = Path(__file__).with_name("task222.onnx")


def init(name: str, value) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value), name=name)


def build() -> onnx.ModelProto:
    # The rectangle is strictly inside the 16x16 active grid. Cropping rows/cols
    # 1..14 keeps every possible 2x2 seed while deleting the guaranteed border.
    crop_code = np.zeros((1, 10, 1, 1), dtype=np.float32)
    crop_code[0, :, 0, 0] = np.arange(10, dtype=np.float32) * 3

    # Select solid 2x2 seeds that have a 4-neighbor seed. This rejects isolated
    # 2x2 noise while preserving all legal rectangle seed components.
    select_kernel = np.array([[[[0, 1, 0], [1, 20, 1], [0, 1, 0]]]], dtype=np.uint8)

    row_canvas = np.zeros((1, 1, 30, 1), dtype=np.float16)
    row_canvas[:, :, :16, :] = 1
    col_canvas = np.zeros((1, 1, 1, 30), dtype=np.float16)
    col_canvas[:, :, :, :16] = 1
    bg_chan = np.zeros((1, 1, 10, 1), dtype=np.float16)
    bg_chan[:, :, 0, :] = 1
    scaled_lut = (np.arange(10, dtype=np.uint8) * 3).reshape(1, 1, 10, 1)

    inits = [
        init("crop_code", crop_code),
        init("q_scale", np.float32(1.0)),
        init("avg_y_scale", np.float32(4.0)),
        init("select_y_scale", np.float32(41.0)),
        init("q_zp", np.uint8(0)),
        init("sum_kernel_q", np.ones((1, 1, 2, 2), dtype=np.uint8)),
        init("select_kernel_q", select_kernel),
        init("pad_row_seed16", np.array([0, 0, 1, 0, 0, 0, 2, 0], dtype=np.int64)),
        init("pad_col_seed16", np.array([0, 0, 0, 1, 0, 0, 0, 2], dtype=np.int64)),
        init("pad_row30", np.array([0, 0, 0, 0, 0, 0, 14, 0], dtype=np.int64)),
        init("pad_col30", np.array([0, 0, 0, 0, 0, 0, 0, 14], dtype=np.int64)),
        init("row_canvas", row_canvas),
        init("col_canvas", col_canvas),
        init("bg_chan", bg_chan),
        init("scaled_lut", scaled_lut),
    ]

    nodes = [
        helper.make_node(
            "Conv",
            ["input", "crop_code"],
            ["grid14f"],
            kernel_shape=[1, 1],
            pads=[-1, -1, -15, -15],
        ),
        helper.make_node("Cast", ["grid14f"], ["code14"], to=TensorProto.UINT8),
        helper.make_node("MaxPool", ["code14"], ["mx2"], kernel_shape=[2, 2], strides=[1, 1]),
        helper.make_node(
            "QLinearConv",
            [
                "code14",
                "q_scale",
                "q_zp",
                "sum_kernel_q",
                "q_scale",
                "q_zp",
                "avg_y_scale",
                "q_zp",
            ],
            ["avg2"],
            kernel_shape=[2, 2],
        ),
        helper.make_node("Equal", ["avg2", "mx2"], ["solid"]),
        helper.make_node("Cast", ["solid"], ["solid_u8"], to=TensorProto.UINT8),
        helper.make_node(
            "QLinearConv",
            [
                "solid_u8",
                "q_scale",
                "q_zp",
                "select_kernel_q",
                "q_scale",
                "q_zp",
                "select_y_scale",
                "q_zp",
            ],
            ["selected_u8"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        ),
        helper.make_node("Mul", ["selected_u8", "mx2"], ["selected_code_map"]),
        helper.make_node(
            "MaxPool",
            ["selected_code_map"],
            ["row_seed"],
            kernel_shape=[1, 13],
            strides=[1, 1],
        ),
        helper.make_node(
            "MaxPool",
            ["selected_code_map"],
            ["col_seed"],
            kernel_shape=[13, 1],
            strides=[1, 1],
        ),
        helper.make_node("Pad", ["row_seed", "pad_row_seed16", "q_zp"], ["row_seed16"]),
        helper.make_node("Pad", ["col_seed", "pad_col_seed16", "q_zp"], ["col_seed16"]),
        helper.make_node(
            "MaxPool",
            ["row_seed16"],
            ["row_rect_u8"],
            kernel_shape=[2, 1],
            pads=[1, 0, 0, 0],
            strides=[1, 1],
        ),
        helper.make_node(
            "MaxPool",
            ["col_seed16"],
            ["col_rect_u8"],
            kernel_shape=[1, 2],
            pads=[0, 1, 0, 0],
            strides=[1, 1],
        ),
        helper.make_node("Pad", ["row_rect_u8", "pad_row30", "q_zp"], ["row_rect30"]),
        helper.make_node("Pad", ["col_rect_u8", "pad_col30", "q_zp"], ["col_rect30"]),
        helper.make_node("Cast", ["row_rect30"], ["row_rect_f16"], to=TensorProto.FLOAT16),
        helper.make_node("Cast", ["col_rect30"], ["col_rect_f16"], to=TensorProto.FLOAT16),
        helper.make_node("Concat", ["row_canvas", "row_rect_f16"], ["row_comp"], axis=1),
        helper.make_node("Concat", ["col_canvas", "col_rect_f16"], ["col_comp"], axis=1),
        helper.make_node(
            "MaxPool",
            ["selected_code_map"],
            ["selected_code"],
            kernel_shape=[13, 13],
            strides=[1, 1],
        ),
        helper.make_node("Equal", ["selected_code", "scaled_lut"], ["selected_chan_b"]),
        helper.make_node("Cast", ["selected_chan_b"], ["selected_chan"], to=TensorProto.FLOAT16),
        helper.make_node("Sub", ["selected_chan", "bg_chan"], ["delta_chan"]),
        helper.make_node("Concat", ["bg_chan", "delta_chan"], ["chan_comp"], axis=1),
        helper.make_node(
            "Einsum",
            ["row_comp", "col_comp", "chan_comp"],
            ["output"],
            equation="ntrq,ntqc,ntkq->nkrc",
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "task222_att12_selected_code_factor",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    del model.graph.value_info[:]
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    onnx.save(build(), OUT)
    print(OUT)
