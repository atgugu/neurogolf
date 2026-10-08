#!/usr/bin/env python3
from __future__ import annotations

import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


H = W = 13
REL = 25
CENTER = 12


def init(name: str, arr) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr), name=name)


def stair_offsets() -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    for dr, dc in [(-1, 1), (1, -1)]:
        v, h, r, c = 2, 0, 0, 0
        while True:
            if v:
                r += dr
                v -= 1
                if r < -CENTER or r > CENTER:
                    break
                offsets.append((r, c))
                if not v:
                    h = 2
            else:
                c += dc
                h -= 1
                if c < -CENTER or c > CENTER:
                    break
                offsets.append((r, c))
                if not h:
                    v = 2
    return offsets


def rel25(*, gray_code: int = 5, cyan_code: int = 8) -> np.ndarray:
    canvas = np.zeros((1, 1, REL, REL), dtype=np.uint8)
    canvas[0, 0, CENTER, CENTER] = cyan_code
    for dr, dc in stair_offsets():
        canvas[0, 0, CENTER + dr, CENTER + dc] = gray_code
    return canvas


def value_info(name: str, elem_type: int, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, elem_type, shape)


def add_scalar_seed_slice(nodes: list[onnx.NodeProto]) -> None:
    nodes.extend(
        [
            helper.make_node(
                "Einsum",
                ["input", "cyan_selector", "pos30"],
                ["seed_row_f"],
                equation="nchw,c,h->n",
            ),
            helper.make_node(
                "Einsum",
                ["input", "cyan_selector", "pos30"],
                ["seed_col_f"],
                equation="nchw,c,w->n",
            ),
            helper.make_node("Cast", ["seed_row_f"], ["seed_row_i64"], to=TensorProto.INT64),
            helper.make_node("Cast", ["seed_col_f"], ["seed_col_i64"], to=TensorProto.INT64),
            helper.make_node("Sub", ["twelve_i64", "seed_row_i64"], ["start_r"]),
            helper.make_node("Sub", ["twelve_i64", "seed_col_i64"], ["start_c"]),
            helper.make_node("Add", ["start_r", "thirteen_i64"], ["end_r"]),
            helper.make_node("Add", ["start_c", "thirteen_i64"], ["end_c"]),
            helper.make_node("Concat", ["start_r", "start_c"], ["starts"], axis=0),
            helper.make_node("Concat", ["end_r", "end_c"], ["ends"], axis=0),
            helper.make_node("Slice", ["rel25", "starts", "ends", "axes_hw"], ["color13"]),
        ]
    )


def scalar_value_infos(include_color13_output: bool) -> list[onnx.ValueInfoProto]:
    infos = [
        value_info("seed_row_f", TensorProto.FLOAT, [1]),
        value_info("seed_col_f", TensorProto.FLOAT, [1]),
        value_info("seed_row_i64", TensorProto.INT64, [1]),
        value_info("seed_col_i64", TensorProto.INT64, [1]),
        value_info("start_r", TensorProto.INT64, [1]),
        value_info("start_c", TensorProto.INT64, [1]),
        value_info("end_r", TensorProto.INT64, [1]),
        value_info("end_c", TensorProto.INT64, [1]),
        value_info("starts", TensorProto.INT64, [2]),
        value_info("ends", TensorProto.INT64, [2]),
    ]
    if not include_color13_output:
        infos.append(value_info("color13", TensorProto.UINT8, [1, 1, H, W]))
    return infos


def common_initializers(*, gray_code: int = 5, cyan_code: int = 8) -> list[onnx.TensorProto]:
    cyan_selector = np.zeros(10, dtype=np.float32)
    cyan_selector[8] = 1.0
    return [
        init("cyan_selector", cyan_selector),
        init("pos30", np.arange(30, dtype=np.float32)),
        init("rel25", rel25(gray_code=gray_code, cyan_code=cyan_code)),
        init("twelve_i64", np.array([12], dtype=np.int64)),
        init("thirteen_i64", np.array([13], dtype=np.int64)),
        init("axes_hw", np.array([2, 3], dtype=np.int64)),
    ]


def build_colorgrid() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    add_scalar_seed_slice(nodes)
    nodes[-1].output[0] = "output"
    graph = helper.make_graph(
        nodes,
        "task323_relative_template_colorgrid",
        [value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [value_info("output", TensorProto.UINT8, [1, 1, H, W])],
        common_initializers(),
        value_info=scalar_value_infos(include_color13_output=True),
    )
    return helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 14)])


def build_pad_equal() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits = common_initializers()
    inits.extend(
        [
            init("pad_13_to_30", np.array([0, 0, 0, 0, 0, 0, 17, 17], dtype=np.int64)),
            init("pad_sentinel_u8", np.array(255, dtype=np.uint8)),
            init("colors10", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1)),
        ]
    )
    add_scalar_seed_slice(nodes)
    nodes.extend(
        [
            helper.make_node(
                "Pad",
                ["color13", "pad_13_to_30", "pad_sentinel_u8"],
                ["color30"],
                mode="constant",
            ),
            helper.make_node("Equal", ["color30", "colors10"], ["output"]),
        ]
    )
    graph = helper.make_graph(
        nodes,
        "task323_relative_template_pad_equal",
        [value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
        inits,
        value_info=scalar_value_infos(include_color13_output=False)
        + [value_info("color30", TensorProto.UINT8, [1, 1, 30, 30])],
    )
    return helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 14)])


def decoder_kernel5() -> np.ndarray:
    w = np.zeros((10, 1, 5, 5), dtype=np.int8)
    w[0, 0] = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, -127, 125, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.int8,
    )
    w[5, 0] = np.array(
        [
            [1, 0, -2, 0, 1],
            [0, 1, 0, 1, -1],
            [1, 0, 7, 0, 1],
            [0, 1, 0, 1, 1],
            [1, -1, -2, 1, 2],
        ],
        dtype=np.int8,
    )
    w[8, 0] = np.array(
        [
            [-54, 8, 124, 17, -15],
            [48, -46, 24, -4, -19],
            [-36, -7, -1, -18, -32],
            [-6, -4, 14, 11, 15],
            [-13, 29, 123, -25, -115],
        ],
        dtype=np.int8,
    )
    return w


def decoder_kernel3() -> np.ndarray:
    # The center 3x3 neighborhood is the smallest tested candidate that can
    # retain context for separating the middle color from the two extremes.
    return decoder_kernel5()[:, :, 1:4, 1:4]


def decoder_bias5() -> np.ndarray:
    b = np.zeros(10, dtype=np.int32)
    b[0] = -1
    b[5] = -23
    b[8] = -1118
    return b


def build_qconv_decoder() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits = common_initializers(gray_code=8, cyan_code=5)
    inits.extend(
        [
            init("one_f", np.array(1.0, dtype=np.float32)),
            init("x_zp", np.array(1, dtype=np.uint8)),
            init("w_zp", np.array(0, dtype=np.int8)),
            init("decoder_w", decoder_kernel3()),
            init("decoder_b", decoder_bias5()),
        ]
    )
    add_scalar_seed_slice(nodes)
    nodes.append(
        helper.make_node(
            "QLinearConv",
            ["color13", "one_f", "x_zp", "decoder_w", "one_f", "w_zp", "one_f", "x_zp", "decoder_b"],
            ["output"],
            pads=[1, 1, 18, 18],
        )
    )
    graph = helper.make_graph(
        nodes,
        "task323_relative_template_qconv_decoder",
        [value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [value_info("output", TensorProto.UINT8, [1, 10, 30, 30])],
        inits,
        value_info=scalar_value_infos(include_color13_output=False),
    )
    return helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 14)])


def build() -> onnx.ModelProto:
    mode = os.environ.get("TASK323_MODE", "qconv")
    if mode == "colorgrid":
        model = build_colorgrid()
    elif mode == "pad_equal":
        model = build_pad_equal()
    elif mode == "qconv":
        model = build_qconv_decoder()
    else:
        raise ValueError(f"unknown TASK323_MODE={mode!r}")
    model.ir_version = 10
    return model


if __name__ == "__main__":
    model = build()
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, "task323.onnx")
    print("wrote task323.onnx")
