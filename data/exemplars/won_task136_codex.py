#!/usr/bin/env python3
"""Build task136 ONNX.

Rule: preserve both 2x2 blocks, extend color 1 northwest from its top-left,
and extend color 2 southeast from its top-left.

Attempt 5 family: FINAL-CONVINTEGER-FUSE.  Keep the packed no-Div anchor
extractor from attempt 4, but do not materialize bool color masks, a
background mask, or a 3-channel bool scaffold.  Instead convert each dynamic
horizontal interval to a uint8 false-flag plane:

    flag = 0 where the interval is active, 2 otherwise.

The final ConvInteger is free as the graph output.  With x_zero_point=1 it
computes:
    ch0 = (f1 - 1) + (f2 - 1)       >0 only when both flags are false
    ch1 = -(f1 - 1)                 >0 only on color 1
    ch2 = -(f2 - 1)                 >0 only on color 2
and uses Conv padding to expand native 10x10 to 30x30.
"""
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper, numpy_helper


OUT = Path("task136.onnx")


def init(name, array, dtype=None):
    arr = np.asarray(array)
    if dtype is not None:
        arr = arr.astype(dtype)
    return numpy_helper.from_array(arr, name)


def vi(name, elem_type, shape):
    return helper.make_tensor_value_info(name, elem_type, shape)


def node(op, inputs, outputs, **attrs):
    return helper.make_node(op, inputs, outputs, **attrs)


def build():
    # q[i] makes a 2x2 block at rows r,r+1 sum to r:
    # 2*q[r] + 2*q[r+1] == r.
    coord = (np.floor(np.arange(30) / 2) / 2).astype(np.float32)
    code_sel = np.zeros((10,), np.float32)
    code_sel[1] = 1.0
    code_sel[2] = 16.0

    w = np.zeros((10, 2, 1, 1), np.int8)
    w[0, :, 0, 0] = [1, 1]
    w[1, :, 0, 0] = [-1, 0]
    w[2, :, 0, 0] = [0, -1]

    inits = [
        init("code_sel", code_sel),
        init("coord", coord),
        init("axes_12", np.array([1, 2]), np.int64),
        init("zero_u8", np.array([0]), np.uint8),
        init("one_u8", np.array([1]), np.uint8),
        init("two_u8", np.array([2]), np.uint8),
        init("sixteen_u8", np.array([16]), np.uint8),
        init("rows10", np.arange(10).reshape(1, 1, 10, 1), np.uint8),
        init("cols10", np.arange(10).reshape(1, 1, 1, 10), np.uint8),
        init("lane0_is_nw", np.array([True, False]).reshape(1, 2, 1, 1), np.bool_),
        init("conv_w", w),
        init("x_zp", np.array([1], dtype=np.uint8)),
    ]

    nodes = [
        node("Einsum", ["input", "code_sel", "coord"], ["row_code_f"], equation="nchw,c,h->n"),
        node("Einsum", ["input", "code_sel", "coord"], ["col_code_f"], equation="nchw,c,w->n"),
        node("Cast", ["row_code_f"], ["row_code"], to=TP.UINT8),
        node("Cast", ["col_code_f"], ["col_code"], to=TP.UINT8),
        node("Mod", ["row_code", "sixteen_u8"], ["row_lo"], fmod=0),
        node("Div", ["row_code", "sixteen_u8"], ["row_hi"]),
        node("Mod", ["col_code", "sixteen_u8"], ["col_lo"], fmod=0),
        node("Div", ["col_code", "sixteen_u8"], ["col_hi"]),
        node("Concat", ["row_lo", "row_hi"], ["row_u"], axis=0),
        node("Concat", ["col_lo", "col_hi"], ["col_u"], axis=0),
        node("Unsqueeze", ["row_u", "axes_12"], ["row"]),
        node("Unsqueeze", ["col_u", "axes_12"], ["col"]),
        node("Greater", ["rows10", "row"], ["row_gt"]),
        node("Sub", ["rows10", "row"], ["row_diff"]),
        node("LessOrEqual", ["row_diff", "one_u8"], ["row_block"]),
        node("Equal", ["row_gt", "lane0_is_nw"], ["mask_cond"]),
        node("Add", ["row_diff", "col"], ["diag_cols"]),
        node("Where", ["mask_cond", "sixteen_u8", "diag_cols"], ["diag_masked"]),
        node("Where", ["row_block", "col", "diag_masked"], ["left2"]),
        node("Where", ["row_block", "one_u8", "zero_u8"], ["width2"]),
        # uint8 underflow is useful here: cols < left wraps above width, so
        # diff > width is exactly "outside this row's active interval".
        node("Sub", ["cols10", "left2"], ["diff2"]),
        node("Greater", ["diff2", "width2"], ["outside2"]),
        node("Where", ["outside2", "two_u8", "zero_u8"], ["flag2"]),
        node("ConvInteger", ["flag2", "conv_w", "x_zp"], ["output"], pads=[0, 0, 20, 20]),
    ]

    graph = helper.make_graph(
        nodes,
        "task136_final_convinteger_fuse",
        [vi("input", TP.FLOAT, [1, 10, 30, 30])],
        [vi("output", TP.INT32, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, OUT)
    print(f"saved {OUT}")


if __name__ == "__main__":
    build()
