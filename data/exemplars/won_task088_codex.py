#!/usr/bin/env python3
"""task088 attempt 010: scalar marker moments + thresholded QLinearConv.

Rule: crop the rectangle enclosed by the color appearing exactly four times,
then recolor every non-background interior cell to that marker color.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper, numpy_helper


OUT = Path(__file__).resolve().with_name("task088.onnx")
K = 10


def _c(name: str, arr: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr), name=name)


def build() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []

    # Encoded QLinearConv weights use w_zero_point=1, so values 0/1/2 mean
    # effective weights -1/0/+1. base_bg is [2, 1] for output channel 0 and
    # [1, 1] elsewhere before marker-specific +/- adjustment.
    base_bg_w = np.ones((10, 1, 1, 1), dtype=np.uint8)
    base_bg_w[0, 0, 0, 0] = 2
    bias = np.zeros((10,), dtype=np.int32)
    bias[0] = -1

    inits: list[onnx.TensorProto] = [
        _c("four_f", np.array(4.0, np.float32)),
        _c("one_f", np.array(1.0, np.float32)),
        _c("zero_u8", np.array(0, np.uint8)),
        _c("one_u8", np.array(1, np.uint8)),
        _c("channel0_i32", np.array([0], np.int32)),
        _c("channel1_i32", np.array([1], np.int32)),
        _c("ten_i32", np.array([10], np.int32)),
        _c("row_ids_i32", np.arange(10, dtype=np.int32).reshape(1, 1, 10, 1)),
        _c("col_ids_i32", np.arange(10, dtype=np.int32).reshape(1, 1, 1, 10)),
        _c("half_f", np.array(0.5, np.float32)),
        _c("coord30_f", np.arange(30, dtype=np.float32)),
        _c("coord30_sq_f", (np.arange(30, dtype=np.float32) ** 2).astype(np.float32)),
        _c("base_bg_w_u8", base_bg_w),
        _c("qconv_bias_i32", bias),
        _c("pad_axes_i64", np.array([2, 3], np.int64)),
        _c("axes_chw_i32", np.array([1, 2, 3], np.int32)),
    ]

    def add(op: str, ins: list[str], outs: list[str], **kw) -> None:
        nodes.append(helper.make_node(op, ins, outs, **kw))

    add("ReduceSum", ["input", "pad_axes_i64"], ["color_counts"], keepdims=1)
    add("Equal", ["color_counts", "four_f"], ["marker_is_four"])
    add("Cast", ["marker_is_four"], ["marker_mask_f"], to=TP.FLOAT)

    # The marker channel has exactly four pixels: two on the top marker row and
    # two on the bottom marker row, symmetrically for columns. First and second
    # coordinate moments recover the two extrema as scalar tensors.
    add("Einsum", ["input", "marker_mask_f", "coord30_f"], ["row_sum"], equation="nchw,ncij,h->n")
    add(
        "Einsum",
        ["input", "marker_mask_f", "coord30_sq_f"],
        ["row_sq_sum"],
        equation="nchw,ncij,h->n",
    )
    add("Einsum", ["input", "marker_mask_f", "coord30_f"], ["col_sum"], equation="nchw,ncij,w->n")
    add(
        "Einsum",
        ["input", "marker_mask_f", "coord30_sq_f"],
        ["col_sq_sum"],
        equation="nchw,ncij,w->n",
    )

    def extrema_from_moments(prefix: str, sum_name: str, sq_sum_name: str, lo: str, hi: str) -> None:
        pair_sum = f"{prefix}_pair_sum"
        pair_sum_sq = f"{prefix}_pair_sum_sq"
        diff_sq = f"{prefix}_diff_sq"
        diff = f"{prefix}_diff"
        lo2 = f"{prefix}_lo2"
        hi2 = f"{prefix}_hi2"
        lo_f = f"{prefix}_lo_f"
        hi_f = f"{prefix}_hi_f"
        add("Mul", [sum_name, "half_f"], [pair_sum])
        add("Mul", [pair_sum, pair_sum], [pair_sum_sq])
        add("Sub", [sq_sum_name, pair_sum_sq], [diff_sq])
        add("Sqrt", [diff_sq], [diff])
        add("Sub", [pair_sum, diff], [lo2])
        add("Add", [pair_sum, diff], [hi2])
        add("Mul", [lo2, "half_f"], [lo_f])
        add("Mul", [hi2, "half_f"], [hi_f])
        add("Cast", [lo_f], [lo], to=TP.INT32)
        add("Cast", [hi_f], [hi], to=TP.INT32)

    extrema_from_moments("row", "row_sum", "row_sq_sum", "top", "bottom")
    extrema_from_moments("col", "col_sum", "col_sq_sum", "left", "right")

    add("Add", ["top", "channel1_i32"], ["row_start"])
    add("Add", ["left", "channel1_i32"], ["col_start"])
    add("Sub", ["bottom", "row_start"], ["height"])
    add("Sub", ["right", "col_start"], ["width"])
    add("Add", ["row_start", "ten_i32"], ["row_stop10"])
    add("Add", ["col_start", "ten_i32"], ["col_stop10"])
    add("Concat", ["channel0_i32", "row_start", "col_start"], ["ch0_starts"], axis=0)
    add("Concat", ["channel1_i32", "row_stop10", "col_stop10"], ["ch0_ends"], axis=0)
    add("Slice", ["input", "ch0_starts", "ch0_ends", "axes_chw_i32"], ["ch0_crop"])
    add("Cast", ["ch0_crop"], ["bg_u8"], to=TP.UINT8)

    add("Less", ["row_ids_i32", "height"], ["row_valid"])
    add("Less", ["col_ids_i32", "width"], ["col_valid"])
    add("Cast", ["row_valid"], ["row_valid_u8"], to=TP.UINT8)
    add("Cast", ["col_valid"], ["col_valid_u8"], to=TP.UINT8)
    add("Mul", ["row_valid_u8", "col_valid_u8"], ["valid_u8"])
    add("Concat", ["valid_u8", "bg_u8"], ["features"], axis=1)

    add("Cast", ["marker_is_four"], ["marker_w_batched"], to=TP.UINT8)
    add("Transpose", ["marker_w_batched"], ["marker_w_u8"], perm=[1, 0, 2, 3])
    add("Add", ["base_bg_w_u8", "marker_w_u8"], ["w_valid"])
    add("Sub", ["base_bg_w_u8", "marker_w_u8"], ["w_bg"])
    add("Concat", ["w_valid", "w_bg"], ["dyn_w"], axis=1)
    add(
        "QLinearConv",
        [
            "features",
            "one_f",
            "zero_u8",
            "dyn_w",
            "one_f",
            "one_u8",
            "one_f",
            "zero_u8",
            "qconv_bias_i32",
        ],
        ["output"],
        pads=[0, 0, 20, 20],
    )

    vi = [
        helper.make_tensor_value_info("color_counts", TP.FLOAT, [1, 10, 1, 1]),
        helper.make_tensor_value_info("marker_is_four", TP.BOOL, [1, 10, 1, 1]),
        helper.make_tensor_value_info("marker_mask_f", TP.FLOAT, [1, 10, 1, 1]),
        helper.make_tensor_value_info("row_sum", TP.FLOAT, [1]),
        helper.make_tensor_value_info("row_sq_sum", TP.FLOAT, [1]),
        helper.make_tensor_value_info("col_sum", TP.FLOAT, [1]),
        helper.make_tensor_value_info("col_sq_sum", TP.FLOAT, [1]),
        helper.make_tensor_value_info("row_pair_sum", TP.FLOAT, [1]),
        helper.make_tensor_value_info("row_pair_sum_sq", TP.FLOAT, [1]),
        helper.make_tensor_value_info("row_diff_sq", TP.FLOAT, [1]),
        helper.make_tensor_value_info("row_diff", TP.FLOAT, [1]),
        helper.make_tensor_value_info("row_lo2", TP.FLOAT, [1]),
        helper.make_tensor_value_info("row_hi2", TP.FLOAT, [1]),
        helper.make_tensor_value_info("row_lo_f", TP.FLOAT, [1]),
        helper.make_tensor_value_info("row_hi_f", TP.FLOAT, [1]),
        helper.make_tensor_value_info("col_pair_sum", TP.FLOAT, [1]),
        helper.make_tensor_value_info("col_pair_sum_sq", TP.FLOAT, [1]),
        helper.make_tensor_value_info("col_diff_sq", TP.FLOAT, [1]),
        helper.make_tensor_value_info("col_diff", TP.FLOAT, [1]),
        helper.make_tensor_value_info("col_lo2", TP.FLOAT, [1]),
        helper.make_tensor_value_info("col_hi2", TP.FLOAT, [1]),
        helper.make_tensor_value_info("col_lo_f", TP.FLOAT, [1]),
        helper.make_tensor_value_info("col_hi_f", TP.FLOAT, [1]),
        helper.make_tensor_value_info("top", TP.INT32, [1]),
        helper.make_tensor_value_info("left", TP.INT32, [1]),
        helper.make_tensor_value_info("bottom", TP.INT32, [1]),
        helper.make_tensor_value_info("right", TP.INT32, [1]),
        helper.make_tensor_value_info("row_start", TP.INT32, [1]),
        helper.make_tensor_value_info("col_start", TP.INT32, [1]),
        helper.make_tensor_value_info("height", TP.INT32, [1]),
        helper.make_tensor_value_info("width", TP.INT32, [1]),
        helper.make_tensor_value_info("row_stop10", TP.INT32, [1]),
        helper.make_tensor_value_info("col_stop10", TP.INT32, [1]),
        helper.make_tensor_value_info("ch0_starts", TP.INT32, [3]),
        helper.make_tensor_value_info("ch0_ends", TP.INT32, [3]),
        helper.make_tensor_value_info("ch0_crop", TP.FLOAT, [1, 1, K, K]),
        helper.make_tensor_value_info("bg_u8", TP.UINT8, [1, 1, K, K]),
        helper.make_tensor_value_info("row_valid", TP.BOOL, [1, 1, K, 1]),
        helper.make_tensor_value_info("col_valid", TP.BOOL, [1, 1, 1, K]),
        helper.make_tensor_value_info("row_valid_u8", TP.UINT8, [1, 1, K, 1]),
        helper.make_tensor_value_info("col_valid_u8", TP.UINT8, [1, 1, 1, K]),
        helper.make_tensor_value_info("valid_u8", TP.UINT8, [1, 1, K, K]),
        helper.make_tensor_value_info("features", TP.UINT8, [1, 2, K, K]),
        helper.make_tensor_value_info("marker_w_batched", TP.UINT8, [1, 10, 1, 1]),
        helper.make_tensor_value_info("marker_w_u8", TP.UINT8, [10, 1, 1, 1]),
        helper.make_tensor_value_info("w_valid", TP.UINT8, [10, 1, 1, 1]),
        helper.make_tensor_value_info("w_bg", TP.UINT8, [10, 1, 1, 1]),
        helper.make_tensor_value_info("dyn_w", TP.UINT8, [10, 2, 1, 1]),
    ]
    graph = helper.make_graph(
        nodes,
        "task088_attempt010_scalar_moments_qlinearconv",
        [helper.make_tensor_value_info("input", TP.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TP.UINT8, [1, 10, 30, 30])],
        inits,
        value_info=vi,
    )
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    onnx.save(build(), OUT)
    print(OUT)
