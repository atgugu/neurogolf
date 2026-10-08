#!/usr/bin/env python3
"""Task333 att12: shifted/max-green native fill + exact QLinearConv renderer.

The trusted strip/scatter computation is preserved exactly.  Colors use shifted
codes z=color+1 (with green promoted to sentinel 11), so zero padding is distinct
from every in-grid cell and green dominates the directional maxima.  The final
QLinearConv uses [z,z^2] and folds the constant feature into an exact one-hot bias.

STEP ZERO: generator read; true rule = locate 2x2 green box, paint green, place seeds, ray-fill colors on box rows (horiz to box) and box cols (vert to box).
FALLBACK: keep pin algo, crop to cert box (10x10), relower dtypes exact, suppress optionals.
"""
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = Path(__file__).with_name("task333.onnx")


def init(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name)


def node(op, inputs, outputs, name=None, **attrs):
    return helper.make_node(op, inputs, outputs, name=name or outputs[0], **attrs)


def cast(src, dst, to):
    return node("Cast", [src], [dst], to=to)


def pool10(name, x, direction):
    if direction == "right":
        kernel, pads = [1, 10], [0, 9, 0, 0]
    elif direction == "left":
        kernel, pads = [1, 10], [0, 0, 0, 9]
    elif direction == "down":
        kernel, pads = [10, 1], [9, 0, 0, 0]
    elif direction == "up":
        kernel, pads = [10, 1], [0, 0, 9, 0]
    else:
        raise ValueError(direction)
    return node(
        "MaxPool",
        [x],
        [name],
        kernel_shape=kernel,
        pads=pads,
        strides=[1, 1],
    )


def main():
    spatial_select = np.zeros((10, 30), dtype=np.float32)
    for i in range(10):
        spatial_select[i, i] = 1.0

    # Green is a maximal sentinel, allowing each pair of directional results
    # to be selected with one Where while preserving the two green box cells.
    codes = np.arange(1, 11, dtype=np.int16)
    codes[3] = 11
    tail_w = np.stack([2 * codes, -np.ones(10, dtype=np.int16)], axis=1)
    tail_w = tail_w.astype(np.int8).reshape(10, 2, 1, 1)
    tail_bias = (1 - codes.astype(np.int32) ** 2)

    inits = [
        init("color_weights", codes.astype(np.float32).reshape(1, 10)),
        init("spatial_select", spatial_select),
        init("green_chan", (np.eye(1, 10, 3, dtype=np.float32) * 0.25).reshape(10)),
        init("row_iota", np.arange(10, dtype=np.float32).reshape(1, 1, 10, 1)),
        init("col_iota", np.arange(10, dtype=np.float32).reshape(1, 1, 1, 10)),
        init("two_i32", np.array([2], dtype=np.int32)),
        init("axes_h_i32", np.array([2], dtype=np.int32)),
        init("axes_w_i32", np.array([3], dtype=np.int32)),
        init("row_nd_offs", np.array([0, 1], dtype=np.int32).reshape(1, 1, 2, 1)),
        init("row_nd_zeros", np.zeros((1, 1, 2, 2), dtype=np.int64)),
        init("offs2_w_i32", np.array([0, 1], dtype=np.int32).reshape(1, 1, 1, 2)),
        init("col_idx_shape", np.array([1, 1, 10, 2], dtype=np.int64)),
        init("tail_w", tail_w),
        init("tail_bias", tail_bias),
        init("unit_scale", np.array([1], dtype=np.float32)),
        init("zero_u8", np.array([0], dtype=np.uint8)),
        init("zero_i8", np.array([0], dtype=np.int8)),
    ]

    nodes = [
        node(
            "Einsum",
            ["input", "color_weights", "spatial_select", "spatial_select"],
            ["idx_f32"],
            equation="bchw,oc,rh,sw->bors",
        ),
        cast("idx_f32", "idx_z", TensorProto.UINT8),
        node(
            "Einsum",
            ["input", "green_chan", "spatial_select", "row_iota"],
            ["boxR_sum"],
            equation="bchw,c,rh,pqrl->b",
        ),
        node(
            "Einsum",
            ["input", "green_chan", "spatial_select", "col_iota"],
            ["boxC_sum"],
            equation="bchw,c,sw,pqks->b",
        ),
        cast("boxR_sum", "boxR_i", TensorProto.INT32),
        cast("boxC_sum", "boxC_i", TensorProto.INT32),
        node("Less", ["col_iota", "boxC_sum"], ["leftMask"]),
        node("Less", ["row_iota", "boxR_sum"], ["upMask"]),
        node("Add", ["boxR_i", "two_i32"], ["rowEnd_i"]),
        node("Add", ["boxC_i", "two_i32"], ["colEnd_i"]),
        node("Slice", ["idx_z", "boxR_i", "rowEnd_i", "axes_h_i32"], ["rowStrip"]),
        node("Slice", ["idx_z", "boxC_i", "colEnd_i", "axes_w_i32"], ["colStrip"]),
        pool10("PLs", "rowStrip", "right"),
        pool10("PRs", "rowStrip", "left"),
        pool10("PUs", "colStrip", "down"),
        pool10("PDs", "colStrip", "up"),
        node("Where", ["leftMask", "PLs", "PRs"], ["rowFill"]),
        node("Where", ["upMask", "PUs", "PDs"], ["colFill"]),
        node("Add", ["boxR_i", "row_nd_offs"], ["row_positions_i32"]),
        cast("row_positions_i32", "row_positions_i64", TensorProto.INT64),
        node("Concat", ["row_nd_zeros", "row_positions_i64"], ["row_nd_idx"], axis=3),
        node("Add", ["boxC_i", "offs2_w_i32"], ["col_idx_base"]),
        node("Expand", ["col_idx_base", "col_idx_shape"], ["col_scatter_idx"]),
        node(
            "ScatterND",
            ["idx_z", "row_nd_idx", "rowFill"],
            ["row_scattered"],
        ),
        node(
            "ScatterElements",
            ["row_scattered", "col_scatter_idx", "colFill"],
            ["filled_z"],
            axis=3,
        ),
        node("Mul", ["filled_z", "filled_z"], ["filled_sq"]),
        node("Concat", ["filled_z", "filled_sq"], ["tail_feat"], axis=1),
        node(
            "QLinearConv",
            [
                "tail_feat", "unit_scale", "zero_u8",
                "tail_w", "unit_scale", "zero_i8",
                "unit_scale", "zero_u8", "tail_bias",
            ],
            ["output"],
            pads=[0, 0, 20, 20],
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "task333_native_scatter",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.UINT8, [1, 10, 30, 30])],
        inits,
    )
    graph.value_info.extend(
        [
            helper.make_tensor_value_info("rowStrip", TensorProto.UINT8, [1, 1, 2, 10]),
            helper.make_tensor_value_info("colStrip", TensorProto.UINT8, [1, 1, 10, 2]),
        ]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, OUT)
    pcount = sum(np.asarray(numpy_helper.to_array(i)).size for i in inits)
    print(f"saved {OUT} nodes={len(nodes)} params={pcount}")


if __name__ == "__main__":
    main()
