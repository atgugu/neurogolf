#!/usr/bin/env python3
"""Task 178: exact 13-cell axis samplers plus the proven u8 run renderer."""
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "runner"))
from ngolf import G, T  # noqa: E402


def build() -> None:
    g = G(task=178)
    g.opset = 18

    # A zero second tap at dilation 17 makes each Conv's native output length
    # exactly 13.  This is value-identical to the reference's 30-cell Conv
    # followed by Slice(0:13), without materialising either full-width plane.
    row_w = np.zeros((1, 10, 1, 2), np.float32)
    row_w[0, :, 0, 0] = np.arange(10, dtype=np.float32)
    col_w = np.zeros((1, 10, 2, 1), np.float32)
    col_w[0, :, 0, 0] = np.arange(10, dtype=np.float32)
    row4 = g.n(
        "Conv", [g.input, g.init(row_w)], [1, 1, 1, 13], "f32",
        kernel_shape=[1, 2], dilations=[1, 17], strides=[30, 1],
    )
    col4 = g.n(
        "Conv", [g.input, g.init(col_w)], [1, 1, 13, 1], "f32",
        kernel_shape=[2, 1], dilations=[17, 1], strides=[1, 30],
    )
    rows = g.reshape(g.cast(row4, "u8"), [1, 13])
    cols = g.reshape(g.cast(col4, "u8"), [1, 13])

    start0 = g.init(np.array([0], np.int64))
    start1 = g.init(np.array([1], np.int64))
    end4 = g.init(np.array([4], np.int64))
    end12 = g.init(np.array([12], np.int64))
    end13 = g.init(np.array([13], np.int64))
    axis1 = g.init(np.array([1], np.int64))
    reduce_axis = g.init(np.array([1], np.int64))

    # Exact reference orientation test: among row positions 1..3, a valid
    # color differing from position zero exists iff the input was transposed.
    row_curr3 = g.n("Slice", [rows, start1, end4, axis1], [1, 3], "u8")
    row_first = g.n("Slice", [rows, start0, start1, axis1], [1, 1], "u8")
    row_changed = g.and_(g.cast(row_curr3, "b"), g.not_(g.eq(row_curr3, row_first)))
    row_changed_u8 = g.cast(row_changed, "u8")
    orient_u8 = g.n(
        "ReduceMax", [row_changed_u8, reduce_axis], [1], "u8", keepdims=0
    )
    is_row = g.cast(orient_u8, "b")
    line = g.n("Where", [is_row, rows, cols], [1, 13], "u8")

    curr = g.n("Slice", [line, start1, end13, axis1], [1, 12], "u8")
    prev = g.n("Slice", [line, start0, end12, axis1], [1, 12], "u8")
    starts_tail = g.and_(g.cast(curr, "b"), g.not_(g.eq(curr, prev)))
    starts = g.concat([g.init(np.array([[True]], np.bool_)), starts_tail], axis=1)

    scores = g.n(
        "Where",
        [starts,
         g.init(np.arange(13, 0, -1, dtype=np.float16).reshape(1, 13)),
         g.init(np.array(0, np.float16))],
        [1, 13], "f16",
    )
    k5 = g.init(np.array([5], np.int64))
    val_name, idx_name = g._nm(), g._nm()
    g.nodes.append(helper.make_node(
        "TopK", [scores.name, k5.name], [val_name, idx_name],
        axis=1, largest=1, sorted=1,
    ))
    top_values = T(val_name, [1, 5], "f16")
    top_idx = T(idx_name, [1, 5], "i64")
    g.charged.extend([top_values, top_idx])
    labels = g.n("GatherElements", [line, top_idx], [1, 5], "u8", axis=1)
    valid = g.n("GatherElements", [starts, top_idx], [1, 5], "b", axis=1)
    labels_masked = g.n(
        "Where", [valid, labels, g.init(np.array(10, np.uint8))], [1, 5], "u8"
    )

    row_shape = g.init(np.array([1, 1, 1, 5], np.int64))
    col_shape = g.init(np.array([1, 1, 5, 1], np.int64))
    label_shape = g.n("Where", [is_row, row_shape, col_shape], [4], "i64")
    labels4 = g.n("Reshape", [labels_masked, label_shape], [1, 1, 5, 1], "u8")
    channels = g.init(np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1))
    answer_line = g.eq(channels, labels4)
    output_axes = g.n(
        "Where",
        [is_row,
         g.init(np.array([3, 2], np.int32)),
         g.init(np.array([2, 3], np.int32))],
        [2], "i32",
    )
    g.n(
        "Pad",
        [answer_line,
         g.init(np.array([0, 0, 25, 29], np.int64)),
         g.init(np.array(False, np.bool_)), output_axes],
        [1, 10, 30, 30], "b", is_output=True, mode="constant",
    )

    print(g.budget())
    out = HERE / "task178.onnx"
    g.save(out)

    # Shape selection is data-dependent but has exactly these two layouts.
    m = onnx.load(out)
    m.graph.value_info.extend([
        helper.make_tensor_value_info(labels4.name, TensorProto.UINT8, [1, 1, 5, 1]),
        helper.make_tensor_value_info(answer_line.name, TensorProto.BOOL, [1, 10, 5, 1]),
    ])
    m = onnx.shape_inference.infer_shapes(m, strict_mode=True)
    onnx.checker.check_model(m, full_check=True)
    onnx.save(m, out)


if __name__ == "__main__":
    build()
