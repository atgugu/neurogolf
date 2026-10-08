#!/usr/bin/env python3
"""Build task192 packed-row terminal renderer.

True rule: keep exactly the cells that participate in at least one nonzero
2x2 block, repaint those cells with the dominant nonzero box color, emit
channel-0 background inside the true HxW canvas, and leave padding outside
HxW all-false.

This replaces the failed packed-bitset attempt's expensive u32 20x20 unpack
and scalar Pad/Equal tail with a packed terminal renderer:
  final output = BitwiseAnd(row_channel_bitsets, column_bit_masks)
The gate thresholds raw > 0, so uint32 bit values are a valid one-hot output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


OUT = Path(__file__).resolve().parent / "task192.onnx"


def arr(name: str, values, dtype) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values, dtype=dtype), name)


def build() -> onnx.ModelProto:
    f32 = np.float32
    u32 = np.uint32
    i64 = np.int64
    b = np.bool_

    bit_h30 = np.zeros((30,), dtype=f32)
    bit_h30[:20] = [float(1 << i) for i in range(20)]
    row_masks30 = np.asarray([1 << i for i in range(30)], dtype=np.uint32).reshape(1, 1, 30, 1)
    bg_idx = np.zeros((10, 1), dtype=np.int32)
    bg_idx[0, 0] = 1

    inits = [
        arr("fg_ch", [0.0] + [1.0] * 9, f32),
        arr("bit_h30", bit_h30, f32),
        arr("starts0", [0], i64),
        arr("starts1", [1], i64),
        arr("ends19", [19], i64),
        arr("ends20", [20], i64),
        arr("axis1", [1], i64),
        arr("shift1", [1], u32),
        arr("zero_col", np.zeros((1, 1), dtype=u32), u32),
        arr("zero_tail", np.zeros((1, 10), dtype=u32), u32),
        arr("zero_bits", np.zeros((1, 30), dtype=u32), u32),
        arr("idx10", np.arange(10, dtype=i64).reshape(10, 1), i64),
        arr("bg_idx", bg_idx, np.int32),
        arr("sel_keep", [2], np.int32),
        arr("row_masks30", row_masks30, u32),
    ]

    nodes = [
        # Packed in-grid mask by column: every true-canvas cell has exactly
        # one hot input channel, while padded cells have none.
        helper.make_node("Einsum", ["input", "bit_h30"], ["grid_bits_f"], equation="nchw,h->nw"),
        helper.make_node("Cast", ["grid_bits_f"], ["grid_bits"], to=TensorProto.UINT32),
        # Packed nonzero columns, used to detect 2x2 all-nonzero anchors.
        helper.make_node("Einsum", ["input", "fg_ch", "bit_h30"], ["col_bits_f"], equation="nchw,c,h->nw"),
        helper.make_node("Cast", ["col_bits_f"], ["col_bits"], to=TensorProto.UINT32),
        helper.make_node("Slice", ["col_bits", "starts0", "ends19", "axis1"], ["bits_a"]),
        helper.make_node("Slice", ["col_bits", "starts1", "ends20", "axis1"], ["bits_b"]),
        helper.make_node("BitwiseAnd", ["bits_a", "bits_b"], ["vpair"]),
        helper.make_node("BitShift", ["vpair", "shift1"], ["vpair_r"], direction="RIGHT"),
        helper.make_node("BitwiseAnd", ["vpair", "vpair_r"], ["anchor_bits"]),
        # Equivalent to dilating left/right first, but it keeps the vertical
        # dilation on the 19-column anchor lane and removes four 20-wide lanes.
        helper.make_node("BitShift", ["anchor_bits", "shift1"], ["anchor_down"], direction="LEFT"),
        helper.make_node("BitwiseOr", ["anchor_bits", "anchor_down"], ["anchor_v"]),
        helper.make_node("Concat", ["anchor_v", "zero_col"], ["anch_left"], axis=1),
        helper.make_node("Concat", ["zero_col", "anchor_v"], ["anch_right"], axis=1),
        helper.make_node("BitwiseOr", ["anch_left", "anch_right"], ["keep_bits"]),
        helper.make_node("Concat", ["keep_bits", "zero_tail"], ["keep_bits30"], axis=1),
        helper.make_node("BitwiseXor", ["grid_bits", "keep_bits30"], ["bg_bits"]),
        # Dominant nonzero color. fg_ch zeroes channel 0, so ArgMax returns the
        # actual color id among channels 1..9.
        helper.make_node("Einsum", ["input", "fg_ch"], ["counts10"], equation="nchw,c->nc"),
        helper.make_node("ArgMax", ["counts10"], ["arg10"], axis=1, keepdims=1),
        helper.make_node("Equal", ["idx10", "arg10"], ["dom_chan"]),
        # Build packed row bitsets per output channel.
        helper.make_node("Concat", ["zero_bits", "bg_bits", "keep_bits30"], ["choices"], axis=0),
        helper.make_node("Where", ["dom_chan", "sel_keep", "bg_idx"], ["sel"]),
        helper.make_node("Gather", ["choices", "sel"], ["state30"], axis=0),
        # Terminal renderer. The uint32 bit value at each column is positive
        # iff that output channel owns the cell; the graph output is free.
        helper.make_node("BitwiseAnd", ["state30", "row_masks30"], ["output"]),
    ]

    graph = helper.make_graph(
        nodes,
        "task192_lean_packed_terminal",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.UINT32, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    return onnx.shape_inference.infer_shapes(model, strict_mode=True)


def main() -> None:
    model = build()
    onnx.save(model, OUT)
    print(OUT)
    mem = 0
    for vi in model.graph.value_info:
        tt = vi.type.tensor_type
        dims = [d.dim_value for d in tt.shape.dim]
        nbytes = int(np.prod(dims)) * np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tt.elem_type)).itemsize
        mem += nbytes
        print(f"{nbytes:4} {vi.name:14} {TensorProto.DataType.Name(tt.elem_type):8} {dims}")
    params = sum(int(np.prod(t.dims)) for t in model.graph.initializer)
    print("params", params, "mem", mem, "cost", params + mem)


if __name__ == "__main__":
    main()
