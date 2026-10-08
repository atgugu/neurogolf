#!/usr/bin/env python3
"""Packed-bitset rebuild attempt for task165.

True rule: find the 10-cell kite and, for each relative kite column that has a
scatter pixel strictly below the kite in that column, paint that column from
the bottom up to just below the kite using the scatter color.

The audited packed plan names PackBitsU32/UnpackBits helpers, but this repo has
no custom ONNX ops or ngolf helpers with those names.  This file lowers the
same representation to standard ONNX: row bits are packed by an Einsum+Cast,
and all kite detection is done with BitShift/BitwiseAnd over uint32 row words.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


OUT = Path("task165.onnx")
OPSET = 18


def init(name: str, values, dtype=None) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values, dtype=dtype), name)


def node(op: str, ins: list[str], out: str, **attrs) -> onnx.NodeProto:
    return helper.make_node(op, ins, [out], name=out, **attrs)


def bit_and(a: str, b: str, out: str) -> onnx.NodeProto:
    return node("BitwiseAnd", [a, b], out)


def bit_shift(x: str, amount: str, direction: str, out: str) -> onnx.NodeProto:
    return node("BitShift", [x, amount], out, direction=direction)


def add_budget(model: onnx.ModelProto) -> None:
    g = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    mem = 0
    rows = []
    for vi in g.graph.value_info:
        tt = vi.type.tensor_type
        if not tt.HasField("shape"):
            continue
        dims = [d.dim_value for d in tt.shape.dim]
        if not dims:
            dims = []
        nbytes = int(np.prod(dims or [1])) * np.dtype(
            onnx.helper.tensor_dtype_to_np_dtype(tt.elem_type)
        ).itemsize
        mem += nbytes
        rows.append((nbytes, vi.name, TensorProto.DataType.Name(tt.elem_type), dims))
    params = sum(int(np.prod(t.dims or [1])) for t in g.graph.initializer)
    cost = params + mem
    print(f"BUDGET: params={params} memory={mem} cost={cost} -> {25 - math.log(cost):.4f} pts")
    for nbytes, name, dt, dims in sorted(rows, reverse=True)[:16]:
        print(f"  {nbytes:5d} B  {dt:<8} {dims}  {name}")


def build() -> onnx.ModelProto:
    f32 = np.float32
    u8 = np.uint8
    u32 = np.uint32
    i32 = np.int32
    i64 = np.int64

    col_bits = (1 << np.arange(30, dtype=np.uint32)).astype(np.float32)
    valid_centers = np.uint32(sum(1 << c for c in range(5, 15)))

    inits = [
        init("fg_ch", [0.0] + [1.0] * 9, f32),
        init("fg4", np.array([0.0] + [1.0] * 9, dtype=f32).reshape(1, 10, 1, 1), None),
        init("col_bits_f", col_bits, f32),
        init("axis_row", [1], i64),
        init("axes_reduce_row", [2], i64),
        init("axes_reduce_rel", [3], i64),
        init("axes_spatial", [2, 3], i64),
        init("axes4", [0, 1, 2, 3], i64),
        init("zero_1", [0], i64),
        init("one_1", [1], i64),
        init("two_1", [2], i64),
        init("three_i64", [3], i64),
        init("five_i64", [5], i64),
        init("seven_1", [7], i64),
        init("ten_1", [10], i64),
        init("nine_1", [9], i64),
        init("sixteen_1", [16], i64),
        init("shape1", [1], i64),
        init("shape16", [16], i64),
        init("pow10_u32", (1 << np.arange(10, dtype=np.uint32)).reshape(1, 10), None),
        init("zero_u32", np.array(0, dtype=u32), None),
        init("zero_u8_scalar", np.array(0, dtype=u8), None),
        init("valid_centers_u32", np.array(valid_centers, dtype=u32), None),
        init("sh1", np.array(1, dtype=u32), None),
        init("sh2", np.array(2, dtype=u32), None),
        init("sh3", np.array(3, dtype=u32), None),
        init("five_u32", np.array(5, dtype=u32), None),
        init("starts_rows19", [1], i64),
        init("ends_rows19", [20], i64),
        init("row0_start", [1], i64),
        init("row0_end", [11], i64),
        init("row1_start", [2], i64),
        init("row1_end", [12], i64),
        init("row2_start", [3], i64),
        init("row2_end", [13], i64),
        init("row3_start", [4], i64),
        init("row3_end", [14], i64),
        init("pads_bottom25", [0, 9, 0, 9], i64),
        init("pads_bottom30", [0, 2, 0, 12], i64),
        init("g7", np.array([3, 2, 2, 1, 2, 2, 3], dtype=u8).reshape(1, 7), None),
        init("gplus2_u32", np.array([5, 4, 4, 3, 4, 4, 5], dtype=u32).reshape(1, 7), None),
        init(
            "row_iota30",
            np.array([0, 0, 0] + list(range(2, 19)) + [0] * 10, dtype=u8).reshape(30, 1),
            None,
        ),
        init("row_iota19", np.arange(19, dtype=u8).reshape(1, 1, 19, 1), None),
        init("u0", np.array([0], dtype=u8), None),
        init("u6", np.array([6], dtype=u8), None),
        init("u8v", np.array([8], dtype=u8), None),
        init("u9", np.array([9], dtype=u8), None),
        init("u16", np.array([16], dtype=u8), None),
        init("u99", np.array([99], dtype=u8), None),
    ]

    nodes: list[onnx.NodeProto] = [
        # Pack every row's non-background cells into one uint32 word.
        node("Einsum", ["input", "fg_ch", "col_bits_f"], "row_words_f", equation="nchw,c,w->nh"),
        node("Cast", ["row_words_f"], "row_words", to=TensorProto.UINT32),
    ]

    # Candidate rows are top rows 1..10.  Align the kite stencil taps so a
    # true origin has its center bit in the same position across all taps.
    # Row 2 is checked as two adjacent pairs, which certifies the four row-2
    # kite cells with two aligned taps instead of four.
    for dr in range(3):
        nodes.append(
            node(
                "Slice",
                ["row_words", f"row{dr}_start", f"row{dr}_end", "axis_row"],
                f"r{dr}",
            )
        )

    nodes.extend(
        [
            bit_shift("r1", "sh1", "LEFT", "r1_l1"),
            bit_shift("r1", "sh1", "RIGHT", "r1_r1"),
            bit_shift("r2", "sh1", "RIGHT", "r2_s1"),
            bit_and("r2", "r2_s1", "r2_pair"),
            bit_shift("r2_pair", "sh2", "LEFT", "r2_pair_l"),
            bit_shift("r2_pair", "sh1", "RIGHT", "r2_pair_r"),
        ]
    )
    aligned = ["r0", "r1_l1", "r1_r1", "r2_pair_l", "r2_pair_r"]

    cur = aligned[0]
    for i, x in enumerate(aligned[1:], 1):
        out = f"kand_{i}"
        nodes.append(bit_and(cur, x, out))
        cur = out
    nodes.append(bit_and(cur, "valid_centers_u32", "candidate_bits"))
    nodes.extend(
        [
            node("Cast", ["candidate_bits"], "candidate_i32", to=TensorProto.INT32),
            node("ArgMax", ["candidate_i32"], "i_idx", axis=1, keepdims=1),
            node("ReduceMax", ["candidate_i32", "axis_row"], "origin_i32", keepdims=1),
            node("Cast", ["origin_i32"], "origin_word", to=TensorProto.UINT32),
            node("BitShift", ["origin_word", "five_u32"], "origin_word_s5", direction="RIGHT"),
            node("Equal", ["origin_word_s5", "pow10_u32"], "origin_eq10"),
            node("Cast", ["origin_eq10"], "origin_eq10_u8", to=TensorProto.UINT8),
            node("ArgMax", ["origin_eq10_u8"], "center_rel", axis=1, keepdims=1),
            node("Add", ["center_rel", "five_i64"], "center_col"),
            node("Cast", ["i_idx"], "i_u8", to=TensorProto.UINT8),
            node("Cast", ["i_idx"], "i_u32", to=TensorProto.UINT32),
            node("Cast", ["center_rel"], "j_u8", to=TensorProto.UINT8),
        ]
    )

    # Build column row-bit words as a second packed view.  This avoids the
    # charged [19,7] relative-column expansion: each of the seven kite columns
    # is shifted below its certified lowest kite row, then tested for any bit.
    nodes.extend(
        [
            node("Einsum", ["input", "fg_ch", "col_bits_f"], "col_words_f", equation="nchw,c,h->nw"),
            node("Cast", ["col_words_f"], "col_words", to=TensorProto.UINT32),
            node("Sub", ["center_col", "three_i64"], "left_col"),
            node("Reshape", ["left_col", "shape1"], "left_col_1"),
            node("Add", ["left_col_1", "seven_1"], "left_col_end"),
            node("Slice", ["col_words", "left_col_1", "left_col_end", "axis_row"], "col7_words"),
            node("Add", ["i_u32", "gplus2_u32"], "shift7"),
            node("BitShift", ["col7_words", "shift7"], "col7_below", direction="RIGHT"),
            node("Greater", ["col7_below", "zero_u32"], "active7_flat"),
        ]
    )

    # The dot color is the non-background color present in the input after
    # removing the kite color sampled at the detected top vertex.
    nodes.extend(
        [
            node("Add", ["i_idx", "one_1"], "top_row"),
            node("Reshape", ["top_row", "shape1"], "top_row_1"),
            node("Reshape", ["center_col", "shape1"], "center_col_1"),
            node("Add", ["top_row_1", "one_1"], "top_row_end"),
            node("Add", ["center_col_1", "one_1"], "center_col_end"),
            node("Concat", ["zero_1", "zero_1", "top_row_1", "center_col_1"], "kite_starts", axis=0),
            node("Concat", ["one_1", "ten_1", "top_row_end", "center_col_end"], "kite_ends", axis=0),
            node("Slice", ["input", "kite_starts", "kite_ends", "axes4"], "kite10_float"),
            node("ReduceMax", ["input", "axes_spatial"], "present10_float", keepdims=1),
            node("Mul", ["present10_float", "fg4"], "present_fg_float"),
            node("Sub", ["present_fg_float", "kite10_float"], "dot10_float"),
            node("Add", ["g7", "i_u8"], "bottom7"),
            node("Where", ["active7_flat", "bottom7", "u99"], "bottom7_active"),
            node("Pad", ["bottom7_active", "pads_bottom25", "u99"], "bottom25", mode="constant"),
            node("Sub", ["nine_1", "center_rel"], "slice_start_raw"),
            node("Reshape", ["slice_start_raw", "shape1"], "slice_start"),
            node("Add", ["slice_start", "sixteen_1"], "slice_end"),
            node("Slice", ["bottom25", "slice_start", "slice_end", "axis_row"], "bottom16_active"),
            node("Pad", ["bottom16_active", "pads_bottom30", "u99"], "bottom30", mode="constant"),
            node("Greater", ["row_iota30", "bottom30"], "fill30"),
            node("Where", ["fill30", "dot10_float", "input"], "output"),
        ]
    )

    used = {name for nd in nodes for name in nd.input if name}
    inits = [t for t in inits if t.name in used]

    graph = helper.make_graph(
        nodes,
        "task165_packed_detector",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        inits,
        value_info=[
            helper.make_tensor_value_info("col7_words", TensorProto.UINT32, [1, 7]),
            helper.make_tensor_value_info("col7_below", TensorProto.UINT32, [1, 7]),
            helper.make_tensor_value_info("active7_flat", TensorProto.BOOL, [1, 7]),
            helper.make_tensor_value_info("bottom16_active", TensorProto.UINT8, [1, 16]),
            helper.make_tensor_value_info("kite10_float", TensorProto.FLOAT, [1, 10, 1, 1]),
            helper.make_tensor_value_info("dot10_float", TensorProto.FLOAT, [1, 10, 1, 1]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


def main() -> None:
    model = build()
    add_budget(model)
    onnx.save(model, OUT)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
