#!/usr/bin/env python3
"""Build task396.onnx with a compact row-bitset locator and sparse renderer.

The prior banked graph localizes the widest box by slicing the selected color
to an 18x18 fp32 plane and materializing an 18x18 run-code plane.  This rebuild
projects the selected color directly from the free one-hot input into 18 row
bitsets, finds the longest contiguous run with uint32/uint16 bit operations,
then renders with a final fp16 Einsum.  The renderer uses signed logits:
background cells are positive on channel 0, nonzero cells are positive on the
dynamic marker channel, and all false classes are negative or zero.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


OUT = Path(__file__).resolve().with_name("task396.onnx")


def arr(name: str, values: np.ndarray, dtype: np.dtype | None = None) -> onnx.TensorProto:
    a = np.asarray(values, dtype=dtype)
    return numpy_helper.from_array(a, name)


def vi(name: str, dtype: int, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape)


def n(op: str, inputs: list[str], outputs: list[str], **attrs) -> onnx.NodeProto:
    return helper.make_node(op, inputs, outputs, **attrs)


def main() -> None:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    def add_init(name: str, values, dtype=None) -> str:
        inits.append(arr(name, values, dtype))
        return name

    # Opset 18 uses axes as an input for Reduce* and enables bitwise ops.
    add_init("axes23", np.array([2, 3], np.int64))
    add_init("mask0_f32", np.array([[0] + [1] * 9], np.float32))
    add_init("arange10_i64", np.arange(10, dtype=np.int64))

    col_pow = np.zeros(30, dtype=np.float32)
    col_pow[:18] = (2.0 ** np.arange(18)).astype(np.float32)
    add_init("col_pow30_f32", col_pow)
    add_init("slice0", np.array([0], np.int64))
    add_init("slice18", np.array([18], np.int64))

    add_init("shift1_u32", np.array(1, np.uint32))
    add_init("two_u16", np.array(2, np.uint16))
    add_init("zero_u16", np.array(0, np.uint16))
    add_init("zero_u8", np.array(0, np.uint8))
    for k in range(4, 9):
        add_init(f"w{k}_u8", np.array(k, np.uint8))
    add_init("one_i64", np.array([1], np.int64))
    add_init("four_i64", np.array(4, np.int64))
    add_init("eight_i32", np.array(8, np.int32))
    add_init("zero_i32_v", np.array([0], np.int32))
    add_init("one_i32_v", np.array([1], np.int32))
    add_init("axes123_i32", np.array([1, 2, 3], np.int32))
    add_init("ln2_f32", np.array(np.log(2.0), np.float32))
    add_init("half_f32", np.array(0.5, np.float32))
    add_init("half_f16", np.array(0.5, np.float16))
    black_vec = np.zeros((1, 10), dtype=np.float16)
    black_vec[0, 0] = 1
    add_init("black_vec_f16", black_vec)

    add_init("arange8_i64", np.arange(8, dtype=np.int64))
    map8_to_30 = np.zeros((8, 30), dtype=np.float16)
    map8_to_30[np.arange(8), np.arange(8)] = 1
    add_init("map8_to_30_f16", map8_to_30)

    # Color counts: dominant nonzero color is the box color; least nonzero is
    # the marker color used to repaint the crop.
    nodes += [
        n("ReduceSum", ["input", "axes23"], ["chsum"], keepdims=0),
        n("Mul", ["chsum", "mask0_f32"], ["chsum_no0"]),
        n("ArgMax", ["chsum_no0"], ["box_c"], axis=1, keepdims=0),
        n("Greater", ["chsum_no0", "half_f32"], ["has_pixels"]),
        n("Equal", ["arange10_i64", "box_c"], ["box_sel_b"]),
        n("Xor", ["has_pixels", "box_sel_b"], ["marker_vec_b"]),
        n("Cast", ["marker_vec_b"], ["marker_vec_f16"], to=TensorProto.FLOAT16),
        n("Cast", ["box_sel_b"], ["box_sel_f32"], to=TensorProto.FLOAT),
    ]

    # Project the dynamically selected color channel to row bitsets:
    # rowbits[r] = sum_c,w input[0,c,r,w] * 1[c == box_c] * 2**w.
    nodes += [
        n("Einsum", ["input", "box_sel_f32", "col_pow30_f32"], ["rowbits30_f32"], equation="nchw,c,w->nh"),
        n("Slice", ["rowbits30_f32", "slice0", "slice18", "one_i64"], ["rowbits18_f32"]),
        n("Cast", ["rowbits18_f32"], ["rowbits_u32"], to=TensorProto.UINT32),
    ]

    nodes += [
        n("BitShift", ["rowbits_u32", "shift1_u32"], ["rowbits_s1"], direction="RIGHT"),
        n("BitwiseAnd", ["rowbits_u32", "rowbits_s1"], ["run_ge2"]),
        n("BitShift", ["run_ge2", "shift1_u32"], ["run_ge2_s1_u32"], direction="RIGHT"),
        n("BitwiseAnd", ["run_ge2", "run_ge2_s1_u32"], ["run_ge3_u32"]),
        n("Cast", ["run_ge3_u32"], ["run_ge3"], to=TensorProto.UINT16),
    ]

    prev = "run_ge3"
    run_names: dict[int, str] = {}
    for k in range(4, 9):
        shifted = f"run_ge{k - 1}_s1"
        out = f"run_ge{k}"
        nodes.append(n("Div", [prev, "two_u16"], [shifted]))
        nodes.append(n("BitwiseAnd", [prev, shifted], [out]))
        prev = out
        run_names[k] = out

    # Per row, keep the longest run length in uint8.  Widths are generator-
    # certified to be in [4, 8] for the widest box, and box widths are sampled without replacement,
    # so the widest run identifies the target box's top and bottom rows.
    width_prev = "zero_u8"
    for k in range(4, 9):
        has = f"has_run{k}"
        width = f"row_width{k}"
        nodes.append(n("Greater", [run_names[k], "zero_u16"], [has]))
        nodes.append(n("Where", [has, f"w{k}_u8", width_prev], [width]))
        width_prev = width
    nodes += [
        n("ReduceMax", [width_prev, "one_i64"], ["OW_u8"], keepdims=1),
        n("Equal", [width_prev, "OW_u8"], ["row_is_widest"]),
        n("Cast", ["row_is_widest"], ["row_is_widest_u8"], to=TensorProto.UINT8),
        n("ArgMax", ["row_is_widest_u8"], ["r0"], axis=1, keepdims=1, select_last_index=0),
        n("ArgMax", ["row_is_widest_u8"], ["r1"], axis=1, keepdims=1, select_last_index=1),
        n("Sub", ["r1", "r0"], ["OH_m1"]),
        n("Add", ["OH_m1", "one_i64"], ["OH"]),
        n("Cast", ["OW_u8"], ["OW"], to=TensorProto.INT64),
        n("Cast", ["r0"], ["r0_i32"], to=TensorProto.INT32),
        n("Reshape", ["r0_i32", "one_i64"], ["r0_v"]),
    ]

    # Select the run-start bitset for the target top row and decode its single
    # set bit to the left edge c0.
    for k in range(4, 9):
        nodes.append(n("Gather", [run_names[k], "r0_v"], [f"run{k}_at_r0"], axis=1))
    nodes += [
        n(
            "Concat",
            [f"run{k}_at_r0" for k in range(4, 9)],
            ["run_at_r0_stack"],
            axis=1,
        ),
        n("Sub", ["OW", "four_i64"], ["widx_raw"]),
        n("Clip", ["widx_raw", "slice0", "four_i64"], ["widx_clip"]),
        n("GatherElements", ["run_at_r0_stack", "widx_clip"], ["start_bits"], axis=1),
    ]
    nodes += [
        n("Cast", ["start_bits"], ["start_bits_f32"], to=TensorProto.FLOAT),
        n("Log", ["start_bits_f32"], ["c0_ln"]),
        n("Div", ["c0_ln", "ln2_f32"], ["c0_log2"]),
        n("Add", ["c0_log2", "half_f32"], ["c0_round"]),
        n("Cast", ["c0_round"], ["c0"], to=TensorProto.INT64),
    ]

    # Dynamic 8x8 crop of input channel 0.  Channel 0 is one for black, zero
    # for all nonzero content; Equal(..., 0) is therefore the output content mask.
    nodes += [
        n("Cast", ["c0"], ["c0_i32"], to=TensorProto.INT32),
        n("Reshape", ["c0_i32", "one_i64"], ["c0_v"]),
        n("Add", ["r0_i32", "eight_i32"], ["r8"]),
        n("Add", ["c0_i32", "eight_i32"], ["c8"]),
        n("Reshape", ["r8", "one_i64"], ["r8_v"]),
        n("Reshape", ["c8", "one_i64"], ["c8_v"]),
        n("Concat", ["zero_i32_v", "r0_v", "c0_v"], ["bg_starts"], axis=0),
        n("Concat", ["one_i32_v", "r8_v", "c8_v"], ["bg_ends"], axis=0),
        n("Slice", ["input", "bg_starts", "bg_ends", "axes123_i32"], ["bg8_f32"]),
        n("Less", ["arange8_i64", "OH"], ["win_r"]),
        n("Less", ["arange8_i64", "OW"], ["win_c"]),
        n("Sub", ["black_vec_f16", "marker_vec_f16"], ["class_coeff_f16"]),
        n("Cast", ["bg8_f32"], ["bg8_f16"], to=TensorProto.FLOAT16),
        n("Sub", ["bg8_f16", "half_f16"], ["sgn8_f16"]),
        n("Cast", ["win_r"], ["win_r_f16"], to=TensorProto.FLOAT16),
        n("Cast", ["win_c"], ["win_c_f16"], to=TensorProto.FLOAT16),
        n(
            "Einsum",
            [
                "sgn8_f16",
                "win_r_f16",
                "win_c_f16",
                "class_coeff_f16",
                "map8_to_30_f16",
                "map8_to_30_f16",
            ],
            ["output"],
            equation="nqij,ni,nj,nk,ir,jc->nkrc",
        ),
    ]

    input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 10, 30, 30])
    value_info = [
        vi("chsum", TensorProto.FLOAT, [1, 10]),
        vi("chsum_no0", TensorProto.FLOAT, [1, 10]),
        vi("box_c", TensorProto.INT64, [1]),
        vi("has_pixels", TensorProto.BOOL, [1, 10]),
        vi("box_sel_b", TensorProto.BOOL, [10]),
        vi("marker_vec_b", TensorProto.BOOL, [1, 10]),
        vi("marker_vec_f16", TensorProto.FLOAT16, [1, 10]),
        vi("box_sel_f32", TensorProto.FLOAT, [10]),
        vi("rowbits30_f32", TensorProto.FLOAT, [1, 30]),
        vi("rowbits18_f32", TensorProto.FLOAT, [1, 18]),
        vi("rowbits_u32", TensorProto.UINT32, [1, 18]),
    ]
    value_info += [
        vi("rowbits_s1", TensorProto.UINT32, [1, 18]),
        vi("run_ge2", TensorProto.UINT32, [1, 18]),
        vi("run_ge2_s1_u32", TensorProto.UINT32, [1, 18]),
        vi("run_ge3_u32", TensorProto.UINT32, [1, 18]),
        vi("run_ge3", TensorProto.UINT16, [1, 18]),
    ]
    for k in range(4, 9):
        value_info.append(vi(f"run_ge{k - 1}_s1", TensorProto.UINT16, [1, 18]))
        value_info.append(vi(f"run_ge{k}", TensorProto.UINT16, [1, 18]))
    for k in range(4, 9):
        value_info.append(vi(f"has_run{k}", TensorProto.BOOL, [1, 18]))
        value_info.append(vi(f"row_width{k}", TensorProto.UINT8, [1, 18]))
        value_info.append(vi(f"run{k}_at_r0", TensorProto.UINT16, [1, 1]))
    value_info += [
        vi("OW_u8", TensorProto.UINT8, [1, 1]),
        vi("row_is_widest", TensorProto.BOOL, [1, 18]),
        vi("row_is_widest_u8", TensorProto.UINT8, [1, 18]),
        vi("r0", TensorProto.INT64, [1, 1]),
        vi("r1", TensorProto.INT64, [1, 1]),
        vi("OH_m1", TensorProto.INT64, [1, 1]),
        vi("OH", TensorProto.INT64, [1, 1]),
        vi("OW", TensorProto.INT64, [1, 1]),
        vi("r0_i32", TensorProto.INT32, [1, 1]),
        vi("r0_v", TensorProto.INT32, [1]),
    ]
    value_info += [
        vi("run_at_r0_stack", TensorProto.UINT16, [1, 5]),
        vi("widx_raw", TensorProto.INT64, [1, 1]),
        vi("widx_clip", TensorProto.INT64, [1, 1]),
        vi("start_bits", TensorProto.UINT16, [1, 1]),
        vi("start_bits_f32", TensorProto.FLOAT, [1, 1]),
        vi("c0_ln", TensorProto.FLOAT, [1, 1]),
        vi("c0_log2", TensorProto.FLOAT, [1, 1]),
        vi("c0_round", TensorProto.FLOAT, [1, 1]),
        vi("c0", TensorProto.INT64, [1, 1]),
        vi("c0_i32", TensorProto.INT32, [1, 1]),
        vi("c0_v", TensorProto.INT32, [1]),
        vi("r8", TensorProto.INT32, [1, 1]),
        vi("c8", TensorProto.INT32, [1, 1]),
        vi("r8_v", TensorProto.INT32, [1]),
        vi("c8_v", TensorProto.INT32, [1]),
        vi("bg_starts", TensorProto.INT32, [3]),
        vi("bg_ends", TensorProto.INT32, [3]),
        vi("bg8_f32", TensorProto.FLOAT, [1, 1, 8, 8]),
        vi("win_r", TensorProto.BOOL, [1, 8]),
        vi("win_c", TensorProto.BOOL, [1, 8]),
        vi("class_coeff_f16", TensorProto.FLOAT16, [1, 10]),
        vi("bg8_f16", TensorProto.FLOAT16, [1, 1, 8, 8]),
        vi("sgn8_f16", TensorProto.FLOAT16, [1, 1, 8, 8]),
        vi("win_r_f16", TensorProto.FLOAT16, [1, 8]),
        vi("win_c_f16", TensorProto.FLOAT16, [1, 8]),
    ]

    graph = helper.make_graph(nodes, "task396_sparse_rowbitset", [input_vi], [output_vi], inits)
    graph.value_info.extend(value_info)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    model.producer_name = "task396_sparse_rowbitset_build"

    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, OUT)
    sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(f"wrote {OUT} sha256={sha} bytes={OUT.stat().st_size}")


if __name__ == "__main__":
    main()
