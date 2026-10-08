#!/usr/bin/env python3
"""task363 slim exception: att04 matcher minus W1z/k4 z-fingerprint (full-gate clean @ 2325)."""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def init(name, arr, dtype=None):
    arr = np.asarray(arr, dtype=dtype)
    return numpy_helper.from_array(arr, name)


nodes = [
    helper.make_node("Slice", ["input", "s_z_start", "s_z_end", "axes_chw"], ["z32"]),
    helper.make_node("Slice", ["input", "s_r_start", "s_r_end", "axes_chw"], ["r32"]),
    helper.make_node("Cast", ["z32"], ["z8"], to=TensorProto.UINT8),
    helper.make_node("Cast", ["r32"], ["r8"], to=TensorProto.UINT8),
    helper.make_node("ReduceMax", ["r8"], ["rowmask"], axes=[3], keepdims=1),
    helper.make_node("ReduceMax", ["r8"], ["colmask"], axes=[2], keepdims=1),
    helper.make_node("ArgMax", ["rowmask"], ["minrow64"], axis=2, keepdims=1),
    helper.make_node("ArgMax", ["colmask"], ["mincol64"], axis=3, keepdims=1),
    helper.make_node("Cast", ["minrow64"], ["minrow32"], to=TensorProto.INT32),
    helper.make_node("Cast", ["mincol64"], ["mincol32"], to=TensorProto.INT32),
    helper.make_node("Reshape", ["minrow32", "shape1"], ["minrow1"]),
    helper.make_node("Reshape", ["mincol32", "shape1"], ["mincol1"]),
    helper.make_node("Add", ["minrow1", "iota4_i32"], ["row_idx"]),
    helper.make_node("Add", ["mincol1", "iota4_i32"], ["col_idx"]),
    helper.make_node("Less", ["row_idx", "ten_i32"], ["row_valid"]),
    helper.make_node("Less", ["col_idx", "ten_i32"], ["col_valid"]),
    helper.make_node("Where", ["row_valid", "row_idx", "zero_i32"], ["row_clip"]),
    helper.make_node("Where", ["col_valid", "col_idx", "zero_i32"], ["col_clip"]),
    helper.make_node("Gather", ["r8", "row_clip"], ["gat_rows"], axis=2),
    helper.make_node("Gather", ["gat_rows", "col_clip"], ["normT"], axis=3),
    helper.make_node(
        "Slice", ["normT", "ft_start", "ft_end", "axes_hw", "ft_steps"], ["normTflip"]
    ),
    helper.make_node(
        "QLinearConv",
        ["z8", "one_scale", "zero_u8", "normT", "one_scale", "zero_u8", "one_scale", "zero_u8"],
        ["match"],
        kernel_shape=[4, 4],
        pads=[0, 0, 3, 3],
    ),
    helper.make_node("ReduceMax", ["match"], ["k"], keepdims=1),
    helper.make_node("Equal", ["match", "k"], ["valid"]),
    helper.make_node("Slice", ["r8", "r0_start", "r0_end", "axes_hw"], ["r0_box"]),
    helper.make_node(
        "QLinearConv",
        ["r0_box", "one_scale", "zero_u8", "W0r", "one_scale", "zero_u8", "one_scale", "zero_u8"],
        ["s0r"],
        kernel_shape=[3, 3],
    ),
    helper.make_node("Equal", ["s0r", "four_u8"], ["e0r"]),
    helper.make_node("Slice", ["r8", "r1_start", "r1_end", "axes_hw"], ["r1_box"]),
    helper.make_node("ReduceMin", ["r1_box"], ["s1r"], axes=[2, 3], keepdims=1),
    helper.make_node("Equal", ["s1r", "one_u8"], ["e1r_shape"]),
    helper.make_node("Or", ["e0r", "e1r_shape"], ["exception_case"]),
    helper.make_node("GatherND", ["valid", "anchor_erase_idx"], ["anchor_orig"]),
    helper.make_node("Not", ["exception_case"], ["not_exception_1111"]),
    helper.make_node("And", ["anchor_orig", "not_exception_1111"], ["anchor_updates"]),
    helper.make_node("ScatterND", ["valid", "anchor_erase_idx", "anchor_updates"], ["valid_fix"]),
    helper.make_node("Cast", ["valid_fix"], ["valid8"], to=TensorProto.UINT8),
    helper.make_node(
        "QLinearConv",
        ["valid8", "one_scale", "zero_u8", "normTflip", "one_scale", "zero_u8", "one_scale", "zero_u8"],
        ["paint8"],
        kernel_shape=[4, 4],
        pads=[3, 3, 0, 0],
    ),
    helper.make_node("Concat", ["z8", "r8", "ones10", "paint8"], ["state4"], axis=1),
    helper.make_node(
        "ConvInteger",
        ["state4", "Wout"],
        ["output"],
        kernel_shape=[1, 1],
        pads=[0, 0, 20, 20],
    ),
]

Wout = np.zeros((10, 4, 1, 1), dtype=np.int8)
Wout[0, 0, 0, 0] = 1
Wout[0, 3, 0, 0] = -1
Wout[2, 1, 0, 0] = 1
Wout[2, 3, 0, 0] = 1
Wout[5, 0, 0, 0] = -1
Wout[5, 1, 0, 0] = -1
Wout[5, 2, 0, 0] = 1

inits = [
    init("s_z_start", [0, 0, 0], np.int64),
    init("s_z_end", [1, 10, 10], np.int64),
    init("s_r_start", [2, 0, 0], np.int64),
    init("s_r_end", [3, 10, 10], np.int64),
    init("axes_chw", [1, 2, 3], np.int64),
    init("axes_hw", [2, 3], np.int64),
    init("r0_start", [0, 5], np.int64),
    init("r0_end", [3, 8], np.int64),
    init("r1_start", [5, 1], np.int64),
    init("r1_end", [6, 5], np.int64),
    init("one_scale", np.array(1.0, dtype=np.float32)),
    init("zero_u8", np.array(0, dtype=np.uint8)),
    init("one_u8", np.array([[[[1]]]], dtype=np.uint8)),
    init("ones10", np.ones((1, 1, 10, 10), dtype=np.uint8)),
    init("four_u8", np.array([[[[4]]]], dtype=np.uint8)),
    init(
        "anchor_erase_idx",
        np.array([[0, 0, 1, 3], [0, 0, 2, 6], [0, 0, 5, 1]], dtype=np.int64).reshape(1, 1, 1, 3, 4),
    ),
    init("W0r", np.array([[[[0, 1, 0], [1, 0, 1], [0, 1, 0]]]], dtype=np.uint8)),
    init("iota4_i32", [0, 1, 2, 3], np.int32),
    init("shape1", [1], np.int64),
    init("ten_i32", np.array(10, dtype=np.int32)),
    init("zero_i32", np.array(0, dtype=np.int32)),
    init("ft_start", [3, 3], np.int64),
    init("ft_end", [-5, -5], np.int64),
    init("ft_steps", [-1, -1], np.int64),
    init("Wout", Wout),
]

graph = helper.make_graph(
    nodes,
    "task363_slimexc",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
    [helper.make_tensor_value_info("output", TensorProto.INT32, [1, 10, 30, 30])],
    inits,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8
onnx.checker.check_model(model, full_check=True)
onnx.shape_inference.infer_shapes(model, strict_mode=True)
onnx.save(model, "task363.onnx")
print("saved task363.onnx")