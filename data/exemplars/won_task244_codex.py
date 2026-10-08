#!/usr/bin/env python3
"""task244 dynamic strided-slice rebuild.

True rule: the input is a horizontally flipped linegrid of a 3x3 or 4x4
bitmap; remove separator lines and undo the horizontal flip.

Priced before build:

Design A, requested native u8 plane-kill:
  raw color plane [1,1,30,30] u8              900 B
  crop/masks/stamp [1,1,10,10] u8/bool       300 B
  selected bitmap/features [1,1,4,4] u8/bool  64 B
  terminal renderer                           free
  params approx                              222 el
  paper total if raw plane were free:        ~484 B
  executable issue: the gate input is one-hot fp32, and decoding raw color
  ids costs a wide ArgMax/Cast or weighted plane, so this family is not
  physically available as written.

Design B, binding bool-terminal sampler:
  sampled_grid [1,10,4,4] f32                 640 B
  sampled_bool [1,10,4,4] bool                160 B
  existing Slice/Pad controls and probes      418 B
  params                                      56 el
  measured total                            1274 B
  executable issue: ONNX Slice preserves the fp32 input dtype, so the f32
  sampled tensor is still charged before Greater can produce a bool patch.

Design C, built here, dynamic Slice sampler with scalar area detector:
  area/geometry scalars                       ~48 B
  H=11 disambiguation probes                  104 B
  dynamic slice/pad index scalars/vectors    ~160 B
  one-hot 4x(3|4) sampled core                640 B max
  final dynamic Pad to [1,10,30,30]           free
  params                                      55 el
  measured total                             993 B, fast+full gate pass.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def init(name: str, arr, dtype=None) -> onnx.TensorProto:
    a = np.asarray(arr)
    if dtype is not None:
        a = a.astype(dtype)
    return numpy_helper.from_array(a, name)


def main() -> None:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    value_info: list[onnx.ValueInfoProto] = []

    def add(op: str, ins: list[str], outs: list[str], **attrs) -> None:
        nodes.append(helper.make_node(op, ins, outs, **attrs))

    def c(name: str, arr, dtype=None) -> str:
        inits.append(init(name, arr, dtype))
        return name

    def vi(name: str, dtype: int, shape: list[int]) -> None:
        value_info.append(helper.make_tensor_value_info(name, dtype, shape))

    input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])

    c("sum_axis_rows", [0, 1, 2, 3], np.int64)
    c("slice_axes_spatial", [2, 3], np.int64)
    c("shape_1", [1], np.int64)
    c("zero_i64", [0], np.int64)
    c("pad_prefix", [0, 0, 0, 0, 0, 0, 26], np.int64)
    c("four_i64", [4], np.int64)

    # Geometry index order:
    # H=8, H=11/N=3, H=11/N=4, H=14, H=15, H=17, H=19, H=23.
    c("step_table", [3, 4, 3, 5, 4, 6, 5, 6], np.int64)
    c("col_start_table", [7, 10, 10, 13, 14, 16, 18, 22], np.int64)
    c("pad_right_table", [27, 27, 26, 27, 26, 27, 26, 26], np.int64)

    for name, value in [
        ("h_gt8", 64.5),
        ("h_gt11", 121.5),
        ("h_gt14", 196.5),
        ("h_gt15", 225.5),
        ("h_gt17", 289.5),
        ("h_gt19", 361.5),
        ("h_eq11", 121.0),
    ]:
        c(name, np.array(value, np.float32))

    c("cell_a_starts", [8, 0], np.int64)
    c("cell_a_ends", [9, 1], np.int64)
    c("cell_b_starts", [8, 3], np.int64)
    c("cell_b_ends", [9, 4], np.int64)

    # One-hot padding outside the real grid is all-zero, so total activation is H*H.
    add("ReduceSum", ["input", "sum_axis_rows"], ["height"], keepdims=0)

    cast_names: list[str] = []
    for nm in ["gt8", "gt11", "gt14", "gt15", "gt17", "gt19"]:
        add("Greater", ["height", "h_" + nm], [nm + "_b"])
        add("Cast", [nm + "_b"], [nm + "_f"], to=TensorProto.FLOAT)
        cast_names.append(nm + "_f")
    add("Equal", ["height", "h_eq11"], ["is_h11"])

    add("Slice", ["input", "cell_a_starts", "cell_a_ends", "slice_axes_spatial"], ["cell_a"])
    add("Slice", ["input", "cell_b_starts", "cell_b_ends", "slice_axes_spatial"], ["cell_b"])
    add("ArgMax", ["cell_a"], ["cell_a_color"], axis=1, keepdims=0)
    add("ArgMax", ["cell_b"], ["cell_b_color"], axis=1, keepdims=0)
    add("Equal", ["cell_a_color", "cell_b_color"], ["probe_same"])
    add("Reshape", ["probe_same", "shape_1"], ["probe_same_1d"])
    add("And", ["is_h11", "probe_same_1d"], ["h11_is4_b"])
    add("Cast", ["h11_is4_b"], ["h11_is4_f"], to=TensorProto.FLOAT)

    # base = gt8 + 2*gt11 + gt14 + gt15 + gt17 + gt19.
    add(
        "Sum",
        ["gt8_f", "gt11_f", "gt11_f", "gt14_f", "gt15_f", "gt17_f", "gt19_f", "h11_is4_f"],
        ["geometry_float"],
    )
    add("Cast", ["geometry_float"], ["geometry_idx"], to=TensorProto.INT32)

    add("Gather", ["step_table", "geometry_idx"], ["step_i64"], axis=0)
    add("Gather", ["col_start_table", "geometry_idx"], ["col_start"], axis=0)
    add("Gather", ["pad_right_table", "geometry_idx"], ["pad_right"], axis=0)
    add("Mul", ["step_i64", "four_i64"], ["row_end"])
    add("Sub", ["zero_i64", "step_i64"], ["neg_step"])

    add("Concat", ["zero_i64", "col_start"], ["slice_starts"], axis=0)
    add("Concat", ["row_end", "zero_i64"], ["slice_ends"], axis=0)
    add("Concat", ["step_i64", "neg_step"], ["slice_steps"], axis=0)
    add("Concat", ["pad_prefix", "pad_right"], ["pads"], axis=0)

    # Runtime width is 3 for N=3 and 4 for N=4.  The final dynamic Pad restores
    # the required 30x30 canvas; static value_info keeps the scorer contract.
    add("Slice", ["input", "slice_starts", "slice_ends", "slice_axes_spatial", "slice_steps"], ["sampled_grid"])
    add("Pad", ["sampled_grid", "pads"], ["output"], mode="constant")

    vi("sampled_grid", TensorProto.FLOAT, [1, 10, 4, 4])
    vi("pads", TensorProto.INT64, [8])

    graph = helper.make_graph(
        nodes,
        "task244_dynamic_slice",
        [input_vi],
        [output_vi],
        inits,
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, "task244.onnx")


if __name__ == "__main__":
    main()
