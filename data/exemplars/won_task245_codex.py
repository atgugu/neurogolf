#!/usr/bin/env python3
"""Build task245.onnx with relowered u8 state and a terminal ConvInteger renderer.

Rule: shift red by green_ul - red_ul + (1, 1), keep green corners, and clear old red.

Priced designs before build:
- u8 relower + synthesized green frame (built): memory ~= 1555, params ~= 159,
  cost ~= 1714. Tensors: red f32/u8 [1,1,9,9] (324+81), green-anchor
  f32/u8 [1,1,4,4] (64+16), row/col projections and anchors (88), two i32
  gather indices [9] (72), shifted-red gather/pad (81+81+100), four tiny
  c0 extent probes plus inside mask (198), green edge-LUT gathers/reshapes
  and mask (140), terminal tail3 (300).
- Full 10x10 green plane relower: memory ~= 1603, params ~= 123, cost ~= 1726.
  It closes only at the exact ceiling and is less robust to repricing, so the
  9x9 red crop + LUT frame is built instead.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as TP
from onnx import helper, numpy_helper


OUT = Path(__file__).with_name("task245.onnx")
H = W = 10
RH = RW = 9


def init(name: str, arr) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr), name=name)


def anchor_nodes(prefix: str, mask: str, n: int) -> list[onnx.NodeProto]:
    return [
        helper.make_node("ReduceMax", [mask], [f"{prefix}_rows"], axes=[1, 3], keepdims=0),
        helper.make_node("ReduceMax", [mask], [f"{prefix}_cols"], axes=[1, 2], keepdims=0),
        helper.make_node(
            "ArgMax",
            [f"{prefix}_rows"],
            [f"{prefix}_r64"],
            axis=1,
            keepdims=0,
            select_last_index=0,
        ),
        helper.make_node(
            "ArgMax",
            [f"{prefix}_cols"],
            [f"{prefix}_c64"],
            axis=1,
            keepdims=0,
            select_last_index=0,
        ),
        helper.make_node("Cast", [f"{prefix}_r64"], [f"{prefix}_r32"], to=TP.INT32),
        helper.make_node("Cast", [f"{prefix}_c64"], [f"{prefix}_c32"], to=TP.INT32),
    ]


def slice_node(st: str, en: str, out: str) -> onnx.NodeProto:
    return helper.make_node("Slice", ["input", st, en], [out])


def edge_lut() -> np.ndarray:
    lut = np.zeros((4, 10), dtype=np.uint8)
    for i in range(4):
        lut[i, i] = 1
        lut[i, i + 6] = 1
    return lut


def tail_weights() -> np.ndarray:
    w = np.zeros((10, 3, 1, 1), dtype=np.int8)
    # tail channels are [shifted_red, green, inside_grid].
    w[0, :, 0, 0] = [-1, -1, 1]
    w[2, :, 0, 0] = [1, 0, 0]
    w[3, :, 0, 0] = [0, 1, 0]
    return w


def build() -> onnx.ModelProto:
    inits = [
        init("ch2_st", np.array([0, 2, 0, 0], np.int64)),
        init("ch2_en", np.array([1, 3, RH, RW], np.int64)),
        init("g4_st", np.array([0, 3, 0, 0], np.int64)),
        init("g4_en", np.array([1, 4, 4, 4], np.int64)),
        init("c0_r0_st", np.array([0, 0, 7, 0], np.int64)),
        init("c0_r0_en", np.array([1, 1, 10, 1], np.int64)),
        init("c0_r5_st", np.array([0, 0, 7, 5], np.int64)),
        init("c0_r5_en", np.array([1, 1, 10, 6], np.int64)),
        init("c0_c0_st", np.array([0, 0, 0, 7], np.int64)),
        init("c0_c0_en", np.array([1, 1, 1, 10], np.int64)),
        init("c0_c5_st", np.array([0, 0, 5, 7], np.int64)),
        init("c0_c5_en", np.array([1, 1, 6, 10], np.int64)),
        init("row_base", np.ones((1, 1, 7, 1), dtype=np.uint8)),
        init("col_base", np.ones((1, 1, 1, 7), dtype=np.uint8)),
        init("iota9_i32", np.arange(RH, dtype=np.int32)),
        init("one_i32", np.array(1, np.int32)),
        init("shift_pad", np.array([0, 0, 0, 0, 0, 0, 1, 1], np.int64)),
        init("zero_u8", np.array(0, dtype=np.uint8)),
        init("edge_lut", edge_lut()),
        init("row_shape", np.array([1, 1, H, 1], dtype=np.int64)),
        init("col_shape", np.array([1, 1, 1, W], dtype=np.int64)),
        init("tail_w", tail_weights()),
    ]

    nodes: list[onnx.NodeProto] = [
        slice_node("ch2_st", "ch2_en", "s2_f"),
        slice_node("g4_st", "g4_en", "g4_f"),
        helper.make_node("Cast", ["s2_f"], ["shape_u8"], to=TP.UINT8),
        helper.make_node("Cast", ["g4_f"], ["g4_u8"], to=TP.UINT8),
    ]
    nodes += anchor_nodes("s", "shape_u8", RH)
    nodes += anchor_nodes("m", "g4_u8", 4)
    nodes += [
        helper.make_node("Sub", ["m_r32", "s_r32"], ["dr0"]),
        helper.make_node("Sub", ["m_c32", "s_c32"], ["dc0"]),
        helper.make_node("Add", ["dr0", "one_i32"], ["dr_s"]),
        helper.make_node("Add", ["dc0", "one_i32"], ["dc_s"]),
        helper.make_node("Sub", ["iota9_i32", "dr_s"], ["idx_r"]),
        helper.make_node("Sub", ["iota9_i32", "dc_s"], ["idx_c"]),
        helper.make_node("Gather", ["shape_u8", "idx_r"], ["g_row"], axis=2),
        helper.make_node("Gather", ["g_row", "idx_c"], ["shifted9_u8"], axis=3),
        helper.make_node("Pad", ["shifted9_u8", "shift_pad", "zero_u8"], ["shifted_u8"]),
        helper.make_node("Gather", ["edge_lut", "m_r32"], ["green_row_flat"], axis=0),
        helper.make_node("Gather", ["edge_lut", "m_c32"], ["green_col_flat"], axis=0),
        helper.make_node("Reshape", ["green_row_flat", "row_shape"], ["green_row"]),
        helper.make_node("Reshape", ["green_col_flat", "col_shape"], ["green_col"]),
        helper.make_node("Mul", ["green_row", "green_col"], ["green_u8"]),
        # Height > 7/8/9 is detected by channel-0 probes at columns 0 and 5.
        slice_node("c0_r0_st", "c0_r0_en", "c0_r0_f"),
        slice_node("c0_r5_st", "c0_r5_en", "c0_r5_f"),
        helper.make_node("Max", ["c0_r0_f", "c0_r5_f"], ["h_extra_f"]),
        helper.make_node("Cast", ["h_extra_f"], ["h_extra_u8"], to=TP.UINT8),
        helper.make_node("Concat", ["row_base", "h_extra_u8"], ["row_inside"], axis=2),
        # Width > 7/8/9 is symmetric, using channel-0 probes at rows 0 and 5.
        slice_node("c0_c0_st", "c0_c0_en", "c0_c0_f"),
        slice_node("c0_c5_st", "c0_c5_en", "c0_c5_f"),
        helper.make_node("Max", ["c0_c0_f", "c0_c5_f"], ["w_extra_f"]),
        helper.make_node("Cast", ["w_extra_f"], ["w_extra_u8"], to=TP.UINT8),
        helper.make_node("Concat", ["col_base", "w_extra_u8"], ["col_inside"], axis=3),
        helper.make_node("Mul", ["row_inside", "col_inside"], ["inside_u8"]),
        helper.make_node("Concat", ["shifted_u8", "green_u8", "inside_u8"], ["tail3"], axis=1),
        helper.make_node("ConvInteger", ["tail3", "tail_w"], ["output"], pads=[0, 0, 20, 20]),
    ]

    graph = helper.make_graph(
        nodes,
        "task245_sparse_extent",
        [helper.make_tensor_value_info("input", TP.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TP.INT32, [1, 10, 30, 30])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


def static_budget(model: onnx.ModelProto) -> tuple[int, int]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    value_info = {v.name: v for v in list(inferred.value_info) + list(inferred.output)}
    memory = 0
    for node in inferred.node:
        for out in node.output:
            if not out or out == "output":
                continue
            tt = value_info[out].type.tensor_type
            dims = [d.dim_value for d in tt.shape.dim]
            itemsize = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tt.elem_type)).itemsize
            memory += math.prod(dims) * itemsize
    params = sum(math.prod(t.dims) if t.dims else 1 for t in model.graph.initializer)
    return memory, params


def main() -> None:
    model = build()
    so = ort.SessionOptions()
    so.log_severity_level = 4
    sess = ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])
    probe = np.zeros((1, 10, 30, 30), dtype=np.float32)
    probe[0, 0, 0, 0] = 1.0
    sess.run(["output"], {"input": probe})
    onnx.save(model, OUT)
    memory, params = static_budget(model)
    print(f"saved {OUT}")
    print(f"static budget: mem={memory} params={params} cost={memory + params}")


if __name__ == "__main__":
    main()
