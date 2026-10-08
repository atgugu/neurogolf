#!/usr/bin/env python3
"""task134 attempt 8: scaled-fp16 selector with a rank-3 color broadcast trim.

True rule: detect the magnified solid-block Conway sprite color, sample its 3x3
macro cells at stride K, and render occupied cells in the other nonzero color.

Budgeted families before build:
  A. Sparse-stamp LUT decoder, target 512B:
     planned params [10,3,3] u8 sprite table = 90, [30] row weights = 30,
     [30] col weights = 30, [10] colors/counts = 10, packed LUT = 352.
     Rejected after `probe_sparse_stamp.py`: 304 visible/generated cases contain
     143 distinct Conway patterns, so the 3x3 pattern is not keyed by color or a
     10-entry sprite table and must be sampled from runtime input.
  B. Current verified selector with rank-3 color broadcast, target <1083:
     tensors: selector f32 moments 3*[10]=120B; selector fp16 scratch 11*[10]*2
     =220B; color/output masks 3*[10]=30B; row projection [30] f32=120B;
     weighted column projection [30] f32=120B; row bool/u8 occupancy 2*[30]=60B;
     int64 argmax scalars 4*8=32B; int32 slice arithmetic scalars/vectors=76B;
     sampled tile 36B; pattern bool 9B; rank-3 color broadcast [10,1,1] u8=10B;
     small output [1,10,3,3] u8=90B; params 139; scorer target 1082B.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def init(name: str, array, dtype) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(array, dtype=dtype), name=name)


def scalar(name: str, value, dtype) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)


def vi(name: str, dtype: int, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape)


def build() -> onnx.ModelProto:
    nodes = [
        helper.make_node("Einsum", ["input", "nz_mask"], ["n_f32"], equation="bcyx,c->c"),
        helper.make_node("Einsum", ["input", "coord", "nz_mask"], ["Sy_f32"], equation="bcyx,y,c->c"),
        helper.make_node("Einsum", ["input", "coordsq", "nz_mask"], ["Syy_f32"], equation="bcyx,y,c->c"),
        helper.make_node("Cast", ["n_f32"], ["n"], to=TensorProto.FLOAT16),
        helper.make_node("Cast", ["Sy_f32"], ["Sy"], to=TensorProto.FLOAT16),
        helper.make_node("Cast", ["Syy_f32"], ["Syy"], to=TensorProto.FLOAT16),
        helper.make_node("Mul", ["n", "Syy"], ["nSyy"]),
        helper.make_node("Mul", ["Sy", "Sy"], ["Sy2"]),
        helper.make_node("Sub", ["nSyy", "Sy2"], ["Uy"]),
        helper.make_node("Add", ["Uy", "eps16"], ["den"]),
        helper.make_node("Mul", ["n", "eps16"], ["n_scaled"]),
        helper.make_node("Pow", ["n_scaled", "three16"], ["n3_scaled"]),
        helper.make_node("Div", ["n3_scaled", "den"], ["score"]),
        helper.make_node("ArgMax", ["score"], ["m_idx1"], axis=0, keepdims=1),
        helper.make_node("Equal", ["color_ids", "m_idx1"], ["ohm_b"]),
        helper.make_node("Cast", ["ohm_b"], ["ohm_f"], to=TensorProto.FLOAT),
        helper.make_node("Cast", ["n"], ["n_u8"], to=TensorProto.UINT8),
        helper.make_node("Where", ["ohm_b", "zero_u8", "n_u8"], ["ohs_u8"]),
        helper.make_node("Einsum", ["ohm_f", "input"], ["rm"], equation="c,bcyx->y"),
        helper.make_node("Einsum", ["ohm_f", "input", "cweight"], ["cscore"], equation="c,bcyx,x->x"),
        helper.make_node("Greater", ["rm", "zero_f"], ["rocc_b"]),
        helper.make_node("Cast", ["rocc_b"], ["rocc"], to=TensorProto.UINT8),
        helper.make_node("ArgMax", ["rocc"], ["rmin"], axis=0, keepdims=1),
        helper.make_node("ArgMax", ["cscore"], ["cmin"], axis=0, keepdims=1),
        helper.make_node("ArgMax", ["rocc"], ["rmax"], axis=0, keepdims=1, select_last_index=1),
        helper.make_node("Cast", ["m_idx1"], ["m_i32"], to=TensorProto.INT32),
        helper.make_node("Cast", ["rmin"], ["rmin_i32"], to=TensorProto.INT32),
        helper.make_node("Cast", ["cmin"], ["cmin_i32"], to=TensorProto.INT32),
        helper.make_node("Cast", ["rmax"], ["rmax_i32"], to=TensorProto.INT32),
        helper.make_node("Sub", ["rmax_i32", "rmin_i32"], ["rspan"]),
        helper.make_node("Add", ["rspan", "one_i32"], ["rext"]),
        helper.make_node("Div", ["rext", "three_i32"], ["mag"]),
        helper.make_node("Concat", ["m_i32", "rmin_i32", "cmin_i32"], ["starts"], axis=0),
        helper.make_node("Concat", ["one_i32", "rext", "rext"], ["span"], axis=0),
        helper.make_node("Add", ["starts", "span"], ["ends"]),
        helper.make_node("Concat", ["one_i32", "mag", "mag"], ["steps"], axis=0),
        helper.make_node("Slice", ["input", "starts", "ends", "axes123", "steps"], ["sel"]),
        helper.make_node("Greater", ["sel", "zero_f"], ["patt"]),
        helper.make_node("Unsqueeze", ["ohs_u8", "unsq_axes"], ["ohs_r"]),
        helper.make_node("Where", ["patt", "ohs_r", "e0_4"], ["small"]),
        helper.make_node("Pad", ["small", "pads", "zero_u8"], ["output"], mode="constant"),
    ]

    inits = [
        init("coord", np.arange(30, dtype=np.float32) / 30.0, np.float32),
        init("coordsq", (np.arange(30, dtype=np.float32) / 30.0) ** 2, np.float32),
        init("nz_mask", [0.0] + [1.0] * 9, np.float32),
        scalar("eps16", 0.001, np.float16),
        scalar("three16", 3.0, np.float16),
        scalar("zero_f", 0.0, np.float32),
        init("cweight", 4.0 ** (29 - np.arange(30, dtype=np.float32)), np.float32),
        init("color_ids", np.arange(10, dtype=np.int64), np.int64),
        init("three_i32", [3], np.int32),
        init("one_i32", [1], np.int32),
        init("axes123", [1, 2, 3], np.int32),
        init("unsq_axes", [1, 2], np.int64),
        init("e0_4", np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8),
        init("pads", [0, 0, 0, 0, 0, 0, 27, 27], np.int64),
        scalar("zero_u8", 0, np.uint8),
    ]

    graph = helper.make_graph(
        nodes,
        "task134_one_axis_moment",
        [vi("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [vi("output", TensorProto.UINT8, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 14)])
    model = onnx.shape_inference.infer_shapes(model, strict_mode=False)

    manual = {
        "n_f32": (TensorProto.FLOAT, [10]),
        "Sy_f32": (TensorProto.FLOAT, [10]),
        "Syy_f32": (TensorProto.FLOAT, [10]),
        "n": (TensorProto.FLOAT16, [10]),
        "Sy": (TensorProto.FLOAT16, [10]),
        "Syy": (TensorProto.FLOAT16, [10]),
        "nSyy": (TensorProto.FLOAT16, [10]),
        "Sy2": (TensorProto.FLOAT16, [10]),
        "Uy": (TensorProto.FLOAT16, [10]),
        "den": (TensorProto.FLOAT16, [10]),
        "n_scaled": (TensorProto.FLOAT16, [10]),
        "n3_scaled": (TensorProto.FLOAT16, [10]),
        "score": (TensorProto.FLOAT16, [10]),
        "m_idx1": (TensorProto.INT64, [1]),
        "ohm_b": (TensorProto.BOOL, [10]),
        "ohm_f": (TensorProto.FLOAT, [10]),
        "n_u8": (TensorProto.UINT8, [10]),
        "ohs_u8": (TensorProto.UINT8, [10]),
        "rm": (TensorProto.FLOAT, [30]),
        "cscore": (TensorProto.FLOAT, [30]),
        "rocc_b": (TensorProto.BOOL, [30]),
        "rocc": (TensorProto.UINT8, [30]),
        "rmin": (TensorProto.INT64, [1]),
        "cmin": (TensorProto.INT64, [1]),
        "rmax": (TensorProto.INT64, [1]),
        "m_i32": (TensorProto.INT32, [1]),
        "rmin_i32": (TensorProto.INT32, [1]),
        "cmin_i32": (TensorProto.INT32, [1]),
        "rmax_i32": (TensorProto.INT32, [1]),
        "rspan": (TensorProto.INT32, [1]),
        "rext": (TensorProto.INT32, [1]),
        "mag": (TensorProto.INT32, [1]),
        "starts": (TensorProto.INT32, [3]),
        "span": (TensorProto.INT32, [3]),
        "ends": (TensorProto.INT32, [3]),
        "steps": (TensorProto.INT32, [3]),
        "sel": (TensorProto.FLOAT, [1, 1, 3, 3]),
        "patt": (TensorProto.BOOL, [1, 1, 3, 3]),
        "ohs_r": (TensorProto.UINT8, [10, 1, 1]),
        "small": (TensorProto.UINT8, [1, 10, 3, 3]),
    }
    manual_names = set(manual)
    kept_vi = [x for x in model.graph.value_info if x.name not in manual_names]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_vi)
    for name, (dtype, shape) in manual.items():
        model.graph.value_info.append(vi(name, dtype, shape))

    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    onnx.save(build(), "task134.onnx")
    print("saved task134.onnx")
