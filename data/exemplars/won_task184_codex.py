#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


W = 30
Z = 3


def cst(name, value, dtype):
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)


def node(op, inputs, outputs, **attrs):
    return helper.make_node(op, inputs, outputs, **attrs)


def segment_nodes(prefix: str, count: str, out: str):
    return [
        helper.make_node("Equal", [count, "zero_f"], [f"{prefix}_sep"]),
        helper.make_node("Cast", [f"{prefix}_sep"], [f"{prefix}_sep_h"], to=TensorProto.FLOAT16),
        helper.make_node("CumSum", [f"{prefix}_sep_h", "axis1"], [f"{prefix}_id_h"]),
        helper.make_node("Equal", ["zone_h", f"{prefix}_id_h"], [f"{prefix}_mask_b"]),
        helper.make_node("Cast", [f"{prefix}_mask_b"], [out], to=TensorProto.FLOAT),
    ]


def segment_poly_nodes(prefix: str, count: str, out: str):
    return [
        helper.make_node("Less", [count, "one30"], [f"{prefix}_sep"]),
        helper.make_node("Cast", [f"{prefix}_sep"], [f"{prefix}_sep_f"], to=TensorProto.FLOAT),
        helper.make_node("CumSum", [f"{prefix}_sep_f", "axis1"], [f"{prefix}_id_f"]),
        helper.make_node("Concat", ["one30", f"{prefix}_id_f"], [out], axis=0),
    ]


def build():
    place = np.zeros((Z, W), dtype=np.float32)
    place[np.arange(Z), np.arange(Z)] = 1.0
    inits = [
        cst("fg10", [0, 1, 1, 1, 1, 1, 1, 1, 1, 1], np.float32),
        cst("zero_f", 0.0, np.float32),
        cst("axis1", 1, np.int64),
        cst("zone_h", np.arange(Z).reshape(Z, 1), np.float16),
        cst("place", place, np.float32),
    ]
    nodes = [
        helper.make_node("Einsum", ["input", "fg10"], ["row_cnt"], equation="bchw,c->bh"),
        helper.make_node("Einsum", ["input", "fg10"], ["col_cnt"], equation="bchw,c->bw"),
        *segment_nodes("r", "row_cnt", "R"),
        *segment_nodes("c", "col_cnt", "C"),
        helper.make_node(
            "Einsum",
            ["R", "input", "C", "fg10", "place", "place"],
            ["output"],
            equation="rh,bchw,qw,c,rp,qs->bcps",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "task184_sep_f16_equal",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, W, W])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, W, W])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def task184_sep_exact_relower():
    place = np.zeros((Z, W), dtype=np.float32)
    place[np.arange(Z), np.arange(Z)] = 1.0
    coeff = np.zeros((Z, 2, 2), dtype=np.float32)
    coeff[0, 0, 0] = 1.0
    coeff[0, 0, 1] = -1.5
    coeff[0, 1, 1] = 0.5
    coeff[1, 0, 1] = 2.0
    coeff[1, 1, 1] = -1.0
    coeff[2, 0, 1] = -0.5
    coeff[2, 1, 1] = 0.5
    inits = [
        cst("fg10", [0, 1, 1, 1, 1, 1, 1, 1, 1, 1], np.float32),
        cst("one30", np.ones((1, W), dtype=np.float32), np.float32),
        cst("axis1", 1, np.int64),
        cst("coeff", coeff, np.float32),
        cst("place", place, np.float32),
    ]
    nodes = [
        helper.make_node("Einsum", ["input", "fg10"], ["row_cnt"], equation="bchw,c->bh"),
        helper.make_node("Einsum", ["input", "fg10"], ["col_cnt"], equation="bchw,c->bw"),
        *segment_poly_nodes("r", "row_cnt", "RU"),
        *segment_poly_nodes("c", "col_cnt", "CU"),
        helper.make_node(
            "Einsum",
            ["coeff", "RU", "RU", "input", "coeff", "CU", "CU", "fg10", "place", "place"],
            ["output"],
            equation="rad,ah,dh,bchw,qel,ew,lw,c,rp,qs->bcps",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "task184_sep_exact_relower",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, W, W])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, W, W])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def sample_axis_nodes(prefix: str, count: str, idx_out: str):
    return [
        node("Equal", [count, "zero_f"], [f"{prefix}_sep"]),
        node("Cast", [f"{prefix}_sep"], [f"{prefix}_sep_h"], to=TensorProto.FLOAT16),
        node("TopK", [f"{prefix}_sep_h", "k2"], [f"{prefix}_sep_v", f"{prefix}_sep_i"], axis=1, largest=1, sorted=1),
        node("Gather", [f"{prefix}_sep_i", "idx0"], [f"{prefix}_s0"], axis=1),
        node("Gather", [f"{prefix}_sep_i", "idx1"], [f"{prefix}_s1"], axis=1),
        node("Gather", [f"{prefix}_sep_v", "idx1"], [f"{prefix}_v1"], axis=1),
        node("Greater", [f"{prefix}_v1", "zero_h"], [f"{prefix}_has2"]),
        node("Add", [f"{prefix}_s0", "one_i64"], [f"{prefix}_s0p1"]),
        node("Add", [f"{prefix}_s1", "one_i64"], [f"{prefix}_s1p1_raw"]),
        node("Min", [f"{prefix}_s1p1_raw", "max29_i64"], [f"{prefix}_s1p1"]),
        node("Where", [f"{prefix}_has2", f"{prefix}_s1p1", f"{prefix}_s0"], [f"{prefix}_third"]),
        node("Concat", ["one_1x1", f"{prefix}_s0p1", f"{prefix}_third"], [idx_out], axis=1),
    ]


def task184_slice_sample_pad():
    inits = [
        cst("fg10", [0, 1, 1, 1, 1, 1, 1, 1, 1, 1], np.float32),
        cst("zero_f", 0.0, np.float32),
        cst("zero_h", 0.0, np.float16),
        cst("k2", [2], np.int64),
        cst("idx0", [0], np.int64),
        cst("idx1", [1], np.int64),
        cst("one_i64", np.array([[1]], dtype=np.int64), np.int64),
        cst("one_1x1", np.array([[1]], dtype=np.int64), np.int64),
        cst("max29_i64", np.array([[29]], dtype=np.int64), np.int64),
        cst("row_shape", [1, 1, 3, 1, 1], np.int64),
        cst("col_shape", [1, 1, 1, 3, 1], np.int64),
        cst("pair_shape", [1, 10, 3, 3, 1], np.int64),
        cst("slice_starts", [1], np.int64),
        cst("slice_ends", [10], np.int64),
        cst("slice_axes", [1], np.int64),
        cst("pads_out", [0, 1, 0, 0, 0, 0, 27, 27], np.int64),
    ]
    nodes = [
        node("Einsum", ["input", "fg10"], ["row_cnt"], equation="bchw,c->bh"),
        node("Einsum", ["input", "fg10"], ["col_cnt"], equation="bchw,c->bw"),
        *sample_axis_nodes("r", "row_cnt", "row_idx"),
        *sample_axis_nodes("c", "col_cnt", "col_idx"),
        node("Reshape", ["row_idx", "row_shape"], ["row_idx5"]),
        node("Reshape", ["col_idx", "col_shape"], ["col_idx5"]),
        node("Expand", ["row_idx5", "pair_shape"], ["row_pair"]),
        node("Expand", ["col_idx5", "pair_shape"], ["col_pair"]),
        node("Concat", ["row_pair", "col_pair"], ["pair_idx"], axis=4),
        node("GatherND", ["input", "pair_idx"], ["sample10"], batch_dims=2),
        node("Slice", ["sample10", "slice_starts", "slice_ends", "slice_axes"], ["sample9"]),
        node("Pad", ["sample9", "pads_out", "zero_f"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes,
        "task184_slice_sample_pad",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, W, W])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, W, W])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


MODELS = {
    "task184_sep_f16_equal": build,
    "task184_sep_exact_relower": task184_sep_exact_relower,
    "task184_slice_sample_pad": task184_slice_sample_pad,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="task184_sep_f16_equal")
    parser.add_argument("--out", default="task184.onnx")
    args = parser.parse_args()
    onnx.save(MODELS[args.model](), Path(args.out))
