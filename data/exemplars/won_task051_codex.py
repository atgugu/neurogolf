#!/usr/bin/env python3
"""task051 att12: att7 extraction with a tagged Hodel-ray terminal.

True rule: find the singleton marker color in the triangle and shoot that color
from the marker along the emitter axis through background cells only.

Priced before build:

Design A, infra-fused Hodel terminal (this file)
  x background plane              [1,1,30,30] uint8        900
  row_eq,row_side                 [1,1,30,1] bool x2        60
  col_eq,col_side                 [1,1,1,30] bool x2        60
  marker_oh,bg_oh                 [1,10,1,1] bool x2        20
  small extraction/flags          scalar/vector outputs    178
  params                          initializers/constants   122
  paper total                                             1340

Design B, incumbent dense terminal Einsum
  measured by fast_verify: 1477 = 1094 memory + 383 params; value-exact but
  below the +0.15 registration floor.

Rejected lower-bound family: one-mask standard Where measured 1377 but paints
through non-background cells, so correctness repairs only increase cost.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper as H
from onnx import numpy_helper


RUNNER = Path("../../runner")
sys.path.insert(0, str(RUNNER))
from ngolf import relower_onehot_plane, terminal_hodel_ray  # noqa: E402


def init(name: str, arr: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr), name=name)


def main() -> None:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    inits.extend(
        [
            init("one_f", np.array(1.0, np.float32)),
            init("nonzero_color_f", np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1], np.float32)),
            init("ids_f", np.arange(30, dtype=np.float32)),
            init("row_ids_i", np.arange(30, dtype=np.int8).reshape(1, 1, 30, 1)),
            init("col_ids_i", np.arange(30, dtype=np.int8).reshape(1, 1, 1, 30)),
            init("ax23", np.array([2, 3], dtype=np.int64)),
            init("zero_i4", np.zeros((1, 1, 1, 1), dtype=np.int8)),
        ]
    )

    x_nodes, x_inits, _ = relower_onehot_plane(
        "input",
        "x_u8",
        channel=0,
        crop=((0, 1), (0, 1), (0, 30), (0, 30)),
        dtype="u8",
        starts_name="x_starts",
        ends_name="x_ends",
    )
    nodes.extend(x_nodes)
    for name, arr in x_inits:
        inits.append(init(name, np.asarray(arr, dtype=np.int64)))

    bg = np.zeros((1, 10, 1, 1), dtype=np.bool_)
    bg[0, 0, 0, 0] = True
    nodes.append(H.make_node("Constant", [], ["bg_oh_b"], value=init("bg_oh_value", bg)))

    def add(op: str, inputs: list[str], outputs: list[str], **attrs) -> None:
        nodes.append(H.make_node(op, inputs, outputs, **attrs))

    add("ReduceSum", ["input", "ax23"], ["color_counts"], keepdims=1)
    add("Equal", ["color_counts", "one_f"], ["special_color_b"])
    add("Cast", ["special_color_b"], ["special_vec_f"], to=TP.FLOAT)

    add("Einsum", ["input", "special_vec_f", "ids_f"], ["special_row_f"], equation="bchw,bcxy,h->bxy")
    add("Einsum", ["input", "special_vec_f", "ids_f"], ["special_col_f"], equation="bchw,bcxy,w->bxy")
    add("Cast", ["special_row_f"], ["special_row_i"], to=TP.INT8)
    add("Cast", ["special_col_f"], ["special_col_i"], to=TP.INT8)

    add("Einsum", ["input", "nonzero_color_f"], ["total_colored_f"], equation="bchw,c->b")
    add("Einsum", ["input", "nonzero_color_f", "ids_f"], ["row_moment_f"], equation="bchw,c,h->b")
    add("Einsum", ["input", "nonzero_color_f", "ids_f"], ["col_moment_f"], equation="bchw,c,w->b")

    add("Mul", ["special_row_f", "total_colored_f"], ["center_row_moment_f"])
    add("Mul", ["special_col_f", "total_colored_f"], ["center_col_moment_f"])
    add("Sub", ["row_moment_f", "center_row_moment_f"], ["row_moment_delta_f"])
    add("Sub", ["col_moment_f", "center_col_moment_f"], ["col_moment_delta_f"])
    add("Cast", ["row_moment_delta_f"], ["row_moment_delta_i"], to=TP.INT8)
    add("Cast", ["col_moment_delta_f"], ["col_moment_delta_i"], to=TP.INT8)
    add("Sign", ["row_moment_delta_i"], ["vert_sign_i"])
    add("Sign", ["col_moment_delta_i"], ["horiz_sign_i"])

    add("Mul", ["row_ids_i", "vert_sign_i"], ["row_scaled_i"])
    add("Mul", ["special_row_i", "vert_sign_i"], ["row_center_scaled_i"])
    add("Mul", ["col_ids_i", "horiz_sign_i"], ["col_scaled_i"])
    add("Mul", ["special_col_i", "horiz_sign_i"], ["col_center_scaled_i"])

    add("Greater", ["row_scaled_i", "row_center_scaled_i"], ["row_forward"])
    add("Greater", ["col_scaled_i", "col_center_scaled_i"], ["col_forward"])
    add("Equal", ["row_ids_i", "special_row_i"], ["same_row"])
    add("Equal", ["col_ids_i", "special_col_i"], ["same_col"])

    add("Equal", ["horiz_sign_i", "zero_i4"], ["hzero_b"])
    add("Not", ["hzero_b"], ["hflag_b"])
    add("Equal", ["vert_sign_i", "zero_i4"], ["vzero_b"])
    add("Not", ["vzero_b"], ["vflag_b"])

    term_nodes, _, _ = terminal_hodel_ray(
        "input",
        "x_u8",
        "same_row",
        "row_forward",
        "same_col",
        "col_forward",
        "special_color_b",
        "bg_oh_b",
        "hflag_b",
        "vflag_b",
        "output",
    )
    nodes.extend(term_nodes)

    graph = H.make_graph(
        nodes,
        "task051_att12_terminal_hodel_ray",
        [H.make_tensor_value_info("input", TP.FLOAT, [1, 10, 30, 30])],
        [H.make_tensor_value_info("output", TP.FLOAT, [1, 10, 30, 30])],
        initializer=inits,
    )
    model = H.make_model(graph, opset_imports=[H.make_opsetid("", 14)])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, "task051.onnx")
    print(f"saved task051.onnx nodes={len(nodes)} inits={len(inits)}")


if __name__ == "__main__":
    main()
