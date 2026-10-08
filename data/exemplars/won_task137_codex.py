#!/usr/bin/env python3
"""Build task137.onnx by a value-exact u8 relowering of the proven graph.

True rule: the median of the three marker coordinates is the center, their
Chebyshev offset is the spacing, and the output paints every clipped square
perimeter at a positive multiple of that spacing.

Two priced families/designs (all node outputs are charged):

1. Proven f32-template graph (behavioral oracle)
   scalar statistics: 9 * [1] f32 = 36 B
   scalar u8 state: 8 * [1] u8 = 8 B
   row/column u8/bool arithmetic: 24 * 30 = 720 B
   pre-concat casts: 2 * 30 * 4 = 240 B
   f32 concatenated templates: 2 * 60 * 4 = 480 B
   initializers: 186 elements
   total: 1453 B memory + 186 params = 1639 B

2. PLANE-KILL + u8 rank/template relower (built)
   statistic/scalar and row/column arithmetic: 583 B
   u8 concatenated templates: 2 * 60 = 120 B
   one f32 cast per complete template: 2 * 60 * 4 = 480 B
   initializers: 186 elements
   total: 1183 B memory + 186 params = 1369 B

The rewrite swaps Cast->Concat for Concat(u8)->Cast on both separable
templates. ORT requires every Einsum operand to share a dtype, so those final
casts cannot legally be removed. Two affine rank branches reuse already
computed scalar offsets. The reference then maps integer ranks r,c to 2r and
2c+1 merely to break ties toward background. Over the certified rank range
[1,30], 0.99*r-c has the same strict sign and makes ties negative, deleting
three more vectors while preserving the thresholded terminal result.
"""

from __future__ import annotations

import onnx
import numpy as np
from onnx import TensorProto, helper, numpy_helper


SOURCE = "task137_plane_kill_source.onnx"
OUTPUT = "task137.onnx"


def main() -> None:
    model = onnx.load(SOURCE)
    graph = model.graph

    # The all-one row/column template initializers remain exact in uint8.
    for i, initializer in enumerate(graph.initializer):
        if initializer.name in {"t11", "t12"}:
            array = numpy_helper.to_array(initializer).astype("uint8")
            graph.initializer[i].CopyFrom(numpy_helper.from_array(array, initializer.name))

    # Preserve all statistics and integral arithmetic.  Only relower the two
    # template assembly pairs, keeping the names consumed by the final Einsum.
    nodes = list(graph.node)

    # Reuse qmax+2 and qmax+1 directly, deleting one row and one column vector.
    # Locate by output names after the distance rewrite to avoid brittle indices.
    by_output = {node.output[0]: i for i, node in enumerate(nodes)}
    nodes[by_output["t42"]] = helper.make_node("Add", ["t41", "t34"], ["t43"])
    del nodes[by_output["t43"]]
    by_output = {node.output[0]: i for i, node in enumerate(nodes)}
    nodes[by_output["t46"]] = helper.make_node("Sub", ["t40", "t35"], ["t47"])
    del nodes[by_output["t47"]]

    # The last three u8 vectors only encoded 2*r versus 2*c+1. A coefficient
    # ratio just below one gives exactly the same ordering and background ties.
    nodes = [node for node in nodes if not any(o in {"t50", "t51", "t52"} for o in node.output)]
    for i, initializer in enumerate(graph.initializer):
        if initializer.name == "t13":
            table = numpy_helper.to_array(initializer).astype("float32")
            table[:, 0] *= np.float32(0.99)
            graph.initializer[i].CopyFrom(numpy_helper.from_array(table, initializer.name))

    by_output = {node.output[0]: i for i, node in enumerate(nodes)}
    row_concat = helper.make_node("Concat", ["t45", "t11"], ["t53"], axis=1)
    row_cast = helper.make_node("Cast", ["t53"], ["t54"], to=TensorProto.FLOAT)
    col_concat = helper.make_node("Concat", ["t12", "t49"], ["t55"], axis=1)
    col_cast = helper.make_node("Cast", ["t55"], ["t56"], to=TensorProto.FLOAT)
    tail_start = by_output["t53"]
    nodes[tail_start:tail_start + 4] = [row_concat, row_cast, col_concat, col_cast]
    del graph.node[:]
    graph.node.extend(nodes)

    onnx.checker.check_model(model)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, OUTPUT)


if __name__ == "__main__":
    main()
