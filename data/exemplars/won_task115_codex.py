#!/usr/bin/env python3
import math

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def init(name, arr, proto=None):
    arr = np.asarray(arr)
    if proto is None:
        return numpy_helper.from_array(arr, name)
    return helper.make_tensor(name, proto, arr.shape, arr.ravel().tolist())


def node(op, inputs, outputs, **attrs):
    return helper.make_node(op, inputs, outputs, **attrs)


def main():
    nodes = [
        node("Einsum", ["input"], ["counts_f"], equation="nchw->nc"),
        node("Einsum", ["input", "coords"], ["sum_x_f"], equation="nchw,w->nc"),
        node("Einsum", ["input", "coords"], ["sum_y_f"], equation="nchw,h->nc"),
        node("Cast", ["counts_f"], ["counts"], to=TensorProto.FLOAT16),
        node("Cast", ["sum_x_f"], ["sum_x"], to=TensorProto.FLOAT16),
        node("Cast", ["sum_y_f"], ["sum_y"], to=TensorProto.FLOAT16),
        node("Max", ["counts", "one_h"], ["safe_counts"]),
        node("Div", ["sum_x", "safe_counts"], ["mean_x"]),
        node("Div", ["sum_y", "safe_counts"], ["mean_y"]),
        node("Greater", ["counts", "one_h"], ["present"]),
        node("Where", ["present", "mean_x", "one_h"], ["mean_x_masked"]),
        node("Where", ["present", "mean_y", "one_h"], ["mean_y_masked"]),
        node("ReduceMin", ["mean_x_masked"], ["min_x"], axes=[1], keepdims=1),
        node("ReduceMax", ["mean_x"], ["max_x"], axes=[1], keepdims=1),
        node("ReduceMin", ["mean_y_masked"], ["min_y"], axes=[1], keepdims=1),
        node("ReduceMax", ["mean_y"], ["max_y"], axes=[1], keepdims=1),
        node("Sub", ["max_x", "min_x"], ["range_x"]),
        node("Sub", ["max_y", "min_y"], ["range_y"]),
        node("Greater", ["range_x", "range_y"], ["row_mode"]),
        node("Where", ["row_mode", "mean_x_masked", "mean_y_masked"], ["active_mean"]),
        node("TopK", ["active_mean", "k4"], ["top_values", "top_indices"], largest=0, sorted=1),
        node("Cast", ["top_indices"], ["top_u8"], to=TensorProto.UINT8),
        node("Concat", ["top_u8", "zero_color"], ["seq5"], axis=1),
        node("Where", ["row_mode", "row_gather_idx", "col_gather_idx"], ["gather_idx"]),
        node("Gather", ["seq5", "gather_idx"], ["color_block"], axis=1),
        node("Equal", ["color_block", "chan_ids_1_9"], ["block"]),
        node("Pad", ["block", "pads_out"], ["output"], mode="constant"),
    ]

    initializers = [
        init("coords", (np.arange(30, dtype=np.float32) / 30.0)),
        init("one_h", np.array(1, dtype=np.float16)),
        init("k4", np.array([4], dtype=np.int64)),
        init("chan_ids_1_9", np.arange(1, 10, dtype=np.uint8).reshape(1, 9, 1, 1)),
        init("zero_color", np.zeros((1, 1), dtype=np.uint8)),
        init(
            "row_gather_idx",
            np.array(
                [
                    [0, 1, 2, 3],
                    [4, 4, 4, 4],
                    [4, 4, 4, 4],
                    [4, 4, 4, 4],
                ],
                dtype=np.int32,
            ),
        ),
        init(
            "col_gather_idx",
            np.array(
                [
                    [0, 4, 4, 4],
                    [1, 4, 4, 4],
                    [2, 4, 4, 4],
                    [3, 4, 4, 4],
                ],
                dtype=np.int32,
            ),
        ),
        init("pads_out", np.array([0, 1, 0, 0, 0, 0, 26, 26], dtype=np.int64)),
    ]

    graph = helper.make_graph(
        nodes,
        "task115_tail_gather",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(inferred, "task115.onnx")

    elem = {
        TensorProto.FLOAT: 4,
        TensorProto.FLOAT16: 2,
        TensorProto.UINT8: 1,
        TensorProto.BOOL: 1,
        TensorProto.INT32: 4,
        TensorProto.INT64: 8,
    }
    value_info = list(inferred.graph.value_info)
    mem = 0
    rows = []
    for vi in value_info:
        tt = vi.type.tensor_type
        shape = [d.dim_value for d in tt.shape.dim]
        nbytes = int(np.prod(shape)) * elem[tt.elem_type]
        mem += nbytes
        rows.append((nbytes, vi.name, tt.elem_type, shape))
    params = sum(numpy_helper.to_array(t).size for t in inferred.graph.initializer)
    cost = params + mem
    pts = 25.0 - math.log(cost)
    print(f"BUDGET: params={params} memory={mem} cost={cost} pts={pts:.4f}")
    for nbytes, name, dtype, shape in sorted(rows, reverse=True)[:16]:
        print(f"{nbytes:5d} B {name:16s} dtype={dtype:2d} shape={shape}")


if __name__ == "__main__":
    main()
