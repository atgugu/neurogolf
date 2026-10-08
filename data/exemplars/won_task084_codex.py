#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


OUT = Path("task084.onnx")


def init(name: str, arr: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr), name=name)


def linear_coeff(const: float, row: float, col: float, size: float) -> np.ndarray:
    """Coefficients for const + row*r + col*c + size*N over [1,x] features."""
    out = np.zeros((2, 2, 2), dtype=np.float32)
    out[0, 0, 0] = const
    out[1, 0, 0] = row
    out[0, 1, 0] = col
    out[0, 0, 1] = size
    return out


def factor_coefficients():
    one = linear_coeff(1.0, 0.0, 0.0, 0.0)
    red_minus = linear_coeff(0.5, 1.0, 1.0, -1.0)
    red_plus = linear_coeff(1.5, 1.0, 1.0, -1.0)
    yellow_minus = linear_coeff(0.5, 1.0, 0.0, -1.0)
    yellow_plus = linear_coeff(1.5, 1.0, 0.0, -1.0)

    coeff0 = np.stack([one, red_minus]).astype(np.float32)
    coeff1 = np.stack([one, red_plus]).astype(np.float32)
    coeff2 = np.stack([one, yellow_minus]).astype(np.float32)
    coeff3 = np.stack([one, yellow_plus]).astype(np.float32)

    # Terms: anchor, background, red, yellow.
    sel0 = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=np.float32)
    sel1 = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=np.float32)
    sel2 = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32)
    sel3 = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32)
    return (coeff0, coeff1, coeff2, coeff3), (sel0, sel1, sel2, sel3)


def build() -> onnx.ModelProto:
    coord = np.stack([np.ones(30, dtype=np.float32), np.arange(30, dtype=np.float32)], axis=1)

    inmap = np.zeros((4, 10), dtype=np.float32)
    inmap[0, 1:] = 1.0
    inmap[1:, 0] = 1.0

    outmap = np.zeros((4, 10, 2), dtype=np.float32)
    outmap[0, 1:, 1] = 1.0
    outmap[1, 0, 0] = 1.0
    outmap[2, 2, 0] = -1.0
    outmap[3, 4, 0] = -1.0

    coeffs, selectors = factor_coefficients()
    inits = [
        init("coord", coord),
        init("one_out", np.ones((1, 10, 1, 1), dtype=np.float32)),
        init("one_size", np.ones((1,), dtype=np.float32)),
        init("inmap", inmap),
        init("outmap", outmap),
    ]
    for idx, arr in enumerate(coeffs):
        inits.append(init(f"coeff{idx}", arr))
    for idx, arr in enumerate(selectors):
        inits.append(init(f"sel{idx}", arr))

    equation = (
        "nirw,ti,pouv,top,"
        "ra,wb,c,tq,qabc,"
        "rd,we,f,ts,sdef,"
        "rg,wh,j,tl,lghj,"
        "rm,wk,z,ty,ymkz->norw"
    )

    nodes = [
        helper.make_node("ReduceMax", ["input"], ["present"], axes=[0, 2, 3], keepdims=1),
        helper.make_node("Concat", ["one_out", "present"], ["outfeat"], axis=0),
        helper.make_node("ReduceSum", ["input"], ["area"], axes=[0, 1, 2, 3], keepdims=0),
        helper.make_node("Sqrt", ["area"], ["n"]),
        helper.make_node("Unsqueeze", ["n"], ["n1"], axes=[0]),
        helper.make_node("Concat", ["one_size", "n1"], ["size2"], axis=0),
        helper.make_node(
            "Einsum",
            [
                "input",
                "inmap",
                "outfeat",
                "outmap",
                "coord",
                "coord",
                "size2",
                "sel0",
                "coeff0",
                "coord",
                "coord",
                "size2",
                "sel1",
                "coeff1",
                "coord",
                "coord",
                "size2",
                "sel2",
                "coeff2",
                "coord",
                "coord",
                "size2",
                "sel3",
                "coeff3",
            ],
            ["output"],
            equation=equation,
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "task084_factored_linear_terminal_einsum",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def budget(model: onnx.ModelProto) -> str:
    dtype_bytes = {
        TensorProto.FLOAT: 4,
        TensorProto.FLOAT16: 2,
        TensorProto.INT64: 8,
        TensorProto.INT32: 4,
        TensorProto.UINT32: 4,
        TensorProto.UINT8: 1,
        TensorProto.INT8: 1,
        TensorProto.BOOL: 1,
    }
    inferred = shape_inference.infer_shapes(model)
    output_names = {o.name for o in model.graph.output}
    value_info = {v.name: v for v in list(inferred.graph.value_info) + list(inferred.graph.output)}
    params = sum(int(np.prod(t.dims)) for t in model.graph.initializer)
    mem = 0
    rows = []
    for node in model.graph.node:
        for name in node.output:
            if name in output_names:
                continue
            tt = value_info[name].type.tensor_type
            shape = [d.dim_value for d in tt.shape.dim]
            size = int(np.prod(shape)) * dtype_bytes[tt.elem_type]
            mem += size
            rows.append((size, name, tt.elem_type, shape))
    cost = params + mem
    lines = [f"BUDGET params={params} memory={mem} cost={cost} pts={25.0 - math.log(cost):.4f}"]
    for size, name, elem, shape in sorted(rows, reverse=True):
        lines.append(f"  {size:4d} B  elem={elem:<2} {shape}  {name}")
    return "\n".join(lines)


def main() -> int:
    model = build()
    onnx.save(model, OUT)
    print(budget(model))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
