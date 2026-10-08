from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

sys.path.insert(0, "../../runner")
from ngolf import RELOWER_DOC, relower_onehot_plane


OUT = Path("task369.onnx")


def init(name: str, arr) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr), name=name)


def build() -> onnx.ModelProto:
    inits: list[onnx.TensorProto] = []

    def c(name: str, arr) -> str:
        inits.append(init(name, arr))
        return name

    relower_nodes, relower_inits, _ = relower_onehot_plane(
        "input",
        "black",
        channel=0,
        crop=((0, 1), (0, 1), (0, 10), (0, 10)),
        dtype="u8",
        starts_name="slice_starts",
        ends_name="slice_ends",
    )
    for name, arr in relower_inits:
        c(name, arr)

    c("q_scale", np.array([1.0], dtype=np.float32))
    c("u8_zero", np.array([0], dtype=np.uint8))
    c("one_u8", np.array([1], dtype=np.uint8))
    c("i8_zero", np.array([0], dtype=np.int8))

    # Two-hop morphology state:
    #   h0 = black + 1, a real-canvas sentinel used by the terminal x_zero_point=1.
    #   h1 = hub = black cell with four-neighbor degree 2.
    # The terminal's 3x3 stencil reads neighboring hubs, which is the near_hub fact.
    w1 = np.zeros((2, 1, 3, 3), dtype=np.int8)
    b1 = np.array([1, -8], dtype=np.int32)
    w1[0, 0, 1, 1] = 1
    w1[1, 0, 1, 1] = 5
    for rr, cc in ((0, 1), (1, 0), (1, 2), (2, 1)):
        w1[1, 0, rr, cc] = 2
    c("w1", w1)
    c("b1", b1)

    # Direct terminal logits. With x_zero_point=1, h0 contributes black/not-black
    # and h1 contributes hub-vs-real-nonhub while implicit padding contributes zero.
    w2 = np.zeros((10, 2, 3, 3), dtype=np.int8)
    b2 = np.zeros((10,), dtype=np.int32)

    b2[1] = -25
    w2[1] = np.array(
        [
            [[4, 8, 4], [8, 36, 8], [4, 8, 4]],
            [[-4, 10, -4], [10, 4, 10], [-4, 10, -4]],
        ],
        dtype=np.int8,
    )
    b2[2] = -49
    w2[2] = np.array(
        [
            [[-2, 6, -2], [6, 24, 6], [-2, 6, -2]],
            [[2, -5, 2], [-5, -12, -5], [2, -5, 2]],
        ],
        dtype=np.int8,
    )
    b2[3] = -1
    w2[3] = np.array(
        [
            [[0, -1, 0], [-1, 2, -1], [0, -1, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ],
        dtype=np.int8,
    )
    b2[5] = -1
    w2[5] = np.array(
        [
            [[0, 0, 0], [0, -2, 0], [0, 0, 0]],
            [[0, 0, 0], [0, -2, 0], [0, 0, 0]],
        ],
        dtype=np.int8,
    )
    c("w2", w2)
    c("b2", b2)

    nodes = [
        *relower_nodes,
        helper.make_node(
            "QLinearConv",
            [
                "black",
                "q_scale",
                "u8_zero",
                "w1",
                "q_scale",
                "i8_zero",
                "q_scale",
                "u8_zero",
                "b1",
            ],
            ["h"],
            pads=[1, 1, 1, 1],
        ),
        helper.make_node(
            "QLinearConv",
            [
                "h",
                "q_scale",
                "one_u8",
                "w2",
                "q_scale",
                "i8_zero",
                "q_scale",
                "u8_zero",
                "b2",
            ],
            ["output"],
            pads=[1, 1, 21, 21],
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "task369_relowered_two_hop_terminal",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.UINT8, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    shape_inference.infer_shapes(model, strict_mode=True)
    return model


def budget(model: onnx.ModelProto) -> str:
    dtype_bytes = {
        TensorProto.FLOAT: 4,
        TensorProto.FLOAT16: 2,
        TensorProto.INT64: 8,
        TensorProto.INT32: 4,
        TensorProto.UINT8: 1,
        TensorProto.INT8: 1,
        TensorProto.BOOL: 1,
    }
    inferred = shape_inference.infer_shapes(model, strict_mode=True)
    output_names = {o.name for o in inferred.graph.output}
    relower_tmps = {
        node.output[0]
        for node in inferred.graph.node
        if node.op_type == "Slice"
        and node.output
        and node.doc_string.startswith(RELOWER_DOC)
    }
    vi = {v.name: v for v in list(inferred.graph.value_info) + list(inferred.graph.output)}
    params = sum(int(np.prod(t.dims)) for t in inferred.graph.initializer)
    rows = []
    mem = 0
    for node in inferred.graph.node:
        for name in node.output:
            if not name or name in output_names or name in relower_tmps:
                continue
            tt = vi[name].type.tensor_type
            shape = [d.dim_value for d in tt.shape.dim]
            size = int(np.prod(shape)) * dtype_bytes[tt.elem_type]
            rows.append((size, name, shape, tt.elem_type))
            mem += size
    cost = params + mem
    lines = [f"BUDGET params={params} memory={mem} cost={cost} pts={25 - math.log(cost):.4f}"]
    for size, name, shape, elem in sorted(rows, reverse=True):
        lines.append(f"  {size:4d} B elem={elem:<2} {shape} {name}")
    return "\n".join(lines)


def main() -> int:
    model = build()
    onnx.save(model, OUT)
    print(budget(model))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
