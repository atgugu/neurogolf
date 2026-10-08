#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runner"))
from ngolf import relower_onehot_plane


OUT = Path(__file__).resolve().parent / "task160.onnx"


def init(name: str, arr, dtype=None):
    a = np.asarray(arr, dtype=dtype)
    return numpy_helper.from_array(a, name=name)


def qconv_node(x: str, w: str, y: str, b: str, *, x_zp: str = "zero_u8",
               w_zp: str = "zero_i8", pads):
    return helper.make_node(
        "QLinearConv",
        [x, "one_f", x_zp, w, "one_f", w_zp, "one_f", "zero_u8", b],
        [y],
        pads=list(pads),
    )


def main() -> None:
    nodes, relower_inits, _ = relower_onehot_plane(
        input_name="input",
        output_name="blue_u8",
        channel=1,
        crop=((0, 1), (1, 2), (0, 10), (0, 10)),
        dtype="u8",
        starts_name="blue_starts",
        ends_name="blue_ends",
    )

    inits = [init(name, arr) for name, arr in relower_inits]
    inits += [
        init("one_f", np.array(1.0, np.float32)),
        init("zero_u8", np.array(0, np.uint8)),
        init("zero_i8", np.array(0, np.int8)),
    ]

    feature_w = np.zeros((2, 1, 3, 3), dtype=np.uint8)
    feature_w[0, 0, 0, 0] = 1
    feature_w[1, 0, :, :] = 1
    feature_b = np.array([0, -7], dtype=np.int32)
    inits += [init("feature_w", feature_w), init("feature_b", feature_b)]

    code_delta = np.array(
        [[-1, 1, -1], [1, 2, 1], [-1, 1, -1]],
        dtype=np.int8,
    )
    code2_delta = np.array(
        [[-3, 5, -3], [5, 8, 5], [-3, 5, -3]],
        dtype=np.int8,
    )
    state_w = np.zeros((2, 2, 3, 3), dtype=np.int8)
    state_w[0, 0, 2, 2] = 1
    state_w[0, 1, :, :] = code_delta
    state_w[1, 0, 2, 2] = 3
    state_w[1, 1, :, :] = code2_delta
    # Shifted class code: 1=black, 2=blue, 3=red. This keeps QLinearConv's
    # bottom/right zero padding distinct from in-crop black.
    state_b = np.array([1, 1], dtype=np.int32)
    inits += [init("state_w", state_w), init("state_b", state_b)]

    final_w = np.zeros((10, 2, 1, 1), dtype=np.int8)
    final_b = np.zeros((10,), dtype=np.int32)
    # Separators over (code, code^2): pad=(0,0), black=(1,1), blue=(2,4), red=(3,9).
    final_w[0, :, 0, 0] = [2, -1]
    final_w[1, :, 0, 0] = [4, -1]
    final_b[1] = -3
    final_w[2, :, 0, 0] = [-2, 1]
    final_b[2] = -2
    inits += [init("final_w", final_w), init("final_b", final_b)]

    nodes += [
        qconv_node(
            "blue_u8",
            "feature_w",
            "features_u8",
            "feature_b",
            w_zp="zero_u8",
            pads=[0, 0, 2, 2],
        ),
        qconv_node("features_u8", "state_w", "state_u8", "state_b", pads=[2, 2, 0, 0]),
        qconv_node(
            "state_u8",
            "final_w",
            "output",
            "final_b",
            pads=[0, 0, 20, 20],
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "task160_scalar_quadratic_u8",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.UINT8, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, OUT)

    params = sum(np.prod(t.dims, dtype=np.int64) for t in model.graph.initializer)
    memory = 100 + 200 + 200
    cost = int(params + memory)
    print(f"params={params} memory={memory} cost={cost} pts={25 - math.log(cost):.4f}")


if __name__ == "__main__":
    main()
