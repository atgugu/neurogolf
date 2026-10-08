#!/usr/bin/env python3
from __future__ import annotations

import math

import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper as H
from onnx import numpy_helper


OUT = "task189.onnx"


def init(name: str, arr) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr), name=name)


def vi(name: str, dtype: int, shape: list[int]) -> onnx.ValueInfoProto:
    return H.make_tensor_value_info(name, dtype, shape)


def build() -> onnx.ModelProto:
    nodes = [
        H.make_node("GatherND", ["input", "h_idx"], ["h_f"], batch_dims=1),
        H.make_node("Cast", ["h_f"], ["h"], to=TP.BOOL),
        H.make_node("GatherND", ["input", "v_idx"], ["v_f"], batch_dims=1),
        H.make_node("Cast", ["v_f"], ["v"], to=TP.BOOL),
        H.make_node("Where", ["v", "i0", "i3"], ["pat_rs"]),
        H.make_node("Where", ["h", "i0", "i3"], ["pat_cs"]),
        H.make_node("Where", ["v", "i7", "i0"], ["pal_rs"]),
        H.make_node("Where", ["h", "i7", "i0"], ["pal_cs"]),
        H.make_node("Concat", ["i3", "pat_rs", "pat_cs"], ["pat_starts"], axis=0),
        H.make_node("Add", ["pat_starts", "pat_sizes"], ["pat_ends"]),
        H.make_node("Concat", ["i1", "pal_rs", "pal_cs"], ["pal_starts"], axis=0),
        H.make_node("Add", ["pal_starts", "pal_sizes"], ["pal_ends"]),
        H.make_node("Slice", ["input", "pat_starts", "pat_ends", "axes"], ["pattern_f"]),
        H.make_node("Cast", ["pattern_f"], ["pattern_b"], to=TP.BOOL),
        H.make_node("Slice", ["input", "pal_starts", "pal_ends", "axes"], ["palette_f"]),
        H.make_node("Conv", ["palette_f", "color_values", "color_bias"], ["palette_plus"]),
        H.make_node("Cast", ["palette_plus"], ["palette_plus_u8"], to=TP.UINT8),
        H.make_node(
            "Resize",
            ["palette_plus_u8", "", "resize_scales"],
            ["palette_big_u8"],
            mode="nearest",
            coordinate_transformation_mode="asymmetric",
            nearest_mode="floor",
        ),
        H.make_node("Where", ["pattern_b", "palette_big_u8", "one_u8"], ["code_u8"]),
        H.make_node("Cast", ["code_u8"], ["code_f16"], to=TP.FLOAT16),
        H.make_node("Pow", ["code_f16", "pow_exp_f16"], ["code_feat_f16"]),
        H.make_node(
            "ConvTranspose",
            ["code_feat_f16", "decode_w_f16", "decode_b_f16"],
            ["output"],
            pads=[0, 0, -24, -24],
        ),
    ]

    initializers = [
        init("axes", np.array([1, 2, 3], dtype=np.int32)),
        init("i0", np.array([0], dtype=np.int32)),
        init("i1", np.array([1], dtype=np.int32)),
        init("i3", np.array([3], dtype=np.int32)),
        init("i7", np.array([7], dtype=np.int32)),
        init("pat_sizes", np.array([1, 6, 6], dtype=np.int32)),
        init("pal_sizes", np.array([9, 2, 2], dtype=np.int32)),
        init("color_values", np.arange(1, 10, dtype=np.float32).reshape(1, 9, 1, 1)),
        init("color_bias", np.array([1], dtype=np.float32)),
        init("resize_scales", np.array([1, 1, 3, 3], dtype=np.float32)),
        init("one_u8", np.array([1], dtype=np.uint8)),
        init("pow_exp_f16", np.array([1, 2], dtype=np.float16).reshape(1, 2, 1, 1)),
        init(
            "decode_w_f16",
            np.stack(
                [
                    np.array([2 * (k + 1), -1], dtype=np.float16)
                    for k in range(10)
                ],
                axis=1,
            ).reshape(2, 10, 1, 1),
        ),
        init(
            "decode_b_f16",
            np.array([-(k + 1) * (k + 1) + 0.25 for k in range(10)], dtype=np.float16),
        ),
        init("h_idx", np.array([[8, 0, 6]], dtype=np.int64)),
        init("v_idx", np.array([[8, 6, 0]], dtype=np.int64)),
    ]

    value_info = [
        vi("h_f", TP.FLOAT, [1]),
        vi("h", TP.BOOL, [1]),
        vi("v_f", TP.FLOAT, [1]),
        vi("v", TP.BOOL, [1]),
        vi("pat_rs", TP.INT32, [1]),
        vi("pat_cs", TP.INT32, [1]),
        vi("pal_rs", TP.INT32, [1]),
        vi("pal_cs", TP.INT32, [1]),
        vi("pat_starts", TP.INT32, [3]),
        vi("pat_ends", TP.INT32, [3]),
        vi("pal_starts", TP.INT32, [3]),
        vi("pal_ends", TP.INT32, [3]),
        vi("pattern_f", TP.FLOAT, [1, 1, 6, 6]),
        vi("pattern_b", TP.BOOL, [1, 1, 6, 6]),
        vi("palette_f", TP.FLOAT, [1, 9, 2, 2]),
        vi("palette_plus", TP.FLOAT, [1, 1, 2, 2]),
        vi("palette_plus_u8", TP.UINT8, [1, 1, 2, 2]),
        vi("palette_big_u8", TP.UINT8, [1, 1, 6, 6]),
        vi("code_u8", TP.UINT8, [1, 1, 6, 6]),
        vi("code_f16", TP.FLOAT16, [1, 1, 6, 6]),
        vi("code_feat_f16", TP.FLOAT16, [1, 2, 6, 6]),
    ]

    graph = H.make_graph(
        nodes,
        "task189_equal_terminal",
        [vi("input", TP.FLOAT, [1, 10, 30, 30])],
        [vi("output", TP.FLOAT16, [1, 10, 30, 30])],
        initializer=initializers,
        value_info=value_info,
    )
    model = H.make_model(graph, opset_imports=[H.make_operatorsetid("", 18)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    return onnx.shape_inference.infer_shapes(model, strict_mode=True)


def main() -> None:
    model = build()
    onnx.save(model, OUT)
    sizes = {
        TP.FLOAT: 4,
        TP.FLOAT16: 2,
        TP.BOOL: 1,
        TP.UINT8: 1,
        TP.INT32: 4,
        TP.INT64: 8,
    }
    rows: list[tuple[int, str, int, list[int]]] = []
    memory = 0
    for value in model.graph.value_info:
        tt = value.type.tensor_type
        shape = [d.dim_value for d in tt.shape.dim]
        nbytes = int(np.prod(shape)) * sizes[tt.elem_type]
        memory += nbytes
        rows.append((nbytes, value.name, tt.elem_type, shape))
    params = sum(numpy_helper.to_array(t).size for t in model.graph.initializer)
    cost = params + memory
    print(f"saved {OUT}")
    print(f"BUDGET params={params} memory={memory} cost={cost} pts={25 - math.log(cost):.4f}")
    for nbytes, name, dtype, shape in sorted(rows, reverse=True):
        print(f"{nbytes:4d} B {name:16s} dtype={dtype:2d} shape={shape}")


if __name__ == "__main__":
    main()
