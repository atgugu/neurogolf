#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8
I8 = TensorProto.INT8
I32 = TensorProto.INT32


def init(name: str, value, dtype=None):
    arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype)
    return numpy_helper.from_array(arr, name=name)


def selector(offset: int) -> np.ndarray:
    s = np.zeros((14, 30), dtype=np.float32)
    for i in range(14):
        s[i, i + offset] = 1.0
    return s


def classifier_weights() -> np.ndarray:
    w = np.zeros((10, 3, 1, 1), dtype=np.int8)
    for c in range(10):
        # features are [1, code, code^2].  Score = 1 - (code - c)^2.
        # The grader thresholds output > 0, so only the matching channel survives.
        w[c, 0, 0, 0] = 1 - c * c
        w[c, 1, 0, 0] = 2 * c
        w[c, 2, 0, 0] = -1
    return w


def build() -> onnx.ModelProto:
    inits = [
        init("A", [0, 1, 1, 1, 1, 0, 1, 1, 1, 1], np.float32),
        init("Cg", [0, 0, 0, 0, 0, 1, 0, 0, 0, 0], np.float32),
        init("Bc", np.arange(10, dtype=np.float32)),
        init("one1", [1], np.float32),
        init("S0", selector(0)),
        init("S1", selector(1)),
        init("S2", selector(2)),
        init("gray5", [5], np.float32),
        init(
            "tpl",
            np.array(
                [
                    [
                        [
                            [0, 0, 4, 1, 1, 1, 4, 2, 2],
                            [0, 4, 4, 4, 1, 4, 4, 4, 2],
                            [4, 4, 4, 4, 4, 4, 4, 4, 4],
                            [3, 4, 4, 4, 4, 4, 4, 4, 5],
                            [3, 3, 4, 4, 4, 4, 4, 5, 5],
                            [3, 4, 4, 4, 4, 4, 4, 4, 5],
                            [4, 4, 4, 4, 4, 4, 4, 4, 4],
                            [6, 4, 4, 4, 7, 4, 4, 4, 8],
                            [6, 6, 4, 7, 7, 7, 4, 8, 8],
                        ]
                    ]
                ],
                dtype=np.int64,
            ),
        ),
        init("ones99", np.ones((1, 1, 9, 9), dtype=np.uint8)),
        init("eqw", classifier_weights(), np.int8),
    ]

    n = helper.make_node
    ein4 = "Zajs,a,Hj,Ws,Zbkt,b,Hk,Wt,Zclu,c,Hl,Wu,Zdmv,d,Hm,Wv,Q->Q"
    ein5 = "Zajs,a,Hj,Ws,Zbkt,b,Hk,Wt,Zclu,c,Hl,Wu,Zdmv,d,Hm,Wv,Zenw,e,Hn,Ww,Q->Q"
    nodes = [
        n("Einsum", ["input", "Bc", "S0", "S0", "input", "A", "S0", "S1", "input", "A", "S1", "S0", "input", "Cg", "S2", "S2", "one1"], ["e0"], equation=ein4),
        n("Einsum", ["input", "Bc", "S0", "S0", "input", "A", "S0", "S1", "input", "A", "S0", "S2", "input", "A", "S1", "S1", "one1"], ["e1"], equation=ein4),
        n("Einsum", ["input", "Bc", "S0", "S1", "input", "A", "S0", "S2", "input", "A", "S1", "S2", "input", "Cg", "S0", "S0", "input", "Cg", "S2", "S2", "one1"], ["e2"], equation=ein5),
        n("Einsum", ["input", "Bc", "S0", "S0", "input", "A", "S1", "S0", "input", "A", "S1", "S1", "input", "A", "S2", "S0", "one1"], ["e3"], equation=ein4),
        n("Einsum", ["input", "Bc", "S0", "S2", "input", "A", "S1", "S1", "input", "A", "S1", "S2", "input", "A", "S2", "S2", "one1"], ["e5"], equation=ein4),
        n("Einsum", ["input", "Bc", "S1", "S0", "input", "A", "S2", "S0", "input", "A", "S2", "S1", "input", "Cg", "S0", "S1", "input", "Cg", "S2", "S2", "one1"], ["e6"], equation=ein5),
        n("Einsum", ["input", "Bc", "S1", "S1", "input", "A", "S2", "S0", "input", "A", "S2", "S1", "input", "A", "S2", "S2", "one1"], ["e7"], equation=ein4),
        n("Einsum", ["input", "Bc", "S1", "S2", "input", "A", "S2", "S1", "input", "A", "S2", "S2", "input", "Cg", "S0", "S0", "input", "Cg", "S1", "S1", "one1"], ["e8"], equation=ein5),
        n("Concat", ["e0", "e1", "e2", "e3", "gray5", "e5", "e6", "e7", "e8"], ["bankf"], axis=0),
        n("Cast", ["bankf"], ["bank"], to=U8),
        n("Gather", ["bank", "tpl"], ["code"], axis=0),
        n("Mul", ["code", "code"], ["code_sq"]),
        n("Concat", ["ones99", "code", "code_sq"], ["features"], axis=1),
        n("ConvInteger", ["features", "eqw"], ["output"], pads=[0, 0, 21, 21]),
    ]

    graph = helper.make_graph(
        nodes,
        "task264_convint_tail",
        [helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", I32, [1, 10, 30, 30])],
        initializer=inits,
    )
    model = helper.make_model(graph, producer_name="task264_codex", opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


if __name__ == "__main__":
    print("Design A (terminal ConvInteger renderer swap, build):")
    print("  params: extractor 1292 + tpl 81 + ones99 81 + eqw 30 = 1484")
    print("  tensors: e[8] f32[1]=32B; bankf f32[9]=36B; bank u8[9]=9B;")
    print("           code u8[1,1,9,9]=81B; code_sq u8[1,1,9,9]=81B; features u8[1,3,9,9]=243B")
    print("  total: memory 482 + params 1484 = 1966")
    print("Design B (native 16x16 crop/conv extractor, rejected):")
    print("  params: ramp 10 + kernels/templates about 96 + renderer 111 = 217")
    print("  tensors: crop f32[1,10,16,16]=10240B; color f32[1,1,16,16]=1024B;")
    print("           masks/scores at least u8/i32[1,8,14,14]>=1568B; bool tail/render >=810B")
    print("  total: >13642 before locator scratch, misses the 2030 bar")
    onnx.save(build(), "task264.onnx")
    print("saved task264.onnx")
