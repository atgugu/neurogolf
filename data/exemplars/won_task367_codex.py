#!/usr/bin/env python3
"""Build task367.onnx.

True rule: preserve gray borders/connectors and recolor only solid black box
interiors yellow.

Priced designs before build:

1. Target family: packed row-bitset flood.
   tensor budget if the seed union were expressible directly in row bits:
   v_sl f32[1,1,20,20]=1600; row/col profiles + packed black/valid masks
   about 500; seed row-bit masks about 720; five packed dilations about 1280;
   unpack uint32[1,1,20,20]=1600; v_fill bool[1,1,20,20]=400;
   v_pad bool[1,1,30,30]=900; params about 90. Paper target: about 7090,
   below the pin but not enough after exact corner-OR seed expansion; rejected
   for this fallback attempt because the exact seed formulas add too many
   charged bitwise temporaries.

2. Built fallback: integrated 6x6 QLinearConv seed bank with top/bottom
   boundary predicates merged into one value-exact linear channel.
   tensor budget: v_sl f32[1,1,20,20]=1600; v_z u8[1,1,20,20]=400;
   v_row u8[1,1,20,1]=20; v_colmask u8[1,1,1,20]=20;
   v_valid u8[1,1,20,20]=400; v_u8 u8[1,2,20,20]=800;
   v_main u8[1,7,20,20]=2800; r1c/r1/r2c/r2/r3c/r3/r4c/r4/r5c=9*400;
   v_fill bool[1,1,20,20]=400; v_pad bool[1,1,30,30]=900.
   params: slice/pad/scalars/e4=31; Wm=7*2*6*6=504; Bm=7;
   Wc1=1*7*3*3=63; Wc=9. Total params=614, memory=10940,
   cost=11554.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper as nh, shape_inference


def make_wm5() -> tuple[np.ndarray, np.ndarray]:
    wm = np.zeros((10, 2, 5, 5), dtype=np.int8)
    bm = np.array([0, 0, -3, -4, -3, -4, -3, -3, -3, -3], dtype=np.int32)

    wm[0, 0, 1, 2] = -1
    wm[0, 0, 2, 2] = 1
    wm[0, 1, 1, 2] = -1

    wm[1, 0, 2, 2] = 1
    wm[1, 0, 3, 2] = -1
    wm[1, 1, 3, 2] = -1

    for ch, bias in ((2, -3), (3, -4)):
        bm[ch] = bias
        wm[ch, 0, 2, 2] = 2
        wm[ch, 0, 2, 3] = 1
        wm[ch, 0, 3, 2] = 1
        wm[ch, 0, 3, 3] = 1

    for ch, bias in ((4, -3), (5, -4)):
        bm[ch] = bias
        wm[ch, 0, 1, 1] = 1
        wm[ch, 0, 1, 2] = 1
        wm[ch, 0, 2, 1] = 1
        wm[ch, 0, 2, 2] = 2

    wm[6, 0, 1, 2] = -2
    wm[6, 0, 2, 1] = -2
    wm[6, 0, 2, 2] = 3
    wm[6, 1, 0, 1] = 1
    wm[6, 1, 1, 0] = 1

    wm[7, 0, 1, 2] = -2
    wm[7, 0, 2, 2] = 3
    wm[7, 0, 2, 3] = -2
    wm[7, 1, 0, 3] = 1
    wm[7, 1, 1, 4] = 1

    wm[8, 0, 2, 1] = -2
    wm[8, 0, 2, 2] = 3
    wm[8, 0, 3, 2] = -2
    wm[8, 1, 3, 0] = 1
    wm[8, 1, 4, 1] = 1

    wm[9, 0, 2, 2] = 3
    wm[9, 0, 2, 3] = -2
    wm[9, 0, 3, 2] = -2
    wm[9, 1, 3, 4] = 1
    wm[9, 1, 4, 3] = 1

    # Replace [black, gray] by [black, valid], where gray = valid - black.
    return np.stack([wm[:, 0] - wm[:, 1], wm[:, 1]], axis=1).astype(np.int8), bm


def make_integrated_bank() -> tuple[np.ndarray, np.ndarray]:
    wm5, bm5 = make_wm5()
    wm = np.zeros((7, 2, 6, 6), dtype=np.int8)
    bm = np.zeros((7,), dtype=np.int32)

    # Merge the old top and bottom boundary predicates:
    #   B(center) & (~V(up) | ~V(down))
    # as the linear threshold 2*B(center) - V(up) - V(down) > 0.
    wm[0, 0, 2, 2] = 2
    wm[0, 1, 1, 2] = -1
    wm[0, 1, 3, 2] = -1

    for out_ch, old_ch in enumerate((6, 7, 8, 9), start=1):
        wm[out_ch, :, 0:5, 0:5] = wm5[old_ch]
        bm[out_ch] = bm5[old_ch]

    # Two length-6 black-run probes replace the separate Wh/Wv planes.
    wm[5, 0, 2, :] = 1
    bm[5] = -5
    wm[6, 0, :, 2] = 1
    bm[6] = -5
    return wm, bm


def main() -> None:
    wm, bm = make_integrated_bank()

    wc1 = np.zeros((1, wm.shape[0], 3, 3), dtype=np.int8)
    wc1[0, :, 1, 1] = 1
    wc1[0, :, 0, 1] = 1
    wc1[0, :, 2, 1] = 1
    wc1[0, :, 1, 0] = 1
    wc1[0, :, 1, 2] = 1

    wc = np.array([[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]], dtype=np.int8)
    e4 = np.zeros((1, 10, 1, 1), dtype=np.float32)
    e4[0, 4, 0, 0] = 1.0

    inits = [
        nh.from_array(np.array([0, 0, 0], dtype=np.int64), "st3"),
        nh.from_array(np.array([1, 20, 20], dtype=np.int64), "en3"),
        nh.from_array(np.array([1, 2, 3], dtype=np.int64), "ax3"),
        nh.from_array(np.array(1.0, dtype=np.float32), "qs"),
        nh.from_array(np.array(0, dtype=np.uint8), "zu"),
        nh.from_array(np.array(0, dtype=np.int8), "zi"),
        nh.from_array(np.array([0, 0, 0, 0, 0, 0, 10, 10], dtype=np.int64), "pad8"),
        nh.from_array(np.array(False, dtype=np.bool_), "fls"),
        nh.from_array(e4, "e4"),
        nh.from_array(wm, "Wm"),
        nh.from_array(bm, "Bm"),
        nh.from_array(wc1, "Wc1"),
        nh.from_array(wc, "Wc"),
    ]

    nodes = [
        helper.make_node("Slice", ["input", "st3", "en3", "ax3"], ["v_sl"]),
        helper.make_node("Cast", ["v_sl"], ["v_z"], to=TensorProto.UINT8),
        helper.make_node("ReduceMax", ["v_z"], ["v_row"], axes=[3], keepdims=1),
        helper.make_node("ReduceMax", ["v_z"], ["v_colmask"], axes=[2], keepdims=1),
        helper.make_node("Min", ["v_row", "v_colmask"], ["v_valid"]),
        helper.make_node("Concat", ["v_z", "v_valid"], ["v_u8"], axis=1),
        helper.make_node(
            "QLinearConv",
            ["v_u8", "qs", "zu", "Wm", "qs", "zi", "qs", "zu", "Bm"],
            ["v_main"],
            pads=[2, 2, 3, 3],
        ),
        # v_main channels are positive seed predicates; this first propagation
        # stencil directly OR-dilates them, deleting the old v_col combiner plane.
        helper.make_node(
            "QLinearConv",
            ["v_main", "qs", "zu", "Wc1", "qs", "zi", "qs", "zu"],
            ["r1c"],
            pads=[1, 1, 1, 1],
        ),
        helper.make_node("Min", ["r1c", "v_z"], ["r1"]),
    ]

    prev = "r1"
    for i in range(2, 6):
        conv = f"r{i}c"
        nodes.append(
            helper.make_node(
                "QLinearConv",
                [prev, "qs", "zu", "Wc", "qs", "zi", "qs", "zu"],
                [conv],
                pads=[1, 1, 1, 1],
            )
        )
        if i < 5:
            masked = f"r{i}"
            nodes.append(helper.make_node("Min", [conv, "v_z"], [masked]))
            prev = masked

    nodes.extend(
        [
            helper.make_node("Less", ["r5c", "v_z"], ["v_fill"]),
            helper.make_node("Pad", ["v_fill", "pad8", "fls"], ["v_pad"]),
            helper.make_node("Where", ["v_pad", "e4", "input"], ["output"]),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task367_integrated_direct_r1",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 14)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.save(model, "task367.onnx")
    print("saved task367.onnx")


if __name__ == "__main__":
    main()
