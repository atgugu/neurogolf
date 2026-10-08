#!/usr/bin/env python3
"""Build task069.onnx after probing the dense uint8 color-grid contract.

True rule: erase the original multicolor sprite, then stamp its normalized
coloring onto every same-shaped cyan sprite at the cyan component bbox TL.

Design prices before building:

1. Dense bbox-stamp rule rebuild, target <=939 B:
   - color extraction: dense_f [1,1,10,10] f32 = 400 B, x [1,1,10,10] u8 = 100 B,
     params color/row/col selectors = 310.
   - masks/base: cyan, fg, template, base as 10x10 bool/u8 tensors = ~400 B.
   - bbox-anchor/stamp: standard ONNX Slice/Pad shifts over 10x10 masks cost
     another ~100 B per shifted tensor, with >=12 such tensors before stamping.
   - conclusion: executable standard lowering lands above the 1617 registration
     bar before the 3x4 color stamp cascade, so it does not close.

2. Dense qtail-core with terminal dense contract, target <=939 B:
   - reuse verified non-cyan color projection, template crop, and cyan correlation
     core from the prior correct qtail family.
   - remove the one-hot/output padding tail entirely; final Gather extracts the
     color-code channel from stamp_code [1,2,10,10] u8 as output [1,1,10,10] u8.
   - probe result: price accepted the dense tensor, but gate124 still compares
     raw[0] to a [10,30,30] one-hot target, so dense color IDs are not a valid
     correctness contract.

3. Implemented fallback: one-hot sentinel tail under the old contract.
   - generator cert gives template height <=3 and width <=4, so crop indices
     row0+[0,1,2] and col0+[0,1,2,3] are either in bounds or past the
     bottom/right edge. Invalid indices are routed to absolute index 0, which
     is guaranteed zero in Tcol for exactly those edge cases.
   - stamp_code uses scalar output zero point 1, encoding in-grid cells as
     z=color+1 via features (z^2, z) while padded outside remains (0,0).
   - final 1x1 QLinearConv renders each color from that 2-feature code.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, "../../runner")
from ngolf import relower_onehot_plane


OUT = Path(__file__).with_name("task069.onnx")
SZ = 10
KH = 3
KW = 4


def init(name: str, arr) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr), name=name)


def build() -> onnx.ModelProto:
    # Project the 10x10 active input area from one-hot to color IDs, with cyan
    # mapped to zero so the only nonzero component is the template.
    colid = np.arange(10, dtype=np.float32).reshape(1, 10)
    colid[0, 8] = 0.0
    selector = np.eye(SZ, 30, dtype=np.float32)

    relower_nodes, relower_inits, relower_shape = relower_onehot_plane(
        "input",
        "Eu8",
        channel=8,
        crop=((0, 1), (8, 9), (0, SZ), (0, SZ)),
        dtype="u8",
        starts_name="eu8_starts",
        ends_name="eu8_ends",
    )
    assert relower_shape == [1, 1, SZ, SZ]

    inits = [
        init("colidW", colid),
        init("sel10x30", selector),
        init("qscale", np.array(1.0, dtype=np.float32)),
        init("c0", np.array(0, dtype=np.uint8)),
        init("c0_i32", np.array(0, dtype=np.int32)),
        init("c10_i32", np.array(10, dtype=np.int32)),
        init("row_offsets", np.array([0, 1, 2], dtype=np.int32)),
        init("col_offsets", np.array([0, 1, 2, 3], dtype=np.int32)),
        init("flip_starts", np.array([KH - 1, KW - 1], dtype=np.int64)),
        init("flip_ends", np.array([-KH - 1, -KW - 1], dtype=np.int64)),
        init("flip_axes", np.array([2, 3], dtype=np.int64)),
        init("flip_steps", np.array([-1, -1], dtype=np.int64)),
    ]
    inits.extend(init(name, arr) for name, arr in relower_inits)

    tail_w = np.zeros((10, 2, 1, 1), dtype=np.int8)
    tail_b = np.zeros((10,), dtype=np.int32)
    for k in range(10):
        z = k + 1
        tail_w[k, 0, 0, 0] = -1
        tail_w[k, 1, 0, 0] = 2 * z
        tail_b[k] = 1 - z * z
    inits.extend(
        [
            init("tail_w", tail_w),
            init("tail_b", tail_b),
            init("c0_i8", np.array(0, dtype=np.int8)),
        ]
    )

    nodes = [
        helper.make_node(
            "Einsum",
            ["input", "colidW", "sel10x30", "sel10x30"],
            ["Tproj"],
            equation="nchw,oc,rh,sw->nors",
        ),
        helper.make_node("Cast", ["Tproj"], ["Tcol"], to=TensorProto.UINT8),
        helper.make_node("ReduceMax", ["Tcol"], ["row_color"], axes=[0, 1, 3], keepdims=0),
        helper.make_node("ReduceMax", ["Tcol"], ["col_color"], axes=[0, 1, 2], keepdims=0),
        helper.make_node("Min", ["row_color", "c0"], ["row_has_bad"]),
    ]

    # Min with c0 above would be wrong; overwrite the tiny row/col occupancy
    # using Clip semantics from the original qtail build: Min(color, 1).
    nodes.pop()
    inits.append(init("c1", np.array(1, dtype=np.uint8)))
    nodes.extend(
        [
            helper.make_node("Min", ["row_color", "c1"], ["row_has"]),
            helper.make_node("Min", ["col_color", "c1"], ["col_has"]),
            helper.make_node("ArgMax", ["row_has"], ["row0_i64"], axis=0, keepdims=0),
            helper.make_node("ArgMax", ["col_has"], ["col0_i64"], axis=0, keepdims=0),
            helper.make_node("Cast", ["row0_i64"], ["row0"], to=TensorProto.INT32),
            helper.make_node("Cast", ["col0_i64"], ["col0"], to=TensorProto.INT32),
            helper.make_node("Add", ["row0", "row_offsets"], ["row_idx"]),
            helper.make_node("Add", ["col0", "col_offsets"], ["col_idx"]),
            helper.make_node("Less", ["row_idx", "c10_i32"], ["row_valid"]),
            helper.make_node("Less", ["col_idx", "c10_i32"], ["col_valid"]),
            helper.make_node("Where", ["row_valid", "row_idx", "c0_i32"], ["row_idx_safe"]),
            helper.make_node("Where", ["col_valid", "col_idx", "c0_i32"], ["col_idx_safe"]),
            helper.make_node("Gather", ["Tcol", "row_idx_safe"], ["Tc_rows"], axis=2),
            helper.make_node("Gather", ["Tc_rows", "col_idx_safe"], ["Tc_crop"], axis=3),
        ]
    )
    nodes.extend(relower_nodes)
    nodes.extend(
        [
            # Divide the cyan/template correlation by its maximum.  Because
            # generated boxes never overlap and all cyan copies have the same
            # shape as the template, Du8 is one exactly at copy bbox anchors.
            helper.make_node(
                "QLinearConv",
                ["Eu8", "qscale", "c0", "Tc_crop", "qscale", "c0", "qscale", "c0"],
                ["corr_u8"],
                pads=[0, 0, KH - 1, KW - 1],
            ),
            helper.make_node(
                "ReduceMax",
                ["corr_u8"],
                ["corr_max"],
                axes=[0, 1, 2, 3],
                keepdims=0,
            ),
            helper.make_node("Div", ["corr_u8", "corr_max"], ["Du8"]),
            helper.make_node(
                "Slice",
                ["Tc_crop", "flip_starts", "flip_ends", "flip_axes", "flip_steps"],
                ["Tc_flip"],
            ),
            helper.make_node("Mul", ["Tc_flip", "Tc_flip"], ["Tc_sq"]),
            helper.make_node("Add", ["Tc_flip", "Tc_flip"], ["Tc_double"]),
            helper.make_node("Add", ["Tc_sq", "Tc_double"], ["Tc_zsq_minus1"]),
            helper.make_node("Concat", ["Tc_zsq_minus1", "Tc_flip"], ["Tc_pair"], axis=0),
            helper.make_node(
                "QLinearConv",
                ["Du8", "qscale", "c0", "Tc_pair", "qscale", "c0", "qscale", "c1"],
                ["stamp_code"],
                pads=[KH - 1, KW - 1, 0, 0],
            ),
            helper.make_node(
                "QLinearConv",
                [
                    "stamp_code",
                    "qscale",
                    "c0",
                    "tail_w",
                    "qscale",
                    "c0_i8",
                    "qscale",
                    "c0",
                    "tail_b",
                ],
                ["output"],
                kernel_shape=[1, 1],
                pads=[0, 0, 20, 20],
            ),
        ]
    )

    graph = helper.make_graph(
        nodes,
        "task069_sentinel_onehot_tail",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.UINT8, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    model = build()
    onnx.save(model, OUT)
    params = sum(int(np.prod(t.dims)) for t in model.graph.initializer)
    print(f"saved {OUT} nodes={len(model.graph.node)} params={params}")


if __name__ == "__main__":
    main()
