#!/usr/bin/env python3
"""Build task325.onnx.

True rule: count 4-connected cyan components; emit an N x N black square
with cyan on the main diagonal.

Priced designs considered:

1. PACKED-BITBOARD-BFS, audited family.  The literal plan is not directly
   executable here: ONNX/ORT rejects signed int64 BitShift, so the words must
   be uint64/u32; the mandated 10x10 crop is also source-invalid because the
   generator and validation examples use grids up to 16x16 with sprites below
   row 10.  Expanding to a source-valid 16x16 domain needs four words, and a
   legal SSA unroll charges every shift/or tensor, so the quoted 42-word budget
   is not physically attainable without an unavailable fused/in-place op.

2. Exact scalar Euler+hole counter with compact ConvTranspose terminal.  Keep
   the pin's proven scalar component formula, but replace the 9-channel compact
   Concat+Pad renderer with a one-channel ternary f16 patch: background=-1,
   outside=0, diagonal=+1.  The final ConvTranspose maps the two signed
   extremes to channels 0 and 8 and pads to [1,10,30,30].  Paper budget:
   params 945, memory 301, cost 1246.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper as H
from onnx import numpy_helper as nh


def init(name: str, arr) -> onnx.TensorProto:
    return nh.from_array(np.asarray(arr), name=name)


def main() -> None:
    e = np.zeros(10, dtype=np.float32)
    e[8] = 1.0

    s = np.zeros((30, 30), dtype=np.float32)
    for i in range(29):
        s[i, i + 1] = 1.0

    row = np.arange(6, dtype=np.uint8).reshape(1, 1, 6, 1)
    col = np.arange(6, dtype=np.uint8).reshape(1, 1, 1, 6)

    # ConvTranspose input is a one-channel ternary state:
    #   -1 = active black, 0 = outside compact output, +1 = cyan diagonal.
    # Bias -0.5 keeps all padded/no-data cells below the >0 harness threshold.
    w = np.zeros((1, 10, 1, 1), dtype=np.float16)
    w[0, 0, 0, 0] = -1.0
    w[0, 8, 0, 0] = 1.0
    bias = np.full(10, -0.5, dtype=np.float16)

    inits = [
        init("e", e),
        init("S", s),
        init("row", row),
        init("col", col),
        init("w", w),
        init("bias", bias),
        init("zero", np.array(0.0, dtype=np.float16)),
        init("neg", np.array(-1.0, dtype=np.float16)),
        init("pos", np.array(1.0, dtype=np.float16)),
    ]

    nodes = [
        H.make_node("Einsum", ["input", "e"], ["V"], equation="zchw,c->"),
        H.make_node(
            "Einsum",
            ["input", "e", "input", "e", "S"],
            ["Eh"],
            equation="zchw,c,zkhv,k,wv->",
        ),
        H.make_node(
            "Einsum",
            ["input", "e", "input", "e", "S"],
            ["Ev"],
            equation="zchw,c,zkgw,k,hg->",
        ),
        H.make_node(
            "Einsum",
            ["input", "e", "input", "e", "input", "e", "S", "input", "e", "S"],
            ["F"],
            equation="zchw,c,zkhv,k,zlgw,l,hg,zmgv,m,wv->",
        ),
        H.make_node(
            "Einsum",
            [
                "input",
                "e",
                "input",
                "e",
                "input",
                "e",
                "S",
                "input",
                "e",
                "input",
                "e",
                "S",
                "input",
                "e",
                "S",
                "input",
                "e",
                "S",
                "input",
                "e",
            ],
            ["Hole"],
            equation="zcad,c,zkae,k,zlaf,l,ab,zmbd,m,znbf,n,bg,zpgd,p,de,zqge,q,ef,zrgf,r->",
        ),
        H.make_node("Sub", ["V", "Eh"], ["t1"]),
        H.make_node("Sub", ["t1", "Ev"], ["t2"]),
        H.make_node("Add", ["t2", "F"], ["t3"]),
        H.make_node("Add", ["t3", "Hole"], ["nf"]),
        H.make_node("Cast", ["nf"], ["n8"], to=TP.UINT8),
        H.make_node("Less", ["row", "n8"], ["rowin"]),
        H.make_node("Less", ["col", "n8"], ["colin"]),
        H.make_node("And", ["rowin", "colin"], ["active"]),
        H.make_node("Equal", ["row", "col"], ["diagfull"]),
        H.make_node("And", ["rowin", "diagfull"], ["diag"]),
        H.make_node("Where", ["active", "neg", "zero"], ["bg_state"]),
        H.make_node("Where", ["diag", "pos", "bg_state"], ["state"]),
        H.make_node("ConvTranspose", ["state", "w", "bias"], ["output"], pads=[0, 0, -24, -24]),
    ]

    graph = H.make_graph(
        nodes,
        "task325_exact_scalar_convtranspose_terminal",
        [H.make_tensor_value_info("input", TP.FLOAT, [1, 10, 30, 30])],
        [H.make_tensor_value_info("output", TP.FLOAT16, [1, 10, 30, 30])],
        inits,
    )
    model = H.make_model(graph, opset_imports=[H.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, "task325.onnx")
    print("saved task325.onnx")


if __name__ == "__main__":
    main()
