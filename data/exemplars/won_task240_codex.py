#!/usr/bin/env python3
"""Build task240.onnx.

True rule: mirror the sparse odd-lattice corner seed into all four quadrants,
and use each diagonal seed's adjacent right/below color to draw the nested
dotted square frame for that radius.

Priced designs:
  A. Shared-bank polynomial terminal Einsum, chosen:
     bank[9,30] 270 + psel[4,9] 36 + src_r[8,4] 32
     + src_c[8,4] 32 + chfeat[3,10] 30 + fgate[3,2] 6
     + sgate[2,8] 16 + coeff[9,8] 72 + chan[3,10] 30;
     memory 0; cost 524.
     This replaces the banked 706-cost channel-map renderer by sharing the
     output bank for source orbits and using the sign polynomial
     1 - (color - output_channel)^2.
  B. Source-minus-black terminal Einsum:
     coord bank Z+P_i[5,30] 150 + row maps[8,2,5] 80
     + col maps[8,2,5] 80 + spatial coeff[8,9] 72
     + spatial bank[9,30] 270; memory 0; cost 652.
     This avoids channel tables by subtracting a known black cell, but misses
     the current registration ceiling.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def shared_bank():
    """Rows: full 19, nested odd intervals, then endpoint pairs."""
    bank = np.zeros((9, 30), np.float32)
    bank[0, :19] = 1.0
    for q in range(1, 5):
        lo = 2 * q - 1
        hi = 19 - lo
        bank[q, lo:hi:2] = 1.0
    for q in range(5, 9):
        i = q - 5
        bank[q, 1 + 2 * i] = 1.0
        bank[q, 17 - 2 * i] = 1.0
    return bank


def source_maps():
    # s0,s2,s4,s6 are diagonal colors; s1,s3,s5 are the right/below
    # frame colors; s7 is the first diagonal orbit used only for the
    # global background polynomial's constant feature.
    pairs = [
        (0, 0),
        (0, 1),
        (1, 1),
        (1, 2),
        (2, 2),
        (2, 3),
        (3, 3),
        (0, 0),
    ]
    sr = np.zeros((8, 4), np.float32)
    sc = np.zeros((8, 4), np.float32)
    for s, (r, c) in enumerate(pairs):
        sr[s, r] = 1.0
        sc[s, c] = 1.0
    return sr, sc


def endpoint_selector():
    p = np.zeros((4, 9), np.float32)
    for i in range(4):
        p[i, 5 + i] = 1.0
    return p


def channel_features():
    k = np.arange(10, dtype=np.float32)
    feat = np.zeros((3, 10), np.float32)
    # The first diagonal orbit always contains exactly one colored source and
    # three black mirrors, so channel-0 sum / 3 is a stable constant one.
    feat[0, 0] = 1.0 / 3.0
    feat[1] = k
    feat[2] = k * k
    return feat


def feature_gate():
    # g0: background constant only. g1: color/color^2 source features only.
    fg = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], np.float32)
    sg = np.zeros((2, 8), np.float32)
    sg[0, 7] = 1.0
    sg[1, :7] = 1.0
    return fg, sg


def spatial_coeffs():
    q = np.zeros((9, 8), np.float32)
    q[0, 7] = 1.0
    q[1, 1] = 1.0
    q[2, 1] = -1.0
    q[2, 3] = 1.0
    q[3, 3] = -1.0
    q[3, 5] = 1.0
    q[4, 5] = -1.0
    q[5, 0] = 1.0
    q[5, 1] = -1.0
    q[6, 2] = 1.0
    q[6, 3] = -1.0
    q[7, 4] = 1.0
    q[7, 5] = -1.0
    q[8, 6] = 1.0
    return q


def channel_poly():
    o = np.arange(10, dtype=np.float32)
    c = np.zeros((3, 10), np.float32)
    c[0] = 1.0 - o * o
    c[1] = 2.0 * o
    c[2] = -1.0
    return c


def init(name, arr):
    return numpy_helper.from_array(np.asarray(arr, np.float32), name=name)


def main():
    bank = shared_bank()
    src_r, src_c = source_maps()
    fgate, sgate = feature_gate()
    nodes = [
        helper.make_node(
            "Einsum",
            [
                "input",
                "bank",
                "bank",
                "psel",
                "psel",
                "src_r",
                "src_c",
                "chfeat",
                "fgate",
                "sgate",
                "coeff",
                "bank",
                "bank",
                "chan",
            ],
            ["output"],
            equation="nkrc,ar,bc,da,eb,sd,se,fk,fg,gs,qs,qh,qw,fo->nohw",
        )
    ]
    inits = [
        init("bank", bank),
        init("psel", endpoint_selector()),
        init("src_r", src_r),
        init("src_c", src_c),
        init("chfeat", channel_features()),
        init("fgate", fgate),
        init("sgate", sgate),
        init("coeff", spatial_coeffs()),
        init("chan", channel_poly()),
    ]
    graph = helper.make_graph(
        nodes,
        "task240_shared_bank_poly",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.save(model, "task240.onnx")

    params = sum(int(np.prod(numpy_helper.to_array(t).shape)) for t in model.graph.initializer)
    print(f"saved task240.onnx; params={params} memory=0 cost={params}")


if __name__ == "__main__":
    main()
