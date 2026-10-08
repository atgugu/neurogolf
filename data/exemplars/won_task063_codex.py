#!/usr/bin/env python3
"""Build task063 with the reconciled rule and a shared rank-2 terminal renderer.

The literal Lane-A crop is physically over budget: Slice(input) would create a
charged f32[1,10,14,14] tensor (7,840 B).  The permitted secondary family keeps
the Kaggle-proven algorithm and its f32 terminal Einsum, while sharing one
rank-2 color basis between source and destination.  A 2-vector selects the
background row of that same basis for the two count contractions.

Pre-build scorer budget (every non-output node):

    row_count       f32[1,1,30,1]  = 120 B
    col_count       f32[1,1,1,30]  = 120 B
    area_count      f32[1]         =   4 B
    size_count      f32[1]         =   4 B
    target_count    f32[1]         =   4 B
    row_empty30     bool[1,1,30,1] =  30 B
    col_empty30     bool[1,1,1,30] =  30 B
    row_basis       f32[1,2,30,1]  = 240 B
    col_basis       f32[1,2,1,30]  = 240 B
    ------------------------------------------------
    charged memory                         792 B

Initializers: color_basis 20 + bg_axis 2 + interior_sel 30 +
two_scalar 1 + active_basis 2 + special_basis 2 + pair_coeff 8 = 65 params.
Paper total: 792 + 65 = 857, below the Lane-A ceiling of 864.
The final f32[1,10,30,30] Einsum output is the graph output and costs 0 B.
"""
import sys

import numpy as np

sys.path.insert(0, "../../runner")
from ngolf import G  # noqa: E402


M = 14
INTERIOR = M - 2


def build() -> G:
    g = G(task=63)
    g.opset = 14

    # Shared source/destination embedding.  Background and green are opposite
    # on axis 0; red and cyan are opposite on axis 1.  The input certificate
    # excludes green, so the -1 in source column 3 is inert before rendering.
    color_basis_np = np.zeros((2, 10), dtype=np.float32)
    color_basis_np[0, 0] = 1.0
    color_basis_np[0, 3] = -1.0
    color_basis_np[1, 2] = 1.0
    color_basis_np[1, 8] = -1.0
    color_basis = g.init(color_basis_np)

    # Reuse axis 0 of color_basis as the background-channel selector.
    bg_axis = g.init(np.array([1.0, 0.0], dtype=np.float32))

    interior_np = np.zeros((30, 1, 1), dtype=np.float32)
    interior_np[1 : INTERIOR + 1, 0, 0] = 1.0
    interior_sel = g.init(interior_np)

    two_scalar = g.init(np.array([2.0], dtype=np.float32))

    active_np = np.zeros((1, 2, 1, 1), dtype=np.float32)
    active_np[:, 0, :, :] = 1.0
    special_np = np.zeros((1, 2, 1, 1), dtype=np.float32)
    special_np[:, 1, :, :] = 1.0
    active_basis = g.init(active_np)
    special_basis = g.init(special_np)

    pair_np = np.ones((2, 2, 2), dtype=np.float32)
    pair_np[:, :, 0] = np.array(
        [[1.0, -1.0], [-1.0, -1.0]], dtype=np.float32
    )
    pair_coeff = g.init(pair_np)

    row_count = g.einsum(
        "nkhw,pk,p,wab->nahb",
        [g.input, color_basis, bg_axis, interior_sel],
        [1, 1, 30, 1],
        "f32",
    )
    col_count = g.einsum(
        "nkhw,pk,p,hab->nabw",
        [g.input, color_basis, bg_axis, interior_sel],
        [1, 1, 1, 30],
        "f32",
    )

    area_count = g.n("ReduceSum", [g.input], [1], "f32", keepdims=0)
    size_count = g.n("Sqrt", [area_count], [1], "f32")
    target_count = g.n("Sub", [size_count, two_scalar], [1], "f32")

    row_empty = g.n("Equal", [row_count, target_count], [1, 1, 30, 1], "b")
    col_empty = g.n("Equal", [col_count, target_count], [1, 1, 1, 30], "b")
    row_basis = g.n(
        "Where",
        [row_empty, special_basis, active_basis],
        [1, 2, 30, 1],
        "f32",
    )
    col_basis = g.n(
        "Where",
        [col_empty, special_basis, active_basis],
        [1, 2, 1, 30],
        "f32",
    )

    g.n(
        "Einsum",
        [g.input, row_basis, col_basis, pair_coeff, color_basis, color_basis],
        [1, 10, 30, 30],
        "f32",
        equation="nkhw,nahx,nbxw,abp,pk,pc->nchw",
        is_output=True,
    )
    return g


if __name__ == "__main__":
    graph = build()
    print(graph.budget())
    graph.save("task063.onnx")
