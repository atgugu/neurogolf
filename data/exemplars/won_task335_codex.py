import os
#!/usr/bin/env python3
"""Build task335 candidate.

The audited 3x3 LUT family is not source-equivalent for this task: a blank
3x3 neighborhood can be either background or yellow depending on remote
endpoint coordinates. This build keeps the verified rank-2 rule but tries a
smaller terminal renderer: concatenate u8 rank terms before one float cast and
factor the final coefficient tensor through a rank-3 latent.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import TensorProto as T
from onnx import helper as h
from onnx import numpy_helper as nh


ROOT = Path(__file__).resolve().parent
CLEAN = Path((os.environ.get("NEUROGOLF_CLEAN", "") + ""))
sys.path.insert(0, str(CLEAN))

from neurogolf.builders.common import fp32_input, fp32_output, make_model  # noqa:E402


N = 17


def build() -> onnx.ModelProto:
    nodes = []
    inits = []

    def init(name: str, arr) -> None:
        inits.append(nh.from_array(np.asarray(arr), name=name))

    def i64(name: str, vals) -> None:
        init(name, np.asarray(vals, dtype=np.int64))

    # Latents: black, red, cyan. The same matrix is used as the input selector
    # and output selector. The black latent has an extra -1 at yellow; generated
    # inputs never contain yellow, so endpoint extraction still sees pure black,
    # while the terminal renderer gets the yellow correction for path cells.
    cmap = np.zeros((10, 3), dtype=np.float32)
    cmap[0, 0] = 1.0
    cmap[4, 0] = -1.0
    cmap[2, 1] = 1.0
    cmap[8, 2] = 1.0
    init("cmap", cmap)
    init("coord", np.arange(30, dtype=np.float32))

    nodes.append(h.make_node("Einsum", ["input", "cmap", "coord"], ["row_f"], equation="nchw,cl,h->nl"))
    nodes.append(h.make_node("Einsum", ["input", "cmap", "coord"], ["col_f"], equation="nchw,cl,w->nl"))
    nodes.append(h.make_node("Cast", ["row_f"], ["row_u"], to=T.UINT8))
    nodes.append(h.make_node("Cast", ["col_f"], ["col_u"], to=T.UINT8))

    i64("i1", [1])
    i64("i2", [2])
    for src, idx, out in (
        ("row_u", "i1", "r2"),
        ("row_u", "i2", "r8"),
        ("col_u", "i1", "c2"),
        ("col_u", "i2", "c8"),
    ):
        nodes.append(h.make_node("Gather", [src, idx], [out], axis=1))

    init("idx17", np.arange(1, N + 1, dtype=np.uint8).reshape(1, 1, N))
    init("one_u", np.array(1, dtype=np.uint8))
    init("two_u", np.array(2, dtype=np.uint8))

    # row_score: 2 on the red row, 1 on the vertical leg, 0 elsewhere.
    nodes.append(h.make_node("Less", ["r2", "r8"], ["r2_lt_r8"]))
    nodes.append(h.make_node("Sub", ["r2", "one_u"], ["r2_m1"]))
    nodes.append(h.make_node("Add", ["r2", "one_u"], ["r2_p1"]))
    nodes.append(h.make_node("Where", ["r2_lt_r8", "r2_m1", "r8"], ["vlo_m1"]))
    nodes.append(h.make_node("Where", ["r2_lt_r8", "r8", "r2_p1"], ["vhi_p1"]))
    nodes.append(h.make_node("Greater", ["idx17", "vlo_m1"], ["v_ge_lo"]))
    nodes.append(h.make_node("Less", ["idx17", "vhi_p1"], ["v_le_hi"]))
    nodes.append(h.make_node("And", ["v_ge_lo", "v_le_hi"], ["vrow"]))
    nodes.append(h.make_node("Equal", ["idx17", "r2"], ["r2_row"]))
    nodes.append(h.make_node("Cast", ["vrow"], ["vrow_u"], to=T.UINT8))
    nodes.append(h.make_node("Where", ["r2_row", "two_u", "vrow_u"], ["row_score17"]))

    # inv17: 2 on the cyan column, 1 on the horizontal leg, 0 elsewhere.
    nodes.append(h.make_node("Less", ["c2", "c8"], ["c2_lt_c8"]))
    nodes.append(h.make_node("Sub", ["c8", "one_u"], ["c8_m1"]))
    nodes.append(h.make_node("Add", ["c8", "one_u"], ["c8_p1"]))
    nodes.append(h.make_node("Where", ["c2_lt_c8", "c2", "c8_m1"], ["hlo_m1"]))
    nodes.append(h.make_node("Where", ["c2_lt_c8", "c8_p1", "c2"], ["hhi_p1"]))
    nodes.append(h.make_node("Greater", ["idx17", "hlo_m1"], ["h_ge_lo"]))
    nodes.append(h.make_node("Less", ["idx17", "hhi_p1"], ["h_le_hi"]))
    nodes.append(h.make_node("And", ["h_ge_lo", "h_le_hi"], ["hcol"]))
    nodes.append(h.make_node("Equal", ["idx17", "c8"], ["c8_col"]))
    nodes.append(h.make_node("Cast", ["hcol"], ["hcol_u"], to=T.UINT8))
    nodes.append(h.make_node("Where", ["c8_col", "two_u", "hcol_u"], ["inv17"]))

    i64("pads_17_30", [1, 12])
    init("ones30_u", np.ones((1, 1, 30), dtype=np.uint8))
    nodes.append(h.make_node("Pad", ["row_score17", "pads_17_30", "", "i2"], ["row_score"], mode="constant"))
    nodes.append(h.make_node("Pad", ["inv17", "pads_17_30", "", "i2"], ["inv_col"], mode="constant"))
    nodes.append(h.make_node("Concat", ["ones30_u", "row_score"], ["row_terms_u"], axis=1))
    nodes.append(h.make_node("Concat", ["ones30_u", "inv_col"], ["col_terms_u"], axis=1))
    nodes.append(h.make_node("Cast", ["row_terms_u"], ["row_terms"], to=T.FLOAT))
    nodes.append(h.make_node("Cast", ["col_terms_u"], ["col_terms"], to=T.FLOAT))

    # Factorized coeff[k, latent, term]. `cmap` supplies both input and output
    # color factors; term_lat supplies the identity/path score per latent.
    term_lat = np.array(
        [
            [1.5, 1.0, 1.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    init("term_lat", term_lat)
    nodes.append(
        h.make_node(
            "Einsum",
            ["input", "cmap", "row_terms", "col_terms", "cmap", "term_lat"],
            ["output"],
            equation="nchw,cl,nth,ntw,kl,tl->nkhw",
        )
    )

    model = make_model(
        nodes,
        [fp32_input()],
        [fp32_output()],
        inits,
        opset=18,
        name="task335_rank2_u8concat_cpfinal",
    )
    del model.graph.value_info[:]
    onnx.checker.check_model(model)
    return model


def main() -> None:
    out = ROOT / "task335.onnx"
    onnx.save(build(), out)
    print(out)


if __name__ == "__main__":
    main()
