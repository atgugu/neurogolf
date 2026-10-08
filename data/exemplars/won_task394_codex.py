#!/usr/bin/env python3
"""Task 394: factored terminal crop with bounded uint8 coordinate arithmetic."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


OUT = Path(__file__).with_name("task394.onnx")


def init(name, value, dtype=None):
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name)


def selector(rows):
    """Dense selector from the certified 0..6 crop box into 0..29 canvas coords."""
    out = np.zeros((rows, 30), dtype=np.float32)
    out[np.arange(rows), np.arange(rows)] = 1
    return out


def build():
    # P selects an input coordinate in the source certificate box.  D factors the
    # old 3x30 output selector as D(3x7) @ P(7x30) inside the free final Einsum.
    p = selector(7)
    d = np.zeros((3, 7), dtype=np.float32)
    d[np.arange(3), np.arange(3)] = 1
    inits = [
        init("e0", [1] + [0] * 9, np.float32),
        init("P", p),
        init("D", d),
        init("iota3", [0, 1, 2], np.uint8),
        init("depth7", np.array(7), np.int32),
        init("onehot_vals", [0, 1], np.float32),
        init("v49", np.array(49), np.float32),
        init("three_i", np.array(3), np.uint8),
        init("two_i", np.array(2), np.uint8),
        init("thirty_i", np.array(30), np.uint8),
    ]
    n = []
    add = lambda op, ins, outs, **attrs: n.append(helper.make_node(op, ins, outs, **attrs))

    # The black channel's row/column occupancy locates the bite; its maximum is
    # the bite side.  All following values are certified to lie in [0, 30].
    add("Einsum", ["input", "e0", "P"], ["hr"], equation="bkrc,k,ar->a")
    add("Einsum", ["input", "e0", "P"], ["hc"], equation="bkrc,k,ac->a")
    add("ArgMax", ["hr"], ["row64"], axis=0, keepdims=0)
    add("ArgMax", ["hc"], ["col64"], axis=0, keepdims=0)
    add("ReduceMax", ["hr"], ["bs_f"], axes=[0], keepdims=0)
    add("ReduceSum", ["input"], ["total"], keepdims=0)
    add("Equal", ["total", "v49"], ["is7"])
    add("Where", ["is7", "three_i", "two_i"], ["period"])
    add("Cast", ["row64"], ["row"], to=TensorProto.UINT8)
    add("Cast", ["col64"], ["col"], to=TensorProto.UINT8)
    add("Cast", ["bs_f"], ["bite"], to=TensorProto.UINT8)

    # Choose a congruent intact source row.  Modulo 2*period replaces the old
    # signed up/down branch and is exact for the certified 0..6 coordinates.
    add("Add", ["row", "iota3"], ["rbase"])
    add("Add", ["period", "period"], ["period2"])
    add("Add", ["rbase", "period"], ["rpre"])
    add("Mod", ["rpre", "period2"], ["source_row"], fmod=0)
    add("Less", ["iota3", "bite"], ["in_bite"])
    add("Where", ["in_bite", "source_row", "thirty_i"], ["row_idx"])
    add("Add", ["col", "iota3"], ["cbase"])
    add("Where", ["in_bite", "cbase", "thirty_i"], ["col_idx"])
    # ORT's OneHot kernel accepts int32 indices only; keep this widening at the
    # last possible point so the rest of the bounded coordinate path stays bytewide.
    add("Cast", ["row_idx"], ["row_idx32"], to=TensorProto.INT32)
    add("Cast", ["col_idx"], ["col_idx32"], to=TensorProto.INT32)
    add("OneHot", ["row_idx32", "depth7", "onehot_vals"], ["S_row"], axis=-1)
    add("OneHot", ["col_idx32", "depth7", "onehot_vals"], ["S_col"], axis=-1)

    add(
        "Einsum",
        ["D", "P", "S_row", "P", "input", "P", "S_col", "D", "P"],
        ["output"],
        equation="iA,AI,ia,ar,bkrc,ec,je,jE,EJ->bkIJ",
    )

    graph = helper.make_graph(
        n,
        "task394_factored_crop",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, producer_name="codex-task394", opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    return model


if __name__ == "__main__":
    onnx.save(build(), OUT)
    print(OUT)
