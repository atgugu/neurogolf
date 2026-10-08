#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
TASK = "task085.onnx"
H = W = 30
K = 10
MIDDLE_ROWS = np.array([2, 5, 6, 8, 9, 10, 11, 12, 13, 14], dtype=np.int64)


def _init(name: str, array) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(array), name=name)


def _terminal_bases() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rank-3 threshold-exact factorization of the terminal channel table.

    The identity term scores source colour j at output colour k as
    1-(k-j)^2.  At a punched cell the correction cancels the j terms,
    leaving 1-k^2, which selects background channel zero only.
    """
    k = np.arange(K, dtype=np.float32)
    j = np.arange(K, dtype=np.float32)
    chan_k = np.stack((1.0 - k * k, k, np.ones(K, dtype=np.float32)), axis=1)
    chan_j = np.stack((np.ones(K, dtype=np.float32), j, j * j), axis=1)
    chan_a = np.array(
        [
            [1.0, 2.0, -1.0],  # 1 - (k-j)^2
            [0.0, -2.0, 1.0],  # correction to 1 - k^2
            [0.0, -2.0, 1.0],
        ],
        dtype=np.float32,
    )
    return chan_k, chan_j, chan_a


def _row_selector() -> np.ndarray:
    # Selector slot 0 means "all rows" for the identity term. Slots 1..10
    # select the only generator-reachable middle rows.
    selector = np.zeros((1 + len(MIDDLE_ROWS), H), dtype=np.float32)
    selector[0, :] = 1.0
    for i, row in enumerate(MIDDLE_ROWS, start=1):
        selector[i, row] = 1.0
    return selector


def _col_terms() -> np.ndarray:
    cols = np.zeros((3, W), dtype=np.float32)
    cols[0, :] = 1.0
    cols[1, 0::2] = 1.0  # correction for the first parity family
    cols[2, 1::2] = 1.0  # correction for the second parity family
    return cols


def build() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []

    def add(op: str, inputs: list[str], outputs: list[str], **attrs) -> None:
        nodes.append(helper.make_node(op, inputs, outputs, **attrs))

    code_w = np.zeros(K, dtype=np.float32)
    for color in range(1, K):
        code_w[color] = 13.0 * color + 1.0

    mid_in_code15 = MIDDLE_ROWS - 1
    chan_k, chan_j, chan_a = _terminal_bases()
    inits = [
        _init("code_w", code_w),
        _init("sel", np.array([0.0] + [1.0] * 9, dtype=np.float32)),
        _init("alt_w", (1.0 - 2.0 * (np.arange(W) % 2)).astype(np.float32)),
        _init("s1", np.array([1], dtype=np.int64)),
        _init("s16", np.array([16], dtype=np.int64)),
        _init("ax1", np.array([1], dtype=np.int64)),
        _init("idx_top", (mid_in_code15 - 1).astype(np.int64)),
        _init("idx_mid", mid_in_code15.astype(np.int64)),
        _init("idx_bot", (mid_in_code15 + 1).astype(np.int64)),
        _init("idx_rows", MIDDLE_ROWS),
        _init("z16", np.array(0, dtype=np.float16)),
        _init("z32", np.array(0, dtype=np.float32)),
        _init("zlead", np.zeros((1, 1), dtype=np.float32)),
        _init("row0", np.array([[1.0] + [0.0] * len(MIDDLE_ROWS)], dtype=np.float32)),
        _init("row_sel", _row_selector()),
        _init("col_terms", _col_terms()),
        _init("chan_k", chan_k),
        _init("chan_j", chan_j),
        _init("chan_a", chan_a),
    ]

    # One collision-free row code replaces the live graph's separate color-sum
    # and foreground-count projections for generator rows.
    add("Einsum", ["input", "code_w"], ["Code32"], equation="nkhw,k->nh")
    add("Einsum", ["input", "sel", "alt_w"], ["Alt32"], equation="nkhw,k,w->nh")
    add("Slice", ["Code32", "s1", "s16", "ax1"], ["Code15_32"])
    add("Cast", ["Code15_32"], ["Code15"], to=TensorProto.FLOAT16)
    add("Gather", ["Code15", "idx_top"], ["CodeTop"], axis=1)
    add("Gather", ["Code15", "idx_mid"], ["CodeMid"], axis=1)
    add("Gather", ["Code15", "idx_bot"], ["CodeBot"], axis=1)
    add("Equal", ["CodeMid", "CodeTop"], ["eq_top"])
    add("Equal", ["CodeMid", "CodeBot"], ["eq_bot"])
    add("Greater", ["CodeMid", "z16"], ["row_pos"])
    add("And", ["eq_top", "eq_bot"], ["eq_both"])
    add("And", ["eq_both", "row_pos"], ["middle"])

    add("Gather", ["Alt32", "idx_rows"], ["AltMid"], axis=1)
    add("Greater", ["AltMid", "z32"], ["left_even"])
    add("Not", ["left_even"], ["left_odd_any"])
    add("And", ["middle", "left_odd_any"], ["left_odd"])
    add("And", ["middle", "left_even"], ["left_even_mid"])
    add("Cast", ["left_odd"], ["left_odd_f"], to=TensorProto.FLOAT)
    add("Cast", ["left_even_mid"], ["left_even_f"], to=TensorProto.FLOAT)
    add("Concat", ["zlead", "left_odd_f"], ["row_odd"], axis=1)
    add("Concat", ["zlead", "left_even_f"], ["row_even"], axis=1)
    add("Concat", ["row0", "row_odd", "row_even"], ["row_terms"], axis=0)

    add(
        "Einsum",
        ["input", "row_terms", "row_sel", "col_terms", "chan_k", "chan_j", "chan_a"],
        ["output"],
        equation="njhw,am,mh,aw,kr,jr,ar->nkhw",
    )

    graph = helper.make_graph(
        nodes,
        "task085_rowcode_terminal_einsum",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, K, H, W])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, K, H, W])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 18)])
    model.ir_version = 10
    del model.graph.value_info[:]
    return model


def main() -> None:
    out = HERE / TASK
    onnx.save(build(), out)
    print(out)


if __name__ == "__main__":
    main()
