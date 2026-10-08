#!/usr/bin/env python3
"""Build task251.onnx.

True rule: preserve red and fill black cells blue only when their black
component is not connected to the real grid boundary.

Diagnostic fallback: banked complement repair with `near_red` removed.  This
checks whether the banked repair's local-near guard is load-bearing after the
attempt-13 packed-flood family proved physically over budget.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runner"))
from ngolf import relower_onehot_plane  # noqa: E402


OUT = Path(__file__).resolve().parent / "task251.onnx"
REPAIR_ROUNDS = 3

F32 = TensorProto.FLOAT
U8 = TensorProto.UINT8


def init(name: str, value, dtype=None) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name)


def node(nodes: list[onnx.NodeProto], op: str, ins: list[str], out: str, **attrs) -> str:
    nodes.append(helper.make_node(op, ins, [out], **attrs))
    return out


def build_no_near_complement() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits = [init("pad30", [0, 0, 0, 0, 0, 7, 18, 18], np.int64)]

    z_nodes, z_inits, _ = relower_onehot_plane(
        "input",
        "Z",
        channel=0,
        crop=((0, 1), (0, 1), (0, 12), (0, 12)),
        dtype="u8",
        starts_name="st_z",
        ends_name="en_z",
    )
    c2_nodes, c2_inits, _ = relower_onehot_plane(
        "input",
        "C2u",
        channel=2,
        crop=((0, 1), (2, 3), (0, 12), (0, 12)),
        dtype="u8",
        starts_name="st_c2",
        ends_name="en_c2",
    )
    nodes.extend(z_nodes)
    nodes.extend(c2_nodes)
    inits.extend(init(name, arr) for name, arr in z_inits)
    inits.extend(init(name, arr) for name, arr in c2_inits)

    node(nodes, "MaxPool", ["C2u"], "red_l", kernel_shape=[1, 5], pads=[0, 4, 0, 0], strides=[1, 1])
    node(nodes, "MaxPool", ["C2u"], "red_r", kernel_shape=[1, 5], pads=[0, 0, 0, 4], strides=[1, 1])
    node(nodes, "MaxPool", ["C2u"], "red_u", kernel_shape=[5, 1], pads=[4, 0, 0, 0], strides=[1, 1])
    node(nodes, "MaxPool", ["C2u"], "red_d", kernel_shape=[5, 1], pads=[0, 0, 4, 0], strides=[1, 1])
    node(nodes, "Min", ["red_l", "red_r", "red_u", "red_d", "Z"], "cand_z")
    node(nodes, "Sub", ["Z", "cand_z"], "R0")

    cur = "R0"
    for i in range(REPAIR_ROUNDS):
        node(nodes, "MaxPool", [cur], f"mp{i}", kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1])
        node(nodes, "Min", [f"mp{i}", "Z"], f"R{i + 1}")
        cur = f"R{i + 1}"

    node(nodes, "Sub", ["Z", cur], "fill_u")
    node(nodes, "Concat", [cur, "fill_u", "C2u"], "stack", axis=1)
    node(nodes, "Pad", ["stack", "pad30"], "output")

    graph = helper.make_graph(
        nodes,
        "task251_no_near_complement_diagnostic",
        [helper.make_tensor_value_info("input", F32, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", U8, [1, 10, 30, 30])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)], ir_version=10)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    onnx.save(build_no_near_complement(), OUT)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
