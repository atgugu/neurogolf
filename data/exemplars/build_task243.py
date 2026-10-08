#!/usr/bin/env python3
"""task243 build — TRUE-RULE 4-conn zero flood adjacent to color-1.

Morphological reconstruction on 18×18 uint8 crop (certified W=20, grids ≤18).
Assembly trim: ch2–9 bool crop 18×18 + single Concat-before-Pad (vs per-channel Pad).
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

STEPS = 28
CROP = 18


def _c(name: str, arr) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr), name=name)


def build(steps: int = STEPS) -> onnx.ModelProto:
    nodes: list = []
    inits: list = []

    inits += [
        _c("ax_all", np.array([0, 1, 2, 3], np.int64)),
        _c("st_ch0", np.array([0, 0, 0, 0], np.int64)),
        _c("en_ch0", np.array([1, 1, CROP, CROP], np.int64)),
        _c("st_ch1", np.array([0, 1, 0, 0], np.int64)),
        _c("en_ch1", np.array([1, 2, CROP, CROP], np.int64)),
        _c("st_ch2", np.array([0, 2, 0, 0], np.int64)),
        _c("en_ch10", np.array([1, 10, CROP, CROP], np.int64)),
        _c("pad_back", np.array([0, 0, 0, 0, 0, 0, 30 - CROP, 30 - CROP], np.int64)),
    ]

    nodes.append(helper.make_node("Cast", ["input"], ["input_b"], to=TensorProto.BOOL))
    nodes.append(helper.make_node("Slice", ["input_b", "st_ch0", "en_ch0", "ax_all"], ["z18_b"]))
    nodes.append(helper.make_node("Slice", ["input_b", "st_ch1", "en_ch1", "ax_all"], ["o18_b"]))
    nodes.append(helper.make_node("Cast", ["z18_b"], ["zero"], to=TensorProto.UINT8))
    nodes.append(helper.make_node("Cast", ["o18_b"], ["cur0"], to=TensorProto.UINT8))
    nodes.append(helper.make_node("Max", ["zero", "cur0"], ["mask"]))

    cur = "cur0"
    for i in range(steps):
        v, h, d, nxt = f"v{i}", f"h{i}", f"d{i}", f"c{i + 1}"
        nodes.append(helper.make_node(
            "MaxPool", [cur], [v], kernel_shape=[3, 1], pads=[1, 0, 1, 0], strides=[1, 1]))
        nodes.append(helper.make_node(
            "MaxPool", [cur], [h], kernel_shape=[1, 3], pads=[0, 1, 0, 1], strides=[1, 1]))
        nodes.append(helper.make_node("Max", [v, h], [d]))
        nodes.append(helper.make_node("Min", [d, "mask"], [nxt]))
        cur = nxt

    nodes.append(helper.make_node("Min", [cur, "zero"], ["fill"]))
    nodes.append(helper.make_node("Cast", ["fill"], ["fill_b"], to=TensorProto.BOOL))
    nodes.append(helper.make_node("Or", ["o18_b", "fill_b"], ["ch1n"]))
    nodes.append(helper.make_node("Xor", ["z18_b", "fill_b"], ["ch0n"]))
    nodes.append(helper.make_node("Slice", ["input_b", "st_ch2", "en_ch10", "ax_all"], ["rest18_b"]))
    nodes.append(helper.make_node("Concat", ["ch0n", "ch1n", "rest18_b"], ["stack18"], axis=1))
    nodes.append(helper.make_node("Pad", ["stack18", "pad_back"], ["output"]))

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    out = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    g = helper.make_graph(nodes, "task243_u8_flood", [inp], [out], inits)
    m = helper.make_model(g, producer_name="ngolf-243", opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 10
    try:
        m = onnx.shape_inference.infer_shapes(m)
    except Exception:
        pass
    return m


if __name__ == "__main__":
    from collections import Counter

    m = build()
    onnx.checker.check_model(m)
    outp = "candidates/task243.onnx"
    onnx.save(m, outp)
    print(f"Saved {outp}")
    print(f"nodes={len(m.graph.node)} inits={len(m.graph.initializer)}")
    print("ops:", dict(Counter(n.op_type for n in m.graph.node)))