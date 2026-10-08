#!/usr/bin/env python3
"""task345 — interleaved uint8 rowprop state with strided ConvInteger renderer."""
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as TP, helper, numpy_helper

OPSET = 18
OUT = Path(__file__).resolve().parent / "task345.onnx"


def c(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name=name)


def build():
    nodes = []
    inits = []

    def add(name, arr):
        inits.append(c(name, arr))

    add("starts_red", np.array([2, 9, 0], dtype=np.int64))
    add("ends_red", np.array([3, 10, 10], dtype=np.int64))
    for r in range(2, 7):
        add(f"starts_gray_{r}", np.array([5, r, 0], dtype=np.int64))
        add(f"ends_gray_{r}", np.array([6, r + 1, 10], dtype=np.int64))
    add("axes_chw", np.array([1, 2, 3], dtype=np.int64))
    add("shift_idx", np.array([9, 0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64))
    add("zero_row", np.zeros((1, 1, 1, 10), dtype=np.uint8))
    add("xzp", np.array(1, dtype=np.uint8))

    w = np.zeros((10, 1, 2, 1), dtype=np.int8)
    w[0, 0, :, 0] = [-1, -1]  # background: red=0 and gray=0 -> positive
    w[2, 0, :, 0] = [1, 0]     # red priority, including red over gray
    w[5, 0, :, 0] = [-1, 1]    # gray only
    add("Wout", w)

    def n(op, ins, outs, **kw):
        nodes.append(helper.make_node(op, ins, outs, **kw))

    n("Slice", ["input", "starts_red", "ends_red", "axes_chw"], ["red_f"])
    n("Cast", ["red_f"], ["active_9_u1"], to=TP.UINT8)
    n("Add", ["active_9_u1", "active_9_u1"], ["active_9"])
    prev = "active_9"
    outs = []
    for r in range(6, 1, -1):
        n("Slice", ["input", f"starts_gray_{r}", f"ends_gray_{r}", "axes_chw"], [f"gray_{r}_f"])
        n("Cast", [f"gray_{r}_f"], [f"blocker_{r}_u1"], to=TP.UINT8)
        n("Add", [f"blocker_{r}_u1", f"blocker_{r}_u1"], [f"blocker_{r}"])
        n("BitwiseAnd", [prev, f"blocker_{r}"], [f"hit_{r}"])
        n("Gather", [f"hit_{r}", "shift_idx"], [f"shift_{r}"], axis=3)
        n("BitwiseOr", [prev, f"shift_{r}"], [f"out_{r}"])
        n("BitwiseXor", [f"out_{r}", f"hit_{r}"], [f"active_{r}"])
        outs.append((r, f"out_{r}", f"active_{r}"))
        prev = f"active_{r}"

    act2 = outs[-1][2]
    red_rows = [act2, act2]
    red_rows.append(act2)
    for _, out_r, _ in reversed(outs):
        red_rows.append(out_r)
    red_rows.extend(["active_9", "active_9"])
    gray_rows = [
        "zero_row",
        "zero_row",
        "blocker_2",
        "blocker_3",
        "blocker_4",
        "blocker_5",
        "blocker_6",
        "zero_row",
        "zero_row",
        "zero_row",
    ]
    packed_rows = []
    for red_row, gray_row in zip(red_rows, gray_rows):
        packed_rows.extend([red_row, gray_row])
    n("Concat", packed_rows, ["state_interleaved"], axis=2)
    n(
        "ConvInteger",
        ["state_interleaved", "Wout", "xzp"],
        ["output"],
        kernel_shape=[2, 1],
        pads=[0, 0, 40, 20],
        strides=[2, 1],
    )

    graph = helper.make_graph(
        nodes,
        "task345",
        [helper.make_tensor_value_info("input", TP.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TP.INT32, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = 10
    del model.graph.value_info[:]
    return model


if __name__ == "__main__":
    m = build()
    onnx.checker.check_model(m)
    onnx.save(m, OUT)
    print(f"saved {OUT}")
