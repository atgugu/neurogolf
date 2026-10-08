from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def init(name, arr):
    return numpy_helper.from_array(np.asarray(arr), name=name)


def plus7(radius: int) -> np.ndarray:
    k = np.zeros((1, 1, 7, 7), dtype=np.uint8)
    for d in range(-radius, radius + 1):
        k[0, 0, 3, 3 + d] = 1
        k[0, 0, 3 + d, 3] = 1
    return k


def build() -> onnx.ModelProto:
    # Pin-FAITHFUL (gray-neutral) rebuild of the task118 inverse rule.
    #   code:  red(ch2)->65, gray(ch5)->64, black->0   (single f32 Conv + crop to 25x28)
    #   score: QLinearConv x_zero=64 => (code-64)= red:+6 gray:0 black:-64; uint8 clamp =>
    #          6*redcount at a center IFF the whole plus footprint is black-free, else 0.
    #   L:     global radius = (max s3 > max s2).  centers = local-max & score>=L.
    #   paint: gray cells covered by a radius-L plus stamp -> cyan(8).
    # Score/center tensors live on a shifted 21x24 frame (rows 2..22, cols 1..24) to match
    # the pin's graded edge behavior while keeping the compact tensor area.
    inits = [
        init("W1", _w1()),
        init("sc", np.array(1.0, np.float32)),
        init("ys65", np.array(1.0 / 65.0, np.float32)),
        init("z64", np.array(64, np.uint8)),
        init("z0", np.array(0, np.uint8)),
        init("K3", plus7(3)),
        init("K2", plus7(2)),
        # thresholds are 6*radius because red decodes to +6 under x_zero=64 coding
        init("t2", np.array(12, np.uint8)),
        init("t3", np.array(18, np.uint8)),
        init("pads_out", np.array([0, 0, 0, 0, 0, 0, 5, 2], np.int64)),
        init("false_b", np.array(False, np.bool_)),
        init("col8", _col8()),
    ]

    nodes = [
        helper.make_node("Conv", ["input", "W1"], ["v"], pads=[0, 0, -5, -2]),
        helper.make_node("Cast", ["v"], ["vc"], to=TensorProto.UINT8),
        # black-free red-count scores on the compact shifted frame.
        helper.make_node("QLinearConv",
                         ["vc", "sc", "z64", "K3", "sc", "z0", "sc", "z0"],
                         ["s3"], pads=[1, 2, 1, 0]),
        helper.make_node("QLinearConv",
                         ["vc", "sc", "z64", "K2", "sc", "z0", "sc", "z0"],
                         ["s2"], pads=[1, 2, 1, 0]),
        helper.make_node("ReduceMax", ["s3"], ["mx3"], axes=[2, 3], keepdims=1),
        helper.make_node("ReduceMax", ["s2"], ["mx2"], axes=[2, 3], keepdims=1),
        helper.make_node("Greater", ["mx3", "mx2"], ["L"]),
        helper.make_node("Where", ["L", "s3", "s2"], ["score"]),
        helper.make_node("Where", ["L", "t3", "t2"], ["tp1"]),
        helper.make_node("MaxPool", ["score"], ["mp"],
                         kernel_shape=[7, 7], pads=[3, 3, 3, 3], strides=[1, 1]),
        helper.make_node("Max", ["mp", "tp1"], ["mpt"]),
        helper.make_node("GreaterOrEqual", ["score", "mpt"], ["cen"]),
        helper.make_node("Cast", ["cen"], ["cenu"], to=TensorProto.UINT8),
        helper.make_node("Where", ["L", "K3", "K2"], ["sten"]),
        # stamp centers (21x24) onto the 25x28 paint frame; y_scale=1/65 => a covered
        # cell decodes to 65.  Greater(pcode, vc) paints gray(64)-covered cells but not
        # red(70)-covered ones (65>64 but 65<70), folding the whole paint tail into 1 op.
        helper.make_node("QLinearConv",
                         ["cenu", "sc", "z0", "sten", "sc", "z0", "ys65", "z0"],
                         ["pcode"], pads=[5, 4, 5, 6]),
        helper.make_node("Greater", ["pcode", "vc"], ["cyan"]),
        helper.make_node("Pad", ["cyan", "pads_out", "false_b"], ["Spad"], mode="constant"),
        helper.make_node("Where", ["Spad", "col8", "input"], ["output"]),
    ]

    graph = helper.make_graph(
        nodes, "task118_faithful_planekill",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 10
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model)
    return model


def _w1():
    w = np.zeros((1, 10, 1, 1), np.float32)
    w[0, 2, 0, 0] = 70.0   # red  -> +6 under x_zero=64
    w[0, 5, 0, 0] = 64.0   # gray -> 0
    return w


def _col8():
    c = np.zeros((1, 10, 1, 1), np.float32)
    c[0, 8, 0, 0] = 1.0
    return c


if __name__ == "__main__":
    onnx.save(build(), "task118.onnx")
    print("wrote task118.onnx")
