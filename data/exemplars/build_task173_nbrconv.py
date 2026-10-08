"""task173 — conv25 + 8-ch neighbor Conv replaces pad27 + 8 shift slices."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

OUT = Path("candidates/task173_nbrconv.onnx")

# neighbor order: hl, hr, vu, vd, nw, ne, sw, se
NB_TAPS = [
    (1, 0),
    (1, 2),
    (0, 1),
    (2, 1),
    (0, 0),
    (0, 2),
    (2, 0),
    (2, 2),
]


def build() -> None:
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    out = helper.make_tensor_value_info("output", TensorProto.BOOL, [1, 10, 30, 30])
    nodes: list = []
    inits: list = []

    def init(name: str, dtype: int, value) -> None:
        arr = np.asarray(value)
        inits.append(helper.make_tensor(name, dtype, arr.shape, arr.flatten().tolist()))

    def node(op: str, inputs: list[str], outputs: list[str], **attrs) -> None:
        nodes.append(helper.make_node(op, inputs, outputs, **attrs))

    w = np.zeros((1, 10, 6, 6), dtype=np.float32)
    w[0, :, 0, 0] = np.arange(1, 11, dtype=np.float32)
    init("colors_w25", TensorProto.FLOAT, w)
    init("palette_u8", TensorProto.UINT8, np.arange(1, 11, dtype=np.uint8).reshape(1, 10, 1, 1))
    init("zeros11_u8", TensorProto.UINT8, np.zeros(11, dtype=np.uint8))
    init("iota11_u8", TensorProto.UINT8, np.arange(11, dtype=np.uint8))
    init("val_0_u8", TensorProto.UINT8, [0])
    init("val_1_u8", TensorProto.UINT8, [1])
    init("shape_11_1", TensorProto.INT64, [11, 1])
    init("axis4", TensorProto.INT64, [4])
    init("axis1", TensorProto.INT64, [1])
    init("axes_ch", TensorProto.INT64, [1])

    nkw = np.zeros((8, 1, 3, 3), dtype=np.float32)
    for ch, (r, c) in enumerate(NB_TAPS):
        nkw[ch, 0, r, c] = 1.0
    init("nbr_w", TensorProto.FLOAT, nkw)

    node("Conv", ["input", "colors_w25"], ["label25_f32"])
    node("Cast", ["label25_f32"], ["label25"], to=TensorProto.UINT8)
    node("Equal", ["label25", "val_1_u8"], ["iz"])
    node("Cast", ["label25"], ["label25_cf32"], to=TensorProto.FLOAT)
    node("Conv", ["label25_cf32", "nbr_w"], ["nbr_f32"], auto_pad="SAME_UPPER")
    node("Cast", ["nbr_f32"], ["nbr_u8"], to=TensorProto.UINT8)

    def nbr_ch(i: int, name: str) -> None:
        init(f"c_{name}", TensorProto.INT64, [i])
        init(f"c_{name}_e", TensorProto.INT64, [i + 1])
        node("Slice", ["nbr_u8", f"c_{name}", f"c_{name}_e", "axes_ch"], [name])

    nbr_ch(0, "hl")
    nbr_ch(1, "hr")
    nbr_ch(2, "vu")
    nbr_ch(3, "vd")
    nbr_ch(4, "nw")
    nbr_ch(5, "ne")
    nbr_ch(6, "sw")
    nbr_ch(7, "se")

    node("Unsqueeze", ["label25", "axis4"], ["label25_idx_u8"])
    node("Cast", ["label25_idx_u8"], ["idx64"], to=TensorProto.INT64)
    node("Cast", ["label25"], ["idx32"], to=TensorProto.INT32)

    node("Equal", ["hl", "hr"], ["eq_H"])
    node("Where", ["eq_H", "hl", "val_0_u8"], ["flankv_H"])
    node("Equal", ["vu", "vd"], ["eq_V"])
    node("Where", ["eq_V", "vu", "val_0_u8"], ["flankv_V"])
    node("Equal", ["nw", "ne"], ["eq_X1"])
    node("Equal", ["nw", "sw"], ["eq_X2"])
    node("Equal", ["nw", "se"], ["eq_X3"])
    node("And", ["eq_X1", "eq_X2"], ["and_X1"])
    node("And", ["and_X1", "eq_X3"], ["eq_X"])
    node("Where", ["eq_X", "nw", "val_0_u8"], ["flankv_X"])

    def assoc(axis: str, flankv: str) -> None:
        node("Where", ["iz", "val_0_u8", flankv], [f"masked_{axis}"])
        node("ScatterND", ["zeros11_u8", "idx64", f"masked_{axis}"], [f"T2_raw_{axis}"], reduction="max")
        node("Greater", [f"T2_raw_{axis}", "val_1_u8"], [f"T2_ok_{axis}"])
        node("Where", [f"T2_ok_{axis}", f"T2_raw_{axis}", "val_0_u8"], [f"T2_{axis}"])
        node("Greater", [f"T2_{axis}", "val_0_u8"], [f"T2_{axis}_nz"])
        node("Where", [f"T2_{axis}_nz", "iota11_u8", "val_0_u8"], [f"upd_f_{axis}"])
        node("Reshape", [f"T2_{axis}", "shape_11_1"], [f"T2_{axis}_11_1"])
        node("Cast", [f"T2_{axis}_11_1"], [f"idx_f_{axis}"], to=TensorProto.INT64)
        node("ScatterND", ["zeros11_u8", f"idx_f_{axis}", f"upd_f_{axis}"], [f"f_{axis}"], reduction="max")

    assoc("H", "flankv_H")
    assoc("V", "flankv_V")
    assoc("X", "flankv_X")

    for axis in ("H", "V", "X"):
        node("Gather", [f"f_{axis}", "idx32"], [f"pf_{axis}"])

    node("MaxPool", ["pf_H"], ["c_H"], kernel_shape=[1, 3], pads=[0, 1, 0, 1], strides=[1, 1])
    node("MaxPool", ["pf_V"], ["c_V"], kernel_shape=[3, 1], pads=[1, 0, 1, 0], strides=[1, 1])
    node("MaxPool", ["pf_X"], ["c_X"], kernel_shape=[2, 2], pads=[1, 1, 1, 1], strides=[1, 1], dilations=[2, 2])
    node("Where", ["eq_H", "c_H", "val_0_u8"], ["raw_cH"])
    node("Where", ["eq_V", "c_V", "val_0_u8"], ["raw_cV"])
    node("Where", ["eq_X", "c_X", "val_0_u8"], ["raw_cX"])

    for axis in ("H", "V", "X"):
        node("Gather", [f"T2_{axis}", "idx32"], [f"pa_{axis}"])

    node("MaxPool", ["pa_H"], ["raw_arm_H"], kernel_shape=[1, 3], pads=[0, 1, 0, 1], strides=[1, 1])
    node("MaxPool", ["pa_V"], ["raw_arm_V"], kernel_shape=[3, 1], pads=[1, 0, 1, 0], strides=[1, 1])
    node("MaxPool", ["pa_X"], ["raw_arm_X"], kernel_shape=[2, 2], pads=[1, 1, 1, 1], strides=[1, 1], dilations=[2, 2])
    node("Max", ["raw_cH", "raw_cV", "raw_cX", "raw_arm_H", "raw_arm_V", "raw_arm_X"], ["raw_fills"])
    node("Max", ["raw_fills", "val_1_u8"], ["fills1"])
    node("Where", ["iz", "fills1", "label25"], ["out25"])

    init("pads_to_30", TensorProto.INT64, [0, 0, 0, 0, 0, 0, 5, 5])
    node("Pad", ["out25", "pads_to_30", "val_0_u8"], ["acc30"])
    node("Equal", ["acc30", "palette_u8"], ["output"])

    graph = helper.make_graph(nodes, "task173_nbrconv", [inp], [out], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.save(model, OUT)
    print(OUT)


if __name__ == "__main__":
    build()