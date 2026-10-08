from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


OUT = Path(__file__).with_name("task270.onnx")
IR_VERSION = 10
OPSET = 18

nodes: list[onnx.NodeProto] = []
inits: list[onnx.TensorProto] = []
seen: set[str] = set()


def const(name: str, arr) -> str:
    if name not in seen:
        seen.add(name)
        inits.append(numpy_helper.from_array(np.asarray(arr), name=name))
    return name


def add(op: str, inputs: list[str], output: str, **attrs) -> str:
    nodes.append(helper.make_node(op, inputs, [output], name=output, **attrs))
    return output


inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
out = helper.make_tensor_value_info("output", TensorProto.INT64, [1, 10, 30, 30])

const("chan_w", np.array([0, 1, 2, 12, 0, 0, 0, 96, 0, 0], np.float32))
const("axis1_vec", np.array([1], np.int64))
const("two_i64", np.array([[2]], np.int64))
const("one_u64", np.array([[1]], np.uint64))
const("zero_i64", np.array([[0]], np.int64))

const("marker_starts", np.array([0], np.int64))
const("marker_ends", np.array([15], np.int64))
const("mask1", np.array(1, np.uint8))
const("mask2", np.array(2, np.uint8))
const("mask3", np.array(8, np.uint8))
const("mask7", np.array(64, np.uint8))

base_state = np.zeros((30, 1), dtype=np.int64)
base_state[:15, 0] = np.int64((1 << 15) - 1)
const("base_state", base_state)

col_mask = np.zeros((1, 10, 1, 30), dtype=np.int64)
for c in range(15):
    col_mask[0, 0, 0, c] = np.int64(1 << c)
for c in range(2, 13):
    col_mask[0, 1, 0, c] = np.int64(1 << (c + 13))
    col_mask[0, 2, 0, c] = np.int64(1 << (c + 24))
for c in range(1, 14):
    col_mask[0, 3, 0, c] = np.int64(1 << (c + 36))
    col_mask[0, 7, 0, c] = np.int64(1 << (c + 49))
const("col_mask", col_mask)

for offset in (13, 24, 36, 49):
    const(f"mul{offset}", np.array([[(np.int64(1) << np.int64(offset)) - np.int64(1)]], dtype=np.int64))

row_pack = add("Einsum", ["input", "chan_w"], "row_pack", equation="bchw,c->bh")
col_pack = add("Einsum", ["input", "chan_w"], "col_pack", equation="bchw,c->bw")
row_pack_u = add("Cast", [row_pack], "row_pack_u", to=TensorProto.UINT8)
col_pack_u = add("Cast", [col_pack], "col_pack_u", to=TensorProto.UINT8)

row_mark = add("Slice", [row_pack_u, "marker_starts", "marker_ends", "axis1_vec"], "row_marker_pack")
col_mark = add("Slice", [col_pack_u, "marker_starts", "marker_ends", "axis1_vec"], "col_marker_pack")

row1_u = add("BitwiseAnd", [row_mark, "mask1"], "row1_i")
row2_u = add("BitwiseAnd", [row_mark, "mask2"], "row2_i")
row3_u = add("BitwiseAnd", [row_mark, "mask3"], "row3_i")
row7_u = add("BitwiseAnd", [row_mark, "mask7"], "row7_i")
col1_u = add("BitwiseAnd", [col_mark, "mask1"], "col1_i")
col2_u = add("BitwiseAnd", [col_mark, "mask2"], "col2_i")
col3_u = add("BitwiseAnd", [col_mark, "mask3"], "col3_i")
col7_u = add("BitwiseAnd", [col_mark, "mask7"], "col7_i")


def center_index(core: str, name: str) -> str:
    return add("ArgMax", [core], name, axis=1, keepdims=1)


cr1 = center_index(row1_u, "cr1")
cc1 = center_index(col1_u, "cc1")
cr2 = center_index(row2_u, "cr2")
cc2 = center_index(col2_u, "cc2")


def binary_occ(src: str, name: str) -> str:
    return add("Min", [src, "mask1"], name)


def direction_flags(
    vec_r_u: str, vec_c_u: str, cr: str, cc: str, tag: str
) -> tuple[str, str, str, str]:
    occ_r = vec_r_u
    occ_c = vec_c_u
    any_r_u = add("ReduceMax", [occ_r, "axis1_vec"], f"row_any_u{tag}", keepdims=1)
    any_c_u = add("ReduceMax", [occ_c, "axis1_vec"], f"col_any_u{tag}", keepdims=1)
    any_r = add("Cast", [any_r_u], f"row_any{tag}", to=TensorProto.BOOL)
    any_c = add("Cast", [any_c_u], f"col_any{tag}", to=TensorProto.BOOL)

    min_r = add("ArgMax", [occ_r], f"row_min{tag}", axis=1, keepdims=1)
    max_r = add("ArgMax", [occ_r], f"row_max{tag}", axis=1, keepdims=1, select_last_index=1)
    min_c = add("ArgMax", [occ_c], f"col_min{tag}", axis=1, keepdims=1)
    max_c = add("ArgMax", [occ_c], f"col_max{tag}", axis=1, keepdims=1, select_last_index=1)

    up_b = add("And", [any_r, add("Less", [min_r, cr], f"up_raw{tag}")], f"up_b{tag}")
    down_b = add("And", [any_r, add("Greater", [max_r, cr], f"down_raw{tag}")], f"down_b{tag}")
    left_b = add("And", [any_c, add("Less", [min_c, cc], f"left_raw{tag}")], f"left_b{tag}")
    right_b = add("And", [any_c, add("Greater", [max_c, cc], f"right_raw{tag}")], f"right_b{tag}")
    return up_b, down_b, left_b, right_b


up3, down3, left3, right3 = direction_flags(row3_u, col3_u, cr2, cc2, "3")
up7, down7, left7, right7 = direction_flags(row7_u, col7_u, cr1, cc1, "7")


def sub1(x: str, name: str) -> str:
    return add("Sub", [x, "axis1_vec"], name)


def add1(x: str, name: str) -> str:
    return add("Add", [x, "axis1_vec"], name)


cr1m, cr1p = sub1(cr1, "cr1m"), add1(cr1, "cr1p")
cr2m, cr2p = sub1(cr2, "cr2m"), add1(cr2, "cr2p")


def bg_bit(col_idx: str, name: str) -> str:
    sh = add("Cast", [col_idx], f"{name}_sh_u64", to=TensorProto.UINT64)
    return add("BitShift", ["one_u64", sh], name, direction="LEFT")


def shift_bit(bit: str, offset: int, name: str) -> str:
    const(f"{name}_off", np.array([[offset]], dtype=np.uint64))
    return add("BitShift", [bit, f"{name}_off"], name, direction="LEFT")


def delta_from_bg(bg: str, color_offset: int, name: str) -> str:
    return add("Mul", [bg, f"mul{color_offset}"], name)


def cast_i64(x: str, name: str) -> str:
    return add("Cast", [x], name, to=TensorProto.INT64)


c1_bg = bg_bit(cc1, "c1_bg")
c1_bg_i = cast_i64(c1_bg, "c1_bg_i")
c1l_bg = add("Div", [c1_bg_i, "two_i64"], "c1l_bg")
c1r_bg = add("Add", [c1_bg_i, c1_bg_i], "c1r_bg")
c2_bg = bg_bit(cc2, "c2_bg")
c2_bg_i = cast_i64(c2_bg, "c2_bg_i")
c2l_bg = add("Div", [c2_bg_i, "two_i64"], "c2l_bg")
c2r_bg = add("Add", [c2_bg_i, c2_bg_i], "c2r_bg")

d1 = delta_from_bg(c1_bg_i, 13, "d1")
d2 = delta_from_bg(c2_bg_i, 24, "d2")
d3m = delta_from_bg(c2_bg_i, 36, "d3m")
d3l = delta_from_bg(c2l_bg, 36, "d3l")
d3r = delta_from_bg(c2r_bg, 36, "d3r")
d7m = delta_from_bg(c1_bg_i, 49, "d7m")
d7l = delta_from_bg(c1l_bg, 49, "d7l")
d7r = delta_from_bg(c1r_bg, 49, "d7r")


def gated(delta: str, flag: str | None, name: str) -> str:
    val = delta
    if flag is not None:
        val = add("Where", [flag, delta, "zero_i64"], f"{name}_gated")
    return val


p3u = gated(d3m, up3, "p3u")
p3d = gated(d3m, down3, "p3d")
p3l = gated(d3l, left3, "p3l")
p3r = gated(d3r, right3, "p3r")
p7u = gated(d7m, up7, "p7u")
p7d = gated(d7m, down7, "p7d")
p7l = gated(d7l, left7, "p7l")
p7r = gated(d7r, right7, "p7r")

d2_lr = add("Add", [add("Add", [d2, p3l], "d2_plus_l"), p3r], "d2_with_lr")
d1_lr = add("Add", [add("Add", [d1, p7l], "d1_plus_l"), p7r], "d1_with_lr")

updates = [d1_lr, d2_lr, p3u, p3d, p7u, p7d]
rows = [cr1, cr2, cr2m, cr2p, cr1m, cr1p]

updates_t = add("Concat", updates, "updates", axis=0)
rows_t = add("Concat", rows, "rows", axis=0)
state = add("ScatterND", ["base_state", rows_t, updates_t], "state", reduction="add")
add("BitwiseAnd", [state, "col_mask"], "output")

graph = helper.make_graph(nodes, "task270_i64_rowword_scatter", [inp], [out], inits)
model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", OPSET)])
onnx.checker.check_model(model, full_check=True)
model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
onnx.save(model, OUT)
print(OUT)
