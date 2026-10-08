#!/usr/bin/env python3
"""task062 att15 — red-adjacency core + box-plane ConvInteger terminal."""
from __future__ import annotations

import sys

import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper as H
from onnx import numpy_helper as NH

sys.path.insert(0, "../../runner")
from ngolf import relower_onehot_plane  # noqa: E402

OPSET = 17
IR_VERSION = 10
OUT = "task062.onnx"


def map8(axis: int, high: bool, active: bool) -> np.ndarray:
    out = np.arange(8, dtype=np.int32)
    if not active:
        return out
    base = 2 * axis + (1 if high else -1)
    for j in range(8):
        visible = j > axis if high else j < axis
        if not visible:
            src = base - j
            out[j] = src if 0 <= src < 8 else 8
    return out


def map_lut() -> np.ndarray:
    rows: list[np.ndarray] = [map8(0, True, False)]
    for axis in range(8):
        rows.append(map8(axis, True, True))
    for axis in range(8):
        rows.append(map8(axis, False, True))
    return np.stack(rows).astype(np.int32)


def build() -> onnx.ModelProto:
    inits = []
    nodes = []

    def init(name: str, arr) -> str:
        inits.append(NH.from_array(np.asarray(arr), name))
        return name

    def node(op: str, ins: list[str], out: str, **attrs) -> str:
        nodes.append(H.make_node(op, ins, [out], **attrs))
        return out

    def relower(channel: int, name: str) -> str:
        ns, its, _shape = relower_onehot_plane(
            "input",
            name,
            channel=channel,
            crop=((0, 1), (channel, channel + 1), (1, 9), (1, 9)),
            dtype="u8",
            starts_name=f"{name}_starts",
            ends_name=f"{name}_ends",
        )
        nodes.extend(ns)
        for n, arr in its:
            init(n, arr)
        return name

    black_u8 = relower(0, "black_u8")
    red_u8 = relower(2, "red_u8")

    init("one_u8", np.array(1, np.uint8))
    init("zero_i32", np.array(0, np.int32))
    init("one_i32", np.array(1, np.int32))
    init("seven_i32", np.array(7, np.int32))
    init("nine_i32", np.array(9, np.int32))
    init("eight_i32", np.array(8, np.int32))
    init("shape_1_i64", np.array([1], np.int64))
    init("weight_shape_i64", np.array([10, 1, 1, 1], np.int64))
    init("squeeze3_axes", np.array([0, 1, 2], np.int64))
    init("paint_allow_i8", np.array([0, 1, 0, 0, 1, 1, 1, 1, 1, 1], np.int8))
    init("green_neg_i8", np.array([0, 0, 0, -1, 0, 0, 0, 0, 0, 0], np.int8))
    init("box_w_i8", np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], np.int8).reshape(10, 1, 1, 1))
    init("map_lut_i32", map_lut())
    init("pad9_pads", np.array([0, 0, 0, 0, 0, 0, 1, 1], np.int64))
    init("pad10_pads", np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64))
    init("sem2_channel_pads", np.array([0, 0, 0, 0, 0, 1, 0, 0], np.int64))

    node("Sub", ["one_u8", black_u8], "fg_u8")

    # One non-red, non-green paint color is present in every generated input.
    node("ReduceMax", ["input"], "present_f", axes=[0, 2, 3], keepdims=0)
    node("Cast", ["present_f"], "present_i8", to=TP.INT8)
    node("Mul", ["present_i8", "paint_allow_i8"], "paint10_i8")
    node("Add", ["paint10_i8", "green_neg_i8"], "obj_w_i8")
    node("Reshape", ["obj_w_i8", "weight_shape_i64"], "obj_w_i8_4d")
    node("Concat", ["obj_w_i8_4d", "box_w_i8"], "conv_w", axis=1)

    node("Pad", ["fg_u8", "pad9_pads"], "obj_grid9", mode="constant")

    # Red axis: first occupied red row/column. The inactive axis is ignored by keys.
    node("ReduceMax", [red_u8], "red_rows", axes=[3], keepdims=1)
    node("ReduceMax", [red_u8], "red_cols", axes=[2], keepdims=1)
    node("ArgMax", ["red_rows"], "axis_row_i64_k", axis=2, keepdims=0)
    node("Squeeze", ["axis_row_i64_k", "squeeze3_axes"], "axis_row_i64")
    node("Cast", ["axis_row_i64"], "axis_row", to=TP.INT32)
    node("ArgMax", ["red_cols"], "axis_col_i64_k", axis=3, keepdims=0)
    node("Squeeze", ["axis_col_i64_k", "squeeze3_axes"], "axis_col_i64")
    node("Cast", ["axis_col_i64"], "axis_col", to=TP.INT32)

    # Neighbor probes use colored-only cells masked by the red line coordinates.
    node("Add", ["axis_col", "one_i32"], "axis_col_p1_raw")
    node("Min", ["axis_col_p1_raw", "seven_i32"], "axis_col_p1")
    node("Sub", ["axis_col", "one_i32"], "axis_col_m1_raw")
    node("Max", ["axis_col_m1_raw", "zero_i32"], "axis_col_m1")
    node("Add", ["axis_row", "one_i32"], "axis_row_p1_raw")
    node("Min", ["axis_row_p1_raw", "seven_i32"], "axis_row_p1")

    node("Reshape", ["axis_col_p1", "shape_1_i64"], "axis_col_p1_v")
    node("Reshape", ["axis_col_m1", "shape_1_i64"], "axis_col_m1_v")
    node("Reshape", ["axis_row_p1", "shape_1_i64"], "axis_row_p1_v")

    node("Gather", ["fg_u8", "axis_col_p1_v"], "right_col_fg", axis=3)
    node("Gather", [red_u8, "axis_col_p1_v"], "right_col_red", axis=3)
    node("Sub", ["right_col_fg", "right_col_red"], "right_col")
    node("Mul", ["right_col", "red_rows"], "right_on_red_rows")
    node("ReduceMax", ["right_on_red_rows"], "right_any", axes=[0, 1, 2, 3], keepdims=0)
    node("Gather", ["fg_u8", "axis_col_m1_v"], "left_col_fg", axis=3)
    node("Gather", [red_u8, "axis_col_m1_v"], "left_col_red", axis=3)
    node("Sub", ["left_col_fg", "left_col_red"], "left_col")
    node("Mul", ["left_col", "red_rows"], "left_on_red_rows")
    node("ReduceMax", ["left_on_red_rows"], "left_any", axes=[0, 1, 2, 3], keepdims=0)

    node("Gather", ["fg_u8", "axis_row_p1_v"], "down_row_fg", axis=2)
    node("Gather", [red_u8, "axis_row_p1_v"], "down_row_red", axis=2)
    node("Sub", ["down_row_fg", "down_row_red"], "down_row")
    node("Mul", ["down_row", "red_cols"], "down_on_red_cols")
    node("ReduceMax", ["down_on_red_cols"], "down_any", axes=[0, 1, 2, 3], keepdims=0)

    node("Add", ["right_any", "left_any"], "sel_h_u8")

    # Map keys: 0 inactive identity, 1..8 active/high, 9..16 active/low.
    node("Add", ["axis_col", "nine_i32"], "axis_col_p9_key")
    node("Add", ["axis_row", "nine_i32"], "axis_row_p9_key")
    node("Cast", ["right_any"], "right_i32", to=TP.INT32)
    node("Cast", ["down_any"], "down_i32", to=TP.INT32)
    node("Cast", ["sel_h_u8"], "sel_h_i32", to=TP.INT32)
    node("Mul", ["right_i32", "eight_i32"], "right_x8")
    node("Mul", ["down_i32", "eight_i32"], "down_x8")
    node("Sub", ["axis_col_p9_key", "right_x8"], "active_col_key")
    node("Sub", ["axis_row_p9_key", "down_x8"], "active_row_key")
    node("Mul", ["sel_h_i32", "active_col_key"], "col_key")
    node("Sub", ["one_i32", "sel_h_i32"], "sel_v_i32")
    node("Mul", ["sel_v_i32", "active_row_key"], "row_key")
    node("Gather", ["map_lut_i32", "row_key"], "row_idx10", axis=0)
    node("Gather", ["map_lut_i32", "col_key"], "col_idx10", axis=0)

    node("Gather", ["obj_grid9", "row_idx10"], "row_gather", axis=2)
    node("Gather", ["row_gather", "col_idx10"], "obj8_mapped", axis=3)
    node("Pad", ["obj8_mapped", "pad10_pads"], "obj10_u8", mode="constant")
    node("Pad", ["obj10_u8", "sem2_channel_pads", "one_u8"], "sem2", mode="constant")
    node("ConvInteger", ["sem2", "conv_w"], "output", pads=[0, 0, 20, 20], strides=[1, 1])

    graph = H.make_graph(
        nodes,
        "task062_att15_convinteger_boxpad",
        [H.make_tensor_value_info("input", TP.FLOAT, [1, 10, 30, 30])],
        [H.make_tensor_value_info("output", TP.INT32, [1, 10, 30, 30])],
        inits,
    )
    model = H.make_model(graph, opset_imports=[H.make_operatorsetid("", OPSET)])
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


if __name__ == "__main__":
    onnx.save(build(), OUT)
    print(f"saved {OUT}")
