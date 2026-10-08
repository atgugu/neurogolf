#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as TP, helper, numpy_helper

N = 13
OUT = Path(__file__).resolve().with_name("task089.onnx")


def init(inits: list, name: str, value, dtype) -> None:
    inits.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), name))


def wrap_positive(arr: np.ndarray) -> np.ndarray:
    return np.where(arr > 0, arr - (N * N), arr).astype(np.int32)


def wrap_axis(arr: np.ndarray) -> np.ndarray:
    return np.where(arr > 0, arr - N, arr).astype(np.int32)


def oriented_offsets(*, mirror_red: bool) -> np.ndarray:
    vals = []
    for kr in range(-2, 3):
        for kc in range(-2, 3):
            if mirror_red:
                vals.append((-kr) * N + kc)
            else:
                vals.append((-kr) * N + (-kc))
    return wrap_positive(np.asarray(vals, dtype=np.int32)).reshape(1, 1, 5, 5)


def source_patch_nodes(
    nodes: list,
    *,
    prefix: str,
    color13: str,
    anchor: str,
    neighbor: str,
    row_offset: str,
    col_offset: str,
) -> str:
    p = prefix
    nodes += [
        helper.make_node("Where", [anchor, neighbor, "zero_u8"], [f"{p}_src_u8"]),
        helper.make_node("ReduceMax", [f"{p}_src_u8"], [f"{p}_src_rows"], axes=[0, 1, 3], keepdims=0),
        helper.make_node("ReduceMax", [f"{p}_src_u8"], [f"{p}_src_cols"], axes=[0, 1, 2], keepdims=0),
        helper.make_node("ArgMax", [f"{p}_src_rows"], [f"{p}_src_row_i64"], axis=0, keepdims=1),
        helper.make_node("ArgMax", [f"{p}_src_cols"], [f"{p}_src_col_i64"], axis=0, keepdims=1),
        helper.make_node("Cast", [f"{p}_src_row_i64"], [f"{p}_src_row"], to=TP.INT32),
        helper.make_node("Cast", [f"{p}_src_col_i64"], [f"{p}_src_col"], to=TP.INT32),
        helper.make_node("ReduceMax", [f"{p}_src_u8"], [f"{p}_payload"], axes=[0, 1, 2, 3], keepdims=0),
        helper.make_node("Add", [f"{p}_src_row", row_offset], [f"{p}_local_rows"]),
        helper.make_node("Add", [f"{p}_src_col", col_offset], [f"{p}_local_cols"]),
        helper.make_node("Gather", [color13, f"{p}_local_rows"], [f"{p}_local_rowpick"], axis=2),
        helper.make_node("Gather", [f"{p}_local_rowpick", f"{p}_local_cols"], [f"{p}_local"], axis=3),
        helper.make_node("Equal", [f"{p}_local", f"{p}_payload"], [f"{p}_match"]),
        helper.make_node("Where", [f"{p}_match", f"{p}_payload", "zero_u8"], [f"{p}_upd"]),
    ]
    return f"{p}_upd"


def weighted_source_patch_nodes(
    nodes: list,
    *,
    prefix: str,
    anchor: str,
    neighbor: str,
    payload_fg: str,
    mirror_red: bool,
) -> str:
    """Build the oriented 5x5 body kernel for one marker color.

    The source plane contains the body color B at the unique source marker.
    QLinearConv(source_B, payload_fg_13x13) emits B * local_payload in the
    reversed Conv kernel order.  Comparing against B*B filters unrelated
    nearby payload colors without the row/column ArgMax gather chain.
    """
    p = prefix
    raw = f"{p}_raw"
    oriented = raw
    nodes += [
        helper.make_node("Where", [anchor, neighbor, "zero_u8"], [f"{p}_src_u8"]),
        helper.make_node(
            "QLinearConv",
            [f"{p}_src_u8", "q_scale", "zero_u8", payload_fg, "q_scale", "zero_u8", "q_scale", "zero_u8"],
            [raw],
            kernel_shape=[13, 13],
            pads=[2, 2, 2, 2],
        ),
    ]
    if mirror_red:
        oriented = f"{p}_raw_m"
        nodes.append(helper.make_node("Slice", [raw, "rev5_st", "rev5_en", "rev5_ax3", "rev5_step"], [oriented]))
    nodes += [
        helper.make_node("ReduceMax", [f"{p}_src_u8"], [f"{p}_payload"], axes=[0, 1, 2, 3], keepdims=0),
        helper.make_node("Mul", [f"{p}_payload", f"{p}_payload"], [f"{p}_payload_sq"]),
        helper.make_node("Equal", [oriented, f"{p}_payload_sq"], [f"{p}_match"]),
        helper.make_node("Where", [f"{p}_match", f"{p}_payload", "zero_u8"], [f"{p}_upd"]),
    ]
    return f"{p}_upd"


def build_model(*, terminal_onehot: bool = False) -> onnx.ModelProto:
    nodes: list = []
    inits: list = []

    color_w = np.zeros((1, 10, 2, 2), dtype=np.float32)
    for color in range(10):
        color_w[0, color, 0, 0] = color

    for name, value, dtype in [
        ("color_w", color_w, np.float32),
        ("u8_2", 2, np.uint8),
        ("u8_3", 3, np.uint8),
        ("zero_u8", 0, np.uint8),
        ("sentinel10_u8", 10, np.uint8),
        ("rev5_st", [4], np.int64),
        ("rev5_en", [-9223372036854775807], np.int64),
        ("rev5_ax3", [3], np.int64),
        ("rev5_step", [-1], np.int64),
        ("pad_out4", [0, 0, 0, 0, 0, 0, 30 - N, 30 - N], np.int64),
        ("color_bank", np.arange(10, dtype=np.uint8).reshape(1, 10, 1, 1), np.uint8),
        ("q_scale", 1.0, np.float32),
    ]:
        init(inits, name, value, dtype)

    nodes += [
        helper.make_node("Conv", ["input", "color_w"], ["idx_f32"], kernel_shape=[2, 2], dilations=[17, 17]),
        helper.make_node("Cast", ["idx_f32"], ["color13"], to=TP.UINT8),
        helper.make_node("Equal", ["color13", "u8_2"], ["red_a"]),
        helper.make_node("Equal", ["color13", "u8_3"], ["green_a"]),
        helper.make_node("Or", ["red_a", "green_a"], ["marker_b"]),
        helper.make_node("Where", ["marker_b", "zero_u8", "color13"], ["payload_fg"]),
        helper.make_node("MaxPool", ["payload_fg"], ["neighbor4"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]),
    ]

    red_upd = weighted_source_patch_nodes(
        nodes,
        prefix="c2",
        anchor="red_a",
        neighbor="neighbor4",
        payload_fg="payload_fg",
        mirror_red=True,
    )
    green_upd = weighted_source_patch_nodes(
        nodes,
        prefix="c3",
        anchor="green_a",
        neighbor="neighbor4",
        payload_fg="payload_fg",
        mirror_red=False,
    )

    nodes += [
        helper.make_node("Equal", ["neighbor4", "zero_u8"], ["neighbor_zero"]),
        helper.make_node("And", ["red_a", "neighbor_zero"], ["loneR"]),
        helper.make_node("Cast", ["loneR"], ["loneR_u8"], to=TP.UINT8),
        helper.make_node("Cast", ["green_a"], ["green_tgt_u8"], to=TP.UINT8),
        helper.make_node(
            "QLinearConv",
            ["loneR_u8", "q_scale", "zero_u8", red_upd, "q_scale", "zero_u8", "q_scale", "zero_u8"],
            ["red_stamped"],
            kernel_shape=[5, 5],
            pads=[2, 2, 2, 2],
        ),
        helper.make_node(
            "QLinearConv",
            ["green_tgt_u8", "q_scale", "zero_u8", green_upd, "q_scale", "zero_u8", "q_scale", "zero_u8"],
            ["green_stamped"],
            kernel_shape=[5, 5],
            pads=[2, 2, 2, 2],
        ),
        helper.make_node("Max", ["color13", "red_stamped", "green_stamped"], ["out_u8"]),
    ]

    if terminal_onehot:
        nodes += [
            helper.make_node("Equal", ["out_u8", "color_bank"], ["onehot13"]),
            helper.make_node("Pad", ["onehot13", "pad_out4"], ["output"], mode="constant"),
        ]
        graph_name = "task089_u8_terminal_onehot_pad"
    else:
        nodes += [
            helper.make_node("Pad", ["out_u8", "pad_out4", "sentinel10_u8"], ["color30"], mode="constant"),
            helper.make_node("Equal", ["color30", "color_bank"], ["output"]),
        ]
        graph_name = "task089_direct_kernel_qconv"

    graph = helper.make_graph(
        nodes,
        graph_name,
        [helper.make_tensor_value_info("input", TP.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TP.BOOL, [1, 10, 30, 30])],
        inits,
    )
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 16)])
    onnx.checker.check_model(model, full_check=True)
    return model


def build_u8_terminal() -> onnx.ModelProto:
    return build_model(terminal_onehot=True)


def build_grouped9() -> onnx.ModelProto:
    # The audited 9x9 crop misses visible train/test cells; keep the corrected
    # weighted-patch grouped-conv family on the full 13x13 certificate window.
    return build_model(terminal_onehot=False)


def build() -> onnx.ModelProto:
    return build_grouped9()


def main() -> None:
    model = build()
    onnx.save(model, OUT)
    print(f"saved {OUT} bytes={OUT.stat().st_size} nodes={len(model.graph.node)}")


if __name__ == "__main__":
    main()
