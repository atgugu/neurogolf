#!/usr/bin/env python3
"""task163 attempt 9: direct uint8 color-grid terminal renderer.

This keeps the verified yellow locator, then crops the selected 3x3 source
block from the free one-hot input, decodes color ids as uint8, builds the
native 11x11 lattice, and lets the final Pad produce the graph output.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = Path(__file__).resolve().parent / "task163.onnx"


def init(name: str, arr) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr), name=name)


def node(op: str, inputs: list[str], outputs: list[str], **attrs) -> onnx.NodeProto:
    return helper.make_node(op, inputs, outputs, **attrs)


def vi(name: str, dtype: int, shape: list[int]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, dtype, shape)


def build_task163_plane_kill_u8() -> onnx.ModelProto:
    yellow = np.zeros(10, dtype=np.float32)
    yellow[4] = 1.0

    block_map = np.zeros((3, 30), dtype=np.float16)
    off = np.zeros((4, 30), dtype=np.float16)
    for block_id in range(3):
        for local_pos in range(4):
            pos = block_id * 4 + local_pos
            if pos < 11:
                block_map[block_id, pos] = 1.0
                off[local_pos, pos] = 1.0

    local = np.zeros((2, 3, 4), dtype=np.float16)
    for r in range(3):
        local[0, r, r] = 1.0
        local[1, r, 3] = 1.0

    basis = np.zeros((10, 3, 3), dtype=np.float16)
    for k in range(10):
        basis[k, 0, 0] = -1.0
        basis[k, 1, 0] = np.float16(2 * k)
        basis[k, 2, 0] = np.float16(-k * k + 0.25)
    basis[0, 2, 1] = 1.0
    basis[5, 2, 2] = 1.0

    case = np.zeros((3, 2, 2, 2), dtype=np.float16)
    case[1, 0, 0, 0] = 0.25  # black base for every interior cell.
    case[2, 0, 0, 1] = 1.0
    case[2, 0, 1, 0] = 1.0
    case[2, 0, 1, 1] = 1.0  # every separator term is gray.
    case[0, 1, 0, 0] = 1.0
    case[1, 1, 0, 0] = -0.25  # selected color polynomial minus black base.

    inits = [
        init("yellow", yellow),
        init("iota30", np.arange(30, dtype=np.float32)),
        init("axes0", np.array([0], dtype=np.int64)),
        init("axes123", np.array([1, 2, 3], dtype=np.int32)),
        init("four_u8", np.array(4, dtype=np.uint8)),
        init("src_ch_start", np.array([1], dtype=np.uint8)),
        init("src_extent", np.array([9, 3, 3], dtype=np.int32)),
        init("convw", np.arange(1, 10, dtype=np.float32).reshape(1, 9, 1, 1)),
        init("idx_r", np.arange(3, dtype=np.uint8).reshape(1, 1, 3, 1)),
        init("idx_c", np.arange(3, dtype=np.uint8).reshape(1, 1, 1, 3)),
        init("one3", np.ones((1, 1, 3, 3), dtype=np.float16)),
        init("basis", basis),
        init("case", case),
        init("local", local),
        init("block", block_map),
        init("off", off),
    ]

    nodes = [
        node("Einsum", ["input", "yellow", "iota30"], ["R_f"], equation="nchw,c,h->"),
        node("Einsum", ["input", "yellow", "iota30"], ["C_f"], equation="nchw,c,w->"),
        node("Cast", ["R_f"], ["R_u8"], to=TensorProto.UINT8),
        node("Cast", ["C_f"], ["C_u8"], to=TensorProto.UINT8),
        node("Mod", ["R_u8", "four_u8"], ["R_loc"]),
        node("Mod", ["C_u8", "four_u8"], ["C_loc"]),
        node("Sub", ["R_u8", "R_loc"], ["R_src"]),
        node("Sub", ["C_u8", "C_loc"], ["C_src"]),
        node("Unsqueeze", ["R_src", "axes0"], ["R_start"]),
        node("Unsqueeze", ["C_src", "axes0"], ["C_start"]),
        node("Concat", ["src_ch_start", "R_start", "C_start"], ["src_starts_u8"], axis=0),
        node("Cast", ["src_starts_u8"], ["src_starts"], to=TensorProto.INT32),
        node("Add", ["src_starts", "src_extent"], ["src_ends"]),
        node("Slice", ["input", "src_starts", "src_ends", "axes123"], ["src_9"]),
        node("Conv", ["src_9", "convw"], ["src_f"], kernel_shape=[1, 1]),
        node("Cast", ["src_f"], ["x"], to=TensorProto.FLOAT16),
        node("Mul", ["x", "x"], ["x2"]),
        node("Concat", ["x2", "x", "one3"], ["features"], axis=1),
        node("Equal", ["R_loc", "idx_r"], ["r_hit"]),
        node("Equal", ["C_loc", "idx_c"], ["c_hit"]),
        node("And", ["r_hit", "c_hit"], ["loc_b"]),
        node("Cast", ["loc_b"], ["loc"], to=TensorProto.FLOAT16),
        node("Concat", ["one3", "loc"], ["locs"], axis=1),
        node(
            "Einsum",
            [
                "features",
                "locs",
                "basis",
                "case",
                "local",
                "local",
                "block",
                "off",
                "block",
                "off",
            ],
            ["output"],
            equation="nfrs,ndab,kfm,mduv,urp,vsq,ah,ph,bw,qw->nkhw",
        ),
    ]

    value_info = [
        vi("R_f", TensorProto.FLOAT, []),
        vi("C_f", TensorProto.FLOAT, []),
        vi("R_u8", TensorProto.UINT8, []),
        vi("C_u8", TensorProto.UINT8, []),
        vi("R_loc", TensorProto.UINT8, []),
        vi("C_loc", TensorProto.UINT8, []),
        vi("R_src", TensorProto.UINT8, []),
        vi("C_src", TensorProto.UINT8, []),
        vi("R_start", TensorProto.UINT8, [1]),
        vi("C_start", TensorProto.UINT8, [1]),
        vi("src_starts_u8", TensorProto.UINT8, [3]),
        vi("src_starts", TensorProto.INT32, [3]),
        vi("src_ends", TensorProto.INT32, [3]),
        vi("src_9", TensorProto.FLOAT, [1, 9, 3, 3]),
        vi("src_f", TensorProto.FLOAT, [1, 1, 3, 3]),
        vi("x", TensorProto.FLOAT16, [1, 1, 3, 3]),
        vi("x2", TensorProto.FLOAT16, [1, 1, 3, 3]),
        vi("features", TensorProto.FLOAT16, [1, 3, 3, 3]),
        vi("r_hit", TensorProto.BOOL, [1, 1, 3, 1]),
        vi("c_hit", TensorProto.BOOL, [1, 1, 1, 3]),
        vi("loc_b", TensorProto.BOOL, [1, 1, 3, 3]),
        vi("loc", TensorProto.FLOAT16, [1, 1, 3, 3]),
        vi("locs", TensorProto.FLOAT16, [1, 2, 3, 3]),
    ]

    graph = helper.make_graph(
        nodes,
        "task163_factorized_lattice",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 10, 30, 30])],
        inits,
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def build_task163_direct_u8_terminal() -> onnx.ModelProto:
    yellow = np.zeros(10, dtype=np.float32)
    yellow[4] = 1.0

    dest_idx_table = np.array(
        [
            [0, 1, 2, 3, 4, 4, 4, 3, 4, 4, 4],
            [4, 4, 4, 3, 0, 1, 2, 3, 4, 4, 4],
            [4, 4, 4, 3, 4, 4, 4, 3, 0, 1, 2],
        ],
        dtype=np.int32,
    )

    inits = [
        init("yellow", yellow),
        init("iota30", np.arange(30, dtype=np.float32)),
        init("axes0", np.array([0], dtype=np.int64)),
        init("axes123", np.array([1, 2, 3], dtype=np.int32)),
        init("four_u8", np.array(4, dtype=np.uint8)),
        init("src_ch_start", np.array([0], dtype=np.uint8)),
        init("src_extent", np.array([10, 3, 3], dtype=np.int32)),
        init("dest_idx_table", dest_idx_table),
        init("col5", np.full((1, 1, 3, 1), 5, dtype=np.uint8)),
        init("col0", np.zeros((1, 1, 3, 1), dtype=np.uint8)),
        init("row5", np.full((1, 1, 1, 5), 5, dtype=np.uint8)),
        init("row0", np.array([[[[0, 0, 0, 5, 0]]]], dtype=np.uint8)),
        init("pads_final", np.array([0, 0, 0, 0, 0, 0, 19, 19], dtype=np.int64)),
        init("pad0", np.array(0, dtype=np.uint8)),
    ]

    nodes = [
        node("Einsum", ["input", "yellow", "iota30"], ["R_f"], equation="nchw,c,h->"),
        node("Einsum", ["input", "yellow", "iota30"], ["C_f"], equation="nchw,c,w->"),
        node("Cast", ["R_f"], ["R_u8"], to=TensorProto.UINT8),
        node("Cast", ["C_f"], ["C_u8"], to=TensorProto.UINT8),
        node("Mod", ["R_u8", "four_u8"], ["R_loc"]),
        node("Mod", ["C_u8", "four_u8"], ["C_loc"]),
        node("Sub", ["R_u8", "R_loc"], ["R_src"]),
        node("Sub", ["C_u8", "C_loc"], ["C_src"]),
        node("Cast", ["R_loc"], ["R_loc_i32"], to=TensorProto.INT32),
        node("Cast", ["C_loc"], ["C_loc_i32"], to=TensorProto.INT32),
        node("Unsqueeze", ["R_src", "axes0"], ["R_start"]),
        node("Unsqueeze", ["C_src", "axes0"], ["C_start"]),
        node("Concat", ["src_ch_start", "R_start", "C_start"], ["src_starts_u8"], axis=0),
        node("Cast", ["src_starts_u8"], ["src_starts"], to=TensorProto.INT32),
        node("Add", ["src_starts", "src_extent"], ["src_ends"]),
        node("Slice", ["input", "src_starts", "src_ends", "axes123"], ["src"]),
        node("ArgMax", ["src"], ["src_argmax_i64"], axis=1, keepdims=1),
        node("Cast", ["src_argmax_i64"], ["src_ids"], to=TensorProto.UINT8),
        node("Concat", ["src_ids", "col5", "col0"], ["tile_top"], axis=3),
        node("Concat", ["tile_top", "row5", "row0"], ["tile5"], axis=2),
        node("Gather", ["dest_idx_table", "R_loc_i32"], ["r_dest_inds"], axis=0),
        node("Gather", ["dest_idx_table", "C_loc_i32"], ["c_dest_inds"], axis=0),
        node("Gather", ["tile5", "r_dest_inds"], ["G_r_dest"], axis=2),
        node("Gather", ["G_r_dest", "c_dest_inds"], ["native11"], axis=3),
        node("Pad", ["native11", "pads_final", "pad0"], ["output"], mode="constant"),
    ]

    value_info = [
        vi("R_f", TensorProto.FLOAT, []),
        vi("C_f", TensorProto.FLOAT, []),
        vi("R_u8", TensorProto.UINT8, []),
        vi("C_u8", TensorProto.UINT8, []),
        vi("R_loc", TensorProto.UINT8, []),
        vi("C_loc", TensorProto.UINT8, []),
        vi("R_src", TensorProto.UINT8, []),
        vi("C_src", TensorProto.UINT8, []),
        vi("R_loc_i32", TensorProto.INT32, []),
        vi("C_loc_i32", TensorProto.INT32, []),
        vi("R_start", TensorProto.UINT8, [1]),
        vi("C_start", TensorProto.UINT8, [1]),
        vi("src_starts_u8", TensorProto.UINT8, [3]),
        vi("src_starts", TensorProto.INT32, [3]),
        vi("src_ends", TensorProto.INT32, [3]),
        vi("src", TensorProto.FLOAT, [1, 10, 3, 3]),
        vi("src_argmax_i64", TensorProto.INT64, [1, 1, 3, 3]),
        vi("src_ids", TensorProto.UINT8, [1, 1, 3, 3]),
        vi("tile_top", TensorProto.UINT8, [1, 1, 3, 5]),
        vi("tile5", TensorProto.UINT8, [1, 1, 5, 5]),
        vi("r_dest_inds", TensorProto.INT32, [11]),
        vi("c_dest_inds", TensorProto.INT32, [11]),
        vi("G_r_dest", TensorProto.UINT8, [1, 1, 11, 5]),
        vi("native11", TensorProto.UINT8, [1, 1, 11, 11]),
    ]

    graph = helper.make_graph(
        nodes,
        "task163_direct_u8_terminal",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.UINT8, [1, 1, 30, 30])],
        inits,
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def build_task163_direct_input_onehot_terminal() -> onnx.ModelProto:
    yellow = np.zeros(10, dtype=np.float32)
    yellow[4] = 1.0

    block = np.zeros((3, 30), dtype=np.float32)
    off4 = np.zeros((4, 30), dtype=np.float32)
    for block_id in range(3):
        for local_pos in range(4):
            pos = block_id * 4 + local_pos
            if pos < 11:
                block[block_id, pos] = 1.0
                off4[local_pos, pos] = 1.0
    off3 = off4[:3].copy()

    local = np.zeros((2, 3, 4), dtype=np.float32)
    for r in range(3):
        local[0, r, r] = 1.0
        local[1, r, 3] = 1.0

    coef = np.zeros((10, 10, 3), dtype=np.float32)
    for k in range(10):
        coef[k, k, 0] = 1.0
    coef[0, :, 1] = 1.0
    coef[5, :, 2] = 1.0

    case = np.zeros((3, 2, 2, 2), dtype=np.float32)
    case[1, 0, 0, 0] = 1.0
    case[2, 0, 0, 1] = 1.0
    case[2, 0, 1, 0] = 1.0
    case[2, 0, 1, 1] = 1.0
    case[0, 1, 0, 0] = 1.0
    case[1, 1, 0, 0] = -1.0

    inits = [
        init("yellow", yellow),
        init("iota30", np.arange(30, dtype=np.float32)),
        init("four_u8", np.array(4, dtype=np.uint8)),
        init("src_blocks", np.array([0, 4, 8], dtype=np.uint8)),
        init("idx_r", np.arange(3, dtype=np.uint8).reshape(1, 1, 3, 1)),
        init("idx_c", np.arange(3, dtype=np.uint8).reshape(1, 1, 1, 3)),
        init("one3", np.ones((1, 1, 3, 3), dtype=np.float32)),
        init("block", block),
        init("off3", off3),
        init("off4", off4),
        init("local", local),
        init("coef", coef),
        init("case", case),
    ]

    nodes = [
        node("Einsum", ["input", "yellow", "iota30"], ["R_f"], equation="nchw,c,h->"),
        node("Einsum", ["input", "yellow", "iota30"], ["C_f"], equation="nchw,c,w->"),
        node("Cast", ["R_f"], ["R_u8"], to=TensorProto.UINT8),
        node("Cast", ["C_f"], ["C_u8"], to=TensorProto.UINT8),
        node("Mod", ["R_u8", "four_u8"], ["R_loc"]),
        node("Mod", ["C_u8", "four_u8"], ["C_loc"]),
        node("Sub", ["R_u8", "R_loc"], ["R_src"]),
        node("Sub", ["C_u8", "C_loc"], ["C_src"]),
        node("Equal", ["R_src", "src_blocks"], ["src_r_b"]),
        node("Equal", ["C_src", "src_blocks"], ["src_c_b"]),
        node("Cast", ["src_r_b"], ["src_r"], to=TensorProto.FLOAT),
        node("Cast", ["src_c_b"], ["src_c"], to=TensorProto.FLOAT),
        node("Equal", ["R_loc", "idx_r"], ["r_hit"]),
        node("Equal", ["C_loc", "idx_c"], ["c_hit"]),
        node("And", ["r_hit", "c_hit"], ["loc_b"]),
        node("Cast", ["loc_b"], ["loc"], to=TensorProto.FLOAT),
        node("Concat", ["one3", "loc"], ["locs"], axis=1),
        node(
            "Einsum",
            [
                "input",
                "src_r",
                "src_c",
                "block",
                "off3",
                "block",
                "off3",
                "locs",
                "coef",
                "case",
                "local",
                "local",
                "block",
                "off4",
                "block",
                "off4",
            ],
            ["output"],
            equation="njxy,a,b,ax,rx,by,sy,ndol,kjm,mduv,urp,vsq,oh,ph,lw,qw->nkhw",
        ),
    ]

    value_info = [
        vi("R_f", TensorProto.FLOAT, []),
        vi("C_f", TensorProto.FLOAT, []),
        vi("R_u8", TensorProto.UINT8, []),
        vi("C_u8", TensorProto.UINT8, []),
        vi("R_loc", TensorProto.UINT8, []),
        vi("C_loc", TensorProto.UINT8, []),
        vi("R_src", TensorProto.UINT8, []),
        vi("C_src", TensorProto.UINT8, []),
        vi("src_r_b", TensorProto.BOOL, [3]),
        vi("src_c_b", TensorProto.BOOL, [3]),
        vi("src_r", TensorProto.FLOAT, [3]),
        vi("src_c", TensorProto.FLOAT, [3]),
        vi("r_hit", TensorProto.BOOL, [1, 1, 3, 1]),
        vi("c_hit", TensorProto.BOOL, [1, 1, 1, 3]),
        vi("loc_b", TensorProto.BOOL, [1, 1, 3, 3]),
        vi("loc", TensorProto.FLOAT, [1, 1, 3, 3]),
        vi("locs", TensorProto.FLOAT, [1, 2, 3, 3]),
    ]

    graph = helper.make_graph(
        nodes,
        "task163_direct_input_onehot_terminal",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        inits,
        value_info=value_info,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def build() -> onnx.ModelProto:
    return build_task163_direct_input_onehot_terminal()


if __name__ == "__main__":
    onnx.save(build(), OUT)
    print(f"saved {OUT}")
