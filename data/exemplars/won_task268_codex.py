#!/usr/bin/env python3
"""Attempt 15: compress the proven packed-bitrow parser feeding the legal terminal.

Priced inventory ceiling (fixed scorer):
  packed parser outputs (excluding final mask)   <= 995 B
  out4mask bool [1,1,10,10]                        100 B
  terminal_updates f32 [1,2,10,10]                 800 B
  initializer elements                            <= 260
  terminal ScatterElements output                    free
  total                                           <= 2155

The first exact rewrite removes per-ray `position < grid_size` guards.  All row
bitsets are instead intersected once with the already-computed low/high grid masks.
That is algebraically identical and keeps padded columns clear for every size 5..10.
"""

from pathlib import Path

import onnx
import numpy as np
from onnx import helper, numpy_helper


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "task268_terminal_probe.onnx"
OUT = ROOT / "task268.onnx"


def dead_code_elim(model: onnx.ModelProto) -> None:
    graph = model.graph
    needed = {out.name for out in graph.output}
    kept = []
    for node in reversed(graph.node):
        if any(out in needed for out in node.output):
            kept.append(node)
            needed.update(name for name in node.input if name)
    kept.reverse()
    del graph.node[:]
    graph.node.extend(kept)

    used = {name for node in graph.node for name in node.input if name}
    old = list(graph.initializer)
    del graph.initializer[:]
    graph.initializer.extend(x for x in old if x.name in used)
    del graph.value_info[:]


def build() -> onnx.ModelProto:
    model = onnx.load(SOURCE)
    graph = model.graph

    # Let each profile contraction carry its two singleton axes.  Slicing then
    # produces the exact rank-4 tensors consumed by the parser, deleting both
    # charged Reshapes and their eight shape parameters.
    ch0 = next(x for x in graph.initializer if x.name == "ch0_sel")
    ch0.CopyFrom(numpy_helper.from_array(
        np.array([1] + [16] * 9, np.float32).reshape(10, 1, 1), "ch0_sel"))
    prof_ax = next(x for x in graph.initializer if x.name == "prof_ax")
    prof_ax.CopyFrom(numpy_helper.from_array(np.array([2], np.int64), "prof_ax"))
    profile_eq = {
        "row_prof30": b"nchw,cxy->nxhy",
        "col_prof30": b"nchw,cxy->nxwy",
    }
    for node in graph.node:
        if node.output and node.output[0] in profile_eq:
            next(a for a in node.attribute if a.name == "equation").s = profile_eq[node.output[0]]
        if node.op_type == "Slice" and node.output and node.output[0] == "col_black1d":
            node.input[3] = "prof_ax"
        for i, name in enumerate(node.input):
            if name == "row_black":
                node.input[i] = "row_black1d"
            elif name == "col_black":
                node.input[i] = "col_black1d"

    graph.initializer.extend([
        numpy_helper.from_array(np.array([2], np.uint8), "TWO8_NEW"),
        numpy_helper.from_array(np.array([15], np.uint8), "FIFTEEN8"),
        numpy_helper.from_array(np.array([3], np.uint8), "THREE8"),
        numpy_helper.from_array(np.array([10], np.uint8), "TEN8"),
    ])

    # Encoded profile p = black + 16*colored = grid_size + 15*colored.
    # Thus p>15 identifies affected rows/columns without confusing zero padding,
    # while p%15 recovers the square grid size from any occupied row.
    profile_insert = next(i for i, n in enumerate(graph.node)
                          if n.output and n.output[0] == "row_size")
    graph.node.insert(profile_insert, helper.make_node(
        "Greater", ["row_black1d", "FIFTEEN8"], ["rowhas_encoded"],
        name="encoded_affected_rows"))
    graph.node.insert(profile_insert + 1, helper.make_node(
        "Greater", ["col_black1d", "FIFTEEN8"], ["colhas_encoded"],
        name="encoded_affected_columns"))
    graph.node.insert(profile_insert + 2, helper.make_node(
        "Mod", ["row_black1d", "FIFTEEN8"], ["row_grid_sizes"],
        name="decode_grid_size_rows"))
    graph.node.insert(profile_insert + 3, helper.make_node(
        "ReduceMax", ["row_grid_sizes"], ["grid_size_encoded"],
        name="encoded_grid_size", keepdims=1))

    # The affected flags are contiguous. ArgMax(first/last) produces both box
    # endpoints and can be reused directly as Gather indices, eliminating four
    # coordinate-Where vectors, four reductions, all four i32 index casts, and
    # the now-unused column coordinate initializer.
    endpoint_insert = profile_insert + 4
    endpoint_nodes = [
        helper.make_node("Cast", ["rowhas_encoded"], ["rowhas_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("Cast", ["colhas_encoded"], ["colhas_u8"], to=onnx.TensorProto.UINT8),
        helper.make_node("ArgMax", ["rowhas_u8"], ["r0_idx"], axis=2, keepdims=1,
                         select_last_index=0),
        helper.make_node("ArgMax", ["rowhas_u8"], ["r1_idx"], axis=2, keepdims=1,
                         select_last_index=1),
        helper.make_node("ArgMax", ["colhas_u8"], ["k0_idx"], axis=2, keepdims=1,
                         select_last_index=0),
        helper.make_node("ArgMax", ["colhas_u8"], ["k1_idx"], axis=2, keepdims=1,
                         select_last_index=1),
        helper.make_node("Cast", ["r0_idx"], ["r0"], to=onnx.TensorProto.UINT8),
        helper.make_node("Cast", ["r1_idx"], ["r1"], to=onnx.TensorProto.UINT8),
        helper.make_node("Cast", ["k0_idx"], ["k0"], to=onnx.TensorProto.UINT8),
        helper.make_node("Cast", ["k1_idx"], ["k1"], to=onnx.TensorProto.UINT8),
    ]
    for node in graph.node:
        if node.output and node.output[0] in {"r0", "r1", "k0", "k1"}:
            node.output[0] = "old_" + node.output[0]
    for off, node in enumerate(endpoint_nodes):
        graph.node.insert(endpoint_insert + off, node)
    for node in graph.node:
        if node.op_type == "GatherElements" and node.output:
            idx = {"topCnt": "r0_idx", "botCnt": "r1_idx",
                   "leftCnt": "k0_idx", "rightCnt": "k1_idx"}.get(node.output[0])
            if idx:
                node.input[1] = idx
            if node.output[0] in {"leftCnt", "rightCnt"}:
                next(a for a in node.attribute if a.name == "axis").i = 2

    encoded_replace = {
        "rowhas": "rowhas_encoded",
        "colhas": "colhas_encoded",
        "grid_size": "grid_size_encoded",
    }
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name in encoded_replace:
                node.input[i] = encoded_replace[name]
        if node.output:
            # More colored cells means a larger encoded profile, the reverse of
            # the reference's black-count comparison.
            if node.output[0] in {"topOpen", "botOpen", "rightOpen"}:
                node.op_type = {
                    "topOpen": "Less",
                    "botOpen": "Greater",
                    "rightOpen": "Greater",
                }[node.output[0]]

    # A wrapped u8 distance labels the horizontal straight band with its high
    # bit and labels the two adjacent interior rows with zero.  This one packed
    # state replaces the duplicated >=/<= predicates and their separate Min.
    range_insert = next(i for i, n in enumerate(graph.node)
                        if n.output and n.output[0] == "s_le_h")
    range_nodes = [
        helper.make_node("Add", ["r0", "ONE8"], ["r0p1_new"]),
        helper.make_node("Sub", ["r1", "ONE8"], ["r1m1_new"]),
        helper.make_node("Sub", ["r0p1_new", "R"], ["dist_top_base"]),
        helper.make_node("Sub", ["R", "r1m1_new"], ["dist_bot_base"]),
        helper.make_node("Min", ["dist_top_base", "dist_bot_base"], ["dist_base"]),
        helper.make_node("Greater", ["dist_base", "EIGHT8"], ["s_band_new"]),
        helper.make_node("Equal", ["dist_base", "ZERO8"], ["interior_edge"]),
        helper.make_node("Or", ["interior_edge", "s_band_new"], ["inR_new"]),
        helper.make_node("Sub", ["k0", "ONE8"], ["k0m1_new"]),
        helper.make_node("Add", ["k1", "ONE8"], ["k1p1_new"]),
    ]
    for off, node in enumerate(range_nodes):
        graph.node.insert(range_insert + off, node)

    range_replace = {
        "inR": "inR_new",
        "s_band_h": "s_band_new",
        "dist_h": "dist_base",
    }
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name in range_replace:
                node.input[i] = range_replace[name]
        if node.output and node.output[0] == "h_left_pos":
            node.input[0] = "k0m1_new"
        elif node.output and node.output[0] == "h_right_pos":
            node.input[0] = "k1p1_new"
        elif node.output and node.output[0] == "k1m2":
            node.input[0] = "k1"
            node.input[1] = "TWO8_NEW"
        elif node.output and node.output[0] == "k0p2":
            node.input[0] = "k0"
            node.input[1] = "TWO8_NEW"

    # Rebase the two interval shifts on 2 so k0+1 and k1-1 never need to
    # materialize: 1<<(k0+1) == 2<<k0 and 1<<(k1-1) == 2<<(k1-2).
    for node in graph.node:
        if node.output and node.output[0] == "int_lo_before_shift":
            node.input[0] = "TWO8_NEW"
            node.input[1] = "k0"
        elif node.output and node.output[0] == "open_lo_end_shift":
            node.input[0] = "TWO8_NEW"
            node.input[1] = "k1m2"

    # size's two high-column bits are 3 >> (10-size): 0 for sizes <=8,
    # 1 for size 9, and 3 for size 10.  This removes the underflow guard.
    for node in graph.node:
        if node.output and node.output[0] == "suff_hi_end_amt":
            node.input[0] = "TEN8"
            node.input[1] = "grid_size_encoded"
        elif node.output and node.output[0] == "suff_hi_end_shift":
            node.input[0] = "THREE8"
            next(a for a in node.attribute if a.name == "direction").s = b"RIGHT"

    for node in graph.node:
        for i, name in enumerate(node.input):
            if name == "suff_hi_end":
                node.input[i] = "suff_hi_end_shift"

    # Horizontal diagonals must stop at the bottom of the source grid.  Reuse
    # the decoded per-row grid-size profile to replace an invalid position with
    # grid_size itself, whose bit is cleared by the dynamic column mask.
    ray_insert = next(i for i, n in enumerate(graph.node)
                      if n.output and n.output[0] == "h_ray_lo_raw")
    graph.node.insert(ray_insert, helper.make_node(
        "Greater", ["row_grid_sizes", "ZERO8"], ["row_is_valid"],
        name="encoded_valid_rows"))
    graph.node.insert(ray_insert + 1, helper.make_node(
        "Where", ["row_is_valid", "h_ray_pos", "grid_size_encoded"],
        ["h_ray_pos_grid"], name="stop_horizontal_ray_at_grid"))
    for node in graph.node:
        if node.output and node.output[0] in {"h_ray_lo_raw", "h_ray_hi_raw_sh"}:
            node.input[0 if node.output[0] == "h_ray_hi_raw_sh" else 1] = "h_ray_pos_grid"

    # A ray position outside the source grid may set bits in padded columns.  The
    # reference clears each ray separately.  Clear the merged packed rows once
    # with suff_lo_end/suff_hi_end, which are exactly the size-bit masks.
    replace = {
        "h_ray_lo": "h_ray_lo_raw",
        "h_ray_hi": "h_ray_hi_raw",
        "v_right_lo": "v_right_lo_raw",
        "v_right_hi": "v_right_hi_raw",
        # Horizontal rays past the bottom/right edge necessarily land outside
        # the dynamic grid column mask below; the separate row-valid Where is
        # therefore redundant as well.
        "h_lo": "h_raw_lo",
        "h_hi": "h_raw_hi",
    }
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name in replace:
                node.input[i] = replace[name]

    pix_nodes = {"pix_lo8": "cmask_lo_grid", "pix_hi2": "cmask_hi_grid"}
    insert_at = next(i for i, n in enumerate(graph.node) if "pix_lo8" in n.output)
    graph.node.insert(
        insert_at,
        helper.make_node("BitwiseAnd", ["cmask_lo8", "suff_lo_end"], ["cmask_lo_grid"],
                         name="mask_columns_to_grid_low"),
    )
    graph.node.insert(
        insert_at + 1,
        helper.make_node("BitwiseAnd", ["cmask_hi2", "suff_hi_end_shift"], ["cmask_hi_grid"],
                         name="mask_columns_to_grid_high"),
    )
    for node in graph.node:
        if node.output and node.output[0] in pix_nodes:
            node.input[1] = pix_nodes[node.output[0]]

    dead_code_elim(model)
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return model


def main() -> int:
    model = build()
    onnx.save(model, OUT)
    print(f"saved {OUT} nodes={len(model.graph.node)} inits={len(model.graph.initializer)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
