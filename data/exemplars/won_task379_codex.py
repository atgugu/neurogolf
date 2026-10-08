#!/usr/bin/env python3
"""Build task379 with coordinate-preserving ramp projections.

True rule: copy the cyan guide line(s), draw blocked perpendicular red rays
from each red seed to reachable guides, and stamp a cyan 3x3 cap at each hit
with the center restored to red; the generator may transpose the whole grid.

Priced design A, rejected histogram LUT family:
  [1,4,30] f32 x2 = 960, [1,4,30] i32 x2 = 960,
  [1,30] i32 x2 = 240, two reshapes = 240, Equal mask = 900,
  params = 305, total = 4565.  It closes on cost but has a row-signature
  collision on train[0] and cannot render the three paint transitions.

Priced design B, built ramp/rank family:
  Four direct [1,30] f32 projections = 480, active extents = 240,
  compact [1,30] bool/u8 predicates and caps, one [7,30] slot table,
  two factor casts [11,30] f16 = 1320, terminal rank Einsum is free.
  This is the coordinate-preserving family mandated for attempt 19.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


OUT = Path("task379.onnx")
DATA = Path("extracted/task379.json")
SEED_SLOTS = 5
RANKS = SEED_SLOTS + 4


def c(name: str, arr) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(arr), name=name)


def n(op: str, inputs: list[str], outputs: list[str], **attrs) -> onnx.NodeProto:
    return helper.make_node(op, inputs, outputs, **attrs)


def solve_horizontal(grid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(grid)
    h, w = grid.shape
    lines = [r for r in range(h) if np.all(grid[r, :] == 8)]
    seeds = [(r, col) for r in range(h) for col in range(w) if grid[r, col] == 2]
    seeds.sort(key=lambda rc: rc[1])

    for line in lines:
        out[line, :] = 8

    for row, col in seeds:
        for line in lines:
            r = row
            dr = -1 if line < row else 1
            while r != line:
                if out[r, col] == 8:
                    break
                out[r, col] = 2
                r += dr
            if r == line:
                for rr in range(r - 1, r + 2):
                    for cc in range(col - 1, col + 2):
                        out[rr, cc] = 8
                out[r, col] = 2
    return out


def rule_np(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.int64)
    if any(np.all(grid[r, :] == 8) for r in range(grid.shape[0])):
        return solve_horizontal(grid)
    return solve_horizontal(grid.T).T


def wide(v: np.ndarray) -> np.ndarray:
    out = v.copy()
    out[1:] |= v[:-1]
    out[:-1] |= v[1:]
    return out


def palette(dtype=np.float32) -> np.ndarray:
    color = np.zeros((RANKS, 10), dtype=np.float32)
    color[0, 0] = 1.0
    color[1, 0] = -0.125
    color[1, 8] = 0.03125
    for rank in (2, 3):
        color[rank, 0] = -1.0
        color[rank, 2] = -1.0
        color[rank, 8] = 2.0
    for rank in range(4, RANKS):
        color[rank, 0] = -1.0
        color[rank, 2] = 1.0
        color[rank, 8] = -1.0
    return color.astype(dtype)


def numpy_factors(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.asarray(grid, dtype=np.int64)
    h, w = grid.shape
    g30 = np.zeros((30, 30), dtype=np.int64)
    g30[:h, :w] = grid

    fg = np.zeros((30, 30), dtype=bool)
    fg[:h, :w] = True
    cyan = g30 == 8
    seed = g30 == 2

    active_rows = fg.max(axis=1)
    active_cols = fg.max(axis=0)
    row_line_counts = cyan.sum(axis=1).astype(np.uint8)
    col_line_counts = cyan.sum(axis=0).astype(np.uint8)
    line_rows_orig = row_line_counts > 3
    line_cols_orig = col_line_counts > 3

    seed_row_plus1_h = np.zeros(30, dtype=np.uint8)
    seed_col_plus1_v = np.zeros(30, dtype=np.uint8)
    for rr, cc in zip(*np.where(seed)):
        seed_row_plus1_h[cc] = rr + 1
        seed_col_plus1_v[rr] = cc + 1

    h_any = bool(line_rows_orig.max())
    line_profile = row_line_counts if h_any else col_line_counts
    slot_plus1 = seed_row_plus1_h if h_any else seed_col_plus1_v

    line_idxs = np.flatnonzero(line_profile > 3)
    h_first = int(line_idxs[0])
    h_last = int(line_idxs[-1])

    row_ids = np.arange(30, dtype=np.uint8)
    top_band = (row_ids == h_first - 1) | (row_ids == h_first + 1)
    bottom_band = (row_ids == h_last - 1) | (row_ids == h_last + 1)

    present = slot_plus1 > 0
    touch_top = present & (slot_plus1 <= h_last + 1)
    touch_bottom = present & (slot_plus1 > h_first)
    wide_top = wide(touch_top.astype(np.uint8)).astype(bool)
    wide_bottom = wide(touch_bottom.astype(np.uint8)).astype(bool)

    positions = np.flatnonzero(slot_plus1 > 0)
    order = sorted(positions, key=lambda pos: int(slot_plus1[pos]), reverse=True)[:SEED_SLOTS]
    slot_cols = np.zeros((SEED_SLOTS, 30), dtype=bool)
    red_slots = np.zeros((SEED_SLOTS, 30), dtype=bool)
    for slot, pos in enumerate(order):
        coord = int(slot_plus1[pos]) - 1
        slot_cols[slot, pos] = True
        if coord < h_first:
            lo, hi = coord, h_first
        elif coord > h_last:
            lo, hi = h_last, coord
        else:
            lo, hi = h_first, h_last
        red_slots[slot, lo : hi + 1] = True

    row_line_comp = line_profile if h_any else active_rows.astype(np.uint8)
    col_line_comp = active_cols.astype(np.uint8) if h_any else line_profile
    row_top_comp = top_band if h_any else wide_top
    col_top_comp = wide_top if h_any else top_band
    row_bottom_comp = bottom_band if h_any else wide_bottom
    col_bottom_comp = wide_bottom if h_any else bottom_band
    row_red = red_slots if h_any else slot_cols
    col_red = slot_cols if h_any else red_slots

    row_fac = np.concatenate(
        [
            active_rows.reshape(1, 30),
            row_line_comp.reshape(1, 30),
            row_top_comp.reshape(1, 30),
            row_bottom_comp.reshape(1, 30),
            row_red,
        ],
        axis=0,
    )
    col_fac = np.concatenate(
        [
            active_cols.reshape(1, 30),
            col_line_comp.reshape(1, 30),
            col_top_comp.reshape(1, 30),
            col_bottom_comp.reshape(1, 30),
            col_red,
        ],
        axis=0,
    )
    logits = np.einsum(
        "nk,nr,nc->krc",
        palette(),
        row_fac.astype(np.float32),
        col_fac.astype(np.float32),
    )
    return logits, row_fac, col_fac


def collision_audit() -> None:
    data = json.loads(DATA.read_text())
    max_rank = 0
    max_seeds = 0
    for split in ("train", "test", "arc-gen"):
        for idx, ex in enumerate(data.get(split, [])):
            inp = np.array(ex["input"], dtype=np.int64)
            exp = np.array(ex["output"], dtype=np.int64)
            if not np.array_equal(rule_np(inp), exp):
                raise AssertionError(f"local rule mismatch {split}[{idx}]")
            logits, row_fac, col_fac = numpy_factors(inp)
            got = logits > 0
            want = np.zeros((10, 30, 30), dtype=bool)
            for r, row in enumerate(exp):
                for col, value in enumerate(row):
                    want[value, r, col] = True
            if not np.array_equal(got, want):
                raise AssertionError(f"factor renderer collision {split}[{idx}]")
            max_rank = max(
                max_rank,
                int(np.linalg.matrix_rank(row_fac.astype(np.float32))),
                int(np.linalg.matrix_rank(col_fac.astype(np.float32))),
            )
            max_seeds = max(max_seeds, int((inp == 2).sum()))
    if max_rank > RANKS:
        raise AssertionError(f"rank guard failed: observed {max_rank}, R={RANKS}")
    if max_seeds > SEED_SLOTS:
        raise AssertionError(f"slot guard failed: observed {max_seeds}, slots={SEED_SLOTS}")


def build() -> onnx.ModelProto:
    collision_audit()

    inits: list[onnx.TensorProto] = [
        c("c8", np.eye(10, dtype=np.float32)[8]),
        c("c2", np.eye(10, dtype=np.float32)[2]),
        c("ramp_plus1", np.arange(1, 31, dtype=np.float32)),
        c("row_ids30", np.arange(30, dtype=np.uint8).reshape(1, 30)),
        c("zero_slots_u8", np.zeros((SEED_SLOTS, 30), dtype=np.uint8)),
        c("zero_u8", np.array(0, dtype=np.uint8)),
        c("zero_f32", np.array(0, dtype=np.float32)),
        c("one_u8", np.array(1, dtype=np.uint8)),
        c("three_u8", np.array(3, dtype=np.uint8)),
        c("shape_Sx1", np.array([SEED_SLOTS, 1], dtype=np.int64)),
        c("shape_11130", np.array([1, 1, 1, 30], dtype=np.int64)),
        c("shape_1x30", np.array([1, 30], dtype=np.int64)),
        c("kS", np.array([SEED_SLOTS], dtype=np.int64)),
        c("rank_on", np.ones((1, RANKS), dtype=np.float16)),
        c("palette", palette(np.float16)),
    ]

    nodes: list[onnx.NodeProto] = [
        n("Einsum", ["input", "c8"], ["row_line_f"], equation="bcij,c->bi"),
        n("Cast", ["row_line_f"], ["row_line"], to=TensorProto.UINT8),
        n("Einsum", ["input", "c8"], ["col_line_f"], equation="bcij,c->bj"),
        n("Cast", ["col_line_f"], ["col_line"], to=TensorProto.UINT8),
        n("Einsum", ["input", "c2", "ramp_plus1"], ["seed_col_plus1_v_f"], equation="bcij,c,j->bi"),
        n("Cast", ["seed_col_plus1_v_f"], ["seed_col_plus1_v"], to=TensorProto.UINT8),
        n("Einsum", ["input", "c2", "ramp_plus1"], ["seed_row_plus1_h_f"], equation="bcij,c,i->bj"),
        n("Cast", ["seed_row_plus1_h_f"], ["seed_row_plus1_h"], to=TensorProto.UINT8),
        n("ReduceMax", ["input"], ["active_rows_f"], axes=[1, 3], keepdims=0),
        n("Cast", ["active_rows_f"], ["active_rows"], to=TensorProto.UINT8),
        n("ReduceMax", ["input"], ["active_cols_f"], axes=[1, 2], keepdims=0),
        n("Cast", ["active_cols_f"], ["active_cols"], to=TensorProto.UINT8),
        n("ReduceMax", ["row_line"], ["h_any_u8"], axes=[1], keepdims=1),
        n("Greater", ["h_any_u8", "three_u8"], ["h_any"]),
        n("Where", ["h_any", "row_line", "col_line"], ["line_profile"]),
        n("Where", ["h_any", "seed_row_plus1_h", "seed_col_plus1_v"], ["slot_plus1"]),
        n("ArgMax", ["line_profile"], ["h_first_i"], axis=1, keepdims=1),
        n("ArgMax", ["line_profile"], ["h_last_i"], axis=1, keepdims=1, select_last_index=1),
        n("Cast", ["h_first_i"], ["h_first"], to=TensorProto.UINT8),
        n("Cast", ["h_last_i"], ["h_last"], to=TensorProto.UINT8),
        n("Sub", ["h_first", "one_u8"], ["h_first_m1"]),
        n("Add", ["h_first", "one_u8"], ["h_first_p1"]),
        n("Sub", ["h_last", "one_u8"], ["h_last_m1"]),
        n("Add", ["h_last", "one_u8"], ["h_last_p1"]),
        n("Equal", ["row_ids30", "h_first_m1"], ["top_m1"]),
        n("Equal", ["row_ids30", "h_first_p1"], ["top_p1"]),
        n("Or", ["top_m1", "top_p1"], ["top_band_b"]),
        n("Cast", ["top_band_b"], ["top_band"], to=TensorProto.UINT8),
        n("Equal", ["row_ids30", "h_last_m1"], ["bottom_m1"]),
        n("Equal", ["row_ids30", "h_last_p1"], ["bottom_p1"]),
        n("Or", ["bottom_m1", "bottom_p1"], ["bottom_band_b"]),
        n("Cast", ["bottom_band_b"], ["bottom_band"], to=TensorProto.UINT8),
        n("Greater", ["slot_plus1", "zero_u8"], ["slot_axis_present"]),
        n("LessOrEqual", ["slot_plus1", "h_last_p1"], ["touch_top_raw"]),
        n("And", ["slot_axis_present", "touch_top_raw"], ["touch_top"]),
        n("Greater", ["slot_plus1", "h_first"], ["touch_bottom_raw"]),
        n("And", ["slot_axis_present", "touch_bottom_raw"], ["touch_bottom"]),
        n("Cast", ["touch_top"], ["touch_top_u8"], to=TensorProto.UINT8),
        n("Reshape", ["touch_top_u8", "shape_11130"], ["touch_top_4d"]),
        n("MaxPool", ["touch_top_4d"], ["wide_top_u8_4d"], kernel_shape=[1, 3], pads=[0, 1, 0, 1]),
        n("Reshape", ["wide_top_u8_4d", "shape_1x30"], ["wide_top"]),
        n("Cast", ["touch_bottom"], ["touch_bottom_u8"], to=TensorProto.UINT8),
        n("Reshape", ["touch_bottom_u8", "shape_11130"], ["touch_bottom_4d"]),
        n("MaxPool", ["touch_bottom_4d"], ["wide_bottom_u8_4d"], kernel_shape=[1, 3], pads=[0, 1, 0, 1]),
        n("Reshape", ["wide_bottom_u8_4d", "shape_1x30"], ["wide_bottom"]),
        n("Cast", ["slot_plus1"], ["slot_scores"], to=TensorProto.FLOAT),
        n("TopK", ["slot_scores", "kS"], ["slot_vals", "slot_idx"], axis=1, largest=1, sorted=1),
        n("Greater", ["slot_vals", "zero_f32"], ["slot_present_1xS"]),
        n("Reshape", ["slot_present_1xS", "shape_Sx1"], ["slot_present"]),
        n("Cast", ["slot_present"], ["slot_present_u8"], to=TensorProto.UINT8),
        n("Reshape", ["slot_idx", "shape_Sx1"], ["slot_idx_Sx1"]),
        n("Cast", ["slot_vals"], ["slot_coord_plus1_1xS"], to=TensorProto.UINT8),
        n("Reshape", ["slot_coord_plus1_1xS", "shape_Sx1"], ["slot_coord_plus1"]),
        n("Sub", ["slot_coord_plus1", "one_u8"], ["slot_coord"]),
        n("ScatterElements", ["zero_slots_u8", "slot_idx_Sx1", "slot_present_u8"], ["slot_cols"], axis=1),
        n("Less", ["slot_coord", "h_first"], ["slot_above_raw"]),
        n("And", ["slot_above_raw", "slot_present"], ["slot_above"]),
        n("Greater", ["slot_coord", "h_last"], ["slot_below_raw"]),
        n("And", ["slot_below_raw", "slot_present"], ["slot_below"]),
        n("Where", ["slot_above", "slot_coord", "h_first"], ["red_lo_above_else_first"]),
        n("Where", ["slot_below", "h_last", "red_lo_above_else_first"], ["red_lo"]),
        n("Where", ["slot_above", "h_first", "h_last"], ["red_hi_above_else_last"]),
        n("Where", ["slot_below", "slot_coord", "red_hi_above_else_last"], ["red_hi"]),
        n("Sub", ["row_ids30", "red_lo"], ["red_row_delta"]),
        n("Sub", ["red_hi", "red_lo"], ["red_span"]),
        n("LessOrEqual", ["red_row_delta", "red_span"], ["red_slots_b"]),
        n("Cast", ["red_slots_b"], ["red_slots"], to=TensorProto.UINT8),
        n("Where", ["h_any", "line_profile", "active_rows"], ["row_line_comp"]),
        n("Where", ["h_any", "active_cols", "line_profile"], ["col_line_comp"]),
        n("Where", ["h_any", "top_band", "wide_top"], ["row_top_comp"]),
        n("Where", ["h_any", "wide_top", "top_band"], ["col_top_comp"]),
        n("Where", ["h_any", "bottom_band", "wide_bottom"], ["row_bottom_comp"]),
        n("Where", ["h_any", "wide_bottom", "bottom_band"], ["col_bottom_comp"]),
        n("Where", ["h_any", "red_slots", "slot_cols"], ["row_red_slots"]),
        n("Where", ["h_any", "slot_cols", "red_slots"], ["col_red_slots"]),
        n(
            "Concat",
            ["active_rows", "row_line_comp", "row_top_comp", "row_bottom_comp", "row_red_slots"],
            ["row_u8"],
            axis=0,
        ),
        n(
            "Concat",
            ["active_cols", "col_line_comp", "col_top_comp", "col_bottom_comp", "col_red_slots"],
            ["col_u8"],
            axis=0,
        ),
        n("Cast", ["row_u8"], ["row_factor"], to=TensorProto.FLOAT16),
        n("Cast", ["col_u8"], ["col_factor"], to=TensorProto.FLOAT16),
        n(
            "Einsum",
            ["rank_on", "palette", "row_factor", "col_factor"],
            ["output"],
            equation="bn,nk,nr,nc->bkrc",
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "task379_coord_ramp_rank",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 10, 30, 30])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 10
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(inferred)
    for node in inferred.graph.node:
        if node.doc_string:
            raise AssertionError("hidden/native/tagged node doc_string is forbidden")
    return inferred


def main() -> None:
    onnx.save(build(), OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
