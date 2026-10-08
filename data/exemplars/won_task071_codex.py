#!/usr/bin/env python3
"""Build task071 with an axis-index terminal Gather renderer.

The decider's literal fixed-side recipe is impossible because ARC-GEN randomizes
flip.  We retain the selected family and cheaply classify the dirty half from
two guaranteed mirrored box samples plus one top-row sprite-color sample.
"""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


nodes = []
inits = []


def init(name, value, dtype=None):
    arr = np.asarray(value, dtype=dtype)
    inits.append(numpy_helper.from_array(arr, name=name))
    return name


def node(op, inputs, outputs, **attrs):
    nodes.append(helper.make_node(op, inputs, outputs, **attrs))


# Fixed active certificate: generated grids are at most 16x16, the sprite top
# is in rows 2..5, and all top-row endpoints lie in columns 0..12.
starts = init("starts", [0, 0, 0, 0], np.int64)
ends = init("ends", [1, 1, 6, 16], np.int64)
axes4 = init("axes4", [0, 1, 2, 3], np.int64)
sq01 = init("sq01", [0, 1], np.int64)
rev13 = init("rev13", np.arange(15, -1, -1), np.int32)
c12 = init("c12", [15], np.int32)
c2 = init("c2", [2], np.int32)
c5 = init("c5", [5], np.int32)
zero1 = init("zero1", [0], np.int32)

# Locate the first row containing a non-background cell from channel zero.
node("Slice", ["input", starts, ends, axes4], ["active_bg"])
node("ReduceMin", ["active_bg"], ["row_min"], axes=[3], keepdims=1)
node("ArgMin", ["row_min"], ["top_raw"], axis=2, keepdims=0)
node("Squeeze", ["top_raw", sq01], ["top"])
node("Gather", ["active_bg", "top"], ["top_f32"], axis=2)
node("Cast", ["top_f32"], ["top_u8"], to=TensorProto.UINT8)
node("Gather", ["top_u8", rev13], ["top_rev"], axis=3)

# Dynamic symmetry-axis endpoints L,R and their sum S.
node("ArgMin", ["top_u8"], ["L_raw"], axis=3, keepdims=0)
node("ArgMin", ["top_rev"], ["RR_raw"], axis=3, keepdims=0)
node("Squeeze", ["L_raw", sq01], ["L64"])
node("Squeeze", ["RR_raw", sq01], ["RR64"])
node("Cast", ["L64"], ["L"], to=TensorProto.INT32)
node("Cast", ["RR64"], ["RR"], to=TensorProto.INT32)
node("Sub", [c12, "RR"], ["R"])
node("Add", ["L", "R"], ["S"])

# At row top+2, floor(S/2)+2 is guaranteed inside the unflipped box;
# its mirror is guaranteed inside the flipped box.  GatherND reads the ten
# channels of those two pixels and of the top-left sprite endpoint.
node("Cast", ["top"], ["top32"], to=TensorProto.INT32)
node("Add", ["top32", c2], ["sample_row"])
node("Div", ["S", c2], ["halfS"])
node("Add", ["halfS", c2], ["right_col"])
node("Sub", ["S", "right_col"], ["left_col"])

# GatherND uses batch_dims=2, so batch/channel are implicit and each index only
# stores (row,col).  This halves the mandatory int64 index workspace.
base = np.zeros((1, 10, 3, 2), dtype=np.int64)
base_idx = init("base_idx", base)
dyn_shape = init("dyn_shape", [1, 1, 3, 2], np.int64)
node("Cast", ["L"], ["L_idx"], to=TensorProto.INT64)
node("Cast", ["sample_row"], ["sample_row_idx"], to=TensorProto.INT64)
node("Cast", ["left_col"], ["left_col_idx"], to=TensorProto.INT64)
node("Cast", ["right_col"], ["right_col_idx"], to=TensorProto.INT64)
node(
    "Concat",
    ["top", "L_idx", "sample_row_idx", "left_col_idx",
     "sample_row_idx", "right_col_idx"],
    ["dynamic_flat"],
    axis=0,
)
node("Reshape", ["dynamic_flat", dyn_shape], ["dynamic_idx"])
node("Add", [base_idx, "dynamic_idx"], ["pixel_indices"])
node("GatherND", ["input", "pixel_indices"], ["pixels"], batch_dims=2)
node("ArgMax", ["pixels"], ["colors"], axis=1, keepdims=0)
sq0 = init("sq0", [0], np.int64)
node("Squeeze", ["colors", sq0], ["colors_vec"])

# The left sample is clean iff its color is background or the sprite color.
idx_sprite = init("idx_sprite", np.asarray(0, dtype=np.int32))
idx_left = init("idx_left", np.asarray(1, dtype=np.int32))
zero64 = init("zero64", np.asarray(0, dtype=np.int64))
node("Gather", ["colors_vec", idx_sprite], ["sprite_color"], axis=0)
node("Gather", ["colors_vec", idx_left], ["left_color"], axis=0)
node("Equal", ["left_color", "sprite_color"], ["left_sprite"])
node("Equal", ["left_color", zero64], ["left_bg"])
node("Or", ["left_sprite", "left_bg"], ["left_clean"])

# Construct both reflection schedules and select the schedule sourced from the
# clean half.  The final Gather is the only full-grid result and is free.
j = init("j", np.arange(30, dtype=np.int32))
node("Sub", ["S", j], ["mirror_j"])
node("Min", [j, "mirror_j"], ["from_left"])
node("Max", [j, "mirror_j"], ["from_right"])
node("Where", ["left_clean", "from_left", "from_right"], ["idx"])
node("Sub", ["halfS", c5], ["active_left"])
node("Add", ["halfS", c5], ["active_right"])
node("GreaterOrEqual", [j, "active_left"], ["in_left"])
node("LessOrEqual", [j, "active_right"], ["in_right"])
node("And", ["in_left", "in_right"], ["in_active"])
node("Where", ["in_active", "idx", j], ["bounded_idx"])
node("Gather", ["input", "bounded_idx"], ["output"], axis=3)

graph = helper.make_graph(
    nodes,
    "task071_axis_index",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
    initializer=inits,
)
model = helper.make_model(
    graph,
    opset_imports=[helper.make_opsetid("", 13)],
    producer_name="task071-att16",
)
model.ir_version = 8
onnx.checker.check_model(model)
model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
onnx.checker.check_model(model)
onnx.save(model, "task071.onnx")
print("wrote task071.onnx")
