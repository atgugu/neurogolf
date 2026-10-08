#!/usr/bin/env python3
"""Certified native-10 rewrite of the task392 paying pin.

The generator fixes size=10.  The pin already renders that native 10x10 state by
terminal ConvTranspose (which expands it to the required 30x30 one-hot canvas).
This surgery only replaces its color/count decoder.

Cost budget (all non-terminal node outputs are charged):
  old decoder: color_counts f32[1,10] 40 + nonblack_counts f32[1,9] 36
       + color_idx0 i64[1] 8 + colored_count f32[1] 4 = 88 B; 4 params.
  new decoder: stats f32[1,2] 8 + Split count/color_code f32[1,1] 8
       + color_div f32[1,1] 4 + color_i8/color_idx_raw/color_idx0 i8 each 1
       + reshaped count f32[1] 4 = 27 B; 22 params.
  unchanged pin: 643 - 88 + 27 = 582 B memory; 221 - 4 + 22 = 239 params.
  paper total = 821 B, at the Lane-A registration ceiling.
"""
import os
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

PIN = (os.environ.get("NEUROGOLF_CLEAN", "") + "/knowledge_proven_paying/task392/0f3b237c.onnx")


def init(name, a):
    return numpy_helper.from_array(np.asarray(a), name)


def build():
    m = onnx.load(PIN)
    g = m.graph
    # Input has one non-black paint color.  The contraction emits (N, c*N):
    # N is the colored-cell count required by the pin hash; c is color 1..9.
    # Thus floor(c*N/N)-1 is precisely the pin's old non-black ArgMax index.
    stats_weights = np.zeros((10, 2), np.float32)
    stats_weights[1:, 0] = 1
    stats_weights[:, 1] = np.arange(10, dtype=np.float32)
    g.initializer.extend([
        init("stats_weights", stats_weights),
        init("one_i8_decoder", np.array(1, np.int8)),
        init("one_shape", np.array([1], np.int64)),
    ])
    # Delete only constants used exclusively by the removed decoder.
    keep = {x.name for x in g.initializer} - {"color_starts", "color_ends", "rs_axes"}
    kept = [x for x in g.initializer if x.name in keep]
    del g.initializer[:]
    g.initializer.extend(kept)
    # Equal is integer-type strict; its terminal color-ID table can be narrowed
    # value-exactly because IDs are only -1..8.
    for x in g.initializer:
        if x.name == "render_color_ids":
            x.CopyFrom(init("render_color_ids", np.arange(-1, 9, dtype=np.int8).reshape(1, 10, 1, 1)))

    old = list(g.node)
    # first four pin nodes are exactly ReduceSum, Slice, ArgMax, ReduceMax
    replacement = [
        helper.make_node("Einsum", ["input", "stats_weights"], ["stats"],
                         equation="abcd,be->ae"),
        helper.make_node("Split", ["stats"], ["colored_count_raw", "color_code"],
                         axis=1),
        helper.make_node("Reshape", ["colored_count_raw", "one_shape"], ["colored_count"]),
        helper.make_node("Div", ["color_code", "colored_count_raw"], ["color_div"]),
        helper.make_node("Cast", ["color_div"], ["color_i8"], to=TensorProto.INT8),
        helper.make_node("Sub", ["color_i8", "one_i8_decoder"], ["color_idx_raw"]),
        helper.make_node("Reshape", ["color_idx_raw", "one_shape"], ["color_idx0"]),
    ]
    del g.node[:]
    g.node.extend(replacement + old[4:])
    onnx.checker.check_model(m, full_check=True)
    return m


if __name__ == "__main__":
    onnx.save(build(), "task392.onnx")
