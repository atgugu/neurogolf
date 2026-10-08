#!/usr/bin/env python3
"""task397 exact source rule with direct-left gather and native code shift.

The exact generator branch table is in rule.py.  This retains all eight possible
source rows, because num_colors=2 permits a box at rows 6..7.

Two value-exact graph identities remove charged state from the Kaggle-proven graph:

1. GatherElements(ids_u8, row) already reads columns 0..8 because the index tensor
   has width 9.  Therefore the 8x9 left_cols Slice is redundant; only the shifted
   right_cols Slice is materialized.
2. Compute code8=ids_u8+1 before padding, then Pad with constant 1 to 10x10.  This
   is identical to Pad(ids_u8)+1 inside the renderer support but charges an 8x10
   Add output instead of a 10x10 Add output.  Final-convolution exterior padding
   remains zero, preserving the reference output byte-for-byte after thresholding.

PAPER BUDGET (every non-output node tensor is charged):
  code_f              [1,1,8,10] f32       320
  ids_u8, nz_u      2x[1,1,8,10] u8        160
  box_count           [1,1,7,9] u8          63
  anchor              [1,1,7,9] bool        63
  row64, row1       2x[1,1,1,9] i64        144
  right_cols          [1,1,8,9] u8          72
  a,b,c,d           4x[1,1,1,9] u8          36
  equality/seen     9x[1,1,1,9] bool         81
  duplicate casts  3x[1,1,1,9] u8           27
  duplicate adds   2x[1,1,1,9] u8           18
  height_raw          [1,1,1,9] u8            9
  height_plane        [1,1,7,9] u8           63
  shadow_q            [1,1,10,10] u8         100
  code8               [1,1,8,10] u8          80
  code10, code10_sq 2x[1,1,10,10] u8         200
  poly3               [1,3,10,10] u8         300
  ------------------------------------------------
  charged memory                              1736
  initializer elements                         112
  paper total                                 1848  (registration ceiling)

The requested 924-byte TerminalArcRender atlas is physically unavailable in legal
ONNX: TerminalArcRender is not a standard operator, nonstandard domains are banned,
and the actual graph input is float32 one-hot rather than a uint8 color plane.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "refs" / "kaggle_proven_best.onnx"
OUTPUT = HERE / "task397.onnx"


def _replace_initializer(model: onnx.ModelProto, name: str, value: np.ndarray) -> None:
    replacement = numpy_helper.from_array(value, name=name)
    for i, old in enumerate(model.graph.initializer):
        if old.name == name:
            model.graph.initializer[i].CopyFrom(replacement)
            return
    raise KeyError(name)


def build() -> onnx.ModelProto:
    model = onnx.load(REFERENCE)
    model.graph.name = "task397_rule_reconciled_direct_left_native_shift"

    # row64 has shape [1,1,1,9], so gathering it from ids_u8 naturally selects
    # source columns 0..8.  Remove the redundant left-column materialization.
    kept = []
    for node in model.graph.node:
        if list(node.output) == ["left_cols"]:
            if node.op_type != "Slice":
                raise RuntimeError("unexpected left_cols producer")
            continue
        for i, name in enumerate(node.input):
            if name == "left_cols":
                node.input[i] = "ids_u8"
        kept.append(node)
    del model.graph.node[:]
    model.graph.node.extend(kept)

    # Reuse one_i64 as the right Slice start.  A length-1 tensor broadcasts in
    # row64+1 and also satisfies Slice's one-dimensional starts requirement.
    _replace_initializer(model, "one_i64", np.array([1], dtype=np.int64))
    for node in model.graph.node:
        if list(node.output) == ["right_cols"]:
            node.input[1] = "one_i64"

    # Remove constants that belonged only to the deleted/reused Slice starts.
    unused = {"left_start", "left_end", "right_start"}
    retained_inits = [x for x in model.graph.initializer if x.name not in unused]
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained_inits)

    # Turn Pad(ids)+1 into Pad(ids+1, constant=1).  Reorder these two nodes so the
    # graph stays topological and keep the proven polynomial renderer unchanged.
    pad_node = None
    add_node = None
    other_nodes = []
    insert_at = None
    for node in model.graph.node:
        if list(node.output) == ["ids10"]:
            pad_node = node
            if insert_at is None:
                insert_at = len(other_nodes)
            continue
        if list(node.output) == ["code10"]:
            add_node = node
            if insert_at is None:
                insert_at = len(other_nodes)
            continue
        other_nodes.append(node)
    if pad_node is None or add_node is None or insert_at is None:
        raise RuntimeError("reference renderer prefix not found")
    add_node.input[:] = ["ids_u8", "one_u"]
    add_node.output[:] = ["code8"]
    pad_node.input[:] = ["code8", "pad_ids10", "one_u"]
    pad_node.output[:] = ["code10"]
    other_nodes[insert_at:insert_at] = [add_node, pad_node]
    del model.graph.node[:]
    model.graph.node.extend(other_nodes)

    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model, full_check=True)
    return model


if __name__ == "__main__":
    onnx.save(build(), OUTPUT)
    print(f"wrote {OUTPUT}")
