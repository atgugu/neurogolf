"""Build task340 from the Kaggle-proven graph with two value-exact byte reductions.

The rule is proved in rule.py.  This build preserves the proven graph's operators and
replaces only (1) a nine-entry row-bit lookup with an equivalent uint8 right shift and
(2) a rank-2 Gather plus Reshape with a rank-aware Gather that emits the renderer input
directly.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "refs" / "kaggle_proven_best.onnx"
TARGET = HERE / "task340.onnx"


def attr(node, name, value):
    del node.attribute[:]
    node.attribute.extend([helper.make_attribute(name, value)])


def build():
    model = onnx.load(SOURCE)
    graph = model.graph

    nodes = {out: node for node in graph.node for out in node.output if out}

    # Preserve the kind-selection logic but keep its singleton axes through both
    # Gathers.  The second Gather then directly has [1,10,1,30], so the old final
    # Reshape and its 300-byte output disappear.
    shape10 = next(x for x in graph.initializer if x.name == "shape10")
    shape10.CopyFrom(numpy_helper.from_array(np.array([1, 10, 1], np.int64), "shape10"))

    col_lut = nodes["col_lut"]
    attr(col_lut, "axis", 0)

    col_bits_gather = nodes["col_bits10x30"]
    attr(col_bits_gather, "axis", 3)
    col_bits_gather.output[0] = "col_bits"

    old_col_reshape = nodes["col_bits"]

    # row_type is in 0..8.  128 >> (8-row_type) is exactly the old LUT
    # [0,1,2,4,8,16,32,64,128] under uint8 shift semantics.
    old_row_cast = nodes["row_type_i"]
    old_row_gather = nodes["row_bits30"]
    old_row_reshape = nodes["row_bits"]

    row_sub = helper.make_node("Sub", ["k8", "row_type"], ["row_shift"], name="row_shift")
    row_shift = helper.make_node(
        "BitShift", ["bit128", "row_shift"], ["row_bits30"],
        name="row_bits_lookup", direction="RIGHT"
    )

    rebuilt = []
    for node in graph.node:
        if node is old_row_cast:
            rebuilt.extend([row_sub, row_shift])
        elif node is old_row_gather or node is old_col_reshape:
            continue
        else:
            rebuilt.append(node)
    del graph.node[:]
    graph.node.extend(rebuilt)

    # The lookup initializer is no longer used; the reshape initializer for the
    # deleted col_bits node is likewise dead.
    kept = [
        x for x in graph.initializer
        if x.name not in {"row_bit_lut", "shape1x10x1x30"}
    ]
    del graph.initializer[:]
    graph.initializer.extend(kept)
    graph.initializer.append(numpy_helper.from_array(np.array([128], np.uint8), "bit128"))

    # The model already uses BitwiseAnd and therefore opset 18; keep that contract.
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(inferred, full_check=True)
    onnx.save(inferred, TARGET)


if __name__ == "__main__":
    build()
