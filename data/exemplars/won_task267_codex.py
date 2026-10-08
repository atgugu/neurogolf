"""task267: rank-2 marker remap with one terminal Einsum.

True rule: erase the bottom-left marker, recolor every other non-background
cell with the marker color, and retain color-0 background everywhere else.

Priced budget before build:
  marker   f32 [1,10,1,1] = 40 B
  marker_b bool[1,10,1,1] = 10 B
  selector f32 [1,10,1,2] = 80 B
  palette  f32 [1,10,1,2] = 80 B
  initializers = 39 elements
Total: 210 B memory + 39 params = 249.
"""
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def init(name, value):
    return numpy_helper.from_array(np.asarray(value), name)


base = np.zeros((1, 10, 1, 2), dtype=np.float32)
base[:, 0, :, 0] = 1.0
base[:, 1:, :, 1] = 1.0
erase = np.array([1.0, 0.0], dtype=np.float32).reshape(1, 1, 1, 2)
e0 = np.zeros((1, 10, 1, 1), dtype=np.float32)
e0[:, 0] = 1.0

nodes = [
    helper.make_node("Slice", ["input", "m_st", "m_en", "axes_hw"], ["marker"]),
    helper.make_node("Greater", ["marker", "zero"], ["marker_b"]),
    helper.make_node("Where", ["marker_b", "erase", "base"], ["selector"]),
    helper.make_node("Concat", ["e0", "marker"], ["palette"], axis=3),
    helper.make_node(
        "Einsum",
        ["input", "selector", "palette"],
        ["output"],
        equation="nchw,ncqr,nkqr->nkhw",
    ),
]

inits = [
    # The live standardized input keeps the generator's marker at grid [6, 0].
    # The audited [5, 1] coordinates read background on every corpus example.
    init("m_st", np.array([6, 0], dtype=np.int64)),
    init("m_en", np.array([7, 1], dtype=np.int64)),
    init("axes_hw", np.array([2, 3], dtype=np.int64)),
    init("zero", np.array(0.0, dtype=np.float32)),
    init("base", base),
    init("erase", erase),
    init("e0", e0),
]

graph = helper.make_graph(
    nodes,
    "task267_rank2_marker_remap",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
    inits,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
onnx.checker.check_model(model)
onnx.save(model, "task267.onnx")
